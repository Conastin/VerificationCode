/* Background service worker: right-click menu "配置本站验证码" tells the
 * content script to open the visual config bar; the config it saves is
 * picked up by the content script via chrome.storage.onChanged. Also owns
 * the fail-sample store (IndexedDB): content scripts report failed
 * recognitions/submits, the popup triggers ZIP export / clearing.
 */
"use strict";

const CONTENT_FILES = [
  "vendor/onnxruntime-web/ort.wasm.min.js",
  "shared/recognizer.js",
  "content.js",
];

// The fail-sample snapshot (image + prediction) must survive a whole-page
// reload, but content scripts cannot touch chrome.storage.session directly
// ("Access to storage is not allowed from this context"). All session
// storage goes through these background-side handlers instead (see
// storeFailSnap / getFailSnap / clearFailSnap below).

chrome.runtime.onInstalled.addListener(() => {
  chrome.contextMenus.removeAll(() => {
    chrome.contextMenus.create({
      id: "config-captcha",
      title: "配置本站验证码",
      contexts: ["page"],
    });
  });
});

async function ensureContentScript(tabId) {
  // Content scripts only auto-inject on pages loaded after the extension was
  // installed/updated; inject manually for older open tabs.
  try {
    await chrome.scripting.executeScript({
      target: { tabId },
      files: CONTENT_FILES,
    });
  } catch (err) {
    console.warn("[captcha-autofill] manual injection failed:", err);
  }
}

async function sendToTab(tabId, message) {
  try {
    return await chrome.tabs.sendMessage(tabId, message);
  } catch {
    // content script not present (page opened before install) -> inject + retry
    await ensureContentScript(tabId);
    return await chrome.tabs.sendMessage(tabId, message);
  }
}

chrome.contextMenus.onClicked.addListener(async (info, tab) => {
  if (info.menuItemId !== "config-captcha" || !tab?.id) return;
  try {
    await sendToTab(tab.id, { type: "startConfig" });
  } catch (err) {
    console.error("[captcha-autofill] startConfig failed:", err);
  }
});

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  const handle = {
    saveFailSample: () => storeFailSample({
      ...message,
      origin: sender.origin || "",
      pageUrl: (sender.tab && sender.tab.url) || "",
    }),
    countFails,
    exportFails,
    clearFails,
    // Fail-snapshot relay: content scripts are barred from storage.session,
    // so reads/writes are proxied here (trusted context).
    storeFailSnap: async () => {
      await chrome.storage.session.set({ failSnap: message.snap });
      return { ok: true };
    },
    getFailSnap: async () => {
      const { failSnap } = await chrome.storage.session.get("failSnap");
      return { failSnap: failSnap || null };
    },
    clearFailSnap: async () => {
      await chrome.storage.session.remove("failSnap");
      return { ok: true };
    },
  }[message?.type];
  if (!handle) return;
  handle().then(sendResponse).catch((err) => {
    console.error("[captcha-autofill]", message.type, "failed:", err);
    sendResponse({ ok: false, error: String(err) });
  });
  return true; // async response
});

// ------------------------------------------- fail-sample storage (IndexedDB)
// chrome.storage.local (10 MB) is too small for images; keep samples in the
// service worker's IndexedDB and export them as a ZIP on demand.
const DB_NAME = "captcha-fails";
const DB_STORE = "samples";
const MAX_SAMPLES = 2000;

function openDB() {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, 1);
    req.onupgradeneeded = () => {
      req.result.createObjectStore(DB_STORE, { keyPath: "hash" });
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

function hashBytes(bytes) {
  // FNV-1a 32-bit over the JPEG payload; used for content dedup.
  let h = 0x811c9dc5;
  const view = new Uint8Array(bytes);
  for (let i = 0; i < view.length; i++) {
    h ^= view[i];
    h = Math.imul(h, 0x01000193) >>> 0;
  }
  return h.toString(16).padStart(8, "0");
}

async function storeFailSample(msg) {
  // Binary travels as base64 (runtime messages are JSON-serialized; raw
  // TypedArray/ArrayBuffer arrive as empty objects and hash to a constant,
  // deduping every sample into "duplicate").
  let bytes;
  try {
    const bin = atob(msg.b64 || "");
    bytes = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  } catch (err) {
    console.error("[captcha-autofill] bg base64 decode failed:", msg.label, err);
    return { ok: false, error: "bad base64" };
  }
  if (bytes.length === 0) {
    console.error("[captcha-autofill] bg rejected empty sample bytes:", msg.label, msg.reason);
    return { ok: false, error: "empty bytes" };
  }
  const db = await openDB();
  const store = db.transaction(DB_STORE, "readwrite").objectStore(DB_STORE);
  const hash = hashBytes(bytes);
  const existing = await new Promise((resolve, reject) => {
    const req = store.get(hash);
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
  if (existing) return { ok: true, dup: true };
  const count = await new Promise((resolve, reject) => {
    const req = store.count();
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
  if (count >= MAX_SAMPLES) return { ok: false, full: true };
  await new Promise((resolve, reject) => {
    const req = store.put({
      hash,
      name: `fail_${String(msg.ts)}_${hash.slice(0, 6)}.jpg`,
      blob: new Blob([bytes], { type: "image/jpeg" }),
      label: msg.label,
      confs: msg.confs,
      conf: msg.conf,
      reason: msg.reason,
      origin: msg.origin,
      pageUrl: msg.pageUrl,
      imgSrc: msg.imgSrc,
      ts: msg.ts,
      attempts: msg.attempts || 0,
    });
    req.onsuccess = () => resolve();
    req.onerror = () => reject(req.error);
  });
  return { ok: true };
}

async function countFails() {
  const db = await openDB();
  const count = await new Promise((resolve, reject) => {
    const req = db.transaction(DB_STORE).objectStore(DB_STORE).count();
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
  return { count };
}

async function clearFails() {
  const db = await openDB();
  await new Promise((resolve, reject) => {
    const req = db.transaction(DB_STORE, "readwrite").objectStore(DB_STORE).clear();
    req.onsuccess = () => resolve();
    req.onerror = () => reject(req.error);
  });
  return { ok: true };
}

// ------------------------------------------------------------------- ZIP (store)
const CRC_TABLE = (() => {
  const table = new Uint32Array(256);
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    table[n] = c >>> 0;
  }
  return table;
})();

function crc32(bytes) {
  let c = 0xffffffff;
  for (let i = 0; i < bytes.length; i++) {
    c = CRC_TABLE[(c ^ bytes[i]) & 0xff] ^ (c >>> 8);
  }
  return (c ^ 0xffffffff) >>> 0;
}

function dosDateTime(ts = Date.now()) {
  const d = new Date(ts);
  return {
    date: ((d.getFullYear() - 1980) << 9) | ((d.getMonth() + 1) << 5) | d.getDate(),
    time: (d.getHours() << 11) | (d.getMinutes() << 5) | (d.getSeconds() >> 1),
  };
}

function makeZip(entries) {
  // Entries: [{ name, data: Uint8Array }]. Stored uncompressed (JPEG/text are
  // already compact) so no deflate implementation is needed.
  const { date, time } = dosDateTime();
  const encoder = new TextEncoder();
  const parts = [];
  const central = [];
  let offset = 0;
  for (const entry of entries) {
    const nameBytes = encoder.encode(entry.name);
    const crc = crc32(entry.data);
    const header = new DataView(new ArrayBuffer(30));
    header.setUint32(0, 0x04034b50, true);
    header.setUint16(4, 20, true);
    header.setUint16(6, 0, true);
    header.setUint16(8, 0, true); // method 0 = store
    header.setUint16(10, time, true);
    header.setUint16(12, date, true);
    header.setUint32(14, crc, true);
    header.setUint32(18, entry.data.length, true);
    header.setUint32(22, entry.data.length, true);
    header.setUint16(26, nameBytes.length, true);
    header.setUint16(28, 0, true);
    parts.push(header.buffer, nameBytes, entry.data);

    const cd = new DataView(new ArrayBuffer(46));
    cd.setUint32(0, 0x02014b50, true);
    cd.setUint16(4, 20, true);
    cd.setUint16(6, 20, true);
    cd.setUint16(8, 0, true);
    cd.setUint16(10, 0, true);
    cd.setUint16(12, time, true);
    cd.setUint16(14, date, true);
    cd.setUint32(16, crc, true);
    cd.setUint32(20, entry.data.length, true);
    cd.setUint32(24, entry.data.length, true);
    cd.setUint16(28, nameBytes.length, true);
    cd.setUint16(30, 0, true);
    cd.setUint16(32, 0, true);
    cd.setUint16(34, 0, true);
    cd.setUint16(36, 0, true);
    cd.setUint32(38, 0, true);
    cd.setUint32(42, offset, true);
    central.push(cd.buffer, nameBytes);
    offset += 30 + nameBytes.length + entry.data.length;
  }
  const cdSize = central.reduce((sum, part) => sum + part.byteLength, 0);
  const eocd = new DataView(new ArrayBuffer(22));
  eocd.setUint32(0, 0x06054b50, true);
  eocd.setUint16(4, 0, true);
  eocd.setUint16(6, 0, true);
  eocd.setUint16(8, entries.length, true);
  eocd.setUint16(10, entries.length, true);
  eocd.setUint32(12, cdSize, true);
  eocd.setUint32(16, offset, true);
  eocd.setUint16(20, 0, true);
  return new Blob([...parts, ...central, eocd], { type: "application/zip" });
}

async function exportFails() {
  const db = await openDB();
  const all = await new Promise((resolve, reject) => {
    const req = db.transaction(DB_STORE).objectStore(DB_STORE).getAll();
    req.onsuccess = () => resolve(req.result || []);
    req.onerror = () => reject(req.error);
  });
  if (!all.length) return { ok: false, empty: true };
  all.sort((a, b) => a.ts - b.ts);

  const encoder = new TextEncoder();
  const entries = [];
  const rows = [["source_file", "label", "predicted_file", "min_confidence", "reason"]];
  const meta = [];
  for (const s of all) {
    entries.push({ name: s.name, data: new Uint8Array(await s.blob.arrayBuffer()) });
    rows.push([s.name, s.label, s.name, s.conf.toFixed(4), s.reason]);
    const { blob, ...rest } = s;
    meta.push(rest);
  }
  entries.push({
    name: "capture_manifest.csv",
    data: encoder.encode(rows.map((r) => r.join(",")).join("\n") + "\n"),
  });
  entries.push({
    name: "failed_meta.json",
    data: encoder.encode(JSON.stringify(meta, null, 2)),
  });

  const zip = makeZip(entries);
  // Hand the ZIP back as base64: response messages are JSON-serialized too,
  // so a raw ArrayBuffer would arrive as an empty object (corrupt download).
  // The popup decodes and downloads it (SW blob URLs also die with the worker).
  const data = new Uint8Array(await zip.arrayBuffer());
  let bin = "";
  for (let i = 0; i < data.length; i += 0x8000) {
    bin += String.fromCharCode.apply(null, data.subarray(i, i + 0x8000));
  }
  const stamp = new Date().toISOString().slice(0, 19).replace(/[:T]/g, "-");
  return { ok: true, count: all.length, filename: `captcha_failures_${stamp}.zip`, b64: btoa(bin) };
}

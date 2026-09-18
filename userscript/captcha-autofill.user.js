// ==UserScript==
// @name         验证码自动识别填充（通用 CRNN）
// @namespace    https://github.com/Conastin/VerificationCode
// @version      0.1.3
// @description  通用验证码识别：自动发现验证码与输入框，本地 CRNN 模型推理（无需服务器），低置信自动刷新重试。PoC 版。
// @author       Conastin
// @match        *://*/*
// @run-at       document-idle
// @connect      cdn.jsdelivr.net
// @connect      raw.githubusercontent.com
// @connect      github.com
// @connect      objects.githubusercontent.com
// @connect      fastly.jsdelivr.net
// @require      https://cdn.jsdelivr.net/npm/onnxruntime-web@1.19.2/dist/ort.min.js
// @grant        GM_xmlhttpRequest
// @grant        GM_getResourceURL
// @grant        GM_setValue
// @grant        GM_getValue
// @grant        GM_deleteValue
// @grant        GM_registerMenuCommand
// @noframes
// ==/UserScript==

/* eslint-disable no-undef */
"use strict";

(() => {
  // ------------------------------------------------------------ 常量
  const MODEL_VERSION = "v11-fp32-3.0.0";
  const MODEL_URLS = [
    "https://fastly.jsdelivr.net/gh/Conastin/VerificationCode@v3.0.0/browser/extension/model.onnx",
    "https://cdn.jsdelivr.net/gh/Conastin/VerificationCode@v3.0.0/browser/extension/model.onnx",
    "https://raw.githubusercontent.com/Conastin/VerificationCode/v3.0.0/browser/extension/model.onnx",
  ];
  const ORT_WASM_PATHS = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.19.2/dist/";
  const CHARSET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ";
  const BLANK = 36;
  const IMG_H = 48;
  const CONF_THRESHOLD = 0.9;
  const MAX_ATTEMPTS = 5;

  const state = {
    session: null,
    status: "idle", // idle | loading | ready | error
    config: null,   // {imgSelector, inputSelector, refreshSelector}
    chip: null,
  };

  // ------------------------------------------------------------ 工具
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

  function shortSelector(el) {
    if (el.id) return "#" + CSS.escape(el.id);
    if (el.name) {
      const byName = document.querySelector(
        `${el.tagName.toLowerCase()}[name="${CSS.escape(el.name)}"]`);
      if (byName === el) return `${el.tagName.toLowerCase()}[name="${CSS.escape(el.name)}"]`;
    }
    const path = [];
    let node = el;
    while (node && node.nodeType === 1 && path.length < 5) {
      const parent = node.parentElement;
      if (!parent) { path.unshift(node.tagName.toLowerCase()); break; }
      const sibs = Array.from(parent.children).filter((c) => c.tagName === node.tagName);
      const idx = sibs.indexOf(node);
      path.unshift(node.tagName.toLowerCase() + (sibs.length > 1 ? `:nth-of-type(${idx + 1})` : ""));
      node = parent;
    }
    return path.join(">");
  }

  function setChip(text, kind) {
    if (!state.chip) return;
    state.chip.textContent = text;
    state.chip.dataset.kind = kind || "info";
  }

  // ------------------------------------------------------------ 模型加载（三级容灾）
  async function fetchWithGM(url, onProgress) {
    return new Promise((resolve, reject) => {
      const req = GM_xmlhttpRequest({
        method: "GET", url, responseType: "arraybuffer",
        headers: { "Cache-Control": "no-cache" },
        onprogress: (e) => {
          if (onProgress && e.lengthComputable) onProgress(e.loaded, e.total);
        },
        onload: (res) => (res.status === 200 ? resolve(res.response) : reject(new Error(url + " -> " + res.status))),
        onerror: () => reject(new Error("network: " + url)),
        ontimeout: () => reject(new Error("timeout: " + url)),
      });
      if (req && req.abort) { /* keep handle */ }
    });
  }

  async function loadModelBytes() {
    // 1) Cache API 本地缓存（按版本, 只在 https 页面可用）
    if ("caches" in window && window.isSecureContext) {
      try {
        const cache = await caches.open("captcha-model");
        const hit = await cache.match(MODEL_VERSION);
        if (hit) return { bytes: await hit.arrayBuffer(), cached: true };
      } catch (e) { /* 页面 CSP 等原因, 走下一级 */ }
    }
    // 2) @resource（管理器预下载）
    try {
      const url = GM_getResourceURL("model", true);
      if (url && url.startsWith("blob:")) {
        const resp = await fetch(url);
        if (resp.ok) return { bytes: await resp.arrayBuffer(), cached: true };
      }
    } catch (e) { /* 管理器差异, 走直拉 */ }
    // 3) 多 CDN 直拉
    let lastErr = null;
    for (const url of MODEL_URLS) {
      try {
        const bytes = await fetchWithGM(url, (loaded, total) =>
          setChip(`模型下载 ${((loaded / total) * 100) | 0}%（首次约10MB，之后本地缓存）`, "load"));
        if ("caches" in window && window.isSecureContext) {
          try {
            const cache = await caches.open("captcha-model");
            await cache.put(MODEL_VERSION, new Response(bytes));
          } catch (e) { /* 缓存失败不影响使用 */ }
        }
        return { bytes, cached: false };
      } catch (e) { lastErr = e; }
    }
    throw lastErr || new Error("all model sources failed");
  }

  async function initModel() {
    if (state.session) return;
    state.status = "loading";
    setChip("模型加载中…", "load");
    try {
      const { bytes } = await loadModelBytes();
      ort.env.wasm.wasmPaths = ORT_WASM_PATHS;
      ort.env.wasm.numThreads = 1;
      state.session = await ort.InferenceSession.create(bytes, {
        executionProviders: ["wasm"],
        graphOptimizationLevel: "all",
      });
      state.status = "ready";
      setChip("模型就绪", "ok");
    } catch (e) {
      state.status = "error";
      setChip("模型加载失败: " + e.message, "err");
      throw e;
    }
  }

  // ------------------------------------------------------------ 识别（CRNN + CTC）
  function imageToTensor(image) {
    const natural = image.naturalWidth || image.width || 80;
    const naturalH = image.naturalHeight || image.height || 34;
    const w = Math.max(8, Math.round((natural * IMG_H) / naturalH));
    const canvas = document.createElement("canvas");
    canvas.width = w; canvas.height = IMG_H;
    const ctx = canvas.getContext("2d", { willReadFrequently: true });
    ctx.drawImage(image, 0, 0, w, IMG_H);
    const rgba = ctx.getImageData(0, 0, w, IMG_H).data;
    const plane = w * IMG_H;
    const data = new Float32Array(3 * plane);
    for (let p = 0; p < plane; p++) {
      data[p] = rgba[p * 4] / 255;
      data[plane + p] = rgba[p * 4 + 1] / 255;
      data[2 * plane + p] = rgba[p * 4 + 2] / 255;
    }
    return new ort.Tensor("float32", data, [1, 3, IMG_H, w]);
  }

  async function recognize(image) {
    const results = await state.session.run({ image: imageToTensor(image) });
    const logits = results.logits.data;
    const T = results.logits.dims[1];
    const label = []; const confs = [];
    let prev = -1;
    for (let t = 0; t < T; t++) {
      const base = t * (BLANK + 1);
      let max = -Infinity, best = 0;
      for (let c = 0; c <= BLANK; c++) {
        const v = logits[base + c];
        if (v > max) { max = v; best = c; }
      }
      let sum = 0;
      for (let c = 0; c <= BLANK; c++) sum += Math.exp(logits[base + c] - max);
      if (best !== prev && best !== BLANK) { label.push(CHARSET[best]); confs.push(1 / sum); }
      prev = best;
    }
    const conf = confs.length ? confs.reduce((a, b) => a + b, 0) / confs.length : 0;
    return { label: label.join(""), conf, perSlot: confs };
  }

  // ------------------------------------------------------------ 自动发现
  const IMG_KEYWORDS = /(captcha|verify|rand|seccode|vcode|kaptcha|checkcode|imgcode|validcode|randcode|getrandcode)/i;
  const INPUT_KEYWORDS = /(验证码|校验码|captcha|verification|randcode|vcode|seccode)/i;

  function plausibleImage(img) {
    const w = img.naturalWidth || img.width;
    const h = img.naturalHeight || img.height;
    if (!w || !h || w < 40 || w > 260 || h < 14 || h > 100) return false;
    const ratio = w / h;
    if (ratio < 1.2 || ratio > 8) return false;
    if (/logo|icon|avatar|banner|qrcode|barcode/i.test(img.src + " " + (img.id || "") + " " + (img.className || ""))) return false;
    if (img.src && /\.svg(\?|$)/i.test(img.src)) return false;
    const rect = img.getBoundingClientRect();
    if (rect.width === 0 || rect.height === 0) return false;
    return true;
  }

  function discover() {
    const imgs = Array.from(document.querySelectorAll("img, canvas"))
      .filter(plausibleImage);
    let best = null;
    for (const img of imgs) {
      const sig = `${img.src || ""} ${img.id || ""} ${img.className || ""} ${img.getAttribute("onclick") || ""}`;
      const kw = IMG_KEYWORDS.test(sig);
      // 输入框: 从最近公共容器往外找
      let scope = img.closest("form, div, section");
      for (let depth = 0; depth < 4 && scope; depth++) {
        const inputs = Array.from(scope.querySelectorAll('input[type="text"], input:not([type])'))
          .filter((inp) => {
            if (inp.disabled || inp.readOnly || inp.offsetParent === null) return false;
            const s = `${inp.id || ""} ${inp.name || ""} ${inp.placeholder || ""} ${inp.getAttribute("aria-label") || ""} ${inp.maxLength || 8}`;
            if (/\b(email|tel|phone|user|password|search)\b/i.test(s)) return false;
            return INPUT_KEYWORDS.test(s) || (inp.maxLength >= 4 && inp.maxLength <= 8);
          });
        if (inputs.length) {
          const input = inputs[0];
          const score = (kw ? 10 : 0) + (depth === 0 ? 5 : 3 - depth);
          if (!best || score > best.score) {
            best = { img, input, score, keyword: kw };
          }
          break;
        }
        scope = scope.parentElement;
      }
    }
    return best;
  }

  // ------------------------------------------------------------ UI（状态角标 + 发现横幅）
  function injectStyle() {
    const style = document.createElement("style");
    style.textContent = `
.cap-chip{position:fixed;right:14px;bottom:14px;z-index:2147483000;background:#24292f;color:#fff;
  padding:6px 12px;border-radius:8px;font:12px/1.6 Consolas,"Microsoft YaHei",monospace;
  box-shadow:0 2px 10px rgba(0,0,0,.35);opacity:.92}
.cap-chip[data-kind="ok"]{background:#1a7f37}.cap-chip[data-kind="err"]{background:#b62324}
.cap-chip[data-kind="load"]{background:#9a6700}.cap-chip[data-kind="busy"]{background:#0969da}
.cap-banner{position:fixed;top:0;left:50%;transform:translateX(-50%);z-index:2147483000;
  background:#fff;border:1px solid #d0d7de;border-top:none;border-radius:0 0 10px 10px;
  padding:10px 14px;display:flex;align-items:center;gap:12px;
  font:13px/1.5 "Microsoft YaHei",sans-serif;color:#1f2328;box-shadow:0 2px 12px rgba(0,0,0,.18)}
.cap-banner img{width:96px;height:40px;object-fit:contain;background:#f6f8fa;border:1px solid #d0d7de;border-radius:4px}
.cap-btn{cursor:pointer;border:1px solid #d0d7de;background:#f6f8fa;border-radius:6px;padding:4px 12px;font-size:13px}
.cap-btn.primary{background:#1f883d;color:#fff;border-color:#1f883d}
.cap-btn:hover{filter:brightness(.97)}
.cap-hl{outline:3px solid #0969da !important;outline-offset:2px}
.cap-note{color:#656d76;font-size:12px}
    `;
    document.head.appendChild(style);
  }

  function makeChip() {
    const chip = document.createElement("div");
    chip.className = "cap-chip";
    chip.dataset.kind = "info";
    chip.textContent = "验证码脚本已加载";
    document.body.appendChild(chip);
    state.chip = chip;
  }

  function showDiscoveryBanner(cand) {
    if (document.querySelector(".cap-banner")) return;
    const banner = document.createElement("div");
    banner.className = "cap-banner";
    const img = document.createElement("img");
    img.src = cand.img.src || "";
    if (cand.img.tagName === "CANVAS") img.src = cand.img.toDataURL();
    const text = document.createElement("div");
    text.innerHTML =
      `<b>检测到验证码</b><div class="cap-note">输入框: <code>${shortSelector(cand.input)}</code>` +
      `${cand.keyword ? "" : "（低置信猜测，请确认）"}</div>`;
    const btnYes = document.createElement("button");
    btnYes.className = "cap-btn primary"; btnYes.textContent = "启用自动填充";
    const btnAdj = document.createElement("button");
    btnAdj.className = "cap-btn"; btnAdj.textContent = "微调";
    const btnNo = document.createElement("button");
    btnNo.className = "cap-btn"; btnNo.textContent = "忽略本站";
    banner.append(img, text, btnYes, btnAdj, btnNo);
    document.body.appendChild(banner);

    cand.img.classList.add("cap-hl");
    cand.input.classList.add("cap-hl");

    const dismiss = () => { banner.remove(); cand.img.classList.remove("cap-hl"); cand.input.classList.remove("cap-hl"); };
    btnYes.onclick = async () => {
      state.config = {
        imgSelector: shortSelector(cand.img),
        inputSelector: shortSelector(cand.input),
        refreshSelector: shortSelector(cand.img),
      };
      await GM_setValue("cfg:" + location.origin, state.config);
      dismiss();
      runAutoFill();
    };
    btnAdj.onclick = () => { dismiss(); startAdjust(cand); };
    btnNo.onclick = async () => {
      await GM_setValue("ignore:" + location.origin, true);
      dismiss();
    };
    // 20 秒无操作自动收起
    setTimeout(dismiss, 20000);
  }

  function startAdjust(cand) {
    const tip = document.createElement("div");
    tip.className = "cap-banner";
    tip.innerHTML = "<b>微调</b>：点击验证码图片 → 再点击输入框";
    document.body.appendChild(tip);
    const pick = (label, filter) => new Promise((resolve) => {
      const handler = (e) => {
        if (!filter(e.target)) return;
        e.preventDefault(); e.stopPropagation();
        document.removeEventListener("click", handler, true);
        resolve(e.target);
      };
      document.addEventListener("click", handler, true);
    });
    (async () => {
      const img = await pick("图片", (t) => t.tagName === "IMG" || t.tagName === "CANVAS");
      const input = await pick("输入框", (t) => (t.tagName === "INPUT" && t.type !== "password"));
      state.config = {
        imgSelector: shortSelector(img),
        inputSelector: shortSelector(input),
        refreshSelector: shortSelector(img),
      };
      await GM_setValue("cfg:" + location.origin, state.config);
      tip.remove();
      runAutoFill();
    })();
  }

  // ------------------------------------------------------------ 自动填充主循环
  function waitImage(img, timeoutMs = 10000) {
    return new Promise((resolve) => {
      if (img.complete && img.naturalWidth > 0) return resolve(true);
      const t = setTimeout(() => { cleanup(); resolve(false); }, timeoutMs);
      const ok = () => { clearTimeout(t); cleanup(); resolve(true); };
      const cleanup = () => { img.removeEventListener("load", ok); img.removeEventListener("error", ok); };
      img.addEventListener("load", ok); img.addEventListener("error", ok);
    });
  }

  function refreshCaptcha(img) {
    try { img.click(); } catch (e) { /* ignore */ }
  }

  function fillInput(input, value) {
    const proto = input instanceof HTMLTextAreaElement ? HTMLTextAreaElement.prototype : HTMLInputElement.prototype;
    const setter = Object.getOwnPropertyDescriptor(proto, "value").set;
    setter.call(input, value);
    input.dispatchEvent(new Event("input", { bubbles: true }));
    input.dispatchEvent(new Event("change", { bubbles: true }));
  }

  async function runAutoFill() {
    if (!state.config) return;
    await initModel();
    for (let attempt = 1; attempt <= MAX_ATTEMPTS; attempt++) {
      const img = document.querySelector(state.config.imgSelector);
      const input = document.querySelector(state.config.inputSelector);
      if (!img || !input) { setChip("等待页面元素…", "load"); return; }
      // 用户已手动填写则不再覆盖
      if (input.value && input.value.length >= 4) { setChip("已填充（手动）", "ok"); return; }
      if (!(await waitImage(img))) continue;
      setChip(`识别中…（${attempt}/${MAX_ATTEMPTS}）`, "busy");
      let result;
      try { result = await recognize(img); }
      catch (e) { setChip("识别失败: " + e.message, "err"); return; }
      if (result.conf >= CONF_THRESHOLD && result.label.length === 4) {
        fillInput(input, result.label);
        setChip(`已填充 ${result.label}（${result.conf.toFixed(2)}）`, "ok");
        return;
      }
      setChip(`置信度不足 ${result.conf.toFixed(2)}，刷新重试`, "load");
      refreshCaptcha(img);
      await sleep(1200);
    }
    setChip("多次置信度不足，请手动输入", "err");
  }

  // ------------------------------------------------------------ 入口
  async function main() {
    if (window.top !== window.self) return; // @noframes 兜底
    injectStyle();
    makeChip();

    GM_registerMenuCommand("手动识别当前页", () => {
      state.config = state.config || GM_getValue("cfg:" + location.origin);
      if (!state.config) {
        const cand = discover();
        if (cand) return showDiscoveryBanner(cand);
        return setChip("未检测到验证码", "err");
      }
      runAutoFill();
    });
    GM_registerMenuCommand("清除本站配置", async () => {
      await GM_deleteValue("cfg:" + location.origin);
      location.reload();
    });

    const ignored = await GM_getValue("ignore:" + location.origin);
    if (ignored) { setChip("本站已忽略（菜单可重置）", "info"); return; }

    state.config = await GM_getValue("cfg:" + location.origin);
    if (state.config) {
      runAutoFill();
      return;
    }
    // 无配置: 延迟自动发现（等页面渲染稳定）
    await sleep(1500);
    const cand = discover();
    if (cand) showDiscoveryBanner(cand);
    else setChip("未检测到验证码", "info");
  }

  main().catch((e) => console.error("[captcha-us]", e));
})();

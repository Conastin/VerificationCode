/* Universal content script.
 *
 * Two captcha modes, configured per-origin from the visual config bar
 * (right-click "配置本站验证码" or the popup button), by picking elements
 * on the page:
 *
 * - image: an <img> captcha. Auto-fill recognizes the image (conf >= 0.9)
 *   and fills the input; low-confidence attempts click the image to refresh
 *   and retry.
 * - text: the captcha is rendered as plain DOM text (e.g. skylumo.cc).
 *   Auto-fill reads the element text, extracts the code (textPattern), and
 *   fills the input. No ONNX model is loaded in this mode.
 *
 * A MutationObserver covers manual refreshes (image src change / text
 * element change). Status badge auto-hides after a few seconds.
 */
"use strict";

(() => {
  const MAX_ATTEMPTS = 5;
  const RESET_WINDOW_MS = 60000;
  const CONF_THRESHOLD = 0.9;
  const BADGE_HIDE_MS = 4000;
  const EXT_BASE = chrome.runtime.getURL("");
  // Default extraction regex for text-mode captchas (3-8 alphanumerics).
  const DEFAULT_TEXT_PATTERN = /^[0-9A-Za-z]{3,8}$/;
  // Fail-sample collection: server-side captcha-reject hints (dynamic only,
  // see setupFailHintMonitor) and dedup window.
  const FAIL_HINT_PATTERN =
    /验证码.{0,8}(不正确|错误|已失效|已过期|无效|校验失败|请重新|有误)/;
  const FAIL_HINT_DEDUP_MS = 3000;

  let badge = null;
  let badgeTimer = null;
  let attemptCount = 0;
  let lastAttemptAt = 0;
  let busy = false;
  let lastFilled = "";
  let autoRetry = false; // extension clicked the refresh button; may overwrite input
  let config = null; // { mode, imgSelector?, textSelector?, inputSelector, textPattern?, refreshSelector? }
  let lastSnap = null; // { dataUrl, imgSrc, label, confs, conf } of the last fill
  let failHintLastAt = 0;

  // ------------------------------------------------------------------ status
  function showStatus(text, kind) {
    if (!badge) {
      badge = document.createElement("div");
      badge.style.cssText =
        "position:fixed;right:16px;bottom:16px;z-index:999999;" +
        "padding:10px 16px;border-radius:8px;font-size:13px;color:#fff;" +
        "font-family:'Microsoft YaHei',sans-serif;box-shadow:0 2px 8px rgba(0,0,0,.3);" +
        "max-width:340px;opacity:1;transition:opacity .5s;";
      document.body.appendChild(badge);
    }
    badge.style.background =
      kind === "ok" ? "#2f6b0f" : kind === "fail" ? "#c00" : "#b8860b";
    badge.textContent = text;
    badge.style.opacity = "1";
    clearTimeout(badgeTimer);
    badgeTimer = setTimeout(() => {
      badge.style.opacity = "0";
      setTimeout(() => {
        if (badge) { badge.remove(); badge = null; }
      }, 600);
    }, BADGE_HIDE_MS);
  }

  // ------------------------------------------------------------- DOM helpers
  function waitFor(selector, timeoutMs = 15000) {
    return new Promise((resolve) => {
      const el = document.querySelector(selector);
      if (el) return resolve(el);
      const observer = new MutationObserver(() => {
        const found = document.querySelector(selector);
        if (found) { observer.disconnect(); resolve(found); }
      });
      observer.observe(document.body, { childList: true, subtree: true });
      setTimeout(() => { observer.disconnect(); resolve(null); }, timeoutMs);
    });
  }

  function waitForImageLoad(img, timeoutMs = 10000) {
    return new Promise((resolve) => {
      if (img.complete && img.naturalWidth > 0) return resolve(true);
      const onLoad = () => { cleanup(); resolve(true); };
      const onError = () => { cleanup(); resolve(false); };
      const cleanup = () => {
        img.removeEventListener("load", onLoad);
        img.removeEventListener("error", onError);
        clearTimeout(timer);
      };
      const timer = setTimeout(() => { cleanup(); resolve(img.complete); }, timeoutMs);
      img.addEventListener("load", onLoad);
      img.addEventListener("error", onError);
    });
  }

  function setInputValue(el, value) {
    const proto = el instanceof HTMLTextAreaElement
      ? window.HTMLTextAreaElement.prototype
      : window.HTMLInputElement.prototype;
    Object.getOwnPropertyDescriptor(proto, "value").set.call(el, value);
    el.dispatchEvent(new Event("input", { bubbles: true }));
    el.dispatchEvent(new Event("change", { bubbles: true }));
  }

  function bestSelector(el) {
    if (el.id) return "#" + CSS.escape(el.id);
    if (el.name) return el.tagName.toLowerCase() + '[name="' + CSS.escape(el.name) + '"]';
    const path = [];
    let node = el;
    while (node && node.nodeType === 1 && node !== document.body) {
      let part = node.tagName.toLowerCase();
      const siblings = Array.from(node.parentNode.children)
        .filter((c) => c.tagName === node.tagName);
      if (siblings.length > 1) {
        part += ":nth-of-type(" + (siblings.indexOf(node) + 1) + ")";
      }
      path.unshift(part);
      node = node.parentNode;
    }
    return path.join(" > ");
  }

  // Selector for persisted configs: prefer id/name, then the first class that
  // alone uniquely identifies the element, then the full class list, and fall
  // back to the DOM path. Short selectors survive SPA re-renders better than
  // paths and stay readable in the config bar.
  function uniqueSelector(el) {
    if (el.id) return "#" + CSS.escape(el.id);
    if (el.name) return el.tagName.toLowerCase() + '[name="' + CSS.escape(el.name) + '"]';
    const tag = el.tagName.toLowerCase();
    if (el.classList.length) {
      for (const c of el.classList) {
        const sel = tag + "." + CSS.escape(c);
        if (el.matches(sel) && document.querySelectorAll(sel).length === 1) return sel;
      }
      const full = tag + "." + Array.from(el.classList).map((c) => CSS.escape(c)).join(".");
      if (el.matches(full) && document.querySelectorAll(full).length === 1) return full;
    }
    return bestSelector(el);
  }

  // Click targets are often bare text nodes/span wrappers with no stable
  // hooks; walk up to the nearest element that carries an id/name/class so
  // the generated selector survives SPA re-renders. Interactive elements
  // (img/input/textarea/canvas) are kept as-is.
  function resolvePickElement(el) {
    if (["IMG", "CANVAS", "INPUT", "TEXTAREA"].includes(el.tagName)) return el;
    let node = el;
    while (node && node !== document.body) {
      if (node.id || node.name || node.classList.length) return node;
      node = node.parentElement;
    }
    return el;
  }

  // ------------------------------------------------------------ fail samples
  async function snapshotImage(img) {
    const w = img.naturalWidth || img.width;
    const h = img.naturalHeight || img.height;
    if (!w || !h) return null;
    const canvas = document.createElement("canvas");
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext("2d", { willReadFrequently: true });
    ctx.drawImage(img, 0, 0, w, h);
    try {
      return canvas.toDataURL("image/jpeg", 0.85);
    } catch (err) {
      // Cross-origin image tainted the canvas; nothing we can save.
      console.warn("[captcha-autofill] snapshot skipped (tainted canvas):", err);
      return null;
    }
  }

  async function saveFailSample(reason, snap, extra = {}) {
    if (!snap || !snap.dataUrl) return;
    try {
      const byteStr = atob(snap.dataUrl.split(",")[1]);
      const bytes = new Uint8Array(byteStr.length);
      for (let i = 0; i < byteStr.length; i++) bytes[i] = byteStr.charCodeAt(i);
      // runtime messages are JSON-serialized and cannot carry binary
      // (TypedArray/ArrayBuffer arrive as empty objects) — send base64.
      let bin = "";
      for (let i = 0; i < bytes.length; i += 0x8000) {
        bin += String.fromCharCode.apply(null, bytes.subarray(i, i + 0x8000));
      }
      const resp = await chrome.runtime.sendMessage({
        type: "saveFailSample",
        reason,
        label: snap.label,
        confs: snap.confs,
        conf: snap.conf,
        imgSrc: snap.imgSrc,
        attempts: extra.attempts || 0,
        ts: Date.now(),
        b64: btoa(bin),
      });
      if (resp?.ok) {
        showStatus(
          resp.dup ? "失败样本已存在（跳过）" : "已保存失败样本，可导出修正",
          "ok"
        );
      } else if (resp?.full) {
        showStatus("失败样本已达上限，请打开扩展导出", "fail");
      }
    } catch (err) {
      console.warn("[captcha-autofill] save fail sample failed:", err);
    }
  }

  async function captureSubmitFailure() {
    // Prefer the snapshot of the captcha that was actually submitted (the
    // site usually refreshes the image after a rejected submit).
    let snap = lastSnap;
    if (!snap) {
      // Manual submit: best-effort — recognize the current image on the fly.
      const img = document.querySelector(config.imgSelector);
      if (!img) return;
      try {
        const dataUrl = await snapshotImage(img);
        if (!dataUrl) return;
        const result = await CaptchaRecognizer.recognize(img);
        snap = {
          dataUrl,
          imgSrc: img.src,
          label: result.label,
          confs: result.perSlot,
          conf: result.conf,
        };
      } catch (err) {
        console.warn("[captcha-autofill] manual failure capture failed:", err);
        return;
      }
    }
    await saveFailSample("submit_failed", snap);
  }

  async function checkPageLoadSubmitFailure() {
    // Rejected submits often reload the whole page: the reject hint is
    // already in the DOM when this script runs (no mutation to observe) and
    // the in-memory snapshot is gone. The snapshot cached at fill time
    // survives via the background (storage.session) — use it if a fresh hint
    // is present.
    const SNAP_TTL_MS = 60000;
    try {
      const resp = await chrome.runtime.sendMessage({ type: "getFailSnap" });
      const failSnap = resp?.failSnap;
      if (!failSnap || Date.now() - failSnap.ts > SNAP_TTL_MS) return;
      // Same host: a rejected submit may land on a different path (e.g.
      // check.action) than the login page the snapshot was taken on.
      const sameHost =
        failSnap.pageUrl && new URL(failSnap.pageUrl).host === location.host;
      if (!sameHost) return;
      const bodyText = document.body.textContent || "";
      if (!FAIL_HINT_PATTERN.test(bodyText)) return;
      await saveFailSample("submit_failed", failSnap);
      await chrome.runtime.sendMessage({ type: "clearFailSnap" }).catch(() => {});
    } catch (err) {
      console.warn("[captcha-autofill] page-load failure check skipped:", err);
    }
  }

  function setupFailHintMonitor() {
    // Detect rejected submits on pages that update in place (no reload).
    // Three shapes: newly inserted text node, newly inserted element, or
    // text changed inside an existing element (characterData). Static page
    // text (e.g. the input label "图形验证码") is already in the DOM before
    // this observer starts, so it can never match.
    const monitor = new MutationObserver((mutations) => {
      if (Date.now() - failHintLastAt < FAIL_HINT_DEDUP_MS) return;
      const input = document.querySelector(config.inputSelector);
      if (!(lastFilled || (input && input.value))) return; // nothing submitted
      for (const m of mutations) {
        let text = "";
        if (m.type === "characterData") {
          text = m.target.nodeValue || "";
        } else {
          for (const node of m.addedNodes) {
            if (node.nodeType === Node.TEXT_NODE) {
              text = node.nodeValue || "";
              break;
            }
            if (node.nodeType === Node.ELEMENT_NODE) {
              const t = node.textContent || "";
              if (t.length <= 5000) { text = t; break; }
            }
          }
        }
        if (!text || text.length > 5000) continue;
        if (FAIL_HINT_PATTERN.test(text)) {
          failHintLastAt = Date.now();
          if (config.mode === "text") {
            // Text captchas carry no image snapshot: retry via the optional
            // refresh button, otherwise just warn. autoRetry lets the next
            // fill overwrite the stale wrong value left in the input.
            if (config.refreshSelector) {
              const btn = document.querySelector(config.refreshSelector);
              if (btn) {
                autoRetry = true;
                btn.click();
                showStatus("验证码不正确，已点击更换验证码重试", "retry");
                return;
              }
            }
            showStatus("验证码不正确，请手动刷新验证码", "fail");
            return;
          }
          captureSubmitFailure();
          return;
        }
      }
    });
    monitor.observe(document.body, {
      childList: true,
      subtree: true,
      characterData: true,
    });
  }

  // ------------------------------------------------------------- auto-fill
  async function runTextFillCore() {
    const input = document.querySelector(config.inputSelector);
    if (!input) { showStatus("未找到验证码输入框", "fail"); return; }
    if (input.value && input.value !== lastFilled && !autoRetry) return; // user typed
    autoRetry = false;
    attemptCount++;
    lastAttemptAt = Date.now();

    const textEl = document.querySelector(config.textSelector);
    if (!textEl) { showStatus("未找到验证码文本元素", "fail"); return; }
    const raw = (textEl.textContent || "").trim();
    let pattern = DEFAULT_TEXT_PATTERN;
    try {
      pattern = config.textPattern ? new RegExp(config.textPattern) : DEFAULT_TEXT_PATTERN;
    } catch (err) {
      console.warn("[captcha-autofill] invalid textPattern, falling back:", err);
    }
    const m = raw.match(pattern);
    if (!m) {
      showStatus(`验证码文本无法匹配: ${raw.slice(0, 24) || "(空)"}`, "fail");
      return;
    }
    const code = m[0];
    if (code === lastFilled && input.value === code) return; // already filled
    setInputValue(input, code);
    lastFilled = code;
    showStatus(`已填充 ${code}`, "ok");
  }

  async function runAutoFill() {
    if (busy || !config) return;
    busy = true;
    try {
      if (config.mode === "text") {
        await runTextFillCore();
        return;
      }
      const input = document.querySelector(config.inputSelector);
      if (!input) { showStatus("未找到验证码输入框", "fail"); return; }
      if (input.value && input.value !== lastFilled) return; // user typed
      attemptCount++;
      lastAttemptAt = Date.now();

      const img = document.querySelector(config.imgSelector);
      if (!img) { showStatus("未找到验证码图片", "fail"); return; }

      const loaded = await waitForImageLoad(img);
      if (!loaded) { showStatus("验证码图片加载异常", "fail"); return; }

      let result;
      try {
        result = await CaptchaRecognizer.recognize(img);
      } catch (err) {
        console.error("[captcha-autofill] recognize failed:", err);
        showStatus("识别失败: " + err, "fail");
        return;
      }

      if (result.conf >= CONF_THRESHOLD) {
        setInputValue(input, result.label);
        lastFilled = result.label;
        const dataUrl = await snapshotImage(img);
        if (dataUrl) {
          const snap = { dataUrl, imgSrc: img.src, label: result.label, confs: result.perSlot, conf: result.conf };
          lastSnap = snap;
          // Survive a whole-page reload after a rejected submit: relayed to
          // the background (see checkPageLoadSubmitFailure).
          chrome.runtime.sendMessage({
            type: "storeFailSnap",
            snap: { ...snap, pageUrl: location.href, ts: Date.now() },
          }).catch((err) => console.warn("[captcha-autofill] storeFailSnap failed:", err));
        }
        showStatus(`已填充 ${result.label}（置信度 ${result.conf.toFixed(2)}）`, "ok");
        return;
      }
      // Low-confidence attempts are hard cases worth collecting for retraining.
      const dataUrl = await snapshotImage(img);
      if (dataUrl) {
        await saveFailSample("low_conf", {
          dataUrl,
          imgSrc: img.src,
          label: result.label,
          confs: result.perSlot,
          conf: result.conf,
        }, { attempts: attemptCount });
      }
      if (attemptCount >= MAX_ATTEMPTS) {
        showStatus(`连续 ${MAX_ATTEMPTS} 次置信度不足，请手动输入`, "fail");
        return;
      }
      showStatus(`置信度不足 ${result.conf.toFixed(2)}，正在刷新重试（${attemptCount}/${MAX_ATTEMPTS}）`, "retry");
      img.click();
    } finally {
      busy = false;
    }
  }

  function scheduleAutoFill() {
    if (!config) return;
    const input = document.querySelector(config.inputSelector);
    if (input && input.value && input.value !== lastFilled && !autoRetry) return;
    if (busy) return;
    if (Date.now() - lastAttemptAt > RESET_WINDOW_MS) attemptCount = 0;
    if (attemptCount >= MAX_ATTEMPTS) return;
    setTimeout(runAutoFill, 250);
  }

  async function setupAutoFill(cfg) {
    config = cfg;
    // Check for a rejected submit that landed us on a reloaded page before
    // the auto-fill below overwrites the persisted snapshot.
    await checkPageLoadSubmitFailure();
    try {
      await CaptchaRecognizer.init(EXT_BASE + "model.onnx", EXT_BASE + "vendor/onnxruntime-web/");
    } catch (err) {
      console.error("[captcha-autofill] model init failed:", err);
      showStatus("识别模型加载失败", "fail");
      return;
    }
    const img = await waitFor(config.imgSelector);
    if (!img) return;
    new MutationObserver(() => scheduleAutoFill()).observe(
      img, { attributes: true, attributeFilter: ["src"] }
    );
    // Page may replace the whole image after a failed submit.
    new MutationObserver(() => {
      const current = document.querySelector(config.imgSelector);
      if (current && current !== img) scheduleAutoFill();
    }).observe(document.body, { childList: true, subtree: true });
    setupFailHintMonitor();
    scheduleAutoFill();
  }

  async function setupTextAutoFill(cfg) {
    config = cfg;
    // Text mode needs no ONNX model: the captcha is read straight from the
    // DOM, so auto-fill starts with zero warm-up cost.
    const textEl = await waitFor(config.textSelector);
    if (!textEl) {
      showStatus("未找到验证码文本元素，请检查选择器", "fail");
      return;
    }
    // Manual "更换验证码" refreshes replace the text -> re-read and re-fill.
    new MutationObserver(() => {
      if (!busy && Date.now() - lastAttemptAt > 500) scheduleAutoFill();
    }).observe(textEl, { childList: true, characterData: true, subtree: true });
    // The whole page may rebuild the element after a rejected submit.
    new MutationObserver(() => {
      const current = document.querySelector(config.textSelector);
      if (current && current !== textEl) scheduleAutoFill();
    }).observe(document.body, { childList: true, subtree: true });
    setupFailHintMonitor();
    scheduleAutoFill();
  }

  // --------------------------------------------------- selector test (popup)
  async function handleTestSelectors(sel) {
    const query = (selector) => {
      if (!selector) return { skipped: true };
      try {
        const els = Array.from(document.querySelectorAll(selector));
        return { found: els.length > 0, count: els.length };
      } catch (err) {
        return { found: false, error: String(err) };
      }
    };
    const paint = (selector, color) => {
      if (!selector) return;
      let els = [];
      try { els = Array.from(document.querySelectorAll(selector)); } catch { return; }
      for (const el of els.slice(0, 8)) {
        el.style.outline = `3px solid ${color}`;
        setTimeout(() => { el.style.outline = ""; }, 4000);
      }
    };
    paint(sel.textSelector, "#e11d48");   // text element -> red
    paint(sel.inputSelector, "#2f6b0f");  // input -> green
    paint(sel.refreshSelector, "#2563eb"); // refresh button -> blue
    const out = {
      origin: location.origin,
      text: { ...query(sel.textSelector) },
      input: { ...query(sel.inputSelector) },
      refresh: { ...query(sel.refreshSelector) },
    };
    if (out.text.found) {
      const el = document.querySelector(sel.textSelector);
      const raw = (el.textContent || "").trim();
      out.text.sample = raw.slice(0, 30);
      let pattern = DEFAULT_TEXT_PATTERN;
      try {
        pattern = sel.textPattern ? new RegExp(sel.textPattern) : DEFAULT_TEXT_PATTERN;
      } catch { /* keep default */ }
      const m = raw.match(pattern);
      out.text.extracted = m ? m[0] : null;
    }
    return out;
  }

  // --------------------------------------------------- visual config bar
  // Click-to-pick config flow: a floating bar on top of the page lets the
  // user click the captcha element (image or text), the input, and optionally
  // the refresh button. Selectors are generated automatically; saving applies
  // instantly via chrome.storage.onChanged.
  const PICK_LABEL = { captcha: "验证码元素", input: "输入框", refresh: "更换验证码按钮" };
  const PICK_COLOR = { captcha: "#e11d48", input: "#2f6b0f", refresh: "#b8860b" };
  const PICK_GLOW = {
    captcha: "rgba(225,29,72,.75)",
    input: "rgba(47,107,15,.75)",
    refresh: "rgba(184,134,11,.75)",
    hover: "rgba(37,99,235,.75)",
  };

  // Highlight = 3px colored outline + white isolation ring + outer glow, so
  // it stays clearly visible on any page background / framework (plain
  // outlines get lost against busy or light-colored layouts). An inset ring
  // mirrors the outline inside the element: when an overflow:hidden ancestor
  // clips the outer ring (common in card-style login forms), the inner ring
  // keeps the highlight fully visible.
  function paintHighlight(el, kind) {
    if (!el || !el.style) return;
    const color = kind === "hover" ? "#2563eb" : PICK_COLOR[kind];
    el.style.outline = `3px solid ${color}`;
    el.style.boxShadow =
      `inset 0 0 0 3px ${color}, 0 0 0 2px #fff, 0 0 12px 3px ${PICK_GLOW[kind]}`;
  }

  function clearHighlight(el) {
    if (!el || !el.style) return;
    el.style.outline = "";
    el.style.boxShadow = "";
  }

  // A picked element may serve several roles (e.g. on image captchas the
  // "更换验证码按钮" is the image itself). Keep the first role's highlight;
  // later roles reuse it without repainting over it.
  function highlightKindOf(el, excludeKind) {
    return Object.keys(configPicked).find(
      (k) => k !== excludeKind && configPicked[k] && configPicked[k].el === el
    );
  }

  // Restore a picked element's role color after hover moves away; elements
  // that are not picked simply lose their highlight.
  function restoreOrClearHighlight(el) {
    const k = highlightKindOf(el);
    if (k) paintHighlight(el, k);
    else clearHighlight(el);
  }

  let configBarEl = null;
  let configPick = null;   // "captcha" | "input" | "refresh" | null
  let configMode = null;   // "text" | "image", derived from the picked captcha
  let configPicked = {};   // { captcha: {el, sel}, input: {...}, refresh: {...} }
  let configHoverEl = null;

  function configStatus(text) {
    const el = configBarEl && configBarEl.querySelector(".vcbar-status");
    if (el) el.textContent = text;
  }

  function configBarButton(kind) {
    return configBarEl && configBarEl.querySelector(`[data-kind="${kind}"]`);
  }

  function refreshConfigBar() {
    if (!configBarEl) return;
    for (const kind of ["captcha", "input", "refresh"]) {
      const btn = configBarButton(kind);
      const sel = configPicked[kind] && configPicked[kind].sel;
      const label = sel
        ? `${PICK_LABEL[kind]} · ${sel}`
        : PICK_LABEL[kind] + (kind === "refresh" ? "(可选)" : "");
      btn.textContent = label;
      btn.classList.toggle("picked", !!sel);
      btn.classList.toggle("active", configPick === kind);
    }
    const typeEl = configBarEl.querySelector(".vcbar-type");
    if (typeEl) {
      typeEl.textContent = configMode
        ? `类型: ${configMode === "image" ? "图片验证码" : "纯文字验证码"}`
        : "类型: 未选择";
    }
  }

  function clearConfigPick() {
    configPick = null;
    if (configHoverEl) { restoreOrClearHighlight(configHoverEl); configHoverEl = null; }
    refreshConfigBar();
  }

  function startConfigPick(kind) {
    if (configPick === kind) { clearConfigPick(); return; }
    clearConfigPick();
    configPick = kind;
    configStatus(`请点击页面上的${PICK_LABEL[kind]}（悬停高亮，Esc 取消）`);
    refreshConfigBar();
  }

  function onConfigMove(e) {
    if (!configPick) return;
    const raw = e.target;
    if (!raw || raw.nodeType !== 1 || raw === document.body || raw === document.documentElement) return;
    if (configBarEl && configBarEl.contains(raw)) return;
    const el = resolvePickElement(raw);
    if (configHoverEl && configHoverEl !== el) restoreOrClearHighlight(configHoverEl);
    if (configHoverEl !== el) {
      if (highlightKindOf(el)) return; // already picked: keep its role color
      configHoverEl = el;
      paintHighlight(el, "hover");
    }
  }

  function onConfigClick(e) {
    if (!configPick) return;
    if (configBarEl && configBarEl.contains(e.target)) return;
    e.preventDefault();
    e.stopImmediatePropagation();
    const raw = e.target;
    if (!raw || raw.nodeType !== 1 || raw === document.body || raw === document.documentElement) return;
    const el = resolvePickElement(raw);
    if (configHoverEl) { restoreOrClearHighlight(configHoverEl); configHoverEl = null; }
    const kind = configPick;
    // Replacing a previously picked element: clear its old highlight, unless
    // another role still points at it (then repaint in that role's color).
    if (configPicked[kind] && configPicked[kind].el) {
      const old = configPicked[kind].el;
      const otherKind = highlightKindOf(old, kind);
      if (otherKind) paintHighlight(old, otherKind);
      else clearHighlight(old);
    }
    if (kind === "captcha") {
      configMode = el.tagName === "IMG" ? "image" : "text";
    }
    configPicked[kind] = { el, sel: uniqueSelector(el) };
    const mergedKind = highlightKindOf(el, kind);
    if (mergedKind) {
      configStatus(`"${PICK_LABEL[kind]}"与"${PICK_LABEL[mergedKind]}"为同一元素，已合并标记`);
    } else {
      paintHighlight(el, kind);
      configStatus(`已选中${PICK_LABEL[kind]}: ${configPicked[kind].sel}`);
    }
    configPick = null;
    refreshConfigBar();
  }

  function onConfigKey(e) {
    if (e.key === "Escape") {
      if (configPick) clearConfigPick();
      else hideConfigBar();
    }
  }

  function saveConfigBar() {
    if (!configPicked.captcha) { configStatus("请先选择验证码元素"); return; }
    if (!configPicked.input) { configStatus("请先选择输入框"); return; }
    let cfg;
    if (configMode === "image") {
      cfg = {
        mode: "image",
        imgSelector: configPicked.captcha.sel,
        inputSelector: configPicked.input.sel,
        refreshSelector: configPicked.refresh ? configPicked.refresh.sel : undefined,
      };
    } else {
      cfg = {
        mode: "text",
        textSelector: configPicked.captcha.sel,
        inputSelector: configPicked.input.sel,
        refreshSelector: configPicked.refresh ? configPicked.refresh.sel : undefined,
      };
    }
    chrome.storage.local.set({ [location.origin]: cfg }).then(() => {
      configStatus("已保存，配置自动生效");
      setTimeout(hideConfigBar, 1200);
    });
  }

  function showConfigBar() {
    if (configBarEl) return;
    const bar = document.createElement("div");
    bar.className = "vcbar";
    bar.style.cssText =
      "position:fixed;top:0;left:0;right:0;z-index:2147483647;" +
      "background:#1f2937;color:#fff;font:13px/1.5 'Microsoft YaHei',sans-serif;" +
      "padding:10px 16px;box-shadow:0 2px 12px rgba(0,0,0,.4);" +
      "display:flex;flex-wrap:wrap;align-items:center;gap:8px;";
    bar.innerHTML =
      '<span style="font-weight:bold">配置验证码 · <span class="vcbar-origin" ' +
      'style="font-weight:normal;color:#9ca3af"></span></span>' +
      '<button data-kind="captcha" class="vcbar-btn">验证码元素</button>' +
      '<button data-kind="input" class="vcbar-btn">输入框</button>' +
      '<button data-kind="refresh" class="vcbar-btn">更换验证码按钮(可选)</button>' +
      '<span class="vcbar-type" style="color:#9ca3af;font-size:12px"></span>' +
      '<span class="vcbar-status" style="flex:1;color:#34d399;font-size:12px;text-align:right"></span>' +
      '<button class="vcbar-save">保存配置</button>' +
      '<button class="vcbar-cancel">取消</button>';
    const style = document.createElement("style");
    style.textContent =
      ".vcbar-btn{background:#374151;color:#fff;border:1px solid #4b5563;border-radius:6px;" +
      "padding:6px 12px;cursor:pointer;font-size:12px}" +
      ".vcbar-btn:hover{background:#4b5563}" +
      ".vcbar-btn.active{outline:2px solid #f59e0b}" +
      ".vcbar-btn.picked[data-kind=\"captcha\"]{background:#e11d48;border-color:#e11d48}" +
      ".vcbar-btn.picked[data-kind=\"input\"]{background:#2f6b0f;border-color:#2f6b0f}" +
      ".vcbar-btn.picked[data-kind=\"refresh\"]{background:#b8860b;border-color:#b8860b}" +
      ".vcbar-save{background:#2f6b0f;color:#fff;border:none;border-radius:6px;" +
      "padding:6px 14px;cursor:pointer;font-size:12px;font-weight:bold}" +
      ".vcbar-save:hover{background:#3a8313}" +
      ".vcbar-cancel{background:#4b5563;color:#fff;border:none;border-radius:6px;" +
      "padding:6px 14px;cursor:pointer;font-size:12px}" +
      ".vcbar-cancel:hover{background:#6b7280}";
    bar.appendChild(style);
    bar.querySelector(".vcbar-origin").textContent = location.origin;
    bar.querySelector('[data-kind="captcha"]').addEventListener("click", () => startConfigPick("captcha"));
    bar.querySelector('[data-kind="input"]').addEventListener("click", () => startConfigPick("input"));
    bar.querySelector('[data-kind="refresh"]').addEventListener("click", () => startConfigPick("refresh"));
    bar.querySelector(".vcbar-save").addEventListener("click", saveConfigBar);
    bar.querySelector(".vcbar-cancel").addEventListener("click", hideConfigBar);
    configBarEl = bar;
    document.body.appendChild(bar);
    document.addEventListener("mousemove", onConfigMove, true);
    document.addEventListener("click", onConfigClick, true);
    document.addEventListener("keydown", onConfigKey, true);
    refreshConfigBar();
    configStatus("点击按钮后在页面上点选对应元素，保存后自动生效");
    loadExistingConfig();
  }

  // Load a previously saved config into the bar: shows the stored selectors
  // on the buttons and highlights the configured elements on the page so the
  // user sees what is already wired up (and can re-pick any of them).
  async function loadExistingConfig() {
    try {
      const data = await chrome.storage.local.get(location.origin);
      const cfg = data[location.origin];
      if (!cfg) return;
      // SPA pages may render the elements after the bar opens; retry briefly.
      const targetSel = cfg.mode === "text" ? cfg.textSelector : cfg.imgSelector;
      for (let i = 0; i < 10; i++) {
        let ready = true;
        for (const sel of [targetSel, cfg.inputSelector]) {
          if (sel) {
            try { if (!document.querySelector(sel)) ready = false; } catch { ready = false; }
          }
        }
        if (ready || i === 9) break;
        await new Promise((r) => setTimeout(r, 500));
      }
      const pickFrom = (sel, kind) => {
        if (!sel) return;
        let el = null;
        try { el = document.querySelector(sel); } catch { return; }
        if (!el) return;
        // Same element serving several roles: keep the first role's color.
        if (!highlightKindOf(el, kind)) paintHighlight(el, kind);
        configPicked[kind] = { el, sel };
      };
      if (cfg.mode === "text") {
        configMode = "text";
        pickFrom(cfg.textSelector, "captcha");
        pickFrom(cfg.inputSelector, "input");
        pickFrom(cfg.refreshSelector, "refresh");
      } else {
        configMode = "image";
        pickFrom(cfg.imgSelector, "captcha");
        pickFrom(cfg.inputSelector, "input");
        pickFrom(cfg.refreshSelector, "refresh");
      }
      refreshConfigBar();
      const n = Object.keys(configPicked).length;
      if (n) {
        configStatus(`已加载本站已有配置（${n} 项已高亮），可点击按钮重新选择`);
      }
    } catch (err) {
      console.warn("[captcha-autofill] loadExistingConfig failed:", err);
    }
  }

  function hideConfigBar() {
    clearConfigPick();
    document.removeEventListener("mousemove", onConfigMove, true);
    document.removeEventListener("click", onConfigClick, true);
    document.removeEventListener("keydown", onConfigKey, true);
    if (configBarEl) { configBarEl.remove(); configBarEl = null; }
    for (const kind of Object.keys(configPicked)) {
      clearHighlight(configPicked[kind] && configPicked[kind].el);
    }
    configPicked = {};
    configMode = null;
  }

  chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
    if (message?.type === "testSelectors") {
      handleTestSelectors(message)
        .then(sendResponse)
        .catch((err) => sendResponse({ ok: false, error: String(err) }));
      return true; // async response
    }
    if (message?.type === "startConfig") {
      showConfigBar();
      sendResponse({ ok: true });
      return;
    }
  });

  // ---------------------------------------------------------------- startup
  chrome.storage.local.get(null).then((data) => {
    const cfg = data[location.origin];
    if (cfg && cfg.mode === "text") {
      setupTextAutoFill(cfg);
    } else if (cfg && cfg.imgSelector && cfg.inputSelector) {
      setupAutoFill(cfg);
    }
  });

  // Config saved from the popup applies without a page reload.
  chrome.storage.onChanged.addListener((changes, area) => {
    if (area !== "local") return;
    const change = changes[location.origin];
    if (!change) return;
    const cfg = change.newValue;
    if (cfg && cfg.mode === "text") {
      setupTextAutoFill(cfg);
    } else if (cfg && cfg.imgSelector && cfg.inputSelector) {
      setupAutoFill(cfg);
    } else {
      config = null;
      showStatus("本站验证码配置已清除", "fail");
    }
  });
})();

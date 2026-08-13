/* Universal content script.
 *
 * - Right-click "标记为验证码并自动填充" on any page image -> locate the
 *   image, auto-find the captcha input, persist the config per-origin, then
 *   the background reloads the page.
 * - On every page load: if a config exists for this origin, auto-fill the
 *   captcha (recognize -> fill if conf >= 0.9, else click the image to
 *   refresh and retry). A MutationObserver on the image src covers manual
 *   refreshes too.
 * - Status badge auto-hides after a few seconds.
 */
"use strict";

(() => {
  const MAX_ATTEMPTS = 5;
  const RESET_WINDOW_MS = 60000;
  const CONF_THRESHOLD = 0.9;
  const BADGE_HIDE_MS = 4000;
  const EXT_BASE = chrome.runtime.getURL("");
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
  let config = null; // { imgSelector, inputSelector } for this origin
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

  function findCaptchaInput(img) {
    // 1) Attribute hints common for captcha inputs.
    const pattern = /captcha|verif|imgcode|validate|checkcode|yanzheng|code|verify/i;
    const candidates = document.querySelectorAll(
      'input[type="text"], input[type="tel"], input:not([type]), textarea'
    );
    for (const el of candidates) {
      const hay = [el.name, el.id, el.placeholder, el.className].join(" ");
      if (pattern.test(hay)) return el;
    }
    // 2) Nearest text input: same container as the image, then ancestors.
    let node = img.parentElement;
    for (let i = 0; i < 4 && node; i++) {
      const inputs = node.querySelectorAll('input[type="text"], input:not([type])');
      if (inputs.length) return inputs[inputs.length - 1];
      node = node.parentElement;
    }
    // 3) Single short text input on the page.
    if (candidates.length === 1) return candidates[0];
    return null;
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
      const resp = await chrome.runtime.sendMessage({
        type: "saveFailSample",
        reason,
        label: snap.label,
        confs: snap.confs,
        conf: snap.conf,
        imgSrc: snap.imgSrc,
        attempts: extra.attempts || 0,
        bytes: bytes.buffer,
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

  function setupFailHintMonitor() {
    // Only hints that *appear* after page load count as a rejected submit;
    // static page text (e.g. the input label "图形验证码") must not trigger.
    const monitor = new MutationObserver((mutations) => {
      if (Date.now() - failHintLastAt < FAIL_HINT_DEDUP_MS) return;
      const input = document.querySelector(config.inputSelector);
      if (!(lastFilled || (input && input.value))) return; // nothing submitted
      for (const m of mutations) {
        for (const node of m.addedNodes) {
          if (node.nodeType !== Node.TEXT_NODE) continue;
          const text = node.nodeValue || "";
          if (text.length > 500) continue;
          if (FAIL_HINT_PATTERN.test(text)) {
            failHintLastAt = Date.now();
            captureSubmitFailure();
            return;
          }
        }
      }
    });
    monitor.observe(document.body, { childList: true, subtree: true });
  }

  // ------------------------------------------------------------- auto-fill
  async function runAutoFill() {
    if (busy || !config) return;
    busy = true;
    try {
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
          lastSnap = { dataUrl, imgSrc: img.src, label: result.label, confs: result.perSlot, conf: result.conf };
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
    if (input && input.value && input.value !== lastFilled) return;
    if (busy) return;
    if (Date.now() - lastAttemptAt > RESET_WINDOW_MS) attemptCount = 0;
    if (attemptCount >= MAX_ATTEMPTS) return;
    setTimeout(runAutoFill, 250);
  }

  async function setupAutoFill(cfg) {
    config = cfg;
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

  // ------------------------------------------------------------------ mark
  async function handleMark(srcUrl) {
    const imgs = Array.from(document.querySelectorAll("img"));
    const img = imgs.find((el) => {
      const src = el.currentSrc || el.src || "";
      return src && (src === srcUrl || src.endsWith(srcUrl) || srcUrl.endsWith(src.split("?")[0]));
    }) || imgs.find((el) => {
      const src = el.currentSrc || el.src || "";
      return src && src.split("?")[0] === srcUrl.split("?")[0];
    });
    if (!img) throw new Error("未找到图片元素（可能已加载完毕被移除）");

    const input = findCaptchaInput(img);
    if (!input) throw new Error("未找到验证码输入框，请确认输入框为文本类型");

    const cfg = { imgSelector: bestSelector(img), inputSelector: bestSelector(input) };
    await chrome.storage.local.set({ [location.origin]: cfg });
    showStatus("已标记，正在刷新页面测试…", "ok");
    return cfg;
  }

  chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
    if (message?.type === "markCaptcha") {
      handleMark(message.srcUrl)
        .then((cfg) => {
          sendResponse({ ok: true, imgSelector: cfg.imgSelector, inputSelector: cfg.inputSelector });
          setTimeout(() => chrome.runtime.sendMessage({ type: "marked" }), 800);
        })
        .catch((err) => {
          showStatus("标记失败: " + err.message, "fail");
          sendResponse({ ok: false, error: String(err) });
        });
      return true; // async response
    }
  });

  // ---------------------------------------------------------------- startup
  chrome.storage.local.get(null).then((data) => {
    const cfg = data[location.origin];
    if (cfg && cfg.imgSelector && cfg.inputSelector) {
      setupAutoFill(cfg);
    }
  });
})();

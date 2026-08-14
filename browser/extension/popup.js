/* Popup: per-origin captcha configuration (text/image modes) + fail-sample
 * tools. The "测试选择器" button asks the content script to query and
 * highlight the configured selectors on the current page and report what it
 * extracted, so the config can be verified before saving.
 * ZIP export / clear stay in the background service worker.
 */
"use strict";

const $ = (id) => document.getElementById(id);
const siteEl = $("site");
const statusEl = $("status");

let tab = null;
let origin = null;

function setStatus(text) {
  statusEl.textContent = text;
}

function currentMode() {
  return document.querySelector('input[name="mode"]:checked').value;
}

function setMode(mode) {
  $("textFields").style.display = mode === "text" ? "" : "none";
  $("imageFields").style.display = mode === "image" ? "" : "none";
}

for (const radio of document.querySelectorAll('input[name="mode"]')) {
  radio.addEventListener("change", () => setMode(radio.value));
}

async function init() {
  const [t] = await chrome.tabs.query({ active: true, currentWindow: true });
  tab = t;
  let url = null;
  try {
    url = t && t.url ? new URL(t.url) : null;
  } catch { url = null; }
  if (!url || !/^https?:$/.test(url.protocol)) {
    siteEl.textContent = "当前页面无法配置（需要 http/https 站点）";
    for (const el of document.querySelectorAll("button, input")) el.disabled = true;
    return;
  }
  origin = url.origin;
  siteEl.textContent = "当前站点: " + origin;
  const data = await chrome.storage.local.get(origin);
  const cfg = data[origin];
  if (cfg) fillForm(cfg);
}

function fillForm(cfg) {
  const mode = cfg.mode === "text" ? "text" : "image";
  document.querySelector(`input[name="mode"][value="${mode}"]`).checked = true;
  setMode(mode);
  $("textSelector").value = cfg.textSelector || "";
  $("inputSelector").value = cfg.inputSelector || "";
  $("textPattern").value = cfg.textPattern || "";
  $("refreshSelector").value = cfg.refreshSelector || "";
  $("imgSelector").value = cfg.imgSelector || "";
  $("imageInputSelector").value = cfg.inputSelector || "";
}

$("visualBtn").addEventListener("click", async () => {
  if (!tab) return;
  try {
    await chrome.tabs.sendMessage(tab.id, { type: "startConfig" });
    window.close();
  } catch (err) {
    setStatus("无法启动可视化配置：请刷新页面后重试");
  }
});

$("saveBtn").addEventListener("click", async () => {
  if (!origin) return;
  const mode = currentMode();
  let cfg;
  if (mode === "text") {
    const textSelector = $("textSelector").value.trim();
    const inputSelector = $("inputSelector").value.trim();
    if (!textSelector || !inputSelector) {
      setStatus("请填写文字选择器和输入框选择器");
      return;
    }
    cfg = {
      mode: "text",
      textSelector,
      inputSelector,
      textPattern: $("textPattern").value.trim() || undefined,
      refreshSelector: $("refreshSelector").value.trim() || undefined,
    };
  } else {
    const imgSelector = $("imgSelector").value.trim();
    const inputSelector = $("imageInputSelector").value.trim();
    if (!imgSelector || !inputSelector) {
      setStatus("请填写图片选择器和输入框选择器");
      return;
    }
    cfg = { mode: "image", imgSelector, inputSelector };
  }
  await chrome.storage.local.set({ [origin]: cfg });
  setStatus("已保存，页面自动生效（无需刷新）");
});

$("clearCfgBtn").addEventListener("click", async () => {
  if (!origin) return;
  if (!confirm(`确定清除 ${origin} 的验证码配置？`)) return;
  await chrome.storage.local.remove(origin);
  for (const id of ["textSelector", "inputSelector", "textPattern", "refreshSelector", "imgSelector", "imageInputSelector"]) {
    $(id).value = "";
  }
  setStatus("已清除本站配置");
});

$("testBtn").addEventListener("click", async () => {
  if (!tab || !origin) return;
  const mode = currentMode();
  const payload = { type: "testSelectors" };
  if (mode === "text") {
    payload.textSelector = $("textSelector").value.trim();
    payload.inputSelector = $("inputSelector").value.trim();
    payload.refreshSelector = $("refreshSelector").value.trim() || "";
    payload.textPattern = $("textPattern").value.trim() || "";
  } else {
    payload.textSelector = "";
    payload.inputSelector = $("imageInputSelector").value.trim();
    payload.refreshSelector = "";
  }
  if (!payload.textSelector && !payload.inputSelector) {
    setStatus("请先填写至少一个选择器");
    return;
  }
  setStatus("正在测试…（命中元素将高亮 4 秒）");
  try {
    const resp = await chrome.tabs.sendMessage(tab.id, payload);
    if (!resp) { setStatus("页面无响应，请刷新页面后重试"); return; }
    const lines = [];
    const fmt = (name, r) => {
      if (!r) return;
      if (r.skipped) return;
      if (r.found) {
        let s = `- ${name}: 命中 ${r.count} 个`;
        if (name === "文字元素" && r.sample !== undefined) {
          s += `\n  文本: "${r.sample}"`;
          s += `\n  提取: "${r.extracted}"`;
        }
        lines.push(s);
      } else {
        lines.push(`- ${name}: 未找到` + (r.error ? `（${r.error}）` : ""));
      }
    };
    fmt("文字元素", resp.text);
    fmt("输入框", resp.input);
    fmt("更换验证码按钮", resp.refresh);
    setStatus(lines.join("\n") || "未填写任何选择器");
  } catch (err) {
    setStatus("测试失败: 页面未注入扩展脚本，请刷新页面后重试");
  }
});

// ------------------------------------------------------- fail-sample tools
const exportBtn = $("exportBtn");
const clearFailsBtn = $("clearFailsBtn");
const countEl = $("count");

async function refreshCount() {
  try {
    const resp = await chrome.runtime.sendMessage({ type: "countFails" });
    countEl.textContent = `已收集 ${resp.count} 个失败样本（最多 2000）`;
    exportBtn.disabled = resp.count === 0;
  } catch (err) {
    countEl.textContent = "获取数量失败";
    console.error("[captcha-autofill] count failed:", err);
  }
}

exportBtn.addEventListener("click", async () => {
  exportBtn.disabled = true;
  setStatus("正在导出…");
  try {
    const resp = await chrome.runtime.sendMessage({ type: "exportFails" });
    if (resp.ok) {
      // Decode base64 and download from the popup document (SW blob URLs die
      // with the worker; response messages are JSON-serialized).
      const bin = atob(resp.b64);
      const bytes = new Uint8Array(bin.length);
      for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
      const blob = new Blob([bytes], { type: "application/zip" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = resp.filename;
      a.click();
      setTimeout(() => URL.revokeObjectURL(url), 60000);
      setStatus(`已下载 ${resp.filename}（${resp.count} 个样本）`);
    } else if (resp.empty) {
      setStatus("没有可导出的样本");
    }
  } catch (err) {
    setStatus("导出失败: " + err);
  }
  exportBtn.disabled = false;
});

clearFailsBtn.addEventListener("click", async () => {
  if (!confirm("确定清空所有已收集的失败样本？此操作不可恢复。")) return;
  try {
    const resp = await chrome.runtime.sendMessage({ type: "clearFails" });
    if (resp.ok) {
      setStatus("已清空");
      await refreshCount();
    }
  } catch (err) {
    setStatus("清空失败: " + err);
  }
});

init();
refreshCount();

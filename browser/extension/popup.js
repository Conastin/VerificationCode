/* Popup: show collected fail-sample count, trigger ZIP export / clear.
 * The heavy lifting (IndexedDB read, ZIP assembly, download) runs in the
 * background service worker via runtime messages.
 */
"use strict";

const countEl = document.getElementById("count");
const exportBtn = document.getElementById("exportBtn");
const clearBtn = document.getElementById("clearBtn");
const statusEl = document.getElementById("status");

function setStatus(text) {
  statusEl.textContent = text;
}

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
      console.log("[captcha-autofill] export resp:", resp.filename,
        "count=", resp.count, "dataBytes=", resp.data?.byteLength);
      // Download from the popup document: SW blob URLs die with the worker.
      const blob = new Blob([resp.data], { type: "application/zip" });
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

clearBtn.addEventListener("click", async () => {
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

refreshCount();

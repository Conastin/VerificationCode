/* Background service worker: right-click menu on any image to mark it as a
 * CAPTCHA. After the content script stores the config and asks for a reload,
 * the page refreshes and auto-fill kicks in from the persisted config.
 */
"use strict";

const CONTENT_FILES = [
  "vendor/onnxruntime-web/ort.wasm.min.js",
  "shared/recognizer.js",
  "content.js",
];

chrome.runtime.onInstalled.addListener(() => {
  chrome.contextMenus.removeAll(() => {
    chrome.contextMenus.create({
      id: "mark-captcha",
      title: "标记为验证码并自动填充",
      contexts: ["image"],
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

chrome.contextMenus.onClicked.addListener(async (info, tab) => {
  if (info.menuItemId !== "mark-captcha" || !tab?.id) return;
  try {
    await chrome.tabs.sendMessage(tab.id, { type: "markCaptcha", srcUrl: info.srcUrl });
  } catch {
    // content script not present (page opened before install) -> inject + retry
    await ensureContentScript(tab.id);
    try {
      await chrome.tabs.sendMessage(tab.id, { type: "markCaptcha", srcUrl: info.srcUrl });
    } catch (err) {
      console.error("[captcha-autofill] mark failed:", err);
    }
  }
});

chrome.runtime.onMessage.addListener((message, sender) => {
  if (message?.type === "marked") {
    // Content script stored the config; reload so auto-fill runs fresh.
    chrome.tabs.reload(sender.tab.id);
  }
});

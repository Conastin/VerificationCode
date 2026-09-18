# -*- coding: utf-8 -*-
"""生成自包含 HTML 标注工具：内嵌 base64 图片，回车跳下一张，导出 labels.txt。"""
import base64
import json
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "data" / "portal_label"
OUT = SRC / "label_tool.html"

files = sorted(SRC.glob("q_*.jpg"))
images = {f.name: base64.b64encode(f.read_bytes()).decode("ascii") for f in files}

HTML = """<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>portal 验证码标注</title>
<style>
  body { font-family: Consolas, "Microsoft YaHei", monospace; background: #222; color: #eee;
         display: flex; flex-direction: column; align-items: center; margin: 0; padding: 24px; }
  h3 { margin: 0 0 8px; font-weight: normal; color: #9ecbff; }
  #progress { color: #888; margin-bottom: 12px; }
  #img { image-rendering: auto; border: 2px solid #555; background: #fff; }
  #input { font-size: 28px; width: 220px; text-align: center; letter-spacing: 8px;
           padding: 6px; margin-top: 14px; text-transform: none; }
  #hint { color: #888; font-size: 13px; margin-top: 10px; line-height: 1.7; }
  kbd { background: #333; border: 1px solid #555; border-radius: 3px; padding: 0 5px; }
  button { margin: 16px 6px 0; padding: 8px 18px; font-size: 15px; cursor: pointer; }
  #done { position: fixed; top: 12px; right: 16px; color: #6f6; display: none; }
</style>
</head>
<body>
<h3>portal 验证码标注工具</h3>
<div id="progress"></div>
<img id="img" width="480" height="204">
<input id="input" maxlength="6" autocomplete="off" autofocus spellcheck="false">
<div id="hint">
  <kbd>输入 4 个字符</kbd> → <kbd>Enter</kbd> 下一张 &nbsp;|&nbsp; <kbd>←</kbd>/<kbd>→</kbd> 前后翻 &nbsp;|&nbsp;
  <kbd>空格+Enter</kbd> 或留空 Enter = 跳过（看不清）<br>
  大小写：看到小写就输小写，不确定就输大写。标完或想休息时点「导出」，可断点续标（进度自动存在本机）。
</div>
<button id="export">导出 labels.txt</button>
<button id="copy">复制到剪贴板</button>
<div id="done">已导出</div>
<script>
const IMAGES = __IMAGES_JSON__;
const names = Object.keys(IMAGES);
const STORE_KEY = "portal_label_progress_v1";
let labels = {};
try { labels = JSON.parse(localStorage.getItem(STORE_KEY) || "{}"); } catch (e) {}
let idx = parseInt(localStorage.getItem(STORE_KEY + "_idx") || "0", 10) || 0;
if (idx >= names.length) idx = 0;

const img = document.getElementById("img");
const input = document.getElementById("input");
const progress = document.getElementById("progress");

function save() {
  localStorage.setItem(STORE_KEY, JSON.stringify(labels));
  localStorage.setItem(STORE_KEY + "_idx", String(idx));
}
function render() {
  const name = names[idx];
  img.src = "data:image/jpeg;base64," + IMAGES[name];
  input.value = labels[name] || "";
  const doneCount = Object.keys(labels).filter(k => labels[k]).length;
  progress.textContent = `第 ${idx + 1} / ${names.length} 张   已标 ${doneCount} 张`;
  input.focus(); input.select();
}
function next() {
  const name = names[idx];
  const v = input.value.trim();
  if (v) labels[name] = v; else delete labels[name];
  save();
  idx = (idx + 1) % names.length;
  render();
}
input.addEventListener("keydown", (e) => {
  if (e.key === "Enter") { e.preventDefault(); next(); }
  if (e.key === "ArrowRight") { e.preventDefault(); idx = Math.min(idx + 1, names.length - 1); render(); }
  if (e.key === "ArrowLeft") { e.preventDefault(); idx = Math.max(idx - 1, 0); render(); }
});
function labelText() {
  return names.filter(n => labels[n]).map(n => `${n} ${labels[n]}`).join("\\n");
}
document.getElementById("export").onclick = () => {
  const blob = new Blob([labelText() + "\\n"], { type: "text/plain" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "labels.txt";
  a.click();
  document.getElementById("done").style.display = "block";
};
document.getElementById("copy").onclick = async () => {
  try { await navigator.clipboard.writeText(labelText()); 
        document.getElementById("done").textContent = "已复制"; 
        document.getElementById("done").style.display = "block"; } catch (e) {}
};
render();
</script>
</body>
</html>
"""

html = HTML.replace("__IMAGES_JSON__", json.dumps(images))
OUT.write_text(html, encoding="utf-8")
print(f"{OUT} ({len(files)} images, {OUT.stat().st_size / 1024:.0f} KB)")

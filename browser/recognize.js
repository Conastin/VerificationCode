/* Demo page logic only; recognition core lives in shared/recognizer.js. */
"use strict";

let session = null;
let modelStatus = "loading";
let sampleMeta = null;

// ---------------------------------------------------------------- model setup
async function initModel() {
  const status = document.getElementById("status");
  try {
    await CaptchaRecognizer.init(
      "model.onnx",
      location.origin + "/browser/vendor/onnxruntime-web/"
    );
    modelStatus = "ok";
    status.className = "status ok";
    status.textContent =
      "模型加载成功 · onnxruntime-web " + ort.env.version +
      (window.crossOriginIsolated ? "（多线程）" : "（单线程）");
    document.getElementById("runBtn").disabled = false;
    document.getElementById("pickBtn").disabled = false;
  } catch (err) {
    modelStatus = "error";
    status.className = "status err";
    status.textContent = "模型加载失败: " + err;
  }
}

// --------------------------------------------------------------------- views
function loadImage(src) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error("图片加载失败: " + src));
    img.src = src;
  });
}

async function runAll() {
  const tbody = document.querySelector("#resultTable tbody");
  tbody.innerHTML = "";
  let consistent = 0, correct = 0;
  for (const rec of sampleMeta.samples) {
    const img = await loadImage("samples/" + rec.file);
    const res = await CaptchaRecognizer.recognize(img);
    const tr = document.createElement("tr");
    const matchPy = res.label === rec.python_pred ? "✓" : "✗";
    const matchTrue = res.label === rec.true ? "✓" : "✗";
    if (matchPy === "✓") consistent++;
    if (matchTrue === "✓") correct++;
    tr.innerHTML =
      "<td>" + rec.file + "</td>" +
      "<td>" + rec.true + "</td>" +
      "<td class='js'>" + res.label + "</td>" +
      "<td>" + rec.python_pred + "</td>" +
      "<td>" + res.conf.toFixed(3) + "</td>" +
      "<td>JS/真实: " + matchTrue + " · JS/Python: " + matchPy + "</td>";
    tbody.appendChild(tr);
  }
  const summary = document.createElement("tr");
  summary.innerHTML = "<td colspan='6'><b>汇总: 识别正确 " + correct + "/" +
    sampleMeta.samples.length + " · 与 Python 一致 " + consistent + "/" +
    sampleMeta.samples.length + "</b></td>";
  tbody.appendChild(summary);
}

async function runCustom(file) {
  const url = URL.createObjectURL(file);
  const img = await loadImage(url);
  const res = await CaptchaRecognizer.recognize(img);
  URL.revokeObjectURL(url);
  const box = document.getElementById("customResult");
  box.innerHTML =
    "预测: <b>" + res.label + "</b> &nbsp; 最低置信度: " + res.conf.toFixed(3) +
    " &nbsp; 逐位: " + res.perSlot.map((v) => v.toFixed(2)).join(" / ");
}

// ---------------------------------------------------------------------- init
(async () => {
  const metaResp = await fetch("samples/samples.json");
  sampleMeta = await metaResp.json();
  const grid = document.getElementById("grid");
  for (const rec of sampleMeta.samples) {
    const card = document.createElement("div");
    card.className = "card";
    card.innerHTML =
      "<img src='samples/" + rec.file + "' alt='" + rec.file + "'>" +
      "<div class='label'>" + rec.file + " · 真实: " + rec.true + "</div>";
    grid.appendChild(card);
  }
  document.getElementById("runBtn").onclick = runAll;
  document.getElementById("pickBtn").onclick = () => document.getElementById("fileInput").click();
  document.getElementById("fileInput").onchange = (e) => {
    if (e.target.files.length) runCustom(e.target.files[0]);
  };
  await initModel();
})();

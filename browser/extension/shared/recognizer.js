/* Shared CAPTCHA recognition core (no DOM dependencies).
 * Used by both the demo page and the browser extension.
 * Preprocessing mirrors the Python pipeline: RGB float32 /255, CHW, 1x3x20x60.
 * Output logits (1,4,31) -> softmax -> argmax per slot + min confidence.
 */
"use strict";

const CaptchaRecognizer = (() => {
  const CHARSET = "23456789ABCDEFGHJKMNPQRSTUVWXYZ";
  let session = null;

  async function init(modelUrl, wasmPaths) {
    ort.env.wasm.wasmPaths = wasmPaths;
    // Content scripts / classic pages are not cross-origin isolated, so
    // threads (SharedArrayBuffer) are unavailable -> single thread is fine
    // for a 60x20 image (few ms).
    ort.env.wasm.numThreads = 1;
    session = await ort.InferenceSession.create(modelUrl, {
      executionProviders: ["wasm"],
      graphOptimizationLevel: "all",
    });
    return true;
  }

  function imageToTensor(image) {
    const canvas = document.createElement("canvas");
    canvas.width = 60;
    canvas.height = 20;
    const ctx = canvas.getContext("2d", { willReadFrequently: true });
    ctx.drawImage(image, 0, 0, 60, 20);
    const rgba = ctx.getImageData(0, 0, 60, 20).data;
    const data = new Float32Array(3 * 20 * 60);
    for (let p = 0; p < 60 * 20; p++) {
      data[p] = rgba[p * 4] / 255;
      data[1200 + p] = rgba[p * 4 + 1] / 255;
      data[2400 + p] = rgba[p * 4 + 2] / 255;
    }
    return new ort.Tensor("float32", data, [1, 3, 20, 60]);
  }

  async function recognize(image) {
    if (!session) throw new Error("recognizer not initialized");
    const results = await session.run({ image: imageToTensor(image) });
    const logits = results.logits.data; // (1, 4, 31)
    const label = [];
    const confs = [];
    for (let slot = 0; slot < 4; slot++) {
      const base = slot * 31;
      let max = -Infinity;
      for (let c = 0; c < 31; c++) max = Math.max(max, logits[base + c]);
      let sum = 0;
      for (let c = 0; c < 31; c++) sum += Math.exp(logits[base + c] - max);
      let best = 0, bestP = 0;
      for (let c = 0; c < 31; c++) {
        const p = Math.exp(logits[base + c] - max) / sum;
        if (p > bestP) { bestP = p; best = c; }
      }
      label.push(CHARSET[best]);
      confs.push(bestP);
    }
    return { label: label.join(""), conf: Math.min(...confs), perSlot: confs };
  }

  return { init, recognize, CHARSET };
})();

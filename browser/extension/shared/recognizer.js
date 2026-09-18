/* Shared CAPTCHA recognition core (no DOM dependencies).
 * Used by both the demo page and the browser extension.
 *
 * Universal CRNN (v11) + CTC: any captcha size, height normalized to 48
 * keeping aspect ratio; logits (1, T, 37) -> per-frame softmax -> greedy
 * CTC decode (collapse repeats, drop blank=36). Confidence is the mean
 * per-character probability of the decoded output.
 */
"use strict";

const CaptchaRecognizer = (() => {
  const CHARSET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ";
  const BLANK = 36;
  const IMG_H = 48;
  let session = null;

  async function init(modelUrl, wasmPaths) {
    ort.env.wasm.wasmPaths = wasmPaths;
    // Content scripts / classic pages are not cross-origin isolated, so
    // threads (SharedArrayBuffer) are unavailable -> single thread is fine
    // (a 48x~120 CRNN forward takes a few ms).
    ort.env.wasm.numThreads = 1;
    session = await ort.InferenceSession.create(modelUrl, {
      executionProviders: ["wasm"],
      graphOptimizationLevel: "all",
    });
    return true;
  }

  function imageToTensor(image) {
    const natural = image.naturalWidth || image.width || 80;
    const naturalH = image.naturalHeight || image.height || 34;
    const w = Math.max(8, Math.round(natural * IMG_H / naturalH));
    const canvas = document.createElement("canvas");
    canvas.width = w;
    canvas.height = IMG_H;
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
    if (!session) throw new Error("recognizer not initialized");
    const results = await session.run({ image: imageToTensor(image) });
    const logits = results.logits.data; // (1, T, 37)
    const T = results.logits.dims[1];

    // per-frame softmax argmax + greedy CTC decode
    const label = [];
    const confs = [];
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
      const prob = 1 / sum; // probability of the argmax class
      if (best !== prev && best !== BLANK) {
        label.push(CHARSET[best]);
        confs.push(prob);
      }
      prev = best;
    }
    const conf = confs.length ? confs.reduce((a, b) => a + b, 0) / confs.length : 0;
    return { label: label.join(""), conf, perSlot: confs };
  }

  return { init, recognize, CHARSET };
})();

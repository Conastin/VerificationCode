# -*- coding: utf-8 -*-
"""定宽训练吞吐测试: collate 固定 pad 到 192."""
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from verification_code.synth_universal import SynthCaptchaGenerator
from verification_code.train_universal import CrnnCaptcha, collate, run_batch
from verification_code.finetune_universal import load_labeled_folder, MixedStream

FIXED_W = 224


def fixed_collate(batch):
    from verification_code.train_universal import encode_text
    images, texts = zip(*batch)
    imgs = np.zeros((len(images), 48, FIXED_W, 3), dtype=np.float32)
    for i, img in enumerate(images):
        if img.shape[1] > FIXED_W:  # rare over-wide sample: squeeze to fit
            img = np.asarray(
                __import__("PIL.Image", fromlist=["Image"]).fromarray(
                    (img * 255).astype(np.uint8)).resize((FIXED_W, 48)),
                dtype=np.float32) / 255.0
        imgs[i, :, : img.shape[1], :] = img
    tensor = torch.from_numpy(imgs).permute(0, 3, 1, 2)
    targets = torch.tensor([c for t in texts for c in encode_text(t)], dtype=torch.long)
    target_lens = torch.tensor([len(encode_text(t)) for t in texts], dtype=torch.long)
    input_lens = torch.full((len(images),), FIXED_W // 4, dtype=torch.long)
    return tensor, targets, input_lens, target_lens, list(texts)


def main() -> None:
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    pool = load_labeled_folder(Path("data/train31"))
    synth = SynthCaptchaGenerator(seed=9)
    stream = MixedStream(pool, synth, 400 * 128, 0.65)
    loader = DataLoader(stream, batch_size=128, shuffle=False, num_workers=12,
                        collate_fn=fixed_collate, persistent_workers=True,
                        drop_last=True, pin_memory=True)
    model = CrnnCaptcha().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    scaler = torch.amp.GradScaler("cuda")
    n = 0
    t0 = time.time()
    for batch in loader:
        run_batch(model, batch, device, opt, scaler)
        n += 1
        if n == 30:
            t_warm = time.time() - t0
    dt = (time.time() - t0 - t_warm) / max(n - 30, 1)
    print(f"fixed-w{FIXED_W}: {n} steps, warm {t_warm:.0f}s, "
          f"{dt*1000:.0f} ms/step, {1/dt:.1f} steps/s, {128/dt:.0f} imgs/s")


if __name__ == "__main__":
    main()

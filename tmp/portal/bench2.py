# -*- coding: utf-8 -*-
"""分离瓶颈: A 纯 DataLoader 循环; B 纯 GPU 训练步(固定张量)."""
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from verification_code.synth_universal import SynthCaptchaGenerator
from verification_code.train_universal import CrnnCaptcha, collate, run_batch
from verification_code.finetune_universal import load_labeled_folder, MixedStream


def bench_loader() -> None:
    pool = load_labeled_folder(Path("data/train31"))
    synth = SynthCaptchaGenerator(seed=9)
    stream = MixedStream(pool, synth, 300 * 128, 0.65)
    loader = DataLoader(stream, batch_size=128, shuffle=False, num_workers=12,
                        collate_fn=collate, persistent_workers=True, drop_last=True)
    it = iter(loader)
    for _ in range(20):
        next(it)  # warm
    t0 = time.time()
    n = 0
    for _ in range(200):
        next(it)
        n += 1
    dt = time.time() - t0
    print(f"A) loader only: {dt/n*1000:.0f} ms/batch, {128*n/dt:.0f} imgs/s")


def bench_gpu() -> None:
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    model = CrnnCaptcha().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    scaler = torch.amp.GradScaler("cuda")
    # 固定宽度 160 的假 batch
    for width in (112, 160):
        images = torch.rand(128, 3, 48, width, device=device)
        batch = (images, torch.randint(0, 36, (128 * 4,), device=device),
                 torch.full((128,), width // 4, dtype=torch.long, device=device),
                 torch.full((128,), 4, dtype=torch.long, device=device),
                 ["AAAA"] * 128)
        for _ in range(10):
            run_batch(model, batch, device, opt, scaler)
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(50):
            run_batch(model, batch, device, opt, scaler)
        torch.cuda.synchronize()
        dt = (time.time() - t0) / 50
        print(f"B) gpu only w={width}: {dt*1000:.0f} ms/step ({128/dt:.0f} imgs/s)")


if __name__ == "__main__":
    bench_gpu()
    bench_loader()

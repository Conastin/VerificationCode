# -*- coding: utf-8 -*-
"""吞吐基准: batch128 workers12 MixedStream 400 步."""
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from verification_code.synth_universal import SynthCaptchaGenerator
from verification_code.train_universal import CrnnCaptcha, collate, run_batch
from verification_code.finetune_universal import load_labeled_folder, MixedStream


def main() -> None:
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    pool = load_labeled_folder(Path("data/train31"))
    synth = SynthCaptchaGenerator(seed=9)
    stream = MixedStream(pool, synth, 400 * 128, 0.65)
    loader = DataLoader(stream, batch_size=128, shuffle=False, num_workers=12,
                        collate_fn=collate, persistent_workers=True, drop_last=True,
                        pin_memory=True)
    model = CrnnCaptcha().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    scaler = torch.amp.GradScaler("cuda")
    n = 0
    t0 = time.time()
    for batch in loader:
        loss, ok, cnt = run_batch(model, batch, device, opt, scaler)
        n += 1
        if n == 20:
            t_warm = time.time() - t0
    dt = (time.time() - t0 - t_warm) / max(n - 20, 1)
    print(f"{n} steps, warm {t_warm:.0f}s, steady: {dt*1000:.0f} ms/step, "
          f"{1/dt:.1f} steps/s, {128/dt:.0f} imgs/s")


if __name__ == "__main__":
    main()

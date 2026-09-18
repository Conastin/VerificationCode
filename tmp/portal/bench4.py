# -*- coding: utf-8 -*-
"""分段计时: 数据等待 vs 前向/反向 vs 解码同步."""
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from verification_code.synth_universal import SynthCaptchaGenerator
from verification_code.train_universal import (CrnnCaptcha, collate, ctc_greedy,
                                               run_batch)
from verification_code.finetune_universal import load_labeled_folder, MixedStream
import torch.nn as nn


def main() -> None:
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    pool = load_labeled_folder(Path("data/train31"))
    synth = SynthCaptchaGenerator(seed=9)
    stream = MixedStream(pool, synth, 300 * 128, 0.65)
    loader = DataLoader(stream, batch_size=128, shuffle=False, num_workers=12,
                        collate_fn=collate, persistent_workers=True,
                        drop_last=True, pin_memory=True)
    model = CrnnCaptcha().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    scaler = torch.amp.GradScaler("cuda")

    t_data = t_h2d = t_fwd = t_bwd = t_dec = 0.0
    n = 0
    it = iter(loader)
    for _ in range(20):  # warm
        batch = next(it)
        run_batch(model, batch, device, opt, scaler)

    torch.cuda.synchronize()
    t_end = time.time() + 60  # 采样 60 秒
    while time.time() < t_end:
        t0 = time.time()
        batch = next(it)
        t1 = time.time()
        images, targets, input_lens, target_lens, texts = batch
        images = images.to(device, non_blocking=True)
        targets = targets.to(device)
        input_lens = input_lens.to(device)
        target_lens = target_lens.to(device)
        with torch.autocast("cuda", dtype=torch.float16):
            logits = model(images)
        t2 = time.time()
        loss = nn.functional.ctc_loss(
            logits.log_softmax(-1).permute(1, 0, 2).float(), targets,
            input_lens, target_lens, blank=36, zero_infinity=True)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        torch.cuda.synchronize()
        t3 = time.time()
        preds = ctc_greedy(logits.float())
        _ = sum(p == t.upper() for (p, _), t in zip(preds, texts))
        t4 = time.time()
        t_data += t1 - t0
        t_h2d += t2 - t1 - 0  # fwd 含 h2d 等待
        t_fwd += 0
        t_bwd += t3 - t2
        t_dec += t4 - t3
        n += 1
    total = t_data + t_h2d + t_bwd + t_dec
    print(f"steps={n}  平均 {total/n*1000:.0f} ms/step")
    print(f"  取数据:   {t_data/n*1000:6.1f} ms")
    print(f"  前向+h2d: {t_h2d/n*1000:6.1f} ms")
    print(f"  损失+反向+同步: {t_bwd/n*1000:6.1f} ms")
    print(f"  解码+对账: {t_dec/n*1000:6.1f} ms")


if __name__ == "__main__":
    main()

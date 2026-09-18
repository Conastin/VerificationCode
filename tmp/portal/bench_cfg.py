# -*- coding: utf-8 -*-
"""单配置吞吐+内存基准: python bench_cfg.py <batch> <workers> <steps>"""
import sys
import time


def main() -> None:
    import psutil
    import torch
    from torch.utils.data import DataLoader

    sys.path.insert(0, ".")
    from verification_code.synth_universal import SynthCaptchaGenerator
    from verification_code.train_universal import (CrnnCaptcha, SynthStream,
                                                   collate, run_batch)

    batch, workers, steps = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    gen = SynthCaptchaGenerator(seed=9)
    stream = SynthStream(gen, steps * batch)
    loader = DataLoader(stream, batch_size=batch, shuffle=False,
                        num_workers=workers, collate_fn=collate,
                        persistent_workers=True, drop_last=True, pin_memory=True)
    model = CrnnCaptcha().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    scaler = torch.amp.GradScaler("cuda")
    proc = psutil.Process()

    def total_rss() -> int:
        ram = proc.memory_info().rss
        for ch in proc.children(recursive=True):
            try:
                ram += ch.memory_info().rss
            except psutil.Error:
                pass
        return ram

    n = 0
    t0 = time.time()
    for b in loader:
        if n == 30:
            t_warm = time.time() - t0
        run_batch(model, b, device, opt, scaler, with_metrics=False)
        n += 1
    torch.cuda.synchronize()
    dt = (time.time() - t0 - t_warm) / max(n - 30, 1)
    print(f"batch={batch} workers={workers}: {dt*1000:.0f} ms/step, "
          f"{batch/dt:.0f} imgs/s, RAM~{total_rss()/2**30:.1f}GB, "
          f"VRAM={torch.cuda.max_memory_allocated()/2**30:.2f}GB", flush=True)


if __name__ == "__main__":
    main()

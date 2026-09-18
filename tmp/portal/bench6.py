# -*- coding: utf-8 -*-
"""配置对比: (batch, workers) 组合的吞吐与内存占用."""
import importlib
import subprocess
import sys
import time
from pathlib import Path

CONFIGS = [(128, 6), (192, 8), (192, 6)]

BENCH = '''
import time, psutil, torch
from torch.utils.data import DataLoader
from verification_code.synth_universal import SynthCaptchaGenerator
from verification_code.train_universal import CrnnCaptcha, collate, run_batch, SynthStream
import sys
BATCH, WORKERS, STEPS = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
device = torch.device("cuda")
torch.backends.cudnn.benchmark = True
gen = SynthCaptchaGenerator(seed=9)
stream = SynthStream(gen, STEPS * BATCH)
loader = DataLoader(stream, batch_size=BATCH, shuffle=False, num_workers=WORKERS,
                    collate_fn=collate, persistent_workers=True, drop_last=True,
                    pin_memory=True)
model = CrnnCaptcha().to(device)
opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
scaler = torch.amp.GradScaler("cuda")
proc = psutil.Process()
n = 0
t0 = time.time()
worker_ram = 0
for batch in loader:
    if n == 30:
        t_warm = time.time() - t0
        ram0 = proc.memory_info().rss
        for ch in proc.children(recursive=True):
            try: ram0 += ch.memory_info().rss
            except Exception: pass
    run_batch(model, batch, device, opt, scaler, with_metrics=False)
    n += 1
torch.cuda.synchronize()
dt = (time.time() - t0 - t_warm) / max(n - 30, 1)
ram1 = proc.memory_info().rss
for ch in proc.children(recursive=True):
    try: ram1 += ch.memory_info().rss
    except Exception: pass
print(f"batch={BATCH} workers={WORKERS}: {dt*1000:.0f} ms/step, {BATCH/dt:.0f} imgs/s, "
      f"RAM~{(ram1)/2**30:.1f}GB, VRAM={torch.cuda.max_memory_allocated()/2**30:.2f}GB")
'''

Path("tmp/portal/bench_cfg.py").write_text(BENCH, encoding="utf-8")


def main() -> None:
    for batch, workers in CONFIGS:
        r = subprocess.run([sys.executable, "tmp/portal/bench_cfg.py", str(batch),
                            str(workers), "240"], capture_output=True, text=True,
                           cwd=".", timeout=600)
        out = (r.stdout + r.stderr).strip().splitlines()
        print(out[-1] if out else f"batch={batch} workers={workers}: FAILED")
        if "FAILED" in (out[-1] if out else "") or r.returncode != 0:
            print("\n".join(out[-10:]))


if __name__ == "__main__":
    main()

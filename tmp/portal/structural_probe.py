# -*- coding: utf-8 -*-
"""结构性探针:
A. 饱和度可分性: 字符(彩色) vs 干扰线(灰黑) 在 HSV 饱和度轴上是否分离
B. 字符落位: 用饱和度掩码做垂直投影, 测字符中心 vs 理想 20px 等分格的抖动
样本: 金标准(易) + 已知难例(hard 各批)
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, ".")


def analyze(img: Image.Image):
    rgb = np.asarray(img.convert("RGB"), dtype=np.float32)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    v = rgb.max(axis=2)
    # 饱和度: (max-min)/max, 背景(白) S~0
    s = np.where(v > 0, (v - rgb.min(axis=2)) / np.maximum(v, 1), 0)
    bg = v >= 235                       # 近白背景
    dark = (v < 140) & (s < 0.25)       # 灰黑笔画候选(干扰线/暗字)
    chroma = s >= 0.25                  # 彩色像素(字符主体)
    # B: 垂直投影上的彩色像素 -> 字符列位置
    col = (chroma & ~bg).sum(axis=0)
    th = col.max() * 0.2 if col.max() > 0 else 0
    on = col > th
    # 找连续段
    segs = []
    start = None
    for x, flag in enumerate(on):
        if flag and start is None:
            start = x
        elif not flag and start is not None:
            segs.append((start, x)); start = None
    if start is not None:
        segs.append((start, len(on)))
    segs = [(a, b) for a, b in segs if b - a >= 3]
    centers = [(a + b) / 2 for a, b in segs]
    ideal = [10 + 20 * i for i in range(4)]
    return {
        "bg_pct": bg.mean(),
        "dark_px": int((dark & ~bg).sum()),
        "chroma_px": int((chroma & ~bg).sum()),
        "n_segs": len(segs),
        "centers": [round(c, 1) for c in centers[:6]],
        "ideal": ideal,
        "dev": [round(min(abs(c - i) for i in ideal), 1) for c in centers[:4]]
               if len(centers) >= 4 else None,
    }


def main() -> None:
    easy = sorted(Path("data/portal_dataset/server_confirmed/batch_20260918_143724").glob("*.jpg"))[:200]
    hard = []
    for d in ("hard", "hard2", "hard34", "hard56"):
        hard += sorted(Path("data/portal_label", d).glob("*.jpg"))
    hard = hard[:300]

    for name, files in [("易例(金标准)", easy), ("难例(模型曾错)", hard)]:
        stats = [analyze(Image.open(f)) for f in files]
        n = len(stats)
        dark_total = sum(x["dark_px"] for x in stats)
        chroma_total = sum(x["chroma_px"] for x in stats)
        devs = [d for x in stats if x["dev"] for d in x["dev"]]
        seg4 = sum(1 for x in stats if x["n_segs"] == 4)
        print(f"== {name} n={n} ==")
        print(f"  灰黑像素(干扰线主体): {dark_total/n:.0f}px/图, 彩色像素(字符主体): {chroma_total/n:.0f}px/图")
        print(f"  彩色:灰黑 比 = {chroma_total/max(dark_total,1):.1f}:1")
        print(f"  投影恰好分出 4 段: {seg4/n:.0%}")
        if devs:
            print(f"  字符中心偏离理想格: mean={np.mean(devs):.1f}px, p90={np.percentile(devs,90):.1f}px (格宽20px)")
        # 饱和度直方图对比: 干扰线 vs 字符
        sample = stats[:5]


if __name__ == "__main__":
    main()

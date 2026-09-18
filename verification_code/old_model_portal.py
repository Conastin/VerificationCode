"""Run the OLD slot model on portal captures (what the extension does today:
resize 80x34 -> 60x20) and dump a comparison sheet with predictions.

Usage:
    python -m verification_code.old_model_portal --data data/portal_raw
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from .model import build_model_from_checkpoint, decode_logits


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Old slot model on portal images.")
    parser.add_argument("--data", type=Path, default=Path("data/portal_raw"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/best.pt"))
    parser.add_argument("--out-csv", type=Path, default=Path("reports/portal_old_model.csv"))
    parser.add_argument("--sheet", type=Path, default=Path("reports/portal_old_sheet.png"))
    return parser


def main() -> int:
    args = build_parser().parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    charset = ckpt["charset"]
    model = build_model_from_checkpoint(ckpt, charset).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    files = sorted(args.data.glob("*.jpg"))
    rows = []
    with torch.no_grad():
        for path in files:
            with Image.open(path) as im:
                arr = np.asarray(im.convert("RGB").resize((60, 20), Image.BILINEAR),
                                 dtype=np.float32) / 255.0
            x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
            label, conf = decode_logits(model(x), charset)
            rows.append((path.stem, label[0], conf[0].min().item()))

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["file", "pred_old_model", "min_confidence"])
        writer.writerows(rows)

    # sheet: original image + old-model prediction, aligned with analyze_portal layout
    step = max(1, len(rows) // 30)
    picked = rows[::step][:30]
    scale = 3
    cw, ch = 80 * scale, 34 * scale
    pad, label_h = 10, 34
    cols = 5
    rows_n = (len(picked) + cols - 1) // cols
    W = cols * (cw + 2 * pad) + pad
    H = rows_n * (ch + label_h + 2 * pad) + pad
    sheet = Image.new("RGB", (W, H), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.truetype("C:/Windows/Fonts/consolab.ttf", 22)
    for i, (name, pred, conf) in enumerate(picked):
        r, c = divmod(i, cols)
        x = pad + c * (cw + 2 * pad)
        y = pad + r * (ch + label_h + 2 * pad)
        with Image.open(args.data / f"{name}.jpg") as im:
            sheet.paste(im.resize((cw, ch), Image.LANCZOS), (x, y))
        color = (0, 120, 0) if conf >= 0.9 else (200, 120, 0) if conf >= 0.7 else (200, 0, 0)
        draw.text((x, y + ch + 4), f"{pred}  {conf:.2f}", fill=color, font=font)
    sheet.save(args.sheet)

    confs = [c for _, _, c in rows]
    n = len(confs)
    print(f"old model on portal: n={n} mean-conf={sum(confs) / n:.3f}")
    for th in (0.9, 0.8, 0.7, 0.5):
        print(f"conf>={th}: {sum(c >= th for c in confs) / n:.1%}")
    print(f"csv: {args.out_csv}\nsheet: {args.sheet}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

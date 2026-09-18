"""Run the universal CRNN on the portal captures and report confidence stats.

No ground truth is used here (VLM labeling was unreliable); outputs:
- per-image prediction + confidence CSV
- a preview sheet of 30 predictions for human eyeballing

Usage:
    python -m verification_code.analyze_portal --data data/portal_raw
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader

from .train_universal import (CrnnCaptcha, collate, ctc_greedy, preprocess,
                              set_img_h)


def portal_files(root: Path) -> list[Path]:
    return sorted(root.glob("*.jpg"))


def predict_folder(model: CrnnCaptcha, files: list[Path], device, batch_size=128):
    rows = []
    for s in range(0, len(files), batch_size):
        chunk = files[s:s + batch_size]
        images = []
        for path in chunk:
            with Image.open(path) as im:
                images.append(preprocess(im.convert("RGB"), model.img_h))
        batch = collate(list(zip(images, ["0000"] * len(chunk))))  # dummy labels
        with torch.no_grad():
            logits = model(batch[0].to(device))
        for path, (pred, conf) in zip(chunk, ctc_greedy(logits)):
            rows.append((path.stem, pred, conf))
    return rows


def make_sheet(rows: list[tuple[str, str, float]], src: Path, out: Path,
              sample: int = 30) -> None:
    step = max(1, len(rows) // sample)
    picked = rows[::step][:sample]
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
        with Image.open(src / f"{name}.jpg") as im:
            sheet.paste(im.resize((cw, ch), Image.LANCZOS), (x, y))
        color = (0, 120, 0) if conf >= 0.9 else (200, 120, 0) if conf >= 0.7 else (200, 0, 0)
        draw.text((x, y + ch + 4), f"{pred}  {conf:.2f}", fill=color, font=font)
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Portal inference stats (no GT).")
    parser.add_argument("--data", type=Path, default=Path("data/portal_raw"))
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("checkpoints/universal_crnn.pt"))
    parser.add_argument("--out-csv", type=Path, default=Path("reports/portal_predictions.csv"))
    parser.add_argument("--sheet", type=Path, default=Path("reports/portal_sheet.png"))
    return parser


def main() -> int:
    args = build_parser().parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    set_img_h(ckpt.get("img_h", 32))
    model = CrnnCaptcha().to(device)
    model.load_state_dict(ckpt["model_state"])

    folder_files = portal_files(args.data)
    rows = predict_folder(model, folder_files, device)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["file", "pred", "confidence"])
        writer.writerows(rows)

    confs = [c for _, _, c in rows]
    confs_sorted = sorted(confs)
    n = len(confs)
    lens = {}
    for _, p, _ in rows:
        lens[len(p)] = lens.get(len(p), 0) + 1
    print(f"n={n}")
    print(f"conf: mean={sum(confs) / n:.3f} median={confs_sorted[n // 2]:.3f}")
    for th in (0.9, 0.8, 0.7, 0.5):
        frac = sum(c >= th for c in confs) / n
        print(f"conf>={th}: {frac:.1%}")
    print(f"pred length distribution: {dict(sorted(lens.items()))}")
    make_sheet(rows, args.data, args.sheet)
    print(f"csv: {args.out_csv}\nsheet: {args.sheet}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

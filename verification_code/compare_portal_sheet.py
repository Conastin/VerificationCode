"""Side-by-side comparison sheet: portal captures with OLD slot-model vs NEW
universal-CRNN predictions on the same image (for human eyeballing, no GT).

Usage:
    python -m verification_code.compare_portal_sheet
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Old vs new model comparison sheet.")
    parser.add_argument("--data", type=Path, default=Path("data/portal_raw"))
    parser.add_argument("--old-csv", type=Path, default=Path("reports/portal_old_model.csv"))
    parser.add_argument("--new-csv", type=Path, default=Path("reports/portal_predictions.csv"))
    parser.add_argument("--out", type=Path, default=Path("reports/portal_compare_sheet.png"))
    parser.add_argument("--sample", type=int, default=30)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    old = {r["file"]: (r["pred_old_model"], float(r["min_confidence"]))
           for r in csv.DictReader(args.old_csv.open(encoding="utf-8"))}
    new = {r["file"]: (r["pred"], float(r["confidence"]))
           for r in csv.DictReader(args.new_csv.open(encoding="utf-8"))}
    common = sorted(set(old) & set(new))
    step = max(1, len(common) // args.sample)
    picked = common[::step][: args.sample]

    scale = 3
    cw, ch = 80 * scale, 34 * scale
    pad, label_h = 10, 34
    cols = 5
    rows_n = (len(picked) + cols - 1) // cols
    W = cols * (cw + 2 * pad) + pad
    H = rows_n * (ch + label_h * 2 + 2 * pad) + pad + 30
    sheet = Image.new("RGB", (W, H), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.truetype("C:/Windows/Fonts/consolab.ttf", 22)
    head_font = ImageFont.truetype("C:/Windows/Fonts/msyhbd.ttc", 24)
    draw.text((pad, 4), f"上排=旧模型预测    下排=新模型预测（{len(picked)} 张 portal 验证码）",
              fill=(0, 0, 0), font=head_font)
    for i, name in enumerate(picked):
        r, c = divmod(i, cols)
        x = pad + c * (cw + 2 * pad)
        y = 30 + pad + r * (ch + label_h * 2 + 2 * pad)
        with Image.open(args.data / f"{name}.jpg") as im:
            sheet.paste(im.resize((cw, ch), Image.LANCZOS), (x, y))
        op, oc = old[name]
        np_, nc = new[name]
        draw.text((x, y + ch + 2), f"旧 {op} {oc:.2f}", fill=(180, 0, 0), font=font)
        draw.text((x, y + ch + label_h + 4), f"新 {np_} {nc:.2f}",
                  fill=(0, 120, 0), font=font)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.out)
    print(f"saved: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# -*- coding: utf-8 -*-
"""把 portal_raw 的验证码拼成大图便于人工标注：每格 3x 放大 + 红色序号。"""
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

SRC = Path(__file__).resolve().parents[2] / "data" / "portal_raw"
OUT = Path(__file__).parent / "sheets"
SCALE = 3
COLS, ROWS = 4, 5
FONT = ImageFont.truetype("C:/Windows/Fonts/consola.ttf", 28)


def main() -> int:
    OUT.mkdir(exist_ok=True)
    files = sorted(SRC.glob("p_*.jpg"))
    per_sheet = COLS * ROWS
    cell_w, cell_h = 80 * SCALE, 34 * SCALE
    pad, header = 8, 40
    for sheet_idx in range(0, len(files), per_sheet):
        chunk = files[sheet_idx:sheet_idx + per_sheet]
        W = COLS * (cell_w + pad) + pad
        H = ROWS * (cell_h + header + pad) + pad
        canvas = Image.new("RGB", (W, H), (255, 255, 255))
        draw = ImageDraw.Draw(canvas)
        for j, path in enumerate(chunk):
            r, c = divmod(j, COLS)
            x = pad + c * (cell_w + pad)
            y = pad + r * (cell_h + header + pad)
            im = Image.open(path).resize((cell_w, cell_h), Image.LANCZOS)
            canvas.paste(im, (x, y + header))
            idx = sheet_idx + j
            draw.text((x + 4, y + 4), f"{idx:03d}", fill=(255, 0, 0), font=FONT)
            draw.rectangle([x - 1, y + header - 1, x + cell_w, y + header + cell_h], outline=(180, 180, 180))
        out_path = OUT / f"sheet_{sheet_idx // per_sheet:02d}.png"
        canvas.save(out_path)
        print(out_path, len(chunk))
    return 0


if __name__ == "__main__":
    sys.exit(main())

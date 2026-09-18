"""Domain-randomized synthetic CAPTCHA generator for universal CRNN training.

Renders randomized captchas covering a broad style space instead of replicating
one site: many fonts, random sizes, per-char colors (incl. grayscale), scribble
line interference, noise, blur, JPEG artifacts, elastic-ish warp, and variable
length (3-6) / variable canvas size. Output: PIL image + ground-truth string.

Usage (library):
    from verification_code.synth_universal import SynthCaptchaGenerator
    gen = SynthCaptchaGenerator()
    image, text = gen.render()          # image: PIL RGB, text: e.g. "K7VF"

CLI (dump a preview sheet):
    python -m verification_code.synth_universal --count 40 --out tmp/synth_sheet.png
"""

from __future__ import annotations

import argparse
import io
import math
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

# Full alphanumeric; case folded at training time (see train_universal.py).
CHARSET_UPPER = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
CHARSET_FULL = CHARSET_UPPER + "abcdefghijklmnopqrstuvwxyz"

# pairs the real sites' captchas actually confuse (v1-v3 error analysis)
CONFUSABLE_PAIRS = [
    ("0", "O"), ("1", "I"), ("1", "L"), ("2", "Z"), ("5", "S"), ("8", "B"),
    ("6", "G"), ("V", "W"), ("U", "V"), ("C", "G"), ("M", "N"), ("Q", "O"),
    ("I", "J"), ("A", "4"), ("H", "N"), ("K", "X"),
]

_FONT_CANDIDATES = [
    "arial.ttf", "arialbd.ttf", "ariblk.ttf", "tahoma.ttf", "tahomabd.ttf",
    "times.ttf", "timesbd.ttf", "verdana.ttf", "verdanab.ttf", "cour.ttf",
    "courbd.ttf", "consola.ttf", "consolab.ttf", "comic.ttf", "comicbd.ttf",
    "georgia.ttf", "georgiab.ttf", "trebuc.ttf", "trebucbd.ttf", "impact.ttf",
    "palab.ttf", "pala.ttf", "Gabriola.ttf", "seguihis.ttf", "framd.ttf",
    "Candara.ttf", "Candarab.ttf", "Constantine.ttf", "CORBEL.TTF", "GOTHIC.TTF",
    "GOTHICB.TTF", "MATURASC.TTF", "PLAYBILL.TTF", "STENCIL.TTF", "papyrus.ttf",
    "BRITANIC.TTF", "INFROMAN.TTF", "RAGE.TTF", "SEGOEPR.TTF", "SEGOESC.TTF",
]


class SynthCaptchaGenerator:
    """Random-style CAPTCHA renderer with exact ground truth."""

    def __init__(self, font_dir: str = "C:/Windows/Fonts", seed: int | None = None,
                 lowercase_prob: float = 0.2, confusable_prob: float = 0.25) -> None:
        self.font_dir = Path(font_dir)
        self.rng = random.Random(seed)
        fonts = [f for f in _FONT_CANDIDATES if (self.font_dir / f).is_file()]
        if not fonts:
            raise FileNotFoundError(f"no usable fonts under {font_dir}")
        self.fonts = fonts
        self.lowercase_prob = lowercase_prob
        self.confusable_prob = confusable_prob

    # ------------------------------------------------------------------ utils
    def _rand_color(self, mode: str) -> tuple[int, int, int]:
        rng = self.rng
        if mode == "gray":
            v = rng.randint(0, 130)
            return (v, v, v)
        if mode == "dark":
            return (rng.randint(0, 90), rng.randint(0, 90), rng.randint(0, 90))
        # vivid / per-char colored style
        return (rng.randint(20, 170), rng.randint(20, 170), rng.randint(20, 170))

    def _rand_bg(self) -> tuple[int, int, int]:
        rng = self.rng
        mode = rng.random()
        if mode < 0.35:
            return (255, 255, 255)
        if mode < 0.6:
            v = rng.randint(200, 245)
            return (v, v, v)
        return (rng.randint(190, 255), rng.randint(190, 255), rng.randint(190, 255))

    def _scribble(self, draw: ImageDraw.ImageDraw, w: int, h: int, count: int) -> None:
        """Dense diagonal scribble interference like the portal style."""
        rng = self.rng
        for _ in range(count):
            x0 = rng.randint(-10, w)
            y0 = rng.randint(-10, h + 10)
            angle = rng.uniform(math.pi * 0.15, math.pi * 0.35)  # ~45 deg family
            length = rng.uniform(w * 0.4, w * 1.4)
            x1 = x0 + length * math.cos(angle)
            y1 = y0 + length * math.sin(angle)
            color = self._rand_color("gray")
            width = rng.randint(1, 2)
            draw.line([x0, y0, x1, y1], fill=color, width=width)

    def _warp(self, image: Image.Image, strength: float) -> Image.Image:
        """Cheap sinusoidal warp (row/column displacement), numpy-vectorized."""
        if strength <= 0:
            return image
        rng = self.rng
        w, h = image.size
        period = rng.uniform(18.0, 42.0)
        amp = rng.uniform(0.5, 1.5) * strength
        phase_x = rng.uniform(0, math.tau)
        phase_y = rng.uniform(0, math.tau)
        ys = np.arange(h, dtype=np.int64)
        xs = np.arange(w, dtype=np.int64)
        dx = (amp * np.sin(2 * np.pi * ys / period + phase_x)).astype(np.int64)
        dy = (0.7 * amp * np.sin(2 * np.pi * xs / period + phase_y)).astype(np.int64)
        grid_x = xs[None, :] + dx[:, None]        # (h, w)
        grid_y = ys[:, None] + dy[None, :]        # (h, w)
        np.clip(grid_x, 0, w - 1, out=grid_x)
        np.clip(grid_y, 0, h - 1, out=grid_y)
        arr = np.asarray(image)
        return Image.fromarray(arr[grid_y, grid_x])

    # ------------------------------------------------------------------ render
    def render(self, length: int | None = None) -> tuple[Image.Image, str]:
        rng = self.rng
        if length is None:
            length = rng.choices([4, 5, 3, 6], weights=[0.75, 0.15, 0.05, 0.05])[0]

        # canvas: heights cover both real sites (20 / 34) plus headroom
        h = rng.choice([20, 24, 28, 32, 34, 36, 40])
        char_h = int(h * rng.uniform(0.6, 0.8))
        width_per_char = rng.uniform(0.55, 1.1) * char_h
        margin = rng.randint(2, 8)
        w = int(margin * 2 + length * width_per_char)
        w = max(w, 24)

        bg = self._rand_bg()
        image = Image.new("RGB", (w, h), bg)
        draw = ImageDraw.Draw(image)

        # interference before chars (chars drawn on top) and/or after
        pre_lines = rng.randint(2, 10)
        self._scribble(draw, w, h, pre_lines)

        # character rendering
        color_mode = rng.random()
        text_chars: list[str] = []
        font_mode = rng.random()
        same_font = font_mode < 0.7
        base_font_name = rng.choice(self.fonts)
        for i in range(length):
            char = rng.choice(CHARSET_UPPER)
            if rng.random() < self.lowercase_prob and char.isalpha():
                char = char.lower()
            text_chars.append(char)

        # confusable-pair boost: place a real confusion pair into adjacent
        # slots so the model must learn the fine distinction
        if rng.random() < self.confusable_prob and length >= 2:
            pair = rng.choice(CONFUSABLE_PAIRS)
            pos = rng.randrange(length - 1)
            text_chars[pos], text_chars[pos + 1] = pair[0], pair[1]

        for i, char in enumerate(text_chars):
            font_name = base_font_name if same_font else rng.choice(self.fonts)
            size = max(10, int(char_h * rng.uniform(0.85, 1.15)))
            try:
                font = ImageFont.truetype(str(self.font_dir / font_name), size)
            except OSError:
                font = ImageFont.truetype(str(self.font_dir / self.fonts[0]), size)

            if color_mode < 0.5:  # per-char color (portal style)
                color = self._rand_color("vivid")
            elif color_mode < 0.8:  # single dark color (old-site style)
                color = self._rand_color("dark")
            else:  # grayscale
                color = self._rand_color("gray")

            cx = margin + i * width_per_char + rng.uniform(-1.5, 1.5)
            cy = (h - size) / 2 + rng.uniform(-2.5, 2.5)
            angle = rng.uniform(-22, 22)

            # render char to its own layer, rotate, paste
            pad = size + 8
            layer = Image.new("RGBA", (pad * 2, pad * 2), (0, 0, 0, 0))
            ldraw = ImageDraw.Draw(layer)
            ldraw.text((pad, pad), char, font=font, fill=color + (255,))
            if angle:
                layer = layer.rotate(angle, resample=Image.BICUBIC, center=(pad, pad))
            image.paste(layer, (int(cx - pad), int(cy - pad)), layer)

        # post interference (chars partially hidden behind lines)
        post_lines = rng.randint(0, 6)
        draw = ImageDraw.Draw(image)
        self._scribble(draw, w, h, post_lines)

        # noise dots
        if rng.random() < 0.7:
            for _ in range(rng.randint(10, 120)):
                x, y = rng.randint(0, w - 1), rng.randint(0, h - 1)
                draw.point((x, y), fill=self._rand_color("gray"))

        # warp / blur / jpeg artifacts (moderate: cover real styles, not extremes)
        if rng.random() < 0.5:
            image = self._warp(image, strength=rng.uniform(0.3, 1.2))
        if rng.random() < 0.5:
            image = image.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 0.9)))
        if rng.random() < 0.45:
            buf = io.BytesIO()
            image.save(buf, format="JPEG", quality=rng.randint(50, 88))
            buf.seek(0)
            image = Image.open(buf).copy().convert("RGB")

        return image, "".join(text_chars)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Synthetic CAPTCHA preview sheet.")
    parser.add_argument("--count", type=int, default=40)
    parser.add_argument("--out", type=Path, default=Path("tmp/synth_sheet.png"))
    parser.add_argument("--seed", type=int, default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    gen = SynthCaptchaGenerator(seed=args.seed)
    cell = 200
    cols = 5
    rows = (args.count + cols - 1) // cols
    sheet = Image.new("RGB", (cols * cell, rows * (cell // 2 + 24)), (240, 240, 240))
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.truetype("C:/Windows/Fonts/consola.ttf", 18)
    for i in range(args.count):
        img, text = gen.render()
        r, c = divmod(i, cols)
        x0, y0 = c * cell + 10, r * (cell // 2 + 24) + 20
        sheet.paste(img, (x0, y0))
        draw.text((x0, y0 + img.height + 4), f"{i:02d}: {text}", fill=(0, 0, 0), font=font)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.out)
    print(f"sheet saved: {args.out} ({args.count} samples)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

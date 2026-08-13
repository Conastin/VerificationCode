"""Run inference on CAPTCHA images with a trained checkpoint.

Usage:
    python -m verification_code.infer --checkpoint checkpoints/best.pt image1.jpg image2.jpg
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image
from torch import nn

import numpy as np

from .model import DEFAULT_CHARSET
from .model import build_model_from_checkpoint, decode_logits


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Predict labels for CAPTCHA images.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--charset", default=DEFAULT_CHARSET)
    parser.add_argument("--device", default="auto")
    parser.add_argument("images", nargs="+", type=Path, help="image files to classify")
    return parser


@torch.no_grad()
def predict_image(
    model: nn.Module,
    image_path: Path,
    device: torch.device,
    charset: str,
) -> tuple[str, torch.Tensor]:
    with Image.open(image_path) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
    logits = model(tensor.to(device))
    label, conf = decode_logits(logits, charset)
    return label[0], conf[0]


def main() -> int:
    args = build_parser().parse_args()
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    charset = checkpoint.get("charset", args.charset)
    model = build_model_from_checkpoint(checkpoint, charset).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    for image_path in args.images:
        label, conf = predict_image(model, image_path, device, charset)
        conf_str = ", ".join(f"{value:.3f}" for value in conf.tolist())
        print(f"{image_path}: {label}  (per-slot confidence: {conf_str})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

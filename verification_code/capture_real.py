"""Capture real CAPTCHAs from the target login page and label them with the
trained recognizer.

Downloads fresh CAPTCHA images from the target site, runs the SlotCaptchaModel
on each, and renames every file to its recognized label (with a numeric suffix
on ties so no file is overwritten). A manifest CSV keeps the mapping.

Usage:
    python -m verification_code.capture_real --count 100 \
        --out data/real_captcha --checkpoint checkpoints/best.pt
"""

import os
from __future__ import annotations

import argparse
import csv
import io
import ssl
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .model import DEFAULT_CHARSET
from .model import build_model_from_checkpoint, decode_logits

DEFAULT_URL = os.environ.get("OLD_SITE_BASE", "https://old.example.internal") + "/fort/pages/commons/image.jsp"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Capture and label real CAPTCHAs.")
    parser.add_argument("--url", default=DEFAULT_URL, help="CAPTCHA image endpoint")
    parser.add_argument("--count", type=int, default=100, help="number of images to capture")
    parser.add_argument("--out", type=Path, default=Path("data/real_captcha"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/best.pt"))
    parser.add_argument("--charset", default=None, help="overrides checkpoint charset")
    parser.add_argument("--no-verify", action="store_true", default=True,
                        help="skip TLS certificate verification (self-signed test target)")
    return parser


def fetch_image(url: str, context: ssl.SSLContext | None) -> bytes:
    request = urllib.request.Request(url)
    with urllib.request.urlopen(request, timeout=20, context=context) as response:
        return response.read()


@torch.no_grad()
def predict_image_bytes(model: torch.nn.Module, data: bytes, device: torch.device,
                        charset: str) -> tuple[str, torch.Tensor]:
    with Image.open(io.BytesIO(data)) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
    logits = model(tensor.to(device))
    label, conf = decode_logits(logits, charset)
    return label[0], conf[0]


def main() -> int:
    args = build_parser().parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    charset = args.charset or checkpoint.get("charset", DEFAULT_CHARSET)
    model = build_model_from_checkpoint(checkpoint, charset).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    context = ssl.create_default_context()
    if args.no_verify:
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE

    args.out.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out / "capture_manifest.csv"
    used_names: defaultdict[str, int] = defaultdict(int)

    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["source_file", "label", "predicted_file", "min_confidence"])
        captured = 0
        for index in range(args.count):
            raw_name = f"raw_{index:04d}.jpg"
            raw_path = args.out / raw_name
            try:
                data = fetch_image(args.url, context)
            except (urllib.error.URLError, TimeoutError, OSError) as error:
                print(f"capture {index}: fetch failed: {error}")
                continue
            raw_path.write_bytes(data)
            label, conf = predict_image_bytes(model, data, device, charset)
            used_names[label] += 1
            predicted_name = f"{label}_{used_names[label]}.jpg" if used_names[label] > 1 else f"{label}.jpg"
            predicted_path = args.out / predicted_name
            raw_path.rename(predicted_path)
            writer.writerow([raw_name, label, predicted_name, round(conf.min().item(), 4)])
            captured += 1
            print(f"{index:4d} {raw_name} -> {predicted_name}  conf={conf.min().item():.3f}")

    print(f"captured={captured} output={args.out} manifest={manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

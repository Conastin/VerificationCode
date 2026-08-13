"""Export a trained CaptchaCNN checkpoint to ONNX.

Usage:
    python -m verification_code.export_onnx --checkpoint checkpoints/best.pt \
        --out checkpoints/best.onnx
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .model import DEFAULT_CHARSET
from .model import build_model_from_checkpoint


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export CaptchaCNN to ONNX.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("checkpoints/model.onnx"))
    parser.add_argument("--charset", default=DEFAULT_CHARSET)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    charset = checkpoint.get("charset", args.charset)
    model = build_model_from_checkpoint(checkpoint, charset)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    dummy = torch.randn(1, 3, 20, 60)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy,
        args.out,
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
    )
    print(f"exported {args.out} ({args.out.stat().st_size / 1024:.1f} KiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

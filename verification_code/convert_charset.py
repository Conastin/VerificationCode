"""Convert a checkpoint to a reduced charset (e.g. 36 -> 31 classes).

The real CAPTCHA charset excludes ambiguous characters (0, 1, I, L, O), so
the model output layer can be shrunk: weights of the removed classes are
dropped and the remaining classes remapped. Only the final Linear layer is
rebuilt; the shared backbone is copied unchanged.

Usage:
    python -m verification_code.convert_charset \
        --checkpoint checkpoints/finetuned_2200/best.pt \
        --charset "23456789ABCDEFGHJKMNPQRSTUVWXYZ" \
        --out checkpoints/best_31.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .model import DEFAULT_CHARSET
from .model import SlotCaptchaModel


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert checkpoint to a reduced charset.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--charset", required=True, help="new charset (subset of the old one)")
    parser.add_argument("--out", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    old_ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    old_charset = old_ckpt.get("charset", DEFAULT_CHARSET)
    if args.charset == old_charset:
        raise SystemExit("new charset equals old charset; nothing to convert")

    index_map = [old_charset.index(char) for char in args.charset]
    old_state = old_ckpt["model_state"]

    new_model = SlotCaptchaModel(length=4, num_classes=len(args.charset))
    new_state = new_model.state_dict()
    # Copy shared layers, remap the final Linear layer's output rows.
    for key, value in old_state.items():
        if key not in new_state:
            continue
        if "char_net.14." in key:  # final Linear(256, num_classes)
            if "weight" in key:
                new_state[key] = value[index_map]
            else:
                new_state[key] = value[index_map]
        else:
            new_state[key] = value

    new_ckpt = dict(old_ckpt)
    new_ckpt["charset"] = args.charset
    new_ckpt["model_state"] = new_state
    new_ckpt["converted_from"] = str(args.checkpoint)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(new_ckpt, args.out)
    print(f"converted {len(old_charset)} -> {len(args.charset)} classes: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

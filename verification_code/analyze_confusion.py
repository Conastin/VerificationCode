"""Analyze character-level confusions of a trained checkpoint on a test set.

Prints the most common (true, predicted) confusions per slot and per character
so the charset can be pruned (e.g. O/0, I/1) or training can be improved.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .dataset import CaptchaDataset
from .model import DEFAULT_CHARSET
from .model import CaptchaCNN


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Character confusion analysis.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--charset", default=DEFAULT_CHARSET)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--top", type=int, default=15)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    charset = checkpoint.get("charset", args.charset)
    model = CaptchaCNN(length=checkpoint.get("length", 4), num_classes=len(charset)).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    dataset = CaptchaDataset(args.data, charset=charset)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    per_slot = [Counter() for _ in range(4)]
    per_char = Counter()
    total_chars = 0
    with torch.no_grad():
        for images, labels in loader:
            logits = model(images.to(device))
            preds = logits.argmax(dim=-1).cpu()
            for slot in range(4):
                for true_id, pred_id in zip(labels[:, slot].tolist(), preds[:, slot].tolist()):
                    per_slot[slot][(charset[true_id], charset[pred_id])] += 1
                    per_char[(charset[true_id], charset[pred_id])] += 1
                    total_chars += 1

    print(f"total characters: {total_chars}, overall char accuracy: "
          f"{sum(v for k, v in per_char.items() if k[0] == k[1]) / total_chars:.4f}")
    for slot in range(4):
        correct = sum(v for k, v in per_slot[slot].items() if k[0] == k[1])
        total = sum(per_slot[slot].values())
        print(f"\nslot {slot + 1} accuracy: {correct / total:.4f}  top confusions:")
        for (true, pred), count in per_slot[slot].most_common(args.top):
            if true != pred:
                print(f"  {true} -> {pred}: {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

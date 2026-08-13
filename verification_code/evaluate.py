"""Evaluate a trained CaptchaCNN checkpoint on a held-out test set.

Reports per-slot accuracy, full-image accuracy, char-level accuracy, and a
rejection (low-confidence) analysis: for a series of minimum-confidence
thresholds it shows coverage and accuracy when samples below the threshold
are refused.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from .dataset import CaptchaDataset
from .model import DEFAULT_CHARSET
from .model import build_model_from_checkpoint, decode_logits


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float, list[float]]:
    """Return (full accuracy, char accuracy, per-slot accuracy)."""
    model.eval()
    full_correct = 0
    char_correct = 0
    char_total = 0
    slot_correct = [0] * model.length
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        preds = logits.argmax(dim=-1)
        per_slot = (preds == labels)
        full_correct += per_slot.all(dim=1).sum().item()
        char_correct += per_slot.sum().item()
        char_total += labels.numel()
        for slot in range(model.length):
            slot_correct[slot] += per_slot[:, slot].sum().item()
    total = len(loader.dataset)
    return (
        full_correct / total,
        char_correct / char_total,
        [count / total for count in slot_correct],
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a CaptchaCNN checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True, help="test data directory with labels.txt")
    parser.add_argument("--charset", default=DEFAULT_CHARSET)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", type=Path, default=None, help="write metrics JSON to this path")
    return parser


@torch.no_grad()
def rejection_analysis(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    charset: str,
    thresholds: tuple[float, ...] = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99),
) -> list[dict[str, float]]:
    """Coverage/accuracy pairs when refusing samples below each threshold.

    Confidence is the minimum per-slot softmax probability. Samples whose
    minimum confidence falls below the threshold are refused (not counted as
    correct), which is the behavior a real auto-fill integration would use.
    """
    model.eval()
    results: list[dict[str, float]] = []
    for threshold in thresholds:
        accepted = 0
        accepted_correct = 0
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            _, conf = decode_logits(logits, charset)
            mask = conf.min(dim=1).values >= threshold
            if not mask.any():
                continue
            preds = logits.argmax(dim=-1)[mask]
            accepted += mask.sum().item()
            accepted_correct += (preds == labels[mask]).all(dim=1).sum().item()
        total = len(loader.dataset)
        results.append(
            {
                "threshold": threshold,
                "coverage": accepted / total,
                "accuracy": accepted_correct / accepted if accepted else 1.0,
            }
        )
    return results


def main() -> int:
    args = build_parser().parse_args()
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    charset = checkpoint.get("charset", args.charset)
    model = build_model_from_checkpoint(checkpoint, charset).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    test_set = CaptchaDataset(args.data, charset=charset)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    full_acc, char_acc, slot_acc = evaluate(model, test_loader, device)
    print(f"checkpoint: {args.checkpoint}")
    print(f"test samples: {len(test_set)}")
    print(f"full-image accuracy: {full_acc:.4f}")
    print(f"char-level accuracy: {char_acc:.4f}")
    print(f"per-slot accuracy:   {[f'{x:.4f}' for x in slot_acc]}")

    rows = rejection_analysis(model, test_loader, device, charset)
    print("\nrejection analysis (min per-slot confidence threshold):")
    print(f"{'threshold':>9} {'coverage':>9} {'accuracy':>9}")
    for row in rows:
        print(f"{row['threshold']:>9.2f} {row['coverage']:>9.4f} {row['accuracy']:>9.4f}")

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        metrics = {
            "checkpoint": str(args.checkpoint),
            "test_samples": len(test_set),
            "full_accuracy": full_acc,
            "char_accuracy": char_acc,
            "slot_accuracy": slot_acc,
            "rejection": rows,
        }
        args.out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(f"metrics written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

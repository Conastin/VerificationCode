"""Fine-tune the recognizer on manually corrected real CAPTCHAs.

Mixes the corrected real samples with random synthetic samples every epoch so
the model adapts to the real distribution without forgetting the synthetic
one. A small holdout of real samples is used for validation.

Usage:
    python -m verification_code.finetune_real \
        --base checkpoints/best.pt --real data/real_captcha \
        --out checkpoints/finetuned --epochs 15
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .dataset import CaptchaDataset
from .model import DEFAULT_CHARSET
from .model import build_model_from_checkpoint, encode_label


class RealCaptchaDataset(Dataset):
    """Corrected real CAPTCHAs, optionally with light augmentation.

    Reads ``labels_corrected.txt`` (filename label) produced by
    verify_captcha.py; falls back to the recognizer's predicted labels from
    capture_manifest.csv when the file is missing. Augmentation is a tiny
    random translation (<=1px) and brightness jitter to combat overfitting on
    the small real set.
    """

    def __init__(
        self,
        directory: Path,
        charset: str = DEFAULT_CHARSET,
        augment: bool = False,
    ) -> None:
        self.directory = Path(directory)
        self.charset = charset
        self.augment = augment
        self.items: list[tuple[Path, str]] = []
        directory = self.directory

        labels_path = directory / "labels_corrected.txt"
        if labels_path.is_file():
            with labels_path.open(encoding="utf-8") as handle:
                for line in handle:
                    parts = line.split()
                    path = directory / parts[0]
                    if len(parts) == 2 and path.is_file():
                        self.items.append((path, parts[1]))
        else:
            manifest_path = directory / "capture_manifest.csv"
            if not manifest_path.is_file():
                raise FileNotFoundError(f"need labels_corrected.txt or capture_manifest.csv in {directory}")
            with manifest_path.open(encoding="utf-8") as handle:
                next(handle, None)
                for row in handle:
                    parts = row.strip().split(",")
                    if len(parts) >= 3 and parts[1] and parts[2]:
                        self.items.append((directory / parts[2], parts[1]))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        path, label = self.items[index]
        with Image.open(path) as image:
            array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1)
        if self.augment:
            # Real captchas drift up to ~3px from the synthetic slot grid, so
            # the translation augmentation must cover that range.
            dy = random.randint(-3, 3)
            dx = random.randint(-3, 3)
            tensor = torch.roll(tensor, shifts=(dy, dx), dims=(1, 2))
            tensor *= random.uniform(0.9, 1.1)
            tensor = tensor.clamp(0.0, 1.0)
        label_ids = torch.tensor(encode_label(label, self.charset), dtype=torch.long)
        return tensor, label_ids


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fine-tune on corrected real CAPTCHAs.")
    parser.add_argument("--base", type=Path, default=Path("checkpoints/best.pt"))
    parser.add_argument("--real", type=Path, default=Path("data/real_captcha"))
    parser.add_argument("--syn-train", type=Path, default=Path("data/train"))
    parser.add_argument("--syn-val", type=Path, default=Path("data/val"))
    parser.add_argument("--out", type=Path, default=Path("checkpoints/finetuned"))
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--syn-ratio", type=float, default=0.25,
                        help="fraction of synthetic samples mixed per epoch "
                             "(0.25 = real:synthetic 4:1)")
    parser.add_argument("--holdout", type=int, default=20,
                        help="real samples held out for validation (fixed seed)")
    parser.add_argument("--syn-val-size", type=int, default=2000,
                        help="synthetic samples used per validation")
    parser.add_argument("--seed", type=int, default=42)
    return parser


@torch.no_grad()
def evaluate_subsample(model: nn.Module, dataset: Dataset, size: int,
                       device: torch.device, rng: random.Random) -> tuple[float, float]:
    """Full-image and char accuracy on a random subsample of a dataset."""
    indices = rng.sample(range(len(dataset)), min(size, len(dataset)))
    full_correct = char_correct = 0
    char_total = 0
    model.eval()
    for start in range(0, len(indices), 256):
        batch = [dataset[i] for i in indices[start:start + 256]]
        images = torch.stack([item[0] for item in batch])
        labels = torch.stack([item[1] for item in batch])
        logits = model(images.to(device)).cpu()
        per_slot = logits.argmax(-1) == labels
        full_correct += per_slot.all(dim=1).sum().item()
        char_correct += per_slot.sum().item()
        char_total += labels.numel()
    return full_correct / len(indices), char_correct / char_total


def main() -> int:
    args = build_parser().parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base = torch.load(args.base, map_location=device, weights_only=False)
    charset = base.get("charset", DEFAULT_CHARSET)
    model = build_model_from_checkpoint(base, charset).to(device)
    model.load_state_dict(base["model_state"])
    print(f"loaded base: {args.base} (arch={base.get('arch', 'full')})")

    real_all = RealCaptchaDataset(args.real, charset=charset, augment=True)
    rng = random.Random(args.seed)
    indices = list(range(len(real_all)))
    rng.shuffle(indices)
    holdout_idx = indices[:args.holdout]
    train_idx = indices[args.holdout:]
    if not train_idx:
        raise SystemExit(f"need more than {args.holdout} real samples; have {len(real_all)}")
    real_train = list(train_idx)
    print(f"real samples: total={len(real_all)} train={len(real_train)} holdout={len(holdout_idx)}")

    syn_pool = CaptchaDataset(args.syn_train, charset=charset)
    syn_val = CaptchaDataset(args.syn_val, charset=charset)
    print(f"synthetic pool: {len(syn_pool)}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    args.out.mkdir(parents=True, exist_ok=True)

    # baseline on holdout before fine-tuning
    holdout_ds = RealCaptchaDataset(args.real, charset=charset, augment=False)
    holdout_items = [holdout_ds[i] for i in holdout_idx]
    baseline_full, baseline_char = 0.0, 0.0
    if holdout_items:
        model.eval()
        with torch.no_grad():
            images = torch.stack([item[0] for item in holdout_items])
            labels = torch.stack([item[1] for item in holdout_items])
            per_slot = model(images.to(device)).cpu().argmax(-1) == labels
            baseline_full = per_slot.all(dim=1).float().mean().item()
            baseline_char = per_slot.float().mean().item()
    print(f"before: real holdout full={baseline_full:.3f} char={baseline_char:.3f}")

    best_holdout = baseline_full
    for epoch in range(args.epochs):
        model.train()
        # Re-sample real items every epoch so augmentation is re-applied.
        syn_count = int(len(train_idx) * args.syn_ratio / (1 - args.syn_ratio))
        syn_indices = rng.sample(range(len(syn_pool)), min(syn_count, len(syn_pool)))
        combined = [real_all[i] for i in train_idx] + [syn_pool[i] for i in syn_indices]
        rng.shuffle(combined)
        running_loss = 0.0
        batches = 0
        for start in range(0, len(combined), args.batch_size):
            batch = combined[start:start + args.batch_size]
            images = torch.stack([item[0] for item in batch])
            labels = torch.stack([item[1] for item in batch])
            optimizer.zero_grad()
            logits = model(images.to(device))
            loss = criterion(logits.reshape(-1, len(charset)), labels.reshape(-1).to(device))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            batches += 1

        holdout_full, holdout_char = 0.0, 0.0
        if holdout_items:
            model.eval()
            with torch.no_grad():
                images = torch.stack([item[0] for item in holdout_items])
                labels = torch.stack([item[1] for item in holdout_items])
                per_slot = model(images.to(device)).cpu().argmax(-1) == labels
                holdout_full = per_slot.all(dim=1).float().mean().item()
                holdout_char = per_slot.float().mean().item()
        syn_full, syn_char = evaluate_subsample(model, syn_val, args.syn_val_size, device, rng)
        print(
            f"epoch {epoch + 1}/{args.epochs} | loss {running_loss / batches:.4f} "
            f"| real-holdout full {holdout_full:.3f} char {holdout_char:.3f} "
            f"| syn-val full {syn_full:.3f} char {syn_char:.3f}"
        )

        if holdout_full >= best_holdout:
            best_holdout = holdout_full
            torch.save(
                {
                    "epoch": epoch,
                    "arch": base.get("arch", "full"),
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "val_acc": holdout_full,
                    "charset": charset,
                    "length": 4,
                    "finetuned_on": str(args.real),
                },
                args.out / "best.pt",
            )
    torch.save(
        {
            "epoch": args.epochs - 1,
            "arch": base.get("arch", "full"),
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "val_acc": holdout_full,
            "charset": charset,
            "length": 4,
            "finetuned_on": str(args.real),
        },
        args.out / "last.pt",
    )
    print(f"best real-holdout full accuracy: {best_holdout:.3f} -> {args.out / 'best.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

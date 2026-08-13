"""Sanity check: how well can a single-char classifier learn cropped slots?

Crops each character slot from generated CAPTCHAs into 15x20 tiles and trains
a small single-head CNN. High accuracy here proves the images carry enough
information and the full-image 4-head model is the bottleneck; low accuracy
means the rendered characters themselves are too hard.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset

import numpy as np

from .model import DEFAULT_CHARSET


class SlotCropDataset(Dataset):
    def __init__(self, data_dir: str | Path, limit: int | None = None) -> None:
        self.data_dir = Path(data_dir)
        self.charset = DEFAULT_CHARSET
        self.items: list[tuple[Path, int]] = []
        with (self.data_dir / "labels.txt").open(encoding="utf-8") as handle:
            for line in handle:
                filename, label = line.split()
                for slot, char in enumerate(label):
                    self.items.append(
                        (self.data_dir / filename, slot, self.charset.index(char))
                    )
                    if limit is not None and len(self.items) >= limit:
                        return

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        path, slot, char_id = self.items[index]
        with Image.open(path) as image:
            crop = image.crop((slot * 15, 0, slot * 15 + 15, 20))
            array = np.asarray(crop, dtype=np.float32) / 255.0
        return torch.from_numpy(array).permute(2, 0, 1), torch.tensor(char_id, dtype=torch.long)


class SingleCharCNN(nn.Module):
    def __init__(self, num_classes: int = 36) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 7x10
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(128 * 7 * 10, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Single-char slot crop sanity check.")
    parser.add_argument("--data", type=Path, default=Path("data/train"))
    parser.add_argument("--val", type=Path, default=Path("data/val"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--limit", type=int, default=None, help="max crop samples")
    parser.add_argument("--out", type=Path, default=Path("checkpoints/slot_crop.pt"))
    return parser


def main() -> int:
    args = build_parser().parse_args()
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_set = SlotCropDataset(args.data, limit=args.limit)
    val_set = SlotCropDataset(args.val, limit=args.limit)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0)
    print(f"crop samples: train={len(train_set)} val={len(val_set)}")

    model = SingleCharCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best = 0.0
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        scheduler.step()

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                preds = model(images.to(device)).argmax(dim=-1).cpu()
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        acc = correct / total
        best = max(best, acc)
        print(f"epoch {epoch + 1}/{args.epochs} | loss {running_loss / len(train_loader):.4f} | val acc {acc:.4f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "val_acc": best}, args.out)
    print(f"best val acc {best:.4f} -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

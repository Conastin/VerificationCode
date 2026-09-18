"""Universal CRNN + CTC model and its training loop.

Goal: ONE variable-size, variable-length recognizer covering arbitrary text
captcha styles (no site-specific slot geometry or fixed canvas size).

- Input: RGB image of any size; the dataset resizes height to IMG_H and pads
  width to a batch max (right-side pad).
- Output: per-frame class probabilities; CTC decodes to a 3-6 char string.
- Charset: case-folded alphanumerics (36 classes) + CTC blank. Lowercase in
  training images is folded to uppercase targets; at inference the prediction
  is uppercase (login backends here are case-insensitive in practice, and the
  two real sites use uppercase/digits).

Usage:
    python -m verification_code.train_universal --steps 8000
    python -m verification_code.train_universal --evaluate-old data/real_test_independent
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .synth_universal import SynthCaptchaGenerator

FOLD_CHARSET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"  # 36 classes, case-folded
# (portal 登录实测大小写不敏感，统一折叠大写训练；如将来需要区分大小写再扩 62 类)
BLANK = 36  # CTC blank index

IMG_H = 32  # default training height; checkpoints may override (see set_img_h)


def set_img_h(img_h: int) -> None:
    """Switch the module-wide input height (must match the loaded checkpoint)."""
    global IMG_H
    IMG_H = img_h


def encode_text(text: str) -> list[int]:
    return [FOLD_CHARSET.index(ch) for ch in text.upper() if ch in FOLD_CHARSET]


def preprocess(image: Image.Image, img_h: int = IMG_H, channels: int = 3) -> np.ndarray:
    """Height-normalize keeping aspect ratio; returns HxWxC float32.

    channels=4 appends a saturation map (max-min)/max: portal chars are
    chromatic while scribble lines are gray, so S highlights strokes.
    """
    w, h = image.size
    new_w = max(8, round(w * img_h / h))
    resized = image.resize((new_w, img_h), Image.BILINEAR)
    arr = np.asarray(resized, dtype=np.float32) / 255.0
    if channels == 3:
        return arr
    v = arr.max(axis=2)
    s = np.where(v > 0, (v - arr.min(axis=2)) / np.maximum(v, 1e-6), 0.0)
    return np.concatenate([arr, s[..., None]], axis=2)


class CrnnCaptcha(nn.Module):
    """Compact CNN -> BiLSTM -> CTC head for variable-width captchas."""

    def __init__(self, num_classes: int = 36, lstm_hidden: int = 192,
                 img_h: int | None = None, lstm_layers: int = 2,
                 in_channels: int = 3) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.img_h = img_h or IMG_H
        self.in_channels = in_channels
        channels = (32, 64, 128)
        c1, c2, c3 = channels
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, c1, 3, padding=1), nn.BatchNorm2d(c1), nn.ReLU(True),
            nn.MaxPool2d(2),                                # H/2
            nn.Conv2d(c1, c2, 3, padding=1), nn.BatchNorm2d(c2), nn.ReLU(True),
            nn.MaxPool2d(2),                                # H/4
            nn.Conv2d(c2, c3, 3, padding=1), nn.BatchNorm2d(c3), nn.ReLU(True),
            nn.MaxPool2d((2, 1)),                           # H/8, W unchanged
            nn.Conv2d(c3, c3, 3, padding=1), nn.BatchNorm2d(c3), nn.ReLU(True),
        )
        # flatten channel*height into the LSTM input instead of pooling
        # (pooling destroyed stroke detail).
        feat_dim = c3 * (self.img_h // 8)
        self.rnn = nn.LSTM(feat_dim, lstm_hidden, num_layers=lstm_layers,
                           bidirectional=True, batch_first=False)
        self.head = nn.Linear(2 * lstm_hidden, num_classes + 1)  # + blank

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B,3,H,W) -> logits (B, T, C+1) with T = W/4."""
        feats = self.backbone(x)               # (B, C, H/8, W/4)
        b, c, h, w = feats.shape
        feats = feats.permute(3, 0, 1, 2).reshape(w, b, c * h)  # (T,B,CH)
        seq, _ = self.rnn(feats)
        logits = self.head(seq)                # (T, B, C+1)
        return logits.permute(1, 0, 2)         # (B, T, C+1)


def ctc_greedy(logits: torch.Tensor, label_mode: bool = True) -> list[tuple[str, float]]:
    """Greedy CTC decode; returns list of (text, avg_nonblank_prob)."""
    probs = torch.softmax(logits, dim=-1)
    frames = probs.argmax(dim=-1)              # (B, T)
    frame_probs = probs.max(dim=-1).values
    results: list[tuple[str, float]] = []
    for b in range(frames.shape[0]):
        chars: list[str] = []
        confs: list[float] = []
        prev = BLANK
        for t, idx in enumerate(frames[b].tolist()):
            if idx != prev and idx != BLANK:
                chars.append(FOLD_CHARSET[idx])
                confs.append(frame_probs[b, t].item())
            prev = idx
        results.append(("".join(chars), float(np.mean(confs)) if confs else 0.0))
    return results


class SynthStream(Dataset):
    """Infinite synthetic stream; each epoch yields fresh random renders."""

    def __init__(self, gen: SynthCaptchaGenerator, size: int,
                 img_h: int | None = None, channels: int = 3) -> None:
        self.gen = gen
        self.size = size
        self.img_h = img_h or IMG_H
        self.channels = channels

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> tuple[np.ndarray, str]:
        image, text = self.gen.render()
        return preprocess(image, self.img_h, self.channels), text


class EvalFolder(Dataset):
    """Labeled folder: either labels.txt ("<file> <label>" per line) or files
    named <label>.jpg (old-site convention)."""

    def __init__(self, root: Path, img_h: int | None = None,
                 channels: int = 3) -> None:
        self.img_h = img_h or IMG_H
        self.channels = channels
        self.samples = []
        root = Path(root)
        labels_file = root / "labels.txt"
        if labels_file.is_file():
            for line in labels_file.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if len(parts) == 2 and (root / parts[0]).is_file() \
                        and all(ch in FOLD_CHARSET for ch in parts[1].upper()):
                    self.samples.append((root / parts[0], parts[1].upper()))
        else:
            for path in sorted(root.glob("*.jpg")):
                label = path.stem.upper()
                if all(ch in FOLD_CHARSET for ch in label):
                    self.samples.append((path, label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[np.ndarray, str]:
        path, label = self.samples[index]
        with Image.open(path) as image:
            return preprocess(image.convert("RGB"), self.img_h, self.channels), label


def collate(batch: list[tuple[np.ndarray, str]]):
    """Pad widths to batch max (rounded up to a multiple of 16 so conv shapes
    stay in a small bucket set); build CTC tensors. Height is taken from the
    arrays themselves (workers don't inherit module-global IMG_H on Windows)."""
    images, texts = zip(*batch)
    img_h = images[0].shape[0]
    n_ch = images[0].shape[2]
    max_w = max(img.shape[1] for img in images)
    max_w = math.ceil(max_w / 16) * 16
    imgs = np.zeros((len(images), img_h, max_w, n_ch), dtype=np.float32)
    for i, img in enumerate(images):
        imgs[i, :, : img.shape[1], :] = img
    tensor = torch.from_numpy(imgs).permute(0, 3, 1, 2)  # (B,3,H,W)
    targets = torch.tensor([c for t in texts for c in encode_text(t)], dtype=torch.long)
    target_lens = torch.tensor([len(encode_text(t)) for t in texts], dtype=torch.long)
    input_lens = torch.full((len(images),), imgs.shape[2] // 4, dtype=torch.long)
    return tensor, targets, input_lens, target_lens, list(texts)


def run_batch(model: CrnnCaptcha, batch, device, optimizer=None, scaler=None,
              with_metrics: bool = False):
    images, targets, input_lens, target_lens, texts = batch
    images = images.to(device, non_blocking=True)
    targets = targets.to(device)
    input_lens = input_lens.to(device)
    target_lens = target_lens.to(device)

    ctx = torch.autocast("cuda", dtype=torch.float16) if scaler else torch.autocast("cpu", enabled=False)
    with ctx:
        logits = model(images)
        if scaler:
            loss = nn.functional.ctc_loss(
                logits.log_softmax(-1).permute(1, 0, 2).float(), targets,
                input_lens, target_lens, blank=BLANK, zero_infinity=True)
        else:
            loss = nn.functional.ctc_loss(
                logits.log_softmax(-1).permute(1, 0, 2), targets,
                input_lens, target_lens, blank=BLANK, zero_infinity=True)
    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)
        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
    correct = total = 0
    if with_metrics:
        with torch.no_grad():
            preds = ctc_greedy(logits.float())
            correct = sum(1 for (p, _), t in zip(preds, texts) if p == t.upper())
            total = len(texts)
    return loss.detach(), correct, total


def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    print(f"device={device}")
    gen = SynthCaptchaGenerator(seed=args.seed)
    stream = SynthStream(gen, args.steps * args.batch_size, img_h=IMG_H)
    loader = DataLoader(stream, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, collate_fn=collate,
                        persistent_workers=args.workers > 0, drop_last=True)
    model = CrnnCaptcha().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=args.steps, pct_start=0.1)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)
    step = 0
    t0 = time.time()
    running_loss = 0.0
    running_correct = running_total = 0
    for batch in loader:
        loss, correct, total = run_batch(model, batch, device, optimizer, scaler,
                                         with_metrics=(step + 1) % 200 == 0)
        scheduler.step()
        running_loss += loss
        running_correct += correct
        running_total += total
        step += 1
        if step % 200 == 0:
            elapsed = time.time() - t0
            print(f"step {step}/{args.steps} loss={running_loss.item() / 200:.3f} "
                  f"synth-acc={running_correct / max(running_total, 1):.4f} "
                  f"({elapsed:.0f}s)", flush=True)
            running_loss, running_correct, running_total = 0.0, 0, 0
        if step % 2000 == 0 or step == args.steps:
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "charset": FOLD_CHARSET,
                    "arch": "crnn",
                    "img_h": IMG_H,
                },
                checkpoint_dir / "universal_crnn.pt",
            )
    print(f"training done in {time.time() - t0:.0f}s -> checkpoints/universal_crnn.pt")


def evaluate_folder(model: CrnnCaptcha, root: Path, device, batch_size=128) -> dict:
    folder = EvalFolder(root, img_h=model.img_h,
                        channels=getattr(model, "in_channels", 3))
    loader = DataLoader(folder, batch_size=batch_size, shuffle=False,
                        num_workers=2, collate_fn=collate)
    model.eval()
    correct = total = 0
    char_correct = char_total = 0
    details = []
    with torch.no_grad():
        for batch in loader:
            images = batch[0].to(device)
            texts = batch[-1]
            logits = model(images)
            preds = ctc_greedy(logits)
            for (p, conf), t in zip(preds, texts):
                total += 1
                ok = p == t
                correct += ok
                char_correct += sum(a == b for a, b in zip(p, t))
                char_total += len(t)
                details.append((t, p, round(conf, 3), ok))
    model.train()
    acc = correct / max(total, 1)
    char_acc = char_correct / max(char_total, 1)
    print(f"[{root}] n={total} full-acc={acc:.4f} char-acc={char_acc:.4f}")
    wrong = [d for d in details if not d[3]][:20]
    for t, p, conf, _ in wrong:
        print(f"  true={t} pred={p} conf={conf}")
    return {"n": total, "full_accuracy": acc, "char_accuracy": char_acc,
            "details": details}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train universal CRNN captcha model.")
    parser.add_argument("--steps", type=int, default=8000)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--lr", type=float, default=1.5e-3)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--evaluate-old", type=Path, default=None,
                        help="also evaluate on old-site labeled folder")
    parser.add_argument("--evaluate-dir", type=Path, default=None,
                        help="evaluate only (no training): labeled folder")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.evaluate_dir:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load("checkpoints/universal_crnn.pt", map_location=device,
                          weights_only=False)
        set_img_h(ckpt.get("img_h", 32))
        model = CrnnCaptcha(img_h=ckpt.get("img_h", 32),
                            lstm_hidden=ckpt.get("lstm_hidden", 192),
                            lstm_layers=ckpt.get("lstm_layers", 2)).to(device)
        model.load_state_dict(ckpt["model_state"])
        evaluate_folder(model, args.evaluate_dir, device)
        return 0

    train(args)
    if args.evaluate_old:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load("checkpoints/universal_crnn.pt", map_location=device,
                          weights_only=False)
        set_img_h(ckpt.get("img_h", 32))
        model = CrnnCaptcha(img_h=ckpt.get("img_h", 32),
                            lstm_hidden=ckpt.get("lstm_hidden", 192),
                            lstm_layers=ckpt.get("lstm_layers", 2)).to(device)
        model.load_state_dict(ckpt["model_state"])
        evaluate_folder(model, args.evaluate_old, device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

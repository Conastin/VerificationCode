"""Fine-tune the universal CRNN on real labeled data (old-site captures) mixed
with the synthetic stream, keeping portal/general coverage intact.

Data conventions supported:
- labels.txt style: "<file> <label>" per line (train31/val31)
- filename-as-label style: <label>.jpg (real_all/real_trusted/real_human)

Usage:
    python -m verification_code.finetune_universal --steps 6000 \
        --real data/train31:10000 data/real_all:10000
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .synth_universal import SynthCaptchaGenerator
from .train_universal import (BLANK, FOLD_CHARSET, IMG_H, CrnnCaptcha, collate,
                              encode_text, preprocess, run_batch, set_img_h)


def load_labeled_folder(root: Path) -> list[tuple[Path, str]]:
    root = Path(root)
    samples: list[tuple[Path, str]] = []
    labels_file = root / "labels.txt"
    if labels_file.is_file():
        for line in labels_file.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) == 2 and all(ch in FOLD_CHARSET for ch in parts[1].upper()):
                samples.append((root / parts[0], parts[1].upper()))
    else:
        for path in sorted(root.glob("*.jpg")):
            label = path.stem.upper()
            if all(ch in FOLD_CHARSET for ch in label):
                samples.append((path, label))
    return samples


class RealCaptchaSet(Dataset):
    def __init__(self, samples: list[tuple[Path, str]], img_h: int | None = None,
                 rng: random.Random | None = None, channels: int = 3) -> None:
        self.samples = samples
        self.img_h = img_h or IMG_H
        self.channels = channels
        self.rng = rng or random.Random(123)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        with Image.open(path) as image:
            arr = preprocess(image.convert("RGB"), self.img_h, self.channels)
        return arr, label


class MixedStream(Dataset):
    """Each index draws real or synthetic with prob `real_prob` (real is
    sampled uniformly with replacement; synthetic is fresh every epoch)."""

    def __init__(self, real_samples, synth: SynthCaptchaGenerator, size: int,
                 real_prob: float, img_h: int | None = None,
                 channels: int = 3) -> None:
        self.real = RealCaptchaSet(real_samples, img_h=img_h, channels=channels)
        self.synth = synth
        self.size = size
        self.real_prob = real_prob
        self.img_h = img_h or IMG_H
        self.channels = channels
        self.offset = 0

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int):
        if self.rng_draw(index) < self.real_prob:
            return self.real[self.rng_index(index)]
        image, text = self.synth.render()
        return preprocess(image, self.img_h, self.channels), text

    # deterministic per-index randomness keeps workers decorrelated
    def rng_draw(self, index: int) -> float:
        return random.Random(index * 7919 + self.offset).random()

    def rng_index(self, index: int) -> int:
        return random.Random(index * 104729 + self.offset).randrange(len(self.real))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fine-tune universal CRNN.")
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("checkpoints/universal_crnn.pt"))
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--lr", type=float, default=4e-4)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--real-prob", type=float, default=0.5,
                        help="fraction of each batch drawn from real data")
    parser.add_argument("--real", nargs="+", default=["data/train31:6000",
                        "data/real_all:6000", "data/real_trusted:1000"],
                        help="labeled folders as path:weight entries")
    parser.add_argument("--evaluate-old", type=Path,
                        default=Path("data/real_test_independent"))
    parser.add_argument("--out", type=Path, default=Path("checkpoints/universal_crnn_ft.pt"))
    parser.add_argument("--lstm-hidden", type=int, default=192,
                        help="capacity experiment: wider LSTM")
    parser.add_argument("--lstm-layers", type=int, default=2,
                        help="capacity experiment: deeper LSTM")
    parser.add_argument("--channels", type=int, default=3, choices=(3, 4),
                        help="4 = RGB + saturation (portal: colored chars vs gray lines)")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    # weighted pool of real samples
    pool: list[tuple[Path, str]] = []
    for entry in args.real:
        root, _, weight = entry.partition(":")
        samples = load_labeled_folder(Path(root))
        take = samples if not weight else samples
        count = int(weight) if weight else len(samples)
        # weight = desired draw count: replicate proportionally
        factor = max(1, round(count / max(len(samples), 1)))
        pool.extend(take * factor)
        print(f"real {root}: {len(samples)} samples x{factor}")
    print(f"real pool size: {len(pool)}")

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    set_img_h(ckpt.get("img_h", 32))
    ckpt_img_h = ckpt.get("img_h", 32)

    synth = SynthCaptchaGenerator(seed=202)
    stream = MixedStream(pool, synth, args.steps * args.batch_size, args.real_prob,
                         img_h=ckpt_img_h, channels=args.channels)
    loader = DataLoader(stream, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, collate_fn=collate,
                        persistent_workers=args.workers > 0, drop_last=True)

    bigger = args.lstm_hidden != 192 or args.lstm_layers != 2 or args.channels != 3
    model = CrnnCaptcha(img_h=ckpt_img_h, lstm_hidden=args.lstm_hidden,
                        lstm_layers=args.lstm_layers,
                        in_channels=args.channels).to(device)
    if bigger:
        # capacity/channels probe: keep only shape-compatible weights
        state = {k: v for k, v in ckpt["model_state"].items()
                 if k in model.state_dict()
                 and model.state_dict()[k].shape == v.shape}
        model.load_state_dict(state, strict=False)
        print(f"warm-start: {len(state)}/{len(ckpt['model_state'])} tensors "
              f"transferred (conv1/rnn/head fresh as needed)")
    else:
        model.load_state_dict(ckpt["model_state"])

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=args.steps, pct_start=0.15)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    step = 0
    for batch in loader:
        loss, correct, total = run_batch(model, batch, device, optimizer, scaler,
                                         with_metrics=(step + 1) % 200 == 0)
        scheduler.step()
        step += 1
        if step % 200 == 0:
            print(f"ft step {step}/{args.steps} loss={loss.item():.3f} "
                  f"batch-acc={correct}/{total}", flush=True)
        if step % 2000 == 0 or step == args.steps:
            torch.save({"model_state": model.state_dict(), "charset": FOLD_CHARSET,
                        "arch": "crnn", "img_h": ckpt_img_h,
                        "lstm_hidden": args.lstm_hidden,
                        "lstm_layers": args.lstm_layers,
                        "channels": args.channels}, args.out)
    print(f"fine-tune done -> {args.out}")

    if args.evaluate_old:
        from .train_universal import evaluate_folder
        evaluate_folder(model, args.evaluate_old, device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

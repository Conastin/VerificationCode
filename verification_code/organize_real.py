"""Organize captured CAPTCHA batches into human-verified and auto-trusted sets.

Batches verified entirely by hand (labels_corrected.txt covers every image)
go into the human set. Batches that were reviewed with --max-conf only (low
confidence reviewed, high confidence trusted) are split at a confidence
threshold: below it -> human set, at/above it -> trusted set (label = model
prediction).

Outputs both sets as directories with labels.txt + labels_corrected.txt
(same content), compatible with finetune_real.py.

Usage:
    python -m verification_code.organize_real \
        --human-dirs data/real_captcha data/real_captcha2 data/real_captcha3 data/real_check \
        --split-dirs data/real_captcha4 --threshold 0.9
"""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Organize real CAPTCHA batches.")
    parser.add_argument("--human-dirs", nargs="+", type=Path, required=True,
                        help="batches where every image was manually verified")
    parser.add_argument("--split-dirs", nargs="*", type=Path, default=[],
                        help="batches reviewed with --max-conf; split by confidence")
    parser.add_argument("--threshold", type=float, default=0.9,
                        help="confidence at/above which labels are trusted without review")
    parser.add_argument("--out-human", type=Path, default=Path("data/real_human"))
    parser.add_argument("--out-trusted", type=Path, default=Path("data/real_trusted"))
    return parser


def read_labels(directory: Path) -> dict[str, str]:
    labels: dict[str, str] = {}
    path = directory / "labels_corrected.txt"
    if not path.is_file():
        raise FileNotFoundError(f"missing {path}")
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            parts = line.split()
            if len(parts) == 2:
                labels[parts[0]] = parts[1]
    return labels


def read_manifest_conf(directory: Path) -> dict[str, float]:
    conf: dict[str, float] = {}
    path = directory / "capture_manifest.csv"
    if not path.is_file():
        raise FileNotFoundError(f"missing {path}")
    with path.open(encoding="utf-8") as handle:
        for row in csv.reader(handle):
            if len(row) >= 4 and row[1] and row[1] != "label":
                try:
                    conf[row[2]] = float(row[3])
                except ValueError:
                    pass
    return conf


def copy_into(target: Path, source: Path, name: str, label: str,
              used: dict[str, bool]) -> None:
    base = Path(name).stem
    new_name = name
    if new_name in used:
        suffix = 2
        while f"{base}_{suffix}.jpg" in used:
            suffix += 1
        new_name = f"{base}_{suffix}.jpg"
    shutil.copy2(source, target / new_name)
    used[new_name] = True
    return new_name


def write_labelset(directory: Path, rows: list[tuple[str, str]]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for name in ("labels.txt", "labels_corrected.txt"):
        with (directory / name).open("w", encoding="utf-8", newline="\n") as handle:
            for filename, label in rows:
                handle.write(f"{filename} {label}\n")


def main() -> int:
    args = build_parser().parse_args()
    # Reset both output directories before rebuilding (idempotent).
    for directory in (args.out_human, args.out_trusted):
        if directory.is_dir():
            for old in directory.glob("*.jpg"):
                old.unlink()
            for old in directory.glob("labels*.txt"):
                old.unlink()
    args.out_human.mkdir(parents=True, exist_ok=True)
    args.out_trusted.mkdir(parents=True, exist_ok=True)
    human_rows: list[tuple[str, str]] = []
    trusted_rows: list[tuple[str, str]] = []
    human_used: dict[str, bool] = {}
    trusted_used: dict[str, bool] = {}

    for directory in args.human_dirs:
        labels = read_labels(directory)
        for name, label in labels.items():
            source = directory / name
            if source.is_file():
                new_name = copy_into(args.out_human, source, name, label, human_used)
                human_rows.append((new_name, label))

    for directory in args.split_dirs:
        labels = read_labels(directory)
        conf = read_manifest_conf(directory)
        for name, label in labels.items():
            source = directory / name
            if not source.is_file():
                continue
            if conf.get(name, 0.0) >= args.threshold:
                new_name = copy_into(args.out_trusted, source, name, label, trusted_used)
                trusted_rows.append((new_name, label))
            else:
                new_name = copy_into(args.out_human, source, name, label, human_used)
                human_rows.append((new_name, label))

    write_labelset(args.out_human, human_rows)
    write_labelset(args.out_trusted, trusted_rows)
    print(f"human-verified: {len(human_rows)} -> {args.out_human}")
    print(f"auto-trusted:   {len(trusted_rows)} -> {args.out_trusted}")
    print(f"total: {len(human_rows) + len(trusted_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

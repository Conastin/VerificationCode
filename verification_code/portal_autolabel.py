"""Server-feedback auto-labeling loop for portal.example.com.

Each iteration: fresh session -> fresh captcha -> model predicts -> submit with
dummy credentials -> server verdict labels the sample:

- captcha OK   ("用户不存在")  -> the prediction is SERVER-CONFIRMED ground
  truth: archived to server_confirmed/<batch>/ (golden data, append-only)
- captcha wrong ("验证码输入错误...") -> hard sample: archived to
  server_failed/<batch>/ with the wrong prediction (candidates for human
  labeling)

Archival discipline (datasets are precious):
- every batch gets a manifest.json (timestamp, checkpoint, counts) and a
  manifest.csv row per attempt (verdict, confidence, sha256)
- images deduped by sha256 across the whole portal_dataset tree
- append-only: existing files are never overwritten

Usage:
    python -m verification_code.portal_autolabel --count 400 --interval 2.2
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import random
import time
from datetime import datetime
from pathlib import Path

import torch
from PIL import Image

from .portal_login_validate import (classify, fetch_captcha, fetch_login_page,
                                    make_opener, recognize, submit_login)
from .train_universal import CrnnCaptcha, set_img_h

ROOT = Path("data/portal_dataset")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Server-feedback auto-labeling.")
    parser.add_argument("--count", type=int, default=400)
    parser.add_argument("--interval", type=float, default=2.2)
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("checkpoints/universal_crnn_ft_portal.pt"))
    parser.add_argument("--user", default="testuser01")
    parser.add_argument("--password", default="Test123456!")
    parser.add_argument("--seed", type=int, default=None)
    return parser


def known_hashes(root: Path) -> set[str]:
    """sha256 of every jpg already archived anywhere under root."""
    hashes = set()
    for path in root.rglob("*.jpg"):
        try:
            hashes.add(hashlib.sha256(path.read_bytes()).hexdigest())
        except OSError:
            continue
    return hashes


def main() -> int:
    args = build_parser().parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    set_img_h(ckpt.get("img_h", 32))
    model = CrnnCaptcha(img_h=ckpt.get("img_h", 32)).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    batch_name = datetime.now().strftime("batch_%Y%m%d_%H%M%S")
    confirmed_dir = ROOT / "server_confirmed" / batch_name
    failed_dir = ROOT / "server_failed" / batch_name
    confirmed_dir.mkdir(parents=True, exist_ok=True)
    failed_dir.mkdir(parents=True, exist_ok=True)

    seen = known_hashes(ROOT)
    rng = random.Random(args.seed)

    manifest = {
        "batch": batch_name,
        "started": datetime.now().isoformat(timespec="seconds"),
        "checkpoint": str(args.checkpoint),
        "attempts": args.count,
        "interval_s": args.interval,
    }
    (confirmed_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (failed_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    confirmed_handle = (confirmed_dir / "labels.txt").open("w", encoding="utf-8")
    failed_handle = (failed_dir / "failed.csv").open("w", encoding="utf-8", newline="")
    failed_csv = csv.writer(failed_handle)
    failed_csv.writerow(["file", "model_pred", "confidence", "verdict", "hint"])
    log_handle = (ROOT / f"autolabel_{batch_name}.csv").open("w", encoding="utf-8",
                                                             newline="")
    log_csv = csv.writer(log_handle)
    log_csv.writerow(["attempt", "verdict", "pred", "conf", "sha256",
                      "confirmed_file", "failed_file"])

    stats = {"confirmed": 0, "failed": 0, "dup": 0, "error": 0, "unknown": 0}
    for attempt in range(1, args.count + 1):
        try:
            opener = make_opener()
            page = fetch_login_page(opener)
            if not page["lt"]:
                raise RuntimeError("lt ticket missing")
            data = fetch_captcha(opener, page["lt"])
            pred, conf = recognize(model, data, model.img_h, device)
            status, resp = submit_login(opener, page, args.user, args.password, pred)
            verdict, hint = classify(resp)

            digest = hashlib.sha256(data).hexdigest()
            dup = digest in seen
            seen.add(digest)

            c_file = f_file = ""
            if verdict == "captcha_ok":
                stats["confirmed"] += 1
                c_file = f"{pred}_{attempt:04d}.jpg"
                (confirmed_dir / c_file).write_bytes(data)
                confirmed_handle.write(f"{c_file} {pred}\n")
                confirmed_handle.flush()
                if dup:
                    stats["dup"] += 1
            elif verdict == "captcha_wrong":
                stats["failed"] += 1
                f_file = f"fail_{attempt:04d}.jpg"
                (failed_dir / f_file).write_bytes(data)
                failed_csv.writerow([f_file, pred, round(conf, 4), verdict,
                                     hint[:80]])
            else:
                stats["unknown"] += 1
                (failed_dir / f"unknown_{attempt:04d}.jpg").write_bytes(data)
                failed_csv.writerow([f"unknown_{attempt:04d}.jpg", pred,
                                     round(conf, 4), verdict, hint[:80]])
            log_csv.writerow([attempt, verdict, pred, round(conf, 4), digest[:16],
                              c_file, f_file])
            mark = {"captcha_ok": "OK ", "captcha_wrong": "X  "}.get(verdict, "?  ")
            print(f"{attempt:4d}/{args.count} {mark} {pred} conf={conf:.2f} {verdict}"
                  f"{' (dup)' if dup and verdict == 'captcha_ok' else ''}", flush=True)
        except Exception as error:  # noqa: BLE001 - keep the loop alive
            stats["error"] += 1
            print(f"{attempt:4d}/{args.count} !! {error}", flush=True)
            log_csv.writerow([attempt, "error", "", "", "", "", ""])
        time.sleep(args.interval + rng.uniform(0, 0.6))

    confirmed_handle.close()
    failed_handle.close()
    log_handle.close()
    judged = stats["confirmed"] + stats["failed"]
    print(f"\n确认(金标准) {stats['confirmed']} | 拒绝(难例) {stats['failed']} | "
          f"未知 {stats['unknown']} | 异常 {stats['error']} | 重复图 {stats['dup']}")
    if judged:
        print(f"单次识别率: {stats['confirmed'] / judged:.1%}")
    print(f"金标准: {confirmed_dir}\n难例:   {failed_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

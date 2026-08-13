"""Evaluate a multimodal LLM (OpenAI-compatible API) on labeled real CAPTCHAs.

Uses the 1000 manually-corrected samples in data/real_all as ground truth to
measure how well a VLM reads the CAPTCHAs, and optionally compares against
the fine-tuned CNN baseline.

Credentials: the API key is read from the OPENAI_API_KEY environment
variable (set it yourself, do not paste keys into chat). Base URL and model
come from --base-url / --model or OPENAI_BASE_URL.

Usage:
    set OPENAI_API_KEY=...
    python -m verification_code.vlm_eval --sample 50 --model gpt-4o-mini
"""

from __future__ import annotations

import argparse
import base64
import csv
import os
import re
import time
from pathlib import Path

from openai import OpenAI

from .finetune_real import RealCaptchaDataset

PROMPT = (
    "这是一张验证码图片，其中包含4个字符（大写字母 A-Z 或数字 0-9）。"
    "请只输出这4个字符本身，不要输出任何其他内容、解释或标点。"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="VLM CAPTCHA recognition evaluation.")
    parser.add_argument("--data", type=Path, default=Path("data/real_all"))
    parser.add_argument("--sample", type=int, default=50, help="number of images to test (0 = all)")
    parser.add_argument("--offset", type=int, default=0, help="skip this many images first")
    parser.add_argument("--model", default=os.environ.get("VLM_MODEL", "gpt-4o-mini"))
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", None))
    parser.add_argument("--out", type=Path, default=Path("reports/vlm_eval.csv"))
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--retries", type=int, default=3)
    return parser


def load_api_key() -> str:
    """Read the API key from the environment or the protected local file."""
    key = os.environ.get("OPENAI_API_KEY")
    if key:
        return key
    key_file = Path.home() / ".captcha_vlm_key"
    if key_file.is_file():
        return key_file.read_text(encoding="utf-8").strip()
    raise SystemExit("no API key found; set OPENAI_API_KEY or write it to ~/.captcha_vlm_key")


def clean_output(text: str) -> str:
    """Keep only the 4 alphanumeric characters the model should have printed."""
    cleaned = re.sub(r"[^A-Za-z0-9]", "", text or "").upper()
    return cleaned[:4]


def recognize(client: OpenAI, model: str, image_bytes: bytes, max_tokens: int,
              retries: int) -> str:
    encoded = base64.b64encode(image_bytes).decode("ascii")
    url = f"data:image/jpeg;base64,{encoded}"
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": PROMPT},
                            {"type": "image_url", "image_url": {"url": url}},
                        ],
                    }
                ],
                max_tokens=max_tokens,
                temperature=0,
            )
            return clean_output(response.choices[0].message.content)
        except Exception as error:  # noqa: BLE001 - retry any transient failure
            if attempt == retries - 1:
                print(f"  error after {retries} attempts: {error}")
                return ""
            time.sleep(2 * (attempt + 1))
    return ""


def main() -> int:
    args = build_parser().parse_args()
    client = OpenAI(api_key=load_api_key(), base_url=args.base_url)

    dataset = RealCaptchaDataset(args.data, augment=False)
    total = len(dataset)
    indices = list(range(args.offset, min(args.offset + args.sample, total) if args.sample else total))
    print(f"model={args.model} testing {len(indices)}/{total} samples")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    full_correct = char_correct = 0
    start_time = time.time()
    with args.out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["file", "true", "pred", "full_ok", "char_ok"])
        for index in indices:
            image, label_ids = dataset[index]
            charset = dataset.charset
            true_label = "".join(charset[i] for i in label_ids.tolist())
            image_path = dataset.items[index][0]
            pred = recognize(client, args.model, image_path.read_bytes(),
                             args.max_tokens, args.retries)
            full_ok = pred == true_label
            char_ok = sum(1 for a, b in zip(pred, true_label) if a == b)
            full_correct += full_ok
            char_correct += char_ok
            writer.writerow([image_path.name, true_label, pred, int(full_ok), char_ok])
            print(f"{index:4d} true={true_label} pred={pred or '?'} {'OK' if full_ok else 'X'}")

    elapsed = time.time() - start_time
    n = len(indices)
    print(f"\nVLM {args.model}: {n} 张 | 全对 {full_correct}/{n} ({full_correct / n:.1%}) "
          f"| 字符级 {char_correct / (n * 4):.1%} | 用时 {elapsed:.0f}s ({elapsed / n:.2f}s/张)")
    print(f"detail: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

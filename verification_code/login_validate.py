"""Real login-validation of the CAPTCHA recognizer against the target site.

Each iteration: new session -> fetch CAPTCHA -> recognize -> submit login
with a test account. The server response discriminates the result:

- "验证码不正确" in the response -> recognition was WRONG (captcha rejected)
- otherwise -> captcha passed (username/password check ran instead)

Runs at a conservative rate (2s between attempts) and only performs login
attempts with the provided test account for captcha validation purposes.

Usage:
    python -m verification_code.login_validate --user MES02 --password xxx \
        --attempts 50 --checkpoint checkpoints/best.pt
"""

import os
from __future__ import annotations

import argparse
import csv
import http.cookiejar
import io
import re
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .model import build_model_from_checkpoint, decode_logits

BASE = os.environ.get("OLD_SITE_BASE", "https://old.example.internal")
# Dynamic rejection markers only; page-fixed labels like "图形验证码" (input
# placeholder) and "获取图形验证码" must NOT be used as rejection signals.
REJECT_MARKERS = ("验证码不正确", "验证码错误，请重新输入", "验证码已失效")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate recognizer via real login checks.")
    parser.add_argument("--user", required=True)
    parser.add_argument("--password", required=True)
    parser.add_argument("--attempts", type=int, default=50)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/best.pt"))
    parser.add_argument("--interval", type=float, default=2.0, help="seconds between attempts")
    parser.add_argument("--out", type=Path, default=Path("reports/login_validation.csv"))
    return parser


def new_session() -> urllib.request.OpenerDirector:
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    jar = http.cookiejar.CookieJar()
    opener = urllib.request.build_opener(
        urllib.request.HTTPCookieProcessor(jar),
        urllib.request.HTTPSHandler(context=ctx),
    )
    opener.addheaders = [("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0")]
    opener.open(f"{BASE}/fort/pages/login.jsp", timeout=20).read()
    return opener


def fetch_captcha(opener: urllib.request.OpenerDirector) -> bytes:
    url = f"{BASE}/fort/pages/commons/image.jsp?seed=" + str(int(time.time() * 1000))
    return opener.open(url, timeout=20).read()


def recognize(model: torch.nn.Module, charset: str, image_bytes: bytes) -> tuple[str, float]:
    with Image.open(io.BytesIO(image_bytes)) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    x = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
    with torch.no_grad():
        label, conf = decode_logits(model(x), charset)
    return label[0], conf[0].min().item()


def submit_login(opener: urllib.request.OpenerDirector, user: str, password: str,
                 code: str) -> tuple[int, str]:
    form = {
        "li": "", "code": "", "finger": "testfinger", "ca_login": "0",
        "empUSBCode": "", "empTaxusbkeySN": "", "rad-authType": "0",
        "loginName": user, "password": password,
        "imgCode": code, "para": "1",
    }
    request = urllib.request.Request(
        f"{BASE}/fort/login/check.action",
        data=urllib.parse.urlencode(form).encode(),
    )
    request.add_header("Referer", f"{BASE}/fort/pages/login.jsp")
    try:
        response = opener.open(request, timeout=20)
        return response.status, response.read().decode("utf-8", "ignore")
    except urllib.error.HTTPError as error:
        return error.code, error.read().decode("utf-8", "ignore")


def classify(status: int, response: str) -> tuple[str, str]:
    """Return (verdict, evidence) based on server-side dynamic hints.

    - "验证码不正确" -> captcha_wrong
    - credential-check hints (用户名或密码错误 etc.) -> captcha_ok
    - login-success redirect (frame.action) -> captcha_ok
    """
    text = re.sub(r"<[^>]+>", " ", response)
    text = re.sub(r"\s+", " ", text).strip()

    def evidence(keywords: tuple[str, ...]) -> str | None:
        for keyword in keywords:
            pos = text.find(keyword)
            if pos >= 0:
                return text[max(0, pos - 60):pos + 80]
        return None

    reject = evidence(("验证码不正确", "验证码错误，请重新输入", "验证码已失效"))
    if reject:
        return "captcha_wrong", reject
    cred = evidence(("用户名或密码错误", "密码错误", "用户名错误", "账号或密码", "口令错误", "密码不正确"))
    if cred:
        return "captcha_ok", cred
    success = evidence(("frame.action", "top.location", "window.location.href='/fort/frame"))
    if success:
        return "captcha_ok", success
    return "captcha_ok", text[:250]


def main() -> int:
    args = build_parser().parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    charset = checkpoint.get("charset")
    model = build_model_from_checkpoint(checkpoint, charset=charset)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    results = []
    with args.out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["attempt", "predicted", "confidence", "http", "verdict", "hint"])
        for attempt in range(1, args.attempts + 1):
            try:
                opener = new_session()
                image = fetch_captcha(opener)
                predicted, conf = recognize(model, charset, image)
                status, response = submit_login(opener, args.user, args.password, predicted)
                verdict, hint = classify(status, response)
                writer.writerow([attempt, predicted, round(conf, 4), status, verdict, hint[:200]])
                results.append(verdict)
                mark = "OK " if verdict == "captcha_ok" else "X  "
                print(f"{attempt:3d}/{args.attempts} {mark} {predicted} conf={conf:.3f} http={status} {verdict}")
            except Exception as error:  # noqa: BLE001 - keep going on transient failures
                print(f"{attempt:3d}/{args.attempts} !! error: {error}")
                results.append("error")
            time.sleep(args.interval)

    ok = results.count("captcha_ok")
    wrong = results.count("captcha_wrong")
    errors = results.count("error")
    print(f"\n总尝试 {len(results)} | 验证码通过 {ok} | 验证码错误 {wrong} | 异常 {errors}")
    if ok + wrong:
        print(f"识别准确率: {ok / (ok + wrong):.2%}  ({ok}/{ok + wrong})")
    print(f"明细: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

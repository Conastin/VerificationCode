"""Real login-validation of the universal CRNN against portal.example.com.

Server semantics (user-confirmed): any non-empty username/password works for
captcha validation. POST /login response discriminates:
- "验证码输入错误" family  -> captcha WRONG
- account/password error   -> captcha PASSED (uppercase submission also tests
  the case-insensitivity assumption)

Each attempt uses a fresh session (new lt ticket + fresh captcha bound to it).

Usage:
    python -m verification_code.portal_login_validate \
        --checkpoint checkpoints/universal_crnn_ft_portal.pt --attempts 60
"""

from __future__ import annotations

import argparse
import csv
import http.cookiejar
import io
import re
import ssl
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .train_universal import (CrnnCaptcha, ctc_greedy, preprocess, set_img_h)

BASE = "https://portal.example.com"
# dynamic markers only; page-fixed JS template strings (e.g. 人脸认证失败)
# must NOT be used - the verdict comes exclusively from the #msg div that the
# server injects into the POST /login response HTML.
CAPTCHA_WRONG = ("验证码输入错误", "验证码不正确", "验证码错误", "请您重新输入")
CAPTCHA_OK_HINTS = ("用户不存在", "用户名或密码", "密码错误", "账号或密码", "账户密码错误",
                    "口令错误", "用户名不存在", "账号不存在", "锁定", "冻结", "认证类型", "扫码")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Portal login validation.")
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("checkpoints/universal_crnn_ft_portal.pt"))
    parser.add_argument("--attempts", type=int, default=60)
    parser.add_argument("--interval", type=float, default=2.5)
    parser.add_argument("--user", default="testuser01")
    parser.add_argument("--password", default="Test123456!")
    parser.add_argument("--fold-case", action="store_true", default=False,
                        help="submit predictions uppercased (default: as-is)")
    parser.add_argument("--out", type=Path, default=Path("reports/portal_login_validation.csv"))
    return parser


def make_opener() -> urllib.request.OpenerDirector:
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    jar = http.cookiejar.CookieJar()
    opener = urllib.request.build_opener(
        urllib.request.HTTPCookieProcessor(jar),
        urllib.request.HTTPSHandler(context=ctx),
    )
    opener.addheaders = [("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Edge/120")]
    return opener


def fetch_login_page(opener) -> dict[str, str]:
    html = opener.open(f"{BASE}/login", timeout=20).read().decode("utf-8", "ignore")

    def grab(pattern: str) -> str:
        m = re.search(pattern, html)
        return m.group(1) if m else ""

    return {
        "lt": grab(r'name="lt"[^>]*value="([^"]+)"'),
        "csrfToken": grab(r'name="csrfToken"[^>]*value="([^"]+)"'),
        "execution": grab(r'name="execution"[^>]*value="([^"]+)"'),
    }


def fetch_captcha(opener, lt: str) -> bytes:
    url = f"{BASE}/image/getRandcode/{lt}_KEY?d={time.time() * 1000:.0f}"
    return opener.open(url, timeout=20).read()


def submit_login(opener, page: dict[str, str], user: str, password: str,
                 code: str) -> tuple[int, str]:
    form = {
        "eid": "esc",
        "isShowRandomCode": "1",
        "keyCacheCode": f"{page['lt']}_KEY",
        "lt": page["lt"],
        "execution": page["execution"],
        "_eventId": "submit",
        "authType": "pwd",
        "cert": "",
        "csrfToken": page["csrfToken"],
        "username": user,
        "password": password,
        "randomCode": code,
    }
    request = urllib.request.Request(
        f"{BASE}/login", data=urllib.parse.urlencode(form).encode(),
        headers={"Referer": f"{BASE}/login",
                 "Accept-Language": "zh-CN,zh;q=0.9",
                 "Content-Type": "application/x-www-form-urlencoded"})
    try:
        response = opener.open(request, timeout=20)
        raw = response.read()
    except urllib.error.HTTPError as error:
        return error.code, error.read().decode("utf-8", "ignore")
    text = raw.decode("utf-8", "ignore")
    if "id=\"msg\"" not in text and "id='msg'" not in text:
        # server may negotiate GBK when the client lacks Accept-Language
        text = raw.decode("gbk", "ignore")
    return response.status, text


def classify(response: str) -> tuple[str, str]:
    """Verdict from the server-injected <div id="msg"> element only."""
    m = re.search(r"<div id=\"msg\"[^>]*>([^<]*)</div>", response)
    if not m:
        return "unknown", "no #msg element in response"
    text = m.group(1).strip()
    if any(k in text for k in CAPTCHA_WRONG):
        return "captcha_wrong", text
    if any(k in text for k in CAPTCHA_OK_HINTS):
        return "captcha_ok", text
    return "unknown", text


def recognize(model, data: bytes, img_h: int, device) -> tuple[str, float]:
    with Image.open(io.BytesIO(data)) as image:
        arr = preprocess(image.convert("RGB"), img_h, getattr(model, "in_channels", 3))
    x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.no_grad():
        (pred, conf), = ctc_greedy(model(x))
    return pred, conf


def main() -> int:
    args = build_parser().parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    set_img_h(ckpt.get("img_h", 32))
    model = CrnnCaptcha(img_h=ckpt.get("img_h", 32),
                        lstm_hidden=ckpt.get("lstm_hidden", 192),
                        lstm_layers=ckpt.get("lstm_layers", 2),
                        in_channels=ckpt.get("channels", 3)).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    results = []
    with args.out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["attempt", "predicted", "confidence", "verdict", "hint"])
        for attempt in range(1, args.attempts + 1):
            try:
                opener = make_opener()
                page = fetch_login_page(opener)
                if not page["lt"]:
                    raise RuntimeError("lt ticket not found in login page")
                image = fetch_captcha(opener, page["lt"])
                predicted, conf = recognize(model, image, model.img_h, device)
                submit = predicted.upper() if args.fold_case else predicted
                status, response = submit_login(opener, page, args.user, args.password, submit)
                verdict, hint = classify(response)
                writer.writerow([attempt, predicted, round(conf, 4), verdict, hint[:200]])
                results.append(verdict)
                mark = "OK " if verdict == "captcha_ok" else ("X  " if verdict == "captcha_wrong" else "?  ")
                print(f"{attempt:3d}/{args.attempts} {mark} {predicted} conf={conf:.2f} "
                      f"http={status} {verdict} | {hint[:80]}")
            except Exception as error:  # noqa: BLE001 - keep going on transient failures
                print(f"{attempt:3d}/{args.attempts} !! error: {error}")
                results.append("error")
            time.sleep(args.interval)

    ok = results.count("captcha_ok")
    wrong = results.count("captcha_wrong")
    unknown = results.count("unknown")
    errors = results.count("error")
    n = ok + wrong
    print(f"\n总尝试 {len(results)} | 验证码通过 {ok} | 验证码错误 {wrong} | "
          f"无法判别 {unknown} | 异常 {errors}")
    if n:
        print(f"识别准确率(通过/可判别): {ok / n:.1%}  ({ok}/{n})")
    print(f"明细: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

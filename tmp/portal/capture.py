# -*- coding: utf-8 -*-
"""采集 portal.example.com 验证码样本（纯 HTTP）。

流程: GET /login 提取 lt ticket -> 同 KEY 加 cache-buster 反复取新图。
数据落到 data/portal_raw/（.gitignore data/* 已忽略，真实样本不入库）。
"""
import re
import ssl
import sys
import time
import urllib.request
from pathlib import Path

OUT = Path(__file__).resolve().parents[2] / "data" / "portal_raw"
BASE = "https://portal.example.com"
COUNT = int(sys.argv[1]) if len(sys.argv) > 1 else 150


def main() -> int:
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    opener = urllib.request.build_opener(
        urllib.request.HTTPSHandler(context=ctx),
        urllib.request.HTTPCookieProcessor(),
    )
    opener.addheaders = [("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Edge/120")]

    html = opener.open(f"{BASE}/login", timeout=20).read().decode("utf-8", "ignore")
    m = re.search(r'name="lt"[^>]*value="([^"]+)"', html)
    if not m:
        print("lt not found")
        return 1
    lt = m.group(1)
    print(f"lt={lt}")

    OUT.mkdir(parents=True, exist_ok=True)
    ok = 0
    for i in range(COUNT):
        url = f"{BASE}/image/getRandcode/{lt}_KEY?d={time.time() * 1000:.0f}"
        try:
            raw = opener.open(url, timeout=20).read()
        except OSError as e:
            print(f"{i}: fetch failed: {e}")
            time.sleep(1.0)
            continue
        if len(raw) < 200 or not raw[:2] == b"\xff\xd8":
            print(f"{i}: unexpected payload len={len(raw)}")
            time.sleep(1.0)
            continue
        (OUT / f"p_{i:04d}.jpg").write_bytes(raw)
        ok += 1
        time.sleep(0.3)
    print(f"saved {ok}/{COUNT} -> {OUT}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

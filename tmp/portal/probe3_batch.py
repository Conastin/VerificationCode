# -*- coding: utf-8 -*-
"""Probe 3: 验证批量采集可行性。

A. 同一会话内同一 KEY 加 cache-buster 连续 GET 5 次 -> 字节是否不同(是否换新图)
B. 同一浏览器连续开两次 /login -> lt 是否变化
C. 纯 HTTP(无浏览器)复刻: requests 风格, 用 urllib 带 JSESSIONID + lt 直接 GET,
   验证脱离浏览器也能采集
"""
import hashlib
import json
import re
import ssl
import sys
import time
import urllib.request
from pathlib import Path

from playwright.sync_api import sync_playwright

OUT = Path(__file__).parent
URL = "https://portal.example.com/login"


def md5(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()[:12]


def main() -> int:
    results = {}

    with sync_playwright() as p:
        browser = p.chromium.launch(channel="msedge", headless=True)
        context = browser.new_context(ignore_https_errors=True, locale="zh-CN")
        page = context.new_page()

        # ---- B: 两次加载 /login, 比较 lt ----
        page.goto(URL, wait_until="domcontentloaded", timeout=60000)
        page.wait_for_timeout(1500)
        lt1 = page.evaluate("() => document.querySelector('input[name=lt]').value")
        page.goto(URL, wait_until="domcontentloaded", timeout=60000)
        page.wait_for_timeout(1500)
        lt2 = page.evaluate("() => document.querySelector('input[name=lt]').value")
        results["lt_first"] = lt1
        results["lt_second"] = lt2
        results["lt_changes_per_pageload"] = lt1 != lt2
        print(f"B) lt#1={lt1}")
        print(f"B) lt#2={lt2}  changed={lt1 != lt2}")

        # ---- A: 同一 KEY 连续刷新 5 次 ----
        key = page.evaluate("() => document.querySelector('#keyCacheCode').value")
        hashes = []
        for i in range(5):
            data = page.evaluate(
                """async (key) => {
                  const res = await fetch('/image/getRandcode/' + key + '?d=' + Math.random(),
                                          {credentials: 'include'});
                  const blob = await res.blob();
                  const buf = await blob.arrayBuffer();
                  const bytes = new Uint8Array(buf);
                  let bin = '';
                  for (let j = 0; j < bytes.length; j += 8192) {
                    bin += String.fromCharCode.apply(null, bytes.subarray(j, j + 8192));
                  }
                  return btoa(bin);
                }""",
                key,
            )
            import base64

            raw = base64.b64decode(data)
            name = f"refresh_{i}.jpg"
            (OUT / name).write_bytes(raw)
            h = md5(raw)
            hashes.append(h)
            print(f"A) refresh#{i} size={len(raw)} md5={h} -> {name}")
        results["same_key_refresh_unique_hashes"] = sorted(set(hashes)) != [hashes[0]]
        results["same_key_refresh_hashes"] = hashes
        print(f"A) 同 KEY 刷新产生不同图片: {results['same_key_refresh_unique_hashes']}")

        cookies = {c["name"]: c["value"] for c in context.cookies()}
        results["cookies"] = cookies
        results["key"] = key
        browser.close()

    # ---- C: 纯 HTTP 复刻(无浏览器) ----
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    opener = urllib.request.build_opener(
        urllib.request.HTTPSHandler(context=ctx),
        urllib.request.HTTPCookieProcessor(),
    )
    opener.addheaders = [("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Edge/120")]
    html = opener.open(URL, timeout=20).read().decode("utf-8", "ignore")
    m = re.search(r'name="lt"[^>]*value="([^"]+)"', html)
    if not m:
        print("C) FAILED: lt not found in HTML")
    else:
        lt = m.group(1)
        print(f"C) pure-HTTP lt={lt}")
        for i in range(3):
            u = f"https://portal.example.com/image/getRandcode/{lt}_KEY?d={time.time()}"
            raw = opener.open(u, timeout=20).read()
            (OUT / f"purehttp_{i}.jpg").write_bytes(raw)
            print(f"C) pure-HTTP GET#{i} size={len(raw)} md5={md5(raw)}")
        results["pure_http_ok"] = True

    (OUT / "20_batch_feasibility.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

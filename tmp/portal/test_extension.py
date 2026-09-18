# -*- coding: utf-8 -*-
"""扩展 v3.0.0 E2E 实测: persistent-context Edge + 扩展, 双站自动填充验证."""
import shutil
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

EXT_DIR = str(Path(__file__).resolve().parents[2] / "browser" / "extension")
PROFILE = Path(__file__).parent / "edge_profile"

PORTAL_CFG = {
    "mode": "image",
    "imgSelector": "#codeImg",
    "inputSelector": "#randomCode",
    "refreshSelector": "#codeImg",
}
OLD_CFG = {
    "mode": "image",
    "imgSelector": "img[src*='image.jsp']",
    "inputSelector": "input[name=imgCode]",
    "refreshSelector": "img[src*='image.jsp']",
}


def main() -> int:
    if PROFILE.exists():
        shutil.rmtree(PROFILE)
    with sync_playwright() as p:
        context = p.chromium.launch_persistent_context(
            str(PROFILE), channel="msedge", headless=True,
            ignore_https_errors=True, locale="zh-CN",
            args=[f"--disable-extensions-except={EXT_DIR}",
                  f"--load-extension={EXT_DIR}", "--headless=new"],
        )
        ext_id = None
        warm = context.new_page()
        warm.goto("about:blank")
        for _ in range(40):
            for worker in context.service_workers:
                if worker.url.startswith("chrome-extension://"):
                    ext_id = worker.url.split("/")[2]
                    break
            if ext_id:
                break
            warm.reload()
            time.sleep(0.5)
        warm.close()
        print("extension id:", ext_id)
        assert ext_id, "extension service worker not found"

        # 注入两站配置
        ext_page = context.new_page()
        ext_page.goto(f"chrome-extension://{ext_id}/popup.html",
                      wait_until="domcontentloaded")
        ext_page.evaluate(
            """([origin, cfg]) => chrome.storage.local.set({ [origin]: cfg })""",
            ["https://portal.example.com", PORTAL_CFG])
        ext_page.evaluate(
            """([origin, cfg]) => chrome.storage.local.set({ [origin]: cfg })""",
            ["https://old.example.internal", OLD_CFG])
        ext_page.close()

        results = {}
        # portal
        page = context.new_page()
        page.goto("https://portal.example.com/login",
                  wait_until="domcontentloaded", timeout=60000)
        page.wait_for_timeout(2500)
        page.fill("#username_show", "testuser01")
        page.click("#password_show")
        page.wait_for_timeout(2500)
        filled = ""
        for _ in range(45):
            el = page.query_selector("#randomCode")
            filled = page.eval_on_selector("#randomCode", "e => e.value") if el else ""
            if len(filled) == 4:
                break
            page.wait_for_timeout(1000)
        results["portal"] = filled
        print(f"[portal] 填充值: {filled!r}")
        page.close()

        # old site
        page = context.new_page()
        page.goto("https://old.example.internal/fort/pages/login.jsp",
                  wait_until="domcontentloaded", timeout=60000)
        page.wait_for_timeout(2500)
        filled = ""
        for _ in range(45):
            el = page.query_selector("input[name=imgCode]")
            filled = page.eval_on_selector("input[name=imgCode]", "e => e.value") if el else ""
            if len(filled) == 4:
                break
            page.wait_for_timeout(1000)
        results["old"] = filled
        print(f"[old] 填充值: {filled!r}")
        page.close()
        context.close()

    ok = all(len(v) == 4 for v in results.values())
    print("\nE2E 结果:", "PASS" if ok else "FAIL", results)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

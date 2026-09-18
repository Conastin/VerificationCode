# -*- coding: utf-8 -*-
"""Probe portal.example.com/login: 触发验证码加载并抓取页面/网络信息。

Step 1: 打开登录页，截图 + dump 表单 DOM + 记录网络请求
Step 2: 输入账号 -> 点击密码框 -> 等待验证码出现 -> 截图 + 找验证码 img/背景图
"""
import json
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

OUT = Path(__file__).parent
URL = "https://portal.example.com/login"

network_log = []


def main() -> int:
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="msedge", headless=True)
        context = browser.new_context(
            ignore_https_errors=True,
            viewport={"width": 1440, "height": 900},
            locale="zh-CN",
        )
        page = context.new_page()

        def on_response(resp):
            try:
                req = resp.request
                network_log.append(
                    {
                        "method": req.method,
                        "url": req.url,
                        "status": resp.status,
                        "type": req.resource_type,
                    }
                )
            except Exception:
                pass

        page.on("response", on_response)

        page.goto(URL, wait_until="domcontentloaded", timeout=60000)
        page.wait_for_timeout(5000)
        page.screenshot(path=str(OUT / "01_login_initial.png"), full_page=True)

        # dump 登录表单区域的可交互元素
        info = page.evaluate(
            """() => {
              const pick = (el) => ({
                tag: el.tagName.toLowerCase(),
                type: el.getAttribute('type'),
                placeholder: el.getAttribute('placeholder'),
                name: el.getAttribute('name'),
                id: el.id,
                cls: el.className,
                visible: !!(el.offsetWidth || el.offsetHeight),
              });
              const els = [...document.querySelectorAll('input,button,img,canvas,[role=button]')];
              return els.slice(0, 80).map(pick);
            }"""
        )
        (OUT / "02_dom_inputs.json").write_text(
            json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print("== inputs/buttons/images ==")
        for item in info:
            if item.get("visible"):
                print(json.dumps(item, ensure_ascii=False))

        browser.close()

    (OUT / "03_network_initial.json").write_text(
        json.dumps(network_log, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nnetwork requests captured: {len(network_log)}")
    for item in network_log:
        print(item["status"], item["type"], item["url"][:160])
    return 0


if __name__ == "__main__":
    sys.exit(main())

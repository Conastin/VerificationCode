# -*- coding: utf-8 -*-
"""Playwright: 错验证码提交, 抓 DOM 消息 + 网络流, 定位服务器反馈信号。"""
import json
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

OUT = Path(__file__).parent

def main() -> int:
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="msedge", headless=True)
        context = browser.new_context(ignore_https_errors=True, locale="zh-CN",
                                      viewport={"width": 1440, "height": 900})
        page = context.new_page()
        net = []
        page.on("response", lambda r: net.append({"method": r.request.method, "url": r.url,
                                                  "status": r.status, "type": r.request.resource_type}))
        page.goto("https://portal.example.com/login", wait_until="domcontentloaded", timeout=60000)
        page.wait_for_timeout(3000)
        page.fill("#username_show", "testuser01")
        page.click("#password_show")
        page.fill("#password_show", "Test123456!")
        page.wait_for_timeout(2000)  # 验证码加载
        page.fill("#randomCode", "ZZZZ")
        page.click("#btn_login")
        page.wait_for_timeout(5000)
        page.screenshot(path=str(OUT / "30_wrong_captcha.png"))
        # DOM 消息
        msg = page.evaluate("""() => {
          const out = [];
          for (const sel of ['#errMsg', '#msg', '.error', '.messager-window', '.window-body',
                             '.tooltip', '.alert', '#btn_login ~ div', '.validateMsg']) {
            document.querySelectorAll(sel).forEach(el => {
              const t = (el.innerText || '').trim();
              if (t) out.push({sel, text: t.slice(0, 120)});
            });
          }
          // 兜底: body 上任何包含关键词的可见文本
          const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
          const seen = new Set();
          while (walker.nextNode()) {
            const t = walker.currentNode.textContent.trim();
            if (t && /验证码|密码|重新输入|错误/.test(t) && !seen.has(t)) {
              seen.add(t);
              if (seen.size < 10) out.push({sel: 'text', text: t.slice(0, 120)});
            }
          }
          return out;
        }""")
        print("DOM messages:", json.dumps(msg, ensure_ascii=False, indent=1))
        (OUT / "31_login_net.json").write_text(json.dumps(net, ensure_ascii=False, indent=1),
                                               encoding="utf-8")
        print("network after click:")
        for item in net:
            if item["type"] in ("xhr", "fetch", "document") or "login" in item["url"]:
                print(" ", item["status"], item["method"], item["type"], item["url"][:130])
        browser.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())

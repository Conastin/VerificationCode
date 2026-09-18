# -*- coding: utf-8 -*-
"""Probe 2: 输入账号 -> 点击密码框 -> 捕获验证码出现后的 DOM/网络/图片字节。"""
import json
import sys
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
                body_len = -1
                ct = resp.headers.get("content-type", "")
                if "image" in ct:
                    try:
                        body_len = len(resp.body())
                    except Exception:
                        pass
                network_log.append(
                    {
                        "method": req.method,
                        "url": req.url,
                        "status": resp.status,
                        "type": req.resource_type,
                        "content_type": ct,
                        "body_len": body_len,
                    }
                )
            except Exception:
                pass

        page.on("response", on_response)

        page.goto(URL, wait_until="domcontentloaded", timeout=60000)
        page.wait_for_timeout(3000)

        # ---- 交互: 输入账号, 点击密码框 ----
        page.fill("#username_show", "MES02")
        page.click("#password_show")
        page.wait_for_timeout(4000)  # 等待验证码加载
        page.screenshot(path=str(OUT / "10_after_click_password.png"), full_page=True)

        # ---- 找到验证码相关元素 ----
        captcha_info = page.evaluate(
            """() => {
              const out = {imgs: [], hidden: [], randcodeInHtml: null};
              document.querySelectorAll('img').forEach(im => {
                const r = im.getBoundingClientRect();
                out.imgs.push({
                  src: im.src, cls: im.className, id: im.id,
                  visible: !!(r.width || r.height), w: im.naturalWidth, h: im.naturalHeight,
                  style: (im.getAttribute('style') || '').slice(0, 120),
                  parentCls: im.parentElement ? im.parentElement.className : null,
                  parentId: im.parentElement ? im.parentElement.id : null,
                });
              });
              const html = document.documentElement.outerHTML;
              const m = html.match(/getRandcode[^"']*/);
              out.randcodeInHtml = m ? m[0] : null;
              document.querySelectorAll('input[type=hidden]').forEach(h => {
                out.hidden.push({name: h.name, id: h.id, value: (h.value || '').slice(0, 80)});
              });
              return out;
            }"""
        )
        (OUT / "11_captcha_dom.json").write_text(
            json.dumps(captcha_info, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print("== imgs after interaction ==")
        for im in captcha_info["imgs"]:
            print(json.dumps(im, ensure_ascii=False))
        print("randcode in html:", captcha_info["randcodeInHtml"])
        print("hidden inputs:", json.dumps(captcha_info["hidden"], ensure_ascii=False))

        # ---- 保存整页 HTML 供离线分析 ----
        (OUT / "12_page.html").write_text(page.content(), encoding="utf-8")

        # ---- 用页面会话 fetch 验证码原图字节并保存 ----
        saved = page.evaluate(
            """async () => {
              const im = [...document.querySelectorAll('img')]
                .find(i => i.src.includes('getRandcode'));
              if (!im) return null;
              const res = await fetch(im.src, {credentials: 'include'});
              const blob = await res.blob();
              return {status: res.status, type: blob.type, size: blob.size,
                      dataUrl: await new Promise(r => {
                        const fr = new FileReader();
                        fr.onload = () => r(fr.result);
                        fr.readAsDataURL(blob);
                      })};
            }"""
        )
        if saved:
            (OUT / "13_captcha_fetch.json").write_text(
                json.dumps(
                    {k: v for k, v in saved.items() if k != "dataUrl"},
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            b64 = saved["dataUrl"].split(",", 1)[1]
            (OUT / "captcha_sample1.bin").write_bytes(__import__("base64").b64decode(b64))
            print("captcha image saved:", saved["status"], saved["type"], saved["size"], "bytes")

        # ---- 点击验证码图片看是否刷新 ----
        clicked = page.evaluate(
            """() => {
              const im = [...document.querySelectorAll('img')]
                .find(i => i.src.includes('getRandcode'));
              if (!im) return false;
              im.click();
              return true;
            }"""
        )
        page.wait_for_timeout(3000)
        page.screenshot(path=str(OUT / "14_after_captcha_click.png"), full_page=True)
        print("clicked captcha img:", clicked)

        # ---- cookies ----
        cookies = context.cookies()
        (OUT / "15_cookies.json").write_text(
            json.dumps(cookies, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print("== cookies ==")
        for c in cookies:
            print(c["name"], "=", (c["value"] or "")[:40], "domain=" + c["domain"])

        browser.close()

    (OUT / "16_network_interact.json").write_text(
        json.dumps(network_log, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nnetwork requests captured: {len(network_log)}")
    for item in network_log:
        mark = "  ***" if "randcode" in item["url"].lower() or item["method"] == "POST" else ""
        print(item["status"], item["method"], item["type"], item["url"][:150], mark)
    return 0


if __name__ == "__main__":
    sys.exit(main())

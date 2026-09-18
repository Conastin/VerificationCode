# -*- coding: utf-8 -*-
"""重放可行性实验: 抓图后延迟 D 再提交, 观察验证码检查是否仍运行/匹配.

每个延迟档位:
  A) 错码提交(ZZZZ) -> 期望"验证码输入错误" = 检查仍在运行且绑定原图
  B) 模型预测提交    -> 期望"用户不存在" = 存储的码仍对应该张图(完整重放)
"""
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
from verification_code.portal_login_validate import (classify, make_opener,
                                                     fetch_login_page,
                                                     fetch_captcha, submit_login)
from verification_code.train_universal import CrnnCaptcha, ctc_greedy, preprocess, set_img_h

CKPT = "checkpoints/universal_crnn_ft_portal_v7.pt"


def predict(model, data: bytes) -> str:
    import io
    from PIL import Image
    with Image.open(io.BytesIO(data)) as im:
        arr = preprocess(im.convert("RGB"), model.img_h)
    x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.no_grad():
        (p, _), = ctc_greedy(model(x))
    return p


def one_round(model, delay_s: float, mode: str) -> str:
    opener = make_opener()
    page = fetch_login_page(opener)
    data = fetch_captcha(opener, page["lt"])
    if mode == "wrong":
        code = "ZZZZ"
    else:
        code = predict(model, data)
    if delay_s:
        time.sleep(delay_s)
    status, resp = submit_login(opener, page, "testuser01", "Test123456!", code)
    verdict, hint = classify(resp)
    extra = ""
    if verdict == "unknown":
        m = re.search(r"<div id=\"msg\"[^>]*>([^<]*)</div>", resp)
        extra = m.group(1) if m else "(no msg div)"
    return f"delay={delay_s}s {mode} code={code} -> {verdict} | {hint} {extra}"


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(CKPT, map_location=device, weights_only=False)
    set_img_h(ckpt.get("img_h", 32))
    model = CrnnCaptcha(img_h=ckpt.get("img_h", 32),
                        lstm_hidden=ckpt.get("lstm_hidden", 192),
                        lstm_layers=ckpt.get("lstm_layers", 2)).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    print(one_round(model, 0, "wrong"))       # 对照: 立即错码
    print(one_round(model, 0, "model"))       # 对照: 立即模型码
    print(one_round(model, 60, "wrong"))      # 1 分钟后
    print(one_round(model, 60, "model"))
    print(one_round(model, 600, "wrong"))     # 10 分钟后
    print(one_round(model, 600, "model"))

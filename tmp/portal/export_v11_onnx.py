# -*- coding: utf-8 -*-
"""导出 v11 通用 CRNN 为动态宽度 ONNX."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from verification_code.train_universal import CrnnCaptcha


def main() -> None:
    ckpt = torch.load("checkpoints/universal_crnn_ft_portal_v11.pt",
                      map_location="cpu", weights_only=False)
    img_h = ckpt.get("img_h", 48)
    model = CrnnCaptcha(img_h=img_h,
                        lstm_hidden=ckpt.get("lstm_hidden", 192),
                        lstm_layers=ckpt.get("lstm_layers", 2),
                        in_channels=ckpt.get("channels", 3))
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    dummy = torch.randn(1, 3, img_h, 160)
    out = Path("browser/extension/model.onnx")
    torch.onnx.export(
        model, dummy, str(out),
        input_names=["image"], output_names=["logits"],
        dynamic_axes={"image": {0: "batch", 3: "width"},
                      "logits": {0: "batch", 1: "frames"}},
        opset_version=17, do_constant_folding=True,
    )
    print(f"exported {out} ({out.stat().st_size/1e6:.1f} MB), img_h={img_h}")
    # 验证: torch vs onnxruntime 一致性(两个宽度)
    import numpy as np
    import onnxruntime as ort
    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    for w in (113, 144):
        x = torch.randn(1, 3, img_h, w)
        with torch.no_grad():
            t = model(x).numpy()
        o = sess.run(None, {"image": x.numpy()})[0]
        print(f"w={w}: torch{Ttuple(t.shape)} vs onnx{o.shape}, max_diff={np.abs(t-o).max():.2e}")


def Ttuple(shape):
    return tuple(shape)


if __name__ == "__main__":
    main()

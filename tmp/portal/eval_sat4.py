# -*- coding: utf-8 -*-
"""sat4 模型评估: 旧站 + portal test-60."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from verification_code.train_universal import CrnnCaptcha, evaluate_folder, set_img_h


def main() -> None:
    device = torch.device("cuda")
    ckpt = torch.load("checkpoints/universal_crnn_sat4.pt", map_location=device,
                      weights_only=False)
    set_img_h(ckpt.get("img_h", 32))
    model = CrnnCaptcha(img_h=ckpt.get("img_h", 32),
                        lstm_hidden=ckpt.get("lstm_hidden", 192),
                        lstm_layers=ckpt.get("lstm_layers", 2),
                        in_channels=ckpt.get("channels", 3)).to(device)
    model.load_state_dict(ckpt["model_state"])
    print("== sat4 @ 旧站 ==")
    evaluate_folder(model, Path("data/real_test_independent"), device)
    print("== sat4 @ portal test-60 ==")
    evaluate_folder(model, Path("data/portal_label/test"), device)


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""portal 留出测试集评估."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from verification_code.train_universal import CrnnCaptcha, evaluate_folder, set_img_h


def main() -> None:
    device = torch.device("cuda")
    ckpt = torch.load("checkpoints/universal_crnn_ft_portal_v12.pt", map_location=device,
                      weights_only=False)
    set_img_h(ckpt.get("img_h", 32))
    model = CrnnCaptcha(img_h=ckpt.get("img_h", 32),
                        lstm_hidden=ckpt.get("lstm_hidden", 192),
                        lstm_layers=ckpt.get("lstm_layers", 2)).to(device)
    model.load_state_dict(ckpt["model_state"])
    print("== portal 留出测试集 (60 张, 用户标注, 大小写折叠) ==")
    evaluate_folder(model, Path("data/portal_label/test"), device)


if __name__ == "__main__":
    main()

"""Dataset loading for labeled CAPTCHA image directories.

Expects the layout produced by ``CaptchaGenerator.generate_dataset``:
a directory of JPEGs plus a ``labels.txt`` file with one
``<filename> <label>`` line per image.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .model import DEFAULT_CHARSET
from .model import encode_label


class CaptchaDataset(Dataset):
    """Read a generated CAPTCHA directory into image/label tensors."""

    def __init__(
        self,
        data_dir: str | Path,
        charset: str = DEFAULT_CHARSET,
        length: int = 4,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.charset = charset
        self.length = length
        self.samples: list[tuple[Path, str]] = []
        labels_path = self.data_dir / "labels.txt"
        if not labels_path.is_file():
            raise FileNotFoundError(f"missing labels file: {labels_path}")
        with labels_path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                filename, label = line.split()
                self.samples.append((self.data_dir / filename, label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        path, label = self.samples[index]
        with Image.open(path) as image:
            array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1)
        return tensor, torch.tensor(encode_label(label, self.charset), dtype=torch.long)

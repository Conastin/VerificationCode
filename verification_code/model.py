"""CNN model for fixed-length CAPTCHA recognition.

Architecture follows the README roadmap: a shared CNN backbone with four
classification heads, one per character slot (60x20 RGB input, 4 characters).
"""

from __future__ import annotations

import torch
import torch.nn as nn

# Real CAPTCHA charset: excludes confusable characters (0, 1, I, L, O).
DEFAULT_CHARSET = "23456789ABCDEFGHJKMNPQRSTUVWXYZ"


def encode_label(label: str, charset: str = DEFAULT_CHARSET) -> list[int]:
    """Map a label string to charset indices (one per character slot)."""
    return [charset.index(char) for char in label]


def build_model_from_checkpoint(
    checkpoint: dict,
    charset: str = DEFAULT_CHARSET,
) -> CaptchaCNN | SlotCaptchaModel:
    """Instantiate the model architecture recorded in a training checkpoint."""
    arch = checkpoint.get("arch", "full")
    length = checkpoint.get("length", 4)
    if arch == "slot":
        return SlotCaptchaModel(length=length, num_classes=len(charset))
    return CaptchaCNN(length=length, num_classes=len(charset))


def decode_logits(logits: torch.Tensor, charset: str = DEFAULT_CHARSET) -> tuple[list[str], torch.Tensor]:
    """Decode per-slot logits into predicted labels and per-slot probabilities.

    Args:
        logits: tensor of shape (batch, length, num_classes).
        charset: character set used for the output classes.

    Returns:
        (labels, probs) where labels are the argmax predictions and probs is
        the softmax probability of the predicted class per slot.
    """
    probs = torch.softmax(logits, dim=-1)
    pred_ids = probs.argmax(dim=-1)
    labels = ["".join(charset[idx] for idx in row) for row in pred_ids.tolist()]
    conf = probs.gather(-1, pred_ids.unsqueeze(-1)).squeeze(-1)
    return labels, conf


class CaptchaCNN(nn.Module):
    """Shared CNN backbone with four independent classification heads.

    The CAPTCHA is small (60x20), so the backbone stays shallow; the three
    conv layers with max pooling reduce it to a 15x5 feature map before the
    heads. Each head classifies one character slot over `num_classes`.
    """

    def __init__(
        self,
        length: int = 4,
        num_classes: int = 36,
        backbone_channels: tuple[int, int, int] = (32, 64, 128),
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.length = length
        self.num_classes = num_classes

        c1, c2, c3 = backbone_channels
        self.backbone = nn.Sequential(
            nn.Conv2d(3, c1, kernel_size=3, padding=1),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 30x10
            nn.Conv2d(c1, c2, kernel_size=3, padding=1),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 15x5
            nn.Conv2d(c2, c3, kernel_size=3, padding=1),
            nn.BatchNorm2d(c3),
            nn.ReLU(inplace=True),
        )
        # Flattened feature size: c3 * 15 * 5 for a 60x20 input.
        self.feature_dim = c3 * 15 * 5
        self.embedding = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.heads = nn.ModuleList(
            [nn.Linear(256, num_classes) for _ in range(length)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return logits of shape (batch, length, num_classes)."""
        features = self.embedding(self.backbone(x))
        return torch.stack([head(features) for head in self.heads], dim=1)


class SlotCaptchaModel(nn.Module):
    """Slot-cropped CAPTCHA recognizer.

    Crops each fixed 15px character slot out of the full 60x20 image and runs
    a shared single-character CNN over all four crops. This reaches ~99.9%
    char accuracy on the generated data, far above the full-image shared
    backbone (the 60x20 image is too small for a pooled feature map to keep
    the four slot regions separable).
    """

    def __init__(
        self,
        length: int = 4,
        num_classes: int = 36,
        slot_width: int = 15,
        channels: tuple[int, int] = (32, 64),
    ) -> None:
        super().__init__()
        self.length = length
        self.num_classes = num_classes
        self.slot_width = slot_width

        c1, c2 = channels
        self.char_net = nn.Sequential(
            nn.Conv2d(3, c1, kernel_size=3, padding=1),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 7x10
            nn.Conv2d(c1, c2, kernel_size=3, padding=1),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
            nn.Conv2d(c2, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(128 * 7 * 10, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return logits of shape (batch, length, num_classes)."""
        crops = [x[:, :, :, s * self.slot_width:(s + 1) * self.slot_width] for s in range(self.length)]
        return torch.stack([self.char_net(crop) for crop in crops], dim=1)

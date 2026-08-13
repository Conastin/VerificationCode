"""Tests for the CAPTCHA model, dataset loading, and training plumbing."""

import shutil
from pathlib import Path

import torch

from verification_code.dataset import CaptchaDataset
from verification_code.model import (
    DEFAULT_CHARSET,
    CaptchaCNN,
    SlotCaptchaModel,
    build_model_from_checkpoint,
    decode_logits,
    encode_label,
)

SYNTHETIC_SOURCE = Path("data/val31")  # 31-char synthetic samples (fixture data)


def _make_labeled_dir(tmp_path: Path, count: int) -> Path:
    """Copy a few labeled samples from the synthetic fixture into tmp_path."""
    labels = []
    with (SYNTHETIC_SOURCE / "labels.txt").open(encoding="utf-8") as handle:
        for line in handle:
            parts = line.split()
            if len(parts) == 2:
                labels.append(parts)
            if len(labels) == count:
                break
    for name, _label in labels:
        shutil.copy2(SYNTHETIC_SOURCE / name, tmp_path / name)
    with (tmp_path / "labels.txt").open("w", encoding="utf-8", newline="\n") as handle:
        for name, label in labels:
            handle.write(f"{name} {label}\n")
    return tmp_path


def test_forward_output_shape():
    model = CaptchaCNN(length=4, num_classes=len(DEFAULT_CHARSET))
    logits = model(torch.randn(3, 3, 20, 60))
    assert logits.shape == (3, 4, len(DEFAULT_CHARSET))


def test_slot_model_forward_output_shape():
    model = SlotCaptchaModel(length=4, num_classes=len(DEFAULT_CHARSET))
    logits = model(torch.randn(3, 3, 20, 60))
    assert logits.shape == (3, 4, len(DEFAULT_CHARSET))


def test_build_model_from_checkpoint():
    checkpoint = {"arch": "slot", "length": 4}
    assert isinstance(build_model_from_checkpoint(checkpoint), SlotCaptchaModel)
    checkpoint = {"length": 4}
    assert isinstance(build_model_from_checkpoint(checkpoint), CaptchaCNN)


def test_encode_decode_roundtrip():
    label = "A7K3"
    ids = encode_label(label)
    logits = torch.full((1, 4, len(DEFAULT_CHARSET)), -10.0)
    for slot, class_id in enumerate(ids):
        logits[0, slot, class_id] = 5.0
    decoded, conf = decode_logits(logits)
    assert decoded == [label]
    assert torch.all(conf[0] > 0.99)


def test_dataset_reads_labeled_directory(tmp_path: Path):
    _make_labeled_dir(tmp_path, count=5)
    dataset = CaptchaDataset(tmp_path)
    assert len(dataset) == 5
    image, label = dataset[0]
    assert image.shape == (3, 20, 60)
    assert image.min() >= 0.0 and image.max() <= 1.0
    assert label.shape == (4,)
    assert all(0 <= int(item) < len(DEFAULT_CHARSET) for item in label)


def test_training_step_runs(tmp_path: Path):
    _make_labeled_dir(tmp_path, count=8)
    dataset = CaptchaDataset(tmp_path)
    loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)

    model = SlotCaptchaModel(length=4, num_classes=len(DEFAULT_CHARSET))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()

    images, labels = next(iter(loader))
    logits = model(images)
    loss = criterion(logits.reshape(-1, len(DEFAULT_CHARSET)), labels.reshape(-1))
    loss.backward()
    optimizer.step()

    assert loss.item() > 0.0
    assert all(param.grad is not None for param in model.char_net.parameters())

"""Data-loader smoke test against a tiny synthetic dataset."""

from __future__ import annotations

from pathlib import Path

import torch

from traffic_signs.config import DataConfig, TrainConfig
from traffic_signs.data.gtsrb import build_dataloaders


def test_dataloaders_shape_and_split(synthetic_gtsrb: dict[str, Path]):
    data_cfg = DataConfig(
        train_dir=synthetic_gtsrb["train_dir"],
        test_dir=synthetic_gtsrb["test_dir"],
        image_size=48,
        num_classes=3,
        val_split=0.25,
        num_workers=0,
    )
    train_cfg = TrainConfig(epochs=1, batch_size=4, learning_rate=1e-3)
    train_loader, val_loader, test_loader, classes = build_dataloaders(data_cfg, train_cfg, seed=0)

    assert classes == ["00000", "00001", "00002"]

    # Train + val must exactly equal the training set
    total_train_samples = sum(1 for _ in train_loader.dataset) + sum(1 for _ in val_loader.dataset)
    assert total_train_samples == 3 * 8

    images, labels = next(iter(train_loader))
    assert images.shape[1:] == (3, 48, 48)
    assert labels.dtype == torch.int64

    # Test loader is separate, never part of the split
    assert sum(1 for _ in test_loader.dataset) == 3 * 4

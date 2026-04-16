"""Shared pytest fixtures — synthetic GTSRB-like dataset on disk."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image


def _write_synthetic_split(
    root: Path,
    classes: int,
    samples_per_class: int,
    image_size: int,
    seed: int,
) -> None:
    rng = np.random.default_rng(seed)
    for c in range(classes):
        class_dir = root / f"{c:05d}"
        class_dir.mkdir(parents=True, exist_ok=True)
        for i in range(samples_per_class):
            arr = rng.integers(0, 256, size=(image_size, image_size, 3), dtype=np.uint8)
            # Bias channel mean per class so the task is at least weakly learnable —
            # the tests never train to convergence but this catches silent dtype bugs.
            arr[..., c % 3] = np.clip(arr[..., c % 3] + c * 3, 0, 255)
            Image.fromarray(arr).save(class_dir / f"img_{i:03d}.png")


@pytest.fixture(scope="session")
def synthetic_gtsrb(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """Create a miniature GTSRB-like dataset (3 classes × 6 samples)."""

    root = tmp_path_factory.mktemp("gtsrb_synth")
    train_dir = root / "train"
    test_dir = root / "test"
    _write_synthetic_split(train_dir, classes=3, samples_per_class=8, image_size=48, seed=0)
    _write_synthetic_split(test_dir, classes=3, samples_per_class=4, image_size=48, seed=1)
    return {"train_dir": train_dir, "test_dir": test_dir}

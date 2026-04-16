"""GTSRB dataset loading.

Fixes introduced versus the original notebook
---------------------------------------------
1. The notebook used the TEST set as validation during training. That is a
   data leak: the test metrics later reported in the comparison table are
   tainted. Here the validation set is carved out of the TRAINING folder with
   a stratified split, and the test set is only touched by
   :mod:`traffic_signs.evaluation`.
2. ``horizontal_flip`` defaults to OFF — GTSRB contains directional signs
   ("turn left only", "no entry", arrow markers) whose label becomes wrong
   under a horizontal flip. The setting is still configurable for users who
   want to experiment, but the safe default is correct.
3. Paths come from :class:`~traffic_signs.config.DataConfig`, not from
   hard-coded ``C:\\Users\\…`` strings.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder

from traffic_signs.config import DataConfig, TrainConfig

log = logging.getLogger(__name__)

# ImageNet-style normalisation is convenient when fine-tuning pretrained
# backbones. For the small bespoke CNNs in this repo, 0.5/0.5 is equally
# valid; we use the latter for consistency with the notebook's ``Normalize``.
_MEAN: tuple[float, float, float] = (0.5, 0.5, 0.5)
_STD: tuple[float, float, float] = (0.5, 0.5, 0.5)


def build_transforms(
    data_cfg: DataConfig,
    train_cfg: TrainConfig,
) -> tuple[transforms.Compose, transforms.Compose]:
    """Return ``(train_transform, eval_transform)`` based on the config."""

    size = data_cfg.image_size

    train_tf: list[torch.nn.Module] = [
        transforms.Resize((size, size)),
        transforms.ColorJitter(
            brightness=train_cfg.color_jitter,
            contrast=train_cfg.color_jitter,
            saturation=train_cfg.color_jitter,
        ),
        transforms.RandomAffine(
            degrees=train_cfg.rotation_degrees,
            translate=(0.1, 0.1),
            shear=5,
        ),
    ]
    if train_cfg.horizontal_flip:
        log.warning(
            "horizontal_flip=True on GTSRB is a known labelling hazard for "
            "directional signs. Proceeding because it was requested explicitly."
        )
        train_tf.append(transforms.RandomHorizontalFlip())

    train_tf.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(_MEAN, _STD),
        ]
    )

    eval_tf = transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(_MEAN, _STD),
        ]
    )

    return transforms.Compose(train_tf), eval_tf


def _stratified_indices(
    targets: Sequence[int],
    val_fraction: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    """Per-class stratified split producing ``(train_idx, val_idx)``."""

    generator = torch.Generator().manual_seed(seed)
    per_class: dict[int, list[int]] = {}
    for idx, label in enumerate(targets):
        per_class.setdefault(int(label), []).append(idx)

    train_idx: list[int] = []
    val_idx: list[int] = []
    for label, indices in per_class.items():
        n = len(indices)
        n_val = max(1, int(round(n * val_fraction)))
        perm = torch.randperm(n, generator=generator).tolist()
        shuffled = [indices[i] for i in perm]
        val_idx.extend(shuffled[:n_val])
        train_idx.extend(shuffled[n_val:])

    return train_idx, val_idx


def build_dataloaders(
    data_cfg: DataConfig,
    train_cfg: TrainConfig,
    *,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, DataLoader, list[str]]:
    """Build train, validation and test dataloaders.

    Returns
    -------
    train_loader, val_loader, test_loader, class_names
    """

    train_dir = Path(data_cfg.train_dir)
    test_dir = Path(data_cfg.test_dir)
    if not train_dir.is_dir():
        raise FileNotFoundError(
            f"Training directory not found: {train_dir!s}\n"
            "See data/README.md for how to prepare the GTSRB dataset."
        )
    if not test_dir.is_dir():
        raise FileNotFoundError(
            f"Test directory not found: {test_dir!s}\n"
            "See data/README.md for how to prepare the GTSRB dataset."
        )

    train_tf, eval_tf = build_transforms(data_cfg, train_cfg)

    # We instantiate the training folder twice: once with the train transform
    # (used by the training subset) and once with the eval transform (used by
    # the validation subset). This avoids augmenting validation samples.
    full_train = ImageFolder(str(train_dir), transform=train_tf)
    full_train_eval = ImageFolder(str(train_dir), transform=eval_tf)
    test_dataset = ImageFolder(str(test_dir), transform=eval_tf)

    n_classes_found = len(full_train.classes)
    if n_classes_found != data_cfg.num_classes:
        log.warning(
            "Dataset has %d classes but config.num_classes=%d. "
            "Continuing with %d.",
            n_classes_found,
            data_cfg.num_classes,
            n_classes_found,
        )

    train_idx, val_idx = _stratified_indices(
        full_train.targets, data_cfg.val_split, seed=seed
    )

    train_subset = Subset(full_train, train_idx)
    val_subset = Subset(full_train_eval, val_idx)

    common = {
        "batch_size": train_cfg.batch_size,
        "num_workers": data_cfg.num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": data_cfg.num_workers > 0,
    }

    train_loader = DataLoader(train_subset, shuffle=True, **common)
    val_loader = DataLoader(val_subset, shuffle=False, **common)
    test_loader = DataLoader(test_dataset, shuffle=False, **common)

    log.info(
        "Datasets ready — train=%d, val=%d, test=%d, classes=%d",
        len(train_subset),
        len(val_subset),
        len(test_dataset),
        n_classes_found,
    )

    return train_loader, val_loader, test_loader, list(full_train.classes)


def load_class_names(train_dir: str | Path) -> list[str]:
    """Return sorted class-folder names from a training directory."""

    return sorted(p.name for p in Path(train_dir).iterdir() if p.is_dir())

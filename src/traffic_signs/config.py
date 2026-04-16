"""Typed experiment configuration.

Rationale
---------
The original notebook hard-codes Windows paths, mixes hyperparameters into
training code, and re-defines constants in every cell (BATCH_SIZE, EPOCHS, …).
That is the primary reason the notebook is not reproducible on another machine.

This module replaces all of that with a single, strictly-typed, YAML-backed
config object. Every training, evaluation or inference entry point accepts an
``ExperimentConfig`` — hyperparameters never appear as module-level constants.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator

ModelName = Literal["traffic_sign_net", "traffic_sign_net_stn", "deep_traffic_net"]


class DataConfig(BaseModel):
    """Filesystem layout for the GTSRB dataset.

    The repo does not ship the data; ``data/README.md`` explains how to fetch
    it. All paths are resolved relative to the project root so that the config
    is portable across machines and OSes.
    """

    train_dir: Path = Field(
        default=Path("data/raw/GTSRB/Final_Training/Images"),
        description="Folder containing one subfolder per class (00000, 00001, …).",
    )
    test_dir: Path = Field(
        default=Path("data/raw/GTSRB/Final_Test/Images"),
        description="Test folder in ImageFolder layout (see data/README.md).",
    )
    image_size: int = Field(default=48, ge=16, le=256)
    num_classes: int = Field(default=43, ge=2)
    val_split: float = Field(default=0.2, ge=0.0, lt=0.5)
    num_workers: int = Field(default=4, ge=0)


class TrainConfig(BaseModel):
    """Optimization hyperparameters."""

    epochs: int = Field(default=20, ge=1)
    batch_size: int = Field(default=64, ge=1)
    learning_rate: float = Field(default=1e-3, gt=0)
    weight_decay: float = Field(default=1e-4, ge=0)
    optimizer: Literal["adam", "adamw", "sgd"] = "adamw"
    scheduler: Literal["plateau", "cosine", "none"] = "plateau"
    early_stopping_patience: int = Field(default=7, ge=1)
    # Augmentation is kept modest — GTSRB contains directional signs (arrows,
    # "no entry", etc.), so horizontal-flip is OFF by default. The original
    # notebook enabled it; that is a silent labelling bug and is corrected here.
    horizontal_flip: bool = False
    color_jitter: float = Field(default=0.2, ge=0.0, le=0.5)
    rotation_degrees: float = Field(default=10.0, ge=0.0, le=45.0)


class OutputConfig(BaseModel):
    checkpoints_dir: Path = Path("checkpoints")
    reports_dir: Path = Path("reports")
    figures_dir: Path = Path("reports/figures")
    metrics_dir: Path = Path("reports/metrics")


class ExperimentConfig(BaseModel):
    """Top-level config object consumed by every entry point."""

    model: ModelName = "deep_traffic_net"
    seed: int = 42
    device: Literal["auto", "cpu", "cuda", "mps"] = "auto"

    data: DataConfig = Field(default_factory=DataConfig)
    train: TrainConfig = Field(default_factory=TrainConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)

    # --- validators -----------------------------------------------------
    @field_validator("data")
    @classmethod
    def _resolve_data_paths(cls, v: DataConfig) -> DataConfig:
        # Leave paths relative; resolving happens at runtime against the CWD,
        # which for CLI/notebooks is always the repository root.
        return v


# -------------------------------------------------------------------------
# Loader
# -------------------------------------------------------------------------
def load_config(path: str | Path | None = None) -> ExperimentConfig:
    """Load an :class:`ExperimentConfig` from YAML.

    Passing ``None`` returns a config populated entirely from defaults — useful
    for tests and for the ``configs/default.yaml`` path assumption.
    """

    if path is None:
        return ExperimentConfig()

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    return ExperimentConfig.model_validate(raw)

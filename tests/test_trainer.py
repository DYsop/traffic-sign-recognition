"""End-to-end trainer test (one epoch) on the synthetic dataset."""

from __future__ import annotations

from pathlib import Path

import torch

from traffic_signs.config import DataConfig, TrainConfig
from traffic_signs.data.gtsrb import build_dataloaders
from traffic_signs.models import build_model
from traffic_signs.training.trainer import Trainer


def test_trainer_runs_one_epoch(synthetic_gtsrb: dict[str, Path], tmp_path: Path):
    data_cfg = DataConfig(
        train_dir=synthetic_gtsrb["train_dir"],
        test_dir=synthetic_gtsrb["test_dir"],
        image_size=48,
        num_classes=3,
        val_split=0.25,
        num_workers=0,
    )
    train_cfg = TrainConfig(
        epochs=1,
        batch_size=4,
        learning_rate=1e-3,
        optimizer="adamw",
        scheduler="none",
        early_stopping_patience=3,
    )
    train_loader, val_loader, _, classes = build_dataloaders(data_cfg, train_cfg, seed=0)
    model = build_model("traffic_sign_net", num_classes=len(classes), image_size=48)
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=train_cfg,
        device=torch.device("cpu"),
        model_name="traffic_sign_net",
        checkpoint_dir=tmp_path / "ckpt",
    )
    history = trainer.fit()
    assert len(history.records) == 1
    assert (tmp_path / "ckpt" / "traffic_sign_net" / "best.pt").is_file()
    assert (tmp_path / "ckpt" / "traffic_sign_net" / "last.pt").is_file()

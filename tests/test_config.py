"""Config round-tripping and validation."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from traffic_signs.config import ExperimentConfig, load_config


def test_default_config_is_valid():
    cfg = ExperimentConfig()
    assert cfg.model in {"traffic_sign_net", "traffic_sign_net_stn", "deep_traffic_net"}
    assert cfg.seed == 42
    assert 0 < cfg.train.learning_rate < 1
    assert cfg.data.num_classes == 43


def test_load_config_roundtrip(tmp_path: Path):
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "model": "traffic_sign_net",
                "seed": 7,
                "train": {"epochs": 3, "batch_size": 8, "learning_rate": 0.01},
            }
        )
    )
    cfg = load_config(cfg_path)
    assert cfg.model == "traffic_sign_net"
    assert cfg.seed == 7
    assert cfg.train.epochs == 3
    assert cfg.train.batch_size == 8


def test_load_config_missing_file_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "nope.yaml")


def test_bad_values_are_rejected():
    with pytest.raises(Exception):
        ExperimentConfig.model_validate({"train": {"learning_rate": -1}})
    with pytest.raises(Exception):
        ExperimentConfig.model_validate({"train": {"epochs": 0}})

"""Evaluation metrics sanity."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from traffic_signs.evaluation.metrics import evaluate
from traffic_signs.evaluation.reports import plot_confusion_matrix, save_classification_report


class _IdentityHead(nn.Module):
    """Model that ignores input and always predicts class 0 — deterministic."""

    def __init__(self, num_classes: int = 3):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        logits = torch.full((batch, self.num_classes), -10.0)
        logits[:, 0] = 10.0
        return logits


def _fake_loader() -> DataLoader:
    x = torch.randn(8, 3, 48, 48)
    y = torch.tensor([0, 0, 0, 0, 1, 1, 2, 2])  # 4 correct out of 8 for a "always-0" model
    return DataLoader(TensorDataset(x, y), batch_size=4)


def test_evaluate_produces_consistent_metrics():
    model = _IdentityHead(num_classes=3)
    loader = _fake_loader()
    result = evaluate(
        model, loader, device=torch.device("cpu"), model_name="dummy", class_names=["a", "b", "c"]
    )
    assert result.num_samples == 8
    assert abs(result.accuracy - 0.5) < 1e-6
    assert result.model == "dummy"
    assert len(result.confusion_matrix) == 3


def test_report_files_are_written(tmp_path: Path):
    model = _IdentityHead(num_classes=3)
    loader = _fake_loader()
    result = evaluate(
        model, loader, device=torch.device("cpu"), model_name="dummy", class_names=["a", "b", "c"]
    )

    cm_path = plot_confusion_matrix(result, tmp_path / "cm.png")
    rep_path = save_classification_report(result, tmp_path / "rep.csv")
    assert cm_path.is_file() and cm_path.stat().st_size > 0
    assert rep_path.is_file() and rep_path.stat().st_size > 0

"""Model evaluation on a held-out loader.

Produces a single, JSON-serialisable :class:`EvaluationResult`. All comparison
tables in the notebook and in the README are built from these objects — so by
construction the same metric (test accuracy, test loss) is reported for every
model, eliminating the apples-to-oranges issue in the original notebook.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    matthews_corrcoef,
    top_k_accuracy_score,
)
from torch import nn
from torch.utils.data import DataLoader

log = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    model: str
    loss: float
    accuracy: float
    top5_accuracy: float
    mcc: float
    kappa: float
    num_samples: int
    confusion_matrix: list[list[int]] = field(default_factory=list)
    per_class_report: dict[str, dict[str, float]] = field(default_factory=dict)

    def to_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2)
        log.info("Evaluation result written to %s", path)


@torch.inference_mode()
def _collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    model.eval()
    all_logits: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    loss_sum = 0.0
    total = 0
    criterion = nn.CrossEntropyLoss(reduction="sum")

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        loss_sum += float(criterion(logits, labels).item())
        total += labels.size(0)
        all_logits.append(logits.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    logits = np.concatenate(all_logits, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    preds = logits.argmax(axis=1)
    avg_loss = loss_sum / max(1, total)
    return labels, preds, logits, avg_loss


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    model_name: str,
    class_names: list[str] | None = None,
) -> EvaluationResult:
    """Run a full evaluation pass."""

    labels, preds, logits, loss = _collect_predictions(model, loader, device)

    acc = float(accuracy_score(labels, preds))
    probs = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
    num_classes = logits.shape[1]
    top5 = float(top_k_accuracy_score(labels, probs, k=5, labels=list(range(num_classes))))
    mcc = float(matthews_corrcoef(labels, preds))
    kappa = float(cohen_kappa_score(labels, preds))
    cm = confusion_matrix(labels, preds).tolist()

    target_names = class_names if class_names and len(class_names) == num_classes else None
    report_dict = classification_report(
        labels,
        preds,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )
    # classification_report returns float64 values — convert for JSON safety
    report_dict = {
        k: {
            kk: float(vv) if isinstance(vv, (int, float, np.floating)) else vv
            for kk, vv in v.items()
        }
        if isinstance(v, dict)
        else v
        for k, v in report_dict.items()
    }

    return EvaluationResult(
        model=model_name,
        loss=loss,
        accuracy=acc,
        top5_accuracy=top5,
        mcc=mcc,
        kappa=kappa,
        num_samples=int(labels.shape[0]),
        confusion_matrix=cm,
        per_class_report=report_dict,
    )

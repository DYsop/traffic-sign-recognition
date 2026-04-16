"""Report generation (figures, CSVs)."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from traffic_signs.evaluation.metrics import EvaluationResult
from traffic_signs.training.trainer import TrainingHistory

log = logging.getLogger(__name__)

sns.set_theme(style="whitegrid", context="notebook")


def plot_training_curves(history: TrainingHistory, out_path: Path) -> Path:
    """Save a two-panel figure: accuracy and loss by epoch."""

    out_path.parent.mkdir(parents=True, exist_ok=True)
    records = history.records
    epochs = [r.epoch for r in records]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(epochs, [r.train_acc for r in records], label="train", marker="o", ms=3)
    axes[0].plot(epochs, [r.val_acc for r in records], label="val", marker="o", ms=3)
    axes[0].set(title=f"{history.model} — Accuracy", xlabel="Epoch", ylabel="Accuracy")
    axes[0].legend()

    axes[1].plot(epochs, [r.train_loss for r in records], label="train", marker="o", ms=3)
    axes[1].plot(epochs, [r.val_loss for r in records], label="val", marker="o", ms=3)
    axes[1].set(title=f"{history.model} — Loss", xlabel="Epoch", ylabel="Cross-Entropy")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Wrote training curves to %s", out_path)
    return out_path


def plot_confusion_matrix(result: EvaluationResult, out_path: Path) -> Path:
    """Save a normalised confusion-matrix heatmap."""

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cm = np.asarray(result.confusion_matrix, dtype=float)
    with np.errstate(invalid="ignore"):
        cm_norm = cm / cm.sum(axis=1, keepdims=True)
    cm_norm = np.nan_to_num(cm_norm)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm_norm,
        cmap="Blues",
        cbar=True,
        square=True,
        xticklabels=False,
        yticklabels=False,
        ax=ax,
    )
    ax.set(
        title=f"{result.model} — Normalised Confusion Matrix (test set)",
        xlabel="Predicted class",
        ylabel="True class",
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Wrote confusion matrix to %s", out_path)
    return out_path


def save_classification_report(result: EvaluationResult, out_path: Path) -> Path:
    """Save the per-class report as a CSV so it renders nicely on GitHub."""

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(result.per_class_report).T
    df.to_csv(out_path, float_format="%.4f")
    log.info("Wrote classification report to %s", out_path)
    return out_path


def build_comparison_table(results: list[EvaluationResult]) -> pd.DataFrame:
    """Build the three-model comparison table used in the README."""

    rows = []
    for r in results:
        rows.append(
            {
                "Model": r.model,
                "Test Accuracy": r.accuracy,
                "Top-5 Accuracy": r.top5_accuracy,
                "Test Loss": r.loss,
                "MCC": r.mcc,
                "Cohen's Kappa": r.kappa,
                "N (test)": r.num_samples,
            }
        )
    return pd.DataFrame(rows)

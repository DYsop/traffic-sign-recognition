"""Generate Figure 7.7: worst-5-classes-per-model bar grid.

Produces a 1x3 panel figure showing the five lowest-accuracy classes per model
as horizontal bar charts, side by side. This visualises the per-class weakness
profile documented in Table 7.4 and makes the architecture-specific failure
patterns (the focus of § 8.4) immediately visible to the reader.

The key visual story is:
- TrafficSignNet's weaknesses are moderate (90-96 percent range)
- TrafficSignNet-STN's weakest classes cluster around 87-94 percent
- DeepTrafficNet has two dramatic failures (classes 21, 30) below 80 percent

Output: reports/figures/comparison/worst_classes_per_model.png
Referenced from: docs/07_results.md § 7.4 (future update)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# GTSRB class names (subset used for labelling)
CLASS_NAMES = {
    0: "Speed limit 20",
    3: "Speed limit 60",
    6: "End of 80",
    12: "Priority road",
    19: "Danger curve L",
    21: "Double curve",
    22: "Bumpy road",
    27: "Pedestrians",
    30: "Ice/snow",
    31: "Wild animals",
    39: "Keep left",
    40: "Roundabout",
}

MODELS = [
    ("traffic_sign_net", "TrafficSignNet"),
    ("traffic_sign_net_stn", "TrafficSignNet-STN"),
    ("deep_traffic_net", "DeepTrafficNet"),
]

# Colours: dark blue for TSN, teal for STN, warm orange for Deep
# Chosen to be legible in black-and-white print and accessible
PANEL_COLOUR = {
    "traffic_sign_net": "#1f3c88",  # deep blue
    "traffic_sign_net_stn": "#0d8b8b",  # teal
    "deep_traffic_net": "#d2691e",  # burnt orange
}


def compute_worst_classes(test_metrics: dict, n: int = 5) -> list[tuple[int, float, int, int]]:
    """Return the n classes with lowest per-class accuracy.

    Returns: list of (class_idx, accuracy, correct, support) tuples, ascending.
    """
    cm = np.array(test_metrics["confusion_matrix"])
    support = cm.sum(axis=1)
    correct = cm.diagonal()
    # Guard against zero-support classes (shouldn't happen in GTSRB test)
    acc = np.where(support > 0, correct / support, 1.0)

    worst_indices = np.argsort(acc)[:n]
    return [(int(c), float(acc[c]), int(correct[c]), int(support[c])) for c in worst_indices]


def load_metrics(metrics_root: Path) -> dict[str, dict]:
    """Load test_metrics.json for each of the three models."""
    result = {}
    for model_key, _ in MODELS:
        path = metrics_root / model_key / "test_metrics.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing metrics file: {path}")
        result[model_key] = json.loads(path.read_text())
    return result


def render_panel(
    ax: plt.Axes,
    model_key: str,
    display_name: str,
    worst: list[tuple[int, float, int, int]],
) -> None:
    """Render a single panel showing worst-5 classes for one model."""
    # Reverse so the worst bar is at top
    worst = list(reversed(worst))

    labels = []
    accuracies = []
    for cls_idx, acc, _correct, _support in worst:
        name = CLASS_NAMES.get(cls_idx, f"Class {cls_idx}")
        labels.append(f"{cls_idx:02d} · {name}")
        accuracies.append(acc * 100)

    y_pos = np.arange(len(labels))
    colour = PANEL_COLOUR[model_key]

    bars = ax.barh(y_pos, accuracies, color=colour, alpha=0.85, edgecolor="black", linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlim(60, 100)
    ax.set_xlabel("Per-class accuracy (%)", fontsize=10)
    ax.set_title(display_name, fontsize=11, fontweight="bold")
    ax.axvline(x=95, color="grey", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.axvline(x=100, color="black", linewidth=0.5, alpha=0.3)

    # Annotate each bar with its exact value
    for bar, acc, (_, _, correct, support) in zip(bars, accuracies, worst, strict=True):
        width = bar.get_width()
        ax.text(
            width + 0.5,
            bar.get_y() + bar.get_height() / 2,
            f"{acc:.1f}%  ({correct}/{support})",
            va="center",
            fontsize=8,
            color="black",
        )

    ax.set_axisbelow(True)
    ax.grid(axis="x", alpha=0.25, linestyle=":")


def render_figure(metrics: dict[str, dict]) -> plt.Figure:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=120, sharey=False)
    fig.suptitle(
        "Figure 7.7 — Five lowest-accuracy classes per model on the GTSRB test set",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )

    for ax, (model_key, display_name) in zip(axes, MODELS, strict=True):
        worst = compute_worst_classes(metrics[model_key], n=5)
        render_panel(ax, model_key, display_name, worst)

    fig.tight_layout()
    return fig


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metrics-root",
        type=Path,
        default=Path("reports/metrics"),
        help="Path to reports/metrics directory",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/figures/comparison/worst_classes_per_model.png"),
        help="Destination PNG path",
    )
    args = parser.parse_args()

    metrics = load_metrics(args.metrics_root)
    fig = render_figure(metrics)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight", dpi=150)
    plt.close(fig)

    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Generate the three figures referenced in docs/03_dataset.md.

The figures are saved under reports/figures/dataset/ in PNG format at
150 DPI, which is a good compromise between GitHub's rendering quality
and repository size.

Usage:
    # From the repository root, with the venv active and the dataset
    # already placed under data/raw/GTSRB/:
    python scripts/generate_dataset_figures.py

Outputs:
    reports/figures/dataset/sample_grid.png
    reports/figures/dataset/class_distribution.png
    reports/figures/dataset/directional_flips.png
"""

from __future__ import annotations

import logging
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image, ImageOps

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
TRAIN_DIR = REPO_ROOT / "data" / "raw" / "GTSRB" / "Final_Training" / "Images"
OUT_DIR = REPO_ROOT / "reports" / "figures" / "dataset"

# Deterministic sampling for reproducibility of the figure contents.
SEED = 42


def _ensure_dataset_present() -> None:
    if not TRAIN_DIR.is_dir():
        raise SystemExit(
            f"Training directory not found at {TRAIN_DIR}.\n"
            "Place the extracted GTSRB dataset under data/raw/GTSRB/ "
            "before running this script (see data/README.md)."
        )


def _random_image_for_class(class_dir: Path, rng: random.Random) -> Image.Image:
    candidates = sorted(
        list(class_dir.glob("*.ppm")) + list(class_dir.glob("*.png"))
    )
    if not candidates:
        raise RuntimeError(f"No images in {class_dir}")
    path = rng.choice(candidates)
    return Image.open(path).convert("RGB")


# ---------------------------------------------------------------------------
# Figure 3.1 — sample grid, one image per class
# ---------------------------------------------------------------------------
def make_sample_grid(out_path: Path) -> None:
    """Render one random image per class on an 8 by 6 grid."""

    rng = random.Random(SEED)
    class_dirs = sorted(p for p in TRAIN_DIR.iterdir() if p.is_dir())
    log.info("Found %d class directories", len(class_dirs))

    cols = 8
    rows = (len(class_dirs) + cols - 1) // cols  # 6 for 43 classes
    fig, axes = plt.subplots(rows, cols, figsize=(12, rows * 1.8))

    for ax, class_dir in zip(axes.flat, class_dirs, strict=False):
        img = _random_image_for_class(class_dir, rng).resize((64, 64))
        ax.imshow(img)
        ax.set_title(f"{int(class_dir.name)}", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    # Hide any unused subplots.
    for ax in axes.flat[len(class_dirs):]:
        ax.axis("off")

    fig.suptitle("GTSRB — one random image per class", fontsize=12, y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Wrote %s", out_path.relative_to(REPO_ROOT))


# ---------------------------------------------------------------------------
# Figure 3.2 — class distribution histogram
# ---------------------------------------------------------------------------
def make_class_distribution(out_path: Path) -> None:
    """Bar chart of training-set sample counts per class."""

    class_dirs = sorted(p for p in TRAIN_DIR.iterdir() if p.is_dir())
    counts_dict = {
        int(d.name): sum(
            1 for _ in list(d.glob("*.ppm")) + list(d.glob("*.png"))
        )
        for d in class_dirs
    }
    counts = pd.Series(counts_dict).sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(13, 4.5))
    sns.barplot(x=counts.index.astype(str), y=counts.values, color="#3b7dd8", ax=ax)
    ax.set_xlabel("Class id (sorted by sample count)")
    ax.set_ylabel("Sample count")
    ax.set_title(
        "GTSRB training partition — samples per class "
        f"(min={counts.min()}, median={int(counts.median())}, "
        f"max={counts.max()}, ratio={counts.max() / counts.min():.1f})"
    )
    ax.tick_params(axis="x", rotation=90, labelsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Wrote %s", out_path.relative_to(REPO_ROOT))

    # Also emit a summary CSV that chapter 3 can cite.
    summary_path = out_path.parent / "class_distribution.csv"
    counts.rename_axis("class_id").to_frame("sample_count").to_csv(summary_path)
    log.info("Wrote %s", summary_path.relative_to(REPO_ROOT))


# ---------------------------------------------------------------------------
# Figure 3.3 — directional signs under horizontal flip
# ---------------------------------------------------------------------------
# Class indices chosen to illustrate the flip problem:
#   33 = turn right only
#   17 = no entry (visually symmetric but semantically asymmetric)
#   34 = turn left only
FLIP_CLASSES = [33, 17, 34]
FLIP_CAPTIONS = ["Turn right only (33)", "No entry (17)", "Turn left only (34)"]


def make_directional_flips(out_path: Path) -> None:
    """Grid showing three directional signs and their horizontal flips."""

    rng = random.Random(SEED + 1)
    fig, axes = plt.subplots(2, 3, figsize=(9, 6))
    fig.suptitle(
        "Figure 3.3 — Directional signs under horizontal flip",
        fontsize=13,
        y=0.99,
    )

    for col, (cls, caption) in enumerate(zip(FLIP_CLASSES, FLIP_CAPTIONS, strict=True)):
        class_dir = TRAIN_DIR / f"{cls:05d}"
        img = _random_image_for_class(class_dir, rng).resize((96, 96))
        flipped = ImageOps.mirror(img)

        axes[0, col].imshow(img)
        axes[0, col].set_title(f"{caption}\noriginal", fontsize=10)
        axes[1, col].imshow(flipped)
        axes[1, col].set_title("horizontally flipped", fontsize=10)
        for ax in (axes[0, col], axes[1, col]):
            ax.set_xticks([])
            ax.set_yticks([])

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Wrote %s", out_path.relative_to(REPO_ROOT))


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
def main() -> int:
    _ensure_dataset_present()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    np.random.seed(SEED)
    sns.set_theme(style="whitegrid", context="notebook")

    make_sample_grid(OUT_DIR / "sample_grid.png")
    make_class_distribution(OUT_DIR / "class_distribution.png")
    make_directional_flips(OUT_DIR / "directional_flips.png")

    log.info("All figures written to %s", OUT_DIR.relative_to(REPO_ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

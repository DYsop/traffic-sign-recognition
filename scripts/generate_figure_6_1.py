"""Generate Figure 6.1: augmentation examples grid.

Produces a 2x3 grid showing one GTSRB image on the left (original, unaugmented)
and five augmented versions on the right, demonstrating the range of photometric
and geometric variability introduced by the training-time augmentation pipeline
documented in chapter 6.

The image used is class 14 (stop sign). A medium-sized, canonical sample is
selected to make the augmentations clearly visible.

Output: reports/figures/dataset/augmentation_examples.png
Referenced from: docs/06_training_setup.md § 6.3.1, Figure 6.1
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from PIL import Image
from torchvision import transforms

SEED = 42
TARGET_CLASS = 14  # Stop sign
N_AUGMENTATIONS = 5
OUTPUT_PATH = Path("reports/figures/dataset/augmentation_examples.png")


def build_augmentation_pipeline() -> transforms.Compose:
    """The augmentation pipeline documented in chapter 6.3.

    Matches configs/*.yaml exactly, minus the final ToTensor + Normalize
    (we want PIL images for display).
    """
    return transforms.Compose(
        [
            transforms.Resize((48, 48)),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
            ),
            transforms.RandomAffine(
                degrees=15, translate=(0.1, 0.1), shear=10
            ),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
        ]
    )


def find_sample_image(data_root: Path, target_class: int) -> Path:
    """Locate a stop sign (class 14) image.

    GTSRB Final_Training has class folders named 00000, 00001, ..., 00042.
    We pick a mid-range image index within the target class folder for
    visual clarity.
    """
    class_dir = data_root / f"{target_class:05d}"
    if not class_dir.exists():
        raise FileNotFoundError(
            f"Class directory not found: {class_dir}\n"
            f"Expected layout: {data_root}/00000/, {data_root}/00001/, ..."
        )

    ppm_files = sorted(class_dir.glob("*.ppm"))
    if not ppm_files:
        raise FileNotFoundError(f"No .ppm images found in {class_dir}")

    # Pick a middle image deterministically — usually of better quality
    # than the first or last files in a GTSRB track sequence
    return ppm_files[len(ppm_files) // 2]


def render_figure(original: Image.Image, augmented: list[Image.Image]) -> plt.Figure:
    """Render a 2x3 grid: original + 5 augmentations."""
    fig, axes = plt.subplots(2, 3, figsize=(9, 6), dpi=120)
    fig.suptitle(
        "Figure 6.1 — Augmentation examples (class 14, stop sign)",
        fontsize=12,
        fontweight="bold",
        y=0.99,
    )

    # Top-left: original
    axes[0, 0].imshow(original)
    axes[0, 0].set_title("Original", fontsize=10)
    axes[0, 0].axis("off")

    # Remaining five panels: augmented versions
    flat_axes = [axes[0, 1], axes[0, 2], axes[1, 0], axes[1, 1], axes[1, 2]]
    for i, (ax, img) in enumerate(zip(flat_axes, augmented, strict=True)):
        ax.imshow(img)
        ax.set_title(f"Augmented #{i + 1}", fontsize=10)
        ax.axis("off")

    fig.tight_layout()
    fig.subplots_adjust(top=0.92)
    return fig


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/raw/GTSRB/Final_Training/Images"),
        help="Path to GTSRB Final_Training/Images directory",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_PATH,
        help="Destination PNG path",
    )
    args = parser.parse_args()

    # Deterministic augmentation output
    random.seed(SEED)
    torch.manual_seed(SEED)

    sample_path = find_sample_image(args.data_root, TARGET_CLASS)
    print(f"Using sample: {sample_path}")

    original = Image.open(sample_path).convert("RGB")
    original_resized = original.resize((48, 48), Image.BILINEAR)

    pipeline = build_augmentation_pipeline()
    augmented = [pipeline(original) for _ in range(N_AUGMENTATIONS)]

    fig = render_figure(original_resized, augmented)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight", dpi=150)
    plt.close(fig)

    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

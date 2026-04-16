"""Reshape the flat GTSRB test folder into a per-class ImageFolder layout.

The official GTSRB test download places all 12 630 images in one flat folder
plus a CSV of labels. `torchvision.datasets.ImageFolder` expects a
``<class_id>/<image>.ppm`` layout. This script moves the files into place.

Usage
-----
    python scripts/prepare_test_set.py \\
        --test-dir data/raw/GTSRB/Final_Test/Images \\
        --annotations data/raw/GTSRB/GT-final_test.csv
"""

from __future__ import annotations

import argparse
import csv
import logging
import shutil
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--test-dir",
        type=Path,
        required=True,
        help="Folder containing the flat list of test images.",
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        required=True,
        help="CSV with columns Filename;...;ClassId (official GTSRB format).",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if not args.test_dir.is_dir():
        raise SystemExit(f"Test dir not found: {args.test_dir}")
    if not args.annotations.is_file():
        raise SystemExit(f"Annotations CSV not found: {args.annotations}")

    moved = 0
    with args.annotations.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            filename = row["Filename"]
            class_id = int(row["ClassId"])
            src = args.test_dir / filename
            if not src.is_file():
                logging.warning("skip missing %s", src)
                continue
            dst_dir = args.test_dir / f"{class_id:05d}"
            dst = dst_dir / filename
            if args.dry_run:
                logging.info("would move %s -> %s", src, dst)
            else:
                dst_dir.mkdir(parents=True, exist_ok=True)
                shutil.move(str(src), str(dst))
            moved += 1

    logging.info("done — %d files %s", moved, "would be moved" if args.dry_run else "moved")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Train all three models and emit a single comparison report.

This is the script that produces the tables and figures shown in the README.
It expects the GTSRB data to be in place and simply invokes the library code
with one of the shipped YAML configs for each model.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch

from traffic_signs.config import load_config
from traffic_signs.data.gtsrb import build_dataloaders
from traffic_signs.evaluation.metrics import evaluate
from traffic_signs.evaluation.reports import (
    build_comparison_table,
    plot_confusion_matrix,
    plot_training_curves,
    save_classification_report,
)
from traffic_signs.models.registry import build_model
from traffic_signs.training.trainer import Trainer
from traffic_signs.utils.device import resolve_device
from traffic_signs.utils.logging_setup import configure_logging
from traffic_signs.utils.seed import set_seed

CONFIGS = [
    Path("configs/traffic_sign_net.yaml"),
    Path("configs/traffic_sign_net_stn.yaml"),
    Path("configs/deep_traffic_net.yaml"),
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", type=Path, nargs="*", default=CONFIGS)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    configure_logging(args.log_level, log_file=Path("logs/train_all.log"))
    log = logging.getLogger(__name__)

    results = []
    for cfg_path in args.configs:
        log.info("=" * 70)
        log.info("Starting %s", cfg_path)
        log.info("=" * 70)
        cfg = load_config(cfg_path)
        set_seed(cfg.seed)
        device = resolve_device(cfg.device)

        train_loader, val_loader, test_loader, class_names = build_dataloaders(
            cfg.data, cfg.train, seed=cfg.seed
        )
        model = build_model(cfg.model, num_classes=len(class_names), image_size=cfg.data.image_size)
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            cfg=cfg.train,
            device=device,
            model_name=cfg.model,
            checkpoint_dir=cfg.output.checkpoints_dir,
        )
        history = trainer.fit()
        history.to_json(cfg.output.metrics_dir / cfg.model / "training_history.json")
        plot_training_curves(history, cfg.output.figures_dir / cfg.model / "training_curves.png")

        best_ckpt = cfg.output.checkpoints_dir / cfg.model / "best.pt"
        state = torch.load(best_ckpt, map_location=device)
        model.load_state_dict(state["state_dict"])

        result = evaluate(
            model, test_loader, device=device, model_name=cfg.model, class_names=class_names
        )
        result.to_json(cfg.output.metrics_dir / cfg.model / "test_metrics.json")
        plot_confusion_matrix(result, cfg.output.figures_dir / cfg.model / "confusion_matrix.png")
        save_classification_report(
            result, cfg.output.metrics_dir / cfg.model / "classification_report.csv"
        )
        results.append(result)

    # Persist the global comparison
    df = build_comparison_table(results)
    out_dir = Path("reports/metrics")
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "comparison.csv", index=False, float_format="%.4f")
    (out_dir / "comparison.json").write_text(
        json.dumps(df.to_dict(orient="records"), indent=2), encoding="utf-8"
    )
    log.info("\n%s", df.to_string(index=False))
    log.info("Comparison table written to reports/metrics/comparison.{csv,json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

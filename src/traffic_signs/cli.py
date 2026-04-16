"""Command-line entry points.

Exposes three commands, wired up via ``[project.scripts]`` in ``pyproject.toml``::

    traffic-signs-train   --config configs/deep_traffic_net.yaml
    traffic-signs-eval    --config configs/deep_traffic_net.yaml --checkpoint checkpoints/deep_traffic_net/best.pt
    traffic-signs-predict --checkpoint checkpoints/deep_traffic_net/best.pt --image path/to/sign.png
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
from traffic_signs.inference.predict import load_predictor
from traffic_signs.models.registry import build_model, list_models
from traffic_signs.training.trainer import Trainer
from traffic_signs.utils.device import resolve_device
from traffic_signs.utils.logging_setup import configure_logging
from traffic_signs.utils.seed import set_seed

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------
def train_cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="traffic-signs-train")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    configure_logging(args.log_level, log_file=Path("logs/train.log"))
    cfg = load_config(args.config)
    set_seed(cfg.seed)
    device = resolve_device(cfg.device)

    train_loader, val_loader, test_loader, class_names = build_dataloaders(
        cfg.data, cfg.train, seed=cfg.seed
    )
    model = build_model(
        cfg.model,
        num_classes=len(class_names),
        image_size=cfg.data.image_size,
    )
    log.info(
        "Model %s has %d trainable parameters",
        cfg.model,
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    )

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

    # Persist history + curves + class names
    history_path = cfg.output.metrics_dir / cfg.model / "training_history.json"
    history.to_json(history_path)
    plot_training_curves(history, cfg.output.figures_dir / cfg.model / "training_curves.png")

    class_path = cfg.output.checkpoints_dir / cfg.model / "class_names.json"
    class_path.write_text(json.dumps(class_names, indent=2), encoding="utf-8")

    # Final test-set evaluation with the BEST checkpoint, not last
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

    log.info("Training complete. Test accuracy = %.4f", result.accuracy)
    return 0


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------
def evaluate_cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="traffic-signs-eval")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    configure_logging(args.log_level)
    cfg = load_config(args.config)
    set_seed(cfg.seed)
    device = resolve_device(cfg.device)

    _, _, test_loader, class_names = build_dataloaders(cfg.data, cfg.train, seed=cfg.seed)
    model = build_model(cfg.model, num_classes=len(class_names), image_size=cfg.data.image_size)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state.get("state_dict", state))

    result = evaluate(
        model, test_loader, device=device, model_name=cfg.model, class_names=class_names
    )
    result.to_json(cfg.output.metrics_dir / cfg.model / "test_metrics.json")
    plot_confusion_matrix(result, cfg.output.figures_dir / cfg.model / "confusion_matrix.png")
    save_classification_report(
        result, cfg.output.metrics_dir / cfg.model / "classification_report.csv"
    )

    print(build_comparison_table([result]).to_string(index=False))
    return 0


# ---------------------------------------------------------------------------
# predict
# ---------------------------------------------------------------------------
def predict_cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="traffic-signs-predict")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--model", choices=list_models(), default="deep_traffic_net")
    parser.add_argument("--num-classes", type=int, default=43)
    parser.add_argument("--image-size", type=int, default=48)
    parser.add_argument(
        "--class-names",
        type=Path,
        default=None,
        help="Optional path to class_names.json produced at training time.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args(argv)

    configure_logging("INFO")
    device = resolve_device(args.device)
    predictor = load_predictor(
        args.checkpoint,
        model_name=args.model,
        num_classes=args.num_classes,
        image_size=args.image_size,
        class_names_file=args.class_names,
        device=device,
    )
    pred = predictor.predict(args.image, top_k=args.top_k)
    print(
        json.dumps(
            {
                "class_id": pred.class_id,
                "class_name": pred.class_name,
                "probability": round(pred.probability, 4),
                "top_k": [{"class": c, "probability": round(p, 4)} for c, p in pred.top_k],
            },
            indent=2,
        )
    )
    return 0

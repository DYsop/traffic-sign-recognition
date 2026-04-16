"""Generic PyTorch trainer.

Replaces three copy-pasted training loops from the notebook with a single,
tested implementation that supports:

* early stopping on validation loss
* optional plateau / cosine LR scheduling
* JSON-serialisable training history (no pickles)
* deterministic checkpoint naming — ``checkpoints/<model>/best.pt`` and
  ``checkpoints/<model>/last.pt``, rather than auto-incrementing versions in
  the working directory.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
from torch import nn, optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from traffic_signs.config import TrainConfig

log = logging.getLogger(__name__)


@dataclass
class EpochRecord:
    epoch: int
    train_loss: float
    train_acc: float
    val_loss: float
    val_acc: float
    lr: float
    seconds: float


@dataclass
class TrainingHistory:
    model: str
    records: list[EpochRecord] = field(default_factory=list)
    total_seconds: float = 0.0
    best_val_loss: float = float("inf")
    best_val_acc: float = 0.0
    best_epoch: int = -1

    def to_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": self.model,
            "total_seconds": self.total_seconds,
            "best_val_loss": self.best_val_loss,
            "best_val_acc": self.best_val_acc,
            "best_epoch": self.best_epoch,
            "records": [asdict(r) for r in self.records],
        }
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        log.info("Training history written to %s", path)


class Trainer:
    """Single-model, single-GPU trainer."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        *,
        cfg: TrainConfig,
        device: torch.device,
        model_name: str,
        checkpoint_dir: Path,
    ) -> None:
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg
        self.device = device
        self.model_name = model_name
        self.checkpoint_dir = Path(checkpoint_dir) / model_name
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------
    def _build_optimizer(self) -> optim.Optimizer:
        params = self.model.parameters()
        if self.cfg.optimizer == "adam":
            return optim.Adam(params, lr=self.cfg.learning_rate, weight_decay=self.cfg.weight_decay)
        if self.cfg.optimizer == "adamw":
            return optim.AdamW(
                params, lr=self.cfg.learning_rate, weight_decay=self.cfg.weight_decay
            )
        if self.cfg.optimizer == "sgd":
            return optim.SGD(
                params,
                lr=self.cfg.learning_rate,
                momentum=0.9,
                weight_decay=self.cfg.weight_decay,
                nesterov=True,
            )
        raise ValueError(f"Unknown optimizer {self.cfg.optimizer!r}")

    def _build_scheduler(self) -> object | None:
        if self.cfg.scheduler == "plateau":
            return ReduceLROnPlateau(self.optimizer, mode="min", factor=0.5, patience=3)
        if self.cfg.scheduler == "cosine":
            return CosineAnnealingLR(self.optimizer, T_max=self.cfg.epochs)
        return None

    # ------------------------------------------------------------------
    # Epoch loops
    # ------------------------------------------------------------------
    def _run_epoch(self, loader: DataLoader, *, train: bool) -> tuple[float, float]:
        self.model.train(mode=train)
        loss_sum = 0.0
        correct = 0
        total = 0
        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            for images, labels in loader:
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                logits = self.model(images)
                loss = self.criterion(logits, labels)

                if train:
                    self.optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    self.optimizer.step()

                loss_sum += float(loss.item()) * labels.size(0)
                correct += int((logits.argmax(1) == labels).sum().item())
                total += labels.size(0)

        return loss_sum / total, correct / total

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def fit(self) -> TrainingHistory:
        history = TrainingHistory(model=self.model_name)
        patience_counter = 0
        start_all = time.perf_counter()

        for epoch in tqdm(range(1, self.cfg.epochs + 1), desc=f"Training {self.model_name}"):
            t0 = time.perf_counter()
            train_loss, train_acc = self._run_epoch(self.train_loader, train=True)
            val_loss, val_acc = self._run_epoch(self.val_loader, train=False)
            dt = time.perf_counter() - t0

            # LR scheduler
            if isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step(val_loss)
            elif self.scheduler is not None:
                self.scheduler.step()

            lr = self.optimizer.param_groups[0]["lr"]
            history.records.append(
                EpochRecord(
                    epoch=epoch,
                    train_loss=train_loss,
                    train_acc=train_acc,
                    val_loss=val_loss,
                    val_acc=val_acc,
                    lr=lr,
                    seconds=dt,
                )
            )
            log.info(
                "epoch=%02d  train_loss=%.4f  train_acc=%.4f  val_loss=%.4f  val_acc=%.4f  lr=%.2e  t=%.1fs",
                epoch,
                train_loss,
                train_acc,
                val_loss,
                val_acc,
                lr,
                dt,
            )

            # Checkpointing + early stopping
            improved = val_loss < history.best_val_loss
            if improved:
                history.best_val_loss = val_loss
                history.best_val_acc = val_acc
                history.best_epoch = epoch
                self._save_checkpoint("best.pt", epoch=epoch, val_loss=val_loss, val_acc=val_acc)
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.cfg.early_stopping_patience:
                    log.info(
                        "Early stopping at epoch %d (patience=%d).",
                        epoch,
                        self.cfg.early_stopping_patience,
                    )
                    break

        # Always save the last state too
        self._save_checkpoint(
            "last.pt",
            epoch=history.records[-1].epoch,
            val_loss=history.records[-1].val_loss,
            val_acc=history.records[-1].val_acc,
        )
        history.total_seconds = time.perf_counter() - start_all
        return history

    # ------------------------------------------------------------------
    def _save_checkpoint(self, name: str, *, epoch: int, val_loss: float, val_acc: float) -> None:
        path = self.checkpoint_dir / name
        torch.save(
            {
                "model_name": self.model_name,
                "state_dict": self.model.state_dict(),
                "epoch": epoch,
                "val_loss": val_loss,
                "val_acc": val_acc,
            },
            path,
        )
        log.debug("Saved checkpoint to %s", path)

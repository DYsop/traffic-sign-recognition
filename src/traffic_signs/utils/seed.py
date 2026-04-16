"""Seed management for reproducible runs.

The original notebook sets no seed at all — training results therefore drift
between runs. ``set_seed`` seeds Python, NumPy and PyTorch (CPU + CUDA) and
flips cuDNN into deterministic mode. A tiny throughput cost, but worth it
for a public repo where results must be reproducible.
"""

from __future__ import annotations

import logging
import os
import random

import numpy as np

log = logging.getLogger(__name__)


def set_seed(seed: int, *, deterministic: bool = True) -> None:
    """Seed all RNGs used in the project.

    Parameters
    ----------
    seed
        Integer seed, typically from :class:`ExperimentConfig`.
    deterministic
        If ``True`` (default) configure cuDNN for deterministic behaviour.
        Set to ``False`` on production training runs where a ~5-10 % speedup
        matters more than bit-exact reproducibility.
    """

    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        else:
            torch.backends.cudnn.benchmark = True
    except ImportError:  # pragma: no cover - torch is a hard dep in practice
        log.warning("PyTorch not available; skipping torch seeding.")

    log.info("Random seed set to %d (deterministic=%s)", seed, deterministic)

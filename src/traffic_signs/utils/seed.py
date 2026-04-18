"""Seed management + Blackwell GPU compatibility switches."""

from __future__ import annotations

import logging
import os
import random

import numpy as np

log = logging.getLogger(__name__)


def set_seed(seed: int, *, deterministic: bool = True) -> None:
    """Seed all RNGs + configure PyTorch for stable Blackwell execution."""

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

        # -----------------------------------------------------------------
        # Blackwell (sm_120) compatibility switches for PyTorch 2.11 nightly
        # -----------------------------------------------------------------
        # Problem: the cuBLAS/cuDNN kernels shipped with current nightly
        # PyTorch builds hit intermittent execution failures on RTX 50-series
        # GPUs for certain fused ops (BN, GEMM autoselected to TF32).
        # Workarounds:
        #   1. Disable cuDNN -> fall back to native CUDA kernels.
        #   2. Force FP32 matmul precision -> avoid flaky TF32 kernel paths.
        # Remove these once a stable cu128 build is published.
        if os.environ.get("TRAFFIC_SIGNS_DISABLE_CUDNN") == "1":
            torch.backends.cudnn.enabled = False
            log.warning("cuDNN disabled via TRAFFIC_SIGNS_DISABLE_CUDNN=1")

        if os.environ.get("TRAFFIC_SIGNS_FORCE_FP32") == "1":
            torch.set_float32_matmul_precision("highest")
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            log.warning("TF32 disabled, FP32 matmul forced")

    except ImportError:
        log.warning("PyTorch not available; skipping torch seeding.")

    log.info("Random seed set to %d (deterministic=%s)", seed, deterministic)
"""Device resolution with a sensible 'auto' default."""

from __future__ import annotations

import logging

import torch

log = logging.getLogger(__name__)


def resolve_device(preference: str = "auto") -> torch.device:
    """Return a ``torch.device`` based on the user's preference.

    ``auto`` picks CUDA if available, otherwise MPS (Apple Silicon), otherwise
    CPU. An explicit preference is honoured but only if the device is usable;
    asking for ``cuda`` on a CPU-only host raises a clear error rather than
    silently degrading.
    """

    preference = preference.lower()

    if preference == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    elif preference == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")
        device = torch.device("cuda")
    elif preference == "mps":
        if not (getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()):
            raise RuntimeError("MPS requested but not available on this system.")
        device = torch.device("mps")
    elif preference == "cpu":
        device = torch.device("cpu")
    else:
        raise ValueError(f"Unknown device preference: {preference!r}")

    log.info("Using device: %s", device)
    return device

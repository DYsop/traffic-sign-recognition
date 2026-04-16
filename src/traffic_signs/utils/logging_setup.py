"""Centralised logging configuration.

Uses :mod:`rich` for readable console output and falls back to a plain handler
when rich is not installed. A single call to :func:`configure_logging` is made
by every CLI entry point.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path


def configure_logging(
    level: str = "INFO",
    log_file: Path | None = None,
) -> None:
    """Configure the root logger.

    Repeated calls are idempotent — any existing handlers on the root logger
    are replaced, which is important in notebooks where a kernel may re-import
    the module many times.
    """

    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)

    fmt = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    try:
        from rich.logging import RichHandler

        handler: logging.Handler = RichHandler(
            rich_tracebacks=True,
            markup=False,
            show_path=False,
        )
        handler.setFormatter(logging.Formatter("%(message)s"))
    except ImportError:  # pragma: no cover
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter(fmt, datefmt=datefmt))

    root.addHandler(handler)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
        root.addHandler(file_handler)

    root.setLevel(level.upper())

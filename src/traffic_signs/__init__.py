"""Traffic-sign recognition on GTSRB.

Public package entry point. Re-exports the most commonly used symbols so that
library users can write ``from traffic_signs import train`` instead of drilling
into submodules.
"""

from __future__ import annotations

__version__ = "0.2.0"

from traffic_signs.config import ExperimentConfig, load_config
from traffic_signs.utils.seed import set_seed

__all__ = ["ExperimentConfig", "__version__", "load_config", "set_seed"]

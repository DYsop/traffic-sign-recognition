"""Seed reproducibility."""

from __future__ import annotations

import numpy as np
import torch

from traffic_signs.utils.seed import set_seed


def test_numpy_reproducible():
    set_seed(123)
    a = np.random.rand(5)
    set_seed(123)
    b = np.random.rand(5)
    assert np.allclose(a, b)


def test_torch_reproducible():
    set_seed(7)
    a = torch.randn(4, 4)
    set_seed(7)
    b = torch.randn(4, 4)
    assert torch.allclose(a, b)

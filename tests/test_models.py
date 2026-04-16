"""Model forward-pass smoke tests."""

from __future__ import annotations

import pytest
import torch

from traffic_signs.models import build_model, list_models


@pytest.mark.parametrize("name", ["traffic_sign_net", "traffic_sign_net_stn", "deep_traffic_net"])
def test_forward_shape(name: str):
    model = build_model(name, num_classes=43, image_size=48)
    model.eval()
    x = torch.randn(2, 3, 48, 48)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (2, 43)


def test_all_registered_models_are_buildable():
    for name in list_models():
        model = build_model(name, num_classes=5, image_size=48)
        assert sum(p.numel() for p in model.parameters()) > 0


def test_unknown_model_raises():
    with pytest.raises(KeyError):
        build_model("nope", num_classes=43, image_size=48)

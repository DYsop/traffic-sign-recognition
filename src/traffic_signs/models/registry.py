"""Model registry — one place that maps a name to a constructor."""

from __future__ import annotations

from collections.abc import Callable

from torch import nn

from traffic_signs.models.deep_traffic_net import DeepTrafficNet
from traffic_signs.models.stn import TrafficSignNetSTN
from traffic_signs.models.traffic_sign_net import TrafficSignNet

_REGISTRY: dict[str, Callable[..., nn.Module]] = {
    "traffic_sign_net": TrafficSignNet,
    "traffic_sign_net_stn": TrafficSignNetSTN,
    "deep_traffic_net": DeepTrafficNet,
}


def list_models() -> list[str]:
    return sorted(_REGISTRY)


def build_model(
    name: str,
    *,
    num_classes: int,
    image_size: int,
    **kwargs: object,
) -> nn.Module:
    """Instantiate a model by name."""

    if name not in _REGISTRY:
        raise KeyError(f"Unknown model {name!r}. Available: {', '.join(list_models())}")
    ctor = _REGISTRY[name]
    return ctor(num_classes=num_classes, image_size=image_size, **kwargs)

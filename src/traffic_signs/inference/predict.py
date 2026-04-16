"""Single-image and batch inference from a trained checkpoint."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from traffic_signs.models.registry import build_model

log = logging.getLogger(__name__)

_EVAL_NORMALISE = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))


@dataclass
class Prediction:
    class_id: int
    class_name: str
    probability: float
    top_k: list[tuple[str, float]]


class Predictor:
    """Wraps a trained model and the transforms needed to feed it raw images."""

    def __init__(
        self,
        model: torch.nn.Module,
        class_names: list[str],
        image_size: int,
        device: torch.device,
    ) -> None:
        self.model = model.to(device).eval()
        self.class_names = class_names
        self.device = device
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                _EVAL_NORMALISE,
            ]
        )

    @torch.inference_mode()
    def predict(self, image: str | Path | Image.Image, top_k: int = 5) -> Prediction:
        if not isinstance(image, Image.Image):
            image = Image.open(image).convert("RGB")
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        logits = self.model(tensor)
        probs = torch.softmax(logits, dim=1)[0].cpu()
        topk = torch.topk(probs, k=min(top_k, probs.numel()))
        top_k_list = [
            (self.class_names[int(idx)], float(probs[int(idx)])) for idx in topk.indices.tolist()
        ]
        cls_id = int(probs.argmax().item())
        return Prediction(
            class_id=cls_id,
            class_name=self.class_names[cls_id],
            probability=float(probs[cls_id].item()),
            top_k=top_k_list,
        )


def load_predictor(
    checkpoint_path: str | Path,
    *,
    model_name: str,
    num_classes: int,
    image_size: int,
    class_names: list[str] | None = None,
    class_names_file: str | Path | None = None,
    device: str | torch.device = "cpu",
) -> Predictor:
    """Load a :class:`Predictor` from a checkpoint on disk."""

    if isinstance(device, str):
        device = torch.device(device)

    model = build_model(model_name, num_classes=num_classes, image_size=image_size)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state.get("state_dict", state))

    if class_names is None:
        if class_names_file is None:
            class_names = [f"class_{i:05d}" for i in range(num_classes)]
        else:
            class_names = json.loads(Path(class_names_file).read_text(encoding="utf-8"))

    return Predictor(model, class_names=class_names, image_size=image_size, device=device)

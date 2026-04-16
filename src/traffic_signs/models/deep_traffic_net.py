"""DeepTrafficNet — five-block CNN, adapted from the notebook."""

from __future__ import annotations

import torch
from torch import nn


class DeepTrafficNet(nn.Module):
    """Five convolutional blocks followed by a three-layer MLP head.

    The notebook used a ``DROPOUT_RATE`` module-level constant; here the rate
    is a constructor argument so that the model is self-contained and can be
    instantiated with different regularisation settings from a config file.
    """

    def __init__(
        self,
        num_classes: int = 43,
        image_size: int = 48,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        def block(in_ch: int, out_ch: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(
            block(3, 32),
            block(32, 64),
            block(64, 128),
            block(128, 256),
            block(256, 512),
        )
        flat = self._infer_flat(image_size)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def _infer_flat(self, image_size: int) -> int:
        with torch.no_grad():
            dummy = torch.zeros(1, 3, image_size, image_size)
            return int(self.features(dummy).view(1, -1).size(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))

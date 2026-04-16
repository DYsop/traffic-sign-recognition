"""TrafficSignNet — the Keras baseline from the notebook, ported to PyTorch.

Why port from Keras to PyTorch?
Mixing two deep-learning frameworks in one repository doubles the dependency
surface and gives users pointless friction. The baseline was simple enough
to translate 1-to-1, so we keep the architecture identical (same channel
widths, same dropout ladder, same 48×48 input) but the whole training stack
becomes uniformly PyTorch. The Keras baseline can still be reconstructed from
git history if someone insists — see CHANGELOG.md.
"""

from __future__ import annotations

import torch
from torch import nn


class TrafficSignNet(nn.Module):
    """Three-block CNN with BatchNorm and increasing dropout."""

    def __init__(self, num_classes: int = 43, image_size: int = 48) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
            nn.Dropout(0.2),
            #
            nn.Conv2d(32, 64, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.Dropout(0.3),
            #
            nn.Conv2d(64, 128, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
            nn.Dropout(0.4),
        )
        flattened = self._infer_flattened_size(image_size)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def _infer_flattened_size(self, image_size: int) -> int:
        with torch.no_grad():
            dummy = torch.zeros(1, 3, image_size, image_size)
            return int(self.features(dummy).view(1, -1).size(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))

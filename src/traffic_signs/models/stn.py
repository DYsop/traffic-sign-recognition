"""TrafficSignNetSTN — CNN with a Spatial Transformer Network front-end."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class TrafficSignNetSTN(nn.Module):
    """Spatial Transformer + 3-block CNN.

    Cleanup versus the notebook version
    -----------------------------------
    * Flattened sizes are computed with a single helper pass, not two.
    * ``_infer_flattened_size`` is explicit about the intermediate shape so
      that slight image-size changes don't silently break the model.
    * The localisation head is initialised to the identity transform as
      recommended in the original STN paper.
    """

    def __init__(self, num_classes: int = 43, image_size: int = 48) -> None:
        super().__init__()

        # ------------------------------------------------------------------
        # Localisation network (produces the 6 affine transformation params)
        # ------------------------------------------------------------------
        self.localization = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=7),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        loc_feat = self._infer_size(self.localization, image_size)

        self.fc_loc = nn.Sequential(
            nn.Linear(loc_feat, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 6),
        )
        # Init as identity transform (STN paper recommendation).
        self.fc_loc[-1].weight.data.zero_()
        self.fc_loc[-1].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

        # ------------------------------------------------------------------
        # Main classifier
        # ------------------------------------------------------------------
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            #
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            #
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        cls_feat = self._infer_size(self.features, image_size)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(cls_feat, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    @staticmethod
    def _infer_size(module: nn.Module, image_size: int) -> int:
        with torch.no_grad():
            dummy = torch.zeros(1, 3, image_size, image_size)
            return int(module(dummy).view(1, -1).size(1))

    def _stn(self, x: torch.Tensor) -> torch.Tensor:
        xs = self.localization(x).flatten(1)
        theta = self.fc_loc(xs).view(-1, 2, 3)
        grid = F.affine_grid(theta, x.size(), align_corners=False)
        return F.grid_sample(x, grid, align_corners=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._stn(x)
        x = self.features(x)
        return self.classifier(x)

# -*- coding: utf-8 -*-
"""
EfficientNetSteg — EfficientNet-B0 fine-tuned for image steganalysis.

Architecture
------------
ImageNet normalisation  : registered buffers (mean / std), applied in forward()
EfficientNet-B0         : pretrained on ImageNet1K, backbone features (1 280-dim).
Classifier head         : Dropout → Linear(1 280 → 512) → ReLU → Dropout →
                          Linear(512 → num_classes)

Raw RGB pixels (in [0, 1]) are passed directly to the backbone after standard
ImageNet normalisation.  No SRM preprocessing is used — EfficientNet is large
enough to learn noise-residual features from raw pixels, and feeding SRM
residuals into ImageNet-pretrained weights destroys the transfer learning
advantage entirely (as shown by the ALASKA2 Kaggle 3rd-place solution,
pfnet-research/kaggle-alaska2-3rd-place-solution).

Training guidance
-----------------
Use SGD with Nesterov momentum (lr≈0.01, momentum=0.9) rather than Adam.
The weak steganographic signal benefits from the consistent gradient
accumulation that SGD+momentum provides.

References
----------
Tan, M. & Le, Q. V., "EfficientNet: Rethinking Model Scaling for Convolutional
Neural Networks", ICML 2019.

Pfnet-research, "ALASKA2 3rd Place Solution" (Kaggle 2020) — no SRM,
raw RGB + ImageNet normalisation + SGD + CutMix.
"""

import torch
import torch.nn as nn

try:
    from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
    _WEIGHTS_ENUM = True
except ImportError:
    from torchvision.models import efficientnet_b0
    _WEIGHTS_ENUM = False

from ..base import BaseSteganalyzer


class EfficientNetSteg(BaseSteganalyzer):
    """
    EfficientNet-B0 steganalyzer trained on raw RGB pixels.

    Parameters
    ----------
    in_channels    : input image channels (must be 3)
    num_classes    : output logits (default 2: cover / stego)
    freeze_backbone: freeze EfficientNet backbone weights during training.
                     Useful for a short warm-up of the head; call
                     :meth:`unfreeze_backbone` to unlock all layers later.
    dropout        : dropout probability in the classifier head (default 0.4)
    """

    def __init__(
        self,
        in_channels: int  = 3,
        num_classes: int  = 2,
        freeze_backbone: bool = False,
        dropout: float = 0.4,
    ) -> None:
        super().__init__(in_channels=in_channels, num_classes=num_classes)

        # ── ImageNet normalisation (applied in forward, stays with the model) ──
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.register_buffer("img_mean", mean)
        self.register_buffer("img_std",  std)

        # ── EfficientNet-B0 backbone (pretrained) ────────────────────────────
        if _WEIGHTS_ENUM:
            backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        else:
            backbone = efficientnet_b0(pretrained=True)

        self.features = backbone.features       # outputs (N, 1280, H/32, W/32)
        self.avgpool  = backbone.avgpool        # AdaptiveAvgPool2d → (N, 1280, 1, 1)

        backbone_out = 1280

        # ── Custom steganalysis head ─────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(backbone_out, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout * 0.5),
            nn.Linear(512, num_classes),
        )

        self._init_head()

        if freeze_backbone:
            self.freeze_backbone()

    # ── Initialisation ────────────────────────────────────────────────────────

    def _init_head(self) -> None:
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    # ── Backbone freeze / unfreeze helpers ────────────────────────────────────

    def freeze_backbone(self) -> None:
        """Freeze all EfficientNet backbone parameters."""
        for p in self.features.parameters():
            p.requires_grad_(False)

    def unfreeze_backbone(self) -> None:
        """Unfreeze all EfficientNet backbone parameters."""
        for p in self.features.parameters():
            p.requires_grad_(True)

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.img_mean) / self.img_std   # ImageNet normalisation
        x = self.features(x)                      # (N, 1280, H/32, W/32)
        x = self.avgpool(x)                       # (N, 1280, 1, 1)
        x = torch.flatten(x, 1)                  # (N, 1280)
        return self.classifier(x)                 # (N, num_classes)

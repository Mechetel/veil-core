# -*- coding: utf-8 -*-
"""
Edge-Guided Dual-Stream U-Net Encoder.
Ji, Zhang, Lv – Applied Sciences 2025.
"""

from typing import List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ....models.base import BaseEncoder
from .attention  import MSMAModule
from .inception  import InceptionDMKModule
from .blocks     import ContractingBlock, BottleneckBlock, ExpandingBlock
from .fusion     import DenseMessageFusion
from .recurrent  import ConvGRUCell, PerturbationNetwork


class EdgeGuidedDualStreamUNetEncoder(BaseEncoder):
    """
    Edge-Guided Dual-Stream U-Net for Secure Image Steganography.
    Ji, Zhang, Lv – Applied Sciences, 2025.

    Architecture summary (Table 1 + Sections 3.1–3.4)
    --------------------------------------------------
    1. Sobel edge enhancement: E = C + α·‖∇C‖  (Eq. 1)
    2. Dual-stream contracting path
         Cover stream  (G1C–G5C): 3→16→32→64→128→128, each with MSMA
         Edge stream   (G1E–G5E): same channels, G1E–G4E plain, G5E+InceptionDMK
    3. Shared expanding path (G6–G10): cross-stream skip fusion at each scale
    4. Dense Block: fuse U-Net output (3 ch) with secret message M (D ch)
    5. ConvGRU iterative optimisation (T steps):
         Training  → returns list of T stego images for the weighted loss
         Inference → returns only the final stego image

    Parameters
    ----------
    data_depth  : bits per pixel (D)
    T           : GRU optimisation iterations (default 10)
    eta         : perturbation step size η (default 1.0)
    gamma       : iterative loss decay factor γ (default 0.8)
    alpha       : image-quality loss weight α (default 100.0)
    sobel_alpha : edge enhancement strength α in Eq. 1 (default 1.0)
    hidden_ch   : ConvGRU hidden channels (default 32)
    """

    def __init__(
        self,
        data_depth:  int,
        T:           int   = 10,
        eta:         float = 1.0,
        gamma:       float = 0.8,
        alpha:       float = 100.0,
        sobel_alpha: float = 1.0,
        hidden_ch:   int   = 32,
    ) -> None:
        super().__init__(data_depth)
        self.T           = T
        self.eta         = eta
        self.gamma       = gamma
        self.alpha       = alpha
        self.sobel_alpha = sobel_alpha

        # ── Fixed Sobel kernels ───────────────────────────────────────────
        Kx = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32
        ).view(1, 1, 3, 3)
        Ky = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32
        ).view(1, 1, 3, 3)
        self.register_buffer("Kx", Kx)
        self.register_buffer("Ky", Ky)

        # ── Dual-stream contracting path ─────────────────────────────────
        # Cover stream: every contracting block uses MSMA
        self.G1C = ContractingBlock(3,   16,  use_msma=True)
        self.G2C = ContractingBlock(16,  32,  use_msma=True)
        self.G3C = ContractingBlock(32,  64,  use_msma=True)
        self.G4C = ContractingBlock(64,  128, use_msma=True)
        self.G5C = BottleneckBlock (128, 128, use_msma=True, use_inception=False)

        # Edge stream: plain contracting; bottleneck uses InceptionDMK
        self.G1E = ContractingBlock(3,   16,  use_msma=False)
        self.G2E = ContractingBlock(16,  32,  use_msma=False)
        self.G3E = ContractingBlock(32,  64,  use_msma=False)
        self.G4E = ContractingBlock(64,  128, use_msma=False)
        self.G5E = BottleneckBlock (128, 128, use_msma=False, use_inception=True)

        # ── Shared expanding path ─────────────────────────────────────────
        # Channel accounting after skip fusion at each level:
        #   G6 input: G5E(128)           → 128 ch
        #   G7 input: G4E(128)+G6(128)+G4C(128) = concat(256)
        #   G8 input: G3E(64) +G7(64) +G3C(64)  = concat(128)
        #   G9 input: G2E(32) +G8(32) +G2C(32)  = concat(64)
        #  G10 input: G1E(16) +G9(16) +G1C(16)  = concat(32)
        self.G6  = ExpandingBlock(128, 128)
        self.G7  = ExpandingBlock(256, 64)
        self.G8  = ExpandingBlock(128, 32)
        self.G9  = ExpandingBlock(64,  16)
        self.G10 = nn.Sequential(
            nn.Conv2d(32, 3, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True),
        )

        # ── Dense message-fusion block ────────────────────────────────────
        UNET_OUT  = 3
        DENSE_OUT = 32
        self.dense_fusion = DenseMessageFusion(UNET_OUT, data_depth, out_ch=DENSE_OUT)

        # ── ConvGRU iterative optimisation ────────────────────────────────
        # GRU input: [δ(3), ∇δ(3), F(DENSE_OUT)] = 3+3+32 = 38
        gru_input_ch = 3 + 3 + DENSE_OUT
        self.gru_cell       = ConvGRUCell(gru_input_ch, hidden_ch)
        self.perturb_net    = PerturbationNetwork(hidden_ch)
        self._hidden_ch     = hidden_ch

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _sobel_enhance(self, image: torch.Tensor) -> torch.Tensor:
        """E = C + α·‖∇C‖  (Eq. 1).  Operates channel-wise."""
        N, C, H, W = image.shape
        x = image.view(N * C, 1, H, W)
        Gx = F.conv2d(x, self.Kx, padding=1)
        Gy = F.conv2d(x, self.Ky, padding=1)
        grad_mag = (Gx ** 2 + Gy ** 2 + 1e-8).sqrt().view(N, C, H, W)
        return (image + self.sobel_alpha * grad_mag).clamp(-1.0, 1.0)

    @staticmethod
    def _pad_to_multiple(x: torch.Tensor, multiple: int = 16):
        """Pad spatial dims to the next multiple of *multiple* (bottom/right).
        Returns (padded_tensor, (original_H, original_W)).
        """
        _, _, H, W = x.shape
        pH = (multiple - H % multiple) % multiple
        pW = (multiple - W % multiple) % multiple
        if pH or pW:
            x = F.pad(x, (0, pW, 0, pH), mode="reflect")
        return x, (H, W)

    def _unet_forward(self, cover: torch.Tensor,
                      edge: torch.Tensor) -> torch.Tensor:
        """Dual-stream U-Net; returns (N, 3, H, W) feature map."""
        # Contracting
        xC1, sC1 = self.G1C(cover);  xE1, sE1 = self.G1E(edge)
        xC2, sC2 = self.G2C(xC1);    xE2, sE2 = self.G2E(xE1)
        xC3, sC3 = self.G3C(xC2);    xE3, sE3 = self.G3E(xE2)
        xC4, sC4 = self.G4C(xC3);    xE4, sE4 = self.G4E(xE3)
        bC       = self.G5C(xC4);    bE       = self.G5E(xE4)

        # Expanding with cross-stream skip fusion
        g6 = self.G6(bE)
        g7 = self.G7(torch.cat([sC4, sE4 + g6],  dim=1))
        g8 = self.G8(torch.cat([sC3, sE3 + g7],  dim=1))
        g9 = self.G9(torch.cat([sC2, sE2 + g8],  dim=1))
        return self.G10(torch.cat([sC1, sE1 + g9], dim=1))

    def _iterative_refine(
        self,
        cover:     torch.Tensor,
        features:  torch.Tensor,
        training:  bool,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """GRU-based iterative perturbation refinement (Section 3.4)."""
        N, _, H, W = cover.shape
        delta  = torch.zeros_like(cover)
        h_t    = torch.zeros(N, self._hidden_ch, H, W,
                             device=cover.device, dtype=cover.dtype)
        stego_list: List[torch.Tensor] = []

        for _ in range(self.T):
            grad_delta = torch.zeros_like(delta)
            x_t = torch.cat([delta, grad_delta, features], dim=1)
            h_t = self.gru_cell(x_t, h_t)
            delta = delta + self.eta * self.perturb_net(h_t)
            S_t = (cover + delta).clamp(-1.0, 1.0)
            if training:
                stego_list.append(S_t)

        return stego_list if training else S_t

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        cover:    torch.Tensor,
        payload:  torch.Tensor,
        training: bool = None,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        Parameters
        ----------
        cover    : (N, 3, H, W) cover image in [-1, 1]
        payload  : (N, D, H, W) binary payload in {0, 1}
        training : override train/eval mode (None → uses self.training)

        Returns
        -------
        training=True  → List[Tensor(N,3,H,W)]  length T
        training=False → Tensor(N,3,H,W)
        """
        if training is None:
            training = self.training

        # Pad to a multiple of 16 so every max-pool/deconv pair is exact;
        # crop back to the original size after the U-Net.
        cover_p, (H, W) = self._pad_to_multiple(cover)
        payload_p, _    = self._pad_to_multiple(payload)

        edge     = self._sobel_enhance(cover_p)
        unet_out = self._unet_forward(cover_p, edge)
        features = self.dense_fusion(unet_out, payload_p)
        result   = self._iterative_refine(cover_p, features, training)

        # Crop padding away
        if isinstance(result, list):
            return [s[..., :H, :W] for s in result]
        return result[..., :H, :W]

# -*- coding: utf-8 -*-
"""
Edge-Aware DenseASPP Encoder for image steganography.

Novel architecture combining three pillars:
  (A) EdgeNet       — lightweight learned edge detector producing soft edge maps
  (B) DenseASPP     — multi-scale feature backbone at full resolution with MSMA
  (C) Edge-Masked   — ConvGRU iterative refinement with edge-masked perturbations
      Refinement      that concentrate data embedding in edge regions

The key novelty is the edge-masked iterative refinement: at each GRU step,
the raw perturbation is element-wise multiplied by an edge mask derived from
EdgeNet, forcing the encoder to embed steganographic data primarily in edge
regions where human vision is least sensitive to modifications.

An epsilon floor (default 0.05) on the mask prevents dead gradients in flat
regions while still directing 95% of the perturbation budget to edges.
"""

from typing import List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from ...base import BaseEncoder
from .recurrent import ConvGRUCell, PerturbationNetwork
from .edge_net import EdgeNet
from .dense_aspp import DenseASPPBackbone
from .edge_fusion import EdgeAwareDenseMessageFusion


class EdgeAwareDenseASPPEncoder(BaseEncoder):
    """
    Edge-Aware DenseASPP Encoder.

    Architecture summary
    --------------------
    1. EdgeNet:      cover → edge_map (N, 1, H, W) in [0, 1]
    2. DenseASPP:    cover → features (N, 64, H, W) with MSMA + InceptionDMK
    3. Edge Fusion:  (features, payload, edge_map) → fused (N, 64, H, W)
    4. Iterative:    ConvGRU refinement with edge-masked perturbations
         Training  → returns list of T stego images for weighted loss
         Inference → returns only the final stego image

    Parameters
    ----------
    data_depth  : bits per pixel (D)
    T           : GRU optimisation iterations (default 8)
    eta         : perturbation step size η (default 1.0)
    gamma       : iterative loss decay factor γ (default 0.8)
    alpha       : image-quality loss weight α (default 100.0)
    hidden_ch   : ConvGRU hidden channels (default 48)
    edge_epsilon: minimum mask value to prevent dead gradients (default 0.05)
    """

    def __init__(
        self,
        data_depth:   int,
        T:            int   = 8,
        eta:          float = 1.0,
        gamma:        float = 0.8,
        alpha:        float = 100.0,
        hidden_ch:    int   = 48,
        edge_epsilon: float = 0.05,
    ) -> None:
        super().__init__(data_depth)
        self.T            = T
        self.eta          = eta
        self.gamma        = gamma
        self.alpha        = alpha
        self.edge_epsilon = edge_epsilon
        self._hidden_ch   = hidden_ch

        # ── Pillar A: Learned edge detector ─────────────────────────────
        self.edge_net = EdgeNet()

        # ── Pillar B: DenseASPP feature backbone ────────────────────────
        feat_ch = 64
        self.backbone = DenseASPPBackbone(in_ch=3, out_ch=feat_ch)

        # ── Edge-aware dense message fusion ─────────────────────────────
        self.fusion = EdgeAwareDenseMessageFusion(
            feat_ch=feat_ch,
            data_depth=data_depth,
            out_ch=feat_ch,
        )

        # ── Pillar C: Edge-masked ConvGRU iterative refinement ──────────
        # GRU input: [delta(3), grad_delta(3), features(feat_ch), edge_map(1)]
        gru_input_ch = 3 + 3 + feat_ch + 1
        self.gru_cell    = ConvGRUCell(gru_input_ch, hidden_ch)
        self.perturb_net = PerturbationNetwork(hidden_ch)

    # ── Helpers ──────────────────────────────────────────────────────────

    def _gru_step(
        self,
        delta:    torch.Tensor,
        features: torch.Tensor,
        edge_map: torch.Tensor,
        h_prev:   torch.Tensor,
    ) -> torch.Tensor:
        """One GRU step including x_t construction and perturbation.

        Wrapped as a single checkpoint unit so x_t (71ch × batch × H × W)
        is never stored between steps — it is reconstructed during backward.
        Saves ~297 MB × T per training step at batch=8, 360².
        Returns (h_t, raw_pert) packed as a tuple.
        """
        grad_delta = torch.zeros_like(delta)
        x_t        = torch.cat([delta, grad_delta, features, edge_map], dim=1)
        h_t        = self.gru_cell(x_t, h_prev)
        raw_pert   = self.perturb_net(h_t)
        return h_t, raw_pert

    def _iterative_refine(
        self,
        cover:    torch.Tensor,
        features: torch.Tensor,
        edge_map: torch.Tensor,
        training: bool,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        ConvGRU iterative refinement with edge-masked perturbations.

        At each step t:
          1. GRU receives [delta, grad_delta, features, edge_map]
          2. PerturbationNetwork produces raw perturbation
          3. Perturbation is masked: pert *= (ε + (1-ε) * edge_map)
          4. delta += η * masked_perturbation
          5. S_t = clamp(cover + delta, -1, 1)

        The epsilon floor ensures gradient flow through flat regions.
        """
        N, _, H, W = cover.shape
        delta = torch.zeros_like(cover)
        h_t   = torch.zeros(
            N, self._hidden_ch, H, W,
            device=cover.device, dtype=cover.dtype,
        )
        stego_list: List[torch.Tensor] = []

        # Pre-compute 3-channel edge mask with epsilon floor
        eps          = self.edge_epsilon
        edge_mask_3ch = eps + (1.0 - eps) * edge_map.expand(-1, 3, -1, -1)

        for _ in range(self.T):
            if training:
                # Single checkpoint boundary: x_t construction + GRU + perturb.
                # Inputs stored: delta(3ch), features(64ch), edge_map(1ch), h_prev(48ch).
                # NOT stored: x_t(71ch) — reconstructed on backward. Saves ~297 MB/step.
                h_t, raw_pert = checkpoint(
                    self._gru_step, delta, features, edge_map, h_t,
                    use_reentrant=False,
                )
            else:
                h_t, raw_pert = self._gru_step(delta, features, edge_map, h_t)

            masked_pert = raw_pert * edge_mask_3ch
            delta       = delta + self.eta * masked_pert
            S_t         = (cover + delta).clamp(-1.0, 1.0)

            if training:
                stego_list.append(S_t)

        return stego_list if training else S_t

    # ── Forward ──────────────────────────────────────────────────────────

    def forward(
        self,
        cover:   torch.Tensor,
        payload: torch.Tensor,
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
        training=True  → List[Tensor(N,3,H,W)]  length T  (+ edge_map stored)
        training=False → Tensor(N,3,H,W)
        """
        if training is None:
            training = self.training

        # 1. Learned edge detection
        edge_map = self.edge_net(cover)          # (N, 1, H, W)

        # 2. DenseASPP feature extraction with MSMA + InceptionDMK
        features = self.backbone(cover)          # (N, 64, H, W)

        # 3. Edge-aware dense message fusion
        fused = self.fusion(features, payload, edge_map)  # (N, 64, H, W)

        # 4. Edge-masked iterative refinement
        result = self._iterative_refine(cover, fused, edge_map, training)

        # Store edge_map for loss computation (edge regularisation)
        self._last_edge_map = edge_map

        return result

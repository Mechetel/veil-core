# -*- coding: utf-8 -*-
"""
Edge-aware dense message fusion module.

Extends the DenseMessageFusion pattern from Ji et al. (2025) with
explicit edge gating: the backbone features are multiplied by the
edge map to create edge-emphasised features, which are concatenated
alongside the original features, payload, and edge map.  Dense skip
connections ensure every fusion layer has direct access to all inputs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class EdgeAwareDenseMessageFusion(nn.Module):
    """
    Edge-aware dense message fusion.

    Architecture
    ------------
    F_edge = features * edge_map              (soft edge gating)
    base   = cat(features, F_edge, payload, edge_map)
           → (N, 2*feat_ch + data_depth + 1, H, W)

    h1: Conv(base_ch → out_ch) → BN → LReLU
    h2: Conv(out_ch + base_ch → out_ch) → BN → LReLU    [cat(h1, base)]
    h3: Conv(2*out_ch + base_ch → out_ch) → BN → LReLU  [cat(h2, h1, base)]

    Parameters
    ----------
    feat_ch    : number of backbone feature channels (default 64)
    data_depth : bits per pixel D
    out_ch     : output channels (default 64)

    Input : features (N, feat_ch, H, W), payload (N, D, H, W), edge_map (N, 1, H, W)
    Output: fused (N, out_ch, H, W)
    """

    def __init__(
        self,
        feat_ch: int = 64,
        data_depth: int = 1,
        out_ch: int = 64,
    ) -> None:
        super().__init__()
        # base = features (feat_ch) + edge-gated features (feat_ch) + payload (D) + edge_map (1)
        base_ch = 2 * feat_ch + data_depth + 1

        self.conv1 = nn.Sequential(
            nn.Conv2d(base_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_ch + base_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(2 * out_ch + base_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def _forward_impl(
        self,
        features: torch.Tensor,
        payload: torch.Tensor,
        edge_map: torch.Tensor,
    ) -> torch.Tensor:
        f_edge = features * edge_map
        base   = torch.cat([features, f_edge, payload, edge_map], dim=1)
        h1     = self.conv1(base)
        h2     = self.conv2(torch.cat([h1, base], dim=1))
        h3     = self.conv3(torch.cat([h2, h1, base], dim=1))
        return h3

    def forward(
        self,
        features: torch.Tensor,
        payload: torch.Tensor,
        edge_map: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        features : (N, feat_ch, H, W) backbone features
        payload  : (N, D, H, W) binary payload
        edge_map : (N, 1, H, W) edge probability map

        Returns
        -------
        fused : (N, out_ch, H, W)
        """
        if self.training and features.requires_grad:
            return checkpoint(self._forward_impl, features, payload, edge_map,
                              use_reentrant=False)
        return self._forward_impl(features, payload, edge_map)

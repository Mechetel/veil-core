# -*- coding: utf-8 -*-
"""
Convolutional GRU and perturbation network for iterative optimisation.
Ji, Zhang, Lv – Applied Sciences 2025, Section 3.4.
"""

import torch
import torch.nn as nn


class ConvGRUCell(nn.Module):
    """
    2-D Convolutional GRU cell.

    All gate operations use 3×3 spatial convolutions instead of
    fully-connected layers, preserving spatial structure.

    Parameters
    ----------
    input_ch  : channels of the input tensor x_t
    hidden_ch : channels of the hidden state h_t
    """

    def __init__(self, input_ch: int, hidden_ch: int) -> None:
        super().__init__()
        total = input_ch + hidden_ch
        self.conv_update = nn.Conv2d(total, hidden_ch, kernel_size=3, padding=1)
        self.conv_reset  = nn.Conv2d(total, hidden_ch, kernel_size=3, padding=1)
        self.conv_new    = nn.Conv2d(total, hidden_ch, kernel_size=3, padding=1)

    def forward(self, x_t: torch.Tensor,
                h_prev: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x_t    : (N, input_ch,  H, W) — current input
        h_prev : (N, hidden_ch, H, W) — previous hidden state

        Returns
        -------
        h_next : (N, hidden_ch, H, W)
        """
        combined = torch.cat([h_prev, x_t], dim=1)
        z = torch.sigmoid(self.conv_update(combined))       # update gate
        r = torch.sigmoid(self.conv_reset(combined))        # reset gate
        h_tilde = torch.tanh(self.conv_new(
            torch.cat([r * h_prev, x_t], dim=1)))           # candidate state
        return (1.0 - z) * h_prev + z * h_tilde


class PerturbationNetwork(nn.Module):
    """
    Maps the GRU hidden state to a 3-channel perturbation direction δ.

    Architecture: Conv3×3 → LeakyReLU → Conv3×3 → Tanh

    Parameters
    ----------
    hidden_ch : number of GRU hidden channels
    """

    def __init__(self, hidden_ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(hidden_ch, hidden_ch // 2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_ch // 2, 3, kernel_size=3, padding=1),
            nn.Tanh(),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """(N, hidden_ch, H, W) → (N, 3, H, W)"""
        return self.net(h)

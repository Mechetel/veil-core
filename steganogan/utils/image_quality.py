# -*- coding: utf-8 -*-
"""
Perceptual image quality metrics: SSIM implementation and a torch-friendly
module-level helper function.
"""

from math import exp
from typing import Tuple

import torch
import torch.nn.functional as F
from torch.nn.functional import conv2d


class SSIMCalculator:
    """
    Structural Similarity Index Measure (SSIM) between two image tensors.

    All methods are static; this class is a namespace for SSIM utilities.
    Reference: Wang et al., "Image quality assessment: from error visibility
    to structural similarity", IEEE TIP 2004.
    """

    @staticmethod
    def gaussian_kernel(window_size: int, sigma: float = 1.5) -> torch.Tensor:
        """
        Build a normalised 1-D Gaussian window.

        Parameters
        ----------
        window_size : number of taps
        sigma       : standard deviation
        """
        weights = torch.tensor(
            [exp(-(x - window_size // 2) ** 2 / (2 * sigma ** 2)) for x in range(window_size)]
        )
        return weights / weights.sum()

    @staticmethod
    def build_window(window_size: int, n_channels: int) -> torch.Tensor:
        """
        Expand a 1-D Gaussian into a 2-D separable filter for *n_channels*.

        Returns a tensor of shape (n_channels, 1, window_size, window_size).
        """
        k1d = SSIMCalculator.gaussian_kernel(window_size).unsqueeze(1)
        k2d = k1d.mm(k1d.t()).float().unsqueeze(0).unsqueeze(0)
        return k2d.expand(n_channels, 1, window_size, window_size).contiguous()

    @staticmethod
    def _compute(
        img1: torch.Tensor,
        img2: torch.Tensor,
        window: torch.Tensor,
        window_size: int,
        n_channels: int,
        reduce: bool = True,
    ) -> torch.Tensor:
        """Low-level SSIM computation (no window creation overhead)."""
        pad = window_size // 2

        mu1 = conv2d(img1, window, padding=pad, groups=n_channels)
        mu2 = conv2d(img2, window, padding=pad, groups=n_channels)
        mu1_sq, mu2_sq, mu1_mu2 = mu1.pow(2), mu2.pow(2), mu1 * mu2

        sigma1_sq = conv2d(img1 * img1, window, padding=pad, groups=n_channels) - mu1_sq
        sigma2_sq = conv2d(img2 * img2, window, padding=pad, groups=n_channels) - mu2_sq
        sigma12   = conv2d(img1 * img2, window, padding=pad, groups=n_channels) - mu1_mu2

        C1, C2 = 0.01 ** 2, 0.03 ** 2
        ssim_map = (
            ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2))
            / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        )
        return ssim_map.mean() if reduce else ssim_map.mean(1).mean(1).mean(1)

    @staticmethod
    def calculate(
        img1: torch.Tensor,
        img2: torch.Tensor,
        window_size: int = 11,
        reduce: bool = True,
    ) -> torch.Tensor:
        """
        Compute SSIM between two batches of images.

        Parameters
        ----------
        img1, img2  : (N, C, H, W) image tensors in the same value range
        window_size : Gaussian kernel size (default 11)
        reduce      : if True, return a scalar mean; else per-image scores
        """
        n_channels = img1.size(1)
        window = SSIMCalculator.build_window(window_size, n_channels)
        window = window.to(img1.device).type_as(img1)
        return SSIMCalculator._compute(img1, img2, window, window_size, n_channels, reduce)


# ── Convenience alias ─────────────────────────────────────────────────────────

def ssim(
    img1: torch.Tensor,
    img2: torch.Tensor,
    window_size: int = 11,
    size_average: bool = True,
) -> torch.Tensor:
    """Module-level SSIM helper (backward-compatible signature)."""
    return SSIMCalculator.calculate(img1, img2, window_size, reduce=size_average)


class FSIMCalculator:
    """
    Feature Similarity Index Measure (FSIM).

    Zhang et al., "FSIM: A Feature Similarity Index for Image Quality
    Assessment", IEEE TIP 2011.

    Phase congruency is approximated via Laplacian-of-Gaussian at two scales
    (σ=1, σ=2), normalised to [0, 1].  Gradient magnitude uses Sobel filters.
    Constants T1=0.85 and T2=160 follow Table I of the original paper.

    Input : (N, C, H, W) tensors in [-1, 1]
    Output: scalar in [0, 1]  (higher = more similar)
    """

    @staticmethod
    def _to_luminance(x: torch.Tensor) -> torch.Tensor:
        if x.size(1) == 1:
            return x
        w = x.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
        return (x * w).sum(dim=1, keepdim=True)

    @staticmethod
    def _gradient_magnitude(lum: torch.Tensor) -> torch.Tensor:
        kx = lum.new_tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                             dtype=torch.float32).view(1, 1, 3, 3)
        ky = lum.new_tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                             dtype=torch.float32).view(1, 1, 3, 3)
        gx = F.conv2d(lum, kx, padding=1)
        gy = F.conv2d(lum, ky, padding=1)
        return (gx.pow(2) + gy.pow(2)).sqrt()

    @staticmethod
    def _log_kernel(sigma: float, device: torch.device,
                    dtype: torch.dtype) -> torch.Tensor:
        size = int(6 * sigma + 1) | 1
        half = size // 2
        y, x = torch.meshgrid(
            torch.arange(-half, half + 1, dtype=dtype, device=device),
            torch.arange(-half, half + 1, dtype=dtype, device=device),
            indexing="ij",
        )
        r2 = x.pow(2) + y.pow(2)
        s2 = sigma ** 2
        k = (r2 / s2 - 2.0) * torch.exp(-r2 / (2.0 * s2))
        k = k - k.mean()
        return k.view(1, 1, size, size)

    @staticmethod
    def _phase_congruency(lum: torch.Tensor) -> torch.Tensor:
        pc = torch.zeros_like(lum)
        for sigma in (1.0, 2.0):
            k = FSIMCalculator._log_kernel(sigma, lum.device, lum.dtype)
            pc = pc + F.conv2d(lum, k, padding=k.shape[-1] // 2).abs()
        return pc

    @staticmethod
    def calculate(img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        img1, img2 : (N, C, H, W) in [-1, 1]

        Returns
        -------
        Scalar FSIM ∈ [0, 1].
        """
        # Scale to [0, 255] equivalent so T2=160 is meaningful
        l1 = FSIMCalculator._to_luminance(img1) * 127.5 + 127.5
        l2 = FSIMCalculator._to_luminance(img2) * 127.5 + 127.5

        pc1_raw = FSIMCalculator._phase_congruency(l1)
        pc2_raw = FSIMCalculator._phase_congruency(l2)
        pc_max  = torch.max(pc1_raw.max(), pc2_raw.max()).clamp(min=1e-8)
        pc1, pc2 = pc1_raw / pc_max, pc2_raw / pc_max   # normalise to [0, 1]

        gm1 = FSIMCalculator._gradient_magnitude(l1)
        gm2 = FSIMCalculator._gradient_magnitude(l2)

        T1, T2 = 0.85, 160.0
        S_pc = (2.0 * pc1 * pc2 + T1) / (pc1.pow(2) + pc2.pow(2) + T1)
        S_gm = (2.0 * gm1 * gm2 + T2) / (gm1.pow(2) + gm2.pow(2) + T2)

        pc_m = torch.max(pc1, pc2)
        return (S_pc * S_gm * pc_m).sum() / pc_m.sum().clamp(min=1e-8)


class WPSNRCalculator:
    """
    Weighted Peak Signal-to-Noise Ratio (WPSNR).

    Applies a local-variance perceptual mask: flat regions (where the HVS is
    most sensitive to distortion) receive high weight; textured/edge regions
    receive low weight (masking effect).

    WPSNR = 10 · log10(max_val² / WMSE)
    WMSE  = Σ[w · (x−y)²] / Σ[w]
    w     = 1 / (1 + local_variance(x))

    Input : (N, C, H, W) tensors in [-1, 1]
    Output: dB value  (higher = better)
    """

    @staticmethod
    def _local_variance(x: torch.Tensor, window: int = 7) -> torch.Tensor:
        n_ch = x.size(1)
        k    = x.new_ones(n_ch, 1, window, window) / (window * window)
        pad  = window // 2
        mu   = F.conv2d(x,        k, padding=pad, groups=n_ch)
        mu2  = F.conv2d(x.pow(2), k, padding=pad, groups=n_ch)
        return (mu2 - mu.pow(2)).clamp(min=0.0)

    @staticmethod
    def calculate(
        img1:    torch.Tensor,
        img2:    torch.Tensor,
        max_val: float = 2.0,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        img1, img2 : (N, C, H, W) in [-1, 1]
        max_val    : signal range; 2.0 for [-1, 1] images

        Returns
        -------
        WPSNR in dB.
        """
        w      = 1.0 / (1.0 + WPSNRCalculator._local_variance(img1))
        sq_err = (img1 - img2).pow(2)
        wmse   = (w * sq_err).sum() / w.sum().clamp(min=1e-8)
        max2   = img1.new_tensor(max_val ** 2)
        return 10.0 * torch.log10(max2 / wmse.clamp(min=1e-10))


def first_element(storage: object, loc: object) -> object:
    """Pickle map_location helper – returns the first argument."""
    return storage

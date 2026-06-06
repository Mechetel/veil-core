# -*- coding: utf-8 -*-
"""
Steganography training and validation metrics.
"""

from typing import Dict, Tuple

import torch
import torch.nn.functional as F

from ..utils.image_quality import SSIMCalculator, FSIMCalculator, WPSNRCalculator


class SteganographyMetrics:
    """
    Computes the standard suite of steganography evaluation metrics.

    All methods are static; this class is a pure-utility namespace.

    Metrics
    -------
    encoder_mse  : pixel-level MSE  between stego and cover  (↓ is better)
    decoder_loss : BCE loss between decoded logits and payload bits  (↓)
    decoder_acc  : fraction of bits decoded correctly  (↑)
    ssim         : structural similarity  (↑, max 1)
    psnr         : peak signal-to-noise ratio in dB  (↑)
    fsim         : feature similarity index measure  (↑, max 1)
    wpsnr        : weighted PSNR with local-variance masking in dB  (↑)
    rsbpp        : robust steganographic bits per pixel  (↑)
    """

    @staticmethod
    def coding_scores(
        cover:    torch.Tensor,
        stego:    torch.Tensor,
        payload:  torch.Tensor,
        decoded:  torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute encoder MSE, decoder BCE loss, and bit accuracy.

        Returns
        -------
        (encoder_mse, decoder_loss, decoder_acc)
        """
        enc_mse  = F.mse_loss(stego, cover)
        dec_loss = F.binary_cross_entropy_with_logits(decoded, payload.float())
        dec_acc  = (
            (decoded >= 0.0).eq(payload >= 0.5).sum().float() / payload.numel()
        )
        return enc_mse, dec_loss, dec_acc

    @staticmethod
    def full_report(
        cover:      torch.Tensor,
        stego:      torch.Tensor,
        payload:    torch.Tensor,
        decoded:    torch.Tensor,
        data_depth: int,
    ) -> Dict[str, float]:
        """
        Compute the full validation metric report.

        Returns a dict with keys:
        ``encoder_mse``, ``decoder_loss``, ``decoder_acc``,
        ``ssim``, ``psnr``, ``fsim``, ``wpsnr``, ``rsbpp``.
        """
        enc_mse, dec_loss, dec_acc = SteganographyMetrics.coding_scores(
            cover, stego, payload, decoded
        )
        return {
            "encoder_mse":  enc_mse.item(),
            "decoder_loss": dec_loss.item(),
            "decoder_acc":  dec_acc.item(),
            "ssim":         SSIMCalculator.calculate(cover, stego).item(),
            "psnr":         10.0 * torch.log10(4.0 / enc_mse).item(),
            "fsim":         FSIMCalculator.calculate(cover, stego).item(),
            "wpsnr":        WPSNRCalculator.calculate(cover, stego).item(),
            "rsbpp":        data_depth * (2.0 * dec_acc.item() - 1.0),
        }

# -*- coding: utf-8 -*-
"""
Steganography loss functions.

  SteganographyLoss        : standard single-step loss (encoder MSE + decoder BCE ± critic)
  IterativeLoss            : weighted multi-step loss for iterative encoders (Ji et al., Eq. 18)
  EdgeAwareIterativeLoss   : extends IterativeLoss with VGG perceptual loss + edge regularisation
  VGGPerceptualLoss        : feature-matching loss using frozen VGG-16
  SobelEdgeRegularisation  : soft consistency loss between learned edge map and Sobel magnitude
"""

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models


class SteganographyLoss:
    """
    Standard single-step steganography loss.

    L_total = λ_enc · MSE(S, C) + BCE(M', M) [- critic_score(S)]

    Parameters
    ----------
    encoder_weight : MSE loss scale factor (default 100)
    """

    def __init__(self, encoder_weight: float = 100.0) -> None:
        self.encoder_weight = encoder_weight

    def __call__(
        self,
        cover:      torch.Tensor,
        stego:      torch.Tensor,
        payload:    torch.Tensor,
        decoded:    torch.Tensor,
        gen_score:  Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        cover, stego    : (N,3,H,W) image tensors
        payload         : (N,D,H,W) ground-truth bits
        decoded         : (N,D,H,W) decoder output logits
        gen_score       : scalar critic score for *stego* (WGAN, optional)
        """
        enc_loss = F.mse_loss(stego, cover)
        dec_loss = F.binary_cross_entropy_with_logits(decoded, payload.float())
        total    = self.encoder_weight * enc_loss + dec_loss
        if gen_score is not None:
            total = total - gen_score
        return total


class IterativeLoss:
    """
    Weighted iterative loss for ConvGRU-based encoders (Eq. 18, Ji et al. 2025).

    L_total = Σ_{t=0}^{T-1}  γ^{T-1-t} · [L_D(M, M'_t) + α·L_E(C, S_t) + β·L_C(C, S_t)]

      L_D : BCE between recovered bits M'_t and ground-truth M
      L_E : pixel-level MSE  (image quality)
      L_C : Wasserstein proxy  critic(S_t) – critic(C)   [optional]

    More recent steps receive higher weight (γ^0 = 1 at t = T-1).

    Parameters
    ----------
    decoder    : decoder module (produces logits from a stego image)
    gamma      : per-step discount factor  (0 < γ ≤ 1, default 0.8)
    alpha      : image-quality loss weight (default 100.0)
    critic     : optional critic module for adversarial loss term
    """

    def __init__(
        self,
        decoder: nn.Module,
        gamma:   float = 0.8,
        alpha:   float = 100.0,
        critic:  Optional[nn.Module] = None,
    ) -> None:
        self.decoder = decoder
        self.gamma   = gamma
        self.alpha   = alpha
        self.critic  = critic

    def __call__(
        self,
        cover:      torch.Tensor,
        payload:    torch.Tensor,
        stego_list: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        cover      : (N,3,H,W) cover image
        payload    : (N,D,H,W) ground-truth bits
        stego_list : list of T stego tensors from the iterative encoder
        """
        T     = len(stego_list)
        total = torch.tensor(0.0, device=cover.device)

        # critic(cover) is the same for every step; compute it once under
        # no_grad (cover is a fixed input, not a learnable parameter).
        critic_cover_score: Optional[torch.Tensor] = None
        if self.critic is not None:
            with torch.no_grad():
                critic_cover_score = torch.mean(self.critic(cover))

        for t, S_t in enumerate(stego_list):
            weight  = self.gamma ** (T - 1 - t)     # most recent → weight 1

            M_prime = self.decoder(S_t)
            L_D = F.binary_cross_entropy_with_logits(M_prime, payload.float())
            L_E = F.mse_loss(S_t, cover)

            L_C = torch.tensor(0.0, device=cover.device)
            if self.critic is not None:
                # Encoder wants to MAXIMISE critic(stego) → minimise −critic(stego).
                # L_C = cover_score − gen_score; since cover_score is detached
                # the gradient only flows through −gen_score, pushing it upward.
                L_C = critic_cover_score - torch.mean(self.critic(S_t))

            total = total + weight * (L_D + self.alpha * L_E + L_C)

        return total


# ═══════════════════════════════════════════════════════════════════════
# VGG Perceptual Loss
# ═══════════════════════════════════════════════════════════════════════

class VGGPerceptualLoss(nn.Module):
    """
    Feature-matching perceptual loss using a frozen VGG-16 network.

    Computes MSE between intermediate feature maps of the cover and stego
    images at layers relu1_2, relu2_2, relu3_3 of VGG-16.  All VGG
    parameters are frozen (no gradient tracking).

    The input images are expected in [-1, 1] range and are internally
    re-normalised to ImageNet statistics before feature extraction.

    Parameters
    ----------
    layer_weights : optional dict mapping layer index to loss weight
                    (default: equal weight for all three layers)
    """

    # VGG feature extraction points (after ReLU):
    #   relu1_2 → layer index 3
    #   relu2_2 → layer index 8
    #   relu3_3 → layer index 15
    _LAYER_INDICES = [3, 8, 15]

    def __init__(self, layer_weights: Optional[dict] = None) -> None:
        super().__init__()

        vgg = tv_models.vgg16(weights=tv_models.VGG16_Weights.DEFAULT)
        max_idx = max(self._LAYER_INDICES) + 1
        self.features = nn.Sequential(*list(vgg.features.children())[:max_idx])

        # Freeze all parameters
        for p in self.features.parameters():
            p.requires_grad_(False)
        self.features.eval()

        # ImageNet normalisation constants (applied to [-1,1] → [0,1] → normalised)
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

        self.layer_weights = layer_weights or {i: 1.0 for i in self._LAYER_INDICES}

    def _normalise(self, x: torch.Tensor) -> torch.Tensor:
        """Convert from [-1, 1] to ImageNet-normalised range."""
        x01 = (x + 1.0) * 0.5  # [-1,1] → [0,1]
        return (x01 - self.mean) / self.std

    def forward(self, cover: torch.Tensor, stego: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        cover : (N, 3, H, W) in [-1, 1]
        stego : (N, 3, H, W) in [-1, 1]

        Returns
        -------
        Scalar perceptual loss (sum of per-layer MSE).
        """
        self.features.eval()  # ensure eval mode (no dropout, fixed BN)

        cover_n = self._normalise(cover)
        stego_n = self._normalise(stego)

        # Pre-compute cover features with no_grad — cover is target only,
        # no gradient needs to flow through the VGG cover path.
        cover_feats: dict = {}
        with torch.no_grad():
            x = cover_n
            for idx, layer in enumerate(self.features):
                x = layer(x)
                if idx in self.layer_weights:
                    cover_feats[idx] = x

        # Stego path — gradients flow back through VGG to the stego image.
        loss = torch.tensor(0.0, device=cover.device)
        stego_feat = stego_n
        for idx, layer in enumerate(self.features):
            stego_feat = layer(stego_feat)
            if idx in self.layer_weights:
                loss = loss + self.layer_weights[idx] * F.mse_loss(
                    stego_feat, cover_feats[idx]
                )

        return loss


# ═══════════════════════════════════════════════════════════════════════
# Sobel Edge Regularisation
# ═══════════════════════════════════════════════════════════════════════

class SobelEdgeRegularisation(nn.Module):
    """
    Soft consistency loss between the learned edge map and Sobel magnitude.

    Provides a warm-start bias toward traditional edges without
    over-constraining the learned detector.

    L_edge = MSE(edge_map, normalised_sobel_magnitude(cover))

    The Sobel magnitude is normalised to [0, 1] per-image.
    """

    def __init__(self) -> None:
        super().__init__()
        Kx = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32
        ).view(1, 1, 3, 3)
        Ky = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32
        ).view(1, 1, 3, 3)
        self.register_buffer("Kx", Kx)
        self.register_buffer("Ky", Ky)

    def _sobel_normalised(self, image: torch.Tensor) -> torch.Tensor:
        """Compute normalised Sobel edge magnitude → (N, 1, H, W) in [0, 1]."""
        N, C, H, W = image.shape
        x = image.view(N * C, 1, H, W)
        Gx = F.conv2d(x, self.Kx, padding=1)
        Gy = F.conv2d(x, self.Ky, padding=1)
        mag = (Gx ** 2 + Gy ** 2 + 1e-8).sqrt().view(N, C, H, W)
        mag = mag.mean(dim=1, keepdim=True)  # average across RGB → (N, 1, H, W)
        # Per-image min-max normalisation to [0, 1]
        mag_flat = mag.view(N, -1)
        mn = mag_flat.min(dim=1).values.view(N, 1, 1, 1)
        mx = mag_flat.max(dim=1).values.view(N, 1, 1, 1)
        return (mag - mn) / (mx - mn + 1e-8)

    def forward(self, edge_map: torch.Tensor,
                cover: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        edge_map : (N, 1, H, W) predicted edge map from EdgeNet, in [0, 1]
        cover    : (N, 3, H, W) cover image in [-1, 1]

        Returns
        -------
        Scalar MSE loss.
        """
        target = self._sobel_normalised(cover)
        return F.mse_loss(edge_map, target.detach())


# ═══════════════════════════════════════════════════════════════════════
# Edge-Aware Iterative Loss
# ═══════════════════════════════════════════════════════════════════════

class EdgeAwareIterativeLoss:
    """
    Extended iterative loss for EdgeAwareDenseASPPEncoder.

    L_total = Σ γ^(T-1-t) × [L_D + α×L_E + L_C]
            + λ_edge  × SobelEdgeRegularisation(edge_map, cover)
            + λ_vgg   × VGGPerceptualLoss(cover, S_final)
            + λ_stega × CE(Steganalyzer(S_final → [0,1]), target=cover)

    The steganalyzer term is an optional "second critic": a frozen
    pretrained binary cover/stego classifier whose cross-entropy against the
    cover label (0) the encoder is pushed to minimise.  This makes the
    encoder produce stego images that the steganalyzer classifies as cover.

    Parameters
    ----------
    decoder       : decoder module
    gamma         : per-step discount factor (default 0.8)
    alpha         : image-quality loss weight (default 100.0)
    critic        : optional WGAN critic module
    lambda_edge   : edge regularisation weight (default 0.01)
    lambda_vgg    : VGG perceptual loss weight (default 0.1)
    steganalyzer  : optional frozen binary steganalyzer (logits over [cover, stego]).
                    Input is expected in [0, 1]; the loss handles the
                    [-1, 1] → [0, 1] conversion internally.
    lambda_stega  : steganalyzer term weight (default 1.0)
    """

    def __init__(
        self,
        decoder:      nn.Module,
        gamma:        float = 0.8,
        alpha:        float = 100.0,
        critic:       Optional[nn.Module] = None,
        lambda_edge:  float = 0.01,
        lambda_vgg:   float = 0.1,
        steganalyzer: Optional[nn.Module] = None,
        lambda_stega: float = 1.0,
    ) -> None:
        self.decoder      = decoder
        self.gamma        = gamma
        self.alpha        = alpha
        self.critic       = critic
        self.lambda_edge  = lambda_edge
        self.lambda_vgg   = lambda_vgg
        self.steganalyzer = steganalyzer
        self.lambda_stega = lambda_stega

        self.vgg_loss  = VGGPerceptualLoss()
        self.edge_reg  = SobelEdgeRegularisation()

    def to(self, device: torch.device) -> "EdgeAwareIterativeLoss":
        """Move auxiliary modules to the target device."""
        self.vgg_loss = self.vgg_loss.to(device)
        self.edge_reg = self.edge_reg.to(device)
        if self.steganalyzer is not None:
            self.steganalyzer = self.steganalyzer.to(device)
        return self

    def __call__(
        self,
        cover:      torch.Tensor,
        payload:    torch.Tensor,
        stego_list: List[torch.Tensor],
        edge_map:   Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        cover      : (N,3,H,W) cover image in [-1, 1]
        payload    : (N,D,H,W) ground-truth bits
        stego_list : list of T stego tensors from the iterative encoder in [-1, 1]
        edge_map   : (N,1,H,W) predicted edge map (optional; needed for edge reg)
        """
        T     = len(stego_list)
        total = torch.tensor(0.0, device=cover.device)

        # Critic(cover) — compute once
        critic_cover_score: Optional[torch.Tensor] = None
        if self.critic is not None:
            with torch.no_grad():
                critic_cover_score = torch.mean(self.critic(cover))

        # Weighted iterative loss (same as IterativeLoss)
        for t, S_t in enumerate(stego_list):
            weight = self.gamma ** (T - 1 - t)

            M_prime = self.decoder(S_t)
            L_D = F.binary_cross_entropy_with_logits(M_prime, payload.float())
            L_E = F.mse_loss(S_t, cover)

            L_C = torch.tensor(0.0, device=cover.device)
            if self.critic is not None:
                # Encoder maximises critic(stego) → minimise cover_score − gen_score.
                L_C = critic_cover_score - torch.mean(self.critic(S_t))

            total = total + weight * (L_D + self.alpha * L_E + L_C)

        # VGG perceptual loss on final stego
        S_final = stego_list[-1]
        total = total + self.lambda_vgg * self.vgg_loss(cover, S_final)

        # Edge regularisation
        if edge_map is not None and self.lambda_edge > 0:
            total = total + self.lambda_edge * self.edge_reg(edge_map, cover)

        # Steganalyzer term — push frozen detector toward "cover" on stego.
        if self.steganalyzer is not None and self.lambda_stega > 0:
            stego_01 = ((S_final + 1.0) * 0.5).clamp(0.0, 1.0)
            logits   = self.steganalyzer(stego_01)
            cover_t  = torch.zeros(stego_01.size(0),
                                   dtype=torch.long, device=stego_01.device)
            total = total + self.lambda_stega * F.cross_entropy(logits, cover_t)

        return total

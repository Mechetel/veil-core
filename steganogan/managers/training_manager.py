"""
TrainingManager
───────────────
Manages one epoch of encoder/decoder (and optional critic) training.

Supports two encoder modes:
  • Standard   – encoder returns a single (N,3,H,W) tensor.
  • Iterative  – encoder returns a list of T stego tensors.
                 Detected automatically; uses the weighted iterative loss
                 from Ji et al. 2025 (Eq. 18):

                   L_total = Σ_{t=1}^{T}  γ^{T-t} · [L_D + α·L_E + β·L_C]
"""

import gc
import torch
import torch.nn.functional as F
from torch.optim import Adam
from tqdm import tqdm

from ..utils import ssim
from ..metrics_calculator import MetricsCalculator
from ..generators.payload_generator import PayloadGenerator


def _is_iterative(output) -> bool:
    """Return True if encoder output is a list/tuple of stego images."""
    return isinstance(output, (list, tuple))


class TrainingManager:
    """Manages the training process with optional critic (adversarial) training."""

    def __init__(self, encoder, decoder, data_depth, device, verbose=False,
                 critic=None, critic_train_steps=5):
        self.encoder           = encoder
        self.decoder           = decoder
        self.critic            = critic
        self.data_depth        = data_depth
        self.device            = device
        self.verbose           = verbose
        self.optimizer         = None
        self.critic_optimizer  = None
        self.critic_train_steps = critic_train_steps
        self.payload_generator = PayloadGenerator(device)

        # Iterative-loss hyper-params (mirrors encoder defaults; overridden if
        # the encoder exposes these attributes directly)
        self._gamma = getattr(encoder, 'gamma', 0.8)
        self._alpha = getattr(encoder, 'alpha', 100.0)

    # ── Properties ───────────────────────────────────────────────────────────

    @property
    def use_critic(self) -> bool:
        return self.critic is not None

    # ── Optimisers ───────────────────────────────────────────────────────────

    def get_optimizer(self):
        params = list(self.decoder.parameters()) + list(self.encoder.parameters())
        return Adam(params, lr=1e-4)

    def get_critic_optimizer(self):
        if self.critic is None:
            return None
        return Adam(self.critic.parameters(), lr=1e-4)

    # ── Critic helpers ───────────────────────────────────────────────────────

    def _critic_score(self, image: torch.Tensor) -> torch.Tensor:
        return torch.mean(self.critic(image))

    def _fit_critic_step(self, cover: torch.Tensor):
        """One WGAN critic update step."""
        gc.collect()
        payload   = self.payload_generator.random_data(cover, self.data_depth)
        raw_out   = self.encoder(cover, payload)
        generated = raw_out[-1] if _is_iterative(raw_out) else raw_out

        cover_score     = self._critic_score(cover)
        generated_score = self._critic_score(generated.detach())

        self.critic_optimizer.zero_grad()
        (generated_score - cover_score).backward()
        self.critic_optimizer.step()

        for p in self.critic.parameters():
            p.data.clamp_(-0.1, 0.1)

        return cover_score.item(), generated_score.item()

    # ── Encoding / decoding ───────────────────────────────────────────────────

    def encode_decode(self, cover: torch.Tensor, quantize: bool = False):
        """
        Run encoder + decoder on a batch.

        Returns
        -------
        generated  : final stego image (N,3,H,W)
        payload    : secret bits        (N,D,H,W)
        decoded    : recovered bits     (N,D,H,W)
        raw_output : raw encoder output (tensor or list of tensors)
        """
        payload   = self.payload_generator.random_data(cover, self.data_depth)
        raw_output = self.encoder(cover, payload)

        if _is_iterative(raw_output):
            generated = raw_output[-1]          # final stego image
        else:
            generated = raw_output

        if quantize:
            generated = (255.0 * (generated + 1.0) / 2.0).long()
            generated = 2.0 * generated.float() / 255.0 - 1.0

        decoded = self.decoder(generated)
        return generated, payload, decoded, raw_output

    # ── Iterative loss  (Eq. 18 of Ji et al. 2025) ───────────────────────────

    def _iterative_loss(self, cover: torch.Tensor, payload: torch.Tensor,
                        stego_list: list) -> torch.Tensor:
        """
        L_total = Σ_{t=0}^{T-1}  γ^{T-1-t} · [L_D(M, M'_t) + α·L_E(X, S_t) + β·L_C(X, S_t)]

        L_D : binary cross-entropy between recovered and original bits
        L_E : MSE between stego and cover image
        L_C : critic(S_t) – critic(X)  [Wasserstein distance proxy]
        """
        T      = len(stego_list)
        gamma  = self._gamma
        alpha  = self._alpha
        total  = torch.tensor(0.0, device=cover.device)

        for t, S_t in enumerate(stego_list):
            weight = gamma ** (T - 1 - t)          # more recent = higher weight

            # Decoding loss  (BCE)
            M_prime = self.decoder(S_t)
            # BCEWithLogitsLoss: payload in {0,1}, M_prime is logits
            L_D = F.binary_cross_entropy_with_logits(M_prime, payload.float())

            # Image quality loss  (MSE)
            L_E = F.mse_loss(S_t, cover)

            # Critic loss
            if self.use_critic:
                L_C = self._critic_score(S_t) - self._critic_score(cover)
            else:
                L_C = torch.tensor(0.0, device=cover.device)

            total = total + weight * (L_D + alpha * L_E + L_C)

        return total

    # ── Training loop ─────────────────────────────────────────────────────────

    def fit_coders(self, train, metrics: dict):
        """Train encoder + decoder (+ critic) for one epoch."""
        pbar = tqdm(train, disable=not self.verbose, desc='Training',
                    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')

        for cover, _ in pbar:
            gc.collect()
            cover = cover.to(self.device)

            # ── Critic training ──────────────────────────────────────────
            if self.use_critic:
                for _ in range(self.critic_train_steps):
                    cover_score, generated_score = self._fit_critic_step(cover)
                metrics['train.cover_score'].append(cover_score)
                metrics['train.generated_score'].append(generated_score)

            # ── Encoder + Decoder training ───────────────────────────────
            generated, payload, decoded, raw_output = self.encode_decode(cover)

            if _is_iterative(raw_output):
                # ─── Iterative encoder loss (Eq. 18) ───────────────────
                total_loss = self._iterative_loss(cover, payload, raw_output)
                # Also add the standard coding scores for logging
                with torch.no_grad():
                    enc_mse, dec_loss, dec_acc = MetricsCalculator.coding_scores(
                        cover, generated, payload, decoded)
            else:
                # ─── Standard loss ──────────────────────────────────────
                enc_mse, dec_loss, dec_acc = MetricsCalculator.coding_scores(
                    cover, generated, payload, decoded)
                if self.use_critic:
                    gen_score  = self._critic_score(generated)
                    total_loss = 100.0 * enc_mse + dec_loss - gen_score
                else:
                    total_loss = 100.0 * enc_mse + dec_loss

            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

            # Logging
            metrics['train.encoder_mse'].append(enc_mse.item())
            metrics['train.decoder_loss'].append(dec_loss.item())
            metrics['train.decoder_acc'].append(dec_acc.item())

            postfix = {
                'enc_mse' : f'{enc_mse.item():.4f}',
                'dec_loss': f'{dec_loss.item():.4f}',
                'dec_acc' : f'{dec_acc.item():.4f}',
            }
            if self.use_critic:
                postfix['cover'] = f'{metrics["train.cover_score"][-1]:.4f}'
                postfix['gen']   = f'{metrics["train.generated_score"][-1]:.4f}'
            pbar.set_postfix(postfix)

    def validate(self, validate, metrics: dict):
        """Validate the model (no gradient)."""
        pbar = tqdm(validate, disable=not self.verbose, desc='Validation',
                    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')

        with torch.no_grad():
            for cover, _ in pbar:
                gc.collect()
                cover = cover.to(self.device)

                generated, payload, decoded, raw_output = self.encode_decode(
                    cover, quantize=True)

                enc_mse, dec_loss, dec_acc = MetricsCalculator.coding_scores(
                    cover, generated, payload, decoded)

                metrics['val.encoder_mse'].append(enc_mse.item())
                metrics['val.decoder_loss'].append(dec_loss.item())
                metrics['val.decoder_acc'].append(dec_acc.item())
                metrics['val.ssim'].append(ssim(cover, generated).item())
                metrics['val.psnr'].append(10 * torch.log10(4 / enc_mse).item())
                metrics['val.rsbpp'].append(
                    self.data_depth * (2 * dec_acc.item() - 1))

                if self.use_critic:
                    cover_score     = self._critic_score(cover)
                    generated_score = self._critic_score(generated)
                    metrics['val.cover_score'].append(cover_score.item())
                    metrics['val.generated_score'].append(generated_score.item())

                pbar.set_postfix({
                    'ssim' : f'{metrics["val.ssim"][-1]:.4f}',
                    'psnr' : f'{metrics["val.psnr"][-1]:.2f}',
                    'rsbpp': f'{metrics["val.rsbpp"][-1]:.4f}',
                })

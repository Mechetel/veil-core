# -*- coding: utf-8 -*-
"""
SteganoGAN – top-level model orchestrator.

Wires together the encoder, decoder, critic, trainer, inference services,
payload factory, visualizer, and training history into one coherent object.
"""

import gc
import inspect
import os
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import torch

from .models.base    import BaseEncoder, BaseDecoder, BaseCritic
from .training       import Trainer
from .inference      import EncoderService, DecoderService
from .utils.device        import DeviceManager
from .utils.payload       import PayloadFactory
from .utils.history       import TrainingHistory
from .utils.checkpoint    import ModelCheckpoint
from .utils.visualization import SampleGridVisualizer


# ── Metric field registries ───────────────────────────────────────────────────

_BASE_METRICS: List[str] = [
    "train.encoder_mse", "train.decoder_loss", "train.decoder_acc",
    "val.encoder_mse",   "val.decoder_loss",   "val.decoder_acc",
    "val.ssim",          "val.psnr",           "val.wpsnr",
    "val.fsim",          "val.rsbpp",
]

_CRITIC_METRICS: List[str] = [
    "train.cover_score", "train.generated_score",
    "val.cover_score",   "val.generated_score",
]


class SteganoGAN:
    """
    SteganoGAN model orchestrator.

    Accepts encoder / decoder / critic either as **already-instantiated
    modules** or as **class objects** (they will be instantiated
    automatically with the matching ``data_depth`` kwarg).

    Parameters
    ----------
    data_depth : bits per pixel (D)
    encoder    : encoder module or class
    decoder    : decoder module or class
    critic     : optional WGAN critic module or class
    gpu        : request GPU acceleration
    verbose    : print training progress
    log_dir    : directory for checkpoints, metrics, and sample grids
    """

    def __init__(
        self,
        data_depth: int,
        encoder:    Union[BaseEncoder, Type[BaseEncoder]],
        decoder:    Union[BaseDecoder, Type[BaseDecoder]],
        critic:     Optional[Union[BaseCritic, Type[BaseCritic]]] = None,
        gpu:        bool = True,
        verbose:    bool = False,
        log_dir:    Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        self.verbose    = verbose
        self.data_depth = data_depth
        kwargs["data_depth"] = data_depth

        # ── Instantiate sub-modules ───────────────────────────────────────
        self.encoder: BaseEncoder = self._instantiate(encoder, kwargs)
        self.decoder: BaseDecoder = self._instantiate(decoder, kwargs)
        self.critic:  Optional[BaseCritic] = (
            self._instantiate(critic, kwargs) if critic is not None else None
        )

        # ── Device setup ──────────────────────────────────────────────────
        self.device_manager = DeviceManager(gpu=gpu, verbose=verbose)
        self.device         = self.device_manager.device
        self.gpu            = self.device_manager.gpu
        modules             = [self.encoder, self.decoder]
        if self.critic is not None:
            modules.append(self.critic)
        self.device_manager.to_device(*modules)

        # ── Core services ─────────────────────────────────────────────────
        self._payload_factory = PayloadFactory(self.device)

        self._trainer = Trainer(
            encoder=self.encoder,
            decoder=self.decoder,
            data_depth=self.data_depth,
            device=self.device,
            verbose=verbose,
            critic=self.critic,
        )

        self._encoder_svc = EncoderService(
            self.encoder, self._payload_factory,
            self.data_depth, self.device, verbose,
        )
        self._decoder_svc = DecoderService(self.decoder, self.device, verbose)

        # ── Transient training state ──────────────────────────────────────
        self.encoder_decoder_optimizer: Optional[torch.optim.Optimizer] = None
        self.critic_optimizer:          Optional[torch.optim.Optimizer] = None
        self.fit_metrics:               Optional[Dict[str, Any]]        = None
        self.history:                   List[Dict[str, Any]]            = []

        # ── Logging / persistence ─────────────────────────────────────────
        self.log_dir = log_dir
        self._history_mgr: Optional[TrainingHistory] = None
        self.samples_path: Optional[str] = None

        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            self.samples_path = os.path.join(log_dir, "samples")
            os.makedirs(self.samples_path, exist_ok=True)
            self._history_mgr = TrainingHistory(log_dir)

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _instantiate(
        cls_or_obj: Union[Any, type],
        kwargs:     Dict[str, Any],
    ) -> Any:
        """Return *cls_or_obj* directly if already instantiated; else construct it."""
        if not inspect.isclass(cls_or_obj):
            return cls_or_obj
        args = inspect.getfullargspec(cls_or_obj.__init__).args
        args.remove("self")
        return cls_or_obj(**{k: kwargs[k] for k in args if k in kwargs})

    # ── Device ────────────────────────────────────────────────────────────────

    def set_device(self) -> None:
        """Re-detect and re-apply the best available compute device."""
        self.device_manager.set_device()
        self.device = self.device_manager.device
        self.gpu    = self.device_manager.gpu
        modules = [self.encoder, self.decoder]
        if self.critic is not None:
            modules.append(self.critic)
        self.device_manager.to_device(*modules)

    def set_depth(self, data_depth: int) -> None:
        """
        Switch the active data_depth for all components.

        For DepthAgnosticEncoder / DepthAgnosticDecoder this controls the
        number of bit-planes embedded / recovered at inference time.
        For standard fixed-shape models the call is a no-op at inference
        (their weight shapes are already committed to the original D).
        """
        self.data_depth               = data_depth
        self.encoder.data_depth       = data_depth
        self.decoder.data_depth       = data_depth
        self._encoder_svc.data_depth  = data_depth
        self._trainer.data_depth      = data_depth

    # ── Training ──────────────────────────────────────────────────────────────

    def fit(
        self,
        train:       torch.utils.data.DataLoader,
        validate:    torch.utils.data.DataLoader,
        epochs:      int = 32,
        start_epoch: int = 1,
        data_depth:  Optional[int] = None,
    ) -> None:
        """
        Train the model.

        Parameters
        ----------
        train, validate : DataLoaders for training and validation
        epochs          : number of epochs to run
        start_epoch     : starting epoch index (for resuming)
        data_depth      : override self.data_depth if needed
        """
        if data_depth is not None:
            self.data_depth = data_depth

        # Lazy optimiser init
        if self.encoder_decoder_optimizer is None:
            self.encoder_decoder_optimizer = self._trainer.build_optimizer()
            self._trainer.optimizer = self.encoder_decoder_optimizer

        if self.critic is not None and self.critic_optimizer is None:
            self.critic_optimizer = self._trainer.build_critic_optimizer()
            self._trainer.critic_optimizer = self.critic_optimizer

        if self._history_mgr:
            self.history = self._history_mgr.records

        metric_fields = _BASE_METRICS + (
            _CRITIC_METRICS if self.critic is not None else []
        )
        end_epoch = start_epoch + epochs

        for epoch in range(start_epoch, end_epoch):
            metrics: Dict[str, List[float]] = {f: [] for f in metric_fields}

            if self.verbose:
                print(f"\n{'=' * 60}")
                print(f"Epoch {epoch}/{end_epoch - 1}")
                print(f"{'=' * 60}")

            self._trainer.train_epoch(train, metrics)
            self._trainer.validate_epoch(validate, metrics)

            self.fit_metrics = {k: sum(v) / len(v) for k, v in metrics.items()}
            self.fit_metrics["epoch"] = epoch

            if self.verbose and self._history_mgr:
                self._history_mgr.print_epoch(self.fit_metrics)

            if self.log_dir:
                if self._history_mgr:
                    self._history_mgr.append(self.fit_metrics)
                    self.history = self._history_mgr.records

                save_this_epoch = (
                    epoch == start_epoch
                    or epoch % 5 == 0
                    or epoch == end_epoch - 1
                )
                if save_this_epoch:
                    ckpt_name = f"{epoch}.rsbpp-{self.fit_metrics['val.rsbpp']:.6f}.p"
                    self.save(os.path.join(self.log_dir, ckpt_name))

                    SampleGridVisualizer(
                        encoder=self.encoder,
                        payload_factory=self._payload_factory,
                        device=self.device,
                    ).save_grid(
                        self.samples_path, epoch,
                        "Hello, SteganoGAN!", self.data_depth,
                    )

            # Free GPU memory between epochs
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            elif self.device.type == "mps":
                torch.mps.empty_cache()
            gc.collect()

    # ── Inference ─────────────────────────────────────────────────────────────

    def encode(self, cover_path: str, output_path: str, message: str) -> None:
        """Embed *message* into *cover_path* and write to *output_path*."""
        self._encoder_svc.encode(cover_path, output_path, message)

    def decode(self, image_path: str) -> str:
        """Extract and return the hidden message from *image_path*."""
        return self._decoder_svc.decode(image_path)

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Serialise this model to *path*."""
        ModelCheckpoint.save(self, path, verbose=self.verbose)

    @classmethod
    def load(
        cls,
        path:    str,
        gpu:     bool = True,
        verbose: bool = False,
        log_dir: Optional[str] = None,
    ) -> "SteganoGAN":
        """Load and return a SteganoGAN model from a checkpoint file."""
        return ModelCheckpoint.load(path, gpu=gpu, verbose=verbose, log_dir=log_dir)

    # ── Convenience properties ────────────────────────────────────────────────

    @property
    def parameter_count(self) -> Dict[str, int]:
        """Return dict with encoder / decoder / critic / total parameter counts."""
        enc = sum(p.numel() for p in self.encoder.parameters())
        dec = sum(p.numel() for p in self.decoder.parameters())
        crt = sum(p.numel() for p in self.critic.parameters()) if self.critic is not None else 0
        return {"encoder": enc, "decoder": dec, "critic": crt, "total": enc + dec + crt}

    def __repr__(self) -> str:  # pragma: no cover
        p = self.parameter_count
        return (
            f"SteganoGAN("
            f"encoder={type(self.encoder).__name__}, "
            f"decoder={type(self.decoder).__name__}, "
            f"critic={type(self.critic).__name__ if self.critic else None}, "
            f"data_depth={self.data_depth}, "
            f"device={self.device}, "
            f"params={p['total']:,})"
        )

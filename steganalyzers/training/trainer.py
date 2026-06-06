# -*- coding: utf-8 -*-
"""
Steganalysis Trainer.

Manages one full training epoch plus validation for any BaseSteganalyzer.

Features
--------
- Mixed-precision (AMP) on CUDA, transparent fallback to FP32 on CPU/MPS
- Gradient clipping (max_norm=1.0)
- Callback system: per-epoch hooks for logging, checkpointing, scheduling
- Tqdm progress bars with live accuracy display
- Accumulates raw logits + labels during validation to compute the full
  metric suite (accuracy, balanced_accuracy, AUC-ROC, TPR@FPR0.1, F1)
"""

from typing import Callable, Dict, List, Optional

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
try:
    from torch.amp import GradScaler, autocast
    _AMP_SUPPORTS_DEVICE_TYPE = True
except ImportError:
    from torch.cuda.amp import GradScaler, autocast
    _AMP_SUPPORTS_DEVICE_TYPE = False
from tqdm import tqdm

from .metrics import SteganalysisMetrics
from ..base   import BaseSteganalyzer


class Trainer:
    """
    Training/validation manager for steganalysis networks.

    Parameters
    ----------
    model       : the steganalyzer to train
    optimizer   : pre-built optimiser (e.g. Adam)
    device      : torch.device for computation
    criterion   : loss function (default: CrossEntropyLoss)
    scheduler   : optional LR scheduler (stepped once per epoch after val)
    verbose     : show tqdm bars
    callbacks   : list of callables invoked at end of each epoch with
                  signature ``cb(epoch, train_metrics, val_metrics, trainer)``
    """

    def __init__(
        self,
        model:      BaseSteganalyzer,
        optimizer:  Optimizer,
        device:     torch.device,
        criterion:  Optional[nn.Module]     = None,
        scheduler                           = None,
        verbose:    bool                    = True,
        callbacks:  Optional[List[Callable]] = None,
    ) -> None:
        self.model     = model.to(device)
        self.optimizer = optimizer
        self.device    = device
        self.criterion = criterion or nn.CrossEntropyLoss()
        self.scheduler = scheduler
        self.verbose   = verbose
        self.callbacks = callbacks or []

        # AMP support
        self._amp_device = device.type if isinstance(device, torch.device) else str(device)
        self._use_amp    = (self._amp_device == "cuda")
        self._scaler: Optional[GradScaler] = (
            GradScaler(device=self._amp_device) if self._use_amp else None
        )

        # History accumulated across all epochs
        self.history: Dict[str, List[float]] = {}

    # ── Training epoch ─────────────────────────────────────────────────────────

    def train_epoch(self, loader: DataLoader) -> Dict[str, float]:
        """Run one training epoch.  Returns mean metrics for the epoch."""
        self.model.train()

        running: Dict[str, List[float]] = {
            "train_loss": [],
            "train_acc":  [],
        }

        pbar = tqdm(
            loader,
            disable=not self.verbose,
            desc="Train",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
        )

        for images, labels in pbar:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            with (autocast(device_type=self._amp_device, enabled=self._use_amp) if _AMP_SUPPORTS_DEVICE_TYPE else autocast(enabled=self._use_amp)):
                logits = self.model(images)
                loss   = self.criterion(logits, labels)

            self.optimizer.zero_grad()
            if self._use_amp:
                self._scaler.scale(loss).backward()
                self._scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                self._scaler.step(self.optimizer)
                self._scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                self.optimizer.step()

            with torch.no_grad():
                acc = (logits.argmax(dim=1) == labels).float().mean().item()

            running["train_loss"].append(loss.item())
            running["train_acc"].append(acc)

            pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{acc:.4f}")

        return SteganalysisMetrics.average(running)

    # ── Validation epoch ───────────────────────────────────────────────────────

    def val_epoch(self, loader: DataLoader) -> Dict[str, float]:
        """Run one validation pass.  Returns full metric suite."""
        self.model.eval()

        all_logits: List[torch.Tensor] = []
        all_labels: List[torch.Tensor] = []
        val_losses: List[float]        = []

        pbar = tqdm(
            loader,
            disable=not self.verbose,
            desc="Val  ",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
        )

        with torch.no_grad():
            for images, labels in pbar:
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                with (autocast(device_type=self._amp_device, enabled=self._use_amp) if _AMP_SUPPORTS_DEVICE_TYPE else autocast(enabled=self._use_amp)):
                    logits = self.model(images)
                    loss   = self.criterion(logits, labels)

                val_losses.append(loss.item())
                all_logits.append(logits.cpu())
                all_labels.append(labels.cpu())

                acc = (logits.argmax(dim=1) == labels).float().mean().item()
                pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{acc:.4f}")

        all_logits_t = torch.cat(all_logits)
        all_labels_t = torch.cat(all_labels)

        full_metrics = SteganalysisMetrics.from_logits(all_logits_t, all_labels_t)
        full_metrics["val_loss"] = float(torch.tensor(val_losses).mean().item())

        return full_metrics

    # ── Full training loop ─────────────────────────────────────────────────────

    def fit(
        self,
        train_loader: DataLoader,
        val_loader:   DataLoader,
        epochs:       int,
        start_epoch:  int = 1,
    ) -> Dict[str, List[float]]:
        """
        Train for *epochs* epochs and return the full history dict.

        Parameters
        ----------
        train_loader : training DataLoader
        val_loader   : validation DataLoader
        epochs       : total number of training epochs
        start_epoch  : epoch to start from (useful when resuming)

        Returns
        -------
        history dict: keys → list of per-epoch values
        """
        for epoch in range(start_epoch, start_epoch + epochs):
            if self.verbose:
                print(f"\nEpoch {epoch}/{start_epoch + epochs - 1}")
                print("-" * 40)

            train_metrics = self.train_epoch(train_loader)
            val_metrics   = self.val_epoch(val_loader)

            # Step LR scheduler
            if self.scheduler is not None:
                if hasattr(self.scheduler, "step"):
                    # ReduceLROnPlateau needs the monitored metric
                    if hasattr(self.scheduler, "patience"):
                        self.scheduler.step(val_metrics.get("val_loss", 0.0))
                    else:
                        self.scheduler.step()

            # Accumulate history
            for k, v in {**train_metrics, **val_metrics}.items():
                self.history.setdefault(k, []).append(v)

            # Print epoch summary
            if self.verbose:
                self._print_summary(epoch, train_metrics, val_metrics)

            # Fire callbacks
            stop = False
            for cb in self.callbacks:
                cb(epoch, train_metrics, val_metrics, self)
                if getattr(cb, "stop", False):
                    stop = True

            if stop:
                break

        return self.history

    # ── Helpers ────────────────────────────────────────────────────────────────

    @staticmethod
    def _print_summary(
        epoch:        int,
        train_metrics: Dict[str, float],
        val_metrics:   Dict[str, float],
    ) -> None:
        print(
            f"  train_loss={train_metrics['train_loss']:.4f}  "
            f"train_acc={train_metrics['train_acc']:.4f}  │  "
            f"val_loss={val_metrics['val_loss']:.4f}  "
            f"val_acc={val_metrics['accuracy']:.4f}  "
            f"bal_acc={val_metrics['balanced_accuracy']:.4f}  "
            f"AUC={val_metrics['auc_roc']:.4f}  "
            f"TPR@FPR10={val_metrics['tpr_at_fpr10']:.4f}  "
            f"F1={val_metrics['f1']:.4f}"
        )

    # ── Pickle support ─────────────────────────────────────────────────────────

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("_scaler", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        use_amp = getattr(self, "_use_amp", False)
        if use_amp:
            amp_dev = getattr(self, "_amp_device", "cuda")
            try:
                self._scaler = GradScaler(device=amp_dev)
            except TypeError:
                self._scaler = GradScaler()
        else:
            self._scaler = None

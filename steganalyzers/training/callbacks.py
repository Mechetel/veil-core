# -*- coding: utf-8 -*-
"""
Training callbacks for steganalysis networks.

All callbacks follow the signature:
    cb(epoch: int, train_metrics: dict, val_metrics: dict, trainer: Trainer)

Available callbacks
-------------------
MetricsLogger     — writes per-epoch metrics to a TSV log file
CheckpointSaver   — saves the best model and optionally periodic checkpoints
EarlyStopping     — stops training when a monitored metric stops improving
LRMonitor         — prints current learning rate each epoch
"""

import csv
import os
from pathlib import Path
from typing import Any, Dict, Optional, TYPE_CHECKING

import torch

from ..utils.checkpoint import Checkpoint

if TYPE_CHECKING:
    from .trainer import Trainer


# ── Metrics logger ─────────────────────────────────────────────────────────────

class MetricsLogger:
    """
    Appends per-epoch metrics to a TSV file.

    Creates the file on the first epoch and writes one row per epoch.
    The header is written automatically from the metric keys.

    Parameters
    ----------
    log_path : path to the output TSV file
    verbose  : print a confirmation message each epoch
    """

    def __init__(self, log_path: str, verbose: bool = False) -> None:
        self.log_path = str(log_path)
        self.verbose  = verbose
        self._header_written = False

    def __call__(
        self,
        epoch:         int,
        train_metrics: Dict[str, float],
        val_metrics:   Dict[str, float],
        trainer:       "Trainer",
    ) -> None:
        row = {"epoch": epoch, **train_metrics, **val_metrics}
        os.makedirs(os.path.dirname(os.path.abspath(self.log_path)), exist_ok=True)

        mode = "w" if not self._header_written else "a"
        with open(self.log_path, mode, newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()), delimiter="\t")
            if not self._header_written:
                writer.writeheader()
                self._header_written = True
            writer.writerow({k: f"{v:.6f}" if isinstance(v, float) else v for k, v in row.items()})

        if self.verbose:
            print(f"  [MetricsLogger] epoch {epoch} → {self.log_path}")


# ── Checkpoint saver ───────────────────────────────────────────────────────────

class CheckpointSaver:
    """
    Saves the model checkpoint when a monitored metric improves.

    Parameters
    ----------
    checkpoint_dir  : directory to write checkpoints
    monitor         : metric name to watch (from val_metrics)
    mode            : "max" or "min" — whether higher or lower is better
    save_best_only  : if True only save when monitor improves
    save_every      : also save every N epochs regardless of improvement
    verbose         : print save messages
    config          : CONFIG dict to embed in every checkpoint
    """

    def __init__(
        self,
        checkpoint_dir: str,
        monitor:        str   = "auc_roc",
        mode:           str   = "max",
        save_best_only: bool  = True,
        save_every:     int   = 0,
        verbose:        bool  = True,
        config:         Optional[Dict[str, Any]] = None,
    ) -> None:
        assert mode in ("max", "min"), f"mode must be 'max' or 'min', got {mode!r}"
        self.checkpoint_dir = Path(checkpoint_dir)
        self.monitor        = monitor
        self.mode           = mode
        self.save_best_only = save_best_only
        self.save_every     = save_every
        self.verbose        = verbose
        self.config         = config

        self._best: float = float("-inf") if mode == "max" else float("inf")
        self.best_path: Optional[str] = None
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def __call__(
        self,
        epoch:         int,
        train_metrics: Dict[str, float],
        val_metrics:   Dict[str, float],
        trainer:       "Trainer",
    ) -> None:
        current = val_metrics.get(self.monitor)
        if current is None:
            return

        improved = (
            (self.mode == "max" and current > self._best) or
            (self.mode == "min" and current < self._best)
        )

        if improved:
            self._best = current
            best_path  = str(self.checkpoint_dir / f"best_epoch{epoch:04d}.pt")
            Checkpoint.save(
                trainer, best_path, epoch,
                metrics=val_metrics, config=self.config,
                verbose=self.verbose,
            )
            self.best_path = best_path

            # Remove previous best to avoid cluttering the directory
            for old in self.checkpoint_dir.glob("best_epoch*.pt"):
                if str(old) != best_path:
                    old.unlink(missing_ok=True)
                    _json = str(old).replace(".pt", "_metrics.json")
                    Path(_json).unlink(missing_ok=True)

        elif not self.save_best_only and self.save_every > 0 and epoch % self.save_every == 0:
            periodic_path = str(self.checkpoint_dir / f"epoch{epoch:04d}.pt")
            Checkpoint.save(
                trainer, periodic_path, epoch,
                metrics=val_metrics, config=self.config,
                verbose=self.verbose,
            )

    def is_improved(self) -> bool:
        """True if the most recent epoch was an improvement."""
        return self.best_path is not None


# ── Early stopping ─────────────────────────────────────────────────────────────

class EarlyStopping:
    """
    Stops training when the monitored metric stops improving.

    Raises :class:`StopIteration` inside the Trainer.fit loop via a flag
    that :meth:`Trainer.fit` checks after each callback invocation.

    Parameters
    ----------
    monitor   : metric name from val_metrics
    patience  : epochs without improvement before stopping
    mode      : "max" or "min"
    min_delta : minimum change to qualify as an improvement
    verbose   : print message when triggered
    """

    def __init__(
        self,
        monitor:   str   = "auc_roc",
        patience:  int   = 10,
        mode:      str   = "max",
        min_delta: float = 1e-4,
        verbose:   bool  = True,
    ) -> None:
        assert mode in ("max", "min")
        self.monitor   = monitor
        self.patience  = patience
        self.mode      = mode
        self.min_delta = min_delta
        self.verbose   = verbose

        self._best:   float = float("-inf") if mode == "max" else float("inf")
        self._wait:   int   = 0
        self.stop:    bool  = False     # Trainer.fit checks this flag

    def __call__(
        self,
        epoch:         int,
        train_metrics: Dict[str, float],
        val_metrics:   Dict[str, float],
        trainer:       "Trainer",
    ) -> None:
        current = val_metrics.get(self.monitor)
        if current is None:
            return

        if self.mode == "max":
            improved = current > self._best + self.min_delta
        else:
            improved = current < self._best - self.min_delta

        if improved:
            self._best = current
            self._wait = 0
        else:
            self._wait += 1
            if self._wait >= self.patience:
                self.stop = True
                if self.verbose:
                    print(
                        f"\n[EarlyStopping] No improvement in {self.monitor!r} "
                        f"for {self.patience} epochs. Stopping at epoch {epoch}."
                    )


# ── LR monitor ────────────────────────────────────────────────────────────────

class LRMonitor:
    """Prints the current learning rate at the end of each epoch."""

    def __call__(
        self,
        epoch:         int,
        train_metrics: Dict[str, float],
        val_metrics:   Dict[str, float],
        trainer:       "Trainer",
    ) -> None:
        lrs = [pg["lr"] for pg in trainer.optimizer.param_groups]
        lr_str = ", ".join(f"{lr:.2e}" for lr in lrs)
        print(f"  [LRMonitor] epoch {epoch}  lr = {lr_str}")

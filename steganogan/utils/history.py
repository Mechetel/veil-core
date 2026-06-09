# -*- coding: utf-8 -*-
"""Training history: append-only log that persists metrics to JSON on disk."""

import json
import os
from typing import Any, Dict, List, Optional


class TrainingHistory:
    """
    Append-only container for per-epoch training metrics.

    On construction, any pre-existing ``metrics.log`` in *log_dir* is loaded
    so that training can be resumed seamlessly.

    Parameters
    ----------
    log_dir : directory where ``metrics.log`` is stored
    """

    _LOG_FILENAME = "metrics.log"

    def __init__(self, log_dir: str) -> None:
        self.log_dir: str = log_dir
        self.records: List[Dict[str, Any]] = []
        self._log_path: Optional[str] = (
            os.path.join(log_dir, self._LOG_FILENAME) if log_dir else None
        )
        self._load()

    # ── Public API ────────────────────────────────────────────────────────────

    def append(self, metrics: Dict[str, Any]) -> None:
        """Append one epoch's metrics dict and flush to disk immediately."""
        self.records.append(metrics)
        self._flush()

    def print_epoch(self, metrics: Dict[str, Any]) -> None:
        """Pretty-print a single epoch's metrics dict to stdout."""
        ep = metrics["epoch"]
        print(f"\nEpoch {ep} metrics:")
        print(
            f"  [train]  enc_mse={metrics['train.encoder_mse']:.6f}"
            f"  dec_loss={metrics['train.decoder_loss']:.4f}"
            f"  dec_acc={metrics['train.decoder_acc']:.4f}"
        )
        if "train.cover_score" in metrics:
            print(
                f"  [train]  cover_score={metrics['train.cover_score']:.6f}"
                f"  gen_score={metrics['train.generated_score']:.6f}"
            )
        print(
            f"  [val]    enc_mse={metrics['val.encoder_mse']:.6f}"
            f"  dec_loss={metrics['val.decoder_loss']:.4f}"
            f"  dec_acc={metrics['val.decoder_acc']:.4f}"
        )
        print(
            f"  [val]    ssim={metrics['val.ssim']:.4f}"
            f"  psnr={metrics['val.psnr']:.2f}"
            f"  wpsnr={metrics['val.wpsnr']:.2f}"
            f"  fsim={metrics['val.fsim']:.4f}"
            f"  rsbpp={metrics['val.rsbpp']:.4f}"
        )
        if "val.cover_score" in metrics:
            print(
                f"  [val]    cover_score={metrics['val.cover_score']:.6f}"
                f"  gen_score={metrics['val.generated_score']:.6f}"
            )

    # ── Backward-compat aliases ───────────────────────────────────────────────

    @property
    def history(self) -> List[Dict[str, Any]]:
        """Alias for :attr:`records` (backward compat)."""
        return self.records

    def append_and_save(self, metrics: Dict[str, Any]) -> None:
        """Alias for :meth:`append` (backward compat)."""
        self.append(metrics)

    def print_metrics(self, metrics: Dict[str, Any]) -> None:
        """Alias for :meth:`print_epoch` (backward compat)."""
        self.print_epoch(metrics)

    # ── Internal ─────────────────────────────────────────────────────────────

    def _load(self) -> None:
        if self._log_path and os.path.exists(self._log_path):
            try:
                with open(self._log_path, "r") as fh:
                    self.records = json.load(fh)
                    print(f"Resumed training history: {len(self.records)} epochs loaded.")
            except (json.JSONDecodeError, IOError) as exc:
                print(f"Warning: could not load {self._log_path}: {exc}")
                self.records = []

    def _flush(self) -> None:
        if self._log_path:
            with open(self._log_path, "w") as fh:
                json.dump(self.records, fh, indent=4)

# -*- coding: utf-8 -*-
"""
Checkpoint utilities for steganalysis networks.

Saves and loads the full training state: model weights, optimiser state,
scheduler state, training history, and CONFIG dict.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from ..training.trainer import Trainer


class Checkpoint:
    """
    Serialises and deserialises trainer state to disk.

    File layout
    -----------
    <path>.pt   — PyTorch state dict bundle
    <path>.json — human-readable metrics snapshot

    All methods are static; this class is a pure-utility namespace.
    """

    @staticmethod
    def save(
        trainer:    "Trainer",
        path:       str,
        epoch:      int,
        metrics:    Optional[Dict[str, float]] = None,
        config:     Optional[Dict[str, Any]]   = None,
        verbose:    bool = True,
    ) -> None:
        """
        Persist the full trainer state to *path*.

        Parameters
        ----------
        trainer : Trainer whose model, optimizer, scheduler to save
        path    : destination file path (extension will be .pt)
        epoch   : current epoch number (stored in the bundle)
        metrics : latest validation metrics to embed in the checkpoint
        config  : CONFIG dict to embed for reproducibility
        verbose : print confirmation message
        """
        path = str(path)
        if not path.endswith(".pt"):
            path += ".pt"

        bundle: Dict[str, Any] = {
            "epoch":           epoch,
            "model_state":     trainer.model.state_dict(),
            "optimizer_state": trainer.optimizer.state_dict(),
            "history":         trainer.history,
        }
        if trainer.scheduler is not None:
            bundle["scheduler_state"] = trainer.scheduler.state_dict()
        if metrics is not None:
            bundle["metrics"] = metrics
        if config is not None:
            bundle["config"] = config

        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save(bundle, path)

        # Companion JSON for quick inspection
        json_path = path.replace(".pt", "_metrics.json")
        if metrics or config:
            with open(json_path, "w") as f:
                json.dump(
                    {"epoch": epoch, "metrics": metrics or {}, "config": config or {}},
                    f, indent=2,
                )

        if verbose:
            print(f"Checkpoint saved → {path}")

    @staticmethod
    def load(
        trainer:  "Trainer",
        path:     str,
        device:   Optional[torch.device] = None,
        verbose:  bool = True,
    ) -> int:
        """
        Load a checkpoint into *trainer* in-place.

        Parameters
        ----------
        trainer : Trainer to restore state into
        path    : checkpoint file (.pt)
        device  : map_location (defaults to trainer.device)
        verbose : print confirmation message

        Returns
        -------
        int — the epoch stored in the checkpoint (use as start_epoch)
        """
        if not path.endswith(".pt"):
            path += ".pt"

        map_loc = device or trainer.device
        bundle  = torch.load(path, map_location=map_loc, weights_only=False)

        trainer.model.load_state_dict(bundle["model_state"])
        trainer.optimizer.load_state_dict(bundle["optimizer_state"])

        if "scheduler_state" in bundle and trainer.scheduler is not None:
            trainer.scheduler.load_state_dict(bundle["scheduler_state"])

        if "history" in bundle:
            trainer.history = bundle["history"]

        epoch = bundle.get("epoch", 0)

        if verbose:
            metrics_str = ""
            if "metrics" in bundle:
                m = bundle["metrics"]
                metrics_str = (
                    f"  acc={m.get('accuracy', 0):.4f}  "
                    f"auc={m.get('auc_roc', 0):.4f}"
                )
            print(f"Checkpoint loaded ← {path}  (epoch {epoch}){metrics_str}")

        return epoch

    @staticmethod
    def best_checkpoint_path(checkpoint_dir: str) -> Optional[str]:
        """Return the path of the best checkpoint in *checkpoint_dir*, or None."""
        checkpoint_dir = Path(checkpoint_dir)
        pts = sorted(checkpoint_dir.glob("best_*.pt"))
        return str(pts[-1]) if pts else None

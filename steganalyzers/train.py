#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Steganalysis training script.

Trains any steganalysis network on the ALASKA2 dataset.
Configure the run entirely via the CONFIG dict at the top of this file.

Supported networks
------------------
  xunet      : XuNet (Xu et al., IEEE SPL 2016)
  yenet      : YeNet (Ye et al., IEEE TIFS 2017)
  srnet      : SRNet (Boroumand et al., IEEE TIFS 2019)
  yedroudjnet: Yedroudj-Net (Yedroudj et al., ICASSP 2018)
  zhunet     : ZhuNet (Zhu et al., 2020)
  siastegnet : SIAStegNet (Spatial-channel Integrated Attention)

Example
-------
    python steganalyzers/train.py

Outputs
-------
  <log_dir>/best_epoch<N>.pt          — best checkpoint (by AUC-ROC)
  <log_dir>/metrics.tsv               — per-epoch metrics table
  <log_dir>/config.json               — serialised CONFIG
"""

import gc
import json
import os
import sys

sys.warnoptions.append("ignore::UserWarning:multiprocessing.resource_tracker")

from time import time
from typing import Any, Dict

os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

# ── Package imports ────────────────────────────────────────────────────────────
# Allow running from the project root: python steganalyzers/train.py
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

from steganalyzers.models     import XuNet, YeNet, SRNet, YedroudjNet, ZhuNet, SIAStegNet, EfficientNetSteg
from steganalyzers.data       import Alaska2DataLoaderFactory, SteganoganDataLoaderFactory
from steganalyzers.training   import (
    Trainer, MetricsLogger, CheckpointSaver, LRMonitor,
)


# ── Configuration ──────────────────────────────────────────────────────────────

CONFIG: Dict[str, Any] = {
    # ── Hardware ──────────────────────────────────────────────────────────────
    "gpu":              True,

    # ── Network ───────────────────────────────────────────────────────────────
    # Choices: xunet+ | yenet+ | srnet+ | yedroudjnet+ | zhunet- | siastegnet- | efficientnetsteg+
    "network":          "xunet",

    # ── Network-specific hyper-parameters ────────────────────────────────────
    # YeNet / ZhuNet / SiaStegNet
    "srm_trainable":    False,      # keep SRM fixed (frozen) — training it corrupts HPF
    "tlu_threshold":    3.0,        # YeNet TLU clamp value

    # YedroudjNet / ZhuNet / SIAStegNet
    "abs_layer":        True,       # apply |·| after HPF (YedroudjNet)
    "clamp_val":        3.0,        # clamp after HPF

    # SIAStegNet
    "ca_reduction":     8,          # channel attention reduction ratio
    "dropout":          0.5,        # dropout before FC

    # EfficientNetSteg
    "freeze_backbone":  False,      # freeze EfficientNet backbone (warm-up phase)

    # ── Training ─────────────────────────────────────────────────────────────
    "epochs":           100,
    "batch_size":       32,
    "num_workers":      4,
    "lr":               1e-2,       # SGD base LR (scaled from 0.2 * bs/128)
    "momentum":         0.9,        # SGD momentum
    "nesterov":         True,       # Nesterov momentum
    "weight_decay":     1e-5,

    # LR scheduler: "cosine" | "plateau" | None
    "scheduler":        "plateau",
    "lr_min":           1e-6,       # cosine min LR
    "lr_patience":      5,          # plateau patience (ReduceLROnPlateau)

    # ── Data ─────────────────────────────────────────────────────────────────
    # Dataset selector: "alaska2" | "steganogan"
    "dataset":          "steganogan",
    # "data_root":        os.path.expanduser("/workspace/alaska2-image-steganalysis"),
    "data_root":        os.path.expanduser("/workspace/steganogan-dataset"),
    "crop_size":        512,
    # ALASKA2 algs: ["JMiPOD", "JUNIWARD", "UERD"]
    # SteganoGAN algs: ["basic", "dense", "residual"]
    "stego_algs":       ["basic", "dense", "residual"],
    "val_frac":         0.1,
    "test_frac":        0.1,
    "balanced":         True,       # balance cover/stego per batch
    "max_samples":      None,       # None = use all; int = quick experiment cap

    # ── Checkpointing ─────────────────────────────────────────────────────────
    "monitor":          "auc_roc",  # metric to track for best checkpoint label
}


# ── Network registry ───────────────────────────────────────────────────────────

def _build_network(cfg: Dict[str, Any]) -> torch.nn.Module:
    choice = cfg["network"].lower()
    common = dict(in_channels=3, num_classes=2)

    if choice == "xunet":
        return XuNet(**common)

    elif choice == "yenet":
        return YeNet(
            **common,
            srm_trainable=cfg.get("srm_trainable", False),
            tlu_threshold=cfg.get("tlu_threshold", 3.0),
        )

    elif choice == "srnet":
        return SRNet(**common)

    elif choice == "yedroudjnet":
        return YedroudjNet(
            **common,
            abs_layer=cfg.get("abs_layer", True),
            clamp_val=cfg.get("clamp_val", 3.0),
        )

    elif choice == "zhunet":
        return ZhuNet(
            **common,
            srm_trainable=cfg.get("srm_trainable", False),
        )

    elif choice == "siastegnet":
        return SIAStegNet(
            **common,
            srm_trainable=cfg.get("srm_trainable", False),
        )

    elif choice == "efficientnetsteg":
        return EfficientNetSteg(
            **common,
            freeze_backbone=cfg.get("freeze_backbone", False),
            dropout=cfg.get("dropout", 0.4),
        )

    else:
        raise ValueError(f"Unknown network: {choice!r}")


def _build_scheduler(optimizer, cfg: Dict[str, Any]):
    choice = cfg.get("scheduler")
    if choice is None:
        return None
    elif choice == "cosine":
        return CosineAnnealingLR(
            optimizer,
            T_max=cfg["epochs"],
            eta_min=cfg.get("lr_min", 1e-6),
        )
    elif choice == "plateau":
        return ReduceLROnPlateau(
            optimizer,
            mode="max",
            patience=cfg.get("lr_patience", 5),
            factor=0.5,
        )
    else:
        raise ValueError(f"Unknown scheduler: {choice!r}")


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    torch.manual_seed(42)

    print("=" * 60)
    print("Steganalysis Training")
    print("=" * 60)
    for k, v in CONFIG.items():
        print(f"  {k:<22}: {v}")
    print()

    # ── Device ────────────────────────────────────────────────────────────────
    if CONFIG["gpu"] and torch.cuda.is_available():
        device = torch.device("cuda")
    elif CONFIG["gpu"] and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}\n")

    # ── Data ──────────────────────────────────────────────────────────────────
    dataset_choice = CONFIG.get("dataset", "alaska2").lower()
    if dataset_choice == "steganogan":
        factory = SteganoganDataLoaderFactory
    elif dataset_choice == "alaska2":
        factory = Alaska2DataLoaderFactory
    else:
        raise ValueError(f"Unknown dataset: {dataset_choice!r}")

    train_loader, val_loader = factory.create(
        root=CONFIG["data_root"],
        batch_size=CONFIG["batch_size"],
        num_workers=CONFIG["num_workers"],
        crop_size=CONFIG["crop_size"],
        stego_algs=CONFIG["stego_algs"],
        val_frac=CONFIG["val_frac"],
        test_frac=CONFIG["test_frac"],
        balanced=CONFIG["balanced"],
        max_samples=CONFIG.get("max_samples"),
        pin_memory=(device.type == "cuda"),
    )
    print(f"Train: {len(train_loader.dataset)}")
    print(f"Val  : {len(val_loader.dataset)}\n")

    # ── Model ─────────────────────────────────────────────────────────────────
    network = _build_network(CONFIG)
    print(f"Network : {type(network).__name__}")
    print(f"Params  : {network.num_parameters:,}\n")

    # ── Log dir ───────────────────────────────────────────────────────────────
    log_dir = os.path.join(
        os.path.dirname(__file__), "runs",
        f"{CONFIG['network']}_{int(time())}",
    )
    os.makedirs(log_dir, exist_ok=True)
    with open(os.path.join(log_dir, "config.json"), "w") as f:
        json.dump(CONFIG, f, indent=2, default=str)

    # ── Optimiser & scheduler ─────────────────────────────────────────────────
    optimizer = SGD(
        network.parameters(),
        lr=CONFIG["lr"],
        momentum=CONFIG.get("momentum", 0.9),
        nesterov=CONFIG.get("nesterov", True),
        weight_decay=CONFIG.get("weight_decay", 1e-5),
    )
    scheduler = _build_scheduler(optimizer, CONFIG)

    # ── Callbacks ─────────────────────────────────────────────────────────────
    callbacks = [
        LRMonitor(),
        MetricsLogger(
            log_path=os.path.join(log_dir, "metrics.tsv"),
            verbose=False,
        ),
        CheckpointSaver(
            checkpoint_dir=log_dir,
            monitor=CONFIG["monitor"],
            mode="max",
            save_best_only=False,
            save_every=1,
            verbose=True,
            config=CONFIG,
        ),
    ]

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer = Trainer(
        model=network,
        optimizer=optimizer,
        device=device,
        scheduler=scheduler,
        verbose=True,
        callbacks=callbacks,
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    print(f"Logs → {log_dir}\n")
    history = trainer.fit(train_loader, val_loader, epochs=CONFIG["epochs"])

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("Training complete.")
    print(f"  Log dir   : {log_dir}")

    ckpt_saver = next((cb for cb in callbacks if isinstance(cb, CheckpointSaver)), None)
    if ckpt_saver and ckpt_saver.best_path:
        print(f"  Best ckpt : {ckpt_saver.best_path}")

    if history:
        last_auc = history.get("auc_roc", [None])[-1]
        last_acc = history.get("accuracy", [None])[-1]
        if last_auc is not None:
            print(f"  Final AUC : {last_auc:.4f}")
        if last_acc is not None:
            print(f"  Final acc : {last_acc:.4f}")
    print("=" * 60)

    gc.collect()
    os._exit(0)


if __name__ == "__main__":
    main()

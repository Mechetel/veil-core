#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Steganalysis evaluation / inference script.

Loads a trained checkpoint and evaluates it on the ALASKA2 test split (or
the validation split).  Reports the full metric suite: accuracy, balanced
accuracy, AUC-ROC, TPR@FPR=0.1, precision, recall, F1.

Configure via the CONFIG dict at the top of this file.

Usage
-----
    python steganalyzers/test.py

Output
------
  Prints a metric table to stdout.
  Optionally writes results to <output_dir>/test_results.json.
"""

import json
import os
import sys
from typing import Any, Dict, Optional

os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
from tqdm import tqdm

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

from steganalyzers.models     import XuNet, YeNet, SRNet, YedroudjNet, ZhuNet, SIAStegNet, EfficientNetSteg
from steganalyzers.data       import Alaska2DataLoaderFactory, SteganoganDataLoaderFactory
from steganalyzers.training   import SteganalysisMetrics
from steganalyzers.utils      import Checkpoint

try:
    from torch.amp import autocast
except ImportError:
    from torch.cuda.amp import autocast


# ── Configuration ──────────────────────────────────────────────────────────────

CONFIG: Dict[str, Any] = {
    # ── Hardware ──────────────────────────────────────────────────────────────
    "gpu":          False,

    # ── Network ───────────────────────────────────────────────────────────────
    # Choices: xunet+ | yenet+ | srnet+ | yedroudjnet+ | zhunet- | siastegnet- | efficientnetsteg+
    "network":      "efficientnetsteg",    # must match the checkpoint

    # Network-specific hyper-parameters (only needed if building from scratch)
    "srm_trainable": True,
    "tlu_threshold": 3.0,
    "abs_layer":     True,
    "clamp_val":     3.0,
    "ca_reduction":  8,
    "dropout":       0.5,

    # ── Checkpoint ────────────────────────────────────────────────────────────
    # Path to a .pt checkpoint file produced by train.py.
    # Set to None to evaluate a randomly initialised model (sanity check).
    "checkpoint":   "steganalyzers/runs/efficientnetsteg/efficientnetsteg_1780660488/epoch0041.pt",       # e.g. "steganalyzers/runs/srnet_1234/best_epoch0032.pt"

    # ── Evaluation split ──────────────────────────────────────────────────────
    "split":        "test",     # "test" | "val"

    # ── Data ─────────────────────────────────────────────────────────────────
    # Dataset selector: "alaska2" | "steganogan"
    "dataset":      "steganogan",
    # "data_root":    os.path.expanduser("~/Projects/datasets/alaska2-image-steganalysis"),
    "data_root":    os.path.expanduser("~/Projects/datasets/steganogan-dataset"),
    "crop_size":    512,
    "batch_size":   32,
    "num_workers":  0,
    # ALASKA2 algs: ["JMiPOD", "JUNIWARD", "UERD"]
    # SteganoGAN algs: ["basic", "dense", "residual"]
    "stego_algs":   ["basic", "dense", "residual"],
    "val_frac":     0.1,
    "test_frac":    0.1,

    # ── Output ────────────────────────────────────────────────────────────────
    "output_dir":   'steganalyzers/runs/efficientnetsteg/efficientnetsteg_1780660488',       # if set, write test_results.json here
    "verbose":      True,
}


# ── Network registry ───────────────────────────────────────────────────────────

def _build_network(cfg: Dict[str, Any]) -> torch.nn.Module:
    """Build a network instance from CONFIG."""
    choice = cfg["network"].lower()
    common = dict(in_channels=3, num_classes=2)

    if choice == "xunet":
        return XuNet(**common)
    elif choice == "yenet":
        return YeNet(**common,
                     srm_trainable=cfg.get("srm_trainable", True),
                     tlu_threshold=cfg.get("tlu_threshold", 3.0))
    elif choice == "srnet":
        return SRNet(**common)
    elif choice == "yedroudjnet":
        return YedroudjNet(**common,
                           abs_layer=cfg.get("abs_layer", True),
                           clamp_val=cfg.get("clamp_val", 3.0))
    elif choice == "zhunet":
        return ZhuNet(**common, srm_trainable=cfg.get("srm_trainable", True))
    elif choice == "siastegnet":
        return SIAStegNet(**common,
                          srm_trainable=cfg.get("srm_trainable", True),
                          ca_reduction=cfg.get("ca_reduction", 8),
                          dropout=cfg.get("dropout", 0.5))
    elif choice == "efficientnetsteg":
        return EfficientNetSteg(**common,
                                freeze_backbone=cfg.get("freeze_backbone", False),
                                dropout=cfg.get("dropout", 0.4))
    else:
        raise ValueError(f"Unknown network: {choice!r}")


# ── Evaluation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(
    model:      torch.nn.Module,
    loader:     torch.utils.data.DataLoader,
    device:     torch.device,
    verbose:    bool = True,
) -> Dict[str, float]:
    """
    Run inference on *loader* and return the full metric suite.

    Parameters
    ----------
    model   : trained steganalyzer (already on *device*)
    loader  : DataLoader for the evaluation split
    device  : compute device
    verbose : show tqdm progress bar

    Returns
    -------
    dict: accuracy, balanced_accuracy, auc_roc, tpr_at_fpr10, precision, recall, f1
    """
    model.eval()
    amp_device = device.type if isinstance(device, torch.device) else str(device)
    use_amp    = (amp_device == "cuda")

    all_logits, all_labels = [], []

    pbar = tqdm(loader, disable=not verbose, desc="Eval ",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]")

    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        with autocast(device_type=amp_device, enabled=use_amp):
            logits = model(images)
        all_logits.append(logits.cpu())
        all_labels.append(labels)

        # Live accuracy display
        acc = (logits.argmax(1).cpu() == labels).float().mean().item()
        pbar.set_postfix(acc=f"{acc:.4f}")

    all_logits_t = torch.cat(all_logits)
    all_labels_t = torch.cat(all_labels)
    return SteganalysisMetrics.from_logits(all_logits_t, all_labels_t)


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("Steganalysis Evaluation")
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
    split          = CONFIG.get("split", "test")
    dataset_choice = CONFIG.get("dataset", "alaska2").lower()
    if dataset_choice == "steganogan":
        factory = SteganoganDataLoaderFactory
    elif dataset_choice == "alaska2":
        factory = Alaska2DataLoaderFactory
    else:
        raise ValueError(f"Unknown dataset: {dataset_choice!r}")

    if split == "test":
        loader = factory.create_test(
            root=CONFIG["data_root"],
            batch_size=CONFIG["batch_size"],
            num_workers=CONFIG["num_workers"],
            crop_size=CONFIG["crop_size"],
            stego_algs=CONFIG["stego_algs"],
            val_frac=CONFIG["val_frac"],
            test_frac=CONFIG["test_frac"],
            pin_memory=(device.type == "cuda"),
        )
    else:
        _, loader = factory.create(
            root=CONFIG["data_root"],
            batch_size=CONFIG["batch_size"],
            num_workers=CONFIG["num_workers"],
            crop_size=CONFIG["crop_size"],
            stego_algs=CONFIG["stego_algs"],
            val_frac=CONFIG["val_frac"],
            test_frac=CONFIG["test_frac"],
            pin_memory=(device.type == "cuda"),
        )

    print(f"Evaluating on {split!r} split: {len(loader.dataset)} images\n")

    # ── Model ─────────────────────────────────────────────────────────────────
    network = _build_network(CONFIG).to(device)

    ckpt_path = CONFIG.get("checkpoint")
    if ckpt_path:
        bundle = torch.load(ckpt_path, map_location=device, weights_only=False)
        network.load_state_dict(bundle["model_state"])
        loaded_epoch = bundle.get("epoch", "?")
        print(f"Loaded checkpoint: {ckpt_path}  (epoch {loaded_epoch})\n")
    else:
        print("Warning: no checkpoint specified — using randomly initialised model.\n")

    # ── Evaluate ──────────────────────────────────────────────────────────────
    metrics = evaluate(network, loader, device, verbose=CONFIG.get("verbose", True))

    # ── Print results ─────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"Results on {split!r} split")
    print(f"  Network : {CONFIG['network']}")
    print(f"  Samples : {len(loader.dataset)}")
    print("-" * 60)
    print(f"  Accuracy          : {metrics['accuracy']:.4f}")
    print(f"  Balanced accuracy : {metrics['balanced_accuracy']:.4f}")
    print(f"  AUC-ROC           : {metrics['auc_roc']:.4f}")
    print(f"  TPR @ FPR=0.10    : {metrics['tpr_at_fpr10']:.4f}")
    print(f"  Precision         : {metrics['precision']:.4f}")
    print(f"  Recall            : {metrics['recall']:.4f}")
    print(f"  F1                : {metrics['f1']:.4f}")
    print("=" * 60)

    # ── Save results ──────────────────────────────────────────────────────────
    output_dir = CONFIG.get("output_dir")
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        results = {
            "network":    CONFIG["network"],
            "checkpoint": ckpt_path,
            "split":      split,
            "num_samples": len(loader.dataset),
            "metrics":    metrics,
        }
        out_path = os.path.join(output_dir, "test_results.json")
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved → {out_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Full-dataset ROC / AUC analysis for a trained steganalyzer.

Loads ALL labelled images from Cover + stego directories (no train/val/test
split) and runs inference with a saved checkpoint.  Outputs:
  • metrics printed to stdout
  • <run_dir>/roc_curve.png  — ROC plot
  • <run_dir>/roc_results.json — metrics + curve arrays

Usage
-----
    python steganalyzers/analyze_roc.py

Configure the paths in CONFIG below.
"""

import json
import os
import sys
from pathlib import Path
from typing import List, Sequence, Tuple

os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

from steganalyzers.models import XuNet, YeNet, SRNet, YedroudjNet, ZhuNet, SIAStegNet, EfficientNetSteg

try:
    from torch.amp import autocast
    _AMP_SUPPORTS_DEVICE_TYPE = True
except ImportError:
    from torch.cuda.amp import autocast
    _AMP_SUPPORTS_DEVICE_TYPE = False


# ── Configuration ──────────────────────────────────────────────────────────────

CONFIG = {
    # ── Hardware ──────────────────────────────────────────────────────────────
    "gpu":          False,

    # ── Dataset ───────────────────────────────────────────────────────────────
    # "alaska2"   → Cover/ + JMiPOD/JUNIWARD/UERD/  (*.jpg)
    # "steganogan"→ cover/ + basic/dense/residual/  (*.png)
    "dataset":    "steganogan",

    # ── Network ───────────────────────────────────────────────────────────────
    # "network":      "XuNet",
    # "network":      "YeNet",
    # "network":      "YedroudjNet",
    # "network":      "SRNet",
    "network":      "EfficientNetSteg",

    # Network hyper-params (must match checkpoint)
    "srm_trainable": False,
    "tlu_threshold": 3.0,
    "abs_layer":     True,
    "clamp_val":     3.0,
    "ca_reduction":  8,
    "dropout":       0.4,
    "freeze_backbone": False,

    # ── Checkpoint ────────────────────────────────────────────────────────────
    "checkpoint": (
        "/Users/dmitryhoma/Projects/phd_dissertation/state_3/"
        "Attention-Steganogan/steganalyzers/runs/efficientnetsteg/"
        # "xunet_1780657578/epoch0040.pt"
        # "yenet_1780658225/epoch0040.pt"
        # "yedroudjnet_1780659810/epoch0041.pt"
        # "srnet_1780658944/epoch0030.pt"
        "efficientnetsteg_1780660488/epoch0041.pt"
    ),

    # ── Data ─────────────────────────────────────────────────────────────────
    # ALASKA2 default: "/Users/dmitryhoma/Projects/datasets/alaska2-image-steganalysis"
    "data_root":  "/Users/dmitryhoma/Projects/datasets/steganogan-dataset",
    # ALASKA2: ["JMiPOD", "JUNIWARD", "UERD"]
    # SteganoGAN: ["basic", "dense", "residual"]
    "stego_algs": ["basic", "dense", "residual"],
    "crop_size":  512,
    "batch_size": 32,
    "num_workers": 4,
    "max_cover":  500,
    "max_stego":  500,

    # ── Output ────────────────────────────────────────────────────────────────
    "output_dir": (
        "/Users/dmitryhoma/Projects/phd_dissertation/state_3/"
        "Attention-Steganogan/steganalyzers/runs/efficientnetsteg/"
        # "xunet_1780657578"
        # "yenet_1780658225"
        # "yedroudjnet_1780659810"
        # "srnet_1780658944"
        "efficientnetsteg_1780660488"
    ),
}


# ── Full-dataset loader ────────────────────────────────────────────────────────

class FullAlaska2Dataset(Dataset):
    """Labelled images from Cover + stego dirs — no train/val split."""

    def __init__(
        self,
        root:        str,
        stego_algs:  Sequence[str],
        transform,
        max_cover:   int = 500,
        max_stego:   int = 500,
    ) -> None:
        root = Path(root).expanduser()
        samples: List[Tuple[Path, int]] = []

        cover_paths = sorted((root / "Cover").glob("*.jpg"))[:max_cover]
        for p in cover_paths:
            samples.append((p, 0))

        # Distribute max_stego evenly across algorithms
        per_alg = max_stego // len(stego_algs)
        for alg in stego_algs:
            alg_paths = sorted((root / alg).glob("*.jpg"))[:per_alg]
            for p in alg_paths:
                samples.append((p, 1))

        self.samples   = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label


class FullSteganoganDataset(Dataset):
    """Labelled images from cover + basic/dense/residual dirs — no split."""

    def __init__(
        self,
        root:        str,
        stego_algs:  Sequence[str],
        transform,
        max_cover:   int = 500,
        max_stego:   int = 500,
    ) -> None:
        root = Path(root).expanduser()
        samples: List[Tuple[Path, int]] = []

        cover_paths = sorted((root / "cover").glob("*.png"))[:max_cover]
        for p in cover_paths:
            samples.append((p, 0))

        per_alg = max_stego // len(stego_algs)
        for alg in stego_algs:
            alg_paths = sorted((root / alg).glob("*.png"))[:per_alg]
            for p in alg_paths:
                samples.append((p, 1))

        self.samples   = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label


# ── Network builder ───────────────────────────────────────────────────────────

def _build_network(cfg) -> nn.Module:
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
                          srm_trainable=cfg.get("srm_trainable", False))
    elif choice == "efficientnetsteg":
        return EfficientNetSteg(**common,
                                freeze_backbone=cfg.get("freeze_backbone", False),
                                dropout=cfg.get("dropout", 0.4))
    else:
        raise ValueError(f"Unknown network: {choice!r}")


# ── Inference ────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_inference(model, loader, device) -> Tuple[np.ndarray, np.ndarray]:
    """Returns (probs_stego, labels) as numpy arrays."""
    model.eval()
    amp_device = device.type
    use_amp    = (amp_device == "cuda")

    all_probs, all_labels = [], []

    pbar = tqdm(loader, desc="Inference",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]")

    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        if _AMP_SUPPORTS_DEVICE_TYPE:
            ctx = autocast(device_type=amp_device, enabled=use_amp)
        else:
            ctx = autocast(enabled=use_amp)
        with ctx:
            logits = model(images)

        probs = torch.softmax(logits.float(), dim=1)[:, 1]
        all_probs.append(probs.cpu().numpy())
        all_labels.append(labels.numpy())

    return np.concatenate(all_probs), np.concatenate(all_labels)


# ── Metrics ──────────────────────────────────────────────────────────────────

def compute_roc(probs: np.ndarray, labels: np.ndarray):
    """Returns (fpr_vals, tpr_vals, auc)."""
    order        = np.argsort(-probs)
    sorted_lbl   = labels[order]
    n_pos = int(sorted_lbl.sum())
    n_neg = len(sorted_lbl) - n_pos

    fpr_vals, tpr_vals = [0.0], [0.0]
    tp = fp = 0
    for lbl in sorted_lbl:
        if lbl == 1:
            tp += 1
        else:
            fp += 1
        tpr_vals.append(tp / n_pos)
        fpr_vals.append(fp / n_neg)
    tpr_vals.append(1.0)
    fpr_vals.append(1.0)

    fpr_arr = np.array(fpr_vals)
    tpr_arr = np.array(tpr_vals)
    trapz   = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    auc     = float(trapz(tpr_arr, fpr_arr))
    return fpr_arr, tpr_arr, auc


def compute_metrics(probs: np.ndarray, labels: np.ndarray, fpr_target=0.1):
    preds = (probs >= 0.5).astype(int)

    acc     = float((preds == labels).mean())
    classes = np.unique(labels)
    recalls = [(preds[labels == c] == c).mean() for c in classes]
    bal_acc = float(np.mean(recalls))

    fpr_arr, tpr_arr, auc = compute_roc(probs, labels)

    # TPR @ FPR = fpr_target
    idx    = np.searchsorted(fpr_arr, fpr_target)
    tpr10  = float(tpr_arr[min(idx, len(tpr_arr) - 1)])

    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    return {
        "accuracy":          acc,
        "balanced_accuracy": bal_acc,
        "auc_roc":           auc,
        "tpr_at_fpr10":      tpr10,
        "precision":         float(prec),
        "recall":            float(rec),
        "f1":                float(f1),
    }, fpr_arr, tpr_arr


# ── Plot ─────────────────────────────────────────────────────────────────────

def plot_roc(fpr: np.ndarray, tpr: np.ndarray, auc: float, out_path: str,
             network_name: str = "", dataset_name: str = "ALASKA2") -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping ROC plot.")
        return

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color="#2196F3", lw=2, label=f"{network_name}  AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--", label="Random")
    ax.axvline(x=0.1, color="#F44336", lw=1, linestyle=":", label="FPR = 0.10")
    ax.set_xlabel("False Positive Rate", fontsize=13)
    ax.set_ylabel("True Positive Rate", fontsize=13)
    ax.set_title(f"ROC Curve — {network_name} on {dataset_name} (full dataset)", fontsize=14)
    ax.legend(fontsize=11)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"ROC plot saved → {out_path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    network_name   = CONFIG["network"]
    dataset_choice = CONFIG.get("dataset", "alaska2").lower()
    dataset_label  = {"alaska2": "ALASKA2", "steganogan": "SteganoGAN"}.get(
        dataset_choice, dataset_choice
    )
    print("=" * 60)
    print(f"{network_name} — Full {dataset_label} ROC / AUC Analysis")
    print("=" * 60)

    # Device
    if CONFIG["gpu"] and torch.cuda.is_available():
        device = torch.device("cuda")
    elif CONFIG["gpu"] and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device : {device}")

    # Data
    transform = transforms.Compose([
        transforms.CenterCrop(CONFIG["crop_size"]),
        transforms.ToTensor(),
    ])
    if dataset_choice == "steganogan":
        dataset_cls = FullSteganoganDataset
    elif dataset_choice == "alaska2":
        dataset_cls = FullAlaska2Dataset
    else:
        raise ValueError(f"Unknown dataset: {dataset_choice!r}")

    dataset = dataset_cls(
        root=CONFIG["data_root"],
        stego_algs=CONFIG["stego_algs"],
        transform=transform,
        max_cover=CONFIG.get("max_cover", 500),
        max_stego=CONFIG.get("max_stego", 500),
    )
    n_cover = sum(1 for _, l in dataset.samples if l == 0)
    n_stego = sum(1 for _, l in dataset.samples if l == 1)
    print(f"Images : {len(dataset)}  (cover={n_cover}, stego={n_stego})\n")

    loader = DataLoader(
        dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=CONFIG["num_workers"],
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    # Model
    model = _build_network(CONFIG).to(device)
    ckpt_path = CONFIG["checkpoint"]
    bundle = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(bundle["model_state"])
    epoch = bundle.get("epoch", "?")
    print(f"Checkpoint : {ckpt_path}")
    print(f"Epoch      : {epoch}\n")

    # Inference
    probs, labels = run_inference(model, loader, device)

    # Metrics
    metrics, fpr_arr, tpr_arr = compute_metrics(probs, labels)

    print(f"\n{'=' * 60}")
    print(f"Results — full {dataset_label} dataset")
    print(f"  Network : {CONFIG['network']}")
    print(f"  Samples : {len(dataset)}")
    print("-" * 60)
    print(f"  Accuracy          : {metrics['accuracy']:.4f}")
    print(f"  Balanced accuracy : {metrics['balanced_accuracy']:.4f}")
    print(f"  AUC-ROC           : {metrics['auc_roc']:.4f}")
    print(f"  TPR @ FPR=0.10    : {metrics['tpr_at_fpr10']:.4f}")
    print(f"  Precision         : {metrics['precision']:.4f}")
    print(f"  Recall            : {metrics['recall']:.4f}")
    print(f"  F1                : {metrics['f1']:.4f}")
    print("=" * 60)

    # Save
    out_dir = CONFIG["output_dir"]
    os.makedirs(out_dir, exist_ok=True)

    results = {
        "network":    CONFIG["network"],
        "dataset":    dataset_choice,
        "checkpoint": ckpt_path,
        "num_samples": len(dataset),
        "num_cover":   n_cover,
        "num_stego":   n_stego,
        "metrics":    metrics,
        "roc": {
            "fpr": fpr_arr.tolist(),
            "tpr": tpr_arr.tolist(),
        },
    }
    json_path = os.path.join(out_dir, "roc_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved → {json_path}")

    plot_roc(fpr_arr, tpr_arr, metrics["auc_roc"],
             out_path=os.path.join(out_dir, "roc_curve.png"),
             network_name=network_name,
             dataset_name=dataset_label)


if __name__ == "__main__":
    main()

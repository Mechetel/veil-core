#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ===================================================================================
# Summary — AUC-ROC
# ===================================================================================
# Steganalyzer                 SteganoGAN-Dense        Edge-UNet           Edge-ASPP
# -----------------------------------------------------------------------------------
# XuNet                             0.3534              0.5534              0.4772
# YeNet                             0.4281              0.5601              0.5343
# YedroudjNet                       0.4466              0.4123              0.6436
# SRNet                             0.2645              0.2578              0.4165
# EfficientNetSteg                  0.5006              0.5612              0.6344
# ===================================================================================

"""
ROC / AUC analysis of all trained steganalyzers on the generated
SteganoGAN stego-image dataset (steganogan-dense, edge-unet, edge-aspp).

Produces one ROC plot per steganalyzer with 3 curves (one per steganogan
variant) and saves metrics to JSON.

Usage:
    python steganalyzers/analyze_steganogan_roc.py
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

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

from steganalyzers.models import (
    XuNet, YeNet, SRNet, YedroudjNet, ZhuNet, SIAStegNet, EfficientNetSteg,
)

try:
    from torch.amp import autocast
    _AMP_SUPPORTS_DEVICE_TYPE = True
except ImportError:
    from torch.cuda.amp import autocast
    _AMP_SUPPORTS_DEVICE_TYPE = False


# ── Configuration ─────────────────────────────────────────────────────────────

DATASET_ROOT = os.path.expanduser("~/Projects/datasets/khoma-stego-images")

STEGO_METHODS = {
    "SteganoGAN-Dense": "steganogan-dense",
    "Edge-UNet":        "edge-unet",
    "Edge-ASPP":        "edge-aspp",
}

RUNS_DIR = os.path.join(_HERE, "runs")

STEGANALYZERS = {
    "XuNet": {
        "network":    "XuNet",
        "checkpoint": os.path.join(RUNS_DIR, "xunet/xunet_1780657578/epoch0040.pt"),
        "params":     {},
    },
    "YeNet": {
        "network":    "YeNet",
        "checkpoint": os.path.join(RUNS_DIR, "yenet/yenet_1780658225/epoch0040.pt"),
        "params":     {"srm_trainable": False, "tlu_threshold": 3.0},
    },
    "YedroudjNet": {
        "network":    "YedroudjNet",
        "checkpoint": os.path.join(RUNS_DIR, "yedroudjnet/yedroudjnet_1780659810/epoch0041.pt"),
        "params":     {"abs_layer": True, "clamp_val": 3.0},
    },
    "SRNet": {
        "network":    "SRNet",
        "checkpoint": os.path.join(RUNS_DIR, "srnet/srnet_1780658944/epoch0030.pt"),
        "params":     {},
    },
    "EfficientNetSteg": {
        "network":    "EfficientNetSteg",
        "checkpoint": os.path.join(RUNS_DIR, "efficientnetsteg/efficientnetsteg_1780660488/epoch0041.pt"),
        "params":     {"freeze_backbone": False, "dropout": 0.4},
    },
}

GPU = False
BATCH_SIZE = 1
NUM_WORKERS = 0
OUTPUT_DIR = os.path.join(RUNS_DIR, "steganogan_roc")


# ── Dataset ───────────────────────────────────────────────────────────────────

class CoverStegoDataset(Dataset):
    """Pairs cover images (label=0) with stego images (label=1)."""

    def __init__(self, cover_dir: str, stego_dir: str, transform) -> None:
        self.transform = transform
        self.samples: List[Tuple[str, int]] = []

        for fname in sorted(os.listdir(cover_dir)):
            if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                self.samples.append((os.path.join(cover_dir, fname), 0))

        for fname in sorted(os.listdir(stego_dir)):
            if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                self.samples.append((os.path.join(stego_dir, fname), 1))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label


# ── Network builder ───────────────────────────────────────────────────────────

def _build_network(name: str, params: dict) -> nn.Module:
    common = dict(in_channels=3, num_classes=2)
    choice = name.lower()
    if choice == "xunet":
        return XuNet(**common)
    elif choice == "yenet":
        return YeNet(**common,
                     srm_trainable=params.get("srm_trainable", True),
                     tlu_threshold=params.get("tlu_threshold", 3.0))
    elif choice == "srnet":
        return SRNet(**common)
    elif choice == "yedroudjnet":
        return YedroudjNet(**common,
                           abs_layer=params.get("abs_layer", True),
                           clamp_val=params.get("clamp_val", 3.0))
    elif choice == "zhunet":
        return ZhuNet(**common, srm_trainable=params.get("srm_trainable", True))
    elif choice == "siastegnet":
        return SIAStegNet(**common, srm_trainable=params.get("srm_trainable", False))
    elif choice == "efficientnetsteg":
        return EfficientNetSteg(**common,
                                freeze_backbone=params.get("freeze_backbone", False),
                                dropout=params.get("dropout", 0.4))
    else:
        raise ValueError(f"Unknown network: {name!r}")


# ── Inference ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_inference(model: nn.Module, loader: DataLoader,
                  device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    amp_device = device.type
    use_amp = (amp_device == "cuda")

    all_probs, all_labels = [], []

    for images, labels in tqdm(loader, desc="  Inference",
                               bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"):
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


# ── ROC / Metrics ─────────────────────────────────────────────────────────────

def compute_roc(probs: np.ndarray, labels: np.ndarray):
    order = np.argsort(-probs)
    sorted_lbl = labels[order]
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
    trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    auc = float(trapz(tpr_arr, fpr_arr))
    return fpr_arr, tpr_arr, auc


def compute_metrics(probs: np.ndarray, labels: np.ndarray) -> Dict:
    preds = (probs >= 0.5).astype(int)

    acc = float((preds == labels).mean())
    classes = np.unique(labels)
    recalls = [(preds[labels == c] == c).mean() for c in classes]
    bal_acc = float(np.mean(recalls))

    _, _, auc = compute_roc(probs, labels)

    # TPR @ FPR = 0.1
    fpr_arr, tpr_arr, _ = compute_roc(probs, labels)
    idx = np.searchsorted(fpr_arr, 0.1)
    tpr10 = float(tpr_arr[min(idx, len(tpr_arr) - 1)])

    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    return {
        "accuracy":          acc,
        "balanced_accuracy": bal_acc,
        "auc_roc":           auc,
        "tpr_at_fpr10":      tpr10,
        "precision":         float(prec),
        "recall":            float(rec),
        "f1":                float(f1),
    }


# ── Plot ──────────────────────────────────────────────────────────────────────

CURVE_COLORS = ["#2196F3", "#F44336", "#4CAF50"] # Blue, Red, Green (for 3 curves)

def plot_roc_multi(
    curves: List[Tuple[np.ndarray, np.ndarray, float, str]],
    title: str,
    out_path: str,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 7))

    for i, (fpr, tpr, auc, label) in enumerate(curves):
        color = CURVE_COLORS[i % len(CURVE_COLORS)]
        ax.plot(fpr, tpr, color=color, lw=2, label=f"{label}  AUC = {auc:.4f}")

    ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--", label="Random")
    ax.axvline(x=0.1, color="#9E9E9E", lw=1, linestyle=":", alpha=0.6, label="FPR = 0.10")

    ax.set_xlabel("False Positive Rate", fontsize=13)
    ax.set_ylabel("True Positive Rate", fontsize=13)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=11, loc="lower right")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  ROC plot saved → {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 70)
    print("Steganalyzer ROC evaluation on SteganoGAN stego dataset")
    print("=" * 70)

    # Device
    if GPU and torch.cuda.is_available():
        device = torch.device("cuda")
    elif GPU and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}\n")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    cover_dir = os.path.join(DATASET_ROOT, "cover")
    transform = transforms.Compose([transforms.ToTensor()])

    all_results: Dict[str, Dict] = {}

    for analyzer_name, analyzer_cfg in STEGANALYZERS.items():
        print(f"\n{'─' * 70}")
        print(f"Steganalyzer: {analyzer_name}")
        print(f"{'─' * 70}")

        # Load model
        model = _build_network(analyzer_cfg["network"], analyzer_cfg["params"]).to(device)
        bundle = torch.load(analyzer_cfg["checkpoint"], map_location=device, weights_only=False)
        model.load_state_dict(bundle["model_state"])
        epoch = bundle.get("epoch", "?")
        print(f"  Checkpoint: {analyzer_cfg['checkpoint']}")
        print(f"  Epoch: {epoch}")

        curves = []
        analyzer_results = {}

        for method_label, method_dir in STEGO_METHODS.items():
            stego_dir = os.path.join(DATASET_ROOT, method_dir)
            print(f"\n  vs {method_label} ({stego_dir})")

            dataset = CoverStegoDataset(cover_dir, stego_dir, transform)
            n_cover = sum(1 for _, l in dataset.samples if l == 0)
            n_stego = sum(1 for _, l in dataset.samples if l == 1)
            print(f"  Samples: {len(dataset)} (cover={n_cover}, stego={n_stego})")

            loader = DataLoader(
                dataset,
                batch_size=BATCH_SIZE,
                shuffle=False,
                num_workers=NUM_WORKERS,
                pin_memory=(device.type == "cuda"),
                drop_last=False,
            )

            probs, labels = run_inference(model, loader, device)
            fpr_arr, tpr_arr, auc = compute_roc(probs, labels)
            metrics = compute_metrics(probs, labels)

            print(f"  AUC={metrics['auc_roc']:.4f}  "
                  f"Acc={metrics['accuracy']:.4f}  "
                  f"BalAcc={metrics['balanced_accuracy']:.4f}  "
                  f"TPR@FPR0.1={metrics['tpr_at_fpr10']:.4f}")

            curves.append((fpr_arr, tpr_arr, auc, method_label))
            analyzer_results[method_label] = {
                **metrics,
                "fpr": fpr_arr.tolist(),
                "tpr": tpr_arr.tolist(),
            }

        # Plot
        plot_path = os.path.join(OUTPUT_DIR, f"{analyzer_name.lower()}_roc.png")
        plot_roc_multi(
            curves,
            title=f"ROC — {analyzer_name} vs SteganoGAN variants",
            out_path=plot_path,
        )

        all_results[analyzer_name] = analyzer_results

        # Free memory
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Save all metrics
    json_path = os.path.join(OUTPUT_DIR, "results.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll metrics saved → {json_path}")

    # Print summary table
    print(f"\n{'=' * 70}")
    print("Summary — AUC-ROC")
    print(f"{'=' * 70}")
    header = f"{'Steganalyzer':<20}" + "".join(f"{m:>20}" for m in STEGO_METHODS)
    print(header)
    print("-" * 70)
    for analyzer_name, methods in all_results.items():
        row = f"{analyzer_name:<20}"
        for method_label in STEGO_METHODS:
            row += f"{methods[method_label]['auc_roc']:>20.4f}"
        print(row)
    print("=" * 70)


def replot() -> None:
    """Rebuild all ROC plots from saved results.json (no inference needed)."""
    json_path = os.path.join(OUTPUT_DIR, "results.json")
    print(f"Loading saved results from {json_path} …")
    with open(json_path) as f:
        all_results = json.load(f)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for analyzer_name, methods in all_results.items():
        curves = []
        for method_label in STEGO_METHODS:
            data = methods[method_label]
            fpr = np.array(data["fpr"])
            tpr = np.array(data["tpr"])
            auc = data["auc_roc"]
            curves.append((fpr, tpr, auc, method_label))

        plot_path = os.path.join(OUTPUT_DIR, f"{analyzer_name.lower()}_roc.png")
        plot_roc_multi(
            curves,
            title=f"ROC — {analyzer_name} vs SteganoGAN variants",
            out_path=plot_path,
        )

    print("Done — all plots rebuilt.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--replot", action="store_true",
                        help="Rebuild plots from saved results.json")
    args = parser.parse_args()

    if args.replot:
        replot()
    else:
        main()

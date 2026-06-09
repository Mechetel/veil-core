#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Plot ROC curves for multiple trained steganalyzers on one figure.

Reads roc_results.json files produced by analyze_roc.py and overlays
all curves on a single plot.

Usage
-----
    python steganalyzers/plot_roc_comparison.py

Configure the list of run directories in RUNS below.
Output is saved to OUTPUT_PATH.
"""

import json
import os
from typing import List, Tuple

import numpy as np

# ── Configuration ──────────────────────────────────────────────────────────────

RUNS: List[Tuple[str, str]] = [
    # (path_to_roc_results.json, display_label)
    (
        "steganalyzers/runs/xunet/xunet_1780657578/roc_results.json",
        "XuNet",
    ),
    (
        "steganalyzers/runs/yenet/yenet_1780658225/roc_results.json",
        "YeNet",
    ),
    (
        "steganalyzers/runs/yedroudjnet/yedroudjnet_1780659810/roc_results.json",
        "YedroudjNet",
    ),
    (
        "steganalyzers/runs/srnet/srnet_1780658944/roc_results.json",
        "SRNet",
    ),
    (
        "steganalyzers/runs/efficientnetsteg/efficientnetsteg_1780660488/roc_results.json",
        "EfficientNetSteg",
    ),
]

OUTPUT_PATH = "steganalyzers/runs/roc_comparison.png"

# ── Colours per model (extend if you add more) ────────────────────────────────

COLORS = [
    "#2196F3",  # blue   — XuNet
    "#4CAF50",  # green  — YeNet
    "#F44336",  # red
    "#FF9800",  # orange
    "#9C27B0",  # purple
    "#00BCD4",  # cyan
    "#795548",  # brown
]

# ── Plot ──────────────────────────────────────────────────────────────────────

def main() -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed.")
        return

    # Resolve paths relative to the project root (one level above this script)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir   = os.path.dirname(script_dir)

    fig, ax = plt.subplots(figsize=(8, 7))

    for i, (rel_path, label) in enumerate(RUNS):
        abs_path = os.path.join(root_dir, rel_path)
        if not os.path.exists(abs_path):
            print(f"Warning: not found — {abs_path}")
            continue

        with open(abs_path) as f:
            data = json.load(f)

        fpr = np.array(data["roc"]["fpr"])
        tpr = np.array(data["roc"]["tpr"])
        m   = data["metrics"]
        auc = m["auc_roc"]
        acc = m["accuracy"]
        bal = m["balanced_accuracy"]

        color = COLORS[i % len(COLORS)]
        ax.plot(fpr, tpr, color=color, lw=2,
                label=f"{label}  AUC={auc:.4f}  acc={acc:.4f}  bal={bal:.4f}")

    # Diagonal reference + FPR=0.1 marker
    ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--", label="Random")
    ax.axvline(x=0.1, color="#607D8B", lw=1, linestyle=":", label="FPR = 0.10")

    ax.set_xlabel("False Positive Rate", fontsize=13)
    ax.set_ylabel("True Positive Rate", fontsize=13)
    ax.set_title("ROC Curve Comparison — ALASKA2", fontsize=14)
    ax.legend(fontsize=10, loc="lower right")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.3)

    out_abs = os.path.join(root_dir, OUTPUT_PATH)
    os.makedirs(os.path.dirname(out_abs), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_abs, dpi=150)
    plt.close(fig)
    print(f"Saved → {out_abs}")

    # Print metric table
    print(f"\n{'Model':<16} {'AUC':>8} {'Acc':>8} {'Bal Acc':>10} {'TPR@FPR10':>12} {'F1':>8}")
    print("-" * 66)
    for rel_path, label in RUNS:
        abs_path = os.path.join(root_dir, rel_path)
        if not os.path.exists(abs_path):
            continue
        with open(abs_path) as f:
            m = json.load(f)["metrics"]
        print(f"{label:<16} {m['auc_roc']:>8.4f} {m['accuracy']:>8.4f} "
              f"{m['balanced_accuracy']:>10.4f} {m['tpr_at_fpr10']:>12.4f} {m['f1']:>8.4f}")


if __name__ == "__main__":
    main()

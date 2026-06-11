#!/usr/bin/env python3
"""Collect + canonically rename model weights from the dissertation repo into
veil-core/weights/.

Steganography checkpoints (.steg) -> weights/steg/<key>.steg
Steganalyzer checkpoints (.pt)    -> weights/analyzers/<key>.pt

The weights/ directory is git-ignored and, in production, is a persistent volume
synced to the server with scripts/sync_weights_to_server.sh. Re-run this whenever
the dissertation models change.
"""
from __future__ import annotations

import glob
import shutil
from pathlib import Path

SRC = Path.home() / "Projects/phd_dissertation/state_3/Attention-Steganogan"
DEST = Path(__file__).resolve().parent.parent / "weights"

# canonical key -> glob (relative to SRC). Latest match (sorted) wins.
STEG: dict[str, str] = {
    "dense-div2k":        "models/div2k/dense_dense/*/weights.steg",
    "edge_unet-div2k":    "models/div2k/edge_unet_dense/*/weights.steg",
}
for ds in ("div2k", "mscoco"):
    for d in (1, 2, 3, 4):
        STEG[f"edge_aspp-{ds}-d{d}"] = (
            f"models/{ds}/edge_aspp_edge_aware_dense/*-d{d}/weights.steg"
        )

NETS = ("efficientnetsteg", "srnet", "xunet", "yedroudjnet", "yenet")
ANALYZERS: dict[str, str] = {}
for net in NETS:
    ANALYZERS[f"{net}-stego"] = f"steganalyzers/runs/{net}/*/epoch*.pt"
    ANALYZERS[f"{net}-alaska2"] = f"steganalyzers/runs_alaska/{net}/*/best_epoch*.pt"


def _pick(pattern: str) -> Path | None:
    matches = sorted(glob.glob(str(SRC / pattern)))
    return Path(matches[-1]) if matches else None


def _collect(mapping: dict[str, str], subdir: str, suffix: str) -> int:
    out = DEST / subdir
    out.mkdir(parents=True, exist_ok=True)
    ok = 0
    for key, pattern in mapping.items():
        src = _pick(pattern)
        target = out / f"{key}{suffix}"
        if src is None:
            print(f"  MISSING  {key:28s}  ({pattern})")
            continue
        shutil.copy2(src, target)
        size_mb = target.stat().st_size / 1e6
        print(f"  ok       {key:28s}  {size_mb:6.1f} MB  <- {src.relative_to(SRC)}")
        ok += 1
    return ok


def main() -> None:
    print(f"Source: {SRC}")
    print(f"Dest:   {DEST}\n")
    print("Steganography models (.steg):")
    n_steg = _collect(STEG, "steg", ".steg")
    print(f"\nSteganalyzers (.pt):")
    n_an = _collect(ANALYZERS, "analyzers", ".pt")
    print(f"\nCollected {n_steg}/{len(STEG)} steg + {n_an}/{len(ANALYZERS)} analyzers")


if __name__ == "__main__":
    main()

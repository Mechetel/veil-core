# -*- coding: utf-8 -*-
"""
SteganoGAN dataset loader for steganalysis.

Dataset structure
-----------------
    steganogan-dataset/
        cover/      *.png   — unmodified cover images, 512×512 (label 0)
        basic/      *.png   — BasicEncoder stego images (label 1)
        dense/      *.png   — DenseEncoder stego images (label 1)
        residual/   *.png   — ResidualEncoder stego images (label 1)

Filenames are paired across folders (0000.png in cover/ matches 0000.png in
each stego folder), which lets us split by filename and pull all variants
together for that split.

Usage
-----
    ds = SteganoganDataset(root="~/datasets/steganogan-dataset", split="train")
    img, label = ds[0]
"""

import random
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset


# ── Supported stego variants ──────────────────────────────────────────────────

STEGO_DIRS: Tuple[str, ...] = ("basic", "dense", "residual")
COVER_DIR:  str             = "cover"


# ── Dataset ───────────────────────────────────────────────────────────────────

class SteganoganDataset(Dataset):
    """
    SteganoGAN binary steganalysis dataset (cover vs stego).

    Parameters
    ----------
    root        : root directory containing cover/, basic/, dense/, residual/
    split       : "train" | "val" | "test"
    transform   : torchvision transform applied to every image
    stego_algs  : which stego variants to include (default: all three)
    val_frac    : fraction of cover+stego pairs reserved for validation
    test_frac   : fraction reserved for testing
    seed        : RNG seed for reproducible splits
    max_samples : hard cap on total samples (useful for quick experiments)
    """

    LABEL_COVER: int = 0
    LABEL_STEGO: int = 1

    def __init__(
        self,
        root:        str,
        split:       str                    = "train",
        transform:   Optional[Callable]     = None,
        stego_algs:  Sequence[str]          = STEGO_DIRS,
        val_frac:    float                  = 0.1,
        test_frac:   float                  = 0.1,
        seed:        int                    = 42,
        max_samples: Optional[int]          = None,
    ) -> None:
        assert split in ("train", "val", "test"), f"Invalid split: {split!r}"
        self.root      = Path(root).expanduser()
        self.split     = split
        self.transform = transform

        cover_dir  = self.root / COVER_DIR
        all_covers = sorted(cover_dir.glob("*.png"))
        filenames  = [p.name for p in all_covers]

        if not filenames:
            raise FileNotFoundError(
                f"No PNG images found in {cover_dir}. "
                "Check that the SteganoGAN dataset is unpacked correctly."
            )

        # Reproducible split
        rng = random.Random(seed)
        rng.shuffle(filenames)

        n      = len(filenames)
        n_test = max(1, int(n * test_frac))
        n_val  = max(1, int(n * val_frac))

        test_files  = filenames[:n_test]
        val_files   = filenames[n_test: n_test + n_val]
        train_files = filenames[n_test + n_val:]

        split_files: List[str] = {
            "train": train_files,
            "val":   val_files,
            "test":  test_files,
        }[split]

        # Build sample list: (path, label)
        samples: List[Tuple[Path, int]] = []
        for fname in split_files:
            cover_path = cover_dir / fname
            if cover_path.exists():
                samples.append((cover_path, self.LABEL_COVER))

            for alg in stego_algs:
                stego_path = self.root / alg / fname
                if stego_path.exists():
                    samples.append((stego_path, self.LABEL_STEGO))

        if max_samples is not None:
            rng.shuffle(samples)
            samples = samples[:max_samples]

        self.samples = samples

    # ── Dataset interface ─────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    # ── Helpers ───────────────────────────────────────────────────────────────

    @property
    def num_cover(self) -> int:
        return sum(1 for _, l in self.samples if l == self.LABEL_COVER)

    @property
    def num_stego(self) -> int:
        return sum(1 for _, l in self.samples if l == self.LABEL_STEGO)

    def __repr__(self) -> str:
        return (
            f"SteganoganDataset(split={self.split!r}, "
            f"cover={self.num_cover}, stego={self.num_stego}, "
            f"total={len(self)})"
        )

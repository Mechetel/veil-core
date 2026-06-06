# -*- coding: utf-8 -*-
"""
ALASKA2 dataset loader.

Dataset structure
-----------------
    alaska2-image-steganalysis/
        Cover/       *.jpg   — unmodified JPEG images, 512×512 (label 0)
        JMiPOD/      *.jpg   — JMiPOD steganography (label 1)
        JUNIWARD/    *.jpg   — J-UNIWARD steganography (label 1)
        UERD/        *.jpg   — UERD steganography (label 1)
        Test/        *.jpg   — unlabeled Kaggle competition images (not used)

Each split (train / val / test) is defined by a list of filenames or by
a fixed random seed + fraction.

Usage
-----
    ds = Alaska2Dataset(root="~/datasets/alaska2", split="train")
    img, label = ds[0]
"""

import os
import random
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


# ── Supported stego algorithms ─────────────────────────────────────────────────

ALASKA2_STEGO_DIRS: Tuple[str, ...] = ("JMiPOD", "JUNIWARD", "UERD")
ALASKA2_COVER_DIR:  str             = "Cover"


# ── Dataset ────────────────────────────────────────────────────────────────────

class Alaska2Dataset(Dataset):
    """
    ALASKA2 binary steganalysis dataset (cover vs stego).

    Parameters
    ----------
    root        : root directory containing Cover/, JMiPOD/, JUNIWARD/, UERD/
    split       : "train" | "val" | "test"
    transform   : torchvision transform applied to every image
    stego_algs  : which stego algorithms to include (default: all three)
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
        stego_algs:  Sequence[str]          = ALASKA2_STEGO_DIRS,
        val_frac:    float                  = 0.1,
        test_frac:   float                  = 0.1,
        seed:        int                    = 42,
        max_samples: Optional[int]          = None,
    ) -> None:
        assert split in ("train", "val", "test"), f"Invalid split: {split!r}"
        self.root      = Path(root).expanduser()
        self.split     = split
        self.transform = transform

        # Collect all cover filenames (basenames only)
        cover_dir  = self.root / ALASKA2_COVER_DIR
        all_covers = sorted(cover_dir.glob("*.jpg"))
        filenames  = [p.name for p in all_covers]

        if not filenames:
            raise FileNotFoundError(
                f"No JPEG images found in {cover_dir}. "
                "Check that ALASKA2 is unpacked correctly."
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
            # Cover
            cover_path = cover_dir / fname
            if cover_path.exists():
                samples.append((cover_path, self.LABEL_COVER))

            # Stego variants
            for alg in stego_algs:
                stego_path = self.root / alg / fname
                if stego_path.exists():
                    samples.append((stego_path, self.LABEL_STEGO))

        if max_samples is not None:
            rng.shuffle(samples)
            samples = samples[:max_samples]

        self.samples = samples

    # ── Dataset interface ──────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    # ── Helpers ────────────────────────────────────────────────────────────────

    @property
    def num_cover(self) -> int:
        return sum(1 for _, l in self.samples if l == self.LABEL_COVER)

    @property
    def num_stego(self) -> int:
        return sum(1 for _, l in self.samples if l == self.LABEL_STEGO)

    def __repr__(self) -> str:
        return (
            f"Alaska2Dataset(split={self.split!r}, "
            f"cover={self.num_cover}, stego={self.num_stego}, "
            f"total={len(self)})"
        )

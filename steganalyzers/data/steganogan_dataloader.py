# -*- coding: utf-8 -*-
"""
DataLoader factory for the SteganoGAN steganalysis dataset.

Mirrors `DataLoaderFactory` (ALASKA2) but binds to `SteganoganDataset`
so we can keep the ALASKA2 path frozen.
"""

from typing import Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms

from .steganogan import SteganoganDataset, STEGO_DIRS


# ── Default transforms ─────────────────────────────────────────────────────────

def _train_transform(crop_size: int) -> transforms.Compose:
    """Augmented pipeline: random crop + flip + tensor in [0,1]."""
    return transforms.Compose([
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        # Pixel values kept in [0, 1] — HPF layers are sensitive to absolute
        # magnitudes, so no mean/std normalisation here.
    ])


def _eval_transform(crop_size: int) -> transforms.Compose:
    """Minimal pipeline: centre crop + tensor."""
    return transforms.Compose([
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
    ])


# ── Factory ────────────────────────────────────────────────────────────────────

class SteganoganDataLoaderFactory:
    """Creates train / val / test DataLoaders for the SteganoGAN dataset."""

    @staticmethod
    def create(
        root:         str,
        batch_size:   int             = 32,
        num_workers:  int             = 4,
        crop_size:    int             = 256,
        stego_algs:   Sequence[str]   = STEGO_DIRS,
        val_frac:     float           = 0.1,
        test_frac:    float           = 0.1,
        seed:         int             = 42,
        balanced:     bool            = True,
        max_samples:  Optional[int]   = None,
        pin_memory:   bool            = True,
    ) -> Tuple[DataLoader, DataLoader]:
        train_ds = SteganoganDataset(
            root=root, split="train",
            transform=_train_transform(crop_size),
            stego_algs=stego_algs,
            val_frac=val_frac, test_frac=test_frac,
            seed=seed, max_samples=max_samples,
        )
        val_ds = SteganoganDataset(
            root=root, split="val",
            transform=_eval_transform(crop_size),
            stego_algs=stego_algs,
            val_frac=val_frac, test_frac=test_frac,
            seed=seed,
        )

        train_sampler = (
            SteganoganDataLoaderFactory._balanced_sampler(train_ds)
            if balanced else None
        )
        shuffle_train = (train_sampler is None)

        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=shuffle_train,
            sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        )
        return train_loader, val_loader

    @staticmethod
    def create_test(
        root:        str,
        batch_size:  int           = 32,
        num_workers: int           = 4,
        crop_size:   int           = 256,
        stego_algs:  Sequence[str] = STEGO_DIRS,
        val_frac:    float         = 0.1,
        test_frac:   float         = 0.1,
        seed:        int           = 42,
        pin_memory:  bool          = True,
    ) -> DataLoader:
        test_ds = SteganoganDataset(
            root=root, split="test",
            transform=_eval_transform(crop_size),
            stego_algs=stego_algs,
            val_frac=val_frac, test_frac=test_frac,
            seed=seed,
        )
        return DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        )

    @staticmethod
    def _balanced_sampler(dataset: SteganoganDataset) -> WeightedRandomSampler:
        """Equalise cover / stego class frequencies (1:3 → ~1:1 per batch)."""
        labels = torch.tensor([lbl for _, lbl in dataset.samples])
        class_counts = torch.bincount(labels)
        class_weights = 1.0 / class_counts.float()
        sample_weights = class_weights[labels]
        return WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )

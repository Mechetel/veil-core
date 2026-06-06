# -*- coding: utf-8 -*-
"""
DataLoader factory for the ALASKA2 steganalysis dataset.

Provides standardised transform pipelines and a single factory entry-point
for creating train / validation / test loaders.
"""

from typing import Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms

from .alaska2 import Alaska2Dataset, ALASKA2_STEGO_DIRS


# ── Default transforms ─────────────────────────────────────────────────────────

def _train_transform(crop_size: int) -> transforms.Compose:
    """Augmented pipeline: random crop + flip + tensor + normalise [0,1]."""
    return transforms.Compose([
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        # Keep pixel values in [0, 1] — steganalysis HPF layers are sensitive
        # to absolute pixel magnitudes so we do NOT mean/std normalise here.
    ])


def _eval_transform(crop_size: int) -> transforms.Compose:
    """Minimal pipeline: centre crop + tensor."""
    return transforms.Compose([
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
    ])


# ── Factory ────────────────────────────────────────────────────────────────────

class Alaska2DataLoaderFactory:
    """
    Creates train / val / test DataLoaders for ALASKA2.

    All factory methods are static; this class is a utility namespace.
    """

    @staticmethod
    def create(
        root:         str,
        batch_size:   int             = 32,
        num_workers:  int             = 4,
        crop_size:    int             = 256,
        stego_algs:   Sequence[str]   = ALASKA2_STEGO_DIRS,
        val_frac:     float           = 0.1,
        test_frac:    float           = 0.1,
        seed:         int             = 42,
        balanced:     bool            = True,
        max_samples:  Optional[int]   = None,
        pin_memory:   bool            = True,
    ) -> Tuple[DataLoader, DataLoader]:
        """
        Create train and validation DataLoaders.

        Parameters
        ----------
        root        : ALASKA2 root directory
        batch_size  : samples per batch
        num_workers : parallel data-loading workers
        crop_size   : spatial crop size (both train and val)
        stego_algs  : which steganography algorithms to include
        val_frac    : fraction of data for validation
        test_frac   : fraction of data for testing (excluded from train+val)
        seed        : reproducibility seed
        balanced    : use WeightedRandomSampler so cover≈stego per batch
        max_samples : cap on total samples per split
        pin_memory  : pin host memory for faster GPU transfer

        Returns
        -------
        (train_loader, val_loader)
        """
        train_ds = Alaska2Dataset(
            root=root, split="train",
            transform=_train_transform(crop_size),
            stego_algs=stego_algs,
            val_frac=val_frac, test_frac=test_frac,
            seed=seed, max_samples=max_samples,
        )
        val_ds = Alaska2Dataset(
            root=root, split="val",
            transform=_eval_transform(crop_size),
            stego_algs=stego_algs,
            val_frac=val_frac, test_frac=test_frac,
            seed=seed,
        )

        train_sampler = (
            Alaska2DataLoaderFactory._balanced_sampler(train_ds) if balanced else None
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
        stego_algs:  Sequence[str] = ALASKA2_STEGO_DIRS,
        val_frac:    float         = 0.1,
        test_frac:   float         = 0.1,
        seed:        int           = 42,
        pin_memory:  bool          = True,
    ) -> DataLoader:
        """Create the test DataLoader (no augmentation, no shuffling)."""
        test_ds = Alaska2Dataset(
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
    def _balanced_sampler(dataset: Alaska2Dataset) -> WeightedRandomSampler:
        """
        WeightedRandomSampler that equalises cover / stego class frequencies.

        Ensures each mini-batch has approximately equal cover and stego
        samples regardless of the dataset's natural class imbalance (which
        can be 1 cover : 3 stego when all three ALASKA2 algorithms are used).
        """
        labels = torch.tensor([lbl for _, lbl in dataset.samples])
        class_counts = torch.bincount(labels)
        # Weight for each sample = 1 / (count of its class)
        class_weights = 1.0 / class_counts.float()
        sample_weights = class_weights[labels]
        return WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )

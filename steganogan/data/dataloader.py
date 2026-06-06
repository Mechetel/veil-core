# -*- coding: utf-8 -*-
"""DataLoader and factory for steganography training/validation datasets."""

import math
from typing import Optional, Tuple

import torch

from .dataset    import SteganographyDataset
from .transforms import TransformFactory
from torchvision import transforms


class SteganographyDataLoader(torch.utils.data.DataLoader):
    """
    DataLoader pre-wired for :class:`SteganographyDataset`.

    Parameters
    ----------
    root        : image folder path (ImageFolder layout)
    transform   : torchvision pipeline (default: TransformFactory.train())
    limit       : optional hard cap on dataset size
    shuffle     : shuffle between epochs
    num_workers : parallel loading workers
    batch_size  : images per mini-batch
    """

    def __init__(
        self,
        root:        str,
        transform:   Optional[transforms.Compose] = None,
        limit:       float = math.inf,
        shuffle:     bool  = True,
        num_workers: int   = 8,
        batch_size:  int   = 4,
        **kwargs,
    ) -> None:
        if transform is None:
            transform = TransformFactory.train()

        super().__init__(
            SteganographyDataset(root, transform, limit),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            **kwargs,
        )


class DataLoaderFactory:
    """Convenience factory that returns matched train / validation loader pairs."""

    @staticmethod
    def create(
        train_root:  str,
        val_root:    str,
        batch_size:  int   = 4,
        num_workers: int   = 8,
        crop_size:   int   = 360,
        train_limit: float = math.inf,
        val_limit:   float = math.inf,
    ) -> Tuple[SteganographyDataLoader, SteganographyDataLoader]:
        """
        Build and return ``(train_loader, val_loader)``.

        Parameters
        ----------
        train_root, val_root : root directories for train / validation images
        batch_size           : mini-batch size
        num_workers          : parallel workers
        crop_size            : spatial crop size for both pipelines
        train_limit          : optional cap on training set size
        val_limit            : optional cap on validation set size
        """
        pin = torch.cuda.is_available()

        train_loader = SteganographyDataLoader(
            root=train_root,
            transform=TransformFactory.train(crop_size),
            limit=train_limit,
            shuffle=True,
            num_workers=num_workers,
            batch_size=batch_size,
            pin_memory=pin,
        )
        val_loader = SteganographyDataLoader(
            root=val_root,
            transform=TransformFactory.validation(crop_size),
            limit=val_limit,
            shuffle=False,
            num_workers=num_workers,
            batch_size=batch_size,
            pin_memory=pin,
        )
        return train_loader, val_loader

# -*- coding: utf-8 -*-
"""Backward-compat shim — use steganogan.data instead."""
from .data.transforms  import TransformFactory
from .data.dataset     import SteganographyDataset as ImageFolder
from .data.dataloader  import SteganographyDataLoader as DataLoader, DataLoaderFactory

TransformBuilder = TransformFactory          # old name alias
DEFAULT_TRANSFORM = TransformFactory.train()

__all__ = ["TransformBuilder", "TransformFactory", "ImageFolder",
           "DataLoader", "DataLoaderFactory", "DEFAULT_TRANSFORM"]

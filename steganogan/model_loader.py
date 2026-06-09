# -*- coding: utf-8 -*-
"""Backward-compat shim — use steganogan.utils.checkpoint instead."""
from .utils.checkpoint import ModelCheckpoint

ModelLoader = ModelCheckpoint               # old name alias
__all__ = ["ModelLoader", "ModelCheckpoint"]

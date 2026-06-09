# -*- coding: utf-8 -*-
"""Backward-compat shim — use steganogan.models.critics instead."""
from .critics import BasicCritic, MultiScaleEdgeAwareCritic
__all__ = ["BasicCritic", "MultiScaleEdgeAwareCritic"]

# -*- coding: utf-8 -*-
"""Backward-compat shim — use steganogan.training.metrics instead."""
from .training.metrics import SteganographyMetrics

MetricsCalculator = SteganographyMetrics     # old name alias
__all__ = ["MetricsCalculator", "SteganographyMetrics"]

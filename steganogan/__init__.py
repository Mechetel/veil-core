# -*- coding: utf-8 -*-
"""
steganogan — public package API.

Typical import:
    from steganogan import SteganoGAN
    from steganogan.models import BasicEncoder, DenseDecoder, BasicCritic
    from steganogan.data   import DataLoaderFactory
"""

from .engine import SteganoGAN

from .models    import (
    BaseEncoder, BaseDecoder, BaseCritic,
    BasicEncoder, ResidualEncoder, DenseEncoder,
    EdgeGuidedDualStreamUNetEncoder,
    BasicDecoder, DenseDecoder, EdgeAwareDenseDecoder,
    BasicCritic, MultiScaleEdgeAwareCritic,
)
from .data      import TransformFactory, SteganographyDataset
from .data      import SteganographyDataLoader, DataLoaderFactory
from .training  import SteganographyLoss, IterativeLoss, SteganographyMetrics, Trainer
from .inference import EncoderService, DecoderService
from .utils     import (
    MessageCodec, DeviceManager, PayloadFactory,
    TrainingHistory, ModelCheckpoint, SampleGridVisualizer,
    SSIMCalculator, ssim,
)

__all__ = [
    # Main class
    "SteganoGAN",
    # Models
    "BaseEncoder", "BaseDecoder", "BaseCritic",
    "BasicEncoder", "ResidualEncoder", "DenseEncoder",
    "EdgeGuidedDualStreamUNetEncoder",
    "BasicDecoder", "DenseDecoder",
    "BasicCritic",
    # Data
    "TransformFactory", "SteganographyDataset",
    "SteganographyDataLoader", "DataLoaderFactory",
    # Training
    "SteganographyLoss", "IterativeLoss", "SteganographyMetrics", "Trainer",
    # Inference
    "EncoderService", "DecoderService",
    # Utils
    "MessageCodec", "DeviceManager", "PayloadFactory",
    "TrainingHistory", "ModelCheckpoint", "SampleGridVisualizer",
    "SSIMCalculator", "ssim",
]

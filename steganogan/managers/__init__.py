# -*- coding: utf-8 -*-
"""
Backward-compat shims for the old managers package.
Use steganogan.training and steganogan.inference instead.
"""
from ..training.trainer      import Trainer as TrainingManager
from ..inference.encoder_service import EncoderService as EncoderManager
from ..inference.decoder_service import DecoderService as DecoderManager
from ..utils.device   import DeviceManager
from ..utils.history  import TrainingHistory as HistoryManager

__all__ = [
    "TrainingManager", "EncoderManager", "DecoderManager",
    "DeviceManager", "HistoryManager",
]

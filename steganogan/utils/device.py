# -*- coding: utf-8 -*-
"""Device selection and model placement utilities."""

import torch
from torch import nn


class DeviceManager:
    """
    Selects the best available compute device and moves models to it.

    Priority order: CUDA → Apple MPS → CPU.

    Parameters
    ----------
    gpu     : request GPU acceleration (CUDA or MPS) when True
    verbose : print device selection messages
    """

    def __init__(self, gpu: bool = True, verbose: bool = False) -> None:
        self.gpu_requested: bool        = gpu
        self.verbose:       bool        = verbose
        self.gpu:           bool        = False
        self.device:        torch.device = torch.device("cpu")
        self._select_device()

    def _select_device(self) -> None:
        """Detect and configure the best available device."""
        if self.gpu_requested:
            if torch.cuda.is_available():
                self.gpu    = True
                self.device = torch.device("cuda")
                self._log("Using NVIDIA GPU (CUDA).")
            elif torch.backends.mps.is_available():
                self.gpu    = True
                self.device = torch.device("mps")
                self._log("Using Apple GPU (MPS).")
            else:
                self.gpu    = False
                self.device = torch.device("cpu")
                self._log("GPU requested but unavailable – falling back to CPU.")
        else:
            self.gpu    = False
            self.device = torch.device("cpu")
            self._log("Using CPU.")

    # kept for backward compat (engine.py calls set_device())
    def set_device(self) -> None:
        """Re-detect and apply the best available device."""
        self._select_device()

    def to_device(self, *models: nn.Module) -> None:
        """Move one or more modules to the selected device in-place."""
        for model in models:
            model.to(self.device)

    def _log(self, message: str) -> None:
        if self.verbose:
            print(message)

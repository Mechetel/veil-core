import torch
from torch import nn


class DeviceManager:
    """Manages device selection and model placement."""

    def __init__(self, gpu: bool = True, verbose: bool = False) -> None:
        self.verbose: bool           = verbose
        self.gpu_requested: bool     = gpu
        self.gpu: bool               = False
        self.device: torch.device    = torch.device('cpu')
        self.set_device()

    def set_device(self) -> None:
        """Detect and configure the best available compute device."""
        if self.gpu_requested:
            if torch.cuda.is_available():
                self.gpu    = True
                self.device = torch.device('cuda')
                if self.verbose:
                    print('Using NVIDIA GPU (CUDA).')
            elif torch.backends.mps.is_available():
                self.gpu    = True
                self.device = torch.device('mps')
                if self.verbose:
                    print('Using Apple GPU (MPS).')
            else:
                self.gpu    = False
                self.device = torch.device('cpu')
                if self.verbose:
                    print('GPU requested but not available. Falling back to CPU.')
        else:
            self.gpu    = False
            self.device = torch.device('cpu')
            if self.verbose:
                print('Using CPU.')

    def to_device(self, *models: nn.Module) -> None:
        """Move one or more nn.Module instances to the current device."""
        for model in models:
            model.to(self.device)

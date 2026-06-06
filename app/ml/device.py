import torch

from app.config import settings


def get_device() -> torch.device:
    """Inference device: CUDA when enabled via env and actually available, else CPU."""
    if settings.cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

# -*- coding: utf-8 -*-
"""Image transformation pipelines for training and validation."""

from torchvision import transforms

_MU    = [0.5, 0.5, 0.5]
_SIGMA = [0.5, 0.5, 0.5]


class TransformFactory:
    """
    Factory for standard steganography image pre-processing pipelines.

    All images are normalised to [-1, 1] via ``Normalize(0.5, 0.5)``.
    """

    @staticmethod
    def train(crop_size: int = 360) -> transforms.Compose:
        """
        Training pipeline: random horizontal flip + random crop + normalise.

        Parameters
        ----------
        crop_size : spatial size of the random crop (default 360)
        """
        return transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(crop_size, pad_if_needed=True),
            transforms.ToTensor(),
            transforms.Normalize(_MU, _SIGMA),
        ])

    @staticmethod
    def validation(crop_size: int = 360) -> transforms.Compose:
        """
        Validation pipeline: deterministic center crop + normalise.

        Parameters
        ----------
        crop_size : spatial size of the center crop (default 360)
        """
        return transforms.Compose([
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize(_MU, _SIGMA),
        ])

    # ── Backward-compat aliases ───────────────────────────────────────────────

    @staticmethod
    def default_transform(crop_size: int = 360) -> transforms.Compose:
        """Alias for :meth:`train` (backward compat)."""
        return TransformFactory.train(crop_size)

    @staticmethod
    def validation_transform(crop_size: int = 360) -> transforms.Compose:
        """Alias for :meth:`validation` (backward compat)."""
        return TransformFactory.validation(crop_size)

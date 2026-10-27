"""Segmentation backends and the factory that picks one."""

from __future__ import annotations

import logging

from ..config import Settings
from ..errors import BackendError
from .base import BackendInfo, SegmentationBackend
from .classical import ClassicalBackend

log = logging.getLogger(__name__)

__all__ = ["BackendInfo", "ClassicalBackend", "SegmentationBackend", "create_backend"]


def create_backend(settings: Settings) -> SegmentationBackend:
    """Build the backend named in ``settings``.

    ``auto`` prefers SAM 2.1 when PyTorch with CUDA and the ``sam2`` package are present,
    and otherwise falls back to the CPU baseline (with a warning in the log).
    """
    if settings.backend == "classical":
        return ClassicalBackend()

    from .sam2_backend import Sam2Backend

    if settings.backend == "sam2":
        return Sam2Backend.from_settings(settings)

    try:
        import torch

        if settings.device in ("auto", "cuda") and not torch.cuda.is_available():
            raise BackendError("no CUDA GPU is available")
        return Sam2Backend.from_settings(settings)
    except (ImportError, BackendError) as exc:
        log.warning("SAM 2.1 is unavailable (%s). Using the CPU baseline instead.", exc)
        return ClassicalBackend()

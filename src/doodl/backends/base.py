"""The interface every segmentation model implements.

The rest of the app only knows this contract, so SAM 2.1, a future SAM 3.1 backend and
the CPU baseline are interchangeable.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..prompts import ObjectPrompt

Masks = dict[int, np.ndarray]


@dataclass(frozen=True)
class BackendInfo:
    name: str  # short id: "sam2" | "classical"
    label: str  # shown in the UI, e.g. "SAM 2.1 tiny"
    device: str  # "cuda" | "cpu" | ...


class SegmentationBackend(ABC):
    """A promptable video object segmenter.

    A *handle* is an opaque per-video session created by :meth:`open_session`. All masks
    are boolean arrays at the resolution of the frames in ``frames_dir``.
    """

    info: BackendInfo

    @abstractmethod
    def open_session(self, frames_dir: Path, num_frames: int) -> Any:
        """Load a video's frames (``00000.jpg`` ...) and return a handle."""

    @abstractmethod
    def set_prompt(
        self, handle: Any, frame_idx: int, obj_id: int, prompt: ObjectPrompt
    ) -> np.ndarray:
        """Set the complete prompt for one object on one frame and return its mask there.

        Calling it again for the same (object, frame) *replaces* the earlier prompt, so
        callers pass the full accumulated outline and clicks every time.
        """

    @abstractmethod
    def propagate(
        self, handle: Any, *, start_frame: int, reverse: bool, obj_ids: Sequence[int]
    ) -> Iterator[tuple[int, Masks]]:
        """Track from ``start_frame`` to the end (or to frame 0 when ``reverse``).

        Yields ``(frame_idx, {obj_id: mask})``. The start frame is included.
        """

    @abstractmethod
    def remove_object(self, handle: Any, obj_id: int) -> None:
        """Forget an object."""

    @abstractmethod
    def close_session(self, handle: Any) -> None:
        """Release the session's memory."""

    def reset_stats(self) -> None:  # noqa: B027 - optional hook
        """Start measuring (for example, GPU peak memory)."""

    def stats(self) -> dict[str, float]:
        """Measurements since :meth:`reset_stats`, such as ``{"peak_gpu_gb": 2.3}``."""
        return {}

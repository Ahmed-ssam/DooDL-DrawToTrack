"""CPU baseline: GrabCut for the first frame, optical flow plus GrabCut to follow it.

It exists for three reasons: the app runs anywhere without a GPU, the tests need a
deterministic model, and the benchmark needs a classical baseline to compare SAM 2.1
against. It is much weaker than SAM 2.1 on occlusion, fast motion and similar colours.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..errors import PromptError
from ..prompts import ObjectPrompt, clean_mask, grabcut_refine, rough_mask
from .base import BackendInfo, Masks, SegmentationBackend


@dataclass
class _Handle:
    frames_dir: Path
    num_frames: int
    seeds: dict[int, dict[int, np.ndarray]] = field(default_factory=dict)


class ClassicalBackend(SegmentationBackend):
    info = BackendInfo(
        name="classical", label="Classical baseline (GrabCut + optical flow)", device="cpu"
    )

    def __init__(self, work_long_side: int = 480, grabcut_iterations: int = 2) -> None:
        self.work_long_side = work_long_side
        self.grabcut_iterations = grabcut_iterations

    # ------------------------------------------------------------------ session

    def open_session(self, frames_dir: Path, num_frames: int) -> _Handle:
        return _Handle(frames_dir=frames_dir, num_frames=num_frames)

    def close_session(self, handle: _Handle) -> None:
        handle.seeds.clear()

    def remove_object(self, handle: _Handle, obj_id: int) -> None:
        handle.seeds.pop(obj_id, None)

    def _read(self, handle: _Handle, index: int) -> np.ndarray:
        frame = cv2.imread(str(handle.frames_dir / f"{index:05d}.jpg"))
        if frame is None:
            raise PromptError(f"Frame {index} could not be read.")
        return frame

    # ------------------------------------------------------------------- prompt

    def set_prompt(
        self, handle: _Handle, frame_idx: int, obj_id: int, prompt: ObjectPrompt
    ) -> np.ndarray:
        frame = self._read(handle, frame_idx)
        shape = frame.shape[:2]
        rough = rough_mask(prompt, shape)
        positives = [p for p, lab in zip(prompt.points, prompt.labels, strict=True) if lab == 1]
        negatives = [p for p, lab in zip(prompt.points, prompt.labels, strict=True) if lab == 0]
        if rough is None:
            if not positives:
                raise PromptError("Draw an outline around the object, or click on it.")
            rough = np.zeros(shape, bool)
            for x, y in positives:
                cv2.circle(
                    rough.view(np.uint8), (round(x), round(y)), max(12, min(shape) // 12), 1, -1
                )
        if prompt.mask is not None:
            mask = rough
        else:
            mask = grabcut_refine(
                frame,
                rough,
                foreground_points=positives,
                background_points=negatives,
                iterations=max(self.grabcut_iterations, 3),
                work_long_side=self.work_long_side,
            )
        handle.seeds.setdefault(obj_id, {})[frame_idx] = mask
        return mask

    # --------------------------------------------------------------- propagation

    def propagate(
        self, handle: _Handle, *, start_frame: int, reverse: bool, obj_ids: Sequence[int]
    ) -> Iterator[tuple[int, Masks]]:
        step = -1 if reverse else 1
        index = start_frame
        current: Masks = {}
        previous_gray: np.ndarray | None = None
        scale = 1.0

        while 0 <= index < handle.num_frames:
            frame = self._read(handle, index)
            if previous_gray is None:
                scale = min(1.0, self.work_long_side / max(frame.shape[:2]))
            small = self._small(frame, scale)
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

            flow = None
            if previous_gray is not None and current:
                # Flow from the new frame back to the previous one lets us pull the old
                # mask forward with a single remap.
                flow = cv2.calcOpticalFlowFarneback(
                    gray, previous_gray, None, 0.5, 3, 21, 3, 5, 1.1, 0
                )

            updated: Masks = {}
            for obj_id in obj_ids:
                seed = handle.seeds.get(obj_id, {}).get(index)
                if seed is not None and (not reverse or index == start_frame):
                    updated[obj_id] = seed
                elif obj_id in current and flow is not None:
                    updated[obj_id] = self._follow(small, current[obj_id], flow, frame.shape[:2])
                elif obj_id in current:
                    updated[obj_id] = current[obj_id]
            current = updated
            previous_gray = gray
            if current:
                yield index, dict(current)
            index += step

    def _small(self, frame: np.ndarray, scale: float) -> np.ndarray:
        if scale >= 1:
            return frame
        size = (round(frame.shape[1] * scale), round(frame.shape[0] * scale))
        return cv2.resize(frame, size, interpolation=cv2.INTER_AREA)

    def _follow(
        self, small: np.ndarray, previous: np.ndarray, flow: np.ndarray, full_shape: tuple[int, int]
    ) -> np.ndarray:
        """Warp the previous mask onto this frame, then snap it to colour edges."""
        sh, sw = small.shape[:2]
        prev_small = cv2.resize(
            previous.astype(np.uint8), (sw, sh), interpolation=cv2.INTER_NEAREST
        )
        grid_x, grid_y = np.meshgrid(
            np.arange(sw, dtype=np.float32), np.arange(sh, dtype=np.float32)
        )
        warped = cv2.remap(
            prev_small.astype(np.float32),
            grid_x + flow[..., 0],
            grid_y + flow[..., 1],
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        warped_mask = warped > 0.5
        if warped_mask.sum() < 16:
            return np.zeros(full_shape, bool)

        refined = self._snap(small, warped_mask)
        if refined.sum() < 0.3 * warped_mask.sum():  # GrabCut drifted; trust the flow
            refined = warped_mask
        full = cv2.resize(
            refined.astype(np.uint8),
            (full_shape[1], full_shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
        return clean_mask(full.astype(bool))

    def _snap(self, small: np.ndarray, warped: np.ndarray) -> np.ndarray:
        radius = max(2, round(0.02 * max(small.shape[:2])))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
        binary = warped.astype(np.uint8)
        core = cv2.erode(binary, kernel).astype(bool)
        outer = cv2.dilate(binary, kernel).astype(bool)
        labels = np.full(binary.shape, cv2.GC_BGD, np.uint8)
        labels[outer] = cv2.GC_PR_BGD
        labels[warped] = cv2.GC_PR_FGD
        labels[core] = cv2.GC_FGD
        if not core.any():
            return warped
        try:
            bgd = np.zeros((1, 65), np.float64)
            fgd = np.zeros((1, 65), np.float64)
            cv2.grabCut(
                small, labels, None, bgd, fgd, self.grabcut_iterations, cv2.GC_INIT_WITH_MASK
            )
        except cv2.error:
            return warped
        return (labels == cv2.GC_FGD) | (labels == cv2.GC_PR_FGD)


def create(settings: Any = None) -> ClassicalBackend:
    return ClassicalBackend()

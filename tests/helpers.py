"""Shared test helpers (imported by conftest and the test modules)."""

from __future__ import annotations

import math

import numpy as np

SIZE = (320, 180)  # width, height of the test clip
FRAMES = 40


def circle_polygon_px(
    mask: np.ndarray, grow: float = 1.35, points: int = 36
) -> list[tuple[float, float]]:
    """A loose circle around a mask, in pixel coordinates (like a rough hand-drawn outline)."""
    ys, xs = np.nonzero(mask)
    cx, cy = xs.mean(), ys.mean()
    radius = math.sqrt(mask.sum() / math.pi) * grow
    return [
        (
            cx + radius * math.cos(2 * math.pi * i / points),
            cy + radius * math.sin(2 * math.pi * i / points),
        )
        for i in range(points)
    ]


def circle_polygon_unit(
    mask: np.ndarray, grow: float = 1.35, points: int = 36
) -> list[tuple[float, float]]:
    """Same, normalised to 0-1 as the API expects."""
    width, height = SIZE
    return [(x / width, y / height) for x, y in circle_polygon_px(mask, grow, points)]


def iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b).sum()
    return 1.0 if union == 0 else float(np.logical_and(a, b).sum() / union)

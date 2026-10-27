"""A synthetic clip with exact ground truth.

Two textured objects cross paths over a panning background, so one of them is partly
occluded mid-clip. It needs no footage, no licences and no network, which makes it the
fixture for tests, the "Use a sample clip" button and the CPU benchmark.
"""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np

from .video import VideoWriter

BALL_ID = 1
SQUARE_ID = 2


def _background(width: int, height: int, pan: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    wide = width + pan
    gradient = np.linspace(0, 1, wide, dtype=np.float32)[None, :, None]
    base = (1 - gradient) * np.array([150, 122, 98], np.float32) + gradient * np.array(
        [112, 96, 84], np.float32
    )
    canvas = np.repeat(base, height, axis=0)
    coarse = rng.normal(0, 14, (height // 24 + 2, wide // 24 + 2, 1)).astype(np.float32)
    coarse = cv2.resize(coarse, (wide, height), interpolation=cv2.INTER_CUBIC)[..., None]
    fine = cv2.GaussianBlur(rng.normal(0, 9, (height, wide)).astype(np.float32), (0, 0), 1.2)
    return np.clip(canvas + coarse + fine[..., None], 0, 255).astype(np.uint8)


def _stripes(
    layer: np.ndarray, mask: np.ndarray, phase: float, color: tuple[int, int, int]
) -> None:
    height, width = mask.shape
    for k in range(-height, width + height, 14):
        x0 = k + int(phase)
        cv2.line(layer, (x0, 0), (x0 - height, height), color, 4, cv2.LINE_8)
    del width


def sample_frames(
    n_frames: int = 72, size: tuple[int, int] = (640, 360), seed: int = 7
) -> tuple[list[np.ndarray], list[dict[int, np.ndarray]]]:
    """Return BGR frames and, per frame, ``{object_id: visible_mask}`` ground truth."""
    width, height = size
    pan = 24
    background = _background(width, height, pan, seed)
    frames: list[np.ndarray] = []
    truths: list[dict[int, np.ndarray]] = []
    radius = round(0.12 * height)
    half = round(0.13 * height)

    for i in range(n_frames):
        t = i / max(n_frames - 1, 1)
        shift = round(t * pan)
        frame = background[:, shift : shift + width].copy()

        # Ball: left to right on a gentle wave.
        cx = round(width * (0.14 + 0.72 * t))
        cy = round(height * (0.40 + 0.08 * math.sin(2 * math.pi * 1.5 * t)))
        ball = np.zeros((height, width), np.uint8)
        cv2.circle(ball, (cx, cy), radius, 1, -1)
        layer = frame.copy()
        cv2.circle(layer, (cx, cy), radius, (40, 140, 240), -1)
        _stripes(layer, ball, 6 * i, (235, 235, 245))
        ball_mask = ball.astype(bool)
        frame[ball_mask] = layer[ball_mask]

        # Square: right to left, spinning. It is drawn on top, so it can hide the ball.
        sx = round(width * (0.86 - 0.72 * t))
        sy = round(height * (0.58 + 0.06 * math.sin(2 * math.pi * 1.0 * t + 1.0)))
        angle = math.radians(80 * t)
        corners = np.array([[-half, -half], [half, -half], [half, half], [-half, half]], np.float32)
        rotation = np.array(
            [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]], np.float32
        )
        polygon = np.round(corners @ rotation.T + [sx, sy]).astype(np.int32)
        square = np.zeros((height, width), np.uint8)
        cv2.fillPoly(square, [polygon], 1)
        layer = frame.copy()
        cv2.fillPoly(layer, [polygon], (70, 150, 60))
        _stripes(layer, square, -5 * i, (30, 90, 30))
        square_mask = square.astype(bool)
        frame[square_mask] = layer[square_mask]

        frames.append(frame)
        truths.append({BALL_ID: ball_mask & ~square_mask, SQUARE_ID: square_mask})
    return frames, truths


def write_sample_video(
    path: Path, n_frames: int = 72, size: tuple[int, int] = (640, 360), fps: float = 30.0
) -> None:
    frames, _ = sample_frames(n_frames, size)
    with VideoWriter(path, size[0], size[1], fps) as writer:
        for frame in frames:
            writer.write(frame)

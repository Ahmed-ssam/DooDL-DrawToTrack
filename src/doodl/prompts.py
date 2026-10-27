"""Turn what the user drew (a rough outline, a few clicks) into model prompts.

SAM 2 takes points, boxes or masks. A hand-drawn outline is none of those, so this
module converts it. The strategy is configurable because it is the main quality lever
on the first frame (see docs/evaluation.md for the ablation protocol).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum

import cv2
import numpy as np

from .errors import PromptError

Point = tuple[float, float]
Box = tuple[float, float, float, float]

MIN_AREA_FRACTION = 0.0005


class PromptStrategy(str, Enum):
    BOX_POINTS = "box_points"  # box from the outline + clicks inside and just outside it
    MASK = "mask"  # feed the rasterised outline to the model as-is
    GRABCUT = "grabcut"  # refine the outline with GrabCut, then box_points


@dataclass
class ObjectPrompt:
    """Everything the user said about one object on one frame (pixel coordinates)."""

    polygon: list[Point] | None = None
    box: Box | None = None
    points: list[Point] = field(default_factory=list)
    labels: list[int] = field(default_factory=list)
    mask: np.ndarray | None = None

    def __post_init__(self) -> None:
        if len(self.points) != len(self.labels):
            raise PromptError("Every click needs a label (add or remove).")
        if any(label not in (0, 1) for label in self.labels):
            raise PromptError("Click labels must be 1 (add) or 0 (remove).")


@dataclass
class SamPrompt:
    """Prompt in the shape SAM 2's video predictor expects."""

    points: np.ndarray  # (N, 2) float32, pixel xy
    labels: np.ndarray  # (N,) int32
    box: np.ndarray | None = None  # (4,) float32 x0, y0, x1, y1
    mask: np.ndarray | None = None  # (H, W) bool


# --------------------------------------------------------------------------- masks


def polygon_to_mask(polygon: list[Point], shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    pts = np.asarray(polygon, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 3:
        raise PromptError(
            "An outline needs at least three points. Draw all the way around the object."
        )
    pts[:, 0] = np.clip(pts[:, 0], 0, width - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, height - 1)
    canvas = np.zeros((height, width), np.uint8)
    cv2.fillPoly(canvas, [np.round(pts).astype(np.int32)], 1)
    if canvas.sum() < MIN_AREA_FRACTION * height * width:
        raise PromptError("That outline is too small. Draw it around the whole object.")
    return canvas.astype(bool)


def box_to_mask(box: Box, shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    x0, y0, x1, y1 = box
    mask = np.zeros((height, width), bool)
    xa, xb = sorted((round(min(max(x0, 0), width)), round(min(max(x1, 0), width))))
    ya, yb = sorted((round(min(max(y0, 0), height)), round(min(max(y1, 0), height))))
    mask[ya:yb, xa:xb] = True
    if mask.sum() < MIN_AREA_FRACTION * height * width:
        raise PromptError("That box is too small. Draw it around the whole object.")
    return mask


def mask_to_box(mask: np.ndarray) -> Box | None:
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    return float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(a, b).sum() / union)


def clean_mask(mask: np.ndarray, *, min_component_fraction: float = 0.02) -> np.ndarray:
    """Close pin-holes, drop specks and fill small interior holes."""
    binary = mask.astype(np.uint8)
    if not binary.any():
        return mask.astype(bool)
    height, width = binary.shape
    radius = max(1, round(0.004 * max(height, width)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if count > 2:
        areas = stats[1:, cv2.CC_STAT_AREA]
        threshold = max(4, min_component_fraction * areas.max())
        keep = 1 + np.nonzero(areas >= threshold)[0]
        binary = np.isin(labels, keep).astype(np.uint8)

    # Fill background components that do not touch the border and are small.
    inverse = (1 - binary).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(inverse, connectivity=4)
    max_hole = max(64, 0.005 * float(binary.sum()))
    border = set(np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]])))
    for index in range(1, count):
        if index not in border and stats[index, cv2.CC_STAT_AREA] <= max_hole:
            binary[labels == index] = 1
    return binary.astype(bool)


# ------------------------------------------------------------------ point sampling


def sample_positive_points(mask: np.ndarray, count: int) -> list[Point]:
    """Well-spread points deep inside the mask (distance-transform core, farthest-point)."""
    if count <= 0 or not mask.any():
        return []
    dist = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    peak = float(dist.max())
    if peak <= 0:
        return []
    ys, xs = np.nonzero(dist >= 0.35 * peak)
    if xs.size > 4000:
        pick = np.random.default_rng(0).choice(xs.size, 4000, replace=False)
        xs, ys = xs[pick], ys[pick]
    candidates = np.stack([xs, ys], axis=1).astype(np.float32)
    start = np.unravel_index(int(np.argmax(dist)), dist.shape)
    chosen = [np.array([start[1], start[0]], dtype=np.float32)]
    nearest = np.linalg.norm(candidates - chosen[0], axis=1)
    while len(chosen) < count and nearest.max() > 1.0:
        nxt = candidates[int(np.argmax(nearest))]
        chosen.append(nxt)
        nearest = np.minimum(nearest, np.linalg.norm(candidates - nxt, axis=1))
    return [(float(x), float(y)) for x, y in chosen]


def sample_negative_points(mask: np.ndarray, count: int) -> list[Point]:
    """Points just outside the outline, spread around it by angle."""
    box = mask_to_box(mask)
    if count <= 0 or box is None:
        return []
    x0, y0, x1, y1 = box
    diagonal = math.hypot(x1 - x0, y1 - y0)
    outer = max(6, round(0.05 * diagonal))
    inner = max(2, outer // 2)

    def grow(radius: int) -> np.ndarray:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
        return cv2.dilate(mask.astype(np.uint8), kernel).astype(bool)

    ring = grow(outer) & ~grow(inner)
    ys, xs = np.nonzero(ring)
    if xs.size == 0:
        return []
    cy, cx = (y0 + y1) / 2, (x0 + x1) / 2
    angles = np.arctan2(ys - cy, xs - cx)
    points: list[Point] = []
    for k in range(count):
        target = -math.pi + 2 * math.pi * (k + 0.5) / count
        delta = np.abs((angles - target + math.pi) % (2 * math.pi) - math.pi)
        best = np.flatnonzero(delta <= delta.min() + 1e-6)
        index = best[int(np.argmax(np.hypot(xs[best] - cx, ys[best] - cy)))]
        points.append((float(xs[index]), float(ys[index])))
    return points


# ----------------------------------------------------------------------- GrabCut


def grabcut_refine(
    frame_bgr: np.ndarray,
    rough: np.ndarray,
    *,
    foreground_points: list[Point] | None = None,
    background_points: list[Point] | None = None,
    iterations: int = 3,
    work_long_side: int = 480,
) -> np.ndarray:
    """Snap a loose outline to the object's colour edges.

    Inside the outline is "probably object", its deep core is "certainly object", a thin
    band outside is "probably background" and everything else is "certainly background".
    """
    height, width = rough.shape
    scale = min(1.0, work_long_side / max(height, width))
    size = (max(8, round(width * scale)), max(8, round(height * scale)))
    small = cv2.resize(frame_bgr, size, interpolation=cv2.INTER_AREA) if scale < 1 else frame_bgr
    seed = (
        cv2.resize(rough.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST)
        if scale < 1
        else rough.astype(np.uint8)
    )
    if not seed.any():
        raise PromptError("That outline is too small. Draw it around the whole object.")

    labels = np.full(seed.shape, cv2.GC_BGD, np.uint8)
    inside = seed.astype(bool)
    dist = cv2.distanceTransform(seed, cv2.DIST_L2, 5)
    radius = max(2, round(0.03 * max(seed.shape)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    band = cv2.dilate(seed, kernel).astype(bool) & ~inside
    labels[band] = cv2.GC_PR_BGD
    labels[inside] = cv2.GC_PR_FGD
    labels[dist >= 0.5 * dist.max()] = cv2.GC_FGD

    stroke = max(2, round(0.012 * max(seed.shape)))
    for pts, value in ((foreground_points, cv2.GC_FGD), (background_points, cv2.GC_BGD)):
        for x, y in pts or []:
            cv2.circle(labels, (round(x * scale), round(y * scale)), stroke, int(value), -1)

    if not (labels == cv2.GC_BGD).any() or not (labels == cv2.GC_FGD).any():
        return rough.copy()
    try:
        bgd = np.zeros((1, 65), np.float64)
        fgd = np.zeros((1, 65), np.float64)
        cv2.grabCut(small, labels, None, bgd, fgd, iterations, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return rough.copy()

    refined = ((labels == cv2.GC_FGD) | (labels == cv2.GC_PR_FGD)).astype(np.uint8)
    if scale < 1:
        refined = cv2.resize(
            refined.astype(np.float32), (width, height), interpolation=cv2.INTER_LINEAR
        )
        refined = (refined > 0.5).astype(np.uint8)
    result = clean_mask(refined.astype(bool))
    # If GrabCut collapsed onto a sliver, the outline is the better answer.
    if result.sum() < 0.05 * rough.sum():
        return rough.copy()
    return result


# ------------------------------------------------------------------ SAM 2 prompts


def rough_mask(prompt: ObjectPrompt, shape: tuple[int, int]) -> np.ndarray | None:
    if prompt.mask is not None:
        return prompt.mask.astype(bool)
    if prompt.polygon is not None:
        return polygon_to_mask(prompt.polygon, shape)
    if prompt.box is not None:
        return box_to_mask(prompt.box, shape)
    return None


def build_sam_prompt(
    prompt: ObjectPrompt,
    frame_bgr: np.ndarray,
    strategy: PromptStrategy,
    *,
    positive_points: int = 4,
    negative_points: int = 4,
) -> SamPrompt:
    shape = frame_bgr.shape[:2]
    rough = rough_mask(prompt, shape)
    clicks = list(prompt.points)
    click_labels = list(prompt.labels)
    if rough is None and not clicks:
        raise PromptError("Draw an outline around the object, or click on it.")

    if rough is not None and strategy is PromptStrategy.MASK and not clicks:
        return SamPrompt(
            points=np.zeros((0, 2), np.float32), labels=np.zeros((0,), np.int32), mask=rough
        )

    box: np.ndarray | None = None
    points: list[Point] = []
    labels: list[int] = []
    if rough is not None:
        if strategy is PromptStrategy.GRABCUT and prompt.mask is None:
            rough = grabcut_refine(
                frame_bgr,
                rough,
                foreground_points=[
                    p for p, lab in zip(clicks, click_labels, strict=True) if lab == 1
                ],
                background_points=[
                    p for p, lab in zip(clicks, click_labels, strict=True) if lab == 0
                ],
            )
        found = mask_to_box(rough)
        if found is not None:
            box = np.asarray(found, dtype=np.float32)
        for p in sample_positive_points(rough, positive_points):
            points.append(p)
            labels.append(1)
        for p in sample_negative_points(rough, negative_points):
            points.append(p)
            labels.append(0)
    points.extend(clicks)
    labels.extend(click_labels)

    return SamPrompt(
        points=np.asarray(points, dtype=np.float32).reshape(-1, 2),
        labels=np.asarray(labels, dtype=np.int32),
        box=box,
    )

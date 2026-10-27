"""Segmentation quality metrics in the DAVIS style: J (region), F (boundary), J&F."""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np


def region_similarity(pred: np.ndarray, truth: np.ndarray) -> float:
    """Jaccard index (intersection over union). Two empty masks count as a perfect match."""
    pred, truth = pred.astype(bool), truth.astype(bool)
    union = np.logical_or(pred, truth).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(pred, truth).sum() / union)


def _boundary(mask: np.ndarray) -> np.ndarray:
    binary = mask.astype(np.uint8)
    eroded = cv2.erode(
        binary, np.ones((3, 3), np.uint8), borderType=cv2.BORDER_CONSTANT, borderValue=0
    )
    return (binary - eroded).astype(bool)


def boundary_f_measure(pred: np.ndarray, truth: np.ndarray, tolerance: float = 0.008) -> float:
    """F-measure of boundary pixels within ``tolerance`` x image diagonal of each other."""
    pred, truth = pred.astype(bool), truth.astype(bool)
    if not pred.any() and not truth.any():
        return 1.0
    if not pred.any() or not truth.any():
        return 0.0
    diagonal = float(np.hypot(*pred.shape))
    radius = max(1, round(tolerance * diagonal))
    pred_edge, truth_edge = _boundary(pred), _boundary(truth)
    near_truth = cv2.distanceTransform((~truth_edge).astype(np.uint8), cv2.DIST_L2, 3) <= radius
    near_pred = cv2.distanceTransform((~pred_edge).astype(np.uint8), cv2.DIST_L2, 3) <= radius
    precision = float((pred_edge & near_truth).sum() / max(pred_edge.sum(), 1))
    recall = float((truth_edge & near_pred).sum() / max(truth_edge.sum(), 1))
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def j_and_f(
    preds: Sequence[np.ndarray], truths: Sequence[np.ndarray], skip_first: bool = True
) -> dict[str, float]:
    """Mean J, mean F and their average over a sequence.

    The first frame is skipped by default because it is the prompt frame (standard
    semi-supervised VOS protocol).
    """
    if len(preds) != len(truths):
        raise ValueError("Predictions and ground truth must have the same number of frames")
    start = 1 if skip_first and len(preds) > 1 else 0
    j_values = [region_similarity(p, t) for p, t in zip(preds[start:], truths[start:], strict=True)]
    f_values = [
        boundary_f_measure(p, t) for p, t in zip(preds[start:], truths[start:], strict=True)
    ]
    j_mean, f_mean = float(np.mean(j_values)), float(np.mean(f_values))
    return {"J": j_mean, "F": f_mean, "J&F": (j_mean + f_mean) / 2}

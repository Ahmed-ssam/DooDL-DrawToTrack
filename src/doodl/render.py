"""Colours, overlays and label maps."""

from __future__ import annotations

import cv2
import numpy as np

# Eight hues picked to stay apart on both dark and light footage. The outline the
# user draws is always red-coral, so no object colour sits near it.
PALETTE: tuple[str, ...] = (
    "#8EE040",  # lime (the logo green)
    "#F5A623",  # amber
    "#8B7BFF",  # violet
    "#18C6C0",  # teal
    "#4DA8FF",  # sky
    "#E86BD0",  # magenta
    "#FF6B5E",  # coral
    "#B8C0D8",  # silver
)

FILL_ALPHA = 0.42


def hex_to_bgr(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    r, g, b = (int(value[i : i + 2], 16) for i in (0, 2, 4))
    return b, g, r


def color_hex(obj_id: int) -> str:
    return PALETTE[(obj_id - 1) % len(PALETTE)]


def _contour_thickness(shape: tuple[int, int]) -> int:
    return max(2, round(min(shape) / 300))


def overlay_masks(
    frame_bgr: np.ndarray,
    masks: dict[int, np.ndarray],
    *,
    alpha: float = FILL_ALPHA,
    contours: bool = True,
) -> np.ndarray:
    """Tint each mask and trace its outline in the object's colour."""
    out = frame_bgr.astype(np.float32)
    thickness = _contour_thickness(frame_bgr.shape[:2])
    for obj_id in sorted(masks):
        mask = masks[obj_id]
        if not mask.any():
            continue
        color = np.array(hex_to_bgr(color_hex(obj_id)), np.float32)
        out[mask] = out[mask] * (1 - alpha) + color * alpha
    result = np.clip(out, 0, 255).astype(np.uint8)
    if contours:
        for obj_id in sorted(masks):
            if masks[obj_id].any():
                found, _ = cv2.findContours(
                    masks[obj_id].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
                )
                cv2.drawContours(
                    result, found, -1, hex_to_bgr(color_hex(obj_id)), thickness, cv2.LINE_AA
                )
    return result


def isolate_masks(frame_bgr: np.ndarray, masks: dict[int, np.ndarray]) -> np.ndarray:
    """Keep only the tracked objects; everything else goes black."""
    keep = np.zeros(frame_bgr.shape[:2], bool)
    for mask in masks.values():
        keep |= mask
    out = np.zeros_like(frame_bgr)
    out[keep] = frame_bgr[keep]
    return out


def masks_to_overlay_png(masks: dict[int, np.ndarray], shape: tuple[int, int]) -> bytes:
    """Transparent RGBA PNG of the tinted masks, drawn over the frame by the browser."""
    height, width = shape
    rgba = np.zeros((height, width, 4), np.uint8)
    thickness = _contour_thickness(shape)
    for obj_id in sorted(masks):
        mask = masks[obj_id]
        if not mask.any():
            continue
        b, g, r = hex_to_bgr(color_hex(obj_id))
        rgba[mask] = (b, g, r, round(255 * FILL_ALPHA))
        found, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(rgba, found, -1, (b, g, r, 255), thickness, cv2.LINE_AA)
    ok, buffer = cv2.imencode(".png", rgba)
    if not ok:
        raise RuntimeError("PNG encoding failed")
    return bytes(buffer.tobytes())


def label_map_from_masks(masks: dict[int, np.ndarray], shape: tuple[int, int]) -> np.ndarray:
    """Pack object masks into one uint8 image (0 = background, otherwise the object id).

    Where two objects overlap, the higher id wins. The label map is what gets stored and
    exported, so the exports always agree with what the overlay shows.
    """
    label_map = np.zeros(shape, np.uint8)
    for obj_id in sorted(masks):
        label_map[masks[obj_id]] = obj_id
    return label_map


def masks_from_label_map(label_map: np.ndarray, obj_ids: list[int]) -> dict[int, np.ndarray]:
    return {obj_id: label_map == obj_id for obj_id in obj_ids}

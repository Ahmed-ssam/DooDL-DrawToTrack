"""Mask storage and exports: indexed PNG label maps, COCO-style RLE, zip."""

from __future__ import annotations

import json
import zipfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from . import __version__


def rle_encode(mask: np.ndarray) -> dict[str, Any]:
    """COCO uncompressed RLE: column-major run lengths, starting with a run of zeros."""
    height, width = mask.shape
    flat = np.asarray(mask, dtype=bool).ravel(order="F")
    if flat.size == 0:
        return {"size": [height, width], "counts": []}
    change = np.flatnonzero(flat[1:] != flat[:-1]) + 1
    bounds = np.concatenate(([0], change, [flat.size]))
    counts = np.diff(bounds).tolist()
    if flat[0]:
        counts = [0, *counts]
    return {"size": [height, width], "counts": counts}


def rle_decode(rle: dict[str, Any]) -> np.ndarray:
    height, width = rle["size"]
    flat = np.zeros(height * width, bool)
    position, value = 0, False
    for run in rle["counts"]:
        if value:
            flat[position : position + run] = True
        position += run
        value = not value
    return flat.reshape((height, width), order="F")


def bbox_xywh(mask: np.ndarray) -> list[int]:
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return [0, 0, 0, 0]
    return [
        int(xs.min()),
        int(ys.min()),
        int(xs.max() - xs.min() + 1),
        int(ys.max() - ys.min() + 1),
    ]


class MaskStore:
    """One indexed PNG per frame (pixel value = object id), kept on disk.

    Storing label maps instead of per-object arrays keeps a 450-frame 720p track under
    a few MB of disk and a flat memory profile.
    """

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        directory.mkdir(parents=True, exist_ok=True)

    def path(self, frame_idx: int) -> Path:
        return self.directory / f"{frame_idx:05d}.png"

    def clear(self) -> None:
        for png in self.directory.glob("*.png"):
            png.unlink()

    def has(self, frame_idx: int) -> bool:
        return self.path(frame_idx).exists()

    def read(self, frame_idx: int) -> np.ndarray | None:
        path = self.path(frame_idx)
        if not path.exists():
            return None
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None:
            return None
        return image if image.ndim == 2 else image[..., 0]

    def update(self, frame_idx: int, masks: dict[int, np.ndarray], shape: tuple[int, int]) -> None:
        """Set the given objects on this frame, leaving other objects untouched."""
        label_map = self.read(frame_idx)
        if label_map is None or label_map.shape != shape:
            label_map = np.zeros(shape, np.uint8)
        for obj_id in masks:
            label_map[label_map == obj_id] = 0
        for obj_id in sorted(masks):
            label_map[masks[obj_id]] = obj_id
        cv2.imwrite(str(self.path(frame_idx)), label_map, [cv2.IMWRITE_PNG_COMPRESSION, 3])

    def count(self) -> int:
        return sum(1 for _ in self.directory.glob("*.png"))


def frame_annotation(
    frame_idx: int, fps: float, label_map: np.ndarray, objects: Iterable[int]
) -> dict[str, Any]:
    entries = []
    for obj_id in objects:
        mask = label_map == obj_id
        area = int(mask.sum())
        if area == 0:
            continue
        entries.append(
            {
                "id": obj_id,
                "bbox": bbox_xywh(mask),
                "area": area,
                "segmentation": rle_encode(mask),
            }
        )
    return {"index": frame_idx, "time_s": round(frame_idx / fps, 4), "objects": entries}


def write_annotations(
    path: Path,
    *,
    width: int,
    height: int,
    fps: float,
    frame_count: int,
    objects: list[dict[str, Any]],
    frames: list[dict[str, Any]],
) -> None:
    document = {
        "doodl_version": __version__,
        "format": "COCO-style uncompressed RLE (column-major, starts with background)",
        "video": {"width": width, "height": height, "fps": fps, "frames": frame_count},
        "objects": objects,
        "frames": frames,
    }
    path.write_text(json.dumps(document, separators=(",", ":")), encoding="utf-8")


def zip_masks(store: MaskStore, destination: Path, frame_count: int) -> None:
    """Zip the label-map PNGs (already compressed, so stored as-is)."""
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_STORED) as archive:
        for index in range(frame_count):
            png = store.path(index)
            if png.exists():
                archive.write(png, arcname=f"masks/{index:05d}.png")
        archive.writestr(
            "README.txt",
            "Each PNG is an 8-bit label map: pixel value 0 is background, value N is object N.\n",
        )

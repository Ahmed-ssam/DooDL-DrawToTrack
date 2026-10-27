from __future__ import annotations

import json
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from doodl.export import (
    MaskStore,
    bbox_xywh,
    frame_annotation,
    rle_decode,
    rle_encode,
    write_annotations,
    zip_masks,
)
from doodl.metrics import boundary_f_measure, j_and_f, region_similarity
from doodl.render import (
    PALETTE,
    color_hex,
    hex_to_bgr,
    isolate_masks,
    label_map_from_masks,
    masks_from_label_map,
    masks_to_overlay_png,
    overlay_masks,
)

SHAPE = (90, 160)


def box(y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
    mask = np.zeros(SHAPE, bool)
    mask[y0:y1, x0:x1] = True
    return mask


# ----------------------------------------------------------------------- render


def test_palette_has_eight_distinct_valid_colours() -> None:
    assert len(PALETTE) == len(set(PALETTE)) == 8
    assert hex_to_bgr("#8EE040") == (0x40, 0xE0, 0x8E)
    assert color_hex(1) == PALETTE[0]
    assert color_hex(9) == PALETTE[0]  # wraps instead of failing


def test_overlay_only_changes_masked_pixels_without_contours() -> None:
    frame = np.full((*SHAPE, 3), 100, np.uint8)
    mask = box(20, 60, 30, 100)
    out = overlay_masks(frame, {1: mask}, contours=False)
    assert (out[~mask] == 100).all()
    assert not (out[mask] == 100).all(axis=-1).any()
    assert frame.max() == 100  # input untouched


def test_overlay_draws_contour_in_the_object_colour() -> None:
    frame = np.full((*SHAPE, 3), 100, np.uint8)
    out = overlay_masks(frame, {2: box(20, 60, 30, 100)})
    target = np.array(hex_to_bgr(color_hex(2)))
    assert (np.abs(out.astype(int) - target).sum(axis=-1) < 6).any()


def test_isolate_keeps_only_objects() -> None:
    frame = np.full((*SHAPE, 3), 200, np.uint8)
    mask = box(10, 40, 10, 40)
    out = isolate_masks(frame, {1: mask})
    assert (out[mask] == 200).all() and (out[~mask] == 0).all()


def test_label_map_roundtrip_and_overlap_rule() -> None:
    a, b = box(10, 50, 10, 60), box(30, 70, 40, 90)
    label_map = label_map_from_masks({1: a, 2: b}, SHAPE)
    assert label_map.dtype == np.uint8
    assert (label_map[a & b] == 2).all()  # higher id wins
    back = masks_from_label_map(label_map, [1, 2])
    assert (back[2] == b).all()
    assert (back[1] == (a & ~b)).all()


def test_overlay_png_is_transparent_outside_masks() -> None:
    png = masks_to_overlay_png({1: box(20, 60, 30, 100)}, SHAPE)
    image = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_UNCHANGED)
    assert image.shape == (*SHAPE, 4)
    assert image[5, 5, 3] == 0
    assert image[40, 60, 3] > 0


# ----------------------------------------------------------------------- export


def test_rle_matches_coco_column_major_example() -> None:
    mask = np.array([[1, 0], [1, 1]], bool)  # column-major flatten: 1,1,0,1
    assert rle_encode(mask)["counts"] == [0, 2, 1, 1]
    assert rle_encode(np.zeros((2, 3), bool))["counts"] == [6]
    assert rle_encode(np.ones((2, 3), bool))["counts"] == [0, 6]


@pytest.mark.parametrize("shape", [(1, 1), (7, 13), (64, 48)])
def test_rle_roundtrip_random_masks(shape: tuple[int, int]) -> None:
    rng = np.random.default_rng(shape[0] * 31 + shape[1])
    for density in (0.0, 0.1, 0.5, 0.9, 1.0):
        mask = rng.random(shape) < density
        assert (rle_decode(rle_encode(mask)) == mask).all()


def test_bbox_xywh() -> None:
    assert bbox_xywh(box(10, 20, 30, 50)) == [30, 10, 20, 10]
    assert bbox_xywh(np.zeros(SHAPE, bool)) == [0, 0, 0, 0]


def test_mask_store_merges_objects_and_clears(tmp_path: Path) -> None:
    store = MaskStore(tmp_path / "masks")
    assert store.read(0) is None and not store.has(0)
    a, b = box(10, 30, 10, 30), box(40, 60, 50, 70)
    store.update(0, {1: a}, SHAPE)
    store.update(0, {2: b}, SHAPE)
    label_map = store.read(0)
    assert label_map is not None
    assert (label_map[a] == 1).all() and (label_map[b] == 2).all()
    moved = box(12, 32, 12, 32)
    store.update(0, {1: moved}, SHAPE)  # replaces object 1 only
    label_map = store.read(0)
    assert label_map is not None
    assert (label_map == 1).sum() == moved.sum() and (label_map == 2).sum() == b.sum()
    assert store.count() == 1
    store.clear()
    assert store.count() == 0


def test_annotations_and_zip(tmp_path: Path) -> None:
    store = MaskStore(tmp_path / "masks")
    a = box(10, 30, 10, 30)
    for i in range(3):
        store.update(i, {1: a}, SHAPE)
    label_map = store.read(0)
    assert label_map is not None
    record = frame_annotation(0, 30.0, label_map, [1, 2])
    assert [o["id"] for o in record["objects"]] == [1]  # object 2 is absent, so skipped
    assert rle_decode(record["objects"][0]["segmentation"]).sum() == a.sum()

    out = tmp_path / "annotations.json"
    write_annotations(
        out,
        width=160,
        height=90,
        fps=30.0,
        frame_count=3,
        objects=[{"id": 1, "name": "Object 1", "color": "#8EE040"}],
        frames=[record],
    )
    document = json.loads(out.read_text())
    assert (
        document["video"]["frames"] == 3 and document["frames"][0]["objects"][0]["area"] == a.sum()
    )

    archive = tmp_path / "masks.zip"
    zip_masks(store, archive, 3)
    with zipfile.ZipFile(archive) as z:
        names = z.namelist()
    assert "README.txt" in names and sum(n.endswith(".png") for n in names) == 3


# ---------------------------------------------------------------------- metrics


def test_region_similarity() -> None:
    a = box(10, 50, 10, 50)
    assert region_similarity(a, a) == 1.0
    assert region_similarity(a, box(60, 80, 100, 140)) == 0.0
    assert region_similarity(np.zeros(SHAPE, bool), np.zeros(SHAPE, bool)) == 1.0
    half = box(10, 50, 10, 30)
    assert region_similarity(half, a) == pytest.approx(0.5)


def test_boundary_f_measure_tolerates_small_shifts_only() -> None:
    a = box(20, 60, 30, 100)
    assert boundary_f_measure(a, a) == 1.0
    assert boundary_f_measure(box(21, 61, 31, 101), a) > 0.9
    assert boundary_f_measure(box(20, 60, 90, 150), a) < 0.5
    assert boundary_f_measure(np.zeros(SHAPE, bool), a) == 0.0


def test_j_and_f_skips_the_prompt_frame() -> None:
    truth = [box(20, 60, 30, 100)] * 3
    preds = [box(0, 5, 0, 5), truth[1], truth[2]]  # frame 0 is wrong but is the prompt frame
    assert j_and_f(preds, truth)["J&F"] == pytest.approx(1.0)
    assert j_and_f(preds, truth, skip_first=False)["J&F"] < 1.0
    with pytest.raises(ValueError):
        j_and_f(preds, truth[:2])

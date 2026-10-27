from __future__ import annotations

import numpy as np
import pytest

from doodl.errors import PromptError
from doodl.prompts import (
    ObjectPrompt,
    PromptStrategy,
    box_to_mask,
    build_sam_prompt,
    clean_mask,
    grabcut_refine,
    mask_iou,
    mask_to_box,
    polygon_to_mask,
    sample_negative_points,
    sample_positive_points,
)
from doodl.samples import BALL_ID
from helpers import circle_polygon_px, iou

SHAPE = (180, 320)


def disk(cx: int, cy: int, r: int, shape: tuple[int, int] = SHAPE) -> np.ndarray:
    yy, xx = np.mgrid[: shape[0], : shape[1]]
    return (xx - cx) ** 2 + (yy - cy) ** 2 <= r**2


class TestMasks:
    def test_polygon_area_matches_geometry(self) -> None:
        mask = polygon_to_mask([(10, 10), (110, 10), (110, 60), (10, 60)], SHAPE)
        assert mask.dtype == bool
        assert abs(mask.sum() - 100 * 50) < 400

    def test_polygon_points_outside_frame_are_clamped(self) -> None:
        mask = polygon_to_mask([(-50, -50), (500, -50), (500, 300), (-50, 300)], SHAPE)
        assert mask.all()

    def test_polygon_needs_three_points(self) -> None:
        with pytest.raises(PromptError, match="three points"):
            polygon_to_mask([(1, 1), (5, 5)], SHAPE)

    def test_tiny_polygon_is_rejected(self) -> None:
        with pytest.raises(PromptError, match="too small"):
            polygon_to_mask([(10, 10), (12, 10), (12, 12), (10, 12)], SHAPE)

    def test_box_roundtrip(self) -> None:
        mask = box_to_mask((20, 30, 120, 90), SHAPE)
        assert mask_to_box(mask) == (20.0, 30.0, 120.0, 90.0)

    def test_empty_mask_has_no_box(self) -> None:
        assert mask_to_box(np.zeros(SHAPE, bool)) is None

    def test_mask_iou(self) -> None:
        a = disk(80, 80, 30)
        assert mask_iou(a, a) == 1.0
        assert mask_iou(a, disk(250, 80, 20)) == 0.0
        assert mask_iou(np.zeros(SHAPE, bool), np.zeros(SHAPE, bool)) == 1.0


class TestCleanMask:
    def test_removes_specks_and_fills_pinholes(self) -> None:
        mask = disk(100, 90, 40)
        mask[90, 100] = False  # pinhole
        mask[10, 10] = True  # speck far away
        cleaned = clean_mask(mask)
        assert cleaned[90, 100]
        assert not cleaned[10, 10]
        assert mask_iou(cleaned, disk(100, 90, 40)) > 0.97

    def test_keeps_large_interior_hole(self) -> None:
        mask = disk(100, 90, 60) & ~disk(100, 90, 25)
        cleaned = clean_mask(mask)
        assert not cleaned[90, 100]

    def test_empty_mask_stays_empty(self) -> None:
        assert not clean_mask(np.zeros(SHAPE, bool)).any()


class TestPointSampling:
    def test_positive_points_are_inside_and_spread_out(self) -> None:
        mask = disk(160, 90, 60)
        points = sample_positive_points(mask, 4)
        assert 2 <= len(points) <= 4
        for x, y in points:
            assert mask[int(y), int(x)]
        distances = [
            np.hypot(a[0] - b[0], a[1] - b[1])
            for i, a in enumerate(points)
            for b in points[i + 1 :]
        ]
        assert min(distances) > 10

    def test_positive_points_on_empty_mask(self) -> None:
        assert sample_positive_points(np.zeros(SHAPE, bool), 4) == []
        assert sample_positive_points(disk(50, 50, 20), 0) == []

    def test_negative_points_are_outside_and_surround_the_object(self) -> None:
        mask = disk(160, 90, 40)
        points = sample_negative_points(mask, 4)
        assert len(points) == 4
        for x, y in points:
            assert not mask[int(y), int(x)]
            assert 0 <= x < SHAPE[1] and 0 <= y < SHAPE[0]
        angles = sorted(np.arctan2(y - 90, x - 160) for x, y in points)
        gaps = np.diff(angles)
        assert gaps.max() < np.pi * 1.2  # no big uncovered side

    def test_negative_points_when_mask_fills_the_frame(self) -> None:
        assert sample_negative_points(np.ones(SHAPE, bool), 4) == []


class TestGrabCut:
    def test_snaps_a_loose_outline_to_the_object(self, truth: tuple[list, list]) -> None:
        frames, truths = truth
        ball = truths[0][BALL_ID]
        rough = polygon_to_mask(circle_polygon_px(ball, grow=1.5), SHAPE)
        refined = grabcut_refine(frames[0], rough)
        assert iou(refined, ball) > 0.85
        assert iou(refined, ball) > iou(rough, ball)

    def test_clicks_change_the_result(self, truth: tuple[list, list]) -> None:
        frames, truths = truth
        ball = truths[0][BALL_ID]
        rough = polygon_to_mask(circle_polygon_px(ball, grow=1.5), SHAPE)
        ys, xs = np.nonzero(rough & ~ball)
        background_click = (float(xs[len(xs) // 2]), float(ys[len(ys) // 2]))
        refined = grabcut_refine(frames[0], rough, background_points=[background_click])
        assert not refined[int(background_click[1]), int(background_click[0])]


class TestBuildSamPrompt:
    def frame(self) -> np.ndarray:
        return np.zeros((*SHAPE, 3), np.uint8)

    def prompt(self) -> ObjectPrompt:
        return ObjectPrompt(polygon=[(100, 50), (220, 50), (220, 130), (100, 130)])

    def test_box_points_strategy(self) -> None:
        sam = build_sam_prompt(self.prompt(), self.frame(), PromptStrategy.BOX_POINTS)
        assert sam.mask is None
        assert sam.box is not None
        assert sam.box.tolist() == pytest.approx([100, 50, 221, 131], abs=2)
        assert sam.points.shape == (len(sam.labels), 2)
        assert (sam.labels == 1).sum() >= 1 and (sam.labels == 0).sum() >= 1
        positives = sam.points[sam.labels == 1]
        assert ((positives[:, 0] > 100) & (positives[:, 0] < 220)).all()

    def test_mask_strategy_passes_the_rasterised_outline(self) -> None:
        sam = build_sam_prompt(self.prompt(), self.frame(), PromptStrategy.MASK)
        assert sam.mask is not None and sam.mask.sum() > 0
        assert len(sam.points) == 0 and sam.box is None

    def test_clicks_with_mask_strategy_fall_back_to_box_and_points(self) -> None:
        prompt = self.prompt()
        prompt.points, prompt.labels = [(150.0, 90.0)], [0]
        sam = build_sam_prompt(prompt, self.frame(), PromptStrategy.MASK)
        assert sam.mask is None and sam.box is not None
        assert sam.points[-1].tolist() == [150.0, 90.0]
        assert sam.labels[-1] == 0

    def test_clicks_only_prompt_has_no_box(self) -> None:
        prompt = ObjectPrompt(points=[(50.0, 50.0)], labels=[1])
        sam = build_sam_prompt(prompt, self.frame(), PromptStrategy.BOX_POINTS)
        assert sam.box is None and sam.points.shape == (1, 2)

    def test_nothing_to_go_on_is_an_error(self) -> None:
        with pytest.raises(PromptError, match="outline"):
            build_sam_prompt(ObjectPrompt(), self.frame(), PromptStrategy.BOX_POINTS)

    def test_grabcut_strategy_tightens_the_box(self, truth: tuple[list, list]) -> None:
        frames, truths = truth
        ball = truths[0][BALL_ID]
        prompt = ObjectPrompt(polygon=circle_polygon_px(ball, grow=1.6))
        loose = build_sam_prompt(prompt, frames[0], PromptStrategy.BOX_POINTS)
        tight = build_sam_prompt(prompt, frames[0], PromptStrategy.GRABCUT)
        assert loose.box is not None and tight.box is not None
        area = lambda b: (b[2] - b[0]) * (b[3] - b[1])  # noqa: E731
        assert area(tight.box) < area(loose.box)


def test_object_prompt_validates_labels() -> None:
    with pytest.raises(PromptError):
        ObjectPrompt(points=[(1.0, 1.0)], labels=[])
    with pytest.raises(PromptError):
        ObjectPrompt(points=[(1.0, 1.0)], labels=[2])

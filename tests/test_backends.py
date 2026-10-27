from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

from doodl.backends import ClassicalBackend, create_backend
from doodl.backends import sam2_backend as sam2_module
from doodl.backends.sam2_backend import (
    SAM2_MODELS,
    Sam2Backend,
    _resolve_dtype,
    checkpoint_urls,
    ensure_checkpoint,
)
from doodl.config import Settings
from doodl.errors import BackendError, PromptError
from doodl.pipeline import Pass, plan_passes
from doodl.prompts import ObjectPrompt, PromptStrategy
from doodl.samples import BALL_ID, SQUARE_ID
from helpers import FRAMES, circle_polygon_px, iou

# ------------------------------------------------------------- classical backend


@pytest.fixture
def classical() -> ClassicalBackend:
    return ClassicalBackend()


@pytest.mark.slow
class TestClassical:
    def test_prompt_snaps_to_object(self, classical, frames_dir, truth) -> None:
        _, truths = truth
        handle = classical.open_session(frames_dir, FRAMES)
        ball = truths[0][BALL_ID]
        mask = classical.set_prompt(handle, 0, 1, ObjectPrompt(polygon=circle_polygon_px(ball)))
        assert iou(mask, ball) > 0.85

    def test_tracks_two_objects_forward(self, classical, frames_dir, truth) -> None:
        _, truths = truth
        handle = classical.open_session(frames_dir, FRAMES)
        classical.set_prompt(
            handle, 0, 1, ObjectPrompt(polygon=circle_polygon_px(truths[0][BALL_ID]))
        )
        classical.set_prompt(
            handle, 0, 2, ObjectPrompt(polygon=circle_polygon_px(truths[0][SQUARE_ID]))
        )
        seen: dict[int, dict[int, np.ndarray]] = {}
        for index, masks in classical.propagate(
            handle, start_frame=0, reverse=False, obj_ids=[1, 2]
        ):
            seen[index] = masks
        assert sorted(seen) == list(range(FRAMES))
        ball_j = np.mean([iou(seen[i][1], truths[i][BALL_ID]) for i in range(1, FRAMES)])
        square_j = np.mean([iou(seen[i][2], truths[i][SQUARE_ID]) for i in range(1, FRAMES)])
        assert ball_j > 0.85
        assert square_j > 0.7

    def test_reverse_pass_covers_earlier_frames(self, classical, frames_dir, truth) -> None:
        _, truths = truth
        handle = classical.open_session(frames_dir, FRAMES)
        classical.set_prompt(
            handle, 20, 1, ObjectPrompt(polygon=circle_polygon_px(truths[20][BALL_ID]))
        )
        back = dict(classical.propagate(handle, start_frame=20, reverse=True, obj_ids=[1]))
        assert sorted(back) == list(range(0, 21))
        assert iou(back[10][1], truths[10][BALL_ID]) > 0.7

    def test_object_appears_only_from_its_first_prompt(self, classical, frames_dir, truth) -> None:
        _, truths = truth
        handle = classical.open_session(frames_dir, FRAMES)
        classical.set_prompt(
            handle, 0, 1, ObjectPrompt(polygon=circle_polygon_px(truths[0][BALL_ID]))
        )
        classical.set_prompt(
            handle, 10, 2, ObjectPrompt(polygon=circle_polygon_px(truths[10][SQUARE_ID]))
        )
        frames: dict[int, set[int]] = {
            i: set(m)
            for i, m in classical.propagate(handle, start_frame=0, reverse=False, obj_ids=[1, 2])
        }
        assert frames[5] == {1} and frames[10] == {1, 2}

    def test_mask_prompt_is_used_as_is(self, classical, frames_dir, truth) -> None:
        _, truths = truth
        handle = classical.open_session(frames_dir, FRAMES)
        exact = truths[0][BALL_ID]
        assert (classical.set_prompt(handle, 0, 1, ObjectPrompt(mask=exact)) == exact).all()

    def test_remove_object_stops_tracking_it(self, classical, frames_dir, truth) -> None:
        _, truths = truth
        handle = classical.open_session(frames_dir, FRAMES)
        classical.set_prompt(
            handle, 0, 1, ObjectPrompt(polygon=circle_polygon_px(truths[0][BALL_ID]))
        )
        classical.remove_object(handle, 1)
        assert list(classical.propagate(handle, start_frame=0, reverse=False, obj_ids=[1])) == []

    def test_prompt_without_outline_or_positive_click_is_an_error(
        self, classical, frames_dir
    ) -> None:
        handle = classical.open_session(frames_dir, FRAMES)
        with pytest.raises(PromptError):
            classical.set_prompt(handle, 0, 1, ObjectPrompt())

    def test_missing_frame_is_a_prompt_error(self, classical, frames_dir) -> None:
        handle = classical.open_session(frames_dir, FRAMES + 5)
        with pytest.raises(PromptError, match="could not be read"):
            classical.set_prompt(
                handle, FRAMES + 2, 1, ObjectPrompt(points=[(5.0, 5.0)], labels=[1])
            )


# ------------------------------------------------------ SAM 2.1 wrapper (stand-in)


class FakePredictor:
    """Mimics the parts of sam2's video predictor API that DooDL relies on."""

    def __init__(self, shape: tuple[int, int] = (60, 80)) -> None:
        self.shape = shape
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.objects: dict[int, tuple[int, int, int, int]] = {}
        self.generators_closed = 0
        self.remove_accepts_need_output = True

    def init_state(
        self, video_path: str, offload_video_to_cpu: bool, offload_state_to_cpu: bool, **_: Any
    ) -> dict:
        self.calls.append(
            (
                "init",
                {
                    "video_path": video_path,
                    "cpu": offload_video_to_cpu,
                    "state_cpu": offload_state_to_cpu,
                },
            )
        )
        return {"id": 1}

    def _logits(self) -> tuple[list[int], np.ndarray]:
        ids = list(self.objects)
        logits = np.full((len(ids), 1, *self.shape), -5.0, np.float32)
        for k, obj in enumerate(ids):
            x0, y0, x1, y1 = self.objects[obj]
            logits[k, 0, y0:y1, x0:x1] = 5.0
        return ids, logits

    def add_new_points_or_box(
        self,
        state: dict,
        frame_idx: int,
        obj_id: int,
        points: Any = None,
        labels: Any = None,
        clear_old_points: bool = True,
        normalize_coords: bool = True,
        box: Any = None,
    ) -> tuple:
        self.calls.append(
            (
                "points_or_box",
                {
                    "frame_idx": frame_idx,
                    "obj_id": obj_id,
                    "points": points,
                    "labels": labels,
                    "clear_old_points": clear_old_points,
                    "normalize_coords": normalize_coords,
                    "box": box,
                },
            )
        )
        if (points is None) != (labels is None):
            raise ValueError("points and labels must be provided together")
        if points is None and box is None:
            raise ValueError("at least one of points or box must be provided as input")
        x0, y0, x1, y1 = (int(v) for v in box) if box is not None else (5, 5, 20, 20)
        self.objects[obj_id] = (x0, y0, x1, y1)
        ids, logits = self._logits()
        return frame_idx, ids, logits

    def add_new_mask(self, state: dict, frame_idx: int, obj_id: int, mask: Any) -> tuple:
        self.calls.append(("mask", {"frame_idx": frame_idx, "obj_id": obj_id, "mask": mask}))
        ys, xs = np.nonzero(mask)
        self.objects[obj_id] = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
        ids, logits = self._logits()
        return frame_idx, ids, logits

    def propagate_in_video(
        self,
        state: dict,
        start_frame_idx: int | None = None,
        max_frame_num_to_track: int | None = None,
        reverse: bool = False,
    ):
        self.calls.append(("propagate", {"start": start_frame_idx, "reverse": reverse}))
        frames = range(start_frame_idx, -1, -1) if reverse else range(start_frame_idx, 6)
        try:
            for index in frames:
                ids, logits = self._logits()
                yield index, ids, logits
        finally:
            self.generators_closed += 1

    def remove_object(
        self, state: dict, obj_id: int, strict: bool = False, need_output: bool = True
    ) -> None:
        if not self.remove_accepts_need_output and need_output is False:
            raise TypeError("unexpected keyword")
        self.calls.append(("remove", {"obj_id": obj_id}))
        self.objects.pop(obj_id, None)

    def reset_state(self, state: dict) -> None:
        self.calls.append(("reset", {}))


@pytest.fixture
def tiny_frames(tmp_path: Path) -> Path:
    directory = tmp_path / "frames"
    directory.mkdir()
    for i in range(6):
        cv2.imwrite(str(directory / f"{i:05d}.jpg"), np.full((60, 80, 3), 40 * i, np.uint8))
    return directory


def make_backend(predictor: FakePredictor, **kwargs: Any) -> Sam2Backend:
    return Sam2Backend(predictor, **kwargs)


class TestSam2Wrapper:
    outline = ObjectPrompt(polygon=[(10, 10), (60, 10), (60, 45), (10, 45)])

    def test_open_session_offloads_frames_and_state_by_size(self, tiny_frames: Path) -> None:
        fake = FakePredictor()
        backend = make_backend(fake, offload_state_frames=3)
        backend.open_session(tiny_frames, 6)
        backend.open_session(tiny_frames, 2)
        first, second = (c[1] for c in fake.calls if c[0] == "init")
        assert first["cpu"] and first["state_cpu"] is True
        assert second["state_cpu"] is False
        assert first["video_path"] == str(tiny_frames)

    def test_box_points_prompt_uses_documented_arguments(self, tiny_frames: Path) -> None:
        fake = FakePredictor()
        backend = make_backend(fake)
        handle = backend.open_session(tiny_frames, 6)
        mask = backend.set_prompt(handle, 2, 7, self.outline)
        call = next(c[1] for c in fake.calls if c[0] == "points_or_box")
        assert call["frame_idx"] == 2 and call["obj_id"] == 7
        assert call["clear_old_points"] is True and call["normalize_coords"] is True
        assert call["box"].shape == (4,) and call["points"].shape[1] == 2
        assert len(call["points"]) == len(call["labels"]) > 0
        assert {0, 1} <= set(call["labels"].tolist())
        assert mask.dtype == bool and mask.shape == (60, 80) and mask[20, 30] and not mask[55, 75]

    def test_mask_strategy_calls_add_new_mask(self, tiny_frames: Path) -> None:
        fake = FakePredictor()
        backend = make_backend(fake, strategy=PromptStrategy.MASK)
        handle = backend.open_session(tiny_frames, 6)
        backend.set_prompt(handle, 0, 1, self.outline)
        name, call = fake.calls[-1]
        assert name == "mask" and call["mask"].dtype == bool

    def test_clicks_only_prompt_sends_points_without_box(self, tiny_frames: Path) -> None:
        fake = FakePredictor()
        backend = make_backend(fake)
        handle = backend.open_session(tiny_frames, 6)
        backend.set_prompt(handle, 0, 1, ObjectPrompt(points=[(30.0, 30.0)], labels=[1]))
        call = fake.calls[-1][1]
        assert call["box"] is None and call["points"].shape == (1, 2)

    def test_missing_frame_is_a_prompt_error(self, tiny_frames: Path) -> None:
        backend = make_backend(FakePredictor())
        handle = backend.open_session(tiny_frames, 6)
        with pytest.raises(PromptError):
            backend.set_prompt(handle, 99, 1, self.outline)

    def test_propagate_directions_and_object_filter(self, tiny_frames: Path) -> None:
        fake = FakePredictor()
        backend = make_backend(fake)
        handle = backend.open_session(tiny_frames, 6)
        backend.set_prompt(handle, 3, 1, self.outline)
        backend.set_prompt(handle, 3, 2, self.outline)
        forward = list(backend.propagate(handle, start_frame=3, reverse=False, obj_ids=[1]))
        backward = list(backend.propagate(handle, start_frame=3, reverse=True, obj_ids=[1, 2]))
        assert [i for i, _ in forward] == [3, 4, 5]
        assert all(set(m) == {1} for _, m in forward)  # object 2 filtered out
        assert [i for i, _ in backward] == [3, 2, 1, 0]
        assert all(set(m) == {1, 2} for _, m in backward)
        propagate_calls = [c[1] for c in fake.calls if c[0] == "propagate"]
        assert propagate_calls == [{"start": 3, "reverse": False}, {"start": 3, "reverse": True}]

    def test_abandoned_propagation_closes_the_generator(self, tiny_frames: Path) -> None:
        fake = FakePredictor()
        backend = make_backend(fake)
        handle = backend.open_session(tiny_frames, 6)
        backend.set_prompt(handle, 0, 1, self.outline)
        iterator = backend.propagate(handle, start_frame=0, reverse=False, obj_ids=[1])
        next(iterator)
        iterator.close()
        assert fake.generators_closed == 1

    def test_remove_object_falls_back_for_older_sam2(self, tiny_frames: Path) -> None:
        fake = FakePredictor()
        fake.remove_accepts_need_output = False
        backend = make_backend(fake)
        handle = backend.open_session(tiny_frames, 6)
        backend.set_prompt(handle, 0, 1, self.outline)
        backend.remove_object(handle, 1)
        assert 1 not in fake.objects

    def test_close_session_resets_state(self, tiny_frames: Path) -> None:
        fake = FakePredictor()
        backend = make_backend(fake)
        handle = backend.open_session(tiny_frames, 6)
        backend.close_session(handle)
        assert ("reset", {}) in fake.calls and handle.state is None

    def test_missing_object_in_model_output_is_reported(self) -> None:
        with pytest.raises(BackendError):
            make_backend(FakePredictor())._extract([2], np.zeros((1, 1, 4, 4)), 9)

    def test_info_label(self) -> None:
        backend = make_backend(FakePredictor(), model_name="base_plus", device="cuda")
        assert backend.info.label == "SAM 2.1 base plus" and backend.info.device == "cuda"


class FakeTorch:
    class _Cuda:
        def __init__(self, major: int) -> None:
            self.major = major

        def get_device_capability(self) -> tuple[int, int]:
            return self.major, 0

    float16, bfloat16 = "fp16", "bf16"

    def __init__(self, major: int) -> None:
        self.cuda = self._Cuda(major)


class TestPrecision:
    def test_t4_defaults_to_full_precision(self) -> None:
        assert _resolve_dtype("auto", "cuda", FakeTorch(7)) is None

    def test_ampere_defaults_to_bfloat16(self) -> None:
        assert _resolve_dtype("auto", "cuda", FakeTorch(8)) == "bf16"

    def test_explicit_choices_win(self) -> None:
        assert _resolve_dtype("fp16", "cuda", FakeTorch(7)) == "fp16"
        assert _resolve_dtype("fp32", "cuda", FakeTorch(9)) is None

    def test_cpu_never_autocasts(self) -> None:
        assert _resolve_dtype("bf16", "cpu", FakeTorch(9)) is None


class TestCheckpoints:
    def test_every_model_has_urls_and_a_sam21_config(self) -> None:
        for name, model in SAM2_MODELS.items():
            urls = checkpoint_urls(model)
            assert urls[0].startswith("https://dl.fbaipublicfiles.com/") and urls[1].startswith(
                "https://huggingface.co/"
            )
            assert model.file in urls[0] and model.config.startswith("configs/sam2.1/"), name

    def test_existing_checkpoint_is_reused_without_network(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        target = tmp_path / SAM2_MODELS["tiny"].file
        with target.open("wb") as f:
            f.truncate(60 * 1024 * 1024)
        monkeypatch.setattr(
            sam2_module.urllib.request,
            "urlopen",
            lambda *a, **k: pytest.fail("no network expected"),
        )
        assert ensure_checkpoint("tiny", tmp_path) == target

    def test_download_failure_lists_every_source_tried(self, tmp_path: Path, monkeypatch) -> None:
        def refuse(*_a: Any, **_k: Any) -> None:
            raise OSError("blocked")

        monkeypatch.setattr(sam2_module.urllib.request, "urlopen", refuse)
        with pytest.raises(BackendError) as info:
            ensure_checkpoint("tiny", tmp_path)
        message = str(info.value)
        assert (
            "dl.fbaipublicfiles.com" in message
            and "huggingface.co" in message
            and "manually" in message
        )
        assert not list(tmp_path.glob("*.part"))


class TestFactory:
    def test_sam2_without_torch_explains_what_to_install(self, monkeypatch) -> None:
        monkeypatch.setitem(sys.modules, "torch", None)
        with pytest.raises(BackendError, match="PyTorch"):
            create_backend(Settings(backend="sam2"))

    def test_sam2_without_the_package_explains_what_to_install(self, monkeypatch) -> None:
        monkeypatch.setitem(sys.modules, "torch", type(sys)("torch"))
        monkeypatch.setitem(sys.modules, "sam2", None)
        monkeypatch.setitem(sys.modules, "sam2.build_sam", None)
        with pytest.raises(BackendError, match="pip install"):
            create_backend(Settings(backend="sam2"))

    def test_auto_falls_back_to_the_cpu_baseline(self, monkeypatch, caplog) -> None:
        monkeypatch.setitem(sys.modules, "torch", None)
        backend = create_backend(Settings(backend="auto"))
        assert isinstance(backend, ClassicalBackend)
        assert "CPU baseline" in caplog.text

    def test_classical_is_explicit(self) -> None:
        assert create_backend(Settings(backend="classical")).info.name == "classical"

    def test_importing_the_app_never_needs_torch(self) -> None:
        """sys.modules[torch]=None makes any `import torch` raise, so a stray import fails."""
        code = (
            "import sys; sys.modules['torch'] = None; sys.modules['sam2'] = None\n"
            "import doodl.api, doodl.backends.sam2_backend, doodl.service, doodl.cli, doodl.colab\n"
        )
        result = subprocess.run(
            [sys.executable, "-I", "-c", code], capture_output=True, text=True, check=False
        )
        assert result.returncode == 0, result.stderr


# -------------------------------------------------------------- pass planning


class TestPlanPasses:
    def test_single_object_on_frame_zero_is_one_forward_pass(self) -> None:
        assert plan_passes({1: 0}) == [Pass(0, False, (1,))]

    def test_mid_clip_prompt_adds_a_backward_pass(self) -> None:
        assert plan_passes({1: 12}) == [Pass(12, False, (1,)), Pass(12, True, (1,))]

    def test_objects_prompted_on_different_frames(self) -> None:
        passes = plan_passes({1: 0, 2: 10, 3: 10, 4: 25})
        assert passes[0] == Pass(0, False, (1, 2, 3, 4))
        assert passes[1:] == [Pass(10, True, (2, 3)), Pass(25, True, (4,))]

    def test_pass_lengths(self) -> None:
        from doodl.pipeline import _pass_length

        assert _pass_length(Pass(5, False, (1,)), 40) == 35
        assert _pass_length(Pass(5, True, (1,)), 40) == 6

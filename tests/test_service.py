from __future__ import annotations

import json
import shutil
import threading
import zipfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from doodl.backends import ClassicalBackend
from doodl.config import Settings
from doodl.errors import BackendError, ConflictError, LimitError, NotFoundError, PromptError
from doodl.jobs import JobStatus
from doodl.metrics import j_and_f
from doodl.samples import BALL_ID, SQUARE_ID
from doodl.service import DooDLService
from doodl.sessions import Session, SessionStatus
from doodl.video import probe_video
from helpers import FRAMES, SIZE, circle_polygon_unit, iou

pytestmark = pytest.mark.slow


class GatedBackend(ClassicalBackend):
    """Holds propagation until released, so tests can act while a job is running."""

    def __init__(self) -> None:
        super().__init__()
        self.started = threading.Event()
        self.release = threading.Event()

    def propagate(self, handle: Any, **kwargs: Any) -> Iterator:
        self.started.set()
        assert self.release.wait(20), "test never released the backend"
        yield from super().propagate(handle, **kwargs)


class CountingBackend(ClassicalBackend):
    def __init__(self) -> None:
        super().__init__()
        self.opened = 0
        self.closed = 0

    def open_session(self, frames_dir: Path, num_frames: int) -> Any:
        self.opened += 1
        return super().open_session(frames_dir, num_frames)

    def close_session(self, handle: Any) -> None:
        self.closed += 1
        super().close_session(handle)


def make_service(settings: Settings, backend: Any = None) -> DooDLService:
    return DooDLService(settings, backend or ClassicalBackend())


def ingest(service: DooDLService, clip: Path) -> Session:
    session = service.begin_session()
    source = session.dir / "source.mp4"
    shutil.copy(clip, source)
    return service.finish_ingest(session, source)


@pytest.fixture
def service(settings: Settings) -> Iterator[DooDLService]:
    svc = make_service(settings)
    yield svc
    svc.close()


def ball(truth: tuple[list, list], frame: int = 0) -> list[tuple[float, float]]:
    return circle_polygon_unit(truth[1][frame][BALL_ID])


def square(truth: tuple[list, list], frame: int = 0) -> list[tuple[float, float]]:
    return circle_polygon_unit(truth[1][frame][SQUARE_ID])


class TestIngest:
    def test_ingest_builds_frames(self, service: DooDLService, clip: Path) -> None:
        session = ingest(service, clip)
        assert (session.width, session.height) == SIZE and session.frame_count == FRAMES
        assert session.status is SessionStatus.READY

    def test_unsupported_extension(self, service: DooDLService, tmp_path: Path) -> None:
        session = service.begin_session()
        source = tmp_path / "notes.txt"
        source.write_text("hi")
        with pytest.raises(PromptError, match="Unsupported"):
            service.finish_ingest(session, source)

    def test_sample_clip_is_ready_to_use(self, service: DooDLService) -> None:
        session = service.ingest_sample()
        assert session.frame_count == 72 and (session.width, session.height) == (640, 360)
        assert probe_video(session.source_path).has_audio is False  # type: ignore[arg-type]


class TestObjects:
    def test_add_object_returns_a_snapped_preview(self, service, clip, truth) -> None:
        session = ingest(service, clip)
        preview = service.add_object(session, 0, ball(truth))
        assert iou(preview.mask, truth[1][0][BALL_ID]) > 0.85
        assert preview.object.obj_id == 1 and preview.object.color.startswith("#")
        assert preview.png[:4] == b"\x89PNG" and 0.0 < preview.coverage < 0.2

    def test_objects_get_distinct_ids_and_colours(self, service, clip, truth) -> None:
        session = ingest(service, clip)
        a = service.add_object(session, 0, ball(truth)).object
        b = service.add_object(session, 0, square(truth)).object
        assert (a.obj_id, b.obj_id) == (1, 2) and a.color != b.color

    def test_object_limit(self, settings, clip, truth) -> None:
        svc = make_service(settings.replace(max_objects=1))
        try:
            session = ingest(svc, clip)
            svc.add_object(session, 0, ball(truth))
            with pytest.raises(LimitError, match="up to 1 objects"):
                svc.add_object(session, 0, square(truth))
        finally:
            svc.close()

    def test_failed_outline_leaves_no_trace(self, service, clip) -> None:
        session = ingest(service, clip)
        with pytest.raises(PromptError):
            service.add_object(session, 0, [(0.5, 0.5), (0.5005, 0.5), (0.5, 0.5005)])
        assert session.objects == {} and session.next_obj_id == 1

    def test_frame_out_of_range(self, service, clip, truth) -> None:
        session = ingest(service, clip)
        with pytest.raises(PromptError, match="outside this video"):
            service.add_object(session, FRAMES, ball(truth))

    def test_refine_accumulates_and_undo_removes_clicks(self, service, clip, truth) -> None:
        session = ingest(service, clip)
        service.add_object(session, 0, ball(truth))
        w, h = SIZE
        corner = (0.02, 0.9)  # clearly background
        service.refine_object(session, 1, 0, [corner], [0])
        service.refine_object(session, 1, 0, [(0.15, 0.4)], [1])
        record = session.objects[1].records[0]
        assert record.labels == [0, 1] and len(record.points) == 2
        assert record.points[0] == pytest.approx((corner[0] * w, corner[1] * h))
        service.refine_object(session, 1, 0, [], [], undo=True)
        assert session.objects[1].records[0].labels == [0]  # refine stores a fresh copy each time

    def test_undo_with_nothing_to_undo(self, service, clip, truth) -> None:
        session = ingest(service, clip)
        service.add_object(session, 0, ball(truth))
        with pytest.raises(PromptError, match="no point to undo"):
            service.refine_object(session, 1, 0, [], [], undo=True)

    def test_refining_on_a_new_frame_adds_a_prompt_frame(self, service, clip, truth) -> None:
        session = ingest(service, clip)
        service.add_object(session, 0, ball(truth))
        cx = float(np.nonzero(truth[1][20][BALL_ID])[1].mean()) / SIZE[0]
        cy = float(np.nonzero(truth[1][20][BALL_ID])[0].mean()) / SIZE[1]
        service.refine_object(session, 1, 20, [(cx, cy)], [1])
        assert sorted(session.objects[1].records) == [0, 20]

    def test_refine_unknown_object(self, service, clip) -> None:
        session = ingest(service, clip)
        with pytest.raises(NotFoundError):
            service.refine_object(session, 5, 0, [(0.5, 0.5)], [1])

    def test_remove_object(self, service, clip, truth) -> None:
        session = ingest(service, clip)
        service.add_object(session, 0, ball(truth))
        service.remove_object(session, 1)
        assert session.objects == {}
        with pytest.raises(NotFoundError):
            service.remove_object(session, 1)


class TestTracking:
    def test_end_to_end_quality_and_outputs(self, service, clip, truth) -> None:
        session = ingest(service, clip)
        service.add_object(session, 0, ball(truth))
        service.add_object(session, 0, square(truth))
        job = service.wait(service.start_tracking(session), timeout=120)
        assert job.status is JobStatus.DONE, job.error
        assert job.progress == 1.0 and session.status is SessionStatus.DONE
        assert session.result is not None and session.result.frames == FRAMES

        for obj_id, key in ((1, BALL_ID), (2, SQUARE_ID)):
            preds = [(session.masks.read(i) == obj_id) for i in range(FRAMES)]
            gts = [truth[1][i][key] for i in range(FRAMES)]
            assert j_and_f(preds, gts)["J&F"] > (0.85 if obj_id == 1 else 0.7)

        results = session.results_dir
        assert {p.name for p in results.iterdir()} == {
            "overlay.mp4",
            "cutout.mp4",
            "masks.zip",
            "annotations.json",
        }
        assert probe_video(results / "overlay.mp4").width == SIZE[0]
        with zipfile.ZipFile(results / "masks.zip") as z:
            assert sum(n.endswith(".png") for n in z.namelist()) == FRAMES
        annotations = json.loads((results / "annotations.json").read_text())
        assert len(annotations["frames"]) == FRAMES
        assert {o["id"] for o in annotations["objects"]} == {1, 2}

    def test_prompt_mid_clip_tracks_both_directions(self, service, clip, truth) -> None:
        session = ingest(service, clip)
        service.add_object(session, 20, ball(truth, 20))
        job = service.wait(service.start_tracking(session), timeout=120)
        assert job.status is JobStatus.DONE, job.error
        assert all(session.masks.has(i) for i in range(FRAMES))
        assert iou(session.masks.read(5) == 1, truth[1][5][BALL_ID]) > 0.6

    def test_tracking_without_objects_is_refused(self, service, clip) -> None:
        session = ingest(service, clip)
        with pytest.raises(PromptError, match="Outline at least one"):
            service.start_tracking(session)
        assert session.status is SessionStatus.READY

    def test_edits_are_blocked_while_tracking_and_cancel_restores_state(
        self, settings, clip, truth
    ) -> None:
        backend = GatedBackend()
        svc = make_service(settings, backend)
        try:
            session = ingest(svc, clip)
            svc.add_object(session, 0, ball(truth))
            job = svc.start_tracking(session)
            assert backend.started.wait(10)
            assert session.status is SessionStatus.TRACKING
            with pytest.raises(ConflictError):
                svc.add_object(session, 0, square(truth))
            with pytest.raises(ConflictError):
                svc.start_tracking(session)
            svc.cancel(job.id)
            backend.release.set()
            svc.wait(job, timeout=30)
            assert job.status is JobStatus.CANCELLED
            assert session.status is SessionStatus.READY and session.result is None
            assert session.masks.count() == 0
        finally:
            backend.release.set()
            svc.close()

    def test_results_go_stale_when_objects_change(self, service, clip, truth) -> None:
        session = ingest(service, clip)
        service.add_object(session, 0, ball(truth))
        service.wait(service.start_tracking(session), timeout=120)
        assert session.result is not None and not session.result_stale
        service.add_object(session, 0, square(truth))
        assert session.result_stale
        service.remove_object(session, 1)
        service.remove_object(session, 2)
        assert session.result is None and session.status is SessionStatus.READY

    def test_unexpected_failure_hides_internals_and_recovers(self, settings, clip, truth) -> None:
        class Boom(ClassicalBackend):
            def propagate(self, handle: Any, **kwargs: Any) -> Iterator:
                raise RuntimeError("secret internal detail at /srv/x.py")
                yield  # pragma: no cover

        svc = make_service(settings, Boom())
        try:
            session = ingest(svc, clip)
            svc.add_object(session, 0, ball(truth))
            job = svc.wait(svc.start_tracking(session), timeout=30)
            assert job.status is JobStatus.FAILED
            assert "secret" not in (job.error or "") and "server log" in (job.error or "")
            assert session.status is SessionStatus.READY
        finally:
            svc.close()

    def test_known_errors_pass_through_and_oom_is_explained(self, settings, clip, truth) -> None:
        class OutOfMemoryError(Exception):
            pass

        for exc, expected in (
            (BackendError("The model is warming up."), "warming up"),
            (OutOfMemoryError(), "ran out of memory"),
        ):

            class Failing(ClassicalBackend):
                def propagate(self, handle: Any, **kwargs: Any) -> Iterator:
                    raise exc  # noqa: B023
                    yield  # pragma: no cover

            svc = make_service(settings, Failing())
            try:
                session = ingest(svc, clip)
                svc.add_object(session, 0, ball(truth))
                job = svc.wait(svc.start_tracking(session), timeout=30)
                assert job.status is JobStatus.FAILED and expected in (job.error or "")
            finally:
                svc.close()


class TestSessionLifecycle:
    def test_oldest_idle_session_is_evicted_at_the_limit(self, settings, clip) -> None:
        svc = make_service(settings.replace(max_sessions=1))
        try:
            first = ingest(svc, clip)
            second = svc.begin_session()
            assert len(svc.store) == 1 and not first.dir.exists() and second.dir.exists()
        finally:
            svc.close()

    def test_busy_server_refuses_new_sessions(self, settings, clip) -> None:
        svc = make_service(settings.replace(max_sessions=1))
        try:
            first = ingest(svc, clip)
            first.status = SessionStatus.TRACKING
            with pytest.raises(ConflictError, match="busy"):
                svc.begin_session()
        finally:
            first.status = SessionStatus.READY
            svc.close()

    def test_expired_sessions_are_swept_but_tracking_ones_are_not(self, settings, clip) -> None:
        svc = make_service(settings)
        try:
            old, busy = ingest(svc, clip), ingest(svc, clip)
            for s in (old, busy):
                s.last_access -= 3 * 3600
            busy.status = SessionStatus.TRACKING
            assert svc.sweep() == 1
            assert not old.dir.exists() and busy.dir.exists()
            with pytest.raises(NotFoundError, match="expired"):
                svc.store.get(old.id)
        finally:
            busy.status = SessionStatus.READY
            svc.close()

    def test_evicted_handle_is_rebuilt_by_replaying_prompts(self, settings, clip, truth) -> None:
        backend = CountingBackend()
        svc = make_service(settings.replace(max_active_handles=1), backend)
        try:
            a, b = ingest(svc, clip), ingest(svc, clip)
            svc.add_object(a, 0, ball(truth))
            svc.add_object(b, 0, ball(truth))  # evicts A's handle
            assert a.handle is None and b.handle is not None and backend.closed == 1
            job = svc.wait(svc.start_tracking(a), timeout=120)  # replays A's outline
            assert job.status is JobStatus.DONE, job.error
            assert backend.opened == 3
            preds = [(a.masks.read(i) == 1) for i in range(FRAMES)]
            assert j_and_f(preds, [truth[1][i][BALL_ID] for i in range(FRAMES)])["J&F"] > 0.85
        finally:
            svc.close()

    def test_discard_closes_the_handle_and_removes_files(self, settings, clip, truth) -> None:
        backend = CountingBackend()
        svc = make_service(settings, backend)
        try:
            session = ingest(svc, clip)
            svc.add_object(session, 0, ball(truth))
            svc.discard(session)
            assert backend.closed == 1 and not session.dir.exists() and len(svc.store) == 0
        finally:
            svc.close()


def test_job_lookup_errors(service: DooDLService) -> None:
    with pytest.raises(NotFoundError):
        service.job("nope")
    with pytest.raises(NotFoundError):
        service.cancel("nope")

"""Application logic: sessions, objects, prompts and tracking jobs.

The HTTP layer is a thin wrapper around this class, so everything here is testable
without a web server.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .backends.base import SegmentationBackend
from .config import Settings
from .errors import ConflictError, LimitError, NotFoundError, PromptError
from .jobs import Job, JobCancelled, JobManager, JobStatus
from .pipeline import run_tracking
from .render import masks_to_overlay_png
from .samples import write_sample_video
from .sessions import PromptRecord, Session, SessionStatus, SessionStore, TrackedObject, TrackResult
from .video import VIDEO_EXTENSIONS, extract_frames, probe_video

log = logging.getLogger(__name__)


@dataclass
class Preview:
    object: TrackedObject
    frame_idx: int
    mask: np.ndarray
    png: bytes

    @property
    def coverage(self) -> float:
        return float(self.mask.mean())


class DooDLService:
    def __init__(
        self,
        settings: Settings,
        backend: SegmentationBackend,
        store: SessionStore | None = None,
        jobs: JobManager | None = None,
    ) -> None:
        self.settings = settings
        self.backend = backend
        self.store = store or SessionStore(settings.data_dir, settings.session_ttl_min * 60)
        self.jobs = jobs or JobManager()
        # One model, one GPU: every call that touches the backend takes this lock.
        self._backend_lock = threading.RLock()
        self._result_counter = 0

    # ------------------------------------------------------------------ sessions

    def begin_session(self) -> Session:
        self.sweep()
        if len(self.store) >= self.settings.max_sessions:
            idle = [s for s in self.store.all() if s.status is not SessionStatus.TRACKING]
            if not idle:
                raise ConflictError(
                    "The server is busy tracking other videos. Try again in a minute."
                )
            self.discard(min(idle, key=lambda s: s.last_access))
        return self.store.create()

    def finish_ingest(self, session: Session, source: Path) -> Session:
        """Validate the uploaded file and turn it into frames."""
        if source.suffix.lower() not in VIDEO_EXTENSIONS:
            raise PromptError(
                "Unsupported file type. Use "
                + ", ".join(sorted(e.lstrip(".").upper() for e in VIDEO_EXTENSIONS))
                + "."
            )
        s = self.settings
        info = probe_video(source)
        frames = extract_frames(
            source,
            session.frames_dir,
            info,
            max_long_side=s.max_long_side,
            max_fps=s.max_fps,
            max_duration_s=s.max_duration_s,
            max_frames=s.max_frames,
        )
        session.source_path = source
        session.video = info
        session.frames = frames
        return session

    def ingest_sample(self) -> Session:
        session = self.begin_session()
        try:
            source = session.dir / "source.mp4"
            write_sample_video(source)
            return self.finish_ingest(session, source)
        except BaseException:
            self.discard(session)
            raise

    def discard(self, session: Session) -> None:
        with session.lock:
            if session.job_id:
                self.jobs.cancel(session.job_id)
            self._close_handle(session)
        self.store.remove(session)

    def sweep(self) -> int:
        expired = self.store.expired()
        for session in expired:
            log.info("removing expired session %s", session.id)
            self.discard(session)
        return len(expired)

    # ------------------------------------------------------------------- handles

    def _close_handle(self, session: Session) -> None:
        if session.handle is not None:
            with self._backend_lock:
                try:
                    self.backend.close_session(session.handle)
                except Exception:
                    log.exception("closing backend session failed")
            session.handle = None

    def _ensure_handle(self, session: Session) -> Any:
        """Open the model's view of this video, replaying prompts if it was evicted."""
        if session.handle is not None:
            return session.handle
        others = [s for s in self.store.all() if s.handle is not None and s is not session]
        others.sort(key=lambda s: s.last_access)
        while len(others) >= self.settings.max_active_handles:
            victim = others.pop(0)
            if victim.status is SessionStatus.TRACKING:
                continue
            with victim.lock:
                self._close_handle(victim)
        assert session.frames is not None
        handle = self.backend.open_session(session.frames.directory, session.frame_count)
        try:
            for obj in session.objects.values():
                for record in obj.records.values():
                    self.backend.set_prompt(
                        handle, record.frame_idx, obj.obj_id, record.to_prompt()
                    )
        except BaseException:
            self.backend.close_session(handle)
            raise
        session.handle = handle
        return handle

    # ------------------------------------------------------------------- objects

    def _require_idle(self, session: Session) -> None:
        if session.status is SessionStatus.TRACKING:
            raise ConflictError("Tracking is running. Wait for it to finish or cancel it first.")

    def _pixels(
        self, session: Session, points: list[tuple[float, float]]
    ) -> list[tuple[float, float]]:
        w, h = session.width, session.height
        return [(min(max(x, 0.0), 1.0) * w, min(max(y, 0.0), 1.0) * h) for x, y in points]

    def _check_frame(self, session: Session, frame_idx: int) -> None:
        if not 0 <= frame_idx < session.frame_count:
            raise PromptError(
                f"Frame {frame_idx} is outside this video (0 to {session.frame_count - 1})."
            )

    def _preview(
        self, session: Session, obj: TrackedObject, frame_idx: int, mask: np.ndarray
    ) -> Preview:
        png = masks_to_overlay_png({obj.obj_id: mask}, (session.height, session.width))
        return Preview(obj, frame_idx, mask, png)

    def add_object(
        self, session: Session, frame_idx: int, polygon: list[tuple[float, float]]
    ) -> Preview:
        """Create an object from an outline (normalised 0-1 coordinates)."""
        with session.lock:
            self._require_idle(session)
            self._check_frame(session, frame_idx)
            if len(session.objects) >= self.settings.max_objects:
                raise LimitError(
                    f"You can track up to {self.settings.max_objects} objects at a time."
                )
            record = PromptRecord(frame_idx=frame_idx, polygon=self._pixels(session, polygon))
            obj_id = session.next_obj_id
            with self._backend_lock:
                handle = self._ensure_handle(session)
                try:
                    mask = self.backend.set_prompt(handle, frame_idx, obj_id, record.to_prompt())
                except BaseException:
                    self.backend.remove_object(handle, obj_id)
                    raise
            obj = TrackedObject(obj_id=obj_id, name=f"Object {obj_id}", records={frame_idx: record})
            session.objects[obj_id] = obj
            session.next_obj_id += 1
            self._mark_stale(session)
            return self._preview(session, obj, frame_idx, mask)

    def refine_object(
        self,
        session: Session,
        obj_id: int,
        frame_idx: int,
        points: list[tuple[float, float]],
        labels: list[int],
        undo: bool = False,
    ) -> Preview:
        """Add (or undo) add/remove clicks for an object on a frame and return the new mask."""
        with session.lock:
            self._require_idle(session)
            self._check_frame(session, frame_idx)
            obj = self._object(session, obj_id)
            record = (obj.records.get(frame_idx) or PromptRecord(frame_idx=frame_idx)).copy()
            if undo:
                if not record.points:
                    raise PromptError("There is no point to undo on this frame.")
                record.points.pop()
                record.labels.pop()
            record.points.extend(self._pixels(session, points))
            record.labels.extend(labels)
            if not record.points and record.polygon is None:
                raise PromptError("Click on the object to add a point.")
            with self._backend_lock:
                handle = self._ensure_handle(session)
                mask = self.backend.set_prompt(handle, frame_idx, obj_id, record.to_prompt())
            obj.records[frame_idx] = record
            self._mark_stale(session)
            return self._preview(session, obj, frame_idx, mask)

    def remove_object(self, session: Session, obj_id: int) -> None:
        with session.lock:
            self._require_idle(session)
            self._object(session, obj_id)
            if session.handle is not None:
                with self._backend_lock:
                    self.backend.remove_object(session.handle, obj_id)
            del session.objects[obj_id]
            self._mark_stale(session)
            if not session.objects:
                session.result = None
                session.status = SessionStatus.READY

    def _object(self, session: Session, obj_id: int) -> TrackedObject:
        obj = session.objects.get(obj_id)
        if obj is None:
            raise NotFoundError("That object no longer exists.")
        return obj

    @staticmethod
    def _mark_stale(session: Session) -> None:
        if session.result is not None:
            session.result_stale = True

    # ------------------------------------------------------------------ tracking

    def start_tracking(self, session: Session) -> Job:
        with session.lock:
            self._require_idle(session)
            if not session.objects:
                raise PromptError("Outline at least one object before tracking.")
            session.status = SessionStatus.TRACKING
            self._result_counter += 1
            version = self._result_counter

        def work(job: Job) -> dict[str, Any]:
            try:
                with self._backend_lock:
                    job.check_cancelled()
                    job.update(0.0, "Loading the video into the model")
                    with session.lock:
                        handle = self._ensure_handle(session)
                    result = run_tracking(session, self.backend, handle, job, version)
                with session.lock:
                    session.result = TrackResult(**result)
                    session.result_stale = False
                    session.status = SessionStatus.DONE
                return result
            except JobCancelled:
                with session.lock:
                    session.result = None
                    session.status = SessionStatus.READY
                    session.masks.clear()
                raise
            except BaseException:
                with session.lock:
                    session.status = SessionStatus.READY
                raise

        job = self.jobs.submit(session.id, work)
        session.job_id = job.id
        return job

    def job(self, job_id: str) -> Job:
        job = self.jobs.get(job_id)
        if job is None:
            raise NotFoundError("That job no longer exists.")
        return job

    def cancel(self, job_id: str) -> Job:
        job = self.jobs.cancel(job_id)
        if job is None:
            raise NotFoundError("That job no longer exists.")
        return job

    def wait(self, job: Job, timeout: float = 120.0) -> Job:
        """Block until a job ends (used by tests and scripts)."""
        deadline = time.time() + timeout
        while job.status in (JobStatus.QUEUED, JobStatus.RUNNING):
            if time.time() > deadline:
                raise TimeoutError("job did not finish in time")
            time.sleep(0.05)
        return job

    def close(self) -> None:
        self.jobs.shutdown()
        for session in self.store.all():
            self._close_handle(session)

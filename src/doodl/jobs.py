"""A tiny single-worker job queue: one GPU, one tracking job at a time."""

from __future__ import annotations

import logging
import secrets
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .errors import DooDLError

log = logging.getLogger(__name__)


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobCancelled(Exception):
    """Raised inside a job when the user cancels it."""


@dataclass
class Job:
    id: str
    session_id: str
    status: JobStatus = JobStatus.QUEUED
    progress: float = 0.0
    message: str = "Waiting for the GPU"
    error: str | None = None
    created: float = field(default_factory=time.time)
    started: float | None = None
    finished: float | None = None
    result: dict[str, Any] = field(default_factory=dict)
    cancel_event: threading.Event = field(default_factory=threading.Event)

    def update(self, progress: float, message: str) -> None:
        self.progress = max(0.0, min(1.0, progress))
        self.message = message

    def check_cancelled(self) -> None:
        if self.cancel_event.is_set():
            raise JobCancelled

    @property
    def elapsed_s(self) -> float:
        if self.started is None:
            return 0.0
        return (self.finished or time.time()) - self.started

    @property
    def finished_ok(self) -> bool:
        return self.status in (JobStatus.DONE, JobStatus.FAILED, JobStatus.CANCELLED)


def friendly_error(exc: BaseException) -> str:
    """Words for the person using the app. Details stay in the server log."""
    if isinstance(exc, DooDLError):
        return str(exc)
    if type(exc).__name__ == "OutOfMemoryError":
        return (
            "The GPU ran out of memory. Try a shorter clip, fewer objects, "
            "or a smaller model (DOODL_SAM2_MODEL=tiny)."
        )
    return "Tracking failed unexpectedly. The server log has the details."


class JobManager:
    def __init__(self, workers: int = 1, keep_seconds: float = 3600) -> None:
        self._pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="doodl-job")
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._keep = keep_seconds

    def submit(self, session_id: str, work: Callable[[Job], dict[str, Any]]) -> Job:
        job = Job(id=secrets.token_urlsafe(9), session_id=session_id)
        with self._lock:
            self._prune()
            self._jobs[job.id] = job
        self._pool.submit(self._run, job, work)
        return job

    def _run(self, job: Job, work: Callable[[Job], dict[str, Any]]) -> None:
        if job.cancel_event.is_set():
            job.status = JobStatus.CANCELLED
            job.finished = time.time()
            return
        job.status = JobStatus.RUNNING
        job.started = time.time()
        job.message = "Starting"
        try:
            job.result = work(job)
            job.status = JobStatus.DONE
            job.progress = 1.0
            job.message = "Finished"
        except JobCancelled:
            job.status = JobStatus.CANCELLED
            job.message = "Cancelled"
        except Exception as exc:
            log.exception("job %s failed", job.id)
            job.status = JobStatus.FAILED
            job.error = friendly_error(exc)
            job.message = "Failed"
        finally:
            job.finished = time.time()

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def cancel(self, job_id: str) -> Job | None:
        job = self.get(job_id)
        if job is not None and not job.finished_ok:
            job.cancel_event.set()
        return job

    def _prune(self) -> None:
        cutoff = time.time() - self._keep
        stale = [i for i, j in self._jobs.items() if j.finished and j.finished < cutoff]
        for job_id in stale:
            del self._jobs[job_id]

    def shutdown(self) -> None:
        for job in list(self._jobs.values()):
            job.cancel_event.set()
        self._pool.shutdown(wait=False, cancel_futures=True)

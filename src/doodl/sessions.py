"""In-memory session registry. Frames, masks and results live on disk under the data dir."""

from __future__ import annotations

import secrets
import shutil
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from .errors import NotFoundError
from .export import MaskStore
from .prompts import Box, ObjectPrompt, Point
from .render import color_hex
from .video import FrameSet, VideoInfo

DOWNLOADS = ("overlay.mp4", "cutout.mp4", "masks.zip", "annotations.json")


class SessionStatus(str, Enum):
    READY = "ready"
    TRACKING = "tracking"
    DONE = "done"


@dataclass
class PromptRecord:
    """One object's prompt on one frame, in pixel coordinates."""

    frame_idx: int
    polygon: list[Point] | None = None
    box: Box | None = None
    points: list[Point] = field(default_factory=list)
    labels: list[int] = field(default_factory=list)

    def to_prompt(self) -> ObjectPrompt:
        return ObjectPrompt(
            polygon=list(self.polygon) if self.polygon else None,
            box=self.box,
            points=list(self.points),
            labels=list(self.labels),
        )

    def copy(self) -> PromptRecord:
        return PromptRecord(
            self.frame_idx,
            list(self.polygon) if self.polygon else None,
            self.box,
            list(self.points),
            list(self.labels),
        )


@dataclass
class TrackedObject:
    obj_id: int
    name: str
    records: dict[int, PromptRecord] = field(default_factory=dict)

    @property
    def color(self) -> str:
        return color_hex(self.obj_id)

    @property
    def first_frame(self) -> int:
        return min(self.records)


@dataclass
class TrackResult:
    version: int
    frames: int
    elapsed_s: float
    fps: float
    peak_gpu_gb: float | None
    files: dict[str, str]


class Session:
    def __init__(self, session_id: str, directory: Path) -> None:
        self.id = session_id
        self.dir = directory
        self.frames_dir = directory / "frames"
        self.results_dir = directory / "results"
        self.thumbs_dir = directory / "thumbs"
        self.masks = MaskStore(directory / "masks")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.thumbs_dir.mkdir(parents=True, exist_ok=True)

        self.video: VideoInfo | None = None
        self.frames: FrameSet | None = None
        self.source_path: Path | None = None
        self.objects: dict[int, TrackedObject] = {}
        self.next_obj_id = 1
        self.status = SessionStatus.READY
        self.result: TrackResult | None = None
        self.result_stale = False
        self.job_id: str | None = None
        self.handle: Any = None

        self.created = time.time()
        self.last_access = self.created
        self.lock = threading.RLock()

    def touch(self) -> None:
        self.last_access = time.time()

    @property
    def width(self) -> int:
        assert self.frames is not None
        return self.frames.width

    @property
    def height(self) -> int:
        assert self.frames is not None
        return self.frames.height

    @property
    def frame_count(self) -> int:
        assert self.frames is not None
        return self.frames.count


class SessionStore:
    def __init__(self, root: Path, ttl_seconds: float) -> None:
        self.root = root / "sessions"
        self.root.mkdir(parents=True, exist_ok=True)
        self.ttl = ttl_seconds
        self._sessions: dict[str, Session] = {}
        self._lock = threading.Lock()

    def create(self) -> Session:
        session_id = secrets.token_urlsafe(9)
        directory = self.root / session_id
        directory.mkdir(parents=True)
        session = Session(session_id, directory)
        with self._lock:
            self._sessions[session_id] = session
        return session

    def get(self, session_id: str) -> Session:
        with self._lock:
            session = self._sessions.get(session_id)
        if session is None:
            raise NotFoundError(
                "This session has expired or does not exist. Upload the video again."
            )
        session.touch()
        return session

    def remove(self, session: Session) -> None:
        with self._lock:
            self._sessions.pop(session.id, None)
        shutil.rmtree(session.dir, ignore_errors=True)

    def all(self) -> list[Session]:
        with self._lock:
            return list(self._sessions.values())

    def expired(self, now: float | None = None) -> list[Session]:
        now = time.time() if now is None else now
        return [
            s
            for s in self.all()
            if s.status is not SessionStatus.TRACKING and now - s.last_access > self.ttl
        ]

    def __len__(self) -> int:
        with self._lock:
            return len(self._sessions)

"""Request and response models for the HTTP API."""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, Field, field_validator

Unit = Annotated[float, Field(ge=0.0, le=1.0)]
UnitPoint = tuple[Unit, Unit]


class ObjectCreate(BaseModel):
    """A hand-drawn outline in normalised coordinates (0 to 1 across the frame)."""

    frame_idx: int = Field(ge=0)
    polygon: list[UnitPoint] = Field(min_length=3, max_length=500)


class ClicksIn(BaseModel):
    frame_idx: int = Field(ge=0)
    points: list[UnitPoint] = Field(default_factory=list, max_length=20)
    labels: list[int] = Field(default_factory=list, max_length=20)
    undo: bool = False

    @field_validator("labels")
    @classmethod
    def _binary(cls, value: list[int]) -> list[int]:
        if any(v not in (0, 1) for v in value):
            raise ValueError("labels must be 1 (add) or 0 (remove)")
        return value


class ObjectOut(BaseModel):
    id: int
    name: str
    color: str
    prompt_frames: list[int]


class PreviewOut(BaseModel):
    object: ObjectOut
    frame_idx: int
    png_b64: str
    coverage: float


class ResultOut(BaseModel):
    version: int
    frames: int
    elapsed_s: float
    fps: float
    peak_gpu_gb: float | None
    stale: bool
    downloads: dict[str, str]


class SessionOut(BaseModel):
    id: str
    width: int
    height: int
    fps: float
    frame_count: int
    duration_s: float
    trimmed: bool
    has_audio: bool
    status: str
    objects: list[ObjectOut]
    result: ResultOut | None
    job_id: str | None


class JobOut(BaseModel):
    id: str
    session_id: str
    status: str
    progress: float
    message: str
    error: str | None
    elapsed_s: float


class LimitsOut(BaseModel):
    max_upload_mb: int
    max_duration_s: float
    max_frames: int
    max_objects: int
    extensions: list[str]


class HealthOut(BaseModel):
    status: str
    version: str
    backend: str
    backend_label: str
    device: str
    auth_required: bool
    limits: LimitsOut
    palette: list[str]

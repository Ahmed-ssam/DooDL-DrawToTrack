"""Runtime settings, read from DOODL_* environment variables."""

from __future__ import annotations

import dataclasses
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

ENV_PREFIX = "DOODL_"

BACKENDS = ("auto", "sam2", "classical")
SAM2_MODELS = ("tiny", "small", "base_plus", "large")
PRECISIONS = ("auto", "fp32", "fp16", "bf16")
STRATEGIES = ("box_points", "mask", "grabcut")


@dataclass(frozen=True)
class Settings:
    """Everything tunable. Defaults are chosen for a free Colab T4 session."""

    data_dir: Path = Path(tempfile.gettempdir()) / "doodl"
    checkpoint_dir: Path = Path.home() / ".cache" / "doodl" / "checkpoints"

    # Model
    backend: str = "auto"  # auto | sam2 | classical
    sam2_model: str = "tiny"  # tiny | small | base_plus | large
    device: str = "auto"  # auto | cuda | cpu | mps
    precision: str = "auto"  # auto | fp32 | fp16 | bf16
    prompt_strategy: str = "box_points"  # box_points | mask | grabcut
    positive_points: int = 4
    negative_points: int = 4

    # Input limits. SAM 2 keeps every frame in memory (about 12.6 MB per frame
    # at its 1024x1024 input size), so the frame cap protects Colab's RAM.
    max_upload_mb: int = 200
    max_duration_s: float = 30.0
    max_frames: int = 450
    max_fps: float = 30.0
    max_long_side: int = 1280
    max_objects: int = 8
    offload_state_frames: int = 150

    # Housekeeping
    max_sessions: int = 6
    max_active_handles: int = 2
    session_ttl_min: int = 60

    # Set to require ?token=... (or the X-DooDL-Token header) on every API call.
    access_token: str | None = None

    def __post_init__(self) -> None:
        _choice("backend", self.backend, BACKENDS)
        _choice("sam2_model", self.sam2_model, SAM2_MODELS)
        _choice("precision", self.precision, PRECISIONS)
        _choice("prompt_strategy", self.prompt_strategy, STRATEGIES)
        for name in ("max_upload_mb", "max_frames", "max_long_side", "max_objects", "max_sessions"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be at least 1")
        if self.max_duration_s <= 0 or self.max_fps <= 0:
            raise ValueError("max_duration_s and max_fps must be positive")
        if not 1 <= self.max_objects <= 8:
            raise ValueError("max_objects must be between 1 and 8 (the colour palette size)")

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Settings:
        import os

        source = os.environ if env is None else env
        defaults = cls()
        values: dict[str, object] = {}
        for field in dataclasses.fields(cls):
            raw = source.get(ENV_PREFIX + field.name.upper())
            if raw is None or raw == "":
                continue
            current = getattr(defaults, field.name)
            try:
                values[field.name] = _coerce(raw, current)
            except ValueError as exc:
                raise ValueError(
                    f"{ENV_PREFIX}{field.name.upper()}={raw!r} is invalid: {exc}"
                ) from exc
        return cls(**values)  # type: ignore[arg-type]

    def replace(self, **changes: object) -> Settings:
        return dataclasses.replace(self, **changes)  # type: ignore[arg-type]

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024


def _choice(name: str, value: str, allowed: tuple[str, ...]) -> None:
    if value not in allowed:
        raise ValueError(f"{name} must be one of {', '.join(allowed)} (got {value!r})")


def _coerce(raw: str, current: object) -> object:
    if isinstance(current, bool):
        return raw.strip().lower() in {"1", "true", "yes", "on"}
    if isinstance(current, int):
        return int(raw)
    if isinstance(current, float):
        return float(raw)
    if isinstance(current, Path):
        return Path(raw).expanduser()
    return raw

from __future__ import annotations

from pathlib import Path

import pytest

from doodl.config import Settings
from doodl.samples import sample_frames, write_sample_video
from doodl.video import extract_frames, probe_video
from helpers import FRAMES, SIZE


@pytest.fixture(scope="session")
def clip(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A 40-frame, 320x180 synthetic video."""
    path = tmp_path_factory.mktemp("clips") / "sample.mp4"
    write_sample_video(path, FRAMES, SIZE)
    return path


@pytest.fixture(scope="session")
def truth() -> tuple[list, list]:
    """The same clip as arrays, with exact per-frame ground-truth masks."""
    return sample_frames(FRAMES, SIZE)


@pytest.fixture(scope="session")
def frames_dir(clip: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    directory = tmp_path_factory.mktemp("frames")
    extract_frames(
        clip,
        directory,
        probe_video(clip),
        max_long_side=1280,
        max_fps=30,
        max_duration_s=30,
        max_frames=450,
    )
    return directory


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        data_dir=tmp_path / "data",
        checkpoint_dir=tmp_path / "checkpoints",
        max_upload_mb=5,
        backend="classical",
    )

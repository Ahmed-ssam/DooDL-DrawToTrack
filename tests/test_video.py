from __future__ import annotations

import subprocess
from pathlib import Path

import cv2
import numpy as np
import pytest

from doodl.errors import VideoError
from doodl.video import VideoInfo, VideoWriter, extract_frames, ffmpeg_exe, mux_audio, probe_video
from helpers import FRAMES, SIZE

pytestmark = pytest.mark.slow


def test_probe_reads_size_rate_and_duration(clip: Path) -> None:
    info = probe_video(clip)
    assert (info.width, info.height) == SIZE
    assert info.fps == pytest.approx(30, abs=0.5)
    assert info.duration_s == pytest.approx(FRAMES / 30, abs=0.15)
    assert info.has_audio is False


def test_probe_rejects_non_video(tmp_path: Path) -> None:
    junk = tmp_path / "notes.mp4"
    junk.write_bytes(b"this is not a video" * 100)
    with pytest.raises(VideoError):
        probe_video(junk)


def test_extract_writes_numbered_jpegs(clip: Path, tmp_path: Path) -> None:
    result = extract_frames(
        clip,
        tmp_path / "f",
        probe_video(clip),
        max_long_side=1280,
        max_fps=30,
        max_duration_s=30,
        max_frames=450,
    )
    assert result.count == FRAMES
    assert (result.width, result.height) == SIZE
    assert result.trimmed is False
    assert result.path(0).name == "00000.jpg" and result.path(FRAMES - 1).exists()


def test_extract_downscales_to_even_dimensions(clip: Path, tmp_path: Path) -> None:
    result = extract_frames(
        clip,
        tmp_path / "f",
        probe_video(clip),
        max_long_side=161,
        max_fps=30,
        max_duration_s=30,
        max_frames=450,
    )
    assert max(result.width, result.height) <= 161
    assert result.width % 2 == 0 and result.height % 2 == 0
    image = cv2.imread(str(result.path(0)))
    assert image.shape[:2] == (result.height, result.width)


def test_extract_trims_long_clips_and_reports_it(clip: Path, tmp_path: Path) -> None:
    result = extract_frames(
        clip,
        tmp_path / "f",
        probe_video(clip),
        max_long_side=1280,
        max_fps=30,
        max_duration_s=0.5,
        max_frames=450,
    )
    assert result.trimmed is True
    assert 13 <= result.count <= 16


def test_extract_caps_frame_rate(clip: Path, tmp_path: Path) -> None:
    result = extract_frames(
        clip,
        tmp_path / "f",
        probe_video(clip),
        max_long_side=1280,
        max_fps=15,
        max_duration_s=30,
        max_frames=450,
    )
    assert result.fps == 15
    assert abs(result.count - FRAMES / 2) <= 2


def test_extract_fails_cleanly_on_garbage(tmp_path: Path) -> None:
    junk = tmp_path / "x.mp4"
    junk.write_bytes(b"\x00" * 2048)
    info = VideoInfo(width=320, height=180, fps=30.0, duration_s=1.0, has_audio=False)
    with pytest.raises(VideoError):
        extract_frames(
            junk,
            tmp_path / "f",
            info,
            max_long_side=640,
            max_fps=30,
            max_duration_s=30,
            max_frames=450,
        )


def test_writer_produces_a_playable_h264_file(tmp_path: Path) -> None:
    out = tmp_path / "out.mp4"
    with VideoWriter(out, 64, 48, 25.0) as writer:
        for i in range(10):
            writer.write(np.full((48, 64, 3), i * 20, np.uint8))
    info = probe_video(out)
    assert (info.width, info.height) == (64, 48)
    assert info.duration_s == pytest.approx(0.4, abs=0.1)


def test_writer_rejects_wrong_frame_size(tmp_path: Path) -> None:
    writer = VideoWriter(tmp_path / "o.mp4", 64, 48, 25.0)
    try:
        with pytest.raises(VideoError, match="size"):
            writer.write(np.zeros((10, 10, 3), np.uint8))
    finally:
        writer.abort()


def test_writer_aborts_on_error_without_hanging(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError), VideoWriter(tmp_path / "o.mp4", 64, 48, 25.0) as writer:
        writer.write(np.zeros((48, 64, 3), np.uint8))
        raise RuntimeError("boom")


def test_mux_audio_copies_the_source_track(clip: Path, tmp_path: Path) -> None:
    with_audio = tmp_path / "with_audio.mp4"
    subprocess.run(
        [
            ffmpeg_exe(),
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(clip),
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=1.3",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-shortest",
            str(with_audio),
        ],
        check=True,
    )
    assert probe_video(with_audio).has_audio

    silent = tmp_path / "silent.mp4"
    with VideoWriter(silent, *SIZE, 30.0) as writer:
        for _ in range(FRAMES):
            writer.write(np.zeros((SIZE[1], SIZE[0], 3), np.uint8))
    final = tmp_path / "final.mp4"
    mux_audio(silent, with_audio, final)
    assert probe_video(final).has_audio
    assert not silent.exists()

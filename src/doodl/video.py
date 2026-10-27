"""Video probing, frame extraction and encoding (ffmpeg does the heavy lifting)."""

from __future__ import annotations

import logging
import math
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType

import cv2
import numpy as np

from .errors import VideoError

log = logging.getLogger(__name__)

VIDEO_EXTENSIONS = frozenset({".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"})

_DURATION_RE = re.compile(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)")
_FPS_RE = re.compile(r"(\d+(?:\.\d+)?)\s+fps")


@dataclass(frozen=True)
class VideoInfo:
    width: int
    height: int
    fps: float
    duration_s: float
    has_audio: bool


@dataclass(frozen=True)
class FrameSet:
    """The JPEG frames SAM 2 reads: ``00000.jpg``, ``00001.jpg``, ..."""

    directory: Path
    count: int
    width: int
    height: int
    fps: float
    trimmed: bool

    def path(self, index: int) -> Path:
        return self.directory / f"{index:05d}.jpg"


def ffmpeg_exe() -> str:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg

        return str(imageio_ffmpeg.get_ffmpeg_exe())
    except Exception as exc:
        raise VideoError(
            "ffmpeg is not installed. Install it (for example `apt install ffmpeg`) and try again."
        ) from exc


def _ffmpeg_text(path: Path) -> str:
    try:
        proc = subprocess.run(
            [ffmpeg_exe(), "-hide_banner", "-i", str(path)],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise VideoError("Reading this file took too long. Try a different video.") from exc
    return proc.stderr


def probe_video(path: Path) -> VideoInfo:
    """Read size, frame rate, duration and audio presence without decoding the clip."""
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise VideoError("This file could not be read as a video. Try an MP4 (H.264) file.")
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        count = float(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()

    text = _ffmpeg_text(path)
    if not math.isfinite(fps) or fps <= 0:
        match = _FPS_RE.search(text)
        fps = float(match.group(1)) if match else 30.0

    match = _DURATION_RE.search(text)
    if match:
        hours, minutes, seconds = match.groups()
        duration = int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    elif math.isfinite(count) and count > 0:
        duration = count / fps
    else:
        duration = 0.0

    if width <= 0 or height <= 0:
        raise VideoError("This file has no readable video track.")
    return VideoInfo(
        width=width,
        height=height,
        fps=fps,
        duration_s=duration,
        has_audio="Audio:" in text,
    )


def extract_frames(
    source: Path,
    directory: Path,
    info: VideoInfo,
    *,
    max_long_side: int,
    max_fps: float,
    max_duration_s: float,
    max_frames: int,
) -> FrameSet:
    """Decode ``source`` to numbered JPEGs, capped in length, frame rate and size."""
    directory.mkdir(parents=True, exist_ok=True)
    out_fps = min(info.fps, max_fps)
    cap_seconds = min(max_duration_s, max_frames / out_fps)
    trimmed = info.duration_s > cap_seconds + 0.05

    side = int(max_long_side)
    scale = (
        f"scale='trunc(iw*min(1,{side}/max(iw,ih))/2)*2':"
        f"'trunc(ih*min(1,{side}/max(iw,ih))/2)*2':flags=lanczos"
    )
    cmd = [
        ffmpeg_exe(),
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source),
        "-t",
        f"{cap_seconds:.3f}",
        "-frames:v",
        str(max_frames),
        "-an",
        "-vf",
        f"fps={out_fps:.6f},{scale}",
        "-q:v",
        "2",
        "-start_number",
        "0",
        str(directory / "%05d.jpg"),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600, check=False)
    except subprocess.TimeoutExpired as exc:
        raise VideoError("Decoding this video took too long. Try a shorter clip.") from exc
    if proc.returncode != 0:
        log.warning("ffmpeg failed: %s", proc.stderr.strip())
        raise VideoError("This video could not be decoded. Try re-exporting it as MP4 (H.264).")

    frames = sorted(directory.glob("*.jpg"))
    if not frames:
        raise VideoError("No frames could be read from this video.")
    first = cv2.imread(str(frames[0]))
    if first is None:
        raise VideoError("The first frame of this video could not be read.")
    height, width = first.shape[:2]
    return FrameSet(
        directory=directory,
        count=len(frames),
        width=width,
        height=height,
        fps=out_fps,
        trimmed=trimmed,
    )


class VideoWriter:
    """Pipe BGR frames into ffmpeg and get a browser-playable H.264 file."""

    def __init__(self, path: Path, width: int, height: int, fps: float, crf: int = 20) -> None:
        self.path = path
        self.width = width
        self.height = height
        cmd = [
            ffmpeg_exe(),
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{width}x{height}",
            "-r",
            f"{fps:.6f}",
            "-i",
            "-",
            "-vf",
            "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            str(crf),
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(path),
        ]
        self._proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
        )

    def write(self, frame: np.ndarray) -> None:
        if frame.shape[:2] != (self.height, self.width):
            raise VideoError("Frame size changed while writing the video.")
        assert self._proc.stdin is not None
        try:
            self._proc.stdin.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())
        except BrokenPipeError as exc:
            raise VideoError(f"Encoding failed: {self._stderr()}") from exc

    def _stderr(self) -> str:
        assert self._proc.stderr is not None
        return self._proc.stderr.read().decode(errors="replace").strip()

    def close(self) -> None:
        assert self._proc.stdin is not None
        self._proc.stdin.close()
        code = self._proc.wait()
        if code != 0:
            raise VideoError(f"Encoding failed: {self._stderr()}")

    def abort(self) -> None:
        self._proc.kill()
        self._proc.wait()

    def __enter__(self) -> VideoWriter:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if exc_type is not None:
            self.abort()
        else:
            self.close()


def mux_audio(video_only: Path, source: Path, destination: Path) -> None:
    """Copy the original audio track onto a rendered video (stops at the shorter stream)."""
    cmd = [
        ffmpeg_exe(),
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(video_only),
        "-i",
        str(source),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0?",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-b:a",
        "160k",
        "-shortest",
        "-movflags",
        "+faststart",
        str(destination),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300, check=False)
    if proc.returncode != 0:
        log.warning("audio mux failed, keeping silent video: %s", proc.stderr.strip())
        shutil.move(str(video_only), str(destination))
    else:
        video_only.unlink(missing_ok=True)

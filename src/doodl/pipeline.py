"""The tracking job: propagate every object, then render the outputs."""

from __future__ import annotations

import logging
import shutil
import time
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from .backends.base import SegmentationBackend
from .export import frame_annotation, write_annotations, zip_masks
from .jobs import Job
from .render import (
    PALETTE,
    isolate_masks,
    masks_from_label_map,
    overlay_masks,
)
from .sessions import Session
from .video import VideoWriter, mux_audio

log = logging.getLogger(__name__)

TRACK_WEIGHT = 0.75  # share of the progress bar given to tracking (the rest is rendering)


@dataclass(frozen=True)
class Pass:
    start: int
    reverse: bool
    obj_ids: tuple[int, ...]


def plan_passes(first_frame_by_obj: dict[int, int]) -> list[Pass]:
    """One forward pass for everything, plus one backward pass per distinct first frame.

    Each object is tracked in both directions from the earliest frame the user prompted
    it on, so prompting mid-clip still covers the frames before it.
    """
    first_overall = min(first_frame_by_obj.values())
    passes = [Pass(first_overall, False, tuple(sorted(first_frame_by_obj)))]
    for start in sorted({f for f in first_frame_by_obj.values() if f > 0}):
        ids = tuple(sorted(o for o, f in first_frame_by_obj.items() if f == start))
        passes.append(Pass(start, True, ids))
    return passes


def _pass_length(p: Pass, frame_count: int) -> int:
    return p.start + 1 if p.reverse else frame_count - p.start


def run_tracking(
    session: Session, backend: SegmentationBackend, handle: Any, job: Job, version: int
) -> dict[str, Any]:
    """Track all of the session's objects and write overlay, cutout, masks and annotations."""
    assert session.frames is not None and session.video is not None
    frames = session.frames
    shape = (frames.height, frames.width)
    first_by_obj = {oid: obj.first_frame for oid, obj in session.objects.items()}
    obj_ids = sorted(first_by_obj)
    passes = plan_passes(first_by_obj)
    total_track = sum(_pass_length(p, frames.count) for p in passes)

    session.masks.clear()
    backend.reset_stats()
    started = time.perf_counter()
    done = 0
    for p in passes:
        wanted = set(p.obj_ids)
        for frame_idx, masks in backend.propagate(
            handle, start_frame=p.start, reverse=p.reverse, obj_ids=p.obj_ids
        ):
            job.check_cancelled()
            if p.reverse:
                keep = {o: m for o, m in masks.items() if o in wanted and frame_idx < p.start}
            else:
                keep = {o: m for o, m in masks.items() if frame_idx >= first_by_obj[o]}
            if keep:
                session.masks.update(frame_idx, keep, shape)
            done += 1
            job.update(
                TRACK_WEIGHT * done / total_track,
                f"Tracking frame {frame_idx + 1} of {frames.count}",
            )
    track_seconds = time.perf_counter() - started

    files = _render_outputs(session, obj_ids, job)
    stats = backend.stats()
    return {
        "version": version,
        "frames": frames.count,
        "elapsed_s": round(track_seconds, 2),
        "fps": round(frames.count / track_seconds, 2) if track_seconds > 0 else 0.0,
        "peak_gpu_gb": stats.get("peak_gpu_gb"),
        "files": files,
    }


def _render_outputs(session: Session, obj_ids: list[int], job: Job) -> dict[str, str]:
    assert (
        session.frames is not None and session.video is not None and session.source_path is not None
    )
    frames = session.frames
    shape = (frames.height, frames.width)
    results = session.results_dir
    overlay_tmp = results / "overlay.silent.mp4"
    cutout_tmp = results / "cutout.silent.mp4"
    annotations: list[dict[str, Any]] = []

    with (
        VideoWriter(overlay_tmp, frames.width, frames.height, frames.fps) as overlay_out,
        VideoWriter(cutout_tmp, frames.width, frames.height, frames.fps) as cutout_out,
    ):
        for index in range(frames.count):
            job.check_cancelled()
            frame = cv2.imread(str(frames.path(index)))
            if frame is None:
                raise RuntimeError(f"Frame {index} is missing")
            label_map = session.masks.read(index)
            if label_map is None:
                label_map = np.zeros(shape, np.uint8)
            masks = masks_from_label_map(label_map, obj_ids)
            overlay_out.write(overlay_masks(frame, masks))
            cutout_out.write(isolate_masks(frame, masks))
            annotations.append(frame_annotation(index, frames.fps, label_map, obj_ids))
            job.update(
                TRACK_WEIGHT + (1 - TRACK_WEIGHT) * (index + 1) / frames.count, "Rendering video"
            )

    for temp, name in ((overlay_tmp, "overlay.mp4"), (cutout_tmp, "cutout.mp4")):
        destination = results / name
        if session.video.has_audio:
            mux_audio(temp, session.source_path, destination)
        else:
            shutil.move(str(temp), str(destination))

    objects = [
        {"id": oid, "name": session.objects[oid].name, "color": PALETTE[(oid - 1) % len(PALETTE)]}
        for oid in obj_ids
    ]
    write_annotations(
        results / "annotations.json",
        width=frames.width,
        height=frames.height,
        fps=frames.fps,
        frame_count=frames.count,
        objects=objects,
        frames=annotations,
    )
    zip_masks(session.masks, results / "masks.zip", frames.count)
    return {name: name for name in ("overlay.mp4", "cutout.mp4", "masks.zip", "annotations.json")}

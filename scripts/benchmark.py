"""Benchmark a backend on the synthetic clip (exact ground truth): quality, speed, memory.

    python scripts/benchmark.py --backend classical
    python scripts/benchmark.py --backend sam2 --model tiny --strategy box_points
    python scripts/benchmark.py --backend sam2 --model small --strategy mask --markdown

The clip has a ball and a square that cross each other (occlusion) over a panning
background, so J&F here is a sanity check of the whole pipeline, not a replacement for
DAVIS/YouTube-VOS numbers.
"""

from __future__ import annotations

import argparse
import json
import math
import tempfile
import time
from pathlib import Path

import numpy as np

from doodl.backends import create_backend
from doodl.config import Settings
from doodl.metrics import j_and_f
from doodl.prompts import ObjectPrompt
from doodl.samples import BALL_ID, SQUARE_ID, sample_frames, write_sample_video
from doodl.video import extract_frames, probe_video


def loose_circle(
    mask: np.ndarray, grow: float = 1.4, points: int = 36
) -> list[tuple[float, float]]:
    ys, xs = np.nonzero(mask)
    cx, cy = float(xs.mean()), float(ys.mean())
    radius = math.sqrt(mask.sum() / math.pi) * grow
    return [
        (
            cx + radius * math.cos(2 * math.pi * i / points),
            cy + radius * math.sin(2 * math.pi * i / points),
        )
        for i in range(points)
    ]


def run(args: argparse.Namespace) -> dict:
    size = (args.width, args.height)
    _, truths = sample_frames(args.frames, size)
    settings = Settings(
        backend=args.backend,
        sam2_model=args.model,
        prompt_strategy=args.strategy,
        precision=args.precision,
        device=args.device,
    )
    backend = create_backend(settings)
    with tempfile.TemporaryDirectory() as tmp:
        clip = Path(tmp) / "clip.mp4"
        write_sample_video(clip, args.frames, size)
        extract_frames(
            clip,
            Path(tmp) / "f",
            probe_video(clip),
            max_long_side=1280,
            max_fps=30,
            max_duration_s=60,
            max_frames=1000,
        )
        handle = backend.open_session(Path(tmp) / "f", args.frames)
        for obj_id in (BALL_ID, SQUARE_ID):
            backend.set_prompt(
                handle, 0, obj_id, ObjectPrompt(polygon=loose_circle(truths[0][obj_id]))
            )
        backend.reset_stats()
        start = time.perf_counter()
        preds: dict[int, dict[int, np.ndarray]] = {}
        for idx, masks in backend.propagate(
            handle, start_frame=0, reverse=False, obj_ids=[BALL_ID, SQUARE_ID]
        ):
            preds[idx] = masks
        elapsed = time.perf_counter() - start
        stats = backend.stats()
        backend.close_session(handle)

    empty = np.zeros((args.height, args.width), bool)
    per_object = {}
    for name, obj_id in (("ball", BALL_ID), ("square", SQUARE_ID)):
        p = [preds.get(i, {}).get(obj_id, empty) for i in range(args.frames)]
        t = [truths[i][obj_id] for i in range(args.frames)]
        per_object[name] = j_and_f(p, t)
    return {
        "backend": backend.info.label,
        "device": backend.info.device,
        "strategy": args.strategy,
        "frames": args.frames,
        "resolution": f"{args.width}x{args.height}",
        "fps": round(args.frames / elapsed, 2),
        "peak_gpu_gb": stats.get("peak_gpu_gb"),
        "J&F": round(float(np.mean([v["J&F"] for v in per_object.values()])), 3),
        "per_object": per_object,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--backend", default="classical", choices=["classical", "sam2"])
    ap.add_argument("--model", default="tiny")
    ap.add_argument("--strategy", default="box_points")
    ap.add_argument("--precision", default="auto")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--frames", type=int, default=72)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=360)
    ap.add_argument("--markdown", action="store_true", help="print a README table row")
    args = ap.parse_args()
    result = run(args)
    if args.markdown:
        print("| Backend | Device | Strategy | Resolution | FPS | Peak VRAM (GB) | J&F |")
        print("|---|---|---|---|---|---|---|")
        print(
            f"| {result['backend']} | {result['device']} | {result['strategy']} | "
            f"{result['resolution']} | {result['fps']} | "
            f"{result['peak_gpu_gb'] or 'n/a'} | {result['J&F']} |"
        )
    else:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

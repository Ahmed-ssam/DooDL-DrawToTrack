"""Command line: ``doodl serve``, ``doodl check`` and ``doodl sample``."""

from __future__ import annotations

import argparse
import logging
import os
import platform
import shutil
import sys
import threading
import webbrowser
from pathlib import Path

from . import __version__
from .config import BACKENDS, PRECISIONS, SAM2_MODELS, STRATEGIES, Settings


def _settings_from_args(args: argparse.Namespace) -> Settings:
    settings = Settings.from_env()
    overrides = {
        "backend": args.backend,
        "sam2_model": args.model,
        "device": args.device,
        "precision": args.precision,
        "prompt_strategy": args.strategy,
        "access_token": args.token,
        "data_dir": Path(args.data_dir).expanduser() if args.data_dir else None,
    }
    return settings.replace(**{k: v for k, v in overrides.items() if v is not None})


def _serve(args: argparse.Namespace) -> int:
    import uvicorn

    from .api import create_app

    settings = _settings_from_args(args)
    if args.host not in ("127.0.0.1", "localhost") and not settings.access_token:
        print(
            "Warning: listening on a public address without --token. Anyone who can reach this "
            "port can use your GPU. Set --token (or DOODL_ACCESS_TOKEN) to require one.",
            file=sys.stderr,
        )
    app = create_app(settings)
    url = f"http://{'127.0.0.1' if args.host in ('0.0.0.0', '::') else args.host}:{args.port}"
    print(f"\nDooDL is running at {url}  (press Ctrl+C to stop)\n", flush=True)
    if args.open:
        threading.Timer(1.5, webbrowser.open, args=(url,)).start()
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)
    return 0


def _check(_: argparse.Namespace) -> int:
    from .video import ffmpeg_exe

    print(f"DooDL {__version__} on Python {platform.python_version()} ({platform.system()})")
    try:
        print(f"ffmpeg:  {ffmpeg_exe()}")
    except Exception as exc:
        print(f"ffmpeg:  MISSING ({exc})")
    try:
        import cv2

        print(f"opencv:  {cv2.__version__}")
    except ImportError:
        print("opencv:  MISSING")
    try:
        import torch

        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no CUDA GPU"
        print(f"torch:   {torch.__version__} ({gpu})")
    except ImportError:
        print("torch:   not installed (the CPU baseline still works)")
    try:
        import sam2  # noqa: F401

        print("sam2:    installed")
    except ImportError:
        print(
            "sam2:    not installed (pip install git+https://github.com/facebookresearch/sam2.git)"
        )
    settings = Settings.from_env()
    print(f"backend: {settings.backend} (model {settings.sam2_model}), data in {settings.data_dir}")
    probe = settings.data_dir if settings.data_dir.exists() else settings.data_dir.parent
    free_gb = shutil.disk_usage(probe if probe.exists() else Path("/")).free / 1e9
    print(f"disk:    {free_gb:.1f} GB free")
    return 0


def _sample(args: argparse.Namespace) -> int:
    from .samples import write_sample_video

    write_sample_video(Path(args.output), n_frames=args.frames)
    print(f"Wrote {args.output}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="doodl", description="Outline an object once, track it through the video."
    )
    parser.add_argument("--version", action="version", version=f"doodl {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="start the web app")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--backend", choices=BACKENDS)
    serve.add_argument("--model", choices=SAM2_MODELS, help="SAM 2.1 size")
    serve.add_argument("--device", help="auto, cuda, cpu or mps")
    serve.add_argument("--precision", choices=PRECISIONS)
    serve.add_argument("--strategy", choices=STRATEGIES, help="how outlines become model prompts")
    serve.add_argument(
        "--token", default=os.environ.get("DOODL_ACCESS_TOKEN"), help="require this access token"
    )
    serve.add_argument("--open", action="store_true", help="open the app in your browser")
    serve.add_argument("--data-dir")
    serve.add_argument("--log-level", default=os.environ.get("DOODL_LOG_LEVEL", "info"))
    serve.set_defaults(run=_serve)

    check = sub.add_parser("check", help="report what this machine can run")
    check.set_defaults(run=_check)

    sample = sub.add_parser("sample", help="write the synthetic demo clip")
    sample.add_argument("output")
    sample.add_argument("--frames", type=int, default=72)
    sample.set_defaults(run=_sample)
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    args = build_parser().parse_args(argv)
    try:
        return int(args.run(args))
    except (ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

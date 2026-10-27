"""FastAPI application: JSON API under /api, the web UI at /."""

from __future__ import annotations

import base64
import logging
import secrets
import threading
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import cv2
from fastapi import FastAPI, Query, Request, Response
from fastapi.concurrency import run_in_threadpool
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from . import __version__
from .backends import create_backend
from .backends.base import SegmentationBackend
from .config import Settings
from .errors import DooDLError, LimitError, NotFoundError, PromptError
from .jobs import Job
from .render import PALETTE, masks_from_label_map, masks_to_overlay_png
from .schemas import (
    ClicksIn,
    HealthOut,
    JobOut,
    LimitsOut,
    ObjectCreate,
    ObjectOut,
    PreviewOut,
    ResultOut,
    SessionOut,
)
from .service import DooDLService, Preview
from .sessions import DOWNLOADS, Session
from .video import VIDEO_EXTENSIONS

log = logging.getLogger(__name__)
STATIC_DIR = Path(__file__).parent / "static"
SWEEP_INTERVAL_S = 60.0


def _object_out(obj_id: int, session: Session) -> ObjectOut:
    obj = session.objects[obj_id]
    return ObjectOut(
        id=obj.obj_id, name=obj.name, color=obj.color, prompt_frames=sorted(obj.records)
    )


def _session_out(session: Session) -> SessionOut:
    assert session.video is not None and session.frames is not None
    result = None
    if session.result is not None:
        r = session.result
        result = ResultOut(
            version=r.version,
            frames=r.frames,
            elapsed_s=r.elapsed_s,
            fps=r.fps,
            peak_gpu_gb=r.peak_gpu_gb,
            stale=session.result_stale,
            downloads={name: f"/api/sessions/{session.id}/downloads/{name}" for name in r.files},
        )
    return SessionOut(
        id=session.id,
        width=session.frames.width,
        height=session.frames.height,
        fps=session.frames.fps,
        frame_count=session.frames.count,
        duration_s=round(session.frames.count / session.frames.fps, 2),
        trimmed=session.frames.trimmed,
        has_audio=session.video.has_audio,
        status=session.status.value,
        objects=[_object_out(oid, session) for oid in sorted(session.objects)],
        result=result,
        job_id=session.job_id,
    )


def _job_out(job: Job) -> JobOut:
    return JobOut(
        id=job.id,
        session_id=job.session_id,
        status=job.status.value,
        progress=round(job.progress, 4),
        message=job.message,
        error=job.error,
        elapsed_s=round(job.elapsed_s, 2),
    )


def _preview_out(preview: Preview, session: Session) -> PreviewOut:
    return PreviewOut(
        object=_object_out(preview.object.obj_id, session),
        frame_idx=preview.frame_idx,
        png_b64=base64.b64encode(preview.png).decode("ascii"),
        coverage=round(preview.coverage, 4),
    )


def create_app(
    settings: Settings | None = None,
    backend: SegmentationBackend | None = None,
) -> FastAPI:
    settings = settings or Settings.from_env()
    backend = backend or create_backend(settings)
    service = DooDLService(settings, backend)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        stop = threading.Event()

        def sweeper() -> None:
            while not stop.wait(SWEEP_INTERVAL_S):
                try:
                    service.sweep()
                except Exception:
                    log.exception("session sweep failed")

        thread = threading.Thread(target=sweeper, name="doodl-sweeper", daemon=True)
        thread.start()
        try:
            yield
        finally:
            stop.set()
            service.close()

    app = FastAPI(
        title="DooDL",
        version=__version__,
        description="Outline an object once, track it through the whole video.",
        lifespan=lifespan,
    )
    app.state.service = service
    app.state.settings = settings

    # ------------------------------------------------------------ middleware

    @app.middleware("http")
    async def guard(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        path = request.url.path
        token = settings.access_token
        if token and path.startswith("/api/") and path != "/api/health":
            supplied = (
                request.headers.get("x-doodl-token") or request.query_params.get("token") or ""
            )
            if not secrets.compare_digest(supplied.encode(), token.encode()):
                return JSONResponse(
                    {
                        "error": {
                            "code": "unauthorized",
                            "message": "This link needs its access token.",
                        }
                    },
                    status_code=401,
                )
        response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        if path.startswith("/api/") and "Cache-Control" not in response.headers:
            response.headers["Cache-Control"] = "no-store"
        return response

    @app.exception_handler(DooDLError)
    async def doodl_error(_: Request, exc: DooDLError) -> JSONResponse:
        return JSONResponse(
            {"error": {"code": exc.code, "message": str(exc)}}, status_code=exc.status_code
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        first = exc.errors()[0] if exc.errors() else {}
        where = ".".join(str(p) for p in first.get("loc", [])[1:])
        message = (
            f"Invalid request ({where}): {first.get('msg', 'check the values')}"
            if where
            else "Invalid request."
        )
        return JSONResponse(
            {"error": {"code": "invalid_request", "message": message}}, status_code=422
        )

    # ----------------------------------------------------------------- routes

    @app.get("/api/health", response_model=HealthOut)
    def health() -> HealthOut:
        return HealthOut(
            status="ok",
            version=__version__,
            backend=backend.info.name,
            backend_label=backend.info.label,
            device=backend.info.device,
            auth_required=bool(settings.access_token),
            limits=LimitsOut(
                max_upload_mb=settings.max_upload_mb,
                max_duration_s=settings.max_duration_s,
                max_frames=settings.max_frames,
                max_objects=settings.max_objects,
                extensions=sorted(VIDEO_EXTENSIONS),
            ),
            palette=list(PALETTE),
        )

    @app.post("/api/sessions", response_model=SessionOut, status_code=201)
    async def upload(
        request: Request, filename: Annotated[str, Query(min_length=1, max_length=255)]
    ) -> SessionOut:
        """Upload a video as the raw request body (``curl --data-binary @clip.mp4``)."""
        extension = Path(filename).suffix.lower()
        if extension not in VIDEO_EXTENSIONS:
            raise PromptError(
                "Unsupported file type. Use "
                + ", ".join(sorted(e.lstrip(".").upper() for e in VIDEO_EXTENSIONS))
                + "."
            )
        declared = request.headers.get("content-length")
        if declared and declared.isdigit() and int(declared) > settings.max_upload_bytes:
            raise LimitError(f"This file is larger than the {settings.max_upload_mb} MB limit.")

        session = await run_in_threadpool(service.begin_session)
        try:
            source = session.dir / f"source{extension}"
            written = 0
            with source.open("wb") as out:
                async for chunk in request.stream():
                    written += len(chunk)
                    if written > settings.max_upload_bytes:
                        raise LimitError(
                            f"This file is larger than the {settings.max_upload_mb} MB limit."
                        )
                    out.write(chunk)
            if written == 0:
                raise PromptError("The upload was empty.")
            await run_in_threadpool(service.finish_ingest, session, source)
        except BaseException:
            await run_in_threadpool(service.discard, session)
            raise
        return _session_out(session)

    @app.post("/api/samples/synthetic", response_model=SessionOut, status_code=201)
    async def sample() -> SessionOut:
        session = await run_in_threadpool(service.ingest_sample)
        return _session_out(session)

    @app.get("/api/sessions/{session_id}", response_model=SessionOut)
    def get_session(session_id: str) -> SessionOut:
        return _session_out(service.store.get(session_id))

    @app.delete("/api/sessions/{session_id}", status_code=204)
    def delete_session(session_id: str) -> Response:
        service.discard(service.store.get(session_id))
        return Response(status_code=204)

    @app.get("/api/sessions/{session_id}/frames/{index}")
    def frame(
        session_id: str, index: int, w: Annotated[int | None, Query(ge=32, le=640)] = None
    ) -> FileResponse:
        session = service.store.get(session_id)
        if not 0 <= index < session.frame_count:
            raise NotFoundError("That frame does not exist.")
        path = session.frames_dir / f"{index:05d}.jpg"
        headers = {"Cache-Control": "private, max-age=3600"}
        if w is None:
            return FileResponse(path, media_type="image/jpeg", headers=headers)
        thumb = session.thumbs_dir / f"{w}_{index:05d}.jpg"
        if not thumb.exists():
            image = cv2.imread(str(path))
            if image is None:
                raise NotFoundError("That frame could not be read.")
            height = max(1, round(image.shape[0] * w / image.shape[1]))
            small = cv2.resize(image, (w, height), interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(thumb), small, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return FileResponse(thumb, media_type="image/jpeg", headers=headers)

    @app.post("/api/sessions/{session_id}/objects", response_model=PreviewOut, status_code=201)
    def add_object(session_id: str, body: ObjectCreate) -> PreviewOut:
        session = service.store.get(session_id)
        preview = service.add_object(session, body.frame_idx, [(x, y) for x, y in body.polygon])
        return _preview_out(preview, session)

    @app.post("/api/sessions/{session_id}/objects/{object_id}/clicks", response_model=PreviewOut)
    def refine(session_id: str, object_id: int, body: ClicksIn) -> PreviewOut:
        session = service.store.get(session_id)
        if len(body.points) != len(body.labels):
            raise PromptError("Every point needs a label.")
        preview = service.refine_object(
            session,
            object_id,
            body.frame_idx,
            [(x, y) for x, y in body.points],
            body.labels,
            body.undo,
        )
        return _preview_out(preview, session)

    @app.delete("/api/sessions/{session_id}/objects/{object_id}", status_code=204)
    def delete_object(session_id: str, object_id: int) -> Response:
        service.remove_object(service.store.get(session_id), object_id)
        return Response(status_code=204)

    @app.post("/api/sessions/{session_id}/track", response_model=JobOut, status_code=202)
    def track(session_id: str) -> JobOut:
        return _job_out(service.start_tracking(service.store.get(session_id)))

    @app.get("/api/jobs/{job_id}", response_model=JobOut)
    def get_job(job_id: str) -> JobOut:
        return _job_out(service.job(job_id))

    @app.post("/api/jobs/{job_id}/cancel", response_model=JobOut)
    def cancel_job(job_id: str) -> JobOut:
        return _job_out(service.cancel(job_id))

    @app.get("/api/sessions/{session_id}/overlay/{index}.png")
    def overlay(session_id: str, index: int) -> Response:
        """Tinted masks for one frame of the finished track (transparent PNG)."""
        session = service.store.get(session_id)
        label_map = session.masks.read(index) if 0 <= index < session.frame_count else None
        if session.result is None or label_map is None:
            raise NotFoundError("No tracking result for this frame yet.")
        masks = masks_from_label_map(label_map, sorted(session.objects))
        png = masks_to_overlay_png({k: v for k, v in masks.items() if v.any()}, label_map.shape)
        return Response(
            png, media_type="image/png", headers={"Cache-Control": "private, max-age=3600"}
        )

    @app.get("/api/sessions/{session_id}/downloads/{name}")
    def download(session_id: str, name: str, save: bool = False) -> FileResponse:
        session = service.store.get(session_id)
        if name not in DOWNLOADS or session.result is None:
            raise NotFoundError("That file is not available yet. Track the video first.")
        path = session.results_dir / name
        if not path.exists():
            raise NotFoundError("That file is not available yet. Track the video first.")
        media = {
            ".mp4": "video/mp4",
            ".zip": "application/zip",
            ".json": "application/json",
        }[path.suffix]
        return FileResponse(
            path,
            media_type=media,
            filename=f"doodl-{name}",
            content_disposition_type="attachment" if save else "inline",
            headers={"Cache-Control": "private, max-age=3600"},
        )

    # The UI is plain files: no build step. Mounted last so /api/* wins.
    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="ui")
    return app


__all__ = ["create_app"]

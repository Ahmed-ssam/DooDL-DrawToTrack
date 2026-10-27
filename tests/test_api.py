from __future__ import annotations

import base64
import time
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from doodl import __version__
from doodl.api import create_app
from doodl.backends import ClassicalBackend
from doodl.config import Settings
from doodl.samples import BALL_ID, SQUARE_ID
from helpers import SIZE, circle_polygon_unit

pytestmark = pytest.mark.slow


def make_client(settings: Settings) -> TestClient:
    return TestClient(create_app(settings, ClassicalBackend()))


@pytest.fixture
def client(settings: Settings) -> Iterator[TestClient]:
    with make_client(settings) as c:  # the context manager runs startup and shutdown
        yield c


def upload(client: TestClient, clip: Path, name: str = "clip.mp4", **kwargs):
    return client.post(
        "/api/sessions", params={"filename": name}, content=clip.read_bytes(), **kwargs
    )


def wait_for(client: TestClient, job_id: str, timeout: float = 120) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] in ("done", "failed", "cancelled"):
            return job
        time.sleep(0.1)
    raise AssertionError("job did not finish")


def error_of(response) -> dict:
    assert "error" in response.json(), response.text
    return response.json()["error"]


class TestBasics:
    def test_health_reports_backend_limits_and_palette(self, client: TestClient) -> None:
        body = client.get("/api/health").json()
        assert body["status"] == "ok" and body["version"] == __version__
        assert (
            body["backend"] == "classical"
            and body["device"] == "cpu"
            and body["auth_required"] is False
        )
        assert body["limits"]["max_upload_mb"] == 5 and ".mp4" in body["limits"]["extensions"]
        assert len(body["palette"]) == 8

    def test_ui_is_served_with_security_headers(self, client: TestClient) -> None:
        page = client.get("/")
        assert page.status_code == 200 and "text/html" in page.headers["content-type"]
        assert "app.js" in page.text and page.headers["x-content-type-options"] == "nosniff"
        assert (
            client.get("/app.js").status_code == 200
            and client.get("/styles.css").status_code == 200
        )

    def test_api_responses_are_not_cached(self, client: TestClient) -> None:
        assert client.get("/api/health").headers["cache-control"] == "no-store"


class TestUploadValidation:
    def test_rejects_unsupported_extension(self, client: TestClient, clip: Path) -> None:
        response = upload(client, clip, "movie.exe")
        assert (
            response.status_code == 422 and "Unsupported file type" in error_of(response)["message"]
        )

    def test_rejects_empty_upload(self, client: TestClient) -> None:
        response = client.post("/api/sessions", params={"filename": "a.mp4"}, content=b"")
        assert response.status_code == 422 and "empty" in error_of(response)["message"]

    def test_rejects_declared_oversize_before_reading(self, client: TestClient) -> None:
        response = client.post(
            "/api/sessions", params={"filename": "a.mp4"}, content=b"0" * (5 * 1024 * 1024 + 1)
        )
        assert response.status_code == 413 and "5 MB" in error_of(response)["message"]
        assert len(client.app.state.service.store) == 0  # type: ignore[attr-defined]

    def test_rejects_oversize_chunked_upload_mid_stream(self, client: TestClient) -> None:
        def chunks() -> Iterator[bytes]:
            for _ in range(7):
                yield b"0" * (1024 * 1024)

        response = client.post("/api/sessions", params={"filename": "a.mp4"}, content=chunks())
        assert response.status_code == 413
        assert len(client.app.state.service.store) == 0  # type: ignore[attr-defined]

    def test_rejects_garbage_and_leaves_nothing_behind(self, client: TestClient) -> None:
        response = client.post(
            "/api/sessions", params={"filename": "fake.mp4"}, content=b"not a video" * 500
        )
        assert response.status_code == 422
        assert len(client.app.state.service.store) == 0  # type: ignore[attr-defined]

    def test_filename_is_required(self, client: TestClient, clip: Path) -> None:
        assert client.post("/api/sessions", content=clip.read_bytes()).status_code == 422

    def test_path_traversal_in_filename_is_harmless(self, client: TestClient, clip: Path) -> None:
        response = upload(client, clip, "../../etc/evil.mp4")
        assert response.status_code == 201
        session = client.app.state.service.store.get(response.json()["id"])  # type: ignore[attr-defined]
        assert session.source_path.parent == session.dir


class TestFullFlow:
    def test_upload_outline_refine_track_download(
        self, client: TestClient, clip: Path, truth
    ) -> None:
        created = upload(client, clip)
        assert created.status_code == 201
        session = created.json()
        sid = session["id"]
        assert (session["width"], session["height"]) == SIZE and session["frame_count"] == 40
        assert (
            session["status"] == "ready"
            and session["objects"] == []
            and session["trimmed"] is False
        )

        frame = client.get(f"/api/sessions/{sid}/frames/0")
        assert frame.status_code == 200 and frame.headers["content-type"] == "image/jpeg"
        thumb = client.get(f"/api/sessions/{sid}/frames/0", params={"w": 64})
        assert cv2.imdecode(np.frombuffer(thumb.content, np.uint8), 1).shape[1] == 64
        assert client.get(f"/api/sessions/{sid}/frames/999").status_code == 404

        polygon = [[x, y] for x, y in circle_polygon_unit(truth[1][0][BALL_ID])]
        added = client.post(
            f"/api/sessions/{sid}/objects", json={"frame_idx": 0, "polygon": polygon}
        )
        assert added.status_code == 201
        body = added.json()
        assert body["object"] == {
            "id": 1,
            "name": "Object 1",
            "color": body["object"]["color"],
            "prompt_frames": [0],
        }
        overlay = cv2.imdecode(
            np.frombuffer(base64.b64decode(body["png_b64"]), np.uint8), cv2.IMREAD_UNCHANGED
        )
        assert overlay.shape == (SIZE[1], SIZE[0], 4) and 0.02 < body["coverage"] < 0.2

        second = client.post(
            f"/api/sessions/{sid}/objects",
            json={"frame_idx": 0, "polygon": circle_polygon_unit(truth[1][0][SQUARE_ID])},
        )
        assert second.json()["object"]["id"] == 2

        refined = client.post(
            f"/api/sessions/{sid}/objects/1/clicks",
            json={"frame_idx": 0, "points": [[0.02, 0.9]], "labels": [0]},
        )
        assert refined.status_code == 200 and refined.json()["object"]["prompt_frames"] == [0]

        job = client.post(f"/api/sessions/{sid}/track")
        assert job.status_code == 202
        done = wait_for(client, job.json()["id"])
        assert done["status"] == "done" and done["progress"] == 1.0, done

        final = client.get(f"/api/sessions/{sid}").json()
        result = final["result"]
        assert final["status"] == "done" and result["frames"] == 40 and result["fps"] > 0
        assert set(result["downloads"]) == {
            "overlay.mp4",
            "cutout.mp4",
            "masks.zip",
            "annotations.json",
        }

        tinted = client.get(f"/api/sessions/{sid}/overlay/20.png")
        assert tinted.status_code == 200 and tinted.headers["content-type"] == "image/png"

        types = {
            "overlay.mp4": "video/mp4",
            "cutout.mp4": "video/mp4",
            "masks.zip": "application/zip",
            "annotations.json": "application/json",
        }
        for name, media in types.items():
            response = client.get(result["downloads"][name])
            assert response.status_code == 200 and response.headers["content-type"].startswith(
                media
            ), name
            assert "inline" in response.headers["content-disposition"]
        saved = client.get(result["downloads"]["masks.zip"], params={"save": "true"})
        assert "attachment" in saved.headers["content-disposition"]
        ranged = client.get(result["downloads"]["overlay.mp4"], headers={"Range": "bytes=0-99"})
        assert ranged.status_code == 206 and len(ranged.content) == 100

        assert client.delete(f"/api/sessions/{sid}/objects/2").status_code == 204
        assert client.get(f"/api/sessions/{sid}").json()["result"]["stale"] is True
        assert client.delete(f"/api/sessions/{sid}").status_code == 204
        assert client.get(f"/api/sessions/{sid}").status_code == 404

    def test_sample_endpoint(self, client: TestClient) -> None:
        response = client.post("/api/samples/synthetic")
        assert response.status_code == 201 and response.json()["frame_count"] == 72

    def test_downloads_and_overlay_before_tracking_are_404(
        self, client: TestClient, clip: Path
    ) -> None:
        sid = upload(client, clip).json()["id"]
        assert client.get(f"/api/sessions/{sid}/downloads/overlay.mp4").status_code == 404
        assert client.get(f"/api/sessions/{sid}/overlay/0.png").status_code == 404
        assert client.get(f"/api/sessions/{sid}/downloads/../../etc/passwd").status_code == 404
        assert client.get(f"/api/sessions/{sid}/downloads/other.txt").status_code == 404


class TestErrors:
    @pytest.fixture
    def sid(self, client: TestClient, clip: Path) -> str:
        return upload(client, clip).json()["id"]

    def test_unknown_session_job_and_object(self, client: TestClient, sid: str) -> None:
        assert client.get("/api/sessions/nope").status_code == 404
        assert "expired" in error_of(client.get("/api/sessions/nope"))["message"]
        assert client.get("/api/jobs/nope").status_code == 404
        assert client.post("/api/jobs/nope/cancel").status_code == 404
        assert client.delete(f"/api/sessions/{sid}/objects/9").status_code == 404

    def test_tracking_without_objects(self, client: TestClient, sid: str) -> None:
        response = client.post(f"/api/sessions/{sid}/track")
        assert (
            response.status_code == 422 and "Outline at least one" in error_of(response)["message"]
        )

    @pytest.mark.parametrize(
        "body",
        [
            {"frame_idx": 0, "polygon": [[0.1, 0.1], [0.2, 0.2]]},  # too few points
            {"frame_idx": 0, "polygon": [[0.1, 0.1], [1.5, 0.2], [0.3, 0.3]]},  # outside 0..1
            {"frame_idx": -1, "polygon": [[0.1, 0.1], [0.5, 0.1], [0.3, 0.5]]},
            {"frame_idx": 0},
            {"frame_idx": 0, "polygon": "nope"},
        ],
    )
    def test_invalid_outline_bodies_are_422_with_a_message(
        self, client: TestClient, sid: str, body: dict
    ) -> None:
        response = client.post(f"/api/sessions/{sid}/objects", json=body)
        assert response.status_code == 422 and error_of(response)["code"] == "invalid_request"

    def test_outline_on_missing_frame(self, client: TestClient, sid: str) -> None:
        response = client.post(
            f"/api/sessions/{sid}/objects",
            json={"frame_idx": 400, "polygon": [[0.1, 0.1], [0.5, 0.1], [0.3, 0.5]]},
        )
        assert response.status_code == 422 and "outside this video" in error_of(response)["message"]

    def test_clicks_validation(self, client: TestClient, sid: str) -> None:
        bad_labels = client.post(
            f"/api/sessions/{sid}/objects/1/clicks",
            json={"frame_idx": 0, "points": [[0.5, 0.5]], "labels": [3]},
        )
        assert bad_labels.status_code == 422
        mismatch = client.post(
            f"/api/sessions/{sid}/objects/1/clicks",
            json={"frame_idx": 0, "points": [[0.5, 0.5]], "labels": []},
        )
        assert mismatch.status_code == 422 and "label" in error_of(mismatch)["message"]


class TestAccessToken:
    @pytest.fixture
    def secured(self, settings: Settings) -> Iterator[TestClient]:
        with make_client(settings.replace(access_token="s3cret")) as c:
            yield c

    def test_health_is_open_and_says_a_token_is_needed(self, secured: TestClient) -> None:
        assert secured.get("/api/health").json()["auth_required"] is True

    def test_api_needs_the_token(self, secured: TestClient, clip: Path) -> None:
        assert upload(secured, clip).status_code == 401
        assert error_of(secured.post("/api/samples/synthetic"))["code"] == "unauthorized"
        assert secured.get("/api/sessions/x", headers={"X-DooDL-Token": "wrong"}).status_code == 401

    def test_header_and_query_token_both_work(self, secured: TestClient, clip: Path) -> None:
        by_header = secured.post(
            "/api/sessions",
            params={"filename": "c.mp4"},
            content=clip.read_bytes(),
            headers={"X-DooDL-Token": "s3cret"},
        )
        assert by_header.status_code == 201
        sid = by_header.json()["id"]
        assert secured.get(f"/api/sessions/{sid}/frames/0").status_code == 401
        assert (
            secured.get(f"/api/sessions/{sid}/frames/0", params={"token": "s3cret"}).status_code
            == 200
        )

    def test_the_ui_itself_stays_reachable(self, secured: TestClient) -> None:
        assert secured.get("/").status_code == 200

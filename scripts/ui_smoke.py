"""Drive the real UI in headless Chromium and save screenshots.

Not part of the unit tests (it needs a browser). Run it manually after UI changes:

    pip install playwright && playwright install chromium
    python scripts/ui_smoke.py --out docs/img

It starts a server with the CPU baseline, loads the synthetic sample clip, draws a loose
outline around the ball with the mouse, tracks it, and checks that the result appears.
"""

from __future__ import annotations

import argparse
import math
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path


def wait_for_server(base: str, timeout: float = 30) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(f"{base}/api/health", timeout=2)
            return
        except Exception:
            time.sleep(0.3)
    raise RuntimeError("server did not start")


def draw_outline(page, cx: float, cy: float, rx: float, ry: float) -> None:
    box = page.locator("#ink").bounding_box()
    assert box is not None
    page.mouse.move(box["x"] + box["width"] * (cx + rx), box["y"] + box["height"] * cy)
    page.mouse.down()
    steps = 48
    for i in range(1, steps + 1):
        angle = 2 * math.pi * i / steps
        wobble = 1 + 0.06 * math.sin(5 * angle)
        page.mouse.move(
            box["x"] + box["width"] * (cx + rx * wobble * math.cos(angle)),
            box["y"] + box["height"] * (cy + ry * wobble * math.sin(angle)),
        )
    page.mouse.up()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="docs/img")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--dark", action="store_true", help="use the dark colour scheme")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    base = f"http://127.0.0.1:{args.port}"

    from playwright.sync_api import sync_playwright

    data_dir = tempfile.mkdtemp(prefix="doodl-ui-")
    server = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "doodl.cli",
            "serve",
            "--backend",
            "classical",
            "--port",
            str(args.port),
            "--data-dir",
            data_dir,
            "--log-level",
            "warning",
        ],
    )
    problems: list[str] = []
    try:
        wait_for_server(base)
        with sync_playwright() as p:
            browser = p.chromium.launch()
            scheme = "dark" if args.dark else "light"
            page = browser.new_page(viewport={"width": 1280, "height": 860}, color_scheme=scheme)
            page.on(
                "console", lambda m: m.type == "error" and problems.append(f"console: {m.text}")
            )
            page.on("pageerror", lambda e: problems.append(f"pageerror: {e}"))

            page.goto(base)
            page.wait_for_timeout(2600)  # let the start-screen animation finish
            page.screenshot(path=str(out / "ui-empty.png"))

            page.click("#sample")
            page.wait_for_selector("#viewport:not([hidden])")
            page.wait_for_function(
                "(() => { const f = document.getElementById('frame');"
                " return f.complete && f.naturalWidth > 0; })()"
            )
            page.screenshot(path=str(out / "ui-loaded.png"))

            # At frame 0 the ball is near (0.14, 0.40) of the frame, radius 0.12 of its height.
            aspect = 640 / 360
            draw_outline(page, 0.14, 0.40, 0.12 / aspect * 1.45, 0.12 * 1.45)
            page.wait_for_selector(".obj")
            page.wait_for_timeout(900)
            page.screenshot(path=str(out / "ui-outlined.png"))

            # A second object: the square.
            draw_outline(page, 0.86, 0.58, 0.13 / aspect * 1.6, 0.13 * 1.6)
            page.wait_for_function("document.querySelectorAll('.obj').length === 2")
            page.wait_for_timeout(900)

            page.click("#track")
            page.wait_for_selector("#result:not([hidden])", timeout=180_000)
            # Playwright's bundled Chromium has no H.264 decoder, so <video> readiness is not
            # checked. Verify what a real browser needs instead: the file is served, and
            # byte-range requests (used for seeking) are honoured.
            src = page.get_attribute("#result-video", "src")
            full = page.request.get(f"{base}{src}")
            ranged = page.request.get(f"{base}{src}", headers={"Range": "bytes=0-99"})
            print(
                "video:",
                full.status,
                full.headers.get("content-type"),
                len(full.body()),
                "bytes;",
                "range:",
                ranged.status,
            )
            if full.status != 200 or ranged.status != 206 or len(full.body()) < 2000:
                problems.append("result video is not served correctly")
            page.screenshot(path=str(out / "ui-result.png"))

            summary = page.inner_text("#result-summary")
            links = page.locator("#downloads a").count()
            print("summary:", summary)
            print("download links:", links)
            if links != 4:
                problems.append(f"expected 4 download links, found {links}")

            # Fix a frame: back to the editor with the tracked overlay visible.
            page.click("#fix")
            page.wait_for_selector("#viewport:not([hidden])")
            page.fill("#scrub", "30")
            page.dispatch_event("#scrub", "input")
            page.wait_for_timeout(700)
            page.screenshot(path=str(out / "ui-fix.png"))

            # Phone width: the layout must stack and never scroll sideways.
            phone = browser.new_context(
                viewport={"width": 390, "height": 844},
                device_scale_factor=2,
                has_touch=True,
                color_scheme=scheme,
            )
            mobile = phone.new_page()
            mobile.on("pageerror", lambda e: problems.append(f"mobile pageerror: {e}"))
            mobile.goto(base)
            mobile.wait_for_timeout(2600)
            mobile.screenshot(path=str(out / "ui-mobile-empty.png"))
            mobile.click("#sample")
            mobile.wait_for_selector("#viewport:not([hidden])")
            mobile.wait_for_timeout(800)
            overflow = mobile.evaluate(
                "document.documentElement.scrollWidth - document.documentElement.clientWidth"
            )
            if overflow > 0:
                problems.append(f"page scrolls sideways on a phone by {overflow}px")
            mobile.screenshot(path=str(out / "ui-mobile.png"), full_page=True)
            phone.close()
            browser.close()
    finally:
        server.terminate()
        server.wait(timeout=10)

    if problems:
        print("PROBLEMS:")
        for line in problems:
            print(" -", line)
        return 1
    print("UI smoke test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Launch DooDL from a Colab (or any Jupyter) notebook.

Two ways to open the app:
  * inline: Colab's own port proxy shows the UI inside the notebook, visible only to you;
  * share:  a temporary public https link through a Cloudflare quick tunnel, protected by
            a random access token, so you can send it to someone for a few minutes.
"""

from __future__ import annotations

import re
import secrets
import stat
import subprocess
import threading
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

CLOUDFLARED_URL = (
    "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"
)
_TUNNEL_RE = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")


@dataclass
class Running:
    port: int
    url: str | None = None
    token: str | None = None
    _server: object | None = field(default=None, repr=False)
    _tunnel: subprocess.Popen[str] | None = field(default=None, repr=False)

    def stop(self) -> None:
        if self._tunnel is not None:
            self._tunnel.terminate()
        if self._server is not None:
            self._server.should_exit = True  # type: ignore[attr-defined]


def _start_server(port: int, settings: object) -> object:
    import uvicorn

    from .api import create_app

    app = create_app(settings)  # type: ignore[arg-type]
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    threading.Thread(target=server.run, name="doodl-server", daemon=True).start()
    for _ in range(100):
        if getattr(server, "started", False):
            return server
        time.sleep(0.1)
    raise RuntimeError("The DooDL server did not start in 10 seconds.")


def _start_tunnel(port: int) -> tuple[subprocess.Popen[str], str]:
    exe = Path("/tmp/cloudflared")
    if not exe.exists():
        urllib.request.urlretrieve(CLOUDFLARED_URL, exe)
        exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    proc = subprocess.Popen(
        [str(exe), "tunnel", "--url", f"http://127.0.0.1:{port}", "--no-autoupdate"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    deadline = time.time() + 40
    assert proc.stderr is not None
    while time.time() < deadline:
        line = proc.stderr.readline()
        match = _TUNNEL_RE.search(line)
        if match:
            return proc, match.group(0)
        if proc.poll() is not None:
            break
    proc.terminate()
    raise RuntimeError("Could not open a public tunnel. Use the inline view instead (share=False).")


def launch(
    *,
    share: bool = False,
    port: int = 8000,
    height: int = 900,
    backend: str = "auto",
    sam2_model: str = "tiny",
    precision: str = "auto",
    prompt_strategy: str = "box_points",
) -> Running:
    """Start the app. Returns a handle with ``.stop()``."""
    from .config import Settings

    token = secrets.token_urlsafe(16) if share else None
    settings = Settings.from_env().replace(
        backend=backend,
        sam2_model=sam2_model,
        precision=precision,
        prompt_strategy=prompt_strategy,
        access_token=token,
    )
    running = Running(port=port, token=token)
    running._server = _start_server(port, settings)

    if share:
        running._tunnel, base = _start_tunnel(port)
        running.url = f"{base}/#token={token}"
        print("Share this link (it stops working when this notebook stops):")
        print(running.url)
        return running

    try:
        from google.colab import output

        output.serve_kernel_port_as_iframe(port, height=height)
    except ImportError:
        running.url = f"http://127.0.0.1:{port}/"
        print(f"Open {running.url}")
    return running

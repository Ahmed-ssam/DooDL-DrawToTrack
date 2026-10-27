from __future__ import annotations

from pathlib import Path

import pytest

from doodl import __version__, cli
from doodl.video import probe_video


def test_version_matches_pyproject() -> None:
    text = (Path(__file__).parents[1] / "pyproject.toml").read_text()
    assert f'version = "{__version__}"' in text


def test_check_reports_the_environment(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["check"]) == 0
    out = capsys.readouterr().out
    assert f"DooDL {__version__}" in out and "ffmpeg:" in out and "backend:" in out


@pytest.mark.slow
def test_sample_command_writes_a_playable_video(tmp_path: Path) -> None:
    target = tmp_path / "demo.mp4"
    assert cli.main(["sample", str(target), "--frames", "12"]) == 0
    assert probe_video(target).width == 640


def test_invalid_environment_is_a_clean_error(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("DOODL_MAX_FRAMES", "lots")
    assert cli.main(["serve"]) == 2
    assert "DOODL_MAX_FRAMES" in capsys.readouterr().err


def test_serving_publicly_without_a_token_warns(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    import uvicorn

    seen: dict[str, object] = {}
    monkeypatch.setattr(uvicorn, "run", lambda app, **kw: seen.update(kw))
    monkeypatch.delenv("DOODL_ACCESS_TOKEN", raising=False)
    code = cli.main(
        [
            "serve",
            "--backend",
            "classical",
            "--host",
            "0.0.0.0",
            "--port",
            "9123",
            "--data-dir",
            str(tmp_path),
        ]
    )
    assert code == 0 and seen["host"] == "0.0.0.0" and seen["port"] == 9123
    assert "without --token" in capsys.readouterr().err


def test_serving_locally_does_not_warn(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    import uvicorn

    monkeypatch.setattr(uvicorn, "run", lambda app, **kw: None)
    monkeypatch.delenv("DOODL_ACCESS_TOKEN", raising=False)
    assert cli.main(["serve", "--backend", "classical", "--data-dir", str(tmp_path)]) == 0
    assert "without --token" not in capsys.readouterr().err


def test_flags_override_the_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DOODL_BACKEND", "sam2")
    args = cli.build_parser().parse_args(
        ["serve", "--backend", "classical", "--model", "small", "--token", "t"]
    )
    settings = cli._settings_from_args(args)
    assert (settings.backend, settings.sam2_model, settings.access_token) == (
        "classical",
        "small",
        "t",
    )


def test_unknown_choice_is_rejected_by_the_parser() -> None:
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["serve", "--backend", "gpt"])


def test_open_flag_opens_the_browser_once_started(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import uvicorn

    opened: list[str] = []
    monkeypatch.setattr(uvicorn, "run", lambda app, **kw: None)
    monkeypatch.setattr(cli.webbrowser, "open", lambda url: opened.append(url))
    monkeypatch.setattr(
        cli.threading,
        "Timer",
        lambda _delay, fn, args=(): type("T", (), {"start": lambda s: fn(*args)})(),
    )
    args = ["serve", "--backend", "classical", "--port", "8123", "--data-dir", str(tmp_path)]
    assert cli.main([*args, "--open"]) == 0
    assert opened == ["http://127.0.0.1:8123"]
    opened.clear()
    assert cli.main(args) == 0
    assert opened == []

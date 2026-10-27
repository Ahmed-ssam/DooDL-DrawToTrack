"""Checks on the static UI that do not need a browser (the browser flow is scripts/ui_smoke.py)."""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

STATIC = Path(__file__).parents[1] / "src" / "doodl" / "static"
NODE = shutil.which("node")


def html_ids() -> set[str]:
    return set(re.findall(r'\sid="([^"]+)"', (STATIC / "index.html").read_text()))


def test_every_element_the_script_looks_up_exists_in_the_html() -> None:
    script = (STATIC / "app.js").read_text()
    block = re.search(r"Object\.fromEntries\(\s*\[(.*?)\]\.map", script, re.S)
    assert block, "could not find the element id list in app.js"
    wanted = set(re.findall(r"'([a-z0-9-]+)'", block.group(1)))
    wanted |= set(re.findall(r"\$\('([a-z0-9-]+)'\)", script))
    missing = wanted - html_ids()
    assert not missing, f"app.js looks up ids that index.html does not define: {sorted(missing)}"
    assert len(wanted) > 25


def test_html_ids_are_unique() -> None:
    ids = re.findall(r'\sid="([^"]+)"', (STATIC / "index.html").read_text())
    assert len(ids) == len(set(ids))


def test_page_has_the_basics_for_accessibility() -> None:
    page = (STATIC / "index.html").read_text()
    assert '<html lang="en">' in page and 'name="viewport"' in page
    assert 'role="alert"' in page and 'aria-label="Frame"' in page
    assert "<h1>" in page and page.count("<h1>") == 1


def test_stylesheet_respects_reduced_motion_and_focus() -> None:
    css = (STATIC / "styles.css").read_text()
    assert "prefers-reduced-motion: reduce" in css and ":focus-visible" in css


def test_no_external_resources() -> None:
    for name in ("index.html", "styles.css", "app.js", "api.js", "stage.js"):
        text = (STATIC / name).read_text()
        assert not re.search(r"(src|href)=\"https?://", text), name
        assert "fonts.googleapis" not in text and "cdn." not in text, name


@pytest.mark.skipif(NODE is None, reason="node is not installed")
@pytest.mark.parametrize("name", ["api.js", "stage.js", "app.js"])
def test_scripts_parse_as_modules(name: str, tmp_path: Path) -> None:
    copy = tmp_path / (name[:-3] + ".mjs")
    copy.write_text((STATIC / name).read_text())
    result = subprocess.run(
        [str(NODE), "--check", str(copy)], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_outline_simplification() -> None:
    script = Path(__file__).parent / "js" / "check_simplify.mjs"
    result = subprocess.run([str(NODE), str(script)], capture_output=True, text=True, check=False)
    assert result.returncode == 0 and "ok" in result.stdout, result.stderr

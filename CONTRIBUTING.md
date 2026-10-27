# Contributing

1. `pip install -e ".[dev]"` and `pre-commit install` (ffmpeg must be on PATH).
2. Make a change with a test. Keep `ruff check .`, `ruff format --check .`, `mypy src` and `pytest` green.
3. New models implement `SegmentationBackend` in `src/doodl/backends/` and register in `create_backend`.
4. After UI changes, run `python scripts/ui_smoke.py` and check the screenshots, including phone width.

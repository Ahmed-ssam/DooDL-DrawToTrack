# Changelog

## 0.1.0

First release: upload, outline, multi-object tracking, exports, SAM 2.1 and CPU baseline backends, FastAPI service, browser UI, Colab launcher, tests, CI.

## Unreleased

- New logo and interface design; the backend name is no longer shown in the UI.
- One-click launchers (`start.bat`, `start.sh`), `doodl serve --open`, bundled ffmpeg via `imageio-ffmpeg`.
- Simpler Colab notebook: Run all, falls back to the CPU tracker when no GPU is present.

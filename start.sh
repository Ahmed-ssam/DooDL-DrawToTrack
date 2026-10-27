#!/usr/bin/env sh
# One-click start for macOS and Linux: sets itself up the first time, then opens DooDL.
cd "$(dirname "$0")" || exit 1
PY=$(command -v python3 || command -v python)
if [ -z "$PY" ]; then
  echo "Python 3.10 or newer is required. Install it from https://www.python.org/downloads/"
  exit 1
fi
if [ ! -x .venv/bin/python ]; then
  echo "Setting up DooDL for the first time. This takes a minute or two..."
  "$PY" -m venv .venv || exit 1
fi
if [ ! -f .venv/.installed ]; then
  .venv/bin/python -m pip install --quiet --upgrade pip && \
  .venv/bin/python -m pip install --quiet -e . && touch .venv/.installed || exit 1
fi
exec .venv/bin/python -m doodl.cli serve --open

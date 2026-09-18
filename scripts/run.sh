#!/usr/bin/env bash
# Start the Crop Disease Detection app (creates the environment on first run).
set -euo pipefail
cd "$(dirname "$0")/.."

if command -v uv >/dev/null 2>&1; then
  uv sync --extra dev
  PY="uv run python"
else
  if [ ! -d .venv ]; then
    python3 -m venv .venv
    .venv/bin/pip install --upgrade pip
    .venv/bin/pip install -r requirements.txt
  fi
  PY=".venv/bin/python"
fi

export CROP_PORT="${CROP_PORT:-5001}"
echo "Open http://localhost:${CROP_PORT}/"
exec $PY backend/app.py

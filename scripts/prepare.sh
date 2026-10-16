#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
uv sync --locked --python 3.12
uv run --no-sync python scripts/doctor.py
uv run --no-sync python scripts/make_synthetic.py
uv run --no-sync pytest -q

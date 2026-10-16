#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
exec .venv/bin/jupyter lab --ip=127.0.0.1 --no-browser

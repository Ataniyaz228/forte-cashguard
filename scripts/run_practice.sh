#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
.venv/bin/python scripts/make_synthetic.py
.venv/bin/datathon audit data/synthetic/train.csv --target target --output reports/eda/practice
.venv/bin/datathon train --config configs/synthetic.yaml

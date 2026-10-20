#!/usr/bin/env bash
set -euo pipefail
python -m tissuemap.cli demo --output runs/first_smoke_test --epochs 1
pytest -q

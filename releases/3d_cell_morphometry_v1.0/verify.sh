#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
python tools/finalize_workflow07.py --verify
python -m pytest -q tests/test_cell_morphometry_v1.py

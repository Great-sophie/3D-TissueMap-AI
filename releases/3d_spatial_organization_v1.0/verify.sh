#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
python tools/freeze_workflow08.py --verify
python -m pytest -q tests/test_spatial_organization_v1.py

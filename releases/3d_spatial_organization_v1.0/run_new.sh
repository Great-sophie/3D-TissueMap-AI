#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
if [[ $# -ne 1 ]]; then
  echo "Usage: bash releases/3d_spatial_organization_v1.0/run_new.sh runs/NEW_OUTPUT" >&2
  exit 2
fi
if [[ -e "$1" ]]; then
  echo "Refusing to overwrite existing output: $1" >&2
  exit 2
fi
python scripts/run_3d_spatial_organization.py \
  --data-dir data/inspect/LVILymph \
  --morphometry-dir runs/3d_cell_morphometry_v1 \
  --out "$1" --assume-zyx --xy-radius-px 24 --knn-k 5

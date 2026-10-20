#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ "$#" -ne 1 ]]; then
  echo "Usage: bash releases/3d_cell_morphometry_v1.0/run_new.sh runs/NEW_OUTPUT" >&2
  exit 2
fi
cd "$ROOT"
if [[ -e "$1" ]]; then
  echo "Refusing to overwrite existing directory: $1" >&2
  exit 2
fi
python scripts/run_3d_cell_morphometry.py --data-dir data/inspect/LVILymph --out "$1" --assume-zyx

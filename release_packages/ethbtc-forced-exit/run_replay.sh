#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
package_root="$(cd "$(dirname "$0")" && pwd)"
cd "$package_root"
python sources/build_v22_grid_dca_forced_exit_v2.py \
  --result-dir inputs/frozen_v22 \
  --grid-candles inputs/candles/grid \
  --dca-candles inputs/candles/dca \
  --output-dir reproduced

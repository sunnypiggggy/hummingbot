$ErrorActionPreference = 'Stop'
$env:PYTHONDONTWRITEBYTECODE = '1'
$PackageRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Push-Location $PackageRoot
try {
    python sources/build_v22_grid_dca_forced_exit_v2.py `
      --result-dir inputs/frozen_v22 `
      --grid-candles inputs/candles/grid `
      --dca-candles inputs/candles/dca `
      --output-dir reproduced
} finally { Pop-Location }

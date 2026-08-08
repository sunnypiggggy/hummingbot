#!/usr/bin/env python3
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


package = Path(__file__).resolve().parent
environment = os.environ.copy()
environment["PYTHONDONTWRITEBYTECODE"] = "1"
command = [
    sys.executable,
    str(package / "sources/build_v22_grid_dca_forced_exit_v2.py"),
    "--result-dir", str(package / "inputs/frozen_v22"),
    "--grid-candles", str(package / "inputs/candles/grid"),
    "--dca-candles", str(package / "inputs/candles/dca"),
    "--output-dir", str(package / "reproduced"),
]
raise SystemExit(subprocess.run(command, cwd=package, env=environment, check=False).returncode)

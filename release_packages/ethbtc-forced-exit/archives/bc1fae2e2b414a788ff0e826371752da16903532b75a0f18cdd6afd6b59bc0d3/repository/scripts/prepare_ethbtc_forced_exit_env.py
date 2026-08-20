#!/usr/bin/env python3
"""Prepare non-secret OCI environment switches for the approved v22 cutover."""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path


CONFIRMATION = "PREPARE-ETHBTC-FORCED-EXIT-LIVE"
VALUES = {
    "GRID_V21_IN_GUARD_ENABLED": "false",
    "GRID_V21_LIVE_AUTHORIZED": "false",
    "GRID_V22_EXECUTION_MODE": "live",
    "GRID_RISK_V22_WEEKLY_GATE_ENABLED": "true",
    "DCA_RISK_V22_WEEKLY_GATE_ENABLED": "true",
    "DCA_RISK_AUTO_REENTRY_ENABLED": "true",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--backup", type=Path, required=True)
    parser.add_argument("--confirm", required=True)
    args = parser.parse_args()
    if args.confirm != CONFIRMATION:
        raise ValueError(f"explicit --confirm {CONFIRMATION} is required")
    original = args.env_file.read_text(encoding="utf-8")
    output: list[str] = []
    seen: set[str] = set()
    for line in original.splitlines():
        key = line.split("=", 1)[0].strip() if "=" in line and not line.lstrip().startswith("#") else ""
        if key in VALUES:
            output.append(f"{key}={VALUES[key]}")
            seen.add(key)
        else:
            output.append(line)
    for key, value in VALUES.items():
        if key not in seen:
            output.append(f"{key}={value}")
    args.backup.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.env_file, args.backup)
    temporary = args.env_file.with_suffix(args.env_file.suffix + ".v22.tmp")
    temporary.write_text("\n".join(output) + "\n", encoding="utf-8")
    os.replace(temporary, args.env_file)
    for key in VALUES:
        print(f"{key}={VALUES[key]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

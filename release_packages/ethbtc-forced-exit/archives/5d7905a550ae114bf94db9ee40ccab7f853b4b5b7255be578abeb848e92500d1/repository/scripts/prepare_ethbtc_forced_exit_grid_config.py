#!/usr/bin/env python3
"""Atomically prepare the live Grid config for v22 and automatic re-entry."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path


CONFIRMATION = "ENABLE-GRID-V22-AUTO-REENTRY"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--backup", type=Path, required=True)
    parser.add_argument("--production-lock", type=Path, required=True)
    parser.add_argument("--confirm", required=True)
    args = parser.parse_args()
    if args.confirm != CONFIRMATION:
        raise ValueError(f"explicit --confirm {CONFIRMATION} is required")
    original = args.config.read_text(encoding="utf-8")
    production = json.loads(args.production_lock.read_text(encoding="utf-8"))
    model_sha = str(production.get("model_sha256", ""))
    feature_sha = str(production.get("feature_schema_sha256", ""))
    if any(len(value) != 64 or any(char not in "0123456789abcdef" for char in value)
           for value in (model_sha, feature_sha)):
        raise ValueError("production lock is missing valid v22 model/feature hashes")
    lines = original.splitlines()
    replacements = {
        "technical_buy_gate_enabled": "true",
        "technical_buy_gate_file": "data/xgboost_risk_gate.json",
        "technical_model_sha256": model_sha,
        "technical_feature_sha256": feature_sha,
        "risk_auto_reentry_enabled": "true",
    }
    seen: set[str] = set()
    output: list[str] = []
    for line in lines:
        stripped = line.strip()
        key = stripped.split(":", 1)[0] if ":" in stripped else ""
        if key in replacements and not line.startswith((" ", "\t")):
            output.append(f"{key}: {replacements[key]}")
            seen.add(key)
        else:
            output.append(line)
    for key, value in replacements.items():
        if key not in seen:
            output.append(f"{key}: {value}")
    candidate = "\n".join(output) + "\n"
    if "technical_buy_gate_file: data/xgboost_risk_gate.json" not in candidate:
        raise RuntimeError("v22 Grid contract path was not prepared")
    if "risk_auto_reentry_enabled: true" not in candidate:
        raise RuntimeError("Grid automatic re-entry was not enabled")
    args.backup.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.config, args.backup)
    temporary = args.config.with_suffix(args.config.suffix + ".v22.tmp")
    temporary.write_text(candidate, encoding="utf-8")
    os.replace(temporary, args.config)
    print(f"prepared={args.config}")
    print(f"backup={args.backup}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

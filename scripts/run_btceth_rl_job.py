#!/usr/bin/env python3
"""Retain a research subprocess handle and its terminal exit evidence."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def stamp():
    return datetime.now(timezone.utc).isoformat()


def write(path, record):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=("fit", "test", "validate"))
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output_dir.resolve()
    proof = output / (args.phase + "_process.json")
    log = output / (args.phase + ".log")
    if proof.exists() or log.exists():
        raise FileExistsError("Preserve existing process evidence")
    command = [sys.executable, "-u", str(root / "scripts/backtest_btceth_rl.py"),
               "--phase", args.phase, "--dataset", str(args.dataset.resolve()),
               "--output-dir", str(output)]
    record = {"phase": args.phase, "started_at": stamp(), "supervisor_pid": os.getpid(),
              "command": command, "cwd": str(root), "status": "launching",
              "frozen_spec_sha256": hashlib.sha256((output / "frozen_spec.json").read_bytes()).hexdigest(),
              "log_relative_path": log.name}
    with log.open("x", encoding="utf-8") as stream:
        process = subprocess.Popen(command, cwd=root, stdin=subprocess.DEVNULL,
                                   stdout=stream, stderr=subprocess.STDOUT)
    record.update(status="running", child_pid=process.pid)
    write(proof, record)
    code = process.wait()
    record.update(status="completed" if code == 0 else "failed", exit_code=code,
                  completed_at=stamp(), log_sha256=hashlib.sha256(log.read_bytes()).hexdigest())
    write(proof, record)
    raise SystemExit(code)


if __name__ == "__main__":
    main()

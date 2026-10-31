#!/usr/bin/env python3
"""Record process and terminal evidence for a sealed research overlay phase."""
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


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=("validate", "test"))
    parser.add_argument("--main-output", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output, original, dataset = args.output_dir.resolve(), args.main_output.resolve(), args.dataset.resolve()
    if output == original or output.is_relative_to(original) or original.is_relative_to(output):
        raise ValueError("Original and overlay studies must remain separate")
    protocol_path = output / "protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8-sig"))
    runner = root / "scripts/btceth_rl_exposure_cap.py"
    if (sha(runner) != protocol["source_sha256"]["scripts/btceth_rl_exposure_cap.py"]
            or sha(dataset) != protocol["dataset_sha256"]
            or sha(original / "frozen_spec.json") != protocol["original_frozen_spec_sha256"]):
        raise ValueError("Runner, data or original study differs from the sealed protocol")
    proof, log = output / (args.phase + "_process.json"), output / (args.phase + ".log")
    if proof.exists() or log.exists():
        raise FileExistsError("Preserve existing process evidence")
    command = [sys.executable, "-u", str(runner), "--phase", args.phase,
               "--main-output", str(original), "--dataset", str(dataset), "--output-dir", str(output)]
    record = dict(phase=args.phase, started_at=stamp(), supervisor_pid=os.getpid(),
                  protocol_sha256=sha(protocol_path), runner_sha256=sha(runner),
                  command=command, cwd=str(root), log_relative_path=log.name, status="launching")
    with log.open("x", encoding="utf-8") as stream:
        process = subprocess.Popen(command, cwd=root, stdin=subprocess.DEVNULL,
                                   stdout=stream, stderr=subprocess.STDOUT)
    record.update(status="running", child_pid=process.pid)
    write(proof, record)
    code = process.wait()
    record.update(status="completed" if code == 0 else "failed", completed_at=stamp(),
                  exit_code=code, log_sha256=sha(log))
    write(proof, record)
    raise SystemExit(code)


if __name__ == "__main__":
    main()

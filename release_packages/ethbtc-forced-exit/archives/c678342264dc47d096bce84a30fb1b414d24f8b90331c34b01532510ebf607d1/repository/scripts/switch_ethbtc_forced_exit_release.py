#!/usr/bin/env python3
"""Atomically point ethbtc-forced-exit/current at a validated release hash."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path,
                        default=Path("release_packages/ethbtc-forced-exit"))
    parser.add_argument("--release-sha256", required=True)
    args = parser.parse_args()
    if len(args.release_sha256) != 64 or any(c not in "0123456789abcdef" for c in args.release_sha256):
        raise ValueError("invalid release SHA-256")
    root = args.root.resolve()
    target = root / "releases" / args.release_sha256
    lock = json.loads((target / "production_lock.json").read_text(encoding="utf-8"))
    if lock.get("release_sha256") != args.release_sha256:
        raise ValueError("release directory and production lock disagree")
    current = root / "current"
    if os.name == "nt":
        is_junction = getattr(os.path, "isjunction", lambda _: False)(current)
        if current.exists() and not (current.is_symlink() or is_junction):
            raise RuntimeError("refusing to replace a non-link current directory")
        if current.exists():
            os.rmdir(current)
        try:
            os.symlink(target, current, target_is_directory=True)
        except OSError as exc:
            # Windows junctions do not require Developer Mode or symlink privilege.
            result = subprocess.run(
                ["cmd", "/c", "mklink", "/J", str(current), str(target)],
                capture_output=True, text=True, check=False,
            )
            if result.returncode:
                raise RuntimeError(result.stderr or result.stdout) from exc
    else:
        temporary = root / ".current.next"
        temporary.unlink(missing_ok=True)
        temporary.symlink_to(Path("releases") / args.release_sha256, target_is_directory=True)
        os.replace(temporary, current)
    print(json.dumps({"current": str(current), "release_sha256": args.release_sha256}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

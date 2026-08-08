#!/usr/bin/env python3
"""Verify the integrity and deployment posture of an ethbtc-forced-exit package."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


PACKAGE_ID = "ethbtc-forced-exit"
MANIFEST = "MANIFEST.sha256"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: object) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def documentation_hash(package_dir: Path) -> str:
    documentation = package_dir / "documentation"
    if not documentation.is_dir() or not (documentation / "README.md").is_file():
        raise ValueError("release documentation is missing")
    hashes = {
        path.relative_to(documentation).as_posix(): sha256_file(path)
        for path in sorted(documentation.rglob("*"))
        if path.is_file()
    }
    return canonical_hash(hashes)


def parse_manifest(path: Path) -> dict[str, str]:
    entries: dict[str, str] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            digest, relative = line.split("  ", 1)
        except ValueError as exc:
            raise ValueError(f"invalid manifest line {line_number}") from exc
        if len(digest) != 64 or relative in entries:
            raise ValueError(f"invalid manifest entry {relative!r}")
        entries[relative] = digest
    return entries


def verify(package_dir: Path, require_deployable: bool = False) -> dict[str, object]:
    package_dir = package_dir.resolve()
    entries = parse_manifest(package_dir / MANIFEST)
    actual = set()
    for path in package_dir.rglob("*"):
        if not path.is_file() or path.name == MANIFEST:
            continue
        relative = path.relative_to(package_dir)
        # Replays are deliberately written to a mutable, non-manifested output
        # directory. Python bytecode is also runtime residue, not release input.
        if relative.parts[0] == "reproduced" or "__pycache__" in relative.parts or path.suffix == ".pyc":
            continue
        actual.add(relative.as_posix())
    expected = set(entries)
    if missing := sorted(expected - actual):
        raise ValueError(f"package files missing: {missing}")
    if unexpected := sorted(actual - expected):
        raise ValueError(f"unmanifested package files: {unexpected}")
    mismatches = [name for name, digest in entries.items()
                  if sha256_file(package_dir / name) != digest]
    if mismatches:
        raise ValueError(f"package hash mismatch: {mismatches}")

    release = json.loads((package_dir / "release.json").read_text(encoding="utf-8"))
    production_path = package_dir / "production_lock.json"
    if production_path.is_file():
        production = json.loads(production_path.read_text(encoding="utf-8"))
        lock_path = package_dir / "shadow_package/shadow_lock.json"
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        model = package_dir / "shadow_package/models/xgboost_long_risk_gate_v22_weekly.joblib"
        expected_documentation_sha = production.get("documentation_sha256")
        checks = {
            "package_id": production.get("package_id") == PACKAGE_ID == release.get("package_id"),
            "release_sha": production.get("release_sha256") == release.get("release_sha256"),
            "shadow_lock_sha": sha256_file(lock_path) == production.get("shadow_lock_sha256"),
            "model_sha": sha256_file(model) == lock.get("model_sha256") == production.get("model_sha256"),
            "feature_sha": lock.get("feature_schema_sha256") == production.get("feature_schema_sha256"),
            "strategy_sha": lock.get("strategy_schema_sha256") == production.get("strategy_schema_sha256"),
            "effective_end": int(lock.get("effective_end", 0)) == int(production.get("effective_end", -1)),
            "fallback_forbidden": production.get("previous_model_fallback_allowed") is False,
            "candidate_closed": production.get("deployment_allowed") is False,
            # Releases staged before documentation became part of the release
            # identity remain verifiable and immutable. All newer releases bind
            # the complete UTF-8 documentation tree to the production lock.
            "documentation_sha": (
                expected_documentation_sha is None
                or documentation_hash(package_dir) == expected_documentation_sha
            ),
        }
        if not all(checks.values()):
            raise ValueError(f"production candidate validation failed: {checks}")
        if require_deployable:
            raise RuntimeError(
                "production candidate is intentionally closed; deployment requires "
                "24h observation and the OCI-local hash-bound approval receipt"
            )
        return {
            "package_id": PACKAGE_ID,
            "integrity": "PASS",
            "files": len(entries),
            "deployment_allowed": False,
            "release_stage": release.get("release_stage"),
            "release_sha256": production["release_sha256"],
            "effective_end": production["effective_end"],
            "checks": checks,
        }
    summary = json.loads((package_dir / "evidence/summary.json").read_text(encoding="utf-8"))
    policy = json.loads((package_dir / "evidence/execution_policy.json").read_text(encoding="utf-8"))
    lock = json.loads((package_dir / "inputs/frozen_v22/shadow_package/shadow_lock.json").read_text(encoding="utf-8"))
    for name, value in (("release", release), ("summary", summary), ("policy", policy)):
        if value.get("package_id") != PACKAGE_ID:
            raise ValueError(f"{name} package_id mismatch")
    if summary.get("execution_policy_sha256") != policy.get("execution_policy_sha256"):
        raise ValueError("summary/execution policy hash mismatch")
    model = package_dir / "inputs/frozen_v22/shadow_package/models/xgboost_long_risk_gate_v22_weekly.joblib"
    if sha256_file(model) != lock.get("model_sha256"):
        raise ValueError("frozen model hash mismatch")
    states = package_dir / "inputs/frozen_v22/application_bundle/risk_states.csv.gz"
    if sha256_file(states) != summary.get("frozen_inputs", {}).get("risk_states_sha256"):
        raise ValueError("frozen risk-state hash mismatch")
    deployable = bool(release.get("deployment_allowed")) and summary.get("verdict") == "GO"
    if require_deployable and not deployable:
        raise RuntimeError("package integrity is valid, but deployment gate is closed")
    return {
        "package_id": PACKAGE_ID,
        "integrity": "PASS",
        "files": len(entries),
        "deployment_allowed": deployable,
        "release_stage": release.get("release_stage"),
        "blockers": release.get("deployment_blockers", []),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package_dir", nargs="?", type=Path, default=Path.cwd())
    parser.add_argument("--require-deployable", action="store_true")
    args = parser.parse_args()
    print(json.dumps(verify(args.package_dir, args.require_deployable), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

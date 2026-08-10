#!/usr/bin/env python3
"""Verify an OCI archive, and optionally compare it with the live OCI source."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
from pathlib import Path


EXCLUDED_NAMES = {
    "archives",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
}
EXCLUDED_SUFFIXES = (".pyc", ".pyo")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_list(path: Path) -> dict[str, str]:
    entries: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        digest, relative = line.split("  ", 1)
        if len(digest) != 64 or relative in entries:
            raise ValueError(f"invalid hash-list entry: {relative}")
        entries[relative] = digest
    return entries


def png_dimensions(path: Path) -> tuple[int, int]:
    with path.open("rb") as stream:
        header = stream.read(24)
    if len(header) != 24 or header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
        raise ValueError(f"invalid PNG: {path}")
    return struct.unpack(">II", header[16:24])


def source_file_names(source_root: Path, source_paths: list[str]) -> set[str]:
    names: set[str] = set()
    for relative in source_paths:
        source = source_root / relative
        if source.is_file():
            names.add(Path(relative).as_posix())
            continue
        if not source.is_dir():
            raise FileNotFoundError(source)
        for directory, directories, files in os.walk(source, followlinks=True):
            directories[:] = sorted(name for name in directories if name not in EXCLUDED_NAMES)
            for name in sorted(files):
                if name in EXCLUDED_NAMES or name.endswith(EXCLUDED_SUFFIXES):
                    continue
                path = Path(directory) / name
                names.add(path.relative_to(source_root).as_posix())
    return names


def verify(root: Path, source_root: Path | None = None) -> dict:
    root = root.resolve()
    entries = hash_list(root / "MANIFEST.sha256")
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path != root / "MANIFEST.sha256"
    }
    if actual != set(entries):
        raise ValueError(
            {
                "missing": sorted(set(entries) - actual),
                "unexpected": sorted(actual - set(entries)),
            }
        )
    mismatches = [
        name for name, digest in entries.items() if sha256_file(root / name) != digest
    ]
    if mismatches:
        raise ValueError(f"archive hash mismatch: {mismatches}")
    if root.name != sha256_file(root / "MANIFEST.sha256"):
        raise ValueError("archive directory is not the manifest content hash")

    forbidden = [
        path
        for path in root.rglob("*")
        if path.is_file()
        and (
            path.name in {".env", ".env.control"}
            or path.suffix in {".pem", ".key"}
            or "secrets" in path.parts
        )
    ]
    if forbidden:
        raise ValueError(f"secret-bearing paths found: {forbidden}")

    repository = root / "repository"
    metadata = json.loads((root / "archive.json").read_text(encoding="utf-8"))
    snapshot = hash_list(root / "SOURCE_SNAPSHOT.sha256")
    repository_actual = {
        path.relative_to(repository).as_posix()
        for path in repository.rglob("*")
        if path.is_file()
    }
    if repository_actual != set(snapshot):
        raise ValueError(
            {
                "repository_missing": sorted(set(snapshot) - repository_actual),
                "repository_unexpected": sorted(repository_actual - set(snapshot)),
            }
        )
    repository_mismatches = [
        name
        for name, digest in snapshot.items()
        if sha256_file(repository / name) != digest
    ]
    if repository_mismatches:
        raise ValueError(f"repository snapshot mismatch: {repository_mismatches}")
    if metadata["repository_file_count"] != len(snapshot):
        raise ValueError("repository file count does not match source snapshot")

    for relative in metadata["source_paths"]:
        if not (repository / relative).exists():
            raise FileNotFoundError(relative)

    notification_settings = {}
    for line in (repository / "telegram-notify.env").read_text(encoding="utf-8").splitlines():
        if line and not line.startswith("#"):
            key, value = line.split("=", 1)
            notification_settings[key] = value
    allowed_notification_keys = {
        "TELEGRAM_NOTIFY_ENABLED",
        "TELEGRAM_PROFIT_REPORT_ENABLED",
        "TELEGRAM_NOTIFY_CHANNEL_ID",
    }
    if set(notification_settings) != allowed_notification_keys:
        raise ValueError("telegram-notify.env contains an unexpected or sensitive key")

    for dockerfile_name in (
        "Dockerfile.dca-live-guard",
        "Dockerfile.grid-live-guard",
        "Dockerfile.grid-live-fdusd-scheduler",
    ):
        dockerfile = (repository / dockerfile_name).read_text(encoding="utf-8")
        for line in dockerfile.splitlines():
            if line.startswith("COPY "):
                source = line.split()[1]
                if not (repository / source).exists():
                    raise FileNotFoundError(f"Docker COPY dependency missing: {source}")

    evidence = (
        repository
        / "release_packages/ethbtc-forced-exit/audits/v22-png-backtest-current-v2"
    )
    report = json.loads((evidence / "manifest.json").read_text(encoding="utf-8"))
    event = json.loads((evidence / "telegram_event.json").read_text(encoding="utf-8"))
    expected = {
        (strategy, pair, window)
        for strategy, pair in (
            ("grid", "BTC-FDUSD"),
            ("grid", "ETH-FDUSD"),
            ("dca", "BTC-USDT"),
            ("dca", "ETH-USDT"),
        )
        for window in ("360d", "2026_jan_feb", "2026_may_june")
    }
    observed = {
        (row["strategy"], row["pair"], row["window"]) for row in report["images"]
    }
    if observed != expected:
        raise ValueError("four-robot/three-window image contract is incomplete")
    if len(event["attachments"]) != 12 or any(
        row["kind"] != "photo" for row in event["attachments"]
    ):
        raise ValueError("Telegram update must contain exactly 12 photos and no documents")
    for row in report["images"]:
        # The evidence manifest is portable and may have been generated on Windows.
        image_name = str(row["path"]).replace("\\", "/").rsplit("/", 1)[-1]
        image_path = evidence / image_name
        if sha256_file(image_path) != row["sha256"]:
            raise ValueError(f"image hash mismatch: {image_path.name}")
        if png_dimensions(image_path) != (1440, 2400):
            raise ValueError(f"invalid mobile image dimensions: {image_path.name}")

    source_match: bool | None = None
    if source_root is not None:
        source_root = source_root.resolve()
        live_names = source_file_names(source_root, metadata["source_paths"])
        if live_names != set(snapshot):
            raise ValueError(
                {
                    "oci_missing_from_archive": sorted(live_names - set(snapshot)),
                    "archive_missing_from_oci": sorted(set(snapshot) - live_names),
                }
            )
        live_mismatches = [
            name
            for name, digest in snapshot.items()
            if sha256_file(source_root / name) != digest
        ]
        if live_mismatches:
            raise ValueError(f"OCI/archive source mismatch: {live_mismatches}")
        source_match = True

    return {
        "integrity": "PASS",
        "archive_sha256": root.name,
        "archive_files": len(entries),
        "repository_files": len(snapshot),
        "oci_source_match": source_match,
        "compose_services": metadata["compose_service_count"],
        "telegram_photos": 12,
        "telegram_documents": 0,
        "secrets_included": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--source-root", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            verify(args.archive, source_root=args.source_root),
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

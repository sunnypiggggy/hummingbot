from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(package: Path) -> str:
    manifest_path = package / "MANIFEST.sha256"
    expected: dict[str, str] = {}
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        digest, relative = line.split("  ", 1)
        if relative.startswith("/") or ".." in Path(relative).parts:
            raise RuntimeError(f"unsafe manifest path: {relative}")
        expected[relative] = digest
    actual = {
        path.relative_to(package).as_posix()
        for path in package.rglob("*")
        if path.is_file() and path.name != manifest_path.name
    }
    if actual != set(expected):
        raise RuntimeError(f"manifest file set mismatch: missing={set(expected) - actual}, extra={actual - set(expected)}")
    mismatches = [relative for relative, digest in expected.items() if sha256(package / relative) != digest]
    if mismatches:
        raise RuntimeError(f"hash mismatch: {mismatches}")

    source = json.loads((package / "SOURCE_MANIFEST.json").read_text(encoding="utf-8"))
    canonical = b"".join(
        relative.encode("utf-8") + b"\0" + item["sha256"].encode("ascii") + b"\n"
        for relative, item in sorted(source["files"].items())
    )
    release_sha = hashlib.sha256(canonical).hexdigest()
    if release_sha != source["release_sha256"] or package.name != release_sha:
        raise RuntimeError("release digest or content-addressed directory mismatch")
    if json.loads((package / "config/binance_stocks_credentials.paper.json").read_text(encoding="utf-8")):
        raise RuntimeError("Paper credential scaffold is not empty")
    return release_sha


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("package", type=Path)
    args = parser.parse_args()
    print(verify(args.package.resolve()))

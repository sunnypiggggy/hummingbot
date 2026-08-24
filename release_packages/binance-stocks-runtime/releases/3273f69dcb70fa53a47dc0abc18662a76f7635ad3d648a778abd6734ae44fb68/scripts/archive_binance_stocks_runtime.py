from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from install_binance_stocks_runtime import SECRET_BLOCK, SERVICE_BLOCK


PACKAGE_ID = "binance-stocks-runtime"
SCHEMA = "binance-stocks-runtime-package-v2"

FILES = (
    ".env.control.example",
    "Dockerfile.binance-stocks-runtime",
    "Dockerfile.trading-management-bot",
    "docker-compose.yml",
    "config/binance_stocks_credentials.paper.json",
    "docs/BINANCE_STOCKS_CONNECTOR.md",
    "docs/BINANCE_STOCKS_PAPER_TRADING.md",
    "docs/BINANCE_STOCKS_RUNTIME.md",
    "docs/TRADING_MANAGEMENT_BOT_V3.md",
    "hummingbot/strategy_v2/executors/binance_stocks_order_executor.py",
    "hummingbot/strategy_v2/executors/binance_stocks_position_executor.py",
    "scripts/archive_binance_stocks_runtime.py",
    "scripts/audit_binance_stocks_runtime_deployment.py",
    "scripts/check_binance_stocks_runtime.py",
    "scripts/deploy_binance_stocks_limits_v3.sh",
    "scripts/install_binance_stocks_runtime.py",
    "scripts/migrate_binance_stocks_limits_v3.sql",
    "scripts/run_binance_stocks_paper_scenario.py",
    "scripts/smoke_binance_stocks_paper_api.py",
    "scripts/smoke_binance_stocks_async_queue.py",
    "scripts/verify_binance_stocks_runtime_package.py",
    "test/test_binance_stocks_runtime_deployment.py",
    "test/test_trading_management_bot.py",
    "test/__init__.py",
    "test/hummingbot/__init__.py",
    "test/hummingbot/connector/__init__.py",
    "test/hummingbot/connector/exchange/__init__.py",
)

DIRECTORIES = (
    "hummingbot/connector/exchange/binance_stocks",
    "stocks_runtime",
    "management_bot",
    "test/hummingbot/connector/exchange/binance_stocks",
    "test/stocks_runtime",
)

GENERATED = {
    "deployment/binance-stocks-runtime.compose.yml": ("services:\n" + SERVICE_BLOCK).encode("utf-8"),
    "deployment/binance-stocks-runtime.secret.yml": ("secrets:\n" + SECRET_BLOCK).encode("utf-8"),
}


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def collect(root: Path) -> dict[str, bytes]:
    content: dict[str, bytes] = {}
    for relative in FILES:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        content[relative] = path.read_bytes()
    for relative in DIRECTORIES:
        directory = root / relative
        if not directory.is_dir():
            raise FileNotFoundError(directory)
        for path in sorted(directory.rglob("*")):
            if not path.is_file() or "__pycache__" in path.parts or path.suffix in {".pyc", ".pyo"}:
                continue
            rel = path.relative_to(root).as_posix()
            content[rel] = path.read_bytes()
    content.update(GENERATED)
    return dict(sorted(content.items()))


def release_digest(content: dict[str, bytes]) -> str:
    canonical = b"".join(
        path.encode("utf-8") + b"\0" + sha256_bytes(payload).encode("ascii") + b"\n"
        for path, payload in sorted(content.items())
    )
    return sha256_bytes(canonical)


def reject_secrets(content: dict[str, bytes]) -> None:
    forbidden_names = {".env.control", "binance_stocks_credentials.json"}
    bad = [path for path in content if Path(path).name in forbidden_names]
    if bad:
        raise RuntimeError(f"refusing to archive secret-bearing files: {bad}")
    paper = json.loads(content["config/binance_stocks_credentials.paper.json"].decode("utf-8"))
    if paper:
        raise RuntimeError("Paper credential scaffold must remain an empty JSON object")


def build(root: Path, output_root: Path) -> Path:
    content = collect(root)
    reject_secrets(content)
    release_sha = release_digest(content)
    release_dir = output_root / "releases" / release_sha
    source_manifest = {
        "schema": SCHEMA,
        "package_id": PACKAGE_ID,
        "release_sha256": release_sha,
        "files": {
            path: {"bytes": len(payload), "sha256": sha256_bytes(payload)}
            for path, payload in content.items()
        },
    }
    package = {
        "schema": SCHEMA,
        "package_id": PACKAGE_ID,
        "release_sha256": release_sha,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "runtime_mode": "PAPER",
        "live_authorized": False,
        "secrets_included": False,
        "position_source": "paper_ledger",
        "external_positions_unknown": False,
        "economic_http_requests_enabled": False,
        "paper_initial_usdc": "2000",
        "base_image": "hummingbot-api-orchestration:local",
    }

    if release_dir.exists():
        existing = json.loads((release_dir / "SOURCE_MANIFEST.json").read_text(encoding="utf-8"))
        if existing != source_manifest:
            raise RuntimeError(f"immutable release directory does not match source: {release_dir}")
    else:
        staging = output_root / ".staging" / release_sha
        if staging.exists():
            shutil.rmtree(staging)
        for relative, payload in content.items():
            atomic_write(staging / relative, payload)
        atomic_write(
            staging / "SOURCE_MANIFEST.json",
            (json.dumps(source_manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8"),
        )
        atomic_write(
            staging / "PACKAGE.json",
            (json.dumps(package, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8"),
        )
        entries = []
        for path in sorted(p for p in staging.rglob("*") if p.is_file()):
            relative = path.relative_to(staging).as_posix()
            entries.append(f"{sha256_bytes(path.read_bytes())}  {relative}")
        atomic_write(staging / "MANIFEST.sha256", ("\n".join(entries) + "\n").encode("utf-8"))
        release_dir.parent.mkdir(parents=True, exist_ok=True)
        staging.replace(release_dir)

    current = {
        "schema": SCHEMA,
        "package_id": PACKAGE_ID,
        "release_sha256": release_sha,
        "path": f"releases/{release_sha}",
    }
    atomic_write(
        output_root / "CURRENT.json",
        (json.dumps(current, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    return release_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    source_root = args.root.resolve()
    output = args.output or source_root / "release_packages" / PACKAGE_ID
    print(build(source_root, output.resolve()))

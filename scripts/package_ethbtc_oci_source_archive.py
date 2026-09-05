#!/usr/bin/env python3
"""Build a content-addressed archive from the authoritative OCI source tree."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any


# This is the deployed Grid/DCA/v22/Telegram source and dependency closure. Runtime
# databases, balances, credentials and logs are deliberately outside these paths.
SOURCE_PATHS = (
    "Dockerfile.dca-live-guard",
    "Dockerfile.grid-live-guard",
    "Dockerfile.dca-live-runtime",
    "Dockerfile.portfolio-grid-runtime",
    "Dockerfile.grid-live-fdusd-scheduler",
    "docker-compose.yml",
    "telegram-notify.env",
    "requirements-grid-xgboost.txt",
    "pyproject.toml",
    "setup.py",
    "live_guard",
    "scheduler",
    "scripts",
    "test",
    "controllers/market_making",
    "conf/controllers",
    "hummingbot/strategy_v2/executors/dca_executor",
    "hummingbot/connector/exchange/binance/binance_exchange.py",
    "release_packages/ethbtc-forced-exit",
)
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


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def ignored(_directory: str, names: list[str]) -> set[str]:
    return {
        name
        for name in names
        if name in EXCLUDED_NAMES or name.endswith(EXCLUDED_SUFFIXES)
    }


def copy_path(source_root: Path, staging: Path, relative: str) -> None:
    source = source_root / relative
    target = staging / "repository" / relative
    if source.is_dir():
        shutil.copytree(source, target, ignore=ignored, symlinks=False)
    elif source.is_file():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    else:
        raise FileNotFoundError(source)


def command_output(source_root: Path, command: list[str]) -> str:
    return subprocess.run(
        command,
        cwd=source_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def write_hash_list(root: Path, output: Path, files: list[Path]) -> None:
    output.write_text(
        "\n".join(
            f"{sha256_file(path)}  {path.relative_to(root).as_posix()}" for path in files
        )
        + "\n",
        encoding="utf-8",
    )


def build(source_root: Path, target_root: Path, deployment_receipt: Path | None) -> Path:
    source_root = source_root.resolve()
    target_root = target_root.resolve()
    staging = target_root / ".oci-source-archive.staging"
    if staging.exists():
        raise FileExistsError(staging)
    staging.mkdir(parents=True)
    try:
        for relative in SOURCE_PATHS:
            copy_path(source_root, staging, relative)

        repository_files = sorted(
            path for path in (staging / "repository").rglob("*") if path.is_file()
        )
        write_hash_list(
            staging / "repository",
            staging / "SOURCE_SNAPSHOT.sha256",
            repository_files,
        )

        if deployment_receipt is not None:
            target = staging / "OCI_DEPLOYMENT_RECEIPT.json"
            target.write_bytes(deployment_receipt.read_bytes())

        services = command_output(
            source_root, ["docker", "compose", "config", "--services"]
        ).splitlines()
        metadata = {
            "schema": "ethbtc-oci-source-archive-v1",
            "package_id": "ethbtc-forced-exit",
            "archive_scope": (
                "authoritative OCI Grid/DCA source, tests, release family, "
                "runtime configuration and build dependency closure"
            ),
            "source_paths": list(SOURCE_PATHS),
            "repository_file_count": len(repository_files),
            "excluded_runtime_paths": [
                "api-files",
                "dca-live-data",
                "grid-live-fdusd-data",
                "account-inventory-data",
                "dca-macro-data",
                "mcp-files",
                "secrets",
                ".env",
                ".env.control",
            ],
            "compose_services": services,
            "compose_service_count": len(services),
            "telegram_model_binary_attachment_allowed": False,
            "release_model_binary_snapshot_included": True,
            "deployment_authority_granted": False,
        }
        atomic_json(staging / "archive.json", metadata)
        (staging / "README.md").write_text(
            "# ethbtc-forced-exit OCI 完整源码归档\n\n"
            "本目录是 OCI 当前 Grid、DCA、v22、Telegram 运行源码、配置、测试、"
            "Docker/Compose、发布族和模型依赖的内容寻址快照。`repository/` 保持 OCI "
            "相对路径，可用于逐文件差异核验和重建。\n\n"
            "运行状态、数据库、余额、日志、API/Telegram/交易所密钥和环境文件不归档。"
            "release 中的模型二进制仅用于依赖闭包和回溯，不会作为 Telegram 附件发送，"
            "也不授予交易权限。\n\n"
            "包内完整性校验：\n\n"
            "```bash\n"
            "python tools/verify_ethbtc_oci_source_archive.py <archive-dir>\n"
            "```\n\n"
            "与 OCI 源目录逐文件校验：\n\n"
            "```bash\n"
            "python tools/verify_ethbtc_oci_source_archive.py <archive-dir> "
            "--source-root /home/ubuntu/extra_drive/hummingbot\n"
            "```\n",
            encoding="utf-8",
        )
        verifier = source_root / "scripts/verify_ethbtc_oci_source_archive.py"
        (staging / "tools").mkdir(parents=True)
        shutil.copy2(verifier, staging / "tools" / verifier.name)

        files = sorted(path for path in staging.rglob("*") if path.is_file())
        manifest_path = staging / "MANIFEST.sha256"
        write_hash_list(staging, manifest_path, files)
        archive_sha = sha256_file(manifest_path)
        target = target_root / archive_sha
        if target.exists():
            raise FileExistsError(target)
        os.replace(staging, target)
        atomic_json(
            target_root / "current.json",
            {
                "schema": "ethbtc-oci-source-archive-pointer-v1",
                "archive_sha256": archive_sha,
                "path": f"archives/{archive_sha}",
            },
        )
        return target
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path.cwd())
    parser.add_argument("--target-root", type=Path, required=True)
    parser.add_argument("--deployment-receipt", type=Path)
    args = parser.parse_args()
    print(build(args.source_root, args.target_root, args.deployment_receipt))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

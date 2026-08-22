from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


CONTAINER = "binance-stocks-runtime"
SITE_PACKAGES = "/opt/conda/envs/hummingbot-api/lib/python3.12/site-packages"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def command(*args: str) -> str:
    return subprocess.run(args, check=True, capture_output=True, text=True).stdout.strip()


def image_path(relative: str) -> str | None:
    if relative.startswith("stocks_runtime/"):
        return f"/hummingbot-api/{relative}"
    if relative.startswith("hummingbot/connector/exchange/binance_stocks/"):
        return f"{SITE_PACKAGES}/{relative}"
    if relative in {
        "hummingbot/strategy_v2/executors/binance_stocks_order_executor.py",
        "hummingbot/strategy_v2/executors/binance_stocks_position_executor.py",
    }:
        return f"{SITE_PACKAGES}/{relative}"
    return None


def extract_service(compose: str) -> str:
    start = compose.index("  binance-stocks-runtime:\n")
    end = compose.index("  hummingbot-mcp:\n", start)
    return compose[start:end]


def audit(root: Path, package: Path) -> dict:
    source = json.loads((package / "SOURCE_MANIFEST.json").read_text(encoding="utf-8"))
    mismatches: list[str] = []
    image_checked = 0
    generated = {path for path in source["files"] if path.startswith("deployment/")}
    for relative, metadata in source["files"].items():
        if relative in generated:
            continue
        path = root / relative
        if not path.is_file() or sha256(path) != metadata["sha256"]:
            mismatches.append(f"host:{relative}")
            continue
        target = image_path(relative)
        if target:
            output = command("docker", "exec", CONTAINER, "sha256sum", target)
            if output.split()[0] != metadata["sha256"]:
                mismatches.append(f"image:{relative}")
            image_checked += 1

    compose = (root / "docker-compose.yml").read_text(encoding="utf-8").replace("\r\n", "\n")
    expected_service = (package / "deployment/binance-stocks-runtime.compose.yml").read_text(
        encoding="utf-8"
    ).replace("\r\n", "\n").removeprefix("services:\n")
    if extract_service(compose) != expected_service:
        mismatches.append("host:docker-compose.service")
    expected_secret = (package / "deployment/binance-stocks-runtime.secret.yml").read_text(
        encoding="utf-8"
    ).replace("\r\n", "\n").removeprefix("secrets:\n")
    if expected_secret not in compose:
        mismatches.append("host:docker-compose.secret")

    inspect = json.loads(command("docker", "inspect", CONTAINER))[0]
    environment = dict(item.split("=", 1) for item in inspect["Config"]["Env"] if "=" in item)
    health = (inspect.get("State", {}).get("Health") or {}).get("Status")
    if inspect.get("State", {}).get("Status") != "running" or health != "healthy":
        mismatches.append("container:health")
    if environment.get("BINANCE_STOCKS_RUNTIME_MODE") != "PAPER":
        mismatches.append("container:runtime_mode")
    if environment.get("BINANCE_STOCKS_LIVE_AUTHORIZED", "").lower() != "false":
        mismatches.append("container:live_authorized")
    if mismatches:
        raise RuntimeError("deployment integrity mismatch: " + ", ".join(mismatches))
    return {
        "release_sha256": source["release_sha256"],
        "host_files_checked": len(source["files"]) - len(generated),
        "image_files_checked": image_checked,
        "container_image": inspect["Image"],
        "container_status": "running/healthy",
        "runtime_mode": "PAPER",
        "live_authorized": False,
        "mismatches": 0,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--package", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.root.resolve(), args.package.resolve()), sort_keys=True))

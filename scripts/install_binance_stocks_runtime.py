from __future__ import annotations

import argparse
import os
import secrets
import shutil
from datetime import datetime, timezone
from pathlib import Path


SERVICE_BLOCK = '''  binance-stocks-runtime:
    profiles:
      - stocks
    container_name: binance-stocks-runtime
    build:
      context: .
      dockerfile: Dockerfile.binance-stocks-runtime
    image: hummingbot/binance-stocks-runtime:local
    restart: unless-stopped
    env_file:
      - .env.control
    environment:
      BINANCE_STOCKS_CREDENTIALS_FILE: /run/secrets/binance_stocks_credentials
      BINANCE_STOCKS_ACCOUNT: stocks_managed
      BINANCE_STOCKS_DATABASE_NAME: hummingbot_stocks
      BINANCE_STOCKS_ORDER_PREFIX: x-HBSTK
      BINANCE_STOCKS_MAX_ORDER_USDC: "200"
      BINANCE_STOCKS_MAX_SYMBOL_USDC: "200"
      BINANCE_STOCKS_MAX_EXPOSURE_USDC: "2000"
      BINANCE_STOCKS_DAILY_LOSS_USDC: "200"
    ports:
      - "127.0.0.1:8001:8000"
    volumes:
      - ./api-files/stocks-runtime/bots:/hummingbot-api/bots
      - ./api-files/stocks-runtime/logs:/hummingbot-api/logs
    secrets:
      - binance_stocks_credentials
    depends_on:
      - postgres
    networks:
      - hummingbot-control

'''

SECRET_BLOCK = '''  binance_stocks_credentials:
    file: "${BINANCE_STOCKS_CREDENTIALS_PATH:-./config/binance_stocks_credentials.paper.json}"
'''


def append_env_if_missing(path: Path, key: str, value: str) -> None:
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    if any(line.startswith(f"{key}=") for line in text.splitlines()):
        return
    with path.open("a", encoding="utf-8") as output:
        if text and not text.endswith("\n"):
            output.write("\n")
        output.write(f"{key}={value}\n")


def install(root: Path) -> None:
    compose = root / "docker-compose.yml"
    env_file = root / ".env.control"
    if not compose.exists() or not env_file.exists():
        raise FileNotFoundError("target must contain docker-compose.yml and .env.control")
    unwritable = [str(path) for path in (compose, env_file) if not os.access(path, os.W_OK)]
    if unwritable:
        raise PermissionError(
            "installer requires write access before making any changes: " + ", ".join(unwritable)
        )
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    backup = root / ".deployment-backups" / f"binance-stocks-runtime-{stamp}"
    backup.mkdir(parents=True, exist_ok=False)
    shutil.copy2(compose, backup / compose.name)
    shutil.copy2(env_file, backup / env_file.name)

    text = compose.read_text(encoding="utf-8")
    if "  binance-stocks-runtime:\n" in text:
        start = text.index("  binance-stocks-runtime:\n")
        end_marker = "  hummingbot-mcp:\n"
        end = text.index(end_marker, start)
        text = text[:start] + SERVICE_BLOCK + text[end:]
    else:
        anchor = "  hummingbot-mcp:\n"
        if text.count(anchor) != 1:
            raise RuntimeError("compose service anchor changed; refusing non-audited edit")
        text = text.replace(anchor, SERVICE_BLOCK + anchor, 1)
    if "  binance_stocks_credentials:\n" not in text:
        anchor = "  dca_binance_emergency_credentials:\n"
        if text.count(anchor) != 1:
            raise RuntimeError("compose secrets anchor changed; refusing non-audited edit")
        text = text.replace(anchor, SECRET_BLOCK + anchor, 1)
    temporary = compose.with_suffix(".yml.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(compose)

    append_env_if_missing(env_file, "STOCKS_API_USERNAME", "stocks-admin")
    append_env_if_missing(env_file, "STOCKS_API_PASSWORD", secrets.token_urlsafe(32))
    append_env_if_missing(env_file, "BINANCE_STOCKS_RUNTIME_MODE", "PAPER")
    append_env_if_missing(env_file, "BINANCE_STOCKS_LIVE_AUTHORIZED", "false")
    append_env_if_missing(env_file, "BINANCE_STOCKS_DISCLAIMER_CONFIRMED", "false")
    append_env_if_missing(env_file, "BINANCE_STOCKS_DEFAULT_WHITELIST", "AAPL,TSLA,SPY,QQQ")
    print(f"installed Paper-only service; backup={backup}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    install(args.root.resolve())

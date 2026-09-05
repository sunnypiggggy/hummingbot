#!/usr/bin/env python3
"""Deploy a separate zero-maker-fee FDUSD paper portfolio-grid instance."""

import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

import requests
import yaml


PAIRS = ["BTC-FDUSD", "ETH-FDUSD", "BNB-FDUSD", "SOL-FDUSD", "XRP-FDUSD", "DOGE-FDUSD", "LINK-FDUSD"]
PROFILE = "fdusd_paper"
CONFIG_NAME = "walk_forward_portfolio_grid_fdusd_maker0.yml"


def main() -> None:
    bots = Path(os.getenv("BOTS_PATH", "/workspace/bots"))
    scripts_dir = bots / "scripts"
    config_dir = bots / "conf" / "scripts"
    profile_dir = bots / "credentials" / PROFILE
    for directory in (scripts_dir, config_dir, profile_dir):
        directory.mkdir(parents=True, exist_ok=True)

    shutil.copy2("/app/walk_forward_portfolio_grid.py", scripts_dir / "walk_forward_portfolio_grid.py")
    verification = bots / "credentials" / "master_account" / ".password_verification"
    if not verification.exists():
        raise RuntimeError("master_account is missing .password_verification; initialize the Hummingbot profile first")
    shutil.copy2(verification, profile_dir / ".password_verification")
    balances = {
        "FDUSD": 100000, "BTC": 1, "ETH": 20, "BNB": 100, "SOL": 100,
        "XRP": 100000, "DOGE": 1000000, "LINK": 10000,
    }
    client_config = {
        "instance_id": "api-managed-fdusd-grid", "log_level": "INFO", "db_mode": {"db_engine": "sqlite"},
        "paper_trade": {"paper_trade_exchanges": ["binance"], "paper_trade_account_balance": balances},
        "mqtt_bridge": {"mqtt_host": "127.0.0.1", "mqtt_port": 1883, "mqtt_namespace": "hbot",
                        "mqtt_ssl": False, "mqtt_autostart": False},
    }
    (profile_dir / "conf_client.yml").write_text(yaml.safe_dump(client_config, sort_keys=False), encoding="utf-8")
    # Binance FDUSD pairs are configured as zero maker fee for this isolated
    # paper profile. Taker fee remains aligned with the other paper bots.
    (profile_dir / "conf_fee_overrides.yml").write_text(
        "template_version: 14\nbinance_maker_percent_fee: 0.0\nbinance_taker_percent_fee: 0.02\n",
        encoding="utf-8",
    )
    config = {
        "script_file_name": "walk_forward_portfolio_grid.py", "controllers_config": [],
        "parameter_version": "fdusd-maker0-manual", "exchange": "binance_paper_trade",
        "trading_pairs": PAIRS, "quote_asset": "FDUSD", "grid_range": 0.08, "grid_levels": 24,
        "order_quote_pct": 0.02, "take_profit": 0.003, "move_threshold": 0.005,
        "portfolio_stop_loss": 0.08, "order_refresh_time": 60, "min_grid_move_seconds": 0,
        "cooldown_seconds": 86400, "min_order_quote": 10, "initial_peak_equity": 0,
        "initial_cooldown_until": 0, "initial_grid_states": {},
    }
    (config_dir / CONFIG_NAME).write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    api_url = os.getenv("HUMMINGBOT_API_URL", "http://hummingbot-api:8000").rstrip("/")
    response = requests.post(
        f"{api_url}/bot-orchestration/deploy-v2-script",
        auth=(os.environ["USERNAME"], os.environ["PASSWORD"]), timeout=60,
        json={"instance_name": "walk-forward-portfolio-grid-fdusd", "credentials_profile": PROFILE,
              "image": os.getenv(
                  "PORTFOLIO_RUNTIME_IMAGE", "hummingbot/portfolio-grid-runtime:local"
              ), "script": "walk_forward_portfolio_grid",
              "script_config": CONFIG_NAME, "headless": True},
    )
    response.raise_for_status()
    result = response.json()
    if not result.get("unique_instance_name"):
        raise RuntimeError(f"API did not return an instance name: {result}")
    state_dir = Path("/workspace/state") / "fdusd_grid"
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / "deployment.json").write_text(json.dumps({
        "deployed_at": datetime.now(timezone.utc).isoformat(), "config": config, "api_response": result,
    }, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

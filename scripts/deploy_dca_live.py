#!/usr/bin/env python3
"""Prepare, preflight, and explicitly arm the isolated BTC/ETH live DCA bots."""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import time
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, Iterable

import requests
import yaml

from dca_live_common import (
    ACCOUNT_NAME,
    CONNECTOR,
    LIVE_PAIRS,
    RESERVE_QUOTE,
    extract_balances,
    live_controller_config,
    side_budget,
    validate_config,
    validate_exchange_filters,
)


LOG = logging.getLogger("dca-live-manager")
CONFIRMATION = "LIVE-DCA-200"
BINANCE_API = "https://api.binance.com"


class ApiClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.session.auth = (os.environ["USERNAME"], os.environ["PASSWORD"])

    def request(self, method: str, path: str, payload: Dict[str, Any] | None = None) -> Any:
        response = self.session.request(method, f"{self.base_url}{path}", json=payload, timeout=60)
        if response.status_code >= 400:
            raise RuntimeError(f"{method} {path} failed ({response.status_code}): {response.text[:800]}")
        return response.json() if response.content else {}

    def accounts(self) -> Any:
        return self.request("GET", "/accounts/")

    def add_account(self, account_name: str) -> Any:
        return self.request("POST", f"/accounts/add-account?account_name={account_name}")

    def credentials(self, account_name: str) -> Any:
        return self.request("GET", f"/accounts/{account_name}/credentials")

    def portfolio(self, account_name: str) -> Any:
        return self.request("POST", "/portfolio/state", {
            "account_names": [account_name],
            "connector_names": [CONNECTOR],
        })

    def status(self) -> Any:
        return self.request("GET", "/bot-orchestration/status")

    def active_containers(self) -> Any:
        return self.request("GET", "/docker/active-containers")

    def deploy(self, bot_name: str, config_name: str, account_name: str) -> Any:
        return self.request("POST", "/bot-orchestration/deploy-v2-script?use_timestamp=false", {
            "instance_name": bot_name,
            "credentials_profile": account_name,
            "script": "v2_with_controllers",
            "script_config": f"{bot_name}.yml",
            "image": os.getenv(
                "DCA_LIVE_RUNTIME_IMAGE", "hummingbot/dca-live-runtime:local"
            ),
            "headless": True,
        })


def stage_configs(bots_path: Path) -> list[Path]:
    controller_source = Path(
        os.getenv(
            "DCA_LIVE_CONTROLLER_SOURCE",
            "/app/dman_maker_v3_macro.py",
        )
    )
    if not controller_source.exists():
        controller_source = (
            Path(__file__).resolve().parents[1]
            / "controllers"
            / "market_making"
            / "dman_maker_v3_macro.py"
        )
    if not controller_source.exists():
        raise RuntimeError("dman_maker_v3_macro.py is missing from the deploy image")
    controller_target = (
        bots_path
        / "controllers"
        / "market_making"
        / "dman_maker_v3_macro.py"
    )
    controller_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(controller_source, controller_target)

    target = bots_path / "conf" / "controllers"
    target.mkdir(parents=True, exist_ok=True)
    written = [controller_target]
    for spec in LIVE_PAIRS.values():
        config = live_controller_config(spec.trading_pair)
        validate_config(config)
        path = target / f"{spec.config_name}.yml"
        path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        written.append(path)
        script_path = bots_path / "conf" / "scripts" / f"{spec.bot_name}.yml"
        script_path.parent.mkdir(parents=True, exist_ok=True)
        script_path.write_text(
            yaml.safe_dump(
                {
                    "script_file_name": "v2_with_controllers.py",
                    "controllers_config": [f"{spec.config_name}.yml"],
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        written.append(script_path)
    return written


def public_market_state(pairs: Iterable[str]) -> Dict[str, Decimal]:
    prices: Dict[str, Decimal] = {}
    session = requests.Session()
    for pair in pairs:
        symbol = pair.replace("-", "")
        info = session.get(f"{BINANCE_API}/api/v3/exchangeInfo",
                           params={"symbol": symbol}, timeout=20)
        info.raise_for_status()
        ticker = session.get(f"{BINANCE_API}/api/v3/ticker/price",
                             params={"symbol": symbol}, timeout=20)
        ticker.raise_for_status()
        price = Decimal(str(ticker.json()["price"]))
        validate_exchange_filters(info.json()["symbols"][0], price)
        prices[pair] = price
    return prices


def credentials_include_binance(payload: Any) -> bool:
    return "binance" in json.dumps(payload, ensure_ascii=True).lower()


def status_contains_bot(payload: Any, bot_name: str) -> bool:
    return bot_name in json.dumps(payload, ensure_ascii=True)


def active_container_exists(payload: Any, bot_name: str) -> bool:
    values = payload.get("data", payload) if isinstance(payload, dict) else payload
    return any(
        isinstance(item, dict)
        and str(item.get("name", "")).lstrip("/") == bot_name
        and str(item.get("status", "")).lower() == "running"
        for item in values if isinstance(values, list)
    )


def load_guard_state(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def can_start_eth(state_path: Path, now: float) -> tuple[bool, str]:
    state = load_guard_state(state_path)
    btc = state.get("bots", {}).get(LIVE_PAIRS["BTC-USDT"].bot_name, {})
    if btc.get("tripped"):
        return False, "BTC canary has tripped its circuit breaker."
    started_at = float(btc.get("started_at", 0))
    if not started_at or now - started_at < 86400:
        return False, "BTC canary has not completed 24 hours under guard monitoring."
    latest_at = float(btc.get("latest", {}).get("updated_at", 0))
    if not latest_at or now - latest_at > 60:
        return False, "BTC canary is not under fresh guard monitoring."
    return True, "BTC canary completed 24 hours."


def selected_requirements(pairs: Iterable[str], prices: Dict[str, Decimal]) -> Dict[str, Decimal]:
    selected = list(pairs)
    requirements = {"USDT": (side_budget() + RESERVE_QUOTE) * len(selected)}
    for pair in selected:
        spec = LIVE_PAIRS[pair]
        requirements[spec.base_asset] = side_budget() / prices[pair]
    return requirements


def preflight(api: ApiClient, pairs: list[str], account_name: str) -> Dict[str, Any]:
    credentials = api.credentials(account_name)
    if not credentials_include_binance(credentials):
        raise RuntimeError(f"Account {account_name} has no encrypted Binance credential.")
    prices = public_market_state(pairs)
    balances = extract_balances(api.portfolio(account_name), account_name, CONNECTOR)
    requirements = selected_requirements(pairs, prices)
    missing = {
        asset: str(required - balances.get(asset, Decimal("0")))
        for asset, required in requirements.items()
        if balances.get(asset, Decimal("0")) < required
    }
    if missing:
        raise RuntimeError(f"Insufficient live balances: {missing}")
    return {
        "account": account_name,
        "pairs": pairs,
        "prices": {pair: str(value) for pair, value in prices.items()},
        "balances": {asset: str(balances.get(asset, Decimal("0"))) for asset in requirements},
        "requirements": {asset: str(value) for asset, value in requirements.items()},
        "layer_quote_amounts_per_side": ["9.5", "19", "28.5", "38"],
        "checked_at": datetime.now(timezone.utc).isoformat(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", choices=["BTC-USDT", "ETH-USDT", "all"], default="BTC-USDT")
    parser.add_argument("--prepare-profile", action="store_true")
    parser.add_argument("--arm", action="store_true")
    parser.add_argument("--confirm", default="")
    parser.add_argument("--override-canary", action="store_true")
    parser.add_argument("--bots-path", default=os.getenv("BOTS_PATH", "/workspace/bots"))
    parser.add_argument("--state-path", default=os.getenv("DCA_LIVE_STATE_PATH", "/workspace/state"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    staged = stage_configs(Path(args.bots_path))
    LOG.info("Staged isolated live configs: %s", ", ".join(str(path) for path in staged))
    api = ApiClient(os.getenv("HUMMINGBOT_API_URL", "http://hummingbot-api:8000"))
    accounts_payload = api.accounts()
    if ACCOUNT_NAME not in json.dumps(accounts_payload):
        if args.prepare_profile:
            api.add_account(ACCOUNT_NAME)
            LOG.info("Created empty encrypted credential profile %s", ACCOUNT_NAME)
        else:
            raise RuntimeError(f"Credential profile {ACCOUNT_NAME} is absent; run with --prepare-profile first.")

    pairs = list(LIVE_PAIRS) if args.pair == "all" else [args.pair]
    report = preflight(api, pairs, ACCOUNT_NAME)
    state_dir = Path(args.state_path)
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / "latest_preflight.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    ownership_path = state_dir / "managed_inventory.json"
    ownership = (
        json.loads(ownership_path.read_text(encoding="utf-8"))
        if ownership_path.exists() else {"schema_version": 1, "account": ACCOUNT_NAME, "pairs": {}}
    )
    if ownership.get("account") != ACCOUNT_NAME:
        raise RuntimeError("managed inventory account mismatch")
    for pair in pairs:
        asset = LIVE_PAIRS[pair].base_asset
        ownership["pairs"][pair] = {
            "base_asset": asset,
            "managed_base": report["requirements"][asset],
            "verified_at": report["checked_at"],
        }
    temporary = ownership_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(ownership, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(ownership_path)
    LOG.info("Live preflight passed: %s", json.dumps(report, ensure_ascii=False))
    if not args.arm:
        LOG.info("Dry-run complete. No live bot was deployed.")
        return 0
    if os.getenv("DCA_LIVE_TRADING_ENABLED", "false").lower() != "true":
        raise RuntimeError("DCA_LIVE_TRADING_ENABLED is not true; refusing live deployment.")
    if args.confirm != CONFIRMATION:
        raise RuntimeError(f"Explicit --confirm {CONFIRMATION} is required.")

    active_containers = api.active_containers()
    for pair in pairs:
        spec = LIVE_PAIRS[pair]
        if active_container_exists(active_containers, spec.bot_name):
            raise RuntimeError(f"Bot {spec.bot_name} already exists; refusing a duplicate deployment.")
        if pair == "ETH-USDT" and not args.override_canary:
            allowed, reason = can_start_eth(state_dir / "guard_state.json", time.time())
            if not allowed:
                raise RuntimeError(reason)
        response = api.deploy(spec.bot_name, spec.config_name, ACCOUNT_NAME)
        if not response.get("success") or response.get("unique_instance_name") != spec.bot_name:
            raise RuntimeError(
                f"Fixed-name deployment failed for {spec.bot_name}: {response}"
            )
        LOG.warning("LIVE BOT DEPLOYED: %s response=%s", spec.bot_name, response)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

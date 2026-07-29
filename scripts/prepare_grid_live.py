#!/usr/bin/env python3
"""Generate a non-deploying live-grid preflight and capital reservation manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict

import requests
import yaml

from grid_live_common import (
    MIN_ORDER_QUOTE,
    PORTFOLIOS,
    SIDE_BUDGET,
    build_live_config,
    required_balances,
    validate_exchange_filters,
    validate_live_config,
)


BINANCE_API = "https://api.binance.com"


class ApiClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.session.auth = (os.environ["USERNAME"], os.environ["PASSWORD"])

    def request(self, method: str, path: str, payload: Dict[str, Any] | None = None) -> Any:
        response = self.session.request(method, f"{self.base_url}{path}", json=payload, timeout=30)
        if response.status_code >= 400:
            raise RuntimeError(f"{method} {path} failed ({response.status_code}): {response.text[:500]}")
        return response.json() if response.content else {}

    def credentials(self, profile: str) -> Any:
        return self.request("GET", f"/accounts/{profile}/credentials")

    def portfolio(self, profiles: list[str]) -> Any:
        return self.request("POST", "/portfolio/state", {
            "account_names": profiles, "connector_names": ["binance"],
        })


def public_state() -> tuple[Dict[str, Decimal], Dict[str, Any]]:
    prices, rules = {}, {}
    session = requests.Session()
    pairs = [pair for portfolio in PORTFOLIOS.values() for pair in portfolio.pairs]
    for pair in pairs:
        symbol = pair.replace("-", "")
        info = session.get(f"{BINANCE_API}/api/v3/exchangeInfo", params={"symbol": symbol}, timeout=20)
        ticker = session.get(f"{BINANCE_API}/api/v3/ticker/price", params={"symbol": symbol}, timeout=20)
        info.raise_for_status()
        ticker.raise_for_status()
        rules[pair] = info.json()["symbols"][0]
        prices[pair] = Decimal(str(ticker.json()["price"]))
        validate_exchange_filters(rules[pair], SIDE_BUDGET / Decimal("5"))
    return prices, rules


def credential_has_binance(payload: Any) -> bool:
    return "binance" in json.dumps(payload, ensure_ascii=True).lower()


def flatten_balances(payload: Any) -> Dict[str, Decimal]:
    """Use the first profile because both profiles point at the same main account."""
    balances: Dict[str, Decimal] = {}
    text_nodes = payload.values() if isinstance(payload, dict) else []
    for account in text_nodes:
        node = account.get("binance", account) if isinstance(account, dict) else {}
        node = node.get("balances", node) if isinstance(node, dict) else {}
        if not isinstance(node, dict):
            continue
        for asset, raw in node.items():
            raw = raw.get("total", raw.get("units", raw.get("balance", 0))) if isinstance(raw, dict) else raw
            try:
                balances[str(asset).upper()] = Decimal(str(raw))
            except Exception:
                pass
        if balances:
            break
    return balances


def main() -> int:
    parser = argparse.ArgumentParser(description="Dry-run only; this command cannot deploy a bot.")
    parser.add_argument("--output-dir", type=Path, default=Path("results/backtests/grid_live_validation_500"))
    parser.add_argument("--usdt-maker-fee", type=Decimal, default=Decimal("0.001"))
    parser.add_argument("--fdusd-maker-fee", type=Decimal, default=Decimal("0"))
    parser.add_argument("--private-api-check", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prices, rules = public_state()
    fees = {"USDT": args.usdt_maker_fee, "FDUSD": args.fdusd_maker_fee}
    configs, hashes = {}, {}
    for key, portfolio in PORTFOLIOS.items():
        config = build_live_config(portfolio, prices, fees[key], trading_enabled=False)
        validate_live_config(config)
        serialized = yaml.safe_dump(config, sort_keys=False)
        path = args.output_dir / portfolio.config_name
        path.write_text(serialized, encoding="utf-8")
        configs[key] = str(path)
        hashes[key] = hashlib.sha256(path.read_bytes()).hexdigest()

    required = required_balances(prices)
    private_status: Dict[str, Any] = {"complete": False, "reason": "not requested"}
    if args.private_api_check:
        api = ApiClient(os.getenv("HUMMINGBOT_API_URL", "http://hummingbot-api:8000"))
        profiles = [portfolio.profile_name for portfolio in PORTFOLIOS.values()]
        missing_credentials = [profile for profile in profiles if not credential_has_binance(api.credentials(profile))]
        if missing_credentials:
            raise RuntimeError(f"Profiles without encrypted Binance credentials: {missing_credentials}")
        balances = flatten_balances(api.portfolio(profiles))
        missing = {asset: str(amount - balances.get(asset, Decimal("0")))
                   for asset, amount in required.items() if balances.get(asset, Decimal("0")) < amount}
        if missing:
            raise RuntimeError(f"Main-account balances do not cover global reservations: {missing}")
        private_status = {
            "complete": False,
            "credentials": "present",
            "balances": {asset: str(balances.get(asset, Decimal("0"))) for asset in required},
            "test_order": "pending: no real-order endpoint is called by this dry-run tool",
            "commission": "must be captured from the signed Binance commission endpoint before arming",
        }

    manifest = {
        "version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "deployment_allowed": False,
        "trading_enabled": False,
        "connector": "binance",
        "shared_main_account": True,
        "profiles": {key: portfolio.profile_name for key, portfolio in PORTFOLIOS.items()},
        "bots": {key: portfolio.bot_name for key, portfolio in PORTFOLIOS.items()},
        "configs": configs,
        "config_sha256": hashes,
        "prices": {pair: str(price) for pair, price in prices.items()},
        "global_required_balances": {asset: str(amount) for asset, amount in required.items()},
        "reservations": {
            key: {
                "quote": str(SIDE_BUDGET * 2 + Decimal("25")),
                "base": {pair.split("-")[0]: str(SIDE_BUDGET / prices[pair]) for pair in portfolio.pairs},
            } for key, portfolio in PORTFOLIOS.items()
        },
        "minimum_order_quote": str(MIN_ORDER_QUOTE),
        "private_preflight": private_status,
        "next_step": "Review validation_result.md and report.html. This manifest cannot deploy.",
    }
    output = args.output_dir / "capital_reservations.json"
    output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

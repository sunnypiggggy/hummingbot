#!/usr/bin/env python3
"""Prepare the FDUSD quote-only live grid and perform non-trading safety checks."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict

import requests

from grid_live_common import (
    PORTFOLIOS,
    active_portfolio_pairs,
    budget_for_live_pairs,
    extract_balances,
    validate_exchange_filters,
)


BINANCE_API = "https://api.binance.com"
MAX_BOOTSTRAP_SLIPPAGE = Decimal("0.001")


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

    def portfolio(self, profile: str) -> Any:
        return self.request("POST", "/portfolio/state", {
            "account_names": [profile],
            "connector_names": ["binance"],
        })


def weighted_ask(asks: list[list[str]], quote_amount: Decimal) -> tuple[Decimal, Decimal, Decimal]:
    if not asks:
        raise ValueError("Order book has no asks.")
    best = Decimal(str(asks[0][0]))
    remaining = quote_amount
    base_amount = Decimal("0")
    spent = Decimal("0")
    for raw_price, raw_amount in asks:
        price = Decimal(str(raw_price))
        available_base = Decimal(str(raw_amount))
        available_quote = price * available_base
        take_quote = min(remaining, available_quote)
        base_amount += take_quote / price
        spent += take_quote
        remaining -= take_quote
        if remaining <= 0:
            break
    if remaining > Decimal("0.000001"):
        raise ValueError(f"Order book depth is short by {remaining} quote units.")
    average = spent / base_amount
    return average, (average / best) - Decimal("1"), base_amount


def public_checks() -> dict:
    session = requests.Session()
    result = {}
    portfolio = PORTFOLIOS["FDUSD"]
    pairs = active_portfolio_pairs(portfolio)
    budget = budget_for_live_pairs("FDUSD", pairs)
    for pair in pairs:
        symbol = pair.replace("-", "")
        info = session.get(f"{BINANCE_API}/api/v3/exchangeInfo",
                           params={"symbol": symbol}, timeout=20)
        depth = session.get(f"{BINANCE_API}/api/v3/depth",
                            params={"symbol": symbol, "limit": 100}, timeout=20)
        info.raise_for_status()
        depth.raise_for_status()
        symbol_info = info.json()["symbols"][0]
        validate_exchange_filters(symbol_info, budget.side_budget)
        average, slippage, estimated_base = weighted_ask(
            depth.json()["asks"], budget.side_budget
        )
        result[pair] = {
            "average_ask": str(average),
            "estimated_slippage": str(slippage),
            "estimated_base": str(estimated_base),
            "slippage_passed": slippage <= MAX_BOOTSTRAP_SLIPPAGE,
            "status": symbol_info["status"],
        }
    return result


def load_receipt(path: Path | None, name: str) -> dict:
    if path is None:
        return {"complete": False, "reason": f"{name} receipt was not supplied"}
    payload = json.loads(path.read_text(encoding="utf-8"))
    checked_at = int(payload.get("checked_at_ts", 0))
    if time.time() - checked_at > 86400:
        raise ValueError(f"{name} receipt is older than 24 hours.")
    return payload


def validate_receipts(commission: dict, permissions: dict, test_order: dict) -> dict:
    profile = PORTFOLIOS["FDUSD"].profile_name
    pairs = active_portfolio_pairs(PORTFOLIOS["FDUSD"])
    commission_ok = (
        commission.get("profile") == profile
        and Decimal(str(commission.get("maker_fee", "-1"))) >= 0
        and Decimal(str(commission.get("taker_fee", "-1"))) >= 0
        and bool(commission.get("signed_endpoint_verified"))
    )
    permissions_ok = (
        permissions.get("profile") == profile
        and bool(permissions.get("read_enabled"))
        and bool(permissions.get("spot_trading_enabled"))
        and bool(permissions.get("ip_restricted"))
        and not bool(permissions.get("withdraw_enabled"))
        and not bool(permissions.get("futures_enabled"))
        and not bool(permissions.get("margin_enabled"))
    )
    test_order_ok = (
        test_order.get("profile") == profile
        and set(test_order.get("pairs", [])) == set(pairs)
        and bool(test_order.get("passed"))
        and bool(test_order.get("no_fill"))
    )
    return {
        "commission_verified": commission_ok,
        "permissions_verified": permissions_ok,
        "test_order_verified": test_order_ok,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="This command never places an order or deploys a bot.")
    parser.add_argument("--validation-dir", type=Path,
                        default=Path("results/backtests/grid_live_fdusd_400_walk_forward"))
    parser.add_argument("--state-dir", type=Path, default=Path("grid-live-fdusd-data"))
    parser.add_argument("--bots-path", type=Path)
    parser.add_argument("--private-api-check", action="store_true")
    parser.add_argument("--commission-receipt", type=Path)
    parser.add_argument("--permission-receipt", type=Path)
    parser.add_argument("--test-order-receipt", type=Path)
    args = parser.parse_args()
    args.state_dir.mkdir(parents=True, exist_ok=True)
    pairs = active_portfolio_pairs(PORTFOLIOS["FDUSD"])
    budget = budget_for_live_pairs("FDUSD", pairs)

    validation = json.loads((args.validation_dir / "validation_result.json").read_text(encoding="utf-8"))
    quantitative_go = all(
        value for key, value in validation["gates"].items() if key != "private_fee_verified"
    )
    public = public_checks()
    public_ok = all(item["slippage_passed"] for item in public.values())
    profile = PORTFOLIOS["FDUSD"].profile_name
    account_status: dict = {"credentials_present": False, "fdusd_balance": "unknown"}
    if args.private_api_check:
        api = ApiClient(os.getenv("HUMMINGBOT_API_URL", "http://hummingbot-api:8000"))
        credentials = api.credentials(profile)
        account_status["credentials_present"] = "binance" in json.dumps(
            credentials, ensure_ascii=True
        ).lower()
        balances = extract_balances(api.portfolio(profile))
        account_status["fdusd_balance"] = str(balances.get("FDUSD", Decimal("0")))
        account_status["balance_passed"] = (
            balances.get("FDUSD", Decimal("0")) >= budget.capital_limit
        )

    commission = load_receipt(args.commission_receipt, "commission")
    permissions = load_receipt(args.permission_receipt, "permissions")
    test_order = load_receipt(args.test_order_receipt, "test-order")
    receipt_status = validate_receipts(commission, permissions, test_order)
    fee_matches_validation = False
    if receipt_status["commission_verified"]:
        fee_matches_validation = (
            Decimal(str(commission["maker_fee"]))
            == Decimal(str(validation.get("maker_fee_used", "-1")))
            and Decimal(str(commission["taker_fee"]))
            == Decimal(str(validation.get("taker_fee_used", "-1")))
        )
    receipt_status["validation_fee_matches"] = fee_matches_validation
    private_complete = (
        bool(account_status.get("credentials_present"))
        and bool(account_status.get("balance_passed"))
        and all(receipt_status.values())
    )
    eligible = quantitative_go and public_ok and private_complete
    maker_fee = str(commission.get("maker_fee", "unknown"))
    taker_fee = str(commission.get("taker_fee", "unknown"))
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "fetched_at": int(time.time()),
        "profile": profile,
        "pairs": list(pairs),
        "required_fdusd": str(budget.capital_limit),
        "recommended_fdusd": str(budget.recommended_balance),
        "bootstrap_quote_per_pair": str(budget.side_budget),
        "maximum_bootstrap_slippage": str(MAX_BOOTSTRAP_SLIPPAGE),
        "public_checks": public,
        "account": account_status,
        "receipt_status": receipt_status,
        "maker_fee": maker_fee,
        "taker_fee": taker_fee,
        "private_preflight_complete": private_complete,
        "quantitative_validation_passed": quantitative_go,
        "eligible_for_manual_approval": eligible,
        "deployment_allowed": False,
        "trading_enabled": False,
        "reason": (
            "All checks passed; rerun the 182-day validation with the verified fee before manual approval."
            if eligible
            else "One or more quantitative, public-market, credential, balance, permission, or test-order gates failed."
        ),
    }
    target = args.state_dir / "private_preflight.json"
    temporary = target.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2), encoding="utf-8")
    temporary.replace(target)

    for name in (
        PORTFOLIOS["FDUSD"].config_name,
        "active_selection.json",
        "capital_reservations.json",
        "validation_result.json",
    ):
        source = args.validation_dir / name
        if source.exists():
            shutil.copy2(source, args.state_dir / name)
    if args.bots_path:
        scripts = args.bots_path / "scripts"
        configs = args.bots_path / "conf" / "scripts"
        scripts.mkdir(parents=True, exist_ok=True)
        configs.mkdir(parents=True, exist_ok=True)
        shutil.copy2(Path(__file__).with_name("walk_forward_portfolio_grid_live.py"),
                     scripts / "walk_forward_portfolio_grid_live.py")
        shutil.copy2(Path(__file__).with_name("grid_live_common.py"),
                     scripts / "grid_live_common.py")
        shutil.copy2(Path(__file__).with_name("grid_macro_gate.py"),
                     scripts / "grid_macro_gate.py")
        shutil.copy2(Path(__file__).with_name("grid_technical_gate.py"),
                     scripts / "grid_technical_gate.py")
        shutil.copy2(args.validation_dir / PORTFOLIOS["FDUSD"].config_name,
                     configs / PORTFOLIOS["FDUSD"].config_name)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if eligible else 2


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Explicitly bootstrap and deploy the FDUSD live grid after every gate passes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
from datetime import datetime, timezone
from decimal import Decimal, ROUND_DOWN
from pathlib import Path
from typing import Any, Dict

import requests
import yaml

from grid_live_common import (
    CONFIRMATION,
    FDUSD_BUDGET,
    PORTFOLIOS,
    build_live_config,
    extract_balances,
    validate_live_config,
)
from grid_macro_gate import load_runtime_macro_gate
from grid_technical_gate import load_runtime_technical_gate
from prepare_fdusd_live_grid import MAX_BOOTSTRAP_SLIPPAGE, weighted_ask


BINANCE_API = "https://api.binance.com"
NO_GO_OVERRIDE_CONFIRMATION = "ACCEPT-NO-GO-GRID-FDUSD-400"


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

    def portfolio(self, profile: str) -> Dict[str, Decimal]:
        payload = self.request("POST", "/portfolio/state", {
            "account_names": [profile],
            "connector_names": ["binance"],
        })
        return extract_balances(payload)

    def status(self) -> Any:
        return self.request("GET", "/bot-orchestration/status")

    def active_containers(self) -> Any:
        return self.request("GET", "/docker/active-containers")

    def market_buy(self, profile: str, pair: str, amount: Decimal) -> Any:
        return self.request("POST", "/trading/orders", {
            "account_name": profile,
            "connector_name": "binance",
            "trading_pair": pair,
            "trade_type": "BUY",
            "amount": str(amount),
            "order_type": "MARKET",
            "position_action": "OPEN",
        })

    def deploy(self, profile: str, config_name: str) -> Any:
        return self.request("POST", "/bot-orchestration/deploy-v2-script?use_timestamp=false", {
            "instance_name": PORTFOLIOS["FDUSD"].bot_name,
            "credentials_profile": profile,
            "image": "hummingbot/hummingbot:latest",
            "script": "walk_forward_portfolio_grid_live",
            "script_config": config_name,
            "headless": True,
        })


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def symbol_plan(pair: str) -> dict:
    session = requests.Session()
    symbol = pair.replace("-", "")
    info_response = session.get(
        f"{BINANCE_API}/api/v3/exchangeInfo", params={"symbol": symbol}, timeout=20
    )
    depth_response = session.get(
        f"{BINANCE_API}/api/v3/depth", params={"symbol": symbol, "limit": 100}, timeout=20
    )
    info_response.raise_for_status()
    depth_response.raise_for_status()
    info = info_response.json()["symbols"][0]
    average, slippage, estimated_base = weighted_ask(
        depth_response.json()["asks"], FDUSD_BUDGET.side_budget
    )
    if slippage > MAX_BOOTSTRAP_SLIPPAGE:
        raise RuntimeError(f"{pair} estimated bootstrap slippage {slippage:.4%} exceeds 0.1%.")
    filters = {item["filterType"]: item for item in info["filters"]}
    step = Decimal(str(filters["LOT_SIZE"]["stepSize"]))
    amount = (estimated_base / step).to_integral_value(rounding=ROUND_DOWN) * step
    if amount <= 0:
        raise RuntimeError(f"{pair} bootstrap amount rounds to zero.")
    return {
        "pair": pair,
        "base": pair.split("-")[0],
        "amount": amount,
        "average_ask": average,
        "estimated_slippage": slippage,
    }


def wait_for_base_delta(api: ApiClient, profile: str, base: str, before: Decimal,
                        expected: Decimal, timeout: int = 90) -> Decimal:
    deadline = time.time() + timeout
    last = Decimal("0")
    while time.time() < deadline:
        balances = api.portfolio(profile)
        last = balances.get(base, Decimal("0")) - before
        if last >= expected * Decimal("0.98"):
            return last
        time.sleep(3)
    raise RuntimeError(f"{base} bootstrap fill is incomplete: received {last}, expected {expected}.")


def guard_readiness(state: Dict[str, Any], containers: Any,
                    now: float | None = None) -> Dict[str, bool]:
    now = time.time() if now is None else now
    last_success = float(state.get("last_success_at", 0))
    age = now - last_success
    return {
        "guard_container_running": "grid-live-guard" in json.dumps(
            containers, ensure_ascii=True
        ),
        "guard_emergency_ready": bool(state.get("emergency_ready")),
        "guard_observation_fresh": 0 <= age <= 30,
        "guard_armed": bool(state.get("armed")),
        "guard_not_shadow": not bool(state.get("shadow", True)),
    }


def validation_authorization(
    validation: Dict[str, Any], *, accept_no_go: bool, override_confirmation: str,
) -> tuple[bool, bool]:
    decision = str(validation.get("validation_decision", ""))
    override = bool(
        decision == "NO-GO"
        and accept_no_go
        and override_confirmation == NO_GO_OVERRIDE_CONFIRMATION
    )
    return decision == "CONDITIONAL GO" or override, override


def load_bootstrap_receipt(path: Path, profile: str) -> tuple[dict, str]:
    raw = path.read_bytes()
    payload = json.loads(raw)
    if payload.get("schema_version") != "fdusd-bootstrap-receipt-v1":
        raise ValueError("unsupported FDUSD bootstrap receipt schema")
    if payload.get("source") != "binance-signed-api":
        raise ValueError("bootstrap receipt must come from the Binance signed API")
    if payload.get("profile") != profile:
        raise ValueError("bootstrap receipt belongs to another profile")
    orders = payload.get("orders", {})
    if set(orders) != set(PORTFOLIOS["FDUSD"].pairs):
        raise ValueError("bootstrap receipt must contain exactly both FDUSD pairs")
    for pair, order in orders.items():
        executed = Decimal(str(order.get("executed_base", "0")))
        quote = Decimal(str(order.get("quote_spent", "0")))
        if order.get("side") != "BUY" or order.get("status") != "FILLED":
            raise ValueError(f"{pair} bootstrap receipt is not a filled BUY")
        if not str(order.get("order_id", "")).strip() or executed <= 0:
            raise ValueError(f"{pair} bootstrap receipt lacks an executed order")
        if not Decimal("95") <= quote <= Decimal("105"):
            raise ValueError(f"{pair} bootstrap quote is outside the approved 100 FDUSD band")
    return payload, hashlib.sha256(raw).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-dir", type=Path, default=Path("grid-live-fdusd-data"))
    parser.add_argument("--bots-path", type=Path, default=Path(os.getenv("BOTS_PATH", "/workspace/bots")))
    parser.add_argument("--arm", action="store_true")
    parser.add_argument("--confirm", default="")
    parser.add_argument("--accept-no-go", action="store_true")
    parser.add_argument("--no-go-confirm", default="")
    parser.add_argument("--bootstrap-receipt", type=Path)
    args = parser.parse_args()

    validation = json.loads((args.state_dir / "validation_result.json").read_text(encoding="utf-8"))
    preflight = json.loads((args.state_dir / "private_preflight.json").read_text(encoding="utf-8"))
    selection = json.loads((args.state_dir / "active_selection.json").read_text(encoding="utf-8"))
    macro_gate = load_runtime_macro_gate(args.state_dir / "macro_gate.json")
    technical_gate = load_runtime_technical_gate(
        args.state_dir / "technical_buy_gate.json"
    )
    guard_state_path = args.state_dir / "guard_state.json"
    guard_state = (
        json.loads(guard_state_path.read_text(encoding="utf-8"))
        if guard_state_path.exists()
        else {}
    )
    api = ApiClient(os.getenv("HUMMINGBOT_API_URL", "http://hummingbot-api:8000"))
    guard_checks = guard_readiness(guard_state, api.active_containers())
    validation_allowed, validation_override = validation_authorization(
        validation,
        accept_no_go=args.accept_no_go,
        override_confirmation=args.no_go_confirm,
    )
    profile = PORTFOLIOS["FDUSD"].profile_name
    receipt = None
    receipt_sha256 = None
    if args.bootstrap_receipt is not None:
        receipt, receipt_sha256 = load_bootstrap_receipt(args.bootstrap_receipt, profile)
    checks = {
        "quantitative_validation_or_explicit_override": validation_allowed,
        "private_preflight": bool(preflight.get("private_preflight_complete")),
        "manual_approval_eligible": bool(preflight.get("eligible_for_manual_approval")),
        "environment_enabled": os.getenv("GRID_LIVE_TRADING_ENABLED", "false").lower() == "true",
        "confirmation": args.confirm == CONFIRMATION,
        "fomc_gate_healthy": bool(macro_gate.get("runtime_gate_healthy")),
        "fomc_execution_enabled": bool(macro_gate.get("execution_enabled")),
        "outside_fomc_pause_window": not bool(macro_gate.get("pause_new_orders")),
        "technical_gate_healthy": bool(technical_gate.get("runtime_gate_healthy")),
        "technical_buy_enabled": bool(technical_gate.get("buy_enabled")),
        **guard_checks,
    }
    print(json.dumps({
        "dry_run": not args.arm,
        "checks": checks,
        "validation_decision": validation.get("validation_decision"),
        "validation_override": validation_override,
        "bootstrap_receipt_sha256": receipt_sha256,
    }, indent=2))
    if not args.arm:
        return 0
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(f"Live FDUSD deployment is blocked by: {failed}")

    if PORTFOLIOS["FDUSD"].bot_name in json.dumps(api.status(), ensure_ascii=True):
        raise RuntimeError("The FDUSD live-grid bot already exists; duplicate deployment is refused.")
    balances_before = api.portfolio(profile)
    acquired_quote = sum(
        (Decimal(str(order["quote_spent"])) for order in receipt["orders"].values()),
        Decimal("0"),
    ) if receipt else Decimal("0")
    if balances_before.get("FDUSD", Decimal("0")) + acquired_quote < FDUSD_BUDGET.capital_limit:
        raise RuntimeError(
            f"The account has less than the required {FDUSD_BUDGET.capital_limit} FDUSD."
        )

    authorization = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "confirmation": args.confirm,
        "consumed": False,
        "trading_enabled": True,
        "validation_decision": validation.get("validation_decision"),
        "validation_override": validation_override,
        "validation_override_confirmation": (
            NO_GO_OVERRIDE_CONFIRMATION if validation_override else None
        ),
        "bootstrap_receipt_sha256": receipt_sha256,
    }
    authorization_path = args.state_dir / "startup_authorization.json"
    atomic_json(authorization_path, authorization)
    bootstrap = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "profile": profile,
        "quote_budget_per_pair": str(FDUSD_BUDGET.side_budget),
        "orders": {},
        "completed": False,
        "bot_deployed": False,
    }
    bootstrap_path = args.state_dir / "bootstrap_state.json"
    atomic_json(bootstrap_path, bootstrap)

    reservations: Dict[str, Decimal] = {}
    plans = {pair: symbol_plan(pair) for pair in PORTFOLIOS["FDUSD"].pairs}
    prices: Dict[str, Decimal] = {}
    if receipt:
        for pair, order in receipt["orders"].items():
            executed = Decimal(str(order["executed_base"]))
            quote = Decimal(str(order["quote_spent"]))
            reservations[pair] = executed
            prices[pair] = quote / executed
            bootstrap["orders"][pair] = {**order, "source": receipt["source"]}
        bootstrap["bootstrap_receipt_sha256"] = receipt_sha256
        atomic_json(bootstrap_path, bootstrap)
    try:
        if receipt:
            pass
        else:
            for pair in PORTFOLIOS["FDUSD"].pairs:
                plan = plans[pair]
                before_base = api.portfolio(profile).get(plan["base"], Decimal("0"))
                response = api.market_buy(profile, pair, plan["amount"])
                bootstrap["orders"][pair] = {
                    "requested_base": str(plan["amount"]),
                    "estimated_slippage": str(plan["estimated_slippage"]),
                    "estimated_average_ask": str(plan["average_ask"]),
                    "api_response": response,
                    "confirmation_status": "submitted_waiting_for_balance_delta",
                }
                atomic_json(bootstrap_path, bootstrap)
                actual_delta = wait_for_base_delta(
                    api, profile, plan["base"], before_base, plan["amount"]
                )
                reservations[pair] = actual_delta
                prices[pair] = Decimal(str(plan["average_ask"]))
                bootstrap["orders"][pair].update({
                    "actual_base_delta": str(actual_delta),
                    "confirmation_status": "confirmed",
                })
                atomic_json(bootstrap_path, bootstrap)
    except Exception as exc:
        bootstrap["failure"] = str(exc)
        bootstrap["manual_action_required"] = (
            "Keep the acquired assets, do not start Grid, and inspect Binance plus the audit record."
        )
        atomic_json(bootstrap_path, bootstrap)
        raise

    maker_fee = Decimal(str(preflight["maker_fee"]))
    if not prices:
        prices = {pair: Decimal(str(plans[pair]["average_ask"]))
                  for pair in PORTFOLIOS["FDUSD"].pairs}
    config = build_live_config(
        PORTFOLIOS["FDUSD"],
        prices,
        maker_fee,
        trading_enabled=True,
        reserved_base_by_pair=reservations,
        bootstrap_from_quote=True,
        bootstrap_completed=True,
    )
    params = selection["parameters"]
    config.update({
        "grid_range": float(Decimal(str(params["half_range"])) * Decimal("2")),
        "grid_levels": int(params.get("levels") or config["grid_levels"]),
        "take_profit": float(max(
            Decimal(str(params["take_profit"])), maker_fee * Decimal("2") + Decimal("0.004")
        )),
        "move_threshold": float(params["move_threshold"]),
        "min_grid_move_seconds": int(params["min_grid_move_seconds"]),
        "active_parameter_version": selection["parameter_version"],
    })
    validate_live_config(config)
    scripts_dir = args.bots_path / "scripts"
    configs_dir = args.bots_path / "conf" / "scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    configs_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(__file__).with_name("walk_forward_portfolio_grid_live.py"),
                 scripts_dir / "walk_forward_portfolio_grid_live.py")
    shutil.copy2(Path(__file__).with_name("grid_live_common.py"),
                 scripts_dir / "grid_live_common.py")
    shutil.copy2(Path(__file__).with_name("grid_macro_gate.py"),
                 scripts_dir / "grid_macro_gate.py")
    shutil.copy2(Path(__file__).with_name("grid_technical_gate.py"),
                 scripts_dir / "grid_technical_gate.py")
    config_path = configs_dir / PORTFOLIOS["FDUSD"].config_name
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    atomic_json(args.state_dir / "capital_reservations.json", {
        "version": 3,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "deployment_allowed": True,
        "trading_enabled": True,
        "bootstrap_completed": True,
        "profiles": {"FDUSD": profile},
        "bots": {"FDUSD": PORTFOLIOS["FDUSD"].bot_name},
        "prices": {pair: str(price) for pair, price in prices.items()},
        "reservations": {
            "FDUSD": {
                "quote": str(
                    FDUSD_BUDGET.capital_limit - FDUSD_BUDGET.side_budget * Decimal("2")
                ),
                "base": {
                    pair.split("-")[0]: str(amount)
                    for pair, amount in reservations.items()
                },
            },
        },
        "config": str(config_path),
    })
    response = api.deploy(profile, config_path.name)
    if not response.get("unique_instance_name"):
        raise RuntimeError(f"API did not return an instance name: {response}")
    bootstrap.update({
        "completed": True,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "reservations": {pair: str(value) for pair, value in reservations.items()},
        "bot_deployed": True,
        "api_response": response,
    })
    atomic_json(bootstrap_path, bootstrap)
    authorization["consumed"] = True
    authorization["consumed_at"] = datetime.now(timezone.utc).isoformat()
    atomic_json(authorization_path, authorization)
    print(json.dumps(response, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

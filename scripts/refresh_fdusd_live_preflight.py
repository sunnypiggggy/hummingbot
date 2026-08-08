#!/usr/bin/env python3
"""Build an auditable FDUSD live preflight from the current shadow Guard state."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

from deploy_fdusd_live_grid import ApiClient, symbol_plan
from grid_live_common import FDUSD_BUDGET, FDUSD_RECOMMENDED_BALANCE, PORTFOLIOS
from grid_macro_gate import load_runtime_macro_gate
from grid_xgboost_risk_gate import load_runtime_xgboost_gate


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-dir", type=Path, default=Path("/workspace/state"))
    args = parser.parse_args()
    state = json.loads((args.state_dir / "guard_state.json").read_text(encoding="utf-8"))
    validation = json.loads((args.state_dir / "validation_result.json").read_text(encoding="utf-8"))
    shadow = state.get("shadow_preflight", {})
    pairs = list(PORTFOLIOS["FDUSD"].pairs)
    api = ApiClient("http://hummingbot-api:8000")
    balances = api.portfolio(PORTFOLIOS["FDUSD"].profile_name)
    plans = {pair: symbol_plan(pair) for pair in pairs}
    commissions = shadow.get("commissions", {})
    fees_complete = all(
        pair in commissions
        and Decimal(str(commissions[pair]["maker_fee"])) >= 0
        and Decimal(str(commissions[pair]["taker_fee"])) >= 0
        for pair in pairs
    )
    maker_fee = max(
        (Decimal(str(commissions[pair]["maker_fee"])) for pair in pairs),
        default=Decimal("-1"),
    )
    taker_fee = max(
        (Decimal(str(commissions[pair]["taker_fee"])) for pair in pairs),
        default=Decimal("-1"),
    )
    permissions_ok = all(bool(shadow.get(field)) for field in (
        "account_read", "spot_trading", "withdrawals_disabled", "ip_restricted",
        "futures_disabled", "margin_disabled", "test_order_no_fill",
    ))
    no_orders = all(int(shadow.get("open_order_counts", {}).get(pair, -1)) == 0 for pair in pairs)
    balance = balances.get("FDUSD", Decimal("0"))
    macro = load_runtime_macro_gate(args.state_dir / "macro_gate.json")
    technical = load_runtime_xgboost_gate(args.state_dir / "xgboost_risk_gate.json")
    checks = {
        "balance_at_least_420_fdusd": balance >= FDUSD_BUDGET.capital_limit,
        "permissions_and_test_orders": permissions_ok,
        "no_existing_pair_orders": no_orders,
        "commission_verified": fees_complete,
        "public_slippage_within_0_1pct": all(
            plan["estimated_slippage"] <= Decimal("0.001") for plan in plans.values()
        ),
        "macro_gate_healthy": bool(macro.get("runtime_gate_healthy")),
        "outside_macro_pause": not bool(macro.get("pause_new_orders")),
        "technical_gate_healthy": bool(technical.get("runtime_gate_healthy")),
        "technical_buy_enabled": bool(
            technical.get("pairs")
            and all(value.get("buy_enabled") for value in technical["pairs"].values())
        ),
    }
    complete = all(checks.values())
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "fetched_at": int(time.time()),
        "profile": PORTFOLIOS["FDUSD"].profile_name,
        "pairs": pairs,
        "required_fdusd": str(FDUSD_BUDGET.capital_limit),
        "recommended_fdusd": str(FDUSD_RECOMMENDED_BALANCE),
        "account": {"fdusd_balance": str(balance), "balance_passed": checks["balance_at_least_420_fdusd"]},
        "checks": checks,
        "maker_fee": str(maker_fee),
        "taker_fee": str(taker_fee),
        "public_checks": {
            pair: {
                "average_ask": str(plan["average_ask"]),
                "estimated_slippage": str(plan["estimated_slippage"]),
                "estimated_base": str(plan["amount"]),
            }
            for pair, plan in plans.items()
        },
        "validation_decision": validation.get("validation_decision"),
        "private_preflight_complete": complete,
        "eligible_for_manual_approval": complete,
        "deployment_allowed": False,
        "trading_enabled": False,
    }
    atomic_json(args.state_dir / "private_preflight.json", report)
    print(json.dumps(report, indent=2))
    return 0 if complete else 2


if __name__ == "__main__":
    raise SystemExit(main())

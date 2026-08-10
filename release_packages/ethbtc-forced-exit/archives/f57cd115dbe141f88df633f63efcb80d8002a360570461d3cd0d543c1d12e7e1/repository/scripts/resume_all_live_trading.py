#!/usr/bin/env python3
"""Hash-bound, fail-closed state preparation for resuming all live bots.

This command never starts a container.  It reconciles the stopped Grid runtime
to quote-only inventory, clears only the two reviewed historical latches, and
leaves Grid in COOLDOWN so its existing automatic re-entry path rebuilds base
inventory after three healthy Guard cycles.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Mapping


CONFIRMATION = "RESUME-ALL-ETHBTC-V22"
GRID_BOT = "grid-live-fdusd-400"
DCA_BOTS = ("dca-live-btcusdt-200", "dca-live-ethusdt-200")
PAIRS = ("BTC-FDUSD", "ETH-FDUSD")
EXPECTED_GRID_REASON = "recoverable PORTFOLIO exit remained incomplete for 4.4s"
EXPECTED_DCA_REASON = "monitor unavailable for 60s"


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def atomic(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".resume.tmp")
    temporary.write_text(
        json.dumps(dict(value), indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def update_grid_model_hash(path: Path, model_sha256: str) -> None:
    lines = path.read_text(encoding="utf-8").splitlines()
    matches = [index for index, line in enumerate(lines)
               if line.lstrip().startswith("technical_model_sha256:")]
    if len(matches) != 1:
        raise RuntimeError("Grid instance config has no unique technical model hash")
    index = matches[0]
    indentation = lines[index][:-len(lines[index].lstrip())]
    lines[index] = f"{indentation}technical_model_sha256: {model_sha256}"
    temporary = path.with_suffix(path.suffix + ".resume.tmp")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def active_state() -> dict[str, Any]:
    return {
        "phase": "ACTIVE", "mechanism": "", "scope": "",
        "triggered_at": None, "exit_target": "quote_only",
        "remaining_base": {}, "exit_completed_at": None,
        "cooldown_until": None, "healthy_cycles": 0,
        "reentry": {}, "episode_baseline": {},
    }


def validate_contract(contract: Mapping[str, Any], release: str, now: float) -> None:
    if contract.get("schema") != "ethbtc-forced-exit-live-contract-v1":
        raise RuntimeError("v22 live contract schema is invalid")
    if contract.get("release_sha256") != release:
        raise RuntimeError("v22 live contract release does not match approval")
    if not contract.get("source_healthy") or not contract.get("execution_authorized"):
        raise RuntimeError("v22 live contract is not healthy and authorized")
    generated = datetime.fromisoformat(str(contract["generated_at"]).replace("Z", "+00:00"))
    if not 0 <= now - generated.timestamp() <= 150:
        raise RuntimeError("v22 live contract is stale")
    pairs = contract.get("pairs", {})
    if set(pairs) != set(PAIRS):
        raise RuntimeError("v22 live contract does not contain exactly BTC/ETH FDUSD")
    if not all(item.get("buy_enabled") and not item.get("force_exit") for item in pairs.values()):
        raise RuntimeError("one or more v22 pair gates currently block normal trading")


def validate_inventory(status: Mapping[str, Any], now: float) -> None:
    if status.get("schema") != "account-inventory-status-v2" or not status.get("healthy"):
        raise RuntimeError("shared account inventory v2 is not healthy")
    if not 0 <= now - float(status.get("generated_at") or 0) <= 15:
        raise RuntimeError("shared account inventory is stale")
    if any(int(value) for value in status.get("open_order_counts", {}).values()):
        raise RuntimeError("exchange still has active BTC/ETH Grid or DCA orders")
    for asset in ("BTC", "ETH"):
        row = status.get("assets", {}).get(asset, {})
        if Decimal(str(row.get("ownership_deficit", "0"))) != 0:
            raise RuntimeError(f"shared account inventory has a {asset} ownership deficit")
        owner = Decimal(str(row.get("owners", {}).get(f"grid:{GRID_BOT}", "-1")))
        if owner != 0:
            raise RuntimeError(f"Grid {asset} ownership must be quote-only before recovery")


def prepare(
    *, grid_state: dict[str, Any], dca_state: dict[str, Any],
    runtime: dict[str, Any], inventory: Mapping[str, Any],
    contract: Mapping[str, Any], release: str, now: float,
) -> dict[str, Any]:
    validate_contract(contract, release, now)
    validate_inventory(inventory, now)
    grid = grid_state.get("bots", {}).get(GRID_BOT, {})
    if not grid.get("tripped") or grid.get("reason") != EXPECTED_GRID_REASON:
        raise RuntimeError("Grid latch is not the reviewed stale-contract incident")
    latest_pairs = grid.get("latest", {}).get("pairs", {})
    if set(latest_pairs) != set(PAIRS):
        raise RuntimeError("Grid audited pair PnL is incomplete")
    if runtime.get("portfolio_recovery", {}).get("mechanism") != "infrastructure_integrity_breaker":
        raise RuntimeError("Grid runtime latch is not an infrastructure integrity incident")
    if set(runtime.get("ledgers", {})) != set(PAIRS):
        raise RuntimeError("Grid runtime ledgers are incomplete")

    pair_equity: dict[str, str] = {}
    for pair in PAIRS:
        ledger = runtime["ledgers"][pair]
        equity = Decimal("200") + Decimal(str(latest_pairs[pair]["pnl"]))
        if equity <= 0:
            raise RuntimeError(f"Grid {pair} reconciled quote equity is not positive")
        ledger.update({
            "quote": str(equity), "base": "0", "base_cost_quote": "0",
            "halted": True, "open_order_ids": [],
            "peak_equity": str(equity), "episode_equity_baseline": str(equity),
        })
        pair_equity[pair] = str(equity)
    portfolio_equity = Decimal("20") + sum(map(Decimal, pair_equity.values()), Decimal("0"))
    runtime.update({
        "portfolio_tripped": True,
        "pair_recovery": {pair: active_state() for pair in PAIRS},
        "portfolio_recovery": {
            **active_state(), "phase": "COOLDOWN",
            "mechanism": "infrastructure_integrity_breaker",
            "scope": "infrastructure", "triggered_at": now,
            "reason": "manual hash-bound recovery after v22 and inventory repair",
            "trigger_value": release, "exit_completed_at": now,
            "cooldown_until": now, "latch_after_exit": False,
        },
        "pending_flatten": [], "flatten_order_ids": {},
        "reentry_order_ids": {}, "pending_inventory_exit": [],
        "inventory_exit_order_ids": {}, "buy_order_ids": [], "sell_order_ids": [],
        "peak_equity": str(portfolio_equity),
        "portfolio_episode_baseline": str(portfolio_equity),
        "updated_at": now,
    })
    runtime.setdefault("runtime_events", []).append({
        "timestamp": datetime.fromtimestamp(now, timezone.utc).isoformat(),
        "event": "manual_integrity_recovery_approved",
        "release_sha256": release, "pair_equity": pair_equity,
        "next_phase": "COOLDOWN_THEN_AUTOMATIC_REENTRY",
    })
    runtime["runtime_events"] = runtime["runtime_events"][-100:]

    grid.update({"tripped": False, "action_complete": False, "started_at": now})
    for key in ("reason", "tripped_at", "flatten", "stop", "post_stop_snapshot",
                "reconciled_snapshot", "last_action_error"):
        grid.pop(key, None)
    grid_state["first_failure_at"] = None
    grid_state["last_success_at"] = now

    for name in DCA_BOTS:
        bot = dca_state.get("bots", {}).get(name, {})
        if not bot.get("tripped") or bot.get("trip_reason") != EXPECTED_DCA_REASON:
            raise RuntimeError(f"{name} latch is not the reviewed monitor incident")
        if not bot.get("manual_exit_required"):
            raise RuntimeError(f"{name} does not retain reviewed managed inventory")
        bot.update({
            "tripped": False, "action_complete": False, "started_at": now,
            "recovery": active_state(),
        })
        for key in ("trip_reason", "tripped_at", "manual_exit_required", "exit_status",
                    "flatten_response", "stop_response", "last_action_error"):
            bot.pop(key, None)
    dca_state.pop("first_failure_at", None)
    dca_state.pop("last_monitor_error", None)
    dca_state["last_success_at"] = now
    return {"pair_equity": pair_equity, "portfolio_equity": str(portfolio_equity)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--release-sha256", required=True)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--confirm", default="")
    args = parser.parse_args()
    if len(args.release_sha256) != 64:
        raise RuntimeError("release SHA-256 is invalid")
    root, now = args.root.resolve(), time.time()
    paths = {
        "grid": root / "grid-live-fdusd-data/guard_state.json",
        "dca": root / "dca-live-data/guard_state.json",
        "runtime": root / "api-files/bots/instances/grid-live-fdusd-400/data/live_grid_runtime_state.json",
        "inventory": root / "account-inventory-data/account_inventory_status.json",
        "contract": root / "grid-live-fdusd-data/xgboost_risk_gate.json",
        "grid_config": root / (
            "api-files/bots/instances/grid-live-fdusd-400/conf/scripts/"
            "walk_forward_portfolio_grid_live_fdusd_400.yml"
        ),
    }
    grid_state, dca_state, runtime = load(paths["grid"]), load(paths["dca"]), load(paths["runtime"])
    contract = load(paths["contract"])
    if "technical_model_sha256:" not in paths["grid_config"].read_text(encoding="utf-8"):
        raise RuntimeError("Grid instance config lacks the v22 model hash binding")
    summary = prepare(
        grid_state=grid_state, dca_state=dca_state, runtime=runtime,
        inventory=load(paths["inventory"]), contract=contract,
        release=args.release_sha256, now=now,
    )
    result = {
        "dry_run": not args.apply, "release_sha256": args.release_sha256,
        "model_sha256": contract["model_sha256"], **summary,
    }
    if not args.apply:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if args.confirm != CONFIRMATION:
        raise RuntimeError(f"explicit --confirm {CONFIRMATION} is required")
    stamp = datetime.fromtimestamp(now, timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    for path in paths.values():
        if path in (paths["inventory"], paths["contract"]):
            continue
        shutil.copy2(path, path.with_name(f"{path.stem}.before_resume_all.{stamp}{path.suffix}"))
    update_grid_model_hash(paths["grid_config"], str(contract["model_sha256"]))
    atomic(paths["runtime"], runtime)
    atomic(paths["grid"], grid_state)
    atomic(paths["dca"], dca_state)
    audit = root / "grid-live-fdusd-data/risk_audit.jsonl"
    with audit.open("a", encoding="utf-8") as output:
        output.write(json.dumps({
            "timestamp": datetime.fromtimestamp(now, timezone.utc).isoformat(),
            "event": "resume_all_live_trading_state_prepared",
            "approval": "explicit_user_authorization",
            **result,
        }, ensure_ascii=False) + "\n")
    result["applied"] = True
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

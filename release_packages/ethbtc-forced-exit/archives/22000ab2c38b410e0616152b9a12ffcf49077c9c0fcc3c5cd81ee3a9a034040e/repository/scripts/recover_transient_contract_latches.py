#!/usr/bin/env python3
"""Recover only reviewed transport-error latches after strict live preflight."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from risk_recovery import active_state


def read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SystemExit(f"refusing recovery: {path} is not a JSON object")
    return value


def write_atomic(path: Path, value: dict[str, Any]) -> None:
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as output:
        json.dump(value, output, indent=2, sort_keys=True)
        temporary = Path(output.name)
    os.chmod(temporary, path.stat().st_mode)
    os.chown(temporary, path.stat().st_uid, path.stat().st_gid)
    temporary.replace(path)


def transport_latch(value: Any) -> bool:
    text = str(value or "").lower()
    return "connectionreseterror" in text or "connection reset by peer" in text


def cooldown_state(*, now: float, mechanism: str, reason: str) -> dict[str, Any]:
    return {
        **active_state(),
        "phase": "COOLDOWN",
        "mechanism": mechanism,
        "scope": "technical",
        "triggered_at": now,
        "exit_completed_at": now,
        "cooldown_until": now,
        "reason": reason,
        "latch_after_exit": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--confirm", required=True)
    args = parser.parse_args()
    if args.confirm != "RECOVER-REVIEWED-TRANSIENT-CONTRACT-LATCHES":
        raise SystemExit("confirmation mismatch")

    root = args.root.resolve()
    grid_guard_path = root / "grid-live-fdusd-data/guard_state.json"
    dca_guard_path = root / "dca-live-data/guard_state.json"
    grid_runtime_path = root / (
        "api-files/bots/instances/grid-live-fdusd-400/data/live_grid_runtime_state.json"
    )
    contract_path = root / "grid-live-fdusd-data/xgboost_risk_gate.json"
    inventory_path = root / "account-inventory-data/account_inventory_status.json"
    paths = (grid_guard_path, dca_guard_path, grid_runtime_path, contract_path, inventory_path)
    if not all(path.exists() for path in paths):
        raise SystemExit("refusing recovery: required state file is missing")

    contract = read_object(contract_path)
    inventory = read_object(inventory_path)
    grid_runtime = read_object(grid_runtime_path)
    dca_guard = read_object(dca_guard_path)
    grid_guard = read_object(grid_guard_path)
    now = time.time()
    generated = datetime.fromisoformat(
        str(contract["generated_at"]).replace("Z", "+00:00")
    ).timestamp()
    valid_until = datetime.fromisoformat(
        str(contract["valid_until"]).replace("Z", "+00:00")
    ).timestamp()
    if not (
        contract.get("source_healthy") is True
        and contract.get("execution_authorized") is True
        and now - generated < 150
        and now <= valid_until
        and set(contract.get("pairs", {})) == {"BTC-FDUSD", "ETH-FDUSD"}
    ):
        raise SystemExit("refusing recovery: v22 contract is not currently healthy and authorized")
    if inventory.get("healthy") is not True or int(inventory.get("active_order_count", -1)) != 0:
        raise SystemExit("refusing recovery: inventory contract is unhealthy or orders are active")
    for asset in ("BTC", "ETH"):
        if float(inventory.get("assets", {}).get(asset, {}).get("ownership_deficit", 0)) > 0:
            raise SystemExit(f"refusing recovery: {asset} ownership deficit")

    portfolio = grid_runtime.get("portfolio_recovery", {})
    if portfolio.get("phase") != "LATCHED" or not transport_latch(portfolio.get("reason")):
        raise SystemExit("refusing recovery: Grid latch is not the reviewed transport error")
    for bot_name in ("dca-live-btcusdt-200", "dca-live-ethusdt-200"):
        recovery = dca_guard.get("bots", {}).get(bot_name, {}).get("recovery", {})
        if recovery.get("phase") != "LATCHED" or not transport_latch(recovery.get("reason")):
            raise SystemExit(f"refusing recovery: {bot_name} latch is not the reviewed transport error")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = root / "deploy_backups/transient_contract_grace" / f"recovery-{stamp}"
    backup.mkdir(parents=True)
    for label, path in (
        ("grid_guard_state.json", grid_guard_path),
        ("dca_guard_state.json", dca_guard_path),
        ("grid_runtime_state.json", grid_runtime_path),
    ):
        shutil.copy2(path, backup / label)

    grid_runtime["portfolio_recovery"] = active_state()
    grid_runtime["portfolio_tripped"] = False
    grid_runtime.pop("integrity_failure_grace", None)
    for pair, signal in contract["pairs"].items():
        risk_off = bool(signal.get("risk_off_active") or signal.get("force_exit"))
        if risk_off:
            grid_runtime["ledgers"][pair]["halted"] = True
        else:
            grid_runtime["ledgers"][pair]["halted"] = True
            grid_runtime["pair_recovery"][pair] = cooldown_state(
                now=now,
                mechanism="infrastructure_integrity_breaker",
                reason="reviewed transient transport latch recovered; rebuild under current gates",
            )
    grid_runtime.setdefault("runtime_events", []).append({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event": "reviewed_transient_contract_latch_recovered",
        "current_v22": {
            pair: bool(signal.get("buy_enabled"))
            for pair, signal in contract["pairs"].items()
        },
    })

    pair_map = {"dca-live-btcusdt-200": "BTC-FDUSD", "dca-live-ethusdt-200": "ETH-FDUSD"}
    for bot_name, source_pair in pair_map.items():
        signal = contract["pairs"][source_pair]
        bot = dca_guard["bots"][bot_name]
        bot["tripped"] = False
        bot["action_complete"] = False
        bot["recovery"] = (
            active_state()
            if signal.get("risk_off_active")
            else cooldown_state(
                now=now,
                mechanism="infrastructure_integrity_breaker",
                reason="reviewed transient transport latch recovered; rebuild under current gates",
            )
        )
        for key in ("trip_reason", "tripped_at", "last_action_error"):
            bot.pop(key, None)
    dca_guard.pop("integrity_failure_grace", None)
    dca_guard["last_success_at"] = now

    write_atomic(grid_runtime_path, grid_runtime)
    write_atomic(dca_guard_path, dca_guard)
    audit = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event": "reviewed_transient_contract_latch_recovered",
        "approval": "explicit_user_fix_request",
        "backup": str(backup),
        "contract_release": contract.get("release_sha256"),
        "current_v22": {
            pair: bool(signal.get("buy_enabled"))
            for pair, signal in contract["pairs"].items()
        },
    }
    for audit_path in (
        root / "grid-live-fdusd-data/risk_audit.jsonl",
        root / "dca-live-data/risk_audit.jsonl",
    ):
        with audit_path.open("a", encoding="utf-8") as output:
            output.write(json.dumps(audit, ensure_ascii=False) + "\n")
    print(json.dumps(audit, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

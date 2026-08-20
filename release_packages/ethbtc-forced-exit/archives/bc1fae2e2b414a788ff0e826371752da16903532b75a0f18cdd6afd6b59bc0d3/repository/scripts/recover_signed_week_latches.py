#!/usr/bin/env python3
"""Recover stale-signed-week latches after a newly approved v22 contract is healthy."""

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

try:
    from risk_recovery import active_state
except ModuleNotFoundError:
    from scripts.risk_recovery import active_state


CONFIRMATION = "RECOVER-APPROVED-SIGNED-WEEK-LATCHES"
BOT_PAIRS = {
    "dca-live-btcusdt-200": "BTC-FDUSD",
    "dca-live-ethusdt-200": "ETH-FDUSD",
}


def read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"{path} is not a JSON object")
    return value


def write_atomic(path: Path, value: dict[str, Any]) -> None:
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False,
    ) as output:
        json.dump(value, output, indent=2, sort_keys=True)
        temporary = Path(output.name)
    os.chmod(temporary, path.stat().st_mode)
    if hasattr(os, "chown"):
        os.chown(temporary, path.stat().st_uid, path.stat().st_gid)
    temporary.replace(path)


def signed_week_reason(value: Any) -> bool:
    text = str(value or "").lower()
    return any(marker in text for marker in (
        "contract is stale", "no signed weekly model covers", "signed_week_unavailable",
    ))


def cooldown_state(now: float) -> dict[str, Any]:
    return {
        **active_state(),
        "phase": "COOLDOWN",
        "mechanism": "infrastructure_integrity_breaker",
        "scope": "infrastructure",
        "triggered_at": now,
        "exit_target": "quote_only",
        "remaining_base": {},
        "exit_completed_at": now,
        "cooldown_until": now,
        "healthy_cycles": 0,
        "reentry": {},
        "episode_baseline": {},
        "latch_after_exit": False,
        "reason": "approved signed week restored; rebuild under current gates",
    }


def recover(
    root: Path, *, release_sha256: str, runtime_generation: str,
    confirm: str, observed_at: float | None = None,
) -> dict[str, Any]:
    if confirm != CONFIRMATION:
        raise RuntimeError("confirmation mismatch")
    root = root.resolve()
    now = time.time() if observed_at is None else observed_at
    paths = {
        "grid_guard": root / "grid-live-fdusd-data/guard_state.json",
        "dca_guard": root / "dca-live-data/guard_state.json",
        "grid_runtime": root / "api-files/bots/instances/grid-live-fdusd-400/data/live_grid_runtime_state.json",
        "contract": root / "grid-live-fdusd-data/ethbtc_forced_exit_observation.json",
        "execution_contract": root / "grid-live-fdusd-data/xgboost_risk_gate.json",
        "inventory": root / "account-inventory-data/account_inventory_status.json",
        "automation": root / "results/ethbtc_forced_exit_weekly/automation_state.json",
        "grid_macro": root / "grid-live-fdusd-data/macro_gate.json",
        "dca_macro": root / "dca-macro-data/state.json",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise RuntimeError(f"required recovery evidence is missing: {missing}")
    values = {name: read_object(path) for name, path in paths.items()}
    contract = values["contract"]
    execution = values["execution_contract"]
    generated = datetime.fromisoformat(
        str(contract["generated_at"]).replace("Z", "+00:00")
    ).timestamp()
    valid_until = datetime.fromisoformat(
        str(contract["valid_until"]).replace("Z", "+00:00")
    ).timestamp()
    identity_ok = all(
        item.get("release_sha256") == release_sha256
        and item.get("runtime_generation") == runtime_generation
        and item.get("source_healthy") is True
        and item.get("execution_authorized") is True
        for item in (contract, execution)
    )
    if not (
        identity_ok and now - generated < 150 and now <= valid_until
        and set(contract.get("pairs", {})) == {"BTC-FDUSD", "ETH-FDUSD"}
    ):
        raise RuntimeError("approved v22 contract is not healthy, fresh, and identity-bound")
    automation = values["automation"]
    if not (
        automation.get("phase") == "ACTIVE"
        and automation.get("candidate_release_sha256") == release_sha256
        and automation.get("runtime_generation") == runtime_generation
    ):
        raise RuntimeError("weekly automation has not finalized the approved generation")
    inventory = values["inventory"]
    if not (
        inventory.get("healthy") is True
        and inventory.get("sources_healthy") is True
        and int(inventory.get("active_order_count", -1)) == 0
        and all(
            float(inventory.get("assets", {}).get(asset, {}).get("ownership_deficit", 1)) == 0
            for asset in ("BTC", "ETH")
        )
    ):
        raise RuntimeError("inventory is unhealthy, has orders, or has an ownership deficit")
    grid_macro = values["grid_macro"]
    dca_macro = values["dca_macro"]
    if not (
        grid_macro.get("source_healthy") is True
        and grid_macro.get("pause_new_orders") is False
        and dca_macro.get("desired_gates") == {"buy": True, "sell": True}
    ):
        raise RuntimeError("FOMC gates do not currently allow recovery")

    grid_runtime = values["grid_runtime"]
    dca_guard = values["dca_guard"]
    portfolio = grid_runtime.get("portfolio_recovery", {})
    if portfolio.get("phase") != "LATCHED" or not signed_week_reason(portfolio.get("reason")):
        raise RuntimeError("Grid is not latched by the reviewed signed-week incident")
    for bot_name in BOT_PAIRS:
        recovery = dca_guard.get("bots", {}).get(bot_name, {}).get("recovery", {})
        if recovery.get("phase") != "LATCHED" or not signed_week_reason(recovery.get("reason")):
            raise RuntimeError(f"{bot_name} is not latched by the reviewed signed-week incident")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = root / "deploy_backups/signed_week_recovery" / f"recovery-{stamp}"
    backup.mkdir(parents=True, exist_ok=False)
    for label in ("grid_guard", "dca_guard", "grid_runtime"):
        shutil.copy2(paths[label], backup / f"{label}.json")

    grid_runtime["portfolio_recovery"] = active_state()
    grid_runtime["portfolio_tripped"] = False
    grid_runtime.pop("integrity_failure_grace", None)
    for pair, signal in contract["pairs"].items():
        ledger = grid_runtime["ledgers"][pair]
        ledger["halted"] = True
        ledger["open_order_ids"] = []
        if not bool(signal.get("risk_off_active") or signal.get("force_exit")):
            grid_runtime["pair_recovery"][pair] = cooldown_state(now)
    grid_runtime.setdefault("runtime_events", []).append({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event": "approved_signed_week_latch_recovered",
        "release_sha256": release_sha256,
        "runtime_generation": runtime_generation,
    })

    for bot_name in BOT_PAIRS:
        bot = dca_guard["bots"][bot_name]
        bot["tripped"] = False
        bot["action_complete"] = False
        bot["recovery"] = cooldown_state(now)
        for key in ("trip_reason", "tripped_at", "last_action_error"):
            bot.pop(key, None)
    dca_guard.pop("integrity_failure_grace", None)
    dca_guard["last_success_at"] = now

    write_atomic(paths["grid_runtime"], grid_runtime)
    write_atomic(paths["dca_guard"], dca_guard)
    audit = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event": "approved_signed_week_latch_recovered",
        "approval": "explicit_user_authorization",
        "release_sha256": release_sha256,
        "runtime_generation": runtime_generation,
        "backup": str(backup),
        "signals": {
            pair: str(signal.get("model_signal"))
            for pair, signal in contract["pairs"].items()
        },
    }
    for audit_path in (
        root / "grid-live-fdusd-data/risk_audit.jsonl",
        root / "dca-live-data/risk_audit.jsonl",
    ):
        with audit_path.open("a", encoding="utf-8") as output:
            output.write(json.dumps(audit, ensure_ascii=False) + "\n")
    return audit


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--release-sha256", required=True)
    parser.add_argument("--runtime-generation", required=True)
    parser.add_argument("--confirm", required=True)
    args = parser.parse_args()
    print(json.dumps(recover(
        args.root, release_sha256=args.release_sha256,
        runtime_generation=args.runtime_generation, confirm=args.confirm,
    ), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

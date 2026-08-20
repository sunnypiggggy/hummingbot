#!/usr/bin/env python3
"""Reset only the approved monitor-unavailable DCA breaker event."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path


BOT_NAMES = ("dca-live-btcusdt-200", "dca-live-ethusdt-200")
EXPECTED_REASON = "monitor unavailable for 60s"
CONFIRMATION = "RESET-MONITOR-UNAVAILABLE-60S"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--confirm", required=True)
    args = parser.parse_args()
    if args.confirm != CONFIRMATION:
        raise SystemExit("confirmation mismatch")

    state_dir = Path(os.getenv("DCA_LIVE_STATE_PATH", "/workspace/state"))
    state_path = state_dir / "guard_state.json"
    audit_path = state_dir / "risk_audit.jsonl"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    previous = {}
    for bot_name in BOT_NAMES:
        bot_state = state.get("bots", {}).get(bot_name)
        if not isinstance(bot_state, dict) or not bot_state.get("tripped"):
            raise SystemExit(f"refusing reset: {bot_name} is not tripped")
        if bot_state.get("trip_reason") != EXPECTED_REASON:
            raise SystemExit(
                f"refusing reset: {bot_name} reason is {bot_state.get('trip_reason')!r}"
            )
        previous[bot_name] = {
            "trip_reason": bot_state.get("trip_reason"),
            "tripped_at": bot_state.get("tripped_at"),
            "action_complete": bot_state.get("action_complete"),
        }

    now = time.time()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = state_dir / f"guard_state.before_manual_reset.{stamp}.json"
    shutil.copy2(state_path, backup)
    for bot_name in BOT_NAMES:
        bot_state = state["bots"][bot_name]
        bot_state.update({
            "tripped": False,
            "action_complete": False,
            "started_at": now,
            "recovery": {
                # The integrity exit sold the managed startup inventory.  Do
                # not claim ACTIVE before it has been rebuilt.  COOLDOWN has
                # no additional delay here; the Guard still requires three
                # healthy cycles and every independent BUY gate before its
                # existing re-entry path can buy the owned baseline again.
                "phase": "COOLDOWN",
                "mechanism": "infrastructure_integrity_breaker",
                "scope": "infrastructure",
                "triggered_at": previous[bot_name]["tripped_at"],
                "exit_target": "quote_only",
                "remaining_base": dict(
                    bot_state.get("recovery", {}).get("remaining_base", {})
                ),
                "exit_completed_at": bot_state.get("recovery", {}).get(
                    "exit_completed_at", now
                ),
                "cooldown_until": now,
                "healthy_cycles": 0,
                "reentry": {},
                "episode_baseline": {},
                "latch_after_exit": False,
                "reason": "manual recovery after reviewed monitor incident",
            },
        })
        for key in (
            "trip_reason", "tripped_at", "last_action_error",
            "stop_response", "flatten_response",
        ):
            bot_state.pop(key, None)
    state.pop("last_monitor_error", None)
    state.pop("first_failure_at", None)
    state["last_success_at"] = now

    temporary = state_path.with_suffix(".reset.tmp")
    temporary.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(state_path)
    audit = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event": "manual_breaker_reset",
        "approval": "explicit_user_authorization",
        "scope": list(BOT_NAMES),
        "expected_reason": EXPECTED_REASON,
        "previous": previous,
        "backup": str(backup),
    }
    with audit_path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(audit, ensure_ascii=False) + "\n")
    print(json.dumps({"reset": list(BOT_NAMES), "backup": str(backup)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

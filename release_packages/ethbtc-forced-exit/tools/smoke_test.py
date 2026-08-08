#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

sys.dont_write_bytecode = True

import pandas as pd


def epoch(value: str) -> int:
    return int(datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp())


def main() -> int:
    package = Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
    sys.path.insert(0, str(package / "sources"))
    from build_v22_grid_dca_forced_exit_v2 import PACKAGE_ID, POLICY, inventory_overlay

    assert PACKAGE_ID == "ethbtc-forced-exit" == POLICY["package_id"]
    frame = pd.DataFrame([
        {"timestamp": 1000, "open": 10_000.0, "close": 10_000.0},
        {"timestamp": 1300, "open": 9_900.0, "close": 9_900.0},
    ])
    inventory, actions = inventory_overlay(frame, pd.Series([False, False]), "BTC-USDT", True)
    assert inventory.phase.tolist() == ["EXITING", "COOLDOWN"]
    assert actions[["signal_ts", "execution_ts"]].iloc[0].tolist() == [1000, 1300]

    summary = json.loads((package / "evidence/summary.json").read_text(encoding="utf-8"))
    assert summary["package_id"] == PACKAGE_ID and summary["deployment_allowed"] is False
    evidence = pd.read_csv(package / "evidence/execution_actions.csv")
    targets = {
        ("grid", "BTC-FDUSD", epoch("2026-05-13T00:00:00Z")): epoch("2026-05-13T00:05:00Z"),
        ("grid", "ETH-FDUSD", epoch("2026-05-23T00:00:00Z")): epoch("2026-05-23T00:05:00Z"),
        ("dca", "BTC-USDT", epoch("2026-05-13T00:00:00Z")): epoch("2026-05-13T00:05:00Z"),
        ("dca", "ETH-USDT", epoch("2026-05-23T00:00:00Z")): epoch("2026-05-23T00:05:00Z"),
    }
    exits = evidence[evidence.action.eq("MARKET_EXIT")]
    for (strategy, pair, signal), execution in targets.items():
        row = exits[(exits.strategy == strategy) & (exits.pair == pair) & (exits.signal_ts == signal)]
        assert len(row) == 1 and int(row.iloc[0].execution_ts) == execution
    print(json.dumps({"package_id": PACKAGE_ID, "smoke_test": "PASS", "target_exits": len(targets)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

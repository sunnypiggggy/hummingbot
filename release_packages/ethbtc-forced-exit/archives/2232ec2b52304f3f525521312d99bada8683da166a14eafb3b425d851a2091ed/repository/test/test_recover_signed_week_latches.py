import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from scripts.recover_signed_week_latches import CONFIRMATION, recover


def write(root: Path, relative: str, value: dict) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_recovery_requires_bound_healthy_contract_and_enters_gated_reentry(tmp_path: Path):
    release, generation = "a" * 64, "b" * 64
    contract = {
        "generated_at": datetime.fromtimestamp(90, timezone.utc).isoformat(),
        "valid_until": datetime.fromtimestamp(200, timezone.utc).isoformat(),
        "release_sha256": release,
        "runtime_generation": generation,
        "source_healthy": True,
        "execution_authorized": True,
        "pairs": {
            "BTC-FDUSD": {"model_signal": "RISK_OFF", "risk_off_active": True, "force_exit": True},
            "ETH-FDUSD": {"model_signal": "RISK_ON", "risk_off_active": False, "force_exit": False},
        },
    }
    write(tmp_path, "grid-live-fdusd-data/ethbtc_forced_exit_observation.json", contract)
    write(tmp_path, "grid-live-fdusd-data/xgboost_risk_gate.json", contract)
    write(tmp_path, "grid-live-fdusd-data/guard_state.json", {})
    write(tmp_path, "account-inventory-data/account_inventory_status.json", {
        "healthy": True, "sources_healthy": True, "active_order_count": 0,
        "assets": {"BTC": {"ownership_deficit": "0"}, "ETH": {"ownership_deficit": "0"}},
    })
    write(tmp_path, "results/ethbtc_forced_exit_weekly/automation_state.json", {
        "phase": "ACTIVE", "candidate_release_sha256": release,
        "runtime_generation": generation,
    })
    write(tmp_path, "grid-live-fdusd-data/macro_gate.json", {
        "source_healthy": True, "pause_new_orders": False,
    })
    write(tmp_path, "dca-macro-data/state.json", {
        "desired_gates": {"buy": True, "sell": True},
    })
    latch = {"phase": "LATCHED", "reason": "fail_closed:contract is stale"}
    write(tmp_path, "api-files/bots/instances/grid-live-fdusd-400/data/live_grid_runtime_state.json", {
        "portfolio_tripped": True, "portfolio_recovery": latch,
        "pair_recovery": {"BTC-FDUSD": {"phase": "REENTRY"}, "ETH-FDUSD": {"phase": "ACTIVE"}},
        "ledgers": {
            "BTC-FDUSD": {"halted": True, "open_order_ids": ["stale"]},
            "ETH-FDUSD": {"halted": True, "open_order_ids": ["stale"]},
        },
    })
    write(tmp_path, "dca-live-data/guard_state.json", {
        "bots": {
            "dca-live-btcusdt-200": {"recovery": latch, "tripped": False},
            "dca-live-ethusdt-200": {"recovery": latch, "tripped": False},
        },
    })

    audit = recover(
        tmp_path, release_sha256=release, runtime_generation=generation,
        confirm=CONFIRMATION, observed_at=100,
    )
    assert audit["signals"] == {"BTC-FDUSD": "RISK_OFF", "ETH-FDUSD": "RISK_ON"}
    grid = json.loads((tmp_path / "api-files/bots/instances/grid-live-fdusd-400/data/live_grid_runtime_state.json").read_text())
    dca = json.loads((tmp_path / "dca-live-data/guard_state.json").read_text())
    assert grid["portfolio_tripped"] is False
    assert grid["portfolio_recovery"]["phase"] == "ACTIVE"
    assert grid["pair_recovery"]["BTC-FDUSD"]["phase"] == "REENTRY"
    assert grid["pair_recovery"]["ETH-FDUSD"]["phase"] == "COOLDOWN"
    assert all(not row["open_order_ids"] for row in grid["ledgers"].values())
    assert all(bot["recovery"]["phase"] == "COOLDOWN" for bot in dca["bots"].values())
    assert Path(audit["backup"]).is_dir()

    with pytest.raises(RuntimeError, match="Grid is not latched"):
        recover(
            tmp_path, release_sha256=release, runtime_generation=generation,
            confirm=CONFIRMATION, observed_at=101,
        )

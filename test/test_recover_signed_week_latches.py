import json
from datetime import datetime, timezone
from decimal import Decimal
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
    write(tmp_path, "grid-live-fdusd-data/guard_state.json", {"bots": {
        "grid-live-fdusd-400": {
            "tripped": True, "action_complete": True, "stop_complete": True,
            "stop": {"verified_no_active_orders": True, "verified_no_live_instances": True},
            "latest": {"pairs": {
                "BTC-FDUSD": {"pnl": "2", "mark": "50000"},
                "ETH-FDUSD": {"pnl": "-1", "mark": "2000"},
            }},
        },
    }})
    write(tmp_path, "account-inventory-data/account_inventory_status.json", {
        "healthy": True, "sources_healthy": True, "active_order_count": 0,
        "assets": {
            "BTC": {"ownership_deficit": "0", "exchange": {"total": "0.001"},
                    "owners": {"grid:grid-live-fdusd-400": "0.0001"}},
            "ETH": {"ownership_deficit": "0", "exchange": {"total": "0.01"},
                    "owners": {"grid:grid-live-fdusd-400": "0.001"}},
        },
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
    write(tmp_path, "grid-live-fdusd-data/capital_reservations.json", {
        "prices": {"BTC-FDUSD": "50000", "ETH-FDUSD": "2000"},
        "reservations": {"FDUSD": {"base": {"BTC": "0.002", "ETH": "0.05"}}},
    })
    latch = {"phase": "LATCHED", "reason": "fail_closed:signed week has expired"}
    write(tmp_path, "api-files/bots/instances/grid-live-fdusd-400/data/live_grid_runtime_state.json", {
        "portfolio_tripped": True, "portfolio_recovery": latch,
        "pair_recovery": {"BTC-FDUSD": {"phase": "REENTRY"}, "ETH-FDUSD": {"phase": "ACTIVE"}},
        "ledgers": {
            "BTC-FDUSD": {"initial_quote": "100", "halted": True, "open_order_ids": ["stale"]},
            "ETH-FDUSD": {"initial_quote": "100", "halted": True, "open_order_ids": ["stale"]},
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
    assert grid["ledgers"]["BTC-FDUSD"]["base"] == "0.0001"
    assert grid["ledgers"]["BTC-FDUSD"]["quote"] == "197.0000"
    assert grid["ledgers"]["ETH-FDUSD"]["base"] == "0.001"
    assert grid["ledgers"]["ETH-FDUSD"]["quote"] == "197.000"
    assert all(bot["recovery"]["phase"] == "COOLDOWN" for bot in dca["bots"].values())
    guard = json.loads((tmp_path / "grid-live-fdusd-data/guard_state.json").read_text())
    assert guard["bots"]["grid-live-fdusd-400"]["tripped"] is False
    assert Path(audit["backup"]).is_dir()

    with pytest.raises(RuntimeError, match="Grid is not latched"):
        recover(
            tmp_path, release_sha256=release, runtime_generation=generation,
            confirm=CONFIRMATION, observed_at=101,
        )


def test_recovery_accepts_durable_exit_evidence_after_guard_restart(tmp_path: Path):
    release, generation = "c" * 64, "d" * 64
    contract = {
        "generated_at": datetime.fromtimestamp(90, timezone.utc).isoformat(),
        "valid_until": datetime.fromtimestamp(200, timezone.utc).isoformat(),
        "release_sha256": release, "runtime_generation": generation,
        "source_healthy": True, "execution_authorized": True,
        "pairs": {
            "BTC-FDUSD": {"model_signal": "RISK_OFF", "risk_off_active": True, "force_exit": True},
            "ETH-FDUSD": {"model_signal": "RISK_ON", "risk_off_active": False, "force_exit": False},
        },
    }
    write(tmp_path, "grid-live-fdusd-data/ethbtc_forced_exit_observation.json", contract)
    write(tmp_path, "grid-live-fdusd-data/xgboost_risk_gate.json", contract)
    write(tmp_path, "results/ethbtc_forced_exit_weekly/automation_state.json", {
        "phase": "ACTIVE", "candidate_release_sha256": release,
        "runtime_generation": generation,
    })
    write(tmp_path, "grid-live-fdusd-data/macro_gate.json", {
        "source_healthy": True, "pause_new_orders": False,
    })
    write(tmp_path, "dca-macro-data/state.json", {"desired_gates": {"buy": True, "sell": True}})
    write(tmp_path, "grid-live-fdusd-data/capital_reservations.json", {
        "prices": {"BTC-FDUSD": "50000", "ETH-FDUSD": "2000"},
        "reservations": {"FDUSD": {"base": {"BTC": "0.002", "ETH": "0.05"}}},
    })
    latch = {"phase": "LATCHED", "reason": "fail_closed:signed week has expired"}
    write(tmp_path, "grid-live-fdusd-data/guard_state.json", {"bots": {
        "grid-live-fdusd-400": {"tripped": False, "action_complete": False,
            "latest": {"pairs": {"BTC-FDUSD": {"pnl": "2", "mark": "50000"},
                                  "ETH-FDUSD": {"pnl": "-1", "mark": "2000"}}}},
    }})
    write(tmp_path, "account-inventory-data/account_inventory_status.json", {
        "healthy": True, "sources_healthy": True, "active_order_count": 0,
        "assets": {
            "BTC": {"ownership_deficit": "0", "exchange": {"total": "0.00001"},
                    "owners": {"grid:grid-live-fdusd-400": "0.00001"}},
            "ETH": {"ownership_deficit": "0", "exchange": {"total": "0.0001"},
                    "owners": {"grid:grid-live-fdusd-400": "0.0001"}},
        },
    })
    write(tmp_path, "api-files/bots/instances/grid-live-fdusd-400/data/live_grid_runtime_state.json", {
        "portfolio_tripped": True,
        "portfolio_recovery": {**latch, "exit_completed_at": 95,
                               "execution": {"target": "quote_only", "last_fill": {"filled_at": 94}}},
        "pair_recovery": {"BTC-FDUSD": {"phase": "REENTRY"}, "ETH-FDUSD": {"phase": "ACTIVE"}},
        "ledgers": {"BTC-FDUSD": {"initial_quote": "100", "halted": True, "open_order_ids": []},
                    "ETH-FDUSD": {"initial_quote": "100", "halted": True, "open_order_ids": []}},
    })
    write(tmp_path, "dca-live-data/guard_state.json", {"bots": {
        "dca-live-btcusdt-200": {"recovery": latch},
        "dca-live-ethusdt-200": {"recovery": latch},
    }})

    audit = recover(
        tmp_path, release_sha256=release, runtime_generation=generation,
        confirm=CONFIRMATION, observed_at=100,
        minimum_notionals={"BTC-FDUSD": Decimal("5"), "ETH-FDUSD": Decimal("5")},
    )
    assert audit["grid_exit_evidence"] == "durable_exit_audit_and_live_exchange_filters"


def test_recovery_rejects_tradable_grid_residual_without_stop_snapshot(tmp_path: Path):
    # Reuse the durable fixture construction, but make the BTC owner tradable.
    test_recovery_accepts_durable_exit_evidence_after_guard_restart(tmp_path)
    # The successful helper call above consumed the latch, so this test only
    # verifies the predicate directly through a fresh in-memory payload.
    from scripts.recover_signed_week_latches import durable_grid_exit_verified
    assert durable_grid_exit_verified(
        portfolio={"exit_completed_at": 1, "execution": {"target": "quote_only"}},
        grid_bot={"latest": {"pairs": {"BTC-FDUSD": {"mark": "50000"},
                                        "ETH-FDUSD": {"mark": "2000"}}}},
        inventory={"assets": {
            "BTC": {"owners": {"grid:grid-live-fdusd-400": "0.001"}},
            "ETH": {"owners": {"grid:grid-live-fdusd-400": "0.0001"}},
        }},
        minimum_notionals={"BTC-FDUSD": Decimal("5"), "ETH-FDUSD": Decimal("5")},
    ) is False

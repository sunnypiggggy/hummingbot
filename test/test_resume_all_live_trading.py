from __future__ import annotations

from scripts.resume_all_live_trading import (
    DCA_BOTS, EXPECTED_DCA_REASON, EXPECTED_GRID_REASON, PAIRS, prepare,
)


def test_prepare_reconciles_grid_to_cash_and_preserves_dca_inventory() -> None:
    release = "a" * 64
    now = 1_800_000_000.0
    grid = {"bots": {"grid-live-fdusd-400": {
        "tripped": True, "reason": EXPECTED_GRID_REASON, "action_complete": True,
        "latest": {"pairs": {
            "BTC-FDUSD": {"pnl": "8.5"}, "ETH-FDUSD": {"pnl": "0.25"},
        }},
    }}}
    dca = {"bots": {name: {
        "tripped": True, "trip_reason": EXPECTED_DCA_REASON,
        "manual_exit_required": True, "managed_base_target": "0.01",
        "recovery": {"phase": "LATCHED"},
    } for name in DCA_BOTS}}
    runtime = {
        "portfolio_recovery": {"mechanism": "infrastructure_integrity_breaker"},
        "ledgers": {pair: {
            "quote": "1", "base": "9", "base_cost_quote": "9",
            "halted": True, "open_order_ids": ["stale"],
        } for pair in PAIRS},
        "runtime_events": [],
    }
    inventory = {
        "schema": "account-inventory-status-v3", "healthy": True,
        "generated_at": now, "open_order_counts": {pair: 0 for pair in PAIRS},
        "assets": {
            "BTC": {"ownership_deficit": "0", "owners": {
                "grid:grid-live-fdusd-400": "0",
                "dca:dca-live-btcusdt-200": "0.01",
            }},
            "ETH": {"ownership_deficit": "0", "owners": {
                "grid:grid-live-fdusd-400": "0",
                "dca:dca-live-ethusdt-200": "0.01",
            }},
        },
    }
    contract = {
        "schema": "ethbtc-forced-exit-live-contract-v1",
        "release_sha256": release, "source_healthy": True,
        "execution_authorized": True, "generated_at": "2027-01-15T08:00:00Z",
        "pairs": {pair: {"buy_enabled": True, "force_exit": False} for pair in PAIRS},
    }

    result = prepare(
        grid_state=grid, dca_state=dca, runtime=runtime,
        inventory=inventory, contract=contract, release=release, now=now,
    )

    assert result["portfolio_equity"] == "428.75"
    assert runtime["ledgers"]["BTC-FDUSD"]["base"] == "0"
    assert runtime["ledgers"]["BTC-FDUSD"]["quote"] == "208.5"
    assert runtime["portfolio_recovery"]["phase"] == "COOLDOWN"
    assert runtime["portfolio_tripped"] is True
    assert grid["bots"]["grid-live-fdusd-400"]["tripped"] is False
    for name in DCA_BOTS:
        assert dca["bots"][name]["tripped"] is False
        assert dca["bots"][name]["managed_base_target"] == "0.01"
        assert dca["bots"][name]["recovery"]["phase"] == "COOLDOWN"
        assert dca["bots"][name]["recovery"]["latch_after_exit"] is False

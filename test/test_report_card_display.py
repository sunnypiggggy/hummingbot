from copy import deepcopy
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from live_guard import dca_live_report


NOW = datetime(2026, 10, 21, tzinfo=timezone.utc)
DCA_BOT = "dca-live-btcusdt-200"
GRID_BOT = "grid-live-fdusd-400"


@pytest.fixture
def reporting(tmp_path):
    instance = dca_live_report.UnifiedTelegramReporting.__new__(dca_live_report.UnifiedTelegramReporting)
    instance.dca_state, instance.grid_state = tmp_path/"dca", tmp_path/"grid"
    instance.bots_path = tmp_path/"bots"
    instance.inventory_status_path = tmp_path/"inventory.json"
    runtime_path = instance.bots_path/f"instances/{GRID_BOT}/data/live_grid_runtime_state.json"
    runtime_path.parent.mkdir(parents=True)
    runtime_path.write_text("{}")
    trace, inventory_reads = [], []
    inventory = {
        "sources_healthy": True, "runtime": {"robots": {DCA_BOT: {"running": True}}},
        "assets": {"BTC": {"owners": {f"dca:{DCA_BOT}": ".01"}}, "ETH": {}},
    }
    guard = {
        "last_success_at": NOW.timestamp(), "bots": {DCA_BOT: {"recovery": {"phase": "ACTIVE"}}},
        "gate_aggregate": {
            "bots": {DCA_BOT: {"controller_applied": True, "v22_buy_enabled": True,
                                "long_only_enabled": True}},
            "v22": {"healthy": True, "pairs": {"BTC-USDT": {"model_signal": "RISK_ON"}}},
            "macro": {"healthy": True, "buy_enabled": True, "sell_enabled": True},
            "capital": {"buy_ready": False, "reason": "insufficient_quote_budget"},
        },
    }
    grid = {
        "bots": {GRID_BOT: {"latest": {
            "updated_at": NOW.timestamp(), "pairs": {"BTC-FDUSD": {"pnl": 0}, "ETH-FDUSD": {"pnl": 1}},
        }}},
        "xgboost_risk_gate": {"source_healthy": True, "pairs": {
            pair: {"buy_enabled": True} for pair in ("BTC-FDUSD", "ETH-FDUSD")
        }},
    }
    runtime = {
        "technical_buy_gate": {"pairs": {
            pair: {"healthy": True, "buy_enabled": True} for pair in ("BTC-FDUSD", "ETH-FDUSD")
        }},
        "macro_gate": {"healthy": True, "paused": False},
        "order_build_status": {"BTC-FDUSD": {"state": "MAKER_WAIT"}, "ETH-FDUSD": {"state": "HEALTHY"}},
    }
    sources = {
        instance.dca_state/"guard_state.json": guard,
        instance.grid_state/"guard_state.json": grid,
        instance.grid_state/"macro_gate.json": {"pause_new_orders": False},
        runtime_path: runtime,
    }

    def load(path):
        trace.append(("load", path.relative_to(tmp_path).as_posix()))
        if path == instance.inventory_status_path:
            return deepcopy(inventory_reads.pop(0) if inventory_reads else inventory)
        return deepcopy(sources.get(path, {}))

    def history(strategy, pair, *, days):
        trace.append(("history", strategy, pair, days))
        return [{"equity": 200}]

    def trade_counts(database, pair):
        trace.append(("trade_counts", pair))
        return 1, 2, .1

    instance._load = load
    instance.outbox = SimpleNamespace(profit_history=history)
    instance._grid_trade_counts = trade_counts
    for method, label in (("_inventory_dust", "dust"), ("_inventory_row", "inventory_gate")):
        original = getattr(instance, method)

        def observe(asset, original=original, label=label):
            trace.append((label, asset))
            return original(asset)

        setattr(instance, method, observe)
    report = {"bots": [{
        "trading_pair": "BTC-USDT", "bot_name": DCA_BOT,
        "profit": {"all_time_mtm_quote": "0"}, "database_available": True,
        "database_age_seconds": 100_000, "market_data_healthy": True,
        "market_data_age_seconds": 10, "mark_price": "100000",
    }]}
    return SimpleNamespace(instance=instance, trace=trace, inventory=inventory,
                           inventory_reads=inventory_reads, guard=guard, report=report)


def test_dca_quiet_database_budget_alert_and_long_only_keep_protective_sell(reporting):
    card, = reporting.instance._dca_cards(reporting.report, NOW)
    status = card["trading_status"]
    gates = {gate["mechanism"]: gate for gate in status["gate_statuses"]}
    assert status["trading_normal"] is True
    assert status["final_permissions"] == {"buy_enabled": True, "sell_enabled": True}
    assert gates["capital_budget_gate"]["state"] == "ALERT_ONLY"
    assert gates["strategy_mode_gate"]["state"] == "LONG_ONLY"
    assert card["data_age_seconds"] == 10
    assert card["quote_asset"] == "USDT"


def test_dca_inventory_observations_stay_separate_for_owner_dust_and_gate(reporting):
    dust = deepcopy(reporting.inventory)
    dust["assets"]["BTC"].update(inventory_phase="DUST", estimated_notional="2", unattributed=".00002")
    deficit = deepcopy(reporting.inventory)
    deficit["assets"]["BTC"]["ownership_deficit"] = ".001"
    reporting.inventory_reads.extend([reporting.inventory, dust, deficit])
    card, = reporting.instance._dca_cards(reporting.report, NOW)
    assert card["owned_base"] == ".01"
    assert card["owned_inventory_mtm_quote"] == 1000
    assert card["unattributed_dust"]["estimated_notional"] == "2"
    assert card["trading_status"]["final_permissions"] == {"buy_enabled": False, "sell_enabled": False}
    assert [row for row in reporting.trace if row == ("load", "inventory.json")] == [("load", "inventory.json")] * 3


def test_grid_uses_per_pair_wall_clock_after_trade_read_before_dust(reporting, monkeypatch):
    readings = iter((10, 20))

    def clock():
        reading = next(readings)
        reporting.trace.append(("clock", reading))
        return NOW.timestamp()+reading

    monkeypatch.setattr(dca_live_report, "time", SimpleNamespace(time=clock))
    cards = reporting.instance._grid_cards(datetime.fromtimestamp(NOW.timestamp()+5000, timezone.utc))
    assert [card["data_age_seconds"] for card in cards] == [10, 20]
    assert all(card["quote_asset"] == "FDUSD" for card in cards)
    assert cards[0]["trading_status"]["system_health"] == "DEGRADED"
    assert cards[0]["trading_status"]["final_permissions"] == {"buy_enabled": True, "sell_enabled": True}
    assert cards[1]["trading_status"]["trading_normal"] is True
    assert [row for row in reporting.trace if row[0] != "load"] == [
        ("history", "grid", "BTC-FDUSD", 370), ("trade_counts", "BTC-FDUSD"),
        ("clock", 10), ("dust", "BTC"), ("inventory_gate", "BTC"),
        ("history", "grid", "ETH-FDUSD", 370), ("trade_counts", "ETH-FDUSD"),
        ("clock", 20), ("dust", "ETH"), ("inventory_gate", "ETH"),
    ]


@pytest.mark.parametrize("field,expected_reads", [("all_time", []), ("peak", [("history", "dca", "BTC-USDT", 370)])])
def test_invalid_dca_valuation_keeps_its_original_read_boundary(reporting, field, expected_reads):
    if field == "all_time":
        reporting.report["bots"][0]["profit"]["all_time_mtm_quote"] = "invalid"
    else:
        reporting.guard["bots"][DCA_BOT]["peak_equity"] = "invalid"
    with pytest.raises(ValueError):
        reporting.instance._dca_cards(reporting.report, NOW)
    assert [row for row in reporting.trace if row[0] != "load"] == expected_reads


def test_empty_dca_report_does_not_require_a_history_reader(reporting):
    del reporting.instance.outbox
    assert reporting.instance._dca_cards({"bots": []}, NOW) == []

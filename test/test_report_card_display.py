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
        "generated_at": NOW.timestamp(), "sources_healthy": True, "runtime": {"robots": {
            DCA_BOT: {"running": True}, GRID_BOT: {"running": True},
        }},
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
                           inventory_reads=inventory_reads, guard=guard, report=report,
                           grid=grid, runtime=runtime)


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
    assert all(card["trading_status"]["process_running"] is True for card in cards)
    assert [row for row in reporting.trace if row[0] != "load"] == [
        ("history", "grid", "BTC-FDUSD", 370), ("trade_counts", "BTC-FDUSD"),
        ("clock", 10), ("dust", "BTC"), ("inventory_gate", "BTC"),
        ("history", "grid", "ETH-FDUSD", 370), ("trade_counts", "ETH-FDUSD"),
        ("clock", 20), ("dust", "ETH"), ("inventory_gate", "ETH"),
    ]


def test_grid_stopped_process_is_not_inferred_from_fresh_guard_and_leftover_runtime(reporting, monkeypatch):
    monkeypatch.setattr(dca_live_report, "time", SimpleNamespace(time=lambda: NOW.timestamp() + 10))
    before = reporting.instance._grid_cards(NOW)
    reporting.inventory["runtime"]["robots"][GRID_BOT]["running"] = False
    stopped = reporting.instance._grid_cards(NOW)
    for previous, current in zip(before, stopped):
        assert current["trading_status"]["process_running"] is False
        assert current["trading_status"]["trade_mode"] == "STOPPED"
        assert current["trading_status"]["trading_normal"] is False
        # Process evidence corrects display, while source risk gates retain
        # their original decisions independently of container execution.
        assert current["trading_status"]["gate_statuses"] == previous["trading_status"]["gate_statuses"]
        assert current["trading_status"]["final_permissions"] == previous["trading_status"]["final_permissions"]
        assert current["profit"] == previous["profit"]


@pytest.mark.parametrize("observation", [None, {}, {"running": None}, {"running": "false"}, {"running": 1}])
def test_grid_missing_or_invalid_process_observation_never_claims_running(reporting, monkeypatch, observation):
    monkeypatch.setattr(dca_live_report, "time", SimpleNamespace(time=lambda: NOW.timestamp() + 10))
    if observation is None:
        reporting.inventory["runtime"]["robots"].pop(GRID_BOT)
    else:
        reporting.inventory["runtime"]["robots"][GRID_BOT] = observation
    cards = reporting.instance._grid_cards(NOW)
    assert all(card["trading_status"]["process_running"] is False for card in cards)
    assert all(card["trading_status"]["trading_normal"] is False for card in cards)


def test_grid_running_process_keeps_original_stale_snapshot_gate(reporting, monkeypatch):
    monkeypatch.setattr(dca_live_report, "time", SimpleNamespace(time=lambda: NOW.timestamp() + 120))
    reporting.inventory["generated_at"] = NOW.timestamp() + 115
    cards = reporting.instance._grid_cards(NOW)
    for card in cards:
        assert card["trading_status"]["process_running"] is True
        assert card["trading_status"]["trading_normal"] is False
        gates = {gate["mechanism"]: gate for gate in card["trading_status"]["gate_statuses"]}
        assert gates["infrastructure_integrity_breaker"]["state"] == "BLOCK"
        assert gates["infrastructure_integrity_breaker"]["buy_enabled"] is False
        assert gates["infrastructure_integrity_breaker"]["sell_enabled"] is False


@pytest.mark.parametrize("generated_at", [None, "invalid", float("nan"), float("inf"),
                                         NOW.timestamp() - 91, NOW.timestamp() + 60])
def test_grid_invalid_or_stale_process_evidence_does_not_claim_running(reporting, monkeypatch, generated_at):
    monkeypatch.setattr(dca_live_report, "time", SimpleNamespace(time=lambda: NOW.timestamp() + 10))
    before = reporting.instance._grid_cards(NOW)
    reporting.inventory["generated_at"] = generated_at
    cards = reporting.instance._grid_cards(NOW)
    for previous, current in zip(before, cards):
        assert current["trading_status"]["process_running"] is False
        assert current["trading_status"]["trading_normal"] is False
        assert current["trading_status"]["gate_statuses"] == previous["trading_status"]["gate_statuses"]
        assert current["trading_status"]["final_permissions"] == previous["trading_status"]["final_permissions"]


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


def grid_controller_mismatch(reporting):
    reporting.grid["xgboost_risk_gate"]["pairs"]["BTC-FDUSD"].update(
        buy_enabled=False, risk_off_active=True, model_signal="RISK_OFF",
    )
    reporting.runtime["portfolio_recovery"] = {
        "phase": "EXITING", "mechanism": "portfolio_drawdown_breaker",
        "scope": "portfolio", "triggered_at": NOW.timestamp() - 86400,
    }


def permission_decisions(status):
    return {gate["mechanism"]: (gate["buy_enabled"], gate["sell_enabled"])
            for gate in status["gate_statuses"]}


def test_grid_fresh_stopped_mismatch_waits_without_releasing_any_permission(reporting, monkeypatch):
    from cardputer_monitor.collector import sanitize_gates

    monkeypatch.setattr(dca_live_report, "time", SimpleNamespace(time=lambda: NOW.timestamp() + 10))
    grid_controller_mismatch(reporting)
    running = reporting.instance._grid_cards(NOW)[0]
    assert running["trading_status"]["system_health"] == "FAILED"
    reporting.inventory["runtime"]["robots"][GRID_BOT]["running"] = False
    original = deepcopy((reporting.grid, reporting.runtime, reporting.inventory))
    stopped = reporting.instance._grid_cards(NOW)[0]
    status = stopped["trading_status"]
    gates = {gate["mechanism"]: gate for gate in status["gate_statuses"]}
    controller = gates["controller_application_gate"]
    assert (controller["state"], controller["health"]) == ("WAITING", "DEGRADED")
    assert controller["enabled"] is True and controller["applicable"] is True
    assert "机器人已停止" in controller["reason"]
    assert controller["buy_enabled"] is False and controller["sell_enabled"] is False
    assert status["system_health"] == "DEGRADED"
    assert status["trade_mode"] == "STOPPED" and status["process_running"] is False
    assert status["trading_normal"] is False and status["phase"] == "EXITING"
    assert permission_decisions(status) == permission_decisions(running["trading_status"])
    assert status["final_permissions"] == running["trading_status"]["final_permissions"] == {
        "buy_enabled": False, "sell_enabled": False,
    }
    assert stopped["profit"] == running["profit"]
    assert (reporting.grid, reporting.runtime, reporting.inventory) == original
    public = next(gate for gate in sanitize_gates(status["gate_statuses"])
                  if gate["mechanism"] == "controller_application_gate")
    assert (public["state"], public["health"]) == ("WAITING", "DEGRADED")
    assert public["buy_enabled"] is False and public["sell_enabled"] is False


@pytest.mark.parametrize("observation", [None, {}, {"running": None}, {"running": "false"},
                                        {"running": "true"}, {"running": 0}, {"running": 1},
                                        {"running": True}])
def test_grid_mismatch_requires_an_explicit_boolean_stopped_observation(reporting, monkeypatch, observation):
    monkeypatch.setattr(dca_live_report, "time", SimpleNamespace(time=lambda: NOW.timestamp() + 10))
    grid_controller_mismatch(reporting)
    if observation is None:
        reporting.inventory["runtime"]["robots"].pop(GRID_BOT)
    else:
        reporting.inventory["runtime"]["robots"][GRID_BOT] = observation
    status = reporting.instance._grid_cards(NOW)[0]["trading_status"]
    controller = next(gate for gate in status["gate_statuses"]
                      if gate["mechanism"] == "controller_application_gate")
    assert (controller["state"], controller["health"]) == ("MISMATCH", "FAILED")
    assert status["system_health"] == "FAILED"
    assert status["final_permissions"] == {"buy_enabled": False, "sell_enabled": False}


@pytest.mark.parametrize("age,waiting", [(-5, True), (90, True), (-5.001, False), (90.001, False)])
def test_grid_stopped_controller_waiting_respects_exact_observation_age_boundary(reporting, monkeypatch, age, waiting):
    observed = NOW.timestamp() + 10
    monkeypatch.setattr(dca_live_report, "time", SimpleNamespace(time=lambda: observed))
    grid_controller_mismatch(reporting)
    reporting.inventory["runtime"]["robots"][GRID_BOT]["running"] = False
    reporting.inventory["generated_at"] = observed - age
    status = reporting.instance._grid_cards(NOW)[0]["trading_status"]
    controller = next(gate for gate in status["gate_statuses"]
                      if gate["mechanism"] == "controller_application_gate")
    assert controller["state"] == ("WAITING" if waiting else "MISMATCH")
    assert status["system_health"] == ("DEGRADED" if waiting else "FAILED")
    assert status["trade_mode"] == "STOPPED"


@pytest.mark.parametrize("generated_at", [None, "invalid", float("nan"), float("inf")])
def test_grid_stopped_mismatch_with_invalid_source_time_remains_failed(reporting, monkeypatch, generated_at):
    monkeypatch.setattr(dca_live_report, "time", SimpleNamespace(time=lambda: NOW.timestamp() + 10))
    grid_controller_mismatch(reporting)
    reporting.inventory["runtime"]["robots"][GRID_BOT]["running"] = False
    reporting.inventory["generated_at"] = generated_at
    status = reporting.instance._grid_cards(NOW)[0]["trading_status"]
    controller = next(gate for gate in status["gate_statuses"]
                      if gate["mechanism"] == "controller_application_gate")
    assert (controller["state"], controller["health"]) == ("MISMATCH", "FAILED")
    assert status["system_health"] == "FAILED"


@pytest.mark.parametrize("fault", ["inventory", "stale_snapshot", "monitor", "execution", "v22_source"])
def test_grid_stopped_waiting_never_hides_an_independent_failed_gate(reporting, monkeypatch, fault):
    monkeypatch.setattr(dca_live_report, "time", SimpleNamespace(time=lambda: NOW.timestamp() + 10))
    grid_controller_mismatch(reporting)
    if fault == "inventory":
        reporting.inventory["sources_healthy"] = False
    elif fault == "stale_snapshot":
        reporting.grid["bots"][GRID_BOT]["latest"]["updated_at"] = NOW.timestamp() - 100
    elif fault == "monitor":
        reporting.grid["last_monitor_error"] = "test-only monitor error"
    elif fault == "execution":
        reporting.runtime["order_build_status"]["BTC-FDUSD"]["state"] = "FAILED"
    else:
        reporting.grid["xgboost_risk_gate"]["source_healthy"] = False
    running = reporting.instance._grid_cards(NOW)[0]["trading_status"]
    reporting.inventory["runtime"]["robots"][GRID_BOT]["running"] = False
    status = reporting.instance._grid_cards(NOW)[0]["trading_status"]
    assert status["system_health"] == "FAILED" and status["trade_mode"] == "STOPPED"
    assert any(gate["health"] == "FAILED" and gate["mechanism"] != "controller_application_gate"
               for gate in status["gate_statuses"])
    assert permission_decisions(status) == permission_decisions(running)
    assert status["final_permissions"] == running["final_permissions"]


def test_grid_stopped_without_a_frozen_controller_record_remains_failed(reporting, monkeypatch):
    monkeypatch.setattr(dca_live_report, "time", SimpleNamespace(time=lambda: NOW.timestamp() + 10))
    grid_controller_mismatch(reporting)
    reporting.runtime["technical_buy_gate"]["pairs"].pop("BTC-FDUSD")
    reporting.inventory["runtime"]["robots"][GRID_BOT]["running"] = False
    status = reporting.instance._grid_cards(NOW)[0]["trading_status"]
    controller = next(gate for gate in status["gate_statuses"]
                      if gate["mechanism"] == "controller_application_gate")
    assert (controller["state"], controller["health"]) == ("MISMATCH", "FAILED")
    assert status["system_health"] == "FAILED" and status["trade_mode"] == "STOPPED"
    assert status["final_permissions"] == {"buy_enabled": False, "sell_enabled": False}

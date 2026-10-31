from copy import deepcopy
from decimal import Decimal
import json
from pathlib import Path
import sqlite3
import sys
import time
from unittest.mock import Mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from live_guard.account_inventory import canonical_sha256
from live_guard.dca_live_guard import Guard as DcaGuard
from live_guard.grid_fill_reconciliation import fold_flatten_fills, read_grid_accounting_view
from live_guard.grid_live_guard import Guard as GridGuard


NOW = 1792658250
PAIRS = ("BTC-FDUSD", "ETH-FDUSD")


def evidence():
    runtime = {"schema_version": 13, "updated_at": NOW + 2, "trading_pairs": list(PAIRS),
        "portfolio_recovery": {"phase": "EXITING", "mechanism": "portfolio_drawdown_breaker",
            "scope": "portfolio", "triggered_at": NOW, "first_exit_order_at": NOW + 2},
        "pair_recovery": {pair: {"phase": "ACTIVE"} for pair in PAIRS},
        "flatten_order_ids": dict(zip(PAIRS, ("flatten-btc", "flatten-eth"))),
        "reentry_intents": {"BTC-FDUSD": {"state": "COMPLETED", "submitted_at": NOW - 20,
            "terminal_observed_at": NOW - 19, "reconciled_at": NOW - 18}}, "ledgers": {}}
    orders, fills = [], []
    source = {"market": "binance", "strategy": "walk_forward_portfolio_grid_live",
              "config_file_path": "walk_forward_portfolio_grid_live_fdusd_400.yml"}
    for pair, base, initial, quantity, price, fees in (
        (PAIRS[0], "0.00167971", "0.00156", (1000, 670), 82053040000, ("0.08205304", "0.05497554")),
        (PAIRS[1], "0.055976", "0.052", (55900,), 2410234000, ("0.13473242",)),
    ):
        order_id = runtime["flatten_order_ids"][pair]
        runtime["ledgers"][pair] = {"trading_pair": pair, "base": base,
            "initial_base": initial, "quote": "70", "base_cost_quote": "140",
            "fees_quote": "1", "sells": 10, "open_order_ids": [order_id]}
        orders.append({**source, "id": order_id, "symbol": pair, "order_type": "MARKET",
            "amount": sum(quantity), "last_status": "SellOrderCompleted",
            "creation_timestamp": (NOW + 2) * 1000 + 100,
            "last_update_timestamp": (NOW + 2) * 1000 + 200})
        for index, (amount, fee) in enumerate(zip(quantity, fees)):
            fills.append({**source, "order_id": order_id, "symbol": pair,
                "trade_type": "SELL", "order_type": "MARKET", "amount": amount,
                "price": price, "timestamp": (NOW + 2) * 1000,
                "exchange_trade_id": pair + str(index),
                "trade_fee": json.dumps({"percent": "0", "percent_token": "FDUSD",
                    "flat_fees": [{"token": "FDUSD", "amount": fee}]}),
                "trade_fee_in_quote": int(Decimal(fee) * 1000000)})
    return runtime, orders, fills


def database(tmp_path):
    runtime, orders, fills = evidence()
    path = tmp_path / "grid.sqlite"
    with sqlite3.connect(path) as db:
        for table, rows in (("Order", orders), ("TradeFill", fills)):
            keys = list(rows[0])
            db.execute(f'CREATE TABLE "{table}" ({",".join(keys)})')
            db.executemany(f'INSERT INTO "{table}" VALUES({",".join("?" for _ in keys)})',
                           [tuple(row[k] for k in keys) for row in rows])
    state = tmp_path / "live_grid_runtime_state.json"
    state.write_text(json.dumps(runtime), encoding="utf-8")
    return path, state, runtime


def test_complete_same_second_flatten_view_is_repeatable_and_charges_exact_quote_fees_once():
    runtime, orders, fills = evidence()
    original = deepcopy(runtime)
    view, receipt = fold_flatten_fills(runtime, orders, fills, stopped=True)
    assert receipt["status"] == "applied"
    assert Decimal(view["ledgers"][PAIRS[0]]["base"]) == Decimal("0.00000971")
    assert Decimal(view["ledgers"][PAIRS[1]]["base"]) == Decimal("0.000076")
    assert Decimal(view["ledgers"][PAIRS[0]]["fees_quote"]) == Decimal("1.13702858")
    assert Decimal(view["ledgers"][PAIRS[0]]["quote"]) == (
        Decimal("70") + Decimal("82053.04") * Decimal("0.00167") - Decimal("0.13702858"))
    assert runtime == original
    assert fold_flatten_fills(runtime, orders, fills, stopped=True) == (view, receipt)
    assert view["portfolio_recovery"] == runtime["portfolio_recovery"]
    assert view["updated_at"] == runtime["updated_at"]


def test_a_checkpoint_with_exit_marker_is_never_replayed_again():
    runtime, orders, fills = evidence()
    booked, _ = fold_flatten_fills(runtime, orders, fills, stopped=True)
    booked["portfolio_recovery"]["last_exit_fill"] = {"amount": "0.0559"}
    view, receipt = fold_flatten_fills(booked, orders, fills, stopped=True)
    assert view == booked
    assert receipt["status"] == "unchanged"


@pytest.mark.parametrize("fault", [
    "running", "unknown_writer", "partial", "cancelled", "duplicate", "foreign_fill",
    "old_order", "wrong_pair", "foreign_config", "negative", "nan_fee", "base_fee",
    "percent_fee", "missing_identity", "booked_base", "pending_reentry", "late_checkpoint",
    "missing_checkpoint_time", "unsupported_schema", "missing_order", "excessive_fee", "malformed_fee",
])
def test_ambiguous_candidate_never_changes_the_original_ledger(fault):
    runtime, orders, fills = evidence()
    stopped = True
    if fault == "running": stopped = False
    elif fault == "unknown_writer": stopped = None
    elif fault == "partial": fills.pop(0)
    elif fault == "cancelled": orders[0]["last_status"] = "OrderCancelled"
    elif fault == "duplicate": fills.append(deepcopy(fills[0]))
    elif fault == "foreign_fill": fills[0]["order_id"] = "ordinary-order"
    elif fault == "old_order": orders[0]["creation_timestamp"] = (NOW - 10) * 1000
    elif fault == "wrong_pair": fills[0]["symbol"] = "BTC-USDT"
    elif fault == "foreign_config": fills[0]["config_file_path"] = "another-bot.yml"
    elif fault == "negative": fills[0]["amount"] = -1
    elif fault == "nan_fee": fills[0]["trade_fee"] = '{"percent":"NaN"}'
    elif fault == "base_fee": fills[0]["trade_fee"] = '{"percent":0,"flat_fees":[{"token":"BTC","amount":".0001"}]}'
    elif fault == "percent_fee": fills[0]["trade_fee"] = '{"percent":".001"}'
    elif fault == "missing_identity": fills[0]["exchange_trade_id"] = ""
    elif fault == "booked_base": runtime["ledgers"][PAIRS[0]]["base"] = ".00000971"
    elif fault == "pending_reentry": runtime["reentry_intents"][PAIRS[0]]["state"] = "SUBMITTED"
    elif fault == "late_checkpoint": runtime["reentry_intents"][PAIRS[0]]["terminal_observed_at"] = NOW + 1
    elif fault == "missing_checkpoint_time": runtime["reentry_intents"][PAIRS[0]].pop("reconciled_at")
    elif fault == "unsupported_schema": runtime["schema_version"] = 12
    elif fault == "missing_order": orders.pop(0)
    elif fault == "excessive_fee":
        fills[0]["trade_fee"] = '{"percent":0,"flat_fees":[{"token":"FDUSD","amount":"1000000"}]}'
        fills[0]["trade_fee_in_quote"] = 1000000000000
    elif fault == "malformed_fee": fills[0]["trade_fee"] = '{"percent":0,"flat_fees":[null]}'
    before = deepcopy(runtime)
    view, receipt = fold_flatten_fills(runtime, orders, fills, stopped=stopped)
    assert receipt["status"] == "unverifiable"
    assert view == before == runtime


def test_sql_reader_is_read_only_and_rejects_a_changed_checkpoint(tmp_path):
    path, state, runtime = database(tmp_path)
    before = path.read_bytes()
    assert read_grid_accounting_view(runtime, path, stopped=True, runtime_path=state)[1]["status"] == "applied"
    assert path.read_bytes() == before
    changed = deepcopy(runtime)
    changed["updated_at"] += 1
    state.write_text(json.dumps(changed), encoding="utf-8")
    view, receipt = read_grid_accounting_view(runtime, path, stopped=True, runtime_path=state)
    assert receipt["status"] == "unverifiable"
    assert view == runtime


def test_guard_completes_existing_dust_path_without_economic_requests(tmp_path):
    path, state, runtime = database(tmp_path)
    guard = GridGuard.__new__(GridGuard)
    guard.database = Mock(return_value=path)
    guard.price = Mock(side_effect=lambda pair: Decimal("82000") if pair.startswith("BTC") else Decimal("2400"))
    guard.state = {"bots": {"grid-live-fdusd-400": {"action_complete": False, "last_action_error": "old"}}}
    guard.emergency_docker = Mock()
    guard.emergency_docker.matching_containers.return_value = []
    guard.emergency_exchange = Mock()
    guard.emergency_exchange.open_orders.return_value = []
    guard.api = Mock()
    guard.api.active_containers.return_value = []
    guard.market_filter = Mock(return_value=(Decimal(".00001"), Decimal("5")))
    guard.manifest = {"reservations": {"FDUSD": {"base": {"BTC": ".00156", "ETH": ".052"}}}}
    guard.audit = Mock()
    guard.save = Mock()
    snapshot = guard.snapshot("FDUSD")
    assert snapshot["runtime_state_sha256"] == canonical_sha256(runtime)
    assert snapshot["flatten_fill_reconciliation"]["status"] == "applied"
    assert snapshot["recovery_episodes"]["PORTFOLIO"]["triggered_at"] == NOW
    guard.trip("FDUSD", "existing recoverable PORTFOLIO exit incomplete", snapshot)
    bot = guard.state["bots"]["grid-live-fdusd-400"]
    assert bot["action_complete"] is True
    assert bot["tripped"] is True
    assert bot["flatten"] == {p: "dust" for p in PAIRS}
    guard.emergency_exchange.market_order.assert_not_called()
    guard.emergency_exchange.cancel_all_orders.assert_not_called()
    assert json.loads(state.read_text()) == runtime


@pytest.mark.parametrize("complete", [True, False])
def test_both_consumers_use_the_same_grid_view_without_borrowing_dca_inventory(tmp_path, complete):
    path, state, runtime = database(tmp_path)
    if not complete:
        with sqlite3.connect(path) as db:
            db.execute("DELETE FROM TradeFill WHERE rowid=1")
    guard = DcaGuard.__new__(DcaGuard)
    guard.grid_reservations_path = tmp_path / "reservations.json"
    guard.grid_inventory_state_path = tmp_path / "grid-guard.json"
    guard.managed_inventory_path = tmp_path / "managed.json"
    for file in (guard.grid_reservations_path, guard.grid_inventory_state_path, guard.managed_inventory_path):
        file.write_text("{}")
    guard._database = Mock(return_value=path)
    guard.emergency_docker = Mock()
    guard.emergency_docker.matching_containers.side_effect = lambda name: [] if name.startswith("grid") else [name]
    guard.state = {"bots": {name: {"latest": {"observed_at": time.time()},
        "equity_ledger": {"schema": "dca-equity-ledger-v1", "reconciliation_status": "RECONCILED",
                          "owned_base": quantity}}
        for name, quantity in (("dca-live-btcusdt-200", ".001131882327"), ("dca-live-ethusdt-200", ".038452820907"))}}
    ownership, _, healthy = guard._inventory_evidence()
    assert healthy is complete
    assert ownership["BTC"]["grid:grid-live-fdusd-400"] == Decimal(".00000971" if complete else ".00167971")
    assert ownership["ETH"]["grid:grid-live-fdusd-400"] == Decimal(".000076" if complete else ".055976")
    assert ownership["BTC"]["dca:dca-live-btcusdt-200"] == Decimal(".001131882327")
    assert ownership["ETH"]["dca:dca-live-ethusdt-200"] == Decimal(".038452820907")
    assert json.loads(state.read_text()) == runtime


def test_unverifiable_current_exit_never_reaches_guard_compensation(tmp_path):
    path, state, runtime = database(tmp_path)
    with sqlite3.connect(path) as db:
        db.execute('DELETE FROM TradeFill WHERE rowid=1')
    guard = GridGuard.__new__(GridGuard)
    guard.database = Mock(return_value=path)
    guard.emergency_docker = Mock()
    guard.emergency_docker.matching_containers.return_value = []
    guard.emergency_exchange = Mock()
    guard.state = {"bots": {"grid-live-fdusd-400": {}}}
    guard.price = Mock(return_value=Decimal("82000"))
    with pytest.raises(RuntimeError, match="accounting unverifiable"):
        guard.snapshot("FDUSD")
    guard.emergency_exchange.market_order.assert_not_called()
    guard.emergency_exchange.cancel_all_orders.assert_not_called()
    assert json.loads(state.read_text()) == runtime

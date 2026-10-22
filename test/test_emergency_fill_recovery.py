"""Replay the Oct 8 filled-but-unverified incident against real SQLite files."""
import json
import sqlite3
import sys
import time
from decimal import Decimal
from pathlib import Path
from unittest.mock import Mock

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "live_guard"))
sys.path.insert(0, str(ROOT / "scripts"))
from account_inventory import UnifiedInventoryLedger
from dca_live_guard import Guard, LIVE_PAIRS
from emergency_execution import execute_market_liquidation, ExecutionPending

BOT = LIVE_PAIRS["ETH-USDT"].bot_name
PAIR = "ETH-USDT"
ORDER = {
    "orderId": 50383325305, "status": "FILLED", "side": "SELL",
    "executedQty": "0.05700000", "cummulativeQuoteQty": "138.61488000",
    "fills": [
        {"tradeId": 4417694127, "qty": "0.005", "price": "2431.84", "commission": "0.01215920", "commissionAsset": "USDT"},
        {"tradeId": 4417694128, "qty": "0.052", "price": "2431.84", "commission": "0.12645568", "commissionAsset": "USDT"},
    ],
}


def fixture(tmp_path):
    ledger = UnifiedInventoryLedger(tmp_path / "shared")
    ledger.bind_account("test-account")
    ledger.start_job(job_id="incident", asset="ETH", scope=f"dca:{BOT}", pair=PAIR,
                     requested_quantity=Decimal("0.057"), client_order_id="inv-original")
    ledger.start_attempt(job_id="incident", sequence=0, client_order_id="inv-original", requested_quantity="0.057")
    ledger.finish_attempt(job_id="incident", sequence=0, status="FILLED", response=ORDER)
    ledger.finish_job("incident", status="FAILED", error="original four-part check failed")
    database = tmp_path / "bot.sqlite"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE TradeFill (trade_type TEXT,price REAL,amount REAL,trade_fee_in_quote REAL,timestamp REAL,trade_fee TEXT,symbol TEXT,exchange_trade_id TEXT)")
    guard = Guard.__new__(Guard)
    guard.state = {"bots": {BOT: {"managed_base_target": "0.05701442090706493699903730061", "emergency_adjustments": [],
                                   "recovery": {"phase": "EXITING"}, "peak_equity": "208", "pnl_offset_quote": "-26.8"}}}
    guard.inventory_ledger = ledger
    guard.emergency_exchange = Mock()
    guard.emergency_exchange.order_by_client_id.return_value = ORDER
    guard.emergency_exchange.open_orders.return_value = []
    guard.emergency_exchange.account_balances.return_value = {"ETH": {"total": Decimal("0.04048400")}}
    guard._database = lambda name: database if name == BOT else None
    guard._price = lambda pair: Decimal("2500")
    guard._managed_base_target = lambda pair: Decimal("0.05701442090706493699903730061")
    guard._latest_stop_loss_event = lambda database: {}
    guard._lot_filter = lambda pair: (Decimal("0.0001"), Decimal("5"))
    guard._save = Mock()
    guard._audit = Mock()
    guard.runtime_errors = Mock()
    return guard, ledger, database


def test_exact_incident_recovers_facts_without_selling_and_survives_restart(tmp_path):
    guard, ledger, database = fixture(tmp_path)
    for index in range(3):
        guard._recover_pending_exit_fills(time.time() + index * 20)
        snapshot = guard._snapshot(BOT, PAIR)
        assert Decimal(snapshot["owned_base"]) == Decimal("0.00001442090706493699903730061")
        assert Decimal(snapshot["quote_balance"]) == Decimal("95") + Decimal("138.47626512")
        assert Decimal(snapshot["fees_quote"]) == Decimal("0.13861488")
    assert len(ledger.economic_fills(f"dca:{BOT}")) == 2
    assert guard.state["bots"][BOT]["peak_equity"] == "208"
    assert guard.state["bots"][BOT]["pnl_offset_quote"] == "-26.8"
    guard.inventory_ledger = UnifiedInventoryLedger(tmp_path / "shared")
    assert guard._snapshot(BOT, PAIR)["owned_base"] == snapshot["owned_base"]
    guard.emergency_exchange.market_order.assert_not_called()
    assert ledger.get_job("incident")["status"] == "EXECUTED_PENDING_VERIFICATION"
    assert "original" in ledger.get_job("incident")["error"]


def test_failed_check_preserves_execution_and_verification_flags(tmp_path):
    guard, ledger, _ = fixture(tmp_path)
    guard._recover_pending_exit_fills(time.time())
    flags = {"order_verified": True, "balance_verified": False, "no_active_orders": True, "requested_quantity_verified": True}
    ledger.finish_job("incident", status="FAILED", error="delayed balance", verification=flags)
    job = ledger.get_job("incident")
    assert job["executed_quantity"] == "0.05700000"
    assert job["quote_quantity"] == "138.61488000"
    assert json.loads(job["verification_json"])["balance_verified"] is False
    assert not ledger.completed_job_verified(job)


def test_normal_fill_overlap_not_counted_twice(tmp_path):
    guard, ledger, database = fixture(tmp_path)
    guard._recover_pending_exit_fills(time.time())
    with sqlite3.connect(database) as connection:
        for fill in ORDER["fills"]:
            connection.execute("INSERT INTO TradeFill VALUES(?,?,?,?,?,?,?,?)", (
                "SELL", str(Decimal(fill["price"]) * 1000000), str(Decimal(fill["qty"]) * 1000000), str(Decimal(fill["commission"]) * 1000000), time.time(),
                json.dumps({"percent": "0", "flat_fees": [{"token": "USDT", "amount": fill["commission"]}]}), PAIR, str(fill["tradeId"])))
    snapshot = guard._snapshot(BOT, PAIR)
    assert abs(Decimal(snapshot["owned_base"]) - Decimal("0.00001442090706493699903730061")) < Decimal("1e-15")
    assert abs(Decimal(snapshot["quote_balance"]) - Decimal("233.47626512")) < Decimal("1e-15")


def test_deficit_and_live_executor_still_prevent_completion(tmp_path):
    guard, ledger, _ = fixture(tmp_path)
    guard._recover_pending_exit_fills(time.time())
    snapshot = guard._snapshot(BOT, PAIR)
    status = {"sources_healthy": True, "generated_at": time.time(), "assets": {"ETH": {
        "ownership_deficit": "0", "owners": {"grid:grid-live-fdusd-400": "0.040198", f"dca:{BOT}": snapshot["owned_base"]}}}}
    ledger.status_path.write_text(json.dumps(status))
    assert not guard._verify_pending_exit_jobs(BOT, snapshot, {"terminal": False})
    status["assets"]["ETH"]["ownership_deficit"] = "0.01"
    ledger.status_path.write_text(json.dumps(status))
    assert not guard._verify_pending_exit_jobs(BOT, snapshot, {"terminal": True})
    status["assets"]["ETH"]["ownership_deficit"] = "0"
    ledger.status_path.write_text(json.dumps(status))
    assert guard._verify_pending_exit_jobs(BOT, snapshot, {"terminal": True})
    assert ledger.completed_job_verified(ledger.get_job("incident"))
    guard.emergency_exchange.market_order.assert_not_called()


def test_unknown_order_cannot_be_resubmitted(tmp_path):
    guard, ledger, _ = fixture(tmp_path)
    ledger.finish_attempt(job_id="incident", sequence=0, status="UNKNOWN", error="response lost")
    guard.emergency_exchange.order_by_client_id.return_value = None
    with ledger.lease("ETH", "test", ttl_seconds=45), pytest.raises(ExecutionPending):
        execute_market_liquidation(exchange=guard.emergency_exchange, ledger=ledger, job_id="incident", pair=PAIR, side="SELL",
                                  target_quantity=Decimal("0.057"), client_order_id="inv-original", step_size=Decimal("0.0001"),
                                  minimum_notional=Decimal("5"), mark_price=Decimal("2500"), lease_asset="ETH", lease_holder="test")
    guard.emergency_exchange.market_order.assert_not_called()


def test_conflicting_or_incomplete_trade_evidence_rejected(tmp_path):
    _, ledger, _ = fixture(tmp_path)
    ledger.record_economic_fills("incident", ORDER, side="SELL")
    changed = json.loads(json.dumps(ORDER))
    changed["fills"][0]["commission"] = "10"
    with pytest.raises(ValueError, match="conflicting"):
        ledger.record_economic_fills("incident", changed, side="SELL")
    changed["fills"] = changed["fills"][:1]
    with pytest.raises(ValueError, match="cover"):
        ledger.record_economic_fills("incident", changed, side="SELL")
    assert len(ledger.economic_fills(f"dca:{BOT}")) == 2


def test_report_consumes_same_facts_at_original_execution_time(tmp_path):
    from live_guard.dca_live_report import DcaLiveReportCollector
    guard, ledger, database = fixture(tmp_path)
    guard._recover_pending_exit_fills(time.time())
    collector = DcaLiveReportCollector(tmp_path, tmp_path)
    collector.inventory_status_path = ledger.status_path
    adjustments = collector._economic_adjustments(BOT, PAIR, database, [])
    assert len(adjustments) == 2
    assert sum((Decimal(row["quote_cashflow"]) - Decimal(row["fee_quote"]) for row in adjustments), Decimal("0")) == Decimal("138.47626512")
    assert sum((Decimal(row["base_delta"]) for row in adjustments), Decimal("0")) == Decimal("-0.057")
    assert collector._economic_adjustments(BOT, PAIR, database, [{"order_id": str(ORDER["orderId"])}]) == [{"order_id": str(ORDER["orderId"])}]


def test_report_readonly_export_does_not_require_wal_access(tmp_path):
    from live_guard.dca_live_report import read_economic_adjustments
    guard, ledger, database = fixture(tmp_path)
    guard._recover_pending_exit_fills(time.time())
    export = ledger.directory / "economic_fills_readonly.json"
    assert export.is_file()
    ledger.database_path.rename(ledger.database_path.with_suffix(".archived"))
    rows = read_economic_adjustments(ledger.database_path, BOT, PAIR, database, [])
    assert len(rows) == 2
    payload = json.loads(export.read_text())
    payload["fills"][0]["quantity"] = "1"
    export.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="integrity"):
        read_economic_adjustments(ledger.database_path, BOT, PAIR, database, [])


def test_report_rejects_stale_economic_export(tmp_path):
    from live_guard.dca_live_report import read_economic_adjustments
    guard, ledger, database = fixture(tmp_path)
    guard._recover_pending_exit_fills(time.time())
    ledger.publish_economic_fills(time.time() - 181)
    with pytest.raises(ValueError, match="stale"):
        read_economic_adjustments(ledger.database_path, BOT, PAIR, database, [])


def test_grid_order_evidence_refreshes_without_overriding_deficit(tmp_path):
    from live_guard.grid_live_guard import Guard as GridGuard
    guard = GridGuard.__new__(GridGuard)
    guard.bots_path = tmp_path / "bots"
    target = guard.bots_path / "instances/grid-live-fdusd-400/data"
    target.mkdir(parents=True)
    guard.inventory_ledger = UnifiedInventoryLedger(tmp_path / "shared")
    guard.inventory_ledger.status_path.write_text(json.dumps({
        "schema": "account-inventory-status-v3", "generated_at": time.time(), "sources_healthy": True,
        "account_fingerprint": "test-account", "evidence_sha256": "evidence", "active_order_count": 0,
        "assets": {"ETH": {"ownership_deficit": "0.056728", "owners": {"grid:grid-live-fdusd-400": "0.04"}}}}))
    guard.emergency_exchange = Mock()
    guard.emergency_exchange.verify_ready.return_value = {"spot_bnb_burn_disabled": True}
    guard.emergency_exchange.open_orders.return_value = []
    guard.emergency_exchange.account_balances.return_value = {"ETH": {"total": Decimal("0.040484")}}
    guard.emergency_exchange._signed.side_effect = lambda method, path, *args: (
        {"enableWithdrawals": False, "ipRestrict": True, "enableFutures": False, "enableMargin": False}
        if path.endswith("apiRestrictions") else [{"makerCommission": "0", "takerCommission": "0.001"}]
        if path.endswith("tradeFee") else {}
    )
    with pytest.raises(RuntimeError, match="ownership deficit"):
        guard.verify_shadow_exchange_ready(["ETH-FDUSD"])
    evidence = json.loads((target / "grid_exchange_order_snapshot.json").read_text())
    assert evidence["pairs"]["ETH-FDUSD"]["orders"] == []
    assert evidence["account_fingerprint"] == "test-account"
    assert time.time() - evidence["generated_at"] < 5
    guard.emergency_exchange.market_order.assert_not_called()

"""Risk episodes cannot borrow a previous episode's completed liquidation."""
from copy import deepcopy
from contextlib import closing
from decimal import Decimal
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import Mock
from urllib.parse import parse_qs, urlencode, urlparse
from urllib.request import ProxyHandler, Request, build_opener

try:
    from dca_live_guard import Guard
    from account_inventory import UnifiedInventoryLedger, canonical_sha256
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from live_guard.dca_live_guard import Guard
    from live_guard.account_inventory import UnifiedInventoryLedger, canonical_sha256


BOT = "dca-live-btcusdt-200"
PAIR = "BTC-USDT"
OWNED = Decimal("0.0011318823")
OTHER = Decimal("0.00000971")
AMOUNT = Decimal("0.00113000")
FLAGS = {key: True for key in ("order_verified", "balance_verified", "no_active_orders", "requested_quantity_verified")}


class LocalExchange:
    def __init__(self, test):
        self.balance, self.free = OWNED + OTHER, None
        self.orders, self.posts, self.read_ids = {}, [], []
        backend = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                request = urlparse(self.path)
                if request.path == "/order":
                    identity = parse_qs(request.query)["id"][0]
                    backend.read_ids.append(identity)
                    value = backend.orders.get(identity)
                elif request.path == "/balances":
                    value = {"BTC": {"total": str(backend.balance), "free": str(backend.balance if backend.free is None else backend.free)}}
                elif request.path == "/open-orders": value = []
                else: raise AssertionError("unexpected local read")
                self.respond(value)

            def do_POST(self):
                assert self.path == "/market"
                value = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                backend.posts.append(value)
                quantity = Decimal(value["amount"])
                order = backend.order(value["client_id"], quantity)
                backend.balance -= quantity
                self.respond(order)

            def respond(self, value):
                body = json.dumps(value).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        test.addCleanup(self.close)
        self.base = "http://127.0.0.1:" + str(self.server.server_port)
        self.opener = build_opener(ProxyHandler({}))

    def close(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=3)

    def call(self, path, data=None):
        request = Request(self.base + path, data=None if data is None else json.dumps(data).encode(),
                          headers={"Content-Type": "application/json"})
        with self.opener.open(request, timeout=3) as response:
            return json.load(response)

    def order(self, client_id, quantity, status="FILLED"):
        fee = quantity * Decimal("80000") * Decimal(".001")
        value = {"clientOrderId": client_id, "orderId": "fake-" + client_id,
                 "status": status, "side": "SELL", "executedQty": str(quantity),
                 "cummulativeQuoteQty": str(quantity * Decimal("80000")),
                 "fills": [{"tradeId": "trade-" + client_id, "qty": str(quantity), "price": "80000",
                            "commission": str(fee), "commissionAsset": "USDT"}] if quantity else []}
        self.orders[client_id] = value
        return value

    def order_by_client_id(self, pair, client_id):
        return self.call("/order?" + urlencode({"id": client_id}))

    def account_balances(self):
        return {asset: {key: Decimal(value) for key, value in row.items()}
                for asset, row in self.call("/balances").items()}

    def open_orders(self, pair):
        return self.call("/open-orders")

    def cancel_all_orders(self, pair):
        # No open orders exist in these scenarios. This is the existing
        # recoverable path's cancellation check, not an economic simulator.
        return []

    def market_order(self, pair, side, amount, client_id):
        assert pair == PAIR and side == "SELL"
        return self.call("/market", {"pair": pair, "side": side, "amount": str(amount), "client_id": client_id})


class DcaExitIdentityTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.now = time.time()
        self.ledger = UnifiedInventoryLedger(self.root / "shared")
        self.ledger.bind_account("fake-account")
        self.database = self.root / "bot.sqlite"
        with closing(sqlite3.connect(self.database)) as db, db:
            db.execute("CREATE TABLE TradeFill (trade_type TEXT,price REAL,amount REAL,trade_fee_in_quote REAL,timestamp REAL,trade_fee TEXT,symbol TEXT,exchange_trade_id TEXT)")
        self.exchange = LocalExchange(self)
        self.guard = guard = Guard.__new__(Guard)
        guard.state = {"bots": {BOT: {"managed_base_target": str(OWNED), "emergency_adjustments": [],
            "recovery": {"phase": "EXITING", "scope": "technical", "mechanism": "v22_weekly_buy_gate", "triggered_at": self.now - 10}}},
            "gate_aggregate": {"bots": {BOT: {"controller_applied": True,
                "controller_actual_buy_enabled": False, "controller_actual_sell_enabled": False}}}}
        guard.inventory_ledger = self.ledger
        guard.emergency_exchange = self.exchange
        guard._database = lambda name: self.database if name == BOT else None
        guard._price = lambda pair: Decimal("80000")
        guard._managed_base_target = lambda pair: OWNED
        guard._latest_stop_loss_event = lambda database: {}
        guard._lot_filter = lambda pair: (Decimal("0.00001000"), Decimal("5"))
        guard._exit_runtime_status = lambda snapshot, pair: {"terminal": True, "active_executors": 0, "exchange_order_count": 0}
        guard._save, guard._audit, guard._notify, guard.runtime_errors = Mock(), Mock(), Mock(), Mock()
        guard._executor_counts = lambda database: {"active_buy_executors": 0}
        guard.auto_reentry_enabled = True
        self.publish()

    def snapshot(self):
        return self.guard._snapshot(BOT, PAIR)

    def publish(self):
        snapshot = self.snapshot()
        self.ledger.reconcile(account_fingerprint="fake-account", balances=self.exchange.account_balances(),
            ownership={"BTC": {f"dca:{BOT}": Decimal(snapshot["owned_base"]), "grid:grid-live-fdusd-400": OTHER}},
            evidence_sha256="fake-current-evidence", open_order_counts={}, sources_healthy=True, now=time.time())
        return snapshot

    def process(self):
        snapshot = self.publish()
        self.guard._process_recoverable(BOT, snapshot, macro={"healthy": True, "buy_enabled": True, "sell_enabled": True},
            technical={"buy_enabled": False, "execution_authorized": True, "force_exit": True}, now=time.time())

    def old_job(self, *, completed=False, quantity=AMOUNT, status="FILLED", current=False):
        digest = canonical_sha256({"bot": BOT, "pair": PAIR, "amount": str(AMOUNT), "triggered_at": None})
        client = "inv-" + digest[:24]
        self.ledger.start_job(job_id=digest, asset="BTC", scope=f"dca:{BOT}", pair=PAIR,
            requested_quantity=AMOUNT, client_order_id=client, now=self.now - (5 if current else 10000))
        at = self.now - (5 if current else 10000)
        self.ledger.start_attempt(job_id=digest, sequence=0, client_order_id=client, requested_quantity=AMOUNT, now=at)
        order = self.exchange.order(client, quantity, status)
        self.ledger.finish_attempt(job_id=digest, sequence=0, status=status, response=order, now=at)
        self.ledger.finish_job(digest, status="COMPLETED" if completed else "FAILED", verification=FLAGS if completed else None,
                               executed_quantity=quantity, now=at)
        return digest, client, order

    def test_previous_completed_same_amount_is_not_this_episode_and_reaches_real_exit_verification(self):
        old_id, old_client, order = self.old_job(completed=True)
        self.ledger.record_economic_fills(old_id, order, side="SELL")
        # A later normal BUY rebuilt the position after that genuine old SELL.
        with closing(sqlite3.connect(self.database)) as db, db:
            db.execute("INSERT INTO TradeFill VALUES(?,?,?,?,?,?,?,?)", ("BUY", 80000000000, 1130, 0,
                       self.now - 100, '{"percent":0,"flat_fees":[]}', PAIR, "later-normal-buy"))
        old_before = self.ledger.get_job(old_id)
        historical = deepcopy(self.guard.state["bots"][BOT]["emergency_adjustments"])
        self.process()
        self.assertEqual(1, len(self.exchange.posts))
        self.assertNotEqual(old_client, self.exchange.posts[0]["client_id"])
        self.assertEqual("EXITING", self.guard.state["bots"][BOT]["recovery"]["phase"])
        self.process()
        bot = self.guard.state["bots"][BOT]
        self.assertEqual("COOLDOWN", bot["recovery"]["phase"])
        self.assertTrue(bot["action_complete"])
        self.assertTrue(all(bot["recovery"]["exit_verification"][k] for k in (
            "order_verified", "trade_verified", "balance_verified", "runtime_verified", "residual_below_exchange_minimum")))
        fees = self.snapshot()["fees_quote"]
        self.guard.inventory_ledger = UnifiedInventoryLedger(self.root / "shared")
        self.process()
        self.assertEqual("COOLDOWN", bot["recovery"]["phase"])
        self.assertEqual(0, bot["recovery"]["healthy_cycles"])
        self.assertEqual(1, len(self.exchange.posts))
        self.assertEqual(fees, self.snapshot()["fees_quote"])
        self.assertEqual(old_before, self.ledger.get_job(old_id))
        self.assertEqual(historical, bot["emergency_adjustments"])
        self.assertFalse(self.guard.state["gate_aggregate"]["bots"][BOT]["controller_actual_buy_enabled"])
        self.assertFalse(self.guard.state["gate_aggregate"]["bots"][BOT]["controller_actual_sell_enabled"])

    def test_two_new_risk_episodes_same_amount_have_distinct_ids(self):
        self.guard._flatten(self.publish(), BOT)
        first = self.exchange.posts[-1]["client_id"]
        # Isolate identity from accounting using a separately confirmed position.
        self.guard._verified_owned_base = lambda *args, **kwargs: (OWNED, {"verified": True})
        self.exchange.balance = OWNED + OTHER
        self.guard.state["bots"][BOT]["recovery"]["triggered_at"] = time.time()
        self.guard._flatten(self.snapshot(), BOT)
        self.assertNotEqual(first, self.exchange.posts[-1]["client_id"])
        self.assertEqual(2, len(self.exchange.posts))

    def test_same_episode_completed_with_unreconciled_owner_never_subtracts_or_sells_again(self):
        self.guard._flatten(self.publish(), BOT)
        self.guard._verified_owned_base = lambda *args, **kwargs: (OWNED, {"verified": True})
        self.exchange.balance = OWNED + OTHER
        self.guard.inventory_ledger = UnifiedInventoryLedger(self.root / "shared")
        with self.assertRaisesRegex(RuntimeError, "completed liquidation requires reconciled ownership"):
            self.guard._flatten(self.snapshot(), BOT)
        self.assertEqual(1, len(self.exchange.posts))
        self.assertEqual([], self.guard.state["bots"][BOT]["emergency_adjustments"])

    def test_restart_between_fill_and_final_verification_never_sells_twice(self):
        self.process()
        self.assertEqual("EXITING", self.guard.state["bots"][BOT]["recovery"]["phase"])
        restarted = Guard.__new__(Guard)
        restarted.__dict__.update(self.guard.__dict__)
        restarted.state = json.loads(json.dumps(self.guard.state))
        restarted.inventory_ledger = UnifiedInventoryLedger(self.root / "shared")
        self.guard = restarted
        self.process()
        self.assertEqual("COOLDOWN", self.guard.state["bots"][BOT]["recovery"]["phase"])
        self.assertEqual(1, len(self.exchange.posts))

    def test_upgrade_current_legacy_completed_without_accounting_never_creates_a_new_sell(self):
        job, _, _ = self.old_job(current=True, completed=True)
        before = self.ledger.get_job(job)
        with self.assertRaisesRegex(RuntimeError, "legacy liquidation requires risk-episode reconciliation"):
            self.guard._flatten(self.publish(), BOT)
        self.assertEqual([], self.exchange.posts)
        self.assertEqual([], self.guard.state["bots"][BOT]["emergency_adjustments"])
        self.assertEqual(before, self.ledger.get_job(job))
        self.assertEqual([], self.ledger.economic_fills(f"dca:{BOT}"))

    def test_upgrade_legacy_completion_cannot_be_bypassed_by_filter_format_or_quantity(self):
        self.old_job(current=True, completed=True)
        self.guard._lot_filter = lambda pair: (Decimal(".00001"), Decimal("5"))
        for owned in (OWNED, OWNED + Decimal(".0001")):
            with self.subTest(owned=str(owned)):
                self.guard._verified_owned_base = lambda *args, **kwargs: (owned, {"verified": True})
                with self.assertRaisesRegex(RuntimeError, "legacy liquidation requires risk-episode reconciliation"):
                    self.guard._flatten(self.snapshot(), BOT)
        self.assertEqual([], self.exchange.posts)

    def test_same_episode_new_identity_is_stable_when_filter_precision_changes(self):
        self.guard._flatten(self.publish(), BOT)
        self.guard._lot_filter = lambda pair: (Decimal(".00001"), Decimal("5"))
        self.guard._verified_owned_base = lambda *args, **kwargs: (OWNED, {"verified": True})
        self.exchange.balance = OWNED + OTHER
        with self.assertRaisesRegex(RuntimeError, "completed liquidation requires reconciled ownership"):
            self.guard._flatten(self.snapshot(), BOT)
        self.assertEqual(1, len(self.exchange.posts))

    def test_uncertain_historical_metadata_or_missing_same_quantity_facts_blocks_new_sell(self):
        job, _, _ = self.old_job(completed=True)
        # No economic facts cover this old completion: even its matching
        # quantity must not be discarded solely because its job dates are old.
        with self.assertRaisesRegex(RuntimeError, "legacy liquidation requires risk-episode reconciliation"):
            self.guard._flatten(self.publish(), BOT)
        with closing(sqlite3.connect(self.ledger.database_path)) as db, db:
            db.execute("UPDATE liquidation_jobs SET requested_quantity='0.00124' WHERE job_id=?", (job,))
            db.execute("UPDATE liquidation_attempts SET updated_at=? WHERE job_id=?", (time.time(), job))
        with self.assertRaisesRegex(RuntimeError, "legacy liquidation requires risk-episode reconciliation"):
            self.guard._flatten(self.publish(), BOT)
        self.assertEqual([], self.exchange.posts)

    def test_legacy_pending_filled_is_reconciled_under_original_id_without_new_sell(self):
        job, client, _ = self.old_job(current=True)
        self.exchange.balance -= AMOUNT
        self.guard._recover_pending_exit_fills(time.time())
        self.process()
        self.assertIn(client, self.exchange.read_ids)
        self.assertEqual([], self.exchange.posts)
        self.assertTrue(self.ledger.completed_job_verified(self.ledger.get_job(job)))
        self.assertEqual("COOLDOWN", self.guard.state["bots"][BOT]["recovery"]["phase"])

    def test_legacy_partial_uses_original_verified_attempt_and_residual_child(self):
        job, client, _ = self.old_job(current=True, quantity=Decimal(".00060000"), status="CANCELED")
        self.exchange.balance -= Decimal(".00060000")
        self.guard._recover_pending_exit_fills(time.time())
        self.process()
        self.assertEqual(1, len(self.exchange.posts))
        self.assertEqual(client + "-r1", self.exchange.posts[0]["client_id"])
        self.assertEqual(Decimal(".00053000"), Decimal(self.exchange.posts[0]["amount"]))
        self.process()
        self.assertTrue(self.ledger.completed_job_verified(self.ledger.get_job(job)))
        self.assertEqual("COOLDOWN", self.guard.state["bots"][BOT]["recovery"]["phase"])

    def test_unknown_legacy_order_never_gets_a_new_id_or_submission(self):
        job, client, _ = self.old_job(current=True, quantity=Decimal("0"), status="UNKNOWN")
        self.exchange.orders.pop(client)
        self.guard._recover_pending_exit_fills(time.time())
        self.process()
        self.assertIn(client, self.exchange.read_ids)
        self.assertEqual([], self.exchange.posts)
        self.assertTrue(self.guard.state["bots"][BOT]["recovery"]["ownership_reconciliation_pending"])
        self.assertEqual(client, self.ledger.attempts(job)[0]["client_order_id"])

    def test_invalid_epoch_is_rejected_before_any_new_job_or_order(self):
        for epoch in (None, 0, -1, "NaN", "Infinity", True):
            with self.subTest(epoch=epoch):
                self.guard.state["bots"][BOT]["recovery"]["triggered_at"] = epoch
                with self.assertRaisesRegex(RuntimeError, "risk episode timestamp"):
                    self.guard._flatten(self.publish(), BOT)
        self.assertEqual([], self.exchange.posts)
        self.assertEqual([], self.ledger.pending_jobs(f"dca:{BOT}"))

    def test_actual_hard_trip_timestamp_keeps_legacy_identity(self):
        self.guard.state["bots"][BOT].update(tripped=True, tripped_at=1234)
        self.guard._flatten(self.publish(), BOT)
        expected = canonical_sha256({"bot": BOT, "pair": PAIR, "amount": str(AMOUNT), "triggered_at": 1234})
        self.assertEqual("inv-" + expected[:24], self.exchange.posts[0]["client_id"])

    def test_locked_funds_or_busy_inventory_lease_never_submit(self):
        self.exchange.free = Decimal("0")
        with self.assertRaisesRegex(RuntimeError, "locked"):
            self.guard._flatten(self.publish(), BOT)
        self.exchange.free = None
        with self.ledger.lease("BTC", "another-guard", ttl_seconds=45):
            with self.assertRaises(RuntimeError):
                self.guard._flatten(self.publish(), BOT)
        self.assertEqual([], self.exchange.posts)


if __name__ == "__main__":
    unittest.main()

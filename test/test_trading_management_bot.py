import json
import sqlite3
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from unittest import TestCase

import yaml

from management_bot.approvals import ApprovalStore
from management_bot.clients import ContractReader, OperationsReportReader
from management_bot.config import Settings
from management_bot.app import TradingManagementBot
from management_bot.storage import BotStore
from management_bot.telegram_api import TelegramAPI, TelegramError


ROOT = Path(__file__).resolve().parents[1]


class TradingManagementBotDeploymentTests(TestCase):
    def test_compose_has_dedicated_secret_scoped_bot_and_no_condor(self):
        compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
        services = compose["services"]
        self.assertNotIn("condor", services)
        service = services["sunnypiggy-trade-bot"]
        self.assertEqual("sunnypiggy-trade-bot", service["container_name"])
        self.assertEqual(["telegram_management_bot_token"], service["secrets"])
        volumes = service.get("volumes", [])
        self.assertFalse(any("docker.sock" in value for value in volumes))
        self.assertFalse(any("binance" in value.lower() and "credentials" in value.lower() for value in volumes))
        self.assertNotIn("telegram_notify_bot_token", service["secrets"])

    def test_management_image_has_no_condor_or_exchange_material(self):
        dockerfile = (ROOT / "Dockerfile.trading-management-bot").read_text(encoding="utf-8")
        self.assertIn("management_bot/", dockerfile)
        self.assertNotIn("condor", dockerfile.lower())
        self.assertNotIn("binance_stocks_credentials", dockerfile)


class BotStoreTests(TestCase):
    def test_update_and_action_idempotency_survive_reopen(self):
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "bot.sqlite"
            store = BotStore(path)
            self.assertTrue(store.claim_update(42))
            self.assertFalse(store.claim_update(42))
            session = store.create_session(7, 7, "stock_order", 11)
            store.update_session(session["session_id"], step="side", payload={"symbol": "AAPL"})
            claimed, _ = store.claim_action("order-1")
            self.assertTrue(claimed)
            store.finish_action("order-1", "COMPLETE", {"order": "ok"})
            store.close()
            reopened = BotStore(path)
            self.assertEqual("AAPL", reopened.get_session(session["session_id"])["payload"]["symbol"])
            claimed, existing = reopened.claim_action("order-1")
            self.assertFalse(claimed)
            self.assertEqual("ok", existing["result"]["order"])
            reopened.close()


class ApprovalStoreTests(TestCase):
    def test_hash_bound_decision_is_atomic_and_idempotent(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            requests = root / "weekly"
            evidence = root / "evidence"
            decisions = root / "decisions"
            requests.mkdir()
            evidence.mkdir()
            release = "a" * 64
            request = {
                "schema": "ethbtc-forced-exit-default-approval-request-v1",
                "release_sha256": release,
                "model_sha256": "b" * 64,
                "review_started_at": 1,
                "review_deadline": int(time.time()) + 3600,
                "checks": {"hashes": True},
            }
            (requests / f"approval-request-{release}.json").write_text(json.dumps(request), encoding="utf-8")
            (requests / "automation_state.json").write_text(json.dumps({
                "phase": "AWAITING_APPROVAL", "candidate_release_sha256": release,
            }), encoding="utf-8")
            store = ApprovalStore(requests, evidence, decisions)
            candidate = store.find(release[:16])
            self.assertEqual("PENDING", candidate["status"])
            result = store.decide(
                candidate, "approve", operator="telegram:7", reason="",
                telegram_user_id=7, telegram_chat_id=7, telegram_update_id=9,
                telegram_callback_query_id="callback-1",
            )
            self.assertEqual("approve", result["decision"])
            self.assertTrue((decisions / "review_decision.json").is_file())
            self.assertTrue((decisions / f"{release}.json").is_file())

    def test_reject_requires_reason(self):
        store = ApprovalStore(Path("missing"), Path("missing"), Path("missing"))
        with self.assertRaisesRegex(ValueError, "reason"):
            store.decide(
                {"candidate_id": "x", "model_type": "x", "release_sha256": "x", "model_sha256": "x",
                 "request_sha256": "x"},
                "reject", operator="telegram:7", reason="", telegram_user_id=7,
                telegram_chat_id=7, telegram_update_id=1, telegram_callback_query_id="x",
            )


class ContractReaderTests(TestCase):
    def test_latched_contract_blocks_resume(self):
        snapshot = {"sources": {"grid": {"phase": "LATCHED", "last_success_at": time.time()}}}
        allowed, reason = ContractReader.resume_allowed("grid", snapshot)
        self.assertFalse(allowed)
        self.assertIn("锁存", reason)


class OperationsReportReaderTests(TestCase):
    def test_reads_fresh_owned_mtm_snapshot_and_status(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            status = root / "trading_status.json"
            status.write_text(json.dumps({
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "robots": [{"strategy": "grid", "pair": "BTC-FDUSD", "trading_normal": True}],
            }), encoding="utf-8")
            database = root / "outbox.sqlite"
            connection = sqlite3.connect(database)
            connection.execute(
                "CREATE TABLE profit_snapshot(strategy TEXT,pair TEXT,observed_at REAL,"
                "mtm_quote REAL,equity REAL,drawdown_pct REAL,payload_json TEXT)"
            )
            connection.execute(
                "INSERT INTO profit_snapshot VALUES (?,?,?,?,?,?,?)",
                ("grid", "BTC-FDUSD", time.time(), 1.25, 201.25, 0.4,
                 json.dumps({"profit": {"all_time_mtm_quote": 1.25}})),
            )
            connection.commit()
            connection.close()
            reader = OperationsReportReader(status, database, 300)
            self.assertTrue(reader.status()["robots"][0]["trading_normal"])
            self.assertEqual(1.25, reader.profits()["robots"][0]["profit"]["all_time_mtm_quote"])


class FakeTelegram:
    def __init__(self):
        self.sent = []
        self.edited = []
        self.answers = []

    def send(self, chat_id, text, rows=None):
        self.sent.append((chat_id, text, rows))
        return {"message_id": 1}

    def edit(self, chat_id, message_id, text, rows=None):
        self.edited.append((chat_id, message_id, text, rows))
        return {"message_id": message_id}

    def answer_callback(self, callback_id, text="", alert=False):
        self.answers.append((callback_id, text, alert))


class FakeStocks:
    def __init__(self):
        self.created = []

    def health(self):
        return {
            "status": "healthy", "runtime_mode": "PAPER", "connector_ready": True,
            "paper_recovery_required": False, "economic_requests_enabled": False,
            "live_authorized": False,
        }

    def whitelist(self):
        return [{"symbol": "AAPL", "enabled": True, "max_position_notional": "200"}]

    def quote(self, _symbol):
        return {"bidPrice": "199", "askPrice": "200"}

    def preview(self, config):
        return {"allowed": True, "fee_reserve": "0.35", "executor_id": config["id"]}

    def preview_order(self, payload):
        return {"allowed": True, "fee_reserve": "0.35", "executor_id": payload["id"],
                "execution_scope": "paper", "binance_economic_request": False}

    def preview_position(self, payload):
        return self.preview_order(payload)

    def create_order(self, payload):
        self.created.append(("order", payload))
        return {"executor": {"status": "RUNNING"}, "execution_scope": "paper"}

    def create_position(self, payload):
        self.created.append(("position", payload))
        return {"executor": {"status": "RUNNING"}, "execution_scope": "paper"}

    def paper_summary(self):
        return {
            "paper_run_id": "paper-run-1", "quote_health": "FRESH", "valuation_complete": True,
            "account": {
                "equity": "2000.90", "peak_equity": "2002", "drawdown_pct": "0.0549",
                "available_cash": "1900", "positions_value": "101.25", "recovery_required": False,
            },
            "windows": {
                "4h": {"pnl": "0.25", "window_complete": False},
                "24h": {"pnl": "0.90", "window_complete": True},
                "7d": {"pnl": "0.90", "window_complete": True},
                "all": {"pnl": "0.90", "window_complete": True},
            },
            "totals": {"fees": "0.35", "fill_count": 1, "open_order_count": 0,
                       "active_executor_count": 1},
            "positions": [{
                "symbol": "AAPL", "total": "0.5", "available": "0.5", "cost_quote": "100",
                "average_cost": "200", "mark_bid": "202.5", "market_value": "101.25",
                "realized_pnl": "0", "unrealized_pnl": "1.25", "fees": "0.35",
                "net_pnl": "0.90",
            }],
            "reconciliation": {"ok": True, "error": None},
        }

    def paper_trades(self, limit=20):
        return [{"symbol": "AAPL", "side": "BUY", "quantity": "0.5", "price": "200",
                 "fee_delta": "0.35", "created_at": "2026-08-22T01:02:03+00:00"}]


class FakeReports:
    def profits(self):
        rows = []
        for strategy, pair, value in (
            ("grid", "BTC-FDUSD", "1.2345"), ("grid", "ETH-FDUSD", "2.3456"),
            ("dca", "BTC-USDT", "0.4567"), ("dca", "ETH-USDT", "-0.1234"),
        ):
            rows.append({
                "strategy": strategy, "pair": pair,
                "profit": {
                    "four_hour_mtm_quote": value, "twenty_four_hour_mtm_quote": value,
                    "seven_day_mtm_quote": value, "all_time_mtm_quote": value,
                },
            })
        return {"age_seconds": 8, "robots": rows}

    def status(self):
        rows = []
        for strategy, pair in (
            ("grid", "BTC-FDUSD"), ("grid", "ETH-FDUSD"),
            ("dca", "BTC-USDT"), ("dca", "ETH-USDT"),
        ):
            gates = []
            if strategy == "dca":
                gates.append({
                    "mechanism": "capital_budget_gate", "label": "资金预算告警",
                    "applicable": True, "health": "HEALTHY", "state": "ALERT_ONLY",
                    "buy_enabled": True, "sell_enabled": True,
                    "reason": "insufficient_quote_budget",
                })
            rows.append({
                "strategy": strategy, "pair": pair, "bot": f"{strategy}-{pair}",
                "trading_normal": True, "phase": "ACTIVE",
                "final_permissions": {"buy_enabled": True, "sell_enabled": True},
                "blockers": [], "gate_statuses": gates,
            })
        return {"age_seconds": 8, "robots": rows}


class FakeHummingbot:
    def status(self):
        return {"data": {"dca-BTC-USDT": {"error_logs": [{
            "timestamp": time.time() - 3600, "msg": "Not enough budget to create DCA."
        }]}}}


class TelegramFlowTests(TestCase):
    def _bot(self, root: Path, *, paper_enabled: bool = False) -> TradingManagementBot:
        token = root / "token"
        token.write_text("123456:test-token", encoding="utf-8")
        guard = root / "guard.json"
        guard.write_text(json.dumps({"last_success_at": time.time(), "phase": "ACTIVE"}), encoding="utf-8")
        inventory = root / "inventory.json"
        inventory.write_text(json.dumps({"healthy": True}), encoding="utf-8")
        settings = Settings(
            token_file=token,
            admin_user_id=7,
            state_dir=root / "state",
            health_path=root / "state" / "health.json",
            hummingbot_api_url="http://api",
            hummingbot_api_username="u",
            hummingbot_api_password="p",
            stocks_api_url="http://stocks",
            stocks_api_username="u",
            stocks_api_password="p",
            grid_guard_path=guard,
            dca_guard_path=guard,
            inventory_path=inventory,
            approval_request_root=root / "weekly",
            approval_evidence_root=root / "evidence",
            approval_decision_root=root / "decisions",
            mutations_enabled=False,
            bots={"grid": {"bot_name": "grid", "script": "grid", "conf": "grid"}},
            stocks_paper_trading_enabled=paper_enabled,
        )
        bot = TradingManagementBot(settings)
        bot.telegram = FakeTelegram()
        bot.stocks = FakeStocks()
        return bot

    def test_private_owner_only_and_duplicate_update_suppression(self):
        with tempfile.TemporaryDirectory() as raw:
            bot = self._bot(Path(raw))
            bot.handle_update({
                "update_id": 1,
                "message": {"from": {"id": 7}, "chat": {"id": 7, "type": "private"}, "text": "/start"},
            })
            bot.handle_update({
                "update_id": 1,
                "message": {"from": {"id": 7}, "chat": {"id": 7, "type": "private"}, "text": "/start"},
            })
            self.assertEqual(1, len(bot.telegram.sent))
            bot.handle_update({
                "update_id": 2,
                "message": {"from": {"id": 8}, "chat": {"id": 8, "type": "private"}, "text": "/start"},
            })
            self.assertEqual(1, len(bot.telegram.sent))
            bot.store.close()

    def test_stock_order_preview_is_inline_and_read_only_does_not_submit(self):
        with tempfile.TemporaryDirectory() as raw:
            bot = self._bot(Path(raw))
            text, rows = bot._new_stock_session("stock_order", 7, 7, 11)
            self.assertIn("请选择", text)
            sid = rows[0][0][1].split(":")[1]
            session = bot.store.get_session(sid)
            bot._stock_order_step(session, "symbol", "AAPL")
            session = bot.store.get_session(sid)
            bot._stock_order_step(session, "side", "BUY")
            session = bot.store.get_session(sid)
            bot._stock_order_step(session, "otype", "MARKET")
            session = bot.store.get_session(sid)
            preview, _ = bot._stock_order_step(session, "amount", "100")
            self.assertIn("权威预检通过", preview)
            session = bot.store.get_session(sid)
            result, _ = bot._stock_order_step(session, "confirm", "-")
            self.assertIn("Paper下单开关已关闭", result)
            bot.store.close()

    def test_stock_paper_trading_is_independent_from_global_mutation_switch(self):
        with tempfile.TemporaryDirectory() as raw:
            bot = self._bot(Path(raw), paper_enabled=True)
            _, rows = bot._new_stock_session("stock_order", 7, 7, 11)
            sid = rows[0][0][1].split(":")[1]
            bot._stock_order_step(bot.store.get_session(sid), "symbol", "AAPL")
            bot._stock_order_step(bot.store.get_session(sid), "side", "BUY")
            bot._stock_order_step(bot.store.get_session(sid), "otype", "MARKET")
            preview, _ = bot._stock_order_step(bot.store.get_session(sid), "amount", "100")
            self.assertIn("执行范围：PAPER", preview)
            self.assertIn("不会发送Binance真实", preview)
            result, _ = bot._stock_order_step(bot.store.get_session(sid), "confirm", "-")
            self.assertIn("PAPER创建请求已处理", result)
            self.assertEqual("order", bot.stocks.created[0][0])
            self.assertNotIn("connector_name", bot.stocks.created[0][1])
            bot.store.close()

    def test_callback_data_stays_below_telegram_limit(self):
        with tempfile.TemporaryDirectory() as raw:
            bot = self._bot(Path(raw))
            _, rows = bot._new_stock_session("stock_order", 7, 7, 11)
            for row in rows:
                for _, callback in row:
                    self.assertLessEqual(len(callback.encode()), 64)
            bot.store.close()

    def test_profit_uses_owned_mtm_snapshots_for_grid_and_dca(self):
        with tempfile.TemporaryDirectory() as raw:
            bot = self._bot(Path(raw))
            bot.reports = FakeReports()
            text = bot._profit()
            self.assertIn("BTC-FDUSD", text)
            self.assertIn("+1.2345 FDUSD", text)
            self.assertIn("ETH-USDT", text)
            self.assertIn("Stock PAPER", text)
            self.assertIn("+0.9000 USDC", text)
            self.assertNotIn("无可信数据", text)
            bot.store.close()

    def test_stock_paper_profit_positions_and_trades_are_clear(self):
        with tempfile.TemporaryDirectory() as raw:
            bot = self._bot(Path(raw))
            profit = bot._stock_paper_profit()
            positions = bot._stock_paper_positions()
            trades = bot._stock_paper_trades()
            self.assertIn("模拟交易正常", profit)
            self.assertIn("运行期不足", profit)
            self.assertIn("AAPL", positions)
            self.assertIn("净收益 +0.9000 USDC", positions)
            self.assertIn("AAPL BUY", trades)
            bot.store.close()

    def test_stock_paper_reconciliation_failure_hides_profit_values(self):
        with tempfile.TemporaryDirectory() as raw:
            bot = self._bot(Path(raw))
            bot.reports = FakeReports()
            broken = bot.stocks.paper_summary()
            broken["valuation_complete"] = False
            broken["reconciliation"] = {"ok": False, "error": "missing_position_mark:AAPL"}
            bot.stocks.paper_summary = lambda: broken
            detail = bot._stock_paper_profit()
            combined = bot._profit()
            self.assertIn("收益数值已隐藏", detail)
            self.assertIn("暂不展示收益数值", combined)
            self.assertNotIn("+0.9000 USDC", combined)
            bot.store.close()

    def test_current_errors_are_grouped_by_effect_and_ignore_old_logs(self):
        with tempfile.TemporaryDirectory() as raw:
            bot = self._bot(Path(raw))
            bot.reports = FakeReports()
            bot.hummingbot = FakeHummingbot()
            text = bot._errors()
            self.assertIn("4/4 正常交易", text)
            self.assertIn("仅告警，BUY/SELL仍放行", text)
            self.assertNotIn("Not enough budget", text)
            self.assertNotIn("交易阻塞：", text)
            bot.store.close()

    def test_risk_status_uses_final_per_robot_permissions(self):
        with tempfile.TemporaryDirectory() as raw:
            bot = self._bot(Path(raw))
            bot.reports = FakeReports()
            overview = bot._risk()
            self.assertIn("4/4 正常交易", overview)
            self.assertIn("GRID BTC-FDUSD", overview)
            self.assertIn("BUY 放行｜SELL 放行", overview)
            self.assertNotIn("合同已过期", overview)
            detail = bot._risk_detail("dca", "BTC-USDT")
            self.assertIn("✅ 正常交易", detail)
            self.assertIn("资金预算告警：提醒（不阻塞）", detail)
            self.assertIn("报价币预算不足，仅告警", detail)
            self.assertIn("当前没有阻塞交易的风控门", detail)
            for row in bot._risk_rows():
                for _, callback in row:
                    self.assertLessEqual(len(callback.encode()), 64)
            bot.store.close()


class TelegramTransportTests(TestCase):
    def test_transport_error_does_not_leak_token(self):
        api = TelegramAPI("123456:super-secret")

        class Broken:
            @staticmethod
            def post(*_args, **_kwargs):
                raise RuntimeError("network down")

        api.session = Broken()
        with self.assertRaises(TelegramError) as raised:
            api.get_updates(0)
        self.assertNotIn("super-secret", str(raised.exception))

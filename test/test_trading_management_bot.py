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

    def send_file(self, chat_id, path, caption=""):
        self.sent.append((chat_id, caption, path))
        return {"message_id": 2}


class FakeParameters:
    def catalog(self):
        return {
            "age_seconds": 2,
            "catalog_sha256": "f" * 64,
            "grid": {
                "application_state": "APPLIED", "configured_sha256": "1" * 64,
                "runtime_sha256": "2" * 64,
                "configured": {"pair_budget_quote": 200, "side_budget_quote": 100,
                               "grid_range": .06, "grid_levels": 10, "take_profit": .006,
                               "move_threshold": .015, "min_grid_move_seconds": 1800,
                               "order_refresh_time": 7200, "min_order_quote": 5.25,
                               "fee_rate": 0},
                "runtime": {"active_parameter_version": "fixed-v1"},
                "pairs": {"BTC-FDUSD": {"effective": {}}, "ETH-FDUSD": {"effective": {}}},
            },
            "dca": {
                "BTC-USDT": {"application_state": "APPLIED", "parameter_sha256": "3" * 64,
                             "effective": {"total_amount_quote": 190,
                                           "dca_spreads": [.01, .02, .04, .08],
                                           "dca_amounts": [.1, .2, .3, .4],
                                           "take_profit": .02, "stop_loss": .05,
                                           "executor_refresh_time": 18000,
                                           "time_limit": 18000, "cooldown_time": 15,
                                           "long_only_enabled": True}},
                "ETH-USDT": {"application_state": "APPLIED", "parameter_sha256": "4" * 64,
                             "effective": {}},
            },
            "risks": {"grid": {"parameters": {"strategy_loss_breaker": {
                "loss_limit_quote": "6", "cooldown_seconds": 21600,
            }}}, "dca": {"parameters": {}}, "current": {
                "grid:BTC-FDUSD": {"trading_normal": True,
                    "final_permissions": {"buy_enabled": True, "sell_enabled": True},
                    "gates": [{"mechanism": "strategy_loss_breaker",
                               "label": "策略亏损熔断", "enabled": True,
                               "state": "ALLOW", "buy_enabled": True, "sell_enabled": True}]},
            }},
            "models": {"active": {"release_sha256": "a" * 64,
                "runtime_generation": "b" * 64, "model_sha256": "c" * 64,
                "feature_schema_sha256": "d" * 64, "valid_until": "2026-08-23T00:00:00Z",
                "cutover_phase": "ACTIVE", "pairs": {"BTC-FDUSD": {
                    "model_signal": "RISK_ON", "probability": .2,
                    "entry_threshold": .5, "model_week": 39}}},
                "candidate": [{"release_sha256": "e" * 64, "status": "PENDING"}],
                "history": [{"release_sha256": "9" * 64, "effective_end": 1}]},
            "history": [{"catalog_sha256": "8" * 64,
                         "recorded_at": "2026-08-22T00:00:00Z"}],
        }

    def evidence(self):
        return {"sets": []}

    def history(self, _digest):
        value = self.catalog()
        return {"recorded_at": "2026-08-22T00:00:00Z", "catalog_sha256": "8" * 64,
                "grid": value["grid"], "dca": value["dca"], "models": value["models"]}


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
        return {"bidPrice": "199", "askPrice": "200", "eventTime": int(time.time() * 1000)}

    def market_status(self, _symbol):
        return {
            "market_phase": "MARKET_OPEN", "trading_status": "TRADING",
            "tradability": "BUY_SELL", "quote_fresh": True,
        }

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


class FakeSystemMetrics:
    def snapshot(self):
        return {
            "cpu": {"used_pct": 12.5, "cores": 4},
            "load": {"one": 0.1, "five": 0.2, "fifteen": 0.3},
            "memory": {
                "used_bytes": 2 * 1024**3, "total_bytes": 8 * 1024**3,
                "available_bytes": 6 * 1024**3, "used_pct": 25.0,
            },
            "disks": {
                "root": {
                    "used_bytes": 20 * 1024**3, "total_bytes": 100 * 1024**3,
                    "available_bytes": 80 * 1024**3, "used_pct": 20.0,
                },
                "extra": {
                    "used_bytes": 80 * 1024**3, "total_bytes": 100 * 1024**3,
                    "available_bytes": 20 * 1024**3, "used_pct": 80.0,
                },
            },
            "uptime_seconds": 90000,
            "errors": [],
        }


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

    def test_position_limit_wizard_uses_live_quote_and_price_based_risk_preview(self):
        with tempfile.TemporaryDirectory() as raw:
            bot = self._bot(Path(raw), paper_enabled=True)
            _, rows = bot._new_stock_session("stock_position", 7, 7, 11)
            sid = rows[0][0][1].split(":")[1]
            bot._stock_position_step(bot.store.get_session(sid), "symbol", "AAPL")
            bot._stock_position_step(bot.store.get_session(sid), "amount", "100")
            price_step, price_rows = bot._stock_position_step(
                bot.store.get_session(sid), "entry", "LIMIT"
            )
            self.assertIn("AAPL 最新行情", price_step)
            self.assertIn("买一 Bid：199", price_step)
            self.assertIn("卖一 Ask：200", price_step)
            self.assertIn("行情时间：", price_step)
            self.assertIn("按最新卖一 200", price_rows[0][0][0])

            barrier_step, _ = bot._stock_position_step(
                bot.store.get_session(sid), "price_latest", "-"
            )
            self.assertIn("LIMIT入场价：200", barrier_step)
            self.assertIn("止盈3% → 约 206.0000", barrier_step)
            self.assertIn("止损2% → 约 196.0000", barrier_step)
            self.assertIn("预期盈亏比：1.50:1", barrier_step)

            preview, _ = bot._stock_position_step(
                bot.store.get_session(sid), "barriers", "default"
            )
            self.assertIn("止盈触发参考：206.0000", preview)
            self.assertIn("止损触发参考：196.0000", preview)
            self.assertIn("限价相对最新卖一：0.0000%", preview)
            session = bot.store.get_session(sid)
            self.assertEqual("200", session["payload"]["request"]["entry_price"])
            result, _ = bot._stock_position_step(session, "confirm", "-")
            self.assertIn("PAPER创建请求已处理", result)
            self.assertEqual("position", bot.stocks.created[0][0])
            bot.store.close()

    def test_custom_limit_input_prompt_contains_current_market_reference(self):
        with tempfile.TemporaryDirectory() as raw:
            bot = self._bot(Path(raw))
            _, rows = bot._new_stock_session("stock_position", 7, 7, 11)
            sid = rows[0][0][1].split(":")[1]
            bot._stock_position_step(bot.store.get_session(sid), "symbol", "AAPL")
            bot._stock_position_step(bot.store.get_session(sid), "amount", "100")
            bot._stock_position_step(bot.store.get_session(sid), "entry", "LIMIT")
            session = bot.store.get_session(sid)
            bot.store.update_session(sid, step="input_price")
            context, _ = bot._stock_quote_context(session["payload"]["symbol"], "BUY")
            prompt = f"{context}\n\n请输入自定义LIMIT入场价（USDC）："
            self.assertIn("买一 Bid：199", prompt)
            self.assertIn("卖一 Ask：200", prompt)
            self.assertIn("价差：1", prompt)
            self.assertIn("数据年龄：", prompt)
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

    def test_overview_includes_host_cpu_memory_and_two_disks(self):
        with tempfile.TemporaryDirectory() as raw:
            bot = self._bot(Path(raw))
            bot.hummingbot = FakeHummingbot()
            bot.system_metrics = FakeSystemMetrics()
            text = bot._overview()
            self.assertIn("OCI宿主机资源", text)
            self.assertIn("CPU：12.5% / 4核", text)
            self.assertIn("内存：2.0/8.0 GiB （25.0%）", text)
            self.assertIn("根盘 /：20.0/100.0 GiB （20.0%）", text)
            self.assertIn("数据盘 extra_drive：80.0/100.0 GiB （80.0%）", text)
            self.assertIn("宿主机运行：1天1小时", text)
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

    def test_models_and_parameters_are_read_only_inline_pages(self):
        with tempfile.TemporaryDirectory() as raw:
            bot = self._bot(Path(raw))
            bot.parameters = FakeParameters()
            menu, rows = bot._models()
            self.assertIn("模型与参数（只读）", menu)
            self.assertIn("最近历史版本：1", menu)
            self.assertTrue(any(callback == "p:grid" for row in rows for _, callback in row))
            grid, _ = bot._grid_parameters("BTC")
            self.assertIn("Grid总范围：6.0000%", grid)
            self.assertIn("Maker费用：0.0000%", grid)
            dca, _ = bot._dca_parameters("BTC")
            self.assertIn("止损：5.0000%", dca)
            risk, _ = bot._risk_parameters("grid", "BTC")
            self.assertIn("交易结论：正常交易", risk)
            self.assertIn("最终权限：BUY=放行 / SELL=放行", risk)
            self.assertIn("亏损阈值=6", risk)
            model, _ = bot._v22_model()
            self.assertIn("概率/阈值：0.2 / 0.5", model)
            history, _ = bot._parameter_versions()
            self.assertIn("最近3个版本", history)
            for _, callback in [button for row in rows for button in row]:
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

import json
import tempfile
import time
from pathlib import Path
from unittest import TestCase

import yaml

from management_bot.approvals import ApprovalStore
from management_bot.clients import ContractReader
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
    def whitelist(self):
        return [{"symbol": "AAPL", "enabled": True, "max_position_notional": "200"}]

    def quote(self, _symbol):
        return {"bidPrice": "199", "askPrice": "200"}

    def preview(self, config):
        return {"allowed": True, "fee_reserve": "0.35", "executor_id": config["id"]}


class TelegramFlowTests(TestCase):
    def _bot(self, root: Path) -> TradingManagementBot:
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
            self.assertIn("只读观察模式", result)
            bot.store.close()

    def test_callback_data_stays_below_telegram_limit(self):
        with tempfile.TemporaryDirectory() as raw:
            bot = self._bot(Path(raw))
            _, rows = bot._new_stock_session("stock_order", 7, 7, 11)
            for row in rows:
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

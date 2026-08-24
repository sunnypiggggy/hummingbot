from datetime import datetime, timedelta, timezone
import asyncio
import json
from decimal import Decimal, ROUND_DOWN
from types import SimpleNamespace
from unittest import IsolatedAsyncioTestCase

from stocks_runtime.async_orders import AsyncStocksOrderScheduler, session_eligible
from stocks_runtime.ledger import _decode_jsonb_fields


class FakeConnector:
    def __init__(self):
        self.market_phase = "MARKET_CLOSED"
        self.quote = (Decimal("99"), Decimal("100"))
        self._trading_status = {"AAPL": "TRADING"}
        self._tradability = {"AAPL": "BOTH"}

    def latest_quote(self, _symbol):
        return self.quote

    def quantize_order_amount(self, _pair, amount):
        return Decimal(amount).quantize(Decimal("0.001"), rounding=ROUND_DOWN)


class FakePolicy:
    trading_date = "2026-08-24"

    def __init__(self):
        self.revalidated = []

    async def revalidate_reserved(self, config, *_args):
        self.revalidated.append(dict(config))
        return {"allowed": True}


class FakeService:
    def __init__(self):
        self.runtime = {}

    async def get_executor(self, executor_id):
        return self.runtime.get(executor_id)


class FakeLedger:
    def __init__(self, row):
        self.row = dict(row)
        self.transitions = []
        self.terminal = []

    async def due_scheduled_rows(self):
        return [dict(self.row)] if self.row["status"] not in {"ACTIVE", "CANCELED", "EXPIRED"} else []

    async def transition_schedule(self, schedule_id, status, **kwargs):
        expected = kwargs.get("expected")
        if expected is not None and self.row["status"] not in expected:
            return None
        self.row["status"] = status
        for key in ("target_trading_date", "executor_config", "requested_shares", "resulting_executor_id"):
            if kwargs.get(key) is not None:
                self.row[key] = kwargs[key]
        if kwargs.get("reason") is not None:
            self.row["last_block_reason"] = kwargs["reason"]
        self.transitions.append((status, dict(kwargs)))
        return dict(self.row)

    async def terminalize_schedule(self, schedule_id, status, reason):
        self.row["status"] = status
        self.row["last_block_reason"] = reason
        self.terminal.append((status, reason))
        return dict(self.row)

    async def executor_record(self, _executor_id):
        return None

    async def scheduled_record(self, schedule_id):
        return dict(self.row) if schedule_id == self.row["schedule_id"] else None


def row(*, order_type="LIMIT", budget=None):
    config = {
        "id": "tg-aapl-async-00000001", "type": "order_executor",
        "connector_name": "binance_stocks", "trading_pair": "AAPL-USDC",
        "side": 1, "amount": "1", "execution_strategy": order_type,
    }
    if order_type == "LIMIT":
        config["price"] = "98"
    return {
        "schedule_id": "sch-test", "executor_id": config["id"],
        "executor_config": config, "request_payload": {"symbol": "AAPL", "side": "BUY"},
        "target_session": "EXTENDED" if order_type == "LIMIT" else "MARKET_OPEN",
        "amount_basis": "QUOTE_BUDGET" if budget is not None else "FIXED_SHARES",
        "quote_budget": budget, "frozen_price": "98" if order_type == "LIMIT" else None,
        "requested_shares": Decimal("1"), "status": "QUEUED", "attempt_count": 0,
        "target_trading_date": None, "cancel_requested": False,
        "hard_expires_at": datetime.now(timezone.utc) + timedelta(days=7),
    }


class AsyncStocksOrderSchedulerTests(IsolatedAsyncioTestCase):
    def make_scheduler(self, value):
        connector, policy, ledger, service = FakeConnector(), FakePolicy(), FakeLedger(value), FakeService()
        created = []

        async def raw_create(config, *_args):
            created.append(dict(config))
            service.runtime[config["id"]] = {"status": "RUNNING"}
            return {"executor_id": config["id"]}

        app = SimpleNamespace(state=SimpleNamespace(
            stocks_connector=connector, stocks_policy=policy, stocks_ledger=ledger,
            stocks_settings=SimpleNamespace(mode="PAPER", live_authorized=False, disclaimer_confirmed=False),
            stocks_paper_broker=SimpleNamespace(trading_date="2026-08-24"),
            executor_service=service, stocks_raw_create=raw_create,
        ))
        return AsyncStocksOrderScheduler(app), connector, policy, ledger, created

    def test_type_specific_session_rules(self):
        self.assertTrue(session_eligible("EXTENDED", "PRE_MARKET"))
        self.assertTrue(session_eligible("EXTENDED", "POST_MARKET"))
        self.assertFalse(session_eligible("MARKET_OPEN", "PRE_MARKET"))
        self.assertTrue(session_eligible("MARKET_OPEN", "MARKET_OPEN"))

    async def test_limit_waits_closed_then_activates_in_pre_with_frozen_terms(self):
        scheduler, connector, _, ledger, created = self.make_scheduler(row(order_type="LIMIT"))
        await scheduler.tick()
        self.assertEqual("WAITING_SESSION", ledger.row["status"])
        self.assertEqual([], created)
        connector.market_phase = "PRE_MARKET"
        await scheduler.tick()
        self.assertEqual("ACTIVE", ledger.row["status"])
        self.assertEqual("98", created[0]["price"])
        self.assertEqual("1", created[0]["amount"])

    async def test_market_ignores_pre_and_recomputes_shares_from_fixed_budget_at_rth(self):
        scheduler, connector, policy, ledger, created = self.make_scheduler(
            row(order_type="MARKET", budget=Decimal("100"))
        )
        connector.market_phase = "PRE_MARKET"
        await scheduler.tick()
        self.assertEqual("WAITING_SESSION", ledger.row["status"])
        connector.market_phase = "MARKET_OPEN"
        connector.quote = (Decimal("109"), Decimal("110"))
        await scheduler.tick()
        self.assertEqual("ACTIVE", ledger.row["status"])
        self.assertEqual("0.909", created[0]["amount"])
        self.assertLessEqual(Decimal(created[0]["amount"]) * Decimal("110"), Decimal("100"))
        self.assertNotIn("price", created[0])
        self.assertEqual(1, len(policy.revalidated))

    async def test_stale_quote_never_creates_executor(self):
        scheduler, connector, _, ledger, created = self.make_scheduler(row(order_type="MARKET", budget=Decimal("100")))
        connector.market_phase = "MARKET_OPEN"
        connector.quote = None
        await scheduler.tick()
        self.assertEqual("WAITING_PREFLIGHT", ledger.row["status"])
        self.assertEqual([], created)

    async def test_cancel_is_terminal_and_idempotent(self):
        scheduler, _, _, ledger, _ = self.make_scheduler(row())
        first = await scheduler.cancel("sch-test")
        second = await scheduler.cancel("sch-test")
        self.assertEqual("CANCELED", first["schedule"]["status"])
        self.assertTrue(second["idempotent_replay"])
        self.assertEqual(1, len(ledger.terminal))

    async def test_concurrent_ticks_create_only_one_executor(self):
        scheduler, connector, _, ledger, created = self.make_scheduler(
            row(order_type="MARKET", budget=Decimal("100"))
        )
        connector.market_phase = "MARKET_OPEN"
        await asyncio.gather(scheduler.tick(), scheduler.tick())
        self.assertEqual("ACTIVE", ledger.row["status"])
        self.assertEqual(1, len(created))

    def test_asyncpg_json_text_row_is_normalized_before_scheduler_consumes_it(self):
        value = row(order_type="MARKET", budget=Decimal("100"))
        value["executor_config"] = json.dumps(value["executor_config"])
        value["request_payload"] = json.dumps(value["request_payload"])

        normalized = _decode_jsonb_fields(value, "request_payload", "executor_config")

        self.assertIsInstance(normalized["executor_config"], dict)
        self.assertIsInstance(normalized["request_payload"], dict)
        self.assertEqual("AAPL-USDC", normalized["executor_config"]["trading_pair"])

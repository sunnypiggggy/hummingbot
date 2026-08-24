from decimal import Decimal
from unittest import IsolatedAsyncioTestCase

from stocks_runtime.ledger import ACTIVE_INTENT_STATES, LedgerLimits
from stocks_runtime.policy import PolicyViolation, StocksExecutorPolicy


class FakeLedger:
    limits = LedgerLimits()

    def __init__(self):
        self.reservations = []
        self.usage = {
            "positions_mtm": Decimal("0"),
            "pending_buy_principal": Decimal("100"),
            "fee_reserve": Decimal("0"),
            "principal_and_mtm": Decimal("100"),
            "by_symbol": {"AAPL": Decimal("0")},
            "positions_by_symbol": {"AAPL": Decimal("0")},
            "quote_total": Decimal("2000"),
            "quote_available": Decimal("1900"),
        }

    async def managed_pnl(self, _):
        return Decimal("0")

    async def active_limits(self):
        return self.limits

    async def whitelist_entry(self, symbol):
        if symbol == "AAPL":
            return {
                "symbol": symbol,
                "enabled": True,
                "max_position_notional": self.limits.max_symbol_exposure,
            }
        return None

    async def set_trading_date(self, *_):
        pass

    async def daily_pnl(self, _):
        return Decimal("0")

    async def managed_exposure(self, _):
        return Decimal("100")

    async def capital_usage(self, _prices, exclude_executor_id=None):
        return self.usage

    async def managed_symbol_exposure(self, _symbol, _price):
        return Decimal("0")

    async def managed_available(self, owner, symbol):
        return Decimal("10") if owner == "unassigned" and symbol == "AAPL" else Decimal("0")

    async def reserve_intent(self, **kwargs):
        self.reservations.append(kwargs)
        return {"executor_id": kwargs["executor_id"], "status": "RESERVED"}


class StocksExecutorPolicyTests(IsolatedAsyncioTestCase):
    def setUp(self):
        self.ledger = FakeLedger()
        self.policy = StocksExecutorPolicy(self.ledger, "SHADOW", False)
        self.policy.update_market({"AAPL"}, {"AAPL": Decimal("200")}, "2026-08-21")

    def test_position_hold_is_terminal_not_a_second_exposure_reservation(self):
        self.assertNotIn("POSITION_HOLD", ACTIVE_INTENT_STATES)

    async def test_valid_order_is_shadow_only_and_reserved_idempotently(self):
        result = await self.policy.validate_and_reserve(
            {
                "id": "order-aapl-0001",
                "type": "order_executor",
                "connector_name": "binance_stocks",
                "trading_pair": "AAPL-USDC",
                "side": "BUY",
                "amount": "0.5",
                "price": "199",
                "execution_strategy": "LIMIT",
            },
            "stocks_managed",
            "stocks-runtime",
        )
        self.assertFalse(result["would_submit"])
        self.assertEqual(Decimal("99.5"), self.ledger.reservations[0]["estimated_notional"])

    async def test_paper_order_is_submitted_to_local_venue(self):
        policy = StocksExecutorPolicy(self.ledger, "PAPER", False)
        policy.update_market({"AAPL"}, {"AAPL": Decimal("200")}, "2026-08-21")
        result = await policy.validate_and_reserve(
            {
                "id": "order-aapl-paper",
                "type": "order_executor",
                "connector_name": "binance_stocks",
                "trading_pair": "AAPL-USDC",
                "side": "BUY",
                "amount": "0.5",
                "price": "199",
                "execution_strategy": "LIMIT",
            },
            "stocks_managed",
            "stocks-runtime",
        )
        self.assertTrue(result["would_submit"])
        self.assertFalse(result["live_authorized"])

    async def test_numeric_hummingbot_enums_have_same_policy_semantics(self):
        policy = StocksExecutorPolicy(self.ledger, "PAPER", False)
        policy.update_market({"AAPL"}, {"AAPL": Decimal("200")}, "2026-08-21")
        result = await policy.preview(
            {
                "id": "position-aapl-enum", "type": "position_executor",
                "connector_name": "binance_stocks", "trading_pair": "AAPL-USDC",
                "side": 1, "amount": "0.5", "entry_price": "200",
                "triple_barrier_config": {
                    "stop_loss": "0.02", "time_limit": 3600,
                    "open_order_type": 2, "take_profit_order_type": 2,
                },
            },
            "stocks_managed", "stocks-runtime",
        )
        self.assertTrue(result["allowed"])

    async def test_preview_uses_same_checks_without_reserving(self):
        result = await self.policy.preview(
            {
                "id": "order-aapl-preview",
                "type": "order_executor",
                "connector_name": "binance_stocks",
                "trading_pair": "AAPL-USDC",
                "side": "BUY",
                "amount": "0.25",
                "price": "200",
                "execution_strategy": "LIMIT",
            },
            "stocks_managed",
            "telegram-management-bot",
        )
        self.assertTrue(result["allowed"])
        self.assertTrue(result["preflight_only"])
        self.assertEqual([], self.ledger.reservations)

    async def test_fee_reserve_does_not_consume_symbol_principal_limit(self):
        self.ledger.limits = LedgerLimits(
            max_order_notional=Decimal("500"), max_symbol_exposure=Decimal("200"),
            max_managed_exposure=Decimal("2000"), daily_loss_limit=Decimal("200"),
        )
        self.ledger.usage.update({
            "principal_and_mtm": Decimal("100"), "pending_buy_principal": Decimal("100"),
            "fee_reserve": Decimal("0.35"), "by_symbol": {"AAPL": Decimal("100")},
        })
        result = await self.policy.preview({
            "id": "order-aapl-fees-01", "type": "order_executor",
            "connector_name": "binance_stocks", "trading_pair": "AAPL-USDC",
            "side": "BUY", "amount": "0.5", "price": "200", "execution_strategy": "LIMIT",
        }, "stocks_managed", "telegram-management-bot")
        self.assertTrue(result["allowed"])
        self.assertEqual("0.35", result["fee_reserve"])

    async def test_three_hundred_usdc_is_allowed_by_new_runtime_limit(self):
        self.ledger.usage.update({"principal_and_mtm": Decimal("0"), "by_symbol": {"AAPL": Decimal("0")}})
        result = await self.policy.preview({
            "id": "order-aapl-300-01", "type": "order_executor",
            "connector_name": "binance_stocks", "trading_pair": "AAPL-USDC",
            "side": "BUY", "amount": "1.5", "price": "200", "execution_strategy": "LIMIT",
        }, "stocks_managed", "telegram-management-bot")
        self.assertTrue(result["allowed"])

    async def test_buy_requires_enabled_operator_whitelist(self):
        self.policy.update_market({"AAPL", "TSLA"}, {"AAPL": Decimal("200"), "TSLA": Decimal("100")})
        with self.assertRaisesRegex(Exception, "whitelist"):
            await self.policy.preview(
                {
                    "id": "order-tsla-preview",
                    "type": "order_executor",
                    "connector_name": "binance_stocks",
                    "trading_pair": "TSLA-USDC",
                    "side": "BUY",
                    "amount": "0.5",
                    "price": "100",
                    "execution_strategy": "LIMIT",
                },
                "stocks_managed",
                None,
            )

    async def test_position_requires_stop_and_time_limit(self):
        with self.assertRaisesRegex(ValueError, "stop_loss"):
            await self.policy.validate_and_reserve(
                {
                    "id": "position-aapl-01",
                    "type": "position_executor",
                    "connector_name": "binance_stocks",
                    "trading_pair": "AAPL-USDC",
                    "side": "BUY",
                    "amount": "0.5",
                    "triple_barrier_config": {"time_limit": 3600},
                },
                "stocks_managed",
                None,
            )

    async def test_rejects_short_bstocks_and_large_order(self):
        base = {
            "id": "position-aapl-02",
            "type": "position_executor",
            "connector_name": "binance_stocks",
            "trading_pair": "AAPL-USDC",
            "side": "SELL",
            "amount": "0.5",
            "triple_barrier_config": {"stop_loss": "0.05", "time_limit": 3600},
        }
        with self.assertRaisesRegex(ValueError, "long-only"):
            await self.policy.validate_and_reserve(base, "stocks_managed", None)
        base.update({
            "id": "position-aaplb-03",
            "side": "BUY",
            "trading_pair": "AAPLB-USDC",
        })
        with self.assertRaisesRegex(ValueError, "exchangeInfo"):
            await self.policy.validate_and_reserve(base, "stocks_managed", None)
        base.update({"id": "position-aapl-04", "trading_pair": "AAPL-USDC", "amount": "3"})
        with self.assertRaises(PolicyViolation) as raised:
            await self.policy.validate_and_reserve(base, "stocks_managed", None)
        self.assertEqual("500", raised.exception.context["limit"])

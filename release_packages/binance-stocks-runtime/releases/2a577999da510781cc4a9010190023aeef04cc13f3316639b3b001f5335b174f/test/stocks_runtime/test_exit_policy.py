from decimal import Decimal
from unittest import TestCase

from hummingbot.core.data_type.common import OrderType
from hummingbot.strategy_v2.executors.binance_stocks_position_executor import BinanceStocksExitPolicy


class BinanceStocksExitPolicyTests(TestCase):
    def test_market_is_used_only_in_rth(self):
        instruction = BinanceStocksExitPolicy.instruction("MARKET_OPEN", Decimal("200"), Decimal("0.01"))
        self.assertIs(instruction.order_type, OrderType.MARKET)
        self.assertEqual("RTH_MARKET", instruction.phase)

    def test_extended_session_uses_marketable_day_limit(self):
        instruction = BinanceStocksExitPolicy.instruction("PRE_MARKET", Decimal("200"), Decimal("0.01"))
        self.assertIs(instruction.order_type, OrderType.LIMIT)
        self.assertEqual(Decimal("199.99"), instruction.price)

    def test_closed_session_remains_exit_pending(self):
        instruction = BinanceStocksExitPolicy.instruction("OVERNIGHT", Decimal("200"), Decimal("0.01"))
        self.assertIsNone(instruction.order_type)

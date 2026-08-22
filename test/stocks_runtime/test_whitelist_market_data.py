import unittest
from decimal import Decimal

from stocks_runtime.whitelist_market_data import (
    enabled_whitelist_symbols,
    executable_mid,
    refresh_whitelist_quotes,
)


class FakeLedger:
    async def whitelist_rows(self):
        return [
            {"symbol": "AAPL", "enabled": True},
            {"symbol": "TSLA", "enabled": False},
            {"symbol": "SPY", "enabled": True},
            {"symbol": "NOTLISTED", "enabled": True},
        ]


class FakeClient:
    def __init__(self):
        self.requested = []

    async def quote(self, symbol):
        self.requested.append(symbol)
        if symbol == "SPY":
            raise IOError("temporary quote error")
        return {"symbol": symbol, "bidPrice": "100", "askPrice": "102"}


class FakeConnector:
    def __init__(self):
        self.events = []

    def process_quote_event(self, event):
        self.events.append(event)


class FakeMarketDataService:
    def __init__(self):
        self.prices = {}

    def set_price(self, pair, price):
        self.prices[pair] = price


class WhitelistMarketDataTests(unittest.IsolatedAsyncioTestCase):
    def test_enabled_symbols_are_intersection_of_whitelist_and_catalog(self):
        rows = [
            {"symbol": "aapl", "enabled": True},
            {"symbol": "TSLA", "enabled": False},
            {"symbol": "MISSING", "enabled": True},
        ]
        self.assertEqual(("AAPL",), enabled_whitelist_symbols(rows, {"AAPL", "TSLA"}))

    def test_mid_uses_bbo_and_rejects_invalid_values(self):
        self.assertEqual(Decimal("101"), executable_mid({"bidPrice": "100", "askPrice": "102"}))
        self.assertEqual(Decimal("0"), executable_mid({"bidPrice": "bad", "askPrice": "102"}))

    async def test_refresh_never_requests_disabled_or_non_whitelist_symbols(self):
        client = FakeClient()
        connector = FakeConnector()
        market_data = FakeMarketDataService()
        result = await refresh_whitelist_quotes(
            client=client,
            ledger=FakeLedger(),
            available_symbols={"AAPL", "TSLA", "SPY", "QQQ"},
            connector=connector,
            market_data_service=market_data,
        )

        self.assertEqual(("AAPL", "SPY"), result.symbols)
        self.assertEqual(["AAPL", "SPY"], client.requested)
        self.assertEqual({"AAPL": Decimal("101")}, result.prices)
        self.assertIn("SPY", result.errors)
        self.assertEqual([{"symbol": "AAPL", "bidPrice": "100", "askPrice": "102"}], connector.events)
        self.assertEqual({"AAPL-USDC": Decimal("101")}, market_data.prices)


if __name__ == "__main__":
    unittest.main()

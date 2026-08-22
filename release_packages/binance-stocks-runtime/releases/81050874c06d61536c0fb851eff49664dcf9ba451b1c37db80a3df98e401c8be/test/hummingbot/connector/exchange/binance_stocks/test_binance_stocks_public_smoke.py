import os
from unittest import IsolatedAsyncioTestCase, skipUnless

from bidict import bidict

from hummingbot.connector.exchange.binance_stocks.binance_stocks_exchange import BinanceStocksExchange
from hummingbot.core.web_assistant.connections.connections_factory import ConnectionsFactory


@skipUnless(
    os.environ.get("BINANCE_STOCKS_PUBLIC_SMOKE") == "1" and os.environ.get("BINANCE_STOCKS_PUBLIC_API_KEY"),
    "explicit public smoke opt-in and read-only API key required",
)
class BinanceStocksPublicSmokeTests(IsolatedAsyncioTestCase):
    async def asyncTearDown(self):
        await ConnectionsFactory().close()

    async def test_aapl_public_quote_is_bbo_only(self):
        exchange = BinanceStocksExchange(
            os.environ["BINANCE_STOCKS_PUBLIC_API_KEY"],
            "",
            trading_pairs=["AAPL-USDC"],
            trading_required=False,
        )
        exchange._set_trading_pair_symbol_map(bidict({"AAPL": "AAPL-USDC"}))
        message = await exchange._orderbook_ds._order_book_snapshot("AAPL-USDC")
        self.assertLessEqual(len(message.bids), 1)
        self.assertLessEqual(len(message.asks), 1)
        self.assertTrue(message.bids and message.asks)

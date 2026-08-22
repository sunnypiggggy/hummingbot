from unittest import TestCase

from hummingbot.connector.exchange.binance_stocks.binance_stocks_order_book import BinanceStocksOrderBook
from hummingbot.core.data_type.order_book import OrderBook


class BinanceStocksOrderBookTests(TestCase):
    def test_each_quote_is_a_complete_one_level_snapshot(self):
        first = BinanceStocksOrderBook.snapshot_message_from_exchange(
            {"s": "AAPL", "bp": "199.90", "bs": "10", "ap": "200.00", "as": "8", "T": 1000},
            timestamp=1,
            metadata={"trading_pair": "AAPL-USDC"},
        )
        second = BinanceStocksOrderBook.snapshot_message_from_exchange(
            {"s": "AAPL", "bp": "200.10", "bs": "5", "ap": "200.20", "as": "6", "T": 2000},
            timestamp=2,
            metadata={"trading_pair": "AAPL-USDC"},
        )
        book = OrderBook()
        book.apply_snapshot(first.bids, first.asks, first.update_id)
        book.apply_snapshot(second.bids, second.asks, second.update_id)

        bids = list(book.bid_entries())
        asks = list(book.ask_entries())
        self.assertEqual(1, len(bids))
        self.assertEqual(1, len(asks))
        self.assertEqual(200.10, bids[0].price)
        self.assertEqual(200.20, asks[0].price)

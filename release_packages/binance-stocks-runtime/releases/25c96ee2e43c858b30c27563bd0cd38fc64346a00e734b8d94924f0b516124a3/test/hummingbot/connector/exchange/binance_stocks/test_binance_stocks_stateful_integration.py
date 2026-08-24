from decimal import Decimal
from unittest import IsolatedAsyncioTestCase

from bidict import bidict

from hummingbot.connector.exchange.binance_stocks import binance_stocks_constants as CONSTANTS
from hummingbot.connector.exchange.binance_stocks.binance_stocks_exchange import BinanceStocksExchange
from hummingbot.connector.exchange.binance_stocks.binance_stocks_position_provider import (
    EquityAccountSnapshot,
    EquityPosition,
    EquityPositionProvider,
)
from hummingbot.core.data_type.common import OrderType, TradeType
from hummingbot.core.data_type.in_flight_order import InFlightOrder
from hummingbot.core.web_assistant.connections.connections_factory import ConnectionsFactory

from .binance_stocks_stateful_simulator import BinanceStocksStatefulSimulator


class SimulatorPositionProvider(EquityPositionProvider):
    def __init__(self, timestamp: float):
        self.timestamp = timestamp

    @property
    def available(self) -> bool:
        return True

    async def get_snapshot(self):
        return EquityAccountSnapshot(
            positions={"AAPL": EquityPosition(Decimal("2"), Decimal("2"))},
            quote_total=Decimal("1000"),
            quote_available=Decimal("1000"),
            source="stateful-simulator",
            timestamp=self.timestamp,
        )


class BinanceStocksStatefulIntegrationTests(IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        await ConnectionsFactory().close()
        self.simulator = BinanceStocksStatefulSimulator()
        await self.simulator.start()
        self.original_rest_url = CONSTANTS.REST_URL
        self.original_ws_url = CONSTANTS.WS_URL
        CONSTANTS.REST_URL = self.simulator.base_url
        CONSTANTS.WS_URL = self.simulator.ws_url
        self.now = 1_700_000_000
        self.exchange = BinanceStocksExchange(
            "key",
            "secret",
            trading_pairs=["AAPL-USDC"],
            disclaimer_confirmed=True,
            position_provider=SimulatorPositionProvider(self.now),
        )
        self.exchange._set_current_timestamp(self.now)
        self.exchange._set_trading_pair_symbol_map(bidict({"AAPL": "AAPL-USDC"}))
        self.exchange.set_account_authorized(True)
        self.exchange.process_market_state_event({"e": "calendar", "phase": "MARKET_OPEN"})
        self.exchange.process_market_state_event({"e": "tradingStatus", "s": "AAPL", "status": "TRADING"})
        self.exchange.process_market_state_event({"e": "tradability", "s": "AAPL", "tradability": "BOTH"})
        self.exchange.process_quote_event(
            {
                "e": "quote",
                "s": "AAPL",
                "bp": "199.90",
                "ap": "200.00",
                "T": self.now * 1000,
            }
        )
        self.exchange._fractional_supported["AAPL"] = True
        await self.exchange._update_balances()

    async def asyncTearDown(self):
        await ConnectionsFactory().close()
        CONSTANTS.REST_URL = self.original_rest_url
        CONSTANTS.WS_URL = self.original_ws_url
        await self.simulator.stop()

    async def test_real_rest_assistant_signs_places_queries_and_cancels_stateful_order(self):
        exchange_order_id, _ = await self.exchange._place_order(
            "deterministic-client-id",
            "AAPL-USDC",
            Decimal("0.5"),
            TradeType.BUY,
            OrderType.LIMIT,
            Decimal("200"),
        )
        self.assertEqual("1", exchange_order_id)
        self.assertEqual(1, self.simulator.place_requests)
        self.assertEqual("EXTENDED", self.simulator.orders["deterministic-client-id"]["tradingSession"])

        tracked = InFlightOrder(
            client_order_id="deterministic-client-id",
            exchange_order_id=exchange_order_id,
            trading_pair="AAPL-USDC",
            order_type=OrderType.LIMIT,
            trade_type=TradeType.BUY,
            amount=Decimal("0.5"),
            price=Decimal("200"),
            creation_timestamp=self.now,
        )
        status = await self.exchange._request_order_status(tracked)
        self.assertEqual("OPEN", status.new_state.name)
        accepted = await self.exchange._place_cancel("deterministic-client-id", tracked)
        self.assertTrue(accepted)
        canceled = await self.exchange._request_order_status(tracked)
        self.assertEqual("CANCELED", canceled.new_state.name)

    async def test_public_quote_uses_real_http_and_builds_only_one_bbo_level(self):
        message = await self.exchange._orderbook_ds._order_book_snapshot("AAPL-USDC")
        self.assertEqual(1, len(message.bids))
        self.assertEqual(1, len(message.asks))
        self.assertEqual(199.90, message.bids[0].price)
        self.assertEqual(200.00, message.asks[0].price)

    async def test_real_websocket_and_listen_key_path_deliver_order_report(self):
        data_source = self.exchange._create_user_stream_data_source()
        key = await data_source._get_listen_key()
        self.assertEqual("test-listen-key", key)
        data_source._current_listen_key = key
        data_source._listen_key_initialized_event.set()
        websocket = await data_source._connected_websocket_assistant()
        response = await websocket.receive()
        self.assertEqual("orderReport", response.data["e"])
        await websocket.disconnect()
        await data_source.stop()

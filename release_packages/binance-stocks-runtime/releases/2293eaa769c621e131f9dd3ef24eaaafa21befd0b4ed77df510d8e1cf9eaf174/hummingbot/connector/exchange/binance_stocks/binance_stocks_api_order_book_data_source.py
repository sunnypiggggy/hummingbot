import asyncio
from typing import TYPE_CHECKING, Any, Dict, List, Optional
from urllib.parse import quote

from hummingbot.core.data_type.order_book_message import OrderBookMessage
from hummingbot.core.data_type.order_book_tracker_data_source import OrderBookTrackerDataSource
from hummingbot.core.web_assistant.connections.data_types import RESTMethod
from hummingbot.core.web_assistant.web_assistants_factory import WebAssistantsFactory
from hummingbot.core.web_assistant.ws_assistant import WSAssistant

from . import binance_stocks_constants as CONSTANTS, binance_stocks_web_utils as web_utils
from .binance_stocks_order_book import BinanceStocksOrderBook
from .binance_stocks_utils import extract_payload

if TYPE_CHECKING:
    from .binance_stocks_exchange import BinanceStocksExchange


class BinanceStocksAPIOrderBookDataSource(OrderBookTrackerDataSource):
    """BBO-only stock quote source.

    Each quote is a complete one-level snapshot.  It is deliberately not
    exposed as L2 depth or a public trade stream.
    """

    def __init__(
        self,
        trading_pairs: List[str],
        connector: "BinanceStocksExchange",
        api_factory: WebAssistantsFactory,
        domain: str = CONSTANTS.DEFAULT_DOMAIN,
    ):
        super().__init__(trading_pairs)
        self._connector = connector
        self._api_factory = api_factory
        self._domain = domain

    async def get_last_traded_prices(self, trading_pairs: List[str], domain: Optional[str] = None) -> Dict[str, float]:
        result: Dict[str, float] = {}
        for trading_pair in trading_pairs:
            quote_payload = await self._request_quote(trading_pair)
            bid = float(quote_payload.get("bp", quote_payload.get("bidPrice", 0)))
            ask = float(quote_payload.get("ap", quote_payload.get("askPrice", 0)))
            if bid <= 0 or ask <= 0:
                raise IOError(f"Incomplete Binance Stocks quote for {trading_pair}")
            result[trading_pair] = (bid + ask) / 2
        return result

    async def _request_quote(self, trading_pair: str) -> Dict[str, Any]:
        symbol = await self._connector.exchange_symbol_associated_to_pair(trading_pair=trading_pair)
        assistant = await self._api_factory.get_rest_assistant()
        response = await assistant.execute_request(
            url=web_utils.public_rest_url(CONSTANTS.QUOTE_PATH_URL, self._domain),
            params={"symbol": symbol},
            headers=self._connector.market_data_headers,
            method=RESTMethod.GET,
            throttler_limit_id=CONSTANTS.QUOTE_PATH_URL,
        )
        payload = extract_payload(response)
        if isinstance(payload, list):
            payload = payload[0] if payload else {}
        if not isinstance(payload, dict):
            raise IOError(f"Unexpected quote response for {trading_pair}: {response}")
        self._connector.process_quote_event(payload)
        return payload

    async def _order_book_snapshot(self, trading_pair: str) -> OrderBookMessage:
        payload = await self._request_quote(trading_pair)
        return BinanceStocksOrderBook.snapshot_message_from_exchange(
            payload,
            timestamp=self._time(),
            metadata={"trading_pair": trading_pair},
        )

    async def _connected_websocket_assistant(self) -> WSAssistant:
        streams = ["calendar"]
        for trading_pair in self._trading_pairs:
            symbol = await self._connector.exchange_symbol_associated_to_pair(trading_pair=trading_pair)
            streams.extend([f"{symbol}@quote", f"{symbol}@tradingStatus", f"{symbol}@tradability"])
        stream_path = "/".join(quote(stream, safe="@_") for stream in streams)
        websocket = await self._api_factory.get_ws_assistant()
        await websocket.connect(
            ws_url=f"{CONSTANTS.WS_URL}/stream?streams={stream_path}",
            ping_timeout=CONSTANTS.WS_HEARTBEAT_SECONDS,
        )
        return websocket

    async def _subscribe_channels(self, ws: WSAssistant):
        # Stocks streams are selected in the connection URL; no SUBSCRIBE RPC exists.
        self.logger().info("Connected to Binance Stocks BBO and market-state streams.")

    def _channel_originating_message(self, event_message: Dict[str, Any]) -> str:
        payload = event_message.get("data", event_message)
        return self._snapshot_messages_queue_key if payload.get("e") == "quote" else ""

    async def _process_websocket_messages(self, websocket_assistant: WSAssistant):
        async for response in websocket_assistant.iter_messages():
            wrapper = response.data
            if not isinstance(wrapper, dict):
                continue
            payload = wrapper.get("data", wrapper)
            event_type = payload.get("e")
            if event_type == "quote":
                self._connector.process_quote_event(payload)
                self._message_queue[self._snapshot_messages_queue_key].put_nowait(payload)
            elif event_type in {"calendar", "tradingStatus", "tradability"}:
                self._connector.process_market_state_event(payload)

    async def _parse_order_book_snapshot_message(self, raw_message: Dict[str, Any], message_queue: asyncio.Queue):
        symbol = raw_message.get("s", raw_message.get("symbol"))
        trading_pair = await self._connector.trading_pair_associated_to_exchange_symbol(symbol=symbol)
        message_queue.put_nowait(
            BinanceStocksOrderBook.snapshot_message_from_exchange(
                raw_message,
                timestamp=self._time(),
                metadata={"trading_pair": trading_pair},
            )
        )

    async def _parse_trade_message(self, raw_message: Dict[str, Any], message_queue: asyncio.Queue):
        return

    async def _parse_order_book_diff_message(self, raw_message: Dict[str, Any], message_queue: asyncio.Queue):
        return

    async def subscribe_to_trading_pair(self, trading_pair: str) -> bool:
        if trading_pair in self._trading_pairs:
            return True
        self.add_trading_pair(trading_pair)
        if self._ws_assistant is not None:
            await self._ws_assistant.disconnect()
        return True

    async def unsubscribe_from_trading_pair(self, trading_pair: str) -> bool:
        if trading_pair not in self._trading_pairs:
            return True
        self.remove_trading_pair(trading_pair)
        if self._ws_assistant is not None:
            await self._ws_assistant.disconnect()
        return True

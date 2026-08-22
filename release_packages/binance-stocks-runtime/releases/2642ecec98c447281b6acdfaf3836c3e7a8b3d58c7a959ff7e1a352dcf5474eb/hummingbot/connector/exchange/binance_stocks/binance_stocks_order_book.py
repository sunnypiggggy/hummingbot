from typing import Any, Dict, Optional

from hummingbot.core.data_type.order_book import OrderBook
from hummingbot.core.data_type.order_book_message import OrderBookMessage, OrderBookMessageType


class BinanceStocksOrderBook(OrderBook):
    @classmethod
    def snapshot_message_from_exchange(
        cls,
        msg: Dict[str, Any],
        timestamp: float,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> OrderBookMessage:
        payload = dict(msg)
        if metadata:
            payload.update(metadata)
        event_ms = int(payload.get("T", payload.get("E", timestamp * 1e3)))
        bid_price = payload.get("bp", payload.get("bidPrice"))
        bid_size = payload.get("bs", payload.get("bidQty", "0"))
        ask_price = payload.get("ap", payload.get("askPrice"))
        ask_size = payload.get("as", payload.get("askQty", "0"))
        bids = [[bid_price, bid_size]] if bid_price is not None else []
        asks = [[ask_price, ask_size]] if ask_price is not None else []
        return OrderBookMessage(
            OrderBookMessageType.SNAPSHOT,
            {
                "trading_pair": payload["trading_pair"],
                "update_id": event_ms,
                "bids": bids,
                "asks": asks,
            },
            timestamp=event_ms * 1e-3,
        )

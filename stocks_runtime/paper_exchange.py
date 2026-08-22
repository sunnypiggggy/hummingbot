from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, Optional, Tuple

from hummingbot.connector.exchange.binance_stocks import binance_stocks_constants as CONSTANTS
from hummingbot.connector.exchange.binance_stocks.binance_stocks_exchange import BinanceStocksExchange
from hummingbot.connector.exchange.binance_stocks.binance_stocks_position_provider import EquityAccountSnapshot
from hummingbot.core.data_type.common import OrderType, TradeType
from hummingbot.core.data_type.in_flight_order import InFlightOrder, OrderUpdate

from stocks_runtime.paper_broker import EXTENDED_PHASES, RTH_PHASES, PostgresPaperBroker


class BinanceStocksPaperExchange(BinanceStocksExchange):
    """Binance market data plus a local persistent execution venue.

    This class intentionally overrides every economic request hook. It cannot
    place or cancel an order at Binance, even if a trading-enabled key is
    accidentally mounted into a PAPER container.
    """

    def __init__(self, *args, paper_broker: PostgresPaperBroker, **kwargs):
        kwargs["trading_required"] = False
        kwargs["disclaimer_confirmed"] = True
        super().__init__(*args, **kwargs)
        self.paper_broker = paper_broker
        self.economic_http_request_count = 0
        self._paper_quote_lock = asyncio.Lock()
        self._account_authorized = True
        self._position_reconciliation_ready = True

    @property
    def display_name(self) -> str:
        return "binance_stocks_PaperTrade"

    @property
    def status_dict(self) -> Dict[str, bool]:
        status = super().status_dict
        status.update({
            "paper_broker_ready": bool(self.paper_broker.run_id),
            "economic_http_requests_disabled": self.economic_http_request_count == 0,
        })
        return status

    def process_quote_event(self, event: Dict[str, Any]):
        super().process_quote_event(event)
        try:
            asyncio.get_running_loop().create_task(self._process_paper_quote(dict(event)))
        except RuntimeError:
            pass

    def process_market_state_event(self, event: Dict[str, Any]):
        super().process_market_state_event(event)
        trading_date = event.get("tradingDate", event.get("trading_date"))
        self.paper_broker.update_market_state(
            self.market_phase, self._trading_status, self._tradability, trading_date
        )

    async def _process_paper_quote(self, event: Dict[str, Any]):
        async with self._paper_quote_lock:
            self.paper_broker.update_market_state(
                self.market_phase, self._trading_status, self._tradability,
                self.paper_broker.trading_date,
            )
            changed = await self.paper_broker.process_quote(event)
            if changed and self.in_flight_orders:
                await self._update_order_status()
            await self._update_balances()

    async def _api_post(self, *args, **kwargs):
        self.economic_http_request_count += 1
        raise PermissionError("PAPER connector has no Binance economic HTTP path")

    async def _place_order(
        self,
        order_id: str,
        trading_pair: str,
        amount: Decimal,
        trade_type: TradeType,
        order_type: OrderType,
        price: Decimal,
        **kwargs,
    ) -> Tuple[str, float]:
        symbol = await self.exchange_symbol_associated_to_pair(trading_pair=trading_pair)
        if order_type not in {OrderType.LIMIT, OrderType.MARKET}:
            raise ValueError("PAPER Stocks supports LIMIT and MARKET only")
        quote = self.paper_broker.latest_quote(symbol)
        if quote is None:
            raise PermissionError(f"paper quote for {symbol} is stale")
        phase = self.market_phase
        if order_type is OrderType.MARKET and phase not in RTH_PHASES:
            raise PermissionError("paper MARKET orders are allowed only during MARKET_OPEN")
        if order_type is OrderType.LIMIT and phase not in EXTENDED_PHASES:
            raise PermissionError("paper LIMIT order requested outside EXTENDED session")
        if not self.paper_broker._direction_allowed(symbol, trade_type.name):
            raise PermissionError(f"paper market state blocks {trade_type.name} for {symbol}")
        if not self._fractional_supported.get(symbol, False) and amount != amount.to_integral_value():
            raise ValueError(f"{symbol} does not support fractional quantity")
        executor_id = self._runtime_executor_id()
        trading_date = self.paper_broker.trading_date or datetime.now(timezone.utc).date().isoformat()
        row = await self.paper_broker.create_order(
            client_order_id=order_id,
            executor_id=executor_id,
            symbol=symbol,
            side=trade_type.name,
            order_type=order_type.name,
            amount=amount,
            limit_price=None if order_type is OrderType.MARKET else price,
            trading_date=trading_date,
        )
        try:
            if hasattr(self._position_provider, "register_order"):
                await self._position_provider.register_order(
                    client_order_id=order_id,
                    executor_id=executor_id,
                    symbol=symbol,
                    side=trade_type.name,
                    requested_base=amount,
                    order_type=order_type.name,
                )
            if hasattr(self._position_provider, "bind_exchange_order"):
                await self._position_provider.bind_exchange_order(order_id, str(row["exchange_order_id"]))
        except Exception:
            await self.paper_broker.cancel_order(order_id, "REJECTED")
            raise
        return str(row["exchange_order_id"]), row["accepted_at"].timestamp()

    async def _place_cancel(self, order_id: str, tracked_order: InFlightOrder):
        return await self.paper_broker.cancel_order(order_id)

    async def _request_order_detail(
        self,
        tracked_order: Optional[InFlightOrder] = None,
        client_order_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        order_id = client_order_id or (tracked_order.client_order_id if tracked_order else None)
        if not order_id:
            raise ValueError("clientOrderId is required")
        row = await self.paper_broker.order(order_id)
        if row is None:
            raise IOError(f"paper order not found: {order_id}")
        average = Decimal(row["filled_quote"]) / Decimal(row["filled_base"]) if Decimal(row["filled_base"]) else 0
        return {
            "clientOrderId": row["client_order_id"],
            "orderId": row["exchange_order_id"],
            "orderStatus": row["status"],
            "filledQuantity": str(row["filled_base"]),
            "filledAmount": str(row["filled_quote"]),
            "avgFilledPrice": str(average),
            "fee": str(row["cumulative_fee"]),
            "updateTime": row["updated_at"].timestamp(),
        }

    async def _request_order_status(self, tracked_order: InFlightOrder) -> OrderUpdate:
        detail = await self._request_order_detail(tracked_order=tracked_order)
        status = str(detail["orderStatus"]).upper()
        if status not in CONSTANTS.ORDER_STATE:
            raise IOError(f"unknown paper order state {status}")
        return OrderUpdate(
            trading_pair=tracked_order.trading_pair,
            update_timestamp=float(detail["updateTime"]),
            new_state=CONSTANTS.ORDER_STATE[status],
            client_order_id=tracked_order.client_order_id,
            exchange_order_id=str(detail["orderId"]),
        )

    async def _update_balances(self):
        account = await self.paper_broker.account()
        positions = await self.paper_broker.ledger.managed_positions()
        self._account_balances.clear()
        self._account_available_balances.clear()
        self._account_balances[self._quote_asset] = Decimal(account["cash_balance"])
        self._account_available_balances[self._quote_asset] = Decimal(account["available_cash"])
        for symbol, position in positions.items():
            self._account_balances[symbol] = position.total
            self._account_available_balances[symbol] = position.available
        self._last_position_snapshot = EquityAccountSnapshot(
            quote_total=Decimal(account["cash_balance"]),
            quote_available=Decimal(account["available_cash"]),
            positions=positions,
            source="paper_ledger",
            timestamp=self._clock_time(),
        )
        self._position_reconciliation_ready = True

    async def _paper_idle(self):
        while True:
            await asyncio.sleep(3600)

    def _create_user_stream_tracker_task(self):
        return asyncio.create_task(self._paper_idle())

    async def _user_stream_event_listener(self):
        await self._paper_idle()

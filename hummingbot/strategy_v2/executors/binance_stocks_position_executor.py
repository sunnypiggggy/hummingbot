from __future__ import annotations

from dataclasses import dataclass
from decimal import ROUND_DOWN, Decimal
from typing import Optional

from hummingbot.core.data_type.common import OrderType, PositionAction, TradeType
from hummingbot.core.event.events import OrderCancelledEvent, OrderFilledEvent
from hummingbot.strategy_v2.executors.position_executor.position_executor import PositionExecutor
from hummingbot.strategy_v2.models.base import RunnableStatus
from hummingbot.strategy_v2.models.executors import CloseType, TrackedOrder


@dataclass(frozen=True)
class StocksExitInstruction:
    order_type: Optional[OrderType]
    price: Decimal = Decimal("NaN")
    phase: str = "EXIT_PENDING"


class BinanceStocksExitPolicy:
    """Translate a risk-exit trigger into a valid Binance Stocks order."""

    @staticmethod
    def instruction(market_phase: str, best_bid: Decimal, tick_size: Decimal) -> StocksExitInstruction:
        phase = market_phase.upper()
        if phase == "MARKET_OPEN":
            return StocksExitInstruction(order_type=OrderType.MARKET, phase="RTH_MARKET")
        if phase in {"PRE_MARKET", "POST_MARKET"}:
            if best_bid <= 0 or tick_size <= 0:
                return StocksExitInstruction(order_type=None)
            price = ((best_bid - tick_size) / tick_size).to_integral_value(rounding=ROUND_DOWN) * tick_size
            return StocksExitInstruction(order_type=OrderType.LIMIT, price=max(tick_size, price), phase="EXTENDED_LIMIT")
        return StocksExitInstruction(order_type=None)


class BinanceStocksPositionExecutor(PositionExecutor):
    """Long-only PositionExecutor with session-aware protective exits.

    The upstream executor hard-codes MARKET for stop/time exits.  Binance
    Stocks permits MARKET only in RTH, so this subclass uses a marketable DAY
    limit in PRE/POST and keeps an explicit EXIT_PENDING state while closed.
    """

    EXIT_REPRICE_SECONDS = 5.0

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.config.connector_name != "binance_stocks":
            raise ValueError("BinanceStocksPositionExecutor requires binance_stocks")
        if self.config.side is not TradeType.BUY:
            raise ValueError("Binance Stocks PositionExecutor is long-only")
        barriers = self.config.triple_barrier_config
        if barriers.stop_loss is None or barriers.stop_loss <= 0:
            raise ValueError("Binance Stocks live positions require stop_loss")
        if barriers.time_limit is None or barriers.time_limit <= 0:
            raise ValueError("Binance Stocks live positions require time_limit")
        self._first_fill_timestamp: Optional[float] = None
        self._pending_close_type: Optional[CloseType] = None
        self._close_order_extended = False
        self._close_order_submitted_at = 0.0
        self._entry_filled_backup = Decimal("0")
        self._entry_quote_backup = Decimal("0")
        self._exit_filled_backup = Decimal("0")
        self._exit_quote_backup = Decimal("0")
        self._fees_quote_backup = Decimal("0")
        self._external_entry_frozen = False

    @property
    def end_time(self) -> Optional[float]:
        if self._first_fill_timestamp is None:
            return None
        return self._first_fill_timestamp + self.config.triple_barrier_config.time_limit

    @property
    def open_filled_amount(self) -> Decimal:
        current = Decimal("0")
        if self._open_order and self._open_order.order:
            current = self._open_order.order.executed_amount_base
        return self.connectors[self.config.connector_name].quantize_order_amount(
            self.config.trading_pair, self._entry_filled_backup + current
        )

    @property
    def entry_price(self) -> Decimal:
        current_base = current_quote = Decimal("0")
        if self._open_order and self._open_order.order:
            current_base = self._open_order.order.executed_amount_base
            current_quote = self._open_order.order.executed_amount_quote
        total_base = self._entry_filled_backup + current_base
        total_quote = self._entry_quote_backup + current_quote
        return total_quote / total_base if total_base > 0 and total_quote > 0 else self.config.entry_price

    @property
    def close_filled_amount(self) -> Decimal:
        current = Decimal("0")
        if self._close_order and self._close_order.order:
            current = self._close_order.order.executed_amount_base
        return self._exit_filled_backup + current

    @property
    def amount_to_close(self) -> Decimal:
        return max(Decimal("0"), self.open_filled_amount - self.close_filled_amount)

    def get_cum_fees_quote(self) -> Decimal:
        return self._fees_quote_backup + super().get_cum_fees_quote()

    def open_and_close_volume_match(self):
        return self.open_filled_amount == 0 or self.amount_to_close < self.trading_rules.min_order_size

    def control_barriers(self):
        # Protect every confirmed partial fill; waiting for the full DAY entry
        # order to fill would leave fractional inventory unprotected.
        if self.open_filled_amount >= self.trading_rules.min_order_size:
            self.control_stop_loss()
            if self.status != RunnableStatus.RUNNING:
                return
            self.control_trailing_stop()
            if self.status != RunnableStatus.RUNNING:
                return
            self.control_take_profit()
            if self.status != RunnableStatus.RUNNING:
                return
        self.control_time_limit()

    def control_open_order(self):
        if not self._external_entry_frozen:
            super().control_open_order()

    def freeze_entry_due_external_activity(self):
        """Cancel only unfilled entry risk; existing protective exits remain active."""
        self._external_entry_frozen = True
        if self._open_order and self._open_order.order and self._open_order.order.is_open:
            self._strategy.cancel(
                connector_name=self.config.connector_name,
                trading_pair=self.config.trading_pair,
                order_id=self._open_order.order_id,
            )

    def process_order_filled_event(self, _, market, event: OrderFilledEvent):
        super().process_order_filled_event(_, market, event)
        if self._open_order and event.order_id == self._open_order.order_id and self._first_fill_timestamp is None:
            self._first_fill_timestamp = self._strategy.current_timestamp

    def process_order_canceled_event(self, _, market, event: OrderCancelledEvent):
        if self._open_order and event.order_id == self._open_order.order_id and self._open_order.order:
            filled = self._open_order.order.executed_amount_base
            if filled > 0:
                self._entry_filled_backup += filled
                self._entry_quote_backup += self._open_order.order.executed_amount_quote
                self._fees_quote_backup += self._open_order.cum_fees_quote
        if self._close_order and event.order_id == self._close_order.order_id and self._close_order.order:
            filled = self._close_order.order.executed_amount_base
            if filled > 0:
                self._exit_filled_backup += filled
                self._exit_quote_backup += self._close_order.order.executed_amount_quote
                self._fees_quote_backup += self._close_order.cum_fees_quote
        elif (
            self._take_profit_limit_order
            and event.order_id == self._take_profit_limit_order.order_id
            and self._take_profit_limit_order.order
        ):
            filled = self._take_profit_limit_order.order.executed_amount_base
            if filled > 0:
                self._exit_filled_backup += filled
                self._exit_quote_backup += self._take_profit_limit_order.order.executed_amount_quote
                self._fees_quote_backup += self._take_profit_limit_order.cum_fees_quote
        super().process_order_canceled_event(_, market, event)

    def place_close_order_and_cancel_open_orders(self, close_type: CloseType, price: Decimal = Decimal("NaN")):
        self.cancel_open_orders()
        self._pending_close_type = close_type
        self.close_type = close_type
        self.close_timestamp = self._strategy.current_timestamp
        self._status = RunnableStatus.SHUTTING_DOWN

    def _submit_session_appropriate_close(self):
        if self._close_order is not None or self.amount_to_close < self.trading_rules.min_order_size:
            return
        # Do not race a risk exit against a still-open entry or take-profit
        # order. Their cancel events carry the final partial fills into the
        # backup counters before the replacement exit quantity is calculated.
        if self._open_order and self._open_order.order and self._open_order.order.is_open:
            return
        if (
            self._take_profit_limit_order
            and self._take_profit_limit_order.order
            and self._take_profit_limit_order.order.is_open
        ):
            return
        connector = self.connectors[self.config.connector_name]
        quote = connector.latest_quote(self.config.trading_pair.split("-")[0])
        best_bid = quote[0] if quote else Decimal("0")
        instruction = BinanceStocksExitPolicy.instruction(
            connector.market_phase,
            best_bid,
            self.trading_rules.min_price_increment,
        )
        if instruction.order_type is None:
            return
        order_id = self.place_order(
            connector_name=self.config.connector_name,
            trading_pair=self.config.trading_pair,
            order_type=instruction.order_type,
            amount=self.amount_to_close,
            price=instruction.price,
            side=TradeType.SELL,
            position_action=PositionAction.CLOSE,
        )
        self._close_order = TrackedOrder(order_id=order_id)
        self._close_order_extended = instruction.order_type is OrderType.LIMIT
        self._close_order_submitted_at = self._strategy.current_timestamp

    async def control_close_order(self):
        connector = self.connectors[self.config.connector_name]
        open_orders = [self._open_order, self._take_profit_limit_order]
        if any(order and order.order and order.order.is_open for order in open_orders):
            self.cancel_open_orders()
            return
        if self._close_order and self._close_order.order and self._close_order.order.is_open:
            age = self._strategy.current_timestamp - self._close_order_submitted_at
            must_switch_to_market = self._close_order_extended and connector.market_phase == "MARKET_OPEN"
            must_reprice = self._close_order_extended and age >= self.EXIT_REPRICE_SECONDS
            if must_switch_to_market or must_reprice:
                self._strategy.cancel(
                    connector_name=self.config.connector_name,
                    trading_pair=self.config.trading_pair,
                    order_id=self._close_order.order_id,
                )
                return
        if self._close_order is None:
            self._submit_session_appropriate_close()
            return
        await super().control_close_order()

    def get_custom_info(self):
        info = super().get_custom_info()
        connector = self.connectors[self.config.connector_name]
        info.update(
            {
                "first_fill_timestamp": self._first_fill_timestamp,
                "exit_phase": "EXIT_PENDING" if self._close_order is None and self._pending_close_type else (
                    "EXTENDED_LIMIT" if self._close_order_extended else "RTH_MARKET"
                ),
                "market_phase": connector.market_phase,
                "external_positions_unknown": True,
                "external_entry_frozen": self._external_entry_frozen,
            }
        )
        return info

"""API-managed execution half of the walk-forward portfolio grid.

The scheduler selects parameters and deploys a versioned YAML configuration.
This script deliberately only executes that immutable configuration.
"""

import json
import logging
import os
import tempfile
from dataclasses import asdict, dataclass
from decimal import Decimal
from pathlib import Path
from typing import Dict, List

from pydantic import Field, field_validator

from hummingbot.connector.connector_base import ConnectorBase
from hummingbot.core.data_type.common import MarketDict, PriceType
from hummingbot.core.event.events import OrderFilledEvent, OrderType
from hummingbot.strategy.strategy_v2_base import StrategyV2Base, StrategyV2ConfigBase


@dataclass
class GridState:
    lower: Decimal
    upper: Decimal
    levels: List[Decimal]
    moves: int = 0
    last_move_ts: float = 0


class WalkForwardPortfolioGridConfig(StrategyV2ConfigBase):
    script_file_name: str = os.path.basename(__file__)
    controllers_config: List[str] = []
    parameter_version: str = Field("manual")
    exchange: str = Field("binance_paper_trade")
    trading_pairs: List[str] = Field(default=["BTC-USDT", "ETH-USDT"])
    quote_asset: str = Field("USDT")
    grid_range: Decimal = Field(Decimal("0.04"))
    grid_levels: int = Field(24)
    order_quote_pct: Decimal = Field(Decimal("0.02"))
    take_profit: Decimal = Field(Decimal("0.003"))
    move_threshold: Decimal = Field(Decimal("0.005"))
    portfolio_stop_loss: Decimal = Field(Decimal("0.08"))
    order_refresh_time: int = Field(60)
    min_grid_move_seconds: int = Field(0)
    cooldown_seconds: int = Field(86400)
    min_order_quote: Decimal = Field(Decimal("10"))
    initial_peak_equity: Decimal = Field(Decimal("0"))
    initial_cooldown_until: float = Field(0)
    initial_grid_states: Dict[str, Dict] = Field(default_factory=dict)

    @field_validator("trading_pairs", mode="before")
    @classmethod
    def parse_trading_pairs(cls, value):
        if isinstance(value, str):
            return [pair.strip().upper() for pair in value.split(",") if pair.strip()]
        return value

    def update_markets(self, markets: MarketDict) -> MarketDict:
        markets[self.exchange] = markets.get(self.exchange, set()) | set(self.trading_pairs)
        return markets


class WalkForwardPortfolioGrid(StrategyV2Base):
    create_timestamp = 0

    def __init__(self, connectors: Dict[str, ConnectorBase], config: WalkForwardPortfolioGridConfig):
        super().__init__(connectors, config)
        self.config = config
        self.grid_states = self._restore_grid_states(config.initial_grid_states)
        self.peak_equity = config.initial_peak_equity
        self.cooldown_until = config.initial_cooldown_until
        self.liquidated = self.cooldown_until > 0

    @property
    def connector(self) -> ConnectorBase:
        return self.connectors[self.config.exchange]

    def on_tick(self):
        if self.current_timestamp < self.create_timestamp:
            return
        equity = self.portfolio_equity()
        if equity <= 0:
            self._schedule_next_tick()
            return
        self.peak_equity = max(self.peak_equity, equity)
        drawdown = (equity - self.peak_equity) / self.peak_equity if self.peak_equity else Decimal("0")
        if drawdown <= -self.config.portfolio_stop_loss:
            self.liquidate_portfolio(equity, drawdown)
            self._persist_state(equity)
            self._schedule_next_tick()
            return
        if self.current_timestamp >= self.cooldown_until:
            self.liquidated = False
            self.cancel_all_orders()
            for pair in self.config.trading_pairs:
                self.refresh_pair_grid(pair, equity)
        self._persist_state(equity)
        self._schedule_next_tick()

    def _schedule_next_tick(self):
        self.create_timestamp = self.current_timestamp + self.config.order_refresh_time

    def refresh_pair_grid(self, pair: str, equity: Decimal):
        price = self.reference_price(pair)
        if price <= 0:
            return
        state = self.grid_states.get(pair)
        if state is None:
            state = self.new_grid_state(price)
            self.grid_states[pair] = state
        elif price > state.upper * (Decimal("1") + self.config.move_threshold) or price < state.lower * (Decimal("1") - self.config.move_threshold):
            if self.current_timestamp - state.last_move_ts >= self.config.min_grid_move_seconds:
                state = self.new_grid_state(price, state.moves + 1)
                self.grid_states[pair] = state
                self.notify(f"Moved grid for {pair}: center={price:.8f}, moves={state.moves}")
        allocation = equity / Decimal(len(self.config.trading_pairs))
        order_quote = max(allocation * self.config.order_quote_pct, self.config.min_order_quote)
        quote_available = self.connector.get_available_balance(self.config.quote_asset)
        for level in (level for level in state.levels if level < price):
            if quote_available < order_quote:
                break
            self.buy(self.config.exchange, pair, order_quote / level, OrderType.LIMIT, level)
            quote_available -= order_quote
        base_available = self.connector.get_available_balance(pair.split("-")[0])
        sell_levels = [level for level in state.levels if level > price]
        if base_available > 0 and sell_levels:
            amount = base_available / Decimal(len(sell_levels))
            for level in sell_levels:
                sell_price = max(level, price * (Decimal("1") + self.config.take_profit))
                if amount * sell_price >= self.config.min_order_quote:
                    self.sell(self.config.exchange, pair, amount, OrderType.LIMIT, sell_price)

    def new_grid_state(self, center: Decimal, moves: int = 0) -> GridState:
        lower = center * (Decimal("1") - self.config.grid_range / Decimal("2"))
        upper = center * (Decimal("1") + self.config.grid_range / Decimal("2"))
        step = (upper - lower) / Decimal(max(self.config.grid_levels - 1, 1))
        levels = [lower + step * Decimal(index) for index in range(self.config.grid_levels)]
        return GridState(lower=lower, upper=upper, levels=levels, moves=moves, last_move_ts=self.current_timestamp)

    def reference_price(self, pair: str) -> Decimal:
        price = self.connector.get_price_by_type(pair, PriceType.LastTrade)
        if price is None or price.is_nan() or price <= 0:
            price = self.connector.get_price_by_type(pair, PriceType.MidPrice)
        return price if price is not None and not price.is_nan() else Decimal("0")

    def portfolio_equity(self) -> Decimal:
        equity = self.connector.get_balance(self.config.quote_asset)
        for pair in self.config.trading_pairs:
            equity += self.connector.get_balance(pair.split("-")[0]) * self.reference_price(pair)
        return equity

    def liquidate_portfolio(self, equity: Decimal, drawdown: Decimal):
        self.cancel_all_orders()
        for pair in self.config.trading_pairs:
            amount = self.connector.get_available_balance(pair.split("-")[0])
            if amount > 0:
                self.sell(self.config.exchange, pair, amount, OrderType.MARKET)
        self.cooldown_until = self.current_timestamp + self.config.cooldown_seconds
        self.liquidated = True
        self.peak_equity = equity
        self.notify(f"Portfolio liquidation: equity={equity:.2f}, drawdown={drawdown:.2%}")

    def cancel_all_orders(self):
        for order in self.get_active_orders(self.config.exchange):
            self.cancel(self.config.exchange, order.trading_pair, order.client_order_id)

    def _restore_grid_states(self, values: Dict[str, Dict]) -> Dict[str, GridState]:
        states = {}
        for pair, raw in values.items():
            try:
                states[pair] = GridState(
                    lower=Decimal(str(raw["lower"])), upper=Decimal(str(raw["upper"])),
                    levels=[Decimal(str(value)) for value in raw["levels"]], moves=int(raw.get("moves", 0)),
                    last_move_ts=float(raw.get("last_move_ts", 0)),
                )
            except (KeyError, ValueError, TypeError):
                self.logger().warning("Ignoring invalid persisted grid state for %s", pair)
        return states

    def _persist_state(self, equity: Decimal):
        payload = {
            "parameter_version": self.config.parameter_version,
            "updated_at": self.current_timestamp,
            "current_equity": str(equity), "peak_equity": str(self.peak_equity),
            "cooldown_until": self.cooldown_until, "liquidated": self.liquidated,
            "grid_states": {pair: {**asdict(state), "lower": str(state.lower), "upper": str(state.upper), "levels": [str(level) for level in state.levels]} for pair, state in self.grid_states.items()},
        }
        target = Path("data/runtime_state.json")
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=target.parent, delete=False) as handle:
            json.dump(payload, handle, ensure_ascii=True)
            temp_name = handle.name
        Path(temp_name).replace(target)

    def did_fill_order(self, event: OrderFilledEvent):
        self.notify(f"{event.trade_type.name} {event.amount:.8f} {event.trading_pair} at {event.price:.8f}")

    def notify(self, message: str):
        self.log_with_clock(logging.INFO, message)
        self.notify_hb_app_with_timestamp(message)

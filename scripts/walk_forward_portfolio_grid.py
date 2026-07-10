import logging
import os
from dataclasses import dataclass
from decimal import Decimal
from typing import Dict, List, Optional

import pandas as pd
from pydantic import Field, field_validator

from hummingbot.connector.connector_base import ConnectorBase
from hummingbot.core.data_type.common import MarketDict, PriceType
from hummingbot.core.event.events import OrderFilledEvent, OrderType
from hummingbot.data_feed.candles_feed.data_types import CandlesConfig
from hummingbot.strategy.strategy_v2_base import StrategyV2Base, StrategyV2ConfigBase
from scripts.backtest_3commas_grid import GridParams
from scripts.backtest_walk_forward_portfolio_grid import (
    default_walk_forward_search_space,
    optimize_params_parallel,
)


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
    exchange: str = Field("binance")
    trading_pairs: List[str] = Field(default=[
        "BTC-USDT", "ETH-USDT", "SOL-USDT", "BNB-USDT", "XRP-USDT",
        "DOGE-USDT", "ADA-USDT", "LINK-USDT", "AVAX-USDT", "TRX-USDT",
    ])
    quote_asset: str = Field("USDT")
    candles_connector: str = Field("binance")
    candles_interval: str = Field("5m")
    lookback_days: Decimal = Field(Decimal("7"))
    initial_training_days: Decimal = Field(Decimal("30"))
    reoptimize_seconds: int = Field(604800)
    max_candidates: int = Field(576)
    optimization_workers: int = Field(4)
    drawdown_penalty: Decimal = Field(Decimal("1.5"))
    liquidation_penalty: Decimal = Field(Decimal("0.25"))
    fee_rate: Decimal = Field(Decimal("0.0002"))
    grid_range: Decimal = Field(Decimal("0.04"))
    grid_levels: int = Field(24)
    order_quote_pct: Decimal = Field(Decimal("0.02"))
    take_profit: Decimal = Field(Decimal("0.003"))
    move_threshold: Decimal = Field(Decimal("0.005"))
    min_candidate_move_threshold: Decimal = Field(Decimal("0.005"))
    stop_loss: Decimal = Field(Decimal("0.04"))
    portfolio_stop_loss: Decimal = Field(Decimal("0.08"))
    order_refresh_time: int = Field(60)
    min_grid_move_seconds: int = Field(0)
    cooldown_seconds: int = Field(86400)
    min_order_quote: Decimal = Field(Decimal("10"))

    @field_validator("trading_pairs", mode="before")
    @classmethod
    def parse_trading_pairs(cls, value):
        if isinstance(value, str):
            return [pair.strip().upper() for pair in value.split(",") if pair.strip()]
        return value

    def update_markets(self, markets: MarketDict) -> MarketDict:
        markets[self.exchange] = markets.get(self.exchange, set()) | set(self.trading_pairs)
        return markets

    def candles_records(self) -> int:
        interval_seconds = {
            "1m": 60,
            "3m": 180,
            "5m": 300,
            "15m": 900,
            "30m": 1800,
            "1h": 3600,
        }.get(self.candles_interval, 300)
        days = max(self.initial_training_days, self.lookback_days)
        return int(days * Decimal(86400) / Decimal(interval_seconds)) + 100


class WalkForwardPortfolioGrid(StrategyV2Base):
    create_timestamp = 0

    def __init__(self, connectors: Dict[str, ConnectorBase], config: WalkForwardPortfolioGridConfig):
        super().__init__(connectors, config)
        self.config = config
        self.grid_states: Dict[str, GridState] = {}
        self.peak_equity = Decimal("0")
        self.cooldown_until = 0
        self.liquidated = False
        self.current_params = self.config_params()
        self.next_reoptimization_ts = 0
        self.optimization_ready_notified = False
        self.candidates = [
            candidate for candidate in default_walk_forward_search_space()
            if Decimal(str(candidate.move_threshold)) >= self.config.min_candidate_move_threshold
        ][:self.config.max_candidates]
        self.initialize_candles_feeds()

    @property
    def connector(self) -> ConnectorBase:
        return self.connectors[self.config.exchange]

    def on_tick(self):
        if self.current_timestamp < self.create_timestamp:
            return

        if not self.candles_ready():
            if not self.optimization_ready_notified:
                self.notify("Waiting for candle feeds before starting walk-forward grid.")
                self.optimization_ready_notified = True
            self.create_timestamp = self.current_timestamp + self.config.order_refresh_time
            return

        if self.current_timestamp >= self.next_reoptimization_ts:
            self.reoptimize_params()

        equity = self.portfolio_equity()
        if equity <= 0:
            self.create_timestamp = self.current_timestamp + self.config.order_refresh_time
            return

        self.peak_equity = max(self.peak_equity, equity)
        drawdown = (equity - self.peak_equity) / self.peak_equity if self.peak_equity > 0 else Decimal("0")
        if drawdown <= -self.config.portfolio_stop_loss:
            self.liquidate_portfolio(equity, drawdown)
            self.create_timestamp = self.current_timestamp + self.config.order_refresh_time
            return

        if self.current_timestamp < self.cooldown_until:
            self.create_timestamp = self.current_timestamp + self.config.order_refresh_time
            return

        self.cancel_all_orders()
        for trading_pair in self.config.trading_pairs:
            self.refresh_pair_grid(trading_pair, equity)

        self.create_timestamp = self.current_timestamp + self.config.order_refresh_time

    def refresh_pair_grid(self, trading_pair: str, equity: Decimal):
        price = self.reference_price(trading_pair)
        if price <= 0:
            return

        state = self.grid_states.get(trading_pair)
        if state is None:
            state = self.new_grid_state(price)
            self.grid_states[trading_pair] = state
        elif price > state.upper * (Decimal("1") + self.current_params.move_threshold) or \
                price < state.lower * (Decimal("1") - self.current_params.move_threshold):
            if self.current_timestamp - state.last_move_ts >= self.config.min_grid_move_seconds:
                state = self.new_grid_state(price, state.moves + 1)
                self.grid_states[trading_pair] = state
                self.notify(f"Moved grid for {trading_pair}: center={price:.8f}, moves={state.moves}")

        allocation = equity / Decimal(len(self.config.trading_pairs))
        order_quote = max(allocation * self.current_params.order_quote_pct, self.config.min_order_quote)
        available_quote = self.connector.get_available_balance(self.config.quote_asset)

        buy_levels = [level for level in state.levels if level < price]
        for level in buy_levels:
            if available_quote < order_quote:
                break
            amount = order_quote / level
            self.buy(self.config.exchange, trading_pair, amount, OrderType.LIMIT, level)
            available_quote -= order_quote

        base_asset = trading_pair.split("-")[0]
        available_base = self.connector.get_available_balance(base_asset)
        sell_levels = [level for level in state.levels if level > price]
        if available_base > 0 and sell_levels:
            sell_amount = available_base / Decimal(len(sell_levels))
            for level in sell_levels:
                tp_price = max(level, price * (Decimal("1") + self.current_params.take_profit))
                if sell_amount * tp_price >= self.config.min_order_quote:
                    self.sell(self.config.exchange, trading_pair, sell_amount, OrderType.LIMIT, tp_price)

    def new_grid_state(self, center_price: Decimal, moves: int = 0) -> GridState:
        lower = center_price * (Decimal("1") - self.current_params.grid_range / Decimal("2"))
        upper = center_price * (Decimal("1") + self.current_params.grid_range / Decimal("2"))
        if self.current_params.grid_levels <= 1:
            levels = [center_price]
        else:
            step = (upper - lower) / Decimal(self.current_params.grid_levels - 1)
            levels = [lower + step * Decimal(idx) for idx in range(self.current_params.grid_levels)]
        return GridState(
            lower=lower,
            upper=upper,
            levels=levels,
            moves=moves,
            last_move_ts=self.current_timestamp,
        )

    def initialize_candles_feeds(self):
        for trading_pair in self.config.trading_pairs:
            self.market_data_provider.initialize_candles_feed(CandlesConfig(
                connector=self.config.candles_connector,
                trading_pair=trading_pair,
                interval=self.config.candles_interval,
                max_records=self.config.candles_records(),
            ))
        self.logger().info(
            f"Initialized {len(self.config.trading_pairs)} candle feeds for walk-forward optimization."
        )

    def candles_ready(self) -> bool:
        for trading_pair in self.config.trading_pairs:
            candle_config = self.candle_config(trading_pair)
            feed = self.market_data_provider.get_candles_feed(candle_config)
            if not feed.ready or feed.candles_df.empty:
                return False
        return True

    def candle_config(self, trading_pair: str) -> CandlesConfig:
        return CandlesConfig(
            connector=self.config.candles_connector,
            trading_pair=trading_pair,
            interval=self.config.candles_interval,
            max_records=self.config.candles_records(),
        )

    def get_training_candles(self) -> Optional[Dict[str, pd.DataFrame]]:
        candles_by_pair = {}
        cutoff = self.current_timestamp - float(self.config.lookback_days) * 86400
        if self.next_reoptimization_ts == 0:
            cutoff = self.current_timestamp - float(self.config.initial_training_days) * 86400
        for trading_pair in self.config.trading_pairs:
            candles = self.market_data_provider.get_candles_df(
                connector_name=self.config.candles_connector,
                trading_pair=trading_pair,
                interval=self.config.candles_interval,
                max_records=self.config.candles_records(),
            ).copy()
            if candles.empty or "timestamp" not in candles.columns:
                return None
            candles["timestamp"] = candles["timestamp"].astype(float)
            candles = candles[candles["timestamp"] >= cutoff].sort_values("timestamp").reset_index(drop=True)
            if len(candles) < 20:
                return None
            candles_by_pair[trading_pair] = candles
        return candles_by_pair

    def reoptimize_params(self):
        training_candles = self.get_training_candles()
        if not training_candles:
            self.notify("Skipping walk-forward optimization: not enough candle history yet.")
            self.next_reoptimization_ts = self.current_timestamp + self.config.order_refresh_time
            return

        params, result, score, _ = optimize_params_parallel(
            candles_by_pair=training_candles,
            candidates=self.candidates,
            initial_quote=float(self.portfolio_equity() or Decimal("10000")),
            fee_rate=float(self.config.fee_rate),
            portfolio_stop_loss=float(self.config.portfolio_stop_loss),
            cooldown_hours=float(Decimal(self.config.cooldown_seconds) / Decimal(3600)),
            drawdown_penalty=float(self.config.drawdown_penalty),
            liquidation_penalty=float(self.config.liquidation_penalty),
            min_grid_move_seconds=float(self.config.min_grid_move_seconds),
            workers=int(self.config.optimization_workers),
        )
        self.apply_params(params)
        self.next_reoptimization_ts = self.current_timestamp + self.config.reoptimize_seconds
        self.notify(
            "Walk-forward params selected: "
            f"range={params.grid_range}, levels={params.grid_levels}, order_pct={params.order_quote_pct}, "
            f"tp={params.take_profit}, move={params.move_threshold}, stop={params.stop_loss}, "
            f"score={score:.6f}, train_pnl={result['net_pnl_pct']:.2%}, train_dd={result['max_drawdown_pct']:.2%}"
        )

    def apply_params(self, params: GridParams):
        self.current_params = self.decimal_params(params)
        self.config.grid_range = self.current_params.grid_range
        self.config.grid_levels = self.current_params.grid_levels
        self.config.order_quote_pct = self.current_params.order_quote_pct
        self.config.take_profit = self.current_params.take_profit
        self.config.move_threshold = self.current_params.move_threshold
        prices = {
            trading_pair: self.reference_price(trading_pair)
            for trading_pair in self.config.trading_pairs
        }
        for trading_pair, price in prices.items():
            if price > 0:
                old_moves = self.grid_states.get(trading_pair).moves if trading_pair in self.grid_states else 0
                self.grid_states[trading_pair] = self.new_grid_state(price, old_moves)

    def config_params(self) -> GridParams:
        return GridParams(
            grid_range=self.config.grid_range,
            grid_levels=self.config.grid_levels,
            order_quote_pct=self.config.order_quote_pct,
            take_profit=self.config.take_profit,
            move_threshold=self.config.move_threshold,
            stop_loss=self.config.stop_loss,
        )

    @staticmethod
    def decimal_params(params: GridParams) -> GridParams:
        return GridParams(
            grid_range=Decimal(str(params.grid_range)),
            grid_levels=int(params.grid_levels),
            order_quote_pct=Decimal(str(params.order_quote_pct)),
            take_profit=Decimal(str(params.take_profit)),
            move_threshold=Decimal(str(params.move_threshold)),
            stop_loss=Decimal(str(params.stop_loss)),
        )

    def reference_price(self, trading_pair: str) -> Decimal:
        price = self.connector.get_price_by_type(trading_pair, PriceType.LastTrade)
        if price is None or price.is_nan() or price <= 0:
            price = self.connector.get_price_by_type(trading_pair, PriceType.MidPrice)
        return price if price is not None and not price.is_nan() else Decimal("0")

    def portfolio_equity(self) -> Decimal:
        equity = self.connector.get_balance(self.config.quote_asset)
        for trading_pair in self.config.trading_pairs:
            base_asset = trading_pair.split("-")[0]
            base_balance = self.connector.get_balance(base_asset)
            if base_balance > 0:
                equity += base_balance * self.reference_price(trading_pair)
        return equity

    def liquidate_portfolio(self, equity: Decimal, drawdown: Decimal):
        self.cancel_all_orders()
        for trading_pair in self.config.trading_pairs:
            base_asset = trading_pair.split("-")[0]
            amount = self.connector.get_available_balance(base_asset)
            price = self.reference_price(trading_pair)
            if amount > 0 and price > 0 and amount * price >= self.config.min_order_quote:
                self.sell(self.config.exchange, trading_pair, amount, OrderType.MARKET)
        self.liquidated = True
        self.cooldown_until = self.current_timestamp + self.config.cooldown_seconds
        self.peak_equity = equity
        self.notify(
            f"Portfolio liquidation triggered. equity={equity:.2f} "
            f"drawdown={drawdown:.2%} cooldown={self.config.cooldown_seconds}s"
        )

    def cancel_all_orders(self):
        for order in self.get_active_orders(self.config.exchange):
            self.cancel(self.config.exchange, order.trading_pair, order.client_order_id)

    def did_fill_order(self, event: OrderFilledEvent):
        msg = (
            f"{event.trade_type.name} {event.amount:.8f} {event.trading_pair} "
            f"at {event.price:.8f} on {self.config.exchange}"
        )
        self.log_with_clock(logging.INFO, msg)
        self.notify_hb_app_with_timestamp(msg)

    def notify(self, msg: str):
        self.log_with_clock(logging.INFO, msg)
        self.notify_hb_app_with_timestamp(msg)

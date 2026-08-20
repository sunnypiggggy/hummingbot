from pathlib import Path

import numpy as np
import pandas as pd

from scripts.backtest_binance_short_sideways_long_vs_bidirectional_360d import (
    END_TS,
    MINIMUM_ORDER,
    PRESETS,
    ROWS_PER_PAIR,
    START_TS,
    ExchangeFilter,
    build_grid_orders,
    initialise_state,
    place_grid,
    process_maker_fills,
    validate_candles,
)


def test_short_sideways_mapping_is_18_levels_and_10_fdusd_safe():
    preset = PRESETS["short_sideways_10usd"]
    exchange_filter = ExchangeFilter(tick_size=0.01, step_size=0.0001, minimum_notional=5.0)
    orders, clipped = build_grid_orders(
        pair="ETH-FDUSD",
        mode="bidirectional",
        preset=preset,
        exchange_filter=exchange_filter,
        center=4500.0,
        quote_available=100.0,
        base_available=100.0 / 4500.0,
        bar=0,
    )
    assert preset.levels == 18
    assert len([order for order in orders if order.side == "BUY"]) <= 9
    assert len([order for order in orders if order.side == "SELL"]) <= 9
    assert all(order.notional + 1e-9 >= MINIMUM_ORDER for order in orders)
    assert sum(order.notional for order in orders if order.side == "BUY") <= 100.0 + 1e-8
    assert sum(order.quantity for order in orders if order.side == "SELL") <= 100.0 / 4500.0 + 1e-12
    assert clipped >= 0


def test_long_only_has_buy_grid_but_no_unbacked_grid_sell():
    preset = PRESETS["short_sideways_10usd"]
    exchange_filter = ExchangeFilter(tick_size=0.01, step_size=0.00001, minimum_notional=5.0)
    orders, _ = build_grid_orders(
        pair="BTC-FDUSD",
        mode="long_only",
        preset=preset,
        exchange_filter=exchange_filter,
        center=100_000.0,
        quote_available=100.0,
        base_available=0.0,
        bar=0,
    )
    assert orders
    assert {order.side for order in orders} == {"BUY"}
    assert all(order.notional >= MINIMUM_ORDER for order in orders)


def test_long_only_sell_is_created_only_after_a_buy_fill():
    preset = PRESETS["short_sideways_10usd"]
    exchange_filter = ExchangeFilter(tick_size=0.01, step_size=0.0001, minimum_notional=5.0)
    state = initialise_state("ETH-FDUSD", "long_only", preset, exchange_filter, 4000.0)
    place_grid(state, 0, 4000.0)
    assert state.base == 0.0
    assert all(order.side == "BUY" for order in state.orders)
    buy = max(state.orders, key=lambda order: order.price)
    first = pd.Series({"timestamp": START_TS + 300, "open": 4000.0, "high": 4000.0,
                       "low": buy.price - 0.01, "close": buy.price})
    process_maker_fills(state, first, 1)
    assert state.base > 0.0
    take_profits = [order for order in state.orders if order.kind == "TAKE_PROFIT"]
    assert take_profits
    assert sum(order.quantity for order in take_profits) <= state.base + 1e-12
    target = max(order.price for order in take_profits)
    second = pd.Series({"timestamp": START_TS + 600, "open": target, "high": target + 0.01,
                        "low": target, "close": target})
    process_maker_fills(state, second, 2)
    assert state.base >= -1e-12


def test_candle_contract_accepts_exact_360_day_window():
    timestamps = np.arange(START_TS, END_TS, 300, dtype=np.int64)
    assert len(timestamps) == ROWS_PER_PAIR
    frame = pd.DataFrame({
        "timestamp": timestamps,
        "open": 100.0,
        "high": 101.0,
        "low": 99.0,
        "close": 100.5,
        "volume": 1.0,
    })
    validate_candles("TEST-FDUSD", frame)


def test_existing_inputs_cover_requested_window():
    root = Path(__file__).resolve().parents[1]
    for pair in ("BTC-FDUSD", "ETH-FDUSD"):
        frame = pd.read_csv(root / f"data/backtesting_candles/binance_{pair}_5m.csv", usecols=["timestamp"])
        count = ((frame.timestamp >= START_TS) & (frame.timestamp < END_TS)).sum()
        assert int(count) == ROWS_PER_PAIR

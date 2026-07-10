#!/usr/bin/env python
"""
Backtest a 3Commas-style moving spot grid with stop loss and parameter search.

Examples:
    python scripts/backtest_3commas_grid.py --days 30 --trading-pair BTC-USDT --optimize
    python scripts/backtest_3commas_grid.py --days 30 --grid-range 0.08 --grid-levels 24 --take-profit 0.003
"""

import argparse
import asyncio
import itertools
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from hummingbot.data_feed.candles_feed.data_types import CandlesConfig  # noqa: E402
from hummingbot.strategy_v2.backtesting.backtesting_data_provider import BacktestingDataProvider  # noqa: E402


@dataclass(frozen=True)
class GridParams:
    grid_range: float
    grid_levels: int
    order_quote_pct: float
    take_profit: float
    move_threshold: float
    stop_loss: float


@dataclass
class BacktestStats:
    net_pnl_quote: float
    net_pnl_pct: float
    max_drawdown_pct: float
    trades: int
    completed_cycles: int
    grid_moves: int
    stopped: bool
    final_quote: float
    final_base: float
    final_price: float
    params: GridParams


def make_levels(center_price: float, grid_range: float, grid_levels: int) -> Tuple[float, float, List[float]]:
    lower = center_price * (1 - grid_range / 2)
    upper = center_price * (1 + grid_range / 2)
    if grid_levels <= 1:
        return lower, upper, [center_price]
    step = (upper - lower) / (grid_levels - 1)
    return lower, upper, [lower + step * idx for idx in range(grid_levels)]


def simulate_grid(
    candles: pd.DataFrame,
    params: GridParams,
    initial_quote: float = 1000.0,
    fee_rate: float = 0.0002,
) -> BacktestStats:
    first_price = float(candles.iloc[0]["close"])
    center = first_price
    lower, upper, levels = make_levels(center, params.grid_range, params.grid_levels)
    quote = initial_quote
    base = 0.0
    lots: List[Dict[str, float]] = []
    active_buy_levels = set()
    equity_peak = initial_quote
    max_drawdown = 0.0
    trades = 0
    completed_cycles = 0
    grid_moves = 0
    stopped = False
    order_quote = initial_quote * params.order_quote_pct

    for _, row in candles.iterrows():
        low = float(row["low"])
        high = float(row["high"])
        close = float(row["close"])

        if close > upper * (1 + params.move_threshold) or close < lower * (1 - params.move_threshold):
            center = close
            lower, upper, levels = make_levels(center, params.grid_range, params.grid_levels)
            active_buy_levels.clear()
            grid_moves += 1

        for level in levels:
            if level >= close or level in active_buy_levels:
                continue
            if low <= level and quote >= order_quote:
                spent = order_quote
                acquired_base = (spent * (1 - fee_rate)) / level
                quote -= spent
                base += acquired_base
                lots.append({"entry": level, "base": acquired_base, "tp": level * (1 + params.take_profit)})
                active_buy_levels.add(level)
                trades += 1

        remaining_lots = []
        for lot in lots:
            if high >= lot["tp"]:
                proceeds = lot["base"] * lot["tp"] * (1 - fee_rate)
                quote += proceeds
                base -= lot["base"]
                active_buy_levels.discard(lot["entry"])
                completed_cycles += 1
                trades += 1
            else:
                remaining_lots.append(lot)
        lots = remaining_lots

        equity = quote + base * close
        equity_peak = max(equity_peak, equity)
        drawdown = (equity - equity_peak) / equity_peak
        max_drawdown = min(max_drawdown, drawdown)
        if drawdown <= -params.stop_loss:
            quote += base * close * (1 - fee_rate)
            base = 0.0
            lots.clear()
            stopped = True
            trades += 1
            break

    final_price = float(candles.iloc[-1]["close"])
    final_equity = quote + base * final_price
    net_pnl = final_equity - initial_quote
    return BacktestStats(
        net_pnl_quote=net_pnl,
        net_pnl_pct=net_pnl / initial_quote,
        max_drawdown_pct=max_drawdown,
        trades=trades,
        completed_cycles=completed_cycles,
        grid_moves=grid_moves,
        stopped=stopped,
        final_quote=quote,
        final_base=base,
        final_price=final_price,
        params=params,
    )


def default_search_space() -> List[GridParams]:
    return [
        GridParams(*values)
        for values in itertools.product(
            [0.04, 0.06, 0.08, 0.12],
            [12, 24, 36],
            [0.01, 0.02],
            [0.001, 0.002, 0.003],
            [0.005, 0.015],
            [0.04, 0.08],
        )
    ]


def stats_to_row(stats: BacktestStats) -> Dict[str, float]:
    row = asdict(stats)
    params = row.pop("params")
    row.update(params)
    return row


async def load_candles(connector: str, trading_pair: str, interval: str, days: float) -> pd.DataFrame:
    end_ts = int(time.time())
    start_ts = end_ts - int(days * 24 * 3600)
    provider = BacktestingDataProvider(connectors={})
    provider.update_backtesting_time(start_ts, end_ts)
    await provider.initialize_candles_feed(CandlesConfig(
        connector=connector,
        trading_pair=trading_pair,
        interval=interval,
        max_records=500,
    ))
    candles = provider.get_candles_df(connector, trading_pair, interval)
    if candles.empty:
        raise RuntimeError(f"No candles loaded for {connector} {trading_pair} {interval}")
    return candles.reset_index(drop=True)


async def main():
    parser = argparse.ArgumentParser(description="Backtest and optimize a moving 3Commas-style grid")
    parser.add_argument("--connector", default="binance")
    parser.add_argument("--trading-pair", default="BTC-USDT")
    parser.add_argument("--interval", default="5m")
    parser.add_argument("--days", type=float, default=30)
    parser.add_argument("--initial-quote", type=float, default=1000)
    parser.add_argument("--fee-rate", type=float, default=0.0002)
    parser.add_argument("--optimize", action="store_true")
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--output", default="backtest_3commas_grid_btc_1m.csv")
    parser.add_argument("--grid-range", type=float, default=0.08)
    parser.add_argument("--grid-levels", type=int, default=24)
    parser.add_argument("--order-quote-pct", type=float, default=0.02)
    parser.add_argument("--take-profit", type=float, default=0.002)
    parser.add_argument("--move-threshold", type=float, default=0.015)
    parser.add_argument("--stop-loss", type=float, default=0.08)
    args = parser.parse_args()

    candles = await load_candles(args.connector, args.trading_pair, args.interval, args.days)
    print(f"Loaded {len(candles)} candles for {args.connector} {args.trading_pair} {args.interval}")

    if args.optimize:
        candidates = default_search_space()
    else:
        candidates = [GridParams(
            args.grid_range,
            args.grid_levels,
            args.order_quote_pct,
            args.take_profit,
            args.move_threshold,
            args.stop_loss,
        )]

    results = [
        simulate_grid(candles, params, initial_quote=args.initial_quote, fee_rate=args.fee_rate)
        for params in candidates
    ]
    rows = [stats_to_row(stats) for stats in results]
    df = pd.DataFrame(rows)
    df.sort_values(by=["net_pnl_quote", "max_drawdown_pct", "completed_cycles"], ascending=[False, False, False],
                   inplace=True)
    output_path = Path(args.output)
    df.to_csv(output_path, index=False)

    print(f"\nSaved results to {output_path}")
    print("\nTop results:")
    display_cols = [
        "net_pnl_quote", "net_pnl_pct", "max_drawdown_pct", "trades", "completed_cycles", "grid_moves", "stopped",
        "grid_range", "grid_levels", "order_quote_pct", "take_profit", "move_threshold", "stop_loss",
    ]
    print(df[display_cols].head(args.top).to_string(index=False))


if __name__ == "__main__":
    asyncio.run(main())

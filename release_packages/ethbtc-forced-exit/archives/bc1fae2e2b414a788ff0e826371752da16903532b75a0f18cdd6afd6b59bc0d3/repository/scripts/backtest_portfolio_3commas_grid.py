#!/usr/bin/env python
"""
Backtest a portfolio of 3Commas-style moving spot grids with account-level drawdown liquidation.

The portfolio keeps one moving grid per USDT trading pair. Risk control is shared:
when current equity falls below peak equity by the configured drawdown threshold, the
script cancels all grids, liquidates all non-USDT balances at the current close price,
stops opening new positions, and observes a cooldown period.
"""

import argparse
import asyncio
import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, List

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.backtest_3commas_grid import GridParams, make_levels  # noqa: E402
from hummingbot.data_feed.candles_feed.data_types import CandlesConfig  # noqa: E402
from hummingbot.strategy_v2.backtesting.backtesting_data_provider import BacktestingDataProvider  # noqa: E402


DEFAULT_PAIRS = [
    "BTC-USDT",
    "ETH-USDT",
    "SOL-USDT",
    "BNB-USDT",
    "XRP-USDT",
    "DOGE-USDT",
    "ADA-USDT",
    "LINK-USDT",
    "AVAX-USDT",
    "TRX-USDT",
]


@dataclass
class PairState:
    trading_pair: str
    quote: float
    base: float
    lower: float
    upper: float
    levels: List[float]
    lots: List[Dict[str, float]]
    active_buy_levels: set
    trades: int = 0
    completed_cycles: int = 0
    grid_moves: int = 0
    last_grid_move_ts: float = 0.0


@dataclass
class MoveEvent:
    timestamp: float
    trading_pair: str
    direction: str
    trigger_price: float
    old_lower: float
    old_upper: float
    new_lower: float
    new_upper: float
    move_threshold: float


def initialize_pair_state(trading_pair: str, initial_quote: float, first_price: float,
                          params: GridParams) -> PairState:
    lower, upper, levels = make_levels(first_price, params.grid_range, params.grid_levels)
    return PairState(
        trading_pair=trading_pair,
        quote=initial_quote,
        base=0.0,
        lower=lower,
        upper=upper,
        levels=levels,
        lots=[],
        active_buy_levels=set(),
    )


def equity_at_price(state: PairState, price: float) -> float:
    return state.quote + state.base * price


async def load_pair_candles(
    connector: str,
    trading_pairs: List[str],
    interval: str,
    days: float,
    end_ts: int | None = None,
) -> Dict[str, pd.DataFrame]:
    end_ts = int(end_ts or time.time())
    start_ts = end_ts - int(days * 24 * 3600)
    provider = BacktestingDataProvider(connectors={})
    provider.update_backtesting_time(start_ts, end_ts)
    candles_by_pair = {}
    for trading_pair in trading_pairs:
        await provider.initialize_candles_feed(CandlesConfig(
            connector=connector,
            trading_pair=trading_pair,
            interval=interval,
            max_records=500,
        ))
        candles = provider.get_candles_df(connector, trading_pair, interval).copy()
        if candles.empty:
            raise RuntimeError(f"No candles loaded for {connector} {trading_pair} {interval}")
        candles_by_pair[trading_pair] = candles.reset_index(drop=True)
    return candles_by_pair


def simulate_portfolio(
    candles_by_pair: Dict[str, pd.DataFrame],
    params: GridParams,
    initial_quote: float,
    fee_rate: float,
    portfolio_stop_loss: float,
    cooldown_hours: float,
    min_grid_move_seconds: float = 0.0,
) -> tuple[Dict[str, float], pd.DataFrame, pd.DataFrame]:
    allocation = initial_quote / len(candles_by_pair)
    states = {
        pair: initialize_pair_state(pair, allocation, float(candles.iloc[0]["close"]), params)
        for pair, candles in candles_by_pair.items()
    }
    steps = min(len(candles) for candles in candles_by_pair.values())
    peak_equity = initial_quote
    max_drawdown = 0.0
    liquidated = False
    liquidation_step = None
    cooldown_steps = int(cooldown_hours * 3600 / (float(candles_by_pair[next(iter(candles_by_pair))].iloc[1]["timestamp"])
                                                  - float(candles_by_pair[next(iter(candles_by_pair))].iloc[0]["timestamp"])))
    cooldown_until = -1
    move_events: List[MoveEvent] = []
    equity_rows: List[Dict[str, float]] = []

    for idx in range(steps):
        prices = {pair: float(candles.iloc[idx]["close"]) for pair, candles in candles_by_pair.items()}
        lows = {pair: float(candles.iloc[idx]["low"]) for pair, candles in candles_by_pair.items()}
        highs = {pair: float(candles.iloc[idx]["high"]) for pair, candles in candles_by_pair.items()}
        timestamp = float(candles_by_pair[next(iter(candles_by_pair))].iloc[idx]["timestamp"])

        if idx >= cooldown_until and not liquidated:
            for pair, state in states.items():
                close = prices[pair]
                can_move = timestamp - state.last_grid_move_ts >= min_grid_move_seconds
                if can_move and (
                    close > state.upper * (1 + params.move_threshold)
                    or close < state.lower * (1 - params.move_threshold)
                ):
                    old_lower = state.lower
                    old_upper = state.upper
                    direction = "up" if close > state.upper * (1 + params.move_threshold) else "down"
                    state.lower, state.upper, state.levels = make_levels(close, params.grid_range, params.grid_levels)
                    state.active_buy_levels.clear()
                    state.grid_moves += 1
                    state.last_grid_move_ts = timestamp
                    move_events.append(MoveEvent(
                        timestamp=timestamp,
                        trading_pair=pair,
                        direction=direction,
                        trigger_price=close,
                        old_lower=old_lower,
                        old_upper=old_upper,
                        new_lower=state.lower,
                        new_upper=state.upper,
                        move_threshold=params.move_threshold,
                    ))

                order_quote = allocation * params.order_quote_pct
                for level in state.levels:
                    if level >= close or level in state.active_buy_levels:
                        continue
                    if lows[pair] <= level and state.quote >= order_quote:
                        acquired_base = (order_quote * (1 - fee_rate)) / level
                        state.quote -= order_quote
                        state.base += acquired_base
                        state.lots.append({"entry": level, "base": acquired_base, "tp": level * (1 + params.take_profit)})
                        state.active_buy_levels.add(level)
                        state.trades += 1

                remaining_lots = []
                for lot in state.lots:
                    if highs[pair] >= lot["tp"]:
                        state.quote += lot["base"] * lot["tp"] * (1 - fee_rate)
                        state.base -= lot["base"]
                        state.active_buy_levels.discard(lot["entry"])
                        state.completed_cycles += 1
                        state.trades += 1
                    else:
                        remaining_lots.append(lot)
                state.lots = remaining_lots

        current_equity = sum(equity_at_price(state, prices[pair]) for pair, state in states.items())
        peak_equity = max(peak_equity, current_equity)
        drawdown = (current_equity - peak_equity) / peak_equity
        max_drawdown = min(max_drawdown, drawdown)
        equity_rows.append({
            "timestamp": timestamp,
            "equity": current_equity,
            "peak_equity": peak_equity,
            "drawdown_pct": drawdown,
        })

        if drawdown <= -portfolio_stop_loss and not liquidated:
            for pair, state in states.items():
                if state.base > 0:
                    state.quote += state.base * prices[pair] * (1 - fee_rate)
                    state.base = 0.0
                    state.lots.clear()
                    state.active_buy_levels.clear()
                    state.trades += 1
            liquidated = True
            liquidation_step = idx
            cooldown_until = idx + cooldown_steps
            break

    final_prices = {pair: float(candles.iloc[min(steps - 1, liquidation_step or steps - 1)]["close"])
                    for pair, candles in candles_by_pair.items()}
    final_equity = sum(equity_at_price(state, final_prices[pair]) for pair, state in states.items())
    total_trades = sum(state.trades for state in states.values())
    total_cycles = sum(state.completed_cycles for state in states.values())
    total_moves = sum(state.grid_moves for state in states.values())
    open_positions = sum(1 for state in states.values() if state.base > 0)

    result = {
        "initial_quote": initial_quote,
        "final_equity": final_equity,
        "net_pnl_quote": final_equity - initial_quote,
        "net_pnl_pct": (final_equity - initial_quote) / initial_quote,
        "max_drawdown_pct": max_drawdown,
        "liquidated": liquidated,
        "liquidation_step": liquidation_step if liquidation_step is not None else -1,
        "cooldown_hours": cooldown_hours,
        "trades": total_trades,
        "completed_cycles": total_cycles,
        "grid_moves": total_moves,
        "open_positions": open_positions,
        "portfolio_stop_loss": portfolio_stop_loss,
    }
    return result, pd.DataFrame(move_events), pd.DataFrame(equity_rows)


def write_html_report(output_path: str, result: Dict[str, float], move_events: pd.DataFrame,
                      equity_curve: pd.DataFrame) -> None:
    if equity_curve.empty:
        return
    chart_points = []
    min_equity = equity_curve["equity"].min()
    max_equity = equity_curve["equity"].max()
    span = max(max_equity - min_equity, 1)
    width = 960
    height = 260
    for idx, row in equity_curve.reset_index(drop=True).iterrows():
        x = idx / max(len(equity_curve) - 1, 1) * width
        y = height - ((row["equity"] - min_equity) / span * height)
        chart_points.append(f"{x:.2f},{y:.2f}")
    move_summary = (
        move_events.groupby(["trading_pair", "direction"]).size().unstack(fill_value=0).reset_index()
        if not move_events.empty else pd.DataFrame(columns=["trading_pair", "up", "down"])
    )
    if "up" not in move_summary.columns:
        move_summary["up"] = 0
    if "down" not in move_summary.columns:
        move_summary["down"] = 0
    move_summary["total"] = move_summary["up"] + move_summary["down"]
    move_summary.sort_values("total", ascending=False, inplace=True)

    summary_rows = "\n".join(
        f"<tr><td>{row.trading_pair}</td><td>{int(row.total)}</td><td>{int(row.up)}</td><td>{int(row.down)}</td></tr>"
        for row in move_summary.itertuples(index=False)
    )
    event_rows = "\n".join(
        "<tr>"
        f"<td>{pd.to_datetime(row.timestamp, unit='s')}</td>"
        f"<td>{row.trading_pair}</td>"
        f"<td>{row.direction}</td>"
        f"<td>{row.trigger_price:.6f}</td>"
        f"<td>{row.old_lower:.6f} - {row.old_upper:.6f}</td>"
        f"<td>{row.new_lower:.6f} - {row.new_upper:.6f}</td>"
        "</tr>"
        for row in move_events.tail(200).itertuples(index=False)
    )
    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>Portfolio Grid Backtest Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #17202a; }}
    h1, h2 {{ margin: 0 0 12px; }}
    .metrics {{ display: grid; grid-template-columns: repeat(4, minmax(150px, 1fr)); gap: 12px; margin: 18px 0; }}
    .metric {{ border: 1px solid #d7dde5; border-radius: 6px; padding: 12px; }}
    .metric b {{ display: block; font-size: 13px; color: #5f6b7a; margin-bottom: 6px; }}
    svg {{ width: 100%; height: 300px; border: 1px solid #d7dde5; border-radius: 6px; background: #fbfcfe; }}
    table {{ border-collapse: collapse; width: 100%; margin: 12px 0 26px; font-size: 13px; }}
    th, td {{ border: 1px solid #d7dde5; padding: 7px 9px; text-align: right; }}
    th:first-child, td:first-child, td:nth-child(2) {{ text-align: left; }}
    th {{ background: #f1f4f8; }}
  </style>
</head>
<body>
  <h1>组合动态网格回测报告</h1>
  <div class="metrics">
    <div class="metric"><b>净利润</b>{result["net_pnl_quote"]:.2f} USDT ({result["net_pnl_pct"] * 100:.2f}%)</div>
    <div class="metric"><b>最大回撤</b>{result["max_drawdown_pct"] * 100:.2f}%</div>
    <div class="metric"><b>网格移动次数</b>{int(result["grid_moves"])}</div>
    <div class="metric"><b>是否触发清仓</b>{result["liquidated"]}</div>
  </div>
  <h2>权益曲线</h2>
  <svg viewBox="0 0 {width} {height + 30}" preserveAspectRatio="none">
    <polyline fill="none" stroke="#2563eb" stroke-width="2" points="{" ".join(chart_points)}" />
    <text x="8" y="18" font-size="12">max {max_equity:.2f}</text>
    <text x="8" y="{height - 6}" font-size="12">min {min_equity:.2f}</text>
  </svg>
  <h2>网格移动汇总</h2>
  <table>
    <thead><tr><th>交易对</th><th>总移动</th><th>向上移动</th><th>向下移动</th></tr></thead>
    <tbody>{summary_rows}</tbody>
  </table>
  <h2>最近 200 次移动触发明细</h2>
  <table>
    <thead><tr><th>时间</th><th>交易对</th><th>方向</th><th>触发价</th><th>旧网格</th><th>新网格</th></tr></thead>
    <tbody>{event_rows}</tbody>
  </table>
</body>
</html>"""
    with open(output_path, "w", encoding="utf-8") as outfile:
        outfile.write(html)


async def main():
    parser = argparse.ArgumentParser(description="Backtest a portfolio moving grid with drawdown liquidation")
    parser.add_argument("--connector", default="binance")
    parser.add_argument("--trading-pairs", default=",".join(DEFAULT_PAIRS))
    parser.add_argument("--interval", default="5m")
    parser.add_argument("--days", type=float, default=30)
    parser.add_argument("--end-ts", type=int, default=None)
    parser.add_argument("--initial-quote", type=float, default=10000)
    parser.add_argument("--fee-rate", type=float, default=0.0002)
    parser.add_argument("--portfolio-stop-loss", type=float, default=0.08)
    parser.add_argument("--cooldown-hours", type=float, default=24)
    parser.add_argument("--min-grid-move-seconds", type=float, default=0)
    parser.add_argument("--grid-range", type=float, default=0.04)
    parser.add_argument("--grid-levels", type=int, default=36)
    parser.add_argument("--order-quote-pct", type=float, default=0.02)
    parser.add_argument("--take-profit", type=float, default=0.002)
    parser.add_argument("--move-threshold", type=float, default=0.005)
    parser.add_argument("--output", default="backtest_portfolio_3commas_grid.csv")
    parser.add_argument("--moves-output", default="backtest_portfolio_3commas_grid_moves.csv")
    parser.add_argument("--equity-output", default="backtest_portfolio_3commas_grid_equity.csv")
    parser.add_argument("--html-output", default="backtest_portfolio_3commas_grid_report.html")
    args = parser.parse_args()

    trading_pairs = [pair.strip().upper() for pair in args.trading_pairs.split(",") if pair.strip()]
    params = GridParams(
        args.grid_range,
        args.grid_levels,
        args.order_quote_pct,
        args.take_profit,
        args.move_threshold,
        args.portfolio_stop_loss,
    )
    candles_by_pair = await load_pair_candles(args.connector, trading_pairs, args.interval, args.days, args.end_ts)
    result, move_events, equity_curve = simulate_portfolio(
        candles_by_pair=candles_by_pair,
        params=params,
        initial_quote=args.initial_quote,
        fee_rate=args.fee_rate,
        portfolio_stop_loss=args.portfolio_stop_loss,
        cooldown_hours=args.cooldown_hours,
        min_grid_move_seconds=args.min_grid_move_seconds,
    )
    df = pd.DataFrame([{**result, **{
        "trading_pairs": ",".join(trading_pairs),
        "grid_range": args.grid_range,
        "grid_levels": args.grid_levels,
        "order_quote_pct": args.order_quote_pct,
        "take_profit": args.take_profit,
        "move_threshold": args.move_threshold,
        "min_grid_move_seconds": args.min_grid_move_seconds,
    }}])
    df.to_csv(args.output, index=False)
    move_events.to_csv(args.moves_output, index=False)
    equity_curve.to_csv(args.equity_output, index=False)
    write_html_report(args.html_output, result, move_events, equity_curve)

    print(f"Saved result to {args.output}")
    print(f"Saved move events to {args.moves_output}")
    print(f"Saved equity curve to {args.equity_output}")
    print(f"Saved HTML report to {args.html_output}")
    print(df.T.to_string(header=False))


if __name__ == "__main__":
    asyncio.run(main())

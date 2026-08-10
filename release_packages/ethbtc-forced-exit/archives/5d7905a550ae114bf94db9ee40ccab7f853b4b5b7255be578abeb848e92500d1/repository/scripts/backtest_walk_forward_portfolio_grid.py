#!/usr/bin/env python
"""
Walk-forward backtest for a portfolio of moving spot grids.

The first month is used for initial parameter selection. After that, each
trading week picks the best grid parameters from the previous lookback window
and trades the next week with those parameters.
"""

import argparse
import asyncio
import concurrent.futures
import html
import itertools
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from hummingbot.data_feed.candles_feed.data_types import CandlesConfig  # noqa: E402
from hummingbot.strategy_v2.backtesting.backtesting_data_provider import BacktestingDataProvider  # noqa: E402
from scripts.backtest_3commas_grid import GridParams, make_levels  # noqa: E402
from scripts.backtest_portfolio_3commas_grid import (  # noqa: E402
    DEFAULT_PAIRS,
    MoveEvent,
    PairState,
    equity_at_price,
    initialize_pair_state,
    simulate_portfolio,
)


SECONDS_PER_DAY = 24 * 60 * 60


def default_walk_forward_search_space() -> List[GridParams]:
    return [
        GridParams(*values)
        for values in itertools.product(
            [0.04, 0.06, 0.08, 0.10],
            [8, 12, 16, 24],
            [0.01, 0.015, 0.02],
            [0.003, 0.005, 0.008, 0.01],
            [0.005, 0.01, 0.015],
            [0.04],
        )
    ]


def parse_pairs(raw_pairs: str, limit: int | None = None) -> List[str]:
    pairs = [pair.strip().upper() for pair in raw_pairs.split(",") if pair.strip()]
    return pairs[:limit] if limit else pairs


async def load_pair_candles_range(
    connector: str,
    trading_pairs: List[str],
    interval: str,
    start_ts: int,
    end_ts: int,
) -> Dict[str, pd.DataFrame]:
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
        candles["timestamp"] = candles["timestamp"].astype(float)
        candles_by_pair[trading_pair] = candles.sort_values("timestamp").reset_index(drop=True)
    return candles_by_pair


def slice_candles(candles_by_pair: Dict[str, pd.DataFrame], start_ts: float, end_ts: float) -> Dict[str, pd.DataFrame]:
    sliced = {}
    for pair, candles in candles_by_pair.items():
        mask = (candles["timestamp"] >= start_ts) & (candles["timestamp"] < end_ts)
        window = candles.loc[mask].reset_index(drop=True)
        if window.empty:
            return {}
        sliced[pair] = window
    return sliced


def align_window(candles_by_pair: Dict[str, pd.DataFrame]) -> Tuple[Dict[str, pd.DataFrame], int]:
    steps = min(len(candles) for candles in candles_by_pair.values())
    return {pair: candles.iloc[:steps].reset_index(drop=True) for pair, candles in candles_by_pair.items()}, steps


def score_result(result: Dict[str, float], drawdown_penalty: float, liquidation_penalty: float) -> float:
    score = result["net_pnl_pct"] - abs(result["max_drawdown_pct"]) * drawdown_penalty
    if result["liquidated"]:
        score -= liquidation_penalty
    return score


def optimize_params(
    candles_by_pair: Dict[str, pd.DataFrame],
    candidates: Iterable[GridParams],
    initial_quote: float,
    fee_rate: float,
    portfolio_stop_loss: float,
    cooldown_hours: float,
    min_grid_move_seconds: float,
    drawdown_penalty: float,
    liquidation_penalty: float,
) -> Tuple[GridParams, Dict[str, float], float, pd.DataFrame]:
    best_params = None
    best_result = None
    best_score = float("-inf")
    candidate_rows = []
    for params in candidates:
        result, _, _ = simulate_portfolio(
            candles_by_pair=candles_by_pair,
            params=params,
            initial_quote=initial_quote,
            fee_rate=fee_rate,
            portfolio_stop_loss=portfolio_stop_loss,
            cooldown_hours=cooldown_hours,
            min_grid_move_seconds=min_grid_move_seconds,
        )
        score = score_result(result, drawdown_penalty, liquidation_penalty)
        candidate_rows.append({
            **asdict(params),
            "selection_score": score,
            "selection_net_pnl_quote": result["net_pnl_quote"],
            "selection_net_pnl_pct": result["net_pnl_pct"],
            "selection_max_drawdown_pct": result["max_drawdown_pct"],
            "selection_liquidated": result["liquidated"],
            "selection_trades": result["trades"],
            "selection_completed_cycles": result["completed_cycles"],
            "selection_grid_moves": result["grid_moves"],
            "selection_open_positions": result["open_positions"],
        })
        if score > best_score:
            best_score = score
            best_params = params
            best_result = result
    if best_params is None or best_result is None:
        raise RuntimeError("No parameter candidates could be evaluated.")
    return best_params, best_result, best_score, pd.DataFrame(candidate_rows)


def _evaluate_optimization_candidate(args: Tuple[
    Dict[str, pd.DataFrame],
    GridParams,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
]) -> Tuple[GridParams, Dict[str, float], float, Dict[str, float]]:
    (
        candles_by_pair,
        params,
        initial_quote,
        fee_rate,
        portfolio_stop_loss,
        cooldown_hours,
        min_grid_move_seconds,
        drawdown_penalty,
        liquidation_penalty,
    ) = args
    result, _, _ = simulate_portfolio(
        candles_by_pair=candles_by_pair,
        params=params,
        initial_quote=initial_quote,
        fee_rate=fee_rate,
        portfolio_stop_loss=portfolio_stop_loss,
        cooldown_hours=cooldown_hours,
        min_grid_move_seconds=min_grid_move_seconds,
    )
    score = score_result(result, drawdown_penalty, liquidation_penalty)
    row = {
        **asdict(params),
        "selection_score": score,
        "selection_net_pnl_quote": result["net_pnl_quote"],
        "selection_net_pnl_pct": result["net_pnl_pct"],
        "selection_max_drawdown_pct": result["max_drawdown_pct"],
        "selection_liquidated": result["liquidated"],
        "selection_grid_moves": result["grid_moves"],
        "selection_trades": result["trades"],
        "selection_open_positions": result["open_positions"],
    }
    return params, result, score, row


def _evaluate_optimization_chunk(args: Tuple[
    Dict[str, pd.DataFrame],
    List[GridParams],
    float,
    float,
    float,
    float,
    float,
    float,
    float,
]) -> List[Tuple[GridParams, Dict[str, float], float, Dict[str, float]]]:
    (
        candles_by_pair,
        candidates,
        initial_quote,
        fee_rate,
        portfolio_stop_loss,
        cooldown_hours,
        min_grid_move_seconds,
        drawdown_penalty,
        liquidation_penalty,
    ) = args
    return [
        _evaluate_optimization_candidate((
            candles_by_pair,
            params,
            initial_quote,
            fee_rate,
            portfolio_stop_loss,
            cooldown_hours,
            min_grid_move_seconds,
            drawdown_penalty,
            liquidation_penalty,
        ))
        for params in candidates
    ]


def optimize_params_parallel(
    candles_by_pair: Dict[str, pd.DataFrame],
    candidates: Iterable[GridParams],
    initial_quote: float,
    fee_rate: float,
    portfolio_stop_loss: float,
    cooldown_hours: float,
    drawdown_penalty: float,
    liquidation_penalty: float,
    min_grid_move_seconds: float = 0.0,
    workers: int = 1,
) -> Tuple[GridParams, Dict[str, float], float, pd.DataFrame]:
    candidate_list = list(candidates)
    if workers <= 1 or len(candidate_list) <= 1:
        return optimize_params(
            candles_by_pair=candles_by_pair,
            candidates=candidate_list,
            initial_quote=initial_quote,
            fee_rate=fee_rate,
            portfolio_stop_loss=portfolio_stop_loss,
            cooldown_hours=cooldown_hours,
            min_grid_move_seconds=min_grid_move_seconds,
            drawdown_penalty=drawdown_penalty,
            liquidation_penalty=liquidation_penalty,
        )

    worker_count = max(1, min(workers, len(candidate_list)))
    chunks = [candidate_list[idx::worker_count] for idx in range(worker_count)]
    task_args = [
        (
            candles_by_pair,
            chunk,
            initial_quote,
            fee_rate,
            portfolio_stop_loss,
            cooldown_hours,
            min_grid_move_seconds,
            drawdown_penalty,
            liquidation_penalty,
        )
        for chunk in chunks
        if chunk
    ]
    best_params = None
    best_result = None
    best_score = float("-inf")
    candidate_rows = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=worker_count) as executor:
        for chunk_results in executor.map(_evaluate_optimization_chunk, task_args):
            for params, result, score, row in chunk_results:
                candidate_rows.append(row)
                if score > best_score:
                    best_score = score
                    best_params = params
                    best_result = result
    if best_params is None or best_result is None:
        raise RuntimeError("No parameter candidates could be evaluated.")
    return best_params, best_result, best_score, pd.DataFrame(candidate_rows)


def portfolio_equity(states: Dict[str, PairState], prices: Dict[str, float]) -> float:
    return sum(equity_at_price(state, prices[pair]) for pair, state in states.items())


def regrid_states(states: Dict[str, PairState], prices: Dict[str, float], params: GridParams) -> None:
    for pair, state in states.items():
        state.lower, state.upper, state.levels = make_levels(prices[pair], params.grid_range, params.grid_levels)
        state.active_buy_levels.clear()


def simulate_trading_window(
    candles_by_pair: Dict[str, pd.DataFrame],
    states: Dict[str, PairState],
    params: GridParams,
    fee_rate: float,
    portfolio_stop_loss: float,
    cooldown_hours: float,
    min_grid_move_seconds: float,
    risk_peak_equity: float,
    cooldown_until_ts: float,
    global_step: int,
    week_index: int,
) -> Tuple[Dict[str, float], pd.DataFrame, pd.DataFrame, float, float, int]:
    candles_by_pair, steps = align_window(candles_by_pair)
    if steps < 2:
        raise RuntimeError("Trading window must contain at least two candles per pair.")

    first_pair = next(iter(candles_by_pair))
    step_seconds = float(candles_by_pair[first_pair].iloc[1]["timestamp"]) - float(candles_by_pair[first_pair].iloc[0]["timestamp"])
    cooldown_seconds = max(cooldown_hours * 3600, step_seconds)
    start_equity = None
    max_drawdown = 0.0
    trades_start = sum(state.trades for state in states.values())
    cycles_start = sum(state.completed_cycles for state in states.values())
    moves_start = sum(state.grid_moves for state in states.values())
    liquidations = 0
    move_events: List[MoveEvent] = []
    equity_rows: List[Dict[str, float]] = []

    for idx in range(steps):
        timestamp = float(candles_by_pair[first_pair].iloc[idx]["timestamp"])
        prices = {pair: float(candles.iloc[idx]["close"]) for pair, candles in candles_by_pair.items()}
        lows = {pair: float(candles.iloc[idx]["low"]) for pair, candles in candles_by_pair.items()}
        highs = {pair: float(candles.iloc[idx]["high"]) for pair, candles in candles_by_pair.items()}
        current_equity = portfolio_equity(states, prices)
        if start_equity is None:
            start_equity = current_equity

        if timestamp >= cooldown_until_ts:
            allocation = max(current_equity / max(len(states), 1), 0.0)
            for pair, state in states.items():
                close = prices[pair]
                up_trigger = close > state.upper * (1 + params.move_threshold)
                down_trigger = close < state.lower * (1 - params.move_threshold)
                can_move = timestamp - state.last_grid_move_ts >= min_grid_move_seconds
                if can_move and (up_trigger or down_trigger):
                    old_lower = state.lower
                    old_upper = state.upper
                    direction = "up" if up_trigger else "down"
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
                    if lows[pair] <= level and state.quote >= order_quote > 0:
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

        current_equity = portfolio_equity(states, prices)
        risk_peak_equity = max(risk_peak_equity, current_equity)
        drawdown = (current_equity - risk_peak_equity) / risk_peak_equity if risk_peak_equity else 0.0
        max_drawdown = min(max_drawdown, drawdown)

        if drawdown <= -portfolio_stop_loss:
            for pair, state in states.items():
                if state.base > 0:
                    state.quote += state.base * prices[pair] * (1 - fee_rate)
                    state.base = 0.0
                    state.lots.clear()
                    state.active_buy_levels.clear()
                    state.trades += 1
            liquidations += 1
            cooldown_until_ts = timestamp + cooldown_seconds
            current_equity = portfolio_equity(states, prices)
            risk_peak_equity = current_equity
            drawdown = 0.0

        equity_rows.append({
            "global_step": global_step + idx,
            "week_index": week_index,
            "timestamp": timestamp,
            "equity": current_equity,
            "risk_peak_equity": risk_peak_equity,
            "drawdown_pct": drawdown,
            "cooldown_active": timestamp < cooldown_until_ts,
        })

    final_prices = {pair: float(candles.iloc[steps - 1]["close"]) for pair, candles in candles_by_pair.items()}
    final_equity = portfolio_equity(states, final_prices)
    start_equity = start_equity if start_equity is not None else final_equity
    result = {
        "start_equity": start_equity,
        "final_equity": final_equity,
        "net_pnl_quote": final_equity - start_equity,
        "net_pnl_pct": (final_equity - start_equity) / start_equity if start_equity else 0.0,
        "max_drawdown_pct": max_drawdown,
        "liquidations": liquidations,
        "trades": sum(state.trades for state in states.values()) - trades_start,
        "completed_cycles": sum(state.completed_cycles for state in states.values()) - cycles_start,
        "grid_moves": sum(state.grid_moves for state in states.values()) - moves_start,
        "open_positions": sum(1 for state in states.values() if state.base > 0),
    }
    return result, pd.DataFrame(move_events), pd.DataFrame(equity_rows), risk_peak_equity, cooldown_until_ts, global_step + steps


def dataframe_to_rows(df: pd.DataFrame, columns: List[str], formats: Dict[str, str] | None = None, limit: int | None = None) -> str:
    if df.empty:
        return "<tr><td colspan=\"{}\">No rows</td></tr>".format(len(columns))
    rows = []
    view = df.tail(limit) if limit else df
    formats = formats or {}
    for _, row in view.iterrows():
        cells = []
        for column in columns:
            value = row.get(column, "")
            if column in formats and pd.notna(value):
                value = formats[column].format(value)
            cells.append(f"<td>{html.escape(str(value))}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return "\n".join(rows)


def write_html_report(
    output_path: str,
    summary: pd.DataFrame,
    moves: pd.DataFrame,
    equity: pd.DataFrame,
    combo_report: pd.DataFrame | None = None,
) -> None:
    if equity.empty:
        return
    width = 980
    height = 280
    min_equity = float(equity["equity"].min())
    max_equity = float(equity["equity"].max())
    span = max(max_equity - min_equity, 1.0)
    points = []
    for idx, row in equity.reset_index(drop=True).iterrows():
        x = idx / max(len(equity) - 1, 1) * width
        y = height - ((float(row["equity"]) - min_equity) / span * height)
        points.append(f"{x:.2f},{y:.2f}")

    move_summary = (
        moves.groupby(["trading_pair", "direction"]).size().unstack(fill_value=0).reset_index()
        if not moves.empty else pd.DataFrame(columns=["trading_pair", "up", "down"])
    )
    for column in ["up", "down"]:
        if column not in move_summary.columns:
            move_summary[column] = 0
    move_summary["total"] = move_summary["up"] + move_summary["down"]
    move_summary.sort_values("total", ascending=False, inplace=True)

    final_equity = float(equity.iloc[-1]["equity"])
    start_equity = float(summary.iloc[0]["start_equity"]) if not summary.empty else final_equity
    net_pct = (final_equity - start_equity) / start_equity if start_equity else 0.0
    max_dd = float(equity["drawdown_pct"].min())
    liquidations = int(summary["liquidations"].sum()) if not summary.empty else 0
    total_moves = int(summary["grid_moves"].sum()) if not summary.empty else 0

    weekly_columns = [
        "week_index", "trade_start", "trade_end", "net_pnl_pct", "max_drawdown_pct", "grid_moves",
        "liquidations", "grid_range", "grid_levels", "order_quote_pct", "take_profit", "move_threshold",
    ]
    combo_columns = [
        "rank", "grid_range", "grid_levels", "order_quote_pct", "take_profit", "move_threshold", "stop_loss",
        "avg_selection_score", "compound_net_pnl_pct", "avg_net_pnl_pct", "worst_max_drawdown_pct", "win_rate",
        "liquidation_count",
    ]
    move_columns = ["timestamp", "trading_pair", "direction", "trigger_price", "old_lower", "old_upper", "new_lower", "new_upper"]
    summary_formats = {
        "net_pnl_pct": "{:.4%}",
        "max_drawdown_pct": "{:.4%}",
        "avg_selection_score": "{:.6f}",
        "compound_net_pnl_pct": "{:.4%}",
        "avg_net_pnl_pct": "{:.4%}",
        "worst_max_drawdown_pct": "{:.4%}",
        "win_rate": "{:.2%}",
    }
    move_formats = {column: "{:.6f}" for column in ["trigger_price", "old_lower", "old_upper", "new_lower", "new_upper"]}

    document = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>Walk-forward Portfolio Grid Backtest</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #18212f; background: #ffffff; }}
    h1, h2 {{ margin: 0 0 12px; }}
    h2 {{ margin-top: 28px; }}
    .metrics {{ display: grid; grid-template-columns: repeat(4, minmax(150px, 1fr)); gap: 12px; margin: 18px 0; }}
    .metric {{ border: 1px solid #d7dde5; border-radius: 6px; padding: 12px; background: #fbfcfe; }}
    .metric b {{ display: block; font-size: 13px; color: #5d6878; margin-bottom: 6px; }}
    svg {{ width: 100%; height: 320px; border: 1px solid #d7dde5; border-radius: 6px; background: #fbfcfe; }}
    table {{ border-collapse: collapse; width: 100%; margin: 12px 0 26px; font-size: 13px; }}
    th, td {{ border: 1px solid #d7dde5; padding: 7px 9px; text-align: right; white-space: nowrap; }}
    th:first-child, td:first-child, td:nth-child(2), td:nth-child(3) {{ text-align: left; }}
    th {{ background: #f2f5f9; }}
  </style>
</head>
<body>
  <h1>Walk-forward Portfolio Grid Backtest</h1>
  <div class="metrics">
    <div class="metric"><b>Net PnL</b>{final_equity - start_equity:.2f} USDT ({net_pct:.2%})</div>
    <div class="metric"><b>Max Drawdown</b>{max_dd:.2%}</div>
    <div class="metric"><b>Grid Moves</b>{total_moves}</div>
    <div class="metric"><b>Liquidations</b>{liquidations}</div>
  </div>
  <h2>Equity Curve</h2>
  <svg viewBox="0 0 {width} {height + 34}" preserveAspectRatio="none">
    <polyline fill="none" stroke="#2563eb" stroke-width="2" points="{" ".join(points)}" />
    <text x="8" y="18" font-size="12">max {max_equity:.2f}</text>
    <text x="8" y="{height - 6}" font-size="12">min {min_equity:.2f}</text>
  </svg>
  <h2>Weekly Parameters and Returns</h2>
  <table>
    <thead><tr>{"".join(f"<th>{html.escape(column)}</th>" for column in weekly_columns)}</tr></thead>
    <tbody>{dataframe_to_rows(summary, weekly_columns, summary_formats)}</tbody>
  </table>
  <h2>Top Candidate Parameter Combos</h2>
  <table>
    <thead><tr>{"".join(f"<th>{html.escape(column)}</th>" for column in combo_columns)}</tr></thead>
    <tbody>{dataframe_to_rows(combo_report.head(20) if combo_report is not None else pd.DataFrame(), combo_columns, summary_formats)}</tbody>
  </table>
  <h2>Grid Move Summary</h2>
  <table>
    <thead><tr><th>trading_pair</th><th>up</th><th>down</th><th>total</th></tr></thead>
    <tbody>{dataframe_to_rows(move_summary, ["trading_pair", "up", "down", "total"])}</tbody>
  </table>
  <h2>Recent Move Events</h2>
  <table>
    <thead><tr>{"".join(f"<th>{html.escape(column)}</th>" for column in move_columns)}</tr></thead>
    <tbody>{dataframe_to_rows(moves, move_columns, move_formats, limit=250)}</tbody>
  </table>
</body>
</html>"""
    Path(output_path).write_text(document, encoding="utf-8")


def build_param_analysis(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    group_cols = ["grid_range", "grid_levels", "order_quote_pct", "take_profit", "move_threshold", "stop_loss"]
    grouped = summary.groupby(group_cols, dropna=False)
    rows = []
    for params, group in grouped:
        net_pnl_quote = float(group["net_pnl_quote"].sum())
        net_pnl_pct = float((1 + group["net_pnl_pct"]).prod() - 1)
        max_dd = float(group["max_drawdown_pct"].min())
        mean_weekly = float(group["net_pnl_pct"].mean())
        win_rate = float((group["net_pnl_pct"] > 0).mean())
        liquidation_count = int(group["liquidations"].sum())
        realized_score = mean_weekly - abs(max_dd) * 1.5 - (0.25 if liquidation_count else 0.0)
        row = dict(zip(group_cols, params))
        row.update({
            "weeks_used": len(group),
            "first_week": int(group["week_index"].min()),
            "last_week": int(group["week_index"].max()),
            "realized_net_pnl_quote": net_pnl_quote,
            "realized_net_pnl_pct": net_pnl_pct,
            "mean_weekly_pnl_pct": mean_weekly,
            "max_drawdown_pct": max_dd,
            "win_rate": win_rate,
            "total_grid_moves": int(group["grid_moves"].sum()),
            "total_trades": int(group["trades"].sum()),
            "liquidations": liquidation_count,
            "realized_score": realized_score,
        })
        rows.append(row)
    analysis = pd.DataFrame(rows)
    if not analysis.empty:
        analysis.sort_values(
            by=["realized_score", "realized_net_pnl_pct", "weeks_used", "max_drawdown_pct"],
            ascending=[False, False, False, False],
            inplace=True,
        )
    return analysis


def build_combo_report(candidate_evaluations: pd.DataFrame) -> pd.DataFrame:
    if candidate_evaluations.empty:
        return pd.DataFrame()
    group_cols = ["grid_range", "grid_levels", "order_quote_pct", "take_profit", "move_threshold", "stop_loss"]
    grouped = candidate_evaluations.groupby(group_cols, dropna=False)
    rows = []
    for params, group in grouped:
        row = dict(zip(group_cols, params))
        row.update({
            "windows_evaluated": len(group),
            "avg_selection_score": float(group["selection_score"].mean()),
            "median_selection_score": float(group["selection_score"].median()),
            "avg_net_pnl_pct": float(group["selection_net_pnl_pct"].mean()),
            "median_net_pnl_pct": float(group["selection_net_pnl_pct"].median()),
            "compound_net_pnl_pct": float((1 + group["selection_net_pnl_pct"]).prod() - 1),
            "avg_max_drawdown_pct": float(group["selection_max_drawdown_pct"].mean()),
            "worst_max_drawdown_pct": float(group["selection_max_drawdown_pct"].min()),
            "win_rate": float((group["selection_net_pnl_pct"] > 0).mean()),
            "liquidation_count": int(group["selection_liquidated"].sum()),
            "avg_trades": float(group["selection_trades"].mean()),
            "avg_grid_moves": float(group["selection_grid_moves"].mean()),
        })
        rows.append(row)
    report = pd.DataFrame(rows)
    if not report.empty:
        report.sort_values(
            by=["avg_selection_score", "compound_net_pnl_pct", "worst_max_drawdown_pct"],
            ascending=[False, False, False],
            inplace=True,
        )
        report.insert(0, "rank", range(1, len(report) + 1))
    return report


def write_result_markdown(
    output_path: str,
    summary: pd.DataFrame,
    param_analysis: pd.DataFrame,
    moves: pd.DataFrame,
    combo_report: pd.DataFrame | None = None,
) -> None:
    if summary.empty:
        Path(output_path).write_text("# Walk-forward Portfolio Grid Result\n\nNo summary rows generated.\n", encoding="utf-8")
        return

    start_equity = float(summary.iloc[0]["start_equity"])
    final_equity = float(summary.iloc[-1]["final_equity"])
    total_net_pct = (final_equity - start_equity) / start_equity if start_equity else 0.0
    total_moves = int(summary["grid_moves"].sum())
    total_liquidations = int(summary["liquidations"].sum())
    max_dd = float(summary["max_drawdown_pct"].min())
    best_week = summary.sort_values("net_pnl_pct", ascending=False).iloc[0]
    worst_week = summary.sort_values("net_pnl_pct", ascending=True).iloc[0]
    best_params = param_analysis.iloc[0] if not param_analysis.empty else None
    best_combo = combo_report.iloc[0] if combo_report is not None and not combo_report.empty else None
    robust_params = None
    if not param_analysis.empty:
        robust_pool = param_analysis[param_analysis["weeks_used"] >= 2]
        if robust_pool.empty:
            robust_pool = param_analysis
        robust_params = robust_pool.sort_values(
            by=["realized_net_pnl_pct", "max_drawdown_pct", "weeks_used"],
            ascending=[False, False, False],
        ).iloc[0]
    move_summary = (
        moves.groupby(["trading_pair", "direction"]).size().unstack(fill_value=0).reset_index()
        if not moves.empty else pd.DataFrame(columns=["trading_pair", "up", "down"])
    )
    for column in ["up", "down"]:
        if column not in move_summary.columns:
            move_summary[column] = 0
    if not move_summary.empty:
        move_summary["total"] = move_summary["up"] + move_summary["down"]
        move_summary.sort_values("total", ascending=False, inplace=True)

    lines = [
        "# Walk-forward Portfolio Grid Result",
        "",
        "## Overall",
        f"- Period: {summary.iloc[0]['trade_start']} to {summary.iloc[-1]['trade_end']}",
        f"- Start equity: {start_equity:.2f} USDT",
        f"- Final equity: {final_equity:.2f} USDT",
        f"- Net PnL: {final_equity - start_equity:.2f} USDT ({total_net_pct:.2%})",
        f"- Max weekly-window drawdown: {max_dd:.2%}",
        f"- Total grid moves: {total_moves}",
        f"- Total liquidations: {total_liquidations}",
        "",
        "## Best Realized Parameter Combination",
    ]
    if best_params is not None:
        lines.extend([
            f"- grid_range: {best_params['grid_range']}",
            f"- grid_levels: {int(best_params['grid_levels'])}",
            f"- order_quote_pct: {best_params['order_quote_pct']}",
            f"- take_profit: {best_params['take_profit']}",
            f"- move_threshold: {best_params['move_threshold']}",
            f"- stop_loss: {best_params['stop_loss']}",
            f"- weeks_used: {int(best_params['weeks_used'])}",
            f"- realized_net_pnl: {best_params['realized_net_pnl_quote']:.2f} USDT ({best_params['realized_net_pnl_pct']:.2%})",
            f"- mean_weekly_pnl: {best_params['mean_weekly_pnl_pct']:.2%}",
            f"- max_drawdown: {best_params['max_drawdown_pct']:.2%}",
            f"- win_rate: {best_params['win_rate']:.2%}",
            f"- realized_score: {best_params['realized_score']:.6f}",
        ])
    else:
        lines.append("- No parameter analysis rows.")

    lines.extend([
        "",
        "## Best Candidate Combo From Training Windows",
    ])
    if best_combo is not None:
        lines.extend([
            f"- rank: {int(best_combo['rank'])}",
            f"- grid_range: {best_combo['grid_range']}",
            f"- grid_levels: {int(best_combo['grid_levels'])}",
            f"- order_quote_pct: {best_combo['order_quote_pct']}",
            f"- take_profit: {best_combo['take_profit']}",
            f"- move_threshold: {best_combo['move_threshold']}",
            f"- stop_loss: {best_combo['stop_loss']}",
            f"- windows_evaluated: {int(best_combo['windows_evaluated'])}",
            f"- avg_selection_score: {best_combo['avg_selection_score']:.6f}",
            f"- compound_training_pnl: {best_combo['compound_net_pnl_pct']:.2%}",
            f"- avg_training_pnl: {best_combo['avg_net_pnl_pct']:.2%}",
            f"- worst_training_drawdown: {best_combo['worst_max_drawdown_pct']:.2%}",
            f"- win_rate: {best_combo['win_rate']:.2%}",
            f"- liquidation_count: {int(best_combo['liquidation_count'])}",
        ])
    else:
        lines.append("- No candidate combo report rows.")

    lines.extend([
        "",
        "## Robust Recommendation",
    ])
    if robust_params is not None:
        lines.extend([
            "This combination was selected from parameter groups used in at least two trading weeks, then ranked by realized return with drawdown as the tie breaker.",
            f"- grid_range: {robust_params['grid_range']}",
            f"- grid_levels: {int(robust_params['grid_levels'])}",
            f"- order_quote_pct: {robust_params['order_quote_pct']}",
            f"- take_profit: {robust_params['take_profit']}",
            f"- move_threshold: {robust_params['move_threshold']}",
            f"- stop_loss: {robust_params['stop_loss']}",
            f"- weeks_used: {int(robust_params['weeks_used'])}",
            f"- realized_net_pnl: {robust_params['realized_net_pnl_quote']:.2f} USDT ({robust_params['realized_net_pnl_pct']:.2%})",
            f"- mean_weekly_pnl: {robust_params['mean_weekly_pnl_pct']:.2%}",
            f"- max_drawdown: {robust_params['max_drawdown_pct']:.2%}",
            f"- win_rate: {robust_params['win_rate']:.2%}",
        ])
    else:
        lines.append("- No robust recommendation available.")

    lines.extend([
        "",
        "## Best and Worst Weeks",
        f"- Best week: #{int(best_week['week_index'])}, {best_week['trade_start']} to {best_week['trade_end']}, PnL {best_week['net_pnl_pct']:.2%}, moves {int(best_week['grid_moves'])}",
        f"- Worst week: #{int(worst_week['week_index'])}, {worst_week['trade_start']} to {worst_week['trade_end']}, PnL {worst_week['net_pnl_pct']:.2%}, moves {int(worst_week['grid_moves'])}",
        "",
        "## Grid Move Leaders",
    ])
    if move_summary.empty:
        lines.append("- No grid move events.")
    else:
        for row in move_summary.head(10).itertuples(index=False):
            lines.append(f"- {row.trading_pair}: total {int(row.total)}, up {int(row.up)}, down {int(row.down)}")

    Path(output_path).write_text("\n".join(lines) + "\n", encoding="utf-8")


async def main() -> None:
    parser = argparse.ArgumentParser(description="Walk-forward portfolio grid backtest")
    parser.add_argument("--connector", default="binance")
    parser.add_argument("--trading-pairs", default=",".join(DEFAULT_PAIRS))
    parser.add_argument("--pair-limit", type=int, default=None)
    parser.add_argument("--interval", default="5m")
    parser.add_argument("--days", type=float, default=182)
    parser.add_argument("--end-ts", type=int, default=None)
    parser.add_argument("--train-days", type=float, default=30)
    parser.add_argument("--lookback-days", type=float, default=7)
    parser.add_argument("--trade-days", type=float, default=7)
    parser.add_argument("--initial-quote", type=float, default=10000)
    parser.add_argument("--fee-rate", type=float, default=0.0002)
    parser.add_argument("--portfolio-stop-loss", type=float, default=0.08)
    parser.add_argument("--cooldown-hours", type=float, default=24)
    parser.add_argument("--min-grid-move-seconds", type=float, default=0)
    parser.add_argument("--min-candidate-move-threshold", type=float, default=0)
    parser.add_argument("--drawdown-penalty", type=float, default=1.5)
    parser.add_argument("--liquidation-penalty", type=float, default=0.25)
    parser.add_argument("--optimization-workers", type=int, default=1)
    parser.add_argument("--max-candidates", type=int, default=None)
    parser.add_argument("--output-prefix", default="backtest_walk_forward_portfolio_grid")
    args = parser.parse_args()

    trading_pairs = parse_pairs(args.trading_pairs, args.pair_limit)
    end_ts = int(args.end_ts or time.time())
    start_ts = end_ts - int(args.days * SECONDS_PER_DAY)
    train_end_ts = start_ts + int(args.train_days * SECONDS_PER_DAY)
    if train_end_ts >= end_ts:
        raise ValueError("--days must be larger than --train-days")

    print(f"Loading {len(trading_pairs)} pairs from {pd.to_datetime(start_ts, unit='s')} to {pd.to_datetime(end_ts, unit='s')}")
    candles_by_pair = await load_pair_candles_range(args.connector, trading_pairs, args.interval, start_ts, end_ts)
    candidates = default_walk_forward_search_space()
    if args.min_candidate_move_threshold > 0:
        candidates = [
            candidate for candidate in candidates
            if candidate.move_threshold >= args.min_candidate_move_threshold
        ]
    if args.max_candidates:
        candidates = candidates[:args.max_candidates]
    print(f"Evaluating {len(candidates)} parameter candidates per selection window")

    train_candles = slice_candles(candles_by_pair, start_ts, train_end_ts)
    if len(train_candles) != len(trading_pairs):
        raise RuntimeError("Initial training window is missing candles for at least one pair.")
    initial_params, initial_train_result, initial_score, initial_candidate_evaluations = optimize_params_parallel(
        candles_by_pair=train_candles,
        candidates=candidates,
        initial_quote=args.initial_quote,
        fee_rate=args.fee_rate,
        portfolio_stop_loss=args.portfolio_stop_loss,
        cooldown_hours=args.cooldown_hours,
        drawdown_penalty=args.drawdown_penalty,
        liquidation_penalty=args.liquidation_penalty,
        min_grid_move_seconds=args.min_grid_move_seconds,
        workers=args.optimization_workers,
    )
    print(f"Initial params from first month: {initial_params} score={initial_score:.6f}")

    first_trade_window = slice_candles(candles_by_pair, train_end_ts, min(train_end_ts + int(args.trade_days * SECONDS_PER_DAY), end_ts))
    if len(first_trade_window) != len(trading_pairs):
        raise RuntimeError("First trading window is missing candles for at least one pair.")
    first_prices = {pair: float(candles.iloc[0]["close"]) for pair, candles in first_trade_window.items()}
    allocation = args.initial_quote / len(trading_pairs)
    states = {
        pair: initialize_pair_state(pair, allocation, first_prices[pair], initial_params)
        for pair in trading_pairs
    }
    risk_peak_equity = args.initial_quote
    cooldown_until_ts = 0.0
    global_step = 0
    week_index = 1
    week_start = train_end_ts
    summary_rows = []
    all_moves = []
    all_equity = []
    all_candidate_evaluations = []

    while week_start < end_ts:
        week_end = min(week_start + int(args.trade_days * SECONDS_PER_DAY), end_ts)
        lookback_start = max(start_ts, week_start - int(args.lookback_days * SECONDS_PER_DAY))
        if week_index == 1:
            opt_start, opt_end = start_ts, train_end_ts
            params, train_result, train_score = initial_params, initial_train_result, initial_score
            candidate_evaluations = initial_candidate_evaluations.copy()
        else:
            opt_start, opt_end = lookback_start, week_start
            lookback_candles = slice_candles(candles_by_pair, opt_start, opt_end)
            if len(lookback_candles) != len(trading_pairs):
                print(f"Skipping week {week_index}: missing lookback candles")
                week_start = week_end
                continue
            params, train_result, train_score, candidate_evaluations = optimize_params_parallel(
                candles_by_pair=lookback_candles,
                candidates=candidates,
                initial_quote=args.initial_quote,
                fee_rate=args.fee_rate,
                portfolio_stop_loss=args.portfolio_stop_loss,
                cooldown_hours=args.cooldown_hours,
                drawdown_penalty=args.drawdown_penalty,
                liquidation_penalty=args.liquidation_penalty,
                min_grid_move_seconds=args.min_grid_move_seconds,
                workers=args.optimization_workers,
            )
        trade_candles = slice_candles(candles_by_pair, week_start, week_end)
        if len(trade_candles) != len(trading_pairs):
            print(f"Stopping at week {week_index}: missing trading candles")
            break
        candidate_evaluations.insert(0, "week_index", week_index)
        candidate_evaluations.insert(1, "train_start", pd.to_datetime(opt_start, unit="s"))
        candidate_evaluations.insert(2, "train_end", pd.to_datetime(opt_end, unit="s"))
        candidate_evaluations.insert(3, "trade_start", pd.to_datetime(week_start, unit="s"))
        candidate_evaluations.insert(4, "trade_end", pd.to_datetime(week_end, unit="s"))
        all_candidate_evaluations.append(candidate_evaluations)

        current_prices = {pair: float(candles.iloc[0]["close"]) for pair, candles in trade_candles.items()}
        regrid_states(states, current_prices, params)
        result, moves, equity, risk_peak_equity, cooldown_until_ts, global_step = simulate_trading_window(
            trade_candles,
            states,
            params,
            args.fee_rate,
            args.portfolio_stop_loss,
            args.cooldown_hours,
            args.min_grid_move_seconds,
            risk_peak_equity,
            cooldown_until_ts,
            global_step,
            week_index,
        )
        params_row = asdict(params)
        row = {
            "week_index": week_index,
            "train_start": pd.to_datetime(opt_start, unit="s"),
            "train_end": pd.to_datetime(opt_end, unit="s"),
            "trade_start": pd.to_datetime(week_start, unit="s"),
            "trade_end": pd.to_datetime(week_end, unit="s"),
            "selection_score": train_score,
            "selection_net_pnl_pct": train_result["net_pnl_pct"],
            "selection_max_drawdown_pct": train_result["max_drawdown_pct"],
            **result,
            **params_row,
        }
        summary_rows.append(row)
        if not moves.empty:
            moves.insert(0, "week_index", week_index)
            all_moves.append(moves)
        if not equity.empty:
            all_equity.append(equity)
        print(
            f"Week {week_index:02d} {row['trade_start']} -> {row['trade_end']} "
            f"pnl={result['net_pnl_pct']:.2%} dd={result['max_drawdown_pct']:.2%} "
            f"moves={result['grid_moves']} params={params}"
        )
        week_index += 1
        week_start = week_end

    summary = pd.DataFrame(summary_rows)
    moves = pd.concat(all_moves, ignore_index=True) if all_moves else pd.DataFrame()
    equity = pd.concat(all_equity, ignore_index=True) if all_equity else pd.DataFrame()
    candidate_evaluations = (
        pd.concat(all_candidate_evaluations, ignore_index=True) if all_candidate_evaluations else pd.DataFrame()
    )
    prefix = Path(args.output_prefix)
    summary_output = prefix.with_name(prefix.name + "_summary.csv")
    moves_output = prefix.with_name(prefix.name + "_moves.csv")
    equity_output = prefix.with_name(prefix.name + "_equity.csv")
    html_output = prefix.with_name(prefix.name + "_report.html")
    param_analysis_output = prefix.with_name(prefix.name + "_param_analysis.csv")
    candidate_evaluations_output = prefix.with_name(prefix.name + "_candidate_evaluations.csv")
    combo_report_output = prefix.with_name(prefix.name + "_combo_report.csv")
    result_output = prefix.with_name(prefix.name + "_result.md")
    param_analysis = build_param_analysis(summary)
    combo_report = build_combo_report(candidate_evaluations)
    summary.to_csv(summary_output, index=False)
    moves.to_csv(moves_output, index=False)
    equity.to_csv(equity_output, index=False)
    param_analysis.to_csv(param_analysis_output, index=False)
    candidate_evaluations.to_csv(candidate_evaluations_output, index=False)
    combo_report.to_csv(combo_report_output, index=False)
    write_html_report(str(html_output), summary, moves, equity, combo_report)
    write_result_markdown(str(result_output), summary, param_analysis, moves, combo_report)

    print(f"Saved weekly summary to {summary_output}")
    print(f"Saved move events to {moves_output}")
    print(f"Saved equity curve to {equity_output}")
    print(f"Saved parameter analysis to {param_analysis_output}")
    print(f"Saved candidate evaluations to {candidate_evaluations_output}")
    print(f"Saved combo report to {combo_report_output}")
    print(f"Saved result summary to {result_output}")
    print(f"Saved HTML report to {html_output}")
    if not summary.empty:
        start_equity = float(summary.iloc[0]["start_equity"])
        final_equity = float(summary.iloc[-1]["final_equity"])
        print(f"Final equity: {final_equity:.2f} USDT ({(final_equity - start_equity) / start_equity:.2%})")
        print(f"Max weekly drawdown: {summary['max_drawdown_pct'].min():.2%}")
        print(f"Total grid moves: {int(summary['grid_moves'].sum())}")
        print(f"Total liquidations: {int(summary['liquidations'].sum())}")


if __name__ == "__main__":
    asyncio.run(main())

#!/usr/bin/env python
"""
Staged optimizer for the walk-forward portfolio grid strategy.

Stage 1 evaluates a compact fixed-parameter search space. Stage 2 expands the
best fixed candidates across move cooldown and portfolio stop-loss settings.
Stage 3 runs walk-forward validation using the top fixed candidates only.
"""

import argparse
import asyncio
import concurrent.futures
import itertools
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.backtest_3commas_grid import GridParams  # noqa: E402
from scripts.backtest_portfolio_3commas_grid import DEFAULT_PAIRS, load_pair_candles, simulate_portfolio  # noqa: E402
from scripts.backtest_walk_forward_portfolio_grid import (  # noqa: E402
    build_combo_report,
    build_param_analysis,
    optimize_params_parallel,
    regrid_states,
    simulate_trading_window,
    slice_candles,
    write_html_report,
    write_result_markdown,
)
from scripts.backtest_portfolio_3commas_grid import initialize_pair_state  # noqa: E402


SECONDS_PER_DAY = 24 * 60 * 60


def parse_pairs(raw_pairs: str, limit: int | None = None) -> List[str]:
    pairs = [pair.strip().upper() for pair in raw_pairs.split(",") if pair.strip()]
    return pairs[:limit] if limit else pairs


def stage_one_candidates() -> List[dict]:
    rows = []
    for values in itertools.product(
        [0.06, 0.08, 0.10],
        [8, 12, 16],
        [0.005, 0.01, 0.015],
        [0.005, 0.008, 0.01],
        [0.005, 0.01, 0.015],
    ):
        rows.append({
            "grid_range": values[0],
            "grid_levels": values[1],
            "order_quote_pct": values[2],
            "take_profit": values[3],
            "move_threshold": values[4],
            "stop_loss": 0.04,
            "min_grid_move_seconds": 0.0,
            "portfolio_stop_loss": 0.08,
            "stage": 1,
        })
    return rows


def params_from_row(row: dict) -> GridParams:
    return GridParams(
        float(row["grid_range"]),
        int(row["grid_levels"]),
        float(row["order_quote_pct"]),
        float(row["take_profit"]),
        float(row["move_threshold"]),
        float(row.get("stop_loss", 0.04)),
    )


def fixed_score(result: Dict[str, float]) -> float:
    return float(result["net_pnl_pct"])


def evaluate_fixed_candidate(args: Tuple[Dict[str, pd.DataFrame], dict, float, float, float]) -> dict:
    candles_by_pair, candidate, initial_quote, fee_rate, cooldown_hours = args
    params = params_from_row(candidate)
    result, _, _ = simulate_portfolio(
        candles_by_pair=candles_by_pair,
        params=params,
        initial_quote=initial_quote,
        fee_rate=fee_rate,
        portfolio_stop_loss=float(candidate["portfolio_stop_loss"]),
        cooldown_hours=cooldown_hours,
        min_grid_move_seconds=float(candidate["min_grid_move_seconds"]),
    )
    return {
        **candidate,
        **result,
        "score": fixed_score(result),
    }


def evaluate_fixed_candidates(
    candles_by_pair: Dict[str, pd.DataFrame],
    candidates: Iterable[dict],
    initial_quote: float,
    fee_rate: float,
    cooldown_hours: float,
    workers: int,
) -> pd.DataFrame:
    candidate_list = list(candidates)
    task_args = [(candles_by_pair, candidate, initial_quote, fee_rate, cooldown_hours) for candidate in candidate_list]
    if workers <= 1:
        rows = [evaluate_fixed_candidate(args) for args in task_args]
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            rows = list(executor.map(evaluate_fixed_candidate, task_args))
    df = pd.DataFrame(rows)
    df.sort_values(
        by=["score", "net_pnl_pct", "max_drawdown_pct", "liquidated", "trades"],
        ascending=[False, False, False, True, False],
        inplace=True,
    )
    df.reset_index(drop=True, inplace=True)
    df.insert(0, "rank", range(1, len(df) + 1))
    return df


def expand_top_candidates(stage_one: pd.DataFrame, top_n: int) -> List[dict]:
    expanded = []
    seen = set()
    param_cols = ["grid_range", "grid_levels", "order_quote_pct", "take_profit", "move_threshold", "stop_loss"]
    for _, row in stage_one.head(top_n).iterrows():
        for min_move_seconds, portfolio_stop_loss in itertools.product([0.0, 900.0, 1800.0], [0.08, 0.10, 0.12]):
            candidate = {column: row[column] for column in param_cols}
            candidate.update({
                "min_grid_move_seconds": min_move_seconds,
                "portfolio_stop_loss": portfolio_stop_loss,
                "stage": 2,
            })
            key = tuple(candidate[column] for column in [
                "grid_range", "grid_levels", "order_quote_pct", "take_profit", "move_threshold",
                "stop_loss", "min_grid_move_seconds", "portfolio_stop_loss",
            ])
            if key not in seen:
                seen.add(key)
                expanded.append(candidate)
    return expanded


async def run_top_candidate_walk_forward(
    candles_by_pair: Dict[str, pd.DataFrame],
    candidates: List[dict],
    trading_pairs: List[str],
    start_ts: int,
    end_ts: int,
    train_days: float,
    lookback_days: float,
    trade_days: float,
    initial_quote: float,
    fee_rate: float,
    cooldown_hours: float,
    drawdown_penalty: float,
    liquidation_penalty: float,
    workers: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_end_ts = start_ts + int(train_days * SECONDS_PER_DAY)
    candidate_params = [params_from_row(row) for row in candidates]
    candidate_settings = {
        (
            row["grid_range"], row["grid_levels"], row["order_quote_pct"], row["take_profit"],
            row["move_threshold"], row["stop_loss"],
        ): row
        for row in candidates
    }

    train_candles = slice_candles(candles_by_pair, start_ts, train_end_ts)
    initial_params, initial_train_result, initial_score, initial_candidate_evaluations = optimize_params_parallel(
        candles_by_pair=train_candles,
        candidates=candidate_params,
        initial_quote=initial_quote,
        fee_rate=fee_rate,
        portfolio_stop_loss=0.08,
        cooldown_hours=cooldown_hours,
        drawdown_penalty=drawdown_penalty,
        liquidation_penalty=liquidation_penalty,
        min_grid_move_seconds=0.0,
        workers=workers,
    )
    first_window = slice_candles(candles_by_pair, train_end_ts, min(train_end_ts + int(trade_days * SECONDS_PER_DAY), end_ts))
    first_prices = {pair: float(candles.iloc[0]["close"]) for pair, candles in first_window.items()}
    allocation = initial_quote / len(trading_pairs)
    states = {
        pair: initialize_pair_state(pair, allocation, first_prices[pair], initial_params)
        for pair in trading_pairs
    }
    risk_peak_equity = initial_quote
    cooldown_until_ts = 0.0
    global_step = 0
    week_index = 1
    week_start = train_end_ts
    summary_rows = []
    all_moves = []
    all_equity = []
    all_candidate_evaluations = []

    while week_start < end_ts:
        week_end = min(week_start + int(trade_days * SECONDS_PER_DAY), end_ts)
        lookback_start = max(start_ts, week_start - int(lookback_days * SECONDS_PER_DAY))
        if week_index == 1:
            opt_start, opt_end = start_ts, train_end_ts
            params = initial_params
            train_result = initial_train_result
            train_score = initial_score
            candidate_evaluations = initial_candidate_evaluations.copy()
        else:
            opt_start, opt_end = lookback_start, week_start
            lookback_candles = slice_candles(candles_by_pair, opt_start, opt_end)
            params, train_result, train_score, candidate_evaluations = optimize_params_parallel(
                candles_by_pair=lookback_candles,
                candidates=candidate_params,
                initial_quote=initial_quote,
                fee_rate=fee_rate,
                portfolio_stop_loss=0.08,
                cooldown_hours=cooldown_hours,
                drawdown_penalty=drawdown_penalty,
                liquidation_penalty=liquidation_penalty,
                min_grid_move_seconds=0.0,
                workers=workers,
            )
        key = (
            params.grid_range, params.grid_levels, params.order_quote_pct, params.take_profit,
            params.move_threshold, params.stop_loss,
        )
        selected = candidate_settings.get(key, {})
        selected_min_move = float(selected.get("min_grid_move_seconds", 0.0))
        selected_stop = float(selected.get("portfolio_stop_loss", 0.08))

        trade_candles = slice_candles(candles_by_pair, week_start, week_end)
        current_prices = {pair: float(candles.iloc[0]["close"]) for pair, candles in trade_candles.items()}
        regrid_states(states, current_prices, params)
        result, moves, equity, risk_peak_equity, cooldown_until_ts, global_step = simulate_trading_window(
            trade_candles,
            states,
            params,
            fee_rate,
            selected_stop,
            cooldown_hours,
            selected_min_move,
            risk_peak_equity,
            cooldown_until_ts,
            global_step,
            week_index,
        )

        candidate_evaluations.insert(0, "week_index", week_index)
        candidate_evaluations.insert(1, "train_start", pd.to_datetime(opt_start, unit="s"))
        candidate_evaluations.insert(2, "train_end", pd.to_datetime(opt_end, unit="s"))
        candidate_evaluations.insert(3, "trade_start", pd.to_datetime(week_start, unit="s"))
        candidate_evaluations.insert(4, "trade_end", pd.to_datetime(week_end, unit="s"))
        all_candidate_evaluations.append(candidate_evaluations)

        summary_rows.append({
            "week_index": week_index,
            "train_start": pd.to_datetime(opt_start, unit="s"),
            "train_end": pd.to_datetime(opt_end, unit="s"),
            "trade_start": pd.to_datetime(week_start, unit="s"),
            "trade_end": pd.to_datetime(week_end, unit="s"),
            "selection_score": train_score,
            "selection_net_pnl_pct": train_result["net_pnl_pct"],
            "selection_max_drawdown_pct": train_result["max_drawdown_pct"],
            **result,
            **asdict(params),
            "min_grid_move_seconds": selected_min_move,
            "selected_portfolio_stop_loss": selected_stop,
        })
        if not moves.empty:
            moves.insert(0, "week_index", week_index)
            all_moves.append(moves)
        if not equity.empty:
            all_equity.append(equity)
        week_index += 1
        week_start = week_end

    summary = pd.DataFrame(summary_rows)
    moves = pd.concat(all_moves, ignore_index=True) if all_moves else pd.DataFrame()
    equity = pd.concat(all_equity, ignore_index=True) if all_equity else pd.DataFrame()
    candidate_evaluations = (
        pd.concat(all_candidate_evaluations, ignore_index=True) if all_candidate_evaluations else pd.DataFrame()
    )
    param_analysis = build_param_analysis(summary)
    combo_report = build_combo_report(candidate_evaluations)
    return summary, moves, equity, param_analysis, combo_report, candidate_evaluations


def write_fixed_result(path: str, fixed: pd.DataFrame, expanded: pd.DataFrame, wf_summary: pd.DataFrame) -> None:
    best = expanded.iloc[0]
    lines = [
        "# Staged Portfolio Grid Optimization",
        "",
        "## Fixed Search Best",
        f"- final_equity: {best['final_equity']:.2f} USDT",
        f"- net_pnl: {best['net_pnl_quote']:.2f} USDT ({best['net_pnl_pct']:.2%})",
        f"- max_drawdown: {best['max_drawdown_pct']:.2%}",
        f"- liquidated: {bool(best['liquidated'])}",
        f"- trades: {int(best['trades'])}",
        f"- grid_moves: {int(best['grid_moves'])}",
        f"- grid_range: {best['grid_range']}",
        f"- grid_levels: {int(best['grid_levels'])}",
        f"- order_quote_pct: {best['order_quote_pct']}",
        f"- take_profit: {best['take_profit']}",
        f"- move_threshold: {best['move_threshold']}",
        f"- min_grid_move_seconds: {best['min_grid_move_seconds']}",
        f"- portfolio_stop_loss: {best['portfolio_stop_loss']}",
        "",
        "## Search Counts",
        f"- stage_one_candidates: {len(fixed)}",
        f"- expanded_candidates: {len(expanded)}",
    ]
    if not wf_summary.empty:
        start = float(wf_summary.iloc[0]["start_equity"])
        final = float(wf_summary.iloc[-1]["final_equity"])
        lines.extend([
            "",
            "## Walk-forward Top Candidate Validation",
            f"- final_equity: {final:.2f} USDT",
            f"- net_pnl: {final - start:.2f} USDT ({(final - start) / start:.2%})",
            f"- max_drawdown: {wf_summary['max_drawdown_pct'].min():.2%}",
            f"- liquidations: {int(wf_summary['liquidations'].sum())}",
            f"- grid_moves: {int(wf_summary['grid_moves'].sum())}",
        ])
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


async def main() -> None:
    parser = argparse.ArgumentParser(description="Staged portfolio grid optimizer")
    parser.add_argument("--connector", default="binance")
    parser.add_argument("--trading-pairs", default=",".join(DEFAULT_PAIRS))
    parser.add_argument("--pair-limit", type=int, default=10)
    parser.add_argument("--interval", default="5m")
    parser.add_argument("--days", type=float, default=182)
    parser.add_argument("--end-ts", type=int, default=1783566900)
    parser.add_argument("--initial-quote", type=float, default=10000)
    parser.add_argument("--fee-rate", type=float, default=0.0002)
    parser.add_argument("--cooldown-hours", type=float, default=24)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-stage-one-candidates", type=int, default=None)
    parser.add_argument("--top-stage-one", type=int, default=20)
    parser.add_argument("--top-walk-forward", type=int, default=10)
    parser.add_argument("--train-days", type=float, default=30)
    parser.add_argument("--lookback-days", type=float, default=7)
    parser.add_argument("--trade-days", type=float, default=7)
    parser.add_argument("--drawdown-penalty", type=float, default=1.5)
    parser.add_argument("--liquidation-penalty", type=float, default=0.25)
    parser.add_argument("--output-prefix", default="results/backtests/optimize_fixed_top10_6m_fee002")
    parser.add_argument("--walk-forward-prefix", default="results/backtests/walk_forward_top_candidates_top10_6m_fee002")
    args = parser.parse_args()

    trading_pairs = parse_pairs(args.trading_pairs, args.pair_limit)
    print(f"Loading {len(trading_pairs)} pairs, days={args.days}, end_ts={args.end_ts}")
    candles_by_pair = await load_pair_candles(args.connector, trading_pairs, args.interval, args.days, args.end_ts)

    first_stage_candidates = stage_one_candidates()
    if args.max_stage_one_candidates:
        first_stage_candidates = first_stage_candidates[:args.max_stage_one_candidates]
    stage_one = evaluate_fixed_candidates(
        candles_by_pair,
        first_stage_candidates,
        args.initial_quote,
        args.fee_rate,
        args.cooldown_hours,
        args.workers,
    )
    expanded_candidates = expand_top_candidates(stage_one, args.top_stage_one)
    expanded = evaluate_fixed_candidates(
        candles_by_pair,
        expanded_candidates,
        args.initial_quote,
        args.fee_rate,
        args.cooldown_hours,
        args.workers,
    )

    prefix = Path(args.output_prefix)
    candidates_output = prefix.with_name(prefix.name + "_candidates.csv")
    top_output = prefix.with_name(prefix.name + "_top.csv")
    result_output = prefix.with_name(prefix.name + "_result.md")
    stage_one.to_csv(candidates_output, index=False)
    expanded.to_csv(top_output, index=False)

    start_ts = int(args.end_ts - args.days * SECONDS_PER_DAY)
    top_candidates = (
        expanded.drop_duplicates(
            subset=["grid_range", "grid_levels", "order_quote_pct", "take_profit", "move_threshold", "stop_loss"],
            keep="first",
        )
        .head(args.top_walk_forward)
        .to_dict("records")
    )
    wf_summary, wf_moves, wf_equity, wf_param_analysis, wf_combo_report, wf_candidate_evaluations = (
        await run_top_candidate_walk_forward(
            candles_by_pair,
            top_candidates,
            trading_pairs,
            start_ts,
            args.end_ts,
            args.train_days,
            args.lookback_days,
            args.trade_days,
            args.initial_quote,
            args.fee_rate,
            args.cooldown_hours,
            args.drawdown_penalty,
            args.liquidation_penalty,
            args.workers,
        )
    )

    wf_prefix = Path(args.walk_forward_prefix)
    wf_summary_output = wf_prefix.with_name(wf_prefix.name + "_summary.csv")
    wf_moves_output = wf_prefix.with_name(wf_prefix.name + "_moves.csv")
    wf_equity_output = wf_prefix.with_name(wf_prefix.name + "_equity.csv")
    wf_candidate_output = wf_prefix.with_name(wf_prefix.name + "_candidate_evaluations.csv")
    wf_combo_output = wf_prefix.with_name(wf_prefix.name + "_combo_report.csv")
    wf_report_output = wf_prefix.with_name(wf_prefix.name + "_report.html")
    wf_result_output = wf_prefix.with_name(wf_prefix.name + "_result.md")
    wf_param_output = wf_prefix.with_name(wf_prefix.name + "_param_analysis.csv")

    wf_summary.to_csv(wf_summary_output, index=False)
    wf_moves.to_csv(wf_moves_output, index=False)
    wf_equity.to_csv(wf_equity_output, index=False)
    wf_candidate_evaluations.to_csv(wf_candidate_output, index=False)
    wf_combo_report.to_csv(wf_combo_output, index=False)
    wf_param_analysis.to_csv(wf_param_output, index=False)
    write_html_report(str(wf_report_output), wf_summary, wf_moves, wf_equity, wf_combo_report)
    write_result_markdown(str(wf_result_output), wf_summary, wf_param_analysis, wf_moves, wf_combo_report)
    write_fixed_result(str(result_output), stage_one, expanded, wf_summary)

    print(f"Saved fixed candidates to {candidates_output}")
    print(f"Saved expanded top candidates to {top_output}")
    print(f"Saved fixed result summary to {result_output}")
    print(f"Saved walk-forward summary to {wf_summary_output}")
    print(f"Saved walk-forward candidate evaluations to {wf_candidate_output}")
    print(f"Saved walk-forward combo report to {wf_combo_output}")
    print(f"Saved walk-forward report to {wf_report_output}")
    print(f"Saved walk-forward result to {wf_result_output}")


if __name__ == "__main__":
    asyncio.run(main())

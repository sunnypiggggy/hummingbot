"""Shared walk-forward selection helpers for the BTC/ETH FDUSD live grid."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timedelta
from typing import Dict, Iterable
from zoneinfo import ZoneInfo

import pandas as pd

from validate_grid_live import (
    Candidate,
    InventoryExitPolicy,
    candidates,
    simulate,
    slice_window,
)


SHANGHAI = ZoneInfo("Asia/Shanghai")
DAY_SECONDS = 86400
WEEK_SECONDS = 7 * DAY_SECONDS
INITIAL_LOOKBACK_DAYS = 30
WEEKLY_LOOKBACK_DAYS = 14
SCORE_DRAWDOWN_PENALTY = 1.5


def local_day_start(timestamp: int) -> int:
    local = datetime.fromtimestamp(timestamp, SHANGHAI)
    return int(local.replace(hour=0, minute=0, second=0, microsecond=0).timestamp())


def weekly_cutoff(now_ts: int) -> tuple[int, int]:
    """Return the latest eligible Monday 00:00 cutoff and its 00:10 run time."""
    local = datetime.fromtimestamp(now_ts, SHANGHAI)
    monday = (local - timedelta(days=local.weekday())).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    run_at = monday + timedelta(minutes=10)
    if local < run_at:
        monday -= timedelta(days=7)
        run_at -= timedelta(days=7)
    return int(monday.timestamp()), int(run_at.timestamp())


def rolling_validation_windows(start_ts: int, end_ts: int) -> list[tuple[int, int, int, int]]:
    """Create a 30-day first train fold followed by 14-day weekly walk-forward folds."""
    start = local_day_start(start_ts)
    end = local_day_start(end_ts)
    first_test_start = start + INITIAL_LOOKBACK_DAYS * DAY_SECONDS
    windows: list[tuple[int, int, int, int]] = []
    test_start = first_test_start
    fold = 0
    while test_start + WEEK_SECONDS <= end:
        lookback_days = INITIAL_LOOKBACK_DAYS if fold == 0 else WEEKLY_LOOKBACK_DAYS
        train_start = test_start - lookback_days * DAY_SECONDS
        windows.append((train_start, test_start, test_start, test_start + WEEK_SECONDS))
        test_start += WEEK_SECONDS
        fold += 1
    return windows


def score_result(result: dict) -> float:
    return float(result["net_pnl_pct"]) - SCORE_DRAWDOWN_PENALTY * abs(
        float(result["max_drawdown_pct"])
    )


def pair_risk_triggered(pair_stats: Dict[str, dict]) -> bool:
    return any(int(metrics.get("liquidations", 0)) > 0 for metrics in pair_stats.values())


def select_candidate(candles: Dict[str, pd.DataFrame], maker_fee: float,
                     taker_fee: float | None = None,
                     candidate_pool: Iterable[Candidate] | None = None,
                     require_eligible: bool = True,
                     technical_buy_gate: Dict[int, bool] | None = None,
                     cost_floor_enabled: bool = True,
                     inventory_exit_policy: InventoryExitPolicy | None = None,
                     ) -> tuple[Candidate, pd.DataFrame]:
    rows = []
    ranked = []
    for candidate in candidate_pool or candidates():
        result, _, per_pair = simulate(
            candles, candidate, maker_fee, taker_fee=taker_fee,
            technical_buy_gate=technical_buy_gate,
            cost_floor_enabled=cost_floor_enabled,
            inventory_exit_policy=inventory_exit_policy,
        )
        pair_stop = pair_risk_triggered(per_pair)
        eligible = not bool(result["liquidated"]) and not pair_stop
        score = score_result(result)
        row = {
            **asdict(candidate),
            "levels": candidate.levels,
            "score": score,
            "eligible": eligible,
            "portfolio_liquidated": bool(result["liquidated"]),
            "pair_stop_triggered": pair_stop,
            **result,
        }
        rows.append(row)
        ranked.append((eligible, score, candidate))
    eligible = [item for item in ranked if item[0]]
    if not eligible:
        if require_eligible:
            raise RuntimeError("All 81 FDUSD grid candidates triggered a risk limit.")
        selected = max(ranked, key=lambda item: item[1])[2]
    else:
        selected = max(eligible, key=lambda item: item[1])[2]
    frame = pd.DataFrame(rows)
    frame.attrs["eligible_count"] = len(eligible)
    return selected, frame


def run_walk_forward(candles: Dict[str, pd.DataFrame], maker_fee: float,
                     start_ts: int, end_ts: int,
                     taker_fee: float | None = None,
                     technical_buy_gate: Dict[int, bool] | None = None,
                     cost_floor_enabled: bool = True,
                     inventory_exit_policy: InventoryExitPolicy | None = None,
                     ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    evaluations = []
    summaries = []
    pair_rows = []
    folds = rolling_validation_windows(start_ts, end_ts)
    if not folds:
        raise RuntimeError("The validation range is too short for a 30-day train and 7-day test fold.")
    for fold, (train_start, train_end, test_start, test_end) in enumerate(folds, 1):
        training = slice_window(candles, train_start, train_end)
        testing = slice_window(candles, test_start, test_end)
        selected, candidate_rows = select_candidate(
            training,
            maker_fee,
            taker_fee=taker_fee,
            require_eligible=False,
            technical_buy_gate=technical_buy_gate,
            cost_floor_enabled=cost_floor_enabled,
            inventory_exit_policy=inventory_exit_policy,
        )
        eligible_count = int(candidate_rows.attrs.get("eligible_count", 0))
        candidate_rows.insert(0, "fold", fold)
        candidate_rows.insert(1, "train_start", train_start)
        candidate_rows.insert(2, "train_end", train_end)
        evaluations.append(candidate_rows)
        result, _, pair_stats = simulate(
            testing, selected, maker_fee, taker_fee=taker_fee,
            technical_buy_gate=technical_buy_gate,
            cost_floor_enabled=cost_floor_enabled,
            inventory_exit_policy=inventory_exit_policy,
        )
        summaries.append({
            "fold": fold,
            "train_start": train_start,
            "train_end": train_end,
            "test_start": test_start,
            "test_end": test_end,
            **asdict(selected),
            "levels": selected.levels,
            "training_eligible_candidates": eligible_count,
            "diagnostic_fallback": eligible_count == 0,
            **result,
        })
        for pair, metrics in pair_stats.items():
            pair_rows.append({"fold": fold, "pair": pair, **metrics})
    return pd.concat(evaluations, ignore_index=True), pd.DataFrame(summaries), pd.DataFrame(pair_rows)

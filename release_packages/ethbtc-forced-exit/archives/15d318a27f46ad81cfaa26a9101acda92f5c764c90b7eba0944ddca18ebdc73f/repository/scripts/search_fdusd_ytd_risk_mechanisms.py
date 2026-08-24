#!/usr/bin/env python3
"""Search FDUSD Grid risk mechanisms 1-3 independently over 2026 YTD."""

from __future__ import annotations

import argparse
import html
import json
import math
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Mapping

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from validate_grid_live import (
    Candidate,
    PairBreakerPolicy,
    load_candles,
    simulate,
)
from grid_technical_gate import roc_sqz_signal_from_klines


DAY_SECONDS = 86_400
BAR_SECONDS = 300
WARMUP_DAYS = 8
PAIRS = ("BTC-FDUSD", "ETH-FDUSD")
PAIR_COLORS = {"BTC-FDUSD": "#0891B2", "ETH-FDUSD": "#7C3AED"}
SCENARIO_COLORS = {
    "baseline": "#64748B",
    "mechanism_1": "#2563EB",
    "mechanism_2": "#D97706",
    "mechanism_3": "#7C3AED",
}
ROC_TRIGGER_THRESHOLDS = (-4.0, -5.0, -6.0, -7.0, -8.0, -9.0, -10.0, -12.0)
SQZ_TRIGGER_THRESHOLDS = (-1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -8.0)
ROC_RECOVERY_THRESHOLDS = tuple(float(value) for value in range(-6, 7))
SQZ_RECOVERY_THRESHOLDS = (-3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0)
COOLDOWN_DAYS = tuple(range(1, 8))
RESET_MODES = (False, True)
BASE_CANDIDATE = Candidate(0.03, 0.006, 0.006, 0.015, 1800)
_WINDOW_CACHE: dict[tuple[int, int, int], pd.DataFrame] = {}
_SIGNAL_CACHE: dict[int, tuple[list[int], list[dict]]] = {}


@dataclass(frozen=True)
class TechnicalParameters:
    roc_trigger_pct: float
    sqz_trigger_pct: float
    roc_recovery_pct: float
    sqz_recovery_pct: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2026-01-01T00:00:00+00:00")
    parser.add_argument("--end-ts", type=int)
    parser.add_argument("--cache-dir", type=Path, default=Path("data/backtesting_candles"))
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("results/backtests/fdusd_ytd_risk_mechanisms_1_3"),
    )
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--top-trigger-count", type=int, default=10)
    parser.add_argument(
        "--reuse-technical-dir", type=Path,
        help="Reuse a completed mechanism-1 search with the exact same period.",
    )
    return parser.parse_args()


def iso_timestamp(value: str) -> int:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        raise ValueError("start must include a timezone")
    return int(parsed.astimezone(timezone.utc).timestamp())


def latest_complete_bar(now: float | None = None) -> int:
    observed = time.time() if now is None else now
    return math.floor(observed / BAR_SECONDS) * BAR_SECONDS


def data_quality(frame: pd.DataFrame, pair: str, start_ts: int, split_ts: int,
                 end_ts: int, warmup_start: int) -> dict:
    evaluation = frame[(frame.timestamp >= start_ts) & (frame.timestamp < end_ts)]
    expected = (end_ts - start_ts) // BAR_SECONDS
    timestamps = evaluation.timestamp.astype("int64")
    duplicate_count = int(timestamps.duplicated().sum())
    actual_set = set(timestamps.tolist())
    missing_count = sum(
        timestamp not in actual_set for timestamp in range(start_ts, end_ts, BAR_SECONDS)
    )
    warmup_rows = int(((frame.timestamp >= warmup_start) & (frame.timestamp < start_ts)).sum())
    return {
        "pair": pair,
        "warmup_start": warmup_start,
        "evaluation_start": start_ts,
        "split_ts": split_ts,
        "evaluation_end": end_ts,
        "first_timestamp": int(frame.timestamp.min()),
        "last_timestamp": int(frame.timestamp.max()),
        "expected_rows": expected,
        "actual_rows": len(evaluation),
        "coverage_pct": len(evaluation) / expected * 100,
        "missing_rows": missing_count,
        "duplicate_rows": duplicate_count,
        "warmup_rows": warmup_rows,
        "warmup_complete": warmup_rows >= WARMUP_DAYS * DAY_SECONDS // BAR_SECONDS * 0.98,
    }


def slice_pair(frame: pd.DataFrame, start_ts: int, end_ts: int) -> pd.DataFrame:
    key = (id(frame), start_ts, end_ts)
    cached = _WINDOW_CACHE.get(key)
    if cached is None:
        cached = frame[
            (frame.timestamp >= start_ts) & (frame.timestamp < end_ts)
        ].reset_index(drop=True)
        _WINDOW_CACHE[key] = cached
    return cached


def gate_pause_windows(timeline: Mapping[int, bool], start_ts: int, end_ts: int) -> int:
    previous = True
    windows = 0
    for timestamp, enabled in sorted(timeline.items()):
        if timestamp < start_ts or timestamp >= end_ts:
            continue
        if previous and not enabled:
            windows += 1
        previous = enabled
    return windows


def technical_timeline(frame: pd.DataFrame, params: TechnicalParameters) -> dict[int, bool]:
    timestamps, observations = technical_observations(frame)
    active = False
    transitions: list[tuple[int, bool]] = []
    for observation in observations:
        roc = float(observation["roc_48h_pct"])
        sqz = float(observation["sqzmom_pct"])
        adverse = roc <= params.roc_trigger_pct and sqz <= params.sqz_trigger_pct
        improving = float(observation["sqzmom"]) > float(observation["sqzmom_previous"])
        if not active and adverse:
            active = True
        elif (
            active and not adverse and improving
            and roc >= params.roc_recovery_pct
            and sqz >= params.sqz_recovery_pct
        ):
            active = False
        transitions.append((int(observation["timestamp"]), not active))
    timeline: dict[int, bool] = {}
    pointer = 0
    enabled = False
    for timestamp in timestamps:
        while pointer < len(transitions) and transitions[pointer][0] <= timestamp:
            enabled = transitions[pointer][1]
            pointer += 1
        timeline[timestamp] = enabled
    return timeline


def technical_observations(frame: pd.DataFrame) -> tuple[list[int], list[dict]]:
    cached = _SIGNAL_CACHE.get(id(frame))
    if cached is not None:
        return cached
    source = frame.sort_values("timestamp")
    source = source.assign(bucket=(source.timestamp.astype("int64") // 14_400) * 14_400)
    bars = source.groupby("bucket", sort=True).agg(
        open=("open", "first"), high=("high", "max"), low=("low", "min"),
        close=("close", "last"), rows=("close", "size"),
    ).reset_index()
    bars = bars[bars.rows == 48]
    klines: list[list] = []
    observations: list[dict] = []
    for row in bars.itertuples(index=False):
        close_time = int(row.bucket) + 14_400 - 1
        klines.append([
            int(row.bucket) * 1000, row.open, row.high, row.low, row.close, 0,
            close_time * 1000,
        ])
        if len(klines) < 40:
            continue
        signal = roc_sqz_signal_from_klines(klines[-64:])
        observations.append({"timestamp": close_time + 1, **signal})
    result = (source.timestamp.astype("int64").tolist(), observations)
    _SIGNAL_CACHE[id(frame)] = result
    return result


def evaluate_pair(
    frame: pd.DataFrame,
    pair: str,
    start_ts: int,
    end_ts: int,
    *,
    gate: dict[int, bool] | None = None,
    policy: PairBreakerPolicy | None = None,
    record_curve: bool = False,
    trade_log: list[dict] | None = None,
) -> tuple[dict, pd.DataFrame]:
    result, curve, pair_stats = simulate(
        {pair: slice_pair(frame, start_ts, end_ts)},
        BASE_CANDIDATE,
        maker_fee=0.0,
        taker_fee=0.001,
        order_refresh_seconds=7200,
        technical_buy_gate={pair: gate} if gate is not None else None,
        trade_log=trade_log,
        risk_breakers_enabled=False,
        cost_floor_enabled=False,
        inventory_exit_policy=None,
        pair_breaker_policy=policy,
        record_curve=record_curve,
    )
    stats = pair_stats[pair]
    metrics = {
        "net_pnl_quote": float(stats["net_pnl_quote"]),
        "return_pct": float(stats["net_pnl_quote"]) / 200 * 100,
        "max_drawdown_pct": float(stats["max_drawdown_pct"]) * 100,
        "pause_hours": float(
            stats["technical_risk_off_seconds"] if gate is not None else stats["halted_seconds"]
        ) / 3600,
        "trades": int(stats["buys"] + stats["sells"]),
        "trigger_count": int(stats["liquidations"]),
        "fees_quote": float(stats["fees_quote"]),
        "original_final_pnl_quote": float(stats["net_pnl_quote"]),
        "loss_reference_equity": float(stats["loss_reference_equity"]),
    }
    return metrics, curve


def add_percentile_score(rows: pd.DataFrame) -> pd.DataFrame:
    ranked = rows.copy()
    ranked["return_percentile"] = ranked.return_pct.rank(method="average", pct=True)
    ranked["drawdown_percentile"] = ranked.max_drawdown_pct.rank(method="average", pct=True)
    ranked["pause_percentile"] = ranked.pause_hours.rank(
        method="average", pct=True, ascending=False,
    )
    ranked["score"] = (
        ranked.return_percentile * 0.50
        + ranked.drawdown_percentile * 0.30
        + ranked.pause_percentile * 0.20
    )
    return ranked


def search_technical_pair(frame: pd.DataFrame, pair: str, start_ts: int, split_ts: int,
                          top_count: int) -> tuple[pd.DataFrame, pd.DataFrame, TechnicalParameters]:
    stage_one_rows = []
    for roc_trigger in ROC_TRIGGER_THRESHOLDS:
        for sqz_trigger in SQZ_TRIGGER_THRESHOLDS:
            params = TechnicalParameters(roc_trigger, sqz_trigger, 1.0, -3.0)
            timeline = technical_timeline(frame, params)
            metrics, _ = evaluate_pair(frame, pair, start_ts, split_ts, gate=timeline)
            stage_one_rows.append({
                "pair": pair, **asdict(params),
                "pause_windows": gate_pause_windows(timeline, start_ts, split_ts),
                **metrics,
            })
    stage_one = add_percentile_score(pd.DataFrame(stage_one_rows))
    stage_one["eligible"] = stage_one.pause_windows >= 2
    eligible = stage_one[stage_one.eligible].sort_values(
        ["score", "return_pct", "max_drawdown_pct"], ascending=False,
    )
    if eligible.empty:
        raise RuntimeError(f"{pair} has no eligible technical trigger parameter")
    top_triggers = eligible.head(top_count)

    stage_two_rows = []
    for trigger in top_triggers.itertuples(index=False):
        for roc_recovery in ROC_RECOVERY_THRESHOLDS:
            for sqz_recovery in SQZ_RECOVERY_THRESHOLDS:
                params = TechnicalParameters(
                    trigger.roc_trigger_pct,
                    trigger.sqz_trigger_pct,
                    roc_recovery,
                    sqz_recovery,
                )
                timeline = technical_timeline(frame, params)
                metrics, _ = evaluate_pair(frame, pair, start_ts, split_ts, gate=timeline)
                stage_two_rows.append({
                    "pair": pair, **asdict(params),
                    "pause_windows": gate_pause_windows(timeline, start_ts, split_ts),
                    **metrics,
                })
    stage_two = add_percentile_score(pd.DataFrame(stage_two_rows))
    stage_two["eligible"] = stage_two.pause_windows >= 2
    ranked = stage_two[stage_two.eligible].sort_values(
        ["score", "return_pct", "max_drawdown_pct"], ascending=False,
    )
    if ranked.empty:
        raise RuntimeError(f"{pair} has no eligible technical recovery parameter")
    winner = ranked.iloc[0]
    selected = TechnicalParameters(
        float(winner.roc_trigger_pct), float(winner.sqz_trigger_pct),
        float(winner.roc_recovery_pct), float(winner.sqz_recovery_pct),
    )
    stage_one["selected_for_stage_two"] = stage_one.apply(
        lambda row: bool(((top_triggers.roc_trigger_pct == row.roc_trigger_pct)
                          & (top_triggers.sqz_trigger_pct == row.sqz_trigger_pct)).any()), axis=1,
    )
    stage_two["selected"] = (
        (stage_two.roc_trigger_pct == selected.roc_trigger_pct)
        & (stage_two.sqz_trigger_pct == selected.sqz_trigger_pct)
        & (stage_two.roc_recovery_pct == selected.roc_recovery_pct)
        & (stage_two.sqz_recovery_pct == selected.sqz_recovery_pct)
    )
    return stage_one, stage_two, selected


def search_cooldown_pair(frame: pd.DataFrame, pair: str, start_ts: int, split_ts: int,
                         mechanism: str) -> tuple[pd.DataFrame, PairBreakerPolicy]:
    trigger = "loss" if mechanism == "mechanism_2" else "drawdown"
    rows = []
    for days in COOLDOWN_DAYS:
        for reset in RESET_MODES:
            policy = PairBreakerPolicy(trigger, days * DAY_SECONDS, reset, (pair,))
            metrics, _ = evaluate_pair(
                frame, pair, start_ts, split_ts, policy=policy,
            )
            rows.append({
                "mechanism": mechanism,
                "pair": pair,
                "trigger": trigger,
                "cooldown_days": days,
                "reset_baseline": reset,
                **metrics,
            })
    results = add_percentile_score(pd.DataFrame(rows))
    ranked = results.sort_values(
        ["score", "return_pct", "max_drawdown_pct"], ascending=False,
    )
    winner = ranked.iloc[0]
    selected = PairBreakerPolicy(
        trigger,
        int(winner.cooldown_days) * DAY_SECONDS,
        bool(winner.reset_baseline),
        (pair,),
    )
    results["selected"] = (
        (results.cooldown_days == int(winner.cooldown_days))
        & (results.reset_baseline == bool(winner.reset_baseline))
    )
    return results, selected


def paused_intervals(curve: pd.DataFrame, end_ts: int) -> list[tuple[int, int]]:
    intervals: list[tuple[int, int]] = []
    start = None
    for row in curve.itertuples(index=False):
        if bool(row.paused) and start is None:
            start = int(row.timestamp)
        elif not bool(row.paused) and start is not None:
            intervals.append((start, int(row.timestamp)))
            start = None
    if start is not None:
        intervals.append((start, end_ts))
    return intervals


def replay_pair_scenario(
    frame: pd.DataFrame,
    pair: str,
    scenario: str,
    start_ts: int,
    end_ts: int,
    *,
    technical: TechnicalParameters | None = None,
    policy: PairBreakerPolicy | None = None,
) -> tuple[dict, pd.DataFrame, list[dict]]:
    gate = technical_timeline(frame, technical) if technical is not None else None
    trades: list[dict] = []
    metrics, curve = evaluate_pair(
        frame, pair, start_ts, end_ts,
        gate=gate, policy=policy, record_curve=True, trade_log=trades,
    )
    result = pd.DataFrame({
        "timestamp": curve.timestamp.astype("int64"),
        "pair_equity": 200 + curve[f"{pair}_pnl_quote"],
        "pair_drawdown_pct": curve[f"{pair}_drawdown_pct"],
        "paused": (
            curve.timestamp.astype("int64").map(lambda value: not gate.get(int(value), False))
            if gate is not None else curve[f"{pair}_halted"].astype(bool)
        ),
    })
    result["scenario"] = scenario
    result["pair"] = pair
    events = [{"scenario": scenario, **item} for item in trades]
    for pause_start, pause_end in paused_intervals(result, end_ts):
        events.append({
            "scenario": scenario,
            "pair": pair,
            "timestamp": pause_start,
            "end_timestamp": pause_end,
            "side": "PAUSE",
            "reason": "technical_pause_interval" if gate is not None else "breaker_pause_interval",
            "trigger": "technical_gate" if gate is not None else f"pair_{policy.trigger}",
        })
    return metrics, result, events


def portfolio_curve(pair_curves: list[pd.DataFrame], scenario: str,
                    segment: str) -> pd.DataFrame:
    merged = None
    for frame in pair_curves:
        pair = str(frame.pair.iloc[0])
        values = frame[["timestamp", "pair_equity"]].rename(
            columns={"pair_equity": f"{pair}_equity"},
        )
        merged = values if merged is None else merged.merge(values, on="timestamp", how="inner")
    if merged is None or merged.empty:
        raise RuntimeError(f"No curves were available for {scenario}/{segment}")
    merged["equity"] = 20 + sum(merged[f"{pair}_equity"] for pair in PAIRS)
    merged["peak_equity"] = merged.equity.cummax().clip(lower=420)
    merged["drawdown_pct"] = (merged.equity / merged.peak_equity - 1) * 100
    merged["scenario"] = scenario
    merged["segment"] = segment
    return merged


def portfolio_metrics(curve: pd.DataFrame, pause_hours: float) -> dict:
    final_equity = float(curve.equity.iloc[-1])
    return {
        "net_pnl_quote": final_equity - 420,
        "return_pct": (final_equity - 420) / 420 * 100,
        "max_drawdown_pct": float(curve.drawdown_pct.min()),
        "pause_hours": pause_hours,
        "trades": 0,
        "trigger_count": 0,
        "fees_quote": 0.0,
    }


def locked_replay(
    candles: dict[str, pd.DataFrame],
    start_ts: int,
    split_ts: int,
    end_ts: int,
    technical_winners: dict[str, TechnicalParameters],
    cooldown_winners: dict[str, dict[str, PairBreakerPolicy]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    scenario_definitions = {
        "baseline": {},
        "mechanism_1": {"technical": technical_winners},
        "mechanism_2": {"policy": cooldown_winners["mechanism_2"]},
        "mechanism_3": {"policy": cooldown_winners["mechanism_3"]},
    }
    metric_rows: list[dict] = []
    curve_rows: list[pd.DataFrame] = []
    event_rows: list[dict] = []
    for segment, left, right in (
        ("development", start_ts, split_ts),
        ("holdout", split_ts, end_ts),
    ):
        for scenario, definition in scenario_definitions.items():
            pair_curves = []
            pair_pause_hours = 0.0
            pair_metrics_rows = []
            for pair in PAIRS:
                metrics, curve, events = replay_pair_scenario(
                    candles[pair], pair, scenario, left, right,
                    technical=definition.get("technical", {}).get(pair),
                    policy=definition.get("policy", {}).get(pair),
                )
                curve["segment"] = segment
                pair_curves.append(curve)
                curve_rows.append(curve)
                pair_pause_hours += metrics["pause_hours"]
                pair_metrics_rows.append(metrics)
                metric_rows.append({
                    "segment": segment, "scenario": scenario, "scope": pair, **metrics,
                })
                event_rows.extend({"segment": segment, **item} for item in events)
            combined = portfolio_curve(pair_curves, scenario, segment)
            curve_rows.append(combined)
            combined_metrics = portfolio_metrics(combined, pair_pause_hours)
            combined_metrics["trades"] = sum(item["trades"] for item in pair_metrics_rows)
            combined_metrics["trigger_count"] = sum(
                item["trigger_count"] for item in pair_metrics_rows
            )
            combined_metrics["fees_quote"] = sum(item["fees_quote"] for item in pair_metrics_rows)
            metric_rows.append({
                "segment": segment, "scenario": scenario, "scope": "PORTFOLIO",
                **combined_metrics,
            })
    metrics = pd.DataFrame(metric_rows)
    holdout_portfolio = metrics[
        (metrics.segment == "holdout") & (metrics.scope == "PORTFOLIO")
    ].copy()
    scored = add_percentile_score(holdout_portfolio)
    baseline_score = float(scored[scored.scenario == "baseline"].score.iloc[0])
    baseline_drawdown = float(
        scored[scored.scenario == "baseline"].max_drawdown_pct.iloc[0]
    )
    scored["beats_baseline_score"] = scored.score > baseline_score
    scored["drawdown_not_worse"] = scored.max_drawdown_pct >= baseline_drawdown
    scored["eligible_for_integrated_validation"] = (
        scored.beats_baseline_score & scored.drawdown_not_worse
    )
    for row in scored.itertuples(index=False):
        mask = (
            (metrics.segment == "holdout")
            & (metrics.scope == "PORTFOLIO")
            & (metrics.scenario == row.scenario)
        )
        for column in (
            "return_percentile", "drawdown_percentile", "pause_percentile", "score",
            "beats_baseline_score", "drawdown_not_worse", "eligible_for_integrated_validation",
        ):
            metrics.loc[mask, column] = getattr(row, column)
    curves = pd.concat(curve_rows, ignore_index=True, sort=False)
    events = pd.DataFrame(event_rows)
    return metrics, curves, events


def technical_heatmap(stage_one: pd.DataFrame) -> go.Figure:
    fig = make_subplots(rows=1, cols=2, subplot_titles=PAIRS, horizontal_spacing=0.12)
    for column, pair in enumerate(PAIRS, start=1):
        frame = stage_one[stage_one.pair == pair]
        matrix = frame.pivot(
            index="sqz_trigger_pct", columns="roc_trigger_pct", values="score",
        ).sort_index(ascending=False).sort_index(axis=1, ascending=False)
        fig.add_trace(go.Heatmap(
            x=matrix.columns, y=matrix.index, z=matrix.values,
            colorscale="Blues", zmin=0, zmax=1, showscale=column == 2,
            colorbar={"title": "开发集评分"} if column == 2 else None,
            hovertemplate="ROC≤%{x:.1f}%<br>SQZMOM≤%{y:.1f}%<br>评分=%{z:.3f}<extra></extra>",
        ), row=1, col=column)
        selected = frame[frame.selected_for_stage_two]
        fig.add_trace(go.Scatter(
            x=selected.roc_trigger_pct, y=selected.sqz_trigger_pct,
            mode="markers", name=f"{pair} 前10", showlegend=True,
            marker={"symbol": "x", "size": 10, "color": "#111827"},
        ), row=1, col=column)
    fig.update_xaxes(title_text="ROC48触发阈值（%）")
    fig.update_yaxes(title_text="SQZMOM触发阈值（%）")
    fig.update_layout(
        title={"text": "机制1：逐交易对ROC/SQZMOM触发参数搜索<br><sup>仅目标技术门启用；开发集前10组进入恢复参数联合搜索</sup>", "x": 0.02},
        template="plotly_white", height=600, margin={"t": 110, "l": 70, "r": 80},
        font={"family": "Arial, Microsoft YaHei, sans-serif"},
    )
    return fig


def technical_leaderboard(stage_two: pd.DataFrame) -> go.Figure:
    fig = make_subplots(rows=1, cols=2, subplot_titles=PAIRS, horizontal_spacing=0.18)
    for column, pair in enumerate(PAIRS, start=1):
        frame = stage_two[(stage_two.pair == pair) & stage_two.eligible].nlargest(10, "score").copy()
        frame = frame.sort_values("score")
        frame["label"] = frame.apply(
            lambda row: (
                f"T({row.roc_trigger_pct:g},{row.sqz_trigger_pct:g}) "
                f"R({row.roc_recovery_pct:g},{row.sqz_recovery_pct:g})"
            ), axis=1,
        )
        fig.add_trace(go.Bar(
            x=frame.score, y=frame.label, orientation="h", name=pair,
            marker_color=PAIR_COLORS[pair], showlegend=False,
            customdata=frame[["return_pct", "max_drawdown_pct", "pause_hours"]],
            hovertemplate=("评分=%{x:.3f}<br>收益=%{customdata[0]:.2f}%"
                           "<br>回撤=%{customdata[1]:.2f}%<br>暂停=%{customdata[2]:.1f}小时<extra></extra>"),
        ), row=1, col=column)
    fig.update_xaxes(title_text="开发集百分位综合评分", range=[0, 1.02])
    fig.update_layout(
        title={"text": "机制1：恢复参数联合搜索前10名<br><sup>T=触发阈值，R=恢复阈值；BTC与ETH分别锁定冠军</sup>", "x": 0.02},
        template="plotly_white", height=620, margin={"t": 110, "l": 150, "r": 40},
        font={"family": "Arial, Microsoft YaHei, sans-serif"},
    )
    return fig


def cooldown_figure(cooldown: pd.DataFrame) -> go.Figure:
    fig = make_subplots(
        rows=2, cols=2, shared_xaxes=True,
        subplot_titles=("机制2 · BTC", "机制2 · ETH", "机制3 · BTC", "机制3 · ETH"),
        vertical_spacing=0.14,
    )
    for row_index, mechanism in enumerate(("mechanism_2", "mechanism_3"), start=1):
        for column, pair in enumerate(PAIRS, start=1):
            frame = cooldown[(cooldown.mechanism == mechanism) & (cooldown.pair == pair)]
            for reset, dash, label in ((False, "solid", "不重置基准"), (True, "dash", "重置基准")):
                mode = frame[frame.reset_baseline == reset].sort_values("cooldown_days")
                fig.add_trace(go.Scatter(
                    x=mode.cooldown_days, y=mode.score, mode="lines+markers",
                    name=label, legendgroup=label,
                    showlegend=row_index == 1 and column == 1,
                    line={"color": PAIR_COLORS[pair], "dash": dash, "width": 2},
                    marker={"symbol": "circle" if not reset else "diamond"},
                    customdata=mode[["return_pct", "max_drawdown_pct", "pause_hours", "trigger_count"]],
                    hovertemplate=("冷却=%{x}天<br>评分=%{y:.3f}<br>收益=%{customdata[0]:.2f}%"
                                   "<br>回撤=%{customdata[1]:.2f}%<br>暂停=%{customdata[2]:.1f}小时"
                                   "<br>触发=%{customdata[3]:.0f}<extra></extra>"),
                ), row=row_index, col=column)
            selected = frame[frame.selected]
            fig.add_trace(go.Scatter(
                x=selected.cooldown_days, y=selected.score, mode="markers",
                name="锁定冠军", legendgroup="winner",
                showlegend=row_index == 1 and column == 1,
                marker={"symbol": "x", "size": 14, "color": "#111827", "line": {"width": 2}},
            ), row=row_index, col=column)
    fig.update_xaxes(title_text="冷却天数", dtick=1)
    fig.update_yaxes(title_text="开发集评分", range=[0, 1.02])
    fig.update_layout(
        title={"text": "机制2/3：1–7天冷却与基准重置模式搜索<br><sup>机制2=单对亏损6 FDUSD；机制3=单对峰值回撤3%；其余风控关闭</sup>", "x": 0.02},
        template="plotly_white", height=900, margin={"t": 115, "l": 70, "r": 40},
        font={"family": "Arial, Microsoft YaHei, sans-serif"},
    )
    return fig


def holdout_scatter(metrics: pd.DataFrame) -> go.Figure:
    frame = metrics[(metrics.segment == "holdout") & (metrics.scope == "PORTFOLIO")].copy()
    frame["drawdown_magnitude"] = -frame.max_drawdown_pct
    fig = go.Figure()
    for row in frame.itertuples(index=False):
        fig.add_trace(go.Scatter(
            x=[row.drawdown_magnitude], y=[row.return_pct], mode="markers+text",
            text=[row.scenario], textposition="top center", name=row.scenario,
            marker={"size": 16, "color": SCENARIO_COLORS[row.scenario],
                    "symbol": "diamond" if bool(getattr(row, "eligible_for_integrated_validation", False)) else "circle"},
            customdata=[[row.pause_hours, row.score, row.trigger_count]],
            hovertemplate=("最大回撤=%{x:.2f}%<br>收益=%{y:.2f}%<br>暂停=%{customdata[0]:.1f}对小时"
                           "<br>评分=%{customdata[1]:.3f}<br>触发=%{customdata[2]:.0f}<extra></extra>"),
        ))
    fig.update_layout(
        title={"text": "锁定样本外：收益—回撤对比<br><sup>越靠左上越好；菱形表示评分优于无风控基线且回撤未恶化</sup>", "x": 0.02},
        template="plotly_white", height=620, margin={"t": 105, "l": 75, "r": 45},
        xaxis_title="最大回撤绝对值（%）", yaxis_title="样本外收益（%）",
        font={"family": "Arial, Microsoft YaHei, sans-serif"},
    )
    fig.add_hline(y=0, line={"color": "#64748B", "dash": "dot"})
    return fig


def equity_figure(curves: pd.DataFrame, split_ts: int) -> go.Figure:
    portfolio = curves[curves.equity.notna()].copy()
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.10,
                        subplot_titles=("组合权益", "组合回撤"))
    for scenario, color in SCENARIO_COLORS.items():
        for segment in ("development", "holdout"):
            frame = portfolio[(portfolio.scenario == scenario) & (portfolio.segment == segment)]
            fig.add_trace(go.Scatter(
                x=pd.to_datetime(frame.timestamp, unit="s", utc=True), y=frame.equity,
                mode="lines", name=scenario, legendgroup=scenario,
                showlegend=segment == "development", line={"color": color, "width": 1.6},
            ), row=1, col=1)
            fig.add_trace(go.Scatter(
                x=pd.to_datetime(frame.timestamp, unit="s", utc=True), y=frame.drawdown_pct,
                mode="lines", name=scenario, legendgroup=scenario, showlegend=False,
                line={"color": color, "width": 1.3},
            ), row=2, col=1)
    split_time = pd.to_datetime(split_ts, unit="s", utc=True)
    fig.add_vline(x=split_time, line={"color": "#111827", "dash": "dash", "width": 1.5})
    fig.add_annotation(x=split_time, y=1.02, xref="x", yref="paper", text="70/30锁参边界",
                       showarrow=False, xanchor="left")
    fig.update_yaxes(title_text="FDUSD", row=1, col=1)
    fig.update_yaxes(title_text="回撤（%）", row=2, col=1)
    fig.update_layout(
        title={"text": "无风控基线与三个锁定机制：开发集/样本外权益曲线<br><sup>样本外段从420 FDUSD重新计量，参数不再重新选择</sup>", "x": 0.02},
        template="plotly_white", height=850, margin={"t": 115, "l": 75, "r": 40},
        hovermode="x unified", font={"family": "Arial, Microsoft YaHei, sans-serif"},
    )
    return fig


def pause_timeline(events: pd.DataFrame) -> go.Figure:
    intervals = events[events.reason.isin(["technical_pause_interval", "breaker_pause_interval"])].copy()
    intervals["lane"] = intervals.scenario + " · " + intervals.pair
    fig = go.Figure()
    for scenario, color in SCENARIO_COLORS.items():
        frame = intervals[intervals.scenario == scenario]
        if frame.empty:
            continue
        fig.add_trace(go.Bar(
            x=(frame.end_timestamp - frame.timestamp) * 1000,
            base=pd.to_datetime(frame.timestamp, unit="s", utc=True),
            y=frame.lane, orientation="h", name=scenario, marker_color=color,
            customdata=(frame.end_timestamp - frame.timestamp) / 3600,
            hovertemplate="%{y}<br>开始=%{base|%Y-%m-%d %H:%M UTC}<br>时长=%{customdata:.1f}小时<extra></extra>",
        ))
    fig.update_layout(
        title={"text": "锁定参数的停止交易区间<br><sup>技术门仅禁止对应交易对买入；机制2/3在冷却期间停止该交易对网格</sup>", "x": 0.02},
        template="plotly_white", height=620, margin={"t": 105, "l": 175, "r": 40},
        barmode="overlay", xaxis_title="UTC时间", yaxis_title="机制 · 交易对",
        font={"family": "Arial, Microsoft YaHei, sans-serif"},
    )
    return fig


def figure_html(fig: go.Figure, include_plotlyjs: bool) -> str:
    return fig.to_html(
        full_html=False, include_plotlyjs=include_plotlyjs,
        config={"displaylogo": False, "responsive": True, "scrollZoom": True},
    )


def write_report(output_dir: Path, figures: list[go.Figure], summary: dict,
                 metrics: pd.DataFrame) -> None:
    winners = summary["locked_parameters"]
    cards = []
    for mechanism in ("mechanism_1", "mechanism_2", "mechanism_3"):
        decision = summary["holdout_decisions"][mechanism]
        cards.append(
            f'<article><h3>{html.escape(mechanism)}</h3>'
            f'<p class="decision">{html.escape(decision["decision"])}</p>'
            f'<pre>{html.escape(json.dumps(winners[mechanism], ensure_ascii=False, indent=2))}</pre>'
            f'<p>{html.escape(decision["reason"])}</p></article>'
        )
    sections = "".join(
        f'<section class="chart">{figure_html(figure, index == 0)}</section>'
        for index, figure in enumerate(figures)
    )
    holdout = metrics[(metrics.segment == "holdout") & (metrics.scope == "PORTFOLIO")]
    rows = "".join(
        f"<tr><td>{html.escape(str(row.scenario))}</td><td>{row.return_pct:+.2f}%</td>"
        f"<td>{row.max_drawdown_pct:.2f}%</td><td>{row.pause_hours:.1f}</td>"
        f"<td>{getattr(row, 'score', float('nan')):.3f}</td></tr>"
        for row in holdout.itertuples(index=False)
    )
    document = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>FDUSD Grid 机制1–3年度搜索</title>
<style>body{{margin:0;background:#f8fafc;color:#1f2937;font-family:Arial,'Microsoft YaHei',sans-serif}}
main{{max-width:1500px;margin:auto;padding:28px 20px 60px}}header,.chart,article{{background:white;border:1px solid #cbd5e1;border-radius:10px}}
header{{padding:24px 28px}}.cards{{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin:20px 0}}article{{padding:18px}}
.chart{{margin:22px 0;padding:8px}}pre{{white-space:pre-wrap;background:#f1f5f9;padding:10px;font-size:12px}}table{{border-collapse:collapse;width:100%}}
th,td{{border-bottom:1px solid #e2e8f0;padding:9px;text-align:right}}th:first-child,td:first-child{{text-align:left}}.decision{{font-weight:700}}
@media(max-width:900px){{.cards{{grid-template-columns:1fr}}main{{padding:12px 6px}}}}</style></head><body><main>
<header><h1>FDUSD Grid 机制1–3：2026年度独立参数搜索</h1>
<p>BTC与ETH分别选参；前70%开发、后30%锁定样本外。搜索时仅启用目标机制，Maker 0%、Taker 0.1%、2小时挂单、成本底线关闭。</p>
<p><b>数据：</b>{html.escape(summary['period']['start_iso'])} 至 {html.escape(summary['period']['end_iso'])}；本报告只给出后续联合验证建议，不授权部署OCI。</p>
<h2>样本外组合结果</h2><table><thead><tr><th>场景</th><th>收益</th><th>最大回撤</th><th>暂停对小时</th><th>评分</th></tr></thead><tbody>{rows}</tbody></table>
</header><div class="cards">{''.join(cards)}</div>{sections}</main></body></html>"""
    (output_dir / "report.html").write_text(document, encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.top_trigger_count <= 0 or args.top_trigger_count > 56:
        raise ValueError("top-trigger-count must be between 1 and 56")
    start_ts = iso_timestamp(args.start) // BAR_SECONDS * BAR_SECONDS
    end_ts = (args.end_ts or latest_complete_bar()) // BAR_SECONDS * BAR_SECONDS
    if end_ts <= start_ts + 30 * DAY_SECONDS:
        raise ValueError("evaluation period must exceed 30 days")
    total_bars = (end_ts - start_ts) // BAR_SECONDS
    split_ts = start_ts + int(total_bars * 0.70) * BAR_SECONDS
    warmup_start = start_ts - WARMUP_DAYS * DAY_SECONDS
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading and validating 2026 YTD candles...", flush=True)
    candles = {
        pair: load_candles(
            pair, warmup_start, end_ts, args.cache_dir, allow_download=not args.no_download,
        )
        for pair in PAIRS
    }
    quality = pd.DataFrame([
        data_quality(candles[pair], pair, start_ts, split_ts, end_ts, warmup_start)
        for pair in PAIRS
    ])
    if not bool((quality.coverage_pct >= 98).all()):
        raise RuntimeError("evaluation candle coverage is below 98%")
    if not bool(quality.warmup_complete.all()):
        raise RuntimeError("the eight-day technical indicator warmup is incomplete")
    quality.to_csv(args.output_dir / "data_quality.csv", index=False)

    technical_winners: dict[str, TechnicalParameters] = {}
    if args.reuse_technical_dir is not None:
        prior_summary = json.loads(
            (args.reuse_technical_dir / "summary.json").read_text(encoding="utf-8")
        )
        prior_period = prior_summary["period"]
        if (
            int(prior_period["start_ts"]) != start_ts
            or int(prior_period["split_ts"]) != split_ts
            or int(prior_period["end_ts"]) != end_ts
        ):
            raise ValueError("reused mechanism-1 results do not match the fixed period")
        stage_one = pd.read_csv(args.reuse_technical_dir / "mechanism1_trigger_search.csv")
        stage_two = pd.read_csv(args.reuse_technical_dir / "mechanism1_joint_search.csv")
        for pair in PAIRS:
            selected_mask = stage_two.selected.astype(str).str.lower().eq("true")
            selected = stage_two[(stage_two.pair == pair) & selected_mask]
            if len(selected) != 1:
                raise ValueError(f"reused mechanism-1 results require one winner for {pair}")
            row = selected.iloc[0]
            technical_winners[pair] = TechnicalParameters(
                float(row.roc_trigger_pct), float(row.sqz_trigger_pct),
                float(row.roc_recovery_pct), float(row.sqz_recovery_pct),
            )
        print(f"Reused mechanism 1 search from {args.reuse_technical_dir}", flush=True)
    else:
        stage_one_parts = []
        stage_two_parts = []
        for pair in PAIRS:
            print(f"Searching mechanism 1 for {pair}...", flush=True)
            stage_one, stage_two, winner = search_technical_pair(
                candles[pair], pair, start_ts, split_ts, args.top_trigger_count,
            )
            stage_one_parts.append(stage_one)
            stage_two_parts.append(stage_two)
            technical_winners[pair] = winner
        stage_one = pd.concat(stage_one_parts, ignore_index=True)
        stage_two = pd.concat(stage_two_parts, ignore_index=True)

    cooldown_parts = []
    cooldown_winners: dict[str, dict[str, PairBreakerPolicy]] = {
        "mechanism_2": {}, "mechanism_3": {},
    }
    for mechanism in ("mechanism_2", "mechanism_3"):
        for pair in PAIRS:
            print(f"Searching {mechanism} cooldown for {pair}...", flush=True)
            results, winner = search_cooldown_pair(
                candles[pair], pair, start_ts, split_ts, mechanism,
            )
            cooldown_parts.append(results)
            cooldown_winners[mechanism][pair] = winner
    cooldown = pd.concat(cooldown_parts, ignore_index=True)

    print("Replaying locked winners on development and holdout segments...", flush=True)
    metrics, curves, events = locked_replay(
        candles, start_ts, split_ts, end_ts,
        technical_winners, cooldown_winners,
    )
    holdout_portfolio = metrics[
        (metrics.segment == "holdout") & (metrics.scope == "PORTFOLIO")
    ].set_index("scenario")
    decisions = {}
    for mechanism in ("mechanism_1", "mechanism_2", "mechanism_3"):
        row = holdout_portfolio.loc[mechanism]
        eligible = bool(row.eligible_for_integrated_validation)
        decisions[mechanism] = {
            "decision": "可进入完整线上机制联合验证" if eligible else "样本外否决",
            "reason": (
                "样本外综合评分高于无风控基线，且最大回撤未恶化。"
                if eligible else
                "未同时满足样本外综合评分高于无风控基线、最大回撤不恶化两项条件。"
            ),
            "score": float(row.score),
            "return_pct": float(row.return_pct),
            "max_drawdown_pct": float(row.max_drawdown_pct),
            "pause_hours": float(row.pause_hours),
        }
    summary = {
        "schema_version": "fdusd-ytd-risk-mechanism-search-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "period": {
            "warmup_start_ts": warmup_start,
            "start_ts": start_ts,
            "split_ts": split_ts,
            "end_ts": end_ts,
            "start_iso": datetime.fromtimestamp(start_ts, timezone.utc).isoformat(),
            "split_iso": datetime.fromtimestamp(split_ts, timezone.utc).isoformat(),
            "end_iso": datetime.fromtimestamp(end_ts, timezone.utc).isoformat(),
            "development_fraction": 0.70,
            "holdout_fraction": 0.30,
        },
        "base_grid": {**asdict(BASE_CANDIDATE), "levels": BASE_CANDIDATE.levels},
        "execution": {
            "pair_budget_fdusd": 200,
            "reserve_fdusd": 20,
            "maker_fee": 0.0,
            "taker_fee": 0.001,
            "order_lifetime_seconds": 7200,
            "cost_floor_enabled": False,
            "other_risk_controls_enabled": False,
        },
        "search_space": {
            "technical_trigger": {
                "roc_pct": ROC_TRIGGER_THRESHOLDS,
                "sqzmom_pct": SQZ_TRIGGER_THRESHOLDS,
            },
            "technical_recovery": {
                "roc_pct": ROC_RECOVERY_THRESHOLDS,
                "sqzmom_pct": SQZ_RECOVERY_THRESHOLDS,
                "top_trigger_count": args.top_trigger_count,
            },
            "cooldown_days": COOLDOWN_DAYS,
            "reset_baseline": RESET_MODES,
        },
        "score": "candidate percentiles: return 50%, maximum drawdown 30%, pause duration 20%",
        "locked_parameters": {
            "mechanism_1": {
                pair: asdict(technical_winners[pair]) for pair in PAIRS
            },
            "mechanism_2": {
                pair: asdict(cooldown_winners["mechanism_2"][pair]) for pair in PAIRS
            },
            "mechanism_3": {
                pair: asdict(cooldown_winners["mechanism_3"][pair]) for pair in PAIRS
            },
        },
        "holdout_decisions": decisions,
        "deployment_authorized": False,
        "deployment_note": "Research output only; OCI and live parameters remain unchanged.",
    }

    stage_one.sort_values(["pair", "score"], ascending=[True, False]).to_csv(
        args.output_dir / "mechanism1_trigger_search.csv", index=False,
    )
    stage_two.sort_values(["pair", "score"], ascending=[True, False]).to_csv(
        args.output_dir / "mechanism1_joint_search.csv", index=False,
    )
    cooldown.sort_values(["mechanism", "pair", "score"], ascending=[True, True, False]).to_csv(
        args.output_dir / "mechanism2_3_cooldown_search.csv", index=False,
    )
    metrics.to_csv(args.output_dir / "locked_segment_metrics.csv", index=False)
    curves.to_csv(args.output_dir / "locked_curves.csv", index=False)
    events.to_csv(args.output_dir / "locked_events.csv", index=False)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8",
    )

    figures = [
        technical_heatmap(stage_one),
        technical_leaderboard(stage_two),
        cooldown_figure(cooldown),
        holdout_scatter(metrics),
        equity_figure(curves, split_ts),
        pause_timeline(events),
    ]
    chart_map = [
        {"section": "机制1触发", "family": "heatmap", "question": "每对触发阈值如何影响开发集评分"},
        {"section": "机制1联合参数", "family": "bar", "question": "每对联合参数前10名是什么"},
        {"section": "机制2/3冷却", "family": "line", "question": "延迟天数和重置模式如何影响评分"},
        {"section": "样本外比较", "family": "scatter", "question": "收益与回撤是否优于无风控基线"},
        {"section": "权益轨迹", "family": "line", "question": "锁定机制在开发与样本外如何表现"},
        {"section": "停止区间", "family": "timeline", "question": "何时以及多久停止交易"},
    ]
    (args.output_dir / "chart_map.json").write_text(
        json.dumps(chart_map, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    write_report(args.output_dir, figures, summary, metrics)
    print(json.dumps({
        "output_dir": str(args.output_dir),
        "period": summary["period"],
        "locked_parameters": summary["locked_parameters"],
        "holdout_decisions": decisions,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

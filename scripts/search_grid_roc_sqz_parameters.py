#!/usr/bin/env python3
"""Search ROC/SQZMOM Grid risk-off thresholds with maroon recovery."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from validate_grid_live import (
    Candidate,
    read_cache,
    simulate,
    technical_buy_gate_timeline,
)


DAY_SECONDS = 86_400
WARMUP_SECONDS = 8 * DAY_SECONDS
PAIRS = ("BTC-FDUSD", "ETH-FDUSD")
ROC_THRESHOLDS = (-4.0, -5.0, -6.0, -7.0, -8.0, -9.0, -10.0, -12.0)
SQZMOM_THRESHOLDS = (-1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -8.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=180)
    parser.add_argument("--train-days", type=int, default=120)
    parser.add_argument("--end-ts", type=int, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path("data/backtesting_candles"))
    parser.add_argument("--validation-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--maker-fee", type=float, default=0.0)
    parser.add_argument("--taker-fee", type=float, default=0.001)
    return parser.parse_args()


def load_candles(cache_dir: Path, start_ts: int, end_ts: int) -> dict[str, pd.DataFrame]:
    result = {}
    for pair in PAIRS:
        frame = read_cache(cache_dir / f"binance_{pair}_5m.csv")
        frame = frame[
            (frame.timestamp >= start_ts - WARMUP_SECONDS)
            & (frame.timestamp < end_ts)
        ].sort_values("timestamp").reset_index(drop=True)
        expected = (end_ts - start_ts) // 300
        actual = len(frame[(frame.timestamp >= start_ts) & (frame.timestamp < end_ts)])
        if actual < expected * 0.98:
            raise RuntimeError(f"{pair} coverage is only {actual}/{expected}")
        result[pair] = frame
    return result


def slice_candles(
    candles: dict[str, pd.DataFrame], start_ts: int, end_ts: int,
) -> dict[str, pd.DataFrame]:
    return {
        pair: frame[(frame.timestamp >= start_ts) & (frame.timestamp < end_ts)].reset_index(drop=True)
        for pair, frame in candles.items()
    }


def window_stats(timeline: dict[int, bool], start_ts: int, end_ts: int) -> tuple[int, float]:
    values = [
        enabled for timestamp, enabled in sorted(timeline.items())
        if start_ts <= timestamp < end_ts
    ]
    windows = 0
    previous = True
    for enabled in values:
        if previous and not enabled:
            windows += 1
        previous = enabled
    hours = sum(not enabled for enabled in values) * 5 / 60
    return windows, hours


def segment_metrics(
    candles: dict[str, pd.DataFrame], timeline: dict[int, bool], candidate: Candidate,
    start_ts: int, end_ts: int, maker_fee: float, taker_fee: float,
) -> dict:
    segment = slice_candles(candles, start_ts, end_ts)
    result, _, _ = simulate(
        segment, candidate, maker_fee, taker_fee=taker_fee,
        technical_buy_gate=timeline, risk_breakers_enabled=False,
    )
    windows, hours = window_stats(timeline, start_ts, end_ts)
    days = (end_ts - start_ts) / DAY_SECONDS
    normalized_return = result["net_pnl_pct"] * 30 / days
    score = normalized_return - 0.75 * abs(result["max_drawdown_pct"])
    return {
        "return_pct": result["net_pnl_pct"] * 100,
        "max_drawdown_pct": result["max_drawdown_pct"] * 100,
        "risk_adjusted_score": score,
        "trades": result["trades"],
        "risk_off_windows": windows,
        "risk_off_hours": hours,
    }


def heatmap_frame(rows: pd.DataFrame, value: str) -> pd.DataFrame:
    return rows.pivot(index="sqzmom_risk_off_pct", columns="roc_risk_off_pct", values=value).sort_index(
        ascending=False
    ).sort_index(axis=1, ascending=False)


def build_figure(rows: pd.DataFrame, selected: pd.Series) -> go.Figure:
    fig = make_subplots(
        rows=2, cols=2, horizontal_spacing=0.10, vertical_spacing=0.15,
        subplot_titles=(
            "Robust score: min(train, validation)",
            "180-day return (%)",
            "180-day maximum drawdown (%)",
            "180-day BUY-off hours",
        ),
    )
    specifications = (
        ("robust_score", "Cividis", 1, 1, ".3f", 0.45, 0.79, ""),
        ("full_return_pct", "Blues", 1, 2, ".2f", 1.02, 0.79, "%"),
        ("full_max_drawdown_pct", "Oranges_r", 2, 1, ".2f", 0.45, 0.22, ""),
        ("full_risk_off_hours", "Blues", 2, 2, ".1f", 1.02, 0.22, "hours"),
    )
    for index, (value, colorscale, row, col, fmt, colorbar_x, colorbar_y, colorbar_title) in enumerate(
        specifications, 1
    ):
        matrix = heatmap_frame(rows, value)
        coloraxis = "coloraxis" if index == 1 else f"coloraxis{index}"
        fig.add_trace(go.Heatmap(
            x=matrix.columns, y=matrix.index, z=matrix.values,
            coloraxis=coloraxis,
            customdata=matrix.values,
            hovertemplate=(
                "ROC threshold=%{x:.1f}%<br>SQZMOM threshold=%{y:.1f}%"
                f"<br>{value}=%{{customdata:{fmt}}}<extra></extra>"
            ),
        ), row=row, col=col)
        fig.update_layout(**{
            coloraxis: {
                "colorscale": colorscale,
                "showscale": True,
                "colorbar": {
                    "x": colorbar_x, "y": colorbar_y, "len": 0.31,
                    "thickness": 11, "title": {"text": colorbar_title, "side": "right"},
                },
            }
        })
        fig.add_trace(go.Scatter(
            x=[selected["roc_risk_off_pct"]], y=[selected["sqzmom_risk_off_pct"]],
            mode="markers", showlegend=False,
            marker={"symbol": "x", "size": 14, "color": "#111827", "line": {"width": 2}},
            hovertemplate="selected<extra></extra>",
        ), row=row, col=col)
    fig.update_xaxes(title_text="ROC 48h threshold (%)")
    fig.update_yaxes(title_text="SQZMOM threshold (%)")
    fig.update_layout(
        title={
            "text": (
                "FDUSD Grid ROC/SQZMOM threshold search"
                "<br><sup>AND trigger; first maroon bar recovers BUY; 120-day train / 60-day validation; no hard breaker</sup>"
            ),
            "x": 0.02,
        },
        template="plotly_white", height=1050, margin={"l": 75, "r": 100, "t": 115, "b": 60},
    )
    return fig


def main() -> int:
    args = parse_args()
    if args.train_days <= 0 or args.train_days >= args.days:
        raise ValueError("train-days must be between zero and days")
    end_ts = args.end_ts // 300 * 300
    start_ts = end_ts - args.days * DAY_SECONDS
    split_ts = start_ts + args.train_days * DAY_SECONDS
    candles = load_candles(args.cache_dir, start_ts, end_ts)
    selection = json.loads((args.validation_dir / "active_selection.json").read_text(encoding="utf-8"))
    params = selection["parameters"]
    candidate = Candidate(
        half_range=float(params["half_range"]),
        min_spread=float(params["minimum_spread"]),
        take_profit=float(params["take_profit"]),
        move_threshold=float(params["move_threshold"]),
        move_cooldown_seconds=int(params["min_grid_move_seconds"]),
    )
    rows = []
    for roc_threshold in ROC_THRESHOLDS:
        for sqz_threshold in SQZMOM_THRESHOLDS:
            timeline = technical_buy_gate_timeline(
                candles["BTC-FDUSD"],
                roc_risk_off_pct=roc_threshold,
                sqzmom_risk_off_pct=sqz_threshold,
            )
            train = segment_metrics(
                candles, timeline, candidate, start_ts, split_ts,
                args.maker_fee, args.taker_fee,
            )
            validation = segment_metrics(
                candles, timeline, candidate, split_ts, end_ts,
                args.maker_fee, args.taker_fee,
            )
            full = segment_metrics(
                candles, timeline, candidate, start_ts, end_ts,
                args.maker_fee, args.taker_fee,
            )
            eligible = (
                train["risk_off_windows"] >= 1
                and validation["risk_off_windows"] >= 1
                and full["risk_off_windows"] >= 2
            )
            rows.append({
                "roc_risk_off_pct": roc_threshold,
                "sqzmom_risk_off_pct": sqz_threshold,
                "eligible": eligible,
                "robust_score": min(train["risk_adjusted_score"], validation["risk_adjusted_score"]),
                "mean_score": (train["risk_adjusted_score"] + validation["risk_adjusted_score"]) / 2,
                **{f"train_{key}": value for key, value in train.items()},
                **{f"validation_{key}": value for key, value in validation.items()},
                **{f"full_{key}": value for key, value in full.items()},
            })
    results = pd.DataFrame(rows)
    eligible = results[results.eligible]
    if eligible.empty:
        raise RuntimeError("no threshold pair triggered in both train and validation periods")
    selected_index = eligible.sort_values(
        ["robust_score", "mean_score", "full_risk_adjusted_score"], ascending=False
    ).index[0]
    results["selected"] = results.index == selected_index
    selected = results.loc[selected_index]
    results = results.sort_values(
        ["eligible", "robust_score", "mean_score"], ascending=False
    ).reset_index(drop=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output_dir / "parameter_search.csv", index=False)
    payload = {
        "schema_version": "grid-roc-sqz-parameter-search-v1",
        "period": {"start_ts": start_ts, "split_ts": split_ts, "end_ts": end_ts,
                   "days": args.days, "train_days": args.train_days,
                   "validation_days": args.days - args.train_days},
        "search_space": {"roc_risk_off_pct": ROC_THRESHOLDS,
                         "sqzmom_risk_off_pct": SQZMOM_THRESHOLDS},
        "candidate": {**asdict(candidate), "levels": candidate.levels},
        "trigger": "ROC48 <= threshold AND SQZMOM <= threshold",
        "recovery": "first maroon SQZMOM bar while risk-off is active",
        "risk_breakers_enabled": False,
        "eligibility": "at least one risk-off window in train and validation; at least two full-period windows",
        "score": "min(30-day-normalized return - 0.75*abs(max drawdown)) across train and validation",
        "selected": selected.to_dict(),
        "evaluated_combinations": len(results),
        "eligible_combinations": int(results.eligible.sum()),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    fig = build_figure(results, selected)
    fig.write_html(args.output_dir / "parameter_search_heatmaps.html", include_plotlyjs=True, full_html=True)
    fig.write_json(args.output_dir / "parameter_search_heatmaps.plotly.json")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Search combined ROC/SQZMOM recovery thresholds for the FDUSD Grid."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from grid_technical_gate import roc_sqz_signal_from_klines
from search_grid_roc_sqz_parameters import (
    DAY_SECONDS,
    PAIRS,
    WARMUP_SECONDS,
    load_candles,
    segment_metrics,
)
from validate_grid_live import Candidate
from validate_grid_live import technical_buy_gate_timeline


ROC_RECOVERY_THRESHOLDS = (
    -6.0, -5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
)
SQZMOM_RECOVERY_THRESHOLDS = (-3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0)


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
    parser.add_argument("--roc-risk-off-pct", type=float, default=-5.0)
    parser.add_argument("--sqzmom-risk-off-pct", type=float, default=-1.0)
    return parser.parse_args()


def advance_combined_gate(
    active: bool,
    signal: dict,
    *,
    roc_risk_off_pct: float,
    sqzmom_risk_off_pct: float,
    roc_recovery_pct: float,
    sqzmom_recovery_pct: float,
) -> tuple[bool, bool, bool]:
    roc = float(signal["roc_48h_pct"])
    sqz = float(signal["sqzmom_pct"])
    improving = float(signal["sqzmom"]) > float(signal["sqzmom_previous"])
    adverse = roc <= roc_risk_off_pct and sqz <= sqzmom_risk_off_pct
    trigger = bool(not active and adverse)
    recover = bool(
        active
        and not adverse
        and improving
        and roc >= roc_recovery_pct
        and sqz >= sqzmom_recovery_pct
    )
    if trigger:
        active = True
    elif recover:
        active = False
    return active, trigger, recover


def combined_recovery_timeline(
    frame: pd.DataFrame,
    *,
    roc_risk_off_pct: float,
    sqzmom_risk_off_pct: float,
    roc_recovery_pct: float,
    sqzmom_recovery_pct: float,
) -> dict[int, bool]:
    """Build a closed-4h-bar, no-lookahead BUY gate with combined recovery."""
    timeline, _, _ = combined_recovery_diagnostics(
        frame,
        roc_risk_off_pct=roc_risk_off_pct,
        sqzmom_risk_off_pct=sqzmom_risk_off_pct,
        roc_recovery_pct=roc_recovery_pct,
        sqzmom_recovery_pct=sqzmom_recovery_pct,
    )
    return timeline


def combined_recovery_diagnostics(
    frame: pd.DataFrame,
    *,
    roc_risk_off_pct: float,
    sqzmom_risk_off_pct: float,
    roc_recovery_pct: float,
    sqzmom_recovery_pct: float,
) -> tuple[dict[int, bool], list[dict], list[dict]]:
    """Return the no-lookahead gate, transition events, and closed-bar signals."""
    source = frame.sort_values("timestamp").copy()
    source["bucket"] = (source.timestamp.astype("int64") // 14_400) * 14_400
    bars = source.groupby("bucket", sort=True).agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        rows=("close", "size"),
    ).reset_index()
    bars = bars[bars.rows == 48]
    klines: list[list] = []
    transitions: list[tuple[int, bool]] = []
    events: list[dict] = []
    observations: list[dict] = []
    active = False
    for row in bars.itertuples(index=False):
        close_time = int(row.bucket) + 14_400 - 1
        klines.append([
            int(row.bucket) * 1000,
            row.open,
            row.high,
            row.low,
            row.close,
            0,
            close_time * 1000,
        ])
        if len(klines) < 40:
            continue
        signal = roc_sqz_signal_from_klines(klines[-64:])
        active, trigger, recover = advance_combined_gate(
            active,
            signal,
            roc_risk_off_pct=roc_risk_off_pct,
            sqzmom_risk_off_pct=sqzmom_risk_off_pct,
            roc_recovery_pct=roc_recovery_pct,
            sqzmom_recovery_pct=sqzmom_recovery_pct,
        )
        effective_at = close_time + 1
        transitions.append((effective_at, not active))
        observation = {
            "timestamp": effective_at,
            "roc_48h_pct": float(signal["roc_48h_pct"]),
            "sqzmom_pct": float(signal["sqzmom_pct"]),
            "sqzmom": float(signal["sqzmom"]),
            "sqzmom_previous": float(signal["sqzmom_previous"]),
            "sqzmom_color": str(signal["sqzmom_color"]),
            "buy_enabled": not active,
        }
        observations.append(observation)
        if trigger or recover:
            events.append({
                **observation,
                "event": "risk_off" if trigger else "recovery",
            })
    timeline: dict[int, bool] = {}
    pointer = 0
    enabled = False
    for raw_timestamp in source.timestamp:
        timestamp = int(raw_timestamp)
        while pointer < len(transitions) and transitions[pointer][0] <= timestamp:
            enabled = transitions[pointer][1]
            pointer += 1
        timeline[timestamp] = enabled
    return timeline, events, observations


def gate_intervals(
    timeline: dict[int, bool], start_ts: int, end_ts: int,
) -> list[tuple[int, int, bool]]:
    """Compress five-minute gate values into contiguous plotting intervals."""
    values = [(timestamp, enabled) for timestamp, enabled in sorted(timeline.items())
              if start_ts <= timestamp < end_ts]
    if not values:
        return []
    intervals: list[tuple[int, int, bool]] = []
    interval_start, enabled = values[0]
    for timestamp, next_enabled in values[1:]:
        if next_enabled != enabled:
            intervals.append((interval_start, timestamp, enabled))
            interval_start, enabled = timestamp, next_enabled
    intervals.append((interval_start, end_ts, enabled))
    return intervals


def heatmap(rows: pd.DataFrame, value: str) -> pd.DataFrame:
    return rows.pivot(
        index="sqzmom_recovery_pct", columns="roc_recovery_pct", values=value
    ).sort_index(ascending=False).sort_index(axis=1)


def build_figure(rows: pd.DataFrame, selected: pd.Series) -> go.Figure:
    fig = make_subplots(
        rows=2,
        cols=2,
        horizontal_spacing=0.10,
        vertical_spacing=0.15,
        subplot_titles=(
            "Robust train/validation score",
            "180-day return (%)",
            "180-day maximum drawdown (%)",
            "180-day BUY-off hours",
        ),
    )
    specs = (
        ("robust_score", "Cividis", 1, 1),
        ("full_return_pct", "Blues", 1, 2),
        ("full_max_drawdown_pct", "Oranges_r", 2, 1),
        ("full_risk_off_hours", "Blues", 2, 2),
    )
    for index, (field, colorscale, row, col) in enumerate(specs, 1):
        matrix = heatmap(rows, field)
        coloraxis = "coloraxis" if index == 1 else f"coloraxis{index}"
        fig.add_trace(go.Heatmap(
            x=matrix.columns,
            y=matrix.index,
            z=matrix.values,
            coloraxis=coloraxis,
            hovertemplate=(
                "ROC recovery=%{x:.1f}%<br>SQZMOM recovery=%{y:.1f}%"
                f"<br>{field}=%{{z:.4f}}<extra></extra>"
            ),
        ), row=row, col=col)
        fig.add_trace(go.Scatter(
            x=[selected["roc_recovery_pct"]],
            y=[selected["sqzmom_recovery_pct"]],
            mode="markers",
            marker={"symbol": "x", "size": 14, "color": "#111827", "line": {"width": 2}},
            showlegend=False,
            hovertemplate="selected<extra></extra>",
        ), row=row, col=col)
        fig.update_layout(**{coloraxis: {"colorscale": colorscale, "showscale": False}})
    fig.update_xaxes(title_text="ROC recovery threshold (%)")
    fig.update_yaxes(title_text="SQZMOM recovery threshold (%)")
    fig.update_layout(
        title={
            "text": (
                "FDUSD Grid combined ROC/SQZMOM recovery search"
                "<br><sup>Risk-off fixed at ROC≤-5% AND SQZMOM≤-1%; "
                "recovery requires both thresholds plus improving SQZMOM; 120d/60d split</sup>"
            ),
            "x": 0.02,
        },
        template="plotly_white",
        height=1000,
        margin={"l": 80, "r": 45, "t": 115, "b": 65},
    )
    return fig


def build_recovery_timeline_figure(
    candles: dict[str, pd.DataFrame], timeline: dict[int, bool], events: list[dict],
    observations: list[dict], *, start_ts: int, end_ts: int,
    roc_risk_off_pct: float, sqzmom_risk_off_pct: float,
    roc_recovery_pct: float, sqzmom_recovery_pct: float,
) -> go.Figure:
    """Plot prices, risk-off spans, recovery points, and the driving signals."""
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.045,
        row_heights=[0.36, 0.36, 0.28],
        subplot_titles=("BTC-FDUSD", "ETH-FDUSD", "Closed 4h recovery signals"),
    )
    sliced: dict[str, pd.DataFrame] = {}
    colors = {"BTC-FDUSD": "#2563eb", "ETH-FDUSD": "#7c3aed"}
    for row, pair in enumerate(PAIRS, 1):
        frame = candles[pair]
        frame = frame[(frame.timestamp >= start_ts) & (frame.timestamp < end_ts)].copy()
        frame["datetime"] = pd.to_datetime(frame.timestamp, unit="s", utc=True)
        sliced[pair] = frame
        fig.add_trace(go.Scattergl(
            x=frame.datetime, y=frame.close, mode="lines", name=pair,
            line={"color": colors[pair], "width": 1},
            hovertemplate="%{x|%Y-%m-%d %H:%M UTC}<br>%{y:.4f}<extra>" + pair + "</extra>",
        ), row=row, col=1)

    for left, right, enabled in gate_intervals(timeline, start_ts, end_ts):
        if enabled:
            continue
        fig.add_vrect(
            x0=pd.to_datetime(left, unit="s", utc=True),
            x1=pd.to_datetime(right, unit="s", utc=True),
            fillcolor="#ef4444", opacity=0.13, line_width=0,
            layer="below", row="all", col=1,
        )

    btc = sliced["BTC-FDUSD"].set_index("timestamp").close
    for kind, symbol, color, label in (
        ("risk_off", "x", "#dc2626", "Risk-off: BUY disabled"),
        ("recovery", "diamond", "#16a34a", "Recovery: BUY enabled"),
    ):
        selected_events = [event for event in events
                           if event["event"] == kind and start_ts <= event["timestamp"] < end_ts]
        if not selected_events:
            continue
        event_times = [event["timestamp"] for event in selected_events]
        event_prices = [float(btc.iloc[btc.index.get_indexer([timestamp], method="nearest")[0]])
                        for timestamp in event_times]
        custom = [[event["roc_48h_pct"], event["sqzmom_pct"], event["sqzmom_color"]]
                  for event in selected_events]
        fig.add_trace(go.Scatter(
            x=pd.to_datetime(event_times, unit="s", utc=True), y=event_prices,
            mode="markers", name=label,
            marker={"symbol": symbol, "size": 10, "color": color,
                    "line": {"color": "white", "width": 1}},
            customdata=custom,
            hovertemplate=(
                "%{x|%Y-%m-%d %H:%M UTC}<br>ROC48=%{customdata[0]:.2f}%"
                "<br>SQZMOM=%{customdata[1]:.2f}%<br>color=%{customdata[2]}"
                f"<extra>{label}</extra>"
            ),
        ), row=1, col=1)

    signal_frame = pd.DataFrame(observations)
    signal_frame = signal_frame[(signal_frame.timestamp >= start_ts)
                                & (signal_frame.timestamp < end_ts)].copy()
    signal_frame["datetime"] = pd.to_datetime(signal_frame.timestamp, unit="s", utc=True)
    fig.add_trace(go.Scatter(
        x=signal_frame.datetime, y=signal_frame.roc_48h_pct, mode="lines+markers",
        name="ROC48 (%)", line={"color": "#0f766e", "width": 1.4}, marker={"size": 3},
    ), row=3, col=1)
    fig.add_trace(go.Scatter(
        x=signal_frame.datetime, y=signal_frame.sqzmom_pct, mode="lines+markers",
        name="SQZMOM (%)", line={"color": "#d97706", "width": 1.4}, marker={"size": 3},
    ), row=3, col=1)
    for threshold, color, dash, annotation in (
        (roc_risk_off_pct, "#0f766e", "dot", "ROC risk-off"),
        (sqzmom_risk_off_pct, "#d97706", "dot", "SQZ risk-off"),
        (roc_recovery_pct, "#0f766e", "dash", "ROC recovery"),
        (sqzmom_recovery_pct, "#d97706", "dash", "SQZ recovery"),
    ):
        fig.add_hline(y=threshold, line={"color": color, "dash": dash, "width": 1},
                      annotation_text=annotation, row=3, col=1)
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Price", row=2, col=1)
    fig.update_yaxes(title_text="Percent", row=3, col=1)
    fig.update_xaxes(title_text="UTC", row=3, col=1, rangeslider={"visible": True})
    fig.update_layout(
        title={
            "text": (
                "FDUSD Grid risk-off and combined recovery timeline"
                f"<br><sup>red background = BUY disabled; recovery: ROC ≥ {roc_recovery_pct:g}% "
                f"AND SQZMOM ≥ {sqzmom_recovery_pct:g}% AND improving</sup>"
            ),
            "x": 0.02,
        },
        template="plotly_white", height=1200, hovermode="x unified",
        legend={"orientation": "h", "y": 1.04, "x": 0},
        margin={"l": 75, "r": 50, "t": 125, "b": 90},
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
    selection_path = args.validation_dir / "active_selection.json"
    if selection_path.exists():
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        params = selection["parameters"]
    else:
        source_summary = json.loads(
            (args.validation_dir / "summary.json").read_text(encoding="utf-8")
        )
        params = source_summary["candidate"]
    candidate = Candidate(
        half_range=float(params["half_range"]),
        min_spread=float(params.get("minimum_spread", params.get("min_spread"))),
        take_profit=float(params["take_profit"]),
        move_threshold=float(params["move_threshold"]),
        move_cooldown_seconds=int(
            params.get("min_grid_move_seconds", params.get("move_cooldown_seconds", 1800))
        ),
    )
    rows = []
    for roc_recovery in ROC_RECOVERY_THRESHOLDS:
        for sqz_recovery in SQZMOM_RECOVERY_THRESHOLDS:
            timeline = combined_recovery_timeline(
                candles["BTC-FDUSD"],
                roc_risk_off_pct=args.roc_risk_off_pct,
                sqzmom_risk_off_pct=args.sqzmom_risk_off_pct,
                roc_recovery_pct=roc_recovery,
                sqzmom_recovery_pct=sqz_recovery,
            )
            train = segment_metrics(
                candles, timeline, candidate, start_ts, split_ts, args.maker_fee, args.taker_fee
            )
            validation = segment_metrics(
                candles, timeline, candidate, split_ts, end_ts, args.maker_fee, args.taker_fee
            )
            full = segment_metrics(
                candles, timeline, candidate, start_ts, end_ts, args.maker_fee, args.taker_fee
            )
            eligible = (
                train["risk_off_windows"] >= 1
                and validation["risk_off_windows"] >= 1
                and full["risk_off_windows"] >= 2
            )
            rows.append({
                "roc_recovery_pct": roc_recovery,
                "sqzmom_recovery_pct": sqz_recovery,
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
        raise RuntimeError("no recovery pair produced windows in both train and validation")
    selected_index = eligible.sort_values(
        ["robust_score", "mean_score", "full_risk_adjusted_score"], ascending=False
    ).index[0]
    results["selected"] = results.index == selected_index
    selected = results.loc[selected_index]
    results = results.sort_values(
        ["eligible", "robust_score", "mean_score"], ascending=False
    ).reset_index(drop=True)
    maroon_timeline = technical_buy_gate_timeline(
        candles["BTC-FDUSD"],
        roc_risk_off_pct=args.roc_risk_off_pct,
        sqzmom_risk_off_pct=args.sqzmom_risk_off_pct,
    )
    maroon_baseline = segment_metrics(
        candles, maroon_timeline, candidate, start_ts, end_ts, args.maker_fee, args.taker_fee
    )
    always_enabled = {int(timestamp): True for timestamp in candles["BTC-FDUSD"].timestamp}
    ungated_baseline = segment_metrics(
        candles, always_enabled, candidate, start_ts, end_ts, args.maker_fee, args.taker_fee
    )
    selected_timeline, selected_events, selected_observations = combined_recovery_diagnostics(
        candles["BTC-FDUSD"],
        roc_risk_off_pct=args.roc_risk_off_pct,
        sqzmom_risk_off_pct=args.sqzmom_risk_off_pct,
        roc_recovery_pct=float(selected["roc_recovery_pct"]),
        sqzmom_recovery_pct=float(selected["sqzmom_recovery_pct"]),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output_dir / "recovery_parameter_search.csv", index=False)
    payload = {
        "schema_version": "grid-roc-sqz-recovery-search-v1",
        "period": {
            "start_ts": start_ts,
            "split_ts": split_ts,
            "end_ts": end_ts,
            "days": args.days,
            "train_days": args.train_days,
            "validation_days": args.days - args.train_days,
        },
        "risk_off": {
            "roc_pct": args.roc_risk_off_pct,
            "sqzmom_pct": args.sqzmom_risk_off_pct,
            "rule": "ROC48 <= threshold AND SQZMOM <= threshold",
        },
        "recovery": {
            "rule": (
                "risk-off condition cleared AND ROC48 >= recovery threshold AND "
                "SQZMOM >= recovery threshold AND SQZMOM improving"
            ),
            "roc_thresholds": ROC_RECOVERY_THRESHOLDS,
            "sqzmom_thresholds": SQZMOM_RECOVERY_THRESHOLDS,
        },
        "candidate": {**asdict(candidate), "levels": candidate.levels},
        "risk_breakers_enabled": False,
        "score": "min(30-day-normalized return - 0.75*abs(max drawdown)) across train and validation",
        "selected": selected.to_dict(),
        "baselines": {
            "maroon_recovery": maroon_baseline,
            "buy_always_enabled": ungated_baseline,
        },
        "evaluated_combinations": len(results),
        "eligible_combinations": int(results.eligible.sum()),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    figure = build_figure(results, selected)
    figure.write_html(
        args.output_dir / "recovery_parameter_heatmaps.html",
        include_plotlyjs=True,
        full_html=True,
    )
    figure.write_json(args.output_dir / "recovery_parameter_heatmaps.plotly.json")
    timeline_figure = build_recovery_timeline_figure(
        candles, selected_timeline, selected_events, selected_observations,
        start_ts=start_ts, end_ts=end_ts,
        roc_risk_off_pct=args.roc_risk_off_pct,
        sqzmom_risk_off_pct=args.sqzmom_risk_off_pct,
        roc_recovery_pct=float(selected["roc_recovery_pct"]),
        sqzmom_recovery_pct=float(selected["sqzmom_recovery_pct"]),
    )
    timeline_figure.write_html(
        args.output_dir / "risk_off_recovery_timeline.html",
        include_plotlyjs=True, full_html=True,
    )
    timeline_figure.write_json(args.output_dir / "risk_off_recovery_timeline.plotly.json")
    pd.DataFrame(selected_events).to_csv(
        args.output_dir / "risk_off_recovery_events.csv", index=False,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

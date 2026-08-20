#!/usr/bin/env python3
"""Create a self-contained Plotly chart with exact Risk-off entry/exit times."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


PAIRS = ("BTC-FDUSD", "ETH-FDUSD")
PAIR_COLORS = {"BTC-FDUSD": "#2563EB", "ETH-FDUSD": "#D97706"}
ENTRY_COLOR = "#9A6700"
RECOVERY_COLOR = "#334155"
RESET_COLOR = "#7C3AED"


def utc_text(ts: int) -> str:
    return pd.to_datetime(int(ts), unit="s", utc=True).strftime("%Y-%m-%d %H:%M UTC")


def build_timing_events(states: pd.DataFrame, intervals: pd.DataFrame) -> pd.DataFrame:
    """Return every entry and every effective exit, including weekly resets."""
    rows: list[dict[str, Any]] = []
    ordered = states.sort_values(["fold", "pair", "signal_ts"])
    for interval in intervals.itertuples(index=False):
        pair_states = ordered[(ordered.fold == int(interval.fold)) & (ordered.pair == interval.pair)]
        entry_row = pair_states[pair_states.signal_ts == int(interval.start_ts)]
        if entry_row.empty:
            raise RuntimeError(f"Missing entry state for fold={interval.fold}, pair={interval.pair}")
        entry_item = entry_row.iloc[0]
        rows.append({
            "fold": int(interval.fold), "pair": interval.pair, "event": "ENTER",
            "timestamp": int(interval.start_ts), "time_utc": utc_text(interval.start_ts),
            "probability": float(entry_item.probability),
            "entry_threshold": float(entry_item.entry_threshold),
            "recovery_threshold": float(entry_item.recovery_threshold),
            "duration_hours": float(interval.duration_hours),
            "end_reason": str(interval.end_reason),
        })
        if interval.end_reason == "recover":
            exit_row = pair_states[pair_states.signal_ts == int(interval.end_ts)]
            if exit_row.empty:
                raise RuntimeError(f"Missing recovery state for fold={interval.fold}, pair={interval.pair}")
            exit_item = exit_row.iloc[0]
            event, probability = "EXIT_RECOVER", float(exit_item.probability)
        else:
            before = pair_states[pair_states.signal_ts < int(interval.end_ts)].tail(1)
            if before.empty:
                raise RuntimeError(f"Missing pre-reset state for fold={interval.fold}, pair={interval.pair}")
            event, probability = "EXIT_WEEK_RESET", float(before.iloc[0].probability)
        rows.append({
            "fold": int(interval.fold), "pair": interval.pair, "event": event,
            "timestamp": int(interval.end_ts), "time_utc": utc_text(interval.end_ts),
            "probability": probability,
            "entry_threshold": float(entry_item.entry_threshold),
            "recovery_threshold": float(entry_item.recovery_threshold),
            "duration_hours": float(interval.duration_hours),
            "end_reason": str(interval.end_reason),
        })
    result = pd.DataFrame(rows).sort_values(["timestamp", "pair", "event"]).reset_index(drop=True)
    if result.empty or len(result) != 2 * len(intervals):
        raise AssertionError("Each Risk-off interval must have exactly one entry and one effective exit")
    return result


def add_event_markers(fig: go.Figure, events: pd.DataFrame, pair: str, row: int) -> None:
    definitions = {
        "ENTER": ("triangle-up", ENTRY_COLOR, "进入Risk-off"),
        "EXIT_RECOVER": ("circle-open", RECOVERY_COLOR, "模型恢复BUY"),
        "EXIT_WEEK_RESET": ("square-open", RESET_COLOR, "周度重置退出"),
    }
    for event, (symbol, color, label) in definitions.items():
        frame = events[(events.pair == pair) & (events.event == event)].copy()
        if frame.empty:
            continue
        custom = np.column_stack([
            frame.time_utc, frame.fold, frame.entry_threshold,
            frame.recovery_threshold, frame.duration_hours, frame.end_reason,
        ])
        fig.add_trace(go.Scatter(
            x=pd.to_datetime(frame.timestamp, unit="s", utc=True), y=frame.probability,
            mode="markers", name=label, legendgroup=event,
            showlegend=(row == 1),
            marker={"symbol": symbol, "size": 11, "color": color, "line": {"color": color, "width": 2}},
            customdata=custom,
            hovertemplate=(
                f"<b>{pair} · {label}</b><br>"
                "时间 %{customdata[0]}<br>周折 W%{customdata[1]}<br>"
                "概率 %{y:.6f}<br>进入阈值 %{customdata[2]:.6f}<br>"
                "恢复阈值 %{customdata[3]:.6f}<br>区间时长 %{customdata[4]:.1f}h<br>"
                "结束原因 %{customdata[5]}<extra></extra>"
            ),
        ), row=row, col=1)


def build_figure(states: pd.DataFrame, intervals: pd.DataFrame, events: pd.DataFrame) -> go.Figure:
    figure = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.09,
        subplot_titles=("BTC-FDUSD 风险概率", "ETH-FDUSD 风险概率"),
    )
    for row, pair in enumerate(PAIRS, 1):
        frame = states[states.pair == pair].sort_values("signal_ts").copy()
        time = pd.to_datetime(frame.signal_ts, unit="s", utc=True)
        common_hover = np.column_stack([
            frame.fold, frame.consecutive_recovery_bars,
            frame.risk_off_active, frame.reason,
        ])
        figure.add_trace(go.Scatter(
            x=time, y=frame.probability, mode="lines", name=f"{pair} probability",
            legendgroup=pair, line={"color": PAIR_COLORS[pair], "width": 1.6},
            customdata=common_hover,
            hovertemplate=(
                f"<b>{pair}</b><br>%{{x|%Y-%m-%d %H:%M UTC}}<br>"
                "概率 %{y:.6f}<br>周折 W%{customdata[0]}<br>"
                "连续恢复计数 %{customdata[1]}<br>Risk-off %{customdata[2]}<br>"
                "原因 %{customdata[3]}<extra></extra>"
            ),
        ), row=row, col=1)
        figure.add_trace(go.Scatter(
            x=time, y=frame.entry_threshold, mode="lines", name=f"{pair} entry threshold",
            legendgroup=f"{pair}-thresholds", line={"color": "#475569", "width": 1.2, "dash": "dash"},
            hovertemplate="进入阈值 %{y:.6f}<br>%{x|%Y-%m-%d %H:%M UTC}<extra></extra>",
        ), row=row, col=1)
        figure.add_trace(go.Scatter(
            x=time, y=frame.recovery_threshold, mode="lines", name=f"{pair} recovery threshold",
            legendgroup=f"{pair}-thresholds", line={"color": "#7C3AED", "width": 1.2, "dash": "dot"},
            hovertemplate="恢复阈值 %{y:.6f}<br>%{x|%Y-%m-%d %H:%M UTC}<extra></extra>",
        ), row=row, col=1)
        for interval in intervals[intervals.pair == pair].itertuples(index=False):
            figure.add_vrect(
                x0=pd.to_datetime(interval.start_ts, unit="s", utc=True),
                x1=pd.to_datetime(interval.end_ts, unit="s", utc=True),
                fillcolor=PAIR_COLORS[pair], opacity=0.09, line_width=0,
                layer="below", row=row, col=1,
            )
        add_event_markers(figure, events, pair, row)

    figure.update_yaxes(title_text="Risk probability", range=[0, 1], tickformat=".2f", row=1, col=1)
    figure.update_yaxes(title_text="Risk probability", range=[0, 1], tickformat=".2f", row=2, col=1)
    figure.update_xaxes(title_text="UTC", rangeslider={"visible": True, "thickness": 0.07}, row=2, col=1)
    figure.update_layout(
        title={
            "text": (
                "XGBoost Risk-off 精确进入/退出时间"
                "<br><sup>蓝/橙线=概率 · 灰虚线=进入 · 紫点线=恢复"
                "<br>▲进入 · ○模型恢复 · □周度重置 · UTC</sup>"
            ),
            "x": 0.02, "xanchor": "left",
        },
        template="plotly_white", height=940,
        margin={"l": 78, "r": 50, "t": 135, "b": 80},
        showlegend=False,
        hovermode="closest", font={"family": "Arial, Microsoft YaHei, sans-serif", "color": "#17202A"},
    )
    return figure


def generate_plot(states_path: Path, intervals_path: Path, output_html: Path, events_csv: Path) -> dict[str, int]:
    states = pd.read_csv(states_path)
    intervals = pd.read_csv(intervals_path)
    required_states = {"fold", "pair", "signal_ts", "probability", "entry_threshold", "recovery_threshold", "risk_off_active", "consecutive_recovery_bars", "reason"}
    required_intervals = {"fold", "pair", "start_ts", "end_ts", "duration_hours", "end_reason"}
    if required_states.difference(states.columns) or required_intervals.difference(intervals.columns):
        raise RuntimeError("Risk timing inputs do not match the expected schema")
    events = build_timing_events(states, intervals)
    events.to_csv(events_csv, index=False)
    figure = build_figure(states, intervals, events)
    output_html.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(
        output_html, include_plotlyjs=True, full_html=True,
        config={"responsive": True, "displaylogo": False, "scrollZoom": True},
    )
    return {
        "states": len(states), "intervals": len(intervals), "entries": int((events.event == "ENTER").sum()),
        "model_recoveries": int((events.event == "EXIT_RECOVER").sum()),
        "weekly_resets": int((events.event == "EXIT_WEEK_RESET").sum()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("results/backtests/xgboost_grid_risk_gate_v1"))
    args = parser.parse_args()
    counts = generate_plot(
        args.input_dir / "revalidation_risk_states.csv.gz",
        args.input_dir / "revalidation_risk_off_intervals.csv",
        args.input_dir / "risk_gate_entry_exit_plotly.html",
        args.input_dir / "risk_gate_entry_exit_events.csv",
    )
    print(counts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

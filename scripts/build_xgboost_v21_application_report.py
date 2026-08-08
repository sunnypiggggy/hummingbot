#!/usr/bin/env python3
"""Replay the exact frozen v21 bundle and render its application-equivalent Plotly."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import retrain_xgboost_long_risk_gate_250d_v19 as research
from compare_independent_gate_ml_stops import load_candles as load_feature_candles
from xgboost_long_risk_gate_v21 import (
    MODEL_VERSION, PAIRS, build_inference_panel, run_bundle_strategy,
    strategy_schema_sha256, validate_strategy_bundle,
)


ROOT = Path("results/backtests/xgboost_grid_long_risk_gate_v21_250d")
PACKAGE = ROOT / "shadow_package"
SOURCE = Path("results/backtests/eth_xgboost_long_risk_gate_v15_250d")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, default=ROOT)
    parser.add_argument("--package-dir", type=Path, default=PACKAGE)
    parser.add_argument("--source-dir", type=Path, default=SOURCE)
    return parser.parse_args()


def intervals_from_states(states: pd.DataFrame) -> pd.DataFrame:
    intervals: list[dict[str, Any]] = []
    for pair in PAIRS:
        start: int | None = None
        for row in states[states.pair.eq(pair)].sort_values("signal_ts").itertuples(index=False):
            if row.transition == "enter":
                if start is not None:
                    raise RuntimeError(f"{pair} entered twice without recovery")
                start = int(row.signal_ts)
            elif row.transition == "recover":
                if start is None:
                    raise RuntimeError(f"{pair} recovered without an open interval")
                intervals.append({"pair": pair, "start_ts": start, "end_ts": int(row.signal_ts),
                                  "duration_hours": (int(row.signal_ts) - start) / 3600,
                                  "end_reason": "bundle_adaptive_relief"})
                start = None
        if start is not None:
            intervals.append({"pair": pair, "start_ts": start, "end_ts": research.END_TS,
                              "duration_hours": (research.END_TS - start) / 3600,
                              "end_reason": "research_period_end"})
    return pd.DataFrame(intervals)


def timelines_from_states(states: pd.DataFrame) -> dict[str, dict[int, bool]]:
    timelines = {pair: {timestamp: False for timestamp in range(research.START_TS, research.END_TS, 300)}
                 for pair in PAIRS}
    for pair in PAIRS:
        rows = states[states.pair.eq(pair)].sort_values("signal_ts").reset_index(drop=True)
        for index, row in rows.iterrows():
            start = max(int(row.signal_ts), research.START_TS)
            end = min(int(rows.iloc[index + 1].signal_ts) if index + 1 < len(rows) else research.END_TS,
                      research.END_TS)
            for timestamp in range(start, end, 300):
                timelines[pair][timestamp] = bool(row.recommended_buy_enabled)
    return timelines


def render_plot(states: pd.DataFrame, intervals: pd.DataFrame,
                candles: Mapping[str, pd.DataFrame], summary: Mapping[str, Any], target: Path) -> None:
    figure = make_subplots(
        rows=4, cols=1, shared_xaxes=True, vertical_spacing=.035,
        subplot_titles=("BTC-FDUSD price", "BTC frozen-bundle probability",
                        "ETH-FDUSD price", "ETH frozen-bundle probability"),
    )
    shapes: dict[str, list[int]] = {pair: [] for pair in PAIRS}
    for offset, pair in enumerate(PAIRS):
        price_row, probability_row = 1 + offset * 2, 2 + offset * 2
        price = candles[pair].iloc[::12].copy()
        state = states[states.pair.eq(pair)].sort_values("signal_ts")
        figure.add_trace(go.Scatter(
            x=pd.to_datetime(price.timestamp, unit="s", utc=True), y=price.close,
            name=f"{pair} close", line={"width": 1.2}), row=price_row, col=1)
        figure.add_trace(go.Scatter(
            x=pd.to_datetime(state.signal_ts, unit="s", utc=True), y=state.probability,
            name=f"{pair} application probability", line={"width": 1}), row=probability_row, col=1)
        figure.add_trace(go.Scatter(
            x=pd.to_datetime(state.signal_ts, unit="s", utc=True), y=state.entry_threshold,
            name=f"{pair} frozen absolute threshold", line={"dash": "dot", "width": 1}),
            row=probability_row, col=1)
        for number, interval in enumerate(intervals[intervals.pair.eq(pair)].itertuples(index=False)):
            figure.add_vrect(
                x0=pd.to_datetime(interval.start_ts, unit="s", utc=True),
                x1=pd.to_datetime(interval.end_ts, unit="s", utc=True),
                fillcolor="#f59e0b", opacity=.16, line_width=0, row=price_row, col=1,
                name=f"{pair} Risk-off", legendgroup=f"{pair}-risk-off", showlegend=number == 0,
            )
            shapes[pair].append(len(figure.layout.shapes) - 1)
        events = state[state.transition.isin(("enter", "recover"))]
        for transition, symbol, color in (("enter", "triangle-down", "#dc2626"),
                                           ("recover", "triangle-up", "#16a34a")):
            points = events[events.transition.eq(transition)]
            if points.empty:
                continue
            marks = pd.merge_asof(
                points[["signal_ts"]].sort_values("signal_ts"),
                price[["timestamp", "close"]].sort_values("timestamp"),
                left_on="signal_ts", right_on="timestamp", direction="backward",
            )
            figure.add_trace(go.Scatter(
                x=pd.to_datetime(marks.signal_ts, unit="s", utc=True), y=marks.close,
                mode="markers", marker={"symbol": symbol, "color": color, "size": 9},
                name=f"{pair} {transition}"), row=price_row, col=1)
    for _, start, end in research.ANCHOR_WINDOWS:
        for row in (1, 3):
            figure.add_vrect(
                x0=pd.to_datetime(start, unit="s", utc=True),
                x1=pd.to_datetime(end, unit="s", utc=True), fillcolor="#7c3aed",
                opacity=.08, line_width=1, line_dash="dash", row=row, col=1)

    def visibility(pair: str, enabled: bool) -> dict[str, bool]:
        return {f"shapes[{index}].visible": enabled for index in shapes[pair]}

    metrics = summary["grid_metrics"]
    figure.update_layout(
        height=1150, template="plotly_white", hovermode="x unified", margin={"t": 155},
        title=(f"v21 frozen application bundle exact replay | PnL {metrics['oos_pnl_fdusd']:.3f} FDUSD | "
               f"DD {metrics['stitched_max_drawdown_pct']:.3f}% | absolute thresholds | NO-GO"),
        updatemenus=[
            {"type": "buttons", "direction": "right", "x": 0, "y": 1.105, "buttons": [
                {"label": "BTC Risk-off ON", "method": "relayout", "args": [visibility("BTC-FDUSD", True)]},
                {"label": "BTC Risk-off OFF", "method": "relayout", "args": [visibility("BTC-FDUSD", False)]},
            ]},
            {"type": "buttons", "direction": "right", "x": .36, "y": 1.105, "buttons": [
                {"label": "ETH Risk-off ON", "method": "relayout", "args": [visibility("ETH-FDUSD", True)]},
                {"label": "ETH Risk-off OFF", "method": "relayout", "args": [visibility("ETH-FDUSD", False)]},
            ]},
        ],
        annotations=[*list(figure.layout.annotations), {
            "text": "This chart and the application use the same serialized models, absolute thresholds, feature order and state machine.",
            "xref": "paper", "yref": "paper", "x": 0, "y": 1.145,
            "showarrow": False, "xanchor": "left", "font": {"size": 12, "color": "#475569"},
        }],
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(target, include_plotlyjs=True, full_html=True)


def main() -> int:
    args = parse_args()
    lock_path = args.package_dir / "shadow_lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    model_path = Path(lock["model_path"])
    if not model_path.exists():
        model_path = args.package_dir / "models" / model_path.name
    if research.sha256_file(model_path) != lock["model_sha256"]:
        raise RuntimeError("frozen model hash mismatch")
    bundle = joblib.load(model_path)
    validate_strategy_bundle(bundle)
    if lock["strategy_schema_sha256"] != strategy_schema_sha256():
        raise RuntimeError("strategy lock hash mismatch")

    feature_candles, _ = load_feature_candles(args.source_dir / "extended_candles")
    panel = build_inference_panel(feature_candles)
    panel = panel[panel.signal_ts.between(research.START_TS, research.END_TS, inclusive="left")].copy()
    state_parts = []
    for pair in PAIRS:
        rows = panel[panel.pair.eq(pair)].copy()
        pair_states, _ = run_bundle_strategy(rows, pair=pair, pair_bundle=bundle["pairs"][pair])
        state_parts.append(pair_states)
    states = pd.concat(state_parts, ignore_index=True).sort_values(["pair", "signal_ts"])
    intervals = intervals_from_states(states)
    events = states[states.transition.isin(("enter", "recover"))].copy()

    grid_candles = research.load_candles(args.source_dir)
    selections = pd.read_csv(args.source_dir / "grid_selections.csv")
    detail = research.exact_replay(
        grid_candles, selections, research.combine_timelines(timelines_from_states(states)),
        return_details=True,
    )
    prefix = args.result_dir / "application_bundle"
    prefix.mkdir(parents=True, exist_ok=True)
    states.to_csv(prefix / "risk_states.csv.gz", index=False, compression="gzip")
    events.to_csv(prefix / "risk_events.csv", index=False)
    intervals.to_csv(prefix / "risk_intervals.csv", index=False)
    for name in ("weekly", "pairs", "equity", "trades"):
        detail[name].to_csv(prefix / f"grid_{name}.csv.gz", index=False, compression="gzip")

    research_summary = json.loads((args.result_dir / "summary.json").read_text(encoding="utf-8"))
    report_summary = {
        "schema": "xgboost-v21-application-bundle-replay-v1",
        "model_version": MODEL_VERSION,
        "model_sha256": lock["model_sha256"],
        "strategy_schema_sha256": lock["strategy_schema_sha256"],
        "probability_semantics": "final_refit_model_with_frozen_absolute_threshold",
        "grid_metrics": detail["summary"],
        "research_walk_forward_metrics": research_summary.get("metrics", {}),
        "pair_intervals": {
            pair: {"count": int(intervals.pair.eq(pair).sum()),
                   "hours": float(intervals[intervals.pair.eq(pair)].duration_hours.sum())}
            for pair in PAIRS
        },
        "deployment_allowed": False,
        "promotion_authorized": False,
        "historical_verdict": "NO-GO",
    }
    research.atomic_json(prefix / "summary.json", report_summary)
    main_report = args.result_dir / f"{MODEL_VERSION}_plotly.html"
    research_report = args.result_dir / f"{MODEL_VERSION}_research_walk_forward_plotly.html"
    if main_report.exists() and not research_report.exists():
        shutil.copy2(main_report, research_report)
    render_plot(states, intervals, grid_candles, report_summary, main_report)
    report_summary.update({"plotly": main_report.as_posix(),
                           "research_walk_forward_plotly": research_report.as_posix()})
    research.atomic_json(prefix / "summary.json", report_summary)
    research_summary.update({
        "plotly": main_report.as_posix(),
        "plotly_semantics": "frozen_application_bundle_exact_replay",
        "research_walk_forward_plotly": research_report.as_posix(),
        "application_bundle_summary": (prefix / "summary.json").as_posix(),
    })
    research.atomic_json(args.result_dir / "summary.json", research_summary)
    lock.update({
        "application_plotly": main_report.as_posix(),
        "application_plotly_sha256": research.sha256_file(main_report),
        "application_bundle_summary": (prefix / "summary.json").as_posix(),
        "application_bundle_summary_sha256": research.sha256_file(prefix / "summary.json"),
        "application_risk_states_sha256": research.sha256_file(prefix / "risk_states.csv.gz"),
        "application_risk_intervals_sha256": research.sha256_file(prefix / "risk_intervals.csv"),
        "application_replay_semantics": "final_refit_model_frozen_absolute_threshold_shared_bundle_runner",
    })
    research.atomic_json(lock_path, lock)
    print(json.dumps(report_summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

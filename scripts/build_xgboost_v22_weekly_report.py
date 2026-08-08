#!/usr/bin/env python3
"""Build the application-equivalent v22 weekly-bundle replay and Plotly."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import joblib
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import retrain_xgboost_long_risk_gate_250d_v19 as research
import xgboost_long_risk_gate_v22 as v22


ROOT = Path("results/backtests/xgboost_grid_long_risk_gate_v22_weekly_250d")
PACKAGE = ROOT / "shadow_package"
SOURCE = Path("results/backtests/eth_xgboost_long_risk_gate_v15_250d")
V19 = Path("results/backtests/xgboost_grid_long_risk_gate_v19_250d")
V21 = Path("results/backtests/xgboost_grid_long_risk_gate_v21_250d")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, default=ROOT)
    parser.add_argument("--package-dir", type=Path, default=PACKAGE)
    parser.add_argument("--source-dir", type=Path, default=SOURCE)
    parser.add_argument("--v19-dir", type=Path, default=V19)
    parser.add_argument("--v21-dir", type=Path, default=V21)
    return parser.parse_args()


def intervals_from_states(states: pd.DataFrame) -> pd.DataFrame:
    result: list[dict[str, Any]] = []
    for pair in v22.PAIRS:
        start: int | None = None
        for row in states[states.pair.eq(pair)].sort_values("signal_ts").itertuples(index=False):
            if row.transition == "enter":
                if start is not None: raise RuntimeError(f"{pair} duplicate enter")
                start = int(row.signal_ts)
            elif row.transition == "recover":
                if start is None: raise RuntimeError(f"{pair} recover without enter")
                result.append({"pair": pair, "start_ts": start, "end_ts": int(row.signal_ts),
                               "duration_hours": (int(row.signal_ts) - start) / 3600,
                               "end_reason": "adaptive_structural_relief"})
                start = None
        if start is not None:
            result.append({"pair": pair, "start_ts": start, "end_ts": research.END_TS,
                           "duration_hours": (research.END_TS - start) / 3600,
                           "end_reason": "research_period_end"})
    return pd.DataFrame(result)


def timelines(states: pd.DataFrame) -> dict[str, dict[int, bool]]:
    output = {pair: {ts: False for ts in range(research.START_TS, research.END_TS, 300)}
              for pair in v22.PAIRS}
    for pair in v22.PAIRS:
        rows = states[states.pair.eq(pair)].sort_values("signal_ts").reset_index(drop=True)
        for index, row in rows.iterrows():
            right = int(rows.iloc[index + 1].signal_ts) if index + 1 < len(rows) else research.END_TS
            for ts in range(max(research.START_TS, int(row.signal_ts)), min(right, research.END_TS), 300):
                output[pair][ts] = bool(row.recommended_buy_enabled)
    return output


def render(states: pd.DataFrame, intervals: pd.DataFrame, candles: Mapping[str, pd.DataFrame],
           summary: Mapping[str, Any], target: Path) -> None:
    figure = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=.035,
        subplot_titles=("BTC-FDUSD price", "BTC weekly walk-forward probability",
                        "ETH-FDUSD price", "ETH weekly walk-forward probability"))
    shape_indices = {pair: [] for pair in v22.PAIRS}
    for offset, pair in enumerate(v22.PAIRS):
        price_row, probability_row = 1 + 2 * offset, 2 + 2 * offset
        price = candles[pair].iloc[::12].copy()
        state = states[states.pair.eq(pair)].sort_values("signal_ts")
        figure.add_trace(go.Scatter(x=pd.to_datetime(price.timestamp, unit="s", utc=True), y=price.close,
            name=f"{pair} close", line={"width": 1.2}), row=price_row, col=1)
        figure.add_trace(go.Scatter(x=pd.to_datetime(state.signal_ts, unit="s", utc=True), y=state.probability,
            customdata=state[["fold", "entry_threshold"]],
            hovertemplate="p=%{y:.6f}<br>fold=%{customdata[0]}<br>threshold=%{customdata[1]:.6f}<extra></extra>",
            name=f"{pair} weekly probability", line={"width": 1}), row=probability_row, col=1)
        figure.add_trace(go.Scatter(x=pd.to_datetime(state.signal_ts, unit="s", utc=True), y=state.entry_threshold,
            name=f"{pair} fold-local threshold", line={"dash": "dot", "width": 1}), row=probability_row, col=1)
        for number, item in enumerate(intervals[intervals.pair.eq(pair)].itertuples(index=False)):
            figure.add_vrect(x0=pd.to_datetime(item.start_ts, unit="s", utc=True),
                x1=pd.to_datetime(item.end_ts, unit="s", utc=True), fillcolor="#f59e0b", opacity=.16,
                line_width=0, row=price_row, col=1, name=f"{pair} Risk-off",
                legendgroup=f"{pair}-risk", showlegend=number == 0)
            shape_indices[pair].append(len(figure.layout.shapes) - 1)
        events = state[state.transition.isin(("enter", "recover"))]
        for transition, symbol, color in (("enter", "triangle-down", "#dc2626"),
                                           ("recover", "triangle-up", "#16a34a")):
            points = events[events.transition.eq(transition)]
            if points.empty: continue
            marks = pd.merge_asof(points[["signal_ts"]].sort_values("signal_ts"),
                price[["timestamp", "close"]].sort_values("timestamp"), left_on="signal_ts",
                right_on="timestamp", direction="backward")
            figure.add_trace(go.Scatter(x=pd.to_datetime(marks.signal_ts, unit="s", utc=True), y=marks.close,
                mode="markers", marker={"symbol": symbol, "color": color, "size": 9},
                name=f"{pair} {transition}"), row=price_row, col=1)
    for _, start, end in research.ANCHOR_WINDOWS:
        for row in (1, 3):
            figure.add_vrect(x0=pd.to_datetime(start, unit="s", utc=True),
                x1=pd.to_datetime(end, unit="s", utc=True), fillcolor="#7c3aed", opacity=.08,
                line_width=1, line_dash="dash", row=row, col=1)
    def visible(pair: str, enabled: bool) -> dict[str, bool]:
        return {f"shapes[{index}].visible": enabled for index in shape_indices[pair]}
    metrics = summary["grid_metrics"]
    figure.update_layout(height=1150, template="plotly_white", hovermode="x unified", margin={"t": 155},
        title=(f"v22 weekly walk-forward bundle exact replay | PnL {metrics['oos_pnl_fdusd']:.3f} FDUSD | "
               f"DD {metrics['stitched_max_drawdown_pct']:.3f}% | fold-local thresholds | NO-GO"),
        updatemenus=[
            {"type": "buttons", "direction": "right", "x": 0, "y": 1.105, "buttons": [
                {"label": "BTC Risk-off ON", "method": "relayout", "args": [visible("BTC-FDUSD", True)]},
                {"label": "BTC Risk-off OFF", "method": "relayout", "args": [visible("BTC-FDUSD", False)]}]},
            {"type": "buttons", "direction": "right", "x": .36, "y": 1.105, "buttons": [
                {"label": "ETH Risk-off ON", "method": "relayout", "args": [visible("ETH-FDUSD", True)]},
                {"label": "ETH Risk-off OFF", "method": "relayout", "args": [visible("ETH-FDUSD", False)]}]},
        ], annotations=[*list(figure.layout.annotations), {
            "text": "Plotly and application replay use the same signed weekly models, fold-local thresholds and continuous state machine.",
            "xref": "paper", "yref": "paper", "x": 0, "y": 1.145, "showarrow": False,
            "xanchor": "left", "font": {"size": 12, "color": "#475569"}}])
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(target, include_plotlyjs=True, full_html=True)


def main() -> int:
    args = parse_args(); output = args.result_dir / "application_bundle"; output.mkdir(parents=True, exist_ok=True)
    lock_path = args.package_dir / "shadow_lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8")); model_path = Path(lock["model_path"])
    if not model_path.exists(): model_path = args.package_dir / "models" / model_path.name
    if research.sha256_file(model_path) != lock["model_sha256"]: raise RuntimeError("v22 bundle hash mismatch")
    bundle = joblib.load(model_path); v22.validate_weekly_bundle(bundle)
    panel = pd.read_csv(args.v19_dir / "feature_panel.csv.gz")
    panel = panel[panel.signal_ts.between(research.START_TS, research.END_TS, inclusive="left")]
    parts = []
    for pair in v22.PAIRS:
        state, _ = v22.run_weekly_bundle_strategy(panel[panel.pair.eq(pair)], pair=pair,
                                                   pair_bundle=bundle["pairs"][pair])
        parts.append(state)
    states = pd.concat(parts, ignore_index=True).sort_values(["pair", "signal_ts"])
    old = pd.read_csv(args.v21_dir / "final_risk_states.csv.gz").sort_values(["pair", "signal_ts"])
    parity = {
        "risk_state_mismatches": int((states.risk_off_active.to_numpy() != old.risk_off_active.to_numpy()).sum()),
        "transition_mismatches": int((states.transition.to_numpy() != old.transition.to_numpy()).sum()),
        "maximum_probability_absolute_error": float((states.probability.to_numpy() - old.probability.to_numpy()).__abs__().max()),
        "maximum_threshold_absolute_delta": float((states.entry_threshold.to_numpy() - old.entry_threshold.to_numpy()).__abs__().max()),
    }
    if parity["risk_state_mismatches"] or parity["transition_mismatches"]:
        raise AssertionError(f"v22 is not state-equivalent to old weekly replay: {parity}")
    intervals = intervals_from_states(states); events = states[states.transition.isin(("enter", "recover"))]
    candles = research.load_candles(args.source_dir); selections = pd.read_csv(args.source_dir / "grid_selections.csv")
    detail = research.exact_replay(candles, selections, research.combine_timelines(timelines(states)), return_details=True)
    states.to_csv(output / "risk_states.csv.gz", index=False, compression="gzip")
    events.to_csv(output / "risk_events.csv", index=False); intervals.to_csv(output / "risk_intervals.csv", index=False)
    for name in ("weekly", "pairs", "equity", "trades"):
        detail[name].to_csv(output / f"grid_{name}.csv.gz", index=False, compression="gzip")
    summary = {"schema": "xgboost-v22-weekly-application-replay-v1", "model_version": v22.MODEL_VERSION,
        "model_sha256": lock["model_sha256"], "strategy_schema_sha256": lock["strategy_schema_sha256"],
        "probability_semantics": lock["probability_semantics"], "grid_metrics": detail["summary"],
        "old_walk_forward_parity": parity, "weeks_per_pair": lock["weeks_per_pair"],
        "deployment_allowed": False, "promotion_authorized": False, "historical_verdict": "NO-GO"}
    target = args.result_dir / f"{v22.MODEL_VERSION}_plotly.html"; render(states, intervals, candles, summary, target)
    summary["plotly"] = target.as_posix(); research.atomic_json(output / "summary.json", summary)
    lock.update({"application_plotly": target.as_posix(), "application_plotly_sha256": research.sha256_file(target),
        "application_bundle_summary": (output / "summary.json").as_posix(),
        "application_bundle_summary_sha256": research.sha256_file(output / "summary.json"),
        "application_replay_semantics": "signed_weekly_models_fold_local_thresholds_continuous_state"})
    research.atomic_json(lock_path, lock)
    print(json.dumps(summary, ensure_ascii=False, indent=2)); return 0


if __name__ == "__main__":
    raise SystemExit(main())

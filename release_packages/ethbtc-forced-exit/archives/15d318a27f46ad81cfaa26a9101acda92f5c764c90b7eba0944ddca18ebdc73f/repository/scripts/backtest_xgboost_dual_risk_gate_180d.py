#!/usr/bin/env python3
"""180-day diagnostic replay of two independent XGBoost Grid BUY gates.

The long channel predicts a persistent 72-hour decline.  The short channel
predicts a one-hour spike down or a fast adverse move within 24 hours.  Both
channels are trained and calibrated independently with mature labels only.
The combined strategy disables an ordinary pair BUY whenever either channel
is Risk-off; it never requests an immediate sell.
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import (
    balanced_accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

import backtest_xgboost_long_risk_gate_180d as base
from compare_independent_gate_ml_stops import (
    ALL_FEATURES,
    FIVE_MINUTES,
    HOUR,
    PAIRS,
    hourly_bars,
    load_candles,
)
from grid_xgboost_risk_gate import PairGateState, advance_pair_gate
from search_fdusd_inventory_exit import aggregate_rows
from tune_xgboost_momentum_stop_v2 import (
    fit_one_group,
    sha256_file,
    split_mature_training,
    write_json,
    xgb_configurations,
)


MODEL_VERSION = "xgboost-grid-dual-risk-gate-v4"
PERIOD = "diagnostic_180d"
DEFAULT_OUTPUT = Path("results/backtests/xgboost_dual_risk_gate_180d_v4")
SOURCE_LONG_PANEL = Path(
    "results/backtests/xgboost_persistent_risk_gate_180d_v3/multi_horizon_feature_panel.csv.gz"
)

STRATEGIES: dict[str, dict[str, Any]] = {
    "long_persistent_72h": {
        "label": "72h持续下跌",
        "target": "target_long",
        "ready": "label_ready_ts_long",
        "entry_quantile": 0.75,
        "recovery_quantile": 0.65,
        "color": "#C2410C",
        "fill": "rgba(194,65,12,0.11)",
        "enter_symbol": "triangle-down",
        "recover_symbol": "triangle-up",
    },
    "short_spike_1h_24h": {
        "label": "1h/24h快速下跌",
        "target": "target_short",
        "ready": "label_ready_ts_short",
        "entry_quantile": 0.90,
        "recovery_quantile": 0.80,
        "color": "#2563EB",
        "fill": "rgba(37,99,235,0.09)",
        "enter_symbol": "diamond",
        "recover_symbol": "circle-open",
    },
}

SCENARIOS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("XGBoost long only", ("long_persistent_72h",)),
    ("XGBoost short only", ("short_spike_1h_24h",)),
    ("XGBoost dual OR gate", tuple(STRATEGIES)),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=Path("data/backtesting_candles"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--source-weekly-results", type=Path, default=base.DEFAULT_WEEKLY_RESULTS)
    parser.add_argument("--source-long-panel", type=Path, default=SOURCE_LONG_PANEL)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def augment_dual_targets(
    panel: pd.DataFrame, candles: Mapping[str, pd.DataFrame]
) -> pd.DataFrame:
    """Add the fixed short-horizon label without changing momentum features."""
    item = panel.copy()
    item["target_long"] = item.target
    item["label_ready_ts_long"] = item.signal_ts.astype("int64") + 72 * HOUR
    parts = []
    for pair, bars in hourly_bars(candles).items():
        parts.append(pd.DataFrame({
            "pair": pair,
            "bar_open_ts": bars.index.astype("int64") // 10**9,
            "future_min_return_1h": (
                bars.low.shift(-1).to_numpy(float) / bars.close.to_numpy(float) - 1.0
            ),
        }))
    one_hour = pd.concat(parts, ignore_index=True)
    if one_hour.duplicated(["pair", "bar_open_ts"]).any():
        raise AssertionError("One-hour label frame has duplicate keys")
    if "future_min_return_1h" in item.columns:
        item = item.drop(columns="future_min_return_1h")
    before = len(item)
    item = item.merge(one_hour, on=["pair", "bar_open_ts"], how="left", validate="one_to_one")
    if len(item) != before:
        raise AssertionError("Short-label join changed feature-panel grain")
    item["short_threshold_1h"] = np.maximum(0.008, 1.5 * item.atr_pct)
    item["short_threshold_24h"] = np.maximum(0.030, 3.0 * item.atr_pct)
    item["target_short_1h"] = (
        item.future_min_return_1h <= -item.short_threshold_1h
    ).astype(float)
    item["target_short_24h"] = (
        item.future_min_return_24h <= -item.short_threshold_24h
    ).astype(float)
    short_mature = item.future_min_return_24h.notna()
    item["target_short"] = (
        item[["target_short_1h", "target_short_24h"]].max(axis=1)
    ).where(short_mature)
    item["label_ready_ts_short"] = item.signal_ts.astype("int64") + 24 * HOUR
    return item.sort_values(["signal_ts", "pair"]).reset_index(drop=True)


def validate_target_rates(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for pair in PAIRS:
        pair_frame = panel[panel.pair == pair]
        for strategy, spec in STRATEGIES.items():
            rate = float(pair_frame[spec["target"]].mean())
            rows.append({
                "pair": pair, "strategy": strategy,
                "rows": int(pair_frame[spec["target"]].notna().sum()),
                "positive_rate": rate,
                "rate_reasonable": bool(0.05 <= rate <= 0.40),
            })
    result = pd.DataFrame(rows)
    if not result.rate_reasonable.all():
        raise RuntimeError(f"Dual-label positive rate outside 5%-40%:\n{result}")
    return result


def train_dual_walk_forward(
    panel: pd.DataFrame, selections: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    config = next(item for item in xgb_configurations() if item["config_id"] == "xgb_21")
    features = list(ALL_FEATURES)
    prediction_parts, audit_rows, importance_rows = [], [], []
    for strategy, spec in STRATEGIES.items():
        working = panel.copy()
        working["target"] = working[spec["target"]]
        working["label_ready_ts"] = working[spec["ready"]]
        for block in selections.itertuples(index=False):
            mature, core, validation = split_mature_training(working, int(block.train_end))
            testing = working[
                (working.signal_ts >= int(block.test_start))
                & (working.signal_ts < int(block.test_end))
            ].copy()
            model, fit_audit = fit_one_group(config, features, mature, core, validation)
            probability = model.predict_proba(testing[features])[:, 1]
            calibration_probability = model.predict_proba(validation[features])[:, 1]
            calibration = validation[["pair"]].copy()
            calibration["probability"] = calibration_probability
            entry, recovery = {}, {}
            for pair in PAIRS:
                values = calibration.loc[calibration.pair == pair, "probability"]
                entry[pair] = float(values.quantile(float(spec["entry_quantile"])))
                recovery[pair] = float(values.quantile(float(spec["recovery_quantile"])))
            keep = [
                "pair", "bar_open_utc", "bar_open_ts", "signal_ts",
                "last_complete_1h_ts", "last_complete_4h_ts", "target_long",
                "target_short", "target_short_1h", "target_short_24h",
                "future_min_return_1h", "future_min_return_24h",
                "future_close_return_72h", "future_below_current_fraction_72h",
                "roc_48h_4h", "sqzmom_pct_4h", "close_at_signal",
            ]
            out = testing[keep].copy()
            out["target"] = testing.target
            out["label_ready_ts"] = testing.label_ready_ts
            out["strategy"] = strategy
            out["strategy_label"] = spec["label"]
            out["variant"] = f"xgb_21 | shared | {strategy}"
            out["probability"] = np.asarray(probability, dtype=float)
            out["entry_threshold"] = out.pair.map(entry)
            out["recovery_threshold"] = out.pair.map(recovery)
            out["period"] = PERIOD
            out["fold"] = int(block.fold)
            prediction_parts.append(out)
            gains = np.asarray(model.feature_importances_, dtype=float)
            if gains.sum() > 0:
                gains = gains / gains.sum()
            importance_rows.extend({
                "strategy": strategy, "fold": int(block.fold),
                "feature": feature, "gain_importance": float(gain),
            } for feature, gain in zip(features, gains))
            audit_rows.append({
                "strategy": strategy, "fold": int(block.fold),
                "label_maturity_hours": 72 if strategy.startswith("long") else 24,
                "train_cutoff_ts": int(block.train_end),
                "mature_rows": len(mature), "target_rate": float(mature.target.mean()),
                "train_last_label_ready_ts": int(mature.label_ready_ts.max()),
                "calibration_first_signal_ts": int(validation.signal_ts.min()),
                "calibration_last_signal_ts": int(validation.signal_ts.max()),
                "test_first_signal_ts": int(testing.signal_ts.min()),
                **fit_audit,
                **{f"{pair}_entry_threshold": entry[pair] for pair in PAIRS},
                **{f"{pair}_recovery_threshold": recovery[pair] for pair in PAIRS},
            })
            print(f"XGB {strategy} {int(block.fold):02d}/{len(selections)}", flush=True)
    predictions = pd.concat(prediction_parts, ignore_index=True)
    audit = pd.DataFrame(audit_rows)
    if not bool((audit.train_last_label_ready_ts <= audit.train_cutoff_ts).all()):
        raise AssertionError("A dual-channel model used an immature label")
    if not bool((audit.calibration_last_signal_ts < audit.test_first_signal_ts).all()):
        raise AssertionError("Calibration records overlap a test fold")
    if not np.isfinite(predictions.probability).all() or not predictions.probability.between(0, 1).all():
        raise AssertionError("Invalid dual-channel probabilities")
    return predictions, audit, pd.DataFrame(importance_rows)


def build_strategy_gate(
    predictions: pd.DataFrame, strategy: str, start_ts: int, end_ts: int,
) -> tuple[dict[str, dict[int, bool]], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    spec = STRATEGIES[strategy]
    gates: dict[str, dict[int, bool]] = {pair: {} for pair in PAIRS}
    states, events, intervals = [], [], []
    for pair in PAIRS:
        records = list(predictions[
            (predictions.strategy == strategy) & (predictions.pair == pair)
            & (predictions.signal_ts >= start_ts) & (predictions.signal_ts < end_ts)
        ].sort_values("signal_ts").itertuples(index=False))
        if not records:
            raise RuntimeError(f"No {strategy}/{pair} predictions for fold")
        state = PairGateState()
        interval_start = None
        for index, row in enumerate(records):
            state, signal = advance_pair_gate(
                pair=pair, probability=float(row.probability),
                entry_threshold=float(row.entry_threshold),
                recovery_threshold=float(row.recovery_threshold),
                signal_ts=int(row.signal_ts), previous=state,
                model_version=f"{MODEL_VERSION}|{strategy}",
            )
            right = min(
                int(records[index + 1].signal_ts) if index + 1 < len(records) else end_ts,
                end_ts,
            )
            for timestamp in range(max(start_ts, int(row.signal_ts)), right, FIVE_MINUTES):
                gates[pair][timestamp] = bool(signal["buy_enabled"])
            states.append({
                "strategy": strategy, "strategy_label": spec["label"], "pair": pair,
                "signal_ts": int(row.signal_ts), "probability": float(row.probability),
                "entry_threshold": float(row.entry_threshold),
                "recovery_threshold": float(row.recovery_threshold),
                "risk_off_active": bool(state.risk_off_active),
                "buy_enabled": bool(signal["buy_enabled"]),
                "transition": signal["transition"], "reason": signal["reason"],
                "event_id": signal["event_id"],
            })
            if signal["transition"] == "enter":
                interval_start = int(row.signal_ts)
            elif signal["transition"] == "recover" and interval_start is not None:
                intervals.append({
                    "strategy": strategy, "strategy_label": spec["label"], "pair": pair,
                    "start_ts": interval_start, "end_ts": int(row.signal_ts),
                    "duration_hours": (int(row.signal_ts) - interval_start) / HOUR,
                    "end_reason": "recover",
                })
                interval_start = None
            if signal["transition"] in {"enter", "recover"}:
                events.append({
                    "strategy": strategy, "strategy_label": spec["label"],
                    "timestamp": int(row.signal_ts), "pair": pair,
                    "event": signal["transition"], "probability": float(row.probability),
                    "entry_threshold": float(row.entry_threshold),
                    "recovery_threshold": float(row.recovery_threshold),
                    "event_id": signal["event_id"],
                })
        if interval_start is not None:
            intervals.append({
                "strategy": strategy, "strategy_label": spec["label"], "pair": pair,
                "start_ts": interval_start, "end_ts": int(end_ts),
                "duration_hours": (int(end_ts) - interval_start) / HOUR,
                "end_reason": "weekly_reinitialization",
            })
            last = records[-1]
            events.append({
                "strategy": strategy, "strategy_label": spec["label"],
                "timestamp": int(end_ts), "pair": pair,
                "event": "weekly_reinitialization", "probability": float(last.probability),
                "entry_threshold": float(last.entry_threshold),
                "recovery_threshold": float(last.recovery_threshold),
                "event_id": f"weekly-reset-{strategy}-{pair}-{end_ts}",
            })
    return gates, pd.DataFrame(states), pd.DataFrame(events), pd.DataFrame(intervals)


def combine_channel_gates(
    channel_gates: Sequence[Mapping[str, Mapping[int, bool]]],
    start_ts: int, end_ts: int,
) -> dict[str, dict[int, bool]]:
    """Combine only channels that belong to each pair.

    Pair-specific channel builders return an empty mapping for the other pair.
    Empty therefore means "not applicable".  A missing timestamp inside an
    applicable channel remains fail-closed.
    """
    if not channel_gates:
        raise ValueError("At least one risk channel is required")
    combined: dict[str, dict[int, bool]] = {pair: {} for pair in PAIRS}
    for pair in PAIRS:
        applicable = [gate[pair] for gate in channel_gates if gate.get(pair)]
        if not applicable:
            raise ValueError(f"No applicable risk channel for {pair}")
        for timestamp in range(int(start_ts), int(end_ts), FIVE_MINUTES):
            combined[pair][timestamp] = all(
                bool(gate.get(timestamp, False)) for gate in applicable
            )
    return combined


def replay_channels(
    candles: Mapping[str, pd.DataFrame], selections: pd.DataFrame,
    predictions: pd.DataFrame, strategies: Sequence[str], scenario: str,
) -> dict[str, Any]:
    weekly, pair_rows, curves, trades = [], [], [], []
    state_parts, event_parts, interval_parts, stop_rows = [], [], [], []
    cumulative = 0.0
    for selection in selections.itertuples(index=False):
        fold_predictions = predictions[predictions.fold == int(selection.fold)]
        channel_gates = []
        for strategy in strategies:
            gate, states, events, intervals = build_strategy_gate(
                fold_predictions, strategy, int(selection.test_start), int(selection.test_end)
            )
            channel_gates.append(gate)
            for frame in (states, events, intervals):
                if not frame.empty:
                    frame["period"], frame["fold"] = PERIOD, int(selection.fold)
            state_parts.append(states); event_parts.append(events); interval_parts.append(intervals)
        combined = combine_channel_gates(
            channel_gates, int(selection.test_start), int(selection.test_end)
        )
        result, curve, pair_metrics, trade_frame, stop = base.simulate_fold(
            candles, selection, combined, record_details=True
        )
        weekly.append({"scenario": scenario, **selection._asdict(), **result, **stop})
        pair_rows.extend({
            "scenario": scenario, "period": PERIOD, "fold": int(selection.fold),
            "pair": pair, **metrics,
        } for pair, metrics in pair_metrics.items())
        if not curve.empty:
            curve = curve.copy()
            curve["scenario"], curve["period"], curve["fold"] = scenario, PERIOD, int(selection.fold)
            curve["cumulative_oos_pnl"] = cumulative + curve.equity - base.INITIAL_EQUITY
            curves.append(curve)
        if not trade_frame.empty:
            trade_frame["scenario"], trade_frame["period"], trade_frame["fold"] = scenario, PERIOD, int(selection.fold)
            trades.append(trade_frame)
        for event in trade_frame.to_dict("records") if not trade_frame.empty else []:
            if event.get("reason") == "pair_breaker_flatten":
                stop_rows.append({
                    "scenario": scenario, "fold": int(selection.fold), "scope": event["pair"],
                    "kind": "pair_stop", "start_ts": int(event["timestamp"]),
                    "end_ts": int(selection.test_end),
                })
        if result["liquidated"] and not curve.empty:
            stop_rows.append({
                "scenario": scenario, "fold": int(selection.fold), "scope": "PORTFOLIO",
                "kind": "portfolio_stop", "start_ts": int(curve.timestamp.iloc[-1]),
                "end_ts": int(selection.test_end),
            })
        cumulative += float(result["net_pnl_quote"])
    summary = aggregate_rows(weekly, pair_rows)
    pair_frame = pd.DataFrame(pair_rows)
    weekly_frame = pd.DataFrame(weekly)
    summary["risk_off_pair_hours"] = float(pair_frame.technical_risk_off_seconds.sum() / HOUR)
    summary["momentum_stop_exits"] = int(weekly_frame.momentum_stop_exits.sum())
    return {
        "summary": summary, "weekly": weekly_frame, "pairs": pair_frame,
        "equity": pd.concat(curves, ignore_index=True),
        "trades": pd.concat(trades, ignore_index=True) if trades else pd.DataFrame(),
        "states": pd.concat(state_parts, ignore_index=True),
        "events": pd.concat(event_parts, ignore_index=True),
        "intervals": pd.concat(interval_parts, ignore_index=True),
        "stops": pd.DataFrame(stop_rows),
    }


def classification_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (strategy, pair), frame in predictions.groupby(["strategy", "pair"]):
        item = frame[frame.target.notna()].copy()
        y = item.target.astype(int).to_numpy()
        probability = item.probability.to_numpy(float)
        decision = (probability >= item.entry_threshold.to_numpy(float)).astype(int)
        rows.append({
            "strategy": strategy, "pair": pair, "rows": len(item),
            "positive_rate": float(y.mean()),
            "roc_auc": float(roc_auc_score(y, probability)) if len(np.unique(y)) > 1 else np.nan,
            "precision_at_entry": float(precision_score(y, decision, zero_division=0)),
            "recall_at_entry": float(recall_score(y, decision, zero_division=0)),
            "balanced_accuracy_at_entry": float(balanced_accuracy_score(y, decision)),
        })
    return pd.DataFrame(rows)


def _merge_effective_intervals(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (strategy, pair), group in frame.groupby(["strategy", "pair"]):
        merged: list[list[int]] = []
        for interval in group.sort_values(["start_ts", "end_ts"]).itertuples(index=False):
            left, right = int(interval.start_ts), int(interval.end_ts)
            if merged and left <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], right)
            else:
                merged.append([left, right])
        rows.extend({
            "strategy": strategy, "strategy_label": STRATEGIES[strategy]["label"],
            "pair": pair, "start_ts": left, "end_ts": right,
            "duration_hours": (right - left) / HOUR,
        } for left, right in merged)
    return pd.DataFrame(rows)


def june_coverage(intervals: pd.DataFrame) -> pd.DataFrame:
    start = int(pd.Timestamp("2026-06-01T00:00:00Z").timestamp())
    end = int(pd.Timestamp("2026-06-06T00:00:00Z").timestamp())
    effective = _merge_effective_intervals(intervals)
    rows = []
    for pair in PAIRS:
        for strategy in STRATEGIES:
            hit = effective[
                (effective.pair == pair) & (effective.strategy == strategy)
                & (effective.start_ts < end) & (effective.end_ts > start)
            ].sort_values("start_ts").head(1)
            row = hit.iloc[0] if not hit.empty else None
            rows.append({
                "pair": pair, "strategy": strategy,
                "strategy_label": STRATEGIES[strategy]["label"],
                "covers_june_1_start": bool(row is not None and int(row.start_ts) <= start < int(row.end_ts)),
                "effective_entry_utc": (
                    pd.to_datetime(int(row.start_ts), unit="s", utc=True).strftime("%Y-%m-%d %H:%M UTC")
                    if row is not None else None
                ),
                "effective_exit_utc": (
                    pd.to_datetime(int(row.end_ts), unit="s", utc=True).strftime("%Y-%m-%d %H:%M UTC")
                    if row is not None else None
                ),
            })
    return pd.DataFrame(rows)


def build_plotly(
    cache_dir: Path, output_dir: Path, states: pd.DataFrame,
    events: pd.DataFrame, intervals: pd.DataFrame,
    metrics: pd.DataFrame, coverage: pd.DataFrame,
    anchor_windows: Sequence[tuple[str, int, int]] | None = None,
) -> Path:
    # Fifteen-minute plotting grain preserves the price path while keeping the
    # self-contained dual-channel HTML responsive. Backtest calculations still
    # use the complete five-minute candles.
    prices = {}
    for pair in PAIRS:
        full_price = base._price_frame(
            cache_dir / f"binance_{pair}_5m.csv", base.START_TS, base.END_TS
        )
        prices[pair] = full_price.iloc[::3].copy()
    event_parts = []
    for pair in PAIRS:
        item = events[events.pair == pair].sort_values("timestamp").copy()
        item["timestamp"] = item.timestamp.astype("int64")
        item = pd.merge_asof(
            item, prices[pair][["timestamp", "close"]].sort_values("timestamp"),
            on="timestamp", direction="backward",
        )
        item["time"] = pd.to_datetime(item.timestamp, unit="s", utc=True)
        item["time_utc"] = item.time.dt.strftime("%Y-%m-%d %H:%M UTC")
        event_parts.append(item)
    plotted_events = pd.concat(event_parts, ignore_index=True).sort_values(
        ["timestamp", "pair", "strategy"]
    )
    plotted_events.to_csv(output_dir / "plotly_dual_entry_exit_events.csv", index=False)

    figure = make_subplots(
        rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.05,
        row_heights=[0.29, 0.21, 0.29, 0.21],
        subplot_titles=(
            "BTC价格与双预测区间",
            "BTC长期/短期概率",
            "ETH价格与双预测区间",
            "ETH长期/短期概率",
        ),
    )
    # Plotly's add_vrect mutates and revalidates the complete shape collection
    # on every call.  Short-horizon gates can produce hundreds of intervals,
    # turning report generation into an O(n^2) operation.  Build the equivalent
    # subplot-bound shapes in memory and attach them once instead.
    plot_shapes: list[dict[str, object]] = []

    def append_vrect(
        chart_row: int, x0: pd.Timestamp, x1: pd.Timestamp, *,
        fillcolor: str, line_width: float, layer: str,
        line_color: str | None = None,
    ) -> None:
        axis_suffix = "" if chart_row == 1 else str(chart_row)
        line: dict[str, object] = {"width": line_width}
        if line_color is not None:
            line["color"] = line_color
        plot_shapes.append({
            "type": "rect",
            "xref": f"x{axis_suffix}",
            "yref": f"y{axis_suffix} domain",
            "x0": x0,
            "x1": x1,
            "y0": 0,
            "y1": 1,
            "fillcolor": fillcolor,
            "line": line,
            "layer": layer,
        })

    price_color = "#30343B"
    for pair, price_row, probability_row in (("BTC-FDUSD", 1, 2), ("ETH-FDUSD", 3, 4)):
        price = prices[pair]
        figure.add_trace(go.Scattergl(
            x=price.time, y=price.close, mode="lines", name=f"{pair[:3]} close",
            line={"color": price_color, "width": 1.3}, legendgroup=f"{pair}-price",
            hovertemplate="%{x|%Y-%m-%d %H:%M UTC}<br>Close %{y:,.4f} FDUSD<extra></extra>",
        ), row=price_row, col=1)
        for strategy, spec in STRATEGIES.items():
            state = states[(states.pair == pair) & (states.strategy == strategy)].sort_values("signal_ts").copy()
            state["time"] = pd.to_datetime(state.signal_ts, unit="s", utc=True)
            figure.add_trace(go.Scattergl(
                x=state.time, y=state.probability, mode="lines",
                name=f"{pair[:3]} {spec['label']} probability",
                line={"color": spec["color"], "width": 1.25},
                legendgroup=f"{pair}-{strategy}",
                customdata=np.column_stack((state.entry_threshold, state.recovery_threshold, state.transition)),
                hovertemplate=("%{x|%Y-%m-%d %H:%M UTC}<br>Probability %{y:.6f}"
                               "<br>Entry %{customdata[0]:.6f}<br>Recovery %{customdata[1]:.6f}"
                               "<br>State %{customdata[2]}<extra></extra>"),
            ), row=probability_row, col=1)
            figure.add_trace(go.Scatter(
                x=state.time, y=state.entry_threshold, mode="lines",
                name=f"{pair[:3]} {spec['label']} entry threshold",
                line={"color": spec["color"], "width": 1, "dash": "dot"},
                legendgroup=f"{pair}-{strategy}",
                hovertemplate="%{x|%Y-%m-%d %H:%M UTC}<br>Entry %{y:.6f}<extra></extra>",
            ), row=probability_row, col=1)
            for event, label, symbol in (
                ("enter", "进入", spec["enter_symbol"]),
                ("recover", "退出", spec["recover_symbol"]),
            ):
                marked = plotted_events[
                    (plotted_events.pair == pair) & (plotted_events.strategy == strategy)
                    & (plotted_events.event == event)
                ]
                figure.add_trace(go.Scatter(
                    x=marked.time, y=marked.close, mode="markers",
                    name=f"{pair[:3]} {spec['label']} {label}",
                    marker={"symbol": symbol, "size": 9, "color": spec["color"],
                            "line": {"color": "#111827", "width": 0.65}},
                    legendgroup=f"{pair}-{strategy}-{event}",
                    customdata=np.column_stack((marked.probability, marked.entry_threshold, marked.time_utc)) if not marked.empty else None,
                    hovertemplate=(f"<b>{spec['label']} {label}</b><br>%{{customdata[2]}}"
                                   "<br>Price %{y:,.4f}<br>Probability %{customdata[0]:.6f}"
                                   "<br>Entry %{customdata[1]:.6f}<extra></extra>"),
                ), row=price_row, col=1)
            for interval in intervals[
                (intervals.pair == pair) & (intervals.strategy == strategy)
            ].itertuples(index=False):
                for chart_row in (price_row, probability_row):
                    append_vrect(
                        chart_row,
                        pd.to_datetime(interval.start_ts, unit="s", utc=True),
                        pd.to_datetime(interval.end_ts, unit="s", utc=True),
                        fillcolor=spec["fill"], line_width=0, layer="below",
                    )
        highlighted = anchor_windows or ((
            "jun_01_05", int(pd.Timestamp("2026-06-01T00:00:00Z").timestamp()),
            int(pd.Timestamp("2026-06-06T00:00:00Z").timestamp()),
        ),)
        for _, anchor_start, anchor_end in highlighted:
            for chart_row in (price_row, probability_row):
                append_vrect(
                    chart_row,
                    pd.to_datetime(anchor_start, unit="s", utc=True),
                    pd.to_datetime(anchor_end, unit="s", utc=True),
                    fillcolor="rgba(0,0,0,0)", line_width=1.2,
                    line_color="rgba(55,65,81,0.75)", layer="above",
                )
        figure.update_yaxes(title_text=f"{pair[:3]} price (FDUSD)", row=price_row, col=1)
        figure.update_yaxes(title_text="Probability", range=[0, 1], row=probability_row, col=1)
    figure.update_xaxes(showgrid=True, gridcolor="#F3F4F6")
    figure.update_xaxes(
        title_text="UTC", row=4, col=1,
        rangeslider={"visible": True, "thickness": 0.045},
        rangeselector={"buttons": [
            {"count": 5, "label": "5d", "step": "day", "stepmode": "backward"},
            {"count": 30, "label": "30d", "step": "day", "stepmode": "backward"},
            {"count": 90, "label": "90d", "step": "day", "stepmode": "backward"},
            {"step": "all", "label": "180d"},
        ]},
    )
    figure.update_layout(
        template="plotly_white", height=1500, hovermode="x unified",
        margin={"l": 72, "r": 30, "t": 220, "b": 65},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.025, "x": 0, "font": {"size": 9}},
        font={"family": "Arial, Microsoft YaHei, sans-serif", "color": "#1F2937"},
        shapes=plot_shapes,
    )
    plot = figure.to_html(
        full_html=False, include_plotlyjs=True, div_id="dual-risk-gate-plot",
        config={"responsive": True, "displaylogo": False, "scrollZoom": True},
    )
    drawdown_column = (
        "stitched_max_drawdown_pct"
        if "stitched_max_drawdown_pct" in metrics.columns else "worst_drawdown_pct"
    )
    baseline_profit = float(metrics.iloc[0].oos_pnl_fdusd)
    baseline_drawdown = float(metrics.iloc[0][drawdown_column])
    metric_rows = "".join(
        f"<tr><td>{html.escape(str(row.scenario))}</td><td>{row.oos_pnl_fdusd:+.4f}</td>"
        f"<td>{float(getattr(row, drawdown_column)):.4f}%</td>"
        f"<td>{float(getattr(row, 'worst_drawdown_pct', getattr(row, drawdown_column))):.4f}%</td>"
        f"<td>{float(row.oos_pnl_fdusd) - baseline_profit:+.4f}</td>"
        f"<td>{float(getattr(row, drawdown_column)) - baseline_drawdown:+.4f} pp</td>"
        f"<td>{int(row.pair_stop_events)}</td><td>{int(row.portfolio_stop_events)}</td>"
        f"<td>{row.risk_off_pair_hours:.1f}</td></tr>"
        for row in metrics.itertuples(index=False)
    )
    coverage_rows = "".join(
        f"<tr><td>{html.escape(row.pair)}</td><td>{html.escape(row.strategy_label)}</td>"
        f"<td>{'是' if row.covers_june_1_start else '否'}</td>"
        f"<td>{html.escape(str(row.effective_entry_utc))}</td>"
        f"<td>{html.escape(str(row.effective_exit_utc))}</td></tr>"
        for row in coverage.itertuples(index=False)
    )
    coverage_section = (
        '<h2>6月1–5日两种预测区间</h2><div class="table-wrap"><table><thead><tr>'
        '<th>Pair</th><th>策略</th><th>6月1日已Risk-off</th><th>有效进入UTC</th>'
        f'<th>有效退出UTC</th></tr></thead><tbody>{coverage_rows}</tbody></table></div>'
        if coverage_rows else ""
    )
    anchor_labels = {"feb_03_06": "2月3–6日", "jun_01_06": "6月1–6日",
                     "jun_01_05": "6月1–5日"}
    highlighted_names = "、".join(
        anchor_labels.get(name, name)
        for name, _, _ in (anchor_windows or (("jun_01_05", 0, 0),))
    )
    event_rows = "".join(
        f"<tr><td>{row.time_utc}</td><td>{row.pair}</td><td>{html.escape(row.strategy_label)}</td>"
        f"<td>{row.event}</td><td>{row.probability:.6f}</td><td>{row.entry_threshold:.6f}</td>"
        f"<td>{row.recovery_threshold:.6f}</td><td>{row.close:,.4f}</td></tr>"
        for row in plotted_events.itertuples(index=False)
    )
    page = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>XGBoost双风险策略180天回测</title><style>
html,body{{max-width:100%;overflow-x:hidden}}body{{margin:0;background:#fff;color:#1f2937;font-family:Arial,'Microsoft YaHei',sans-serif}}
main{{box-sizing:border-box;width:100%;max-width:1540px;margin:auto;padding:16px}}h1{{font-size:26px;margin:8px 0 4px;overflow-wrap:anywhere}}
.sub{{color:#4b5563;margin:0 0 12px;overflow-wrap:anywhere}}.note{{padding:12px 14px;background:#f9fafb;border-left:4px solid #374151;line-height:1.55;overflow-wrap:anywhere}}
#dual-risk-gate-plot,.plotly-graph-div{{width:100%!important;max-width:100%!important}}h2{{margin-top:28px}}
.shadow-switches{{position:sticky;top:8px;z-index:1000;display:flex;align-items:center;gap:14px;flex-wrap:wrap;width:max-content;max-width:calc(100% - 24px);margin:12px 0 4px;padding:9px 12px;background:rgba(255,255,255,.96);border:1px solid #d1d5db;border-radius:9px;box-shadow:0 3px 12px rgba(17,24,39,.12);font-size:13px}}
.shadow-switches-title{{font-weight:700}}.shadow-switch{{display:inline-flex;align-items:center;gap:7px;cursor:pointer;white-space:nowrap}}.shadow-switch input{{position:absolute;opacity:0;pointer-events:none}}
.shadow-track{{position:relative;width:34px;height:19px;background:#9ca3af;border-radius:999px;transition:background .16s ease;box-shadow:inset 0 0 0 1px rgba(17,24,39,.18)}}.shadow-track::after{{content:'';position:absolute;left:2px;top:2px;width:15px;height:15px;background:#fff;border-radius:50%;box-shadow:0 1px 2px rgba(0,0,0,.3);transition:transform .16s ease}}.shadow-switch input:checked+.shadow-track::after{{transform:translateX(15px)}}
#toggle-long-shadow:checked+.shadow-track{{background:#C2410C}}#toggle-short-shadow:checked+.shadow-track{{background:#2563EB}}.shadow-switch input:focus-visible+.shadow-track{{outline:3px solid rgba(37,99,235,.28);outline-offset:2px}}.shadow-switches-note{{color:#6b7280;font-size:11px}}
.table-wrap{{overflow:auto;border:1px solid #e5e7eb;border-radius:8px;margin-top:10px}}table{{border-collapse:collapse;width:100%;font-size:13px}}
th,td{{padding:8px 10px;border-bottom:1px solid #e5e7eb;text-align:right;white-space:nowrap}}th{{background:#f9fafb;position:sticky;top:0}}
th:first-child,td:first-child,th:nth-child(2),td:nth-child(2),th:nth-child(3),td:nth-child(3),th:nth-child(4),td:nth-child(4){{text-align:left}}
@media(max-width:600px){{main{{padding:8px}}h1{{font-size:19px}}.sub,.note{{font-size:12px}}.shadow-switches{{top:4px;gap:8px;padding:7px 9px;font-size:12px}}.shadow-switches-note{{width:100%}}}}
</style></head><body><main><h1>XGBoost双风险策略：180天诊断回测</h1>
<p class="sub">2026-02-01 15:00 至 2026-07-31 15:00 UTC｜{html.escape(STRATEGIES['long_persistent_72h']['label'])} + {html.escape(STRATEGIES['short_spike_1h_24h']['label'])}</p>
<div class="note"><b>读图：</b>橙色阴影/三角为长期下降通道，蓝色阴影/菱形与圆圈为短期插针通道；灰色边框标出{html.escape(highlighted_names)}。两通道独立预测，组合Grid在任一通道Risk-off时暂停对应交易对普通BUY，不触发即时卖出。该区间已被查看，全部结果均为诊断性回放。</div>
{plot}<h2>180天Grid策略对比</h2><div class="table-wrap"><table><thead><tr><th>方案</th><th>收益 FDUSD</th><th>拼接最大回撤</th><th>最差周内回撤</th><th>相对机制1收益</th><th>相对机制1回撤</th><th>单对停止</th><th>组合停止</th><th>Risk-off pair-hours</th></tr></thead><tbody>{metric_rows}</tbody></table></div>
{coverage_section}
<h2>全部精确进入与退出时间</h2><div class="table-wrap"><table><thead><tr><th>UTC</th><th>Pair</th><th>策略</th><th>事件</th><th>概率</th><th>进入阈值</th><th>恢复阈值</th><th>价格</th></tr></thead><tbody>{event_rows}</tbody></table></div>
</main><div class="shadow-switches" id="shadow-switches" role="group" aria-label="预测区间阴影显示">
<span class="shadow-switches-title">阴影显示</span>
<label class="shadow-switch" for="toggle-long-shadow"><input id="toggle-long-shadow" type="checkbox" checked><span class="shadow-track" aria-hidden="true"></span><span>{html.escape(STRATEGIES['long_persistent_72h']['label'])}</span></label>
<label class="shadow-switch" for="toggle-short-shadow"><input id="toggle-short-shadow" type="checkbox" checked><span class="shadow-track" aria-hidden="true"></span><span>{html.escape(STRATEGIES['short_spike_1h_24h']['label'])}</span></label>
<span class="shadow-switches-note">仅控制背景阴影</span></div>
<script>(function(){{const chart=document.getElementById('dual-risk-gate-plot');const controls=document.getElementById('shadow-switches');if(chart&&controls)chart.parentNode.insertBefore(controls,chart);function normalize(value){{return String(value||'').replace(/\\s/g,'').toLowerCase();}}function setShadow(fill,visible){{if(!chart||!window.Plotly||!chart.layout)return;const target=normalize(fill);const update={{}};(chart.layout.shapes||[]).forEach(function(shape,index){{if(normalize(shape.fillcolor)===target)update['shapes['+index+'].visible']=visible;}});if(Object.keys(update).length)Plotly.relayout(chart,update);}}document.getElementById('toggle-long-shadow').addEventListener('change',function(event){{setShadow('rgba(194,65,12,0.11)',event.target.checked);}});document.getElementById('toggle-short-shadow').addEventListener('change',function(event){{setShadow('rgba(37,99,235,0.09)',event.target.checked);}});function adapt(){{if(!chart||!window.Plotly)return;const mobile=window.innerWidth<=600;chart.style.height=mobile?'1840px':'1500px';Plotly.relayout(chart,mobile?{{width:Math.max(window.innerWidth-16,320),height:1840,'margin.l':52,'margin.r':12,'margin.t':500,'margin.b':65,'legend.orientation':'v','legend.x':0.02,'legend.y':1.02,'legend.font.size':8}}:{{width:Math.min(window.innerWidth-32,1540),height:1500,'margin.l':72,'margin.r':30,'margin.t':220,'margin.b':65,'legend.orientation':'h','legend.x':0,'legend.y':1.025,'legend.font.size':9}});Plotly.Plots.resize(chart);}}window.addEventListener('load',adapt);let timer;window.addEventListener('resize',function(){{clearTimeout(timer);timer=setTimeout(adapt,120);}});}})();</script></body></html>"""
    output = output_dir / "xgboost_dual_risk_gate_180d_plotly.html"
    output.write_text(page, encoding="utf-8")
    return output


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    candles, quality = load_candles(args.cache_dir)
    quality.to_csv(args.output_dir / "data_quality.csv", index=False)
    windows = base.diagnostic_windows()
    selections_path = args.output_dir / "grid_selections.csv"
    grid_audit_path = args.output_dir / "grid_selection_audit.csv"
    if args.resume and selections_path.exists() and grid_audit_path.exists():
        selections = pd.read_csv(selections_path); grid_audit = pd.read_csv(grid_audit_path)
    else:
        selections, grid_audit = base.frozen_grid_sequence(windows, args.source_weekly_results)
        selections.to_csv(selections_path, index=False); grid_audit.to_csv(grid_audit_path, index=False)

    panel_path = args.output_dir / "dual_target_feature_panel.csv.gz"
    if args.resume and panel_path.exists():
        panel = pd.read_csv(panel_path)
    else:
        source_panel = pd.read_csv(args.source_long_panel) if args.source_long_panel.exists() else base.build_multi_horizon_panel(candles)
        panel = augment_dual_targets(source_panel, candles)
        panel.to_csv(panel_path, index=False, compression="gzip")
    target_quality = validate_target_rates(panel)
    target_quality.to_csv(args.output_dir / "target_quality.csv", index=False)

    prediction_path = args.output_dir / "dual_walk_forward_predictions.csv.gz"
    audit_path = args.output_dir / "dual_training_audit.csv"
    importance_path = args.output_dir / "dual_gain_feature_importance.csv"
    if args.resume and prediction_path.exists() and audit_path.exists() and importance_path.exists():
        predictions = pd.read_csv(prediction_path)
        audit = pd.read_csv(audit_path)
        importance = pd.read_csv(importance_path)
    else:
        predictions, audit, importance = train_dual_walk_forward(panel, selections)
        predictions.to_csv(prediction_path, index=False, compression="gzip")
        audit.to_csv(audit_path, index=False)
        importance.to_csv(importance_path, index=False)

    baseline = base.replay(
        candles, selections, scenario="Mechanism 1 (BTC ROC/SQZMOM)",
        baseline_gate=base.mechanism1_gate(candles), record_details=True,
    )
    results = {"Mechanism 1 (BTC ROC/SQZMOM)": baseline}
    for scenario, strategies in SCENARIOS:
        results[scenario] = replay_channels(
            candles, selections, predictions, strategies, scenario
        )
    combined = results["XGBoost dual OR gate"]

    pd.concat([item["weekly"] for item in results.values()], ignore_index=True).to_csv(
        args.output_dir / "weekly_results.csv", index=False
    )
    pd.concat([item["pairs"] for item in results.values()], ignore_index=True).to_csv(
        args.output_dir / "pair_results.csv", index=False
    )
    pd.concat([item["equity"] for item in results.values()], ignore_index=True).to_csv(
        args.output_dir / "equity_curves.csv.gz", index=False, compression="gzip"
    )
    pd.concat([item["trades"] for item in results.values()], ignore_index=True).to_csv(
        args.output_dir / "trade_events.csv.gz", index=False, compression="gzip"
    )
    pd.concat([item["stops"] for item in results.values()], ignore_index=True).to_csv(
        args.output_dir / "stop_events.csv", index=False
    )
    combined["states"].to_csv(args.output_dir / "dual_risk_states.csv.gz", index=False, compression="gzip")
    combined["events"].to_csv(args.output_dir / "dual_risk_gate_events.csv", index=False)
    combined["intervals"].to_csv(args.output_dir / "dual_risk_off_intervals.csv", index=False)
    effective_intervals = _merge_effective_intervals(combined["intervals"])
    effective_intervals[
        effective_intervals.strategy == "long_persistent_72h"
    ].to_csv(args.output_dir / "long_prediction_intervals.csv", index=False)
    effective_intervals[
        effective_intervals.strategy == "short_spike_1h_24h"
    ].to_csv(args.output_dir / "short_prediction_intervals.csv", index=False)
    metrics = pd.DataFrame([{"scenario": name, **item["summary"]} for name, item in results.items()])
    metrics.to_csv(args.output_dir / "metrics.csv", index=False)
    classification = classification_metrics(predictions)
    classification.to_csv(args.output_dir / "classification_metrics.csv", index=False)
    coverage = june_coverage(combined["intervals"])
    coverage.to_csv(args.output_dir / "june_1_5_strategy_coverage.csv", index=False)
    plot = build_plotly(
        args.cache_dir, args.output_dir, combined["states"], combined["events"],
        combined["intervals"], metrics, coverage,
    )

    base_summary, dual_summary = baseline["summary"], combined["summary"]
    gates = {
        "long_channel_covers_june_1_for_btc_eth": bool(
            coverage[coverage.strategy == "long_persistent_72h"].covers_june_1_start.all()
        ),
        "dual_pnl_positive": bool(dual_summary["oos_pnl_fdusd"] > 0),
        "dual_pnl_better_than_mechanism1": bool(dual_summary["oos_pnl_fdusd"] > base_summary["oos_pnl_fdusd"]),
        "dual_drawdown_not_worse": bool(dual_summary["worst_drawdown_pct"] >= base_summary["worst_drawdown_pct"]),
        "dual_pair_stops_fewer": bool(dual_summary["pair_stop_events"] < base_summary["pair_stop_events"]),
        "dual_portfolio_stops_zero": bool(dual_summary["portfolio_stop_events"] == 0),
        "no_immediate_momentum_sell": bool(dual_summary["momentum_stop_exits"] == 0),
    }
    summary = {
        "schema": "xgboost-dual-risk-gate-180d-diagnostic-v1",
        "model_version": MODEL_VERSION,
        "evidence_status": "diagnostic_replay_after_interval_review",
        "deployment_authorized": False,
        "verdict": "NEXT_STAGE_JOINT_VALIDATION" if all(gates.values()) else "NO-GO",
        "start_utc": pd.to_datetime(base.START_TS, unit="s", utc=True).isoformat(),
        "end_utc": pd.to_datetime(base.END_TS, unit="s", utc=True).isoformat(),
        "calendar_days": 180, "weekly_folds": len(selections),
        "strategies": {
            "long_persistent_72h": {
                "target": "72h close <= -max(3%, 3xATR%) and >=2/3 future hourly closes below current",
                "maturity_hours": 72, "entry_quantile": 0.75, "recovery_quantile": 0.65,
            },
            "short_spike_1h_24h": {
                "target": "next-1h low <= -max(0.8%,1.5xATR%) OR next-24h low <= -max(3%,3xATR%)",
                "maturity_hours": 24, "entry_quantile": 0.90, "recovery_quantile": 0.80,
            },
        },
        "configuration": next(item for item in xgb_configurations() if item["config_id"] == "xgb_21"),
        "metrics": {name: item["summary"] for name, item in results.items()},
        "acceptance_gates": gates,
        "target_quality": target_quality.to_dict("records"),
        "classification_metrics": classification.to_dict("records"),
        "june_1_5_coverage": coverage.to_dict("records"),
        "no_lookahead_checks": {
            "all_training_labels_mature": bool((audit.train_last_label_ready_ts <= audit.train_cutoff_ts).all()),
            "calibration_precedes_test": bool((audit.calibration_last_signal_ts < audit.test_first_signal_ts).all()),
            "probabilities_finite_and_bounded": bool(np.isfinite(predictions.probability).all() and predictions.probability.between(0, 1).all()),
        },
        "artifacts": {"plotly_html": str(plot)},
        "input_hashes": {
            "candles": {pair: sha256_file(args.cache_dir / f"binance_{pair}_5m.csv") for pair in PAIRS},
            "feature_panel": sha256_file(panel_path), "grid": sha256_file(selections_path),
            "predictions": sha256_file(prediction_path),
        },
        "limitations": [
            "The June 1-5 interval and full 180-day period have been viewed; this is diagnostic replay, not fresh OOS evidence.",
            "Weekly validation reinitializes Grid inventory and both gate states.",
            "Funding, OI, taker-buy ratio and macro/FOMC history remain unavailable.",
        ],
    }
    write_json(args.output_dir / "summary.json", summary)
    print(json.dumps({
        "verdict": summary["verdict"], "metrics": summary["metrics"],
        "coverage": summary["june_1_5_coverage"], "plot": str(plot),
    }, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

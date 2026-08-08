#!/usr/bin/env python3
"""Diagnostic 180-day replay for a multi-horizon XGBoost Grid BUY gate.

This study keeps the production-like Grid, inventory-exit policy and breakers.
Only the legacy BTC ROC/SQZMOM ordinary-BUY gate is replaced.  The target is a
fixed 72h closing-decline plus price-persistence label. Every weekly fit and
probability threshold uses only labels mature at the cutoff.

The June 1-5 interval motivated this study, so the result is diagnostic replay
evidence, never fresh out-of-sample evidence and never deployment authority.
"""

from __future__ import annotations

import argparse
import html
import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from compare_independent_gate_ml_stops import (
    ALL_FEATURES,
    DAY,
    FIVE_MINUTES,
    HOUR,
    PAIRS,
    build_feature_panel,
    hourly_bars,
    load_candles,
)
from fdusd_live_grid_optimizer import select_candidate
from grid_xgboost_risk_gate import PairGateState, advance_pair_gate
from search_fdusd_inventory_exit import aggregate_rows
from tune_xgboost_grid_risk_gate_v1 import (
    INITIAL_EQUITY,
    POLICY,
    TAKER_FEE,
    candidate_from_row,
    mechanism1_gate,
    replay,
    simulate_fold,
)
from tune_xgboost_momentum_stop_v2 import (
    fit_one_group,
    sha256_file,
    split_mature_training,
    write_json,
    xgb_configurations,
)
from validate_grid_live import Candidate, slice_window


MODEL_VERSION = "xgboost-grid-persistent-risk-gate-v3"
VARIANT = "xgb_21 | shared | persistent-72h"
PERIOD = "diagnostic_180d"
START_TS = int(pd.Timestamp("2026-02-01T15:00:00Z").timestamp())
END_TS = int(pd.Timestamp("2026-07-31T15:00:00Z").timestamp())
ENTRY_QUANTILE = 0.75
RECOVERY_QUANTILE = 0.65
INITIAL_APPROVED = Candidate(0.03, 0.006, 0.006, 0.015, 1800)
DEFAULT_OUTPUT = Path("results/backtests/xgboost_persistent_risk_gate_180d_v3")
DEFAULT_WEEKLY_RESULTS = Path(
    "results/backtests/fdusd_inventory_exit_parameter_search/weekly_results.csv"
)

PAIR_LABEL = {"BTC-FDUSD": "BTC", "ETH-FDUSD": "ETH"}
PAIR_COLOR = {"BTC-FDUSD": "#2563EB", "ETH-FDUSD": "#C2417B"}
RISK_COLOR = "#C2410C"
RECOVERY_COLOR = "#2563EB"
RISK_FILL = "rgba(194,65,12,0.11)"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=Path("data/backtesting_candles"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--source-weekly-results", type=Path, default=DEFAULT_WEEKLY_RESULTS)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def diagnostic_windows(start_ts: int = START_TS, end_ts: int = END_TS) -> pd.DataFrame:
    """Return weekly tests, including a final partial week, over exactly 180 days."""
    if end_ts - start_ts != 180 * DAY:
        raise ValueError("Diagnostic interval must span exactly 180 days")
    rows = []
    test_start = int(start_ts)
    fold = 1
    while test_start < end_ts:
        test_end = min(test_start + 7 * DAY, end_ts)
        rows.append({
            "period": PERIOD,
            "fold": fold,
            "train_start": test_start - 14 * DAY,
            "train_end": test_start,
            "test_start": test_start,
            "test_end": test_end,
        })
        test_start = test_end
        fold += 1
    result = pd.DataFrame(rows)
    if int((result.test_end - result.test_start).sum()) != 180 * DAY:
        raise AssertionError("Weekly windows do not cover the complete interval")
    return result


def _future_min_return(frame: pd.DataFrame, hours: int) -> pd.Series:
    future_low = pd.concat(
        [frame.low.shift(-offset) for offset in range(1, hours + 1)], axis=1
    ).min(axis=1, skipna=False)
    return future_low / frame.close - 1.0


def build_multi_horizon_panel(candles: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Create fixed 6h/24h/72h labels while preserving the v1 feature schema."""
    panel = build_feature_panel(candles, horizon_hours=72).copy()
    hourly = hourly_bars(candles)
    label_parts = []
    for pair in PAIRS:
        bars = hourly[pair]
        item = pd.DataFrame({
            "bar_open_ts": bars.index.astype("int64") // 10**9,
            "close_at_signal": bars.close.to_numpy(float),
            "future_min_return_6h": _future_min_return(bars, 6).to_numpy(float),
            "future_min_return_24h": _future_min_return(bars, 24).to_numpy(float),
            "future_min_return_72h": _future_min_return(bars, 72).to_numpy(float),
            "future_close_return_72h": (
                bars.close.shift(-72).to_numpy(float) / bars.close.to_numpy(float) - 1.0
            ),
            "future_below_current_fraction_72h": (
                pd.concat([bars.close.shift(-offset) for offset in range(1, 73)], axis=1)
                .lt(bars.close, axis=0).sum(axis=1).to_numpy(float) / 72.0
            ),
        })
        item["pair"] = pair
        label_parts.append(item)
    labels = pd.concat(label_parts, ignore_index=True)
    panel = panel.drop(columns=["target", "future_min_return", "adverse_threshold"]).merge(
        labels, on=["pair", "bar_open_ts"], how="left", validate="one_to_one"
    )
    panel["adverse_threshold_6h"] = np.maximum(0.004, panel.atr_pct)
    panel["adverse_threshold_24h"] = np.maximum(0.015, 2.0 * panel.atr_pct)
    panel["adverse_threshold_72h"] = np.maximum(0.030, 3.0 * panel.atr_pct)
    panel["target_6h"] = (
        panel.future_min_return_6h <= -panel.adverse_threshold_6h
    ).astype(float)
    panel["target_24h"] = (
        panel.future_min_return_24h <= -panel.adverse_threshold_24h
    ).astype(float)
    panel["target_72h"] = (
        (panel.future_close_return_72h <= -panel.adverse_threshold_72h)
        & (panel.future_below_current_fraction_72h >= 2.0 / 3.0)
    ).astype(float)
    mature = panel.future_close_return_72h.notna()
    panel["target"] = panel.target_72h.where(mature)
    # Compatibility names used by the shared trainer and audit tooling.
    panel["future_min_return"] = panel.future_min_return_72h
    panel["adverse_threshold"] = panel.adverse_threshold_72h
    panel["label_ready_ts"] = panel.signal_ts.astype("int64") + 72 * HOUR
    return panel.sort_values(["signal_ts", "pair"]).reset_index(drop=True)


def select_grid_sequence(
    candles: Mapping[str, pd.DataFrame], windows: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select weekly Grid parameters using only the legacy gate and prior 14 days."""
    gate = mechanism1_gate(candles)
    previous = INITIAL_APPROVED
    selections, audit = [], []
    for row in windows.itertuples(index=False):
        selected, evaluations = select_candidate(
            slice_window(dict(candles), int(row.train_start), int(row.train_end)),
            0.0,
            taker_fee=TAKER_FEE,
            require_eligible=False,
            technical_buy_gate=gate,
            cost_floor_enabled=True,
            inventory_exit_policy=POLICY,
        )
        eligible = int(evaluations.attrs.get("eligible_count", 0))
        retained = eligible == 0
        if retained:
            selected = previous
        else:
            previous = selected
        selections.append({**row._asdict(), **asdict(selected)})
        audit.append({
            "fold": int(row.fold),
            "eligible_candidates": eligible,
            "parameters_retained": retained,
            "best_training_score": float(evaluations.score.max()),
        })
        print(f"GRID {int(row.fold):02d}/{len(windows)}", flush=True)
    return pd.DataFrame(selections), pd.DataFrame(audit)


def frozen_grid_sequence(
    windows: pd.DataFrame, source_weekly_results: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carry forward the most recently available approved Grid without lookahead."""
    source = pd.read_csv(source_weekly_results)
    source = source[source.scenario == "new"].sort_values("test_start").drop_duplicates(
        "test_start", keep="last"
    )
    parameter_columns = [
        "half_range", "min_spread", "take_profit", "move_threshold",
        "move_cooldown_seconds",
    ]
    missing = set(["test_start", *parameter_columns]).difference(source.columns)
    if missing:
        raise RuntimeError(f"Weekly Grid source is missing {sorted(missing)}")
    selections, audit = [], []
    for row in windows.itertuples(index=False):
        eligible = source[source.test_start <= int(row.test_start)].tail(1)
        if eligible.empty:
            selected = INITIAL_APPROVED
            approved_at = None
            origin = "initial_approved"
        else:
            source_row = eligible.iloc[0]
            selected = Candidate(
                float(source_row.half_range), float(source_row.min_spread),
                float(source_row.take_profit), float(source_row.move_threshold),
                int(source_row.move_cooldown_seconds),
            )
            approved_at = int(source_row.test_start)
            origin = "latest_preexisting_approved_week"
        if approved_at is not None and approved_at > int(row.test_start):
            raise AssertionError("Grid schedule used a future approval")
        selections.append({**row._asdict(), **asdict(selected)})
        audit.append({
            "fold": int(row.fold), "source": origin,
            "approved_test_start": approved_at,
            "approval_not_after_fold_start": approved_at is None or approved_at <= int(row.test_start),
        })
    return pd.DataFrame(selections), pd.DataFrame(audit)


def train_walk_forward(
    panel: pd.DataFrame, selections: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fit one fixed shared XGB configuration and calibrate fold-local thresholds."""
    config = next(item for item in xgb_configurations() if item["config_id"] == "xgb_21")
    features = list(ALL_FEATURES)
    predictions, audits, importance = [], [], []
    for block in selections.itertuples(index=False):
        mature, core, validation = split_mature_training(panel, int(block.train_end))
        testing = panel[
            (panel.signal_ts >= int(block.test_start))
            & (panel.signal_ts < int(block.test_end))
        ].copy()
        model, fit_audit = fit_one_group(config, features, mature, core, validation)
        test_probability = model.predict_proba(testing[features])[:, 1]
        calibration_probability = model.predict_proba(validation[features])[:, 1]
        calibration = validation[["pair", "signal_ts"]].copy()
        calibration["probability"] = calibration_probability
        entry, recovery = {}, {}
        for pair in PAIRS:
            values = calibration.loc[calibration.pair == pair, "probability"]
            if values.empty:
                raise RuntimeError(f"Missing calibration probabilities for {pair}")
            entry[pair] = float(values.quantile(ENTRY_QUANTILE))
            recovery[pair] = float(values.quantile(RECOVERY_QUANTILE))
        keep = [
            "pair", "bar_open_utc", "bar_open_ts", "signal_ts", "label_ready_ts",
            "last_complete_1h_ts", "last_complete_4h_ts", "target",
            "target_6h", "target_24h", "target_72h",
            "future_min_return_6h", "future_min_return_24h", "future_min_return_72h",
            "future_close_return_72h", "future_below_current_fraction_72h",
            "adverse_threshold_6h", "adverse_threshold_24h", "adverse_threshold_72h",
            "roc_48h_4h", "sqzmom_pct_4h", "sqzmom_value_4h", "sqzmom_slope_4h",
            "sqzmom_improving_4h", "close_at_signal",
        ]
        out = testing[keep].copy()
        out["algorithm"] = "xgb_21"
        out["architecture"] = "shared"
        out["variant"] = VARIANT
        out["probability"] = np.asarray(test_probability, dtype=float)
        out["entry_threshold"] = out.pair.map(entry)
        out["recovery_threshold"] = out.pair.map(recovery)
        out["period"] = PERIOD
        out["fold"] = int(block.fold)
        predictions.append(out)
        gains = np.asarray(model.feature_importances_, dtype=float)
        if gains.sum() > 0:
            gains = gains / gains.sum()
        importance.extend({
            "fold": int(block.fold), "feature": feature, "gain_importance": float(gain)
        } for feature, gain in zip(features, gains))
        audits.append({
            "fold": int(block.fold),
            "train_cutoff_ts": int(block.train_end),
            "mature_rows": len(mature),
            "target_rate": float(mature.target.mean()),
            "train_last_label_ready_ts": int(mature.label_ready_ts.max()),
            "calibration_first_signal_ts": int(validation.signal_ts.min()),
            "calibration_last_signal_ts": int(validation.signal_ts.max()),
            "test_first_signal_ts": int(testing.signal_ts.min()),
            **fit_audit,
            **{f"{pair}_entry_threshold": entry[pair] for pair in PAIRS},
            **{f"{pair}_recovery_threshold": recovery[pair] for pair in PAIRS},
        })
        print(f"XGB  {int(block.fold):02d}/{len(selections)}", flush=True)
    predictions_frame = pd.concat(predictions, ignore_index=True)
    if not np.isfinite(predictions_frame.probability).all():
        raise AssertionError("Non-finite walk-forward probability")
    audit_frame = pd.DataFrame(audits)
    if not bool((audit_frame.train_last_label_ready_ts <= audit_frame.train_cutoff_ts).all()):
        raise AssertionError("A weekly fit used an immature 72-hour label")
    return predictions_frame, audit_frame, pd.DataFrame(importance)


def build_adaptive_buy_gate(
    predictions: pd.DataFrame, start_ts: int, end_ts: int
) -> tuple[dict[str, dict[int, bool]], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    gates: dict[str, dict[int, bool]] = {pair: {} for pair in PAIRS}
    states, events, intervals = [], [], []
    for pair in PAIRS:
        records = list(
            predictions[
                (predictions.pair == pair)
                & (predictions.signal_ts >= start_ts)
                & (predictions.signal_ts < end_ts)
            ].sort_values("signal_ts").itertuples(index=False)
        )
        if not records:
            raise RuntimeError(f"No {pair} predictions for fold")
        state = PairGateState()
        interval_start = None
        for index, row in enumerate(records):
            state, signal = advance_pair_gate(
                pair=pair,
                probability=float(row.probability),
                entry_threshold=float(row.entry_threshold),
                recovery_threshold=float(row.recovery_threshold),
                signal_ts=int(row.signal_ts),
                previous=state,
                model_version=MODEL_VERSION,
            )
            right = min(
                int(records[index + 1].signal_ts) if index + 1 < len(records) else end_ts,
                end_ts,
            )
            for timestamp in range(max(int(row.signal_ts), start_ts), right, FIVE_MINUTES):
                gates[pair][timestamp] = bool(signal["buy_enabled"])
            states.append({
                "variant": VARIANT, "architecture": "shared", "pair": pair,
                "signal_ts": int(row.signal_ts), "probability": float(row.probability),
                "entry_threshold": float(row.entry_threshold),
                "recovery_threshold": float(row.recovery_threshold),
                "risk_off_active": bool(state.risk_off_active),
                "buy_enabled": bool(signal["buy_enabled"]),
                "consecutive_recovery_bars": int(state.consecutive_recovery_bars),
                "risk_off_since_ts": state.risk_off_since,
                "transition": signal["transition"], "reason": signal["reason"],
                "event_id": signal["event_id"],
                "last_complete_1h_ts": int(row.last_complete_1h_ts),
                "last_complete_4h_ts": int(row.last_complete_4h_ts),
            })
            if signal["transition"] == "enter":
                interval_start = int(row.signal_ts)
            elif signal["transition"] == "recover" and interval_start is not None:
                intervals.append({
                    "pair": pair, "start_ts": interval_start, "end_ts": int(row.signal_ts),
                    "duration_hours": (int(row.signal_ts) - interval_start) / HOUR,
                    "end_reason": "recover",
                })
                interval_start = None
            if signal["transition"] in {"enter", "recover"}:
                events.append({
                    "timestamp": int(row.signal_ts), "pair": pair,
                    "event": signal["transition"], "probability": float(row.probability),
                    "entry_threshold": float(row.entry_threshold),
                    "recovery_threshold": float(row.recovery_threshold),
                    "event_id": signal["event_id"],
                })
        if interval_start is not None:
            intervals.append({
                "pair": pair, "start_ts": interval_start, "end_ts": int(end_ts),
                "duration_hours": (int(end_ts) - interval_start) / HOUR,
                "end_reason": "weekly_reinitialization",
            })
            last = records[-1]
            events.append({
                "timestamp": int(end_ts), "pair": pair,
                "event": "weekly_reinitialization", "probability": float(last.probability),
                "entry_threshold": float(last.entry_threshold),
                "recovery_threshold": float(last.recovery_threshold),
                "event_id": f"weekly-reset-{pair}-{end_ts}",
            })
    return gates, pd.DataFrame(states), pd.DataFrame(events), pd.DataFrame(intervals)


def replay_adaptive(
    candles: Mapping[str, pd.DataFrame], selections: pd.DataFrame,
    predictions: pd.DataFrame,
) -> dict[str, Any]:
    weekly, pair_rows, curves, trades = [], [], [], []
    states, gate_events, intervals, stop_rows = [], [], [], []
    cumulative = 0.0
    for selection in selections.itertuples(index=False):
        fold_predictions = predictions[predictions.fold == int(selection.fold)]
        gate, fold_states, fold_events, fold_intervals = build_adaptive_buy_gate(
            fold_predictions, int(selection.test_start), int(selection.test_end)
        )
        result, curve, pair_metrics, trade_frame, stop = simulate_fold(
            candles, selection, gate, record_details=True
        )
        weekly.append({
            "scenario": VARIANT, **selection._asdict(), **result, **stop,
        })
        pair_rows.extend({
            "scenario": VARIANT, "period": PERIOD, "fold": int(selection.fold),
            "pair": pair, **metrics,
        } for pair, metrics in pair_metrics.items())
        if not curve.empty:
            curve = curve.copy()
            curve["scenario"], curve["period"], curve["fold"] = VARIANT, PERIOD, int(selection.fold)
            curve["cumulative_oos_pnl"] = cumulative + curve.equity - INITIAL_EQUITY
            curves.append(curve)
        if not trade_frame.empty:
            trade_frame["scenario"], trade_frame["period"], trade_frame["fold"] = VARIANT, PERIOD, int(selection.fold)
            trades.append(trade_frame)
        for frame in (fold_states, fold_events, fold_intervals):
            if not frame.empty:
                frame["period"], frame["fold"] = PERIOD, int(selection.fold)
        states.append(fold_states); gate_events.append(fold_events); intervals.append(fold_intervals)
        for item in trade_frame.to_dict("records") if not trade_frame.empty else []:
            if item.get("reason") == "pair_breaker_flatten":
                stop_rows.append({
                    "scenario": VARIANT, "fold": int(selection.fold), "scope": item["pair"],
                    "kind": "pair_stop", "start_ts": int(item["timestamp"]),
                    "end_ts": int(selection.test_end),
                })
        if result["liquidated"] and not curve.empty:
            stop_rows.append({
                "scenario": VARIANT, "fold": int(selection.fold), "scope": "PORTFOLIO",
                "kind": "portfolio_stop", "start_ts": int(curve.timestamp.iloc[-1]),
                "end_ts": int(selection.test_end),
            })
        cumulative += float(result["net_pnl_quote"])
    weekly_frame, pair_frame = pd.DataFrame(weekly), pd.DataFrame(pair_rows)
    summary = aggregate_rows(weekly, pair_rows)
    summary["risk_off_pair_hours"] = float(pair_frame.technical_risk_off_seconds.sum() / HOUR)
    summary["momentum_stop_exits"] = int(weekly_frame.momentum_stop_exits.sum())
    return {
        "summary": summary, "weekly": weekly_frame, "pairs": pair_frame,
        "equity": pd.concat(curves, ignore_index=True),
        "trades": pd.concat(trades, ignore_index=True) if trades else pd.DataFrame(),
        "states": pd.concat(states, ignore_index=True),
        "events": pd.concat(gate_events, ignore_index=True) if gate_events else pd.DataFrame(),
        "intervals": pd.concat(intervals, ignore_index=True) if intervals else pd.DataFrame(),
        "stops": pd.DataFrame(stop_rows),
    }


def june_diagnostic(
    candles: Mapping[str, pd.DataFrame], states: pd.DataFrame, intervals: pd.DataFrame
) -> pd.DataFrame:
    start = int(pd.Timestamp("2026-06-01T00:00:00Z").timestamp())
    end = int(pd.Timestamp("2026-06-06T00:00:00Z").timestamp())
    old_entries = {"BTC-FDUSD": "2026-06-04 10:00 UTC", "ETH-FDUSD": "2026-06-04 11:00 UTC"}
    rows = []
    for pair in PAIRS:
        raw = candles[pair]
        price = raw[(raw.timestamp >= start) & (raw.timestamp < end)]
        entries = states[
            (states.pair == pair) & (states.signal_ts >= start)
            & (states.signal_ts < end) & (states.transition == "enter")
        ].sort_values("signal_ts")
        pair_intervals = intervals[intervals.pair == pair].sort_values(["start_ts", "end_ts"])
        # Merge touching weekly intervals so weekly validation resets do not hide
        # the effective continuous Risk-off start seen by the user.
        merged: list[list[int]] = []
        for interval in pair_intervals.itertuples(index=False):
            left, right = int(interval.start_ts), int(interval.end_ts)
            if merged and left <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], right)
            else:
                merged.append([left, right])
        covering = next((item for item in merged if item[0] < end and item[1] > start), None)
        active_at_start = bool(covering and covering[0] <= start < covering[1])
        entry_state = None
        if covering:
            matched = states[
                (states.pair == pair) & (states.signal_ts == int(covering[0]))
                & (states.transition == "enter")
            ].sort_values("fold").head(1)
            entry_state = matched.iloc[0] if not matched.empty else None
        rows.append({
            "pair": pair,
            "interval_start_utc": "2026-06-01 00:00 UTC",
            "interval_end_utc": "2026-06-06 00:00 UTC",
            "price_return_pct": (float(price.close.iloc[-1]) / float(price.close.iloc[0]) - 1) * 100,
            "minimum_return_pct": (float(price.low.min()) / float(price.close.iloc[0]) - 1) * 100,
            "old_v1_first_entry_utc": old_entries[pair],
            "persistent_risk_off_at_interval_start": active_at_start,
            "persistent_effective_entry_utc": (
                pd.to_datetime(int(covering[0]), unit="s", utc=True).strftime("%Y-%m-%d %H:%M UTC")
                if covering else None
            ),
            "persistent_effective_exit_utc": (
                pd.to_datetime(int(covering[1]), unit="s", utc=True).strftime("%Y-%m-%d %H:%M UTC")
                if covering else None
            ),
            "persistent_entry_probability": (
                float(entry_state.probability) if entry_state is not None else np.nan
            ),
            "persistent_entry_threshold": (
                float(entry_state.entry_threshold) if entry_state is not None else np.nan
            ),
            "new_entries_inside_interval": len(entries),
        })
    return pd.DataFrame(rows)


def _price_frame(path: Path, start_ts: int, end_ts: int) -> pd.DataFrame:
    frame = pd.read_csv(path, usecols=["timestamp", "close"])
    frame["timestamp"] = frame.timestamp.astype("int64")
    frame = frame[(frame.timestamp >= start_ts) & (frame.timestamp < end_ts)].copy()
    frame["time"] = pd.to_datetime(frame.timestamp, unit="s", utc=True)
    return frame.sort_values("timestamp")


def build_plotly(
    cache_dir: Path, output_dir: Path, states: pd.DataFrame,
    events: pd.DataFrame, intervals: pd.DataFrame, metrics: pd.DataFrame,
    june: pd.DataFrame,
) -> Path:
    prices = {
        pair: _price_frame(cache_dir / f"binance_{pair}_5m.csv", START_TS, END_TS)
        for pair in PAIRS
    }
    event_parts = []
    for pair in PAIRS:
        item = events[events.pair == pair].sort_values("timestamp").copy()
        item = pd.merge_asof(
            item, prices[pair][["timestamp", "close"]], on="timestamp", direction="backward"
        )
        item["time"] = pd.to_datetime(item.timestamp, unit="s", utc=True)
        event_parts.append(item)
    plotted_events = pd.concat(event_parts, ignore_index=True).sort_values(["timestamp", "pair"])
    plotted_events["time_utc"] = plotted_events.time.dt.strftime("%Y-%m-%d %H:%M UTC")
    plotted_events.to_csv(output_dir / "plotly_entry_exit_events.csv", index=False)

    figure = make_subplots(
        rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.045,
        row_heights=[0.28, 0.20, 0.28, 0.20],
        subplot_titles=(
            "BTC-FDUSD price", "BTC 72h 持续下跌概率",
            "ETH-FDUSD price", "ETH 72h 持续下跌概率",
        ),
    )
    for pair, price_row, probability_row in (("BTC-FDUSD", 1, 2), ("ETH-FDUSD", 3, 4)):
        label = PAIR_LABEL[pair]
        price = prices[pair]
        state = states[states.pair == pair].sort_values("signal_ts").copy()
        state["time"] = pd.to_datetime(state.signal_ts, unit="s", utc=True)
        figure.add_trace(go.Scattergl(
            x=price.time, y=price.close, mode="lines", name=f"{label} close",
            line={"color": PAIR_COLOR[pair], "width": 1.35}, legendgroup=pair,
            hovertemplate="%{x|%Y-%m-%d %H:%M UTC}<br>Close %{y:,.4f} FDUSD<extra></extra>",
        ), row=price_row, col=1)
        figure.add_trace(go.Scattergl(
            x=state.time, y=state.probability, mode="lines", name=f"{label} risk probability",
            line={"color": RISK_COLOR, "width": 1.25}, legendgroup=f"{pair}-risk",
            customdata=np.column_stack((state.entry_threshold, state.recovery_threshold, state.transition)),
            hovertemplate=("%{x|%Y-%m-%d %H:%M UTC}<br>Probability %{y:.6f}"
                           "<br>Entry %{customdata[0]:.6f}<br>Recovery %{customdata[1]:.6f}"
                           "<br>State %{customdata[2]}<extra></extra>"),
        ), row=probability_row, col=1)
        figure.add_trace(go.Scatter(
            x=state.time, y=state.entry_threshold, mode="lines", name=f"{label} entry threshold",
            line={"color": "#374151", "width": 1, "dash": "dot"},
            legendgroup=f"{pair}-threshold",
            hovertemplate="%{x|%Y-%m-%d %H:%M UTC}<br>Entry %{y:.6f}<extra></extra>",
        ), row=probability_row, col=1)
        figure.add_trace(go.Scatter(
            x=state.time, y=state.recovery_threshold, mode="lines", name=f"{label} recovery threshold",
            line={"color": "#6B7280", "width": 1, "dash": "dash"},
            legendgroup=f"{pair}-threshold",
            hovertemplate="%{x|%Y-%m-%d %H:%M UTC}<br>Recovery %{y:.6f}<extra></extra>",
        ), row=probability_row, col=1)
        for event, event_label, symbol, color in (
            ("enter", "进入 Risk-off", "triangle-down", RISK_COLOR),
            ("recover", "退出 Risk-off", "triangle-up", RECOVERY_COLOR),
            ("weekly_reinitialization", "周度重置", "x", "#B7791F"),
        ):
            marked = plotted_events[(plotted_events.pair == pair) & (plotted_events.event == event)]
            figure.add_trace(go.Scatter(
                x=marked.time, y=marked.close, mode="markers", name=f"{label} {event_label}",
                marker={"symbol": symbol, "size": 10, "color": color,
                        "line": {"color": "#111827", "width": 0.7}},
                customdata=np.column_stack((marked.probability, marked.entry_threshold,
                                             marked.recovery_threshold, marked.time_utc)) if not marked.empty else None,
                hovertemplate=(f"<b>{event_label}</b><br>%{{customdata[3]}}<br>Price %{{y:,.4f}} FDUSD"
                               "<br>Probability %{customdata[0]:.6f}<br>Entry %{customdata[1]:.6f}"
                               "<br>Recovery %{customdata[2]:.6f}<extra></extra>"),
            ), row=price_row, col=1)
        for interval in intervals[intervals.pair == pair].itertuples(index=False):
            for chart_row in (price_row, probability_row):
                figure.add_vrect(
                    x0=pd.to_datetime(interval.start_ts, unit="s", utc=True),
                    x1=pd.to_datetime(interval.end_ts, unit="s", utc=True),
                    fillcolor=RISK_FILL, line_width=0, layer="below", row=chart_row, col=1,
                )
        for chart_row in (price_row, probability_row):
            figure.add_vrect(
                x0=pd.Timestamp("2026-06-01T00:00:00Z"), x1=pd.Timestamp("2026-06-06T00:00:00Z"),
                fillcolor="rgba(183,121,31,0.055)", line_width=1,
                line_color="rgba(183,121,31,0.65)", layer="below", row=chart_row, col=1,
            )
        figure.update_yaxes(title_text=f"{label} price (FDUSD)", row=price_row, col=1)
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
        template="plotly_white", height=1450, hovermode="x unified",
        margin={"l": 72, "r": 30, "t": 190, "b": 65},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.025, "x": 0, "font": {"size": 10}},
        font={"family": "Arial, Microsoft YaHei, sans-serif", "color": "#1F2937"},
    )
    plot = figure.to_html(
        full_html=False, include_plotlyjs=True, div_id="long-risk-gate-plot",
        config={"responsive": True, "displaylogo": False, "scrollZoom": True},
    )
    metric_rows = "".join(
        f"<tr><td>{html.escape(str(row.scenario))}</td><td>{row.oos_pnl_fdusd:+.4f}</td>"
        f"<td>{row.worst_drawdown_pct:.4f}%</td><td>{int(row.pair_stop_events)}</td>"
        f"<td>{int(row.portfolio_stop_events)}</td><td>{row.risk_off_pair_hours:.1f}</td></tr>"
        for row in metrics.itertuples(index=False)
    )
    june_rows = "".join(
        f"<tr><td>{html.escape(row.pair)}</td><td>{row.price_return_pct:+.2f}%</td>"
        f"<td>{row.minimum_return_pct:+.2f}%</td><td>{html.escape(str(row.old_v1_first_entry_utc))}</td>"
        f"<td>{html.escape(str(row.persistent_effective_entry_utc))}</td>"
        f"<td>{html.escape(str(row.persistent_effective_exit_utc))}</td>"
        f"<td>{'是' if row.persistent_risk_off_at_interval_start else '否'}</td></tr>"
        for row in june.itertuples(index=False)
    )
    event_rows = "".join(
        f"<tr><td>{html.escape(row.time_utc)}</td><td>{html.escape(row.pair)}</td>"
        f"<td>{html.escape(row.event)}</td><td>{row.probability:.6f}</td>"
        f"<td>{row.entry_threshold:.6f}</td><td>{row.recovery_threshold:.6f}</td>"
        f"<td>{row.close:,.4f}</td></tr>"
        for row in plotted_events.itertuples(index=False)
    )
    page = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>XGBoost持续下跌Risk-off门：180天诊断回测</title><style>
html,body{{max-width:100%;overflow-x:hidden}}body{{margin:0;background:#fff;color:#1f2937;font-family:Arial,'Microsoft YaHei',sans-serif}}
main{{box-sizing:border-box;width:100%;max-width:1540px;margin:auto;padding:16px}}h1{{font-size:26px;margin:8px 0 4px;overflow-wrap:anywhere}}h2{{margin-top:28px}}
.sub{{color:#4b5563;margin:0 0 12px;overflow-wrap:anywhere;word-break:break-word}}.note{{padding:12px 14px;background:#fff7ed;border-left:4px solid #c2410c;line-height:1.55;overflow-wrap:anywhere}}
#long-risk-gate-plot,.plotly-graph-div{{width:100%!important;max-width:100%!important}}
.table-wrap{{overflow:auto;border:1px solid #e5e7eb;border-radius:8px;margin-top:10px}}table{{border-collapse:collapse;width:100%;font-size:13px}}
th,td{{padding:8px 10px;border-bottom:1px solid #e5e7eb;text-align:right;white-space:nowrap}}th{{background:#f9fafb;position:sticky;top:0}}
th:first-child,td:first-child,th:nth-child(2),td:nth-child(2),th:nth-child(3),td:nth-child(3){{text-align:left}}
@media(max-width:600px){{main{{padding:8px}}h1{{font-size:19px}}.sub,.note{{font-size:12px}}}}
</style></head><body><main><h1>XGBoost持续下跌门：180天诊断回测</h1>
<p class="sub">2026-02-01 15:00 至 2026-07-31 15:00 UTC｜完整180天｜每周滚动训练与Grid重置</p>
<div class="note"><b>证据边界：</b>6月1–5日已经被查看，因此这是诊断性回放，不是全新样本外证据。橙色阴影为Risk-off；橙色倒三角是进入，蓝色正三角是模型恢复，金色×是周度重置。金色细框标出6月1–5日。</div>
{plot}<h2>180天Grid结果</h2><div class="table-wrap"><table><thead><tr><th>方案</th><th>收益 FDUSD</th><th>最大回撤</th><th>单对停止</th><th>组合停止</th><th>Risk-off pair-hours</th></tr></thead><tbody>{metric_rows}</tbody></table></div>
<h2>6月1–5日覆盖诊断</h2><div class="table-wrap"><table><thead><tr><th>Pair</th><th>区间收益</th><th>最低收益</th><th>旧版首次进入</th><th>持续版有效进入</th><th>持续版有效退出</th><th>6月1日已Risk-off</th></tr></thead><tbody>{june_rows}</tbody></table></div>
<h2>具体进入与退出时间（UTC）</h2><div class="table-wrap"><table><thead><tr><th>UTC</th><th>Pair</th><th>事件</th><th>概率</th><th>进入阈值</th><th>恢复阈值</th><th>价格</th></tr></thead><tbody>{event_rows}</tbody></table></div>
</main><script>(function(){{
const chart=document.getElementById('long-risk-gate-plot');
function adapt(){{if(!chart||!window.Plotly)return;const mobile=window.innerWidth<=600;chart.style.height=mobile?'1720px':'1450px';
Plotly.relayout(chart,mobile?{{width:Math.max(window.innerWidth-16,320),height:1720,
'margin.l':52,'margin.r':12,'margin.t':430,'margin.b':65,'legend.font.size':9,
'legend.orientation':'v','legend.x':0.02,'legend.y':1.02}}:
{{width:Math.min(window.innerWidth-32,1540),height:1450,'margin.l':72,'margin.r':30,
'margin.t':190,'margin.b':65,'legend.font.size':10,'legend.y':1.025,
'legend.orientation':'h','legend.x':0}});Plotly.Plots.resize(chart);}}
window.addEventListener('load',adapt);let timer;window.addEventListener('resize',function(){{clearTimeout(timer);timer=setTimeout(adapt,120);}});
}})();</script></body></html>"""
    output = output_dir / "xgboost_persistent_risk_gate_180d_plotly.html"
    output.write_text(page, encoding="utf-8")
    return output


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    candles, quality = load_candles(args.cache_dir)
    quality.to_csv(args.output_dir / "data_quality.csv", index=False)
    windows = diagnostic_windows()
    selections_path = args.output_dir / "grid_selections.csv"
    grid_audit_path = args.output_dir / "grid_selection_audit.csv"
    if args.resume and selections_path.exists() and grid_audit_path.exists():
        selections = pd.read_csv(selections_path)
        grid_audit = pd.read_csv(grid_audit_path)
    else:
        selections, grid_audit = frozen_grid_sequence(windows, args.source_weekly_results)
        selections.to_csv(selections_path, index=False)
        grid_audit.to_csv(grid_audit_path, index=False)

    panel_path = args.output_dir / "multi_horizon_feature_panel.csv.gz"
    if args.resume and panel_path.exists():
        panel = pd.read_csv(panel_path)
    else:
        panel = build_multi_horizon_panel(candles)
        panel.to_csv(panel_path, index=False, compression="gzip")

    predictions_path = args.output_dir / "walk_forward_predictions.csv.gz"
    audit_path = args.output_dir / "training_audit.csv"
    importance_path = args.output_dir / "gain_feature_importance.csv"
    if args.resume and predictions_path.exists() and audit_path.exists() and importance_path.exists():
        predictions = pd.read_csv(predictions_path)
        training_audit = pd.read_csv(audit_path)
        importance = pd.read_csv(importance_path)
    else:
        predictions, training_audit, importance = train_walk_forward(panel, selections)
        predictions.to_csv(predictions_path, index=False, compression="gzip")
        training_audit.to_csv(audit_path, index=False)
        importance.to_csv(importance_path, index=False)

    baseline = replay(
        candles, selections, scenario="Mechanism 1 (BTC ROC/SQZMOM)",
        baseline_gate=mechanism1_gate(candles), record_details=True,
    )
    model = replay_adaptive(candles, selections, predictions)
    pd.concat([baseline["weekly"], model["weekly"]], ignore_index=True).to_csv(
        args.output_dir / "weekly_results.csv", index=False
    )
    pd.concat([baseline["pairs"], model["pairs"]], ignore_index=True).to_csv(
        args.output_dir / "pair_results.csv", index=False
    )
    pd.concat([baseline["equity"], model["equity"]], ignore_index=True).to_csv(
        args.output_dir / "equity_curves.csv.gz", index=False, compression="gzip"
    )
    pd.concat([baseline["trades"], model["trades"]], ignore_index=True).to_csv(
        args.output_dir / "trade_events.csv.gz", index=False, compression="gzip"
    )
    pd.concat([baseline["stops"], model["stops"]], ignore_index=True).to_csv(
        args.output_dir / "stop_events.csv", index=False
    )
    model["states"].to_csv(args.output_dir / "risk_states.csv.gz", index=False, compression="gzip")
    model["events"].to_csv(args.output_dir / "risk_gate_events.csv", index=False)
    model["intervals"].to_csv(args.output_dir / "risk_off_intervals.csv", index=False)
    metrics = pd.DataFrame([
        {"scenario": "Mechanism 1 (BTC ROC/SQZMOM)", **baseline["summary"]},
        {"scenario": VARIANT, **model["summary"]},
    ])
    metrics.to_csv(args.output_dir / "metrics.csv", index=False)
    june = june_diagnostic(candles, model["states"], model["intervals"])
    june.to_csv(args.output_dir / "june_1_5_diagnostic.csv", index=False)
    plot_path = build_plotly(
        args.cache_dir, args.output_dir, model["states"], model["events"],
        model["intervals"], metrics, june,
    )
    gates = {
        "covers_june_1_5_for_both_pairs": bool(june.persistent_risk_off_at_interval_start.all()),
        "pnl_better_than_mechanism1": bool(model["summary"]["oos_pnl_fdusd"] > baseline["summary"]["oos_pnl_fdusd"]),
        "pnl_positive": bool(model["summary"]["oos_pnl_fdusd"] > 0),
        "drawdown_not_worse": bool(model["summary"]["worst_drawdown_pct"] >= baseline["summary"]["worst_drawdown_pct"]),
        "pair_stops_fewer": bool(model["summary"]["pair_stop_events"] < baseline["summary"]["pair_stop_events"]),
        "portfolio_stops_not_increased": bool(model["summary"]["portfolio_stop_events"] <= baseline["summary"]["portfolio_stop_events"]),
        "portfolio_stops_zero": bool(model["summary"]["portfolio_stop_events"] == 0),
    }
    summary = {
        "schema": "xgboost-long-risk-gate-180d-diagnostic-v1",
        "model_version": MODEL_VERSION,
        "evidence_status": "diagnostic_replay_after_june_interval_review",
        "deployment_authorized": False,
        "verdict": "NEXT_STAGE_JOINT_VALIDATION" if all(gates.values()) else "NO-GO",
        "acceptance_gates": gates,
        "start_utc": pd.to_datetime(START_TS, unit="s", utc=True).isoformat(),
        "end_utc": pd.to_datetime(END_TS, unit="s", utc=True).isoformat(),
        "calendar_days": 180,
        "weekly_folds": len(selections),
        "label": {
            "target": "72h closing decline AND persistent price weakness",
            "72h_close_threshold": "max(3.0%, 3 x 1h ATR%)",
            "persistence": "at least two thirds of the next 72 hourly closes below current close",
            "maturity_hours": 72,
        },
        "thresholds": {
            "entry_quantile": ENTRY_QUANTILE,
            "recovery_quantile": RECOVERY_QUANTILE,
            "calibration": "fold-local last 14 mature days only",
        },
        "configuration": next(item for item in xgb_configurations() if item["config_id"] == "xgb_21"),
        "baseline": baseline["summary"],
        "model": model["summary"],
        "pnl_difference_fdusd": float(model["summary"]["oos_pnl_fdusd"] - baseline["summary"]["oos_pnl_fdusd"]),
        "june_1_5": june.to_dict("records"),
        "no_lookahead_checks": {
            "all_training_labels_mature": bool(
                (training_audit.train_last_label_ready_ts <= training_audit.train_cutoff_ts).all()
            ),
            "calibration_precedes_test": bool(
                (training_audit.calibration_last_signal_ts < training_audit.test_first_signal_ts).all()
            ),
            "prediction_probabilities_finite": bool(np.isfinite(predictions.probability).all()),
            "prediction_probabilities_in_unit_interval": bool(predictions.probability.between(0, 1).all()),
            "xgboost_gate_has_no_momentum_stop_exit": bool(model["summary"]["momentum_stop_exits"] == 0),
        },
        "artifacts": {"plotly_html": str(plot_path)},
        "input_hashes": {
            "candles": {
                pair: sha256_file(args.cache_dir / f"binance_{pair}_5m.csv") for pair in PAIRS
            },
            "feature_panel": sha256_file(panel_path),
            "grid_selections": sha256_file(selections_path),
            "source_weekly_results": sha256_file(args.source_weekly_results),
            "predictions": sha256_file(predictions_path),
        },
        "limitations": [
            "June 1-5 was inspected before this target was specified; results are diagnostic, not fresh OOS evidence.",
            "Weekly Grid validation reinitializes inventory and model gate state at each fold boundary.",
            "Funding, OI, taker-buy ratio and macro/FOMC history remain unavailable.",
        ],
    }
    write_json(args.output_dir / "summary.json", summary)
    print(json.dumps({
        "baseline": baseline["summary"], "model": model["summary"],
        "june_1_5": june.to_dict("records"), "plot": str(plot_path),
    }, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

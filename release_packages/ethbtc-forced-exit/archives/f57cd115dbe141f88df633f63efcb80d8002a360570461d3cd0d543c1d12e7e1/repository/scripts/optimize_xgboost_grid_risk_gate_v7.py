#!/usr/bin/env python3
"""Search and lock the XGBoost v7 long/short Grid BUY risk gate.

Mechanism 1 is replayed only as a comparison.  Every XGBoost scenario reaches
the Grid through a pair-specific BUY gate and can never request a sell.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import multiprocessing as mp
import os
from dataclasses import asdict
from itertools import product
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

import backtest_xgboost_dual_risk_gate_180d as dual
import backtest_xgboost_long_risk_gate_180d as base
import optimize_xgboost_dual_risk_gate_180d_v5 as v5
from compare_independent_gate_ml_stops import (
    ALL_FEATURES, FIVE_MINUTES, HOUR, PAIRS, SEPARATE_FEATURES,
    hourly_bars, load_candles,
)
from search_fdusd_inventory_exit import aggregate_rows
from tune_xgboost_grid_risk_gate_v1 import candidate_from_row
from tune_xgboost_momentum_stop_v2 import (
    fit_one_group, sha256_file, split_mature_training, write_json,
    xgb_configurations,
)
from validate_grid_live import crash_candles, simulate, slice_window


MODEL_VERSION = "xgboost-grid-risk-gate-v7"
OUTPUT_DIR = Path("results/backtests/xgboost_grid_risk_gate_v7")
SOURCE_DIR = Path("results/backtests/xgboost_dual_risk_gate_180d_v5")
START_TS = int(pd.Timestamp("2026-02-01T15:00:00Z").timestamp())
END_TS = int(pd.Timestamp("2026-07-31T15:00:00Z").timestamp())
ENTRY_QUANTILES = (0.90, 0.925, 0.95, 0.96, 0.97, 0.98, 0.985, 0.99)
ANCHOR_WINDOWS = (
    ("feb_03_06", int(pd.Timestamp("2026-02-03T00:00:00Z").timestamp()),
     int(pd.Timestamp("2026-02-07T00:00:00Z").timestamp())),
    ("jun_01_06", int(pd.Timestamp("2026-06-01T00:00:00Z").timestamp()),
     int(pd.Timestamp("2026-06-07T00:00:00Z").timestamp())),
)
TARGETS = {
    "long_72h": {"channel": "long", "label": "长期持续下跌 72h", "ready_hours": 72,
                 "definition_version": "long-endpoint-72h-below-two-thirds-v1"},
    "long_120h": {"channel": "long", "label": "长期持续下跌 120h", "ready_hours": 120,
                  "definition_version": "long-endpoint-120h-below-80pct-v1"},
    "short_1h_6h": {"channel": "short", "label": "1h快速插针", "ready_hours": 6,
                    "definition_version": "next-1h-low-rebound-closes-hours-2-through-6-v2"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prepare", "screen", "search", "finalize", "plot", "all"), default="all")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--cache-dir", type=Path, default=Path("data/backtesting_candles"))
    parser.add_argument("--source-dir", type=Path, default=SOURCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--screen-top", type=int, default=10)
    parser.add_argument("--workers", type=int, default=1)
    return parser.parse_args()


def _sha256_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(payload).hexdigest()


def _future_fraction_below(close: pd.Series, hours: int) -> np.ndarray:
    return (
        pd.concat([close.shift(-offset) for offset in range(1, hours + 1)], axis=1)
        .lt(close, axis=0).sum(axis=1).to_numpy(float) / float(hours)
    )


def relabel_panel(panel: pd.DataFrame, candles: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Build the two persistent targets and the strict 1h-drop/6h-rebound target."""
    item = panel.copy()
    derived = []
    for pair, bars in hourly_bars(candles).items():
        current = bars.close.to_numpy(float)
        future_lows_1h = bars.low.shift(-1).to_numpy(float)
        # The recovery must occur after the one-hour drop candle has completed;
        # including that candle's close would not establish OHLC event ordering.
        future_max_close_6h = pd.concat(
            [bars.close.shift(-offset) for offset in range(2, 7)], axis=1
        ).max(axis=1).to_numpy(float)
        derived.append(pd.DataFrame({
            "pair": pair,
            "bar_open_ts": bars.index.astype("int64") // 10**9,
            "future_close_return_72h_v7": bars.close.shift(-72).to_numpy(float) / current - 1.0,
            "future_close_return_120h_v7": bars.close.shift(-120).to_numpy(float) / current - 1.0,
            "future_below_fraction_72h_v7": _future_fraction_below(bars.close, 72),
            "future_below_fraction_120h_v7": _future_fraction_below(bars.close, 120),
            "future_min_return_1h_v7": future_lows_1h / current - 1.0,
            "future_max_close_return_6h_v7": future_max_close_6h / current - 1.0,
        }))
    columns = [
        "future_close_return_72h_v7", "future_close_return_120h_v7",
        "future_below_fraction_72h_v7", "future_below_fraction_120h_v7",
        "future_min_return_1h_v7", "future_max_close_return_6h_v7",
    ]
    item = item.drop(columns=[name for name in columns if name in item.columns])
    item = item.merge(pd.concat(derived, ignore_index=True), on=["pair", "bar_open_ts"],
                      how="left", validate="one_to_one")
    item["long_threshold_72h_v7"] = np.maximum(0.03, 3.0 * item.atr_pct)
    item["long_threshold_120h_v7"] = np.maximum(0.05, 5.0 * item.atr_pct)
    item["short_threshold_1h_v7"] = np.maximum(0.008, 2.0 * item.atr_pct)
    valid72 = item.future_close_return_72h_v7.notna()
    valid120 = item.future_close_return_120h_v7.notna()
    valid_short = item.future_min_return_1h_v7.notna() & item.future_max_close_return_6h_v7.notna()
    item["target_long_72h"] = (
        (item.future_close_return_72h_v7 <= -item.long_threshold_72h_v7)
        & (item.future_below_fraction_72h_v7 >= 2.0 / 3.0)
    ).astype(float).where(valid72)
    item["target_long_120h"] = (
        (item.future_close_return_120h_v7 <= -item.long_threshold_120h_v7)
        & (item.future_below_fraction_120h_v7 >= 0.80)
    ).astype(float).where(valid120)
    drop = item.future_min_return_1h_v7.abs()
    rebound = item.future_max_close_return_6h_v7 - item.future_min_return_1h_v7
    item["target_short_1h_6h"] = (
        (item.future_min_return_1h_v7 <= -item.short_threshold_1h_v7)
        & (rebound >= 0.50 * drop)
    ).astype(float).where(valid_short)
    item["label_ready_ts_long_72h"] = item.signal_ts.astype("int64") + 72 * HOUR
    item["label_ready_ts_long_120h"] = item.signal_ts.astype("int64") + 120 * HOUR
    item["label_ready_ts_short_1h_6h"] = item.signal_ts.astype("int64") + 6 * HOUR
    return item.sort_values(["signal_ts", "pair"]).reset_index(drop=True)


def working_target(panel: pd.DataFrame, target: str) -> pd.DataFrame:
    columns = {
        "long_72h": ("target_long_72h", "label_ready_ts_long_72h"),
        "long_120h": ("target_long_120h", "label_ready_ts_long_120h"),
        "short_1h_6h": ("target_short_1h_6h", "label_ready_ts_short_1h_6h"),
    }
    target_column, ready_column = columns[target]
    item = panel.copy()
    item["target"] = item[target_column]
    item["label_ready_ts"] = item[ready_column]
    return item


def target_quality(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for pair in PAIRS:
        frame = panel[panel.pair == pair]
        for target, (column, _) in {
            "long_72h": ("target_long_72h", 72),
            "long_120h": ("target_long_120h", 120),
            "short_1h_6h": ("target_short_1h_6h", 6),
        }.items():
            values = frame[column].dropna()
            rows.append({"pair": pair, "target": target, "rows": len(values),
                         "positive_rate": float(values.mean())})
    return pd.DataFrame(rows)


def _fit_block(
    working: pd.DataFrame, block: Any, config: Mapping[str, Any], architecture: str,
    test_start: int | None = None, test_end: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    mature, core, validation = split_mature_training(working, int(block.train_end))
    left = int(test_start if test_start is not None else block.test_start)
    right = int(test_end if test_end is not None else block.test_end)
    testing = working[(working.signal_ts >= left) & (working.signal_ts < right)].copy()
    features = list(ALL_FEATURES if architecture == "shared" else SEPARATE_FEATURES)
    if architecture == "shared":
        groups: Iterable[tuple[str, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]] = [
            ("ALL", mature, core, validation, testing)
        ]
    else:
        groups = [
            (pair, mature[mature.pair == pair], core[core.pair == pair],
             validation[validation.pair == pair], testing[testing.pair == pair])
            for pair in PAIRS
        ]
    predictions, calibration, audits = [], [], []
    for model_pair, train, early, validate, test in groups:
        model, fit_audit = fit_one_group(config, features, train, early, validate)
        predicted = test[["pair", "signal_ts", "target"]].copy()
        predicted["probability"] = model.predict_proba(test[features])[:, 1]
        predictions.append(predicted)
        calibrated = validate[["pair", "signal_ts", "target"]].copy()
        calibrated["probability"] = model.predict_proba(validate[features])[:, 1]
        calibration.append(calibrated)
        audits.append({
            "model_pair": model_pair, "train_cutoff_ts": int(block.train_end),
            "last_mature_label_ready_ts": int(train.label_ready_ts.max()),
            "last_calibration_signal_ts": int(validate.signal_ts.max()),
            "first_test_signal_ts": int(test.signal_ts.min()), **fit_audit,
        })
    return pd.concat(predictions, ignore_index=True), pd.concat(calibration, ignore_index=True), audits


def _needed_quantiles(channel: str) -> tuple[float, ...]:
    return tuple(sorted({
        *ENTRY_QUANTILES,
        *(max(0.50, quantile - 0.10) for quantile in ENTRY_QUANTILES),
    }))


def _attach_thresholds(prediction: pd.DataFrame, calibration: pd.DataFrame,
                       channel: str) -> pd.DataFrame:
    output = prediction.copy()
    for quantile in _needed_quantiles(channel):
        values = {
            pair: float(calibration.loc[calibration.pair == pair, "probability"].quantile(quantile))
            for pair in PAIRS
        }
        output[v5.quantile_column(quantile)] = output.pair.map(values)
    return output


def fixed_origin_prediction(panel: pd.DataFrame, target: str, config: Mapping[str, Any],
                            architecture: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    block = SimpleNamespace(train_end=START_TS, test_start=START_TS, test_end=END_TS, fold=0)
    predicted, calibration, audits = _fit_block(
        working_target(panel, target), block, config, architecture, START_TS, END_TS
    )
    predicted = _attach_thresholds(predicted, calibration, TARGETS[target]["channel"])
    predicted["strategy"] = target
    predicted["fold"] = 0
    return predicted, pd.DataFrame(audits)


def weekly_prediction(panel: pd.DataFrame, selections: pd.DataFrame, target: str,
                      config: Mapping[str, Any], architecture: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    working = working_target(panel, target)
    outputs, audit_rows = [], []
    for block in selections.itertuples(index=False):
        predicted, calibration, audits = _fit_block(working, block, config, architecture)
        predicted = _attach_thresholds(predicted, calibration, TARGETS[target]["channel"])
        predicted["strategy"] = target
        predicted["fold"] = int(block.fold)
        outputs.append(predicted)
        audit_rows.extend({"target": target, "config_id": config["config_id"],
                           "architecture": architecture, "fold": int(block.fold), **row}
                          for row in audits)
    result = pd.concat(outputs, ignore_index=True)
    audit = pd.DataFrame(audit_rows)
    if not np.isfinite(result.probability).all() or not result.probability.between(0, 1).all():
        raise AssertionError("walk-forward probabilities are invalid")
    if not (audit.last_mature_label_ready_ts <= audit.train_cutoff_ts).all():
        raise AssertionError("walk-forward fit used an immature label")
    return result, audit


def model_key(target: str, config_id: str, architecture: str) -> str:
    return f"{target}|{config_id}|{architecture}"


def fixed_gate(channel: str, entry: float) -> v5.GateParameters:
    if channel == "long":
        return v5.GateParameters(entry, max(0.50, entry - 0.10), 1, 4, 24, 120, 48)
    return v5.GateParameters(entry, max(0.50, entry - 0.10), 2, 1, 1, 6, 0)


def gate_id(channel: str, gate: v5.GateParameters) -> str:
    maximum = "none" if gate.maximum_hours is None else str(gate.maximum_hours)
    return (f"{channel}-q{gate.entry_quantile:g}-r{gate.recovery_quantile:g}"
            f"-e{gate.entry_bars}-x{gate.recovery_bars}-min{gate.minimum_hours}"
            f"-max{maximum}-cd{gate.cooldown_hours}")


def combine_gates(channel_gates: Sequence[Mapping[str, Mapping[int, bool]]]) -> dict[str, dict[int, bool]]:
    return dual.combine_channel_gates(channel_gates, START_TS, END_TS)


def replay_metrics(candles: Mapping[str, pd.DataFrame], selections: pd.DataFrame,
                   predictions: Sequence[pd.DataFrame], gates: Sequence[v5.GateParameters],
                   taker_fee: float = base.TAKER_FEE, slippage: float = 0.0) -> dict[str, Any]:
    channel_gates = []
    for prediction, gate in zip(predictions, gates):
        strategy = str(prediction.strategy.iloc[0])
        timeline, _, _, _ = v5.build_continuous_gate(prediction, strategy, gate, START_TS, END_TS)
        channel_gates.append(timeline)
    timeline = combine_gates(channel_gates)
    weekly, pair_rows, curves = [], [], []
    cumulative = 0.0
    for selection in selections.itertuples(index=False):
        result, curve, pairs = simulate(
            slice_window(dict(candles), int(selection.test_start), int(selection.test_end)),
            candidate_from_row(selection), maker_fee=0.0, taker_fee=taker_fee,
            slippage=slippage, order_refresh_seconds=7200, technical_buy_gate=timeline,
            momentum_stop_timeline=None, trade_log=None, risk_breakers_enabled=True,
            cost_floor_enabled=True, inventory_exit_policy=base.POLICY, record_curve=True,
        )
        pair_stop_events = sum(int(value.get("liquidations", 0)) for value in pairs.values())
        weekly.append({
            "fold": int(selection.fold), **result,
            "pair_stop_events": pair_stop_events, "pair_stop_hours": 0.0,
            "portfolio_stop_events": int(bool(result["liquidated"])),
            "portfolio_stop_hours": 0.0,
        })
        pair_rows.extend({"fold": int(selection.fold), "pair": pair, **value}
                         for pair, value in pairs.items())
        if not curve.empty:
            item = curve[["timestamp", "equity"]].copy()
            item["cumulative_oos_pnl"] = cumulative + item.equity - base.INITIAL_EQUITY
            curves.append(item)
        cumulative += float(result["net_pnl_quote"])
    summary = aggregate_rows(weekly, pair_rows)
    curve = pd.concat(curves, ignore_index=True).sort_values("timestamp")
    equity = base.INITIAL_EQUITY + curve.cumulative_oos_pnl.astype(float)
    summary["stitched_max_drawdown_pct"] = float((equity / equity.cummax() - 1.0).min() * 100.0)
    pair_frame = pd.DataFrame(pair_rows)
    summary["risk_off_pair_hours"] = float(pair_frame.technical_risk_off_seconds.sum() / HOUR)
    summary["pair_stop_events"] = int(pair_frame.liquidations.sum())
    summary["portfolio_stop_events"] = int(sum(bool(row["liquidated"]) for row in weekly))
    summary["momentum_stop_exits"] = int(sum(int(row["momentum_stop_exits"]) for row in weekly))
    return summary


def baseline_metrics(candles: Mapping[str, pd.DataFrame], selections: pd.DataFrame) -> dict[str, Any]:
    result = base.replay(candles, selections, scenario="Mechanism 1 (comparison only)",
                         baseline_gate=base.mechanism1_gate(candles), record_details=True)
    curve = result["equity"].sort_values(["fold", "timestamp"])
    equity = base.INITIAL_EQUITY + curve.cumulative_oos_pnl.astype(float)
    output = dict(result["summary"])
    output["stitched_max_drawdown_pct"] = float((equity / equity.cummax() - 1.0).min() * 100.0)
    return output


def score_frame(frame: pd.DataFrame) -> pd.DataFrame:
    item = frame.copy()
    item["profit_percentile"] = item.oos_pnl_fdusd.rank(method="average", pct=True)
    item["drawdown_percentile"] = item.stitched_max_drawdown_pct.rank(method="average", pct=True)
    item["objective_score"] = 0.5 * item.profit_percentile + 0.5 * item.drawdown_percentile
    sort = ["eligible", "objective_score", "portfolio_stop_events", "pair_stop_events",
            "risk_off_pair_hours", "oos_pnl_fdusd", "stitched_max_drawdown_pct"]
    ascending = [False, False, True, True, True, False, False]
    item = item.sort_values(sort, ascending=ascending).reset_index(drop=True)
    item["rank"] = np.arange(1, len(item) + 1)
    return item


def anchor_metrics(intervals: pd.DataFrame) -> dict[str, Any]:
    values: dict[str, Any] = {}
    all_pass = True
    overlap_total = 0.0
    for pair in PAIRS:
        group = intervals[intervals.pair == pair]
        values[f"{pair[:3]}_interval_count"] = int(len(group))
        all_pass = all_pass and len(group) <= 8
        for name, start, end in ANCHOR_WINDOWS:
            overlaps = group.assign(
                overlap=lambda x: np.maximum(0, np.minimum(x.end_ts, end) - np.maximum(x.start_ts, start))
            )
            coverage = float(overlaps.overlap.sum() / (end - start))
            timely = bool(((group.start_ts <= start + 12 * HOUR) & (group.end_ts > start)).any())
            values[f"{pair[:3]}_{name}_coverage"] = coverage
            values[f"{pair[:3]}_{name}_timely"] = timely
            overlap_total += float(overlaps.overlap.sum())
            all_pass = all_pass and timely and coverage >= 0.70
    total_seconds = float(intervals.end_ts.sub(intervals.start_ts).sum())
    outside_share = max(total_seconds - overlap_total, 0.0) / (len(PAIRS) * (END_TS - START_TS))
    values["long_outside_anchor_pair_hour_share"] = outside_share
    values["long_anchor_pass"] = bool(all_pass and outside_share <= 0.20)
    return values


def channel_overlap(long_states: pd.DataFrame, short_states: pd.DataFrame) -> float:
    left = long_states[["pair", "signal_ts", "risk_off_active"]].rename(columns={"risk_off_active": "long"})
    right = short_states[["pair", "signal_ts", "risk_off_active"]].rename(columns={"risk_off_active": "short"})
    merged = left.merge(right, on=["pair", "signal_ts"], how="inner")
    union = merged.long | merged.short
    return float((merged.long & merged.short).sum() / union.sum()) if union.any() else 0.0


def sampled_gates(channel: str) -> list[v5.GateParameters]:
    """Exhaust the approved state-machine grid with fixed 10-point hysteresis."""
    if channel == "long":
        combinations = product(
            ENTRY_QUANTILES, (1, 2), (4, 8), (12, 24),
            (72, 120, 168), (24, 48, 72),
        )
    else:
        combinations = product(ENTRY_QUANTILES, (1, 2), (1, 2), (1,), (2, 4, 6), (0, 2))
    gates = [
        v5.GateParameters(
            float(entry), max(0.50, float(entry) - 0.10), int(entry_bars),
            int(recovery_bars), int(minimum), int(maximum), int(cooldown),
        )
        for entry, entry_bars, recovery_bars, minimum, maximum, cooldown in combinations
    ]
    expected = 576 if channel == "long" else 192
    if len(gates) != expected or len({gate_id(channel, gate) for gate in gates}) != expected:
        raise AssertionError(f"expected {expected} unique {channel} gates, got {len(gates)}")
    return gates


def prediction_path(output_dir: Path, stage: str, key: str) -> Path:
    safe = key.replace("|", "__")
    return output_dir / "prediction_cache" / stage / f"{safe}.csv.gz"


def prediction_cache_metadata(
    args: argparse.Namespace, *, stage: str, target: str,
    config: Mapping[str, Any], architecture: str,
) -> dict[str, Any]:
    """Describe every immutable input used to produce a probability cache."""
    return {
        "schema": "xgboost-grid-risk-gate-v7-prediction-cache-v1",
        "stage": stage,
        "target": target,
        "architecture": architecture,
        "configuration_sha256": _sha256_json(dict(config)),
        "source_feature_panel_sha256": sha256_file(
            args.source_dir / "dual_target_feature_panel.csv.gz"
        ),
        "source_candle_sha256": {
            pair: sha256_file(args.cache_dir / f"binance_{pair}_5m.csv") for pair in PAIRS
        },
        "feature_schema_sha256": _sha256_json(
            list(ALL_FEATURES if architecture == "shared" else SEPARATE_FEATURES)
        ),
        "target_definition_version": TARGETS[target]["definition_version"],
        "grid_sequence_sha256": sha256_file(args.output_dir / "grid_selections.csv"),
        "prediction_pipeline_sha256": _sha256_json({
            "fit_block": inspect.getsource(_fit_block),
            "attach_thresholds": inspect.getsource(_attach_thresholds),
            "fit_one_group": inspect.getsource(fit_one_group),
        }),
    }


def _cache_meta_path(path: Path) -> Path:
    return path.with_name(path.name + ".metadata.json")


def write_csv_atomic(frame: pd.DataFrame, path: Path, **kwargs: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_csv(temporary, index=False, **kwargs)
    os.replace(temporary, path)


def load_prediction_cache(path: Path, expected: Mapping[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    metadata_path = _cache_meta_path(path)
    audit_path = path.with_suffix(".audit.csv")
    if not metadata_path.exists():
        raise ValueError(f"prediction cache metadata is missing: {metadata_path}")
    observed = json.loads(metadata_path.read_text(encoding="utf-8"))
    if observed != dict(expected):
        raise ValueError(f"prediction cache hash mismatch: {path}")
    prediction = pd.read_csv(path)
    audit = pd.read_csv(audit_path) if audit_path.exists() else pd.DataFrame()
    return prediction, audit


def save_prediction_cache(
    path: Path, prediction: pd.DataFrame, audit: pd.DataFrame,
    metadata: Mapping[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_csv_atomic(prediction, path, compression="gzip")
    write_csv_atomic(audit, path.with_suffix(".audit.csv"))
    write_json(_cache_meta_path(path), dict(metadata))


def screen_models(args: argparse.Namespace, panel: pd.DataFrame, candles: Mapping[str, pd.DataFrame],
                  selections: pd.DataFrame, baseline: Mapping[str, Any]) -> pd.DataFrame:
    target_path = args.output_dir / "model_screen_40x2x3x8.csv"
    existing = pd.read_csv(target_path) if args.resume and target_path.exists() else pd.DataFrame()
    done = set(existing.candidate_id) if not existing.empty else set()
    rows = existing.to_dict("records") if not existing.empty else []
    configs = xgb_configurations()
    for target, config, architecture in product(TARGETS, configs, ("shared", "separate")):
        key = model_key(target, config["config_id"], architecture)
        path = prediction_path(args.output_dir, "screen", key)
        metadata = prediction_cache_metadata(
            args, stage="screen", target=target, config=config, architecture=architecture
        )
        if args.resume and path.exists():
            prediction, _ = load_prediction_cache(path, metadata)
        else:
            prediction, audit = fixed_origin_prediction(panel, target, config, architecture)
            save_prediction_cache(path, prediction, audit, metadata)
        for entry in ENTRY_QUANTILES:
            gate = fixed_gate(TARGETS[target]["channel"], entry)
            candidate_id = f"{key}|{gate_id(TARGETS[target]['channel'], gate)}"
            if candidate_id in done:
                continue
            metrics = replay_metrics(candles, selections, [prediction], [gate])
            rows.append({
                "candidate_id": candidate_id, "model_key": key, "target": target,
                "channel": TARGETS[target]["channel"], "config_id": config["config_id"],
                "architecture": architecture, **asdict(gate), **metrics,
                "eligible": bool(metrics["oos_pnl_fdusd"] > baseline["oos_pnl_fdusd"]
                                 and metrics["stitched_max_drawdown_pct"] >= baseline["stitched_max_drawdown_pct"]),
            })
            write_csv_atomic(pd.DataFrame(rows), target_path)
        print(f"SCREEN {key}", flush=True)
    ranked = pd.concat(
        [score_frame(group) for _, group in pd.DataFrame(rows).groupby("channel")],
        ignore_index=True,
    ).sort_values(["channel", "rank"])
    write_csv_atomic(ranked, target_path)
    return ranked


def finalists(screen: pd.DataFrame, top: int) -> pd.DataFrame:
    best_per_model = screen.sort_values("rank").drop_duplicates("model_key", keep="first")
    return pd.concat([
        best_per_model[best_per_model.channel == channel].nsmallest(top, "rank")
        for channel in ("long", "short")
    ], ignore_index=True)


def load_weekly_predictions(args: argparse.Namespace, panel: pd.DataFrame,
                            selections: pd.DataFrame, selected: pd.DataFrame) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    configs = {item["config_id"]: item for item in xgb_configurations()}
    predictions: dict[str, pd.DataFrame] = {}
    audits = []
    for row in selected.itertuples(index=False):
        key = str(row.model_key)
        path = prediction_path(args.output_dir, "weekly", key)
        config = configs[str(row.config_id)]
        metadata = prediction_cache_metadata(
            args, stage="weekly", target=str(row.target), config=config,
            architecture=str(row.architecture),
        )
        if args.resume and path.exists():
            prediction, audit = load_prediction_cache(path, metadata)
        else:
            prediction, audit = weekly_prediction(
                panel, selections, str(row.target), config, str(row.architecture)
            )
            save_prediction_cache(path, prediction, audit, metadata)
        predictions[key] = prediction
        audits.append(audit)
        print(f"WEEKLY {key}", flush=True)
    return predictions, pd.concat(audits, ignore_index=True)


_REFINE_CANDLES: Mapping[str, pd.DataFrame] | None = None
_REFINE_SELECTIONS: pd.DataFrame | None = None
_REFINE_PREDICTIONS: Mapping[str, pd.DataFrame] | None = None
_REFINE_BASELINE: Mapping[str, Any] | None = None


def _initialize_replay_worker(
    candles: Mapping[str, pd.DataFrame], selections: pd.DataFrame,
    predictions: Mapping[str, pd.DataFrame], baseline: Mapping[str, Any],
) -> None:
    global _REFINE_CANDLES, _REFINE_SELECTIONS, _REFINE_PREDICTIONS, _REFINE_BASELINE
    _REFINE_CANDLES = candles
    _REFINE_SELECTIONS = selections
    _REFINE_PREDICTIONS = predictions
    _REFINE_BASELINE = baseline


def _refine_candidate(task: tuple[dict[str, Any], v5.GateParameters]) -> dict[str, Any]:
    model, gate = task
    if any(value is None for value in (
        _REFINE_CANDLES, _REFINE_SELECTIONS, _REFINE_PREDICTIONS, _REFINE_BASELINE
    )):
        raise RuntimeError("replay worker was not initialized")
    key, channel = str(model["model_key"]), str(model["channel"])
    prediction = _REFINE_PREDICTIONS[key]  # type: ignore[index]
    metrics = replay_metrics(
        _REFINE_CANDLES, _REFINE_SELECTIONS, [prediction], [gate]  # type: ignore[arg-type]
    )
    extra: dict[str, Any] = {}
    if channel == "long":
        _, _, _, intervals = v5.build_continuous_gate(
            prediction, str(model["target"]), gate, START_TS, END_TS
        )
        extra = anchor_metrics(intervals)
    eligible = bool(
        metrics["oos_pnl_fdusd"] > _REFINE_BASELINE["oos_pnl_fdusd"]  # type: ignore[index]
        and metrics["stitched_max_drawdown_pct"] >= _REFINE_BASELINE["stitched_max_drawdown_pct"]  # type: ignore[index]
        and (channel != "long" or extra.get("long_anchor_pass", False))
    )
    return {
        "candidate_id": f"{key}|{gate_id(channel, gate)}", **model,
        **asdict(gate), **metrics, **extra, "eligible": eligible,
    }


def refine_single(args: argparse.Namespace, candles: Mapping[str, pd.DataFrame],
                  selections: pd.DataFrame, selected: pd.DataFrame,
                  predictions: Mapping[str, pd.DataFrame], baseline: Mapping[str, Any]) -> pd.DataFrame:
    target_path = args.output_dir / "single_channel_refined_search.csv"
    existing = pd.read_csv(target_path) if args.resume and target_path.exists() else pd.DataFrame()
    done = set(existing.candidate_id) if not existing.empty else set()
    rows = existing.to_dict("records") if not existing.empty else []
    tasks: list[tuple[dict[str, Any], v5.GateParameters]] = []
    for model in selected.itertuples(index=False):
        record = {
            "model_key": str(model.model_key), "target": str(model.target),
            "channel": str(model.channel), "config_id": str(model.config_id),
            "architecture": str(model.architecture),
        }
        for gate in sampled_gates(record["channel"]):
            if f"{record['model_key']}|{gate_id(record['channel'], gate)}" not in done:
                tasks.append((record, gate))
    workers = max(1, int(args.workers))
    if workers == 1:
        _initialize_replay_worker(candles, selections, predictions, baseline)
        iterator = map(_refine_candidate, tasks)
        pool = None
    else:
        context = mp.get_context("spawn")
        pool = context.Pool(
            processes=workers, initializer=_initialize_replay_worker,
            initargs=(candles, selections, predictions, baseline),
            maxtasksperchild=128,
        )
        iterator = pool.imap(_refine_candidate, tasks, chunksize=2)
    try:
        for index, result in enumerate(iterator, 1):
            rows.append(result)
            if index % 8 == 0 or index == len(tasks):
                write_csv_atomic(pd.DataFrame(rows), target_path)
                print(f"REFINE {index:05d}/{len(tasks):05d}", flush=True)
    finally:
        if pool is not None:
            pool.close(); pool.join()
    ranked = pd.concat(
        [score_frame(group) for _, group in pd.DataFrame(rows).groupby("channel")],
        ignore_index=True,
    ).sort_values(["channel", "rank"])
    write_csv_atomic(ranked, target_path)
    return ranked


def gate_from_row(row: Mapping[str, Any]) -> v5.GateParameters:
    return v5.GateParameters(
        float(row["entry_quantile"]), float(row["recovery_quantile"]),
        int(row["entry_bars"]), int(row["recovery_bars"]), int(row["minimum_hours"]),
        int(row["maximum_hours"]), int(row["cooldown_hours"]),
    )


def dual_search(args: argparse.Namespace, candles: Mapping[str, pd.DataFrame],
                selections: pd.DataFrame, single: pd.DataFrame,
                predictions: Mapping[str, pd.DataFrame], baseline: Mapping[str, Any]) -> pd.DataFrame:
    path = args.output_dir / "dual_channel_search.csv"
    existing = pd.read_csv(path) if args.resume and path.exists() else pd.DataFrame()
    done = set(existing.candidate_id) if not existing.empty else set()
    rows = existing.to_dict("records") if not existing.empty else []
    top_long = single[single.channel == "long"].nsmallest(10, "rank")
    top_short = single[single.channel == "short"].nsmallest(10, "rank")
    for long_row, short_row in product(top_long.to_dict("records"), top_short.to_dict("records")):
        candidate_id = f"dual|{long_row['candidate_id']}||{short_row['candidate_id']}"
        if candidate_id in done:
            continue
        long_prediction = predictions[str(long_row["model_key"])]
        short_prediction = predictions[str(short_row["model_key"])]
        long_gate, short_gate = gate_from_row(long_row), gate_from_row(short_row)
        metrics = replay_metrics(candles, selections, [long_prediction, short_prediction], [long_gate, short_gate])
        _, long_states, _, long_intervals = v5.build_continuous_gate(
            long_prediction, str(long_row["target"]), long_gate, START_TS, END_TS
        )
        _, short_states, _, _ = v5.build_continuous_gate(
            short_prediction, str(short_row["target"]), short_gate, START_TS, END_TS
        )
        anchors = anchor_metrics(long_intervals)
        overlap = channel_overlap(long_states, short_states)
        eligible = bool(
            metrics["oos_pnl_fdusd"] > baseline["oos_pnl_fdusd"]
            and metrics["stitched_max_drawdown_pct"] >= baseline["stitched_max_drawdown_pct"]
            and anchors["long_anchor_pass"] and overlap <= 0.15
        )
        rows.append({
            "candidate_id": candidate_id, "long_candidate_id": long_row["candidate_id"],
            "short_candidate_id": short_row["candidate_id"],
            "long_model_key": long_row["model_key"], "short_model_key": short_row["model_key"],
            "long_target": long_row["target"], "short_target": short_row["target"],
            **{f"long_{key}": value for key, value in asdict(long_gate).items()},
            **{f"short_{key}": value for key, value in asdict(short_gate).items()},
            **metrics, **anchors, "active_jaccard": overlap, "eligible": eligible,
        })
        write_csv_atomic(pd.DataFrame(rows), path)
        print(f"DUAL {len(rows):03d}/100", flush=True)
    ranked = score_frame(pd.DataFrame(rows))
    write_csv_atomic(ranked, path)
    return ranked


def prefixed_gate(row: Mapping[str, Any], prefix: str) -> v5.GateParameters:
    return v5.GateParameters(
        float(row[f"{prefix}entry_quantile"]), float(row[f"{prefix}recovery_quantile"]),
        int(row[f"{prefix}entry_bars"]), int(row[f"{prefix}recovery_bars"]),
        int(row[f"{prefix}minimum_hours"]), int(row[f"{prefix}maximum_hours"]),
        int(row[f"{prefix}cooldown_hours"]),
    )


def final_result(candles: Mapping[str, pd.DataFrame], selections: pd.DataFrame,
                 winner: Mapping[str, Any], predictions: Mapping[str, pd.DataFrame]) -> dict[str, Any]:
    long_prediction = predictions[str(winner["long_model_key"])]
    short_prediction = predictions[str(winner["short_model_key"])]
    combined = pd.concat([long_prediction, short_prediction], ignore_index=True)
    gates = {
        str(winner["long_target"]): prefixed_gate(winner, "long_"),
        str(winner["short_target"]): prefixed_gate(winner, "short_"),
    }
    return v5.replay(candles, selections, combined, gates, tuple(gates), "XGBoost v7 dual OR gate")


def train_final_models(panel: pd.DataFrame, winner: Mapping[str, Any], output_dir: Path) -> tuple[Path, dict[str, Any]]:
    configs = {item["config_id"]: item for item in xgb_configurations()}
    bundle: dict[str, Any] = {"model_version": MODEL_VERSION, "channels": {}}
    for prefix, channel in (("long_", "long"), ("short_", "short")):
        key = str(winner[f"{channel}_model_key"])
        target, config_id, architecture = key.split("|")
        working = working_target(panel, target)
        mature, core, validation = split_mature_training(working, END_TS)
        features = list(ALL_FEATURES if architecture == "shared" else SEPARATE_FEATURES)
        groups = [
            ("ALL", mature, core, validation)
        ] if architecture == "shared" else [
            (pair, mature[mature.pair == pair], core[core.pair == pair], validation[validation.pair == pair])
            for pair in PAIRS
        ]
        models, thresholds = {}, {pair: {} for pair in PAIRS}
        for model_pair, train, early, validate in groups:
            model, _ = fit_one_group(configs[config_id], features, train, early, validate)
            models[model_pair] = model
            probability = model.predict_proba(validate[features])[:, 1]
            calibrated = validate[["pair"]].copy(); calibrated["probability"] = probability
            threshold_pairs = PAIRS if architecture == "shared" else (model_pair,)
            for pair in threshold_pairs:
                values = calibrated.loc[calibrated.pair == pair, "probability"]
                gate = prefixed_gate(winner, prefix)
                thresholds[pair] = {
                    "entry": float(values.quantile(gate.entry_quantile)),
                    "recovery": float(values.quantile(gate.recovery_quantile)),
                }
        bundle["channels"][channel] = {
            "target": target, "config_id": config_id, "architecture": architecture,
            "features": features, "models": models, "thresholds": thresholds,
            "gate": asdict(prefixed_gate(winner, prefix)),
        }
    path = output_dir / "models" / "xgboost_grid_risk_gate_v7.joblib"
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)
    return path, bundle


def bundle_probability(bundle: Mapping[str, Any], channel: str, rows: pd.DataFrame) -> np.ndarray:
    specification = bundle["channels"][channel]
    features = list(specification["features"])
    if specification["architecture"] == "shared":
        return specification["models"]["ALL"].predict_proba(rows[features])[:, 1]
    output = np.empty(len(rows), dtype=float)
    for pair in PAIRS:
        mask = rows.pair.eq(pair)
        output[mask.to_numpy()] = specification["models"][pair].predict_proba(
            rows.loc[mask, features]
        )[:, 1]
    return output


def write_model_diagnostics(
    output_dir: Path, bundle: Mapping[str, Any], panel: pd.DataFrame,
    predictions: Mapping[str, pd.DataFrame], winner: Mapping[str, Any],
) -> dict[str, Any]:
    classification_rows: list[dict[str, Any]] = []
    importance_rows: list[dict[str, Any]] = []
    for channel in ("long", "short"):
        key = str(winner[f"{channel}_model_key"])
        predicted = predictions[key]
        for pair in PAIRS:
            sample = predicted[predicted.pair.eq(pair)].dropna(subset=["target", "probability"])
            values = sample.target.astype(int)
            classification_rows.append({
                "channel": channel, "pair": pair, "rows": len(sample),
                "positive_rate": float(values.mean()),
                "roc_auc": float(roc_auc_score(values, sample.probability))
                if values.nunique() > 1 else float("nan"),
                "average_precision": float(average_precision_score(values, sample.probability))
                if values.nunique() > 1 else float("nan"),
            })
        specification = bundle["channels"][channel]
        features = list(specification["features"])
        per_model = []
        for model_pair, model in specification["models"].items():
            values = np.asarray(model.feature_importances_, dtype=float)
            per_model.append(values)
            for feature, gain in zip(features, values):
                importance_rows.append({
                    "channel": channel, "model_pair": model_pair,
                    "feature": feature, "gain": float(gain),
                })
        mean_gain = np.mean(per_model, axis=0)
        importance_rows.extend({
            "channel": channel, "model_pair": "MEAN",
            "feature": feature, "gain": float(gain),
        } for feature, gain in zip(features, mean_gain))
    pd.DataFrame(classification_rows).to_csv(output_dir / "classification_metrics.csv", index=False)
    pd.DataFrame(importance_rows).sort_values(
        ["channel", "model_pair", "gain"], ascending=[True, True, False]
    ).to_csv(output_dir / "xgboost_gain_feature_importance.csv", index=False)

    sample = panel[(panel.signal_ts < END_TS)].groupby("pair", group_keys=False).tail(32)
    roundtrip = joblib.load(output_dir / "models" / "xgboost_grid_risk_gate_v7.joblib")
    maximum_error = 0.0
    for channel in ("long", "short"):
        before = bundle_probability(bundle, channel, sample)
        after = bundle_probability(roundtrip, channel, sample)
        maximum_error = max(maximum_error, float(np.max(np.abs(before - after))))
    result = {
        "rows": int(len(sample)), "maximum_probability_absolute_error": maximum_error,
        "passed": bool(maximum_error <= 1e-12),
    }
    write_json(output_dir / "model_serialization_check.json", result)
    if not result["passed"]:
        raise AssertionError("serialized XGBoost probabilities changed")
    return result


def finalize(args: argparse.Namespace, panel: pd.DataFrame, candles: Mapping[str, pd.DataFrame],
             selections: pd.DataFrame, baseline: Mapping[str, Any], dual_ranked: pd.DataFrame,
             predictions: Mapping[str, pd.DataFrame], audit: pd.DataFrame) -> dict[str, Any]:
    winner = dual_ranked.iloc[0].to_dict()
    detailed = final_result(candles, selections, winner, predictions)
    metrics = dict(detailed["summary"])
    curve = detailed["equity"].sort_values(["fold", "timestamp"])
    equity = base.INITIAL_EQUITY + curve.cumulative_oos_pnl.astype(float)
    metrics["stitched_max_drawdown_pct"] = float((equity / equity.cummax() - 1.0).min() * 100.0)
    long_prediction = predictions[str(winner["long_model_key"])]
    short_prediction = predictions[str(winner["short_model_key"])]
    long_gate, short_gate = prefixed_gate(winner, "long_"), prefixed_gate(winner, "short_")
    crash_path = crash_candles(dict(candles), 0.15)
    crash_panel = relabel_panel(base.build_multi_horizon_panel(crash_path), crash_path)
    configs = {item["config_id"]: item for item in xgb_configurations()}
    crash_predictions = []
    for channel in ("long", "short"):
        target, config_id, architecture = str(winner[f"{channel}_model_key"]).split("|")
        predicted, _ = weekly_prediction(
            crash_panel, selections, target, configs[config_id], architecture
        )
        crash_predictions.append(predicted)
    pressure_rows = []
    for name, scenario_candles, scenario_predictions, taker_fee, slippage in (
        ("base", candles, [long_prediction, short_prediction], base.TAKER_FEE, 0.0),
        ("taker_150pct", candles, [long_prediction, short_prediction], base.TAKER_FEE * 1.5, 0.0),
        ("slippage_0_05pct", candles, [long_prediction, short_prediction], base.TAKER_FEE, 0.0005),
        ("slippage_0_10pct", candles, [long_prediction, short_prediction], base.TAKER_FEE, 0.0010),
        ("single_day_15pct_drop", crash_path, crash_predictions, base.TAKER_FEE, 0.0),
    ):
        stress = replay_metrics(
            scenario_candles, selections, scenario_predictions,
            [long_gate, short_gate], taker_fee=taker_fee, slippage=slippage,
        )
        pressure_rows.append({"scenario": name, **stress,
                              "no_stops": int(stress["pair_stop_events"]) == 0
                              and int(stress["portfolio_stop_events"]) == 0})
    pressure = pd.DataFrame(pressure_rows)
    pressure.to_csv(args.output_dir / "pressure_tests.csv", index=False)
    accepted = {
        "positive_profit": bool(metrics["oos_pnl_fdusd"] >= 0),
        "beats_mechanism1_profit": bool(metrics["oos_pnl_fdusd"] > baseline["oos_pnl_fdusd"]),
        "drawdown_not_worse": bool(metrics["stitched_max_drawdown_pct"] >= baseline["stitched_max_drawdown_pct"]),
        "zero_portfolio_stops": bool(int(metrics["portfolio_stop_events"]) == 0),
        "fewer_pair_stops": bool(int(metrics["pair_stop_events"]) < int(baseline["pair_stop_events"])),
        "long_anchor_pass": bool(winner["long_anchor_pass"]),
        "channel_overlap_pass": bool(float(winner["active_jaccard"]) <= 0.15),
        "all_pressure_scenarios_no_stops": bool(pressure.no_stops.all()),
    }
    deployment_allowed = bool(all(accepted.values()))
    model_path, bundle = train_final_models(panel, winner, args.output_dir)
    model_hash = sha256_file(model_path)
    feature_hash = _sha256_json({name: value["features"] for name, value in bundle["channels"].items()})
    serialization = write_model_diagnostics(
        args.output_dir, bundle, panel, predictions, winner
    )
    prediction_hashes = {
        channel: sha256_file(prediction_path(
            args.output_dir, "weekly", str(winner[f"{channel}_model_key"])
        )) for channel in ("long", "short")
    }
    data_hashes = {
        pair: sha256_file(args.cache_dir / f"binance_{pair}_5m.csv") for pair in PAIRS
    }
    lock = {
        "schema": "xgboost-grid-risk-gate-v7-lock-v1", "model_version": MODEL_VERSION,
        "selection_basis": {"period": [START_TS, END_TS], "profit_weight": 0.5,
                            "drawdown_weight": 0.5, "mechanism1_runtime_fallback": False,
                            "xgboost_configurations": 40, "screen_candidates": 1920,
                            "long_gate_candidates_per_model": 576,
                            "short_gate_candidates_per_model": 192,
                            "refined_candidates": 7680, "dual_candidates": 100,
                            "recovery_quantile_rule": "entry_quantile_minus_0.10"},
        "winner": winner, "acceptance": accepted, "deployment_allowed": deployment_allowed,
        "model_path": model_path.as_posix(), "model_sha256": model_hash,
        "feature_schema_sha256": feature_hash,
        "training_data_sha256": data_hashes,
        "feature_panel_sha256": sha256_file(args.output_dir / "dual_target_feature_panel.csv.gz"),
        "grid_sequence_sha256": sha256_file(args.output_dir / "grid_selections.csv"),
        "prediction_sha256": prediction_hashes,
        "serialization_check": serialization,
    }
    write_json(args.output_dir / "locked_configuration.json", lock)
    pd.DataFrame([{"scenario": "Mechanism 1 (comparison only)", **baseline},
                  {"scenario": "XGBoost v7 dual OR gate", **metrics}]).to_csv(
                      args.output_dir / "final_metrics.csv", index=False)
    detailed["weekly"].to_csv(args.output_dir / "final_weekly_results.csv", index=False)
    detailed["equity"].to_csv(args.output_dir / "final_equity_curve.csv.gz", index=False, compression="gzip")
    detailed["trades"].to_csv(args.output_dir / "final_trade_events.csv.gz", index=False, compression="gzip")
    detailed["states"].to_csv(args.output_dir / "final_risk_states.csv.gz", index=False, compression="gzip")
    detailed["events"].to_csv(args.output_dir / "final_risk_events.csv", index=False)
    detailed["intervals"].to_csv(args.output_dir / "final_risk_intervals.csv", index=False)
    detailed["stops"].to_csv(args.output_dir / "final_stop_events.csv", index=False)
    audit.to_csv(args.output_dir / "walk_forward_training_audit.csv", index=False)
    summary = {
        "schema": "xgboost-grid-risk-gate-v7-summary-v1", "model_version": MODEL_VERSION,
        "evidence_status": "full_180d_in_sample_targeted_optimization",
        "verdict": "NEXT_STAGE_JOINT_VALIDATION" if deployment_allowed else "NO-GO",
        "deployment_allowed": deployment_allowed, "mechanism1_runtime_fallback": False,
        "baseline": baseline, "winner_metrics": metrics, "acceptance": accepted,
        "locked_configuration": lock,
        "no_lookahead": {
            "all_labels_mature": bool((audit.last_mature_label_ready_ts <= audit.train_cutoff_ts).all()),
            "calibration_precedes_test": bool((audit.last_calibration_signal_ts < audit.first_test_signal_ts).all()),
        },
        "limitations": [
            "The same 180-day path and both anchor windows are used for selection; this is not fresh OOS evidence.",
            "Funding, OI, taker-buy ratio and macro/FOMC history are unavailable.",
            "If the lock is not deployment allowed, the runtime gate fails closed instead of falling back to Mechanism 1.",
        ],
    }
    write_json(args.output_dir / "summary.json", summary)
    artifact = {
        "schema": "xgboost-grid-risk-gate-v7-artifact-v1",
        "model_version": MODEL_VERSION,
        "evidence_status": summary["evidence_status"],
        "verdict": summary["verdict"],
        "deployment_allowed": deployment_allowed,
        "primary_report": "xgboost_v7_riskoff_entry_exit_plotly.html",
        "sources": {
            "candles": data_hashes,
            "feature_panel": lock["feature_panel_sha256"],
            "grid_sequence": lock["grid_sequence_sha256"],
        },
        "metrics": {
            "mechanism1": baseline,
            "xgboost_v7": metrics,
        },
        "acceptance": accepted,
        "files": [
            "model_screen_40x2x3x8.csv", "single_channel_refined_search.csv",
            "dual_channel_search.csv", "locked_configuration.json", "final_metrics.csv",
            "final_weekly_results.csv", "final_equity_curve.csv.gz",
            "final_risk_states.csv.gz", "final_risk_events.csv", "final_risk_intervals.csv",
            "pressure_tests.csv", "classification_metrics.csv",
            "xgboost_gain_feature_importance.csv", "models/xgboost_grid_risk_gate_v7.joblib",
            "technical_summary.md", "xgboost_v7_riskoff_entry_exit_plotly.html",
        ],
    }
    write_json(args.output_dir / "artifact.json", artifact)
    (args.output_dir / "technical_summary.md").write_text(
        "\n".join([
            "# XGBoost v7 Risk-off 180天研究摘要",
            "",
            f"- 结论：`{summary['verdict']}`；`deployment_allowed={str(deployment_allowed).lower()}`。",
            f"- XGBoost净盈利：{metrics['oos_pnl_fdusd']:+.6f} FDUSD；机制1：{baseline['oos_pnl_fdusd']:+.6f} FDUSD。",
            f"- XGBoost拼接最大回撤：{metrics['stitched_max_drawdown_pct']:.6f}%；机制1：{baseline['stitched_max_drawdown_pct']:.6f}%。",
            f"- 单对/组合停止：{int(metrics['pair_stop_events'])}/{int(metrics['portfolio_stop_events'])}。",
            f"- 长期目标：`{winner['long_target']}`；短期目标：`{winner['short_target']}`；长短Risk-off Jaccard：{float(winner['active_jaccard']):.4%}。",
            "- XGBoost仅控制对应交易对普通BUY；SELL、风控基准恢复BUY和48小时额外库存退出不受模型门驱动。",
            "- 该180天路径参与了模型选择，属于样本内定向优化，不是全新样本外证据。",
            "- 未通过锁定验收时，运行时只会fail-closed暂停普通BUY，不回退机制1。",
            "",
        ]),
        encoding="utf-8",
    )
    return summary


def build_plot(args: argparse.Namespace) -> Path:
    lock = json.loads((args.output_dir / "locked_configuration.json").read_text(encoding="utf-8"))
    states = pd.read_csv(args.output_dir / "final_risk_states.csv.gz")
    events = pd.read_csv(args.output_dir / "final_risk_events.csv")
    intervals = pd.read_csv(args.output_dir / "final_risk_intervals.csv")
    metrics = pd.read_csv(args.output_dir / "final_metrics.csv")
    coverage = pd.DataFrame(anchor_metrics(intervals[intervals.strategy == lock["winner"]["long_target"]]), index=[0])
    coverage.to_csv(args.output_dir / "anchor_window_coverage.csv", index=False)
    strategy_map = {
        str(lock["winner"]["long_target"]): "long_persistent_72h",
        str(lock["winner"]["short_target"]): "short_spike_1h_24h",
    }
    for frame in (states, events, intervals):
        frame["strategy"] = frame.strategy.map(strategy_map)
        frame["strategy_label"] = frame.strategy.map({
            "long_persistent_72h": "长期持续下跌",
            "short_spike_1h_24h": "1h快速插针",
        })
    # The legacy Plotly builder reads its strategy metadata from a module-level
    # mapping.  Keep the v7 aliases local to this call so importing/running this
    # research entry cannot corrupt legacy tests or another report build in the
    # same Python process.
    original_strategies = {key: value.copy() for key, value in dual.STRATEGIES.items()}
    try:
        dual.STRATEGIES.clear()
        dual.STRATEGIES.update({
            "long_persistent_72h": {
                **original_strategies["long_persistent_72h"],
                "label": "长期持续下跌",
            },
            "short_spike_1h_24h": {
                **original_strategies["short_spike_1h_24h"],
                "label": "1h快速插针",
            },
        })
        plot = dual.build_plotly(
            args.cache_dir, args.output_dir, states, events, intervals, metrics,
            pd.DataFrame(), anchor_windows=ANCHOR_WINDOWS,
        )
    finally:
        dual.STRATEGIES.clear()
        dual.STRATEGIES.update(original_strategies)
    page = plot.read_text(encoding="utf-8")
    page = page.replace("XGBoost双风险策略：180天诊断回放", "XGBoost v7 Risk-off直接驱动Grid")
    page = page.replace("120h持续下跌", "长期持续下跌")
    verdict = "NEXT_STAGE_JOINT_VALIDATION" if lock["deployment_allowed"] else "NO-GO"
    banner = (
        f'<div class="note"><b>锁定结论：{verdict}</b>｜'
        f'deployment_allowed={str(bool(lock["deployment_allowed"])).lower()}｜'
        '同一180天路径参与定向选择，并非全新样本外证据。'
        '未授权时Grid普通BUY保持fail-closed，且不回退机制1。</div>'
    )
    page = page.replace("</h1>", f"</h1>{banner}", 1)
    target = args.output_dir / "xgboost_v7_riskoff_entry_exit_plotly.html"
    target.write_text(page, encoding="utf-8")
    artifact_path = args.output_dir / "artifact.json"
    if artifact_path.exists():
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
        artifact["primary_report_sha256"] = sha256_file(target)
        write_json(artifact_path, artifact)
    return target


def main() -> int:
    args = parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)
    candles, quality = load_candles(args.cache_dir)
    quality.to_csv(args.output_dir / "data_quality.csv", index=False)
    selections = pd.read_csv(args.source_dir / "grid_selections.csv")
    selections.to_csv(args.output_dir / "grid_selections.csv", index=False)
    panel_path = args.output_dir / "dual_target_feature_panel.csv.gz"
    if args.resume and panel_path.exists():
        panel = pd.read_csv(panel_path)
    else:
        source = pd.read_csv(args.source_dir / "dual_target_feature_panel.csv.gz")
        panel = relabel_panel(source, candles)
        panel.to_csv(panel_path, index=False, compression="gzip")
    target_quality(panel).to_csv(args.output_dir / "target_quality.csv", index=False)
    xgb_configs = pd.DataFrame(xgb_configurations())
    xgb_configs.to_csv(args.output_dir / "xgboost_40_parameters.csv", index=False)
    baseline_path = args.output_dir / "mechanism1_baseline.json"
    if args.resume and baseline_path.exists():
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    else:
        baseline = baseline_metrics(candles, selections); write_json(baseline_path, baseline)
    if args.stage == "prepare":
        return 0

    screen_path = args.output_dir / "model_screen_40x2x3x8.csv"
    if args.stage in {"screen", "all"}:
        screen = screen_models(args, panel, candles, selections, baseline)
    elif screen_path.exists():
        screen = pd.read_csv(screen_path)
    else:
        raise FileNotFoundError("run --stage screen first")
    if args.stage == "screen":
        return 0

    selected = finalists(screen, args.screen_top)
    selected.to_csv(args.output_dir / "model_finalists.csv", index=False)
    predictions, audit = load_weekly_predictions(args, panel, selections, selected)
    single_path = args.output_dir / "single_channel_refined_search.csv"
    dual_path = args.output_dir / "dual_channel_search.csv"
    if args.stage in {"search", "all"}:
        single = refine_single(args, candles, selections, selected, predictions, baseline)
        ranked = dual_search(args, candles, selections, single, predictions, baseline)
    elif single_path.exists() and dual_path.exists():
        single, ranked = pd.read_csv(single_path), pd.read_csv(dual_path)
    else:
        raise FileNotFoundError("run --stage search first")
    if args.stage == "search":
        return 0

    if args.stage in {"finalize", "all"}:
        summary = finalize(args, panel, candles, selections, baseline, ranked, predictions, audit)
        print(json.dumps({"verdict": summary["verdict"], "metrics": summary["winner_metrics"]},
                         ensure_ascii=False, indent=2), flush=True)
    if args.stage in {"plot", "all"}:
        plot = build_plot(args)
        print(json.dumps({"plotly": str(plot)}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

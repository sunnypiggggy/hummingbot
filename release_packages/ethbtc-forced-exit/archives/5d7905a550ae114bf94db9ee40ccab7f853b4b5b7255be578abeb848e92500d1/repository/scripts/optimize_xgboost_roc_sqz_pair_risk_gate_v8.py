#!/usr/bin/env python3
"""Search pair-independent ROC/SQZMOM XGBoost BUY risk gates over 180 days.

The Grid remains the trading strategy.  Models can only pause ordinary BUY
orders for their own pair; they never request a market sell.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import multiprocessing as mp
import os
import time
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
import optimize_xgboost_grid_risk_gate_v7 as v7
from compare_independent_gate_ml_stops import HOUR, PAIRS, load_candles
from search_fdusd_inventory_exit import aggregate_rows
from tune_xgboost_grid_risk_gate_v1 import candidate_from_row
from tune_xgboost_momentum_stop_v2 import (
    fit_one_group, sha256_file, split_mature_training, write_json,
    xgb_configurations,
)
from validate_grid_live import crash_candles, simulate, slice_window


MODEL_VERSION = "xgboost-roc-sqz-pair-risk-gate-v8"
CONFIGURATION_FAMILY = "XGBoost"
OUTPUT_DIR = Path("results/backtests/xgboost_roc_sqz_pair_risk_gate_v8")
SOURCE_DIR = Path("results/backtests/xgboost_grid_risk_gate_v7")
START_TS = v7.START_TS
END_TS = v7.END_TS
ENTRY_QUANTILES = v7.ENTRY_QUANTILES
ANCHOR_WINDOWS = v7.ANCHOR_WINDOWS
TARGETS = v7.TARGETS
FIVE_MINUTES = 300

# The feature contract is intentionally narrow.  No price level, RSI, ATR,
# volume, pair id, BTC context, or clock feature can enter the model.
ROC_SQZ_FEATURES = (
    "roc_5",
    "roc_20",
    "sqzmom_value",
    "sqzmom_slope",
    "roc_48h_4h",
    "sqzmom_pct_4h",
    "sqzmom_value_4h",
    "sqzmom_slope_4h",
    "sqzmom_improving_4h",
    "roc_to_entry_4h",
    "sqz_to_entry_4h",
    "roc_to_recovery_4h",
    "sqz_to_recovery_4h",
)
FEATURES_BY_TARGET: dict[str, tuple[str, ...]] = {}
FEATURES_BY_PAIR_TARGET: dict[tuple[str, str], tuple[str, ...]] = {}
CONFIGURATION_PROVIDER: Any = None
# Optional research-version hook used when a synthetic pressure path needs
# features that are not produced by the legacy multi-horizon panel builder.
CRASH_PANEL_BUILDER: Any = None
MODEL_ARTIFACT_FILENAME = "xgboost_roc_sqz_pair_risk_gate_v8.joblib"
MODEL_SCHEMA = "xgboost-roc-sqz-pair-risk-gate-v8-model-v1"
LOCK_SCHEMA = "xgboost-roc-sqz-pair-risk-gate-v8-lock-v1"
SUMMARY_SCHEMA = "xgboost-roc-sqz-pair-risk-gate-v8-summary-v1"
PREDICTION_CACHE_SCHEMA = "xgboost-roc-sqz-pair-v8-prediction-cache-v1"
STRATEGY_LABEL = "XGBoost v8 ROC/SQZ pair-independent gate"
PLOT_FILENAME = "xgboost_v8_roc_sqz_pair_riskoff_plotly.html"
PLOT_TITLE = "XGBoost v8：BTC/ETH独立ROC/SQZMOM Risk-off驱动Grid"
FEATURE_NOTE = "特征仅限ROC/SQZMOM"
FEATURE_LIMITATION = "The feature set is intentionally restricted to ROC and SQZMOM derivatives."
LONG_CHANNEL_LABEL = "长期ROC/SQZ风险"
SHORT_CHANNEL_LABEL = "1h快速下跌"
PARAMETERS_FILENAME = "xgboost_40_parameters.csv"
IMPORTANCE_FILENAME = "xgboost_gain_feature_importance.csv"
SCREEN_FILENAME = "model_screen_40x2pairsx3targetsx8.csv"
SINGLE_FILENAME = "single_pair_channel_refined_search.csv"
PAIR_FILENAME = "pair_independent_long_short_search.csv"
PORTFOLIO_FILENAME = "btc_eth_independent_portfolio_search.csv"


def features_for_target(target: str, pair: str | None = None) -> tuple[str, ...]:
    """Return the immutable feature contract for one prediction target."""
    if pair is not None and (pair, target) in FEATURES_BY_PAIR_TARGET:
        return FEATURES_BY_PAIR_TARGET[(pair, target)]
    return FEATURES_BY_TARGET.get(target, ROC_SQZ_FEATURES)


def features_for_model(
    target: str, pair: str, config: Mapping[str, Any] | None = None,
) -> tuple[str, ...]:
    if config is not None and config.get("features"):
        return tuple(str(value) for value in config["features"])
    return features_for_target(target, pair)


def configurations_for(target: str, pair: str) -> list[dict[str, Any]]:
    if CONFIGURATION_PROVIDER is None:
        return xgb_configurations()
    return list(CONFIGURATION_PROVIDER(target, pair))


def all_configurations() -> list[dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for target, pair in product(TARGETS, PAIRS):
        for config in configurations_for(target, pair):
            config_id = str(config["config_id"])
            if config_id in output and output[config_id] != config:
                raise ValueError(f"configuration id is not globally unique: {config_id}")
            output[config_id] = config
    return list(output.values())


def feature_contract() -> dict[str, list[str]]:
    if FEATURES_BY_PAIR_TARGET:
        return {
            f"{pair}|{target}": list(features_for_target(target, pair))
            for pair, target in product(PAIRS, TARGETS)
        }
    return {target: list(features_for_target(target)) for target in TARGETS}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", choices=("prepare", "screen", "search", "finalize", "plot", "all"),
        default="all",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--cache-dir", type=Path, default=Path("data/backtesting_candles"))
    parser.add_argument("--source-dir", type=Path, default=SOURCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--screen-top", type=int, default=10)
    parser.add_argument("--workers", type=int, default=1)
    return parser.parse_args()


def sha256_json(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(raw).hexdigest()


def write_csv_atomic(frame: pd.DataFrame, path: Path, **kwargs: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # A fixed .tmp name is unsafe when a killed/restarted multiprocessing run
    # briefly overlaps its successor on Windows.  A unique sibling preserves
    # atomic replacement without allowing one writer to steal another's file.
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    frame.to_csv(temporary, index=False, **kwargs)
    # Windows indexers and concurrent read-only progress checks can briefly
    # hold the destination.  Preserve atomic replacement while tolerating a
    # bounded transient lock instead of discarding an otherwise valid run.
    for attempt in range(12):
        try:
            os.replace(temporary, path)
            return
        except PermissionError:
            if attempt == 11:
                raise
            time.sleep(0.10 * (attempt + 1))


def pair_model_key(target: str, pair: str, config_id: str) -> str:
    return f"{target}|{pair}|{config_id}"


def prediction_path(output_dir: Path, stage: str, key: str) -> Path:
    return output_dir / "prediction_cache" / stage / f"{key.replace('|', '__')}.csv.gz"


def metadata_path(path: Path) -> Path:
    return path.with_name(path.name + ".metadata.json")


def prediction_metadata(
    args: argparse.Namespace, *, stage: str, target: str, pair: str,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema": PREDICTION_CACHE_SCHEMA,
        "stage": stage,
        "target": target,
        "pair": pair,
        "configuration_sha256": sha256_json(dict(config)),
        "feature_schema_sha256": sha256_json(features_for_model(target, pair, config)),
        "feature_panel_sha256": sha256_file(args.output_dir / "feature_panel.csv.gz"),
        "source_candle_sha256": sha256_file(
            args.cache_dir / f"binance_{pair}_5m.csv"
        ),
        "grid_sequence_sha256": sha256_file(args.output_dir / "grid_selections.csv"),
        "target_definition_version": TARGETS[target]["definition_version"],
        "pipeline_sha256": sha256_json({
            "fit": inspect.getsource(fit_pair_block),
            "thresholds": inspect.getsource(attach_thresholds),
            "fit_one_group": inspect.getsource(fit_one_group),
        }),
    }


def load_prediction(path: Path, expected: Mapping[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not metadata_path(path).exists():
        raise ValueError(f"prediction cache metadata is missing: {path}")
    observed = json.loads(metadata_path(path).read_text(encoding="utf-8"))
    if observed != dict(expected):
        raise ValueError(f"prediction cache hash mismatch: {path}")
    audit_path = path.with_suffix(".audit.csv")
    return pd.read_csv(path), pd.read_csv(audit_path)


def save_prediction(
    path: Path, prediction: pd.DataFrame, audit: pd.DataFrame,
    metadata: Mapping[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_csv_atomic(prediction, path, compression="gzip")
    write_csv_atomic(audit, path.with_suffix(".audit.csv"))
    write_json(metadata_path(path), dict(metadata))


def attach_thresholds(prediction: pd.DataFrame, calibration: pd.DataFrame) -> pd.DataFrame:
    output = prediction.copy()
    needed = sorted({*ENTRY_QUANTILES, *(max(0.50, q - 0.10) for q in ENTRY_QUANTILES)})
    for quantile in needed:
        output[v5.quantile_column(quantile)] = float(
            calibration.probability.quantile(quantile)
        )
    return output


def fit_pair_block(
    panel: pd.DataFrame, target: str, pair: str, config: Mapping[str, Any],
    block: Any, test_start: int | None = None, test_end: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], Any]:
    working = v7.working_target(panel, target)
    working = working[working.pair.eq(pair)].copy()
    mature, core, validation = split_mature_training(working, int(block.train_end))
    left = int(test_start if test_start is not None else block.test_start)
    right = int(test_end if test_end is not None else block.test_end)
    testing = working[(working.signal_ts >= left) & (working.signal_ts < right)].copy()
    features = list(features_for_model(target, pair, config))
    model, fit_audit = fit_one_group(config, features, mature, core, validation)
    predicted = testing[["pair", "signal_ts", "target"]].copy()
    predicted["probability"] = model.predict_proba(
        testing[features]
    )[:, 1]
    calibrated = validation[["pair", "signal_ts", "target"]].copy()
    calibrated["probability"] = model.predict_proba(
        validation[features]
    )[:, 1]
    predicted = attach_thresholds(predicted, calibrated)
    predicted["strategy"] = target
    audit = {
        "target": target,
        "pair": pair,
        "train_cutoff_ts": int(block.train_end),
        "last_mature_label_ready_ts": int(mature.label_ready_ts.max()),
        "last_calibration_signal_ts": int(validation.signal_ts.max()),
        "first_test_signal_ts": int(testing.signal_ts.min()),
        **fit_audit,
    }
    if audit["last_mature_label_ready_ts"] > audit["train_cutoff_ts"]:
        raise AssertionError("pair model used an immature label")
    if audit["last_calibration_signal_ts"] >= audit["first_test_signal_ts"]:
        raise AssertionError("pair calibration overlaps prediction period")
    return predicted, calibrated, audit, model


def fixed_origin_prediction(
    panel: pd.DataFrame, target: str, pair: str, config: Mapping[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    block = SimpleNamespace(train_end=START_TS, test_start=START_TS, test_end=END_TS)
    prediction, _, audit, _ = fit_pair_block(
        panel, target, pair, config, block, START_TS, END_TS
    )
    return prediction, pd.DataFrame([audit])


def weekly_prediction(
    panel: pd.DataFrame, selections: pd.DataFrame, target: str,
    pair: str, config: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    predictions, audits = [], []
    for block in selections.itertuples(index=False):
        prediction, _, audit, _ = fit_pair_block(panel, target, pair, config, block)
        prediction["fold"] = int(block.fold)
        predictions.append(prediction)
        audits.append({"config_id": config["config_id"], "fold": int(block.fold), **audit})
    output = pd.concat(predictions, ignore_index=True)
    if not np.isfinite(output.probability).all() or not output.probability.between(0, 1).all():
        raise AssertionError("pair model probabilities are invalid")
    return output, pd.DataFrame(audits)


def gate_id(channel: str, gate: v5.GateParameters) -> str:
    return v7.gate_id(channel, gate)


def fixed_gate(channel: str, entry: float) -> v5.GateParameters:
    return v7.fixed_gate(channel, entry)


def refinement_gates(channel: str) -> list[v5.GateParameters]:
    """Deterministically cover the full approved state-machine space.

    XGBoost's 40 configurations and all eight entry quantiles are fully
    screened.  The more expensive Grid refinement uses a seed-42 subset of
    the approved Cartesian state space, retaining both endpoint tuples.
    """
    full = v7.sampled_gates(channel)
    count = 128 if channel == "long" else 64
    rng = np.random.default_rng(42 if channel == "long" else 43)
    middle = rng.choice(np.arange(1, len(full) - 1), size=count - 2, replace=False)
    indices = sorted({0, len(full) - 1, *(int(value) for value in middle)})
    result = [full[index] for index in indices]
    if len(result) != count or len({gate_id(channel, gate) for gate in result}) != count:
        raise AssertionError(f"expected {count} deterministic {channel} refinement gates")
    return result


def gate_from_row(row: Mapping[str, Any], prefix: str = "") -> v5.GateParameters:
    return v5.GateParameters(
        float(row[f"{prefix}entry_quantile"]),
        float(row[f"{prefix}recovery_quantile"]),
        int(row[f"{prefix}entry_bars"]),
        int(row[f"{prefix}recovery_bars"]),
        int(row[f"{prefix}minimum_hours"]),
        int(row[f"{prefix}maximum_hours"]),
        int(row[f"{prefix}cooldown_hours"]),
    )


def strategy_name(pair: str, channel: str, target: str) -> str:
    return f"{pair[:3].lower()}_{channel}_{target}"


def build_pair_gate(
    prediction: pd.DataFrame, pair: str, channel: str, target: str,
    gate: v5.GateParameters, start_ts: int = START_TS, end_ts: int = END_TS,
) -> tuple[dict[str, dict[int, bool]], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    name = strategy_name(pair, channel, target)
    records = list(prediction[
        prediction.pair.eq(pair)
        & prediction.signal_ts.between(start_ts, end_ts, inclusive="left")
    ].sort_values("signal_ts").itertuples(index=False))
    timeline = {item: {} for item in PAIRS}
    states, events, intervals = [], [], []
    state, interval_start = v5.GateState(), None
    for index, row in enumerate(records):
        entry = float(getattr(row, v5.quantile_column(gate.entry_quantile)))
        recovery = float(getattr(row, v5.quantile_column(gate.recovery_quantile)))
        state, transition, reason = v5.step_gate(
            float(row.probability), entry, recovery, int(row.signal_ts), state, gate
        )
        right = min(
            int(records[index + 1].signal_ts) if index + 1 < len(records) else end_ts,
            end_ts,
        )
        for timestamp in range(max(start_ts, int(row.signal_ts)), right, FIVE_MINUTES):
            timeline[pair][timestamp] = not state.active
        states.append({
            "strategy": name, "channel": channel, "target": target, "pair": pair,
            "signal_ts": int(row.signal_ts), "probability": float(row.probability),
            "entry_threshold": entry, "recovery_threshold": recovery,
            "risk_off_active": bool(state.active), "buy_enabled": not bool(state.active),
            "transition": transition, "reason": reason,
        })
        if transition == "enter":
            interval_start = int(row.signal_ts)
        elif transition == "recover" and interval_start is not None:
            intervals.append({
                "strategy": name, "channel": channel, "target": target, "pair": pair,
                "start_ts": interval_start, "end_ts": int(row.signal_ts),
                "duration_hours": (int(row.signal_ts) - interval_start) / HOUR,
                "end_reason": reason,
            })
            interval_start = None
        if transition in {"enter", "recover"}:
            events.append({
                "strategy": name, "channel": channel, "target": target, "pair": pair,
                "timestamp": int(row.signal_ts), "event": transition,
                "probability": float(row.probability), "entry_threshold": entry,
                "recovery_threshold": recovery,
                "event_id": f"{MODEL_VERSION}-{name}-{int(row.signal_ts)}-{transition}",
            })
    if interval_start is not None:
        intervals.append({
            "strategy": name, "channel": channel, "target": target, "pair": pair,
            "start_ts": interval_start, "end_ts": end_ts,
            "duration_hours": (end_ts - interval_start) / HOUR,
            "end_reason": "research_period_end",
        })
    return timeline, pd.DataFrame(states), pd.DataFrame(events), pd.DataFrame(intervals)


def combine_pair_gates(
    specifications: Sequence[tuple[pd.DataFrame, str, str, str, v5.GateParameters]],
) -> tuple[dict[str, dict[int, bool]], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    timelines, states, events, intervals = [], [], [], []
    for prediction, pair, channel, target, gate in specifications:
        timeline, state, event, interval = build_pair_gate(
            prediction, pair, channel, target, gate
        )
        timelines.append(timeline)
        states.append(state)
        if not event.empty:
            events.append(event)
        if not interval.empty:
            intervals.append(interval)
    combined = dual.combine_channel_gates(timelines, START_TS, END_TS)
    return (
        combined,
        pd.concat(states, ignore_index=True),
        pd.concat(events, ignore_index=True) if events else pd.DataFrame(),
        pd.concat(intervals, ignore_index=True) if intervals else pd.DataFrame(),
    )


def replay_metrics(
    candles: Mapping[str, pd.DataFrame], selections: pd.DataFrame,
    specifications: Sequence[tuple[pd.DataFrame, str, str, str, v5.GateParameters]],
    taker_fee: float = base.TAKER_FEE, slippage: float = 0.0,
) -> dict[str, Any]:
    timeline, _, _, _ = combine_pair_gates(specifications)
    weekly, pair_rows, curves = [], [], []
    cumulative = 0.0
    for selection in selections.itertuples(index=False):
        result, curve, pairs = simulate(
            slice_window(dict(candles), int(selection.test_start), int(selection.test_end)),
            candidate_from_row(selection), maker_fee=0.0, taker_fee=taker_fee,
            slippage=slippage, order_refresh_seconds=7200,
            technical_buy_gate=timeline, momentum_stop_timeline=None, trade_log=None,
            risk_breakers_enabled=True, cost_floor_enabled=True,
            inventory_exit_policy=base.POLICY, record_curve=True,
        )
        weekly.append({
            "fold": int(selection.fold), **result,
            "pair_stop_events": sum(int(value.get("liquidations", 0)) for value in pairs.values()),
            "pair_stop_hours": 0.0,
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
    pair_frame = pd.DataFrame(pair_rows)
    summary.update({
        "stitched_max_drawdown_pct": float((equity / equity.cummax() - 1).min() * 100),
        "risk_off_pair_hours": float(pair_frame.technical_risk_off_seconds.sum() / HOUR),
        "pair_stop_events": int(pair_frame.liquidations.sum()),
        "portfolio_stop_events": int(sum(bool(row["liquidated"]) for row in weekly)),
        "momentum_stop_exits": int(sum(int(row["momentum_stop_exits"]) for row in weekly)),
    })
    return summary


def pair_anchor_metrics(intervals: pd.DataFrame, pair: str) -> dict[str, Any]:
    group = intervals[intervals.pair.eq(pair)] if not intervals.empty else intervals
    values: dict[str, Any] = {"interval_count": int(len(group))}
    passed = len(group) <= 8
    anchor_seconds = 0.0
    for name, start, end in ANCHOR_WINDOWS:
        if group.empty:
            coverage, timely = 0.0, False
        else:
            overlap = np.maximum(
                0, np.minimum(group.end_ts.to_numpy(), end)
                - np.maximum(group.start_ts.to_numpy(), start),
            )
            coverage = float(overlap.sum() / (end - start))
            timely = bool(((group.start_ts <= start + 12 * HOUR) & (group.end_ts > start)).any())
            anchor_seconds += float(overlap.sum())
        values[f"{name}_coverage"] = coverage
        values[f"{name}_timely"] = timely
        passed = passed and timely and coverage >= 0.70
    total = float((group.end_ts - group.start_ts).sum()) if not group.empty else 0.0
    outside_share = max(total - anchor_seconds, 0.0) / float(END_TS - START_TS)
    values["outside_anchor_share"] = outside_share
    values["anchor_pass"] = bool(passed and outside_share <= 0.20)
    return values


def pair_channel_overlap(long_states: pd.DataFrame, short_states: pd.DataFrame, pair: str) -> float:
    left = long_states[long_states.pair.eq(pair)][["signal_ts", "risk_off_active"]].rename(
        columns={"risk_off_active": "long"}
    )
    right = short_states[short_states.pair.eq(pair)][["signal_ts", "risk_off_active"]].rename(
        columns={"risk_off_active": "short"}
    )
    merged = left.merge(right, on="signal_ts", how="inner")
    union = merged.long | merged.short
    return float((merged.long & merged.short).sum() / union.sum()) if union.any() else 0.0


def score_frame(frame: pd.DataFrame, group_columns: Sequence[str]) -> pd.DataFrame:
    parts = []
    for _, group in frame.groupby(list(group_columns), dropna=False):
        item = group.copy()
        item["profit_percentile"] = item.oos_pnl_fdusd.rank(method="average", pct=True)
        item["drawdown_percentile"] = item.stitched_max_drawdown_pct.rank(method="average", pct=True)
        item["objective_score"] = 0.5 * item.profit_percentile + 0.5 * item.drawdown_percentile
        item = item.sort_values(
            ["eligible", "objective_score", "portfolio_stop_events", "pair_stop_events",
             "risk_off_pair_hours", "oos_pnl_fdusd", "stitched_max_drawdown_pct"],
            ascending=[False, False, True, True, True, False, False],
        ).reset_index(drop=True)
        item["rank"] = np.arange(1, len(item) + 1)
        parts.append(item)
    return pd.concat(parts, ignore_index=True)


def screen_models(
    args: argparse.Namespace, panel: pd.DataFrame, candles: Mapping[str, pd.DataFrame],
    selections: pd.DataFrame, baseline: Mapping[str, Any],
) -> pd.DataFrame:
    path = args.output_dir / SCREEN_FILENAME
    existing = pd.read_csv(path) if args.resume and path.exists() else pd.DataFrame()
    rows = existing.to_dict("records") if not existing.empty else []
    done = set(existing.candidate_id) if not existing.empty else set()
    jobs = [
        (target, pair, config)
        for target, pair in product(TARGETS, PAIRS)
        for config in configurations_for(target, pair)
    ]
    if args.workers > 1:
        with mp.get_context("spawn").Pool(
            processes=args.workers,
            initializer=initialize_screen_worker,
            initargs=(args, panel, candles, selections, baseline, done),
        ) as pool:
            for key, batch in pool.imap_unordered(screen_model_worker, jobs):
                rows.extend(batch)
                write_csv_atomic(pd.DataFrame(rows), path)
                print(f"SCREEN {key}", flush=True)
        ranked = score_frame(pd.DataFrame(rows), ("pair", "channel"))
        write_csv_atomic(ranked, path)
        return ranked
    for target, pair, config in jobs:
        key = pair_model_key(target, pair, config["config_id"])
        cache = prediction_path(args.output_dir, "screen", key)
        metadata = prediction_metadata(
            args, stage="screen", target=target, pair=pair, config=config
        )
        cache_valid = False
        if args.resume and cache.exists():
            try:
                prediction, _ = load_prediction(cache, metadata)
                cache_valid = True
            except ValueError:
                cache_valid = False
        if not cache_valid:
            prediction, audit = fixed_origin_prediction(panel, target, pair, config)
            save_prediction(cache, prediction, audit, metadata)
        channel = TARGETS[target]["channel"]
        for entry in ENTRY_QUANTILES:
            gate = fixed_gate(channel, float(entry))
            candidate_id = f"{key}|{gate_id(channel, gate)}"
            if candidate_id in done:
                continue
            metrics = replay_metrics(
                candles, selections, [(prediction, pair, channel, target, gate)]
            )
            rows.append({
                "candidate_id": candidate_id, "model_key": key, "pair": pair,
                "target": target, "channel": channel, "config_id": config["config_id"],
                **asdict(gate), **metrics,
                "eligible": bool(
                    metrics["oos_pnl_fdusd"] > baseline["oos_pnl_fdusd"]
                    and metrics["stitched_max_drawdown_pct"] >= baseline["stitched_max_drawdown_pct"]
                ),
            })
            write_csv_atomic(pd.DataFrame(rows), path)
        print(f"SCREEN {key}", flush=True)
    ranked = score_frame(pd.DataFrame(rows), ("pair", "channel"))
    write_csv_atomic(ranked, path)
    return ranked


_SCREEN_ARGS: argparse.Namespace | None = None
_SCREEN_PANEL: pd.DataFrame | None = None
_SCREEN_CANDLES: Mapping[str, pd.DataFrame] | None = None
_SCREEN_SELECTIONS: pd.DataFrame | None = None
_SCREEN_BASELINE: Mapping[str, Any] | None = None
_SCREEN_DONE: set[str] = set()


def initialize_screen_worker(
    args: argparse.Namespace, panel: pd.DataFrame, candles: Mapping[str, pd.DataFrame],
    selections: pd.DataFrame, baseline: Mapping[str, Any], done: set[str],
) -> None:
    global _SCREEN_ARGS, _SCREEN_PANEL, _SCREEN_CANDLES, _SCREEN_SELECTIONS, _SCREEN_BASELINE, _SCREEN_DONE
    _SCREEN_ARGS, _SCREEN_PANEL = args, panel
    _SCREEN_CANDLES, _SCREEN_SELECTIONS, _SCREEN_BASELINE = candles, selections, baseline
    _SCREEN_DONE = done


def screen_model_worker(job: tuple[str, str, Mapping[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    if any(value is None for value in (
        _SCREEN_ARGS, _SCREEN_PANEL, _SCREEN_CANDLES, _SCREEN_SELECTIONS, _SCREEN_BASELINE
    )):
        raise RuntimeError("screen worker was not initialized")
    target, pair, config = job
    key = pair_model_key(target, pair, str(config["config_id"]))
    cache = prediction_path(_SCREEN_ARGS.output_dir, "screen", key)  # type: ignore[union-attr]
    metadata = prediction_metadata(
        _SCREEN_ARGS, stage="screen", target=target, pair=pair, config=config  # type: ignore[arg-type]
    )
    cache_valid = False
    if _SCREEN_ARGS.resume and cache.exists():  # type: ignore[union-attr]
        try:
            prediction, _ = load_prediction(cache, metadata)
            cache_valid = True
        except ValueError:
            cache_valid = False
    if not cache_valid:
        prediction, audit = fixed_origin_prediction(_SCREEN_PANEL, target, pair, config)  # type: ignore[arg-type]
        save_prediction(cache, prediction, audit, metadata)
    channel = TARGETS[target]["channel"]
    batch = []
    for entry in ENTRY_QUANTILES:
        gate = fixed_gate(channel, float(entry))
        candidate_id = f"{key}|{gate_id(channel, gate)}"
        if candidate_id in _SCREEN_DONE:
            continue
        metrics = replay_metrics(
            _SCREEN_CANDLES, _SCREEN_SELECTIONS,  # type: ignore[arg-type]
            [(prediction, pair, channel, target, gate)],
        )
        batch.append({
            "candidate_id": candidate_id, "model_key": key, "pair": pair,
            "target": target, "channel": channel, "config_id": config["config_id"],
            **asdict(gate), **metrics,
            "eligible": bool(
                metrics["oos_pnl_fdusd"] > _SCREEN_BASELINE["oos_pnl_fdusd"]  # type: ignore[index]
                and metrics["stitched_max_drawdown_pct"] >= _SCREEN_BASELINE["stitched_max_drawdown_pct"]  # type: ignore[index]
            ),
        })
    return key, batch


def finalists(screen: pd.DataFrame, top: int) -> pd.DataFrame:
    best = screen.sort_values("rank").drop_duplicates("model_key", keep="first")
    parts = []
    for pair, channel in product(PAIRS, ("long", "short")):
        parts.append(best[(best.pair == pair) & (best.channel == channel)].nsmallest(top, "rank"))
    return pd.concat(parts, ignore_index=True)


_WEEKLY_ARGS: argparse.Namespace | None = None
_WEEKLY_PANEL: pd.DataFrame | None = None
_WEEKLY_SELECTIONS: pd.DataFrame | None = None


def initialize_weekly_worker(
    args: argparse.Namespace, panel: pd.DataFrame, selections: pd.DataFrame,
) -> None:
    global _WEEKLY_ARGS, _WEEKLY_PANEL, _WEEKLY_SELECTIONS
    _WEEKLY_ARGS, _WEEKLY_PANEL, _WEEKLY_SELECTIONS = args, panel, selections


def weekly_prediction_worker(
    job: tuple[str, str, str, Mapping[str, Any]],
) -> tuple[str, pd.DataFrame, pd.DataFrame]:
    if _WEEKLY_ARGS is None or _WEEKLY_PANEL is None or _WEEKLY_SELECTIONS is None:
        raise RuntimeError("weekly prediction worker was not initialized")
    key, target, pair, config = job
    cache = prediction_path(_WEEKLY_ARGS.output_dir, "weekly", key)
    metadata = prediction_metadata(
        _WEEKLY_ARGS, stage="weekly", target=target, pair=pair, config=config
    )
    cache_rejected = False
    if _WEEKLY_ARGS.resume and cache.exists():
        try:
            prediction, audit = load_prediction(cache, metadata)
        except ValueError:
            cache_rejected = True
    if not (_WEEKLY_ARGS.resume and cache.exists() and not cache_rejected):
        prediction, audit = weekly_prediction(
            _WEEKLY_PANEL, _WEEKLY_SELECTIONS, target, pair, config
        )
        save_prediction(cache, prediction, audit, metadata)
    if cache_rejected:
        audit = audit.copy()
        audit["cache_hash_mismatch_retrained"] = True
    return key, prediction, audit


def load_weekly_predictions(
    args: argparse.Namespace, panel: pd.DataFrame, selections: pd.DataFrame,
    selected: pd.DataFrame,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    configs = {item["config_id"]: item for item in all_configurations()}
    predictions, audits = {}, []
    jobs = []
    for row in selected.itertuples(index=False):
        key = str(row.model_key)
        target, pair, config_id = key.split("|")
        jobs.append((key, target, pair, configs[config_id]))
    workers = max(1, int(args.workers))
    if workers == 1:
        initialize_weekly_worker(args, panel, selections)
        iterator = map(weekly_prediction_worker, jobs)
        pool = None
    else:
        pool = mp.get_context("spawn").Pool(
            processes=workers, initializer=initialize_weekly_worker,
            initargs=(args, panel, selections), maxtasksperchild=4,
        )
        iterator = pool.imap_unordered(weekly_prediction_worker, jobs, chunksize=1)
    try:
        for key, prediction, audit in iterator:
            predictions[key] = prediction
            audits.append(audit)
            print(f"WEEKLY {key}", flush=True)
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    return predictions, pd.concat(audits, ignore_index=True)


_REFINE_CANDLES: Mapping[str, pd.DataFrame] | None = None
_REFINE_SELECTIONS: pd.DataFrame | None = None
_REFINE_PREDICTIONS: Mapping[str, pd.DataFrame] | None = None
_REFINE_BASELINE: Mapping[str, Any] | None = None
_REFINE_DONE: set[str] = set()


def initialize_refine_worker(
    candles: Mapping[str, pd.DataFrame], selections: pd.DataFrame,
    predictions: Mapping[str, pd.DataFrame], baseline: Mapping[str, Any],
    done: set[str],
) -> None:
    global _REFINE_CANDLES, _REFINE_SELECTIONS, _REFINE_PREDICTIONS, _REFINE_BASELINE, _REFINE_DONE
    _REFINE_CANDLES = candles
    _REFINE_SELECTIONS = selections
    _REFINE_PREDICTIONS = predictions
    _REFINE_BASELINE = baseline
    _REFINE_DONE = done


def refine_model_worker(model: Mapping[str, Any]) -> tuple[str, list[dict[str, Any]], int]:
    if any(value is None for value in (
        _REFINE_CANDLES, _REFINE_SELECTIONS, _REFINE_PREDICTIONS, _REFINE_BASELINE
    )):
        raise RuntimeError("refine worker was not initialized")
    key, pair, target, channel = (
        str(model["model_key"]), str(model["pair"]),
        str(model["target"]), str(model["channel"]),
    )
    prediction = _REFINE_PREDICTIONS[key]  # type: ignore[index]
    cache: dict[tuple[tuple[int, int], ...], tuple[dict[str, Any], dict[str, Any]]] = {}
    output = []
    for gate in refinement_gates(channel):
        candidate_id = f"{key}|{gate_id(channel, gate)}"
        if candidate_id in _REFINE_DONE:
            continue
        _, _, _, intervals = build_pair_gate(prediction, pair, channel, target, gate)
        signature = tuple(
            (int(row.start_ts), int(row.end_ts))
            for row in intervals.sort_values(["start_ts", "end_ts"]).itertuples(index=False)
        ) if not intervals.empty else ()
        if signature not in cache:
            metrics = replay_metrics(
                _REFINE_CANDLES, _REFINE_SELECTIONS,  # type: ignore[arg-type]
                [(prediction, pair, channel, target, gate)],
            )
            extra = pair_anchor_metrics(intervals, pair) if channel == "long" else {}
            cache[signature] = (metrics, extra)
        else:
            metrics, extra = cache[signature]
        output.append({
            "candidate_id": candidate_id, "model_key": key, "pair": pair,
            "target": target, "channel": channel, "config_id": str(model["config_id"]),
            **asdict(gate), **dict(metrics), **dict(extra),
            "eligible": bool(
                metrics["oos_pnl_fdusd"] > _REFINE_BASELINE["oos_pnl_fdusd"]  # type: ignore[index]
                and metrics["stitched_max_drawdown_pct"]
                >= _REFINE_BASELINE["stitched_max_drawdown_pct"]  # type: ignore[index]
                and (channel != "long" or extra.get("anchor_pass", False))
            ),
        })
    return key, output, len(cache)


def refine_single(
    args: argparse.Namespace, candles: Mapping[str, pd.DataFrame], selections: pd.DataFrame,
    selected: pd.DataFrame, predictions: Mapping[str, pd.DataFrame],
    baseline: Mapping[str, Any],
) -> pd.DataFrame:
    path = args.output_dir / SINGLE_FILENAME
    existing = pd.read_csv(path) if args.resume and path.exists() else pd.DataFrame()
    rows = existing.to_dict("records") if not existing.empty else []
    done = set(existing.candidate_id) if not existing.empty else set()
    models = []
    for model in selected.to_dict("records"):
        channel, key = str(model["channel"]), str(model["model_key"])
        if any(f"{key}|{gate_id(channel, gate)}" not in done for gate in refinement_gates(channel)):
            models.append(model)
    workers = max(1, int(args.workers))
    if workers == 1:
        initialize_refine_worker(candles, selections, predictions, baseline, done)
        iterator: Iterable[tuple[str, list[dict[str, Any]], int]] = map(refine_model_worker, models)
        pool = None
    else:
        context = mp.get_context("spawn")
        pool = context.Pool(
            processes=workers, initializer=initialize_refine_worker,
            initargs=(candles, selections, predictions, baseline, done),
            maxtasksperchild=4,
        )
        iterator = pool.imap_unordered(refine_model_worker, models, chunksize=1)
    try:
        for index, (key, model_rows, unique) in enumerate(iterator, 1):
            rows.extend(model_rows)
            write_csv_atomic(pd.DataFrame(rows), path)
            print(
                f"REFINE-MODEL {index:02d}/{len(models):02d} {key} "
                f"rows={len(model_rows)} unique_timelines={unique} total_rows={len(rows)}",
                flush=True,
            )
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    ranked = score_frame(pd.DataFrame(rows), ("pair", "channel"))
    write_csv_atomic(ranked, path)
    return ranked


def pair_dual_search(
    args: argparse.Namespace, candles: Mapping[str, pd.DataFrame], selections: pd.DataFrame,
    single: pd.DataFrame, predictions: Mapping[str, pd.DataFrame],
    baseline: Mapping[str, Any],
) -> pd.DataFrame:
    path = args.output_dir / PAIR_FILENAME
    existing = pd.read_csv(path) if args.resume and path.exists() else pd.DataFrame()
    rows = existing.to_dict("records") if not existing.empty else []
    done = set(existing.candidate_id) if not existing.empty else set()
    for pair in PAIRS:
        long_top = single[(single.pair == pair) & (single.channel == "long")].nsmallest(10, "rank")
        short_top = single[(single.pair == pair) & (single.channel == "short")].nsmallest(10, "rank")
        for long_row, short_row in product(long_top.to_dict("records"), short_top.to_dict("records")):
            candidate_id = f"{pair}|{long_row['candidate_id']}||{short_row['candidate_id']}"
            if candidate_id in done:
                continue
            long_gate, short_gate = gate_from_row(long_row), gate_from_row(short_row)
            long_prediction = predictions[str(long_row["model_key"])]
            short_prediction = predictions[str(short_row["model_key"])]
            specs = [
                (long_prediction, pair, "long", str(long_row["target"]), long_gate),
                (short_prediction, pair, "short", str(short_row["target"]), short_gate),
            ]
            metrics = replay_metrics(candles, selections, specs)
            _, long_states, _, long_intervals = build_pair_gate(
                long_prediction, pair, "long", str(long_row["target"]), long_gate
            )
            _, short_states, _, _ = build_pair_gate(
                short_prediction, pair, "short", str(short_row["target"]), short_gate
            )
            anchors = pair_anchor_metrics(long_intervals, pair)
            overlap = pair_channel_overlap(long_states, short_states, pair)
            rows.append({
                "candidate_id": candidate_id, "pair": pair,
                "long_candidate_id": long_row["candidate_id"],
                "short_candidate_id": short_row["candidate_id"],
                "long_model_key": long_row["model_key"],
                "short_model_key": short_row["model_key"],
                "long_target": long_row["target"], "short_target": short_row["target"],
                **{f"long_{key}": value for key, value in asdict(long_gate).items()},
                **{f"short_{key}": value for key, value in asdict(short_gate).items()},
                **metrics, **anchors, "active_jaccard": overlap,
                "eligible": bool(
                    metrics["oos_pnl_fdusd"] > baseline["oos_pnl_fdusd"]
                    and metrics["stitched_max_drawdown_pct"] >= baseline["stitched_max_drawdown_pct"]
                    and anchors["anchor_pass"] and overlap <= 0.15
                ),
            })
            write_csv_atomic(pd.DataFrame(rows), path)
            print(f"PAIR-DUAL {pair} {len(rows):03d}/200", flush=True)
    ranked = score_frame(pd.DataFrame(rows), ("pair",))
    write_csv_atomic(ranked, path)
    return ranked


def pair_specifications(
    row: Mapping[str, Any], predictions: Mapping[str, pd.DataFrame]
) -> list[tuple[pd.DataFrame, str, str, str, v5.GateParameters]]:
    pair = str(row["pair"])
    return [
        (predictions[str(row["long_model_key"])], pair, "long", str(row["long_target"]),
         gate_from_row(row, "long_")),
        (predictions[str(row["short_model_key"])], pair, "short", str(row["short_target"]),
         gate_from_row(row, "short_")),
    ]


def portfolio_search(
    args: argparse.Namespace, candles: Mapping[str, pd.DataFrame], selections: pd.DataFrame,
    pair_ranked: pd.DataFrame, predictions: Mapping[str, pd.DataFrame],
    baseline: Mapping[str, Any],
) -> pd.DataFrame:
    path = args.output_dir / PORTFOLIO_FILENAME
    existing = pd.read_csv(path) if args.resume and path.exists() else pd.DataFrame()
    rows = existing.to_dict("records") if not existing.empty else []
    done = set(existing.candidate_id) if not existing.empty else set()
    btc = pair_ranked[pair_ranked.pair.eq("BTC-FDUSD")].nsmallest(10, "rank")
    eth = pair_ranked[pair_ranked.pair.eq("ETH-FDUSD")].nsmallest(10, "rank")
    for btc_row, eth_row in product(btc.to_dict("records"), eth.to_dict("records")):
        candidate_id = f"portfolio|{btc_row['candidate_id']}|||{eth_row['candidate_id']}"
        if candidate_id in done:
            continue
        specs = pair_specifications(btc_row, predictions) + pair_specifications(eth_row, predictions)
        metrics = replay_metrics(candles, selections, specs)
        rows.append({
            "candidate_id": candidate_id,
            "BTC_pair_candidate_id": btc_row["candidate_id"],
            "ETH_pair_candidate_id": eth_row["candidate_id"],
            **metrics,
            "BTC_anchor_pass": bool(btc_row["anchor_pass"]),
            "ETH_anchor_pass": bool(eth_row["anchor_pass"]),
            "BTC_active_jaccard": float(btc_row["active_jaccard"]),
            "ETH_active_jaccard": float(eth_row["active_jaccard"]),
            "eligible": bool(
                metrics["oos_pnl_fdusd"] > baseline["oos_pnl_fdusd"]
                and metrics["stitched_max_drawdown_pct"] >= baseline["stitched_max_drawdown_pct"]
                and bool(btc_row["anchor_pass"]) and bool(eth_row["anchor_pass"])
                and float(btc_row["active_jaccard"]) <= 0.15
                and float(eth_row["active_jaccard"]) <= 0.15
            ),
        })
        write_csv_atomic(pd.DataFrame(rows), path)
        print(f"PORTFOLIO {len(rows):03d}/100", flush=True)
    ranked = pd.DataFrame(rows)
    ranked["profit_percentile"] = ranked.oos_pnl_fdusd.rank(method="average", pct=True)
    ranked["drawdown_percentile"] = ranked.stitched_max_drawdown_pct.rank(method="average", pct=True)
    ranked["objective_score"] = 0.5 * ranked.profit_percentile + 0.5 * ranked.drawdown_percentile
    ranked = ranked.sort_values(
        ["eligible", "objective_score", "portfolio_stop_events", "pair_stop_events",
         "risk_off_pair_hours", "oos_pnl_fdusd", "stitched_max_drawdown_pct"],
        ascending=[False, False, True, True, True, False, False],
    ).reset_index(drop=True)
    ranked["rank"] = np.arange(1, len(ranked) + 1)
    write_csv_atomic(ranked, path)
    return ranked


def baseline_metrics(candles: Mapping[str, pd.DataFrame], selections: pd.DataFrame) -> dict[str, Any]:
    return v7.baseline_metrics(candles, selections)


def selected_pair_rows(
    portfolio_winner: Mapping[str, Any], pair_ranked: pd.DataFrame,
) -> dict[str, dict[str, Any]]:
    output = {}
    for pair in PAIRS:
        candidate_id = str(portfolio_winner[f"{pair[:3]}_pair_candidate_id"])
        match = pair_ranked[pair_ranked.candidate_id.eq(candidate_id)]
        if len(match) != 1:
            raise ValueError(f"locked {pair} candidate is missing or duplicated")
        output[pair] = match.iloc[0].to_dict()
    return output


def locked_specifications(
    portfolio_winner: Mapping[str, Any], pair_ranked: pd.DataFrame,
    predictions: Mapping[str, pd.DataFrame],
) -> list[tuple[pd.DataFrame, str, str, str, v5.GateParameters]]:
    rows = selected_pair_rows(portfolio_winner, pair_ranked)
    return [spec for pair in PAIRS for spec in pair_specifications(rows[pair], predictions)]


def detailed_replay(
    candles: Mapping[str, pd.DataFrame], selections: pd.DataFrame,
    specifications: Sequence[tuple[pd.DataFrame, str, str, str, v5.GateParameters]],
    scenario: str,
) -> dict[str, Any]:
    timeline, states, events, intervals = combine_pair_gates(specifications)
    weekly, pair_rows, curves, trades, stops = [], [], [], [], []
    cumulative = 0.0
    for selection in selections.itertuples(index=False):
        result, curve, pairs, trade_frame, stop = base.simulate_fold(
            candles, selection, timeline, record_details=True
        )
        weekly.append({"scenario": scenario, **selection._asdict(), **result, **stop})
        pair_rows.extend({
            "scenario": scenario, "period": dual.PERIOD, "fold": int(selection.fold),
            "pair": pair, **value,
        } for pair, value in pairs.items())
        if not curve.empty:
            curve = curve.copy()
            curve["scenario"] = scenario
            curve["period"] = dual.PERIOD
            curve["fold"] = int(selection.fold)
            curve["cumulative_oos_pnl"] = cumulative + curve.equity - base.INITIAL_EQUITY
            curves.append(curve)
        if not trade_frame.empty:
            trade_frame = trade_frame.copy()
            trade_frame["scenario"] = scenario
            trade_frame["period"] = dual.PERIOD
            trade_frame["fold"] = int(selection.fold)
            trades.append(trade_frame)
            for event in trade_frame.to_dict("records"):
                if event.get("reason") == "pair_breaker_flatten":
                    stops.append({
                        "scenario": scenario, "fold": int(selection.fold),
                        "scope": event["pair"], "kind": "pair_stop",
                        "start_ts": int(event["timestamp"]),
                        "end_ts": int(selection.test_end),
                    })
        if result["liquidated"] and not curve.empty:
            stops.append({
                "scenario": scenario, "fold": int(selection.fold),
                "scope": "PORTFOLIO", "kind": "portfolio_stop",
                "start_ts": int(curve.timestamp.iloc[-1]),
                "end_ts": int(selection.test_end),
            })
        cumulative += float(result["net_pnl_quote"])
    weekly_frame, pair_frame = pd.DataFrame(weekly), pd.DataFrame(pair_rows)
    summary = aggregate_rows(weekly, pair_rows)
    equity_frame = pd.concat(curves, ignore_index=True)
    stitched = base.INITIAL_EQUITY + equity_frame.cumulative_oos_pnl.astype(float)
    summary.update({
        "risk_off_pair_hours": float(pair_frame.technical_risk_off_seconds.sum() / HOUR),
        "momentum_stop_exits": int(weekly_frame.momentum_stop_exits.sum()),
        "pair_stop_events": int(pair_frame.liquidations.sum()),
        "portfolio_stop_events": int(weekly_frame.liquidated.astype(bool).sum()),
        "stitched_max_drawdown_pct": float((stitched / stitched.cummax() - 1).min() * 100),
    })
    return {
        "summary": summary, "weekly": weekly_frame, "pairs": pair_frame,
        "equity": equity_frame,
        "trades": pd.concat(trades, ignore_index=True) if trades else pd.DataFrame(),
        "states": states, "events": events, "intervals": intervals,
        "stops": pd.DataFrame(stops),
    }


def train_final_models(
    panel: pd.DataFrame, pair_rows: Mapping[str, Mapping[str, Any]],
    output_dir: Path,
) -> tuple[Path, dict[str, Any]]:
    configs = {item["config_id"]: item for item in all_configurations()}
    bundle: dict[str, Any] = {
        "schema": MODEL_SCHEMA,
        "model_version": MODEL_VERSION,
        "feature_contract": feature_contract(),
        "pairs": {},
    }
    for pair in PAIRS:
        bundle["pairs"][pair] = {"channels": {}}
        row = pair_rows[pair]
        for channel in ("long", "short"):
            target, model_pair, config_id = str(row[f"{channel}_model_key"]).split("|")
            if model_pair != pair:
                raise AssertionError("pair-independent lock references another pair")
            working = v7.working_target(panel, target)
            working = working[working.pair.eq(pair)].copy()
            mature, core, validation = split_mature_training(working, END_TS)
            features = list(features_for_model(target, pair, configs[config_id]))
            model, fit_audit = fit_one_group(
                configs[config_id], features, mature, core, validation
            )
            probability = model.predict_proba(validation[features])[:, 1]
            gate = gate_from_row(row, f"{channel}_")
            bundle["pairs"][pair]["channels"][channel] = {
                "target": target, "config_id": config_id,
                "features": features, "model": model,
                "thresholds": {
                    "entry": float(pd.Series(probability).quantile(gate.entry_quantile)),
                    "recovery": float(pd.Series(probability).quantile(gate.recovery_quantile)),
                },
                "gate": asdict(gate), "fit_audit": fit_audit,
            }
    path = output_dir / "models" / MODEL_ARTIFACT_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)
    return path, bundle


def bundle_probability(
    bundle: Mapping[str, Any], pair: str, channel: str, rows: pd.DataFrame,
) -> np.ndarray:
    specification = bundle["pairs"][pair]["channels"][channel]
    return specification["model"].predict_proba(
        rows[list(specification["features"])]
    )[:, 1]


def write_model_diagnostics(
    output_dir: Path, bundle: Mapping[str, Any], panel: pd.DataFrame,
    predictions: Mapping[str, pd.DataFrame], pair_rows: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    classification, importance = [], []
    for pair in PAIRS:
        row = pair_rows[pair]
        for channel in ("long", "short"):
            key = str(row[f"{channel}_model_key"])
            predicted = predictions[key].dropna(subset=["target", "probability"])
            target = predicted.target.astype(int)
            classification.append({
                "pair": pair, "channel": channel, "model_key": key,
                "rows": len(predicted), "positive_rate": float(target.mean()),
                "roc_auc": float(roc_auc_score(target, predicted.probability))
                if target.nunique() > 1 else float("nan"),
                "average_precision": float(average_precision_score(target, predicted.probability))
                if target.nunique() > 1 else float("nan"),
            })
            model = bundle["pairs"][pair]["channels"][channel]["model"]
            features = bundle["pairs"][pair]["channels"][channel]["features"]
            gains = np.asarray(model.feature_importances_, dtype=float)
            if float(gains.sum()) > 0:
                gains = gains / float(gains.sum())
            importance.extend({
                "pair": pair, "channel": channel, "feature": feature, "gain": float(gain),
            } for feature, gain in zip(features, gains))
    pd.DataFrame(classification).to_csv(
        output_dir / "classification_metrics.csv", index=False
    )
    pd.DataFrame(importance).sort_values(
        ["pair", "channel", "gain"], ascending=[True, True, False]
    ).to_csv(output_dir / IMPORTANCE_FILENAME, index=False)

    loaded = joblib.load(output_dir / "models" / MODEL_ARTIFACT_FILENAME)
    maximum_error, row_count = 0.0, 0
    for pair in PAIRS:
        sample = panel[(panel.pair == pair) & (panel.signal_ts < END_TS)].tail(32)
        row_count += len(sample)
        for channel in ("long", "short"):
            before = bundle_probability(bundle, pair, channel, sample)
            after = bundle_probability(loaded, pair, channel, sample)
            maximum_error = max(maximum_error, float(np.max(np.abs(before - after))))
    result = {
        "rows": row_count, "maximum_probability_absolute_error": maximum_error,
        "passed": bool(maximum_error <= 1e-12),
    }
    write_json(output_dir / "model_serialization_check.json", result)
    if not result["passed"]:
        raise AssertionError("serialized pair model probabilities changed")
    return result


def finalize(
    args: argparse.Namespace, panel: pd.DataFrame, candles: Mapping[str, pd.DataFrame],
    selections: pd.DataFrame, baseline: Mapping[str, Any], pair_ranked: pd.DataFrame,
    portfolio: pd.DataFrame, predictions: Mapping[str, pd.DataFrame], audit: pd.DataFrame,
) -> dict[str, Any]:
    winner = portfolio.iloc[0].to_dict()
    pair_rows = selected_pair_rows(winner, pair_ranked)
    specifications = locked_specifications(winner, pair_ranked, predictions)
    detailed = detailed_replay(
        candles, selections, specifications, STRATEGY_LABEL
    )
    metrics = detailed["summary"]

    crash_path = crash_candles(dict(candles), 0.15)
    if CRASH_PANEL_BUILDER is None:
        crash_panel = v7.relabel_panel(base.build_multi_horizon_panel(crash_path), crash_path)
    else:
        crash_panel = CRASH_PANEL_BUILDER(crash_path)
    configs = {item["config_id"]: item for item in all_configurations()}
    crash_predictions: dict[str, pd.DataFrame] = {}
    for pair in PAIRS:
        row = pair_rows[pair]
        for channel in ("long", "short"):
            key = str(row[f"{channel}_model_key"])
            target, model_pair, config_id = key.split("|")
            prediction, _ = weekly_prediction(
                crash_panel, selections, target, model_pair, configs[config_id]
            )
            crash_predictions[key] = prediction
    crash_specifications = locked_specifications(winner, pair_ranked, crash_predictions)
    pressure_rows = []
    for name, scenario_candles, specs, taker_fee, slippage in (
        ("base", candles, specifications, base.TAKER_FEE, 0.0),
        ("taker_150pct", candles, specifications, base.TAKER_FEE * 1.5, 0.0),
        ("slippage_0_05pct", candles, specifications, base.TAKER_FEE, 0.0005),
        ("slippage_0_10pct", candles, specifications, base.TAKER_FEE, 0.0010),
        ("single_day_15pct_drop", crash_path, crash_specifications, base.TAKER_FEE, 0.0),
    ):
        result = replay_metrics(
            scenario_candles, selections, specs, taker_fee=taker_fee, slippage=slippage
        )
        pressure_rows.append({
            "scenario": name, **result,
            "no_stops": int(result["pair_stop_events"]) == 0
            and int(result["portfolio_stop_events"]) == 0,
        })
    pressure = pd.DataFrame(pressure_rows)
    pressure.to_csv(args.output_dir / "pressure_tests.csv", index=False)

    accepted = {
        "positive_profit": bool(metrics["oos_pnl_fdusd"] >= 0),
        "beats_mechanism1_profit": bool(metrics["oos_pnl_fdusd"] > baseline["oos_pnl_fdusd"]),
        "drawdown_not_worse": bool(
            metrics["stitched_max_drawdown_pct"] >= baseline["stitched_max_drawdown_pct"]
        ),
        "zero_portfolio_stops": bool(int(metrics["portfolio_stop_events"]) == 0),
        "fewer_pair_stops": bool(
            int(metrics["pair_stop_events"]) < int(baseline["pair_stop_events"])
        ),
        "BTC_anchor_pass": bool(pair_rows["BTC-FDUSD"]["anchor_pass"]),
        "ETH_anchor_pass": bool(pair_rows["ETH-FDUSD"]["anchor_pass"]),
        "BTC_channel_overlap_pass": bool(pair_rows["BTC-FDUSD"]["active_jaccard"] <= 0.15),
        "ETH_channel_overlap_pass": bool(pair_rows["ETH-FDUSD"]["active_jaccard"] <= 0.15),
        "all_pressure_scenarios_no_stops": bool(pressure.no_stops.all()),
    }
    deployment_allowed = bool(all(accepted.values()))
    model_path, bundle = train_final_models(panel, pair_rows, args.output_dir)
    serialization = write_model_diagnostics(
        args.output_dir, bundle, panel, predictions, pair_rows
    )
    prediction_hashes = {
        pair: {
            channel: sha256_file(prediction_path(
                args.output_dir, "weekly", str(pair_rows[pair][f"{channel}_model_key"])
            )) for channel in ("long", "short")
        } for pair in PAIRS
    }
    lock = {
        "schema": LOCK_SCHEMA,
        "model_version": MODEL_VERSION,
        "selection_basis": {
            "period": [START_TS, END_TS], "profit_weight": 0.5,
            "drawdown_weight": 0.5, "feature_contract": feature_contract(),
            "pair_parameters_independent": True,
            "mechanism1_runtime_fallback": False,
            "configuration_family": CONFIGURATION_FAMILY,
            "model_configurations": len(all_configurations()),
            # Kept for backward compatibility with the v8 notebook schema.
            "xgboost_configurations": (
                len(all_configurations()) if CONFIGURATION_FAMILY == "XGBoost" else 0
            ),
            "screen_candidates": len(pd.read_csv(args.output_dir / SCREEN_FILENAME)),
            "single_refined_candidates": len(pd.read_csv(args.output_dir / SINGLE_FILENAME)),
            "pair_long_short_candidates": len(pair_ranked),
            "portfolio_candidates": len(portfolio),
        },
        "portfolio_winner": winner, "pair_winners": pair_rows,
        "acceptance": accepted, "deployment_allowed": deployment_allowed,
        "model_path": model_path.as_posix(), "model_sha256": sha256_file(model_path),
        "feature_schema_sha256": sha256_json(feature_contract()),
        "training_data_sha256": {
            pair: sha256_file(args.cache_dir / f"binance_{pair}_5m.csv") for pair in PAIRS
        },
        "feature_panel_sha256": sha256_file(args.output_dir / "feature_panel.csv.gz"),
        "grid_sequence_sha256": sha256_file(args.output_dir / "grid_selections.csv"),
        "prediction_sha256": prediction_hashes,
        "serialization_check": serialization,
    }
    write_json(args.output_dir / "locked_configuration.json", lock)

    pd.DataFrame([
        {"scenario": "Mechanism 1 (comparison only)", **baseline},
        {"scenario": STRATEGY_LABEL, **metrics},
    ]).to_csv(args.output_dir / "final_metrics.csv", index=False)
    detailed["weekly"].to_csv(args.output_dir / "final_weekly_results.csv", index=False)
    detailed["equity"].to_csv(
        args.output_dir / "final_equity_curve.csv.gz", index=False, compression="gzip"
    )
    detailed["trades"].to_csv(
        args.output_dir / "final_trade_events.csv.gz", index=False, compression="gzip"
    )
    detailed["states"].to_csv(
        args.output_dir / "final_risk_states.csv.gz", index=False, compression="gzip"
    )
    detailed["events"].to_csv(args.output_dir / "final_risk_events.csv", index=False)
    detailed["intervals"].to_csv(args.output_dir / "final_risk_intervals.csv", index=False)
    detailed["stops"].to_csv(args.output_dir / "final_stop_events.csv", index=False)
    audit.to_csv(args.output_dir / "walk_forward_training_audit.csv", index=False)
    summary = {
        "schema": SUMMARY_SCHEMA,
        "model_version": MODEL_VERSION,
        "evidence_status": "full_180d_in_sample_targeted_optimization",
        "verdict": "NEXT_STAGE_JOINT_VALIDATION" if deployment_allowed else "NO-GO",
        "deployment_allowed": deployment_allowed,
        "mechanism1_runtime_fallback": False,
        "baseline": baseline, "winner_metrics": metrics,
        "pair_winners": pair_rows, "acceptance": accepted,
        "no_lookahead": {
            "all_labels_mature": bool(
                (audit.last_mature_label_ready_ts <= audit.train_cutoff_ts).all()
            ),
            "calibration_precedes_test": bool(
                (audit.last_calibration_signal_ts < audit.first_test_signal_ts).all()
            ),
        },
        "limitations": [
            "The same 180-day path and both anchor windows are used for selection; this is not fresh OOS evidence.",
            FEATURE_LIMITATION,
            "Funding, OI, taker-buy ratio and macro/FOMC history are unavailable.",
            "A rejected lock fails closed and never falls back to Mechanism 1.",
        ],
    }
    write_json(args.output_dir / "summary.json", summary)
    return summary


def build_plot(args: argparse.Namespace) -> Path:
    lock = json.loads((args.output_dir / "locked_configuration.json").read_text(encoding="utf-8"))
    states = pd.read_csv(args.output_dir / "final_risk_states.csv.gz")
    events = pd.read_csv(args.output_dir / "final_risk_events.csv")
    intervals = pd.read_csv(args.output_dir / "final_risk_intervals.csv")
    metrics = pd.read_csv(args.output_dir / "final_metrics.csv")
    mapping = {"long": "long_persistent_72h", "short": "short_spike_1h_24h"}
    labels = {"long": LONG_CHANNEL_LABEL, "short": SHORT_CHANNEL_LABEL}
    for frame in (states, events, intervals):
        frame["strategy"] = frame.channel.map(mapping)
        frame["strategy_label"] = frame.channel.map(labels)
    original = {key: value.copy() for key, value in dual.STRATEGIES.items()}
    try:
        dual.STRATEGIES.clear()
        dual.STRATEGIES.update({
            "long_persistent_72h": {**original["long_persistent_72h"], "label": labels["long"]},
            "short_spike_1h_24h": {**original["short_spike_1h_24h"], "label": labels["short"]},
        })
        source = dual.build_plotly(
            args.cache_dir, args.output_dir, states, events, intervals, metrics,
            pd.DataFrame(), anchor_windows=ANCHOR_WINDOWS,
        )
    finally:
        dual.STRATEGIES.clear()
        dual.STRATEGIES.update(original)
    page = source.read_text(encoding="utf-8")
    page = page.replace(
        "<title>XGBoost双风险策略180天回测</title>",
        f"<title>{PLOT_TITLE}</title>",
        1,
    )
    verdict = "NEXT_STAGE_JOINT_VALIDATION" if lock["deployment_allowed"] else "NO-GO"
    note = (
        f'<div class="note"><b>锁定结论：{verdict}</b>｜BTC/ETH模型、阈值和状态机参数完全独立｜'
        f'{FEATURE_NOTE}｜模型只暂停普通BUY，不触发即时卖出。</div>'
    )
    page = page.replace("</h1>", f"</h1>{note}", 1)
    page = page.replace(
        "<h1>XGBoost双风险策略：180天诊断回测</h1>",
        f"<h1>{PLOT_TITLE}</h1>",
    )
    page = page.replace("XGBoost双风险策略：180天诊断回放", "XGBoost v8 ROC/SQZ独立Risk-off驱动Grid")
    target = args.output_dir / PLOT_FILENAME
    target.write_text(page, encoding="utf-8")
    return target


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    candles, quality = load_candles(args.cache_dir)
    quality.to_csv(args.output_dir / "data_quality.csv", index=False)
    selections = pd.read_csv(args.source_dir / "grid_selections.csv")
    selections.to_csv(args.output_dir / "grid_selections.csv", index=False)
    panel_path = args.output_dir / "feature_panel.csv.gz"
    if args.resume and panel_path.exists():
        panel = pd.read_csv(panel_path)
    else:
        source = pd.read_csv(args.source_dir / "dual_target_feature_panel.csv.gz")
        requested_features = sorted({feature for features in feature_contract().values() for feature in features})
        missing = sorted(set(requested_features) - set(source.columns))
        if missing:
            raise ValueError(f"missing ROC/SQZ features: {missing}")
        panel = source.copy()
        panel.to_csv(panel_path, index=False, compression="gzip")
    pd.DataFrame([
        {"target": target, "feature": feature}
        for target, features in feature_contract().items() for feature in features
    ]).to_csv(
        args.output_dir / "feature_schema.csv", index=False
    )
    v7.target_quality(panel).to_csv(args.output_dir / "target_quality.csv", index=False)
    pd.DataFrame(all_configurations()).to_csv(
        args.output_dir / PARAMETERS_FILENAME, index=False
    )
    baseline_path = args.output_dir / "mechanism1_baseline.json"
    if args.resume and baseline_path.exists():
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    else:
        baseline = baseline_metrics(candles, selections)
        write_json(baseline_path, baseline)
    if args.stage == "prepare":
        return 0

    screen_path = args.output_dir / SCREEN_FILENAME
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
    single_path = args.output_dir / SINGLE_FILENAME
    pair_path = args.output_dir / PAIR_FILENAME
    portfolio_path = args.output_dir / PORTFOLIO_FILENAME
    if args.stage in {"search", "all"}:
        single = refine_single(args, candles, selections, selected, predictions, baseline)
        pair_ranked = pair_dual_search(args, candles, selections, single, predictions, baseline)
        portfolio = portfolio_search(args, candles, selections, pair_ranked, predictions, baseline)
    elif single_path.exists() and pair_path.exists() and portfolio_path.exists():
        single = pd.read_csv(single_path)
        pair_ranked = pd.read_csv(pair_path)
        portfolio = pd.read_csv(portfolio_path)
    else:
        raise FileNotFoundError("run --stage search first")
    if args.stage == "search":
        return 0

    if args.stage in {"finalize", "all"}:
        summary = finalize(
            args, panel, candles, selections, baseline, pair_ranked,
            portfolio, predictions, audit,
        )
        print(json.dumps({"verdict": summary["verdict"], "metrics": summary["winner_metrics"]},
                         ensure_ascii=False, indent=2), flush=True)
    if args.stage in {"plot", "all"}:
        plot = build_plot(args)
        print(json.dumps({"plotly": str(plot)}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

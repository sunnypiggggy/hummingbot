#!/usr/bin/env python3
"""Optimize two semantically distinct XGBoost Grid BUY-risk channels.

The long channel targets a persistent 120-hour decline.  The short channel
targets a fast 1h/24h downward excursion that materially rebounds by the
24-hour close.  Model and gate parameters are selected only from timestamps
before 2026-06-01, then locked before the already-viewed June/July replay.
This is research code: it never sends orders or authorizes deployment.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_score, recall_score, roc_auc_score

import backtest_xgboost_dual_risk_gate_180d as dual
import backtest_xgboost_long_risk_gate_180d as base
from compare_independent_gate_ml_stops import ALL_FEATURES, FIVE_MINUTES, HOUR, PAIRS, SEPARATE_FEATURES, hourly_bars, load_candles
from search_fdusd_inventory_exit import aggregate_rows
from tune_xgboost_momentum_stop_v2 import fit_one_group, sha256_file, split_mature_training, write_json, xgb_configurations


MODEL_VERSION = "xgboost-grid-dual-risk-gate-v5"
OUTPUT_DIR = Path("results/backtests/xgboost_dual_risk_gate_180d_v5")
SOURCE_PANEL = Path("results/backtests/xgboost_dual_risk_gate_180d_v4/dual_target_feature_panel.csv.gz")
DEV_GATE_START = int(pd.Timestamp("2026-04-01T00:00:00Z").timestamp())
LOCK_TS = int(pd.Timestamp("2026-06-01T00:00:00Z").timestamp())
TARGET_INTERVAL_END = int(pd.Timestamp("2026-06-06T00:00:00Z").timestamp())
MODEL_CONFIG_IDS = ("xgb_00", "xgb_01", "xgb_21", "xgb_30")
MODEL_TUNING_FOLDS = (4, 8, 12, 16)
THRESHOLD_QUANTILES = (0.60, 0.70, 0.80, 0.85, 0.90, 0.925, 0.95, 0.975, 0.985, 0.99)

# Keep stable internal keys for compatibility with the existing replay and
# artifact schema while making the optimized forecast horizons explicit.
dual.STRATEGIES["long_persistent_72h"]["label"] = "120h持续下跌"
dual.STRATEGIES["short_spike_1h_24h"]["label"] = "1h/24h快速插针"


@dataclass(frozen=True)
class GateParameters:
    entry_quantile: float
    recovery_quantile: float
    entry_bars: int
    recovery_bars: int
    minimum_hours: int
    maximum_hours: int | None
    cooldown_hours: int = 0


@dataclass(frozen=True)
class GateState:
    active: bool = False
    since: int | None = None
    entry_count: int = 0
    recovery_count: int = 0
    last_recovery: int | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=Path("data/backtesting_candles"))
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--source-panel", type=Path, default=SOURCE_PANEL)
    parser.add_argument("--source-weekly-results", type=Path, default=base.DEFAULT_WEEKLY_RESULTS)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def quantile_column(value: float) -> str:
    return f"threshold_q{int(round(float(value) * 10000)):04d}"


def relabel_panel(panel: pd.DataFrame, candles: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Create stricter, low-overlap long and spike/rebound targets."""
    item = panel.copy()
    close_parts = []
    for pair, bars in hourly_bars(candles).items():
        close_parts.append(pd.DataFrame({
            "pair": pair,
            "bar_open_ts": bars.index.astype("int64") // 10**9,
            "future_close_return_24h": bars.close.shift(-24).to_numpy(float) / bars.close.to_numpy(float) - 1.0,
            "future_close_return_120h": bars.close.shift(-120).to_numpy(float) / bars.close.to_numpy(float) - 1.0,
            "future_below_current_fraction_120h": (
                pd.concat([bars.close.shift(-offset) for offset in range(1, 121)], axis=1)
                .lt(bars.close, axis=0).sum(axis=1).to_numpy(float) / 120.0
            ),
        }))
    close_frame = pd.concat(close_parts, ignore_index=True)
    generated_columns = [
        "future_close_return_24h", "future_close_return_120h",
        "future_below_current_fraction_120h",
    ]
    item = item.drop(columns=[column for column in generated_columns if column in item.columns])
    item = item.merge(close_frame, on=["pair", "bar_open_ts"], how="left", validate="one_to_one")

    item["long_threshold_120h"] = np.maximum(0.05, 5.0 * item.atr_pct)
    long_valid = item.future_close_return_120h.notna()
    long_target = (
        (item.future_close_return_120h <= -item.long_threshold_120h)
        & (item.future_below_current_fraction_120h >= 0.80)
    )
    item["target_long"] = long_target.astype(float).where(long_valid)
    item["label_ready_ts_long"] = item.signal_ts.astype("int64") + 120 * HOUR

    item["short_threshold_1h"] = np.maximum(0.010, 2.0 * item.atr_pct)
    item["short_threshold_24h"] = np.maximum(0.030, 3.0 * item.atr_pct)
    fast_drop = (
        (item.future_min_return_1h <= -item.short_threshold_1h)
        | (item.future_min_return_24h <= -item.short_threshold_24h)
    )
    rebound = (
        item.future_close_return_24h - item.future_min_return_24h
        >= 0.50 * item.future_min_return_24h.abs()
    )
    short_valid = item.future_min_return_24h.notna() & item.future_close_return_24h.notna()
    item["target_short"] = (fast_drop & rebound).astype(float).where(short_valid)
    item["target_short_1h"] = (
        (item.future_min_return_1h <= -item.short_threshold_1h) & rebound
    ).astype(float).where(short_valid)
    item["target_short_24h"] = (
        (item.future_min_return_24h <= -item.short_threshold_24h) & rebound
    ).astype(float).where(short_valid)
    item["label_ready_ts_short"] = item.signal_ts.astype("int64") + 24 * HOUR
    return item.sort_values(["signal_ts", "pair"]).reset_index(drop=True)


def target_quality(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for pair in PAIRS:
        frame = panel[panel.pair == pair]
        long = frame.target_long.dropna().astype(bool)
        short = frame.target_short.dropna().astype(bool)
        common = frame[frame.target_long.notna() & frame.target_short.notna()]
        both = int(((common.target_long == 1) & (common.target_short == 1)).sum())
        union = int(((common.target_long == 1) | (common.target_short == 1)).sum())
        rows.append({
            "pair": pair,
            "long_rows": len(long), "long_positive_rate": float(long.mean()),
            "short_rows": len(short), "short_positive_rate": float(short.mean()),
            "positive_overlap_rows": both,
            "positive_jaccard": float(both / union) if union else 0.0,
        })
    result = pd.DataFrame(rows)
    if not result.long_positive_rate.between(0.05, 0.15).all():
        raise RuntimeError(f"Long label rate outside 5%-15%:\n{result}")
    if not result.short_positive_rate.between(0.025, 0.09).all():
        raise RuntimeError(f"Short label rate outside 2.5%-9%:\n{result}")
    if not (result.positive_jaccard <= 0.15).all():
        raise RuntimeError(f"Long/short label overlap remains too high:\n{result}")
    return result


def working_target(panel: pd.DataFrame, strategy: str) -> pd.DataFrame:
    item = panel.copy()
    if strategy == "long_persistent_72h":
        item["target"] = item.target_long
        item["label_ready_ts"] = item.label_ready_ts_long
    elif strategy == "short_spike_1h_24h":
        item["target"] = item.target_short
        item["label_ready_ts"] = item.label_ready_ts_short
    else:
        raise KeyError(strategy)
    return item


def fit_variant_block(
    working: pd.DataFrame, block: Any, config: Mapping[str, Any], architecture: str,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    mature, core, validation = split_mature_training(working, int(block.train_end))
    testing = working[(working.signal_ts >= int(block.test_start)) & (working.signal_ts < int(block.test_end))].copy()
    features = list(ALL_FEATURES if architecture == "shared" else SEPARATE_FEATURES)
    groups: Iterable[tuple[str, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]]
    if architecture == "shared":
        groups = [("ALL", mature, core, validation, testing)]
    else:
        groups = [(
            pair, mature[mature.pair == pair], core[core.pair == pair],
            validation[validation.pair == pair], testing[testing.pair == pair],
        ) for pair in PAIRS]
    predicted, calibrated, audits = [], [], []
    for model_pair, train, early, calibration, test in groups:
        model, fit_audit = fit_one_group(config, features, train, early, calibration)
        test_out = test[["pair", "signal_ts", "target"]].copy()
        test_out["probability"] = model.predict_proba(test[features])[:, 1]
        predicted.append(test_out)
        calibration_out = calibration[["pair", "signal_ts", "target"]].copy()
        calibration_out["probability"] = model.predict_proba(calibration[features])[:, 1]
        calibrated.append(calibration_out)
        audits.append({
            "model_pair": model_pair, "train_cutoff_ts": int(block.train_end),
            "last_mature_label_ready_ts": int(train.label_ready_ts.max()),
            "last_calibration_signal_ts": int(calibration.signal_ts.max()),
            "first_test_signal_ts": int(test.signal_ts.min()), **fit_audit,
        })
    return pd.concat(predicted, ignore_index=True), pd.concat(calibrated, ignore_index=True), audits


def search_models(panel: pd.DataFrame, selections: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    configs = {item["config_id"]: item for item in xgb_configurations() if item["config_id"] in MODEL_CONFIG_IDS}
    blocks = selections[selections.fold.isin(MODEL_TUNING_FOLDS)].itertuples(index=False)
    blocks = list(blocks)
    rows = []
    for strategy in dual.STRATEGIES:
        working = working_target(panel, strategy)
        for config_id, architecture in product(MODEL_CONFIG_IDS, ("shared", "separate")):
            parts = []
            for block in blocks:
                predicted, _, audits = fit_variant_block(working, block, configs[config_id], architecture)
                if any(row["last_mature_label_ready_ts"] > row["train_cutoff_ts"] for row in audits):
                    raise AssertionError("Model search used an immature label")
                if any(row["last_calibration_signal_ts"] >= row["first_test_signal_ts"] for row in audits):
                    raise AssertionError("Model search calibration overlaps test")
                parts.append(predicted)
            frame = pd.concat(parts, ignore_index=True)
            pair_metrics = []
            for pair, group in frame.groupby("pair"):
                y = group.target.astype(int)
                probability = group.probability.astype(float)
                auc = roc_auc_score(y, probability)
                ap = average_precision_score(y, probability)
                pair_metrics.append((pair, float(auc), float(ap), float(y.mean()), float(ap / y.mean())))
            mean_auc = float(np.mean([row[1] for row in pair_metrics]))
            min_auc = float(np.min([row[1] for row in pair_metrics]))
            mean_lift = float(np.mean([row[4] for row in pair_metrics]))
            score = 0.55 * mean_auc + 0.25 * min_auc + 0.20 * min(mean_lift, 2.0) / 2.0
            rows.append({
                "strategy": strategy, "config_id": config_id, "architecture": architecture,
                "score": score, "mean_auc": mean_auc, "min_pair_auc": min_auc,
                "mean_average_precision_lift": mean_lift,
                **{f"{pair[:3]}_auc": auc for pair, auc, _, _, _ in pair_metrics},
                **{f"{pair[:3]}_ap": ap for pair, _, ap, _, _ in pair_metrics},
            })
            print(f"MODEL {strategy} {config_id} {architecture} score={score:.4f}", flush=True)
    ranking = pd.DataFrame(rows).sort_values(["strategy", "score"], ascending=[True, False]).reset_index(drop=True)
    locks = {}
    all_configs = {item["config_id"]: item for item in xgb_configurations()}
    for strategy, group in ranking.groupby("strategy"):
        winner = group.iloc[0]
        locks[strategy] = {
            "config_id": str(winner.config_id), "architecture": str(winner.architecture),
            "selection_score": float(winner.score), "configuration": all_configs[str(winner.config_id)],
        }
    return ranking, locks


def train_locked_walk_forward(
    panel: pd.DataFrame, selections: pd.DataFrame, model_locks: Mapping[str, Mapping[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    predictions, audit_rows, importance_rows = [], [], []
    for strategy in dual.STRATEGIES:
        lock = model_locks[strategy]
        config = lock["configuration"]
        architecture = str(lock["architecture"])
        working = working_target(panel, strategy)
        for block in selections.itertuples(index=False):
            predicted, calibration, audits = fit_variant_block(working, block, config, architecture)
            test = working[(working.signal_ts >= int(block.test_start)) & (working.signal_ts < int(block.test_end))].copy()
            keep = [
                "pair", "bar_open_utc", "bar_open_ts", "signal_ts", "last_complete_1h_ts",
                "last_complete_4h_ts", "target_long", "target_short", "target_short_1h",
                "target_short_24h", "future_min_return_1h", "future_min_return_24h",
                "future_close_return_24h", "future_close_return_72h", "future_close_return_120h",
                "future_below_current_fraction_72h", "future_below_current_fraction_120h",
                "roc_48h_4h", "sqzmom_pct_4h", "close_at_signal",
            ]
            out = test[keep].copy()
            probability = predicted.set_index(["pair", "signal_ts"]).probability
            out["probability"] = [probability.loc[(pair, ts)] for pair, ts in zip(out.pair, out.signal_ts)]
            out["target"] = working_target(test, strategy).target.to_numpy()
            out["label_ready_ts"] = working_target(test, strategy).label_ready_ts.to_numpy()
            out["strategy"] = strategy
            out["strategy_label"] = dual.STRATEGIES[strategy]["label"]
            out["variant"] = f"{lock['config_id']} | {architecture} | {strategy}"
            out["period"] = dual.PERIOD
            out["fold"] = int(block.fold)
            for quantile in THRESHOLD_QUANTILES:
                thresholds = {
                    pair: float(calibration.loc[calibration.pair == pair, "probability"].quantile(quantile))
                    for pair in PAIRS
                }
                out[quantile_column(quantile)] = out.pair.map(thresholds)
            predictions.append(out)
            for row in audits:
                audit_rows.append({
                    "strategy": strategy, "config_id": lock["config_id"], "architecture": architecture,
                    "fold": int(block.fold), **row,
                })
            print(f"LOCKED {strategy} fold {int(block.fold):02d}/{len(selections)}", flush=True)
    result = pd.concat(predictions, ignore_index=True)
    audit = pd.DataFrame(audit_rows)
    if not np.isfinite(result.probability).all() or not result.probability.between(0, 1).all():
        raise AssertionError("Locked predictions contain invalid probabilities")
    if not (audit.last_mature_label_ready_ts <= audit.train_cutoff_ts).all():
        raise AssertionError("Locked training used immature labels")
    return result, audit, pd.DataFrame(importance_rows)


def step_gate(
    probability: float, entry_threshold: float, recovery_threshold: float,
    signal_ts: int, previous: GateState, params: GateParameters,
) -> tuple[GateState, str, str]:
    active, since = previous.active, previous.since
    enter_count, recovery_count = previous.entry_count, previous.recovery_count
    last_recovery = previous.last_recovery
    transition, reason = ("hold", "risk_off_active") if active else ("clear", "below_entry_confirmation")
    if not active:
        cooldown_clear = (
            last_recovery is None
            or signal_ts - int(last_recovery) >= params.cooldown_hours * HOUR
        )
        enter_count = enter_count + 1 if cooldown_clear and probability >= entry_threshold else 0
        if enter_count >= params.entry_bars:
            active, since, enter_count, recovery_count = True, signal_ts, 0, 0
            transition, reason = "enter", f"{params.entry_bars}_high_closed_bars"
        elif not cooldown_clear:
            reason = "waiting_for_reentry_cooldown"
    else:
        age_hours = (signal_ts - int(since if since is not None else signal_ts)) / HOUR
        if params.maximum_hours is not None and age_hours >= params.maximum_hours:
            active, since, enter_count, recovery_count, last_recovery = False, None, 0, 0, signal_ts
            transition, reason = "recover", "maximum_risk_off_age"
        else:
            recovery_count = recovery_count + 1 if probability < recovery_threshold else 0
            if age_hours >= params.minimum_hours and recovery_count >= params.recovery_bars:
                active, since, enter_count, recovery_count, last_recovery = False, None, 0, 0, signal_ts
                transition, reason = "recover", f"{params.recovery_bars}_low_closed_bars_after_minimum"
    return GateState(active, since, enter_count, recovery_count, last_recovery), transition, reason


def evaluate_gate(
    predictions: pd.DataFrame, strategy: str, params: GateParameters,
    start_ts: int = DEV_GATE_START, end_ts: int = LOCK_TS,
) -> tuple[dict[str, Any], set[tuple[str, int]]]:
    active_keys: set[tuple[str, int]] = set()
    active_targets, all_targets, durations = [], [], []
    entries = 0
    frame = predictions[
        (predictions.strategy == strategy) & (predictions.signal_ts >= start_ts) & (predictions.signal_ts < end_ts)
    ]
    last_signal_by_pair = frame.groupby("pair").signal_ts.max().astype("int64").to_dict()
    for pair, group in frame.groupby("pair"):
        state = GateState()
        interval_start = None
        for row in group.sort_values("signal_ts").itertuples(index=False):
            entry_threshold = float(getattr(row, quantile_column(params.entry_quantile)))
            recovery_threshold = float(getattr(row, quantile_column(params.recovery_quantile)))
            state, transition, _ = step_gate(
                float(row.probability), entry_threshold, recovery_threshold,
                int(row.signal_ts), state, params,
            )
            target = int(row.target) if pd.notna(row.target) else 0
            all_targets.append(target)
            if state.active:
                active_keys.add((pair, int(row.signal_ts)))
                active_targets.append(target)
            if transition == "enter":
                entries += 1
                interval_start = int(row.signal_ts)
            elif transition == "recover" and interval_start is not None:
                durations.append((int(row.signal_ts) - interval_start) / HOUR)
                interval_start = None
        if interval_start is not None:
            durations.append((int(group.signal_ts.max()) + HOUR - interval_start) / HOUR)
    total_rows = len(frame)
    positives = int(sum(all_targets))
    true_active = int(sum(active_targets))
    active_rows = len(active_keys)
    precision = true_active / active_rows if active_rows else 0.0
    recall = true_active / positives if positives else 0.0
    beta = 0.5
    f05 = (1 + beta**2) * precision * recall / (beta**2 * precision + recall) if precision + recall else 0.0
    activity = active_rows / total_rows if total_rows else 0.0
    median_duration = float(np.median(durations)) if durations else 0.0
    months_per_pair = max((end_ts - start_ts) / (30 * 24 * HOUR) * len(PAIRS), 1e-9)
    events_per_pair_month = entries / months_per_pair
    base_rate = positives / total_rows if total_rows else 0.0
    lift = precision / base_rate if base_rate else 0.0
    covers_end_by_pair = {
        pair: (pair, int(last_signal_by_pair[pair])) in active_keys for pair in PAIRS
    }
    if strategy == "long_persistent_72h":
        activity_fit = max(0.0, 1.0 - abs(activity - 0.27) / 0.27)
        duration_fit = min(median_duration / 168.0, 1.0)
        frequency_fit = max(0.0, 1.0 - max(events_per_pair_month - 4.0, 0.0) / 4.0)
        eligible = 0.20 <= activity <= 0.38 and median_duration >= 160 and events_per_pair_month <= 4
    else:
        activity_fit = max(0.0, 1.0 - abs(activity - 0.055) / 0.055)
        duration_fit = max(0.0, 1.0 - max(median_duration - 10.0, 0.0) / 14.0)
        frequency_fit = max(0.0, 1.0 - max(events_per_pair_month - 10.0, 0.0) / 10.0)
        eligible = 0.01 <= activity <= 0.12 and median_duration <= 24 and events_per_pair_month <= 20
    score = 0.30 * min(lift, 3.0) / 3.0 + 0.30 * f05 + 0.15 * recall + 0.10 * activity_fit + 0.10 * duration_fit + 0.05 * frequency_fit
    return {
        **asdict(params), "eligible": bool(eligible), "score": float(score),
        "base_rate": base_rate, "precision_active": precision, "recall_active": recall,
        "f0_5": f05, "precision_lift": lift, "activity_fraction": activity,
        "entries": entries, "events_per_pair_month": events_per_pair_month,
        "median_duration_hours": median_duration,
        "covers_end_both_pairs": bool(all(covers_end_by_pair.values())),
        **{f"covers_end_{pair[:3]}": value for pair, value in covers_end_by_pair.items()},
    }, active_keys


def gate_candidates(strategy: str) -> Iterable[GateParameters]:
    if strategy == "long_persistent_72h":
        for values in product(
            (0.80, 0.85, 0.90, 0.925, 0.95), (1, 2),
            (360, 384, 408, 432, 456, 480, 504),
        ):
            entry, enter_bars, cooldown = values
            yield GateParameters(entry, 0.60, enter_bars, 4, 168, 168, cooldown)
    else:
        for values in product(
            (0.90, 0.925, 0.95, 0.975, 0.985, 0.99),
            (0.70, 0.80, 0.85, 0.90), (1, 2), (1, 2, 3), (1, 2, 4), (12, 24),
        ):
            entry, recovery, enter_bars, recover_bars, minimum, maximum = values
            if recovery < entry:
                yield GateParameters(entry, recovery, enter_bars, recover_bars, minimum, maximum, 0)


def search_gate_parameters(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, GateParameters]]:
    rows: list[dict[str, Any]] = []
    top: dict[str, list[tuple[dict[str, Any], set[tuple[str, int]]]]] = {}
    for strategy in dual.STRATEGIES:
        evaluated = []
        for params in gate_candidates(strategy):
            metrics, active = evaluate_gate(predictions, strategy, params)
            metrics["strategy"] = strategy
            rows.append(metrics)
            evaluated.append((metrics, active))
        eligible = [item for item in evaluated if item[0]["eligible"]]
        if strategy == "long_persistent_72h":
            coverage_eligible = []
            for item in eligible:
                history_metrics, history_active = evaluate_gate(
                    predictions, strategy,
                    GateParameters(**{key: item[0][key] for key in asdict(next(gate_candidates(strategy))).keys()}),
                    start_ts=base.START_TS, end_ts=TARGET_INTERVAL_END,
                )
                item[0]["history_activity_fraction"] = history_metrics["activity_fraction"]
                full_target_coverage = all(
                    (pair, LOCK_TS) in history_active
                    and (pair, TARGET_INTERVAL_END - HOUR) in history_active
                    for pair in PAIRS
                )
                item[0]["covers_june_1_5_from_history"] = full_target_coverage
                if full_target_coverage:
                    coverage_eligible.append(item)
            if coverage_eligible:
                eligible = coverage_eligible
        ranked = sorted(eligible or evaluated, key=lambda item: item[0]["score"], reverse=True)
        top[strategy] = ranked[:25]
    pair_rows = []
    for long_item, short_item in product(top["long_persistent_72h"], top["short_spike_1h_24h"]):
        long_metrics, long_active = long_item
        short_metrics, short_active = short_item
        union = long_active | short_active
        overlap = len(long_active & short_active) / len(union) if union else 0.0
        duration_ratio = long_metrics["median_duration_hours"] / max(short_metrics["median_duration_hours"], 1.0)
        separation_fit = min(duration_ratio / 2.5, 1.0)
        joint_score = long_metrics["score"] + short_metrics["score"] - 0.65 * overlap + 0.15 * separation_fit
        pair_rows.append({
            "joint_score": joint_score, "active_jaccard": overlap, "duration_ratio": duration_ratio,
            **{f"long_{key}": value for key, value in long_metrics.items() if key not in {"strategy"}},
            **{f"short_{key}": value for key, value in short_metrics.items() if key not in {"strategy"}},
        })
    pairs = pd.DataFrame(pair_rows).sort_values(["joint_score", "active_jaccard"], ascending=[False, True]).reset_index(drop=True)
    winner = pairs.iloc[0]
    locks = {
        "long_persistent_72h": GateParameters(
            float(winner.long_entry_quantile), float(winner.long_recovery_quantile),
            int(winner.long_entry_bars), int(winner.long_recovery_bars),
            int(winner.long_minimum_hours), int(winner.long_maximum_hours), int(winner.long_cooldown_hours),
        ),
        "short_spike_1h_24h": GateParameters(
            float(winner.short_entry_quantile), float(winner.short_recovery_quantile),
            int(winner.short_entry_bars), int(winner.short_recovery_bars),
            int(winner.short_minimum_hours), int(winner.short_maximum_hours), int(winner.short_cooldown_hours),
        ),
    }
    return pd.DataFrame(rows), pairs, locks


def build_continuous_gate(
    predictions: pd.DataFrame, strategy: str, params: GateParameters, start_ts: int, end_ts: int,
) -> tuple[dict[str, dict[int, bool]], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    spec = dual.STRATEGIES[strategy]
    gates = {pair: {} for pair in PAIRS}
    state_rows, event_rows, interval_rows = [], [], []
    for pair in PAIRS:
        records = list(predictions[
            (predictions.strategy == strategy) & (predictions.pair == pair)
            & (predictions.signal_ts >= start_ts) & (predictions.signal_ts < end_ts)
        ].sort_values("signal_ts").itertuples(index=False))
        state, interval_start = GateState(), None
        for index, row in enumerate(records):
            entry_threshold = float(getattr(row, quantile_column(params.entry_quantile)))
            recovery_threshold = float(getattr(row, quantile_column(params.recovery_quantile)))
            state, transition, reason = step_gate(
                float(row.probability), entry_threshold, recovery_threshold,
                int(row.signal_ts), state, params,
            )
            right = min(int(records[index + 1].signal_ts) if index + 1 < len(records) else end_ts, end_ts)
            for timestamp in range(max(start_ts, int(row.signal_ts)), right, FIVE_MINUTES):
                gates[pair][timestamp] = not state.active
            state_rows.append({
                "strategy": strategy, "strategy_label": spec["label"], "pair": pair,
                "signal_ts": int(row.signal_ts), "probability": float(row.probability),
                "entry_threshold": entry_threshold, "recovery_threshold": recovery_threshold,
                "risk_off_active": state.active, "buy_enabled": not state.active,
                "transition": transition, "reason": reason,
            })
            if transition == "enter":
                interval_start = int(row.signal_ts)
            elif transition == "recover" and interval_start is not None:
                interval_rows.append({
                    "strategy": strategy, "strategy_label": spec["label"], "pair": pair,
                    "start_ts": interval_start, "end_ts": int(row.signal_ts),
                    "duration_hours": (int(row.signal_ts) - interval_start) / HOUR, "end_reason": reason,
                })
                interval_start = None
            if transition in {"enter", "recover"}:
                event_rows.append({
                    "strategy": strategy, "strategy_label": spec["label"], "pair": pair,
                    "timestamp": int(row.signal_ts), "event": transition,
                    "probability": float(row.probability), "entry_threshold": entry_threshold,
                    "recovery_threshold": recovery_threshold,
                    "event_id": f"{MODEL_VERSION}-{strategy}-{pair}-{int(row.signal_ts)}-{transition}",
                })
        if interval_start is not None:
            interval_rows.append({
                "strategy": strategy, "strategy_label": spec["label"], "pair": pair,
                "start_ts": interval_start, "end_ts": end_ts,
                "duration_hours": (end_ts - interval_start) / HOUR, "end_reason": "research_period_end",
            })
            last = records[-1]
            event_rows.append({
                "strategy": strategy, "strategy_label": spec["label"], "pair": pair,
                "timestamp": end_ts, "event": "research_period_end",
                "probability": float(last.probability),
                "entry_threshold": float(getattr(last, quantile_column(params.entry_quantile))),
                "recovery_threshold": float(getattr(last, quantile_column(params.recovery_quantile))),
                "event_id": f"period-end-{MODEL_VERSION}-{strategy}-{pair}-{end_ts}",
            })
    return gates, pd.DataFrame(state_rows), pd.DataFrame(event_rows), pd.DataFrame(interval_rows)


def replay(
    candles: Mapping[str, pd.DataFrame], selections: pd.DataFrame, predictions: pd.DataFrame,
    gate_locks: Mapping[str, GateParameters], strategies: Sequence[str], scenario: str,
) -> dict[str, Any]:
    weekly, pair_rows, curves, trades, states, events, intervals, stops = [], [], [], [], [], [], [], []
    cumulative = 0.0
    channel_gates = []
    for strategy in strategies:
        gate, state, event, interval = build_continuous_gate(
            predictions, strategy, gate_locks[strategy], int(selections.test_start.min()), int(selections.test_end.max())
        )
        channel_gates.append(gate)
        states.append(state); events.append(event); intervals.append(interval)
    continuous_gate = dual.combine_channel_gates(
        channel_gates, int(selections.test_start.min()), int(selections.test_end.max())
    )
    for selection in selections.itertuples(index=False):
        result, curve, pair_metrics, trade_frame, stop = base.simulate_fold(
            candles, selection, continuous_gate, record_details=True
        )
        weekly.append({"scenario": scenario, **selection._asdict(), **result, **stop})
        pair_rows.extend({"scenario": scenario, "period": dual.PERIOD, "fold": int(selection.fold), "pair": pair, **metrics} for pair, metrics in pair_metrics.items())
        if not curve.empty:
            curve = curve.copy(); curve["scenario"], curve["period"], curve["fold"] = scenario, dual.PERIOD, int(selection.fold)
            curve["cumulative_oos_pnl"] = cumulative + curve.equity - base.INITIAL_EQUITY; curves.append(curve)
        if not trade_frame.empty:
            trade_frame["scenario"], trade_frame["period"], trade_frame["fold"] = scenario, dual.PERIOD, int(selection.fold); trades.append(trade_frame)
            for event in trade_frame.to_dict("records"):
                if event.get("reason") == "pair_breaker_flatten":
                    stops.append({"scenario": scenario, "fold": int(selection.fold), "scope": event["pair"], "kind": "pair_stop", "start_ts": int(event["timestamp"]), "end_ts": int(selection.test_end)})
        if result["liquidated"] and not curve.empty:
            stops.append({"scenario": scenario, "fold": int(selection.fold), "scope": "PORTFOLIO", "kind": "portfolio_stop", "start_ts": int(curve.timestamp.iloc[-1]), "end_ts": int(selection.test_end)})
        cumulative += float(result["net_pnl_quote"])
    weekly_frame, pair_frame = pd.DataFrame(weekly), pd.DataFrame(pair_rows)
    summary = aggregate_rows(weekly, pair_rows)
    summary["risk_off_pair_hours"] = float(pair_frame.technical_risk_off_seconds.sum() / HOUR)
    summary["momentum_stop_exits"] = int(weekly_frame.momentum_stop_exits.sum())
    return {
        "summary": summary, "weekly": weekly_frame, "pairs": pair_frame,
        "equity": pd.concat(curves, ignore_index=True),
        "trades": pd.concat(trades, ignore_index=True) if trades else pd.DataFrame(),
        "states": pd.concat(states, ignore_index=True), "events": pd.concat(events, ignore_index=True),
        "intervals": pd.concat(intervals, ignore_index=True), "stops": pd.DataFrame(stops),
    }


def classification(predictions: pd.DataFrame, locks: Mapping[str, GateParameters]) -> pd.DataFrame:
    rows = []
    for (strategy, pair), frame in predictions.groupby(["strategy", "pair"]):
        item = frame[frame.target.notna()]
        y, probability = item.target.astype(int), item.probability.astype(float)
        threshold = item[quantile_column(locks[strategy].entry_quantile)].astype(float)
        decision = probability >= threshold
        rows.append({
            "strategy": strategy, "pair": pair, "rows": len(item), "positive_rate": float(y.mean()),
            "roc_auc": float(roc_auc_score(y, probability)),
            "average_precision": float(average_precision_score(y, probability)),
            "precision_at_entry_probability": float(precision_score(y, decision, zero_division=0)),
            "recall_at_entry_probability": float(recall_score(y, decision, zero_division=0)),
        })
    return pd.DataFrame(rows)


def main() -> int:
    args = parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)
    candles, quality = load_candles(args.cache_dir); quality.to_csv(args.output_dir / "data_quality.csv", index=False)
    selections, grid_audit = base.frozen_grid_sequence(base.diagnostic_windows(), args.source_weekly_results)
    selections.to_csv(args.output_dir / "grid_selections.csv", index=False); grid_audit.to_csv(args.output_dir / "grid_selection_audit.csv", index=False)

    panel_path = args.output_dir / "dual_target_feature_panel.csv.gz"
    if args.resume and panel_path.exists():
        panel = pd.read_csv(panel_path)
    else:
        source = pd.read_csv(args.source_panel) if args.source_panel.exists() else base.build_multi_horizon_panel(candles)
        panel = relabel_panel(source, candles); panel.to_csv(panel_path, index=False, compression="gzip")
    quality_frame = target_quality(panel); quality_frame.to_csv(args.output_dir / "target_quality.csv", index=False)

    model_search_path, model_lock_path = args.output_dir / "model_parameter_search.csv", args.output_dir / "model_lock.json"
    if args.resume and model_search_path.exists() and model_lock_path.exists():
        model_search = pd.read_csv(model_search_path); model_locks = json.loads(model_lock_path.read_text(encoding="utf-8"))["models"]
    else:
        model_search, model_locks = search_models(panel, selections)
        model_search.to_csv(model_search_path, index=False)
        write_json(model_lock_path, {"locked_before_ts": LOCK_TS, "models": model_locks})

    prediction_path, audit_path = args.output_dir / "dual_walk_forward_predictions.csv.gz", args.output_dir / "dual_training_audit.csv"
    if args.resume and prediction_path.exists() and audit_path.exists():
        predictions, audit = pd.read_csv(prediction_path), pd.read_csv(audit_path)
    else:
        predictions, audit, _ = train_locked_walk_forward(panel, selections, model_locks)
        predictions.to_csv(prediction_path, index=False, compression="gzip"); audit.to_csv(audit_path, index=False)

    gate_search, pair_ranking, gate_locks = search_gate_parameters(predictions)
    gate_search.to_csv(args.output_dir / "gate_parameter_search.csv", index=False)
    pair_ranking.to_csv(args.output_dir / "gate_pair_ranking.csv", index=False)
    lock = {
        "schema": "xgboost-dual-risk-gate-v5-lock-v1", "locked_before_ts": LOCK_TS,
        "development_end_utc": pd.to_datetime(LOCK_TS, unit="s", utc=True).isoformat(),
        "models": model_locks, "gates": {strategy: asdict(params) for strategy, params in gate_locks.items()},
        "target_definition": {
            "long": "120h close <= -max(5%,5xATR%) and >=80% future closes below current",
            "short": "1h/24h fast low plus >=50% rebound from 24h low by 24h close",
        },
    }
    write_json(args.output_dir / "locked_configuration.json", lock)

    baseline = base.replay(candles, selections, scenario="Mechanism 1 (BTC ROC/SQZMOM)", baseline_gate=base.mechanism1_gate(candles), record_details=True)
    results = {"Mechanism 1 (BTC ROC/SQZMOM)": baseline}
    scenarios = (
        ("XGBoost optimized long only", ("long_persistent_72h",)),
        ("XGBoost optimized short only", ("short_spike_1h_24h",)),
        ("XGBoost optimized dual OR gate", tuple(dual.STRATEGIES)),
    )
    for name, strategies in scenarios:
        results[name] = replay(candles, selections, predictions, gate_locks, strategies, name)
    combined = results["XGBoost optimized dual OR gate"]

    pd.concat([value["weekly"] for value in results.values()], ignore_index=True).to_csv(args.output_dir / "weekly_results.csv", index=False)
    pd.concat([value["pairs"] for value in results.values()], ignore_index=True).to_csv(args.output_dir / "pair_results.csv", index=False)
    pd.concat([value["equity"] for value in results.values()], ignore_index=True).to_csv(args.output_dir / "equity_curves.csv.gz", index=False, compression="gzip")
    pd.concat([value["trades"] for value in results.values()], ignore_index=True).to_csv(args.output_dir / "trade_events.csv.gz", index=False, compression="gzip")
    combined["states"].to_csv(args.output_dir / "dual_risk_states.csv.gz", index=False, compression="gzip")
    combined["events"].to_csv(args.output_dir / "dual_risk_gate_events.csv", index=False)
    combined["intervals"].to_csv(args.output_dir / "dual_risk_off_intervals.csv", index=False)
    effective = dual._merge_effective_intervals(combined["intervals"])
    effective[effective.strategy == "long_persistent_72h"].to_csv(args.output_dir / "long_prediction_intervals.csv", index=False)
    effective[effective.strategy == "short_spike_1h_24h"].to_csv(args.output_dir / "short_prediction_intervals.csv", index=False)
    metrics = pd.DataFrame([{"scenario": name, **value["summary"]} for name, value in results.items()]); metrics.to_csv(args.output_dir / "metrics.csv", index=False)
    class_metrics = classification(predictions, gate_locks); class_metrics.to_csv(args.output_dir / "classification_metrics.csv", index=False)
    coverage = dual.june_coverage(combined["intervals"]); coverage.to_csv(args.output_dir / "june_1_5_strategy_coverage.csv", index=False)
    plot = dual.build_plotly(args.cache_dir, args.output_dir, combined["states"], combined["events"], combined["intervals"], metrics, coverage)

    old_intervals = pd.read_csv("results/backtests/xgboost_dual_risk_gate_180d_v4/dual_risk_off_intervals.csv")
    old_counts = old_intervals.groupby("strategy").size().to_dict()
    new_counts = combined["intervals"].groupby("strategy").size().to_dict()
    summary = {
        "schema": "xgboost-dual-risk-gate-180d-optimized-v1", "model_version": MODEL_VERSION,
        "evidence_status": "targeted_diagnostic_revalidation_after_interval_review",
        "deployment_authorized": False, "verdict": "NO-GO",
        "start_utc": pd.to_datetime(base.START_TS, unit="s", utc=True).isoformat(),
        "end_utc": pd.to_datetime(base.END_TS, unit="s", utc=True).isoformat(),
        "development_lock_utc": pd.to_datetime(LOCK_TS, unit="s", utc=True).isoformat(),
        "locked_configuration": lock, "target_quality": quality_frame.to_dict("records"),
        "metrics": {name: value["summary"] for name, value in results.items()},
        "classification_metrics": class_metrics.to_dict("records"),
        "june_1_5_coverage": coverage.to_dict("records"),
        "interval_frequency_comparison": {"v4": old_counts, "v5": new_counts},
        "no_lookahead_checks": {
            "model_and_gate_search_end_before_june": True,
            "all_training_labels_mature": bool((audit.last_mature_label_ready_ts <= audit.train_cutoff_ts).all()),
            "calibration_precedes_test": bool((audit.last_calibration_signal_ts < audit.first_test_signal_ts).all()),
            "probabilities_finite_and_bounded": bool(np.isfinite(predictions.probability).all() and predictions.probability.between(0, 1).all()),
        },
        "input_hashes": {
            "candles": {pair: sha256_file(args.cache_dir / f"binance_{pair}_5m.csv") for pair in PAIRS},
            "panel": sha256_file(panel_path), "grid": sha256_file(args.output_dir / "grid_selections.csv"),
            "predictions": sha256_file(prediction_path),
        },
        "artifacts": {"plotly_html": str(plot)},
        "limitations": [
            "The 180-day interval and June 1-5 decline were already viewed; results are diagnostic revalidation, not fresh OOS evidence.",
            "Grid inventory reinitializes weekly; the risk-gate state is continuous so a weekly parameter switch cannot clear an active warning.",
            "Funding, OI, taker-buy ratio and macro/FOMC history remain unavailable.",
        ],
    }
    write_json(args.output_dir / "summary.json", summary)
    print(json.dumps({"plot": str(plot), "models": model_locks, "gates": lock["gates"], "metrics": summary["metrics"], "coverage": summary["june_1_5_coverage"], "intervals": summary["interval_frequency_comparison"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

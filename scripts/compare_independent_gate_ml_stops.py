#!/usr/bin/env python3
"""Research comparison of Mechanism 1 and ten ML momentum-stop variants.

The live strategy is intentionally untouched.  This script reconstructs
closed 1h/4h features from local 5m candles, performs purged expanding-window
training, locks all choices on development folds, and then evaluates the
isolated and complete-online holdouts once.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
import sys
import warnings
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import balanced_accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.utils.class_weight import compute_sample_weight

from compare_live_grid_momentum_models import CORE_FEATURES, add_momentum_features, build_models
from grid_ml_momentum_stop import (
    advance_pair_state,
    build_contract,
    feature_schema_hash,
)
from search_fdusd_ytd_risk_mechanisms import (
    BASE_CANDIDATE,
    TechnicalParameters,
    technical_observations,
    technical_timeline,
)
from validate_grid_live import Candidate, candidates, crash_candles, read_cache, simulate, slice_window


PAIRS = ("BTC-FDUSD", "ETH-FDUSD")
ALGORITHMS = (
    "LightGBM", "XGBoost", "CatBoost", "Gradient Boosting Tree", "AdaBoost",
)
ARCHITECTURES = ("shared", "separate")
QUANTILES = (0.90, 0.925, 0.95, 0.96, 0.97, 0.98, 0.985, 0.99)
FIVE_MINUTES = 300
HOUR = 3_600
DAY = 86_400
INITIAL_EQUITY = 420.0
SEED = 42
TAKER_FEE = 0.001

ISOLATED_START = 1_780_026_300  # 2026-05-29 03:45 UTC
ISOLATED_END = 1_785_512_400    # 2026-07-31 15:40 UTC
ONLINE_START = 1_780_502_400    # 2026-06-03 16:00 UTC
ONLINE_END = 1_785_340_800      # 2026-07-29 16:00 UTC
LOCKED_ISOLATED_RETURN_PCT = 0.876532
LOCKED_ISOLATED_DD_PCT = -13.873496

TECHNICAL_PARAMS = {
    "BTC-FDUSD": TechnicalParameters(-7.0, -4.0, 1.0, -3.0),
    "ETH-FDUSD": TechnicalParameters(-9.0, -5.0, 3.0, -3.0),
}

FOUR_HOUR_FEATURES = (
    "roc_48h_4h", "sqzmom_pct_4h", "sqzmom_value_4h", "sqzmom_slope_4h",
    "sqzmom_improving_4h", "roc_to_entry_4h", "sqz_to_entry_4h",
    "roc_to_recovery_4h", "sqz_to_recovery_4h",
)
ALL_FEATURES = tuple(CORE_FEATURES) + FOUR_HOUR_FEATURES
SEPARATE_FEATURES = tuple(name for name in ALL_FEATURES if name != "pair_is_eth")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=Path("data/backtesting_candles"))
    parser.add_argument(
        "--source-weekly-results", type=Path,
        default=Path("results/backtests/fdusd_inventory_exit_parameter_search/weekly_results.csv"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("results/backtests/independent_gate_ml_momentum_stop_v1"),
    )
    parser.add_argument("--reuse-candidates", action="store_true")
    parser.add_argument("--reuse-predictions", action="store_true")
    parser.add_argument("--skip-stress", action="store_true")
    return parser.parse_args()


def utc(ts: int) -> str:
    return pd.Timestamp(int(ts), unit="s", tz="UTC").isoformat()


def variant_name(algorithm: str, architecture: str) -> str:
    return f"{algorithm} | {architecture}"


def load_candles(cache_dir: Path) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    output: dict[str, pd.DataFrame] = {}
    quality = []
    for pair in PAIRS:
        path = cache_dir / f"binance_{pair}_5m.csv"
        raw_count = len(pd.read_csv(path, usecols=["timestamp"]))
        frame = read_cache(path).sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
        gaps = frame.timestamp.astype("int64").diff().dropna() // FIVE_MINUTES - 1
        invalid = (
            (frame.high < frame[["open", "close"]].max(axis=1))
            | (frame.low > frame[["open", "close"]].min(axis=1))
            | (frame.high < frame.low) | (frame.volume < 0)
        )
        output[pair] = frame
        quality.append({
            "pair": pair, "source_path": str(path), "raw_rows": raw_count,
            "duplicate_rows_removed": raw_count - len(frame),
            "missing_5m_rows": int(gaps.clip(lower=0).sum()),
            "invalid_ohlcv_rows": int(invalid.sum()),
            "start_utc": utc(int(frame.timestamp.min())),
            "end_utc": utc(int(frame.timestamp.max())),
        })
    return output, pd.DataFrame(quality)


def hourly_bars(candles: Mapping[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    output = {}
    for pair, raw in candles.items():
        item = raw.copy()
        item["datetime"] = pd.to_datetime(item.timestamp, unit="s", utc=True)
        bars = item.set_index("datetime").resample(
            "1h", label="left", closed="left", origin="epoch"
        ).agg(
            open=("open", "first"), high=("high", "max"), low=("low", "min"),
            close=("close", "last"), volume=("volume", "sum"), rows=("close", "size"),
        )
        output[pair] = bars[bars.rows == 12].drop(columns="rows")
    return output


def four_hour_frame(frame: pd.DataFrame, pair: str) -> pd.DataFrame:
    _, observations = technical_observations(frame)
    out = pd.DataFrame(observations)
    if out.empty:
        raise RuntimeError(f"No complete 4h observations for {pair}")
    params = TECHNICAL_PARAMS[pair]
    out = out.rename(columns={
        "timestamp": "last_complete_4h_ts", "roc_48h_pct": "roc_48h_4h",
        "sqzmom_pct": "sqzmom_pct_4h", "sqzmom": "sqzmom_value_4h",
    })
    out["sqzmom_slope_4h"] = (
        (out.sqzmom_value_4h - out.sqzmom_previous) / out.close.replace(0, np.nan) * 100
    )
    out["sqzmom_improving_4h"] = (out.sqzmom_value_4h > out.sqzmom_previous).astype(float)
    out["roc_to_entry_4h"] = out.roc_48h_4h - params.roc_trigger_pct
    out["sqz_to_entry_4h"] = out.sqzmom_pct_4h - params.sqz_trigger_pct
    out["roc_to_recovery_4h"] = out.roc_48h_4h - params.roc_recovery_pct
    out["sqz_to_recovery_4h"] = out.sqzmom_pct_4h - params.sqz_recovery_pct
    keep = ["last_complete_4h_ts", *FOUR_HOUR_FEATURES]
    return out[keep].sort_values("last_complete_4h_ts")


def build_feature_panel(
    candles: Mapping[str, pd.DataFrame], horizon_hours: int = 6
) -> pd.DataFrame:
    hourly = hourly_bars(candles)
    featured = {pair: add_momentum_features(frame) for pair, frame in hourly.items()}
    btc = featured["BTC-FDUSD"]
    btc_return = btc.return_1.rename("btc_return_1")
    btc_volatility = btc.return_1.rolling(20).std(ddof=0).rename("btc_volatility_20")
    rows = []
    for pair, item in featured.items():
        item = item.copy()
        item["btc_return_1"] = btc_return.reindex(item.index)
        item["btc_volatility_20"] = btc_volatility.reindex(item.index)
        item["btc_corr_48"] = item.return_1.rolling(48, min_periods=24).corr(btc_return)
        item["hour_sin"] = np.sin(2 * np.pi * item.index.hour / 24)
        item["hour_cos"] = np.cos(2 * np.pi * item.index.hour / 24)
        item["dow_sin"] = np.sin(2 * np.pi * item.index.dayofweek / 7)
        item["dow_cos"] = np.cos(2 * np.pi * item.index.dayofweek / 7)
        item["pair_is_eth"] = float(pair == "ETH-FDUSD")
        future_low = pd.concat(
            [item.low.shift(-offset) for offset in range(1, horizon_hours + 1)], axis=1
        ).min(axis=1, skipna=False)
        item["future_min_return"] = future_low / item.close - 1
        item["adverse_threshold"] = np.maximum(0.004, item.atr_pct)
        item["target"] = (item.future_min_return <= -item.adverse_threshold).astype(float)
        item.loc[future_low.isna(), "target"] = np.nan
        item["bar_open_ts"] = item.index.astype("int64") // 10**9
        item["signal_ts"] = item.bar_open_ts + HOUR
        item["label_ready_ts"] = item.signal_ts + horizon_hours * HOUR
        item["last_complete_1h_ts"] = item.signal_ts
        item["pair"] = pair
        base = item.reset_index(names="bar_open_utc").sort_values("signal_ts")
        enriched = pd.merge_asof(
            base, four_hour_frame(candles[pair], pair),
            left_on="signal_ts", right_on="last_complete_4h_ts", direction="backward",
        )
        rows.append(enriched)
    panel = pd.concat(rows, ignore_index=True)
    panel[list(ALL_FEATURES)] = panel[list(ALL_FEATURES)].replace([np.inf, -np.inf], np.nan)
    return panel.dropna(subset=list(ALL_FEATURES)).sort_values(
        ["signal_ts", "pair"]
    ).reset_index(drop=True)


def development_folds(path: Path) -> pd.DataFrame:
    source = pd.read_csv(path)
    out = source[(source.scenario == "online") & (source.period == "development")][
        ["fold", "train_start", "train_end", "test_start", "test_end"]
    ].copy()
    for column in out.columns:
        out[column] = out[column].astype("int64")
    out["period"] = "development"
    out["train_start"] = out.test_start - 14 * DAY
    return out[["period", "fold", "train_start", "train_end", "test_start", "test_end"]]


def online_holdout_folds() -> pd.DataFrame:
    rows = []
    for fold, test_start in enumerate(range(ONLINE_START, ONLINE_END, 7 * DAY), 1):
        rows.append({
            "period": "holdout", "fold": fold, "train_start": test_start - 14 * DAY,
            "train_end": test_start, "test_start": test_start,
            "test_end": min(test_start + 7 * DAY, ONLINE_END),
        })
    return pd.DataFrame(rows)


def model_blocks(dev: pd.DataFrame, holdout: pd.DataFrame) -> pd.DataFrame:
    rows = dev.to_dict("records")
    rows.append({
        "period": "holdout", "fold": 0, "train_start": 0, "train_end": ISOLATED_START,
        "test_start": ISOLATED_START, "test_end": ONLINE_START,
    })
    rows.extend(holdout.to_dict("records"))
    rows.append({
        "period": "holdout", "fold": int(holdout.fold.max()) + 1,
        "train_start": 0, "train_end": ONLINE_END,
        "test_start": ONLINE_END, "test_end": ISOLATED_END,
    })
    return pd.DataFrame(rows)


def mechanism1_gates(candles: Mapping[str, pd.DataFrame]) -> dict[str, dict[int, bool]]:
    return {
        pair: technical_timeline(candles[pair], TECHNICAL_PARAMS[pair]) for pair in PAIRS
    }


def candidate_key(candidate: Candidate) -> tuple[float, float, float, float, int]:
    return (
        candidate.half_range, candidate.min_spread, candidate.take_profit,
        candidate.move_threshold, candidate.move_cooldown_seconds,
    )


def regenerate_grid_selections(
    candles: Mapping[str, pd.DataFrame], gates: Mapping[str, Mapping[int, bool]],
    folds: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Use only the preceding 14 days and Mechanism 1 for weekly selection."""
    selected_rows, evaluation_rows = [], []
    previous = BASE_CANDIDATE
    pool = candidates()
    for fold in folds.itertuples(index=False):
        training = slice_window(dict(candles), int(fold.train_start), int(fold.train_end))
        ranked = []
        for candidate in pool:
            result, _, pair_stats = simulate(
                training, candidate, maker_fee=0.0, taker_fee=TAKER_FEE,
                technical_buy_gate=dict(gates), risk_breakers_enabled=True,
                cost_floor_enabled=True, record_curve=False,
            )
            pair_stops = int(sum(int(value["liquidations"]) for value in pair_stats.values()))
            eligible = not bool(result["liquidated"]) and pair_stops == 0
            score = float(result["net_pnl_pct"] - 1.5 * abs(result["max_drawdown_pct"]))
            ranked.append((eligible, score, candidate))
            evaluation_rows.append({
                "period": fold.period, "fold": int(fold.fold), **asdict(candidate),
                "score": score, "eligible": eligible,
                "portfolio_stop_events": int(bool(result["liquidated"])),
                "pair_stop_events": pair_stops, **result,
            })
        eligible_rows = [item for item in ranked if item[0]]
        if eligible_rows:
            selected = max(eligible_rows, key=lambda value: value[1])[2]
            retained = False
        else:
            selected = previous
            retained = True
        previous = selected
        selected_rows.append({
            **fold._asdict(), **asdict(selected), "levels": selected.levels,
            "eligible_count": len(eligible_rows), "retained_previous": retained,
        })
    return pd.DataFrame(selected_rows), pd.DataFrame(evaluation_rows)


def feature_columns(architecture: str) -> tuple[str, ...]:
    return ALL_FEATURES if architecture == "shared" else SEPARATE_FEATURES


def fit_predict_block(
    panel: pd.DataFrame, block: Any, seed: int = SEED,
) -> tuple[pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]]]:
    training = panel[panel.target.notna() & (panel.label_ready_ts <= int(block.train_end))]
    testing = panel[
        (panel.signal_ts >= int(block.test_start)) & (panel.signal_ts < int(block.test_end))
    ].copy()
    if training.empty or testing.empty:
        raise RuntimeError(f"Empty model block: {block.period}/{block.fold}")
    if int(training.label_ready_ts.max()) > int(block.train_end):
        raise AssertionError("Six-hour label purge failed")
    base_columns = [
        "pair", "bar_open_utc", "bar_open_ts", "signal_ts", "label_ready_ts",
        "last_complete_1h_ts", "last_complete_4h_ts", "target",
        "future_min_return", "adverse_threshold", "roc_48h_4h", "sqzmom_pct_4h",
        "sqzmom_value_4h", "sqzmom_slope_4h", "sqzmom_improving_4h",
    ]
    prediction_rows, importance_rows = [], []
    for architecture in ARCHITECTURES:
        features = list(feature_columns(architecture))
        groups: Iterable[tuple[str, pd.DataFrame, pd.DataFrame]]
        if architecture == "shared":
            groups = [("ALL", training, testing)]
        else:
            groups = [
                (pair, training[training.pair == pair], testing[testing.pair == pair])
                for pair in PAIRS
            ]
        architecture_output = []
        for model_pair, train_part, test_part in groups:
            weights = compute_sample_weight(
                class_weight="balanced", y=train_part.target.astype(int)
            )
            models = build_models(seed)
            group_output = test_part[base_columns].copy()
            for algorithm, model in models.items():
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model.fit(
                        train_part[features], train_part.target.astype(int),
                        sample_weight=weights,
                    )
                long = group_output.copy()
                long["algorithm"] = algorithm
                long["architecture"] = architecture
                long["variant"] = variant_name(algorithm, architecture)
                long["probability"] = model.predict_proba(test_part[features])[:, 1]
                architecture_output.append(long)
                values = getattr(model, "feature_importances_", None)
                if values is not None:
                    values = np.asarray(values, dtype=float)
                    if values.sum() > 0:
                        values = values / values.sum()
                    importance_rows.extend({
                        "period": block.period, "fold": int(block.fold),
                        "algorithm": algorithm, "architecture": architecture,
                        "variant": variant_name(algorithm, architecture),
                        "model_pair": model_pair, "feature": feature,
                        "importance": float(value),
                    } for feature, value in zip(features, values))
        prediction_rows.extend(architecture_output)
    predictions = pd.concat(prediction_rows, ignore_index=True)
    predictions["period"] = block.period
    predictions["fold"] = int(block.fold)
    audit = [{
        "period": block.period, "fold": int(block.fold),
        "train_cutoff_ts": int(block.train_end), "train_rows": len(training),
        "train_last_signal_ts": int(training.signal_ts.max()),
        "train_last_label_ready_ts": int(training.label_ready_ts.max()),
        "test_rows_per_variant": len(testing),
        "test_first_signal_ts": int(testing.signal_ts.min()),
        "test_last_signal_ts": int(testing.signal_ts.max()),
        "target_rate": float(training.target.mean()),
    }]
    return predictions, importance_rows, audit


def train_predictions(
    panel: pd.DataFrame, blocks: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    predictions, importances, audits = [], [], []
    for block in blocks.itertuples(index=False):
        block_predictions, block_importances, block_audit = fit_predict_block(panel, block)
        predictions.append(block_predictions)
        importances.extend(block_importances)
        audits.extend(block_audit)
    return pd.concat(predictions, ignore_index=True), pd.DataFrame(importances), pd.DataFrame(audits)


def thresholds_from_development(predictions: pd.DataFrame) -> pd.DataFrame:
    dev = predictions[predictions.period == "development"]
    rows = []
    for variant in sorted(dev.variant.unique()):
        for quantile in QUANTILES:
            row = {"variant": variant, "quantile": quantile}
            for pair in PAIRS:
                values = dev[(dev.variant == variant) & (dev.pair == pair)].probability
                row[f"{pair}_threshold"] = float(values.quantile(quantile))
            rows.append(row)
    return pd.DataFrame(rows)


def threshold_map(row: Any) -> dict[str, float]:
    if isinstance(row, Mapping) or isinstance(row, pd.Series):
        return {pair: float(row[f"{pair}_threshold"]) for pair in PAIRS}
    return {pair: float(getattr(row, f"{pair}_threshold")) for pair in PAIRS}


def recovery_condition(row: Any, pair: str) -> bool:
    params = TECHNICAL_PARAMS[pair]
    return bool(
        float(row.roc_48h_4h) >= params.roc_recovery_pct
        and float(row.sqzmom_pct_4h) >= params.sqz_recovery_pct
        and bool(row.sqzmom_improving_4h)
    )


def build_risk_timeline(
    predictions: pd.DataFrame, variant: str, thresholds: Mapping[str, float],
    start_ts: int, end_ts: int,
) -> tuple[dict[str, dict[int, float]], pd.DataFrame, pd.DataFrame]:
    """Create independent pair states; Mechanism 1 is used only for recovery."""
    timeline: dict[str, dict[int, float]] = {pair: {} for pair in PAIRS}
    state_rows, event_rows = [], []
    selected = predictions[
        (predictions.variant == variant)
        & (predictions.signal_ts >= start_ts) & (predictions.signal_ts < end_ts)
    ]
    algorithm, architecture = variant.rsplit(" | ", 1)
    for pair in PAIRS:
        pair_rows = selected[selected.pair == pair].sort_values("signal_ts")
        active = False
        records = list(pair_rows.itertuples(index=False))
        for index, row in enumerate(records):
            signal_ts = int(row.signal_ts)
            recovery_met = recovery_condition(row, pair)
            previous = active
            signal = advance_pair_state(
                pair=pair, probability=float(row.probability),
                entry_threshold=float(thresholds[pair]), previous_risk_off=active,
                recovery_condition_met=recovery_met, signal_ts=signal_ts,
                last_complete_1h_ts=int(row.last_complete_1h_ts),
                last_complete_4h_ts=int(row.last_complete_4h_ts),
                model_version=variant,
                recovery_details={
                    "roc_48h_pct": float(row.roc_48h_4h),
                    "sqzmom_pct": float(row.sqzmom_pct_4h),
                    "sqzmom_improving": bool(row.sqzmom_improving_4h),
                    "roc_recovery_threshold_pct": TECHNICAL_PARAMS[pair].roc_recovery_pct,
                    "sqzmom_recovery_threshold_pct": TECHNICAL_PARAMS[pair].sqz_recovery_pct,
                },
            )
            active = bool(signal["risk_off_active"])
            right = min(
                int(records[index + 1].signal_ts) if index + 1 < len(records) else end_ts,
                end_ts,
            )
            if active:
                for timestamp in range(max(signal_ts, start_ts), right, FIVE_MINUTES):
                    timeline[pair][timestamp] = float(row.probability)
            transition = "enter" if not previous and active else "recover" if previous and not active else "hold" if active else "clear"
            state_rows.append({
                "variant": variant, "algorithm": algorithm, "architecture": architecture,
                "pair": pair, "signal_ts": signal_ts, "probability": float(row.probability),
                "entry_threshold": float(thresholds[pair]), "risk_off_active": active,
                "recovery_condition_met": recovery_met, "transition": transition,
                "last_complete_1h_ts": int(row.last_complete_1h_ts),
                "last_complete_4h_ts": int(row.last_complete_4h_ts),
                "roc_48h_pct": float(row.roc_48h_4h),
                "sqzmom_pct": float(row.sqzmom_pct_4h),
                "sqzmom_improving": bool(row.sqzmom_improving_4h),
                "event_id": signal["event_id"],
            })
            if transition in {"enter", "recover"}:
                event_rows.append({
                    "variant": variant, "timestamp": signal_ts, "pair": pair,
                    "side": "PAUSE" if transition == "enter" else "RESUME",
                    "reason": signal["reason"], "probability": float(row.probability),
                    "entry_threshold": float(thresholds[pair]),
                    "stop_excess_inventory": bool(signal["stop_excess_inventory"]),
                    "event_id": signal["event_id"],
                })
    return timeline, pd.DataFrame(state_rows), pd.DataFrame(event_rows)


def candidate_from_row(row: Any) -> Candidate:
    return Candidate(
        float(row.half_range), float(row.min_spread), float(row.take_profit),
        float(row.move_threshold), int(row.move_cooldown_seconds),
    )


def simulate_one(
    candles: Mapping[str, pd.DataFrame], start_ts: int, end_ts: int, candidate: Candidate,
    *, gates: Mapping[str, Mapping[int, bool]] | None = None,
    timeline: Mapping[str, Mapping[int, float]] | None = None,
    risk_breakers_enabled: bool, cost_floor_enabled: bool,
    taker_fee: float = TAKER_FEE, slippage: float = 0.0,
    record_details: bool = False,
) -> tuple[dict[str, Any], pd.DataFrame, dict[str, dict[str, Any]], pd.DataFrame]:
    trade_log: list[dict[str, Any]] | None = [] if record_details else None
    result, curve, pairs = simulate(
        slice_window(dict(candles), start_ts, end_ts), candidate,
        maker_fee=0.0, taker_fee=taker_fee, slippage=slippage,
        order_refresh_seconds=7_200,
        technical_buy_gate=dict(gates) if gates is not None else None,
        momentum_stop_timeline=dict(timeline) if timeline is not None else None,
        momentum_stop_threshold=0.5, trade_log=trade_log,
        risk_breakers_enabled=risk_breakers_enabled,
        cost_floor_enabled=cost_floor_enabled, inventory_exit_policy=None,
        record_curve=record_details,
    )
    return result, curve, pairs, pd.DataFrame(trade_log or [])


def replay_weekly(
    candles: Mapping[str, pd.DataFrame], selections: pd.DataFrame,
    *, scenario: str, track: str,
    gates: Mapping[str, Mapping[int, bool]] | None = None,
    predictions: pd.DataFrame | None = None, variant: str | None = None,
    thresholds: Mapping[str, float] | None = None,
    record_details: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    weekly_rows, pair_rows, curves, events, states = [], [], [], [], []
    cumulative_pnl = 0.0
    for selection in selections.itertuples(index=False):
        start_ts, end_ts = int(selection.test_start), int(selection.test_end)
        timeline = None
        if variant is not None:
            if predictions is None or thresholds is None:
                raise ValueError("ML weekly replay requires predictions and thresholds")
            fold_predictions = predictions[
                (predictions.period == selection.period) & (predictions.fold == int(selection.fold))
            ]
            timeline, fold_states, state_events = build_risk_timeline(
                fold_predictions, variant, thresholds, start_ts, end_ts
            )
            if not fold_states.empty:
                fold_states["track"] = track
                fold_states["period"] = selection.period
                fold_states["fold"] = int(selection.fold)
                states.append(fold_states)
            if not state_events.empty:
                state_events["track"] = track
                state_events["period"] = selection.period
                state_events["fold"] = int(selection.fold)
                events.append(state_events)
        result, curve, pair_stats, trade_events = simulate_one(
            candles, start_ts, end_ts, candidate_from_row(selection),
            gates=gates if variant is None else None, timeline=timeline,
            risk_breakers_enabled=(track == "online"),
            cost_floor_enabled=(track == "online"), record_details=record_details,
        )
        weekly_rows.append({
            "scenario": scenario, "track": track, "period": selection.period,
            "fold": int(selection.fold), "test_start": start_ts, "test_end": end_ts,
            **asdict(candidate_from_row(selection)), **result,
        })
        pair_rows.extend({
            "scenario": scenario, "track": track, "period": selection.period,
            "fold": int(selection.fold), "pair": pair, **metrics,
        } for pair, metrics in pair_stats.items())
        if record_details and not curve.empty:
            item = curve.copy()
            item["scenario"] = scenario
            item["track"] = track
            item["period"] = selection.period
            item["fold"] = int(selection.fold)
            item["cumulative_oos_pnl"] = cumulative_pnl + item.equity - INITIAL_EQUITY
            curves.append(item)
        if not trade_events.empty:
            trade_events["scenario"] = scenario
            trade_events["track"] = track
            trade_events["period"] = selection.period
            trade_events["fold"] = int(selection.fold)
            events.append(trade_events)
        cumulative_pnl += float(result["net_pnl_quote"])
    return (
        pd.DataFrame(weekly_rows), pd.DataFrame(pair_rows),
        pd.concat(curves, ignore_index=True) if curves else pd.DataFrame(),
        pd.concat(events, ignore_index=True) if events else pd.DataFrame(),
        pd.concat(states, ignore_index=True) if states else pd.DataFrame(),
    )


def replay_isolated_continuous(
    candles: Mapping[str, pd.DataFrame], *, scenario: str,
    gates: Mapping[str, Mapping[int, bool]] | None = None,
    predictions: pd.DataFrame | None = None, variant: str | None = None,
    thresholds: Mapping[str, float] | None = None,
    record_details: bool = False,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    timeline, states, state_events = None, pd.DataFrame(), pd.DataFrame()
    if variant is not None:
        if predictions is None or thresholds is None:
            raise ValueError("ML isolated replay requires predictions and thresholds")
        timeline, states, state_events = build_risk_timeline(
            predictions[predictions.period == "holdout"], variant, thresholds,
            ISOLATED_START, ISOLATED_END,
        )
    result, curve, pairs, trade_events = simulate_one(
        candles, ISOLATED_START, ISOLATED_END, BASE_CANDIDATE,
        gates=gates if variant is None else None, timeline=timeline,
        risk_breakers_enabled=False, cost_floor_enabled=False,
        record_details=record_details,
    )
    if not curve.empty:
        curve["scenario"] = scenario
        curve["track"] = "isolated"
        curve["cumulative_oos_pnl"] = curve.equity - INITIAL_EQUITY
    pair_frame = pd.DataFrame([
        {"scenario": scenario, "track": "isolated", "pair": pair, **metrics}
        for pair, metrics in pairs.items()
    ])
    if not trade_events.empty:
        trade_events["scenario"] = scenario
        trade_events["track"] = "isolated"
    if not state_events.empty:
        state_events["scenario"] = scenario
        state_events["track"] = "isolated"
        trade_events = pd.concat([state_events, trade_events], ignore_index=True, sort=False)
    if not states.empty:
        states["scenario"] = scenario
        states["track"] = "isolated"
    return result, curve, pair_frame, trade_events, states


def summarize_weekly(weekly: pd.DataFrame, pairs: pd.DataFrame) -> dict[str, Any]:
    returns = weekly.net_pnl_quote / INITIAL_EQUITY
    std = float(returns.std(ddof=0))
    return {
        "pnl_fdusd": float(weekly.net_pnl_quote.sum()),
        "return_pct": float(weekly.net_pnl_quote.sum() / (INITIAL_EQUITY * len(weekly)) * 100),
        "max_drawdown_pct": float(weekly.max_drawdown_pct.min() * 100),
        "positive_folds": int((weekly.net_pnl_quote > 0).sum()),
        "portfolio_stop_events": int(weekly.liquidated.astype(bool).sum()),
        "pair_stop_events": int(pairs.liquidations.sum()),
        "risk_off_pair_hours": float(pairs.momentum_risk_off_seconds.sum() / 3600),
        "momentum_stop_exits": int(weekly.momentum_stop_exits.sum()),
        "weekly_sharpe": float(returns.mean() / std * math.sqrt(52)) if std else 0.0,
        "folds": len(weekly),
    }


def summarize_continuous(result: Mapping[str, Any], pairs: pd.DataFrame) -> dict[str, Any]:
    return {
        "pnl_fdusd": float(result["net_pnl_quote"]),
        "return_pct": float(result["net_pnl_pct"] * 100),
        "max_drawdown_pct": float(result["max_drawdown_pct"] * 100),
        "portfolio_stop_events": int(bool(result["liquidated"])),
        "pair_stop_events": int(pairs.liquidations.sum()),
        "risk_off_pair_hours": float(pairs.momentum_risk_off_seconds.sum() / 3600),
        "momentum_stop_exits": int(result["momentum_stop_exits"]),
        "folds": 1,
    }


def isolated_dev_selections(dev: pd.DataFrame) -> pd.DataFrame:
    out = dev.copy()
    for key, value in asdict(BASE_CANDIDATE).items():
        out[key] = value
    out["levels"] = BASE_CANDIDATE.levels
    return out


def development_selection(
    candles: Mapping[str, pd.DataFrame], gates: Mapping[str, Mapping[int, bool]],
    dev_grid: pd.DataFrame, dev_predictions: pd.DataFrame, thresholds: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, Any]]:
    iso_grid = isolated_dev_selections(dev_grid[
        ["period", "fold", "train_start", "train_end", "test_start", "test_end"]
    ])
    base_iso_weekly, base_iso_pairs, *_ = replay_weekly(
        candles, iso_grid, scenario="Mechanism 1", track="isolated", gates=gates
    )
    base_online_weekly, base_online_pairs, *_ = replay_weekly(
        candles, dev_grid, scenario="Mechanism 1", track="online", gates=gates
    )
    baseline_iso = summarize_weekly(base_iso_weekly, base_iso_pairs)
    baseline_online = summarize_weekly(base_online_weekly, base_online_pairs)
    rows = []
    for _, setting in thresholds.iterrows():
        variant = str(setting["variant"])
        pair_thresholds = threshold_map(setting)
        iso_weekly, iso_pairs, *_ = replay_weekly(
            candles, iso_grid, scenario=variant, track="isolated",
            predictions=dev_predictions, variant=variant, thresholds=pair_thresholds,
        )
        online_weekly, online_pairs, *_ = replay_weekly(
            candles, dev_grid, scenario=variant, track="online",
            predictions=dev_predictions, variant=variant, thresholds=pair_thresholds,
        )
        iso = summarize_weekly(iso_weekly, iso_pairs)
        online = summarize_weekly(online_weekly, online_pairs)
        eligible = bool(
            iso["return_pct"] > 0 and online["return_pct"] > 0
            and iso["max_drawdown_pct"] >= baseline_iso["max_drawdown_pct"]
            and online["max_drawdown_pct"] >= baseline_online["max_drawdown_pct"]
            and online["portfolio_stop_events"] <= baseline_online["portfolio_stop_events"]
        )
        row = {
            "variant": variant, "algorithm": variant.rsplit(" | ", 1)[0],
            "architecture": variant.rsplit(" | ", 1)[1],
            "quantile": float(setting["quantile"]),
            "BTC-FDUSD_threshold": pair_thresholds["BTC-FDUSD"],
            "ETH-FDUSD_threshold": pair_thresholds["ETH-FDUSD"],
            "eligible": eligible,
        }
        for track, metrics in (("isolated", iso), ("online", online)):
            row.update({f"{track}_{key}": value for key, value in metrics.items()})
            row[f"{track}_safety_burden"] = (
                metrics["risk_off_pair_hours"] + 168 * metrics["pair_stop_events"]
                + 336 * metrics["portfolio_stop_events"]
            )
        rows.append(row)
    table = pd.DataFrame(rows)
    for track in ("isolated", "online"):
        table[f"{track}_return_percentile"] = table[f"{track}_return_pct"].rank(pct=True)
        table[f"{track}_drawdown_percentile"] = table[f"{track}_max_drawdown_pct"].rank(pct=True)
        table[f"{track}_safety_percentile"] = 1 - table[f"{track}_safety_burden"].rank(pct=True) + 1 / len(table)
        table[f"{track}_score"] = (
            0.5 * table[f"{track}_return_percentile"]
            + 0.3 * table[f"{track}_drawdown_percentile"]
            + 0.2 * table[f"{track}_safety_percentile"]
        )
    table["joint_score"] = table[["isolated_score", "online_score"]].min(axis=1)
    table["average_score"] = table[["isolated_score", "online_score"]].mean(axis=1)
    table["total_stop_events"] = (
        table.isolated_pair_stop_events + table.online_pair_stop_events
        + table.isolated_portfolio_stop_events + table.online_portfolio_stop_events
    )
    table["combined_return_pct"] = table.isolated_return_pct + table.online_return_pct
    table = table.sort_values(
        ["eligible", "joint_score", "average_score", "total_stop_events", "combined_return_pct"],
        ascending=[False, False, False, True, False],
    ).reset_index(drop=True)
    table["global_rank"] = np.arange(1, len(table) + 1)

    locked_rows = []
    for variant, group in table.groupby("variant", sort=False):
        pool = group[group.eligible]
        winner = (pool if not pool.empty else group).sort_values(
            ["joint_score", "average_score", "total_stop_events", "combined_return_pct"],
            ascending=[False, False, True, False],
        ).iloc[0].to_dict()
        winner["variant_has_eligible_candidate"] = bool(not pool.empty)
        locked_rows.append(winner)
    locked = pd.DataFrame(locked_rows).sort_values(
        ["eligible", "joint_score", "average_score", "total_stop_events", "combined_return_pct"],
        ascending=[False, False, False, True, False],
    ).reset_index(drop=True)
    locked["locked_rank"] = np.arange(1, len(locked) + 1)
    return table, locked, baseline_iso, baseline_online


def classification_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    holdout = predictions[(predictions.period == "holdout") & predictions.target.notna()]
    for variant in sorted(holdout.variant.unique()):
        for pair_scope, data in [("ALL", holdout[holdout.variant == variant])]:
            y = data.target.astype(int)
            probability = data.probability.clip(1e-8, 1 - 1e-8)
            rows.append({
                "variant": variant, "pair": pair_scope, "rows": len(data),
                "positive_rate": float(y.mean()),
                "roc_auc": float(roc_auc_score(y, probability)),
                "log_loss": float(log_loss(y, probability, labels=[0, 1])),
                "brier_score": float(brier_score_loss(y, probability)),
                "balanced_accuracy_0_5": float(balanced_accuracy_score(y, probability >= 0.5)),
            })
        for pair in PAIRS:
            data = holdout[(holdout.variant == variant) & (holdout.pair == pair)]
            y = data.target.astype(int)
            probability = data.probability.clip(1e-8, 1 - 1e-8)
            rows.append({
                "variant": variant, "pair": pair, "rows": len(data),
                "positive_rate": float(y.mean()),
                "roc_auc": float(roc_auc_score(y, probability)),
                "log_loss": float(log_loss(y, probability, labels=[0, 1])),
                "brier_score": float(brier_score_loss(y, probability)),
                "balanced_accuracy_0_5": float(balanced_accuracy_score(y, probability >= 0.5)),
            })
    return pd.DataFrame(rows)


def holdout_evaluation(
    candles: Mapping[str, pd.DataFrame], gates: Mapping[str, Mapping[int, bool]],
    online_grid: pd.DataFrame, predictions: pd.DataFrame, locked: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base_iso_result, base_iso_curve, base_iso_pairs, base_iso_events, _ = replay_isolated_continuous(
        candles, scenario="Mechanism 1", gates=gates, record_details=True
    )
    base_iso = summarize_continuous(base_iso_result, base_iso_pairs)
    if abs(base_iso["return_pct"] - LOCKED_ISOLATED_RETURN_PCT) > 1e-6:
        raise AssertionError(f"Mechanism 1 return parity failed: {base_iso['return_pct']}")
    if abs(base_iso["max_drawdown_pct"] - LOCKED_ISOLATED_DD_PCT) > 1e-6:
        raise AssertionError(f"Mechanism 1 drawdown parity failed: {base_iso['max_drawdown_pct']}")

    base_online_weekly, base_online_pairs, base_online_curve, base_online_events, _ = replay_weekly(
        candles, online_grid, scenario="Mechanism 1", track="online",
        gates=gates, record_details=True,
    )
    base_online = summarize_weekly(base_online_weekly, base_online_pairs)
    metric_rows = []
    curves = [base_iso_curve, base_online_curve]
    weekly_frames = [base_online_weekly]
    event_frames = [base_iso_events, base_online_events]
    state_frames = []
    pair_frames = [base_iso_pairs, base_online_pairs]
    for _, setting in locked.iterrows():
        variant = str(setting["variant"])
        thresholds = threshold_map(setting)
        iso_result, iso_curve, iso_pairs, iso_events, iso_states = replay_isolated_continuous(
            candles, scenario=variant, predictions=predictions, variant=variant,
            thresholds=thresholds, record_details=True,
        )
        iso = summarize_continuous(iso_result, iso_pairs)
        online_weekly, online_pairs, online_curve, online_events, online_states = replay_weekly(
            candles, online_grid, scenario=variant, track="online",
            predictions=predictions, variant=variant, thresholds=thresholds,
            record_details=True,
        )
        online = summarize_weekly(online_weekly, online_pairs)
        isolated_pass = bool(
            iso["return_pct"] > LOCKED_ISOLATED_RETURN_PCT
            and iso["max_drawdown_pct"] >= LOCKED_ISOLATED_DD_PCT
        )
        online_pass = bool(
            online["return_pct"] > 0
            and online["return_pct"] >= base_online["return_pct"]
            and online["max_drawdown_pct"] >= base_online["max_drawdown_pct"]
            and online["portfolio_stop_events"] == 0
            and online["pair_stop_events"] <= base_online["pair_stop_events"]
        )
        metric_rows.append({
            "variant": variant, "algorithm": setting["algorithm"],
            "architecture": setting["architecture"], "locked_quantile": float(setting["quantile"]),
            "development_eligible": bool(setting["eligible"]),
            "development_joint_score": float(setting["joint_score"]),
            "BTC-FDUSD_threshold": thresholds["BTC-FDUSD"],
            "ETH-FDUSD_threshold": thresholds["ETH-FDUSD"],
            **{f"isolated_{key}": value for key, value in iso.items()},
            **{f"online_{key}": value for key, value in online.items()},
            "isolated_success": isolated_pass, "online_success": online_pass,
            "joint_holdout_success": isolated_pass and online_pass,
        })
        curves.extend([iso_curve, online_curve])
        weekly_frames.append(online_weekly)
        event_frames.extend([iso_events, online_events])
        state_frames.extend([iso_states, online_states])
        pair_frames.extend([iso_pairs, online_pairs])
    metrics = pd.DataFrame(metric_rows).sort_values(
        ["joint_holdout_success", "isolated_return_pct", "online_return_pct"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    metrics["holdout_rank"] = np.arange(1, len(metrics) + 1)
    return (
        metrics, base_iso, base_online,
        pd.concat(curves, ignore_index=True, sort=False),
        pd.concat(weekly_frames, ignore_index=True, sort=False),
        pd.concat(event_frames, ignore_index=True, sort=False),
        pd.concat(state_frames, ignore_index=True, sort=False) if state_frames else pd.DataFrame(),
        pd.concat(pair_frames, ignore_index=True, sort=False),
    )


def curve_weekly_blocks(curve: pd.DataFrame, start_ts: int, end_ts: int) -> pd.DataFrame:
    source = curve[(curve.timestamp >= start_ts) & (curve.timestamp < end_ts)].copy()
    source = source.sort_values("timestamp")
    source["equity_change"] = source.equity.diff()
    if not source.empty:
        source.loc[source.index[0], "equity_change"] = float(source.equity.iloc[0]) - INITIAL_EQUITY
    source["block"] = ((source.timestamp - start_ts) // (7 * DAY)).astype(int) + 1
    rows = []
    for block, item in source.groupby("block", sort=True):
        if item.empty:
            continue
        local_dd = float((item.equity / item.equity.cummax() - 1).min() * 100)
        rows.append({
            "block": int(block), "pnl_fdusd": float(item.equity_change.sum()),
            "drawdown_pct": local_dd,
        })
    return pd.DataFrame(rows)


def paired_block_bootstrap(
    model: pd.DataFrame, baseline: pd.DataFrame, *, weekly_reset: bool,
    samples: int = 10_000, seed: int = SEED,
) -> dict[str, Any]:
    merged = model.merge(baseline, on="block", suffixes=("_model", "_baseline"))
    pnl_delta = (merged.pnl_fdusd_model - merged.pnl_fdusd_baseline).to_numpy(float)
    dd_delta = (merged.drawdown_pct_model - merged.drawdown_pct_baseline).to_numpy(float)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(merged), size=(samples, len(merged)))
    sampled_pnl = pnl_delta[indices]
    return_samples = (
        sampled_pnl.mean(axis=1) if weekly_reset else sampled_pnl.sum(axis=1)
    ) / INITIAL_EQUITY * 100
    dd_samples = dd_delta[indices].mean(axis=1)
    observed_return_delta = (
        pnl_delta.mean() if weekly_reset else pnl_delta.sum()
    ) / INITIAL_EQUITY * 100
    return {
        "blocks": len(merged), "samples": samples,
        "return_difference_pct": float(observed_return_delta),
        "return_difference_95ci_pct": [float(x) for x in np.quantile(return_samples, [0.025, 0.975])],
        "mean_block_drawdown_difference_pct": float(dd_delta.mean()),
        "mean_block_drawdown_difference_95ci_pct": [float(x) for x in np.quantile(dd_samples, [0.025, 0.975])],
        "return_ci_crosses_zero": bool(np.quantile(return_samples, 0.025) <= 0 <= np.quantile(return_samples, 0.975)),
        "short_sample_warning": len(merged) < 20,
    }


def bootstrap_final(
    curves: pd.DataFrame, weekly: pd.DataFrame, final_variant: str,
) -> dict[str, Any]:
    base_iso = curve_weekly_blocks(
        curves[(curves.track == "isolated") & (curves.scenario == "Mechanism 1")],
        ISOLATED_START, ISOLATED_END,
    )
    model_iso = curve_weekly_blocks(
        curves[(curves.track == "isolated") & (curves.scenario == final_variant)],
        ISOLATED_START, ISOLATED_END,
    )
    base_online = weekly[weekly.scenario == "Mechanism 1"].copy()
    model_online = weekly[weekly.scenario == final_variant].copy()
    base_online = base_online.rename(columns={"fold": "block", "max_drawdown_pct": "drawdown_raw"})
    model_online = model_online.rename(columns={"fold": "block", "max_drawdown_pct": "drawdown_raw"})
    base_online["drawdown_pct"] = base_online.drawdown_raw * 100
    model_online["drawdown_pct"] = model_online.drawdown_raw * 100
    return {
        "method": "paired weekly block bootstrap with 10,000 seeded resamples",
        "isolated": paired_block_bootstrap(model_iso, base_iso, weekly_reset=False),
        "online": paired_block_bootstrap(
            model_online[["block", "net_pnl_quote", "drawdown_pct"]].rename(columns={"net_pnl_quote": "pnl_fdusd"}),
            base_online[["block", "net_pnl_quote", "drawdown_pct"]].rename(columns={"net_pnl_quote": "pnl_fdusd"}),
            weekly_reset=True,
        ),
    }


def refit_variant_predict(
    training_panel: pd.DataFrame, testing_panel: pd.DataFrame,
    variant: str, cutoff: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    algorithm, architecture = variant.rsplit(" | ", 1)
    features = list(feature_columns(architecture))
    train = training_panel[
        training_panel.target.notna() & (training_panel.label_ready_ts <= cutoff)
    ]
    test = testing_panel[
        (testing_panel.signal_ts >= cutoff) & (testing_panel.signal_ts < ISOLATED_END)
    ]
    outputs, fitted = [], {}
    groups = [("ALL", train, test)] if architecture == "shared" else [
        (pair, train[train.pair == pair], test[test.pair == pair]) for pair in PAIRS
    ]
    for model_pair, train_part, test_part in groups:
        weights = compute_sample_weight(class_weight="balanced", y=train_part.target.astype(int))
        model = build_models(SEED)[algorithm]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(train_part[features], train_part.target.astype(int), sample_weight=weights)
        item = test_part[[
            "pair", "bar_open_utc", "bar_open_ts", "signal_ts", "label_ready_ts",
            "last_complete_1h_ts", "last_complete_4h_ts", "target", "future_min_return",
            "adverse_threshold", "roc_48h_4h", "sqzmom_pct_4h", "sqzmom_value_4h",
            "sqzmom_slope_4h", "sqzmom_improving_4h",
        ]].copy()
        item["algorithm"], item["architecture"], item["variant"] = algorithm, architecture, variant
        item["probability"] = model.predict_proba(test_part[features])[:, 1]
        outputs.append(item)
        fitted[model_pair] = model
    return pd.concat(outputs, ignore_index=True), fitted


def stress_test(
    candles: Mapping[str, pd.DataFrame], panel: pd.DataFrame, final_variant: str,
    thresholds: Mapping[str, float], final_candidate: Candidate,
) -> tuple[pd.DataFrame, dict[str, Any], bytes]:
    stress_start = ISOLATED_END - 7 * DAY
    cutoffs = [ONLINE_END - 7 * DAY, ONLINE_END]
    actual_predictions, crash_predictions, latest_models = [], [], {}
    stressed_candles = crash_candles(dict(candles), drop=0.15)
    stressed_panel = build_feature_panel(stressed_candles)
    for cutoff in cutoffs:
        actual, models = refit_variant_predict(panel, panel, final_variant, cutoff)
        crashed, _ = refit_variant_predict(panel, stressed_panel, final_variant, cutoff)
        next_cutoff = ONLINE_END if cutoff < ONLINE_END else ISOLATED_END
        actual_predictions.append(actual[(actual.signal_ts >= max(cutoff, stress_start)) & (actual.signal_ts < next_cutoff)])
        crash_predictions.append(crashed[(crashed.signal_ts >= max(cutoff, stress_start)) & (crashed.signal_ts < next_cutoff)])
        latest_models = models
    actual_predictions = pd.concat(actual_predictions, ignore_index=True)
    crash_predictions = pd.concat(crash_predictions, ignore_index=True)
    scenarios = [
        ("base_taker_fee", dict(candles), actual_predictions, TAKER_FEE, 0.0),
        ("taker_fee_150pct", dict(candles), actual_predictions, TAKER_FEE * 1.5, 0.0),
        ("slippage_0.05pct", dict(candles), actual_predictions, TAKER_FEE, 0.0005),
        ("slippage_0.10pct", dict(candles), actual_predictions, TAKER_FEE, 0.0010),
        ("one_day_15pct_crash", stressed_candles, crash_predictions, TAKER_FEE, 0.0),
    ]
    rows = []
    for name, scenario_candles, scenario_predictions, fee, slippage in scenarios:
        timeline, _, _ = build_risk_timeline(
            scenario_predictions, final_variant, thresholds, stress_start, ISOLATED_END
        )
        result, _, pairs, _ = simulate_one(
            scenario_candles, stress_start, ISOLATED_END, final_candidate,
            timeline=timeline, risk_breakers_enabled=True, cost_floor_enabled=True,
            taker_fee=fee, slippage=slippage,
        )
        pair_stops = int(sum(int(value["liquidations"]) for value in pairs.values()))
        rows.append({
            "scenario": name, "return_pct": float(result["net_pnl_pct"] * 100),
            "max_drawdown_pct": float(result["max_drawdown_pct"] * 100),
            "portfolio_stop_events": int(bool(result["liquidated"])),
            "pair_stop_events": pair_stops,
            "momentum_stop_exits": int(result["momentum_stop_exits"]),
            "stress_gate_pass": not bool(result["liquidated"]) and pair_stops == 0,
        })
    blob = pickle.dumps({
        "variant": final_variant, "features": list(feature_columns(final_variant.rsplit(" | ", 1)[1])),
        "models": latest_models,
    }, protocol=pickle.HIGHEST_PROTOCOL)
    return pd.DataFrame(rows), {"stressed_panel_rows": len(stressed_panel)}, blob


def final_signal_contract(
    states: pd.DataFrame, final_variant: str, thresholds: Mapping[str, float],
    model_blob: bytes,
) -> dict[str, Any]:
    version = f"grid-ml-momentum-stop-v1:{final_variant.replace(' ', '_')}:seed42"
    signals = {}
    generated_at = 0
    for pair in PAIRS:
        history = states[
            (states.track == "isolated") & (states.scenario == final_variant)
            & (states.pair == pair)
        ].sort_values("signal_ts")
        if history.empty:
            raise RuntimeError(f"No final signal history for {pair}")
        current = history.iloc[-1]
        previous_active = bool(history.iloc[-2].risk_off_active) if len(history) > 1 else False
        signal = advance_pair_state(
            pair=pair, probability=float(current.probability),
            entry_threshold=float(thresholds[pair]), previous_risk_off=previous_active,
            recovery_condition_met=bool(current.recovery_condition_met),
            signal_ts=int(current.signal_ts),
            last_complete_1h_ts=int(current.last_complete_1h_ts),
            last_complete_4h_ts=int(current.last_complete_4h_ts),
            model_version=version,
            recovery_details={
                "roc_48h_pct": float(current.roc_48h_pct),
                "sqzmom_pct": float(current.sqzmom_pct),
                "sqzmom_improving": bool(current.sqzmom_improving),
                "roc_recovery_threshold_pct": TECHNICAL_PARAMS[pair].roc_recovery_pct,
                "sqzmom_recovery_threshold_pct": TECHNICAL_PARAMS[pair].sqz_recovery_pct,
            },
        )
        signals[pair] = signal
        generated_at = max(generated_at, int(current.signal_ts))
    return build_contract(
        pair_signals=signals, generated_at=generated_at, valid_until=generated_at + 150,
        model_version=version, model_sha256=hashlib.sha256(model_blob).hexdigest(),
        feature_schema_sha256=feature_schema_hash(list(feature_columns(final_variant.rsplit(" | ", 1)[1]))),
        source_healthy=True,
    )


def create_report(
    path: Path, *, summary: Mapping[str, Any], holdout: pd.DataFrame,
    curves: pd.DataFrame, weekly: pd.DataFrame, states: pd.DataFrame, events: pd.DataFrame,
    classifications: pd.DataFrame, importance: pd.DataFrame,
    final_variant: str, stress: pd.DataFrame,
) -> None:
    fig = make_subplots(
        rows=4, cols=2,
        subplot_titles=(
            "隔离轨累计权益", "完整线上轨累计样本外盈亏",
            "逐周盈亏（线上轨）", "回撤",
            "BTC/ETH 风险概率与风险切换", "10种锁定模型排行榜",
            "最佳模型特征重要性", "样本外分类 ROC AUC",
        ),
        vertical_spacing=0.09, horizontal_spacing=0.08,
    )
    colors = {"Mechanism 1": "#111827", final_variant: "#dc2626"}
    for scenario in ("Mechanism 1", final_variant):
        item = curves[(curves.track == "isolated") & (curves.scenario == scenario)].copy()
        if not item.empty:
            fig.add_trace(go.Scatter(
                x=pd.to_datetime(item.timestamp, unit="s", utc=True), y=item.equity,
                name=scenario, legendgroup=scenario, mode="lines",
                line=dict(color=colors[scenario], width=2.4),
            ), row=1, col=1)
        online = curves[(curves.track == "online") & (curves.scenario == scenario)].copy()
        if not online.empty:
            fig.add_trace(go.Scatter(
                x=pd.to_datetime(online.timestamp, unit="s", utc=True),
                y=online.cumulative_oos_pnl, name=scenario, legendgroup=scenario,
                showlegend=False, mode="lines", line=dict(color=colors[scenario], width=2.2),
            ), row=1, col=2)
            fig.add_trace(go.Scatter(
                x=pd.to_datetime(online.timestamp, unit="s", utc=True),
                y=online.drawdown_pct * 100, name=f"{scenario} DD",
                legendgroup=scenario, showlegend=False, mode="lines",
                line=dict(color=colors[scenario], width=1.7),
            ), row=2, col=2)
    weekly_plot = weekly[weekly.scenario.isin(["Mechanism 1", final_variant])].copy()
    for scenario in ("Mechanism 1", final_variant):
        item = weekly_plot[weekly_plot.scenario == scenario]
        fig.add_trace(go.Bar(
            x=[f"W{int(value)}" for value in item.fold], y=item.net_pnl_quote,
            name=scenario, legendgroup=scenario, showlegend=False,
            marker_color=colors[scenario], opacity=0.8,
        ), row=2, col=1)

    risk = states[(states.track == "isolated") & (states.scenario == final_variant)]
    pair_colors = {"BTC-FDUSD": "#2563eb", "ETH-FDUSD": "#7c3aed"}
    for pair in PAIRS:
        item = risk[risk.pair == pair]
        fig.add_trace(go.Scatter(
            x=pd.to_datetime(item.signal_ts, unit="s", utc=True),
            y=np.where(item.risk_off_active.astype(bool), 1.0, np.nan),
            name=f"{pair} risk region", mode="lines", fill="tozeroy",
            fillcolor="rgba(239,68,68,0.08)", line=dict(width=0), connectgaps=False,
            hoverinfo="skip",
        ), row=3, col=1)
        fig.add_trace(go.Scatter(
            x=pd.to_datetime(item.signal_ts, unit="s", utc=True), y=item.probability,
            name=f"{pair} probability", mode="lines", line=dict(color=pair_colors[pair], width=1.5),
        ), row=3, col=1)
        fig.add_trace(go.Scatter(
            x=pd.to_datetime(item.signal_ts, unit="s", utc=True), y=item.entry_threshold,
            name=f"{pair} threshold", mode="lines", line=dict(color=pair_colors[pair], dash="dash", width=1),
        ), row=3, col=1)
        entered = item[item.transition == "enter"]
        fig.add_trace(go.Scatter(
            x=pd.to_datetime(entered.signal_ts, unit="s", utc=True), y=entered.probability,
            name=f"{pair} risk-off", mode="markers",
            marker=dict(color=pair_colors[pair], symbol="x", size=9),
        ), row=3, col=1)
    actual_exits = events[
        (events.track == "isolated") & (events.scenario == final_variant)
        & (events.reason == "momentum_stop_exit")
    ] if "reason" in events else pd.DataFrame()
    if not actual_exits.empty:
        fig.add_trace(go.Scatter(
            x=pd.to_datetime(actual_exits.timestamp, unit="s", utc=True),
            y=np.full(len(actual_exits), 0.98), name="actual excess-inventory stop",
            mode="markers", marker=dict(color="#111827", symbol="triangle-down", size=8),
            text=actual_exits.pair,
            hovertemplate="%{text}<br>%{x}<br>actual excess-inventory Taker stop<extra></extra>",
        ), row=3, col=1)

    ranking = holdout.sort_values("development_joint_score")
    fig.add_trace(go.Bar(
        x=ranking.development_joint_score, y=ranking.variant, orientation="h",
        name="开发集联合分", marker_color=["#dc2626" if value == final_variant else "#94a3b8" for value in ranking.variant],
        showlegend=False, text=ranking.holdout_rank, textposition="auto",
        customdata=np.stack([ranking.isolated_return_pct, ranking.online_return_pct], axis=-1),
        hovertemplate="%{y}<br>开发联合分 %{x:.3f}<br>隔离收益 %{customdata[0]:.3f}%<br>线上收益 %{customdata[1]:.3f}%<extra></extra>",
    ), row=3, col=2)

    best_importance = importance[importance.variant == final_variant].groupby("feature", as_index=False).importance.mean()
    best_importance = best_importance.nlargest(15, "importance").sort_values("importance")
    fig.add_trace(go.Bar(
        x=best_importance.importance, y=best_importance.feature, orientation="h",
        name="重要性", marker_color="#0f766e", showlegend=False,
    ), row=4, col=1)
    auc = classifications[classifications.pair == "ALL"].sort_values("roc_auc")
    fig.add_trace(go.Bar(
        x=auc.roc_auc, y=auc.variant, orientation="h", name="ROC AUC",
        marker_color=["#dc2626" if value == final_variant else "#64748b" for value in auc.variant],
        showlegend=False,
    ), row=4, col=2)
    fig.update_layout(
        template="plotly_white", height=1500, barmode="group",
        margin=dict(l=90, r=35, t=135, b=60),
        legend=dict(orientation="h", yanchor="bottom", y=1.015, x=0),
        hovermode="x unified",
    )
    fig.update_yaxes(title_text="FDUSD", row=1, col=1)
    fig.update_yaxes(title_text="累计盈亏 FDUSD", row=1, col=2)
    fig.update_yaxes(title_text="周盈亏 FDUSD", row=2, col=1)
    fig.update_yaxes(title_text="回撤 %", row=2, col=2)
    fig.update_yaxes(title_text="概率", range=[0, 1], row=3, col=1)
    figure_html = fig.to_html(full_html=False, include_plotlyjs=True, config={"responsive": True})

    def table(frame: pd.DataFrame) -> str:
        return frame.to_html(index=False, border=0, classes="data", float_format=lambda x: f"{x:.6f}")

    leaderboard_columns = [
        "holdout_rank", "variant", "locked_quantile", "development_eligible",
        "isolated_return_pct", "isolated_max_drawdown_pct", "isolated_success",
        "online_return_pct", "online_max_drawdown_pct", "online_portfolio_stop_events",
        "online_pair_stop_events", "online_success", "joint_holdout_success",
    ]
    stress_columns = [
        "scenario", "return_pct", "max_drawdown_pct", "portfolio_stop_events",
        "pair_stop_events", "stress_gate_pass",
    ]
    status_class = "go" if summary["research_verdict"] == "NEXT_STAGE_JOINT_VALIDATION" else "nogo"
    html = f"""<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'>
<meta name='viewport' content='width=device-width,initial-scale=1'>
<title>BTC/ETH ML Momentum Stop Research</title><style>
body{{margin:0;background:#f3f4f6;color:#111827;font-family:Inter,Segoe UI,Arial,sans-serif}}
main{{max-width:1500px;margin:auto;padding:22px}}h1{{font-size:30px;margin:0 0 8px}}h2{{margin-top:30px}}
.banner{{background:white;border-left:6px solid #dc2626;padding:16px 18px;border-radius:8px;box-shadow:0 1px 3px #0001}}
.banner.go{{border-color:#059669}}.banner strong{{font-size:22px}}.muted{{color:#64748b}}
.cards{{display:grid;grid-template-columns:repeat(4,minmax(170px,1fr));gap:12px;margin:16px 0}}
.card{{background:white;padding:14px;border-radius:8px;box-shadow:0 1px 3px #0001}}.card span{{display:block;color:#64748b;font-size:13px}}.card b{{font-size:21px}}
.plot,.section{{background:white;padding:12px;border-radius:8px;margin-top:16px;box-shadow:0 1px 3px #0001}}
.scroll{{overflow-x:auto}}table.data{{border-collapse:collapse;width:100%;font-size:12px;white-space:nowrap}}
table.data th,table.data td{{padding:8px;border-bottom:1px solid #e5e7eb;text-align:right}}table.data th:nth-child(2),table.data td:nth-child(2){{text-align:left}}
@media(max-width:760px){{main{{padding:10px}}h1{{font-size:22px}}.cards{{grid-template-columns:1fr 1fr}}.plot{{padding:0;overflow-x:auto}}.plot>div{{min-width:700px}}}}
</style></head><body><main>
<div class='banner {status_class}'><h1>BTC/ETH 机器学习智能止损研究</h1>
<strong>{summary['research_verdict']}</strong><p>{summary['verdict_reason']}</p>
<div class='muted'>deployment_authorized=false；未修改、未接入实时下单。</div></div>
<div class='cards'>
<div class='card'><span>开发集锁定模型</span><b>{final_variant}</b></div>
<div class='card'><span>隔离轨收益</span><b>{summary['final_holdout']['isolated_return_pct']:.3f}%</b></div>
<div class='card'><span>线上轨收益</span><b>{summary['final_holdout']['online_return_pct']:.3f}%</b></div>
<div class='card'><span>压力测试通过</span><b>{summary['stress_all_pass']}</b></div>
</div>
<div class='plot'>{figure_html}</div>
<div class='section'><h2>10种锁定模型排行榜</h2><div class='scroll'>{table(holdout[leaderboard_columns])}</div></div>
<div class='section'><h2>压力测试</h2><div class='scroll'>{table(stress[stress_columns])}</div></div>
<div class='section'><h2>验证说明</h2><ul>
<li>所有信号来自完整收盘的1h/4h K线；未来6小时标签在成熟前不可进入训练。</li>
<li>ML完全替代机制1的风险进入条件；机制1的分对 ROC/SQZMOM 只参与恢复。</li>
<li>资金费率、OI、主动买入占比因本地数据不存在而排除；宏观/FOMC历史状态统一排除。</li>
<li>短样本和 bootstrap 区间跨零会限制结论；任何结果均不构成部署授权。</li>
</ul></div></main></body></html>"""
    path.write_text(html, encoding="utf-8")


def write_notebook(path: Path, output_dir: Path) -> None:
    import nbformat as nbf
    resolved_output = output_dir.resolve()
    repository_root = Path(__file__).resolve().parents[1]
    script_path = Path(__file__).resolve()
    notebook = nbf.v4.new_notebook()
    notebook.cells = [
        nbf.v4.new_markdown_cell(
            "# BTC/ETH 独立机制1 vs ML智能止损\n\n"
            "**tl;dr**：本 notebook 读取脚本生成的锁定开发集与一次性样本外产物。"
            "默认不重跑昂贵训练；将 `REBUILD=True` 可完整复现。"
        ),
        nbf.v4.new_code_cell(
            "from pathlib import Path\nimport json, subprocess, sys\nimport pandas as pd\n"
            f"OUTPUT = Path(r'{resolved_output.as_posix()}')\n"
            f"REPOSITORY = Path(r'{repository_root.as_posix()}')\n"
            f"SCRIPT = Path(r'{script_path.as_posix()}')\nREBUILD = False\n"
            "if REBUILD:\n"
            "    subprocess.run([sys.executable, str(SCRIPT), '--output-dir', str(OUTPUT)], "
            "check=True, cwd=REPOSITORY)\n"
            "summary = json.loads((OUTPUT/'research_summary.json').read_text(encoding='utf-8'))\nsummary"
        ),
        nbf.v4.new_markdown_cell(
            "## 方法与无前视约束\n\n1h/4h 仅使用完整K线；标签成熟时间为信号后6小时；"
            "每周训练仅纳入 `label_ready_ts <= train_end` 的记录。阈值、架构和算法只在开发集锁定。"
        ),
        nbf.v4.new_code_cell(
            "audit = pd.read_csv(OUTPUT/'training_audit.csv')\n"
            "assert (audit.train_last_label_ready_ts <= audit.train_cutoff_ts).all()\n"
            "audit.tail()"
        ),
        nbf.v4.new_markdown_cell("## 数据质量与锁定结果"),
        nbf.v4.new_code_cell(
            "quality = pd.read_csv(OUTPUT/'data_quality.csv')\n"
            "holdout = pd.read_csv(OUTPUT/'holdout_metrics.csv')\n"
            "display(quality)\ndisplay(holdout)"
        ),
        nbf.v4.new_markdown_cell("## 结论与限制"),
        nbf.v4.new_code_cell(
            "print(summary['research_verdict'])\nprint(summary['verdict_reason'])\n"
            "print(json.dumps(summary['bootstrap'], ensure_ascii=False, indent=2))"
        ),
    ]
    notebook.metadata.kernelspec = {"display_name": "Python 3", "language": "python", "name": "python3"}
    notebook.metadata.language_info = {"name": "python", "version": f"{sys.version_info.major}.{sys.version_info.minor}"}
    path.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, path)


def json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value)!r}")


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    candles, quality = load_candles(args.cache_dir)
    quality.to_csv(args.output_dir / "data_quality.csv", index=False)
    panel = build_feature_panel(candles)
    panel[[
        "pair", "bar_open_utc", "bar_open_ts", "signal_ts", "label_ready_ts",
        "last_complete_1h_ts", "last_complete_4h_ts", "target",
        "future_min_return", "adverse_threshold", *ALL_FEATURES,
    ]].to_csv(args.output_dir / "feature_panel.csv.gz", index=False, compression="gzip")
    gates = mechanism1_gates(candles)

    dev_folds = development_folds(args.source_weekly_results)
    holdout_folds = online_holdout_folds()
    dev_selection_path = args.output_dir / "development_grid_selections.csv"
    holdout_selection_path = args.output_dir / "holdout_grid_selections.csv"
    if args.reuse_candidates and dev_selection_path.exists() and holdout_selection_path.exists():
        dev_grid = pd.read_csv(dev_selection_path)
        online_grid = pd.read_csv(holdout_selection_path)
        candidate_evaluations = pd.DataFrame()
    else:
        dev_grid, dev_evaluations = regenerate_grid_selections(candles, gates, dev_folds)
        online_grid, holdout_evaluations = regenerate_grid_selections(candles, gates, holdout_folds)
        candidate_evaluations = pd.concat([dev_evaluations, holdout_evaluations], ignore_index=True)
        dev_grid.to_csv(dev_selection_path, index=False)
        online_grid.to_csv(holdout_selection_path, index=False)
        candidate_evaluations.to_csv(args.output_dir / "grid_candidate_evaluations.csv.gz", index=False, compression="gzip")

    blocks = model_blocks(dev_folds, holdout_folds)
    predictions_path = args.output_dir / "predictions.csv.gz"
    importance_path = args.output_dir / "feature_importance.csv"
    audit_path = args.output_dir / "training_audit.csv"
    if args.reuse_predictions and predictions_path.exists() and importance_path.exists() and audit_path.exists():
        predictions = pd.read_csv(predictions_path)
        importance = pd.read_csv(importance_path)
        audit = pd.read_csv(audit_path)
    else:
        predictions, importance, audit = train_predictions(panel, blocks)
        predictions.to_csv(predictions_path, index=False, compression="gzip")
        importance.to_csv(importance_path, index=False)
        audit.to_csv(audit_path, index=False)
    if not np.isfinite(predictions.probability).all():
        raise AssertionError("Predictions contain non-finite probabilities")
    if not predictions.probability.between(0, 1).all():
        raise AssertionError("Predictions are outside [0, 1]")
    if not (audit.train_last_label_ready_ts <= audit.train_cutoff_ts).all():
        raise AssertionError("Training audit detected label leakage")

    thresholds = thresholds_from_development(predictions)
    thresholds.to_csv(args.output_dir / "development_probability_thresholds.csv", index=False)
    dev_candidates, locked, dev_base_iso, dev_base_online = development_selection(
        candles, gates, dev_grid,
        predictions[predictions.period == "development"], thresholds,
    )
    dev_candidates.to_csv(args.output_dir / "development_selection_table.csv", index=False)
    locked.to_csv(args.output_dir / "locked_variant_choices.csv", index=False)
    final_setting = locked.iloc[0]
    final_variant = str(final_setting["variant"])
    final_thresholds = {
        pair: float(final_setting[f"{pair}_threshold"]) for pair in PAIRS
    }

    (
        holdout, base_iso, base_online, curves, weekly, events, states, pair_metrics,
    ) = holdout_evaluation(candles, gates, online_grid, predictions, locked)
    holdout.to_csv(args.output_dir / "holdout_metrics.csv", index=False)
    curves.to_csv(args.output_dir / "equity_curves.csv.gz", index=False, compression="gzip")
    weekly.to_csv(args.output_dir / "weekly_pnl.csv", index=False)
    events.to_csv(args.output_dir / "trade_and_signal_events.csv.gz", index=False, compression="gzip")
    states.to_csv(args.output_dir / "risk_probability_states.csv.gz", index=False, compression="gzip")
    pair_metrics.to_csv(args.output_dir / "pair_metrics.csv", index=False)
    classifications = classification_metrics(predictions)
    classifications.to_csv(args.output_dir / "classification_metrics.csv", index=False)

    final_candidate = candidate_from_row(online_grid.iloc[-1])
    if args.skip_stress:
        stress = pd.DataFrame([{
            "scenario": "SKIPPED", "return_pct": np.nan, "max_drawdown_pct": np.nan,
            "portfolio_stop_events": 0, "pair_stop_events": 0,
            "momentum_stop_exits": 0, "stress_gate_pass": False,
        }])
        stress_audit, model_blob = {"skipped": True}, pickle.dumps({"variant": final_variant})
    else:
        stress, stress_audit, model_blob = stress_test(
            candles, panel, final_variant, final_thresholds, final_candidate
        )
    stress.to_csv(args.output_dir / "stress_tests.csv", index=False)
    model_path = args.output_dir / "best_model.pkl"
    model_path.write_bytes(model_blob)

    bootstrap = bootstrap_final(curves, weekly, final_variant)
    (args.output_dir / "bootstrap.json").write_text(
        json.dumps(bootstrap, ensure_ascii=False, indent=2, default=json_default), encoding="utf-8"
    )
    contract = final_signal_contract(states, final_variant, final_thresholds, model_blob)
    (args.output_dir / "grid_ml_momentum_stop_v1.example.json").write_text(
        json.dumps(contract, ensure_ascii=False, indent=2, default=json_default), encoding="utf-8"
    )

    model_parameters = {}
    for name, model in build_models(SEED).items():
        params = model.get_params()
        model_parameters[name] = {
            key: value for key, value in params.items()
            if isinstance(value, (str, int, float, bool, type(None)))
        }
    manifest = {
        "schema": "grid-ml-momentum-stop-v1-research-manifest",
        "seed": SEED, "algorithms": list(ALGORITHMS), "architectures": list(ARCHITECTURES),
        "variants": [variant_name(a, b) for a in ALGORITHMS for b in ARCHITECTURES],
        "quantiles": list(QUANTILES), "features_shared": list(ALL_FEATURES),
        "features_separate": list(SEPARATE_FEATURES),
        "feature_schema_sha256_shared": feature_schema_hash(list(ALL_FEATURES)),
        "feature_schema_sha256_separate": feature_schema_hash(list(SEPARATE_FEATURES)),
        "model_parameters": model_parameters,
        "label": "future 6h minimum return <= -max(0.4%, current 1h ATR_pct)",
        "training_rule": "expanding; label_ready_ts <= weekly train cutoff",
        "entry_rule": "pair probability >= pair threshold; Mechanism 1 entry is not evaluated",
        "recovery_rule": {
            pair: {
                "probability_below_threshold": True,
                "roc_gte_pct": TECHNICAL_PARAMS[pair].roc_recovery_pct,
                "sqzmom_gte_pct": TECHNICAL_PARAMS[pair].sqz_recovery_pct,
                "sqzmom_improving": True,
            } for pair in PAIRS
        },
        "excluded_missing_local_sources": ["funding_rate", "open_interest", "taker_buy_ratio"],
        "excluded_correlated_v1_features": ["Williams %R", "CMO", "TRIX", "KST"],
        "macro_fomc_gate": "excluded uniformly because historical state is unavailable",
        "deployment_allowed": False,
        "best_model_sha256": hashlib.sha256(model_blob).hexdigest(),
        "best_model_file": model_path.name,
    }
    (args.output_dir / "model_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=json_default), encoding="utf-8"
    )

    final_holdout = holdout[holdout.variant == final_variant].iloc[0].to_dict()
    development_eligible = bool(final_setting["eligible"])
    holdout_pass = bool(final_holdout["joint_holdout_success"])
    stress_all_pass = bool(stress.stress_gate_pass.all())
    gate_pass = development_eligible and holdout_pass and stress_all_pass
    verdict = "NEXT_STAGE_JOINT_VALIDATION" if gate_pass else "NO-GO"
    reasons = []
    if not development_eligible:
        reasons.append("开发集没有满足双轨安全约束的候选")
    if not holdout_pass:
        reasons.append("锁定模型未同时通过两条样本外成功门槛")
    if not stress_all_pass:
        reasons.append("至少一个压力场景触发停止或未执行")
    if bootstrap["isolated"]["return_ci_crosses_zero"] or bootstrap["online"]["return_ci_crosses_zero"]:
        reasons.append("周度块bootstrap收益差95%区间跨零，统计证据有限")
    if bootstrap["isolated"]["short_sample_warning"] or bootstrap["online"]["short_sample_warning"]:
        reasons.append("周块少于20，置信区间仅作诊断")
    summary = {
        "research_verdict": verdict,
        "verdict_reason": "；".join(reasons) if reasons else "全部研究门槛通过，但仍不授权部署",
        "deployment_authorized": False,
        "runtime_modified": False,
        "final_model_selected_on_development_only": final_variant,
        "final_locked_quantile": float(final_setting["quantile"]),
        "final_thresholds": final_thresholds,
        "development_eligible": development_eligible,
        "development_baselines": {"isolated": dev_base_iso, "online": dev_base_online},
        "locked_isolated_baseline": base_iso,
        "online_holdout_baseline": base_online,
        "final_holdout": final_holdout,
        "stress_all_pass": stress_all_pass,
        "stress_audit": stress_audit,
        "bootstrap": bootstrap,
        "periods": {
            "isolated_holdout": [utc(ISOLATED_START), utc(ISOLATED_END)],
            "online_holdout": [utc(ONLINE_START), utc(ONLINE_END)],
            "development_folds": len(dev_grid), "online_holdout_folds": len(online_grid),
        },
        "fixed_assumptions": {
            "pairs": list(PAIRS), "maker_fee": 0.0, "taker_fee": TAKER_FEE,
            "order_lifetime_seconds": 7_200, "base_slippage": 0.0,
            "initial_grid": asdict(BASE_CANDIDATE), "capital_fdusd": INITIAL_EQUITY,
        },
    }
    (args.output_dir / "research_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=json_default), encoding="utf-8"
    )
    markdown = f"""# BTC/ETH ML Momentum Stop Research Summary

**Verdict: {verdict}**

Selected on development only: `{final_variant}`, q={float(final_setting['quantile']):.3f}.
Deployment remains unauthorized and the live runtime was not modified.

## Holdout

- Isolated: {final_holdout['isolated_return_pct']:.6f}% return, {final_holdout['isolated_max_drawdown_pct']:.6f}% max drawdown.
- Complete online: {final_holdout['online_return_pct']:.6f}% return, {final_holdout['online_max_drawdown_pct']:.6f}% worst weekly drawdown.
- Mechanism 1 isolated parity: {base_iso['return_pct']:.9f}% / {base_iso['max_drawdown_pct']:.9f}%.

## Limits

{summary['verdict_reason']}
"""
    (args.output_dir / "research_summary.md").write_text(markdown, encoding="utf-8")
    create_report(
        args.output_dir / "interactive_comparison.html", summary=summary,
        holdout=holdout, curves=curves, weekly=weekly, states=states, events=events,
        classifications=classifications, importance=importance,
        final_variant=final_variant, stress=stress,
    )
    write_notebook(args.output_dir / "reproducible_analysis.ipynb", args.output_dir)
    print(json.dumps({
        "output_dir": str(args.output_dir), "verdict": verdict,
        "final_variant": final_variant, "development_eligible": development_eligible,
        "holdout_pass": holdout_pass, "stress_all_pass": stress_all_pass,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

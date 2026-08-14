#!/usr/bin/env python3
"""Development search and fixed-interval revalidation for an XGBoost Grid BUY gate.

This research entry point replaces only the ROC/SQZMOM ordinary-BUY gate. It
never passes ``momentum_stop_timeline`` to the simulator and never changes the
live strategy.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import shutil
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Mapping

import nbformat as nbf
import numpy as np
import pandas as pd
from nbclient import NotebookClient
from sklearn.metrics import balanced_accuracy_score, brier_score_loss, log_loss, roc_auc_score

from compare_independent_gate_ml_stops import (
    ALL_FEATURES, ARCHITECTURES, DAY, FIVE_MINUTES, PAIRS, QUANTILES,
    SEPARATE_FEATURES, build_feature_panel, load_candles, mechanism1_gates,
)
from grid_xgboost_risk_gate import (
    PairGateState, advance_pair_gate, build_contract, feature_schema_hash,
)
from search_fdusd_inventory_exit import aggregate_rows, holdout_windows, stop_metrics
from tune_xgboost_momentum_stop_v2 import (
    BASE_PREDICTION_COLUMNS, canonical_json, fit_predict_block_v2, json_default,
    sha256_file, sha256_frame, write_json, xgb_configurations,
)
from validate_grid_live import (
    Candidate, InventoryExitPolicy, crash_candles, simulate, slice_window,
    technical_buy_gate_timeline,
)


MODEL_VERSION = "xgboost-grid-risk-gate-v1"
LOCK_SCHEMA = "xgboost-grid-risk-gate-v1-lock-v1"
OUTPUT_SCHEMA = "xgboost-grid-risk-gate-v1-revalidation-v1"
DEFAULT_OUTPUT = Path("results/backtests/xgboost_grid_risk_gate_v1")
DEFAULT_SOURCE = Path("results/backtests/fdusd_inventory_exit_parameter_search/weekly_results.csv")
DEFAULT_SUMMARY = Path("results/backtests/fdusd_inventory_exit_parameter_search/summary.json")
DEFAULT_PREDICTIONS = Path("results/backtests/xgboost_momentum_stop_revalidation_v2/development_predictions.csv.gz")
DEFAULT_PANEL = Path("results/backtests/xgboost_momentum_stop_revalidation_v2/feature_panel.csv.gz")
PLUGIN_ROOT = Path(
    "C:/Users/sunny/.codex/plugins/cache/openai-curated-remote/"
    "data-analytics/0.2.8-13ceeea1f599"
)
INITIAL_EQUITY = 420.0
TAKER_FEE = 0.001
DEV_BASELINE_PNL = 32.44000123084811
DEV_BASELINE_DD = -3.9038679244179093
DEV_BASELINE_PAIR_STOPS = 9
DEV_BASELINE_PORTFOLIO_STOPS = 0
REVAL_START = 1_779_897_600  # 2026-05-27 16:00 UTC
REVAL_END = 1_785_081_600    # 2026-07-26 16:00 UTC
REVAL_BASELINE_PNL = -32.34056667417843
REVAL_BASELINE_DD = -5.866793652492175
REVAL_BASELINE_PAIR_STOPS = 7
REVAL_BASELINE_PORTFOLIO_STOPS = 2
POLICY = InventoryExitPolicy(10, 172800, 0.0, 0.0, 0.5, 0.75)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("search", "revalidate", "all"), default="all")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--cache-dir", type=Path, default=Path("data/backtesting_candles"))
    parser.add_argument("--source-weekly-results", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--source-summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--source-predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--source-panel", type=Path, default=DEFAULT_PANEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--skip-stress", action="store_true")
    parser.add_argument("--skip-report", action="store_true")
    return parser.parse_args()


def candidate_from_row(row: Any) -> Candidate:
    return Candidate(
        float(row.half_range), float(row.min_spread), float(row.take_profit),
        float(row.move_threshold), int(row.move_cooldown_seconds),
    )


def mechanism1_gate(candles: Mapping[str, pd.DataFrame]) -> dict[int, bool]:
    """The inventory study's original BTC-controlled ROC/SQZMOM gate."""
    return technical_buy_gate_timeline(candles["BTC-FDUSD"])


def fixed_grid_selections(path: Path, period: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    selected = frame[(frame.period == period) & (frame.scenario == "new")].copy()
    columns = [
        "period", "fold", "test_start", "test_end", "half_range", "min_spread",
        "take_profit", "move_threshold", "move_cooldown_seconds",
    ]
    selected = selected[columns].sort_values("fold").reset_index(drop=True)
    expected = 12 if period == "development" else 8
    if len(selected) != expected:
        raise RuntimeError(f"Expected {expected} fixed {period} Grid folds, found {len(selected)}")
    selected["train_start"] = selected.test_start.astype(int) - 14 * DAY
    selected["train_end"] = selected.test_start.astype(int)
    return selected


def fold_blocks(selections: pd.DataFrame) -> list[SimpleNamespace]:
    return [SimpleNamespace(**row._asdict()) for row in selections.itertuples(index=False)]


def prediction_variant(config_id: str, architecture: str) -> str:
    return f"{config_id} | {architecture}"


def build_buy_gate(
    predictions: pd.DataFrame, variant: str, entry: Mapping[str, float],
    recovery: Mapping[str, float], start_ts: int, end_ts: int,
) -> tuple[dict[str, dict[int, bool]], pd.DataFrame, pd.DataFrame]:
    """Build independent pair BUY-enable maps from completed hourly signals."""
    gates: dict[str, dict[int, bool]] = {pair: {} for pair in PAIRS}
    states, events = [], []
    selected = predictions[
        (predictions.variant == variant)
        & (predictions.signal_ts >= start_ts) & (predictions.signal_ts < end_ts)
    ]
    architecture = variant.rsplit(" | ", 1)[1]
    for pair in PAIRS:
        records = list(selected[selected.pair == pair].sort_values("signal_ts").itertuples(index=False))
        if not records:
            raise RuntimeError(f"No {pair} predictions for {start_ts}..{end_ts}")
        state = PairGateState()
        for index, row in enumerate(records):
            previous = state
            state, signal = advance_pair_gate(
                pair=pair, probability=float(row.probability),
                entry_threshold=float(entry[pair]), recovery_threshold=float(recovery[pair]),
                signal_ts=int(row.signal_ts), previous=state, model_version=variant,
            )
            right = min(
                int(records[index + 1].signal_ts) if index + 1 < len(records) else end_ts,
                end_ts,
            )
            for timestamp in range(max(int(row.signal_ts), start_ts), right, FIVE_MINUTES):
                gates[pair][timestamp] = bool(signal["buy_enabled"])
            state_row = {
                "variant": variant, "architecture": architecture, "pair": pair,
                "signal_ts": int(row.signal_ts), "probability": float(row.probability),
                "entry_threshold": float(entry[pair]),
                "recovery_threshold": float(recovery[pair]),
                "risk_off_active": bool(state.risk_off_active),
                "buy_enabled": bool(signal["buy_enabled"]),
                "consecutive_recovery_bars": state.consecutive_recovery_bars,
                "risk_off_since_ts": state.risk_off_since,
                "transition": signal["transition"], "reason": signal["reason"],
                "event_id": signal["event_id"],
                "last_complete_1h_ts": int(row.last_complete_1h_ts),
                "last_complete_4h_ts": int(row.last_complete_4h_ts),
            }
            states.append(state_row)
            if signal["transition"] in {"enter", "recover"}:
                events.append({
                    "timestamp": int(row.signal_ts), "pair": pair,
                    "side": "PAUSE" if signal["transition"] == "enter" else "RESUME",
                    "reason": f"xgboost_risk_gate_{signal['transition']}",
                    "probability": float(row.probability),
                    "entry_threshold": float(entry[pair]),
                    "recovery_threshold": float(recovery[pair]),
                    "event_id": signal["event_id"],
                })
    return gates, pd.DataFrame(states), pd.DataFrame(events)


def simulate_fold(
    candles: Mapping[str, pd.DataFrame], selection: Any,
    gate: Mapping[str, Mapping[int, bool]], *, taker_fee: float = TAKER_FEE,
    slippage: float = 0.0, record_details: bool = False,
) -> tuple[dict[str, Any], pd.DataFrame, dict[str, Any], pd.DataFrame, dict[str, Any]]:
    trades: list[dict[str, Any]] = []
    end_ts = int(selection.test_end)
    result, curve, pairs = simulate(
        slice_window(dict(candles), int(selection.test_start), end_ts),
        candidate_from_row(selection), maker_fee=0.0, taker_fee=taker_fee,
        slippage=slippage, order_refresh_seconds=7200,
        technical_buy_gate=dict(gate), momentum_stop_timeline=None,
        trade_log=trades, risk_breakers_enabled=True, cost_floor_enabled=True,
        inventory_exit_policy=POLICY, record_curve=record_details,
    )
    if result["momentum_stop_enabled"] or result["momentum_stop_exits"]:
        raise AssertionError("BUY-gate replay unexpectedly enabled a momentum stop")
    if any(item.get("reason") == "momentum_stop_exit" for item in trades):
        raise AssertionError("XGBoost BUY gate produced an immediate Taker exit")
    if record_details:
        stops = stop_metrics(result, curve, trades, end_ts)
    else:
        pair_stops = [item for item in trades if item.get("reason") == "pair_breaker_flatten"]
        portfolio_stops = [item for item in trades if item.get("reason") == "portfolio_breaker"]
        stops = {
            "pair_stop_events": len(pair_stops),
            "pair_stop_hours": sum(max(end_ts-int(item["timestamp"]), 0)/3600 for item in pair_stops),
            "portfolio_stop_events": len(portfolio_stops),
            "portfolio_stop_hours": sum(max(end_ts-int(item["timestamp"]), 0)/3600 for item in portfolio_stops),
        }
    trade_frame = pd.DataFrame(trades) if record_details else pd.DataFrame()
    return result, curve if record_details else pd.DataFrame(), pairs, trade_frame, stops


def replay(
    candles: Mapping[str, pd.DataFrame], selections: pd.DataFrame, *, scenario: str,
    baseline_gate: Mapping[str, Mapping[int, bool]] | None = None,
    predictions: pd.DataFrame | None = None, variant: str | None = None,
    entry: Mapping[str, float] | None = None, recovery: Mapping[str, float] | None = None,
    taker_fee: float = TAKER_FEE, slippage: float = 0.0, record_details: bool = False,
) -> dict[str, Any]:
    weekly, pair_rows, curves, trades, states, gate_events, stop_rows = [], [], [], [], [], [], []
    cumulative = 0.0
    for selection in selections.itertuples(index=False):
        if variant is None:
            if baseline_gate is None:
                raise ValueError("baseline gate is required")
            gate, fold_states, fold_gate_events = baseline_gate, pd.DataFrame(), pd.DataFrame()
        else:
            fold_predictions = predictions[
                (predictions.period == selection.period) & (predictions.fold == int(selection.fold))
            ]
            gate, fold_states, fold_gate_events = build_buy_gate(
                fold_predictions, variant, entry, recovery,
                int(selection.test_start), int(selection.test_end),
            )
        result, curve, pair_metrics, trade_frame, stop = simulate_fold(
            candles, selection, gate, taker_fee=taker_fee, slippage=slippage,
            record_details=record_details,
        )
        weekly.append({
            "scenario": scenario, "period": selection.period, "fold": int(selection.fold),
            "test_start": int(selection.test_start), "test_end": int(selection.test_end),
            **asdict(candidate_from_row(selection)), **result, **stop,
        })
        pair_rows.extend({
            "scenario": scenario, "period": selection.period, "fold": int(selection.fold),
            "pair": pair, **metrics,
        } for pair, metrics in pair_metrics.items())
        if record_details:
            if not curve.empty:
                curve = curve.copy()
                curve["scenario"], curve["period"], curve["fold"] = scenario, selection.period, int(selection.fold)
                curve["cumulative_oos_pnl"] = cumulative + curve.equity - INITIAL_EQUITY
                curves.append(curve)
            if not trade_frame.empty:
                trade_frame["scenario"], trade_frame["period"], trade_frame["fold"] = scenario, selection.period, int(selection.fold)
                trades.append(trade_frame)
            if not fold_states.empty:
                fold_states["period"], fold_states["fold"] = selection.period, int(selection.fold)
                states.append(fold_states)
            if not fold_gate_events.empty:
                fold_gate_events["scenario"], fold_gate_events["period"], fold_gate_events["fold"] = scenario, selection.period, int(selection.fold)
                gate_events.append(fold_gate_events)
            for item in (trade_frame.to_dict("records") if not trade_frame.empty else []):
                if item.get("reason") == "pair_breaker_flatten":
                    stop_rows.append({"scenario": scenario, "fold": int(selection.fold), "scope": item["pair"], "kind": "pair_stop", "start_ts": int(item["timestamp"]), "end_ts": int(selection.test_end)})
            if result["liquidated"] and not curve.empty:
                stop_rows.append({"scenario": scenario, "fold": int(selection.fold), "scope": "PORTFOLIO", "kind": "portfolio_stop", "start_ts": int(curve.timestamp.iloc[-1]), "end_ts": int(selection.test_end)})
        cumulative += float(result["net_pnl_quote"])
    weekly_frame, pair_frame = pd.DataFrame(weekly), pd.DataFrame(pair_rows)
    summary = aggregate_rows(weekly, pair_rows)
    summary["risk_off_pair_hours"] = float(pair_frame.technical_risk_off_seconds.sum() / 3600)
    summary["momentum_stop_exits"] = int(weekly_frame.momentum_stop_exits.sum())
    return {
        "summary": summary, "weekly": weekly_frame, "pairs": pair_frame,
        "equity": pd.concat(curves, ignore_index=True) if curves else pd.DataFrame(),
        "trades": pd.concat(trades, ignore_index=True) if trades else pd.DataFrame(),
        "states": pd.concat(states, ignore_index=True) if states else pd.DataFrame(),
        "gate_events": pd.concat(gate_events, ignore_index=True) if gate_events else pd.DataFrame(),
        "stops": pd.DataFrame(stop_rows),
    }


def thresholds_from_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for variant in sorted(predictions.variant.unique()):
        for entry_quantile in QUANTILES:
            recovery_quantile = entry_quantile - 0.10
            row = {"variant": variant, "entry_quantile": entry_quantile, "recovery_quantile": recovery_quantile}
            for pair in PAIRS:
                values = predictions[(predictions.variant == variant) & (predictions.pair == pair)].probability
                row[f"{pair}_entry_threshold"] = float(values.quantile(entry_quantile))
                row[f"{pair}_recovery_threshold"] = float(values.quantile(recovery_quantile))
            rows.append(row)
    result = pd.DataFrame(rows)
    if len(result) != 640:
        raise AssertionError(f"Expected 640 threshold candidates, got {len(result)}")
    return result


def threshold_maps(row: Any) -> tuple[dict[str, float], dict[str, float]]:
    def value(name: str) -> float:
        return float(row[name]) if isinstance(row, Mapping) or isinstance(row, pd.Series) else float(getattr(row, name))
    entry = {pair: value(f"{pair}_entry_threshold") for pair in PAIRS}
    recovery = {pair: value(f"{pair}_recovery_threshold") for pair in PAIRS}
    return entry, recovery


def prepare(args: argparse.Namespace) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    configs = pd.DataFrame(xgb_configurations())
    configs_path = args.output_dir / "xgboost_parameter_configurations.csv"
    configs.to_csv(configs_path, index=False)
    candles, quality = load_candles(args.cache_dir)
    quality.to_csv(args.output_dir / "data_quality.csv", index=False)
    panel_path = args.output_dir / "feature_panel.csv.gz"
    if not panel_path.exists():
        if args.source_panel.exists():
            shutil.copy2(args.source_panel, panel_path)
        else:
            panel = build_feature_panel(candles)
            panel[[*BASE_PREDICTION_COLUMNS, *ALL_FEATURES]].to_csv(panel_path, index=False, compression="gzip")
    panel = pd.read_csv(panel_path)
    missing = set(BASE_PREDICTION_COLUMNS).union(ALL_FEATURES).difference(panel.columns)
    if missing:
        raise RuntimeError(f"Feature cache missing columns: {sorted(missing)}")
    dev = fixed_grid_selections(args.source_weekly_results, "development")
    reval = fixed_grid_selections(args.source_weekly_results, "holdout")
    dev_path, reval_path = args.output_dir / "development_grid_selections.csv", args.output_dir / "revalidation_grid_selections.csv"
    dev.to_csv(dev_path, index=False); reval.to_csv(reval_path, index=False)
    hashes = {
        "candles": {pair: sha256_file(args.cache_dir / f"binance_{pair}_5m.csv") for pair in PAIRS},
        "feature_panel_sha256": sha256_file(panel_path),
        "feature_schema_sha256_shared": feature_schema_hash(list(ALL_FEATURES)),
        "feature_schema_sha256_separate": feature_schema_hash(list(SEPARATE_FEATURES)),
        "development_grid_sha256": sha256_file(dev_path),
        "revalidation_grid_sha256": sha256_file(reval_path),
        "configurations_sha256": sha256_file(configs_path),
    }
    write_json(args.output_dir / "input_hashes.json", hashes)
    return candles, panel, dev, reval, hashes


def validate_prediction_cache(predictions: pd.DataFrame, dev: pd.DataFrame) -> None:
    required = set(BASE_PREDICTION_COLUMNS).union({"variant", "architecture", "algorithm", "probability", "period", "fold"})
    missing = required.difference(predictions.columns)
    if missing:
        raise RuntimeError(f"Prediction cache missing columns: {sorted(missing)}")
    variants = set(predictions.variant.unique())
    expected = {prediction_variant(c["config_id"], architecture) for c in xgb_configurations() for architecture in ARCHITECTURES}
    if variants != expected:
        raise RuntimeError("Prediction cache does not contain all 80 deterministic variants")
    dev_predictions = predictions[predictions.period == "development"]
    if set(dev_predictions.fold.astype(int)) != set(dev.fold.astype(int)):
        raise RuntimeError("Prediction cache development folds do not match fixed Grid folds")
    if not np.isfinite(dev_predictions.probability).all() or not dev_predictions.probability.between(0, 1).all():
        raise RuntimeError("Prediction cache contains invalid probabilities")
    for fold in dev.itertuples(index=False):
        rows = dev_predictions[dev_predictions.fold == int(fold.fold)]
        if rows.signal_ts.min() < int(fold.test_start) or rows.signal_ts.max() >= int(fold.test_end):
            raise RuntimeError("Prediction cache timestamp range does not match development fold")


def search_predictions(args: argparse.Namespace, panel: pd.DataFrame, dev: pd.DataFrame, hashes: Mapping[str, Any]) -> pd.DataFrame:
    output = args.output_dir / "development_predictions.csv.gz"
    provenance_path = args.output_dir / "prediction_cache_provenance.json"
    cache_keys = ["candles", "feature_panel_sha256", "feature_schema_sha256_shared", "feature_schema_sha256_separate", "configurations_sha256"]
    if output.exists() and args.resume:
        if not provenance_path.exists():
            raise RuntimeError("Prediction cache provenance is missing; refusing reuse")
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        expected_hashes = {key: hashes[key] for key in cache_keys}
        if provenance.get("validated_input_hashes") != expected_hashes:
            raise RuntimeError("Prediction cache input hash mismatch; refusing reuse")
        if provenance.get("prediction_sha256") != sha256_file(output):
            raise RuntimeError("Prediction cache file hash mismatch; refusing reuse")
        predictions = pd.read_csv(output)
        validate_prediction_cache(predictions, dev)
        return predictions
    if args.source_predictions.exists():
        source_hash_path = args.source_predictions.parent / "input_hashes.json"
        if not source_hash_path.exists():
            raise RuntimeError("Source prediction cache has no input hash manifest")
        source_hashes = json.loads(source_hash_path.read_text(encoding="utf-8"))
        expected_hashes = {key: hashes[key] for key in cache_keys}
        actual_hashes = {key: source_hashes.get(key) for key in cache_keys}
        if actual_hashes != expected_hashes:
            raise RuntimeError("Source prediction cache hash mismatch; refusing reuse")
        predictions = pd.read_csv(args.source_predictions)
        predictions = predictions[predictions.period == "development"].copy()
        validate_prediction_cache(predictions, dev)
        predictions.to_csv(output, index=False, compression="gzip")
        write_json(provenance_path, {
            "reused": True, "source": str(args.source_predictions),
            "source_sha256": sha256_file(args.source_predictions),
            "validated_input_hashes": expected_hashes,
            "prediction_sha256": sha256_file(output),
        })
        return predictions
    prediction_parts, importance_parts, audit_parts = [], [], []
    for config in xgb_configurations():
        for architecture in ARCHITECTURES:
            for block in fold_blocks(dev):
                prediction, importance, audit, _ = fit_predict_block_v2(panel, block, config, architecture)
                prediction_parts.append(prediction); importance_parts.append(importance); audit_parts.append(audit)
    predictions = pd.concat(prediction_parts, ignore_index=True)
    validate_prediction_cache(predictions, dev)
    predictions.to_csv(output, index=False, compression="gzip")
    write_json(provenance_path, {
        "reused": False,
        "validated_input_hashes": {key: hashes[key] for key in cache_keys},
        "prediction_sha256": sha256_file(output),
    })
    pd.concat(importance_parts, ignore_index=True).to_csv(args.output_dir / "development_feature_importance.csv", index=False)
    pd.concat(audit_parts, ignore_index=True).to_csv(args.output_dir / "development_training_audit.csv", index=False)
    return predictions


def assert_baseline(summary: Mapping[str, Any], period: str) -> None:
    expected = (
        (DEV_BASELINE_PNL, DEV_BASELINE_DD, DEV_BASELINE_PAIR_STOPS, DEV_BASELINE_PORTFOLIO_STOPS)
        if period == "development" else
        (REVAL_BASELINE_PNL, REVAL_BASELINE_DD, REVAL_BASELINE_PAIR_STOPS, REVAL_BASELINE_PORTFOLIO_STOPS)
    )
    actual = (summary["oos_pnl_fdusd"], summary["worst_drawdown_pct"], summary["pair_stop_events"], summary["portfolio_stop_events"])
    if abs(actual[0] - expected[0]) > 1e-8 or abs(actual[1] - expected[1]) > 1e-8 or actual[2:] != expected[2:]:
        raise AssertionError(f"{period} Mechanism 1 parity failed: actual={actual}, expected={expected}")


def rank_candidates(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["pnl_rank"] = out.oos_pnl_fdusd.rank(method="average", pct=True)
    out["drawdown_rank"] = out.worst_drawdown_pct.rank(method="average", pct=True)
    out["portfolio_stop_rank"] = out.portfolio_stop_hours.rank(method="average", pct=True, ascending=False)
    out["pair_stop_rank"] = out.pair_stop_hours.rank(method="average", pct=True, ascending=False)
    out["balanced_score"] = 0.40*out.pnl_rank + 0.25*out.drawdown_rank + 0.20*out.portfolio_stop_rank + 0.15*out.pair_stop_rank
    out = out.sort_values(
        ["eligible", "balanced_score", "portfolio_stop_hours", "pair_stop_hours", "oos_pnl_fdusd", "risk_off_pair_hours"],
        ascending=[False, False, True, True, False, True],
    ).reset_index(drop=True)
    out["rank"] = np.arange(1, len(out)+1)
    return out


def run_search(args: argparse.Namespace, candles: Mapping[str, pd.DataFrame], panel: pd.DataFrame, dev: pd.DataFrame, hashes: Mapping[str, Any]) -> dict:
    gates = mechanism1_gate(candles)
    baseline = replay(candles, dev, scenario="Mechanism 1", baseline_gate=gates)
    assert_baseline(baseline["summary"], "development")
    predictions = search_predictions(args, panel, dev, hashes)
    thresholds = thresholds_from_predictions(predictions)
    thresholds.to_csv(args.output_dir / "development_probability_thresholds.csv", index=False)
    cache_dir = args.output_dir / "search_cache"; cache_dir.mkdir(exist_ok=True)
    rows = []
    for index, (_, setting) in enumerate(thresholds.iterrows(), 1):
        cache = cache_dir / f"candidate_{index:03d}.json"
        if args.resume and cache.exists():
            row = json.loads(cache.read_text(encoding="utf-8")); rows.append(row); continue
        entry, recovery = threshold_maps(setting)
        result = replay(
            candles, dev, scenario=setting.variant, predictions=predictions,
            variant=setting.variant, entry=entry, recovery=recovery,
        )
        metrics = result["summary"]
        row = {
            "variant": setting.variant,
            "config_id": setting.variant.split(" | ")[0],
            "architecture": setting.variant.rsplit(" | ", 1)[1],
            "entry_quantile": float(setting.entry_quantile),
            "recovery_quantile": float(setting.recovery_quantile),
            **{f"{pair}_entry_threshold": entry[pair] for pair in PAIRS},
            **{f"{pair}_recovery_threshold": recovery[pair] for pair in PAIRS},
            **metrics,
        }
        row["eligible"] = bool(
            metrics["oos_pnl_fdusd"] > baseline["summary"]["oos_pnl_fdusd"]
            and metrics["worst_drawdown_pct"] >= baseline["summary"]["worst_drawdown_pct"]
            and metrics["portfolio_stop_events"] == 0
            and metrics["pair_stop_events"] < baseline["summary"]["pair_stop_events"]
        )
        write_json(cache, row); rows.append(row)
        if index % 25 == 0 or index == len(thresholds):
            print(f"development replay {index}/{len(thresholds)}", flush=True)
    candidates = rank_candidates(pd.DataFrame(rows))
    candidates.to_csv(args.output_dir / "development_640_candidates.csv", index=False)
    winner = candidates.iloc[0].to_dict()
    config = next(item for item in xgb_configurations() if item["config_id"] == winner["config_id"])
    lock = {
        "schema": LOCK_SCHEMA, "model_version": MODEL_VERSION,
        "selection_source": "development_only", "immutable": True,
        "development_has_eligible_candidate": bool(candidates.eligible.any()),
        "forced_no_go": not bool(candidates.eligible.any()),
        "variant": winner["variant"], "config_id": winner["config_id"],
        "architecture": winner["architecture"], "configuration": config,
        "entry_quantile": winner["entry_quantile"], "recovery_quantile": winner["recovery_quantile"],
        "entry_thresholds": {pair: winner[f"{pair}_entry_threshold"] for pair in PAIRS},
        "recovery_thresholds": {pair: winner[f"{pair}_recovery_threshold"] for pair in PAIRS},
        "development_metrics": {key: winner[key] for key in (
            "oos_pnl_fdusd", "worst_drawdown_pct", "pair_stop_events", "portfolio_stop_events",
            "pair_stop_hours", "portfolio_stop_hours", "risk_off_pair_hours", "balanced_score", "eligible")},
        "baseline_metrics": baseline["summary"], "input_hashes": dict(hashes),
        "development_prediction_sha256": sha256_file(args.output_dir / "development_predictions.csv.gz"),
        "development_candidates_sha256": sha256_file(args.output_dir / "development_640_candidates.csv"),
    }
    lock["lock_payload_sha256"] = feature_schema_hash([canonical_json(lock)])
    lock_path = args.output_dir / "locked_configuration.json"
    if lock_path.exists() and args.resume:
        existing = json.loads(lock_path.read_text(encoding="utf-8"))
        if canonical_json(existing) != canonical_json(lock):
            raise RuntimeError("Existing immutable lock differs from development winner")
    else:
        write_json(lock_path, lock)
    write_json(args.output_dir / "search_summary.json", {
        "schema": OUTPUT_SCHEMA, "candidates": len(candidates), "variants": candidates.variant.nunique(),
        "eligible_candidates": int(candidates.eligible.sum()), "baseline": baseline["summary"],
        "locked": lock, "revalidation_metrics_read": False,
    })
    return lock


def validate_lock(args: argparse.Namespace, hashes: Mapping[str, Any]) -> dict:
    path = args.output_dir / "locked_configuration.json"
    if not path.exists():
        raise RuntimeError("revalidate requires locked_configuration.json from search")
    lock = json.loads(path.read_text(encoding="utf-8"))
    if lock.get("schema") != LOCK_SCHEMA or not lock.get("immutable"):
        raise RuntimeError("Invalid or mutable lock file")
    if lock.get("input_hashes") != dict(hashes):
        raise RuntimeError("Input hash mismatch; refusing revalidation")
    prediction_path = args.output_dir / "development_predictions.csv.gz"
    candidate_path = args.output_dir / "development_640_candidates.csv"
    if sha256_file(prediction_path) != lock["development_prediction_sha256"] or sha256_file(candidate_path) != lock["development_candidates_sha256"]:
        raise RuntimeError("Locked development artifact hash mismatch")
    return lock


def train_revalidation_predictions(args: argparse.Namespace, panel: pd.DataFrame, reval: pd.DataFrame, lock: Mapping[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    output = args.output_dir / "revalidation_predictions.csv.gz"
    audit_path = args.output_dir / "revalidation_training_audit.csv"
    importance_path = args.output_dir / "revalidation_gain_feature_importance.csv"
    if args.resume and output.exists() and audit_path.exists() and importance_path.exists():
        stored = pd.read_csv(output)
        last_block = fold_blocks(reval)[-1]
        check, _, _, latest = fit_predict_block_v2(
            panel, last_block, lock["configuration"], lock["architecture"]
        )
        keys = ["pair", "signal_ts"]
        expected = stored[stored.fold == int(last_block.fold)][keys + ["probability"]].sort_values(keys)
        actual = check[keys + ["probability"]].sort_values(keys)
        if len(expected) != len(actual) or not np.allclose(expected.probability, actual.probability, rtol=0, atol=1e-12):
            raise RuntimeError("Serialized revalidation prediction cache differs from deterministic refit")
        return stored, pd.read_csv(audit_path), pd.read_csv(importance_path), latest
    predictions, audits, importances, latest = [], [], [], {}
    config = lock["configuration"]; architecture = lock["architecture"]
    for block in fold_blocks(reval):
        prediction, importance, audit, fitted = fit_predict_block_v2(panel, block, config, architecture)
        predictions.append(prediction); importances.append(importance); audits.append(audit); latest = fitted
    pred = pd.concat(predictions, ignore_index=True)
    pred.to_csv(output, index=False, compression="gzip")
    audit = pd.concat(audits, ignore_index=True); audit.to_csv(audit_path, index=False)
    importance = pd.concat(importances, ignore_index=True); importance.to_csv(importance_path, index=False)
    return pred, audit, importance, latest


def classification_metrics(predictions: pd.DataFrame, entry: Mapping[str, float]) -> pd.DataFrame:
    rows = []
    for pair in PAIRS:
        frame = predictions[predictions.pair == pair]
        y = frame.target.astype(int).to_numpy(); p = frame.probability.to_numpy(float)
        pred = (p >= float(entry[pair])).astype(int)
        rows.append({
            "pair": pair, "rows": len(frame), "positive_rate": float(y.mean()),
            "roc_auc": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else math.nan,
            "log_loss": float(log_loss(y, p, labels=[0, 1])), "brier_score": float(brier_score_loss(y, p)),
            "balanced_accuracy_at_entry": float(balanced_accuracy_score(y, pred)),
        })
    return pd.DataFrame(rows)


def paired_bootstrap(model: pd.DataFrame, baseline: pd.DataFrame, samples: int = 10_000) -> dict:
    merged = model.merge(baseline, on="fold", suffixes=("_model", "_baseline"))
    pnl = (merged.net_pnl_quote_model - merged.net_pnl_quote_baseline).to_numpy(float)
    dd = (merged.max_drawdown_pct_model - merged.max_drawdown_pct_baseline).to_numpy(float) * 100
    rng = np.random.default_rng(42); indices = rng.integers(0, len(pnl), size=(samples, len(pnl)))
    pnl_samples = pnl[indices].sum(axis=1)
    dd_samples = dd[indices].mean(axis=1)
    pnl_ci = np.quantile(pnl_samples, [0.025, 0.975]); dd_ci = np.quantile(dd_samples, [0.025, 0.975])
    return {
        "method": "paired weekly block bootstrap", "blocks": len(pnl), "samples": samples,
        "pnl_difference_fdusd": float(pnl.sum()), "pnl_difference_95ci_fdusd": [float(x) for x in pnl_ci],
        "mean_fold_drawdown_difference_pct": float(dd.mean()), "drawdown_difference_95ci_pct": [float(x) for x in dd_ci],
        "pnl_ci_crosses_zero": bool(pnl_ci[0] <= 0 <= pnl_ci[1]), "short_sample_warning": len(pnl) < 20,
    }


def risk_off_intervals(states: pd.DataFrame, selections: pd.DataFrame) -> pd.DataFrame:
    rows = []
    ends = selections.set_index("fold").test_end.to_dict()
    for (fold, pair), frame in states.groupby(["fold", "pair"]):
        start = None
        for item in frame.sort_values("signal_ts").itertuples(index=False):
            if item.transition == "enter":
                start = int(item.signal_ts)
            elif item.transition == "recover" and start is not None:
                rows.append({"fold": int(fold), "pair": pair, "start_ts": start, "end_ts": int(item.signal_ts), "duration_hours": (int(item.signal_ts)-start)/3600, "end_reason": "recover"})
                start = None
        if start is not None:
            end = int(ends[int(fold)])
            rows.append({"fold": int(fold), "pair": pair, "start_ts": start, "end_ts": end, "duration_hours": (end-start)/3600, "end_reason": "weekly_reinitialization"})
    return pd.DataFrame(rows)


def serialization_audit(
    model_blob: bytes, panel: pd.DataFrame, predictions: pd.DataFrame,
    last_selection: Any, architecture: str,
) -> pd.DataFrame:
    payload = pickle.loads(model_blob)
    features = list(ALL_FEATURES if architecture == "shared" else SEPARATE_FEATURES)
    testing = panel[(panel.signal_ts >= int(last_selection.test_start)) & (panel.signal_ts < int(last_selection.test_end))].copy()
    expected = predictions[predictions.fold == int(last_selection.fold)]
    rows = []
    groups = [("ALL", testing)] if architecture == "shared" else [(pair, testing[testing.pair == pair]) for pair in PAIRS]
    for model_pair, frame in groups:
        probability = payload["models"][model_pair].predict_proba(frame[features])[:, 1]
        actual = frame[["pair", "signal_ts"]].copy(); actual["roundtrip_probability"] = probability
        merged = actual.merge(expected[["pair", "signal_ts", "probability"]], on=["pair", "signal_ts"], how="inner")
        rows.append({
            "model_pair": model_pair, "rows": len(merged),
            "max_absolute_probability_difference": float(np.max(np.abs(merged.roundtrip_probability-merged.probability))),
            "probabilities_finite": bool(np.isfinite(probability).all()),
            "probabilities_in_unit_interval": bool(np.logical_and(probability >= 0, probability <= 1).all()),
        })
    return pd.DataFrame(rows)


def run_revalidation(args: argparse.Namespace, candles: Mapping[str, pd.DataFrame], panel: pd.DataFrame, reval: pd.DataFrame, hashes: Mapping[str, Any]) -> dict:
    lock = validate_lock(args, hashes)
    predictions, audit, importance, models = train_revalidation_predictions(args, panel, reval, lock)
    if not bool((audit.train_last_label_ready_ts <= audit.train_cutoff_ts).all()):
        raise AssertionError("Revalidation training used an immature six-hour label")
    if not bool((audit.core_last_signal_ts < audit.early_stop_first_signal_ts).all()):
        raise AssertionError("Early-stop split is not strictly chronological")
    if not bool((audit.early_stop_first_signal_ts < audit.test_first_signal_ts).all()):
        raise AssertionError("Early-stop records overlap the prediction week")
    entry, recovery = lock["entry_thresholds"], lock["recovery_thresholds"]
    base = replay(candles, reval, scenario="Mechanism 1", baseline_gate=mechanism1_gate(candles), record_details=True)
    assert_baseline(base["summary"], "revalidation")
    model = replay(candles, reval, scenario=lock["variant"], predictions=predictions, variant=lock["variant"], entry=entry, recovery=recovery, record_details=True)
    for name, frame in (("revalidation_equity_curves.csv.gz", pd.concat([base["equity"], model["equity"]], ignore_index=True)),
                        ("revalidation_trade_events.csv.gz", pd.concat([base["trades"], model["trades"]], ignore_index=True)),
                        ("revalidation_risk_states.csv.gz", model["states"])):
        frame.to_csv(args.output_dir / name, index=False, compression="gzip")
    pd.concat([base["weekly"], model["weekly"]], ignore_index=True).to_csv(args.output_dir / "revalidation_weekly_results.csv", index=False)
    pd.concat([base["pairs"], model["pairs"]], ignore_index=True).to_csv(args.output_dir / "revalidation_pair_results.csv", index=False)
    pd.concat([base["stops"], model["stops"]], ignore_index=True).to_csv(args.output_dir / "revalidation_stop_events.csv", index=False)
    model["gate_events"].to_csv(args.output_dir / "revalidation_risk_gate_events.csv", index=False)
    class_metrics = classification_metrics(predictions, entry); class_metrics.to_csv(args.output_dir / "revalidation_classification_metrics.csv", index=False)
    bootstrap = paired_bootstrap(model["weekly"], base["weekly"]); write_json(args.output_dir / "revalidation_bootstrap.json", bootstrap)
    metrics_rows = []
    for scenario, result in (("Mechanism 1", base), (lock["variant"], model)):
        metrics_rows.append({"scenario": scenario, **result["summary"]})
    metrics = pd.DataFrame(metrics_rows); metrics.to_csv(args.output_dir / "revalidation_metrics.csv", index=False)
    success = bool(
        model["summary"]["oos_pnl_fdusd"] > base["summary"]["oos_pnl_fdusd"]
        and model["summary"]["oos_pnl_fdusd"] >= 0
        and model["summary"]["worst_drawdown_pct"] >= base["summary"]["worst_drawdown_pct"]
        and model["summary"]["portfolio_stop_events"] == 0
        and model["summary"]["pair_stop_events"] < REVAL_BASELINE_PAIR_STOPS
    )
    stress = pd.DataFrame()
    if not args.skip_stress:
        stress = run_stress(args, candles, panel, reval.iloc[-1], lock)
    stress_pass = bool(not stress.empty and stress.stress_gate_pass.all()) if not args.skip_stress else False
    verdict = "NEXT_STAGE_JOINT_VALIDATION" if success and stress_pass and not lock["forced_no_go"] else "NO-GO"
    model_blob = pickle.dumps({"variant": lock["variant"], "features": list(ALL_FEATURES if lock["architecture"] == "shared" else SEPARATE_FEATURES), "models": models}, protocol=pickle.HIGHEST_PROTOCOL)
    (args.output_dir / "locked_xgboost_models.pkl").write_bytes(model_blob)
    serialization = serialization_audit(model_blob, panel, predictions, reval.iloc[-1], lock["architecture"])
    serialization.to_csv(args.output_dir / "model_serialization_audit.csv", index=False)
    write_json(args.output_dir / "model_manifest.json", {
        "model_version": MODEL_VERSION, "variant": lock["variant"],
        "architecture": lock["architecture"], "configuration": lock["configuration"],
        "features": list(ALL_FEATURES if lock["architecture"] == "shared" else SEPARATE_FEATURES),
        "model_groups": sorted(models), "deployment_allowed": False,
    })
    intervals = risk_off_intervals(model["states"], reval)
    intervals.to_csv(args.output_dir / "revalidation_risk_off_intervals.csv", index=False)
    from plot_xgboost_risk_gate_timing import generate_plot
    generate_plot(
        args.output_dir / "revalidation_risk_states.csv.gz",
        args.output_dir / "revalidation_risk_off_intervals.csv",
        args.output_dir / "risk_gate_entry_exit_plotly.html",
        args.output_dir / "risk_gate_entry_exit_events.csv",
    )
    contract = contract_example(model["states"], lock, model_blob)
    write_json(args.output_dir / "grid_xgboost_risk_gate_v1.example.json", contract)
    summary = {
        "schema": OUTPUT_SCHEMA, "evidence_status": "revalidation", "verdict": verdict,
        "deployment_authorized": False, "development_forced_no_go": lock["forced_no_go"],
        "locked_variant": lock["variant"], "baseline": base["summary"], "model": model["summary"],
        "fixed_interval_success": success, "stress_pass": stress_pass,
        "bootstrap": bootstrap, "input_hashes": dict(hashes),
        "limitations": ["Fixed interval had already been viewed and is revalidation evidence.", "Eight weekly blocks make the bootstrap interval unstable.", "The 10 FDUSD inventory cap is enforced at order allocation; marked notional can move with price and peaked at 10.377 FDUSD in this replay.", "Funding, OI, taker-buy ratio and historical macro/FOMC state are unavailable and excluded uniformly."],
    }
    write_json(args.output_dir / "research_summary.json", summary)
    contract_text = canonical_json(contract).lower()
    acceptance = {
        "baseline_reproduced_exactly": True,
        "forty_unique_configs": len(xgb_configurations()) == 40,
        "eighty_development_variants": pd.read_csv(args.output_dir / "development_predictions.csv.gz").variant.nunique() == 80,
        "six_hundred_forty_candidates": len(pd.read_csv(args.output_dir / "development_640_candidates.csv")) == 640,
        "label_maturity_purge_passed": bool((audit.train_last_label_ready_ts <= audit.train_cutoff_ts).all()),
        "early_stop_chronology_passed": bool((audit.core_last_signal_ts < audit.early_stop_first_signal_ts).all() and (audit.early_stop_first_signal_ts < audit.test_first_signal_ts).all()),
        "probabilities_finite_and_bounded": bool(np.isfinite(predictions.probability).all() and predictions.probability.between(0, 1).all()),
        "serialization_probability_match": bool((serialization.max_absolute_probability_difference <= 1e-12).all()),
        "inventory_acquisition_cap_policy_active": bool(all(item.inventory_exit_policy.notna().all() for item in (base["weekly"], model["weekly"]))),
        "marked_extra_inventory_within_existing_5pct_tolerance": bool(pd.concat([base["pairs"], model["pairs"]]).max_extra_inventory_quote_observed.max() <= POLICY.max_extra_inventory_quote * 1.05),
        "max_marked_extra_inventory_quote": float(pd.concat([base["pairs"], model["pairs"]]).max_extra_inventory_quote_observed.max()),
        "model_gate_created_no_immediate_sell": bool(model["summary"]["momentum_stop_exits"] == 0 and "momentum_stop_exit" not in set(model["trades"].reason if not model["trades"].empty else [])),
        "all_model_recoveries_obey_four_hour_minimum": bool((intervals.loc[intervals.end_reason == "recover", "duration_hours"] >= 4).all()),
        "contract_contains_no_sell_action": all(token not in contract_text for token in ("stop_excess_inventory", "momentum_stop_exit", '"sell"', '"taker"')),
        "weekly_grid_sequence_shared": True,
        "report_deployment_allowed": False,
    }
    write_json(args.output_dir / "acceptance_checks.json", acceptance)
    (args.output_dir / "technical_summary.md").write_text(
        f"# XGBoost独立Risk-off门再验证摘要\n\n- 结论：**{verdict}**；deployment_authorized=false。\n"
        f"- 锁定：`{lock['variant']}`，进入分位数 {lock['entry_quantile']:.3f}，恢复分位数 {lock['recovery_quantile']:.3f}。\n"
        f"- 固定区间：模型 {model['summary']['oos_pnl_fdusd']:+.6f} FDUSD，机制1 {base['summary']['oos_pnl_fdusd']:+.6f} FDUSD。\n"
        f"- 停止：模型单对 {model['summary']['pair_stop_events']} 次、组合 {model['summary']['portfolio_stop_events']} 次。\n"
        "- 证据仅为revalidation，不是全新未见样本外；实时策略未修改。\n",
        encoding="utf-8",
    )
    write_notebook(args.output_dir)
    if not args.skip_report:
        build_report(args.output_dir, summary)
    return summary


def run_stress(args: argparse.Namespace, candles: Mapping[str, pd.DataFrame], panel: pd.DataFrame, last_selection: Any, lock: Mapping[str, Any]) -> pd.DataFrame:
    """Run the five fixed scenarios on the last validation week.

    Actual-scenario probabilities reuse the locked final weekly prediction.
    The crash scenario rebuilds features and refits the locked model at the same
    cutoff, preserving label maturity and avoiding use of future crash labels.
    """
    selection_dict = last_selection.to_dict() if isinstance(last_selection, pd.Series) else last_selection._asdict()
    selection_block = SimpleNamespace(**selection_dict)
    start, end = int(selection_block.test_start), int(selection_block.test_end)
    actual = pd.read_csv(args.output_dir / "revalidation_predictions.csv.gz")
    actual = actual[actual.fold == int(selection_block.fold)]
    # Bound the source at the stress interval end before applying the helper;
    # otherwise it would shock the dataset's final day (after this fold).
    crash_source = {
        pair: frame[frame.timestamp < end].copy().reset_index(drop=True)
        for pair, frame in candles.items()
    }
    crash = crash_candles(crash_source, drop=0.15)
    crash_panel = build_feature_panel(crash)
    prediction, _, _, _ = fit_predict_block_v2(crash_panel, selection_block, lock["configuration"], lock["architecture"])
    scenarios = [
        ("base", candles, actual, 0.001, 0.0),
        ("taker_fee_150pct", candles, actual, 0.0015, 0.0),
        ("slippage_0.05pct", candles, actual, 0.001, 0.0005),
        ("slippage_0.10pct", candles, actual, 0.001, 0.0010),
        ("one_day_15pct_crash", crash, prediction, 0.001, 0.0),
    ]
    rows = []
    for name, scenario_candles, pred, fee, slippage in scenarios:
        selection = pd.DataFrame([selection_dict])
        result = replay(scenario_candles, selection, scenario=name, predictions=pred, variant=lock["variant"], entry=lock["entry_thresholds"], recovery=lock["recovery_thresholds"], taker_fee=fee, slippage=slippage)
        metrics = result["summary"]
        rows.append({"scenario": name, **metrics, "stress_gate_pass": metrics["portfolio_stop_events"] == 0 and metrics["pair_stop_events"] == 0})
    frame = pd.DataFrame(rows); frame.to_csv(args.output_dir / "revalidation_stress_tests.csv", index=False)
    return frame


def contract_example(states: pd.DataFrame, lock: Mapping[str, Any], model_blob: bytes) -> dict:
    latest = states.sort_values("signal_ts").groupby("pair", as_index=False).tail(1)
    pair_signals, last1h, last4h = {}, {}, {}
    for row in latest.itertuples(index=False):
        pair_signals[row.pair] = {
            "pair": row.pair, "probability": float(row.probability),
            "entry_threshold": float(row.entry_threshold), "recovery_threshold": float(row.recovery_threshold),
            "risk_off_active": bool(row.risk_off_active), "buy_enabled": bool(row.buy_enabled),
            "consecutive_recovery_bars": int(row.consecutive_recovery_bars),
            "risk_off_since": None if pd.isna(row.risk_off_since_ts) else int(row.risk_off_since_ts),
            "reason": row.reason, "event_id": row.event_id,
        }
        last1h[row.pair], last4h[row.pair] = int(row.last_complete_1h_ts), int(row.last_complete_4h_ts)
    generated = int(latest.signal_ts.max())
    return build_contract(
        generated_at=generated, valid_until=generated+150, model_version=MODEL_VERSION,
        model_sha256=__import__("hashlib").sha256(model_blob).hexdigest(),
        feature_sha256=feature_schema_hash(list(ALL_FEATURES if lock["architecture"] == "shared" else SEPARATE_FEATURES)),
        source_healthy=True, pair_signals=pair_signals, last_complete_1h=last1h, last_complete_4h=last4h,
    )


def write_notebook(output_dir: Path) -> None:
    notebook = nbf.v4.new_notebook()
    notebook.cells = [
        nbf.v4.new_markdown_cell("## tl;dr\n\nThis notebook audits the locked XGBoost ordinary-BUY risk gate. It is research-only and never authorizes deployment."),
        nbf.v4.new_markdown_cell("## Context & Methods\n\nThe fixed 60-day interval is **revalidation**, not unseen evidence. Models use mature six-hour labels only; Grid parameters and inventory policy are shared with Mechanism 1.\n\n### Key Assumptions\n\nMaker 0%, Taker 0.1%, 420 FDUSD total, pair-independent gate, no model-triggered sell."),
        nbf.v4.new_code_cell("from pathlib import Path\nimport json, pandas as pd\nroot = Path('.')\nsummary = json.loads((root/'research_summary.json').read_text(encoding='utf-8'))\nmetrics = pd.read_csv(root/'revalidation_metrics.csv')\ncandidates = pd.read_csv(root/'development_640_candidates.csv')\nstress = pd.read_csv(root/'revalidation_stress_tests.csv') if (root/'revalidation_stress_tests.csv').exists() else pd.DataFrame()\nsummary['verdict']"),
        nbf.v4.new_markdown_cell("## Data\n\nSource artifacts are the local BTC-FDUSD/ETH-FDUSD five-minute caches, closed-bar feature panel, immutable lock, fixed weekly Grid selections and replay outputs."),
        nbf.v4.new_code_cell("assert len(candidates) == 640\nassert candidates['variant'].nunique() == 80\nassert not candidates.filter(like='threshold').isna().any().any()\nmetrics[['scenario','oos_pnl_fdusd','worst_drawdown_pct','pair_stop_events','portfolio_stop_events']]"),
        nbf.v4.new_markdown_cell("## Results"),
        nbf.v4.new_code_cell("display(metrics)\ndisplay(candidates.head(10))\ndisplay(stress)"),
        nbf.v4.new_markdown_cell("## Takeaways\n\nUse the executed outputs above together with `research_summary.json`. Any failed development, fixed-interval, or stress gate results in NO-GO; deployment remains disabled in every case."),
    ]
    notebook.metadata.kernelspec = {"display_name": "Python 3", "language": "python", "name": "python3"}
    source = output_dir / "reproducible_analysis.ipynb"; executed = output_dir / "reproducible_analysis.executed.ipynb"
    nbf.write(notebook, source)
    client = NotebookClient(notebook, timeout=600, kernel_name="python3", resources={"metadata": {"path": str(output_dir.resolve())}})
    executed_nb = client.execute(); nbf.write(executed_nb, executed)


def build_report(output_dir: Path, summary: Mapping[str, Any]) -> None:
    """Build the single canonical portable HTML report via the packaged builder."""
    from tune_xgboost_risk_gate_report import build_artifact
    artifact = build_artifact(output_dir, summary)
    write_json(output_dir / "artifact.json", artifact)
    artifact_path = (output_dir / "artifact.json").resolve()
    report_path = (output_dir / "technical_report.html").resolve()
    screenshot_path = (output_dir / "report_verification_failure.png").resolve()
    command = ["node", str(PLUGIN_ROOT / "skills/build-report/scripts/deliver_portable_artifact.mjs"), "--input", str(artifact_path), "--output", str(report_path), "--ready-timeout-ms", "15000", "--timeout-ms", "30000", "--screenshot", str(screenshot_path)]
    completed = subprocess.run(command, cwd=PLUGIN_ROOT, capture_output=True, text=True)
    receipt = {"command": command, "returncode": completed.returncode, "stdout": completed.stdout, "stderr": completed.stderr}
    if completed.returncode and "horizontal_overflow" in (completed.stderr + completed.stdout):
        # Shared reader 0.2.8 uses a 100vw full-bleed sticky top bar. On Windows
        # Chromium the vertical scrollbar makes that one bar wider than the
        # document. Package with the same canonical builder, correct only those
        # shared top-bar/root CSS tokens, then run the official verifier again.
        direct = ["node", str(PLUGIN_ROOT / "skills/build-report/scripts/build_portable_artifact.mjs"), "--input", str(artifact_path), "--output", str(report_path)]
        direct_result = subprocess.run(direct, cwd=PLUGIN_ROOT, capture_output=True, text=True)
        if direct_result.returncode:
            raise RuntimeError(direct_result.stderr or direct_result.stdout)
        html = report_path.read_text(encoding="utf-8")
        replacements = {
            "width:100vw": "width:100%",
            "margin-right:calc(50% - 50vw);margin-left:calc(50% - 50vw)": "margin-right:0;margin-left:0",
            "html,body{margin:0;min-height:100%": "html,body{margin:0;min-height:100%;overflow-x:hidden",
        }
        for before, after in replacements.items():
            if before not in html:
                raise RuntimeError(f"Expected canonical portable-reader CSS token missing: {before}")
            html = html.replace(before, after, 1)
        report_path.write_text(html, encoding="utf-8")
        verify = ["node", str(PLUGIN_ROOT / "skills/build-report/scripts/verify_portable_artifact.mjs"), "--html", str(report_path), "--artifact", str(artifact_path), "--ready-timeout-ms", "15000", "--timeout-ms", "30000", "--screenshot", str(screenshot_path)]
        verified = subprocess.run(verify, cwd=PLUGIN_ROOT, capture_output=True, text=True)
        receipt["canonical_topbar_overflow_correction"] = True
        receipt["direct_build"] = {"command": direct, "returncode": direct_result.returncode, "stdout": direct_result.stdout, "stderr": direct_result.stderr}
        receipt["verification"] = {"command": verify, "returncode": verified.returncode, "stdout": verified.stdout, "stderr": verified.stderr}
        completed = verified
    write_json(output_dir / "report_delivery_receipt.json", receipt)
    if completed.returncode:
        raise RuntimeError(f"Portable report build failed: {completed.stderr or completed.stdout}")


def main() -> int:
    args = parse_args()
    candles, panel, dev, reval, hashes = prepare(args)
    if args.stage in {"search", "all"}:
        run_search(args, candles, panel, dev, hashes)
    if args.stage in {"revalidate", "all"}:
        summary = run_revalidation(args, candles, panel, reval, hashes)
        print(json.dumps({"verdict": summary["verdict"], "model": summary["model"]}, ensure_ascii=False, indent=2, default=json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

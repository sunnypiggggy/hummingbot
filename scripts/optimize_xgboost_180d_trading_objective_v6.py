#!/usr/bin/env python3
"""Search XGBoost Grid risk gates on 180-day profit and stitched drawdown.

The search is intentionally diagnostic and in-sample with respect to trading
performance: all 180 replay days participate in the final profit/drawdown
ranking.  Hourly probabilities remain walk-forward and label-maturity purged,
but the selected configuration is not fresh out-of-sample evidence and never
authorizes deployment or order submission.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from dataclasses import asdict
from itertools import product
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

import backtest_xgboost_dual_risk_gate_180d as dual
import backtest_xgboost_long_risk_gate_180d as base
import optimize_xgboost_dual_risk_gate_180d_v5 as v5
from compare_independent_gate_ml_stops import HOUR, PAIRS, load_candles
from tune_xgboost_momentum_stop_v2 import sha256_file, write_json, xgb_configurations


MODEL_VERSION = "xgboost-grid-trading-objective-v6"
OUTPUT_DIR = Path("results/backtests/xgboost_180d_trading_objective_v6")
SOURCE_V5 = Path("results/backtests/xgboost_dual_risk_gate_180d_v5")
PLUGIN_ROOT = Path(
    r"C:\Users\sunny\.codex\plugins\cache\openai-curated-remote\data-analytics\0.2.8-13ceeea1f599"
)
PRESCREEN_FOLDS = (4, 8, 12, 16)
FINALISTS_PER_ARCHITECTURE = 2
PROFIT_WEIGHT = 0.50
DRAW_DOWN_WEIGHT = 0.50
INCUMBENTS = {
    "long_persistent_72h": ("xgb_00", "separate"),
    "short_spike_1h_24h": ("xgb_30", "shared"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", choices=("prescreen", "search", "report", "plot", "all"), default="all"
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--cache-dir", type=Path, default=Path("data/backtesting_candles"))
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--source-dir", type=Path, default=SOURCE_V5)
    parser.add_argument("--plugin-root", type=Path, default=PLUGIN_ROOT)
    return parser.parse_args()


def safe_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    return json.loads(frame.replace({np.nan: None, np.inf: None, -np.inf: None}).to_json(orient="records"))


def gate_id(strategy: str, gate: v5.GateParameters) -> str:
    prefix = "long" if strategy == "long_persistent_72h" else "short"
    maximum = "none" if gate.maximum_hours is None else str(gate.maximum_hours)
    return (
        f"{prefix}-q{gate.entry_quantile:g}-r{gate.recovery_quantile:g}"
        f"-e{gate.entry_bars}-x{gate.recovery_bars}-min{gate.minimum_hours}"
        f"-max{maximum}-cd{gate.cooldown_hours}"
    )


def long_gate_candidates() -> list[v5.GateParameters]:
    gates = [
        v5.GateParameters(entry, 0.60, 1, 4, duration, duration, cooldown)
        for entry, duration, cooldown in product(
            (0.85, 0.90, 0.925, 0.95), (120, 168), (240, 480)
        )
    ]
    gates.extend(
        v5.GateParameters(entry, 0.60, 2, 4, duration, duration, 480)
        for entry, duration in product((0.90, 0.925), (120, 168))
    )
    return gates


def short_gate_candidates() -> list[v5.GateParameters]:
    return [
        v5.GateParameters(entry, 0.90, entry_bars, 1, 1, maximum, 0)
        for entry, entry_bars, maximum in product(
            (0.95, 0.975, 0.985, 0.99), (1, 2), (6, 12, 24)
        )
    ]


def add_pareto_and_score(frame: pd.DataFrame) -> pd.DataFrame:
    """Rank candidates with equal-weight profit/drawdown percentile utility."""
    ranked = frame.copy()
    ranked["profit_percentile"] = ranked.oos_pnl_fdusd.rank(method="average", pct=True)
    ranked["drawdown_percentile"] = ranked.stitched_max_drawdown_pct.rank(method="average", pct=True)
    ranked["trading_objective_score"] = (
        PROFIT_WEIGHT * ranked.profit_percentile
        + DRAW_DOWN_WEIGHT * ranked.drawdown_percentile
    )
    points = ranked[["oos_pnl_fdusd", "stitched_max_drawdown_pct"]].to_numpy(float)
    pareto = []
    for index, point in enumerate(points):
        weak = (points[:, 0] >= point[0]) & (points[:, 1] >= point[1])
        strict = (points[:, 0] > point[0]) | (points[:, 1] > point[1])
        pareto.append(not bool(np.any(weak & strict & (np.arange(len(points)) != index))))
    ranked["pareto_front"] = pareto
    sort_columns = [
        "trading_objective_score", "oos_pnl_fdusd", "stitched_max_drawdown_pct",
        "portfolio_stop_events", "pair_stop_events", "risk_off_pair_hours",
    ]
    ascending = [False, False, False, True, True, True]
    ranked = ranked.sort_values(sort_columns, ascending=ascending).reset_index(drop=True)
    ranked["rank"] = np.arange(1, len(ranked) + 1)
    return ranked


def stitched_drawdown(result: Mapping[str, Any]) -> float:
    curve = result["equity"].sort_values(["fold", "timestamp"])
    if curve.empty:
        return 0.0
    equity = base.INITIAL_EQUITY + curve.cumulative_oos_pnl.astype(float)
    return float((equity / equity.cummax() - 1.0).min() * 100.0)


def trading_row(result: Mapping[str, Any]) -> dict[str, Any]:
    row = dict(result["summary"])
    row["stitched_max_drawdown_pct"] = stitched_drawdown(result)
    return row


def model_prescreen(
    panel: pd.DataFrame, selections: pd.DataFrame, output_path: Path, resume: bool,
) -> pd.DataFrame:
    configs = {item["config_id"]: item for item in xgb_configurations()}
    blocks = list(selections[selections.fold.isin(PRESCREEN_FOLDS)].itertuples(index=False))
    existing = pd.read_csv(output_path) if resume and output_path.exists() else pd.DataFrame()
    done = set()
    if not existing.empty:
        done = set(zip(existing.strategy, existing.config_id, existing.architecture))
    rows = existing.to_dict("records") if not existing.empty else []
    total = len(dual.STRATEGIES) * len(configs) * 2
    for strategy, config_id, architecture in product(
        dual.STRATEGIES, configs, ("shared", "separate")
    ):
        if (strategy, config_id, architecture) in done:
            continue
        working = v5.working_target(panel, strategy)
        parts = []
        for block in blocks:
            predicted, _, audits = v5.fit_variant_block(
                working, block, configs[config_id], architecture
            )
            if any(item["last_mature_label_ready_ts"] > item["train_cutoff_ts"] for item in audits):
                raise AssertionError("Prescreen used an immature label")
            if any(item["last_calibration_signal_ts"] >= item["first_test_signal_ts"] for item in audits):
                raise AssertionError("Prescreen calibration overlaps test")
            parts.append(predicted)
        predictions = pd.concat(parts, ignore_index=True)
        pair_metrics = []
        for pair, group in predictions.groupby("pair"):
            target = group.target.astype(int)
            probability = group.probability.astype(float)
            auc = float(roc_auc_score(target, probability))
            ap = float(average_precision_score(target, probability))
            pair_metrics.append((pair, auc, ap, float(target.mean())))
        mean_auc = float(np.mean([item[1] for item in pair_metrics]))
        min_auc = float(np.min([item[1] for item in pair_metrics]))
        mean_lift = float(np.mean([item[2] / item[3] for item in pair_metrics]))
        diagnostic_score = 0.55 * mean_auc + 0.25 * min_auc + 0.20 * min(mean_lift, 2.0) / 2.0
        row = {
            "strategy": strategy, "config_id": config_id, "architecture": architecture,
            "diagnostic_score": diagnostic_score, "mean_auc": mean_auc,
            "min_pair_auc": min_auc, "mean_average_precision_lift": mean_lift,
            **{f"{pair[:3]}_auc": auc for pair, auc, _, _ in pair_metrics},
            **{key: value for key, value in configs[config_id].items() if key != "config_id"},
        }
        rows.append(row)
        pd.DataFrame(rows).to_csv(output_path, index=False)
        print(
            f"PRESCREEN {len(rows):03d}/{total} {strategy} {config_id} {architecture} "
            f"score={diagnostic_score:.4f}", flush=True,
        )
    return pd.DataFrame(rows).sort_values(
        ["strategy", "diagnostic_score"], ascending=[True, False]
    ).reset_index(drop=True)


def select_finalists(prescreen: pd.DataFrame) -> dict[str, list[dict[str, Any]]]:
    configs = {item["config_id"]: item for item in xgb_configurations()}
    finalists: dict[str, list[dict[str, Any]]] = {}
    for strategy, strategy_rows in prescreen.groupby("strategy"):
        chosen = []
        for architecture, group in strategy_rows.groupby("architecture"):
            selected = group.nlargest(FINALISTS_PER_ARCHITECTURE, "diagnostic_score")
            chosen.extend(selected.to_dict("records"))
        incumbent_config, incumbent_architecture = INCUMBENTS[strategy]
        if not any(
            item["config_id"] == incumbent_config and item["architecture"] == incumbent_architecture
            for item in chosen
        ):
            incumbent = strategy_rows[
                (strategy_rows.config_id == incumbent_config)
                & (strategy_rows.architecture == incumbent_architecture)
            ].iloc[0]
            chosen.append(incumbent.to_dict())
        finalists[strategy] = [
            {
                "strategy": strategy,
                "config_id": str(item["config_id"]),
                "architecture": str(item["architecture"]),
                "diagnostic_score": float(item["diagnostic_score"]),
                "configuration": configs[str(item["config_id"])],
            }
            for item in chosen
        ]
    return finalists


def prediction_paths(output_dir: Path, strategy: str, config_id: str, architecture: str) -> tuple[Path, Path]:
    stem = f"{strategy}__{config_id}__{architecture}"
    folder = output_dir / "prediction_cache"
    return folder / f"{stem}.csv.gz", folder / f"{stem}.audit.csv"


def train_variant_walk_forward(
    panel: pd.DataFrame, selections: pd.DataFrame, finalist: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    strategy = str(finalist["strategy"])
    working = v5.working_target(panel, strategy)
    predictions, audit_rows = [], []
    for block in selections.itertuples(index=False):
        predicted, calibration, audits = v5.fit_variant_block(
            working, block, finalist["configuration"], str(finalist["architecture"])
        )
        test = working[
            (working.signal_ts >= int(block.test_start))
            & (working.signal_ts < int(block.test_end))
        ].copy()
        out = test[["pair", "signal_ts", "target", "label_ready_ts"]].copy()
        probability = predicted.set_index(["pair", "signal_ts"]).probability
        out["probability"] = [
            probability.loc[(pair, timestamp)]
            for pair, timestamp in zip(out.pair, out.signal_ts)
        ]
        out["strategy"] = strategy
        out["strategy_label"] = dual.STRATEGIES[strategy]["label"]
        out["config_id"] = finalist["config_id"]
        out["architecture"] = finalist["architecture"]
        out["fold"] = int(block.fold)
        for quantile in v5.THRESHOLD_QUANTILES:
            thresholds = {
                pair: float(calibration.loc[calibration.pair == pair, "probability"].quantile(quantile))
                for pair in PAIRS
            }
            out[v5.quantile_column(quantile)] = out.pair.map(thresholds)
        predictions.append(out)
        audit_rows.extend({
            "strategy": strategy, "config_id": finalist["config_id"],
            "architecture": finalist["architecture"], "fold": int(block.fold), **item,
        } for item in audits)
        print(
            f"PREDICT {strategy} {finalist['config_id']} {finalist['architecture']} "
            f"fold={int(block.fold):02d}/{len(selections)}", flush=True,
        )
    result, audit = pd.concat(predictions, ignore_index=True), pd.DataFrame(audit_rows)
    if not np.isfinite(result.probability).all() or not result.probability.between(0, 1).all():
        raise AssertionError("Walk-forward probabilities are invalid")
    if not (audit.last_mature_label_ready_ts <= audit.train_cutoff_ts).all():
        raise AssertionError("Walk-forward fitting used immature labels")
    if not (audit.last_calibration_signal_ts < audit.first_test_signal_ts).all():
        raise AssertionError("Walk-forward calibration overlaps test")
    return result, audit


def load_or_train_predictions(
    panel: pd.DataFrame, selections: pd.DataFrame, finalists: Mapping[str, Sequence[Mapping[str, Any]]],
    output_dir: Path, resume: bool,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    output: dict[str, pd.DataFrame] = {}
    audits = []
    for strategy in dual.STRATEGIES:
        for finalist in finalists[strategy]:
            key = f"{strategy}|{finalist['config_id']}|{finalist['architecture']}"
            prediction_path, audit_path = prediction_paths(
                output_dir, strategy, str(finalist["config_id"]), str(finalist["architecture"])
            )
            if resume and prediction_path.exists() and audit_path.exists():
                prediction = pd.read_csv(prediction_path)
                audit = pd.read_csv(audit_path)
            else:
                prediction, audit = train_variant_walk_forward(panel, selections, finalist)
                prediction_path.parent.mkdir(parents=True, exist_ok=True)
                prediction.to_csv(prediction_path, index=False, compression="gzip")
                audit.to_csv(audit_path, index=False)
            output[key] = prediction
            audits.append(audit)
    return output, pd.concat(audits, ignore_index=True)


def row_gate(row: Mapping[str, Any], prefix: str = "") -> v5.GateParameters:
    maximum = row[f"{prefix}maximum_hours"]
    if pd.isna(maximum):
        maximum = None
    return v5.GateParameters(
        float(row[f"{prefix}entry_quantile"]), float(row[f"{prefix}recovery_quantile"]),
        int(row[f"{prefix}entry_bars"]), int(row[f"{prefix}recovery_bars"]),
        int(row[f"{prefix}minimum_hours"]), None if maximum is None else int(maximum),
        int(row[f"{prefix}cooldown_hours"]),
    )


def search_single_channels(
    candles: Mapping[str, pd.DataFrame], selections: pd.DataFrame,
    finalists: Mapping[str, Sequence[Mapping[str, Any]]], predictions: Mapping[str, pd.DataFrame],
    baseline_metrics: Mapping[str, Any], output_path: Path, resume: bool,
) -> pd.DataFrame:
    existing = pd.read_csv(output_path) if resume and output_path.exists() else pd.DataFrame()
    done = set(existing.candidate_id) if not existing.empty else set()
    rows = existing.to_dict("records") if not existing.empty else []
    gates_by_strategy = {
        "long_persistent_72h": long_gate_candidates(),
        "short_spike_1h_24h": short_gate_candidates(),
    }
    for strategy in dual.STRATEGIES:
        for finalist in finalists[strategy]:
            model_key = f"{strategy}|{finalist['config_id']}|{finalist['architecture']}"
            for gate in gates_by_strategy[strategy]:
                candidate_id = f"{model_key}|{gate_id(strategy, gate)}"
                if candidate_id in done:
                    continue
                result = v5.replay(
                    candles, selections, predictions[model_key], {strategy: gate},
                    (strategy,), candidate_id,
                )
                row = {
                    "candidate_id": candidate_id,
                    "channel": "long" if strategy == "long_persistent_72h" else "short",
                    "strategy": strategy, "model_key": model_key,
                    "config_id": finalist["config_id"], "architecture": finalist["architecture"],
                    "diagnostic_score": finalist["diagnostic_score"], "gate_id": gate_id(strategy, gate),
                    **asdict(gate), **trading_row(result),
                }
                row["beats_baseline_profit"] = row["oos_pnl_fdusd"] > baseline_metrics["oos_pnl_fdusd"]
                row["beats_baseline_drawdown"] = row["stitched_max_drawdown_pct"] >= baseline_metrics["stitched_max_drawdown_pct"]
                rows.append(row)
            pd.DataFrame(rows).to_csv(output_path, index=False)
            print(f"TRADE SEARCH completed {model_key}", flush=True)
    ranked_parts = []
    result = pd.DataFrame(rows)
    for channel, group in result.groupby("channel"):
        ranked_parts.append(add_pareto_and_score(group))
    ranked = pd.concat(ranked_parts, ignore_index=True)
    ranked = ranked.sort_values(["channel", "rank"]).reset_index(drop=True)
    ranked.to_csv(output_path, index=False)
    return ranked


def search_joint_channels(
    candles: Mapping[str, pd.DataFrame], selections: pd.DataFrame,
    predictions: Mapping[str, pd.DataFrame], single: pd.DataFrame,
    baseline_metrics: Mapping[str, Any], output_path: Path, resume: bool,
) -> pd.DataFrame:
    long_top = single[single.channel == "long"].nsmallest(5, "rank")
    short_top = single[single.channel == "short"].nsmallest(5, "rank")
    existing = pd.read_csv(output_path) if resume and output_path.exists() else pd.DataFrame()
    done = set(existing.candidate_id) if not existing.empty else set()
    rows = existing.to_dict("records") if not existing.empty else []
    for long_row, short_row in product(
        long_top.to_dict("records"), short_top.to_dict("records")
    ):
        candidate_id = f"dual|{long_row['candidate_id']}||{short_row['candidate_id']}"
        if candidate_id in done:
            continue
        long_prediction = predictions[str(long_row["model_key"])]
        short_prediction = predictions[str(short_row["model_key"])]
        combined = pd.concat([long_prediction, short_prediction], ignore_index=True)
        gates = {
            "long_persistent_72h": row_gate(long_row),
            "short_spike_1h_24h": row_gate(short_row),
        }
        result = v5.replay(
            candles, selections, combined, gates, tuple(dual.STRATEGIES), candidate_id
        )
        row = {
            "candidate_id": candidate_id, "channel": "dual",
            "long_candidate_id": long_row["candidate_id"],
            "short_candidate_id": short_row["candidate_id"],
            "long_model_key": long_row["model_key"], "short_model_key": short_row["model_key"],
            **{f"long_{key}": value for key, value in asdict(gates["long_persistent_72h"]).items()},
            **{f"short_{key}": value for key, value in asdict(gates["short_spike_1h_24h"]).items()},
            **trading_row(result),
        }
        row["beats_baseline_profit"] = row["oos_pnl_fdusd"] > baseline_metrics["oos_pnl_fdusd"]
        row["beats_baseline_drawdown"] = row["stitched_max_drawdown_pct"] >= baseline_metrics["stitched_max_drawdown_pct"]
        rows.append(row)
        pd.DataFrame(rows).to_csv(output_path, index=False)
        print(
            f"JOINT {len(rows):02d}/25 pnl={row['oos_pnl_fdusd']:+.3f} "
            f"dd={row['stitched_max_drawdown_pct']:.3f}%", flush=True,
        )
    ranked = add_pareto_and_score(pd.DataFrame(rows))
    ranked.to_csv(output_path, index=False)
    return ranked


def detailed_result_for_single(
    row: Mapping[str, Any], candles: Mapping[str, pd.DataFrame], selections: pd.DataFrame,
    predictions: Mapping[str, pd.DataFrame], scenario: str,
) -> dict[str, Any]:
    strategy = str(row["strategy"])
    return v5.replay(
        candles, selections, predictions[str(row["model_key"])],
        {strategy: row_gate(row)}, (strategy,), scenario,
    )


def detailed_result_for_dual(
    row: Mapping[str, Any], candles: Mapping[str, pd.DataFrame], selections: pd.DataFrame,
    predictions: Mapping[str, pd.DataFrame], scenario: str,
) -> dict[str, Any]:
    combined = pd.concat([
        predictions[str(row["long_model_key"])], predictions[str(row["short_model_key"])],
    ], ignore_index=True)
    gates = {
        "long_persistent_72h": row_gate(row, "long_"),
        "short_spike_1h_24h": row_gate(row, "short_"),
    }
    return v5.replay(candles, selections, combined, gates, tuple(dual.STRATEGIES), scenario)


def finalize_results(
    candles: Mapping[str, pd.DataFrame], selections: pd.DataFrame,
    predictions: Mapping[str, pd.DataFrame], single: pd.DataFrame, joint: pd.DataFrame,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any], pd.DataFrame]:
    best_long = single[single.channel == "long"].nsmallest(1, "rank").iloc[0].to_dict()
    best_short = single[single.channel == "short"].nsmallest(1, "rank").iloc[0].to_dict()
    best_dual = joint.nsmallest(1, "rank").iloc[0].to_dict()
    baseline = base.replay(
        candles, selections, scenario="Mechanism 1 (BTC ROC/SQZMOM)",
        baseline_gate=base.mechanism1_gate(candles), record_details=True,
    )
    results = {
        "Mechanism 1 (BTC ROC/SQZMOM)": baseline,
        "XGBoost 180d best long": detailed_result_for_single(
            best_long, candles, selections, predictions, "XGBoost 180d best long"
        ),
        "XGBoost 180d best short": detailed_result_for_single(
            best_short, candles, selections, predictions, "XGBoost 180d best short"
        ),
        "XGBoost 180d best dual": detailed_result_for_dual(
            best_dual, candles, selections, predictions, "XGBoost 180d best dual"
        ),
    }
    metric_rows = []
    for scenario, result in results.items():
        metric_rows.append({"scenario": scenario, **trading_row(result)})
    metrics = pd.DataFrame(metric_rows)
    lock = {
        "schema": "xgboost-180d-trading-objective-lock-v1",
        "model_version": MODEL_VERSION,
        "selection_basis": {
            "period": "full_180d_in_sample_trading_objective",
            "profit_weight": PROFIT_WEIGHT,
            "stitched_drawdown_weight": DRAW_DOWN_WEIGHT,
            "tie_breakers": ["higher_profit", "shallower_drawdown", "fewer_stops", "less_risk_off"],
        },
        "best_long": best_long,
        "best_short": best_short,
        "best_dual": best_dual,
        "deployment_authorized": False,
    }
    return results, lock, metrics


def write_final_artifacts(
    output_dir: Path, results: Mapping[str, Mapping[str, Any]], lock: Mapping[str, Any],
    metrics: pd.DataFrame, prescreen: pd.DataFrame, single: pd.DataFrame, joint: pd.DataFrame,
    audit: pd.DataFrame, source_dir: Path,
) -> dict[str, Any]:
    write_json(output_dir / "locked_configuration.json", lock)
    metrics.to_csv(output_dir / "final_metrics.csv", index=False)
    prescreen.to_csv(output_dir / "model_prescreen_40x2x2.csv", index=False)
    single.to_csv(output_dir / "single_channel_trading_search.csv", index=False)
    joint.to_csv(output_dir / "dual_channel_trading_search.csv", index=False)
    audit.to_csv(output_dir / "walk_forward_training_audit.csv", index=False)
    weekly, equity, events, intervals = [], [], [], []
    for scenario, result in results.items():
        weekly.append(result["weekly"])
        equity.append(result["equity"])
        if "events" in result and not result["events"].empty:
            events.append(result["events"].assign(scenario=scenario))
        if "intervals" in result and not result["intervals"].empty:
            intervals.append(result["intervals"].assign(scenario=scenario))
    pd.concat(weekly, ignore_index=True).to_csv(output_dir / "final_weekly_results.csv", index=False)
    pd.concat(equity, ignore_index=True).to_csv(output_dir / "final_equity_curves.csv.gz", index=False, compression="gzip")
    pd.concat(events, ignore_index=True).to_csv(output_dir / "final_risk_events.csv", index=False) if events else None
    pd.concat(intervals, ignore_index=True).to_csv(output_dir / "final_risk_intervals.csv", index=False) if intervals else None
    baseline = metrics[metrics.scenario == "Mechanism 1 (BTC ROC/SQZMOM)"].iloc[0]
    winner = metrics[metrics.scenario == "XGBoost 180d best dual"].iloc[0]
    gates = {
        "passed_profit": bool(winner.oos_pnl_fdusd > baseline.oos_pnl_fdusd),
        "passed_stitched_drawdown": bool(winner.stitched_max_drawdown_pct >= baseline.stitched_max_drawdown_pct),
        "no_portfolio_stops": bool(int(winner.portfolio_stop_events) == 0),
        "no_momentum_stop_sales": bool(int(winner.momentum_stop_exits) == 0),
    }
    summary = {
        "schema": "xgboost-180d-trading-objective-summary-v1",
        "model_version": MODEL_VERSION,
        "evidence_status": "full_180d_in_sample_parameter_optimization",
        "deployment_authorized": False,
        "verdict": "DIAGNOSTIC_ONLY" if all(gates.values()) else "NO-GO",
        "metric_definition": {
            "profit": "sum of the 26 weekly Grid-fold net PnL values in FDUSD",
            "stitched_max_drawdown_pct": "max drawdown of 420 FDUSD plus cumulative weekly-fold PnL over 180 days",
            "score": "50% profit percentile + 50% stitched-drawdown percentile",
            "worst_drawdown_pct": "worst within-week drawdown retained as a secondary metric",
        },
        "baseline": safe_records(metrics[metrics.scenario.str.startswith("Mechanism")])[0],
        "best_dual": safe_records(metrics[metrics.scenario == "XGBoost 180d best dual"])[0],
        "acceptance": gates,
        "candidate_counts": {
            "prescreen": int(len(prescreen)), "single_channel": int(len(single)),
            "dual_channel": int(len(joint)),
        },
        "lock": lock,
        "no_lookahead_checks": {
            "all_labels_mature": bool((audit.last_mature_label_ready_ts <= audit.train_cutoff_ts).all()),
            "calibration_precedes_test": bool((audit.last_calibration_signal_ts < audit.first_test_signal_ts).all()),
        },
        "source_hashes": {
            "panel": sha256_file(source_dir / "dual_target_feature_panel.csv.gz"),
            "grid": sha256_file(source_dir / "grid_selections.csv"),
        },
        "limitations": [
            "The same 180-day trading path is used for parameter selection and evaluation; this is in-sample optimization, not fresh out-of-sample evidence.",
            "Forty XGBoost configurations are classification-prescreened on four walk-forward folds; only the top two per architecture and the incumbent receive full trading replay.",
            "The stitched drawdown assumes a 420 FDUSD reference plus cumulative weekly PnL while inventory itself still reinitializes every week.",
        ],
    }
    write_json(output_dir / "summary.json", summary)
    return summary


def file_source(source_id: str, label: str, path: str) -> dict[str, Any]:
    reader = "read_json_auto" if path.endswith(".json") else "read_csv_auto"
    return {
        "id": source_id, "label": label, "path": path,
        "query": {
            "engine": "duckdb", "language": "sql",
            "sql": f"SELECT * FROM {reader}('{path}')",
            "description": f"Reproducible local-file read for {label}.",
            "tables_used": [path],
        },
    }


def build_report_artifact(output_dir: Path, summary: Mapping[str, Any]) -> dict[str, Any]:
    metrics = pd.read_csv(output_dir / "final_metrics.csv")
    single = pd.read_csv(output_dir / "single_channel_trading_search.csv")
    joint = pd.read_csv(output_dir / "dual_channel_trading_search.csv")
    curves = pd.read_csv(output_dir / "final_equity_curves.csv.gz")
    weekly = pd.read_csv(output_dir / "final_weekly_results.csv")
    baseline = metrics[metrics.scenario.str.startswith("Mechanism")].iloc[0]
    winner = metrics[metrics.scenario == "XGBoost 180d best dual"].iloc[0]
    primary_scenarios = ["Mechanism 1 (BTC ROC/SQZMOM)", "XGBoost 180d best dual"]
    curves = curves[curves.scenario.isin(primary_scenarios)].copy()
    weekly = weekly[weekly.scenario.isin(primary_scenarios)].copy()
    candidates = pd.concat([
        single.assign(search_scope=single.channel), joint.assign(search_scope="dual")
    ], ignore_index=True, sort=False)
    candidates = add_pareto_and_score(candidates)
    candidates["display_id"] = candidates.apply(
        lambda row: f"{row.search_scope} #{int(row['rank'])}", axis=1
    )
    candidates["profit_fdusd"] = candidates.oos_pnl_fdusd.astype(float)
    candidates["drawdown_pct"] = candidates.stitched_max_drawdown_pct.astype(float)
    top = candidates.nsmallest(20, "rank").copy()
    curve_rows = []
    drawdown_rows = []
    for scenario, group in curves.groupby("scenario"):
        group = group.sort_values(["fold", "timestamp"])
        # The portable report caps each dataset at 2,000 rows.  Twelve-hour
        # sampling preserves roughly 360 observations per scenario while the
        # full five-minute curves remain available in the source CSV.
        sampled = group[group.groupby("fold").cumcount().mod(144).eq(0)].copy()
        sampled["time"] = pd.to_datetime(sampled.timestamp, unit="s", utc=True).astype(str)
        sampled["stitched_equity"] = base.INITIAL_EQUITY + sampled.cumulative_oos_pnl.astype(float)
        running_peak = sampled.stitched_equity.cummax()
        sampled["drawdown_pct"] = (sampled.stitched_equity / running_peak - 1.0) * 100.0
        curve_rows.extend(safe_records(sampled[["time", "scenario", "stitched_equity"]]))
        drawdown_rows.extend(safe_records(sampled[["time", "scenario", "drawdown_pct"]]))
    weekly["week"] = weekly.fold.map(lambda value: f"W{int(value):02d}")
    weekly["pnl_fdusd"] = weekly.net_pnl_quote.astype(float)
    sources = [
        file_source("summary", "180-day optimization summary", "summary.json"),
        file_source("metrics", "Final Grid replay metrics", "final_metrics.csv"),
        file_source("search", "Trading-objective candidate search", "single_channel_trading_search.csv"),
        file_source("joint", "Dual-channel candidate search", "dual_channel_trading_search.csv"),
        file_source("equity", "Final stitched equity curves", "final_equity_curves.csv.gz"),
        file_source("weekly", "Final weekly Grid results", "final_weekly_results.csv"),
        {"id": "implementation", "label": "Reproducible optimizer", "path": "scripts/optimize_xgboost_180d_trading_objective_v6.py"},
    ]
    cards = [
        {"id": "pnl", "dataset": "headline", "sourceId": "metrics", "description": "Selected dual-channel net PnL across 26 weekly Grid folds.", "metrics": [{"label": "180d net PnL", "field": "pnl", "format": "number", "unit": " FDUSD", "signed": True}, {"label": "Mechanism 1", "field": "baseline_pnl", "format": "number", "unit": " FDUSD", "signed": True}]},
        {"id": "dd", "dataset": "headline", "sourceId": "metrics", "description": "Maximum drawdown of the stitched 420 FDUSD reference equity path.", "metrics": [{"label": "180d stitched max DD", "field": "drawdown", "format": "number", "unit": "%", "signed": True}, {"label": "Mechanism 1", "field": "baseline_drawdown", "format": "number", "unit": "%", "signed": True}]},
        {"id": "stops", "dataset": "headline", "sourceId": "metrics", "description": "Safety stops remain guardrails and are not included in the 50/50 score.", "metrics": [{"label": "Portfolio stops", "field": "portfolio_stops", "format": "integer"}, {"label": "Pair stops", "field": "pair_stops", "format": "integer"}]},
    ]
    charts = [
        {"id": "pareto", "title": "180天盈利与拼接最大回撤", "subtitle": "每个点是一组完整Grid回放；越靠右上越好。", "type": "scatter", "dataset": "candidates", "sourceId": "search", "encodings": {"x": {"field": "drawdown_pct", "type": "quantitative", "label": "Stitched max drawdown %"}, "y": {"field": "profit_fdusd", "type": "quantitative", "label": "Net PnL FDUSD"}, "color": {"field": "search_scope", "type": "nominal", "label": "Search scope"}}, "layout": "full"},
        {"id": "ranking", "title": "盈利/回撤综合分前20名", "subtitle": "综合分为盈利百分位与回撤百分位各50%。", "type": "bar", "dataset": "top_candidates", "sourceId": "search", "encodings": {"x": {"field": "display_id", "type": "ordinal", "label": "Candidate rank"}, "y": {"field": "trading_objective_score", "type": "quantitative", "label": "Trading objective score"}, "color": {"field": "search_scope", "type": "nominal", "label": "Search scope"}}, "layout": "full"},
        {"id": "equity", "title": "180天拼接权益路径", "subtitle": "以420 FDUSD为参考起点，叠加每周重新初始化Grid的累计盈亏。", "type": "line", "dataset": "equity", "sourceId": "equity", "encodings": {"x": {"field": "time", "type": "temporal", "label": "UTC"}, "y": {"field": "stitched_equity", "type": "quantitative", "label": "Reference equity FDUSD"}, "color": {"field": "scenario", "type": "nominal", "label": "Scenario"}}, "layout": "full"},
        {"id": "drawdown", "title": "180天拼接回撤路径", "subtitle": "同一时间点越接近0%越好；该图对应主优化回撤指标。", "type": "line", "dataset": "drawdown", "sourceId": "equity", "encodings": {"x": {"field": "time", "type": "temporal", "label": "UTC"}, "y": {"field": "drawdown_pct", "type": "quantitative", "label": "Drawdown %"}, "color": {"field": "scenario", "type": "nominal", "label": "Scenario"}}, "layout": "full"},
        {"id": "weekly", "title": "26个周折盈亏", "subtitle": "每周420 FDUSD重新初始化，展示策略差异的时间分布。", "type": "bar", "dataset": "weekly", "sourceId": "weekly", "encodings": {"x": {"field": "week", "type": "ordinal", "label": "Fold"}, "y": {"field": "pnl_fdusd", "type": "quantitative", "label": "PnL FDUSD"}, "color": {"field": "scenario", "type": "nominal", "label": "Scenario"}}, "layout": "full"},
    ]
    # Exact candidate lookup stays in the source CSV.  A seven-column native
    # table caused the packaged reader to create page-level horizontal
    # overflow; the ranked chart carries the reader-facing comparison.
    tables: list[dict[str, Any]] = []
    title = "XGBoost 180天盈利/回撤参数搜索"
    verdict = summary["verdict"]
    blocks = [
        {"id": "title", "type": "markdown", "body": f"# {title}"},
        {"id": "summary", "type": "markdown", "sourceId": "summary", "body": f"## 技术结论：{verdict}\n\n本轮直接用完整180天Grid净盈利和拼接最大回撤选择参数。最终双通道净盈利为 **{winner.oos_pnl_fdusd:+.3f} FDUSD**，拼接最大回撤为 **{winner.stitched_max_drawdown_pct:.3f}%**；机制1分别为 **{baseline.oos_pnl_fdusd:+.3f} FDUSD** 和 **{baseline.stitched_max_drawdown_pct:.3f}%**。由于同一区间同时用于选择和评价，结果仅是区间内诊断，不是样本外证据。"},
        {"id": "headline", "type": "metric-strip", "cardIds": ["pnl", "dd", "stops"]},
        {"id": "finding", "type": "markdown", "sourceId": "joint", "body": "## 盈利与回撤共同决定胜者\n\n散点图使用完全相同的180天、Grid参数、成本与预算。右上方向表示盈利更高且回撤更浅；Pareto前沿候选不会被另一候选在两个指标上同时支配。综合分仅用于从前沿附近作确定性选择。"},
        {"id": "pareto_block", "type": "chart", "chartId": "pareto", "layout": "full"},
        {"id": "rank_note", "type": "markdown", "sourceId": "search", "body": "## 排名反映交易结果，不再由AUC主导\n\n40组XGBoost参数仍先用四个无前视周折做计算量预筛；每种标签、每种共享/独立架构的前两名和上一版胜者进入完整180天Grid回放。最终排名只使用盈利和拼接回撤，停止次数仅用于平分。"},
        {"id": "ranking_block", "type": "chart", "chartId": "ranking", "layout": "full"},
        {"id": "path_note", "type": "markdown", "sourceId": "equity", "body": "## 权益和回撤路径揭示结果集中在哪些周\n\n权益图将26个每周重置的Grid盈亏拼接到420 FDUSD参考权益上；回撤图使用同一路径计算。这能回答180天累计盈利/回撤目标，但不代表真实账户在周边界连续持仓。"},
        {"id": "equity_block", "type": "chart", "chartId": "equity", "layout": "full"},
        {"id": "drawdown_block", "type": "chart", "chartId": "drawdown", "layout": "full"},
        {"id": "weekly_note", "type": "markdown", "sourceId": "weekly", "body": "## 周度分布用于检查单周主导\n\n逐周柱形图保留全部26个周折。若总收益由少数周贡献、而多数周恶化，则即使综合分领先也不能视为稳健。"},
        {"id": "weekly_block", "type": "chart", "chartId": "weekly", "layout": "full"},
        {"id": "scope", "type": "markdown", "sourceId": "implementation", "body": "## 范围、指标与模型规格\n\n市场为Binance Spot BTC-FDUSD与ETH-FDUSD，UTC时间，5分钟数据生成完整1小时/4小时特征。模型标签仍为120小时持续下跌和1小时/24小时快速插针；预测按周扩展训练，且只使用在截止点前成熟的标签。Grid保持每对200 FDUSD、20 FDUSD组合储备、Maker 0%、风险Taker 0.1%、2小时撤单和既有库存退出/停止机制。"},
        {"id": "method", "type": "markdown", "sourceId": "search", "body": "## 方法：两阶段确定性搜索\n\n第一阶段比较40组固定XGBoost参数、共享/独立架构和两类标签，并只用四个分散周折的AUC/AP作计算量预筛。第二阶段对入围模型生成完整26周无前视概率，分别搜索长期20组与短期24组Risk-off门，再组合各自前5名形成25个双通道候选。每个候选都运行完整Grid会计。"},
        {"id": "limits", "type": "markdown", "sourceId": "summary", "body": "## 限制与稳健性\n\n本轮最大的限制是区间内优化：180天既是选择集也是指标展示集，会高估表现。分类预筛也可能淘汰AUC一般但交易结果更好的配置。拼接回撤基于每周重置后的累计盈亏，并非连续库存账户回撤。所有预测训练和阈值校准仍通过标签成熟与时间边界检查，Risk-off只控制普通BUY，不产生即时Taker卖出。"},
        {"id": "next", "type": "markdown", "body": "## 建议下一步\n\n冻结本轮胜者，只在未来至少8个全新周折上验证；不再用这些新周折调参。只有未来区间同时保持正收益、回撤不劣于机制1且无组合停止，才考虑进入联合运行验证。"},
        {"id": "questions", "type": "markdown", "body": "## 待回答问题\n\n需要继续确认：不同盈利/回撤权重下胜者是否稳定；删除6月已知下跌窗口后排名是否改变；以及最佳配置在Taker费率150%和0.05%/0.10%滑点下是否仍位于Pareto前沿。"},
    ]
    artifact = {
        "surface": "report",
        "manifest": {
            "version": 1, "surface": "report", "title": title,
            "description": "以完整180天Grid盈利和拼接最大回撤直接优化XGBoost Risk-off参数。",
            "generatedAt": pd.Timestamp.now(tz="UTC").isoformat(),
            "cards": cards, "charts": charts, "tables": tables,
            "sources": sources, "blocks": blocks,
        },
        "snapshot": {
            "version": 1, "generatedAt": pd.Timestamp.now(tz="UTC").isoformat(),
            "status": "ready", "datasets": {
                "headline": [{
                    "pnl": float(winner.oos_pnl_fdusd), "baseline_pnl": float(baseline.oos_pnl_fdusd),
                    "drawdown": float(winner.stitched_max_drawdown_pct),
                    "baseline_drawdown": float(baseline.stitched_max_drawdown_pct),
                    "portfolio_stops": int(winner.portfolio_stop_events),
                    "pair_stops": int(winner.pair_stop_events),
                }],
                "candidates": safe_records(candidates[["display_id", "search_scope", "profit_fdusd", "drawdown_pct", "trading_objective_score", "pareto_front"]]),
                "top_candidates": safe_records(top[["rank", "display_id", "search_scope", "profit_fdusd", "drawdown_pct", "trading_objective_score", "portfolio_stop_events", "pair_stop_events"]]),
                "equity": curve_rows, "drawdown": drawdown_rows,
                "weekly": safe_records(weekly[["week", "scenario", "pnl_fdusd"]]),
            },
        },
        "sources": sources,
        "package_info": {"root": "xgboost_180d_trading_objective_v6", "manifestPath": "artifact.json", "snapshotPath": "artifact.json"},
    }
    return artifact


def deliver_report(output_dir: Path, plugin_root: Path) -> Path:
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    artifact = build_report_artifact(output_dir, summary)
    artifact_path, report_path = output_dir / "artifact.json", output_dir / "report.html"
    write_json(artifact_path, artifact)
    builder = plugin_root / "skills/build-report/scripts/deliver_portable_artifact.mjs"
    subprocess.run(
        [
            "node", str(builder.resolve()), "--input", str(artifact_path.resolve()),
            "--output", str(report_path.resolve()), "--ready-timeout-ms", "15000",
            "--timeout-ms", "30000",
        ],
        cwd=output_dir, check=True,
    )
    return report_path


def build_latest_plotly(output_dir: Path, source_dir: Path, cache_dir: Path) -> Path:
    """Render the locked v6 long/short entry and exit timeline."""
    lock = json.loads((output_dir / "locked_configuration.json").read_text(encoding="utf-8"))
    best = lock["best_dual"]
    selections = pd.read_csv(source_dir / "grid_selections.csv")
    start_ts, end_ts = int(selections.test_start.min()), int(selections.test_end.max())
    state_parts, event_parts, interval_parts = [], [], []
    for strategy, prefix in (
        ("long_persistent_72h", "long_"),
        ("short_spike_1h_24h", "short_"),
    ):
        model_key = str(best[f"{prefix}model_key"])
        _, config_id, architecture = model_key.split("|")
        prediction_path, _ = prediction_paths(
            output_dir, strategy, config_id, architecture
        )
        if not prediction_path.exists():
            raise FileNotFoundError(f"Missing locked prediction cache: {prediction_path}")
        prediction = pd.read_csv(prediction_path)
        gate = row_gate(best, prefix)
        _, states, events, intervals = v5.build_continuous_gate(
            prediction, strategy, gate, start_ts, end_ts
        )
        state_parts.append(states)
        event_parts.append(events)
        interval_parts.append(intervals)
    states = pd.concat(state_parts, ignore_index=True)
    events = pd.concat(event_parts, ignore_index=True)
    intervals = pd.concat(interval_parts, ignore_index=True)
    states.to_csv(output_dir / "latest_plotly_risk_states.csv.gz", index=False, compression="gzip")
    events.to_csv(output_dir / "latest_plotly_entry_exit_events.csv", index=False)
    intervals.to_csv(output_dir / "latest_plotly_risk_intervals.csv", index=False)
    coverage = dual.june_coverage(intervals)
    coverage.to_csv(output_dir / "latest_plotly_june_coverage.csv", index=False)
    metrics = pd.read_csv(output_dir / "final_metrics.csv")
    plot = dual.build_plotly(
        cache_dir, output_dir, states, events, intervals, metrics, coverage
    )
    long_gate, short_gate = row_gate(best, "long_"), row_gate(best, "short_")
    details = (
        f"<br><b>最新锁定：</b>长期 {best['long_model_key']}，"
        f"q={long_gate.entry_quantile:g}，持续{long_gate.maximum_hours}h，"
        f"冷却{long_gate.cooldown_hours}h；短期 {best['short_model_key']}，"
        f"q={short_gate.entry_quantile:g}，连续{short_gate.entry_bars}根确认，"
        f"最长{short_gate.maximum_hours}h。"
    )
    page = plot.read_text(encoding="utf-8")
    page = page.replace(
        "<title>XGBoost双风险策略180天回测</title>",
        "<title>最新XGBoost v6进入退出时间</title>",
    ).replace(
        "<h1>XGBoost双风险策略：180天诊断回测</h1>",
        "<h1>最新XGBoost v6：BTC/ETH进入退出时间</h1>",
    ).replace(
        "该区间已被查看，全部结果均为诊断性回放。</div>",
        f"该区间已被查看，全部结果均为诊断性回放。{details}</div>",
    )
    page = page.replace("120h持续下跌", "长期持续下跌")
    plot.write_text(page, encoding="utf-8")
    return plot


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    configs = pd.DataFrame(xgb_configurations())
    configs.to_csv(args.output_dir / "xgboost_40_parameters.csv", index=False)
    panel = pd.read_csv(args.source_dir / "dual_target_feature_panel.csv.gz")
    selections = pd.read_csv(args.source_dir / "grid_selections.csv")
    prescreen_path = args.output_dir / "model_prescreen_40x2x2.csv"

    if args.stage in {"prescreen", "all"}:
        prescreen = model_prescreen(panel, selections, prescreen_path, args.resume)
    else:
        if not prescreen_path.exists():
            raise FileNotFoundError("Run --stage prescreen first")
        prescreen = pd.read_csv(prescreen_path)
    if args.stage == "prescreen":
        return 0

    if args.stage in {"search", "all"}:
        finalists = select_finalists(prescreen)
        write_json(args.output_dir / "model_finalists.json", finalists)
        predictions, audit = load_or_train_predictions(
            panel, selections, finalists, args.output_dir, args.resume
        )
        candles, quality = load_candles(args.cache_dir)
        quality.to_csv(args.output_dir / "data_quality.csv", index=False)
        baseline_result = base.replay(
            candles, selections, scenario="Mechanism 1 (BTC ROC/SQZMOM)",
            baseline_gate=base.mechanism1_gate(candles), record_details=True,
        )
        baseline_metrics = trading_row(baseline_result)
        single = search_single_channels(
            candles, selections, finalists, predictions, baseline_metrics,
            args.output_dir / "single_channel_trading_search.csv", args.resume,
        )
        joint = search_joint_channels(
            candles, selections, predictions, single, baseline_metrics,
            args.output_dir / "dual_channel_trading_search.csv", args.resume,
        )
        results, lock, metrics = finalize_results(
            candles, selections, predictions, single, joint
        )
        summary = write_final_artifacts(
            args.output_dir, results, lock, metrics, prescreen, single, joint,
            audit, args.source_dir,
        )
        print(json.dumps({
            "verdict": summary["verdict"], "best_dual": summary["best_dual"],
            "baseline": summary["baseline"],
        }, ensure_ascii=False, indent=2), flush=True)
    elif not (args.output_dir / "summary.json").exists():
        raise FileNotFoundError("Run --stage search first")

    if args.stage in {"report", "all"}:
        report = deliver_report(args.output_dir, args.plugin_root)
        print(json.dumps({"report": str(report)}, ensure_ascii=False), flush=True)
    if args.stage in {"plot", "all"}:
        plot = build_latest_plotly(args.output_dir, args.source_dir, args.cache_dir)
        print(json.dumps({"plotly": str(plot)}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

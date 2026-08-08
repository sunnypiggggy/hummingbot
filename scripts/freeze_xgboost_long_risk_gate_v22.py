#!/usr/bin/env python3
"""Freeze exact weekly walk-forward models and thresholds into the v22 bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd

import retrain_xgboost_long_risk_gate_250d_v19 as v19
import tune_xgboost_momentum_stop_v2 as tune
import xgboost_long_risk_gate_v22 as v22


DEFAULT_V21 = Path("results/backtests/xgboost_grid_long_risk_gate_v21_250d")
DEFAULT_V19 = Path("results/backtests/xgboost_grid_long_risk_gate_v19_250d")
DEFAULT_SOURCE = Path("results/backtests/eth_xgboost_long_risk_gate_v15_250d")
DEFAULT_OUTPUT = Path("results/backtests/xgboost_grid_long_risk_gate_v22_weekly_250d/shadow_package")
# XGBoost hist uses parallel floating-point reductions; refitting the same
# recipe can differ by a few 1e-8 across process schedules.  State/event
# parity is checked separately and remains exact.
RETRAIN_PROBABILITY_TOLERANCE = 1e-6
RETRAIN_THRESHOLD_TOLERANCE = 1e-12
_PANEL: pd.DataFrame | None = None
_SPECS: dict[str, dict[str, Any]] = {}
_PREDICTIONS: dict[str, pd.DataFrame] = {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v21-dir", type=Path, default=DEFAULT_V21)
    parser.add_argument("--v19-dir", type=Path, default=DEFAULT_V19)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--xgb-threads", type=int, default=2)
    return parser.parse_args()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _model_sha256(model: Any) -> str:
    return _sha256_bytes(bytes(model.get_booster().save_raw(raw_format="ubj")))


def _init(panel_path: str, selected_specs: list[dict[str, Any]], prediction_paths: Mapping[str, str], threads: int) -> None:
    global _PANEL, _SPECS, _PREDICTIONS
    _PANEL = pd.read_csv(panel_path)
    _SPECS = {str(item["pair"]): item for item in selected_specs}
    _PREDICTIONS = {pair: pd.read_csv(path) for pair, path in prediction_paths.items()}
    tune.XGB_N_JOBS = int(threads)
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = "1"


def _fit_week(job: tuple[str, dict[str, Any]]) -> tuple[str, dict[str, Any]]:
    pair, block = job
    if _PANEL is None:
        raise RuntimeError("v22 worker is not initialized")
    spec = _SPECS[pair]
    frame = v19.target_frame(_PANEL, str(spec["target"]), pair)
    model, calibration, audit = v19.fit_leakage_safe(
        frame, int(block["train_end"]), spec["config"], spec["features"],
    )
    quantile = v22.GATES[pair].entry_quantile
    calibration_threshold = float(calibration.probability.quantile(quantile))
    test = frame[frame.signal_ts.between(int(block["test_start"]), int(block["test_end"]), inclusive="left")]
    probability = model.predict_proba(test[list(spec["features"])])[:, 1]
    cached = _PREDICTIONS[pair]
    wanted = cached[cached.fold.eq(int(block["fold"]))].sort_values("signal_ts")
    test = test.sort_values("signal_ts")
    if not np.array_equal(test.signal_ts.to_numpy(np.int64), wanted.signal_ts.to_numpy(np.int64)):
        raise AssertionError(f"{pair} fold {block['fold']} cached timestamps differ")
    probability_error = float(np.max(np.abs(probability - wanted.probability.to_numpy(float))))
    threshold_col = v19.legacy.v5.quantile_column(quantile)
    cached_threshold = float(wanted[threshold_col].iloc[0])
    threshold_error = float(abs(calibration_threshold - cached_threshold))
    if (probability_error > RETRAIN_PROBABILITY_TOLERANCE
            or threshold_error > RETRAIN_THRESHOLD_TOLERANCE):
        raise AssertionError(
            f"{pair} fold {block['fold']} does not reproduce cache: p={probability_error}, q={threshold_error}"
        )
    # Histogram refits can move tied score levels by a few 1e-8.  Preserve the
    # old fold's exact >= classification with the nearest representable
    # threshold; retain the raw calibration quantile for audit.
    cached_above = wanted.probability.to_numpy(float) >= cached_threshold
    if np.array_equal(probability >= calibration_threshold, cached_above):
        execution_threshold = calibration_threshold
    elif cached_above.all():
        execution_threshold = float(np.min(probability))
    elif not cached_above.any():
        execution_threshold = float(np.nextafter(np.float32(np.max(probability)), np.float32(np.inf)))
    else:
        maximum_below = float(np.max(probability[~cached_above]))
        minimum_above = float(np.min(probability[cached_above]))
        if not maximum_below < minimum_above:
            raise AssertionError(f"{pair} fold {block['fold']} score classes cannot be separated")
        if maximum_below < calibration_threshold <= minimum_above:
            execution_threshold = calibration_threshold
        elif calibration_threshold <= maximum_below:
            execution_threshold = float(np.nextafter(np.float32(maximum_below), np.float32(np.inf)))
        else:
            execution_threshold = minimum_above
    if not np.array_equal(probability >= execution_threshold, cached_above):
        raise AssertionError(f"{pair} fold {block['fold']} threshold parity failed")
    signed_test_end = max(int(block["test_end"]), int(block["test_start"]) + 7 * v19.DAY)
    return pair, {
        "fold": int(block["fold"]), "train_cutoff": int(block["train_end"]),
        "test_start": int(block["test_start"]), "test_end": signed_test_end,
        "research_test_end": int(block["test_end"]),
        "entry_threshold": execution_threshold, "calibration_threshold": calibration_threshold,
        "entry_quantile": quantile,
        "execution_threshold_adjustment": execution_threshold - calibration_threshold,
        "best_tree_count": int(audit["best_tree_count"]),
        "last_label_ready_ts": int(audit["last_label_ready_ts"]),
        "development_last_ts": int(audit["development_last_ts"]),
        "calibration_first_ts": int(audit["calibration_first_ts"]),
        "calibration_rows": int(audit["calibration_rows"]),
        "model_sha256": _model_sha256(model), "model": model,
        "cached_probability_max_abs_error": probability_error,
        "cached_threshold_abs_error": threshold_error,
    }


def _selected_specs(args: argparse.Namespace) -> list[dict[str, Any]]:
    lock = json.loads((args.v21_dir / "locked_configuration.json").read_text(encoding="utf-8"))
    finalists = json.loads((args.v19_dir / "weekly_finalists.json").read_text(encoding="utf-8"))
    by_key = {str(item["model_key"]): item for item in finalists}
    selected = []
    for pair in v22.PAIRS:
        candidate = str(lock["candidate"][f"{pair[:3]}_candidate_id"])
        model_key = "|".join(candidate.split("|")[:4])
        if model_key not in by_key:
            raise KeyError(f"selected v21 model is absent from v19 finalists: {model_key}")
        selected.append(dict(by_key[model_key]))
    return selected


def main() -> int:
    mp.freeze_support()
    args = parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = args.output_dir / "models"; model_dir.mkdir(parents=True, exist_ok=True)
    selected = _selected_specs(args)
    predictions = {
        str(spec["pair"]): str(v19.cache_path(type("Args", (), {"output_dir": args.v19_dir})(), "weekly", spec))
        for spec in selected
    }
    selections = pd.read_csv(args.source_dir / "grid_selections.csv").sort_values("fold")
    blocks = selections[["fold", "train_end", "test_start", "test_end"]].to_dict("records")
    jobs = [(pair, block) for pair in v22.PAIRS for block in blocks]
    workers = min(max(1, int(args.workers)), len(jobs))
    initargs = (str(args.v19_dir / "feature_panel.csv.gz"), selected, predictions, int(args.xgb_threads))
    if workers == 1:
        _init(*initargs); results = [_fit_week(job) for job in jobs]
    else:
        with mp.get_context("spawn").Pool(workers, initializer=_init, initargs=initargs, maxtasksperchild=8) as pool:
            results = list(pool.imap_unordered(_fit_week, jobs))
    spec_by_pair = {str(item["pair"]): item for item in selected}
    pair_weeks = {pair: [] for pair in v22.PAIRS}
    for pair, week in results:
        pair_weeks[pair].append(week)
    bundle = {
        "schema": v22.MODEL_BUNDLE_SCHEMA, "model_version": v22.MODEL_VERSION,
        "feature_schema_sha256": v22.feature_schema_sha256(),
        "strategy_schema_sha256": v22.strategy_schema_sha256(),
        "strategy_spec": v22.strategy_spec(),
        "training_panel_sha256": v19.sha256_file(args.v19_dir / "feature_panel.csv.gz"),
        "grid_sequence_sha256": v19.sha256_file(args.source_dir / "grid_selections.csv"),
        "pairs": {},
    }
    for pair in v22.PAIRS:
        spec = spec_by_pair[pair]
        bundle["pairs"][pair] = {
            "model_key": spec["model_key"], "target": spec["target"],
            "features": list(spec["features"]), "config": dict(spec["config"]),
            "gate": asdict(v22.GATES[pair]),
            "weeks": sorted(pair_weeks[pair], key=lambda item: int(item["test_start"])),
        }
    v22.validate_weekly_bundle(bundle)
    model_path = model_dir / "xgboost_long_risk_gate_v22_weekly.joblib"
    temporary = model_path.with_name(f".{model_path.name}.{os.getpid()}.tmp")
    joblib.dump(bundle, temporary, compress=3); os.replace(temporary, model_path)
    model_hash = v19.sha256_file(model_path)
    source_lock = args.v21_dir / "locked_configuration.json"
    lock = {
        "schema": "xgboost-grid-long-risk-gate-v22-weekly-shadow-lock-v1",
        "model_version": v22.MODEL_VERSION, "shadow_mode": True,
        "deployment_allowed": False, "promotion_authorized": False,
        "historical_verdict": "NO-GO", "short_spike_enabled": False,
        "market_sell_action": False, "mechanism1_fallback_allowed": False,
        "probability_semantics": "weekly_walk_forward_model_with_fold_local_calibration_threshold",
        "model_path": model_path.as_posix(), "model_sha256": model_hash,
        "feature_schema_sha256": v22.feature_schema_sha256(),
        "strategy_schema_sha256": v22.strategy_schema_sha256(),
        "training_panel_sha256": bundle["training_panel_sha256"],
        "grid_sequence_sha256": bundle["grid_sequence_sha256"],
        "source_v21_lock_sha256": v19.sha256_file(source_lock),
        "weeks_per_pair": len(blocks),
        "effective_start": int(selections.test_start.min()),
        "effective_end": max(int(item["test_end"]) for item in bundle["pairs"][v22.PAIRS[0]]["weeks"]),
        "future_week_policy": "signed_week_required_else_fail_closed",
        "retrain_probability_tolerance": RETRAIN_PROBABILITY_TOLERANCE,
        "retrain_threshold_tolerance": RETRAIN_THRESHOLD_TOLERANCE,
    }
    v19.atomic_json(args.output_dir / "shadow_lock.json", lock)
    print(json.dumps({**lock, "model_path": str(model_path)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

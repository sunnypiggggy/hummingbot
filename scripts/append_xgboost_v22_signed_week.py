#!/usr/bin/env python3
"""Train and append one future signed week to a staged v22 bundle.

The source package is never modified.  The staged package remains shadow-only
and requires review before it can replace another v22 package.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import joblib
import pandas as pd

from compare_independent_gate_ml_stops import build_feature_panel, load_candles
import optimize_xgboost_grid_risk_gate_v7 as labels
import retrain_xgboost_long_risk_gate_250d_v19 as v19
import tune_xgboost_momentum_stop_v2 as tune
import xgboost_long_risk_gate_v22 as v22


DEFAULT_PACKAGE = Path("results/backtests/xgboost_grid_long_risk_gate_v22_weekly_250d/shadow_package")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-package", type=Path, default=DEFAULT_PACKAGE)
    parser.add_argument("--candle-dir", type=Path, required=True)
    parser.add_argument("--cutoff", required=True, help="UTC ISO timestamp or epoch seconds")
    parser.add_argument("--output-package", type=Path, required=True)
    parser.add_argument("--xgb-threads", type=int, default=2)
    return parser.parse_args()


def parse_cutoff(value: str) -> int:
    try: return int(value)
    except ValueError: return int(pd.Timestamp(value).timestamp())


def combined_candle_hash(directory: Path) -> str:
    digest = hashlib.sha256()
    for pair in v22.PAIRS: digest.update(v19.sha256_file(directory / f"binance_{pair}_5m.csv").encode())
    return digest.hexdigest()


def main() -> int:
    args = parse_args(); cutoff = parse_cutoff(args.cutoff)
    source_lock_path = args.source_package / "shadow_lock.json"
    source_lock = json.loads(source_lock_path.read_text(encoding="utf-8"))
    source_model = Path(source_lock["model_path"])
    if not source_model.exists(): source_model = args.source_package / "models" / source_model.name
    if v19.sha256_file(source_model) != source_lock["model_sha256"]: raise RuntimeError("source model hash mismatch")
    bundle = joblib.load(source_model); v22.validate_weekly_bundle(bundle)
    expected_cutoff = max(int(week["test_end"]) for week in bundle["pairs"][v22.PAIRS[0]]["weeks"])
    if cutoff != expected_cutoff:
        raise ValueError(f"new cutoff must equal signed manifest end {expected_cutoff}, got {cutoff}")
    candles, quality = load_candles(args.candle_dir)
    for row in quality.itertuples(index=False):
        if int(row.missing_5m_rows) or int(row.invalid_ohlcv_rows): raise RuntimeError(f"{row.pair} candle quality failed")
    if min(int(frame.timestamp.max()) for frame in candles.values()) < cutoff - 300:
        raise RuntimeError("candle history does not reach the requested weekly cutoff")
    panel = v19.prepare_panel(labels.relabel_panel(build_feature_panel(candles, horizon_hours=6), candles))
    tune.XGB_N_JOBS = int(args.xgb_threads)
    next_fold = max(int(week["fold"]) for week in bundle["pairs"][v22.PAIRS[0]]["weeks"]) + 1
    for pair in v22.PAIRS:
        item = bundle["pairs"][pair]
        frame = v19.target_frame(panel, str(item["target"]), pair)
        model, calibration, audit = v19.fit_leakage_safe(frame, cutoff, item["config"], item["features"])
        threshold = float(calibration.probability.quantile(v22.GATES[pair].entry_quantile))
        model_hash = hashlib.sha256(bytes(model.get_booster().save_raw(raw_format="ubj"))).hexdigest()
        item["weeks"].append({"fold": next_fold, "train_cutoff": cutoff, "test_start": cutoff,
            "test_end": cutoff + 7 * v19.DAY, "research_test_end": None,
            "entry_threshold": threshold, "calibration_threshold": threshold,
            "entry_quantile": v22.GATES[pair].entry_quantile, "execution_threshold_adjustment": 0.0,
            "best_tree_count": int(audit["best_tree_count"]),
            "last_label_ready_ts": int(audit["last_label_ready_ts"]),
            "development_last_ts": int(audit["development_last_ts"]),
            "calibration_first_ts": int(audit["calibration_first_ts"]),
            "calibration_rows": int(audit["calibration_rows"]), "model_sha256": model_hash,
            "model": model, "cached_probability_max_abs_error": None, "cached_threshold_abs_error": None})
    bundle["training_candle_sha256"] = combined_candle_hash(args.candle_dir)
    bundle["last_signed_week_cutoff"] = cutoff; v22.validate_weekly_bundle(bundle)
    model_dir = args.output_package / "models"; model_dir.mkdir(parents=True, exist_ok=True)
    target = model_dir / "xgboost_long_risk_gate_v22_weekly.joblib"
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    joblib.dump(bundle, temporary, compress=3); os.replace(temporary, target)
    staged_lock = {**source_lock, "model_path": target.as_posix(), "model_sha256": v19.sha256_file(target),
        "weeks_per_pair": next_fold, "effective_end": cutoff + 7 * v19.DAY,
        "training_candle_sha256": bundle["training_candle_sha256"],
        "source_package_lock_sha256": v19.sha256_file(source_lock_path),
        "staged_for_review": True, "deployment_allowed": False, "promotion_authorized": False}
    for key in list(staged_lock):
        if key.startswith("application_"): staged_lock.pop(key)
    v19.atomic_json(args.output_package / "shadow_lock.json", staged_lock)
    print(json.dumps({"fold": next_fold, "cutoff": cutoff, "valid_until": cutoff + 7 * v19.DAY,
        "model_sha256": staged_lock["model_sha256"], "staged_for_review": True,
        "deployment_allowed": False}, indent=2)); return 0


if __name__ == "__main__":
    raise SystemExit(main())


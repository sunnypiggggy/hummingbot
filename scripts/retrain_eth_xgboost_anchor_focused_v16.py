#!/usr/bin/env python3
"""Anchor-focused, no-lookahead ETH long XGBoost retraining.

The known February/June windows are used only for model selection. Each fixed
origin or weekly fit uses labels mature at its cutoff. The model controls only
ordinary Grid BUY through the existing v14 persistent-evidence state machine.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import multiprocessing as mp
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

import optimize_eth_xgboost_long_risk_gate_v15 as v15
import optimize_xgboost_roc_sqz_pair_risk_gate_v8 as engine
import refine_xgboost_v9_long_entry_persistence_v14 as v14
import tune_xgboost_momentum_stop_v2 as tune


MODEL_VERSION = "eth-xgboost-anchor-focused-v16"
OUTPUT_DIR = Path("results/backtests/eth_xgboost_anchor_focused_v16")
V11_DIR = Path("results/backtests/xgboost_feature_selected_pair_risk_gate_v11")
PAIR = "ETH-FDUSD"
TARGETS = ("long_72h", "long_120h")
WEIGHT_PROFILES = ("balanced", "positive_x2", "persistent_severity")

FEATURE_SETS = {
    "long_72h": {
        "base": ("adx_14", "di_spread", "atr_pct", "btc_volatility_20"),
        "structure": ("adx_14", "di_spread", "atr_pct", "btc_volatility_20",
                      "below_ema20_ratio_72h", "drawdown_from_high_168h",
                      "btc_downside_beta_72h", "expected_shortfall_72h"),
        "structure_roc_sqz": ("below_ema20_ratio_72h", "drawdown_from_high_168h",
                              "btc_downside_beta_72h", "expected_shortfall_72h",
                              "roc_48h_4h", "sqzmom_pct_4h", "sqzmom_slope_4h",
                              "adx_14", "di_spread", "atr_pct"),
    },
    "long_120h": {
        "base": ("adx_14", "di_spread", "atr_pct", "btc_volatility_20"),
        "structure": ("adx_14", "di_spread", "atr_pct", "btc_volatility_20",
                      "btc_downside_beta_72h", "historical_var_72h",
                      "drawdown_from_high_168h", "vol_of_vol_72h"),
        "structure_roc_sqz": ("btc_downside_beta_72h", "historical_var_72h",
                              "drawdown_from_high_168h", "vol_of_vol_72h",
                              "below_ema20_ratio_72h", "drawdown_duration_168h",
                              "roc_48h_4h", "sqzmom_pct_4h", "sqzmom_slope_4h",
                              "di_spread"),
    },
}

OUTCOME_COLUMNS = {
    "long_72h": ("future_below_fraction_72h_v7", "future_close_return_72h_v7", "long_threshold_72h_v7"),
    "long_120h": ("future_below_fraction_120h_v7", "future_close_return_120h_v7", "long_threshold_120h_v7"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("screen", "weekly", "search", "extended", "finalize", "plot", "all"), default="all")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--xgb-threads", type=int, default=2)
    parser.add_argument("--finalists-per-target", type=int, default=12)
    parser.add_argument("--grid-top", type=int, default=160)
    parser.add_argument("--v9-dir", type=Path, default=v15.V9_DIR)
    parser.add_argument("--v14-dir", type=Path, default=v15.V14_DIR)
    parser.add_argument("--v11-dir", type=Path, default=V11_DIR)
    parser.add_argument("--cache-dir", type=Path, default=Path("data/backtesting_candles"))
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def model_specs() -> list[dict[str, Any]]:
    output = []
    for target in TARGETS:
        for feature_id, features in FEATURE_SETS[target].items():
            for profile in WEIGHT_PROFILES:
                for config in tune.xgb_configurations():
                    output.append({
                        "model_key": f"{target}|{feature_id}|{profile}|{config['config_id']}",
                        "target": target, "feature_id": feature_id, "features": list(features),
                        "weight_profile": profile, "config": config,
                    })
    return output


def weighted_samples(frame: pd.DataFrame, target: str, profile: str) -> np.ndarray:
    weights = np.asarray(compute_sample_weight("balanced", frame.target.astype(int)), dtype=float)
    positive = frame.target.astype(int).to_numpy() == 1
    if profile == "positive_x2":
        weights[positive] *= 2.0
    elif profile == "persistent_severity":
        fraction_col, return_col, threshold_col = OUTCOME_COLUMNS[target]
        fraction = frame[fraction_col].fillna(0).clip(0, 1).to_numpy(float)
        threshold = frame[threshold_col].abs().replace(0, np.nan).to_numpy(float)
        severity = np.nan_to_num((-frame[return_col].to_numpy(float)) / threshold, nan=0.0, posinf=2.0)
        weights[positive] *= (1.0 + 2.0 * fraction[positive] + np.clip(severity[positive], 0, 2))
    return weights


def fit_model(spec: Mapping[str, Any], mature: pd.DataFrame, core: pd.DataFrame,
              validation: pd.DataFrame) -> tuple[XGBClassifier, dict[str, Any]]:
    config, features = spec["config"], list(spec["features"])
    profile, target = str(spec["weight_profile"]), str(spec["target"])
    cap, best_trees, best_score = int(config["n_estimators"]), int(config["n_estimators"]), np.nan
    if bool(config["uses_early_stopping"]):
        early = XGBClassifier(**tune.config_params(config), early_stopping_rounds=50)
        early.fit(core[features], core.target.astype(int),
                  sample_weight=weighted_samples(core, target, profile),
                  eval_set=[(validation[features], validation.target.astype(int))],
                  sample_weight_eval_set=[weighted_samples(validation, target, profile)], verbose=False)
        if getattr(early, "best_iteration", None) is not None:
            best_trees = max(1, min(cap, int(early.best_iteration) + 1))
        if getattr(early, "best_score", None) is not None:
            best_score = float(early.best_score)
    final = XGBClassifier(**tune.config_params(config, n_estimators=best_trees))
    final.fit(mature[features], mature.target.astype(int),
              sample_weight=weighted_samples(mature, target, profile), verbose=False)
    return final, {"best_tree_count": best_trees, "best_validation_logloss": best_score}


def predict_block(panel: pd.DataFrame, spec: Mapping[str, Any], block: Any) -> tuple[pd.DataFrame, dict[str, Any]]:
    working = engine.v7.working_target(panel, str(spec["target"]))
    working = working[working.pair.eq(PAIR)].copy()
    mature, core, validation = tune.split_mature_training(working, int(block.train_end))
    test = working[(working.signal_ts >= int(block.test_start)) & (working.signal_ts < int(block.test_end))].copy()
    model, audit = fit_model(spec, mature, core, validation)
    features = list(spec["features"])
    predicted = test[["pair", "signal_ts", "target"]].copy()
    predicted["probability"] = model.predict_proba(test[features])[:, 1]
    calibrated = validation[["pair", "signal_ts", "target"]].copy()
    calibrated["probability"] = model.predict_proba(validation[features])[:, 1]
    predicted = engine.attach_thresholds(predicted, calibrated)
    predicted["strategy"] = str(spec["target"])
    return predicted, {
        "model_key": spec["model_key"], "train_cutoff_ts": int(block.train_end),
        "last_mature_label_ready_ts": int(mature.label_ready_ts.max()),
        "last_calibration_signal_ts": int(validation.signal_ts.max()),
        "first_test_signal_ts": int(test.signal_ts.min()), **audit,
    }


def cache_path(args: argparse.Namespace, stage: str, model_key: str) -> Path:
    return args.output_dir / "prediction_cache" / stage / f"{model_key.replace('|', '__')}.csv.gz"


def cache_meta(args: argparse.Namespace, stage: str, spec: Mapping[str, Any]) -> dict[str, Any]:
    payload = {
        "model_version": MODEL_VERSION, "stage": stage, "spec": spec,
        "feature_panel_sha256": v15.sha256_file(args.v11_dir / "feature_panel.csv.gz"),
        "grid_sha256": v15.sha256_file(args.v11_dir / "grid_selections.csv"),
    }
    return {"payload_sha256": hashlib.sha256(v15.canonical(payload)).hexdigest(), **payload}


_PANEL: pd.DataFrame | None = None
_SELECTIONS: pd.DataFrame | None = None
_ARGS: argparse.Namespace | None = None


def init_worker(panel: pd.DataFrame, selections: pd.DataFrame, args: argparse.Namespace) -> None:
    global _PANEL, _SELECTIONS, _ARGS
    _PANEL, _SELECTIONS, _ARGS = panel, selections, args
    tune.XGB_N_JOBS = int(args.xgb_threads)
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = "1"


def save_cache(path: Path, prediction: pd.DataFrame, audit: pd.DataFrame, meta: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    prediction.to_csv(temporary, index=False, compression="gzip"); os.replace(temporary, path)
    audit.to_csv(path.with_suffix(".audit.csv"), index=False)
    meta = {**meta, "prediction_sha256": v15.sha256_file(path), "rows": len(prediction)}
    v15.atomic_json(path.with_name(path.name + ".metadata.json"), meta)


def load_cache(args: argparse.Namespace, stage: str, spec: Mapping[str, Any]) -> pd.DataFrame | None:
    path = cache_path(args, stage, str(spec["model_key"])); meta_path = path.with_name(path.name + ".metadata.json")
    if not (args.resume and path.exists() and meta_path.exists()):
        return None
    observed = json.loads(meta_path.read_text(encoding="utf-8"))
    expected = cache_meta(args, stage, spec)
    if any(observed.get(key) != value for key, value in expected.items()):
        return None
    if observed.get("prediction_sha256") != v15.sha256_file(path):
        return None
    return pd.read_csv(path)


def training_worker(job: tuple[str, dict[str, Any]]) -> tuple[str, str]:
    if _PANEL is None or _SELECTIONS is None or _ARGS is None:
        raise RuntimeError("worker not initialized")
    stage, spec = job
    cached = load_cache(_ARGS, stage, spec)
    if cached is not None:
        return str(spec["model_key"]), "reused"
    predictions, audits = [], []
    blocks = ([SimpleNamespace(train_end=engine.START_TS, test_start=engine.START_TS, test_end=engine.END_TS)]
              if stage == "screen" else list(_SELECTIONS.itertuples(index=False)))
    for block in blocks:
        prediction, audit = predict_block(_PANEL, spec, block)
        predictions.append(prediction); audits.append(audit)
    save_cache(cache_path(_ARGS, stage, str(spec["model_key"])), pd.concat(predictions, ignore_index=True),
               pd.DataFrame(audits), cache_meta(_ARGS, stage, spec))
    return str(spec["model_key"]), "trained"


def run_jobs(args: argparse.Namespace, panel: pd.DataFrame, selections: pd.DataFrame,
             stage: str, specs: list[dict[str, Any]]) -> None:
    jobs = [(stage, spec) for spec in specs]
    if args.workers == 1:
        init_worker(panel, selections, args); iterator = map(training_worker, jobs); pool = None
    else:
        pool = mp.get_context("spawn").Pool(args.workers, initializer=init_worker,
                                             initargs=(panel, selections, args), maxtasksperchild=4)
        iterator = pool.imap_unordered(training_worker, jobs, chunksize=1)
    try:
        for index, (key, status) in enumerate(iterator, 1):
            print(f"{stage.upper()} {index}/{len(jobs)} {key} [{status}]", flush=True)
    finally:
        if pool is not None: pool.close(); pool.join()


def anchor_probability_screen(args: argparse.Namespace, specs: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for spec in specs:
        prediction = pd.read_csv(cache_path(args, "screen", str(spec["model_key"])))
        for quantile in engine.ENTRY_QUANTILES:
            threshold = prediction[engine.v5.quantile_column(quantile)].to_numpy(float)
            active = prediction.probability.to_numpy(float) >= threshold
            ts = prediction.signal_ts.to_numpy(np.int64)
            values = {"model_key": spec["model_key"], "target": spec["target"],
                      "feature_id": spec["feature_id"], "weight_profile": spec["weight_profile"],
                      "config_id": spec["config"]["config_id"], "entry_quantile": quantile}
            coverages, timely = [], []
            anchor_mask = np.zeros(len(ts), dtype=bool)
            for name, start, end in engine.ANCHOR_WINDOWS:
                mask = (ts >= start) & (ts < end); anchor_mask |= mask
                coverage = float(active[mask].mean()) if mask.any() else 0.0
                is_timely = bool(active[(ts >= start) & (ts <= start + 12 * engine.HOUR)].any())
                values[f"{name}_coverage"] = coverage; values[f"{name}_timely"] = is_timely
                coverages.append(coverage); timely.append(is_timely)
            outside = float(active[~anchor_mask].mean()) if (~anchor_mask).any() else 0.0
            values.update({"minimum_anchor_coverage": min(coverages), "timely_anchor_count": sum(timely),
                           "outside_active_share": outside, "screen_frequency_pass": outside <= .20,
                           # Penalize total outside activity, not merely the
                           # excess over 20%; otherwise an always-on model can
                           # outrank a selective precursor model.
                           "screen_score": min(coverages) + 0.5 * sum(timely) - 2.0 * outside})
            rows.append(values)
    ranked = pd.DataFrame(rows).sort_values(
        ["screen_frequency_pass", "screen_score", "minimum_anchor_coverage",
         "timely_anchor_count", "outside_active_share"],
        ascending=[False, False, False, False, True],
    )
    ranked.to_csv(args.output_dir / "fixed_origin_anchor_screen.csv", index=False)
    return ranked


def select_finalists(args: argparse.Namespace, ranked: pd.DataFrame, specs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = []
    for target in TARGETS:
        best = ranked[ranked.target.eq(target)].drop_duplicates("model_key").head(args.finalists_per_target)
        keys.extend(best.model_key.astype(str))
    # Always retain the exact v15 incumbent and its 120h legacy counterpart.
    # A fixed-origin prescreen can otherwise discard a model whose strength
    # comes from weekly refitting, making the comparison unfair.
    keys.extend(("long_72h|base|balanced|xgb_35", "long_120h|base|balanced|xgb_03"))
    selected = [spec for spec in specs if spec["model_key"] in set(keys)]
    Path(args.output_dir / "weekly_finalists.json").write_text(
        json.dumps(selected, indent=2, default=str), encoding="utf-8"
    )
    return selected


def ensure_incumbents(selected: list[dict[str, Any]], specs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    required = {"long_72h|base|balanced|xgb_35", "long_120h|base|balanced|xgb_03"}
    by_key = {str(item["model_key"]): item for item in [*selected, *specs]}
    keys = {str(item["model_key"]) for item in selected} | required
    return [by_key[key] for key in sorted(keys)]


def run_state_grid_search(args: argparse.Namespace, panel: pd.DataFrame,
                          selections: pd.DataFrame, candles: Mapping[str, pd.DataFrame],
                          specs: list[dict[str, Any]]) -> pd.DataFrame:
    context = panel[["pair", "signal_ts", *v14.CONTEXT_FEATURES]]
    structural, predictions = [], {}
    for spec in specs:
        path = cache_path(args, "weekly", str(spec["model_key"])); pred = pd.read_csv(path)
        sha = v15.sha256_file(path); predictions[str(spec["model_key"])] = (pred, sha)
        enriched = v14.attach_entry_evidence(pred, context, PAIR)
        for gate in v15.gate_candidates():
            row = v15.structural_row(enriched, context, str(spec["target"]), str(spec["model_key"]), sha, gate)
            row.update({"model_key": spec["model_key"], "feature_id": spec["feature_id"],
                        "weight_profile": spec["weight_profile"], "xgb_config_id": spec["config"]["config_id"]})
            structural.append(row)
    structure = pd.DataFrame(structural)
    structure["frequency_constraints_pass"] = (structure.interval_count <= 8) & (structure.outside_anchor_share <= .2)
    structure["minimum_anchor_coverage"] = structure[[f"{n}_coverage" for n, _, _ in engine.ANCHOR_WINDOWS]].min(axis=1)
    structure["timely_anchor_count"] = sum(structure[f"{n}_timely"].astype(int) for n, _, _ in engine.ANCHOR_WINDOWS)
    structure = structure.sort_values(
        ["anchor_pass", "frequency_constraints_pass", "minimum_anchor_coverage", "timely_anchor_count", "outside_anchor_share"],
        ascending=[False, False, False, False, True],
    )
    structure.to_csv(args.output_dir / "weekly_structural_search.csv", index=False)
    evaluated = structure.head(args.grid_top)
    _, _, _, fixed = v15.fixed_specifications(args)
    original, grid_rows = engine.combine_pair_gates, []
    try:
        engine.combine_pair_gates = v14.filtered_combiner(context)
        for index, row in enumerate(evaluated.itertuples(index=False), 1):
            pred, sha = predictions[str(row.model_key)]
            if sha != row.prediction_sha256: raise RuntimeError("prediction hash changed")
            metrics = engine.replay_metrics(candles, selections,
                [*fixed, (pred, PAIR, "long", str(row.target), engine.gate_from_row(row._asdict()))])
            grid_rows.append({**row._asdict(), **metrics})
            if index % 10 == 0: print(f"GRID {index}/{len(evaluated)}", flush=True)
    finally:
        engine.combine_pair_gates = original
    ranked = pd.DataFrame(grid_rows)
    ranked["profit_percentile"] = ranked.oos_pnl_fdusd.rank(pct=True)
    ranked["drawdown_percentile"] = ranked.stitched_max_drawdown_pct.rank(pct=True)
    ranked["objective_score"] = .5 * ranked.profit_percentile + .5 * ranked.drawdown_percentile
    ranked["eligible"] = (ranked.anchor_pass & (ranked.oos_pnl_fdusd > 4.08906229455954)
                           & (ranked.stitched_max_drawdown_pct >= -9.263364315297606)
                           & (ranked.portfolio_stop_events == 0) & (ranked.pair_stop_events < 7)
                           & (ranked.btc_pnl_fdusd >= 0) & (ranked.eth_pnl_fdusd >= 0))
    ranked = ranked.sort_values(
        ["eligible", "objective_score", "anchor_pass", "frequency_constraints_pass",
         "minimum_anchor_coverage", "timely_anchor_count", "active_hours"],
        ascending=[False, False, False, False, False, False, True],
    ).reset_index(drop=True); ranked["rank"] = np.arange(1, len(ranked) + 1)
    ranked.to_csv(args.output_dir / "grid_search.csv", index=False)
    winner = ranked.iloc[0].to_dict()
    lock = {"model_version": MODEL_VERSION, "deployment_allowed": False,
            "evidence_status": "known_anchor_in_sample_targeted_optimization",
            "candidate": winner, "prediction_file": cache_path(args, "weekly", str(winner["model_key"])).as_posix(),
            "prediction_sha256": winner["prediction_sha256"],
            "feature_panel_sha256": v15.sha256_file(args.v11_dir / "feature_panel.csv.gz")}
    v15.atomic_json(args.output_dir / "locked_configuration.json", lock)
    return ranked


def extended_cooldown_search(args: argparse.Namespace, panel: pd.DataFrame,
                             selections: pd.DataFrame, candles: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Resolve the coverage/frequency tradeoff for the strongest 120h model."""
    model_key = "long_120h|base|balanced|xgb_03"
    path = cache_path(args, "weekly", model_key); prediction = pd.read_csv(path)
    prediction_sha = v15.sha256_file(path)
    context = panel[["pair", "signal_ts", *v14.CONTEXT_FEATURES]]
    enriched = v14.attach_entry_evidence(prediction, context, PAIR)
    rows = []
    for entry, entry_bars, recovery_bars, minimum, maximum, cooldown in itertools.product(
        engine.ENTRY_QUANTILES, (1, 2), (4, 8), (12, 24), (120, 168), (72, 120, 168, 336),
    ):
        gate = engine.v5.GateParameters(
            float(entry), max(0.50, float(entry) - 0.10), entry_bars,
            recovery_bars, minimum, maximum, cooldown,
        )
        intervals = v15.fast_long_intervals(enriched, gate)
        metrics = engine.pair_anchor_metrics(intervals, PAIR)
        rows.append({
            "candidate_id": f"{model_key}|{engine.gate_id('long', gate)}",
            "model_key": model_key, "target": "long_120h", "feature_id": "base",
            "weight_profile": "balanced", "xgb_config_id": "xgb_03",
            "prediction_sha256": prediction_sha, **gate.__dict__, **metrics,
            "active_hours": float(intervals.duration_hours.sum()) if not intervals.empty else 0.0,
        })
    structural = pd.DataFrame(rows)
    structural["frequency_constraints_pass"] = (
        (structural.interval_count <= 8) & (structural.outside_anchor_share <= .20)
    )
    structural["minimum_anchor_coverage"] = structural[
        [f"{name}_coverage" for name, _, _ in engine.ANCHOR_WINDOWS]
    ].min(axis=1)
    structural["timely_anchor_count"] = sum(
        structural[f"{name}_timely"].astype(int) for name, _, _ in engine.ANCHOR_WINDOWS
    )
    structural = structural.sort_values(
        ["anchor_pass", "frequency_constraints_pass", "minimum_anchor_coverage",
         "timely_anchor_count", "interval_count", "outside_anchor_share"],
        ascending=[False, False, False, False, True, True],
    ).reset_index(drop=True)
    structural.to_csv(args.output_dir / "extended_cooldown_structural_search.csv", index=False)
    _, _, _, fixed = v15.fixed_specifications(args)
    original, results = engine.combine_pair_gates, []
    try:
        engine.combine_pair_gates = v14.filtered_combiner(context)
        for row in structural.head(80).itertuples(index=False):
            metrics = engine.replay_metrics(
                candles, selections,
                [*fixed, (prediction, PAIR, "long", "long_120h", engine.gate_from_row(row._asdict()))],
            )
            results.append({**row._asdict(), **metrics})
    finally:
        engine.combine_pair_gates = original
    ranked = pd.DataFrame(results)
    ranked["profit_percentile"] = ranked.oos_pnl_fdusd.rank(pct=True)
    ranked["drawdown_percentile"] = ranked.stitched_max_drawdown_pct.rank(pct=True)
    ranked["objective_score"] = .5 * ranked.profit_percentile + .5 * ranked.drawdown_percentile
    ranked["eligible"] = (
        ranked.anchor_pass & (ranked.oos_pnl_fdusd > 4.08906229455954)
        & (ranked.stitched_max_drawdown_pct >= -9.263364315297606)
        & (ranked.portfolio_stop_events == 0) & (ranked.pair_stop_events < 7)
        & (ranked.btc_pnl_fdusd >= 0) & (ranked.eth_pnl_fdusd >= 0)
    )
    ranked = ranked.sort_values(
        ["eligible", "objective_score", "anchor_pass", "frequency_constraints_pass",
         "minimum_anchor_coverage", "timely_anchor_count", "interval_count", "outside_anchor_share"],
        ascending=[False, False, False, False, False, False, True, True],
    ).reset_index(drop=True); ranked["rank"] = np.arange(1, len(ranked) + 1)
    ranked.to_csv(args.output_dir / "extended_cooldown_grid_search.csv", index=False)
    winner = ranked.iloc[0].to_dict()
    lock = {"model_version": MODEL_VERSION, "deployment_allowed": False,
            "evidence_status": "known_anchor_in_sample_targeted_optimization",
            "candidate": winner, "prediction_file": path.as_posix(),
            "prediction_sha256": prediction_sha,
            "feature_panel_sha256": v15.sha256_file(args.v11_dir / "feature_panel.csv.gz")}
    v15.atomic_json(args.output_dir / "locked_configuration.json", lock)
    return ranked


def finalize(args: argparse.Namespace, panel: pd.DataFrame, selections: pd.DataFrame,
             candles: Mapping[str, pd.DataFrame]) -> dict[str, Any]:
    lock = json.loads((args.output_dir / "locked_configuration.json").read_text(encoding="utf-8")); row = lock["candidate"]
    path = Path(lock["prediction_file"])
    if v15.sha256_file(path) != lock["prediction_sha256"]: raise RuntimeError("locked prediction mismatch")
    pred = pd.read_csv(path); _, context, _, fixed = v15.fixed_specifications(args)
    specs = [*fixed, (pred, PAIR, "long", str(row["target"]), engine.gate_from_row(row))]
    original = engine.combine_pair_gates
    try:
        engine.combine_pair_gates = v14.filtered_combiner(context)
        detail = engine.detailed_replay(candles, selections, specs, MODEL_VERSION)
    finally: engine.combine_pair_gates = original
    if abs(detail["summary"]["oos_pnl_fdusd"] - row["oos_pnl_fdusd"]) > 1e-9: raise RuntimeError("search/final mismatch")
    outputs = {"final_risk_states.csv.gz": detail["states"], "final_risk_events.csv": detail["events"],
               "final_risk_intervals.csv": detail["intervals"], "final_equity_curve.csv.gz": detail["equity"],
               "final_trades.csv.gz": detail["trades"], "final_stop_events.csv": detail["stops"]}
    for name, frame in outputs.items(): frame.to_csv(args.output_dir / name, index=False, compression="gzip" if name.endswith(".gz") else None)
    comparison = pd.DataFrame([
        {"scenario": "XGBoost v14", **json.loads((args.v14_dir / "summary.json").read_text(encoding="utf-8"))["refined_metrics"]},
        {"scenario": "ETH XGBoost v15", **json.loads((v15.OUTPUT_DIR / "summary.json").read_text(encoding="utf-8"))["final_metrics"]},
        {"scenario": "ETH XGBoost v16 anchor-focused", **detail["summary"]},
    ]); comparison.to_csv(args.output_dir / "comparison.csv", index=False)
    result = {"model_version": MODEL_VERSION, "deployment_allowed": False,
              "evidence_status": "known_anchor_in_sample_targeted_optimization",
              "locked_candidate": row, "final_metrics": detail["summary"],
              "verdict": "NEXT_STAGE_JOINT_VALIDATION" if row.get("eligible") else "NO-GO"}
    v15.atomic_json(args.output_dir / "summary.json", result); return result


def build_plot(args: argparse.Namespace) -> Path:
    source = v14.build_plot(args, pd.read_csv(args.output_dir / "comparison.csv"))
    target = args.output_dir / "eth_xgboost_v16_anchor_focused_plotly.html"
    page = source.read_text(encoding="utf-8").replace("XGBoost v14", "ETH XGBoost v16 anchor-focused")
    target.write_text(page, encoding="utf-8"); return target


def main() -> int:
    args = parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)
    panel = pd.read_csv(args.v11_dir / "feature_panel.csv.gz")
    selections = pd.read_csv(args.v11_dir / "grid_selections.csv")
    candles, _ = engine.load_candles(args.cache_dir); specs = model_specs()
    if args.stage in {"screen", "all"}:
        run_jobs(args, panel, selections, "screen", specs)
        selected = select_finalists(args, anchor_probability_screen(args, specs), specs)
    else:
        selected = json.loads((args.output_dir / "weekly_finalists.json").read_text(encoding="utf-8"))
    selected = ensure_incumbents(selected, specs)
    if args.stage == "screen": return 0
    if args.stage in {"weekly", "all"}: run_jobs(args, panel, selections, "weekly", selected)
    if args.stage == "weekly": return 0
    if args.stage in {"search", "all"}: run_state_grid_search(args, panel, selections, candles, selected)
    if args.stage == "search": return 0
    if args.stage in {"extended", "all"}: extended_cooldown_search(args, panel, selections, candles)
    if args.stage == "extended": return 0
    result = finalize(args, panel, selections, candles) if args.stage in {"finalize", "all"} else json.loads(
        (args.output_dir / "summary.json").read_text(encoding="utf-8"))
    if args.stage in {"plot", "all"}:
        result["plotly"] = build_plot(args).as_posix(); v15.atomic_json(args.output_dir / "summary.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str)); return 0


if __name__ == "__main__":
    mp.freeze_support(); raise SystemExit(main())

#!/usr/bin/env python3
"""Build the non-deploying BTC/ETH XGBoost v16 long-only risk-gate package."""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
from plotly import graph_objects as go
from plotly.subplots import make_subplots

import optimize_xgboost_roc_sqz_pair_risk_gate_v8 as engine
import refine_xgboost_v9_long_entry_persistence_v14 as persistence
import tune_xgboost_momentum_stop_v2 as tune
from validate_grid_live import crash_candles


MODEL_VERSION = "xgboost-grid-long-risk-gate-v16"
LOCK_SCHEMA = "xgboost-grid-long-risk-gate-v16-lock-v1"
PREDICTION_SCHEMA = "xgboost-grid-long-risk-gate-v16-prediction-v1"
OUTPUT_DIR = Path("results/backtests/xgboost_grid_long_risk_gate_v16")
SOURCE_DIR = Path("results/backtests/xgboost_regime_spike_pair_risk_gate_v9")
V15_DIR = Path("results/backtests/eth_xgboost_long_risk_gate_v15")
FEATURES = ("adx_14", "di_spread", "atr_pct", "btc_volatility_20")
TARGETS = ("long_72h", "long_120h")
PAIRS = tuple(engine.PAIRS)
HOUR = engine.HOUR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("predict", "search", "finalize", "package", "plot", "all"), default="all")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--xgb-threads", type=int, default=2)
    parser.add_argument("--grid-top", type=int, default=80)
    parser.add_argument("--combo-top", type=int, default=8)
    parser.add_argument("--cache-dir", type=Path, default=Path("data/backtesting_candles"))
    parser.add_argument("--source-dir", type=Path, default=SOURCE_DIR)
    parser.add_argument("--v15-dir", type=Path, default=V15_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def reference_metrics(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return Mechanism 1 and the invalid legacy score used only as a hurdle."""
    summary_path = args.source_dir / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if isinstance(summary.get("baseline"), Mapping) and isinstance(summary.get("metrics"), Mapping):
            return dict(summary["baseline"]), dict(summary["metrics"])
    metrics = pd.read_csv(args.source_dir / "final_metrics.csv")
    return metrics.iloc[0].to_dict(), {
        "oos_pnl_fdusd": 4.08906229455954,
        "stitched_max_drawdown_pct": -9.263364315297606,
    }


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    os.replace(temporary, path)


def prediction_path(output_dir: Path, pair: str, target: str, config_id: str) -> Path:
    return output_dir / "prediction_cache" / f"{pair}__{target}__{config_id}.csv.gz"


def apply_120h_purge(panel: pd.DataFrame) -> pd.DataFrame:
    """Make every long target mature no earlier than 120 hours after signal."""
    output = panel.copy()
    floor = output.signal_ts.astype("int64") + 120 * HOUR
    for target in TARGETS:
        column = f"label_ready_ts_{target}"
        output[column] = np.maximum(output[column].astype("int64"), floor)
    return output


def prediction_metadata(args: argparse.Namespace, pair: str, target: str, config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema": PREDICTION_SCHEMA,
        "model_version": MODEL_VERSION,
        "pair": pair,
        "target": target,
        "configuration_sha256": sha256_json(dict(config)),
        "features": list(FEATURES),
        "feature_panel_sha256": sha256_file(args.source_dir / "feature_panel.csv.gz"),
        "grid_sequence_sha256": sha256_file(args.source_dir / "grid_selections.csv"),
        "target_definition_version": engine.TARGETS[target]["definition_version"],
    }


_PANEL: pd.DataFrame | None = None
_SELECTIONS: pd.DataFrame | None = None
_ARGS: argparse.Namespace | None = None


def init_worker(panel: pd.DataFrame, selections: pd.DataFrame, args: argparse.Namespace) -> None:
    global _PANEL, _SELECTIONS, _ARGS
    _PANEL, _SELECTIONS, _ARGS = panel, selections, args
    tune.XGB_N_JOBS = int(args.xgb_threads)
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = "1"


def prediction_worker(job: tuple[str, str, dict[str, Any]]) -> tuple[str, str, str, str]:
    if _PANEL is None or _SELECTIONS is None or _ARGS is None:
        raise RuntimeError("prediction worker is not initialized")
    pair, target, base_config = job
    config = {**base_config, "features": list(FEATURES)}
    config_id = str(config["config_id"])
    path = prediction_path(_ARGS.output_dir, pair, target, config_id)
    meta_path = path.with_name(path.name + ".metadata.json")
    expected = prediction_metadata(_ARGS, pair, target, config)
    if _ARGS.resume and path.exists() and meta_path.exists():
        observed = json.loads(meta_path.read_text(encoding="utf-8"))
        if observed.get("input") == expected and observed.get("prediction_sha256") == sha256_file(path):
            return pair, target, config_id, "reused"
    prediction, audit = engine.weekly_prediction(_PANEL, _SELECTIONS, target, pair, config)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    prediction.to_csv(temporary, index=False, compression="gzip")
    os.replace(temporary, path)
    audit.to_csv(path.with_suffix(".audit.csv"), index=False)
    meta = {
        "input": expected,
        "prediction_sha256": sha256_file(path),
        "rows": len(prediction),
        "maturity_pass": bool((audit.last_mature_label_ready_ts <= audit.train_cutoff_ts).all()),
        "purge_hours": 120,
    }
    atomic_json(meta_path, meta)
    return pair, target, config_id, "trained"


def generate_predictions(args: argparse.Namespace, panel: pd.DataFrame, selections: pd.DataFrame) -> None:
    jobs = [(pair, target, config) for pair in PAIRS for target in TARGETS for config in tune.xgb_configurations()]
    workers = max(1, int(args.workers))
    if workers == 1:
        init_worker(panel, selections, args)
        iterator = map(prediction_worker, jobs)
        pool = None
    else:
        pool = mp.get_context("spawn").Pool(
            workers, initializer=init_worker, initargs=(panel, selections, args), maxtasksperchild=2,
        )
        iterator = pool.imap_unordered(prediction_worker, jobs, chunksize=1)
    try:
        for index, (pair, target, config_id, status) in enumerate(iterator, 1):
            print(f"PREDICT {index:03d}/{len(jobs)} {pair} {target} {config_id} [{status}]", flush=True)
    finally:
        if pool is not None:
            pool.close(); pool.join()


def load_prediction(args: argparse.Namespace, pair: str, target: str, config: Mapping[str, Any]) -> tuple[pd.DataFrame, str]:
    configured = {**config, "features": list(FEATURES)}
    path = prediction_path(args.output_dir, pair, target, str(config["config_id"]))
    meta = json.loads(path.with_name(path.name + ".metadata.json").read_text(encoding="utf-8"))
    actual = sha256_file(path)
    if meta.get("input") != prediction_metadata(args, pair, target, configured):
        raise RuntimeError(f"prediction input hash mismatch: {path}")
    if meta.get("prediction_sha256") != actual or not meta.get("maturity_pass") or meta.get("purge_hours") != 120:
        raise RuntimeError(f"prediction audit mismatch: {path}")
    return pd.read_csv(path), actual


def long_intervals(enriched: pd.DataFrame, pair: str, gate: engine.v5.GateParameters) -> pd.DataFrame:
    entry_col = engine.v5.quantile_column(gate.entry_quantile)
    recovery_col = engine.v5.quantile_column(gate.recovery_quantile)
    state = engine.v5.GateState()
    start: int | None = None
    output: list[dict[str, Any]] = []
    columns = ["signal_ts", "probability", entry_col, recovery_col,
               "probability_lag_2h", "probability_rising_3h", "roc_sqz_worsening_8h"]
    for timestamp, probability, entry, recovery, lag2, rising, worsening in enriched[columns].itertuples(index=False, name=None):
        minimum_rise = max(1e-4, 0.25 * max(float(entry) - float(recovery), 0.0))
        evidence = bool((bool(rising) and np.isfinite(lag2) and probability - lag2 >= minimum_rise) or worsening)
        effective = float(probability) if state.active or evidence else min(float(probability), float(np.nextafter(entry, -np.inf)))
        state, transition, reason = engine.v5.step_gate(effective, float(entry), float(recovery), int(timestamp), state, gate)
        if transition == "enter":
            start = int(timestamp)
        elif transition == "recover" and start is not None:
            output.append({"pair": pair, "start_ts": start, "end_ts": int(timestamp),
                           "duration_hours": (int(timestamp) - start) / HOUR, "end_reason": reason})
            start = None
    if start is not None:
        output.append({"pair": pair, "start_ts": start, "end_ts": engine.END_TS,
                       "duration_hours": (engine.END_TS - start) / HOUR, "end_reason": "research_period_end"})
    return pd.DataFrame(output, columns=["pair", "start_ts", "end_ts", "duration_hours", "end_reason"])


_STRUCTURE_PANEL: pd.DataFrame | None = None
_STRUCTURE_ARGS: argparse.Namespace | None = None


def init_structure_worker(panel: pd.DataFrame, args: argparse.Namespace) -> None:
    global _STRUCTURE_PANEL, _STRUCTURE_ARGS
    _STRUCTURE_PANEL, _STRUCTURE_ARGS = panel, args
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = "1"


def structure_worker(job: tuple[str, str, dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    if _STRUCTURE_PANEL is None or _STRUCTURE_ARGS is None:
        raise RuntimeError("structure worker is not initialized")
    pair, target, config = job
    config_id = str(config["config_id"])
    prediction, digest = load_prediction(_STRUCTURE_ARGS, pair, target, config)
    key = f"{pair}|{target}|{config_id}"
    enriched = persistence.attach_entry_evidence(prediction, _STRUCTURE_PANEL, pair)
    rows = []
    for gate in engine.refinement_gates("long"):
        intervals = long_intervals(enriched, pair, gate)
        metrics = engine.pair_anchor_metrics(intervals, pair)
        coverage = sum(float(metrics[f"{name}_coverage"]) for name, _, _ in engine.ANCHOR_WINDOWS)
        timely = sum(int(bool(metrics[f"{name}_timely"])) for name, _, _ in engine.ANCHOR_WINDOWS)
        score = 2.0 * timely + coverage - 3.0 * max(float(metrics["outside_anchor_share"]) - 0.20, 0.0)
        rows.append({
            "candidate_id": f"{key}|{engine.gate_id('long', gate)}",
            "model_key": key, "pair": pair, "target": target, "config_id": config_id,
            "prediction_sha256": digest, **asdict(gate), **metrics,
            "active_hours": float(intervals.duration_hours.sum()) if not intervals.empty else 0.0,
            "structure_score": score, "structure_eligible": bool(metrics["anchor_pass"]),
        })
    return key, rows


def structural_candidates(args: argparse.Namespace, panel: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    configs = tune.xgb_configurations()
    jobs = [(pair, target, config) for pair in PAIRS for target in TARGETS for config in configs]
    rows: list[dict[str, Any]] = []
    workers = max(1, int(args.workers))
    if workers == 1:
        init_structure_worker(panel, args)
        iterator = map(structure_worker, jobs)
        pool = None
    else:
        pool = mp.get_context("spawn").Pool(
            workers, initializer=init_structure_worker, initargs=(panel, args), maxtasksperchild=4,
        )
        iterator = pool.imap_unordered(structure_worker, jobs, chunksize=1)
    try:
        for index, (key, batch) in enumerate(iterator, 1):
            rows.extend(batch)
            print(f"STRUCTURE {index:03d}/{len(jobs)} {key}", flush=True)
    finally:
        if pool is not None:
            pool.close(); pool.join()
    predictions: dict[str, pd.DataFrame] = {}
    for pair, target, config in jobs:
        prediction, _ = load_prediction(args, pair, target, config)
        key = f"{pair}|{target}|{config['config_id']}"
        predictions[key] = persistence.attach_entry_evidence(prediction, panel, pair)
    frame = pd.DataFrame(rows).sort_values(
        ["pair", "structure_eligible", "structure_score", "interval_count", "outside_anchor_share"],
        ascending=[True, False, False, True, True],
    ).reset_index(drop=True)
    frame.to_csv(args.output_dir / "structural_search.csv", index=False)
    return frame, predictions


def row_spec(row: Mapping[str, Any], predictions: Mapping[str, pd.DataFrame]) -> tuple[Any, str, str, str, Any]:
    return (predictions[str(row["model_key"])], str(row["pair"]), "long", str(row["target"]), engine.gate_from_row(row))


def objective_rank(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["profit_percentile"] = result.oos_pnl_fdusd.rank(pct=True)
    result["drawdown_percentile"] = result.stitched_max_drawdown_pct.rank(pct=True)
    result["objective_score"] = 0.5 * result.profit_percentile + 0.5 * result.drawdown_percentile
    return result


def search(args: argparse.Namespace, panel: pd.DataFrame, candles: Mapping[str, pd.DataFrame], selections: pd.DataFrame) -> dict[str, Any]:
    structural, predictions = structural_candidates(args, panel)
    pools: dict[str, pd.DataFrame] = {}
    for pair in PAIRS:
        group = structural[structural.pair.eq(pair)]
        eligible = group[group.structure_eligible]
        pools[pair] = (eligible if not eligible.empty else group).head(max(1, args.grid_top)).copy()
    references = {pair: pools[pair].iloc[0].to_dict() for pair in PAIRS}
    original = engine.combine_pair_gates
    pair_rows: list[dict[str, Any]] = []
    try:
        engine.combine_pair_gates = persistence.filtered_combiner(panel)
        for pair in PAIRS:
            other = next(item for item in PAIRS if item != pair)
            for index, row in enumerate(pools[pair].to_dict("records"), 1):
                metrics = engine.replay_metrics(candles, selections, [row_spec(row, predictions), row_spec(references[other], predictions)])
                pair_rows.append({**row, **metrics})
                if index % 10 == 0:
                    pd.DataFrame(pair_rows).to_csv(args.output_dir / "pair_grid_search.partial.csv", index=False)
                    print(f"PAIR GRID {pair} {index}/{len(pools[pair])}", flush=True)
        pair_grid = pd.concat([
            objective_rank(pd.DataFrame(pair_rows)[lambda value: value.pair.eq(pair)]) for pair in PAIRS
        ], ignore_index=True).sort_values(["pair", "objective_score"], ascending=[True, False])
        pair_grid.to_csv(args.output_dir / "pair_grid_search.csv", index=False)
        finalists = {pair: pair_grid[pair_grid.pair.eq(pair)].head(max(1, args.combo_top)).to_dict("records") for pair in PAIRS}
        combos: list[dict[str, Any]] = []
        for btc in finalists["BTC-FDUSD"]:
            for eth in finalists["ETH-FDUSD"]:
                metrics = engine.replay_metrics(candles, selections, [row_spec(btc, predictions), row_spec(eth, predictions)])
                combined = {
                    "candidate_id": f"{btc['candidate_id']}||{eth['candidate_id']}",
                    "BTC_candidate_id": btc["candidate_id"], "ETH_candidate_id": eth["candidate_id"],
                    **metrics,
                }
                for pair, row in (("BTC", btc), ("ETH", eth)):
                    for key in ("model_key", "target", "config_id", "prediction_sha256", "entry_quantile",
                                "recovery_quantile", "entry_bars", "recovery_bars", "minimum_hours",
                                "maximum_hours", "cooldown_hours", "interval_count", "outside_anchor_share",
                                "feb_03_06_coverage", "feb_03_06_timely", "jun_01_06_coverage", "jun_01_06_timely",
                                "anchor_pass", "active_hours"):
                        combined[f"{pair}_{key}"] = row[key]
                combos.append(combined)
        ranked = objective_rank(pd.DataFrame(combos))
    finally:
        engine.combine_pair_gates = original
    _, legacy_reference = reference_metrics(args)
    profit_floor = float(legacy_reference["oos_pnl_fdusd"])
    drawdown_floor = float(legacy_reference["stitched_max_drawdown_pct"])
    ranked["eligible"] = (
        (ranked.oos_pnl_fdusd > profit_floor)
        & (ranked.stitched_max_drawdown_pct >= drawdown_floor)
        & (ranked.btc_pnl_fdusd >= 0) & (ranked.eth_pnl_fdusd >= 0)
        & (ranked.portfolio_stop_events == 0) & (ranked.pair_stop_events < 7)
        & ranked.BTC_anchor_pass.astype(bool) & ranked.ETH_anchor_pass.astype(bool)
    )
    ranked = ranked.sort_values(
        ["eligible", "objective_score", "portfolio_stop_events", "pair_stop_events", "risk_off_pair_hours"],
        ascending=[False, False, True, True, True],
    ).reset_index(drop=True)
    ranked["rank"] = np.arange(1, len(ranked) + 1)
    ranked.to_csv(args.output_dir / "combined_grid_search.csv", index=False)
    winner = ranked.iloc[0].to_dict()
    pair_locks = {}
    for prefix, pair in (("BTC", "BTC-FDUSD"), ("ETH", "ETH-FDUSD")):
        pair_locks[pair] = {key: winner[f"{prefix}_{key}"] for key in (
            "model_key", "target", "config_id", "prediction_sha256", "entry_quantile", "recovery_quantile",
            "entry_bars", "recovery_bars", "minimum_hours", "maximum_hours", "cooldown_hours",
            "interval_count", "outside_anchor_share", "feb_03_06_coverage", "feb_03_06_timely",
            "jun_01_06_coverage", "jun_01_06_timely", "anchor_pass", "active_hours",
        )}
        pair_locks[pair]["features"] = list(FEATURES)
    lock = {
        "schema": LOCK_SCHEMA, "model_version": MODEL_VERSION,
        "evidence_status": f"same_{int((engine.END_TS - engine.START_TS) // (24 * HOUR))}d_in_sample_targeted_revalidation",
        "deployment_allowed": False, "shadow_only": True, "short_spike_enabled": False,
        "verdict": "SEARCH_LOCKED_DIAGNOSTIC", "candidate_id": winner["candidate_id"],
        "pairs": pair_locks, "search_metrics": winner,
        "profit_acceptance_floor": profit_floor, "drawdown_acceptance_floor": drawdown_floor,
        "feature_panel_sha256": sha256_file(args.source_dir / "feature_panel.csv.gz"),
        "grid_sequence_sha256": sha256_file(args.source_dir / "grid_selections.csv"),
    }
    atomic_json(args.output_dir / "locked_configuration.search.json", lock)
    return lock


def locked_specs(args: argparse.Namespace, panel: pd.DataFrame, lock: Mapping[str, Any]) -> tuple[list[Any], dict[str, pd.DataFrame]]:
    configs = {str(item["config_id"]): item for item in tune.xgb_configurations()}
    specs, predictions = [], {}
    for pair, selected in lock["pairs"].items():
        prediction, digest = load_prediction(args, pair, str(selected["target"]), configs[str(selected["config_id"])])
        if digest != selected["prediction_sha256"]:
            raise RuntimeError(f"locked prediction hash mismatch for {pair}")
        enriched = persistence.attach_entry_evidence(prediction, panel, pair)
        predictions[pair] = enriched
        specs.append((enriched, pair, "long", str(selected["target"]), engine.gate_from_row(selected)))
    return specs, predictions


def pressure_tests(args: argparse.Namespace, panel: pd.DataFrame, candles: Mapping[str, pd.DataFrame],
                   selections: pd.DataFrame, lock: Mapping[str, Any], specs: Sequence[Any]) -> pd.DataFrame:
    original = engine.combine_pair_gates
    rows = []
    try:
        engine.combine_pair_gates = persistence.filtered_combiner(panel)
        for name, fee, slippage in (
            ("base", engine.base.TAKER_FEE, 0.0),
            ("taker_150pct", engine.base.TAKER_FEE * 1.5, 0.0),
            ("slippage_0_05pct", engine.base.TAKER_FEE, 0.0005),
            ("slippage_0_10pct", engine.base.TAKER_FEE, 0.0010),
        ):
            result = engine.replay_metrics(candles, selections, specs, taker_fee=fee, slippage=slippage)
            rows.append({"scenario": name, **result,
                         "no_stops": result["pair_stop_events"] == 0 and result["portfolio_stop_events"] == 0})
        stressed_candles = crash_candles(dict(candles), 0.15)
        stressed_panel = apply_120h_purge(
            engine.v7.relabel_panel(engine.base.build_multi_horizon_panel(stressed_candles), stressed_candles)
        )
        configs = {str(item["config_id"]): {**item, "features": list(FEATURES)} for item in tune.xgb_configurations()}
        stressed_specs = []
        for pair, selected in lock["pairs"].items():
            predicted, _ = engine.weekly_prediction(stressed_panel, selections, str(selected["target"]), pair,
                                                    configs[str(selected["config_id"])])
            enriched = persistence.attach_entry_evidence(predicted, stressed_panel, pair)
            stressed_specs.append((enriched, pair, "long", str(selected["target"]), engine.gate_from_row(selected)))
        engine.combine_pair_gates = persistence.filtered_combiner(stressed_panel)
        result = engine.replay_metrics(stressed_candles, selections, stressed_specs)
        rows.append({"scenario": "single_day_15pct_drop", **result,
                     "no_stops": result["pair_stop_events"] == 0 and result["portfolio_stop_events"] == 0})
    finally:
        engine.combine_pair_gates = original
    return pd.DataFrame(rows)


def train_bundle(args: argparse.Namespace, panel: pd.DataFrame, lock: Mapping[str, Any]) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    configs = {str(item["config_id"]): item for item in tune.xgb_configurations()}
    models, thresholds, gates, targets, config_ids, audits = {}, {}, {}, {}, {}, {}
    for pair, selected in lock["pairs"].items():
        working = engine.v7.working_target(panel, str(selected["target"]))
        working = working[working.pair.eq(pair)].copy()
        mature, core, validation = engine.split_mature_training(working, engine.END_TS)
        config = {**configs[str(selected["config_id"])], "features": list(FEATURES)}
        model, audit = engine.fit_one_group(config, list(FEATURES), mature, core, validation)
        values = model.predict_proba(validation[list(FEATURES)])[:, 1]
        entry = float(pd.Series(values).quantile(float(selected["entry_quantile"])))
        recovery = float(pd.Series(values).quantile(float(selected["recovery_quantile"])))
        if not np.isfinite([entry, recovery]).all() or entry <= recovery:
            raise RuntimeError(f"degenerate production hysteresis for {pair}: {entry} <= {recovery}")
        models[pair] = model
        thresholds[pair] = {"entry": entry, "recovery": recovery}
        gates[pair] = asdict(engine.gate_from_row(selected))
        targets[pair] = selected["target"]
        config_ids[pair] = selected["config_id"]
        audits[pair] = {
            **audit, "mature_rows": len(mature), "core_rows": len(core), "calibration_rows": len(validation),
            "last_label_ready_ts": int(mature.label_ready_ts.max()), "training_cutoff_ts": engine.END_TS,
        }
    bundle = {
        "schema": "xgboost-grid-long-risk-gate-v16-model-bundle-v1",
        "model_version": MODEL_VERSION,
        "channels": {"long": {
            "architecture": "separate", "features": list(FEATURES), "models": models,
            "thresholds": thresholds, "gates": gates, "targets": targets, "config_ids": config_ids,
            "entry_evidence": "probability_rising_3h_or_roc_sqz_worsening_8h",
        }},
        "training_audit": audits,
    }
    path = args.output_dir / "models" / "xgboost_grid_long_risk_gate_v16.joblib"
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)
    loaded = joblib.load(path)
    sample = panel[panel.signal_ts < engine.END_TS].groupby("pair", group_keys=False).tail(32)
    maximum_error = 0.0
    for pair in PAIRS:
        rows = sample[sample.pair.eq(pair)]
        before = models[pair].predict_proba(rows[list(FEATURES)])[:, 1]
        after = loaded["channels"]["long"]["models"][pair].predict_proba(rows[list(FEATURES)])[:, 1]
        maximum_error = max(maximum_error, float(np.max(np.abs(before - after))))
    serialization = {"rows": len(sample), "maximum_probability_absolute_error": maximum_error,
                     "passed": maximum_error <= 1e-12}
    return path, bundle, serialization


def finalize(args: argparse.Namespace, panel: pd.DataFrame, candles: Mapping[str, pd.DataFrame], selections: pd.DataFrame) -> dict[str, Any]:
    lock_path = args.output_dir / "locked_configuration.search.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if sha256_file(args.source_dir / "feature_panel.csv.gz") != lock["feature_panel_sha256"]:
        raise RuntimeError("locked feature panel hash mismatch")
    specs, predictions = locked_specs(args, panel, lock)
    original = engine.combine_pair_gates
    try:
        engine.combine_pair_gates = persistence.filtered_combiner(panel)
        detailed = engine.detailed_replay(candles, selections, specs, MODEL_VERSION)
        neutral_specs = []
        for prediction, pair, channel, target, gate in specs:
            neutral = prediction.copy()
            neutral["probability"] = 0.0
            neutral[engine.v5.quantile_column(gate.entry_quantile)] = 1.0
            neutral[engine.v5.quantile_column(gate.recovery_quantile)] = 0.5
            neutral_specs.append((neutral, pair, channel, target, gate))
        no_gate_metrics = engine.replay_metrics(candles, selections, neutral_specs)

        corrected_v15_metrics = None
        v15_lock_path = args.v15_dir / "locked_eth_long_configuration.json"
        if (args.source_dir / "final_metrics.csv").exists() and v15_lock_path.exists():
            v9_summary, _, v9_predictions = persistence.load_locked_inputs(args.source_dir)
            v9_specs = persistence.specifications(v9_summary, v9_predictions)
            btc_v14_long = [spec for spec in v9_specs if spec[1] == "BTC-FDUSD" and spec[2] == "long"]
            v15_lock = json.loads(v15_lock_path.read_text(encoding="utf-8"))
            v15_selected = v15_lock["candidate"]
            v15_prediction_path = Path(v15_lock["prediction_file"])
            if sha256_file(v15_prediction_path) != v15_lock["prediction_sha256"]:
                raise RuntimeError("v15 comparison prediction hash mismatch")
            v15_prediction = pd.read_csv(v15_prediction_path)
            corrected_v15_specs = [
                *btc_v14_long,
                (v15_prediction, "ETH-FDUSD", "long", str(v15_selected["target"]), engine.gate_from_row(v15_selected)),
            ]
            corrected_v15_metrics = engine.replay_metrics(candles, selections, corrected_v15_specs)
    finally:
        engine.combine_pair_gates = original
    for filename, frame, compression in (
        ("final_risk_states.csv.gz", detailed["states"], "gzip"),
        ("final_risk_events.csv", detailed["events"], None),
        ("final_risk_intervals.csv", detailed["intervals"], None),
        ("final_equity_curve.csv.gz", detailed["equity"], "gzip"),
        ("final_trades.csv.gz", detailed["trades"], "gzip"),
        ("final_stop_events.csv", detailed["stops"], None),
    ):
        frame.to_csv(args.output_dir / filename, index=False, compression=compression)
    pressure = pressure_tests(args, panel, candles, selections, lock, specs)
    pressure.to_csv(args.output_dir / "pressure_tests.csv", index=False)
    model_path, bundle, serialization = train_bundle(args, panel, lock)
    model_hash = sha256_file(model_path)
    feature_hash = sha256_json({"long": list(FEATURES)})
    metrics = detailed["summary"]
    mechanism, legacy_reference = reference_metrics(args)
    profit_floor = float(legacy_reference["oos_pnl_fdusd"])
    drawdown_floor = float(legacy_reference["stitched_max_drawdown_pct"])
    acceptance = {
        "profit_above_legacy_reference": metrics["oos_pnl_fdusd"] > profit_floor,
        "drawdown_not_worse_than_legacy_reference": metrics["stitched_max_drawdown_pct"] >= drawdown_floor,
        "both_pair_pnl_nonnegative": metrics["btc_pnl_fdusd"] >= 0 and metrics["eth_pnl_fdusd"] >= 0,
        "zero_portfolio_stops": metrics["portfolio_stop_events"] == 0,
        "fewer_than_7_pair_stops": metrics["pair_stop_events"] < 7,
        "both_pair_anchor_pass": all(bool(value["anchor_pass"]) for value in lock["pairs"].values()),
        "all_pressure_scenarios_no_stops": bool(pressure.no_stops.all()),
        "serialization_exact": bool(serialization["passed"]),
    }
    research_passed = bool(all(acceptance.values()))
    final_lock = {
        **lock,
        "schema": LOCK_SCHEMA,
        "evidence_status": f"same_{int((engine.END_TS - engine.START_TS) // (24 * HOUR))}d_in_sample_targeted_revalidation",
        "verdict": "NEXT_STAGE_JOINT_VALIDATION" if research_passed else "NO-GO",
        "deployment_allowed": False,
        "research_acceptance_passed": research_passed,
        "acceptance": acceptance,
        "final_metrics": metrics,
        "model_path": model_path.as_posix(), "model_sha256": model_hash,
        "feature_schema_sha256": feature_hash,
        "training_data_sha256": {
            pair: sha256_file(args.cache_dir / f"binance_{pair}_5m.csv") for pair in PAIRS
        },
        "serialization_check": serialization,
        "contract_schema": "grid-xgboost-long-risk-gate-v1",
    }
    atomic_json(args.output_dir / "locked_configuration.json", final_lock)
    summary = {
        "model_version": MODEL_VERSION, "deployment_allowed": False,
        "short_spike_enabled": False,
        "evidence_status": f"same_{int((engine.END_TS - engine.START_TS) // (24 * HOUR))}d_in_sample_targeted_revalidation",
        "verdict": final_lock["verdict"], "metrics": metrics,
        "acceptance": acceptance, "pairs": lock["pairs"],
        "model_sha256": model_hash, "feature_schema_sha256": feature_hash,
    }
    atomic_json(args.output_dir / "summary.json", summary)
    mechanism["scenario"] = "Mechanism 1 (historical reference)"
    comparison_rows = [
        {**mechanism, "evidence_valid": True},
        {"scenario": "No technical BUY gate", **no_gate_metrics, "evidence_valid": True},
        {"scenario": "Original legacy result (invalid pair-channel replay)",
         **legacy_reference, "evidence_valid": False},
        {"scenario": "XGBoost v16 corrected long-only", **metrics, "evidence_valid": True},
    ]
    if corrected_v15_metrics is not None:
        comparison_rows.insert(-1, {"scenario": "Corrected v15-equivalent long-only",
                                    **corrected_v15_metrics, "evidence_valid": True})
    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(args.output_dir / "comparison.csv", index=False)
    pd.DataFrame([{"pair": pair, **audit} for pair, audit in bundle["training_audit"].items()]).to_csv(
        args.output_dir / "final_training_audit.csv", index=False
    )
    return summary


def build_plot(args: argparse.Namespace) -> Path:
    states = pd.read_csv(args.output_dir / "final_risk_states.csv.gz")
    events = pd.read_csv(args.output_dir / "final_risk_events.csv")
    intervals = pd.read_csv(args.output_dir / "final_risk_intervals.csv")
    equity = pd.read_csv(args.output_dir / "final_equity_curve.csv.gz")
    candles, _ = engine.load_candles(args.cache_dir)
    period_days = int((engine.END_TS - engine.START_TS) // (24 * HOUR))
    fig = make_subplots(rows=5, cols=1, shared_xaxes=True,
                        row_heights=[0.23, 0.18, 0.18, 0.23, 0.18], vertical_spacing=0.025,
                        subplot_titles=("BTC-FDUSD", "BTC长期概率", "ETH-FDUSD", "ETH长期概率", "Grid权益与回撤"))
    shadow_indices = []
    colors = {"BTC-FDUSD": "#f59e0b", "ETH-FDUSD": "#2563eb"}
    rows_by_pair = {"BTC-FDUSD": (1, 2), "ETH-FDUSD": (3, 4)}
    for pair, (price_row, probability_row) in rows_by_pair.items():
        price = candles[pair].copy()
        price["timestamp"] = price.timestamp.astype("int64")
        price = price[price.timestamp.between(engine.START_TS, engine.END_TS, inclusive="left")]
        x = pd.to_datetime(price.timestamp, unit="s", utc=True)
        fig.add_trace(go.Scatter(x=x, y=price.close, name=f"{pair}价格", line={"width": 1.2}), row=price_row, col=1)
        sample = states[states.pair.eq(pair)].sort_values("signal_ts")
        sx = pd.to_datetime(sample.signal_ts, unit="s", utc=True)
        fig.add_trace(go.Scatter(x=sx, y=sample.probability, name=f"{pair}长期概率", line={"color": colors[pair]}), row=probability_row, col=1)
        fig.add_trace(go.Scatter(x=sx, y=sample.entry_threshold, name=f"{pair}进入阈值", line={"dash": "dash", "color": "#dc2626"}), row=probability_row, col=1)
        fig.add_trace(go.Scatter(x=sx, y=sample.recovery_threshold, name=f"{pair}恢复阈值", line={"dash": "dot", "color": "#16a34a"}), row=probability_row, col=1)
        for index, interval in enumerate(intervals[intervals.pair.eq(pair)].itertuples(index=False)):
            start, end = pd.to_datetime(interval.start_ts, unit="s", utc=True), pd.to_datetime(interval.end_ts, unit="s", utc=True)
            y0, y1 = float(price.close.min()), float(price.close.max())
            shadow_indices.append(len(fig.data))
            fig.add_trace(go.Scatter(
                x=[start, end, end, start, start], y=[y0, y0, y1, y1, y0], fill="toself",
                fillcolor="rgba(245,158,11,0.16)", line={"width": 0},
                name="长期Risk-off阴影", legendgroup="long_shadow", showlegend=(pair == PAIRS[0] and index == 0),
                hovertemplate=f"{pair}<br>{start}<br>{end}<extra></extra>",
            ), row=price_row, col=1)
        pair_events = events[events.pair.eq(pair)]
        for event_name, symbol, color in (("enter", "triangle-down", "#dc2626"), ("recover", "triangle-up", "#16a34a")):
            selected = pair_events[pair_events.event.eq(event_name)].copy()
            if not selected.empty:
                selected["timestamp"] = selected.timestamp.astype("int64")
                merged = pd.merge_asof(selected.sort_values("timestamp"), price[["timestamp", "close"]].sort_values("timestamp"),
                                       left_on="timestamp", right_on="timestamp", direction="backward")
                fig.add_trace(go.Scatter(x=pd.to_datetime(merged.timestamp, unit="s", utc=True), y=merged.close,
                                         mode="markers", marker={"symbol": symbol, "size": 10, "color": color},
                                         name=f"{pair} {'进入' if event_name == 'enter' else '退出'}"), row=price_row, col=1)
    ex = pd.to_datetime(equity.timestamp, unit="s", utc=True)
    fig.add_trace(go.Scatter(x=ex, y=equity.cumulative_oos_pnl, name="累计净收益", line={"color": "#111827"}), row=5, col=1)
    fig.add_trace(go.Scatter(x=ex, y=equity.drawdown_pct, name="回撤%", line={"color": "#dc2626"}), row=5, col=1)
    for start, end in (("2026-02-03", "2026-02-07"), ("2026-06-01", "2026-06-07")):
        fig.add_vrect(x0=start, x1=end, fillcolor="rgba(168,85,247,0.08)", line_width=1, line_dash="dot")
    visible_on = [True] * len(fig.data)
    visible_off = [False if index in shadow_indices else True for index in range(len(fig.data))]
    fig.update_layout(
        title={"text": f"XGBoost v16 BTC/ETH独立长期Risk-off（{period_days}天）<br>"
                       "<sup>无短期插针 · deployment_allowed=false</sup>",
               "x": 0.01, "xanchor": "left", "font": {"size": 18}},
        height=1250, hovermode="x unified",
        updatemenus=[{"type": "buttons", "direction": "right", "buttons": [
            {"label": "显示长期阴影", "method": "update", "args": [{"visible": visible_on}]},
            {"label": "隐藏长期阴影", "method": "update", "args": [{"visible": visible_off}]},
        ], "x": 0.0, "xanchor": "left", "y": 1.06}],
        margin={"l": 55, "r": 20, "t": 125, "b": 150},
        legend={"orientation": "h", "x": 0.0, "xanchor": "left", "y": -0.08, "yanchor": "top"},
    )
    fig.update_yaxes(automargin=False)
    html = fig.to_html(full_html=True, include_plotlyjs=True)
    html = html.replace("<head>", "<head><meta name='viewport' content='width=device-width,initial-scale=1'>", 1)
    path = args.output_dir / "xgboost_v16_long_only_riskoff_plotly.html"
    path.write_text(html, encoding="utf-8")
    return path


def write_package_docs(args: argparse.Namespace) -> None:
    rollback = """# Rollback\n\nThis package is not activated. If a future shadow service is started, stop only the signal producer and keep the Grid fail-closed. Never fall back to Mechanism 1 and never issue a market sell.\n"""
    shadow = """# Shadow runbook\n\n1. Verify model, feature, data and lock hashes.\n2. Run the producer without any authorization flag.\n3. Confirm deployment_allowed=false, shadow_mode=true and short_spike_enabled=false.\n4. Observe at least eight complete forward weeks before creating a separately signed activation lock.\n5. Signal failures must pause ordinary BUY only; SELL and the 48-hour inventory exit remain unchanged.\n"""
    (args.output_dir / "ROLLBACK.md").write_text(rollback, encoding="utf-8")
    (args.output_dir / "SHADOW_RUNBOOK.md").write_text(shadow, encoding="utf-8")
    atomic_json(args.output_dir / "producer_config.json", {
        "model_version": MODEL_VERSION,
        "contract_schema": "grid-xgboost-long-risk-gate-v1",
        "signal_filename": "xgboost_risk_gate.json",
        "poll_seconds": 60,
        "stale_after_seconds": 150,
        "deployment_allowed": False,
        "shadow_mode": True,
        "short_spike_enabled": False,
        "authorize_flag_supported": False,
    })
    atomic_json(args.output_dir / "state_transition_examples.json", {
        "clear": {"risk_off_active": False, "buy_enabled": True, "market_sell_action": False},
        "long_entry": {"risk_off_active": True, "buy_enabled": False, "market_sell_action": False},
        "long_recovery": {"risk_off_active": False, "buy_enabled": True, "market_sell_action": False},
        "fail_closed": {"risk_off_active": True, "buy_enabled": False, "market_sell_action": False},
    })
    package_files = [
        "locked_configuration.json", "summary.json", "comparison.csv", "pressure_tests.csv",
        "final_training_audit.csv", "models/xgboost_grid_long_risk_gate_v16.joblib",
        "xgboost_risk_gate.signal_sample.json", "xgboost_risk_gate.state_sample.json",
        "xgboost_v16_long_only_riskoff_plotly.html", "ROLLBACK.md", "SHADOW_RUNBOOK.md",
        "producer_config.json", "state_transition_examples.json", "docker_build_validation.json",
    ]
    present = {
        name: {"sha256": sha256_file(args.output_dir / name), "bytes": (args.output_dir / name).stat().st_size}
        for name in package_files if (args.output_dir / name).exists()
    }
    atomic_json(args.output_dir / "package_manifest.json", {
        "schema": "xgboost-grid-long-risk-gate-v16-package-v1",
        "model_version": MODEL_VERSION,
        "deployment_allowed": False,
        "short_spike_enabled": False,
        "files": present,
    })


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    panel = apply_120h_purge(pd.read_csv(args.source_dir / "feature_panel.csv.gz"))
    selections = pd.read_csv(args.source_dir / "grid_selections.csv")
    candles, quality = engine.load_candles(args.cache_dir)
    quality.to_csv(args.output_dir / "data_quality.csv", index=False)
    if args.stage in {"predict", "all"}:
        generate_predictions(args, panel, selections)
    if args.stage == "predict":
        return 0
    if args.stage in {"search", "all"}:
        search(args, panel, candles, selections)
    if args.stage == "search":
        return 0
    if args.stage in {"finalize", "all"}:
        result = finalize(args, panel, candles, selections)
    else:
        result = json.loads((args.output_dir / "summary.json").read_text(encoding="utf-8"))
    if args.stage in {"package", "all"}:
        write_package_docs(args)
    if args.stage in {"plot", "all"}:
        result["plotly"] = build_plot(args).as_posix()
        atomic_json(args.output_dir / "summary.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(main())

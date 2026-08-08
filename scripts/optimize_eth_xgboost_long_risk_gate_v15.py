#!/usr/bin/env python3
"""Re-optimize only the ETH-FDUSD long XGBoost BUY risk gate.

BTC long and both short-spike channels are frozen to the v9/v14 artifacts.
Every candidate uses weekly walk-forward predictions.  Structural long-risk
constraints are evaluated before the expensive 180-day Grid replay, and the
lock binds the exact prediction bytes used by search and finalization.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

import optimize_xgboost_roc_sqz_pair_risk_gate_v8 as engine
import refine_xgboost_v9_long_entry_persistence_v14 as v14
import tune_xgboost_momentum_stop_v2 as tune


MODEL_VERSION = "eth-xgboost-long-risk-gate-v15"
V9_DIR = Path("results/backtests/xgboost_regime_spike_pair_risk_gate_v9")
V14_DIR = Path("results/backtests/xgboost_v9_long_entry_persistence_v14")
OUTPUT_DIR = Path("results/backtests/eth_xgboost_long_risk_gate_v15")
PAIR = "ETH-FDUSD"
TARGETS = ("long_72h", "long_120h")
LONG_FEATURES = ("adx_14", "di_spread", "atr_pct", "btc_volatility_20")
LOCK_SCHEMA = "eth-xgboost-long-risk-gate-v15-lock-v1"
PREDICTION_SCHEMA = "eth-xgboost-long-risk-gate-v15-prediction-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("predict", "search", "lock", "finalize", "plot", "all"), default="all")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--xgb-threads", type=int, default=2)
    parser.add_argument("--grid-top", type=int, default=160)
    parser.add_argument("--cache-dir", type=Path, default=Path("data/backtesting_candles"))
    parser.add_argument("--v9-dir", type=Path, default=V9_DIR)
    parser.add_argument("--v14-dir", type=Path, default=V14_DIR)
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


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    os.replace(temporary, path)


def prediction_path(output_dir: Path, target: str, config_id: str) -> Path:
    return output_dir / "prediction_cache" / f"{target}__{PAIR}__{config_id}.csv.gz"


def metadata(args: argparse.Namespace, target: str, config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema": PREDICTION_SCHEMA,
        "model_version": MODEL_VERSION,
        "target": target,
        "pair": PAIR,
        "configuration_sha256": hashlib.sha256(canonical(dict(config))).hexdigest(),
        "features": list(LONG_FEATURES),
        "feature_panel_sha256": sha256_file(args.v9_dir / "feature_panel.csv.gz"),
        "grid_sequence_sha256": sha256_file(args.v9_dir / "grid_selections.csv"),
        "target_definition_version": engine.TARGETS[target]["definition_version"],
    }


_PANEL: pd.DataFrame | None = None
_SELECTIONS: pd.DataFrame | None = None
_ARGS: argparse.Namespace | None = None


def init_worker(panel: pd.DataFrame, selections: pd.DataFrame, args: argparse.Namespace) -> None:
    global _PANEL, _SELECTIONS, _ARGS
    _PANEL, _SELECTIONS, _ARGS = panel, selections, args
    tune.XGB_N_JOBS = int(args.xgb_threads)
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"


def prediction_worker(job: tuple[str, dict[str, Any]]) -> tuple[str, str, str]:
    if _PANEL is None or _SELECTIONS is None or _ARGS is None:
        raise RuntimeError("prediction worker was not initialized")
    target, base_config = job
    config = {**base_config, "features": list(LONG_FEATURES)}
    path = prediction_path(_ARGS.output_dir, target, str(config["config_id"]))
    meta_path = path.with_name(path.name + ".metadata.json")
    expected = metadata(_ARGS, target, config)
    if _ARGS.resume and path.exists() and meta_path.exists():
        observed = json.loads(meta_path.read_text(encoding="utf-8"))
        if observed == expected and observed.get("prediction_sha256") == sha256_file(path):
            return target, str(config["config_id"]), "reused"
    prediction, audit = engine.weekly_prediction(_PANEL, _SELECTIONS, target, PAIR, config)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    prediction.to_csv(temporary, index=False, compression="gzip")
    os.replace(temporary, path)
    audit.to_csv(path.with_suffix(".audit.csv"), index=False)
    expected["prediction_sha256"] = sha256_file(path)
    expected["rows"] = len(prediction)
    expected["last_label_mature_ok"] = bool((audit.last_mature_label_ready_ts <= audit.train_cutoff_ts).all())
    atomic_json(meta_path, expected)
    return target, str(config["config_id"]), "trained"


def generate_predictions(args: argparse.Namespace, panel: pd.DataFrame, selections: pd.DataFrame) -> None:
    jobs = [(target, config) for target in TARGETS for config in tune.xgb_configurations()]
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
        for index, (target, config_id, status) in enumerate(iterator, 1):
            print(f"PREDICT {index:02d}/80 {target} {config_id} [{status}]", flush=True)
    finally:
        if pool is not None:
            pool.close(); pool.join()


def load_prediction_bound(args: argparse.Namespace, target: str, config: Mapping[str, Any]) -> tuple[pd.DataFrame, str]:
    configured = {**config, "features": list(LONG_FEATURES)}
    path = prediction_path(args.output_dir, target, str(config["config_id"]))
    observed = json.loads(path.with_name(path.name + ".metadata.json").read_text(encoding="utf-8"))
    actual = sha256_file(path)
    if observed != {**metadata(args, target, configured), "prediction_sha256": actual, "rows": observed.get("rows"),
                    "last_label_mature_ok": observed.get("last_label_mature_ok")}:
        raise RuntimeError(f"prediction metadata mismatch: {path}")
    if not observed.get("last_label_mature_ok"):
        raise RuntimeError(f"immature label audit: {path}")
    return pd.read_csv(path), actual


def gate_candidates() -> list[engine.v5.GateParameters]:
    # The approved deterministic v8 long-state search includes all entry
    # quantiles while bounding the expensive state/Grid Cartesian product.
    return engine.refinement_gates("long")


def structural_row(prediction: pd.DataFrame, panel: pd.DataFrame, target: str,
                   config_id: str, prediction_sha: str, gate: engine.v5.GateParameters) -> dict[str, Any]:
    intervals = fast_long_intervals(prediction, gate)
    metrics = engine.pair_anchor_metrics(intervals, PAIR)
    active_hours = float(intervals.duration_hours.sum()) if not intervals.empty else 0.0
    coverage_sum = sum(float(metrics[f"{name}_coverage"]) for name, _, _ in engine.ANCHOR_WINDOWS)
    timely_count = sum(bool(metrics[f"{name}_timely"]) for name, _, _ in engine.ANCHOR_WINDOWS)
    overlap_penalty = max(int(metrics["interval_count"]) - 8, 0)
    structure_score = (
        2.0 * timely_count + coverage_sum - 3.0 * max(float(metrics["outside_anchor_share"]) - 0.20, 0.0)
        - 0.10 * overlap_penalty
    )
    return {
        "candidate_id": f"{target}|{PAIR}|{config_id}|{engine.gate_id('long', gate)}",
        "target": target, "config_id": config_id, "prediction_sha256": prediction_sha,
        **asdict(gate), **metrics, "active_hours": active_hours,
        "structure_score": float(structure_score), "structure_eligible": bool(metrics["anchor_pass"]),
        "state_rows": len(prediction),
    }


def fast_long_intervals(enriched: pd.DataFrame, gate: engine.v5.GateParameters) -> pd.DataFrame:
    """Evaluate structural constraints without constructing a 5-minute gate.

    Full timeline materialization is required for Grid replay but is wasteful
    during the first-stage 10,240-state structural screen.
    """
    entry_col = engine.v5.quantile_column(gate.entry_quantile)
    recovery_col = engine.v5.quantile_column(gate.recovery_quantile)
    columns = [
        "signal_ts", "probability", entry_col, recovery_col,
        "probability_lag_2h", "probability_rising_3h", "roc_sqz_worsening_8h",
    ]
    state = engine.v5.GateState()
    interval_start: int | None = None
    rows: list[dict[str, Any]] = []
    for row in enriched[columns].itertuples(index=False, name=None):
        timestamp, probability, entry, recovery, lag2, rising, worsening = row
        minimum_rise = max(1e-4, 0.25 * max(float(entry) - float(recovery), 0.0))
        evidence = bool(
            (bool(rising) and np.isfinite(lag2) and float(probability) - float(lag2) >= minimum_rise)
            or bool(worsening)
        )
        effective = float(probability)
        if not state.active and not evidence:
            effective = min(effective, float(np.nextafter(float(entry), -np.inf)))
        state, transition, reason = engine.v5.step_gate(
            effective, float(entry), float(recovery), int(timestamp), state, gate
        )
        if transition == "enter":
            interval_start = int(timestamp)
        elif transition == "recover" and interval_start is not None:
            rows.append({
                "pair": PAIR, "start_ts": interval_start, "end_ts": int(timestamp),
                "duration_hours": (int(timestamp) - interval_start) / engine.HOUR,
                "end_reason": reason,
            })
            interval_start = None
    if interval_start is not None:
        rows.append({
            "pair": PAIR, "start_ts": interval_start, "end_ts": engine.END_TS,
            "duration_hours": (engine.END_TS - interval_start) / engine.HOUR,
            "end_reason": "research_period_end",
        })
    return pd.DataFrame(rows, columns=["pair", "start_ts", "end_ts", "duration_hours", "end_reason"])


def fixed_specifications(args: argparse.Namespace) -> tuple[dict[str, Any], pd.DataFrame, dict[str, pd.DataFrame], list[Any]]:
    summary, context, predictions = v14.load_locked_inputs(args.v9_dir)
    specs = v14.specifications(summary, predictions)
    fixed = [spec for spec in specs if not (spec[1] == PAIR and spec[2] == "long")]
    return summary, context, predictions, fixed


def run_search(args: argparse.Namespace, panel: pd.DataFrame, candles: Mapping[str, pd.DataFrame],
               selections: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    loaded: dict[tuple[str, str], tuple[pd.DataFrame, str]] = {}
    configs = {str(item["config_id"]): item for item in tune.xgb_configurations()}
    for target in TARGETS:
        for config_id, config in configs.items():
            prediction, prediction_sha = load_prediction_bound(args, target, config)
            loaded[(target, config_id)] = (prediction, prediction_sha)
            enriched = v14.attach_entry_evidence(prediction, panel, PAIR)
            for gate in gate_candidates():
                rows.append(structural_row(enriched, panel, target, config_id, prediction_sha, gate))
        print(f"STRUCTURE {target} complete", flush=True)
    structural = pd.DataFrame(rows).sort_values(
        ["structure_eligible", "structure_score", "interval_count", "outside_anchor_share"],
        ascending=[False, False, True, True],
    ).reset_index(drop=True)
    structural.to_csv(args.output_dir / "eth_long_structural_search.csv", index=False)
    eligible = structural[structural.structure_eligible].copy()
    pool = eligible if not eligible.empty else structural
    evaluated = pool.head(max(1, int(args.grid_top))).copy()
    _, _, _, fixed = fixed_specifications(args)
    original = engine.combine_pair_gates
    grid_rows = []
    try:
        engine.combine_pair_gates = v14.filtered_combiner(panel)
        for index, row in enumerate(evaluated.itertuples(index=False), 1):
            prediction, actual_sha = loaded[(str(row.target), str(row.config_id))]
            if actual_sha != str(row.prediction_sha256):
                raise RuntimeError("search prediction hash changed before Grid replay")
            gate = engine.gate_from_row(row._asdict())
            metrics = engine.replay_metrics(candles, selections, [*fixed, (prediction, PAIR, "long", str(row.target), gate)])
            grid_rows.append({**row._asdict(), **metrics})
            if index % 10 == 0:
                pd.DataFrame(grid_rows).to_csv(args.output_dir / "eth_long_grid_search.partial.csv", index=False)
                print(f"GRID {index}/{len(evaluated)}", flush=True)
    finally:
        engine.combine_pair_gates = original
    ranked = pd.DataFrame(grid_rows)
    ranked["profit_percentile"] = ranked.oos_pnl_fdusd.rank(pct=True)
    ranked["drawdown_percentile"] = ranked.stitched_max_drawdown_pct.rank(pct=True)
    ranked["objective_score"] = 0.5 * ranked.profit_percentile + 0.5 * ranked.drawdown_percentile
    ranked["frequency_constraints_pass"] = (
        (ranked.interval_count <= 8) & (ranked.outside_anchor_share <= 0.20)
    )
    ranked["minimum_anchor_coverage"] = ranked[
        [f"{name}_coverage" for name, _, _ in engine.ANCHOR_WINDOWS]
    ].min(axis=1)
    ranked["timely_anchor_count"] = sum(
        ranked[f"{name}_timely"].astype(bool).astype(int) for name, _, _ in engine.ANCHOR_WINDOWS
    )
    ranked["eligible"] = (
        ranked.structure_eligible & (ranked.oos_pnl_fdusd > 4.08906229455954)
        & (ranked.stitched_max_drawdown_pct >= -9.263364315297606)
        & (ranked.portfolio_stop_events == 0) & (ranked.pair_stop_events < 7)
        & (ranked.btc_pnl_fdusd >= 0) & (ranked.eth_pnl_fdusd >= 0)
    )
    ranked = ranked.sort_values(
        ["eligible", "objective_score", "frequency_constraints_pass", "minimum_anchor_coverage",
         "timely_anchor_count", "portfolio_stop_events", "pair_stop_events", "active_hours"],
        ascending=[False, False, False, False, False, True, True, True],
    ).reset_index(drop=True)
    ranked["rank"] = np.arange(1, len(ranked) + 1)
    ranked.to_csv(args.output_dir / "eth_long_grid_search.csv", index=False)
    winner = ranked.iloc[0].to_dict()
    lock = {
        "schema": LOCK_SCHEMA, "model_version": MODEL_VERSION,
        "created_from": "same_180d_in_sample_targeted_optimization",
        "deployment_allowed": False, "verdict": "DIAGNOSTIC_ONLY",
        "candidate": winner,
        "prediction_file": prediction_path(args.output_dir, str(winner["target"]), str(winner["config_id"])).as_posix(),
        "prediction_sha256": str(winner["prediction_sha256"]),
        "feature_panel_sha256": sha256_file(args.v9_dir / "feature_panel.csv.gz"),
        "grid_sequence_sha256": sha256_file(args.v9_dir / "grid_selections.csv"),
    }
    atomic_json(args.output_dir / "locked_eth_long_configuration.json", lock)
    return ranked


def relock_existing(args: argparse.Namespace) -> dict[str, Any]:
    """Re-rank completed Grid rows without retraining or replaying them."""
    ranked = pd.read_csv(args.output_dir / "eth_long_grid_search.csv")
    ranked["frequency_constraints_pass"] = (
        (ranked.interval_count <= 8) & (ranked.outside_anchor_share <= 0.20)
    )
    ranked["minimum_anchor_coverage"] = ranked[
        [f"{name}_coverage" for name, _, _ in engine.ANCHOR_WINDOWS]
    ].min(axis=1)
    ranked["timely_anchor_count"] = sum(
        ranked[f"{name}_timely"].astype(bool).astype(int) for name, _, _ in engine.ANCHOR_WINDOWS
    )
    ranked = ranked.sort_values(
        ["eligible", "objective_score", "frequency_constraints_pass", "minimum_anchor_coverage",
         "timely_anchor_count", "portfolio_stop_events", "pair_stop_events", "active_hours"],
        ascending=[False, False, False, False, False, True, True, True],
    ).reset_index(drop=True)
    ranked["rank"] = np.arange(1, len(ranked) + 1)
    ranked.to_csv(args.output_dir / "eth_long_grid_search.csv", index=False)
    winner = ranked.iloc[0].to_dict()
    lock = {
        "schema": LOCK_SCHEMA, "model_version": MODEL_VERSION,
        "created_from": "same_180d_in_sample_targeted_optimization",
        "deployment_allowed": False, "verdict": "DIAGNOSTIC_ONLY",
        "candidate": winner,
        "prediction_file": prediction_path(args.output_dir, str(winner["target"]), str(winner["config_id"])).as_posix(),
        "prediction_sha256": str(winner["prediction_sha256"]),
        "feature_panel_sha256": sha256_file(args.v9_dir / "feature_panel.csv.gz"),
        "grid_sequence_sha256": sha256_file(args.v9_dir / "grid_selections.csv"),
    }
    atomic_json(args.output_dir / "locked_eth_long_configuration.json", lock)
    return lock


def finalize(args: argparse.Namespace, panel: pd.DataFrame, candles: Mapping[str, pd.DataFrame],
             selections: pd.DataFrame) -> dict[str, Any]:
    lock = json.loads((args.output_dir / "locked_eth_long_configuration.json").read_text(encoding="utf-8"))
    candidate = lock["candidate"]
    path = Path(lock["prediction_file"])
    if sha256_file(path) != lock["prediction_sha256"]:
        raise RuntimeError("locked prediction hash mismatch; refusing finalization")
    if sha256_file(args.v9_dir / "feature_panel.csv.gz") != lock["feature_panel_sha256"]:
        raise RuntimeError("locked feature hash mismatch; refusing finalization")
    prediction = pd.read_csv(path)
    _, _, _, fixed = fixed_specifications(args)
    gate = engine.gate_from_row(candidate)
    specs = [*fixed, (prediction, PAIR, "long", str(candidate["target"]), gate)]
    original = engine.combine_pair_gates
    try:
        engine.combine_pair_gates = v14.filtered_combiner(panel)
        detailed = engine.detailed_replay(candles, selections, specs, MODEL_VERSION)
    finally:
        engine.combine_pair_gates = original
    if abs(float(detailed["summary"]["oos_pnl_fdusd"]) - float(candidate["oos_pnl_fdusd"])) > 1e-9:
        raise RuntimeError("locked search/final profit mismatch")
    for name, frame, compression in (
        ("final_risk_states.csv.gz", detailed["states"], "gzip"),
        ("final_risk_events.csv", detailed["events"], None),
        ("final_risk_intervals.csv", detailed["intervals"], None),
        ("final_equity_curve.csv.gz", detailed["equity"], "gzip"),
        ("final_trades.csv.gz", detailed["trades"], "gzip"),
        ("final_stop_events.csv", detailed["stops"], None),
    ):
        frame.to_csv(args.output_dir / name, index=False, compression=compression)
    old = json.loads((args.v9_dir / "summary.json").read_text(encoding="utf-8"))["winner_metrics"]
    comparison = pd.DataFrame([
        {"scenario": "XGBoost v9", **old},
        {"scenario": "XGBoost v14 persistence filter", **json.loads((args.v14_dir / "summary.json").read_text(encoding="utf-8"))["refined_metrics"]},
        {"scenario": "ETH XGBoost v15 optimized", **detailed["summary"]},
    ])
    comparison.to_csv(args.output_dir / "comparison.csv", index=False)
    result = {
        "model_version": MODEL_VERSION, "deployment_allowed": False,
        "evidence_status": "same_180d_in_sample_targeted_optimization",
        "locked_candidate": candidate, "final_metrics": detailed["summary"],
        "search_final_prediction_hash_match": True,
        "verdict": "NO-GO" if not bool(candidate.get("eligible")) else "NEXT_STAGE_JOINT_VALIDATION",
    }
    atomic_json(args.output_dir / "summary.json", result)
    return result


def build_plot(args: argparse.Namespace) -> Path:
    comparison = pd.read_csv(args.output_dir / "comparison.csv")
    source = v14.build_plot(args, comparison)
    target = args.output_dir / "eth_xgboost_v15_long_riskoff_plotly.html"
    page = source.read_text(encoding="utf-8")
    page = page.replace("XGBoost v14", "ETH XGBoost v15")
    target.write_text(page, encoding="utf-8")
    return target


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    panel = pd.read_csv(args.v9_dir / "feature_panel.csv.gz")
    selections = pd.read_csv(args.v9_dir / "grid_selections.csv")
    candles, _ = engine.load_candles(args.cache_dir)
    if args.stage in {"predict", "all"}:
        generate_predictions(args, panel, selections)
    if args.stage == "predict":
        return 0
    if args.stage in {"search", "all"}:
        run_search(args, panel, candles, selections)
    if args.stage == "search":
        return 0
    if args.stage == "lock":
        print(json.dumps(relock_existing(args), ensure_ascii=False, indent=2, default=str))
        return 0
    if args.stage in {"finalize", "all"}:
        result = finalize(args, panel, candles, selections)
    else:
        result = json.loads((args.output_dir / "summary.json").read_text(encoding="utf-8"))
    if args.stage in {"plot", "all"}:
        plot = build_plot(args)
        result["plotly"] = plot.as_posix()
        atomic_json(args.output_dir / "summary.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(main())

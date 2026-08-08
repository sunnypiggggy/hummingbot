#!/usr/bin/env python3
"""Retrain independent BTC/ETH long XGBoost Grid BUY gates over 250 days.

The two known long-risk windows are hard, in-sample selection constraints.
The short-spike XGBoost gates are frozen; no model emits a market sell.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np
import pandas as pd

import optimize_xgboost_roc_sqz_pair_risk_gate_v8 as engine
import optimize_xgboost_grid_risk_gate_v7 as v7
import refine_xgboost_v9_long_entry_persistence_v14 as v14
import revalidate_eth_xgboost_long_risk_gate_v15_250d as source250
import retrain_eth_xgboost_anchor_focused_v16 as v16
import tune_xgboost_momentum_stop_v2 as tune


MODEL_VERSION = "xgboost-independent-long-risk-gate-v17-250d"
OUTPUT_DIR = Path("results/backtests/xgboost_independent_long_risk_gate_v17_250d")
SOURCE_DIR = Path("results/backtests/eth_xgboost_long_risk_gate_v15_250d")
V9_DIR = Path("results/backtests/xgboost_regime_spike_pair_risk_gate_v9")
PAIRS = ("BTC-FDUSD", "ETH-FDUSD")
TARGETS = ("long_72h", "long_120h")
START_TS, END_TS = source250.START_TS, source250.END_TS
LOCK_SCHEMA = "xgboost-independent-long-risk-gate-v17-lock-v1"

FEATURE_SETS = {
    "regime": (
        "adx_14", "di_spread", "atr_pct", "btc_volatility_20",
    ),
    "regime_roc_sqz": (
        "adx_14", "di_spread", "atr_pct", "btc_volatility_20",
        "roc_20", "roc_48h_4h", "sqzmom_pct_4h", "sqzmom_slope_4h",
    ),
}
WEIGHT_PROFILES = ("balanced", "persistent_severity")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("screen", "weekly", "search", "finalize", "plot", "all"), default="all")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--xgb-threads", type=int, default=2)
    parser.add_argument("--finalists-per-pair-target", type=int, default=4)
    parser.add_argument("--pair-structural-top", type=int, default=12)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--source-dir", type=Path, default=SOURCE_DIR)
    parser.add_argument("--v9-dir", type=Path, default=V9_DIR)
    return parser.parse_args()


def configure_period() -> None:
    if END_TS - START_TS != 250 * 86400:
        raise AssertionError("research interval must be exactly 250 days")
    engine.START_TS = v7.START_TS = START_TS
    engine.END_TS = v7.END_TS = END_TS


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    os.replace(temporary, path)


def specs() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for pair in PAIRS:
        for target in TARGETS:
            for feature_id, features in FEATURE_SETS.items():
                for profile in WEIGHT_PROFILES:
                    for config in tune.xgb_configurations():
                        key = "|".join((pair, target, feature_id, profile, str(config["config_id"])))
                        rows.append({"model_key": key, "pair": pair, "target": target,
                                     "feature_id": feature_id, "features": list(features),
                                     "weight_profile": profile, "config": config})
    return rows


def cache_path(args: argparse.Namespace, stage: str, spec: Mapping[str, Any]) -> Path:
    return args.output_dir / "prediction_cache" / stage / f"{str(spec['model_key']).replace('|', '__')}.csv.gz"


def cache_payload(args: argparse.Namespace, stage: str, spec: Mapping[str, Any]) -> dict[str, Any]:
    payload = {
        "schema": "xgboost-v17-prediction-v1", "model_version": MODEL_VERSION,
        "stage": stage, "spec": spec,
        "panel_sha256": sha256_file(args.source_dir / "feature_panel.csv.gz"),
        "grid_sha256": sha256_file(args.source_dir / "grid_selections.csv"),
        "start_ts": START_TS, "end_ts": END_TS,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    return {**payload, "payload_sha256": hashlib.sha256(encoded).hexdigest()}


_PANEL: pd.DataFrame | None = None
_SELECTIONS: pd.DataFrame | None = None
_ARGS: argparse.Namespace | None = None


def init_worker(panel: pd.DataFrame, selections: pd.DataFrame, args: argparse.Namespace) -> None:
    global _PANEL, _SELECTIONS, _ARGS
    _PANEL, _SELECTIONS, _ARGS = panel, selections, args
    tune.XGB_N_JOBS = int(args.xgb_threads)
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = "1"


def predict_block(panel: pd.DataFrame, spec: Mapping[str, Any], block: Any) -> tuple[pd.DataFrame, dict[str, Any]]:
    working = engine.v7.working_target(panel, str(spec["target"]))
    working = working[working.pair.eq(str(spec["pair"]))].copy()
    mature, core, validation = tune.split_mature_training(working, int(block.train_end))
    # v16 applies mature-label filtering and 120-hour target readiness in the
    # shared splitter. The model itself is then refit on all mature rows.
    model, audit = v16.fit_model(spec, mature, core, validation)
    test = working[(working.signal_ts >= int(block.test_start)) &
                   (working.signal_ts < int(block.test_end))].copy()
    features = list(spec["features"])
    prediction = test[["pair", "signal_ts", "target"]].copy()
    prediction["probability"] = model.predict_proba(test[features])[:, 1]
    calibration = validation[["pair", "signal_ts", "target"]].copy()
    calibration["probability"] = model.predict_proba(validation[features])[:, 1]
    prediction = engine.attach_thresholds(prediction, calibration)
    prediction["strategy"] = str(spec["target"])
    return prediction, {
        "model_key": spec["model_key"], "train_cutoff_ts": int(block.train_end),
        "last_mature_label_ready_ts": int(mature.label_ready_ts.max()),
        "first_test_signal_ts": int(test.signal_ts.min()), **audit,
    }


def worker(job: tuple[str, dict[str, Any]]) -> tuple[str, str]:
    if _PANEL is None or _SELECTIONS is None or _ARGS is None:
        raise RuntimeError("worker not initialized")
    stage, spec = job
    path = cache_path(_ARGS, stage, spec); meta_path = path.with_name(path.name + ".metadata.json")
    expected = cache_payload(_ARGS, stage, spec)
    if _ARGS.resume and path.exists() and meta_path.exists():
        observed = json.loads(meta_path.read_text(encoding="utf-8"))
        if all(observed.get(k) == v for k, v in expected.items()) and observed.get("prediction_sha256") == sha256_file(path):
            return str(spec["model_key"]), "reused"
    blocks = ([SimpleNamespace(train_end=START_TS, test_start=START_TS, test_end=END_TS)]
              if stage == "screen" else list(_SELECTIONS.itertuples(index=False)))
    predictions, audits = [], []
    for block in blocks:
        prediction, audit = predict_block(_PANEL, spec, block)
        predictions.append(prediction); audits.append(audit)
    audit_frame = pd.DataFrame(audits)
    if not (audit_frame.last_mature_label_ready_ts <= audit_frame.train_cutoff_ts).all():
        raise AssertionError("immature label entered training")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    pd.concat(predictions, ignore_index=True).to_csv(temporary, index=False, compression="gzip")
    os.replace(temporary, path)
    audit_frame.to_csv(path.with_suffix(".audit.csv"), index=False)
    atomic_json(meta_path, {**expected, "prediction_sha256": sha256_file(path),
                            "rows": sum(map(len, predictions))})
    return str(spec["model_key"]), "trained"


def run_jobs(args: argparse.Namespace, panel: pd.DataFrame, selections: pd.DataFrame,
             stage: str, selected: list[dict[str, Any]]) -> None:
    jobs = [(stage, spec) for spec in selected]
    workers = min(max(1, int(args.workers)), len(jobs))
    if workers == 1:
        init_worker(panel, selections, args); iterator = map(worker, jobs); pool = None
    else:
        pool = mp.get_context("spawn").Pool(workers, initializer=init_worker,
                                             initargs=(panel, selections, args), maxtasksperchild=4)
        iterator = pool.imap_unordered(worker, jobs, chunksize=1)
    try:
        for index, (key, status) in enumerate(iterator, 1):
            print(f"{stage.upper()} {index}/{len(jobs)} {key} [{status}]", flush=True)
    finally:
        if pool is not None:
            pool.close(); pool.join()


def screen(args: argparse.Namespace, all_specs: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for spec in all_specs:
        prediction = pd.read_csv(cache_path(args, "screen", spec))
        ts = prediction.signal_ts.to_numpy(np.int64)
        for quantile in engine.ENTRY_QUANTILES:
            active = prediction.probability.to_numpy(float) >= prediction[engine.v5.quantile_column(quantile)].to_numpy(float)
            anchor_mask = np.zeros(len(ts), dtype=bool); values: dict[str, Any] = {}
            coverages, timely = [], []
            for name, start, end in engine.ANCHOR_WINDOWS:
                mask = (ts >= start) & (ts < end); anchor_mask |= mask
                coverage = float(active[mask].mean()) if mask.any() else 0.0
                entered = bool(active[(ts >= start) & (ts <= start + 12 * engine.HOUR)].any())
                values.update({f"{name}_coverage": coverage, f"{name}_timely": entered})
                coverages.append(coverage); timely.append(entered)
            outside = float(active[~anchor_mask].mean()) if (~anchor_mask).any() else 1.0
            rows.append({"model_key": spec["model_key"], "pair": spec["pair"], "target": spec["target"],
                         "feature_id": spec["feature_id"], "weight_profile": spec["weight_profile"],
                         "config_id": spec["config"]["config_id"], "entry_quantile": quantile,
                         **values, "minimum_anchor_coverage": min(coverages),
                         "timely_anchor_count": sum(timely), "outside_active_share": outside,
                         "screen_pass": min(coverages) >= .70 and all(timely) and outside <= .20,
                         "screen_score": min(coverages) + .5 * sum(timely) - 2 * outside})
    ranked = pd.DataFrame(rows).sort_values(
        ["screen_pass", "screen_score", "minimum_anchor_coverage", "outside_active_share"],
        ascending=[False, False, False, True])
    ranked.to_csv(args.output_dir / "fixed_origin_anchor_screen.csv", index=False)
    return ranked


def finalists(args: argparse.Namespace, ranked: pd.DataFrame,
              all_specs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys: set[str] = set()
    for pair in PAIRS:
        for target in TARGETS:
            part = ranked[(ranked.pair == pair) & (ranked.target == target)]
            keys.update(part.drop_duplicates("model_key").head(args.finalists_per_pair_target).model_key.astype(str))
    selected = [item for item in all_specs if str(item["model_key"]) in keys]
    atomic_json(args.output_dir / "weekly_finalists.json", selected)
    return selected


def bind_interval_pair(intervals: pd.DataFrame, pair: str) -> pd.DataFrame:
    """Correctly bind intervals returned by the legacy ETH-only fast path."""
    output = intervals.copy()
    if not output.empty:
        output["pair"] = pair
    return output


def structural_candidates(args: argparse.Namespace, panel: pd.DataFrame,
                          selected: list[dict[str, Any]]) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    context = panel[["pair", "signal_ts", *v14.CONTEXT_FEATURES]]
    rows, predictions = [], {}
    for spec in selected:
        path = cache_path(args, "weekly", spec); prediction = pd.read_csv(path)
        prediction_sha = sha256_file(path); predictions[str(spec["model_key"])] = prediction
        enriched = v14.attach_entry_evidence(prediction, context, str(spec["pair"]))
        for gate in v16.v15.gate_candidates():
            intervals = v16.v15.fast_long_intervals(enriched, gate)
            # The reused v15 fast evaluator was originally ETH-only and writes
            # its module-level PAIR into every interval. Rebind here so BTC
            # anchor metrics cannot silently evaluate an empty group.
            intervals = bind_interval_pair(intervals, str(spec["pair"]))
            metrics = engine.pair_anchor_metrics(intervals, str(spec["pair"]))
            rows.append({"model_key": spec["model_key"], "pair": spec["pair"], "target": spec["target"],
                         "feature_id": spec["feature_id"], "weight_profile": spec["weight_profile"],
                         "config_id": spec["config"]["config_id"], "prediction_sha256": prediction_sha,
                         **gate.__dict__, **metrics,
                         "active_hours": float(intervals.duration_hours.sum()) if not intervals.empty else 0.0})
    frame = pd.DataFrame(rows)
    frame["minimum_anchor_coverage"] = frame[[f"{name}_coverage" for name, _, _ in engine.ANCHOR_WINDOWS]].min(axis=1)
    frame["structure_pass"] = frame.anchor_pass & (frame.interval_count <= 8) & (frame.outside_anchor_share <= .20)
    frame = frame.sort_values(["structure_pass", "minimum_anchor_coverage", "outside_anchor_share", "interval_count", "active_hours"],
                              ascending=[False, False, True, True, True])
    frame.to_csv(args.output_dir / "weekly_structural_search.csv", index=False)
    return frame, predictions


def frozen_short_specs(args: argparse.Namespace) -> list[tuple[pd.DataFrame, str, str, str, Any]]:
    summary = json.loads((args.v9_dir / "summary.json").read_text(encoding="utf-8"))
    result = []
    for pair, config_id in (("BTC-FDUSD", "xgb_35"), ("ETH-FDUSD", "xgb_12")):
        path = args.source_dir / "prediction_cache" / f"short_1h_6h__{pair}__{config_id}.csv.gz"
        if not path.exists():
            raise FileNotFoundError(path)
        row = summary["pair_winners"][pair]
        result.append((pd.read_csv(path), pair, "short", "short_1h_6h", engine.gate_from_row(row, "short_")))
    return result


def search(args: argparse.Namespace, panel: pd.DataFrame, selections: pd.DataFrame,
           candles: Mapping[str, pd.DataFrame], selected: list[dict[str, Any]]) -> pd.DataFrame:
    structure, predictions = structural_candidates(args, panel, selected)
    choices = {pair: structure[structure.pair.eq(pair)].head(args.pair_structural_top) for pair in PAIRS}
    shorts = frozen_short_specs(args); context = panel[["pair", "signal_ts", *v14.CONTEXT_FEATURES]]
    rows = []; original = engine.combine_pair_gates
    try:
        engine.combine_pair_gates = v14.filtered_combiner(context)
        total = len(choices[PAIRS[0]]) * len(choices[PAIRS[1]])
        index = 0
        for btc in choices[PAIRS[0]].itertuples(index=False):
            for eth in choices[PAIRS[1]].itertuples(index=False):
                index += 1
                specifications = [*shorts]
                for row in (btc, eth):
                    specifications.append((predictions[str(row.model_key)], str(row.pair), "long", str(row.target),
                                           engine.gate_from_row(row._asdict())))
                metrics = engine.replay_metrics(candles, selections, specifications)
                rows.append({"candidate_id": f"{btc.model_key}::{eth.model_key}::{index}",
                             "btc_model_key": btc.model_key, "eth_model_key": eth.model_key,
                             "btc_target": btc.target, "eth_target": eth.target,
                             "btc_prediction_sha256": btc.prediction_sha256,
                             "eth_prediction_sha256": eth.prediction_sha256,
                             **{f"btc_{k}": v for k, v in btc._asdict().items() if k not in {"pair", "model_key", "target"}},
                             **{f"eth_{k}": v for k, v in eth._asdict().items() if k not in {"pair", "model_key", "target"}},
                             **metrics})
                if index % 10 == 0: print(f"GRID {index}/{total}", flush=True)
    finally:
        engine.combine_pair_gates = original
    ranked = pd.DataFrame(rows)
    ranked["profit_percentile"] = ranked.oos_pnl_fdusd.rank(pct=True)
    ranked["drawdown_percentile"] = ranked.stitched_max_drawdown_pct.rank(pct=True)
    ranked["objective_score"] = .5 * ranked.profit_percentile + .5 * ranked.drawdown_percentile
    ranked["window_pass"] = ranked.btc_structure_pass & ranked.eth_structure_pass
    ranked["eligible"] = (ranked.window_pass & (ranked.oos_pnl_fdusd > 0) &
                          (ranked.portfolio_stop_events == 0) &
                          (ranked.btc_pnl_fdusd >= 0) & (ranked.eth_pnl_fdusd >= 0))
    ranked = ranked.sort_values(["eligible", "window_pass", "objective_score", "portfolio_stop_events",
                                 "pair_stop_events", "risk_off_pair_hours"],
                                ascending=[False, False, False, True, True, True]).reset_index(drop=True)
    ranked["rank"] = np.arange(1, len(ranked) + 1)
    ranked.to_csv(args.output_dir / "grid_search.csv", index=False)
    winner = ranked.iloc[0].to_dict()
    atomic_json(args.output_dir / "locked_configuration.json", {
        "schema": LOCK_SCHEMA, "model_version": MODEL_VERSION, "deployment_allowed": False,
        "evidence_status": "250d_known_window_in_sample_targeted_optimization",
        "candidate": winner, "panel_sha256": sha256_file(args.source_dir / "feature_panel.csv.gz"),
        "grid_sha256": sha256_file(args.source_dir / "grid_selections.csv"),
    })
    return ranked


def winner_specs(args: argparse.Namespace, selected: list[dict[str, Any]]) -> tuple[list[Any], dict[str, Any]]:
    lock = json.loads((args.output_dir / "locked_configuration.json").read_text(encoding="utf-8")); row = lock["candidate"]
    by_key = {str(item["model_key"]): item for item in selected}; output = frozen_short_specs(args)
    for prefix, pair in (("btc", "BTC-FDUSD"), ("eth", "ETH-FDUSD")):
        spec = by_key[str(row[f"{prefix}_model_key"])]
        path = cache_path(args, "weekly", spec)
        if sha256_file(path) != row[f"{prefix}_prediction_sha256"]:
            raise RuntimeError(f"locked {pair} prediction hash mismatch")
        gate_data = {key[len(prefix) + 1:]: value for key, value in row.items() if key.startswith(prefix + "_")}
        output.append((pd.read_csv(path), pair, "long", str(row[f"{prefix}_target"]), engine.gate_from_row(gate_data)))
    return output, lock


def finalize(args: argparse.Namespace, panel: pd.DataFrame, selections: pd.DataFrame,
             candles: Mapping[str, pd.DataFrame], selected: list[dict[str, Any]]) -> dict[str, Any]:
    specifications, lock = winner_specs(args, selected); original = engine.combine_pair_gates
    try:
        engine.combine_pair_gates = v14.filtered_combiner(panel[["pair", "signal_ts", *v14.CONTEXT_FEATURES]])
        detail = engine.detailed_replay(candles, selections, specifications, MODEL_VERSION)
    finally:
        engine.combine_pair_gates = original
    candidate = lock["candidate"]
    if abs(float(detail["summary"]["oos_pnl_fdusd"]) - float(candidate["oos_pnl_fdusd"])) > 1e-8:
        raise RuntimeError("locked search and final replay differ")
    for name, frame, compression in (
        ("final_risk_states.csv.gz", detail["states"], "gzip"),
        ("final_risk_events.csv", detail["events"], None),
        ("final_risk_intervals.csv", detail["intervals"], None),
        ("final_equity_curve.csv.gz", detail["equity"], "gzip"),
        ("final_trades.csv.gz", detail["trades"], "gzip"),
        ("final_stop_events.csv", detail["stops"], None),
    ):
        frame.to_csv(args.output_dir / name, index=False, compression=compression)
    baseline = engine.baseline_metrics(candles, selections)
    comparison = pd.DataFrame([{"scenario": "Mechanism 1 - 250d", **baseline},
                               {"scenario": "XGBoost v17 - 250d", **detail["summary"]}])
    comparison.to_csv(args.output_dir / "comparison.csv", index=False)
    result = {"model_version": MODEL_VERSION, "period_days": 250,
              "start_utc": pd.to_datetime(START_TS, unit="s", utc=True).isoformat(),
              "end_utc": pd.to_datetime(END_TS, unit="s", utc=True).isoformat(),
              "deployment_allowed": False, "evidence_status": lock["evidence_status"],
              "baseline": baseline, "metrics": detail["summary"], "locked_candidate": candidate,
              "verdict": "NEXT_STAGE_JOINT_VALIDATION" if bool(candidate.get("eligible")) else "NO-GO"}
    atomic_json(args.output_dir / "summary.json", result)
    return result


def plot(args: argparse.Namespace) -> Path:
    # Reuse the self-contained BTC/ETH Plotly builder: long/short shadows keep
    # independent legend switches and transition markers remain exact.
    args.cache_dir = args.source_dir / "extended_candles"
    source = v14.build_plot(args, pd.read_csv(args.output_dir / "comparison.csv"))
    target = args.output_dir / "xgboost_v17_250d_riskoff_plotly.html"
    page = source.read_text(encoding="utf-8")
    title = "XGBoost v17：BTC/ETH独立长期Risk-off（250天）"
    page = re.sub(r"<title>.*?</title>", f"<title>{title}</title>", page, count=1, flags=re.S)
    page = re.sub(r"<h1>.*?</h1>", f"<h1>{title}</h1>", page, count=1, flags=re.S)
    target.write_text(page, encoding="utf-8")
    return target


def main() -> int:
    mp.freeze_support(); configure_period(); args = parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)
    panel = pd.read_csv(args.source_dir / "feature_panel.csv.gz")
    selections = pd.read_csv(args.source_dir / "grid_selections.csv")
    candles, quality = engine.load_candles(args.source_dir / "extended_candles")
    quality.to_csv(args.output_dir / "data_quality.csv", index=False)
    all_specs = specs()
    if args.stage in {"screen", "all"}:
        run_jobs(args, panel, selections, "screen", all_specs)
        selected = finalists(args, screen(args, all_specs), all_specs)
    else:
        selected = json.loads((args.output_dir / "weekly_finalists.json").read_text(encoding="utf-8"))
    if args.stage == "screen": return 0
    if args.stage in {"weekly", "all"}: run_jobs(args, panel, selections, "weekly", selected)
    if args.stage == "weekly": return 0
    if args.stage in {"search", "all"}: search(args, panel, selections, candles, selected)
    if args.stage == "search": return 0
    result = finalize(args, panel, selections, candles, selected) if args.stage in {"finalize", "all"} else json.loads(
        (args.output_dir / "summary.json").read_text(encoding="utf-8"))
    if args.stage in {"plot", "all"}:
        result["plotly"] = plot(args).as_posix(); atomic_json(args.output_dir / "summary.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str)); return 0


if __name__ == "__main__":
    raise SystemExit(main())

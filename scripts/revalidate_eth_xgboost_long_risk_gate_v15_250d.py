#!/usr/bin/env python3
"""Revalidate the locked ETH XGBoost v15 gate over the preceding 250 days."""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import time
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import requests

import backtest_xgboost_long_risk_gate_180d as base
import optimize_eth_xgboost_long_risk_gate_v15 as v15
import optimize_xgboost_grid_risk_gate_v7 as v7
import optimize_xgboost_roc_sqz_pair_risk_gate_v8 as engine
import refine_xgboost_v9_long_entry_persistence_v14 as v14
import tune_xgboost_momentum_stop_v2 as tune


MODEL_VERSION = "eth-xgboost-long-risk-gate-v15-250d-revalidation"
START_TS = int(pd.Timestamp("2025-11-23T15:00:00Z").timestamp())
END_TS = int(pd.Timestamp("2026-07-31T15:00:00Z").timestamp())
RAW_START_TS = int(pd.Timestamp("2025-10-01T00:00:00Z").timestamp())
OUTPUT_DIR = Path("results/backtests/eth_xgboost_long_risk_gate_v15_250d")
DATA_DIR_NAME = "extended_candles"
SOURCE_WEEKLY = Path("results/backtests/fdusd_inventory_exit_parameter_search/weekly_results.csv")
LOCKED_V15 = Path("results/backtests/eth_xgboost_long_risk_gate_v15/locked_eth_long_configuration.json")
PAIRS = ("BTC-FDUSD", "ETH-FDUSD")
LONG_FEATURES = ("adx_14", "di_spread", "atr_pct", "btc_volatility_20")
SHORT_FEATURES = ("price_to_ema20_atr", "volume_zscore", "di_spread")
PREDICTION_SCHEMA = "eth-xgboost-v15-250d-prediction-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prepare", "predict", "replay", "plot", "all"), default="all")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--xgb-threads", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--source-weekly-results", type=Path, default=SOURCE_WEEKLY)
    parser.add_argument("--locked-v15", type=Path, default=LOCKED_V15)
    return parser.parse_args()


def configure_period() -> None:
    if END_TS - START_TS != 250 * 86400:
        raise AssertionError("250-day interval is not exact")
    engine.START_TS = START_TS
    engine.END_TS = END_TS
    v7.START_TS = START_TS
    v7.END_TS = END_TS


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    os.replace(temporary, path)


def download_pair(pair: str, path: Path) -> None:
    symbol = pair.replace("-", "")
    cursor = RAW_START_TS * 1000
    end_ms = (END_TS + 5 * 60) * 1000
    rows: list[list[Any]] = []
    session = requests.Session()
    while cursor < end_ms:
        response = session.get(
            "https://api.binance.com/api/v3/klines",
            params={"symbol": symbol, "interval": "5m", "startTime": cursor,
                    "endTime": end_ms - 1, "limit": 1000}, timeout=30,
        )
        response.raise_for_status()
        batch = response.json()
        if not batch:
            break
        rows.extend(batch)
        cursor = int(batch[-1][0]) + 300_000
        time.sleep(0.03)
    frame = pd.DataFrame(rows, columns=[
        "timestamp", "open", "high", "low", "close", "volume", "close_time",
        "quote_volume", "trades", "taker_base", "taker_quote", "ignore",
    ])
    if frame.empty:
        raise RuntimeError(f"Binance returned no candles for {pair}")
    frame["timestamp"] = frame.timestamp.astype("int64") // 1000
    frame = frame.drop_duplicates("timestamp").sort_values("timestamp")
    expected = pd.RangeIndex(RAW_START_TS, END_TS + 300, 300)
    missing = expected.difference(pd.Index(frame.timestamp.astype("int64")))
    if len(missing):
        raise RuntimeError(f"{pair} has {len(missing)} missing 5-minute candles; refusing 250d replay")
    path.parent.mkdir(parents=True, exist_ok=True)
    frame[["timestamp", "open", "high", "low", "close", "volume"]].to_csv(path, index=False)


def windows() -> pd.DataFrame:
    rows, cursor, fold = [], START_TS, 1
    while cursor < END_TS:
        right = min(cursor + 7 * 86400, END_TS)
        rows.append({"period": "revalidation_250d", "fold": fold,
                     "train_start": cursor - 14 * 86400, "train_end": cursor,
                     "test_start": cursor, "test_end": right})
        cursor, fold = right, fold + 1
    result = pd.DataFrame(rows)
    if int((result.test_end - result.test_start).sum()) != 250 * 86400:
        raise AssertionError("weekly folds do not span 250 days")
    return result


def prepare(args: argparse.Namespace) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame, Path]:
    data_dir = args.output_dir / DATA_DIR_NAME
    for pair in PAIRS:
        path = data_dir / f"binance_{pair}_5m.csv"
        if not (args.resume and path.exists()):
            download_pair(pair, path)
    candles, quality = engine.load_candles(data_dir)
    quality.to_csv(args.output_dir / "data_quality.csv", index=False)
    selection_path = args.output_dir / "grid_selections.csv"
    if args.resume and selection_path.exists():
        selections = pd.read_csv(selection_path)
    else:
        selections, audit = base.frozen_grid_sequence(windows(), args.source_weekly_results)
        selections.to_csv(selection_path, index=False)
        audit.to_csv(args.output_dir / "grid_selection_audit.csv", index=False)
    panel_path = args.output_dir / "feature_panel.csv.gz"
    if args.resume and panel_path.exists():
        panel = pd.read_csv(panel_path)
    else:
        panel = v7.relabel_panel(base.build_multi_horizon_panel(candles), candles)
        panel.to_csv(panel_path, index=False, compression="gzip")
    if panel.signal_ts.min() > START_TS - 30 * 86400:
        raise RuntimeError("feature panel lacks sufficient pre-period training history")
    return candles, panel, selections, data_dir


def model_jobs(args: argparse.Namespace) -> list[dict[str, Any]]:
    lock = json.loads(args.locked_v15.read_text(encoding="utf-8"))["candidate"]
    return [
        {"key": "long_120h|BTC-FDUSD|xgb_03", "target": "long_120h", "pair": "BTC-FDUSD",
         "config_id": "xgb_03", "features": LONG_FEATURES},
        {"key": "short_1h_6h|BTC-FDUSD|xgb_35", "target": "short_1h_6h", "pair": "BTC-FDUSD",
         "config_id": "xgb_35", "features": SHORT_FEATURES},
        {"key": f"{lock['target']}|ETH-FDUSD|{lock['config_id']}", "target": str(lock["target"]),
         "pair": "ETH-FDUSD", "config_id": str(lock["config_id"]), "features": LONG_FEATURES},
        {"key": "short_1h_6h|ETH-FDUSD|xgb_12", "target": "short_1h_6h", "pair": "ETH-FDUSD",
         "config_id": "xgb_12", "features": SHORT_FEATURES},
    ]


_PANEL: pd.DataFrame | None = None
_SELECTIONS: pd.DataFrame | None = None
_ARGS: argparse.Namespace | None = None


def init_worker(panel: pd.DataFrame, selections: pd.DataFrame, args: argparse.Namespace) -> None:
    global _PANEL, _SELECTIONS, _ARGS
    _PANEL, _SELECTIONS, _ARGS = panel, selections, args
    tune.XGB_N_JOBS = int(args.xgb_threads)
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = "1"


def predict_worker(job: Mapping[str, Any]) -> tuple[str, str]:
    if _PANEL is None or _SELECTIONS is None or _ARGS is None:
        raise RuntimeError("worker not initialized")
    configs = {item["config_id"]: item for item in tune.xgb_configurations()}
    config = {**configs[str(job["config_id"])], "features": list(job["features"])}
    path = _ARGS.output_dir / "prediction_cache" / f"{str(job['key']).replace('|', '__')}.csv.gz"
    meta_path = path.with_name(path.name + ".metadata.json")
    expected = {
        "schema": PREDICTION_SCHEMA, "job": dict(job),
        "feature_panel_sha256": sha256_file(_ARGS.output_dir / "feature_panel.csv.gz"),
        "grid_sequence_sha256": sha256_file(_ARGS.output_dir / "grid_selections.csv"),
        "locked_v15_sha256": sha256_file(_ARGS.locked_v15),
    }
    if _ARGS.resume and path.exists() and meta_path.exists():
        observed = json.loads(meta_path.read_text(encoding="utf-8"))
        if observed == {**expected, "prediction_sha256": sha256_file(path)}:
            return str(job["key"]), "reused"
    prediction, audit = engine.weekly_prediction(
        _PANEL, _SELECTIONS, str(job["target"]), str(job["pair"]), config
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    prediction.to_csv(path, index=False, compression="gzip")
    audit.to_csv(path.with_suffix(".audit.csv"), index=False)
    if not (audit.last_mature_label_ready_ts <= audit.train_cutoff_ts).all():
        raise AssertionError("immature training label")
    write_json(meta_path, {**expected, "prediction_sha256": sha256_file(path)})
    return str(job["key"]), "trained"


def predict_all(args: argparse.Namespace, panel: pd.DataFrame, selections: pd.DataFrame) -> None:
    jobs = model_jobs(args)
    workers = min(max(1, int(args.workers)), len(jobs))
    if workers == 1:
        init_worker(panel, selections, args); iterator = map(predict_worker, jobs); pool = None
    else:
        pool = mp.get_context("spawn").Pool(workers, initializer=init_worker,
                                             initargs=(panel, selections, args))
        iterator = pool.imap_unordered(predict_worker, jobs)
    try:
        for key, status in iterator:
            print(f"PREDICT {key} [{status}]", flush=True)
    finally:
        if pool is not None:
            pool.close(); pool.join()


def specifications(args: argparse.Namespace) -> list[Any]:
    v9 = json.loads(v15.V9_DIR.joinpath("summary.json").read_text(encoding="utf-8"))
    lock = json.loads(args.locked_v15.read_text(encoding="utf-8"))["candidate"]
    rows = v9["pair_winners"]
    result = []
    for job in model_jobs(args):
        path = args.output_dir / "prediction_cache" / f"{str(job['key']).replace('|', '__')}.csv.gz"
        meta = json.loads(path.with_name(path.name + ".metadata.json").read_text(encoding="utf-8"))
        if meta["prediction_sha256"] != sha256_file(path):
            raise RuntimeError(f"prediction hash mismatch: {path}")
        pair, target = str(job["pair"]), str(job["target"])
        channel = "short" if target.startswith("short") else "long"
        gate_row = lock if pair == "ETH-FDUSD" and channel == "long" else rows[pair]
        gate = engine.gate_from_row(gate_row, "" if gate_row is lock else f"{channel}_")
        result.append((pd.read_csv(path), pair, channel, target, gate))
    return result


def replay(args: argparse.Namespace, candles: Mapping[str, pd.DataFrame], panel: pd.DataFrame,
           selections: pd.DataFrame) -> dict[str, Any]:
    baseline = engine.baseline_metrics(candles, selections)
    original = engine.combine_pair_gates
    try:
        engine.combine_pair_gates = v14.filtered_combiner(panel)
        detailed = engine.detailed_replay(candles, selections, specifications(args), MODEL_VERSION)
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
    eth_long = detailed["intervals"][(detailed["intervals"].pair == "ETH-FDUSD")
                                      & (detailed["intervals"].channel == "long")]
    structure = engine.pair_anchor_metrics(eth_long, "ETH-FDUSD")
    comparison = pd.DataFrame([
        {"scenario": "Mechanism 1 — 250d", **baseline},
        {"scenario": "Locked ETH XGBoost v15 — 250d", **detailed["summary"]},
    ])
    comparison.to_csv(args.output_dir / "comparison.csv", index=False)
    result = {
        "model_version": MODEL_VERSION, "period_days": 250,
        "start_utc": pd.to_datetime(START_TS, unit="s", utc=True).isoformat(),
        "end_utc": pd.to_datetime(END_TS, unit="s", utc=True).isoformat(),
        "parameter_selection": "locked_from_180d_v15_no_250d_retuning",
        "deployment_allowed": False, "baseline": baseline,
        "metrics": detailed["summary"], "eth_long_structure": structure,
        "eth_long_active_hours": float(eth_long.duration_hours.sum()),
        "prediction_hashes": {
            job["key"]: sha256_file(args.output_dir / "prediction_cache" /
                                    f"{str(job['key']).replace('|', '__')}.csv.gz")
            for job in model_jobs(args)
        },
        "verdict": "REVALIDATION_ONLY",
    }
    write_json(args.output_dir / "summary.json", result)
    return result


def build_plot(args: argparse.Namespace, data_dir: Path) -> Path:
    metrics = pd.read_csv(args.output_dir / "comparison.csv")
    source = v14.build_plot(args, metrics)
    target = args.output_dir / "eth_xgboost_v15_250d_riskoff_plotly.html"
    page = source.read_text(encoding="utf-8")
    page = page.replace(
        "XGBoost v14：持续概率或ROC/SQZ恶化长期Risk-off",
        "ETH XGBoost v15：锁定参数250天复测",
    )
    page = page.replace("180天", "250天").replace("2026-02-01 15:00", "2025-11-23 15:00")
    target.write_text(page, encoding="utf-8")
    return target


def main() -> int:
    mp.freeze_support(); configure_period()
    args = parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)
    candles, panel, selections, data_dir = prepare(args)
    args.cache_dir = data_dir
    if args.stage == "prepare": return 0
    if args.stage in {"predict", "all"}: predict_all(args, panel, selections)
    if args.stage == "predict": return 0
    if args.stage in {"replay", "all"}: result = replay(args, candles, panel, selections)
    else: result = json.loads((args.output_dir / "summary.json").read_text(encoding="utf-8"))
    if args.stage in {"plot", "all"}:
        plot = build_plot(args, data_dir); result["plotly"] = plot.as_posix()
        write_json(args.output_dir / "summary.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

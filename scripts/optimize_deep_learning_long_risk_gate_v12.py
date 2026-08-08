#!/usr/bin/env python3
"""Train and validate a dual-resolution deep-learning long Risk-off BUY gate.

The Grid remains the trading strategy.  The PyTorch model replaces only the
long channel; the locked v11 XGBoost short channel remains unchanged.  No model
signal can create a sell order or authorize deployment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import platform
import sys
import time
from dataclasses import asdict
from itertools import product
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import backtest_xgboost_long_risk_gate_180d as base
import optimize_xgboost_dual_risk_gate_180d_v5 as v5
import optimize_xgboost_feature_selected_pair_risk_gate_v11 as v11
import optimize_xgboost_grid_risk_gate_v7 as v7
import optimize_xgboost_roc_sqz_pair_risk_gate_v8 as engine
from compare_independent_gate_ml_stops import HOUR, PAIRS, load_candles
from deep_learning_long_risk_models_v12 import (
    DualBranchLongRiskModel, RobustSequenceScaler, deterministic_configurations,
    seed_everything,
)
from tune_xgboost_momentum_stop_v2 import sha256_file, write_json
from validate_grid_live import crash_candles


MODEL_VERSION = "deep-learning-long-risk-gate-v12"
SCHEMA = "grid-hybrid-risk-gate-v1"
OUTPUT_DIR = Path("results/backtests/deep_learning_long_risk_gate_v12")
V11_DIR = Path("results/backtests/xgboost_feature_selected_pair_risk_gate_v11")
START_TS, END_TS = v7.START_TS, v7.END_TS
ENTRY_QUANTILES = v7.ENTRY_QUANTILES
SEEDS = (42, 43, 44)
OLD_PNL = 4.08906229455954
OLD_DRAWDOWN = -9.263364315297606
OLD_PAIR_STOPS = 7
PREDICTION_CACHE_COMPATIBLE_TRAINER_HASHES = {
    # CUDA/architecture support revision used for the completed Transformer
    # walk-forward run.  Later edits below only add final-model checkpointing.
    "a569eefaad04fa097b7e96924c9d8683d5ca3b116d5aebaa1f6f2bb48f0df05c",
}

HOURLY_FEATURES = (
    "return_1", "return_5", "return_20", "roc_5", "roc_20", "adx_14",
    "di_spread", "atr_pct", "btc_volatility_20", "drawdown_from_high_72h",
    "drawdown_from_high_168h", "drawdown_duration_168h", "below_ema20_ratio_72h",
    "lower_low_ratio_72h", "downside_semivariance_ratio_24h",
    "downside_semivariance_ratio_72h", "rv_24h_percentile_30d",
    "vol_of_vol_72h", "trend_efficiency_72h", "ema20_slope_atr_12h",
    "historical_var_72h", "expected_shortfall_72h", "negative_skew_72h",
    "taker_sell_share_24h", "taker_sell_share_72h", "trade_count_zscore_72h",
)
FIVE_FEATURES = (
    "log_return", "range_pct", "close_location", "log_volume",
    "log_quote_volume", "log_trades", "taker_buy_ratio",
    "taker_sell_imbalance", "amihud", "cross_return",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prepare", "train", "search", "finalize", "plot", "all"), default="all")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--torch-threads", type=int, default=2)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--architecture", choices=("all", "tcn", "gru", "transformer"), default="all",
        help="Limit training/search to one architecture without changing its deterministic configurations.",
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--v11-dir", type=Path, default=V11_DIR)
    parser.add_argument("--cache-dir", type=Path, default=Path("data/backtesting_candles"))
    parser.add_argument("--max-configs", type=int, default=24, help="Test-only limiter; production default covers all 24.")
    parser.add_argument("--max-epochs", type=int, default=100)
    return parser.parse_args()


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def file_hashes(paths: Sequence[Path]) -> str:
    payload = "|".join(f"{path}:{sha256_file(path)}" for path in paths)
    return hashlib.sha256(payload.encode()).hexdigest()


def prediction_metadata_compatible(cached: Mapping[str, Any], expected: Mapping[str, Any]) -> bool:
    if dict(cached) == dict(expected):
        return True
    if cached.get("trainer_sha256") not in PREDICTION_CACHE_COMPATIBLE_TRAINER_HASHES:
        return False
    cached_without_trainer = {key: value for key, value in cached.items() if key != "trainer_sha256"}
    expected_without_trainer = {key: value for key, value in expected.items() if key != "trainer_sha256"}
    return cached_without_trainer == expected_without_trainer


def make_five_frame(own: pd.DataFrame, cross: pd.DataFrame) -> pd.DataFrame:
    left = own.sort_values("timestamp").copy()
    right = cross[["timestamp", "close"]].sort_values("timestamp").rename(columns={"close": "cross_close"})
    item = left.merge(right, on="timestamp", how="inner", validate="one_to_one")
    log_close = np.log(item.close.astype(float))
    item["log_return"] = log_close.diff()
    item["range_pct"] = (item.high - item.low) / item.open.replace(0, np.nan)
    item["close_location"] = (item.close - item.low) / (item.high - item.low).replace(0, np.nan)
    item["log_volume"] = np.log1p(item.volume)
    item["log_quote_volume"] = np.log1p(item.quote_asset_volume)
    item["log_trades"] = np.log1p(item.n_trades)
    item["taker_buy_ratio"] = item.taker_buy_base_volume / item.volume.replace(0, np.nan)
    item["taker_sell_imbalance"] = 1 - 2 * item.taker_buy_ratio
    item["amihud"] = item.log_return.abs() / item.quote_asset_volume.replace(0, np.nan)
    item["cross_return"] = np.log(item.cross_close.astype(float)).diff()
    return item[["timestamp", *FIVE_FEATURES]].replace([np.inf, -np.inf], np.nan)


def build_sequence_cache(args: argparse.Namespace) -> dict[str, Path]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.output_dir / "sequence_cache"
    cache_dir.mkdir(exist_ok=True)
    panel_path = args.v11_dir / "feature_panel.csv.gz"
    panel = pd.read_csv(panel_path)
    micro_paths = {
        "BTC-FDUSD": args.v11_dir / "source_cache/binance_BTCUSDT_5m_micro.csv.gz",
        "ETH-FDUSD": args.v11_dir / "source_cache/binance_ETHUSDT_5m_micro.csv.gz",
    }
    micro = {pair: pd.read_csv(path) for pair, path in micro_paths.items()}
    source_hash = file_hashes([panel_path, *micro_paths.values()])
    outputs = {}
    quality = []
    for pair in PAIRS:
        target = cache_dir / f"{pair}_dual_sequence.npz"
        metadata = cache_dir / f"{pair}_dual_sequence.metadata.json"
        expected = {
            "schema": "deep-long-v12-dual-sequence-v1", "source_hash": source_hash,
            "pair": pair, "hourly_features": list(HOURLY_FEATURES),
            "five_features": list(FIVE_FEATURES), "hourly_steps": 168, "five_steps": 288,
        }
        if args.resume and target.exists() and metadata.exists() and json.loads(metadata.read_text()) == expected:
            outputs[pair] = target
            with np.load(target) as cached:
                quality.append({"pair": pair, "rows": len(cached["signal_ts"]), "cache_reused": True})
            continue
        group = panel[panel.pair.eq(pair)].sort_values("signal_ts").reset_index(drop=True)
        cross_pair = "ETH-FDUSD" if pair.startswith("BTC") else "BTC-FDUSD"
        five = make_five_frame(micro[pair], micro[cross_pair]).dropna().reset_index(drop=True)
        five_ts = five.timestamp.to_numpy(dtype=np.int64)
        five_values = five[list(FIVE_FEATURES)].to_numpy(dtype=np.float32)
        hourly_values = group[list(HOURLY_FEATURES)].to_numpy(dtype=np.float32)
        hourly_ts = group.signal_ts.to_numpy(dtype=np.int64)
        rows_h, rows_f, rows_y, rows_ts, sequence_end = [], [], [], [], []
        for index in range(167, len(group)):
            signal_ts = int(hourly_ts[index])
            five_right = int(np.searchsorted(five_ts, signal_ts, side="left"))
            five_left = five_right - 288
            if five_left < 0 or five_right <= five_left:
                continue
            selected_ts = five_ts[five_left:five_right]
            if len(selected_ts) != 288 or np.any(np.diff(selected_ts) != 300):
                continue
            hour_window = hourly_values[index - 167:index + 1]
            if not np.isfinite(hour_window).all() or not np.isfinite(five_values[five_left:five_right]).all():
                continue
            rows_h.append(hour_window)
            rows_f.append(five_values[five_left:five_right])
            rows_y.append([float(group.target_long_72h.iloc[index]), float(group.target_long_120h.iloc[index])])
            rows_ts.append(signal_ts)
            sequence_end.append(int(selected_ts[-1]))
        np.savez_compressed(
            target, hourly=np.asarray(rows_h, dtype=np.float32), five=np.asarray(rows_f, dtype=np.float32),
            target=np.asarray(rows_y, dtype=np.float32), signal_ts=np.asarray(rows_ts, dtype=np.int64),
            sequence_end_ts=np.asarray(sequence_end, dtype=np.int64),
        )
        write_json(metadata, expected)
        outputs[pair] = target
        quality.append({
            "pair": pair, "rows": len(rows_ts), "cache_reused": False,
            "first_signal_ts": min(rows_ts), "last_signal_ts": max(rows_ts),
            "all_execute_next_5m": bool(np.all(np.asarray(sequence_end) + 300 == np.asarray(rows_ts))),
        })
    atomic_csv(pd.DataFrame(quality), args.output_dir / "sequence_data_quality.csv")
    pd.DataFrame({"hourly_feature": pd.Series(HOURLY_FEATURES), "five_feature": pd.Series(FIVE_FEATURES)}).to_csv(
        args.output_dir / "sequence_feature_contract.csv", index=False
    )
    return outputs


def sigmoid_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    values = np.clip(logits / temperature, -40, 40)
    return 1.0 / (1.0 + np.exp(-values))


def temperature_fit(logits: np.ndarray, target: np.ndarray) -> np.ndarray:
    temperatures = np.linspace(.5, 3.0, 51)
    result = []
    for column in range(2):
        losses = [log_loss(target[:, column], sigmoid_temperature(logits[:, column], t), labels=[0, 1]) for t in temperatures]
        result.append(float(temperatures[int(np.argmin(losses))]))
    return np.asarray(result, dtype=np.float32)


def tensor_loader(hourly: np.ndarray, five: np.ndarray, target: np.ndarray | None,
                  batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    tensors = [torch.from_numpy(hourly), torch.from_numpy(five)]
    if target is not None:
        tensors.append(torch.from_numpy(target))
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(TensorDataset(*tensors), batch_size=batch_size, shuffle=shuffle,
                      num_workers=0, generator=generator)


def infer_logits(model: nn.Module, hourly: np.ndarray, five: np.ndarray, batch_size: int) -> np.ndarray:
    device = next(model.parameters()).device
    model.eval()
    output = []
    with torch.no_grad():
        for batch_h, batch_f in tensor_loader(hourly, five, None, batch_size, False, 0):
            output.append(model(batch_h.to(device), batch_f.to(device)).cpu().numpy())
    return np.concatenate(output) if output else np.empty((0, 2), dtype=np.float32)


def train_fold(arrays: Mapping[str, np.ndarray], config: Mapping[str, Any], cutoff: int,
               test_start: int, test_end: int, seed: int, threads: int,
               max_epochs: int, device_name: str = "cpu", return_artifacts: bool = False):
    seed_everything(seed, threads)
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    ts, y = arrays["signal_ts"], arrays["target"]
    mature = np.flatnonzero(ts + 120 * HOUR <= cutoff)
    if len(mature) < 240:
        raise ValueError(f"insufficient mature sequence rows at cutoff {cutoff}: {len(mature)}")
    desired_validation_start = int(ts[mature].max() - 14 * 24 * HOUR)
    validation = mature[ts[mature] >= desired_validation_start]
    core = mature[ts[mature] < desired_validation_start]
    validation_policy = "last_14_mature_days"
    if len(core) < 96:
        validation_rows = min(14 * 24, max(48, len(mature) - 96))
        core, validation = mature[:-validation_rows], mature[-validation_rows:]
        validation_policy = "early_fold_shortened_keep_96_core_min_48_validation"
    testing = np.flatnonzero((ts >= test_start) & (ts < test_end))
    if len(core) < 96 or len(validation) < 48 or len(testing) == 0:
        raise ValueError("invalid core/validation/testing split")
    scaler = RobustSequenceScaler.fit(arrays["hourly"][core], arrays["five"][core])
    core_h, core_f = scaler.transform(arrays["hourly"][core], arrays["five"][core])
    val_h, val_f = scaler.transform(arrays["hourly"][validation], arrays["five"][validation])
    test_h, test_f = scaler.transform(arrays["hourly"][testing], arrays["five"][testing])
    model = DualBranchLongRiskModel(core_h.shape[-1], core_f.shape[-1], config).to(device)
    positives = y[core].sum(axis=0)
    weights = np.clip((len(core) - positives) / np.maximum(positives, 1), 1, 10).astype(np.float32)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.from_numpy(weights).to(device))
    validation_criterion = nn.BCEWithLogitsLoss(pos_weight=torch.from_numpy(weights))
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["learning_rate"]),
                                  weight_decay=float(config["weight_decay"]))
    loader = tensor_loader(core_h, core_f, y[core], int(config["batch_size"]), True, seed)
    best_loss, best_state, patience_left, epochs = float("inf"), None, int(config["patience"]), 0
    for epoch in range(min(int(config["max_epochs"]), int(max_epochs))):
        model.train()
        for batch_h, batch_f, batch_y in loader:
            batch_h = batch_h.to(device)
            batch_f = batch_f.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch_h, batch_f), batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        validation_logits = infer_logits(model, val_h, val_f, int(config["batch_size"]))
        validation_loss = float(validation_criterion(
            torch.from_numpy(validation_logits), torch.from_numpy(y[validation])
        ).item())
        epochs = epoch + 1
        if validation_loss < best_loss - 1e-5:
            best_loss = validation_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            patience_left = int(config["patience"])
        else:
            patience_left -= 1
            if patience_left <= 0:
                break
    if best_state is None:
        raise RuntimeError("training did not produce a checkpoint")
    model.load_state_dict(best_state)
    validation_logits = infer_logits(model, val_h, val_f, int(config["batch_size"]))
    temperatures = temperature_fit(validation_logits, y[validation])
    test_logits = infer_logits(model, test_h, test_f, int(config["batch_size"]))
    calibrated_validation = np.column_stack([
        sigmoid_temperature(validation_logits[:, index], float(temperatures[index])) for index in range(2)
    ])
    calibrated_test = np.column_stack([
        sigmoid_temperature(test_logits[:, index], float(temperatures[index])) for index in range(2)
    ])
    frame = pd.DataFrame({
        "signal_ts": ts[testing], "target_72h": y[testing, 0], "target_120h": y[testing, 1],
        "p72": calibrated_test[:, 0], "p120": calibrated_test[:, 1],
    })
    frame["pmean"] = frame[["p72", "p120"]].mean(axis=1)
    for head, column in (("p72", 0), ("p120", 1)):
        for quantile in sorted({*ENTRY_QUANTILES, *(max(.5, value - .10) for value in ENTRY_QUANTILES)}):
            frame[f"{head}_q{quantile:.3f}"] = float(np.quantile(calibrated_validation[:, column], quantile))
    validation_mean = calibrated_validation.mean(axis=1)
    for quantile in sorted({*ENTRY_QUANTILES, *(max(.5, value - .10) for value in ENTRY_QUANTILES)}):
        frame[f"pmean_q{quantile:.3f}"] = float(np.quantile(validation_mean, quantile))
    audit = {
        "cutoff": cutoff, "test_start": test_start, "test_end": test_end,
        "last_mature_label_ready_ts": int((ts[mature] + 120 * HOUR).max()),
        "last_core_signal_ts": int(ts[core].max()), "first_validation_signal_ts": int(ts[validation].min()),
        "last_validation_signal_ts": int(ts[validation].max()), "first_test_signal_ts": int(ts[testing].min()),
        "core_rows": len(core), "validation_rows": len(validation), "test_rows": len(testing),
        "epochs": epochs, "best_validation_loss": best_loss,
        "validation_policy": validation_policy,
        "temperature_72h": float(temperatures[0]), "temperature_120h": float(temperatures[1]),
        "parameter_count": sum(item.numel() for item in model.parameters()),
        "device": str(device),
        "torch_version": torch.__version__,
    }
    if return_artifacts:
        return frame, audit, {
            "model": model, "scaler": scaler, "test_indices": testing,
            "test_hourly": test_h, "test_five": test_f,
        }
    return frame, audit


def prediction_cache_path(args: argparse.Namespace, pair: str, config_id: str, seed: int) -> Path:
    return args.output_dir / "prediction_cache" / f"{pair}__{config_id}__seed{seed}.csv.gz"


def train_config(args_dict: dict[str, Any], pair: str, sequence_path: str,
                 config: dict[str, Any], seed: int) -> tuple[str, str, pd.DataFrame]:
    args = argparse.Namespace(**args_dict)
    path = prediction_cache_path(args, pair, str(config["config_id"]), seed)
    audit_path = path.with_suffix(".audit.csv")
    metadata_path = path.with_suffix(".metadata.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "schema": "deep-long-v12-weekly-prediction-v1", "pair": pair,
        "config": config, "seed": seed, "sequence_sha256": sha256_file(Path(sequence_path)),
        "run_max_epochs": int(args.max_epochs),
        "device": str(args.device),
        "torch_version": torch.__version__,
        "trainer_sha256": sha256_file(Path(__file__)),
        "model_source_sha256": sha256_file(Path(__file__).with_name("deep_learning_long_risk_models_v12.py")),
    }
    if args.resume and path.exists() and audit_path.exists() and metadata_path.exists():
        if prediction_metadata_compatible(json.loads(metadata_path.read_text()), metadata):
            return pair, str(config["config_id"]), pd.read_csv(path)
    with np.load(sequence_path) as source:
        arrays = {key: source[key] for key in source.files}
    selections = pd.read_csv(Path(args.v11_dir) / "grid_selections.csv")
    parts, audits = [], []
    for fold in selections.itertuples(index=False):
        fold_dir = args.output_dir / "prediction_cache" / "folds" / f"{pair}__{config['config_id']}__seed{seed}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        fold_path = fold_dir / f"fold_{int(fold.fold):02d}.csv.gz"
        fold_audit_path = fold_dir / f"fold_{int(fold.fold):02d}.audit.json"
        fold_metadata = {**metadata, "fold": int(fold.fold), "train_end": int(fold.train_end),
                         "test_start": int(fold.test_start), "test_end": int(fold.test_end)}
        if args.resume and fold_path.exists() and fold_audit_path.exists():
            cached_audit = json.loads(fold_audit_path.read_text())
            if cached_audit.get("metadata") == fold_metadata:
                prediction = pd.read_csv(fold_path)
                audit = cached_audit["audit"]
            else:
                prediction, audit = train_fold(
                    arrays, config, int(fold.train_end), int(fold.test_start), int(fold.test_end),
                    seed, int(args.torch_threads), int(args.max_epochs), str(args.device),
                )
        else:
            prediction, audit = train_fold(
                arrays, config, int(fold.train_end), int(fold.test_start), int(fold.test_end),
                seed, int(args.torch_threads), int(args.max_epochs), str(args.device),
            )
        prediction["fold"] = int(fold.fold)
        prediction.to_csv(fold_path, index=False, compression="gzip")
        write_json(fold_audit_path, {"metadata": fold_metadata, "audit": audit})
        parts.append(prediction)
        audits.append({"pair": pair, "config_id": config["config_id"], "seed": seed,
                       "fold": int(fold.fold), **audit})
    result = pd.concat(parts, ignore_index=True)
    result["pair"] = pair
    result["config_id"] = config["config_id"]
    result.to_csv(path, index=False, compression="gzip")
    atomic_csv(pd.DataFrame(audits), audit_path)
    write_json(metadata_path, metadata)
    return pair, str(config["config_id"]), result


def train_all(args: argparse.Namespace, sequences: Mapping[str, Path], seed: int = 42) -> dict[tuple[str, str], pd.DataFrame]:
    configs = deterministic_configurations()
    if str(args.architecture) != "all":
        configs = [item for item in configs if item["architecture"] == str(args.architecture)]
    configs = configs[:int(args.max_configs)]
    pd.DataFrame(configs).to_csv(args.output_dir / "deep_model_configurations.csv", index=False)
    configs = sorted(configs, key=lambda item: (int(str(item["config_id"])[-2:]), str(item["architecture"])))
    jobs = [(vars(args), pair, str(sequences[pair]), config, seed) for config, pair in product(configs, PAIRS)]
    output = {}
    if int(args.workers) == 1:
        iterator = (train_config(*job) for job in jobs)
        pool = None
    else:
        pool = mp.get_context("spawn").Pool(int(args.workers), maxtasksperchild=2)
        iterator = pool.starmap_async(train_config, jobs).get
        iterator = iterator()
    try:
        for pair, config_id, prediction in iterator:
            output[(pair, config_id)] = prediction
            print(f"TRAIN {pair} {config_id} seed={seed}", flush=True)
    finally:
        if int(args.workers) > 1:
            pool.close(); pool.join()
    audits = []
    for path in (args.output_dir / "prediction_cache").glob(f"*seed{seed}.csv.audit.csv"):
        audits.append(pd.read_csv(path))
    if audits:
        pd.concat(audits, ignore_index=True).to_csv(args.output_dir / f"walk_forward_training_audit_seed{seed}.csv", index=False)
    return output


def head_prediction(frame: pd.DataFrame, pair: str, head: str) -> pd.DataFrame:
    output = frame[["signal_ts", f"target_{'72h' if head == 'p72' else '120h'}"]].copy()
    if head == "pmean":
        output = frame[["signal_ts", "target_72h"]].copy()
    output.columns = ["signal_ts", "target"]
    output["pair"] = pair
    output["probability"] = frame[head].to_numpy()
    for quantile in sorted({*ENTRY_QUANTILES, *(max(.5, value - .10) for value in ENTRY_QUANTILES)}):
        output[v5.quantile_column(quantile)] = frame[f"{head}_q{quantile:.3f}"].to_numpy()
    return output


def load_short_contract(args: argparse.Namespace) -> tuple[dict[str, pd.DataFrame], dict[str, v5.GateParameters]]:
    lock = json.loads((args.v11_dir / "locked_configuration.json").read_text(encoding="utf-8"))
    predictions, gates = {}, {}
    for pair in PAIRS:
        row = lock["pair_winners"][pair]
        key = str(row["short_model_key"])
        target, model_pair, config_id = key.split("|")
        predictions[pair] = pd.read_csv(args.v11_dir / "prediction_cache/weekly" / f"{target}__{model_pair}__{config_id}.csv.gz")
        gates[pair] = engine.gate_from_row(row, "short_")
    return predictions, gates


def selected_long_gates(prediction: pd.DataFrame, pair: str, target: str,
                        short_prediction: pd.DataFrame, short_gate: v5.GateParameters) -> list[tuple[v5.GateParameters, dict[str, Any]]]:
    _, short_states, _, _ = engine.build_pair_gate(short_prediction, pair, "short", "short_1h_6h", short_gate)
    rows = []
    for gate in engine.refinement_gates("long"):
        _, states, _, intervals = engine.build_pair_gate(prediction, pair, "long", target, gate)
        anchors = engine.pair_anchor_metrics(intervals, pair)
        overlap = engine.pair_channel_overlap(states, short_states, pair)
        rows.append((gate, {**anchors, "active_jaccard": overlap}))
    rows.sort(key=lambda item: (
        not bool(item[1]["anchor_pass"]), item[1]["active_jaccard"] > .15,
        -(item[1]["feb_03_06_coverage"] + item[1]["jun_01_06_coverage"]),
        item[1]["outside_anchor_share"], item[1]["interval_count"],
    ))
    eligible = [row for row in rows if row[1]["anchor_pass"] and row[1]["active_jaccard"] <= .15]
    return (eligible[:5] if eligible else rows[:1])


def search(args: argparse.Namespace, predictions: Mapping[tuple[str, str], pd.DataFrame],
           candles: Mapping[str, pd.DataFrame], selections: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    short_predictions, short_gates = load_short_contract(args)
    baseline = json.loads((args.v11_dir / "mechanism1_baseline.json").read_text())
    pair_rows = []
    long_views: dict[str, pd.DataFrame] = {}
    for (pair, config_id), frame in predictions.items():
        config = next(item for item in deterministic_configurations() if item["config_id"] == config_id)
        for head in ("p72", "p120", "pmean"):
            target = "long_72h" if head in ("p72", "pmean") else "long_120h"
            prediction = head_prediction(frame, pair, head)
            model_key = f"{pair}|{config_id}|{head}"
            long_views[model_key] = prediction
            for gate, diagnostics in selected_long_gates(
                prediction, pair, target, short_predictions[pair], short_gates[pair]
            ):
                specs = [
                    (prediction, pair, "long", target, gate),
                    *[(short_predictions[p], p, "short", "short_1h_6h", short_gates[p]) for p in PAIRS],
                ]
                metrics = engine.replay_metrics(candles, selections, specs)
                pair_rows.append({
                    "candidate_id": f"{model_key}|{engine.gate_id('long', gate)}",
                    "pair": pair, "model_key": model_key, "config_id": config_id,
                    "architecture": config["architecture"], "head": head, "target": target,
                    "parameter_count": int(pd.read_csv(args.output_dir / "walk_forward_training_audit_seed42.csv").query(
                        "pair == @pair and config_id == @config_id"
                    ).parameter_count.iloc[0]),
                    **asdict(gate), **metrics, **diagnostics,
                    "eligible": bool(diagnostics["anchor_pass"] and diagnostics["active_jaccard"] <= .15),
                })
    pair_frame = engine.score_frame(pd.DataFrame(pair_rows), ("pair",))
    atomic_csv(pair_frame, args.output_dir / "pair_long_candidate_search.csv")
    portfolio_rows = []
    btc_top = pair_frame[pair_frame.pair.eq("BTC-FDUSD")].nsmallest(10, "rank")
    eth_top = pair_frame[pair_frame.pair.eq("ETH-FDUSD")].nsmallest(10, "rank")
    for btc, eth in product(btc_top.to_dict("records"), eth_top.to_dict("records")):
        specs = []
        for pair, row in (("BTC-FDUSD", btc), ("ETH-FDUSD", eth)):
            specs.append((long_views[row["model_key"]], pair, "long", row["target"], engine.gate_from_row(row)))
            specs.append((short_predictions[pair], pair, "short", "short_1h_6h", short_gates[pair]))
        metrics = engine.replay_metrics(candles, selections, specs)
        portfolio_rows.append({
            "candidate_id": f"portfolio|{btc['candidate_id']}|||{eth['candidate_id']}",
            "BTC_candidate_id": btc["candidate_id"], "ETH_candidate_id": eth["candidate_id"],
            "BTC_anchor_pass": bool(btc["anchor_pass"]), "ETH_anchor_pass": bool(eth["anchor_pass"]),
            "BTC_active_jaccard": btc["active_jaccard"], "ETH_active_jaccard": eth["active_jaccard"],
            "parameter_count": int(btc["parameter_count"] + eth["parameter_count"]), **metrics,
            "eligible": bool(
                btc["anchor_pass"] and eth["anchor_pass"] and btc["active_jaccard"] <= .15
                and eth["active_jaccard"] <= .15 and metrics["oos_pnl_fdusd"] > baseline["oos_pnl_fdusd"]
                and metrics["stitched_max_drawdown_pct"] >= baseline["stitched_max_drawdown_pct"]
            ),
        })
    portfolio = pd.DataFrame(portfolio_rows)
    portfolio["profit_percentile"] = portfolio.oos_pnl_fdusd.rank(method="average", pct=True)
    portfolio["drawdown_percentile"] = portfolio.stitched_max_drawdown_pct.rank(method="average", pct=True)
    portfolio["objective_score"] = .5 * portfolio.profit_percentile + .5 * portfolio.drawdown_percentile
    portfolio = portfolio.sort_values(
        ["eligible", "objective_score", "portfolio_stop_events", "pair_stop_events",
         "risk_off_pair_hours", "parameter_count", "oos_pnl_fdusd"],
        ascending=[False, False, True, True, True, True, False],
    ).reset_index(drop=True)
    portfolio["rank"] = np.arange(1, len(portfolio) + 1)
    atomic_csv(portfolio, args.output_dir / "portfolio_search.csv")
    return pair_frame, portfolio


def candidate_row(pair_frame: pd.DataFrame, candidate_id: str) -> dict[str, Any]:
    match = pair_frame[pair_frame.candidate_id.eq(candidate_id)]
    if len(match) != 1:
        raise ValueError(f"locked candidate missing: {candidate_id}")
    return match.iloc[0].to_dict()


def final_specifications(args: argparse.Namespace, winner: Mapping[str, Any], pair_frame: pd.DataFrame,
                         predictions: Mapping[tuple[str, str], pd.DataFrame]):
    short_predictions, short_gates = load_short_contract(args)
    specs, rows, views = [], {}, {}
    for pair in PAIRS:
        row = candidate_row(pair_frame, str(winner[f"{pair[:3]}_candidate_id"]))
        rows[pair] = row
        view = head_prediction(predictions[(pair, str(row["config_id"]))], pair, str(row["head"]))
        views[pair] = view
        specs.extend([
            (view, pair, "long", str(row["target"]), engine.gate_from_row(row)),
            (short_predictions[pair], pair, "short", "short_1h_6h", short_gates[pair]),
        ])
    return specs, rows, views


def save_detailed(args: argparse.Namespace, detailed: Mapping[str, Any]) -> None:
    detailed["weekly"].to_csv(args.output_dir / "final_weekly_results.csv", index=False)
    detailed["equity"].to_csv(args.output_dir / "final_equity_curve.csv.gz", index=False, compression="gzip")
    detailed["states"].to_csv(args.output_dir / "final_risk_states.csv.gz", index=False, compression="gzip")
    detailed["events"].to_csv(args.output_dir / "final_risk_events.csv", index=False)
    detailed["intervals"].to_csv(args.output_dir / "final_risk_intervals.csv", index=False)
    detailed["trades"].to_csv(args.output_dir / "final_trade_events.csv.gz", index=False, compression="gzip")
    detailed["stops"].to_csv(args.output_dir / "final_stop_events.csv", index=False)


def final_train_model(args: argparse.Namespace, pair: str, row: Mapping[str, Any], sequence_path: Path,
                      seed: int) -> dict[str, Any]:
    config = next(item for item in deterministic_configurations() if item["config_id"] == row["config_id"])
    model_dir = args.output_dir / "models"
    model_dir.mkdir(exist_ok=True)
    model_path = model_dir / f"{pair}__{config['config_id']}__seed{seed}.pt"
    metadata_path = model_path.with_suffix(".metadata.json")
    expected_metadata = {
        "schema": "deep-long-v12-final-model-cache-v1",
        "pair": pair,
        "seed": int(seed),
        "config": config,
        "head": str(row["head"]),
        "sequence_sha256": sha256_file(sequence_path),
        "trainer_sha256": sha256_file(Path(__file__)),
        "model_source_sha256": sha256_file(Path(__file__).with_name("deep_learning_long_risk_models_v12.py")),
        "torch_version": torch.__version__,
        "device": str(args.device),
        "max_epochs": int(args.max_epochs),
    }
    if args.resume and model_path.exists() and metadata_path.exists():
        cached = json.loads(metadata_path.read_text())
        if cached.get("metadata") == expected_metadata and cached.get("result", {}).get("model_sha256") == sha256_file(model_path):
            return cached["result"]
    with np.load(sequence_path) as source:
        arrays = {key: source[key] for key in source.files}
    frame, audit, artifact = train_fold(
        arrays, config, END_TS, END_TS - 7 * 24 * HOUR, END_TS, seed,
        int(args.torch_threads), int(args.max_epochs), str(args.device), return_artifacts=True,
    )
    payload = {
        "schema": "deep-long-v12-pytorch-model-v1", "model_version": MODEL_VERSION,
        "pair": pair, "seed": seed, "config": config, "head": row["head"],
        "hourly_features": HOURLY_FEATURES, "five_features": FIVE_FEATURES,
        "scaler": artifact["scaler"].as_dict(),
        "state_dict": artifact["model"].state_dict(), "fit_audit": audit,
    }
    torch.save(payload, model_path)
    loaded = torch.load(model_path, map_location="cpu", weights_only=False)
    restored = DualBranchLongRiskModel(len(HOURLY_FEATURES), len(FIVE_FEATURES), loaded["config"])
    restored.load_state_dict(loaded["state_dict"])
    restored = restored.to(next(artifact["model"].parameters()).device)
    before = infer_logits(artifact["model"], artifact["test_hourly"][-1:], artifact["test_five"][-1:], 1)
    after = infer_logits(restored, artifact["test_hourly"][-1:], artifact["test_five"][-1:], 1)
    maximum_error = float(np.max(np.abs(before - after)))
    if maximum_error > 1e-7:
        raise AssertionError("serialized PyTorch probability logits changed")
    result = {
        "pair": pair, "seed": seed, "config": config, "head": row["head"], "audit": audit,
        "last_probability": float(frame[str(row["head"])].iloc[-1]),
        "model_path": str(model_path), "model_sha256": sha256_file(model_path),
        "serialization_maximum_logit_error": maximum_error,
    }
    write_json(metadata_path, {"metadata": expected_metadata, "result": result})
    return result


def write_attribution(args: argparse.Namespace, sequences: Mapping[str, Path], rows: Mapping[str, Mapping[str, Any]]) -> None:
    feature_rows, time_rows = [], []
    for pair in PAIRS:
        model_path = args.output_dir / "models" / f"{pair}__{rows[pair]['config_id']}__seed42.pt"
        payload = torch.load(model_path, map_location="cpu", weights_only=False)
        model = DualBranchLongRiskModel(len(HOURLY_FEATURES), len(FIVE_FEATURES), payload["config"])
        model.load_state_dict(payload["state_dict"]); model.eval()
        scaler_values = payload["scaler"]
        scaler = RobustSequenceScaler(*[
            np.asarray(scaler_values[key], dtype=np.float32)
            for key in ("median_hourly", "iqr_hourly", "median_five", "iqr_five")
        ])
        with np.load(sequences[pair]) as source:
            indices = np.flatnonzero((source["signal_ts"] >= START_TS) & (source["signal_ts"] < END_TS))
            indices = indices[np.linspace(0, len(indices) - 1, min(512, len(indices))).astype(int)]
            hourly, five = scaler.transform(source["hourly"][indices], source["five"][indices])
        base_logits = infer_logits(model, hourly, five, 128)
        head_index = 0 if rows[pair]["head"] in ("p72", "pmean") else 1
        base_probability = sigmoid_temperature(base_logits[:, head_index], 1.0)
        for index, feature in enumerate(HOURLY_FEATURES):
            altered = hourly.copy(); altered[:, :, index] = np.roll(altered[:, :, index], 24, axis=0)
            probability = sigmoid_temperature(infer_logits(model, altered, five, 128)[:, head_index], 1.0)
            feature_rows.append({"pair": pair, "branch": "hourly", "feature": feature,
                                 "mean_absolute_probability_change": float(np.mean(np.abs(probability - base_probability)))})
        for index, feature in enumerate(FIVE_FEATURES):
            altered = five.copy(); altered[:, :, index] = np.roll(altered[:, :, index], 24, axis=0)
            probability = sigmoid_temperature(infer_logits(model, hourly, altered, 128)[:, head_index], 1.0)
            feature_rows.append({"pair": pair, "branch": "five_minute", "feature": feature,
                                 "mean_absolute_probability_change": float(np.mean(np.abs(probability - base_probability)))})
        for branch, values, blocks in (
            ("hourly", hourly, ((0, 24), (24, 72), (72, 120), (120, 168))),
            ("five_minute", five, ((0, 72), (72, 144), (144, 216), (216, 288))),
        ):
            for left, right in blocks:
                changed_h, changed_f = hourly.copy(), five.copy()
                (changed_h if branch == "hourly" else changed_f)[:, left:right, :] = 0
                probability = sigmoid_temperature(infer_logits(model, changed_h, changed_f, 128)[:, head_index], 1.0)
                time_rows.append({"pair": pair, "branch": branch, "start_step": left, "end_step": right,
                                  "mean_absolute_probability_change": float(np.mean(np.abs(probability - base_probability)))})
    pd.DataFrame(feature_rows).to_csv(args.output_dir / "permutation_feature_attribution.csv", index=False)
    pd.DataFrame(time_rows).to_csv(args.output_dir / "time_block_attribution.csv", index=False)


def finalize(args: argparse.Namespace, sequences: Mapping[str, Path], predictions: Mapping[tuple[str, str], pd.DataFrame],
             candles: Mapping[str, pd.DataFrame], selections: pd.DataFrame,
             pair_frame: pd.DataFrame, portfolio: pd.DataFrame) -> dict[str, Any]:
    winner = portfolio.iloc[0].to_dict()
    specs, rows, _ = final_specifications(args, winner, pair_frame, predictions)
    detailed = engine.detailed_replay(candles, selections, specs, "Deep v12 + fixed XGBoost v11 short")
    save_detailed(args, detailed)
    metrics = detailed["summary"]
    pressure_rows = []
    for scenario, scenario_candles, fee, slippage in (
        ("base", candles, base.TAKER_FEE, 0.0),
        ("taker_150pct", candles, base.TAKER_FEE * 1.5, 0.0),
        ("slippage_0_05pct", candles, base.TAKER_FEE, .0005),
        ("slippage_0_10pct", candles, base.TAKER_FEE, .001),
        ("single_day_15pct_drop_fixed_locked_signal", crash_candles(dict(candles), .15), base.TAKER_FEE, 0.0),
    ):
        value = engine.replay_metrics(scenario_candles, selections, specs, taker_fee=fee, slippage=slippage)
        pressure_rows.append({"scenario": scenario, **value,
                              "no_stops": value["pair_stop_events"] == 0 and value["portfolio_stop_events"] == 0})
    pressure = pd.DataFrame(pressure_rows)
    pressure.to_csv(args.output_dir / "pressure_tests.csv", index=False)
    seed_rows = []
    for seed in SEEDS:
        seed_predictions: dict[tuple[str, str], pd.DataFrame] = {}
        for pair in PAIRS:
            config_id = str(rows[pair]["config_id"])
            if seed == 42:
                seed_predictions[(pair, config_id)] = predictions[(pair, config_id)]
            else:
                config = next(item for item in deterministic_configurations() if item["config_id"] == config_id)
                _, _, value = train_config(vars(args), pair, str(sequences[pair]), config, seed)
                seed_predictions[(pair, config_id)] = value
        seed_specs, _, _ = final_specifications(args, winner, pair_frame, seed_predictions)
        seed_metrics = engine.replay_metrics(candles, selections, seed_specs)
        seed_rows.append({"seed": seed, **seed_metrics})
    seed_frame = pd.DataFrame(seed_rows)
    seed_frame.to_csv(args.output_dir / "seed_stability_grid_metrics.csv", index=False)
    stability = [final_train_model(args, pair, rows[pair], sequences[pair], seed)
                 for seed, pair in product(SEEDS, PAIRS)]
    write_json(args.output_dir / "seed_stability_training.json", {"runs": stability})
    write_attribution(args, sequences, rows)
    median_seed_profit = float(seed_frame.oos_pnl_fdusd.median())
    worst_seed_drawdown = float(seed_frame.stitched_max_drawdown_pct.min())
    acceptance = {
        "median_seed_profit_above_v11": median_seed_profit > OLD_PNL,
        "worst_seed_drawdown_not_worse_than_v11": worst_seed_drawdown >= OLD_DRAWDOWN,
        "all_seed_btc_non_negative": bool(seed_frame.btc_pnl_fdusd.ge(0).all()),
        "all_seed_eth_non_negative": bool(seed_frame.eth_pnl_fdusd.ge(0).all()),
        "all_seed_zero_portfolio_stops": bool(seed_frame.portfolio_stop_events.eq(0).all()),
        "all_seed_fewer_than_7_pair_stops": bool(seed_frame.pair_stop_events.lt(OLD_PAIR_STOPS).all()),
        "BTC_anchor_pass": bool(rows["BTC-FDUSD"]["anchor_pass"]),
        "ETH_anchor_pass": bool(rows["ETH-FDUSD"]["anchor_pass"]),
        "all_pressure_scenarios_no_stops": bool(pressure.no_stops.all()),
    }
    passed = bool(all(acceptance.values()))
    lock = {
        "schema": "deep-learning-long-risk-v12-lock-v1", "model_version": MODEL_VERSION,
        "portfolio_winner": winner, "pair_winners": rows, "acceptance": acceptance,
        "research_gate_passed": passed, "deployment_allowed": False,
        "evidence_status": "full_180d_in_sample_anchor_targeted_optimization",
        "feature_schema_sha256": hashlib.sha256(json.dumps({"hourly": HOURLY_FEATURES, "five": FIVE_FEATURES}).encode()).hexdigest(),
        "sequence_sha256": {pair: sha256_file(path) for pair, path in sequences.items()},
        "v11_short_lock_sha256": sha256_file(args.v11_dir / "locked_configuration.json"),
    }
    write_json(args.output_dir / "locked_configuration.json", lock)
    summary = {
        "schema": "deep-learning-long-risk-v12-summary-v1", "model_version": MODEL_VERSION,
        "winner_metrics": metrics, "pair_winners": rows, "acceptance": acceptance,
        "median_seed_profit_fdusd": median_seed_profit,
        "worst_seed_stitched_drawdown_pct": worst_seed_drawdown,
        "research_gate_passed": passed, "deployment_allowed": False,
        "verdict": "NEXT_STAGE_JOINT_VALIDATION" if passed else "NO-GO",
        "evidence_status": lock["evidence_status"],
    }
    write_json(args.output_dir / "summary.json", summary)
    old = pd.read_csv(args.v11_dir / "previous_version_comparison.csv")
    old = pd.concat([old, pd.DataFrame([{"version": "Deep v12 hybrid", **metrics}])], ignore_index=True)
    old.to_csv(args.output_dir / "version_comparison.csv", index=False)
    return summary


def main() -> int:
    args = parse_args()
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but CUDA is unavailable")
    if args.device == "cuda" and int(args.workers) > 1:
        print("CUDA training uses one process to avoid competing for a single GPU", flush=True)
        args.workers = 1
    environment_files = [
        Path("environment-grid-deep-risk-base-cu126.yml"),
        Path("requirements-grid-deep-risk-base-cu126.txt"),
    ]
    write_json(args.output_dir / "environment_lock.json", {
        "schema": "deep-risk-python-environment-lock-v1",
        "python": platform.python_version(), "python_executable": str(Path(sys.executable).resolve()),
        "pytorch": torch.__version__, "pytorch_cuda_available": torch.cuda.is_available(),
        "training_device": str(args.device), "architecture_filter": str(args.architecture),
        "numpy": np.__version__, "pandas": pd.__version__,
        "lock_files": {str(path): sha256_file(path) for path in environment_files},
    })
    sequences = build_sequence_cache(args)
    if args.stage == "prepare":
        return 0
    candles = load_candles(args.cache_dir)
    selections = pd.read_csv(args.v11_dir / "grid_selections.csv")
    predictions = train_all(args, sequences, 42)
    if args.stage == "train":
        return 0
    pair_path, portfolio_path = args.output_dir / "pair_long_candidate_search.csv", args.output_dir / "portfolio_search.csv"
    if args.stage in {"search", "all"} or not (args.resume and pair_path.exists() and portfolio_path.exists()):
        pair_frame, portfolio = search(args, predictions, candles, selections)
    else:
        pair_frame, portfolio = pd.read_csv(pair_path), pd.read_csv(portfolio_path)
    if args.stage == "search":
        return 0
    if args.stage in {"finalize", "all"}:
        summary = finalize(args, sequences, predictions, candles, selections, pair_frame, portfolio)
        print(json.dumps({"verdict": summary["verdict"], "metrics": summary["winner_metrics"]}, ensure_ascii=False, indent=2))
    if args.stage in {"plot", "all"}:
        from build_deep_learning_long_risk_gate_v12_artifacts import build_all
        build_all(args.output_dir, args.v11_dir)
    return 0


if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(main())

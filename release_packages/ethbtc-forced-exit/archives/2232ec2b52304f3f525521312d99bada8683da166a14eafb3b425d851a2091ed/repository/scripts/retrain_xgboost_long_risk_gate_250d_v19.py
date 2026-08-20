#!/usr/bin/env python3
"""Leakage-safe, long-only XGBoost Grid BUY-gate research over 250 days.

This supersedes the diagnostic v15-v18 studies.  The model only arms a
pair-specific long-risk gate.  Directional market structure confirms entry
and recovery; probability alone can never recover BUY.  No signal generated
by this module is allowed to sell inventory or authorize deployment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from plotly import graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import roc_auc_score
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

import optimize_xgboost_roc_sqz_pair_risk_gate_v8 as legacy
import search_fdusd_inventory_exit as inventory
import tune_xgboost_grid_risk_gate_v1 as grid
import tune_xgboost_momentum_stop_v2 as tune
from validate_grid_live import simulate, slice_window
from validate_grid_live import crash_candles


MODEL_VERSION = "xgboost-grid-long-risk-gate-v19-250d"
OUTPUT_DIR = Path("results/backtests/xgboost_grid_long_risk_gate_v19_250d")
SOURCE_DIR = Path("results/backtests/eth_xgboost_long_risk_gate_v15_250d")
PAIRS = ("BTC-FDUSD", "ETH-FDUSD")
TARGETS = ("long_event_72h", "long_event_120h")
HOUR = 3600
DAY = 24 * HOUR
START_TS = int(pd.Timestamp("2025-11-23T15:00:00Z").timestamp())
END_TS = int(pd.Timestamp("2026-07-31T15:00:00Z").timestamp())
ANCHOR_WINDOWS = (
    ("feb_03_06", int(pd.Timestamp("2026-02-03T00:00:00Z").timestamp()),
     int(pd.Timestamp("2026-02-07T00:00:00Z").timestamp())),
    ("jun_01_06", int(pd.Timestamp("2026-06-01T00:00:00Z").timestamp()),
     int(pd.Timestamp("2026-06-07T00:00:00Z").timestamp())),
)

FEATURE_SETS: dict[str, tuple[str, ...]] = {
    "directional_persistence": (
        "adx_14", "di_spread", "atr_pct", "btc_volatility_20",
        "roc_48h_4h", "sqzmom_pct_4h", "sqzmom_slope_4h",
        "drawdown_from_high_72h", "drawdown_from_high_168h",
        "drawdown_duration_168h", "below_ema20_ratio_72h",
        "lower_low_ratio_72h", "downside_semivariance_ratio_72h",
        "trend_efficiency_72h", "ema20_slope_atr_12h",
    ),
    "full_structure": (
        "adx_14", "di_spread", "atr_pct", "btc_volatility_20",
        "roc_48h_4h", "sqzmom_pct_4h", "sqzmom_slope_4h",
        "drawdown_from_high_72h", "drawdown_from_high_168h",
        "drawdown_duration_168h", "below_ema20_ratio_72h",
        "lower_low_ratio_72h", "downside_semivariance_ratio_24h",
        "downside_semivariance_ratio_72h", "rv_24h_percentile_30d",
        "vol_of_vol_72h", "trend_efficiency_72h", "ema20_slope_atr_12h",
        "historical_var_72h", "expected_shortfall_72h", "negative_skew_72h",
        "cross_pair_downside_beta_72h", "relative_drawdown_72h",
    ),
}


@dataclass(frozen=True)
class ResearchPeriod:
    start_ts: int
    end_ts: int

    def __post_init__(self) -> None:
        if self.end_ts <= self.start_ts:
            raise ValueError("end_ts must be after start_ts")


@dataclass(frozen=True)
class LongGate:
    entry_quantile: float
    entry_bars: int
    minimum_hours: int
    cooldown_hours: int
    arm_hours: int = 24
    recovery_4h_bars: int = 2


PERIOD = ResearchPeriod(START_TS, END_TS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prepare", "screen", "train", "search", "finalize", "plot", "all"), default="all")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--xgb-threads", type=int, default=2)
    parser.add_argument("--finalists", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--source-dir", type=Path, default=SOURCE_DIR)
    return parser.parse_args()


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


def rolling_duration_from_high(close: pd.Series, window: int) -> pd.Series:
    values = close.to_numpy(float)
    result = np.full(len(values), np.nan)
    for index in range(window - 1, len(values)):
        sample = values[index - window + 1:index + 1]
        result[index] = len(sample) - 1 - int(np.nanargmax(sample))
    return pd.Series(result, index=close.index)


def rolling_percentile(series: pd.Series, window: int = 720, minimum: int = 240) -> pd.Series:
    return series.rolling(window, min_periods=minimum).apply(
        lambda x: float((x[:-1] <= x[-1]).mean()) if len(x) > 1 else np.nan, raw=True,
    )


def expected_shortfall(values: np.ndarray) -> float:
    size = max(1, int(np.ceil(0.05 * len(values))))
    return float(np.mean(np.sort(values)[:size]))


def add_structure_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Build scale-free long-regime features from already completed 1h bars."""
    parts: dict[str, pd.DataFrame] = {}
    returns: dict[str, pd.Series] = {}
    for pair in PAIRS:
        item = panel[panel.pair.eq(pair)].sort_values("signal_ts").copy()
        close, low = item.close.astype(float), item.low.astype(float)
        r1 = np.log(close).diff(); returns[pair] = pd.Series(r1.to_numpy(), index=item.signal_ts)
        negative, total = r1.clip(upper=0).pow(2), r1.pow(2)
        ema20 = close.ewm(span=20, adjust=False).mean()
        rv24 = total.rolling(24, min_periods=12).sum().pow(.5)
        item["drawdown_from_high_72h"] = close / close.rolling(72, min_periods=36).max() - 1
        item["drawdown_from_high_168h"] = close / close.rolling(168, min_periods=84).max() - 1
        item["drawdown_duration_168h"] = rolling_duration_from_high(close, 168)
        item["below_ema20_ratio_72h"] = close.lt(ema20).astype(float).rolling(72, min_periods=36).mean()
        item["lower_low_ratio_72h"] = low.lt(low.shift()).astype(float).rolling(72, min_periods=36).mean()
        for hours in (24, 72):
            item[f"downside_semivariance_ratio_{hours}h"] = (
                negative.rolling(hours, min_periods=hours // 2).sum()
                / total.rolling(hours, min_periods=hours // 2).sum().replace(0, np.nan)
            )
        item["rv_24h_percentile_30d"] = rolling_percentile(rv24)
        item["vol_of_vol_72h"] = rv24.rolling(72, min_periods=36).std(ddof=0) / rv24.rolling(72, min_periods=36).mean().replace(0, np.nan)
        item["trend_efficiency_72h"] = (close - close.shift(72)).abs() / close.diff().abs().rolling(72, min_periods=36).sum().replace(0, np.nan)
        atr_price = item.atr_pct.astype(float) * close
        item["ema20_slope_atr_12h"] = (ema20 - ema20.shift(12)) / atr_price.replace(0, np.nan)
        item["historical_var_72h"] = r1.rolling(72, min_periods=36).quantile(.05)
        item["expected_shortfall_72h"] = r1.rolling(72, min_periods=36).apply(expected_shortfall, raw=True)
        item["negative_skew_72h"] = -r1.rolling(72, min_periods=36).skew()
        parts[pair] = item
    btc = returns["BTC-FDUSD"]; eth = returns["ETH-FDUSD"]
    for pair, own, other in (("BTC-FDUSD", btc, eth), ("ETH-FDUSD", eth, btc)):
        item = parts[pair].set_index("signal_ts")
        down = other.where(other < 0)
        cov = own.where(other < 0).rolling(72, min_periods=24).cov(down)
        item["cross_pair_downside_beta_72h"] = cov / down.rolling(72, min_periods=24).var().replace(0, np.nan)
        other_dd = other.add(1).rolling(72, min_periods=36).apply(lambda x: x.prod(), raw=True) - 1
        own_dd = own.add(1).rolling(72, min_periods=36).apply(lambda x: x.prod(), raw=True) - 1
        item["relative_drawdown_72h"] = own_dd - other_dd
        parts[pair] = item.reset_index()
    output = pd.concat(parts.values(), ignore_index=True).sort_values(["signal_ts", "pair"])
    return output.reset_index(drop=True)


def event_onset_target(regime: pd.Series, horizon: int, lead_hours: int = 24) -> tuple[pd.Series, pd.Series]:
    """Predict a new persistent-risk onset within the next lead window.

    Tail rows without a fully observed regime horizon remain NaN.  Positive
    rows receive inverse-overlap uniqueness weights by onset event.
    """
    values = regime.astype("float").to_numpy()
    valid = np.isfinite(values)
    # The original forward-risk label can flicker around its threshold.  An
    # event begins only when at least 9 of the next 12 labels describe the
    # persistent regime and no more than 6 of the prior 24 do.  This turns a
    # long decline into one onset rather than dozens of correlated positives.
    future_persistence = pd.Series(values).rolling(12, min_periods=12).mean().shift(-11).to_numpy()
    prior_risk = pd.Series(values).rolling(24, min_periods=24).mean().shift(1).to_numpy()
    onset = valid & (future_persistence >= .75) & (prior_risk <= .25)
    # Enforce a 48-hour refractory period between adjacent onset candidates.
    accepted = np.zeros(len(values), dtype=bool); last = -10**9
    for position in np.flatnonzero(onset):
        if position - last >= 48:
            accepted[position] = True; last = int(position)
    onset = accepted
    target = np.full(len(values), np.nan)
    uniqueness = np.ones(len(values), dtype=float)
    last_usable = len(values) - lead_hours - horizon
    for index in range(max(0, last_usable + 1)):
        future = np.flatnonzero(onset[index:index + lead_hours + 1])
        target[index] = float(len(future) > 0)
        if len(future):
            event_index = index + int(future[0])
            eligible = np.arange(max(0, event_index - lead_hours), event_index + 1)
            uniqueness[index] = 1.0 / max(1, len(eligible))
    return pd.Series(target, index=regime.index), pd.Series(uniqueness, index=regime.index)


def prepare_panel(source: pd.DataFrame) -> pd.DataFrame:
    output = add_structure_features(source)
    parts = []
    for pair in PAIRS:
        item = output[output.pair.eq(pair)].sort_values("signal_ts").copy()
        for hours in (72, 120):
            target, unique = event_onset_target(item[f"target_long_{hours}h"], hours)
            item[f"target_long_event_{hours}h"] = target
            item[f"event_uniqueness_{hours}h"] = unique
            item[f"label_ready_ts_long_event_{hours}h"] = item.signal_ts + (hours + 24) * HOUR
        parts.append(item)
    result = pd.concat(parts, ignore_index=True).sort_values(["signal_ts", "pair"])
    all_features = sorted(set().union(*FEATURE_SETS.values()))
    result[all_features] = result[all_features].replace([np.inf, -np.inf], np.nan)
    return result.reset_index(drop=True)


def target_frame(panel: pd.DataFrame, target: str, pair: str) -> pd.DataFrame:
    hours = 72 if target.endswith("72h") else 120
    item = panel[panel.pair.eq(pair)].copy()
    item["target"] = item[f"target_long_event_{hours}h"]
    item["event_uniqueness"] = item[f"event_uniqueness_{hours}h"]
    item["label_ready_ts"] = item[f"label_ready_ts_long_event_{hours}h"]
    return item


def split_training(frame: pd.DataFrame, cutoff: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    mature = frame[frame.target.notna() & (frame.label_ready_ts <= cutoff)].copy()
    calibration = mature[mature.signal_ts >= cutoff - 14 * DAY].copy()
    development = mature[mature.signal_ts < cutoff - 14 * DAY].copy()
    early = development[development.signal_ts >= cutoff - 28 * DAY].copy()
    early_train = development[development.signal_ts < cutoff - 28 * DAY].copy()
    if min(map(len, (calibration, development, early, early_train))) == 0:
        raise RuntimeError(f"insufficient disjoint train/early/calibration rows at {cutoff}")
    if set(development.index) & set(calibration.index):
        raise AssertionError("calibration rows leaked into final fit")
    if int(mature.label_ready_ts.max()) > cutoff:
        raise AssertionError("immature label entered training")
    return mature, development, early_train, early


def sample_weights(frame: pd.DataFrame) -> np.ndarray:
    weights = np.asarray(compute_sample_weight("balanced", frame.target.astype(int)), float)
    weights *= np.where(frame.target.to_numpy(int) == 1, frame.event_uniqueness.to_numpy(float), 1.0)
    return weights / np.mean(weights)


def fit_leakage_safe(frame: pd.DataFrame, cutoff: int, config: Mapping[str, Any], features: Sequence[str]) -> tuple[XGBClassifier, pd.DataFrame, dict[str, Any]]:
    mature, development, early_train, early = split_training(frame, cutoff)
    calibration = mature.loc[~mature.index.isin(development.index)].copy()
    cap = int(config["n_estimators"]); trees = cap; score = np.nan
    if bool(config["uses_early_stopping"]):
        model = XGBClassifier(**tune.config_params(config), early_stopping_rounds=50)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(early_train[list(features)], early_train.target.astype(int),
                      sample_weight=sample_weights(early_train),
                      eval_set=[(early[list(features)], early.target.astype(int))], verbose=False)
        trees = min(cap, max(1, int(getattr(model, "best_iteration", cap - 1)) + 1))
        score = float(getattr(model, "best_score", np.nan))
    final = XGBClassifier(**tune.config_params(config, n_estimators=trees))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        final.fit(development[list(features)], development.target.astype(int),
                  sample_weight=sample_weights(development), verbose=False)
    calibration = calibration[["pair", "signal_ts", "target"]].copy()
    calibration["probability"] = final.predict_proba(mature.loc[calibration.index, list(features)])[:, 1]
    audit = {
        "cutoff": cutoff, "mature_rows": len(mature), "development_rows": len(development),
        "early_train_rows": len(early_train), "early_stop_rows": len(early),
        "calibration_rows": len(calibration), "best_tree_count": trees,
        "best_validation_logloss": score,
        "development_last_ts": int(development.signal_ts.max()),
        "calibration_first_ts": int(calibration.signal_ts.min()),
        "last_label_ready_ts": int(mature.label_ready_ts.max()),
    }
    return final, calibration, audit


def attach_thresholds(prediction: pd.DataFrame, calibration: pd.DataFrame) -> pd.DataFrame:
    result = prediction.copy()
    for quantile in legacy.ENTRY_QUANTILES:
        result[legacy.v5.quantile_column(quantile)] = float(calibration.probability.quantile(quantile))
    return result


def configurations() -> list[dict[str, Any]]:
    return tune.xgb_configurations()


def specs() -> list[dict[str, Any]]:
    return [
        {"model_key": f"{pair}|{target}|{feature_id}|{config['config_id']}", "pair": pair,
         "target": target, "feature_id": feature_id, "features": list(features), "config": config}
        for pair in PAIRS for target in TARGETS for feature_id, features in FEATURE_SETS.items()
        for config in configurations()
    ]


_PANEL: pd.DataFrame | None = None
_SELECTIONS: pd.DataFrame | None = None
_ARGS: argparse.Namespace | None = None
_PERIOD: ResearchPeriod | None = None


def init_worker(panel: pd.DataFrame, selections: pd.DataFrame, args: argparse.Namespace, period: ResearchPeriod) -> None:
    global _PANEL, _SELECTIONS, _ARGS, _PERIOD
    _PANEL, _SELECTIONS, _ARGS, _PERIOD = panel, selections, args, period
    tune.XGB_N_JOBS = int(args.xgb_threads)
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = "1"


def cache_path(args: argparse.Namespace, stage: str, spec: Mapping[str, Any]) -> Path:
    return args.output_dir / "prediction_cache" / stage / f"{str(spec['model_key']).replace('|', '__')}.csv.gz"


def predict_spec(stage: str, spec: Mapping[str, Any]) -> tuple[str, str]:
    if _PANEL is None or _SELECTIONS is None or _ARGS is None or _PERIOD is None:
        raise RuntimeError("worker context missing")
    path = cache_path(_ARGS, stage, spec)
    input_hash = hashlib.sha256(json.dumps({"spec": spec, "period": asdict(_PERIOD),
        "panel": sha256_file(_ARGS.output_dir / "feature_panel.csv.gz")}, sort_keys=True).encode()).hexdigest()
    meta_path = path.with_name(path.name + ".metadata.json")
    if _ARGS.resume and path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("input_hash") == input_hash and meta.get("prediction_sha256") == sha256_file(path):
            return str(spec["model_key"]), "reused"
    frame = target_frame(_PANEL, str(spec["target"]), str(spec["pair"]))
    blocks = ([SimpleNamespace(train_end=_PERIOD.start_ts, test_start=_PERIOD.start_ts, test_end=_PERIOD.end_ts, fold=0)]
              if stage == "screen" else list(_SELECTIONS.itertuples(index=False)))
    outputs, audits = [], []
    for block in blocks:
        model, calibration, audit = fit_leakage_safe(frame, int(block.train_end), spec["config"], spec["features"])
        test = frame[(frame.signal_ts >= int(block.test_start)) & (frame.signal_ts < int(block.test_end))].copy()
        pred = test[["pair", "signal_ts", "target", "last_complete_4h_ts", "roc_48h_4h",
                     "sqzmom_pct_4h", "di_spread", "ema20_slope_atr_12h", "below_ema20_ratio_72h"]].copy()
        pred["probability"] = model.predict_proba(test[list(spec["features"])])[:, 1]
        pred = attach_thresholds(pred, calibration); pred["fold"] = int(block.fold)
        outputs.append(pred); audits.append({"fold": int(block.fold), **audit})
    result = pd.concat(outputs, ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    result.to_csv(temporary, index=False, compression="gzip"); os.replace(temporary, path)
    pd.DataFrame(audits).to_csv(path.with_suffix(".audit.csv"), index=False)
    atomic_json(meta_path, {"input_hash": input_hash, "prediction_sha256": sha256_file(path),
                            "period": asdict(_PERIOD), "rows": len(result)})
    return str(spec["model_key"]), "trained"


def run_jobs(args: argparse.Namespace, panel: pd.DataFrame, selections: pd.DataFrame,
             stage: str, selected: Sequence[Mapping[str, Any]], period: ResearchPeriod = PERIOD) -> None:
    jobs = [(stage, item) for item in selected]
    workers = min(max(1, int(args.workers)), len(jobs))
    if workers == 1:
        init_worker(panel, selections, args, period); iterator = map(lambda x: predict_spec(*x), jobs); pool = None
    else:
        pool = mp.get_context("spawn").Pool(workers, initializer=init_worker,
            initargs=(panel, selections, args, period), maxtasksperchild=4)
        iterator = pool.starmap_async(predict_spec, jobs).get()
    try:
        for index, (key, status) in enumerate(iterator, 1):
            print(f"{stage.upper()} {index}/{len(jobs)} {key} [{status}]", flush=True)
    finally:
        if workers > 1:
            pool.close(); pool.join()


def classification_screen(args: argparse.Namespace, all_specs: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    rows = []
    for spec in all_specs:
        prediction = pd.read_csv(cache_path(args, "screen", spec))
        valid = prediction.target.notna() & prediction.probability.notna()
        auc = roc_auc_score(prediction.loc[valid, "target"], prediction.loc[valid, "probability"]) if prediction.loc[valid, "target"].nunique() == 2 else np.nan
        rows.append({"model_key": spec["model_key"], "pair": spec["pair"], "target": spec["target"],
                     "feature_id": spec["feature_id"], "config_id": spec["config"]["config_id"],
                     "auc": auc, "classification_pass": bool(np.isfinite(auc) and auc >= .55)})
    result = pd.DataFrame(rows).sort_values(["pair", "target", "classification_pass", "auc"], ascending=[True, True, False, False])
    result.to_csv(args.output_dir / "classification_screen.csv", index=False)
    return result


def select_finalists(args: argparse.Namespace, screen: pd.DataFrame, all_specs: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    keys = set()
    for pair in PAIRS:
        for target in TARGETS:
            group = screen[(screen.pair == pair) & (screen.target == target)]
            passed = group[group.classification_pass]
            keys.update((passed if not passed.empty else group).head(args.finalists).model_key)
    selected = [dict(item) for item in all_specs if item["model_key"] in keys]
    atomic_json(args.output_dir / "weekly_finalists.json", selected)
    return selected


def gate_candidates() -> list[LongGate]:
    return [LongGate(q, entry, minimum, cooldown) for q in legacy.ENTRY_QUANTILES
            for entry in (1, 2) for minimum in (24, 48) for cooldown in (24, 48)]


def build_long_state(prediction: pd.DataFrame, pair: str, gate: LongGate,
                     period: ResearchPeriod = PERIOD, include_timeline: bool = True,
                     include_states: bool = True) -> tuple[dict[int, bool], pd.DataFrame, pd.DataFrame]:
    rows = prediction.sort_values("signal_ts").reset_index(drop=True)
    threshold_col = legacy.v5.quantile_column(gate.entry_quantile)
    timestamps = rows.signal_ts.to_numpy(np.int64)
    probabilities = rows.probability.to_numpy(float)
    thresholds = rows[threshold_col].to_numpy(float)
    complete4h = rows.last_complete_4h_ts.to_numpy(np.int64)
    roc = rows.roc_48h_4h.to_numpy(float); sqz = rows.sqzmom_pct_4h.to_numpy(float)
    di = rows.di_spread.to_numpy(float); slope = rows.ema20_slope_atr_12h.to_numpy(float)
    below = rows.below_ema20_ratio_72h.to_numpy(float)
    active = False; above = 0; armed_until = -1; cooldown_until = -1; start = None
    recovery = 0; last_4h = None; previous_4h = None; timeline: dict[int, bool] = {}
    states, intervals = [], []
    for index in range(len(rows)):
        ts, probability, threshold = int(timestamps[index]), float(probabilities[index]), float(thresholds[index])
        above = above + 1 if probability >= threshold else 0
        if not active and above >= gate.entry_bars:
            armed_until = max(armed_until, ts + gate.arm_hours * HOUR)
        new_4h = last_4h != int(complete4h[index])
        current = (float(roc[index]), float(sqz[index]), float(di[index]),
                   float(slope[index]), float(below[index]))
        worsening = False; improving = False
        if new_4h and previous_4h is not None:
            worsening = (current[0] < previous_4h[0] <= 0 and current[1] < previous_4h[1] <= 0
                         and (current[2] < 0 or current[3] < 0 or current[4] >= .5))
            improving = (current[0] > previous_4h[0] and current[1] > previous_4h[1]
                         and (current[2] > previous_4h[2] or current[3] >= 0 or current[4] < .5))
            previous_4h = current
        elif new_4h:
            previous_4h = current
        if new_4h: last_4h = int(complete4h[index])
        transition = "hold" if active else "clear"
        if not active and ts >= cooldown_until and ts <= armed_until and worsening:
            active = True; start = ts; recovery = 0; transition = "enter"
        elif active and new_4h:
            recovery = recovery + 1 if improving else 0
            if start is not None and ts - start >= gate.minimum_hours * HOUR and recovery >= gate.recovery_4h_bars:
                active = False; transition = "recover"; cooldown_until = ts + gate.cooldown_hours * HOUR
                intervals.append({"pair": pair, "start_ts": start, "end_ts": ts,
                                  "duration_hours": (ts - start) / HOUR, "end_reason": "two_4h_structure_improvements"})
                start = None; above = 0; armed_until = -1
        right = min(int(timestamps[index + 1]) if index + 1 < len(rows) else period.end_ts, period.end_ts)
        if include_timeline:
            for timestamp in range(max(ts, period.start_ts), right, 300): timeline[timestamp] = not active
        if include_states:
            states.append({"pair": pair, "signal_ts": ts, "probability": probability,
                           "entry_threshold": threshold, "risk_off_active": active, "buy_enabled": not active,
                           "armed": ts <= armed_until, "structure_worsening": worsening,
                           "structure_recovery_count": recovery, "transition": transition})
    if start is not None:
        intervals.append({"pair": pair, "start_ts": start, "end_ts": period.end_ts,
                          "duration_hours": (period.end_ts - start) / HOUR, "end_reason": "research_period_end"})
    return timeline, pd.DataFrame(states), pd.DataFrame(intervals)


def anchor_metrics(intervals: pd.DataFrame, pair: str, period: ResearchPeriod = PERIOD) -> dict[str, Any]:
    group = intervals[intervals.pair.eq(pair)] if not intervals.empty else intervals
    result: dict[str, Any] = {"interval_count": len(group)}; passed = len(group) <= 8; overlap_total = 0.0
    for name, start, end in ANCHOR_WINDOWS:
        overlap = (np.maximum(0, np.minimum(group.end_ts, end) - np.maximum(group.start_ts, start))
                   if not group.empty else np.array([]))
        coverage = float(overlap.sum() / (end - start)) if len(overlap) else 0.0
        timely = bool(((group.start_ts <= start + 12 * HOUR) & (group.end_ts > start)).any()) if not group.empty else False
        result[f"{name}_coverage"] = coverage; result[f"{name}_timely"] = timely
        overlap_total += float(overlap.sum()) if len(overlap) else 0.0
        passed = passed and coverage >= .70 and timely
    total = float((group.end_ts - group.start_ts).sum()) if not group.empty else 0.0
    result["outside_anchor_share"] = max(0.0, total - overlap_total) / (period.end_ts - period.start_ts)
    result["anchor_pass"] = bool(passed and result["outside_anchor_share"] <= .20)
    return result


def structural_search(args: argparse.Namespace, selected: Sequence[Mapping[str, Any]]) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict[str, dict[int, bool]]]:
    rows, predictions, timelines = [], {}, {}
    for spec in selected:
        prediction = pd.read_csv(cache_path(args, "weekly", spec)); predictions[str(spec["model_key"])] = prediction
        for gate in gate_candidates():
            _, _, intervals = build_long_state(prediction, str(spec["pair"]), gate,
                                                include_timeline=False, include_states=False)
            metrics = anchor_metrics(intervals, str(spec["pair"]))
            candidate_id = f"{spec['model_key']}|{gate.entry_quantile}|{gate.entry_bars}|{gate.minimum_hours}|{gate.cooldown_hours}"
            rows.append({"candidate_id": candidate_id, "model_key": spec["model_key"], "pair": spec["pair"],
                         "target": spec["target"], "feature_id": spec["feature_id"], **asdict(gate), **metrics,
                         "active_hours": float(intervals.duration_hours.sum()) if not intervals.empty else 0.0})
    result = pd.DataFrame(rows); result["structure_pass"] = result.anchor_pass.astype(bool)
    result["minimum_anchor_coverage"] = result[[f"{name}_coverage" for name, _, _ in ANCHOR_WINDOWS]].min(axis=1)
    result = result.sort_values(["pair", "structure_pass", "minimum_anchor_coverage", "interval_count", "outside_anchor_share"], ascending=[True, False, False, True, True])
    result.to_csv(args.output_dir / "structural_search.csv", index=False)
    # Five-minute dictionaries are expensive; materialize only candidates
    # that can actually reach the constrained Grid comparison.
    for pair in PAIRS:
        for row in result[(result.pair == pair) & result.structure_pass].head(8).itertuples(index=False):
            gate = LongGate(float(row.entry_quantile), int(row.entry_bars), int(row.minimum_hours), int(row.cooldown_hours))
            timeline, _, _ = build_long_state(predictions[str(row.model_key)], pair, gate)
            timelines[str(row.candidate_id)] = timeline
    return result, predictions, timelines


def hard_stop_if_no_structure(args: argparse.Namespace, structure: pd.DataFrame) -> None:
    missing = [pair for pair in PAIRS if structure[(structure.pair == pair) & structure.structure_pass].empty]
    if missing:
        payload = {"model_version": MODEL_VERSION, "verdict": "NO-GO", "deployment_allowed": False,
                   "reason": "no_structurally_eligible_candidate", "pairs_without_candidate": missing,
                   "grid_search_executed": False, "period": asdict(PERIOD)}
        atomic_json(args.output_dir / "locked_configuration.json", payload)
        atomic_json(args.output_dir / "summary.json", payload)
        raise RuntimeError(f"hard stop: no structural candidate for {', '.join(missing)}")


def combine_timelines(pair_timelines: Mapping[str, Mapping[int, bool]], period: ResearchPeriod = PERIOD) -> dict[str, dict[int, bool]]:
    result = {pair: {} for pair in PAIRS}
    for pair in PAIRS:
        source = pair_timelines[pair]
        for timestamp in range(period.start_ts, period.end_ts, 300):
            result[pair][timestamp] = bool(source.get(timestamp, False))
    return result


def exact_replay(candles: Mapping[str, pd.DataFrame], selections: pd.DataFrame,
                 technical_gate: Mapping[str, Mapping[int, bool]], taker_fee: float = .001,
                 slippage: float = 0.0, return_details: bool = False) -> dict[str, Any]:
    weekly, pair_rows, curves, all_trades = [], [], [], []; cumulative = 0.0
    for selection in selections.itertuples(index=False):
        trades: list[dict[str, Any]] = []
        result, curve, pairs = simulate(slice_window(dict(candles), int(selection.test_start), int(selection.test_end)),
            grid.candidate_from_row(selection), maker_fee=0.0, taker_fee=taker_fee, slippage=slippage,
            order_refresh_seconds=7200, technical_buy_gate=technical_gate, momentum_stop_timeline=None,
            trade_log=trades, risk_breakers_enabled=True, cost_floor_enabled=True,
            inventory_exit_policy=legacy.base.POLICY, record_curve=True)
        stops = inventory.stop_metrics(result, curve, trades, int(selection.test_end))
        weekly.append({"fold": int(selection.fold), **result, **stops})
        pair_rows.extend({"fold": int(selection.fold), "pair": pair, **value} for pair, value in pairs.items())
        all_trades.extend({"fold": int(selection.fold), **trade} for trade in trades)
        if not curve.empty:
            item = curve[["timestamp", "equity"]].copy(); item["cumulative_oos_pnl"] = cumulative + item.equity - legacy.base.INITIAL_EQUITY
            curves.append(item)
        cumulative += float(result["net_pnl_quote"])
    summary = inventory.aggregate_rows(weekly, pair_rows)
    curve = pd.concat(curves, ignore_index=True).sort_values("timestamp")
    equity = legacy.base.INITIAL_EQUITY + curve.cumulative_oos_pnl
    summary["stitched_max_drawdown_pct"] = float((equity / equity.cummax() - 1).min() * 100)
    summary["risk_off_pair_hours"] = float(pd.DataFrame(pair_rows).technical_risk_off_seconds.sum() / HOUR)
    if not return_details:
        return summary
    return {"summary": summary, "weekly": pd.DataFrame(weekly), "pairs": pd.DataFrame(pair_rows),
            "equity": curve, "trades": pd.DataFrame(all_trades)}


def load_candles(source_dir: Path) -> dict[str, pd.DataFrame]:
    output = {}
    for pair in PAIRS:
        path = source_dir / "extended_candles" / f"binance_{pair}_5m.csv"
        item = pd.read_csv(path)
        item = item[item.timestamp.between(START_TS, END_TS, inclusive="left")].copy()
        expected = np.arange(START_TS, END_TS, 300, dtype=np.int64)
        if len(item) != len(expected) or not np.array_equal(item.timestamp.to_numpy(np.int64), expected):
            raise RuntimeError(f"{pair} candle sequence is incomplete for explicit 250-day period")
        output[pair] = item.reset_index(drop=True)
    return output


def baseline_reference(args: argparse.Namespace) -> dict[str, Any]:
    path = args.source_dir.parent / "xgboost_grid_long_risk_gate_v16_250d" / "summary.json"
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload.get("baseline"), dict): return payload["baseline"]
    return {"oos_pnl_fdusd": -21.6682188545, "stitched_max_drawdown_pct": -18.950285,
            "pair_stop_events": 28, "portfolio_stop_events": 2}


def grid_search(args: argparse.Namespace, structure: pd.DataFrame,
                timelines: Mapping[str, Mapping[int, bool]], candles: Mapping[str, pd.DataFrame],
                selections: pd.DataFrame) -> pd.DataFrame:
    pools = {pair: structure[(structure.pair == pair) & structure.structure_pass].head(8) for pair in PAIRS}
    rows = []; total = len(pools[PAIRS[0]]) * len(pools[PAIRS[1]])
    for index, (btc, eth) in enumerate(((b, e) for _, b in pools[PAIRS[0]].iterrows()
                                       for _, e in pools[PAIRS[1]].iterrows()), 1):
        gate = combine_timelines({"BTC-FDUSD": timelines[btc.candidate_id], "ETH-FDUSD": timelines[eth.candidate_id]})
        metrics = exact_replay(candles, selections, gate)
        rows.append({"candidate_id": f"{btc.candidate_id}||{eth.candidate_id}",
                     "BTC_candidate_id": btc.candidate_id, "ETH_candidate_id": eth.candidate_id,
                     **metrics})
        print(f"GRID {index}/{total}", flush=True)
    result = pd.DataFrame(rows)
    result["profit_percentile"] = result.oos_pnl_fdusd.rank(pct=True)
    result["drawdown_percentile"] = result.stitched_max_drawdown_pct.rank(pct=True)
    result["objective_score"] = .5 * result.profit_percentile + .5 * result.drawdown_percentile
    baseline = baseline_reference(args)
    result["eligible"] = ((result.oos_pnl_fdusd > 0) & (result.oos_pnl_fdusd > float(baseline["oos_pnl_fdusd"]))
        & (result.stitched_max_drawdown_pct >= float(baseline["stitched_max_drawdown_pct"]))
        & (result.btc_pnl_fdusd >= 0) & (result.eth_pnl_fdusd >= 0)
        & (result.portfolio_stop_events == 0) & (result.pair_stop_events < 7))
    result = result.sort_values(["eligible", "objective_score", "portfolio_stop_events", "pair_stop_events",
                                 "risk_off_pair_hours"], ascending=[False, False, True, True, True]).reset_index(drop=True)
    result["rank"] = np.arange(1, len(result) + 1)
    result.to_csv(args.output_dir / "grid_search.csv", index=False)
    winner = result.iloc[0].to_dict()
    atomic_json(args.output_dir / "locked_configuration.json", {
        "schema": "xgboost-grid-long-risk-gate-v19-lock-v1", "model_version": MODEL_VERSION,
        "deployment_allowed": False, "shadow_mode": True, "short_spike_enabled": False,
        "market_sell_action": False, "mechanism1_fallback_allowed": False,
        "evidence_status": "250d_known_window_in_sample_targeted_optimization",
        "verdict": "SEARCH_LOCKED" if bool(winner["eligible"]) else "DIAGNOSTIC_ONLY",
        "candidate": winner, "period": asdict(PERIOD),
        "feature_panel_sha256": sha256_file(args.output_dir / "feature_panel.csv.gz"),
        "grid_sequence_sha256": sha256_file(args.source_dir / "grid_selections.csv"),
    })
    return result


def finalize(args: argparse.Namespace, structure: pd.DataFrame, timelines: Mapping[str, Mapping[int, bool]],
             predictions: Mapping[str, pd.DataFrame], candles: Mapping[str, pd.DataFrame],
             selections: pd.DataFrame) -> dict[str, Any]:
    lock = json.loads((args.output_dir / "locked_configuration.json").read_text(encoding="utf-8"))
    winner = lock["candidate"]
    selected_rows = {pair: structure[structure.candidate_id.eq(winner[f"{pair[:3]}_candidate_id"])].iloc[0]
                     for pair in PAIRS}
    gates = combine_timelines({pair: timelines[row.candidate_id] for pair, row in selected_rows.items()})
    detail = exact_replay(candles, selections, gates, return_details=True)
    states, intervals = [], []
    for pair, row in selected_rows.items():
        gate = LongGate(float(row.entry_quantile), int(row.entry_bars), int(row.minimum_hours), int(row.cooldown_hours))
        _, state, interval = build_long_state(predictions[str(row.model_key)], pair, gate)
        states.append(state); intervals.append(interval)
    state_frame = pd.concat(states, ignore_index=True); interval_frame = pd.concat(intervals, ignore_index=True)
    state_frame.to_csv(args.output_dir / "final_risk_states.csv.gz", index=False, compression="gzip")
    interval_frame.to_csv(args.output_dir / "final_risk_intervals.csv", index=False)
    state_frame[state_frame.transition.isin(["enter", "recover"])].to_csv(
        args.output_dir / "final_risk_events.csv", index=False)
    for name in ("weekly", "pairs", "equity", "trades"):
        detail[name].to_csv(args.output_dir / f"final_{name}.csv.gz", index=False, compression="gzip")
    stress_rows = []
    for scenario, fee, slip, data in (
        ("taker_150pct", .0015, 0.0, candles), ("slippage_0.05pct", .001, .0005, candles),
        ("slippage_0.10pct", .001, .001, candles), ("single_day_15pct_drop", .001, 0.0, crash_candles(dict(candles), .15)),
    ):
        stress_rows.append({"scenario": scenario, **exact_replay(data, selections, gates, fee, slip)})
    stress = pd.DataFrame(stress_rows); stress.to_csv(args.output_dir / "stress_tests.csv", index=False)
    stress_pass = bool(((stress.pair_stop_events == 0) & (stress.portfolio_stop_events == 0)).all())
    verdict = "NEXT_STAGE_JOINT_VALIDATION" if bool(winner["eligible"]) and stress_pass else "NO-GO"
    summary = {"model_version": MODEL_VERSION, "deployment_allowed": False, "short_spike_enabled": False,
               "evidence_status": lock["evidence_status"], "verdict": verdict, "period": asdict(PERIOD),
               "baseline": baseline_reference(args), "metrics": detail["summary"],
               "stress_all_no_stops": stress_pass, "locked_candidate": winner}
    atomic_json(args.output_dir / "summary.json", summary)
    return summary


def build_plot(args: argparse.Namespace, candles: Mapping[str, pd.DataFrame]) -> Path:
    states = pd.read_csv(args.output_dir / "final_risk_states.csv.gz")
    intervals = pd.read_csv(args.output_dir / "final_risk_intervals.csv")
    summary = json.loads((args.output_dir / "summary.json").read_text(encoding="utf-8"))
    figure = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=.035,
                           subplot_titles=("BTC-FDUSD price", "BTC long probability", "ETH-FDUSD price", "ETH long probability"))
    risk_shape_indices: dict[str, list[int]] = {pair: [] for pair in PAIRS}
    for offset, pair in enumerate(PAIRS):
        price_row, probability_row = 1 + 2 * offset, 2 + 2 * offset
        price = candles[pair].iloc[::12]
        figure.add_trace(go.Scatter(x=pd.to_datetime(price.timestamp, unit="s", utc=True), y=price.close,
                                    name=f"{pair} close", line={"width": 1.2}), row=price_row, col=1)
        state = states[states.pair.eq(pair)]
        figure.add_trace(go.Scatter(x=pd.to_datetime(state.signal_ts, unit="s", utc=True), y=state.probability,
                                    name=f"{pair} long probability", line={"width": 1}), row=probability_row, col=1)
        figure.add_trace(go.Scatter(x=pd.to_datetime(state.signal_ts, unit="s", utc=True), y=state.entry_threshold,
                                    name=f"{pair} threshold", line={"dash": "dot", "width": 1}), row=probability_row, col=1)
        group = intervals[intervals.pair.eq(pair)]
        for number, interval in enumerate(group.itertuples(index=False)):
            figure.add_vrect(x0=pd.to_datetime(interval.start_ts, unit="s", utc=True),
                             x1=pd.to_datetime(interval.end_ts, unit="s", utc=True), fillcolor="#f59e0b",
                             opacity=.16, line_width=0, row=price_row, col=1,
                             name=f"{pair} long Risk-off shadow", legendgroup=f"{pair}-long-shadow",
                             showlegend=number == 0)
            risk_shape_indices[pair].append(len(figure.layout.shapes) - 1)
        events = state[state.transition.isin(["enter", "recover"])]
        for event, symbol, color in (("enter", "triangle-down", "#dc2626"), ("recover", "triangle-up", "#16a34a")):
            points = events[events.transition.eq(event)]
            if not points.empty:
                px = pd.merge_asof(points[["signal_ts"]].sort_values("signal_ts"), price[["timestamp", "close"]],
                                   left_on="signal_ts", right_on="timestamp", direction="backward")
                figure.add_trace(go.Scatter(x=pd.to_datetime(px.signal_ts, unit="s", utc=True), y=px.close,
                    mode="markers", marker={"symbol": symbol, "color": color, "size": 9},
                    name=f"{pair} {event}"), row=price_row, col=1)
    for name, start, end in ANCHOR_WINDOWS:
        for row in (1, 3):
            figure.add_vrect(x0=pd.to_datetime(start, unit="s", utc=True), x1=pd.to_datetime(end, unit="s", utc=True),
                             fillcolor="#7c3aed", opacity=.08, line_width=1, line_dash="dash", row=row, col=1)
    metrics = summary.get("metrics", {})
    if metrics:
        title = (f"v19 long-only 250d | PnL {metrics.get('oos_pnl_fdusd', float('nan')):.3f} FDUSD | "
                 f"DD {metrics.get('stitched_max_drawdown_pct', float('nan')):.3f}% | {summary['verdict']}")
    else:
        title = "v19 long-only 250d | structural diagnostic (Grid not executed) | NO-GO"
    def visibility_update(pair: str, visible: bool) -> dict[str, bool]:
        return {f"shapes[{index}].visible": visible for index in risk_shape_indices[pair]}

    figure.update_layout(
        height=1150, template="plotly_white", hovermode="x unified", title=title,
        margin={"t": 145},
        updatemenus=[
            {
                "type": "buttons", "direction": "right", "x": 0.0, "y": 1.105,
                "xanchor": "left", "yanchor": "top", "showactive": True,
                "buttons": [
                    {"label": "BTC Risk-off ON", "method": "relayout",
                     "args": [visibility_update("BTC-FDUSD", True)]},
                    {"label": "BTC Risk-off OFF", "method": "relayout",
                     "args": [visibility_update("BTC-FDUSD", False)]},
                ],
            },
            {
                "type": "buttons", "direction": "right", "x": 0.36, "y": 1.105,
                "xanchor": "left", "yanchor": "top", "showactive": True,
                "buttons": [
                    {"label": "ETH Risk-off ON", "method": "relayout",
                     "args": [visibility_update("ETH-FDUSD", True)]},
                    {"label": "ETH Risk-off OFF", "method": "relayout",
                     "args": [visibility_update("ETH-FDUSD", False)]},
                ],
            },
        ],
        annotations=[
            *list(figure.layout.annotations),
            {"text": "Risk-off阴影独立开关（不影响价格、概率及进入/退出标记）",
             "xref": "paper", "yref": "paper", "x": 0.0, "y": 1.145,
             "xanchor": "left", "yanchor": "top", "showarrow": False,
             "font": {"size": 12, "color": "#475569"}},
        ],
    )
    target = args.output_dir / "xgboost_v19_long_only_250d_plotly.html"
    figure.write_html(target, include_plotlyjs=True, full_html=True)
    return target


def diagnostic_artifacts(args: argparse.Namespace, structure: pd.DataFrame,
                         predictions: Mapping[str, pd.DataFrame], candles: Mapping[str, pd.DataFrame]) -> dict[str, Any]:
    """Render the best constraint diagnostic without running or implying Grid selection."""
    chosen = structure.groupby("pair", sort=False).head(1).copy()
    states, intervals = [], []
    pair_summary: dict[str, Any] = {}
    for row in chosen.itertuples(index=False):
        gate = LongGate(float(row.entry_quantile), int(row.entry_bars), int(row.minimum_hours), int(row.cooldown_hours))
        _, state, interval = build_long_state(predictions[str(row.model_key)], str(row.pair), gate,
                                               include_timeline=False, include_states=True)
        states.append(state); intervals.append(interval)
        pair_summary[str(row.pair)] = {
            "model_key": str(row.model_key), "entry_quantile": float(row.entry_quantile),
            "entry_bars": int(row.entry_bars), "minimum_hours": int(row.minimum_hours),
            "cooldown_hours": int(row.cooldown_hours), "interval_count": int(row.interval_count),
            "outside_anchor_share": float(row.outside_anchor_share),
            **{f"{name}_coverage": float(getattr(row, f"{name}_coverage")) for name, _, _ in ANCHOR_WINDOWS},
            **{f"{name}_timely": bool(getattr(row, f"{name}_timely")) for name, _, _ in ANCHOR_WINDOWS},
        }
    state_frame = pd.concat(states, ignore_index=True)
    state_frame.to_csv(args.output_dir / "final_risk_states.csv.gz", index=False, compression="gzip")
    state_frame[state_frame.transition.isin(["enter", "recover"])].to_csv(
        args.output_dir / "final_risk_events.csv", index=False)
    pd.concat(intervals, ignore_index=True).to_csv(args.output_dir / "final_risk_intervals.csv", index=False)
    chosen.to_csv(args.output_dir / "diagnostic_selection.csv", index=False)
    summary = json.loads((args.output_dir / "summary.json").read_text(encoding="utf-8"))
    summary.update({"diagnostic_pairs": pair_summary, "plotly": build_plot(args, candles).as_posix()})
    atomic_json(args.output_dir / "summary.json", summary)
    return summary


def main() -> int:
    mp.freeze_support(); args = parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)
    selections = pd.read_csv(args.source_dir / "grid_selections.csv")
    if int(selections.test_start.min()) != START_TS or int(selections.test_end.max()) != END_TS:
        raise RuntimeError("grid sequence does not exactly match explicit 250-day period")
    panel_path = args.output_dir / "feature_panel.csv.gz"
    if args.stage in {"prepare", "all"}:
        panel = prepare_panel(pd.read_csv(args.source_dir / "feature_panel.csv.gz"))
        panel.to_csv(panel_path, index=False, compression="gzip")
    else: panel = pd.read_csv(panel_path)
    if args.stage == "prepare": return 0
    all_specs = specs()
    if args.stage in {"screen", "all"}:
        run_jobs(args, panel, selections, "screen", all_specs)
        selected = select_finalists(args, classification_screen(args, all_specs), all_specs)
    else: selected = json.loads((args.output_dir / "weekly_finalists.json").read_text(encoding="utf-8"))
    if args.stage == "screen": return 0
    if args.stage in {"train", "all"}: run_jobs(args, panel, selections, "weekly", selected)
    if args.stage == "train": return 0
    candles = load_candles(args.source_dir)
    structure, predictions, timelines = structural_search(args, selected)
    try:
        hard_stop_if_no_structure(args, structure)
    except RuntimeError:
        result = diagnostic_artifacts(args, structure, predictions, candles)
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str), flush=True)
        return 2
    if args.stage in {"search", "all"}: grid_search(args, structure, timelines, candles, selections)
    if args.stage == "search": return 0
    if args.stage in {"finalize", "all"}: result = finalize(args, structure, timelines, predictions, candles, selections)
    else: result = json.loads((args.output_dir / "summary.json").read_text(encoding="utf-8"))
    if args.stage in {"plot", "all"}:
        result["plotly"] = build_plot(args, candles).as_posix(); atomic_json(args.output_dir / "summary.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

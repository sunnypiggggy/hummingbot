#!/usr/bin/env python3
"""Independent SOL-FDUSD weekly walk-forward v22-style long-risk model.

The module intentionally consumes SOL candles only.  It reuses the frozen
v22 training and state-machine semantics (mature 72h onset label, disjoint
early-stop/calibration windows, fold-local thresholds, and continuous state)
without importing a BTC/ETH model or feature.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

import retrain_xgboost_long_risk_gate_250d_v19 as v19
from xgboost_long_risk_gate_v22_features import (
    add_momentum_features,
    expected_shortfall,
    roc_sqz_signal_from_klines,
    rolling_duration_from_high,
    rolling_percentile,
)


PAIR = "SOL-FDUSD"
MODEL_VERSION = "xgboost-sol-grid-long-risk-gate-v22-weekly-360d"
MODEL_BUNDLE_SCHEMA = "xgboost-sol-grid-long-risk-gate-v22-weekly-bundle-v1"
STATE_SCHEMA = "xgboost-sol-grid-long-risk-gate-v22-weekly-state-v1"
HOUR = 3600
DAY = 86400

# The ETH v22 directional-persistence schema is the scale-appropriate frozen
# starting point for SOL.  BTC dependence is replaced by SOL's own volatility.
FEATURES = (
    "adx_14", "di_spread", "atr_pct", "sol_volatility_20",
    "roc_48h_4h", "sqzmom_pct_4h", "sqzmom_slope_4h",
    "drawdown_from_high_72h", "drawdown_from_high_168h",
    "drawdown_duration_168h", "below_ema20_ratio_72h",
    "lower_low_ratio_72h", "downside_semivariance_ratio_72h",
    "trend_efficiency_72h", "ema20_slope_atr_12h",
)

XGB_CONFIG: dict[str, Any] = {
    "config_id": "xgb_34", "order": 34, "kind": "sampled",
    "uses_early_stopping": True, "learning_rate": 0.015,
    "n_estimators": 1200, "subsample": 0.8, "reg_lambda": 20.0,
    "reg_alpha": 0.5, "min_child_weight": 10, "max_depth": 6,
    "max_bin": 512, "gamma": 0.15, "colsample_bytree": 0.65,
}


@dataclass(frozen=True)
class GateConfig:
    entry_quantile: float = 0.985
    entry_bars: int = 2
    arm_hours: int = 48
    minimum_hours: int = 48
    cooldown_hours: int = 24
    recovery_4h_bars: int = 3
    confirmation_mode: str = "persistent_bearish"
    recovery_mode: str = "adaptive_relief"


GATE = GateConfig()


@dataclass
class GateState:
    active: bool = False
    since: int | None = None
    above_entry_count: int = 0
    armed_until: int | None = None
    cooldown_until: int | None = None
    recovery_count: int = 0
    last_complete_4h_ts: int | None = None
    previous_structure: tuple[float, float, float, float, float] | None = None
    last_signal_ts: int | None = None
    last_event_id: str | None = None
    probability_history: list[tuple[int, float]] = field(default_factory=list)


def sha256_json(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def hourly_bars(candles: pd.DataFrame) -> pd.DataFrame:
    item = candles.copy()
    item["datetime"] = pd.to_datetime(item.timestamp, unit="s", utc=True)
    bars = item.set_index("datetime").resample(
        "1h", label="left", closed="left", origin="epoch",
    ).agg(
        open=("open", "first"), high=("high", "max"), low=("low", "min"),
        close=("close", "last"), volume=("volume", "sum"), rows=("close", "size"),
    )
    return bars[bars.rows.eq(12)].drop(columns="rows")


def four_hour_frame(candles: pd.DataFrame) -> pd.DataFrame:
    source = candles.sort_values("timestamp").assign(
        bucket=lambda value: (value.timestamp.astype("int64") // 14400) * 14400,
    )
    bars = source.groupby("bucket", sort=True).agg(
        open=("open", "first"), high=("high", "max"), low=("low", "min"),
        close=("close", "last"), rows=("close", "size"),
    ).reset_index()
    klines: list[list[Any]] = []
    rows: list[dict[str, float]] = []
    for row in bars[bars.rows.eq(48)].itertuples(index=False):
        close_time = int(row.bucket) + 14399
        klines.append([
            int(row.bucket) * 1000, row.open, row.high, row.low, row.close,
            0, close_time * 1000,
        ])
        if len(klines) >= 40:
            rows.append({
                "last_complete_4h_ts": close_time + 1,
                **roc_sqz_signal_from_klines(klines[-64:]),
            })
    out = pd.DataFrame(rows).rename(columns={
        "roc_48h_pct": "roc_48h_4h", "sqzmom_pct": "sqzmom_pct_4h",
        "sqzmom": "sqzmom_value_4h",
    })
    out["sqzmom_slope_4h"] = (
        (out.sqzmom_value_4h - out.sqzmom_previous)
        / out.close.replace(0, np.nan) * 100
    )
    return out[[
        "last_complete_4h_ts", "roc_48h_4h", "sqzmom_pct_4h",
        "sqzmom_slope_4h",
    ]].sort_values("last_complete_4h_ts")


def _future_fraction_below(close: pd.Series, hours: int) -> np.ndarray:
    return (
        pd.concat([close.shift(-offset) for offset in range(1, hours + 1)], axis=1)
        .lt(close, axis=0).sum(axis=1).to_numpy(float) / float(hours)
    )


def build_panel(candles: pd.DataFrame) -> pd.DataFrame:
    bars = add_momentum_features(hourly_bars(candles))
    close, low = bars.close.astype(float), bars.low.astype(float)
    log_return = np.log(close).diff()
    bars["sol_volatility_20"] = bars.return_1.rolling(20).std(ddof=0)
    bars["bar_open_ts"] = bars.index.astype("int64") // 10**9
    bars["signal_ts"] = bars.bar_open_ts + HOUR
    bars["pair"] = PAIR
    base = bars.reset_index(names="bar_open_utc").sort_values("signal_ts")
    panel = pd.merge_asof(
        base, four_hour_frame(candles), left_on="signal_ts",
        right_on="last_complete_4h_ts", direction="backward",
    )
    close, low = panel.close.astype(float), panel.low.astype(float)
    log_return = np.log(close).diff()
    negative, total = log_return.clip(upper=0).pow(2), log_return.pow(2)
    ema20 = close.ewm(span=20, adjust=False).mean()
    rv24 = total.rolling(24, min_periods=12).sum().pow(.5)
    panel["drawdown_from_high_72h"] = close / close.rolling(72, min_periods=36).max() - 1
    panel["drawdown_from_high_168h"] = close / close.rolling(168, min_periods=84).max() - 1
    panel["drawdown_duration_168h"] = rolling_duration_from_high(close, 168)
    panel["below_ema20_ratio_72h"] = close.lt(ema20).astype(float).rolling(72, min_periods=36).mean()
    panel["lower_low_ratio_72h"] = low.lt(low.shift()).astype(float).rolling(72, min_periods=36).mean()
    panel["downside_semivariance_ratio_72h"] = (
        negative.rolling(72, min_periods=36).sum()
        / total.rolling(72, min_periods=36).sum().replace(0, np.nan)
    )
    panel["rv_24h_percentile_30d"] = rolling_percentile(rv24)
    panel["vol_of_vol_72h"] = (
        rv24.rolling(72, min_periods=36).std(ddof=0)
        / rv24.rolling(72, min_periods=36).mean().replace(0, np.nan)
    )
    panel["trend_efficiency_72h"] = (
        (close - close.shift(72)).abs()
        / close.diff().abs().rolling(72, min_periods=36).sum().replace(0, np.nan)
    )
    panel["ema20_slope_atr_12h"] = (
        (ema20 - ema20.shift(12))
        / (panel.atr_pct.astype(float) * close).replace(0, np.nan)
    )
    panel["historical_var_72h"] = log_return.rolling(72, min_periods=36).quantile(.05)
    panel["expected_shortfall_72h"] = log_return.rolling(72, min_periods=36).apply(
        expected_shortfall, raw=True,
    )
    panel["negative_skew_72h"] = -log_return.rolling(72, min_periods=36).skew()

    future_close = close.shift(-72)
    threshold = np.maximum(.03, 3.0 * panel.atr_pct.astype(float))
    valid = future_close.notna()
    panel["target_long_72h"] = (
        (future_close / close - 1 <= -threshold)
        & (_future_fraction_below(close, 72) >= 2.0 / 3.0)
    ).astype(float).where(valid)
    onset, uniqueness = v19.event_onset_target(panel.target_long_72h, 72)
    panel["target"] = onset
    panel["event_uniqueness"] = uniqueness
    panel["label_ready_ts"] = panel.signal_ts + 96 * HOUR
    panel[list(FEATURES)] = panel[list(FEATURES)].replace([np.inf, -np.inf], np.nan)
    return panel.dropna(subset=list(FEATURES)).sort_values("signal_ts").reset_index(drop=True)


def weekly_folds(start: pd.Timestamp, end: pd.Timestamp) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    start = start.tz_convert("UTC") if start.tzinfo else start.tz_localize("UTC")
    end = end.tz_convert("UTC") if end.tzinfo else end.tz_localize("UTC")
    next_monday = start.normalize() + pd.Timedelta(days=(7 - start.weekday()) % 7)
    if next_monday <= start:
        next_monday += pd.Timedelta(days=7)
    points = [start]
    point = next_monday
    while point < end:
        points.append(point)
        point += pd.Timedelta(days=7)
    points.append(end)
    return list(zip(points[:-1], points[1:]))


def _finite_structure(values: Sequence[float]) -> tuple[float, float, float, float, float]:
    result = tuple(float(value) for value in values)
    if len(result) != 5 or not all(math.isfinite(value) for value in result):
        raise ValueError("SOL directional structure contains non-finite values")
    return result  # type: ignore[return-value]


def _entry_confirm(current: tuple[float, ...]) -> bool:
    roc, sqz, di, slope, below = current
    votes = int(di < 0) + int(slope < 0) + int(below >= .50)
    return bool(roc < 0 and sqz < 0 and votes >= 2 and (below >= .55 or (di < 0 and slope < 0)))


def _recovery_confirm(current: tuple[float, ...], previous: tuple[float, ...] | None) -> bool:
    if previous is None:
        return False
    roc, sqz, di, slope, below = current
    votes = int(di > 0) + int(slope >= 0) + int(below < .50)
    return bool(roc > previous[0] and sqz > previous[1] and votes >= 2)


def advance_gate(
    state: GateState, *, probability: float, threshold: float, signal_ts: int,
    last_complete_4h_ts: int, structure: Sequence[float],
) -> tuple[GateState, str]:
    current = _finite_structure(structure)
    state.probability_history.append((signal_ts, probability))
    state.probability_history = state.probability_history[-48:]
    state.above_entry_count = state.above_entry_count + 1 if probability >= threshold else 0
    if not state.active and state.above_entry_count >= GATE.entry_bars:
        armed = signal_ts + GATE.arm_hours * HOUR
        state.armed_until = max(int(state.armed_until or armed), armed)
    new_4h = state.last_complete_4h_ts != last_complete_4h_ts
    entry_ok = new_4h and _entry_confirm(current)
    recovery_ok = new_4h and _recovery_confirm(current, state.previous_structure)
    transition = "hold" if state.active else "clear"
    if (
        not state.active and signal_ts >= int(state.cooldown_until or -1)
        and signal_ts <= int(state.armed_until or -1) and entry_ok
    ):
        state.active = True
        state.since = signal_ts
        state.recovery_count = 0
        transition = "enter"
    elif state.active and new_4h:
        state.recovery_count = state.recovery_count + 1 if recovery_ok else 0
        strong = (
            (current[0] >= 0 or current[1] >= 0) and current[2] > 0
            and current[3] >= 0 and current[4] < .50
        )
        required = 2 if strong else GATE.recovery_4h_bars
        age = signal_ts - int(state.since or signal_ts)
        if age >= GATE.minimum_hours * HOUR and state.recovery_count >= required:
            state.active = False
            state.since = None
            state.cooldown_until = signal_ts + GATE.cooldown_hours * HOUR
            state.armed_until = None
            state.above_entry_count = 0
            transition = "recover"
    if new_4h:
        state.previous_structure = current
        state.last_complete_4h_ts = last_complete_4h_ts
    state.last_signal_ts = signal_ts
    if transition in {"enter", "recover"}:
        state.last_event_id = hashlib.sha256(
            f"{MODEL_VERSION}|{PAIR}|{signal_ts}|{transition}".encode(),
        ).hexdigest()
    return state, transition


def train_weekly_bundle(
    candles: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp,
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame, pd.DataFrame]:
    panel = build_panel(candles)
    working = panel.copy()
    rows: list[pd.DataFrame] = []
    weeks: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    importance_rows: list[dict[str, Any]] = []
    state = GateState()
    for fold, (test_start, test_end) in enumerate(weekly_folds(start, end), 1):
        cutoff = int(test_start.timestamp())
        test = panel[
            (panel.signal_ts >= cutoff)
            & (panel.signal_ts < int(test_end.timestamp()))
        ].copy()
        if test.empty:
            raise RuntimeError(f"SOL fold {fold} has no OOS rows")
        model, calibration, audit = v19.fit_leakage_safe(
            working, cutoff, XGB_CONFIG, FEATURES,
        )
        threshold = float(calibration.probability.quantile(GATE.entry_quantile))
        test["probability"] = model.predict_proba(test[list(FEATURES)])[:, 1]
        test["entry_threshold"] = threshold
        test["fold"] = fold
        model_hash = hashlib.sha256(
            bytes(model.get_booster().save_raw(raw_format="ubj")),
        ).hexdigest()
        weeks.append({
            "fold": fold, "train_cutoff": cutoff, "test_start": cutoff,
            "test_end": int(test_end.timestamp()), "entry_threshold": threshold,
            "entry_quantile": GATE.entry_quantile,
            "best_tree_count": int(audit["best_tree_count"]),
            "last_label_ready_ts": int(audit["last_label_ready_ts"]),
            "development_last_ts": int(audit["development_last_ts"]),
            "calibration_first_ts": int(audit["calibration_first_ts"]),
            "calibration_rows": int(audit["calibration_rows"]),
            "model_sha256": model_hash, "model": model,
        })
        audit_rows.append({
            "fold": fold, "test_start_utc": test_start.isoformat(),
            "test_end_utc": test_end.isoformat(), "threshold": threshold,
            "model_sha256": model_hash, **audit,
        })
        values = model.feature_importances_.astype(float)
        if values.sum() > 0:
            values /= values.sum()
        importance_rows.extend({
            "fold": fold, "feature": feature, "importance": float(value),
        } for feature, value in zip(FEATURES, values))
        snapshots = []
        for row in test.itertuples(index=False):
            state, transition = advance_gate(
                state, probability=float(row.probability),
                threshold=float(row.entry_threshold), signal_ts=int(row.signal_ts),
                last_complete_4h_ts=int(row.last_complete_4h_ts),
                structure=(
                    row.roc_48h_4h, row.sqzmom_pct_4h, row.di_spread,
                    row.ema20_slope_atr_12h, row.below_ema20_ratio_72h,
                ),
            )
            snapshots.append({
                "risk_off_active": bool(state.active), "transition": transition,
                "event_id": state.last_event_id,
            })
        states = pd.DataFrame(snapshots, index=test.index)
        test[list(states.columns)] = states
        test["model_signal"] = np.where(test.risk_off_active, "RISK_OFF", "RISK_ON")
        rows.append(test[[
            "signal_ts", "close", "target", "probability", "entry_threshold", "fold",
            "risk_off_active", "model_signal", "transition", "event_id",
        ]])
    predictions = pd.concat(rows, ignore_index=True).sort_values("signal_ts")
    expected = np.arange(int(start.timestamp()), int(end.timestamp()), HOUR, dtype=np.int64)
    if not np.array_equal(predictions.signal_ts.to_numpy(np.int64), expected):
        missing = np.setdiff1d(expected, predictions.signal_ts.to_numpy(np.int64))
        raise RuntimeError(f"SOL v22 OOS coverage gap: {missing[:8].tolist()}")
    serializable_weeks = [{k: v for k, v in week.items() if k != "model"} for week in weeks]
    bundle = {
        "schema": MODEL_BUNDLE_SCHEMA, "model_version": MODEL_VERSION,
        "pair": PAIR, "target": "long_event_72h",
        "features": list(FEATURES), "feature_schema_sha256": sha256_json(list(FEATURES)),
        "config": XGB_CONFIG, "gate": asdict(GATE), "weeks": weeks,
        "fold_count": len(weeks),
        "model_lineage_sha256": sha256_json(serializable_weeks),
        "training": {
            "cadence": "weekly_walk_forward", "label_ready_delay_hours": 96,
            "calibration_days": 14, "early_stop_days": 28,
            "state_continuous_across_folds": True,
            "missing_fold_policy": "fail_closed_no_previous_week_fallback",
            "btc_or_eth_dependency": False,
        },
    }
    return predictions, bundle, pd.DataFrame(audit_rows), pd.DataFrame(importance_rows)


def public_bundle_metadata(bundle: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(bundle)
    value["weeks"] = [
        {key: item for key, item in week.items() if key != "model"}
        for week in bundle["weeks"]
    ]
    return value

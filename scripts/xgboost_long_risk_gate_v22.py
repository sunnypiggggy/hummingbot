#!/usr/bin/env python3
"""Weekly walk-forward XGBoost long Risk-off strategy used by v22.

Unlike v21's final-refit bundle, a v22 bundle contains the exact model and
fold-local calibration threshold used for every effective week.  Gate state
is continuous across model rollovers; a missing week is always fail-closed.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from xgboost_long_risk_gate_v22_features import build_inference_panel as _build_inference_panel


MODEL_VERSION = "xgboost-grid-long-risk-gate-v22-weekly-250d"
MODEL_BUNDLE_SCHEMA = "xgboost-grid-long-risk-gate-v22-weekly-bundle-v1"
CONTRACT_SCHEMA = "grid-xgboost-long-risk-gate-v3"
STATE_SCHEMA = "xgboost-long-risk-gate-v22-weekly-state-v1"
STRATEGY_SCHEMA = "xgboost-grid-long-risk-gate-v22-weekly-strategy-v1"
STALE_AFTER_SECONDS = 150
HOUR = 3600
PAIRS = ("BTC-FDUSD", "ETH-FDUSD")
BTC_FEATURES = (
    "adx_14", "di_spread", "atr_pct", "btc_volatility_20", "roc_48h_4h",
    "sqzmom_pct_4h", "sqzmom_slope_4h", "drawdown_from_high_72h",
    "drawdown_from_high_168h", "drawdown_duration_168h", "below_ema20_ratio_72h",
    "lower_low_ratio_72h", "downside_semivariance_ratio_24h",
    "downside_semivariance_ratio_72h", "rv_24h_percentile_30d", "vol_of_vol_72h",
    "trend_efficiency_72h", "ema20_slope_atr_12h", "historical_var_72h",
    "expected_shortfall_72h", "negative_skew_72h", "cross_pair_downside_beta_72h",
    "relative_drawdown_72h",
)
ETH_FEATURES = (
    "adx_14", "di_spread", "atr_pct", "btc_volatility_20", "roc_48h_4h",
    "sqzmom_pct_4h", "sqzmom_slope_4h", "drawdown_from_high_72h",
    "drawdown_from_high_168h", "drawdown_duration_168h", "below_ema20_ratio_72h",
    "lower_low_ratio_72h", "downside_semivariance_ratio_72h",
    "trend_efficiency_72h", "ema20_slope_atr_12h",
)
FEATURES = {"BTC-FDUSD": BTC_FEATURES, "ETH-FDUSD": ETH_FEATURES}


@dataclass(frozen=True)
class GateConfig:
    entry_quantile: float
    entry_bars: int
    arm_hours: int
    minimum_hours: int
    cooldown_hours: int
    recovery_4h_bars: int
    confirmation_mode: str = "persistent_bearish"
    recovery_mode: str = "adaptive_relief"


GATES = {
    "BTC-FDUSD": GateConfig(.98, 1, 48, 48, 48, 3),
    "ETH-FDUSD": GateConfig(.985, 2, 48, 48, 24, 3),
}


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
    structure_history: list[tuple[int, tuple[float, float, float, float, float]]] = field(default_factory=list)


def _base_strategy_spec() -> dict[str, Any]:
    return {
        "schema": "xgboost-grid-long-risk-gate-v21-strategy-v1",
        "implementation_version": "persistent-bearish-adaptive-relief-v1",
        "signal_cadence": "complete_1h_close", "first_execution_delay_seconds": 300,
        "structure_cadence": "complete_4h_close", "pair_state_isolation": True,
        "entry_threshold_comparison": "probability_greater_than_or_equal",
        "entry_confirmation": {"mode": "persistent_bearish", "required": "roc<0 and sqz<0 and at_least_two(di<0,ema_slope<0,below_ema_ratio>=0.50) and (below_ema_ratio>=0.55 or (di<0 and ema_slope<0))"},
        "recovery_confirmation": {"mode": "adaptive_relief", "ordinary": "roc_improves and sqz_improves and at_least_two(di>0,ema_slope>=0,below_ema_ratio<0.50)", "strong": "(roc>=0 or sqz>=0) and di>0 and ema_slope>=0 and below_ema_ratio<0.50", "strong_recovery_4h_bars": 2},
        "actions": {"ordinary_grid_buy": "recommend_pause_while_risk_off", "sell": "unchanged", "market_sell": False, "inventory_timeout": "unchanged", "risk_recovery_buy": "unchanged"},
        "public_shadow_buy_enabled": False,
        "pairs": {pair: {"features": list(FEATURES[pair]), "gate": asdict(GATES[pair])} for pair in PAIRS},
    }


def strategy_spec() -> dict[str, Any]:
    base = _base_strategy_spec()
    return {
        **base,
        "schema": STRATEGY_SCHEMA,
        "model_version": MODEL_VERSION,
        "probability_semantics": "weekly_walk_forward_model_with_fold_local_calibration_threshold",
        "model_rollover": {
            "calendar": "frozen_manifest_test_start_inclusive_test_end_exclusive",
            "forward_staging": "train_cutoff_may_precede_test_start_by_at_most_48h",
            "gate_state_reset": False,
            "missing_week": "fail_closed",
            "previous_week_fallback": False,
            "v21_final_refit_fallback": False,
        },
        "training": {
            "label": "long_event_72h",
            "label_ready_delay_hours": 96,
            "calibration": "last_14_days_of_mature_records",
            "final_fit": "all_mature_development_records_excluding_calibration",
            "early_stopping": "last_14_days_of_development_before_final_fit",
            "purge_hours": 120,
        },
        "pairs": {
            pair: {"features": list(FEATURES[pair]), "gate": asdict(GATES[pair])}
            for pair in PAIRS
        },
    }


def strategy_schema_sha256(spec: Mapping[str, Any] | None = None) -> str:
    encoded = json.dumps(
        strategy_spec() if spec is None else dict(spec),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def feature_schema_sha256() -> str:
    payload = {pair: list(FEATURES[pair]) for pair in PAIRS}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def validate_weekly_bundle(bundle: Mapping[str, Any]) -> None:
    if bundle.get("schema") != MODEL_BUNDLE_SCHEMA or bundle.get("model_version") != MODEL_VERSION:
        raise ValueError("v22 weekly bundle schema/model mismatch")
    embedded_spec = bundle.get("strategy_spec")
    current_spec = strategy_spec()
    legacy_spec = json.loads(json.dumps(current_spec))
    legacy_spec["model_rollover"].pop("forward_staging", None)
    if embedded_spec not in (current_spec, legacy_spec):
        raise ValueError("v22 embedded strategy specification mismatch")
    if bundle.get("strategy_schema_sha256") != strategy_schema_sha256(embedded_spec):
        raise ValueError("v22 strategy hash mismatch")
    if bundle.get("feature_schema_sha256") != feature_schema_sha256():
        raise ValueError("v22 feature hash mismatch")
    for pair in PAIRS:
        item = bundle.get("pairs", {}).get(pair, {})
        if list(item.get("features", [])) != list(FEATURES[pair]):
            raise ValueError(f"{pair} feature order mismatch")
        if item.get("gate") != asdict(GATES[pair]):
            raise ValueError(f"{pair} gate mismatch")
        weeks = list(item.get("weeks", []))
        if not weeks:
            raise ValueError(f"{pair} has no weekly models")
        previous_end = None
        folds: set[int] = set()
        for week in weeks:
            start, end, cutoff = int(week["test_start"]), int(week["test_end"]), int(week["train_cutoff"])
            threshold = float(week["entry_threshold"])
            fold = int(week["fold"])
            if (cutoff > start or start - cutoff > 48 * 3600 or end <= start
                    or not 0 <= threshold <= 1 or not math.isfinite(threshold)):
                raise ValueError(f"{pair} invalid weekly boundary/threshold")
            if previous_end is not None and start != previous_end:
                raise ValueError(f"{pair} weekly manifest is not contiguous")
            if fold in folds or "model" not in week:
                raise ValueError(f"{pair} duplicate fold or missing model")
            previous_end = end
            folds.add(fold)


def week_for_timestamp(pair_bundle: Mapping[str, Any], signal_ts: int) -> Mapping[str, Any]:
    matches = [week for week in pair_bundle["weeks"]
               if int(week["test_start"]) <= int(signal_ts) < int(week["test_end"])]
    if len(matches) != 1:
        raise RuntimeError(f"no unique signed weekly model covers signal_ts={signal_ts}")
    return matches[0]


def run_weekly_bundle_strategy(
    rows: pd.DataFrame, *, pair: str, pair_bundle: Mapping[str, Any],
    state: GateState | None = None,
) -> tuple[pd.DataFrame, GateState]:
    """Predict with fold-local models and advance one continuous gate state."""
    if pair not in PAIRS:
        raise ValueError(f"unsupported pair: {pair}")
    features = list(pair_bundle.get("features", []))
    if features != list(FEATURES[pair]):
        raise ValueError(f"{pair} feature order mismatch")
    required = features + ["signal_ts", "last_complete_4h_ts", "roc_48h_4h",
                           "sqzmom_pct_4h", "di_spread", "ema20_slope_atr_12h",
                           "below_ema20_ratio_72h"]
    missing = [column for column in required if column not in rows]
    if missing:
        raise ValueError(f"{pair} rows missing columns: {missing}")
    ordered = rows.sort_values("signal_ts").copy()
    if ordered[required].replace([np.inf, -np.inf], np.nan).isna().any().any():
        raise ValueError(f"{pair} rows contain missing/non-finite inputs")
    probabilities = np.empty(len(ordered), dtype=float)
    thresholds = np.empty(len(ordered), dtype=float)
    folds = np.empty(len(ordered), dtype=int)
    for week in pair_bundle["weeks"]:
        mask = ordered.signal_ts.between(int(week["test_start"]), int(week["test_end"]), inclusive="left")
        if not mask.any():
            continue
        probabilities[mask.to_numpy()] = week["model"].predict_proba(ordered.loc[mask, features])[:, 1]
        thresholds[mask.to_numpy()] = float(week["entry_threshold"])
        folds[mask.to_numpy()] = int(week["fold"])
    covered = np.array([any(int(w["test_start"]) <= int(ts) < int(w["test_end"])
                            for w in pair_bundle["weeks"]) for ts in ordered.signal_ts], dtype=bool)
    if not covered.all():
        missing_ts = int(ordered.loc[~covered, "signal_ts"].iloc[0])
        raise RuntimeError(f"no signed weekly model covers {pair} signal_ts={missing_ts}")
    if not np.isfinite(probabilities).all() or not ((probabilities >= 0) & (probabilities <= 1)).all():
        raise ValueError(f"{pair} weekly model returned invalid probability")
    current_state = state or GateState()
    snapshots: list[dict[str, Any]] = []
    config = GateConfig(**dict(pair_bundle["gate"]))
    for index, row in enumerate(ordered.itertuples(index=False)):
        current_state, result = advance_gate(
            pair=pair, probability=float(probabilities[index]), entry_threshold=float(thresholds[index]),
            signal_ts=int(row.signal_ts), last_complete_4h_ts=int(row.last_complete_4h_ts),
            structure=(row.roc_48h_4h, row.sqzmom_pct_4h, row.di_spread,
                       row.ema20_slope_atr_12h, row.below_ema20_ratio_72h),
            state=current_state, config=config, model_version=MODEL_VERSION,
        )
        snapshots.append({"signal_ts": int(row.signal_ts), "fold": int(folds[index]), **result})
    return pd.DataFrame(snapshots), current_state


def _finite_structure(values: Sequence[float]) -> tuple[float, float, float, float, float]:
    result = tuple(float(value) for value in values)
    if len(result) != 5 or not all(math.isfinite(value) for value in result):
        raise ValueError("directional structure must contain five finite values")
    return result  # type: ignore[return-value]


def entry_confirm(current: tuple[float, ...], previous: tuple[float, ...] | None) -> bool:
    del previous
    roc, sqz, di, slope, below = current
    votes = int(di < 0) + int(slope < 0) + int(below >= .50)
    return bool(roc < 0 and sqz < 0 and votes >= 2 and (below >= .55 or (di < 0 and slope < 0)))


def recovery_confirm(current: tuple[float, ...], previous: tuple[float, ...] | None) -> bool:
    if previous is None:
        return False
    roc, sqz, di, slope, below = current
    votes = int(di > 0) + int(slope >= 0) + int(below < .50)
    return bool(roc > previous[0] and sqz > previous[1] and votes >= 2)


def event_id(pair: str, signal_ts: int, transition: str, model_version: str = MODEL_VERSION) -> str:
    return hashlib.sha256(f"{CONTRACT_SCHEMA}|{model_version}|{pair}|{int(signal_ts)}|{transition}".encode()).hexdigest()


def snapshot(pair: str, probability: float, threshold: float, state: GateState,
             transition: str, entry_ok: bool, recovery_ok: bool,
             structure: tuple[float, ...] | None, config: GateConfig) -> dict[str, Any]:
    names = ("roc_48h_4h", "sqzmom_pct_4h", "di_spread", "ema20_slope_atr_12h", "below_ema20_ratio_72h")
    reasons = {"enter": "probability_armed_and_persistent_bearish_4h", "recover": "adaptive_structural_relief_confirmed",
               "duplicate": "duplicate_or_late_complete_hour_ignored", "hold": "long_risk_off_waiting_for_structural_relief", "clear": "long_risk_gate_clear"}
    return {
        "pair": pair, "probability": float(probability), "entry_threshold": float(threshold),
        "risk_off_active": bool(state.active), "recommended_buy_enabled": not bool(state.active), "buy_enabled": False,
        "above_entry_count": int(state.above_entry_count),
        "armed": bool(state.armed_until is not None and (state.last_signal_ts or 0) <= state.armed_until),
        "armed_until": state.armed_until, "risk_off_since": state.since, "cooldown_until": state.cooldown_until,
        "structure_recovery_count": int(state.recovery_count), "entry_structure_confirmed": bool(entry_ok),
        "recovery_structure_confirmed": bool(recovery_ok),
        "structure": dict(zip(names, structure)) if structure is not None else None,
        "confirmation_mode": config.confirmation_mode, "recovery_mode": config.recovery_mode,
        "recovery_required_4h_bars": config.recovery_4h_bars, "transition": transition,
        "reason": reasons[transition], "event_id": state.last_event_id,
    }


def advance_gate(*, pair: str, probability: float, entry_threshold: float, signal_ts: int,
                 last_complete_4h_ts: int, structure: Sequence[float], state: GateState,
                 config: GateConfig | None = None, model_version: str = MODEL_VERSION) -> tuple[GateState, dict[str, Any]]:
    config = config or GATES[pair]
    if pair not in PAIRS or not math.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError("pair/probability is invalid")
    if not math.isfinite(entry_threshold) or not 0 <= entry_threshold <= 1:
        raise ValueError("entry threshold is invalid")
    if state.last_signal_ts is not None and signal_ts <= state.last_signal_ts:
        return state, snapshot(pair, probability, entry_threshold, state, "duplicate", False, False, state.previous_structure, config)
    current = _finite_structure(structure); state.probability_history.append((int(signal_ts), float(probability))); state.probability_history = state.probability_history[-48:]
    state.above_entry_count = state.above_entry_count + 1 if probability >= entry_threshold else 0
    if not state.active and state.above_entry_count >= config.entry_bars:
        armed = signal_ts + config.arm_hours * HOUR; state.armed_until = max(int(state.armed_until or armed), armed)
    new_4h = state.last_complete_4h_ts != int(last_complete_4h_ts)
    entry_ok = new_4h and entry_confirm(current, state.previous_structure); recovery_ok = new_4h and recovery_confirm(current, state.previous_structure)
    transition = "hold" if state.active else "clear"; cooldown = int(state.cooldown_until or -1); armed_until = int(state.armed_until or -1)
    if not state.active and signal_ts >= cooldown and signal_ts <= armed_until and entry_ok:
        state.active = True; state.since = int(signal_ts); state.recovery_count = 0; transition = "enter"
    elif state.active and new_4h:
        state.recovery_count = state.recovery_count + 1 if recovery_ok else 0
        strong = (current[0] >= 0 or current[1] >= 0) and current[2] > 0 and current[3] >= 0 and current[4] < .50
        required = 2 if strong else config.recovery_4h_bars; age = signal_ts-int(state.since if state.since is not None else signal_ts)
        if age >= config.minimum_hours*HOUR and state.recovery_count >= required:
            state.active = False; state.since = None; state.cooldown_until = signal_ts+config.cooldown_hours*HOUR
            state.armed_until = None; state.above_entry_count = 0; transition = "recover"
    if new_4h:
        state.previous_structure = current; state.last_complete_4h_ts = int(last_complete_4h_ts)
        state.structure_history.append((int(last_complete_4h_ts), current)); state.structure_history = state.structure_history[-4:]
    state.last_signal_ts = int(signal_ts)
    if transition in {"enter", "recover"}:
        state.last_event_id = event_id(pair, signal_ts, transition, model_version)
    return state, snapshot(pair, probability, entry_threshold, state, transition, entry_ok, recovery_ok, current, config)


def state_to_dict(state: GateState) -> dict[str, Any]:
    value = asdict(state)
    if state.previous_structure is not None: value["previous_structure"] = list(state.previous_structure)
    value["probability_history"] = [[ts,p] for ts,p in state.probability_history]
    value["structure_history"] = [[ts,list(s)] for ts,s in state.structure_history]
    return value


def state_from_dict(value: Mapping[str, Any]) -> GateState:
    previous = value.get("previous_structure")
    probabilities = [(int(ts),float(p)) for ts,p in value.get("probability_history",[])]
    structures = [(int(ts),_finite_structure(s)) for ts,s in value.get("structure_history",[])]
    if any(not math.isfinite(p) or not 0 <= p <= 1 for _,p in probabilities): raise ValueError("state probability history is invalid")
    if any(a[0] >= b[0] for a,b in zip(probabilities,probabilities[1:])): raise ValueError("state probability history is not strictly ordered")
    if any(a[0] >= b[0] for a,b in zip(structures,structures[1:])): raise ValueError("state structure history is not strictly ordered")
    return GateState(active=bool(value.get("active",False)), since=value.get("since"), above_entry_count=int(value.get("above_entry_count",0)),
        armed_until=value.get("armed_until"), cooldown_until=value.get("cooldown_until"), recovery_count=int(value.get("recovery_count",0)),
        last_complete_4h_ts=value.get("last_complete_4h_ts"), previous_structure=_finite_structure(previous) if previous is not None else None,
        last_signal_ts=value.get("last_signal_ts"), last_event_id=value.get("last_event_id"), probability_history=probabilities[-48:], structure_history=structures[-4:])


def build_inference_panel(candles: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    return _build_inference_panel(candles, FEATURES)

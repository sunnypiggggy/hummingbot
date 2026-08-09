#!/usr/bin/env python3
"""XGBoost long-only BUY-gate state and fail-closed runtime contract.

The contract can only pause ordinary Grid BUY orders.  It deliberately has no
sell, Taker, flatten, or excess-inventory action.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA = "grid-xgboost-long-risk-gate-v1"
MODEL_VERSION = "xgboost-grid-long-risk-gate-v21-250d"
STALE_AFTER_SECONDS = 150
REQUIRED_PAIRS = ("BTC-FDUSD", "ETH-FDUSD")
REQUIRED_CHANNELS = ("long",)


def _utc(ts: int) -> str:
    return datetime.fromtimestamp(int(ts), tz=timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_utc(value: Any, field: str) -> datetime:
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError(f"{field} must be timezone-aware")
    return parsed.astimezone(timezone.utc)


def _event_id(model_version: str, pair: str, signal_ts: int, transition: str) -> str:
    raw = f"{SCHEMA}|{model_version}|{pair}|{int(signal_ts)}|{transition}".encode()
    return hashlib.sha256(raw).hexdigest()


def _valid_sha256(value: Any) -> bool:
    text = str(value)
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text.lower())


@dataclass(frozen=True)
class PairGateState:
    """Backward-compatible single-channel hysteresis state."""

    risk_off_active: bool = False
    risk_off_since: int | None = None
    consecutive_recovery_bars: int = 0


def advance_pair_gate(
    *, pair: str, probability: float, entry_threshold: float,
    recovery_threshold: float, signal_ts: int, previous: PairGateState,
    model_version: str, minimum_risk_off_seconds: int = 4 * 60 * 60,
    required_low_bars: int = 2,
) -> tuple[PairGateState, dict[str, Any]]:
    """Advance one channel on a newly completed 1h candle."""
    values = (probability, entry_threshold, recovery_threshold)
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("probability and thresholds must be finite")
    if not 0 <= probability <= 1 or not 0 <= recovery_threshold <= entry_threshold <= 1:
        raise ValueError("probability/thresholds must satisfy 0 <= recovery <= entry <= 1")
    if minimum_risk_off_seconds < 0 or required_low_bars < 1:
        raise ValueError("hysteresis settings are invalid")

    active = bool(previous.risk_off_active)
    since = previous.risk_off_since
    count = int(previous.consecutive_recovery_bars)
    transition = "hold" if active else "clear"
    reason = "risk_probability_below_entry"
    if not active and probability >= entry_threshold:
        active, since, count = True, int(signal_ts), 0
        transition, reason = "enter", "probability_at_or_above_entry_threshold"
    elif active:
        count = count + 1 if probability < recovery_threshold else 0
        age = int(signal_ts) - int(since if since is not None else signal_ts)
        if count >= required_low_bars and age >= minimum_risk_off_seconds:
            active, since, count = False, None, 0
            transition, reason = "recover", "low_closed_bars_after_minimum_pause"
        else:
            reason = (
                "waiting_for_minimum_pause" if count >= required_low_bars
                else "waiting_for_low_closed_bars"
            )
    state = PairGateState(active, since, count)
    return state, {
        "pair": pair,
        "probability": float(probability),
        "entry_threshold": float(entry_threshold),
        "recovery_threshold": float(recovery_threshold),
        "risk_off_active": active,
        "buy_enabled": not active,
        "consecutive_recovery_bars": count,
        "risk_off_since": _utc(since) if since is not None else None,
        "transition": transition,
        "reason": reason,
    }


def combine_pair_channels(
    *, pair: str, channels: Mapping[str, Mapping[str, Any]], signal_ts: int,
    model_version: str,
) -> dict[str, Any]:
    """Normalize the pair's single long-risk channel."""
    if set(channels) != set(REQUIRED_CHANNELS):
        raise ValueError(f"channels must be exactly {REQUIRED_CHANNELS}")
    normalized: dict[str, dict[str, Any]] = {}
    for name in REQUIRED_CHANNELS:
        raw = dict(channels[name])
        probability = float(raw["probability"])
        entry = float(raw["entry_threshold"])
        recovery = float(raw["recovery_threshold"])
        if not all(math.isfinite(value) for value in (probability, entry, recovery)):
            raise ValueError(f"{pair}/{name} probability and thresholds must be finite")
        if not 0 <= probability <= 1 or not 0 <= recovery <= entry <= 1:
            raise ValueError(f"{pair}/{name} probability or thresholds are invalid")
        raw.update({
            "probability": probability,
            "entry_threshold": entry,
            "recovery_threshold": recovery,
            "risk_off_active": bool(raw.get("risk_off_active")),
        })
        normalized[name] = raw
    active_channels = [name for name in REQUIRED_CHANNELS if normalized[name]["risk_off_active"]]
    transition = "+".join(
        f"{name}:{normalized[name].get('transition', 'hold')}" for name in REQUIRED_CHANNELS
    )
    return {
        "pair": pair,
        "channels": normalized,
        "active_channels": active_channels,
        "risk_off_active": bool(active_channels),
        "buy_enabled": not bool(active_channels),
        "transition": transition,
        "reason": "long_risk_off" if active_channels else "long_channel_clear",
        "event_id": _event_id(model_version, pair, signal_ts, transition),
    }


def fail_closed_pair(
    *, pair: str, signal_ts: int, model_version: str, reason: str,
    channels: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Represent unhealthy input without creating a market action."""
    return {
        "pair": pair,
        "channels": dict(channels or {}),
        "active_channels": [],
        "risk_off_active": True,
        "buy_enabled": False,
        "transition": "fail_closed",
        "reason": reason,
        "event_id": _event_id(model_version, pair, signal_ts, "fail_closed"),
    }


def feature_schema_hash(features: Sequence[str]) -> str:
    payload = json.dumps(list(features), ensure_ascii=False, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def build_contract(
    *, generated_at: int, valid_until: int, model_version: str,
    model_sha256: str, feature_sha256: str, data_sha256: str,
    source_healthy: bool, deployment_allowed: bool,
    pair_signals: Mapping[str, Mapping[str, Any]],
    last_complete_1h: Mapping[str, int], last_complete_4h: Mapping[str, int],
) -> dict[str, Any]:
    pairs: dict[str, dict[str, Any]] = {}
    for pair in REQUIRED_PAIRS:
        if pair not in pair_signals:
            raise ValueError(f"missing pair signal: {pair}")
        signal = dict(pair_signals[pair])
        if not source_healthy or not deployment_allowed:
            signal = fail_closed_pair(
                pair=pair, signal_ts=generated_at, model_version=model_version,
                reason="source_unhealthy" if not source_healthy else "model_not_deployment_allowed",
                channels=signal.get("channels"),
            )
        signal["last_complete_1h"] = _utc(last_complete_1h[pair])
        signal["last_complete_4h"] = _utc(last_complete_4h[pair])
        pairs[pair] = signal
    return {
        "schema": SCHEMA,
        "generated_at": _utc(generated_at),
        "valid_until": _utc(valid_until),
        "model_version": model_version,
        "model_sha256": str(model_sha256),
        "feature_schema_sha256": str(feature_sha256),
        "data_sha256": str(data_sha256),
        "source_healthy": bool(source_healthy),
        "stale_after_seconds": STALE_AFTER_SECONDS,
        "deployment_allowed": bool(deployment_allowed),
        "shadow_mode": True,
        "short_spike_enabled": False,
        "market_sell_action": False,
        "mechanism1_fallback_allowed": False,
        "pairs": pairs,
    }


def _failed_runtime(reason: str, observed: datetime) -> dict[str, Any]:
    now_ts = int(observed.timestamp())
    return {
        "schema": SCHEMA,
        "generated_at": _utc(now_ts),
        "source_healthy": False,
        "deployment_allowed": False,
        "shadow_mode": True,
        "short_spike_enabled": False,
        "market_sell_action": False,
        "mechanism1_fallback_allowed": False,
        "runtime_gate_healthy": False,
        "runtime_age_seconds": None,
        "reason": f"fail_closed:{reason}",
        "pairs": {
            pair: fail_closed_pair(
                pair=pair, signal_ts=now_ts, model_version=MODEL_VERSION,
                reason=f"fail_closed:{reason}",
            ) for pair in REQUIRED_PAIRS
        },
    }


def load_runtime_xgboost_gate(
    path: Path, *, now: datetime | None = None,
    max_age_seconds: int = STALE_AFTER_SECONDS,
    expected_model_sha256: str | None = None,
    expected_feature_sha256: str | None = None,
) -> dict[str, Any]:
    """Load and strictly validate the only supported live technical BUY gate."""
    observed = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    try:
        schema = json.loads(path.read_text(encoding="utf-8")).get("schema")
    except Exception:
        schema = None
    if schema == "ethbtc-forced-exit-live-contract-v1":
        # Transitional reader: observation keeps the existing v21 contract in
        # force; after the atomic activation boundary this same file becomes
        # the sole v22 forced-exit contract. There is no fallback once switched.
        try:
            # Hummingbot loads strategy helpers from the ``scripts`` package.
            from scripts.ethbtc_forced_exit_contract import load_runtime_contract
        except ImportError:
            # Guard/scheduler images copy helpers directly into /app.
            from ethbtc_forced_exit_contract import load_runtime_contract
        return load_runtime_contract(path, now=observed, max_age_seconds=max_age_seconds)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema") != SCHEMA:
            raise ValueError("unsupported XGBoost risk-gate schema")
        if payload.get("model_version") != MODEL_VERSION:
            raise ValueError("unsupported XGBoost risk-gate model version")
        if payload.get("short_spike_enabled") is not False:
            raise ValueError("short-spike channel is forbidden")
        if payload.get("market_sell_action") is not False:
            raise ValueError("risk gate cannot request a market sell")
        if payload.get("shadow_mode") is not False:
            raise ValueError("live XGBoost gate must not be a shadow contract")
        if payload.get("mechanism1_fallback_allowed") is not False:
            raise ValueError("Mechanism 1 fallback is forbidden")
        for field in ("model_sha256", "feature_schema_sha256", "data_sha256"):
            if not _valid_sha256(payload.get(field)):
                raise ValueError(f"{field} is not a SHA-256 digest")
        generated = _parse_utc(payload["generated_at"], "generated_at")
        valid_until = _parse_utc(payload["valid_until"], "valid_until")
        age = (observed - generated).total_seconds()
        if age < -10:
            raise ValueError("XGBoost gate timestamp is in the future")
        if age > max_age_seconds or observed > valid_until:
            raise ValueError(f"XGBoost risk gate is stale by {max(0, age):.0f} seconds")
        if not bool(payload.get("source_healthy")):
            raise ValueError("XGBoost signal source is unhealthy")
        if not bool(payload.get("deployment_allowed")):
            raise ValueError("locked XGBoost model is not deployment allowed")
        if expected_model_sha256 and payload.get("model_sha256") != expected_model_sha256:
            raise ValueError("XGBoost model hash mismatch")
        if expected_feature_sha256 and payload.get("feature_schema_sha256") != expected_feature_sha256:
            raise ValueError("XGBoost feature schema hash mismatch")
        raw_pairs = payload.get("pairs")
        if not isinstance(raw_pairs, Mapping) or set(raw_pairs) != set(REQUIRED_PAIRS):
            raise ValueError("XGBoost gate must contain exactly BTC-FDUSD and ETH-FDUSD")
        pairs: dict[str, dict[str, Any]] = {}
        for pair in REQUIRED_PAIRS:
            raw = dict(raw_pairs[pair])
            normalized = combine_pair_channels(
                pair=pair, channels=raw["channels"], signal_ts=int(generated.timestamp()),
                model_version=str(payload["model_version"]),
            )
            if bool(raw.get("buy_enabled")) != bool(normalized["buy_enabled"]):
                raise ValueError(f"{pair} combined buy_enabled is inconsistent")
            if bool(raw.get("risk_off_active")) != bool(normalized["risk_off_active"]):
                raise ValueError(f"{pair} combined risk_off_active is inconsistent")
            completed_1h = _parse_utc(raw["last_complete_1h"], f"{pair}.last_complete_1h")
            completed_4h = _parse_utc(raw["last_complete_4h"], f"{pair}.last_complete_4h")
            if completed_1h > generated or completed_4h > generated:
                raise ValueError(f"{pair} contains an incomplete future candle")
            if (observed - completed_1h).total_seconds() > 5400:
                raise ValueError(f"{pair} latest complete 1h candle is stale")
            if (observed - completed_4h).total_seconds() > 5 * 3600:
                raise ValueError(f"{pair} latest complete 4h candle is stale")
            normalized.update({
                key: raw[key] for key in ("last_complete_1h", "last_complete_4h") if key in raw
            })
            pairs[pair] = normalized
        return {
            **payload,
            "pairs": pairs,
            "runtime_gate_healthy": True,
            "runtime_age_seconds": max(0, int(age)),
            "reason": "xgboost_gate_healthy",
        }
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        return _failed_runtime(str(exc), observed)


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}-", suffix=".tmp", text=True
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(dict(payload), stream, indent=2, ensure_ascii=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def state_to_dict(state: PairGateState) -> dict[str, Any]:
    return asdict(state)


def state_from_dict(value: Mapping[str, Any]) -> PairGateState:
    return PairGateState(
        risk_off_active=bool(value.get("risk_off_active", False)),
        risk_off_since=(
            int(value["risk_off_since"]) if value.get("risk_off_since") is not None else None
        ),
        consecutive_recovery_bars=int(value.get("consecutive_recovery_bars", 0)),
    )


# Compatibility constants used by the existing v1 tests and callers.
MIN_RISK_OFF_SECONDS = 4 * 60 * 60
REQUIRED_LOW_BARS = 2

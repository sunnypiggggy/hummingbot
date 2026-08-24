#!/usr/bin/env python3
"""Research-only contract helpers for the ML momentum stop.

This module deliberately has no dependency on the live grid runtime.  It turns
one closed-bar probability per pair into the proposed
``grid-ml-momentum-stop-v1`` JSON payload and implements its fail-closed data
health behavior.
"""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, Mapping


SCHEMA = "grid-ml-momentum-stop-v1"
PAIRS = ("BTC-FDUSD", "ETH-FDUSD")


def _utc_iso(timestamp: int) -> str:
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def stable_event_id(
    model_version: str, pair: str, signal_ts: int, transition: str
) -> str:
    """Return an idempotent identifier that is stable across file rewrites."""
    raw = f"{SCHEMA}|{model_version}|{pair}|{signal_ts}|{transition}".encode()
    return hashlib.sha256(raw).hexdigest()


def feature_schema_hash(features: list[str] | tuple[str, ...]) -> str:
    canonical = json.dumps(list(features), ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def advance_pair_state(
    *,
    pair: str,
    probability: float,
    entry_threshold: float,
    previous_risk_off: bool,
    recovery_condition_met: bool,
    signal_ts: int,
    last_complete_1h_ts: int,
    last_complete_4h_ts: int,
    model_version: str,
    recovery_details: Mapping[str, Any],
) -> dict[str, Any]:
    """Advance one pair without allowing either pair to affect the other."""
    if pair not in PAIRS:
        raise ValueError(f"Unsupported pair: {pair}")
    probability = float(probability)
    entry_threshold = float(entry_threshold)
    if not 0 <= probability <= 1 or not 0 <= entry_threshold <= 1:
        raise ValueError("probability and entry_threshold must be finite values in [0, 1]")

    entered = not previous_risk_off and probability >= entry_threshold
    recovered = bool(
        previous_risk_off
        and probability < entry_threshold
        and recovery_condition_met
    )
    risk_off = bool(entered or (previous_risk_off and not recovered))
    if entered:
        transition = "enter"
        reason = "model_probability_entered_risk_off"
    elif recovered:
        transition = "recover"
        reason = "probability_low_and_pair_momentum_recovered"
    elif risk_off:
        transition = "hold"
        reason = "risk_off_waiting_for_probability_and_pair_recovery"
    else:
        transition = "clear"
        reason = "model_probability_below_entry_threshold"

    return {
        "probability": probability,
        "entry_threshold": entry_threshold,
        "risk_off_active": risk_off,
        "buy_enabled": not risk_off,
        # Only the inactive -> active edge may cause a market action.
        "stop_excess_inventory": bool(entered),
        "last_complete_1h": _utc_iso(int(last_complete_1h_ts)),
        "last_complete_4h": _utc_iso(int(last_complete_4h_ts)),
        "recovery": {
            "probability_below_threshold": probability < entry_threshold,
            "pair_momentum_condition_met": bool(recovery_condition_met),
            **dict(recovery_details),
        },
        "reason": reason,
        "event_id": stable_event_id(model_version, pair, int(signal_ts), transition),
    }


def build_contract(
    *,
    pair_signals: Mapping[str, Mapping[str, Any]],
    generated_at: int,
    valid_until: int,
    model_version: str,
    model_sha256: str,
    feature_schema_sha256: str,
    source_healthy: bool = True,
) -> dict[str, Any]:
    if set(pair_signals) != set(PAIRS):
        raise ValueError(f"pair_signals must contain exactly {PAIRS}")
    if int(valid_until) <= int(generated_at):
        raise ValueError("valid_until must be after generated_at")
    payload = {
        "schema": SCHEMA,
        "generated_at": _utc_iso(int(generated_at)),
        "valid_until": _utc_iso(int(valid_until)),
        "model_version": str(model_version),
        "model_sha256": str(model_sha256),
        "feature_schema_sha256": str(feature_schema_sha256),
        "source_healthy": bool(source_healthy),
        "deployment_allowed": False,
        "pairs": {pair: dict(pair_signals[pair]) for pair in PAIRS},
    }
    if not source_healthy:
        for signal in payload["pairs"].values():
            signal["buy_enabled"] = False
            signal["stop_excess_inventory"] = False
            signal["reason"] = (
                "source_unhealthy_fail_closed_without_market_sell:"
                f"{signal.get('reason', 'unknown')}"
            )
    validate_contract(payload)
    return payload


def failed_contract(
    *,
    generated_at: int,
    valid_until: int,
    model_version: str,
    model_sha256: str,
    feature_schema_sha256: str,
    reason: str,
) -> dict[str, Any]:
    signals = {}
    for pair in PAIRS:
        signals[pair] = {
            "probability": None,
            "entry_threshold": None,
            "risk_off_active": True,
            "buy_enabled": False,
            "stop_excess_inventory": False,
            "last_complete_1h": None,
            "last_complete_4h": None,
            "recovery": {"probability_below_threshold": False, "pair_momentum_condition_met": False},
            "reason": f"source_unhealthy:{reason}",
            "event_id": stable_event_id(model_version, pair, int(generated_at), "fail_closed"),
        }
    return build_contract(
        pair_signals=signals,
        generated_at=generated_at,
        valid_until=valid_until,
        model_version=model_version,
        model_sha256=model_sha256,
        feature_schema_sha256=feature_schema_sha256,
        source_healthy=False,
    )


def validate_contract(payload: Mapping[str, Any]) -> None:
    required = {
        "schema", "generated_at", "valid_until", "model_version", "model_sha256",
        "feature_schema_sha256", "source_healthy", "deployment_allowed", "pairs",
    }
    missing = required.difference(payload)
    if missing:
        raise ValueError(f"Signal contract is missing: {sorted(missing)}")
    if payload["schema"] != SCHEMA or payload["deployment_allowed"] is not False:
        raise ValueError("Research schema and deployment_allowed=false are mandatory")
    if set(payload["pairs"]) != set(PAIRS):
        raise ValueError(f"Signal contract must contain exactly {PAIRS}")
    pair_required = {
        "probability", "entry_threshold", "risk_off_active", "buy_enabled",
        "stop_excess_inventory", "last_complete_1h", "last_complete_4h",
        "recovery", "reason", "event_id",
    }
    for pair, signal in payload["pairs"].items():
        pair_missing = pair_required.difference(signal)
        if pair_missing:
            raise ValueError(f"{pair} signal is missing: {sorted(pair_missing)}")
        if not payload["source_healthy"] and (
            signal["buy_enabled"] or signal["stop_excess_inventory"]
        ):
            raise ValueError("Unhealthy sources must disable BUY without causing a market sell")


def enforce_freshness(payload: Mapping[str, Any], now_ts: int) -> dict[str, Any]:
    """Fail closed when a previously healthy signal file has expired."""
    validate_contract(payload)
    valid_until = int(
        datetime.fromisoformat(str(payload["valid_until"]).replace("Z", "+00:00")).timestamp()
    )
    if int(now_ts) <= valid_until and bool(payload["source_healthy"]):
        return deepcopy(dict(payload))
    return failed_contract(
        generated_at=int(now_ts), valid_until=int(now_ts) + 150,
        model_version=str(payload["model_version"]),
        model_sha256=str(payload["model_sha256"]),
        feature_schema_sha256=str(payload["feature_schema_sha256"]),
        reason="signal_file_expired" if int(now_ts) > valid_until else "source_unhealthy",
    )

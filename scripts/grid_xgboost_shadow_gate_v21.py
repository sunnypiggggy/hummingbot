#!/usr/bin/env python3
"""Strict, non-authorizing v21 shadow signal contract."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from xgboost_long_risk_gate_v21 import CONTRACT_SCHEMA, MODEL_VERSION, PAIRS, STALE_AFTER_SECONDS


def utc(ts: int) -> str:
    return datetime.fromtimestamp(int(ts), timezone.utc).isoformat().replace("+00:00", "Z")


def parse_utc(value: Any) -> datetime:
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("timestamp must be timezone-aware")
    return parsed.astimezone(timezone.utc)


def valid_sha256(value: Any) -> bool:
    text = str(value)
    return len(text) == 64 and all(char in "0123456789abcdef" for char in text.lower())


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}-", suffix=".tmp", text=True)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(dict(payload), stream, ensure_ascii=False, indent=2)
            stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)


def contract_event_id(pair: str, signal_ts: int, transition: str) -> str:
    return hashlib.sha256(f"{CONTRACT_SCHEMA}|{MODEL_VERSION}|{pair}|{signal_ts}|{transition}".encode()).hexdigest()


def build_contract(*, generated_at: int, model_sha256: str, feature_sha256: str,
                   strategy_sha256: str, training_data_sha256: str, candidate_lock_sha256: str,
                   state_sha256: str, source_healthy: bool,
                   pair_snapshots: Mapping[str, Mapping[str, Any]],
                   last_complete_1h: Mapping[str, int], last_complete_4h: Mapping[str, int],
                   reason: str = "shadow_signal_healthy") -> dict[str, Any]:
    pairs = {}
    for pair in PAIRS:
        raw = dict(pair_snapshots[pair])
        if not source_healthy:
            raw.update({"risk_off_active": True, "recommended_buy_enabled": False,
                        "transition": "fail_closed", "reason": reason,
                        "event_id": contract_event_id(pair, generated_at, "fail_closed")})
        raw.update({"buy_enabled": False, "last_complete_1h": utc(last_complete_1h[pair]),
                    "last_complete_4h": utc(last_complete_4h[pair])})
        pairs[pair] = {
            "long": raw,
            "active_channels": ["long"] if raw["risk_off_active"] else [],
            "risk_off_active": bool(raw["risk_off_active"]),
            "recommended_buy_enabled": bool(raw["recommended_buy_enabled"]),
            "buy_enabled": False,
            "reason": raw.get("reason", reason),
            "event_id": raw["event_id"],
        }
    return {
        "schema": CONTRACT_SCHEMA, "generated_at": utc(generated_at),
        "valid_until": utc(generated_at + STALE_AFTER_SECONDS), "stale_after_seconds": STALE_AFTER_SECONDS,
        "model_version": MODEL_VERSION, "model_sha256": model_sha256,
        "feature_schema_sha256": feature_sha256, "strategy_schema_sha256": strategy_sha256,
        "training_data_sha256": training_data_sha256,
        "candidate_lock_sha256": candidate_lock_sha256, "state_sha256": state_sha256,
        "source_healthy": bool(source_healthy), "shadow_mode": True,
        "deployment_allowed": False, "promotion_authorized": False,
        "short_spike_enabled": False, "market_sell_action": False,
        "mechanism1_fallback_allowed": False, "runtime_action": "observe_only",
        "reason": reason, "pairs": pairs,
    }


def failed_contract(reason: str, generated_at: int) -> dict[str, Any]:
    zero = "0" * 64
    snapshots = {pair: {"probability": 0.0, "entry_threshold": 1.0,
                        "risk_off_active": True, "recommended_buy_enabled": False,
                        "transition": "fail_closed", "reason": reason,
                        "event_id": contract_event_id(pair, generated_at, "fail_closed")}
                 for pair in PAIRS}
    return build_contract(generated_at=generated_at, model_sha256=zero, feature_sha256=zero,
        strategy_sha256=zero,
        training_data_sha256=zero, candidate_lock_sha256=zero, state_sha256=zero,
        source_healthy=False, pair_snapshots=snapshots,
        last_complete_1h={pair: generated_at for pair in PAIRS},
        last_complete_4h={pair: generated_at for pair in PAIRS}, reason=f"fail_closed:{reason}")


def load_shadow_contract(path: Path, *, now: datetime | None = None) -> dict[str, Any]:
    observed = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != CONTRACT_SCHEMA or payload.get("model_version") != MODEL_VERSION:
        raise ValueError("unsupported v21 shadow schema/model")
    for field in ("model_sha256", "feature_schema_sha256", "strategy_schema_sha256", "training_data_sha256",
                  "candidate_lock_sha256", "state_sha256"):
        if not valid_sha256(payload.get(field)): raise ValueError(f"invalid {field}")
    if payload.get("shadow_mode") is not True or payload.get("deployment_allowed") is not False:
        raise ValueError("v21 contract must remain non-authorizing shadow output")
    if payload.get("promotion_authorized") is not False or payload.get("short_spike_enabled") is not False:
        raise ValueError("promotion/short channel is forbidden")
    if payload.get("market_sell_action") is not False or payload.get("mechanism1_fallback_allowed") is not False:
        raise ValueError("sell/fallback action is forbidden")
    generated, valid_until = parse_utc(payload["generated_at"]), parse_utc(payload["valid_until"])
    age = (observed - generated).total_seconds()
    if age < -10 or age > STALE_AFTER_SECONDS or observed > valid_until:
        raise ValueError("v21 shadow heartbeat is stale or in the future")
    if set(payload.get("pairs", {})) != set(PAIRS): raise ValueError("pair set mismatch")
    for pair in PAIRS:
        pair_signal = payload["pairs"][pair]
        if set(pair_signal) & {"short", "short_spike"}:
            raise ValueError("short channel is forbidden")
        channel = pair_signal.get("long", {})
        if channel.get("buy_enabled") is not False: raise ValueError("shadow output may never enable BUY")
        if not 0 <= float(channel["probability"]) <= 1: raise ValueError("invalid probability")
        if pair_signal.get("buy_enabled") is not False: raise ValueError("pair output may never enable BUY")
        if bool(pair_signal.get("risk_off_active")) != bool(channel.get("risk_off_active")):
            raise ValueError("pair/long state mismatch")
        if bool(pair_signal.get("recommended_buy_enabled")) != bool(channel.get("recommended_buy_enabled")):
            raise ValueError("pair/long recommendation mismatch")
    return {**payload, "shadow_contract_healthy": bool(payload.get("source_healthy")),
            "runtime_age_seconds": max(0, int(age))}

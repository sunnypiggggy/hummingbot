#!/usr/bin/env python3
"""Strict non-authorizing contract for the v22 weekly Risk-off gate."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from xgboost_long_risk_gate_v22 import CONTRACT_SCHEMA, MODEL_VERSION, PAIRS, STALE_AFTER_SECONDS


def utc(ts: int) -> str:
    return datetime.fromtimestamp(int(ts), timezone.utc).isoformat().replace("+00:00", "Z")


def valid_sha256(value: Any) -> bool:
    text = str(value); return len(text) == 64 and all(c in "0123456789abcdef" for c in text.lower())


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}-", suffix=".tmp", text=True)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(dict(payload), stream, ensure_ascii=False, indent=2); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)


def event_id(pair: str, ts: int, transition: str) -> str:
    return hashlib.sha256(f"{CONTRACT_SCHEMA}|{MODEL_VERSION}|{pair}|{ts}|{transition}".encode()).hexdigest()


def build_contract(*, generated_at: int, hashes: Mapping[str, str], source_healthy: bool,
                   snapshots: Mapping[str, Mapping[str, Any]], last_1h: Mapping[str, int],
                   last_4h: Mapping[str, int], reason: str) -> dict[str, Any]:
    pairs = {}
    for pair in PAIRS:
        long = dict(snapshots[pair])
        if not source_healthy:
            long.update({"risk_off_active": True, "recommended_buy_enabled": False,
                         "buy_enabled": False, "transition": "fail_closed", "reason": reason,
                         "event_id": event_id(pair, generated_at, "fail_closed")})
        long.update({"buy_enabled": False, "last_complete_1h": utc(last_1h[pair]),
                     "last_complete_4h": utc(last_4h[pair])})
        pairs[pair] = {"long": long, "risk_off_active": bool(long["risk_off_active"]),
                       "recommended_buy_enabled": bool(long["recommended_buy_enabled"]),
                       "buy_enabled": False, "active_channels": ["long"] if long["risk_off_active"] else [],
                       "reason": long.get("reason", reason), "event_id": long["event_id"]}
    return {"schema": CONTRACT_SCHEMA, "model_version": MODEL_VERSION, "generated_at": utc(generated_at),
        "valid_until": utc(generated_at + STALE_AFTER_SECONDS), "stale_after_seconds": STALE_AFTER_SECONDS,
        **dict(hashes), "source_healthy": bool(source_healthy), "shadow_mode": True,
        "deployment_allowed": False, "promotion_authorized": False, "short_spike_enabled": False,
        "market_sell_action": False, "mechanism1_fallback_allowed": False,
        "probability_semantics": "weekly_walk_forward_model_with_fold_local_calibration_threshold",
        "runtime_action": "observe_only", "reason": reason, "pairs": pairs}


def failed_contract(reason: str, generated_at: int) -> dict[str, Any]:
    zero = "0" * 64
    snapshots = {pair: {"probability": 0.0, "entry_threshold": 1.0, "fold": None,
        "risk_off_active": True, "recommended_buy_enabled": False, "buy_enabled": False,
        "transition": "fail_closed", "reason": reason, "event_id": event_id(pair, generated_at, "fail_closed")}
        for pair in PAIRS}
    fields = ("model_sha256", "feature_schema_sha256", "strategy_schema_sha256",
              "training_data_sha256", "candidate_lock_sha256", "state_sha256")
    return build_contract(generated_at=generated_at, hashes={field: zero for field in fields},
        source_healthy=False, snapshots=snapshots, last_1h={p: generated_at for p in PAIRS},
        last_4h={p: generated_at for p in PAIRS}, reason=f"fail_closed:{reason}")


def load_shadow_contract(path: Path, *, now: datetime | None = None) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8")); observed = now or datetime.now(timezone.utc)
    if payload.get("schema") != CONTRACT_SCHEMA or payload.get("model_version") != MODEL_VERSION:
        raise ValueError("unsupported v22 contract")
    for field in ("model_sha256", "feature_schema_sha256", "strategy_schema_sha256",
                  "training_data_sha256", "candidate_lock_sha256", "state_sha256"):
        if not valid_sha256(payload.get(field)): raise ValueError(f"invalid {field}")
    generated = datetime.fromisoformat(payload["generated_at"].replace("Z", "+00:00"))
    valid_until = datetime.fromisoformat(payload["valid_until"].replace("Z", "+00:00"))
    age = (observed.astimezone(timezone.utc) - generated).total_seconds()
    if age < -10 or age > STALE_AFTER_SECONDS or observed > valid_until: raise ValueError("v22 heartbeat stale")
    if payload.get("deployment_allowed") is not False or payload.get("promotion_authorized") is not False:
        raise ValueError("v22 shadow contract cannot authorize deployment")
    if payload.get("market_sell_action") is not False or payload.get("mechanism1_fallback_allowed") is not False:
        raise ValueError("v22 sell/fallback forbidden")
    if set(payload.get("pairs", {})) != set(PAIRS): raise ValueError("pair set mismatch")
    for pair in PAIRS:
        item = payload["pairs"][pair]; long = item.get("long", {})
        if set(item) & {"short", "short_spike"}: raise ValueError("short channel forbidden")
        if long.get("buy_enabled") is not False or item.get("buy_enabled") is not False:
            raise ValueError("shadow output cannot enable BUY")
        if not 0 <= float(long["probability"]) <= 1: raise ValueError("invalid probability")
    return {**payload, "shadow_contract_healthy": bool(payload.get("source_healthy")),
            "runtime_age_seconds": max(0, int(age))}


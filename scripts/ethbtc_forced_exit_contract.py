"""Strict live contract for the ethbtc-forced-exit v22 execution overlay."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


SCHEMA = "ethbtc-forced-exit-live-contract-v1"
PACKAGE_ID = "ethbtc-forced-exit"
EXECUTION_POLICY_VERSION = "v22-risk-off-forced-exit-v2"
MODEL_VERSION = "xgboost-grid-long-risk-gate-v22-weekly-250d"
STALE_AFTER_SECONDS = 150
REQUIRED_PAIRS = ("BTC-FDUSD", "ETH-FDUSD")
HASH_FIELDS = (
    "release_sha256", "model_sha256", "feature_schema_sha256",
    "strategy_schema_sha256", "training_data_sha256",
)


def utc(ts: int | float) -> str:
    return datetime.fromtimestamp(int(ts), timezone.utc).isoformat().replace("+00:00", "Z")


def parse_utc(value: Any, field: str) -> datetime:
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError(f"{field} must be timezone-aware")
    return parsed.astimezone(timezone.utc)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def valid_sha256(value: Any) -> bool:
    text = str(value)
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text.lower())


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}-", suffix=".tmp", text=True,
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(dict(payload), stream, ensure_ascii=False, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def event_id(release_sha256: str, pair: str, signal_ts: int, transition: str) -> str:
    value = f"{SCHEMA}|{release_sha256}|{pair}|{signal_ts}|{transition}"
    return hashlib.sha256(value.encode()).hexdigest()


def failed_contract(*, generated_at: int, reason: str,
                    metadata: Mapping[str, Any] | None = None) -> dict[str, Any]:
    meta = dict(metadata or {})
    hashes = {field: str(meta.get(field, "0" * 64)) for field in HASH_FIELDS}
    pairs = {
        pair: {
            "pair": pair, "source_pair": pair, "signal_ts": generated_at,
            "model_week": None, "week_start": None, "week_end": None,
            "probability": None, "entry_threshold": None,
            "risk_off_active": True, "recommended_buy_enabled": False,
            "buy_enabled": False, "force_exit": True,
            "transition": "fail_closed", "reason": reason,
            "event_id": event_id(hashes["release_sha256"], pair, generated_at, "fail_closed"),
        }
        for pair in REQUIRED_PAIRS
    }
    return {
        "schema": SCHEMA, "package_id": PACKAGE_ID,
        "execution_policy_version": EXECUTION_POLICY_VERSION,
        "model_version": MODEL_VERSION, "generated_at": utc(generated_at),
        "valid_until": utc(generated_at + STALE_AFTER_SECONDS),
        "stale_after_seconds": STALE_AFTER_SECONDS,
        **hashes, "source_healthy": False, "execution_authorized": False,
        "observation_mode": True, "activation_at": None,
        "approval_receipt_sha256": None, "deployment_allowed": False,
        "promotion_authorized": False, "market_sell_action": True,
        "previous_model_fallback_allowed": False,
        "runtime_action": "fail_closed", "reason": reason, "pairs": pairs,
    }


def load_runtime_contract(path: Path, *, now: datetime | None = None,
                          max_age_seconds: int = STALE_AFTER_SECONDS,
                          expected_release_sha256: str | None = None) -> dict[str, Any]:
    observed = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema") != SCHEMA or payload.get("package_id") != PACKAGE_ID:
            raise ValueError("unsupported ethbtc-forced-exit contract")
        if payload.get("execution_policy_version") != EXECUTION_POLICY_VERSION:
            raise ValueError("forced-exit execution policy mismatch")
        if payload.get("model_version") != MODEL_VERSION:
            raise ValueError("v22 model version mismatch")
        if payload.get("market_sell_action") is not True:
            raise ValueError("forced-exit contract must declare market sell authority")
        if payload.get("previous_model_fallback_allowed") is not False:
            raise ValueError("previous model fallback is forbidden")
        for field in HASH_FIELDS:
            if not valid_sha256(payload.get(field)):
                raise ValueError(f"invalid {field}")
        if expected_release_sha256 and payload["release_sha256"] != expected_release_sha256:
            raise ValueError("release hash mismatch")
        generated = parse_utc(payload["generated_at"], "generated_at")
        valid_until = parse_utc(payload["valid_until"], "valid_until")
        age = (observed - generated).total_seconds()
        if age < -10 or age > max_age_seconds or observed > valid_until:
            raise ValueError("ethbtc-forced-exit contract is stale")
        raw_pairs = payload.get("pairs")
        if not isinstance(raw_pairs, Mapping) or set(raw_pairs) != set(REQUIRED_PAIRS):
            raise ValueError("contract must contain exactly BTC-FDUSD and ETH-FDUSD")
        pairs: dict[str, dict[str, Any]] = {}
        for pair in REQUIRED_PAIRS:
            item = dict(raw_pairs[pair])
            if item.get("pair") != pair or item.get("source_pair") != pair:
                raise ValueError(f"{pair} contract mapping mismatch")
            if not valid_sha256(item.get("event_id")):
                raise ValueError(f"{pair} event id is invalid")
            probability = float(item["probability"])
            threshold = float(item["entry_threshold"])
            if not 0 <= probability <= 1 or not 0 <= threshold <= 1:
                raise ValueError(f"{pair} probability or threshold is invalid")
            week_start = int(item["week_start"])
            week_end = int(item["week_end"])
            signal_ts = int(item["signal_ts"])
            if not week_start <= signal_ts < week_end:
                raise ValueError(f"{pair} signal is outside its signed week")
            if int(observed.timestamp()) >= week_end:
                raise ValueError(f"{pair} signed week has expired")
            risk_off = bool(item["risk_off_active"])
            recommended = bool(item["recommended_buy_enabled"])
            if recommended == risk_off:
                raise ValueError(f"{pair} risk/recommendation contradiction")
            authorized = bool(payload.get("execution_authorized"))
            if bool(item.get("buy_enabled")) != bool(authorized and recommended):
                raise ValueError(f"{pair} effective BUY state mismatch")
            if bool(item.get("force_exit")) != bool(authorized and risk_off):
                raise ValueError(f"{pair} force-exit state mismatch")
            pairs[pair] = item
        source_healthy = bool(payload.get("source_healthy"))
        return {
            **payload, "pairs": pairs,
            "runtime_gate_healthy": source_healthy,
            "runtime_age_seconds": max(0, int(age)),
            "reason": payload.get("reason", "healthy" if source_healthy else "source_unhealthy"),
        }
    except Exception as exc:
        return {
            **failed_contract(generated_at=int(observed.timestamp()), reason=f"fail_closed:{exc}"),
            "runtime_gate_healthy": False, "runtime_age_seconds": None,
        }

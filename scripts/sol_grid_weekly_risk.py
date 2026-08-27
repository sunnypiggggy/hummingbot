"""Isolated live contract for the SOL-FDUSD weekly Grid BUY gate.

This contract deliberately has a different schema, model identity and state
lineage from the immutable BTC/ETH v22 package.  A SOL failure can therefore
fail closed only SOL without contaminating BTC/ETH or DCA consumers.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


PAIR = "SOL-FDUSD"
MODEL_VERSION = "sol-grid-weekly-risk-v1"
CONTRACT_SCHEMA = "sol-grid-weekly-risk-live-contract-v1"
STALE_AFTER_SECONDS = 150
FEATURES = (
    "ret_1", "ret_6", "ret_24", "ret_72", "vol_24", "vol_72",
    "ema20_deviation", "ema20_slope_12", "atr_24", "range_24",
    "volume_z_72", "drawdown_72", "drawdown_168",
)


def feature_schema_sha256() -> str:
    payload = json.dumps({PAIR: list(FEATURES)}, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _valid_sha256(value: Any) -> bool:
    text = str(value or "").lower()
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def _parse_utc(value: Any, field: str) -> datetime:
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError(f"{field} must include timezone")
    return parsed.astimezone(timezone.utc)


def failed_runtime(reason: str, observed: datetime) -> dict[str, Any]:
    return {
        "schema": CONTRACT_SCHEMA,
        "model_version": MODEL_VERSION,
        "generated_at": observed.isoformat(),
        "source_healthy": False,
        "deployment_allowed": False,
        "runtime_gate_healthy": False,
        "runtime_age_seconds": None,
        "reason": f"fail_closed:{reason}",
        "pairs": {
            PAIR: {
                "buy_enabled": False,
                "risk_off_active": True,
                "force_exit": True,
                "model_signal": "UNAVAILABLE",
                "reason": f"fail_closed:{reason}",
            }
        },
    }


def load_runtime_sol_gate(
    path: Path, *, now: datetime | None = None,
    max_age_seconds: int = STALE_AFTER_SECONDS,
    expected_model_sha256: str | None = None,
    expected_feature_sha256: str | None = None,
) -> dict[str, Any]:
    observed = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema") != CONTRACT_SCHEMA:
            raise ValueError("unsupported SOL weekly contract schema")
        if payload.get("model_version") != MODEL_VERSION:
            raise ValueError("unsupported SOL weekly model version")
        if set(payload.get("pairs", {})) != {PAIR}:
            raise ValueError("SOL weekly contract must contain exactly SOL-FDUSD")
        for field in (
            "release_sha256", "model_sha256", "feature_schema_sha256",
            "strategy_sha256", "data_sha256", "state_lineage_sha256",
        ):
            if not _valid_sha256(payload.get(field)):
                raise ValueError(f"{field} is not a SHA-256 digest")
        if payload.get("feature_schema_sha256") != feature_schema_sha256():
            raise ValueError("SOL feature schema hash mismatch")
        if expected_model_sha256 and payload.get("model_sha256") != expected_model_sha256:
            raise ValueError("SOL model hash mismatch")
        if expected_feature_sha256 and payload.get("feature_schema_sha256") != expected_feature_sha256:
            raise ValueError("SOL expected feature hash mismatch")
        generated = _parse_utc(payload["generated_at"], "generated_at")
        valid_until = _parse_utc(payload["valid_until"], "valid_until")
        age = (observed - generated).total_seconds()
        if age < -10:
            raise ValueError("SOL contract timestamp is in the future")
        if age > max_age_seconds or observed > valid_until:
            raise ValueError(f"SOL weekly contract is stale by {max(0, age):.0f} seconds")
        if payload.get("source_healthy") is not True:
            raise ValueError("SOL signal source is unhealthy")
        if payload.get("deployment_allowed") is not True:
            raise ValueError("SOL weekly model is not deployment allowed")
        signal = dict(payload["pairs"][PAIR])
        if signal.get("model_signal") not in {"RISK_ON", "RISK_OFF"}:
            raise ValueError("SOL model signal is invalid")
        expected_enabled = signal["model_signal"] == "RISK_ON"
        if bool(signal.get("buy_enabled")) != expected_enabled:
            raise ValueError("SOL buy_enabled is inconsistent with model signal")
        if bool(signal.get("risk_off_active")) == expected_enabled:
            raise ValueError("SOL risk_off_active is inconsistent with model signal")
        if signal.get("model_week") != payload.get("model_week"):
            raise ValueError("SOL model week mismatch")
        return {
            **payload,
            "runtime_gate_healthy": True,
            "runtime_age_seconds": max(0, int(age)),
            "reason": "sol_weekly_gate_healthy",
        }
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        return failed_runtime(str(exc), observed)

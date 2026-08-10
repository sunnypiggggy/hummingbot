"""Fail-closed ROC/SQZMOM BUY gate for the FDUSD live Grid."""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


GRID_TECHNICAL_GATE_SCHEMA = "grid-technical-buy-gate-v3"
DEFAULT_MAX_AGE_SECONDS = 150
DEFAULT_ROC_RISK_OFF_PCT = -5.0
DEFAULT_SQZMOM_RISK_OFF_PCT = -1.0
DEFAULT_ROC_RECOVERY_PCT = 1.0
DEFAULT_SQZMOM_RECOVERY_PCT = -3.0
COMBINED_RECOVERY_RULE_VERSION = "combined-roc1-sqz3-improving-v1"


def _utc(value: str, field: str) -> datetime:
    parsed = datetime.fromisoformat(str(value))
    if parsed.tzinfo is None:
        raise ValueError(f"{field} must be timezone-aware")
    return parsed.astimezone(timezone.utc)


def _linreg_endpoint(values: list[float]) -> float:
    length = len(values)
    if length < 2:
        raise ValueError("linear regression requires at least two values")
    x_mean = (length - 1) / 2
    y_mean = sum(values) / length
    denominator = sum((index - x_mean) ** 2 for index in range(length))
    slope = sum(
        (index - x_mean) * (value - y_mean)
        for index, value in enumerate(values)
    ) / denominator
    return y_mean - slope * x_mean + slope * (length - 1)


def roc_sqz_signal_from_klines(
    klines: list[list[Any]], *, roc_length: int = 12, sqz_length: int = 20
) -> dict:
    """Match the existing DCA 4h ROC48/SQZMOM calculation exactly."""
    if len(klines) < sqz_length * 2:
        raise ValueError(f"at least {sqz_length * 2} closed klines are required")
    highs = [float(item[2]) for item in klines]
    lows = [float(item[3]) for item in klines]
    closes = [float(item[4]) for item in klines]
    if any(value <= 0 for value in closes):
        raise ValueError("kline closes must be positive")
    sources = []
    for index in range(sqz_length - 1, len(klines)):
        start = index - sqz_length + 1
        highest = max(highs[start:index + 1])
        lowest = min(lows[start:index + 1])
        close_sma = sum(closes[start:index + 1]) / sqz_length
        midpoint = ((highest + lowest) / 2 + close_sma) / 2
        sources.append(closes[index] - midpoint)
    current = _linreg_endpoint(sources[-sqz_length:])
    previous = _linreg_endpoint(sources[-sqz_length - 1:-1])
    close = closes[-1]
    color = (
        "lime" if current > 0 and current > previous
        else "green" if current > 0
        else "red" if current < previous
        else "maroon"
    )
    return {
        "bar_open_time": int(klines[-1][0]),
        "bar_close_time": int(klines[-1][6]),
        "close": close,
        "roc_48h_pct": (close / closes[-1 - roc_length] - 1) * 100,
        "sqzmom": current,
        "sqzmom_previous": previous,
        "sqzmom_pct": current / close * 100,
        "sqzmom_red": current < 0 and current < previous,
        "sqzmom_green": current > 0,
        "sqzmom_color": color,
        "sqzmom_lime": color == "lime",
    }


def build_technical_buy_gate(
    signal: Mapping[str, Any], *, previously_active: bool,
    previous_bar_close_time: int | None = None,
    previous_sqzmom_color: str | None = None,
    roc_risk_off_pct: float = DEFAULT_ROC_RISK_OFF_PCT,
    sqzmom_risk_off_pct: float = DEFAULT_SQZMOM_RISK_OFF_PCT,
    roc_recovery_pct: float = DEFAULT_ROC_RECOVERY_PCT,
    sqzmom_recovery_pct: float = DEFAULT_SQZMOM_RECOVERY_PCT,
    now: datetime | None = None, symbol: str = "BTCFDUSD",
) -> dict:
    observed = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    bar_close_time = int(signal.get("bar_close_time", 0))
    new_bar = previous_bar_close_time is None or bar_close_time != previous_bar_close_time
    color = str(signal.get("sqzmom_color") or (
        "lime" if float(signal["sqzmom"]) > 0
        and float(signal["sqzmom"]) > float(signal["sqzmom_previous"])
        else "green" if float(signal["sqzmom"]) > 0
        else "red" if float(signal["sqzmom"]) < float(signal["sqzmom_previous"])
        else "maroon"
    ))
    risk_off_condition = bool(
        float(signal["roc_48h_pct"]) <= roc_risk_off_pct
        and float(signal["sqzmom_pct"]) <= sqzmom_risk_off_pct
    )
    improving = float(signal["sqzmom"]) > float(signal["sqzmom_previous"])
    recovery_condition = bool(
        not risk_off_condition
        and float(signal["roc_48h_pct"]) >= roc_recovery_pct
        and float(signal["sqzmom_pct"]) >= sqzmom_recovery_pct
        and improving
    )
    active = bool(previously_active)
    trigger = bool(new_bar and not active and risk_off_condition)
    recover = bool(new_bar and active and recovery_condition)
    if trigger:
        active = True
    elif recover:
        active = False
    return {
        "schema_version": GRID_TECHNICAL_GATE_SCHEMA,
        "generated_at": observed.isoformat(),
        "source_healthy": True,
        "symbol": symbol,
        "interval": "4h",
        "buy_enabled": not active,
        "risk_off_active": active,
        "trigger": trigger,
        "recover": recover,
        "risk_off_condition": risk_off_condition,
        "roc_risk_off_pct": roc_risk_off_pct,
        "sqzmom_risk_off_pct": sqzmom_risk_off_pct,
        "roc_recovery_pct": roc_recovery_pct,
        "sqzmom_recovery_pct": sqzmom_recovery_pct,
        "sqzmom_improving": improving,
        "recovery_condition": recovery_condition,
        "recovery_rule_version": COMBINED_RECOVERY_RULE_VERSION,
        "last_evaluated_bar_close_time": bar_close_time,
        "last_sqzmom_color": color,
        "decision_rule": (
            f"ROC48 <= {roc_risk_off_pct:g}% and SQZMOM <= "
            f"{sqzmom_risk_off_pct:g}% disables BUY; "
            f"while active, ROC48 >= {roc_recovery_pct:g}% and SQZMOM >= "
            f"{sqzmom_recovery_pct:g}% and SQZMOM improving enables BUY"
        ),
        "signal": dict(signal),
        "reason": (
            "roc_sqzmom_combined_risk_off" if active
            else "roc_sqzmom_combined_recovery" if recover
            else "normal_conditions"
        ),
    }


def failed_technical_buy_gate(reason: str, *, now: datetime | None = None) -> dict:
    observed = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    return {
        "schema_version": GRID_TECHNICAL_GATE_SCHEMA,
        "generated_at": observed.isoformat(),
        "source_healthy": False,
        "buy_enabled": False,
        "risk_off_active": True,
        "reason": f"fail_closed:{reason}",
    }


def load_runtime_technical_gate(
    path: Path, *, now: datetime | None = None,
    max_age_seconds: int = DEFAULT_MAX_AGE_SECONDS,
) -> dict:
    observed = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != GRID_TECHNICAL_GATE_SCHEMA:
            raise ValueError("unsupported Grid technical gate schema")
        generated_at = _utc(str(payload["generated_at"]), "generated_at")
        age = (observed - generated_at).total_seconds()
        if age < -10:
            raise ValueError("Grid technical gate timestamp is in the future")
        if age > max_age_seconds:
            raise ValueError(f"Grid technical gate is stale by {age:.0f} seconds")
        if not bool(payload.get("source_healthy")):
            raise ValueError(str(payload.get("reason", "technical source unhealthy")))
        return {
            **payload,
            "buy_enabled": bool(payload.get("buy_enabled")),
            "runtime_gate_healthy": True,
            "runtime_age_seconds": max(0, int(age)),
        }
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        return {
            **failed_technical_buy_gate(str(exc), now=observed),
            "runtime_gate_healthy": False,
            "runtime_age_seconds": None,
        }


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
        os.chmod(path, 0o644)
        os.chmod(path, 0o644)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)

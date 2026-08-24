"""Fail-closed FOMC gate shared by the live-grid scheduler and strategy."""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


GRID_MACRO_GATE_SCHEMA = "grid-fomc-gate-v1"
MACRO_STATE_SCHEMA_VERSION = 3
DEFAULT_MAX_AGE_SECONDS = 150


def _utc(value: str, field: str) -> datetime:
    parsed = datetime.fromisoformat(str(value))
    if parsed.tzinfo is None:
        raise ValueError(f"{field} must be timezone-aware")
    return parsed.astimezone(timezone.utc)


def _lease_is_active(lease: Mapping[str, Any], now: datetime) -> bool:
    if str(lease.get("event_kind", "")).lower() != "fomc":
        return False
    approval = lease.get("approval")
    if not isinstance(approval, Mapping):
        return False
    if approval.get("status") != "approved" or approval.get("action") != "approve":
        return False
    effective_at = _utc(str(lease["effective_at"]), "effective_at")
    resume_at = _utc(str(lease["resume_at"]), "resume_at")
    revoked_at = lease.get("revoked_at")
    if revoked_at and now >= _utc(str(revoked_at), "revoked_at"):
        return False
    return effective_at <= now < resume_at


def build_grid_macro_gate(
    state: Mapping[str, Any] | None,
    *,
    now: datetime | None = None,
    max_source_age_seconds: int = DEFAULT_MAX_AGE_SECONDS,
    execution_enabled: bool = True,
) -> dict:
    """Convert the approved macro state into a minimal Grid pause document."""
    observed = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    base = {
        "schema_version": GRID_MACRO_GATE_SCHEMA,
        "generated_at": observed.isoformat(),
        "max_age_seconds": int(max_source_age_seconds),
        "event_kind": "fomc",
        "execution_enabled": bool(execution_enabled),
        "pause_new_orders": True,
        "source_healthy": False,
        "active_lease_ids": [],
        "active_events": [],
    }
    try:
        if not isinstance(state, Mapping):
            raise ValueError("macro state is unavailable")
        if int(state.get("schema_version", -1)) != MACRO_STATE_SCHEMA_VERSION:
            raise ValueError("unsupported macro state schema")
        last_reconcile = _utc(str(state["last_reconcile"]), "last_reconcile")
        age = (observed - last_reconcile).total_seconds()
        if age < -10:
            raise ValueError("macro state timestamp is in the future")
        if age > max_source_age_seconds:
            raise ValueError(f"macro state is stale by {age:.0f} seconds")
        leases = state.get("leases", {})
        if not isinstance(leases, Mapping):
            raise ValueError("macro leases are invalid")
        active = [
            dict(lease)
            for lease in leases.values()
            if isinstance(lease, Mapping) and _lease_is_active(lease, observed)
        ]
        would_pause = bool(active)
        base.update({
            "pause_new_orders": would_pause and bool(execution_enabled),
            "shadow_pause_new_orders": would_pause,
            "source_healthy": True,
            "source_last_reconcile": last_reconcile.isoformat(),
            "source_age_seconds": max(0, int(age)),
            "active_lease_ids": sorted(str(item["decision_id"]) for item in active),
            "active_events": [
                {
                    "decision_id": str(item["decision_id"]),
                    "event_id": str(item["event_id"]),
                    "market_impact": str(item["market_impact"]),
                    "effective_at": str(item["effective_at"]),
                    "resume_at": str(item["resume_at"]),
                }
                for item in sorted(active, key=lambda value: str(value["decision_id"]))
            ],
            "reason": (
                "approved_fomc_window_active"
                if active and execution_enabled
                else "shadow_fomc_window_active"
                if active
                else "no_active_fomc_window"
            ),
        })
    except (KeyError, TypeError, ValueError) as exc:
        base["reason"] = f"fail_closed:{exc}"
    return base


def load_runtime_macro_gate(
    path: Path,
    *,
    now: datetime | None = None,
    max_age_seconds: int = DEFAULT_MAX_AGE_SECONDS,
) -> dict:
    """Read a scheduler-produced gate; any validation failure pauses Grid."""
    observed = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != GRID_MACRO_GATE_SCHEMA:
            raise ValueError("unsupported Grid macro gate schema")
        if payload.get("event_kind") != "fomc":
            raise ValueError("Grid macro gate is not scoped to FOMC")
        generated_at = _utc(str(payload["generated_at"]), "generated_at")
        age = (observed - generated_at).total_seconds()
        if age < -10:
            raise ValueError("Grid macro gate timestamp is in the future")
        if age > max_age_seconds:
            raise ValueError(f"Grid macro gate is stale by {age:.0f} seconds")
        if not bool(payload.get("source_healthy")):
            raise ValueError(str(payload.get("reason", "macro source is unhealthy")))
        lease_ids = payload.get("active_lease_ids", [])
        if not isinstance(lease_ids, list):
            raise ValueError("Grid macro gate lease IDs are invalid")
        execution_enabled = bool(payload.get("execution_enabled"))
        if execution_enabled and bool(lease_ids) != bool(payload.get("pause_new_orders")):
            raise ValueError("Grid macro gate pause state does not match active leases")
        if not execution_enabled and bool(payload.get("pause_new_orders")):
            raise ValueError("Grid macro shadow mode may not pause live orders")
        return {
            **payload,
            "pause_new_orders": bool(payload.get("pause_new_orders")),
            "runtime_gate_healthy": True,
            "runtime_age_seconds": max(0, int(age)),
        }
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        return {
            "schema_version": GRID_MACRO_GATE_SCHEMA,
            "generated_at": observed.isoformat(),
            "pause_new_orders": True,
            "source_healthy": False,
            "runtime_gate_healthy": False,
            "active_lease_ids": [],
            "active_events": [],
            "reason": f"fail_closed:{exc}",
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
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)

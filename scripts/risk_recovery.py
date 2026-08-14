"""Persistent, fail-closed recovery state shared by live Grid and DCA guards."""

from __future__ import annotations

from typing import Any, Mapping


ACTIVE = "ACTIVE"
EXITING = "EXITING"
COOLDOWN = "COOLDOWN"
REENTRY = "REENTRY"
LATCHED = "LATCHED"
PHASES = {ACTIVE, EXITING, COOLDOWN, REENTRY, LATCHED}

POSITION_COOLDOWN_SECONDS = 30 * 60
TECHNICAL_COOLDOWN_SECONDS = 0
STRATEGY_COOLDOWN_SECONDS = 6 * 60 * 60
PORTFOLIO_COOLDOWN_SECONDS = 12 * 60 * 60
REQUIRED_HEALTHY_CYCLES = 3
EMERGENCY_ESCALATION_SECONDS = 3
EXIT_CRITICAL_SECONDS = 10

# Only transport failures that are normally recoverable without changing the
# signed model/data contract receive a grace period.  Unknown failures remain
# deterministic and therefore fail closed immediately.
TRANSIENT_TRANSPORT_MARKERS = (
    "connectionreseterror",
    "connection reset",
    "connection aborted",
    "connection refused",
    "remote disconnected",
    "remotedisconnected",
    "read timed out",
    "connect timeout",
    "connecttimeout",
    "readtimeout",
    "temporarily unavailable",
    "temporary failure in name resolution",
    "name resolution",
    "max retries exceeded",
    "http 429",
    "status code 429",
    "http 500",
    "http 502",
    "http 503",
    "http 504",
    "status code 500",
    "status code 502",
    "status code 503",
    "status code 504",
)


def classify_integrity_failure(reason: Any) -> str:
    """Classify a contract/source failure without weakening integrity checks."""
    normalized = str(reason or "").strip().lower()
    if "transient_grace_expired" in normalized:
        return "deterministic_integrity"
    if any(marker in normalized for marker in TRANSIENT_TRANSPORT_MARKERS):
        return "transient_transport"
    return "deterministic_integrity"


def advance_integrity_failure(
    previous: Mapping[str, Any] | None,
    *,
    reason: Any,
    now: float,
    grace_seconds: float,
) -> dict[str, Any]:
    """Advance a persistent transient-failure timer.

    Deterministic integrity failures expire immediately.  Transport failures
    share one episode even if the exception text changes between retries.
    """
    classification = classify_integrity_failure(reason)
    old = dict(previous or {})
    same_episode = old.get("classification") == classification
    first_seen_at = float(old.get("first_seen_at", now)) if same_episode else float(now)
    elapsed = max(0.0, float(now) - first_seen_at)
    grace = max(0.0, float(grace_seconds))
    expired = classification != "transient_transport" or elapsed >= grace
    return {
        "classification": classification,
        "first_seen_at": first_seen_at,
        "last_seen_at": float(now),
        "reason": str(reason or "unknown"),
        "attempts": int(old.get("attempts", 0)) + 1 if same_episode else 1,
        "grace_seconds": grace,
        "elapsed_seconds": elapsed,
        "remaining_seconds": max(0.0, grace - elapsed) if not expired else 0.0,
        "expired": expired,
    }


def cooldown_for_scope(scope: str) -> int:
    return {
        "technical": TECHNICAL_COOLDOWN_SECONDS,
        "position": POSITION_COOLDOWN_SECONDS,
        "strategy": STRATEGY_COOLDOWN_SECONDS,
        "portfolio": PORTFOLIO_COOLDOWN_SECONDS,
    }[scope]


def active_state() -> dict[str, Any]:
    return {
        "phase": ACTIVE,
        "mechanism": "",
        "scope": "",
        "triggered_at": None,
        "exit_target": "quote_only",
        "remaining_base": {},
        "exit_completed_at": None,
        "cooldown_until": None,
        "healthy_cycles": 0,
        "reentry": {},
        "episode_baseline": {},
    }


def normalize_state(value: Mapping[str, Any] | None) -> dict[str, Any]:
    state = active_state()
    if value:
        state.update(dict(value))
    if state["phase"] not in PHASES:
        raise ValueError(f"unknown recovery phase {state['phase']!r}")
    return state


def trigger_state(*, mechanism: str, scope: str, now: float,
                  trigger_value: Any, signal_price: Any,
                  reason: str, latched: bool = False,
                  latch_after_exit: bool = False) -> dict[str, Any]:
    if scope not in {"technical", "position", "strategy", "portfolio", "infrastructure"}:
        raise ValueError(f"unsupported recovery scope {scope!r}")
    return {
        **active_state(),
        "phase": LATCHED if latched else EXITING,
        "latch_after_exit": bool(latch_after_exit),
        "mechanism": mechanism,
        "scope": scope,
        "reason": reason,
        "trigger_value": str(trigger_value),
        "signal_price": str(signal_price),
        "triggered_at": float(now),
        "first_exit_order_at": None,
        "exit_attempts": 0,
        "critical_alerted": False,
    }


def mark_exit_complete(state: Mapping[str, Any], *, now: float,
                       remaining_base: Mapping[str, Any],
                       execution: Mapping[str, Any]) -> dict[str, Any]:
    result = normalize_state(state)
    if result["phase"] == LATCHED:
        return result
    result.update({
        "phase": LATCHED if result.get("latch_after_exit") else COOLDOWN,
        "remaining_base": {key: str(value) for key, value in remaining_base.items()},
        "exit_completed_at": float(now),
        "cooldown_until": (
            None if result.get("latch_after_exit")
            else float(now) + cooldown_for_scope(str(result["scope"]))
        ),
        "healthy_cycles": 0,
        "execution": dict(execution),
    })
    return result


def advance_recovery(state: Mapping[str, Any], *, now: float, healthy: bool,
                     gates_allow_reentry: bool) -> dict[str, Any]:
    result = normalize_state(state)
    if result["phase"] in {ACTIVE, EXITING, LATCHED}:
        return result
    result["healthy_cycles"] = int(result.get("healthy_cycles", 0)) + 1 if healthy else 0
    if (
        result["phase"] == COOLDOWN
        and float(now) >= float(result.get("cooldown_until") or float("inf"))
        and result["healthy_cycles"] >= REQUIRED_HEALTHY_CYCLES
    ):
        result["phase"] = REENTRY
    result["reentry_allowed"] = bool(
        result["phase"] == REENTRY and healthy and gates_allow_reentry
    )
    return result


def mark_reentry_complete(state: Mapping[str, Any], *, now: float,
                          baseline: Mapping[str, Any]) -> dict[str, Any]:
    previous = normalize_state(state)
    if previous["phase"] != REENTRY:
        raise ValueError("reentry can complete only from REENTRY")
    result = active_state()
    result["recovered_at"] = float(now)
    result["previous_mechanism"] = previous.get("mechanism", "")
    result["episode_baseline"] = {key: str(value) for key, value in baseline.items()}
    return result

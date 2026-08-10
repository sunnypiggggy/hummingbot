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

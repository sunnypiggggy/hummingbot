"""Offline-only recovery-counter experiment. Never imported by a live service.

A delegates directly to the unmodified production gate. B reuses that gate's
entry/evidence machinery but replaces ONLY the recovery decision and counters.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field, replace
from typing import Any

import xgboost_long_risk_gate_v22 as v22


@dataclass
class CounterState:
    gate: v22.GateState = field(default_factory=v22.GateState)
    ordinary_count: int = 0
    strong_count: int = 0
    fold: int | None = None

    def dump(self) -> dict[str, Any]:
        return {"gate": v22.state_to_dict(self.gate), "ordinary_count": self.ordinary_count,
                "strong_count": self.strong_count, "fold": self.fold}

    @classmethod
    def restore(cls, value: dict[str, Any]) -> "CounterState":
        return cls(v22.state_from_dict(value["gate"]), int(value["ordinary_count"]),
                   int(value["strong_count"]), value["fold"])


def strong_confirm(structure: tuple[float, ...]) -> bool:
    roc, sqz, di, slope, below = structure
    return bool((roc >= 0 or sqz >= 0) and di > 0 and slope >= 0 and below < .5)


def rollover(state: CounterState, fold: int, boundary: int) -> None:
    """Same release-rollover evidence barrier for both arms, no active reset."""
    if state.fold is not None and fold != state.fold and not state.gate.active:
        state.gate.above_entry_count = 0
        state.gate.armed_until = None
        state.gate.entry_evidence_not_before = boundary
    state.fold = fold


def step(state: CounterState, *, arm: str, pair: str, probability: float,
         threshold: float, ts: int, structure_ts: int, structure: tuple[float, ...],
         fold: int, boundary: int) -> dict[str, Any]:
    if arm not in {"A", "B"}:
        raise ValueError("unknown arm")
    if structure_ts > ts:
        raise ValueError("future 4h structure")
    # Late/duplicate callbacks must not mutate even the rollover metadata.
    duplicate = state.gate.last_signal_ts is not None and ts <= state.gate.last_signal_ts
    if not duplicate and state.gate.last_complete_4h_ts is not None and structure_ts < state.gate.last_complete_4h_ts:
        raise ValueError("backwards 4h structure")
    if not duplicate:
        rollover(state, fold, boundary)
    old = copy.deepcopy(state.gate)
    config = v22.GATES[pair]
    new4 = old.last_complete_4h_ts != structure_ts and not duplicate
    gap = new4 and old.last_complete_4h_ts is not None and structure_ts - old.last_complete_4h_ts != 14400
    ordinary = bool(new4 and v22.recovery_confirm(structure, old.previous_structure))
    strong = strong_confirm(structure)
    kwargs = dict(pair=pair, probability=probability, entry_threshold=threshold,
                  signal_ts=ts, last_complete_4h_ts=structure_ts, structure=structure,
                  state=state.gate)
    # Suppress only the baseline recovery decision; entry and retrigger cooldown
    # still use the exact production implementation (minimum_hours is exit-only).
    selected_config = config if arm == "A" else replace(config, minimum_hours=10**9)
    state.gate, result = v22.advance_gate(**kwargs, config=selected_config)
    recovery_path = ""
    if arm == "A":
        count = state.gate.recovery_count
        ordinary_decision_count, strong_decision_count = count, 0
        if result["transition"] == "recover":
            recovery_path = "strong_shared" if strong and count < 3 else "ordinary"
        state.ordinary_count = count
    else:
        if old.active and new4:
            if gap:
                state.ordinary_count = state.strong_count = 0
            # An unknown intervening 4h bar cannot supply ROC/SQZ improvement.
            state.ordinary_count = state.ordinary_count + 1 if ordinary and not gap else 0
            state.strong_count = state.strong_count + 1 if strong else 0
            age = ts - (old.since if old.since is not None else ts)
            if age >= config.minimum_hours * 3600:
                if state.ordinary_count >= config.recovery_4h_bars:
                    recovery_path = "ordinary"
                elif state.strong_count >= 2:
                    recovery_path = "strong_independent"
            if recovery_path:
                gate = state.gate
                gate.active = False
                gate.since = None
                gate.cooldown_until = ts + config.cooldown_hours * 3600
                gate.armed_until = None
                gate.above_entry_count = 0
                gate.last_event_id = v22.event_id(pair, ts, "recover")
                result["transition"] = "recover"
        ordinary_decision_count, strong_decision_count = state.ordinary_count, state.strong_count
        if result["transition"] in {"enter", "recover"}:
            state.ordinary_count = state.strong_count = 0
        state.gate.recovery_count = state.ordinary_count
        result = v22.snapshot(pair, probability, threshold, state.gate, result["transition"],
                              result["entry_structure_confirmed"], ordinary, structure, config)
    return {"signal_ts": ts, "pair": pair, "arm": arm, "fold": fold,
            "probability": probability, "threshold": threshold, "structure_ts": structure_ts,
            "risk_off": state.gate.active, "since": state.gate.since,
            "transition": result["transition"], "ordinary_condition": ordinary,
            "strong_condition": strong, "new_4h": new4, "missing_4h": gap,
            "ordinary_count": ordinary_decision_count, "strong_count": strong_decision_count,
            "ordinary_stored": state.ordinary_count, "strong_stored": state.strong_count,
            "recovery_path": recovery_path, "cooldown_until": state.gate.cooldown_until,
            "roc": structure[0], "sqzmom": structure[1], "di": structure[2],
            "ema_slope": structure[3], "below_ema_ratio": structure[4]}

from __future__ import annotations

import sys
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from optimize_xgboost_dual_risk_gate_180d_v5 import (  # noqa: E402
    GateParameters,
    GateState,
    HOUR,
    quantile_column,
    step_gate,
)


def test_long_gate_requires_consecutive_high_bars_and_delayed_recovery():
    params = GateParameters(0.95, 0.80, 3, 2, 24, None)
    state = GateState()
    for offset in range(2):
        state, transition, _ = step_gate(0.9, 0.8, 0.4, offset * HOUR, state, params)
        assert not state.active
        assert transition == "clear"
    state, transition, _ = step_gate(0.9, 0.8, 0.4, 2 * HOUR, state, params)
    assert state.active
    assert transition == "enter"
    state, transition, _ = step_gate(0.1, 0.8, 0.4, 3 * HOUR, state, params)
    assert state.active
    state, transition, _ = step_gate(0.1, 0.8, 0.4, 4 * HOUR, state, params)
    assert state.active
    state, transition, _ = step_gate(0.1, 0.8, 0.4, 26 * HOUR, state, params)
    assert not state.active
    assert transition == "recover"


def test_short_gate_has_hard_maximum_duration():
    params = GateParameters(0.95, 0.80, 1, 2, 1, 12)
    state, transition, _ = step_gate(0.9, 0.8, 0.4, 0, GateState(), params)
    assert state.active and transition == "enter"
    state, transition, reason = step_gate(0.9, 0.8, 0.4, 12 * HOUR, state, params)
    assert not state.active
    assert transition == "recover"
    assert reason == "maximum_risk_off_age"


def test_quantile_column_is_stable():
    assert quantile_column(0.925) == "threshold_q9250"
    assert quantile_column(0.99) == "threshold_q9900"

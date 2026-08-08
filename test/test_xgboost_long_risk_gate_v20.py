from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import refine_xgboost_long_risk_gate_v20 as v20


def test_gate_space_is_deterministic_and_long_only() -> None:
    first, second = v20.candidates(), v20.candidates()
    assert first == second
    assert len(first) == 640
    assert all(item.arm_hours in (48, 72) for item in first)
    assert all(item.recovery_4h_bars in (3, 4) for item in first)


def test_relaxed_entry_still_requires_bearish_direction() -> None:
    previous = (-1.0, -1.0, -1.0, -1.0, .8)
    assert v20._entry_confirm((-.8, -1.2, -1.0, -1.0, .8), previous, "directional_relaxed")
    assert not v20._entry_confirm((.1, -1.2, -1.0, -1.0, .8), previous, "directional_relaxed")
    assert not v20._entry_confirm((-.8, -1.2, 1.0, 1.0, .2), previous, "directional_relaxed")


def test_recovery_requires_both_momentum_improvement_and_two_relief_votes() -> None:
    previous = (-2.0, -2.0, -1.0, -1.0, .8)
    assert v20._recovery_confirm((-1.0, -1.0, 1.0, .1, .4), previous, "structural_relief")
    assert not v20._recovery_confirm((-1.0, -2.1, 1.0, .1, .4), previous, "structural_relief")
    assert not v20._recovery_confirm((-1.0, -1.0, -1.0, -1.0, .8), previous, "structural_relief")
    assert not v20._recovery_confirm((-1.0, -1.0, 1.0, .1, .4), previous, "regime_exit")


def test_low_probability_does_not_recover_without_structure() -> None:
    count = 96
    frame = pd.DataFrame({
        "signal_ts": np.arange(count) * v20.v19.HOUR,
        "probability": np.r_[np.full(8, .9), np.full(count - 8, .01)],
        v20.v19.legacy.v5.quantile_column(.90): .5,
        "last_complete_4h_ts": np.repeat(np.arange(24) * 4 * v20.v19.HOUR, 4),
        "roc_48h_4h": np.linspace(-.1, -3, count),
        "sqzmom_pct_4h": np.linspace(-.1, -3, count),
        "di_spread": -1.0, "ema20_slope_atr_12h": -1.0,
        "below_ema20_ratio_72h": .8,
    })
    gate = v20.Gate(.90, 1, 72, 48, 24, 3, "persistent_bearish", "structural_relief")
    _, states, intervals = v20.build_state(frame, "BTC-FDUSD", gate, False, True)
    assert states.transition.eq("enter").sum() == 1
    assert states.transition.eq("recover").sum() == 0
    assert intervals.iloc[0].end_reason == "research_period_end"


def test_v21_focused_space_uses_high_quantiles_and_adaptive_recovery() -> None:
    import refine_xgboost_long_risk_gate_v21 as v21
    gates = v21.focused_candidates()
    assert len(gates) == 48
    assert {item.entry_quantile for item in gates} == {.98, .985, .99}
    assert {item.recovery_mode for item in gates} == {"structural_relief", "adaptive_relief"}
    assert all(item.arm_hours == 48 and item.minimum_hours == 48 for item in gates)

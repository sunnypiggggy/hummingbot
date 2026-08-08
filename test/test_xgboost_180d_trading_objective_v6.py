from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import optimize_xgboost_180d_trading_objective_v6 as research


def test_gate_search_spaces_are_deterministic_and_distinct() -> None:
    long = research.long_gate_candidates()
    short = research.short_gate_candidates()
    assert len(long) == 20
    assert len(short) == 24
    assert len({research.gate_id("long_persistent_72h", item) for item in long}) == 20
    assert len({research.gate_id("short_spike_1h_24h", item) for item in short}) == 24
    assert all(item.minimum_hours >= 120 for item in long)
    assert all(item.maximum_hours <= 24 for item in short)


def test_equal_weight_profit_drawdown_score_and_pareto_front() -> None:
    frame = pd.DataFrame([
        {"candidate_id": "profit", "oos_pnl_fdusd": 10.0, "stitched_max_drawdown_pct": -8.0,
         "portfolio_stop_events": 0, "pair_stop_events": 0, "risk_off_pair_hours": 1.0},
        {"candidate_id": "balanced", "oos_pnl_fdusd": 8.0, "stitched_max_drawdown_pct": -4.0,
         "portfolio_stop_events": 0, "pair_stop_events": 0, "risk_off_pair_hours": 1.0},
        {"candidate_id": "dominated", "oos_pnl_fdusd": 7.0, "stitched_max_drawdown_pct": -5.0,
         "portfolio_stop_events": 0, "pair_stop_events": 0, "risk_off_pair_hours": 1.0},
    ])
    ranked = research.add_pareto_and_score(frame).set_index("candidate_id")
    assert ranked.loc["balanced", "trading_objective_score"] == pytest.approx(5 / 6)
    assert bool(ranked.loc["profit", "pareto_front"])
    assert bool(ranked.loc["balanced", "pareto_front"])
    assert not bool(ranked.loc["dominated", "pareto_front"])


def test_stitched_drawdown_uses_420_reference_equity() -> None:
    result = {
        "equity": pd.DataFrame({
            "fold": [1, 1, 2], "timestamp": [1, 2, 3],
            "cumulative_oos_pnl": [0.0, 42.0, 0.0],
        })
    }
    # 462 to 420 is a 9.0909% drawdown on the stitched reference path.
    assert round(research.stitched_drawdown(result), 6) == -9.090909

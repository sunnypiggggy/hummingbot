from __future__ import annotations

import pandas as pd

import retrain_xgboost_long_risk_gate_250d_v17 as v17


def test_period_is_exactly_250_days() -> None:
    assert v17.END_TS - v17.START_TS == 250 * 86400


def test_deterministic_independent_search_space() -> None:
    specs = v17.specs()
    assert len(specs) == 2 * 2 * 2 * 2 * 40
    assert len({item["model_key"] for item in specs}) == len(specs)
    assert {item["pair"] for item in specs} == set(v17.PAIRS)
    assert any("roc_48h_4h" in item["features"] for item in specs)
    assert any("sqzmom_pct_4h" in item["features"] for item in specs)


def test_legacy_fast_intervals_are_rebound_to_requested_pair() -> None:
    legacy = pd.DataFrame({
        "pair": ["ETH-FDUSD"], "start_ts": [1], "end_ts": [2],
        "duration_hours": [1 / 3600], "end_reason": ["test"],
    })
    rebound = v17.bind_interval_pair(legacy, "BTC-FDUSD")
    assert rebound.pair.tolist() == ["BTC-FDUSD"]
    assert legacy.pair.tolist() == ["ETH-FDUSD"]


def test_research_contract_never_authorizes_deployment() -> None:
    assert "market sell" in v17.__doc__.lower()
    assert v17.MODEL_VERSION.endswith("250d")

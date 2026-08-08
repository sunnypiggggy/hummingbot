from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from compare_independent_gate_ml_stops import (  # noqa: E402
    TECHNICAL_PARAMS,
    build_risk_timeline,
    recovery_condition,
)
from grid_ml_momentum_stop import (  # noqa: E402
    advance_pair_state,
    build_contract,
    enforce_freshness,
    failed_contract,
    feature_schema_hash,
)


def prediction_row(pair: str, probability: float, signal_ts: int, **overrides):
    values = {
        "variant": "AdaBoost | shared",
        "algorithm": "AdaBoost",
        "architecture": "shared",
        "pair": pair,
        "probability": probability,
        "signal_ts": signal_ts,
        "last_complete_1h_ts": signal_ts,
        "last_complete_4h_ts": signal_ts,
        "roc_48h_4h": 5.0,
        "sqzmom_pct_4h": 0.0,
        "sqzmom_value_4h": 2.0,
        "sqzmom_slope_4h": 0.1,
        "sqzmom_improving_4h": 1.0,
    }
    values.update(overrides)
    return values


def test_pair_recovery_thresholds_are_independent():
    btc = type("Row", (), prediction_row("BTC-FDUSD", 0.1, 3_600, roc_48h_4h=2.0))
    eth = type("Row", (), prediction_row("ETH-FDUSD", 0.1, 3_600, roc_48h_4h=2.0))
    assert recovery_condition(btc, "BTC-FDUSD") is True
    assert recovery_condition(eth, "ETH-FDUSD") is False
    assert TECHNICAL_PARAMS["ETH-FDUSD"].roc_recovery_pct == 3.0


def test_btc_risk_does_not_pause_eth_and_recovery_needs_both_conditions():
    rows = [
        prediction_row("BTC-FDUSD", 0.95, 3_600),
        prediction_row("ETH-FDUSD", 0.10, 3_600),
        prediction_row(
            "BTC-FDUSD", 0.10, 7_200, roc_48h_4h=-2.0,
            sqzmom_pct_4h=-4.0, sqzmom_improving_4h=0.0,
        ),
        prediction_row("ETH-FDUSD", 0.10, 7_200),
        prediction_row("BTC-FDUSD", 0.10, 10_800),
        prediction_row("ETH-FDUSD", 0.10, 10_800),
    ]
    timeline, states, events = build_risk_timeline(
        pd.DataFrame(rows), "AdaBoost | shared",
        {"BTC-FDUSD": 0.9, "ETH-FDUSD": 0.9}, 3_600, 14_400,
    )
    assert timeline["BTC-FDUSD"][3_600] == 0.95
    assert 3_600 not in timeline["ETH-FDUSD"]
    btc = states[states.pair == "BTC-FDUSD"].set_index("signal_ts")
    assert bool(btc.loc[7_200, "risk_off_active"]) is True
    assert bool(btc.loc[10_800, "risk_off_active"]) is False
    assert list(events[events.pair == "BTC-FDUSD"].side) == ["PAUSE", "RESUME"]


def test_stop_action_is_only_emitted_on_first_risk_edge():
    first = advance_pair_state(
        pair="BTC-FDUSD", probability=0.9, entry_threshold=0.8,
        previous_risk_off=False, recovery_condition_met=False, signal_ts=3_600,
        last_complete_1h_ts=3_600, last_complete_4h_ts=0,
        model_version="v1", recovery_details={},
    )
    held = advance_pair_state(
        pair="BTC-FDUSD", probability=0.95, entry_threshold=0.8,
        previous_risk_off=True, recovery_condition_met=False, signal_ts=7_200,
        last_complete_1h_ts=7_200, last_complete_4h_ts=0,
        model_version="v1", recovery_details={},
    )
    assert first["stop_excess_inventory"] is True
    assert held["stop_excess_inventory"] is False
    assert first["event_id"] == advance_pair_state(
        pair="BTC-FDUSD", probability=0.9, entry_threshold=0.8,
        previous_risk_off=False, recovery_condition_met=False, signal_ts=3_600,
        last_complete_1h_ts=3_600, last_complete_4h_ts=0,
        model_version="v1", recovery_details={},
    )["event_id"]


def test_contract_is_research_only_and_unhealthy_source_fails_closed_without_sell():
    signal = advance_pair_state(
        pair="BTC-FDUSD", probability=0.9, entry_threshold=0.8,
        previous_risk_off=False, recovery_condition_met=False, signal_ts=3_600,
        last_complete_1h_ts=3_600, last_complete_4h_ts=0,
        model_version="v1", recovery_details={},
    )
    eth = {**signal, "event_id": "eth"}
    payload = build_contract(
        pair_signals={"BTC-FDUSD": signal, "ETH-FDUSD": eth},
        generated_at=3_600, valid_until=3_750, model_version="v1",
        model_sha256="m", feature_schema_sha256=feature_schema_hash(["x"]),
    )
    assert payload["deployment_allowed"] is False
    failed = failed_contract(
        generated_at=3_600, valid_until=3_750, model_version="v1",
        model_sha256="m", feature_schema_sha256="f", reason="stale",
    )
    assert failed["source_healthy"] is False
    assert all(not value["buy_enabled"] for value in failed["pairs"].values())
    assert all(not value["stop_excess_inventory"] for value in failed["pairs"].values())
    stale = enforce_freshness(payload, 3_751)
    assert stale["source_healthy"] is False
    assert all(not value["buy_enabled"] for value in stale["pairs"].values())
    assert all(not value["stop_excess_inventory"] for value in stale["pairs"].values())

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import prepare_xgboost_long_risk_gate_v16 as v16
from build_xgboost_risk_gate_signal import persistent_entry_evidence


def test_v16_is_independent_long_only_and_deterministic() -> None:
    assert v16.PAIRS == ("BTC-FDUSD", "ETH-FDUSD")
    assert v16.TARGETS == ("long_72h", "long_120h")
    assert v16.FEATURES == ("adx_14", "di_spread", "atr_pct", "btc_volatility_20")
    assert len(v16.tune.xgb_configurations()) == 40
    assert len(v16.engine.refinement_gates("long")) == 128


def test_120_hour_purge_applies_to_both_long_targets() -> None:
    frame = pd.DataFrame({
        "signal_ts": [1_000, 2_000],
        "label_ready_ts_long_72h": [1_000 + 72 * v16.HOUR, 2_000 + 130 * v16.HOUR],
        "label_ready_ts_long_120h": [1_000 + 120 * v16.HOUR, 2_000 + 120 * v16.HOUR],
    })
    result = v16.apply_120h_purge(frame)
    assert result.label_ready_ts_long_72h.tolist() == [
        1_000 + 120 * v16.HOUR, 2_000 + 130 * v16.HOUR,
    ]
    assert result.label_ready_ts_long_120h.tolist() == [
        1_000 + 120 * v16.HOUR, 2_000 + 120 * v16.HOUR,
    ]


def test_persistent_probability_evidence_requires_three_rising_hours_and_gap() -> None:
    history = [
        {"probability": 0.70, "roc": 1.0, "sqz": 1.0},
        {"probability": 0.73, "roc": 1.0, "sqz": 1.0},
    ]
    evidence, probability, technical = persistent_entry_evidence(
        history, 0.80, 0.78, 0.58, 1.0, 1.0,
    )
    assert evidence and probability and not technical


def test_persistent_technical_evidence_uses_two_complete_four_hour_steps() -> None:
    history = [
        {"probability": 0.4, "roc": -1.0 - i * 0.1, "sqz": -0.5 - i * 0.1}
        for i in range(8)
    ]
    evidence, probability, technical = persistent_entry_evidence(
        history, 0.4, 0.8, 0.6, -2.0, -1.5,
    )
    assert evidence and technical and not probability


def test_feature_hash_is_order_bound() -> None:
    assert v16.sha256_json({"long": list(v16.FEATURES)}) != v16.sha256_json(
        {"long": list(reversed(v16.FEATURES))}
    )


def test_no_short_or_market_sell_terms_in_model_contract_constants() -> None:
    import grid_xgboost_risk_gate as contract

    assert contract.REQUIRED_CHANNELS == ("long",)
    assert "long" in contract.SCHEMA
    failed = contract._failed_runtime("test", pd.Timestamp("2026-08-01", tz="UTC").to_pydatetime())
    assert failed["short_spike_enabled"] is False
    assert failed["market_sell_action"] is False
    assert all(not value["buy_enabled"] for value in failed["pairs"].values())


def test_250_day_adapter_is_exact_and_keeps_long_only_feature_set() -> None:
    import retest_xgboost_long_risk_gate_v16_250d as retest

    assert retest.END_TS - retest.START_TS == 250 * 86400
    assert "250d" in retest.engine.MODEL_VERSION
    assert retest.engine.FEATURES == ("adx_14", "di_spread", "atr_pct", "btc_volatility_20")
    assert retest.engine.TARGETS == ("long_72h", "long_120h")

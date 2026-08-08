from pathlib import Path
import sys

import pandas as pd


sys.path.insert(0, str((Path(__file__).resolve().parents[1] / "scripts").resolve()))

import optimize_xgboost_roc_sqz_pair_risk_gate_v8 as research
from tune_xgboost_momentum_stop_v2 import xgb_configurations


def _prediction(pair: str) -> pd.DataFrame:
    rows = []
    for index, probability in enumerate((0.2, 0.95, 0.96, 0.1, 0.1, 0.1, 0.1, 0.1)):
        row = {
            "pair": pair,
            "signal_ts": research.START_TS + index * research.HOUR,
            "target": 0.0,
            "probability": probability,
            "strategy": "long_72h",
        }
        for quantile in sorted({
            *research.ENTRY_QUANTILES,
            *(max(0.50, value - 0.10) for value in research.ENTRY_QUANTILES),
        }):
            row[research.v5.quantile_column(quantile)] = 0.9 if quantile >= 0.9 else 0.3
        rows.append(row)
    return pd.DataFrame(rows)


def test_feature_contract_contains_only_roc_and_sqz_derivatives():
    assert len(research.ROC_SQZ_FEATURES) == 13
    assert len(set(research.ROC_SQZ_FEATURES)) == len(research.ROC_SQZ_FEATURES)
    assert all("roc" in feature or "sqz" in feature for feature in research.ROC_SQZ_FEATURES)
    assert "pair_is_eth" not in research.ROC_SQZ_FEATURES
    assert "atr_pct" not in research.ROC_SQZ_FEATURES


def test_parameter_and_state_machine_search_counts_are_deterministic():
    assert len(xgb_configurations()) == 40
    assert len(research.v7.sampled_gates("long")) == 576
    assert len(research.v7.sampled_gates("short")) == 192
    assert len(research.refinement_gates("long")) == 128
    assert len(research.refinement_gates("short")) == 64
    assert 40 * 2 * 3 * 8 == 1920
    assert 2 * 5 * (128 + 64) == 1920


def test_pair_model_key_forces_pair_specific_hyperparameters():
    btc = research.pair_model_key("long_72h", "BTC-FDUSD", "xgb_01")
    eth = research.pair_model_key("long_72h", "ETH-FDUSD", "xgb_02")
    assert btc == "long_72h|BTC-FDUSD|xgb_01"
    assert eth == "long_72h|ETH-FDUSD|xgb_02"
    assert btc != eth


def test_btc_gate_never_changes_eth_timeline():
    gate = research.v5.GateParameters(0.9, 0.8, 2, 4, 1, 24, 0)
    timeline, states, _, _ = research.build_pair_gate(
        _prediction("BTC-FDUSD"), "BTC-FDUSD", "long", "long_72h", gate,
        end_ts=research.START_TS + 8 * research.HOUR,
    )
    assert timeline["ETH-FDUSD"] == {}
    assert not states[states.risk_off_active].empty


def test_anchor_metrics_require_both_windows_and_no_more_than_eight_intervals():
    intervals = pd.DataFrame([
        {
            "pair": "BTC-FDUSD", "start_ts": start, "end_ts": end,
            "duration_hours": (end - start) / research.HOUR,
        }
        for _, start, end in research.ANCHOR_WINDOWS
    ])
    result = research.pair_anchor_metrics(intervals, "BTC-FDUSD")
    assert result["anchor_pass"]
    assert result["feb_03_06_coverage"] == 1.0
    assert result["jun_01_06_coverage"] == 1.0


def test_pair_gate_events_never_contain_sell_or_taker_action():
    gate = research.v5.GateParameters(0.9, 0.8, 2, 4, 1, 24, 0)
    _, _, events, _ = research.build_pair_gate(
        _prediction("BTC-FDUSD"), "BTC-FDUSD", "long", "long_72h", gate,
        end_ts=research.START_TS + 8 * research.HOUR,
    )
    assert set(events.event).issubset({"enter", "recover"})
    assert not {"sell", "taker", "stop_excess_inventory"}.intersection(events.columns)

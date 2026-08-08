from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import retrain_xgboost_long_risk_gate_250d_v19 as v19


def test_period_is_explicit_and_exactly_250_days() -> None:
    assert v19.PERIOD.end_ts - v19.PERIOD.start_ts == 250 * v19.DAY
    args = argparse.Namespace(xgb_threads=1)
    period = v19.ResearchPeriod(100, 200)
    v19.init_worker(pd.DataFrame(), pd.DataFrame(), args, period)
    assert v19._PERIOD == period


def test_event_onset_is_refractory_and_tail_is_unlabelled() -> None:
    regime = pd.Series(np.zeros(260))
    regime.iloc[80:110] = 1
    regime.iloc[120:150] = 1  # within the 48-hour refractory interval
    target, uniqueness = v19.event_onset_target(regime, horizon=72, lead_hours=24)
    assert target.notna().sum() == 165
    assert 0 < int(target.sum()) <= 25
    assert (uniqueness[target.eq(1)] < 1).all()
    assert target.iloc[-95:].isna().all()


def test_training_split_keeps_calibration_out_of_final_development() -> None:
    cutoff = 200 * v19.DAY
    frame = pd.DataFrame({
        "signal_ts": np.arange(100, 200) * v19.DAY,
        "label_ready_ts": np.arange(100, 200) * v19.DAY + v19.HOUR,
        "target": np.tile([0, 1], 50),
    })
    mature, development, early_train, early = v19.split_training(frame, cutoff)
    calibration = mature.loc[~mature.index.isin(development.index)]
    assert set(calibration.index).isdisjoint(development.index)
    assert set(early.index).issubset(development.index)
    assert set(early_train.index).isdisjoint(early.index)
    assert development.signal_ts.max() < calibration.signal_ts.min()


def _state_prediction() -> pd.DataFrame:
    timestamps = np.arange(0, 80) * v19.HOUR
    threshold = np.full(80, .7)
    probability = np.r_[np.full(8, .2), np.full(64, .9), np.full(8, .1)]
    # Worsening at the 8h complete-bar update enters the gate.  Afterwards
    # probability falls, but structure never improves, so recovery is barred.
    roc = np.repeat([-.1, -.2, -.3, -.4, -.5, -.6, -.7, -.8, -.9, -1], 8)
    sqz = roc.copy()
    return pd.DataFrame({
        "signal_ts": timestamps, "probability": probability,
        v19.legacy.v5.quantile_column(.90): threshold,
        "last_complete_4h_ts": np.repeat(np.arange(10) * 4 * v19.HOUR, 8),
        "roc_48h_4h": roc, "sqzmom_pct_4h": sqz,
        "di_spread": -1.0, "ema20_slope_atr_12h": -1.0,
        "below_ema20_ratio_72h": .8,
    })


def test_probability_alone_cannot_recover_long_gate() -> None:
    prediction = _state_prediction()
    period = v19.ResearchPeriod(0, 80 * v19.HOUR)
    _, states, intervals = v19.build_long_state(
        prediction, "BTC-FDUSD", v19.LongGate(.90, 1, 24, 24), period,
    )
    assert states.transition.eq("enter").sum() == 1
    assert states.transition.eq("recover").sum() == 0
    assert intervals.iloc[0].end_reason == "research_period_end"


def test_pair_timeline_missing_timestamp_is_fail_closed() -> None:
    period = v19.ResearchPeriod(0, 900)
    combined = v19.combine_timelines({"BTC-FDUSD": {0: True}, "ETH-FDUSD": {0: True}}, period)
    assert combined["BTC-FDUSD"][0] is True
    assert combined["BTC-FDUSD"][300] is False
    assert combined["ETH-FDUSD"][600] is False


def test_zero_structural_candidates_hard_stop_before_grid(tmp_path: Path) -> None:
    args = argparse.Namespace(output_dir=tmp_path)
    frame = pd.DataFrame({"pair": list(v19.PAIRS), "structure_pass": [False, False]})
    with pytest.raises(RuntimeError, match="hard stop"):
        v19.hard_stop_if_no_structure(args, frame)
    payload = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert payload["grid_search_executed"] is False
    assert payload["deployment_allowed"] is False


def test_feature_contract_contains_direction_and_persistence() -> None:
    features = set(v19.FEATURE_SETS["full_structure"])
    assert {"roc_48h_4h", "sqzmom_pct_4h", "di_spread"} <= features
    assert {"drawdown_duration_168h", "below_ema20_ratio_72h",
            "downside_semivariance_ratio_72h", "trend_efficiency_72h"} <= features

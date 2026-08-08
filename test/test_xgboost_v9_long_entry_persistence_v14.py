from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str((Path(__file__).resolve().parents[1] / "scripts").resolve()))

import refine_xgboost_v9_long_entry_persistence_v14 as research


def test_entry_evidence_requires_persistent_probability_or_joint_technical_worsening():
    timestamps = [research.engine.START_TS + i * research.HOUR for i in range(12)]
    prediction = pd.DataFrame({
        "pair": "BTC-FDUSD", "signal_ts": timestamps, "target": 0,
        "probability": [0.4, 0.5, 0.6, 0.55, 0.54, 0.53, 0.52, 0.51, 0.50, 0.49, 0.48, 0.47],
    })
    panel = pd.DataFrame({
        "pair": "BTC-FDUSD", "signal_ts": timestamps,
        "roc_48h_4h": [-1, -1, -1, -1, -2, -2, -2, -2, -3, -3, -3, -3],
        "sqzmom_pct_4h": [-1, -1, -1, -1, -2, -2, -2, -2, -3, -3, -3, -3],
    })
    output = research.attach_entry_evidence(prediction, panel, "BTC-FDUSD")
    assert bool(output.loc[2, "probability_rising_3h"])
    assert not bool(output.loc[3, "probability_rising_3h"])
    assert bool(output.loc[8, "roc_sqz_worsening_8h"])


def test_entry_evidence_rejects_roc_only_deterioration():
    timestamps = [research.engine.START_TS + i * research.HOUR for i in range(9)]
    prediction = pd.DataFrame({
        "pair": "ETH-FDUSD", "signal_ts": timestamps, "target": 0,
        "probability": [0.5] * 9,
    })
    panel = pd.DataFrame({
        "pair": "ETH-FDUSD", "signal_ts": timestamps,
        "roc_48h_4h": [-1] * 4 + [-2] * 4 + [-3],
        "sqzmom_pct_4h": [-3] * 4 + [-2] * 4 + [-1],
    })
    output = research.attach_entry_evidence(prediction, panel, "ETH-FDUSD")
    assert not bool(output.loc[8, "roc_sqz_worsening_8h"])

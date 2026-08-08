from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS)) if str(SCRIPTS) not in sys.path else None
import retrain_eth_xgboost_anchor_focused_v16 as v16


def test_model_search_is_deterministic_and_eth_long_only() -> None:
    specs = v16.model_specs()
    assert len(specs) == 2 * 3 * 3 * 40
    assert {item["target"] for item in specs} == set(v16.TARGETS)
    assert len({item["model_key"] for item in specs}) == len(specs)


def test_persistent_weighting_only_uses_mature_outcome_columns() -> None:
    frame = pd.DataFrame({"target": [0, 1], "future_below_fraction_72h_v7": [0.1, 1.0],
                          "future_close_return_72h_v7": [-.01, -.10], "long_threshold_72h_v7": [-.03, -.03]})
    balanced = v16.weighted_samples(frame, "long_72h", "balanced")
    focused = v16.weighted_samples(frame, "long_72h", "persistent_severity")
    assert focused[0] == balanced[0]
    assert focused[1] > balanced[1]


def test_feature_sets_have_no_future_columns() -> None:
    for target_sets in v16.FEATURE_SETS.values():
        for features in target_sets.values():
            assert not any(name.startswith("future_") or name.startswith("target") for name in features)

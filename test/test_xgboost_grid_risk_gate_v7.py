from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import optimize_xgboost_grid_risk_gate_v7 as research


def test_xgboost_configuration_space_is_fixed_at_40_unique_rows() -> None:
    configurations = research.xgb_configurations()
    assert len(configurations) == 40
    assert len({research._sha256_json(item) for item in configurations}) == 40
    assert {item["config_id"] for item in configurations} == {
        f"xgb_{index:02d}" for index in range(40)
    }


def test_prediction_cache_rejects_hash_mismatch(tmp_path: Path) -> None:
    cache = tmp_path / "probability.csv.gz"
    prediction = pd.DataFrame({"probability": [0.1]})
    audit = pd.DataFrame({"cutoff": [1]})
    metadata = {"schema": "cache", "feature_hash": "good"}
    research.save_prediction_cache(cache, prediction, audit, metadata)
    loaded, loaded_audit = research.load_prediction_cache(cache, metadata)
    assert loaded.probability.tolist() == [0.1]
    assert loaded_audit.cutoff.tolist() == [1]
    with pytest.raises(ValueError, match="hash mismatch"):
        research.load_prediction_cache(cache, {**metadata, "feature_hash": "changed"})


def test_gate_samples_are_deterministic_unique_and_channel_specific() -> None:
    long_first = research.sampled_gates("long")
    long_second = research.sampled_gates("long")
    short = research.sampled_gates("short")
    assert long_first == long_second
    assert len(long_first) == 576
    assert len(short) == 192
    assert len({research.gate_id("long", item) for item in long_first}) == 576
    assert len({research.gate_id("short", item) for item in short}) == 192
    assert all(item.recovery_quantile == max(0.50, item.entry_quantile - 0.10)
               for item in (*long_first, *short))
    assert all(item.maximum_hours in {72, 120, 168} for item in long_first)
    assert all(item.maximum_hours in {2, 4, 6} for item in short)


def test_anchor_gate_requires_timely_pair_coverage_and_limits_frequency() -> None:
    rows = []
    for pair in research.PAIRS:
        for _, start, end in research.ANCHOR_WINDOWS:
            rows.append({
                "pair": pair, "start_ts": start, "end_ts": end,
                "duration_hours": (end - start) / 3600,
            })
    metrics = research.anchor_metrics(pd.DataFrame(rows))
    assert metrics["long_anchor_pass"]
    assert metrics["BTC_feb_03_06_coverage"] == 1.0
    missed = pd.DataFrame(rows)
    missed.loc[(missed.pair == "BTC-FDUSD") & (missed.start_ts == research.ANCHOR_WINDOWS[0][1]), "start_ts"] += 24 * 3600
    assert not research.anchor_metrics(missed)["long_anchor_pass"]


def test_profit_drawdown_score_is_equal_weight_and_eligibility_first() -> None:
    frame = pd.DataFrame([
        {"candidate_id": "profit", "eligible": False, "oos_pnl_fdusd": 10.0,
         "stitched_max_drawdown_pct": -10.0, "portfolio_stop_events": 0,
         "pair_stop_events": 0, "risk_off_pair_hours": 1.0},
        {"candidate_id": "balanced", "eligible": True, "oos_pnl_fdusd": 5.0,
         "stitched_max_drawdown_pct": -5.0, "portfolio_stop_events": 0,
         "pair_stop_events": 0, "risk_off_pair_hours": 1.0},
    ])
    ranked = research.score_frame(frame)
    assert ranked.iloc[0].candidate_id == "balanced"
    assert ranked.set_index("candidate_id").loc["balanced", "objective_score"] == 0.75


def test_target_ready_times_match_72_120_and_6_hours() -> None:
    panel = pd.DataFrame({
        "signal_ts": [100], "target_long_72h": [1.0], "target_long_120h": [0.0],
        "target_short_1h_6h": [1.0], "label_ready_ts_long_72h": [100 + 72 * 3600],
        "label_ready_ts_long_120h": [100 + 120 * 3600],
        "label_ready_ts_short_1h_6h": [100 + 6 * 3600],
    })
    for target, hours in (("long_72h", 72), ("long_120h", 120), ("short_1h_6h", 6)):
        working = research.working_target(panel, target)
        assert int(working.label_ready_ts.iloc[0]) == 100 + hours * 3600

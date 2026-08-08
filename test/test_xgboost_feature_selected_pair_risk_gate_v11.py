from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str((Path(__file__).resolve().parents[1] / "scripts").resolve()))

import optimize_xgboost_feature_selected_pair_risk_gate_v11 as v11


OUT = Path("results/backtests/xgboost_feature_selected_pair_risk_gate_v11")


def test_candidate_contracts_are_pair_and_channel_specific():
    btc_long = set(v11.candidate_features("BTC-FDUSD", "long_72h"))
    eth_long = set(v11.candidate_features("ETH-FDUSD", "long_72h"))
    btc_short = set(v11.candidate_features("BTC-FDUSD", "short_1h_6h"))
    eth_short = set(v11.candidate_features("ETH-FDUSD", "short_1h_6h"))
    assert "eth_sync_down_ratio_72h" in btc_long
    assert "btc_downside_beta_72h" in eth_long
    assert "eth_short_corr_1h" in btc_short
    assert "btc_short_corr_1h" in eth_short
    assert not (btc_long & {"btc_downside_beta_72h"})


def test_block_permutation_is_deterministic_and_preserves_values():
    source = pd.Series(np.arange(53, dtype=float))
    first = v11.rotate_24h_blocks(source)
    second = v11.rotate_24h_blocks(source)
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(np.sort(first), np.sort(source.to_numpy()))
    assert not np.array_equal(first, source.to_numpy())


def test_locked_selection_outputs_have_no_lookahead_and_are_complete():
    audit = pd.read_csv(OUT / "feature_selection_fold_audit.csv")
    assert (audit.last_label_ready_ts <= audit.train_cutoff_ts).all()
    subsets = json.loads((OUT / "selected_feature_subsets.json").read_text(encoding="utf-8"))["subsets"]
    assert len(subsets) == 30
    counts = pd.DataFrame(subsets).groupby(["pair", "target"]).size()
    assert counts.eq(5).all()
    assert all(3 <= len(item["features"]) <= 8 for item in subsets)
    assert all(len(item["features"]) == len(set(item["features"])) for item in subsets)


def test_full_search_is_40_configs_times_five_subsets_times_thresholds():
    parameters = pd.read_csv(OUT / "xgboost_v11_feature_subset_parameters.csv")
    assert len(parameters) == 1200
    assert parameters.groupby(["base_config_id"]).ngroups == 40
    screen = pd.read_csv(OUT / "model_screen_5subsets_x40_x2pairs_x3targets_x8.csv")
    assert len(screen) == 9600
    assert screen.model_key.nunique() == 1200


def test_final_contract_is_fail_closed_and_never_sells():
    summary = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
    signal = json.loads((OUT / "grid_xgboost_risk_gate_v4_sample.json").read_text(encoding="utf-8"))
    assert summary["verdict"] == "NO-GO"
    assert summary["deployment_allowed"] is False
    assert signal["schema"] == "grid-xgboost-risk-gate-v4"
    assert signal["deployment_allowed"] is False
    assert signal["mechanism1_fallback_allowed"] is False
    assert signal["market_sell_action"] is False
    assert signal["stop_excess_inventory"] is False
    assert all(not item["buy_enabled"] for item in signal["pairs"].values())


def test_drop_column_is_a_real_weekly_grid_replay():
    ablation = pd.read_csv(OUT / "drop_column_grid_ablation.csv")
    assert not ablation.empty
    assert ablation.evaluation_scope.eq("weekly_walk_forward_full_grid").all()
    assert np.isfinite(ablation.grid_composite_contribution).all()
    assert {"full_pnl_fdusd", "drop_pnl_fdusd", "drop_pair_stops"}.issubset(ablation.columns)

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import grid_xgboost_shadow_gate_v22 as contract  # noqa: E402
import ethbtc_forced_exit_contract as live_contract  # noqa: E402
import retrain_xgboost_long_risk_gate_250d_v19 as research  # noqa: E402
import xgboost_long_risk_gate_v22 as v22  # noqa: E402
from build_xgboost_v22_shadow_signal import load_state  # noqa: E402


RESULT = ROOT / "results/backtests/xgboost_grid_long_risk_gate_v22_weekly_250d"
PACKAGE = RESULT / "shadow_package"
MODEL = PACKAGE / "models/xgboost_long_risk_gate_v22_weekly.joblib"


def test_v22_embeds_weekly_probability_semantics_and_denies_fallback() -> None:
    spec = v22.strategy_spec()
    assert spec["probability_semantics"].startswith("weekly_walk_forward")
    assert spec["model_rollover"]["gate_state_reset"] is False
    assert spec["model_rollover"]["missing_week"] == "fail_closed"
    assert spec["model_rollover"]["previous_week_fallback"] is False
    assert spec["model_rollover"]["v21_final_refit_fallback"] is False


def test_signed_contiguous_rollover_preserves_state_when_training_policy_hash_changes(
    tmp_path: Path,
) -> None:
    predecessor_lock = "a" * 64
    state_path = tmp_path / "state.json"
    state_path.write_text(json.dumps({
        "schema": v22.STATE_SCHEMA,
        "model_version": v22.MODEL_VERSION,
        "model_sha256": "b" * 64,
        "feature_schema_sha256": "c" * 64,
        "strategy_schema_sha256": "d" * 64,
        "candidate_lock_sha256": predecessor_lock,
        "training_data_sha256": "e" * 64,
        "manifest_effective_end": 100,
        "pairs": {"BTC-FDUSD": {"gate_state": {"active": True}}},
    }), encoding="utf-8")
    expected = {
        "model_sha256": "f" * 64,
        "feature_schema_sha256": "c" * 64,
        "strategy_schema_sha256": "1" * 64,
        "candidate_lock_sha256": "2" * 64,
        "training_data_sha256": "3" * 64,
        "manifest_effective_end": 200,
    }

    loaded = load_state(
        state_path, expected, rollover_from_lock_sha256=predecessor_lock,
    )

    assert loaded["pairs"]["BTC-FDUSD"]["gate_state"]["active"] is True
    assert loaded["strategy_schema_sha256"] == "1" * 64
    assert loaded["rollover_strategy_changed"] is True


def test_rollover_with_changed_strategy_still_requires_exact_predecessor(tmp_path: Path) -> None:
    state_path = tmp_path / "state.json"
    state_path.write_text(json.dumps({
        "schema": v22.STATE_SCHEMA,
        "model_version": v22.MODEL_VERSION,
        "model_sha256": "b" * 64,
        "feature_schema_sha256": "c" * 64,
        "strategy_schema_sha256": "d" * 64,
        "candidate_lock_sha256": "a" * 64,
        "training_data_sha256": "e" * 64,
        "manifest_effective_end": 100,
        "pairs": {},
    }), encoding="utf-8")
    expected = {
        "model_sha256": "f" * 64,
        "feature_schema_sha256": "c" * 64,
        "strategy_schema_sha256": "1" * 64,
        "candidate_lock_sha256": "2" * 64,
        "training_data_sha256": "3" * 64,
        "manifest_effective_end": 200,
    }

    with pytest.raises(ValueError, match="state hash/manifest mismatch"):
        load_state(state_path, expected, rollover_from_lock_sha256="9" * 64)


def test_frozen_weekly_bundle_is_contiguous_and_hash_locked() -> None:
    lock = json.loads((PACKAGE / "shadow_lock.json").read_text(encoding="utf-8"))
    bundle = joblib.load(MODEL); v22.validate_weekly_bundle(bundle)
    assert research.sha256_file(MODEL) == lock["model_sha256"]
    assert bundle["strategy_schema_sha256"] == lock["strategy_schema_sha256"]
    for pair in v22.PAIRS:
        weeks = bundle["pairs"][pair]["weeks"]
        assert len(weeks) == lock["weeks_per_pair"] == 36
        assert all(left["test_end"] == right["test_start"] for left, right in zip(weeks, weeks[1:]))
        assert max(item["last_label_ready_ts"] - item["train_cutoff"] for item in weeks) <= 0
        assert max(item["cached_probability_max_abs_error"] for item in weeks) <= lock["retrain_probability_tolerance"]


def test_v22_weekly_bundle_exactly_matches_old_plot_states_and_events() -> None:
    bundle = joblib.load(MODEL)
    panel = pd.read_csv(ROOT / "results/backtests/xgboost_grid_long_risk_gate_v19_250d/feature_panel.csv.gz")
    old = pd.read_csv(ROOT / "results/backtests/xgboost_grid_long_risk_gate_v21_250d/final_risk_states.csv.gz")
    for pair in v22.PAIRS:
        rows = panel[(panel.pair == pair) & panel.signal_ts.between(
            research.START_TS, research.END_TS, inclusive="left")]
        actual, _ = v22.run_weekly_bundle_strategy(rows, pair=pair, pair_bundle=bundle["pairs"][pair])
        wanted = old[old.pair.eq(pair)].sort_values("signal_ts")
        actual = actual.sort_values("signal_ts")
        assert np.array_equal(actual.signal_ts.to_numpy(), wanted.signal_ts.to_numpy())
        assert np.array_equal(actual.risk_off_active.to_numpy(), wanted.risk_off_active.to_numpy())
        assert np.array_equal(actual.transition.to_numpy(), wanted.transition.to_numpy())
        assert np.max(np.abs(actual.probability.to_numpy() - wanted.probability.to_numpy())) <= 1e-6
        assert np.max(np.abs(actual.entry_threshold.to_numpy() - wanted.entry_threshold.to_numpy())) <= 1e-8


def test_unsigned_future_week_is_rejected_instead_of_using_last_model() -> None:
    bundle = joblib.load(MODEL); pair_bundle = bundle["pairs"]["BTC-FDUSD"]
    with pytest.raises(RuntimeError, match="no unique signed weekly model"):
        v22.week_for_timestamp(pair_bundle, int(pair_bundle["weeks"][-1]["test_end"]))


def test_failed_contract_is_buy_disabled_for_both_pairs(tmp_path: Path) -> None:
    now = int(datetime.now(timezone.utc).timestamp()); path = tmp_path / "failed.json"
    contract.atomic_json(path, contract.failed_contract("missing signed week", now))
    loaded = contract.load_shadow_contract(path, now=datetime.fromtimestamp(now, timezone.utc))
    assert loaded["source_healthy"] is False
    assert all(loaded["pairs"][pair]["buy_enabled"] is False for pair in v22.PAIRS)
    assert all(loaded["pairs"][pair]["long"]["risk_off_active"] is True for pair in v22.PAIRS)


def test_failed_live_contract_preserves_primary_reason_without_float_none(tmp_path: Path) -> None:
    now = int(datetime.now(timezone.utc).timestamp())
    path = tmp_path / "failed-live.json"
    value = live_contract.failed_contract(
        generated_at=now,
        reason="no signed weekly model covers current signal",
        metadata={
            "release_sha256": "a" * 64, "model_sha256": "b" * 64,
            "feature_schema_sha256": "c" * 64, "strategy_schema_sha256": "d" * 64,
            "training_data_sha256": "e" * 64,
        },
    )
    live_contract.atomic_json(path, value)
    loaded = live_contract.load_runtime_contract(
        path, now=datetime.fromtimestamp(now, timezone.utc),
    )
    assert loaded["runtime_gate_healthy"] is False
    assert loaded["reason"] == "no signed weekly model covers current signal"
    assert all(item["force_exit"] for item in loaded["pairs"].values())

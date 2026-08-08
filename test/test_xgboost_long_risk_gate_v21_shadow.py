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
sys.path.insert(0, str(ROOT / "scripts"))

import grid_xgboost_risk_gate as live_v16  # noqa: E402
import grid_xgboost_shadow_gate_v21 as contract  # noqa: E402
import xgboost_long_risk_gate_v21 as v21  # noqa: E402
from build_xgboost_v21_shadow_signal import load_state  # noqa: E402
from compare_independent_gate_ml_stops import load_candles  # noqa: E402


PACKAGE = ROOT / "results/backtests/xgboost_grid_long_risk_gate_v21_250d/shadow_package"


def _advance(state: v21.GateState, ts: int, structure, probability=1.0, pair="BTC-FDUSD"):
    return v21.advance_gate(
        pair=pair, probability=probability, entry_threshold=.5, signal_ts=ts,
        last_complete_4h_ts=ts - ts % (4 * v21.HOUR), structure=structure,
        state=state,
    )


def test_frozen_feature_schema_and_gate_configuration() -> None:
    assert len(v21.BTC_FEATURES) == 23
    assert len(v21.ETH_FEATURES) == 15
    assert len(set(v21.BTC_FEATURES)) == 23
    assert len(set(v21.ETH_FEATURES)) == 15
    assert v21.GATES["BTC-FDUSD"] == v21.GateConfig(.98, 1, 48, 48, 48, 3)
    assert v21.GATES["ETH-FDUSD"] == v21.GateConfig(.985, 2, 48, 48, 24, 3)
    lock = json.loads((PACKAGE / "shadow_lock.json").read_text(encoding="utf-8"))
    assert lock["feature_schema_sha256"] == v21.feature_schema_sha256()
    assert lock["pairs"]["BTC-FDUSD"]["features"] == list(v21.BTC_FEATURES)
    assert lock["pairs"]["ETH-FDUSD"]["features"] == list(v21.ETH_FEATURES)


def test_live_and_research_feature_rows_match_to_numerical_precision() -> None:
    candle_dir = ROOT / "results/backtests/eth_xgboost_long_risk_gate_v15_250d/extended_candles"
    candles, _ = load_candles(candle_dir)
    live = v21.build_inference_panel(candles)
    research = pd.read_csv(
        ROOT / "results/backtests/xgboost_grid_long_risk_gate_v19_250d/feature_panel.csv.gz"
    )
    for pair in v21.PAIRS:
        left = live[live.pair.eq(pair)].set_index("signal_ts")
        right = research[research.pair.eq(pair)].set_index("signal_ts")
        common = left.index.intersection(right.index)
        assert len(common) > 6000
        for feature in v21.FEATURES[pair]:
            error = np.nanmax(np.abs(
                left.loc[common, feature].to_numpy(float)
                - right.loc[common, feature].to_numpy(float)
            ))
            assert error <= 1e-12, (pair, feature, error)


def test_entry_threshold_is_inclusive_but_requires_complete_bearish_4h() -> None:
    state = v21.GateState()
    # At the threshold the model arms, but a non-bearish structure cannot enter.
    state, snap = _advance(state, 4 * v21.HOUR, (1, 1, 1, 1, .1), probability=.5)
    assert snap["armed"] and not snap["risk_off_active"]
    state, snap = _advance(state, 8 * v21.HOUR, (-1, -1, -1, -1, .8), probability=.1)
    assert snap["transition"] == "enter"
    assert snap["risk_off_active"] and snap["buy_enabled"] is False


def test_eth_requires_two_probability_bars_and_is_independent() -> None:
    eth, btc = v21.GateState(), v21.GateState()
    bearish = (-1, -1, -1, -1, .8)
    eth, first = _advance(eth, 4 * v21.HOUR, bearish, pair="ETH-FDUSD")
    assert not first["risk_off_active"] and first["above_entry_count"] == 1
    eth, second = _advance(eth, 5 * v21.HOUR, bearish, pair="ETH-FDUSD")
    assert second["above_entry_count"] == 2 and not second["risk_off_active"]
    eth, third = _advance(eth, 8 * v21.HOUR, bearish, probability=0, pair="ETH-FDUSD")
    assert third["risk_off_active"]
    assert not btc.active and btc.last_signal_ts is None


def test_low_probability_alone_cannot_recover_and_three_4h_relief_bars_can() -> None:
    state = v21.GateState()
    bearish = (-4, -4, -1, -1, .8)
    state, _ = _advance(state, 0, bearish)
    assert state.active
    # Many low-probability hours with no structural relief do not recover.
    for hour in range(1, 49):
        state, _ = _advance(state, hour * v21.HOUR, bearish, probability=0)
    assert state.active
    # Ordinary (not strong) relief needs three distinct complete 4h structures.
    for index, hour in enumerate((52, 56, 60), start=1):
        current = (-4 + index, -4 + index, 1, .1, .6)
        state, snap = _advance(state, hour * v21.HOUR, current, probability=0)
    assert snap["transition"] == "recover" and not state.active
    assert state.cooldown_until == 108 * v21.HOUR


def test_all_condition_strong_relief_shortens_recovery_to_two_bars() -> None:
    state = v21.GateState()
    state, _ = _advance(state, 0, (-2, -2, -1, -1, .8))
    state, first = _advance(state, 48 * v21.HOUR, (.1, .1, 1, .1, .4), probability=0)
    assert state.active and first["structure_recovery_count"] == 1
    state, second = _advance(state, 52 * v21.HOUR, (.2, .2, 1, .2, .3), probability=0)
    assert second["transition"] == "recover" and not state.active


def test_duplicate_or_late_hour_is_idempotent_and_state_roundtrips() -> None:
    state, first = _advance(v21.GateState(), 0, (-1, -1, -1, -1, .8))
    encoded = v21.state_to_dict(state)
    restored = v21.state_from_dict(encoded)
    restored, duplicate = _advance(restored, 0, (1, 1, 1, 1, .1), probability=0)
    assert duplicate["transition"] == "duplicate"
    assert restored.last_event_id == first["event_id"]
    assert restored.active and restored.previous_structure == (-1, -1, -1, -1, .8)
    assert restored.probability_history == [(0, 1.0)]
    assert restored.structure_history == [(0, (-1, -1, -1, -1, .8))]


def test_state_hash_mismatch_is_not_silently_reset(tmp_path: Path) -> None:
    lock = {"model_sha256": "a" * 64, "feature_schema_sha256": "b" * 64,
            "strategy_schema_sha256": "f" * 64,
            "candidate_lock_sha256": "d" * 64, "training_panel_sha256": "e" * 64}
    path = tmp_path / "state.json"
    path.write_text(json.dumps({"schema": v21.STATE_SCHEMA, "model_version": v21.MODEL_VERSION,
                                "model_sha256": "c" * 64, "feature_schema_sha256": "b" * 64,
                                "strategy_schema_sha256": "f" * 64,
                                "candidate_lock_sha256": "d" * 64,
                                "training_data_sha256": "e" * 64, "pairs": {}}))
    with pytest.raises(ValueError, match="hash mismatch"):
        load_state(path, lock)


def _snap(pair: str, active=False):
    return {"pair": pair, "probability": .2, "entry_threshold": .5,
            "risk_off_active": active, "recommended_buy_enabled": not active,
            "transition": "hold", "event_id": "e"}


def test_v2_contract_is_non_authorizing_and_rejected_by_current_v16(tmp_path: Path) -> None:
    now = 2_000_000_000
    digest = "a" * 64
    payload = contract.build_contract(
        generated_at=now, model_sha256=digest, feature_sha256=digest,
        strategy_sha256=digest,
        training_data_sha256=digest, candidate_lock_sha256=digest,
        state_sha256=digest, source_healthy=True,
        pair_snapshots={pair: _snap(pair) for pair in v21.PAIRS},
        last_complete_1h={pair: now - 300 for pair in v21.PAIRS},
        last_complete_4h={pair: now - 3600 for pair in v21.PAIRS},
    )
    path = tmp_path / "xgboost_risk_gate_v21_shadow.json"
    contract.atomic_json(path, payload)
    loaded = contract.load_shadow_contract(path, now=datetime.fromtimestamp(now, timezone.utc))
    assert loaded["shadow_contract_healthy"]
    assert loaded["deployment_allowed"] is False and loaded["promotion_authorized"] is False
    assert loaded["market_sell_action"] is False and loaded["mechanism1_fallback_allowed"] is False
    assert loaded["short_spike_enabled"] is False
    assert all(loaded["pairs"][pair]["long"]["buy_enabled"] is False for pair in v21.PAIRS)
    assert all(loaded["pairs"][pair]["buy_enabled"] is False for pair in v21.PAIRS)
    assert all(loaded["pairs"][pair]["risk_off_active"] ==
               loaded["pairs"][pair]["long"]["risk_off_active"] for pair in v21.PAIRS)
    rejected = live_v16.load_runtime_xgboost_gate(path, now=datetime.fromtimestamp(now, timezone.utc))
    assert rejected["runtime_gate_healthy"] is False
    assert all(not rejected["pairs"][pair]["buy_enabled"] for pair in v21.PAIRS)


def test_contract_stale_and_future_heartbeat_are_rejected(tmp_path: Path) -> None:
    now = 2_000_000_000
    digest = "a" * 64
    payload = contract.build_contract(
        generated_at=now, model_sha256=digest, feature_sha256=digest,
        strategy_sha256=digest,
        training_data_sha256=digest, candidate_lock_sha256=digest,
        state_sha256=digest, source_healthy=True,
        pair_snapshots={pair: _snap(pair) for pair in v21.PAIRS},
        last_complete_1h={pair: now for pair in v21.PAIRS},
        last_complete_4h={pair: now for pair in v21.PAIRS},
    )
    path = tmp_path / "signal.json"
    contract.atomic_json(path, payload)
    with pytest.raises(ValueError, match="stale or in the future"):
        contract.load_shadow_contract(path, now=datetime.fromtimestamp(now + 151, timezone.utc))
    with pytest.raises(ValueError, match="stale or in the future"):
        contract.load_shadow_contract(path, now=datetime.fromtimestamp(now - 11, timezone.utc))


def test_frozen_models_serialize_exactly_and_return_finite_probabilities() -> None:
    lock = json.loads((PACKAGE / "shadow_lock.json").read_text(encoding="utf-8"))
    assert lock["historical_verdict"] == "NO-GO"
    assert lock["deployment_allowed"] is False and lock["promotion_authorized"] is False
    assert lock["serialization_check"]["maximum_probability_absolute_error"] <= 1e-12
    bundle = joblib.load(ROOT / lock["model_path"])
    v21.validate_strategy_bundle(bundle)
    assert lock["strategy_schema_sha256"] == v21.strategy_schema_sha256()
    for pair in v21.PAIRS:
        model = bundle["pairs"][pair]["model"]
        probability = model.predict_proba(np.zeros((2, len(v21.FEATURES[pair]))))[:, 1]
        assert np.isfinite(probability).all()
        assert np.logical_and(probability >= 0, probability <= 1).all()


def test_v21_is_merged_into_guard_without_a_standalone_container() -> None:
    compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    assert "  grid-xgboost-v21-shadow:" not in compose
    assert "  grid-xgboost-risk-gate:" not in compose
    assert "GRID_V21_IN_GUARD_ENABLED: \"true\"" in compose
    assert "/workspace/package:ro" in compose
    assert "grid-live-guard-v21-candles:/workspace/v21-candles" in compose
    dockerfile = (ROOT / "Dockerfile.grid-live-guard").read_text(encoding="utf-8")
    assert "requirements-grid-xgboost.txt" in dockerfile
    assert "COPY scripts /app" in dockerfile
    requirements = (ROOT / "requirements-grid-xgboost.txt").read_text(encoding="utf-8")
    assert "xgboost==3.3.0" in requirements


def test_shared_online_state_machine_matches_v21_250d_offline_replay() -> None:
    research = ROOT / "results/backtests/xgboost_grid_long_risk_gate_v19_250d/prediction_cache/weekly"
    sources = {
        "BTC-FDUSD": research / "BTC-FDUSD__long_event_72h__full_structure__xgb_34.csv.gz",
        "ETH-FDUSD": research / "ETH-FDUSD__long_event_72h__directional_persistence__xgb_16.csv.gz",
    }
    expected = pd.read_csv(
        ROOT / "results/backtests/xgboost_grid_long_risk_gate_v21_250d/final_risk_states.csv.gz"
    )
    actual = []
    for pair, source in sources.items():
        rows = pd.read_csv(source).sort_values("signal_ts")
        state = v21.GateState()
        qcol = f"threshold_q{int(round(v21.GATES[pair].entry_quantile * 10000)):04d}"
        for row in rows.itertuples(index=False):
            state, snapshot = v21.advance_gate(
                pair=pair, probability=float(row.probability),
                entry_threshold=float(getattr(row, qcol)), signal_ts=int(row.signal_ts),
                last_complete_4h_ts=int(row.last_complete_4h_ts),
                structure=(row.roc_48h_4h, row.sqzmom_pct_4h, row.di_spread,
                           row.ema20_slope_atr_12h, row.below_ema20_ratio_72h), state=state,
            )
            actual.append({"pair": pair, "signal_ts": int(row.signal_ts), **snapshot})
    actual_frame = pd.DataFrame(actual).sort_values(["pair", "signal_ts"]).reset_index(drop=True)
    expected = expected.sort_values(["pair", "signal_ts"]).reset_index(drop=True)
    assert actual_frame["recommended_buy_enabled"].tolist() == expected["buy_enabled"].tolist()
    assert not actual_frame["buy_enabled"].any()  # public shadow safety value
    for field in ("risk_off_active", "armed", "entry_structure_confirmed",
                  "recovery_structure_confirmed", "structure_recovery_count", "transition"):
        assert actual_frame[field].tolist() == expected[field].tolist(), field


def test_application_plotly_states_are_exact_frozen_bundle_outputs() -> None:
    lock = json.loads((PACKAGE / "shadow_lock.json").read_text(encoding="utf-8"))
    bundle = joblib.load(ROOT / lock["model_path"])
    panel = pd.read_csv(
        ROOT / "results/backtests/xgboost_grid_long_risk_gate_v19_250d/feature_panel.csv.gz"
    )
    start = int(pd.Timestamp("2025-11-23T15:00:00Z").timestamp())
    end = int(pd.Timestamp("2026-07-31T15:00:00Z").timestamp())
    panel = panel[panel.signal_ts.between(start, end, inclusive="left")]
    expected = pd.read_csv(
        ROOT / "results/backtests/xgboost_grid_long_risk_gate_v21_250d/application_bundle/risk_states.csv.gz"
    )
    for pair in v21.PAIRS:
        actual, _ = v21.run_bundle_strategy(
            panel[panel.pair.eq(pair)], pair=pair, pair_bundle=bundle["pairs"][pair]
        )
        wanted = expected[expected.pair.eq(pair)].sort_values("signal_ts").reset_index(drop=True)
        actual = actual.sort_values("signal_ts").reset_index(drop=True)
        assert actual.signal_ts.tolist() == wanted.signal_ts.tolist()
        assert np.max(np.abs(actual.probability - wanted.probability)) <= 1e-12
        assert np.max(np.abs(actual.entry_threshold - wanted.entry_threshold)) <= 1e-15
        for field in ("risk_off_active", "recommended_buy_enabled",
                      "above_entry_count", "armed", "structure_recovery_count", "transition"):
            assert actual[field].tolist() == wanted[field].tolist(), (pair, field)

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from tune_xgboost_momentum_stop_v2 import (  # noqa: E402
    ALL_FEATURES,
    DAY,
    LOCK_SCHEMA,
    MODEL_VERSION,
    SEPARATE_FEATURES,
    canonical_json,
    fit_one_group,
    sha256_bytes,
    split_mature_training,
    validate_lock,
    xgb_configurations,
)


def test_approved_40_configurations_are_deterministic_unique_and_include_anchors():
    configurations = xgb_configurations()
    assert len(configurations) == 40
    parameter_keys = {
        canonical_json({
            key: value for key, value in item.items()
            if key not in {"config_id", "order", "kind", "uses_early_stopping"}
        })
        for item in configurations
    }
    assert len(parameter_keys) == 40
    assert configurations[0] == {
        "config_id": "xgb_00", "order": 0, "kind": "legacy",
        "uses_early_stopping": False, "learning_rate": 0.04,
        "n_estimators": 240, "max_depth": 5, "min_child_weight": 15,
        "subsample": 0.85, "colsample_bytree": 0.85, "gamma": 0.0,
        "reg_alpha": 0.0, "reg_lambda": 1.0, "max_bin": 256,
    }
    assert configurations[1]["kind"] == "regularized_anchor"
    assert configurations[1]["n_estimators"] == 800
    assert configurations[1]["max_depth"] == 3
    assert configurations[1]["reg_lambda"] == 8.0
    assert sha256_bytes(canonical_json(configurations).encode()) == (
        "3eb4f24a2e4739a0f24bd0e4d09eaeded1c701eec45822c7533355fa76e3523e"
    )


def test_separate_models_remove_pair_identity_feature_only():
    assert "pair_is_eth" in ALL_FEATURES
    assert "pair_is_eth" not in SEPARATE_FEATURES
    assert set(ALL_FEATURES).difference(SEPARATE_FEATURES) == {"pair_is_eth"}


def test_internal_early_stop_split_is_mature_and_strictly_before_validation():
    cutoff = 100 * DAY
    signal_ts = np.arange(1, 90) * DAY
    panel = pd.DataFrame({
        "signal_ts": signal_ts,
        "label_ready_ts": signal_ts + 6 * 3600,
        "target": (np.arange(len(signal_ts)) % 2).astype(float),
    })
    mature, core, validation = split_mature_training(panel, cutoff)
    assert mature.label_ready_ts.max() <= cutoff
    assert core.signal_ts.max() < validation.signal_ts.min()
    assert validation.signal_ts.min() >= cutoff - 14 * DAY


def test_nonlegacy_early_stop_then_full_mature_refit_and_pickle_roundtrip():
    rng = np.random.default_rng(42)
    rows = 240
    frame = pd.DataFrame({"target": np.tile([0.0, 1.0], rows // 2)})
    for feature in ALL_FEATURES:
        frame[feature] = rng.normal(size=rows)
    core, validation = frame.iloc[:160], frame.iloc[160:]
    config = dict(xgb_configurations()[1])
    config["n_estimators"] = 20
    model, audit = fit_one_group(config, list(ALL_FEATURES), frame, core, validation)
    assert audit["early_stopping_used"] is True
    assert 1 <= audit["best_tree_count"] <= 20
    assert model.get_params()["n_estimators"] == audit["best_tree_count"]
    probability = model.predict_proba(frame[list(ALL_FEATURES)])[:, 1]
    assert np.isfinite(probability).all()
    assert np.logical_and(probability >= 0, probability <= 1).all()
    import pickle
    restored = pickle.loads(pickle.dumps(model))
    np.testing.assert_allclose(
        probability, restored.predict_proba(frame[list(ALL_FEATURES)])[:, 1],
        rtol=0, atol=1e-12,
    )


def test_revalidate_refuses_missing_lock(tmp_path: Path):
    args = type("Args", (), {"output_dir": tmp_path})
    with pytest.raises(RuntimeError, match="locked_configuration.json is missing"):
        validate_lock(args, {}, pd.DataFrame())


def test_revalidate_refuses_hash_mismatch_before_reading_results(tmp_path: Path):
    lock = {
        "schema": LOCK_SCHEMA,
        "model_version": MODEL_VERSION,
        "hashes": {"feature_panel_sha256": "expected"},
        "revalidation_policy": {"configuration_switching_after_lock": False},
    }
    (tmp_path / "locked_configuration.json").write_text(json.dumps(lock), encoding="utf-8")
    args = type("Args", (), {"output_dir": tmp_path})
    with pytest.raises(RuntimeError, match="input hash mismatch"):
        validate_lock(
            args, {"feature_panel_sha256": "actual"},
            pd.DataFrame(columns=["pair", "signal_ts", *ALL_FEATURES]),
        )

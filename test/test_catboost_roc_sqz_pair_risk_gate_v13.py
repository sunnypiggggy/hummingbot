from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str((Path(__file__).resolve().parents[1] / "scripts").resolve()))

import optimize_catboost_roc_sqz_pair_risk_gate_v13 as cat


def test_catboost_configuration_grid_is_deterministic_and_unique() -> None:
    first = cat.catboost_configurations()
    second = cat.catboost_configurations()
    assert first == second
    assert len(first) == 40
    assert len({item["config_id"] for item in first}) == 40
    assert len({cat.canonical({k: v for k, v in item.items() if k not in {"config_id", "order", "kind", "uses_early_stopping"}}) for item in first}) == 40
    assert first[0]["kind"] == "legacy_anchor"
    assert first[1]["kind"] == "regularized_anchor"


def test_feature_contract_contains_only_roc_and_sqzmom_derivatives() -> None:
    features = cat.engine.ROC_SQZ_FEATURES
    assert features
    assert all(("roc" in feature) or ("sqz" in feature) for feature in features)


def test_catboost_fit_returns_finite_probabilities() -> None:
    rng = np.random.default_rng(42)
    features = ["roc_5", "sqzmom_value"]
    frame = pd.DataFrame(rng.normal(size=(320, 2)), columns=features)
    frame["target"] = ((frame.roc_5 + 0.4 * frame.sqzmom_value) < 0).astype(int)
    config = {**cat.catboost_configurations()[1], "iterations": 20}
    model, audit = cat.fit_catboost(
        config, features, frame, frame.iloc[:240], frame.iloc[240:],
    )
    probability = model.predict_proba(frame[features])[:, 1]
    assert np.isfinite(probability).all()
    assert ((probability >= 0) & (probability <= 1)).all()
    assert 1 <= audit["best_tree_count"] <= 20

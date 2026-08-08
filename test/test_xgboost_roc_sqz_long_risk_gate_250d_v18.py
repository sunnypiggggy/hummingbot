from __future__ import annotations

import retrain_xgboost_roc_sqz_long_risk_gate_250d_v18 as v18


def test_v18_uses_only_roc_sqz_features() -> None:
    allowed_fragments = ("roc", "sqz")
    for features in v18.engine.FEATURE_SETS.values():
        assert 3 <= len(features) <= 10
        assert all(any(fragment in feature for fragment in allowed_fragments) for feature in features)


def test_v18_has_independent_deterministic_640_model_screen() -> None:
    specs = v18.engine.specs()
    assert len(specs) == 640
    assert {item["pair"] for item in specs} == {"BTC-FDUSD", "ETH-FDUSD"}
    assert {item["target"] for item in specs} == {"long_72h", "long_120h"}


def test_v18_has_separate_non_deploying_artifact_namespace() -> None:
    assert "v18" in v18.engine.MODEL_VERSION
    assert "v18" in v18.engine.OUTPUT_DIR.as_posix()

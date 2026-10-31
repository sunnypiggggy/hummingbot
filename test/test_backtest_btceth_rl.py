"""Selection and allocation controls, without training or external services."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from scripts.backtest_btceth_rl import (
    ALGORITHMS, COSTS, PROFILES, SEEDS, baseline_signal_table, baseline_weights,
    VALIDATION, hyperparameters, job_id, select_validation, sha, validation_score,
)


def test_risk_balanced_selection_accounts_for_drawdown_and_both_costs():
    good = {"net_return": .08, "max_drawdown": -.10}
    risky = {"net_return": .08, "max_drawdown": -.30}
    stressed = {"net_return": -.20, "max_drawdown": -.30}
    assert validation_score(good, good) > validation_score(risky, risky)
    assert validation_score(good, good) > validation_score(good, stressed)
    assert validation_score(good, stressed) == validation_score(stressed, good)


def test_hyperparameters_are_fresh_and_do_not_mutate_another_model():
    first = hyperparameters("tqc")
    first["policy_kwargs"]["net_arch"][0] = 1
    assert hyperparameters("tqc")["policy_kwargs"]["net_arch"] == [128, 128]
    assert "n_quantiles" not in hyperparameters("sac")["policy_kwargs"]


@pytest.mark.parametrize("kind", ["sma200_trend", "momentum180"])
def test_baseline_never_reads_current_or_future_day(kind):
    hours = pd.date_range("2023-01-01", periods=370 * 24, freq="h", tz="UTC")
    close = pd.DataFrame({"BTCUSDT": np.linspace(100., 200., len(hours)),
                          "ETHUSDT": np.linspace(20., 10., len(hours))}, index=hours)
    day = pd.Timestamp("2023-10-01", tz="UTC")
    original = {"hourly_close": close}
    altered = {"hourly_close": close.copy()}
    altered["hourly_close"].loc[day:] *= 100
    np.testing.assert_array_equal(baseline_weights(original, day, kind),
                                  baseline_weights(altered, day, kind))
    target = baseline_weights(original, day, kind)
    np.testing.assert_allclose(target, [.95, 0., .05])
    prepared = baseline_signal_table(original, kind)
    np.testing.assert_array_equal(target, baseline_weights(original, day, kind, prepared))


@pytest.mark.parametrize("kind,expected", [("cash", [0., 0., 1.]),
    ("btc_hold", [.95, 0., .05]), ("eth_hold", [0., .95, .05]),
    ("equal_hold", [.475, .475, .05]), ("equal_rebalance", [.475, .475, .05])])
def test_baselines_have_same_capital_constraints(kind, expected):
    np.testing.assert_allclose(baseline_weights({}, "2025-01-01", kind), expected)


def write_validation(root, *, damage=None):
    (root / "frozen_spec.json").write_text(json.dumps({"fixed": True}), encoding="utf-8")
    count = len(ALGORITHMS) * len(PROFILES) * len(SEEDS)
    (root / "training_completion.json").write_text(json.dumps({"model_count": count}), encoding="utf-8")
    for algorithm in ALGORITHMS:
        for profile in PROFILES:
            for seed in SEEDS:
                training = root / "training" / job_id(algorithm, profile, seed)
                training.mkdir(parents=True)
                model = training / "model.zip"
                model.write_bytes(b"dedicated-fake-test-checkpoint")
                (training / "record.json").write_text(json.dumps(dict(algorithm=algorithm, profile=profile,
                    seed=seed, model_relative_path=str(model.relative_to(root)), model_sha256=sha(model))), encoding="utf-8")
            directory = root / "evaluation" / "validation" / algorithm / profile
            directory.mkdir(parents=True)
            rows = []
            for scenario in COSTS:
                for seed in SEEDS:
                    rows.append(dict(seed=seed, scenario=scenario, primary_policy=False,
                                     net_return=.1, max_drawdown=-.1))
                rows.append(dict(seed=np.nan, scenario=scenario, primary_policy=True,
                                 net_return=.10 if profile == "balanced" else .09,
                                 max_drawdown=-.1))
            if damage and algorithm == ALGORITHMS[0] and profile == next(iter(PROFILES)):
                damage(rows)
            frame = pd.DataFrame(rows)
            frame["split"], frame["algorithm"], frame["profile"] = "validation", algorithm, profile
            frame["start"], frame["end_exclusive"] = pd.Timestamp(VALIDATION[0]).isoformat(), pd.Timestamp(VALIDATION[1]).isoformat()
            frame["initial_capital"] = 10000.
            frame.to_csv(directory / "summary.csv", index=False)


def test_validation_selects_profiles_without_any_test_directory(tmp_path):
    write_validation(tmp_path)
    selected = select_validation(tmp_path)
    assert selected["selected_profiles"] == {algorithm: "balanced" for algorithm in ALGORITHMS}
    assert not selected["selection_uses_test"]
    assert not (tmp_path / "evaluation" / "test").exists()
    assert len(selected["validation_source_sha256"]) == len(ALGORITHMS) * len(PROFILES)
    assert len(selected["training_source_sha256"]) == 1 + 2 * len(ALGORITHMS) * len(PROFILES) * len(SEEDS)


def test_selector_rejects_missing_seed_with_duplicate_cost_row(tmp_path):
    def damage(rows):
        # Total rows and overall seed set stay unchanged. Per-cost integrity
        # must still reject a missing base seed and duplicated stress seed.
        rows[0]["scenario"] = "stress"
    write_validation(tmp_path, damage=damage)
    with pytest.raises(ValueError, match="Each cost scenario"):
        select_validation(tmp_path)


def test_selector_cannot_run_after_a_test_attempt(tmp_path):
    write_validation(tmp_path)
    (tmp_path / "test_started.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="precede"):
        select_validation(tmp_path)


def test_selector_rejects_changed_checkpoint(tmp_path):
    write_validation(tmp_path)
    model = next((tmp_path / "training").glob("*/model.zip"))
    model.write_bytes(b"changed-fake-checkpoint")
    with pytest.raises(ValueError, match="reconcile"):
        select_validation(tmp_path)


def test_selector_rejects_wrong_validation_dates(tmp_path):
    write_validation(tmp_path)
    path = next((tmp_path / "evaluation").glob("validation/*/*/summary.csv"))
    frame = pd.read_csv(path)
    frame["start"] = "2025-02-01T00:00:00+00:00"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="identity mismatch"):
        select_validation(tmp_path)

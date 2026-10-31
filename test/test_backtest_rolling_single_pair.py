"""Checks for actual schedule, physical data isolation and existing selection."""
from copy import deepcopy
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import pytest

from scripts import backtest_rolling_single_pair as runner

ROOT = Path(__file__).resolve().parents[1]


def spec():
    return dict(selected=deepcopy(runner.SELECTED), recipes={
        k: {"algorithm": a} for k, a in (("sac_default", "sac"), ("sac_gamma099", "sac"),
                                        ("td3_default", "td3"), ("tqc_default", "tqc"))},
        seeds=list(range(5)), test=list(runner.TEST), lag_days=1, train_days=1827)


@pytest.mark.parametrize("cadence,versions", [("fixed", {0}), ("every_4w", set(range(0, 53, 4))),
                                             ("every_week", set(range(53)))])
def test_real_calendar(cadence, versions):
    days = pd.date_range(*runner.TEST, inclusive="left", freq="D")
    actual = [runner.version_for_day(day, runner.TEST[0], cadence) for day in days]
    assert len(actual) == 365 and set(actual) == versions
    for day, version in zip(days, actual):
        assert pd.Timestamp(runner.TEST[0]) + pd.Timedelta(days=7 * version) <= day


def test_unique_models_shared_by_three_arms(tmp_path):
    (tmp_path / "protocol.json").write_text("{}")
    values = runner.contexts(tmp_path, spec())
    assert len(values) == 1590
    assert len({str(runner.model_directory(tmp_path, c)) for c in values}) == 1590
    for c in values:
        assert pd.Timestamp(c["train_end"]) + pd.Timedelta(days=1) == pd.Timestamp(c["effective_day"])
        assert (pd.Timestamp(c["train_end"]) - pd.Timestamp(c["train_start"])).days == 1827


def test_crop_physically_removes_future_other_coin_and_old_warmup():
    index = pd.date_range("2019-01-01", "2026-10-24", freq="h", tz="UTC")
    frame = pd.DataFrame({"BTCUSDT": np.arange(len(index)) + 1., "ETHUSDT": 2.}, index=index)
    data = {"hourly_" + f: frame.copy() for f in ("open", "high", "low", "close", "volume")}
    context = runner.model_context(spec(), "a" * 64, "BTCUSDT", "sac_default", 0, 0)
    cropped = runner.training_data(data, context)
    original_digest = runner.input_market_digest(cropped, "BTCUSDT")
    for f in data:
        data[f].loc[data[f].index >= pd.Timestamp(context["train_end"]), "BTCUSDT"] = -999.
        data[f]["ETHUSDT"] = -777.
    second = runner.training_data(data, context)
    assert original_digest == runner.input_market_digest(second, "BTCUSDT")
    for f in second:
        if f.startswith("hourly_"):
            assert list(second[f].columns) == ["BTCUSDT"]
            assert second[f].index.max() < pd.Timestamp(context["train_end"])
            assert second[f].index.min() == pd.Timestamp(context["train_start"]) - pd.Timedelta(days=200)


@pytest.mark.parametrize("version", [-1, 53, True, 1.2])
def test_reject_undeclared_version(version):
    with pytest.raises(ValueError):
        runner.model_context(spec(), "a" * 64, "BTCUSDT", "sac_default", 0, version)


def selection_inputs(tmp_path):
    research = ROOT / "results/backtests"
    target = tmp_path / "selection_inputs"
    target.mkdir()
    native = research / "single_pair_rl_six_20261030"
    shutil.copyfile(native / "protocol.json", target / "native_protocol.json")
    shutil.copyfile(native / "validation_summary.csv", target / "native_validation_summary.csv")
    for c in ("c01_lr1e4", "c02_gamma099", "c03_lr1e4_gamma099"):
        shutil.copyfile(research / "sac_validation_optimization_20261031/tuning" / c / "validation_summary.csv",
                        target / (c + "_validation_summary.csv"))
    source = ROOT / "results/analysis/all_experiments_rolling_20261031/existing_experiments_inventory_v2.json"
    shutil.copyfile(source, tmp_path / "inventory.json")
    return runner.read(tmp_path / "inventory.json")


def test_actual_validation_only_shortlist_no_annual_access(tmp_path):
    inventory = selection_inputs(tmp_path)
    result = runner.selection_from_validation(tmp_path, inventory)
    assert result["selected"] == runner.SELECTED
    assert result["uses_test"] is False and result["seed_selection"] is False
    assert len(result["candidates"]) == 18
    assert not any("test_" in name or "annual" in name for name in result["source_sha256"])
    cashlike = [r for r in result["candidates"] if r["cashlike_degenerate"]]
    assert len(cashlike) == 2 and all(r["algorithm"] == "recurrent_ppo" for r in cashlike)


def test_annual_rows_cannot_be_used_for_selection(tmp_path):
    inventory = selection_inputs(tmp_path)
    path = tmp_path / "selection_inputs/native_validation_summary.csv"
    data = pd.read_csv(path)
    data["split"] = "test"
    data.to_csv(path, index=False)
    with pytest.raises(ValueError, match="validation"):
        runner.selection_from_validation(tmp_path, inventory)


def test_score_tamper_cannot_change_shortlist(tmp_path):
    inventory = selection_inputs(tmp_path)
    inventory["candidates"][0]["robust_score"] = 999.
    with pytest.raises(ValueError, match="reconcile"):
        runner.selection_from_validation(tmp_path, inventory)

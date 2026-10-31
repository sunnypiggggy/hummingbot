"""Manual portfolio and causal-input tests without downloading model weights."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from scripts.crypto_foundation_portfolio import (
    CONTEXT_DAYS, FIELDS, context_frame, fixed_weights, frame_digest,
    known_exit_today, volatility,
)
from scripts.foundation_model_native import NativeForecaster, date_seed, parameter_digest
from scripts.foundation_model_audit_helpers import target_weights


def market_fixture():
    dates = pd.date_range("2025-01-01", periods=131, freq="D", tz="UTC")
    closes = 100 + np.arange(len(dates)) * .1
    daily = {}
    for name in FIELDS:
        values = closes.copy()
        if name == "volume":
            values[:] = 1000
        elif name == "amount":
            values *= 1000
        daily[name] = pd.DataFrame({"1": values, "1027": values * 2}, index=dates)
    return {"daily": daily}, dates[128]


def test_context_uses_exactly_previous_128_complete_days_and_no_future_values():
    market, decision = market_fixture()
    before = context_frame(market, decision, "1")
    assert before is not None and len(before) == CONTEXT_DAYS
    assert before.index[0] == decision - pd.Timedelta(days=128)
    assert before.index[-1] == decision - pd.Timedelta(days=1)
    assert before.index.max() < decision
    assert list(before.columns) == list(FIELDS)
    digest = frame_digest(before)
    for frame in market["daily"].values():
        frame.loc[frame.index >= decision, "1"] = 1e20
    after = context_frame(market, decision, "1")
    pd.testing.assert_frame_equal(before, after)
    assert frame_digest(after) == digest
    for name in FIELDS:
        market["daily"][name] = market["daily"][name].loc[lambda frame: frame.index < decision]
    pd.testing.assert_frame_equal(before, context_frame(market, decision, "1"))


def test_missing_real_quote_amount_and_short_context_are_not_filled():
    market, decision = market_fixture()
    market["daily"]["amount"].loc[decision - pd.Timedelta(days=1), "1"] = np.nan
    assert context_frame(market, decision, "1") is None
    market, decision = market_fixture()
    for name in FIELDS:
        market["daily"][name] = market["daily"][name].iloc[1:]
    assert context_frame(market, decision, "1") is None


def test_zero_amount_is_allowed_but_zero_recent_base_volume_is_not_ready():
    market, decision = market_fixture()
    market["daily"]["amount"].loc[decision - pd.Timedelta(days=1), "1"] = 0.
    assert context_frame(market, decision, "1") is not None
    market["daily"]["volume"].loc[(market["daily"]["volume"].index >= decision - pd.Timedelta(days=30)) &
                                    (market["daily"]["volume"].index < decision), "1"] = 0.
    assert context_frame(market, decision, "1") is None


def test_volatility_is_sample_std_of_30_causal_daily_returns():
    market, decision = market_fixture()
    frame = context_frame(market, decision, "1")
    expected = np.std(frame.close.to_numpy()[-30:] / frame.close.to_numpy()[-31:-1] - 1, ddof=1)
    assert volatility(frame) == pytest.approx(expected, rel=1e-13, abs=1e-15)
    frame.iloc[:-31, frame.columns.get_loc("close")] = 1e10
    assert volatility(frame) == pytest.approx(expected, rel=1e-13, abs=1e-15)


def test_fixed_hurdle_and_manual_cap_excess_remains_in_cash_without_refill():
    forecasts, risks, valid = np.zeros(10), np.full(10, .01), np.ones(10, bool)
    forecasts[0] = .0024
    assert np.array_equal(fixed_weights(forecasts, risks, valid), np.r_[np.zeros(10), 1.])
    forecasts[:2] = [.0124, .0224]
    risks[1] = .02
    expected = np.r_[.25, .25, np.zeros(8), .5]
    assert np.allclose(fixed_weights(forecasts, risks, valid), expected, rtol=0, atol=1e-14)
    forecasts[:4] = .0124
    risks[:] = .01
    expected = np.r_[np.full(4, .2375), np.zeros(6), .05]
    assert np.allclose(fixed_weights(forecasts, risks, valid), expected, rtol=0, atol=1e-14)
    # 24bp is a declared approximate round-trip hurdle, not an exact ROI.
    exact_break_even = (1.0002 * 1.001) / (.9998 * .999) - 1
    assert exact_break_even > .0024 and exact_break_even == pytest.approx(.002402882883, abs=1e-10)


def test_mapper_excludes_bad_forecasts_risk_and_invalid_slots():
    forecasts, risks, valid = np.ones(10), np.full(10, .03), np.ones(10, bool)
    forecasts[0], forecasts[1], risks[2], risks[3], valid[4] = np.nan, np.inf, np.nan, -.01, False
    actual = fixed_weights(forecasts, risks, valid)
    assert np.array_equal(actual[:5], np.zeros(5))
    assert (actual[:10] <= .25).all() and actual[-1] >= .05 - 1e-12
    assert actual.sum() == pytest.approx(1.)
    assert np.allclose(actual, target_weights(forecasts, risks, valid=valid), rtol=0, atol=1e-14)
    with pytest.raises(ValueError):
        fixed_weights(np.ones(9), np.ones(9), np.ones(9, bool))


def test_known_delisting_matches_original_announced_effective_boundary():
    asset = {"delisting_notice_at": "2026-06-12T06:00:00Z", "effective_until": "2026-06-30T03:00:00Z"}
    assert not known_exit_today(asset, pd.Timestamp("2026-06-11", tz="UTC"))
    assert not known_exit_today(asset, pd.Timestamp("2026-06-29", tz="UTC"))
    assert known_exit_today(asset, pd.Timestamp("2026-06-30", tz="UTC"))
    assert not known_exit_today(asset, pd.Timestamp("2026-07-01", tz="UTC"))


class FakeKronosPredictor:
    def __init__(self):
        self.calls = []

    def predict_batch(self, frames, x_times, y_times, **kwargs):
        self.calls.append((frames, x_times, y_times, kwargs))
        return [pd.DataFrame({"close": [float(torch.rand(()) + 100)]}, index=y_time)
                for y_time in y_times]


def test_kronos_batch_seed_is_repeatable_after_other_requests_and_keeps_dates_causal():
    market, date = market_fixture()
    frames = [context_frame(market, date, key) for key in ("1", "1027")]
    model = NativeForecaster.__new__(NativeForecaster)
    model.model_key = "kronos_small"
    model.predictor = FakeKronosPredictor()
    first, derived = model.predict(frames, date, seed=0, asset_keys=["1", "1027"])
    second, other_seed = model.predict(frames, date, seed=1, asset_keys=["1", "1027"])
    repeated, repeated_seed = model.predict(frames, date, seed=0, asset_keys=["1", "1027"])
    assert first == repeated and derived == repeated_seed
    assert second != first and derived != other_seed
    assert derived == date_seed("kronos_small", 0, date, ["1", "1027"])
    frames, x_times, y_times, kwargs = model.predictor.calls[0]
    assert all(stamps.max() < date.tz_localize(None) for stamps in x_times)
    assert all(list(stamps) == [date.tz_localize(None)] for stamps in y_times)
    assert kwargs == {"pred_len": 1, "T": 1., "top_p": .9, "top_k": 0,
                      "sample_count": 1, "verbose": False}
    with pytest.raises(ValueError):
        model.predict(frames, date, seed=0, asset_keys=["1027", "1"])


class FakeChronosPipeline:
    quantiles = [.1, .5, .9]

    def __init__(self):
        self.calls = []

    def predict(self, inputs, **kwargs):
        self.calls.append((inputs, kwargs))
        return [torch.tensor([[[99.], [100. + i], [102.]]]) for i in range(len(inputs))]


def test_chronos_uses_native_median_only_past_covariates_and_no_cross_asset_grouping():
    market, date = market_fixture()
    frames = [context_frame(market, date, key) for key in ("1", "1027")]
    model = NativeForecaster.__new__(NativeForecaster)
    model.model_key = "chronos2"
    model.config = {}
    model.pipeline = FakeChronosPipeline()
    actual, _ = model.predict(frames, date, seed=0, asset_keys=["1", "1027"])
    different_seed, _ = model.predict(frames, date, seed=4, asset_keys=["1", "1027"])
    assert actual == different_seed == [{"predicted_close": 100.}, {"predicted_close": 101.}]
    inputs, kwargs = model.pipeline.calls[0]
    assert kwargs == {"prediction_length": 1, "context_length": 128,
                      "batch_size": 64, "predict_batches_jointly": False}
    for entry, frame in zip(inputs, frames):
        assert set(entry) == {"target", "past_covariates"}
        assert np.array_equal(entry["target"], frame.close.to_numpy(np.float32))
        assert set(entry["past_covariates"]) == {"open", "high", "low", "volume", "amount"}
        assert all(len(values) == 128 for values in entry["past_covariates"].values())


def test_chronos_synchronous_loader_changes_only_loading_and_restores_factory_after_failure(monkeypatch):
    import sys
    module = sys.modules[FakeChronosPipeline.__module__]
    calls = []

    def original_loader(*args, **kwargs):
        calls.append((args, kwargs))
        return "original_loader_result"

    class FailingPipeline(FakeChronosPipeline):
        def predict(self, inputs, **kwargs):
            actual = module.DataLoader("synthetic_dataset", num_workers=1, pin_memory=True, batch_size=64)
            assert actual == "original_loader_result"
            raise RuntimeError("synthetic inference failure")

    monkeypatch.setattr(module, "DataLoader", original_loader, raising=False)
    market, date = market_fixture()
    model = NativeForecaster.__new__(NativeForecaster)
    model.model_key, model.config = "chronos2", {"synchronous_chronos_loader": True}
    model.pipeline = FailingPipeline()
    with pytest.raises(RuntimeError, match="synthetic inference failure"):
        model._chronos_results([context_frame(market, date, "1")])
    assert calls == [(("synthetic_dataset",), {"num_workers": 0, "pin_memory": False, "batch_size": 64})]
    assert module.DataLoader is original_loader


def test_model_tensor_digest_catches_inference_mutation():
    module = torch.nn.Linear(2, 1)
    holder = NativeForecaster.__new__(NativeForecaster)
    holder.modules = {"model": module}
    holder.initial_parameter_sha256 = parameter_digest(holder.modules)
    assert holder.verify_unchanged() == holder.initial_parameter_sha256
    with torch.no_grad():
        module.bias.add_(1)
    with pytest.raises(AssertionError):
        holder.verify_unchanged()


def test_atomic_cache_contract_accepts_concurrent_identical_writers_and_rejects_substitution(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from scripts.backtest_crypto_foundation_models import _commit_cache_contract, read
    path = tmp_path / "seed_0" / "cache_contract.json"
    contract = {"model_key": "synthetic", "seed": 0, "protocol_sha256": "fixed"}
    with ThreadPoolExecutor(max_workers=8) as workers:
        list(workers.map(lambda _: _commit_cache_contract(path, contract), range(32)))
    assert read(path) == contract
    assert list(path.parent.glob("*.partial")) == []
    with pytest.raises(AssertionError, match="different frozen inputs"):
        _commit_cache_contract(path, {**contract, "protocol_sha256": "substituted"})
    assert read(path) == contract


def test_date_shard_schedule_has_38_jobs_and_is_disjoint_exhaustive_with_stable_batch_rng(tmp_path):
    from scripts.backtest_crypto_foundation_models import forecast_jobs
    spec = {"models": {"kronos_small": {"sampling_seeds": list(range(5))},
                       "kronos_base": {"sampling_seeds": list(range(5))},
                       "chronos2": {"sampling_seeds": []}},
            "date_shards": {"kronos_small": 2, "kronos_base": 4, "chronos2": 8}}
    jobs = forecast_jobs(tmp_path, spec)
    assert len(jobs) == 38
    assert [sum(job["model_key"] == key for job in jobs) for key in spec["models"]] == [10, 20, 8]
    dates = pd.date_range("2025-07-25", periods=455, freq="D", tz="UTC")
    for key, config in spec["models"].items():
        for seed in config["sampling_seeds"] or [None]:
            actual = [job for job in jobs if job["model_key"] == key and job["seed"] == seed]
            pieces = [list(dates[job["shard_index"]::job["shard_count"]]) for job in actual]
            flat = [date for piece in pieces for date in piece]
            assert len(flat) == len(set(flat)) == len(dates) and set(flat) == set(dates)
            if seed is not None:
                original = {day: date_seed(key, seed, day, ["1", "1027"]) for day in dates}
                sharded = {day: date_seed(key, seed, day, ["1", "1027"]) for piece in pieces for day in piece}
                assert sharded == original


def test_mocked_native_shards_merge_repeat_and_recover_partial_commit_without_trusting_tampering(tmp_path, monkeypatch):
    import scripts.backtest_crypto_foundation_models as runner
    import scripts.foundation_model_native as native
    market, first = market_fixture()
    days = pd.date_range(first, periods=3, freq="D")
    records = []
    for date in days:
        for key in ("1", "1027"):
            frame = context_frame(market, date, key)
            records.append({"date": date, "asset_key": key, "symbol": "SYNTHETIC",
                            "context_ready": True, "exclusion_reason": "",
                            "input_start": date - pd.Timedelta(days=128), "input_end_exclusive": date,
                            "input_rows": 128, "input_sha256": frame_digest(frame),
                            "base_close": float(frame.close.iloc[-1]), "volatility_30d": volatility(frame)})
    plan = pd.DataFrame(records)
    (tmp_path / "forecast_request_plan.csv").write_bytes(runner.canonical_csv(plan))
    (tmp_path / "frozen_spec.json").write_text('{"synthetic_test": true}', encoding="utf-8")
    spec = {"models": {"kronos_small": {"sampling_seeds": [0], "first_decision_date": first.isoformat()}},
            "threads_per_model": 2, "date_shards": {"kronos_small": 2},
            "native_repeat_dates": [days[0].isoformat(), days[-1].isoformat()]}
    monkeypatch.setattr(runner, "load_market", lambda output: (market, {}))
    monkeypatch.setattr(runner, "_read_request_plan", lambda output: (plan.copy(deep=True), days))

    class SyntheticNative:
        calls = []
        initial_parameter_sha256 = "synthetic-fixed-parameters"

        def __init__(self, model_key, config, **kwargs):
            self.model_key = model_key

        def predict(self, frames, date, *, seed, asset_keys):
            self.calls.append((date, tuple(asset_keys)))
            # The real native path is not exercised in this cache scheduler test.
            return [{"predicted_close": float(frame.close.iloc[-1]) + 1} for frame in frames], date_seed(
                self.model_key, seed, date, asset_keys)

        def verify_unchanged(self):
            return self.initial_parameter_sha256

    monkeypatch.setattr(native, "NativeForecaster", SyntheticNative)
    jobs = runner.forecast_jobs(tmp_path, spec)
    shards = [runner.forecast_task(job) for job in jobs]
    assert len(SyntheticNative.calls) == 5  # Three requests and two fixed-date native repeats.
    assert all(keys == ("1", "1027") for date, keys in SyntheticNative.calls)
    manifests = runner.merge_forecast_shards(tmp_path, spec, list(reversed(shards)))
    assert len(manifests) == 1 and len(manifests[0]["native_repeat_verifications"]) == 2
    caches, _ = runner.load_forecasts(tmp_path, "kronos_small")
    assert len(caches[0]) == 6
    full_cache = tmp_path / "forecasts/kronos_small/seed_0.csv"
    original_bytes = full_cache.read_bytes()
    SyntheticNative.calls.clear()
    resumed = [runner.forecast_task(job) for job in jobs]
    assert len(SyntheticNative.calls) == 2  # Committed days are read, repeats still call the native model.
    runner.merge_forecast_shards(tmp_path, spec, resumed)
    assert full_cache.read_bytes() == original_bytes
    with pytest.raises(AssertionError, match="incomplete, duplicated or unexpected"):
        runner.merge_forecast_shards(tmp_path, spec, [resumed[0], resumed[0]])
    day = tmp_path / "forecasts/kronos_small/seed_0/days" / (days[0].strftime("%Y-%m-%d") + ".csv")
    sidecar = day.with_suffix(".sha256.json")
    sidecar.unlink()
    before = len(SyntheticNative.calls)
    runner.forecast_task(jobs[0])
    assert len(SyntheticNative.calls) - before == 3 and sidecar.exists()
    # A committed file cannot be silently replaced, even with parseable CSV bytes.
    day.write_bytes(day.read_bytes() + b"\n")
    with pytest.raises(AssertionError, match="Committed native day cache changed"):
        runner.forecast_task(jobs[0])
    # Missing a checksum requires a fresh native recomputation and byte identity.
    sidecar.unlink()
    with pytest.raises(AssertionError, match="uncommitted day cache does not match"):
        runner.forecast_task(jobs[0])

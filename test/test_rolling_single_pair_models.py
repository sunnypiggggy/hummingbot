"""Native CPU shape/counter proofs; never financial experiment outcomes."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json

import numpy as np
import pytest

from scripts import rolling_single_pair_models as rolling


def context(recipe_id="sac_default", symbol="BTCUSDT", seed=0, version=0, *, preflight=True):
    effective = datetime(2025, 10, 23, tzinfo=timezone.utc) + timedelta(days=7 * version)
    end = effective - timedelta(days=1)
    result = dict(recipe_id=recipe_id, symbol=symbol, algorithm=rolling.RECIPE_ALGORITHMS[recipe_id], seed=seed,
        version_index=version, effective_day=effective.isoformat(), train_end=end.isoformat(),
        train_start=(end - timedelta(days=1827)).isoformat(), protocol_sha256="a" * 64)
    if preflight:
        result["nonfinancial_preflight"] = True
    return result


def synthetic_env(symbol):
    import gymnasium as gym

    class ShapeFixture(gym.Env):
        observation_space = gym.spaces.Box(-10., 10., (13,), dtype=np.float32)
        action_space = gym.spaces.Box(-1., 1., (1,), dtype=np.float32)
        SYMBOLS = (symbol,)

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            self.position = 0
            return np.zeros(13, dtype=np.float32), {}

        def step(self, action):
            self.position += 1
            values = np.array([self.position / 32, float(action[0]), *[np.sin(self.position + i) * .1 for i in range(11)]], dtype=np.float32)
            return values, float(action[0]) * .001, False, self.position == 32, {}
    return ShapeFixture()


@pytest.mark.parametrize("rid,algorithm,gamma", [("sac_default", "sac", .995), ("sac_gamma099", "sac", .99),
    ("tqc_default", "tqc", .995), ("td3_default", "td3", .995)])
def test_recipe_is_original_except_declared_sac_gamma(rid, algorithm, gamma):
    original = rolling._base().recipe(algorithm)
    declared = rolling.recipe(algorithm, rid)
    assert declared["gamma"] == gamma
    assert {k: v for k, v in declared.items() if k != "gamma"} == {k: v for k, v in original.items() if k != "gamma"}
    original["policy_kwargs"]["net_arch"].append(999)
    assert rolling.recipe(algorithm, rid)["policy_kwargs"]["net_arch"] == [128, 128]


@pytest.mark.parametrize("version", [0, 1, 4, 52])
def test_context_dates_are_exact_and_immutable(version):
    value = context(version=version)
    checked = rolling.validate_context(value, recipe_id="sac_default", symbol="BTCUSDT", algorithm="sac", seed=0)
    assert checked == value and checked is not value


@pytest.mark.parametrize("key,value", [("recipe_id", "unplanned"), ("algorithm", "td3"), ("seed", True),
    ("seed", 5), ("symbol", "ADAUSDT"), ("version_index", True), ("version_index", 53),
    ("effective_day", "2025-10-24T00:00:00+00:00"), ("train_end", "2025-10-23T00:00:00+00:00"),
    ("train_start", "2020-10-22T00:00:00+00:00"), ("effective_day", "2025-10-23T00:00:00"),
    ("train_end", "2025-10-22T01:00:00+00:00"), ("protocol_sha256", "A" * 64),
    ("protocol_sha256", "a" * 63), ("nonfinancial_preflight", 1)])
def test_reject_context_identity_dates_and_flags(key, value):
    bad = context()
    bad[key] = value
    with pytest.raises(ValueError):
        rolling.validate_context(bad)


def test_context_must_bind_all_fields_and_cannot_relabel():
    for missing in rolling.REQUIRED_CONTEXT:
        value = context()
        value.pop(missing)
        with pytest.raises(ValueError):
            rolling.validate_context(value)
    with pytest.raises(ValueError):
        rolling.validate_context(context(), symbol="ETHUSDT")
    with pytest.raises(ValueError):
        rolling.validate_context(context(), seed=1)
    with pytest.raises(ValueError):
        rolling.validate_context(context(), recipe_id="sac_gamma099")
    with pytest.raises(ValueError):
        rolling.recipe("td3", "sac_default")


@pytest.mark.parametrize("bad", [True, 0, None, "False"])
def test_only_boolean_false_native_serialization_default_allowed(bad):
    with pytest.raises(ValueError):
        rolling.canonical_policy_kwargs({"net_arch": [128, 128], "use_sde": bad})


def test_serialization_default_and_named_classes_canonicalize():
    from torch import nn
    from torch.optim import Adam
    assert rolling.canonical_policy_kwargs({"net_arch": [128, 128], "use_sde": False}) == {"net_arch": [128, 128]}
    assert rolling.canonical_policy_kwargs({"activation_fn": nn.ReLU, "optimizer_class": Adam}) == {
        "activation_fn": "ReLU", "optimizer_class": "Adam"}


@pytest.mark.parametrize("recipe_id,symbol", [("sac_default", "BTCUSDT"), ("sac_gamma099", "ETHUSDT"), ("td3_default", "BTCUSDT")])
def test_real_native_training_serialization_and_inference(tmp_path, recipe_id, symbol):
    env, bound = synthetic_env(symbol), context(recipe_id, symbol)
    algorithm = bound["algorithm"]
    model = rolling.make_model(env, algorithm, recipe_id, 0, bound, trace_path=tmp_path / "timeouts.jsonl", device="cpu")
    before = rolling._rng_evidence(env, "cpu")
    assert rolling._rng_evidence(env, "cpu") == before
    for bad in (512, 32768, True):
        with pytest.raises(ValueError):
            rolling.train_model(model, env, algorithm, bad, 0)
    trained = rolling.train_model(model, env, algorithm, 1024, 0, progress_path=tmp_path / "progress.jsonl")
    assert trained["actual_steps"] == 1024
    assert trained["counters"]["replay_transitions"] == 1024
    assert len((tmp_path / "timeouts.jsonl").read_text().splitlines()) == 32
    expected = {"actor": 256 if algorithm == "td3" else 512, "critic": 512}
    if algorithm != "td3":
        expected["entropy"] = 512
    assert trained["counters"]["optimizer_steps"] == expected
    directory = tmp_path / "checkpoint"
    saved = rolling.save_model(model, env, algorithm, 0, directory, bound)
    assert saved["recipe"] == rolling.recipe(algorithm, recipe_id)
    assert saved["rolling_context"] == bound
    assert not (directory / "single_pair_model.json").exists()
    assert (directory / "rolling_model.json").is_file()
    restored = rolling.load_model(directory, algorithm, symbol, 0, bound, device="cpu")
    assert rolling.digest(model) == rolling.digest(restored)
    for offset in range(8):
        obs = np.array([np.sin(i + offset) * .1 for i in range(13)], dtype=np.float32)
        np.testing.assert_array_equal(rolling.predict(model, obs)[0], rolling.predict(restored, obs)[0])
    assert rolling.digest(model) == saved["parameter_sha256"]
    with pytest.raises(FileExistsError):
        rolling.save_model(model, env, algorithm, 0, directory, bound)
    with pytest.raises(ValueError):
        rolling.train_model(model, env, algorithm, 1024, 0)
    changed_context = deepcopy(bound)
    changed_context["protocol_sha256"] = "b" * 64
    with pytest.raises(ValueError):
        rolling.load_model(directory, algorithm, symbol, 0, changed_context, device="cpu")
    path = directory / "rolling_model.json"
    original = path.read_text()
    for field, value in (("recipe", {"gamma": .98}), ("parameter_sha256", "a" * 64),
                         ("adapter_sha256", "b" * 64), ("recipe_id", "sac_default" if recipe_id != "sac_default" else "sac_gamma099")):
        bad = json.loads(original)
        bad[field] = value
        path.write_text(json.dumps(bad), encoding="utf-8")
        with pytest.raises(ValueError):
            rolling.load_model(directory, algorithm, symbol, 0, bound, device="cpu")
    path.write_text(original, encoding="utf-8")


def test_formal_fragment_counter_checks():
    bound = context(preflight=False)
    first = datetime.fromisoformat(bound["train_start"])
    fragment = dict(start=first.isoformat(), end_exclusive=(first + timedelta(days=32)).isoformat(),
        completed_until=(first + timedelta(days=32)).isoformat(), completed_steps=32, terminated=False, truncated=True)
    record = dict(actual_steps=32768, counters=dict(environment_steps=32768, gradient_update_rounds=32256,
        optimizer_steps={"actor": 32256, "critic": 32256, "entropy": 32256}, replay_add_calls=32768, replay_transitions=32768),
        training_episode_history=[deepcopy(fragment) for _ in range(1024)])
    assert rolling.check_training_record(record, "sac", bound)
    for mutator in (lambda r: r["counters"].update(replay_transitions=32767),
                    lambda r: r["training_episode_history"].pop(),
                    lambda r: r["training_episode_history"][0].update(terminated=True),
                    lambda r: r["training_episode_history"][0].update(completed_steps=31),
                    lambda r: r["training_episode_history"][0].update(end_exclusive=bound["effective_day"])):
        bad = deepcopy(record)
        mutator(bad)
        with pytest.raises(ValueError):
            rolling.check_training_record(bad, "sac", bound)


def test_preflight_cannot_claim_formal_financial_environment():
    with pytest.raises(ValueError):
        rolling.make_model(synthetic_env("BTCUSDT"), "sac", "sac_default", 0, context(preflight=False), device="cpu")


def formal_fixture(*, future_hour=False, warmup_days=210, **kwargs):
    import pandas as pd
    from scripts.single_pair_rl_env import SinglePairEnv

    bound = context(preflight=False)
    first, end = pd.Timestamp(bound["train_start"]), pd.Timestamp(bound["train_end"])
    hours = pd.date_range(first - pd.Timedelta(days=warmup_days), end + pd.Timedelta(hours=int(future_hour)), freq="h", inclusive="left")
    prices = 100 + np.arange(len(hours)) * .0001 + np.sin(np.arange(len(hours)) / 24)
    frame = pd.DataFrame({"BTCUSDT": prices}, index=hours)
    data = {"hourly_" + key: frame.copy() for key in ("open", "high", "low", "close", "volume")}
    return SinglePairEnv(data, "BTCUSDT", first, end, training=True, episode_days=32, record_history=False, **kwargs), bound


def test_formal_environment_retains_exact_physical_cutoff_and_feature_prehistory():
    env, bound = formal_fixture()
    assert rolling._check_environment(env, bound)["symbol"] == "BTCUSDT"
    assert env.training_window_metadata["requested_train_days"] == 1827
    for _ in range(4):
        env.reset()
        episode = env.training_episode_history[-1]
        assert datetime.fromisoformat(episode["start"]) >= datetime.fromisoformat(bound["train_start"])
        assert datetime.fromisoformat(episode["end_exclusive"]) <= datetime.fromisoformat(bound["train_end"])


@pytest.mark.parametrize("changes", [{"future_hour": True}, {"warmup_days": 199},
    {"capital": 9999.}, {"costs": {"fee": .002, "slippage": .0005}}, {"rebalance_band": .03}])
def test_formal_environment_rejects_future_data_insufficient_prehistory_and_financial_change(changes):
    env, bound = formal_fixture(**changes)
    with pytest.raises(ValueError):
        rolling._check_environment(env, bound)

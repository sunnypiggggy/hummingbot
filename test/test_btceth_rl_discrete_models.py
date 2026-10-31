"""Core contracts on synthetic data/CPU; no learning or Contrib installation."""
from pathlib import Path
from types import SimpleNamespace, ModuleType
import hashlib
import json
import subprocess
import sys

import gymnasium as gym
import numpy as np
import pandas as pd
import pytest

from scripts import btceth_rl_discrete_models as core
from scripts.btceth_rl_env import BTCETHPortfolioEnv


class ProbeEnv(gym.Env):
    observation_space = gym.spaces.Box(-10, 10, (27,), dtype=np.float32)
    action_space = gym.spaces.Box(-1, 1, (2,), dtype=np.float32)

    def __init__(self, fragment=32):
        self.fragment, self.histories, self.last_action = fragment, [], None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        self.histories.append(dict(completed_steps=0))
        return np.full(27, -.5, dtype=np.float32), {"seed_seen": seed, "options_seen": options}

    def step(self, action):
        self.last_action = np.asarray(action).copy(); self.steps += 1
        self.histories[-1]["completed_steps"] = self.steps
        return np.full(27, self.steps / self.fragment, dtype=np.float32), .01, False, self.steps == self.fragment, {"date": "synthetic"}


def flat_data(days=2):
    start = pd.Timestamp("2025-01-01T00:00:00Z"); end = start + pd.Timedelta(days=days)
    hours = pd.date_range(start - pd.Timedelta(days=210), end, freq="h", inclusive="left")
    prices = pd.DataFrame({"BTCUSDT": 100., "ETHUSDT": 200.}, index=hours)
    data = {"hourly_" + name: prices.copy() for name in ("open", "high", "low", "close")}
    data["hourly_volume"] = prices * 0 + 100.
    return data, start, end


def replay(**kwargs):
    values = dict(device="cpu", n_envs=1, optimize_memory_usage=False, handle_timeout_termination=True)
    values.update(kwargs)
    return core.observed_replay_class()(50000, gym.spaces.Box(-10, 10, (27,), dtype=np.float32), gym.spaces.Discrete(21), **values)


def test_module_import_does_not_import_torch_or_native_frameworks():
    code = """
import builtins
original = builtins.__import__
def guarded(name,*a,**kw):
    if name.split('.')[0] in ('torch','stable_baselines3','sb3_contrib'):
        raise AssertionError('GPU-aware import before worker isolation: '+name)
    return original(name,*a,**kw)
builtins.__import__=guarded
from scripts import btceth_rl_discrete_models as core
assert len(core.ACTION_TABLE)==21
assert core.training_recipe('qrdqn')['policy_kwargs']['n_quantiles']==200
"""
    result = subprocess.run([sys.executable, "-B", "-c", code], cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_all_21_actions_match_original_box_mapping_and_are_unique():
    data, start, end = flat_data()
    env = BTCETHPortfolioEnv(data, start, end)
    assert core.ACTION_TABLE[0] == (-1., 0.) and len(set(core.ACTION_TABLE)) == 21
    expected = [np.array([0., 0., 1.])]
    for g in (.25, .5, .75, 1.):
        for share in (0., .25, .5, .75, 1.):
            expected.append(np.array([.95 * g * (1 - share), .95 * g * share, 1 - .95 * g]))
    for i, target in enumerate(expected):
        action = core.box_action(np.int64(i))
        assert action.dtype == np.float32 and action.shape == (2,)
        np.testing.assert_allclose(env.action_to_weights(action), target, atol=1e-15, rtol=0)
        np.testing.assert_allclose(core.mapped_target(env, i), target, atol=1e-15, rtol=0)
    # Returned arrays cannot mutate the immutable canonical table.
    action[:] = 0
    assert core.ACTION_TABLE[-1] == (1., 1.)
    assert core.action_table_sha256() == hashlib.sha256(json.dumps(core.ACTION_TABLE, separators=(",", ":")).encode("ascii")).hexdigest()


@pytest.mark.parametrize("invalid", [-1, 21, True, False, 1., 1.5, np.nan, "1", [1], np.array([1, 2]), None])
def test_invalid_action_ids_cannot_be_coerced(invalid):
    with pytest.raises(ValueError):
        core.box_action(invalid)


def test_wrapper_changes_only_action_and_forwards_reset_observation_reward_and_info():
    raw = ProbeEnv(); env = core.DiscretePortfolioActions(raw)
    assert env.observation_space is raw.observation_space and env.action_space.n == 21
    obs, info = env.reset(seed=4, options={"synthetic": True})
    assert info == dict(seed_seen=4, options_seen={"synthetic": True})
    value, reward, terminated, truncated, info = env.step(20)
    np.testing.assert_array_equal(raw.last_action, [1., 1.])
    assert reward == .01 and not terminated and not truncated and info["date"] == "synthetic"
    assert value.shape == obs.shape == (27,)


def test_ensemble_averages_targets_instead_of_nonlinear_box_actions():
    data, start, end = flat_data(); env = BTCETHPortfolioEnv(data, start, end)
    ids = np.array([0, 16, 20, 5, 1])
    mapped = core.mean_mapped_targets(env, ids)
    manual = np.mean([env.action_to_weights(core.box_action(i)) for i in ids], axis=0)
    wrong = env.action_to_weights(np.mean([core.box_action(i) for i in ids], axis=0))
    np.testing.assert_array_equal(mapped, manual)
    assert not np.allclose(mapped, wrong)
    capped = core.mean_mapped_targets(env, ids, exposure_multiplier=.75)
    np.testing.assert_allclose(capped[:2], mapped[:2] * .75, atol=1e-15, rtol=0)
    assert capped.sum() == 1. and capped[2] >= .05
    with pytest.raises(ValueError): core.mean_mapped_targets(env, ids[:-1])
    with pytest.raises(ValueError): core.mapped_target(env, 0, exposure_multiplier=True)


def test_fixed_recipe_native_optimizer_and_real_update_arithmetic(monkeypatch):
    import torch
    recipe = core.training_recipe("qrdqn")
    assert recipe["policy_kwargs"] == dict(net_arch=[128, 128], optimizer_class="torch.optim.Adam", optimizer_kwargs=dict(eps=1e-8), n_quantiles=200)
    assert recipe["learning_starts"] == 1024 and recipe["buffer_size"] == 50000
    assert (32768 - 1024) // 4 == core.expected_training_counters()["optimizer_updates"] == 7936
    assert 32768 // 512 == core.expected_training_counters()["target_sync_calls"] == 64
    recipe["policy_kwargs"]["net_arch"][0] = 7
    assert core.training_recipe("qrdqn")["policy_kwargs"]["net_arch"] == [128, 128]
    for algorithm in ("qrdqn", "dqn"):
        values = core.native_kwargs(algorithm)
        assert values["policy_kwargs"]["optimizer_class"] is torch.optim.Adam
        assert values["policy_kwargs"]["optimizer_kwargs"] == dict(eps=1e-8)
        assert values["replay_buffer_kwargs"]["handle_timeout_termination"] is True
        assert ("n_quantiles" in values["policy_kwargs"]) is (algorithm == "qrdqn")
    fake = ModuleType("sb3_contrib"); fake.QRDQN = object()
    monkeypatch.setitem(sys.modules, "sb3_contrib", fake)
    monkeypatch.setattr(core.importlib.metadata, "version", lambda name: "2.9.0")
    assert core.native_class("qrdqn") is fake.QRDQN
    with pytest.raises(ValueError): core.native_class("ppo")


class TensorPolicy:
    def __init__(self):
        self.eval_calls = 0

    def set_training_mode(self, training):
        assert training is False; self.eval_calls += 1

    def obs_to_tensor(self, obs):
        import torch
        values = np.asarray(obs)
        return torch.as_tensor(values.reshape(-1, 27), dtype=torch.float32), values.ndim == 2


class SyntheticQuantiles:
    action_space = gym.spaces.Discrete(21)
    observation_space = gym.spaces.Box(-10, 10, (27,), dtype=np.float32)

    def __init__(self, q):
        self.policy, self.q = TensorPolicy(), np.asarray(q, dtype=np.float32)

    def quantile_net(self, obs):
        import torch
        return torch.as_tensor(np.repeat(self.q[None], len(obs), axis=0))

    def predict(self, obs, state=None, episode_start=None, deterministic=False):
        assert deterministic is True
        values = np.asarray(obs); ids = self.q.mean(axis=0).argmax()
        return np.array(ids) if values.ndim == 1 else np.full(len(values), ids, dtype=np.int64), state


def test_qr_prefix_tail_is_nominal_unsorted_and_mean_native_adapter_keeps_policy():
    q = np.zeros((200, 21), dtype=np.float32)
    q[:50, 1] = 5.; q[50:, 1] = -5.; q[:, 2] = 1.
    model = SyntheticQuantiles(q)
    mean = core.DiscretePredictor(model, "qrdqn_mean")
    tail = core.DiscretePredictor(model, "qrdqn_lower25")
    obs = np.zeros(27, dtype=np.float32)
    np.testing.assert_array_equal(mean.predict(obs)[0], core.box_action(2))
    np.testing.assert_array_equal(tail.predict(obs)[0], core.box_action(1))
    assert mean.policy is tail.policy is model.policy
    assert np.sort(q, axis=0)[:50].mean(axis=0).argmax() != 1
    diagnostic = tail.last_prediction["diagnostics"]
    assert diagnostic["heads_sorted"] is diagnostic["cvar_training_or_guarantee"] is False
    assert diagnostic["mean_lower_disagreement_rate"] == 1.
    actions, _ = tail.predict(np.zeros((3, 27), dtype=np.float32))
    assert actions.shape == (3, 2) and tail.last_prediction["action_ids"] == [1, 1, 1]
    with pytest.raises(ValueError): tail.predict(obs, deterministic=False)


def test_zero_ties_choose_first_action_and_crossings_are_adjacent_tau_heads():
    q = np.zeros((200, 21), dtype=np.float32)
    values = core.quantile_diagnostics(q)
    assert values["mean_action_ids"] == values["lower25_action_ids"] == [0]
    assert values["quantile_crossing_rate"] == 0
    q[0, 5] = 1.
    values = core.quantile_diagnostics(q)
    assert values["adjacent_crossings"] == 1 and values["adjacent_head_pairs"] == 199 * 21
    assert values["quantile_crossing_rate"] == 1 / (199 * 21)
    with pytest.raises(ValueError): core.quantile_diagnostics(np.zeros((199, 21)))
    with pytest.raises(ValueError): core.quantile_diagnostics(np.full((200, 21), np.nan))


def test_native_replay_vecenv_stores_terminal_instead_of_reset_observation(tmp_path):
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
    raw = ProbeEnv(); vec = DummyVecEnv([lambda: core.DiscretePortfolioActions(raw)])
    vec.seed(4); obs = vec.reset()
    path = tmp_path / "timeouts.jsonl"; buffer = replay(trace_path=path)
    assert not path.exists()
    owner = SimpleNamespace(_vec_normalize_env=None, _last_obs=obs)
    for _ in range(64):
        new_obs, rewards, dones, infos = vec.step(np.array([20], dtype=np.int64))
        OffPolicyAlgorithm._store_transition(owner, buffer, np.array([20], dtype=np.int64), new_obs, rewards, dones, infos)
    assert buffer.observed_transitions == buffer.pos == 64 and len(buffer.timeout_records) == 2
    assert [r["completed_steps"] for r in raw.histories] == [32, 32, 0]
    for row in buffer.timeout_records:
        assert row["stored_done"] == row["stored_timeout"] == 1. and row["effective_done_from_stored_flags"] == 0
        assert row["handle_timeout_termination"] and row["TimeLimit.truncated"] and not row["terminated_from_stored_flags"]
        assert row["terminal_observation_present"] and row["stored_next_matches_terminal"]
        assert row["stored_next_observation_values"] == row["terminal_observation_values"] == [1.] * 27
        assert row["stored_next_observation_values"] != [-.5] * 27  # VecEnv reset value
    samples = buffer._get_samples(np.array([31, 63]))
    assert samples.dones.cpu().numpy().tolist() == [[0.], [0.]]
    assert json.loads(path.read_text().splitlines()[-1])["transition_chain_sha256"] == buffer.transition_chain_sha256
    vec.close()


def test_digest_is_of_actual_post_add_storage_and_cannot_open_existing_trace(tmp_path):
    buffer = replay(); obs = np.zeros((1, 27), dtype=np.float32); nxt = obs + .2
    action = np.array([3], dtype=np.int64); reward = np.array([.5], dtype=np.float32); done = np.array([False])
    buffer.add(obs, nxt, action, reward, done, [{}])
    payload = b"".join(np.asarray(value, dtype=dtype).tobytes() for value, dtype in (
        (buffer.observations[0], "<f4"), (buffer.next_observations[0], "<f4"), (buffer.actions[0], "<i8"),
        (buffer.rewards[0], "<f4"), (buffer.dones[0], "<f4"), (buffer.timeouts[0], "<f4")))
    expected = hashlib.sha256(bytes.fromhex(hashlib.sha256(b"").hexdigest()) + payload).hexdigest()
    assert buffer.transition_chain_sha256 == expected
    path = tmp_path / "retained_trace.jsonl"; path.write_bytes(b"Retain this original trace")
    duplicate = replay(trace_path=path)
    with pytest.raises(FileExistsError):
        duplicate.add(obs, nxt, action, reward, np.array([True]), [{"TimeLimit.truncated": True, "terminal_observation": nxt[0]}])
    assert path.read_bytes() == b"Retain this original trace"


def test_native_dqn_checkpoint_save_load_and_predict_never_append_training_trace(tmp_path):
    import torch
    model = core.native_class("dqn")("MlpPolicy", core.DiscretePortfolioActions(ProbeEnv()),
        device="cpu", seed=0, **core.native_kwargs("dqn", trace_path=tmp_path / "trace.jsonl"))
    # No learn/train call: only native replay serialization and inference.
    assert model._n_updates == model.num_timesteps == 0
    buffer = model.replay_buffer
    obs = np.zeros((1, 27), dtype=np.float32); nxt = obs + .1
    buffer.add(obs, nxt, np.array([0]), np.array([.1]), np.array([True]),
        [{"TimeLimit.truncated": True, "terminal_observation": nxt[0]}])
    trace = (tmp_path / "trace.jsonl").read_bytes()
    model.save(tmp_path / "model.zip")
    loaded = core.native_class("dqn").load(tmp_path / "model.zip", device="cpu")
    assert (tmp_path / "trace.jsonl").read_bytes() == trace
    assert loaded._n_updates == loaded.num_timesteps == 0
    assert all(torch.equal(value, loaded.policy.state_dict()[name]) for name, value in model.policy.state_dict().items())
    adapted = core.DiscretePredictor(loaded, "dqn_mean")
    native, _ = loaded.predict(obs[0], deterministic=True)
    value, _ = adapted.predict(obs[0])
    np.testing.assert_array_equal(value, core.box_action(native))
    assert adapted.policy is loaded.policy and (tmp_path / "trace.jsonl").read_bytes() == trace


def test_real_monthly_contract_reporting_frame_and_independent_reconstruction_dict():
    from scripts.crypto_rl_reporting import summarize
    from scripts.audit_btceth_rl import reconstruct_fills, Checks
    data, start, end = flat_data(); env = core.DiscretePortfolioActions(BTCETHPortfolioEnv(data, start, end))
    env.reset(seed=0)
    for _ in range(2): env.step(0)
    result = env.unwrapped.get_result()
    row, history, months = summarize(result, run_id="synthetic", strategy="dqn", scenario="base",
        split="test", start=start, end=end, capital=10000., fee=.001, slippage=.0002)
    rebuilt = reconstruct_fills(data, result["fills"], result["equity"], result["metadata"],
        start=start, end=end, fee=.001, slippage=.0002, checks=Checks(), label="synthetic")
    assert isinstance(months, pd.DataFrame) and isinstance(rebuilt["monthly"], dict)
    assert dict(zip(months.month, months["return"])) == rebuilt["monthly"]
    assert row["net_return"] == 0 and len(history) == 3
    env.close()

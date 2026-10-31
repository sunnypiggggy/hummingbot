"""Independent synthetic checks; no market reading, SSH, or CUDA dispatch."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts import single_pair_rl_models as models
from scripts import smoke_single_pair_rl_extended as smoke


def test_frozen_core_has_not_changed():
    assert models.sha(models.__file__) == smoke.CORE_SHA256


def test_native_td3_cpu_uses_actual_delayed_updates_and_checkpoint(tmp_path):
    before = models.sha(models.__file__)
    recipe = copy.deepcopy(models.recipe("td3"))
    result = smoke.native_workload(tmp_path/"td3","td3",symbol="ETHUSDT",seed=1)
    assert result["passed"] is True
    assert result["actual_steps"] == 1024
    assert result["training"]["counters"]["optimizer_steps"] == {"actor":256,"critic":512}
    assert result["timeouts"]["boundaries"] == 32
    assert result["initial_parameter_sha256"] != result["trained_parameter_sha256"]
    assert result["restored_parameter_sha256"] == result["trained_parameter_sha256"]
    assert result["action_before_load"] == result["action_after_load"]
    assert result["recipe"] == models.recipe("td3") == recipe
    assert result["synthetic_recipe_override"] is False
    assert models.sha(models.__file__) == before


def test_full_recipe_native_droq_cpu_warmup_and_checkpoint(tmp_path):
    upstream = Path("results/backtests/btceth_rl_droq_20261030_preflight/upstream/DroQ").resolve()
    before = models.sha(models.__file__)
    recipe = copy.deepcopy(models.recipe("droq"))
    result = smoke.native_workload(tmp_path/"droq","droq",upstream=upstream,seed=2)
    assert result["passed"] is True
    assert result["actual_steps"] == 5002
    assert result["training"]["counters"]["q_optimizer_steps"] == [40,40]
    assert result["training"]["counters"]["target_updates_per_network"] == [40,40]
    assert result["training"]["counters"]["policy_optimizer_steps"] == 2
    assert result["training"]["counters"]["alpha_optimizer_steps"] == 2
    assert result["training"]["effective_parameters"]["target_entropy"] == -1
    assert result["timeouts"]["boundaries"] == 156
    assert result["restored_parameter_sha256"] == result["trained_parameter_sha256"]
    assert result["action_before_load"] == result["action_after_load"]
    assert result["recipe"] == models.recipe("droq") == recipe
    assert result["synthetic_recipe_override"] is False
    assert models.sha(models.__file__) == before


@pytest.mark.parametrize("algorithm",["crossq","recurrent_ppo"])
def test_native_contrib_extended_cpu_when_installed(tmp_path,algorithm):
    pytest.importorskip("sb3_contrib")
    result = smoke.native_workload(tmp_path/algorithm,algorithm)
    assert result["passed"] is True
    if algorithm == "crossq":
        assert result["training"]["counters"]["optimizer_steps"]["actor"] == 170
        assert result["training"]["counters"]["optimizer_steps"]["entropy"] == 170
    else:
        assert result["training"]["counters"]["optimizer_steps"] == {"policy":40}
        assert result["recurrent"]["terminal_bootstrap_checks"] == 32


def test_droq_requires_explicit_upstream_before_workload(tmp_path):
    with pytest.raises(ValueError,match="explicit verified"):
        smoke.native_workload(tmp_path/"never-created","droq")
    assert not (tmp_path/"never-created").exists()


def test_cuda_wrapper_refuses_different_physical_uuid_before_compute(tmp_path,monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES","GPU-wrong")
    with pytest.raises(ValueError,match="expected physical"):
        smoke.run_smoke(tmp_path/"failure","td3","GPU-expected")
    failure = json.loads((tmp_path/"failure/failure.json").read_text(encoding="utf-8"))
    assert failure["error_type"] == "ValueError"
    assert not (tmp_path/"failure/native").exists()


@pytest.mark.parametrize("algorithm",smoke.ALGORITHMS)
def test_counters_reject_claimed_updates_without_actual_matching_steps(algorithm):
    counters = {"environment_steps":1024,"gradient_update_rounds":512,"optimizer_steps":{"actor":0,"critic":512},
                "replay_add_calls":1024,"replay_transitions":1024}
    record = {"actual_steps":5002 if algorithm == "droq" else 1024,"counters":counters}
    env = SimpleNamespace(total_steps=record["actual_steps"])
    with pytest.raises(AssertionError):
        smoke.assert_training_counters(algorithm,record,env)


class FakePolicy:
    def __init__(self):
        self.calls = 0

    def predict_values(self,obs,lstm_states,episode_starts):
        self.calls += 1
        return torch.tensor([.25])


class FakeRollout:
    def __init__(self):
        self.pos = 0
        self.calls = 0
        self.rewards = np.zeros((128,1),dtype=np.float32)
        self.observations = np.zeros((128,1,13),dtype=np.float32)
        self.episode_starts = np.zeros((128,1),dtype=np.float32)
        for name in ("hidden_states_pi","cell_states_pi","hidden_states_vf","cell_states_vf"):
            setattr(self,name,np.zeros((128,1,1,64),dtype=np.float32))
        self.returned = object()

    def add(self,obs,actions,reward,starts,*args,**kwargs):
        self.calls += 1
        self.observations[self.pos] = obs
        self.rewards[self.pos] = reward
        self.episode_starts[self.pos] = starts
        self.pos += 1
        return self.returned


def recurrent_fixture():
    model = SimpleNamespace(policy=FakePolicy(),rollout_buffer=FakeRollout(),gamma=.995)
    env = SimpleNamespace(total_steps=32,raw_rewards={32:.001})
    terminal = torch.zeros((1,13))
    terminal[0,0] = 1
    state = (torch.zeros((1,1,64)),torch.zeros((1,1,64)))
    starts = torch.tensor([False])
    return model,env,terminal,state,starts


def test_recurrent_observation_preserves_native_return_and_call_count():
    model,env,terminal,state,starts = recurrent_fixture()
    observer = smoke.RecurrentObservation(model,env)
    try:
        value = model.policy.predict_values(terminal,state,starts)
        reward = np.asarray([.001+.995*.25],dtype=np.float32)
        returned = model.rollout_buffer.add(np.zeros((1,13)),np.zeros((1,1)),reward,[False])
        assert returned is model.rollout_buffer.returned
        assert torch.equal(value,torch.tensor([.25]))
        assert model.policy.calls == model.rollout_buffer.calls == 1
        assert observer.values[0]["observation"][0] == 1
        assert observer.adds[0]["stored_reward"] == float(reward[0])
    finally:
        observer.close()
    assert "predict_values" not in model.policy.__dict__
    assert "add" not in model.rollout_buffer.__dict__


def test_recurrent_observation_detects_missing_terminal_bootstrap():
    model,env,terminal,state,starts = recurrent_fixture()
    observer = smoke.RecurrentObservation(model,env)
    try:
        model.policy.predict_values(terminal,state,starts)
        with pytest.raises(AssertionError,match="bootstrap"):
            model.rollout_buffer.add(np.zeros((1,13)),np.zeros((1,1)),np.asarray([.001],dtype=np.float32),[False])
    finally:
        observer.close()


def test_synthetic_terminal_is_real_observation_not_reset():
    env = smoke.SyntheticPair("BTCUSDT")
    original,_ = env.reset(seed=0)
    for _ in range(32):
        final,reward,terminated,truncated,info = env.step([.5])
    reset,_ = env.reset()
    assert truncated is True and terminated is False
    assert final[0] == 1 and reset[0] == original[0] == 0
    assert env.terminals[0]["observation"] == final.tolist()


def test_timeout_check_rejects_stored_reset_observation(tmp_path):
    env = smoke.SyntheticPair("BTCUSDT")
    env.reset()
    for _ in range(1024):
        _,_,_,done,_ = env.step([.5])
        if done:
            env.reset()
    rows = []
    for row in env.terminals:
        values = row["observation"]
        rows.append({"terminal_observation_values":values,"stored_next_observation_values":[0.]*13,
            "stored_next_observation_sha256":"wrong","terminal_observation_sha256":"correct", "effective_done":0,"timeout":1})
    trace = tmp_path/"trace.jsonl"
    trace.write_text("\n".join(json.dumps(row) for row in rows)+"\n",encoding="utf-8")
    with pytest.raises(AssertionError,match="reset state"):
        smoke.check_timeouts("td3",trace,env)

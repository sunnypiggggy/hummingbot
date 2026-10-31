"""Synthetic, single-symbol native learner checks; no market or GPU run."""
import copy
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import torch

from scripts import single_pair_rl_models as models


class SyntheticPair(gym.Env):
    SYMBOLS = ("BTCUSDT",)
    observation_space = gym.spaces.Box(-10., 10., (13,), dtype=np.float32)
    action_space = gym.spaces.Box(-1., 1., (1,), dtype=np.float32)

    def __init__(self):
        self.resets = 0
        self.total_steps = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.position = 0
        self.resets += 1
        return np.zeros(13,dtype=np.float32), {}

    def step(self, action):
        self.position += 1
        self.total_steps += 1
        values = np.full(13,self.position,dtype=np.float32)
        values[1] = float(action[0])
        return values, float(action[0])*.001, False, self.position == 4, {"date":str(self.total_steps)}


@pytest.mark.parametrize("algorithm",models.ALGORITHMS)
def test_json_recipe_is_fresh_and_declares_original_budget(algorithm):
    value = models.recipe(algorithm)
    json.dumps(value,allow_nan=False)
    reference = copy.deepcopy(value)
    value["mutated_for_synthetic_test"] = True
    assert models.recipe(algorithm) == reference
    assert models.STEPS[algorithm] == (65536 if algorithm == "td3" else 32768)


def test_one_dimensional_td3_noise_and_native_recipe():
    env = SyntheticPair()
    model = models.make_model("td3",env,seed=0)
    actual = models.effective_parameters(model,"td3")
    assert actual["action_noise"]["mean"] == [0.]
    assert actual["action_noise"]["sigma"] == [.1]
    assert model.action_space.shape == (1,)
    assert model.observation_space.shape == (13,)
    assert actual["gradient_steps"] == 1
    assert actual["policy_delay"] == 2
    assert actual["policy_kwargs"]["net_arch"] == [128,128]
    assert model.replay_buffer.action_dim == 1


@pytest.mark.parametrize("seed",[True,False,-1,5,1.5])
def test_seed_identity_is_not_a_rank_or_unbounded_search(seed):
    with pytest.raises(ValueError,match="predeclared"):
        models.seed_everything(seed)


@pytest.mark.parametrize("symbols",[("BTCUSDT","ETHUSDT"),("ADAUSDT",),()])
def test_model_refuses_other_or_multiple_pair_envs(symbols):
    env = SyntheticPair()
    env.SYMBOLS = symbols
    with pytest.raises(ValueError,match="Exactly one"):
        models.make_model("sac",env,seed=0)


def test_model_refuses_old_portfolio_spaces():
    env = SyntheticPair()
    env.observation_space = gym.spaces.Box(-10.,10.,(27,),dtype=np.float32)
    env.action_space = gym.spaces.Box(-1.,1.,(2,),dtype=np.float32)
    with pytest.raises(ValueError,match="observation13/action1"):
        models.make_model("sac",env,seed=0)


def test_actual_sac_training_replay_terminal_chain_and_checkpoint(tmp_path):
    env = SyntheticPair()
    trace = tmp_path/"terminal.jsonl"
    progress = tmp_path/"progress.jsonl"
    model = models.make_model("sac",env,seed=1,trace_path=trace)
    assert not trace.exists()
    record = models.train_model(model,env,1024,seed=1,progress_path=progress)
    assert record["actual_steps"] == 1024
    assert env.total_steps == 1024
    assert record["counters"]["replay_transitions"] == 1024
    assert record["counters"]["replay_add_calls"] == 1024
    assert record["counters"]["gradient_update_rounds"] == 512
    assert record["counters"]["optimizer_steps"]["actor"] == 512
    assert record["counters"]["optimizer_steps"]["critic"] == 512
    terminals = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines()]
    assert len(terminals) == 256
    assert all(row["timeout"] == 1 and row["effective_done"] == 0 for row in terminals)
    assert all(row["terminal_observation_values"][0] == 4 for row in terminals)
    assert all(row["stored_next_observation_values"] == row["terminal_observation_values"] for row in terminals)
    assert all(row["stored_next_observation_sha256"] == row["terminal_observation_sha256"] for row in terminals)
    values = np.linspace(-1,1,13,dtype=np.float32)
    expected,_ = models.predict(model,values,"sac")
    manifest = models.save_model(model,tmp_path/"checkpoint","sac")
    before_trace = trace.read_bytes()
    restored = models.load_model(tmp_path/"checkpoint","sac")
    assert trace.read_bytes() == before_trace
    assert models.parameter_digest(restored,"sac") == manifest["parameter_sha256"]
    observed,_ = models.predict(restored,values,"sac")
    np.testing.assert_array_equal(expected,observed)
    assert manifest["observation_shape"] == [13] and manifest["action_shape"] == [1]
    assert manifest["multi_model_target_average"] is False
    assert manifest["independent_account"] is True
    with pytest.raises(ValueError,match="resume"):
        models.train_model(restored,env,1,seed=1)
    with pytest.raises(FileExistsError):
        models.save_model(model,tmp_path/"checkpoint","sac")


def test_load_rejects_old_multi_asset_manifest_before_native_load(tmp_path):
    (tmp_path/"single_pair_model.json").write_text(json.dumps(dict(algorithm="sac",symbol="BTCUSDT",observation_shape=[27],action_shape=[2])),encoding="utf-8")
    with pytest.raises(ValueError,match="independent single-pair"):
        models.load_model(tmp_path,"sac")


def test_replay_constructor_rejects_old_dimensions_and_never_creates_trace(tmp_path):
    cls = models.observed_sb3_replay_class(tmp_path/"never-opened.jsonl")
    with pytest.raises(ValueError,match="single-pair replay"):
        cls(16,gym.spaces.Box(-1.,1.,(27,),dtype=np.float32),gym.spaces.Box(-1.,1.,(2,),dtype=np.float32),device="cpu")
    assert not (tmp_path/"never-opened.jsonl").exists()


def test_droq_native_dimensions_auto_entropy_and_safe_checkpoint(monkeypatch,tmp_path):
    # Explicitly synthetic small-network recipe for a bounded real optimizer
    # smoke; the formal adapter's public constructor has no tuning override.
    original_recipe = models.recipe
    def tiny_recipe(algorithm):
        value = original_recipe(algorithm)
        if algorithm == "droq":
            value.update(hidden_sizes=[8,8],replay_size=64,batch_size=4,start_steps=2,
                         delay_update_steps=2,utd_ratio=2,policy_update_delay=2)
        return value
    monkeypatch.setattr(models,"recipe",tiny_recipe)
    env = SyntheticPair()
    trace = tmp_path/"droq-terminal.jsonl"
    model = models.make_model("droq",env,seed=2,trace_path=trace)
    assert (model.obs_dim,model.act_dim,model.target_entropy) == (13,1,-1)
    assert model.alpha == .2
    assert model.replay_buffer.obs1_buf.shape == (64,13)
    assert model.replay_buffer.acts_buf.shape == (64,1)
    assert len(model.q_net_list) == 2
    record = models.train_model(model,env,16,seed=2)
    assert record["counters"]["environment_steps"] == 16
    assert record["counters"]["replay_adds"] == 16
    assert record["counters"]["random_exploration_actions"] == 3
    assert record["counters"]["q_optimizer_steps"] == [28,28]
    assert record["counters"]["policy_optimizer_steps"] == 14
    assert record["counters"]["alpha_optimizer_steps"] == 14
    assert env.resets == 4  # Never replace the final terminal observation by reset.
    boundaries = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines()]
    assert len(boundaries) == 4
    assert all(row["time_limit_truncated"] and row["stored_done"] == 0 for row in boundaries)
    assert all(row["terminal_observation_values"] == row["stored_next_observation_values"] for row in boundaries)
    value = np.linspace(-.25,.25,13,dtype=np.float32)
    expected,_ = models.predict(model,value,"droq")
    manifest = models.save_model(model,tmp_path/"droq","droq")
    source_rng = torch.get_rng_state().clone()
    restored = models.load_model(tmp_path/"droq","droq")
    assert torch.equal(source_rng,torch.get_rng_state())
    actual,_ = models.predict(restored,value,"droq")
    np.testing.assert_array_equal(expected,actual)
    assert models.parameter_digest(restored,"droq") == manifest["parameter_sha256"]
    with pytest.raises(ValueError,match="13 finite"):
        models.predict(restored,np.zeros(27,dtype=np.float32),"droq")


def test_replay_cache_install_is_opt_in_and_refuses_cpu():
    class CPUModel:
        _single_pair_identity = {"algorithm":"droq"}
        device = "cpu"
    with pytest.raises(ValueError,match="unused new CUDA"):
        models.install_replay_cache(CPUModel(),Path("never-read"))


@pytest.mark.parametrize("algorithm",["tqc","crossq","recurrent_ppo"])
def test_native_contrib_single_pair_spaces_when_installed(algorithm):
    pytest.importorskip("sb3_contrib")
    model = models.make_model(algorithm,SyntheticPair(),seed=0)
    assert model.action_space.shape == (1,)
    assert model.observation_space.shape == (13,)
    result,_ = models.predict(model,np.zeros(13,dtype=np.float32),algorithm,episode_start=True)
    assert result.shape == (1,)

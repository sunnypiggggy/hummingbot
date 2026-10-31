"""Boundary and tensor-schema tests; no historical training or strategy selection."""
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import pytest
import torch

from scripts.crypto_rl_elegantrl_adapter import (
    ExplicitReplay, TransitionCollector, make_agent, network_tensors,
    tensor_digest, predict, save_checkpoint, load_checkpoint,
)
from scripts.audit_rl_elegantrl import native_checkpoint_digests

VENDOR = "C:/Users/sunny/.cache/codex-research-vendors/elegantrl-24228304867b"
PARAMS = dict(net_dims=[8, 8], learning_rate=.0003, batch_size=4,
              repeat_times=1., gamma=.9, reward_scale=1.,
              soft_update_tau=.005, num_ensembles=2, ratio_clip=.2,
              lambda_gae_adv=.95, lambda_entropy=0., update_freq=2)


class BoundaryEnv:
    """A timeout's real successor is 2, while reset jumps to 100."""
    action_space = SimpleNamespace(shape=(1,))

    def __init__(self):
        self.resets = self.steps = 0
        self.actions = []

    def reset(self, seed=None):
        self.resets += 1
        return np.array([0. if self.resets == 1 else 100., 1.], np.float32), {}

    def step(self, action):
        self.steps += 1
        self.actions.append(np.asarray(action))
        if self.steps > 3:
            raise AssertionError("Crossed the fixed train endpoint into fake validation")
        x = [1., 2., 101.][self.steps - 1]
        return np.array([x, 1.], np.float32), float(self.steps), self.steps == 3, self.steps == 2, {}


class ToyActor:
    def __init__(self, ppo):
        self.ppo = ppo
        self.calls = []

    def get_action(self, state, **kwargs):
        self.calls.append(kwargs)
        action = torch.full((len(state), 1), .2)
        return (action, torch.full((len(state),), -.7)) if self.ppo else action

    @staticmethod
    def convert_action_for_env(action):
        return action.tanh()


def toy_agent(algorithm):
    calls = []
    def critic(state):
        calls.extend(state[:, 0].tolist())
        return state[:, :1] * 10
    return SimpleNamespace(act=ToyActor(algorithm == "ppo"), cri=critic, gamma=.9,
                           reward_scale=1., explore_noise_std=.05, critic_states=calls)


def test_ppo_real_timeout_bootstrap_and_terminal_split():
    env, agent = BoundaryEnv(), toy_agent("ppo")
    collector = TransitionCollector(env, agent, "ppo", 4)
    states, actions, logp, rewards, undones, masks = collector.collect(3, total_steps=3)
    assert states.shape == (3, 2) and actions.shape == (3, 1)
    assert all(x.shape == (3,) for x in (logp, rewards, undones, masks))
    assert states[:, 0].tolist() == [0., 1., 100.]
    assert rewards.tolist() == [1., 20., 3.]  # r2 + gamma*V(real successor2)
    assert undones.tolist() == [True, False, False] and masks.tolist() == [True] * 3
    assert torch.allclose(logp, torch.full((3,), -.7))  # not upstream collector's zero logprobs
    assert agent.critic_states == [2.]  # never reset100 or terminal101
    assert collector.steps == env.steps == 3 and env.resets == 2
    assert isinstance(agent.last_state, np.ndarray) and agent.last_state.tolist() == [101., 1.]
    assert np.allclose(env.actions[0], np.tanh(.2))


@pytest.mark.parametrize("algorithm", ["sac", "td3"])
def test_explicit_next_state_and_timeout_loss_not_discarded(algorithm):
    env, agent = BoundaryEnv(), toy_agent(algorithm)
    replay = ExplicitReplay(8, 2, 1)
    collector = TransitionCollector(env, agent, algorithm, 1)
    collector.collect(3, total_steps=3, replay=replay)
    state, action, reward, undone, mask, successor = [x[:3] for x in replay.arrays]
    assert state[:, 0].tolist() == [0., 1., 100.]
    assert successor[:, 0].tolist() == [1., 2., 101.]  # timeout predecessor never points to reset100
    assert reward[:, 0].tolist() == [1., 2., 3.]
    assert undone[:, 0].tolist() == [1., 1., 0.] and mask[:, 0].tolist() == [1.] * 3
    assert env.resets == 2 and not agent.critic_states
    if algorithm == "td3":
        assert agent.act.calls == [{"action_std": .05}] * 3
    else:
        assert agent.act.calls == [{}] * 3


def test_replay_wraparound_keeps_explicit_successors_and_column_shapes():
    replay = ExplicitReplay(2, 2, 1)
    for i in (1., 2., 3.):
        replay.add([i, 0], [.2], i, i == 2., [i + 10, 0])
    assert replay.cur_size == 2 and replay.p == 1
    with patch("torch.randint", return_value=torch.tensor([0, 1])):
        batch = replay.sample(2)
    assert [tuple(t.shape) for t in batch] == [(2, 2), (2, 1), (2, 1), (2, 1), (2, 1), (2, 2)]
    assert batch[0][:, 0].tolist() == [3., 2.]
    assert batch[-1][:, 0].tolist() == [13., 12.]
    assert replay.samples_drawn == 2


def test_warmup_is_part_of_interaction_budget():
    env, agent = BoundaryEnv(), toy_agent("td3")
    replay = ExplicitReplay(4, 2, 1)
    collector = TransitionCollector(env, agent, "td3", 1)
    collector.collect(3, total_steps=3, replay=replay, warmup=2)
    assert len(agent.act.calls) == 1 and replay.cur_size == collector.steps == 3


def test_native_ppo_gae_hand_case_never_propagates_random_reset_value():
    agent = make_agent("ppo", VENDOR, PARAMS, state_dim=2, action_dim=1)
    agent.cri = lambda state: state[:, :1] * 10
    agent.last_state = np.array([101., 1.], np.float32)
    states = torch.tensor([[0., 1.], [1., 1.], [100., 1.]])
    rewards = torch.tensor([1., 20., 3.])  # second reward includes .9*V(real_next2)
    undones = torch.tensor([True, False, False])
    masks = torch.ones(3, dtype=torch.bool)
    values = torch.tensor([0., 10., 1000.])
    advantages = agent.get_advantages(states, rewards, undones, masks, values)
    assert torch.allclose(advantages, torch.tensor([18.55, 10., -997.]), atol=1e-5)
    assert torch.allclose(advantages + values, torch.tensor([18.55, 20., 3.]), atol=1e-5)
    agent.last_state = np.array([-999., 1.], np.float32)
    assert torch.equal(agent.get_advantages(states, rewards, undones, masks, values), advantages)


@pytest.mark.parametrize("algorithm", ["ppo", "sac", "td3"])
def test_native_helloworld_update_shapes_and_safe_checkpoint_roundtrip(algorithm, tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(1)
    agent = make_agent(algorithm, VENDOR, PARAMS, state_dim=2, action_dim=1)
    before = tensor_digest(network_tensors(agent))
    if algorithm == "ppo":
        states = torch.arange(16, dtype=torch.float32).reshape(8, 2) / 10
        with torch.no_grad():
            actions, logp = agent.act.get_action(states)
        agent.last_state = np.array([1.6, 1.7], np.float32)
        batch = (states, actions, logp, torch.linspace(-1, 1, 8),
                 torch.tensor([1, 1, 1, 0, 1, 1, 1, 0], dtype=torch.bool),
                 torch.ones(8, dtype=torch.bool))
        torch.set_grad_enabled(False)  # native method builds value targets before enabling update grads
    else:
        batch = ExplicitReplay(16, 2, 1)
        for i in range(8):
            batch.add([i / 10, .5], [.2], i / 100, i == 7, [i / 10 + .1, .5])
    try:
        with patch.object(agent, "update_objectives", wraps=agent.update_objectives) as updates, \
                patch.object(agent.act_optimizer, "step", wraps=agent.act_optimizer.step) as actor_steps:
            stats = agent.update_net(batch)
        assert updates.call_count == 2
        assert actor_steps.call_count == (1 if algorithm == "td3" else 2)
    finally:
        torch.set_grad_enabled(True)
    assert all(np.isfinite(v) for v in stats)
    assert tensor_digest(network_tensors(agent)) != before
    assert all(torch.isfinite(value).all() for value in network_tensors(agent).values())
    checkpoint = tmp_path / "native.pth"
    save_checkpoint(checkpoint, agent, {"algorithm": algorithm, "research_only": True})
    safe_payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert "act" in safe_payload and "metadata" in safe_payload
    independent_all, independent_actor = native_checkpoint_digests(str(checkpoint))
    assert independent_all == tensor_digest(network_tensors(agent))
    assert independent_actor == tensor_digest(agent.act.state_dict())
    restored = make_agent(algorithm, VENDOR, PARAMS, state_dim=2, action_dim=1)
    assert load_checkpoint(checkpoint, restored)["research_only"] is True
    assert tensor_digest(network_tensors(restored)) == tensor_digest(network_tensors(agent))
    obs = np.array([.5, .25], np.float32)
    assert np.array_equal(predict(agent, obs), predict(restored, obs))
    frozen = tensor_digest(network_tensors(restored))
    assert np.array_equal(predict(restored, obs), predict(restored, obs))
    assert tensor_digest(network_tensors(restored)) == frozen

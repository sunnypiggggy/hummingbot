"""Explicit Gymnasium transition adapter for unmodified, pinned ElegantRL agents.

Only collection, bounded update scheduling and checkpoint serialization live here.
The financial environment and upstream actor/critic update methods stay unchanged.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch


def tensor_digest(tensors):
    h = hashlib.sha256()
    for key, value in sorted(tensors.items()):
        array = value.detach().cpu().numpy()
        h.update(key.encode())
        h.update(str(array.shape).encode())
        h.update(str(array.dtype).encode())
        h.update(array.tobytes())
    return h.hexdigest()


def network_tensors(agent):
    tensors = {}
    for name in ("act", "cri", "act_target", "cri_target"):
        module = getattr(agent, name, None)
        if module is not None:
            tensors.update({name + "." + key: value for key, value in module.state_dict().items()})
    if hasattr(agent, "alpha_log"):
        tensors["alpha_log"] = agent.alpha_log
    return tensors


def make_agent(algorithm, vendor, parameters, state_dim=92, action_dim=11):
    vendor = str(Path(vendor).resolve())
    helloworld = str(Path(vendor) / "helloworld")
    if helloworld not in sys.path:
        sys.path.insert(0, helloworld)
    import erl_agent
    from erl_config import Config
    from erl_agent import AgentPPO, AgentSAC, AgentTD3
    if Path(erl_agent.__file__).resolve() != Path(helloworld) / "erl_agent.py":
        raise AssertionError("An unpinned ElegantRL module was loaded")
    cls = {"ppo": AgentPPO, "sac": AgentSAC, "td3": AgentTD3}[algorithm]
    args = Config()
    for key, value in parameters.items():
        setattr(args, key, value)
    args.if_discrete = False
    args.if_off_policy = algorithm != "ppo"
    args.num_envs, args.max_step = 1, 128
    agent = cls(net_dims=parameters["net_dims"], state_dim=state_dim,
                action_dim=action_dim, gpu_id=-1, args=args)
    return agent


class ExplicitReplay:
    """Train-only uniform replay storing the real next observation before reset.

    Exposes the sample interface consumed by native AgentSAC/AgentTD3. A timeout
    retains bootstrap; a true terminal does not. Neither discards its critic loss.
    """

    def __init__(self, capacity, state_dim, action_dim):
        self.max_size, self.cur_size, self.p = capacity, 0, 0
        self.arrays = [torch.empty((capacity, width), dtype=torch.float32)
                       for width in (state_dim, action_dim, 1, 1, 1, state_dim)]
        self.samples_drawn = 0

    def add(self, state, action, reward, terminated, next_state):
        items = (state, action, [reward], [float(not terminated)], [1.], next_state)
        for dest, value in zip(self.arrays, items):
            dest[self.p] = torch.as_tensor(value, dtype=torch.float32)
        self.p = (self.p + 1) % self.max_size
        self.cur_size = min(self.cur_size + 1, self.max_size)

    def sample(self, batch_size):
        if self.cur_size < batch_size:
            raise ValueError("Replay is not ready")
        ids = torch.randint(self.cur_size, (batch_size,))
        self.samples_drawn += batch_size
        result = [array[ids] for array in self.arrays]
        return tuple(result)


class TransitionCollector:
    """Collect upstream actions on the audited ledger without autoreset leakage."""

    def __init__(self, env, agent, algorithm, seed):
        self.env, self.agent, self.algorithm = env, agent, algorithm
        self.obs, _ = env.reset(seed=seed)
        self.steps, self.truncations, self.terminals = 0, 0, 0
        self.bootstrap_adjustments = []

    def collect(self, count, *, total_steps, replay=None, warmup=0):
        states, actions, logprobs, rewards, undones = [], [], [], [], []
        with torch.no_grad():
            for _ in range(count):
                state = torch.as_tensor(self.obs, dtype=torch.float32).unsqueeze(0)
                if self.algorithm == "ppo":
                    action, logprob = self.agent.act.get_action(state)
                    action_for_env = self.agent.act.convert_action_for_env(action)[0].cpu().numpy()
                    states.append(state[0].clone())
                    actions.append(action[0].clone())
                    logprobs.append(logprob[0].clone())
                elif self.steps < warmup:
                    action_for_env = np.random.uniform(-1, 1, self.env.action_space.shape).astype(np.float32)
                else:
                    if self.algorithm == "td3":
                        action = self.agent.act.get_action(state, action_std=self.agent.explore_noise_std)
                    else:
                        action = self.agent.act.get_action(state)
                    action_for_env = action[0].cpu().numpy()
                if not np.isfinite(action_for_env).all():
                    raise AssertionError("Nonfinite upstream action")
                real_next, reward, terminated, truncated, _ = self.env.step(action_for_env)
                self.steps += 1
                self.truncations += int(truncated)
                self.terminals += int(terminated)
                if self.algorithm == "ppo":
                    learned_reward = float(reward) * self.agent.reward_scale
                    if truncated and not terminated:
                        terminal = torch.as_tensor(real_next, dtype=torch.float32).unsqueeze(0)
                        bootstrap = float(self.agent.gamma * self.agent.cri(terminal).reshape(-1)[0])
                        learned_reward += bootstrap
                        self.bootstrap_adjustments.append({"step": self.steps, "bootstrap": bootstrap})
                    rewards.append(learned_reward)
                    undones.append(not (terminated or truncated))
                else:
                    replay.add(self.obs, action_for_env, float(reward) * self.agent.reward_scale,
                               terminated, real_next)
                # Store real_next first. The random reset observation is never a
                # terminal transition's next state, nor its bootstrap state.
                if (terminated or truncated) and self.steps < total_steps:
                    self.obs, _ = self.env.reset()
                else:
                    self.obs = real_next
            self.agent.last_state = np.asarray(self.obs, dtype=np.float32).copy()
        if self.algorithm == "ppo":
            r = torch.tensor(rewards, dtype=torch.float32)
            return (torch.stack(states), torch.stack(actions),
                    torch.stack(logprobs), r,
                    torch.tensor(undones, dtype=torch.bool),
                    torch.ones_like(r, dtype=torch.bool))
        return None


def save_checkpoint(path, agent, metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {name: getattr(agent, name).state_dict()
               for name in ("act", "cri", "act_target", "cri_target")
               if getattr(agent, name, None) is not None}
    payload["metadata"] = json.loads(json.dumps(metadata))
    for name in ("act_optimizer", "cri_optimizer", "alpha_optim"):
        if getattr(agent, name, None) is not None:
            payload[name] = getattr(agent, name).state_dict()
    if hasattr(agent, "alpha_log"):
        payload["alpha_log"] = agent.alpha_log.detach().clone()
    torch.save(payload, path)


def load_checkpoint(path, agent):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    for name in ("act", "cri", "act_target", "cri_target"):
        if name in payload:
            getattr(agent, name).load_state_dict(payload[name], strict=True)
    if "alpha_log" in payload:
        with torch.no_grad():
            agent.alpha_log.copy_(payload["alpha_log"])
    agent.act.eval()
    agent.cri.eval()
    return payload["metadata"]


def predict(agent, obs):
    with torch.no_grad():
        action = agent.act(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0))[0].cpu().numpy()
    if not np.isfinite(action).all() or (np.abs(action) > 1.000001).any():
        raise AssertionError("Invalid deterministic ElegantRL action")
    return action

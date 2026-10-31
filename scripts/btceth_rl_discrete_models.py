"""Native QRDQN/DQN primitives for the isolated BTC/ETH discrete study.

Importing this module loads Gym/NumPy only. Call native factories after a
worker's CUDA UUID isolation. No market loading, model training, deployment,
checkpoint writing or financial-month serialization occurs on import.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import json
from pathlib import Path

import gymnasium as gym
import numpy as np

SEEDS, STEPS, EPISODE_DAYS = tuple(range(5)), 32768, 32
N_ACTIONS, N_QUANTILES, LOWER_HEADS = 21, 200, 50
ACTION_TABLE = ((-1., 0.),) + tuple((g, 2 * share - 1)
    for g in (.25, .5, .75, 1.) for share in (0., .25, .5, .75, 1.))
VARIANTS = {"qrdqn_mean": "qrdqn", "qrdqn_lower25": "qrdqn", "dqn_mean": "dqn"}
COMMON = dict(learning_rate=1e-4, buffer_size=50000, learning_starts=1024,
    batch_size=128, gamma=.99, tau=1., train_freq=4, gradient_steps=1,
    target_update_interval=512, exploration_fraction=.25,
    exploration_initial_eps=1., exploration_final_eps=.05, max_grad_norm=10.,
    optimize_memory_usage=False, n_steps=1)
_REPLAY_CLASS = None


def action_table_sha256():
    return hashlib.sha256(json.dumps(ACTION_TABLE, separators=(",", ":"), allow_nan=False).encode("ascii")).hexdigest()


def _ids(value):
    values = np.asarray(value)
    if values.ndim > 1 or not values.size or values.dtype.kind not in "iu" or np.any(values >= N_ACTIONS) or np.any(values < 0):
        raise ValueError("Require integral action IDs in0..20; no float/bool/coercion")
    return values.astype(np.int64, copy=False)


def box_action(action_id):
    value = _ids(action_id)
    if value.ndim != 0:
        raise ValueError("One scalar discrete action is required")
    return np.asarray(ACTION_TABLE[int(value)], dtype=np.float32).copy()


def _box_batch(action_ids):
    values = _ids(action_ids)
    return np.asarray(ACTION_TABLE, dtype=np.float32)[values].copy()


class DiscretePortfolioActions(gym.ActionWrapper):
    """Only the action representation changes; observation/reset/ledger pass through."""
    def __init__(self, env):
        super().__init__(env)
        if (not isinstance(env.action_space, gym.spaces.Box) or env.action_space.shape != (2,)
                or not np.all(env.action_space.low == -1) or not np.all(env.action_space.high == 1)):
            raise ValueError("Wrap the original two-control Box portfolio environment")
        if env.observation_space.shape != (27,):
            raise ValueError("The unchanged27-dimensional state is required")
        self.action_space = gym.spaces.Discrete(N_ACTIONS)

    def action(self, action):
        return box_action(action)


def mapped_target(env, action_id, *, exposure_multiplier=1.):
    if isinstance(exposure_multiplier, (bool, np.bool_)) or exposure_multiplier not in (1., .75):
        raise ValueError("Only the original raw/cap075 evaluation budgets are declared")
    target = np.asarray(env.action_to_weights(box_action(action_id)), dtype=float).copy()
    if target.shape != (3,) or not np.isfinite(target).all():
        raise ValueError("Original three-asset target required")
    target[:2] *= exposure_multiplier
    target[2] = 1 - target[:2].sum()
    return target


def mean_mapped_targets(env, action_ids, *, exposure_multiplier=1.):
    ids = _ids(action_ids)
    if ids.shape != (5,):
        raise ValueError("Exactly five fixed-seed targets must share one funded account")
    return np.mean([mapped_target(env, i, exposure_multiplier=exposure_multiplier) for i in ids], axis=0)


def training_recipe(algorithm):
    if algorithm not in ("qrdqn", "dqn"):
        raise ValueError("Only the two declared native algorithms")
    value = copy.deepcopy(COMMON)
    value.update(policy="MlpPolicy", n_envs=1, steps=STEPS, seeds=list(SEEDS), episode_days=EPISODE_DAYS,
        policy_kwargs=dict(net_arch=[128, 128], optimizer_class="torch.optim.Adam", optimizer_kwargs=dict(eps=1e-8)),
        replay_buffer_kwargs=dict(handle_timeout_termination=True), action_table_sha256=action_table_sha256())
    if algorithm == "qrdqn":
        value["policy_kwargs"]["n_quantiles"] = N_QUANTILES
    return value


def expected_training_counters():
    return dict(environment_steps=STEPS, replay_adds=STEPS, optimizer_updates=7936,
        target_sync_calls=64, completed_fragments=1024, fragment_days=32,
        final_unused_zero_step_reset=1, expected_episode_history_entries=1025)


def native_class(algorithm):
    """Lazy native imports; worker must already have isolated its assigned UUID."""
    if algorithm not in ("qrdqn", "dqn"):
        raise ValueError("Only native qrdqn/dqn are supported")
    if importlib.metadata.version("stable-baselines3") != "2.9.0":
        raise ValueError("Pinned SB3 2.9.0 required")
    if algorithm == "qrdqn":
        if importlib.metadata.version("sb3-contrib") != "2.9.0":
            raise ValueError("Pinned SB3-Contrib 2.9.0 required")
        from sb3_contrib import QRDQN
        return QRDQN
    from stable_baselines3 import DQN
    return DQN


def native_kwargs(algorithm, *, trace_path=None):
    if algorithm not in ("qrdqn", "dqn"):
        raise ValueError("Only native qrdqn/dqn are supported")
    import torch
    values = copy.deepcopy(COMMON)
    values["policy_kwargs"] = dict(net_arch=[128, 128], optimizer_class=torch.optim.Adam,
                                    optimizer_kwargs=dict(eps=1e-8))
    if algorithm == "qrdqn":
        values["policy_kwargs"]["n_quantiles"] = N_QUANTILES
    values["replay_buffer_class"] = observed_replay_class()
    values["replay_buffer_kwargs"] = dict(handle_timeout_termination=True,
        trace_path=str(trace_path) if trace_path is not None else None)
    return values


def _quantiles_array(value):
    array = np.asarray(value)
    if array.shape == (N_QUANTILES, N_ACTIONS):
        array = array[None]
    if (array.ndim != 3 or array.shape[0] < 1 or array.shape[1:] != (N_QUANTILES, N_ACTIONS)
            or array.dtype.kind not in "fiu"
            or not np.isfinite(array).all()):
        raise ValueError("Finite batch×200 nominal-quantile-heads×21 actions required")
    return array


def quantile_diagnostics(quantiles):
    """Adjacent nominal-head crossing and mean/prefix-head action disagreement.

    Heads retain their nominal tau order. There is no sorting, CVaR training,
    CVaR calibration/error assertion, or guaranteed ordering of predicted heads.
    """
    array = _quantiles_array(quantiles)
    mean_ids = array.mean(axis=1).argmax(axis=1)
    lower_ids = array[:, :LOWER_HEADS].mean(axis=1).argmax(axis=1)
    crossings = int(np.count_nonzero(array[:, 1:] < array[:, :-1]))
    pairs = int(array.shape[0] * (N_QUANTILES - 1) * N_ACTIONS)
    disagree = int(np.count_nonzero(mean_ids != lower_ids))
    return dict(n_quantiles=N_QUANTILES, nominal_lower_head_count=LOWER_HEADS,
        nominal_lower_fraction=.25, heads_sorted=False, cvar_training_or_guarantee=False,
        adjacent_crossings=crossings, adjacent_head_pairs=pairs, quantile_crossing_rate=crossings / pairs,
        mean_action_ids=mean_ids.tolist(), lower25_action_ids=lower_ids.tolist(),
        mean_lower_disagreement_count=disagree, mean_lower_disagreement_rate=disagree / array.shape[0])


class DiscretePredictor:
    """Native policy digest plus explicit discrete-to-Box inference semantics."""
    def __init__(self, model, variant, *, collect_diagnostics=True):
        if variant not in VARIANTS:
            raise ValueError("Undeclared inference variant")
        if (not isinstance(model.action_space, gym.spaces.Discrete) or model.action_space.n != N_ACTIONS
                or int(model.action_space.start) != 0):
            raise ValueError("Loaded native21-action model required")
        if model.observation_space.shape != (27,):
            raise ValueError("Loaded original27-dimensional model required")
        self.model, self.policy, self.variant = model, model.policy, variant
        self.collect_diagnostics, self.last_prediction = bool(collect_diagnostics), None

    def _quantile_values(self, observation):
        import torch
        self.policy.set_training_mode(False)
        tensor, vectorized = self.policy.obs_to_tensor(observation)
        with torch.no_grad():
            quantiles = self.model.quantile_net(tensor)
        if quantiles.ndim != 3 or tuple(quantiles.shape[1:]) != (N_QUANTILES, N_ACTIONS):
            raise ValueError("Actual native200×21 quantile head required")
        if not bool(torch.isfinite(quantiles).all()):
            raise ValueError("Nonfinite native quantile predictions")
        mean_ids = quantiles.mean(dim=1).argmax(dim=1)
        lower_ids = quantiles[:, :LOWER_HEADS].mean(dim=1).argmax(dim=1)
        return quantiles, mean_ids, lower_ids, vectorized

    def predict(self, observation, state=None, episode_start=None, deterministic=True):
        if deterministic is not True:
            raise ValueError("Research replay requires deterministic prediction")
        values = np.asarray(observation)
        if (values.shape != (27,) and not (values.ndim == 2 and values.shape[0] > 0 and values.shape[1] == 27)) or not np.isfinite(values).all():
            raise ValueError("Finite original27-dimensional observations required")
        diagnostic = None
        if self.variant == "qrdqn_lower25" or (self.variant == "qrdqn_mean" and self.collect_diagnostics):
            q, mean_ids, lower_ids, vectorized = self._quantile_values(observation)
            mean_values, lower_values = mean_ids.cpu().numpy(), lower_ids.cpu().numpy()
            if self.variant == "qrdqn_mean":
                ids, state = self.model.predict(observation, state=state, episode_start=episode_start, deterministic=True)
                if not np.array_equal(_ids(ids).reshape(-1), mean_values):
                    raise ValueError("Native mean-action and actual quantile mean disagree")
            else:
                ids = lower_values if vectorized else lower_values.squeeze(axis=0)
            if self.collect_diagnostics:
                diagnostic = quantile_diagnostics(q.detach().cpu().numpy())
                # Action decisions use actual native Torch reductions. CPU
                # diagnostic reductions can differ at floating-point near-ties.
                diagnostic.update(mean_action_ids=mean_values.tolist(), lower25_action_ids=lower_values.tolist(),
                    mean_lower_disagreement_count=int(np.count_nonzero(mean_values != lower_values)),
                    mean_lower_disagreement_rate=float(np.mean(mean_values != lower_values)))
        else:
            ids, state = self.model.predict(observation, state=state, episode_start=episode_start, deterministic=True)
        ids = _ids(ids)
        if (values.ndim == 1 and ids.ndim != 0) or (values.ndim == 2 and ids.shape != (len(values),)):
            raise ValueError("Native action batch shape differs from the observation")
        self.last_prediction = dict(variant=self.variant, action_ids=ids.tolist(), diagnostics=diagnostic)
        return _box_batch(ids), state


def _observation_record(value):
    array = np.asarray(value, dtype="<f4")
    if array.shape != (27,) or not np.isfinite(array).all():
        raise ValueError("Exactly27 finite stored observation values required")
    return array.tolist(), hashlib.sha256(array.tobytes()).hexdigest()


def observed_replay_class():
    """Lazy native subclass. Constructor/load never creates or opens a trace."""
    global _REPLAY_CLASS
    if _REPLAY_CLASS is not None:
        return _REPLAY_CLASS
    from stable_baselines3.common.buffers import ReplayBuffer

    class ObservedReplayBuffer(ReplayBuffer):
        def __init__(self, *args, trace_path=None, **kwargs):
            super().__init__(*args, **kwargs)
            if (self.n_envs != 1 or self.obs_shape != (27,) or self.observations.dtype != np.float32
                    or not isinstance(self.action_space, gym.spaces.Discrete) or self.action_space.n != N_ACTIONS
                    or int(self.action_space.start) != 0 or self.optimize_memory_usage or not self.handle_timeout_termination):
                raise ValueError("Original single-env27-state/21-action timeout-aware replay required")
            self.trace_path = str(trace_path) if trace_path is not None else None
            self._trace_started = False
            self.observed_add_calls = self.observed_transitions = 0
            self.transition_chain_sha256 = hashlib.sha256(b"").hexdigest()
            self.timeout_records = []

        def add(self, obs, next_obs, action, reward, done, infos):
            position = self.pos
            super().add(obs, next_obs, action, reward, done, infos)
            self.observed_add_calls += 1
            self.observed_transitions += self.n_envs
            payload = b"".join(np.asarray(value, dtype=dtype).tobytes() for value, dtype in (
                (self.observations[position], "<f4"), (self.next_observations[position], "<f4"),
                (self.actions[position], "<i8"), (self.rewards[position], "<f4"),
                (self.dones[position], "<f4"), (self.timeouts[position], "<f4")))
            self.transition_chain_sha256 = hashlib.sha256(bytes.fromhex(self.transition_chain_sha256) + payload).hexdigest()
            for i in range(self.n_envs):
                if not self.dones[position, i] and not self.timeouts[position, i]:
                    continue
                before, before_sha = _observation_record(self.observations[position, i])
                after, after_sha = _observation_record(self.next_observations[position, i])
                terminal = infos[i].get("terminal_observation")
                terminal_values, terminal_sha = _observation_record(terminal) if terminal is not None else (None, None)
                stored_done, timeout = float(self.dones[position, i]), float(self.timeouts[position, i])
                row = dict(transition_number=self.observed_transitions, replay_position=position, env_index=i,
                    action_id=int(_ids(np.asarray(self.actions[position, i]).reshape(()))),
                    reward=float(self.rewards[position, i]), stored_done=stored_done, stored_timeout=timeout,
                    effective_done_from_stored_flags=stored_done * (1 - timeout),
                    terminated_from_stored_flags=bool(stored_done and not timeout),
                    handle_timeout_termination=bool(self.handle_timeout_termination),
                    terminal_observation_present=terminal is not None,
                    stored_observation_values=before, stored_observation_sha256=before_sha,
                    stored_next_observation_values=after, stored_next_observation_sha256=after_sha,
                    terminal_observation_values=terminal_values, terminal_observation_sha256=terminal_sha,
                    stored_next_matches_terminal=terminal is not None and after_sha == terminal_sha,
                    info_date=str(infos[i].get("date")), transition_chain_sha256=self.transition_chain_sha256)
                row["TimeLimit.truncated"] = bool(infos[i].get("TimeLimit.truncated", False))
                if self.trace_path is not None:
                    mode = "a" if self._trace_started else "x"
                    with Path(self.trace_path).open(mode, encoding="utf-8", newline="\n") as stream:
                        stream.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")
                    self._trace_started = True
                self.timeout_records.append(row)

    ObservedReplayBuffer.__module__ = __name__
    ObservedReplayBuffer.__qualname__ = "ObservedReplayBuffer"
    globals()["ObservedReplayBuffer"] = ObservedReplayBuffer
    _REPLAY_CLASS = ObservedReplayBuffer
    return ObservedReplayBuffer


def __getattr__(name):
    # Allows a native checkpoint's class reference to resolve lazily without
    # making ordinary module import initialize Torch or SB3.
    if name == "ObservedReplayBuffer":
        return observed_replay_class()
    raise AttributeError(name)

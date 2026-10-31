"""Independent single-pair RL model adapters (13 observations, one action).

This file never loads market data, mixes models, transfers cash, or dispatches
training. Constructors use the reviewed native algorithms; orchestration and
the frozen experiment protocol live outside this module. Previous two-asset
checkpoints cannot be loaded through this interface.
"""
from __future__ import annotations

import copy
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import inspect
import json
import os
from pathlib import Path
import random
import time

import numpy as np

OBSERVATIONS, ACTIONS = 13, 1
ALGORITHMS = ("sac", "tqc", "crossq", "recurrent_ppo", "td3", "droq")
SEEDS = (0, 1, 2, 3, 4)
PAIRS = ("BTCUSDT", "ETHUSDT")
STEPS = {algorithm: 65536 if algorithm == "td3" else 32768 for algorithm in ALGORITHMS}
GPU_CACHE_SHA256 = "16e176531ac416e3dc25b60c752e1bef037a003a37edafcc51c5ba2887019009"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def write_new(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def observation(values):
    values = np.asarray(values, dtype="<f4")
    if values.shape != (OBSERVATIONS,) or not np.isfinite(values).all():
        raise ValueError("Exactly 13 finite single-pair observations required")
    return values.tolist(), hashlib.sha256(values.tobytes()).hexdigest()


def action(values):
    values = np.asarray(values, dtype=np.float32)
    if values.shape != (ACTIONS,) or not np.isfinite(values).all() or (np.abs(values) > 1 + 1e-6).any():
        raise ValueError("Exactly one finite normalized action required")
    return np.clip(values, -1., 1.)


def seed_everything(seed, device="cpu"):
    if type(seed) is not int or seed not in SEEDS:
        raise ValueError("Use the five predeclared integer seeds 0..4")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if str(device).startswith("cuda"):
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1 or str(device) not in ("cuda", "cuda:0"):
            raise RuntimeError("Use exactly one UUID-isolated CUDA GPU per model worker")
        torch.cuda.manual_seed_all(seed)


def environment_identity(env):
    raw = env.unwrapped
    symbols = tuple(raw.SYMBOLS)
    if symbols not in (("BTCUSDT",), ("ETHUSDT",)):
        raise ValueError("Exactly one allowed trading pair per model")
    if env.observation_space.shape != (OBSERVATIONS,) or env.action_space.shape != (ACTIONS,):
        raise ValueError("Single-pair model requires observation13/action1 spaces")
    if (not np.all(env.action_space.low == -1) or not np.all(env.action_space.high == 1)
            or np.dtype(env.action_space.dtype) != np.dtype("float32")):
        raise ValueError("A normalized FP32 one-dimensional action Box is required")
    return dict(symbol=symbols[0], observation_shape=[OBSERVATIONS], action_shape=[ACTIONS],
                independent_account=True, multi_model_target_average=False)


def recipe(algorithm):
    """JSON recipe; action-dimension adaptations are explicit, not retuning."""
    if algorithm not in ALGORITHMS:
        raise ValueError("Undeclared algorithm")
    if algorithm in ("sac", "tqc", "crossq", "recurrent_ppo"):
        from scripts.backtest_btceth_rl import hyperparameters
        return copy.deepcopy(hyperparameters(algorithm))
    if algorithm == "td3":
        from scripts.btceth_rl_td3 import recipe as td3_recipe
        values = copy.deepcopy(td3_recipe())
        values["action_noise"]["mean"] = [0.]
        values["action_noise"]["sigma"] = [.1]
        return values
    from scripts.btceth_rl_droq import recipe as droq_recipe
    return dict(droq_recipe(), obs_dim=OBSERVATIONS, act_dim=ACTIONS, act_limit=1.,
                effective_target_entropy=-ACTIONS)


def native_class(algorithm):
    if algorithm in ("sac", "td3"):
        from stable_baselines3 import SAC, TD3
        return {"sac": SAC, "td3": TD3}[algorithm]
    if algorithm == "tqc":
        from sb3_contrib import TQC
        return TQC
    if algorithm == "crossq":
        from sb3_contrib import CrossQ
        return CrossQ
    if algorithm == "recurrent_ppo":
        from sb3_contrib import RecurrentPPO
        return RecurrentPPO
    raise ValueError("DroQ uses the verified author's native Agent")


def observed_sb3_replay_class(trace_path=None):
    """Observe actual stored transitions, including terminal next observations.

    The constructor does not create/open a trace file. Loading a native model
    for inference therefore cannot append to its original training evidence.
    """
    from stable_baselines3.common.buffers import ReplayBuffer

    class SinglePairReplayBuffer(ReplayBuffer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            if (self.obs_shape != (OBSERVATIONS,) or self.action_dim != ACTIONS
                    or self.n_envs != 1 or self.optimize_memory_usage or not self.handle_timeout_termination):
                raise ValueError("Original single-pair replay/timeout contract required")
            self.observed_add_calls = self.observed_transitions = 0
            self.transition_digest = hashlib.sha256()
            self.timeout_records = []

        def add(self, obs, next_obs, act, reward, done, infos):
            position = self.pos
            super().add(obs, next_obs, act, reward, done, infos)
            self.observed_add_calls += 1
            self.observed_transitions += self.n_envs
            for values in (self.observations[position], self.actions[position], self.rewards[position],
                           self.next_observations[position], self.dones[position], self.timeouts[position]):
                self.transition_digest.update(np.asarray(values, dtype="<f4").tobytes())
            if bool(np.asarray(done).reshape(-1)[0]):
                terminal, terminal_sha = observation(np.asarray(next_obs)[0])
                stored, stored_sha = observation(self.next_observations[position, 0])
                record = dict(transition_number=self.observed_transitions, replay_position=position,
                    done=float(self.dones[position, 0]), timeout=float(self.timeouts[position, 0]),
                    effective_done=float(self.dones[position, 0] * (1 - self.timeouts[position, 0])),
                    terminal_observation_values=terminal, terminal_observation_sha256=terminal_sha,
                    stored_next_observation_values=stored, stored_next_observation_sha256=stored_sha)
                self.timeout_records.append(record)
                if trace_path is not None:
                    with Path(trace_path).open("a", encoding="utf-8", newline="\n") as stream:
                        stream.write(json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n")

    return SinglePairReplayBuffer


def native_kwargs(algorithm, trace_path=None):
    values = recipe(algorithm)
    policy = values.pop("policy")
    if algorithm == "td3":
        from stable_baselines3.common.noise import NormalActionNoise
        from torch import nn
        from torch.optim import Adam
        values["train_freq"] = tuple(values["train_freq"])
        values["policy_kwargs"]["activation_fn"] = nn.ReLU
        values["policy_kwargs"]["optimizer_class"] = Adam
        noise = values.pop("action_noise")
        values["action_noise"] = NormalActionNoise(np.asarray(noise["mean"]), np.asarray(noise["sigma"]), dtype=np.float32)
    if algorithm != "recurrent_ppo":
        values["replay_buffer_class"] = observed_sb3_replay_class(trace_path)
        values.setdefault("replay_buffer_kwargs", {}).update(handle_timeout_termination=True)
    return policy, values


def _droq():
    from scripts import btceth_rl_droq
    return btceth_rl_droq


def _observed_droq_replay(raw, trace_path=None):
    # A local subclass is intentional: generic legacy counter/checkpoint helpers
    # recognize the observer, while no old 27-feature observation function runs.
    class SinglePairObservedReplay(_droq().ObservedReplay):
        def store(self, obs, act, reward, next_obs, done):
            position = self.raw.ptr
            observation(obs)
            observation(next_obs)
            action(act)
            self.raw.store(obs, act, reward, next_obs, done)
            self.writes += 1
            for value in (self.raw.obs1_buf[position], self.raw.acts_buf[position],
                          [self.raw.rews_buf[position]], self.raw.obs2_buf[position], [self.raw.done_buf[position]]):
                self.digest.update(np.asarray(value, dtype="<f4").tobytes())
            if self.pending and (self.pending["terminated"] or self.pending["truncated"]):
                before, before_sha = observation(self.raw.obs1_buf[position])
                stored, stored_sha = observation(self.raw.obs2_buf[position])
                terminal, terminal_sha = observation(next_obs)
                record = dict(transition_number=self.writes, replay_position=position, env_index=0,
                    done=float(bool(self.pending["terminated"] or self.pending["truncated"])),
                    timeout=float(bool(self.pending["truncated"] and not self.pending["terminated"])),
                    effective_done=float(bool(self.pending["terminated"])), stored_done=float(self.raw.done_buf[position]),
                    time_limit_truncated=bool(self.pending["truncated"] and not self.pending["terminated"]),
                    terminated=bool(self.pending["terminated"]), truncated=bool(self.pending["truncated"]),
                    info_date=str(self.pending.get("date")),
                    terminal_observation_values=terminal, terminal_observation_sha256=terminal_sha,
                    stored_next_observation_values=stored, stored_next_observation_sha256=stored_sha,
                    stored_observation_values=before, stored_observation_sha256=before_sha)
                self.boundaries.append(record)
                if self.path:
                    with self.path.open("a", encoding="utf-8", newline="\n") as stream:
                        stream.write(json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n")
            self.pending = None
    return SinglePairObservedReplay(raw, trace_path)


def make_model(algorithm, env, seed, device="cpu", upstream_root=None, *, trace_path=None):
    identity = environment_identity(env)
    seed_everything(seed, device)
    if algorithm == "droq":
        upstream_root = Path(upstream_root or _droq().DEFAULT_UPSTREAM)
        cls, _ = _droq().load_upstream(upstream_root)
        values = recipe(algorithm)
        for key in ("obs_dim", "act_dim", "act_limit", "effective_target_entropy"):
            values.pop(key)
        values["hidden_sizes"] = tuple(values["hidden_sizes"])
        model = cls(identity["symbol"], obs_dim=OBSERVATIONS, act_dim=ACTIONS, act_limit=1., device=device, **values)
        model.replay_buffer = _observed_droq_replay(model.replay_buffer, trace_path)
        model.q_target_net_list = [_droq().ObservedTarget(target) for target in model.q_target_net_list]
        if (model.obs_dim, model.act_dim, model.target_entropy) != (OBSERVATIONS, ACTIONS, -ACTIONS):
            raise ValueError("Native DroQ dimensions/automatic target entropy differ")
    else:
        policy, values = native_kwargs(algorithm, trace_path)
        model = native_class(algorithm)(policy, env, seed=seed, device=device, verbose=0, **values)
    model._single_pair_identity = dict(identity, algorithm=algorithm, seed=seed)
    model._single_pair_training = None
    return model


def parameter_digest(model, algorithm):
    if algorithm == "droq":
        return _droq().parameter_digest(model)
    from scripts.backtest_btceth_rl import parameter_digest as digest
    return digest(model.policy)


def effective_parameters(model, algorithm):
    if algorithm == "droq":
        return dict(_droq().effective_parameters(model), obs_dim=int(model.obs_dim), act_dim=int(model.act_dim))
    if algorithm == "td3":
        from scripts.btceth_rl_td3 import effective_parameters as td3_parameters
        return td3_parameters(model)
    values = {key: getattr(model, key) for key in ("learning_rate", "gamma", "n_steps") if hasattr(model, key)}
    for key in ("buffer_size", "learning_starts", "batch_size", "tau", "gradient_steps", "n_epochs", "gae_lambda", "ent_coef", "policy_delay", "top_quantiles_to_drop_per_net"):
        value = getattr(model, key, None)
        if value is not None and type(value) in (str, int, float, bool):
            values[key] = value
    if hasattr(model, "train_freq"):
        values["train_freq"] = [model.train_freq.frequency, model.train_freq.unit.value]
    values.update(observation_shape=list(model.observation_space.shape), action_shape=list(model.action_space.shape),
                  native_class=type(model).__name__)
    return values


def actual_counters(model, algorithm):
    if algorithm == "droq":
        return _droq().actual_counters(model)
    result = dict(environment_steps=int(model.num_timesteps), gradient_update_rounds=int(model._n_updates))
    optimizers = {}
    for name in ("actor", "critic", "policy"):
        optimizer = getattr(getattr(model, name, None), "optimizer", None)
        if optimizer is not None:
            optimizers[name] = _droq().optimizer_steps(optimizer)
    if getattr(model, "ent_coef_optimizer", None) is not None:
        optimizers["entropy"] = _droq().optimizer_steps(model.ent_coef_optimizer)
    result["optimizer_steps"] = optimizers
    buffer = getattr(model, "replay_buffer", None)
    if buffer is not None:
        result.update(replay_add_calls=int(buffer.observed_add_calls), replay_transitions=int(buffer.observed_transitions))
    return result


def train_model(model, env, steps, seed, progress_path=None):
    """Train exactly a new model's requested interactions; never resume it."""
    identity = environment_identity(env)
    if (type(steps) is not int or steps <= 0 or type(seed) is not int or seed not in SEEDS
            or model._single_pair_identity != dict(identity, algorithm=model._single_pair_identity["algorithm"], seed=seed)):
        raise ValueError("Positive exact training budget and original single-pair identity required")
    if model._single_pair_training is not None:
        raise ValueError("Training an existing fitted model is not a supported resume")
    algorithm = model._single_pair_identity["algorithm"]
    if algorithm == "recurrent_ppo" and steps % model.n_steps:
        raise ValueError("RecurrentPPO budget must be a complete rollout multiple")
    started = time.monotonic()
    def progress(actual_steps):
        if progress_path is not None:
            row = dict(at=now(), actual_steps=actual_steps, elapsed_seconds=time.monotonic()-started,
                       counters=actual_counters(model, algorithm))
            with Path(progress_path).open("a", encoding="utf-8", newline="\n") as stream:
                stream.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")
    if algorithm != "droq":
        from stable_baselines3.common.callbacks import BaseCallback
        if model.num_timesteps != 0:
            raise ValueError("Loaded/trained policies cannot be retrained")
        class Progress(BaseCallback):
            def _on_step(self):
                if self.num_timesteps % 2048 == 0:
                    progress(int(self.num_timesteps))
                return True
        model.learn(total_timesteps=steps, callback=Progress(), reset_num_timesteps=True)
        if model.num_timesteps != steps:
            raise ValueError("Native learner exceeded its predeclared interaction budget")
        counters = actual_counters(model, algorithm)
        replay = getattr(model, "replay_buffer", None)
        trace_digest = replay.transition_digest.hexdigest() if replay is not None else None
        if replay is not None and replay.observed_transitions != steps:
            raise ValueError("Every actual environment step must be stored once")
    else:
        if model.replay_buffer.writes or model.replay_buffer.size:
            raise ValueError("DroQ training starts from an unused native Agent")
        obs, _ = env.reset(seed=seed)
        env.action_space.seed(seed)
        observation(obs)
        logger = _droq().MetricSink()
        random_actions = 0
        for step in range(1, steps + 1):
            random_actions += int(model.replay_buffer.size <= model.start_steps)
            act = action(model.get_exploration_action(obs, env))
            next_obs, reward, terminated, truncated, info = env.step(act)
            observation(next_obs)
            model.replay_buffer.pending = dict(terminated=bool(terminated), truncated=bool(truncated), date=info.get("date"))
            model.store_data(obs, act, reward, next_obs, bool(terminated))
            _droq().original_train(model, logger)
            if step % 2048 == 0:
                progress(step)
            if (terminated or truncated) and step < steps:
                obs, _ = env.reset()
            elif not (terminated or truncated):
                obs = next_obs
        counters = _droq().actual_counters(model, environment_steps=steps, random_actions=random_actions)
        if counters["replay_adds"] != steps:
            raise ValueError("Every actual DroQ interaction must be stored once")
        trace_digest = model.replay_buffer.digest.hexdigest()
    result = dict(algorithm=algorithm, symbol=identity["symbol"], seed=seed, completed_at=now(),
        elapsed_seconds=time.monotonic()-started, actual_steps=steps, counters=counters,
        transition_sha256=trace_digest, effective_parameters=effective_parameters(model, algorithm),
        training_episode_history=list(getattr(env.unwrapped, "training_episode_history", [])),
        training_window_quality=getattr(env.unwrapped, "training_window_metadata", {}))
    model._single_pair_training = result
    progress(steps)
    return result


def runtime_versions():
    result = {}
    for name in ("torch", "numpy", "gymnasium", "stable-baselines3", "sb3-contrib"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def save_model(model, directory, algorithm, *, upstream_root=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = directory / "single_pair_model.json"
    if any((directory/name).exists() for name in ("single_pair_model.json", "model.zip", "model.json", "model_tensors.pt")):
        raise FileExistsError("Preserve every attempted single-pair checkpoint")
    identity = getattr(model, "_single_pair_identity", None)
    training = getattr(model, "_single_pair_training", None)
    if not identity or identity.get("algorithm") != algorithm or not training:
        raise ValueError("Only a genuinely trained single-pair model can be saved")
    if algorithm == "droq":
        checkpoint = _droq().save_checkpoint(directory, model, counters=training["counters"],
                                            upstream=upstream_root or _droq().DEFAULT_UPSTREAM)
        files = {name: sha(directory/name) for name in ("model.json", "model_tensors.pt")}
        checkpoint_parameter_sha = checkpoint["parameter_sha256"]
    else:
        model.save(directory / "model.zip")
        files = {"model.zip": sha(directory/"model.zip")}
        checkpoint_parameter_sha = parameter_digest(model, algorithm)
    manifest = dict(schema_version=1, **identity, recipe=recipe(algorithm), training=training,
        parameter_sha256=checkpoint_parameter_sha, checkpoint_files=files, runtime_packages=runtime_versions(),
        adapter_sha256=sha(__file__), no_optimizer_or_replay_resume=True,
        resume_note="Native optimizer state is retained as evidence; these inference loaders do not resume training.")
    write_new(manifest_path, manifest)
    return manifest


class SinglePairDroQPolicy:
    def __init__(self, actor, other):
        self.actor, self._other, self.policy = actor, other, self

    def state_dict(self):
        return {**self._other, **{"policy."+name:value for name,value in self.actor.state_dict().items()}}

    def predict(self, obs, state=None, episode_start=None, deterministic=True):
        if deterministic is not True:
            raise ValueError("Only deterministic native mean-action replay is supported")
        import torch
        observation(obs)
        device = next(self.actor.parameters()).device
        with torch.no_grad():
            result = self.actor(torch.as_tensor(obs,dtype=torch.float32,device=device).unsqueeze(0), deterministic=True,return_log_prob=False)[0]
        return action(result.detach().cpu().numpy().reshape(ACTIONS)), None


def load_model(directory, algorithm, device="cpu", upstream_root=None):
    directory = Path(directory)
    manifest = json.loads((directory/"single_pair_model.json").read_text(encoding="utf-8"))
    if (manifest.get("algorithm") != algorithm or manifest.get("symbol") not in PAIRS
            or manifest.get("observation_shape") != [OBSERVATIONS] or manifest.get("action_shape") != [ACTIONS]
            or manifest.get("independent_account") is not True or manifest.get("multi_model_target_average") is not False
            or manifest.get("adapter_sha256") != sha(__file__)):
        raise ValueError("Checkpoint is not this frozen independent single-pair model")
    for name, expected in manifest["checkpoint_files"].items():
        if name not in ("model.zip", "model.json", "model_tensors.pt") or sha(directory/name) != expected:
            raise ValueError("Checkpoint identity or bytes changed")
    if algorithm == "droq":
        import torch
        checkpoint, tensors = _droq().read_checkpoint(directory)
        if checkpoint["effective_target_entropy"] != -ACTIONS:
            raise ValueError("Old multi-asset DroQ checkpoint is incompatible")
        _, core = _droq().load_upstream(upstream_root or _droq().DEFAULT_UPSTREAM)
        devices = [torch.device(device).index or 0] if str(device).startswith("cuda") else []
        with torch.random.fork_rng(devices=devices):
            actor = core.TanhGaussianPolicy(OBSERVATIONS, ACTIONS, tuple(checkpoint["hidden_sizes"]), action_limit=checkpoint["action_limit"]).to(device)
        actor.load_state_dict({name:tensors[key] for name,key in checkpoint["tensor_groups"]["policy"]["named_tensors"].items()}, strict=True)
        actor.eval()
        other = {group+"."+name:tensors[key] for group in _droq().MODEL_GROUPS if group != "policy"
                 for name,key in checkpoint["tensor_groups"][group]["named_tensors"].items()}
        model = SinglePairDroQPolicy(actor, other)
    else:
        model = native_class(algorithm).load(directory/"model.zip",device=device)
        if model.observation_space.shape != (OBSERVATIONS,) or model.action_space.shape != (ACTIONS,):
            raise ValueError("Loaded native spaces differ")
        model.policy.set_training_mode(False)
    if parameter_digest(model,algorithm) != manifest["parameter_sha256"]:
        raise ValueError("Loaded actual model tensors differ from their checkpoint")
    model._single_pair_identity = {key:manifest[key] for key in ("symbol","observation_shape","action_shape","independent_account","multi_model_target_average","algorithm","seed")}
    model._single_pair_training = manifest["training"]
    return model


def predict(model, obs, algorithm, state=None, episode_start=False):
    observation(obs)
    if model._single_pair_identity["algorithm"] != algorithm:
        raise ValueError("Algorithm identity differs")
    if algorithm == "droq" and not hasattr(model,"predict"):
        return action(model.get_test_action(obs)), None
    result, state = model.predict(obs,state=state,episode_start=np.asarray([bool(episode_start)]),deterministic=True)
    return action(np.asarray(result).reshape(ACTIONS)), state


def install_replay_cache(model, performance_root):
    """Optional dynamic-shape transport; caller must first prove GPU13/1 equality.

    This reuses the pinned general cache class only. It intentionally does not
    call the old 27/2 installer and never patches native module globals/Polyak.
    """
    import torch
    if (model._single_pair_identity["algorithm"] != "droq" or str(model.device) != "cuda:0"
            or model.replay_buffer.writes != 0 or model.replay_buffer.size != 0):
        raise ValueError("Install only on an unused new CUDA single-pair DroQ model")
    path = Path(performance_root)/"benchmark_gpu_replay_cache.py"
    if sha(path) != GPU_CACHE_SHA256:
        raise ValueError("Only the reviewed generic transport class is allowed")
    spec = importlib.util.spec_from_file_location("single_pair_reviewed_replay_cache",path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    before_np, before_cpu, before_cuda = np.random.get_state(), torch.get_rng_state().clone(), torch.cuda.get_rng_state().clone()
    cache = module.GPUReplayCache(model)
    after_np = np.random.get_state()
    if (before_np[0] != after_np[0] or not np.array_equal(before_np[1],after_np[1]) or before_np[2:] != after_np[2:]
            or not torch.equal(before_cpu,torch.get_rng_state()) or not torch.equal(before_cuda,torch.cuda.get_rng_state())):
        raise RuntimeError("Replay transport installation consumed randomness")
    return cache


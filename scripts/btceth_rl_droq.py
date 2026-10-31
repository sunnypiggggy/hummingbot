"""Independent DroQ source preparation and bounded research primitives.

The pinned author's original Agent/core implement the learning mathematics.
This module supplies a Gymnasium driver, observational evidence, safe tensor
checkpoints and cash-funded portfolio evaluation.  It never edits an older
algorithm registry.  Formal market/GPU execution needs a separate frozen
protocol and explicit authorization; this source-preparation entry runs none.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
import hashlib
import importlib
import json
import math
import multiprocessing as mp
import os
from pathlib import Path, PurePosixPath
import random
import shutil
import subprocess
import sys
import time

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_name] = "1"
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from scripts import backtest_btceth_rl as shared
from scripts.crypto_rl_reporting import canonical_csv, result_fingerprints, save_result, summarize

UPSTREAM_COMMIT = "4c7adddbaf7573ff9256b16091a3bbdbd536aad4"
UPSTREAM_MANIFEST_SHA256 = "9b0eb6b5e08fefae4a468061ab31e93bcd4d67606750b6196e926f6e103e7560"
EFFECTIVE_CONFIG_SHA256 = "d76e838b65d333c0669b5b138aa5bb8e81efe9e88921c25a0d00feb266828c1a"
DEFAULT_UPSTREAM = ROOT / "results/backtests/btceth_rl_droq_20261030_preflight/upstream/DroQ"
SOURCE_FILES = (*shared.SOURCE_FILES, "scripts/audit_btceth_rl.py", "scripts/audit_btceth_rl_exposure_cap.py", "scripts/audit_btceth_rl_execution_band.py", "scripts/btceth_rl_droq.py", "scripts/audit_btceth_rl_droq.py")
TRAIN, VALIDATION, TEST = shared.TRAIN, shared.VALIDATION, shared.TEST
SEEDS, STEPS, EPISODE_DAYS = (0, 1, 2, 3, 4), 32768, 32
PROFILE = "balanced"
PROFILE_PARAMETERS = dict(drawdown_penalty=.25, rebalance_band=.025)
BUDGETS = {"raw": 1., "cap075": .75}
COSTS = copy.deepcopy(shared.COSTS)
FLAGS = ("designed_after_first_annual", "test_year_already_seen", "not_fresh_holdout")
CASE_FILES = ("equity_1h.csv.gz", "fills.csv", "weights.csv", "boundary_equity.csv", "metadata.json", "fingerprints.json", "member_targets.csv", "replay.json")
GROUP_FILES = ("summary.csv", "daily_equity.csv", "monthly_returns.csv", "completion.json")
TARGETS = {"primary_base_net_return_at_least": .05, "primary_base_max_drawdown_magnitude_at_most": .20,
    "primary_stress_net_return_at_least": 0., "primary_beats_equal_hold_return_and_drawdown": True,
    "positive_base_training_seeds_at_least": 3, "stronger_claim_requires_beating_simple_trend_controls": True, "targets_are_not_guarantees": True}
MODEL_GROUPS = ("policy", "q0", "q1", "target_q0", "target_q1", "log_alpha")
now, sha = shared.now, shared.sha


def relative(root, name):
    text = str(name).replace("\\", "/"); parsed = PurePosixPath(text)
    if not text or parsed.is_absolute() or ".." in parsed.parts or ":" in text:
        raise ValueError("Research evidence paths must remain relative")
    root = Path(root)
    if root.is_symlink() or hasattr(root, "is_junction") and root.is_junction():
        raise ValueError("Research roots cannot be links or junctions")
    root = root.resolve(); result = root
    for part in parsed.parts:
        result /= part
        if result.is_symlink() or hasattr(result, "is_junction") and result.is_junction():
            raise ValueError("Research evidence cannot cross links or junctions")
    result.resolve().relative_to(root)
    return result


def write_new(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False); stream.write("\n")


def recipe():
    """Author defaults plus the explicitly declared .01/LN/two-Q switches."""
    return dict(hidden_sizes=[256, 256], replay_size=1000000, batch_size=256, lr=.0003, gamma=.99, polyak=.995,
        alpha=.2, auto_alpha=True, target_entropy="auto", start_steps=5000, delay_update_steps="auto",
        utd_ratio=20, num_Q=2, num_min=2, q_target_mode="min", policy_update_delay=20, target_drop_rate=.01, layer_norm=True)


def experiment_spec():
    return dict(algorithm="droq", profile=PROFILE, profile_parameters=PROFILE_PARAMETERS.copy(), seeds=list(SEEDS),
        profiles={PROFILE: PROFILE_PARAMETERS.copy()},
        timesteps_per_model=STEPS, model_count=5, train_episode_days=EPISODE_DAYS, train=list(TRAIN), validation=list(VALIDATION), test=list(TEST),
        capital=10000., costs=copy.deepcopy(COSTS), budgets=BUDGETS.copy(), cash_floor=.05, max_asset_weight=.95,
        hyperparameters=recipe(), initial_effective_alpha=.2, initial_log_alpha=0., effective_target_entropy=-2,
        expected_counters=dict(environment_steps=32768, replay_adds=32768, random_exploration_actions=5001, learning_train_calls=27768,
            q_inner_rounds=555360, q_optimizer_steps=[555360, 555360], total_q_optimizer_steps=1110720,
            target_updates_per_network=[555360, 555360], policy_optimizer_steps=27768, alpha_optimizer_steps=27768),
        timeout_boundaries=1024,
        candidate_count=2, validation_cases=24, validation_repeats=24, test_cases=12, test_repeats=12, baseline_reference_cases=14,
        official_uniform_crypto_defaults=False, recipe_is_pure_official_default=False, dropout_rate_is_prespecified_research_assumption=True,
        seed_selection=False, selection_uses_test=False, new_recipe_search=False, annual_requires_separate_authorization=True,
        training_budget_multiple=1, environment_step_budget_multiple=1, compute_budget_comparable=False,
        training_budget_multiple_units="environment interactions relative to the original32768 only; G20 Q optimization has substantially greater compute",
        q_inner_rounds_per_model=555360, individual_q_optimizer_steps_per_model=1110720,
        selection_score="mean(log1p(net_return)-0.5*abs(max_drawdown)-2*max(abs(max_drawdown)-0.20,0)) across base/stress",
        tie_break="raw before cap075, then profile lexical identifier", research_targets=copy.deepcopy(TARGETS),
        upstream_commit=UPSTREAM_COMMIT, upstream_manifest_sha256=UPSTREAM_MANIFEST_SHA256, effective_config_sha256=EFFECTIVE_CONFIG_SHA256,
        optimizer_order="Q backward; final-inner actor backward; alpha step; Q optimizer steps; final-inner actor step; target Polyak each inner",
        clipping_and_cash_mapping="Original unmodified financial environment; no new action bias/filter/stop",
        inference_adapter="Rebuild only the deterministic original actor; retain all Q/target/LN/alpha tensors for independent structural and optimizer audit, without reconstructing an Agent or enabling target dropout at inference",
        effective_contract_refinement="The upstream preparation contract's entire-Agent replay wording is refined to actor-only deterministic inference; training and all saved Q/target states retain the original source mathematics",
        bounded_driver=True, official_main_entrypoint_used=False, reseed_each_epoch=False, reset_after_final_step=False,
        **{flag: True for flag in FLAGS})


def verify_upstream(root):
    root = Path(root)
    manifest_path = relative(root, "verified_manifest.json")
    if sha(manifest_path) != UPSTREAM_MANIFEST_SHA256:
        raise ValueError("Use the reviewed complete author source manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = manifest.get("files", [])
    if (manifest.get("commit") != UPSTREAM_COMMIT or manifest.get("license") != "MIT" or manifest.get("file_count") != 27
            or manifest.get("all_git_blob_sha1_verified") is not True or len(files) != 27):
        raise ValueError("Pinned27 author sources and MIT notices required")
    found = {}
    for item in files:
        path = relative(root, item["path"]); content = path.read_bytes()
        blob = hashlib.sha1(b"blob " + str(len(content)).encode() + b"\0" + content).hexdigest()
        if len(content) != item["size"] or sha(path) != item["sha256"] or blob != item["git_blob_sha1"]:
            raise ValueError("Official source byte or Git blob changed: " + item["path"])
        found[item["path"]] = item["sha256"]
    return manifest, found


def load_upstream(root=DEFAULT_UPSTREAM):
    """Normal verified imports in a clean worker; no AST/modules patch."""
    _, files = verify_upstream(root); root = Path(root).resolve()
    package = root / "OriginalREDQCodebase"
    for name, module in list(sys.modules.items()):
        if name == "redq" or name.startswith("redq."):
            source = getattr(module, "__file__", None)
            if source is None or not Path(source).resolve().is_relative_to(package):
                raise ValueError("Foreign redq import cache; use a clean research worker")
    old_path, old_bytecode = list(sys.path), sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        sys.path.insert(0, str(package))
        core = importlib.import_module("redq.algos.core")
        algorithm = importlib.import_module("redq.algos.redq_sac")
    finally:
        sys.path[:] = old_path
        sys.dont_write_bytecode = old_bytecode
    for name in ("redq", "redq.algos", "redq.algos.core", "redq.algos.redq_sac"):
        module = sys.modules[name]; expected = package / (name.replace(".", "/") + ("/__init__.py" if name in ("redq", "redq.algos") else ".py"))
        if Path(module.__file__).resolve() != expected.resolve() or sha(expected) != files[expected.relative_to(root).as_posix()]:
            raise ValueError("Actually imported author module differs: " + name)
    if any(getattr(algorithm, key) is not getattr(core, key) for key in ("Mlp", "ReplayBuffer", "TanhGaussianPolicy", "soft_update_model1_with_model2")):
        raise ValueError("Author Agent must use the actual reviewed core objects")
    return algorithm.REDQSACAgent, core


def seed_cpu(seed):
    import torch
    torch.set_num_threads(1)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def make_agent(upstream=DEFAULT_UPSTREAM, *, device="cpu", overrides=None):
    """Overrides exist for synthetic CPU fixtures, never a formal recipe search."""
    cls, _ = load_upstream(upstream)
    values = recipe(); values.update(overrides or {}); values["hidden_sizes"] = tuple(values["hidden_sizes"])
    return cls("BTCETH", obs_dim=27, act_dim=2, act_limit=1., device=device, **values)


def loaded_sources(upstream=DEFAULT_UPSTREAM):
    load_upstream(upstream)
    root = Path(upstream).resolve()
    return {name: dict(file=str(root / name), sha256=sha(root / name)) for name in (
        "OriginalREDQCodebase/redq/algos/core.py", "OriginalREDQCodebase/redq/algos/redq_sac.py")}


def effective_parameters(agent):
    """Read actual objects, separating current alpha from constructor alpha."""
    import torch
    values = {name: getattr(agent, name) for name in ("replay_size", "batch_size", "lr", "gamma", "polyak", "auto_alpha", "start_steps", "delay_update_steps", "utd_ratio", "num_Q", "num_min", "q_target_mode", "policy_update_delay")}
    values.update(hidden_sizes=list(agent.hidden_sizes), target_entropy=float(agent.target_entropy), alpha_current=float(agent.alpha),
        log_alpha_current=float(agent.log_alpha.detach().cpu().item()),
        dropout_per_q=[[float(m.p) for m in q.modules() if isinstance(m, torch.nn.Dropout)] for q in agent.q_net_list],
        layer_norm_per_q=[[list(m.normalized_shape) for m in q.modules() if isinstance(m, torch.nn.LayerNorm)] for q in agent.q_net_list],
        target_training=[bool(_raw(q).training) for q in agent.q_target_net_list])
    return values


def observation(value):
    array = np.asarray(value, dtype="<f4").reshape(-1)
    if array.shape != (27,) or not np.isfinite(array).all():
        raise ValueError("Exactly27 finite causal observation/state features required")
    return array.tolist(), hashlib.sha256(array.tobytes()).hexdigest()


class ObservedReplay:
    """Read after the real store; sampling is the original bound method."""
    def __init__(self, raw, path=None):
        self.raw, self.path, self.pending = raw, Path(path) if path else None, None
        self.writes, self.boundaries = 0, []
        self.digest = hashlib.sha256()

    def __getattr__(self, name):
        return getattr(self.raw, name)

    def store(self, obs, action, reward, next_obs, done):
        position = self.raw.ptr
        self.raw.store(obs, action, reward, next_obs, done)
        self.writes += 1
        for value in (self.raw.obs1_buf[position], self.raw.acts_buf[position], np.asarray([self.raw.rews_buf[position]], dtype="<f4"), self.raw.obs2_buf[position], np.asarray([self.raw.done_buf[position]], dtype="<f4")):
            self.digest.update(np.asarray(value, dtype="<f4").tobytes())
        if self.pending and (self.pending["terminated"] or self.pending["truncated"]):
            before, before_sha = observation(self.raw.obs1_buf[position]); stored, stored_sha = observation(self.raw.obs2_buf[position]); terminal, terminal_sha = observation(next_obs)
            record = dict(transition_number=self.writes, replay_position=position, env_index=0,
                done=float(bool(self.pending["terminated"] or self.pending["truncated"])), timeout=float(bool(self.pending["truncated"] and not self.pending["terminated"])),
                effective_done=float(bool(self.pending["terminated"])), stored_done=float(self.raw.done_buf[position]),
                time_limit_truncated=bool(self.pending["truncated"] and not self.pending["terminated"]),
                terminated=bool(self.pending["terminated"]), truncated=bool(self.pending["truncated"]), info_date=str(self.pending.get("date")),
                terminal_observation_values=terminal, terminal_observation_sha256=terminal_sha,
                stored_next_observation_values=stored, stored_next_observation_sha256=stored_sha,
                stored_observation_values=before, stored_observation_sha256=before_sha)
            self.boundaries.append(record)
            if self.path:
                with self.path.open("a", encoding="utf-8", newline="\n") as stream:
                    stream.write(json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n")
        self.pending = None


class ObservedTarget:
    """Count actual Polyak parameter iteration only while original train runs."""
    def __init__(self, raw):
        self.raw, self.active, self.update_rounds = raw, False, 0

    def __getattr__(self, name):
        return getattr(self.raw, name)

    def __call__(self, *args, **kwargs):
        return self.raw(*args, **kwargs)

    def parameters(self, *args, **kwargs):
        if self.active:
            self.update_rounds += 1
        return self.raw.parameters(*args, **kwargs)


def observe_agent(agent, trace_path=None):
    if isinstance(agent.replay_buffer, ObservedReplay) or any(isinstance(q, ObservedTarget) for q in agent.q_target_net_list):
        raise ValueError("Do not stack observers")
    agent.replay_buffer = ObservedReplay(agent.replay_buffer, trace_path)
    agent.q_target_net_list = [ObservedTarget(q) for q in agent.q_target_net_list]
    return agent


class MetricSink:
    def __init__(self):
        self.calls, self.last = 0, {}

    def store(self, **values):
        self.calls += 1; self.last = values


def original_train(agent, logger):
    targets = [q for q in agent.q_target_net_list if isinstance(q, ObservedTarget)]
    before = targets[0].update_rounds if targets else optimizer_steps(agent.q_optimizer_list[0])
    for q in targets:
        q.active = True
    try:
        agent.train(logger)
    finally:
        for q in targets:
            q.active = False
    after = targets[0].update_rounds if targets else optimizer_steps(agent.q_optimizer_list[0])
    if after > before:
        agent._observed_learning_calls = getattr(agent, "_observed_learning_calls", 0) + 1


def optimizer_steps(optimizer):
    import torch
    values = []
    for state in optimizer.state.values():
        step = state.get("step")
        if not isinstance(step, torch.Tensor) or step.numel() != 1:
            raise ValueError("Actual scalar Adam step tensors required")
        value = float(step.detach().cpu().item())
        if not np.isfinite(value) or value < 0 or value != int(value):
            raise ValueError("Finite integral actual Adam step required")
        values.append(int(value))
    if values and len(set(values)) != 1:
        raise ValueError("Partial parameter optimization cannot masquerade as complete updates")
    return values[0] if values else 0


def actual_counters(agent, *, environment_steps=None, random_actions=None):
    q = [optimizer_steps(opt) for opt in agent.q_optimizer_list]
    if len(q) != 2 or len(set(q)) != 1:
        raise ValueError("Both original Q optimizers must have equal real updates")
    result = dict(q_inner_rounds=q[0], q_optimizer_steps=q, total_q_optimizer_steps=sum(q), policy_optimizer_steps=optimizer_steps(agent.policy_optimizer),
        alpha_optimizer_steps=optimizer_steps(agent.alpha_optim), target_updates_per_network=[t.update_rounds if isinstance(t, ObservedTarget) else None for t in agent.q_target_net_list],
        learning_train_calls=getattr(agent, "_observed_learning_calls", 0))
    if environment_steps is not None:
        result["environment_steps"] = int(environment_steps)
    if isinstance(agent.replay_buffer, ObservedReplay):
        result.update(replay_adds=agent.replay_buffer.writes)
    if random_actions is not None:
        result["random_exploration_actions"] = int(random_actions)
    return result


def drive_steps(agent, env, steps, *, seed=0, progress_path=None):
    """Exactlysteps interactions; store terminal next_obs before any reset."""
    if type(steps) is not int or steps <= 0 or not isinstance(agent.replay_buffer, ObservedReplay):
        raise ValueError("Positive bounded steps and one observational replay wrapper required")
    obs, _ = env.reset(seed=seed); env.action_space.seed(seed)
    observation(obs); random_actions = 0; logger = MetricSink(); started = time.monotonic()
    action_rng_before = copy.deepcopy(env.action_space.np_random.bit_generator.state)
    for step in range(1, steps + 1):
        random_actions += int(agent.replay_buffer.size <= agent.start_steps)
        action = agent.get_exploration_action(obs, env)
        next_obs, reward, terminated, truncated, info = env.step(action)
        observation(next_obs)
        agent.replay_buffer.pending = dict(terminated=bool(terminated), truncated=bool(truncated), date=info.get("date"))
        agent.store_data(obs, action, reward, next_obs, bool(terminated))
        original_train(agent, logger)
        if progress_path and step % 2048 == 0:
            with Path(progress_path).open("a", encoding="utf-8", newline="\n") as stream:
                stream.write(json.dumps(dict(at=now(), elapsed_seconds=time.monotonic() - started, **actual_counters(agent, environment_steps=step, random_actions=random_actions)), allow_nan=False) + "\n")
        if terminated or truncated:
            if step < steps:
                obs, _ = env.reset()
        else:
            obs = next_obs
    counters = actual_counters(agent, environment_steps=steps, random_actions=random_actions)
    if counters["replay_adds"] != steps:
        raise ValueError("Each actual interaction must produce exactly one source replay store")
    return dict(counters=counters, elapsed_seconds=time.monotonic() - started, logger_calls=logger.calls,
        transition_sha256=agent.replay_buffer.digest.hexdigest(), training_episode_history=list(getattr(env, "training_episode_history", [])),
        training_window_quality=getattr(env, "training_window_metadata", {}),
        action_space_rng_before=action_rng_before, action_space_rng_after=copy.deepcopy(env.action_space.np_random.bit_generator.state))


def _raw(module):
    return module.raw if isinstance(module, ObservedTarget) else module


def modules_for(agent):
    return dict(policy=agent.policy_net, q0=agent.q_net_list[0], q1=agent.q_net_list[1], target_q0=_raw(agent.q_target_net_list[0]), target_q1=_raw(agent.q_target_net_list[1]))


def parameter_tensors(agent):
    result = {group + "." + key: value for group, module in modules_for(agent).items() for key, value in module.state_dict().items()}
    result["log_alpha.log_alpha"] = agent.log_alpha.detach()
    return result


def tensor_digest(values):
    digest = hashlib.sha256()
    for name, value in sorted(values.items()):
        tensor = value.detach().cpu().contiguous()
        header = json.dumps([name, str(tensor.dtype), list(tensor.shape)], separators=(",", ":")).encode()
        digest.update(len(header).to_bytes(8, "big")); digest.update(header); digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def parameter_digest(model):
    return tensor_digest(model.state_dict() if hasattr(model, "state_dict") else parameter_tensors(model))


def _module_metadata(module):
    import torch
    return dict({"class": type(module).__name__}, training=bool(module.training),
        dropout_p=[float(m.p) for m in module.modules() if isinstance(m, torch.nn.Dropout)],
        layer_norm_shapes=[list(m.normalized_shape) for m in module.modules() if isinstance(m, torch.nn.LayerNorm)],
        requires_grad_all=all(p.requires_grad for p in module.parameters()), parameter_count=sum(p.numel() for p in module.parameters()))


def save_checkpoint(directory, agent, *, counters, upstream=DEFAULT_UPSTREAM):
    """Only Tensor blobs plus a tagged JSON tree; never pickle an Agent."""
    import torch
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    if any((directory / name).exists() for name in ("model_tensors.pt", "model.json")):
        raise FileExistsError("Preserve every attempted checkpoint; never overwrite original Tensor evidence")
    blob, groups, trees = {}, {}, {}
    def pack(value, group):
        if isinstance(value, torch.Tensor):
            key = group + "/tensor" + str(len(groups[group]["tensor_keys"]))
            blob[key] = value.detach().cpu().clone(); groups[group]["tensor_keys"].append(key)
            return {"tensor": key}
        if isinstance(value, dict):
            return {"dict": [[pack(k, group), pack(v, group)] for k, v in value.items()]}
        if isinstance(value, tuple):
            return {"tuple": [pack(v, group) for v in value]}
        if isinstance(value, list):
            return {"list": [pack(v, group) for v in value]}
        if value is None or type(value) in (str, bool, int, float):
            if isinstance(value, float) and not np.isfinite(value):
                raise ValueError("Checkpoint metadata must be finite")
            return value
        raise TypeError("Non-Tensor/JSON checkpoint object: " + type(value).__name__)
    def begin(name):
        groups[name] = dict(tensor_keys=[], named_tensors={})
    module_meta = {}
    for group, module in modules_for(agent).items():
        begin(group)
        for name, value in module.state_dict().items():
            packed = pack(value, group); groups[group]["named_tensors"][name] = packed["tensor"]
        groups[group]["parameter_count"] = sum(v.numel() for v in module.state_dict().values())
        module_meta[group] = _module_metadata(module)
    begin("log_alpha"); groups["log_alpha"]["named_tensors"]["log_alpha"] = pack(agent.log_alpha, "log_alpha")["tensor"]
    groups["log_alpha"]["parameter_count"] = agent.log_alpha.numel()
    optimizers = dict(optimizer_policy=agent.policy_optimizer, optimizer_q0=agent.q_optimizer_list[0], optimizer_q1=agent.q_optimizer_list[1], optimizer_alpha=agent.alpha_optim)
    for group, optimizer in optimizers.items():
        begin(group); raw_state = optimizer.state_dict(); trees[group] = pack(raw_state, group)
        groups[group].update(step_tensor_keys=[], step_values=[])
        # Use JSON tree references, never equality matching of values, to map
        # every actual optimizer tensor and native per-parameter step tensor.
        def visit(node, trail=()):
            if isinstance(node, dict) and "tensor" in node:
                groups[group]["named_tensors"]["/".join(map(str, trail))] = node["tensor"]
                if trail and trail[-1] == "step":
                    groups[group]["step_tensor_keys"].append(node["tensor"])
                    groups[group]["step_values"].append(float(blob[node["tensor"]].item()))
            elif isinstance(node, dict) and "dict" in node:
                for key, value in node["dict"]:
                    visit(value, (*trail, key))
            elif isinstance(node, dict) and ("list" in node or "tuple" in node):
                for index, value in enumerate(node.get("list", node.get("tuple"))):
                    visit(value, (*trail, index))
        visit(trees[group])
        optimizer_steps(optimizer)
    begin("rng_torch"); groups["rng_torch"]["named_tensors"]["cpu"] = pack(torch.get_rng_state(), "rng_torch")["tensor"]
    if str(agent.device).startswith("cuda"):
        groups["rng_torch"]["named_tensors"]["cuda0"] = pack(torch.cuda.get_rng_state(agent.device), "rng_torch")["tensor"]
    np_state = np.random.get_state(); begin("rng_numpy")
    groups["rng_numpy"]["named_tensors"]["state"] = pack(torch.from_numpy(np_state[1].astype(np.int64)), "rng_numpy")["tensor"]
    begin("counters")
    for key, value in sorted(counters.items()):
        if isinstance(value, list):
            for index, item in enumerate(value):
                if type(item) is not int or item < 0:
                    raise ValueError("Nonnegative actual counter required")
            # Per-Q arrays remain JSON mirrors; actual per-parameter Adam step
            # tensors and target-observer evidence supply their independent pin.
        else:
            if type(value) is not int or value < 0:
                raise ValueError("Nonnegative actual counter required")
            groups["counters"]["named_tensors"][key] = pack(torch.tensor(value, dtype=torch.int64), "counters")["tensor"]
    tensor_path = directory / "model_tensors.pt"
    with tensor_path.open("xb") as stream:
        torch.save(blob, stream)
    manifest = dict(schema_version=1, upstream_commit=UPSTREAM_COMMIT, upstream_manifest_sha256=UPSTREAM_MANIFEST_SHA256,
        tensor_file=tensor_path.name, tensor_sha256=sha(tensor_path), parameter_sha256=parameter_digest(agent),
        tensor_groups=groups, tensors={k: dict(shape=list(v.shape), dtype=str(v.dtype)) for k, v in blob.items()}, trees=trees,
        modules=module_meta, counters=counters, effective_alpha=float(agent.alpha), log_alpha_value=float(agent.log_alpha.detach().cpu().item()),
        effective_target_entropy=int(agent.target_entropy), hidden_sizes=list(agent.hidden_sizes), action_limit=float(agent.act_limit),
        rng_numpy=dict(kind=np_state[0], position=int(np_state[2]), has_gauss=int(np_state[3]), cached_gaussian=float(np_state[4])),
        rng_python=pack(random.getstate(), "rng_numpy"), optimizer_classes={k: type(v).__name__ for k, v in optimizers.items()},
        safe_load_contract="torch.load(weights_only=True); dict[str,Tensor] only; no Agent/model object pickle")
    write_new(directory / "model.json", manifest)
    return manifest


def read_checkpoint(directory):
    """Independent-safe loader does not construct an Agent or sample RNG."""
    import torch
    directory = Path(directory); manifest = json.loads(relative(directory, "model.json").read_text(encoding="utf-8"))
    path = relative(directory, manifest["tensor_file"])
    if manifest.get("upstream_commit") != UPSTREAM_COMMIT or manifest.get("upstream_manifest_sha256") != UPSTREAM_MANIFEST_SHA256 or sha(path) != manifest.get("tensor_sha256"):
        raise ValueError("Original pure-Tensor checkpoint source or bytes changed")
    tensors = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(tensors, dict) or any(not isinstance(k, str) or not isinstance(v, torch.Tensor) for k, v in tensors.items()) or set(tensors) != set(manifest.get("tensors", {})):
        raise ValueError("Only declared Tensor dictionary checkpoints are accepted")
    for key, value in tensors.items():
        if manifest["tensors"][key] != dict(shape=list(value.shape), dtype=str(value.dtype)) or not torch.isfinite(value).all():
            raise ValueError("Checkpoint tensor shape/dtype/value differs: " + key)
    values = {}
    for group in MODEL_GROUPS:
        for name, key in manifest["tensor_groups"][group]["named_tensors"].items():
            values[group + "." + name] = tensors[key]
    if tensor_digest(values) != manifest.get("parameter_sha256"):
        raise ValueError("Actual pure-Tensor parameter digest differs")
    for name, key in manifest["tensor_groups"]["counters"]["named_tensors"].items():
        expected = manifest["counters"]
        for part in name.split("/"):
            expected = expected[int(part)] if isinstance(expected, list) else expected[part]
        if tensors[key].dtype != torch.int64 or tensors[key].numel() != 1 or int(tensors[key].item()) != expected:
            raise ValueError("Actual counter tensors and JSON disagree")
    for group in ("optimizer_policy", "optimizer_q0", "optimizer_q1", "optimizer_alpha"):
        meta = manifest["tensor_groups"][group]
        if meta["step_values"] != [float(tensors[k].item()) for k in meta["step_tensor_keys"]]:
            raise ValueError("Actual optimizer step Tensor/JSON evidence disagrees")
    return manifest, tensors


class FrozenPolicy:
    def __init__(self, actor, other_tensors):
        self.actor, self._other = actor, other_tensors
        self.policy = self

    def state_dict(self):
        return {**self._other, **{"policy." + k: v for k, v in self.actor.state_dict().items()}}

    def predict(self, obs, state=None, episode_start=None, deterministic=True):
        import torch
        if deterministic is not True:
            raise ValueError("Research replay requires deterministic original mean-action inference")
        values = np.asarray(obs, dtype=np.float32)
        if values.shape != (27,) or not np.isfinite(values).all():
            raise ValueError("Exactly27 finite actual observation/state values required")
        device = next(self.actor.parameters()).device
        with torch.no_grad():
            action = self.actor(torch.as_tensor(values, device=device).unsqueeze(0), deterministic=True, return_log_prob=False)[0]
        return action.detach().cpu().numpy().reshape(2), None


def load_policy(directory, *, upstream=DEFAULT_UPSTREAM, device="cpu"):
    import torch
    manifest, tensors = read_checkpoint(directory); _, core = load_upstream(upstream)
    cuda_devices = [torch.device(device).index or 0] if str(device).startswith("cuda") else []
    # Constructing an actor consumes initialization RNG, so isolate that work.
    with torch.random.fork_rng(devices=cuda_devices):
        actor = core.TanhGaussianPolicy(27, 2, tuple(manifest["hidden_sizes"]), action_limit=manifest["action_limit"]).to(device)
    actor.load_state_dict({name: tensors[key] for name, key in manifest["tensor_groups"]["policy"]["named_tensors"].items()}, strict=True)
    actor.eval()
    other = {(group + "." + name): tensors[key]
        for group in MODEL_GROUPS if group != "policy" for name, key in manifest["tensor_groups"][group]["named_tensors"].items()}
    model = FrozenPolicy(actor, other)
    if parameter_digest(model) != manifest["parameter_sha256"]:
        raise ValueError("Loaded original actor differs from the audited checkpoint")
    return model


def cropped_data(data, end):
    cutoff = pd.Timestamp(end)
    return {name: value.loc[value.index < cutoff].copy() if name.startswith("hourly_") else value for name, value in data.items()}


def evaluate(data, models, seeds, family, start, end, costs):
    from scripts.btceth_rl_env import BTCETHPortfolioEnv
    if family not in BUDGETS or len(models) != len(seeds) or len(models) not in (1, 5) or len(set(seeds)) != len(seeds) or not set(seeds).issubset(SEEDS) or len(models) == 5 and set(seeds) != set(SEEDS):
        raise ValueError("One fixed seed or all five original DroQ policies required")
    before = [parameter_digest(model.policy) for model in models]
    env = BTCETHPortfolioEnv(data, start, end, costs=costs, capital=10000., record_history=True, **PROFILE_PARAMETERS)
    obs, _ = env.reset(seed=0); rows = []; done = False
    while not done:
        date = str(env.current_date); values, observation_sha = observation(obs); targets, members = [], []
        for seed, model in zip(seeds, models):
            action, _ = model.predict(obs, deterministic=True); action = np.asarray(action, dtype=float).reshape(2)
            raw = env.action_to_weights(action); mapped = raw.copy(); mapped[:2] *= BUDGETS[family]; mapped[2] = 1 - mapped[:2].sum(); targets.append(mapped)
            members.append(dict(day=date, seed=seed, raw_action0=float(action[0]), raw_action1=float(action[1]),
                **dict(zip(("raw_btc", "raw_eth", "raw_cash"), raw)), **dict(zip(("mapped_btc", "mapped_eth", "mapped_cash"), mapped)),
                **dict(zip(("capped_btc", "capped_eth", "capped_cash"), mapped)), observation_sha256=observation_sha,
                observation_values_json=json.dumps(values, separators=(",", ":")), observation_asof=str(env.current_date - pd.Timedelta(hours=1)), observation_cutoff_exclusive=date,
                **dict(zip(("observation_btc_weight", "observation_eth_weight", "observation_cash_weight", "observation_drawdown"), values[-4:]))))
        target = np.mean(targets, axis=0)
        for member in members:
            member.update(dict(zip(("target_btc", "target_eth", "target_cash"), target))); rows.append(member)
        obs, _, terminated, truncated, _ = env.step_target(target); done = bool(terminated or truncated)
    after = [parameter_digest(model.policy) for model in models]
    if before != after:
        raise ValueError("Actual replay cannot mutate any trained DroQ model parameters")
    result = env.get_result(); env.close()
    result["metadata"].update(algorithm="droq", profile=PROFILE, family=family, model_count=len(models), exposure_multiplier=BUDGETS[family],
        training_budget_multiple=1, q_inner_rounds_per_training_model=555360,
        parameter_sha256_before=before, parameter_sha256_after=after, old_actions_or_equity_scaled=False, deterministic_prediction=True,
        ensemble="mean five actual targets in one real funded10,000 USDT account", **{flag: True for flag in FLAGS})
    return result, pd.DataFrame(rows)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def publish(path, value):
    """Only live process/progress records are replaceable, never sealed inputs."""
    path = Path(path); temporary = path.with_suffix(path.suffix + ".pending")
    with temporary.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False); stream.write("\n")
    os.replace(temporary, path)


def utc(value):
    value = pd.Timestamp(value)
    if value.tzinfo is None:
        raise ValueError("Explicit UTC evidence timestamps required")
    return value.tz_convert("UTC")


def pins(root, names):
    return {name: sha(relative(root, name)) for name in sorted(names)}


def verify_pins(root, values):
    if not isinstance(values, dict) or not values:
        raise ValueError("Nonempty exact input SHA map required")
    for name, expected in values.items():
        if sha(relative(root, name)) != expected:
            raise ValueError("Frozen research evidence changed: " + name)


def financial_imports():
    for name in shared.SOURCE_FILES:
        module = sys.modules.get(name[:-3].replace("/", "."))
        if module is not None and Path(module.__file__).resolve() != (ROOT / name).resolve():
            raise ValueError("Imported financial module is outside the pinned source root: " + name)


def separate(main, output):
    for value in (main, output):
        relative(value, ".")
    main, output = Path(main).resolve(), Path(output).resolve()
    if main == output or main.is_relative_to(output) or output.is_relative_to(main):
        raise ValueError("DroQ research requires a separate, non-nested tree")
    return main, output


def baseline_files():
    folder = "evaluation/test/baselines"
    return {*(f"{folder}/{name}" for name in GROUP_FILES),
        *(f"{folder}/{cost}/{key}/{name}" for cost in COSTS for key in shared.BASELINES for name in CASE_FILES[:6])}


def fit_authorization(value, sources):
    """No source/fixture preparation grants an actual market or GPU run."""
    if (value.get("fit_validation_authorized") is not True or value.get("no_live_trading") is not True
            or value.get("reviewed_source_sha256") != sources or value.get("hyperparameters") != recipe()
            or value.get("seeds") != list(SEEDS) or value.get("timesteps_per_model") != STEPS
            or value.get("budgets") != BUDGETS or any(value.get(flag) is not True for flag in FLAGS)):
        raise ValueError("Separate explicit authorization must pin the reviewed nine sources and exact finite DroQ recipe")
    utc(value["authorized_at"])
    return value


def prepare(main, dataset, output, *, upstream_root, config_contract, authorization, allowance_file=None):
    """Future SHA-only preparation. This never unpickles data or constructs NN."""
    main, output = separate(main, output); dataset = Path(dataset)
    relative(dataset.parent, dataset.name)
    if output.exists():
        raise FileExistsError("Preserve every attempted DroQ study; use a new output")
    financial_imports(); sources = pins(ROOT, SOURCE_FILES)
    approval = fit_authorization(read_json(authorization), sources)
    old, runtime = read_json(main / "frozen_spec.json"), read_json(main / "runtime.json")
    if (old.get("timesteps_per_model") != STEPS or old.get("research_targets") != TARGETS
            or tuple(old.get("train", ())) != TRAIN or tuple(old.get("validation", ())) != VALIDATION or tuple(old.get("test", ())) != TEST):
        raise ValueError("Original dates, environment budget and unlowered research goals required")
    verify_pins(ROOT, old["code_sha256"])
    if sha(dataset) != old["dataset_sha256"] or sha(dataset.parent / "manifest.json") != old["dataset_manifest_sha256"]:
        raise ValueError("Only the original exact BTC/ETH dataset and manifest are permitted")
    original_audit = read_json(main / "independent_audit.json")
    if (original_audit.get("passed") is not True or original_audit.get("test_complete") is not True
            or original_audit.get("status") != "passed" or original_audit.get("run_count") != 172 or original_audit.get("failures") != []):
        raise ValueError("The fourteen referenced controls need the complete original172 audit")
    verify_pins(main, original_audit["input_sha256"])
    manifest, author_files = verify_upstream(upstream_root)
    upstream_hashes = pins(upstream_root, {*author_files, "verified_manifest.json", "github_commit.json", "github_tree.json"})
    if sha(config_contract) != EFFECTIVE_CONFIG_SHA256:
        raise ValueError("Use the independently reviewed effective constructor contract")
    contract = read_json(config_contract)
    if contract.get("effective_constructor_kwargs") != recipe() or contract.get("official_commit") != UPSTREAM_COMMIT:
        raise ValueError("Effective constructor contract differs from the single reviewed recipe")
    gpus = runtime["physical_gpus"]
    if (len(gpus) != 2 or len({v["uuid"] for v in gpus}) != 2 or any("V100" not in v["name"] for v in gpus)
            or old.get("runtime_packages") != shared.RUNTIME_PACKAGES):
        raise ValueError("Original two physical V100 devices and unchanged pinned runtime required")
    spec = experiment_spec()
    spec.update(schema_version=1, prepared_at=now(), original_frozen_spec_sha256=sha(main / "frozen_spec.json"),
        dataset_sha256=sha(dataset), dataset_manifest_sha256=sha(dataset.parent / "manifest.json"),
        source_sha256=sources, original_input_sha256=pins(main, baseline_files() | {"frozen_spec.json", "runtime.json", "independent_audit.json"}),
        upstream_evidence_sha256=upstream_hashes, effective_config_contract_sha256=EFFECTIVE_CONFIG_SHA256,
        fit_validation_authorization_sha256=sha(authorization), physical_gpus=gpus,
        seed_gpu_assignments={str(seed): gpus[seed % 2]["uuid"] for seed in SEEDS},
        family_gpu_assignments={family: gpu["uuid"] for family, gpu in zip(BUDGETS, gpus)},
        runtime_packages=copy.deepcopy(shared.RUNTIME_PACKAGES), allowed_compute_instance=None, allowed_gpu_rows=[])
    if utc(approval["authorized_at"]) > utc(spec["prepared_at"]):
        raise ValueError("Authorization must actually precede preparation")
    if allowance_file is not None:
        allowance = read_json(allowance_file)
        spec.update(allowed_compute_instance=allowance["allowed_instance"], allowed_gpu_rows=allowance["allowed_gpu_rows"],
            resource_allowance_sha256=sha(allowance_file))
    # All checks above precede the first new-study write.
    output.mkdir(parents=True)
    for base, names, source in (("sources", sources, ROOT), ("upstream", upstream_hashes, upstream_root)):
        for name in names:
            target = relative(output, base + "/" + name); target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(relative(source, name), target)
    shutil.copyfile(config_contract, output / "effective_config_contract.json")
    (output / "preflight").mkdir()
    shutil.copyfile(authorization, output / "preflight/fit_validation_authorization.json")
    if allowance_file is not None:
        shutil.copyfile(allowance_file, output / "resource_allowance.json")
    write_new(output / "protocol.json", spec)
    write_new(output / "protocol_receipt.json", dict(protocol_file="protocol.json", protocol_sha256=sha(output / "protocol.json"), sealed_at=now()))
    return spec


def protocol(main, dataset, output):
    financial_imports(); main, output = separate(main, output)
    spec, receipt = read_json(output / "protocol.json"), read_json(output / "protocol_receipt.json")
    if any(spec.get(key) != expected for key, expected in experiment_spec().items()):
        raise ValueError("Exactly one frozen recipe, five seeds and two inference budgets required")
    if (receipt.get("protocol_file") != "protocol.json" or receipt.get("protocol_sha256") != sha(output / "protocol.json")
            or utc(receipt["sealed_at"]) < utc(spec["prepared_at"])
            or spec["original_frozen_spec_sha256"] != sha(main / "frozen_spec.json")
            or sha(dataset) != spec["dataset_sha256"] or sha(Path(dataset).parent / "manifest.json") != spec["dataset_manifest_sha256"]
            or set(spec["source_sha256"]) != set(SOURCE_FILES)):
        raise ValueError("Whole-file protocol, original source or dataset identity changed")
    verify_pins(main, spec["original_input_sha256"])
    verify_pins(ROOT, spec["source_sha256"]); verify_pins(output / "sources", spec["source_sha256"])
    verify_upstream(output / "upstream"); verify_pins(output / "upstream", spec["upstream_evidence_sha256"])
    if sha(output / "effective_config_contract.json") != spec["effective_config_contract_sha256"] or spec["effective_config_contract_sha256"] != EFFECTIVE_CONFIG_SHA256:
        raise ValueError("Reviewed effective constructor contract changed")
    auth_path = output / "preflight/fit_validation_authorization.json"
    if sha(auth_path) != spec["fit_validation_authorization_sha256"]:
        raise ValueError("Original fit/validation authorization changed")
    approval = fit_authorization(read_json(auth_path), spec["source_sha256"])
    if utc(approval["authorized_at"]) > utc(spec["prepared_at"]):
        raise ValueError("Fit/validation authorization must precede freezing")
    if spec.get("resource_allowance_sha256") is not None:
        allowance = read_json(output / "resource_allowance.json")
        if (sha(output / "resource_allowance.json") != spec["resource_allowance_sha256"]
                or allowance.get("allowed_instance") != spec["allowed_compute_instance"] or allowance.get("allowed_gpu_rows") != spec["allowed_gpu_rows"]):
            raise ValueError("Reviewed resource instance allowance changed")
    return spec


def job_id(seed):
    if type(seed) is not int or seed not in SEEDS:
        raise ValueError("One of all five fixed integer seeds required")
    return f"droq__balanced__seed{seed}"


def train_one(task):
    from scripts.btceth_rl_env import BTCETHPortfolioEnv
    main, dataset, output = map(Path, (task["main"], task["dataset"], task["output"]))
    spec = protocol(main, dataset, output); seed = task["seed"]
    if task.get("profile") != PROFILE:
        raise ValueError("Only the frozen balanced reward is trainable")
    directory = output / "training" / job_id(seed); directory.mkdir(parents=True, exist_ok=False)
    device = shared.torch_setup(seed)
    if device["visible_gpu_uuid"] != spec["seed_gpu_assignments"][str(seed)] or device["capability"] != [7, 0] or "V100" not in device["name"]:
        raise ValueError("Actual isolated assigned physical V100 required; no CPU fallback")
    write_new(directory / "started.json", dict(at=now(), pid=os.getpid(), task=task, device=device, protocol_sha256=sha(output / "protocol.json")))
    raw = None; start_time = time.monotonic()
    try:
        # Cropping before feature construction prevents validation prices/features
        # from even entering a training environment.
        data = cropped_data(shared.load_dataset(dataset), TRAIN[1])
        raw = BTCETHPortfolioEnv(data, *TRAIN, training=True, episode_days=EPISODE_DAYS,
            costs=COSTS["base"], capital=10000., record_history=False, **PROFILE_PARAMETERS)
        agent = observe_agent(make_agent(output / "upstream", device="cuda:0"), directory / "training_timeout_transitions.jsonl")
        if agent.alpha != .2 or float(agent.log_alpha.detach().cpu().item()) != 0.:
            raise ValueError("Preserve the real author's initial alpha overwrite before the first update")
        driven = drive_steps(agent, raw, STEPS, seed=seed, progress_path=directory / "training_metrics.jsonl")
        counters = driven["counters"]
        if counters != experiment_spec()["expected_counters"]:
            raise ValueError("Actual replay, Q/actor/alpha optimizer or target-update budget differs")
        manifest = save_checkpoint(directory, agent, counters=counters, upstream=output / "upstream")
        trace = directory / "training_timeout_transitions.jsonl"
        if len(trace.read_text(encoding="utf-8").splitlines()) != 1024:
            raise ValueError("Exactly1024 real32-day Gymnasium truncations must be recorded")
        record = dict(algorithm="droq", profile=PROFILE, seed=seed, completed_at=now(), device=device,
            elapsed_seconds=time.monotonic() - start_time, actual_timesteps=STEPS, actual_counters=counters,
            algorithm_class=type(agent).__module__ + "." + type(agent).__qualname__,
            hyperparameters=recipe(), effective_hyperparameters=effective_parameters(agent),
            upstream_loaded_sources=loaded_sources(output / "upstream"),
            model_tensor_file=(directory / "model_tensors.pt").relative_to(output).as_posix(), model_tensor_sha256=sha(directory / "model_tensors.pt"),
            model_manifest_file=(directory / "model.json").relative_to(output).as_posix(), model_manifest_sha256=sha(directory / "model.json"),
            timeout_trace_relative_path=trace.relative_to(output).as_posix(), timeout_trace_sha256=sha(trace),
            parameter_sha256=manifest["parameter_sha256"], protocol_sha256=sha(output / "protocol.json"),
            features=list(raw.FEATURE_NAMES), training_episode_history=list(raw.training_episode_history),
            training_window_quality=raw.training_window_metadata, training_budget_multiple=1,
            environment_step_budget_multiple=1, compute_budget_comparable=False,
            action_space_rng_before=driven["action_space_rng_before"], action_space_rng_after=driven["action_space_rng_after"],
            actual_transition_sha256=driven["transition_sha256"],
            recipe_is_pure_official_default=False, bounded_driver=True, official_main_entrypoint_used=False,
            replay_semantics="actual terminal next_obs stored before reset; done=terminated; arrays observed only after original store",
            **{flag: True for flag in FLAGS})
        write_new(directory / "record.json", record)
        print("DROQ_TRAIN_COMPLETE", seed, json.dumps(counters), flush=True)
        return record
    except BaseException as error:
        write_new(directory / "failure.json", dict(at=now(), type=type(error).__name__, error=str(error), pid=os.getpid()))
        raise
    finally:
        if raw is not None:
            raw.close()


def training_files(output):
    output = Path(output); complete = read_json(output / "training_completion.json")
    if complete.get("model_count") != 5 or complete.get("protocol_sha256") != sha(output / "protocol.json"):
        raise ValueError("All five new fixed DroQ models must complete before validation")
    records = complete.get("records", [])
    if len(records) != 5 or {v.get("seed") for v in records} != set(SEEDS):
        raise ValueError("No missing, duplicated or selected training seeds")
    files = {"training_completion.json"}
    for seed in SEEDS:
        folder = "training/" + job_id(seed); record = read_json(output / folder / "record.json")
        if (record not in records or record.get("algorithm") != "droq" or record.get("profile") != PROFILE
                or record.get("seed") != seed or record.get("actual_timesteps") != STEPS
                or record.get("actual_counters") != experiment_spec()["expected_counters"]
                or record.get("hyperparameters") != recipe() or record.get("protocol_sha256") != sha(output / "protocol.json")):
            raise ValueError("Exact trained seed, finite recipe or observed optimizer budget changed")
        for field, digest_field, leaf in (("model_tensor_file", "model_tensor_sha256", "model_tensors.pt"),
                ("model_manifest_file", "model_manifest_sha256", "model.json"), ("timeout_trace_relative_path", "timeout_trace_sha256", "training_timeout_transitions.jsonl")):
            name = folder + "/" + leaf
            if record.get(field) != name or sha(relative(output, name)) != record.get(digest_field):
                raise ValueError("Exact trained checkpoint/manifest/timeout trace changed")
            files.add(name)
        files.add(folder + "/record.json")
    return pins(output, files)


def rng_digest():
    """Observe current RNG states; no sampling, seed changes or CUDA creation."""
    import torch
    digest = hashlib.sha256(); state = np.random.get_state()
    digest.update(state[0].encode()); digest.update(state[1].tobytes()); digest.update(repr(state[2:]).encode())
    digest.update(repr(random.getstate()).encode()); digest.update(torch.get_rng_state().numpy().tobytes())
    if torch.cuda.is_initialized():
        for value in torch.cuda.get_rng_state_all():
            digest.update(value.cpu().numpy().tobytes())
    return digest.hexdigest()


def evaluate_worker(task):
    main, dataset, output = map(Path, (task["main"], task["dataset"], task["output"]))
    spec = protocol(main, dataset, output); family, split = task["family"], task["split"]
    if family not in BUDGETS or split not in ("validation", "test"):
        raise ValueError("Only declared raw/cap budget and fixed research splits required")
    device = shared.torch_setup(0)
    if device["visible_gpu_uuid"] != spec["family_gpu_assignments"][family] or device["capability"] != [7, 0]:
        raise ValueError("Actual isolated assigned family V100 required")
    start, end = VALIDATION if split == "validation" else TEST
    data = cropped_data(shared.load_dataset(dataset), end); models, records = [], []
    for seed in SEEDS:
        directory = output / "training" / job_id(seed); record = read_json(directory / "record.json")
        model = load_policy(directory, upstream=output / "upstream", device="cuda:0")
        if parameter_digest(model) != record["parameter_sha256"]:
            raise ValueError("Loaded original actor and all five-network tensors differ from the fixed training record")
        models.append(model); records.append(record)
    directory = output / split / family / PROFILE
    directory.mkdir(parents=True, exist_ok=False); rows, histories, months_all, proofs = [], [], [], []
    for scenario, costs in COSTS.items():
        plans = [(str(seed), [model], [seed], seed) for seed, model in zip(SEEDS, models)] + [("ensemble", models, list(SEEDS), None)]
        for key, members, seeds, seed in plans:
            run_id = f"{split}__{scenario}__{family}__droq__{PROFILE}__{key}"
            rng_before = rng_digest()
            result, trace = evaluate(data, members, seeds, family, start, end, costs)
            rng_first = rng_digest()
            repeat, repeated_trace = evaluate(data, members, seeds, family, start, end, costs)
            rng_after = rng_digest()
            fingerprints, repeated = result_fingerprints(result), result_fingerprints(repeat)
            trace_bytes, repeated_bytes = canonical_csv(trace), canonical_csv(repeated_trace)
            if fingerprints != repeated or trace_bytes != repeated_bytes or rng_before != rng_first or rng_before != rng_after:
                raise AssertionError("Exact mean-action replay must preserve parameters, accounts, traces and all inference RNG states")
            case = directory / scenario / key
            if case.exists():
                raise FileExistsError("Never overwrite a partial or completed account")
            save_result(case, result); (case / "member_targets.csv").write_bytes(trace_bytes)
            proof = dict(run_id=run_id, passed=True, repeat_exact=True, fingerprints=fingerprints, repeated_fingerprints=repeated,
                member_targets_sha256=hashlib.sha256(trace_bytes).hexdigest(), repeated_member_targets_sha256=hashlib.sha256(repeated_bytes).hexdigest(),
                parameter_sha256_before=result["metadata"]["parameter_sha256_before"], parameter_sha256_after=result["metadata"]["parameter_sha256_after"],
                rng_sha256_before=rng_before, rng_sha256_after_first=rng_first, rng_sha256_after_repeat=rng_after,
                model_records=[record for record in records if record["seed"] in seeds], device=device, worker_pid=os.getpid())
            write_new(case / "replay.json", proof); proofs.append(proof)
            row, history, months = summarize(result, run_id=run_id, strategy="droq_" + family, seed=seed, scenario=scenario,
                split=split, start=start, end=end, capital=10000., **costs)
            row.update(algorithm="droq", profile=PROFILE, family=family, primary_policy=seed is None, exposure_multiplier=BUDGETS[family], rebalance_band=.025)
            history["profile"], months["profile"] = PROFILE, PROFILE
            rows.append(row); histories.append(history); months_all.append(months)
            print("DROQ_EVAL", run_id, row["net_return"], row["max_drawdown"], flush=True)
    for name, frame in zip(GROUP_FILES[:3], (pd.DataFrame(rows), pd.concat(histories, ignore_index=True), pd.concat(months_all, ignore_index=True))):
        (directory / name).write_bytes(canonical_csv(frame))
    write_new(directory / "completion.json", dict(at=now(), protocol_sha256=sha(output / "protocol.json"), case_count=12,
        repeat_count=12, repeats=proofs, worker_pid=os.getpid(), device=device))
    return dict(profile=PROFILE, split=split, family=family, device=device, worker_pid=os.getpid())


def process_identity(pid):
    import pwd
    proc = Path("/proc") / str(int(pid))
    uid = int(next(line for line in (proc / "status").read_text().splitlines() if line.startswith("Uid:")).split()[1])
    return dict(pid=int(pid), uid=uid, username=pwd.getpwuid(uid).pw_name, exe=os.readlink(proc / "exe"),
        starttime_ticks=(proc / "stat").read_text().rsplit(")", 1)[1].split()[19],
        boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip())


def resource_rows(stdout):
    rows = []
    for line in stdout.splitlines():
        if line.strip():
            uuid, pid = [value.strip() for value in line.split(",")]
            rows.append(dict(gpu_uuid=uuid, pid=int(pid)))
    return rows


def verify_resource_rows(rows, spec, inspect_process=None):
    if not rows:
        return
    inspect_process = inspect_process or process_identity
    allowed = spec.get("allowed_compute_instance")
    if allowed is None or rows != spec.get("allowed_gpu_rows") or inspect_process(allowed["pid"]) != allowed:
        raise RuntimeError("Unknown existing GPU process; never suppress or stop another task")


def gpu_preflight(spec):
    gpus = shared.gpu_inventory()
    identity_keys = ("index", "uuid", "name", "memory_total_mib")
    if [{k: gpu[k] for k in identity_keys} for gpu in gpus] != [{k: gpu[k] for k in identity_keys} for gpu in spec["physical_gpus"]]:
        raise ValueError("The two original physical V100 devices changed")
    command = ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"]
    stdout = subprocess.check_output(command, text=True)
    verify_resource_rows(resource_rows(stdout), spec)
    if shared.runtime_packages() != spec["runtime_packages"]:
        raise ValueError("No runtime version fallback or package installation is allowed")
    return dict(at=now(), physical_gpus=gpus, command=command, unfiltered_stdout=stdout,
        allowed_compute_instance=spec.get("allowed_compute_instance"), allowed_gpu_rows=spec.get("allowed_gpu_rows", []))


def terminal(output, phase):
    output = Path(output); value = read_json(output / (phase + "_process.json"))
    if (value.get("phase") != phase or value.get("status") != "completed" or type(value.get("exit_code")) is not int or value["exit_code"] != 0
            or any(type(value.get(key)) is not int or value[key] <= 0 for key in ("supervisor_pid", "child_pid"))
            or value["supervisor_pid"] == value["child_pid"] or value.get("source_sha256") != sha(Path(__file__))
            or value.get("protocol_sha256") != sha(output / "protocol.json") or value.get("log_relative_path") != phase + ".log"
            or value.get("log_sha256") != sha(relative(output, phase + ".log"))):
        raise ValueError("Actual matching completed0 process, child and exact original log required: " + phase)
    return value


def no_test(output):
    if any((Path(output) / name).exists() for name in ("test", "test_started.json", "test_completion.json", "test_process.json", "annual_authorization.json")):
        raise ValueError("Validation/selection cannot read or overwrite any annual attempt")


def supervise(main, dataset, output, phase, *, authorization=None):
    output = Path(output); spec = protocol(main, dataset, output)
    if phase not in ("fit", "validate", "test"):
        raise ValueError("Only separately authorized fit, validate or annual test supported")
    if phase in ("fit", "validate"):
        no_test(output)
    if phase == "validate":
        terminal(output, "fit"); training_files(output)
        if (output / "selection.json").exists():
            raise ValueError("Validation cannot be restarted after a selected candidate")
    if phase == "test":
        annual_gate(main, dataset, output, authorization, require_new=True)
    for name in (phase + "_process.json", phase + ".log", phase + "_completion.json", "preflight/" + phase + ".json",
            "training" if phase == "fit" else "validation" if phase == "validate" else "test"):
        if relative(output, name).exists():
            raise FileExistsError("Preserve existing phase/process/log/partial accounts; never silently restart: " + name)
    resource = gpu_preflight(spec)
    write_new(output / "preflight" / (phase + ".json"), resource)
    started_at = now()
    if phase == "test":
        # The marker is a dispatch declaration and precedes process.started_at.
        shutil.copyfile(authorization, output / "annual_authorization.json")
        write_new(output / "test_started.json", dict(at=now(), protocol_sha256=sha(output / "protocol.json"),
            selection_sha256=sha(output / "selection.json"), authorization_sha256=sha(output / "annual_authorization.json")))
        started_at = now()
    command = [sys.executable, "-u", str(Path(__file__).resolve()), "--phase", "_" + phase,
        "--reference-main", str(Path(main).resolve()), "--dataset", str(Path(dataset).resolve()), "--output-dir", str(output.resolve())]
    record = dict(phase=phase, status="launching", started_at=started_at, supervisor_pid=os.getpid(), command=command,
        protocol_sha256=sha(output / "protocol.json"), source_sha256=sha(Path(__file__)), log_relative_path=phase + ".log")
    proof, log = output / (phase + "_process.json"), output / (phase + ".log")
    write_new(proof, record)
    with log.open("x", encoding="utf-8") as stream:
        child = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
        record.update(status="running", child_pid=child.pid); publish(proof, record)
        code = child.wait()
    record.update(status="completed" if code == 0 else "failed", exit_code=code, completed_at=now(), log_sha256=sha(log)); publish(proof, record)
    if code:
        raise RuntimeError("Phase failed; preserve its original terminal/log/partial evidence without restart")
    return record


def private_gate(output, phase):
    # Popen can start before its parent has atomically published child_pid.
    # Wait only for that exact declaration; perform no GPU/data work meanwhile.
    for _ in range(30):
        proof = read_json(Path(output) / (phase + "_process.json"))
        if proof.get("status") != "launching":
            break
        time.sleep(.1)
    if (proof.get("phase") != phase or proof.get("status") != "running" or proof.get("child_pid") != os.getpid()
            or proof.get("supervisor_pid") != os.getppid() or proof.get("source_sha256") != sha(Path(__file__))
            or proof.get("protocol_sha256") != sha(Path(output) / "protocol.json")):
        raise ValueError("Private child requires its actual matching source-bound supervisor and PPID")


def fit_internal(main, dataset, output):
    output = Path(output); private_gate(output, "fit"); spec = protocol(main, dataset, output); no_test(output)
    pools = {gpu["uuid"]: ProcessPoolExecutor(max_workers=1, mp_context=mp.get_context("spawn"), initializer=shared.gpu_initializer,
        initargs=(gpu["uuid"],)) for gpu in spec["physical_gpus"]}
    records = []
    try:
        futures = [pools[spec["seed_gpu_assignments"][str(seed)]].submit(train_one,
            dict(main=str(main), dataset=str(dataset), output=str(output), profile=PROFILE, seed=seed)) for seed in SEEDS]
        for future in as_completed(futures):
            records.append(future.result())
            publish(output / "training_status.json", dict(at=now(), completed=len(records), total=5))
    finally:
        for pool in pools.values():
            pool.shutdown(wait=True, cancel_futures=True)
    if len(records) != 5 or {v["seed"] for v in records} != set(SEEDS):
        raise ValueError("All five fixed seed policies must complete")
    records.sort(key=lambda value: value["seed"])
    write_new(output / "training_completion.json", dict(at=now(), model_count=5, records=records, protocol_sha256=sha(output / "protocol.json")))
    write_new(output / "fit_completion.json", dict(at=now(), model_count=5, protocol_sha256=sha(output / "protocol.json")))


def validate_internal(main, dataset, output):
    output = Path(output); private_gate(output, "validate"); spec = protocol(main, dataset, output)
    no_test(output); terminal(output, "fit"); training_files(output)
    if (output / "selection.json").exists():
        raise ValueError("Both candidate budgets must finish before any selection")
    pools = {family: ProcessPoolExecutor(max_workers=1, mp_context=mp.get_context("spawn"), initializer=shared.gpu_initializer,
        initargs=(spec["family_gpu_assignments"][family],)) for family in BUDGETS}
    try:
        workers = [pool.submit(evaluate_worker, dict(main=str(main), dataset=str(dataset), output=str(output), split="validation", family=family))
            for family, pool in pools.items()]
        workers = [future.result() for future in workers]
    finally:
        for pool in pools.values():
            pool.shutdown(wait=True, cancel_futures=True)
    for name in GROUP_FILES[:3]:
        frames = [pd.read_csv(output / "validation" / family / PROFILE / name, float_precision="round_trip") for family in BUDGETS]
        (output / "validation" / name).write_bytes(canonical_csv(pd.concat(frames, ignore_index=True)))
    training_files(output)
    write_new(output / "validation_completion.json", dict(at=now(), protocol_sha256=sha(output / "protocol.json"), case_count=24, repeat_count=24, workers=workers))
    write_new(output / "validate_completion.json", dict(at=now(), protocol_sha256=sha(output / "protocol.json"), case_count=24, repeat_count=24))


def audit_inputs(main, output, audit):
    verify_pins(output, audit.get("input_sha256")); verify_pins(main, audit.get("main_input_sha256")); verify_pins(ROOT, audit.get("source_sha256"))


def validation_files():
    return {"validation_completion.json", "validate_completion.json", "validate_process.json", "validate.log",
        *("validation/" + name for name in GROUP_FILES[:3]),
        *(f"validation/{family}/{PROFILE}/{name}" for family in BUDGETS for name in GROUP_FILES),
        *(f"validation/{family}/{PROFILE}/{cost}/{key}/{name}" for family in BUDGETS for cost in COSTS for key in (*map(str, SEEDS), "ensemble") for name in CASE_FILES)}


def validation_audit(main, output):
    value = read_json(Path(output) / "independent_audit.validation.json")
    if (value.get("passed") is not True or value.get("status") != "partial_test_pending" or value.get("test_complete") is not False
            or value.get("training_model_count") != 5 or value.get("validation_run_count") != 24 or value.get("run_count") != 24
            or value.get("repeat_count") != 24 or value.get("failure_count") != 0 or value.get("failures") != []):
        raise ValueError("Complete independent24-account validation audit required; partial accounts never permit selection")
    audit_inputs(main, output, value)
    required = validation_files() | set(training_files(output)) | {"protocol.json", "protocol_receipt.json"}
    if not required.issubset(value["input_sha256"]) or set(value["source_sha256"]) != set(SOURCE_FILES):
        raise ValueError("Independent validation audit must bind all case, repeat, training and source inputs")
    return value


def candidate(frame, family):
    if family not in BUDGETS or len(frame) != 12 or frame.run_id.duplicated().any():
        raise ValueError("Two costs, all five seeds and one actual ensemble required")
    rows = {}
    for cost in COSTS:
        accounts = frame.loc[frame.scenario.eq(cost)]
        primary = accounts.loc[accounts.primary_policy]
        members = accounts.loc[~accounts.primary_policy]
        if len(primary) != 1 or len(members) != 5 or set(members.seed) != set(SEEDS) or primary.seed.notna().any():
            raise ValueError("No omitted cost or seed selection")
        row = primary.iloc[0]
        if row.run_id != f"validation__{cost}__{family}__droq__balanced__ensemble":
            raise ValueError("Actual predeclared ensemble validation identity required")
        if (not np.isfinite([row.net_return, row.max_drawdown, row.final_equity, row.fees]).all()
                or row.net_return <= -1 or row.max_drawdown > 0 or row.max_drawdown <= -1 or row.final_equity <= 0 or row.fees < 0):
            raise ValueError("Finite genuine validation return, drawdown and fees required")
        rows[cost] = {key: (str(row[key]) if key == "run_id" else float(row[key])) for key in ("run_id", "net_return", "max_drawdown", "final_equity", "fees")}
    score = sum(math.log1p(row["net_return"]) - .5 * abs(row["max_drawdown"]) - 2 * max(abs(row["max_drawdown"]) - .20, 0.) for row in rows.values()) / 2
    return dict(algorithm="droq", family=family, profile=PROFILE, rebalance_band=.025, exposure_multiplier=BUDGETS[family], score=score, **rows)


def select(main, dataset, output):
    output = Path(output); protocol(main, dataset, output); no_test(output)
    terminal(output, "fit"); terminal(output, "validate"); training = training_files(output); validation_audit(main, output)
    if (output / "selection.json").exists() or (output / "selection_receipt.json").exists():
        raise FileExistsError("Preserve the exact sealed validation choice")
    candidates = [candidate(pd.read_csv(output / "validation" / family / PROFILE / "summary.csv", float_precision="round_trip"), family) for family in BUDGETS]
    candidates.sort(key=lambda v: (-v["score"], 0 if v["family"] == "raw" else 1, v["profile"]))
    value = dict(selected_at=now(), protocol_sha256=sha(output / "protocol.json"), audit_sha256=sha(output / "independent_audit.validation.json"),
        candidates=candidates, candidate=candidates[0], seed_selection=False, selection_uses_test=False, test_results_available_to_selector=False,
        training_source_sha256=training, validation_source_sha256=pins(output, validation_files()), **{flag: True for flag in FLAGS})
    write_new(output / "selection.json", value)
    write_new(output / "selection_receipt.json", dict(sealed_at=now(), selection_file="selection.json", selection_sha256=sha(output / "selection.json"), protocol_sha256=sha(output / "protocol.json")))
    return value


def frozen_selection(main, dataset, output):
    output = Path(output); spec = protocol(main, dataset, output); validation_audit(main, output)
    value, receipt = read_json(output / "selection.json"), read_json(output / "selection_receipt.json")
    if (value.get("seed_selection") is not False or value.get("selection_uses_test") is not False
            or value.get("test_results_available_to_selector") is not False or any(value.get(flag) is not True for flag in FLAGS)
            or value.get("audit_sha256") != sha(output / "independent_audit.validation.json")
            or receipt.get("selection_file") != "selection.json" or receipt.get("selection_sha256") != sha(output / "selection.json")
            or value.get("protocol_sha256") != receipt.get("protocol_sha256") or value["protocol_sha256"] != sha(output / "protocol.json")
            or utc(value["selected_at"]) > utc(receipt["sealed_at"])):
        raise ValueError("Original whole-file validation seal and known-year identity required")
    verify_pins(output, value["training_source_sha256"]); verify_pins(output, value["validation_source_sha256"])
    if set(value["training_source_sha256"]) != set(training_files(output)) or set(value["validation_source_sha256"]) != validation_files():
        raise ValueError("All training and both budget validation sources must remain bound")
    actual = [candidate(pd.read_csv(output / "validation" / family / PROFILE / "summary.csv", float_precision="round_trip"), family) for family in BUDGETS]
    actual.sort(key=lambda v: (-v["score"], 0 if v["family"] == "raw" else 1, v["profile"]))
    stored = value.get("candidates", [])
    if len(stored) != 2 or [(v["family"], v["profile"]) for v in stored] != [(v["family"], v["profile"]) for v in actual] or value.get("candidate") != stored[0]:
        raise ValueError("Exactly the two fixed budget candidates and original raw-first ranking required")
    for saved, truth in zip(stored, actual):
        for key in ("algorithm", "family", "profile", "rebalance_band", "exposure_multiplier"):
            if saved.get(key) != truth[key]:
                raise ValueError("Selected algorithm, band or execution budget changed")
        if not math.isclose(saved["score"], truth["score"], rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("Sealed score no longer matches immutable validation accounts")
        for cost in COSTS:
            if saved[cost]["run_id"] != truth[cost]["run_id"] or any(not math.isclose(saved[cost][k], truth[cost][k], rel_tol=1e-10, abs_tol=1e-10) for k in ("net_return", "max_drawdown", "final_equity", "fees")):
                raise ValueError("Actual sealed validation financial metrics changed")
    return spec, value, receipt


def annual_gate(main, dataset, output, authorization, *, require_new):
    output = Path(output); spec, selection, receipt = frozen_selection(main, dataset, output)
    terminal(output, "fit"); terminal(output, "validate")
    if require_new:
        no_test(output)
    if authorization is None:
        raise ValueError("Separate explicit annual authorization is required after the frozen validation choice")
    auth = read_json(authorization)
    if (auth.get("annual_evaluation_authorized") is not True or auth.get("no_live_trading") is not True
            or auth.get("financial_parameters_fixed") is not True or auth.get("selection_sha256") != sha(output / "selection.json")
            or any(auth.get(flag) is not True for flag in FLAGS) or utc(auth["authorized_at"]) < utc(receipt["sealed_at"])):
        raise ValueError("Annual authorization must identify this actual newly sealed DroQ candidate and disclose the already-observed year")
    if not require_new:
        marker = read_json(output / "test_started.json")
        if (marker.get("protocol_sha256") != sha(output / "protocol.json") or marker.get("selection_sha256") != sha(output / "selection.json")
                or marker.get("authorization_sha256") != sha(authorization) or utc(marker["at"]) < utc(auth["authorized_at"])):
            raise ValueError("Actual annual marker must follow and bind its independent authorization")
    return spec, selection


def test_internal(main, dataset, output):
    output = Path(output); private_gate(output, "test")
    spec, selection = annual_gate(main, dataset, output, output / "annual_authorization.json", require_new=False)
    chosen = selection["candidate"]
    with ProcessPoolExecutor(max_workers=1, mp_context=mp.get_context("spawn"), initializer=shared.gpu_initializer,
            initargs=(spec["family_gpu_assignments"][chosen["family"]],)) as pool:
        worker = pool.submit(evaluate_worker, dict(main=str(main), dataset=str(dataset), output=str(output), split="test", family=chosen["family"])).result()
    training_files(output)
    write_new(output / "test_completion.json", dict(at=now(), case_count=12, repeat_count=12, candidate=chosen,
        protocol_sha256=sha(output / "protocol.json"), selection_sha256=sha(output / "selection.json"), worker=worker))


def verify(main, dataset, output):
    output = Path(output); _, selection = annual_gate(main, dataset, output, output / "annual_authorization.json", require_new=False)
    terminal(output, "test"); audit = read_json(output / "independent_audit.json")
    if (audit.get("passed") is not True or audit.get("status") != "passed" or audit.get("test_complete") is not True
            or audit.get("training_model_count") != 5 or audit.get("validation_run_count") != 24 or audit.get("test_run_count") != 12
            or audit.get("baseline_run_count") != 14 or audit.get("run_count") != 26 or audit.get("repeat_count") != 12
            or audit.get("failure_count") != 0 or audit.get("failures") != []):
        raise ValueError("Complete independent new12+referenced14 audit required; accounting passage differs from financial goal success")
    audit_inputs(main, output, audit)
    targets = audit.get("research_targets", {})
    if (targets.get("candidate") != {k: selection["candidate"][k] for k in ("algorithm", "family", "profile")}
            or targets.get("fresh_holdout_success") is not False or targets.get("training_budget_multiple") != 1
            or any(targets.get(flag) is not True for flag in FLAGS)):
        raise ValueError("Goal evidence must describe the new DroQ candidate and retain known-year/computation disclosure")
    value = dict(at=now(), protocol_sha256=sha(output / "protocol.json"), selection_sha256=sha(output / "selection.json"),
        audit_sha256=sha(output / "independent_audit.json"), accounting_audit_passed=True, **targets)
    write_new(output / "goal_candidate_evidence.json", value)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--describe-source", action="store_true")
    parser.add_argument("--phase", choices=("prepare", "fit", "_fit", "validate", "_validate", "select", "test", "_test", "verify"))
    for name in ("reference-main", "dataset", "output-dir", "upstream-root", "config-contract", "authorization", "allowance"):
        parser.add_argument("--" + name, type=Path)
    args = parser.parse_args()
    if args.describe_source:
        if args.phase:
            parser.error("Source description cannot dispatch a research phase")
        print(json.dumps(dict(source_preparation_only=True, specification=experiment_spec()), ensure_ascii=False, indent=2)); return
    if not args.phase or any(getattr(args, name) is None for name in ("reference_main", "dataset", "output_dir")):
        parser.error("An explicit phase, reference-main, dataset and independent output-dir are required")
    if args.phase == "prepare":
        if args.upstream_root is None or args.config_contract is None or args.authorization is None:
            parser.error("Preparation requires exact upstream/config-contract and a separate source-bound fit/validation authorization")
        value = prepare(args.reference_main, args.dataset, args.output_dir, upstream_root=args.upstream_root,
            config_contract=args.config_contract, authorization=args.authorization, allowance_file=args.allowance)
        print(json.dumps({"prepared_at": value["prepared_at"], "protocol_sha256": sha(args.output_dir / "protocol.json")}))
    elif args.phase in ("fit", "validate", "test"):
        print(json.dumps(supervise(args.reference_main, args.dataset, args.output_dir, args.phase, authorization=args.authorization)))
    elif args.phase == "_fit":
        fit_internal(args.reference_main, args.dataset, args.output_dir)
    elif args.phase == "_validate":
        validate_internal(args.reference_main, args.dataset, args.output_dir)
    elif args.phase == "_test":
        test_internal(args.reference_main, args.dataset, args.output_dir)
    elif args.phase == "select":
        print(json.dumps(select(args.reference_main, args.dataset, args.output_dir)["candidate"]))
    elif args.phase == "verify":
        print(json.dumps(verify(args.reference_main, args.dataset, args.output_dir)))


if __name__ == "__main__":
    main()

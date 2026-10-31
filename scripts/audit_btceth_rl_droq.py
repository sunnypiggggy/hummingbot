"""Independent DroQ research evidence checks; never constructs or trains an agent.

The author's OriginalREDQCodebase is a separate implementation from SB3.
This auditor keeps its strict warmup, per-network Adam steps, stochastic target
dropout and first-alpha behaviour explicit.  Market accounting uses only the
previously frozen independent reconstruction helpers, never the execution env.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import audit_btceth_rl_execution_band as prior
from scripts.audit_btceth_rl import (BASELINES, Checks, _audit_run, _check_summary,
    independent_training_calendar, read_csv, read_equity, reconstruct_fills, verify_saved_fingerprints)
from scripts.audit_btceth_rl_exposure_cap import (CASE_FILES, COSTS, GROUP_FILES, SYMBOLS,
    attach_inventory_replay, candidate_score, compare_group_tables, load_market,
    mapped_target, read_json, safe_file, sha, utc)

UPSTREAM_REPOSITORY = "https://github.com/TakuyaHiraoka/Dropout-Q-Functions-for-Doubly-Efficient-Reinforcement-Learning"
UPSTREAM_COMMIT = "4c7adddbaf7573ff9256b16091a3bbdbd536aad4"
UPSTREAM_MANIFEST_SHA256 = "9b0eb6b5e08fefae4a468061ab31e93bcd4d67606750b6196e926f6e103e7560"
UPSTREAM_FILES = {
    "OriginalREDQCodebase/redq/algos/core.py": "16fec92b1de1ce5942599f57079bc3a002b2869429515b4cd68606bda8fb52db",
    "OriginalREDQCodebase/redq/algos/redq_sac.py": "1417451e076f59584e1d5f3cdea1c47366c6625dff5591e3d864dd7c88152cf7",
    "OriginalREDQCodebase/main.py": "79eef30a55ab1d9ebe8de3eb7b186a139062d9b1ace238938622e19f337d3d09",
    "LICENSE": "bd9702c64430509ea51b727c695abb6086d58889fea46b65752e46183ff91681",
}
TRAIN = ("2020-01-01T00:00:00Z", "2025-01-01T00:00:00Z")
VALIDATION = ("2025-01-01T00:00:00Z", "2025-10-23T00:00:00Z")
TEST = ("2025-10-23T00:00:00Z", "2026-10-23T00:00:00Z")
STEPS, EPISODE_DAYS, SEEDS = 32768, 32, tuple(range(5))
PROFILES, BAND, FAMILIES = ("balanced",), .025, {"raw": 1., "cap075": .75}
FLAGS = ("designed_after_first_annual", "test_year_already_seen", "not_fresh_holdout")
PARAMETER_GROUPS = ("policy", "q0", "q1", "target_q0", "target_q1", "log_alpha")
OPTIMIZER_GROUPS = ("optimizer_policy", "optimizer_q0", "optimizer_q1", "optimizer_alpha")
CONFIG_CONTRACT_SHA256 = "d76e838b65d333c0669b5b138aa5bb8e81efe9e88921c25a0d00feb266828c1a"
RECIPE = dict(hidden_sizes=[256, 256], replay_size=1000000, batch_size=256,
    lr=.0003, gamma=.99, polyak=.995, alpha=.2, auto_alpha=True, target_entropy="auto",
    start_steps=5000, delay_update_steps="auto", utd_ratio=20, num_Q=2, num_min=2,
    q_target_mode="min", policy_update_delay=20, target_drop_rate=.01, layer_norm=True)
TARGETS = {"primary_base_net_return_at_least": .05, "primary_base_max_drawdown_magnitude_at_most": .20,
    "primary_stress_net_return_at_least": 0., "primary_beats_equal_hold_return_and_drawdown": True,
    "positive_base_training_seeds_at_least": 3, "stronger_claim_requires_beating_simple_trend_controls": True,
    "targets_are_not_guarantees": True}
SOURCE_FILES = (*prior.FINANCIAL, "scripts/audit_btceth_rl.py", "scripts/audit_btceth_rl_exposure_cap.py",
                "scripts/audit_btceth_rl_execution_band.py", "scripts/btceth_rl_droq.py", "scripts/audit_btceth_rl_droq.py")


class Inputs:
    def __init__(self, output, main, source, checks):
        original = {"study": Path(output), "main": Path(main), "source": Path(source)}
        if any(path.is_symlink() or hasattr(path, "is_junction") and path.is_junction() for path in original.values()):
            raise ValueError("Original evidence roots cannot be links or junctions")
        self.roots = {key: path.resolve() for key, path in original.items()}
        study, reference = self.roots["study"], self.roots["main"]
        if study == reference or study.is_relative_to(reference) or reference.is_relative_to(study):
            raise ValueError("A separate DroQ study cannot nest or overwrite original evidence")
        self.hashes = {key: {} for key in self.roots}
        self.checks = checks

    def capture(self, root, name):
        path = safe_file(self.roots[root], name)
        digest, name = sha(path), str(name).replace("\\", "/")
        old = self.hashes[root].get(name)
        if old is not None:
            self.checks.check(old == digest, "Input changed between reads:" + root + ":" + name)
        self.hashes[root][name] = digest
        return path

    def json(self, root, name):
        return read_json(self.capture(root, name))

    def csv(self, root, name):
        return read_csv(self.capture(root, name))

    def pin(self, root, values, required, label):
        if not self.checks.check(isinstance(values, dict) and set(required).issubset(values), label + ": all required SHA inputs"):
            raise ValueError("Incomplete immutable source graph:" + label)
        for name, expected in values.items():
            self.checks.check(sha(self.capture(root, name)) == expected, label + ":" + name)

    def unchanged(self):
        for root, values in self.hashes.items():
            for name, expected in values.items():
                self.checks.check(sha(safe_file(self.roots[root], name)) == expected, "Unchanged input:" + root + ":" + name)


def baseline_files():
    folder = "evaluation/test/baselines"
    return {*(f"{folder}/{name}" for name in GROUP_FILES),
        *(f"{folder}/{cost}/{key}/{name}" for cost in COSTS for key in BASELINES for name in CASE_FILES[:6])}


def training_files(*, include_started=False):
    names = ("record.json", "model_tensors.pt", "model.json", "training_timeout_transitions.jsonl")
    if include_started:
        names += ("started.json",)
    return {"training_completion.json", *(f"training/droq__balanced__seed{seed}/{name}" for seed in SEEDS for name in names)}


def validation_files():
    return {"validation_completion.json", "validate_completion.json", "validate_process.json", "validate.log",
        *(f"validation/{name}" for name in GROUP_FILES[:3]),
        *(f"validation/{family}/balanced/{name}" for family in FAMILIES for name in GROUP_FILES),
        *(f"validation/{family}/balanced/{cost}/{key}/{name}" for family in FAMILIES
          for cost in COSTS for key in (*map(str, SEEDS), "ensemble") for name in CASE_FILES)}


def verify_fit_authorization(inputs, spec):
    approval = inputs.json("study", "preflight/fit_validation_authorization.json")
    inputs.checks.check(sha(inputs.roots["study"] / "preflight/fit_validation_authorization.json")
        == spec.get("fit_validation_authorization_sha256")
        and approval.get("fit_validation_authorized") is True and approval.get("no_live_trading") is True
        and approval.get("reviewed_source_sha256") == spec.get("source_sha256")
        and set(approval.get("reviewed_source_sha256", {})) == set(SOURCE_FILES)
        and approval.get("hyperparameters") == RECIPE and approval.get("seeds") == list(SEEDS)
        and approval.get("timesteps_per_model") == STEPS and approval.get("budgets") == FAMILIES
        and all(approval.get(flag) is True for flag in FLAGS)
        and utc(approval["authorized_at"]) <= utc(spec["prepared_at"]),
        "Independent prior fit/validation authorization binds all nine reviewed sources and the finite recipe")
    return approval


def verify_gpu_snapshot(actual, declared, checks, label):
    identity = ("index", "uuid", "name", "memory_total_mib")
    checks.check(isinstance(actual, list) and len(actual) == len(declared) == 2
        and [{k: gpu.get(k) for k in identity} for gpu in actual]
        == [{k: gpu.get(k) for k in identity} for gpu in declared], label + ": static physical GPU identities")
    for gpu in actual:
        values = [gpu.get("memory_used_mib"), gpu.get("utilization_percent")]
        checks.check(all(type(value) in (int, float) and math.isfinite(value) and value >= 0 for value in values),
                     label + ": actual finite nonnegative dynamic resource observations")


def verify_action_space_rng(before, after, seed, random_actions, checks, label):
    """Independent local PCG64 stream; never samples a policy or global RNG."""
    generator = np.random.Generator(np.random.PCG64(seed))
    checks.check(before == generator.bit_generator.state, label + ": actual Gym Box seed-specific initial PCG64 state")
    generator.uniform(low=np.full(2, -1.), high=np.full(2, 1.), size=(random_actions, 2))
    checks.check(after == generator.bit_generator.state, label + ": exactly the declared random two-action warmup draws")


def update_counts(env_steps=STEPS, *, start_steps=5000, delay_update_steps=5000,
                  utd_ratio=20, policy_update_delay=20, num_q=2, auto_alpha=True):
    """Independent count of the author's after-store strict-size train loop."""
    integers = (env_steps, start_steps, delay_update_steps, utd_ratio, policy_update_delay, num_q)
    if any(type(v) is not int or v < 0 for v in integers) or min(utd_ratio, policy_update_delay, num_q) == 0:
        raise ValueError("Nonnegative exact integers and positive update ratios required")
    calls = max(0, env_steps - delay_update_steps)
    q_rounds = calls * utd_ratio
    # The last inner round always updates the actor, even for a nonmultiple.
    actor_per_call = (utd_ratio + policy_update_delay - 1) // policy_update_delay
    return dict(environment_steps=env_steps, replay_adds=env_steps,
        random_exploration_actions=min(env_steps, start_steps + 1),
        learning_train_calls=calls, q_inner_rounds=q_rounds,
        q_optimizer_steps=[q_rounds] * num_q, total_q_optimizer_steps=q_rounds * num_q,
        target_updates_per_network=[q_rounds] * num_q,
        policy_optimizer_steps=calls * actor_per_call,
        alpha_optimizer_steps=calls * actor_per_call if auto_alpha else 0)


def q_target(rewards, effective_done, target_qs, next_log_prob, *, alpha, gamma):
    """DroQ N=2 target, without sampling or consuming either random stream."""
    rewards, done, logp = (np.asarray(v, dtype=float) for v in (rewards, effective_done, next_log_prob))
    values = np.asarray(target_qs, dtype=float)
    if (rewards.ndim != 1 or rewards.shape != done.shape or rewards.shape != logp.shape
            or values.shape != (len(rewards), 2) or not np.isfinite(values).all()
            or not all(np.isfinite(v).all() for v in (rewards, done, logp))
            or not np.isin(done, (0., 1.)).all() or not 0 <= gamma <= 1
            or not np.isfinite(alpha) or alpha < 0):
        raise ValueError("Finite aligned two-Q target inputs required")
    return rewards + gamma * (1. - done) * (values.min(axis=1) - alpha * logp)


def q_loss_per_network(predicted_qs, target):
    """MSE(concat(Q0,Q1), expanded_y)*2 equals the sum of each Q MSE."""
    values, target = np.asarray(predicted_qs, dtype=float), np.asarray(target, dtype=float)
    if values.shape != (len(target), 2) or not np.isfinite(values).all() or not np.isfinite(target).all():
        raise ValueError("Finite aligned two-Q loss inputs required")
    return np.square(values - target[:, None]).mean(axis=0)


def policy_loss(qs, log_prob, alpha):
    values, logp = np.asarray(qs, dtype=float), np.asarray(log_prob, dtype=float)
    if (values.shape != (len(logp), 2) or not np.isfinite(values).all()
            or not np.isfinite(logp).all() or not np.isfinite(alpha) or alpha < 0):
        raise ValueError("Finite two-Q policy inputs required")
    return float((alpha * logp - values.mean(axis=1)).mean())


def observation_digest(values):
    array = np.asarray(values, dtype="<f4")
    if array.shape != (27,) or not np.isfinite(array).all():
        raise ValueError("Exactly27 finite observed float32 values required")
    return hashlib.sha256(array.tobytes()).hexdigest()


def verify_timeout_trace(rows, episodes, features, checks, label, *, steps=STEPS):
    """Read original replay arrays after store, before a training-fragment reset."""
    history = [v for v in episodes if int(v["completed_steps"]) > 0]
    checks.check(len(rows) == len(history) == steps // EPISODE_DAYS,
                 label + ": every actual32-day truncation has one stored transition")
    for index, (row, episode) in enumerate(zip(rows, history), start=1):
        transition, end = index * EPISODE_DAYS, utc(row["info_date"])
        checks.check(row.get("transition_number") == transition and row.get("replay_position") == transition - 1
                     and row.get("env_index") == 0,
                     label + ": exact single-environment after-store slot")
        checks.check(end == utc(episode["end_exclusive"]) == utc(episode["completed_until"])
                     and utc(TRAIN[0]) < end <= utc(TRAIN[1]), label + ": training terminal never crosses cutoff")
        checks.check(row.get("done") == row.get("timeout") == 1.
                     and row.get("effective_done") == row.get("stored_done") == 0.
                     and row.get("time_limit_truncated") is True and row.get("terminated") is False
                     and row.get("truncated") is True,
                     label + ": artificial time limit bootstraps without absorbing done")
        values = {}
        for key in ("stored_observation", "terminal_observation", "stored_next_observation"):
            value = np.asarray(row.get(key + "_values"), dtype=float)
            if not checks.check(value.shape == (27,) and np.isfinite(value).all(), label + ": replay27:" + key):
                continue
            checks.check(row.get(key + "_sha256") == observation_digest(value), label + ": observed bytes:" + key)
            values[key] = value
        if len(values) != 3:
            continue
        checks.vector_close(values["terminal_observation"], values["stored_next_observation"],
                            label + ": actual terminal next observation, never reset", atol=0, rtol=0)
        for key, day in (("stored_observation", end - pd.Timedelta(days=1)),
                         ("terminal_observation", end), ("stored_next_observation", end)):
            checks.vector_close(values[key][:23], features.loc[day].to_numpy().astype(np.float32),
                                label + ": independent causal market prefix:" + key, atol=1e-7, rtol=0)


def verify_upstream(directory, checks):
    directory = Path(directory)
    manifest = read_json(safe_file(directory, "verified_manifest.json"))
    checks.check(sha(directory / "verified_manifest.json") == UPSTREAM_MANIFEST_SHA256,
                 "Previously reviewed whole-file author provenance manifest")
    checks.check(manifest.get("repository") == UPSTREAM_REPOSITORY and manifest.get("commit") == UPSTREAM_COMMIT
                 and manifest.get("license") == "MIT" and manifest.get("file_count") == 27
                 and manifest.get("all_git_blob_sha1_verified") is True, "Exact author implementation, not an SB3 substitute")
    files = manifest.get("files", [])
    checks.check(len(files) == 27 and len({v["path"] for v in files}) == 27, "All authored upstream paths unique")
    for item in files:
        content = safe_file(directory, item["path"]).read_bytes()
        git_sha = hashlib.sha1(b"blob " + str(len(content)).encode() + b"\0" + content).hexdigest()
        checks.check(len(content) == item["size"] and hashlib.sha256(content).hexdigest() == item["sha256"]
                     and git_sha == item["git_blob_sha1"] and item.get("git_blob_verified") is True,
                     "Original raw bytes and Git blob:" + item["path"])
    by_name = {v["path"]: v["sha256"] for v in files}
    checks.check(all(by_name.get(k) == v for k, v in UPSTREAM_FILES.items()), "Core, agent, unused CLI and MIT independently pinned")


def tensor_parameter_digest(tensors, groups):
    digest = hashlib.sha256()
    selected = {name: key for group in PARAMETER_GROUPS for name, key in
                ((group + "." + n, key) for n, key in groups[group]["named_tensors"].items())}
    for name, key in sorted(selected.items()):
        value = tensors[key].detach().cpu().contiguous()
        header = json.dumps([name, str(value.dtype), list(value.shape)], separators=(",", ":")).encode()
        digest.update(len(header).to_bytes(8, "big")); digest.update(header); digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def verify_network_tensors(tensors, groups, modules):
    policy_shapes = {
        "hidden_layers.0.weight": [256, 27], "hidden_layers.0.bias": [256],
        "hidden_layers.1.weight": [256, 256], "hidden_layers.1.bias": [256],
        "last_fc_layer.weight": [2, 256], "last_fc_layer.bias": [2],
        "last_fc_log_std.weight": [2, 256], "last_fc_log_std.bias": [2]}
    q_shapes = {
        "hidden_layers.0.weight": [256, 29], "hidden_layers.0.bias": [256],
        "hidden_layers.2.weight": [256], "hidden_layers.2.bias": [256],
        "hidden_layers.3.weight": [256, 256], "hidden_layers.3.bias": [256],
        "hidden_layers.5.weight": [256], "hidden_layers.5.bias": [256],
        "last_fc_layer.weight": [1, 256], "last_fc_layer.bias": [1]}
    if set(modules) != set(PARAMETER_GROUPS) - {"log_alpha"}:
        raise ValueError("Five actual network module observations required")
    for name in modules:
        names, expected = groups[name]["named_tensors"], policy_shapes if name == "policy" else q_shapes
        if set(names) != set(expected) or any(list(tensors[names[k]].shape) != v for k, v in expected.items()):
            raise ValueError("Original author network architecture mismatch:" + name)
        if any(str(tensors[key].dtype) != "torch.float32" for key in names.values()):
            raise ValueError("Author learner requires actual float32 network parameters")
        module = modules[name]
        if (module.get("class") != ("TanhGaussianPolicy" if name == "policy" else "Mlp")
                or module.get("training") is not True or module.get("requires_grad_all") is not True
                or module.get("dropout_p") != ([] if name == "policy" else [.01, .01])
                or module.get("layer_norm_shapes") != ([] if name == "policy" else [[256], [256]])):
            raise ValueError("Official active target dropout/LayerNorm/mode changed:" + name)
    alpha = list(groups["log_alpha"]["named_tensors"].values())
    if len(alpha) != 1 or list(tensors[alpha[0]].shape) != [1] or str(tensors[alpha[0]].dtype) != "torch.float32":
        raise ValueError("One actual trainable entropy-temperature tensor required")


def decode_tensor_tree(node, tensor_keys):
    """Decode inert JSON tags only; a tensor reference remains a tuple of text."""
    if node is None or type(node) in (str, bool, int, float):
        if isinstance(node, float) and not math.isfinite(node):
            raise ValueError("Finite tagged optimizer metadata required")
        return node
    if not isinstance(node, dict) or len(node) != 1:
        raise ValueError("One inert JSON container tag required")
    tag, value = next(iter(node.items()))
    if tag == "tensor":
        if value not in tensor_keys:
            raise ValueError("Unknown actual tensor reference")
        return ("tensor", value)
    if tag in ("list", "tuple"):
        values = [decode_tensor_tree(v, tensor_keys) for v in value]
        return values if tag == "list" else tuple(values)
    if tag == "dict":
        result = {}
        for key, value in value:
            key = decode_tensor_tree(key, tensor_keys)
            if type(key) not in (str, int) or key in result:
                raise ValueError("Unique inert scalar optimizer state keys required")
            result[key] = decode_tensor_tree(value, tensor_keys)
        return result
    raise ValueError("Executable or unknown checkpoint tag rejected")


def verify_optimizer_tree(tensors, groups, trees):
    if set(trees) != set(OPTIMIZER_GROUPS):
        raise ValueError("All four original Adam state trees required")
    for optimizer, parameter_group in (("optimizer_policy", "policy"), ("optimizer_q0", "q0"),
                                       ("optimizer_q1", "q1"), ("optimizer_alpha", "log_alpha")):
        group = groups[optimizer]
        tree = decode_tensor_tree(trees[optimizer], set(group["tensor_keys"]))
        if set(tree) != {"state", "param_groups"} or len(tree["param_groups"]) != 1:
            raise ValueError("Original single-param-group Adam structure required")
        parameters = tree["param_groups"][0]
        ids = parameters["params"]
        if (len(ids) != len(set(ids)) or set(ids) != set(tree["state"])
                or len(ids) != len(groups[parameter_group]["named_tensors"])
                or parameters.get("lr") != .0003 or list(parameters.get("betas", ())) != [.9, .999]
                or parameters.get("eps") != 1e-8 or parameters.get("weight_decay") != 0
                or parameters.get("amsgrad") is not False):
            raise ValueError("Actual Adam defaults or optimized parameter population changed:" + optimizer)
        actual_steps, actual_refs = [], set()
        expected_parameters = list(groups[parameter_group]["named_tensors"].values())
        for parameter_id, parameter_key in zip(ids, expected_parameters):
            state = tree["state"][parameter_id]
            if set(state) != {"step", "exp_avg", "exp_avg_sq"}:
                raise ValueError("Complete original Adam state required")
            for kind, reference in state.items():
                if not isinstance(reference, tuple) or len(reference) != 2 or reference[0] != "tensor":
                    raise ValueError("Adam state must refer to actual saved Tensor bytes")
                actual_refs.add(reference[1])
                if kind == "step":
                    actual_steps.append(reference[1])
                elif list(tensors[reference[1]].shape) != list(tensors[parameter_key].shape):
                    raise ValueError("Actual Adam moments must match their original parameter shape")
        if set(actual_steps) != set(group["step_tensor_keys"]) or actual_refs != set(group["tensor_keys"]):
            raise ValueError("Adam step mirrors must identify the true tagged state, not unrelated counter tensors")


def read_tensor_checkpoint(tensor_file, manifest_file, *, expected_counters=None):
    """Only weights_only CPU tensors; no model, optimizer or pickle construction."""
    import torch
    before = torch.get_rng_state().clone()
    cuda_before = torch.cuda.is_initialized()
    tensors = torch.load(tensor_file, map_location="cpu", weights_only=True)
    if not isinstance(tensors, dict) or not tensors or any(not isinstance(k, str) or not isinstance(v, torch.Tensor) for k, v in tensors.items()):
        raise ValueError("Checkpoint must be a flat string-to-Tensor dictionary")
    manifest = read_json(manifest_file)
    if (manifest.get("tensor_file") != Path(tensor_file).name or manifest.get("tensor_sha256") != sha(tensor_file)
            or manifest.get("upstream_commit") != UPSTREAM_COMMIT
            or manifest.get("upstream_manifest_sha256") != UPSTREAM_MANIFEST_SHA256
            or manifest.get("effective_target_entropy") != -2 or manifest.get("hidden_sizes") != [256, 256]
            or manifest.get("action_limit") != 1.):
        raise ValueError("Checkpoint author source, Tensor bytes or actual dimensions/entropy changed")
    specs, groups = manifest["tensors"], manifest["tensor_groups"]
    if set(specs) != set(tensors):
        raise ValueError("Checkpoint tensor inventory differs from its metadata")
    for key, value in tensors.items():
        if (value.device.type != "cpu" or list(value.shape) != specs[key]["shape"]
                or str(value.dtype) != specs[key]["dtype"] or not np.isfinite(value.detach().numpy()).all()):
            raise ValueError("Invalid actual finite CPU tensor:" + key)
    required = {*PARAMETER_GROUPS, *OPTIMIZER_GROUPS, "rng_torch", "rng_numpy", "counters"}
    if set(groups) != required:
        raise ValueError("Exactly five networks, log_alpha, four Adam states, RNG and actual counters required")
    covered = []
    for name, group in groups.items():
        keys = group["tensor_keys"]
        if len(set(keys)) != len(keys) or not set(keys).issubset(tensors):
            raise ValueError("Invalid named tensor group:" + name)
        covered.extend(keys)
        if name in PARAMETER_GROUPS and (not group["named_tensors"] or set(group["named_tensors"].values()) != set(keys)):
            raise ValueError("Actual model state names must identify every parameter:" + name)
    if len(covered) != len(set(covered)) or set(covered) != set(tensors):
        raise ValueError("Every stored tensor must belong to exactly one declared group")
    verify_network_tensors(tensors, groups, manifest["modules"])
    verify_optimizer_tree(tensors, groups, manifest["trees"])
    counts = expected_counters or update_counts()
    if (manifest.get("counters") != counts
            or manifest.get("optimizer_classes") != {key: "Adam" for key in OPTIMIZER_GROUPS}):
        raise ValueError("Actual optimizer classes and complete counter mirrors required")
    optimizer_expected = {"optimizer_policy": counts["policy_optimizer_steps"],
        "optimizer_q0": counts["q_optimizer_steps"][0], "optimizer_q1": counts["q_optimizer_steps"][1],
        "optimizer_alpha": counts["alpha_optimizer_steps"]}
    for group_name, expected in optimizer_expected.items():
        group = groups[group_name]; keys = group["step_tensor_keys"]
        parameter_group = {"optimizer_policy": "policy", "optimizer_q0": "q0", "optimizer_q1": "q1", "optimizer_alpha": "log_alpha"}[group_name]
        if (not keys or not set(keys).issubset(group["tensor_keys"]) or len(keys) != len(group["step_values"])
                or len(keys) != len(groups[parameter_group]["named_tensors"])):
            raise ValueError("Actual Adam per-parameter step tensors required:" + group_name)
        for key, mirrored in zip(keys, group["step_values"]):
            value = tensors[key]
            if value.numel() != 1 or value.item() != expected or mirrored != expected:
                raise ValueError("Actual Adam step disagrees with strict author loop:" + group_name)
    counters = groups["counters"]["named_tensors"]
    scalar_counts = {k: v for k, v in counts.items() if isinstance(v, int)}
    if set(counters) != set(scalar_counts):
        raise ValueError("All actual integer environment/learning counters required")
    for name, expected in scalar_counts.items():
        value = tensors[counters[name]]
        if value.numel() != 1 or value.dtype != torch.int64 or value.item() != expected:
            raise ValueError("Actual stored integer counter mismatch:" + name)
    log_alpha = float(tensors[next(iter(groups["log_alpha"]["named_tensors"].values()))].item())
    if (not math.isclose(manifest.get("log_alpha_value", float("nan")), log_alpha, rel_tol=0, abs_tol=1e-12)
            or not math.isclose(manifest.get("effective_alpha", float("nan")), math.exp(log_alpha), rel_tol=1e-6, abs_tol=1e-12)):
        raise ValueError("Trained current alpha must match the actual saved log_alpha tensor")
    if not torch.equal(before, torch.get_rng_state()) or cuda_before != torch.cuda.is_initialized():
        raise ValueError("Independent checkpoint reading must not consume RNG or initialize CUDA")
    digest = tensor_parameter_digest(tensors, groups)
    if digest != manifest.get("parameter_sha256"):
        raise ValueError("Parameter SHA must identify actual model tensors alone")
    return {"parameter_sha256": digest, "tensor_count": len(tensors),
        "actual_counters": counts, "effective_alpha": manifest["effective_alpha"], "log_alpha_value": log_alpha,
        "cuda_initialized_changed": False, "cpu_rng_consumed": False}


def verify_protocol(inputs, dataset_path):
    checks = inputs.checks
    spec, receipt = inputs.json("study", "protocol.json"), inputs.json("study", "protocol_receipt.json")
    old, runtime = inputs.json("main", "frozen_spec.json"), inputs.json("main", "runtime.json")
    fixed = dict(algorithm="droq", profiles={"balanced": {"drawdown_penalty": .25, "rebalance_band": BAND}},
        seeds=list(SEEDS), budgets=FAMILIES, train=list(TRAIN), validation=list(VALIDATION), test=list(TEST),
        capital=10000., costs=COSTS, timesteps_per_model=STEPS, train_episode_days=EPISODE_DAYS,
        model_count=5, validation_cases=24, validation_repeats=24, test_cases=12, test_repeats=12,
        cash_floor=.05, max_asset_weight=.95, hyperparameters=RECIPE, recipe_is_pure_official_default=False,
        seed_selection=False, selection_uses_test=False, new_recipe_search=False,
        annual_requires_separate_authorization=True,
        selection_score="mean(log1p(net_return)-0.5*abs(max_drawdown)-2*max(abs(max_drawdown)-0.20,0)) across base/stress",
        tie_break="raw before cap075, then profile lexical identifier", research_targets=TARGETS)
    for key, expected in fixed.items():
        checks.check(spec.get(key) == expected, "One finite predeclared DroQ experiment:" + key)
    checks.check(all(spec.get(flag) is True for flag in FLAGS), "Known-year disclosure remains explicit")
    checks.check(receipt.get("protocol_file") == "protocol.json"
                 and receipt.get("protocol_sha256") == sha(inputs.roots["study"] / "protocol.json")
                 and utc(spec["prepared_at"]) <= utc(receipt["sealed_at"]), "Whole-file protocol sealed before execution")
    checks.check(spec.get("original_frozen_spec_sha256") == sha(inputs.roots["main"] / "frozen_spec.json")
                 and old.get("timesteps_per_model") == STEPS and old.get("research_targets") == TARGETS
                 and tuple(old["train"]) == TRAIN and tuple(old["validation"]) == VALIDATION
                 and tuple(old["test"]) == TEST, "Original dates, environment budget and five unlowered goals")
    sources = spec.get("source_sha256", {})
    inputs.pin("source", sources, SOURCE_FILES, "Explicit DroQ and independent financial source DAG")
    checks.check(set(sources) == set(SOURCE_FILES)
                 and all(sources[name] == old["code_sha256"][name] for name in prior.FINANCIAL),
                 "Original four financial sources unchanged; no TD3/SB3 substitution")
    for name, expected in sources.items():
        checks.check(sha(inputs.capture("study", "sources/" + name)) == expected, "Archived source bytes:" + name)
    verify_fit_authorization(inputs, spec)
    inputs.pin("main", spec.get("original_input_sha256"),
               baseline_files() | {"frozen_spec.json", "runtime.json", "independent_audit.json"}, "Fourteen original funded controls fixed in advance")
    upstream = inputs.roots["study"] / "upstream"
    verify_upstream(upstream, checks)
    manifest = read_json(upstream / "verified_manifest.json")
    expected_upstream = {v["path"] for v in manifest["files"]} | {"verified_manifest.json", "github_commit.json", "github_tree.json"}
    checks.check(set(spec.get("upstream_evidence_sha256", {})) == expected_upstream, "Complete authored sources and three provenance files")
    for name, expected in spec.get("upstream_evidence_sha256", {}).items():
        checks.check(sha(inputs.capture("study", "upstream/" + name)) == expected, "Pinned official source:" + name)
    contract = inputs.json("study", "effective_config_contract.json")
    checks.check(sha(inputs.roots["study"] / "effective_config_contract.json") == CONFIG_CONTRACT_SHA256
                 == spec.get("effective_config_contract_sha256")
                 and contract.get("effective_constructor_kwargs") == RECIPE
                 and contract.get("official_commit") == UPSTREAM_COMMIT,
                 "Reviewed exact effective arguments; ignored MuJoCo CLI is never the training driver")
    dataset_path = Path(dataset_path)
    checks.check(sha(dataset_path) == spec.get("dataset_sha256") == old["dataset_sha256"]
                 and sha(dataset_path.parent / "manifest.json") == spec.get("dataset_manifest_sha256") == old["dataset_manifest_sha256"],
                 "Original exact market file and manifest, no converted or synthetic training data")
    gpus = spec.get("physical_gpus", [])
    checks.check(gpus == runtime.get("physical_gpus") and len(gpus) == 2
                 and len({gpu["uuid"] for gpu in gpus}) == 2 and all("V100" in gpu["name"] for gpu in gpus), "Original two physical V100 identities")
    if len(gpus) == 2:
        checks.check(spec.get("seed_gpu_assignments") == {str(seed): gpus[seed % 2]["uuid"] for seed in SEEDS}
                     and spec.get("family_gpu_assignments") == {family: gpu["uuid"] for family, gpu in zip(FAMILIES, gpus)},
                     "Fixed round-robin training seeds and separate raw/cap inference workers")
    checks.check(spec.get("runtime_packages") == old.get("runtime_packages"), "No dependency installation or version fallback")
    if spec.get("resource_allowance_sha256") is not None:
        allowance = inputs.json("study", "resource_allowance.json")
        checks.check(sha(inputs.roots["study"] / "resource_allowance.json") == spec["resource_allowance_sha256"]
                     and allowance.get("allowed_instance") == spec.get("allowed_compute_instance")
                     and allowance.get("allowed_gpu_rows") == spec.get("allowed_gpu_rows"), "Only the reviewed UI process instance exception")
    if checks.failed:
        raise ValueError("DroQ source/contract/protocol failed before tensor or financial outcome reads")
    return spec, old


def verify_phase(inputs, phase, spec):
    checks = inputs.checks
    process, complete = inputs.json("study", phase + "_process.json"), inputs.json("study", phase + "_completion.json")
    preflight = inputs.json("study", "preflight/" + phase + ".json")
    receipt = inputs.json("study", "protocol_receipt.json")
    command = process.get("command", [])
    checks.check(process.get("phase") == phase and process.get("status") == "completed" and process.get("exit_code") == 0
                 and process.get("source_sha256") == spec["source_sha256"]["scripts/btceth_rl_droq.py"]
                 and process.get("protocol_sha256") == sha(inputs.roots["study"] / "protocol.json"), phase + ": real source-bound terminal0")
    checks.check(len(command) >= 5 and str(command[2]).replace("\\", "/").endswith("/btceth_rl_droq.py")
                 and command.count("--phase") == 1 and command[command.index("--phase") + 1] == "_" + phase,
                 phase + ": actual private coordinator command")
    checks.check(all(type(process.get(k)) is int and process[k] > 0 for k in ("supervisor_pid", "child_pid"))
                 and process["supervisor_pid"] != process["child_pid"], phase + ": distinct real supervisor and child")
    checks.check(process.get("log_relative_path") == phase + ".log"
                 and process.get("log_sha256") == sha(inputs.capture("study", phase + ".log")), phase + ": original terminal log")
    checks.check(complete.get("protocol_sha256") == sha(inputs.roots["study"] / "protocol.json")
                 and utc(receipt["sealed_at"]) <= utc(preflight["at"]) <= utc(process["started_at"])
                 <= utc(complete["at"]) <= utc(process["completed_at"]), phase + ": freeze precedes execution and completion")
    verify_gpu_snapshot(preflight.get("physical_gpus", []), spec["physical_gpus"], checks, phase)
    checks.check(preflight.get("command") == ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"],
                 phase + ": unfiltered original two-card resource snapshot")
    rows = []
    for line in preflight.get("unfiltered_stdout", "").splitlines():
        if line.strip():
            uuid, pid = [v.strip() for v in line.split(",")]
            rows.append({"gpu_uuid": uuid, "pid": int(pid)})
    checks.check(not rows or rows == spec.get("allowed_gpu_rows"), phase + ": no undeclared compute task suppressed")
    if rows:
        checks.check(preflight.get("allowed_compute_instance") == spec.get("allowed_compute_instance"), phase + ": exact reviewed UI identity")
    return process, complete


def audit_training(inputs, spec, dataset, features):
    checks = inputs.checks
    fit, completion = inputs.json("study", "fit_process.json"), inputs.json("study", "training_completion.json")
    checks.check(completion.get("model_count") == len(completion.get("records", [])) == 5
                 and completion.get("protocol_sha256") == sha(inputs.roots["study"] / "protocol.json"), "Five actual new DroQ models completed")
    completed = {(v["profile"], v["seed"]): v for v in completion.get("records", [])}
    calendar, records = independent_training_calendar(dataset, *TRAIN, EPISODE_DAYS), {}
    expected_counts = update_counts()
    for seed in SEEDS:
        folder, label = f"training/droq__balanced__seed{seed}", f"DroQ balanced seed{seed}"
        record, started = inputs.json("study", folder + "/record.json"), inputs.json("study", folder + "/started.json")
        checks.check((record.get("algorithm"), record.get("profile"), record.get("seed")) == ("droq", "balanced", seed)
                     and record == completed.get(("balanced", seed))
                     and record.get("protocol_sha256") == started.get("protocol_sha256") == sha(inputs.roots["study"] / "protocol.json"), label + ": exact new seed identity")
        task = started.get("task", {})
        checks.check(task.get("seed") == seed and task.get("profile") == "balanced"
                     and utc(fit["started_at"]) <= utc(started["at"]) <= utc(record["completed_at"]) <= utc(fit["completed_at"]), label + ": one completed training attempt")
        checks.check(record.get("actual_timesteps") == STEPS and record.get("actual_counters") == expected_counts
                     and record.get("hyperparameters") == RECIPE and record.get("recipe_is_pure_official_default") is False,
                     label + ": finite real driver budget, per-optimizer counts and single research assumption")
        verify_action_space_rng(record.get("action_space_rng_before"), record.get("action_space_rng_after"),
                                seed, expected_counts["random_exploration_actions"], checks, label)
        transition_sha = record.get("actual_transition_sha256", "")
        checks.check(isinstance(transition_sha, str) and len(transition_sha) == 64
                     and set(transition_sha) <= set("0123456789abcdef"), label + ": actual complete original replay-write fingerprint retained")
        effective = record.get("effective_hyperparameters", {})
        expected_effective = {name: RECIPE[name] for name in ("replay_size", "batch_size", "lr", "gamma", "polyak",
            "auto_alpha", "start_steps", "utd_ratio", "num_Q", "num_min", "q_target_mode", "policy_update_delay")}
        expected_effective.update(hidden_sizes=[256, 256], delay_update_steps=5000, target_entropy=-2.,
            dropout_per_q=[[.01, .01], [.01, .01]], layer_norm_per_q=[[[256], [256]], [[256], [256]]], target_training=[True, True])
        checks.check(set(effective) == {*expected_effective, "alpha_current", "log_alpha_current"}
                     and all(effective.get(k) == v for k, v in expected_effective.items()), label + ": actual live agent hyperparameters, not ignored CLI arguments")
        loaded = record.get("upstream_loaded_sources", {})
        checks.check(set(loaded) == set(UPSTREAM_FILES) - {"LICENSE", "OriginalREDQCodebase/main.py"}
                     and all(v.get("sha256") == UPSTREAM_FILES[k]
                             and str(v.get("file", "")).replace("\\", "/").endswith("/" + k) for k, v in loaded.items())
                     and record.get("algorithm_class") == "redq.algos.redq_sac.REDQSACAgent",
                     label + ": genuine author agent/core, no SB3 replacement or cache ambiguity")
        device = record.get("device", {})
        checks.check(device.get("device") == "cuda:0" and device.get("capability") == [7, 0]
                     and "V100" in device.get("name", "") and device.get("visible_gpu_uuid") == spec["seed_gpu_assignments"][str(seed)]
                     and device == started.get("device"), label + ": exact assigned seed V100")
        paths = {"model_tensor_file": ("model_tensors.pt", "model_tensor_sha256"),
                 "model_manifest_file": ("model.json", "model_manifest_sha256"),
                 "timeout_trace_relative_path": ("training_timeout_transitions.jsonl", "timeout_trace_sha256")}
        captured = {}
        for field, (name, digest_field) in paths.items():
            expected = folder + "/" + name
            checks.check(record.get(field) == expected, label + ": exact owned checkpoint/trace path:" + name)
            captured[name] = inputs.capture("study", expected)
            checks.check(sha(captured[name]) == record.get(digest_field), label + ": exact saved bytes:" + name)
        checkpoint = read_tensor_checkpoint(captured["model_tensors.pt"], captured["model.json"])
        checks.check(checkpoint["parameter_sha256"] == record.get("parameter_sha256")
                     and checkpoint["actual_counters"] == record.get("actual_counters"), label + ": independently read tensor parameters and actual Adam/counter states")
        checks.close(effective.get("alpha_current"), checkpoint["effective_alpha"], label + ": actual trained entropy coefficient", atol=1e-12)
        checks.close(effective.get("log_alpha_current"), checkpoint["log_alpha_value"], label + ": actual trained log_alpha", atol=1e-12)
        total = 0
        for episode in record.get("training_episode_history", []):
            first, end, until = (utc(episode[key]) for key in ("start", "end_exclusive", "completed_until"))
            steps = int(episode["completed_steps"])
            checks.check(first in calendar["eligible"] and end - first == pd.Timedelta(days=EPISODE_DAYS)
                         and utc(TRAIN[0]) <= first <= until <= end <= utc(TRAIN[1])
                         and until - first == pd.Timedelta(days=steps), label + ": complete contiguous causal training fragment")
            checks.check(episode.get("terminated") is False and episode.get("truncated") == (steps == EPISODE_DAYS), label + ": timeout bootstrap including unused final reset")
            total += steps
        checks.check(total == STEPS, label + ": exactly32768 completed market transitions")
        rows = [json.loads(line) for line in captured["training_timeout_transitions.jsonl"].read_text(encoding="utf-8").splitlines() if line.strip()]
        verify_timeout_trace(rows, record["training_episode_history"], features, checks, label)
        quality = record.get("training_window_quality", {})
        checks.check(all(quality.get(key) == calendar[key] for key in ("requested_train_days", "candidate_windows", "eligible_windows", "excluded_windows"))
                     and quality.get("covered_training_days") == calendar["covered_days"]
                     and len(record.get("features", [])) == len(set(record.get("features", []))) == 27, label + ": actual shared27-state market calendar and coverage")
        records[("balanced", seed)] = record
    checks.check(set(records) == set(completed) == {("balanced", seed) for seed in SEEDS}, "All five fixed seeds retained without selection")
    return records, {k: v for k, v in calendar.items() if k not in ("eligible", "market", "features")}


# Independently reconstructed funded-account mathematics; no execution-engine calls.

def action_weights(action, family):
    if family not in FAMILIES:
        raise ValueError("Only the two declared evaluation budgets are permitted")
    values = np.asarray(action, dtype=float)
    if values.shape != (2,):
        raise ValueError("Exactly two finite original actions are required")
    raw, capped = mapped_target(*values)
    return raw, raw.copy() if family == "raw" else capped


def validate_group_rows(frame, *, split, family, profile):
    dates = VALIDATION if split == "validation" else TEST if split == "test" else None
    if dates is None or family not in FAMILIES or profile not in PROFILES:
        raise ValueError("Only predeclared DroQ family/profile/split identities are valid")
    required = {"run_id", "algorithm", "family", "profile", "scenario", "split", "seed", "primary_policy",
        "initial_capital", "start", "end_exclusive", "net_return", "max_drawdown", "final_equity", "fees",
        "rebalance_band", "exposure_multiplier"}
    if not required.issubset(frame) or len(frame) != 12 or frame.run_id.duplicated().any():
        raise ValueError("Each group requires exactly five seeds plus one real ensemble at both costs")
    fixed = dict(algorithm="droq", family=family, profile=profile, split=split,
                 initial_capital=10000., rebalance_band=BAND, exposure_multiplier=FAMILIES[family])
    if any(not frame[key].eq(value).all() for key, value in fixed.items()):
        raise ValueError("The DroQ financial identity/recipe cannot change between candidates")
    if (not pd.to_datetime(frame.start, utc=True).eq(utc(dates[0])).all()
            or not pd.to_datetime(frame.end_exclusive, utc=True).eq(utc(dates[1])).all()):
        raise ValueError("Evaluation must use the complete original fixed window")
    expected = {f"{split}__{cost}__{family}__droq__{profile}__{key}" for cost in COSTS
                for key in (*map(str, SEEDS), "ensemble")}
    if set(frame.run_id) != expected or not frame.primary_policy.map(lambda x: isinstance(x, (bool, np.bool_))).all():
        raise ValueError("All twelve fixed run IDs and true primary-policy flags are required")
    for cost in COSTS:
        rows = frame.loc[frame.scenario.eq(cost)]
        primary, seeds = rows.loc[rows.primary_policy], rows.loc[~rows.primary_policy]
        if len(primary) != 1 or primary.seed.notna().any() or len(seeds) != 5 or set(seeds.seed) != set(SEEDS):
            raise ValueError("No omitted cost or seed selection")
    numbers = frame[["net_return", "max_drawdown", "final_equity", "fees"]].to_numpy(dtype=float)
    if (not np.isfinite(numbers).all() or (frame.net_return <= -1).any() or (frame.max_drawdown > 0).any()
            or (frame.max_drawdown <= -1).any() or (frame.fees < 0).any() or (frame.final_equity <= 0).any()
            or (frame.final_equity / 10000 - 1 - frame.net_return).abs().max() > 1e-10):
        raise ValueError("Financial summary does not reconcile to the funded capital")


def rank_candidates(groups):
    if set(groups) != {(family, profile) for family in FAMILIES for profile in PROFILES}:
        raise ValueError("Exactly two declared validation candidates are required")
    candidates = []
    for (family, profile), frame in groups.items():
        validate_group_rows(frame, split="validation", family=family, profile=profile)
        scenarios = {cost: frame.loc[frame.scenario.eq(cost) & frame.primary_policy].iloc[0] for cost in COSTS}
        candidates.append(dict(family=family, algorithm="droq", profile=profile,
            exposure_multiplier=FAMILIES[family], rebalance_band=BAND, score=candidate_score(scenarios["base"], scenarios["stress"]),
            **{cost: {"run_id": str(row.run_id), **{key: float(row[key]) for key in
                ("net_return", "max_drawdown", "final_equity", "fees")}} for cost, row in scenarios.items()}))
    return sorted(candidates, key=lambda row: (-row["score"], 0 if row["family"] == "raw" else 1, row["profile"]))


def audit_member_targets(trace, weights, rebuilt, fills, *, start, end, seeds, family, features, checks, label):
    fields = {"day", "seed", "raw_action0", "raw_action1", "raw_btc", "raw_eth", "raw_cash",
        "mapped_btc", "mapped_eth", "mapped_cash", "target_btc", "target_eth", "target_cash",
        "observation_values_json", "observation_sha256", "observation_asof", "observation_cutoff_exclusive",
        "observation_btc_weight", "observation_eth_weight", "observation_cash_weight", "observation_drawdown"}
    if not checks.check(fields.issubset(trace), label + ": complete original action, budget, common-state schema"):
        return
    trace, weights = trace.copy(), weights.copy()
    trace["day"], weights["date"] = pd.to_datetime(trace.day, utc=True), pd.to_datetime(weights.date, utc=True)
    days = pd.date_range(utc(start), utc(end), freq="D", inclusive="left")
    checks.check(len(trace) == len(days) * len(seeds) and not trace[["day", "seed"]].duplicated().any()
                 and set(trace.day) == set(days) and set(trace.seed) == set(seeds),
                 label + ": exact complete model/day Cartesian product")
    checks.check(len(weights) == len(days) * 2 and not weights[["date", "asset_key"]].duplicated().any()
                 and set(weights.date) == set(days), label + ": complete two-coin daily weights")
    if not checks.check({"coin_values", "inventory_quantity", "inventory_presence"}.issubset(rebuilt),
                         label + ": independently reconstructed exact inventory required"):
        return
    eq = rebuilt["equity"]
    peaks = eq.equity.cummax().clip(lower=10000.)
    fill_days = set(pd.to_datetime(fills.timestamp, utc=True).dt.normalize()) if len(fills) else set()
    for day in days:
        members = trace.loc[trace.day.eq(day)].sort_values("seed")
        if not checks.check(members.seed.tolist() == sorted(seeds), label + ": all fixed model members exactly once"):
            continue
        before = day - pd.Timedelta(hours=1)
        if day == utc(start):
            account = np.array([0., 0., 1., 0.])
        else:
            state = eq.loc[before]
            account = np.r_[rebuilt["coin_values"].loc[before].to_numpy() / state.equity,
                            state.cash / state.equity, 1 - state.equity / peaks.loc[before]]
        hashes = members.observation_sha256.astype(str)
        checks.check(hashes.nunique() == 1 and all(len(h) == 64 and set(h) <= set("0123456789abcdef") for h in hashes),
                     label + ": all members observe this same actual funded account")
        targets = []
        for row in members.itertuples(index=False):
            obs = np.asarray(json.loads(row.observation_values_json), dtype=float)
            if checks.check(obs.shape == (27,) and np.isfinite(obs).all(), label + ": actual finite27-state input"):
                checks.check(observation_digest(obs) == row.observation_sha256, label + ": actual input float32 SHA")
                checks.vector_close(obs[:23], features.loc[day].to_numpy().astype(np.float32),
                                    label + ": independent causal market prefix", atol=1e-7, rtol=0)
                checks.vector_close(obs[-4:], account, label + ": true actual-account state, not averaged seed NAV", atol=1e-7, rtol=0)
                checks.vector_close([row.observation_btc_weight, row.observation_eth_weight,
                                    row.observation_cash_weight, row.observation_drawdown], obs[-4:],
                                    label + ": explicit current account fields match full input", atol=0, rtol=0)
            checks.check(utc(row.observation_asof) == before and utc(row.observation_cutoff_exclusive) == day,
                         label + ": signal input strictly before execution day")
            if hasattr(row, "feature_input_cutoff"):
                checks.check(utc(row.feature_input_cutoff) == before, label + ": exact previous completed hour cutoff")
            raw, mapped = action_weights([row.raw_action0, row.raw_action1], family)
            checks.vector_close([row.raw_btc, row.raw_eth, row.raw_cash], raw, label + ": original action mapping", atol=1e-12)
            checks.vector_close([row.mapped_btc, row.mapped_eth, row.mapped_cash], mapped,
                                label + ": declared budget acts before ensemble mean and leaves cash residual", atol=1e-12)
            targets.append(mapped)
        target = np.mean(targets, axis=0)
        gross = .95 * FAMILIES[family]
        checks.check(np.isfinite(target).all() and (target >= 0).all() and target[:2].sum() <= gross + 1e-12
                     and target[2] >= 1 - gross - 1e-12, label + ": funded declared gross/cash budget")
        checks.vector_close(members[["target_btc", "target_eth", "target_cash"]],
                            np.tile(target, (len(members), 1)), label + ": true arithmetic mean of all five target vectors", atol=1e-12)
        daily = weights.loc[weights.date.eq(day)]
        if not checks.check(len(daily) == 2 and set(daily.asset_key) == set(SYMBOLS), label + ": original spot coin identities"):
            continue
        daily = daily.set_index("asset_key").loc[list(SYMBOLS)]
        checks.check(daily.slot.tolist() == [0, 1] and daily.symbol.tolist() == ["BTC", "ETH"], label + ": original coin slots")
        checks.vector_close(daily.target_weight, target[:2], label + ": actual requested mean target", atol=1e-12)
        close = day + pd.Timedelta(hours=23)
        actual = rebuilt["coin_values"].loc[close].to_numpy() / eq.loc[close, "equity"]
        checks.vector_close(daily.actual_weight, actual, label + ": actual post-cost marked inventory", atol=1e-10)
        execute = prior.rebalance_decision(target, account, BAND)
        checks.check(daily.rebalanced.map(lambda value: isinstance(value, (bool, np.bool_))).all()
                     and daily.rebalanced.eq(execute).all(), label + ": unchanged2.5pp band and unconditional cash exit")
        if not execute:
            checks.check(day not in fill_days, label + ": no fills when rebalance band skips execution")


def audit_case(directory, dataset, row, records, checks, *, split, family, profile, features, gpus, dates=None):
    start, end = dates or (VALIDATION if split == "validation" else TEST)
    label, cost = row["run_id"], row["scenario"]
    seeds = list(SEEDS) if row["primary_policy"] else [int(row["seed"])]
    equity, fills = read_equity(directory / "equity_1h.csv.gz"), read_csv(directory / "fills.csv")
    metadata, weights = read_json(directory / "metadata.json"), read_csv(directory / "weights.csv")
    fingerprints = verify_saved_fingerprints(directory, checks, label)
    rebuilt = reconstruct_fills(dataset, fills, equity, metadata, start=start, end=end,
                               **COSTS[cost], checks=checks, label=label)
    if rebuilt is None:
        raise ValueError("Incomplete DroQ hourly funded account:" + label)
    inventory = attach_inventory_replay(rebuilt, fills, dataset["hourly_close"], checks=checks, label=label, metadata=metadata)
    _check_summary(row, rebuilt["metrics"], checks, label)
    parameters = [records[(profile, seed)]["parameter_sha256"] for seed in seeds]
    checks.check(metadata.get("algorithm") == "droq" and metadata.get("family") == family
                 and metadata.get("profile") == profile and metadata.get("model_count") == len(seeds),
                 label + ": true fixed DroQ model population")
    checks.check(metadata.get("parameter_sha256_before") == metadata.get("parameter_sha256_after") == parameters,
                 label + ": actual trained policy parameters remain immutable at inference")
    for name, expected in (("rebalance_band", BAND), ("cash_floor_target", .05), ("max_asset_weight_target", .95),
        ("drawdown_penalty", .25 if profile == "balanced" else 0.), ("exposure_multiplier", FAMILIES[family])):
        checks.close(metadata.get(name), expected, label + ": unchanged financial contract:" + name, atol=1e-15, rtol=0)
    checks.check(metadata.get("old_actions_or_equity_scaled") is False
                 and metadata.get("training_budget_multiple") == 1 and metadata.get("deterministic_prediction") is True
                 and all(metadata.get(flag) is True for flag in FLAGS),
                 label + ": new actual funded account and already-observed-year disclosure")
    trace = read_csv(directory / "member_targets.csv")
    audit_member_targets(trace, weights, rebuilt, fills, start=start, end=end, seeds=seeds, family=family,
                         features=features, checks=checks, label=label)
    rebuilt["costs"] = COSTS[cost]
    prior.audit_execution_quantities(dataset, trace, rebuilt, fills, start=start, end=end, band=BAND, checks=checks, label=label)
    replay = read_json(directory / "replay.json")
    checks.check(replay.get("run_id") == label and replay.get("passed") is True and replay.get("repeat_exact") is True
                 and replay.get("fingerprints") == replay.get("repeated_fingerprints") == fingerprints,
                 label + ": exact repeated financial fingerprints")
    trace_sha = sha(directory / "member_targets.csv")
    checks.check(replay.get("member_targets_sha256") == replay.get("repeated_member_targets_sha256") == trace_sha,
                 label + ": exact repeated causal action/observation trace")
    checks.check(replay.get("parameter_sha256_before") == replay.get("parameter_sha256_after") == parameters,
                 label + ": unchanged checkpoint parameters throughout exact repetition")
    rng = replay.get("rng_sha256_before", "")
    checks.check(isinstance(rng, str) and len(rng) == 64 and set(rng) <= set("0123456789abcdef")
                 and rng == replay.get("rng_sha256_after_first") == replay.get("rng_sha256_after_repeat"),
                 label + ": original deterministic actor-only inference consumes no recorded random stream")
    checks.check(replay.get("model_records") == [records[(profile, seed)] for seed in seeds],
                 label + ": exact actual trained model records, not renamed old policies")
    device = replay.get("device", {})
    checks.check(device.get("device") == "cuda:0" and device.get("capability") == [7, 0]
                 and "V100" in device.get("name", "") and device.get("visible_gpu_uuid") in {gpu["uuid"] for gpu in gpus},
                 label + ": actual isolated original V100 inference")
    days = rebuilt["equity"].equity.resample("D").last()
    dd = rebuilt["drawdown"].resample("D").last()
    history = pd.DataFrame({"timestamp": [utc(start), *list(days.index + pd.Timedelta(days=1))],
        "equity": [10000., *days.tolist()], "net_return": [0., *(days / 10000 - 1).tolist()],
        "drawdown": [0., *dd.tolist()]})
    boundary = read_equity(directory / "boundary_equity.csv")
    checks.check(boundary.index.equals(pd.DatetimeIndex(history.timestamp)), label + ": complete daily reward boundaries")
    checks.vector_close(boundary.equity, history.equity, label + ": independent boundary NAV", atol=1e-8)
    rewards, previous_nav, previous_dd = [0.], 10000., 0.
    penalty = .25 if profile == "balanced" else 0.
    for _, hours in rebuilt["equity"].groupby(rebuilt["equity"].index.normalize()):
        deepest, nav = -float(rebuilt["drawdown"].loc[hours.index].min()), float(hours.equity.iloc[-1])
        rewards.append(100 * (math.log(nav / previous_nav) - penalty * max(deepest - previous_dd, 0.)))
        previous_nav, previous_dd = nav, -float(rebuilt["drawdown"].loc[hours.index[-1]])
    checks.vector_close(boundary.reward, rewards, label + ": unchanged independent daily risk reward", atol=1e-9)
    run = {"run_id": label, "hours": len(equity), "reconstructed": True, "repeat_exact": True,
        "fingerprints": fingerprints, "member_targets_sha256": trace_sha, "inventory_replay": inventory,
        **{key: rebuilt["metrics"][key] for key in ("net_return", "max_drawdown", "fees", "final_equity")}}
    return run, history, rebuilt["monthly"]


def compare_tables(inputs, actual, expected, label):
    for frame, truth, keys in zip(actual, expected,
                                  (["run_id"], ["run_id", "timestamp"], ["run_id", "month"])):
        checks = inputs.checks
        checks.check(len(frame) == len(truth) and set(frame.columns) == set(truth.columns), label + ": exact table dimensions")
        if len(frame) != len(truth) or set(frame.columns) != set(truth.columns):
            continue
        frame, truth = frame.sort_values(keys).reset_index(drop=True), truth.sort_values(keys).reset_index(drop=True)
        checks.check(not frame[keys].duplicated().any(), label + ": unique financial row identities")
        for column in truth:
            missing = frame[column].isna() & truth[column].isna()
            if pd.api.types.is_numeric_dtype(truth[column]):
                checks.vector_close(frame.loc[~missing, column], truth.loc[~missing, column],
                                    label + ": financial table:" + column, atol=1e-8)
            else:
                checks.check((missing | frame[column].eq(truth[column])).all(), label + ": financial table:" + column)


def audit_baselines(inputs, dataset, old_spec):
    checks, folder = inputs.checks, "evaluation/test/baselines"
    original = inputs.json("main", "independent_audit.json")
    checks.check(original.get("passed") is True and original.get("status") == "passed"
                 and original.get("run_count") == 172 and original.get("test_complete") is True
                 and original.get("failures") == [], "The predeclared original172 account audit is complete")
    for name in GROUP_FILES:
        inputs.capture("main", folder + "/" + name)
    summary, daily, monthly = (inputs.csv("main", folder + "/" + name) for name in GROUP_FILES[:3])
    expected = {f"test__{cost}__{name}" for cost in COSTS for name in BASELINES}
    if len(summary) != 14 or set(summary.run_id) != expected or summary.run_id.duplicated().any():
        raise ValueError("All seven fixed original controls at both costs are mandatory")
    results = {}
    for row in summary.to_dict("records"):
        name, cost, label = row["strategy"], row["scenario"], row["run_id"]
        if not checks.check(name in BASELINES and cost in COSTS and label == f"test__{cost}__{name}",
                             label + ": original funded baseline identity"):
            continue
        relative = f"{folder}/{cost}/{name}"
        for filename in CASE_FILES[:6]:
            inputs.capture("main", relative + "/" + filename)
        run = _audit_run(safe_file(inputs.roots["main"], relative), dataset, old_spec, {}, row, monthly, daily,
            split="test", scenario=cost, algorithm="baseline", profile="baseline", key=name, checks=checks)
        previous = original.get("runs", {}).get(label, {})
        checks.check(run.get("reconstructed") is True and run.get("hours") == 8760
                     and previous.get("reconstructed") is True and previous.get("hours") == 8760
                     and previous.get("fingerprints") == run["fingerprints"], label + ": original baseline re-reconciles to passed172 evidence")
        run.update(relative_directory=relative, source_root="main", fees=float(row["fees"]), final_equity=float(row["final_equity"]))
        results[label] = run
    return summary, results


def performance_evidence(summary, baseline, candidate):
    validate_group_rows(summary, split="test", family=candidate["family"], profile=candidate["profile"])
    if len(baseline) != 14 or baseline.run_id.duplicated().any():
        raise ValueError("All fourteen predeclared baseline accounts required")
    base = summary.loc[summary.scenario.eq("base") & summary.primary_policy].iloc[0]
    stress = summary.loc[summary.scenario.eq("stress") & summary.primary_policy].iloc[0]
    members = summary.loc[summary.scenario.eq("base") & ~summary.primary_policy].sort_values("seed")
    equal = baseline.loc[baseline.run_id.eq("test__base__equal_hold")].iloc[0]
    positive = int(members.net_return.gt(0).sum())
    gates = dict(base_return_at_least_5pct=bool(base.net_return >= .05),
        base_max_drawdown_at_most_20pct=bool(abs(base.max_drawdown) <= .20),
        stress_return_nonnegative=bool(stress.net_return >= 0), at_least_three_positive_training_seeds=positive >= 3,
        beats_equal_hold_return_and_drawdown=bool(base.net_return > equal.net_return and base.max_drawdown > equal.max_drawdown))
    trend = {}
    for name in ("sma200_trend", "momentum180"):
        row = baseline.loc[baseline.run_id.eq("test__base__" + name)].iloc[0]
        trend[name] = bool(base.net_return > row.net_return and base.max_drawdown > row.max_drawdown)
    return {"candidate": {key: candidate[key] for key in ("algorithm", "family", "profile")},
        "gates": gates, "goal_performance_achieved": all(gates.values()), "all_five_diagnostic_targets_met": all(gates.values()),
        "base_net_return": float(base.net_return), "base_max_drawdown": float(base.max_drawdown),
        "stress_net_return": float(stress.net_return), "stress_max_drawdown": float(stress.max_drawdown),
        "positive_base_training_seeds": positive,
        "base_seed_returns": [{"seed": int(row.seed), "net_return": float(row.net_return)} for row in members.itertuples()],
        "base_fees_usdt": float(base.fees), "stress_fees_usdt": float(stress.fees),
        "beats_simple_trend_controls": trend, "stronger_control_comparison_met": all(trend.values()),
        "fresh_holdout_success": False, "training_budget_multiple": 1, "recipe_is_pure_official_default": False,
        "accounting_passage_is_not_goal_success": True, **{flag: True for flag in FLAGS}}


def audit_group(inputs, dataset, records, spec, *, split, family, features):
    folder, checks = f"{split}/{family}/balanced", inputs.checks
    for name in GROUP_FILES:
        inputs.capture("study", folder + "/" + name)
    summary, daily, monthly = (inputs.csv("study", folder + "/" + name) for name in GROUP_FILES[:3])
    validate_group_rows(summary, split=split, family=family, profile="balanced")
    completion = inputs.json("study", folder + "/completion.json")
    checks.check(completion.get("case_count") == completion.get("repeat_count") == 12
                 and completion.get("protocol_sha256") == sha(inputs.roots["study"] / "protocol.json"), folder + ": twelve cases and exact repeats")
    proofs = completion.get("repeats", [])
    by_id = {v.get("run_id"): v for v in proofs}
    checks.check(len(proofs) == len(by_id) == 12 and set(by_id) == set(summary.run_id), folder + ": all case receipts exactly once")
    device, expected_uuid = completion.get("device", {}), spec["family_gpu_assignments"][family]
    checks.check(device.get("visible_gpu_uuid") == expected_uuid, folder + ": actual assigned budget worker card")
    expected = {}
    for row in summary.to_dict("records"):
        key = "ensemble" if row["primary_policy"] else str(int(row["seed"]))
        relative = folder + "/" + row["scenario"] + "/" + key
        for name in CASE_FILES:
            inputs.capture("study", relative + "/" + name)
        run, history, months = audit_case(safe_file(inputs.roots["study"], relative), dataset, row, records, checks,
            split=split, family=family, profile="balanced", features=features,
            gpus=[v for v in spec["physical_gpus"] if v["uuid"] == expected_uuid])
        proof = inputs.json("study", relative + "/replay.json")
        checks.check(by_id.get(row["run_id"]) == proof and proof.get("worker_pid") == completion.get("worker_pid")
                     and proof.get("device") == device, row["run_id"] + ": actual worker and repeat receipt agree")
        run.update(relative_directory=relative, family=family, profile="balanced", scenario=row["scenario"])
        expected[row["run_id"]] = (run, history, months)
    compare_group_tables(summary, daily, monthly, expected, checks, folder)
    return summary, daily, monthly, expected


def verify_workers(inputs, spec, completion, *, split, chosen=None):
    checks = inputs.checks
    workers = completion.get("workers", []) if split == "validation" else [completion.get("worker", {})]
    families = set(FAMILIES) if split == "validation" else {chosen["family"]}
    checks.check(len(workers) == len(families) and {v.get("family") for v in workers} == families
                 and len({v.get("worker_pid") for v in workers}) == len(families), split + ": real one-worker-per-budget handles")
    for worker in workers:
        family = worker.get("family")
        if family not in families:
            continue
        checks.check(worker.get("split") == split and worker.get("profile") == "balanced"
                     and worker.get("device", {}).get("visible_gpu_uuid") == spec["family_gpu_assignments"][family]
                     and type(worker.get("worker_pid")) is int and worker["worker_pid"] > 0, split + ": assigned actual inference process")
        group = inputs.json("study", f"{split}/{family}/balanced/completion.json")
        checks.check(group.get("worker_pid") == worker.get("worker_pid") and group.get("device") == worker.get("device"), split + ": group and completed coordinator agree")


def verify_selection(inputs):
    checks = inputs.checks
    selected, receipt = inputs.json("study", "selection.json"), inputs.json("study", "selection_receipt.json")
    previous = inputs.json("study", "independent_audit.validation.json")
    checks.check(previous.get("passed") is True and previous.get("status") == "partial_test_pending"
                 and previous.get("test_complete") is False and previous.get("training_model_count") == 5
                 and previous.get("run_count") == previous.get("repeat_count") == previous.get("validation_run_count") == 24
                 and previous.get("failure_count") == 0 and previous.get("failures") == [], "Complete independent24-account validation before selection")
    inputs.pin("study", previous.get("input_sha256"), validation_files(), "Original independent24 validation evidence")
    inputs.pin("main", previous.get("main_input_sha256"), {"frozen_spec.json", "runtime.json"}, "Original validation source context")
    inputs.pin("source", previous.get("source_sha256"), SOURCE_FILES, "Original reviewed auditor/runner")
    checks.check(selected.get("audit_sha256") == sha(inputs.roots["study"] / "independent_audit.validation.json")
                 and receipt.get("selection_file") == "selection.json"
                 and receipt.get("selection_sha256") == sha(inputs.roots["study"] / "selection.json")
                 and receipt.get("protocol_sha256") == selected.get("protocol_sha256") == sha(inputs.roots["study"] / "protocol.json"), "Exact whole-file selected-candidate seal")
    checks.check(selected.get("seed_selection") is False and selected.get("selection_uses_test") is False
                 and selected.get("test_results_available_to_selector") is False
                 and all(selected.get(flag) is True for flag in FLAGS), "No seed or annual selection; known-year disclosure")
    inputs.pin("study", selected.get("training_source_sha256"), training_files(), "All five checkpoints, manifests and timeout traces")
    inputs.pin("study", selected.get("validation_source_sha256"), validation_files(), "Both original budget candidates and exact repeats")
    groups = {(family, "balanced"): inputs.csv("study", f"validation/{family}/balanced/summary.csv") for family in FAMILIES}
    expected = {(v["family"], v["profile"]): v for v in rank_candidates(groups)}
    actual = selected.get("candidates", [])
    checks.check(len(actual) == 2 and {(v.get("family"), v.get("profile")) for v in actual} == set(expected), "Exactly two predeclared budget candidates")
    for row in actual:
        truth = expected.get((row.get("family"), row.get("profile")))
        if truth is None:
            continue
        checks.check(row.get("algorithm") == "droq" and row.get("rebalance_band") == BAND
                     and row.get("exposure_multiplier") == truth["exposure_multiplier"], "No post-validation algorithm or band search")
        checks.close(row.get("score"), truth["score"], "Independent fixed validation score", atol=1e-12)
        for cost in COSTS:
            checks.check(row.get(cost, {}).get("run_id") == truth[cost]["run_id"], "Original actual validation ensemble ID")
            for key in ("net_return", "max_drawdown", "final_equity", "fees"):
                checks.close(row.get(cost, {}).get(key), truth[cost][key], "Original validation metric:" + key, atol=1e-10)
    order = sorted(actual, key=lambda v: (-v["score"], 0 if v["family"] == "raw" else 1, v["profile"]))
    checks.check(actual == order and selected.get("candidate") == (order[0] if order else None), "Validated sealed score bytes and raw-first ties")
    checks.check(utc(previous["audited_at"]) <= utc(selected["selected_at"]) <= utc(receipt["sealed_at"]), "Independent validation precedes the sealed choice")
    return selected["candidate"], selected, receipt


def verify_annual_authorization(inputs, receipt):
    checks = inputs.checks
    auth, marker = inputs.json("study", "annual_authorization.json"), inputs.json("study", "test_started.json")
    checks.check(auth.get("annual_evaluation_authorized") is True and auth.get("no_live_trading") is True
                 and auth.get("financial_parameters_fixed") is True
                 and auth.get("selection_sha256") == sha(inputs.roots["study"] / "selection.json")
                 and all(auth.get(flag) is True for flag in FLAGS), "Explicit separate annual authorization for this new selected study")
    checks.check(marker.get("protocol_sha256") == sha(inputs.roots["study"] / "protocol.json")
                 and marker.get("selection_sha256") == sha(inputs.roots["study"] / "selection.json")
                 and marker.get("authorization_sha256") == sha(inputs.roots["study"] / "annual_authorization.json")
                 and utc(receipt["sealed_at"]) <= utc(auth["authorized_at"]) <= utc(marker["at"]), "Annual marker follows the actual authorization and prior validation seal")
    return auth, marker


def audit(output, main, dataset_path, *, phase="validation", source_root=ROOT):
    checks, started = Checks(), prior.now()
    inputs = Inputs(output, main, source_root, checks)
    records, runs, baseline_runs, research, training = {}, {}, {}, None, None
    validation_count = 0
    try:
        if phase not in ("validation", "test"):
            raise ValueError("Only validation or separately authorized annual audit phases")
        if phase == "validation" and any((inputs.roots["study"] / name).exists() for name in
                ("test", "test_started.json", "test_completion.json", "test_process.json", "annual_authorization.json")):
            raise ValueError("The original validation-only audit cannot inspect later annual evidence")
        if phase == "test" and not all((inputs.roots["study"] / name).is_file() for name in
                ("selection.json", "selection_receipt.json", "annual_authorization.json", "test_started.json")):
            raise ValueError("An annual audit requires the prior sealed choice and explicit separate authorization")
        spec, old = verify_protocol(inputs, dataset_path)
        fit, fit_complete = verify_phase(inputs, "fit", spec)
        validate, validate_complete = verify_phase(inputs, "validate", spec)
        checks.check(fit_complete.get("model_count") == 5 and validate_complete.get("case_count") == validate_complete.get("repeat_count") == 24
                     and utc(fit["completed_at"]) <= utc(validate["started_at"]), "All five models complete before24 actual validation accounts")
        if phase == "test":
            chosen, _, seal = verify_selection(inputs)
            _, marker = verify_annual_authorization(inputs, seal)
            test, complete = verify_phase(inputs, "test", spec)
            checks.check(complete.get("case_count") == complete.get("repeat_count") == 12 and complete.get("candidate") == chosen
                         and utc(marker["at"]) <= utc(test["started_at"]), "Only the authorized sealed winner has twelve complete repeated annual accounts")
        if checks.failed:
            raise ValueError("Source, completed phase, selection or authorization failed before prices or saved tensors")
        cutoff = VALIDATION[1] if phase == "validation" else TEST[1]
        dataset = load_market(dataset_path, cutoff)
        features = prior.independent_market_features(dataset, cutoff=cutoff)
        dates = VALIDATION if phase == "validation" else TEST
        expected_hours = pd.date_range(utc(dates[0]), utc(dates[1]), freq="h", inclusive="left")
        for field in ("open", "high", "low", "close", "volume"):
            values = dataset["hourly_" + field].reindex(expected_hours).to_numpy(dtype=float)
            checks.check(np.isfinite(values).all() and (values >= 0 if field == "volume" else values > 0).all(), "Complete original two-coin hourly field:" + field)
        records, training = audit_training(inputs, spec, dataset, features)
        if phase == "validation":
            groups, assembled = {}, []
            for family in FAMILIES:
                group = audit_group(inputs, dataset, records, spec, split="validation", family=family, features=features)
                groups[(family, "balanced")] = group[0]; assembled.append(group)
                runs.update({k: v[0] for k, v in group[3].items()})
            actual = [inputs.csv("study", "validation/" + name) for name in GROUP_FILES[:3]]
            expected = [pd.concat([group[i] for group in assembled], ignore_index=True) for i in range(3)]
            compare_tables(inputs, actual, expected, "All24 validation parent tables")
            complete = inputs.json("study", "validation_completion.json")
            checks.check(complete.get("case_count") == complete.get("repeat_count") == 24
                         and complete.get("protocol_sha256") == sha(inputs.roots["study"] / "protocol.json"), "Both fixed budget candidates completed")
            verify_workers(inputs, spec, complete, split="validation")
            for name in validation_files():
                inputs.capture("study", name)
            rank_candidates(groups)
            validation_count = len(runs)
        else:
            group = audit_group(inputs, dataset, records, spec, split="test", family=chosen["family"], features=features)
            verify_workers(inputs, spec, complete, split="test", chosen=chosen)
            runs = {k: v[0] for k, v in group[3].items()}
            baseline, baseline_runs = audit_baselines(inputs, dataset, old)
            validation_count = 24
            if checks.failed == 0:
                research = performance_evidence(group[0], baseline, chosen)
        inputs.unchanged()
    except Exception as error:
        checks.check(False, type(error).__name__ + ": " + str(error))
    expected = (24, 0) if phase == "validation" else (12, 14)
    passed = checks.failed == 0 and len(records) == 5 and (len(runs), len(baseline_runs)) == expected
    return dict(schema_version=1, started_at=started, audited_at=prior.now(), phase=phase, passed=passed,
        status=("partial_test_pending" if phase == "validation" else "passed") if passed else "failed",
        validation_complete=passed if phase == "validation" else validation_count == 24, test_complete=passed and phase == "test",
        training_model_count=len(records), run_count=len(runs) + len(baseline_runs), validation_run_count=validation_count,
        test_run_count=len(runs) if phase == "test" else 0, baseline_run_count=len(baseline_runs), repeat_count=len(runs) if passed else 0,
        checks_total=checks.count, failure_count=checks.failed, failures=checks.failures, maximum_errors=checks.maximum_errors,
        input_sha256=inputs.hashes["study"], main_input_sha256=inputs.hashes["main"], source_sha256=inputs.hashes["source"],
        dataset_sha256=sha(dataset_path) if Path(dataset_path).is_file() else None, training_calendar=training, runs=runs, baseline_runs=baseline_runs,
        declared_research_targets=TARGETS, research_targets=research, fresh_holdout_success=False, training_budget_multiple=1,
        recipe_is_pure_official_default=False, accounting_passage_is_not_goal_success=True, **{flag: True for flag in FLAGS})


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("output-dir", "reference-main", "dataset"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--phase", choices=("validation", "test"), default="validation")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = args.report or args.output_dir / ("independent_audit.validation.json" if args.phase == "validation" else "independent_audit.json")
    if report.exists():
        raise FileExistsError("Preserve existing audit bytes and choose another --report")
    result = audit(args.output_dir, args.reference_main, args.dataset, phase=args.phase)
    report.parent.mkdir(parents=True, exist_ok=True)
    with report.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False); stream.write("\n")
    print(json.dumps({key: result[key] for key in ("passed", "status", "training_model_count", "validation_run_count", "test_run_count", "baseline_run_count", "checks_total", "failure_count")}))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()

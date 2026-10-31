"""Independent TD3 research audit, with no execution engine or policy inference.

Training uses the previous fixed financial environment. This module validates
the newly declared recipe, installed official sources, timeout replay evidence
and funded accounts. It never mutates a frozen algorithm registry or a prior
research artifact. The previously inspected year is not a fresh holdout.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
from pathlib import Path
import sys
import zipfile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import audit_btceth_rl_execution_band as prior
from scripts.audit_btceth_rl import BASELINES, Checks, _audit_run, _check_summary, independent_training_calendar, read_csv, read_equity, reconstruct_fills, verify_saved_fingerprints
from scripts.audit_btceth_rl_exposure_cap import CASE_FILES, COSTS, GROUP_FILES, SEEDS, SYMBOLS, attach_inventory_replay, candidate_score, compare_group_tables, load_market, mapped_target, read_json, safe_file, sha, utc

TRAIN = ("2020-01-01T00:00:00Z", "2025-01-01T00:00:00Z")
VALIDATION = ("2025-01-01T00:00:00Z", "2025-10-23T00:00:00Z")
TEST = ("2025-10-23T00:00:00Z", "2026-10-23T00:00:00Z")
PROFILES, FAMILIES = ("return", "balanced"), {"raw": 1., "cap075": .75}
FLAGS = ("designed_after_first_annual", "test_year_already_seen", "not_fresh_holdout")
STEPS, EPISODE_DAYS, BAND = 65536, 32, .025
SB3_COMMIT = "8908708f10c8ff29759c67f55c8acb56cab27463"
UPSTREAM_SHA256 = {
    "stable_baselines3/td3/td3.py": "45107ef95f85cca1224741b666a3de2afa57838a501cc301331fd7e9a41cc2e0",
    "stable_baselines3/td3/policies.py": "5b81fa2347e9e3ce85956b570cb3008f58778d62ad1598c3bc89217c3a143062",
    "stable_baselines3/common/off_policy_algorithm.py": "ec895fa4b7ce96550b0acdfea3a73561db568f863b24c0b77d106f3f67e540a7",
    "stable_baselines3/common/buffers.py": "21db27f000048615f5848facf18af9afc22e6e2a1c26148911c2cba542d1988d",
    "stable_baselines3/common/noise.py": "335d9e322c8b5bf1b43703ac284951a01bb7980822cb30c8c2b94b0f48dc4e1f",
}
LICENSE_SHA256 = "2d8fbeee75e0acd13e75b30470b3521b12fb55c26c90c2060c6bd525345870a9"
SETUP_SHA256 = "7c407d274eddac9884e3beb4ad7718a14cbb8a792940903384894ee9ea8d1e7e"
VERIFIED_MANIFEST_SHA256 = "7ea8c1248651934265a4fe087b1979c231f6b0ab51d2180c2c9b3a55121853d9"
RECIPE = dict(policy="MlpPolicy", learning_rate=.0003, gamma=.995, buffer_size=100000,
    learning_starts=512, batch_size=128, tau=.005, train_freq=[1, "step"], gradient_steps=1,
    n_steps=1, policy_delay=2, target_policy_noise=.2, target_noise_clip=.5,
    optimize_memory_usage=False, replay_buffer_kwargs={"handle_timeout_termination": True},
    policy_kwargs={"net_arch": [128, 128], "activation_fn": "ReLU",
                   "optimizer_class": "Adam", "n_critics": 2, "share_features_extractor": False},
    action_noise={"class_name": "NormalActionNoise", "mean": [0., 0.], "sigma": [.1, .1], "dtype": "float32",
                  "units": "normalized Box[-1,1] action units, not portfolio percentage points"})
TARGETS = {"primary_base_net_return_at_least": .05, "primary_base_max_drawdown_magnitude_at_most": .20,
    "primary_stress_net_return_at_least": 0., "primary_beats_equal_hold_return_and_drawdown": True,
    "positive_base_training_seeds_at_least": 3, "stronger_claim_requires_beating_simple_trend_controls": True,
    "targets_are_not_guarantees": True}
SOURCE_FILES = (*prior.FINANCIAL, "scripts/audit_btceth_rl.py", "scripts/audit_btceth_rl_exposure_cap.py",
                "scripts/audit_btceth_rl_execution_band.py", "scripts/btceth_rl_td3.py", "scripts/audit_btceth_rl_td3.py")


class Inputs:
    def __init__(self, output, main, source, checks):
        self.roots = {"study": Path(output).resolve(), "main": Path(main).resolve(), "source": Path(source).resolve()}
        if self.roots["study"] == self.roots["main"]:
            raise ValueError("New TD3 research cannot overwrite the original study")
        self.hashes = {key: {} for key in self.roots}
        self.checks = checks

    def capture(self, root, name):
        path = safe_file(self.roots[root], name)
        digest, key = sha(path), str(name).replace("\\", "/")
        old = self.hashes[root].get(key)
        if old is not None:
            self.checks.check(old == digest, "Input changed between reads:" + root + ":" + key)
        self.hashes[root][key] = digest
        return path

    def json(self, root, name):
        return read_json(self.capture(root, name))

    def csv(self, root, name):
        return read_csv(self.capture(root, name))

    def pin(self, root, values, required, label):
        if not self.checks.check(isinstance(values, dict) and set(required).issubset(values), label + ": complete immutable inputs"):
            raise ValueError("Missing required immutable inputs:" + label)
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


def validation_files():
    return {"validation_completion.json", "validate_completion.json", "validate_process.json", "validate.log",
        *(f"validation/{name}" for name in GROUP_FILES[:3]),
        *(f"validation/{family}/{profile}/{name}" for family in FAMILIES for profile in PROFILES for name in GROUP_FILES),
        *(f"validation/{family}/{profile}/{cost}/{key}/{name}" for family in FAMILIES for profile in PROFILES
          for cost in COSTS for key in (*map(str, SEEDS), "ensemble") for name in CASE_FILES)}


def verify_protocol(inputs, dataset_path):
    checks = inputs.checks
    spec, receipt = inputs.json("study", "protocol.json"), inputs.json("study", "protocol_receipt.json")
    old, runtime = inputs.json("main", "frozen_spec.json"), inputs.json("main", "runtime.json")
    fixed = dict(algorithm="td3", profiles={"return": {"drawdown_penalty": 0., "rebalance_band": BAND},
        "balanced": {"drawdown_penalty": .25, "rebalance_band": BAND}}, seeds=list(SEEDS), budgets=FAMILIES,
        train=list(TRAIN), validation=list(VALIDATION), test=list(TEST), capital=10000., costs=COSTS,
        timesteps_per_model=STEPS, old_timesteps_per_model=32768, training_budget_multiple=2, train_episode_days=32,
        model_count=10, validation_cases=48, validation_repeats=48, test_cases=12, test_repeats=12,
        cash_floor=.05, max_asset_weight=.95, hyperparameters=RECIPE, recipe_is_pure_official_default=False,
        seed_selection=False, selection_uses_test=False, new_recipe_search=False, annual_requires_separate_authorization=True,
        selection_score="mean(log1p(net_return)-0.5*abs(max_drawdown)-2*max(abs(max_drawdown)-0.20,0)) across base/stress",
        tie_break="raw before cap075, then profile lexical identifier", research_targets=TARGETS)
    for name, expected in fixed.items():
        checks.check(spec.get(name) == expected, "One finite declared TD3 experiment:" + name)
    checks.check(all(spec.get(flag) is True for flag in FLAGS), "Already-seen-year limitations retained")
    checks.check(receipt.get("protocol_file") == "protocol.json"
                 and receipt.get("protocol_sha256") == sha(inputs.roots["study"] / "protocol.json")
                 and utc(spec["prepared_at"]) <= utc(receipt["sealed_at"]), "Own whole-file declaration seal before training")
    checks.check(spec.get("original_frozen_spec_sha256") == sha(inputs.roots["main"] / "frozen_spec.json")
                 and old.get("timesteps_per_model") == 32768
                 and tuple(old["train"]) == TRAIN and tuple(old["validation"]) == VALIDATION and tuple(old["test"]) == TEST
                 and old.get("research_targets") == TARGETS, "Original split, target thresholds and disclosed twofold budget")
    sources = spec.get("source_sha256", {})
    inputs.pin("source", sources, SOURCE_FILES, "Financial/audit/TD3 source bytes")
    checks.check(set(sources) == set(SOURCE_FILES)
                 and all(sources[name] == old["code_sha256"][name] for name in prior.FINANCIAL),
                 "Exact new source DAG; old four financial files untouched")
    for name, expected in sources.items():
        checks.check(sha(inputs.capture("study", "sources/" + name)) == expected, "Archived exact source:" + name)
    inputs.pin("main", spec.get("original_input_sha256"),
               baseline_files() | {"frozen_spec.json", "runtime.json", "independent_audit.json"}, "Original funded controls fixed before TD3")
    upstream_root = inputs.roots["study"] / "upstream"
    verify_upstream(upstream_root, checks)
    expected_files = {*UPSTREAM_SHA256, "LICENSE", "setup.py", "verified_manifest.json",
                      "github_manifest.json", "installed_distribution_observation.json"}
    upstream = spec.get("upstream_evidence_sha256", {})
    checks.check(set(upstream) == expected_files, "All seven official files and three provenance manifests preserved")
    for name, expected in upstream.items():
        checks.check(sha(inputs.capture("study", "upstream/" + name)) == expected, "Declared official source artifact:" + name)
    checks.check(spec.get("sb3_source_sha256") == UPSTREAM_SHA256
                 and spec.get("upstream", {}).get("commit") == SB3_COMMIT
                 and spec.get("upstream", {}).get("version") == "2.9.0"
                 and spec.get("upstream", {}).get("verified_manifest_sha256") == VERIFIED_MANIFEST_SHA256,
                 "The actual installed five-file TD3 implementation matches the reviewed official source")
    dataset_path = Path(dataset_path)
    checks.check(sha(dataset_path) == spec.get("dataset_sha256") == old["dataset_sha256"]
                 and sha(dataset_path.parent / "manifest.json") == spec.get("dataset_manifest_sha256")
                 == old["dataset_manifest_sha256"], "Original source market file/manifest without data conversion")
    gpus = spec.get("physical_gpus", [])
    checks.check(gpus == runtime.get("physical_gpus") and len(gpus) == 2
                 and len({gpu["uuid"] for gpu in gpus}) == 2 and all("V100" in gpu["name"] for gpu in gpus)
                 and spec.get("profile_gpu_assignments") == {profile: gpu["uuid"] for profile, gpu in zip(PROFILES, gpus)},
                 "Two original physical V100s, one fixed profile per card")
    checks.check(spec.get("runtime_packages") == old["runtime_packages"], "Same frozen research dependencies, no installation or fallback")
    if spec.get("resource_allowance_sha256") is not None:
        allowance = inputs.json("study", "resource_allowance.json")
        checks.check(sha(inputs.roots["study"] / "resource_allowance.json") == spec["resource_allowance_sha256"]
                     and allowance.get("allowed_instance") == spec.get("allowed_compute_instance")
                     and allowance.get("allowed_gpu_rows") == spec.get("allowed_gpu_rows"),
                     "Only the independently reviewed process-instance resource exception retained")
    if checks.failed:
        raise ValueError("Protocol/source/upstream/calendar context failed before any model or financial result read")
    return spec, old


def verify_phase(inputs, phase, spec):
    checks = inputs.checks
    process = inputs.json("study", phase + "_process.json")
    complete = inputs.json("study", phase + "_completion.json")
    preflight = inputs.json("study", "preflight/" + phase + ".json")
    receipt = inputs.json("study", "protocol_receipt.json")
    command = process.get("command", [])
    checks.check(process.get("phase") == phase and process.get("status") == "completed" and process.get("exit_code") == 0
                 and process.get("protocol_sha256") == sha(inputs.roots["study"] / "protocol.json")
                 and process.get("source_sha256") == spec["source_sha256"]["scripts/btceth_rl_td3.py"],
                 phase + ": real terminal0 supervisor used the declared TD3 source")
    checks.check(len(command) >= 5 and str(command[2]).replace("\\", "/").endswith("/btceth_rl_td3.py")
                 and "--phase" in command and command[command.index("--phase") + 1] == "_" + phase,
                 phase + ": actual matching private child command")
    checks.check(isinstance(process.get("supervisor_pid"), int) and process["supervisor_pid"] > 0
                 and isinstance(process.get("child_pid"), int) and process["child_pid"] > 0
                 and process["supervisor_pid"] != process["child_pid"], phase + ": distinct actual process handles")
    checks.check(process.get("log_relative_path") == phase + ".log"
                 and process.get("log_sha256") == sha(inputs.capture("study", phase + ".log")),
                 phase + ": immutable true terminal log")
    checks.check(complete.get("protocol_sha256") == sha(inputs.roots["study"] / "protocol.json")
                 and utc(receipt["sealed_at"]) <= utc(preflight["at"]) <= utc(process["started_at"])
                 <= utc(complete["at"]) <= utc(process["completed_at"]), phase + ": sealed declaration precedes real work and completion")
    checks.check(preflight.get("physical_gpus") == spec["physical_gpus"]
                 and preflight.get("command") == ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"],
                 phase + ": original two-card resource preflight was recorded unfiltered")
    rows = []
    for line in preflight.get("unfiltered_stdout", "").splitlines():
        if line.strip():
            uuid, pid = [value.strip() for value in line.split(",")]
            rows.append({"gpu_uuid": uuid, "pid": int(pid)})
    checks.check(not rows or rows == spec.get("allowed_gpu_rows"), phase + ": no undeclared compute process suppressed")
    if rows:
        checks.check(preflight.get("allowed_compute_instance") == spec.get("allowed_compute_instance"),
                     phase + ": exact reviewed UI process identity and start time retained")
    return process, complete


def audit_training(inputs, spec, dataset, features):
    checks = inputs.checks
    fit = inputs.json("study", "fit_process.json")
    completion = inputs.json("study", "training_completion.json")
    checks.check(completion.get("model_count") == 10 and len(completion.get("records", [])) == 10
                 and completion.get("protocol_sha256") == sha(inputs.roots["study"] / "protocol.json"),
                 "Exactly ten new completed TD3 models, not relabeled old weights")
    completed = {(row["profile"], row["seed"]): row for row in completion.get("records", [])}
    calendar = independent_training_calendar(dataset, *TRAIN, EPISODE_DAYS)
    records = {}
    effective = {key: value for key, value in RECIPE.items() if key != "policy"}
    effective["action_noise"] = {key: value for key, value in RECIPE["action_noise"].items() if key != "units"}
    for profile in PROFILES:
        for seed in SEEDS:
            folder = f"training/td3__{profile}__seed{seed}"
            record, started = inputs.json("study", folder + "/record.json"), inputs.json("study", folder + "/started.json")
            label = f"TD3 {profile} seed{seed}"
            checks.check((record.get("algorithm"), record.get("profile"), record.get("seed")) == ("td3", profile, seed)
                         and completed.get((profile, seed)) == record
                         and record.get("protocol_sha256") == started.get("protocol_sha256")
                         == sha(inputs.roots["study"] / "protocol.json"), label + ": actual new job identity")
            task = started.get("task", {})
            checks.check(task.get("profile") == profile and task.get("seed") == seed
                         and utc(fit["started_at"]) <= utc(started["at"]) <= utc(record["completed_at"])
                         <= utc(fit["completed_at"]), label + ": actual training job lies inside its own sealed fit attempt")
            checks.check(record.get("actual_timesteps") == STEPS and record.get("actual_n_updates") == 65024
                         and record.get("actor_update_count") == 32512 and record.get("hyperparameters") == RECIPE
                         and record.get("effective_hyperparameters") == effective
                         and record.get("training_budget_multiple") == 2 and record.get("old_timesteps_per_model") == 32768
                         and record.get("recipe_is_pure_official_default") is False,
                         label + ": exact actual learned budget/attributes and disclosed shared recipe")
            checks.check(record.get("algorithm_class") == "stable_baselines3.td3.td3.TD3"
                         and record.get("policy_class") == "TD3Policy" and record.get("training_buffer_class") == "AuditedReplayBuffer"
                         and record.get("sb3_source_sha256") == spec["sb3_source_sha256"] == UPSTREAM_SHA256,
                         label + ": actual official TD3 policy and purely observing replay subclass")
            semantics = record.get("replay_buffer_semantics", {})
            checks.check(semantics.get("optimize_memory_usage") is False and semantics.get("handle_timeout_termination") is True
                         and semantics.get("observations_captured_after_super_add") is True,
                         label + ": original timeout bootstrap semantics and untouched stored transition")
            device = record.get("device", {})
            checks.check(device.get("device") == "cuda:0" and device.get("capability") == [7, 0]
                         and "V100" in device.get("name", "")
                         and device.get("visible_gpu_uuid") == spec["profile_gpu_assignments"][profile]
                         and started.get("device") == device, label + ": actual assigned profile-specific V100")
            expected_model, expected_trace = folder + "/model.zip", folder + "/training_timeout_transitions.jsonl"
            checks.check(record.get("model_relative_path") == expected_model
                         and record.get("timeout_trace_relative_path") == expected_trace, label + ": unique owned model/trace paths")
            model = inputs.capture("study", expected_model)
            trace = inputs.capture("study", expected_trace)
            checks.check(sha(model) == record.get("model_sha256") and sha(trace) == record.get("timeout_trace_sha256"),
                         label + ": original checkpoint and complete timeout trace bytes")
            checks.check(checkpoint_parameter_digest(model) == record.get("parameter_sha256"),
                         label + ": independently hashed actual saved policy tensors")
            saved = checkpoint_training_metadata(model)
            for key, expected in {"num_timesteps": STEPS, "_n_updates": 65024,
                **{key: RECIPE[key] for key in ("policy_delay", "learning_rate", "learning_starts", "batch_size", "buffer_size",
                    "gradient_steps", "n_steps", "gamma", "tau", "target_policy_noise", "target_noise_clip")}}.items():
                checks.close(saved[key], expected, label + ": actual ZIP training metadata:" + key, atol=1e-15, rtol=0)
            total = 0
            for episode in record.get("training_episode_history", []):
                first, end, until = (utc(episode[key]) for key in ("start", "end_exclusive", "completed_until"))
                steps = int(episode["completed_steps"])
                checks.check(first in calendar["eligible"] and end - first == pd.Timedelta(days=32)
                             and utc(TRAIN[0]) <= first <= until <= end <= utc(TRAIN[1])
                             and until - first == pd.Timedelta(days=steps), label + ": complete causal32-day training fragment")
                checks.check(episode.get("terminated") is False and episode.get("truncated") == (steps == 32),
                             label + ": every market fragment is time-truncated, including final unused reset")
                total += steps
            checks.check(total == STEPS, label + ": no early stop or missing training transitions")
            rows = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines() if line.strip()]
            verify_replay_boundaries(rows, record["training_episode_history"], features, checks, label)
            quality = record.get("training_window_quality", {})
            for key in ("requested_train_days", "candidate_windows", "eligible_windows", "excluded_windows"):
                checks.check(quality.get(key) == calendar[key], label + ": independently reconstructed train calendar:" + key)
            checks.check(quality.get("covered_training_days") == calendar["covered_days"]
                         and len(record.get("features", [])) == len(set(record.get("features", []))) == 27,
                         label + ": original market coverage and fixed27 causal states")
            records[(profile, seed)] = record
    checks.check(set(records) == set(completed) == {(profile, seed) for profile in PROFILES for seed in SEEDS},
                 "All ten distinct profile/seed models retained")
    return records, {key: value for key, value in calendar.items() if key not in ("eligible", "market", "features")}


def observation_digest(values):
    return hashlib.sha256(np.asarray(values, dtype="<f4").tobytes()).hexdigest()


def checkpoint_parameter_digest(path):
    """Hash tensors independently without loading a policy class or inference."""
    import torch
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if names.count("policy.pth") != 1 or names.count("_stable_baselines3_version") != 1:
            raise ValueError("Exactly one official policy state and SB3 version record required")
        if archive.read("_stable_baselines3_version").decode().strip() != "2.9.0":
            raise ValueError("Only the pinned official SB3 version is permitted")
        state = torch.load(io.BytesIO(archive.read("policy.pth")), map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or not state or not all(isinstance(key, str) and isinstance(value, torch.Tensor)
                                                         for key, value in state.items()):
        raise ValueError("Only a tensor state dictionary is audited, never an executable policy object")
    if not any(key.startswith("actor_target.") for key in state) or not any(key.startswith("critic_target.") for key in state):
        raise ValueError("TD3 requires both actor-target and critic-target tensors; old TQC weights cannot be relabeled")
    digest = hashlib.sha256()
    for key, tensor in sorted(state.items()):
        values = tensor.detach().cpu().numpy()
        if not np.isfinite(values).all():
            raise ValueError("Every actual saved policy tensor must be finite")
        digest.update(key.encode())
        digest.update(str((values.shape, values.dtype)).encode())
        digest.update(values.tobytes())
    return digest.hexdigest()


def checkpoint_training_metadata(path):
    """Read plain JSON training counters; never decode cloudpickle attributes."""
    with zipfile.ZipFile(path) as archive:
        if archive.namelist().count("data") != 1:
            raise ValueError("Exactly one official plain JSON model-data member required")
        data = json.loads(archive.read("data"))
    names = ("num_timesteps", "_n_updates", "policy_delay", "learning_rate", "learning_starts", "batch_size",
             "buffer_size", "gradient_steps", "n_steps", "gamma", "tau", "target_policy_noise", "target_noise_clip")
    result = {}
    for key in names:
        value = data.get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError("Plain finite official model metadata missing:" + key)
        result[key] = value
    return result


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
        raise ValueError("Only predeclared TD3 family/profile/split identities are valid")
    required = {"run_id", "algorithm", "family", "profile", "scenario", "split", "seed", "primary_policy",
        "initial_capital", "start", "end_exclusive", "net_return", "max_drawdown", "final_equity", "fees",
        "rebalance_band", "exposure_multiplier"}
    if not required.issubset(frame) or len(frame) != 12 or frame.run_id.duplicated().any():
        raise ValueError("Each group requires exactly five seeds plus one real ensemble at both costs")
    fixed = dict(algorithm="td3", family=family, profile=profile, split=split,
                 initial_capital=10000., rebalance_band=BAND, exposure_multiplier=FAMILIES[family])
    if any(not frame[key].eq(value).all() for key, value in fixed.items()):
        raise ValueError("The TD3 financial identity/recipe cannot change between candidates")
    if (not pd.to_datetime(frame.start, utc=True).eq(utc(dates[0])).all()
            or not pd.to_datetime(frame.end_exclusive, utc=True).eq(utc(dates[1])).all()):
        raise ValueError("Evaluation must use the complete original fixed window")
    expected = {f"{split}__{cost}__{family}__td3__{profile}__{key}" for cost in COSTS
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
        raise ValueError("Exactly four declared validation candidates are required")
    candidates = []
    for (family, profile), frame in groups.items():
        validate_group_rows(frame, split="validation", family=family, profile=profile)
        scenarios = {cost: frame.loc[frame.scenario.eq(cost) & frame.primary_policy].iloc[0] for cost in COSTS}
        candidates.append(dict(family=family, algorithm="td3", profile=profile,
            exposure_multiplier=FAMILIES[family], rebalance_band=BAND, score=candidate_score(scenarios["base"], scenarios["stress"]),
            **{cost: {"run_id": str(row.run_id), **{key: float(row[key]) for key in
                ("net_return", "max_drawdown", "final_equity", "fees")}} for cost, row in scenarios.items()}))
    return sorted(candidates, key=lambda row: (-row["score"], 0 if row["family"] == "raw" else 1, row["profile"]))


def verify_replay_boundaries(rows, episodes, features, checks, label):
    """Check actual stored transitions and true timeout observations, not reset states."""
    history = [episode for episode in episodes if int(episode["completed_steps"]) > 0]
    checks.check(len(rows) == len(history) == STEPS // EPISODE_DAYS,
                 label + ": every32-day time limit recorded exactly once")
    for number, (row, episode) in enumerate(zip(rows, history), start=1):
        transition = number * EPISODE_DAYS
        date = utc(row["info_date"])
        checks.check(row.get("transition_number") == transition
                     and row.get("replay_position") == transition - 1 and row.get("env_index") == 0,
                     label + ": actual ordered single-env non-overwritten replay slot")
        checks.check(date == utc(episode["end_exclusive"]) == utc(episode["completed_until"])
                     and utc(TRAIN[0]) < date <= utc(TRAIN[1]),
                     label + ": timeout boundary strictly within original training")
        checks.check(row.get("done") == row.get("timeout") == 1.
                     and row.get("time_limit_truncated") is True
                     and row.get("effective_done") == 0.,
                     label + ": time limit bootstraps rather than becoming an absorbing terminal")
        values = {}
        for key in ("stored_observation", "terminal_observation", "stored_next_observation"):
            array = np.asarray(row.get(key + "_values"), dtype=float)
            if not checks.check(array.shape == (27,) and np.isfinite(array).all(),
                                label + ": full finite recorded replay observation:" + key):
                continue
            checks.check(row.get(key + "_sha256") == observation_digest(array),
                         label + ": actual float32 replay observation SHA:" + key)
            values[key] = array
        if len(values) != 3:
            continue
        checks.vector_close(values["terminal_observation"], values["stored_next_observation"],
                            label + ": replay stores actual terminal observation, never reset observation",
                            atol=0, rtol=0)
        for key, day in (("stored_observation", date - pd.Timedelta(days=1)),
                         ("terminal_observation", date), ("stored_next_observation", date)):
            checks.vector_close(values[key][:23], features.loc[day].to_numpy().astype(np.float32),
                                label + ": replay market prefix only uses completed prior hours:" + key,
                                atol=1e-7, rtol=0)


def verify_upstream(root, checks):
    root = Path(root)
    manifest = read_json(root / "verified_manifest.json")
    checks.check(manifest.get("commit") == SB3_COMMIT and manifest.get("tag") == "v2.9.0"
                 and manifest.get("license") == "MIT" and manifest.get("all_five_installed_algorithm_files_match_official") is True,
                 "Pinned official SB3 release, MIT and installed-source equality evidence")
    hashes = {}
    checks.check(sha(root / "verified_manifest.json") == VERIFIED_MANIFEST_SHA256,
                 "The independently archived full upstream provenance manifest remains immutable")
    for name, expected in {**UPSTREAM_SHA256, "LICENSE": LICENSE_SHA256, "setup.py": SETUP_SHA256}.items():
        path = safe_file(root, name)
        actual = sha(path)
        checks.check(actual == expected == manifest.get("files", {}).get(name, {}).get("sha256"),
                     "Official source bytes and preserved notice:" + name)
        hashes[name] = actual
    return hashes


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
        raise ValueError("Incomplete TD3 hourly funded account:" + label)
    inventory = attach_inventory_replay(rebuilt, fills, dataset["hourly_close"], checks=checks, label=label, metadata=metadata)
    _check_summary(row, rebuilt["metrics"], checks, label)
    parameters = [records[(profile, seed)]["parameter_sha256"] for seed in seeds]
    checks.check(metadata.get("algorithm") == "td3" and metadata.get("family") == family
                 and metadata.get("profile") == profile and metadata.get("model_count") == len(seeds),
                 label + ": true fixed TD3 model population")
    checks.check(metadata.get("parameter_sha256_before") == metadata.get("parameter_sha256_after") == parameters,
                 label + ": actual trained policy parameters remain immutable at inference")
    for name, expected in (("rebalance_band", BAND), ("cash_floor_target", .05), ("max_asset_weight_target", .95),
        ("drawdown_penalty", .25 if profile == "balanced" else 0.), ("exposure_multiplier", FAMILIES[family])):
        checks.close(metadata.get(name), expected, label + ": unchanged financial contract:" + name, atol=1e-15, rtol=0)
    checks.check(metadata.get("old_actions_or_equity_scaled") is False
                 and metadata.get("training_budget_multiple") == 2 and metadata.get("deterministic_prediction") is True
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


def audit_group(inputs, dataset, records, spec, *, split, family, profile, features):
    folder = f"{split}/{family}/{profile}"
    checks = inputs.checks
    for name in GROUP_FILES:
        inputs.capture("study", folder + "/" + name)
    summary, daily, monthly = (inputs.csv("study", folder + "/" + name) for name in GROUP_FILES[:3])
    validate_group_rows(summary, split=split, family=family, profile=profile)
    completion = inputs.json("study", folder + "/completion.json")
    checks.check(completion.get("case_count") == completion.get("repeat_count") == 12
                 and completion.get("protocol_sha256") == sha(inputs.roots["study"] / "protocol.json"),
                 folder + ": complete twelve-account/twelve-repeat group")
    proofs = completion.get("repeats", [])
    by_id = {proof.get("run_id"): proof for proof in proofs}
    checks.check(len(proofs) == len(by_id) == 12 and set(by_id) == set(summary.run_id),
                 folder + ": exactly all fixed repeated cases")
    expected_uuid = spec["profile_gpu_assignments"][profile]
    device = completion.get("device", {})
    checks.check(device.get("visible_gpu_uuid") == expected_uuid, folder + ": exact assigned profile card")
    expected = {}
    for row in summary.to_dict("records"):
        key = "ensemble" if row["primary_policy"] else str(int(row["seed"]))
        relative = folder + "/" + row["scenario"] + "/" + key
        for name in CASE_FILES:
            inputs.capture("study", relative + "/" + name)
        run, history, months = audit_case(safe_file(inputs.roots["study"], relative), dataset, row, records, checks,
            split=split, family=family, profile=profile, features=features,
            gpus=[gpu for gpu in spec["physical_gpus"] if gpu["uuid"] == expected_uuid])
        proof = inputs.json("study", relative + "/replay.json")
        checks.check(by_id.get(row["run_id"]) == proof
                     and proof.get("worker_pid") == completion.get("worker_pid") and proof.get("device") == device,
                     row["run_id"] + ": actual worker and full case receipt agree")
        run.update(relative_directory=relative, family=family, profile=profile, scenario=row["scenario"])
        expected[row["run_id"]] = (run, history, months)
    compare_group_tables(summary, daily, monthly, expected, checks, folder)
    return summary, daily, monthly, expected


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


def verify_workers(inputs, spec, completion, *, split, chosen=None):
    checks = inputs.checks
    workers = completion.get("workers", []) if split == "validation" else [completion.get("worker", {})]
    profiles = set(PROFILES) if split == "validation" else {chosen["profile"]}
    checks.check(len(workers) == len(profiles) and {row.get("profile") for row in workers} == profiles
                 and len({row.get("worker_pid") for row in workers}) == len(profiles),
                 split + ": exact profile-specific actual worker handles")
    for worker in workers:
        profile = worker.get("profile")
        if profile not in profiles:
            continue
        families = list(FAMILIES) if split == "validation" else [chosen["family"]]
        checks.check(worker.get("split") == split and worker.get("families") == families
                     and worker.get("device", {}).get("visible_gpu_uuid") == spec["profile_gpu_assignments"][profile]
                     and isinstance(worker.get("worker_pid"), int) and worker["worker_pid"] > 0,
                     split + ": true declared profile/budget worker on its assigned UUID")
        for family in families:
            group = inputs.json("study", f"{split}/{family}/{profile}/completion.json")
            checks.check(group.get("worker_pid") == worker.get("worker_pid") and group.get("device") == worker.get("device"),
                         split + ": actual case-group worker belongs to the completed coordinator")


def verify_selection(inputs, records):
    """Keep validated frozen score bytes for cross-platform exact-tie stability."""
    checks = inputs.checks
    selected, receipt = inputs.json("study", "selection.json"), inputs.json("study", "selection_receipt.json")
    previous = inputs.json("study", "independent_audit.validation.json")
    checks.check(previous.get("passed") is True and previous.get("status") == "partial_test_pending"
                 and previous.get("training_model_count") == 10 and previous.get("validation_run_count") == 48
                 and previous.get("run_count") == previous.get("repeat_count") == 48
                 and previous.get("test_complete") is False and previous.get("failure_count") == 0
                 and previous.get("failures") == [], "Actual complete independent validation preceded selection")
    inputs.pin("study", previous.get("input_sha256"), validation_files(), "Original independent48-account validation evidence")
    inputs.pin("main", previous.get("main_input_sha256"), {"frozen_spec.json", "runtime.json"}, "Original validation reference bytes")
    inputs.pin("source", previous.get("source_sha256"), SOURCE_FILES, "Original validation auditor/executor source")
    checks.check(selected.get("audit_sha256") == sha(inputs.roots["study"] / "independent_audit.validation.json")
                 and receipt.get("selection_file") == "selection.json"
                 and receipt.get("selection_sha256") == sha(inputs.roots["study"] / "selection.json")
                 and receipt.get("protocol_sha256") == selected.get("protocol_sha256")
                 == sha(inputs.roots["study"] / "protocol.json"),
                 "Whole-file fixed candidate seal and its original audit remain immutable")
    checks.check(selected.get("seed_selection") is False and selected.get("selection_uses_test") is False
                 and selected.get("test_results_available_to_selector") is False
                 and all(selected.get(flag) is True for flag in FLAGS),
                 "No seed/test selection; already-seen-year disclosure retained")
    expected_training = {"training_completion.json", *(f"training/td3__{profile}__seed{seed}/{name}"
        for profile in PROFILES for seed in SEEDS for name in ("record.json", "model.zip", "training_timeout_transitions.jsonl"))}
    inputs.pin("study", selected.get("training_source_sha256"), expected_training, "All ten original trained models and timeout traces")
    inputs.pin("study", selected.get("validation_source_sha256"), validation_files(), "All48 original validation accounts and repeats")
    groups = {(family, profile): inputs.csv("study", f"validation/{family}/{profile}/summary.csv")
              for family in FAMILIES for profile in PROFILES}
    expected = {(row["family"], row["profile"]): row for row in rank_candidates(groups)}
    actual = selected.get("candidates", [])
    checks.check(len(actual) == 4 and {(row.get("family"), row.get("profile")) for row in actual} == set(expected),
                 "Only the four predeclared profile/budget candidates compete")
    for row in actual:
        truth = expected.get((row.get("family"), row.get("profile")))
        if truth is None:
            continue
        checks.check(row.get("algorithm") == "td3" and row.get("exposure_multiplier") == truth["exposure_multiplier"]
                     and row.get("rebalance_band") == BAND, "Selection keeps fixed algorithm/budget/2.5pp band")
        checks.close(row.get("score"), truth["score"], "Independent original fixed validation score", atol=1e-12)
        for cost in COSTS:
            checks.check(row.get(cost, {}).get("run_id") == truth[cost]["run_id"], "Actual validation ensemble ID")
            for key in ("net_return", "max_drawdown", "final_equity", "fees"):
                checks.close(row.get(cost, {}).get(key), truth[cost][key], "Independent validation source metric:" + key, atol=1e-10)
    order = sorted(actual, key=lambda row: (-row["score"], 0 if row["family"] == "raw" else 1, row["profile"]))
    checks.check(actual == order and selected.get("candidate") == (order[0] if order else None),
                 "Validated frozen score ordering and exact ties raw-first then lexical profile")
    checks.check(utc(previous["audited_at"]) <= utc(selected["selected_at"]) <= utc(receipt["sealed_at"]),
                 "Complete independent validation precedes fixed choice and its seal")
    return selected["candidate"], selected, receipt


def verify_annual_authorization(inputs, spec, receipt):
    checks = inputs.checks
    auth, marker = inputs.json("study", "annual_authorization.json"), inputs.json("study", "test_started.json")
    checks.check(auth.get("annual_evaluation_authorized") is True and auth.get("no_live_trading") is True
                 and auth.get("financial_parameters_fixed") is True
                 and auth.get("selection_sha256") == sha(inputs.roots["study"] / "selection.json")
                 and all(auth.get(flag) is True for flag in FLAGS),
                 "New annual replay needs explicit separate authorization after the sealed validation")
    checks.check(marker.get("protocol_sha256") == sha(inputs.roots["study"] / "protocol.json")
                 and marker.get("selection_sha256") == sha(inputs.roots["study"] / "selection.json")
                 and marker.get("authorization_sha256") == sha(inputs.roots["study"] / "annual_authorization.json")
                 and utc(receipt["sealed_at"]) <= utc(auth["authorized_at"]) <= utc(marker["at"]),
                 "Actual annual start marker identifies this authorization and exact fixed candidate")
    return auth, marker


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
        "fresh_holdout_success": False, "training_budget_multiple": 2, "recipe_is_pure_official_default": False,
        "accounting_passage_is_not_goal_success": True, **{flag: True for flag in FLAGS}}


def audit(output, main, dataset_path, *, phase="validation", source_root=ROOT):
    checks, started = Checks(), prior.now()
    inputs = Inputs(output, main, source_root, checks)
    records, runs, baseline_runs, research, training = {}, {}, {}, None, None
    selected, validation_count = None, 0
    try:
        if phase not in ("validation", "test"):
            raise ValueError("Explicit validation or separately authorized test phase required")
        if phase == "validation" and any((Path(output) / name).exists() for name in
            ("test", "test_started.json", "test_process.json", "test_completion.json", "annual_authorization.json")):
            raise ValueError("Validation auditing cannot inspect a study already dispatched to the new annual phase")
        spec, old = verify_protocol(inputs, dataset_path)
        fit, fit_complete = verify_phase(inputs, "fit", spec)
        validate, validate_complete = verify_phase(inputs, "validate", spec)
        checks.check(fit_complete.get("model_count") == 10
                     and validate_complete.get("case_count") == validate_complete.get("repeat_count") == 48
                     and utc(fit["completed_at"]) <= utc(validate["started_at"]), "All ten fits complete before forty-eight validation accounts")
        if phase == "test":
            selected, _, selection_receipt = verify_selection(inputs, {})
            _, marker = verify_annual_authorization(inputs, spec, selection_receipt)
            test, test_complete = verify_phase(inputs, "test", spec)
            checks.check(test_complete.get("case_count") == test_complete.get("repeat_count") == 12
                         and test_complete.get("candidate") == selected
                         and utc(marker["at"]) <= utc(test["started_at"]), "Only the previously authorized selected twelve annual accounts complete")
        if checks.failed:
            raise ValueError("Source/terminal/selection/authorization checks failed before account or tensor reads")
        cutoff = VALIDATION[1] if phase == "validation" else TEST[1]
        dataset = load_market(dataset_path, cutoff)
        features = prior.independent_market_features(dataset, cutoff=cutoff)
        dates = VALIDATION if phase == "validation" else TEST
        expected_hours = pd.date_range(utc(dates[0]), utc(dates[1]), freq="h", inclusive="left")
        for field in ("open", "high", "low", "close", "volume"):
            data = dataset["hourly_" + field].reindex(expected_hours).to_numpy(dtype=float)
            checks.check(np.isfinite(data).all() and (data >= 0 if field == "volume" else data > 0).all(),
                         "Complete original two-coin market hours without interpolation:" + field)
        records, training = audit_training(inputs, spec, dataset, features)
        if phase == "validation":
            groups = {}
            all_groups = []
            for family in FAMILIES:
                for profile in PROFILES:
                    group = audit_group(inputs, dataset, records, spec, split="validation", family=family, profile=profile, features=features)
                    groups[(family, profile)] = group[0]; all_groups.append(group)
                    runs.update({key: values[0] for key, values in group[3].items()})
            actual = [inputs.csv("study", "validation/" + name) for name in GROUP_FILES[:3]]
            expected = [pd.concat([group[i] for group in all_groups], ignore_index=True) for i in range(3)]
            compare_tables(inputs, actual, expected, "Complete validation parent tables")
            root_complete = inputs.json("study", "validation_completion.json")
            checks.check(root_complete.get("case_count") == root_complete.get("repeat_count") == 48
                         and root_complete.get("protocol_sha256") == sha(inputs.roots["study"] / "protocol.json"),
                         "All four fixed validation candidates and exact repeats completed")
            verify_workers(inputs, spec, root_complete, split="validation")
            for name in validation_files():
                inputs.capture("study", name)
            rank_candidates(groups)
            validation_count = len(runs)
        else:
            group = audit_group(inputs, dataset, records, spec, split="test", family=selected["family"], profile=selected["profile"], features=features)
            verify_workers(inputs, spec, test_complete, split="test", chosen=selected)
            runs = {key: values[0] for key, values in group[3].items()}
            baseline, baseline_runs = audit_baselines(inputs, dataset, old)
            validation_count = 48
            if checks.failed == 0:
                research = performance_evidence(group[0], baseline, selected)
        inputs.unchanged()
    except Exception as exc:
        checks.check(False, type(exc).__name__ + ": " + str(exc))
    expected = (48, 0) if phase == "validation" else (12, 14)
    passed = checks.failed == 0 and len(records) == 10 and (len(runs), len(baseline_runs)) == expected
    return {"schema_version": 1, "started_at": started, "audited_at": prior.now(), "phase": phase,
        "passed": passed, "status": ("partial_test_pending" if phase == "validation" else "passed") if passed else "failed",
        "validation_complete": passed if phase == "validation" else validation_count == 48,
        "test_complete": passed and phase == "test", "training_model_count": len(records),
        "run_count": len(runs) + len(baseline_runs), "validation_run_count": validation_count,
        "test_run_count": len(runs) if phase == "test" else 0, "baseline_run_count": len(baseline_runs),
        "repeat_count": len(runs) if passed else 0, "checks_total": checks.count,
        "failure_count": checks.failed, "failures": checks.failures, "maximum_errors": checks.maximum_errors,
        "input_sha256": inputs.hashes["study"], "main_input_sha256": inputs.hashes["main"],
        "source_sha256": inputs.hashes["source"], "dataset_sha256": sha(dataset_path),
        "training_calendar": training, "runs": runs, "baseline_runs": baseline_runs,
        "declared_research_targets": TARGETS, "research_targets": research,
        "fresh_holdout_success": False, "training_budget_multiple": 2,
        "accounting_passage_is_not_goal_success": True, **{flag: True for flag in FLAGS}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("output-dir", "reference-main", "dataset"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--phase", choices=("validation", "test"), default="validation")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = args.report or args.output_dir / ("independent_audit.validation.json" if args.phase == "validation" else "independent_audit.json")
    if report.exists():
        raise FileExistsError("Preserve the previous audit/receipt; choose a new --report")
    result = audit(args.output_dir, args.reference_main, args.dataset, phase=args.phase)
    report.parent.mkdir(parents=True, exist_ok=True)
    with report.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False); stream.write("\n")
    print(json.dumps({key: result[key] for key in ("passed", "status", "training_model_count",
        "validation_run_count", "test_run_count", "baseline_run_count", "checks_total", "failure_count")}))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()

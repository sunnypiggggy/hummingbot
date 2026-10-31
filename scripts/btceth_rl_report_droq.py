"""Data-only annual DroQ queries, gated by complete independent evidence.

This reader imports no model, execution engine, selector or market pickle.
Five DroQ policies retain32768 environment interactions, but G20 entails
555360 joint-Q updates per model, so compute is not comparable. Their annual
replay is an already-seen-year follow-up, never a fresh holdout, and cannot
replace the original208 or execution-band38 research scopes.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TRAIN = ("2020-01-01T00:00:00Z", "2025-01-01T00:00:00Z")
VALIDATION = (TRAIN[1], "2025-10-23T00:00:00Z")
TEST = (VALIDATION[1], "2026-10-23T00:00:00Z")
PROFILES, BUDGETS = ("balanced",), {"raw": 1., "cap075": .75}
SEEDS = tuple(range(5))
COSTS = {"base": {"fee": .001, "slippage": .0002}, "stress": {"fee": .002, "slippage": .0005}}
BASELINES = ("cash", "btc_hold", "eth_hold", "equal_hold", "equal_rebalance", "sma200_trend", "momentum180")
FLAGS = ("designed_after_first_annual", "test_year_already_seen", "not_fresh_holdout")
SCOPE = "droq_known_year_followup"
CASE_FILES = ("equity_1h.csv.gz", "fills.csv", "weights.csv", "boundary_equity.csv", "metadata.json", "fingerprints.json", "member_targets.csv", "replay.json")
GROUP_FILES = ("summary.csv", "daily_equity.csv", "monthly_returns.csv", "completion.json")
FINANCIAL = ("scripts/backtest_btceth_rl.py", "scripts/btceth_rl_env.py", "scripts/crypto_rl_portfolio_env.py", "scripts/crypto_rl_reporting.py")
SOURCE_FILES = (*FINANCIAL, "scripts/audit_btceth_rl.py", "scripts/audit_btceth_rl_exposure_cap.py", "scripts/audit_btceth_rl_execution_band.py", "scripts/btceth_rl_droq.py", "scripts/audit_btceth_rl_droq.py")
TARGETS = {"primary_base_net_return_at_least": .05, "primary_base_max_drawdown_magnitude_at_most": .20,
    "primary_stress_net_return_at_least": 0., "primary_beats_equal_hold_return_and_drawdown": True,
    "positive_base_training_seeds_at_least": 3, "stronger_claim_requires_beating_simple_trend_controls": True,
    "targets_are_not_guarantees": True}
RECIPE = dict(hidden_sizes=[256, 256], replay_size=1000000, batch_size=256, lr=.0003, gamma=.99, polyak=.995,
    alpha=.2, auto_alpha=True, target_entropy="auto", start_steps=5000, delay_update_steps="auto", utd_ratio=20,
    num_Q=2, num_min=2, q_target_mode="min", policy_update_delay=20, target_drop_rate=.01, layer_norm=True)
UPSTREAM_COMMIT = "4c7adddbaf7573ff9256b16091a3bbdbd536aad4"
MANIFEST_SHA256 = "9b0eb6b5e08fefae4a468061ab31e93bcd4d67606750b6196e926f6e103e7560"
CONFIG_SHA256 = "d76e838b65d333c0669b5b138aa5bb8e81efe9e88921c25a0d00feb266828c1a"
RUNNER_SHA256 = "f1463c5e206aabd6a77e934f7972a55c84c15901b4a380e470c7dcbbcf08443a"
AUDITOR_SHA256 = "bf33c879ff5eeaa13ee288c561fb5e6f18efb3fddc0da5179e6d646339b8c2bc"
COUNTS = dict(environment_steps=32768, replay_adds=32768, random_exploration_actions=5001, learning_train_calls=27768,
    q_inner_rounds=555360, q_optimizer_steps=[555360, 555360], total_q_optimizer_steps=1110720,
    target_updates_per_network=[555360, 555360], policy_optimizer_steps=27768, alpha_optimizer_steps=27768)


def no_link(path):
    path = Path(path)
    if any(p.is_symlink() or hasattr(p, "is_junction") and p.is_junction() for p in (path, *path.parents)):
        raise ValueError("Report evidence cannot cross links or junctions")
    return path


def sha(path):
    digest = hashlib.sha256()
    with no_link(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe(root, name):
    from pathlib import PurePosixPath
    text = str(name).replace("\\", "/"); parsed = PurePosixPath(text)
    if not text or parsed.is_absolute() or ":" in text or ".." in parsed.parts:
        raise ValueError("Report input pins must remain relative to their explicit root")
    root = no_link(root).resolve(); path = no_link(root.joinpath(*parsed.parts)); path.resolve().relative_to(root)
    return path


class Sources:
    def __init__(self):
        self.hashes = {}
    def file(self, path):
        path = no_link(path).resolve()
        if not path.is_file():
            raise FileNotFoundError("Complete report evidence required: " + str(path))
        key, digest = path.as_posix(), sha(path)
        if key in self.hashes and self.hashes[key] != digest:
            raise ValueError("Report evidence changed between reads: " + key)
        self.hashes[key] = digest
        return path
    def json(self, path):
        return json.loads(self.file(path).read_text(encoding="utf-8-sig"))
    def csv(self, path):
        return pd.read_csv(self.file(path), float_precision="round_trip")
    def pins(self, root, values, *, required=(), label="Evidence"):
        if not isinstance(values, dict) or not values or not set(required).issubset(values):
            raise ValueError(label + " has partial SHA evidence")
        for name, expected in values.items():
            if sha(self.file(safe(root, name))) != expected:
                raise ValueError(label + " SHA changed: " + name)
    def unchanged(self):
        for name, expected in self.hashes.items():
            if sha(name) != expected:
                raise ValueError("Report input changed after reading: " + name)


def same_metrics(actual, expected, label):
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise ValueError(label + " metric fields differ")
        for key in expected:
            same_metrics(actual[key], expected[key], label + ":" + key)
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise ValueError(label + " metric lists differ")
        for first, second in zip(actual, expected):
            same_metrics(first, second, label)
    elif isinstance(expected, (bool, np.bool_)):
        if not isinstance(actual, (bool, np.bool_)) or bool(actual) != bool(expected):
            raise ValueError(label + " boolean differs")
    elif isinstance(expected, (int, float, np.number)):
        if isinstance(actual, (bool, np.bool_)) or not isinstance(actual, (int, float, np.number)) or not np.isfinite([actual, expected]).all() or not np.isclose(actual, expected, atol=1e-10, rtol=1e-10):
            raise ValueError(label + " metric value differs")
    elif actual != expected:
        raise ValueError(label + " identity differs")


def utc(value):
    value = pd.Timestamp(value)
    if value.tzinfo is None:
        raise ValueError("DroQ evidence requires timezone-aware timestamps")
    return value.tz_convert("UTC")


def positive_int(value):
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def group_files(split, family, profile):
    folder = f"{split}/{family}/{profile}"
    return {*(f"{folder}/{name}" for name in GROUP_FILES),
        *(f"{folder}/{cost}/{key}/{name}" for cost in COSTS for key in (*map(str, SEEDS), "ensemble") for name in CASE_FILES)}


def baseline_files():
    folder = "evaluation/test/baselines"
    return {*(f"{folder}/{name}" for name in GROUP_FILES),
        *(f"{folder}/{cost}/{key}/{name}" for cost in COSTS for key in BASELINES for name in CASE_FILES[:6])}


def training_files():
    return {"training_completion.json", *(f"training/droq__{profile}__seed{seed}/{name}"
        for profile in PROFILES for seed in SEEDS for name in ("started.json", "record.json", "model_tensors.pt", "model.json", "training_timeout_transitions.jsonl"))}


def validation_files():
    return {"validation_completion.json", "validate_completion.json", "validate_process.json", "validate.log",
        *(f"validation/{name}" for name in GROUP_FILES[:3]),
        *(name for family in BUDGETS for profile in PROFILES for name in group_files("validation", family, profile))}


def _audit(value, *, validation=False):
    wanted = dict(passed=True, status="partial_test_pending" if validation else "passed", phase="validation" if validation else "test",
        test_complete=not validation, validation_complete=True, training_model_count=5, validation_run_count=24,
        test_run_count=0 if validation else 12, baseline_run_count=0 if validation else 14,
        run_count=24 if validation else 26, repeat_count=24 if validation else 12, failure_count=0, failures=[],
        fresh_holdout_success=False, training_budget_multiple=1, accounting_passage_is_not_goal_success=True,
        **{flag: True for flag in FLAGS})
    if (any(value.get(k) != v or type(v) in (bool, int) and type(value.get(k)) is not type(v) for k, v in wanted.items()) or value.get("declared_research_targets") != TARGETS
            or not positive_int(value.get("checks_total"))
            or any(not positive_int(value[k]) for k in ("checks", "checks_total") if k in value)
            or ("checks" in value and value["checks"] != value["checks_total"])):
        raise ValueError("Complete independent DroQ annual26 required; validation24 remains partial_test_pending")


def _bind(sources, roots, value, *, study_required=(), main_required=()):
    sources.pins(roots["study"], value.get("input_sha256"), required=study_required, label="DroQ study audit")
    sources.pins(roots["main"], value.get("main_input_sha256"), required=main_required, label="DroQ original-main audit")
    sources.pins(roots["source"], value.get("source_sha256"), required=SOURCE_FILES, label="DroQ audit implementation")


def _protocol(sources, study, main, source_root):
    p, receipt, old = (sources.json(path) for path in (study / "protocol.json", study / "protocol_receipt.json", main / "frozen_spec.json"))
    fixed = dict(algorithm="droq", profiles={"balanced": {"drawdown_penalty": .25, "rebalance_band": .025}}, budgets=BUDGETS,
        seeds=list(SEEDS), train=list(TRAIN), validation=list(VALIDATION), test=list(TEST), costs=COSTS, capital=10000.,
        timesteps_per_model=32768, training_budget_multiple=1, environment_step_budget_multiple=1, compute_budget_comparable=False,
        train_episode_days=32, model_count=5, validation_cases=24, validation_repeats=24, test_cases=12, test_repeats=12,
        cash_floor=.05, max_asset_weight=.95, hyperparameters=RECIPE, expected_counters=COUNTS, recipe_is_pure_official_default=False,
        seed_selection=False, selection_uses_test=False, new_recipe_search=False, annual_requires_separate_authorization=True,
        research_targets=TARGETS, selection_score="mean(log1p(net_return)-0.5*abs(max_drawdown)-2*max(abs(max_drawdown)-0.20,0)) across base/stress",
        tie_break="raw before cap075, then profile lexical identifier", **{flag: True for flag in FLAGS})
    if (any(p.get(k) != v or isinstance(v, bool) and p.get(k) is not v for k, v in fixed.items())
            or receipt.get("protocol_file") != "protocol.json" or receipt.get("protocol_sha256") != sha(study / "protocol.json")
            or utc(p["prepared_at"]) > utc(receipt["sealed_at"]) or p.get("original_frozen_spec_sha256") != sha(main / "frozen_spec.json")
            or old.get("timesteps_per_model") != 32768 or old.get("research_targets") != TARGETS
            or any(p.get(k) != old.get(k) for k in ("dataset_sha256", "dataset_manifest_sha256"))):
        raise ValueError("Fixed DroQ source recipe,5/24/26 scope and whole-file protocol required")
    code = p.get("source_sha256", {})
    if (set(code) != set(SOURCE_FILES) or any(code[k] != old.get("code_sha256", {}).get(k) for k in FINANCIAL)
            or code.get("scripts/btceth_rl_droq.py") != RUNNER_SHA256 or code.get("scripts/audit_btceth_rl_droq.py") != AUDITOR_SHA256):
        raise ValueError("Exact f146/bf33 nine-source DAG must preserve original financial sources")
    sources.pins(source_root, code, required=SOURCE_FILES, label="Actual DroQ source")
    sources.pins(study / "sources", code, required=SOURCE_FILES, label="Archived DroQ source")
    sources.pins(main, p.get("original_input_sha256"), required={"frozen_spec.json", "runtime.json", "independent_audit.json"} | baseline_files(), label="Predeclared original controls")
    runtime, gpus = sources.json(main / "runtime.json"), p.get("physical_gpus", [])
    if (gpus != runtime.get("physical_gpus") or len(gpus) != 2 or len({g["uuid"] for g in gpus}) != 2 or any("V100" not in g["name"] for g in gpus)
            or p.get("seed_gpu_assignments") != {str(seed): gpus[seed % 2]["uuid"] for seed in SEEDS}
            or p.get("family_gpu_assignments") != {family: gpu["uuid"] for family, gpu in zip(BUDGETS, gpus)}
            or p.get("runtime_packages") != old.get("runtime_packages")):
        raise ValueError("Original two physical V100s, fixed seed/family assignments and dependency pins required")
    upstream = study / "upstream"; manifest = sources.json(upstream / "verified_manifest.json")
    authors = manifest.get("files", []); expected = {v["path"] for v in authors} | {"verified_manifest.json", "github_commit.json", "github_tree.json"}
    if (sha(upstream / "verified_manifest.json") != MANIFEST_SHA256 or manifest.get("commit") != UPSTREAM_COMMIT or manifest.get("license") != "MIT"
            or manifest.get("file_count") != 27 or len(authors) != 27 or manifest.get("all_git_blob_sha1_verified") is not True
            or set(p.get("upstream_evidence_sha256", {})) != expected or p.get("upstream_commit") != UPSTREAM_COMMIT
            or any(p["upstream_evidence_sha256"][v["path"]] != v["sha256"] for v in authors)):
        raise ValueError("All27 exact original author files plus three provenance files and MIT notice required")
    sources.pins(upstream, p["upstream_evidence_sha256"], required=expected, label="DroQ author source/license")
    config = sources.json(study / "effective_config_contract.json")
    if (sha(study / "effective_config_contract.json") != p.get("effective_config_contract_sha256") or p["effective_config_contract_sha256"] != CONFIG_SHA256
            or config.get("official_commit") != UPSTREAM_COMMIT or config.get("effective_constructor_kwargs") != RECIPE):
        raise ValueError("Reviewed exact .01/LN/G20 effective configuration required")
    auth = sources.json(study / "preflight/fit_validation_authorization.json")
    if (sha(study / "preflight/fit_validation_authorization.json") != p.get("fit_validation_authorization_sha256")
            or auth.get("fit_validation_authorized") is not True or auth.get("no_live_trading") is not True
            or auth.get("reviewed_source_sha256") != code or auth.get("hyperparameters") != RECIPE
            or auth.get("seeds") != list(SEEDS) or auth.get("timesteps_per_model") != 32768 or auth.get("budgets") != BUDGETS
            or any(auth.get(k) is not True for k in FLAGS) or utc(auth["authorized_at"]) > utc(p["prepared_at"])):
        raise ValueError("Explicit source-bound fit/validation approval must precede freezing")
    return p, receipt


def _phase(sources, study, phase, p, receipt):
    process, complete, preflight = (sources.json(study / name) for name in (phase + "_process.json", phase + "_completion.json", "preflight/" + phase + ".json"))
    command = process.get("command", []); keys = ("index", "uuid", "name", "memory_total_mib")
    current = preflight.get("physical_gpus", [])
    if (process.get("phase") != phase or process.get("status") != "completed" or type(process.get("exit_code")) is not int or process["exit_code"] != 0
            or process.get("protocol_sha256") != sha(study / "protocol.json") or complete.get("protocol_sha256") != sha(study / "protocol.json")
            or process.get("source_sha256") != p["source_sha256"]["scripts/btceth_rl_droq.py"]
            or len(command) < 5 or not str(command[2]).replace("\\", "/").endswith("/btceth_rl_droq.py")
            or command.count("--phase") != 1 or command.index("--phase") + 1 >= len(command) or command[command.index("--phase") + 1] != "_" + phase
            or not positive_int(process.get("supervisor_pid")) or not positive_int(process.get("child_pid")) or process["supervisor_pid"] == process["child_pid"]
            or process.get("log_relative_path") != phase + ".log" or process.get("log_sha256") != sha(sources.file(study / (phase + ".log")))
            or len(current) != 2 or [{k: gpu[k] for k in keys} for gpu in current] != [{k: gpu[k] for k in keys} for gpu in p["physical_gpus"]]
            or any(not np.isfinite([gpu.get("memory_used_mib", -1), gpu.get("utilization_percent", -1)]).all() or gpu.get("memory_used_mib", -1) < 0
                or not 0 <= gpu.get("utilization_percent", -1) <= 100 for gpu in current)
            or not (utc(receipt["sealed_at"]) <= utc(preflight["at"]) <= utc(process["started_at"]) <= utc(complete["at"]) <= utc(process["completed_at"]))):
        raise ValueError("Real terminal0 source/log, physical GPU identity and time chain required:" + phase)
    good = complete.get("model_count") == 5 if phase == "fit" else complete.get("case_count") == complete.get("repeat_count") == (24 if phase == "validate" else 12)
    if not good:
        raise ValueError("Incomplete declared DroQ phase:" + phase)
    return process, complete


def _training(sources, study, p, fit, audits):
    completion = sources.json(study / "training_completion.json"); records = completion.get("records", [])
    if (completion.get("model_count") != 5 or len(records) != 5 or {(r.get("profile"), r.get("seed")) for r in records} != {("balanced", seed) for seed in SEEDS}
            or completion.get("protocol_sha256") != sha(study / "protocol.json")):
        raise ValueError("All five fixed newly trained DroQ policies required")
    expected = {k: RECIPE[k] for k in ("replay_size", "batch_size", "lr", "gamma", "polyak", "auto_alpha", "start_steps", "utd_ratio", "num_Q", "num_min", "q_target_mode", "policy_update_delay")}
    expected.update(hidden_sizes=[256, 256], delay_update_steps=5000, target_entropy=-2., dropout_per_q=[[.01, .01], [.01, .01]],
        layer_norm_per_q=[[[256], [256]], [[256], [256]]], target_training=[True, True])
    result = []
    for seed in SEEDS:
        folder = f"training/droq__balanced__seed{seed}"; record, started = (sources.json(study / folder / name) for name in ("record.json", "started.json"))
        if (record != next(r for r in records if r["seed"] == seed) or record.get("algorithm") != "droq" or record.get("profile") != "balanced"
                or record.get("actual_timesteps") != 32768 or record.get("actual_counters") != COUNTS or record.get("hyperparameters") != RECIPE
                or record.get("algorithm_class") != "redq.algos.redq_sac.REDQSACAgent" or record.get("recipe_is_pure_official_default") is not False
                or record.get("training_budget_multiple") != 1 or record.get("environment_step_budget_multiple") != 1 or record.get("compute_budget_comparable") is not False
                or record.get("protocol_sha256") != started.get("protocol_sha256") or record["protocol_sha256"] != sha(study / "protocol.json")
                or any(record.get(flag) is not True for flag in FLAGS)):
            raise ValueError("Exact five-model DroQ budget, author class and actual optimizer counts required")
        effective = record.get("effective_hyperparameters", {})
        if set(effective) != {*expected, "alpha_current", "log_alpha_current"} or any(effective.get(k) != v for k, v in expected.items()):
            raise ValueError("Actual active target-dropout/G20/mean-Q effective recipe required")
        for leaf, field, digest in (("model_tensors.pt", "model_tensor_file", "model_tensor_sha256"), ("model.json", "model_manifest_file", "model_manifest_sha256"),
                ("training_timeout_transitions.jsonl", "timeout_trace_relative_path", "timeout_trace_sha256")):
            if record.get(field) != folder + "/" + leaf or sha(sources.file(study / folder / leaf)) != record.get(digest):
                raise ValueError("Actual pure Tensor checkpoint/JSON/timeout bytes changed")
        manifest = sources.json(study / folder / "model.json")
        if (manifest.get("tensor_sha256") != record["model_tensor_sha256"] or manifest.get("parameter_sha256") != record.get("parameter_sha256")
                or manifest.get("counters") != COUNTS or manifest.get("upstream_commit") != UPSTREAM_COMMIT
                or not np.isclose(manifest.get("effective_alpha", np.nan), effective["alpha_current"], atol=1e-12, rtol=1e-12)
                or not np.isclose(manifest.get("log_alpha_value", np.nan), effective["log_alpha_current"], atol=1e-12, rtol=1e-12)):
            raise ValueError("Audited saved Tensor identities, alpha and actual update evidence must agree")
        loaded = record.get("upstream_loaded_sources", {})
        for name in ("OriginalREDQCodebase/redq/algos/core.py", "OriginalREDQCodebase/redq/algos/redq_sac.py"):
            if loaded.get(name, {}).get("sha256") != p["upstream_evidence_sha256"].get(name) or not str(loaded[name].get("file", "")).replace("\\", "/").endswith("/" + name):
                raise ValueError("Actual author Agent/core import identity required")
        device = record.get("device", {})
        if (device != started.get("device") or device.get("visible_gpu_uuid") != p["seed_gpu_assignments"][str(seed)] or device.get("device") != "cuda:0"
                or device.get("capability") != [7, 0] or "V100" not in device.get("name", "")
                or not (utc(fit["started_at"]) <= utc(started["at"]) <= utc(record["completed_at"]) <= utc(fit["completed_at"]))):
            raise ValueError("Actual fixed seed's isolated V100 and completed training time chain required")
        quality = record.get("training_window_quality", {})
        for audit in audits:
            calendar = audit.get("training_calendar", {})
            if (any(quality.get(k) != calendar.get(k) for k in ("requested_train_days", "candidate_windows", "eligible_windows", "excluded_windows"))
                    or quality.get("covered_training_days") != calendar.get("covered_days") or not positive_int(calendar.get("covered_days"))):
                raise ValueError("Independently complete causal TRAIN calendar required for every seed")
        result.append(dict(algorithm="droq", profile="balanced", seed=seed, actual_timesteps=32768, **COUNTS,
            environment_step_budget_multiple=1, compute_budget_comparable=False, training_budget_multiple=1, recipe_is_pure_official_default=False,
            gpu_uuid=device["visible_gpu_uuid"], model_tensor_sha256=record["model_tensor_sha256"], model_manifest_sha256=record["model_manifest_sha256"],
            parameter_sha256=record["parameter_sha256"], covered_training_days=quality["covered_training_days"], eligible_windows=quality["eligible_windows"],
            started_at=started["at"], completed_at=record["completed_at"], scope=SCOPE))
    return result


def _summary(frame, *, split, family=None, profile=None, baseline=False):
    dates = VALIDATION if split == "validation" else TEST
    expected = {f"test__{cost}__{key}" for cost in COSTS for key in BASELINES} if baseline else {
        f"{split}__{cost}__{family}__droq__{profile}__{key}" for cost in COSTS for key in (*map(str, SEEDS), "ensemble")}
    required = {"run_id", "strategy", "scenario", "split", "initial_capital", "start", "end_exclusive", "net_return", "max_drawdown", "final_equity", "fees", "average_exposure"}
    if (not required.issubset(frame) or len(frame) != len(expected) or frame.run_id.duplicated().any() or set(frame.run_id) != expected
            or not frame.split.eq(split).all() or not frame.initial_capital.eq(10000.).all()
            or not pd.to_datetime(frame.start, utc=True).eq(utc(dates[0])).all() or not pd.to_datetime(frame.end_exclusive, utc=True).eq(utc(dates[1])).all()):
        raise ValueError("Complete fixed DroQ/baseline population, window and initial capital required")
    values = frame[["net_return", "max_drawdown", "final_equity", "fees", "average_exposure"]].to_numpy(dtype=float)
    if (not np.isfinite(values).all() or (frame.net_return <= -1).any() or (frame.final_equity <= 0).any()
            or not frame.max_drawdown.between(-1, 0, inclusive="right").all() or (frame.fees < 0).any()
            or not frame.average_exposure.between(0, 1).all() or not np.allclose(frame.final_equity / 10000 - 1, frame.net_return, atol=1e-11, rtol=1e-11)):
        raise ValueError("Reported DroQ financial metrics must be finite actual funded accounts")
    if baseline:
        if any(row.strategy not in BASELINES or row.run_id != f"test__{row.scenario}__{row.strategy}" for row in frame.itertuples()):
            raise ValueError("Baseline identities cannot be relabeled as learned DroQ")
    else:
        fixed = dict(algorithm="droq", family=family, profile=profile, strategy="droq_" + family, exposure_multiplier=BUDGETS[family], rebalance_band=.025)
        if (not {*fixed, "seed", "primary_policy"}.issubset(frame) or any(not frame[k].eq(v).all() for k, v in fixed.items())
                or not frame.primary_policy.map(lambda v: isinstance(v, (bool, np.bool_))).all()):
            raise ValueError("Exact selected DroQ execution/model identity required")
        for cost in COSTS:
            group = frame.loc[frame.scenario.eq(cost)]
            primary, seeds = group.loc[group.primary_policy], group.loc[~group.primary_policy]
            if len(primary) != 1 or primary.seed.notna().any() or len(seeds) != 5 or set(seeds.seed) != set(SEEDS):
                raise ValueError("All five seeds and one genuine common-account ensemble required")
    return expected


def _history(history, months, summary):
    if (not {"run_id", "timestamp", "equity", "net_return", "drawdown"}.issubset(history)
            or not {"run_id", "month", "return"}.issubset(months) or set(history.run_id) != set(summary.run_id) or set(months.run_id) != set(summary.run_id)):
        raise ValueError("Complete annual curve and monthly population required")
    dates = pd.date_range(*TEST, freq="D")
    for row in summary.itertuples():
        curve, monthly = history.loc[history.run_id.eq(row.run_id)], months.loc[months.run_id.eq(row.run_id)]
        if (not pd.DatetimeIndex(pd.to_datetime(curve.timestamp, utc=True)).equals(dates)
                or not np.isfinite(curve[["equity", "net_return", "drawdown"]]).all().all() or (curve.equity <= 0).any()
                or (curve.drawdown > 0).any() or not np.isclose(curve.equity.iloc[0], 10000., atol=1e-8, rtol=0)
                or not np.isclose(curve.equity.iloc[-1], row.final_equity, atol=1e-7, rtol=1e-12)
                or not np.allclose(curve.net_return, curve.equity / 10000 - 1, atol=1e-11, rtol=1e-11)
                or row.max_drawdown > float(curve.drawdown.min()) + 1e-10):
            raise ValueError("Every annual boundary must reconcile with actual NAV and hourly-audited drawdown")
        daily = pd.Series(curve.equity.iloc[1:].to_numpy(), index=dates[1:] - pd.Timedelta(days=1))
        ends = daily.resample("ME").last(); expected = ends.pct_change(); expected.iloc[0] = ends.iloc[0] / 10000 - 1
        if (monthly.month.tolist() != expected.index.strftime("%Y-%m").tolist() or not np.isfinite(monthly["return"]).all()
                or not np.allclose(monthly["return"], expected, atol=1e-10, rtol=1e-10)
                or not np.isclose(np.prod(1 + monthly["return"]) - 1, row.net_return, atol=1e-10, rtol=1e-10)):
            raise ValueError("Annual monthly returns must reconcile with daily boundaries and final NAV")


def _runs(value, name, summary, *, hours):
    runs = value.get(name, {})
    if not isinstance(runs, dict) or set(runs) != set(summary.run_id):
        raise ValueError("Complete independently reconstructed account identities required")
    for row in summary.itertuples():
        proof = runs[row.run_id]
        if (proof.get("reconstructed") is not True or proof.get("hours") != hours or not isinstance(proof.get("fingerprints"), dict) or not proof["fingerprints"]
                or name == "runs" and proof.get("repeat_exact") is not True):
            raise ValueError("Full hourly account reconstruction required")
        for key in ("net_return", "max_drawdown", "fees", "final_equity"):
            same_metrics(proof.get(key), float(getattr(row, key)), "Independently audited DroQ metric:" + key)
    return runs


def _case_readbacks(sources, root, summary, history, proof, *, baseline=False, family=None, training=None):
    """Check actual exported account CSVs, without prices, inference or a simulator.

    The immutable bf33 audit proves execution/price/quantity semantics. This
    additional report check prevents a summary/curve/trade export mismatch.
    """
    dates = pd.date_range(*TEST, freq="h", inclusive="left")
    for row in summary.itertuples():
        key = row.strategy if baseline else "ensemble" if row.primary_policy else str(int(row.seed))
        folder = root / ("evaluation/test/baselines" if baseline else f"test/{family}/balanced") / row.scenario / key
        hourly = sources.csv(folder / "equity_1h.csv.gz"); fills = sources.csv(folder / "fills.csv")
        metadata, fingerprints = sources.json(folder / "metadata.json"), sources.json(folder / "fingerprints.json")
        required = {"timestamp", "equity", "cash", "base_value", "exposure"}
        if not required.issubset(hourly) or not pd.DatetimeIndex(pd.to_datetime(hourly.timestamp, utc=True)).equals(dates):
            raise ValueError("Complete8760 actual hourly rows required for every displayed account")
        numbers = hourly[["equity", "cash", "base_value", "exposure"]].to_numpy(dtype=float)
        if (not np.isfinite(numbers).all() or (hourly.equity <= 0).any() or (hourly.cash < -1e-7).any() or (hourly.base_value < -1e-7).any()
                or not np.allclose(hourly.equity, hourly.cash + hourly.base_value, atol=1e-7, rtol=1e-12)
                or not np.allclose(hourly.exposure, hourly.base_value / hourly.equity, atol=1e-10, rtol=1e-10)):
            raise ValueError("Actual cash plus funded holdings must reconcile to hourly equity")
        index = pd.DatetimeIndex(pd.to_datetime(hourly.timestamp, utc=True))
        nav = pd.Series(hourly.equity.to_numpy(), index=index)
        dd = nav / nav.cummax().clip(lower=10000.) - 1
        for observed, expected, name in ((float(nav.iloc[-1]), row.final_equity, "final NAV"), (float(dd.min()), row.max_drawdown, "hourly drawdown"),
                (float(hourly.exposure.mean()), row.average_exposure, "average actual exposure")):
            same_metrics(observed, float(expected), "Actual hourly export:" + name)
        if not {"timestamp", "side", "qty", "price", "fee"}.issubset(fills):
            raise ValueError("Actual trade export headers required even for a cash policy")
        if len(fills):
            values = fills[["qty", "price", "fee"]].to_numpy(dtype=float); stamps = pd.to_datetime(fills.timestamp, utc=True)
            if (not np.isfinite(values).all() or (fills.qty <= 0).any() or (fills.price <= 0).any() or (fills.fee < 0).any()
                    or not fills.side.isin(("BUY", "SELL")).all() or not ((stamps >= utc(TEST[0])) & (stamps < utc(TEST[1]))).all()
                    or not np.allclose(fills.fee, fills.qty * fills.price * COSTS[row.scenario]["fee"], atol=1e-8, rtol=1e-11)):
                raise ValueError("Actual exported spot trades must retain valid funded fees and window")
        same_metrics(float(fills.fee.sum()), float(row.fees), "Actual exported trade fees")
        if fingerprints != proof[row.run_id]["fingerprints"]:
            raise ValueError("Displayed account must preserve its exact independent-audit fingerprints")
        daily = history.loc[history.run_id.eq(row.run_id)]
        expected_nav = np.r_[10000., nav.resample("D").last().to_numpy()]
        expected_dd = np.r_[0., dd.resample("D").last().to_numpy()]
        if not np.allclose(daily.equity, expected_nav, atol=1e-7, rtol=1e-12) or not np.allclose(daily.drawdown, expected_dd, atol=1e-10, rtol=1e-10):
            raise ValueError("Displayed daily boundaries must reconcile to actual hourly files")
        if not baseline:
            members = training if key == "ensemble" else [r for r in training if r["seed"] == int(key)]
            params = [r["parameter_sha256"] for r in members]
            replay = sources.json(folder / "replay.json")
            if (metadata.get("algorithm") != "droq" or metadata.get("profile") != "balanced" or metadata.get("family") != family
                    or metadata.get("model_count") != len(members) or metadata.get("parameter_sha256_before") != params
                    or metadata.get("parameter_sha256_after") != params or replay.get("passed") is not True or replay.get("repeat_exact") is not True
                    or replay.get("fingerprints") != replay.get("repeated_fingerprints") or replay.get("fingerprints") != fingerprints
                    or replay.get("rng_sha256_before") != replay.get("rng_sha256_after_first") or replay.get("rng_sha256_before") != replay.get("rng_sha256_after_repeat")):
                raise ValueError("True common-account member identities and immutable deterministic DroQ inference required")


def _candidate(frame, family, profile):
    _summary(frame, split="validation", family=family, profile=profile)
    metrics = {cost: frame.loc[frame.scenario.eq(cost) & frame.primary_policy].iloc[0] for cost in COSTS}
    def one(row):
        dd = abs(row.max_drawdown)
        return math.log1p(row.net_return) - .5 * dd - 2 * max(dd - .20, 0.)
    return dict(algorithm="droq", family=family, profile=profile, exposure_multiplier=BUDGETS[family], rebalance_band=.025,
        score=sum(one(row) for row in metrics.values()) / 2,
        **{cost: dict(run_id=row.run_id, **{key: float(getattr(row, key)) for key in ("net_return", "max_drawdown", "fees", "final_equity")}) for cost, row in metrics.items()})


def _targets(frame, baselines, chosen):
    base = frame.loc[frame.scenario.eq("base") & frame.primary_policy].iloc[0]
    stress = frame.loc[frame.scenario.eq("stress") & frame.primary_policy].iloc[0]
    seeds = frame.loc[frame.scenario.eq("base") & ~frame.primary_policy].sort_values("seed")
    equal = baselines.loc[baselines.run_id.eq("test__base__equal_hold")].iloc[0]
    positive = int(seeds.net_return.gt(0).sum())
    gates = dict(base_return_at_least_5pct=bool(base.net_return >= .05), base_max_drawdown_at_most_20pct=bool(abs(base.max_drawdown) <= .20),
        stress_return_nonnegative=bool(stress.net_return >= 0), at_least_three_positive_training_seeds=positive >= 3,
        beats_equal_hold_return_and_drawdown=bool(base.net_return > equal.net_return and base.max_drawdown > equal.max_drawdown))
    trends = {name: bool(base.net_return > baselines.loc[baselines.run_id.eq("test__base__" + name)].iloc[0].net_return
        and base.max_drawdown > baselines.loc[baselines.run_id.eq("test__base__" + name)].iloc[0].max_drawdown) for name in ("sma200_trend", "momentum180")}
    return dict(candidate={key: chosen[key] for key in ("algorithm", "family", "profile")}, gates=gates,
        goal_performance_achieved=all(gates.values()), all_five_diagnostic_targets_met=all(gates.values()),
        base_net_return=float(base.net_return), base_max_drawdown=float(base.max_drawdown), stress_net_return=float(stress.net_return), stress_max_drawdown=float(stress.max_drawdown),
        positive_base_training_seeds=positive, base_seed_returns=[dict(seed=int(row.seed), net_return=float(row.net_return)) for row in seeds.itertuples()],
        base_fees_usdt=float(base.fees), stress_fees_usdt=float(stress.fees), beats_simple_trend_controls=trends, stronger_control_comparison_met=all(trends.values()),
        fresh_holdout_success=False, training_budget_multiple=1, recipe_is_pure_official_default=False, accounting_passage_is_not_goal_success=True,
        **{flag: True for flag in FLAGS})


def load_droq(study_root, main_root, *, audit_path=None, source_root=ROOT):
    """Return seven future annual query sets only after full26/24/5 gates.

    Baseline14 rows are reused reference accounts, not fourteen additional
    policy runs.  Validation scores retain their sealed Linux bytes for exact
    tie ordering; their numerical values are independently checked tolerantly.
    """
    study, main, source_root = map(lambda p: no_link(p).resolve(), (study_root, main_root, source_root))
    if study == main or study.is_relative_to(main) or main.is_relative_to(study):
        raise ValueError("DroQ needs an independent research tree, preserving previous studies")
    sources = Sources(); roots = dict(study=study, main=main, source=source_root)
    selected_path = no_link(audit_path).resolve() if audit_path else study / "independent_audit.local.json"
    if selected_path == study / "independent_audit.json":
        raise ValueError("Explicit local and retained remote audits must remain distinct identities")
    selected_path.relative_to(study)
    audits, audit_paths = [], list(dict.fromkeys((selected_path, study / "independent_audit.json")))
    for path in audit_paths:
        audit = sources.json(path); _audit(audit)
        _bind(sources, roots, audit, study_required={"protocol.json", "protocol_receipt.json", "selection.json", "selection_receipt.json", "independent_audit.validation.json", "annual_authorization.json", "test_started.json", "test_process.json", "test_completion.json"} | training_files() | validation_files(),
            main_required={"frozen_spec.json", "runtime.json", "independent_audit.json"} | baseline_files())
        audits.append(audit)
    if sha(audit_paths[0]) == sha(audit_paths[1]):
        raise ValueError("An independent local audit cannot be a renamed copy of the retained remote audit")
    p, protocol_receipt = _protocol(sources, study, main, source_root)
    if any(audit.get("dataset_sha256") != p["dataset_sha256"] or audit.get("source_sha256") != p["source_sha256"] for audit in audits):
        raise ValueError("Every audit must describe the same original frozen dataset")
    fit, _ = _phase(sources, study, "fit", p, protocol_receipt)
    validate, _ = _phase(sources, study, "validate", p, protocol_receipt)
    test, completion = _phase(sources, study, "test", p, protocol_receipt)
    initial = sources.json(study / "independent_audit.validation.json"); _audit(initial, validation=True)
    _bind(sources, roots, initial, study_required=validation_files() | training_files(), main_required={"frozen_spec.json", "runtime.json"})
    training = _training(sources, study, p, fit, [*audits, initial])
    selection, seal = (sources.json(study / name) for name in ("selection.json", "selection_receipt.json"))
    if (seal.get("selection_file") != "selection.json" or seal.get("selection_sha256") != sha(study / "selection.json")
            or seal.get("protocol_sha256") != selection.get("protocol_sha256") or selection.get("protocol_sha256") != sha(study / "protocol.json")
            or selection.get("audit_sha256") != sha(study / "independent_audit.validation.json")
            or selection.get("seed_selection") is not False or selection.get("selection_uses_test") is not False
            or selection.get("test_results_available_to_selector") is not False or any(selection.get(flag) is not True for flag in FLAGS)):
        raise ValueError("Immutable validation-only two-candidate selection and receipt required")
    selected_training = training_files() - {name for name in training_files() if name.endswith("/started.json")}
    sources.pins(study, selection.get("training_source_sha256"), required=selected_training, label="Sealed five DroQ models")
    sources.pins(study, selection.get("validation_source_sha256"), required=validation_files(), label="Sealed all24 DroQ validation accounts")
    actual, validation_frames = {}, []
    for family in BUDGETS:
        for profile in PROFILES:
            frame = sources.csv(study / f"validation/{family}/{profile}/summary.csv")
            actual[(family, profile)] = _candidate(frame, family, profile)
            validation_frames.append(frame)
    _runs(initial, "runs", pd.concat(validation_frames, ignore_index=True), hours=7080)
    pool = selection.get("candidates", [])
    if len(pool) != 2 or {(r.get("family"), r.get("profile")) for r in pool} != set(actual):
        raise ValueError("Exactly two predeclared DroQ validation candidates required")
    for item in pool:
        same_metrics(item, actual[(item["family"], item["profile"])], "Independently recomputed DroQ validation candidate")
    order = sorted(pool, key=lambda r: (-r["score"], 0 if r["family"] == "raw" else 1, r["profile"]))
    if pool != order or selection.get("candidate") != pool[0]:
        raise ValueError("Frozen score order must preserve raw-first exact ties, without test selection")
    chosen = selection["candidate"]
    annual_folder = f"test/{chosen['family']}/{chosen['profile']}"
    for audit in audits:
        sources.pins(study, audit["input_sha256"], required=group_files("test", chosen["family"], chosen["profile"]) | {"preflight/test.json", "test.log"}, label="Selected12 annual DroQ accounts")
    auth, marker = (sources.json(study / name) for name in ("annual_authorization.json", "test_started.json"))
    if (auth.get("annual_evaluation_authorized") is not True or auth.get("no_live_trading") is not True or auth.get("financial_parameters_fixed") is not True
            or auth.get("selection_sha256") != sha(study / "selection.json") or any(auth.get(flag) is not True for flag in FLAGS)
            or marker.get("protocol_sha256") != sha(study / "protocol.json") or marker.get("selection_sha256") != sha(study / "selection.json")
            or marker.get("authorization_sha256") != sha(study / "annual_authorization.json") or completion.get("candidate") != chosen
            or completion.get("selection_sha256") != sha(study / "selection.json")
            or not (utc(fit["completed_at"]) <= utc(validate["started_at"]) <= utc(validate["completed_at"]) <= utc(initial["audited_at"])
                <= utc(selection["selected_at"]) <= utc(seal["sealed_at"]) <= utc(auth["authorized_at"]) <= utc(marker["at"]) <= utc(test["started_at"]))):
        raise ValueError("Complete validation and sealed choice must precede separately authorized new annual replay")
    main_audit = sources.json(main / "independent_audit.json")
    if (main_audit.get("passed") is not True or main_audit.get("status") != "passed" or main_audit.get("test_complete") is not True
            or main_audit.get("run_count") != 172 or not isinstance(main_audit.get("runs"), dict) or len(main_audit["runs"]) != 172
            or not positive_int(main_audit.get("checks_total")) or main_audit.get("failure_count") != 0 or main_audit.get("failures") != []):
        raise ValueError("The retained original172 reference audit must remain complete")
    sources.pins(main, main_audit.get("input_sha256"), required={"frozen_spec.json", "test_completion.json"} | baseline_files(), label="Original172 completion and controls")
    annual_tables = [sources.csv(study / annual_folder / name) for name in GROUP_FILES[:3]]
    baseline_tables = [sources.csv(main / "evaluation/test/baselines" / name) for name in GROUP_FILES[:3]]
    summary, baseline = annual_tables[0], baseline_tables[0]
    _summary(summary, split="test", family=chosen["family"], profile=chosen["profile"])
    _summary(baseline, split="test", baseline=True)
    for tables in (annual_tables, baseline_tables):
        _history(tables[1], tables[2], tables[0])
    targets = _targets(summary, baseline, chosen)
    for audit in audits:
        _runs(audit, "runs", summary, hours=8760); refs = _runs(audit, "baseline_runs", baseline, hours=8760)
        if (utc(audit["audited_at"]) < utc(test["completed_at"]) or any(refs[key]["fingerprints"] != main_audit.get("runs", {}).get(key, {}).get("fingerprints") for key in refs)):
            raise ValueError("Annual audit must follow real completion and preserve original baseline fingerprints")
        same_metrics(audit.get("research_targets"), targets, "Actual DroQ annual research conditions")
    _case_readbacks(sources, study, summary, annual_tables[1], audits[0]["runs"], family=chosen["family"], training=training)
    _case_readbacks(sources, main, baseline, baseline_tables[1], audits[0]["baseline_runs"], baseline=True)
    goal = sources.json(study / "goal_candidate_evidence.json")
    if (goal.get("protocol_sha256") != sha(study / "protocol.json") or goal.get("selection_sha256") != sha(study / "selection.json")
            or goal.get("audit_sha256") != sha(study / "independent_audit.json") or goal.get("accounting_audit_passed") is not True
            or utc(goal["at"]) < utc(audits[audit_paths.index(study / "independent_audit.json")]["audited_at"])):
        raise ValueError("Goal evidence must retain its actual remote audit identity, independent of local re-audit")
    for key, expected in targets.items():
        same_metrics(goal.get(key), expected, "Own DroQ goal evidence:" + key)
    sources.file(Path(__file__))
    
    tables = [[], [], []]
    for role, source_tables in (("droq_selected", annual_tables), ("original_baseline_reference", baseline_tables)):
        for frame, dest in zip(source_tables, tables):
            item = frame.copy(); item["scope"] = SCOPE; item["droq_role"] = role; item["training_budget_multiple"] = 1 if role == "droq_selected" else None
            dest.append(item)
    target_rows = [dict(gate=key, met=value, category="已观察年度的DroQ自身诊断") for key, value in targets["gates"].items()]
    target_rows.extend(dict(gate=key, met=value, category="简单趋势对照") for key, value in targets["beats_simple_trend_controls"].items())
    audit_rows = [dict(role="本机明确审计" if i == 0 and path != study / "independent_audit.json" else "保留远端审计", file=path.as_posix(), sha256=sha(path),
        passed=True, new_runs=12, baseline_reference_runs=14, validation_runs=24, training_models=5, repeat_count=12, fresh_holdout_success=False) for i, path in enumerate(audit_paths)]
    frames = dict(droq_summary=pd.concat(tables[0], ignore_index=True), droq_equity=pd.concat(tables[1], ignore_index=True), droq_monthly=pd.concat(tables[2], ignore_index=True),
        droq_targets=pd.DataFrame(target_rows), droq_validation_selection=pd.DataFrame(pool), droq_training=pd.DataFrame(training), droq_audit=pd.DataFrame(audit_rows))
    source = dict(label="DroQ原作者数学、G20计算预算与已观察年度研究", classification="modeled", timezone="UTC", period=dict(start=TEST[0], endExclusive=TEST[1]),
        files=list(sources.hashes), fileHashes=dict(sources.hashes), datasetSha256=p["dataset_sha256"],
        caveats=["32768环境步×5模型；G20产生每模型555360次联合Q更新，不能解释为与原SAC计算量相同。.01 dropout为预声明研究参数。", "新年度已观察，不是fresh holdout；不能改写原208、5pp38或原失败目标。", "年度26行含12个新DroQ账户和14个原基准引用；基准并非新增模型或新增资金。", "两个执行预算候选只用295天验证选择；每个成本保留五个种子和一个平均目标的真实资金组合。", "SAC/TQC仅可从先前审计报告另作对照，不能替代DroQ自身种子稳定性条件。", "回撤来自完整小时权益独立重建；曲线展示完整日边界。"])
    queries = {}
    for name, frame in frames.items():
        context = copy.deepcopy(source)
        if name in ("droq_training", "droq_validation_selection"):
            dates = TRAIN if name == "droq_training" else VALIDATION
            context["period"] = dict(start=dates[0], endExclusive=dates[1])
        queries[name] = dict(rows=frame.astype(object).where(pd.notna(frame), None).to_dict("records"), source=context,
            methods=[dict(language="text", code="Read-only, fully SHA-bound DroQ evidence. Complete independent annual26 plus prior24 validation, all5 pure-Tensor checkpoints, sealed two-candidate selection, separate known-year authorization and own recomputed goal required. No inference, execution, price loading or prior-report mutation.")])
    sources.unchanged()
    return dict(queries=queries, source=source, targets=targets, selection=selection, audit=audits[0], external_input_sha256=dict(sources.hashes),
        query_run_count=26, new_run_count=12, baseline_reference_run_count=14, validation_run_count=24, training_model_count=5,
        original_scopes_unchanged=True, fresh_holdout_success=False, training_budget_multiple=1,
        environment_step_budget_multiple=1, compute_budget_comparable=False,
        droqFollowup=dict(algorithm="droq", profile="balanced", scope=SCOPE, chosen=chosen,
            targets=targets, known_year=True, fresh_holdout_success=False, prior_scopes_unchanged=True,
            models=5, timesteps_per_model=32768, environment_step_budget_multiple=1, compute_budget_comparable=False,
            q_inner_rounds_per_model=555360, individual_q_optimizer_steps_per_model=1110720,
            actor_optimizer_steps_per_model=27768, alpha_optimizer_steps_per_model=27768,
            upstream_commit=UPSTREAM_COMMIT, recipe_is_pure_official_default=False))

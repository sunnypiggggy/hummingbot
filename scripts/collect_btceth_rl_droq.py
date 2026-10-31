"""SHA-only snapshots for independently audited DroQ validation or annual work.

This transport never imports a model, financial environment, dataset payload,
selector or reporting module. Validation means5 policies and24 exact repeated
accounts. Full means their immutable24 validation plus12 newly repeated annual
accounts and14 original baseline references; a failed research goal is retained.
Each collection writes a separate sibling snapshot and preserves earlier receipts.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import sys
import tarfile
import types
import uuid

ROOT = Path(__file__).resolve().parents[1] if globals().get("__file__") not in (None, "<stdin>") else Path.cwd()
TRANSPORT_SHA256 = "54ef7a4add8a3ca363ef8855989d797047e3a8c4e3f8cf2e6282fb8abc03a9db"
SOURCE_HASHES = {
    "scripts/backtest_btceth_rl.py": "dbc7ffae69ddc1c7b043302d770b2c36066cd0b9c8577fa231d6d46d8e9e16a1",
    "scripts/btceth_rl_env.py": "9bf67454009ba03bf53720b3fe1f3587430c76599d610acdfce4f03ac8befe02",
    "scripts/crypto_rl_portfolio_env.py": "56091e7d29a79316b0a3f1d3defb2531856de9fd90e9cfb3f4e63a3df48a6179",
    "scripts/crypto_rl_reporting.py": "b77e0c55d23a2adf65dd87aab8df322458754e6c908d4b2152305543af6ff6d5",
    "scripts/audit_btceth_rl.py": "3c5e6840538cadd45f53026423946358de45da35dc8d30f8984571d78ba3728c",
    "scripts/audit_btceth_rl_exposure_cap.py": "d8e941874dd05f96910785bec36d9243bea36dcc2f1ee1cc4bb9142c7483e762",
    "scripts/audit_btceth_rl_execution_band.py": "b4d699bc57b22a1b436491bd501e4ed7af784834e8156f6a0042956f5eb6285f",
    "scripts/btceth_rl_droq.py": "f1463c5e206aabd6a77e934f7972a55c84c15901b4a380e470c7dcbbcf08443a",
    "scripts/audit_btceth_rl_droq.py": "bf33c879ff5eeaa13ee288c561fb5e6f18efb3fddc0da5179e6d646339b8c2bc",
}
UPSTREAM_COMMIT = "4c7adddbaf7573ff9256b16091a3bbdbd536aad4"
UPSTREAM_MANIFEST_SHA256 = "9b0eb6b5e08fefae4a468061ab31e93bcd4d67606750b6196e926f6e103e7560"
CONFIG_SHA256 = "d76e838b65d333c0669b5b138aa5bb8e81efe9e88921c25a0d00feb266828c1a"
REMOTE_ROOT = "/home/sunny/research/btceth_rl_droq_20261030"
REMOTE_SOURCE = "/home/sunny/research/btceth_rl_droq_20261030_preflight/formal_source"
REMOTE_MAIN = "/home/sunny/research/btceth_rl_20261029/experiment"
REMOTE_DATASET = "/home/sunny/research/btceth_rl_20261029/data/crypto_rl_btceth_20261029/dataset.pkl.gz"
REMOTE_TRANSPORT = "/home/sunny/research/btceth_rl_20261029/scripts/collect_btceth_rl_results.py"
FAMILIES, SEEDS, COSTS = ("raw", "cap075"), tuple(range(5)), ("base", "stress")
FLAGS = ("designed_after_first_annual", "test_year_already_seen", "not_fresh_holdout")
FINANCIAL = {"scripts/backtest_btceth_rl.py", "scripts/btceth_rl_env.py", "scripts/crypto_rl_portfolio_env.py", "scripts/crypto_rl_reporting.py"}
CASE_FILES = ("equity_1h.csv.gz", "fills.csv", "weights.csv", "boundary_equity.csv", "metadata.json", "fingerprints.json", "member_targets.csv", "replay.json")
GROUP_FILES = ("summary.csv", "daily_equity.csv", "monthly_returns.csv", "completion.json")
BASELINES = ("cash", "btc_hold", "eth_hold", "equal_hold", "equal_rebalance", "sma200_trend", "momentum180")
GATES = {"base_return_at_least_5pct", "base_max_drawdown_at_most_20pct", "stress_return_nonnegative", "at_least_three_positive_training_seeds", "beats_equal_hold_return_and_drawdown"}
TARGETS = {"primary_base_net_return_at_least": .05, "primary_base_max_drawdown_magnitude_at_most": .20,
    "primary_stress_net_return_at_least": 0., "primary_beats_equal_hold_return_and_drawdown": True,
    "positive_base_training_seeds_at_least": 3, "stronger_claim_requires_beating_simple_trend_controls": True, "targets_are_not_guarantees": True}
RECIPE = dict(hidden_sizes=[256, 256], replay_size=1000000, batch_size=256, lr=.0003, gamma=.99, polyak=.995,
    alpha=.2, auto_alpha=True, target_entropy="auto", start_steps=5000, delay_update_steps="auto", utd_ratio=20,
    num_Q=2, num_min=2, q_target_mode="min", policy_update_delay=20, target_drop_rate=.01, layer_norm=True)
COUNTS = dict(environment_steps=32768, replay_adds=32768, random_exploration_actions=5001, learning_train_calls=27768,
    q_inner_rounds=555360, q_optimizer_steps=[555360, 555360], total_q_optimizer_steps=1110720,
    target_updates_per_network=[555360, 555360], policy_optimizer_steps=27768, alpha_optimizer_steps=27768)
VALIDATION_AUDIT = "independent_audit.validation.json"


def load_transport(path):
    """A private instance of reviewed54ef; no globals or functions are patched."""
    path = Path(path)
    if path.is_symlink() or hasattr(path, "is_junction") and path.is_junction():
        raise ValueError("Reviewed transport source cannot be a link")
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != TRANSPORT_SHA256:
        raise ValueError("Use the exact reviewed54ef safe transport")
    module = types.ModuleType("_droq_frozen_transport"); module.__file__ = str(path.resolve())
    exec(compile(content, module.__file__, "exec"), module.__dict__)
    return module


_transport_path = Path(sys.argv[sys.argv.index("--transport-source") + 1]) if "--transport-source" in sys.argv else ROOT / "scripts/collect_btceth_rl_results.py"
transport = load_transport(_transport_path)
sha, read_json, safe_target = transport.sha, transport.read_json, transport.safe_target


def utc(value):
    value = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if value.tzinfo is None:
        raise ValueError("Timezone-aware evidence timestamps required")
    return value.astimezone(timezone.utc)


def hashes(root, values, label, *, required=()):
    if not isinstance(values, dict) or not values or not set(required).issubset(values):
        raise ValueError(label + " has incomplete immutable SHA dependencies")
    for name, expected in values.items():
        if not re.fullmatch(r"[0-9a-f]{64}", str(expected)) or sha(safe_target(root, name)) != expected:
            raise ValueError(label + " changed: " + str(name))


def training_files(*, started=True):
    leaves = ("record.json", "model_tensors.pt", "model.json", "training_timeout_transitions.jsonl") + (("started.json",) if started else ())
    return {"training_completion.json", *(f"training/droq__balanced__seed{seed}/{leaf}" for seed in SEEDS for leaf in leaves)}


def group_files(split, family):
    folder = f"{split}/{family}/balanced"
    return {*(f"{folder}/{name}" for name in GROUP_FILES),
        *(f"{folder}/{cost}/{key}/{name}" for cost in COSTS for key in (*map(str, SEEDS), "ensemble") for name in CASE_FILES)}


def validation_files():
    return {"validation_completion.json", "validate_completion.json", "validate_process.json", "validate.log",
        *("validation/" + name for name in GROUP_FILES[:3]), *(name for family in FAMILIES for name in group_files("validation", family))}


def baseline_files():
    folder = "evaluation/test/baselines"
    return {*(f"{folder}/{name}" for name in GROUP_FILES),
        *(f"{folder}/{cost}/{key}/{name}" for cost in COSTS for key in BASELINES for name in CASE_FILES[:6])}


def no_annual(root):
    for parent, dirs, files in os.walk(root, followlinks=False):
        for name in (*dirs, *files):
            path = Path(parent) / name
            if transport.is_link(path):
                raise ValueError("DroQ evidence cannot cross links or junctions")
            key = PurePosixPath(path.relative_to(root).as_posix())
            if "test" in key.parts or key.name.startswith(("test_", "test.", "annual")) or key.name in ("independent_audit.json", "goal_candidate_evidence.json"):
                raise ValueError("Validation-only snapshot refuses every annual/test artifact: " + key.as_posix())


def identities(root, main, dataset, source_root=ROOT, *, snapshot=False):
    if any(transport.is_link(p) for p in (root, main, dataset, source_root)):
        raise ValueError("Original research, data and source roots cannot be links or junctions")
    root, main, dataset, source_root = map(lambda p: Path(p).resolve(), (root, main, dataset, source_root))
    if root == main or root.is_relative_to(main) or main.is_relative_to(root):
        raise ValueError("DroQ and original reference must remain separate siblings")
    p, receipt, old = (read_json(safe_target(where, name)) for where, name in ((root, "protocol.json"), (root, "protocol_receipt.json"), (main, "frozen_spec.json")))
    fixed = dict(algorithm="droq", profiles={"balanced": {"drawdown_penalty": .25, "rebalance_band": .025}}, seeds=list(SEEDS),
        budgets={"raw": 1., "cap075": .75}, capital=10000., timesteps_per_model=32768, train_episode_days=32, model_count=5,
        validation_cases=24, validation_repeats=24, test_cases=12, test_repeats=12, cash_floor=.05, max_asset_weight=.95,
        hyperparameters=RECIPE, expected_counters=COUNTS, training_budget_multiple=1, environment_step_budget_multiple=1,
        compute_budget_comparable=False, recipe_is_pure_official_default=False, seed_selection=False, selection_uses_test=False,
        new_recipe_search=False, annual_requires_separate_authorization=True, research_targets=TARGETS, **{flag: True for flag in FLAGS})
    if (any(p.get(k) != v or type(v) is bool and p.get(k) is not v for k, v in fixed.items())
            or receipt.get("protocol_file") != "protocol.json" or receipt.get("protocol_sha256") != sha(root / "protocol.json")
            or utc(p["prepared_at"]) > utc(receipt["sealed_at"]) or p.get("source_sha256") != SOURCE_HASHES
            or p.get("original_frozen_spec_sha256") != sha(main / "frozen_spec.json")
            or any(p.get(k) != old.get(k) for k in ("dataset_sha256", "dataset_manifest_sha256", "train", "validation", "test"))
            or old.get("research_targets") != TARGETS or any(p["source_sha256"][k] != old.get("code_sha256", {}).get(k) for k in FINANCIAL)):
        raise ValueError("Frozen DroQ source, recipe, original identity or protocol receipt changed")
    if sha(dataset) != p["dataset_sha256"] or sha(safe_target(dataset.parent, "manifest.json")) != p["dataset_manifest_sha256"]:
        raise ValueError("Original frozen data/manifest bytes changed")
    hashes(source_root, SOURCE_HASHES, "Actual reviewed nine-source DAG")
    hashes(main, p.get("original_input_sha256"), "Original references", required=baseline_files() | {"frozen_spec.json", "runtime.json", "independent_audit.json"})
    if snapshot:
        hashes(root / "sources", SOURCE_HASHES, "Archived nine-source DAG")
        hashes(root / "upstream", p.get("upstream_evidence_sha256"), "Author source and license")
        upstream = root / "upstream"; manifest = read_json(safe_target(upstream, "verified_manifest.json")); files = manifest.get("files", [])
        expected = {item["path"] for item in files} | {"verified_manifest.json", "github_commit.json", "github_tree.json"}
        if (sha(upstream / "verified_manifest.json") != UPSTREAM_MANIFEST_SHA256 or manifest.get("commit") != UPSTREAM_COMMIT
                or manifest.get("license") != "MIT" or manifest.get("file_count") != 27 or len(files) != 27
                or manifest.get("all_git_blob_sha1_verified") is not True or set(p["upstream_evidence_sha256"]) != expected
                or p.get("upstream_commit") != UPSTREAM_COMMIT or p.get("upstream_manifest_sha256") != UPSTREAM_MANIFEST_SHA256):
            raise ValueError("Exact27 authored sources and three pinned provenance files required")
        for item in files:
            if p["upstream_evidence_sha256"][item["path"]] != item["sha256"]:
                raise ValueError("Author manifest and protocol disagree")
        config = read_json(safe_target(root, "effective_config_contract.json"))
        if (sha(root / "effective_config_contract.json") != CONFIG_SHA256 or p.get("effective_config_contract_sha256") != CONFIG_SHA256
                or config.get("effective_constructor_kwargs") != RECIPE or config.get("official_commit") != UPSTREAM_COMMIT):
            raise ValueError("Reviewed effective constructor contract changed")
    auth = read_json(safe_target(root, "preflight/fit_validation_authorization.json"))
    if (sha(safe_target(root, "preflight/fit_validation_authorization.json")) != p.get("fit_validation_authorization_sha256")
            or auth.get("fit_validation_authorized") is not True or auth.get("no_live_trading") is not True
            or auth.get("reviewed_source_sha256") != SOURCE_HASHES or auth.get("hyperparameters") != RECIPE
            or auth.get("seeds") != list(SEEDS) or auth.get("timesteps_per_model") != 32768 or auth.get("budgets") != fixed["budgets"]
            or any(auth.get(flag) is not True for flag in FLAGS) or utc(auth["authorized_at"]) > utc(p["prepared_at"])):
        raise ValueError("Actual source-bound fit/validation authorization must precede freezing")
    return p, dict(protocol_sha256=sha(root / "protocol.json"), frozen_spec_sha256=p["original_frozen_spec_sha256"],
        dataset_sha256=p["dataset_sha256"], dataset_manifest_sha256=p["dataset_manifest_sha256"], source_sha256=SOURCE_HASHES)


def phase_guard(root, phase, p, *, process_root, process_main, process_dataset, process_source):
    process, complete = (read_json(safe_target(root, phase + suffix)) for suffix in ("_process.json", "_completion.json"))
    command = process.get("command", [])
    if (process.get("phase") != phase or process.get("status") != "completed" or type(process.get("exit_code")) is not int or process["exit_code"] != 0
            or process.get("source_sha256") != SOURCE_HASHES["scripts/btceth_rl_droq.py"] or process.get("protocol_sha256") != sha(root / "protocol.json")
            or complete.get("protocol_sha256") != process["protocol_sha256"] or process.get("log_relative_path") != phase + ".log"
            or process.get("log_sha256") != sha(safe_target(root, phase + ".log"))
            or any(type(process.get(k)) is not int or process[k] <= 0 for k in ("supervisor_pid", "child_pid"))
            or process["supervisor_pid"] == process["child_pid"] or len(command) < 5
            or command[2] != str(PurePosixPath(str(process_source)) / "scripts/btceth_rl_droq.py")):
        raise ValueError("Actual source-bound completed0 DroQ process/log required: " + phase)
    for option, expected in {"--phase": "_" + phase, "--reference-main": str(process_main), "--dataset": str(process_dataset), "--output-dir": str(process_root)}.items():
        if command.count(option) != 1 or command.index(option) + 1 >= len(command) or command[command.index(option) + 1] != expected:
            raise ValueError("DroQ terminal command belongs to another study: " + option)
    preflight, receipt = read_json(safe_target(root, "preflight/" + phase + ".json")), read_json(root / "protocol_receipt.json")
    if not (utc(receipt["sealed_at"]) <= utc(preflight["at"]) <= utc(process["started_at"]) <= utc(complete["at"]) <= utc(process["completed_at"])):
        raise ValueError("DroQ freeze/preflight/start/completion/terminal timestamps differ")
    count_ok = complete.get("model_count") == 5 if phase == "fit" else complete.get("case_count") == complete.get("repeat_count") == (24 if phase == "validate" else 12)
    if not count_ok:
        raise ValueError("Complete5 fits,24 validation or12 annual accounts and repeats required")
    if sys.platform == "linux":
        for pid in (process["supervisor_pid"], process["child_pid"]):
            try:
                proc = Path("/proc") / str(pid); state = (proc / "stat").read_text().rsplit(")", 1)[1].split()[0]
                argv = (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
            except (FileNotFoundError, ProcessLookupError):
                continue
            if state != "Z" and "btceth_rl_droq.py" in argv:
                raise ValueError("DroQ research process is still live")
    return process, complete


def audit_scope(audit, *, full):
    fixed = dict(passed=True, status="passed" if full else "partial_test_pending", phase="test" if full else "validation",
        validation_complete=True, test_complete=full, training_model_count=5, validation_run_count=24,
        test_run_count=12 if full else 0, baseline_run_count=14 if full else 0, run_count=26 if full else 24,
        repeat_count=12 if full else 24, failure_count=0, failures=[], fresh_holdout_success=False, training_budget_multiple=1,
        recipe_is_pure_official_default=False, accounting_passage_is_not_goal_success=True, declared_research_targets=TARGETS,
        **{flag: True for flag in FLAGS})
    if (any(audit.get(k) != v or type(v) in (bool, int) and type(audit.get(k)) is not type(v) for k, v in fixed.items())
            or type(audit.get("checks_total")) is not int or audit["checks_total"] <= 0 or not full and audit.get("research_targets") is not None):
        raise ValueError("Independent complete DroQ partial24 or full26 scope required")


def common_files(p):
    result = {"protocol.json", "protocol_receipt.json", "effective_config_contract.json", "preflight/fit_validation_authorization.json",
        "fit_process.json", "fit_completion.json", "fit.log", "preflight/fit.json", "preflight/validate.json",
        *("sources/" + name for name in SOURCE_HASHES), *("upstream/" + name for name in p["upstream_evidence_sha256"])}
    if p.get("resource_allowance_sha256") is not None:
        result.add("resource_allowance.json")
    return result


def original_validation(root, main, source_root, p, identity, fit, validate):
    previous = read_json(safe_target(root, VALIDATION_AUDIT)); audit_scope(previous, full=False)
    required = common_files(p) | training_files() | validation_files()
    if any("test" in PurePosixPath(name).parts or PurePosixPath(name).name.startswith(("test_", "test.", "annual")) for name in previous.get("input_sha256", {})):
        raise ValueError("Original validation audit cannot bind later DroQ annual evidence")
    hashes(root, previous.get("input_sha256"), "Independent24 validation inputs", required=required)
    hashes(main, previous.get("main_input_sha256"), "Original validation references", required={"frozen_spec.json", "runtime.json"})
    hashes(source_root, previous.get("source_sha256"), "Independent validation implementation", required=SOURCE_HASHES)
    if previous.get("source_sha256") != SOURCE_HASHES or previous.get("dataset_sha256") != p["dataset_sha256"]:
        raise ValueError("Independent validation source/data identity changed")
    ids = {f"validation__{cost}__{family}__droq__balanced__{key}" for family in FAMILIES for cost in COSTS for key in (*map(str, SEEDS), "ensemble")}
    if set(previous.get("runs", {})) != ids or any(v.get("reconstructed") is not True or v.get("hours") != 7080 or v.get("repeat_exact") is not True for v in previous["runs"].values()):
        raise ValueError("Exactly24 distinct independently reconstructed repeated validation accounts required")
    training = read_json(safe_target(root, "training_completion.json")); records = training.get("records", [])
    if (training.get("model_count") != 5 or len(records) != 5 or training.get("protocol_sha256") != identity["protocol_sha256"]
            or {(v.get("profile"), v.get("seed")) for v in records} != {("balanced", seed) for seed in SEEDS}):
        raise ValueError("All five newly trained DroQ policies required")
    for record in records:
        folder = f"training/droq__balanced__seed{record['seed']}"
        if (read_json(safe_target(root, folder + "/record.json")) != record or record.get("algorithm") != "droq"
                or record.get("actual_timesteps") != 32768 or record.get("actual_counters") != COUNTS
                or record.get("hyperparameters") != RECIPE or record.get("protocol_sha256") != identity["protocol_sha256"]):
            raise ValueError("Actual trained DroQ finite budget or record identity changed")
        for leaf, path_key, hash_key in (("model_tensors.pt", "model_tensor_file", "model_tensor_sha256"),
                ("model.json", "model_manifest_file", "model_manifest_sha256"), ("training_timeout_transitions.jsonl", "timeout_trace_relative_path", "timeout_trace_sha256")):
            if record.get(path_key) != folder + "/" + leaf or record.get(hash_key) != sha(safe_target(root, folder + "/" + leaf)):
                raise ValueError("Pure Tensor checkpoint, manifest or actual timeout bytes changed")
        manifest = read_json(safe_target(root, folder + "/model.json"))
        if (manifest.get("parameter_sha256") != record.get("parameter_sha256") or manifest.get("tensor_sha256") != record["model_tensor_sha256"]
                or manifest.get("counters") != COUNTS):
            raise ValueError("Saved tensor identity and independent-audited counter metadata disagree")
    selection, seal = (read_json(safe_target(root, name)) for name in ("selection.json", "selection_receipt.json"))
    if (seal.get("selection_file") != "selection.json" or seal.get("selection_sha256") != sha(root / "selection.json")
            or seal.get("protocol_sha256") != selection.get("protocol_sha256") or selection.get("protocol_sha256") != identity["protocol_sha256"]
            or selection.get("audit_sha256") != sha(root / VALIDATION_AUDIT) or selection.get("seed_selection") is not False
            or selection.get("selection_uses_test") is not False or selection.get("test_results_available_to_selector") is not False
            or any(selection.get(flag) is not True for flag in FLAGS)):
        raise ValueError("Whole-file two-candidate validation choice must already be sealed")
    pool = selection.get("candidates", [])
    if (len(pool) != 2 or {(v.get("family"), v.get("profile")) for v in pool} != {(f, "balanced") for f in FAMILIES}
            or any(v.get("algorithm") != "droq" or type(v.get("score")) not in (int, float) or not math.isfinite(v["score"])
                or v.get("rebalance_band") != .025 or v.get("exposure_multiplier") != (1. if v.get("family") == "raw" else .75) for v in pool)
            or pool != sorted(pool, key=lambda v: (-v["score"], 0 if v["family"] == "raw" else 1, v["profile"])) or selection.get("candidate") != pool[0]):
        raise ValueError("Only two fixed budget candidates and sealed raw-first ties allowed")
    hashes(root, selection.get("training_source_sha256"), "Sealed five policies", required=training_files(started=False))
    hashes(root, selection.get("validation_source_sha256"), "Sealed24 validation accounts", required=validation_files())
    if not (utc(fit["completed_at"]) <= utc(validate["started_at"]) <= utc(validate["completed_at"]) <= utc(previous["audited_at"]) <= utc(selection["selected_at"]) <= utc(seal["sealed_at"])):
        raise ValueError("Fit, validation audit and sealed selection must occur in order")
    return previous, selection, seal


def terminal_guard(root, *, main, dataset, mode, source_root=ROOT, process_root=None, process_main=None, process_dataset=None, process_source=None):
    if mode not in ("validation_only", "full"):
        raise ValueError("Only explicitly separate validation_only or full snapshots supported")
    if any(transport.is_link(p) for p in (root, main)):
        raise ValueError("Original research roots cannot be links or junctions")
    root, main = Path(root).resolve(), Path(main).resolve()
    if mode == "validation_only":
        no_annual(root)
    p, identity = identities(root, main, dataset, source_root, snapshot=True)
    args = dict(process_root=process_root or root, process_main=process_main or main,
        process_dataset=process_dataset or Path(dataset).resolve(), process_source=process_source or Path(source_root).resolve())
    fit, _ = phase_guard(root, "fit", p, **args); validate, _ = phase_guard(root, "validate", p, **args)
    previous, selection, seal = original_validation(root, main, source_root, p, identity, fit, validate)
    audit_file, audit = VALIDATION_AUDIT, previous
    protected = {"protocol.json", "protocol_receipt.json", VALIDATION_AUDIT, "selection.json", "selection_receipt.json", "fit_process.json", "validate_process.json"}
    identity.update(mode=mode, scope="droq_validation24" if mode == "validation_only" else "droq_annual26_with_original_validation24",
        validation_audit_sha256=sha(root / VALIDATION_AUDIT), selection_sha256=sha(root / "selection.json"),
        training_models=5, original_validation_accounts=24, validation_repeats=24, financial_recomputation=False, fresh_holdout_success=False)
    if mode == "full":
        audit_file = "independent_audit.json"; audit = read_json(safe_target(root, audit_file)); audit_scope(audit, full=True)
        auth, marker = (read_json(safe_target(root, name)) for name in ("annual_authorization.json", "test_started.json"))
        if (auth.get("annual_evaluation_authorized") is not True or auth.get("no_live_trading") is not True or auth.get("financial_parameters_fixed") is not True
                or auth.get("selection_sha256") != sha(root / "selection.json") or any(auth.get(flag) is not True for flag in FLAGS)
                or marker.get("protocol_sha256") != identity["protocol_sha256"] or marker.get("selection_sha256") != sha(root / "selection.json")
                or marker.get("authorization_sha256") != sha(root / "annual_authorization.json") or not (utc(seal["sealed_at"]) <= utc(auth["authorized_at"]) <= utc(marker["at"]))):
            raise ValueError("Separate exact-candidate annual authorization must follow the validation seal")
        process, complete = phase_guard(root, "test", p, **args); chosen = selection["candidate"]
        if complete.get("candidate") != chosen or complete.get("selection_sha256") != sha(root / "selection.json") or utc(marker["at"]) > utc(process["started_at"]):
            raise ValueError("Only the separately authorized selected annual account set may be transported")
        required = common_files(p) | training_files() | validation_files() | group_files("test", chosen["family"]) | {
            VALIDATION_AUDIT, "selection.json", "selection_receipt.json", "annual_authorization.json", "test_started.json", "test_process.json", "test_completion.json", "test.log", "preflight/test.json"}
        hashes(root, audit.get("input_sha256"), "Full26 study inputs", required=required)
        hashes(main, audit.get("main_input_sha256"), "Original14 reference accounts", required=baseline_files() | {"frozen_spec.json", "runtime.json", "independent_audit.json"})
        hashes(source_root, audit.get("source_sha256"), "Full independent implementation", required=SOURCE_HASHES)
        if audit.get("source_sha256") != SOURCE_HASHES or audit.get("dataset_sha256") != p["dataset_sha256"] or utc(audit["audited_at"]) < utc(process["completed_at"]):
            raise ValueError("Full annual audit source, data or execution identity changed")
        ids = {f"test__{cost}__{chosen['family']}__droq__balanced__{key}" for cost in COSTS for key in (*map(str, SEEDS), "ensemble")}
        baseline_ids = {f"test__{cost}__{name}" for cost in COSTS for name in BASELINES}
        if (set(audit.get("runs", {})) != ids or set(audit.get("baseline_runs", {})) != baseline_ids
                or any(v.get("reconstructed") is not True or v.get("hours") != 8760 or v.get("repeat_exact") is not True for v in audit["runs"].values())
                or any(v.get("reconstructed") is not True or v.get("hours") != 8760 for v in audit["baseline_runs"].values())):
            raise ValueError("Full26 means12 new repeated accounts plus14 original references")
        original = read_json(safe_target(main, "independent_audit.json"))
        if (original.get("passed") is not True or original.get("status") != "passed" or original.get("test_complete") is not True
                or original.get("run_count") != 172 or original.get("failures") != []
                or any(not v.get("fingerprints") or v.get("fingerprints") != original.get("runs", {}).get(k, {}).get("fingerprints") for k, v in audit["baseline_runs"].items())):
            raise ValueError("Baseline identities must remain bound to the original complete172 audit")
        hashes(main, original.get("input_sha256"), "Original172 indirect completion chain", required=baseline_files() | {"frozen_spec.json", "test_completion.json"})
        research = audit.get("research_targets", {}); gates = research.get("gates", {})
        numbers = ("base_net_return", "base_max_drawdown", "stress_net_return", "stress_max_drawdown", "base_fees_usdt", "stress_fees_usdt")
        members, trends = research.get("base_seed_returns", []), research.get("beats_simple_trend_controls", {})
        if (research.get("candidate") != {k: chosen[k] for k in ("algorithm", "family", "profile")} or set(gates) != GATES
                or any(type(v) is not bool for v in gates.values()) or type(research.get("goal_performance_achieved")) is not bool
                or research["goal_performance_achieved"] != all(gates.values()) or research.get("all_five_diagnostic_targets_met") is not research["goal_performance_achieved"]
                or research.get("fresh_holdout_success") is not False or research.get("training_budget_multiple") != 1
                or research.get("recipe_is_pure_official_default") is not False or research.get("accounting_passage_is_not_goal_success") is not True
                or any(type(research.get(k)) not in (int, float) or not math.isfinite(research[k]) for k in numbers)
                or type(research.get("positive_base_training_seeds")) is not int or not 0 <= research["positive_base_training_seeds"] <= 5
                or len(members) != 5 or {r.get("seed") for r in members} != set(SEEDS)
                or any(type(r.get("net_return")) not in (int, float) or not math.isfinite(r["net_return"]) for r in members)
                or set(trends) != {"sma200_trend", "momentum180"} or any(type(v) is not bool for v in trends.values())
                or type(research.get("stronger_control_comparison_met")) is not bool
                or any(research.get(flag) is not True for flag in FLAGS)):
            raise ValueError("Own-DroQ target declarations and already-seen-year disclosure required; either outcome is valid")
        goal = read_json(safe_target(root, "goal_candidate_evidence.json"))
        if (goal.get("protocol_sha256") != identity["protocol_sha256"] or goal.get("selection_sha256") != sha(root / "selection.json")
                or goal.get("audit_sha256") != sha(root / audit_file) or goal.get("accounting_audit_passed") is not True
                or utc(goal["at"]) < utc(audit["audited_at"]) or any(goal.get(k) != v for k, v in research.items())):
            raise ValueError("Native runner goal evidence must preserve the exact full audit and its outcome")
        identity.update(new_test_accounts=12, baseline_reference_accounts=14, test_repeats=12,
            goal_evidence_sha256=sha(root / "goal_candidate_evidence.json"), goal_performance_achieved=research["goal_performance_achieved"])
        protected |= {"annual_authorization.json", "test_started.json", "test_process.json", "test_completion.json", "test.log", audit_file, "goal_candidate_evidence.json"}
    identity.update(audit_file=audit_file, audit_sha256=sha(root / audit_file), audit_input_sha256=audit["input_sha256"],
        main_input_sha256=audit["main_input_sha256"], audit_source_sha256=audit["source_sha256"],
        terminal_proof_sha256={name: sha(safe_target(root, name)) for name in sorted(protected)})
    return identity


def export_snapshot(root, *, main, dataset, mode, source_root=ROOT, export_parent=None):
    if transport.is_link(root):
        raise ValueError("Original snapshot root cannot be linked")
    root = Path(root).resolve(); first = terminal_guard(root, main=main, dataset=dataset, mode=mode, source_root=source_root)
    before = transport.inventory(root); parent = Path(export_parent or root.parent / "btceth_rl_droq_exports")
    if transport.is_link(parent) or any(parent.resolve().is_relative_to(bound) for bound in (root, Path(main).resolve())):
        raise ValueError("Exports must remain outside both research trees")
    parent = parent.resolve(); parent.mkdir(parents=True, exist_ok=True)
    token = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex
    prefix = "validation-" if mode == "validation_only" else "full-"
    destination = safe_target(parent, prefix + "/" + token); destination.parent.mkdir(exist_ok=True); destination.mkdir(exist_ok=False)
    archive = destination / "experiment_snapshot.tar.gz"
    try:
        with archive.open("xb") as raw, gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as compressed:
            with tarfile.open(mode="w", fileobj=compressed, format=tarfile.PAX_FORMAT) as tar:
                for item in before:
                    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
                    with os.fdopen(os.open(safe_target(root, item["path"]), flags), "rb") as stream:
                        st = os.fstat(stream.fileno())
                        if not stat.S_ISREG(st.st_mode) or st.st_size != item["size"]:
                            raise RuntimeError("Research source changed during snapshot")
                        info = tarfile.TarInfo(item["path"]); info.size, info.mode, info.mtime = item["size"], 0o600, 0
                        info.uid = info.gid = 0; info.uname = info.gname = ""; tar.addfile(info, stream)
        if transport.inventory(root) != before or terminal_guard(root, main=main, dataset=dataset, mode=mode, source_root=source_root) != first or sha(_transport_path) != TRANSPORT_SHA256:
            raise RuntimeError("Research evidence or reviewed transport changed during snapshot")
        manifest = dict(schema_version=1, created_at=transport.stamp(), mode=mode, remote_root=str(root), files=before,
            file_count=len(before), total_bytes=sum(v["size"] for v in before), terminal_guard=first,
            archive_name=archive.name, archive_sha256=sha(archive), archive_size=archive.stat().st_size)
        path = destination / "snapshot_manifest.json"; transport.write_new_json(path, manifest)
        return dict(status="complete", mode=mode, remote_root=str(root), export_directory=str(destination), archive_path=str(archive), manifest_path=str(path),
            archive_sha256=manifest["archive_sha256"], archive_size=manifest["archive_size"], manifest_sha256=sha(path),
            protocol_sha256=first["protocol_sha256"], frozen_spec_sha256=first["frozen_spec_sha256"], audit_file=first["audit_file"], audit_sha256=first["audit_sha256"], reused_transport_sha256=TRANSPORT_SHA256)
    except BaseException as error:
        transport.write_new_json(destination / "export_failure.json", dict(at=transport.stamp(), error_type=type(error).__name__, error=str(error))); raise


def collect(*, output, reference_output, main_output, dataset_path, mode, source_root=ROOT,
        alias=transport.DEFAULT_ALIAS, remote_python=transport.DEFAULT_REMOTE_PYTHON, remote_root=REMOTE_ROOT, remote_main=REMOTE_MAIN,
        remote_dataset=REMOTE_DATASET, remote_source=REMOTE_SOURCE, remote_transport=REMOTE_TRANSPORT, export_parent=None, staging=None):
    if mode not in ("validation_only", "full") or any(transport.is_link(p) for p in (output, reference_output, main_output)):
        raise ValueError("Explicit scope and ordinary nonlinked collection roots required")
    output, reference_output, main_output = map(lambda p: Path(p).resolve(), (output, reference_output, main_output))
    if any(output == b or output.is_relative_to(b) or b.is_relative_to(output) for b in (reference_output, main_output)):
        raise ValueError("Each output must be a separate non-nested sibling, preserving earlier receipts")
    if (output / "transfer_receipt.json").exists() and read_json(output / "transfer_receipt.json").get("mode") != mode:
        raise ValueError("A validation transfer receipt cannot become an annual receipt or vice versa")
    _, local = identities(reference_output, main_output, dataset_path, source_root)
    reference_before = transport.inventory(reference_output); old_receipt = reference_output / "transfer_receipt.json"
    if mode == "full" and not old_receipt.is_file():
        raise ValueError("Annual collection requires the retained original validation snapshot receipt")
    if old_receipt.exists():
        receipt = read_json(old_receipt)
        if receipt.get("mode") != "validation_only" or receipt.get("protocol_sha256") != local["protocol_sha256"] or receipt.get("audit_file") != VALIDATION_AUDIT:
            raise ValueError("Reference receipt must retain its original validation scope and protocol")
        hashes(reference_output, transport._receipt_map(receipt), "Retained validation snapshot receipt",
            required=training_files() | validation_files() | {"protocol.json", "protocol_receipt.json", VALIDATION_AUDIT, "selection.json", "selection_receipt.json"})
    token = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex
    staging = Path(staging or ROOT / "results/backtests/btceth_rl_droq_transfers" / token)
    if transport.is_link(staging) or staging.exists() or any(staging.resolve() == b or staging.resolve().is_relative_to(b) or b.is_relative_to(staging.resolve()) for b in (output, reference_output, main_output)):
        raise ValueError("New independent staging directory required")
    staging.mkdir(parents=True, exist_ok=False)
    if old_receipt.exists():
        shutil.copyfile(old_receipt, staging / "retained_validation_transfer_receipt.json")
    payload = Path(__file__).read_bytes(); parent = export_parent or str(PurePosixPath(str(remote_root)).parent / "btceth_rl_droq_exports")
    common = ["--mode", mode, "--remote-root", str(remote_root), "--remote-export-parent", str(parent), "--main-output", str(remote_main),
        "--dataset", str(remote_dataset), "--source-root", str(remote_source), "--transport-source", str(remote_transport)]
    response = transport._remote_call(alias, remote_python, [*common, "--remote-export-worker"], payload)
    transport._validate_remote_export(response, remote_root, parent, mode == "validation_only")
    audit_file = VALIDATION_AUDIT if mode == "validation_only" else "independent_audit.json"
    if (response.get("protocol_sha256") != local["protocol_sha256"] or response.get("frozen_spec_sha256") != local["frozen_spec_sha256"]
            or response.get("audit_file") != audit_file or response.get("reused_transport_sha256") != TRANSPORT_SHA256):
        raise ValueError("Remote DroQ protocol, scope, audit or transport identity changed")
    transport.write_new_json(staging / "remote_export_receipt.json", response)
    manifest_path, archive = staging / "snapshot_manifest.json", staging / "experiment_snapshot.tar.gz"
    transport._remote_call(alias, remote_python, [*common, "--remote-read-file", response["manifest_path"]], payload, destination=manifest_path)
    if sha(manifest_path) != response["manifest_sha256"]:
        raise ValueError("Downloaded manifest SHA changed")
    manifest = read_json(manifest_path)
    if any(manifest.get(k) != response[k] for k in ("mode", "remote_root", "archive_sha256", "archive_size")) or manifest.get("terminal_guard", {}).get("audit_sha256") != response["audit_sha256"]:
        raise ValueError("Manifest differs from its unique export receipt")
    transport.manifest_entries(manifest)
    transport._remote_call(alias, remote_python, [*common, "--remote-read-file", response["archive_path"]], payload, destination=archive)
    extracted = staging / "verified_files"; entries = transport.verify_extract(archive, manifest, extracted)
    proof = terminal_guard(extracted, main=main_output, dataset=dataset_path, mode=mode, source_root=source_root,
        process_root=remote_root, process_main=remote_main, process_dataset=remote_dataset, process_source=remote_source)
    if proof != manifest["terminal_guard"] or proof["protocol_sha256"] != local["protocol_sha256"] or "transfer_receipt.json" in entries or transport.inventory(reference_output) != reference_before:
        raise ValueError("Snapshot changed, nested a receipt or altered original validation evidence")
    if old_receipt.exists() and read_json(old_receipt).get("audit_sha256") != proof["validation_audit_sha256"]:
        raise ValueError("Retained validation receipt binds another initial audit")
    if identities(reference_output, main_output, dataset_path, source_root)[1] != local or sha(_transport_path) != TRANSPORT_SHA256:
        raise RuntimeError("Prepared identities or safe transport changed before merge")
    merged = transport.merge_verified_files(extracted, entries, output)
    receipt = dict(schema_version=1, collected_at=transport.stamp(), mode=mode, scope=proof["scope"], remote_alias=alias, remote_root=str(remote_root),
        remote_export_directory=response["export_directory"], local_staging_directory=str(staging), archive_sha256=manifest["archive_sha256"], archive_size=manifest["archive_size"],
        manifest_sha256=response["manifest_sha256"], collector_source_sha256=hashlib.sha256(payload).hexdigest(), reused_transport_sha256=TRANSPORT_SHA256,
        protocol_sha256=local["protocol_sha256"], audit_file=audit_file, audit_sha256=proof["audit_sha256"], terminal_guard=proof,
        reference_validation_output=str(reference_output), retained_validation_transfer_receipt_sha256=sha(old_receipt) if old_receipt.exists() else None,
        exactfiles=[dict(path=name, sha256=item["sha256"]) for name, item in sorted(entries.items())], file_count=len(entries), total_bytes=manifest["total_bytes"],
        no_financial_recomputation=True, remote_audits_preserved=True, local_audits_must_use_separate_names=True, **merged)
    transport.write_new_json(staging / "collection_receipt.json", receipt)
    if not merged["receipt_reused"]:
        transport.write_new_json(output / "transfer_receipt.json", receipt)
    return dict(status="collected", mode=mode, scope=proof["scope"], output_directory=str(output), staging_directory=str(staging),
        file_count=len(entries), goal_performance_achieved=proof.get("goal_performance_achieved"))


def stream_generated(path, parent, mode):
    path, parent = Path(path), Path(parent); prefix = "validation-" if mode == "validation_only" else "full-"
    if (path.name not in ("snapshot_manifest.json", "experiment_snapshot.tar.gz") or not re.fullmatch(r"\d{8}T\d{6}Z-[0-9a-f]{32}", path.parent.name)
            or path.parent.parent.name != prefix or path.parent.parent.parent.resolve() != parent.resolve()):
        raise ValueError("Only uniquely generated artifacts of the requested snapshot scope may be streamed")
    with safe_target(parent, path.relative_to(parent).as_posix()).open("rb") as stream:
        shutil.copyfileobj(stream, sys.stdout.buffer, length=1024 * 1024)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("validation_only", "full"), required=True)
    for name in ("output-dir", "reference-output-dir", "staging-dir"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--main-output", type=Path, default=ROOT / "results/backtests/btceth_rl_20261029")
    parser.add_argument("--dataset", type=Path, default=transport.DEFAULT_DATASET)
    parser.add_argument("--source-root", type=Path, default=ROOT)
    parser.add_argument("--transport-source", type=Path, default=_transport_path)
    parser.add_argument("--ssh-alias", default=transport.DEFAULT_ALIAS)
    parser.add_argument("--remote-python", default=transport.DEFAULT_REMOTE_PYTHON)
    for name, default in (("root", REMOTE_ROOT), ("main", REMOTE_MAIN), ("dataset", REMOTE_DATASET), ("source", REMOTE_SOURCE), ("transport", REMOTE_TRANSPORT)):
        parser.add_argument("--remote-" + name, default=default)
    parser.add_argument("--remote-export-parent")
    parser.add_argument("--remote-export-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--remote-read-file", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.remote_read_file:
        if args.remote_export_parent is None:
            parser.error("An explicit generated export parent is required")
        stream_generated(args.remote_read_file, args.remote_export_parent, args.mode); return
    if args.remote_export_worker:
        result = export_snapshot(args.remote_root, main=args.main_output, dataset=args.dataset, source_root=args.source_root, mode=args.mode, export_parent=args.remote_export_parent)
    else:
        base = ROOT / "results/backtests"
        output = args.output_dir or base / ("btceth_rl_droq_20261030_validation_snapshot" if args.mode == "validation_only" else "btceth_rl_droq_20261030_annual_snapshot")
        reference = args.reference_output_dir or base / ("btceth_rl_droq_20261030" if args.mode == "validation_only" else "btceth_rl_droq_20261030_validation_snapshot")
        result = collect(output=output, reference_output=reference, main_output=args.main_output, dataset_path=args.dataset, source_root=args.source_root,
            mode=args.mode, alias=args.ssh_alias, remote_python=args.remote_python, remote_root=args.remote_root, remote_main=args.remote_main,
            remote_dataset=args.remote_dataset, remote_source=args.remote_source, remote_transport=args.remote_transport, export_parent=args.remote_export_parent, staging=args.staging_dir)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()

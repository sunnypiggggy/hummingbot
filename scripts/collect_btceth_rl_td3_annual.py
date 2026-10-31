"""SHA-only transport for a separately authorized, fully audited TD3 year.

No dispatch, trading, inference, accounting or performance calculation occurs
here.  Full26 means twelve new TD3 accounts and fourteen original baseline
references, with their original48 validation and ten policies preserved.
Validation receipts stay in their original tree; the annual output is a new
non-nested sibling.  A failed research goal is a valid transportable result.
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
VALIDATION_COLLECTOR_SHA256 = "0dea90837743edc53cd90dabbbdf718442d2f3e9522d7f8c0d4a626766e2912e"
REMOTE_VALIDATION_COLLECTOR = "/home/sunny/research/btceth_rl_td3_20261030_preflight/transport_dependencies/collect_btceth_rl_td3_validation_0dea9083.py"


def load_validation_helpers(path):
    path = Path(path)
    if path.is_symlink() or hasattr(path, "is_junction") and path.is_junction():
        raise ValueError("Reviewed validation helper cannot be a link")
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != VALIDATION_COLLECTOR_SHA256:
        raise ValueError("Use the reviewed0dea validation metadata helpers")
    module = types.ModuleType("_td3_annual_frozen_validation_helpers")
    module.__file__ = str(path.resolve())
    exec(compile(content, module.__file__, "exec"), module.__dict__)
    return module


_validation_path = Path(sys.argv[sys.argv.index("--validation-collector-source") + 1]) if "--validation-collector-source" in sys.argv else ROOT / "scripts/collect_btceth_rl_td3_validation.py"
validation = load_validation_helpers(_validation_path)
transport = validation.transport
sha, read_json, safe_target, hashes, utc = validation.sha, validation.read_json, validation.safe_target, validation.hashes, validation.utc
SOURCE_NAMES, FLAGS = validation.SOURCE_NAMES, validation.FLAGS
BASELINES = ("cash", "btc_hold", "eth_hold", "equal_hold", "equal_rebalance", "sma200_trend", "momentum180")
GATES = {"base_return_at_least_5pct", "base_max_drawdown_at_most_20pct", "stress_return_nonnegative", "at_least_three_positive_training_seeds", "beats_equal_hold_return_and_drawdown"}
TARGETS = {"primary_base_net_return_at_least": .05, "primary_base_max_drawdown_magnitude_at_most": .20,
    "primary_stress_net_return_at_least": 0., "primary_beats_equal_hold_return_and_drawdown": True,
    "positive_base_training_seeds_at_least": 3, "stronger_claim_requires_beating_simple_trend_controls": True, "targets_are_not_guarantees": True}


def group_files(family, profile):
    folder = f"test/{family}/{profile}"
    return {*(f"{folder}/{n}" for n in validation.GROUP_FILES), *(f"{folder}/{cost}/{key}/{n}" for cost in ("base", "stress") for key in (*map(str, range(5)), "ensemble") for n in validation.CASE_FILES)}


def baseline_files():
    folder = "evaluation/test/baselines"
    return {*(f"{folder}/{n}" for n in validation.GROUP_FILES), *(f"{folder}/{cost}/{key}/{n}" for cost in ("base", "stress") for key in BASELINES for n in validation.CASE_FILES[:6])}


def _audit(value, *, full):
    fixed = dict(passed=True, status="passed" if full else "partial_test_pending", phase="test" if full else "validation",
        validation_complete=True, test_complete=full, training_model_count=10, validation_run_count=48,
        test_run_count=12 if full else 0, baseline_run_count=14 if full else 0, run_count=26 if full else 48,
        repeat_count=12 if full else 48, failure_count=0, failures=[], fresh_holdout_success=False, training_budget_multiple=2,
        accounting_passage_is_not_goal_success=True, **{flag: True for flag in FLAGS})
    if (any(value.get(k) != expected or isinstance(expected, bool) and value.get(k) is not expected for k, expected in fixed.items())
            or type(value.get("checks_total")) is not int or value["checks_total"] <= 0
            or value.get("declared_research_targets") != TARGETS):
        raise ValueError("Independent710 complete annual26 or original partial48 scope required")


def _original_validation(root, main, source_root, p, identity, fit, validate):
    previous = read_json(safe_target(root, validation.AUDIT_FILE)); _audit(previous, full=False)
    required = validation.validation_files() | validation.training_files() | {"protocol.json", "protocol_receipt.json", "fit_process.json", "fit_completion.json", "fit.log", "preflight/fit.json", "preflight/validate.json"}
    if any("test" in PurePosixPath(name).parts or PurePosixPath(name).name.startswith(("test_", "test.", "annual")) for name in previous.get("input_sha256", {})):
        raise ValueError("Original validation audit cannot bind later TD3 annual data")
    hashes(root, previous.get("input_sha256"), "Original independent48 validation bytes", required=required)
    hashes(main, previous.get("main_input_sha256"), "Original validation reference dependencies", required={"frozen_spec.json", "runtime.json"})
    hashes(source_root, previous.get("source_sha256"), "Original validation audit implementation", required=SOURCE_NAMES)
    if previous.get("source_sha256") != p["source_sha256"] or previous.get("dataset_sha256") != p["dataset_sha256"]:
        raise ValueError("Original validation source/data identity changed")
    ids = {f"validation__{cost}__{family}__td3__{profile}__{key}" for family in validation.FAMILIES for profile in validation.PROFILES for cost in ("base", "stress") for key in (*map(str, range(5)), "ensemble")}
    if set(previous.get("runs", {})) != ids or any(r.get("reconstructed") is not True or r.get("hours") != 7080 or r.get("repeat_exact") is not True for r in previous["runs"].values()):
        raise ValueError("All48 original validation accounts and repeats must remain reconstructed")
    training = read_json(safe_target(root, "training_completion.json")); records = training.get("records", [])
    if (training.get("model_count") != 10 or training.get("protocol_sha256") != identity["protocol_sha256"] or len(records) != 10
            or {(r.get("profile"), r.get("seed")) for r in records} != {(profile, seed) for profile in validation.PROFILES for seed in range(5)}):
        raise ValueError("All ten original trained policies must remain present")
    for r in records:
        folder = f"training/td3__{r['profile']}__seed{r['seed']}"
        if (read_json(safe_target(root, folder + "/record.json")) != r or r.get("algorithm") != "td3" or r.get("actual_timesteps") != 65536
                or r.get("actual_n_updates") != 65024 or r.get("actor_update_count") != 32512 or r.get("protocol_sha256") != identity["protocol_sha256"]):
            raise ValueError("Original ten-model TD3 budget or record changed")
        for name, key, relative_key in (("model.zip", "model_sha256", "model_relative_path"), ("training_timeout_transitions.jsonl", "timeout_trace_sha256", "timeout_trace_relative_path")):
            if r.get(relative_key) != folder + "/" + name or r.get(key) != sha(safe_target(root, folder + "/" + name)):
                raise ValueError("Original TD3 checkpoint/timeout bytes changed")
    selected, seal = (read_json(safe_target(root, name)) for name in ("selection.json", "selection_receipt.json"))
    if (seal.get("selection_file") != "selection.json" or seal.get("selection_sha256") != sha(safe_target(root, "selection.json"))
            or seal.get("protocol_sha256") != selected.get("protocol_sha256") or selected.get("protocol_sha256") != identity["protocol_sha256"]
            or selected.get("audit_sha256") != sha(safe_target(root, validation.AUDIT_FILE)) or selected.get("seed_selection") is not False
            or selected.get("selection_uses_test") is not False or selected.get("test_results_available_to_selector") is not False
            or any(selected.get(flag) is not True for flag in FLAGS)):
        raise ValueError("Original validation-only four-candidate seal must remain immutable")
    pool = selected.get("candidates", [])
    if (len(pool) != 4 or {(r.get("family"), r.get("profile")) for r in pool} != {(f, p) for f in validation.FAMILIES for p in validation.PROFILES}
            or any(r.get("algorithm") != "td3" or type(r.get("score")) not in (int, float) or not math.isfinite(r["score"]) for r in pool)
            or pool != sorted(pool, key=lambda r: (-r["score"], 0 if r["family"] == "raw" else 1, r["profile"])) or selected.get("candidate") != pool[0]):
        raise ValueError("Four fixed TD3 candidates and original raw-first tie choice required")
    hashes(root, selected.get("training_source_sha256"), "Sealed ten models", required=validation.training_files(include_started=False))
    hashes(root, selected.get("validation_source_sha256"), "Sealed48 validation accounts", required=validation.validation_files())
    if not (utc(fit["completed_at"]) <= utc(validate["started_at"]) <= utc(validate["completed_at"]) <= utc(previous["audited_at"]) <= utc(selected["selected_at"]) <= utc(seal["sealed_at"])):
        raise ValueError("Original fit, validation audit and sealed choice time chain changed")
    return previous, selected, seal


def _test_process(root, p, *, process_root, process_main, process_dataset, process_source):
    process, complete, preflight = (read_json(safe_target(root, name)) for name in ("test_process.json", "test_completion.json", "preflight/test.json"))
    command = process.get("command", [])
    if (process.get("phase") != "test" or process.get("status") != "completed" or type(process.get("exit_code")) is not int or process["exit_code"] != 0
            or process.get("source_sha256") != p["source_sha256"]["scripts/btceth_rl_td3.py"]
            or process.get("protocol_sha256") != sha(safe_target(root, "protocol.json")) or complete.get("protocol_sha256") != process["protocol_sha256"]
            or process.get("log_relative_path") != "test.log" or process.get("log_sha256") != sha(safe_target(root, "test.log"))
            or complete.get("case_count") != 12 or complete.get("repeat_count") != 12
            or any(type(process.get(k)) is not int or process[k] <= 0 for k in ("supervisor_pid", "child_pid")) or process["supervisor_pid"] == process["child_pid"]
            or len(command) < 5 or str(command[2]) != str(PurePosixPath(str(process_source)) / "scripts/btceth_rl_td3.py")):
        raise ValueError("Actual source-bound completed0 test12+12 process/log required")
    for option, expected in {"--phase": "_test", "--reference-main": str(process_main), "--dataset": str(process_dataset), "--output-dir": str(process_root)}.items():
        if command.count(option) != 1 or command.index(option) + 1 >= len(command) or command[command.index(option) + 1] != expected:
            raise ValueError("Annual command belongs to another study: " + option)
    receipt = read_json(safe_target(root, "protocol_receipt.json"))
    if not (utc(receipt["sealed_at"]) <= utc(preflight["at"]) <= utc(process["started_at"]) <= utc(complete["at"]) <= utc(process["completed_at"])):
        raise ValueError("Annual process freeze/start/terminal time chain changed")
    if sys.platform == "linux":
        for pid in (process["supervisor_pid"], process["child_pid"]):
            try:
                state = (Path("/proc") / str(pid) / "stat").read_text().rsplit(")", 1)[1].split()[0]
                argv = (Path("/proc") / str(pid) / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
            except (FileNotFoundError, ProcessLookupError):
                continue
            if state != "Z" and "btceth_rl_td3.py" in argv:
                raise ValueError("Annual TD3 process is still live")
    return process, complete


def terminal_guard(root, *, main, dataset, source_root=ROOT, process_root=None, process_main=None, process_dataset=None, process_source=None):
    if transport.is_link(root) or transport.is_link(main):
        raise ValueError("Original annual research roots cannot be links or junctions")
    root, main = Path(root).resolve(), Path(main).resolve()
    p, identity = validation.identities(root, main, dataset, source_root, require_snapshot=True)
    if p.get("research_targets") != TARGETS or read_json(safe_target(main, "frozen_spec.json")).get("research_targets") != TARGETS:
        raise ValueError("The original fixed research thresholds must remain unchanged")
    args = dict(process_root=process_root or root, process_main=process_main or main, process_dataset=process_dataset or Path(dataset).resolve(), process_source=process_source or Path(source_root).resolve())
    fit, validate = (validation._phase(root, phase, p, **args) for phase in ("fit", "validate"))
    previous, selected, seal = _original_validation(root, main, source_root, p, identity, fit, validate)
    auth, marker = (read_json(safe_target(root, name)) for name in ("annual_authorization.json", "test_started.json"))
    if (auth.get("annual_evaluation_authorized") is not True or auth.get("no_live_trading") is not True or auth.get("financial_parameters_fixed") is not True
            or auth.get("selection_sha256") != sha(safe_target(root, "selection.json")) or any(auth.get(flag) is not True for flag in FLAGS)
            or marker.get("protocol_sha256") != identity["protocol_sha256"] or marker.get("selection_sha256") != sha(safe_target(root, "selection.json"))
            or marker.get("authorization_sha256") != sha(safe_target(root, "annual_authorization.json"))
            or not (utc(seal["sealed_at"]) <= utc(auth["authorized_at"]) <= utc(marker["at"]))):
        raise ValueError("Separate root annual authorization must follow selection and precede marker")
    test, completion = _test_process(root, p, **args)
    if (completion.get("candidate") != selected["candidate"] or completion.get("selection_sha256") != sha(safe_target(root, "selection.json"))
            or utc(marker["at"]) > utc(test["started_at"])):
        raise ValueError("Annual12 completion must use the pre-authorized frozen candidate")
    audit = read_json(safe_target(root, "independent_audit.json")); _audit(audit, full=True)
    chosen = selected["candidate"]
    required = validation.validation_files() | validation.training_files() | group_files(chosen["family"], chosen["profile"]) | {
        "protocol.json", "protocol_receipt.json", validation.AUDIT_FILE, "selection.json", "selection_receipt.json", "annual_authorization.json", "test_started.json",
        "fit_process.json", "fit.log", "fit_completion.json", "preflight/fit.json", "preflight/validate.json", "test_process.json", "test_completion.json", "test.log", "preflight/test.json"}
    hashes(root, audit.get("input_sha256"), "Full26 study byte dependencies", required=required)
    hashes(main, audit.get("main_input_sha256"), "Original14 baseline references", required=baseline_files() | {"frozen_spec.json", "runtime.json", "independent_audit.json"})
    hashes(source_root, audit.get("source_sha256"), "Full710 implementation", required=SOURCE_NAMES)
    if audit.get("source_sha256") != p["source_sha256"] or audit.get("dataset_sha256") != p["dataset_sha256"] or utc(audit["audited_at"]) < utc(test["completed_at"]):
        raise ValueError("Full annual audit source/data/completion identity changed")
    ids = {f"test__{cost}__{chosen['family']}__td3__{chosen['profile']}__{key}" for cost in ("base", "stress") for key in (*map(str, range(5)), "ensemble")}
    baselines = {f"test__{cost}__{name}" for cost in ("base", "stress") for name in BASELINES}
    if (set(audit.get("runs", {})) != ids or set(audit.get("baseline_runs", {})) != baselines
            or any(r.get("reconstructed") is not True or r.get("hours") != 8760 or r.get("repeat_exact") is not True for r in audit["runs"].values())
            or any(r.get("reconstructed") is not True or r.get("hours") != 8760 for r in audit["baseline_runs"].values())):
        raise ValueError("Full26 requires twelve real repeated TD3 accounts plus fourteen original references")
    original = read_json(safe_target(main, "independent_audit.json"))
    if (original.get("passed") is not True or original.get("status") != "passed" or original.get("test_complete") is not True or original.get("run_count") != 172 or original.get("failures") != []
            or any(original.get("runs", {}).get(key, {}).get("fingerprints") != proof.get("fingerprints") or not proof.get("fingerprints") for key, proof in audit["baseline_runs"].items())):
        raise ValueError("Baseline references must preserve their original172 audit identities")
    # Follow the original audit to its completion bytes;710 binds that audit,
    # rather than claiming it directly read the old global completion marker.
    hashes(main, original.get("input_sha256"), "Original172 immutable completion", required={"frozen_spec.json", "test_completion.json"} | baseline_files())
    research = audit.get("research_targets", {}); gates = research.get("gates", {})
    candidate = {k: chosen[k] for k in ("algorithm", "family", "profile")}
    numbers = ("base_net_return", "base_max_drawdown", "stress_net_return", "stress_max_drawdown", "base_fees_usdt", "stress_fees_usdt")
    members = research.get("base_seed_returns", [])
    trends = research.get("beats_simple_trend_controls", {})
    if (research.get("candidate") != candidate or set(gates) != GATES or any(type(v) is not bool for v in gates.values())
            or type(research.get("goal_performance_achieved")) is not bool or research["goal_performance_achieved"] != all(gates.values())
            or research.get("all_five_diagnostic_targets_met") is not research["goal_performance_achieved"]
            or research.get("fresh_holdout_success") is not False or research.get("training_budget_multiple") != 2
            or research.get("recipe_is_pure_official_default") is not False or research.get("accounting_passage_is_not_goal_success") is not True
            or any(type(research.get(k)) not in (int, float) or not math.isfinite(research[k]) for k in numbers)
            or type(research.get("positive_base_training_seeds")) is not int or not 0 <= research["positive_base_training_seeds"] <= 5
            or len(members) != 5 or {r.get("seed") for r in members} != set(range(5))
            or any(type(r.get("net_return")) not in (int, float) or not math.isfinite(r["net_return"]) for r in members)
            or set(trends) != {"sma200_trend", "momentum180"} or any(type(v) is not bool for v in trends.values())
            or type(research.get("stronger_control_comparison_met")) is not bool
            or any(research.get(flag) is not True for flag in FLAGS)):
        raise ValueError("Original own-TD3 target and known-year declarations required; either goal outcome is allowed")
    goal = read_json(safe_target(root, "goal_candidate_evidence.json"))
    if (goal.get("protocol_sha256") != identity["protocol_sha256"] or goal.get("selection_sha256") != sha(safe_target(root, "selection.json"))
            or goal.get("audit_sha256") != sha(safe_target(root, "independent_audit.json")) or goal.get("accounting_audit_passed") is not True
            or utc(goal["at"]) < utc(audit["audited_at"]) or any(goal.get(k) != v for k, v in research.items())):
        raise ValueError("Native runner goal evidence must retain its original full-audit identity and outcome")
    protected = {"protocol.json", "protocol_receipt.json", validation.AUDIT_FILE, "selection.json", "selection_receipt.json", "annual_authorization.json", "test_started.json", "test_process.json", "test_completion.json", "test.log", "independent_audit.json", "goal_candidate_evidence.json"}
    identity.update(mode="full", scope="td3_annual26_with_original_validation48", audit_file="independent_audit.json", audit_sha256=sha(safe_target(root, "independent_audit.json")),
        validation_audit_sha256=sha(safe_target(root, validation.AUDIT_FILE)), selection_sha256=sha(safe_target(root, "selection.json")),
        goal_evidence_sha256=sha(safe_target(root, "goal_candidate_evidence.json")), goal_performance_achieved=research["goal_performance_achieved"],
        terminal_proof_sha256={name: sha(safe_target(root, name)) for name in sorted(protected)}, audit_input_sha256=audit["input_sha256"],
        main_input_sha256=audit["main_input_sha256"], audit_source_sha256=audit["source_sha256"], new_test_accounts=12, baseline_reference_accounts=14,
        training_models=10, original_validation_accounts=48, test_repeats=12, financial_recomputation=False, fresh_holdout_success=False)
    return identity


def export_snapshot(root, *, main, dataset, source_root=ROOT, export_parent=None):
    if transport.is_link(root):
        raise ValueError("Annual research root cannot be linked")
    root = Path(root).resolve(); first = terminal_guard(root, main=main, dataset=dataset, source_root=source_root); before = transport.inventory(root)
    parent = Path(export_parent or root.parent / "btceth_rl_td3_annual_exports")
    if transport.is_link(parent) or parent.resolve().is_relative_to(root) or parent.resolve().is_relative_to(Path(main).resolve()):
        raise ValueError("Annual exports must remain outside both research trees")
    parent = parent.resolve(); parent.mkdir(parents=True, exist_ok=True)
    token = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex
    destination = safe_target(parent, "full-/" + token); destination.parent.mkdir(exist_ok=True); destination.mkdir(exist_ok=False)
    archive = destination / "experiment_snapshot.tar.gz"
    try:
        with archive.open("xb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with tarfile.open(mode="w", fileobj=compressed, format=tarfile.PAX_FORMAT) as tar:
                for item in before:
                    with os.fdopen(os.open(safe_target(root, item["path"]), os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)), "rb") as stream:
                        st = os.fstat(stream.fileno())
                        if not stat.S_ISREG(st.st_mode) or st.st_size != item["size"]:
                            raise RuntimeError("Annual source changed before snapshot")
                        info = tarfile.TarInfo(item["path"]); info.size, info.mode, info.mtime = item["size"], 0o600, 0
                        info.uid = info.gid = 0; info.uname = info.gname = ""; tar.addfile(info, stream)
        if (transport.inventory(root) != before or terminal_guard(root, main=main, dataset=dataset, source_root=source_root) != first
                or sha(Path(transport.__file__)) != validation.TRANSPORT_SHA256 or sha(_validation_path) != VALIDATION_COLLECTOR_SHA256):
            raise RuntimeError("Annual evidence or reviewed transport changed during snapshot")
        manifest = dict(schema_version=1, created_at=transport.stamp(), mode="full", remote_root=str(root), file_count=len(before), total_bytes=sum(i["size"] for i in before), files=before,
            terminal_guard=first, archive_name=archive.name, archive_sha256=sha(archive), archive_size=archive.stat().st_size)
        manifest_path = destination / "snapshot_manifest.json"; transport.write_new_json(manifest_path, manifest)
        return dict(status="complete", mode="full", remote_root=str(root), export_directory=str(destination), archive_path=str(archive), manifest_path=str(manifest_path),
            archive_sha256=manifest["archive_sha256"], archive_size=manifest["archive_size"], manifest_sha256=sha(manifest_path), protocol_sha256=first["protocol_sha256"],
            frozen_spec_sha256=first["frozen_spec_sha256"], audit_file="independent_audit.json", audit_sha256=first["audit_sha256"], reused_transport_sha256=validation.TRANSPORT_SHA256,
            validation_helpers_sha256=VALIDATION_COLLECTOR_SHA256)
    except BaseException as error:
        transport.write_new_json(destination / "export_failure.json", dict(at=transport.stamp(), error_type=type(error).__name__, error=str(error)))
        raise


def collect(*, output, reference_output, main_output, dataset_path, source_root=ROOT, alias=transport.DEFAULT_ALIAS, remote_python=transport.DEFAULT_REMOTE_PYTHON,
            remote_root=validation.REMOTE_ROOT, remote_main=validation.REMOTE_MAIN, remote_dataset=validation.REMOTE_DATASET, remote_source=validation.REMOTE_SOURCE,
            remote_transport=validation.REMOTE_TRANSPORT, remote_validation_helpers=REMOTE_VALIDATION_COLLECTOR, export_parent=None, staging=None):
    if any(transport.is_link(p) for p in (output, reference_output, main_output)):
        raise ValueError("Annual collection roots cannot be links or junctions")
    output, reference_output, main_output = map(lambda p: Path(p).resolve(), (output, reference_output, main_output))
    if any(output == bound or output.is_relative_to(bound) or bound.is_relative_to(output) for bound in (reference_output, main_output)):
        raise ValueError("Annual output must be a new non-nested sibling, preserving validation receipts")
    if (output / "transfer_receipt.json").exists() and read_json(output / "transfer_receipt.json").get("mode") != "full":
        raise ValueError("An existing validation transfer receipt cannot become an annual receipt")
    _, local = validation.identities(reference_output, main_output, dataset_path, source_root)
    reference_before = transport.inventory(reference_output)
    old_receipt = reference_output / "transfer_receipt.json"
    if old_receipt.exists():
        value = read_json(old_receipt)
        if value.get("mode") != "validation_only" or value.get("protocol_sha256") != local["protocol_sha256"] or value.get("audit_file") != validation.AUDIT_FILE:
            raise ValueError("Original local validation receipt must retain its own scope and protocol")
    token = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex
    staging = Path(staging or ROOT / "results/backtests/btceth_rl_td3_annual_transfers" / token)
    if transport.is_link(staging) or staging.exists() or any(staging.resolve() == b or staging.resolve().is_relative_to(b) or b.is_relative_to(staging.resolve()) for b in (output, reference_output, main_output)):
        raise ValueError("New independent annual staging directory required")
    staging.mkdir(parents=True, exist_ok=False)
    if old_receipt.exists():
        shutil.copyfile(old_receipt, staging / "retained_validation_transfer_receipt.json")
    payload = Path(__file__).read_bytes(); parent = export_parent or str(PurePosixPath(remote_root).parent / "btceth_rl_td3_annual_exports")
    common = ["--remote-root", str(remote_root), "--remote-export-parent", str(parent), "--main-output", str(remote_main), "--dataset", str(remote_dataset), "--source-root", str(remote_source),
        "--transport-source", str(remote_transport), "--validation-collector-source", str(remote_validation_helpers)]
    response = transport._remote_call(alias, remote_python, [*common, "--remote-export-worker"], payload)
    transport._validate_remote_export(response, remote_root, parent, False)
    if (response.get("protocol_sha256") != local["protocol_sha256"] or response.get("frozen_spec_sha256") != local["frozen_spec_sha256"] or response.get("audit_file") != "independent_audit.json"
            or response.get("reused_transport_sha256") != validation.TRANSPORT_SHA256 or response.get("validation_helpers_sha256") != VALIDATION_COLLECTOR_SHA256):
        raise ValueError("Remote annual source/protocol/scope identity changed")
    transport.write_new_json(staging / "remote_export_receipt.json", response)
    manifest_path, archive = staging / "snapshot_manifest.json", staging / "experiment_snapshot.tar.gz"
    transport._remote_call(alias, remote_python, [*common, "--remote-read-file", response["manifest_path"]], payload, destination=manifest_path)
    if sha(manifest_path) != response["manifest_sha256"]:
        raise ValueError("Downloaded annual manifest SHA changed")
    manifest = read_json(manifest_path)
    if any(manifest.get(k) != response[k] for k in ("mode", "remote_root", "archive_sha256", "archive_size")) or manifest.get("terminal_guard", {}).get("audit_sha256") != response["audit_sha256"]:
        raise ValueError("Annual manifest differs from unique export receipt")
    transport.manifest_entries(manifest)
    transport._remote_call(alias, remote_python, [*common, "--remote-read-file", response["archive_path"]], payload, destination=archive)
    extracted = staging / "verified_files"; entries = transport.verify_extract(archive, manifest, extracted)
    proof = terminal_guard(extracted, main=main_output, dataset=dataset_path, source_root=source_root,
        process_root=remote_root, process_main=remote_main, process_dataset=remote_dataset, process_source=remote_source)
    if (proof != manifest["terminal_guard"] or proof["protocol_sha256"] != local["protocol_sha256"] or "transfer_receipt.json" in entries
            or transport.inventory(reference_output) != reference_before):
        raise ValueError("Annual snapshot differs, nests a receipt or changed original validation evidence")
    if old_receipt.exists() and read_json(old_receipt).get("audit_sha256") != proof["validation_audit_sha256"]:
        raise ValueError("Retained validation receipt binds another original validation audit")
    if validation.identities(reference_output, main_output, dataset_path, source_root)[1] != local or sha(Path(transport.__file__)) != validation.TRANSPORT_SHA256 or sha(_validation_path) != VALIDATION_COLLECTOR_SHA256:
        raise RuntimeError("Prepared source or helper bytes changed before annual merge")
    merged = transport.merge_verified_files(extracted, entries, output)
    receipt = dict(schema_version=1, collected_at=transport.stamp(), mode="full", scope="td3_annual26_with_original_validation48", remote_alias=alias, remote_root=str(remote_root),
        remote_export_directory=response["export_directory"], local_staging_directory=str(staging), archive_sha256=manifest["archive_sha256"], archive_size=manifest["archive_size"], manifest_sha256=response["manifest_sha256"],
        collector_source_sha256=hashlib.sha256(payload).hexdigest(), reused_transport_sha256=validation.TRANSPORT_SHA256, validation_helpers_sha256=VALIDATION_COLLECTOR_SHA256,
        protocol_sha256=local["protocol_sha256"], audit_file="independent_audit.json", audit_sha256=proof["audit_sha256"], terminal_guard=proof,
        reference_validation_output=str(reference_output), retained_validation_transfer_receipt_sha256=sha(old_receipt) if old_receipt.exists() else None,
        exactfiles=[dict(path=n, sha256=item["sha256"]) for n, item in sorted(entries.items())], file_count=len(entries), total_bytes=manifest["total_bytes"],
        new_test_accounts=12, baseline_reference_accounts=14, original_validation_accounts=48, training_models=10, no_financial_recomputation=True,
        remote_audits_preserved=True, local_audits_must_use_separate_names=True, **merged)
    transport.write_new_json(staging / "collection_receipt.json", receipt)
    if not merged["receipt_reused"]:
        transport.write_new_json(output / "transfer_receipt.json", receipt)
    return dict(status="collected", mode="full", scope=receipt["scope"], output_directory=str(output), staging_directory=str(staging), file_count=len(entries), goal_performance_achieved=proof["goal_performance_achieved"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/backtests/btceth_rl_td3_20261030_annual_snapshot")
    parser.add_argument("--reference-output-dir", type=Path, default=ROOT / "results/backtests/btceth_rl_td3_20261030")
    parser.add_argument("--main-output", type=Path, default=ROOT / "results/backtests/btceth_rl_20261029")
    parser.add_argument("--dataset", type=Path, default=transport.DEFAULT_DATASET)
    parser.add_argument("--source-root", type=Path, default=ROOT)
    parser.add_argument("--transport-source", type=Path, default=validation._transport_path)
    parser.add_argument("--validation-collector-source", type=Path, default=_validation_path)
    parser.add_argument("--ssh-alias", default=transport.DEFAULT_ALIAS)
    parser.add_argument("--remote-python", default=transport.DEFAULT_REMOTE_PYTHON)
    parser.add_argument("--remote-root", default=validation.REMOTE_ROOT)
    parser.add_argument("--remote-main", default=validation.REMOTE_MAIN)
    parser.add_argument("--remote-dataset", default=validation.REMOTE_DATASET)
    parser.add_argument("--remote-source", default=validation.REMOTE_SOURCE)
    parser.add_argument("--remote-transport", default=validation.REMOTE_TRANSPORT)
    parser.add_argument("--remote-validation-helpers", default=REMOTE_VALIDATION_COLLECTOR)
    parser.add_argument("--remote-export-parent")
    parser.add_argument("--staging-dir", type=Path)
    parser.add_argument("--remote-export-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--remote-read-file", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.remote_read_file:
        path, parent = Path(args.remote_read_file), Path(args.remote_export_parent)
        if (path.name not in ("snapshot_manifest.json", "experiment_snapshot.tar.gz") or not re.fullmatch(r"\d{8}T\d{6}Z-[0-9a-f]{32}", path.parent.name)
                or path.parent.parent.name != "full-" or path.parent.parent.parent.resolve() != parent.resolve()):
            raise ValueError("Only generated full annual snapshot artifacts may be streamed")
        path = safe_target(parent, path.relative_to(parent).as_posix())
        with path.open("rb") as stream:
            shutil.copyfileobj(stream, sys.stdout.buffer, length=1024 * 1024)
        return
    if args.remote_export_worker:
        result = export_snapshot(args.remote_root, main=args.main_output, dataset=args.dataset, source_root=args.source_root, export_parent=args.remote_export_parent)
    else:
        result = collect(output=args.output_dir, reference_output=args.reference_output_dir, main_output=args.main_output, dataset_path=args.dataset, source_root=args.source_root,
            alias=args.ssh_alias, remote_python=args.remote_python, remote_root=args.remote_root, remote_main=args.remote_main, remote_dataset=args.remote_dataset,
            remote_source=args.remote_source, remote_transport=args.remote_transport, remote_validation_helpers=args.remote_validation_helpers,
            export_parent=args.remote_export_parent, staging=args.staging_dir)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()

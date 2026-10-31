"""SHA-only transport for sealed TD3 validation, with no annual capability.

Uses the frozen original collector's path, inventory, extraction and merge
functions.  Archive orchestration is explicit here; no old module globals are
patched, and no financial, model, selector or reporting module is imported.
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
RUNNER_SHA256 = "63204eb6a71eb26d68f5d98c0f3d80a337f60bf41ee93e0cb973a7b0a78cecdf"
AUDITOR_SHA256 = "710746591766666dbd47b7c1c18d0f0a12e7cef04c3e7c36c1496c41184ae79b"
REMOTE_MAIN = "/home/sunny/research/btceth_rl_20261029/experiment"
REMOTE_ROOT = "/home/sunny/research/btceth_rl_td3_20261030"
REMOTE_SOURCE = "/home/sunny/research/btceth_rl_td3_20261030_preflight/formal_source"
REMOTE_DATASET = "/home/sunny/research/btceth_rl_20261029/data/crypto_rl_btceth_20261029/dataset.pkl.gz"
REMOTE_TRANSPORT = "/home/sunny/research/btceth_rl_20261029/scripts/collect_btceth_rl_results.py"
PROFILES, FAMILIES, SEEDS = ("return", "balanced"), ("raw", "cap075"), tuple(range(5))
FLAGS = ("designed_after_first_annual", "test_year_already_seen", "not_fresh_holdout")
FINANCIAL = {"scripts/backtest_btceth_rl.py", "scripts/btceth_rl_env.py", "scripts/crypto_rl_portfolio_env.py", "scripts/crypto_rl_reporting.py"}
SOURCE_NAMES = FINANCIAL | {"scripts/audit_btceth_rl.py", "scripts/audit_btceth_rl_exposure_cap.py", "scripts/audit_btceth_rl_execution_band.py", "scripts/btceth_rl_td3.py", "scripts/audit_btceth_rl_td3.py"}
CASE_FILES = ("equity_1h.csv.gz", "fills.csv", "weights.csv", "boundary_equity.csv", "metadata.json", "fingerprints.json", "member_targets.csv", "replay.json")
GROUP_FILES = ("summary.csv", "daily_equity.csv", "monthly_returns.csv", "completion.json")
AUDIT_FILE = "independent_audit.validation.json"


def load_transport(path):
    """Load exactly54ef without package lookup, .pyc writes or monkeypatches."""
    path = Path(path)
    if path.is_symlink() or hasattr(path, "is_junction") and path.is_junction():
        raise ValueError("Safe transport source cannot be a link")
    path = path.resolve()
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != TRANSPORT_SHA256:
        raise ValueError("Use the reviewed original safe-transport implementation")
    module = types.ModuleType("_td3_validation_frozen_transport")
    module.__file__ = str(path)
    exec(compile(content, str(path), "exec"), module.__dict__)
    return module


_transport_path = Path(sys.argv[sys.argv.index("--transport-source") + 1]) if "--transport-source" in sys.argv else ROOT / "scripts/collect_btceth_rl_results.py"
transport = load_transport(_transport_path)
sha, read_json, safe_target = transport.sha, transport.read_json, transport.safe_target


def utc(value):
    value = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if value.tzinfo is None:
        raise ValueError("Timezone-aware source evidence required")
    return value.astimezone(timezone.utc)


def hashes(root, values, label, *, required=()):
    if not isinstance(values, dict) or not values or not set(required).issubset(values):
        raise ValueError(label + " has incomplete immutable SHA evidence")
    for name, expected in values.items():
        if not re.fullmatch(r"[0-9a-f]{64}", str(expected)) or sha(safe_target(root, name)) != expected:
            raise ValueError(label + " changed: " + str(name))


def validation_files():
    return {"validation_completion.json", "validate_completion.json", "validate_process.json", "validate.log",
        *(f"validation/{name}" for name in GROUP_FILES[:3]),
        *(f"validation/{family}/{profile}/{name}" for family in FAMILIES for profile in PROFILES for name in GROUP_FILES),
        *(f"validation/{family}/{profile}/{cost}/{key}/{name}" for family in FAMILIES for profile in PROFILES for cost in ("base", "stress") for key in (*map(str, SEEDS), "ensemble") for name in CASE_FILES)}


def training_files(*, include_started=True):
    names = ("record.json", "model.zip", "training_timeout_transitions.jsonl", "started.json") if include_started else ("record.json", "model.zip", "training_timeout_transitions.jsonl")
    return {"training_completion.json", *(f"training/td3__{profile}__seed{seed}/{name}" for profile in PROFILES for seed in SEEDS for name in names)}


def no_annual(root):
    """Inspect names, never annual prices or rewards; fail even on a marker."""
    root = Path(root)
    if transport.is_link(root) or not root.is_dir():
        raise ValueError("Validation research root must be an ordinary directory")
    for parent, dirs, files in os.walk(root, followlinks=False):
        for name in (*dirs, *files):
            path = Path(parent) / name
            if transport.is_link(path):
                raise ValueError("Validation evidence cannot cross links or junctions")
            relative = path.relative_to(root).as_posix(); portable = PurePosixPath(relative)
            if ("test" in portable.parts or portable.name.startswith(("test_", "test.", "annual"))
                    or portable.name in ("independent_audit.json", "goal_candidate_evidence.json")):
                raise ValueError("Validation-only transport refuses every TD3 annual/test artifact: " + relative)


def identities(root, main, dataset, source_root, *, require_snapshot=False):
    if any(transport.is_link(path) for path in (root, main, dataset, source_root)):
        raise ValueError("Research, data and source roots cannot be links or junctions")
    root, main, dataset, source_root = map(lambda x: Path(x).resolve(), (root, main, dataset, source_root))
    if root == main or root.is_relative_to(main) or main.is_relative_to(root):
        raise ValueError("New TD3 study and original reference must remain separate")
    p, receipt, old = (read_json(safe_target(where, name)) for where, name in ((root, "protocol.json"), (root, "protocol_receipt.json"), (main, "frozen_spec.json")))
    psha = sha(safe_target(root, "protocol.json"))
    if (receipt.get("protocol_file") != "protocol.json" or receipt.get("protocol_sha256") != psha
            or utc(p["prepared_at"]) > utc(receipt["sealed_at"]) or p.get("algorithm") != "td3"
            or p.get("model_count") != 10 or p.get("timesteps_per_model") != 65536 or p.get("old_timesteps_per_model") != 32768
            or p.get("training_budget_multiple") != 2 or p.get("train_episode_days") != 32
            or p.get("validation_cases") != 48 or p.get("validation_repeats") != 48 or p.get("seeds") != list(SEEDS)
            or set(p.get("profiles", {})) != set(PROFILES) or p.get("budgets") != {"raw": 1., "cap075": .75}
            or p.get("capital") != 10000. or p.get("seed_selection") is not False or p.get("selection_uses_test") is not False
            or p.get("annual_requires_separate_authorization") is not True or p.get("recipe_is_pure_official_default") is not False
            or any(p.get(flag) is not True for flag in FLAGS)
            or p.get("original_frozen_spec_sha256") != sha(safe_target(main, "frozen_spec.json"))
            or any(p.get(key) != old.get(key) for key in ("dataset_sha256", "dataset_manifest_sha256"))
            or set(p.get("source_sha256", {})) != SOURCE_NAMES
            or any(p["source_sha256"][key] != old.get("code_sha256", {}).get(key) for key in FINANCIAL)
            or p["source_sha256"].get("scripts/btceth_rl_td3.py") != RUNNER_SHA256 or p["source_sha256"].get("scripts/audit_btceth_rl_td3.py") != AUDITOR_SHA256):
        raise ValueError("Frozen TD3 source, finite budget, protocol receipt or original identity changed")
    if transport.is_link(dataset) or sha(dataset) != p["dataset_sha256"] or sha(safe_target(dataset.parent, "manifest.json")) != p["dataset_manifest_sha256"]:
        raise ValueError("Original frozen dataset bytes changed")
    hashes(source_root, p["source_sha256"], "Actual reviewed source")
    hashes(main, p.get("original_input_sha256"), "Original reference SHA dependencies", required={"frozen_spec.json", "runtime.json", "independent_audit.json"})
    if require_snapshot:
        hashes(root / "sources", p["source_sha256"], "Archived original source")
        hashes(root / "upstream", p.get("upstream_evidence_sha256"), "Reviewed upstream/MIT evidence")
    return p, dict(mode="validation_only", protocol_sha256=psha, frozen_spec_sha256=p["original_frozen_spec_sha256"],
        dataset_sha256=p["dataset_sha256"], dataset_manifest_sha256=p["dataset_manifest_sha256"], source_sha256=p["source_sha256"])


def _phase(root, phase, p, *, process_root, process_main, process_dataset, process_source):
    process, complete = (read_json(safe_target(root, phase + suffix)) for suffix in ("_process.json", "_completion.json"))
    log = phase + ".log"; command = process.get("command", [])
    if (process.get("phase") != phase or process.get("status") != "completed" or type(process.get("exit_code")) is not int or process["exit_code"] != 0
            or process.get("source_sha256") != p["source_sha256"]["scripts/btceth_rl_td3.py"]
            or process.get("protocol_sha256") != sha(safe_target(root, "protocol.json")) or complete.get("protocol_sha256") != process["protocol_sha256"]
            or process.get("log_relative_path") != log or process.get("log_sha256") != sha(safe_target(root, log))
            or any(type(process.get(k)) is not int or process[k] <= 0 for k in ("supervisor_pid", "child_pid")) or process["supervisor_pid"] == process["child_pid"]
            or len(command) < 5 or str(command[2]) != str(PurePosixPath(str(process_source)) / "scripts/btceth_rl_td3.py")):
        raise ValueError("Actual matching completed0 TD3 process/log required: " + phase)
    options = {"--phase": "_" + phase, "--reference-main": str(process_main), "--dataset": str(process_dataset), "--output-dir": str(process_root)}
    for option, expected in options.items():
        if command.count(option) != 1 or command.index(option) + 1 >= len(command) or command[command.index(option) + 1] != expected:
            raise ValueError("TD3 terminal command belongs to another study: " + option)
    preflight = read_json(safe_target(root, "preflight/" + phase + ".json"))
    receipt = read_json(safe_target(root, "protocol_receipt.json"))
    if not (utc(receipt["sealed_at"]) <= utc(preflight["at"]) <= utc(process["started_at"]) <= utc(complete["at"]) <= utc(process["completed_at"])):
        raise ValueError("TD3 freeze, execution and terminal timestamps differ")
    count_ok = complete.get("model_count") == 10 if phase == "fit" else complete.get("case_count") == complete.get("repeat_count") == 48
    if not count_ok:
        raise ValueError("Complete10 fits and48 validation/repeats required")
    if sys.platform == "linux":
        for pid in (process["supervisor_pid"], process["child_pid"]):
            proc = Path("/proc") / str(pid)
            try:
                state = (proc / "stat").read_text().rsplit(")", 1)[1].split()[0]
                argv = (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
            except (FileNotFoundError, ProcessLookupError):
                continue
            if state != "Z" and "btceth_rl_td3.py" in argv:
                raise ValueError("TD3 process is still live")
    return process


def terminal_guard(root, *, main, dataset, source_root=ROOT, process_root=None, process_main=None, process_dataset=None, process_source=None):
    if transport.is_link(root) or transport.is_link(main):
        raise ValueError("Original research roots cannot be links or junctions")
    root, main = Path(root).resolve(), Path(main).resolve()
    no_annual(root)
    p, identity = identities(root, main, dataset, source_root, require_snapshot=True)
    process_args = dict(process_root=process_root or root, process_main=process_main or main, process_dataset=process_dataset or Path(dataset).resolve(), process_source=process_source or Path(source_root).resolve())
    fit = _phase(root, "fit", p, **process_args); validate = _phase(root, "validate", p, **process_args)
    audit = read_json(safe_target(root, AUDIT_FILE))
    fixed = dict(passed=True, status="partial_test_pending", phase="validation", test_complete=False, validation_complete=True,
        training_model_count=10, validation_run_count=48, run_count=48, test_run_count=0, baseline_run_count=0, repeat_count=48,
        failure_count=0, failures=[], research_targets=None, training_budget_multiple=2, fresh_holdout_success=False,
        **{flag: True for flag in FLAGS})
    if (any(audit.get(k) != v or isinstance(v, bool) and audit.get(k) is not v for k, v in fixed.items())
            or type(audit.get("checks_total")) is not int or audit["checks_total"] <= 0 or audit.get("dataset_sha256") != p["dataset_sha256"]
            or audit.get("source_sha256") != p["source_sha256"]):
        raise ValueError("Only complete independent710 partial_test_pending48 validation may be transported")
    required = validation_files() | training_files() | {"protocol.json", "protocol_receipt.json", "fit_process.json", "fit_completion.json", "fit.log", "preflight/fit.json", "preflight/validate.json"}
    hashes(root, audit.get("input_sha256"), "Independent48 audited study inputs", required=required)
    hashes(main, audit.get("main_input_sha256"), "Independent old reference SHA dependencies", required={"frozen_spec.json", "runtime.json"})
    hashes(source_root, audit.get("source_sha256"), "Independent audit source")
    expected_ids = {f"validation__{cost}__{family}__td3__{profile}__{key}" for family in FAMILIES for profile in PROFILES for cost in ("base", "stress") for key in (*map(str, SEEDS), "ensemble")}
    if set(audit.get("runs", {})) != expected_ids or any(v.get("reconstructed") is not True or v.get("hours") != 7080 or v.get("repeat_exact") is not True for v in audit["runs"].values()):
        raise ValueError("Exactly48 distinct independently rebuilt funded validation accounts required")
    complete = read_json(safe_target(root, "validation_completion.json"))
    if complete.get("protocol_sha256") != identity["protocol_sha256"] or complete.get("case_count") != 48 or complete.get("repeat_count") != 48:
        raise ValueError("Complete48 validation parent receipt required")
    training = read_json(safe_target(root, "training_completion.json")); records = training.get("records", [])
    if (training.get("model_count") != 10 or training.get("protocol_sha256") != identity["protocol_sha256"] or len(records) != 10
            or {(r.get("profile"), r.get("seed")) for r in records} != {(profile, seed) for profile in PROFILES for seed in SEEDS}):
        raise ValueError("Exactly ten original complete training records required")
    for record in records:
        folder = f"training/td3__{record['profile']}__seed{record['seed']}"
        saved = read_json(safe_target(root, folder + "/record.json"))
        if (saved != record or saved.get("algorithm") != "td3" or saved.get("actual_timesteps") != 65536
                or saved.get("actual_n_updates") != 65024 or saved.get("actor_update_count") != 32512
                or saved.get("protocol_sha256") != identity["protocol_sha256"]):
            raise ValueError("Frozen training budget or record identity changed")
        for file, key, relative_key in (("model.zip", "model_sha256", "model_relative_path"), ("training_timeout_transitions.jsonl", "timeout_trace_sha256", "timeout_trace_relative_path")):
            if saved.get(relative_key) != folder + "/" + file or saved.get(key) != sha(safe_target(root, folder + "/" + file)):
                raise ValueError("Training model/timeout byte identity changed")
    selection, seal = (read_json(safe_target(root, name)) for name in ("selection.json", "selection_receipt.json"))
    if (seal.get("selection_file") != "selection.json" or seal.get("selection_sha256") != sha(safe_target(root, "selection.json"))
            or selection.get("protocol_sha256") != seal.get("protocol_sha256") or selection.get("protocol_sha256") != identity["protocol_sha256"]
            or selection.get("audit_sha256") != sha(safe_target(root, AUDIT_FILE)) or selection.get("seed_selection") is not False
            or selection.get("selection_uses_test") is not False or selection.get("test_results_available_to_selector") is not False
            or any(selection.get(flag) is not True for flag in FLAGS)):
        raise ValueError("Whole-file four-candidate validation choice must already be sealed")
    pool = selection.get("candidates", [])
    if (len(pool) != 4 or {(r.get("family"), r.get("profile")) for r in pool} != {(family, profile) for family in FAMILIES for profile in PROFILES}
            or any(r.get("algorithm") != "td3" or isinstance(r.get("score"), bool) or not isinstance(r.get("score"), (int, float)) or not math.isfinite(r["score"]) for r in pool)
            or pool != sorted(pool, key=lambda r: (-r["score"], 0 if r["family"] == "raw" else 1, r["profile"])) or selection.get("candidate") != pool[0]):
        raise ValueError("Only four fixed candidates, sealed raw-first ties and no seed selection allowed")
    hashes(root, selection.get("training_source_sha256"), "Sealed ten trained models", required=training_files(include_started=False))
    hashes(root, selection.get("validation_source_sha256"), "Sealed48 validation accounts", required=validation_files())
    if not (utc(fit["completed_at"]) <= utc(validate["started_at"]) <= utc(validate["completed_at"]) <= utc(audit["audited_at"]) <= utc(selection["selected_at"]) <= utc(seal["sealed_at"])):
        raise ValueError("Complete fit, validation audit and selection seal must occur in order")
    protected_names = {"protocol.json", "protocol_receipt.json", "fit_process.json", "fit.log", "validate_process.json", "validate.log", "validation_completion.json", AUDIT_FILE, "selection.json", "selection_receipt.json"}
    identity.update(audit_file=AUDIT_FILE, audit_sha256=sha(safe_target(root, AUDIT_FILE)), terminal_proof_sha256={n: sha(safe_target(root, n)) for n in sorted(protected_names)},
        audit_input_sha256=audit["input_sha256"], main_input_sha256=audit["main_input_sha256"], audit_source_sha256=audit["source_sha256"],
        selection_sha256=sha(safe_target(root, "selection.json")), training_models=10, validation_cases=48, validation_repeats=48,
        financial_recomputation=False, annual_reads_enabled=False)
    return identity


def export_snapshot(root, *, main, dataset, source_root=ROOT, export_parent=None):
    """Explicit archive orchestration; reuse54ef safety functions unchanged."""
    if transport.is_link(root):
        raise ValueError("Research root cannot be a link or junction")
    root = Path(root).resolve()
    first = terminal_guard(root, main=main, dataset=dataset, source_root=source_root)
    before = transport.inventory(root)
    parent = Path(export_parent or root.parent / "btceth_rl_td3_validation_exports")
    if transport.is_link(parent) or parent.resolve() == root or parent.resolve().is_relative_to(root) or parent.resolve().is_relative_to(Path(main).resolve()):
        raise ValueError("Validation exports must be outside both studies")
    parent = parent.resolve(); parent.mkdir(parents=True, exist_ok=True)
    token = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex
    destination = safe_target(parent, "validation-/" + token); destination.parent.mkdir(exist_ok=True); destination.mkdir(exist_ok=False)
    archive = destination / "experiment_snapshot.tar.gz"
    try:
        with archive.open("xb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with tarfile.open(mode="w", fileobj=compressed, format=tarfile.PAX_FORMAT) as tar:
                for item in before:
                    path = safe_target(root, item["path"])
                    with os.fdopen(os.open(path, os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)), "rb") as stream:
                        details = os.fstat(stream.fileno())
                        if not stat.S_ISREG(details.st_mode) or details.st_size != item["size"]:
                            raise RuntimeError("Source changed before archive snapshot")
                        info = tarfile.TarInfo(item["path"]); info.size, info.mode, info.mtime = item["size"], 0o600, 0
                        info.uid = info.gid = 0; info.uname = info.gname = ""; tar.addfile(info, stream)
        if (transport.inventory(root) != before or terminal_guard(root, main=main, dataset=dataset, source_root=source_root) != first
                or sha(Path(transport.__file__)) != TRANSPORT_SHA256):
            raise RuntimeError("Study or source changed during validation snapshot")
        manifest = dict(schema_version=1, created_at=transport.stamp(), mode="validation_only", remote_root=str(root),
            file_count=len(before), total_bytes=sum(i["size"] for i in before), files=before, terminal_guard=first,
            archive_name=archive.name, archive_sha256=sha(archive), archive_size=archive.stat().st_size)
        manifest_path = destination / "snapshot_manifest.json"; transport.write_new_json(manifest_path, manifest)
        return dict(status="complete", mode="validation_only", remote_root=str(root), export_directory=str(destination), archive_path=str(archive), manifest_path=str(manifest_path),
            archive_sha256=manifest["archive_sha256"], archive_size=manifest["archive_size"], manifest_sha256=sha(manifest_path),
            protocol_sha256=first["protocol_sha256"], frozen_spec_sha256=first["frozen_spec_sha256"], audit_file=AUDIT_FILE, audit_sha256=first["audit_sha256"], reused_transport_sha256=TRANSPORT_SHA256)
    except BaseException as error:
        transport.write_new_json(destination / "export_failure.json", dict(at=transport.stamp(), error_type=type(error).__name__, error=str(error)))
        raise


def collect(*, output, reference_output, main_output, dataset_path, source_root=ROOT, alias=transport.DEFAULT_ALIAS,
            remote_python=transport.DEFAULT_REMOTE_PYTHON, remote_root=REMOTE_ROOT, remote_main=REMOTE_MAIN,
            remote_dataset=REMOTE_DATASET, remote_source=REMOTE_SOURCE, remote_transport=REMOTE_TRANSPORT, export_parent=None, staging=None):
    if any(transport.is_link(path) for path in (output, reference_output, main_output)):
        raise ValueError("Collection roots cannot be links or junctions")
    output, reference_output, main_output = map(lambda p: Path(p).resolve(), (output, reference_output, main_output))
    if (output == main_output or output.is_relative_to(main_output) or main_output.is_relative_to(output)
            or output != reference_output and (output.is_relative_to(reference_output) or reference_output.is_relative_to(output))):
        raise ValueError("Collected TD3 output must be separate from original studies and un-nested")
    no_annual(reference_output)
    _, local = identities(reference_output, main_output, dataset_path, source_root)
    token = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex
    staging = Path(staging or ROOT / "results/backtests/btceth_rl_td3_validation_transfers" / token)
    if transport.is_link(staging) or staging.exists() or any(staging.resolve() == bound or staging.resolve().is_relative_to(bound) or bound.is_relative_to(staging.resolve()) for bound in (output, reference_output, main_output)):
        raise ValueError("A new independent staging tree is required")
    staging.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).read_bytes(); parent = export_parent or str(PurePosixPath(remote_root).parent / "btceth_rl_td3_validation_exports")
    common = ["--remote-root", str(remote_root), "--remote-export-parent", str(parent), "--main-output", str(remote_main), "--dataset", str(remote_dataset), "--source-root", str(remote_source), "--transport-source", str(remote_transport)]
    response = transport._remote_call(alias, remote_python, [*common, "--remote-export-worker"], source)
    transport._validate_remote_export(response, remote_root, parent, True)
    if (response.get("protocol_sha256") != local["protocol_sha256"] or response.get("frozen_spec_sha256") != local["frozen_spec_sha256"]
            or response.get("reused_transport_sha256") != TRANSPORT_SHA256 or response.get("audit_file") != AUDIT_FILE):
        raise ValueError("Remote validation identity differs from the prepared local protocol")
    transport.write_new_json(staging / "remote_export_receipt.json", response)
    manifest_path, archive = staging / "snapshot_manifest.json", staging / "experiment_snapshot.tar.gz"
    transport._remote_call(alias, remote_python, [*common, "--remote-read-file", response["manifest_path"]], source, destination=manifest_path)
    if sha(manifest_path) != response["manifest_sha256"]:
        raise ValueError("Downloaded validation manifest SHA changed")
    manifest = read_json(manifest_path)
    if any(manifest.get(k) != response[k] for k in ("mode", "remote_root", "archive_sha256", "archive_size")) or manifest.get("terminal_guard", {}).get("audit_sha256") != response.get("audit_sha256"):
        raise ValueError("Manifest differs from its unique remote validation receipt")
    transport.manifest_entries(manifest)
    transport._remote_call(alias, remote_python, [*common, "--remote-read-file", response["archive_path"]], source, destination=archive)
    extracted = staging / "verified_files"; entries = transport.verify_extract(archive, manifest, extracted)
    proof = terminal_guard(extracted, main=main_output, dataset=dataset_path, source_root=source_root,
        process_root=remote_root, process_main=remote_main, process_dataset=remote_dataset, process_source=remote_source)
    if proof != manifest["terminal_guard"] or proof["protocol_sha256"] != local["protocol_sha256"] or "transfer_receipt.json" in entries:
        raise ValueError("Extracted validation evidence differs or nests an existing transport receipt")
    no_annual(output) if output.exists() else None
    if identities(reference_output, main_output, dataset_path, source_root)[1] != local or sha(Path(transport.__file__)) != TRANSPORT_SHA256:
        raise RuntimeError("Local inputs or reviewed transport changed before merge")
    merged = transport.merge_verified_files(extracted, entries, output)
    receipt = dict(schema_version=1, collected_at=transport.stamp(), mode="validation_only", remote_alias=alias, remote_root=str(remote_root),
        remote_export_directory=response["export_directory"], local_staging_directory=str(staging), archive_sha256=manifest["archive_sha256"], archive_size=manifest["archive_size"],
        manifest_sha256=response["manifest_sha256"], collector_source_sha256=hashlib.sha256(source).hexdigest(), reused_transport_sha256=TRANSPORT_SHA256,
        protocol_sha256=local["protocol_sha256"], audit_file=AUDIT_FILE, audit_sha256=proof["audit_sha256"], terminal_guard=proof,
        exactfiles=[dict(path=name, sha256=item["sha256"]) for name, item in sorted(entries.items())], file_count=len(entries), total_bytes=manifest["total_bytes"],
        no_annual_capability=True, no_financial_recomputation=True, preserve_remote_audit_use_separate_local_audit=True, **merged)
    transport.write_new_json(staging / "collection_receipt.json", receipt)
    if not merged["receipt_reused"]:
        transport.write_new_json(output / "transfer_receipt.json", receipt)
    return dict(status="collected", mode="validation_only", output_directory=str(output), staging_directory=str(staging), file_count=len(entries),
        copied_files=len(merged["copied_files"]), identical_files_preserved=len(merged["identical_files_preserved"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/backtests/btceth_rl_td3_20261030")
    parser.add_argument("--reference-output-dir", type=Path, default=ROOT / "results/backtests/btceth_rl_td3_20261030")
    parser.add_argument("--main-output", type=Path, default=ROOT / "results/backtests/btceth_rl_20261029")
    parser.add_argument("--dataset", type=Path, default=transport.DEFAULT_DATASET)
    parser.add_argument("--source-root", type=Path, default=ROOT)
    parser.add_argument("--transport-source", type=Path, default=_transport_path)
    parser.add_argument("--ssh-alias", default=transport.DEFAULT_ALIAS)
    parser.add_argument("--remote-python", default=transport.DEFAULT_REMOTE_PYTHON)
    parser.add_argument("--remote-root", default=REMOTE_ROOT)
    parser.add_argument("--remote-main", default=REMOTE_MAIN)
    parser.add_argument("--remote-dataset", default=REMOTE_DATASET)
    parser.add_argument("--remote-source", default=REMOTE_SOURCE)
    parser.add_argument("--remote-transport", default=REMOTE_TRANSPORT)
    parser.add_argument("--remote-export-parent")
    parser.add_argument("--staging-dir", type=Path)
    parser.add_argument("--remote-export-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--remote-read-file", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.remote_read_file:
        path, parent = Path(args.remote_read_file), Path(args.remote_export_parent)
        if (path.name not in ("snapshot_manifest.json", "experiment_snapshot.tar.gz") or not re.fullmatch(r"\d{8}T\d{6}Z-[0-9a-f]{32}", path.parent.name)
                or path.parent.parent.name != "validation-" or path.parent.parent.parent.resolve() != parent.resolve()):
            raise ValueError("Only generated validation snapshot artifacts may be streamed")
        path = safe_target(parent, path.relative_to(parent).as_posix())
        with path.open("rb") as stream:
            shutil.copyfileobj(stream, sys.stdout.buffer, length=1024 * 1024)
        return
    if args.remote_export_worker:
        result = export_snapshot(args.remote_root, main=args.main_output, dataset=args.dataset, source_root=args.source_root, export_parent=args.remote_export_parent)
    else:
        result = collect(output=args.output_dir, reference_output=args.reference_output_dir, main_output=args.main_output, dataset_path=args.dataset, source_root=args.source_root,
            alias=args.ssh_alias, remote_python=args.remote_python, remote_root=args.remote_root, remote_main=args.remote_main, remote_dataset=args.remote_dataset,
            remote_source=args.remote_source, remote_transport=args.remote_transport, export_parent=args.remote_export_parent, staging=args.staging_dir)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()

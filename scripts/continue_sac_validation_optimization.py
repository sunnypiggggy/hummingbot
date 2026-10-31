"""Versioned orchestration handoff; never restart the inherited tuning fit.

The original fourteen sources, plan, protocol, launch and logs remain immutable.
Only the separately sealed v2 metadata auditor is selected for later audits.
No signal, process-group action, market rule or learner mutation occurs here.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
import time

DRIVER = "extension_source/scripts/backtest_sac_validation_optimization.py"
OLD_AUDITOR = "scripts/audit_sac_validation_optimization.py"
NEW_AUDITOR = "audit_revision_source/scripts/audit_sac_validation_optimization_v2.py"
CONTROLLER = "audit_revision_source/scripts/continue_sac_validation_optimization.py"
REVISION_ID = "native_sac_policy_kwargs_default_false_v2"
REASON = "sb3_sac_saved_policy_kwargs_use_sde_false"
NORMALIZATION = "net_arch unchanged; use_sde absent or boolean False only; reject unknown policy kwargs"
CANDIDATES = ("c00_baseline", "c01_lr1e4", "c02_gamma099", "c03_lr1e4_gamma099")
SYMBOLS = ("BTCUSDT", "ETHUSDT")
SEEDS = tuple(range(100, 105))
_STATUS_OWNED = False


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def safe(top, name):
    top, name = Path(top).absolute(), str(name)
    part = PurePosixPath(name)
    if not name or part.as_posix() != name or part.is_absolute() or ".." in part.parts or ":" in name or "\\" in name:
        raise ValueError("Only canonical research-relative paths are permitted")
    path = top.joinpath(*part.parts)
    if any(p.is_symlink() or hasattr(p, "is_junction") and p.is_junction() for p in (path, *path.parents)):
        raise ValueError("Research evidence cannot cross a link")
    path.resolve().relative_to(top.resolve())
    return path


def write_status(top, **fields):
    target = safe(top, "pipeline_status.json")
    temporary = target.with_suffix(".revision-pending")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump({"at": now(), **fields}, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    os.replace(temporary, target)


def process_identity(pid):
    path = Path("/proc") / str(pid)
    try:
        fields = (path / "stat").read_text().rsplit(")", 1)[1].split()
        command = [p.decode() for p in (path / "cmdline").read_bytes().split(b"\0") if p]
        return {"pid": int(pid), "state": fields[0], "start_ticks": str(fields[19]), "command": command}
    except (FileNotFoundError, ProcessLookupError):
        return None


def normalized(command):
    return [str(p) for p in command if str(p)]


def verify_revision(top, dataset):
    top, dataset = Path(top), Path(dataset)
    revision = read(safe(top, "audit_revision.json"))
    receipt = read(safe(top, "audit_revision_receipt.json"))
    original = read(safe(top, "pipeline_launch.json"))
    extension, protocol = read(safe(top, "extension_manifest.json")), read(safe(top, "protocol.json"))
    expected = {"schema_version": 1, "revision_id": REVISION_ID, "reason_code": REASON,
        "revised_audit_source_path": NEW_AUDITOR, "controller_source_path": CONTROLLER,
        "permitted_metadata_normalization": NORMALIZATION,
        "financial_math_changed": False, "model_recipe_changed": False, "training_restarted": False}
    for key, value in expected.items():
        if revision.get(key) != value or isinstance(value, bool) and type(revision.get(key)) is not bool:
            raise ValueError("Only the explicitly declared metadata compatibility revision is allowed: " + key)
    bindings = {"original_protocol_sha256": "protocol.json", "search_plan_sha256": "search_plan.json",
        "original_extension_manifest_sha256": "extension_manifest.json", "original_pipeline_launch_sha256": "pipeline_launch.json",
        "revised_audit_source_sha256": NEW_AUDITOR, "controller_source_sha256": CONTROLLER}
    for field, name in bindings.items():
        if revision.get(field) != sha(safe(top, name)):
            raise ValueError("Immutable original/revised source binding changed: " + name)
    if (receipt.get("sha256") != sha(safe(top, "audit_revision.json"))
            or datetime.fromisoformat(receipt["at_utc"]) < datetime.fromisoformat(revision["created_at_utc"])
            or sha(__file__) != revision["controller_source_sha256"]
            or extension["extension_sources"].get(OLD_AUDITOR) != revision.get("original_audit_source_sha256")
            or protocol["sources"].get("scripts/audit_single_pair_rl.py") != revision.get("financial_oracle_source_sha256")
            or read(safe(top, "protocol_receipt.json"))["protocol_sha256"] != revision["original_protocol_sha256"]
            or read(safe(top, "search_plan_receipt.json"))["sha256"] != revision["search_plan_sha256"]
            or sha(dataset) != protocol["dataset_sha256"]
            or sha(dataset.parent / "manifest.json") != protocol["dataset_manifest_sha256"]):
        raise ValueError("Original protocol, metadata correction receipt or exact data changed")
    for name, expected_sha in protocol["sources"].items():
        if sha(safe(top / "frozen_source", name)) != expected_sha:
            raise ValueError("The original financial/model source changed")
    for name, expected_sha in extension["extension_sources"].items():
        if sha(safe(top / "extension_source", name)) != expected_sha:
            raise ValueError("The original optimization source changed")
    requested = read(safe(top, "controller_handoff_requested.json"))
    confirmed = read(safe(top, "controller_handoff_confirmed.json"))
    keys = ("requested_at_utc", "reason_code", "old_controller_pid", "old_controller_start_ticks", "active_tuning_fit_pid",
            "active_tuning_fit_start_ticks", "original_pipeline_launch_sha256", "protocol_sha256", "original_auditor_sha256",
            "training_restarted", "all_frozen_inputs_preserved")
    if (any(requested.get(k) != confirmed.get(k) for k in keys)
            or confirmed.get("training_restarted") is not False or confirmed.get("all_frozen_inputs_preserved") is not True
            or confirmed.get("active_fit_still_alive") is not True or confirmed.get("reason_code") != REASON
            or confirmed.get("old_controller_pid") != original["pid"]
            or confirmed.get("original_pipeline_launch_sha256") != revision["original_pipeline_launch_sha256"]
            or confirmed.get("protocol_sha256") != revision["original_protocol_sha256"]
            or confirmed.get("original_auditor_sha256") != revision["original_audit_source_sha256"]):
        raise ValueError("The precise non-destructive orchestration handoff is required")
    old = process_identity(original["pid"])
    if old is not None and old["state"] != "Z" and old["start_ticks"] == str(confirmed["old_controller_start_ticks"]):
        raise ValueError("The original controller is still active; do not duplicate orchestration")
    return revision, confirmed


def wait_for_launch(top, revision, handoff, *, timeout=5.):
    target, deadline = safe(top, "pipeline_launch_revision.json"), time.monotonic() + timeout
    while True:
        try:
            launch = read(target)
            break
        except (FileNotFoundError, json.JSONDecodeError):
            if time.monotonic() >= deadline:
                raise ValueError("The parent must seal the new PID/command launch receipt")
            time.sleep(.05)
    required = {"pid": os.getpid(), "original_pipeline_launch_sha256": revision["original_pipeline_launch_sha256"],
        "inherited_tuning_fit_pid": handoff["active_tuning_fit_pid"],
        "audit_revision_sha256": sha(safe(top, "audit_revision.json")), "controller_source_sha256": revision["controller_source_sha256"],
        "revised_audit_source_sha256": revision["revised_audit_source_sha256"], "training_restarted": False}
    if (any(launch.get(k) != v for k, v in required.items())
            or str(launch.get("inherited_tuning_fit_start_ticks")) != str(handoff["active_tuning_fit_start_ticks"])):
        raise ValueError("New controller launch does not belong to the exact inherited fit/revision")
    command = normalized(launch.get("command", []))
    if str(Path(__file__).resolve()) not in command or "tuning-fit" in command or "pipeline" in command:
        raise ValueError("The new launch must invoke this continuation controller, never another fitting pipeline")
    return launch


def validate_finished_tuning(top):
    top = Path(top)
    completed = read(safe(top, "tuning_fit_complete.json"))
    status = read(safe(top, "tuning_fit_status.json"))
    expected_children = {"tuning/" + c for c in CANDIDATES}
    if (completed.get("fresh_tasks") != 40 or set(completed.get("child_counter_audit_sha256", {})) != expected_children
            or status.get("completed") != 40 or status.get("total") != 40 or status.get("failed") != []):
        raise ValueError("The original forty-model fit did not finish successfully; do not restart it")
    expected_ids = {(symbol, seed) for symbol in SYMBOLS for seed in SEEDS}
    parent_sha = sha(safe(top, "protocol.json"))
    extension = read(safe(top, "extension_manifest.json"))
    for candidate in CANDIDATES:
        prefix = "tuning/" + candidate + "/"
        counter = read(safe(top, prefix + "training_counter_audit.json"))
        fit = read(safe(top, prefix + "fit_complete.json"))
        if (completed["child_counter_audit_sha256"]["tuning/" + candidate] != sha(safe(top, prefix + "training_counter_audit.json"))
                or counter.get("passed") is not True or counter.get("failure_count") != 0 or counter.get("failures") != []
                or counter.get("models_checked") != 10 or counter.get("new_models") != 10 or counter.get("retained_models") != 0
                or counter.get("core_protocol_sha256") != sha(safe(top, prefix + "protocol.json"))
                or counter.get("parent_protocol_sha256") != parent_sha
                or counter.get("extension_source_sha256") != extension["extension_sources"]["scripts/backtest_sac_validation_optimization.py"]
                or fit.get("model_count") != 10 or len(fit.get("records", [])) != 10):
            raise ValueError("Each inherited candidate must have its exact ten-model native counter audit")
        rows = {(r["symbol"], r["seed"]): r for r in counter["models"]}
        if len(counter["models"]) != 10 or set(rows) != expected_ids:
            raise ValueError("Inherited tuning model population changed")
        actual_records = []
        for symbol, seed in sorted(expected_ids):
            name = prefix + f"training/{symbol}/sac/seed{seed}/record.json"
            record = read(safe(top, name))
            actual_records.append(record)
            row, c = rows[(symbol, seed)], record["training"]["counters"]
            if (row.get("record_sha256") != sha(safe(top, name)) or row.get("counters") != c
                    or row.get("training_origin") != "fresh" or row.get("candidate_id") != candidate
                    or record.get("symbol") != symbol or record.get("algorithm") != "sac" or record.get("seed") != seed
                    or record.get("stage") != "tuning" or record.get("candidate_id") != candidate
                    or record["training"]["actual_steps"] != 32768
                    or c.get("environment_steps") != 32768 or c.get("gradient_update_rounds") != 32256
                    or c.get("optimizer_steps") != {"actor": 32256, "critic": 32256, "entropy": 32256}
                    or c.get("replay_add_calls") != 32768 or c.get("replay_transitions") != 32768):
                raise ValueError("Inherited native work/identity no longer matches its terminal evidence")
            directory = name.rsplit("/", 1)[0]
            for filename, expected_sha in record["files"].items():
                if sha(safe(top, directory + "/" + filename)) != expected_sha:
                    raise ValueError("Inherited trained checkpoint changed")
        if sorted(fit["records"], key=lambda r: (r["symbol"], r["seed"])) != sorted(actual_records, key=lambda r: (r["symbol"], r["seed"])):
            raise ValueError("Inherited fit receipt no longer contains all exact trained records")
    return completed


def wait_for_existing_fit(top, dataset, handoff, *, interval=30):
    while True:
        verify_revision(top, dataset)
        actual = process_identity(handoff["active_tuning_fit_pid"])
        if actual is None or actual["state"] == "Z":
            return validate_finished_tuning(top)
        if (actual["start_ticks"] != str(handoff["active_tuning_fit_start_ticks"])
                or actual["command"] != normalized(handoff["active_tuning_fit_command"])):
            raise ValueError("Inherited fit PID was reused or changed identity; never touch another process")
        write_status(top, phase="inherited-tuning-fit", status="running", pid=os.getpid(),
                     inherited_tuning_fit_pid=actual["pid"], training_restarted=False)
        time.sleep(interval)


def verify_audit(top, value):
    if value.get("passed") is not True or value.get("failure_count") != 0 or not value.get("inputs_sha256"):
        raise ValueError("An exact successful v2 audit is required")
    revision = read(safe(top, "audit_revision.json"))
    if value.get("audit_source_sha256") != revision["revised_audit_source_sha256"]:
        raise ValueError("The continuation must only consume the declared v2 auditor")
    for name, expected in value["inputs_sha256"].items():
        if sha(safe(top, name)) != expected:
            raise ValueError("Audited continuation input changed")


def continue_pipeline(top, dataset, *, interval=30, max_workers=None):
    global _STATUS_OWNED
    top = Path(top).resolve()
    revision, handoff = verify_revision(top, dataset)
    wait_for_launch(top, revision, handoff)
    _STATUS_OWNED = True
    wait_for_existing_fit(top, dataset, handoff, interval=interval)
    runner, audit = safe(top, DRIVER), safe(top, NEW_AUDITOR)
    phases = [("tuning-validation", "tuning_validation_complete.json")]
    phases += [("audit-tuning:" + c, "tuning/" + c + "/validation_optimization_audit.json") for c in CANDIDATES]
    phases += [("select", "selection.json"), ("audit-selection", "selection_audit.json"),
        ("final-prepare", "final_prepared.json"), ("final-fit", "final_fit_complete.json"),
        ("final-validation", "final_validation_complete.json"), ("audit-final-validation", "final/validation_optimization_audit.json"),
        ("final-test", "final_test_complete.json"), ("audit-final-test", "final/test_optimization_audit.json"),
        ("audit-completion", "completion_contract_audit.json")]
    logs = safe(top, "audit_revision_logs")
    logs.mkdir(exist_ok=True)
    for phase, marker_name in phases:
        verify_revision(top, dataset)
        marker = safe(top, marker_name)
        if marker.exists():
            if phase.startswith("audit-"):
                verify_audit(top, read(marker))
            continue
        if phase.startswith("audit-"):
            mode = "tuning-validation" if phase.startswith("audit-tuning:") else phase.removeprefix("audit-")
            command = [sys.executable, "-B", str(audit), "--study-root", str(top), "--dataset", str(dataset), "--phase", mode]
            if phase.startswith("audit-tuning:"):
                command += ["--candidate", phase.split(":", 1)[1]]
        else:
            command = [sys.executable, "-B", str(runner), phase, "--output", str(top), "--dataset", str(dataset)]
            if max_workers:
                command += ["--max-workers", str(max_workers)]
        if "tuning-fit" in command or "pipeline" in command:
            raise ValueError("The inherited tuning work can never be restarted by continuation")
        write_status(top, phase=phase, status="running", pid=os.getpid(), training_restarted=False,
                     audit_revision_sha256=sha(safe(top, "audit_revision.json")))
        log = logs / (phase.replace(":", "_") + ".log")
        with log.open("x", encoding="utf-8") as stream:
            result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
        if result.returncode or not marker.exists():
            write_status(top, phase=phase, status="failed", pid=os.getpid(), exit_code=result.returncode,
                         training_restarted=False, all_evidence_preserved=True)
            raise RuntimeError("Continuation phase failed; preserve work and logs: " + phase)
        if phase.startswith("audit-"):
            verify_audit(top, read(marker))
    verify_revision(top, dataset)
    write_status(top, status="completed", no_production_change=True, training_restarted=False,
                 audit_revision_sha256=sha(safe(top, "audit_revision.json")))
    return {"completed": True, "inherited_tuning_fit_pid": handoff["active_tuning_fit_pid"], "training_restarted": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", "--study-root", dest="output", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--poll-interval", type=int, default=30)
    parser.add_argument("--max-workers", type=int)
    args = parser.parse_args()
    if not 1 <= args.poll_interval <= 60:
        parser.error("Use a bounded1..60second wait interval")
    try:
        result = continue_pipeline(args.output, args.dataset, interval=args.poll_interval, max_workers=args.max_workers)
    except BaseException as exc:
        if _STATUS_OWNED:
            write_status(args.output, phase="continuation", status="failed", pid=os.getpid(), error_type=type(exc).__name__,
                         error=str(exc), training_restarted=False, all_evidence_preserved=True)
        raise
    print(json.dumps(result))


if __name__ == "__main__":
    main()

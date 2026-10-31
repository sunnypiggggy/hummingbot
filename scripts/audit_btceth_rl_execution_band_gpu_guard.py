"""Independent resource-adapter evidence audit; frozen financial audit unchanged.

The only admissible GPU preflight exception is the declared existing ptyxis
process instance. This source never stops a process, runs a policy, or changes
the original protocol. The original failed preflight remains evidence.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import audit_btceth_rl_execution_band as core
from scripts.audit_btceth_rl import Checks
from scripts.audit_btceth_rl_exposure_cap import read_json, safe_file, sha, utc

EXEMPT_PID = 715236
EXACT_QUERY = ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"]
SNAPSHOT_QUERY = ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
                  "--format=csv,noheader"]
GPU_STATUS_QUERY = ["nvidia-smi", "--query-gpu=index,uuid,name,memory.used,utilization.gpu",
                    "--format=csv,noheader"]
SOURCE_FILES = ("scripts/btceth_rl_execution_band.py", "scripts/audit_btceth_rl_execution_band.py",
    "scripts/btceth_rl_execution_band_gpu_guard.py", "scripts/audit_btceth_rl_execution_band_gpu_guard.py")
INSTANCE_FIELDS = ("pid", "uid", "username", "exe", "starttime_ticks", "boot_id")
FAILURE_FILES = ("runner_process.json", "validation.log", "preflight/launch.log")


def validate_instance(instance, *, expected=None):
    if not isinstance(instance, dict) or not set(INSTANCE_FIELDS).issubset(instance):
        raise ValueError("A complete non-reusable process-instance identity is required")
    if (instance["pid"] != EXEMPT_PID or instance["username"] != "sunny"
            or not isinstance(instance["uid"], int) or instance["uid"] <= 0
            or Path(instance["exe"]).name != "ptyxis" or not str(instance["exe"]).startswith("/")
            or not str(instance["starttime_ticks"]).isdigit() or int(instance["starttime_ticks"]) <= 0
            or not isinstance(instance["boot_id"], str) or len(instance["boot_id"]) != 36):
        raise ValueError("Only the authorized existing sunny ptyxis instance can be exempt")
    if expected is not None and any(instance[name] != expected[name] for name in INSTANCE_FIELDS):
        raise ValueError("PID reuse, changed executable, UID, boot or start time cannot be exempt")
    return {name: instance[name] for name in INSTANCE_FIELDS}


def filtered_query(stdout, allowed_rows):
    """Delete exact frozen row identities and preserve every other byte/row."""
    if not isinstance(stdout, str):
        raise ValueError("Record the original text query response")
    allowed = set()
    for row in allowed_rows:
        if (set(row) != {"gpu_uuid", "pid"} or row["pid"] != EXEMPT_PID
                or not isinstance(row["gpu_uuid"], str) or not row["gpu_uuid"].startswith("GPU-")):
            raise ValueError("Only exact original-GPU/existing-ptyxis rows may be removed")
        key = (row["gpu_uuid"], row["pid"])
        if key in allowed:
            raise ValueError("Duplicate allowed row identities are ambiguous")
        allowed.add(key)
    if not allowed:
        raise ValueError("Freeze the actual preflight exception row identities")
    kept, dropped = [], []
    for line in stdout.splitlines(keepends=True):
        parts = line.strip().split(",")
        try:
            identity = (parts[0].strip(), int(parts[1].strip())) if len(parts) == 2 else None
        except ValueError:
            identity = None
        if identity in allowed:
            if identity in dropped:
                raise ValueError("Duplicated observed process rows cannot be silently suppressed")
            dropped.append(identity)
        else:
            kept.append(line)
    return "".join(kept), [{"gpu_uuid": uuid, "pid": pid} for uuid, pid in dropped]


def verify_query_event(event, instance, allowed_rows, checks, label):
    checks.check(event.get("argv") == EXACT_QUERY and event.get("returncode") == 0,
                 label + ": only the frozen exact successful resource query is adapted")
    try:
        validate_instance(event.get("instance_observed"), expected=instance)
        expected, dropped = filtered_query(event["stdout_original"], allowed_rows)
        checks.check(event.get("stdout_filtered") == expected,
                     label + ": every non-exempt PID and byte remains untouched")
        if "removed_rows" in event:
            checks.check(event["removed_rows"] == dropped, label + ": exact removed row identities")
    except (ValueError, TypeError, KeyError) as exc:
        checks.check(False, label + ": " + str(exc))


def verify_snapshot(snapshot, instance, allowed_rows, gpus, checks, label):
    try:
        validate_instance(snapshot["instance"], expected=instance)
        checks.check(snapshot.get("argv") == SNAPSHOT_QUERY and snapshot.get("gpu_status_argv") == GPU_STATUS_QUERY,
                     label + ": original unfiltered resource and GPU status query identities")
        observed, allowed = set(), {(row["gpu_uuid"], row["pid"]) for row in allowed_rows}
        for line in snapshot["stdout"].splitlines():
            if not line.strip():
                continue
            fields = [part.strip() for part in line.split(",")]
            if len(fields) != 4:
                raise ValueError("Complete real four-column GPU process records required")
            identity = (fields[0], int(fields[1]))
            memory = float(fields[3].removesuffix("MiB").strip())
            if identity in observed or identity not in allowed or Path(fields[2]).name != "ptyxis" or not 0 <= memory < float("inf"):
                raise ValueError("Only the exact declared existing terminal instance may be present")
            observed.add(identity)
        checks.check(observed == allowed, label + ": exact original UI instance rows; no other compute PID")
        rows = {}
        for line in snapshot["gpu_status_stdout"].splitlines():
            if not line.strip():
                continue
            fields = [part.strip() for part in line.split(",")]
            if len(fields) != 5 or fields[1] in rows or not fields[0].isdigit():
                raise ValueError("Unique complete five-column GPU status records required")
            utilization = float(fields[4].removesuffix("%").strip())
            memory = float(fields[3].removesuffix("MiB").strip())
            if not (0 <= utilization <= 100 and 0 <= memory < float("inf")):
                raise ValueError("GPU resource observations must be finite and nonnegative")
            rows[fields[1]] = fields
        checks.check(set(rows) == {g["uuid"] for g in gpus}
                     and all("V100" in row[2] for row in rows.values()),
                     label + ": both original physical V100 cards retain actual utilization/memory evidence")
    except (ValueError, TypeError, KeyError) as exc:
        checks.check(False, label + ": " + str(exc))


def verify_adapter(inputs, original_output):
    """Read-only proof chain; an old failed process is never presented as passed."""
    checks, root = inputs.checks, inputs.roots["extension"]
    original = Path(original_output).resolve()
    if original == root:
        raise ValueError("The retry attempt and original failed output must be separate directories")
    p = inputs.json("extension", "adapter_protocol.json")
    receipt = inputs.json("extension", "adapter_protocol_receipt.json")
    frozen = inputs.json("extension", "protocol.json")
    checks.check(receipt.get("protocol_file") == "adapter_protocol.json"
                 and receipt.get("protocol_sha256") == sha(root / "adapter_protocol.json")
                 and utc(p["prepared_at"]) <= utc(receipt["sealed_at"]),
                 "Separate adapter scope declaration and whole-file seal")
    checks.check(p.get("financial_rules_changed") is False and p.get("no_annual") is True,
                 "Only one existing UI process resource check changes")
    checks.check(p.get("prices_loaded_at_prepare") is False, "Resource declaration was prepared before loading new financial data")
    checks.check(p.get("original_protocol_sha256") == sha(root / "protocol.json")
                 == sha(safe_file(original, "protocol.json"))
                 and p.get("original_protocol_receipt_sha256") == sha(root / "protocol_receipt.json")
                 == sha(safe_file(original, "protocol_receipt.json")),
                 "Retry copies the original declared c30 protocol and receipt without any rewrite")
    source = p.get("source_sha256", {})
    inputs.pin_map("source", source, SOURCE_FILES, "Four actual core/adapter audit and execution sources")
    checks.check(set(source) == set(SOURCE_FILES)
                 and all(source[name] == frozen["source_sha256"][name] for name in SOURCE_FILES[:2]),
                 "Original financial execution and complete financial auditor retain their exact bytes")
    for name in SOURCE_FILES[2:]:
        choices = ("adapter_sources/" + name, "sources/" + name)
        present = [name for name in choices if (root / name).is_file()]
        if not checks.check(bool(present), "Exact archived independent adapter source bytes"):
            continue
        for name in present:
            repo_name = "scripts/" + name.split("scripts/", 1)[1]
            checks.check(sha(inputs.capture("extension", name)) == source[repo_name],
                         "Archived actual source matches sealed adapter declaration:" + name)
    failure_hashes = p.get("original_failed_files", {})
    if not checks.check(isinstance(failure_hashes, dict) and set(failure_hashes) == set(FAILURE_FILES),
                         "All three original failed preflight files preserved"):
        raise ValueError("Partial original failure evidence is not an admissible adapter")
    for name, expected in failure_hashes.items():
        checks.check(sha(safe_file(original, name)) == expected
                     == sha(inputs.capture("extension", "original_failure/" + name)),
                     "Original failed evidence and immutable copy agree:" + name)
    failed = read_json(safe_file(original, "runner_process.json"))
    checks.check(failed.get("status") == "failed" and failed.get("exit_code") != 0
                 and failed.get("source_sha256") == frozen["source_sha256"]["scripts/btceth_rl_execution_band.py"]
                 and failed.get("log_sha256") == failure_hashes["validation.log"],
                 "First preflight failed; no success or process restart is fabricated")
    checks.check(not (original / "validation").exists()
                 and not (original / "validation_completion.json").exists(),
                 "Original failed resource preflight produced no financial cases")
    checks.check(utc(failed["completed_at"]) <= utc(p["prepared_at"]),
                 "Adapter was transparently declared after the original failed attempt")
    instance = validate_instance(p.get("allowed_instance"))
    allowed = p.get("allowed_gpu_rows", [])
    filtered_query("", allowed)
    checks.check(len(allowed) == 1 and {row["gpu_uuid"] for row in allowed}.issubset({g["uuid"] for g in frozen["physical_gpus"]}),
                 "Only the captured UI row on original physical cards may be exempt")
    verify_snapshot(p["before"], instance, allowed, frozen["physical_gpus"], checks, "Declared original preflight snapshot")
    checks.check(utc(p["before"]["at"]) <= utc(p["prepared_at"]), "Real instance snapshot preceded the new scope declaration")
    runtime = inputs.json("extension", "adapter_runtime.json")
    process = inputs.json("extension", "runner_process.json")
    checks.check(runtime.get("status") == "completed" and runtime.get("old_subprocess_run_restored") is True
                 and runtime.get("financial_rules_changed") is False and runtime.get("no_annual") is True,
                 "Resource proxy finished normally and restored the original subprocess function")
    checks.check(runtime.get("adapter_protocol_sha256") == sha(root / "adapter_protocol.json")
                 and runtime.get("error") is None, "Successful actual runtime binds the exact adapter declaration")
    command = process.get("command", [])
    checks.check(len(command) >= 7 and str(command[2]).replace("\\", "/").endswith("/btceth_rl_execution_band_gpu_guard.py")
                 and process.get("actual_entrypoint") == command[2]
                 and "--original-runner" in command
                 and str(command[command.index("--original-runner") + 1]).replace("\\", "/").endswith("/btceth_rl_execution_band.py")
                 and "--phase" in command and command[command.index("--phase") + 1] == "_validate",
                 "Actual child is the disclosed adapter, which explicitly invokes the original core")
    checks.check(process.get("core_source_sha256") == process.get("source_sha256")
                 == source["scripts/btceth_rl_execution_band.py"]
                 and process.get("adapter_source_sha256") == source["scripts/btceth_rl_execution_band_gpu_guard.py"]
                 and bool(process.get("source_adaptation")),
                 "Core-loaded and actual adapter source identities remain separate")
    checks.check(utc(receipt["sealed_at"]) <= utc(process["started_at"])
                 <= utc(runtime["before"]["at"]) <= utc(runtime["after"]["at"])
                 <= utc(process["completed_at"]), "Adapter seal precedes actual bounded resource-check attempt")
    if "pid" in runtime:
        checks.check(runtime["pid"] == process.get("child_pid"), "Runtime evidence belongs to the real private child")
    verify_snapshot(runtime["before"], instance, allowed, frozen["physical_gpus"], checks, "Before original validation")
    verify_snapshot(runtime["after"], instance, allowed, frozen["physical_gpus"], checks, "After original validation")
    events = runtime.get("query_events", [])
    checks.check(len(events) >= 1, "Actual original preflight query was observed by the bounded adapter")
    for i, event in enumerate(events):
        verify_query_event(event, instance, allowed, checks, f"Adapted original query{i}")
        checks.check(utc(runtime["before"]["at"]) <= utc(event["at"]) <= utc(runtime["after"]["at"]),
                     "Query observation lies within actual adapter execution")
    checks.check(not any((root / name).exists() for name in ("test", "test_started.json", "test_completion.json", "test_process.json")),
                 "Adapter cannot dispatch or contain an annual run")
    return {"adapter_protocol_sha256": sha(root / "adapter_protocol.json"),
        "adapter_protocol_receipt_sha256": sha(root / "adapter_protocol_receipt.json"),
        "original_failure_sha256": failure_hashes, "original_failed_attempt_preserved": True,
        "allowed_instance": instance, "actual_query_events": len(events),
        "financial_rules_changed": False, "no_annual": True}


def audit(output, original_output, main, reference, dataset_path, *, source_root=ROOT):
    """Extra resource proofs plus the complete unchanged b4d financial audit."""
    checks = Checks()
    inputs = core.Inputs(output, main, reference, source_root, checks)
    proof, result = {}, None
    try:
        proof = verify_adapter(inputs, original_output)
        if checks.failed:
            raise ValueError("Resource-adapter evidence failed before financial reconstruction")
        result = core.audit(output, main, reference, dataset_path, source_root=source_root)
        for root, values in result["input_sha256"].items():
            for name, expected in values.items():
                checks.check(sha(inputs.capture(root, name)) == expected,
                             "All original financial audit inputs remain unchanged:" + root + ":" + name)
        checks.check(result.get("passed") is True and result.get("status") == "passed"
                     and result.get("run_count") == result.get("comparator_run_count") == result.get("repeat_count") == 12
                     and result.get("test_complete") is False, "Complete unchanged b4d twelve-new/twelve-reference validation audit")
        inputs.unchanged()
        for name, expected in proof["original_failure_sha256"].items():
            checks.check(sha(safe_file(original_output, name)) == expected,
                         "Original failure unchanged after complete financial audit:" + name)
    except Exception as exc:
        checks.check(False, type(exc).__name__ + ": " + str(exc))
    if result is None:
        result = {"passed": False, "status": "failed", "run_count": 0, "repeat_count": 0,
            "comparator_run_count": 0, "checks_total": 0, "failure_count": 0, "failures": [],
            "test_complete": False, "validation_complete": False, "audited_at": core.now()}
    core_checks, core_failures = result["checks_total"], result["failure_count"]
    result["core_financial_audit"] = {"source_sha256": sha(Path(source_root) / SOURCE_FILES[1]),
        "passed": bool(result["passed"]), "checks_total": core_checks, "failure_count": core_failures}
    result["checks_total"] += checks.count
    result["failure_count"] += checks.failed
    result["failures"] = [*result["failures"], *checks.failures]
    result["passed"] = bool(result["passed"] and checks.failed == 0)
    result["status"] = "passed" if result["passed"] else "failed"
    result["validation_complete"] = result["passed"]
    result["input_sha256"], result["source_sha256"] = inputs.hashes, inputs.hashes["source"]
    result["adapter_evidence"], result["resource_adapter_disclosed"] = proof, True
    result["audited_at"] = core.now()
    return result


def dump_new(path, result):
    path = Path(path)
    if path.exists():
        raise FileExistsError("Preserve every previous audit attempt")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("output-dir", "original-output", "main-output", "reference-output", "dataset"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = args.report or args.output_dir / "independent_audit.validation.json"
    if report.exists():
        raise FileExistsError("Preserve the original resource and financial audit evidence")
    result = audit(args.output_dir, args.original_output, args.main_output, args.reference_output, args.dataset)
    dump_new(report, result)
    print(json.dumps({key: result[key] for key in
        ("passed", "status", "run_count", "repeat_count", "comparator_run_count", "checks_total", "failure_count")}))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()

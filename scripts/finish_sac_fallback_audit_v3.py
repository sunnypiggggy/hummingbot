"""Finish the sealed SAC study only when both coins retained the baseline.

Keep the failed v2 import audit and every earlier source unchanged. Import
already audited annual account bytes, aggregate them with the original helpers,
and invoke the separately sealed v3 independent auditor. Never fit or replay.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
import time

DRIVER = "extension_source/scripts/backtest_sac_validation_optimization.py"
PARENT_CONTROLLER = "audit_revision_source/scripts/continue_sac_validation_optimization.py"
AUDITOR = "audit_import_revision_source/scripts/audit_sac_validation_optimization_v3.py"
FINISHER = "audit_import_revision_source/scripts/finish_sac_fallback_audit_v3.py"
VALIDATION_AUDIT = "final/validation_optimization_audit_v3.json"
TEST_AUDIT = "final/test_optimization_audit_v3.json"
FAILED_AUDIT = "final/validation_optimization_audit.json"
REVISION_ID = "retained_import_population_shadowing_v3"
REASON = "import_manifest_expected_digest_shadowing"
PERMITTED_CHANGE = "rename inner imported-file expected digest; preserve exact population and byte hashes"
PARENT_AUDITOR_SHA256 = "a53f412c748899bec6cc961316eedad52c354f522248a5f8b9d6d99b279ab468"
SYMBOLS = ("BTCUSDT", "ETHUSDT")
CANDIDATES = ("c00_baseline", "c01_lr1e4", "c02_gamma099", "c03_lr1e4_gamma099")
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
        raise ValueError("Canonical study-relative evidence paths are required")
    path = top.joinpath(*part.parts)
    if any(p.is_symlink() or hasattr(p, "is_junction") and p.is_junction() for p in (path, *path.parents)):
        raise ValueError("Frozen evidence cannot cross a link")
    path.resolve().relative_to(top.resolve())
    return path


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def write_status(top, **fields):
    path = safe(top, "pipeline_status.json")
    pending = path.with_suffix(".import-revision-pending")
    write_new(pending, {"at": now(), "pid": os.getpid(), "version": "fallback-audit-v3", "training_restarted": False, **fields})
    os.replace(pending, path)


def load_source(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_files(top, mapping):
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError("A nonempty exact evidence hash map is required")
    for name, expected_sha in mapping.items():
        if sha(safe(top, name)) != expected_sha:
            raise ValueError("Bound study evidence changed: " + name)


def verify_audit(top, path, expected_source, *, final=False):
    value = read(safe(top, path))
    if (value.get("passed") is not True or value.get("failure_count") != 0 or value.get("failures")
            or value.get("audit_source_sha256") != expected_source
            or value.get("protocol_sha256") != sha(safe(top, "protocol.json"))):
        raise ValueError("An exact successful versioned audit is required: " + path)
    verify_files(top, value.get("inputs_sha256"))
    for name, expected_sha in value.get("external_inputs_sha256", {}).items():
        if sha(name) != expected_sha:
            raise ValueError("Bound original reference evidence changed")
    if final and value.get("core_financial_audit_passed") is not True:
        raise ValueError("Original independent financial arithmetic must also pass")
    return value


def verify_retained_final(top):
    prepared, counter, fit = (read(safe(top, name)) for name in
        ("final_prepared.json", "final/training_counter_audit.json", "final/fit_complete.json"))
    expected = {(s, n) for s in SYMBOLS for n in range(30)}
    if (prepared.get("retained_models") != 60 or prepared.get("new_models") != 0
            or prepared.get("selection_sha256") != sha(safe(top, "selection.json"))
            or prepared.get("final_protocol_sha256") != sha(safe(top, "final/protocol.json"))
            or counter.get("passed") is not True or counter.get("failure_count") != 0 or counter.get("failures")
            or counter.get("models_checked") != 60 or counter.get("retained_models") != 60 or counter.get("new_models") != 0
            or fit.get("model_count") != 60 or len(fit.get("records", [])) != 60
            or read(safe(top, "final_fit_complete.json")).get("fresh_tasks") != 0
            or read(safe(top, "final_validation_complete.json")).get("fresh_tasks") != 0):
        raise ValueError("Only sixty complete retained models and zero new final work are permitted")
    extension = read(safe(top, "extension_manifest.json"))
    if (counter.get("core_protocol_sha256") != sha(safe(top, "final/protocol.json"))
            or counter.get("parent_protocol_sha256") != sha(safe(top, "protocol.json"))
            or counter.get("extension_source_sha256") != extension["extension_sources"]["scripts/backtest_sac_validation_optimization.py"]
            or read(safe(top, "final_fit_complete.json")).get("child_counter_audit_sha256") != {"final": sha(safe(top, "final/training_counter_audit.json"))}):
        raise ValueError("Retained native training evidence must retain its original source bindings")
    final = read(safe(top, "final/protocol.json"))
    baseline = read(safe(top, "protocol.json"))["base_recipe"]
    if (final.get("recipe_ids_by_symbol") != {s: "c00_baseline" for s in SYMBOLS}
            or final.get("recipes_by_symbol") != {s: baseline for s in SYMBOLS}
            or final.get("seeds") != list(range(30)) or final.get("symbols") != list(SYMBOLS)
            or final.get("algorithms") != ["sac"] or final.get("capital_per_account") != 10000.
            or final.get("capital_shared_between_pairs") is not False):
        raise ValueError("Final accounts must use their unchanged individual baseline recipes")
    rows = {(r["symbol"], r["seed"]): r for r in counter.get("models", [])}
    manifest = read(safe(top, "final/import_manifest.json"))
    imports = {(r["symbol"], r["seed"]): r for r in manifest.get("entries", [])}
    if len(rows) != 60 or set(rows) != expected or len(imports) != 60 or set(imports) != expected:
        raise ValueError("The exact retained seed population is required")
    if manifest.get("selection_sha256") != sha(safe(top, "selection.json")):
        raise ValueError("Retained models must follow the frozen selection")
    records = []
    for symbol, seed in sorted(expected):
        tail = f"training/{symbol}/sac/seed{seed}"
        entry = imports[(symbol, seed)]
        source, destination = "references/" + symbol + "/" + tail, "final/" + tail
        if (entry.get("origin") != "retained_baseline_fallback" or entry.get("source_directory") != source
                or entry.get("destination_directory") != destination):
            raise ValueError("No new training may be disguised as retained fallback")
        files = entry.get("file_sha256")
        verify_files(top, {source + "/" + k: v for k, v in files.items()})
        verify_files(top, {destination + "/" + k: v for k, v in files.items()})
        for directory in (source, destination):
            actual = {p.relative_to(safe(top, directory)).as_posix() for p in safe(top, directory).rglob("*") if p.is_file()}
            if actual != set(files):
                raise ValueError("The complete original model directory must be retained")
        record = read(safe(top, destination + "/record.json")); records.append(record)
        if (rows[(symbol, seed)].get("record_sha256") != sha(safe(top, destination + "/record.json"))
                or rows[(symbol, seed)].get("retained") is not True
                or rows[(symbol, seed)].get("training_origin") != "retained_baseline_fallback"
                or rows[(symbol, seed)].get("counters") != record["training"]["counters"]
                or record.get("symbol") != symbol or record.get("algorithm") != "sac" or record.get("seed") != seed):
            raise ValueError("Retained native counter records must match original bytes")
    if sorted(fit["records"], key=lambda r: (r["symbol"], r["seed"])) != sorted(records, key=lambda r: (r["symbol"], r["seed"])):
        raise ValueError("Final fit receipt must contain the exact retained records")


def verify_contract(top, dataset):
    top = Path(top)
    revision = read(safe(top, "audit_import_revision.json"))
    expected = {"schema_version": 1, "revision_id": REVISION_ID, "reason_code": REASON,
        "parent_auditor_source_sha256": PARENT_AUDITOR_SHA256,
        "revised_audit_source_path": AUDITOR, "finisher_source_path": FINISHER,
        "permitted_change": PERMITTED_CHANGE, "financial_math_changed": False, "model_recipe_changed": False,
        "selection_changed": False, "training_restarted": False}
    for key, value in expected.items():
        if revision.get(key) != value or isinstance(value, bool) and type(revision.get(key)) is not bool:
            raise ValueError("Only the precise import-check variable correction is authorized: " + key)
    bindings = {"original_protocol_sha256": "protocol.json", "search_plan_sha256": "search_plan.json",
        "parent_audit_revision_sha256": "audit_revision.json", "revised_audit_source_sha256": AUDITOR,
        "finisher_source_sha256": FINISHER, "failed_final_validation_audit_sha256": FAILED_AUDIT,
        "selection_sha256": "selection.json", "selection_audit_sha256": "selection_audit.json"}
    for key, name in bindings.items():
        if revision.get(key) != sha(safe(top, name)):
            raise ValueError("Original or versioned immutable evidence changed: " + name)
    receipt = read(safe(top, "audit_import_revision_receipt.json"))
    if (receipt.get("sha256") != sha(safe(top, "audit_import_revision.json"))
            or datetime.fromisoformat(receipt["at_utc"]) < datetime.fromisoformat(revision["created_at_utc"])
            or sha(__file__) != revision["finisher_source_sha256"]):
        raise ValueError("The actual finisher must match its sealed revision receipt")
    parent_revision = read(safe(top, "audit_revision.json"))
    if (sha(safe(top, PARENT_CONTROLLER)) != parent_revision["controller_source_sha256"]
            or parent_revision["revised_audit_source_sha256"] != PARENT_AUDITOR_SHA256):
        raise ValueError("Preserve the preceding frozen controller and auditor")
    parent = load_source(safe(top, PARENT_CONTROLLER), "_sac_fallback_parent_controller")
    parent.verify_revision(top, dataset)
    parent.validate_finished_tuning(top)
    launch = read(safe(top, "pipeline_launch_revision.json"))
    running = parent.process_identity(launch["pid"])
    if running is not None and running["state"] != "Z":
        raise ValueError("Do not run alongside the preceding continuation controller")
    for name in ["tuning/" + c + "/validation_optimization_audit.json" for c in CANDIDATES] + ["selection_audit.json"]:
        verify_audit(top, name, PARENT_AUDITOR_SHA256)
    selection = read(safe(top, "selection.json")); baseline = read(safe(top, "protocol.json"))["base_recipe"]
    if (set(selection.get("selection", {})) != set(SYMBOLS)
            or any(selection["selection"][s].get("candidate_id") != "c00_baseline"
                or selection["selection"][s].get("fallback_to_baseline") is not True
                or selection["selection"][s].get("selected_recipe") != baseline for s in SYMBOLS)
            or selection.get("uses_test") is not False or selection.get("uses_final_validation") is not False
            or selection.get("seed_selection") is not False or selection.get("algorithm_selection") is not False):
        raise ValueError("This finisher only supports the already frozen two-coin c00 fallback")
    failed = read(safe(top, FAILED_AUDIT))
    if (failed.get("passed") is not False or failed.get("failure_count") != 1
            or failed.get("phase") != "final-validation" or failed.get("audit_source_sha256") != PARENT_AUDITOR_SHA256
            or failed.get("failures") != ["AttributeError: 'str' object has no attribute 'get'"]):
        raise ValueError("Preserve the exact diagnosed v2 failure; no broader recovery is permitted")
    verify_retained_final(top)
    return revision


def wait_for_launch(top, revision, *, timeout=5.):
    deadline = time.monotonic() + timeout
    while True:
        try:
            launch = read(safe(top, "pipeline_launch_import_revision.json")); break
        except (FileNotFoundError, json.JSONDecodeError):
            if time.monotonic() >= deadline:
                raise ValueError("The parent must seal the new finisher PID and command")
            time.sleep(.05)
    fields = {"pid": os.getpid(), "audit_import_revision_sha256": sha(safe(top, "audit_import_revision.json")),
        "finisher_source_sha256": revision["finisher_source_sha256"], "revised_audit_source_sha256": revision["revised_audit_source_sha256"],
        "parent_pipeline_launch_revision_sha256": sha(safe(top, "pipeline_launch_revision.json")), "training_restarted": False}
    if any(launch.get(k) != v for k, v in fields.items()):
        raise ValueError("The launch must belong to this exact fallback finisher and revision")
    if str(Path(__file__).resolve()) not in launch.get("command", []):
        raise ValueError("The exact sealed finisher command is required")
    return launch


def annual_bindings(top, revision):
    shared = {"validation_optimization_audit_path": VALIDATION_AUDIT,
        "active_auditor_source_sha256": revision["revised_audit_source_sha256"],
        "audit_import_revision_sha256": sha(safe(top, "audit_import_revision.json"))}
    pins = {"selection_sha256": sha(safe(top, "selection.json")), "selection_audit_sha256": sha(safe(top, "selection_audit.json")),
        "validation_audit_sha256": sha(safe(top, "final/validation_audit.json")),
        "validation_optimization_audit_sha256": sha(safe(top, VALIDATION_AUDIT)),
        "protocol_sha256": sha(safe(top, "protocol.json")), "final_protocol_sha256": sha(safe(top, "final/protocol.json"))}
    compatible = {"uses_test": False, "seed_selection": False, "algorithm_selection": False, "mode": "fixed_sac_recipe_comparison",
        "selection": {s: {"algorithm": "sac", "seed": 0} for s in SYMBOLS}, "top_selection_sha256": pins["selection_sha256"],
        "protocol_sha256": pins["final_protocol_sha256"], "validation_audit_sha256": pins["validation_audit_sha256"],
        "validation_optimization_audit_sha256": pins["validation_optimization_audit_sha256"], **shared}
    return compatible, pins, shared


def aggregate_imported_annual(top, runner):
    import pandas as pd
    reporting_path = safe(top, "frozen_source/scripts/crypto_rl_reporting.py")
    expected_sha = read(safe(top, "protocol.json"))["sources"]["scripts/crypto_rl_reporting.py"]
    if sha(reporting_path) != expected_sha:
        raise ValueError("Use the original frozen CSV serialization")
    reporting = load_source(reporting_path, "_sac_fallback_frozen_reporting")
    child, spec = safe(top, "final"), runner.check_child(top, safe(top, "final"))
    if spec["seeds"] != list(range(30)) or set(spec["costs"]) != {"base", "stress"}:
        raise ValueError("Annual aggregation requires the original sixty-seed two-cost population")
    results = [runner._case_result(child, "test", symbol, algorithm, seed, scenario)
        for symbol in SYMBOLS for algorithm, seed in [("sac", n) for n in range(30)] + [(a, None) for a in runner.BASELINES]
        for scenario in spec["costs"]]
    if len(results) != 136 or len({r["path"] for r in results}) != 136:
        raise ValueError("Exactly136 original annual accounts and their repeats are required")
    summary = pd.DataFrame([r["summary"] for r in results]).sort_values("run_id")
    return results, reporting.canonical_csv(summary)


def import_annual(top, dataset, revision):
    verify_contract(top, dataset)
    verify_audit(top, VALIDATION_AUDIT, revision["revised_audit_source_sha256"], final=True)
    runner = load_source(safe(top, DRIVER), "_sac_fallback_frozen_runner")
    compatible, pins, shared = annual_bindings(top, revision)
    complete = safe(top, "final_test_complete.json")
    if complete.exists():
        started, actual_compatible = read(safe(top, "final/test_started.json")), read(safe(top, "final/validation_selection.json"))
        if (started.get("pins") != pins or any(started.get(k) != v for k, v in shared.items())
                or any(actual_compatible.get(k) != v for k, v in compatible.items())
                or started.get("oracle_compatibility_selection_sha256") != sha(safe(top, "final/validation_selection.json"))):
            raise ValueError("Previously imported annual gate changed")
        manifest = read(safe(top, "final/test_import_manifest.json"))
        if manifest.get("selection_sha256") != pins["selection_sha256"] or len(manifest.get("entries", [])) != 272:
            raise ValueError("Complete original annual import population required")
        for entry in manifest["entries"]:
            for directory in (entry["source_directory"], entry["destination_directory"]):
                verify_files(top, {directory + "/" + k: v for k, v in entry["file_sha256"].items()})
        results, csv = aggregate_imported_annual(top, runner)
        prior, aggregate = read(safe(top, "final/test_complete.json")), read(complete)
        if (safe(top, "final/test_summary.csv").read_bytes() != csv or prior.get("results") != results
                or prior.get("accounts") != 136 or prior.get("repeats") != 136 or prior.get("repeat_equal") is not True
                or aggregate.get("fresh_tasks") != 0 or aggregate.get("child_summary_sha256") != {"final": sha(safe(top, "final/test_summary.csv"))}):
            raise ValueError("Previously aggregated retained annual accounts changed")
        return {"fresh_accounts": 0, "accounts": 136, "referenced_existing": True}
    new_paths = ("final/validation_selection.json", "final/test_started.json", "reference_annual_import.json",
        "final/test_import_manifest.json", "final/evaluation/test", "final/repeat/test", "final/test_summary.csv", "final/test_complete.json")
    if any(safe(top, n).exists() for n in new_paths):
        raise FileExistsError("Preserve partial annual evidence; do not repeat or overwrite an attempted import")
    write_new(safe(top, "final/validation_selection.json"), {"at": now(), **compatible})
    write_new(safe(top, "final/test_started.json"), {"at": now(), "pins": pins,
        "oracle_compatibility_selection_sha256": sha(safe(top, "final/validation_selection.json")), **shared})
    runner.import_final_cases(top, "test")
    results, csv = aggregate_imported_annual(top, runner)
    with safe(top, "final/test_summary.csv").open("xb") as stream:
        stream.write(csv)
    write_new(safe(top, "final/test_complete.json"), {"at": now(), "accounts": 136, "repeats": 136,
        "repeat_equal": True, "results": results})
    write_new(complete, {"at": now(), "fresh_tasks": 0,
        "child_summary_sha256": {"final": sha(safe(top, "final/test_summary.csv"))}})
    return {"fresh_accounts": 0, "accounts": 136, "referenced_existing": False}


def run_audit(top, dataset, revision, phase, marker, *, workers=None):
    verify_contract(top, dataset)
    if safe(top, marker).exists():
        return verify_audit(top, marker, revision["revised_audit_source_sha256"], final=phase != "completion")
    command = [sys.executable, "-B", str(safe(top, AUDITOR)), "--study-root", str(top), "--dataset", str(dataset), "--phase", phase]
    if workers and phase != "completion":
        command += ["--workers", str(workers)]
    logs = safe(top, "audit_import_revision_logs"); logs.mkdir(exist_ok=True)
    with (logs / (phase + ".log")).open("x", encoding="utf-8") as stream:
        result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode or not safe(top, marker).exists():
        raise RuntimeError("Fallback v3 independent audit failed; preserve all evidence: " + phase)
    return verify_audit(top, marker, revision["revised_audit_source_sha256"], final=phase != "completion")


def finish(top, dataset, *, workers=None):
    global _STATUS_OWNED
    top, dataset = Path(top).resolve(), Path(dataset).resolve()
    revision = verify_contract(top, dataset)
    wait_for_launch(top, revision)
    _STATUS_OWNED = True
    for phase, marker in (("final-validation", VALIDATION_AUDIT), ("import-annual", "final_test_complete.json"),
                          ("final-test", TEST_AUDIT), ("completion", "completion_contract_audit.json")):
        write_status(top, phase="fallback-v3:" + phase, status="running",
            audit_import_revision_sha256=sha(safe(top, "audit_import_revision.json")))
        try:
            if phase == "import-annual":
                import_annual(top, dataset, revision)
            else:
                run_audit(top, dataset, revision, phase, marker, workers=workers)
        except BaseException as exc:
            write_status(top, phase="fallback-v3:" + phase, status="failed", error_type=type(exc).__name__,
                error=str(exc), all_evidence_preserved=True)
            raise
    verify_contract(top, dataset)
    write_status(top, phase="fallback-v3:completion", status="completed", no_production_change=True,
        audit_import_revision_sha256=sha(safe(top, "audit_import_revision.json")), retained_models=60, new_models=0, fresh_annual_accounts=0)
    return {"completed": True, "retained_models": 60, "new_models": 0, "fresh_annual_accounts": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", "--study-root", dest="output", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.workers is not None and args.workers < 1:
        parser.error("A positive independent audit worker count is required")
    try:
        if args.verify_only:
            verify_contract(args.output, args.dataset)
            result = {"verified": True, "read_only": True, "retained_models": 60, "new_models": 0}
        else:
            result = finish(args.output, args.dataset, workers=args.workers)
    except BaseException as exc:
        if _STATUS_OWNED and read(safe(args.output, "pipeline_status.json")).get("status") != "failed":
            write_status(args.output, phase="fallback-v3", status="failed", error_type=type(exc).__name__, error=str(exc), all_evidence_preserved=True)
        raise
    print(json.dumps(result))


if __name__ == "__main__":
    main()

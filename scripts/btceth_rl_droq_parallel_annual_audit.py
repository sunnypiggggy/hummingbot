"""Full26 audit for the explicitly declared two-GPU DroQ annual execution.

The sealed52a validation auditor stays immutable. This module independently
checks the true annual process/resource chain, adopts original bf33 CPU case
reconstructions, rechecks the14 original controls and calls the original five
research gates. No model is trained or inferred and no order is submitted.
"""
from __future__ import annotations
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

BASE_SHA = "52aace871ee5d04409536b3e1b8d2f13468bb37894099e1a664dd722634732e6"
EXECUTION_SHA = "ad1b84e28b84e6c59c454cc92443917d6edc267cf1d4eecb6d373c34c47774fd"
VALIDATION_AUDIT = "independent_parallel_audit.validation.json"
MODE = "parallel_cases_explicit_two_gpu_annex"


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def need(condition, label):
    if not condition:
        raise ValueError(label)


def load(root, name, expected):
    path = Path(root) / name
    need(sha(path) == expected, "Declared independent execution/audit source changed:" + name)
    alias = "_droq_annual_" + path.stem
    if str(Path(root).resolve()) not in sys.path:
        sys.path.insert(0, str(Path(root).resolve()))
    if alias not in sys.modules:
        spec = importlib.util.spec_from_file_location(alias, path)
        module = importlib.util.module_from_spec(spec); sys.modules[alias] = module
        spec.loader.exec_module(module)
    need(Path(sys.modules[alias].__file__).resolve() == path.resolve(), "Foreign annual adapter import")
    return sys.modules[alias]


def test_files(family):
    need(family in ("raw", "cap075"), "Only the selected fixed family is annual-tested")
    return {"selection.json", "selection_receipt.json", VALIDATION_AUDIT,
        "parallel_annual_resource_annex.json", "parallel_annual_resource_receipt.json", "annual_authorization.json",
        "test_started.json", "test_completion.json", "parallel_test_process.json", "parallel_test.owner.json",
        "parallel_test_runner.log", "parallel_test_resource_preflight.json",
        *(f"test/{family}/balanced/{name}" for name in ("summary.csv", "daily_equity.csv", "monthly_returns.csv", "completion.json")),
        *(f"test/{family}/balanced/{cost}/{key}/{name}" for cost in ("base", "stress")
          for key in ("0", "1", "2", "3", "4", "ensemble")
          for name in ("equity_1h.csv.gz", "fills.csv", "weights.csv", "boundary_equity.csv", "metadata.json", "fingerprints.json", "member_targets.csv", "replay.json"))}


def verify_annual_process(inputs, bf, spec, selection, annex, execution):
    p = inputs.json("study", "parallel_test_process.json")
    complete = inputs.json("study", "test_completion.json")
    owner = inputs.json("study", "parallel_test.owner.json")
    marker = inputs.json("study", "test_started.json")
    resource = inputs.json("study", "parallel_test_resource_preflight.json")
    expected = annex["case_gpu_assignments"]
    need(p.get("scope") == "explicit_droq_two_gpu_selected_annual" and p.get("phase") == "test"
         and p.get("status") == "completed" and type(p.get("exit_code")) is int and p["exit_code"] == 0
         and p.get("source_sha256") == EXECUTION_SHA and p.get("parallel_source_sha256") == execution.PARALLEL_SHA
         and p.get("original_runner_sha256") == execution.ORIGINAL_SHA and p.get("total_cases") == 12
         and p.get("max_workers_per_gpu") == 6 and p.get("case_gpu_assignments") == expected,
         "Real declared two-GPU annual supervisor/child completion0 required")
    need(type(p.get("supervisor_pid")) is int and p["supervisor_pid"] > 0
         and type(p.get("child_pid")) is int and p["child_pid"] > 0
         and p["supervisor_pid"] != p["child_pid"] == p.get("coordinator_pid")
         and owner.get("pid") == marker.get("supervisor_pid") == p["supervisor_pid"]
         and owner.get("source_sha256") == EXECUTION_SHA and isinstance(owner.get("token"), str) and len(owner["token"]) == 32,
         "Actual distinct annual parent/child and exclusive owner/marker identity required")
    command = p.get("command", [])
    need(len(command) >= 6 and Path(command[3]).name == "btceth_rl_droq_parallel_execution.py"
         and command.count("--phase") == 1 and command[command.index("--phase") + 1] == "_test"
         and command.count("--audit-source-sha") == 1 and command[command.index("--audit-source-sha") + 1] == BASE_SHA,
         "Actual separately source-bound annual private-child command required")
    for field, name in (("protocol_sha256", "protocol.json"), ("selection_sha256", "selection.json"),
            ("authorization_sha256", "annual_authorization.json"), ("test_marker_sha256", "test_started.json"),
            ("resource_annex_sha256", "parallel_annual_resource_annex.json"), ("completion_sha256", "test_completion.json")):
        need(p.get(field) == sha(inputs.capture("study", name)), "Actual annual process immutable dependency mismatch:" + name)
    need(p.get("log_relative_path") == "parallel_test_runner.log"
         and p.get("log_sha256") == sha(inputs.capture("study", "parallel_test_runner.log")), "Actual annual child log bytes changed")
    need(complete.get("protocol_sha256") == p["protocol_sha256"] and complete.get("selection_sha256") == p["selection_sha256"]
         and complete.get("candidate") == selection["candidate"] and complete.get("case_count") == complete.get("repeat_count") == 12
         and complete.get("coordinator_pid") == p["child_pid"] and complete.get("execution_mode") == MODE
         and complete.get("execution_source_sha256") == EXECUTION_SHA and complete.get("resource_annex_sha256") == p["resource_annex_sha256"]
         and complete.get("case_gpu_assignments") == expected, "Only the actual sealed candidate's twelve repeated accounts may complete")
    bf.verify_gpu_snapshot(resource.get("physical_gpus", []), spec["physical_gpus"], inputs.checks, "Annual resource preflight")
    need(resource.get("command") == ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"], "Unfiltered original annual GPU command required")
    rows = [{"gpu_uuid": line.split(",")[0].strip(), "pid": int(line.split(",")[1])}
            for line in resource.get("unfiltered_stdout", "").splitlines() if line.strip()]
    need(rows == resource.get("rows") and (not rows or rows == spec.get("allowed_gpu_rows")), "No undeclared GPU task may be hidden at annual startup")
    if rows:
        allowed = spec["allowed_compute_instance"]; actual = resource["instances"][str(allowed["pid"])]
        need(all(actual.get(k) == v for k, v in allowed.items()), "Original reviewed UI instance changed")
    times = [owner["at"], marker["at"], p["started_at"], resource["at"], complete["at"], p["completed_at"]]
    need(all(bf.utc(a) <= bf.utc(b) for a, b in zip(times, times[1:])), "Annual owner, approval marker, resource snapshot and genuine terminal chronology changed")
    return p, complete


def audit_group(inputs, bf, process, completion, annex, expected, *, family):
    folder = f"test/{family}/balanced"
    summary, daily, monthly = [inputs.csv("study", folder + "/" + name)
                              for name in ("summary.csv", "daily_equity.csv", "monthly_returns.csv")]
    bf.validate_group_rows(summary, split="test", family=family, profile="balanced")
    group = inputs.json("study", folder + "/completion.json")
    need(group.get("case_count") == group.get("repeat_count") == 12 and group.get("execution_mode") == MODE
         and group.get("device") is None and group.get("worker_pid") is None
         and group.get("coordinator_pid") == process["coordinator_pid"]
         and group.get("execution_source_sha256") == EXECUTION_SHA
         and group.get("protocol_sha256") == process["protocol_sha256"]
         and group.get("resource_annex_sha256") == process["resource_annex_sha256"]
         and group.get("case_gpu_assignments") == annex["case_gpu_assignments"], "Truthful two-card group and exact twelve-case annex required")
    proofs = group.get("repeats", []); workers = group.get("case_workers", [])
    by_id = {p.get("run_id"): p for p in proofs}; worker_map = {p.get("run_id"): p for p in workers}
    need(len(proofs) == len(by_id) == len(workers) == len(worker_map) == 12
         and set(by_id) == set(worker_map) == set(summary.run_id) == set(annex["case_gpu_assignments"]),
         "Exactly the twelve selected-model accounts/repeats/workers are required")
    subset, devices = {}, {}
    for row in summary.to_dict("records"):
        run_id = row["run_id"]; key = "ensemble" if row["primary_policy"] else str(int(row["seed"]))
        relative = f"{folder}/{row['scenario']}/{key}"
        proof = inputs.json("study", relative + "/replay.json")
        assigned = annex["case_gpu_assignments"][run_id]
        need(proof == by_id[run_id] and type(proof.get("worker_pid")) is int and proof["worker_pid"] > 0
             and proof["worker_pid"] not in (process["supervisor_pid"], process["coordinator_pid"])
             and proof.get("device", {}).get("visible_gpu_uuid") == proof.get("assigned_gpu_uuid") == assigned
             and proof.get("execution_mode") == MODE and proof.get("execution_source_sha256") == EXECUTION_SHA
             and proof.get("resource_annex_sha256") == process["resource_annex_sha256"]
             and worker_map[run_id] == {k: proof[k] for k in ("run_id", "worker_pid", "device")},
             "Actual per-case PID/card/annex differs from the immutable task assignment")
        need(devices.setdefault(assigned, proof["device"]) == proof["device"], "Per-card runtime evidence changed")
        need(run_id in expected, "Original financial reconstruction receipt missing")
        run, history, months = expected[run_id]
        run.update(relative_directory=relative, family=family, profile="balanced", scenario=row["scenario"])
        # The original worker already checked every summary metric. Recheck the
        # goal-sensitive metrics against its independent ledger here as well.
        bf._check_summary(row, {k: run[k] for k in ("net_return", "max_drawdown", "fees", "final_equity")}, inputs.checks, run_id)
        subset[run_id] = (run, history, months)
    need(len(devices) == 2 and group.get("devices") == [devices[key] for key in sorted(devices)]
         and group.get("worker_pids") == sorted({p["worker_pid"] for p in proofs}), "Both genuine cards and complete worker population required")
    handle = {k: group[k] for k in ("device", "devices", "case_workers", "worker_pids")}
    need(completion.get("workers") == [handle] and completion.get("case_workers") == workers,
         "Top-level annual handles must match real per-case completions")
    need(bf.utc(process["started_at"]) <= bf.utc(group["at"]) <= bf.utc(completion["at"]), "Real group completion lies outside the annual process")
    bf.compare_group_tables(summary, daily, monthly, subset, inputs.checks, folder)
    return summary, daily, monthly, subset


def audit(output, main, dataset_path, *, source_root, extra_source_root, receipt_root, phase="test"):
    need(phase == "test", "Use the already sealed independent52a validation audit for validation")
    base = load(extra_source_root, "btceth_rl_droq_parallel_audit.py", BASE_SHA)
    execution = load(extra_source_root, "btceth_rl_droq_parallel_execution.py", EXECUTION_SHA)
    worker = base.load_extra(extra_source_root, "btceth_rl_droq_case_audit_v2.py")
    bf = worker.load_frozen(source_root)
    checks = bf.Checks(); inputs = bf.Inputs(output, main, source_root, checks)
    started = datetime.now(timezone.utc).isoformat()
    records, runs, baseline_runs, training, research, envelope = {}, {}, {}, None, None, None
    validation_count = 0
    try:
        spec, old = bf.verify_protocol(inputs, dataset_path)
        # This is the explicit immutable execution adapter's gate, never the
        # legacy fabricated-fit/validate terminal path.
        _, _, selection, annex, auth = execution.annual_gate(main, dataset_path, output, source_root,
            extra_source_root, BASE_SHA, require_new=False)
        previous = inputs.json("study", VALIDATION_AUDIT)
        need(previous.get("passed") is True and previous.get("status") == "partial_test_pending"
             and previous.get("training_model_count") == 5 and previous.get("validation_run_count") == previous.get("repeat_count") == 24
             and previous.get("failure_count") == 0 and previous.get("failures") == [], "Original independently completed24-case validation audit required")
        validation_count = 24
        for root, key in (("study", "input_sha256"), ("main", "main_input_sha256"), ("source", "source_sha256")):
            inputs.pin(root, previous[key], previous[key], "Unchanged sealed validation evidence:" + root)
        envelope = base.verify_schedule(inputs, bf, dataset_path, source_root, extra_source_root)
        base.verify_parallel_validation(inputs, bf, spec, envelope)
        process, complete = verify_annual_process(inputs, bf, spec, selection, annex, execution)
        need(checks.failed == 0, "Source, seal, true scheduling or annual metadata checks failed before market/tensor reads")
        data = bf.load_market(dataset_path, bf.TEST[1])
        features = bf.prior.independent_market_features(data, cutoff=bf.TEST[1])
        import pandas as pd
        import numpy as np
        hours = pd.date_range(bf.utc(bf.TEST[0]), bf.utc(bf.TEST[1]), freq="h", inclusive="left")
        for field in ("open", "high", "low", "close", "volume"):
            values = data["hourly_" + field].reindex(hours).to_numpy(dtype=float)
            checks.check(np.isfinite(values).all() and (values >= 0 if field == "volume" else values > 0).all(), "Complete original two-coin annual hourly field:" + field)
        records, training = bf.audit_training(base.TrainingTimeBounds(inputs, envelope), spec, data, features)
        expected = base.adopt_case_receipts(inputs, bf, receipt_root, dataset_path, phase="test")
        chosen = selection["candidate"]
        group = audit_group(inputs, bf, process, complete, annex, expected, family=chosen["family"])
        runs = {key: value[0] for key, value in group[3].items()}
        baseline, baseline_runs = bf.audit_baselines(inputs, data, old)
        if checks.failed == 0:
            research = bf.performance_evidence(group[0], baseline, chosen)
        for name in test_files(chosen["family"]):
            inputs.capture("study", name)
        inputs.unchanged()
    except Exception as error:
        checks.check(False, type(error).__name__ + ": " + str(error))
    passed = checks.failed == 0 and len(records) == 5 and len(runs) == 12 and len(baseline_runs) == 14
    extensions = {name: sha(Path(extra_source_root) / name) for name in base.PINS}
    extensions.update({"btceth_rl_droq_parallel_audit.py": BASE_SHA,
                       "btceth_rl_droq_parallel_execution.py": EXECUTION_SHA,
                       Path(__file__).name: sha(__file__)})
    return dict(schema_version=1, started_at=started, audited_at=datetime.now(timezone.utc).isoformat(), phase="test", passed=passed,
        status="passed" if passed else "failed", validation_complete=validation_count == 24, test_complete=passed,
        training_model_count=len(records), run_count=len(runs) + len(baseline_runs), validation_run_count=validation_count,
        test_run_count=len(runs), baseline_run_count=len(baseline_runs), repeat_count=len(runs) if passed else 0,
        checks_total=checks.count, failure_count=checks.failed, failures=checks.failures, maximum_errors=checks.maximum_errors,
        input_sha256=inputs.hashes["study"], main_input_sha256=inputs.hashes["main"], source_sha256=inputs.hashes["source"],
        extension_source_sha256=extensions, dataset_sha256=sha(dataset_path), training_calendar=training,
        runs=runs, baseline_runs=baseline_runs, declared_research_targets=bf.TARGETS, research_targets=research,
        derived_training_time_bounds=envelope, original_failed_fit_relabelled=False, scheduling_adapter_explicit=True,
        financial_functions_unchanged=True, frozen_financial_auditor_sha256=worker.FROZEN_AUDITOR_SHA,
        fresh_holdout_success=False, training_budget_multiple=1, recipe_is_pure_official_default=False,
        accounting_passage_is_not_goal_success=True, **{flag: True for flag in bf.FLAGS})


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("output-dir", "reference-main", "dataset", "source-root", "extra-source-root", "receipt-root", "report"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--phase", choices=("test",), default="test")
    args = p.parse_args()
    if args.report.exists():
        raise FileExistsError("Never overwrite an independent annual financial audit")
    result = audit(args.output_dir, args.reference_main, args.dataset, source_root=args.source_root,
        extra_source_root=args.extra_source_root, receipt_root=args.receipt_root, phase=args.phase)
    with args.report.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, indent=2, allow_nan=False); stream.write("\n")
    print(json.dumps({k: result[k] for k in ("passed", "status", "training_model_count", "test_run_count", "baseline_run_count", "checks_total", "failure_count", "failures")}))
    raise SystemExit(0 if result["passed"] else 1)

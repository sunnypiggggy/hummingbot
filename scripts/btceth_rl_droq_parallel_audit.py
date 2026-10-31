"""Explicit scheduling adapter around the unchanged bf33 financial auditor.

Validation first: original protocol, five real tensor/Adam/replay checks and
all24 independently reconstructed accounts remain mandatory. The adapter
verifies genuine parallel PID/GPU evidence instead of impersonating legacy
single-worker terminals. A derived training envelope is used ONLY as the
time bound read by the original audit_training; no file is rewritten.
"""
from __future__ import annotations
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

PINS = {
    "btceth_rl_droq_parallel_eval.py": "7616aa59606a24d14a2b69fb4a9774baa6096d71015a42d5f8f2da5b2effe38d",
    "btceth_rl_droq_scheduling_audit.py": "2e7f61ff1015b0449c17b19e37836cbba3a6606e3cec40f9eb5b726d65572be7",
    "btceth_rl_droq_scheduling.py": "52ba244d55abf36fc49cdbaa8374c179c939dc35b07f3d27bbd8c7a07b233453",
    "wait_then_parallel_validation.py": "57c0f8f6ac602b38377df3b9df68d78adbd6271645be1979a7ee80c0e5859669",
    "btceth_rl_droq_case_audit_v2.py": "f15ee7e4f04f693a6c07d5b0c1685b5cecf8d0eeb899783f3c302e3fad162b2b",
}
CASE_FILES = ("equity_1h.csv.gz", "fills.csv", "weights.csv", "boundary_equity.csv", "metadata.json", "fingerprints.json", "member_targets.csv", "replay.json")
GROUP_FILES = ("summary.csv", "daily_equity.csv", "monthly_returns.csv", "completion.json")


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def need(condition, label):
    if not condition:
        raise ValueError(label)


def integer(value):
    return type(value) is int and value > 0


def validation_files():
    return {"validation_completion.json", "validate_completion.json", "parallel_validation_process.json",
        "parallel_validation_authorization.json", "parallel_validation_resource_preflight.json", "parallel_validation_runner.log",
        "resource_batch_process.json", "resource_batch.owner.json", "scheduling_audit.fit.json",
        "accelerated_fit_process.json", "accelerated_fit_completion.json", "accelerated_seed4_worker.json",
        "scheduling_annex.json", "training_completion.json",
        *(f"validation/{name}" for name in GROUP_FILES[:3]),
        *(f"validation/{family}/balanced/{name}" for family in ("raw", "cap075") for name in GROUP_FILES),
        *(f"validation/{family}/balanced/{cost}/{key}/{name}" for family in ("raw", "cap075")
          for cost in ("base", "stress") for key in ("0", "1", "2", "3", "4", "ensemble") for name in CASE_FILES)}


def load_extra(root, name):
    path = Path(root) / name
    need(name in PINS and sha(path) == PINS[name], "Pinned scheduling/receipt source changed: " + name)
    alias = "_droq_extra_" + path.stem
    if str(Path(root).resolve()) not in sys.path:
        sys.path.insert(0, str(Path(root).resolve()))
    if alias not in sys.modules:
        spec = importlib.util.spec_from_file_location(alias, path)
        module = importlib.util.module_from_spec(spec); sys.modules[alias] = module
        spec.loader.exec_module(module)
    need(Path(sys.modules[alias].__file__).resolve() == path.resolve(), "Foreign extension import")
    return sys.modules[alias]


class TrainingTimeBounds:
    """Narrow read view: never returned from verify_phase as a real terminal."""
    def __init__(self, inputs, envelope):
        self.inputs, self.envelope = inputs, envelope
        need(envelope.get("kind") == "derived_training_time_bounds_only"
             and envelope.get("original_terminal_relabelled") is False
             and envelope.get("original_terminal_status") == "failed"
             and envelope.get("accelerated_terminal_status") == "completed"
             and envelope.get("model_count") == 5, "Explicit independently verified training time envelope required")

    def __getattr__(self, name):
        return getattr(self.inputs, name)

    def json(self, root, name):
        actual = self.inputs.json(root, name)
        if root == "study" and name == "fit_process.json":
            need(actual.get("status") == "failed", "Original failed fit must remain failed")
            return {key: self.envelope[key] for key in ("kind", "started_at", "completed_at")}
        return actual


def verify_schedule(inputs, bf, dataset_path, source_root, extra_root):
    helper = load_extra(extra_root, "btceth_rl_droq_scheduling_audit.py")
    actual = helper.verify_accelerated_fit(inputs.roots["study"], inputs.roots["main"], dataset_path, source_root,
        scheduler_source=Path(extra_root) / "btceth_rl_droq_scheduling.py", performance_root=extra_root)
    stored = inputs.json("study", "scheduling_audit.fit.json")
    need(actual.get("passed") is True and actual.get("status") == "passed" and actual.get("training_model_count") == 5,
         "Fresh independent five-model scheduling audit must pass")
    need(stored.get("passed") is True and stored.get("status") == "passed"
         and stored.get("derived_cohort_envelope") == actual.get("derived_cohort_envelope"),
         "Stored and freshly derived cohort evidence must agree")
    for root, key in (("study", "input_sha256"), ("main", "main_input_sha256"), ("source", "source_sha256")):
        inputs.pin(root, actual[key], actual[key], "Fresh verified scheduling dependency graph:" + root)
    return actual["derived_cohort_envelope"]


def verify_parallel_validation(inputs, bf, spec, envelope):
    p = inputs.json("study", "parallel_validation_process.json")
    complete = inputs.json("study", "validation_completion.json")
    short = inputs.json("study", "validate_completion.json")
    auth = inputs.json("study", "parallel_validation_authorization.json")
    batch = inputs.json("study", "resource_batch_process.json")
    owner = inputs.json("study", "resource_batch.owner.json")
    resource = inputs.json("study", "parallel_validation_resource_preflight.json")
    protocol_sha = sha(inputs.roots["study"] / "protocol.json")
    need(p.get("scope") == "droq_parallel_fixed_case_evaluation" and p.get("split") == "validation"
         and p.get("status") == "completed" and p.get("completed_cases") == p.get("total_cases") == 24
         and p.get("source_sha256") == PINS["btceth_rl_droq_parallel_eval.py"]
         and p.get("original_runner_sha256") == spec["source_sha256"]["scripts/btceth_rl_droq.py"]
         and p.get("protocol_sha256") == protocol_sha and p.get("family_gpu_assignments") == spec["family_gpu_assignments"]
         and type(p.get("max_workers_per_family")) is int and 1 <= p["max_workers_per_family"] <= 12,
         "Genuine fixed24 parallel coordinator completion required")
    need(integer(p.get("coordinator_pid")) and p.get("parent_pid") == batch.get("pid")
         and integer(batch.get("pid")) and p["coordinator_pid"] == batch.get("child_pid")
         and p["coordinator_pid"] != batch["pid"], "Actual distinct finite batch and validation coordinator PIDs required")
    need(batch.get("status") == "completed_validation_awaiting_financial_audit"
         and type(batch.get("exit_code")) is int and batch["exit_code"] == 0
         and type(batch.get("child_exit_code")) is int and batch["child_exit_code"] == 0
         and batch.get("source_sha256") == PINS["wait_then_parallel_validation.py"]
         and batch.get("dependencies") == {k: PINS[k] for k in ("btceth_rl_droq_scheduling.py", "btceth_rl_droq_scheduling_audit.py", "btceth_rl_droq_parallel_eval.py")}
         and batch.get("annual_or_selection_authorized") is False and batch.get("production_change") is False
         and batch.get("workers_per_gpu") == 12 and batch.get("validation_cases") == batch.get("validation_repeat_cases") == 24,
         "Actual source-bound batch/child terminal0 and unchanged research scope required")
    need(owner.get("pid") == batch["pid"] and owner.get("source_sha256") == batch["source_sha256"], "Finite batch ownership source/PID changed")
    command = batch.get("command", [])
    need(len(command) == 5 and Path(command[3]).name == "wait_then_parallel_validation.py"
         and command[4] == "--run-validation", "Actual finite batch child invocation required")
    need(batch.get("log_sha256") == sha(inputs.capture("study", "parallel_validation_runner.log"))
         and batch.get("parallel_process_sha256") == sha(inputs.roots["study"] / "parallel_validation_process.json")
         and batch.get("validation_completion_sha256") == p.get("completion_sha256") == sha(inputs.roots["study"] / "validation_completion.json"),
         "Batch, actual parallel process, completion and original log hashes disagree")
    need(auth.get("parallel_case_evaluation_authorized") is True and auth.get("no_live_trading") is True
         and auth.get("financial_parameters_fixed") is True and auth.get("split") == "validation" and auth.get("family") is None
         and auth.get("protocol_sha256") == protocol_sha and auth.get("source_sha256") == p["source_sha256"]
         and auth.get("original_runner_sha256") == p["original_runner_sha256"]
         and sha(inputs.roots["study"] / "parallel_validation_authorization.json") == p.get("authorization_sha256"),
         "Real source/phase-bound prior scheduling authorization required")
    prerequisites = {"scheduling_audit.fit.json", "accelerated_fit_process.json", "accelerated_fit_completion.json",
        "accelerated_seed4_worker.json", "scheduling_annex.json", "training_completion.json", "parallel_validation_resource_preflight.json"}
    inputs.pin("study", auth.get("prerequisite_evidence_sha256"), prerequisites, "All fixed prior scheduling authorization dependencies")
    bf.verify_gpu_snapshot(resource.get("physical_gpus", []), spec["physical_gpus"], inputs.checks, "Parallel validation preflight")
    need(resource.get("command") == ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"], "Original unfiltered GPU process command required")
    rows = [{"gpu_uuid": line.split(",")[0].strip(), "pid": int(line.split(",")[1])}
            for line in resource.get("unfiltered_stdout", "").splitlines() if line.strip()]
    need(rows == resource.get("rows") and (not rows or rows == spec.get("allowed_gpu_rows")), "No unreviewed GPU process may be filtered out")
    if rows:
        allowed = spec["allowed_compute_instance"]
        found = resource["instances"][str(allowed["pid"])]
        need(all(found.get(k) == v for k, v in allowed.items()), "Reviewed UI process instance identity changed")
    for value in (complete, short):
        need(value.get("case_count") == value.get("repeat_count") == 24 and value.get("execution_mode") == "parallel_cases"
             and value.get("protocol_sha256") == protocol_sha and value.get("parallel_source_sha256") == p["source_sha256"],
             "Both fixed candidates and24 actual repeats are required")
    need(complete.get("coordinator_pid") == p["coordinator_pid"], "Completion belongs to a different coordinator")
    times = [envelope["completed_at"], resource["at"], auth["authorized_at"], batch["validation_started_at"],
             p["started_at"], complete["at"], p["completed_at"], batch["completed_at"]]
    need(all(bf.utc(a) <= bf.utc(b) for a, b in zip(times, times[1:])), "Completed fit, resources, authorization and genuine validation chronology changed")
    return p, complete


def merge_checks(checks, receipt):
    need(type(receipt.get("checks_total")) is int and receipt["checks_total"] > 0
         and type(receipt.get("failure_count")) is int and receipt["failure_count"] >= 0,
         "Genuine independent case check counters required")
    checks.count += receipt["checks_total"]
    checks.failed += receipt["failure_count"]
    checks.failures.extend(receipt.get("failures", [])[:max(0, 250 - len(checks.failures))])
    for key, value in receipt.get("maximum_errors", {}).items():
        checks.maximum_errors[key] = max(checks.maximum_errors.get(key, 0.), value)


def adopt_case_receipts(inputs, bf, receipt_root, dataset_path, phase="validation"):
    """Consume actual CPU reconstruction receipts, never trust a passed flag alone."""
    root = Path(receipt_root).resolve(); study = inputs.roots["study"]
    need(root.is_relative_to(study) and root != study, "Case audit receipts must be a separate owned study child")
    prefix = root.relative_to(study).as_posix()
    manifest = inputs.json("study", prefix + "/manifest.json")
    need(manifest.get("scope") == "parallel_bf33_financial_cases_only_not_full_study_audit"
         and manifest.get("phase") == phase and manifest.get("case_count") == (24 if phase == "validation" else 12)
         and manifest.get("source_sha256") == PINS["btceth_rl_droq_case_audit_v2.py"]
         and manifest.get("frozen_auditor_sha256") == spec_hash(inputs, "scripts/audit_btceth_rl_droq.py")
         and manifest.get("protocol_sha256") == sha(study / "protocol.json") and manifest.get("dataset_sha256") == sha(dataset_path),
         "Exact source/data-bound independent account receipt manifest required")
    expected, seen = {}, set()
    for item in manifest.get("records", []):
        run_id = item["run_id"]
        need(run_id not in seen and "/" not in run_id and "\\" not in run_id and ".." not in run_id, "Duplicate or unsafe independent case receipt")
        seen.add(run_id); folder = prefix + "/" + run_id
        need(sha(inputs.capture("study", folder + "/receipt.json")) == item["receipt_sha256"], "Independent case receipt bytes changed")
        receipt = inputs.json("study", folder + "/receipt.json")
        need(receipt.get("scope") == "original_bf33_cpu_financial_case_audit" and receipt.get("run_id") == run_id
             and receipt.get("split") == phase and integer(receipt.get("worker_pid"))
             and receipt.get("wrapper_source_sha256") == manifest["source_sha256"]
             and receipt.get("dataset_sha256") == manifest["dataset_sha256"]
             and receipt.get("source_sha256") == inputs.json("study", "protocol.json")["source_sha256"],
             "Case receipt original financial source/data identity changed")
        directory = receipt["directory_relative"]
        need(set(receipt.get("case_sha256", {})) == set(CASE_FILES), "Independent worker must bind all eight original case files")
        for name, digest in receipt["case_sha256"].items():
            need(sha(inputs.capture("study", directory + "/" + name)) == digest, "Audited financial input changed:" + directory + "/" + name)
        need(set(receipt.get("result_sha256", {})) == {"run.json", "history.csv", "monthly.json"}, "All reconstructed case outputs required")
        for name, digest in receipt["result_sha256"].items():
            need(sha(inputs.capture("study", folder + "/" + name)) == digest, "Independent reconstructed output changed")
        run = inputs.json("study", folder + "/run.json")
        need(run.get("run_id") == run_id and item.get("passed") == receipt.get("passed") == (receipt["failure_count"] == 0),
             "Independent reconstruction identity or true check result changed")
        merge_checks(inputs.checks, receipt)
        expected[run_id] = (run, inputs.csv("study", folder + "/history.csv"), inputs.json("study", folder + "/monthly.json"))
    need(len(expected) == manifest["case_count"] and manifest.get("passed") == all(item.get("passed") is True for item in manifest["records"]),
         "Every real reconstructed account is mandatory")
    return expected


def spec_hash(inputs, source):
    return inputs.json("study", "protocol.json")["source_sha256"][source]


def audit_parallel_group(inputs, bf, process, completion, expected, *, family, split="validation"):
    folder = f"{split}/{family}/balanced"
    summary, daily, monthly = [inputs.csv("study", folder + "/" + name) for name in GROUP_FILES[:3]]
    bf.validate_group_rows(summary, split=split, family=family, profile="balanced")
    group = inputs.json("study", folder + "/completion.json")
    need(group.get("case_count") == group.get("repeat_count") == 12 and group.get("execution_mode") == "parallel_cases"
         and group.get("worker_pid") is None and group.get("coordinator_pid") == process["coordinator_pid"]
         and group.get("parallel_source_sha256") == process["source_sha256"]
         and group.get("protocol_sha256") == process["protocol_sha256"], "Real parallel group completion required")
    wanted = set(summary.run_id)
    proofs = group.get("repeats", [])
    by_id = {p.get("run_id"): p for p in proofs}
    workers = group.get("case_workers", [])
    worker_map = {p.get("run_id"): p for p in workers}
    need(len(proofs) == len(by_id) == len(workers) == len(worker_map) == 12 and set(by_id) == set(worker_map) == wanted,
         "Exactly the twelve genuine repeat and worker identities are mandatory")
    original_uuid = process["family_gpu_assignments"][family]
    need(group.get("device", {}).get("visible_gpu_uuid") == original_uuid, "Original family GPU mapping changed")
    subset = {}
    for row in summary.to_dict("records"):
        run_id = row["run_id"]
        key = "ensemble" if row["primary_policy"] else str(int(row["seed"]))
        actual = inputs.json("study", f"{folder}/{row['scenario']}/{key}/replay.json")
        need(actual == by_id[run_id] and integer(actual.get("worker_pid"))
             and actual["worker_pid"] != process["coordinator_pid"]
             and actual.get("device") == group["device"]
             and worker_map[run_id] == {k: actual[k] for k in ("run_id", "worker_pid", "device")},
             "Genuine per-case PID/GPU/repeat receipts disagree")
        need(run_id in expected, "Financial reconstruction receipt is missing")
        run, history, months = expected[run_id]
        run.update(relative_directory=f"{folder}/{row['scenario']}/{key}", family=family, profile="balanced", scenario=row["scenario"])
        subset[run_id] = (run, history, months)
    need(group.get("worker_pids") == sorted({p["worker_pid"] for p in proofs}), "Actual worker PID population changed")
    need(bf.utc(process["started_at"]) <= bf.utc(group["at"]) <= bf.utc(completion["at"]), "Group chronology is outside actual parallel execution")
    bf.compare_group_tables(summary, daily, monthly, subset, inputs.checks, folder)
    return summary, daily, monthly, subset, group


def audit(output, main, dataset_path, *, source_root, extra_source_root, receipt_root, phase="validation"):
    need(phase == "validation", "Validation-first adapter; annual metadata requires its separate explicit execution annex")
    worker = load_extra(extra_source_root, "btceth_rl_droq_case_audit_v2.py")
    bf = worker.load_frozen(source_root)
    import pandas as pd
    checks = bf.Checks(); started = datetime.now(timezone.utc).isoformat()
    inputs = bf.Inputs(output, main, source_root, checks)
    records, runs, training, envelope = {}, {}, None, None
    try:
        need(not any((Path(output) / name).exists() for name in ("test", "test_started.json", "test_completion.json", "annual_authorization.json")), "Validation-only audit cannot inspect an annual attempt")
        spec, old = bf.verify_protocol(inputs, dataset_path)
        for name, expected in PINS.items():
            need(sha(Path(extra_source_root) / name) == expected, "Explicit scheduling extension source changed:" + name)
        envelope = verify_schedule(inputs, bf, dataset_path, source_root, extra_source_root)
        process, completion = verify_parallel_validation(inputs, bf, spec, envelope)
        need(checks.failed == 0, "Scheduling/protocol prerequisites failed before financial reconstruction")
        data = bf.load_market(dataset_path, bf.VALIDATION[1])
        features = bf.prior.independent_market_features(data, cutoff=bf.VALIDATION[1])
        dates = pd.date_range(bf.utc(bf.VALIDATION[0]), bf.utc(bf.VALIDATION[1]), freq="h", inclusive="left")
        import numpy as np
        for field in ("open", "high", "low", "close", "volume"):
            values = data["hourly_" + field].reindex(dates).to_numpy(dtype=float)
            checks.check(np.isfinite(values).all() and (values >= 0 if field == "volume" else values > 0).all(), "Complete original two-coin hourly field:" + field)
        records, training = bf.audit_training(TrainingTimeBounds(inputs, envelope), spec, data, features)
        expected = adopt_case_receipts(inputs, bf, receipt_root, dataset_path)
        assembled, rank_groups, all_workers = [], {}, []
        for family in bf.FAMILIES:
            group = audit_parallel_group(inputs, bf, process, completion, expected, family=family)
            assembled.append(group); rank_groups[(family, "balanced")] = group[0]
            runs.update({key: value[0] for key, value in group[3].items()})
            all_workers.extend(group[4]["case_workers"])
        need(completion.get("case_workers") == all_workers, "Top-level and both family case-worker populations disagree")
        need(len(completion.get("workers", [])) == 2 and {v.get("family") for v in completion["workers"]} == set(bf.FAMILIES), "Both genuine parallel budget groups required")
        for handle, group in zip(completion["workers"], assembled):
            need(handle.get("profile") == "balanced" and handle.get("split") == "validation" and handle.get("execution_mode") == "parallel_cases"
                 and handle.get("coordinator_pid") == process["coordinator_pid"] and handle.get("worker_pid") is None
                 and all(handle.get(key) == group[4].get(key) for key in ("device", "worker_pids", "case_workers")), "Top-level group handle differs from actual case completion")
        bf.compare_tables(inputs, [inputs.csv("study", "validation/" + name) for name in GROUP_FILES[:3]],
            [pd.concat([group[i] for group in assembled], ignore_index=True) for i in range(3)], "All24 original financial parent tables")
        bf.rank_candidates(rank_groups)
        for name in validation_files():
            inputs.capture("study", name)
        inputs.unchanged()
    except Exception as error:
        checks.check(False, type(error).__name__ + ": " + str(error))
    passed = checks.failed == 0 and len(records) == 5 and len(runs) == 24
    extensions = {name: sha(Path(extra_source_root) / name) for name in PINS}
    extensions[Path(__file__).name] = sha(__file__)
    return dict(schema_version=1, started_at=started, audited_at=datetime.now(timezone.utc).isoformat(), phase="validation", passed=passed,
        status="partial_test_pending" if passed else "failed", validation_complete=passed, test_complete=False,
        training_model_count=len(records), run_count=len(runs), validation_run_count=len(runs), test_run_count=0, baseline_run_count=0,
        repeat_count=len(runs) if passed else 0, checks_total=checks.count, failure_count=checks.failed, failures=checks.failures,
        maximum_errors=checks.maximum_errors, input_sha256=inputs.hashes["study"], main_input_sha256=inputs.hashes["main"],
        source_sha256=inputs.hashes["source"], extension_source_sha256=extensions, dataset_sha256=sha(dataset_path),
        training_calendar=training, runs=runs, baseline_runs={}, declared_research_targets=bf.TARGETS, research_targets=None,
        derived_training_time_bounds=envelope, original_failed_fit_relabelled=False, scheduling_adapter_explicit=True,
        financial_functions_unchanged=True, frozen_financial_auditor_sha256=worker.FROZEN_AUDITOR_SHA,
        fresh_holdout_success=False, training_budget_multiple=1, recipe_is_pure_official_default=False,
        accounting_passage_is_not_goal_success=True, **{flag: True for flag in bf.FLAGS})


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("output-dir", "reference-main", "dataset", "source-root", "extra-source-root", "receipt-root", "report"):
        p.add_argument("--" + name, type=Path, required=True)
    args = p.parse_args()
    if args.report.exists():
        raise FileExistsError("Never overwrite an independent financial audit")
    result = audit(args.output_dir, args.reference_main, args.dataset, source_root=args.source_root,
        extra_source_root=args.extra_source_root, receipt_root=args.receipt_root)
    with args.report.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, indent=2, allow_nan=False); stream.write("\n")
    print(json.dumps({k: result[k] for k in ("passed", "status", "training_model_count", "validation_run_count", "checks_total", "failure_count", "failures")}))
    raise SystemExit(0 if result["passed"] else 1)

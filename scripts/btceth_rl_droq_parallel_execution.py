"""Explicit selection and two-GPU annual adapter for frozen DroQ accounts.

No legacy fit/validate terminal is fabricated. Selection reads validation only;
annual execution needs a later, source/selection/resource-annex-bound approval.
Device allocation is the only change to the frozen case execution recipe.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import importlib
import json
import math
import multiprocessing as mp
import os
from pathlib import Path, PurePosixPath
import sys
import time
import uuid

PARALLEL_SHA = "7616aa59606a24d14a2b69fb4a9774baa6096d71015a42d5f8f2da5b2effe38d"
ORIGINAL_SHA = "f1463c5e206aabd6a77e934f7972a55c84c15901b4a380e470c7dcbbcf08443a"
OLD_AUDIT_SHA = "bf33c879ff5eeaa13ee288c561fb5e6f18efb3fddc0da5179e6d646339b8c2bc"
VALIDATION_AUDIT = "independent_parallel_audit.validation.json"
FINAL_AUDIT = "independent_parallel_audit.json"
MODE = "parallel_cases_explicit_two_gpu_annex"
FLAGS = ("designed_after_first_annual", "test_year_already_seen", "not_fresh_holdout")


def require(value, message):
    if not value:
        raise ValueError(message)


def regular(path):
    path = Path(path)
    for part in (path, *path.parents):
        require(not part.is_symlink() and not (hasattr(part, "is_junction") and part.is_junction()), "No links/junctions in execution evidence")
    return path


def relative(root, name):
    text = str(name).replace("\\", "/"); key = PurePosixPath(text)
    require(bool(text) and not key.is_absolute() and ".." not in key.parts and ":" not in text, "Safe relative execution evidence required")
    root = regular(root).resolve(); path = regular(root.joinpath(*key.parts)); path.resolve().relative_to(root)
    return path


def sha(path):
    with regular(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(regular(path).read_text(encoding="utf-8"))


def now():
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def write_new(path, value):
    with regular(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False); stream.write("\n")


def publish(path, value):
    path = regular(path); temp = path.with_suffix(path.suffix + ".pending")
    write_new(temp, value); os.replace(temp, path)


def verify_pins(root, values):
    require(isinstance(values, dict) and values, "Nonempty immutable source/input map required")
    for name, digest in values.items():
        require(sha(relative(root, name)) == digest, "Changed evidence: " + name)


def load_module(path, digest):
    path = regular(path).resolve(); require(sha(path) == digest, "Reviewed execution dependency SHA required")
    if str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))
    module = importlib.import_module(path.stem)
    require(Path(module.__file__).resolve() == path and sha(module.__file__) == digest, "Actually imported execution dependency differs")
    return module


def dependencies(source_root, extension_root, audit_source_sha):
    parallel = load_module(Path(extension_root) / "btceth_rl_droq_parallel_eval.py", PARALLEL_SHA)
    runner = parallel.get_runner(source_root)
    require(sha(runner.__file__) == ORIGINAL_SHA, "Frozen original financial producer required")
    old = importlib.import_module("scripts.audit_btceth_rl_droq")
    require(Path(old.__file__).resolve() == (Path(source_root) / "scripts/audit_btceth_rl_droq.py").resolve() and sha(old.__file__) == OLD_AUDIT_SHA, "Original independent selection formula required")
    independent = load_module(Path(extension_root) / "btceth_rl_droq_parallel_audit.py", audit_source_sha)
    return runner, parallel, old, independent


def extension_sources(extension_root, audit_source_sha):
    values = {"btceth_rl_droq_parallel_execution.py": sha(__file__), "btceth_rl_droq_parallel_eval.py": PARALLEL_SHA,
              "btceth_rl_droq_parallel_audit.py": audit_source_sha}
    for name, digest in values.items():
        path = Path(__file__) if name == "btceth_rl_droq_parallel_execution.py" else Path(extension_root) / name
        require(sha(path) == digest, "Current explicit adapter/dependency source changed")
    return values


def check_audit_maps(audit, output, main, source_root, extension_root):
    verify_pins(output, audit.get("input_sha256")); verify_pins(main, audit.get("main_input_sha256"))
    verify_pins(source_root, audit.get("source_sha256")); verify_pins(extension_root, audit.get("extension_source_sha256"))


def validation_gate(main, dataset, output, source_root, extension_root, audit_source_sha):
    """Explicit replacement gate; never calls the legacy terminal functions."""
    output = regular(output); runner, parallel, old, independent = dependencies(source_root, extension_root, audit_source_sha)
    spec = runner.protocol(main, dataset, output); audit = read(output / VALIDATION_AUDIT)
    require(audit.get("passed") is True and audit.get("status") == "partial_test_pending" and audit.get("test_complete") is False
            and audit.get("validation_complete") is True and audit.get("failures") == [], "Full independent parallel validation financial audit required")
    counts = dict(training_model_count=5, validation_run_count=24, run_count=24, repeat_count=24, failure_count=0)
    require(all(type(audit.get(k)) is int and audit[k] == v for k, v in counts.items()), "All five models and24 repeated validation accounts required")
    require(audit.get("source_sha256") == spec["source_sha256"], "Exactly the original nine financial source identities required")
    extensions = audit.get("extension_source_sha256", {})
    require(extensions.get("btceth_rl_droq_parallel_audit.py") == audit_source_sha and extensions.get("btceth_rl_droq_parallel_eval.py") == PARALLEL_SHA, "Actual parallel auditor and executor bound by the financial audit")
    check_audit_maps(audit, output, main, source_root, extension_root)
    training = runner.training_files(output); validation = independent.validation_files()
    require(isinstance(validation, set) and VALIDATION_AUDIT not in validation, "Explicit new validation evidence enumeration required")
    required = set(training) | validation | {"protocol.json", "protocol_receipt.json"}
    require(required <= set(audit["input_sha256"]), "Every financial/repeat/training/parallel prerequisite must be audited")
    plans = parallel.plan_cases(); ids = {f"validation__{p['scenario']}__{p['family']}__droq__balanced__{p['key']}" for p in plans}
    runs = audit.get("runs", {})
    require(set(runs) == ids and all(v.get("reconstructed") is True and v.get("hours") == 7080 for v in runs.values()), "24 independently reconstructed295-day accounts required")
    process = read(output / "parallel_validation_process.json")
    require(process.get("status") == "completed" and process.get("completed_cases") == 24 and process.get("source_sha256") == PARALLEL_SHA
            and process.get("protocol_sha256") == sha(output / "protocol.json") and runner.utc(process["completed_at"]) <= runner.utc(audit["audited_at"]), "True parallel validation completion precedes its independent audit")
    return (runner, parallel, old, independent), spec, audit, training, {name: sha(relative(output, name)) for name in sorted(validation)}


def ranked_candidates(runner, old, output):
    import pandas as pd
    groups = {(family, "balanced"): pd.read_csv(Path(output) / "validation" / family / "balanced/summary.csv", float_precision="round_trip") for family in ("raw", "cap075")}
    actual = [runner.candidate(groups[(family, "balanced")], family) for family in ("raw", "cap075")]
    actual.sort(key=lambda row: (-row["score"], 0 if row["family"] == "raw" else 1, row["profile"]))
    expected = old.rank_candidates(groups)
    require(len(expected) == 2 and [r["family"] for r in expected] == [r["family"] for r in actual], "Original independent two-candidate ranking and raw-first ties required")
    for truth, saved in zip(expected, actual):
        require(all(truth[k] == saved[k] for k in ("algorithm", "family", "profile", "rebalance_band", "exposure_multiplier")) and math.isclose(truth["score"], saved["score"], rel_tol=1e-12, abs_tol=1e-12), "Unchanged original score formula required")
        require(all(truth[cost] == saved[cost] for cost in ("base", "stress")), "Original independent ensemble metrics agree")
    return actual


def select(main, dataset, output, source_root, extension_root, audit_source_sha):
    deps, spec, audit, training, validation = validation_gate(main, dataset, output, source_root, extension_root, audit_source_sha)
    runner, _, old, _ = deps; output = Path(output); runner.no_test(output)
    require(not any((output / name).exists() for name in ("selection.json", "selection_receipt.json")), "Never replace a sealed choice")
    candidates = ranked_candidates(runner, old, output)
    value = dict(selected_at=runner.now(), protocol_sha256=sha(output / "protocol.json"), audit_file=VALIDATION_AUDIT, audit_sha256=sha(output / VALIDATION_AUDIT),
        candidates=candidates, candidate=candidates[0], seed_selection=False, selection_uses_test=False, test_results_available_to_selector=False,
        training_source_sha256=training, validation_source_sha256=validation, extension_source_sha256=extension_sources(extension_root, audit_source_sha),
        execution_mode="explicit_parallel_adapter", **{flag: True for flag in FLAGS})
    require(runner.utc(audit["audited_at"]) <= runner.utc(value["selected_at"]), "Independent financial audit precedes selection")
    write_new(output / "selection.json", value)
    write_new(output / "selection_receipt.json", dict(sealed_at=runner.now(), selection_file="selection.json", selection_sha256=sha(output / "selection.json"), protocol_sha256=sha(output / "protocol.json"), execution_mode="explicit_parallel_adapter"))
    return value


def frozen_selection(main, dataset, output, source_root, extension_root, audit_source_sha):
    deps, spec, audit, training, validation = validation_gate(main, dataset, output, source_root, extension_root, audit_source_sha)
    runner, _, old, _ = deps; output = Path(output); selected, receipt = read(output / "selection.json"), read(output / "selection_receipt.json")
    require(selected.get("audit_file") == VALIDATION_AUDIT and selected.get("audit_sha256") == sha(output / VALIDATION_AUDIT)
            and selected.get("protocol_sha256") == receipt.get("protocol_sha256") == sha(output / "protocol.json")
            and selected.get("seed_selection") is False and selected.get("selection_uses_test") is False and selected.get("test_results_available_to_selector") is False
            and all(selected.get(flag) is True for flag in FLAGS) and selected.get("execution_mode") == "explicit_parallel_adapter", "Original pre-annual frozen parallel selection identity required")
    require(receipt.get("selection_file") == "selection.json" and receipt.get("selection_sha256") == sha(output / "selection.json")
            and runner.utc(audit["audited_at"]) <= runner.utc(selected["selected_at"]) <= runner.utc(receipt["sealed_at"]), "Whole-file seal and audit/selection chronology required")
    require(selected.get("training_source_sha256") == training and selected.get("validation_source_sha256") == validation
            and selected.get("extension_source_sha256") == extension_sources(extension_root, audit_source_sha), "No changes after validation selection")
    actual = ranked_candidates(runner, old, output)
    require(selected.get("candidates") == actual and selected.get("candidate") == actual[0], "No annual-based replacement of either fixed validation candidate")
    return deps, spec, selected, receipt


def task_assignments(parallel, spec, family):
    plans = parallel.plan_cases("test", family); gpus = sorted(spec["physical_gpus"], key=lambda row: row["index"])
    require(len(gpus) == 2 and all("V100" in gpu["name"] for gpu in gpus), "Two original physical V100s required")
    return {f"test__{p['scenario']}__{family}__droq__balanced__{p['key']}": gpus[0 if p["key"] in ("0", "2", "4") else 1]["uuid"] for p in plans}


def prepare_annual(main, dataset, output, source_root, extension_root, audit_source_sha):
    deps, spec, selected, receipt = frozen_selection(main, dataset, output, source_root, extension_root, audit_source_sha)
    runner, parallel, _, _ = deps; output = Path(output); runner.no_test(output)
    require(not any((output / name).exists() for name in ("parallel_annual_resource_annex.json", "parallel_annual_resource_receipt.json", "annual_authorization.json")), "Preserve every annual preparation or authorization")
    assignments = task_assignments(parallel, spec, selected["candidate"]["family"])
    value = dict(prepared_at=runner.now(), protocol_sha256=sha(output / "protocol.json"), selection_sha256=sha(output / "selection.json"), selection_receipt_sha256=sha(output / "selection_receipt.json"),
        validation_audit_sha256=sha(output / VALIDATION_AUDIT), family=selected["candidate"]["family"], candidate=selected["candidate"],
        case_gpu_assignments=assignments, physical_gpus=spec["physical_gpus"], original_family_gpu_assignments=spec["family_gpu_assignments"],
        execution_mode=MODE, extension_source_sha256=extension_sources(extension_root, audit_source_sha), per_worker_cpu_threads=1, max_workers_per_gpu=6,
        case_count=12, repeat_count=12, test_window=list(runner.TEST), financial_parameters_changed=False, no_live_trading=True,
        annual_execution_authorized=False, **{flag: True for flag in FLAGS})
    require(runner.utc(receipt["sealed_at"]) <= runner.utc(value["prepared_at"]), "Annual resource annex follows the validation seal")
    write_new(output / "parallel_annual_resource_annex.json", value)
    write_new(output / "parallel_annual_resource_receipt.json", dict(protocol_file="parallel_annual_resource_annex.json", protocol_sha256=sha(output / "parallel_annual_resource_annex.json"), sealed_at=runner.now()))
    return value


def annual_gate(main, dataset, output, source_root, extension_root, audit_source_sha, *, require_new):
    deps, spec, selection, seal = frozen_selection(main, dataset, output, source_root, extension_root, audit_source_sha)
    runner, parallel, _, _ = deps; output = Path(output)
    annex, receipt, auth = (read(output / name) for name in ("parallel_annual_resource_annex.json", "parallel_annual_resource_receipt.json", "annual_authorization.json"))
    expected = task_assignments(parallel, spec, selection["candidate"]["family"])
    require(annex.get("protocol_sha256") == sha(output / "protocol.json") and annex.get("selection_sha256") == sha(output / "selection.json")
            and annex.get("selection_receipt_sha256") == sha(output / "selection_receipt.json") and annex.get("validation_audit_sha256") == sha(output / VALIDATION_AUDIT)
            and annex.get("candidate") == selection["candidate"] and annex.get("family") == selection["candidate"]["family"]
            and annex.get("case_gpu_assignments") == expected and annex.get("original_family_gpu_assignments") == spec["family_gpu_assignments"]
            and annex.get("physical_gpus") == spec["physical_gpus"] and annex.get("test_window") == list(runner.TEST)
            and annex.get("per_worker_cpu_threads") == 1 and annex.get("max_workers_per_gpu") == 6 and annex.get("case_count") == annex.get("repeat_count") == 12
            and annex.get("financial_parameters_changed") is False and annex.get("annual_execution_authorized") is False
            and annex.get("execution_mode") == MODE and annex.get("extension_source_sha256") == extension_sources(extension_root, audit_source_sha)
            and all(annex.get(flag) is True for flag in FLAGS), "Exact predeclared two-GPU12-account resource annex required")
    require(receipt.get("protocol_file") == "parallel_annual_resource_annex.json" and receipt.get("protocol_sha256") == sha(output / "parallel_annual_resource_annex.json"), "Whole-file resource-annex seal required")
    require(auth.get("annual_evaluation_authorized") is True and auth.get("parallel_case_evaluation_authorized") is True and auth.get("no_live_trading") is True
            and auth.get("financial_parameters_fixed") is True and auth.get("selection_sha256") == sha(output / "selection.json")
            and auth.get("resource_annex_sha256") == sha(output / "parallel_annual_resource_annex.json") and auth.get("resource_receipt_sha256") == sha(output / "parallel_annual_resource_receipt.json")
            and auth.get("execution_source_sha256") == sha(__file__) and auth.get("parallel_source_sha256") == PARALLEL_SHA
            and auth.get("validation_audit_sha256") == sha(output / VALIDATION_AUDIT) and auth.get("protocol_sha256") == sha(output / "protocol.json")
            and all(auth.get(flag) is True for flag in FLAGS), "Separate Root annual approval binds selected policy, two-GPU annex and exact execution sources")
    require(runner.utc(seal["sealed_at"]) <= runner.utc(annex["prepared_at"]) <= runner.utc(receipt["sealed_at"]) <= runner.utc(auth["authorized_at"]), "Validation seal then resource seal then separate annual authorization")
    require(runner.utc(auth["authorized_at"]) <= runner.utc(runner.now()), "A future-dated annual approval is not effective yet")
    if require_new:
        require(not any((output / name).exists() for name in ("test", "test_started.json", "test_completion.json", "test_process.json", "parallel_test_process.json", "parallel_test.owner.json", "parallel_test_runner.log")), "Never restart or overwrite an annual attempt; the separately checked authorization is required")
    else:
        marker = read(output / "test_started.json")
        require(marker.get("protocol_sha256") == sha(output / "protocol.json") and marker.get("selection_sha256") == sha(output / "selection.json")
                and marker.get("authorization_sha256") == sha(output / "annual_authorization.json") and marker.get("resource_annex_sha256") == sha(output / "parallel_annual_resource_annex.json")
                and marker.get("execution_source_sha256") == sha(__file__) and runner.utc(auth["authorized_at"]) <= runner.utc(marker["at"]), "True annual marker follows its separate approval")
    return deps, spec, selection, annex, auth


def annual_case_worker(task):
    """7616 case body, with only explicit task-device gate and resource proof."""
    parallel = load_module(Path(task["extension_root"]) / "btceth_rl_droq_parallel_eval.py", PARALLEL_SHA)
    runner = parallel.get_runner(task["source_root"])
    main, dataset, output = map(Path, (task["main"], task["dataset"], task["output"]))
    require(task.get("execution_source_sha256") == sha(__file__) and task.get("protocol_sha256") == sha(output / "protocol.json"), "Actual case uses source-bound declared task")
    require(task.get("resource_annex_sha256") == sha(output / "parallel_annual_resource_annex.json") and task.get("authorization_sha256") == sha(output / "annual_authorization.json") and task.get("test_marker_sha256") == sha(output / "test_started.json"), "Case cannot precede the immutable annual approval/marker")
    spec = runner.protocol(main, dataset, output); annex = read(output / "parallel_annual_resource_annex.json")
    family, scenario, key = (task[k] for k in ("family", "scenario", "key"))
    seed, seeds, run_id = parallel.identity(runner, "test", family, scenario, key)
    assigned = annex["case_gpu_assignments"].get(run_id)
    require(task.get("gpu_uuid") == assigned and assigned in {v["uuid"] for v in spec["physical_gpus"]}, "Only the exact per-account declared original card")
    device = runner.shared.torch_setup(0)
    require(device["visible_gpu_uuid"] == assigned and device["capability"] == [7, 0], "Actual case card agrees with explicit two-GPU annex")
    start, end = runner.TEST; data = runner.cropped_data(runner.shared.load_dataset(dataset), end)
    models, records = [], []
    for member in seeds:
        directory = output / "training" / runner.job_id(member); record = runner.read_json(directory / "record.json")
        require(record.get("seed") == member, "Actual saved training seed changed")
        model = runner.load_policy(directory, upstream=output / "upstream", device="cuda:0")
        require(runner.parameter_digest(model) == record["parameter_sha256"], "Actual actor/checkpoint digest required")
        models.append(model); records.append(record)
    case = output / "test" / family / runner.PROFILE / scenario / key; case.mkdir(parents=True, exist_ok=False)
    rng_before = runner.rng_digest()
    result, trace = runner.evaluate(data, models, seeds, family, start, end, runner.COSTS[scenario]); rng_first = runner.rng_digest()
    repeat, repeated_trace = runner.evaluate(data, models, seeds, family, start, end, runner.COSTS[scenario]); rng_after = runner.rng_digest()
    fingerprints, repeated = runner.result_fingerprints(result), runner.result_fingerprints(repeat)
    trace_bytes, repeated_bytes = runner.canonical_csv(trace), runner.canonical_csv(repeated_trace)
    require(fingerprints == repeated and trace_bytes == repeated_bytes and rng_before == rng_first == rng_after, "Exact accounts/targets/parameters/all inference RNG must repeat")
    runner.save_result(case, result); (case / "member_targets.csv").write_bytes(trace_bytes)
    proof = dict(run_id=run_id, passed=True, repeat_exact=True, fingerprints=fingerprints, repeated_fingerprints=repeated,
        member_targets_sha256=hashlib.sha256(trace_bytes).hexdigest(), repeated_member_targets_sha256=hashlib.sha256(repeated_bytes).hexdigest(),
        parameter_sha256_before=result["metadata"]["parameter_sha256_before"], parameter_sha256_after=result["metadata"]["parameter_sha256_after"],
        rng_sha256_before=rng_before, rng_sha256_after_first=rng_first, rng_sha256_after_repeat=rng_after, model_records=records,
        device=device, worker_pid=os.getpid(), execution_mode=MODE, execution_source_sha256=sha(__file__), resource_annex_sha256=task["resource_annex_sha256"], assigned_gpu_uuid=assigned)
    runner.write_new(case / "replay.json", proof)
    row, history, months = runner.summarize(result, run_id=run_id, strategy="droq_" + family, seed=seed, scenario=scenario, split="test", start=start, end=end, capital=10000., **runner.COSTS[scenario])
    row.update(algorithm="droq", profile=runner.PROFILE, family=family, primary_policy=seed is None, exposure_multiplier=runner.BUDGETS[family], rebalance_band=.025)
    history["profile"], months["profile"] = runner.PROFILE, runner.PROFILE
    return dict(identity=dict(split="test", family=family, scenario=scenario, key=key), row=row, history=history, months=months, proof=proof,
        case_relative_path=case.relative_to(output).as_posix(), case_files_sha256=runner.pins(output, [f"{case.relative_to(output).as_posix()}/{name}" for name in runner.CASE_FILES]))


def merge_annual(runner, parallel, output, family, results, annex):
    import pandas as pd
    output = Path(output); plans = parallel.plan_cases("test", family); by_id = {parallel.case_key(v["identity"]): v for v in results}
    require(len(results) == len(by_id) == 12 and set(by_id) == {parallel.case_key(p) for p in plans}, "All twelve fixed annual accounts once")
    ordered = [by_id[parallel.case_key(p)] for p in plans]; proofs, devices = [], {}
    for result in ordered:
        _, _, run_id = parallel.identity(runner, **result["identity"]); case = relative(output, result["case_relative_path"])
        require(case.relative_to(output).as_posix() == f"test/{family}/balanced/{result['identity']['scenario']}/{result['identity']['key']}", "Actual annual account path unchanged")
        runner.verify_pins(output, result["case_files_sha256"]); proof = read(case / "replay.json")
        require(proof == result["proof"] and proof.get("run_id") == result["row"].get("run_id") == run_id and proof.get("passed") is True
                and proof.get("device", {}).get("visible_gpu_uuid") == annex["case_gpu_assignments"][run_id]
                and proof.get("execution_source_sha256") == sha(__file__), "Actual saved repeat/source/assigned card agree")
        require(all(frame.run_id.eq(run_id).all() for frame in (result["history"], result["months"])), "Actual annual history identity")
        gpu = proof["device"]["visible_gpu_uuid"]
        require(devices.setdefault(gpu, proof["device"]) == proof["device"], "Same per-card runtime evidence")
        proofs.append(proof)
    require(len(devices) == 2, "Both declared GPUs used without relabelling the legacy family card")
    directory = output / "test" / family / runner.PROFILE
    frames = (pd.DataFrame([v["row"] for v in ordered]), pd.concat([v["history"] for v in ordered], ignore_index=True), pd.concat([v["months"] for v in ordered], ignore_index=True))
    for name, frame in zip(runner.GROUP_FILES[:3], frames):
        with (directory / name).open("xb") as stream:
            stream.write(runner.canonical_csv(frame))
    workers = [dict(run_id=p["run_id"], worker_pid=p["worker_pid"], device=p["device"]) for p in proofs]
    group = dict(at=runner.now(), protocol_sha256=sha(output / "protocol.json"), case_count=12, repeat_count=12, repeats=proofs, device=None,
        devices=[devices[gpu] for gpu in sorted(devices)], execution_mode=MODE, worker_pid=None, coordinator_pid=os.getpid(), worker_pids=sorted({p["worker_pid"] for p in proofs}),
        case_workers=workers, case_gpu_assignments=annex["case_gpu_assignments"], execution_source_sha256=sha(__file__), resource_annex_sha256=sha(output / "parallel_annual_resource_annex.json"))
    write_new(directory / "completion.json", group)
    complete = dict(at=runner.now(), protocol_sha256=sha(output / "protocol.json"), selection_sha256=sha(output / "selection.json"), candidate=read(output / "selection.json")["candidate"],
        case_count=12, repeat_count=12, execution_mode=MODE, execution_source_sha256=sha(__file__), resource_annex_sha256=sha(output / "parallel_annual_resource_annex.json"),
        coordinator_pid=os.getpid(), workers=[{k: group[k] for k in ("device", "devices", "case_workers", "worker_pids")}], case_workers=workers, case_gpu_assignments=annex["case_gpu_assignments"])
    write_new(output / "test_completion.json", complete)
    return complete


def private_gate(output, runner):
    for _ in range(30):
        value = read(Path(output) / "parallel_test_process.json")
        if value.get("status") != "launching":
            break
        time.sleep(.1)
    owner = read(Path(output) / "parallel_test.owner.json")
    require(value.get("status") == "running" and value.get("child_pid") == os.getpid() and value.get("supervisor_pid") == os.getppid()
            and owner.get("pid") == os.getppid() and value.get("source_sha256") == sha(__file__), "Only actual independent annual supervisor/child may execute")


def test_internal(main, dataset, output, source_root, extension_root, audit_source_sha):
    deps, spec, selection, annex, _ = annual_gate(main, dataset, output, source_root, extension_root, audit_source_sha, require_new=False)
    runner, parallel, _, _ = deps; output = Path(output); private_gate(output, runner)
    scheduler = load_module(Path(extension_root) / "btceth_rl_droq_scheduling.py", "52ba244d55abf36fc49cdbaa8374c179c939dc35b07f3d27bbd8c7a07b233453")
    write_new(output / "parallel_test_resource_preflight.json", scheduler.resource_snapshot(spec, {}))
    family = selection["candidate"]["family"]; (output / "test" / family / "balanced").mkdir(parents=True, exist_ok=False)
    pools, results = {}, []
    try:
        for gpu in sorted(spec["physical_gpus"], key=lambda row: row["index"]):
            pools[gpu["uuid"]] = ProcessPoolExecutor(max_workers=6, mp_context=mp.get_context("spawn"), initializer=parallel.initialize_worker, initargs=(str(Path(source_root).resolve()), gpu["uuid"]))
        futures = []
        for plan in parallel.plan_cases("test", family):
            run_id = f"test__{plan['scenario']}__{family}__droq__balanced__{plan['key']}"; gpu = annex["case_gpu_assignments"][run_id]
            task = dict(plan, main=str(Path(main).resolve()), dataset=str(Path(dataset).resolve()), output=str(output.resolve()), source_root=str(Path(source_root).resolve()), extension_root=str(Path(extension_root).resolve()),
                gpu_uuid=gpu, execution_source_sha256=sha(__file__), protocol_sha256=sha(output / "protocol.json"), authorization_sha256=sha(output / "annual_authorization.json"),
                resource_annex_sha256=sha(output / "parallel_annual_resource_annex.json"), test_marker_sha256=sha(output / "test_started.json"))
            futures.append(pools[gpu].submit(annual_case_worker, task))
        for future in as_completed(futures):
            results.append(future.result())
        complete = merge_annual(runner, parallel, output, family, results, annex); runner.training_files(output)
    finally:
        for pool in pools.values():
            pool.shutdown(wait=True, cancel_futures=True)
    return complete


def supervise_test(main, dataset, output, source_root, extension_root, audit_source_sha):
    deps, spec, selection, annex, _ = annual_gate(main, dataset, output, source_root, extension_root, audit_source_sha, require_new=True)
    runner = deps[0]; output = Path(output); token = uuid.uuid4().hex
    write_new(output / "parallel_test.owner.json", dict(pid=os.getpid(), token=token, at=runner.now(), source_sha256=sha(__file__)))
    marker = dict(at=runner.now(), protocol_sha256=sha(output / "protocol.json"), selection_sha256=sha(output / "selection.json"), authorization_sha256=sha(output / "annual_authorization.json"),
        resource_annex_sha256=sha(output / "parallel_annual_resource_annex.json"), execution_source_sha256=sha(__file__), parallel_source_sha256=PARALLEL_SHA, supervisor_pid=os.getpid(), execution_mode=MODE)
    write_new(output / "test_started.json", marker)
    command = [sys.executable, "-B", "-u", str(Path(__file__).resolve()), "--phase", "_test", "--reference-main", str(Path(main).resolve()), "--dataset", str(Path(dataset).resolve()),
        "--output-dir", str(output.resolve()), "--source-root", str(Path(source_root).resolve()), "--extension-root", str(Path(extension_root).resolve()), "--audit-source-sha", audit_source_sha]
    process = dict(scope="explicit_droq_two_gpu_selected_annual", phase="test", status="launching", started_at=runner.now(), supervisor_pid=os.getpid(), command=command,
        source_sha256=sha(__file__), original_runner_sha256=ORIGINAL_SHA, parallel_source_sha256=PARALLEL_SHA, protocol_sha256=sha(output / "protocol.json"), selection_sha256=sha(output / "selection.json"),
        authorization_sha256=sha(output / "annual_authorization.json"), test_marker_sha256=sha(output / "test_started.json"), resource_annex_sha256=sha(output / "parallel_annual_resource_annex.json"),
        case_gpu_assignments=annex["case_gpu_assignments"], total_cases=12, max_workers_per_gpu=6, log_relative_path="parallel_test_runner.log")
    path = output / "parallel_test_process.json"; write_new(path, process)
    import subprocess
    try:
        with (output / "parallel_test_runner.log").open("x", encoding="utf-8") as stream:
            child = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
            process.update(status="running", child_pid=child.pid, coordinator_pid=child.pid); publish(path, process); code = child.wait()
        process.update(exit_code=code, completed_at=runner.now(), log_sha256=sha(output / "parallel_test_runner.log"))
        require(type(code) is int and code == 0, "Actual annual child failed; retain partial evidence without restart")
        process.update(status="completed", completion_sha256=sha(output / "test_completion.json")); publish(path, process)
    except BaseException as error:
        process.update(status="failed", completed_at=runner.now(), error_type=type(error).__name__, error=str(error), exit_code=process.get("exit_code", 1), log_sha256=sha(output / "parallel_test_runner.log") if (output / "parallel_test_runner.log").exists() else None)
        publish(path, process); raise
    return process


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("select", "prepare-annual", "test", "_test"), required=True)
    for name in ("reference-main", "dataset", "output-dir", "source-root", "extension-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--audit-source-sha", required=True); args = parser.parse_args()
    function = {"select": select, "prepare-annual": prepare_annual, "test": supervise_test, "_test": test_internal}[args.phase]
    value = function(args.reference_main, args.dataset, args.output_dir, args.source_root, args.extension_root, args.audit_source_sha)
    print(json.dumps({key: value[key] for key in ("candidate", "status", "case_count", "case_gpu_assignments") if key in value}, ensure_ascii=False))


if __name__ == "__main__":
    main()

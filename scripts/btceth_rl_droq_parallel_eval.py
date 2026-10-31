"""Parallelize fixed DroQ inference accounts without changing their mathematics.

Planning is the default. Execution needs a separate SHA-bound authorization,
five complete original training records, and the original frozen protocol.
Each case still runs twice through f146 evaluate/save_result/summarize. This
module never trains, attaches to a process, selects seeds or submits orders.
Parallel process evidence is explicit; it does not impersonate the original
one-process-per-family coordinator and requires its own scheduling audit.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import importlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
DRIVER_SHA = "f1463c5e206aabd6a77e934f7972a55c84c15901b4a380e470c7dcbbcf08443a"
SCOPE = "droq_parallel_fixed_case_evaluation"
_RUNNER = None


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def get_runner(source_root=None):
    global _RUNNER
    source_root = Path(source_root or ROOT).resolve()
    path = source_root / "scripts/btceth_rl_droq.py"
    if sha(path) != DRIVER_SHA:
        raise ValueError("The unchanged pinned f146 financial runner is required")
    if _RUNNER is None:
        if str(source_root) not in sys.path:
            sys.path.insert(0, str(source_root))
        _RUNNER = importlib.import_module("scripts.btceth_rl_droq")
    if Path(_RUNNER.__file__).resolve() != path:
        raise ValueError("Foreign original runner import; use a clean worker")
    return _RUNNER


def plan_cases(split="validation", family=None):
    """Original order: family, base/stress, seeds0..4, real ensemble."""
    if split not in ("validation", "test"):
        raise ValueError("Only the two fixed original evaluation splits exist")
    if split == "validation":
        if family is not None:
            raise ValueError("Validation must retain both original budget families")
        families = ("raw", "cap075")
    else:
        if family not in ("raw", "cap075"):
            raise ValueError("Annual replay needs its already selected family")
        families = (family,)
    return [dict(split=split, family=f, scenario=cost, key=key)
            for f in families for cost in ("base", "stress")
            for key in ("0", "1", "2", "3", "4", "ensemble")]


def identity(runner, split, family, scenario, key):
    if family not in runner.BUDGETS or scenario not in runner.COSTS:
        raise ValueError("Only fixed families and cost scenarios are allowed")
    candidates = plan_cases(split, family if split == "test" else None)
    value = dict(split=split, family=family, scenario=scenario, key=key)
    if value not in candidates:
        raise ValueError("No omitted, extra or selected-seed cases")
    seed = None if key == "ensemble" else int(key)
    seeds = list(runner.SEEDS) if seed is None else [seed]
    run_id = f"{split}__{scenario}__{family}__droq__{runner.PROFILE}__{key}"
    return seed, seeds, run_id


def case_key(plan):
    if set(plan) != {"split", "family", "scenario", "key"}:
        raise ValueError("Exactly one fixed account identity required")
    return tuple(plan[key] for key in ("split", "family", "scenario", "key"))


def initialize_worker(source_root, uuid):
    # Isolate a spawned process before importing any GPU-aware dependency.
    os.environ["CUDA_VISIBLE_DEVICES"] = uuid
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = "1"
    get_runner(source_root).shared.gpu_initializer(uuid)


def case_worker(task):
    """One genuine account plus its exact repetition; only required actors load."""
    runner = get_runner(task.get("source_root"))
    main, dataset, output = map(Path, (task["main"], task["dataset"], task["output"]))
    spec = runner.protocol(main, dataset, output)
    family, split, scenario, key = (task[k] for k in ("family", "split", "scenario", "key"))
    seed, seeds, run_id = identity(runner, split, family, scenario, key)
    if task.get("protocol_sha256") != runner.sha(output / "protocol.json"):
        raise ValueError("Dispatched protocol bytes changed")
    if task.get("source_sha256") != sha(__file__):
        raise ValueError("Dispatched parallel execution source changed")
    device = runner.shared.torch_setup(0)
    if (device["visible_gpu_uuid"] != spec["family_gpu_assignments"][family]
            or device["capability"] != [7, 0]):
        raise ValueError("Actual assigned original family V100 is required")
    start, end = runner.VALIDATION if split == "validation" else runner.TEST
    data = runner.cropped_data(runner.shared.load_dataset(dataset), end)
    models, records = [], []
    for member in seeds:
        directory = output / "training" / runner.job_id(member)
        record = runner.read_json(directory / "record.json")
        if record.get("seed") != member:
            raise ValueError("Actual saved training seed changed")
        model = runner.load_policy(directory, upstream=output / "upstream", device="cuda:0")
        if runner.parameter_digest(model) != record["parameter_sha256"]:
            raise ValueError("Actual loaded actor and saved five-network tensors differ")
        models.append(model); records.append(record)
    case = output / split / family / runner.PROFILE / scenario / key
    # Claim ownership before evaluation; a duplicate task must fail rather than
    # race through save_result (the shared writer permits existing folders).
    case.mkdir(parents=True, exist_ok=False)
    rng_before = runner.rng_digest()
    result, trace = runner.evaluate(data, models, seeds, family, start, end, runner.COSTS[scenario])
    rng_first = runner.rng_digest()
    repeat, repeated_trace = runner.evaluate(data, models, seeds, family, start, end, runner.COSTS[scenario])
    rng_after = runner.rng_digest()
    fingerprints, repeated = runner.result_fingerprints(result), runner.result_fingerprints(repeat)
    trace_bytes, repeated_bytes = runner.canonical_csv(trace), runner.canonical_csv(repeated_trace)
    if (fingerprints != repeated or trace_bytes != repeated_bytes
            or rng_before != rng_first or rng_before != rng_after):
        raise AssertionError("Exact accounts, targets, parameters and all inference RNG must repeat")
    # Only this task owns the case directory claimed above.
    runner.save_result(case, result)
    (case / "member_targets.csv").write_bytes(trace_bytes)
    proof = dict(run_id=run_id, passed=True, repeat_exact=True,
        fingerprints=fingerprints, repeated_fingerprints=repeated,
        member_targets_sha256=hashlib.sha256(trace_bytes).hexdigest(),
        repeated_member_targets_sha256=hashlib.sha256(repeated_bytes).hexdigest(),
        parameter_sha256_before=result["metadata"]["parameter_sha256_before"],
        parameter_sha256_after=result["metadata"]["parameter_sha256_after"],
        rng_sha256_before=rng_before, rng_sha256_after_first=rng_first,
        rng_sha256_after_repeat=rng_after, model_records=records,
        device=device, worker_pid=os.getpid())
    runner.write_new(case / "replay.json", proof)
    row, history, months = runner.summarize(result, run_id=run_id, strategy="droq_" + family,
        seed=seed, scenario=scenario, split=split, start=start, end=end,
        capital=10000., **runner.COSTS[scenario])
    row.update(algorithm="droq", profile=runner.PROFILE, family=family,
        primary_policy=seed is None, exposure_multiplier=runner.BUDGETS[family], rebalance_band=.025)
    history["profile"], months["profile"] = runner.PROFILE, runner.PROFILE
    print("DROQ_PARALLEL_CASE", run_id, row["net_return"], row["max_drawdown"], flush=True)
    return dict(identity=dict(split=split, family=family, scenario=scenario, key=key),
        row=row, history=history, months=months, proof=proof,
        case_relative_path=case.relative_to(output).as_posix(),
        case_files_sha256=runner.pins(output, [f"{case.relative_to(output).as_posix()}/{name}" for name in runner.CASE_FILES]))


def merge_groups(output, split, results, *, family=None, source_root=None):
    """Sort completed cases; never let completion order reorder financial CSVs."""
    runner = get_runner(source_root)
    import pandas as pd
    output = Path(output)
    plans = plan_cases(split, family)
    if len(results) != len(plans):
        raise ValueError("Every original cost/seed/ensemble account must complete")
    by_id = {}
    for result in results:
        key = case_key(result["identity"])
        if key in by_id:
            raise ValueError("Duplicate parallel result")
        by_id[key] = result
    expected = {case_key(plan) for plan in plans}
    if set(by_id) != expected:
        raise ValueError("Missing or extra cost/seed/budget account")
    groups = []
    for current in dict.fromkeys(plan["family"] for plan in plans):
        ordered = [by_id[case_key(plan)] for plan in plans if plan["family"] == current]
        directory = output / split / current / runner.PROFILE
        proofs, devices = [], []
        for result in ordered:
            plan = result["identity"]
            _, _, run_id = identity(runner, **plan)
            case = output / split / current / runner.PROFILE / plan["scenario"] / plan["key"]
            if result["case_relative_path"] != case.relative_to(output).as_posix():
                raise ValueError("Parallel result account path changed")
            runner.verify_pins(output, result["case_files_sha256"])
            proof = runner.read_json(case / "replay.json")
            if (proof != result["proof"] or proof.get("run_id") != run_id
                    or result["row"].get("run_id") != run_id or proof.get("passed") is not True):
                raise ValueError("Returned row and actual saved repeat proof disagree")
            if any(not frame.run_id.eq(run_id).all() for frame in (result["history"], result["months"])):
                raise ValueError("Returned account history identity changed")
            proofs.append(proof); devices.append(proof["device"])
        if not all(device == devices[0] for device in devices):
            raise ValueError("A family moved to a different GPU/runtime")
        frames = (pd.DataFrame([value["row"] for value in ordered]),
                  pd.concat([value["history"] for value in ordered], ignore_index=True),
                  pd.concat([value["months"] for value in ordered], ignore_index=True))
        for name, frame in zip(runner.GROUP_FILES[:3], frames):
            path = directory / name
            with path.open("xb") as stream:
                stream.write(runner.canonical_csv(frame))
        group = dict(at=runner.now(), protocol_sha256=runner.sha(output / "protocol.json"),
            case_count=12, repeat_count=12, repeats=proofs, device=devices[0],
            execution_mode="parallel_cases", worker_pid=None, coordinator_pid=os.getpid(),
            worker_pids=sorted({proof["worker_pid"] for proof in proofs}),
            case_workers=[dict(run_id=proof["run_id"], worker_pid=proof["worker_pid"],
                               device=proof["device"]) for proof in proofs],
            parallel_source_sha256=sha(__file__))
        runner.write_new(directory / "completion.json", group)
        groups.append(dict(profile=runner.PROFILE, split=split, family=current,
            execution_mode="parallel_cases", coordinator_pid=os.getpid(), worker_pid=None,
            worker_pids=group["worker_pids"], device=devices[0], case_workers=group["case_workers"]))
    if split == "validation":
        for name in runner.GROUP_FILES[:3]:
            frames = [pd.read_csv(output / split / current / runner.PROFILE / name,
                                  float_precision="round_trip") for current in ("raw", "cap075")]
            with (output / split / name).open("xb") as stream:
                stream.write(runner.canonical_csv(pd.concat(frames, ignore_index=True)))
    complete = dict(at=runner.now(), protocol_sha256=runner.sha(output / "protocol.json"),
        case_count=len(plans), repeat_count=len(plans), execution_mode="parallel_cases",
        parallel_source_sha256=sha(__file__), coordinator_pid=os.getpid(),
        workers=groups, case_workers=[entry for group in groups for entry in group["case_workers"]])
    if split == "test":
        complete["candidate"] = runner.read_json(output / "selection.json")["candidate"]
        complete["selection_sha256"] = runner.sha(output / "selection.json")
    runner.write_new(output / (split + "_completion.json"), complete)
    if split == "validation":
        runner.write_new(output / "validate_completion.json", dict(at=runner.now(),
            protocol_sha256=runner.sha(output / "protocol.json"), case_count=24, repeat_count=24,
            execution_mode="parallel_cases", parallel_source_sha256=sha(__file__)))
    return complete


def authorize(main, dataset, output, split, family, authorization, source_root=None):
    runner = get_runner(source_root)
    output = Path(output)
    spec = runner.protocol(main, dataset, output)
    auth = runner.read_json(authorization)
    fixed = dict(parallel_case_evaluation_authorized=True, no_live_trading=True,
        financial_parameters_fixed=True, split=split, family=family,
        source_sha256=sha(__file__), original_runner_sha256=DRIVER_SHA,
        protocol_sha256=runner.sha(output / "protocol.json"))
    if any(auth.get(key) != value for key, value in fixed.items()):
        raise ValueError("Separate exact source/phase/protocol-bound scheduling authorization required")
    evidence = auth.get("prerequisite_evidence_sha256")
    runner.verify_pins(output, evidence)
    runner.training_files(output)
    if split == "validation":
        runner.no_test(output)
        if (output / "selection.json").exists():
            raise ValueError("Never re-run validation after an original selected candidate")
    else:
        _, selection, receipt = runner.frozen_selection(main, dataset, output)
        if (selection["candidate"]["family"] != family
                or auth.get("selection_sha256") != runner.sha(output / "selection.json")
                or runner.utc(auth["authorized_at"]) < runner.utc(receipt["sealed_at"])):
            raise ValueError("Annual cases require the actual selected family and later source-bound authorization")
        runner.no_test(output)
    return spec, auth


def run_parallel(main, dataset, output, *, split="validation", family=None,
                 source_root=None, authorization=None, workers_per_gpu=12):
    if type(workers_per_gpu) is not int or not 1 <= workers_per_gpu <= 12:
        raise ValueError("Use1..12 genuine case workers per original family GPU")
    plans = plan_cases(split, family)
    if authorization is None:
        raise ValueError("Planning does not authorize data loading or evaluation")
    runner = get_runner(source_root)
    output = Path(output)
    spec, auth = authorize(main, dataset, output, split, family, authorization, source_root)
    for name in (split, split + "_completion.json", "parallel_" + split + "_process.json"):
        if runner.relative(output, name).exists():
            raise FileExistsError("Preserve all previous phase and partial account evidence: " + name)
    process_path = output / ("parallel_" + split + "_process.json")
    process = dict(scope=SCOPE, split=split, status="running", coordinator_pid=os.getpid(),
        parent_pid=os.getppid(), started_at=runner.now(), source_sha256=sha(__file__),
        original_runner_sha256=DRIVER_SHA, protocol_sha256=runner.sha(output / "protocol.json"),
        authorization_sha256=runner.sha(authorization), max_workers_per_family=workers_per_gpu,
        total_cases=len(plans), family_gpu_assignments=spec["family_gpu_assignments"])
    runner.write_new(process_path, process)
    families = list(dict.fromkeys(plan["family"] for plan in plans))
    results, pools = [], {}
    try:
        for current in families:
            (output / split / current / runner.PROFILE).mkdir(parents=True, exist_ok=False)
            pools[current] = ProcessPoolExecutor(max_workers=workers_per_gpu,
                mp_context=mp.get_context("spawn"), initializer=initialize_worker,
                initargs=(str(Path(source_root or ROOT).resolve()), spec["family_gpu_assignments"][current]))
        futures = [pools[plan["family"]].submit(case_worker, dict(plan,
            main=str(Path(main).resolve()), dataset=str(Path(dataset).resolve()), output=str(output.resolve()),
            source_root=str(Path(source_root or ROOT).resolve()), protocol_sha256=process["protocol_sha256"],
            source_sha256=process["source_sha256"])) for plan in plans]
        for future in as_completed(futures):
            results.append(future.result())
        complete = merge_groups(output, split, results, family=family, source_root=source_root)
        runner.training_files(output)
    except BaseException as exc:
        process.update(status="failed", completed_at=runner.now(),
            error_class=type(exc).__name__, completed_cases=len(results))
        runner.publish(process_path, process)
        raise
    finally:
        for pool in pools.values():
            pool.shutdown(wait=True, cancel_futures=True)
    process.update(status="completed", completed_at=runner.now(), completed_cases=len(results),
        completion_sha256=runner.sha(output / (split + "_completion.json")))
    runner.publish(process_path, process)
    return complete


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--family", choices=("raw", "cap075"))
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--workers-per-gpu", type=int, default=12)
    for name in ("reference-main", "dataset", "output-dir", "source-root", "authorization"):
        parser.add_argument("--" + name, type=Path)
    args = parser.parse_args()
    plans = plan_cases(args.split, args.family)
    if not args.execute:
        print(json.dumps(dict(scope=SCOPE, planning_only=True, no_data_or_model_loaded=True,
            case_count=len(plans), max_workers_per_family=args.workers_per_gpu,
            original_runner_sha256=DRIVER_SHA, source_sha256=sha(__file__), cases=plans), indent=2))
        return
    if any(getattr(args, name) is None for name in ("reference_main", "dataset", "output_dir", "authorization")):
        parser.error("--execute requires source-bound authorization, reference-main, dataset and output-dir")
    run_parallel(args.reference_main, args.dataset, args.output_dir, split=args.split,
        family=args.family, source_root=args.source_root, authorization=args.authorization,
        workers_per_gpu=args.workers_per_gpu)


if __name__ == "__main__":
    main()

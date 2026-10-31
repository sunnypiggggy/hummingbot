"""Frozen single-pair RL research: each policy owns 10,000 USDT and one coin.

Training and both evaluation phases are separate commands. Existing portfolio
models/results are never reused or overwritten. No exchange/private API calls.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.metadata
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_name] = "1"
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

SYMBOLS = ("BTCUSDT", "ETHUSDT")
ALGORITHMS = ("sac", "tqc", "crossq", "recurrent_ppo", "td3", "droq")
SEEDS = (0, 1, 2, 3, 4)
TRAIN = ("2020-01-01T00:00:00Z", "2025-01-01T00:00:00Z")
VALIDATION = (TRAIN[1], "2025-10-23T00:00:00Z")
TEST = (VALIDATION[1], "2026-10-23T00:00:00Z")
COSTS = {"base": {"fee": .001, "slippage": .0002}, "stress": {"fee": .002, "slippage": .0005}}
BASELINES = ("cash", "buy_hold", "sma200_trend", "momentum180")
SOURCE_FILES = ("scripts/backtest_single_pair_rl.py", "scripts/single_pair_rl_env.py", "scripts/single_pair_rl_models.py",
                "scripts/audit_single_pair_rl.py",
                "scripts/backtest_btceth_rl.py", "scripts/btceth_rl_env.py", "scripts/crypto_rl_portfolio_env.py",
                "scripts/crypto_rl_reporting.py", "scripts/crypto_rl_data.py", "scripts/btceth_rl_td3.py", "scripts/btceth_rl_droq.py")


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value, *, exclusive=False):
    from scripts.crypto_rl_reporting import clean_json
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(clean_json(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if exclusive:
        with path.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
    else:
        temporary = path.with_suffix(path.suffix + ".pending")
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, path)


def cropped_data(data, end):
    boundary = pd.Timestamp(end)
    # Build only from explicit causal market inputs, never cached portfolio features.
    return {"assets": data.get("assets", {}), **{"hourly_" + field: data["hourly_" + field].loc[lambda f: f.index < boundary].copy()
            for field in ("open", "high", "low", "close", "volume")}}


def prepare(output, dataset, algorithms):
    from scripts.backtest_btceth_rl import load_dataset
    from scripts.single_pair_rl_env import SinglePairEnv
    from scripts.single_pair_rl_models import recipe, runtime_versions
    algorithms = tuple(algorithms)
    if not algorithms or len(set(algorithms)) != len(algorithms) or not set(algorithms) <= set(ALGORITHMS):
        raise ValueError("A nonempty unique set of reviewed algorithms is required")
    output, dataset = Path(output), Path(dataset)
    if output.exists():
        raise FileExistsError("Preserve an existing research output")
    data = load_dataset(dataset)
    quality = {}
    for symbol in SYMBOLS:
        env = SinglePairEnv(cropped_data(data, TRAIN[1]), symbol, *TRAIN, training=True, episode_days=32, record_history=False)
        quality[symbol] = env.training_window_metadata
        for bounds in (VALIDATION, TEST):
            SinglePairEnv(cropped_data(data, bounds[1]), symbol, *bounds, record_history=False)
    spec = {"study": "independent single-pair RL", "frozen_at": now(), "symbols": list(SYMBOLS), "algorithms": list(algorithms),
            "seeds": list(SEEDS), "primary_seed": 0, "primary_seed_fixed_before_training": True,
            "capital_per_account": 10000., "capital_shared_between_pairs": False, "ensemble": False,
            "train": list(TRAIN), "validation": list(VALIDATION), "test": list(TEST), "costs": COSTS,
            "dataset_sha256": sha(dataset), "dataset_manifest_sha256": sha(dataset.parent / "manifest.json"),
            "sources": {name: sha(ROOT / name) for name in SOURCE_FILES}, "recipes": {a: recipe(a) for a in algorithms},
            "runtime_packages": {**runtime_versions(), "pandas": importlib.metadata.version("pandas")}, "python": sys.version,
            "environment": {"observation_dim": 13, "action_dim": 1, "action": "coin=.95*max(clip(a,-1,1),0); cash=1-coin",
                            "cash_floor": .05, "max_coin_target": .95, "rebalance_band": .025, "drawdown_penalty": .25,
                            "episode_days": 32, "execution": "00UTC sells;01UTC buys;all hourly closes marked",
                            "features": "selected coin10 causal daily market features + own coin/cash weights and drawdown"},
            "training_window_quality": quality, "validation_selection": {"seed_selection": False,
                "algorithm_score": "median across5seeds of mean over base/stress(log1p(return)-.5*abs(DD)-2*max(abs(DD)-.20,0))",
                "tie_break": "algorithm identifier ascending", "selected_algorithm_primary_account": "predeclared seed0 only"},
            "baselines": list(BASELINES), "test_year_already_observed": True, "fresh_holdout": False,
            "research_gates_per_pair": {"base_return_at_least": .05, "base_drawdown_magnitude_at_most": .20,
                "stress_return_at_least": 0., "positive_base_seeds_at_least": 3, "beats_same_coin_hold_return_and_drawdown": True},
            "timesteps_per_model": {a: 65536 if a == "td3" else 32768 for a in algorithms},
            "no_production_change": True, "training_model_count": len(algorithms) * 10}
    output.mkdir(parents=True)
    for name in SOURCE_FILES:
        target = output / "frozen_source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    write(output / "protocol.json", spec, exclusive=True)
    write(output / "protocol_receipt.json", {"protocol_sha256": sha(output / "protocol.json"), "frozen_at": now()}, exclusive=True)
    return spec


def check_protocol(output, dataset):
    from scripts.single_pair_rl_models import runtime_versions
    spec = read(Path(output) / "protocol.json")
    if sha(Path(output) / "protocol.json") != read(Path(output) / "protocol_receipt.json")["protocol_sha256"]:
        raise ValueError("Frozen protocol bytes changed")
    if spec["symbols"] != list(SYMBOLS) or spec["seeds"] != list(SEEDS) or spec["capital_per_account"] != 10000. or spec["ensemble"] is not False:
        raise ValueError("Independent-account protocol changed")
    if spec["train"] != list(TRAIN) or spec["validation"] != list(VALIDATION) or spec["test"] != list(TEST) or spec["costs"] != COSTS:
        raise ValueError("Frozen dates/costs changed")
    if (set(spec["sources"]) != set(SOURCE_FILES) or spec.get("primary_seed") != 0
            or spec.get("primary_seed_fixed_before_training") is not True or spec.get("capital_shared_between_pairs") is not False
            or spec.get("fresh_holdout") is not False or spec.get("test_year_already_observed") is not True
            or not spec["algorithms"] or len(set(spec["algorithms"])) != len(spec["algorithms"])
            or not set(spec["algorithms"]) <= set(ALGORITHMS) or spec["training_model_count"] != len(spec["algorithms"]) * 10):
        raise ValueError("Complete frozen sources and independent-account declarations required")
    required_env = {"observation_dim": 13, "action_dim": 1, "cash_floor": .05, "max_coin_target": .95,
                    "rebalance_band": .025, "drawdown_penalty": .25, "episode_days": 32}
    if any(spec["environment"].get(key) != value for key, value in required_env.items()):
        raise ValueError("Frozen single-pair environment contract changed")
    if sha(dataset) != spec["dataset_sha256"] or sha(Path(dataset).parent / "manifest.json") != spec["dataset_manifest_sha256"]:
        raise ValueError("Frozen market inputs changed")
    for name, expected in spec["sources"].items():
        if sha(ROOT / name) != expected:
            raise ValueError("Executed source differs from frozen source: " + name)
    if {**runtime_versions(), "pandas": importlib.metadata.version("pandas")} != spec["runtime_packages"]:
        raise ValueError("Frozen learner/runtime dependencies changed")
    return spec


def resources():
    result = subprocess.run(["nvidia-smi", "--query-gpu=uuid,memory.free,utilization.gpu", "--format=csv,noheader,nounits"], check=True, capture_output=True, text=True)
    gpus = [{"uuid": (v := line.split(","))[0].strip(), "free_mb": int(v[1]), "utilization": int(v[2])} for line in result.stdout.splitlines()]
    cpus = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count() or 1
    quota = Path("/sys/fs/cgroup/cpu.max")
    if quota.exists():
        limit, period = quota.read_text().split()
        if limit != "max":
            cpus = min(cpus, max(1, int(limit) // int(period)))
    return {"at": now(), "gpus": gpus, "available_cpus": cpus}


def initialize_worker(uuid):
    os.environ["CUDA_VISIBLE_DEVICES"] = uuid
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = "1"


def train_one(task):
    from scripts.backtest_btceth_rl import load_dataset
    from scripts.single_pair_rl_env import SinglePairEnv
    from scripts import single_pair_rl_models as models
    output, symbol, algorithm, seed = Path(task["output"]), task["symbol"], task["algorithm"], task["seed"]
    directory = output / "training" / symbol / algorithm / f"seed{seed}"
    directory.mkdir(parents=True, exist_ok=False)
    started = now()
    try:
        data = cropped_data(load_dataset(task["dataset"]), TRAIN[1])
        env = SinglePairEnv(data, symbol, *TRAIN, training=True, episode_days=32, record_history=False,
                            capital=10000., drawdown_penalty=.25, rebalance_band=.025)
        model = models.make_model(algorithm, env, seed=seed, device="cuda:0", upstream_root=task.get("upstream"))
        training = models.train_model(model, env, steps=models.STEPS[algorithm], seed=seed, progress_path=directory / "progress.jsonl")
        checkpoint = models.save_model(model, directory / "checkpoint", algorithm=algorithm, upstream_root=task.get("upstream"))
        record = {"symbol": symbol, "algorithm": algorithm, "seed": seed, "started_at": started, "completed_at": now(),
                  "capital": 10000., "ensemble": False, "primary_seed": seed == 0, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                  "pid": os.getpid(), "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
                  "training": training, "checkpoint": checkpoint, "observation_shape": [13], "action_shape": [1],
                  "maximum_input_hour": str(data["hourly_close"].index[-1]), "training_window_quality": env.training_window_metadata,
                  "files": {p.relative_to(directory).as_posix(): sha(p) for p in (directory / "checkpoint").rglob("*") if p.is_file()}}
        write(directory / "record.json", record, exclusive=True)
        return record
    except Exception as exc:
        write(directory / "failed.json", {"at": now(), "error_type": type(exc).__name__, "error": str(exc)})
        raise


def run_parallel(output, tasks, function, phase, max_workers=None):
    inventory = resources()
    gpus = inventory["gpus"]
    if len(gpus) < 2:
        raise ValueError("This authorized study requires both available V100 devices")
    # Keep explicit free-memory headroom; real jobs, not utilization busy loops.
    capacity = [max(0, int(g["free_mb"] * .80 // 900)) for g in gpus]
    total = min(len(tasks), max(1, int(inventory["available_cpus"] * .9)), sum(capacity), max_workers or len(tasks))
    if total < 2 or min(capacity) < 1:
        raise ValueError("Insufficient free CPU/GPU resources")
    workers = [0] * len(gpus)
    for _ in range(total):
        slot = min((i for i in range(len(gpus)) if workers[i] < capacity[i]), key=lambda i: workers[i])
        workers[slot] += 1
    ctx = mp.get_context("spawn")
    pools = [ProcessPoolExecutor(max_workers=n, mp_context=ctx, initializer=initialize_worker, initargs=(g["uuid"],)) for n, g in zip(workers, gpus)]
    results, failures = [], []
    write(Path(output) / f"{phase}_resources.json", {**inventory, "worker_counts": workers, "tasks": len(tasks)})
    try:
        futures = {pools[i % len(pools)].submit(function, task): task for i, task in enumerate(tasks)}
        for future in as_completed(futures):
            task = futures[future]
            try:
                results.append(future.result())
            except Exception as exc:
                failures.append({"task": task, "error_type": type(exc).__name__, "error": str(exc)})
            write(Path(output) / f"{phase}_status.json", {"at": now(), "completed": len(results), "failed": failures, "total": len(tasks)})
    finally:
        for pool in pools:
            pool.shutdown(wait=True, cancel_futures=False)
    if failures:
        raise RuntimeError(f"{phase} has {len(failures)} failed attempts; preserve records and completed work")
    return results


def validate_training_record(output, symbol, algorithm, seed, spec):
    directory = Path(output) / "training" / symbol / algorithm / f"seed{seed}"
    record = read(directory / "record.json")
    manifest = read(directory / "checkpoint/single_pair_model.json")
    identity = {"symbol": symbol, "algorithm": algorithm, "seed": seed}
    if any(record.get(key) != value or manifest.get(key) != value for key, value in identity.items()):
        raise ValueError("Checkpoint and training record must belong to this exact pair/algorithm/seed")
    if (not record.get("files") or any(sha(directory / name) != expected for name, expected in record["files"].items())
            or record.get("capital") != 10000. or record.get("ensemble") is not False
            or record.get("observation_shape") != [13] or record.get("action_shape") != [1]
            or manifest.get("observation_shape") != [13] or manifest.get("action_shape") != [1]
            or manifest.get("independent_account") is not True or manifest.get("multi_model_target_average") is not False
            or manifest.get("adapter_sha256") != spec["sources"]["scripts/single_pair_rl_models.py"]):
        raise ValueError("Completed single-pair checkpoint bytes or identity changed")
    training = record["training"]
    steps = spec["timesteps_per_model"][algorithm]
    if (training != manifest.get("training") or record.get("checkpoint") != manifest
            or training.get("actual_steps") != steps or training.get("counters", {}).get("environment_steps") != steps
            or pd.Timestamp(record["maximum_input_hour"]) >= pd.Timestamp(TRAIN[1])):
        raise ValueError("Actual training budget/input cutoff differs from the frozen recipe")
    if algorithm in ("sac", "tqc"):
        counters = training["counters"]
        updates = steps - 512
        if (counters.get("gradient_update_rounds") != updates or counters.get("replay_transitions") != steps
                or counters.get("replay_add_calls") != steps
                or counters.get("optimizer_steps") != {"actor": updates, "critic": updates, "entropy": updates}):
            raise ValueError("Actual actor/critic/entropy/replay update counts differ")
    return record


def fit(output, dataset, upstream=None, max_workers=None):
    spec = check_protocol(output, dataset)
    tasks = []
    for seed in SEEDS:
        for algorithm in spec["algorithms"]:
            for symbol in SYMBOLS:
                directory = Path(output) / "training" / symbol / algorithm / f"seed{seed}"
                if (directory / "record.json").exists():
                    validate_training_record(output, symbol, algorithm, seed, spec)
                    continue
                if directory.exists():
                    raise FileExistsError("Incomplete attempted model needs a separately recorded recovery; never restart silently")
                tasks.append(dict(output=str(output), dataset=str(dataset), upstream=upstream, symbol=symbol, algorithm=algorithm, seed=seed))
    if tasks:
        run_parallel(output, tasks, train_one, "fit", max_workers)
    records = [validate_training_record(output, symbol, algorithm, seed, spec) for symbol in SYMBOLS for algorithm in spec["algorithms"] for seed in SEEDS]
    if len(records) != spec["training_model_count"]:
        raise ValueError("Incomplete model grid")
    write(Path(output) / "fit_complete.json", {"at": now(), "model_count": len(records), "records": records})


def evaluate_case(task):
    from scripts.backtest_btceth_rl import load_dataset
    from scripts.single_pair_rl_env import SinglePairEnv
    from scripts import single_pair_rl_models as models
    from scripts.crypto_rl_reporting import save_result, summarize, canonical_csv, result_fingerprints
    output, symbol, algorithm, seed, scenario, split = (task[key] for key in ("output", "symbol", "algorithm", "seed", "scenario", "split"))
    if (symbol not in SYMBOLS or split not in ("validation", "test") or scenario not in COSTS
            or (seed is None and algorithm not in BASELINES) or (seed is not None and (type(seed) is not int or seed not in SEEDS or algorithm not in ALGORITHMS))):
        raise ValueError("Undeclared single-pair evaluation case")
    bounds = VALIDATION if split == "validation" else TEST
    data = cropped_data(load_dataset(task["dataset"]), bounds[1])
    model = None if seed is None else models.load_model(Path(output) / "training" / symbol / algorithm / f"seed{seed}/checkpoint", algorithm=algorithm, device="cuda:0", upstream_root=task.get("upstream"))
    if model is not None:
        identity = getattr(model, "_single_pair_identity", {})
        if any(identity.get(key) != value for key, value in {"symbol": symbol, "algorithm": algorithm, "seed": seed,
                "observation_shape": [13], "action_shape": [1], "independent_account": True, "multi_model_target_average": False}.items()):
            raise ValueError("Loaded model does not belong to this exact independent trading account")
    before = models.parameter_digest(model, algorithm) if model is not None else None
    run_id = f"{split}__{scenario}__{symbol}__{algorithm}__{'baseline' if seed is None else seed}"
    signatures, action_signatures = [], []
    for repeat in (False, True):
        env = SinglePairEnv(data, symbol, *bounds, costs=COSTS[scenario], capital=10000., drawdown_penalty=.25, rebalance_band=.025)
        obs, _ = env.reset(seed=0)
        state, first, done, actions = None, True, False, []
        while not done:
            day = env.current_date
            if model is not None:
                action, state = models.predict(model, obs, algorithm=algorithm, state=state, episode_start=first)
                obs, _, terminated, truncated, info = env.step(action)
                actions.append({"date": str(day), "action": np.asarray(action).reshape(-1).tolist(), "target": info["target_weights"].tolist()})
            else:
                if algorithm == "buy_hold" and not first:
                    obs, _, terminated, truncated, _ = env.step_hold()
                else:
                    on = algorithm == "buy_hold" or algorithm == "sma200_trend" and env._features[day][7] > 0 or algorithm == "momentum180" and env._features[day][4] > 0
                    obs, _, terminated, truncated, _ = env.step_target([.95 if on else 0., .05 if on else 1.])
            first, done = False, terminated or truncated
        result = env.get_result()
        signatures.append(result_fingerprints(result))
        directory = Path(output) / ("repeat" if repeat else "evaluation") / split / symbol / algorithm / ("baseline" if seed is None else f"seed{seed}") / scenario
        directory.mkdir(parents=True, exist_ok=False)
        save_result(directory, result)
        write(directory / "actions.json", actions)
        action_signatures.append(sha(directory / "actions.json"))
        row, daily, monthly = summarize(result, run_id=run_id, strategy=algorithm, seed=seed, scenario=scenario, split=split, start=bounds[0], end=bounds[1], capital=10000., **COSTS[scenario])
        row.update(symbol=symbol, algorithm=algorithm, primary_seed=seed == 0, ensemble=False, independent_capital=True)
        write(directory / "summary.json", row)
        (directory / "daily_equity.csv").write_bytes(canonical_csv(daily))
        (directory / "monthly_returns.csv").write_bytes(canonical_csv(monthly))
    if signatures[0] != signatures[1] or action_signatures[0] != action_signatures[1] or model is not None and models.parameter_digest(model, algorithm) != before:
        raise ValueError("Deterministic replay or unchanged model check failed")
    canonical = Path(output) / "evaluation" / split / symbol / algorithm / ("baseline" if seed is None else f"seed{seed}") / scenario
    return {"summary": row, "path": canonical.relative_to(Path(output)).as_posix(),
            "repeat_path": directory.relative_to(Path(output)).as_posix(), "repeat_equal": True}


def evaluate(output, dataset, split, upstream=None, max_workers=None):
    from scripts.crypto_rl_reporting import canonical_csv
    spec = check_protocol(output, dataset)
    if not (Path(output) / "fit_complete.json").exists():
        raise ValueError("All fresh independent models must finish training first")
    if split == "test" and not (Path(output) / "validation_selection.json").exists():
        raise ValueError("Freeze validation-only algorithm choice before annual replay")
    if split == "test":
        selection = read(Path(output) / "validation_selection.json")
        if (selection.get("uses_test") is not False or selection.get("seed_selection") is not False
                or selection.get("validation_audit_sha256") != sha(Path(output) / "validation_audit.json")
                or selection.get("validation_summary_sha256") != sha(Path(output) / "validation_summary.csv")
                or selection.get("protocol_sha256") != sha(Path(output) / "protocol.json")):
            raise ValueError("Annual replay requires the exact audited validation-only selection")
        started = Path(output) / "test_started.json"
        pins = {"selection_sha256": sha(Path(output) / "validation_selection.json"), "validation_audit_sha256": sha(Path(output) / "validation_audit.json"),
                "protocol_sha256": sha(Path(output) / "protocol.json"), "dataset_sha256": spec["dataset_sha256"]}
        if started.exists():
            if read(started)["pins"] != pins:
                raise ValueError("Annual pre-start pins changed")
        else:
            write(started, {"at": now(), "pins": pins}, exclusive=True)
    tasks = [dict(output=str(output), dataset=str(dataset), upstream=upstream, symbol=symbol, algorithm=algorithm, seed=seed, scenario=scenario, split=split)
             for symbol in SYMBOLS for algorithm in spec["algorithms"] for seed in SEEDS for scenario in COSTS]
    tasks += [dict(output=str(output), dataset=str(dataset), upstream=upstream, symbol=symbol, algorithm=algorithm, seed=None, scenario=scenario, split=split)
              for symbol in SYMBOLS for algorithm in BASELINES for scenario in COSTS]
    results = run_parallel(output, tasks, evaluate_case, split, max_workers)
    summary = pd.DataFrame([result["summary"] for result in results]).sort_values("run_id")
    (Path(output) / f"{split}_summary.csv").write_bytes(canonical_csv(summary))
    write(Path(output) / f"{split}_complete.json", {"at": now(), "accounts": len(results), "repeats": len(results), "repeat_equal": True, "results": results})
    return summary


def select(output):
    spec = read(Path(output) / "protocol.json")
    audit = read(Path(output) / "validation_audit.json")
    if audit.get("passed") is not True:
        raise ValueError("Independent validation financial audit must pass before selection")
    if (Path(output) / "test_complete.json").exists() or (Path(output) / "test_started.json").exists():
        raise ValueError("Never select again after annual results")
    if (audit.get("summary_sha256") != sha(Path(output) / "validation_summary.csv")
            or audit.get("protocol_sha256") != sha(Path(output) / "protocol.json")):
        raise ValueError("Validation audit is not bound to these exact summary/protocol bytes")
    frame = pd.read_csv(Path(output) / "validation_summary.csv")
    identities = {(symbol, algorithm, seed, scenario) for symbol in SYMBOLS for algorithm in spec["algorithms"] for seed in SEEDS for scenario in COSTS}
    identities |= {(symbol, algorithm, None, scenario) for symbol in SYMBOLS for algorithm in BASELINES for scenario in COSTS}
    observed = {(row.symbol, row.algorithm, None if pd.isna(row.seed) else float(row.seed), row.scenario) for row in frame.itertuples()}
    expected_run_ids = {f"validation__{scenario}__{symbol}__{algorithm}__{'baseline' if seed is None else seed}" for symbol, algorithm, seed, scenario in identities}
    if (len(frame) != len(identities) or observed != identities or set(frame.run_id) != expected_run_ids
            or set(frame.split) != {"validation"} or not frame.initial_capital.eq(10000.).all()):
        raise ValueError("Every single-pair validation identity is required once at both costs")
    selection = {}
    for symbol in SYMBOLS:
        rows = []
        for algorithm in spec["algorithms"]:
            group = frame.loc[(frame.symbol == symbol) & (frame.algorithm == algorithm)]
            if len(group) != 10 or set(group.seed) != set(SEEDS):
                raise ValueError("Need five full seeds at both costs")
            scores = [np.mean([math.log1p(row.net_return) - .5 * abs(row.max_drawdown) - 2 * max(abs(row.max_drawdown) - .2, 0.) for row in group.loc[group.seed == seed].itertuples()]) for seed in SEEDS]
            rows.append({"algorithm": algorithm, "score_median": float(np.median(scores)), "seed_scores": scores})
        rows.sort(key=lambda row: (-row["score_median"], row["algorithm"]))
        selection[symbol] = {"algorithm": rows[0]["algorithm"], "seed": 0, "all_candidates": rows}
    write(Path(output) / "validation_selection.json", {"at": now(), "uses_test": False, "seed_selection": False, "selection": selection,
            "validation_audit_sha256": sha(Path(output) / "validation_audit.json"), "validation_summary_sha256": sha(Path(output) / "validation_summary.csv"),
            "protocol_sha256": sha(Path(output) / "protocol.json")}, exclusive=True)
    return selection


def pipeline(output, dataset, upstream=None, max_workers=None):
    """Continue only dependent, checked phases; preserve every terminal record."""
    phases = ("fit", "validation", "audit-validation", "select", "test", "audit-test")
    script = ROOT / "scripts/backtest_single_pair_rl.py"
    check_protocol(output, dataset)
    for phase in phases:
        marker = {"fit": "fit_complete.json", "validation": "validation_complete.json", "audit-validation": "validation_audit.json",
                  "select": "validation_selection.json", "test": "test_complete.json", "audit-test": "test_audit.json"}[phase]
        if (Path(output) / marker).exists():
            saved = read(Path(output) / marker)
            if phase.startswith("audit-") and saved.get("passed") is not True:
                raise RuntimeError("Preserve failed audit; repair in a separate recorded attempt")
            continue
        command = [sys.executable, "-B", str(script), phase, "--output", str(output), "--dataset", str(dataset)]
        if upstream:
            command += ["--upstream", upstream]
        if max_workers:
            command += ["--max-workers", str(max_workers)]
        write(Path(output) / "pipeline_status.json", {"at": now(), "phase": phase, "status": "running", "pid": os.getpid()})
        with (Path(output) / f"{phase}.log").open("x", encoding="utf-8") as stream:
            process = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
        if process.returncode:
            write(Path(output) / "pipeline_status.json", {"at": now(), "phase": phase, "status": "failed", "exit_code": process.returncode})
            raise RuntimeError("Pipeline phase failed: " + phase)
    write(Path(output) / "pipeline_status.json", {"at": now(), "status": "completed", "no_production_change": True})
    return {"completed": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "fit", "validation", "select", "test", "audit-validation", "audit-test", "pipeline"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--algorithms", nargs="+", default=["sac", "tqc"])
    parser.add_argument("--upstream")
    parser.add_argument("--max-workers", type=int)
    args = parser.parse_args()
    if args.phase == "prepare":
        result = prepare(args.output, args.dataset, args.algorithms)
    elif args.phase == "fit":
        result = fit(args.output, args.dataset, args.upstream, args.max_workers)
    elif args.phase == "select":
        result = select(args.output)
    elif args.phase == "pipeline":
        result = pipeline(args.output, args.dataset, args.upstream, args.max_workers)
    elif args.phase.startswith("audit-"):
        spec = check_protocol(args.output, args.dataset)
        if args.phase == "audit-test":
            expected = {"selection_sha256": sha(args.output / "validation_selection.json"), "validation_audit_sha256": sha(args.output / "validation_audit.json"),
                        "protocol_sha256": sha(args.output / "protocol.json"), "dataset_sha256": spec["dataset_sha256"]}
            if read(args.output / "test_started.json")["pins"] != expected:
                raise ValueError("Annual source/selection pins changed before final audit")
        from scripts.audit_single_pair_rl import audit_phase
        result = audit_phase(args.output, args.dataset, args.phase.removeprefix("audit-"))
        if not result.get("passed"):
            raise RuntimeError("Independent single-pair financial audit failed")
    else:
        result = evaluate(args.output, args.dataset, args.phase, args.upstream, args.max_workers)
        result = {"accounts": len(result)}
    print(json.dumps({"phase": args.phase, "completed_at": now(), "result": result}, default=str))


if __name__ == "__main__":
    main()

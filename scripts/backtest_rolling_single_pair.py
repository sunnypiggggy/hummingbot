"""Frozen historical, validation-selected independent-account retraining study.

No exchange or production access. All three arms share fresh initial weights;
old fixed-weight studies remain separate historical references.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata
import json
import multiprocessing as mp
import os
from pathlib import Path
import subprocess
import sys
import time

for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_key] = "1"
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

TEST = ("2025-10-23T00:00:00Z", "2026-10-23T00:00:00Z")
SEEDS = tuple(range(5))
SELECTED = {"BTCUSDT": ["td3_default", "sac_default", "sac_gamma099"],
            "ETHUSDT": ["sac_default", "tqc_default", "sac_gamma099"]}
CADENCES = {"fixed": None, "every_4w": 28, "every_week": 7}
COSTS = {"base": {"fee": .001, "slippage": .0002}, "stress": {"fee": .002, "slippage": .0005}}
CORE_FILES = ("backtest_single_pair_rl", "single_pair_rl_env", "single_pair_rl_models", "audit_single_pair_rl",
              "backtest_btceth_rl", "btceth_rl_env", "crypto_rl_portfolio_env", "crypto_rl_reporting",
              "crypto_rl_data", "btceth_rl_td3", "btceth_rl_droq")
SOURCES = tuple("scripts/" + name + ".py" for name in CORE_FILES) + (
    "scripts/rolling_single_pair_models.py", "scripts/backtest_rolling_single_pair.py", "scripts/audit_rolling_single_pair.py")
RESULT_FILES = ("equity_1h.csv.gz", "fills.csv", "weights.csv", "boundary_equity.csv", "metadata.json",
                "fingerprints.json", "actions.json", "summary.json", "daily_equity.csv", "monthly_returns.csv", "swaps.json")
_DATA = None


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value, exclusive=False):
    from scripts.crypto_rl_reporting import clean_json
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(clean_json(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if exclusive:
        with path.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(body)
    else:
        temporary = path.with_suffix(path.suffix + ".pending")
        temporary.write_text(body, encoding="utf-8")
        os.replace(temporary, path)


def model_context(spec, protocol_sha256, symbol, recipe_id, seed, version_index):
    if symbol not in spec["selected"] or recipe_id not in spec["selected"][symbol] or seed not in spec["seeds"]:
        raise ValueError("Undeclared model family")
    if type(version_index) is not int or not 0 <= version_index <= 52:
        raise ValueError("Undeclared weekly model version")
    day = pd.Timestamp(spec["test"][0]) + pd.Timedelta(days=7 * version_index)
    end = day - pd.Timedelta(days=spec["lag_days"])
    start = end - pd.Timedelta(days=spec["train_days"])
    return dict(symbol=symbol, algorithm=spec["recipes"][recipe_id]["algorithm"], recipe_id=recipe_id,
                seed=seed, version_index=version_index, effective_day=day.isoformat(),
                train_start=start.isoformat(), train_end=end.isoformat(), protocol_sha256=protocol_sha256)


def version_for_day(day, start, cadence):
    offset = (pd.Timestamp(day) - pd.Timestamp(start)).days
    if not 0 <= offset < 365 or cadence not in CADENCES:
        raise ValueError("Invalid account day or cadence")
    every = CADENCES[cadence]
    return 0 if every is None else offset // every * (every // 7)


def selection_from_validation(output, inventory):
    """Only these validation CSVs and declared native recipes enter selection."""
    from scripts.single_pair_rl_models import recipe as native_recipe
    output = Path(output)
    base = output / "selection_inputs"
    native_spec = read(base / "native_protocol.json")
    native = pd.read_csv(base / "native_validation_summary.csv", float_precision="round_trip")
    input_files = [base / "native_protocol.json", base / "native_validation_summary.csv"]
    populations = [("native_" + alg, native[native.algorithm.eq(alg)], tuple(range(5)), native_recipe(alg))
                   for alg in native_spec["algorithms"]]
    for name, gamma in (("c01_lr1e4", .995), ("c02_gamma099", .99), ("c03_lr1e4_gamma099", .99)):
        path = base / (name + "_validation_summary.csv")
        frame = pd.read_csv(path, float_precision="round_trip")
        params = native_recipe("sac")
        params.update(learning_rate=.0003 if name == "c02_gamma099" else .0001, gamma=gamma)
        populations.append(("sac_" + name, frame[frame.algorithm.eq("sac")], tuple(range(100, 105)), params))
        input_files.append(path)
    stats, rankings = [], {}
    for symbol in SELECTED:
        rows = []
        for identifier, data, seeds, parameters in populations:
            frame = data[data.symbol.eq(symbol)].copy()
            if (len(frame) != 10 or set(frame.split) != {"validation"} or not frame.initial_capital.eq(10000.).all()
                    or set(frame.scenario) != set(COSTS) or frame.duplicated(["seed", "scenario"]).any()
                    or set(frame.seed) != set(seeds)):
                raise ValueError("Incomplete five-seed two-cost validation population")
            for value in frame.start:
                if pd.Timestamp(value) != pd.Timestamp("2025-01-01T00:00:00Z"):
                    raise ValueError("Validation start changed")
            for value in frame.end_exclusive:
                if pd.Timestamp(value) != pd.Timestamp(TEST[0]):
                    raise ValueError("Validation cutoff changed")
            utilities = []
            for seed in seeds:
                account = frame[frame.seed.eq(seed)]
                utilities.append(float(np.mean([np.log1p(r.net_return) - .5 * abs(r.max_drawdown)
                    - 2 * max(abs(r.max_drawdown) - .20, 0) for r in account.itertuples()])))
            score = float(np.median(utilities) - .25 * (np.quantile(utilities, .75) - np.quantile(utilities, .25)))
            alg = str(frame.algorithm.iloc[0])
            if parameters != native_spec["recipes"][alg] and not identifier.startswith("sac_c"):
                raise ValueError("Native recipe differs from frozen reference")
            item = dict(symbol=symbol, candidate_id=identifier, algorithm=alg, recipe=parameters,
                        tuning_seeds=list(seeds), robust_score=score, seed_utilities=utilities,
                        cashlike_degenerate=bool(frame.fills.eq(0).all()))
            matched = [r for r in inventory["candidates"] if r["symbol"] == symbol and r["candidate_id"] == identifier]
            if (len(matched) != 1 or matched[0]["recipe"] != parameters
                    or abs(matched[0]["robust_score"] - score) > 1e-12):
                raise ValueError("Inventory ranking does not reconcile to validation-only inputs")
            rows.append(item)
        ranking = sorted([r for r in rows if not r["cashlike_degenerate"]],
                         key=lambda r: (-r["robust_score"], r["candidate_id"]))
        rankings[symbol] = [r["candidate_id"] for r in ranking]
        stats.extend(rows)
    mapping = {"native_sac": "sac_default", "native_td3": "td3_default", "native_tqc": "tqc_default",
               "sac_c02_gamma099": "sac_gamma099"}
    chosen = {s: [mapping[c] for c in rankings[s][:3]] for s in SELECTED}
    if chosen != SELECTED:
        raise ValueError("Prespecified shortlist differs from independently recomputed validation ranking")
    return dict(at=now(), uses_test=False, seed_selection=False, selected=chosen, rankings=rankings, candidates=stats,
                source_sha256={p.relative_to(output).as_posix(): sha(p) for p in input_files},
                inventory_sha256=sha(output / "inventory.json"),
                caveat="Retrospective research shortlist; not a change to any old acceptance decision or production authority.")


def prepare(output, dataset):
    from scripts.single_pair_rl_models import recipe, runtime_versions
    output, dataset = Path(output), Path(dataset)
    if (output / "protocol.json").exists():
        raise FileExistsError("Preserve an existing frozen protocol")
    inventory = read(output / "inventory.json")
    selected = selection_from_validation(output, inventory)
    recipes = {}
    for identifier, algorithm in (("sac_default", "sac"), ("sac_gamma099", "sac"),
                                  ("td3_default", "td3"), ("tqc_default", "tqc")):
        parameters = recipe(algorithm)
        if identifier == "sac_gamma099":
            parameters["gamma"] = .99
        recipes[identifier] = dict(algorithm=algorithm, parameters=parameters, timesteps=65536 if algorithm == "td3" else 32768)
    write(output / "selection.json", selected, exclusive=True)
    spec = dict(schema_version=1, frozen_at=now(), study="all algorithm independent-pair retraining comparison",
                test=list(TEST), seeds=list(SEEDS), selected=SELECTED, recipes=recipes, cadences=CADENCES,
                train_days=1827, lag_days=1, preheat_days=200, capital_per_account=10000.,
                capital_shared_between_pairs=False, ensemble=False, costs=COSTS,
                selection_uses_test=False, fresh_holdout=False, test_year_already_observed=True,
                inventory_sha256=sha(output / "inventory.json"), selection_sha256=sha(output / "selection.json"),
                dataset_sha256=sha(dataset), dataset_manifest_sha256=sha(dataset.parent / "manifest.json"),
                sources={p: sha(ROOT / p) for p in SOURCES},
                runtime_packages={**runtime_versions(), "pandas": importlib.metadata.version("pandas")}, python=sys.version,
                expected_models=1590, expected_accounts=180, original_fixed_reference_is_separate=True,
                initial_model="Fresh same checkpoint shared by fixed/every_4w/every_week at test start",
                timing="24h information cutoff lag; offline calculation duration is not simulated deployment latency",
                retraining="From scratch at every cutoff, same master seed, original algorithm budget, base training costs",
                continuation="No optimizer/replay resume; completed checkpoints reused by exact context only",
                environment=dict(observation_dim=13, action_dim=1, cash_floor=.05, max_coin_target=.95,
                                 rebalance_band=.025, drawdown_penalty=.25, episode_days=32,
                                 execution="00UTC sells;01UTC buys;hourly close valuations"),
                no_production_change=True)
    write(output / "protocol.json", spec, exclusive=True)
    write(output / "protocol_receipt.json", dict(at=now(), protocol_sha256=sha(output / "protocol.json")), exclusive=True)
    return spec


def check_protocol(output, dataset=None):
    from scripts.single_pair_rl_models import runtime_versions
    output = Path(output)
    spec = read(output / "protocol.json")
    if sha(output / "protocol.json") != read(output / "protocol_receipt.json")["protocol_sha256"]:
        raise ValueError("Protocol bytes changed")
    if (spec["test"] != list(TEST) or spec["selected"] != SELECTED or spec["seeds"] != list(SEEDS)
            or spec["cadences"] != CADENCES or spec["costs"] != COSTS or spec["train_days"] != 1827
            or spec["lag_days"] != 1 or spec["preheat_days"] != 200 or spec["expected_models"] != 1590
            or spec["expected_accounts"] != 180 or spec["capital_per_account"] != 10000.
            or spec["selection_uses_test"] is not False or spec["ensemble"] is not False):
        raise ValueError("Frozen research grid or financial contract changed")
    if sha(output / "inventory.json") != spec["inventory_sha256"] or sha(output / "selection.json") != spec["selection_sha256"]:
        raise ValueError("Selection evidence changed")
    if spec.get("runtime_packages") != {**runtime_versions(), "pandas": importlib.metadata.version("pandas")} or spec.get("python") != sys.version:
        raise ValueError("Executed learner/runtime differs from frozen versions")
    for path, expected in spec["sources"].items():
        if sha(ROOT / path) != expected:
            raise ValueError("Executed source changed: " + path)
    for path, expected in read(output / "selection.json")["source_sha256"].items():
        if sha(output / path) != expected:
            raise ValueError("Validation selection input changed")
    if dataset is not None:
        if sha(dataset) != spec["dataset_sha256"] or sha(Path(dataset).parent / "manifest.json") != spec["dataset_manifest_sha256"]:
            raise ValueError("Market input changed")
    return spec


def dataset_once(path):
    global _DATA
    if _DATA is None:
        from scripts.backtest_btceth_rl import load_dataset
        _DATA = load_dataset(path)
    return _DATA


def training_data(dataset, context):
    symbol = context["symbol"]
    lower = pd.Timestamp(context["train_start"]) - pd.Timedelta(days=200)
    upper = pd.Timestamp(context["train_end"])
    return {"assets": {symbol: dataset.get("assets", {}).get(symbol, {})},
            **{"hourly_" + name: dataset["hourly_" + name].loc[
                lambda f: (f.index >= lower) & (f.index < upper), [symbol]].copy()
               for name in ("open", "high", "low", "close", "volume")}}


def input_market_digest(data, symbol):
    digest = hashlib.sha256()
    for name in ("open", "high", "low", "close", "volume"):
        frame = data["hourly_" + name].loc[:, [symbol]]
        digest.update(name.encode() + b"\0")
        digest.update(np.asarray(frame.shape, dtype="<i8").tobytes())
        digest.update(frame.index.tz_convert("UTC").asi8.astype("<i8").tobytes())
        digest.update(frame.to_numpy(dtype="<f8").tobytes(order="C"))
    return digest.hexdigest()


def model_directory(output, context):
    return Path(output) / "models" / context["symbol"] / context["recipe_id"] / f"seed{context['seed']}" / f"v{context['version_index']:02d}"


def contexts(output, spec):
    digest = sha(Path(output) / "protocol.json")
    return [model_context(spec, digest, symbol, recipe_id, seed, version)
            for version in range(53) for symbol in spec["selected"]
            for recipe_id in spec["selected"][symbol] for seed in spec["seeds"]]


def train_one(task):
    from scripts.single_pair_rl_env import SinglePairEnv
    from scripts import rolling_single_pair_models as models
    output, context = Path(task["output"]), task["context"]
    spec = check_protocol(output)
    directory = model_directory(output, context)
    directory.mkdir(parents=True, exist_ok=False)
    write(directory / "started.json", dict(at=now(), context=context, pid=os.getpid(), gpu=os.environ.get("CUDA_VISIBLE_DEVICES")), exclusive=True)
    try:
        data = training_data(dataset_once(task["dataset"]), context)
        env = SinglePairEnv(data, context["symbol"], context["train_start"], context["train_end"],
                            costs=COSTS["base"], capital=10000., training=True, episode_days=32, record_history=False)
        model = models.make_model(env, context["algorithm"], context["recipe_id"], context["seed"], context,
                                  trace_path=directory / "timeout_trace.jsonl", device="cuda:0")
        training = models.train_model(model, env, context["algorithm"], spec["recipes"][context["recipe_id"]]["timesteps"],
                                      context["seed"], progress_path=directory / "progress.jsonl")
        checkpoint = models.save_model(model, env, context["algorithm"], context["seed"], directory / "checkpoint", context)
        record = dict(**{k: context[k] for k in ("symbol", "algorithm", "seed", "recipe_id")}, context=context,
                      completed_at=now(), training=training, checkpoint=checkpoint,
                      maximum_input_hour=str(data["hourly_close"].index[-1]), input_market_sha256=input_market_digest(data, context["symbol"]),
                      training_window_quality=env.training_window_metadata, capital=10000., ensemble=False,
                      observation_shape=[13], action_shape=[1], pid=os.getpid(), cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
                      cpu_affinity=sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
                      files={p.relative_to(directory).as_posix(): sha(p) for p in (directory / "checkpoint").iterdir() if p.is_file()})
        write(directory / "record.json", record, exclusive=True)
        return dict(path=directory.relative_to(output).as_posix(), sha256=sha(directory / "record.json"), context=context)
    except Exception as error:
        write(directory / "failed.json", dict(at=now(), type=type(error).__name__, error=str(error)), exclusive=True)
        raise
    finally:
        # Native callbacks may form model/environment cycles. Release them
        # between independent fits so a long worker does not accumulate GPUs.
        if "model" in locals():
            del model
        if "env" in locals():
            env.close()
            del env
        gc.collect()
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def worker_init(uuid=None):
    if uuid is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = uuid
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[key] = "1"


def preflight_one(task):
    """Real native GPU fits outside the financial experiment, never reused."""
    from scripts.single_pair_rl_env import SinglePairEnv
    from scripts import rolling_single_pair_models as models
    from scripts.single_pair_rl_models import observation
    import torch
    torch.cuda.reset_peak_memory_stats()
    context, directory = task["context"], Path(task["directory"])
    directory.mkdir(parents=True, exist_ok=False)
    data = training_data(dataset_once(task["dataset"]), context)
    env = SinglePairEnv(data, context["symbol"], context["train_start"], context["train_end"],
                        costs=COSTS["base"], training=True, episode_days=32, record_history=False)
    model = models.make_model(env, context["algorithm"], context["recipe_id"], context["seed"], context,
                              trace_path=directory / "timeouts.jsonl", device="cuda:0")
    training = models.train_model(model, env, context["algorithm"], 1024, context["seed"],
                                  progress_path=directory / "progress.jsonl")
    manifest = models.save_model(model, env, context["algorithm"], context["seed"], directory / "checkpoint", context)
    restored = models.load_model(directory / "checkpoint", context["algorithm"], context["symbol"],
                                 context["seed"], context, device="cuda:0")
    obs, unused = env.reset(seed=0)
    first = models.predict(model, obs, context["algorithm"], episode_start=True)[0]
    second = models.predict(restored, obs, context["algorithm"], episode_start=True)[0]
    completed = [h for h in training["training_episode_history"] if h.get("completed_steps", 0) > 0]
    if (not np.array_equal(first, second) or models.digest(model) != models.digest(restored)
            or training["actual_steps"] != 1024 or len(completed) != 32
            or any(h["completed_steps"] != 32 or not h["truncated"] or h["terminated"] for h in completed)):
        raise ValueError("GPU-native model preflight failed")
    value = dict(at=now(), passed=True, nonfinancial=True, reused_by_formal_study=False,
                 context=context, pid=os.getpid(), gpu=os.environ.get("CUDA_VISIBLE_DEVICES"),
                 training=training, manifest=manifest, observation=observation(obs)[0], action=first.tolist())
    import torch
    value["cuda_memory"] = dict(peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                                peak_reserved_bytes=torch.cuda.max_memory_reserved())
    observed = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                              capture_output=True, text=True, check=True)
    value["cuda_memory"]["driver_process_mb"] = [float(line.split(",")[1].strip()) for line in observed.stdout.splitlines()
           if line.split(",")[0].strip() == str(os.getpid())]
    write(directory / "result.json", value, exclusive=True)
    return dict(directory=str(directory), sha256=sha(directory / "result.json"), passed=True)


def preflight(output, dataset):
    from scripts.backtest_single_pair_rl import resources
    output = Path(output)
    spec = check_protocol(output, dataset)
    inventory = resources()
    if len(inventory["gpus"]) != 2 or any(g["free_mb"] < 2200 for g in inventory["gpus"]):
        raise RuntimeError("Two available GPUs required for actual native preflight")
    tasks = []
    for gpu_index in range(2):
        for identifier in ("sac_default", "sac_gamma099", "td3_default", "tqc_default"):
            symbol = "ETHUSDT" if identifier == "tqc_default" else "BTCUSDT"
            context = model_context(spec, sha(output / "protocol.json"), symbol, identifier, 0, 0)
            context["nonfinancial_preflight"] = True
            tasks.append(dict(context=context, dataset=str(dataset), gpu_index=gpu_index,
                directory=str(output / "preflight" / f"gpu{gpu_index}" / identifier)))
    pools = [ProcessPoolExecutor(max_workers=4, mp_context=mp.get_context("spawn"), initializer=worker_init,
                                 initargs=(g["uuid"],)) for g in inventory["gpus"]]
    try:
        futures = [pools[t["gpu_index"]].submit(preflight_one, t) for t in tasks]
        results = [f.result() for f in futures]
    finally:
        for pool in pools:
            pool.shutdown(wait=True, cancel_futures=False)
    check_protocol(output, dataset)
    write(output / "preflight_complete.json", dict(at=now(), passed=True, actual_gpu_models=8,
          original_sources_unchanged=True, resources=inventory, results=results), exclusive=True)


def fit(output, dataset, max_workers=None):
    from scripts.backtest_single_pair_rl import resources
    output = Path(output)
    spec = check_protocol(output, dataset)
    preflight_evidence = read(output / "preflight_complete.json")
    if preflight_evidence.get("passed") is not True or preflight_evidence.get("actual_gpu_models") != 8:
        raise ValueError("Actual dual-GPU native preflight must pass first")
    observed_memory = []
    for item in preflight_evidence["results"]:
        path = Path(item["directory"]).resolve()
        path.relative_to(output.resolve())
        if sha(path / "result.json") != item["sha256"]:
            raise ValueError("GPU preflight evidence changed")
        measured = read(path / "result.json")["cuda_memory"]["driver_process_mb"]
        if not measured:
            raise ValueError("Actual driver GPU process memory was not measured")
        observed_memory.extend(measured)
    tasks, retained = [], []
    for context in contexts(output, spec):
        directory = model_directory(output, context)
        if (directory / "record.json").exists():
            from scripts.audit_rolling_single_pair import audit_model_evidence
            row = read(directory / "record.json")
            if row["context"] != context or any(sha(directory / p) != h for p, h in row["files"].items()):
                raise ValueError("Completed training record changed")
            audit_model_evidence(row, read(directory / "checkpoint/rolling_model.json"), context,
                spec["recipes"][context["recipe_id"]]["parameters"], spec["recipes"][context["recipe_id"]]["timesteps"], directory)
            retained.append(directory.relative_to(output).as_posix())
        elif directory.exists():
            raise FileExistsError("Preserve incomplete attempted training: " + str(directory))
        else:
            tasks.append(dict(output=str(output), dataset=str(dataset), context=context))
    if not tasks:
        marker = output / "fit_complete.json"
        if marker.exists():
            if read(marker).get("models") != spec["expected_models"]:
                raise ValueError("Existing model completion marker differs")
        else:
            write(marker, dict(at=now(), models=spec["expected_models"], records=[], retained=retained), exclusive=True)
        return
    inventory = resources()
    gpus = inventory["gpus"]
    if len(gpus) != 2:
        raise RuntimeError("Two GPU research configuration required")
    # Include the driver/context allocation and retain at least 10% headroom.
    # A lower bound avoids optimistic capacities from a short native preflight.
    memory_budget_mb = max(900, int(np.ceil(max(observed_memory) * 1.25)))
    capacities = [int(g["free_mb"] * .90 // memory_budget_mb) for g in gpus]
    total = min(len(tasks), int(inventory["available_cpus"] * .9), sum(capacities), max_workers or 9999)
    if min(capacities) < 1 or total < 2:
        raise RuntimeError("Insufficient research resources")
    workers = [0, 0]
    for unused in range(total):
        slot = min((i for i in range(2) if workers[i] < capacities[i]), key=lambda i: workers[i])
        workers[slot] += 1
    resource_path = output / ("fit_resources.json" if not (output / "fit_resources.json").exists()
                              else f"fit_resources_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}.json")
    write(resource_path, dict(**inventory, workers=workers, tasks=len(tasks), retained=retained,
          measured_driver_process_peak_mb=max(observed_memory), memory_budget_per_worker_mb=memory_budget_mb,
          memory_headroom_fraction=.10), exclusive=True)
    pools = [ProcessPoolExecutor(max_workers=n, mp_context=mp.get_context("spawn"), initializer=worker_init,
                                 initargs=(g["uuid"],)) for n, g in zip(workers, gpus)]
    complete, failures = [], []
    try:
        assigned = [0, 0]
        futures = {}
        for task in tasks:
            slot = min((i for i in range(2) if workers[i]), key=lambda i: assigned[i] / workers[i])
            futures[pools[slot].submit(train_one, task)] = task
            assigned[slot] += 1
        for future in as_completed(futures):
            task = futures[future]
            try:
                complete.append(future.result())
            except Exception as error:
                failures.append(dict(context=task["context"], type=type(error).__name__, error=str(error)))
            write(output / "fit_status.json", dict(at=now(), completed=len(complete) + len(retained),
                  total=spec["expected_models"], failed=failures, remaining=len(tasks) - len(complete) - len(failures)))
    finally:
        for pool in pools:
            pool.shutdown(wait=True, cancel_futures=False)
    if failures:
        raise RuntimeError("Training attempts failed; preserve all attempted evidence")
    write(output / "fit_complete.json", dict(at=now(), models=spec["expected_models"], records=complete, retained=retained), exclusive=True)


def snapshot(env):
    from scripts.crypto_rl_reporting import clean_json
    ledger = env.ledger
    value = dict(date=str(env.current_date), cash=ledger.cash, positions=dict(ledger.positions),
                 cost_basis=dict(ledger.cost_basis), realized_pnl=ledger.realized_pnl,
                 nav=env._nav, high_water=env._high_water, fill_count=len(ledger.fills),
                 fills_digest=hashlib.sha256(json.dumps(clean_json(ledger.fills), sort_keys=True,
                    separators=(",", ":"), allow_nan=False).encode()).hexdigest(),
                 equity_rows=len(env._equity_rows), weight_rows=len(env._weight_rows), boundary_rows=len(env._boundary_rows),
                 fees=env._fees, notional=env._notional, slippage_cost=env._slippage_cost, steps=env._steps,
                 last_prices=dict(env._last_prices))
    return clean_json(value)


def evaluate_case(task):
    from scripts.single_pair_rl_env import SinglePairEnv
    from scripts import rolling_single_pair_models as models
    from scripts.crypto_rl_reporting import save_result, summarize, canonical_csv
    output = Path(task["output"])
    spec = check_protocol(output)
    symbol, recipe_id, seed, cadence, scenario = (task[k] for k in ("symbol", "recipe_id", "seed", "cadence", "scenario"))
    algorithm = spec["recipes"][recipe_id]["algorithm"]
    account_root = output / "accounts" / symbol / recipe_id / cadence
    account_id = f"{symbol}__{recipe_id}__{cadence}__seed{seed}__{scenario}"
    receipt_path = account_root / "case_receipts" / f"seed{seed}__{scenario}.json"
    if receipt_path.exists():
        receipt = read(receipt_path)
        if receipt.get("protocol_sha256") != sha(output / "protocol.json") or receipt.get("account_id") != account_id:
            raise ValueError("Completed account belongs to different protocol or identity")
        original = output / receipt["directory"]
        repeat = Path(str(original).replace(os.sep + "evaluation" + os.sep, os.sep + "repeat" + os.sep))
        for name, digest in receipt["files_sha256"].items():
            if sha(original / name) != digest or sha(repeat / name) != digest:
                raise ValueError("Retained completed account changed")
        return dict(summary=read(original / "summary.json"), directory=receipt["directory"], files_sha256=receipt["files_sha256"])
    data = dataset_once(task["dataset"])
    # Evaluation needs the full year. Its raw market features are causal in the frozen environment.
    from scripts.backtest_single_pair_rl import cropped_data
    data = cropped_data(data, TEST[1])
    dirs = []
    for repeated in (False, True):
        directory = account_root / ("repeat" if repeated else "evaluation") / "test" / symbol / algorithm / f"seed{seed}" / scenario
        directory.mkdir(parents=True, exist_ok=False)
        env = SinglePairEnv(data, symbol, *TEST, costs=COSTS[scenario], capital=10000.,
                            rebalance_band=.025, drawdown_penalty=.25)
        obs, unused = env.reset(seed=0)
        model, current_version, current_digest, state, first, done = None, None, None, None, True, False
        actions, swaps = [], []
        while not done:
            day = env.current_date
            version = version_for_day(day, TEST[0], cadence)
            context = model_context(spec, sha(output / "protocol.json"), symbol, recipe_id, seed, version)
            path = model_directory(output, context) / "checkpoint"
            if version != current_version:
                manifest_sha = sha(path / "rolling_model.json")
                if model is not None and models.digest(model, algorithm) != current_digest:
                    raise ValueError("Inference altered model weights")
                before = snapshot(env)
                model = models.load_model(path, algorithm, symbol, seed, context, device="cpu")
                current_digest = models.digest(model, algorithm)
                state, first, current_version = None, True, version
                after = snapshot(env)
                if before != after:
                    raise ValueError("Model switch changed the continuous ledger")
                swaps.append(dict(date=str(day), version_index=version, model_manifest_sha256=manifest_sha,
                                  context=context, before=before, after=after))
            observation = np.asarray(obs, dtype="<f4")
            act, state = models.predict(model, obs, algorithm, state=state, episode_start=first)
            obs, unused, terminated, truncated, info = env.step(act)
            actions.append(dict(date=str(day), action=np.asarray(act).reshape(-1).tolist(),
                                target=info["target_weights"].tolist(), version_index=version,
                                model_manifest_sha256=manifest_sha, observation=observation.tolist(),
                                observation_sha256=hashlib.sha256(observation.tobytes()).hexdigest()))
            first, done = False, terminated or truncated
        if models.digest(model, algorithm) != current_digest:
            raise ValueError("Final inference altered model weights")
        result = env.get_result()
        save_result(directory, result)
        run_id = f"test__{scenario}__{symbol}__{algorithm}__{seed}"
        row, daily, monthly = summarize(result, run_id=run_id, strategy=algorithm, seed=seed, scenario=scenario,
                                       split="test", start=TEST[0], end=TEST[1], capital=10000., **COSTS[scenario])
        row.update(symbol=symbol, algorithm=algorithm, recipe_id=recipe_id, cadence=cadence,
                   account_id=account_id, primary_seed=seed == 0, ensemble=False, independent_capital=True,
                   evaluation_directory=directory.relative_to(output).as_posix().replace("/repeat/", "/evaluation/"))
        write(directory / "actions.json", actions, exclusive=True)
        write(directory / "swaps.json", swaps, exclusive=True)
        write(directory / "summary.json", row, exclusive=True)
        (directory / "daily_equity.csv").write_bytes(canonical_csv(daily))
        (directory / "monthly_returns.csv").write_bytes(canonical_csv(monthly))
        dirs.append(directory)
    for name in RESULT_FILES:
        if sha(dirs[0] / name) != sha(dirs[1] / name):
            raise ValueError("Repeated account bytes differ: " + name)
    answer = dict(summary=read(dirs[0] / "summary.json"), directory=dirs[0].relative_to(output).as_posix(),
                  files_sha256={name: sha(dirs[0] / name) for name in RESULT_FILES})
    write(receipt_path, dict(at=now(), account_id=account_id, protocol_sha256=sha(output / "protocol.json"),
          directory=answer["directory"], files_sha256=answer["files_sha256"]), exclusive=True)
    return answer


def evaluate(output, dataset, max_workers=None):
    from scripts.crypto_rl_reporting import canonical_csv
    output = Path(output)
    spec = check_protocol(output, dataset)
    if not (output / "fit_complete.json").exists():
        raise ValueError("Full model grid required before annual accounts")
    tasks = [dict(output=str(output), dataset=str(dataset), symbol=symbol, recipe_id=recipe_id,
                  seed=seed, cadence=cadence, scenario=scenario)
             for symbol in spec["selected"] for recipe_id in spec["selected"][symbol]
             for seed in spec["seeds"] for cadence in spec["cadences"] for scenario in spec["costs"]]
    rows, failures = [], []
    workers = min(len(tasks), max_workers or max(1, len(os.sched_getaffinity(0)) - 2))
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"), initializer=worker_init) as pool:
        futures = {pool.submit(evaluate_case, task): task for task in tasks}
        for future in as_completed(futures):
            try:
                rows.append(future.result())
            except Exception as error:
                failures.append(dict(task=futures[future], error=str(error), type=type(error).__name__))
            write(output / "evaluation_status.json", dict(at=now(), completed=len(rows), failures=failures, total=len(tasks)))
    if failures:
        raise RuntimeError("Account failures; preserve every attempt")
    rows.sort(key=lambda r: r["summary"]["account_id"])
    (output / "summary.csv").write_bytes(canonical_csv(pd.DataFrame([r["summary"] for r in rows])))
    manifest = dict(accounts=[dict(**{k: r["summary"][k] for k in
          ("account_id", "symbol", "recipe_id", "algorithm", "seed", "cadence", "scenario")},
          directory=r["directory"], files_sha256=r["files_sha256"]) for r in rows])
    if (output / "accounts_manifest.json").exists():
        if read(output / "accounts_manifest.json") != manifest:
            raise ValueError("Existing full account manifest differs")
    else:
        write(output / "accounts_manifest.json", manifest, exclusive=True)
    marker = output / "evaluation_complete.json"
    if marker.exists():
        if read(marker).get("accounts") != 180 or read(marker).get("summary_sha256") != sha(output / "summary.csv"):
            raise ValueError("Existing evaluation completion marker differs")
    else:
        write(marker, dict(at=now(), accounts=len(rows), summary_sha256=sha(output / "summary.csv")), exclusive=True)


def pipeline(output, dataset, max_workers=None):
    output = Path(output)
    from scripts.audit_rolling_single_pair import audit_study
    try:
        for phase, call in (("fit", lambda: fit(output, dataset, max_workers)),
                            ("evaluation", lambda: evaluate(output, dataset)),
                            ("audit", lambda: audit_study(output, dataset))):
            write(output / "pipeline_status.json", dict(at=now(), status="running", phase=phase, pid=os.getpid()))
            call()
        if read(output / "completion_gate.json").get("passed") is not True:
            raise ValueError("All checks must pass before completion")
        write(output / "pipeline_status.json", dict(at=now(), status="completed", phase="complete", pid=os.getpid(), no_production_change=True))
    except Exception as error:
        write(output / "pipeline_status.json", dict(at=now(), status="failed", pid=os.getpid(), type=type(error).__name__, error=str(error)))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "preflight", "fit", "evaluate", "pipeline"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--max-workers", type=int)
    args = parser.parse_args()
    if args.phase == "prepare":
        print(json.dumps(prepare(args.output, args.dataset)))
    elif args.phase == "preflight":
        preflight(args.output, args.dataset)
    elif args.phase == "fit":
        fit(args.output, args.dataset, args.max_workers)
    elif args.phase == "evaluate":
        evaluate(args.output, args.dataset, args.max_workers)
    else:
        pipeline(args.output, args.dataset, args.max_workers)


if __name__ == "__main__":
    main()

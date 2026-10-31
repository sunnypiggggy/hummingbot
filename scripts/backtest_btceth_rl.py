#!/usr/bin/env python3
"""Frozen, GPU-isolated BTC/ETH spot RL experiment; no exchange credentials.

Phases deliberately separate pre-test learning/selection from the already-seen
365-day research test. Existing Top10 studies and their financial code remain
untouched. Run ``prepare``, copy the frozen bundle, then ``fit`` on two GPUs.
Only ``test`` can read test rewards, after validation_selection.json is frozen.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.metadata
import json
import multiprocessing as mp
import os
from pathlib import Path
import pickle
import random
import subprocess
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.crypto_rl_reporting import (canonical_csv, result_fingerprints,
                                         save_result, summarize, write_json)

TRAIN = ("2020-01-01T00:00:00Z", "2025-01-01T00:00:00Z")
VALIDATION = (TRAIN[1], "2025-10-23T00:00:00Z")
TEST = (VALIDATION[1], "2026-10-23T00:00:00Z")
SEEDS = (0, 1, 2, 3, 4)
EPISODE_DAYS = 32
ALGORITHMS = ("recurrent_ppo", "tqc", "crossq", "sac")
PROFILES = {"return": {"drawdown_penalty": 0., "rebalance_band": .025},
            "balanced": {"drawdown_penalty": .25, "rebalance_band": .025}}
COSTS = {"base": {"fee": .001, "slippage": .0002},
         "stress": {"fee": .002, "slippage": .0005}}
BASELINES = ("cash", "btc_hold", "eth_hold", "equal_hold",
             "equal_rebalance", "sma200_trend", "momentum180")
SOURCE_FILES = ("scripts/backtest_btceth_rl.py", "scripts/btceth_rl_env.py",
                "scripts/crypto_rl_portfolio_env.py", "scripts/crypto_rl_reporting.py")
RUNTIME_PACKAGES = {"torch": "2.14.0+cu126", "numpy": "2.4.6", "pandas": "3.0.3",
                    "gymnasium": "1.2.3", "stable-baselines3": "2.9.0", "sb3-contrib": "2.9.0"}


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_dataset(path):
    path = Path(path)
    manifest = read_json(path.parent / "manifest.json")
    if sha(path) != manifest["dataset_sha256"]:
        raise ValueError("BTC/ETH prepared data hash does not match its manifest")
    with gzip.open(path, "rb") as stream:
        data = pickle.load(stream)
    for field in ("open", "high", "low", "close", "volume"):
        if list(data["hourly_" + field].columns) != ["BTCUSDT", "ETHUSDT"]:
            raise ValueError("Expected fixed BTCUSDT, ETHUSDT identity order")
    data["manifest"] = manifest
    return data


def parameter_digest(policy):
    digest = hashlib.sha256()
    for name, tensor in sorted(policy.state_dict().items()):
        value = tensor.detach().cpu().numpy()
        digest.update(name.encode())
        digest.update(str((value.shape, value.dtype)).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def model_class(algorithm):
    from stable_baselines3 import SAC
    from sb3_contrib import CrossQ, RecurrentPPO, TQC
    return {"recurrent_ppo": RecurrentPPO, "tqc": TQC,
            "crossq": CrossQ, "sac": SAC}[algorithm]


def hyperparameters(algorithm):
    common = dict(learning_rate=.0003, gamma=.995)
    if algorithm == "recurrent_ppo":
        return dict(**common, policy="MlpLstmPolicy", n_steps=128,
                    batch_size=128, n_epochs=5, gae_lambda=.95,
                    clip_range=.2, ent_coef=.001,
                    policy_kwargs=dict(lstm_hidden_size=64, n_lstm_layers=1,
                                       net_arch=dict(pi=[64], vf=[64])))
    common.update(policy="MlpPolicy", buffer_size=100000, learning_starts=512,
                  batch_size=128, tau=.005, train_freq=1, gradient_steps=1,
                  ent_coef="auto", policy_kwargs=dict(net_arch=[128, 128]))
    if algorithm == "tqc":
        common["policy_kwargs"].update(n_critics=2, n_quantiles=25)
        common["top_quantiles_to_drop_per_net"] = 2
    if algorithm == "crossq":
        # CrossQ removes target networks; unlike SAC/TQC it has no tau.
        common.pop("tau")
        common["policy_delay"] = 3
    return common


def make_spec(dataset_path, timesteps):
    return {
        "study": "BTCUSDT / ETHUSDT / USDT, new RL allocation implementations",
        "frozen_at": now(), "scope": "research only; no production or economic requests",
        "train": list(TRAIN), "validation": list(VALIDATION), "test": list(TEST),
        "dataset_sha256": sha(dataset_path),
        "dataset_manifest_sha256": sha(Path(dataset_path).parent / "manifest.json"),
        "code_sha256": {name: sha(ROOT / name) for name in SOURCE_FILES},
        "algorithms": list(ALGORITHMS), "profiles": PROFILES, "seeds": list(SEEDS),
        "timesteps_per_model": timesteps,
        "capital": 10000., "cash_floor_target": .05, "single_coin_cap_target": .95,
        "costs": COSTS, "train_episode_days": EPISODE_DAYS,
        "action": "gross=.95*max(action[0],0); ETH share=(action[1]+1)/2; residual USDT",
        "decision": "daily 00UTC; only previous completed daily/hourly data",
        "execution": "00UTC hour-open sells; 01UTC hour-open buys, each side taker",
        "valuation": "all hourly closes; timestamp labels denote hour opens",
        "reward": "100*(log(after-cost daily NAV ratio)-lambda*incremental intraday drawdown)",
        "hyperparameters": {a: hyperparameters(a) for a in ALGORITHMS},
        "runtime_packages": RUNTIME_PACKAGES,
        "baseline_names": list(BASELINES),
        "validation_selection": {
            "unit": "one profile per algorithm; one overall validation leader, all five seeds retained",
            "primary_policy": "mean of five target-weight vectors; execute one funded ledger",
            "score": "mean(base,stress)[log(final/initial)-0.5*abs(maxDD)-2*max(abs(maxDD)-.20,0)]",
            "tie_break": "lexicographic algorithm/profile identifier; never a test result",
            "seed_selection": False, "test_hyperparameter_selection": False,
        },
        "research_targets": {
            "primary_base_net_return_at_least": .05,
            "primary_base_max_drawdown_magnitude_at_most": .20,
            "primary_stress_net_return_at_least": 0.,
            "primary_beats_equal_hold_return_and_drawdown": True,
            "positive_base_training_seeds_at_least": 3,
            "stronger_claim_requires_beating_simple_trend_controls": True,
            "targets_are_not_guarantees": True,
        },
        "known_limitations": [
            "The same test year was already examined in earlier research; this is not a fresh blind holdout.",
            "Historical candle replay, assumed fees/slippage, fractional units; no historical queue/lot filters.",
            "Different random seeds repeat one market history; they are not independent market samples.",
            "Daily/hours execution matches previous RL, not the earlier minute-execution strategy suite.",
            "Two-coin action space and 95% single-coin cap differ from old dynamic-Top10 25% cap.",
        ],
    }


def check_frozen(output, dataset_path):
    spec_path = Path(output) / "frozen_spec.json"
    spec = read_json(spec_path)
    if spec["train"] != list(TRAIN) or spec["validation"] != list(VALIDATION) or spec["test"] != list(TEST):
        raise ValueError("Split boundaries changed")
    if spec["algorithms"] != list(ALGORITHMS) or spec["profiles"] != PROFILES or spec["seeds"] != list(SEEDS):
        raise ValueError("Declared algorithms, profiles, or seeds changed")
    if spec["costs"] != COSTS or spec["hyperparameters"] != {a: hyperparameters(a) for a in ALGORITHMS}:
        raise ValueError("Frozen financial or learning parameters changed")
    if spec["train_episode_days"] != EPISODE_DAYS:
        raise ValueError("Frozen training episode length changed")
    if sha(dataset_path) != spec["dataset_sha256"] or sha(Path(dataset_path).parent / "manifest.json") != spec["dataset_manifest_sha256"]:
        raise ValueError("Frozen market data changed")
    for name, expected in spec["code_sha256"].items():
        if sha(ROOT / name) != expected:
            raise ValueError("Frozen study source changed: " + name)
    return spec


def gpu_initializer(uuid):
    os.environ["CUDA_VISIBLE_DEVICES"] = uuid
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"


def torch_setup(seed):
    import torch
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Exactly one UUID-isolated CUDA GPU is required; no CPU fallback")
    torch.cuda.manual_seed_all(seed)
    a = torch.ones((32, 32), device="cuda:0")
    if float((a @ a)[0, 0].item()) != 32.:
        raise RuntimeError("GPU kernel self-check failed")
    return {"visible_gpu_uuid": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "device": "cuda:0", "name": torch.cuda.get_device_name(0),
            "capability": list(torch.cuda.get_device_capability(0)),
            "torch": torch.__version__, "cuda_runtime": torch.version.cuda}


def job_id(algorithm, profile, seed):
    return f"{algorithm}__{profile}__seed{seed}"


def train_one(task):
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.monitor import Monitor
    from scripts.btceth_rl_env import BTCETHPortfolioEnv

    output, data_path = Path(task["output"]), Path(task["dataset"])
    spec = check_frozen(output, data_path)
    algorithm, profile, seed = task["algorithm"], task["profile"], int(task["seed"])
    name = job_id(algorithm, profile, seed)
    directory = output / "training" / name
    if directory.exists():
        raise FileExistsError("Refuse to overwrite an attempted training job: " + str(directory))
    directory.mkdir(parents=True)
    device = torch_setup(seed)
    started = time.monotonic()
    write_json(directory / "started.json", dict(at=now(), pid=os.getpid(), task=task, device=device,
                                               frozen_spec_sha256=sha(output / "frozen_spec.json")))
    data = load_dataset(data_path)
    raw_env = BTCETHPortfolioEnv(data, *TRAIN, costs=COSTS["base"], training=True,
                                episode_days=spec["train_episode_days"], record_history=False, **PROFILES[profile])
    env = Monitor(raw_env)
    progress = []

    class Progress(BaseCallback):
        def _on_step(self):
            if self.num_timesteps % 2048 == 0:
                row = {"timesteps": int(self.num_timesteps), "elapsed_seconds": time.monotonic() - started}
                row.update({k: float(v) for k, v in self.model.logger.name_to_value.items()
                            if k.startswith("train/") and isinstance(v, (int, float, np.number)) and np.isfinite(v)})
                progress.append(row)
                write_json(directory / "progress.json", dict(at=now(), pid=os.getpid(), device=device, **row))
                print(f"TRAIN {name} steps={self.num_timesteps} elapsed={row['elapsed_seconds']:.1f}s", flush=True)
            return True

    try:
        cls = model_class(algorithm)
        params = dict(spec["hyperparameters"][algorithm])
        policy = params.pop("policy")
        model = cls(policy, env, seed=seed, device="cuda:0", verbose=0, **params)
        model.learn(total_timesteps=spec["timesteps_per_model"], callback=Progress())
        checkpoint = directory / "model.zip"
        model.save(checkpoint)
        record = {"algorithm": algorithm, "profile": profile, "seed": seed,
                  "completed_at": now(), "elapsed_seconds": time.monotonic() - started,
                  "model_relative_path": str(checkpoint.relative_to(output)).replace("\\", "/"),
                  "model_sha256": sha(checkpoint), "parameter_sha256": parameter_digest(model.policy),
                  "actual_timesteps": int(model.num_timesteps), "device": device,
                  "frozen_spec_sha256": sha(output / "frozen_spec.json"),
                  "training_episode_history": list(getattr(raw_env, "training_episode_history", [])),
                  "training_window_quality": getattr(raw_env, "training_window_metadata", {}),
                  "features": list(getattr(raw_env, "FEATURE_NAMES", []))}
        write_json(directory / "record.json", record)
        pd.DataFrame(progress).to_csv(directory / "training_metrics.csv", index=False)
        print(f"COMPLETE {name} elapsed={record['elapsed_seconds']:.1f}s", flush=True)
        return record
    except BaseException as error:
        write_json(directory / "failure.json", dict(at=now(), error_type=type(error).__name__, error=str(error),
                                                   pid=os.getpid(), device=device))
        raise
    finally:
        env.close()


def gpu_inventory():
    command = ["nvidia-smi", "--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu",
               "--format=csv,noheader,nounits"]
    rows = []
    for line in subprocess.check_output(command, text=True).strip().splitlines():
        index, uuid, name, total, used, utilization = [s.strip() for s in line.split(",")]
        rows.append(dict(index=int(index), uuid=uuid, name=name, memory_total_mib=int(total),
                         memory_used_mib=int(used), utilization_percent=int(utilization)))
    if len(rows) != 2 or any("V100" not in row["name"] for row in rows):
        raise RuntimeError("This protocol requires the two previously authorized V100 devices")
    return sorted(rows, key=lambda row: row["index"])


def runtime_packages():
    actual = {name: importlib.metadata.version(name) for name in RUNTIME_PACKAGES}
    if actual != RUNTIME_PACKAGES:
        raise RuntimeError("Formal GPU experiment dependency versions differ from the preflight pins: " + str(actual))
    return actual


def train_all(output, data_path, *, resume=False):
    spec = check_frozen(output, data_path)
    inventory = gpu_inventory()
    packages = runtime_packages()
    write_json(output / "runtime.json", dict(at=now(), python=sys.version, packages=packages,
                                            physical_gpus=inventory, training_pid=os.getpid()))
    tasks = [dict(output=str(output), dataset=str(data_path), algorithm=a, profile=p, seed=s)
             for a in ALGORITHMS for p in PROFILES for s in SEEDS]
    records, pending = [], []
    for i, task in enumerate(tasks):
        path = output / "training" / job_id(task["algorithm"], task["profile"], task["seed"])
        record_path = path / "record.json"
        if path.exists():
            if not resume or not record_path.is_file():
                raise FileExistsError("Existing/incomplete job must be investigated, never silently restarted: " + str(path))
            record = read_json(record_path)
            if sha(output / record["model_relative_path"]) != record["model_sha256"]:
                raise ValueError("Completed checkpoint hash changed")
            if record["frozen_spec_sha256"] != sha(output / "frozen_spec.json"):
                raise ValueError("Completed job belongs to another frozen protocol")
            records.append(record)
        else:
            pending.append((i % 2, task))
    pools = [ProcessPoolExecutor(max_workers=1, mp_context=mp.get_context("spawn"),
                                 initializer=gpu_initializer, initargs=(gpu["uuid"],)) for gpu in inventory]
    try:
        futures = {pools[slot].submit(train_one, task): task for slot, task in pending}
        for future in as_completed(futures):
            record = future.result()
            records.append(record)
            write_json(output / "training_status.json", dict(at=now(), completed=len(records), total=len(tasks),
                                                              latest=job_id(record["algorithm"], record["profile"], record["seed"])))
    finally:
        for pool in pools:
            pool.shutdown(wait=True, cancel_futures=True)
    if len(records) != len(tasks):
        raise AssertionError("Incomplete formal training grid")
    records.sort(key=lambda r: (r["algorithm"], r["profile"], r["seed"]))
    write_json(output / "training_completion.json", dict(completed_at=now(), model_count=len(records),
                                                          records=records, frozen_spec_sha256=sha(output / "frozen_spec.json")))


def baseline_signal_table(dataset, kind):
    """Causal daily controls, built once per evaluation rather than per step."""
    hourly = dataset["hourly_close"]
    close = hourly.loc[hourly.index.hour == 23].copy()
    close.index = close.index.normalize()
    if kind == "sma200_trend":
        signal = close > close.rolling(200, min_periods=200).mean()
    elif kind == "momentum180":
        signal = close > close.shift(180)
    else:
        return None
    signal.index += pd.Timedelta(days=1)
    return signal


def baseline_weights(dataset, day, kind, prepared_signals=None):
    if kind == "cash":
        return np.array([0., 0., 1.])
    if kind == "btc_hold":
        return np.array([.95, 0., .05])
    if kind == "eth_hold":
        return np.array([0., .95, .05])
    if kind in ("equal_hold", "equal_rebalance"):
        return np.array([.475, .475, .05])
    day = pd.Timestamp(day)
    if kind not in ("sma200_trend", "momentum180"):
        raise ValueError("Unknown baseline: " + kind)
    signals = prepared_signals if prepared_signals is not None else baseline_signal_table(dataset, kind)
    signal = signals.loc[day]
    active = signal.fillna(False).to_numpy(dtype=bool)
    weights = np.zeros(3)
    if active.any():
        weights[:2] = .95 * active / active.sum()
    weights[2] = 1 - weights[:2].sum()
    return weights


def evaluate(dataset, models, algorithm, start, end, costs, profile, *, baseline=None):
    from scripts.btceth_rl_env import BTCETHPortfolioEnv
    env = BTCETHPortfolioEnv(dataset, start, end, costs=costs, record_history=True,
                            **PROFILES[profile])
    obs, _ = env.reset(seed=0)
    states = [None] * len(models)
    episode_starts = np.ones((1,), dtype=bool)
    before = [parameter_digest(model.policy) for model in models]
    baseline_signals = baseline_signal_table(dataset, baseline) if baseline else None
    done, initial = False, True
    while not done:
        if models:
            targets = []
            for index, model in enumerate(models):
                if algorithm == "recurrent_ppo":
                    action, states[index] = model.predict(obs, state=states[index],
                                                         episode_start=episode_starts, deterministic=True)
                else:
                    action, _ = model.predict(obs, deterministic=True)
                targets.append(env.action_to_weights(action))
            obs, _, terminated, truncated, _ = env.step_target(np.mean(targets, axis=0))
            episode_starts[:] = False
        elif baseline.endswith("_hold") and not initial:
            obs, _, terminated, truncated, _ = env.step_hold()
        else:
            target = baseline_weights(dataset, env.current_date, baseline, baseline_signals)
            obs, _, terminated, truncated, _ = env.step_target(target)
        done, initial = bool(terminated or truncated), False
    after = [parameter_digest(model.policy) for model in models]
    if before != after:
        raise AssertionError("Evaluation modified a trained policy")
    result = env.get_result()
    result["metadata"].update(algorithm=algorithm, profile=profile, baseline=baseline,
                               model_count=len(models), parameter_sha256_before=before,
                               parameter_sha256_after=after,
                               ensemble="mean target weights, one actual funded ledger" if len(models) > 1 else None)
    env.close()
    return result


def validation_score(base, stress):
    def one(row):
        drawdown = abs(float(row["max_drawdown"]))
        return float(np.log1p(float(row["net_return"])) - .5 * drawdown - 2 * max(drawdown - .20, 0.))
    return (one(base) + one(stress)) / 2


def evaluate_group(task):
    import torch
    output, data_path = Path(task["output"]), Path(task["dataset"])
    spec = check_frozen(output, data_path)
    torch_setup(123456)
    algorithm, profile, split = task["algorithm"], task["profile"], task["split"]
    if split == "test":
        marker = read_json(output / "test_started.json")
        if marker["validation_selection_sha256"] != sha(output / "validation_selection.json"):
            raise ValueError("Frozen selection changed after test dispatch")
    start, end = VALIDATION if split == "validation" else TEST
    dataset = load_dataset(data_path)
    directory = output / "evaluation" / split / algorithm / profile
    if directory.exists():
        raise FileExistsError("Preserve an existing evaluation attempt: " + str(directory))
    directory.mkdir(parents=True)
    records = [read_json(output / "training" / job_id(algorithm, profile, seed) / "record.json") for seed in SEEDS]
    models = []
    for record in records:
        path = output / record["model_relative_path"]
        if sha(path) != record["model_sha256"]:
            raise ValueError("Checkpoint changed")
        model = model_class(algorithm).load(path, device="cuda:0")
        if parameter_digest(model.policy) != record["parameter_sha256"]:
            raise ValueError("Loaded checkpoint parameters do not match training record")
        models.append(model)
    rows, history, months, repeats = [], [], [], []
    for scenario, costs in COSTS.items():
        plans = [(str(seed), [model], seed) for seed, model in zip(SEEDS, models)] + [("ensemble", models, None)]
        for key, selected, seed in plans:
            run_id = f"{split}__{scenario}__{algorithm}__{profile}__{key}"
            result = evaluate(dataset, selected, algorithm, start, end, costs, profile)
            fingerprint = save_result(directory / scenario / key, result)
            repeat = evaluate(dataset, selected, algorithm, start, end, costs, profile)
            repeated = result_fingerprints(repeat)
            if fingerprint != repeated:
                raise AssertionError("Deterministic policy replay mismatch: " + run_id)
            row, daily, monthly = summarize(result, run_id=run_id, strategy=algorithm, seed=seed,
                                             scenario=scenario, split=split, start=start, end=end,
                                             capital=10000., **costs)
            row.update(profile=profile, algorithm=algorithm, primary_policy=seed is None)
            daily["profile"], monthly["profile"] = profile, profile
            rows.append(row)
            history.append(daily)
            months.append(monthly)
            repeats.append(dict(run_id=run_id, passed=True, fingerprints=fingerprint))
            print(f"EVAL {run_id} return={row['net_return']:.4%} dd={row['max_drawdown']:.4%}", flush=True)
    pd.DataFrame(rows).to_csv(directory / "summary.csv", index=False, float_format="%.17g")
    pd.concat(history, ignore_index=True).to_csv(directory / "daily_equity.csv", index=False, float_format="%.17g")
    pd.concat(months, ignore_index=True).to_csv(directory / "monthly_returns.csv", index=False, float_format="%.17g")
    write_json(directory / "completion.json", dict(at=now(), repeats=repeats, rows=rows,
                                                   frozen_spec_sha256=sha(output / "frozen_spec.json")))
    del models
    torch.cuda.empty_cache()
    return dict(algorithm=algorithm, profile=profile, split=split, rows=rows)


def evaluate_baselines(output, data_path, split):
    dataset = load_dataset(data_path)
    start, end = VALIDATION if split == "validation" else TEST
    directory = output / "evaluation" / split / "baselines"
    if directory.exists():
        raise FileExistsError("Preserve an existing baseline evaluation: " + str(directory))
    directory.mkdir(parents=True)
    rows, histories, months = [], [], []
    for scenario, costs in COSTS.items():
        for name in BASELINES:
            run_id = f"{split}__{scenario}__{name}"
            result = evaluate(dataset, [], "baseline", start, end, costs, "return", baseline=name)
            fingerprints = save_result(directory / scenario / name, result)
            repeat = evaluate(dataset, [], "baseline", start, end, costs, "return", baseline=name)
            if result_fingerprints(repeat) != fingerprints:
                raise AssertionError("Baseline repeat mismatch")
            row, history, monthly = summarize(result, run_id=run_id, strategy=name, scenario=scenario,
                                               split=split, start=start, end=end, capital=10000., **costs)
            row.update(profile="baseline", algorithm="baseline", primary_policy=False)
            history["profile"], monthly["profile"] = "baseline", "baseline"
            rows.append(row)
            histories.append(history)
            months.append(monthly)
    pd.DataFrame(rows).to_csv(directory / "summary.csv", index=False, float_format="%.17g")
    pd.concat(histories, ignore_index=True).to_csv(directory / "daily_equity.csv", index=False, float_format="%.17g")
    pd.concat(months, ignore_index=True).to_csv(directory / "monthly_returns.csv", index=False, float_format="%.17g")
    write_json(directory / "completion.json", dict(at=now(), rows=rows))


def training_source_hashes(output):
    completion = output / "training_completion.json"
    manifest = read_json(completion)
    expected_count = len(ALGORITHMS) * len(PROFILES) * len(SEEDS)
    if manifest["model_count"] != expected_count:
        raise ValueError("Selection requires every declared model to complete")
    files = {"training_completion.json": sha(completion)}
    for algorithm in ALGORITHMS:
        for profile in PROFILES:
            for seed in SEEDS:
                record_path = output / "training" / job_id(algorithm, profile, seed) / "record.json"
                record = read_json(record_path)
                if (record["algorithm"], record["profile"], record["seed"]) != (algorithm, profile, seed):
                    raise ValueError("Training record identity mismatch")
                checkpoint = (output / record["model_relative_path"]).resolve()
                checkpoint.relative_to(output.resolve())
                if sha(checkpoint) != record["model_sha256"]:
                    raise ValueError("Training model/record no longer reconcile")
                files[str(record_path.relative_to(output)).replace("\\", "/")] = sha(record_path)
                files[str(checkpoint.relative_to(output.resolve())).replace("\\", "/")] = sha(checkpoint)
    return files


def select_validation(output, *, allow_existing_test=False):
    if not allow_existing_test and any(path.exists() for path in (
            output / "test_started.json", output / "test_completion.json", output / "evaluation" / "test")):
        raise ValueError("Validation selection must precede every test attempt")
    choices, candidates, files = {}, [], {}
    for algorithm in ALGORITHMS:
        scores = []
        for profile in PROFILES:
            path = output / "evaluation" / "validation" / algorithm / profile / "summary.csv"
            data = pd.read_csv(path, float_precision="round_trip")
            selected = data.loc[data.primary_policy.eq(True)]
            if len(data) != 12 or len(selected) != 2 or set(data.seed.dropna().astype(int)) != set(SEEDS):
                raise ValueError("Validation requires both costs and all five training seeds")
            expected_identity = dict(split="validation", algorithm=algorithm, profile=profile,
                                     start=pd.Timestamp(VALIDATION[0]).isoformat(),
                                     end_exclusive=pd.Timestamp(VALIDATION[1]).isoformat(), initial_capital=10000.)
            for key, value in expected_identity.items():
                if key not in data or not data[key].eq(value).all():
                    raise ValueError("Validation summary identity mismatch: " + key)
            for scenario in COSTS:
                scenario_data = data.loc[data.scenario.eq(scenario)]
                if (len(scenario_data) != 6 or scenario_data.seed.dropna().duplicated().any()
                        or set(scenario_data.seed.dropna().astype(int)) != set(SEEDS)
                        or int(scenario_data.primary_policy.eq(True).sum()) != 1):
                    raise ValueError("Each cost scenario must contain exactly five seeds and one actual ensemble")
            base = selected.loc[selected.scenario.eq("base")].iloc[0].to_dict()
            stress = selected.loc[selected.scenario.eq("stress")].iloc[0].to_dict()
            score = validation_score(base, stress)
            candidate = dict(algorithm=algorithm, profile=profile, score=score, base=base, stress=stress)
            scores.append(candidate)
            candidates.append(candidate)
            files[str(path.relative_to(output)).replace("\\", "/")] = sha(path)
        winner = sorted(scores, key=lambda r: (-r["score"], r["profile"]))[0]
        choices[algorithm] = winner["profile"]
    overall = sorted(candidates, key=lambda r: (-r["score"], r["algorithm"], r["profile"]))[0]
    return dict(frozen_at=now(), frozen_spec_sha256=sha(output / "frozen_spec.json"),
                selected_profiles=choices, primary_validation_leader=dict(algorithm=overall["algorithm"], profile=overall["profile"]),
                candidates=candidates, validation_source_sha256=files,
                training_source_sha256=training_source_hashes(output),
                test_results_available_to_selector=False, selection_uses_test=False)


def evaluate_all(output, data_path, split):
    check_frozen(output, data_path)
    runtime_packages()
    inventory = gpu_inventory()
    if not (output / "training_completion.json").is_file():
        raise ValueError("All 40 declared training models must finish first")
    if split == "validation":
        plans = [(a, p) for a in ALGORITHMS for p in PROFILES]
    else:
        selection_path = output / "validation_selection.json"
        selection = read_json(selection_path)
        if selection["frozen_spec_sha256"] != sha(output / "frozen_spec.json"):
            raise ValueError("Selection does not match protocol")
        for name, expected in selection["validation_source_sha256"].items():
            if sha(output / name) != expected:
                raise ValueError("Frozen validation evidence changed")
        if training_source_hashes(output) != selection["training_source_sha256"]:
            raise ValueError("Test models differ from the validation-frozen training artifacts")
        recomputed = select_validation(output)
        for key in ("selected_profiles", "primary_validation_leader", "validation_source_sha256", "training_source_sha256"):
            if recomputed[key] != selection[key]:
                raise ValueError("Selection does not reproduce from frozen validation evidence: " + key)
        plans = [(a, selection["selected_profiles"][a]) for a in ALGORITHMS]
        lock = output / "test_started.json"
        if lock.exists():
            raise FileExistsError("Test already attempted; preserve results and diagnose explicitly")
        marker = dict(at=now(), validation_selection_sha256=sha(selection_path),
                      frozen_spec_sha256=sha(output / "frozen_spec.json"))
        with lock.open("x", encoding="utf-8") as stream:
            json.dump(marker, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
    pools = [ProcessPoolExecutor(max_workers=1, mp_context=mp.get_context("spawn"),
                                 initializer=gpu_initializer, initargs=(gpu["uuid"],)) for gpu in inventory]
    try:
        futures = [pools[i % 2].submit(evaluate_group, dict(output=str(output), dataset=str(data_path),
                                                         algorithm=a, profile=p, split=split)) for i, (a, p) in enumerate(plans)]
        for future in as_completed(futures):
            future.result()
    finally:
        for pool in pools:
            pool.shutdown(wait=True, cancel_futures=True)
    evaluate_baselines(output, data_path, split)
    if split == "validation":
        path = output / "validation_selection.json"
        if path.exists():
            raise FileExistsError("Never replace a frozen validation selection")
        write_json(path, select_validation(output))
    write_json(output / f"{split}_completion.json", dict(at=now(), model_groups=len(plans),
                                                         baselines=len(BASELINES), frozen_spec_sha256=sha(output / "frozen_spec.json")))
    assemble(output)


def assemble(output):
    for name in ("summary.csv", "daily_equity.csv", "monthly_returns.csv"):
        paths = sorted((output / "evaluation").glob(f"*/*/*/{name}"))
        paths.extend(sorted((output / "evaluation").glob(f"*/baselines/{name}")))
        frames = [pd.read_csv(path, float_precision="round_trip") for path in paths]
        if not frames:
            continue
        frame = pd.concat(frames, ignore_index=True)
        if name == "summary.csv" and frame.run_id.duplicated().any():
            raise AssertionError("Duplicate financial run identity")
        (output / name).write_bytes(canonical_csv(frame))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=("prepare", "train", "validate", "fit", "test", "assemble"))
    parser.add_argument("--dataset", type=Path, default=ROOT / "data/crypto_rl_btceth_20261029/dataset.pkl.gz")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/backtests/btceth_rl_20261029")
    parser.add_argument("--timesteps", type=int, default=32768)
    parser.add_argument("--resume-completed-training", action="store_true")
    args = parser.parse_args()
    output, data_path = args.output_dir.resolve(), args.dataset.resolve()
    if args.phase == "prepare":
        if output.exists():
            raise FileExistsError("Freeze only into a new experiment directory")
        if args.timesteps < 8192 or args.timesteps % 128:
            parser.error("timesteps must be >=8192 and a multiple of128")
        load_dataset(data_path)
        output.mkdir(parents=True)
        write_json(output / "frozen_spec.json", make_spec(data_path, args.timesteps))
        print("FROZEN " + str(output / "frozen_spec.json"), flush=True)
        return
    check_frozen(output, data_path)
    if args.phase in ("train", "fit"):
        train_all(output, data_path, resume=args.resume_completed_training)
    if args.phase in ("validate", "fit"):
        evaluate_all(output, data_path, "validation")
    if args.phase == "test":
        evaluate_all(output, data_path, "test")
    if args.phase == "assemble":
        assemble(output)


if __name__ == "__main__":
    main()

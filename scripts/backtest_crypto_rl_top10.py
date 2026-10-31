#!/usr/bin/env python3
"""Train a bounded, reproducible PPO Top10 portfolio study entirely locally.

Use the dedicated research Python environment. Historical data is prepared by
crypto_rl_data.py. No connector, account key, live order, or trading config is
read. Test rewards never update a model or select a seed.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.metadata
import json
from pathlib import Path
import pickle
import platform
import random
import subprocess
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.crypto_rl_portfolio_env import CryptoPortfolioEnv
from scripts.crypto_rl_reporting import (
    make_reviewed_snapshot, result_fingerprints, save_result, seed_statistics,
    summarize, write_json,
)

TRAIN_START, VAL_START = "2023-01-01T00:00:00Z", "2025-07-25T00:00:00Z"
TEST_START, TEST_END = "2025-10-23T00:00:00Z", "2026-10-23T00:00:00Z"
COSTS = {"base": {"fee": .001, "slippage": .0002}, "stress": {"fee": .002, "slippage": .0005}}
SEEDS = (0, 1, 2, 3, 4)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_dataset(path):
    path = Path(path)
    metadata = json.loads((path.parent / "manifest.json").read_text(encoding="utf-8"))
    if sha(path) != metadata["dataset_sha256"]:
        raise AssertionError("Prepared dataset no longer matches its source manifest")
    with gzip.open(path, "rb") as stream:
        dataset = pickle.load(stream)
    dataset["manifest"] = metadata
    return dataset


def parameter_digest(policy):
    digest = hashlib.sha256()
    for name, tensor in sorted(policy.state_dict().items()):
        value = tensor.detach().cpu().numpy()
        digest.update(name.encode("utf-8"))
        digest.update(str((value.shape, value.dtype)).encode("utf-8"))
        digest.update(value.tobytes())
    return digest.hexdigest()


def train_seed(task):
    """Separate process, a single CPU math thread and a predeclared seed."""
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback

    seed, output = int(task["seed"]), Path(task["out"])
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    dataset = load_dataset(task["dataset"])
    env = CryptoPortfolioEnv(dataset, TRAIN_START, VAL_START, costs=COSTS["base"],
                             training=True, episode_days=128)
    rows, started = [], time.monotonic()

    class Progress(BaseCallback):
        def _on_step(self):
            return True

        def _on_rollout_end(self):
            episodes = list(self.model.ep_info_buffer or [])
            row = {"seed": seed, "timesteps": self.num_timesteps,
                   "elapsed_seconds": time.monotonic() - started,
                   "training_episode_log_reward_mean": float(np.mean([e["r"] for e in episodes])) if episodes else None}
            row.update({k: float(v) for k, v in self.model.logger.name_to_value.items()
                        if k.startswith("train/") and isinstance(v, (float, int, np.number))})
            rows.append(row)
            if self.num_timesteps % 4096 == 0:
                print(f"TRAIN seed={seed} steps={self.num_timesteps}/{task['timesteps']}", flush=True)

    model = PPO("MlpPolicy", env, device="cpu", seed=seed, verbose=0,
                learning_rate=3e-4, n_steps=256, batch_size=64, n_epochs=10,
                gamma=.99, gae_lambda=.95, clip_range=.2, ent_coef=0.,
                policy_kwargs={"net_arch": [32, 32]})
    model.learn(total_timesteps=int(task["timesteps"]), callback=Progress())
    model_path = output / "models" / f"ppo_seed_{seed}.zip"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_path)
    progress = output / "training" / f"seed_{seed}_progress.csv"
    progress.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(progress, index=False, float_format="%.12g")
    record = {"seed": seed, "timesteps": model.num_timesteps,
              "elapsed_seconds": time.monotonic() - started,
              "model_path": str(model_path), "model_sha256": sha(model_path),
              "parameter_sha256": parameter_digest(model.policy), "device": "cpu",
              "train_start": TRAIN_START, "train_end_exclusive": VAL_START,
              "training_window_metadata": env.training_window_metadata,
              "training_episode_history": list(env.training_episode_history),
              "observation_shape": list(env.observation_space.shape),
              "fixed_feature_names": list(env.FEATURE_NAMES)}
    write_json(output / "training" / f"seed_{seed}.json", record)
    print(f"FROZEN seed={seed} steps={model.num_timesteps} parameters={record['parameter_sha256']}", flush=True)
    env.close()
    return record


def evaluate(dataset, models, *, strategy, start, end, costs):
    env = CryptoPortfolioEnv(dataset, start, end, costs=costs,
                             training=False, constraints=strategy != "btc_eth_hold")
    obs, _ = env.reset(seed=0)
    initial, done = True, False
    while not done:
        if strategy in ("ppo", "ppo_ensemble"):
            weights = np.mean([env.action_to_weights(model.predict(obs, deterministic=True)[0]) for model in models], axis=0)
            obs, _, terminated, truncated, _ = env.step_target(weights)
        elif strategy == "btc_eth_hold" and not initial:
            obs, _, terminated, truncated, _ = env.step_hold()
        else:
            weights = np.zeros(11, dtype=float)
            valid = env.current_valid_mask
            if strategy == "top10_equal":
                if valid.any():
                    weights[:10] = np.minimum(.95 * valid / valid.sum(), .25)
            elif strategy == "top10_cap":
                caps = pd.to_numeric(env.current_universe["cap"], errors="raise").to_numpy(dtype=float) * valid
                if caps.sum() > 0:
                    weights[:10] = np.minimum(.95 * caps / caps.sum(), .25)
            elif strategy == "btc_eth_hold":
                if "1" not in env.current_assets or "1027" not in env.current_assets:
                    raise ValueError("Passive reference requires BTC/ETH to exist at initial selection")
                weights[env.current_assets.index("1")] = .475
                weights[env.current_assets.index("1027")] = .475
            elif strategy != "cash":
                raise ValueError(f"Unknown evaluation strategy: {strategy}")
            weights[10] = 1 - weights[:10].sum()
            obs, _, terminated, truncated, _ = env.step_target(weights)
        initial, done = False, terminated or truncated
    result = env.get_result()
    result["metadata"].update(strategy=strategy, model_count=len(models),
                               passive_concentration_cap_exception=strategy == "btc_eth_hold")
    env.close()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "data/crypto_rl_top10/dataset.pkl.gz")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/backtests/crypto_rl_top10_20261024")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--skip-report", action="store_true")
    parser.add_argument("--evaluate-only", action="store_true", help="use saved, SHA-verified frozen models; never retrain")
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        parser.error("workers must be 1..4")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    protocol_path = output / "frozen_spec.json"
    if not protocol_path.is_file():
        raise FileNotFoundError("Predeclare the study protocol in output/frozen_spec.json before training")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if tuple(protocol["seeds"]) != SEEDS or protocol["timesteps_per_seed"] != 32768:
        raise ValueError("This entry point runs only the declared five-seed, fixed-budget experiment")
    expected = {"train": [TRAIN_START, VAL_START], "validation": [VAL_START, TEST_START],
                "test": [TEST_START, TEST_END], "capital": 10000, "cash_target_floor": .05,
                "single_coin_target_cap": .25, "base_cost": COSTS["base"], "stress_cost": COSTS["stress"],
                "net_arch": [32, 32], "learning_rate": .0003, "n_steps": 256,
                "batch_size": 64, "n_epochs": 10, "gamma": .99, "gae_lambda": .95,
                "clip_range": .2, "ent_coef": 0., "train_episode_days": 128}
    for key, value in expected.items():
        if protocol.get(key) != value:
            raise ValueError("Frozen protocol is not implemented by this entry point: " + key)
    dataset = load_dataset(args.dataset)
    code_files = [Path(__file__), *sorted((ROOT / "scripts").glob("crypto_rl_*.*"))]
    initial_code_hashes = {str(p.relative_to(ROOT)): sha(p) for p in code_files if p.is_file()}
    preparation = {"prepared_at": datetime.now(timezone.utc).isoformat(), "dataset_sha256": sha(args.dataset),
                   "protocol_sha256": sha(protocol_path), "code_sha256_before_training": initial_code_hashes}
    preparation_path = output / "training_frozen_inputs.json"
    if preparation_path.exists():
        old = json.loads(preparation_path.read_text(encoding="utf-8"))
        if old["dataset_sha256"] != preparation["dataset_sha256"] or old["protocol_sha256"] != preparation["protocol_sha256"]:
            raise ValueError("Frozen data or protocol changed; preserve this experiment and choose a new output")
        if not args.evaluate_only:
            raise ValueError("Saved training inputs exist; preserve trained results and use --evaluate-only or a new experiment")
        frozen_hashes = {key.replace("\\", "/"): value for key, value in old["code_sha256_before_training"].items()}
        for name in ("scripts/backtest_crypto_rl_top10.py", "scripts/crypto_rl_portfolio_env.py",
                     "scripts/crypto_rl_data.py", "scripts/crypto_rl_reporting.py"):
            expected = frozen_hashes.get(name)
            if expected is None or sha(ROOT / name) != expected:
                raise ValueError("Frozen financial implementation changed: " + name)
    else:
        write_json(preparation_path, preparation)
    config = {**protocol, "test_start": TEST_START, "test_end": TEST_END,
              "train_start": TRAIN_START, "train_end": VAL_START, "validation_start": VAL_START,
              "validation_end": TEST_START, "costs": COSTS}
    write_json(output / "config.json", config)
    if args.evaluate_only:
        records = [json.loads((output / "training" / f"seed_{seed}.json").read_text(encoding="utf-8")) for seed in SEEDS]
        if any(sha(row["model_path"]) != row["model_sha256"] for row in records):
            raise ValueError("A frozen model changed")
    else:
        tasks = [dict(seed=seed, out=str(output), dataset=str(args.dataset.resolve()), timesteps=32768) for seed in SEEDS]
        print(f"Training {len(tasks)} predeclared PPO seeds on pre-test data only...", flush=True)
        if args.workers == 1:
            records = [train_seed(task) for task in tasks]
        else:
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                records = list(pool.map(train_seed, tasks))
    import torch
    from stable_baselines3 import PPO
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    models = [PPO.load(row["model_path"], device="cpu") for row in records]
    if any(parameter_digest(model.policy) != row["parameter_sha256"] for model, row in zip(models, records)):
        raise AssertionError("Loaded checkpoint parameters differ from the frozen records")
    rows, histories, months, holdings, run_manifest, repeats = [], [], [], [], {}, []
    for split, start, end in (("validation", VAL_START, TEST_START), ("test", TEST_START, TEST_END)):
        for scenario, costs in COSTS.items():
            plans = [("ppo", seed, [model]) for seed, model in zip(SEEDS, models)]
            plans += [("ppo_ensemble", None, models), *[(name, None, []) for name in ("cash", "top10_equal", "top10_cap", "btc_eth_hold")]]
            for strategy, seed, selected_models in plans:
                run_id = f"{split}__{scenario}__{strategy}" + (f"__seed{seed}" if seed is not None else "")
                run = lambda st=strategy, mm=selected_models: evaluate(dataset, mm, strategy=st, start=start, end=end, costs=costs)
                result = run()
                fingerprints = save_result(output / "runs" / run_id, result)
                if result_fingerprints(run()) != fingerprints:
                    raise AssertionError("Frozen deterministic evaluation failed: " + run_id)
                repeats.append({"run_id": run_id, "passed": True, "fingerprints": fingerprints})
                row, history, monthly = summarize(result, run_id=run_id, strategy=strategy, seed=seed,
                                                   scenario=scenario, split=split, start=start, end=end, **costs)
                rows.append(row)
                histories.append(history)
                months.append(monthly)
                holdings.extend({**h, "run_id": run_id, "seed": seed, "scenario": scenario, "split": split, "strategy": strategy}
                                for h in result["metadata"]["final_holdings"])
                run_manifest[run_id] = {"fingerprints": fingerprints, "metadata": result["metadata"]}
                print(f"EVAL {run_id}: net={row['net_return']:.2%} hourlyDD={row['max_drawdown']:.2%} fees={row['fees']:.2f}", flush=True)
    model_checks = []
    for model, row in zip(models, records):
        unchanged = parameter_digest(model.policy) == row["parameter_sha256"] and sha(row["model_path"]) == row["model_sha256"]
        if not unchanged:
            raise AssertionError("A frozen model changed during validation/test evaluation")
        model_checks.append({"seed": row["seed"], "parameters_and_checkpoint_unchanged_after_evaluation": True})
    summary, history, monthly = pd.DataFrame(rows), pd.concat(histories, ignore_index=True), pd.concat(months, ignore_index=True)
    final_holdings = pd.DataFrame(holdings)
    universe = dataset["universe"].drop_duplicates(["anchor", "asset_key"]).copy()
    training = pd.DataFrame([{**row, "validation_net_return": float(summary.loc[(summary.strategy == "ppo") & (summary.seed == row["seed"]) & (summary.split == "validation") & (summary.scenario == "base"), "net_return"].iloc[0])} for row in records])
    for filename, frame in (("summary.csv", summary), ("daily_equity.csv", history), ("monthly_returns.csv", monthly),
                            ("seed_statistics.csv", seed_statistics(summary)), ("training_metrics.csv", training),
                            ("universe_history.csv", universe), ("final_holdings.csv", final_holdings)):
        frame.to_csv(output / filename, index=False, float_format="%.12g", encoding="utf-8-sig")
    actual_hashes = {str(p.relative_to(ROOT)): sha(p) for p in code_files if p.is_file()}
    if actual_hashes != initial_code_hashes:
        raise AssertionError("Study source changed during model training/evaluation; preserve evidence and investigate")
    manifest = {"created_at": datetime.now(timezone.utc).isoformat(), "config": config,
                "framework_versions": {n: importlib.metadata.version(n) for n in ("torch", "gymnasium", "stable-baselines3", "numpy", "pandas", "requests")},
                "python": sys.version, "python_executable": sys.executable, "platform": platform.platform(),
                "torch_git_version": torch.version.git_version, "device": "cpu", "torch_threads": 1,
                "source_sha256": actual_hashes, "frozen_inputs": preparation, "data": dataset["manifest"],
                "models": records, "models_unchanged_after_evaluation": model_checks,
                "run_results": run_manifest, "repeat_verifications": repeats,
                "reproduction_argv": [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
                "live_trading_authorized": False}
    write_json(output / "manifest.json", manifest)
    write_json(output / "reviewed.json", make_reviewed_snapshot(summary, history, monthly, universe, training,
                                                                 manifest=manifest, holdings=final_holdings))
    if not args.skip_report:
        from scripts.crypto_rl_report_app import build_report
        write_json(output / "report_build.json", build_report(output))
    print(f"Complete: {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

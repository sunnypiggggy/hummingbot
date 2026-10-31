#!/usr/bin/env python3
"""A2C algorithm control on the unchanged audited Top10 portfolio ledger.

TradeMaster lists A2C as a supported algorithm. This uses the installed SB3
implementation, not TradeMaster's trainer/environment and not its paper results.
No downloaded third-party source is executed. Original PPO outputs are read-only.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.backtest_crypto_rl_top10 import (
    COSTS, SEEDS, TEST_END, TEST_START, TRAIN_START, VAL_START,
    evaluate as original_evaluate, load_dataset, parameter_digest, sha,
)
from scripts.crypto_rl_portfolio_env import CryptoPortfolioEnv
from scripts.crypto_rl_reporting import (
    make_reviewed_snapshot, result_fingerprints, save_result, seed_statistics,
    summarize, write_json,
)

A2C_PARAMS = dict(learning_rate=0.0007, n_steps=8, gamma=0.99, gae_lambda=1.0,
                  ent_coef=0.0, vf_coef=0.5, max_grad_norm=0.5,
                  rms_prop_eps=1e-5, use_rms_prop=True, normalize_advantage=False)


def train_seed(task):
    import torch
    from stable_baselines3 import A2C
    from stable_baselines3.common.callbacks import BaseCallback

    seed, output = int(task["seed"]), Path(task["output"])
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
    progress, started = [], time.monotonic()

    class Progress(BaseCallback):
        def _on_step(self):
            if self.num_timesteps % 4096 == 0:
                progress.append({"seed": seed, "timesteps": self.num_timesteps,
                                 "elapsed_seconds": time.monotonic() - started})
                print(f"TRAIN A2C seed={seed} steps={self.num_timesteps}/32768", flush=True)
            return True

    model = A2C("MlpPolicy", env, device="cpu", seed=seed, verbose=0,
                policy_kwargs={"net_arch": [32, 32]}, **A2C_PARAMS)
    model.learn(total_timesteps=32768, callback=Progress())
    if model.num_timesteps != 32768:
        raise AssertionError("Training exceeded the predeclared interaction budget")
    model_path = output / "models" / f"a2c_seed_{seed}.zip"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_path)
    record = {"seed": seed, "algorithm": "SB3 A2C", "timesteps": model.num_timesteps,
              "elapsed_seconds": time.monotonic() - started, "model_path": str(model_path),
              "model_sha256": sha(model_path), "parameter_sha256": parameter_digest(model.policy),
              "device": "cpu", "train_start": TRAIN_START, "train_end_exclusive": VAL_START,
              "training_window_metadata": env.training_window_metadata,
              "training_episode_history": list(env.training_episode_history),
              "observation_shape": list(env.observation_space.shape),
              "fixed_feature_names": list(env.FEATURE_NAMES)}
    write_json(output / "training" / f"seed_{seed}.json", record)
    pd.DataFrame(progress).to_csv(output / "training" / f"seed_{seed}_progress.csv", index=False)
    print(f"FROZEN A2C seed={seed} parameters={record['parameter_sha256']}", flush=True)
    env.close()
    return record


def evaluate(dataset, models, *, strategy, start, end, costs):
    if strategy not in ("a2c", "a2c_ensemble"):
        return original_evaluate(dataset, models, strategy=strategy, start=start, end=end, costs=costs)
    env = CryptoPortfolioEnv(dataset, start, end, costs=costs, training=False)
    obs, _ = env.reset(seed=0)
    done = False
    while not done:
        weights = np.mean([env.action_to_weights(model.predict(obs, deterministic=True)[0])
                           for model in models], axis=0)
        obs, _, terminated, truncated, _ = env.step_target(weights)
        done = terminated or truncated
    result = env.get_result()
    result["metadata"].update(strategy=strategy, model_count=len(models),
                              implementation="SB3 A2C; unified audited ledger")
    env.close()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "data/crypto_rl_top10/dataset.pkl.gz")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/backtests/crypto_rl_a2c_20261024")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--evaluate-only", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        parser.error("workers must be 1..4")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    protocol_path = output / "frozen_spec.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("algorithm_parameters") != A2C_PARAMS or protocol.get("seeds") != list(SEEDS) or protocol.get("timesteps_per_seed") != 32768:
        raise ValueError("Predeclare the fixed A2C parameters and budget before training")
    original = ROOT / "results/backtests/crypto_rl_top10_20261024"
    original_manifest = json.loads((original / "manifest.json").read_text(encoding="utf-8"))
    config = dict(original_manifest["config"])
    config.update(algorithm="Stable-Baselines3 A2C", algorithm_parameters=A2C_PARAMS,
                  study="A2C algorithm control; not a TradeMaster native reproduction",
                  prior_ppo_test_results_already_observed=True)
    for key in ("n_epochs", "batch_size", "clip_range"):
        config.pop(key, None)
    config.update(A2C_PARAMS)
    dataset = load_dataset(args.dataset)
    if sha(args.dataset) != original_manifest["frozen_inputs"]["dataset_sha256"]:
        raise AssertionError("The algorithm comparison must reuse exactly the PPO data")
    files = [Path(__file__), ROOT / "scripts/backtest_crypto_rl_top10.py",
             ROOT / "scripts/crypto_rl_data.py", ROOT / "scripts/crypto_rl_portfolio_env.py",
             ROOT / "scripts/crypto_rl_reporting.py"]
    code_hashes = {str(p.relative_to(ROOT)): sha(p) for p in files}
    frozen = {"prepared_at": datetime.now(timezone.utc).isoformat(),
              "dataset_sha256": sha(args.dataset), "protocol_sha256": sha(protocol_path),
              "code_sha256_before_training": code_hashes}
    prepared_path = output / "training_frozen_inputs.json"
    if prepared_path.exists():
        previous = json.loads(prepared_path.read_text(encoding="utf-8"))
        if any(previous[key] != frozen[key] for key in ("dataset_sha256", "protocol_sha256", "code_sha256_before_training")):
            raise ValueError("Frozen inputs changed; use a new experiment directory")
        if not args.evaluate_only:
            raise ValueError("Preserve existing training; use --evaluate-only or a new output")
        frozen = previous
    else:
        if args.evaluate_only:
            raise ValueError("No frozen training exists")
        write_json(prepared_path, frozen)
    write_json(output / "config.json", config)
    if args.evaluate_only:
        records = [json.loads((output / "training" / f"seed_{seed}.json").read_text(encoding="utf-8")) for seed in SEEDS]
        if any(sha(row["model_path"]) != row["model_sha256"] for row in records):
            raise ValueError("Frozen model changed")
    else:
        tasks = [dict(seed=seed, output=str(output), dataset=str(args.dataset.resolve())) for seed in SEEDS]
        if args.workers == 1:
            records = [train_seed(task) for task in tasks]
        else:
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                records = list(pool.map(train_seed, tasks))
    import torch
    from stable_baselines3 import A2C
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    models = [A2C.load(row["model_path"], device="cpu") for row in records]
    if any(parameter_digest(m.policy) != row["parameter_sha256"] for m, row in zip(models, records)):
        raise AssertionError("Loaded model parameters differ")
    rows, histories, months, holdings, run_manifest, repeats = [], [], [], [], {}, []
    for split, start, end in (("validation", VAL_START, TEST_START), ("test", TEST_START, TEST_END)):
        for scenario, costs in COSTS.items():
            plans = [("a2c", seed, [model]) for seed, model in zip(SEEDS, models)]
            plans += [("a2c_ensemble", None, models), *[(name, None, []) for name in ("cash", "top10_equal", "top10_cap", "btc_eth_hold")]]
            for strategy, seed, selected_models in plans:
                run_id = f"{split}__{scenario}__{strategy}" + (f"__seed{seed}" if seed is not None else "")
                result = evaluate(dataset, selected_models, strategy=strategy, start=start, end=end, costs=costs)
                fingerprints = save_result(output / "runs" / run_id, result)
                repeated = evaluate(dataset, selected_models, strategy=strategy, start=start, end=end, costs=costs)
                if result_fingerprints(repeated) != fingerprints:
                    raise AssertionError("Deterministic repetition failed: " + run_id)
                row, daily, monthly = summarize(result, run_id=run_id, strategy=strategy, seed=seed,
                                                 scenario=scenario, split=split, start=start, end=end, **costs)
                if strategy.startswith("a2c"):
                    label = "A2C五模型平均权重组合" if strategy.endswith("ensemble") else f"A2C单种子 · seed {seed}"
                    row["label"] = label
                    daily["label"], monthly["label"] = label, label
                rows.append(row)
                histories.append(daily)
                months.append(monthly)
                holdings.extend({**h, "run_id": run_id, "seed": seed, "scenario": scenario, "split": split, "strategy": strategy}
                                for h in result["metadata"]["final_holdings"])
                run_manifest[run_id] = {"fingerprints": fingerprints, "metadata": result["metadata"]}
                repeats.append({"run_id": run_id, "passed": True, "fingerprints": fingerprints})
                print(f"EVAL {run_id}: net={row['net_return']:.2%} hourlyDD={row['max_drawdown']:.2%}", flush=True)
    model_checks = []
    for model, row in zip(models, records):
        unchanged = parameter_digest(model.policy) == row["parameter_sha256"] and sha(row["model_path"]) == row["model_sha256"]
        if not unchanged:
            raise AssertionError("Model changed during evaluation")
        model_checks.append({"seed": row["seed"], "parameters_and_checkpoint_unchanged_after_evaluation": True})
    if any(sha(ROOT / name) != value for name, value in code_hashes.items()):
        raise AssertionError("Frozen financial code changed")
    summary, history, monthly = pd.DataFrame(rows), pd.concat(histories, ignore_index=True), pd.concat(months, ignore_index=True)
    universe = dataset["universe"].drop_duplicates(["anchor", "asset_key"]).copy()
    training = pd.DataFrame([{**row, "validation_net_return": float(summary.loc[(summary.strategy == "a2c") & (summary.seed == row["seed"]) & (summary.split == "validation") & (summary.scenario == "base"), "net_return"].iloc[0])} for row in records])
    for name, frame in (("summary.csv", summary), ("daily_equity.csv", history), ("monthly_returns.csv", monthly),
                        ("seed_statistics.csv", seed_statistics(summary)), ("training_metrics.csv", training),
                        ("universe_history.csv", universe), ("final_holdings.csv", pd.DataFrame(holdings))):
        frame.to_csv(output / name, index=False, float_format="%.17g", encoding="utf-8-sig")
    manifest = {"created_at": datetime.now(timezone.utc).isoformat(), "config": config,
                "implementation_scope": "SB3 A2C algorithm control; TradeMaster lists A2C, but its native environment/trainer were not executed",
                "comparison_is_equal_interaction_budget_not_equal_optimization_compute": True,
                "framework_versions": {name: importlib.metadata.version(name) for name in ("torch", "gymnasium", "stable-baselines3", "numpy", "pandas")},
                "source_sha256": code_hashes, "frozen_inputs": frozen, "data": dataset["manifest"],
                "models": records, "models_unchanged_after_evaluation": model_checks,
                "run_results": run_manifest, "repeat_verifications": repeats,
                "reproduction_argv": [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
                "live_trading_authorized": False}
    write_json(output / "manifest.json", manifest)
    snapshot = make_reviewed_snapshot(summary, history, monthly, universe, training,
                                      manifest=manifest, holdings=pd.DataFrame(holdings))
    snapshot["title"] = snapshot["report"]["title"] = "动态Top10：A2C统一账本算法对照"
    for query in snapshot["queries"].values():
        query["source"]["label"] = "历史动态Top10与本地SB3 A2C现货组合模拟；非TradeMaster原版复现"
    write_json(output / "reviewed.json", snapshot)
    print(f"Complete: {output}", flush=True)


if __name__ == "__main__":
    main()

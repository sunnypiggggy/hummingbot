#!/usr/bin/env python3
"""Retrain the unchanged PPO/A2C implementations on historical Top10 ex-ADA.

Independent user-requested, post-hoc sensitivity study. Original sources,
datasets, models and results are read-only. No private or live interface is used.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.backtest_crypto_rl_top10 import (
    COSTS, SEEDS, TRAIN_START, VAL_START, TEST_START, TEST_END,
    load_dataset, parameter_digest, sha, train_seed as train_ppo, evaluate as evaluate_ppo,
)
from scripts.backtest_rl_a2c_comparison import A2C_PARAMS, train_seed as train_a2c, evaluate as evaluate_a2c
from scripts.crypto_rl_reporting import result_fingerprints, save_result, summarize, seed_statistics, write_json

OLD = {"ppo": ROOT / "results/backtests/crypto_rl_top10_20261024",
       "a2c": ROOT / "results/backtests/crypto_rl_a2c_20261024"}
FINANCIAL_FILES = ["scripts/backtest_crypto_rl_top10.py", "scripts/backtest_rl_a2c_comparison.py",
                   "scripts/crypto_rl_data.py", "scripts/crypto_rl_portfolio_env.py",
                   "scripts/crypto_rl_reporting.py"]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def train_task(task):
    algorithm = task["algorithm"]
    if algorithm == "ppo":
        record = train_ppo(dict(seed=task["seed"], out=task["output"],
                                dataset=task["dataset"], timesteps=32768))
    else:
        record = train_a2c(dict(seed=task["seed"], output=task["output"], dataset=task["dataset"]))
    if record["timesteps"] != 32768:
        raise AssertionError("Training budget differs from the original fixed experiment")
    return {"algorithm": algorithm, "record": record}


def finish_algorithm(algorithm, output, dataset, records, config, frozen, source_hashes):
    import torch
    from stable_baselines3 import PPO, A2C
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    model_class, evaluate = (PPO, evaluate_ppo) if algorithm == "ppo" else (A2C, evaluate_a2c)
    models = [model_class.load(row["model_path"], device="cpu") for row in records]
    if any(sha(row["model_path"]) != row["model_sha256"] or parameter_digest(model.policy) != row["parameter_sha256"]
           for model, row in zip(models, records)):
        raise AssertionError("A frozen checkpoint differs from its training record")
    rows, histories, months, holdings, runs, repeats = [], [], [], [], {}, []
    for split, start, end in (("validation", VAL_START, TEST_START), ("test", TEST_START, TEST_END)):
        for scenario, costs in COSTS.items():
            plans = [(algorithm, seed, [model]) for seed, model in zip(SEEDS, models)]
            plans += [(algorithm + "_ensemble", None, models)]
            plans += [(name, None, []) for name in ("cash", "top10_equal", "top10_cap", "btc_eth_hold")]
            for strategy, seed, selected in plans:
                run_id = f"{split}__{scenario}__{strategy}" + (f"__seed{seed}" if seed is not None else "")
                result = evaluate(dataset, selected, strategy=strategy, start=start, end=end, costs=costs)
                fingerprints = save_result(output / "runs" / run_id, result)
                repeated = evaluate(dataset, selected, strategy=strategy, start=start, end=end, costs=costs)
                if fingerprints != result_fingerprints(repeated):
                    raise AssertionError("Frozen evaluation is not deterministic: " + run_id)
                fills, weights = result["fills"], result["weights"]
                if (len(fills) and fills.asset_key.astype(str).eq("2010").any()) or weights.asset_key.astype(str).eq("2010").any():
                    raise AssertionError("ADA is present in an excluded-universe evaluation")
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
                runs[run_id] = {"fingerprints": fingerprints, "metadata": result["metadata"]}
                repeats.append({"run_id": run_id, "passed": True, "fingerprints": fingerprints})
                print(f"EVAL ex-ADA {run_id}: net={row['net_return']:.2%} hourlyDD={row['max_drawdown']:.2%}", flush=True)
    checks = []
    for model, row in zip(models, records):
        if parameter_digest(model.policy) != row["parameter_sha256"] or sha(row["model_path"]) != row["model_sha256"]:
            raise AssertionError("A model changed during validation/test")
        checks.append({"seed": row["seed"], "parameters_and_checkpoint_unchanged_after_evaluation": True})
    summary, history, monthly = pd.DataFrame(rows), pd.concat(histories, ignore_index=True), pd.concat(months, ignore_index=True)
    training = pd.DataFrame([{**row, "validation_net_return": float(summary.loc[(summary.strategy == algorithm) &
                       (summary.seed == row["seed"]) & (summary.split == "validation") &
                       (summary.scenario == "base"), "net_return"].iloc[0])} for row in records])
    universe = dataset["universe"].drop_duplicates(["anchor", "asset_key"]).copy()
    for name, frame in (("summary.csv", summary), ("daily_equity.csv", history), ("monthly_returns.csv", monthly),
                        ("seed_statistics.csv", seed_statistics(summary)), ("training_metrics.csv", training),
                        ("universe_history.csv", universe), ("final_holdings.csv", pd.DataFrame(holdings))):
        frame.to_csv(output / name, index=False, encoding="utf-8-sig", float_format="%.17g")
    manifest = {"created_at": datetime.now(timezone.utc).isoformat(), "config": config,
                "universe_variant": "ex_ada", "excluded_asset_keys": ["2010"],
                "post_hoc_exclusion_based_on_observed_test": True, "parameter_search": False,
                "framework_versions": {name: importlib.metadata.version(name) for name in ("torch", "gymnasium", "stable-baselines3", "numpy", "pandas")},
                "source_sha256": source_hashes, "frozen_inputs": frozen, "data": dataset["manifest"],
                "models": records, "models_unchanged_after_evaluation": checks, "run_results": runs,
                "repeat_verifications": repeats, "live_trading_authorized": False,
                "reproduction_argv": [sys.executable, str(Path(__file__).resolve()), "--evaluate-only"]}
    write_json(output / "manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "data/crypto_rl_top10_ex_ada/dataset.pkl.gz")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/backtests/crypto_rl_ex_ada_20261024")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--evaluate-only", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        parser.error("workers must be 1..4")
    output = args.output_dir.resolve()
    spec_path = output / "frozen_spec.json"
    spec = read(spec_path)
    if (spec.get("excluded_asset_keys") != ["2010"] or spec.get("seeds") != list(SEEDS)
            or spec.get("timesteps_per_seed") != 32768 or spec.get("parameter_search") is not False
            or spec.get("post_hoc_exclusion_based_on_observed_test") is not True
            or spec.get("algorithms", {}).get("a2c", {}).get("algorithm_parameters") != A2C_PARAMS):
        raise ValueError("Predeclare the exact user-authorized no-ADA protocol before training")
    expected_common = {"train": [TRAIN_START, VAL_START], "validation": [VAL_START, TEST_START],
                       "test": [TEST_START, TEST_END], "capital": 10000, "cash_target_floor": .05,
                       "single_coin_target_cap": .25, "base_cost": COSTS["base"],
                       "stress_cost": COSTS["stress"], "train_episode_days": 128}
    if any(spec.get(key) != value for key, value in expected_common.items()):
        raise ValueError("The new study must retain the original common protocol")
    expected_ppo = {"net_arch": [32, 32], "learning_rate": .0003, "n_steps": 256,
                    "batch_size": 64, "n_epochs": 10, "gamma": .99, "gae_lambda": .95,
                    "clip_range": .2, "ent_coef": 0.}
    if spec["algorithms"]["ppo"]["parameters"] != expected_ppo or spec["algorithms"]["a2c"].get("net_arch") != [32, 32]:
        raise ValueError("Algorithm parameters must remain identical to the original study")
    dataset = load_dataset(args.dataset)
    if "2010" in {str(k) for k in dataset["assets"]} or dataset["universe"].asset_key.astype(str).eq("2010").any():
        raise AssertionError("The dataset still contains ADA")
    if not dataset["universe"].groupby("date").asset_key.nunique().eq(10).all():
        raise AssertionError("Each historical pool must contain ten different eligible assets")
    old_manifests = {name: read(folder / "manifest.json") for name, folder in OLD.items()}
    files = [ROOT / name for name in FINANCIAL_FILES] + [Path(__file__), ROOT / "scripts/rl_ex_ada_data.py"]
    hashes = {str(path.relative_to(ROOT)): sha(path) for path in files}
    for algorithm, old in old_manifests.items():
        original_hashes = {key.replace("\\", "/"): value for key, value in old["source_sha256"].items()}
        for filename in FINANCIAL_FILES:
            if filename in original_hashes and sha(ROOT / filename) != original_hashes[filename]:
                raise AssertionError("An original implementation changed: " + filename)
        for key, expected in {"train_start": TRAIN_START, "train_end": VAL_START, "validation_start": VAL_START,
                              "validation_end": TEST_START, "test_start": TEST_START, "test_end": TEST_END,
                              "costs": COSTS, "capital": 10000}.items():
            if old["config"].get(key) != expected:
                raise AssertionError("The original protocol differs: " + key)
    frozen = {"prepared_at": datetime.now(timezone.utc).isoformat(), "dataset_sha256": sha(args.dataset),
              "dataset_path": str(args.dataset.resolve()), "protocol_sha256": sha(spec_path),
              "code_sha256_before_training": hashes,
              "original_manifest_sha256": {name: sha(folder / "manifest.json") for name, folder in OLD.items()}}
    path = output / "training_frozen_inputs.json"
    if path.exists():
        previous = read(path)
        if any(previous[key] != frozen[key] for key in ("dataset_sha256", "protocol_sha256", "code_sha256_before_training", "original_manifest_sha256")):
            raise ValueError("Frozen inputs changed; preserve this study and use a fresh experiment")
        if not args.evaluate_only:
            raise ValueError("Frozen training exists; use --evaluate-only rather than overwrite models")
        frozen = previous
    else:
        if args.evaluate_only:
            raise ValueError("No frozen model study exists")
        write_json(path, frozen)
    configs = {}
    for algorithm in OLD:
        folder = output / algorithm
        folder.mkdir(parents=True, exist_ok=True)
        configs[algorithm] = {**old_manifests[algorithm]["config"], "universe_variant": "ex_ada",
                              "excluded_asset_keys": ["2010"], "parameter_search": False,
                              "post_hoc_exclusion_based_on_observed_test": True,
                              "universe": "monthly historical eligible Top10 after excluding ADA; rank-based replacement",
                              "study": "Ex-ADA retraining sensitivity, unchanged original algorithm and ledger"}
        write_json(folder / "config.json", configs[algorithm])
        write_json(folder / "training_frozen_inputs.json", frozen)
    if args.evaluate_only:
        trained = [{"algorithm": algorithm, "record": read(output / algorithm / "training" / f"seed_{seed}.json")}
                   for algorithm in OLD for seed in SEEDS]
    else:
        tasks = [dict(algorithm=algorithm, seed=seed, output=str(output / algorithm), dataset=str(args.dataset.resolve()))
                 for algorithm in OLD for seed in SEEDS]
        print("Training fixed PPO and A2C seeds on the no-ADA historical universe...", flush=True)
        if args.workers == 1:
            trained = [train_task(task) for task in tasks]
        else:
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                trained = list(pool.map(train_task, tasks))
    manifests = {}
    for algorithm in OLD:
        records = sorted([row["record"] for row in trained if row["algorithm"] == algorithm], key=lambda row: row["seed"])
        if [row["seed"] for row in records] != list(SEEDS):
            raise AssertionError("All five original seeds are required")
        manifests[algorithm] = finish_algorithm(algorithm, output / algorithm, dataset, records,
                                                configs[algorithm], frozen, hashes)
    if any(sha(ROOT / name) != value for name, value in hashes.items()):
        raise AssertionError("A frozen study implementation changed during training/evaluation")
    for algorithm, folder in OLD.items():
        if sha(folder / "manifest.json") != frozen["original_manifest_sha256"][algorithm]:
            raise AssertionError("An original study was modified")
    for run_id, result in manifests["ppo"]["run_results"].items():
        if run_id.endswith(tuple("__" + name for name in ("cash", "top10_equal", "top10_cap", "btc_eth_hold"))):
            if result["fingerprints"] != manifests["a2c"]["run_results"][run_id]["fingerprints"]:
                raise AssertionError("Shared ex-ADA baselines differ between algorithm studies")
    write_json(output / "numeric_completion.json", {"completed_at": datetime.now(timezone.utc).isoformat(),
               "status": "complete", "evaluations": 80, "repeat_checks": 80, "excluded_asset_keys": ["2010"],
               "manifest_sha256": {algorithm: sha(output / algorithm / "manifest.json") for algorithm in OLD},
               "original_manifest_sha256": frozen["original_manifest_sha256"], "shared_baselines_match": True})
    print("NUMERIC COMPLETE: " + str(output), flush=True)


if __name__ == "__main__":
    main()

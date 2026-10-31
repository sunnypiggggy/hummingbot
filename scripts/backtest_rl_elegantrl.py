"""Bounded native ElegantRL study on the unchanged historical Top10 ledger.

All upstream sources, hyperparameters and budgets must be frozen before training.
Old SB3 studies are read-only controls. No production interface is imported.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.backtest_crypto_rl_top10 import (
    COSTS, SEEDS, TRAIN_START, VAL_START, TEST_START, TEST_END,
    load_dataset, sha, evaluate as evaluate_baseline,
)
from scripts.crypto_rl_portfolio_env import CryptoPortfolioEnv
from scripts.crypto_rl_reporting import (
    result_fingerprints, save_result, summarize, seed_statistics, write_json,
)
from scripts.crypto_rl_elegantrl_adapter import (
    ExplicitReplay, TransitionCollector, make_agent, network_tensors, tensor_digest,
    predict, save_checkpoint, load_checkpoint,
)

DEFAULT_OUTPUT = ROOT / "results/backtests/crypto_rl_elegantrl_20261024"
DATASETS = {"original": ROOT / "data/crypto_rl_top10/dataset.pkl.gz",
            "ex_ada": ROOT / "data/crypto_rl_top10_ex_ada/dataset.pkl.gz"}
BASELINES = ("cash", "top10_equal", "top10_cap", "btc_eth_hold")
FINANCIAL_SOURCES = ("scripts/backtest_rl_elegantrl.py", "scripts/crypto_rl_elegantrl_adapter.py",
                     "scripts/backtest_crypto_rl_top10.py", "scripts/crypto_rl_portfolio_env.py",
                     "scripts/crypto_rl_data.py", "scripts/crypto_rl_reporting.py")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def deterministic(seed):
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_task(task):
    algorithm, variant, seed = task["algorithm"], task["variant"], task["seed"]
    output, spec = Path(task["output"]), task["spec"]
    params = spec["algorithms"][algorithm]["parameters"]
    deterministic(seed)
    dataset = load_dataset(DATASETS[variant])
    env = CryptoPortfolioEnv(dataset, TRAIN_START, VAL_START, costs=COSTS["base"],
                             training=True, episode_days=128)
    agent = make_agent(algorithm, spec["vendor_path"], params)
    collector = TransitionCollector(env, agent, algorithm, seed)
    replay = None if algorithm == "ppo" else ExplicitReplay(32768, 92, 11)
    progress, started, update_count, actor_updates = [], time.monotonic(), 0, 0
    budget = spec["timesteps_per_seed"]
    # Native learner enables gradients only for its optimizer stage. Values and
    # GAE targets must be collected with gradients off, including the first call.
    torch.set_grad_enabled(False)
    for completed in range(0, budget, 256):
        batch = collector.collect(256, total_steps=budget, replay=replay, warmup=1024)
        if algorithm == "ppo":
            updates = 40
            stats = agent.update_net(batch)
            actor_updates += updates
        elif replay.cur_size >= 1024:
            # Native update_net schedules int(cur_size*repeat_times/batch_size).
            # Keep exactly 64 updates per 256 new transitions after warmup.
            agent.repeat_times = (64 * params["batch_size"] + .001) / replay.cur_size
            updates = int(replay.cur_size * agent.repeat_times / agent.batch_size)
            if updates != 64:
                raise AssertionError("Bounded native update schedule changed")
            stats = agent.update_net(replay)
            actor_updates += 32 if algorithm == "td3" else updates
        else:
            updates, stats = 0, []
        update_count += updates
        if not all(torch.isfinite(value).all() for value in network_tensors(agent).values()):
            raise AssertionError("Nonfinite trained network")
        if (completed + 256) % 4096 == 0:
            record = {"seed": seed, "timesteps": collector.steps,
                      "gradient_updates": update_count, "actor_updates": actor_updates,
                      "elapsed_seconds": time.monotonic() - started,
                      "training_objectives": list(stats)}
            progress.append(record)
            print(f"TRAIN ElegantRL {variant}/{algorithm} seed={seed}: {collector.steps}/{budget}", flush=True)
    if collector.steps != 32768 or update_count != (5120 if algorithm == "ppo" else 8000):
        raise AssertionError("Actual interactions or native gradient updates differ from frozen budget")
    checkpoint = output / "models" / f"{algorithm}_seed_{seed}.pth"
    metadata = {"implementation": "elegantrl", "algorithm": algorithm, "seed": seed,
                "universe_variant": variant, "commit": spec["upstream_commit"],
                "parameters": params, "train_start": TRAIN_START, "train_end": VAL_START}
    save_checkpoint(checkpoint, agent, metadata)
    record = {**metadata, "timesteps": collector.steps,
              "elapsed_seconds": time.monotonic() - started, "model_path": str(checkpoint),
              "model_sha256": sha(checkpoint), "parameter_sha256": tensor_digest(network_tensors(agent)),
              "actor_parameter_sha256": tensor_digest(agent.act.state_dict()),
              "gradient_updates": update_count, "actor_updates": actor_updates,
              "replay_samples_drawn": replay.samples_drawn if replay else 0,
              "replay_size": replay.cur_size if replay else 0,
              "collector_truncations": collector.truncations, "collector_terminals": collector.terminals,
              "timeout_bootstrap_adjustments": collector.bootstrap_adjustments,
              "device": "cpu", "train_end_exclusive": VAL_START,
              "training_window_metadata": env.training_window_metadata,
              "training_episode_history": list(env.training_episode_history),
              "observation_shape": list(env.observation_space.shape),
              "fixed_feature_names": list(env.FEATURE_NAMES)}
    write_json(output / "training" / f"seed_{seed}.json", record)
    pd.DataFrame(progress).to_csv(output / "training" / f"seed_{seed}_progress.csv", index=False)
    print(f"FROZEN ElegantRL {variant}/{algorithm} seed={seed}", flush=True)
    env.close()
    return {"variant": variant, "algorithm": algorithm, "record": record}


def evaluate(dataset, agents, *, strategy, start, end, costs):
    if strategy in BASELINES:
        return evaluate_baseline(dataset, [], strategy=strategy, start=start, end=end, costs=costs)
    env = CryptoPortfolioEnv(dataset, start, end, costs=costs, training=False)
    obs, _ = env.reset(seed=0)
    done = False
    while not done:
        weights = np.mean([env.action_to_weights(predict(agent, obs)) for agent in agents], axis=0)
        obs, _, terminated, truncated, _ = env.step_target(weights)
        done = terminated or truncated
    result = env.get_result()
    result["metadata"].update(strategy=strategy, model_count=len(agents), implementation="elegantrl")
    env.close()
    return result


def finish_case(task):
    variant, algorithm, output, spec = task["variant"], task["algorithm"], Path(task["output"]), task["spec"]
    deterministic(0)
    records = [read(output / "training" / f"seed_{seed}.json") for seed in SEEDS]
    agents = []
    for record in records:
        if sha(record["model_path"]) != record["model_sha256"]:
            raise AssertionError("Frozen native checkpoint changed")
        agent = make_agent(algorithm, spec["vendor_path"], spec["algorithms"][algorithm]["parameters"])
        load_checkpoint(record["model_path"], agent)
        if tensor_digest(network_tensors(agent)) != record["parameter_sha256"]:
            raise AssertionError("Loaded actor/critic/target/alpha parameters differ")
        agents.append(agent)
    dataset = load_dataset(DATASETS[variant])
    rows, histories, months, holdings, runs, repeats = [], [], [], [], {}, []
    for split, start, end in (("validation", VAL_START, TEST_START), ("test", TEST_START, TEST_END)):
        for scenario, costs in COSTS.items():
            plans = [(algorithm, seed, [agent]) for seed, agent in zip(SEEDS, agents)]
            plans += [(algorithm + "_ensemble", None, agents)]
            plans += [(name, None, []) for name in BASELINES]
            for strategy, seed, selected in plans:
                run_id = f"{split}__{scenario}__{strategy}" + (f"__seed{seed}" if seed is not None else "")
                result = evaluate(dataset, selected, strategy=strategy, start=start, end=end, costs=costs)
                fingerprints = save_result(output / "runs" / run_id, result)
                repeated = evaluate(dataset, selected, strategy=strategy, start=start, end=end, costs=costs)
                if fingerprints != result_fingerprints(repeated):
                    raise AssertionError("Evaluation was not deterministic: " + run_id)
                if variant == "ex_ada":
                    if any(len(result[name]) and result[name].asset_key.astype(str).eq("2010").any()
                           for name in ("fills", "weights")):
                        raise AssertionError("ADA leaked into an excluded pool")
                row, daily, monthly = summarize(result, run_id=run_id, strategy=strategy, seed=seed,
                                                 scenario=scenario, split=split, start=start, end=end, **costs)
                if strategy not in BASELINES:
                    label = f"ElegantRL {algorithm.upper()}" + ("五模型平均权重组合" if seed is None else f"单种子 · seed {seed}")
                    row["label"], daily["label"], monthly["label"] = label, label, label
                rows.append(row)
                histories.append(daily)
                months.append(monthly)
                holdings.extend({**h, "run_id": run_id, "seed": seed, "scenario": scenario, "split": split, "strategy": strategy}
                                for h in result["metadata"]["final_holdings"])
                runs[run_id] = {"fingerprints": fingerprints, "metadata": result["metadata"]}
                repeats.append({"run_id": run_id, "passed": True, "fingerprints": fingerprints})
                print(f"EVAL ElegantRL {variant}/{algorithm}/{run_id}: net={row['net_return']:.2%} DD={row['max_drawdown']:.2%}", flush=True)
    checks = []
    for agent, record in zip(agents, records):
        if tensor_digest(network_tensors(agent)) != record["parameter_sha256"] or sha(record["model_path"]) != record["model_sha256"]:
            raise AssertionError("Native frozen model changed during evaluation")
        checks.append({"seed": record["seed"], "parameters_and_checkpoint_unchanged_after_evaluation": True})
    summary, history, monthly = pd.DataFrame(rows), pd.concat(histories, ignore_index=True), pd.concat(months, ignore_index=True)
    training = pd.DataFrame([{**record, "validation_net_return": float(summary.loc[
        (summary.strategy == algorithm) & (summary.seed == record["seed"]) &
        (summary.split == "validation") & (summary.scenario == "base"), "net_return"].iloc[0])} for record in records])
    for name, frame in (("summary.csv", summary), ("daily_equity.csv", history), ("monthly_returns.csv", monthly),
                        ("seed_statistics.csv", seed_statistics(summary)), ("training_metrics.csv", training),
                        ("universe_history.csv", dataset["universe"].drop_duplicates(["anchor", "asset_key"])),
                        ("final_holdings.csv", pd.DataFrame(holdings))):
        frame.to_csv(output / name, index=False, encoding="utf-8-sig", float_format="%.17g")
    frozen = read(Path(task["root"]) / "training_frozen_inputs.json")
    config = {**read(ROOT / "results/backtests/crypto_rl_top10_20261024/config.json"),
              "implementation": "elegantrl", "algorithm": algorithm.upper(), "universe_variant": variant,
              "algorithm_parameters": spec["algorithms"][algorithm]["parameters"],
              "upstream_commit": spec["upstream_commit"], "parameter_search": False}
    config.update(study="Native ElegantRL official helloworld agents on the unchanged portfolio ledger",
                  frozen_at=spec["frozen_at"], implementation_path="helloworld/erl_agent.py",
                  training_collector="Explicit terminal next state; real timeout bootstrap; no stock collector",
                  gradient_updates_per_seed=5120 if algorithm == "ppo" else 8000)
    config.pop("protocol_pretraining_amendment", None)
    if algorithm != "ppo":
        for key in ("n_epochs", "gae_lambda", "clip_range", "ent_coef"):
            config.pop(key, None)
    write_json(output / "config.json", config)
    write_json(output / "training_frozen_inputs.json", frozen)
    manifest = {"created_at": datetime.now(timezone.utc).isoformat(), "config": config,
                "universe_variant": variant, "implementation": "elegantrl", "excluded_asset_keys": ["2010"] if variant == "ex_ada" else [],
                "framework_versions": {name: importlib.metadata.version(name) for name in ("torch", "gymnasium", "numpy", "pandas")},
                "source_sha256": frozen["code_sha256_before_training"],
                "frozen_inputs": {**frozen, "dataset_sha256": sha(DATASETS[variant])}, "data": dataset["manifest"],
                "models": records, "models_unchanged_after_evaluation": checks,
                "run_results": runs, "repeat_verifications": repeats, "live_trading_authorized": False}
    write_json(output / "manifest.json", manifest)
    return {"variant": variant, "algorithm": algorithm, "manifest_sha256": sha(output / "manifest.json"), "runs": len(runs)}


def verify_vendor(spec):
    manifest = read(spec["vendor_manifest"])
    if manifest["commit"] != spec["upstream_commit"]:
        raise AssertionError("Upstream commit differs from frozen vendor")
    for relative, expected in spec["vendor_file_sha256"].items():
        if sha(Path(spec["vendor_path"]) / relative) != expected:
            raise AssertionError("Pinned upstream file changed: " + relative)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--phase", choices=("all", "train", "evaluate"), default="all")
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        parser.error("workers must be 1..4")
    output = args.output_dir.resolve()
    spec = read(output / "frozen_spec.json")
    if spec["seeds"] != list(SEEDS) or spec["timesteps_per_seed"] != 32768 or spec["parameter_search"] is not False:
        raise ValueError("The predeclared five-seed bounded experiment is required")
    expected = {"train": [TRAIN_START, VAL_START], "validation": [VAL_START, TEST_START],
                "test": [TEST_START, TEST_END], "costs": COSTS, "capital": 10000,
                "train_episode_days": 128, "cash_target_floor": .05, "single_coin_target_cap": .25}
    if any(spec.get(key) != value for key, value in expected.items()):
        raise ValueError("Date, cost, cash and ledger protocol must match the prior study")
    if any(sha(path) != spec["dataset_sha256_by_variant"][variant] for variant, path in DATASETS.items()):
        raise ValueError("Frozen original and ex-ADA dataset bytes are required")
    verify_vendor(spec)
    frozen = {"prepared_at": datetime.now(timezone.utc).isoformat(),
              "dataset_sha256_by_variant": {key: sha(path) for key, path in DATASETS.items()},
              "protocol_sha256": sha(output / "frozen_spec.json"),
              "code_sha256_before_training": {name: sha(ROOT / name) for name in FINANCIAL_SOURCES},
              "vendor_manifest_sha256": sha(spec["vendor_manifest"])}
    frozen_path = output / "training_frozen_inputs.json"
    if frozen_path.exists():
        previous = read(frozen_path)
        if any(previous[key] != value for key, value in frozen.items() if key != "prepared_at"):
            raise ValueError("Frozen inputs changed; preserve this experiment")
        if args.phase != "evaluate":
            raise ValueError("Existing frozen study may only be evaluated, never silently retrained")
    else:
        if args.phase == "evaluate":
            raise ValueError("No frozen training exists")
        write_json(frozen_path, frozen)
    cases = [{"variant": variant, "algorithm": algorithm, "output": str(output / variant / algorithm),
              "root": str(output), "spec": spec} for variant in DATASETS for algorithm in spec["algorithms"]]
    if args.phase != "evaluate":
        tasks = [{**case, "seed": seed} for case in cases for seed in SEEDS]
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for completed in as_completed([pool.submit(train_task, task) for task in tasks]):
                completed.result()
    if args.phase != "train":
        results = []
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for completed in as_completed([pool.submit(finish_case, case) for case in cases]):
                results.append(completed.result())
        write_json(output / "numeric_completion.json", {"status": "complete", "completed_at": datetime.now(timezone.utc).isoformat(),
                   "evaluations": sum(row["runs"] for row in results), "repeat_checks": sum(row["runs"] for row in results),
                   "cases": results, "models": len(cases) * 5,
                   "manifest_sha256": {row["variant"] + "__" + row["algorithm"]: row["manifest_sha256"] for row in results},
                   "live_trading_authorized": False})
    verify_vendor(spec)
    current = read(frozen_path)
    if any(sha(ROOT / key) != value for key, value in current["code_sha256_before_training"].items()):
        raise AssertionError("Frozen implementation changed during execution")
    print("ELEGANTRL PHASE COMPLETE: " + args.phase, flush=True)


if __name__ == "__main__":
    main()

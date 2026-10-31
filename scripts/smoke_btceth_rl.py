#!/usr/bin/env python3
"""Small real-market interface/performance check, using TRAIN dates only."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=512)
    args = parser.parse_args()
    from scripts.backtest_btceth_rl import gpu_initializer
    gpu_initializer(args.gpu_uuid)
    from scripts.backtest_btceth_rl import (ALGORITHMS, COSTS, EPISODE_DAYS, TRAIN, evaluate,
        hyperparameters, load_dataset, model_class, parameter_digest, sha, torch_setup)
    from scripts.btceth_rl_env import BTCETHPortfolioEnv
    from scripts.crypto_rl_reporting import result_fingerprints, write_json
    from stable_baselines3.common.env_checker import check_env
    import numpy as np
    import torch

    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError("Preserve existing smoke evidence")
    output.mkdir(parents=True)
    device = torch_setup(123)
    data = load_dataset(args.dataset)
    env = BTCETHPortfolioEnv(data, *TRAIN, training=True, episode_days=EPISODE_DAYS, record_history=False)
    check_env(env, warn=True)
    episodes = env.training_window_metadata
    rows = []
    for algorithm in ALGORITHMS:
        torch_setup(123)
        env = BTCETHPortfolioEnv(data, *TRAIN, training=True, episode_days=EPISODE_DAYS, record_history=False)
        params = hyperparameters(algorithm)
        policy = params.pop("policy")
        # Exercise gradient updates quickly, without pretending this is a
        # formal experiment or reusing these checkpoints for evaluation.
        if algorithm == "recurrent_ppo":
            params.update(n_steps=64, batch_size=64, n_epochs=1)
        else:
            params.update(learning_starts=32, buffer_size=2048, batch_size=64)
        model = model_class(algorithm)(policy, env, seed=123, device="cuda:0", verbose=0, **params)
        before = parameter_digest(model.policy)
        started = time.monotonic()
        model.learn(total_timesteps=args.steps)
        elapsed = time.monotonic() - started
        trained = parameter_digest(model.policy)
        if before == trained:
            raise AssertionError("Smoke failed to update the policy: " + algorithm)
        path = output / (algorithm + ".zip")
        model.save(path)
        loaded = model_class(algorithm).load(path, device="cuda:0")
        if parameter_digest(loaded.policy) != trained:
            raise AssertionError("Checkpoint reload mismatch")
        # Test the real inference wrapper, including full-day marks and LSTM
        # states, on a short TRAIN-only interval with known complete features.
        result = evaluate(data, [loaded], algorithm, "2024-01-01", "2024-01-15", COSTS["base"], "balanced")
        repeated = evaluate(data, [loaded], algorithm, "2024-01-01", "2024-01-15", COSTS["base"], "balanced")
        fingerprints = result_fingerprints(result)
        if fingerprints != result_fingerprints(repeated):
            raise AssertionError("Real-market inference repeat differs")
        if len(result["equity"]) != 14 * 24:
            raise AssertionError("Real market smoke dropped hourly marks")
        row = dict(algorithm=algorithm, steps=args.steps, elapsed_seconds=elapsed,
                   steps_per_second=args.steps / elapsed, parameters_updated=True,
                   parameter_sha256=trained, checkpoint_sha256=sha(path),
                   reloaded_parameters_exact=True, replay_exact=True,
                   equity_rows=len(result["equity"]), fingerprints=fingerprints,
                   all_parameters_cuda=all(p.is_cuda for p in model.policy.parameters()))
        if not row["all_parameters_cuda"]:
            raise AssertionError("CPU policy fallback")
        rows.append(row)
        print(f"SMOKE {algorithm} {args.steps} steps {elapsed:.2f}s", flush=True)
        env.close()
        del model, loaded
        torch.cuda.empty_cache()
    write_json(output / "verification.json", dict(passed=True, device=device, models=rows,
        training_window_quality=episodes, observation_shape=list(env.observation_space.shape),
        action_shape=list(env.action_space.shape), no_validation_or_test_rewards=True))


if __name__ == "__main__":
    main()

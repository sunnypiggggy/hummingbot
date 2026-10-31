"""Bounded native-GPU proof of the additional seed whitelist, not a backtest."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[name] = "1"
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def run(algorithm, output, core_root):
    import gymnasium as gym
    import numpy as np
    from scripts import backtest_eth_rl_seed_stability as extension
    core, models = extension.apply_runtime_overlay(core_root)

    class SyntheticETH(gym.Env):
        """Only shapes and native optimizer/replay/serialization are tested."""
        observation_space = gym.spaces.Box(-10., 10., (13,), dtype=np.float32)
        action_space = gym.spaces.Box(-1., 1., (1,), dtype=np.float32)
        SYMBOLS = ("ETHUSDT",)

        def __init__(self):
            self.position = 0

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            self.position = 0
            return np.zeros(13, dtype=np.float32), {}

        def step(self, action):
            self.position += 1
            obs = np.array([self.position / 32, float(action[0]), *[np.sin(self.position + n) * .1 for n in range(11)]], dtype=np.float32)
            return obs, float(action[0]) * .001, False, self.position == 32, {}

    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    if core.SYMBOLS != ("ETHUSDT",) or core.SEEDS != tuple(range(30)) or models.SEEDS != tuple(range(30)):
        raise AssertionError("New seeds were not applied in the actual worker")
    env, seed, steps = SyntheticETH(), 29, 1024
    model = models.make_model(algorithm, env, seed=seed, device="cuda:0")
    training = models.train_model(model, env, steps=steps, seed=seed, progress_path=output / "progress.jsonl")
    counters = training["counters"]
    expected_optimizers = {"critic": 512, "actor": 256 if algorithm == "td3" else 512}
    if algorithm == "sac":
        expected_optimizers["entropy"] = 512
    if (training["actual_steps"] != steps or counters["gradient_update_rounds"] != 512
            or counters["optimizer_steps"] != expected_optimizers
            or counters["replay_transitions"] != steps or counters["replay_add_calls"] != steps):
        raise AssertionError("Native post-warmup counters changed")
    checkpoint = models.save_model(model, output / "checkpoint", algorithm=algorithm)
    restored = models.load_model(output / "checkpoint", algorithm=algorithm, device="cuda:0")
    for i in range(16):
        obs = np.array([np.sin(i + j) * .1 for j in range(13)], dtype=np.float32)
        before, _ = models.predict(model, obs, algorithm=algorithm)
        after, _ = models.predict(restored, obs, algorithm=algorithm)
        np.testing.assert_array_equal(before, after)
    import torch
    if torch.cuda.device_count() != 1 or not str(next(model.policy.parameters()).device).startswith("cuda"):
        raise AssertionError("Native model is not on the isolated GPU")
    physical = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader"], text=True)
    devices = [line.split(",")[1].strip() for line in physical.splitlines() if line.split(",")[0].strip() == str(os.getpid())]
    if devices != [os.environ["CUDA_VISIBLE_DEVICES"]]:
        raise AssertionError("Physical PID/GPU assignment differs from the worker isolation")
    result = {"passed": True, "at": extension.now(), "algorithm": algorithm, "seed": seed, "steps": steps,
              "synthetic_nonfinancial_fixture": True, "formal_budget_unchanged": models.STEPS[algorithm],
              "native_counters": counters, "actual_core_seeds": list(core.SEEDS), "actual_model_seeds": list(models.SEEDS),
              "actual_core_symbols": list(core.SYMBOLS), "checkpoint": checkpoint,
              "restore_equal_actions_checked": 16, "pid": os.getpid(), "physical_gpu_uuid": devices[0],
              "source_sha256": extension.sha(__file__), "extension_sha256": extension.sha(extension.__file__),
              "model_source_sha256": models.sha(models.__file__), "runtime_versions": models.runtime_versions()}
    extension.write_new(output / "result.json", result)
    print(json.dumps({k: result[k] for k in ("passed", "algorithm", "seed", "steps", "physical_gpu_uuid")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithm", choices=("sac", "td3"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--core-source-root", type=Path, required=True)
    args = parser.parse_args()
    run(args.algorithm, args.output, args.core_source_root)

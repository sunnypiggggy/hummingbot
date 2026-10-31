"""One native CUDA readiness test on synthetic data; never a portfolio result."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.dont_write_bytecode = True


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False); stream.write("\n")


def publish(path, value):
    temporary = Path(str(path) + ".pending")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def load_source(root):
    sys.path.insert(0, str(Path(root).resolve()))
    from scripts import btceth_rl_td3
    btceth_rl_td3.financial_imports()
    return btceth_rl_td3


def supervised(args):
    td3 = load_source(args.source_root)
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError("Preserve every previous toy attempt; never use a formal experiment directory")
    allowance_path = Path(args.allowance_source)
    allowance = td3.read_json(allowance_path)
    source_sha = digest(allowance_path)
    if source_sha != args.allowance_sha256:
        raise ValueError("Reviewed resource contract bytes changed")
    gpus = td3.shared.gpu_inventory()
    command = ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"]
    raw = subprocess.check_output(command, text=True)
    td3.verify_resource_rows(td3.resource_rows(raw), dict(allowed_compute_instance=allowance["allowed_instance"], allowed_gpu_rows=allowance["allowed_gpu_rows"]))
    # Prefer the card with no existing UI context; this does not terminate it.
    occupied = {row["gpu_uuid"] for row in td3.resource_rows(raw)}
    chosen = next((gpu for gpu in gpus if gpu["uuid"] not in occupied), gpus[0])
    td3.shared.runtime_packages()
    installed = td3.library_sources()
    output.mkdir(parents=True)
    resource = dict(at=td3.now(), physical_gpus=gpus, compute_command=command, unfiltered_compute_stdout=raw,
        selected_uuid=chosen["uuid"], allowed_instance=allowance["allowed_instance"], allowed_gpu_rows=allowance["allowed_gpu_rows"],
        reviewed_contract_file=str(allowance_path), reviewed_contract_sha256=source_sha)
    write_new(output / "resource_preflight.json", resource)
    source_files = {"scripts/" + name: digest(Path(args.source_root) / "scripts" / name) for name in
        ("__init__.py", "btceth_rl_td3.py", "backtest_btceth_rl.py", "btceth_rl_env.py", "crypto_rl_portfolio_env.py", "crypto_rl_reporting.py")}
    source_files["smoke_btceth_rl_td3.py"] = digest(__file__)
    write_new(output / "source_manifest.json", dict(files=source_files, sb3_source_sha256=installed))
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=chosen["uuid"], CUBLAS_WORKSPACE_CONFIG=":4096:8", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
    child_command = [sys.executable, "-u", str(Path(__file__).resolve()), "--phase", "_run", "--source-root", str(Path(args.source_root).resolve()), "--output-dir", str(output.resolve())]
    record = dict(phase="toy_cuda_smoke", status="launching", started_at=td3.now(), supervisor_pid=os.getpid(),
        source_sha256=digest(__file__), runner_sha256=source_files["scripts/btceth_rl_td3.py"],
        resource_sha256=digest(output / "resource_preflight.json"), source_manifest_sha256=digest(output / "source_manifest.json"),
        command=child_command, log_relative_path="smoke.log", formal_training=False)
    proof, log = output / "smoke_process.json", output / "smoke.log"
    write_new(proof, record)
    with log.open("x", encoding="utf-8") as stream:
        child = subprocess.Popen(child_command, env=env, stdout=stream, stderr=subprocess.STDOUT)
        record.update(status="running", child_pid=child.pid); publish(proof, record)
        print(json.dumps(dict(supervisor_pid=os.getpid(), child_pid=child.pid, selected_uuid=chosen["uuid"])), flush=True)
        code = child.wait()
    record.update(status="completed" if code == 0 else "failed", exit_code=code, completed_at=td3.now(), log_sha256=digest(log))
    publish(proof, record)
    print(json.dumps(record), flush=True)
    if code:
        raise RuntimeError("Toy failed; preserve this attempt and its real terminal evidence")


def toy_run(args):
    td3 = load_source(args.source_root); output = Path(args.output_dir)
    proof = td3.read_json(output / "smoke_process.json")
    if proof.get("child_pid") != os.getpid() or proof.get("supervisor_pid") != os.getppid() or proof.get("status") != "running":
        raise ValueError("Actual unique toy supervisor/child required")
    if (proof["source_sha256"] != digest(__file__) or proof["resource_sha256"] != digest(output / "resource_preflight.json")
            or proof["source_manifest_sha256"] != digest(output / "source_manifest.json")):
        raise ValueError("Toy source/preflight changed")
    source = td3.read_json(output / "source_manifest.json")
    for name, expected in source["files"].items():
        path = Path(__file__) if name == "smoke_btceth_rl_td3.py" else Path(args.source_root) / name
        if digest(path) != expected:
            raise ValueError("Toy source bytes changed")
    device = td3.shared.torch_setup(1)
    import gymnasium as gym
    import numpy as np
    import torch
    from stable_baselines3 import TD3
    td3.library_imports(source["sb3_source_sha256"])
    if (device["capability"] != [7, 0] or "V100" not in device["name"]
            or device["visible_gpu_uuid"] != td3.read_json(output / "resource_preflight.json")["selected_uuid"]):
        raise ValueError("Native V100 sm70 required")

    class Toy(gym.Env):
        observation_space = gym.spaces.Box(-10, 10, shape=(27,), dtype=np.float32)
        action_space = gym.spaces.Box(-1, 1, shape=(2,), dtype=np.float32)
        def __init__(self):
            self.episode, self.steps = 0, 0
        def obs(self):
            result = np.linspace(-.5, .5, 27, dtype=np.float32)
            result[0], result[1] = self.steps / 32, self.episode / 100
            return result
        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed); self.episode += 1; self.steps = 0
            return self.obs(), {}
        def step(self, action):
            self.steps += 1
            # Pure synthetic control loss; no prices, assets or portfolio PnL.
            reward = -float(np.square(np.asarray(action, dtype=float) - [.2, -.2]).mean())
            return self.obs(), reward, False, self.steps == 32, {"date": f"toy_episode_{self.episode}_step_{self.steps}"}

    parameters = td3.model_parameters(output / "toy_timeout_transitions.jsonl")
    parent_buffer = parameters["replay_buffer_class"]

    class RNGObservedBuffer(parent_buffer):
        def add(self, *values, **kwargs):
            before = np.random.get_state(); super().add(*values, **kwargs); after = np.random.get_state()
            if before[0] != after[0] or not np.array_equal(before[1], after[1]) or before[2:] != after[2:]:
                raise AssertionError("Observational replay.add consumed RNG")
            self.rng_checks = getattr(self, "rng_checks", 0) + 1
    parameters["replay_buffer_class"] = RNGObservedBuffer
    env = Toy(); model = TD3("MlpPolicy", env, seed=1, device="cuda:0", verbose=0, **parameters)
    before = td3.shared.parameter_digest(model.policy)
    started = time.monotonic(); model.learn(total_timesteps=1024)
    after = td3.shared.parameter_digest(model.policy)
    if model.num_timesteps != 1024 or model._n_updates != 512 or before == after or model.replay_buffer.rng_checks != 1024:
        raise AssertionError("Toy learning/update/observer readiness counts incorrect")
    trace = output / "toy_timeout_transitions.jsonl"
    rows = [json.loads(line) for line in trace.read_text().splitlines()]
    if [row["transition_number"] for row in rows] != list(range(32, 1025, 32)):
        raise AssertionError("All32 toy timeouts required")
    for row in rows:
        if (row["done"] != 1 or row["timeout"] != 1 or row["effective_done"] != 0 or not row["time_limit_truncated"]
                or row["terminal_observation_sha256"] != row["stored_next_observation_sha256"]
                or row["stored_next_observation_values"][0] != 1):
            raise AssertionError("Toy replay must store terminal observation, never reset observation")
    effective = td3.effective_parameters(model)
    write_new(output / "smoke_receipt.json", dict(passed=True, scope="Synthetic native CUDA readiness only, never a portfolio backtest",
        completed_at=td3.now(), device=device, runtime_packages=td3.shared.runtime_packages(), sb3_source_sha256=source["sb3_source_sha256"],
        source_manifest_sha256=digest(output / "source_manifest.json"), resource_sha256=digest(output / "resource_preflight.json"),
        runner_sha256=source["files"]["scripts/btceth_rl_td3.py"], smoke_source_sha256=digest(__file__),
        toy_steps=int(model.num_timesteps), actual_n_updates=int(model._n_updates), actor_update_count=int(model._n_updates // model.policy_delay),
        timeout_count=len(rows), rng_checks=int(model.replay_buffer.rng_checks), observation_dimension=27, action_dimension=2,
        episode_steps=32, parameter_sha256_before=before, parameter_sha256_after=after, effective_hyperparameters=effective,
        trace_sha256=digest(trace), elapsed_seconds=time.monotonic() - started, real_market_data_read=False,
        formal_training=False, formal_candidate_models_created=0, checkpoints_saved=0, financial_returns_produced=False,
        supervisor_pid=os.getppid(), child_pid=os.getpid()))
    env.close()
    print(json.dumps(dict(passed=True, toy_steps=1024, actual_n_updates=512, timeout_count=32, receipt_sha256=digest(output / "smoke_receipt.json"))), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("supervise", "_run"), required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allowance-source", type=Path)
    parser.add_argument("--allowance-sha256")
    args = parser.parse_args()
    (supervised if args.phase == "supervise" else toy_run)(args)


if __name__ == "__main__":
    main()

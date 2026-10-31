"""One UUID-isolated synthetic native SAC/TQC CUDA smoke worker.

Root orchestration runs one worker on each V100 by setting CUDA_VISIBLE_DEVICES
to that worker's expected UUID. This script dispatches no other process, reads
no market data, runs exactly 1,024 synthetic interactions, and submits no order.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_name] = "1"
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

import gymnasium as gym
import numpy as np

from scripts import single_pair_rl_models as models

EXPECTED_PACKAGES = {"torch":"2.14.0+cu126","numpy":"2.4.6","gymnasium":"1.2.3",
                     "stable-baselines3":"2.9.0","sb3-contrib":"2.9.0"}


class SyntheticPair(gym.Env):
    """Bounded numbers for shape/timeout/learner testing; no price feed."""
    observation_space = gym.spaces.Box(-10.,10.,(13,),dtype=np.float32)
    action_space = gym.spaces.Box(-1.,1.,(1,),dtype=np.float32)

    def __init__(self,symbol):
        if symbol not in models.PAIRS:
            raise ValueError("Only the declared single-pair identities are valid")
        self.SYMBOLS = (symbol,)
        self.total_steps = self.resets = 0

    def reset(self,*,seed=None,options=None):
        super().reset(seed=seed)
        self.position = 0
        self.resets += 1
        return np.zeros(13,dtype=np.float32),{}

    def step(self,action):
        self.position += 1
        self.total_steps += 1
        obs = np.zeros(13,dtype=np.float32)
        obs[0] = self.position/32
        obs[1] = float(action[0])
        obs[2] = np.sin(self.position/4)*.1
        reward = float(action[0])*.001
        return obs,reward,False,self.position == 32,{"date":str(self.total_steps)}


def process_gpu_uuids():
    result = subprocess.run(["nvidia-smi","--query-compute-apps=pid,gpu_uuid","--format=csv,noheader,nounits"],
                            capture_output=True,text=True,check=True,timeout=30)
    rows = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) == 2 and parts[0].isdigit() and int(parts[0]) == os.getpid():
            rows.append(parts[1])
    return sorted(set(rows))


def run_smoke(output,algorithm,gpu_uuid,*,seed=0,symbol="BTCUSDT"):
    output = Path(output)
    output.mkdir(parents=True,exist_ok=False)
    started = time.monotonic()
    models.write_new(output/"started.json",dict(started_at=models.now(),pid=os.getpid(),algorithm=algorithm,
        expected_gpu_uuid=gpu_uuid,visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        symbol=symbol,scope="synthetic only; no market loading or formal training",
        smoke_source_sha256=models.sha(__file__),adapter_sha256=models.sha(models.__file__)))
    try:
        if algorithm not in ("sac","tqc"):
            raise ValueError("This bounded first-round smoke is SAC/TQC only")
        if os.environ.get("CUDA_VISIBLE_DEVICES") != gpu_uuid or not gpu_uuid.startswith("GPU-"):
            raise ValueError("The expected physical UUID must exclusively select this worker's GPU")
        versions = models.runtime_versions()
        if versions != EXPECTED_PACKAGES:
            raise ValueError("Use the pinned V100 research runtime: "+json.dumps(versions))
        import torch
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("Exactly one usable CUDA device required, with no CPU fallback")
        if torch.get_num_interop_threads() != 1:
            torch.set_num_interop_threads(1)
        env = SyntheticPair(symbol)
        trace = output/"terminal_transitions.jsonl"
        model = models.make_model(algorithm,env,seed,"cuda:0",trace_path=trace)
        initial_digest = models.parameter_digest(model,algorithm)
        training = models.train_model(model,env,1024,seed,progress_path=output/"progress.jsonl")
        counters = training["counters"]
        if (env.total_steps != 1024 or counters["environment_steps"] != 1024
                or counters["replay_transitions"] != 1024 or counters["replay_add_calls"] != 1024
                or counters["gradient_update_rounds"] != 512
                or counters["optimizer_steps"].get("actor") != 512
                or counters["optimizer_steps"].get("critic") != 512):
            raise AssertionError("The native smoke must execute exactly 1,024 interactions and 512 actor/critic updates")
        boundaries = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines()]
        if len(boundaries) != 32 or any(
            row["timeout"] != 1 or row["effective_done"] != 0 or row["terminal_observation_values"][0] != 1
            or row["terminal_observation_values"] != row["stored_next_observation_values"]
            or row["terminal_observation_sha256"] != row["stored_next_observation_sha256"] for row in boundaries):
            raise AssertionError("A 32-day synthetic timeout must preserve its actual terminal next observation")
        if any(not torch.isfinite(value).all() for value in model.policy.state_dict().values()):
            raise AssertionError("All actually trained native model tensors must be finite")
        trained_digest = models.parameter_digest(model,algorithm)
        if initial_digest == trained_digest:
            raise AssertionError("The synthetic smoke did not actually change model parameters")
        test_obs = np.linspace(-.2,.2,13,dtype=np.float32)
        expected_action,_ = models.predict(model,test_obs,algorithm)
        checkpoint = models.save_model(model,output/"checkpoint",algorithm)
        original_trace = trace.read_bytes()
        restored = models.load_model(output/"checkpoint",algorithm,"cuda:0")
        loaded_action,_ = models.predict(restored,test_obs,algorithm)
        if not np.array_equal(expected_action,loaded_action) or trace.read_bytes() != original_trace:
            raise AssertionError("Checkpoint inference changed the native action or appended to training evidence")
        if models.parameter_digest(restored,algorithm) != trained_digest:
            raise AssertionError("Loaded checkpoint parameters differ from the actually trained policy")
        physical = process_gpu_uuids()
        if physical != [gpu_uuid]:
            raise AssertionError("Actual CUDA process allocation does not match the worker's declared physical GPU")
        result = dict(passed=True,completed_at=models.now(),elapsed_seconds=time.monotonic()-started,
            scope="synthetic shapes/real native CUDA updates/checkpoint/timeout only; no market backtest",
            algorithm=algorithm,seed=seed,symbol=symbol,pid=os.getpid(),gpu_uuid=gpu_uuid,
            observed_process_gpu_uuids=physical,device="cuda:0",device_name=torch.cuda.get_device_name(0),
            compute_capability=list(torch.cuda.get_device_capability(0)),cuda_runtime=torch.version.cuda,
            observation_shape=[13],action_shape=[1],runtime_packages=versions,
            initial_parameter_sha256=initial_digest,trained_parameter_sha256=trained_digest,
            restored_parameter_sha256=models.parameter_digest(restored,algorithm),
            action_before_load=expected_action.tolist(),action_after_load=loaded_action.tolist(),
            checkpoint_files=checkpoint["checkpoint_files"],checkpoint_manifest_sha256=models.sha(output/"checkpoint/single_pair_model.json"),
            training=training,timeout_boundaries_checked=len(boundaries),trace_sha256=models.sha(trace),
            single_pair_adapter_sha256=models.sha(models.__file__),smoke_source_sha256=models.sha(__file__),
            no_market_data=True,no_formal_training=True,no_multi_model_average=True)
        models.write_new(output/"result.json",result)
        return result
    except BaseException as error:
        models.write_new(output/"failure.json",dict(failed_at=models.now(),pid=os.getpid(),algorithm=algorithm,
            expected_gpu_uuid=gpu_uuid,error_type=type(error).__name__,error=str(error)))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",required=True)
    parser.add_argument("--algorithm",required=True,choices=("sac","tqc"))
    parser.add_argument("--gpu-uuid",required=True)
    parser.add_argument("--symbol",choices=models.PAIRS,default="BTCUSDT")
    parser.add_argument("--seed",choices=models.SEEDS,type=int,default=0)
    args = parser.parse_args()
    result = run_smoke(args.output,args.algorithm,args.gpu_uuid,seed=args.seed,symbol=args.symbol)
    print(json.dumps({key:result[key] for key in ("passed","algorithm","symbol","gpu_uuid","elapsed_seconds","trained_parameter_sha256")}),flush=True)


if __name__ == "__main__":
    main()

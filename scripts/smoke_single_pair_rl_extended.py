"""Bounded native CUDA checks for four additional independent-pair models.

Run one explicit worker per UUID; this script launches no worker and reads no
market. Formal recipes and frozen adapters remain unchanged. DroQ uses 5,002
synthetic interactions to cross its original 5,000-step learning warmup.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

for _name in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"):
    os.environ[_name] = "1"
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

import gymnasium as gym
import numpy as np

from scripts import single_pair_rl_models as models

ALGORITHMS = ("crossq","recurrent_ppo","td3","droq")
CORE_SHA256 = "1f7e7267b51465cc0d06c3af6d120b49c8d2868b6e18a55b1c56b37ed78966a9"
EXPECTED_PACKAGES = {"torch":"2.14.0+cu126","numpy":"2.4.6","gymnasium":"1.2.3",
                     "stable-baselines3":"2.9.0","sb3-contrib":"2.9.0"}


class SyntheticPair(gym.Env):
    observation_space = gym.spaces.Box(-10.,10.,(13,),dtype=np.float32)
    action_space = gym.spaces.Box(-1.,1.,(1,),dtype=np.float32)

    def __init__(self,symbol):
        if symbol not in models.PAIRS:
            raise ValueError("Only BTCUSDT or ETHUSDT is allowed")
        self.SYMBOLS = (symbol,)
        self.total_steps = self.resets = self.position = 0
        self.terminals = []
        self.raw_rewards = {}

    def reset(self,*,seed=None,options=None):
        super().reset(seed=seed)
        self.position = 0
        self.resets += 1
        return np.zeros(13,dtype=np.float32),{}

    def step(self,action):
        action = models.action(action)
        self.position += 1
        self.total_steps += 1
        obs = np.asarray([self.position/32,float(action[0]),*[
            np.sin((self.position+offset)/(offset+2))*.1 for offset in range(11)]],dtype=np.float32)
        reward = float(action[0])*.001
        self.raw_rewards[self.total_steps] = reward
        truncated = self.position == 32
        if truncated:
            self.terminals.append(dict(step=self.total_steps,observation=obs.tolist(),reward=reward))
        return obs,reward,False,truncated,{"date":str(self.total_steps)}


class RecurrentObservation:
    """Transparent observations of native time-limit value bootstrapping.

    Both wrappers return the original result unchanged and call the original
    method exactly once. They are removed before checkpoint serialization.
    """
    def __init__(self,model,env):
        self.model,self.env = model,env
        self.values,self.adds = [],[]
        self.terminal_values = {}
        self.last_add_state_shapes = None
        self.original_predict = model.policy.predict_values
        self.original_add = model.rollout_buffer.add
        if "predict_values" in model.policy.__dict__ or "add" in model.rollout_buffer.__dict__:
            raise ValueError("Do not stack observation wrappers")

        def observe_predict(*args,**kwargs):
            value = self.original_predict(*args,**kwargs)
            obs = args[0]
            if tuple(obs.shape) == (1,13) and float(obs[0,0].detach().cpu().item()) == 1.:
                states = args[1] if len(args) > 1 else kwargs["lstm_states"]
                starts = args[2] if len(args) > 2 else kwargs["episode_starts"]
                actual = float(value.reshape(-1)[0].detach().cpu().item())
                row = dict(step=env.total_steps,observation=obs.detach().cpu().numpy().reshape(13).tolist(),
                    state_shapes=[list(state.shape) for state in states],
                    episode_starts=starts.detach().cpu().numpy().reshape(-1).tolist(),value=actual)
                self.values.append(row)
                self.terminal_values[env.total_steps] = actual
            return value

        def observe_add(*args,**kwargs):
            position = model.rollout_buffer.pos
            result = self.original_add(*args,**kwargs)
            if env.total_steps % 32 == 0:
                actual = float(model.rollout_buffer.rewards[position,0])
                raw = float(np.float32(env.raw_rewards[env.total_steps]))
                terminal = self.terminal_values.get(env.total_steps)
                if terminal is None:
                    raise AssertionError("The native rollout add did not follow terminal-value evaluation")
                expected = raw+float(model.gamma)*terminal
                if not np.isclose(actual,expected,rtol=1e-6,atol=1e-7):
                    raise AssertionError("Native TimeLimit reward failed its real terminal-value bootstrap")
                self.adds.append(dict(step=env.total_steps,rollout_position=position,raw_reward=raw,
                    stored_reward=actual,terminal_value=terminal,expected_reward=expected,
                    stored_observation=model.rollout_buffer.observations[position,0].copy().tolist(),
                    episode_start=bool(model.rollout_buffer.episode_starts[position,0])))
                self.last_add_state_shapes = {name:list(getattr(model.rollout_buffer,name).shape) for name in
                    ("hidden_states_pi","cell_states_pi","hidden_states_vf","cell_states_vf")}
            return result

        model.policy.predict_values = observe_predict
        model.rollout_buffer.add = observe_add

    def close(self):
        del self.model.policy.predict_values
        del self.model.rollout_buffer.add

    def evidence(self):
        buffer = self.model.rollout_buffer
        if len(self.values) != 32 or len(self.adds) != 32:
            raise AssertionError("Native recurrent smoke must bootstrap all 32 actual TimeLimit boundaries")
        for row in self.values:
            if row["state_shapes"] != [[1,1,64],[1,1,64]] or row["episode_starts"] != [False]:
                raise AssertionError("Terminal value must use the carried native VF LSTM state")
        if any(len(row["observation"]) != 13 or row["observation"][0] != 1 for row in self.values):
            raise AssertionError("A reset observation cannot replace the actual recurrent terminal observation")
        starts = np.flatnonzero(np.asarray(buffer.episode_starts).reshape(-1)).tolist()
        if starts != [0,32,64,96]:
            raise AssertionError("Native last rollout must retain four true episode starts")
        # Native get() flattens the arrays for minibatches. Shape evidence must
        # come from the real add() before that expected native transformation.
        shapes = self.last_add_state_shapes
        if any(value != [128,1,1,64] for value in shapes.values()):
            raise AssertionError("Actual native recurrent rollout states have unexpected dimensions")
        return dict(terminal_values=self.values,terminal_adds=self.adds,rollout_state_shapes=shapes,
                    last_rollout_episode_starts=starts,terminal_bootstrap_checks=32,
                    original_methods_called_once=True,training_math_changed=False)


def assert_training_counters(algorithm,training,env):
    counters = training["counters"]
    expected_steps = 5002 if algorithm == "droq" else 1024
    if training["actual_steps"] != expected_steps or env.total_steps != expected_steps:
        raise AssertionError("Unexpected actual synthetic interaction count")
    if algorithm == "droq":
        expected = dict(environment_steps=5002,replay_adds=5002,random_exploration_actions=5001,
            q_inner_rounds=40,q_optimizer_steps=[40,40],total_q_optimizer_steps=80,
            policy_optimizer_steps=2,alpha_optimizer_steps=2,target_updates_per_network=[40,40],
            learning_train_calls=2)
        if counters != expected:
            raise AssertionError("Native full-recipe DroQ warmup/update counters differ: "+json.dumps(counters))
    elif algorithm == "recurrent_ppo":
        if (counters.get("environment_steps") != 1024 or counters.get("gradient_update_rounds") != 40
                or counters.get("optimizer_steps") != {"policy":40}):
            raise AssertionError("Eight complete recurrent rollouts must execute exactly 40 native optimizer updates")
    else:
        actor = 170 if algorithm == "crossq" else 256
        optimizers = counters["optimizer_steps"]
        if (counters.get("environment_steps") != 1024 or counters.get("gradient_update_rounds") != 512
                or counters.get("replay_add_calls") != 1024 or counters.get("replay_transitions") != 1024
                or optimizers.get("critic") != 512 or optimizers.get("actor") != actor
                or algorithm == "crossq" and optimizers.get("entropy") != 170):
            raise AssertionError("Native off-policy counters differ: "+json.dumps(counters))


def check_timeouts(algorithm,trace,env):
    if algorithm == "recurrent_ppo":
        return dict(boundaries=len(env.terminals),kind="native recurrent value bootstrap observations")
    rows = [json.loads(line) for line in Path(trace).read_text(encoding="utf-8").splitlines()]
    count = (5002 if algorithm == "droq" else 1024)//32
    if len(rows) != count or len(env.terminals) != count:
        raise AssertionError("Every actual native replay timeout must be observed")
    for row,emitted in zip(rows,env.terminals):
        if (row["terminal_observation_values"] != emitted["observation"]
                or row["stored_next_observation_values"] != emitted["observation"]
                or row["stored_next_observation_sha256"] != row["terminal_observation_sha256"]
                or row["effective_done"] != 0 or row["timeout"] != 1):
            raise AssertionError("TimeLimit replay stored a reset state or absorbed the market")
        if algorithm == "droq" and (row["stored_done"] != 0 or row["time_limit_truncated"] is not True):
            raise AssertionError("Native DroQ must bootstrap actual TimeLimit transitions")
    return dict(boundaries=count,kind="actual post-store replay values",trace_sha256=models.sha(trace))


def _recurrent_inference(model):
    observations = [np.linspace(-.2+i*.01,.2+i*.01,13,dtype=np.float32) for i in range(4)]
    actions,states = [],[]
    state = None
    for index,obs in enumerate(observations):
        act,state = models.predict(model,obs,"recurrent_ppo",state=state,episode_start=index == 0)
        if len(state) != 2 or any(value.shape != (1,1,64) or not np.isfinite(value).all() for value in state):
            raise AssertionError("Native recurrent action inference must preserve its real LSTM state")
        actions.append(act.copy())
        states.append([value.copy() for value in state])
    if not any(np.any(value != 0) for value in state):
        raise AssertionError("The recurrent policy did not carry any hidden state")
    fresh,new_state = models.predict(model,observations[0],"recurrent_ppo",state=None,episode_start=True)
    reset,reset_state = models.predict(model,observations[0],"recurrent_ppo",state=state,episode_start=True)
    if not np.array_equal(fresh,reset) or any(not np.array_equal(a,b) for a,b in zip(new_state,reset_state)):
        raise AssertionError("A new episode must actually reset the native LSTM state")
    return actions,states


def native_workload(output,algorithm,*,symbol="BTCUSDT",seed=0,device="cpu",upstream=None):
    """A bounded native workload; CPU access exists only for local unit tests."""
    import torch
    if algorithm not in ALGORITHMS:
        raise ValueError("Only the four declared extended algorithms are supported")
    if models.sha(models.__file__) != CORE_SHA256:
        raise ValueError("Preserve the already-frozen single-pair model adapter")
    if algorithm == "droq":
        if upstream is None:
            raise ValueError("DroQ smoke requires an explicit verified author upstream directory")
        models._droq().verify_upstream(upstream)
    output = Path(output)
    output.mkdir(parents=True,exist_ok=False)
    env = SyntheticPair(symbol)
    trace = output/"terminal_transitions.jsonl"
    model = models.make_model(algorithm,env,seed,device,upstream_root=upstream,trace_path=trace)
    initial = models.parameter_digest(model,algorithm)
    observed = RecurrentObservation(model,env) if algorithm == "recurrent_ppo" else None
    try:
        training = models.train_model(model,env,5002 if algorithm == "droq" else 1024,seed,progress_path=output/"progress.jsonl")
        recurrent = observed.evidence() if observed is not None else None
    finally:
        if observed is not None:
            observed.close()
    assert_training_counters(algorithm,training,env)
    timeouts = check_timeouts(algorithm,trace,env)
    if algorithm == "td3":
        values = models.effective_parameters(model,algorithm)
        if values["action_noise"]["mean"] != [0.] or values["action_noise"]["sigma"] != [.1]:
            raise AssertionError("TD3 exploration noise must be native one-dimensional noise")
    if algorithm == "droq" and (model.obs_dim,model.act_dim,model.target_entropy) != (13,1,-1):
        raise AssertionError("Native DroQ shapes/automatic entropy remain incompatible")
    state = model.policy.state_dict() if algorithm != "droq" else models._droq().parameter_tensors(model)
    if any(not torch.isfinite(value).all() for value in state.values()):
        raise AssertionError("Native learned tensors must be finite")
    trained = models.parameter_digest(model,algorithm)
    if initial == trained:
        raise AssertionError("A real optimizer update must change actual parameters")
    obs = np.linspace(-.2,.2,13,dtype=np.float32)
    before_action,_ = models.predict(model,obs,algorithm,episode_start=True)
    before_recurrent = _recurrent_inference(model) if algorithm == "recurrent_ppo" else None
    checkpoint = models.save_model(model,output/"checkpoint",algorithm,upstream_root=upstream)
    trace_bytes = trace.read_bytes() if trace.exists() else None
    restored = models.load_model(output/"checkpoint",algorithm,device,upstream_root=upstream)
    after_action,_ = models.predict(restored,obs,algorithm,episode_start=True)
    if not np.array_equal(before_action,after_action) or models.parameter_digest(restored,algorithm) != trained:
        raise AssertionError("The native single-pair checkpoint failed exact inference restoration")
    if (trace.read_bytes() if trace.exists() else None) != trace_bytes:
        raise AssertionError("Checkpoint inference appended to training replay evidence")
    if before_recurrent is not None:
        after_recurrent = _recurrent_inference(restored)
        for left,right in zip(before_recurrent[0],after_recurrent[0]):
            np.testing.assert_array_equal(left,right)
        for left,right in zip(before_recurrent[1],after_recurrent[1]):
            for a,b in zip(left,right):
                np.testing.assert_array_equal(a,b)
    evidence = dict(passed=True,algorithm=algorithm,symbol=symbol,seed=seed,actual_steps=training["actual_steps"],
        training=training,timeouts=timeouts,recurrent=recurrent,recipe=models.recipe(algorithm),
        formal_recipe_unchanged=True,synthetic_recipe_override=False,
        initial_parameter_sha256=initial,trained_parameter_sha256=trained,restored_parameter_sha256=models.parameter_digest(restored,algorithm),
        action_before_load=before_action.tolist(),action_after_load=after_action.tolist(),
        checkpoint_files=checkpoint["checkpoint_files"],checkpoint_manifest_sha256=models.sha(output/"checkpoint/single_pair_model.json"),
        no_market_data=True,no_formal_fit=True,adapter_sha256=models.sha(models.__file__),smoke_source_sha256=models.sha(__file__))
    models.write_new(output/"native_result.json",evidence)
    return evidence


def process_gpu_uuids():
    result = subprocess.run(["nvidia-smi","--query-compute-apps=pid,gpu_uuid","--format=csv,noheader,nounits"],
                            capture_output=True,text=True,check=True,timeout=30)
    return sorted({parts[1].strip() for line in result.stdout.splitlines()
                   if len(parts := line.split(",")) == 2 and parts[0].strip().isdigit() and int(parts[0].strip()) == os.getpid()})


def run_smoke(output,algorithm,gpu_uuid,*,symbol="BTCUSDT",seed=0,upstream=None):
    output = Path(output)
    output.mkdir(parents=True,exist_ok=False)
    started = time.monotonic()
    models.write_new(output/"started.json",dict(at=models.now(),pid=os.getpid(),algorithm=algorithm,expected_gpu_uuid=gpu_uuid,
        visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),scope="bounded native synthetic smoke only",adapter_sha256=models.sha(models.__file__)))
    try:
        if os.environ.get("CUDA_VISIBLE_DEVICES") != gpu_uuid or not gpu_uuid.startswith("GPU-"):
            raise ValueError("Select exactly the expected physical GPU UUID before importing Torch")
        versions = models.runtime_versions()
        if versions != EXPECTED_PACKAGES:
            raise ValueError("Pinned V100 runtime required: "+json.dumps(versions))
        import torch
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("Exactly one UUID-isolated CUDA GPU is required")
        if torch.get_num_interop_threads() != 1:
            torch.set_num_interop_threads(1)
        evidence = native_workload(output/"native",algorithm,symbol=symbol,seed=seed,device="cuda:0",upstream=upstream)
        physical = process_gpu_uuids()
        if physical != [gpu_uuid]:
            raise AssertionError("Actual CUDA process allocation differs from its declared GPU UUID")
        result = dict(**evidence,completed_at=models.now(),elapsed_seconds=time.monotonic()-started,pid=os.getpid(),gpu_uuid=gpu_uuid,
            observed_process_gpu_uuids=physical,device_name=torch.cuda.get_device_name(0),compute_capability=list(torch.cuda.get_device_capability(0)),
            runtime_packages=versions,cuda_runtime=torch.version.cuda)
        models.write_new(output/"result.json",result)
        return result
    except BaseException as error:
        models.write_new(output/"failure.json",dict(at=models.now(),algorithm=algorithm,expected_gpu_uuid=gpu_uuid,
            error_type=type(error).__name__,error=str(error)))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",required=True)
    parser.add_argument("--algorithm",required=True,choices=ALGORITHMS)
    parser.add_argument("--gpu-uuid",required=True)
    parser.add_argument("--symbol",choices=models.PAIRS,default="BTCUSDT")
    parser.add_argument("--seed",type=int,choices=models.SEEDS,default=0)
    parser.add_argument("--upstream",help="Mandatory explicit verified27 author directory for DroQ")
    args = parser.parse_args()
    result = run_smoke(args.output,args.algorithm,args.gpu_uuid,symbol=args.symbol,seed=args.seed,upstream=args.upstream)
    print(json.dumps({key:result[key] for key in ("passed","algorithm","symbol","gpu_uuid","elapsed_seconds","actual_steps","trained_parameter_sha256")}),flush=True)


if __name__ == "__main__":
    main()

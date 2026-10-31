"""Authorized synthetic DroQ CUDA readiness only; no market or portfolio run.

Importing this helper, ``--help`` and ``--phase contract`` use the standard
library only.  The run phases require a separately reviewed SHA-bound
authorization, a new preflight child directory, and an unfiltered GPU guard.
The pinned author and new runner supply all learning and storage mathematics.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import random
import re
import shutil
import struct
import subprocess
import sys
import time

SCOPE = "droq_native_synthetic_smoke"
STEPS, SEED, EPISODE_STEPS = 5120, 1, 32
COMMIT = "4c7adddbaf7573ff9256b16091a3bbdbd536aad4"
MANIFEST_SHA = "9b0eb6b5e08fefae4a468061ab31e93bcd4d67606750b6196e926f6e103e7560"
CONFIG_SHA = "d76e838b65d333c0669b5b138aa5bb8e81efe9e88921c25a0d00feb266828c1a"
DEPENDENCIES = ("scripts/__init__.py", "scripts/backtest_btceth_rl.py",
                "scripts/crypto_rl_reporting.py", "scripts/btceth_rl_droq.py",
                "scripts/audit_btceth_rl_droq.py", "scripts/smoke_btceth_rl_droq.py")
FORBIDDEN_MODULES = ("scripts.btceth_rl_env", "scripts.crypto_rl_portfolio_env")
FROZEN_IMPORTS = {
    "scripts/backtest_btceth_rl.py": "dbc7ffae69ddc1c7b043302d770b2c36066cd0b9c8577fa231d6d46d8e9e16a1",
    "scripts/crypto_rl_reporting.py": "b77e0c55d23a2adf65dd87aab8df322458754e6c908d4b2152305543af6ff6d5",
}
QUERY = ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"]


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


def write_new(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, allow_nan=False); stream.write("\n")


def publish(path, value):
    temporary = Path(str(path) + ".pending")
    write_new(temporary, value)
    os.replace(temporary, path)


def unlinked(path):
    path = Path(path).absolute()
    for item in (path, *path.parents):
        if item.is_symlink() or hasattr(item, "is_junction") and item.is_junction():
            raise ValueError("Synthetic source/output cannot cross links or junctions")
    return path.resolve()


def new_output(preflight, output):
    preflight, output = unlinked(preflight), unlinked(output)
    if (output.parent != preflight or not re.fullmatch(r"native_smoke_[A-Za-z0-9_-]+", output.name)
            or output.exists()):
        raise ValueError("Use a new direct native_smoke_* child of the isolated preflight")
    return output


def expected_counts(steps=STEPS):
    if type(steps) is not int or steps <= 0:
        raise ValueError("Positive integer synthetic steps required")
    learning = max(steps - 5000, 0)
    return dict(environment_steps=steps, replay_adds=steps,
        random_exploration_actions=min(steps, 5001), learning_train_calls=learning,
        q_inner_rounds=20 * learning, q_optimizer_steps=[20 * learning] * 2,
        total_q_optimizer_steps=40 * learning, target_updates_per_network=[20 * learning] * 2,
        policy_optimizer_steps=learning, alpha_optimizer_steps=learning)


def recipe():
    return dict(hidden_sizes=[256, 256], replay_size=1000000, batch_size=256,
        lr=.0003, gamma=.99, polyak=.995, alpha=.2, auto_alpha=True,
        target_entropy="auto", start_steps=5000, delay_update_steps="auto",
        utd_ratio=20, num_Q=2, num_min=2, q_target_mode="min",
        policy_update_delay=20, target_drop_rate=.01, layer_norm=True)


def validate_counts(actual, steps=STEPS):
    expected = expected_counts(steps)
    if any(actual.get(key) != value for key, value in expected.items()):
        raise ValueError("Actual Adam/driver/target counters differ from the fixed smoke")
    return expected


def validate_effective_parameters(actual):
    expected = {key: value for key, value in recipe().items() if key not in
                ("alpha", "target_entropy", "target_drop_rate", "layer_norm")}
    expected.update(delay_update_steps=5000, target_entropy=-2., alpha_current=.2,
        log_alpha_current=0., dropout_per_q=[[.01, .01], [.01, .01]],
        layer_norm_per_q=[[[256], [256]], [[256], [256]]], target_training=[True, True])
    if any(actual.get(key) != value for key, value in expected.items()):
        raise ValueError("Actual original Dropout/LayerNorm/alpha/optimizer recipe differs")
    return expected


def verify_official(upstream):
    upstream = unlinked(upstream)
    if sha(upstream / "verified_manifest.json") != MANIFEST_SHA:
        raise ValueError("Reviewed official27 source manifest required")
    manifest = read_json(upstream / "verified_manifest.json")
    if (manifest.get("commit") != COMMIT or manifest.get("license") != "MIT"
            or manifest.get("file_count") != 27 or len(manifest.get("files", [])) != 27):
        raise ValueError("Exact author commit and MIT inventory required")
    result = {}
    for item in manifest["files"]:
        rel = PurePosixPath(item["path"])
        if rel.is_absolute() or ".." in rel.parts or ":" in str(rel):
            raise ValueError("Official source paths must remain relative")
        path = unlinked(upstream / rel)
        path.relative_to(upstream)
        content = path.read_bytes()
        blob = hashlib.sha1(b"blob " + str(len(content)).encode() + b"\0" + content).hexdigest()
        if len(content) != item["size"] or sha(path) != item["sha256"] or blob != item["git_blob_sha1"]:
            raise ValueError("Official author source changed: " + str(rel))
        result[str(rel)] = item["sha256"]
    return result


def source_hashes(source_root):
    root = unlinked(source_root)
    values = {name: sha(unlinked(root / name)) for name in DEPENDENCIES}
    if any(values.get(name) != value for name, value in FROZEN_IMPORTS.items()):
        raise ValueError("Older shared financial sources must remain unchanged")
    if values["scripts/smoke_btceth_rl_droq.py"] != sha(__file__):
        raise ValueError("Executing smoke must be the reviewed source-root file")
    return values


def validate_authorization(auth, *, hashes, gpu_uuid, allowance_sha, output_name, preflight_root):
    if (auth.get("scope") != SCOPE or auth.get("authorized") is not True
            or any(auth.get(key) is not False for key in
                ("market_training_authorized", "validation_authorized", "annual_evaluation_authorized"))
            or auth.get("synthetic_steps") != STEPS or auth.get("seed") != SEED
            or auth.get("gpu_uuid") != gpu_uuid or auth.get("output_relative_name") != output_name
            or auth.get("preflight_root") != str(unlinked(preflight_root))
            or auth.get("source_sha256") != hashes or auth.get("resource_allowance_sha256") != allowance_sha
            or auth.get("upstream_manifest_sha256") != MANIFEST_SHA
            or auth.get("effective_config_contract_sha256") != CONFIG_SHA
            or auth.get("hyperparameters") != recipe()):
        raise ValueError("Separate exact-source synthetic-only authorization required")


def process_identity(pid):
    import pwd
    root = Path("/proc") / str(int(pid))
    uid = int(next(line for line in (root / "status").read_text().splitlines() if line.startswith("Uid:")).split()[1])
    return dict(pid=int(pid), uid=uid, username=pwd.getpwuid(uid).pw_name,
        exe=os.readlink(root / "exe"), starttime_ticks=(root / "stat").read_text().rsplit(")", 1)[1].split()[19],
        boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip())


def resource_rows(stdout):
    rows = []
    for line in stdout.splitlines():
        if line.strip():
            fields = [part.strip() for part in line.split(",")]
            if len(fields) != 2 or not fields[0].startswith("GPU-"):
                raise ValueError("Unexpected unfiltered GPU process response")
            rows.append(dict(gpu_uuid=fields[0], pid=int(fields[1])))
    if len({(r["gpu_uuid"], r["pid"]) for r in rows}) != len(rows):
        raise ValueError("Duplicate unfiltered process rows")
    return rows


def verify_resource(rows, allowance, inspect_process=process_identity, child=None):
    if allowance.get("scope") != SCOPE:
        raise ValueError("Resource allowance must be specific to this synthetic smoke")
    instance, expected = allowance.get("allowed_instance"), list(allowance.get("allowed_gpu_rows", []))
    if instance is not None:
        if (instance.get("username") != "sunny" or Path(instance.get("exe", "")).name != "ptyxis"
                or inspect_process(instance["pid"]) != instance
                or not expected or any(row["pid"] != instance["pid"] for row in expected)):
            raise ValueError("Reviewed original ptyxis instance identity changed")
    elif expected:
        raise ValueError("Existing GPU rows require their precise instance identity")
    if child is not None:
        expected.append(child)
    key = lambda row: (row["gpu_uuid"], row["pid"])
    if Counter(map(key, rows)) != Counter(map(key, expected)):
        raise ValueError("Unknown GPU task: never filter, stop or terminate it")


def f32_sha(values):
    if len(values) != 27 or not all(math.isfinite(float(v)) for v in values):
        raise ValueError("Exactly27 finite synthetic observations required")
    return hashlib.sha256(struct.pack("<27f", *values)).hexdigest()


def validate_timeouts(rows, steps=STEPS):
    if [row.get("transition_number") for row in rows] != list(range(32, steps + 1, 32)):
        raise ValueError("Every synthetic32-step truncation must be retained")
    for ordinal, row in enumerate(rows, 1):
        if (any(row.get(k) != v for k, v in dict(done=1., timeout=1., effective_done=0., stored_done=0.,
                time_limit_truncated=True, terminated=False, truncated=True).items())
                or row.get("info_date") != f"synthetic_episode_{ordinal}_step_32"
                or row["replay_position"] != ordinal * 32 - 1
                or row["terminal_observation_values"] != row["stored_next_observation_values"]
                or row["terminal_observation_values"][0] != 1.
                or row["stored_observation_values"][0] != 31 / 32):
            raise ValueError("Replay must retain actual terminal next observation, never reset observation")
        for name in ("terminal_observation", "stored_next_observation", "stored_observation"):
            if f32_sha(row[name + "_values"]) != row[name + "_sha256"]:
                raise ValueError("Synthetic float32 observation SHA differs")


def load_runner(root):
    root = unlinked(root)
    old_path, old_bytecode = list(sys.path), sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        sys.path.insert(0, str(root))
        module = importlib.import_module("scripts.btceth_rl_droq")
    finally:
        sys.path[:] = old_path; sys.dont_write_bytecode = old_bytecode
    if Path(module.__file__).resolve() != root / "scripts/btceth_rl_droq.py":
        raise ValueError("Foreign runner import cache")
    if any(name in sys.modules for name in FORBIDDEN_MODULES):
        raise ValueError("Synthetic smoke must not import a financial environment")
    return module


def wait_for_child_claim(path, *, child_pid=None, supervisor_pid=None, reader=read_json,
                         clock=time.monotonic, sleep=time.sleep, timeout=3.):
    """Wait only for the parent's atomic PID publication, with no NN work."""
    child_pid = os.getpid() if child_pid is None else child_pid
    supervisor_pid = os.getppid() if supervisor_pid is None else supervisor_pid
    deadline = clock() + timeout
    while True:
        proof = reader(path)
        if proof.get("phase") != SCOPE or proof.get("supervisor_pid") != supervisor_pid:
            raise ValueError("A matching actual synthetic supervisor must claim this child")
        if proof.get("status") == "running":
            if proof.get("child_pid") != child_pid:
                raise ValueError("Another child owns this synthetic attempt")
            return proof
        if (proof.get("status") != "launching" or proof.get("child_pid") not in (None, child_pid)
                or clock() >= deadline):
            raise ValueError("Matching child PID publication did not complete; no NN work started")
        sleep(.02)


def common_preflight(args):
    hashes = source_hashes(args.source_root)
    auth_path, allowance_path, config_path = map(unlinked, (args.authorization, args.allowance_source, args.effective_config))
    if sha(auth_path) != args.authorization_sha256 or sha(allowance_path) != args.allowance_sha256 or sha(config_path) != CONFIG_SHA:
        raise ValueError("Reviewed authorization, resource or source-configuration SHA changed")
    auth, allowance = read_json(auth_path), read_json(allowance_path)
    config = read_json(config_path)
    if config.get("effective_constructor_kwargs") != recipe():
        raise ValueError("Fixed DroQ recipe changed")
    official = verify_official(args.upstream_root)
    validate_authorization(auth, hashes=hashes, gpu_uuid=args.gpu_uuid,
        allowance_sha=args.allowance_sha256, output_name=Path(args.output_dir).name, preflight_root=args.preflight_root)
    if unlinked(args.output_dir).parent != unlinked(args.preflight_root):
        raise ValueError("Only isolated preflight output permitted")
    if unlinked(args.upstream_root) != unlinked(args.preflight_root) / "upstream/DroQ":
        raise ValueError("Official sources must belong to this isolated preflight")
    return hashes, official, auth, allowance


def supervised(args):
    output = new_output(args.preflight_root, args.output_dir)
    hashes, official, auth, allowance = common_preflight(args)
    raw = subprocess.check_output(QUERY, text=True)
    verify_resource(resource_rows(raw), allowance)
    gpu_command = ["nvidia-smi", "--query-gpu=index,uuid,name", "--format=csv,noheader"]
    gpu_stdout = subprocess.check_output(gpu_command, text=True)
    gpus = [dict(index=int(v[0]), uuid=v[1].strip(), name=v[2].strip()) for line in gpu_stdout.splitlines()
            if line.strip() for v in [[part.strip() for part in line.split(",")]]]
    if not any(g["uuid"] == args.gpu_uuid and "V100" in g["name"] for g in gpus):
        raise ValueError("Explicit actual V100 UUID required")
    output.mkdir()
    for name, source in (("authorization.json", args.authorization), ("resource_allowance.json", args.allowance_source),
                         ("effective_config_contract.json", args.effective_config)):
        shutil.copyfile(source, output / name)
    source = dict(files=hashes, official_files=official, upstream_root=str(unlinked(args.upstream_root)),
                  source_root=str(unlinked(args.source_root)), upstream_manifest_sha256=MANIFEST_SHA)
    write_new(output / "source_manifest.json", source)
    write_new(output / "resource_preflight.json", dict(at=now(), unfiltered_command=QUERY, unfiltered_stdout=raw,
        physical_gpu_command=gpu_command, physical_gpu_stdout=gpu_stdout, selected_uuid=args.gpu_uuid))
    command = [sys.executable, "-u", str(Path(__file__).resolve()), "--phase", "_run"]
    for name in ("source_root", "upstream_root", "preflight_root", "output_dir", "authorization",
                 "allowance_source", "effective_config", "authorization_sha256", "allowance_sha256", "gpu_uuid"):
        command.extend(["--" + name.replace("_", "-"), str(getattr(args, name))])
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu_uuid, CUBLAS_WORKSPACE_CONFIG=":4096:8",
        OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
    process = dict(phase=SCOPE, status="launching", started_at=now(), supervisor_pid=os.getpid(),
        source_manifest_sha256=sha(output / "source_manifest.json"), authorization_sha256=sha(output / "authorization.json"),
        resource_allowance_sha256=sha(output / "resource_allowance.json"), resource_preflight_sha256=sha(output / "resource_preflight.json"),
        smoke_sha256=sha(__file__), command=command, log_relative_path="smoke.log", formal_training=False)
    proof, log = output / "smoke_process.json", output / "smoke.log"
    write_new(proof, process)
    with log.open("x", encoding="utf-8") as stream:
        child = subprocess.Popen(command, env=env, stdout=stream, stderr=subprocess.STDOUT)
        process.update(status="running", child_pid=child.pid); publish(proof, process)
        print(json.dumps(dict(supervisor_pid=os.getpid(), child_pid=child.pid, gpu_uuid=args.gpu_uuid)), flush=True)
        code = child.wait()
    # Persist the actual terminal first; later provenance checks cannot erase it.
    process.update(status="completed" if code == 0 else "failed", exit_code=code, completed_at=now())
    publish(proof, process)
    process["log_sha256"] = sha(log); publish(proof, process)
    if code:
        raise RuntimeError("Synthetic smoke failed; preserve this attempt and its actual exit")
    common_preflight(args)
    verify_resource(resource_rows(subprocess.check_output(QUERY, text=True)), allowance)
    receipt = read_json(output / "smoke_receipt.json")
    if receipt.get("passed") is not True or receipt.get("smoke_source_sha256") != sha(__file__):
        raise ValueError("Actual child readiness receipt required")
    write_new(output / "supervision_receipt.json", dict(passed=True, at=now(), process_sha256=sha(proof),
        smoke_receipt_sha256=sha(output / "smoke_receipt.json"), source_unchanged=True, unfiltered_final_guard_passed=True))
    print(json.dumps(dict(passed=True, smoke_receipt_sha256=sha(output / "smoke_receipt.json"))), flush=True)


def child_run(args):
    output = unlinked(args.output_dir)
    hashes, official, _, allowance = common_preflight(args)
    proof = wait_for_child_claim(output / "smoke_process.json")
    source = read_json(output / "source_manifest.json")
    if (proof.get("status") != "running" or proof.get("child_pid") != os.getpid()
            or proof.get("supervisor_pid") != os.getppid() or proof.get("smoke_sha256") != sha(__file__)
            or proof.get("source_manifest_sha256") != sha(output / "source_manifest.json")
            or proof.get("authorization_sha256") != sha(output / "authorization.json")
            or proof.get("resource_allowance_sha256") != sha(output / "resource_allowance.json")
            or proof.get("resource_preflight_sha256") != sha(output / "resource_preflight.json")
            or sha(output / "authorization.json") != args.authorization_sha256
            or sha(output / "resource_allowance.json") != args.allowance_sha256
            or sha(output / "effective_config_contract.json") != CONFIG_SHA
            or source.get("files") != hashes or source.get("official_files") != official
            or source.get("source_root") != str(unlinked(args.source_root))
            or source.get("upstream_root") != str(unlinked(args.upstream_root))
            or read_json(output / "resource_preflight.json").get("selected_uuid") != args.gpu_uuid
            or os.environ.get("CUDA_VISIBLE_DEVICES") != args.gpu_uuid):
        raise ValueError("Actual exact-source unique authorized supervisor/child required")
    # No Torch/NumPy/Gym import or NN construction occurs before the gates above.
    import numpy as np
    import torch
    import gymnasium as gym
    if torch.cuda.is_initialized() or torch.__version__ != "2.14.0+cu126":
        raise ValueError("Fresh native Torch2.14 CUDA worker required")
    runner = load_runner(args.source_root)
    if runner.recipe() != recipe():
        raise ValueError("Actual runner recipe differs from smoke authorization")
    torch.set_num_threads(1); random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED); torch.use_deterministic_algorithms(True)
    if (torch.cuda.device_count() != 1 or list(torch.cuda.get_device_capability(0)) != [7, 0]
            or "V100" not in torch.cuda.get_device_name(0)):
        raise ValueError("Exactly one explicitly UUID-isolated V100 sm70 required")

    class CountingBox(gym.spaces.Box):
        def sample(self, *values, **kwargs):
            self.sample_calls = getattr(self, "sample_calls", 0) + 1
            return super().sample(*values, **kwargs)

    class Synthetic(gym.Env):
        def __init__(self):
            self.observation_space = gym.spaces.Box(-10, 10, shape=(27,), dtype=np.float32)
            self.action_space = CountingBox(-1, 1, shape=(2,), dtype=np.float32)
            self.episode, self.steps = 0, 0
        def obs(self):
            value = np.linspace(-.5, .5, 27, dtype=np.float32)
            value[0], value[1] = self.steps / 32, self.episode / 1000
            return value
        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed); self.episode += 1; self.steps = 0
            return self.obs(), {}
        def step(self, action):
            self.steps += 1
            reward = -float(np.square(np.asarray(action, dtype=float) - [.2, -.2]).mean())
            return self.obs(), reward, False, self.steps == 32, {"date": f"synthetic_episode_{self.episode}_step_{self.steps}"}

    def rng_digest():
        digest = hashlib.sha256(); state = np.random.get_state()
        digest.update(repr(random.getstate()).encode())
        digest.update(state[0].encode()); digest.update(state[1].tobytes()); digest.update(repr(state[2:]).encode())
        digest.update(torch.get_rng_state().numpy().tobytes())
        digest.update(torch.cuda.get_rng_state(0).cpu().numpy().tobytes())
        return digest.hexdigest()

    stats = dict(replay_rng_checks=0, hook_rng_checks=0, api_train_calls=0, warmup_noop_train_calls=0,
        target_forward_calls=[0, 0], online_forward_calls=[0, 0], first_target_alphas=[], first_alpha_event=None,
        first_learning_post_alpha=None)
    def observed(read):
        before = rng_digest(); value = read()
        if rng_digest() != before:
            raise AssertionError("Observation consumed Torch/NumPy RNG")
        stats["hook_rng_checks"] += 1
        return value

    env = Synthetic(); agent = runner.make_agent(args.upstream_root, device="cuda:0")
    initial = runner.effective_parameters(agent)
    validate_effective_parameters(initial)
    runner.observe_agent(agent, output / "synthetic_timeout_transitions.jsonl")
    modules = runner.modules_for(agent); before_parameters = runner.parameter_digest(agent)
    for group, module in modules.items():
        drops = [float(m.p) for m in module.modules() if isinstance(m, torch.nn.Dropout)]
        norms = [list(m.normalized_shape) for m in module.modules() if isinstance(m, torch.nn.LayerNorm)]
        expected_drop, expected_norm = ([], []) if group == "policy" else ([.01, .01], [[256], [256]])
        if not module.training or drops != expected_drop or norms != expected_norm:
            raise AssertionError("Actual actor/onlineQ/targetQ architecture or training mode differs")
    original_store = agent.replay_buffer.store
    def observed_store(*values, **kwargs):
        before = rng_digest(); original_store(*values, **kwargs)
        if rng_digest() != before:
            raise AssertionError("Original store plus observational trace consumed RNG")
        stats["replay_rng_checks"] += 1
    agent.replay_buffer.store = observed_store
    hooks = []
    for family, key in (("target", "target_q"), ("online", "q")):
        for index in range(2):
            def hook(module, inputs, family=family, index=index):
                def read():
                    if (not module.training or any(not m.training or m.p != .01 for m in module.modules() if isinstance(m, torch.nn.Dropout))):
                        raise AssertionError("Online/target Dropout must stay in training mode")
                    stats[family + "_forward_calls"][index] += 1
                    if family == "target" and index == 0 and len(stats["first_target_alphas"]) < 20:
                        stats["first_target_alphas"].append(float(agent.alpha))
                observed(read)
            hooks.append(modules[key + str(index)].register_forward_pre_hook(hook))
    def alpha_pre(optimizer, values, kwargs):
        if stats["first_alpha_event"] is None:
            stats["first_alpha_event"] = observed(lambda: dict(alpha_before=float(agent.alpha),
                log_alpha_before=float(agent.log_alpha.detach().cpu().item()),
                q_optimizer_steps_before=[runner.optimizer_steps(opt) for opt in agent.q_optimizer_list]))
    hooks.append(agent.alpha_optim.register_step_pre_hook(alpha_pre))
    original_train = agent.train
    def observed_train(logger):
        before = observed(lambda: runner.optimizer_steps(agent.q_optimizer_list[0]))
        original_train(logger)
        after = observed(lambda: runner.optimizer_steps(agent.q_optimizer_list[0]))
        stats["api_train_calls"] += 1
        stats["warmup_noop_train_calls"] += int(after == before)
        if after > before and stats["first_learning_post_alpha"] is None:
            stats["first_learning_post_alpha"] = observed(lambda: dict(alpha=float(agent.alpha),
                exp_log_alpha=float(agent.log_alpha.detach().cpu().exp().item())))
    agent.train = observed_train
    started = time.monotonic()
    result = runner.drive_steps(agent, env, STEPS, seed=SEED, progress_path=output / "synthetic_progress.jsonl")
    counters = result["counters"]; validate_counts(counters)
    if (env.action_space.sample_calls != 5001 or stats["replay_rng_checks"] != STEPS
            or stats["api_train_calls"] != STEPS or stats["warmup_noop_train_calls"] != 5000
            or stats["target_forward_calls"] != [2400, 2400] or stats["online_forward_calls"] != [2520, 2520]
            or stats["first_target_alphas"] != [.2] * 20
            or stats["first_alpha_event"] != dict(alpha_before=.2, log_alpha_before=0., q_optimizer_steps_before=[19, 19])
            or stats["first_learning_post_alpha"]["alpha"] != stats["first_learning_post_alpha"]["exp_log_alpha"]
            or not .9 < stats["first_learning_post_alpha"]["alpha"] < 1.1):
        raise AssertionError("Real warmup/update/alpha/target-mode evidence differs")
    for handle in hooks:
        handle.remove()
    rows = [json.loads(line) for line in (output / "synthetic_timeout_transitions.jsonl").read_text(encoding="utf-8").splitlines()]
    validate_timeouts(rows)
    after_parameters = runner.parameter_digest(agent)
    if before_parameters == after_parameters:
        raise AssertionError("Real synthetic optimizer updates must change parameters")
    storage_rng = rng_digest()
    saved = runner.save_checkpoint(output / "synthetic_tensor_storage", agent, counters=counters, upstream=args.upstream_root)
    loaded = runner.load_policy(output / "synthetic_tensor_storage", upstream=args.upstream_root, device="cuda:0")
    if (runner.parameter_digest(loaded) != after_parameters or saved["parameter_sha256"] != after_parameters
            or runner.parameter_digest(agent) != after_parameters or storage_rng != rng_digest()):
        raise AssertionError("Pure-Tensor weights-only storage/load changed tensors or RNG")
    if any(name in sys.modules for name in FORBIDDEN_MODULES):
        raise AssertionError("Synthetic readiness must not import a financial environment")
    if source_hashes(args.source_root) != hashes or verify_official(args.upstream_root) != official:
        raise AssertionError("Source bytes changed during synthetic smoke")
    verify_resource(resource_rows(subprocess.check_output(QUERY, text=True)), allowance,
        child=dict(gpu_uuid=args.gpu_uuid, pid=os.getpid()))
    effective = runner.effective_parameters(agent)
    receipt = dict(passed=True, scope=SCOPE, completed_at=now(), supervisor_pid=os.getppid(), child_pid=os.getpid(),
        selected_uuid=args.gpu_uuid, device_name=torch.cuda.get_device_name(0), capability=list(torch.cuda.get_device_capability(0)),
        visible_device_count=torch.cuda.device_count(), torch_version=torch.__version__, numpy_version=np.__version__,
        gymnasium_version=gym.__version__, python_version=sys.version, source_manifest_sha256=sha(output / "source_manifest.json"),
        smoke_source_sha256=sha(__file__), runner_source_sha256=hashes["scripts/btceth_rl_droq.py"],
        independent_auditor_source_sha256=hashes["scripts/audit_btceth_rl_droq.py"],
        author_loaded_sources=runner.loaded_sources(args.upstream_root), hyperparameters=recipe(),
        initial_effective_parameters=initial, final_effective_parameters=effective, actual_counters=counters,
        actual_box_sample_calls=env.action_space.sample_calls, observation=stats, truncations=len(rows),
        timeout_trace_sha256=sha(output / "synthetic_timeout_transitions.jsonl"), observation_dimension=27, action_dimension=2,
        episode_steps=32, seed=SEED, parameter_sha256_before=before_parameters, parameter_sha256_after=after_parameters,
        pure_tensor_roundtrip_parameter_sha256=runner.parameter_digest(loaded), pure_tensor_roundtrip_rng_unchanged=True,
        storage_tensor_sha256=saved["tensor_sha256"], elapsed_seconds=time.monotonic() - started,
        real_market_data_read=False, financial_environment_imported=False, financial_returns_produced=False,
        formal_candidate_models_created=0, formal_training=False, portfolio_validation=False, annual_evaluation=False)
    write_new(output / "smoke_receipt.json", receipt)
    env.close()
    print(json.dumps(dict(passed=True, actual_counters=counters, truncations=len(rows))), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("contract", "supervise", "_run"), default="contract")
    for name in ("source-root", "upstream-root", "preflight-root", "output-dir", "authorization", "allowance-source", "effective-config"):
        parser.add_argument("--" + name, type=Path)
    for name in ("authorization-sha256", "allowance-sha256", "gpu-uuid"):
        parser.add_argument("--" + name)
    args = parser.parse_args()
    if args.phase == "contract":
        print(json.dumps(dict(scope=SCOPE, prepared_only=True, synthetic_steps=STEPS, seed=SEED,
            expected_counters=expected_counts(), expected_timeouts=160, hyperparameters=recipe(),
            required_authorization_keys=["scope", "authorized", "market_training_authorized", "validation_authorized",
                "annual_evaluation_authorized", "synthetic_steps", "seed", "gpu_uuid", "output_relative_name", "preflight_root", "source_sha256",
                "resource_allowance_sha256", "upstream_manifest_sha256", "effective_config_contract_sha256", "hyperparameters"],
            model_construction_executed=False, training_executed=False, market_data_read=False)))
        return
    if any(getattr(args, name) is None for name in ("source_root", "upstream_root", "preflight_root", "output_dir", "authorization",
            "allowance_source", "effective_config", "authorization_sha256", "allowance_sha256", "gpu_uuid")):
        parser.error("All source, authorization, resource and isolated-output arguments are required before execution")
    (supervised if args.phase == "supervise" else child_run)(args)


if __name__ == "__main__":
    main()

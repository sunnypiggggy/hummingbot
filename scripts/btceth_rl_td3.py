"""Independent finite TD3 study; source preparation alone authorizes no run.

The shared off-policy recipe is deliberate, not SB3's pure TD3 defaults.
Ten new policies use twice the old training budget. Four fixed validation
candidates choose one profile/budget before a separately authorized replay
of the already observed year. No old module globals or evidence are changed.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
import hashlib
import importlib.metadata
import json
import math
import multiprocessing as mp
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import time

for _thread_name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_thread_name] = "1"
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from scripts import backtest_btceth_rl as shared
from scripts.crypto_rl_reporting import canonical_csv, result_fingerprints, save_result, summarize

TRAIN, VALIDATION, TEST = shared.TRAIN, shared.VALIDATION, shared.TEST
SEEDS = (0, 1, 2, 3, 4)
PROFILES = {"return": {"drawdown_penalty": 0., "rebalance_band": .025},
            "balanced": {"drawdown_penalty": .25, "rebalance_band": .025}}
BUDGETS = {"raw": 1., "cap075": .75}
COSTS = copy.deepcopy(shared.COSTS)
STEPS, EPISODE_DAYS = 65536, 32
FLAGS = ("designed_after_first_annual", "test_year_already_seen", "not_fresh_holdout")
CASE_FILES = ("equity_1h.csv.gz", "fills.csv", "weights.csv", "boundary_equity.csv", "metadata.json", "fingerprints.json", "member_targets.csv", "replay.json")
GROUP_FILES = ("summary.csv", "daily_equity.csv", "monthly_returns.csv", "completion.json")
SOURCE_FILES = (*shared.SOURCE_FILES, "scripts/audit_btceth_rl.py", "scripts/audit_btceth_rl_exposure_cap.py",
                "scripts/audit_btceth_rl_execution_band.py", "scripts/btceth_rl_td3.py", "scripts/audit_btceth_rl_td3.py")
UPSTREAM = dict(project="DLR-RM/stable-baselines3", version="2.9.0",
    commit="8908708f10c8ff29759c67f55c8acb56cab27463",
    td3_sha256="45107ef95f85cca1224741b666a3de2afa57838a501cc301331fd7e9a41cc2e0",
    verified_manifest_sha256="7ea8c1248651934265a4fe087b1979c231f6b0ab51d2180c2c9b3a55121853d9",
    source="https://github.com/DLR-RM/stable-baselines3/blob/v2.9.0/stable_baselines3/td3/td3.py",
    documentation="https://stable-baselines3.readthedocs.io/en/v2.9.0/modules/td3.html")
now, sha, read_json = shared.now, shared.sha, shared.read_json


def relative(root, name):
    name = str(name).replace("\\", "/")
    parsed = PurePosixPath(name)
    if not name or parsed.is_absolute() or ".." in parsed.parts or ":" in name:
        raise ValueError("Evidence paths must remain relative to their explicit root")
    root = Path(root).resolve(); result = root
    for part in parsed.parts:
        result /= part
        if result.is_symlink() or (hasattr(result, "is_junction") and result.is_junction()):
            raise ValueError("Evidence cannot cross a link or junction")
    result.resolve().relative_to(root)
    return result


def write_new(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False); stream.write("\n")


def publish(path, value):
    path = Path(path); temporary = path.with_suffix(path.suffix + ".pending")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False); stream.write("\n")
    os.replace(temporary, path)


def pins(root, names):
    return {name: sha(relative(root, name)) for name in sorted(names)}


def verify_pins(root, values):
    if not isinstance(values, dict) or not values:
        raise ValueError("Nonempty immutable input SHA required")
    for name, expected in values.items():
        if sha(relative(root, name)) != expected:
            raise ValueError("Frozen evidence changed: " + name)


def financial_imports():
    """Byte pins must describe the actually imported financial modules."""
    for name in shared.SOURCE_FILES:
        module = sys.modules.get(name[:-3].replace("/", "."))
        if module is not None and Path(module.__file__).resolve() != (ROOT / name).resolve():
            raise ValueError("Financial import resolved outside the frozen source root: " + name)


def utc(value):
    value = pd.Timestamp(value)
    if value.tzinfo is None:
        raise ValueError("Explicit UTC timestamps required")
    return value.tz_convert("UTC")


def recipe():
    """JSON-safe shared skeleton, with only the single TD3 recipe declared."""
    return dict(policy="MlpPolicy", learning_rate=.0003, gamma=.995, buffer_size=100000,
        learning_starts=512, batch_size=128, tau=.005, train_freq=[1, "step"], gradient_steps=1,
        n_steps=1, policy_delay=2, target_policy_noise=.2, target_noise_clip=.5,
        optimize_memory_usage=False, replay_buffer_kwargs={"handle_timeout_termination": True},
        policy_kwargs=dict(net_arch=[128, 128], activation_fn="ReLU", optimizer_class="Adam",
            n_critics=2, share_features_extractor=False),
        action_noise=dict(class_name="NormalActionNoise", mean=[0., 0.], sigma=[.1, .1],
            dtype="float32",
            units="normalized Box[-1,1] action units, not portfolio percentage points"))


def shared_spec():
    return dict(algorithm="td3", profiles=copy.deepcopy(PROFILES), seeds=list(SEEDS), budgets=copy.deepcopy(BUDGETS),
        train=list(TRAIN), validation=list(VALIDATION), test=list(TEST), capital=10000., costs=copy.deepcopy(COSTS),
        timesteps_per_model=STEPS, old_timesteps_per_model=32768, training_budget_multiple=2,
        train_episode_days=EPISODE_DAYS, model_count=10, validation_cases=48, validation_repeats=48,
        test_cases=12, test_repeats=12, cash_floor=.05, max_asset_weight=.95, hyperparameters=recipe(),
        upstream=copy.deepcopy(UPSTREAM), recipe_is_pure_official_default=False,
        recipe_label="Original off-policy shared skeleton plus official TD3-specific defaults",
        training_budget="raw only; raw/.75 are two fixed validation execution overlays, not retrained policies",
        selection_score="mean(log1p(net_return)-0.5*abs(max_drawdown)-2*max(abs(max_drawdown)-0.20,0)) across base/stress",
        tie_break="raw before cap075, then profile lexical identifier",
        seed_selection=False, selection_uses_test=False, new_recipe_search=False, annual_requires_separate_authorization=True,
        **{flag: True for flag in FLAGS})


def library_sources():
    distribution = importlib.metadata.distribution("stable-baselines3")
    if distribution.version != UPSTREAM["version"]:
        raise ValueError("Use the reviewed SB3 2.9.0 implementation")
    names = ("stable_baselines3/td3/td3.py", "stable_baselines3/td3/policies.py", "stable_baselines3/common/off_policy_algorithm.py", "stable_baselines3/common/buffers.py", "stable_baselines3/common/noise.py")
    result = {name: sha(distribution.locate_file(name)) for name in names}
    if result["stable_baselines3/td3/td3.py"] != UPSTREAM["td3_sha256"]:
        raise ValueError("Installed TD3 differs from the reviewed official source")
    return result


def library_imports(expected):
    for name, digest in expected.items():
        module = sys.modules.get(name[:-3].replace("/", "."))
        if module is None or sha(module.__file__) != digest:
            raise ValueError("Actually imported SB3 implementation differs from reviewed source: " + name)


def separate(main, output):
    main, output = Path(main).resolve(), Path(output).resolve()
    if main == output or main.is_relative_to(output) or output.is_relative_to(main):
        raise ValueError("New TD3 experiment requires an independent sibling tree")
    return main, output


def baseline_files():
    folder = "evaluation/test/baselines"
    return {*(f"{folder}/{name}" for name in GROUP_FILES),
        *(f"{folder}/{cost}/{key}/{name}" for cost in COSTS for key in shared.BASELINES for name in CASE_FILES[:6])}


def upstream_files(root):
    root = Path(root)
    if sha(root / "verified_manifest.json") != UPSTREAM["verified_manifest_sha256"]:
        raise ValueError("Use the independently reviewed official source/MIT evidence")
    verified = read_json(root / "verified_manifest.json")
    if verified.get("commit") != UPSTREAM["commit"] or verified.get("license") != "MIT" or len(verified.get("files", {})) != 7:
        raise ValueError("Full pinned upstream commit and license evidence required")
    for name, value in verified["files"].items():
        path = relative(root, name); content = path.read_bytes()
        blob = hashlib.sha1(b"blob " + str(len(content)).encode() + b"\0" + content).hexdigest()
        if sha(path) != value["sha256"] or blob != value["git_blob_sha1"]:
            raise ValueError("Archived official file differs from reviewed source: " + name)
    return pins(root, {*verified["files"], "verified_manifest.json", "github_manifest.json", "installed_distribution_observation.json"})


def prepare(main, dataset, output, *, upstream_root, allowance_file=None):
    """Future explicit preparation entry. Never called by source/fixture checks."""
    main, output = separate(main, output); dataset = Path(dataset).resolve()
    if output.exists():
        raise FileExistsError("Preserve every attempted TD3 study")
    old = read_json(main / "frozen_spec.json")
    if old.get("timesteps_per_model") != 32768 or tuple(old.get("train", ())) != TRAIN or tuple(old.get("validation", ())) != VALIDATION or tuple(old.get("test", ())) != TEST:
        raise ValueError("Original split/budget must match the disclosed common experiment")
    verify_pins(ROOT, old["code_sha256"])
    if sha(dataset) != old["dataset_sha256"] or sha(dataset.parent / "manifest.json") != old["dataset_manifest_sha256"]:
        raise ValueError("Use the original frozen BTC/ETH data without conversion")
    runtime = read_json(main / "runtime.json"); gpus = runtime["physical_gpus"]
    if len(gpus) != 2 or len({gpu["uuid"] for gpu in gpus}) != 2 or any("V100" not in gpu["name"] for gpu in gpus):
        raise ValueError("Two original distinct physical V100 identities required")
    sources = pins(ROOT, SOURCE_FILES)
    upstream_hashes = upstream_files(upstream_root)
    installed = library_sources(); verified = read_json(Path(upstream_root) / "verified_manifest.json")
    if any(installed[name] != verified["files"][name]["sha256"] for name in installed):
        raise ValueError("All five installed SB3 files must match the official reviewed commit")
    spec = shared_spec()
    spec.update(schema_version=1, prepared_at=now(), original_frozen_spec_sha256=sha(main / "frozen_spec.json"),
        dataset_sha256=sha(dataset), dataset_manifest_sha256=sha(dataset.parent / "manifest.json"),
        original_input_sha256=pins(main, {"frozen_spec.json", "runtime.json", "independent_audit.json"} | baseline_files()), source_sha256=sources,
        physical_gpus=gpus, profile_gpu_assignments={profile: gpu["uuid"] for profile, gpu in zip(PROFILES, gpus)},
        runtime_packages=copy.deepcopy(shared.RUNTIME_PACKAGES), research_targets=old["research_targets"],
        sb3_source_sha256=installed, upstream_evidence_sha256=upstream_hashes,
        allowed_compute_instance=None, allowed_gpu_rows=[])
    if allowance_file is not None:
        allowance = read_json(allowance_file)
        spec.update(allowed_compute_instance=allowance["allowed_instance"], allowed_gpu_rows=allowance["allowed_gpu_rows"],
            resource_allowance_sha256=sha(allowance_file))
    output.mkdir(parents=True)
    for name in sources:
        target = output / "sources" / name; target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(ROOT / name, target)
    for name in upstream_hashes:
        target = output / "upstream" / name; target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(relative(upstream_root, name), target)
    if allowance_file is not None:
        shutil.copyfile(allowance_file, output / "resource_allowance.json")
    write_new(output / "protocol.json", spec)
    write_new(output / "protocol_receipt.json", dict(protocol_file="protocol.json", protocol_sha256=sha(output / "protocol.json"), sealed_at=now()))
    return spec


def protocol(main, dataset, output):
    financial_imports()
    main, output = separate(main, output)
    spec, receipt = read_json(output / "protocol.json"), read_json(output / "protocol_receipt.json")
    if any(spec.get(key) != value for key, value in shared_spec().items()):
        raise ValueError("Exactly one declared TD3 recipe, all profiles/seeds and fixed budgets required")
    if (receipt.get("protocol_file") != "protocol.json" or receipt.get("protocol_sha256") != sha(output / "protocol.json")
            or utc(receipt["sealed_at"]) < utc(spec["prepared_at"])
            or spec["original_frozen_spec_sha256"] != sha(main / "frozen_spec.json")
            or spec["dataset_sha256"] != sha(dataset) or spec["dataset_manifest_sha256"] != sha(Path(dataset).parent / "manifest.json")):
        raise ValueError("Protocol receipt, original source or dataset identity changed")
    verify_pins(main, spec["original_input_sha256"])
    verify_pins(ROOT, spec["source_sha256"]); verify_pins(output / "sources", spec["source_sha256"])
    verify_pins(output / "upstream", spec["upstream_evidence_sha256"])
    return spec


def cropped_data(data, end):
    """Slice before feature construction; never normalize with later samples."""
    cutoff = utc(end)
    return {name: value.loc[value.index < cutoff].copy() if name.startswith("hourly_") else value for name, value in data.items()}


def observation_values(value):
    array = np.asarray(value, dtype="<f4").reshape(-1)
    if array.shape != (27,) or not np.isfinite(array).all():
        raise ValueError("Exactly27 finite frozen features/ledger state required")
    return array.tolist(), hashlib.sha256(array.tobytes()).hexdigest()


def audited_replay_buffer_class(path, base_class=None):
    """Observe official stored transitions after super.add; change no arrays."""
    if base_class is None:
        from stable_baselines3.common.buffers import ReplayBuffer
        base_class = ReplayBuffer
    destination = Path(path)

    class AuditedReplayBuffer(base_class):
        def add(self, obs, next_obs, action, reward, done, infos):
            super().add(obs, next_obs, action, reward, done, infos)
            self.audit_transition_count = getattr(self, "audit_transition_count", 0) + 1
            if self.optimize_memory_usage or not self.handle_timeout_termination:
                raise ValueError("Original timeout bootstrap and explicit next-observation storage required")
            position = (self.pos - 1) % self.buffer_size
            for index, ended in enumerate(np.asarray(done).reshape(-1)):
                if not ended:
                    continue
                terminal, terminal_sha = observation_values(infos[index]["terminal_observation"])
                stored, stored_sha = observation_values(self.next_observations[position, index])
                prior, prior_sha = observation_values(self.observations[position, index])
                record = dict(transition_number=self.audit_transition_count, replay_position=position, env_index=index,
                    done=float(self.dones[position, index]), timeout=float(self.timeouts[position, index]),
                    effective_done=float(self.dones[position, index] * (1 - self.timeouts[position, index])),
                    time_limit_truncated=bool(infos[index].get("TimeLimit.truncated", False)), info_date=str(infos[index].get("date")),
                    terminal_observation_values=terminal, terminal_observation_sha256=terminal_sha,
                    stored_next_observation_values=stored, stored_next_observation_sha256=stored_sha,
                    stored_observation_values=prior, stored_observation_sha256=prior_sha)
                with destination.open("a", encoding="utf-8", newline="\n") as stream:
                    stream.write(json.dumps(record, allow_nan=False, separators=(",", ":")) + "\n")
    return AuditedReplayBuffer


def model_parameters(trace_path):
    from stable_baselines3.common.noise import NormalActionNoise
    from torch import nn
    from torch.optim import Adam
    values = recipe(); values.pop("policy"); values.pop("action_noise")
    values["train_freq"] = tuple(values["train_freq"])
    values["policy_kwargs"]["activation_fn"] = nn.ReLU
    values["policy_kwargs"]["optimizer_class"] = Adam
    values["action_noise"] = NormalActionNoise(mean=np.zeros(2), sigma=.1 * np.ones(2), dtype=np.float32)
    values["replay_buffer_class"] = audited_replay_buffer_class(trace_path)
    return values


def effective_parameters(model):
    """Read actual model/buffer/policy attributes without sampling any RNG."""
    result = {name: getattr(model, name) for name in ("learning_rate", "gamma", "buffer_size", "learning_starts", "batch_size", "tau",
        "gradient_steps", "n_steps", "policy_delay", "target_policy_noise", "target_noise_clip")}
    result.update(train_freq=[model.train_freq.frequency, model.train_freq.unit.value],
        optimize_memory_usage=model.replay_buffer.optimize_memory_usage,
        replay_buffer_kwargs=dict(handle_timeout_termination=model.replay_buffer.handle_timeout_termination),
        policy_kwargs=dict(net_arch=model.policy.net_arch, activation_fn=model.policy.activation_fn.__name__,
            optimizer_class=model.policy.optimizer_class.__name__, n_critics=len(model.critic.q_networks),
            share_features_extractor=model.policy.share_features_extractor),
        action_noise=dict(class_name=type(model.action_noise).__name__, mean=np.asarray(model.action_noise._mu).tolist(),
            sigma=np.asarray(model.action_noise._sigma).tolist(), dtype=np.dtype(model.action_noise._dtype).name))
    return result


def job_id(profile, seed):
    if profile not in PROFILES or seed not in SEEDS:
        raise ValueError("Only the declared profile and seed identities allowed")
    return f"td3__{profile}__seed{seed}"


def train_one(task):
    from stable_baselines3 import TD3
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.monitor import Monitor
    from scripts.btceth_rl_env import BTCETHPortfolioEnv
    output, main, dataset = map(Path, (task["output"], task["main"], task["dataset"]))
    spec = protocol(main, dataset, output); profile, seed = task["profile"], task["seed"]
    library_imports(spec["sb3_source_sha256"])
    directory = output / "training" / job_id(profile, seed)
    directory.mkdir(parents=True, exist_ok=False)
    device = shared.torch_setup(seed)
    if device["visible_gpu_uuid"] != spec["profile_gpu_assignments"][profile] or device["capability"] != [7, 0]:
        raise ValueError("Use the declared UUID-isolated V100, one profile per card")
    write_new(directory / "started.json", dict(at=now(), pid=os.getpid(), task=task, device=device, protocol_sha256=sha(output / "protocol.json")))
    started = time.monotonic(); progress = []
    data = cropped_data(shared.load_dataset(dataset), TRAIN[1])
    raw = BTCETHPortfolioEnv(data, *TRAIN, training=True, episode_days=EPISODE_DAYS, costs=COSTS["base"],
        capital=10000., record_history=False, **PROFILES[profile])
    env = Monitor(raw); trace = directory / "training_timeout_transitions.jsonl"

    class Progress(BaseCallback):
        def _on_step(self):
            if self.num_timesteps % 2048 == 0:
                record = dict(timesteps=int(self.num_timesteps), elapsed_seconds=time.monotonic() - started)
                progress.append(record); publish(directory / "progress.json", dict(at=now(), pid=os.getpid(), **record))
                print("TD3_TRAIN", job_id(profile, seed), self.num_timesteps, flush=True)
            return True
    try:
        model = TD3("MlpPolicy", env, seed=seed, device="cuda:0", verbose=0, **model_parameters(trace))
        model.learn(total_timesteps=STEPS, callback=Progress())
        if model.num_timesteps != STEPS:
            raise ValueError("Exactly65536 timesteps required, no early stopping or search")
        if model._n_updates != STEPS - recipe()["learning_starts"]:
            raise ValueError("All declared post-warmup critic updates required")
        checkpoint = directory / "model.zip"; model.save(checkpoint)
        record = dict(algorithm="td3", profile=profile, seed=seed, completed_at=now(), device=device,
            elapsed_seconds=time.monotonic() - started, actual_timesteps=int(model.num_timesteps),
            actual_n_updates=int(model._n_updates), actor_update_count=int(model._n_updates // model.policy_delay),
            policy_class=type(model.policy).__name__, training_buffer_class=type(model.replay_buffer).__name__,
            algorithm_class=type(model).__module__ + "." + type(model).__qualname__,
            effective_hyperparameters=effective_parameters(model),
            sb3_source_sha256=library_sources(),
            old_timesteps_per_model=32768, training_budget_multiple=2, recipe_is_pure_official_default=False,
            model_relative_path=checkpoint.relative_to(output).as_posix(), model_sha256=sha(checkpoint),
            parameter_sha256=shared.parameter_digest(model.policy), protocol_sha256=sha(output / "protocol.json"),
            timeout_trace_relative_path=trace.relative_to(output).as_posix(), timeout_trace_sha256=sha(trace),
            replay_buffer_semantics=dict(optimize_memory_usage=model.replay_buffer.optimize_memory_usage,
                handle_timeout_termination=model.replay_buffer.handle_timeout_termination,
                observations_captured_after_super_add=True), hyperparameters=recipe(),
            features=list(raw.FEATURE_NAMES), training_episode_history=list(raw.training_episode_history),
            training_window_quality=raw.training_window_metadata, **{flag: True for flag in FLAGS})
        write_new(directory / "record.json", record)
        pd.DataFrame(progress).to_csv(directory / "training_metrics.csv", index=False)
        return record
    except BaseException as error:
        write_new(directory / "failure.json", dict(at=now(), type=type(error).__name__, error=str(error), pid=os.getpid()))
        raise
    finally:
        env.close()


def training_files(output):
    output = Path(output); done = read_json(output / "training_completion.json")
    if done.get("model_count") != 10:
        raise ValueError("All ten new TD3 policies must complete before validation")
    names = {"training_completion.json"}
    for profile in PROFILES:
        for seed in SEEDS:
            folder = "training/" + job_id(profile, seed); record = read_json(output / folder / "record.json")
            if ((record.get("algorithm"), record.get("profile"), record.get("seed"), record.get("actual_timesteps")) != ("td3", profile, seed, STEPS)
                    or record.get("protocol_sha256") != sha(output / "protocol.json") or record.get("hyperparameters") != recipe()):
                raise ValueError("Training identity/budget/recipe changed")
            for name, expected in ((record["model_relative_path"], record["model_sha256"]),
                    (record["timeout_trace_relative_path"], record["timeout_trace_sha256"])):
                if sha(relative(output, name)) != expected:
                    raise ValueError("Trained checkpoint or timeout evidence changed")
                names.add(name)
            names.add(folder + "/record.json")
    return pins(output, names)


def map_target(raw, family):
    raw = np.asarray(raw, dtype=float)
    if family not in BUDGETS or raw.shape != (3,) or not np.isfinite(raw).all() or np.any(raw < 0) or not np.isclose(raw.sum(), 1):
        raise ValueError("Valid original BTC/ETH/cash target and declared budget required")
    target = raw.copy(); target[:2] *= BUDGETS[family]; target[2] = 1 - target[:2].sum()
    return target


def evaluate(data, models, seeds, profile, family, start, end, costs):
    from scripts.btceth_rl_env import BTCETHPortfolioEnv
    financial_imports()
    if profile not in PROFILES or family not in BUDGETS or len(models) != len(seeds) or len(models) not in (1, 5) or len(set(seeds)) != len(seeds) or not set(seeds).issubset(SEEDS) or len(models) == 5 and set(seeds) != set(SEEDS):
        raise ValueError("One fixed seed or all five original TD3 policies required")
    before = [shared.parameter_digest(model.policy) for model in models]
    env = BTCETHPortfolioEnv(data, start, end, costs=costs, capital=10000., record_history=True, **PROFILES[profile])
    obs, _ = env.reset(seed=0); rows = []; done = False
    while not done:
        date = str(env.current_date); observation, observation_sha = observation_values(obs)
        targets, members = [], []
        for seed, model in zip(seeds, models):
            action, _ = model.predict(obs, deterministic=True)
            action = np.asarray(action, dtype=float).reshape(-1)
            if action.shape != (2,) or not np.isfinite(action).all():
                raise ValueError("Finite two-action TD3 output required")
            raw = env.action_to_weights(action); target = map_target(raw, family); targets.append(target)
            members.append(dict(day=date, seed=seed, raw_action0=float(action[0]), raw_action1=float(action[1]),
                **dict(zip(("raw_btc", "raw_eth", "raw_cash"), raw)), **dict(zip(("capped_btc", "capped_eth", "capped_cash"), target)),
                **dict(zip(("mapped_btc", "mapped_eth", "mapped_cash"), target)),
                observation_sha256=observation_sha, observation_values_json=json.dumps(observation, separators=(",", ":")),
                observation_asof=str(env.current_date - pd.Timedelta(hours=1)), observation_cutoff_exclusive=date,
                **dict(zip(("observation_btc_weight", "observation_eth_weight", "observation_cash_weight", "observation_drawdown"), observation[-4:]))))
        mean = np.mean(targets, axis=0)
        for member in members:
            member.update(dict(zip(("target_btc", "target_eth", "target_cash"), mean))); rows.append(member)
        obs, _, terminated, truncated, _ = env.step_target(mean); done = bool(terminated or truncated)
    after = [shared.parameter_digest(model.policy) for model in models]
    if before != after:
        raise ValueError("Evaluation cannot modify policy parameters")
    result = env.get_result(); env.close()
    result["metadata"].update(algorithm="td3", profile=profile, family=family, model_count=len(models),
        parameter_sha256_before=before, parameter_sha256_after=after, exposure_multiplier=BUDGETS[family],
        old_actions_or_equity_scaled=False, training_budget_multiple=2, deterministic_prediction=True,
        ensemble="mean targets in one real funded10,000 USDT account", **{flag: True for flag in FLAGS})
    return result, pd.DataFrame(rows)


def process_identity(pid):
    import pwd
    root = Path("/proc") / str(int(pid))
    uid = int(next(line for line in (root / "status").read_text().splitlines() if line.startswith("Uid:")).split()[1])
    return dict(pid=int(pid), uid=uid, username=pwd.getpwuid(uid).pw_name, exe=os.readlink(root / "exe"),
        starttime_ticks=(root / "stat").read_text().rsplit(")", 1)[1].split()[19],
        boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip())


def resource_rows(stdout):
    result = []
    for line in stdout.splitlines():
        if line.strip():
            uuid, pid = [value.strip() for value in line.split(",")]
            result.append(dict(gpu_uuid=uuid, pid=int(pid)))
    return result


def verify_resource_rows(rows, allowance, inspect_process=process_identity):
    allowed = allowance.get("allowed_compute_instance")
    expected = allowance.get("allowed_gpu_rows", [])
    if not rows:
        return
    if allowed is None or rows != expected or inspect_process(allowed["pid"]) != allowed:
        raise RuntimeError("Unexpected existing GPU process; never filter or terminate another task")


def gpu_preflight(spec):
    gpus = shared.gpu_inventory()
    if [gpu["uuid"] for gpu in gpus] != [gpu["uuid"] for gpu in spec["physical_gpus"]]:
        raise ValueError("Original physical GPU identities changed")
    command = ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"]
    stdout = subprocess.check_output(command, text=True)
    verify_resource_rows(resource_rows(stdout), spec)
    if shared.runtime_packages() != spec["runtime_packages"] or library_sources() != spec["sb3_source_sha256"]:
        raise ValueError("Reviewed runtime and upstream implementation required")
    return dict(at=now(), physical_gpus=gpus, command=command, unfiltered_stdout=stdout,
        allowed_compute_instance=spec.get("allowed_compute_instance"), allowed_gpu_rows=spec.get("allowed_gpu_rows", []))


def terminal(output, phase):
    path = Path(output) / (phase + "_process.json"); value = read_json(path)
    if (value.get("phase") != phase or value.get("status") != "completed" or value.get("exit_code") != 0
            or not isinstance(value.get("supervisor_pid"), int) or value["supervisor_pid"] <= 0
            or not isinstance(value.get("child_pid"), int) or value["child_pid"] <= 0 or value["supervisor_pid"] == value["child_pid"]
            or value.get("protocol_sha256") != sha(Path(output) / "protocol.json")
            or value.get("source_sha256") != sha(Path(__file__))
            or value.get("log_sha256") != sha(relative(output, value["log_relative_path"]))):
        raise ValueError("Real completed0 process and exact terminal log required: " + phase)
    return value


def supervise(main, dataset, output, phase, *, authorization=None):
    output = Path(output); spec = protocol(main, dataset, output)
    if phase not in ("fit", "validate", "test"):
        raise ValueError("Only explicit fit/validate/test process phases supported")
    if phase == "validate":
        terminal(output, "fit"); training_files(output)
    if phase == "test":
        annual_gate(main, dataset, output, authorization, require_new=True)
    for name in (phase + "_process.json", phase + ".log", phase + "_completion.json"):
        if (output / name).exists():
            raise FileExistsError("Preserve existing phase evidence; investigate before retry: " + name)
    if (output / ("training" if phase == "fit" else "validation" if phase == "validate" else "test")).exists():
        raise FileExistsError("Preserve existing phase account/model tree; no silent restart")
    resource = gpu_preflight(spec)
    write_new(output / "preflight" / (phase + ".json"), resource)
    if phase == "test":
        shutil.copyfile(authorization, output / "annual_authorization.json")
        write_new(output / "test_started.json", dict(at=now(), protocol_sha256=sha(output / "protocol.json"),
            selection_sha256=sha(output / "selection.json"), authorization_sha256=sha(output / "annual_authorization.json")))
    command = [sys.executable, "-u", str(Path(__file__).resolve()), "--phase", "_" + phase,
        "--reference-main", str(Path(main).resolve()), "--dataset", str(Path(dataset).resolve()), "--output-dir", str(output.resolve())]
    record = dict(phase=phase, status="launching", started_at=now(), protocol_sha256=sha(output / "protocol.json"),
        source_sha256=sha(Path(__file__)), supervisor_pid=os.getpid(), command=command, log_relative_path=phase + ".log")
    proof, log = output / (phase + "_process.json"), output / (phase + ".log")
    write_new(proof, record)
    with log.open("x", encoding="utf-8") as stream:
        child = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
        record.update(status="running", child_pid=child.pid); publish(proof, record)
        code = child.wait()
    record.update(status="completed" if code == 0 else "failed", exit_code=code, completed_at=now(), log_sha256=sha(log)); publish(proof, record)
    if code:
        raise RuntimeError("Preserve failed phase/log/partial accounts; never silently restart")
    return record


def private_gate(output, phase):
    proof = read_json(Path(output) / (phase + "_process.json"))
    if (proof.get("phase") != phase or proof.get("status") != "running" or proof.get("child_pid") != os.getpid()
            or proof.get("supervisor_pid") != os.getppid()
            or proof.get("source_sha256") != sha(Path(__file__)) or proof.get("protocol_sha256") != sha(Path(output) / "protocol.json")):
        raise ValueError("Private child may run only under its actual matching supervisor")


def fit_internal(main, dataset, output):
    output = Path(output); spec = protocol(main, dataset, output); private_gate(output, "fit")
    tasks = [dict(main=str(main), dataset=str(dataset), output=str(output), profile=profile, seed=seed) for profile in PROFILES for seed in SEEDS]
    pools = {profile: ProcessPoolExecutor(max_workers=1, mp_context=mp.get_context("spawn"),
        initializer=shared.gpu_initializer, initargs=(spec["profile_gpu_assignments"][profile],)) for profile in PROFILES}
    records = []
    try:
        futures = {pools[task["profile"]].submit(train_one, task): task for task in tasks}
        for future in as_completed(futures):
            records.append(future.result())
            publish(output / "training_status.json", dict(at=now(), completed=len(records), total=10))
    finally:
        for pool in pools.values():
            pool.shutdown(wait=True, cancel_futures=True)
    if len(records) != 10:
        raise ValueError("No missing or selected training seeds")
    write_new(output / "training_completion.json", dict(at=now(), model_count=10, records=records, protocol_sha256=sha(output / "protocol.json")))
    write_new(output / "fit_completion.json", dict(at=now(), model_count=10, protocol_sha256=sha(output / "protocol.json")))


def evaluate_worker(task):
    from stable_baselines3 import TD3
    output, main, dataset = map(Path, (task["output"], task["main"], task["dataset"]))
    spec = protocol(main, dataset, output); profile, split = task["profile"], task["split"]
    library_imports(spec["sb3_source_sha256"])
    device = shared.torch_setup(0)
    if device["visible_gpu_uuid"] != spec["profile_gpu_assignments"][profile] or device["capability"] != [7, 0]:
        raise ValueError("Correct original isolated V100 required")
    start, end = VALIDATION if split == "validation" else TEST
    data = cropped_data(shared.load_dataset(dataset), end)
    models, records = [], []
    for seed in SEEDS:
        record = read_json(output / "training" / job_id(profile, seed) / "record.json")
        model = TD3.load(relative(output, record["model_relative_path"]), device="cuda:0")
        if shared.parameter_digest(model.policy) != record["parameter_sha256"] or sha(relative(output, record["model_relative_path"])) != record["model_sha256"]:
            raise ValueError("Loaded policy does not match the fixed trained checkpoint")
        models.append(model); records.append(record)
    for family in task["families"]:
        directory = output / split / family / profile
        directory.mkdir(parents=True, exist_ok=False); rows, histories, monthly_rows, proofs = [], [], [], []
        for scenario, costs in COSTS.items():
            plans = [(str(seed), [model], [seed], seed) for seed, model in zip(SEEDS, models)] + [("ensemble", models, list(SEEDS), None)]
            for key, members, seeds, seed in plans:
                run_id = f"{split}__{scenario}__{family}__td3__{profile}__{key}"
                result, trace = evaluate(data, members, seeds, profile, family, start, end, costs)
                repeat, repeated_trace = evaluate(data, members, seeds, profile, family, start, end, costs)
                fingerprints, repeated = result_fingerprints(result), result_fingerprints(repeat)
                trace_bytes, repeated_bytes = canonical_csv(trace), canonical_csv(repeated_trace)
                if fingerprints != repeated or trace_bytes != repeated_bytes:
                    raise AssertionError("Exact policy/target/account repeat required")
                case = directory / scenario / key; save_result(case, result); (case / "member_targets.csv").write_bytes(trace_bytes)
                proof = dict(run_id=run_id, passed=True, repeat_exact=True, fingerprints=fingerprints, repeated_fingerprints=repeated,
                    member_targets_sha256=hashlib.sha256(trace_bytes).hexdigest(), repeated_member_targets_sha256=hashlib.sha256(repeated_bytes).hexdigest(),
                    parameter_sha256_before=result["metadata"]["parameter_sha256_before"], parameter_sha256_after=result["metadata"]["parameter_sha256_after"],
                    model_records=[record for record in records if record["seed"] in seeds], device=device, worker_pid=os.getpid())
                write_new(case / "replay.json", proof); proofs.append(proof)
                row, history, months = summarize(result, run_id=run_id, strategy="td3_" + family, seed=seed, scenario=scenario,
                    split=split, start=start, end=end, capital=10000., **costs)
                row.update(algorithm="td3", profile=profile, family=family, primary_policy=seed is None, exposure_multiplier=BUDGETS[family], rebalance_band=.025)
                history["profile"], months["profile"] = profile, profile
                rows.append(row); histories.append(history); monthly_rows.append(months)
                print("TD3_EVAL", run_id, row["net_return"], row["max_drawdown"], flush=True)
        for name, frame in zip(GROUP_FILES, (pd.DataFrame(rows), pd.concat(histories, ignore_index=True), pd.concat(monthly_rows, ignore_index=True))):
            frame.to_csv(directory / name, index=False, float_format="%.17g")
        write_new(directory / "completion.json", dict(at=now(), protocol_sha256=sha(output / "protocol.json"), case_count=12, repeat_count=12, repeats=proofs, worker_pid=os.getpid(), device=device))
    return dict(profile=profile, split=split, families=task["families"], device=device, worker_pid=os.getpid())


def validate_internal(main, dataset, output):
    output = Path(output); spec = protocol(main, dataset, output); private_gate(output, "validate")
    terminal(output, "fit"); training_files(output)
    for name in ("test_started.json", "test_completion.json", "test", "selection.json"):
        if (output / name).exists():
            raise ValueError("Validation must finish before selection and every new annual attempt")
    workers = []
    pools = {profile: ProcessPoolExecutor(max_workers=1, mp_context=mp.get_context("spawn"),
        initializer=shared.gpu_initializer, initargs=(spec["profile_gpu_assignments"][profile],)) for profile in PROFILES}
    try:
        futures = [pool.submit(evaluate_worker, dict(main=str(main), dataset=str(dataset), output=str(output),
            profile=profile, split="validation", families=list(BUDGETS))) for profile, pool in pools.items()]
        workers = [future.result() for future in futures]
    finally:
        for pool in pools.values():
            pool.shutdown(wait=True, cancel_futures=True)
    for name in GROUP_FILES[:3]:
        frames = [pd.read_csv(output / "validation" / family / profile / name, float_precision="round_trip") for family in BUDGETS for profile in PROFILES]
        pd.concat(frames, ignore_index=True).to_csv(output / "validation" / name, index=False, float_format="%.17g")
    training_files(output)
    write_new(output / "validation_completion.json", dict(at=now(), protocol_sha256=sha(output / "protocol.json"), case_count=48, repeat_count=48, workers=workers))
    write_new(output / "validate_completion.json", dict(at=now(), case_count=48, repeat_count=48, protocol_sha256=sha(output / "protocol.json")))


def audit_inputs(main, output, audit):
    verify_pins(output, audit.get("input_sha256")); verify_pins(main, audit.get("main_input_sha256")); verify_pins(ROOT, audit.get("source_sha256"))


def validation_audit(main, output):
    output = Path(output); audit = read_json(output / "independent_audit.validation.json")
    if (audit.get("passed") is not True or audit.get("status") != "partial_test_pending" or audit.get("test_complete") is not False
            or audit.get("training_model_count") != 10 or audit.get("validation_run_count") != 48 or audit.get("run_count") != 48
            or audit.get("repeat_count") != 48 or audit.get("failure_count") != 0 or audit.get("failures") != []):
        raise ValueError("Complete independent ten-model/48-account validation audit required before selection")
    audit_inputs(main, output, audit)
    return audit


def validation_files():
    return {"validation_completion.json", "validate_completion.json", "validate_process.json", "validate.log",
        *(f"validation/{name}" for name in GROUP_FILES[:3]),
        *(f"validation/{family}/{profile}/{name}" for family in BUDGETS for profile in PROFILES for name in GROUP_FILES),
        *(f"validation/{family}/{profile}/{cost}/{key}/{name}" for family in BUDGETS for profile in PROFILES
          for cost in COSTS for key in (*map(str, SEEDS), "ensemble") for name in CASE_FILES)}


def group_rows(frame, family, profile, split):
    start, end = VALIDATION if split == "validation" else TEST
    required = {"run_id", "algorithm", "family", "profile", "split", "seed", "primary_policy", "scenario", "initial_capital",
        "start", "end_exclusive", "net_return", "max_drawdown", "final_equity", "fees", "exposure_multiplier", "rebalance_band"}
    if not required.issubset(frame) or len(frame) != 12 or frame.run_id.duplicated().any():
        raise ValueError("Each fixed candidate requires twelve distinct actual accounts")
    expected = {f"{split}__{cost}__{family}__td3__{profile}__{key}" for cost in COSTS for key in (*map(str, SEEDS), "ensemble")}
    fixed = dict(algorithm="td3", family=family, profile=profile, split=split, initial_capital=10000., exposure_multiplier=BUDGETS[family], rebalance_band=.025)
    if set(frame.run_id) != expected or any(not frame[name].eq(value).all() for name, value in fixed.items()):
        raise ValueError("No candidate, seed, cost, capital or execution substitution")
    if not pd.to_datetime(frame.start, utc=True).eq(utc(start)).all() or not pd.to_datetime(frame.end_exclusive, utc=True).eq(utc(end)).all():
        raise ValueError("Only the complete fixed validation/test window allowed")
    if not frame.primary_policy.map(lambda value: isinstance(value, (bool, np.bool_))).all():
        raise ValueError("Actual ensemble booleans required")
    for cost in COSTS:
        rows = frame.loc[frame.scenario.eq(cost)]; primary, seeds = rows.loc[rows.primary_policy], rows.loc[~rows.primary_policy]
        if len(primary) != 1 or primary.seed.notna().any() or len(seeds) != 5 or set(seeds.seed) != set(SEEDS):
            raise ValueError("Every cost needs five unique seeds and one real ensemble")
    values = frame[["net_return", "max_drawdown", "final_equity", "fees"]].to_numpy(dtype=float)
    if (not np.isfinite(values).all() or (frame.net_return <= -1).any() or (frame.max_drawdown > 0).any() or (frame.max_drawdown <= -1).any()
            or (frame.fees < 0).any() or not np.allclose(frame.final_equity / 10000 - 1, frame.net_return, atol=1e-11, rtol=1e-11)):
        raise ValueError("Candidate financial metrics must reconcile to real initial capital")


def candidate(frame, family, profile):
    group_rows(frame, family, profile, "validation")
    costs = {}
    for cost in COSTS:
        row = frame.loc[frame.scenario.eq(cost) & frame.primary_policy].iloc[0]
        costs[cost] = dict(run_id=row.run_id, **{name: float(row[name]) for name in ("net_return", "max_drawdown", "final_equity", "fees")})
    def one(row):
        dd = abs(row["max_drawdown"])
        return math.log1p(row["net_return"]) - .5 * dd - 2 * max(dd - .20, 0.)
    return dict(algorithm="td3", family=family, profile=profile, exposure_multiplier=BUDGETS[family], rebalance_band=.025,
        score=(one(costs["base"]) + one(costs["stress"])) / 2, **costs)


def no_test(output):
    if any((Path(output) / name).exists() for name in ("test", "test_started.json", "test_completion.json", "test_process.json", "annual_authorization.json")):
        raise ValueError("New TD3 selection must precede every new annual attempt")


def select(main, dataset, output):
    output = Path(output); protocol(main, dataset, output); no_test(output)
    terminal(output, "fit"); terminal(output, "validate"); audit = validation_audit(main, output)
    if any((output / name).exists() for name in ("selection.json", "selection_receipt.json")):
        raise FileExistsError("Preserve the original sealed four-candidate decision")
    records = training_files(output)
    if not validation_files().issubset(audit["input_sha256"]):
        raise ValueError("The independent audit must bind all48 account/repeat files")
    choices = [candidate(pd.read_csv(output / "validation" / family / profile / "summary.csv", float_precision="round_trip"), family, profile)
               for family in BUDGETS for profile in PROFILES]
    choices.sort(key=lambda row: (-row["score"], 0 if row["family"] == "raw" else 1, row["profile"]))
    value = dict(selected_at=now(), protocol_sha256=sha(output / "protocol.json"), audit_sha256=sha(output / "independent_audit.validation.json"),
        candidates=choices, candidate=choices[0], training_source_sha256=records, validation_source_sha256=pins(output, validation_files()),
        seed_selection=False, selection_uses_test=False, test_results_available_to_selector=False,
        unchanged_recipe=True, **{flag: True for flag in FLAGS})
    write_new(output / "selection.json", value)
    write_new(output / "selection_receipt.json", dict(selection_file="selection.json", selection_sha256=sha(output / "selection.json"), protocol_sha256=sha(output / "protocol.json"), sealed_at=now()))
    return value


def frozen_selection(main, dataset, output):
    output = Path(output); spec = protocol(main, dataset, output)
    value, receipt = read_json(output / "selection.json"), read_json(output / "selection_receipt.json")
    if (receipt.get("selection_file") != "selection.json" or receipt.get("selection_sha256") != sha(output / "selection.json")
            or receipt.get("protocol_sha256") != value.get("protocol_sha256") or value.get("protocol_sha256") != sha(output / "protocol.json")
            or value.get("audit_sha256") != sha(output / "independent_audit.validation.json")
            or value.get("seed_selection") is not False or value.get("selection_uses_test") is not False
            or value.get("test_results_available_to_selector") is not False or any(value.get(flag) is not True for flag in FLAGS)):
        raise ValueError("Original validation-only receipt and all five seeds must remain immutable")
    validation_audit(main, output)
    verify_pins(output, value["training_source_sha256"]); verify_pins(output, value["validation_source_sha256"])
    choices = value.get("candidates", [])
    if (len(choices) != 4 or {(row.get("family"), row.get("profile")) for row in choices} != {(family, profile) for family in BUDGETS for profile in PROFILES}
            or choices != sorted(choices, key=lambda row: (-row["score"], 0 if row["family"] == "raw" else 1, row["profile"]))
            or value.get("candidate") != choices[0] or utc(value["selected_at"]) > utc(receipt["sealed_at"])):
        raise ValueError("Exactly the four predeclared candidates and raw-first tie rule required")
    for stored in choices:
        actual = candidate(pd.read_csv(output / "validation" / stored["family"] / stored["profile"] / "summary.csv", float_precision="round_trip"), stored["family"], stored["profile"])
        if not math.isclose(stored["score"], actual["score"], abs_tol=1e-12, rel_tol=1e-12):
            raise ValueError("Frozen score no longer matches its actual validation accounts")
        for cost in COSTS:
            if stored[cost]["run_id"] != actual[cost]["run_id"] or any(not math.isclose(stored[cost][key], actual[cost][key], abs_tol=1e-10, rel_tol=1e-10) for key in ("net_return", "max_drawdown", "fees", "final_equity")):
                raise ValueError("Frozen validation source metrics cannot be replaced by test ranking")
    return spec, value, receipt


def annual_gate(main, dataset, output, authorization, *, require_new):
    output = Path(output); spec, selection, receipt = frozen_selection(main, dataset, output)
    terminal(output, "fit"); terminal(output, "validate")
    if require_new:
        no_test(output)
    if authorization is None:
        raise ValueError("Separate explicit known-year annual authorization required; validation never starts annual automatically")
    value = read_json(authorization)
    if (value.get("annual_evaluation_authorized") is not True or value.get("no_live_trading") is not True
            or value.get("selection_sha256") != sha(output / "selection.json") or value.get("financial_parameters_fixed") is not True
            or any(value.get(flag) is not True for flag in FLAGS) or utc(value["authorized_at"]) < utc(receipt["sealed_at"])):
        raise ValueError("Annual authorization must follow and identify this exact sealed validation candidate")
    if not require_new:
        marker = read_json(output / "test_started.json")
        if (marker.get("protocol_sha256") != sha(output / "protocol.json") or marker.get("selection_sha256") != sha(output / "selection.json")
                or marker.get("authorization_sha256") != sha(authorization) or utc(marker["at"]) < utc(value["authorized_at"])):
            raise ValueError("Annual dispatch identity/time must match its authorization and fixed candidate")
    return spec, selection


def test_internal(main, dataset, output):
    output = Path(output); private_gate(output, "test")
    spec, selection = annual_gate(main, dataset, output, output / "annual_authorization.json", require_new=False)
    chosen = selection["candidate"]; profile = chosen["profile"]
    with ProcessPoolExecutor(max_workers=1, mp_context=mp.get_context("spawn"), initializer=shared.gpu_initializer,
            initargs=(spec["profile_gpu_assignments"][profile],)) as pool:
        worker = pool.submit(evaluate_worker, dict(main=str(main), dataset=str(dataset), output=str(output), profile=profile,
            split="test", families=[chosen["family"]])).result()
    training_files(output)
    write_new(output / "test_completion.json", dict(at=now(), case_count=12, repeat_count=12, candidate=chosen,
        protocol_sha256=sha(output / "protocol.json"), selection_sha256=sha(output / "selection.json"), worker=worker))


def verify(main, dataset, output):
    output = Path(output); spec, selection = annual_gate(main, dataset, output, output / "annual_authorization.json", require_new=False)
    terminal(output, "test"); audit = read_json(output / "independent_audit.json")
    if (audit.get("passed") is not True or audit.get("status") != "passed" or audit.get("test_complete") is not True
            or audit.get("training_model_count") != 10 or audit.get("validation_run_count") != 48 or audit.get("test_run_count") != 12
            or audit.get("baseline_run_count") != 14 or audit.get("run_count") != 26 or audit.get("repeat_count") != 12
            or audit.get("failure_count") != 0 or audit.get("failures") != []):
        raise ValueError("Independent selected12/14baseline final audit and immutable48 validation required")
    audit_inputs(main, output, audit)
    targets = audit.get("research_targets", {})
    if targets.get("candidate") != {key: selection["candidate"][key] for key in ("algorithm", "family", "profile")} or targets.get("fresh_holdout_success") is not False:
        raise ValueError("Goal evidence must refer to this new TD3 winner and retain known-year disclosure")
    value = dict(at=now(), protocol_sha256=sha(output / "protocol.json"), selection_sha256=sha(output / "selection.json"),
        audit_sha256=sha(output / "independent_audit.json"), accounting_audit_passed=True, **targets)
    write_new(output / "goal_candidate_evidence.json", value)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("prepare", "fit", "validate", "select", "test", "verify", "_fit", "_validate", "_test"), required=True)
    for name in ("reference-main", "dataset", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--upstream-root", type=Path)
    parser.add_argument("--resource-allowance-file", type=Path)
    parser.add_argument("--annual-authorization", type=Path)
    args = parser.parse_args()
    if args.phase == "prepare":
        if args.upstream_root is None:
            parser.error("--upstream-root required for reviewed official source and license bytes")
        result = prepare(args.reference_main, args.dataset, args.output_dir, upstream_root=args.upstream_root, allowance_file=args.resource_allowance_file)
    elif args.phase in ("fit", "validate", "test"):
        result = supervise(args.reference_main, args.dataset, args.output_dir, args.phase, authorization=args.annual_authorization)
    elif args.phase == "select":
        result = select(args.reference_main, args.dataset, args.output_dir)
    elif args.phase == "verify":
        result = verify(args.reference_main, args.dataset, args.output_dir)
    else:
        result = {"_fit": fit_internal, "_validate": validate_internal, "_test": test_internal}[args.phase](args.reference_main, args.dataset, args.output_dir)
    print(json.dumps(dict(phase=args.phase, completed=True, result=result), ensure_ascii=False, default=str), flush=True)


if __name__ == "__main__":
    main()

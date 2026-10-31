"""Study-bound native models for the independent rolling-account experiment.

The original environment and model helpers remain unchanged.  Every scheduled
fit starts a fresh native learner; these loaders are inference-only and do not
claim optimizer/replay/RNG-complete training recovery.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta, timezone
import hashlib
import importlib
import json
from pathlib import Path
import random

import numpy as np

BASE_ADAPTER_SHA256 = "1f7e7267b51465cc0d06c3af6d120b49c8d2868b6e18a55b1c56b37ed78966a9"
SEEDS = (0, 1, 2, 3, 4)
SYMBOLS = ("BTCUSDT", "ETHUSDT")
RECIPE_ALGORITHMS = {"sac_default": "sac", "sac_gamma099": "sac", "tqc_default": "tqc", "td3_default": "td3"}
STEPS = {"sac": 32768, "tqc": 32768, "td3": 65536}
REQUIRED_CONTEXT = {"recipe_id", "symbol", "algorithm", "seed", "version_index", "effective_day", "train_start", "train_end", "protocol_sha256"}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _base():
    base = importlib.import_module("scripts.single_pair_rl_models")
    expected = Path(__file__).resolve().with_name("single_pair_rl_models.py")
    if Path(base.__file__).resolve() != expected or sha(expected) != BASE_ADAPTER_SHA256:
        raise ValueError("The unchanged frozen single-pair adapter must be imported from this source root")
    if base.SEEDS != SEEDS:
        raise ValueError("Use a clean rolling worker with the declared original seeds 0..4")
    return base


def _json(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def _day(value):
    if not isinstance(value, str):
        raise ValueError("A string UTC day boundary is required")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0) or any((parsed.hour, parsed.minute, parsed.second, parsed.microsecond)):
        raise ValueError("An explicit UTC midnight boundary is required")
    return parsed


def validate_context(context, *, recipe_id=None, algorithm=None, symbol=None, seed=None):
    if not isinstance(context, dict) or not REQUIRED_CONTEXT <= set(context):
        raise ValueError("Complete rolling recipe/date/protocol context is required")
    try:
        copied = json.loads(_json(context))
    except (TypeError, ValueError) as exc:
        raise ValueError("Rolling context must contain finite JSON values") from exc
    rid = copied["recipe_id"]
    if rid not in RECIPE_ALGORITHMS or recipe_id is not None and rid != recipe_id:
        raise ValueError("Unplanned or mismatched rolling recipe")
    if algorithm is not None and RECIPE_ALGORITHMS[rid] != algorithm:
        raise ValueError("Rolling algorithm differs from its recipe")
    if (copied["algorithm"] != RECIPE_ALGORITHMS[rid] or copied["symbol"] not in SYMBOLS
            or type(copied["seed"]) is not int or copied["seed"] not in SEEDS):
        raise ValueError("Rolling context must bind its declared account and native recipe")
    if symbol is not None and symbol not in SYMBOLS or seed is not None and (type(seed) is not int or seed not in SEEDS):
        raise ValueError("Declared single-pair symbol and integer seed 0..4 are required")
    for key, expected in (("algorithm", algorithm), ("symbol", symbol), ("seed", seed)):
        if key in copied and expected is not None and (copied[key] != expected or key == "seed" and type(copied[key]) is not int):
            raise ValueError("Rolling context identity differs: " + key)
    start, end, effective = (_day(copied[key]) for key in ("train_start", "train_end", "effective_day"))
    version = copied["version_index"]
    if (type(version) is not int or version not in range(53)
            or effective != datetime(2025, 10, 23, tzinfo=timezone.utc) + timedelta(days=7 * version)):
        raise ValueError("The rolling version must identify its predeclared weekly effective day")
    if end != effective - timedelta(days=1) or start != end - timedelta(days=1827):
        raise ValueError("Rolling training must use 1827 days ending one day before effective use")
    protocol = copied["protocol_sha256"]
    if not isinstance(protocol, str) or len(protocol) != 64 or any(c not in "0123456789abcdef" for c in protocol):
        raise ValueError("A lower-case protocol SHA256 is required")
    if "nonfinancial_preflight" in copied and type(copied["nonfinancial_preflight"]) is not bool:
        raise ValueError("The nonfinancial preflight flag must be Boolean")
    return copied


def recipe(algorithm, recipe_id):
    if RECIPE_ALGORITHMS.get(recipe_id) != algorithm:
        raise ValueError("Only the frozen SAC/TQC/TD3 rolling recipes are allowed")
    result = deepcopy(_base().recipe(algorithm))
    if recipe_id == "sac_gamma099":
        result["gamma"] = .99
    return result


def _native_value(value):
    if isinstance(value, dict):
        return {str(k): _native_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_native_value(v) for v in value]
    if isinstance(value, type):
        return value.__name__
    if value is None or type(value) in (str, int, float, bool):
        return value
    raise ValueError("Unexpected native policy parameter type: " + type(value).__name__)


def canonical_policy_kwargs(values):
    """Native SAC/TQC save may add the explicit default ``use_sde=False``."""
    if not isinstance(values, dict):
        raise ValueError("A native policy parameter mapping is required")
    result = deepcopy(values)
    if "use_sde" in result:
        if result.pop("use_sde") is not False:
            raise ValueError("Only the native Boolean use_sde=False default is permitted")
    return _native_value(result)


def check_native_parameters(model, algorithm, declared):
    base = _base()
    actual = base.effective_parameters(model, algorithm)
    actual.update(observation_shape=list(model.observation_space.shape), action_shape=list(model.action_space.shape),
                  native_class=type(model).__name__)
    for key in ("learning_rate", "gamma", "tau", "learning_starts", "batch_size", "buffer_size", "gradient_steps", "ent_coef"):
        if key in declared and actual.get(key) != declared[key]:
            raise ValueError("Actual rolling native parameter differs: " + key)
    if (actual.get("observation_shape") != [13] or actual.get("action_shape") != [1]
            or actual.get("native_class") != {"sac": "SAC", "tqc": "TQC", "td3": "TD3"}[algorithm]
            or actual.get("train_freq") != [1, "step"]):
        raise ValueError("Native rolling spaces, class or update frequency changed")
    if canonical_policy_kwargs(model.policy_kwargs) != canonical_policy_kwargs(declared["policy_kwargs"]):
        raise ValueError("Actual native network/optimizer policy parameters differ")
    for key in ("policy_delay", "target_policy_noise", "target_noise_clip", "top_quantiles_to_drop_per_net"):
        if key in declared and getattr(model, key, None) != declared[key]:
            raise ValueError("Actual algorithm-specific native parameter differs: " + key)
    return actual


def _rng_evidence(env, device):
    """Observe state without initializing generators or consuming randomness."""
    import torch
    def binary(value):
        array = np.asarray(value)
        return {"sha256": hashlib.sha256(array.tobytes()).hexdigest(), "shape": list(array.shape), "dtype": str(array.dtype)}
    np_state = np.random.get_state()
    result = {"python_state_sha256": hashlib.sha256(_json(random.getstate()).encode()).hexdigest(),
              "numpy": {"generator": np_state[0], "state": binary(np_state[1]), "position": int(np_state[2]),
                        "has_gauss": int(np_state[3]), "cached_gaussian": float(np_state[4])},
              "torch_cpu": binary(torch.get_rng_state().cpu().numpy())}
    if str(device).startswith("cuda"):
        result["torch_cuda0"] = binary(torch.cuda.get_rng_state(device).cpu().numpy())
    for key, target in (("environment", env.unwrapped), ("action_space", env.action_space)):
        generator = getattr(target, "_np_random", None)
        result[key] = {"initialized": generator is not None}
        if generator is not None:
            state = generator.bit_generator.state
            result[key].update(generator=type(generator.bit_generator).__name__,
                state_sha256=hashlib.sha256(_json(state).encode()).hexdigest())
    return result


def _check_environment(env, context):
    identity = _base().environment_identity(env)
    if context.get("nonfinancial_preflight") is True:
        return identity
    raw = env.unwrapped
    if (getattr(raw, "training", None) is not True or getattr(raw, "episode_days", None) != 32
            or _day(context["train_start"]) != raw.start.to_pydatetime()
            or _day(context["train_end"]) != raw.end.to_pydatetime()):
        raise ValueError("Use the exact rolling training environment and complete 32-day fragments")
    maximum = raw._frames["close"].index.max().to_pydatetime()
    if maximum != _day(context["train_end"]) - timedelta(hours=1):
        raise ValueError("Training inputs must be physically truncated before the frozen end")
    if raw._frames["close"].index.min().to_pydatetime() > _day(context["train_start"]) - timedelta(days=200):
        raise ValueError("Retain at least 200 calendar days of causal feature prehistory outside training fragments")
    if (raw.capital != 10000. or raw.costs != {"fee": .001, "slippage": .0002}
            or raw.cash_floor != .05 or raw.max_asset_weight != .95
            or raw.rebalance_band != .025 or raw.drawdown_penalty != .25):
        raise ValueError("Original independent training capital, costs and financial rules are required")
    return identity


def make_model(env, algorithm, recipe_id, seed, context, trace_path=None, device="cuda:0", upstream=None):
    del upstream  # The declared grid contains native SB3/SB3-contrib algorithms.
    base = _base()
    identity = base.environment_identity(env)
    context = validate_context(context, recipe_id=recipe_id, algorithm=algorithm, symbol=identity["symbol"], seed=seed)
    _check_environment(env, context)
    declared = recipe(algorithm, recipe_id)
    base.seed_everything(seed, device)
    policy, kwargs = base.native_kwargs(algorithm, trace_path)
    if recipe_id == "sac_gamma099":
        kwargs["gamma"] = .99
    model = base.native_class(algorithm)(policy, env, seed=seed, device=device, verbose=0, **kwargs)
    model._single_pair_identity = dict(identity, algorithm=algorithm, seed=seed)
    model._single_pair_training = None
    model._rolling_context, model._rolling_recipe_id, model._rolling_recipe = context, recipe_id, declared
    model._rolling_rng_before_train = _rng_evidence(env, device)
    check_native_parameters(model, algorithm, declared)
    return model


def check_training_record(training, algorithm, context):
    preflight = context.get("nonfinancial_preflight") is True
    steps = training.get("actual_steps")
    expected_steps = 1024 if preflight else STEPS[algorithm]
    if steps != expected_steps:
        raise ValueError("The rolling interaction budget differs from its declared native recipe")
    updates = steps - 512
    counters = training.get("counters", {})
    expected_optimizers = {"actor": updates // (2 if algorithm == "td3" else 1), "critic": updates}
    if algorithm != "td3":
        expected_optimizers["entropy"] = updates
    if (counters.get("environment_steps") != steps or counters.get("gradient_update_rounds") != updates
            or counters.get("optimizer_steps") != expected_optimizers
            or counters.get("replay_add_calls") != steps or counters.get("replay_transitions") != steps):
        raise ValueError("Actual rolling optimizer/replay/update counters differ")
    if not preflight:
        completed = [row for row in training.get("training_episode_history", []) if row.get("completed_steps", 0) > 0]
        start, end = _day(context["train_start"]), _day(context["train_end"])
        if len(completed) != steps // 32:
            raise ValueError("Every original 32-day fragment must be recorded")
        for row in completed:
            first = datetime.fromisoformat(row["start"])
            until, last = (datetime.fromisoformat(row[key]) for key in ("completed_until", "end_exclusive"))
            if (row.get("completed_steps") != 32 or row.get("terminated") is not False or row.get("truncated") is not True
                    or first < start or last > end or last != first + timedelta(days=32) or until != last):
                raise ValueError("Rolling training fragments cannot terminate or cross the physical training window")
    return True


def train_model(model, env, algorithm, steps, seed, progress_path=None):
    base = _base()
    context = validate_context(model._rolling_context, recipe_id=model._rolling_recipe_id,
        algorithm=algorithm, symbol=base.environment_identity(env)["symbol"], seed=seed)
    _check_environment(env, context)
    expected_steps = 1024 if context.get("nonfinancial_preflight") is True else STEPS[algorithm]
    if type(steps) is not int or steps != expected_steps:
        raise ValueError("Only the frozen algorithm budget or explicit nonfinancial 1024-step preflight is permitted")
    check_native_parameters(model, algorithm, model._rolling_recipe)
    training = base.train_model(model, env, steps, seed, progress_path)
    check_training_record(training, algorithm, context)
    check_native_parameters(model, algorithm, model._rolling_recipe)
    model._rolling_rng_after_train = _rng_evidence(env, model.device)
    return training


def save_model(model, env, algorithm, seed, directory, context):
    base = _base()
    identity = base.environment_identity(env)
    expected = validate_context(context, recipe_id=model._rolling_recipe_id, algorithm=algorithm, symbol=identity["symbol"], seed=seed)
    if expected != model._rolling_context or model._single_pair_identity != dict(identity, algorithm=algorithm, seed=seed):
        raise ValueError("Cannot relabel a rolling model's context or account")
    _check_environment(env, expected)
    training = model._single_pair_training
    check_training_record(training or {}, algorithm, expected)
    check_native_parameters(model, algorithm, model._rolling_recipe)
    directory = Path(directory)
    if directory.is_symlink() or directory.exists() and any(directory.iterdir()):
        raise FileExistsError("Preserve every rolling checkpoint attempt")
    directory.mkdir(parents=True, exist_ok=True)
    model.save(directory / "model.zip")
    manifest = {"schema_version": 1, **model._single_pair_identity, "recipe_id": model._rolling_recipe_id,
        "rolling_context": expected, "recipe": model._rolling_recipe, "training": training,
        "parameter_sha256": base.parameter_digest(model, algorithm), "checkpoint_files": {"model.zip": sha(directory / "model.zip")},
        "runtime_packages": base.runtime_versions(), "adapter_sha256": sha(__file__), "base_adapter_sha256": BASE_ADAPTER_SHA256,
        "rng_before_training": model._rolling_rng_before_train, "rng_after_training": model._rolling_rng_after_train,
        "native_optimizer_state_retained": True, "no_optimizer_or_replay_resume": True,
        "resume_note": "Each scheduled model is trained from scratch. RNG digests and replay counters are evidence, not a complete resume checkpoint.",
        "nonfinancial_preflight": expected.get("nonfinancial_preflight") is True}
    with (directory / "rolling_model.json").open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return manifest


def load_model(directory, algorithm, symbol, seed, context, device="cuda:0"):
    base, directory = _base(), Path(directory)
    expected = validate_context(context, algorithm=algorithm, symbol=symbol, seed=seed)
    declared = recipe(algorithm, expected["recipe_id"])
    manifest = json.loads((directory / "rolling_model.json").read_text(encoding="utf-8"))
    identity = dict(symbol=symbol, algorithm=algorithm, seed=seed, observation_shape=[13], action_shape=[1],
                    independent_account=True, multi_model_target_average=False)
    if (manifest.get("schema_version") != 1 or any(manifest.get(k) != v for k, v in identity.items())
            or manifest.get("recipe_id") != expected["recipe_id"] or manifest.get("rolling_context") != expected
            or manifest.get("recipe") != declared or manifest.get("adapter_sha256") != sha(__file__)
            or manifest.get("base_adapter_sha256") != BASE_ADAPTER_SHA256
            or manifest.get("checkpoint_files") != {"model.zip": sha(directory / "model.zip")}
            or manifest.get("native_optimizer_state_retained") is not True or manifest.get("no_optimizer_or_replay_resume") is not True
            or manifest.get("nonfinancial_preflight") is not (expected.get("nonfinancial_preflight") is True)):
        raise ValueError("Rolling checkpoint identity, context, recipe, source or bytes differ")
    check_training_record(manifest.get("training", {}), algorithm, expected)
    model = base.native_class(algorithm).load(directory / "model.zip", device=device)
    if model.observation_space.shape != (13,) or model.action_space.shape != (1,):
        raise ValueError("Loaded rolling spaces differ")
    check_native_parameters(model, algorithm, declared)
    model.policy.set_training_mode(False)
    if base.parameter_digest(model, algorithm) != manifest["parameter_sha256"]:
        raise ValueError("Loaded rolling model tensors differ")
    model._single_pair_identity, model._single_pair_training = identity, manifest["training"]
    model._rolling_context, model._rolling_recipe_id, model._rolling_recipe = expected, expected["recipe_id"], declared
    return model


def predict(model, obs, algorithm=None, state=None, episode_start=False):
    algorithm = algorithm or model._single_pair_identity["algorithm"]
    return _base().predict(model, obs, algorithm, state, episode_start)


def digest(model, algorithm=None):
    return _base().parameter_digest(model, algorithm or model._single_pair_identity["algorithm"])


parameter_digest = digest

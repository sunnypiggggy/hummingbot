"""Native SAC with truthful, study-bound candidate manifests.

The frozen single-pair model helpers still supply seeding, observed replay,
training counters, prediction and tensor digests. Only the declared learning
rate and gamma candidates differ; no financial environment is imported here.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib
import json
from pathlib import Path
import sys

BASE_ADAPTER_SHA256 = "1f7e7267b51465cc0d06c3af6d120b49c8d2868b6e18a55b1c56b37ed78966a9"
SYMBOLS = ("BTCUSDT", "ETHUSDT")
STAGE_SEEDS = {"tuning": tuple(range(100, 105)), "final": tuple(range(30)), "preflight": (29,)}
CANDIDATE_OVERRIDES = {
    "c00_baseline": {"learning_rate": .0003, "gamma": .995},
    "c01_lr1e4": {"learning_rate": .0001, "gamma": .995},
    "c02_gamma099": {"learning_rate": .0003, "gamma": .99},
    "c03_lr1e4_gamma099": {"learning_rate": .0001, "gamma": .99},
}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def base_models(core_root):
    root = Path(core_root).resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    base = importlib.import_module("scripts.single_pair_rl_models")
    if Path(base.__file__).resolve() != root / "scripts/single_pair_rl_models.py" or sha(base.__file__) != BASE_ADAPTER_SHA256:
        raise ValueError("Use a clean process with the unchanged frozen single-pair model adapter")
    return base


def recipe(candidate_id, core_root):
    if candidate_id not in CANDIDATE_OVERRIDES:
        raise ValueError("Only the four predeclared SAC candidates are allowed")
    result = deepcopy(base_models(core_root).recipe("sac"))
    result.update(CANDIDATE_OVERRIDES[candidate_id])
    return result


def context(stage, symbol, seed, candidate_id, parent_protocol_sha256):
    if (stage not in STAGE_SEEDS or type(seed) is not int or seed not in STAGE_SEEDS[stage]
            or symbol not in SYMBOLS or candidate_id not in CANDIDATE_OVERRIDES
            or not isinstance(parent_protocol_sha256, str) or len(parent_protocol_sha256) != 64
            or any(c not in "0123456789abcdef" for c in parent_protocol_sha256)):
        raise ValueError("A complete declared stage/coin/seed/candidate/protocol identity is required")
    return {"stage": stage, "symbol": symbol, "seed": seed, "candidate_id": candidate_id,
            "parent_protocol_sha256": parent_protocol_sha256}


def make_model(env, *, seed, candidate_id, stage, parent_protocol_sha256, core_root, device="cuda:0"):
    base = base_models(core_root)
    identity = base.environment_identity(env)
    bound = context(stage, identity["symbol"], seed, candidate_id, parent_protocol_sha256)
    base.SEEDS = STAGE_SEEDS[stage]
    base.seed_everything(seed, device)
    policy, kwargs = base.native_kwargs("sac")
    kwargs.update(CANDIDATE_OVERRIDES[candidate_id])
    model = base.native_class("sac")(policy, env, seed=seed, device=device, verbose=0, **kwargs)
    model._single_pair_identity = dict(identity, algorithm="sac", seed=seed)
    model._single_pair_training = None
    model._sac_optimization_context = bound
    model._sac_optimization_recipe = recipe(candidate_id, core_root)
    return model


def check_effective_parameters(training, native_recipe, *, preflight=False):
    steps = 1024 if preflight else 32768
    actual = training["effective_parameters"]
    for key in ("learning_rate", "gamma", "tau", "learning_starts", "batch_size", "buffer_size", "gradient_steps", "ent_coef"):
        if actual.get(key) != native_recipe[key]:
            raise ValueError("Actual native SAC parameter differs from its declared recipe: " + key)
    if (actual.get("observation_shape") != [13] or actual.get("action_shape") != [1]
            or actual.get("native_class") != "SAC" or actual.get("train_freq") != [1, "step"]):
        raise ValueError("Native SAC dimensions/update frequency changed")
    counters, updates = training["counters"], steps - 512
    if (training.get("actual_steps") != steps or counters.get("environment_steps") != steps
            or counters.get("gradient_update_rounds") != updates
            or counters.get("optimizer_steps") != {"actor": updates, "critic": updates, "entropy": updates}
            or counters.get("replay_add_calls") != steps or counters.get("replay_transitions") != steps):
        raise ValueError("The original native SAC interaction/optimizer/replay budget changed")


def train_model(model, env, *, core_root, progress_path=None):
    bound = model._sac_optimization_context
    base = base_models(core_root)
    base.SEEDS = STAGE_SEEDS[bound["stage"]]
    preflight = bound["stage"] == "preflight"
    training = base.train_model(model, env, steps=1024 if preflight else 32768,
                                seed=bound["seed"], progress_path=progress_path)
    check_effective_parameters(training, model._sac_optimization_recipe, preflight=preflight)
    return training


def save_model(model, directory, *, core_root):
    base, directory = base_models(core_root), Path(directory)
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError("Preserve every existing candidate checkpoint")
    directory.mkdir(parents=True, exist_ok=True)
    bound = model._sac_optimization_context
    training = model._single_pair_training
    check_effective_parameters(training, model._sac_optimization_recipe, preflight=bound["stage"] == "preflight")
    model.save(directory / "model.zip")
    manifest = {"schema_version": 2, **model._single_pair_identity, "model_context": bound,
        "candidate_id": bound["candidate_id"], "stage": bound["stage"], "recipe": model._sac_optimization_recipe,
        "training": training, "parameter_sha256": base.parameter_digest(model, "sac"),
        "checkpoint_files": {"model.zip": sha(directory / "model.zip")}, "runtime_packages": base.runtime_versions(),
        "adapter_sha256": sha(__file__), "base_adapter_sha256": BASE_ADAPTER_SHA256,
        "no_optimizer_or_replay_resume": True, "nonfinancial_preflight": bound["stage"] == "preflight"}
    with (directory / "single_pair_model.json").open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return manifest


def load_model(directory, *, symbol, seed, candidate_id, stage, parent_protocol_sha256, core_root,
               device="cuda:0", retained_pin=None):
    base, directory = base_models(core_root), Path(directory)
    expected = context(stage, symbol, seed, candidate_id, parent_protocol_sha256)
    manifest = json.loads((directory / "single_pair_model.json").read_text(encoding="utf-8"))
    if retained_pin is not None:
        if stage != "final" or candidate_id != "c00_baseline" or retained_pin.get("origin") != "retained_baseline_fallback":
            raise ValueError("Only a declared c00 final fallback may reuse an original model")
        for name, digest in retained_pin["file_sha256"].items():
            part = Path(name)
            if part.is_absolute() or ".." in part.parts or sha(directory.parent / part) != digest:
                raise ValueError("Retained fallback model bytes differ from their import evidence")
        if (manifest.get("adapter_sha256") != BASE_ADAPTER_SHA256 or manifest.get("recipe") != recipe(candidate_id, core_root)
                or manifest.get("symbol") != symbol or manifest.get("seed") != seed or manifest.get("algorithm") != "sac"):
            raise ValueError("The original fallback model identity or default recipe changed")
        check_effective_parameters(manifest["training"], manifest["recipe"])
        model = base.load_model(directory, "sac", device=device)
        model._sac_optimization_context = expected
        model._sac_optimization_recipe = recipe(candidate_id, core_root)
        return model
    if (manifest.get("schema_version") != 2 or manifest.get("adapter_sha256") != sha(__file__)
            or manifest.get("base_adapter_sha256") != BASE_ADAPTER_SHA256 or manifest.get("model_context") != expected
            or manifest.get("candidate_id") != candidate_id or manifest.get("stage") != stage
            or manifest.get("recipe") != recipe(candidate_id, core_root) or manifest.get("algorithm") != "sac"
            or manifest.get("symbol") != symbol or manifest.get("seed") != seed
            or manifest.get("observation_shape") != [13] or manifest.get("action_shape") != [1]
            or manifest.get("independent_account") is not True or manifest.get("multi_model_target_average") is not False
            or set(manifest.get("checkpoint_files", {})) != {"model.zip"}
            or manifest["checkpoint_files"]["model.zip"] != sha(directory / "model.zip")):
        raise ValueError("Candidate model identity, actual recipe or source bytes changed")
    check_effective_parameters(manifest["training"], manifest["recipe"], preflight=stage == "preflight")
    model = base.native_class("sac").load(directory / "model.zip", device=device)
    if model.observation_space.shape != (13,) or model.action_space.shape != (1,):
        raise ValueError("Restored candidate native spaces changed")
    model.policy.set_training_mode(False)
    if base.parameter_digest(model, "sac") != manifest["parameter_sha256"]:
        raise ValueError("Restored native policy tensors differ")
    model._single_pair_identity = {key: manifest[key] for key in
        ("symbol", "observation_shape", "action_shape", "independent_account", "multi_model_target_average", "algorithm", "seed")}
    model._single_pair_training = manifest["training"]
    model._sac_optimization_context, model._sac_optimization_recipe = expected, manifest["recipe"]
    return model


def predict(model, obs, *, core_root):
    return base_models(core_root).predict(model, obs, algorithm="sac", state=None, episode_start=False)[0]


def parameter_digest(model, core_root):
    return base_models(core_root).parameter_digest(model, "sac")

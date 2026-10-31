"""Finite validation-only SAC search with unchanged independent spot accounting.

Tuning uses seeds100..104. Per-coin parameters freeze before final seeds0..29.
An unchanged c00 fallback reuses original audited models/accounts byte-for-byte.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SYMBOLS = ("BTCUSDT", "ETHUSDT")
TUNING_SEEDS, FINAL_SEEDS = tuple(range(100, 105)), tuple(range(30))
CANDIDATES = ("c00_baseline", "c01_lr1e4", "c02_gamma099", "c03_lr1e4_gamma099")
BASELINES = ("cash", "buy_hold", "sma200_trend", "momentum180")
FINANCIAL_FIELDS = ("train", "validation", "test", "costs", "environment", "capital_per_account",
    "capital_shared_between_pairs", "ensemble", "dataset_sha256", "dataset_manifest_sha256", "sources",
    "runtime_packages", "python", "no_production_change", "test_year_already_observed", "fresh_holdout")
EXTENSION_NAMES = ("scripts/backtest_sac_validation_optimization.py", "scripts/sac_validation_optimization_models.py",
                   "scripts/audit_sac_validation_optimization.py")
RESULT_FILES = ("equity_1h.csv.gz", "fills.csv", "weights.csv", "boundary_equity.csv", "metadata.json",
                "fingerprints.json", "actions.json", "summary.json", "daily_equity.csv", "monthly_returns.csv")


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def safe(root, name):
    name, root = str(name), Path(root).absolute()
    part = PurePosixPath(name)
    if not name or part.as_posix() != name or part.is_absolute() or ".." in part.parts or ":" in name or "\\" in name:
        raise ValueError("Only canonical relative research evidence paths are allowed")
    path = root.joinpath(*part.parts)
    if any(p.is_symlink() or hasattr(p, "is_junction") and p.is_junction() for p in (path, *path.parents)):
        raise ValueError("Research evidence cannot cross a link")
    path.resolve().relative_to(root.resolve())
    return path


def tree(directory):
    directory = Path(directory)
    result = {}
    for path in sorted(directory.rglob("*")):
        name = path.relative_to(directory).as_posix()
        safe(directory, name)
        if path.is_file():
            result[name] = sha(path)
    return result


def copy_file(source, destination):
    source, destination = Path(source), Path(destination)
    if source.is_symlink() or destination.exists() or destination.is_symlink():
        raise FileExistsError("Preserve existing or linked research evidence")
    expected = sha(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as origin, destination.open("xb") as target:
        shutil.copyfileobj(origin, target)
    if sha(source) != expected or sha(destination) != expected:
        raise ValueError("Evidence changed during copying")
    return expected


def copy_tree(source, destination):
    inventory = tree(source)
    if not inventory or Path(destination).exists():
        raise FileExistsError("A complete new evidence destination is required")
    for name, expected in inventory.items():
        if copy_file(safe(source, name), safe(destination, name)) != expected:
            raise ValueError("Copied evidence differs")
    if tree(source) != inventory or tree(destination) != inventory:
        raise ValueError("Model/case evidence changed during copying")
    return inventory


def core_module(core_root):
    core_root = Path(core_root).resolve()
    if str(core_root) not in sys.path:
        sys.path.insert(0, str(core_root))
    module = importlib.import_module("scripts.backtest_single_pair_rl")
    if Path(module.__file__).resolve() != core_root / "scripts/backtest_single_pair_rl.py":
        raise ValueError("Use a clean process with the exact frozen financial core")
    return module


def adapter():
    module = importlib.import_module("scripts.sac_validation_optimization_models")
    if Path(module.__file__).resolve() != ROOT / "scripts/sac_validation_optimization_models.py":
        raise ValueError("The running candidate adapter differs from this frozen extension")
    return module


def candidate_root(top, candidate):
    if candidate not in CANDIDATES:
        raise ValueError("Unknown finite candidate")
    return Path(top) / "tuning" / candidate


def _copy_core(source_root, destination, source_map):
    for name, expected in source_map.items():
        if copy_file(safe(source_root, name), safe(destination / "frozen_source", name)) != expected:
            raise ValueError("Frozen financial source changed")


def _source_audit(source, split, expected=128):
    audit = read(source / (split + "_audit.json"))
    if (audit.get("passed") is not True or audit.get("failure_count") != 0
            or audit.get("accounts_checked") != expected or audit.get("repeats_checked") != expected
            or audit.get("protocol_sha256") != sha(source / "protocol.json")
            or audit.get("summary_sha256") != sha(source / (split + "_summary.csv"))):
        raise ValueError("The reference must be an exact completed independently audited thirty-seed study")
    return audit


def _reference_cases(source, symbol, split):
    audit = _source_audit(source, split)
    wanted = {f"evaluation/{split}/{symbol}/sac/seed{s}/{c}" for s in FINAL_SEEDS for c in ("base", "stress")}
    wanted |= {f"evaluation/{split}/{symbol}/{a}/baseline/{c}" for a in BASELINES for c in ("base", "stress")}
    cases = {r["relative_path"]: r for r in audit["cases"] if r["relative_path"] in wanted}
    if set(cases) != wanted:
        raise ValueError("All original SAC and fixed-baseline reference accounts are required")
    files = {}
    for name, case in cases.items():
        if case.get("passed") is not True or case.get("failure_count") != 0:
            raise ValueError("Reference case audit failed")
        for file, expected in case["inputs_sha256"].items():
            if sha(safe(source, file)) != expected:
                raise ValueError("Reference financial evidence changed after its audit")
            files[file] = expected
    return files


def prepare(top, dataset, btc_reference, eth_reference, plan_path, *, audit_script=None, core_root=None):
    top, dataset = Path(top).resolve(), Path(dataset).resolve()
    refs = {"BTCUSDT": Path(btc_reference).resolve(), "ETHUSDT": Path(eth_reference).resolve()}
    if top.exists() or any(top.is_relative_to(p) or p.is_relative_to(top) for p in refs.values()):
        raise FileExistsError("Preserve the new output and both original reference studies")
    plan_path = Path(plan_path)
    plan, plan_receipt = read(plan_path), read(plan_path.with_name("search_plan_receipt.json"))
    if plan_receipt.get("sha256") != sha(plan_path) or plan.get("symbols") != list(SYMBOLS) or [r["candidate_id"] for r in plan["candidates"]] != list(CANDIDATES):
        raise ValueError("The exact original four-candidate search declaration is required")
    if plan["tuning_seeds"] != list(TUNING_SEEDS) or plan["comparison_seeds"] != list(FINAL_SEEDS) or plan["training_steps_per_model"] != 32768:
        raise ValueError("Predeclared tuning/final seeds and SAC interaction budget changed")
    old = {symbol: read(root / "protocol.json") for symbol, root in refs.items()}
    for symbol, spec in old.items():
        if (read(refs[symbol] / "protocol_receipt.json")["protocol_sha256"] != sha(refs[symbol] / "protocol.json")
                or spec.get("symbols") != [symbol] or spec.get("seeds") != list(FINAL_SEEDS)
                or spec.get("capital_per_account") != 10000. or spec.get("ensemble") is not False
                or spec.get("capital_shared_between_pairs") is not False or spec.get("training_model_count") != 60):
            raise ValueError("Original independent thirty-seed reference contract changed")
        _source_audit(refs[symbol], "validation")
        _source_audit(refs[symbol], "test")
    for key in FINANCIAL_FIELDS:
        if old["BTCUSDT"].get(key) != old["ETHUSDT"].get(key):
            raise ValueError("The two original studies do not share the same financial/runtime contract: " + key)
    base = old["BTCUSDT"]
    core = core_module(core_root or refs["BTCUSDT"] / "frozen_source")
    models = adapter()
    if (base["sources"] != plan["frozen_core_sources"] or sha(dataset) != plan["dataset_sha256"]
            or sha(dataset) != base["dataset_sha256"] or sha(dataset.parent / "manifest.json") != base["dataset_manifest_sha256"]):
        raise ValueError("Original market data or eleven frozen sources differ from the declared plan")
    for key in ("train", "validation", "test", "costs", "environment", "capital_per_account", "runtime_packages"):
        if plan[key] != base[key]:
            raise ValueError("The search declaration changed the financial contract: " + key)
    recipes = {r["candidate_id"]: r["native_recipe"] for r in plan["candidates"]}
    if recipes != {c: models.recipe(c, core.ROOT) for c in CANDIDATES} or any(old[s]["recipes"]["sac"] != recipes["c00_baseline"] for s in SYMBOLS):
        raise ValueError("Only learning rate/gamma may differ from the original SAC recipe")
    runtime = {**models.base_models(core.ROOT).runtime_versions(), "pandas": importlib.metadata.version("pandas")}
    if runtime != base["runtime_packages"] or sys.version != base["python"]:
        raise ValueError("The original native runtime must remain unchanged")
    for name, expected in base["sources"].items():
        if sha(safe(core.ROOT, name)) != expected:
            raise ValueError("Executed financial source differs from the original study")
    source_entries = {s: {"source_root": str(refs[s]), "protocol_sha256": sha(refs[s] / "protocol.json"),
        "validation_audit_sha256": sha(refs[s] / "validation_audit.json"), "test_audit_sha256": sha(refs[s] / "test_audit.json")} for s in SYMBOLS}
    spec = {key: deepcopy(base[key]) for key in FINANCIAL_FIELDS}
    spec.update(schema_version=1, study="SAC finite validation-only optimization", frozen_at=now(), symbols=list(SYMBOLS),
        algorithm="sac", primary_seed=0, primary_seed_fixed_before_training=True, search_plan_sha256=sha(plan_path),
        base_recipe=recipes["c00_baseline"], candidate_order=list(CANDIDATES), candidate_recipes=recipes,
        candidate_overrides=deepcopy(models.CANDIDATE_OVERRIDES), tuning_seeds=list(TUNING_SEEDS), final_seeds=list(FINAL_SEEDS),
        steps_per_model=32768, baselines=list(BASELINES), reference_sources=source_entries,
        training_window_quality={s: deepcopy(old[s]["training_window_quality"][s]) for s in SYMBOLS},
        adapter_source_sha256=sha(models.__file__), base_adapter_sha256=models.BASE_ADAPTER_SHA256,
        selection_contract={k: deepcopy(plan[k]) for k in ("score", "acceptance", "subwindows", "subwindow_metric", "tie_break", "fallback")})
    top.mkdir(parents=True)
    copy_file(plan_path, top / "search_plan.json")
    copy_file(plan_path.with_name("search_plan_receipt.json"), top / "search_plan_receipt.json")
    _copy_core(core.ROOT, top, spec["sources"])
    write_new(top / "protocol.json", spec)
    write_new(top / "protocol_receipt.json", {"sha256": sha(top / "protocol.json"), "protocol_sha256": sha(top / "protocol.json"), "at": now()})
    extension_files = {EXTENSION_NAMES[0]: Path(__file__), EXTENSION_NAMES[1]: Path(models.__file__),
                       EXTENSION_NAMES[2]: Path(audit_script or ROOT / EXTENSION_NAMES[2])}
    extension_sha = {name: copy_file(path, safe(top / "extension_source", name)) for name, path in extension_files.items()}
    write_new(top / "extension_manifest.json", {"at": now(), "extension_sources": extension_sha,
        "protocol_sha256": sha(top / "protocol.json"), "search_plan_sha256": sha(top / "search_plan.json"),
        "runtime_overlay": {"native_model_seed_whitelist_only": {"tuning": list(TUNING_SEEDS), "final": list(FINAL_SEEDS)},
                            "financial_math_changed": False, "default_fallback_retrained": False}})
    reference_files = {}
    for symbol, source in refs.items():
        relative_root = f"references/{symbol}"
        for name in ("protocol.json", "protocol_receipt.json", "validation_audit.json"):
            relative = relative_root + "/" + name
            reference_files[relative] = copy_file(source / name, safe(top, relative))
        for seed in FINAL_SEEDS:
            relative = f"training/{symbol}/sac/seed{seed}"
            core.validate_training_record(source, symbol, "sac", seed, old[symbol])
            files = copy_tree(safe(source, relative), safe(top, relative_root + "/" + relative))
            reference_files.update({relative_root + "/" + relative + "/" + name: digest for name, digest in files.items()})
        for name, expected in _reference_cases(source, symbol, "validation").items():
            relative = relative_root + "/" + name
            if copy_file(safe(source, name), safe(top, relative)) != expected:
                raise ValueError("Reference validation copy changed")
            reference_files[relative] = expected
    write_new(top / "reference_manifest.json", {"at": now(), "sources": source_entries, "files_sha256": reference_files,
        "annual_imported": False, "no_reference_training": True, "original_models": 60})
    write_new(top / "reference_receipt.json", {"at": now(), "reference_manifest_sha256": sha(top / "reference_manifest.json")})
    for candidate in CANDIDATES:
        _prepare_child(top, candidate_root(top, candidate), "tuning", {s: candidate for s in SYMBOLS}, source_core=core.ROOT)
    return {"tuning_models": 40, "maximum_final_fresh_models": 60, "reference_models_copied": 60}


def _prepare_child(top, child, stage, recipe_ids, *, source_core=None):
    spec = read(Path(top) / "protocol.json")
    child_spec = {key: deepcopy(spec[key]) for key in FINANCIAL_FIELDS}
    child_spec.update(study="native SAC parameter " + stage, frozen_at=now(), stage=stage,
        symbols=list(SYMBOLS), algorithms=["sac"], seeds=list(TUNING_SEEDS if stage == "tuning" else FINAL_SEEDS),
        primary_seed=0, primary_seed_fixed_before_training=True, baselines=list(BASELINES),
        training_model_count=10 if stage == "tuning" else 60, timesteps_per_model={"sac": 32768},
        parent_protocol_sha256=sha(Path(top) / "protocol.json"), recipe_ids_by_symbol=recipe_ids,
        recipes_by_symbol={s: deepcopy(spec["candidate_recipes"][recipe_ids[s]]) for s in SYMBOLS},
        adapter_source_sha256=spec["adapter_source_sha256"], base_adapter_sha256=spec["base_adapter_sha256"],
        training_window_quality=spec["training_window_quality"],
        validation_selection={"seed_selection": False, "algorithm_selection": False, "mode": "fixed_sac_recipe_comparison"},
        candidate_id=recipe_ids["BTCUSDT"] if stage == "tuning" else None,
        selection_sha256=sha(Path(top) / "selection.json") if stage == "final" else None)
    child = Path(child)
    if child.exists():
        raise FileExistsError("Preserve an existing stage/candidate study")
    child.mkdir(parents=True)
    _copy_core(source_core or Path(top) / "frozen_source", child, spec["sources"])
    write_new(child / "protocol.json", child_spec)
    write_new(child / "protocol_receipt.json", {"protocol_sha256": sha(child / "protocol.json"), "at": now()})


def check_top(top, dataset):
    top, dataset = Path(top), Path(dataset)
    spec, manifest = read(top / "protocol.json"), read(top / "extension_manifest.json")
    if (read(top / "protocol_receipt.json")["protocol_sha256"] != sha(top / "protocol.json")
            or manifest["protocol_sha256"] != sha(top / "protocol.json")
            or read(top / "search_plan_receipt.json")["sha256"] != sha(top / "search_plan.json")
            or spec["search_plan_sha256"] != sha(top / "search_plan.json") or manifest["search_plan_sha256"] != spec["search_plan_sha256"]
            or spec["symbols"] != list(SYMBOLS) or spec["tuning_seeds"] != list(TUNING_SEEDS)
            or spec["final_seeds"] != list(FINAL_SEEDS) or spec["candidate_order"] != list(CANDIDATES)
            or spec["capital_per_account"] != 10000. or spec["capital_shared_between_pairs"] is not False or spec["ensemble"] is not False
            or spec["steps_per_model"] != 32768 or set(manifest["extension_sources"]) != set(EXTENSION_NAMES)):
        raise ValueError("Immutable finite-search protocol/declaration changed")
    for name, expected in manifest["extension_sources"].items():
        if sha(safe(top / "extension_source", name)) != expected or sha(safe(ROOT, name)) != expected:
            raise ValueError("Frozen or running optimization source changed: " + name)
    core = core_module(top / "frozen_source")
    for name, expected in spec["sources"].items():
        if sha(safe(core.ROOT, name)) != expected:
            raise ValueError("Original financial source changed")
    models = adapter()
    if (models.sha(models.__file__) != spec["adapter_source_sha256"] or spec["candidate_recipes"] != {c: models.recipe(c, core.ROOT) for c in CANDIDATES}
            or sha(dataset) != spec["dataset_sha256"] or sha(dataset.parent / "manifest.json") != spec["dataset_manifest_sha256"]
            or {**models.base_models(core.ROOT).runtime_versions(), "pandas": importlib.metadata.version("pandas")} != spec["runtime_packages"]
            or sys.version != spec["python"]):
        raise ValueError("Native recipes, runtime or original data changed")
    reference = read(top / "reference_manifest.json")
    if read(top / "reference_receipt.json")["reference_manifest_sha256"] != sha(top / "reference_manifest.json") or reference["sources"] != spec["reference_sources"]:
        raise ValueError("Original reference manifest changed")
    for name, expected in reference["files_sha256"].items():
        if sha(safe(top, name)) != expected:
            raise ValueError("Original reference evidence changed: " + name)
    plan = read(top / "search_plan.json")
    for key in ("train", "validation", "test", "costs", "environment", "capital_per_account", "runtime_packages", "dataset_sha256"):
        if spec.get(key) != plan.get(key):
            raise ValueError("The declared search cannot change the original financial contract: " + key)
    if spec["selection_contract"] != {key: plan[key] for key in ("score", "acceptance", "subwindows", "subwindow_metric", "tie_break", "fallback")}:
        raise ValueError("The fixed validation score or acceptance gates changed")
    return spec, core


def check_child(top, child):
    top, child = Path(top), Path(child)
    parent, spec = read(top / "protocol.json"), read(child / "protocol.json")
    if read(child / "protocol_receipt.json")["protocol_sha256"] != sha(child / "protocol.json") or spec["parent_protocol_sha256"] != sha(top / "protocol.json"):
        raise ValueError("Child stage protocol changed")
    for key in FINANCIAL_FIELDS:
        if spec.get(key) != parent.get(key):
            raise ValueError("Child financial/runtime contract changed: " + key)
    if spec.get("stage") not in ("tuning", "final"):
        raise ValueError("Only declared tuning and final child stages are allowed")
    seeds = TUNING_SEEDS if spec["stage"] == "tuning" else FINAL_SEEDS
    if (spec["symbols"] != list(SYMBOLS) or spec["algorithms"] != ["sac"] or spec["seeds"] != list(seeds)
            or spec["training_model_count"] != len(seeds) * 2 or spec["baselines"] != list(BASELINES)
            or spec["timesteps_per_model"] != {"sac": 32768} or spec["adapter_source_sha256"] != parent["adapter_source_sha256"]
            or spec["base_adapter_sha256"] != parent["base_adapter_sha256"]
            or spec["recipes_by_symbol"] != {s: parent["candidate_recipes"][spec["recipe_ids_by_symbol"][s]] for s in SYMBOLS}):
        raise ValueError("Child population/selected actual recipe changed")
    if spec["stage"] == "final" and spec["selection_sha256"] != sha(top / "selection.json"):
        raise ValueError("Final stage recipe selection changed")
    if spec["stage"] == "tuning" and (Path(child).resolve() != candidate_root(top, spec["candidate_id"]).resolve()
            or set(spec["recipe_ids_by_symbol"].values()) != {spec["candidate_id"]}):
        raise ValueError("The candidate subdirectory and both native recipes must have the same declared identity")
    return spec


def model_directory(child, symbol, seed):
    return Path(child) / f"training/{symbol}/sac/seed{seed}"


def imported_model_pin(child, symbol, seed):
    path = Path(child) / "import_manifest.json"
    if not path.exists():
        return None
    entries = [r for r in read(path)["entries"] if r["symbol"] == symbol and r["seed"] == seed]
    if len(entries) > 1:
        raise ValueError("Duplicate fallback model identity")
    return entries[0] if entries else None


def validate_model(top, child, symbol, seed):
    spec = check_child(top, child)
    directory = model_directory(child, symbol, seed)
    record, manifest = read(directory / "record.json"), read(directory / "checkpoint/single_pair_model.json")
    expected_id = {"symbol": symbol, "algorithm": "sac", "seed": seed}
    if any(record.get(k) != v or manifest.get(k) != v for k, v in expected_id.items()):
        raise ValueError("Model record/checkpoint identity changed")
    if (record.get("capital") != 10000. or record.get("ensemble") is not False or record.get("observation_shape") != [13]
            or record.get("action_shape") != [1] or record.get("checkpoint") != manifest
            or manifest.get("independent_account") is not True or manifest.get("multi_model_target_average") is not False
            or record["training"] != manifest["training"] or not record.get("files")):
        raise ValueError("A genuinely trained independent 13/1 account is required")
    for name, expected in record["files"].items():
        if sha(safe(directory, name)) != expected:
            raise ValueError("Checkpoint bytes changed")
    recipe_id = spec["recipe_ids_by_symbol"][symbol]
    pin = imported_model_pin(child, symbol, seed)
    if pin is not None:
        if spec["stage"] != "final" or recipe_id != "c00_baseline" or pin["origin"] != "retained_baseline_fallback":
            raise ValueError("Only a selected c00 fallback can retain original models")
        if tree(directory) != pin["file_sha256"] or tree(safe(top, pin["source_directory"])) != pin["file_sha256"]:
            raise ValueError("Retained original model tree changed")
        if manifest.get("adapter_sha256") != spec["base_adapter_sha256"] or manifest.get("recipe") != spec["recipes_by_symbol"][symbol]:
            raise ValueError("Retained default adapter/recipe changed")
        origin = "retained_baseline_fallback"
    else:
        expected_context = adapter().context(spec["stage"], symbol, seed, recipe_id, sha(Path(top) / "protocol.json"))
        if (manifest.get("schema_version") != 2 or manifest.get("adapter_sha256") != spec["adapter_source_sha256"]
                or manifest.get("base_adapter_sha256") != spec["base_adapter_sha256"] or manifest.get("model_context") != expected_context
                or manifest.get("recipe") != spec["recipes_by_symbol"][symbol] or manifest.get("nonfinancial_preflight") is not False
                or datetime.fromisoformat(record["started_at"]) < datetime.fromisoformat(spec["frozen_at"])):
            raise ValueError("Fresh candidate context/source/time/actual recipe changed")
        origin = "fresh"
    adapter().check_effective_parameters(record["training"], spec["recipes_by_symbol"][symbol])
    if datetime.fromisoformat(record["maximum_input_hour"]) >= datetime.fromisoformat(spec["train"][1]):
        raise ValueError("Training consumed data beyond the original cutoff")
    episodes = [r for r in record["training"]["training_episode_history"] if r["completed_steps"] > 0]
    if len(episodes) != 1024 or sum(r["completed_steps"] for r in episodes) != 32768:
        raise ValueError("All actual 1024 thirty-two-day training fragments are required")
    for episode in episodes:
        start, end, until = (datetime.fromisoformat(episode[k]) for k in ("start", "end_exclusive", "completed_until"))
        if (episode["completed_steps"] != 32 or episode["terminated"] is not False or episode["truncated"] is not True
                or start < datetime.fromisoformat(spec["train"][0]) or end > datetime.fromisoformat(spec["train"][1])
                or end != start + timedelta(days=32) or until != end):
            raise ValueError("Training fragments cannot cross causal cutoff or change timeout semantics")
    return record, origin


def train_one(task):
    top, child = Path(task["top"]), Path(task["child"])
    parent, core = read(top / "protocol.json"), core_module(top / "frozen_source")
    spec = check_child(top, child)
    symbol, seed = task["symbol"], task["seed"]
    if symbol not in SYMBOLS or type(seed) is not int or seed not in spec["seeds"] or task["parent_protocol_sha256"] != sha(top / "protocol.json"):
        raise ValueError("Undeclared training task")
    if spec["stage"] == "final" and spec["recipe_ids_by_symbol"][symbol] == "c00_baseline":
        raise ValueError("A default fallback must reuse original evidence instead of retraining")
    directory = model_directory(child, symbol, seed)
    directory.mkdir(parents=True, exist_ok=False)
    started = now()
    try:
        from scripts.backtest_btceth_rl import load_dataset
        from scripts.single_pair_rl_env import SinglePairEnv
        data = core.cropped_data(load_dataset(task["dataset"]), parent["train"][1])
        env = SinglePairEnv(data, symbol, *parent["train"], training=True, episode_days=32, record_history=False,
                            capital=10000., costs=parent["costs"]["base"], drawdown_penalty=.25, rebalance_band=.025)
        if env.training_window_metadata != parent["training_window_quality"][symbol]:
            raise ValueError("The original eligible complete training fragments changed")
        models = adapter()
        model = models.make_model(env, seed=seed, candidate_id=spec["recipe_ids_by_symbol"][symbol], stage=spec["stage"],
            parent_protocol_sha256=sha(top / "protocol.json"), core_root=core.ROOT, device="cuda:0")
        training = models.train_model(model, env, core_root=core.ROOT, progress_path=directory / "progress.jsonl")
        checkpoint = models.save_model(model, directory / "checkpoint", core_root=core.ROOT)
        record = {"symbol": symbol, "algorithm": "sac", "seed": seed, "candidate_id": spec["recipe_ids_by_symbol"][symbol],
            "stage": spec["stage"], "started_at": started, "completed_at": now(), "capital": 10000., "ensemble": False,
            "primary_seed": seed == 0, "training_origin": "fresh", "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "pid": os.getpid(), "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
            "training": training, "checkpoint": checkpoint, "observation_shape": [13], "action_shape": [1],
            "maximum_input_hour": str(data["hourly_close"].index[-1]), "training_window_quality": env.training_window_metadata,
            "files": {p.relative_to(directory).as_posix(): sha(p) for p in (directory / "checkpoint").rglob("*") if p.is_file()}}
        write_new(directory / "record.json", record)
        return record
    except Exception as exc:
        write_new(directory / "failed.json", {"at": now(), "error_type": type(exc).__name__, "error": str(exc)})
        raise


def finish_fit(top, child):
    spec = check_child(top, child)
    records, rows = [], []
    for symbol in SYMBOLS:
        for seed in spec["seeds"]:
            record, origin = validate_model(top, child, symbol, seed)
            records.append(record)
            rows.append({"symbol": symbol, "algorithm": "sac", "seed": seed,
                "candidate_id": spec["recipe_ids_by_symbol"][symbol], "training_origin": origin,
                "retained": origin == "retained_baseline_fallback", "counters": record["training"]["counters"],
                "record_sha256": sha(model_directory(child, symbol, seed) / "record.json")})
    write_new(Path(child) / "fit_complete.json", {"at": now(), "model_count": len(records), "records": records})
    result = {"audited_at": now(), "passed": True, "models_checked": len(records), "models": rows,
        "retained_models": sum(r["retained"] for r in rows), "new_models": sum(not r["retained"] for r in rows),
        "failure_count": 0, "failures": [], "core_protocol_sha256": sha(Path(child) / "protocol.json"),
        "parent_protocol_sha256": sha(Path(top) / "protocol.json"), "extension_source_sha256": sha(__file__)}
    write_new(Path(child) / "training_counter_audit.json", result)
    return result


def fit(top, dataset, *, stage, max_workers=None):
    _, core = check_top(top, dataset)
    if stage == "final":
        _selection_gate(top)
    roots = [candidate_root(top, c) for c in CANDIDATES] if stage == "tuning" else [Path(top) / "final"]
    tasks = []
    for child in roots:
        spec = check_child(top, child)
        for seed in spec["seeds"]:
            for symbol in SYMBOLS:
                directory = model_directory(child, symbol, seed)
                if (directory / "record.json").exists():
                    validate_model(top, child, symbol, seed)
                    continue
                if directory.exists():
                    raise FileExistsError("Preserve incomplete training; no automatic restart")
                tasks.append({"top": str(top), "child": str(child), "dataset": str(dataset), "symbol": symbol,
                    "seed": seed, "parent_protocol_sha256": sha(Path(top) / "protocol.json")})
    if tasks:
        core.run_parallel(top, tasks, train_one, stage + "_fit", max_workers)
    results = [finish_fit(top, child) for child in roots]
    write_new(Path(top) / (stage + "_fit_complete.json"), {"at": now(), "fresh_tasks": len(tasks),
        "child_counter_audit_sha256": {child.relative_to(top).as_posix(): sha(child / "training_counter_audit.json") for child in roots}})
    return {"fresh_models": len(tasks), "completed_models": sum(r["models_checked"] for r in results)}


def _selection_gate(top):
    top = Path(top)
    selection, receipt, audited = read(top / "selection.json"), read(top / "selection_receipt.json"), read(top / "selection_audit.json")
    if (receipt.get("sha256") != sha(top / "selection.json") or selection.get("protocol_sha256") != sha(top / "protocol.json")
            or selection.get("uses_test") is not False or selection.get("seed_selection") is not False
            or selection.get("uses_final_validation") is not False or audited.get("passed") is not True
            or audited.get("failure_count") != 0
            or audited.get("inputs_sha256", {}).get("selection.json") != sha(top / "selection.json")):
        raise ValueError("A frozen independently audited validation-only parameter choice is required")
    _verify_audit_inputs(top, audited)
    return selection


def prepare_final(top, dataset):
    parent, _ = check_top(top, dataset)
    selection = _selection_gate(top)
    ids = {s: selection["selection"][s]["candidate_id"] for s in SYMBOLS}
    final = Path(top) / "final"
    _prepare_child(top, final, "final", ids)
    entries = []
    for symbol in SYMBOLS:
        if ids[symbol] != "c00_baseline":
            continue
        for seed in FINAL_SEEDS:
            source = f"references/{symbol}/training/{symbol}/sac/seed{seed}"
            destination = f"final/training/{symbol}/sac/seed{seed}"
            files = copy_tree(safe(top, source), safe(top, destination))
            entries.append({"symbol": symbol, "seed": seed, "candidate_id": "c00_baseline",
                "origin": "retained_baseline_fallback", "source_directory": source,
                "destination_directory": destination, "file_sha256": files})
    write_new(final / "import_manifest.json", {"at": now(), "selection_sha256": sha(Path(top) / "selection.json"), "entries": entries})
    write_new(Path(top) / "final_prepared.json", {"at": now(), "selection_sha256": sha(Path(top) / "selection.json"),
        "final_protocol_sha256": sha(final / "protocol.json"), "retained_models": len(entries), "new_models": 60 - len(entries)})
    return {"retained_models": len(entries), "new_models": 60 - len(entries)}


def import_annual_references(top):
    top = Path(top)
    _selection_gate(top)
    if (top / "reference_annual_import.json").exists():
        prior = read(top / "reference_annual_import.json")
        if prior["selection_sha256"] != sha(top / "selection.json"):
            raise ValueError("Annual reference import selection changed")
        for name, expected in prior["files_sha256"].items():
            if sha(safe(top, name)) != expected:
                raise ValueError("Previously imported annual reference changed")
        return
    files = {}
    spec = read(top / "protocol.json")
    for symbol in SYMBOLS:
        source = Path(spec["reference_sources"][symbol]["source_root"])
        if sha(source / "test_audit.json") != spec["reference_sources"][symbol]["test_audit_sha256"]:
            raise ValueError("The original annual reference audit changed")
        for name, expected in _reference_cases(source, symbol, "test").items():
            relative = f"references/{symbol}/" + name
            if copy_file(safe(source, name), safe(top, relative)) != expected:
                raise ValueError("Annual reference changed during import")
            files[relative] = expected
        relative = f"references/{symbol}/test_audit.json"
        files[relative] = copy_file(source / "test_audit.json", safe(top, relative))
    write_new(top / "reference_annual_import.json", {"at": now(), "selection_sha256": sha(top / "selection.json"),
        "selection_audit_sha256": sha(top / "selection_audit.json"), "files_sha256": files})


def _case_directory(split, symbol, algorithm, seed, scenario, kind="evaluation"):
    return f"{kind}/{split}/{symbol}/{algorithm}/{'baseline' if seed is None else 'seed'+str(seed)}/{scenario}"


def import_final_cases(top, split):
    top, final = Path(top), Path(top) / "final"
    _selection_gate(top)
    if split == "test":
        import_annual_references(top)
    spec = check_child(top, final)
    entries = []
    for symbol in SYMBOLS:
        cases = [(a, None, c, "retained_fixed_baseline") for a in BASELINES for c in spec["costs"]]
        if spec["recipe_ids_by_symbol"][symbol] == "c00_baseline":
            cases += [("sac", seed, c, "retained_baseline_fallback") for seed in FINAL_SEEDS for c in spec["costs"]]
        for algorithm, seed, scenario, origin in cases:
            for kind in ("evaluation", "repeat"):
                name = _case_directory(split, symbol, algorithm, seed, scenario, kind)
                source, destination = f"references/{symbol}/" + name, "final/" + name
                files = copy_tree(safe(top, source), safe(top, destination))
                entries.append({"symbol": symbol, "algorithm": algorithm, "seed": seed, "scenario": scenario,
                    "split": split, "kind": kind, "origin": origin, "source_directory": source,
                    "destination_directory": destination, "file_sha256": files})
    write_new(final / (split + "_import_manifest.json"), {"at": now(), "selection_sha256": sha(top / "selection.json"), "entries": entries})


def evaluate_one(task):
    import numpy as np
    top, child = Path(task["top"]), Path(task["child"])
    core, spec = core_module(top / "frozen_source"), check_child(top, child)
    from scripts.backtest_btceth_rl import load_dataset
    from scripts.single_pair_rl_env import SinglePairEnv
    from scripts.crypto_rl_reporting import save_result, summarize, canonical_csv, result_fingerprints
    symbol, algorithm, seed, scenario, split = (task[k] for k in ("symbol", "algorithm", "seed", "scenario", "split"))
    if (symbol not in SYMBOLS or scenario not in spec["costs"] or split not in ("validation", "test")
            or split == "test" and spec["stage"] != "final"
            or seed is not None and (type(seed) is not int or seed not in spec["seeds"] or algorithm != "sac")
            or seed is None and algorithm not in BASELINES):
        raise ValueError("Undeclared stage/account/cost replay")
    bounds = spec[split]
    data = core.cropped_data(load_dataset(task["dataset"]), bounds[1])
    models = adapter()
    model = None
    if seed is not None:
        validate_model(top, child, symbol, seed)
        model = models.load_model(model_directory(child, symbol, seed) / "checkpoint", symbol=symbol, seed=seed,
            candidate_id=spec["recipe_ids_by_symbol"][symbol], stage=spec["stage"], parent_protocol_sha256=sha(top / "protocol.json"),
            core_root=core.ROOT, device="cuda:0", retained_pin=imported_model_pin(child, symbol, seed))
    before = models.parameter_digest(model, core.ROOT) if model is not None else None
    fingerprints, action_hashes = [], []
    run_id = f"{split}__{scenario}__{symbol}__{algorithm}__{'baseline' if seed is None else seed}"
    for repeated in (False, True):
        env = SinglePairEnv(data, symbol, *bounds, costs=spec["costs"][scenario], capital=10000., drawdown_penalty=.25, rebalance_band=.025)
        obs, _ = env.reset(seed=0)
        first, done, actions = True, False, []
        while not done:
            day = env.current_date
            if model is not None:
                action = models.predict(model, obs, core_root=core.ROOT)
                obs, _, terminated, truncated, info = env.step(action)
                actions.append({"date": str(day), "action": np.asarray(action).reshape(-1).tolist(), "target": info["target_weights"].tolist()})
            elif algorithm == "buy_hold" and not first:
                obs, _, terminated, truncated, _ = env.step_hold()
            else:
                on = algorithm == "buy_hold" or algorithm == "sma200_trend" and env._features[day][7] > 0 or algorithm == "momentum180" and env._features[day][4] > 0
                obs, _, terminated, truncated, _ = env.step_target([.95 if on else 0., .05 if on else 1.])
            first, done = False, terminated or truncated
        result = env.get_result()
        directory = child / _case_directory(split, symbol, algorithm, seed, scenario, "repeat" if repeated else "evaluation")
        directory.mkdir(parents=True, exist_ok=False)
        save_result(directory, result)
        write_new(directory / "actions.json", actions)
        row, daily, monthly = summarize(result, run_id=run_id, strategy=algorithm, seed=seed, scenario=scenario, split=split,
            start=bounds[0], end=bounds[1], capital=10000., **spec["costs"][scenario])
        row.update(symbol=symbol, algorithm=algorithm, primary_seed=seed == 0, ensemble=False, independent_capital=True)
        write_new(directory / "summary.json", row)
        (directory / "daily_equity.csv").write_bytes(canonical_csv(daily))
        (directory / "monthly_returns.csv").write_bytes(canonical_csv(monthly))
        fingerprints.append(result_fingerprints(result))
        action_hashes.append(sha(directory / "actions.json"))
    if fingerprints[0] != fingerprints[1] or action_hashes[0] != action_hashes[1] or model is not None and models.parameter_digest(model, core.ROOT) != before:
        raise ValueError("Deterministic account replay or unchanged policy check failed")
    return _case_result(child, split, symbol, algorithm, seed, scenario)


def _case_result(child, split, symbol, algorithm, seed, scenario):
    canonical = _case_directory(split, symbol, algorithm, seed, scenario)
    repeated = _case_directory(split, symbol, algorithm, seed, scenario, "repeat")
    for name in RESULT_FILES:
        if sha(Path(child) / canonical / name) != sha(Path(child) / repeated / name):
            raise ValueError("Every retained/new canonical account must equal its own repeated bytes")
    return {"summary": read(Path(child) / canonical / "summary.json"), "path": canonical, "repeat_path": repeated, "repeat_equal": True}


def _verify_audit_inputs(top, audited):
    if (audited.get("passed") is not True or audited.get("failure_count") != 0
            or not isinstance(audited.get("inputs_sha256"), dict) or not audited["inputs_sha256"]):
        raise ValueError("Independent optimization audit did not pass")
    for name, expected in audited.get("inputs_sha256", {}).items():
        if sha(safe(top, name)) != expected:
            raise ValueError("Audited optimization input changed: " + name)


def evaluate(top, dataset, *, stage, split="validation", max_workers=None):
    import pandas as pd
    _, core = check_top(top, dataset)
    from scripts.crypto_rl_reporting import canonical_csv
    roots = [candidate_root(top, c) for c in CANDIDATES] if stage == "tuning" else [Path(top) / "final"]
    if stage == "tuning" and split != "validation":
        raise ValueError("Tuning has no annual replay entry")
    if stage == "final":
        _selection_gate(top)
        if split == "test":
            _verify_audit_inputs(top, read(Path(top) / "final/validation_optimization_audit.json"))
            # The original arithmetic oracle reads this original algorithm-level
            # schema. Parameters were already frozen at TOP/selection.json.
            write_new(Path(top) / "final/validation_selection.json", {"at": now(), "uses_test": False,
                "seed_selection": False, "algorithm_selection": False, "mode": "fixed_sac_recipe_comparison",
                "selection": {s: {"algorithm": "sac", "seed": 0} for s in SYMBOLS},
                "top_selection_sha256": sha(Path(top) / "selection.json"),
                "protocol_sha256": sha(Path(top) / "final/protocol.json"),
                "validation_audit_sha256": sha(Path(top) / "final/validation_audit.json"),
                "validation_optimization_audit_sha256": sha(Path(top) / "final/validation_optimization_audit.json")})
            write_new(Path(top) / "final/test_started.json", {"at": now(), "pins": {
                "selection_sha256": sha(Path(top) / "selection.json"), "selection_audit_sha256": sha(Path(top) / "selection_audit.json"),
                "validation_audit_sha256": sha(Path(top) / "final/validation_audit.json"),
                "validation_optimization_audit_sha256": sha(Path(top) / "final/validation_optimization_audit.json"),
                "protocol_sha256": sha(Path(top) / "protocol.json"), "final_protocol_sha256": sha(Path(top) / "final/protocol.json")},
                "oracle_compatibility_selection_sha256": sha(Path(top) / "final/validation_selection.json")})
        import_final_cases(top, split)
    tasks = []
    for child in roots:
        spec = check_child(top, child)
        counter = read(child / "training_counter_audit.json")
        if counter.get("passed") is not True or counter.get("models_checked") != spec["training_model_count"]:
            raise ValueError("All native training must pass its counter audit first")
        for symbol in SYMBOLS:
            for algorithm, seed in [("sac", s) for s in spec["seeds"]] + [(a, None) for a in BASELINES]:
                for scenario in spec["costs"]:
                    existing = child / _case_directory(split, symbol, algorithm, seed, scenario)
                    if existing.exists():
                        if stage != "final":
                            raise FileExistsError("Preserve every attempted tuning case")
                        _case_result(child, split, symbol, algorithm, seed, scenario)
                        continue
                    tasks.append({"top": str(top), "child": str(child), "dataset": str(dataset), "symbol": symbol,
                        "algorithm": algorithm, "seed": seed, "scenario": scenario, "split": split})
    if tasks:
        core.run_parallel(top, tasks, evaluate_one, stage + "_" + split, max_workers)
    for child in roots:
        spec = check_child(top, child)
        results = [_case_result(child, split, symbol, a, seed, scenario) for symbol in SYMBOLS
            for a, seed in [("sac", s) for s in spec["seeds"]] + [(a, None) for a in BASELINES] for scenario in spec["costs"]]
        expected = 36 if stage == "tuning" else 136
        if len(results) != expected:
            raise ValueError("Complete stage/cost population is mandatory")
        summary = pd.DataFrame([r["summary"] for r in results]).sort_values("run_id")
        (child / (split + "_summary.csv")).write_bytes(canonical_csv(summary))
        write_new(child / (split + "_complete.json"), {"at": now(), "accounts": len(results), "repeats": len(results),
            "repeat_equal": True, "results": results})
    write_new(Path(top) / (stage + "_" + split + "_complete.json"), {"at": now(), "fresh_tasks": len(tasks),
        "child_summary_sha256": {p.relative_to(top).as_posix(): sha(p / (split + "_summary.csv")) for p in roots}})
    return {"fresh_accounts": len(tasks), "accounts": 144 if stage == "tuning" else 136}


def _utility(ret, drawdown):
    return math.log1p(float(ret)) - .5 * abs(float(drawdown)) - 2 * max(abs(float(drawdown)) - .20, 0.)


def candidate_metrics(child, symbol, subwindows):
    import numpy as np
    import pandas as pd
    child = Path(child)
    frame = pd.read_csv(child / "validation_summary.csv", float_precision="round_trip")
    group = frame.loc[(frame.symbol == symbol) & (frame.algorithm == "sac")]
    if len(group) != 10 or set(group.seed) != set(TUNING_SEEDS):
        raise ValueError("Exactly five distinct tuning seeds at both actual costs are required")
    scores, base_returns, stress_returns, stress_dd, windows = [], [], [], [], [[] for _ in subwindows]
    for seed in TUNING_SEEDS:
        pair = group.loc[group.seed == seed]
        if len(pair) != 2 or set(pair.scenario) != {"base", "stress"}:
            raise ValueError("A tuning seed must have its own actual account at both costs")
        by_cost = {r.scenario: r for r in pair.itertuples()}
        scores.append(float(np.mean([_utility(r.net_return, r.max_drawdown) for r in by_cost.values()])))
        base_returns.append(float(by_cost["base"].net_return))
        stress_returns.append(float(by_cost["stress"].net_return))
        stress_dd.append(abs(float(by_cost["stress"].max_drawdown)))
        per_window = [[] for _ in subwindows]
        for cost in ("base", "stress"):
            path = child / _case_directory("validation", symbol, "sac", seed, cost) / "equity_1h.csv.gz"
            equity = pd.read_csv(path, float_precision="round_trip")
            equity["timestamp"] = pd.to_datetime(equity.timestamp, utc=True)
            equity = equity.set_index("timestamp").equity
            for i, (start, end) in enumerate(subwindows):
                start, end = pd.Timestamp(start), pd.Timestamp(end)
                before = 10000. if i == 0 else float(equity.loc[start - pd.Timedelta(hours=1)])
                selected = equity.loc[(equity.index >= start) & (equity.index < end)].to_numpy(dtype=float)
                if len(selected) != int((end - start) / pd.Timedelta(hours=1)):
                    raise ValueError("Continuous validation subwindow must retain every original hourly mark")
                nav = np.r_[before, selected]
                ret, dd = float(nav[-1] / before - 1), float(np.min(nav / np.maximum.accumulate(nav) - 1))
                per_window[i].append(_utility(ret, dd))
        for i, values in enumerate(per_window):
            windows[i].append(float(np.mean(values)))
    med, iqr = float(np.median(scores)), float(np.quantile(scores, .75, method="linear") - np.quantile(scores, .25, method="linear"))
    return {"seed_scores": scores, "score_median": med, "score_iqr": iqr, "robust_score": med - .25 * iqr,
        "base_positive_seeds": sum(r > 0 for r in base_returns), "stress_median_return": float(np.median(stress_returns)),
        "stress_p90_drawdown": float(np.quantile(stress_dd, .9, method="linear")),
        "subwindow_median_scores": [float(np.median(v)) for v in windows]}


def choose_parameters(top, dataset):
    top = Path(top)
    spec, _ = check_top(top, dataset)
    if (top / "final").exists() or (top / "selection.json").exists():
        raise FileExistsError("Parameter choice freezes once before any final account exists")
    plan = read(top / "search_plan.json")
    evidence = {}
    for candidate in CANDIDATES:
        root = candidate_root(top, candidate)
        audited = read(root / "validation_optimization_audit.json")
        _verify_audit_inputs(top, audited)
        financial = read(root / "validation_audit.json")
        if financial.get("passed") is not True or financial.get("accounts_checked") != 36 or financial.get("repeats_checked") != 36:
            raise ValueError("Every candidate's complete independent validation audit must pass")
        evidence[candidate] = {"summary": sha(root / "validation_summary.csv"), "financial_audit": sha(root / "validation_audit.json"),
            "contract_audit": sha(root / "validation_optimization_audit.json")}
    choices = {}
    for symbol in SYMBOLS:
        metrics = {c: candidate_metrics(candidate_root(top, c), symbol, plan["subwindows"]) for c in CANDIDATES}
        baseline, gates = metrics["c00_baseline"], {}
        for candidate, value in metrics.items():
            gates[candidate] = {
                "score_improvement_at_least": value["robust_score"] >= baseline["robust_score"] + .01,
                "paired_better_seed_count_at_least": sum(a > b for a, b in zip(value["seed_scores"], baseline["seed_scores"])) >= 3,
                "positive_base_seed_count_at_least": value["base_positive_seeds"] >= 3,
                "stress_return_median_at_least": value["stress_median_return"] >= 0,
                "stress_p90_drawdown_above_baseline_at_most": value["stress_p90_drawdown"] <= baseline["stress_p90_drawdown"] + .01,
                "nonworse_subwindows_at_least": sum(a >= b for a, b in zip(value["subwindow_median_scores"], baseline["subwindow_median_scores"])) >= 2}
        eligible = [c for c in CANDIDATES if c == "c00_baseline" or all(gates[c].values())]
        eligible.sort(key=lambda c: (-metrics[c]["robust_score"], c != "c00_baseline", c))
        selected = eligible[0]
        choices[symbol] = {"candidate_id": selected, "selected_recipe": spec["candidate_recipes"][selected],
            "fallback_to_baseline": selected == "c00_baseline", "metrics_by_candidate": metrics,
            "gates_by_candidate": gates, "eligible_candidates": eligible}
    result = {"at": now(), "uses_test": False, "uses_final_validation": False, "seed_selection": False,
        "algorithm_selection": False, "tuning_seeds": list(TUNING_SEEDS), "final_seeds": list(FINAL_SEEDS),
        "selection": choices, "protocol_sha256": sha(top / "protocol.json"), "validation_evidence_sha256": evidence,
        "search_plan_sha256": sha(top / "search_plan.json")}
    write_new(top / "selection.json", result)
    write_new(top / "selection_receipt.json", {"at": now(), "sha256": sha(top / "selection.json")})
    return {s: choices[s]["candidate_id"] for s in SYMBOLS}


def pipeline(top, dataset, *, max_workers=None):
    top = Path(top).resolve()
    _, core = check_top(top, dataset)
    script = safe(top / "extension_source", EXTENSION_NAMES[0])
    audit = safe(top / "extension_source", EXTENSION_NAMES[2])
    stages = [("tuning-fit", top / "tuning_fit_complete.json"), ("tuning-validation", top / "tuning_validation_complete.json")]
    stages += [("audit-tuning:" + c, candidate_root(top, c) / "validation_optimization_audit.json") for c in CANDIDATES]
    stages += [("select", top / "selection.json"), ("audit-selection", top / "selection_audit.json"),
        ("final-prepare", top / "final_prepared.json"), ("final-fit", top / "final_fit_complete.json"),
        ("final-validation", top / "final_validation_complete.json"), ("audit-final-validation", top / "final/validation_optimization_audit.json"),
        ("final-test", top / "final_test_complete.json"), ("audit-final-test", top / "final/test_optimization_audit.json"),
        ("audit-completion", top / "completion_contract_audit.json")]
    for phase, marker in stages:
        if marker.exists():
            if phase.startswith("audit-"):
                _verify_audit_inputs(top, read(marker))
            continue
        if phase.startswith("audit-"):
            mode = "tuning-validation" if phase.startswith("audit-tuning:") else phase.removeprefix("audit-")
            command = [sys.executable, "-B", str(audit), "--study-root", str(top), "--dataset", str(dataset), "--phase", mode]
            if phase.startswith("audit-tuning:"):
                command += ["--candidate", phase.split(":", 1)[1]]
        else:
            command = [sys.executable, "-B", str(script), phase, "--output", str(top), "--dataset", str(dataset)]
            if max_workers:
                command += ["--max-workers", str(max_workers)]
        core.write(top / "pipeline_status.json", {"at": now(), "phase": phase, "status": "running", "pid": os.getpid()})
        log_name = phase.replace(":", "_") + ".log"
        with (top / log_name).open("x", encoding="utf-8") as stream:
            completed = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
        if completed.returncode or not marker.exists():
            core.write(top / "pipeline_status.json", {"at": now(), "phase": phase, "status": "failed", "exit_code": completed.returncode})
            raise RuntimeError("Preserve failed optimization phase; no automatic restart: " + phase)
        if phase.startswith("audit-"):
            _verify_audit_inputs(top, read(marker))
        check_top(top, dataset)
    core.write(top / "pipeline_status.json", {"at": now(), "status": "completed", "no_production_change": True})
    return {"completed": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "pipeline", "tuning-fit", "tuning-validation", "select", "final-prepare", "final-fit", "final-validation", "final-test"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--btc-reference", type=Path)
    parser.add_argument("--eth-reference", type=Path)
    parser.add_argument("--search-plan", type=Path)
    parser.add_argument("--audit-script", type=Path)
    parser.add_argument("--core-source-root", type=Path)
    parser.add_argument("--max-workers", type=int)
    args = parser.parse_args()
    if args.phase == "prepare":
        if any(v is None for v in (args.btc_reference, args.eth_reference, args.search_plan)):
            parser.error("Preparation requires both immutable reference roots and the declared search plan")
        result = prepare(args.output, args.dataset, args.btc_reference, args.eth_reference, args.search_plan,
            audit_script=args.audit_script, core_root=args.core_source_root)
    elif args.phase == "pipeline":
        result = pipeline(args.output, args.dataset, max_workers=args.max_workers)
    elif args.phase in ("tuning-fit", "final-fit"):
        result = fit(args.output, args.dataset, stage=args.phase.split("-")[0], max_workers=args.max_workers)
    elif args.phase == "select":
        result = choose_parameters(args.output, args.dataset)
    elif args.phase == "final-prepare":
        result = prepare_final(args.output, args.dataset)
    else:
        result = evaluate(args.output, args.dataset, stage=args.phase.split("-")[0],
                          split="test" if args.phase == "final-test" else "validation", max_workers=args.max_workers)
    print(json.dumps({"phase": args.phase, "at": now(), "result": result}, ensure_ascii=False))


if __name__ == "__main__":
    main()

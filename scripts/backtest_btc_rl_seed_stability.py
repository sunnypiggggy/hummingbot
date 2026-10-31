"""Predeclared BTC-only SAC/TD3 seed expansion; frozen financial code is reused.

Only orchestration, protocol population checks and seed whitelists change here.
The original model recipes, model implementation, environment and independent
accounting oracle remain byte-identical. Existing studies are never rewritten.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SYMBOLS = ("BTCUSDT",)
ALGORITHMS = ("td3", "sac")
SEEDS = tuple(range(30))
OLD_SEEDS = tuple(range(5))
NEW_SEEDS = tuple(range(5, 30))
MODEL_COUNT, RETAINED_COUNT, NEW_COUNT, ACCOUNT_COUNT = 60, 10, 50, 128
FINANCIAL_FIELDS = ("train", "validation", "test", "primary_seed", "capital_per_account",
    "capital_shared_between_pairs", "ensemble", "costs", "environment", "sources", "runtime_packages",
    "baselines", "research_gates_per_pair", "python", "no_production_change", "fresh_holdout",
    "test_year_already_observed", "primary_seed_fixed_before_training", "dataset_sha256", "dataset_manifest_sha256")
RUNTIME_OVERLAY = {"core_symbols": list(SYMBOLS), "core_seeds": list(SEEDS), "model_seeds": list(SEEDS),
    "oracle_symbols": list(SYMBOLS), "oracle_seeds": list(SEEDS),
    "worker_apply_policy": "each spawned training and evaluation worker applies this exact whitelist",
    "financial_math_changed": False, "model_recipes_changed": False, "performance_mode": "native; no performance patches"}
SEED_EXTENSION = {"previous_seeds": list(OLD_SEEDS), "new_seeds": list(NEW_SEEDS), "full_seeds": list(SEEDS),
    "retained_model_count": RETAINED_COUNT, "new_model_count": NEW_COUNT,
    "primary_analysis_cohort": "new25", "no_seed_selection": True}
DRIVER_NAME = "scripts/backtest_btc_rl_seed_stability.py"
AUDIT_NAME = "scripts/audit_btc_rl_seed_stability.py"


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
    name = str(name)
    parsed = PurePosixPath(name)
    if not name or parsed.as_posix() != name or parsed.is_absolute() or ".." in parsed.parts or ":" in name or "\\" in name:
        raise ValueError("Evidence requires a canonical relative POSIX path")
    raw_root = Path(root).absolute()
    target = raw_root.joinpath(*parsed.parts)
    for path in (target, *target.parents):
        if path.is_symlink() or hasattr(path, "is_junction") and path.is_junction():
            raise ValueError("Evidence cannot cross a link")
    target.resolve().relative_to(raw_root.resolve())
    return target


def model_files(directory):
    directory = Path(directory)
    result = {}
    for path in sorted(directory.rglob("*")):
        if path.is_symlink() or hasattr(path, "is_junction") and path.is_junction():
            raise ValueError("Model evidence cannot contain links")
        if path.is_file():
            result[path.relative_to(directory).as_posix()] = sha(path)
    return result


def core_module(core_source_root=None):
    root = Path(core_source_root or ROOT).resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    core = importlib.import_module("scripts.backtest_single_pair_rl")
    if Path(core.__file__).resolve() != (root / "scripts/backtest_single_pair_rl.py").resolve():
        raise ValueError("Use a clean process for the selected byte-identical frozen core")
    return core


def apply_runtime_overlay(core_source_root=None):
    """Apply only declared population whitelists in every fresh spawned worker."""
    core = core_module(core_source_root)
    models = importlib.import_module("scripts.single_pair_rl_models")
    if Path(models.__file__).resolve() != Path(core.ROOT) / "scripts/single_pair_rl_models.py":
        raise ValueError("Worker imported another model adapter")
    core.SYMBOLS, core.SEEDS = SYMBOLS, SEEDS
    models.SEEDS = SEEDS
    return core, models


def compatible(source, destination):
    if (source.get("symbols") != ["BTCUSDT", "ETHUSDT"] or source.get("seeds") != list(OLD_SEEDS)
            or source.get("training_model_count") != 60
            or set(source.get("algorithms", [])) != {"sac", "tqc", "crossq", "recurrent_ppo", "td3", "droq"}):
        raise ValueError("The retained source must be the completed original six-algorithm study")
    if (destination.get("symbols") != list(SYMBOLS) or destination.get("algorithms") != list(ALGORITHMS)
            or destination.get("seeds") != list(SEEDS) or destination.get("training_model_count") != MODEL_COUNT
            or destination.get("seed_extension") != SEED_EXTENSION or destination.get("runtime_overlay") != RUNTIME_OVERLAY):
        raise ValueError("Exactly BTC, two fixed algorithms and all thirty predeclared seeds are required")
    for key in FINANCIAL_FIELDS:
        if source.get(key) != destination.get(key):
            raise ValueError("Financial/training contract changed: " + key)
    if destination.get("training_window_quality") != {"BTCUSDT": source["training_window_quality"]["BTCUSDT"]}:
        raise ValueError("BTC's original complete-window training quality must remain unchanged")
    for key in ("recipes", "timesteps_per_model"):
        if destination.get(key) != {a: source[key][a] for a in ALGORITHMS}:
            raise ValueError("The original per-algorithm recipe or training budget changed")
    if destination.get("capital_per_account") != 10000. or destination.get("ensemble") is not False or destination.get("capital_shared_between_pairs") is not False:
        raise ValueError("Each model retains its own 10000 USDT spot account")


def _validate_source(source_root, dataset, core, check_runtime=True):
    source_root, dataset = Path(source_root), Path(dataset)
    source = read(source_root / "protocol.json")
    if read(source_root / "protocol_receipt.json")["protocol_sha256"] != sha(source_root / "protocol.json"):
        raise ValueError("Retained protocol receipt changed")
    if source["dataset_sha256"] != sha(dataset) or source["dataset_manifest_sha256"] != sha(dataset.parent / "manifest.json"):
        raise ValueError("The exact original market data is required")
    if set(source["sources"]) != set(core.SOURCE_FILES):
        raise ValueError("The complete original eleven source files are required")
    for name, expected in source["sources"].items():
        if sha(safe(core.ROOT, name)) != expected or sha(safe(source_root / "frozen_source", name)) != expected:
            raise ValueError("Retained or executed frozen source changed: " + name)
    if check_runtime:
        models = importlib.import_module("scripts.single_pair_rl_models")
        versions = {**models.runtime_versions(), "pandas": importlib.metadata.version("pandas")}
        if versions != source["runtime_packages"] or sys.version != source["python"]:
            raise ValueError("Use the identical frozen learner/runtime dependencies")
    fit = read(source_root / "fit_complete.json")
    if fit.get("model_count") != 60 or len(fit.get("records", [])) != 60:
        raise ValueError("The retained study must have all sixty completed models")
    for split in ("validation", "test"):
        audit = read(source_root / (split + "_audit.json"))
        if (audit.get("passed") is not True or audit.get("failure_count") != 0
                or audit.get("protocol_sha256") != sha(source_root / "protocol.json")
                or audit.get("summary_sha256") != sha(source_root / (split + "_summary.csv"))
                or audit.get("accounts_checked") != 136 or audit.get("repeats_checked") != 136):
            raise ValueError("Retained models require their complete original audited study")
    return source


def prepare(source_root, output, dataset, *, core_source_root=None, audit_script=None, check_runtime=True):
    source_root, output, dataset = Path(source_root).resolve(), Path(output).resolve(), Path(dataset).resolve()
    if output.is_relative_to(source_root) or source_root.is_relative_to(output):
        raise ValueError("New and retained studies must be disjoint")
    if output.exists():
        raise FileExistsError("Preserve an existing research output")
    core = core_module(core_source_root or source_root / "frozen_source")
    source = _validate_source(source_root, dataset, core, check_runtime)
    records = {(a, seed): core.validate_training_record(source_root, "BTCUSDT", a, seed, source)
               for a in ALGORITHMS for seed in OLD_SEEDS}
    audit_script = Path(audit_script or ROOT / AUDIT_NAME)
    if not audit_script.is_file():
        raise ValueError("Freeze the independent BTC seed-expansion audit entry before preparing")
    spec = deepcopy(source)
    spec.update(study="BTC single-pair SAC/TD3 seed stability", frozen_at=now(), symbols=list(SYMBOLS),
        algorithms=list(ALGORITHMS), seeds=list(SEEDS), training_model_count=MODEL_COUNT,
        recipes={a: deepcopy(source["recipes"][a]) for a in ALGORITHMS},
        timesteps_per_model={a: source["timesteps_per_model"][a] for a in ALGORITHMS},
        training_window_quality={"BTCUSDT": deepcopy(source["training_window_quality"]["BTCUSDT"])},
        seed_extension=deepcopy(SEED_EXTENSION), runtime_overlay=deepcopy(RUNTIME_OVERLAY),
        validation_selection={"seed_selection": False, "algorithm_selection": False, "mode": "fixed_algorithm_comparison",
            "algorithm_score": "descriptive only: median across30seeds of mean over base/stress(log1p(return)-.5*abs(DD)-2*max(abs(DD)-.20,0))",
            "selected_algorithm_primary_account": "no winner selected; both fixed algorithms retain predeclared seed0",
            "compatibility_anchor_algorithm": "sac"})
    compatible(source, spec)
    output.mkdir(parents=True)
    for name, expected in spec["sources"].items():
        target = safe(output / "frozen_source", name)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(safe(core.ROOT, name), target)
        if sha(target) != expected:
            raise ValueError("Frozen core copy changed")
    write_new(output / "protocol.json", spec)
    write_new(output / "protocol_receipt.json", {"protocol_sha256": sha(output / "protocol.json"), "frozen_at": now()})
    shutil.copyfile(source_root / "protocol.json", output / "retained_protocol.json")
    retained = []
    for (algorithm, seed), record in records.items():
        directory = f"training/BTCUSDT/{algorithm}/seed{seed}"
        original = safe(source_root, directory)
        before = model_files(original)
        shutil.copytree(original, safe(output, directory))
        if model_files(original) != before or model_files(safe(output, directory)) != before:
            raise ValueError("Retained model evidence changed while copying")
        retained.append({"symbol": "BTCUSDT", "algorithm": algorithm, "seed": seed, "source_directory": directory,
            "destination_directory": directory, "file_sha256": before, "source_record_sha256": sha(original / "record.json"),
            "model_parameter_sha256": record["checkpoint"]["parameter_sha256"], "source_completed_at": record["completed_at"]})
    imported_at = now()
    imported = {"schema_version": 1, "source_study_root": str(source_root),
        "source_protocol_sha256": sha(source_root / "protocol.json"), "destination_protocol_sha256": sha(output / "protocol.json"),
        "source_fit_complete_sha256": sha(source_root / "fit_complete.json"),
        "source_validation_audit_sha256": sha(source_root / "validation_audit.json"), "source_test_audit_sha256": sha(source_root / "test_audit.json"),
        "dataset_sha256": spec["dataset_sha256"], "dataset_manifest_sha256": spec["dataset_manifest_sha256"],
        "compatibility_fields": list(FINANCIAL_FIELDS), "retained_model_count": RETAINED_COUNT, "new_model_count": NEW_COUNT,
        "retained_models": retained, "new_models": [{"symbol": "BTCUSDT", "algorithm": a, "seed": seed} for a in ALGORITHMS for seed in NEW_SEEDS],
        "imported_at": imported_at, "import_source_sha256": sha(__file__), "evaluation_results_reused": False,
        "old_test_year_already_observed": True, "no_production_change": True}
    write_new(output / "training_import.json", imported)
    write_new(output / "training_import_receipt.json", {"training_import_sha256": sha(output / "training_import.json"), "imported_at": imported_at})
    sources = {DRIVER_NAME: Path(__file__), AUDIT_NAME: audit_script}
    frozen_sources = {}
    for name, original in sources.items():
        target = safe(output / "extension_source", name)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original, target)
        frozen_sources[name] = sha(target)
    write_new(output / "extension_manifest.json", {"schema_version": 1, "frozen_at": now(),
        "extension_sources": frozen_sources, "driver_relative_path": "extension_source/" + DRIVER_NAME,
        "driver_source_sha256": frozen_sources[DRIVER_NAME], "audit_entry_relative_path": "extension_source/" + AUDIT_NAME,
        "audit_entry_source_sha256": frozen_sources[AUDIT_NAME], "core_protocol_sha256": sha(output / "protocol.json"),
        "training_import_sha256": sha(output / "training_import.json"), "runtime_overlay": deepcopy(RUNTIME_OVERLAY),
        "retained_models": RETAINED_COUNT, "fresh_models": NEW_COUNT, "models": MODEL_COUNT,
        "accounts_per_phase": ACCOUNT_COUNT, "repeat_all_cases": True, "both_costs_each_model": True})
    validate_import(output, source_root)
    return {"retained": RETAINED_COUNT, "pending": NEW_COUNT, "models": MODEL_COUNT, "accounts_per_phase": ACCOUNT_COUNT}


def validate_import(output, source_root):
    output, source_root = Path(output).resolve(), Path(source_root).resolve()
    source, destination = read(source_root / "protocol.json"), read(output / "protocol.json")
    compatible(source, destination)
    imported = read(output / "training_import.json")
    if (read(output / "training_import_receipt.json")["training_import_sha256"] != sha(output / "training_import.json")
            or imported.get("schema_version") != 1 or imported.get("compatibility_fields") != list(FINANCIAL_FIELDS)
            or imported.get("retained_model_count") != RETAINED_COUNT or imported.get("new_model_count") != NEW_COUNT
            or imported.get("import_source_sha256") != sha(__file__)
            or imported.get("evaluation_results_reused") is not False or imported.get("no_production_change") is not True
            or imported.get("old_test_year_already_observed") is not True):
        raise ValueError("The complete unmodified seed-extension import receipt is required")
    for name in ("dataset_sha256", "dataset_manifest_sha256"):
        if imported.get(name) != destination[name]:
            raise ValueError("Imported market evidence changed")
    for field, root, name in (("source_protocol_sha256", source_root, "protocol.json"), ("destination_protocol_sha256", output, "protocol.json"),
            ("source_fit_complete_sha256", source_root, "fit_complete.json"), ("source_validation_audit_sha256", source_root, "validation_audit.json"),
            ("source_test_audit_sha256", source_root, "test_audit.json")):
        if imported.get(field) != sha(root / name):
            raise ValueError("Original/current provenance binding changed: " + field)
    if sha(output / "retained_protocol.json") != imported["source_protocol_sha256"]:
        raise ValueError("The original retained protocol bytes changed")
    wanted = {("BTCUSDT", a, s) for a in ALGORITHMS for s in OLD_SEEDS}
    observed = [(r["symbol"], r["algorithm"], r["seed"]) for r in imported["retained_models"]]
    pending = [(r["symbol"], r["algorithm"], r["seed"]) for r in imported["new_models"]]
    if len(observed) != RETAINED_COUNT or set(observed) != wanted or len(pending) != NEW_COUNT or set(pending) != {("BTCUSDT", a, s) for a in ALGORITHMS for s in NEW_SEEDS}:
        raise ValueError("All retained ten and new fifty identities must appear exactly once")
    for item in imported["retained_models"]:
        directory = f"training/BTCUSDT/{item['algorithm']}/seed{item['seed']}"
        if item["source_directory"] != directory or item["destination_directory"] != directory:
            raise ValueError("A retained model cannot be relabelled")
        original, copied = safe(source_root, directory), safe(output, directory)
        if model_files(original) != item["file_sha256"] or model_files(copied) != item["file_sha256"]:
            raise ValueError("Retained model bytes changed")
        record = read(original / "record.json")
        if (sha(original / "record.json") != item["source_record_sha256"] or record["completed_at"] != item["source_completed_at"]
                or record["checkpoint"]["parameter_sha256"] != item["model_parameter_sha256"]):
            raise ValueError("Retained model record/time/tensors changed")
    return imported


def check_protocol(output, dataset, *, core_source_root=None, check_runtime=True):
    output, dataset = Path(output), Path(dataset)
    core = core_module(core_source_root or output / "frozen_source")
    spec, receipt = read(output / "protocol.json"), read(output / "protocol_receipt.json")
    if receipt["protocol_sha256"] != sha(output / "protocol.json"):
        raise ValueError("Frozen protocol changed")
    if (spec.get("symbols") != list(SYMBOLS) or spec.get("algorithms") != list(ALGORITHMS) or spec.get("seeds") != list(SEEDS)
            or spec.get("training_model_count") != MODEL_COUNT or spec.get("seed_extension") != SEED_EXTENSION
            or spec.get("runtime_overlay") != RUNTIME_OVERLAY or spec.get("capital_per_account") != 10000.
            or spec.get("ensemble") is not False or spec.get("capital_shared_between_pairs") is not False
            or spec.get("primary_seed") != 0 or spec.get("primary_seed_fixed_before_training") is not True
            or spec.get("fresh_holdout") is not False or spec.get("test_year_already_observed") is not True):
        raise ValueError("Frozen thirty-seed independent BTC comparison changed")
    if (spec["train"] != list(core.TRAIN) or spec["validation"] != list(core.VALIDATION) or spec["test"] != list(core.TEST)
            or spec["costs"] != core.COSTS or spec["baselines"] != list(core.BASELINES)
            or spec["timesteps_per_model"] != {"sac": 32768, "td3": 65536}
            or spec["validation_selection"].get("mode") != "fixed_algorithm_comparison"
            or spec["validation_selection"].get("algorithm_selection") is not False
            or spec["validation_selection"].get("seed_selection") is not False):
        raise ValueError("Original dates/costs/budgets or fixed algorithm comparison changed")
    if spec["dataset_sha256"] != sha(dataset) or spec["dataset_manifest_sha256"] != sha(dataset.parent / "manifest.json"):
        raise ValueError("Original market data changed")
    if set(spec["sources"]) != set(core.SOURCE_FILES):
        raise ValueError("The complete unchanged financial/model source inventory is required")
    for name, expected in spec["sources"].items():
        if sha(safe(core.ROOT, name)) != expected or sha(safe(output / "frozen_source", name)) != expected:
            raise ValueError("Executed/frozen core source changed: " + name)
    manifest = read(output / "extension_manifest.json")
    if (manifest.get("core_protocol_sha256") != sha(output / "protocol.json")
            or manifest.get("training_import_sha256") != sha(output / "training_import.json")
            or manifest.get("runtime_overlay") != RUNTIME_OVERLAY or manifest.get("models") != MODEL_COUNT
            or manifest.get("retained_models") != RETAINED_COUNT or manifest.get("fresh_models") != NEW_COUNT
            or manifest.get("accounts_per_phase") != ACCOUNT_COUNT or manifest.get("repeat_all_cases") is not True
            or manifest.get("both_costs_each_model") is not True
            or manifest.get("driver_relative_path") != "extension_source/" + DRIVER_NAME
            or manifest.get("audit_entry_relative_path") != "extension_source/" + AUDIT_NAME
            or set(manifest.get("extension_sources", {})) != {DRIVER_NAME, AUDIT_NAME}
            or manifest.get("driver_source_sha256") != sha(__file__)):
        raise ValueError("Complete frozen extension manifest is required")
    for name, expected in manifest["extension_sources"].items():
        if sha(safe(output / "extension_source", name)) != expected:
            raise ValueError("Frozen extension source changed: " + name)
    if manifest["driver_source_sha256"] != manifest["extension_sources"][DRIVER_NAME] or manifest["audit_entry_source_sha256"] != manifest["extension_sources"][AUDIT_NAME]:
        raise ValueError("Extension entry source bindings differ")
    if check_runtime:
        models = importlib.import_module("scripts.single_pair_rl_models")
        versions = {**models.runtime_versions(), "pandas": importlib.metadata.version("pandas")}
        if versions != spec["runtime_packages"] or sys.version != spec["python"]:
            raise ValueError("Use the identical frozen native runtime")
        if spec["recipes"] != {a: models.recipe(a) for a in ALGORITHMS}:
            raise ValueError("Native training recipes changed")
    return spec


def _task_scope(task):
    if task.get("symbol") != "BTCUSDT" or task.get("protocol_sha256") != sha(Path(task["output"]) / "protocol.json"):
        raise ValueError("Worker task differs from the frozen independent BTC study")
    core, models = apply_runtime_overlay(task["core_root"])
    if task["seed"] is not None and (type(task["seed"]) is not int or task["seed"] not in SEEDS or task["algorithm"] not in ALGORITHMS):
        raise ValueError("Undeclared model seed/algorithm")
    return core, models


def train_one(task):
    core, models = _task_scope(task)
    if task["seed"] not in NEW_SEEDS:
        raise ValueError("Original seed0..4 checkpoints must not be retrained")
    record = core.train_one(task)
    write_new(Path(task["output"]) / f"training/BTCUSDT/{task['algorithm']}/seed{task['seed']}/runtime_overlay.json",
        {"at": now(), "symbol": "BTCUSDT", "algorithm": task["algorithm"], "seed": task["seed"], "pid": os.getpid(),
         "runtime_overlay": deepcopy(RUNTIME_OVERLAY), "actual_core_seeds": list(core.SEEDS), "actual_model_seeds": list(models.SEEDS),
         "actual_core_symbols": list(core.SYMBOLS), "driver_source_sha256": sha(__file__), "record_sha256": sha(Path(task["output"]) / f"training/BTCUSDT/{task['algorithm']}/seed{task['seed']}/record.json")})
    return record


def evaluate_case(task):
    core, _ = _task_scope(task)
    if task["seed"] is None and task["algorithm"] not in core.BASELINES:
        raise ValueError("Only four fixed baseline accounts are allowed")
    return core.evaluate_case(task)


def training_counter_audit(output, core, *, persist=True):
    spec = read(Path(output) / "protocol.json")
    rows, failures = [], []
    for algorithm in ALGORITHMS:
        for seed in SEEDS:
            try:
                record = core.validate_training_record(output, "BTCUSDT", algorithm, seed, spec)
                training, steps = record["training"], spec["timesteps_per_model"][algorithm]
                counters = training["counters"]
                updates = steps - 512
                optimizers = {"actor": updates if algorithm == "sac" else updates // 2, "critic": updates}
                if algorithm == "sac":
                    optimizers["entropy"] = updates
                if (counters.get("gradient_update_rounds") != updates or counters.get("optimizer_steps") != optimizers
                        or counters.get("replay_transitions") != steps or counters.get("replay_add_calls") != steps):
                    raise ValueError("Original native actor/critic/entropy/replay counts differ")
                episodes = [r for r in training["training_episode_history"] if r["completed_steps"] > 0]
                if len(episodes) != steps // 32 or sum(r["completed_steps"] for r in episodes) != steps:
                    raise ValueError("Actual training fragment population differs")
                for episode in episodes:
                    start, end, until = (datetime.fromisoformat(episode[k]) for k in ("start", "end_exclusive", "completed_until"))
                    if (episode["completed_steps"] != 32 or episode["terminated"] is not False or episode["truncated"] is not True
                            or start < datetime.fromisoformat(core.TRAIN[0]) or end > datetime.fromisoformat(core.TRAIN[1])
                            or end != start + timedelta(days=32) or until != end):
                        raise ValueError("Every actual completed fragment must remain a causal 32-day timeout")
                directory = Path(output) / f"training/BTCUSDT/{algorithm}/seed{seed}"
                row = {"symbol": "BTCUSDT", "algorithm": algorithm, "seed": seed, "counters": counters,
                    "record_sha256": sha(directory / "record.json"), "retained": seed in OLD_SEEDS}
                if seed in NEW_SEEDS:
                    applied = read(directory / "runtime_overlay.json")
                    if (applied["runtime_overlay"] != RUNTIME_OVERLAY or applied["actual_model_seeds"] != list(SEEDS)
                            or applied["actual_core_seeds"] != list(SEEDS) or applied["actual_core_symbols"] != list(SYMBOLS)
                            or applied["driver_source_sha256"] != sha(__file__) or applied["record_sha256"] != row["record_sha256"]
                            or any(applied[k] != row[k] for k in ("symbol", "algorithm", "seed"))
                            or datetime.fromisoformat(record["started_at"]) < datetime.fromisoformat(spec["frozen_at"])):
                        raise ValueError("Fresh worker scope/source/time provenance differs")
                    row["runtime_overlay_sha256"] = sha(directory / "runtime_overlay.json")
                rows.append(row)
            except Exception as exc:
                failures.append({"symbol": "BTCUSDT", "algorithm": algorithm, "seed": seed, "error": str(exc)})
    result = {"audited_at": now(), "passed": not failures and len(rows) == MODEL_COUNT, "models_checked": len(rows),
        "retained_models": RETAINED_COUNT, "new_models": NEW_COUNT, "failure_count": len(failures), "failures": failures,
        "models": rows, "extension_source_sha256": sha(__file__), "core_protocol_sha256": sha(Path(output) / "protocol.json")}
    if persist:
        core.write(Path(output) / "training_counter_audit.json", result)
    if not result["passed"]:
        raise ValueError("Native training counter/time/worker-scope audit failed")
    return result


def validate_fit_completion(output, core):
    """Read saved terminal evidence without rewriting a successful audit."""
    output = Path(output)
    fit, saved = read(output / "fit_complete.json"), read(output / "training_counter_audit.json")
    current = training_counter_audit(output, core, persist=False)
    for key in ("passed", "models_checked", "retained_models", "new_models", "failure_count", "failures", "models",
                "extension_source_sha256", "core_protocol_sha256"):
        if saved.get(key) != current[key]:
            raise ValueError("Saved training audit no longer matches all sixty current model records")
    spec = read(output / "protocol.json")
    records = [core.validate_training_record(output, "BTCUSDT", a, seed, spec) for a in ALGORITHMS for seed in SEEDS]
    if fit.get("model_count") != MODEL_COUNT or fit.get("records") != records:
        raise ValueError("Training completion must contain every exact completed model record")
    return fit, saved


def fit(output, dataset, source_root, *, core_source_root=None, max_workers=None):
    spec = check_protocol(output, dataset, core_source_root=core_source_root)
    validate_import(output, source_root)
    core, _ = apply_runtime_overlay(core_source_root or Path(output) / "frozen_source")
    tasks = []
    for seed in SEEDS:
        for algorithm in ALGORITHMS:
            directory = Path(output) / f"training/BTCUSDT/{algorithm}/seed{seed}"
            if (directory / "record.json").exists():
                core.validate_training_record(output, "BTCUSDT", algorithm, seed, spec)
                continue
            if directory.exists():
                raise FileExistsError("Preserve an incomplete attempt; no silent model restart")
            tasks.append(dict(output=str(output), dataset=str(dataset), upstream=None, symbol="BTCUSDT", algorithm=algorithm, seed=seed,
                core_root=str(core.ROOT), protocol_sha256=sha(Path(output) / "protocol.json")))
    if tasks:
        core.run_parallel(output, tasks, train_one, "fit", max_workers)
    records = [core.validate_training_record(output, "BTCUSDT", a, seed, spec) for a in ALGORITHMS for seed in SEEDS]
    if len(records) != MODEL_COUNT:
        raise ValueError("Incomplete sixty-model population")
    core.write(Path(output) / "fit_complete.json", {"at": now(), "model_count": len(records), "records": records})
    validate_import(output, source_root)
    return training_counter_audit(output, core)


def _annual_pins(output, spec):
    return {"selection_sha256": sha(Path(output) / "validation_selection.json"),
        "validation_audit_sha256": sha(Path(output) / "validation_audit.json"), "protocol_sha256": sha(Path(output) / "protocol.json"),
        "dataset_sha256": spec["dataset_sha256"]}


def check_phase_audits(output, split):
    output = Path(output)
    financial = read(output / (split + "_audit.json"))
    contract = read(output / (split + "_seed_contract_audit.json"))
    manifest = read(output / "extension_manifest.json")
    if (financial.get("phase") != split or contract.get("phase") != split
            or financial.get("passed") is not True or financial.get("failure_count") != 0 or financial.get("failures")
            or financial.get("accounts_checked") != ACCOUNT_COUNT or financial.get("repeats_checked") != ACCOUNT_COUNT
            or financial.get("protocol_sha256") != sha(output / "protocol.json")
            or financial.get("summary_sha256") != sha(output / (split + "_summary.csv"))
            or contract.get("passed") is not True or contract.get("failure_count") != 0 or contract.get("failures")
            or contract.get("core_financial_audit_passed") is not True
            or contract.get("core_financial_audit_sha256") != sha(output / (split + "_audit.json"))
            or contract.get("wrapper_source_sha256") != manifest["extension_sources"][AUDIT_NAME]
            or contract.get("inputs_sha256", {}).get("protocol.json") != sha(output / "protocol.json")
            or contract.get("inputs_sha256", {}).get(split + "_audit.json") != sha(output / (split + "_audit.json"))):
        raise ValueError("Both unchanged financial arithmetic and seed-extension contract audits must pass")
    for name, expected in contract.get("inputs_sha256", {}).items():
        if sha(safe(output, name)) != expected:
            raise ValueError("Seed-extension audit inputs changed")
    return financial, contract


def evaluate(output, dataset, source_root, split, *, core_source_root=None, max_workers=None):
    import pandas as pd
    spec = check_protocol(output, dataset, core_source_root=core_source_root)
    validate_import(output, source_root)
    core, _ = apply_runtime_overlay(core_source_root or Path(output) / "frozen_source")
    from scripts.crypto_rl_reporting import canonical_csv
    validate_fit_completion(output, core)
    if split not in ("validation", "test"):
        raise ValueError("Only the fixed validation and test windows are permitted")
    if any((Path(output) / kind / split).exists() for kind in ("evaluation", "repeat")):
        raise FileExistsError("Preserve every existing evaluation attempt")
    if split == "test":
        selection = read(Path(output) / "validation_selection.json")
        audit, _ = check_phase_audits(output, "validation")
        if (selection.get("uses_test") is not False or selection.get("seed_selection") is not False
                or selection.get("algorithm_selection") is not False or selection.get("mode") != "fixed_algorithm_comparison"
                or selection.get("validation_audit_sha256") != sha(Path(output) / "validation_audit.json")
                or selection.get("validation_summary_sha256") != sha(Path(output) / "validation_summary.csv")
                or selection.get("validation_seed_contract_audit_sha256") != sha(Path(output) / "validation_seed_contract_audit.json")
                or selection.get("protocol_sha256") != sha(Path(output) / "protocol.json") or audit.get("passed") is not True):
            raise ValueError("Freeze the audited validation description before annual replay")
        write_new(Path(output) / "test_started.json", {"at": now(), "pins": _annual_pins(output, spec)})
    tasks = [dict(output=str(output), dataset=str(dataset), upstream=None, symbol="BTCUSDT", algorithm=a, seed=seed,
        scenario=scenario, split=split, core_root=str(core.ROOT), protocol_sha256=sha(Path(output) / "protocol.json"))
        for a in ALGORITHMS for seed in SEEDS for scenario in core.COSTS]
    tasks += [dict(output=str(output), dataset=str(dataset), upstream=None, symbol="BTCUSDT", algorithm=a, seed=None,
        scenario=scenario, split=split, core_root=str(core.ROOT), protocol_sha256=sha(Path(output) / "protocol.json"))
        for a in core.BASELINES for scenario in core.COSTS]
    if len(tasks) != ACCOUNT_COUNT:
        raise ValueError("All 128 account/cost identities are mandatory")
    results = core.run_parallel(output, tasks, evaluate_case, split, max_workers)
    summary = pd.DataFrame([result["summary"] for result in results]).sort_values("run_id")
    (Path(output) / (split + "_summary.csv")).write_bytes(canonical_csv(summary))
    core.write(Path(output) / (split + "_complete.json"), {"at": now(), "accounts": len(results), "repeats": len(results),
        "repeat_equal": True, "results": results})
    return {"accounts": len(results), "repeats": len(results)}


def describe_validation(output):
    import math
    import numpy as np
    import pandas as pd
    root = Path(output)
    if (root / "test_started.json").exists() or (root / "test_complete.json").exists():
        raise ValueError("Never change validation description after annual replay")
    audit, _ = check_phase_audits(root, "validation")
    spec = read(root / "protocol.json")
    if (audit.get("passed") is not True or audit.get("failure_count") != 0
            or audit.get("accounts_checked") != ACCOUNT_COUNT or audit.get("repeats_checked") != ACCOUNT_COUNT
            or audit.get("protocol_sha256") != sha(root / "protocol.json")
            or audit.get("summary_sha256") != sha(root / "validation_summary.csv")):
        raise ValueError("Complete exact validation financial audit must pass first")
    frame = pd.read_csv(root / "validation_summary.csv")
    expected = {("BTCUSDT", a, s, c) for a in ALGORITHMS for s in SEEDS for c in spec["costs"]}
    expected |= {("BTCUSDT", a, None, c) for a in spec["baselines"] for c in spec["costs"]}
    actual = {(r.symbol, r.algorithm, None if pd.isna(r.seed) else float(r.seed), r.scenario) for r in frame.itertuples()}
    if len(frame) != ACCOUNT_COUNT or actual != expected or not frame.initial_capital.eq(10000.).all() or set(frame.split) != {"validation"}:
        raise ValueError("All fixed validation identities are required once")
    candidates = []
    for algorithm in ALGORITHMS:
        group = frame.loc[frame.algorithm == algorithm]
        if len(group) != 60 or set(group.seed) != set(SEEDS):
            raise ValueError("Thirty seeds at both costs are required")
        scores = [float(np.mean([math.log1p(r.net_return) - .5 * abs(r.max_drawdown) - 2 * max(abs(r.max_drawdown) - .2, 0.)
            for r in group.loc[group.seed == seed].itertuples()])) for seed in SEEDS]
        candidates.append({"algorithm": algorithm, "score_median": float(np.median(scores)), "seed_scores": scores})
    # This field only keeps the original independent oracle's schema compatible.
    # Neither it nor the descriptive score changes the fixed two-algorithm test.
    result = {"at": now(), "uses_test": False, "seed_selection": False, "algorithm_selection": False,
        "mode": "fixed_algorithm_comparison", "fixed_algorithms": list(ALGORITHMS),
        "selection": {"BTCUSDT": {"algorithm": "sac", "seed": 0, "selection_role": "compatibility_anchor_only", "all_candidates": candidates}},
        "validation_audit_sha256": sha(root / "validation_audit.json"), "validation_summary_sha256": sha(root / "validation_summary.csv"),
        "validation_seed_contract_audit_sha256": sha(root / "validation_seed_contract_audit.json"),
        "protocol_sha256": sha(root / "protocol.json")}
    write_new(root / "validation_selection.json", result)
    return {"algorithms_compared": list(ALGORITHMS), "algorithm_selection": False, "seed_selection": False}


def pipeline(output, dataset, source_root, *, max_workers=None):
    output = Path(output).resolve()
    check_protocol(output, dataset)
    validate_import(output, source_root)
    core = core_module(output / "frozen_source")
    manifest = read(output / "extension_manifest.json")
    script = safe(output, manifest["driver_relative_path"])
    audit_script = safe(output, manifest["audit_entry_relative_path"])
    phases = ("fit", "validation", "audit-validation", "select", "test", "audit-test")
    markers = {"fit": "training_counter_audit.json", "validation": "validation_complete.json", "audit-validation": "validation_audit.json",
        "select": "validation_selection.json", "test": "test_complete.json", "audit-test": "test_audit.json"}
    for phase in phases:
        marker = output / markers[phase]
        if marker.exists():
            saved = read(marker)
            if (phase.startswith("audit-") or phase == "fit") and saved.get("passed") is not True:
                raise ValueError("Preserve a failed terminal audit; no silent recovery")
            if phase == "fit":
                validate_fit_completion(output, core)
            if phase.startswith("audit-"):
                check_phase_audits(output, phase.removeprefix("audit-"))
            continue
        if phase.startswith("audit-"):
            split = phase.removeprefix("audit-")
            if split == "test" and read(output / "test_started.json")["pins"] != _annual_pins(output, read(output / "protocol.json")):
                raise ValueError("Annual pre-start source pins changed")
            command = [sys.executable, "-B", str(audit_script), "--output", str(output), "--dataset", str(dataset),
                "--split", split, "--frozen-core-root", str(output / "frozen_source")]
        else:
            command = [sys.executable, "-B", str(script), phase, "--output", str(output), "--dataset", str(dataset), "--source-study", str(source_root)]
            if max_workers:
                command += ["--max-workers", str(max_workers)]
        core.write(output / "pipeline_status.json", {"at": now(), "phase": phase, "status": "running", "pid": os.getpid()})
        with (output / (phase + ".log")).open("x", encoding="utf-8") as stream:
            completed = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
        if completed.returncode:
            core.write(output / "pipeline_status.json", {"at": now(), "phase": phase, "status": "failed", "exit_code": completed.returncode})
            raise RuntimeError("Pipeline phase failed; all evidence preserved: " + phase)
        if not marker.exists():
            raise ValueError("Successful phase did not produce its terminal evidence")
        if phase.startswith("audit-"):
            check_phase_audits(output, phase.removeprefix("audit-"))
        check_protocol(output, dataset)
        validate_import(output, source_root)
    core.write(output / "pipeline_status.json", {"at": now(), "status": "completed", "no_production_change": True})
    return {"completed": True, "models": MODEL_COUNT, "accounts_per_phase": ACCOUNT_COUNT}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "validate-import", "fit", "validation", "select", "test", "pipeline"))
    parser.add_argument("--source-study", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--core-source-root", type=Path)
    parser.add_argument("--audit-script", type=Path)
    parser.add_argument("--max-workers", type=int)
    args = parser.parse_args()
    if args.phase == "prepare":
        result = prepare(args.source_study, args.output, args.dataset, core_source_root=args.core_source_root, audit_script=args.audit_script)
    elif args.phase == "validate-import":
        result = {"retained": validate_import(args.output, args.source_study)["retained_model_count"]}
    elif args.phase == "fit":
        result = fit(args.output, args.dataset, args.source_study, core_source_root=args.core_source_root, max_workers=args.max_workers)
        result = {key: result[key] for key in ("passed", "models_checked", "failure_count")}
    elif args.phase == "select":
        check_protocol(args.output, args.dataset, core_source_root=args.core_source_root)
        validate_import(args.output, args.source_study)
        result = describe_validation(args.output)
    elif args.phase == "pipeline":
        result = pipeline(args.output, args.dataset, args.source_study, max_workers=args.max_workers)
    else:
        result = evaluate(args.output, args.dataset, args.source_study, args.phase, core_source_root=args.core_source_root, max_workers=args.max_workers)
    print(json.dumps({"phase": args.phase, "at": now(), "result": result}))


if __name__ == "__main__":
    main()

"""Extend the immutable two-algorithm single-pair study without retraining it.

Twenty byte-identical SAC/TQC checkpoints remain attributed to their original
study. Forty new policies train on the same single-coin accounts; all sixty are
then evaluated together. No production credentials or exchange calls are used.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import importlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import sys

COMPATIBILITY_FIELDS = ("train", "validation", "test", "seeds", "primary_seed", "capital_per_account",
    "capital_shared_between_pairs", "ensemble", "costs", "environment", "sources", "runtime_packages",
    "validation_selection", "baselines", "research_gates_per_pair")
EXTRA_COMPATIBILITY = ("training_window_quality", "python", "no_production_change", "fresh_holdout",
                      "test_year_already_observed", "primary_seed_fixed_before_training")
RETAINED = ("sac", "tqc")
# Put slower independent work first without changing any algorithm's recipe.
SIX_ALGORITHMS = ("droq", "td3", "crossq", "recurrent_ppo", "sac", "tqc")
SYMBOLS = ("BTCUSDT", "ETHUSDT")
SEEDS = (0, 1, 2, 3, 4)


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def relative(root, name):
    name = str(name)
    parsed = PurePosixPath(name)
    if parsed.is_absolute() or ".." in parsed.parts or ":" in name or "\\" in name:
        raise ValueError("Evidence needs a relative POSIX path")
    root = Path(root).resolve()
    target = root.joinpath(*parsed.parts)
    target.resolve().relative_to(root)
    for item in (target, *target.parents):
        if item.is_symlink() or hasattr(item, "is_junction") and item.is_junction():
            raise ValueError("Evidence cannot cross a link")
        if item == root:
            break
    return target


def core_module(source_root=None):
    root = Path(source_root or Path(__file__).resolve().parents[1]).resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    core = importlib.import_module("scripts.backtest_single_pair_rl")
    if Path(core.__file__).resolve() != (root / "scripts/backtest_single_pair_rl.py").resolve():
        raise ValueError("Use a clean process for this exact frozen core")
    return core


def compatible(source, destination):
    if source.get("symbols") != list(SYMBOLS) or destination.get("symbols") != list(SYMBOLS):
        raise ValueError("The retained and new studies must have the exact same two independent trading pairs")
    if set(source.get("algorithms", [])) != set(RETAINED) or source.get("training_model_count") != 20:
        raise ValueError("The retained source must be the complete first twenty single-pair models")
    if set(destination.get("algorithms", [])) != set(SIX_ALGORITHMS) or destination.get("training_model_count") != 60:
        raise ValueError("Exactly six algorithms and sixty models are required")
    for key in (*COMPATIBILITY_FIELDS, *EXTRA_COMPATIBILITY, "dataset_sha256", "dataset_manifest_sha256"):
        if source.get(key) != destination.get(key):
            raise ValueError("Retained-model training/financial contract differs: " + key)
    for algorithm in RETAINED:
        for key in ("recipes", "timesteps_per_model"):
            if source[key][algorithm] != destination[key][algorithm]:
                raise ValueError("Retained algorithm recipe/budget differs: " + algorithm)
    if (destination["capital_per_account"] != 10000 or destination["capital_shared_between_pairs"] is not False
            or destination["ensemble"] is not False or destination["primary_seed"] != 0):
        raise ValueError("Independent single-pair capital and fixed primary seed are mandatory")


def model_files(directory):
    files = {}
    for path in sorted(Path(directory).rglob("*")):
        if path.is_symlink():
            raise ValueError("Checkpoint directory cannot contain links")
        if path.is_file():
            files[path.relative_to(directory).as_posix()] = sha(path)
    return files


def prepare_extension(source_root, output, dataset, *, upstream_root=None, core_source_root=None):
    core = core_module(core_source_root)
    source_root, output = Path(source_root).resolve(), Path(output).resolve()
    if output == source_root or output.is_relative_to(source_root) or source_root.is_relative_to(output):
        raise ValueError("The new study and retained evidence must have separate roots")
    source_spec = core.check_protocol(source_root, dataset)
    for phase in ("validation", "test"):
        audit = read(source_root / f"{phase}_audit.json")
        if (audit.get("passed") is not True or audit.get("failure_count") != 0
                or audit.get("protocol_sha256") != sha(source_root / "protocol.json")
                or audit.get("summary_sha256") != sha(source_root / f"{phase}_summary.csv")):
            raise ValueError("Retained models require their original completed audited study")
    records = {(s, a, seed): core.validate_training_record(source_root, s, a, seed, source_spec)
               for s in SYMBOLS for a in RETAINED for seed in SEEDS}
    destination = core.prepare(output, dataset, SIX_ALGORITHMS)
    compatible(source_spec, destination)
    retained = []
    for (symbol, algorithm, seed), record in records.items():
        directory = f"training/{symbol}/{algorithm}/seed{seed}"
        before = model_files(relative(source_root, directory))
        shutil.copytree(relative(source_root, directory), relative(output, directory))
        if model_files(relative(source_root, directory)) != before or model_files(relative(output, directory)) != before:
            raise ValueError("Retained model bytes changed during copying")
        retained.append({"symbol": symbol, "algorithm": algorithm, "seed": seed, "source_directory": directory,
            "destination_directory": directory, "file_sha256": before, "source_record_sha256": sha(relative(source_root, directory) / "record.json"),
            "model_parameter_sha256": record["checkpoint"]["parameter_sha256"], "source_completed_at": record["completed_at"]})
    imported_at = now()
    receipt = {"schema_version": 1, "source_study_root": str(source_root),
        "source_protocol_sha256": sha(source_root / "protocol.json"), "destination_protocol_sha256": sha(output / "protocol.json"),
        "source_fit_complete_sha256": sha(source_root / "fit_complete.json"),
        "source_validation_audit_sha256": sha(source_root / "validation_audit.json"), "source_test_audit_sha256": sha(source_root / "test_audit.json"),
        "dataset_sha256": source_spec["dataset_sha256"], "dataset_manifest_sha256": source_spec["dataset_manifest_sha256"],
        "compatibility_fields": list(COMPATIBILITY_FIELDS), "extra_compatibility_fields": list(EXTRA_COMPATIBILITY),
        "retained_model_count": 20, "new_model_count": 40, "retained_models": retained,
        "new_models": [{"symbol": s, "algorithm": a, "seed": seed} for s in SYMBOLS for a in SIX_ALGORITHMS if a not in RETAINED for seed in SEEDS],
        "imported_at": imported_at, "import_source_sha256": sha(__file__), "evaluation_results_reused": False,
        "old_test_year_already_observed": True, "no_production_change": True}
    write_new(output / "training_import.json", receipt)
    write_new(output / "training_import_receipt.json", {"training_import_sha256": sha(output / "training_import.json"), "imported_at": imported_at})
    script = output / "extension_source/scripts/extend_single_pair_rl.py"
    script.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(__file__, script)
    # Bundle the already-reviewed author source rather than changing an old path.
    droq = importlib.import_module("scripts.btceth_rl_droq")
    upstream = Path(upstream_root or droq.DEFAULT_UPSTREAM).resolve()
    upstream_manifest, upstream_files = droq.verify_upstream(upstream)
    frozen_upstream = output / "extension_source/upstream/DroQ"
    for name in ["verified_manifest.json", *upstream_files]:
        target = relative(frozen_upstream, name)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(relative(upstream, name), target)
    write_new(output / "extension_manifest.json", {"schema_version": 1, "frozen_at": now(),
        "extension_source": "extension_source/scripts/extend_single_pair_rl.py", "extension_source_sha256": sha(script),
        "training_import_sha256": sha(output / "training_import.json"), "core_protocol_sha256": sha(output / "protocol.json"),
        "upstream_root": "extension_source/upstream/DroQ", "upstream_manifest_sha256": sha(frozen_upstream / "verified_manifest.json"),
        "upstream_commit": upstream_manifest["commit"], "upstream_files_sha256": upstream_files,
        "performance_mode": "native; no GPU replay cache or Polyak graph installed", "new_training_models": 40,
        "retained_training_models": 20, "annual_accounts": 136, "validation_accounts": 136, "repeat_every_account": True})
    validate_import(output, source_root)
    return {"retained": 20, "pending": 40, "accounts_per_phase": 136}


def validate_import(output, source_root):
    output, source_root = Path(output).resolve(), Path(source_root).resolve()
    old, new = read(source_root / "protocol.json"), read(output / "protocol.json")
    compatible(old, new)
    imported = read(output / "training_import.json")
    if (read(output / "training_import_receipt.json")["training_import_sha256"] != sha(output / "training_import.json")
            or imported.get("schema_version") != 1 or imported.get("compatibility_fields") != list(COMPATIBILITY_FIELDS)
            or imported.get("extra_compatibility_fields") != list(EXTRA_COMPATIBILITY)
            or imported.get("retained_model_count") != 20 or imported.get("new_model_count") != 40
            or imported.get("import_source_sha256") != sha(__file__)
            or imported.get("dataset_sha256") != new["dataset_sha256"]
            or imported.get("dataset_manifest_sha256") != new["dataset_manifest_sha256"]
            or imported.get("evaluation_results_reused") is not False or imported.get("old_test_year_already_observed") is not True):
        raise ValueError("Complete frozen import receipt required")
    for field, root, name in (("source_protocol_sha256", source_root, "protocol.json"), ("destination_protocol_sha256", output, "protocol.json"),
        ("source_fit_complete_sha256", source_root, "fit_complete.json"), ("source_validation_audit_sha256", source_root, "validation_audit.json"),
        ("source_test_audit_sha256", source_root, "test_audit.json")):
        if imported.get(field) != sha(root / name):
            raise ValueError("Retained-study source binding changed: " + field)
    expected = {(s, a, seed) for s in SYMBOLS for a in RETAINED for seed in SEEDS}
    identities = [(item["symbol"], item["algorithm"], item["seed"]) for item in imported["retained_models"]]
    pending = [(item["symbol"], item["algorithm"], item["seed"]) for item in imported["new_models"]]
    wanted_pending = {(s, a, seed) for s in SYMBOLS for a in SIX_ALGORITHMS if a not in RETAINED for seed in SEEDS}
    if len(identities) != 20 or set(identities) != expected or len(pending) != 40 or set(pending) != wanted_pending:
        raise ValueError("Every retained and new model identity is required exactly once")
    for item in imported["retained_models"]:
        wanted = f"training/{item['symbol']}/{item['algorithm']}/seed{item['seed']}"
        if item["source_directory"] != wanted or item["destination_directory"] != wanted:
            raise ValueError("A copied checkpoint cannot be relabelled")
        original = relative(source_root, wanted)
        copy = relative(output, wanted)
        if model_files(original) != item["file_sha256"] or model_files(copy) != item["file_sha256"]:
            raise ValueError("Retained checkpoint/record/progress bytes changed")
        record = read(original / "record.json")
        if (sha(original / "record.json") != item["source_record_sha256"]
                or record["completed_at"] != item["source_completed_at"]
                or record["checkpoint"]["parameter_sha256"] != item["model_parameter_sha256"]):
            raise ValueError("Retained model identity/time/tensors changed")
    return imported


def training_counter_audit(output, core):
    spec = read(Path(output) / "protocol.json")
    rows, failures = [], []
    for symbol in SYMBOLS:
        for algorithm in spec["algorithms"]:
            for seed in SEEDS:
                try:
                    record = core.validate_training_record(output, symbol, algorithm, seed, spec)
                    train = record["training"]; counters = train["counters"]; steps = spec["timesteps_per_model"][algorithm]
                    if algorithm in ("sac", "tqc", "crossq", "td3"):
                        updates = steps - 512
                        delay = 3 if algorithm == "crossq" else 2 if algorithm == "td3" else 1
                        optimizers = {"actor": updates // delay, "critic": updates}
                        if algorithm != "td3": optimizers["entropy"] = updates // delay
                        if (counters.get("gradient_update_rounds") != updates or counters.get("optimizer_steps") != optimizers
                                or counters.get("replay_transitions") != steps or counters.get("replay_add_calls") != steps):
                            raise ValueError("Native delayed actor/critic/entropy/replay counts differ")
                    elif algorithm == "recurrent_ppo":
                        updates = steps // 128 * 5
                        if counters.get("gradient_update_rounds") != updates or counters.get("optimizer_steps") != {"policy": updates}:
                            raise ValueError("Native recurrent PPO rollout/epoch optimizer count differs")
                    else:
                        calls = steps - 5000; rounds = calls * 20
                        expected = {"q_inner_rounds": rounds, "q_optimizer_steps": [rounds, rounds], "total_q_optimizer_steps": 2 * rounds,
                            "policy_optimizer_steps": calls, "alpha_optimizer_steps": calls, "target_updates_per_network": [rounds, rounds],
                            "learning_train_calls": calls, "replay_adds": steps, "random_exploration_actions": 5001}
                        if any(counters.get(key) != value for key, value in expected.items()):
                            raise ValueError("Native DroQ Q/target/actor/alpha/replay budget differs")
                        effective = train["effective_parameters"]
                        if any(effective.get(key) != value for key, value in {"obs_dim": 13, "act_dim": 1, "target_entropy": -1.,
                                "utd_ratio": 20, "num_Q": 2, "num_min": 2, "policy_update_delay": 20}.items()):
                            raise ValueError("Native one-action DroQ recipe differs")
                    history = train["training_episode_history"]
                    completed = [episode for episode in history if episode["completed_steps"] > 0]
                    if (len(completed) != steps // 32 or sum(episode["completed_steps"] for episode in completed) != steps
                            or any(episode["completed_steps"] != 32 or episode["terminated"] is not False or episode["truncated"] is not True
                                   or not core.TRAIN[0] <= episode["start"].replace(" ", "T").replace("+00:00", "Z") < core.TRAIN[1]
                                   for episode in completed)):
                        raise ValueError("Every actual 32-day training fragment must remain a timeout")
                    for episode in completed:
                        start = datetime.fromisoformat(episode["start"])
                        end = datetime.fromisoformat(episode["end_exclusive"])
                        until = datetime.fromisoformat(episode["completed_until"])
                        if end != start + timedelta(days=32) or until != end or end > datetime.fromisoformat(core.TRAIN[1]):
                            raise ValueError("Actual completed training fragments cannot cross the training boundary")
                    rows.append({"symbol": symbol, "algorithm": algorithm, "seed": seed, "counters": counters,
                        "record_sha256": sha(Path(output) / f"training/{symbol}/{algorithm}/seed{seed}/record.json"),
                        "retained": algorithm in RETAINED})
                except Exception as exc:
                    failures.append({"symbol": symbol, "algorithm": algorithm, "seed": seed, "error": str(exc)})
    result = {"audited_at": now(), "passed": not failures and len(rows) == 60, "models_checked": len(rows),
        "retained_models": 20, "new_models": 40, "failure_count": len(failures), "failures": failures, "models": rows,
        "extension_source_sha256": sha(__file__), "core_protocol_sha256": sha(Path(output) / "protocol.json")}
    core.write(Path(output) / "training_counter_audit.json", result)
    if not result["passed"]:
        raise ValueError("Algorithm-specific training counter audit failed")
    return result


def pipeline(output, source_root, dataset, *, max_workers=None):
    output = Path(output).resolve()
    core = core_module(output / "frozen_source")
    manifest = read(output / "extension_manifest.json")
    if sha(__file__) != manifest["extension_source_sha256"] or sha(output / manifest["extension_source"]) != manifest["extension_source_sha256"]:
        raise ValueError("Run the exact frozen extension entry point")
    if (manifest["training_import_sha256"] != sha(output / "training_import.json")
            or manifest["core_protocol_sha256"] != sha(output / "protocol.json")
            or manifest.get("new_training_models") != 40 or manifest.get("retained_training_models") != 20
            or manifest.get("annual_accounts") != 136 or manifest.get("validation_accounts") != 136
            or manifest.get("repeat_every_account") is not True):
        raise ValueError("Frozen extension must bind this exact import, protocol, and case population")
    core.check_protocol(output, dataset)
    validate_import(output, source_root)
    upstream = relative(output, manifest["upstream_root"])
    if sha(upstream / "verified_manifest.json") != manifest["upstream_manifest_sha256"]:
        raise ValueError("Frozen author source manifest changed")
    for name, expected in manifest["upstream_files_sha256"].items():
        if sha(relative(upstream, name)) != expected: raise ValueError("Frozen author source changed")
    try:
        core.write(output / "pipeline_status.json", {"at": now(), "phase": "fit", "status": "running", "pid": os.getpid(), "retained": 20, "new": 40})
        core.fit(output, dataset, str(upstream), max_workers)
        validate_import(output, source_root)
        training_counter_audit(output, core)
        # The unchanged core skips its now-complete fit, then runs all 136
        # validation accounts, independent audit, selection, annual and audit.
        return core.pipeline(output, dataset, str(upstream), max_workers)
    except Exception as exc:
        core.write(output / "pipeline_status.json", {"at": now(), "status": "failed", "error_type": type(exc).__name__, "error": str(exc)})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "validate-import", "pipeline"))
    parser.add_argument("--source-study", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dataset")
    parser.add_argument("--upstream")
    parser.add_argument("--core-source-root")
    parser.add_argument("--max-workers", type=int)
    args = parser.parse_args()
    if args.phase == "prepare":
        result = prepare_extension(args.source_study, args.output, args.dataset, upstream_root=args.upstream, core_source_root=args.core_source_root)
    elif args.phase == "validate-import": result = {"retained": validate_import(args.output, args.source_study)["retained_model_count"]}
    else: result = pipeline(args.output, args.source_study, args.dataset, max_workers=args.max_workers)
    print(json.dumps({"phase": args.phase, "at": now(), "result": result}))


if __name__ == "__main__":
    main()

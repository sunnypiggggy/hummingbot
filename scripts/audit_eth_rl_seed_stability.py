"""Independent seed-extension contract checks around the frozen financial oracle.

No environment, ledger, runner, model or inference imports. The original oracle
does the same closed-form hourly arithmetic; only the declared account universe
is extended. Exact ETH feature names preserve the cross-asset exclusion check.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import json
from pathlib import Path, PurePosixPath
import sys

SYMBOLS = ("ETHUSDT",)
ALGORITHMS = ("sac", "td3")
SEEDS = tuple(range(30))
BASELINES = ("cash", "buy_hold", "sma200_trend", "momentum180")
COSTS = {"base": {"fee": .001, "slippage": .0002}, "stress": {"fee": .002, "slippage": .0005}}
SPLITS = {"train": ["2020-01-01T00:00:00Z", "2025-01-01T00:00:00Z"],
          "validation": ["2025-01-01T00:00:00Z", "2025-10-23T00:00:00Z"],
          "test": ["2025-10-23T00:00:00Z", "2026-10-23T00:00:00Z"]}
FEATURES = ["ETHUSDT:close23utc_return_1d/.1", "ETHUSDT:close23utc_return_7d/.3",
            "ETHUSDT:close23utc_return_30d/.5", "ETHUSDT:close23utc_return_90d/1",
            "ETHUSDT:close23utc_return_180d/2", "ETHUSDT:close23utc_sma20_deviation/.2",
            "ETHUSDT:close23utc_sma60_deviation/.35", "ETHUSDT:close23utc_sma200_deviation/.5",
            "ETHUSDT:close23utc_volatility30/.1", "ETHUSDT:volume24h_relative30_minus1/2",
            "coin_weight", "cash_weight", "current_drawdown"]
FINANCIAL_FIELDS = ("train", "validation", "test", "primary_seed", "capital_per_account",
    "capital_shared_between_pairs", "ensemble", "costs", "environment", "sources", "runtime_packages",
    "baselines", "research_gates_per_pair", "python", "no_production_change", "fresh_holdout",
    "test_year_already_observed", "primary_seed_fixed_before_training", "dataset_sha256", "dataset_manifest_sha256")
RUNTIME_OVERLAY = {"core_symbols": list(SYMBOLS), "core_seeds": list(SEEDS), "model_seeds": list(SEEDS),
    "oracle_symbols": list(SYMBOLS), "oracle_seeds": list(SEEDS),
    "worker_apply_policy": "each spawned training and evaluation worker applies this exact whitelist",
    "financial_math_changed": False, "model_recipes_changed": False, "performance_mode": "native; no performance patches"}
SEED_EXTENSION = {"previous_seeds": list(range(5)), "new_seeds": list(range(5, 30)), "full_seeds": list(SEEDS),
    "retained_model_count": 10, "new_model_count": 50, "primary_analysis_cohort": "new25", "no_seed_selection": True}


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


def unlinked_root(path):
    path = Path(path).absolute()
    if any(p.is_symlink() or hasattr(p, "is_junction") and p.is_junction() for p in (path, *path.parents)):
        raise ValueError("Evidence root cannot cross a link")
    return path.resolve()


def safe(root, name):
    name = str(name)
    part = PurePosixPath(name)
    if not name or part.as_posix() != name or part.is_absolute() or ".." in part.parts or ":" in name or "\\" in name:
        raise ValueError("Only canonical root-relative evidence paths are permitted")
    original = Path(root).absolute()
    target = original.joinpath(*part.parts)
    if any(p.is_symlink() or hasattr(p, "is_junction") and p.is_junction() for p in (target, *target.parents)):
        raise ValueError("Evidence cannot cross a link")
    target.resolve().relative_to(original.resolve())
    if not target.is_file():
        raise FileNotFoundError(name)
    return target


class Checks:
    def __init__(self):
        self.count, self.failures = 0, []

    def check(self, condition, label):
        self.count += 1
        if not condition:
            self.failures.append(label)


def contract_checks(output, dataset, split, *, core_root=None):
    output = unlinked_root(output)
    core_root = unlinked_root(core_root or output / "frozen_source")
    checks, pins = Checks(), {}

    def take(name):
        path = safe(output, name)
        pins[name] = sha(path)
        return read(path)

    spec = take("protocol.json")
    receipt = take("protocol_receipt.json")
    checks.check(receipt.get("protocol_sha256") == pins["protocol.json"], "frozen protocol receipt")
    expected = {"symbols": list(SYMBOLS), "seeds": list(SEEDS), "primary_seed": 0,
                "primary_seed_fixed_before_training": True, "training_model_count": 60,
                "capital_per_account": 10000., "capital_shared_between_pairs": False, "ensemble": False,
                "costs": COSTS, "baselines": list(BASELINES), "test_year_already_observed": True,
                "fresh_holdout": False, "no_production_change": True,
                "timesteps_per_model": {"sac": 32768, "td3": 65536}}
    for key, value in expected.items():
        checks.check(spec.get(key) == value, "unchanged financial contract:" + key)
    checks.check(set(spec.get("algorithms", [])) == set(ALGORITHMS) and len(spec["algorithms"]) == 2,
                 "only SAC and TD3 compared")
    for key, value in SPLITS.items():
        checks.check(spec.get(key) == value, "unchanged date window:" + key)
    for key, value in {"observation_dim": 13, "action_dim": 1, "cash_floor": .05, "max_coin_target": .95,
                       "rebalance_band": .025, "drawdown_penalty": .25, "episode_days": 32}.items():
        checks.check(spec.get("environment", {}).get(key) == value, "unchanged environment:" + key)
    checks.check(spec.get("dataset_sha256") == sha(dataset), "same original dataset")
    checks.check(spec.get("dataset_manifest_sha256") == sha(Path(dataset).parent / "manifest.json"), "same dataset manifest")
    checks.check(spec.get("runtime_overlay") == RUNTIME_OVERLAY and spec.get("seed_extension") == SEED_EXTENSION,
                 "only predeclared seed and symbol whitelists change")
    retained = take("retained_protocol.json")
    imported = take("training_import.json")
    imported_receipt = take("training_import_receipt.json")
    checks.check(imported_receipt.get("training_import_sha256") == pins["training_import.json"], "immutable import receipt")
    checks.check(imported.get("source_protocol_sha256") == pins["retained_protocol.json"]
                 and imported.get("destination_protocol_sha256") == pins["protocol.json"]
                 and imported.get("compatibility_fields") == list(FINANCIAL_FIELDS), "original and expanded study provenance")
    checks.check(retained.get("symbols") == ["BTCUSDT", "ETHUSDT"] and retained.get("seeds") == list(range(5))
                 and retained.get("training_model_count") == 60
                 and set(retained.get("algorithms", [])) == {"sac", "tqc", "crossq", "recurrent_ppo", "td3", "droq"},
                 "reference is the audited original six-algorithm study")
    for key in FINANCIAL_FIELDS:
        checks.check(spec.get(key) == retained.get(key), "exact retained financial/data/runtime contract:" + key)
    checks.check(spec.get("training_window_quality") == {"ETHUSDT": retained["training_window_quality"]["ETHUSDT"]},
                 "same eligible ETH training fragments")
    checks.check(spec.get("recipes") == {a: retained["recipes"][a] for a in ALGORITHMS}, "unchanged native SAC and TD3 recipes")
    checks.check(imported.get("retained_model_count") == 10 and imported.get("new_model_count") == 50
                 and imported.get("evaluation_results_reused") is False and imported.get("old_test_year_already_observed") is True
                 and imported.get("dataset_sha256") == spec["dataset_sha256"]
                 and imported.get("dataset_manifest_sha256") == spec["dataset_manifest_sha256"], "ten preserved and fifty new models")
    old_models = imported.get("retained_models", [])
    checks.check(len(old_models) == 10 and {(m["symbol"], m["algorithm"], m["seed"]) for m in old_models}
                 == {("ETHUSDT", a, s) for a in ALGORITHMS for s in range(5)}, "full unchanged original model population")
    for model in old_models:
        directory = f"training/ETHUSDT/{model['algorithm']}/seed{model['seed']}"
        checks.check(type(model["seed"]) is int and model.get("destination_directory") == directory
                     and model.get("source_directory") == directory and bool(model.get("file_sha256")), "retained directory identity")
        for name, expected_sha in model.get("file_sha256", {}).items():
            relative = directory + "/" + name
            pins[relative] = sha(safe(output, relative))
            checks.check(pins[relative] == expected_sha, "byte-identical retained model:" + relative)
        checks.check(model.get("source_record_sha256") == pins.get(directory + "/record.json"), "original record identity")
    new_models = imported.get("new_models", [])
    checks.check(len(new_models) == 50 and all(type(m.get("seed")) is int for m in new_models)
                 and {(m["symbol"], m["algorithm"], m["seed"]) for m in new_models}
                 == {("ETHUSDT", a, s) for a in ALGORITHMS for s in range(5, 30)}, "all fifty predeclared new seeds retained")
    extension = take("extension_manifest.json")
    extensions = extension.get("extension_sources", {})
    names = {"scripts/backtest_eth_rl_seed_stability.py", "scripts/audit_eth_rl_seed_stability.py"}
    checks.check(set(extensions) == names, "complete two-file seed extension source inventory")
    checks.check(extension.get("runtime_overlay") == RUNTIME_OVERLAY
                 and extension.get("core_protocol_sha256") == pins["protocol.json"]
                 and extension.get("training_import_sha256") == pins["training_import.json"]
                 and extension.get("models") == 60 and extension.get("retained_models") == 10
                 and extension.get("fresh_models") == 50 and extension.get("accounts_per_phase") == 128
                 and extension.get("repeat_all_cases") is True and extension.get("both_costs_each_model") is True,
                 "frozen extension manifest matches complete comparison")
    for name, expected_sha in extensions.items():
        relative = "extension_source/" + name
        pins[relative] = sha(safe(output, relative))
        checks.check(pins[relative] == expected_sha, "frozen extension source:" + name)
    checks.check(extensions.get("scripts/audit_eth_rl_seed_stability.py") == sha(__file__), "executed independent wrapper hash")
    for name, expected_sha in spec.get("sources", {}).items():
        relative = "frozen_source/" + name
        pins[relative] = sha(safe(output, relative))
        checks.check(pins[relative] == expected_sha and sha(safe(core_root, name)) == expected_sha, "unchanged executed oracle/core source:" + name)
    checks.check(len(spec.get("sources", {})) == 11 and "scripts/audit_single_pair_rl.py" in spec["sources"], "original core source population")
    counter = take("training_counter_audit.json")
    rows = counter.get("models", [])
    checks.check(counter.get("passed") is True and counter.get("failure_count") == 0 and not counter.get("failures")
                 and counter.get("models_checked") == 60 and counter.get("core_protocol_sha256") == pins["protocol.json"]
                 and counter.get("extension_source_sha256") == extensions.get("scripts/backtest_eth_rl_seed_stability.py"),
                 "complete native training audit binds original runtime and extension")
    checks.check(len(rows) == 60 and {(m["symbol"], m["algorithm"], m["seed"]) for m in rows}
                 == {("ETHUSDT", a, s) for a in ALGORITHMS for s in SEEDS}, "all sixty actual training records audited")
    for model in rows:
        name = f"training/ETHUSDT/{model['algorithm']}/seed{model['seed']}/record.json"
        record = take(name)
        steps = spec["timesteps_per_model"][model["algorithm"]]
        updates = steps - 512
        optimizers = {"critic": updates, "actor": updates if model["algorithm"] == "sac" else updates // 2}
        if model["algorithm"] == "sac":
            optimizers["entropy"] = updates
        counters = record["training"]["counters"]
        checks.check(model.get("record_sha256") == pins[name] and model.get("counters") == counters
                     and record["training"]["actual_steps"] == steps
                     and counters.get("environment_steps") == steps and counters.get("gradient_update_rounds") == updates
                     and counters.get("optimizer_steps") == optimizers and counters.get("replay_add_calls") == steps
                     and counters.get("replay_transitions") == steps, "independently recalculated native update budget:" + name)
    if split == "test":
        selection = take("validation_selection.json")
        started = take("test_started.json")
        validation_audit = take("validation_audit.json")
        checks.check(selection.get("uses_test") is False and selection.get("seed_selection") is False
                     and selection.get("algorithm_selection") is False
                     and set(selection.get("fixed_algorithms", [])) == set(ALGORITHMS)
                     and selection.get("mode") == "fixed_algorithm_comparison", "comparison never selects a seed or test-year winner")
        checks.check(validation_audit.get("passed") is True and validation_audit.get("failure_count") == 0,
                     "annual replay follows a passed validation oracle")
        checks.check(selection.get("validation_audit_sha256") == pins["validation_audit.json"]
                     and selection.get("protocol_sha256") == pins["protocol.json"]
                     and selection.get("validation_summary_sha256") == sha(safe(output, "validation_summary.csv")),
                     "validation comparison binds exact audited evidence")
        checks.check(started.get("pins") == {"selection_sha256": pins["validation_selection.json"],
                     "validation_audit_sha256": pins["validation_audit.json"], "protocol_sha256": pins["protocol.json"],
                     "dataset_sha256": spec["dataset_sha256"]}, "annual start pins fixed comparison and sources")
        checks.check(datetime.fromisoformat(started["at"]) >= datetime.fromisoformat(selection["at"])
                     >= datetime.fromisoformat(validation_audit["audited_at"]), "validation before fixed comparison before annual start")
    return spec, checks, pins, core_root


def metadata_checks(output, split, checks, pins):
    expected = {(a, seed, cost) for a in ALGORITHMS for seed in SEEDS for cost in COSTS}
    expected.update((a, None, cost) for a in BASELINES for cost in COSTS)
    for kind in ("evaluation", "repeat"):
        paths = set()
        for algorithm, seed, cost in expected:
            name = f"{kind}/{split}/ETHUSDT/{algorithm}/{'baseline' if seed is None else 'seed'+str(seed)}/{cost}/metadata.json"
            paths.add(name)
            path = safe(output, name)
            pins[name] = sha(path)
            meta = read(path)
            checks.check(meta.get("observation_features") == FEATURES, "exact ETH-only feature dependencies:" + name)
            checks.check(meta.get("pair") == "ETHUSDT" and meta.get("tradable_symbols") == ["ETHUSDT"]
                         and set(meta.get("final_positions", {})) <= {"ETHUSDT"}, "ETH-only inventory:" + name)
            checks.check(meta.get("no_ensemble") is True and meta.get("independent_capital") == 10000.
                         and meta.get("observation_dim") == 13 and meta.get("action_dim") == 1,
                         "single-account dimensions and ownership:" + name)
        observed = {p.relative_to(output).as_posix() for p in (output / kind / split).rglob("metadata.json")}
        checks.check(observed == paths and len(paths) == 128, "exact full account population:" + kind)


def audit(output, dataset, split, *, core_root=None, workers=None):
    if split not in ("validation", "test"):
        raise ValueError("Only the fixed validation and annual windows can be audited")
    output = unlinked_root(output)
    for name in (split + "_audit.json", split + "_seed_contract_audit.json"):
        target = output / name
        if target.exists() or target.is_symlink() or hasattr(target, "is_junction") and target.is_junction():
            raise FileExistsError("Preserve the existing audit; use a separately recorded attempt: " + name)
    checks, pins, financial = Checks(), {}, None
    started = now()
    try:
        spec, checks, pins, core_root = contract_checks(output, dataset, split, core_root=core_root)
        metadata_checks(output, split, checks, pins)
        if checks.failures:
            raise ValueError("Seed-extension contract or ETH feature checks failed")
        if str(core_root) not in sys.path:
            sys.path.insert(0, str(core_root))
        oracle = importlib.import_module("scripts.audit_single_pair_rl")
        if Path(oracle.__file__).resolve() != safe(core_root, "scripts/audit_single_pair_rl.py").resolve():
            raise ValueError("Use a clean audit process for the exact frozen arithmetic oracle")
        oracle.SYMBOLS, oracle.SEEDS = SYMBOLS, SEEDS
        financial = oracle.audit_phase(output, dataset, split, workers=workers)
        checks.check(financial.get("passed") is True and financial.get("failure_count") == 0
                     and financial.get("accounts_checked") == 128 and financial.get("repeats_checked") == 128,
                     "unchanged independent hourly financial arithmetic passed all128 accounts and repeats")
        pins[split + "_audit.json"] = sha(safe(output, split + "_audit.json"))
        for name, expected_sha in pins.items():
            checks.check(sha(safe(output, name)) == expected_sha, "extension inputs unchanged through audit:" + name)
    except Exception as exc:
        checks.check(False, type(exc).__name__ + ": " + str(exc))
    result = {"phase": split, "started_at": started, "audited_at": now(), "passed": not checks.failures,
              "checks_total": checks.count, "failure_count": len(checks.failures), "failures": checks.failures,
              "core_financial_audit_passed": financial is not None and financial.get("passed") is True,
              "core_financial_audit_sha256": pins.get(split + "_audit.json"), "inputs_sha256": pins,
              "wrapper_source_sha256": sha(__file__), "runtime_overlay": {"oracle.SYMBOLS": list(SYMBOLS), "oracle.SEEDS": list(SEEDS)},
              "independent_math_unchanged": True, "no_environment_or_model_imports": True}
    history = output / "audit_history"
    history.mkdir(parents=True, exist_ok=True)
    body = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    (history / f"{split}_seed_contract_audit_{stamp}.json").write_text(body, encoding="utf-8")
    target = output / (split + "_seed_contract_audit.json")
    with target.open("x", encoding="utf-8") as stream:
        stream.write(body)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--split", required=True, choices=("validation", "test"))
    parser.add_argument("--frozen-core-root", type=Path)
    parser.add_argument("--workers", type=int)
    args = parser.parse_args()
    result = audit(args.output, args.dataset, args.split, core_root=args.frozen_core_root, workers=args.workers)
    print(json.dumps({k: result[k] for k in ("phase", "passed", "checks_total", "failure_count", "core_financial_audit_passed")}))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()

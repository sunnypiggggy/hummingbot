"""Audit-gated six-algorithm report for independent BTC and ETH accounts.

No pickle loading, policy construction, inference, training or publication. The
old portfolio report is imported only for its official Data SDK transport.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import backtest_single_pair_rl as runner
from scripts.crypto_rl_reporting import clean_json, summarize
from scripts.btceth_rl_report_app import DataRuntime, host_path

ALGORITHMS = ("sac", "tqc", "crossq", "recurrent_ppo", "td3", "droq")
MODEL_COUNT = 60
ACCOUNT_COUNT = 136
RETAINED_ALGORITHMS = ("sac", "tqc")
IMPORT_COMPATIBILITY_FIELDS = ("train", "validation", "test", "seeds", "primary_seed", "capital_per_account",
    "capital_shared_between_pairs", "ensemble", "costs", "environment", "sources", "runtime_packages",
    "validation_selection", "baselines", "research_gates_per_pair")
EXTRA_IMPORT_COMPATIBILITY_FIELDS = ("training_window_quality", "python", "no_production_change", "fresh_holdout",
    "test_year_already_observed", "primary_seed_fixed_before_training")
RESULT_FILES = ("equity_1h.csv.gz", "fills.csv", "weights.csv", "boundary_equity.csv", "metadata.json",
                "fingerprints.json", "actions.json", "summary.json", "daily_equity.csv", "monthly_returns.csv")
QUERY_IDS = ("account_summary", "equity_history", "monthly_returns", "seed_statistics", "validation_summary",
             "validation_selection", "training_records", "training_import", "research_targets", "audit_evidence", "protocol")
SELECTION_SCORE_ABS_TOL = 1e-14


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean_json(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def relative(root, name):
    name = str(name).replace("\\", "/")
    require(name and ":" not in name and not name.startswith("/") and ".." not in name.split("/"), "Unsafe evidence path")
    root = Path(root).resolve()
    path = (root / name).resolve()
    path.relative_to(root)
    require(path.is_file() and not path.is_symlink(), "Missing or unsafe evidence file: " + name)
    return path


def csv(path, *, indexed=False):
    result = pd.read_csv(path, float_precision="round_trip")
    if indexed:
        result["timestamp"] = pd.to_datetime(result.timestamp, utc=True)
        result = result.set_index("timestamp")
    return result


def equivalent(actual, expected, label):
    """CSV null/float decoding may differ, financial values and identities may not."""
    require(set(actual) == set(expected), "Changed fields: " + label)
    for key, value in expected.items():
        other = actual[key]
        if value is None:
            require(other is None or pd.isna(other), label + ": missing/null " + key)
        elif isinstance(value, (bool, str)):
            require(other == value, label + ": identity " + key)
        elif isinstance(value, (int, float)):
            require(isinstance(other, (int, float, np.number)) and math.isfinite(float(other))
                    and math.isclose(float(other), float(value), rel_tol=1e-10, abs_tol=1e-7), label + ": value " + key)
        else:
            require(other == value, label + ": field " + key)


def frame_equivalent(actual, expected, label):
    require(set(actual.columns) == set(expected.columns) and len(actual) == len(expected), "Changed CSV shape: " + label)
    for a, e in zip(actual[expected.columns].to_dict("records"), expected.to_dict("records")):
        equivalent(a, e, label)


def verify_selection_choice(choice, scored):
    """Permit only tiny platform math differences; preserve frozen identities."""
    require(set(choice) == {"algorithm", "seed", "all_candidates"}
            and choice["algorithm"] == scored[0]["algorithm"] and type(choice["seed"]) is int and choice["seed"] == 0,
            "Validation selected algorithm or fixed primary seed changed")
    actual = choice["all_candidates"]
    require(isinstance(actual, list) and len(actual) == len(scored), "Validation candidate population changed")
    maximum_error = 0.
    for frozen, local in zip(actual, scored):
        require(set(frozen) == {"algorithm", "score_median", "seed_scores"} and frozen["algorithm"] == local["algorithm"],
                "Validation candidate identity or order changed")
        require(isinstance(frozen["seed_scores"], list) and len(frozen["seed_scores"]) == len(runner.SEEDS), "Validation seed score population changed")
        for observed, expected in zip([frozen["score_median"], *frozen["seed_scores"]], [local["score_median"], *local["seed_scores"]]):
            require(type(observed) in (int, float) and math.isfinite(observed)
                    and math.isclose(observed, expected, rel_tol=0., abs_tol=SELECTION_SCORE_ABS_TOL),
                    "Validation selection score differs beyond platform math tolerance")
            maximum_error = max(maximum_error, abs(observed - expected))
    return maximum_error


def verify_training_import(root, retained_root, spec):
    """Verify the original 20 complete model directories, without model loading."""
    root, retained_root = Path(root).resolve(), Path(retained_root).resolve()
    require(root != retained_root, "A separate original retained study is required")
    receipt = read(relative(root, "training_import.json"))
    outer = read(relative(root, "training_import_receipt.json"))
    require(outer.get("training_import_sha256") == sha(relative(root, "training_import.json"))
            and outer.get("imported_at") == receipt.get("imported_at"), "Training import receipt changed")
    require(receipt.get("schema_version") == 1 and receipt.get("retained_model_count") == 20 and receipt.get("new_model_count") == 40,
            "Exactly 20 retained and 40 new model identities are required")
    external = {}
    def original(name):
        path = relative(retained_root, name)
        external[str(path)] = sha(path)
        return path
    source_spec = read(original("protocol.json"))
    require(receipt.get("source_protocol_sha256") == external[str(retained_root / "protocol.json")]
            and receipt.get("destination_protocol_sha256") == sha(relative(root, "protocol.json"))
            and read(original("protocol_receipt.json")).get("protocol_sha256") == receipt["source_protocol_sha256"], "Import protocol binding changed")
    require(source_spec.get("training_model_count") == 20 and set(source_spec.get("algorithms", [])) == set(RETAINED_ALGORITHMS)
            and len(source_spec.get("algorithms", [])) == 2, "Original retained study must be the two-algorithm 20-model study")
    fields = receipt.get("compatibility_fields", [])
    require(isinstance(fields, list) and len(fields) == len(IMPORT_COMPATIBILITY_FIELDS) and set(fields) == set(IMPORT_COMPATIBILITY_FIELDS),
            "Import compatibility field contract changed")
    require(receipt.get("extra_compatibility_fields") == list(EXTRA_IMPORT_COMPATIBILITY_FIELDS)
            and receipt.get("evaluation_results_reused") is False and receipt.get("old_test_year_already_observed") is True
            and receipt.get("no_production_change") is True, "Retained training and fresh evaluation declarations changed")
    require(all(source_spec.get(key) == spec.get(key) for key in (*IMPORT_COMPATIBILITY_FIELDS, *EXTRA_IMPORT_COMPATIBILITY_FIELDS)), "Retained models have different data, source, funding, dates or accounting rules")
    for key in ("dataset_sha256", "dataset_manifest_sha256"):
        require(receipt.get(key) == source_spec.get(key) == spec.get(key), "Import market identity changed: " + key)
    for algorithm in RETAINED_ALGORITHMS:
        require(source_spec["recipes"][algorithm] == spec["recipes"][algorithm]
                and source_spec["timesteps_per_model"][algorithm] == spec["timesteps_per_model"][algorithm], "Imported algorithm recipe or budget changed")
    for name, key in (("fit_complete.json", "source_fit_complete_sha256"), ("validation_audit.json", "source_validation_audit_sha256"), ("test_audit.json", "source_test_audit_sha256")):
        path = original(name)
        require(receipt.get(key) == sha(path), "Original completion or audit bytes changed: " + name)
        value = read(path)
        if name.endswith("_audit.json"):
            require(value.get("passed") is True and value.get("failure_count") == 0 and value.get("protocol_sha256") == receipt["source_protocol_sha256"]
                    and value.get("accounts_checked") == 56, "Original retained financial audit must remain passed and bound")
        else:
            require(value.get("model_count") == 20 and len(value.get("records", [])) == 20, "Original training completion is incomplete")
    importer = ROOT / "scripts/extend_single_pair_rl.py"
    require(importer.is_file() and receipt.get("import_source_sha256") == sha(importer), "Actual training import helper differs from its receipt")
    external[str(importer.resolve())] = sha(importer)
    retained = {(symbol, algorithm, seed) for symbol in runner.SYMBOLS for algorithm in RETAINED_ALGORITHMS for seed in runner.SEEDS}
    fresh = {(symbol, algorithm, seed) for symbol in runner.SYMBOLS for algorithm in ALGORITHMS if algorithm not in RETAINED_ALGORITHMS for seed in runner.SEEDS}
    def identities(rows):
        require(isinstance(rows, list) and all(isinstance(row, dict) and type(row.get("seed")) is int for row in rows), "Exact integer training import identities required")
        values = [(r.get("symbol"), r.get("algorithm"), r["seed"]) for r in rows]
        require(len(values) == len(set(values)), "Duplicate training import identity")
        return set(values)
    require(identities(receipt.get("retained_models")) == retained and identities(receipt.get("new_models")) == fresh, "Incomplete or foreign retained/new model population")
    rows, local_pins, original_records = [], {}, []
    for item in receipt["retained_models"]:
        symbol, algorithm, seed = item["symbol"], item["algorithm"], item["seed"]
        name = f"training/{symbol}/{algorithm}/seed{seed}"
        require(item.get("source_directory") == name and item.get("destination_directory") == name, "Imported model directory identity changed")
        source_record_path = original(name + "/record.json")
        source_directory, destination_directory = source_record_path.parent, relative(root, name + "/record.json").parent
        record = runner.validate_training_record(retained_root, symbol, algorithm, seed, source_spec)
        source_files = {p.relative_to(source_directory).as_posix() for p in source_directory.rglob("*") if p.is_file()}
        destination_files = {p.relative_to(destination_directory).as_posix() for p in destination_directory.rglob("*") if p.is_file()}
        require(isinstance(item.get("file_sha256"), dict) and source_files == destination_files == set(item["file_sha256"]), "Every imported directory file must be preserved")
        for filename, expected in item["file_sha256"].items():
            a, b = original(name + "/" + filename), relative(root, name + "/" + filename)
            require(sha(a) == expected == sha(b), "Retained model bytes differ: " + name + "/" + filename)
            local_pins[name + "/" + filename] = expected
        require(item.get("source_record_sha256") == sha(source_record_path)
                and item.get("model_parameter_sha256") == record["checkpoint"]["parameter_sha256"]
                and item.get("source_completed_at") == record["completed_at"], "Retained model record/tensor identity changed")
        require(pd.Timestamp(receipt["imported_at"]) >= pd.Timestamp(record["completed_at"]), "Model import predates original completion")
        original_records.append(record)
        rows.append({"symbol": symbol, "algorithm": algorithm, "seed": seed, "training_origin": "retained",
                     "source_directory": name, "destination_directory": name, "preserved_files": len(source_files),
                     "source_record_sha256": item["source_record_sha256"], "model_parameter_sha256": item["model_parameter_sha256"],
                     "source_completed_at": item["source_completed_at"], "source_protocol_sha256": receipt["source_protocol_sha256"], "imported_at": receipt["imported_at"]})
    key = lambda row: (row["symbol"], row["algorithm"], row["seed"])
    require(sorted(read(retained_root / "fit_complete.json")["records"], key=key) == sorted(original_records, key=key), "Original completion records do not equal retained model records")
    provenance = {"retained_models": 20, "fresh_models": 40, "retained_algorithms": list(RETAINED_ALGORITHMS),
                  "source_protocol_sha256": receipt["source_protocol_sha256"], "import_receipt_sha256": sha(root / "training_import.json"),
                  "source_fit_complete_sha256": receipt["source_fit_complete_sha256"], "source_validation_audit_sha256": receipt["source_validation_audit_sha256"],
                  "source_test_audit_sha256": receipt["source_test_audit_sha256"], "import_source_sha256": receipt["import_source_sha256"],
                  "prior_sac_tqc_annual_results_observed": True, "all_source_destination_bytes_equal": True}
    return rows, local_pins, external, provenance


def verify_extension_manifest(root, spec):
    manifest = read(relative(root, "extension_manifest.json"))
    require(manifest.get("schema_version") == 1 and manifest.get("extension_source") == "extension_source/scripts/extend_single_pair_rl.py"
            and manifest.get("extension_source_sha256") == sha(relative(root, manifest["extension_source"])) == sha(ROOT / "scripts/extend_single_pair_rl.py")
            and manifest.get("training_import_sha256") == sha(relative(root, "training_import.json"))
            and manifest.get("core_protocol_sha256") == sha(relative(root, "protocol.json"))
            and manifest.get("new_training_models") == 40 and manifest.get("retained_training_models") == 20
            and manifest.get("annual_accounts") == manifest.get("validation_accounts") == ACCOUNT_COUNT
            and manifest.get("repeat_every_account") is True
            and manifest.get("performance_mode") == "native; no GPU replay cache or Polyak graph installed", "Frozen extension source, budgets or native execution mode changed")
    require(manifest.get("upstream_root") == "extension_source/upstream/DroQ", "Unexpected frozen author-source root")
    upstream_name = manifest["upstream_root"] + "/verified_manifest.json"
    upstream = read(relative(root, upstream_name))
    require(sha(relative(root, upstream_name)) == manifest.get("upstream_manifest_sha256")
            and upstream.get("commit") == manifest.get("upstream_commit") and upstream.get("file_count") == 27
            and upstream.get("all_git_blob_sha1_verified") is True
            and len(upstream.get("files", [])) == 27
            and {item["path"]: item["sha256"] for item in upstream["files"]} == manifest.get("upstream_files_sha256"), "Frozen author source manifest changed")
    names = ["extension_manifest.json", manifest["extension_source"], upstream_name]
    for name, expected in manifest["upstream_files_sha256"].items():
        filename = manifest["upstream_root"] + "/" + name
        require(sha(relative(root, filename)) == expected, "Frozen author source changed: " + name)
        names.append(filename)
    return manifest, {name: sha(relative(root, name)) for name in names}


def verify_native_training_counters(record, spec):
    """Recompute native integer budgets and actual fragment boundaries read-only."""
    algorithm, train = record["algorithm"], record["training"]
    steps, counters = spec["timesteps_per_model"][algorithm], train["counters"]
    require(train.get("actual_steps") == steps, "Actual environment budget changed")
    def exact(observed, expected):
        if type(expected) is int:
            return type(observed) is int and observed == expected
        if isinstance(expected, list):
            return isinstance(observed, list) and len(observed) == len(expected) and all(exact(a, b) for a, b in zip(observed, expected))
        if isinstance(expected, dict):
            return isinstance(observed, dict) and set(observed) == set(expected) and all(exact(observed[k], v) for k, v in expected.items())
        return observed == expected
    expected = {"environment_steps": steps}
    if algorithm in ("sac", "tqc", "crossq", "td3"):
        updates = steps - 512
        delay = 3 if algorithm == "crossq" else 2 if algorithm == "td3" else 1
        optimizers = {"actor": updates // delay, "critic": updates}
        if algorithm != "td3":
            optimizers["entropy"] = updates // delay
        expected.update(gradient_update_rounds=updates, optimizer_steps=optimizers, replay_transitions=steps, replay_add_calls=steps)
    elif algorithm == "recurrent_ppo":
        updates = steps // 128 * 5
        expected.update(gradient_update_rounds=updates, optimizer_steps={"policy": updates})
    else:
        require(algorithm == "droq", "Unknown training counter identity")
        calls, rounds = steps - 5000, (steps - 5000) * 20
        expected.update(q_inner_rounds=rounds, q_optimizer_steps=[rounds, rounds], total_q_optimizer_steps=rounds * 2,
            policy_optimizer_steps=calls, alpha_optimizer_steps=calls, target_updates_per_network=[rounds, rounds],
            learning_train_calls=calls, replay_adds=steps, random_exploration_actions=5001)
        require(all(train.get("effective_parameters", {}).get(k) == v for k, v in {"obs_dim": 13, "act_dim": 1, "target_entropy": -1.,
                    "utd_ratio": 20, "num_Q": 2, "num_min": 2, "policy_update_delay": 20}.items()), "Actual DroQ single-action recipe changed")
    require(all(exact(counters.get(k), value) for k, value in expected.items()), "Native training optimizer, replay or target counters changed: " + algorithm)
    history = train.get("training_episode_history", [])
    completed = [episode for episode in history if episode.get("completed_steps", 0) > 0]
    require(len(completed) == steps // 32 and sum(e["completed_steps"] for e in completed) == steps, "Complete actual 32-day training fragments required")
    for episode in completed:
        start, end, until = map(pd.Timestamp, (episode["start"], episode["end_exclusive"], episode["completed_until"]))
        require(type(episode["completed_steps"]) is int and episode["completed_steps"] == 32 and episode.get("terminated") is False
                and episode.get("truncated") is True and pd.Timestamp(runner.TRAIN[0]) <= start < end <= pd.Timestamp(runner.TRAIN[1])
                and end == start + pd.Timedelta(days=32) and until == end, "Training fragment must end at its real timeout inside the training window")


def verify_training_counter_audit(root, spec, records, manifest):
    audit = read(relative(root, "training_counter_audit.json"))
    require(audit.get("passed") is True and audit.get("failure_count") == 0 and audit.get("failures") == []
            and audit.get("models_checked") == MODEL_COUNT and audit.get("retained_models") == 20 and audit.get("new_models") == 40
            and audit.get("extension_source_sha256") == manifest["extension_source_sha256"]
            and audit.get("core_protocol_sha256") == sha(relative(root, "protocol.json")), "Complete source-bound training counter audit must pass")
    models = audit.get("models", [])
    wanted = {(s, a, seed) for s in runner.SYMBOLS for a in ALGORITHMS for seed in runner.SEEDS}
    require(len(records) == len(models) == MODEL_COUNT and all(type(row.get("seed")) is int for row in [*models, *records]), "Exactly 60 training counter rows required")
    by_identity = {(row["symbol"], row["algorithm"], row["seed"]): row for row in models}
    actual = {(row["symbol"], row["algorithm"], row["seed"]): row for row in records}
    require(set(by_identity) == set(actual) == wanted, "Training counter audit has duplicate or foreign model identities")
    for identity, record in actual.items():
        symbol, algorithm, seed = identity
        row = by_identity[identity]
        require(row.get("record_sha256") == sha(relative(root, f"training/{symbol}/{algorithm}/seed{seed}/record.json"))
                and row.get("counters") == record["training"]["counters"]
                and row.get("retained") is (algorithm in RETAINED_ALGORITHMS), "Training counter evidence does not bind this exact record")
        require(pd.Timestamp(audit["audited_at"]) >= pd.Timestamp(record["completed_at"]), "Training counter audit predates actual completion")
        verify_native_training_counters(record, spec)
    require(pd.Timestamp(audit["audited_at"]) >= pd.Timestamp(read(relative(root, "fit_complete.json"))["at"]), "Counter audit must follow complete training")
    resources = read(relative(root, "validation_resources.json"))
    require(resources.get("tasks") == ACCOUNT_COUNT and pd.Timestamp(resources["at"]) >= pd.Timestamp(audit["audited_at"]), "Training counter audit must precede fresh validation execution")
    return {key: audit[key] for key in ("passed", "models_checked", "retained_models", "new_models", "failure_count", "audited_at", "extension_source_sha256", "core_protocol_sha256")}


def expected_cases(split):
    learned = [(s, a, seed, c) for s in runner.SYMBOLS for a in ALGORITHMS for seed in runner.SEEDS for c in runner.COSTS]
    baselines = [(s, a, None, c) for s in runner.SYMBOLS for a in runner.BASELINES for c in runner.COSTS]
    return {(f"{split}__{c}__{s}__{a}__{'baseline' if seed is None else seed}"):
            (s, a, seed, c, f"evaluation/{split}/{s}/{a}/{'baseline' if seed is None else 'seed' + str(seed)}/{c}")
            for s, a, seed, c in learned + baselines}


def verify_case(root, split, identity, row, audited_case):
    symbol, algorithm, seed, scenario, directory = identity
    require(audited_case.get("passed") is True and audited_case.get("failure_count") == 0
            and audited_case.get("relative_path") == directory, "Case audit failed or belongs to another account")
    require(set(audited_case.get("inputs_sha256", {})) == {f"{d}/{name}" for d in (directory, directory.replace("evaluation/", "repeat/", 1)) for name in RESULT_FILES}, "Complete case audit input set required")
    for key, expected in audited_case["inputs_sha256"].items():
        require(sha(relative(root, key)) == expected, "Changed audited case bytes: " + key)
    for name in RESULT_FILES:
        require(sha(relative(root, directory + "/" + name)) == sha(relative(root, directory.replace("evaluation/", "repeat/", 1) + "/" + name)), "Case repeat changed: " + directory)
    p = relative(root, directory + "/summary.json").parent
    actual = read(p / "summary.json")
    equivalent(row, actual, directory + " aggregate")
    require(actual == audited_case.get("summary"), "Audit summary no longer matches case JSON")
    require(actual["symbol"] == symbol and actual["algorithm"] == algorithm and actual["seed"] == seed
            and actual["scenario"] == scenario and actual["split"] == split and actual["ensemble"] is False
            and actual["initial_capital"] == 10000. and actual["primary_seed"] == (seed == 0)
            and actual["independent_capital"] is True, "Independent account identity changed")
    meta = read(p / "metadata.json")
    require(meta.get("pair") == symbol and meta.get("tradable_symbols") == [symbol] and meta.get("no_ensemble") is True
            and meta.get("independent_capital") == 10000. and meta.get("observation_dim") == 13 and meta.get("action_dim") == 1
            and set(meta.get("final_positions", {})) <= {symbol}, "Foreign coin, dimensions or ownership in case metadata")
    fingerprints = read(p / "fingerprints.json")
    require(set(fingerprints) == {"equity", "fills", "weights", "boundary_equity"}, "All four result fingerprints required")
    for key, name in (("equity", "equity_1h.csv.gz"), ("fills", "fills.csv"), ("weights", "weights.csv"), ("boundary_equity", "boundary_equity.csv")):
        payload = (p / name).read_bytes()
        if name.endswith(".gz"):
            payload = gzip.decompress(payload)
        require(hashlib.sha256(payload).hexdigest() == fingerprints[key], "CSV fingerprint changed: " + directory + "/" + key)
    result = {"equity": csv(p / "equity_1h.csv.gz", indexed=True), "fills": csv(p / "fills.csv"),
              "weights": csv(p / "weights.csv"), "boundary_equity": csv(p / "boundary_equity.csv", indexed=True), "metadata": meta}
    for frame in (result["fills"], result["weights"]):
        require(not len(frame) or set(frame.asset_key) == {symbol}, "Foreign inventory in independent account")
    bounds = runner.TEST if split == "test" else runner.VALIDATION
    re_summary, daily, months = summarize(result, run_id=actual["run_id"], strategy=algorithm, seed=seed, scenario=scenario,
        split=split, start=bounds[0], end=bounds[1], capital=10000., **runner.COSTS[scenario])
    equivalent({k: actual[k] for k in re_summary}, re_summary, directory + " serialized readback")
    daily["timestamp"] = daily.timestamp.map(str)
    saved_daily = csv(p / "daily_equity.csv")
    saved_daily["timestamp"] = pd.to_datetime(saved_daily.timestamp, utc=True).map(str)
    frame_equivalent(saved_daily, daily, directory + " daily")
    frame_equivalent(csv(p / "monthly_returns.csv"), months, directory + " monthly")
    for frame in (daily, months):
        frame["symbol"], frame["algorithm"] = symbol, algorithm
    return actual, daily.to_dict("records"), months.to_dict("records")


def make_reviewed_snapshot(study_root, retained_study_root):
    root = Path(study_root).resolve()
    pins = {}
    def take(name):
        path = relative(root, name)
        pins[name] = sha(path)
        return read(path)
    spec = take("protocol.json")
    receipt = take("protocol_receipt.json")
    require(receipt.get("protocol_sha256") == pins["protocol.json"], "Protocol receipt mismatch")
    require(spec.get("symbols") == list(runner.SYMBOLS) and len(spec.get("algorithms", [])) == len(ALGORITHMS) and set(spec.get("algorithms", [])) == set(ALGORITHMS)
            and spec.get("seeds") == list(runner.SEEDS) and spec.get("primary_seed") == 0
            and spec.get("training_model_count") == MODEL_COUNT and spec.get("capital_per_account") == 10000.
            and spec.get("primary_seed_fixed_before_training") is True and spec.get("capital_shared_between_pairs") is False
            and spec.get("ensemble") is False and spec.get("test_year_already_observed") is True and spec.get("fresh_holdout") is False
            and spec.get("no_production_change") is True, "A complete 60-model independent-account protocol is required")
    require(spec.get("train") == list(runner.TRAIN) and spec.get("validation") == list(runner.VALIDATION)
            and spec.get("test") == list(runner.TEST) and spec.get("costs") == runner.COSTS
            and set(spec.get("sources", {})) == set(runner.SOURCE_FILES)
            and spec.get("timesteps_per_model") == {a: 65536 if a == "td3" else 32768 for a in ALGORITHMS}, "Frozen sources, budgets, dates or costs changed")
    require(spec.get("research_gates_per_pair") == {"base_return_at_least": .05, "base_drawdown_magnitude_at_most": .20,
            "stress_return_at_least": 0., "positive_base_seeds_at_least": 3, "beats_same_coin_hold_return_and_drawdown": True}, "Research target contract changed")
    require(spec.get("recipes", {}).get("droq", {}).get("utd_ratio") == 20,
            "The displayed DroQ compute disclosure must match its frozen recipe")
    for key, value in {"observation_dim": 13, "action_dim": 1, "cash_floor": .05, "max_coin_target": .95,
                       "rebalance_band": .025, "drawdown_penalty": .25, "episode_days": 32}.items():
        require(spec["environment"].get(key) == value, "Single-coin environment contract changed: " + key)
    for name, expected in spec["sources"].items():
        pins["frozen_source/" + name] = sha(relative(root, "frozen_source/" + name))
        require(pins["frozen_source/" + name] == expected, "Frozen source changed: " + name)
        require(sha(relative(ROOT, name)) == expected, "Report readback helper differs from frozen source: " + name)
    import_rows, imported_pins, external_pins, provenance = verify_training_import(root, retained_study_root, spec)
    take("training_import.json")
    take("training_import_receipt.json")
    pins.update(imported_pins)
    extension_manifest, extension_pins = verify_extension_manifest(root, spec)
    pins.update(extension_pins)
    training, training_original = [], []
    for symbol in runner.SYMBOLS:
        for algorithm in ALGORITHMS:
            for seed in runner.SEEDS:
                record = runner.validate_training_record(root, symbol, algorithm, seed, spec)
                if algorithm not in RETAINED_ALGORITHMS:
                    require(pd.Timestamp(record["started_at"]) >= pd.Timestamp(spec["frozen_at"]), "Fresh algorithm record predates this six-algorithm study")
                name = f"training/{symbol}/{algorithm}/seed{seed}/record.json"
                take(name)
                directory = relative(root, name).parent
                actual_files = {p.relative_to(directory).as_posix() for p in (directory / "checkpoint").rglob("*") if p.is_file()}
                require(actual_files == set(record["files"]), "Incomplete checkpoint file inventory")
                for key, expected in record["files"].items():
                    filename = name.rsplit("/", 1)[0] + "/" + key
                    pins[filename] = sha(relative(root, filename))
                    require(pins[filename] == expected, "Training checkpoint bytes changed")
                training_original.append(record)
                counters = record["training"]["counters"]
                training.append({"symbol": symbol, "algorithm": algorithm, "seed": seed, "capital": 10000., "observation_dim": 13,
                    "action_dim": 1, "ensemble": False, "actual_steps": record["training"]["actual_steps"],
                    "gradient_update_rounds": counters.get("gradient_update_rounds"), "q_inner_rounds": counters.get("q_inner_rounds"),
                    "learning_train_calls": counters.get("learning_train_calls"), "native_counters_json": json.dumps(counters, ensure_ascii=False, sort_keys=True),
                    "training_origin": "retained" if algorithm in RETAINED_ALGORITHMS else "fresh",
                    "maximum_input_hour": record["maximum_input_hour"], "record_file": name,
                    "record_sha256": pins[name], "adapter_sha256": record["checkpoint"]["adapter_sha256"]})
    expected_records = {r["record_file"] for r in training}
    require({p.relative_to(root).as_posix() for p in (root / "training").rglob("record.json")} == expected_records, "Unexpected or incomplete trained model population")
    fit = take("fit_complete.json")
    require(fit.get("model_count") == MODEL_COUNT and len(fit.get("records", [])) == MODEL_COUNT
            and sorted(fit["records"], key=lambda r: (r["symbol"], r["algorithm"], r["seed"])) == sorted(training_original, key=lambda r: (r["symbol"], r["algorithm"], r["seed"])), "Training completion does not cover all exact records")
    counter_audit = verify_training_counter_audit(root, spec, training_original, extension_manifest)
    take("training_counter_audit.json")
    take("validation_resources.json")
    summaries, histories, monthly, audits = {}, [], [], []
    for split in ("validation", "test"):
        audit = take(split + "_audit.json")
        require(audit.get("phase") == split and audit.get("passed") is True and audit.get("failure_count") == 0
                and not audit.get("failures") and audit.get("checks_total", 0) > 0
                and all(audit.get(k) == ACCOUNT_COUNT for k in ("expected_accounts", "accounts_checked", "repeats_checked", "accounts_count"))
                and audit.get("capital_per_account") == 10000. and audit.get("capital_shared_between_pairs") is False
                and audit.get("ensemble") is False and audit.get("primary_seed") == 0, "Complete passed financial audit required: " + split)
        require(audit.get("protocol_sha256") == pins["protocol.json"] and audit.get("summary_sha256") == sha(relative(root, split + "_summary.csv")), "Audit summary or protocol binding changed")
        for key, expected in audit.get("inputs_sha256", {}).items():
            pins[key] = sha(relative(root, key))
            require(pins[key] == expected, "Changed phase audit input: " + key)
        require(set(audit.get("inputs_sha256", {})) == {"protocol.json", split + "_summary.csv", split + "_complete.json"}, "Complete phase audit input binding required")
        frame = csv(relative(root, split + "_summary.csv"))
        expected = expected_cases(split)
        require(len(frame) == ACCOUNT_COUNT and not frame.run_id.duplicated().any() and set(frame.run_id) == set(expected), "Exactly 136 distinct accounts required: " + split)
        observed_cases = {p.parent.relative_to(root).as_posix() for p in (root / "evaluation" / split).rglob("summary.json")}
        require(observed_cases == {i[-1] for i in expected.values()}, "Unexpected result folders")
        cases = {case.get("relative_path"): case for case in audit.get("cases", [])}
        require(len(audit.get("cases", [])) == ACCOUNT_COUNT and set(cases) == observed_cases, "Every account needs its own passed oracle result")
        completion = take(split + "_complete.json")
        require(completion.get("accounts") == completion.get("repeats") == ACCOUNT_COUNT and completion.get("repeat_equal") is True
                and len(completion.get("results", [])) == ACCOUNT_COUNT, "Incomplete execution completion")
        completed = {r["summary"]["run_id"]: r for r in completion["results"]}
        require(set(completed) == set(expected), "Completion account identities changed")
        rows = []
        for aggregate in frame.to_dict("records"):
            identity = expected[aggregate["run_id"]]
            case = cases[identity[-1]]
            row, daily, months = verify_case(root, split, identity, aggregate, case)
            item = completed[row["run_id"]]
            require(item.get("path") == identity[-1] and item.get("repeat_path") == identity[-1].replace("evaluation/", "repeat/", 1)
                    and item.get("repeat_equal") is True and item.get("summary") == row, "Wrong canonical/repeat completion path")
            pins.update(case["inputs_sha256"])
            rows.append(row)
            if split == "test":
                histories.extend(daily); monthly.extend(months)
        summaries[split] = rows
        audits.append({k: audit[k] for k in ("phase", "passed", "checks_total", "failure_count", "accounts_checked", "repeats_checked", "audited_at", "independent_oracle")})
    selection = take("validation_selection.json")
    require(selection.get("uses_test") is False and selection.get("seed_selection") is False
            and set(selection.get("selection", {})) == set(runner.SYMBOLS)
            and selection.get("protocol_sha256") == pins["protocol.json"]
            and selection.get("validation_audit_sha256") == pins["validation_audit.json"]
            and selection.get("validation_summary_sha256") == pins["validation_summary.csv"], "Selection does not bind audited validation-only evidence")
    candidates, selection_score_errors = [], {}
    # Match the frozen selector's original CSV decoding and operation order;
    # display readback elsewhere deliberately retains full JSON precision.
    score_frame = pd.read_csv(relative(root, "validation_summary.csv"))
    for symbol in runner.SYMBOLS:
        scored = []
        for algorithm in ALGORITHMS:
            group = score_frame.loc[(score_frame.symbol == symbol) & (score_frame.algorithm == algorithm)]
            scores = [float(np.mean([math.log1p(r.net_return) - .5 * abs(r.max_drawdown) - 2 * max(abs(r.max_drawdown) - .2, 0.)
                for r in group.loc[group.seed == seed].itertuples()])) for seed in runner.SEEDS]
            scored.append({"algorithm": algorithm, "score_median": float(np.median(scores)), "seed_scores": scores})
        scored.sort(key=lambda r: (-r["score_median"], r["algorithm"]))
        choice = selection["selection"][symbol]
        selection_score_errors[symbol] = verify_selection_choice(choice, scored)
        candidates.extend({"symbol": symbol, "algorithm": r["algorithm"], "score_median": r["score_median"], "selected": r["algorithm"] == choice["algorithm"],
                           "primary_seed": 0, **{f"seed{i}_score": value for i, value in enumerate(r["seed_scores"])}} for r in choice["all_candidates"])
    started = take("test_started.json")
    require(started.get("pins") == {"selection_sha256": pins["validation_selection.json"], "validation_audit_sha256": pins["validation_audit.json"],
            "protocol_sha256": pins["protocol.json"], "dataset_sha256": spec["dataset_sha256"]}, "Annual execution did not pin this selection before starting")
    require(pd.Timestamp(selection["at"]) >= pd.Timestamp(audits[0]["audited_at"])
            and pd.Timestamp(started["at"]) >= pd.Timestamp(selection["at"])
            and pd.Timestamp(read(relative(root, "test_complete.json"))["at"]) >= pd.Timestamp(started["at"])
            and pd.Timestamp(audits[1]["audited_at"]) >= pd.Timestamp(read(relative(root, "test_complete.json"))["at"]), "Validation, selection, annual start and final audit are out of chronological order")
    stats, gates = [], []
    for symbol in runner.SYMBOLS:
        for algorithm in ALGORITHMS:
            for scenario in runner.COSTS:
                group = [r for r in summaries["test"] if r["symbol"] == symbol and r["algorithm"] == algorithm and r["scenario"] == scenario]
                require(len(group) == 5 and {r["seed"] for r in group} == set(runner.SEEDS), "Five individual seeds required for statistics")
                returns = [r["net_return"] for r in group]
                stats.append({"symbol": symbol, "algorithm": algorithm, "scenario": scenario, "seeds": 5, "positive_seeds": sum(x > 0 for x in returns),
                    "median_return": float(np.median(returns)), "min_return": min(returns), "max_return": max(returns),
                    "worst_drawdown": min(r["max_drawdown"] for r in group)})
        algorithm = selection["selection"][symbol]["algorithm"]
        primary = {r["scenario"]: r for r in summaries["test"] if r["symbol"] == symbol and r["algorithm"] == algorithm and r["seed"] == 0}
        hold = next(r for r in summaries["test"] if r["symbol"] == symbol and r["algorithm"] == "buy_hold" and r["scenario"] == "base")
        positives = next(r["positive_seeds"] for r in stats if r["symbol"] == symbol and r["algorithm"] == algorithm and r["scenario"] == "base")
        measures = [("base_return", "主成本收益至少5%", primary["base"]["net_return"], .05, primary["base"]["net_return"] >= .05),
                    ("base_drawdown", "主成本回撤不超过20%", abs(primary["base"]["max_drawdown"]), .2, abs(primary["base"]["max_drawdown"]) <= .2),
                    ("stress_return", "压力成本收益不低于0", primary["stress"]["net_return"], 0., primary["stress"]["net_return"] >= 0),
                    ("positive_seeds", "主成本至少3/5种子盈利", positives, 3, positives >= 3),
                    ("beats_hold", "收益与回撤同时优于同币95%持有", None, None, primary["base"]["net_return"] > hold["net_return"] and abs(primary["base"]["max_drawdown"]) < abs(hold["max_drawdown"]))]
        gates.extend({"symbol": symbol, "algorithm": algorithm, "seed": 0, "gate": key, "label": label, "actual": value, "threshold": threshold, "passed": passed} for key, label, value, threshold, passed in measures)
    queries = {}
    common = {"label": "独立单币现货研究账户", "classification": "modeled", "timezone": "UTC", "period": {"start": runner.TEST[0], "end": runner.TEST[1]},
        "caveats": ["本年度已被其他研究观察，不是新的盲测样本。", "五随机种子反映同一行情上的训练波动，不是五份独立市场样本。", "最大回撤按完整小时收盘计算，图表为日末显示。", "手续费和滑点为研究假设，不含盘口队列与全部历史过滤器。", "每个账户独立10,000 USDT，无账户间资金、库存、特征或模型平均。"],
        "metricDefinitions": [{"label": "扣费收益", "definition": "期末真实现金加持币市值/10000-1；已扣模拟交易手续费与不利滑点。", "componentIds": ["single-primary", "single-seeds", "single-baselines", "single-details", "single-curve"]},
            {"label": "最大回撤", "definition": "包含期初10000的历史峰值，与完整小时收盘权益之比的最小值。表中显示跌幅绝对值，越小越好。"},
            {"label": "盈利种子", "definition": "同币同算法五份实际单模型主成本年度收益>0的数量，分母固定5。"},
            {"label": "净清算收益", "definition": "期末持仓按统一滑点和手续费假设退出后的权益/10000-1，仅估算，未强制成交。"}],
        "evidenceFlow": [{"title": "独立账户与训练来源", "detail": "每币/算法/种子独立账户，13维本币观测和1维动作。SAC/TQC20模型从首轮逐字节保留，另4算法40模型本轮训练；60份原始训练记录和manifest全量核对。"},
            {"title": "验证先于年度", "detail": "先完整审计验证136账户，再按预设五种子中位评分选算法；主seed0在训练前固定。年度两成本分别推理，无多模型平均。"},
            {"title": "真实输出读回", "detail": "逐个核对两阶段136账户与重复文件、原审计输入SHA、CSV fingerprints，并从保存小时账本重算摘要、每日与月度展示值。"}]}
    def query(key, rows, files, method, period=None):
        source = deepcopy(common)
        names = list(dict.fromkeys([*files, "protocol.json", "protocol_receipt.json", "validation_selection.json", "validation_audit.json", "test_audit.json",
                                   "training_import.json", "training_import_receipt.json", "extension_manifest.json", "training_counter_audit.json"]))
        source.update(files=names, fileHashes={n: pins[n] for n in names}, executedAt=max(a["audited_at"] for a in audits))
        if period:
            source["period"] = {"start": period[0], "end": period[1]}
        queries[key] = {"rows": clean_json(rows), "source": source, "methods": [{"language": "text", "code": method}]}
    query("account_summary", summaries["test"], ["test_summary.csv"], "Exactly136 independent test accounts:60 learned models at2 costs plus4 fixed baselines per coin at2 costs. No cross-coin aggregation or target averaging.")
    query("equity_history", histories, [p for p in pins if p.startswith("evaluation/test/") and p.endswith("daily_equity.csv")], "Original daily boundary points per real account; all366 including initial10000. Plotted return/drawdown fractions are display-only multiplied by100. No median/averaged equity curve.")
    query("monthly_returns", monthly, [p for p in pins if p.startswith("evaluation/test/") and p.endswith("monthly_returns.csv")], "Actual partial first/last months per independent account; compound to each account's net return.")
    query("seed_statistics", stats, ["test_summary.csv"], "Five exact seeds0..4 per coin/algorithm/scenario. Median/min/max describe terminal outcomes only; they are not traded accounts.")
    query("validation_summary", summaries["validation"], ["validation_summary.csv"], "All136 validation accounts, separate from annual test; same capital/cost/ownership definitions.", runner.VALIDATION)
    query("validation_selection", candidates, ["validation_summary.csv", "validation_selection.json"], "Recompute per-seed mean base/stress log1p(return)-.5*abs(DD)-2*max(abs(DD)-.20,0); choose algorithm by median of5 scores and identifier tie-break. Always retain predeclared seed0.", runner.VALIDATION)
    query("training_records", training, ["fit_complete.json", "training_counter_audit.json", "validation_resources.json", *sorted(expected_records)], "Complete60 training manifests and payload hashes, plus mandatory passed source/protocol-bound60-row counter audit. Recompute native integer actor/critic/entropy/replay/target budgets and every32-day timeout fragment from actual records; audit must follow fit and precede validation. TD3 has65536 environment steps, others32768. Report native SB3 update rounds and DroQ Q-inner-rounds separately; DroQ uses20 Q rounds per learning call.20 retained SAC/TQC plus40 fresh models, not equal compute or60 new trainings.", runner.TRAIN)
    queries["training_records"]["payloadColumns"] = ["native_counters_json"]
    queries["training_records"]["source"]["trainingCounterAudit"] = counter_audit
    query("training_import", import_rows, ["training_import.json", "training_import_receipt.json", *sorted(name for name in expected_records if any('/' + algorithm + '/' in name for algorithm in RETAINED_ALGORITHMS))],
          "20 original SAC/TQC identities: original protocol/fit/validation audit/test audit hashes, exact15 compatibility fields, per-algorithm recipes/budgets and complete directory inventory checked. Every source and destination record/progress/checkpoint file is byte-identical. Original source location is mapped explicitly with retained-study-root; no original evaluation/selection is used as six-algorithm evidence.", runner.TRAIN)
    queries["training_import"]["source"]["trainingProvenance"] = provenance
    query("research_targets", gates, ["test_summary.csv", "validation_selection.json"], "Five original per-coin research gates apply to validation-selected algorithm's seed0. Positive-seed denominator comes from all5 base accounts. Better hold means strictly greater return and strictly smaller DD magnitude.")
    query("audit_evidence", audits, ["validation_audit.json", "test_audit.json"], "Original independent closed-form and actual-fill financial audits must both pass all136 accounts; SHA-only rebinding of all inputs happens before report output. Audit success does not imply target attainment.")
    query("protocol", [{"capital_per_account": 10000., "training_models": MODEL_COUNT, "test_accounts": ACCOUNT_COUNT, "seeds_per_algorithm": 5, "primary_seed": 0, "observation_dim": 13, "action_dim": 1,
        "cash_shared": False, "ensemble": False, "test_start": runner.TEST[0], "test_end_exclusive": runner.TEST[1], "test_year_already_observed": True,
        "dataset_sha256": spec["dataset_sha256"], "protocol_sha256": pins["protocol.json"]}], ["protocol.json"], "Frozen protocol bytes and receipt; source inputs are displayed as hashes only. This report never loads the original pickle or constructs a policy.")
    score_readback = {"scope": "validation selection scores only", "absolute_tolerance": SELECTION_SCORE_ABS_TOL, "relative_tolerance": 0.,
                      "maximum_absolute_error": max(selection_score_errors.values()), "per_symbol_maximum_absolute_error": selection_score_errors,
                      "frozen_display_values_preserved": True, "identities_and_candidate_order_exact": True}
    queries["validation_selection"]["source"]["selectionScoreReadback"] = score_readback
    snapshot = {"surface": "report", "status": "reviewed", "buildStatus": "complete", "title": "六算法独立交易：BTC 与 ETH 的收益与回撤", "generatedAt": datetime.now(timezone.utc).isoformat(),
        "filters": [], "queries": queries, "report": {"asOf": "2026-10-22", "scope": "independent_single_pair_six_accounts", "studyConfig": spec,
        "selection": selection, "auditStatus": "passed", "primarySeed": 0, "noEnsemble": True, "noPooledCapital": True,
        "selectionScoreReadback": score_readback, "trainingProvenance": provenance, "trainingCounterAudit": counter_audit}}
    evidence = {"input_sha256": pins, "accounts_per_phase": ACCOUNT_COUNT, "training_models": MODEL_COUNT, "numeric_results_changed": False,
                "pickle_loaded": False, "policy_constructed": False, "selection_score_readback": score_readback,
                "external_input_sha256": external_pins, "training_provenance": provenance, "training_counter_audit": counter_audit,
                "source_sha256": {str(Path(__file__).relative_to(ROOT)): sha(__file__), "scripts/btceth_rl_report_app.py": sha(ROOT / "scripts/btceth_rl_report_app.py")}}
    require(set(queries) == set(QUERY_IDS) and len(histories) == ACCOUNT_COUNT * 366 and len(monthly) == ACCOUNT_COUNT * 13
            and len(training) == MODEL_COUNT and len(candidates) == len(runner.SYMBOLS) * len(ALGORITHMS)
            and len(stats) == len(runner.SYMBOLS) * len(ALGORITHMS) * len(runner.COSTS), "Complete six-algorithm report query population required")
    verify_inputs(root, evidence)
    return snapshot, evidence


def verify_inputs(root, evidence):
    for name, expected in evidence["input_sha256"].items():
        require(sha(relative(root, name)) == expected, "Evidence changed during report preparation: " + name)
    for name, expected in evidence.get("external_input_sha256", {}).items():
        require(sha(Path(name)) == expected, "Original retained evidence or import helper changed: " + name)


def build_report(study_root, *, retained_study_root, output_dir=None, build_mode="native", wsl_node=None, wsl_distro="Ubuntu-26.04", node_path=None, plugin_root=None):
    root = Path(study_root).resolve()
    snapshot, evidence = make_reviewed_snapshot(root, retained_study_root)
    destination = Path(output_dir).resolve() if output_dir else root / "report"
    project = destination / "report_app"
    ownership = project / ".single-pair-rl-six-authoring.json"
    prior = read(ownership) if ownership.is_file() else None
    if destination.exists() and any(destination.iterdir()):
        require(prior is not None, "Preserve nonempty report destination without single-pair ownership")
        require(prior.get("scope") == "independent_single_pair_six_accounts", "Preserve a different report identity")
        for name, expected in prior.get("authored_sha256", {}).items():
            require(sha(relative(project, name)) == expected, "Preserve user presentation edits: " + name)
        for name, expected in prior.get("delivery_sha256", {}).items():
            require(sha(relative(destination, name)) == expected, "Preserve changed delivered evidence: " + name)
    elif project.exists():
        raise FileExistsError("Unowned app project")
    destination.mkdir(parents=True, exist_ok=True)
    reviewed = destination / "reviewed.json"
    write(reviewed, snapshot)
    write(destination / "report_input_manifest.json", evidence)
    runtime = DataRuntime(node_path=node_path, plugin_root=plugin_root, build_mode=build_mode, wsl_node=wsl_node, wsl_distro=wsl_distro)
    if project.exists():
        preparation = runtime.run("data-app.mjs", "prepare", "--project-dir", project, cwd=destination)
    else:
        preparation = runtime.run("prepare-data-app.mjs", "--surface", "report", "--output", project, "--snapshot", reviewed, cwd=destination)
    require("src/content/" in (project / "AGENTS.md").read_text(encoding="utf-8"), "Unexpected official Data authoring boundary")
    documentation = preparation["documentation"]
    host_path(documentation["entryPoint"]).read_text(encoding="utf-8")
    component_root = host_path(documentation.get("referenceRoot", str(runtime.plugin / "docs/components")))
    for name in ("narrative/reports.md", "charts/charts.md", "tables/tables.md", "controls/controls.md", "sources/sources.md"):
        (component_root / name).read_text(encoding="utf-8")
    content = project / "src/content/report"
    sources = {Path(__file__).with_suffix(".jsx"): content / "ReportContent.jsx", Path(__file__).with_suffix(".css"): content / "single-pair-rl-six-report.css"}
    for source, target in sources.items():
        shutil.copyfile(source, target)
    prepared = read(project / "src/data.json")
    snapshot["id"] = prepared["id"]
    if "legacyPresentationTitle" in prepared:
        snapshot["legacyPresentationTitle"] = prepared["legacyPresentationTitle"]
    write(project / "src/data.json", snapshot)
    authored = {path.relative_to(project).as_posix(): sha(path) for path in sources.values()}
    write(ownership, {"scope": "independent_single_pair_six_accounts", "authored_sha256": authored})
    build = runtime.run("data-app.mjs", "build", "--project-dir", project, "--separate-data", cwd=destination)
    exported = project / ".data-app-offline/exports/single-pair-rl-six.html"
    exported_result = runtime.run("data-app.mjs", "export-html", "--project-dir", project, "--output", exported, cwd=destination)
    verify_inputs(root, evidence)
    delivered = destination / "report.html"
    shutil.copyfile(exported, delivered)
    result = {"scope": "independent_single_pair_six_accounts", "html_path": str(delivered), "preview_path": str(project / "dist/index.html"),
              "project_dir": str(project), "reviewed_sha256": sha(reviewed), "html_sha256": sha(delivered), "runtime": build, "export": exported_result,
              "build_mode": build_mode, "published": False, "numeric_results_changed": False,
              "authoring_source_sha256": {p.name: sha(p) for p in (Path(__file__), *sources)}}
    write(destination / "report_build.json", result)
    write(ownership, {"scope": "independent_single_pair_six_accounts", "authored_sha256": authored,
          "delivery_sha256": {p.name: sha(p) for p in (reviewed, destination / "report_input_manifest.json", delivered, destination / "report_build.json")}})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", required=True, type=Path)
    parser.add_argument("--retained-study-root", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--build-mode", choices=("native", "wsl"), default="native")
    parser.add_argument("--wsl-node")
    parser.add_argument("--wsl-distro", default="Ubuntu-26.04")
    parser.add_argument("--node-path")
    parser.add_argument("--plugin-root")
    args = parser.parse_args()
    print(json.dumps(build_report(args.study_root, retained_study_root=args.retained_study_root, output_dir=args.output_dir, build_mode=args.build_mode, wsl_node=args.wsl_node,
          wsl_distro=args.wsl_distro, node_path=args.node_path, plugin_root=args.plugin_root), ensure_ascii=False))


if __name__ == "__main__":
    main()

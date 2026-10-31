"""Audit-gated BTC SAC/TD3 seed-distribution report; no model inference/training."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts import backtest_single_pair_rl as core
from scripts.btceth_rl_report_app import DataRuntime, host_path
from scripts.crypto_rl_reporting import clean_json
from scripts.single_pair_rl_six_report_app import csv, verify_case, verify_native_training_counters

ALGORITHMS = ("sac", "td3")
SEEDS = tuple(range(30))
NEW_SEEDS = tuple(range(5, 30))
COHORTS = {"new25": NEW_SEEDS, "all30": SEEDS}
MODEL_COUNT, ACCOUNT_COUNT = 60, 128
RESULT_FILES = ("equity_1h.csv.gz", "fills.csv", "weights.csv", "boundary_equity.csv", "metadata.json",
                "fingerprints.json", "actions.json", "summary.json", "daily_equity.csv", "monthly_returns.csv")
SCOPE = "btc_sac_td3_seed_stability"
FINANCIAL_FIELDS = ("train", "validation", "test", "primary_seed", "capital_per_account",
    "capital_shared_between_pairs", "ensemble", "costs", "environment", "sources", "runtime_packages",
    "baselines", "research_gates_per_pair", "python", "no_production_change", "fresh_holdout",
    "test_year_already_observed", "primary_seed_fixed_before_training", "dataset_sha256", "dataset_manifest_sha256")
RUNTIME_OVERLAY = {"core_symbols":["BTCUSDT"], "core_seeds":list(SEEDS), "model_seeds":list(SEEDS),
    "oracle_symbols":["BTCUSDT"], "oracle_seeds":list(SEEDS),
    "worker_apply_policy":"each spawned training and evaluation worker applies this exact whitelist",
    "financial_math_changed":False,"model_recipes_changed":False,"performance_mode":"native; no performance patches"}
SEED_EXTENSION = {"previous_seeds":list(range(5)),"new_seeds":list(NEW_SEEDS),"full_seeds":list(SEEDS),
    "retained_model_count":10,"new_model_count":50,"primary_analysis_cohort":"new25","no_seed_selection":True}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean_json(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def relative(root, name):
    name = str(name)
    posix = PurePosixPath(name)
    require(name and posix.as_posix() == name and not posix.is_absolute() and ".." not in posix.parts
            and ":" not in name and "\\" not in name, "Canonical relative POSIX evidence path required")
    root = Path(root).absolute()
    path = root.joinpath(*posix.parts)
    require(not any(p.is_symlink() or hasattr(p, "is_junction") and p.is_junction() for p in (path, *path.parents)), "Linked evidence is not allowed")
    path.resolve().relative_to(root.resolve())
    require(path.is_file(), "Missing evidence: " + name)
    return path


def expected_cases(split):
    cases = [(a, seed, c) for a in ALGORITHMS for seed in SEEDS for c in core.COSTS]
    cases.extend((a, None, c) for a in core.BASELINES for c in core.COSTS)
    return {f"{split}__{c}__BTCUSDT__{a}__{'baseline' if seed is None else seed}":
            ("BTCUSDT", a, seed, c, f"evaluation/{split}/BTCUSDT/{a}/{'baseline' if seed is None else 'seed' + str(seed)}/{c}")
            for a, seed, c in cases}


def cohort_statistics(rows):
    """Descriptive training-seed distribution; quantiles use explicit linear interpolation."""
    output = []
    for cohort, seeds in COHORTS.items():
        for algorithm in ALGORITHMS:
            for scenario in core.COSTS:
                group = [r for r in rows if r["algorithm"] == algorithm and r["scenario"] == scenario and r["seed"] in seeds]
                require(len(group) == len(seeds) and {r["seed"] for r in group} == set(seeds), "Every declared seed must enter its distribution")
                returns = np.array([r["net_return"] for r in group], dtype=float)
                drawdowns = np.abs([r["max_drawdown"] for r in group])
                require(np.isfinite(returns).all() and np.isfinite(drawdowns).all(), "Finite distribution inputs required")
                quantiles = np.quantile(returns, [.10, .25, .50, .75, .90], method="linear")
                output.append({"cohort": cohort, "algorithm": algorithm, "scenario": scenario, "seeds": len(seeds),
                    "positive_seeds": int((returns > 0).sum()), "loss_seeds": int((returns < 0).sum()), "zero_seeds": int((returns == 0).sum()),
                    "positive_rate": float((returns > 0).mean()), "loss_rate": float((returns < 0).mean()),
                    "mean_return": float(returns.mean()), "median_return": float(quantiles[2]), "min_return": float(returns.min()), "max_return": float(returns.max()),
                    "p10_return": float(quantiles[0]), "q1_return": float(quantiles[1]), "q3_return": float(quantiles[3]), "p90_return": float(quantiles[4]),
                    "median_drawdown": float(np.quantile(drawdowns, .5, method="linear")),
                    "p90_drawdown": float(np.quantile(drawdowns, .9, method="linear")), "worst_drawdown": float(drawdowns.max()),
                    "drawdown_above_20_seeds": int((drawdowns > .20).sum()), "quantile_method": "linear"})
    return output


def paired_seed_results(rows):
    output = []
    for seed in SEEDS:
        for scenario in core.COSTS:
            group = {r["algorithm"]: r for r in rows if r["algorithm"] in ALGORITHMS and r["scenario"] == scenario and r["seed"] == seed}
            require(set(group) == set(ALGORITHMS), "Both algorithms for an identical seed label are required")
            sac, td3 = group["sac"], group["td3"]
            output.append({"seed": seed, "scenario": scenario, "cohort": "retained5" if seed < 5 else "new25",
                "sac_return": sac["net_return"], "td3_return": td3["net_return"], "return_difference": td3["net_return"] - sac["net_return"],
                "sac_drawdown": abs(sac["max_drawdown"]), "td3_drawdown": abs(td3["max_drawdown"]),
                "drawdown_difference": abs(td3["max_drawdown"]) - abs(sac["max_drawdown"]),
                "td3_higher_return": td3["net_return"] > sac["net_return"],
                "td3_lower_drawdown": abs(td3["max_drawdown"]) < abs(sac["max_drawdown"]),
                "sac_run_id": sac["run_id"], "td3_run_id": td3["run_id"]})
    return output


def paired_statistics(pairs):
    output = []
    for cohort, seeds in COHORTS.items():
        for scenario in core.COSTS:
            group = [r for r in pairs if r["seed"] in seeds and r["scenario"] == scenario]
            require(len(group) == len(seeds), "Complete same-label seed differences required")
            output.append({"cohort": cohort, "scenario": scenario, "seeds": len(seeds),
                "median_return_difference": float(np.median([r["return_difference"] for r in group])),
                "median_drawdown_difference": float(np.median([r["drawdown_difference"] for r in group])),
                "td3_higher_return_seeds": sum(r["td3_higher_return"] for r in group),
                "td3_lower_drawdown_seeds": sum(r["td3_lower_drawdown"] for r in group),
                "td3_both_better_seeds": sum(r["td3_higher_return"] and r["td3_lower_drawdown"] for r in group)})
    return output


def pin_original(root, name, pins, *, parse=True):
    """Bind any original evidence file; parse only explicitly JSON documents."""
    path=relative(root,name)
    pins[str(path)]=sha(path)
    return read(path) if parse else None


def verify_inputs(root, evidence):
    for name, expected in evidence["input_sha256"].items():
        require(sha(relative(root, name)) == expected, "Evidence changed during report preparation: " + name)
    for name, expected in evidence.get("external_input_sha256", {}).items():
        require(sha(Path(name)) == expected, "Original retained evidence/helper changed: " + name)
    for name, expected in evidence.get("source_sha256", {}).items():
        require(sha(relative(ROOT,name))==expected,"Report/readback helper changed: "+name)


def make_reviewed_snapshot(study_root, retained_study_root):
    for value in (study_root,retained_study_root):
        path=Path(value).absolute()
        require(not any(p.is_symlink() or hasattr(p,"is_junction") and p.is_junction() for p in (path,*path.parents)), "Linked study root is not permitted")
    root, retained_root = Path(study_root).resolve(), Path(retained_study_root).resolve()
    require(not root.is_relative_to(retained_root) and not retained_root.is_relative_to(root), "Separate new and original study folders required")
    pins, external = {}, {}
    def take(name):
        path = relative(root, name); pins[name] = sha(path)
        return read(path)
    def original(name, *, parse=True):
        return pin_original(retained_root,name,external,parse=parse)
    spec = take("protocol.json")
    require(take("protocol_receipt.json").get("protocol_sha256") == pins["protocol.json"], "Protocol receipt changed")
    require(spec.get("symbols") == ["BTCUSDT"] and set(spec.get("algorithms", [])) == set(ALGORITHMS)
            and len(spec["algorithms"]) == 2 and spec.get("seeds") == list(SEEDS) and all(type(s) is int for s in spec["seeds"])
            and spec.get("training_model_count") == MODEL_COUNT
            and spec.get("capital_per_account") == 10000. and spec.get("primary_seed") == 0
            and spec.get("primary_seed_fixed_before_training") is True and spec.get("capital_shared_between_pairs") is False
            and spec.get("ensemble") is False and spec.get("fresh_holdout") is False and spec.get("test_year_already_observed") is True
            and spec.get("no_production_change") is True, "Complete 30-seed single-BTC protocol required")
    require(spec.get("train") == list(core.TRAIN) and spec.get("validation") == list(core.VALIDATION) and spec.get("test") == list(core.TEST)
            and spec.get("costs") == core.COSTS and spec.get("baselines") == list(core.BASELINES)
            and spec.get("timesteps_per_model") == {"sac": 32768, "td3": 65536}
            and set(spec.get("sources", {})) == set(core.SOURCE_FILES), "Financial dates, costs, budgets or original sources changed")
    require(spec.get("runtime_overlay")==RUNTIME_OVERLAY and spec.get("seed_extension")==SEED_EXTENSION
            and spec.get("validation_selection",{}).get("mode")=="fixed_algorithm_comparison"
            and spec["validation_selection"].get("algorithm_selection") is False
            and spec["validation_selection"].get("seed_selection") is False,"Only the explicit seed/symbol whitelist may change")
    for name, expected in spec["sources"].items():
        require(sha(relative(ROOT, name)) == sha(relative(root, "frozen_source/" + name)) == expected, "Original financial/model helper changed: " + name)
        pins["frozen_source/" + name] = expected
    source_spec = original("protocol.json")
    require(source_spec.get("symbols")==["BTCUSDT","ETHUSDT"] and source_spec.get("seeds")==list(range(5))
            and set(source_spec.get("algorithms",[]))=={"sac","tqc","crossq","recurrent_ppo","td3","droq"}
            and source_spec.get("training_model_count")==60,"Original source must remain the audited six-algorithm BTC/ETH study")
    require(take("retained_protocol.json")==source_spec and pins["retained_protocol.json"]==external[str(retained_root/"protocol.json")],"Retained protocol snapshot changed")
    require(original("protocol_receipt.json").get("protocol_sha256") == external[str(retained_root / "protocol.json")], "Original source receipt changed")
    invariant = ("train", "validation", "test", "capital_per_account", "capital_shared_between_pairs", "ensemble", "costs", "environment", "sources", "runtime_packages", "baselines", "research_gates_per_pair", "python", "fresh_holdout", "test_year_already_observed", "primary_seed_fixed_before_training", "no_production_change", "dataset_sha256", "dataset_manifest_sha256")
    require(all(source_spec.get(k) == spec.get(k) for k in invariant), "Funding, execution, causal features, runtime or market bytes differ from original study")
    for algorithm in ALGORITHMS:
        require(source_spec["recipes"][algorithm] == spec["recipes"][algorithm]
                and source_spec["timesteps_per_model"][algorithm] == spec["timesteps_per_model"][algorithm], "Per-algorithm recipe/compute budget changed")
    imported = take("training_import.json")
    require(take("training_import_receipt.json").get("training_import_sha256") == pins["training_import.json"]
            and imported.get("source_protocol_sha256") == external[str(retained_root / "protocol.json")]
            and imported.get("destination_protocol_sha256") == pins["protocol.json"]
            and imported.get("retained_model_count") == 10 and imported.get("new_model_count") == 50
            and imported.get("evaluation_results_reused") is False and imported.get("old_test_year_already_observed") is True
            and imported.get("no_production_change") is True, "Retained/new model provenance changed")
    require(imported.get("compatibility_fields")==list(FINANCIAL_FIELDS)
            and all(imported.get(k)==spec.get(k)==source_spec.get(k) for k in ("dataset_sha256","dataset_manifest_sha256"))
            and spec.get("training_window_quality")=={"BTCUSDT":source_spec["training_window_quality"]["BTCUSDT"]},"Exact financial compatibility declarations and eligible BTC windows required")
    source_fit = original("fit_complete.json")
    require(source_fit.get("model_count") == 60 and len(source_fit.get("records", [])) == 60
            and imported.get("source_fit_complete_sha256") == external[str(retained_root / "fit_complete.json")], "Original 60-model fit evidence changed")
    for split in ("validation", "test"):
        old_audit = original(split + "_audit.json")
        require(old_audit.get("passed") is True and old_audit.get("failure_count") == 0 and old_audit.get("accounts_checked") == 136
                and old_audit.get("protocol_sha256") == imported["source_protocol_sha256"]
                and imported.get("source_" + split + "_audit_sha256") == external[str(retained_root / (split + "_audit.json"))], "Original full study audit changed")
    old_ids = {("BTCUSDT", a, s) for a in ALGORITHMS for s in range(5)}
    new_ids = {("BTCUSDT", a, s) for a in ALGORITHMS for s in NEW_SEEDS}
    def identities(rows):
        require(isinstance(rows, list) and all(type(r.get("seed")) is int for r in rows), "Integer seed identities required")
        values = [(r.get("symbol"), r.get("algorithm"), r["seed"]) for r in rows]
        require(len(values) == len(set(values)), "Duplicate imported/fresh model identity")
        return set(values)
    require(identities(imported.get("retained_models")) == old_ids and identities(imported.get("new_models")) == new_ids, "Exact retained0..4 and new5..29 populations required")
    import_rows = []
    old_fit_records = {(r["symbol"], r["algorithm"], r["seed"]): r for r in source_fit["records"]}
    for item in imported["retained_models"]:
        a, s = item["algorithm"], item["seed"]; directory = f"training/BTCUSDT/{a}/seed{s}"
        require(item["source_directory"] == item["destination_directory"] == directory, "Retained directory identity changed")
        old_directory = relative(retained_root, directory + "/record.json").parent
        new_directory = relative(root, directory + "/record.json").parent
        old_files = {p.relative_to(old_directory).as_posix() for p in old_directory.rglob("*") if p.is_file()}
        new_files = {p.relative_to(new_directory).as_posix() for p in new_directory.rglob("*") if p.is_file()}
        require(old_files == new_files == set(item["file_sha256"]), "Complete retained directory inventory required")
        for name, expected in item["file_sha256"].items():
            filename = directory + "/" + name; original(filename,parse=False)
            require(sha(relative(root, filename)) == expected == external[str(retained_root / filename)], "Retained model bytes changed: " + filename)
            pins[filename] = expected
        record = read(new_directory / "record.json")
        require(record == old_fit_records[("BTCUSDT", a, s)] and item["source_record_sha256"] == pins[directory + "/record.json"]
                and item["model_parameter_sha256"] == record["checkpoint"]["parameter_sha256"]
                and item["source_completed_at"] == record["completed_at"], "Retained model metadata/tensors changed")
        import_rows.append({"algorithm": a, "seed": s, "origin": "retained", "preserved_files": len(old_files),
                            "source_record_sha256": item["source_record_sha256"], "source_completed_at": item["source_completed_at"]})
    extension = take("extension_manifest.json")
    require(extension.get("core_protocol_sha256") == pins["protocol.json"] and extension.get("training_import_sha256") == pins["training_import.json"]
            and extension.get("retained_models") == 10 and extension.get("fresh_models") == 50 and extension.get("models") == 60
            and extension.get("accounts_per_phase") == ACCOUNT_COUNT and extension.get("repeat_all_cases") is True
            and extension.get("both_costs_each_model") is True and extension.get("runtime_overlay")==RUNTIME_OVERLAY
            and set(extension.get("extension_sources",{}))=={"scripts/backtest_btc_rl_seed_stability.py","scripts/audit_btc_rl_seed_stability.py"}, "Extension manifest population/bindings changed")
    for name_key, hash_key in (("driver_relative_path", "driver_source_sha256"), ("audit_entry_relative_path", "audit_entry_source_sha256")):
        name = extension[name_key]; path = relative(root, name); pins[name] = sha(path)
        require(pins[name] == extension[hash_key] and name.startswith("extension_source/scripts/")
                and sha(relative(ROOT, name.replace("extension_source/", "", 1))) == pins[name]
                and extension["extension_sources"].get(name.replace("extension_source/","",1))==pins[name], "Extension executed helper changed")
    require(imported.get("import_source_sha256") == extension["driver_source_sha256"], "Import helper changed")
    training, original_records = [], []
    for a in ALGORITHMS:
        for seed in SEEDS:
            record = core.validate_training_record(root, "BTCUSDT", a, seed, spec)
            require(type(record.get("seed")) is int,"Integer training-record seed required")
            verify_native_training_counters(record, spec)
            name = f"training/BTCUSDT/{a}/seed{seed}/record.json"; take(name)
            checkpoint=relative(root,name).parent/"checkpoint"
            require({p.relative_to(relative(root,name).parent).as_posix() for p in checkpoint.rglob("*") if p.is_file()}==set(record["files"]),"Exact checkpoint payload inventory required")
            if seed >= 5:
                require(pd.Timestamp(record["started_at"]) >= pd.Timestamp(spec["frozen_at"]), "New training predates the extension freeze")
            for filename, expected in record["files"].items():
                path_name = name.rsplit("/", 1)[0] + "/" + filename
                require(sha(relative(root, path_name)) == expected, "Changed checkpoint payload")
                pins[path_name] = expected
            original_records.append(record)
            training.append({"algorithm": a, "seed": seed, "origin": "retained" if seed < 5 else "new",
                "actual_steps": record["training"]["actual_steps"], "maximum_input_hour": record["maximum_input_hour"],
                "record_sha256": pins[name], "record_file": name,
                "native_counters_json": json.dumps(record["training"]["counters"], sort_keys=True)})
    require({p.relative_to(root).as_posix() for p in (root / "training").rglob("record.json")} == {r["record_file"] for r in training}, "Unexpected/incomplete training population")
    fit = take("fit_complete.json"); sort_key = lambda r: (r["algorithm"], r["seed"])
    require(fit.get("model_count") == 60 and sorted(fit["records"], key=sort_key) == sorted(original_records, key=sort_key), "Fit completion does not match every exact record")
    counter = take("training_counter_audit.json")
    require(counter.get("passed") is True and counter.get("failure_count") == 0 and not counter.get("failures")
            and counter.get("models_checked") == 60 and counter.get("retained_models") == 10 and counter.get("new_models") == 50
            and counter.get("core_protocol_sha256") == pins["protocol.json"]
            and counter.get("extension_source_sha256") == extension["driver_source_sha256"], "Complete passed native counter audit required")
    observed = {(r["symbol"], r["algorithm"], r["seed"]): r for r in counter.get("models", [])}
    require(len(counter.get("models", [])) == 60 and set(observed) == old_ids | new_ids, "Full native counter audit identities required")
    for record in original_records:
        item = observed[(record["symbol"], record["algorithm"], record["seed"])]
        require(item["record_sha256"] == pins[f"training/BTCUSDT/{record['algorithm']}/seed{record['seed']}/record.json"]
                and item["counters"] == record["training"]["counters"] and item["retained"] is (record["seed"] < 5), "Counter audit record/counter bytes changed")
        if record["seed"]>=5:
            name=f"training/BTCUSDT/{record['algorithm']}/seed{record['seed']}/runtime_overlay.json";applied=take(name)
            require(item.get("runtime_overlay_sha256")==pins[name] and applied.get("runtime_overlay")==RUNTIME_OVERLAY
                    and applied.get("actual_core_seeds")==applied.get("actual_model_seeds")==list(SEEDS)
                    and applied.get("actual_core_symbols")==["BTCUSDT"] and applied.get("driver_source_sha256")==extension["driver_source_sha256"]
                    and applied.get("record_sha256")==item["record_sha256"]
                    and all(applied.get(k)==record[k] for k in ("symbol","algorithm","seed")),"Actual fresh worker whitelist/source proof changed")
    summaries, daily_rows, monthly_rows, audits = {}, [], [], []
    for split in ("validation", "test"):
        audit = take(split + "_audit.json"); expected = expected_cases(split)
        require(audit.get("passed") is True and audit.get("phase") == split and audit.get("failure_count") == 0 and not audit.get("failures")
                and audit.get("checks_total", 0) > 0 and all(audit.get(k) == ACCOUNT_COUNT for k in ("expected_accounts", "accounts_checked", "repeats_checked", "accounts_count"))
                and audit.get("protocol_sha256") == pins["protocol.json"]
                and audit.get("summary_sha256") == sha(relative(root, split + "_summary.csv")), "Complete bound independent audit required: " + split)
        contract=take(split+"_seed_contract_audit.json")
        require(contract.get("passed") is True and contract.get("phase")==split and contract.get("failure_count")==0 and not contract.get("failures")
                and contract.get("checks_total",0)>0 and contract.get("core_financial_audit_passed") is True
                and contract.get("core_financial_audit_sha256")==pins[split+"_audit.json"]
                and contract.get("wrapper_source_sha256")==extension["audit_entry_source_sha256"]
                and contract.get("runtime_overlay")=={"oracle.SYMBOLS":["BTCUSDT"],"oracle.SEEDS":list(SEEDS)}
                and contract.get("independent_math_unchanged") is True and contract.get("no_environment_or_model_imports") is True,
                "Independent BTC/expanded-seed contract audit must pass: "+split)
        contract_inputs=contract.get("inputs_sha256",{})
        metadata_names={identity[-1].replace("evaluation/",kind+"/",1)+"/metadata.json" for identity in expected.values() for kind in ("evaluation","repeat")}
        require(metadata_names|{"protocol.json","retained_protocol.json","training_import.json","extension_manifest.json",split+"_audit.json"}<=set(contract_inputs),"Full BTC feature/ownership metadata contract bindings required")
        for name,expected_sha in contract_inputs.items():
            require(sha(relative(root,name))==expected_sha,"Changed seed-contract audit input: "+name);pins[name]=expected_sha
        require(set(audit.get("inputs_sha256", {})) == {"protocol.json", split + "_summary.csv", split + "_complete.json"}, "Complete phase audit inputs required")
        for name, expected_sha in audit["inputs_sha256"].items():
            require(sha(relative(root, name)) == expected_sha, "Changed audited phase input")
            pins[name] = expected_sha
        frame = csv(relative(root, split + "_summary.csv"))
        require(len(frame) == ACCOUNT_COUNT and not frame.run_id.duplicated().any() and set(frame.run_id) == set(expected), "Full 128 distinct accounts required")
        for kind in ("evaluation","repeat"):
            observed_paths={p.parent.relative_to(root).as_posix() for p in (root/kind/split).rglob("summary.json")}
            require(observed_paths=={r[-1].replace("evaluation/",kind+"/",1) for r in expected.values()},"Unexpected/missing serialized account folders")
        cases = {r["relative_path"]: r for r in audit["cases"]}
        require(len(audit["cases"]) == ACCOUNT_COUNT and set(cases) == {r[-1] for r in expected.values()}, "Full oracle case set required")
        completion = take(split + "_complete.json")
        require(completion.get("accounts") == completion.get("repeats") == ACCOUNT_COUNT and completion.get("repeat_equal") is True
                and len(completion["results"]) == ACCOUNT_COUNT, "Complete repeated execution required")
        completed = {r["summary"]["run_id"]: r for r in completion["results"]}
        require(set(completed) == set(expected), "Execution case identities changed")
        rows = []
        for aggregate in frame.to_dict("records"):
            identity = expected[aggregate["run_id"]]; case = cases[identity[-1]]
            for name in case["inputs_sha256"]: relative(root, name)
            row, days, months = verify_case(root, split, identity, aggregate, case)
            require(row["seed"] is None or type(row["seed"]) is int,"Exact integer account seed required")
            item = completed[row["run_id"]]
            require(item["summary"] == row and item["path"] == identity[-1] and item["repeat_path"] == identity[-1].replace("evaluation/", "repeat/", 1)
                    and item["repeat_equal"] is True, "Canonical/repeated execution receipt changed")
            pins.update(case["inputs_sha256"]); rows.append(row)
            if split == "test": daily_rows.extend(days); monthly_rows.extend(months)
            if row["seed"] is None or row["seed"] < 5:
                for kind in ("evaluation", "repeat"):
                    directory = identity[-1].replace("evaluation/", kind + "/", 1)
                    for filename in RESULT_FILES:
                        name = directory + "/" + filename; original(name,parse=False)
                        require(external[str(retained_root / name)] == pins[name], "Unchanged original policy/baseline replay differs: " + name)
        summaries[split] = rows
        audits.append({k: audit[k] for k in ("phase", "passed", "checks_total", "failure_count", "accounts_checked", "repeats_checked", "audited_at", "independent_oracle")})
    selection = take("validation_selection.json"); started = take("test_started.json")
    require(selection.get("uses_test") is False and selection.get("seed_selection") is False
            and selection.get("algorithm_selection") is False and selection.get("mode")=="fixed_algorithm_comparison"
            and set(selection.get("fixed_algorithms",[]))==set(ALGORITHMS)
            and selection.get("protocol_sha256") == pins["protocol.json"]
            and selection.get("validation_audit_sha256") == pins["validation_audit.json"]
            and selection.get("validation_summary_sha256") == pins["validation_summary.csv"]
            and selection.get("validation_seed_contract_audit_sha256")==pins["validation_seed_contract_audit.json"], "Validation-only comparison receipt changed")
    require(started.get("pins") == {"selection_sha256": pins["validation_selection.json"], "validation_audit_sha256": pins["validation_audit.json"],
            "protocol_sha256": pins["protocol.json"], "dataset_sha256": spec["dataset_sha256"]}, "Annual source/validation pins changed")
    require(pd.Timestamp(counter["audited_at"]) >= pd.Timestamp(fit["at"])
            and pd.Timestamp(selection["at"]) >= pd.Timestamp(audits[0]["audited_at"])
            and pd.Timestamp(started["at"]) >= pd.Timestamp(selection["at"])
            and pd.Timestamp(audits[1]["audited_at"]) >= pd.Timestamp(read(root / "test_complete.json")["at"]), "Training, validation and annual chronology changed")
    stats = cohort_statistics(summaries["test"]); pairs = paired_seed_results(summaries["test"]); pair_stats = paired_statistics(pairs)
    queries = {}
    common = {"label": "BTC独立现货账户种子复现", "classification": "modeled", "timezone": "UTC", "period": {"start": core.TEST[0], "end": core.TEST[1]},
        "caveats": ["年度行情此前已被观察；新增种子是训练随机性复现，不是新的市场盲测。", "种子不是独立市场样本；分位数、胜率与同编号差值仅为描述统计。", "TD3每模型65536环境步，SAC32768步，保留原配方，计算预算不等。", "主成本单边0.10%手续费+0.02%不利滑点；压力成本0.20%+0.05%。", "所有账户独立10000USDT，只持BTC与自身USDT，不平均模型权重或账户收益。"],
        "metricDefinitions": [{"label": "扣费收益", "definition": "期末实际现金加BTC市值/10000-1；模拟手续费与滑点已扣，期末真实持仓不强平。"},
            {"label": "最大回撤", "definition": "完整8760小时收盘权益相对含初始10000的历史峰值最大跌幅；表与散点显示幅度绝对值。"},
            {"label": "新增25/全部30", "definition": "新增只含预先固定seed5..29；全部为seed0..29，原seed0..4逐字节保留。"},
            {"label": "分位数", "definition": "NumPy quantile(method=linear)，对各个独立账户的年度终值/回撤求分位；不是可交易组合曲线。"}],
        "evidenceFlow": [{"title": "配方保持，扩展种子白名单", "detail": "原11源码与资金、数据、日期、成本和算法配方不变；独立driver/审计入口冻结并记录显式seed0..29和BTC白名单。"},
            {"title": "训练与账本实际核验", "detail": "60模型的原生更新数和真实32天片段全部核对；两阶段128账户及重复执行通过独立金融oracle。"},
            {"title": "统计覆盖全部预先固定种子", "detail": "没有按年度收益丢弃种子；新25与含旧5的全部30分别展示，逐账户小时、日/月数据读回重算。"}]}
    anchor_names = ["protocol.json", "protocol_receipt.json", "training_import.json", "training_import_receipt.json", "extension_manifest.json", "training_counter_audit.json", "validation_selection.json", "validation_audit.json", "test_audit.json","validation_seed_contract_audit.json","test_seed_contract_audit.json"]
    def query(key, rows, files, method, bounds=core.TEST):
        source = deepcopy(common); names = list(dict.fromkeys([*files, *anchor_names]))
        source.update(files=names, fileHashes={n: pins[n] for n in names}, executedAt=max(a["audited_at"] for a in audits), period={"start": bounds[0], "end": bounds[1]})
        queries[key] = {"rows": clean_json(rows), "source": source, "methods": [{"language": "text", "code": method}]}
    query("account_summary", summaries["test"], ["test_summary.csv"], "128 actual accounts: 60 learned policies at two costs, plus four fixed baselines at two costs; no pooled capital.")
    query("equity_history", daily_rows, [n for n in pins if n.startswith("evaluation/test/") and n.endswith("daily_equity.csv")], "366 actual boundary points per account, initial10000 included; no median/mean trading curve.")
    query("monthly_returns", monthly_rows, [n for n in pins if n.startswith("evaluation/test/") and n.endswith("monthly_returns.csv")], "13 calendar fragments per account, compounded from actual account boundaries.")
    query("cohort_statistics", stats, ["test_summary.csv"], "Full new25 seeds5..29 and all30 seeds0..29; exact linear quantiles, profit/loss/zero denominators, drawdown magnitudes; descriptive only.")
    query("paired_seed_results", pairs, ["test_summary.csv"], "Same-label seed TD3 minus SAC, independently funded policies; descriptive differences, not causal pairs or market-independent replications.")
    query("paired_statistics", pair_stats, ["test_summary.csv"], "Median per-seed TD3-SAC differences and exact counts of higher return/lower drawdown, separately for each cost and seed cohort.")
    query("validation_summary", summaries["validation"], ["validation_summary.csv"], "128 actual validation accounts; both algorithms remain in annual comparison regardless of score.", core.VALIDATION)
    query("training_records", training, ["fit_complete.json", *[r["record_file"] for r in training]], "10 unchanged source models plus50 newly trained; native budgets/optimizer/replay/32-day timeout histories checked.", core.TRAIN)
    queries["training_records"]["payloadColumns"] = ["native_counters_json"]
    query("training_import", import_rows, ["training_import.json", "training_import_receipt.json"], "Every record/progress/checkpoint in retained10 model directories is byte-identical to the original six-algorithm study.", core.TRAIN)
    query("audit_evidence", audits, ["validation_audit.json", "test_audit.json"], "Passed independent oracle for both128-account phases and all repeats; original28 policy/baseline accounts per phase and repetition reproduce all1120 serialized files exactly.")
    query("protocol", [{"capital_per_account":10000., "training_models":60, "accounts_per_phase":128, "seeds_per_algorithm":30,
          "retained_models":10, "fresh_models":50, "primary_seed_reference":0, "fresh_holdout":False, "dataset_sha256":spec["dataset_sha256"]}], ["protocol.json"], "The original reference seed0 is preserved, not replaced by the best expanded-study seed.")
    snapshot = {"surface":"report", "status":"reviewed", "buildStatus":"complete", "title":"BTC 的 TD3 与 SAC：扩展种子后的收益、回撤和稳定性", "generatedAt":datetime.now(timezone.utc).isoformat(),
        "filters":[], "queries":queries, "report":{"asOf":"2026-10-22", "scope":SCOPE, "studyConfig":spec, "trainingCounterAudit":counter,
        "auditStatus":"passed", "noEnsemble":True, "noPooledCapital":True, "defaultCohort":"new25", "selectionRole":"comparison_only_no_winner"}}
    evidence = {"input_sha256":pins, "external_input_sha256":external, "accounts_per_phase":128, "training_models":60, "retained_models":10, "new_models":50,
        "numeric_results_changed":False, "pickle_loaded":False, "policy_constructed":False,
        "source_sha256":{n:sha(ROOT/n) for n in ["scripts/btc_rl_seed_stability_report_app.py","scripts/single_pair_rl_six_report_app.py","scripts/btceth_rl_report_app.py"]}}
    require(len(daily_rows)==128*366 and len(monthly_rows)==128*13 and len(stats)==8 and len(pairs)==60 and len(pair_stats)==4, "Complete report query counts required")
    verify_inputs(root, evidence)
    return snapshot, evidence



def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root",required=True,type=Path)
    parser.add_argument("--retained-study-root",required=True,type=Path)
    args=parser.parse_args()
    snapshot,evidence=make_reviewed_snapshot(args.study_root,args.retained_study_root)
    print(json.dumps({"passed":True,"query_counts":{k:len(v["rows"]) for k,v in snapshot["queries"].items()},"input_files":len(evidence["input_sha256"]),"no_html_written":True}))


if __name__=="__main__":main()

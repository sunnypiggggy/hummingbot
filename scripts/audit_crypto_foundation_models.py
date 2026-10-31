"""Independent causal-cache and funded-portfolio audit; never runs a model."""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path, PurePosixPath
import pickle
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.foundation_model_audit_helpers import Checks, read, target_weights, audit_case_economics
from scripts.audit_crypto_rl_top10 import sha, stamp, announced_exit
from scripts.foundation_model_proof_paths import ProofPaths

OUTPUT = ROOT / "results/backtests/crypto_foundation_models_20261024"
DATASETS = {"original": ROOT / "data/crypto_rl_top10/dataset.pkl.gz",
            "ex_ada": ROOT / "data/crypto_rl_top10_ex_ada/dataset.pkl.gz"}
BASELINES = ("cash", "top10_equal", "top10_cap", "btc_eth_hold")


def canonical_context(frame):
    raw = frame.to_csv(index=True, index_label="timestamp", float_format="%.17g", lineterminator="\n").encode()
    return hashlib.sha256(raw).hexdigest()


def same_csv_records(left, right):
    """Exact positional CSV values, independent of per-file numeric inference.

    A daily CSV can infer int64 where the full-year CSV infers float64 because
    another day contains decimals. Convert each Series to Python scalars rather
    than casting the whole frame to float: Python's integer/float comparison
    still rejects e.g.9007199254740993 versus9007199254740992.0. No tolerance is
    applied, and a missing value matches only a missing value in the same cell.
    """
    if left.shape != right.shape or not left.columns.equals(right.columns):
        return False
    for column in left:
        for actual, expected in zip(left[column].tolist(), right[column].tolist()):
            actual_missing, expected_missing = bool(pd.isna(actual)), bool(pd.isna(expected))
            if actual_missing or expected_missing:
                if actual_missing != expected_missing:
                    return False
                continue
            if isinstance(actual, bool) != isinstance(expected, bool) or actual != expected:
                return False
    return True


def independent_plan(market, datasets, start, end):
    """Each day's input contains only that day's previous 128 complete days."""
    pools = {variant: {date: rows.sort_values("slot") for date, rows in data["universe"].groupby("date")}
             for variant, data in datasets.items()}
    plan, batches = {}, {}
    for date in pd.date_range(start, end, freq="D", inclusive="left"):
        union = sorted(set(pools["original"][date].asset_key.astype(str)) |
                       set(pools["ex_ada"][date].asset_key.astype(str)), key=int)
        available = []
        before = pd.date_range(date - pd.Timedelta(days=128), date, freq="D", inclusive="left")
        for key in union:
            frame = pd.DataFrame({field: market["daily"][field].reindex(before)[key]
                                  for field in ("open", "high", "low", "close", "volume", "amount")})
            frame.index.name = "timestamp"
            good = len(frame) == 128 and np.isfinite(frame.to_numpy()).all()
            good &= not (frame[["open", "high", "low", "close"]] <= 0).any().any()
            good &= not (frame[["volume", "amount"]] < 0).any().any()
            good &= frame.volume.iloc[-30:].mean() > 0
            reason = "known_delisting" if announced_exit(datasets["original"]["assets"], key, date) else ""
            if not good:
                reason = "insufficient_complete_128_day_context"
            if good:
                closes = frame.close.to_numpy(float)
                vol = float(np.std(closes[-30:] / closes[-31:-1] - 1, ddof=1))
                digest, base = canonical_context(frame), float(closes[-1])
            else:
                digest, base, vol = None, None, None
            plan[(date, key)] = {"context_ready": not bool(reason), "exclusion_reason": reason,
                                 "input_start": before[0], "input_end_exclusive": date,
                                 "input_rows": 128 if good else 0, "input_sha256": digest,
                                 "base_close": base, "volatility_30d": vol}
            if not reason:
                available.append(key)
        batches[date] = available
    return plan, batches


def derive_seed(model_key, seed, date, assets):
    value = f"{model_key}|{seed}|{date.isoformat()}|" + ",".join(assets)
    return int.from_bytes(hashlib.sha256(value.encode()).digest()[:4], "big") % (2**31)


def assert_preserved(review, output, phase, proof_paths=None):
    path = (proof_paths.local_evidence("original_research_preservation.json") if proof_paths is not None
            else output / "original_research_preservation.json")
    protection = read(path)
    for relative, record in protection["files"].items():
        path = ROOT / relative.replace("\\", "/")
        review.check(path.is_file() and path.stat().st_size == record["size_bytes"] and sha(path) == record["sha256"],
                     phase + ": prior immutable research " + relative)
    return len(protection["files"])


def audit_gpu_backend(review, output, spec, backend, proof_paths):
    """Inspect recorded native CUDA checks without importing Torch or models."""
    control = read(proof_paths.local_evidence("remote_execution_evidence/migration_control.json"))
    cpu_path = proof_paths.local_evidence("evidence/cpu/frozen_spec.json")
    cpu = read(cpu_path)
    review.check(sha(cpu_path) == control["parent_cpu_protocol_sha256"]
                 == spec["migration"]["parent_cpu_protocol_sha256"], "GPU execution traces the unchanged parent CPU protocol")
    allowed = {"frozen_at", "models", "pinned_files", "model_workers", "backend_evidence_path",
               "backend_evidence_sha256", "model_provenance_path", "reproduction_argv", "chronos_loader_adaptation"}
    review.check(all(spec.get(key) == value for key, value in cpu.items() if key not in allowed),
                 "GPU migration changed execution only, preserving all financial/signal settings")
    from collections import Counter
    review.check(Counter(item["sha256"] for item in cpu["pinned_files"])
                 == Counter(item["sha256"] for item in spec["pinned_files"]),
                 "GPU and parent CPU protocols pin identical official source/weight bytes")
    for key, original in cpu["models"].items():
        moved = spec["models"][key]
        review.check(moved["device"] == "cuda:0" and all(
            moved.get(field) == value for field, value in original.items()
            if field not in {"vendor_path", "weights_path", "tokenizer_path", "device"}),
            key + ": GPU native sampling and availability are unchanged")
    devices, checked = backend["gpu_probe"], backend["synthetic_checks"]
    expected_parameters = control["expected_parameter_sha256_by_model"]
    uuids = {item["uuid"] for item in devices}
    review.check(len(devices) == len(uuids) == 2
                 and {item["physical_index"] for item in devices} == set(control["gpu_indices"])
                 and all("V100" in item["name"] for item in devices), "two independently identified physical V100 GPUs")
    review.check(backend.get("no_cpu_fallback") is True and backend.get("new_models_trained") == 0
                 and backend.get("financial_evaluation_performed") is False
                 and backend.get("cpu_gpu_forecast_bitwise_equality_claimed") is False,
                 "GPU native forecasts only, no training/fallback or CPU/GPU forecast equality claim")
    review.check(len(checked) == 2 and {item["binding"]["uuid"] for item in checked} == uuids,
                 "both physical GPU workers passed native synthetic checks")
    for worker in checked:
        binding, models = worker["binding"], worker["models"]
        review.check(binding["uuid"] in uuids and binding["visible_devices"] == binding["uuid"]
                     and binding["torch_current_device"] == 0 and binding["logical_device"] == "cuda:0"
                     and binding["compute_capability"] == [7, 0],
                     binding["uuid"] + ": isolated physical UUID mapped to local CUDA0 with V100 sm70")
        review.check(len(models) == len(spec["models"])
                     and {item["model_key"] for item in models} == set(spec["models"]),
                     binding["uuid"] + ": all three actual native models checked")
        for row in models:
            review.check(row["status"] == "passed" and row["synthetic_only"] is True
                         and row["same_gpu_repeat_equal"] is True and row["input_unchanged"] is True
                         and row["batch"] == 10 and row["context_days"] == 128 and row["horizon_days"] == 1
                         and row["parameter_sha256"] == expected_parameters[row["model_key"]],
                         binding["uuid"] + "/" + row["model_key"] + ": real native CUDA inference repeat/input/parameter checks")


def audit_experiment_contract(review, output, spec, proof_paths=None):
    """Check source evidence and the predeclared rule without model imports."""
    proof_paths = ProofPaths(output, spec) if proof_paths is None else proof_paths
    provenance_path = proof_paths.resolve(spec["model_provenance_path"])
    review.check(sha(provenance_path) == spec["model_provenance_sha256"],
                 "frozen evaluation provenance unchanged")
    provenance = read(provenance_path)
    rows = {row["model_key"]: row for row in provenance["rows"]}
    review.check(len(provenance["rows"]) == len(rows) == 3 and set(rows) == set(spec["models"]),
                 "one source-backed provenance record per real pretrained model")
    for key, config in spec["models"].items():
        row = rows[key]
        review.check(row["local_fit"] is False and row["context_days"] == spec["context_days"] == 128
                     and row["horizon_days"] == spec["horizon_days"] == 1,
                     key + ": provenance agrees with inference-only input contract")
        review.check(stamp(row["first_decision_date"]) == stamp(config["first_decision_date"])
                     and stamp(row["first_usable_decision_utc"]) == stamp(config["first_decision_date"])
                     and stamp(row["first_verified_public_notice_utc"]) < stamp(config["first_decision_date"]),
                     key + ": conservative daily availability matches original public-notice evidence")
        # The GPU spec retains Linux paths; exact local weight aliases are
        # independently pinned, while provenance remains its original bytes.
        configured_weight = str(config["weights_path"]).rstrip("/\\") + "/model.safetensors"
        actual_weight = proof_paths.resolve(configured_weight, row["weights_sha256"])
        review.check(row["source_commit"] == config["source_commit"]
                     and row["weights_revision"] == config["weight_revision"]
                     and (Path(row["weights_path"]) / "model.safetensors").resolve() == actual_weight,
                     key + ": no checkpoint/source substitution in provenance")
    precommit = read(proof_paths.local_evidence("policy_precommit.json"))
    review.check(precommit["context_days"] == 128 and precommit["horizon_days"] == 1
                 and precommit["hurdle"] == spec["forecast_to_weights"]["hurdle"] == .0024
                 and precommit["volatility_floor"] == spec["forecast_to_weights"]["volatility_floor"] == .01
                 and precommit["cash_floor"] == spec["cash_target_floor"] == .05
                 and precommit["coin_cap"] == spec["single_coin_target_cap"] == .25,
                 "score, context and caps match policy declared before any real-market forecasts")
    review.check(precommit["sampling_seeds"] == spec["models"]["kronos_small"]["sampling_seeds"]
                 == spec["models"]["kronos_base"]["sampling_seeds"] == list(range(5))
                 and spec["models"]["chronos2"]["sampling_seeds"] == []
                 and precommit["chronos_quantile"] == spec["models"]["chronos2"]["sampling"]["quantile"] == .5,
                 "five genuine Kronos paths and one deterministic native Chronos median")
    review.check(precommit["policy_module_sha256"] == sha(ROOT / "scripts/crypto_foundation_portfolio.py")
                 and precommit["request_plan_sha256"] == sha(output / "forecast_request_plan.csv")
                 and precommit["real_forecasts_already_generated"] is False
                 and precommit["parameter_search"] is False and spec["parameter_search"] is False,
                 "unchanged causal input plan and portfolio rule committed before inference")
    backend_path = proof_paths.resolve(spec["backend_evidence_path"])
    backend = read(backend_path)
    review.check(sha(backend_path) == spec["backend_evidence_sha256"]
                 and backend.get("status") == "passed", "verified frozen inference backend")
    if proof_paths.imported is not None:
        audit_gpu_backend(review, output, spec, backend, proof_paths)
    elif spec["models"]["chronos2"].get("synchronous_chronos_loader"):
        synchronous = read(proof_paths.local_evidence("sources/chronos_synchronous_loader_synthetic.json"))
        review.check(synchronous["status"] == "passed" and synchronous["synthetic_only"] is True
                     and synchronous["all_native_quantiles_bitwise_equal"] is True
                     and synchronous["native_median_wrapper_equal"] is True
                     and synchronous["official_loader_restored"] is True
                     and synchronous["official_source_modified"] is False
                     and synchronous["adapter_source_sha256"] == sha(ROOT / "scripts/foundation_model_native.py"),
                     "Windows synchronous loading preserves every native quantile and official source")


def audit_gpu_returned_receipt(review, output, spec, proof_paths):
    """Independently trace each returned shard to its recorded physical GPU."""
    control = read(proof_paths.local_evidence("remote_execution_evidence/migration_control.json"))
    completion = read(proof_paths.local_evidence("forecast_completion.json"))
    backend = read(proof_paths.resolve(spec["backend_evidence_path"]))
    review.check(completion["status"] == "complete" and completion["protocol_sha256"] == sha(output / "frozen_spec.json")
                 and completion["parent_cpu_protocol_sha256"] == control["parent_cpu_protocol_sha256"]
                 and completion["new_models_trained"] == 0 and completion["financial_evaluations"] == 0
                 and completion["cpu_forecasts_reused"] is False,
                 "complete newly generated GPU forecasts, no reused partial CPU cache or remote PNL")
    devices, expected_jobs = backend["gpu_probe"], {}
    for key, config in spec["models"].items():
        count = spec["date_shards"][key]
        for seed in config["sampling_seeds"] or [None]:
            for index in range(count):
                expected_jobs[(key, seed, index, count)] = devices[len(expected_jobs) % 2]["uuid"]
    identity = lambda item: (item["model_key"], item["seed"], item["shard_index"], item["shard_count"])
    review.check(len(completion["shards"]) == len(expected_jobs)
                 and {identity(item) for item in completion["shards"]} == set(expected_jobs),
                 "all38 unique native GPU date shards completed")
    remote_experiment = PurePosixPath(control["remote_root"]) / "experiment"

    def returned_file(remote):
        path = PurePosixPath(remote)
        if not path.is_absolute() or ".." in path.parts:
            raise ValueError("Returned GPU artifact path must be absolute without parent traversal")
        return proof_paths.local_evidence(path.relative_to(remote_experiment).as_posix())

    expected_parameters = control["expected_parameter_sha256_by_model"]
    for item in completion["shards"]:
        binding = item["gpu_worker_binding"]
        review.check(binding["uuid"] == expected_jobs[identity(item)] and binding["visible_devices"] == binding["uuid"]
                     and binding["torch_current_device"] == 0 and binding["logical_device"] == "cuda:0",
                     str(identity(item)) + ": actual shard bound to its assigned isolated physical V100")
        path = returned_file(item["path"])
        review.check(sha(path) == item["sha256"], str(identity(item)) + ": unchanged returned native shard evidence")
        shard = read(path)
        review.check(identity(shard) == identity(item) and shard["config"] == spec["models"][item["model_key"]]
                     and shard["parameter_sha256_before"] == shard["parameter_sha256_after"]
                     == expected_parameters[item["model_key"]], str(identity(item)) + ": correct real pretrained GPU tensors")
        for day in shard["day_files"]:
            review.check(sha(returned_file(day["path"])) == day["sha256"],
                         str(identity(item)) + "/" + day["date"] + ": unchanged actual GPU daily forecast bytes")
    models = {item["model_key"]: item for item in completion["models"]}
    review.check(len(completion["models"]) == len(models) == 3 and set(models) == set(spec["models"]),
                 "all three native GPU model manifests returned")
    for key, item in models.items():
        original_path = proof_paths.local_evidence("remote_execution_evidence/forecasts/" + key + "/manifest.json")
        review.check(sha(original_path) == item["manifest_sha256"], key + ": original remote model manifest retained byte-for-byte")
        original, local = read(original_path), read(output / "forecasts" / key / "manifest.json")
        relocation = local.pop("artifact_relocation")
        review.check(relocation["original_remote_manifest_sha256"] == item["manifest_sha256"]
                     and relocation["changed_fields"] == ["forecast_files[].path"]
                     and relocation["prediction_values_changed"] is False
                     and relocation["gpu_protocol_and_cache_contracts_changed"] is False,
                     key + ": manifest relocation is explicit and changes paths only")
        for entry in local["forecast_files"]:
            remote = entry.pop("original_remote_path")
            review.check(Path(entry["path"]).resolve() == returned_file(remote),
                         key + ": local forecast file aliases the exact returned remote artifact")
            entry["path"] = remote
        review.check(local == original, key + ": all original native manifest evidence preserved after alias reversal")


def audit_plan(review, output, expected):
    rows = pd.read_csv(output / "forecast_request_plan.csv", dtype={"asset_key": str}, float_precision="round_trip")
    rows["date"] = pd.to_datetime(rows.date, utc=True)
    review.check(not rows.duplicated(["date", "asset_key"]).any()
                 and set(zip(rows.date, rows.asset_key)) == set(expected), "complete unique same-day union plan")
    for row in rows.itertuples(index=False):
        item = expected[(row.date, row.asset_key)]
        label = f"{row.date.date()}/{row.asset_key}"
        review.check(bool(row.context_ready) == item["context_ready"]
                     and ("" if pd.isna(row.exclusion_reason) else row.exclusion_reason) == item["exclusion_reason"],
                     label + ": declared eligible/missing context")
        review.check(stamp(row.input_start) == item["input_start"] and stamp(row.input_end_exclusive) == row.date
                     and row.input_rows == item["input_rows"], label + ": exact strictly previous 128 dates")
        if item["input_sha256"] is not None:
            review.check(row.input_sha256 == item["input_sha256"], label + ": actual six-channel input digest")
            review.check(row.base_close == item["base_close"]
                         and abs(row.volatility_30d - item["volatility_30d"]) < 1e-12, label + ": causal price/risk")


def audit_caches(review, output, spec, frozen, plan, batches):
    all_caches, summaries = {}, {}
    for model_key, config in spec["models"].items():
        manifest = read(output / "forecasts" / model_key / "manifest.json")
        review.check(manifest.get("local_fit") is False and manifest.get("inference_only") is True,
                     model_key + ": inference only")
        review.check(manifest["config"] == config
                     and manifest["parameter_sha256_before"] == manifest["parameter_sha256_after"],
                     model_key + ": fixed pretrained tensors and model configuration")
        seeds = config["sampling_seeds"] or [None]
        review.check([item["seed"] for item in manifest["forecast_files"]] == seeds,
                     model_key + ": real sampling paths, no invented deterministic seeds")
        available = stamp(config["first_decision_date"])
        expected_pairs = {key for key, item in plan.items() if key[0] >= available and item["context_ready"]}
        caches = {}
        for entry in manifest["forecast_files"]:
            seed, path = entry["seed"], Path(entry["path"])
            review.check(sha(path) == entry["sha256"], model_key + "/" + str(seed) + ": frozen forecast bytes")
            frame = pd.read_csv(path, dtype={"asset_key": str}, float_precision="round_trip")
            frame["date"] = pd.to_datetime(frame.date, utc=True)
            review.check(not frame.duplicated(["date", "asset_key"]).any()
                         and set(zip(frame.date, frame.asset_key)) == expected_pairs and len(frame) == entry["rows"],
                         model_key + "/" + str(seed) + ": complete publicly available request coverage")
            cache = {}
            for row in frame.itertuples(index=False):
                key = row.date, row.asset_key
                item, batch = plan[key], batches[row.date]
                label = f"{model_key}/{seed}/{row.date.date()}/{row.asset_key}"
                review.check(row.model_key == model_key and ((seed is None and pd.isna(row.seed)) or row.seed == seed),
                             label + ": correct model/path")
                review.check(row.input_sha256 == item["input_sha256"] and row.input_rows == 128
                             and stamp(row.input_start) == row.date - pd.Timedelta(days=128)
                             and stamp(row.input_end_exclusive) == row.date, label + ": real causal context")
                review.check(row.base_close == item["base_close"]
                             and abs(row.volatility_30d - item["volatility_30d"]) < 1e-12, label + ": past close/risk")
                review.check(stamp(row.model_available_after) == available and row.date >= available,
                             label + ": publication availability gate")
                review.check(row.batch_asset_keys == ",".join(batch), label + ": exact contemporaneous batch")
                rng = derive_seed(model_key, seed, row.date, batch) if seed is not None else None
                review.check((seed is None and pd.isna(row.derived_rng_seed)) or row.derived_rng_seed == rng,
                             label + ": fixed per-day RNG or unreplicated deterministic path")
                valid = np.isfinite(row.predicted_close) and row.predicted_close > 0
                review.check(bool(row.valid_forecast) == valid, label + ": explicit invalid forecast handling")
                expected_return = row.predicted_close / item["base_close"] - 1 if valid else np.nan
                review.check((pd.isna(expected_return) and pd.isna(row.predicted_return))
                             or abs(row.predicted_return - expected_return) < 1e-12, label + ": correct return fraction")
                cache[key] = row
            name = "deterministic" if seed is None else f"seed_{seed}"
            contract = read(output / "forecasts" / model_key / name / "cache_contract.json")
            review.check(contract["protocol_sha256"] == frozen["protocol_sha256"]
                         and contract["initial_parameter_sha256"] == manifest["parameter_sha256_before"]
                         and contract["plan_sha256"] == sha(output / "forecast_request_plan.csv"),
                         model_key + "/" + str(seed) + ": cache contract binds inputs and weights")
            days_directory = output / "forecasts" / model_key / name / "days"
            expected_dates = {date for date, key in expected_pairs}
            review.check({path.stem for path in days_directory.glob("*.csv")} ==
                         {date.strftime("%Y-%m-%d") for date in expected_dates},
                         model_key + "/" + str(seed) + ": no missing or extra daily caches")
            for date, combined_day in frame.groupby("date"):
                day_path = days_directory / (date.strftime("%Y-%m-%d") + ".csv")
                hash_path = day_path.with_suffix(".sha256.json")
                daily_hash = read(hash_path)
                review.check(daily_hash == {"sha256": sha(day_path), "date": str(date),
                                           "asset_keys": batches[date],
                                           "protocol_sha256": frozen["protocol_sha256"]},
                             model_key + "/" + str(seed) + "/" + str(date.date()) + ": resumable day cache hash")
                day_rows = pd.read_csv(day_path, dtype={"asset_key": str}, float_precision="round_trip")
                day_rows["date"] = pd.to_datetime(day_rows.date, utc=True)
                review.check(same_csv_records(day_rows.sort_values("asset_key").reset_index(drop=True),
                    combined_day.sort_values("asset_key").reset_index(drop=True)),
                    model_key + "/" + str(seed) + "/" + str(date.date()) + ": merged forecast exactly matches immutable daily cache")
            caches[seed] = cache
        expected_repeats = {(seed, stamp(day)) for seed in seeds for day in spec["native_repeat_dates"]
                            if stamp(day) >= available and batches.get(stamp(day))}
        repeated = manifest["native_repeat_verifications"]
        review.check({(entry["seed"], stamp(entry["date"])) for entry in repeated} == expected_repeats
                     and len(repeated) == len(expected_repeats) and all(entry["passed"] for entry in repeated),
                     model_key + ": fixed real-model repeat dates and paths")
        all_caches[model_key] = caches
        summaries[model_key] = {"paths": len(caches), "rows": sum(len(cache) for cache in caches.values()),
                                "first_decision_date": available.isoformat(), "native_repeat_checks": len(repeated),
                                "parameter_sha256": manifest["parameter_sha256_before"]}
    return all_caches, summaries


def audit(output=OUTPUT):
    output = Path(output)
    numeric = read(output / "numeric_completion.json")
    if numeric.get("status") != "complete" or numeric.get("evaluations") != 200 or numeric.get("repeat_checks") != 200:
        raise ValueError("All 200 evaluations and repeats must finish before final auditing")
    review = Checks()
    check = review.check
    spec, frozen = read(output / "frozen_spec.json"), read(output / "inference_frozen_inputs.json")
    proof_paths = ProofPaths(output, spec)
    if proof_paths.imported is not None:
        check(numeric.get("execution") == "returned_native_dual_v100_forecasts_local_original_ledger"
              and numeric.get("gpu_protocol_sha256") == sha(output / "frozen_spec.json")
              and numeric.get("local_import_manifest_sha256") == sha(output / "local_import_manifest.json"),
              "complete local200 evaluations bind the imported dual-V100 forecasts")
    check(sha(output / "frozen_spec.json") == frozen["protocol_sha256"], "predeclared protocol unchanged")
    check(sha(output / "market/manifest.json") == frozen["market_manifest_sha256"], "market provenance unchanged")
    check(sha(output / "sources/source_manifest.json") == frozen["source_manifest_sha256"], "official source manifest unchanged")
    audit_experiment_contract(review, output, spec, proof_paths)
    if proof_paths.imported is not None:
        audit_gpu_returned_receipt(review, output, spec, proof_paths)
    for relative, expected in frozen["source_sha256"].items():
        check(sha(ROOT / relative) == expected, "frozen inference/financial code: " + relative)
    for item in spec["pinned_files"]:
        check(sha(proof_paths.resolve(item["path"])) == item["sha256"], "official source/weight bytes: " + item["path"])
    old_count = assert_preserved(review, output, "before independent audit", proof_paths)
    market_audit = read(proof_paths.local_evidence("sources/independent_market_audit.json"))
    check(market_audit["status"] == "passed" and market_audit["checks_failed"] == 0,
          "independent real OHLCVA archive audit passed")
    market_meta = read(output / "market/manifest.json")
    # Import preserves the original manifest and its historical Windows path.
    # Read the actual returned market copy, whose SHA must equal that evidence.
    market_path = output / "market/daily_ohlcva.pkl.gz"
    check(sha(market_path) == market_meta["market_sha256"] == market_audit["market_sha256"], "same audited six-channel input bytes")
    with gzip.open(market_path, "rb") as stream:
        market = pickle.load(stream)
    datasets = {variant: pd.read_pickle(path) for variant, path in DATASETS.items()}
    for variant, data in datasets.items():
        check(sha(DATASETS[variant]) == frozen["dataset_sha256_by_variant"][variant], variant + ": same frozen original dataset")
        data["assets"] = {str(key): value for key, value in data["assets"].items()}
        data["universe"]["date"] = pd.to_datetime(data["universe"].date, utc=True)
        data["universe"]["asset_key"] = data["universe"].asset_key.astype(str)
    plan, batches = independent_plan(market, datasets, stamp(spec["validation"][0]), stamp(spec["test"][1]))
    audit_plan(review, output, plan)
    all_caches, cache_reviews = audit_caches(review, output, spec, frozen, plan, batches)
    completed_cases = {(entry["variant"], entry["model_key"]): entry for entry in numeric["cases"]}
    check(set(completed_cases) == {(variant, key) for variant in DATASETS for key in spec["models"]},
          "all six distinct actual model/universe cases")
    case_audits = {}
    for variant, dataset in datasets.items():
        prior_path = (ROOT / "results/backtests/crypto_rl_top10_20261024/manifest.json" if variant == "original"
                      else ROOT / "results/backtests/crypto_rl_ex_ada_20261024/ppo/manifest.json")
        prior_manifest = read(prior_path)
        for model_key, config in spec["models"].items():
            folder = output / variant / model_key
            manifest = read(folder / "manifest.json")
            prefix = variant + "/" + model_key
            check(sha(folder / "manifest.json") == completed_cases[(variant, model_key)]["manifest_sha256"],
                  prefix + ": completion-pinned result manifest")
            check(manifest["config"]["model_config"] == config and manifest["config"]["local_fit"] is False,
                  prefix + ": actual frozen model configuration")
            check(manifest["source_sha256"] == frozen["source_sha256"], prefix + ": same frozen implementation")
            check(manifest["forecast_manifest_sha256"] == sha(output / "forecasts" / model_key / "manifest.json"),
                  prefix + ": both universes use the same actual native predictions")
            seeds = config["sampling_seeds"]
            policies = [(model_key, seed) for seed in seeds]
            policies += [(model_key + "_ensemble" if seeds else model_key, None)]
            policies += [(name, None) for name in BASELINES]
            expected_runs = {f"{split}__{scenario}__{strategy}" + (f"__seed{seed}" if seed is not None else "")
                             for split in ("validation", "test") for scenario in ("base", "stress")
                             for strategy, seed in policies}
            check(len(expected_runs) == completed_cases[(variant, model_key)]["runs"] == (40 if seeds else 20),
                  prefix + ": correct stochastic or deterministic run count")

            def expected_target(row, date, current):
                if date < stamp(config["first_decision_date"]):
                    return np.r_[np.zeros(10), 1.]
                selected = seeds if row["strategy"].endswith("_ensemble") else [None] if not seeds else [int(row["seed"])]
                targets = []
                for seed in selected:
                    predictions, risks, valid = np.full(10, np.nan), np.full(10, np.nan), np.zeros(10, bool)
                    for slot, item in current.iterrows():
                        key = str(item.asset_key)
                        cached = all_caches[model_key][seed].get((date, key))
                        if cached is not None and not announced_exit(dataset["assets"], key, date):
                            predictions[slot], risks[slot], valid[slot] = cached.predicted_return, cached.volatility_30d, cached.valid_forecast
                    targets.append(target_weights(predictions, risks, valid=valid))
                return np.mean(targets, axis=0)

            with (folder / "audit.log").open("w", encoding="utf-8") as stream, redirect_stdout(stream):
                ledger = audit_case_economics(folder, DATASETS[variant], expected_runs=expected_runs, target_provider=expected_target)
            (folder / "independent_audit.json").write_text(json.dumps(ledger, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
            check(ledger["status"] == "passed" and ledger["checks_failed"] == 0 and ledger["runs_checked"] == len(expected_runs),
                  prefix + ": full independent target and funded-portfolio audit")
            case_audits[prefix] = ledger
            for run_id in expected_runs:
                if run_id.endswith(tuple("__" + name for name in BASELINES)):
                    check(manifest["run_results"][run_id]["fingerprints"] == prior_manifest["run_results"][run_id]["fingerprints"],
                          prefix + "/" + run_id + ": exact previous four-file baseline")
            if variant == "ex_ada":
                for run_id in expected_runs:
                    for name in ("fills", "weights"):
                        rows = pd.read_csv(folder / "runs" / run_id / (name + ".csv"), dtype={"asset_key": str})
                        check(not rows.asset_key.eq("2010").any(), prefix + "/" + run_id + ": no ADA " + name)
            print(f"AUDIT CASE {prefix}: {ledger['status']} / {ledger['checks_total']} checks", flush=True)
    assert_preserved(review, output, "after independent audit", proof_paths)
    ledger_failures = sum(case["checks_failed"] for case in case_audits.values())
    result = {"created_at_utc": datetime.now(timezone.utc).isoformat(),
              "status": "passed" if not review.failures and not ledger_failures else "failed",
              "phase": "full_pretrained_foundation_independent_audit",
              "runs_checked": sum(case["runs_checked"] for case in case_audits.values()),
              "checks_total": review.count + sum(case["checks_total"] for case in case_audits.values()),
              "checks_failed": len(review.failures) + ledger_failures,
              "wrapper_checks_total": review.count,
              "ledger_checks_total": sum(case["checks_total"] for case in case_audits.values()),
              "failures": review.failures, "original_files_preserved": old_count,
              "cases": {key: {name: value[name] for name in ("status", "checks_total", "checks_failed", "runs_checked",
                                                           "max_absolute_reconstruction_error_usdt")}
                        for key, value in case_audits.items()},
              "forecast_caches": cache_reviews,
              "execution": "returned_dual_v100_native_forecasts_local_unchanged_ledger" if proof_paths.imported is not None else "cpu_native_forecasts_local_unchanged_ledger",
              "proof_path_overrides_verified": len(proof_paths.overrides),
              "original_gpu_protocol_bytes_changed": False,
              "cpu_gpu_forecast_bitwise_equality_claimed": False,
              "daily_vs_merged_csv_comparison": {"method": "exact per-column Python scalar values; dtype-only numeric inference differences allowed",
                                                  "numeric_tolerance": None, "whole_frame_float_cast_used": False,
                                                  "prediction_values_modified_to_pass_audit": False,
                                                  "failed_import_diagnostic_path": str(ROOT / "results/backtests/crypto_foundation_models_v100_csv_storage_diagnostic_20261024.json")
                                                  if proof_paths.imported is not None else None,
                                                  "failed_import_diagnostic_sha256": sha(ROOT / "results/backtests/crypto_foundation_models_v100_csv_storage_diagnostic_20261024.json")
                                                  if proof_paths.imported is not None else None},
              "proof_resolver_source_sha256": sha(ROOT / "scripts/foundation_model_proof_paths.py"),
              "market_input_audit": {name: market_audit[name] for name in ("status", "checks_total", "checks_failed",
                                                                         "raw_archives_checked", "market_sha256")},
              "scope": "Input causality, fixed native caches/parameters, declared availability evidence and independently reproduced targets/ledger. Absence of historical prices from original pretraining corpora cannot be proved by this audit.",
              "auditor_source_sha256": sha(Path(__file__)),
              "independent_economics_source_sha256": sha(ROOT / "scripts/foundation_model_audit_helpers.py"),
              "old_reconstruct_source_sha256": sha(ROOT / "scripts/audit_crypto_rl_top10.py"),
              "old_or_production_results_modified": False}
    (output / "independent_audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    result = audit(args.output_dir.resolve())
    print(json.dumps({name: result[name] for name in ("status", "runs_checked", "checks_total", "checks_failed")}))
    raise SystemExit(int(bool(result["checks_failed"])))

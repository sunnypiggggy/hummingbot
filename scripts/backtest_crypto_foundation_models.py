"""Frozen, inference-only portfolio comparison of real pretrained models.

Forecasts are common to both historical universes and independent of holdings.
Only the immutable funded-spot ledger simulates trades. No production imports.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time
import uuid

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.backtest_crypto_rl_top10 import (
    COSTS, VAL_START, TEST_START, TEST_END, load_dataset, sha, evaluate as baseline_evaluate,
)
from scripts.crypto_rl_portfolio_env import CryptoPortfolioEnv
from scripts.crypto_rl_reporting import (
    canonical_csv, result_fingerprints, save_result, seed_statistics, summarize, write_json,
)
from scripts.crypto_foundation_portfolio import (
    OUTPUT, DATASETS, CONTEXT_DAYS, FIELDS, utc, load_market, context_frame, frame_digest,
    volatility, fixed_weights, known_exit_today,
)

BASELINES = ("cash", "top10_equal", "top10_cap", "btc_eth_hold")
FINANCIAL_SOURCES = ("scripts/backtest_crypto_foundation_models.py", "scripts/crypto_foundation_portfolio.py",
                     "scripts/foundation_model_native.py", "scripts/crypto_rl_portfolio_env.py",
                     "scripts/crypto_rl_reporting.py", "scripts/backtest_crypto_rl_top10.py", "scripts/crypto_rl_data.py")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def request_plan(output=OUTPUT):
    market, _ = load_market(output)
    datasets = {key: load_dataset(path) for key, path in DATASETS.items()}
    pools = {key: {date: list(rows.sort_values("slot").asset_key.astype(str))
                   for date, rows in data["universe"].groupby("date")} for key, data in datasets.items()}
    plans, records = {}, []
    for day in pd.date_range(utc(VAL_START), utc(TEST_END), freq="D", inclusive="left"):
        assets = sorted(set(pools["original"][day]) | set(pools["ex_ada"][day]), key=int)
        available = {}
        for key in assets:
            frame = context_frame(market, day, key)
            reason = "known_delisting" if known_exit_today(datasets["original"]["assets"][key], day) else ""
            if frame is None:
                reason = "insufficient_complete_128_day_context"
            row = {"date": day, "asset_key": key,
                   "symbol": datasets["original"]["assets"][key]["symbol"],
                   "context_ready": not bool(reason), "exclusion_reason": reason,
                   "input_start": day - pd.Timedelta(days=CONTEXT_DAYS), "input_end_exclusive": day,
                   "input_rows": CONTEXT_DAYS if frame is not None else 0,
                   "input_sha256": frame_digest(frame) if frame is not None else None,
                   "base_close": float(frame.close.iloc[-1]) if frame is not None else None,
                   "volatility_30d": volatility(frame) if frame is not None else None}
            records.append(row)
            if not reason:
                available[key] = {"frame": frame, "record": row}
        plans[day] = available
    return plans, pd.DataFrame(records)


def _atomic_bytes(path, data):
    """Publish a complete file; each date/shard has exactly one worker owner."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.partial")
    try:
        temp.write_bytes(data)
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.partial")
    try:
        write_json(temp, value)
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def _commit_cache_contract(path, contract):
    """An atomic hard link commits once, without overwriting another shard."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.partial")
    try:
        write_json(temp, contract)
        try:
            os.link(temp, path)
        except FileExistsError:
            pass
        if read(path) != contract:
            raise AssertionError("A partial prediction cache belongs to different frozen inputs")
    finally:
        temp.unlink(missing_ok=True)


def forecast_jobs(output, spec):
    """Split dates, never assets within the same seeded native model batch."""
    counts = spec.get("date_shards", {"kronos_small": 2, "kronos_base": 4, "chronos2": 8})
    jobs = []
    for key, config in spec["models"].items():
        count = counts[key]
        if not isinstance(count, int) or count < 1:
            raise ValueError("Every native model needs a positive integer date-shard count")
        for seed in config["sampling_seeds"] or [None]:
            for index in range(count):
                jobs.append({"output": str(output), "model_key": key, "spec": spec,
                             "seed": seed, "shard_index": index, "shard_count": count})
    return jobs


def _read_request_plan(output):
    frame = pd.read_csv(Path(output) / "forecast_request_plan.csv", dtype={"asset_key": str},
                        float_precision="round_trip")
    for column in ("date", "input_start", "input_end_exclusive"):
        frame[column] = pd.to_datetime(frame[column], utc=True)
    if frame.duplicated(["date", "asset_key"]).any():
        raise AssertionError("The frozen request plan repeats a prediction identity")
    days = pd.date_range(utc(VAL_START), utc(TEST_END), freq="D", inclusive="left")
    if not pd.DatetimeIndex(sorted(frame.date.unique())).equals(days):
        raise AssertionError("The frozen request plan must cover every decision date")
    return frame, days


def forecast_task(task):
    """One seed/date shard writes disjoint day caches and its own evidence."""
    from scripts.foundation_model_native import NativeForecaster
    output, spec, key = Path(task["output"]), task["spec"], task["model_key"]
    seed, shard_index, shard_count = task["seed"], task["shard_index"], task["shard_count"]
    config = spec["models"][key]
    if not 0 <= shard_index < shard_count or seed not in (config["sampling_seeds"] or [None]):
        raise ValueError("Native shard identity is outside the fixed seed/date schedule")
    started = time.monotonic()
    model = NativeForecaster(key, config, threads=spec["threads_per_model"])
    market, _ = load_market(output)
    plan, days = _read_request_plan(output)
    first = utc(config["first_decision_date"])
    own_days = set(days[shard_index::shard_count])
    own_plan = plan.loc[plan.date.isin(own_days) & (plan.date >= first) & plan.context_ready]
    groups = list(own_plan.groupby("date", sort=True))
    seed_name = "deterministic" if seed is None else f"seed_{seed}"
    folder = output / "forecasts" / key / seed_name
    contract = {"protocol_sha256": sha(output / "frozen_spec.json"), "model_key": key, "seed": seed,
                "initial_parameter_sha256": model.initial_parameter_sha256,
                "plan_sha256": sha(output / "forecast_request_plan.csv"), "models": config}
    _commit_cache_contract(folder / "cache_contract.json", contract)
    records, repeat_checks = [], []
    for number, (day, rows) in enumerate(groups, 1):
        rows = rows.iloc[np.argsort(rows.asset_key.map(int).to_numpy(), kind="stable")]
        batch_keys = rows.asset_key.tolist()
        frames = [context_frame(market, day, asset) for asset in batch_keys]
        for frame, record in zip(frames, rows.itertuples(index=False)):
            if frame is None or frame_digest(frame) != record.input_sha256:
                raise AssertionError("A native shard context differs from the frozen request plan")
        path = folder / "days" / (day.strftime("%Y-%m-%d") + ".csv")
        hash_path = path.with_suffix(".sha256.json")
        if hash_path.exists() and not path.exists():
            raise AssertionError("A committed native day cache is missing its prediction file")
        if path.exists() and hash_path.exists():
            if read(hash_path) != {"sha256": sha(path), "date": str(day), "asset_keys": batch_keys,
                                    "protocol_sha256": contract["protocol_sha256"]}:
                raise AssertionError("Committed native day cache changed before resume")
            frame = pd.read_csv(path, dtype={"asset_key": str}, float_precision="round_trip")
            if frame.asset_key.tolist() != batch_keys or not pd.to_datetime(frame.date, utc=True).eq(day).all():
                raise AssertionError("Cached native prediction batch identity differs")
            if frame.input_sha256.tolist() != rows.input_sha256.tolist():
                raise AssertionError("Cached native prediction context changed")
        else:
            predicted, rng_seed = model.predict(frames, day, seed=seed or 0, asset_keys=batch_keys)
            forecasts = []
            for source, prediction in zip(rows.to_dict("records"), predicted):
                close = prediction["predicted_close"]
                valid = close > 0
                forecasts.append({**source, **prediction, "model_key": key, "seed": seed,
                                  "derived_rng_seed": rng_seed if seed is not None else None,
                                  "batch_asset_keys": ",".join(batch_keys),
                                  "predicted_return": close / source["base_close"] - 1 if valid else None,
                                  "valid_forecast": valid,
                                  "forecast_reason": "" if valid else "nonpositive_forecast_price",
                                  "model_available_after": first})
            frame = pd.DataFrame(forecasts)
            encoded = canonical_csv(frame)
            if path.exists() and path.read_bytes() != encoded:
                raise AssertionError("An uncommitted day cache does not match recomputed native inference")
            _atomic_bytes(path, encoded)
            _atomic_json(hash_path, {"sha256": sha(path), "date": str(day), "asset_keys": batch_keys,
                                     "protocol_sha256": contract["protocol_sha256"]})
        # Recompute native calls on fixed dates only in that date's owning shard.
        if day.isoformat() in spec["native_repeat_dates"]:
            again, _ = model.predict(frames, day, seed=seed or 0, asset_keys=batch_keys)
            if any(row["predicted_close"] != float(saved)
                   for row, saved in zip(again, frame.predicted_close)):
                raise AssertionError("Native inference repeat changed a forecast")
            repeat_checks.append({"model_key": key, "seed": seed, "date": str(day), "passed": True})
        records.append({"date": str(day), "path": str(path.resolve()), "sha256": sha(path), "rows": len(frame)})
        if number % 15 == 0 or number == len(groups):
            print(f"FORECAST {key}/{seed_name}/shard{shard_index + 1}of{shard_count}: {number}/{len(groups)} dates", flush=True)
    after = model.verify_unchanged()
    manifest = {"model_key": key, "seed": seed, "shard_index": shard_index, "shard_count": shard_count,
                "local_fit": False, "inference_only": True, "created_at": datetime.now(timezone.utc).isoformat(),
                "config": config, "elapsed_seconds": time.monotonic() - started, "day_files": records,
                "parameter_sha256_before": model.initial_parameter_sha256, "parameter_sha256_after": after,
                "native_repeat_verifications": repeat_checks,
                "framework_versions": {name: importlib.metadata.version(name) for name in
                                       ("torch", "numpy", "pandas", "transformers", "huggingface-hub")}}
    shard_path = folder / "shards" / f"shard_{shard_index}_of_{shard_count}.json"
    _atomic_json(shard_path, manifest)
    return {"model_key": key, "seed": seed, "shard_index": shard_index, "shard_count": shard_count,
            "path": str(shard_path.resolve()), "sha256": sha(shard_path)}


def merge_forecast_shards(output, spec, results):
    """Only the parent publishes complete per-seed CSVs and model manifests."""
    output = Path(output)
    jobs = forecast_jobs(output, spec)
    identity = lambda entry: (entry["model_key"], entry["seed"], entry["shard_index"], entry["shard_count"])
    if len(results) != len(jobs) or {identity(row) for row in results} != {identity(row) for row in jobs}:
        raise AssertionError("Native date shards are incomplete, duplicated or unexpected")
    plan, days = _read_request_plan(output)
    manifests = []
    for key, config in spec["models"].items():
        first = utc(config["first_decision_date"])
        requested = plan.loc[(plan.date >= first) & plan.context_ready]
        expected_pairs = set(zip(requested.date, requested.asset_key))
        expected_dates = set(requested.date)
        all_records, all_repeats, shards = [], [], []
        before, versions, elapsed = None, None, 0.
        for seed in config["sampling_seeds"] or [None]:
            selected = sorted((entry for entry in results if entry["model_key"] == key and entry["seed"] == seed),
                              key=lambda entry: entry["shard_index"])
            parts, seen_days = [], set()
            for entry in selected:
                if sha(entry["path"]) != entry["sha256"]:
                    raise AssertionError("Native shard evidence changed before the parent merge")
                shard = read(entry["path"])
                if identity(shard) != identity(entry) or shard["config"] != config or shard["local_fit"] is not False:
                    raise AssertionError("Native shard was produced by a different frozen experiment")
                if before is None:
                    before, versions = shard["parameter_sha256_before"], shard["framework_versions"]
                if shard["parameter_sha256_before"] != before or shard["parameter_sha256_after"] != before or shard["framework_versions"] != versions:
                    raise AssertionError("Native shards used different weights or dependency versions")
                own_days = set(days[entry["shard_index"]::entry["shard_count"]]) & expected_dates
                if {utc(day["date"]) for day in shard["day_files"]} != own_days or len(shard["day_files"]) != len(own_days):
                    raise AssertionError("A native shard does not exactly cover its assigned dates")
                for item in shard["day_files"]:
                    day, path = utc(item["date"]), Path(item["path"])
                    if day in seen_days or sha(path) != item["sha256"]:
                        raise AssertionError("Native date shards overlap or a day cache changed")
                    batch = sorted(requested.loc[requested.date == day].asset_key, key=int)
                    if read(path.with_suffix(".sha256.json")) != {"sha256": item["sha256"], "date": str(day),
                                                                 "asset_keys": batch, "protocol_sha256": sha(output / "frozen_spec.json")}:
                        raise AssertionError("Native day cache is not bound to the frozen protocol")
                    frame = pd.read_csv(path, dtype={"asset_key": str}, float_precision="round_trip")
                    frame["date"] = pd.to_datetime(frame.date, utc=True)
                    if frame.asset_key.tolist() != batch or not frame.date.eq(day).all() or len(frame) != item["rows"]:
                        raise AssertionError("Native day batch differs from its declared identity")
                    parts.append(frame)
                    seen_days.add(day)
                all_repeats.extend(shard["native_repeat_verifications"])
                elapsed += shard["elapsed_seconds"]
                shards.append(entry)
            if seen_days != expected_dates or not parts:
                raise AssertionError("A native sampling path has missing forecast dates")
            combined = pd.concat(parts, ignore_index=True).sort_values(["date", "asset_key"], kind="stable")
            if combined.duplicated(["date", "asset_key"]).any() or set(zip(combined.date, combined.asset_key)) != expected_pairs:
                raise AssertionError("Merged native forecasts are not a unique exhaustive request cover")
            seed_name = "deterministic" if seed is None else f"seed_{seed}"
            path = output / "forecasts" / key / f"{seed_name}.csv"
            _atomic_bytes(path, canonical_csv(combined))
            all_records.append({"model_key": key, "seed": seed, "path": str(path.resolve()),
                                "sha256": sha(path), "rows": len(combined)})
        expected_repeats = {(seed, utc(day)) for seed in config["sampling_seeds"] or [None]
                            for day in spec["native_repeat_dates"] if utc(day) in expected_dates}
        if len(all_repeats) != len(expected_repeats) or {(row["seed"], utc(row["date"])) for row in all_repeats} != expected_repeats or not all(row["passed"] for row in all_repeats):
            raise AssertionError("Native repeat dates/paths do not exactly match the predeclared set")
        manifest = {"model_key": key, "local_fit": False, "inference_only": True,
                    "created_at": datetime.now(timezone.utc).isoformat(), "config": config,
                    "elapsed_seconds": elapsed, "elapsed_seconds_definition": "sum of worker shard wall times; concurrent workers overlap",
                    "forecast_files": all_records, "parameter_sha256_before": before, "parameter_sha256_after": before,
                    "native_repeat_verifications": sorted(all_repeats, key=lambda row: (str(row["seed"]), row["date"])),
                    "framework_versions": versions, "shards": shards}
        _atomic_json(output / "forecasts" / key / "manifest.json", manifest)
        manifests.append(manifest)
        print("FROZEN FORECASTS " + key, flush=True)
    return manifests


def load_forecasts(output, model_key):
    manifest = read(Path(output) / "forecasts" / model_key / "manifest.json")
    if manifest["parameter_sha256_before"] != manifest["parameter_sha256_after"]:
        raise AssertionError("Pretrained weights were updated")
    caches = {}
    for item in manifest["forecast_files"]:
        if sha(item["path"]) != item["sha256"]:
            raise AssertionError("A native forecast cache changed")
        frame = pd.read_csv(item["path"], dtype={"asset_key": str}, float_precision="round_trip")
        frame["date"] = pd.to_datetime(frame.date, utc=True)
        if frame.duplicated(["date", "asset_key"]).any():
            raise AssertionError("Repeated native prediction identity")
        caches[item["seed"]] = {(row.date, row.asset_key): row for row in frame.itertuples(index=False)}
    return caches, manifest


def target_for_day(env, caches, seeds, model_config):
    day = env.current_date
    if day < utc(model_config["first_decision_date"]):
        return np.r_[np.zeros(10), 1.]
    targets = []
    for seed in seeds:
        predicted, vol, valid = np.full(10, np.nan), np.full(10, np.nan), np.zeros(10, dtype=bool)
        for slot, asset in enumerate(env.current_assets):
            if not env.current_valid_mask[slot]:
                continue
            item = caches[seed].get((day, asset))
            if item is None:
                # Only an explicitly documented context exclusion can omit an
                # otherwise feature-ready member; preflight tests cover this.
                raise AssertionError(f"Missing cached forecast for eligible {day}/{asset}")
            predicted[slot], vol[slot], valid[slot] = item.predicted_return, item.volatility_30d, item.valid_forecast
        targets.append(fixed_weights(predicted, vol, valid))
    return np.mean(targets, axis=0)


def evaluate(dataset, caches, *, strategy, seeds, model_config, start, end, costs):
    if strategy in BASELINES:
        return baseline_evaluate(dataset, [], strategy=strategy, start=start, end=end, costs=costs)
    env = CryptoPortfolioEnv(dataset, start, end, costs=costs, training=False)
    env.reset(seed=0)
    done = False
    while not done:
        weights = target_for_day(env, caches, seeds, model_config)
        _, _, terminated, truncated, _ = env.step_target(weights)
        done = terminated or truncated
    result = env.get_result()
    result["metadata"].update(strategy=strategy, implementation="foundation", local_fit=False,
                              forecast_paths=len(seeds), first_decision_date=model_config["first_decision_date"])
    env.close()
    return result


def case_task(task):
    output, key, variant, spec = Path(task["output"]), task["model_key"], task["variant"], task["spec"]
    folder = output / variant / key
    config = spec["models"][key]
    caches, forecast_manifest = load_forecasts(output, key)
    dataset = load_dataset(DATASETS[variant])
    sample_seeds = config["sampling_seeds"]
    plans = [(key, seed, [seed], False) for seed in sample_seeds]
    plans += [(key + "_ensemble" if sample_seeds else key, None, sample_seeds or [None], True)]
    plans += [(name, None, [], False) for name in BASELINES]
    rows, histories, months, holdings, runs, repeats = [], [], [], [], {}, []
    for split, start, end in (("validation", VAL_START, TEST_START), ("test", TEST_START, TEST_END)):
        for scenario, costs in COSTS.items():
            for strategy, seed, selected, primary in plans:
                run_id = f"{split}__{scenario}__{strategy}" + (f"__seed{seed}" if seed is not None else "")
                result = evaluate(dataset, caches, strategy=strategy, seeds=selected, model_config=config,
                                  start=start, end=end, costs=costs)
                fingerprints = save_result(folder / "runs" / run_id, result)
                repeated = evaluate(dataset, caches, strategy=strategy, seeds=selected, model_config=config,
                                    start=start, end=end, costs=costs)
                if fingerprints != result_fingerprints(repeated):
                    raise AssertionError("Frozen prediction-ledger evaluation is not deterministic")
                row, daily, monthly = summarize(result, run_id=run_id, strategy=strategy, seed=seed,
                                                 scenario=scenario, split=split, start=start, end=end, **costs)
                role = "baseline" if strategy in BASELINES else "primary" if primary else "sample_seed"
                label = config["label"] + ("五路径平均权重组合" if primary and sample_seeds else
                         "确定性中位数策略" if primary else f"采样路径{seed}") if strategy not in BASELINES else row["label"]
                extra = {"model_key": key if strategy not in BASELINES else "shared", "primary_policy": primary,
                         "policy_role": role, "implementation": "foundation", "algorithm": config["label"]}
                row.update(extra, label=label)
                for frame in (daily, monthly):
                    frame["label"] = label
                    for field, value in extra.items():
                        frame[field] = value
                rows.append(row)
                histories.append(daily)
                months.append(monthly)
                holdings.extend({**h, "run_id": run_id, "seed": seed, "scenario": scenario, "split": split, "strategy": strategy}
                                for h in result["metadata"]["final_holdings"])
                runs[run_id] = {"fingerprints": fingerprints, "metadata": result["metadata"]}
                repeats.append({"run_id": run_id, "passed": True, "fingerprints": fingerprints})
                print(f"EVAL {variant}/{key}/{run_id}: return={row['net_return']:.2%} DD={row['max_drawdown']:.2%}", flush=True)
    summary = pd.DataFrame(rows)
    tables = {"summary": summary, "daily_equity": pd.concat(histories, ignore_index=True),
              "monthly_returns": pd.concat(months, ignore_index=True), "seed_statistics": seed_statistics(summary),
              "final_holdings": pd.DataFrame(holdings)}
    if not sample_seeds:
        tables["seed_statistics"] = pd.DataFrame(columns=["split", "scenario", "seeds", "median_net_return", "mean_net_return", "worst_net_return", "best_net_return", "median_max_drawdown", "worst_max_drawdown", "positive_seeds"])
    for name, frame in tables.items():
        frame.to_csv(folder / (name + ".csv"), index=False, encoding="utf-8-sig", float_format="%.17g")
    common = {"validation_start": VAL_START, "validation_end": TEST_START, "test_start": TEST_START,
              "test_end": TEST_END, "costs": COSTS, "capital": 10000,
              "cash_target_floor": .05, "single_coin_target_cap": .25,
              "model_key": key, "local_fit": False, "universe_variant": variant,
              "model_config": config, "policy": spec["forecast_to_weights"]}
    frozen = read(output / "inference_frozen_inputs.json")
    write_json(folder / "config.json", common)
    write_json(folder / "inference_frozen_inputs.json", frozen)
    write_json(folder / "manifest.json", {"created_at": datetime.now(timezone.utc).isoformat(),
               "config": common, "model_key": key, "implementation": "foundation", "local_fit": False,
               "universe_variant": variant, "source_sha256": frozen["source_sha256"],
               "frozen_inputs": {**frozen, "dataset_sha256": sha(DATASETS[variant])},
               "forecast_manifest": forecast_manifest, "forecast_manifest_sha256": sha(output / "forecasts" / key / "manifest.json"),
               "run_results": runs, "repeat_verifications": repeats, "production_changes": False})
    return {"variant": variant, "model_key": key, "runs": len(runs), "manifest_sha256": sha(folder / "manifest.json")}


def diagnostics(output, spec):
    plan = pd.read_csv(output / "forecast_request_plan.csv", dtype={"asset_key": str})
    plan["date"] = pd.to_datetime(plan.date, utc=True)
    plan_map = {(r.date, r.asset_key): r for r in plan.itertuples(index=False)}
    result = []
    for key, config in spec["models"].items():
        caches, _ = load_forecasts(output, key)
        for variant, path in DATASETS.items():
            dataset = load_dataset(path)
            for day, pool in dataset["universe"].groupby("date"):
                if not utc(VAL_START) <= day < utc(TEST_END):
                    continue
                available = day >= utc(config["first_decision_date"])
                entries = [plan_map[(day, str(asset))] for asset in pool.asset_key]
                counts = [sum((day, str(asset)) in cache for asset in pool.asset_key) for cache in caches.values()]
                positives = [sum(getattr(cache.get((day, str(asset))), "predicted_return", -np.inf) > .0024
                                 for asset in pool.asset_key) for cache in caches.values()]
                result.append({"model_key": key, "universe_variant": variant, "date": day,
                               "asset_count": len(pool), "context_ready_count": sum(e.context_ready for e in entries),
                               "forecast_available_count": min(counts) if counts else 0,
                               "score_positive_count": float(np.mean(positives)) if available else 0,
                               "model_available": available, "sampling_paths": len(caches),
                               "exclusion_reason": "model_not_publicly_available" if not available else
                               ";".join(sorted({e.exclusion_reason for e in entries if not e.context_ready}))})
    pd.DataFrame(result).to_csv(output / "forecast_diagnostics.csv", index=False, encoding="utf-8-sig", float_format="%.17g")


def verify_sources(spec):
    for item in spec["pinned_files"]:
        if sha(item["path"]) != item["sha256"]:
            raise AssertionError("Pinned official source/weight changed: " + item["path"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--phase", choices=("all", "forecast", "evaluate"), default="all")
    parser.add_argument("--resume-forecasts", action="store_true")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    spec = read(output / "frozen_spec.json")
    if spec["test"] != [TEST_START, TEST_END] or spec["validation"] != [VAL_START, TEST_START] or spec["costs"] != COSTS or spec["local_fit"] is not False:
        raise ValueError("The unchanged dates/costs and inference-only protocol are required")
    verify_sources(spec)
    frozen = {"prepared_at": datetime.now(timezone.utc).isoformat(), "protocol_sha256": sha(output / "frozen_spec.json"),
              "source_sha256": {name: sha(ROOT / name) for name in FINANCIAL_SOURCES},
              "dataset_sha256_by_variant": {name: sha(path) for name, path in DATASETS.items()},
              "market_manifest_sha256": sha(output / "market/manifest.json"),
              "source_manifest_sha256": sha(output / "sources/source_manifest.json")}
    frozen_path = output / "inference_frozen_inputs.json"
    if frozen_path.exists():
        previous = read(frozen_path)
        if any(previous[k] != v for k, v in frozen.items() if k != "prepared_at"):
            raise AssertionError("Frozen study inputs changed; preserve this experiment")
        if args.phase != "evaluate" and not args.resume_forecasts:
            raise ValueError("Use explicit --resume-forecasts for the same frozen partial cache")
    else:
        if args.phase == "evaluate":
            raise ValueError("No frozen native prediction study exists")
        write_json(frozen_path, frozen)
    plan_path = output / "forecast_request_plan.csv"
    _, plan = request_plan(output)
    encoded = canonical_csv(plan)
    if plan_path.exists() and plan_path.read_bytes() != encoded:
        raise AssertionError("A frozen causal request plan changed")
    plan_path.write_bytes(encoded)
    if args.phase != "evaluate":
        forecast_results = []
        with ProcessPoolExecutor(max_workers=spec["model_workers"]) as pool:
            tasks = [pool.submit(forecast_task, task) for task in forecast_jobs(output, spec)]
            for future in as_completed(tasks):
                forecast_results.append(future.result())
        merge_forecast_shards(output, spec, forecast_results)
    if args.phase != "forecast":
        tasks = [{"output": str(output), "model_key": key, "variant": variant, "spec": spec}
                 for variant in DATASETS for key in spec["models"]]
        results = []
        with ProcessPoolExecutor(max_workers=4) as pool:
            for future in as_completed([pool.submit(case_task, task) for task in tasks]):
                results.append(future.result())
        diagnostics(output, spec)
        total = sum(row["runs"] for row in results)
        write_json(output / "numeric_completion.json", {"status": "complete", "created_at": datetime.now(timezone.utc).isoformat(),
                   "cases": results, "evaluations": total, "repeat_checks": total,
                   "new_models_trained": 0, "pretrained_models": len(spec["models"]), "production_changes": False})
    verify_sources(spec)
    final = read(frozen_path)
    if any(sha(ROOT / name) != expected for name, expected in final["source_sha256"].items()):
        raise AssertionError("Frozen financial/inference code changed while running")
    print("FOUNDATION PHASE COMPLETE " + args.phase, flush=True)


if __name__ == "__main__":
    main()

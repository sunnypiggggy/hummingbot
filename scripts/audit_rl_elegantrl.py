"""Independent native ElegantRL audit, with no agent/env/trainer imports.

Financial checks reuse the prior independently implemented reconstruct helper.
Native checkpoints are loaded as tensor dictionaries with weights_only=True.
"""
from __future__ import annotations
import argparse
from contextlib import redirect_stdout
from datetime import datetime, timezone
from functools import lru_cache
import gzip
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.audit_crypto_rl_top10 import reconstruct, independent_training_candidates, stamp, announced_exit, sha


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def independent_tensor_digest(tensors):
    digest = hashlib.sha256()
    for name in sorted(tensors):
        array = tensors[name].detach().cpu().numpy()
        if not np.isfinite(array).all():
            raise AssertionError("Nonfinite native model tensor")
        digest.update(name.encode())
        digest.update(str(array.shape).encode())
        digest.update(str(array.dtype).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


@lru_cache(maxsize=64)
def native_checkpoint_digests(path):
    import torch
    payload = torch.load(path, map_location="cpu", weights_only=True)
    tensors = {}
    for scope in ("act", "cri", "act_target", "cri_target"):
        if scope in payload:
            tensors.update({scope + "." + key: value for key, value in payload[scope].items()})
    if "alpha_log" in payload:
        tensors["alpha_log"] = payload["alpha_log"]
    if not tensors or not payload.get("act"):
        raise AssertionError("Checkpoint contains no native actor/network tensors")
    return independent_tensor_digest(tensors), independent_tensor_digest(payload["act"])


def episode_sequence(record):
    return [{"start": stamp(row["start"]).isoformat(),
             "end_exclusive": stamp(row["end_exclusive"]).isoformat(),
             "completed_until": stamp(row["completed_until"]).isoformat(),
             "completed_steps": int(row["completed_steps"])}
            for row in record["training_episode_history"] if int(row["completed_steps"]) > 0]


def paired_sampling(reference, current):
    before, after = episode_sequence(reference), episode_sequence(current)
    differences = [{"episode": i, "sb3_ppo": before[i] if i < len(before) else None,
                    "native": after[i] if i < len(after) else None}
                   for i in range(max(len(before), len(after)))
                   if i >= len(before) or i >= len(after) or before[i] != after[i]]
    digest = lambda value: hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {"seed": int(current["seed"]), "identical": not differences,
            "sb3_nonempty_episodes": len(before), "native_nonempty_episodes": len(after),
            "sb3_completed_steps": sum(row["completed_steps"] for row in before),
            "native_completed_steps": sum(row["completed_steps"] for row in after),
            "sb3_sequence_sha256": digest(before), "native_sequence_sha256": digest(after),
            "different_episode_count": len(differences), "differences": differences,
            "zero_step_reset_records_excluded": True}


def audit(output, dataset_path):
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    config = manifest["config"]
    data_manifest = json.loads((dataset_path.parent / "manifest.json").read_text(encoding="utf-8"))
    checks, failures, runs = 0, [], {}

    def check(condition, label):
        nonlocal checks
        checks += 1
        if not bool(condition):
            failures.append(label)

    check(sha(dataset_path) == data_manifest["dataset_sha256"], "prepared data SHA256")
    check(sha(dataset_path) == manifest["frozen_inputs"]["dataset_sha256"], "training used this immutable prepared data")
    for path, expected in manifest["source_sha256"].items():
        check(sha(ROOT / path) == expected, "unchanged frozen financial source: " + path)
    dataset = pd.read_pickle(dataset_path)
    dataset["assets"] = {str(k): v for k, v in dataset["assets"].items()}
    universe = dataset["universe"].copy()
    for field in ("date", "snapshot_date", "available_after", "anchor"):
        universe[field] = pd.to_datetime(universe[field], utc=True)
    universe["asset_key"] = universe.asset_key.astype(str)
    dataset["universe"] = universe
    complete = None
    for field in ("open", "high", "low", "close", "volume"):
        quotes = dataset["hourly_" + field]
        available = np.isfinite(quotes) & (quotes.ge(0) if field == "volume" else quotes.gt(0))
        complete = available if complete is None else complete & available
    complete_days = complete.resample("D").sum().eq(24)
    daily_volume = dataset["hourly_volume"].resample("D").sum(min_count=24).where(complete_days)
    ready = complete_days.rolling(31, min_periods=31).sum().eq(31) & daily_volume.rolling(30, min_periods=30).mean().gt(0)
    ready.index = ready.index + pd.Timedelta(days=1)
    check((universe.available_after == universe.snapshot_date + pd.Timedelta(days=2)).all(), "declared two-day historical snapshot lag")
    check((universe.available_after <= universe.anchor).all() and (universe.anchor <= universe.date).all(), "universe information available before decision")
    check(universe.groupby("date").slot.apply(lambda x: sorted(x) == list(range(10))).all(), "ten unique historical slots each day")
    check(universe.groupby("date").asset_key.nunique().eq(10).all(), "no duplicate asset identity")
    eligible, covered, bad = independent_training_candidates(dataset, stamp(config["train_start"]), stamp(config["train_end"]))
    for seed in manifest["models"]:
        check(sha(seed["model_path"]) == seed["model_sha256"], f"seed{seed['seed']}: frozen model file")
        check(native_checkpoint_digests(seed["model_path"])[0] == seed["parameter_sha256"],
              f"seed{seed['seed']}: independent policy tensor digest")
        check(native_checkpoint_digests(seed["model_path"])[1] == seed["actor_parameter_sha256"],
              f"seed{seed['seed']}: independent native actor tensor digest")

        after = [item for item in manifest.get("models_unchanged_after_evaluation", []) if item["seed"] == seed["seed"]]
        check(len(after) == 1 and after[0]["parameters_and_checkpoint_unchanged_after_evaluation"],
              f"seed{seed['seed']}: producer before/after evaluation memory+file guard passed")
        quality = seed["training_window_metadata"]
        check(quality["eligible_windows"] == len(eligible), f"seed{seed['seed']}: eligible train starts")
        check(quality["covered_training_days"] == len(covered), f"seed{seed['seed']}: effective train days")
        check({stamp(row["date"]) for row in quality["incomplete_days"]} == bad, f"seed{seed['seed']}: excluded train gap dates")
        history = seed.get("training_episode_history", [])
        check(bool(history), f"seed{seed['seed']}: actual sampled episode history exists")
        total_steps = 0
        for i, episode in enumerate(history):
            start, end = stamp(episode["start"]), stamp(episode["end_exclusive"])
            completed = stamp(episode["completed_until"])
            check(start in eligible and end == start + pd.Timedelta(days=128), f"seed{seed['seed']}: episode{i} only eligible TRAIN data")
            check(start <= completed <= end and (completed - start).days == episode["completed_steps"], f"seed{seed['seed']}: episode{i} completion")
            total_steps += episode["completed_steps"]
        check(total_steps == seed["timesteps"] == 32768, f"seed{seed['seed']}: bounded actual training budget")
    summary = pd.read_csv(output / "summary.csv", float_precision="round_trip")
    daily_table = pd.read_csv(output / "daily_equity.csv", float_precision="round_trip")
    monthly_table = pd.read_csv(output / "monthly_returns.csv", float_precision="round_trip")
    maximum_errors = {"cash": 0., "base_value": 0., "equity": 0.}
    univ = {date: rows.sort_values("slot") for date, rows in universe.groupby("date")}
    repeats = manifest.get("repeat_verifications", [])
    check(len(repeats) == len(summary) == 40 and all(item["passed"] for item in repeats),
          "all forty frozen evaluation repeat checks recorded")
    for row in summary.to_dict("records"):
        run_id = row["run_id"]
        folder = output / "runs" / run_id
        fingerprints = json.loads((folder / "fingerprints.json").read_text(encoding="utf-8"))
        for name in ("equity", "fills", "weights", "boundary_equity"):
            if name == "equity":
                with gzip.open(folder / "equity_1h.csv.gz", "rb") as stream:
                    actual_hash = hashlib.sha256(stream.read()).hexdigest()
            else:
                actual_hash = sha(folder / (name + ".csv"))
            check(actual_hash == fingerprints[name] == manifest["run_results"][run_id]["fingerprints"][name],
                  run_id + ": serialized " + name + " fingerprint")
        start, end = stamp(row["start"]), stamp(row["end_exclusive"])
        costs = config["costs"][row["scenario"]]
        fills = pd.read_csv(folder / "fills.csv", float_precision="round_trip")
        equity = pd.read_csv(folder / "equity_1h.csv.gz", index_col="timestamp", float_precision="round_trip")
        equity.index = pd.to_datetime(equity.index, utc=True)
        initial = float(row["initial_capital"])
        rebuilt, inventories, economics = reconstruct(dataset, start, end, fills, initial, **costs)
        check(initial == 10000, run_id + ": equal initial capital")
        check(equity.index.equals(rebuilt.index), run_id + ": all evaluation hours once")
        check(not economics["errors"], run_id + ": fill/ownership constraints " + str(economics["errors"]))
        entry_information_ok = True
        for entry in fills.loc[fills.side.eq("BUY")].itertuples(index=False):
            date, key = stamp(entry.timestamp).normalize(), str(entry.asset_key)
            entry_information_ok &= key in set(univ[date].asset_key)
            entry_information_ok &= date in ready.index and bool(ready.loc[date, key])
            entry_information_ok &= not announced_exit(dataset["assets"], key, date)
        check(entry_information_ok, run_id + ": every purchase has eligible identity and 31 prior complete days")
        for field in maximum_errors:
            error = float(np.abs(rebuilt[field].to_numpy() - equity[field].to_numpy()).max())
            maximum_errors[field] = max(maximum_errors[field], error)
            check(error < .001, run_id + ": hourly " + field)
        values = rebuilt.equity.to_numpy()
        peaks = np.maximum.accumulate(np.r_[initial, values])[1:]
        drawdown = values / peaks - 1
        day_close = rebuilt.equity.resample("D").last()
        day_return = day_close.pct_change()
        day_return.iloc[0] = day_close.iloc[0] / initial - 1
        sd = day_return.std(ddof=1)
        sharpe = day_return.mean() / sd * np.sqrt(365) if sd > 1e-15 else np.nan
        result_metrics = {
            "final_equity": values[-1], "net_pnl": values[-1] - initial,
            "net_return": values[-1] / initial - 1, "max_drawdown": drawdown.min(),
            "sharpe_daily": sharpe, "fees": economics["fees"], "total_notional": economics["total_notional"],
            "turnover": economics["total_notional"] / initial, "slippage_cost": economics["slippage_cost"],
            "final_cash": rebuilt.cash.iloc[-1], "final_base_value": rebuilt.base_value.iloc[-1],
            "net_liquidation_equity": rebuilt.cash.iloc[-1] + rebuilt.base_value.iloc[-1] * (1 - costs["fee"]) * (1 - costs["slippage"]),
            "average_exposure": float((rebuilt.base_value / rebuilt.equity).mean()),
        }
        for metric, value in result_metrics.items():
            tolerance = .001 if metric in ("final_equity", "net_pnl", "fees", "total_notional", "slippage_cost", "final_cash", "final_base_value", "net_liquidation_equity") else 1e-8
            check((pd.isna(value) and pd.isna(row[metric])) or abs(value - row[metric]) < tolerance, run_id + ": " + metric)
        check(abs(values[-1] - initial - economics["realized_pnl"] - economics["unrealized_pnl"]) < .001, run_id + ": realized/open PNL bridge")
        history = daily_table.loc[daily_table.run_id == run_id]
        check(len(history) == len(day_close) + 1 and np.abs(history.equity.to_numpy() - np.r_[initial, day_close.to_numpy()]).max() < .001, run_id + ": daily plot")
        daily_dd = pd.Series(drawdown, index=rebuilt.index).resample("D").last().to_numpy()
        check(np.abs(history.drawdown.to_numpy() - np.r_[0., daily_dd]).max() < 1e-8,
              run_id + ": daily plotted drawdown retains hourly peaks")
        month = monthly_table.loc[monthly_table.run_id == run_id]
        last_month = day_close.resample("ME").last().to_numpy()
        expected_month = last_month / np.r_[initial, last_month[:-1]] - 1
        check(len(month) == len(expected_month) and np.abs(month["return"].to_numpy() - expected_month).max() < 1e-8, run_id + ": monthly returns")
        weights = pd.read_csv(folder / "weights.csv", float_precision="round_trip")
        weights["date"] = pd.to_datetime(weights.date, utc=True)
        weights["asset_key"] = weights.asset_key.astype(str)
        check(weights.date.nunique() == len(day_close) and len(weights) == len(day_close) * 10,
              run_id + ": complete daily target/actual slot records")
        identity_ok = True
        actual_ok = True
        target_ok = True
        for day, records in weights.groupby("date", sort=True):
            current = univ[day].set_index("slot")
            at = rebuilt.index.get_loc(day + pd.Timedelta(hours=23))
            for w in records.itertuples(index=False):
                key = str(current.loc[w.slot, "asset_key"])
                identity_ok &= str(w.asset_key) == key
                q = inventories[key][at]
                price = float(dataset["hourly_close"].loc[rebuilt.index[at], key])
                expected_weight = q * price / values[at] if abs(q) > 1e-10 else 0.
                actual_ok &= abs(expected_weight - w.actual_weight) < 1e-8
                if pd.notna(w.target_weight):
                    cap = 1. if row["strategy"] == "btc_eth_hold" else .25
                    target_ok &= -1e-10 <= w.target_weight <= cap + 1e-10
                    if w.target_weight > 1e-10:
                        target_ok &= bool(ready.loc[day, key]) and not announced_exit(dataset["assets"], key, day)
            target_sum = records.target_weight.sum()
            target_ok &= target_sum <= .95 + 1e-10
        check(identity_ok, run_id + ": holdings attached to stable identity, not rank")
        check(actual_ok, run_id + ": actual weights reconstruct from holdings")
        check(target_ok, run_id + ": declared target concentration and cash floor")
        boundary_path = folder / "boundary_equity.csv"
        check(boundary_path.is_file(), run_id + ": reward boundary evidence exists")
        if boundary_path.is_file():
            boundary = pd.read_csv(boundary_path, index_col="timestamp", float_precision="round_trip")
            boundary.index = pd.to_datetime(boundary.index, utc=True)
            check(boundary.index.equals(pd.date_range(start, end, freq="D")),
                  run_id + ": complete unique reward decision boundaries")
            expected_values = [initial]
            for date in boundary.index[1:]:
                if date == end:
                    expected_values.append(float(values[-1]))
                else:
                    prior = rebuilt.index.get_loc(date) - 1
                    amount = float(rebuilt.cash.iloc[prior])
                    for key, q in inventories.items():
                        if abs(q[prior]) > 1e-10:
                            amount += q[prior] * float(dataset["hourly_open"].loc[date, key])
                    expected_values.append(amount)
            check(len(boundary) == len(day_close) + 1 and np.abs(boundary.equity.to_numpy() - expected_values).max() < .001, run_id + ": next-open/last-close boundary NAV")
            expected_reward = np.r_[0., 100 * np.log(np.array(expected_values[1:]) / np.array(expected_values[:-1]))]
            check(np.abs(boundary.reward.to_numpy() - expected_reward).max() < 1e-7, run_id + ": actual net-log rewards")
            check(abs(boundary.reward.sum() - 100 * np.log(values[-1] / initial)) < 1e-7, run_id + ": reward telescopes to actual terminal net return")
        runs[run_id] = {"hours": len(rebuilt), "fills": len(fills), "net_return": result_metrics["net_return"],
                        "max_drawdown": result_metrics["max_drawdown"], "fees": economics["fees"]}
        print(f"AUDIT {run_id}: {len(rebuilt)} hours, {len(fills)} fills", flush=True)
    return {
        "created_at": datetime.now(timezone.utc).isoformat(), "status": "passed" if not failures else "failed",
        "independent_of_env_training_and_metric_helpers": True,
        "checks_total": checks, "checks_failed": len(failures), "failures": failures,
        "runs_checked": len(runs), "max_absolute_reconstruction_error_usdt": maximum_errors,
        "training_quality": {"eligible_windows": len(eligible), "covered_days": len(covered),
                             "bad_days": [str(t) for t in sorted(bad)]},
        "snapshot_scope": "retrospective vendor snapshots with declared lag, not proof of original publication vintage",
        "runs": runs,
    }


def audit_study(output):
    numeric = read(output / "numeric_completion.json")
    if numeric.get("status") != "complete" or numeric.get("evaluations") != 240 or numeric.get("repeat_checks") != 240:
        raise AssertionError("All 240 frozen evaluations and repeats must finish before final auditing")
    spec = read(output / "frozen_spec.json")
    frozen = read(output / "training_frozen_inputs.json")
    snapshot = read(output / "original_research_preservation.json")
    checks, failures = 0, []
    def check(condition, label):
        nonlocal checks
        checks += 1
        if not bool(condition):
            failures.append(label)
    check(sha(output / "frozen_spec.json") == frozen["protocol_sha256"], "pretraining protocol SHA")
    check(spec["seeds"] == list(range(5)) and spec["timesteps_per_seed"] == 32768,
          "same five seeds and exact interaction budget")
    check(set(spec["algorithms"]) == {"ppo", "sac", "td3"}, "three predeclared native algorithms")
    for relative, expected in frozen["code_sha256_before_training"].items():
        check(sha(ROOT / relative) == expected, "every frozen implementation: " + relative)
    for relative, expected in snapshot["files"].items():
        path = ROOT / relative
        check(path.is_file() and path.stat().st_size == expected["size_bytes"] and sha(path) == expected["sha256"],
              "original frozen artifact preserved: " + relative)
    for relative, expected in spec["vendor_file_sha256"].items():
        check(sha(Path(spec["vendor_path"]) / relative) == expected, "unmodified official source: " + relative)
    check(sha(spec["vendor_manifest"]) == frozen["vendor_manifest_sha256"], "frozen official source manifest")
    datasets = {"original": ROOT / "data/crypto_rl_top10/dataset.pkl.gz",
                "ex_ada": ROOT / "data/crypto_rl_top10_ex_ada/dataset.pkl.gz"}
    completed = {(case["variant"], case["algorithm"]): case for case in numeric["cases"]}
    check(len(numeric["cases"]) == 6 and numeric["models"] == 30
          and set(completed) == {(v, a) for v in datasets for a in spec["algorithms"]},
          "numeric completion covers all six distinct cases")
    cases, case_manifests, sampling = {}, {}, {}
    for variant, dataset_path in datasets.items():
        check(sha(dataset_path) == frozen["dataset_sha256_by_variant"][variant], variant + " same frozen dataset")
        reference_path = (ROOT / "results/backtests/crypto_rl_top10_20261024/manifest.json" if variant == "original"
                          else ROOT / "results/backtests/crypto_rl_ex_ada_20261024/ppo/manifest.json")
        reference_models = {m["seed"]: m for m in read(reference_path)["models"]}
        for algorithm in ("ppo", "sac", "td3"):
            folder = output / variant / algorithm
            manifest = case_manifests[(variant, algorithm)] = read(folder / "manifest.json")
            check(sha(folder / "manifest.json") == completed[(variant, algorithm)]["manifest_sha256"],
                  variant + "/" + algorithm + " completed manifest SHA")
            check(manifest["config"]["upstream_commit"] == spec["upstream_commit"], variant + "/" + algorithm + " same upstream pin")
            check(manifest["config"]["algorithm_parameters"] == spec["algorithms"][algorithm]["parameters"],
                  variant + "/" + algorithm + " predeclared actual parameters")
            expected = {f"{split}__{scenario}__{strategy}" + (f"__seed{seed}" if seed is not None else "")
                        for split in ("validation", "test") for scenario in ("base", "stress")
                        for strategy, seed in ([(algorithm, s) for s in range(5)] + [(algorithm + "_ensemble", None)]
                                               + [(x, None) for x in ("cash", "top10_equal", "top10_cap", "btc_eth_hold")])}
            check(set(manifest["run_results"]) == expected, variant + "/" + algorithm + " all intended forty runs")
            check(sorted(m["seed"] for m in manifest["models"]) == list(range(5)),
                  variant + "/" + algorithm + " all five model seeds")
            comparisons = []
            for model in manifest["models"]:
                label = f"{variant}/{algorithm}/seed{model['seed']}"
                comparison = paired_sampling(reference_models[model["seed"]], model)
                comparisons.append(comparison)
                check(comparison["identical"], label + " exact same actual nonzero TRAIN episodes as SB3 PPO")
                check(comparison["native_nonempty_episodes"] == comparison["sb3_nonempty_episodes"] == 256
                      and comparison["native_completed_steps"] == comparison["sb3_completed_steps"] == 32768,
                      label + " 256 complete 128-day episodes, no extra interaction")
                check(model["gradient_updates"] == (5120 if algorithm == "ppo" else 8000), label + " fixed gradient update budget")
                check(model["actor_updates"] == (4000 if algorithm == "td3" else model["gradient_updates"]),
                      label + " fixed actor update budget")
                check(model["replay_size"] == (0 if algorithm == "ppo" else 32768), label + " replay size includes warmup")
                check(model["replay_samples_drawn"] == (0 if algorithm == "ppo" else 8000 * spec["algorithms"][algorithm]["parameters"]["batch_size"]),
                      label + " actual uniform replay sample budget")
                history = [r for r in model["training_episode_history"] if r["completed_steps"] > 0]
                truncations = [r for r in history if r.get("truncated")]
                terminals = [r for r in history if r.get("terminated")]
                check(model["collector_truncations"] == len(truncations)
                      and model["collector_terminals"] == len(terminals), label + " actual reset boundary counts")
                adjustments = model["timeout_bootstrap_adjustments"]
                expected_steps, total = [], 0
                for episode in history:
                    total += episode["completed_steps"]
                    if episode.get("truncated"):
                        expected_steps.append(total)
                check([r["step"] for r in adjustments] == (expected_steps if algorithm == "ppo" else []),
                      label + " bootstrap adjustment only at true timeout steps")
                check(all(np.isfinite(r["bootstrap"]) for r in adjustments), label + " finite timeout value targets")
                import torch
                payload = torch.load(model["model_path"], map_location="cpu", weights_only=True)
                metadata = payload["metadata"]
                check(metadata["algorithm"] == algorithm and metadata["universe_variant"] == variant
                      and metadata["seed"] == model["seed"] and metadata["commit"] == spec["upstream_commit"],
                      label + " safe checkpoint metadata matches case")
            sampling[variant + "/" + algorithm] = comparisons
            if variant == "ex_ada":
                dataset = pd.read_pickle(dataset_path)
                check(not dataset["universe"].asset_key.astype(str).eq("2010").any(), "ADA absent from every ex-ADA train/eval slot")
                for run_id, entry in manifest["run_results"].items():
                    for name in ("fills", "weights"):
                        frame = pd.read_csv(folder / "runs" / run_id / (name + ".csv"), dtype={"asset_key": str})
                        check(not frame.asset_key.eq("2010").any(), variant + "/" + algorithm + "/" + run_id + " no ADA " + name)
                    check(all(str(h["asset_key"]) != "2010" for h in entry["metadata"]["final_holdings"]),
                          variant + "/" + algorithm + "/" + run_id + " no final ADA inventory")
            with (folder / "audit.log").open("w", encoding="utf-8") as stream, redirect_stdout(stream):
                ledger = audit(folder, dataset_path)
            (folder / "independent_audit.json").write_text(json.dumps(ledger, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
            cases[variant + "/" + algorithm] = ledger
            check(ledger["status"] == "passed" and ledger["checks_failed"] == 0 and ledger["runs_checked"] == 40,
                  variant + "/" + algorithm + " independent native model and complete ledger audit")
            print(f"AUDIT CASE {variant}/{algorithm}: {ledger['status']} / {ledger['checks_total']} checks", flush=True)
    for variant in datasets:
        baseline = case_manifests[(variant, "ppo")]
        prior_path = (ROOT / "results/backtests/crypto_rl_top10_20261024/manifest.json" if variant == "original"
                      else ROOT / "results/backtests/crypto_rl_ex_ada_20261024/ppo/manifest.json")
        prior = read(prior_path)
        for run_id, entry in baseline["run_results"].items():
            if run_id.endswith(tuple("__" + x for x in ("cash", "top10_equal", "top10_cap", "btc_eth_hold"))):
                check(entry["fingerprints"] == prior["run_results"][run_id]["fingerprints"],
                      variant + "/" + run_id + " exact same benchmark as frozen SB3 study")
                for algorithm in ("sac", "td3"):
                    check(entry["fingerprints"] == case_manifests[(variant, algorithm)]["run_results"][run_id]["fingerprints"],
                          variant + "/" + algorithm + "/" + run_id + " same four-file benchmark fingerprints")
    # Recheck preserved old outputs after all new case files were read and audited.
    for relative, expected in snapshot["files"].items():
        check(sha(ROOT / relative) == expected["sha256"], "old evidence unchanged after independent audit: " + relative)
    ledger_failures = sum(c["checks_failed"] for c in cases.values())
    result = {"created_at_utc": datetime.now(timezone.utc).isoformat(),
              "status": "passed" if not failures and not ledger_failures else "failed",
              "phase": "full_native_elegantrl_independent_audit", "runs_checked": sum(c["runs_checked"] for c in cases.values()),
              "checks_total": checks + sum(c["checks_total"] for c in cases.values()),
              "wrapper_checks_total": checks, "ledger_checks_total": sum(c["checks_total"] for c in cases.values()),
              "checks_failed": len(failures) + ledger_failures, "failures": failures,
              "cases": {key: {k: value[k] for k in ("status", "checks_total", "checks_failed", "runs_checked",
                                                  "max_absolute_reconstruction_error_usdt")} for key, value in cases.items()},
              "original_files_preserved": len(snapshot["files"]),
              "paired_actual_training_sampling": sampling,
              "all_native_and_sb3_seed_training_dates_identical": all(c["identical"] for group in sampling.values() for c in group),
              "auditor_source_sha256": sha(Path(__file__)),
              "original_reconstruct_source_sha256": sha(ROOT / "scripts/audit_crypto_rl_top10.py"),
              "checkpoint_policy": "weights_only native state dictionaries; optimizer and metadata excluded from allnetwork tensor digest, frozen checkpoint bytes checked separately",
              "timeout_policy": "Source-tested real successor replay; native PPO timeout bootstrap before reset, true terminal no bootstrap; frozen collector code and recorded episode boundaries checked",
              "original_or_production_results_modified": False}
    (output / "independent_audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/backtests/crypto_rl_elegantrl_20261024")
    args = parser.parse_args()
    result = audit_study(args.output_dir.resolve())
    print(json.dumps({k: result[k] for k in ("status", "runs_checked", "checks_total", "checks_failed")}))
    return int(bool(result["checks_failed"]))


if __name__ == "__main__":
    raise SystemExit(main())



"""Independent ex-ADA data/protocol wrapper around the immutable ledger auditor."""
from __future__ import annotations

import argparse
import ast
from contextlib import redirect_stdout
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.audit_crypto_rl_top10 import audit as audit_ledger, independent_training_candidates


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def stamp(value):
    t = pd.Timestamp(value)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def compare_training_episodes(original, current):
    """Describe actual sampling equality; a difference is disclosed, not tuned away."""
    def sequence(record):
        return [{"start": stamp(row["start"]).isoformat(),
                 "end_exclusive": stamp(row["end_exclusive"]).isoformat(),
                 "completed_until": stamp(row["completed_until"]).isoformat(),
                 "completed_steps": int(row["completed_steps"])}
                for row in record.get("training_episode_history", [])
                if int(row["completed_steps"]) > 0]
    before, after = sequence(original), sequence(current)
    differences = [{"episode": index, "original": before[index] if index < len(before) else None,
                    "ex_ada": after[index] if index < len(after) else None}
                   for index in range(max(len(before), len(after)))
                   if index >= len(before) or index >= len(after) or before[index] != after[index]]
    digest = lambda rows: hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {"seed": int(current["seed"]), "identical": not differences,
            "original_nonempty_episodes": len(before), "ex_ada_nonempty_episodes": len(after),
            "original_completed_steps": sum(row["completed_steps"] for row in before),
            "ex_ada_completed_steps": sum(row["completed_steps"] for row in after),
            "original_sequence_sha256": digest(before), "ex_ada_sequence_sha256": digest(after),
            "different_episode_count": len(differences), "differences": differences,
            "comparison_fields": ["start", "end_exclusive", "completed_until", "completed_steps"],
            "zero_step_auto_reset_records_excluded": True}


class Review:
    def __init__(self):
        self.checks = 0
        self.failures = []
        self.details = {}

    def check(self, condition, label):
        self.checks += 1
        if not bool(condition):
            self.failures.append(label)


def source_constants():
    """Read literal exclusion facts without importing the financial data module."""
    tree = ast.parse((ROOT / "scripts/crypto_rl_data.py").read_text(encoding="utf-8"))
    wanted = {"STABLE_SYMBOLS", "WRAPPED_SYMBOLS", "EVENTS"}
    found = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in wanted:
                    found[target.id] = ast.literal_eval(node.value)
    if set(found) != wanted:
        raise AssertionError("Cannot independently read frozen literal exclusion policy")
    return found


def preservation(review, output):
    snapshot = read(output / "original_research_preservation.json")
    for relative, expected in snapshot["files"].items():
        path = ROOT / relative
        review.check(path.is_file() and path.stat().st_size == expected["size_bytes"]
                     and sha(path) == expected["sha256"], "old frozen evidence preserved: " + relative)
    for relative in snapshot["originally_absent"]:
        review.check(not (ROOT / relative).exists(), "no backdated file in frozen study: " + relative)
    review.details["preservation_snapshot_sha256"] = sha(output / "original_research_preservation.json")
    review.details["original_files_checked"] = len(snapshot["files"])


def verify_selection(review, dataset, cache):
    constants = source_constants()
    evidence = read(cache / "universe_selection.json")
    old_evidence = {stamp(row["anchor"]): row for row in read(ROOT / "data/crypto_rl_top10/universe_selection.json")}
    universe = dataset["universe"].copy()
    universe["date"] = pd.to_datetime(universe.date, utc=True)
    universe["anchor"] = pd.to_datetime(universe.anchor, utc=True)
    universe["asset_key"] = universe.asset_key.astype(str)
    anchors = {stamp(row["anchor"]) for row in evidence}
    review.check(anchors == set(old_evidence) and len(evidence) == len(anchors) == 48,
                 "same 48 predetermined selection anchors")
    selected_at, changes = {}, []
    for record in evidence:
        anchor = stamp(record["anchor"])
        before_lag = anchor - pd.Timedelta(days=2)
        expected_date = before_lag.normalize() - pd.Timedelta(days=(before_lag.dayofweek + 1) % 7)
        date = stamp(record["snapshot"]["snapshot_date"])
        review.check(date == expected_date, str(anchor) + " latest Sunday with two-day lag")
        html_path = cache / "snapshots" / (date.strftime("%Y%m%d") + ".html.gz")
        raw_gzip = html_path.read_bytes()
        review.check(hashlib.sha256(raw_gzip).hexdigest() == record["snapshot"]["raw_gzip_sha256"]
                     == old_evidence[anchor]["snapshot"]["raw_gzip_sha256"], str(anchor) + " unchanged historical raw page")
        raw = gzip.decompress(raw_gzip)
        review.check(hashlib.sha256(raw).hexdigest() == record["snapshot"]["html_sha256"],
                     str(anchor) + " decompressed raw page hash")
        matched = re.search(rb'<script[^>]*id=["\x27]__NEXT_DATA__["\x27][^>]*>(.*?)</script>', raw, re.S)
        if matched is None:
            raise AssertionError("Historical provider page missing embedded source")
        page = json.loads(matched.group(1))
        review.check(page["props"]["pageProps"]["dateWithHyphens"] == date.strftime("%Y-%m-%d"),
                     str(anchor) + " page source date")
        state = page["props"]["initialState"]
        state = json.loads(state) if isinstance(state, str) else state
        candidates = []
        for asset in state["cryptocurrency"]["listingHistorical"]["data"]:
            quote = asset["quote"].get("USD", asset["quote"])
            cap = float(quote.get("marketCap", quote.get("market_cap", 0)))
            rank = int(asset.get("cmcRank", asset.get("rank", 0)))
            if rank > 0 and np.isfinite(cap) and cap > 0:
                candidates.append((rank, int(asset["id"]), cap, asset))
        candidates.sort(key=lambda x: (x[0], x[1]))
        decisions = {int(row["cmc_id"]): row for row in record["considered"]}
        review.check(len(decisions) == len(record["considered"]), str(anchor) + " no duplicate considered candidate")
        accepted, considered_ids = [], []
        for rank, identity, cap, asset in candidates:
            considered_ids.append(identity)
            if identity not in decisions:
                review.check(False, str(anchor) + " higher-ranked asset skipped without evidence: " + str(identity))
                break
            decision = decisions[identity]
            review.check(int(decision["rank"]) == rank and decision["symbol"] == asset["symbol"],
                         str(anchor) + " source rank/identity " + str(identity))
            tags = {str(x.get("slug", x.get("name", "")) if isinstance(x, dict) else x).lower()
                    for x in asset.get("tags", [])}
            stable = asset["symbol"].upper() in constants["STABLE_SYMBOLS"] or any("stablecoin" in x for x in tags)
            wrapped = (asset["symbol"].upper() in constants["WRAPPED_SYMBOLS"]
                       or "wrapped" in asset["name"].lower() or "staked" in asset["name"].lower()
                       or bool(tags & {"wrapped-tokens", "liquid-staking-derivatives", "liquid-restaking"}))
            if identity == 2010 or stable or wrapped:
                review.check(not decision.get("accepted", False) and bool(decision.get("excluded")),
                             str(anchor) + " required design/ADA exclusion " + str(identity))
                continue
            event = constants["EVENTS"].get(str(identity), {})
            outside = ((event.get("from") and anchor < stamp(event["from"]))
                       or (event.get("until") and anchor >= stamp(event["until"])))
            if outside:
                positive = False
            else:
                probe = read(Path(decision["evidence"]["path"]))
                bars = probe["data"]
                review.check(probe["sha256"] == decision["evidence"]["sha256"],
                             str(anchor) + " eligibility evidence digest " + str(identity))
                expected_end = (anchor - pd.Timedelta(hours=1)).value // 10**6
                recent = isinstance(bars, list) and len(bars) > 0 and int(bars[-1][0]) == expected_end
                positive = recent and any(float(bar[5]) > 0 for bar in bars)
                if isinstance(bars, list) and bars:
                    review.check(all(int(bar[0]) < anchor.value // 10**6 for bar in bars),
                                 str(anchor) + " no future eligibility quote " + str(identity))
            review.check(bool(decision.get("accepted", False)) == bool(positive),
                         str(anchor) + " actual historical trading eligibility " + str(identity))
            if positive:
                accepted.append({"asset_key": str(identity), "rank": rank, "cap": cap})
            if len(accepted) == 10:
                break
        ids = [row["asset_key"] for row in accepted]
        review.check(len(ids) == 10 and "2010" not in ids, str(anchor) + " ten non-ADA eligible coins")
        review.check([int(x["cmc_id"]) for x in record["considered"]] == considered_ids,
                     str(anchor) + " exact unskipped ranking prefix")
        review.check(ids == list(map(str, record["selected_asset_keys"])), str(anchor) + " independently rebuilt Top10")
        actual = universe.loc[universe.date.eq(anchor)].sort_values("slot")
        review.check(actual.asset_key.tolist() == ids, str(anchor) + " daily slots agree with source ranking")
        if len(actual) == len(accepted):
            review.check(actual.cmc_rank.astype(int).tolist() == [x["rank"] for x in accepted],
                         str(anchor) + " rank feature from source")
            review.check(np.allclose(actual.cap.to_numpy(float), [x["cap"] for x in accepted], rtol=1e-13, atol=1e-6),
                         str(anchor) + " historical cap feature from source")
        selected_at[anchor] = ids
        old_ids = list(map(str, old_evidence[anchor]["selected_asset_keys"]))
        review.check([x for x in old_ids if x != "2010"] == [x for x in ids if x in old_ids],
                     str(anchor) + " preserve every previously eligible non-ADA identity and rank")
        if "2010" not in old_ids:
            review.check(old_ids == ids, str(anchor) + " no replacement when ADA was already absent")
        changes.append({"anchor": str(anchor), "removed": sorted(set(old_ids) - set(ids)),
                        "added": sorted(set(ids) - set(old_ids))})
    ordered = sorted(anchors)
    for day, group in universe.groupby("date"):
        expected_anchor = max(t for t in ordered if t <= day)
        review.check(group.anchor.eq(expected_anchor).all()
                     and group.sort_values("slot").asset_key.tolist() == selected_at[expected_anchor],
                     str(day) + " monthly/boundary hold fixed until next predetermined anchor")
    review.details["pool_changes"] = changes


def preflight(review, output, dataset_path):
    preservation(review, output)
    parent_path = ROOT / "data/crypto_rl_top10/dataset.pkl.gz"
    old, new = pd.read_pickle(parent_path), pd.read_pickle(dataset_path)
    meta = read(dataset_path.parent / "manifest.json")
    review.check(sha(dataset_path) == meta["dataset_sha256"], "new prepared dataset digest")
    common = sorted(set(old["hourly_open"].columns.astype(str)) & set(new["hourly_open"].columns.astype(str)))
    for field in ("open", "high", "low", "close", "volume"):
        before, after = old["hourly_" + field], new["hourly_" + field]
        review.check(before.index.equals(after.index), field + " identical original full hourly date grid")
        review.check(after.index.is_unique and after.index.is_monotonic_increasing
                     and after.index.equals(pd.date_range(after.index[0], after.index[-1], freq="h")),
                     field + " complete ordered UTC hourly grid")
        review.check(np.array_equal(before.loc[:, common].to_numpy(), after.loc[:, common].to_numpy(), equal_nan=True),
                     field + " every shared price/volume/NaN exactly identical")
    dates = pd.to_datetime(new["universe"].date, utc=True)
    review.check(pd.DatetimeIndex(sorted(dates.unique())).equals(pd.date_range("2023-01-01", "2026-10-23", freq="D", inclusive="left", tz="UTC")),
                 "same full train/validation/test days, no evaluation-window deletion")
    review.check(not new["universe"].asset_key.astype(str).eq("2010").any(), "ADA never enters any train/validation/test slot")
    review.check(new["hourly_open"].index[0] == stamp("2022-11-01")
                 and new["hourly_open"].index[-1] + pd.Timedelta(hours=1) == stamp("2026-10-23"),
                 "same raw-data warmup and exclusive endpoint")
    # Characterize feature readiness without changing data or filtering evaluation.
    complete = None
    for field in ("open", "high", "low", "close", "volume"):
        frame = new["hourly_" + field]
        good = np.isfinite(frame) & (frame.ge(0) if field == "volume" else frame.gt(0))
        complete = good if complete is None else complete & good
    daily_complete = complete.resample("D").sum().eq(24)
    volume = new["hourly_volume"].resample("D").sum(min_count=24).where(daily_complete)
    ready = daily_complete.rolling(31, min_periods=31).sum().eq(31) & volume.rolling(30, min_periods=30).mean().gt(0)
    ready.index = ready.index + pd.Timedelta(days=1)
    eligible, covered, bad_days = independent_training_candidates(new, stamp("2023-01-01"), stamp("2025-07-25"))
    invalid_slots, incomplete_eval_slots = [], []
    for row in new["universe"].itertuples(index=False):
        day, key = stamp(row.date), str(row.asset_key)
        if day >= stamp("2025-07-25"):
            if not bool(daily_complete.loc[day, key]):
                incomplete_eval_slots.append({"date": str(day), "asset_key": key})
            if day not in ready.index or not bool(ready.loc[day, key]):
                invalid_slots.append({"date": str(day), "asset_key": key})
    review.check(not incomplete_eval_slots, "all selected validation/test days contain all 24 genuine OHLCV hours")
    review.check(bool(eligible) and all(start >= stamp("2023-01-01") and start + pd.Timedelta(days=128) <= stamp("2025-07-25") for start in eligible),
                 "complete training episodes remain confined to original TRAIN boundaries")
    review.details["data_readiness"] = {"raw_hours": len(new["hourly_open"]), "eligible_training_starts": len(eligible),
                                       "covered_training_days": len(covered), "bad_training_days": sorted(str(d) for d in bad_days),
                                       "invalid_31_prior_day_eval_slots": invalid_slots,
                                       "incomplete_eval_slots": incomplete_eval_slots,
                                       "feature_policy": "Only prior 31 complete UTC days; invalid feature slots remain uninvestable. No fabricated or backfilled quotes."}
    verify_selection(review, new, dataset_path.parent)
    review.details.update(dataset_sha256=sha(dataset_path), parent_dataset_sha256=sha(parent_path),
                          common_asset_keys=common, new_union_asset_keys=sorted(new["universe"].asset_key.astype(str).unique()),
                          ada_training_absence_evidence="Every possible historical slot is non-ADA; original reset starts in cash and unchanged rebalance_target accepts only current slot identities. Training fills are not persisted, so no claim of directly inspecting training fills.")
    return new


def final_checks(review, output, dataset_path, algorithms):
    spec = read(output / "frozen_spec.json")
    review.check(list(map(str, spec.get("excluded_asset_keys", []))) == ["2010"], "user-directed ADA exclusion frozen explicitly")
    review.check(spec.get("post_hoc_exclusion_based_on_observed_test") is True,
                 "post-hoc exclusion based on already observed test is disclosed")
    review.check(spec.get("parameter_search") is False, "no extra parameter search declared")
    for algorithm in algorithms:
        folder = output / algorithm
        manifest = read(folder / "manifest.json")
        old_folder = ROOT / "results/backtests" / ("crypto_rl_top10_20261024" if algorithm == "ppo" else "crypto_rl_a2c_20261024")
        original = read(old_folder / "manifest.json")
        runner = ROOT / "scripts/backtest_rl_ex_ada.py"
        hashes = {Path(k).name: v for k, v in manifest["source_sha256"].items()}
        review.check(hashes.get(runner.name) == sha(runner), algorithm + " new orchestration source frozen")
        for relative, expected in manifest["source_sha256"].items():
            review.check(sha(ROOT / relative) == expected, algorithm + " every frozen study source unchanged: " + relative)
        review.check(manifest["frozen_inputs"]["protocol_sha256"] == sha(output / "frozen_spec.json"),
                     algorithm + " same pretraining protocol SHA")
        review.check(manifest["frozen_inputs"]["dataset_sha256"] == sha(dataset_path),
                     algorithm + " immutable ex-ADA dataset SHA")
        for key in ("train_start", "train_end", "validation_start", "validation_end", "test_start", "test_end",
                    "costs", "capital", "seeds", "timesteps_per_seed", "net_arch", "cash_target_floor", "single_coin_target_cap"):
            review.check(manifest["config"].get(key) == original["config"].get(key), algorithm + " original fixed protocol " + key)
        for key in ("algorithm_parameters", "learning_rate", "n_steps", "batch_size", "n_epochs", "gamma", "gae_lambda",
                    "clip_range", "ent_coef", "reward", "feature_normalization", "train_episode_days", "parameter_search"):
            review.check(manifest["config"].get(key) == original["config"].get(key), algorithm + " unchanged algorithm parameter " + key)
        review.check(sorted(m["seed"] for m in manifest["models"]) == [0, 1, 2, 3, 4], algorithm + " all same five seeds")
        old_seeds = {int(model["seed"]): model for model in original["models"]}
        episode_comparisons = []
        for model in manifest["models"]:
            comparison = compare_training_episodes(old_seeds[int(model["seed"])], model)
            review.check(comparison["original_nonempty_episodes"] > 0
                         and comparison["ex_ada_nonempty_episodes"] > 0
                         and comparison["original_completed_steps"] == comparison["ex_ada_completed_steps"] == 32768,
                         algorithm + " paired sampling records cover original fixed budget, seed " + str(model["seed"]))
            episode_comparisons.append(comparison)
        review.details[algorithm + "_paired_training_sampling"] = {
            "all_seeds_same_actual_training_episodes": all(row["identical"] for row in episode_comparisons),
            "seeds": episode_comparisons,
            "interpretation": "This is observed sampling equality, not a prerequisite used to tune or retrain. Any difference is retained explicitly."}
        expected_runs = {f"{split}__{scenario}__{strategy}" + (f"__seed{seed}" if seed is not None else "")
                         for split in ("validation", "test") for scenario in ("base", "stress")
                         for strategy, seed in ([(algorithm, seed) for seed in range(5)]
                                                 + [(algorithm + "_ensemble", None)]
                                                 + [(x, None) for x in ("cash", "top10_equal", "top10_cap", "btc_eth_hold")])}
        review.check(set(manifest["run_results"]) == expected_runs, algorithm + " all forty intended evaluations")
        for run_id, run in manifest["run_results"].items():
            fills = pd.read_csv(folder / "runs" / run_id / "fills.csv", dtype={"asset_key": str}, float_precision="round_trip")
            review.check(not fills.asset_key.eq("2010").any(), run_id + " no ADA fill")
            weights = pd.read_csv(folder / "runs" / run_id / "weights.csv", dtype={"asset_key": str}, float_precision="round_trip")
            review.check(not weights.asset_key.eq("2010").any(), run_id + " no ADA target/inventory row")
            holdings = run["metadata"].get("final_holdings", [])
            review.check(all(str(p["asset_key"]) != "2010" for p in holdings), run_id + " no terminal ADA inventory")
        with (folder / "audit.log").open("w", encoding="utf-8") as stream, redirect_stdout(stream):
            ledger = audit_ledger(folder, dataset_path)
        (folder / "independent_audit.json").write_text(json.dumps(ledger, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        review.check(ledger["status"] == "passed" and ledger["checks_failed"] == 0 and ledger["runs_checked"] == 40,
                     algorithm + " independent original ledger audit passed")
        review.details[algorithm + "_ledger_audit"] = {k: ledger[k] for k in ("status", "checks_total", "checks_failed",
                                                                           "runs_checked", "max_absolute_reconstruction_error_usdt")}
    if set(algorithms) == {"ppo", "a2c"}:
        ppo, a2c = [read(output / algorithm / "manifest.json") for algorithm in ("ppo", "a2c")]
        common = [r for r in ppo["run_results"] if any(r.endswith("__" + x) for x in ("cash", "top10_equal", "top10_cap", "btc_eth_hold"))]
        review.check(len(common) == 16, "sixteen common benchmark evaluations")
        for run_id in common:
            review.check(ppo["run_results"][run_id]["fingerprints"] == a2c["run_results"][run_id]["fingerprints"],
                         run_id + " identical benchmark economics across algorithms")
    preservation(review, output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/backtests/crypto_rl_ex_ada_20261024")
    parser.add_argument("--dataset", type=Path, default=ROOT / "data/crypto_rl_top10_ex_ada/dataset.pkl.gz")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--algorithm", choices=("ppo", "a2c", "all"), default="all")
    args = parser.parse_args()
    output, dataset = args.output_dir.resolve(), args.dataset.resolve()
    review = Review()
    preflight(review, output, dataset)
    if not args.preflight:
        final_checks(review, output, dataset, ("ppo", "a2c") if args.algorithm == "all" else (args.algorithm,))
    ledgers = [value for name, value in review.details.items() if name.endswith("_ledger_audit")]
    ledger_checks = sum(value["checks_total"] for value in ledgers)
    ledger_failures = sum(value["checks_failed"] for value in ledgers)
    result = {"created_at_utc": datetime.now(timezone.utc).isoformat(),
              "status": "passed" if not review.failures and not ledger_failures else "failed",
              "checks_total": review.checks + ledger_checks, "wrapper_checks_total": review.checks,
              "ledger_checks_total": ledger_checks, "checks_failed": len(review.failures) + ledger_failures,
              "runs_checked": sum(value["runs_checked"] for value in ledgers), "failures": review.failures,
              "phase": "data_preflight" if args.preflight else "full_independent_audit",
              "details": review.details, "auditor_source_sha256": sha(Path(__file__)),
              "original_auditor_source_sha256": sha(ROOT / "scripts/audit_crypto_rl_top10.py"),
              "production_or_original_results_modified": False}
    target = output / ("preflight_independent_audit.json" if args.preflight else "independent_audit.json")
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("status", "phase", "checks_total", "checks_failed", "failures")}))
    return int(bool(review.failures))


if __name__ == "__main__":
    raise SystemExit(main())

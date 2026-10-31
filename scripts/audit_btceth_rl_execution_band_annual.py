"""Independent audit for a separately authorized, already-seen annual replay.

Preparation of this source does not authorize execution. No policy, trading
environment or execution ledger is imported. This audit never edits the frozen
5pp validation experiment, its source or the unsuccessful first annual goal.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import audit_btceth_rl_execution_band as band
from scripts import audit_btceth_rl_execution_band_gpu_guard as gpu_proof
from scripts.audit_btceth_rl import BASELINES, Checks, _check_summary, _audit_run, read_csv, read_equity, reconstruct_fills, verify_saved_fingerprints
from scripts.audit_btceth_rl_exposure_cap import CASE_FILES, COSTS, GROUP_FILES, SEEDS, SYMBOLS, attach_inventory_replay, audit_case as audit_reference_case, compare_group_tables, load_market, read_json, safe_file, sha, utc

TEST = ("2025-10-23T00:00:00Z", "2026-10-23T00:00:00Z")
FAMILY, STRATEGY, BAND, MULTIPLIER = band.FAMILY, band.STRATEGY, .05, .75
FLAGS = band.FLAGS
SOURCE_FILES = (*band.SOURCE_FILES, "scripts/btceth_rl_execution_band_gpu_guard.py",
    "scripts/audit_btceth_rl_execution_band_gpu_guard.py", "scripts/btceth_rl_execution_band_annual.py",
    "scripts/audit_btceth_rl_execution_band_annual.py")
TARGETS = {"primary_base_net_return_at_least": .05, "primary_base_max_drawdown_magnitude_at_most": .20,
    "primary_stress_net_return_at_least": 0., "primary_beats_equal_hold_return_and_drawdown": True,
    "positive_base_training_seeds_at_least": 3, "stronger_claim_requires_beating_simple_trend_controls": True,
    "targets_are_not_guarantees": True}


class Inputs(band.Inputs):
    def __init__(self, annual, validation, main, reference, source, checks):
        self.roots = {key: Path(value).resolve() for key, value in
            (("annual", annual), ("validation", validation), ("main", main), ("reference", reference), ("source", source))}
        if len(set(self.roots.values())) != len(self.roots):
            raise ValueError("Annual, validation, main, reference and source roots must be distinct")
        self.hashes = {key: {} for key in self.roots}
        self.checks = checks


def baseline_files():
    folder = "evaluation/test/baselines"
    return {*(f"{folder}/{name}" for name in GROUP_FILES),
        *(f"{folder}/{cost}/{key}/{name}" for cost in COSTS for key in BASELINES for name in CASE_FILES[:6])}


def case_group_files(split):
    folder = f"{split}/balanced"
    return {*(f"{folder}/{name}" for name in GROUP_FILES),
        *(f"{folder}/{cost}/{key}/{name}" for cost in COSTS
          for key in (*map(str, SEEDS), "ensemble") for name in CASE_FILES)}


def validation_files():
    return case_group_files("validation") | {"protocol.json", "protocol_receipt.json", "summary.csv",
        "daily_equity.csv", "monthly_returns.csv", "validation_completion.json", "runner_process.json",
        "validation.log", "selection.json", "selection_receipt.json", "comparison.json",
        "adapter_protocol.json", "adapter_protocol_receipt.json", "adapter_runtime.json"}


def verify_protocol(inputs, dataset_path):
    checks, root = inputs.checks, inputs.roots["annual"]
    p, receipt = inputs.json("annual", "protocol.json"), inputs.json("annual", "protocol_receipt.json")
    spec = inputs.json("main", "frozen_spec.json")
    fixed = dict(family=FAMILY, algorithm="tqc", profile="balanced", test_start=TEST[0], end_exclusive=TEST[1],
        capital=10000., costs=COSTS, seeds=list(SEEDS), cash_floor=.05, max_asset_weight=.95,
        drawdown_penalty=.25, rebalance_band=BAND, exposure_multiplier=MULTIPLIER, new_training=0,
        seed_selection=False, selection_uses_test=False, annual_evaluation_enabled=True,
        research_targets=TARGETS)
    for name, expected in fixed.items():
        checks.check(p.get(name) == expected, "Separately declared known-year annual variant:" + name)
    checks.check(all(p.get(flag) is True for flag in FLAGS)
                 and tuple(spec["test"]) == TEST and tuple(spec["validation"]) == band.VALIDATION
                 and spec.get("research_targets") == TARGETS,
                 "Already-seen annual disclosure, original windows and original performance gates remain unchanged")
    checks.check(receipt.get("protocol_file") == "protocol.json"
                 and receipt.get("protocol_sha256") == sha(root / "protocol.json")
                 and utc(p["prepared_at"]) <= utc(receipt["sealed_at"]),
                 "New annual protocol has its own immutable whole-file seal")
    sources = p.get("source_sha256", {})
    inputs.pin_map("source", sources, SOURCE_FILES, "Actual annual/original/adapter source bytes")
    checks.check(set(sources) == set(SOURCE_FILES)
                 and all(sources[name] == spec["code_sha256"][name] for name in band.FINANCIAL),
                 "Exact14-source DAG and unchanged four financial sources")
    for name, expected in sources.items():
        copied = inputs.capture("annual", "sources/" + name)
        checks.check(sha(copied) == expected, "Archived full annual source DAG:" + name)
    audit_file = p["validation_audit_file"]
    required_main = baseline_files() | {"frozen_spec.json", "training_completion.json", "runtime.json", "independent_audit.json",
        *(f"training/tqc__balanced__seed{s}/{name}" for s in SEEDS for name in ("record.json", "model.zip"))}
    required_reference = case_group_files("test") | {"protocol.json", "protocol_receipt.json",
        "independent_audit.json", "goal_candidate_evidence.json"}
    original = p.get("original_inputs", {})
    for key, required in (("main", required_main), ("reference", required_reference),
                          ("validation", validation_files() | {audit_file})):
        inputs.pin_map(key, original.get(key), required, "Immutable prior research:" + key)
    previous_audit = inputs.json("validation", audit_file)
    checks.check(previous_audit.get("passed") is True and previous_audit.get("status") == "passed"
                 and previous_audit.get("validation_complete") is True
                 and previous_audit.get("selection_complete") is True and previous_audit.get("test_complete") is False
                 and previous_audit.get("run_count") == previous_audit.get("repeat_count")
                 == previous_audit.get("comparator_run_count") == 12
                 and previous_audit.get("failure_count") == 0 and previous_audit.get("failures") == []
                 and previous_audit.get("resource_adapter_disclosed") is True,
                 "All validation accounts, actual adapter proofs and sealed comparison passed before this annual")
    # The previous wrapper used four root names. Resolve the same immutable
    # files here through explicit local roots, never through remote absolutes.
    mapping = {"extension": "validation", "main": "main", "reference": "reference", "source": "source"}
    prior_hashes = previous_audit.get("input_sha256", {})
    checks.check(set(prior_hashes) == set(mapping)
                 and validation_files().issubset(prior_hashes.get("extension", {}))
                 and set(SOURCE_FILES[:-2]).issubset(prior_hashes.get("source", {})),
                 "Passed validation audit retains complete case/adapter/source input hashes, not a bare success flag")
    for key, values in prior_hashes.items():
        for name, expected in values.items():
            checks.check(sha(inputs.capture(mapping[key], name)) == expected,
                         "Actual complete validation audit input unchanged:" + key + ":" + name)
    selected = inputs.json("validation", "selection.json")
    selection_receipt = inputs.json("validation", "selection_receipt.json")
    checks.check(selection_receipt.get("selection_file") == "selection.json"
                 and selection_receipt.get("selection_sha256") == sha(inputs.roots["validation"] / "selection.json")
                 == p.get("validation_selection_sha256")
                 and selected.get("selection_uses_test") is False and selected.get("seed_selection") is False
                 and all(selected.get(flag) is True for flag in FLAGS), "Original fixed validation decision is preserved")
    new_summary = inputs.csv("validation", "summary.csv")
    old_validation = inputs.csv("reference", "validation/balanced/summary.csv")
    candidates = band.compare_validation_candidates(new_summary, old_validation)
    by_family = {item["family"]: item for item in candidates}
    recorded = selected.get("candidates", [])
    checks.check(len(recorded) == 2 and {item.get("family") for item in recorded} == {"cap075", FAMILY},
                 "Exactly two original validation variants remain in the frozen decision")
    for item in recorded:
        expected = by_family.get(item.get("family"))
        if expected is not None:
            checks.close(item.get("score"), expected["score"], "Independent declared validation comparison score", atol=1e-12)
    frozen_order = sorted(recorded, key=lambda item: (-item["score"], 0 if item["family"] == "cap075" else 1))
    winner = selected.get("candidate", {})
    checks.check(len(frozen_order) == 2 and recorded == frozen_order and winner == frozen_order[0]
                 and frozen_order[0]["family"] == FAMILY
                 and frozen_order[0]["score"] > frozen_order[1]["score"]
                 and winner.get("family") == FAMILY and winner.get("algorithm") == "tqc"
                 and winner.get("profile") == "balanced" and winner.get("rebalance_band") == BAND,
                 "Only a strict same-score5pp validation winner may enter this separate annual trial")
    checks.close(winner.get("score"), by_family[FAMILY]["score"], "Independent fixed validation score before annual", atol=1e-12)
    checks.check(utc(selection_receipt["sealed_at"]) <= utc(previous_audit["audited_at"])
                 <= utc(p["prepared_at"]), "Sealed validation and independent post-selection audit precede annual declaration")
    validation_protocol = inputs.json("validation", "protocol.json")
    records, models = {}, []
    for seed in SEEDS:
        record_file, model_file = (f"training/tqc__balanced__seed{seed}/{name}" for name in ("record.json", "model.zip"))
        record = inputs.json("main", record_file)
        checks.check((record.get("algorithm"), record.get("profile"), record.get("seed")) == ("tqc", "balanced", seed)
                     and record.get("model_relative_path") == model_file
                     and record.get("model_sha256") == sha(inputs.capture("main", model_file))
                     and record.get("frozen_spec_sha256") == sha(inputs.roots["main"] / "frozen_spec.json")
                     and record.get("actual_timesteps") == 32768, "Same original trained checkpoint seed:" + str(seed))
        models.append(dict(profile="balanced", seed=seed, record_file=record_file, model_file=model_file,
            record_sha256=sha(inputs.roots["main"] / record_file), model_sha256=record["model_sha256"],
            parameter_sha256=record["parameter_sha256"]))
        records[seed] = record
    checks.check(p.get("models") == validation_protocol.get("models") == models,
                 "All five original models, record bytes and trained parameters reused; no retraining")
    runtime = inputs.json("main", "runtime.json")
    gpus = p.get("physical_gpus", [])
    checks.check(gpus == runtime.get("physical_gpus") and len(gpus) == 2
                 and len({gpu["uuid"] for gpu in gpus}) == 2
                 and all("V100" in gpu["name"] for gpu in gpus)
                 and p.get("gpu_cost_assignments") == {cost: gpu["uuid"] for cost, gpu in zip(COSTS, gpus)}
                 and p.get("runtime_packages") == spec["runtime_packages"], "Original two physical cards, cost workers and software versions")
    dataset_path = Path(dataset_path)
    checks.check(sha(dataset_path) == p.get("dataset_sha256") == spec["dataset_sha256"]
                 and sha(dataset_path.parent / "manifest.json") == p.get("dataset_manifest_sha256")
                 == spec["dataset_manifest_sha256"], "Original true market data and manifest bytes")
    old_goal = verify_original_goal(inputs)
    main_audit = inputs.json("main", "independent_audit.json")
    checks.check(main_audit.get("passed") is True and main_audit.get("status") == "passed"
                 and main_audit.get("test_complete") is True and main_audit.get("run_count") == 172
                 and main_audit.get("failures") == [], "Original funded controls belong to passed172 study")
    if checks.failed:
        raise ValueError("Annual protocol/model/validation/source gates failed before new financial reconstruction")
    return p, spec, records, main_audit, old_goal


def validate_group_rows(frame, *, dates=TEST):
    required = {"run_id", "algorithm", "profile", "family", "strategy", "scenario", "split", "seed",
        "primary_policy", "initial_capital", "start", "end_exclusive", "net_return", "max_drawdown",
        "final_equity", "fees", "exposure_multiplier", "rebalance_band"}
    if not required.issubset(frame) or len(frame) != 12 or frame.run_id.duplicated().any():
        raise ValueError("Exactly twelve annual seed/ensemble cases at the two fixed costs required")
    fixed = {"algorithm": "tqc", "profile": "balanced", "family": FAMILY, "strategy": STRATEGY,
        "split": "test", "initial_capital": 10000., "exposure_multiplier": MULTIPLIER, "rebalance_band": BAND}
    if any(not frame[name].eq(value).all() for name, value in fixed.items()):
        raise ValueError("Annual trial may only reuse the selected five capped TQC balanced models and5pp band")
    if (not pd.to_datetime(frame.start, utc=True).eq(utc(dates[0])).all()
            or not pd.to_datetime(frame.end_exclusive, utc=True).eq(utc(dates[1])).all()):
        raise ValueError("Annual replay cannot shift dates, omit days or substitute validation")
    expected = {f"test__{cost}__{FAMILY}__tqc__balanced__{key}" for cost in COSTS
                for key in (*map(str, SEEDS), "ensemble")}
    if set(frame.run_id) != expected or not frame.primary_policy.map(lambda x: isinstance(x, (bool, np.bool_))).all():
        raise ValueError("Fixed annual case identities and actual ensemble flags required")
    for cost in COSTS:
        rows = frame.loc[frame.scenario.eq(cost)]
        primary, seeds = rows.loc[rows.primary_policy], rows.loc[~rows.primary_policy]
        if len(primary) != 1 or primary.seed.notna().any() or len(seeds) != 5 or set(seeds.seed) != set(SEEDS):
            raise ValueError("No seed selection or missing stress scenario")
    numbers = frame[["net_return", "max_drawdown", "final_equity", "fees"]].to_numpy(dtype=float)
    if (not np.isfinite(numbers).all() or (frame.net_return <= -1).any() or (frame.max_drawdown > 0).any()
            or (frame.max_drawdown <= -1).any() or (frame.fees < 0).any() or (frame.final_equity <= 0).any()
            or (frame.final_equity / 10000 - 1 - frame.net_return).abs().max() > 1e-10):
        raise ValueError("Inconsistent annual summary")


def audit_case(directory, dataset, row, records, protocol, checks, *, features, dates=TEST):
    """Dynamic dates are local arguments; no mutation of frozen b4d globals."""
    start, end = dates
    label, cost = row["run_id"], row["scenario"]
    seeds = list(SEEDS) if row["primary_policy"] else [int(row["seed"])]
    equity, fills = read_equity(directory / "equity_1h.csv.gz"), read_csv(directory / "fills.csv")
    metadata, weights = read_json(directory / "metadata.json"), read_csv(directory / "weights.csv")
    fingerprints = verify_saved_fingerprints(directory, checks, label)
    rebuilt = reconstruct_fills(dataset, fills, equity, metadata, start=start, end=end,
                               **COSTS[cost], checks=checks, label=label)
    if rebuilt is None:
        raise ValueError("Incomplete independent annual hourly account:" + label)
    inventory = attach_inventory_replay(rebuilt, fills, dataset["hourly_close"], checks=checks, label=label, metadata=metadata)
    _check_summary(row, rebuilt["metrics"], checks, label)
    parameters = [records[seed]["parameter_sha256"] for seed in seeds]
    checks.check(metadata.get("algorithm") == "tqc" and metadata.get("profile") == "balanced"
                 and metadata.get("family") == FAMILY and metadata.get("model_count") == len(seeds),
                 label + ": selected fixed model population")
    checks.check(metadata.get("parameter_sha256_before") == metadata.get("parameter_sha256_after") == parameters,
                 label + ": same five original trained parameters before/after inference")
    for name, expected in (("rebalance_band", BAND), ("cash_floor_target", .05),
                           ("max_asset_weight_target", .95), ("drawdown_penalty", .25), ("exposure_multiplier", MULTIPLIER)):
        checks.close(metadata.get(name), expected, label + ": unchanged financial parameter:" + name, atol=1e-15, rtol=0)
    checks.check(metadata.get("old_actions_or_equity_scaled") is False
                 and metadata.get("band_change_only") is True
                 and all(metadata.get(flag) is True for flag in FLAGS),
                 label + ": true funded5pp account and already-seen-year disclosures")
    trace = read_csv(directory / "member_targets.csv")
    band.audit_member_targets(trace, weights, rebuilt, fills, start=start, end=end, seeds=seeds, checks=checks,
        label=label, band=BAND, market_features=features, require_observation_values=True)
    rebuilt["costs"] = COSTS[cost]
    band.audit_execution_quantities(dataset, trace, rebuilt, fills, start=start, end=end, band=BAND, checks=checks, label=label)
    replay = read_json(directory / "replay.json")
    checks.check(replay.get("run_id") == label and replay.get("passed") is True and replay.get("repeat_exact") is True
                 and replay.get("fingerprints") == replay.get("repeated_fingerprints") == fingerprints,
                 label + ": exact independent repeated financial fingerprints")
    trace_sha = sha(directory / "member_targets.csv")
    checks.check(replay.get("member_targets_sha256") == replay.get("repeated_member_targets_sha256") == trace_sha,
                 label + ": exact repeated raw-action/common-state trace")
    checks.check(replay.get("parameter_sha256_before") == replay.get("parameter_sha256_after") == parameters
                 and replay.get("model_records") == [model for model in protocol["models"] if model["seed"] in seeds],
                 label + ": repeated original checkpoint and parameter byte identities")
    device = replay.get("device", {})
    checks.check(device.get("device") == "cuda:0" and device.get("capability") == [7, 0]
                 and "V100" in device.get("name", "")
                 and device.get("visible_gpu_uuid") == protocol["gpu_cost_assignments"][cost],
                 label + ": original UUID-isolated V100 assigned to this cost scenario")
    daily = rebuilt["equity"].equity.resample("D").last()
    dd = rebuilt["drawdown"].resample("D").last()
    history = pd.DataFrame({"timestamp": [utc(start), *list(daily.index + pd.Timedelta(days=1))],
        "equity": [10000., *daily.tolist()], "net_return": [0., *(daily / 10000 - 1).tolist()],
        "drawdown": [0., *dd.tolist()]})
    boundary = read_equity(directory / "boundary_equity.csv")
    checks.check(boundary.index.equals(pd.DatetimeIndex(history.timestamp)), label + ": exact annual daily boundary dates")
    checks.vector_close(boundary.equity, history.equity, label + ": independent annual boundary NAV", atol=1e-8)
    rewards, previous_nav, previous_dd = [0.], 10000., 0.
    for _, hours in rebuilt["equity"].groupby(rebuilt["equity"].index.normalize()):
        deepest, nav = -float(rebuilt["drawdown"].loc[hours.index].min()), float(hours.equity.iloc[-1])
        rewards.append(100 * (math.log(nav / previous_nav) - .25 * max(deepest - previous_dd, 0.)))
        previous_nav, previous_dd = nav, -float(rebuilt["drawdown"].loc[hours.index[-1]])
    checks.vector_close(boundary.reward, rewards, label + ": same hourly-derived balanced reward", atol=1e-9)
    run = {"run_id": label, "hours": len(equity), "reconstructed": True, "repeat_exact": True,
        "fingerprints": fingerprints, "member_targets_sha256": trace_sha, "inventory_replay": inventory,
        **{key: rebuilt["metrics"][key] for key in ("net_return", "max_drawdown", "fees", "final_equity")}}
    return run, history, rebuilt["monthly"]


def audit_baselines(inputs, dataset, spec, original_audit, *, dates=TEST):
    """Independently replay all14 funded original controls, preserving originals."""
    root, folder, checks = "main", "evaluation/test/baselines", inputs.checks
    for name in GROUP_FILES:
        inputs.capture(root, folder + "/" + name)
    frame = inputs.csv(root, folder + "/summary.csv")
    daily, monthly = inputs.csv(root, folder + "/daily_equity.csv"), inputs.csv(root, folder + "/monthly_returns.csv")
    expected = {f"test__{cost}__{name}" for cost in COSTS for name in BASELINES}
    if len(frame) != 14 or set(frame.run_id) != expected or frame.run_id.duplicated().any():
        raise ValueError("Exactly the seven original funded baselines at both costs are required")
    checks.check(set(daily.run_id) == set(monthly.run_id) == expected, "Original baseline table coverage contains no hidden/omitted controls")
    results = {}
    for row in frame.to_dict("records"):
        label, cost, name = row["run_id"], row["scenario"], row["strategy"]
        if not checks.check(cost in COSTS and name in BASELINES
            and label == f"test__{cost}__{name}" and row["split"] == "test"
            and row["initial_capital"] == 10000. and utc(row["start"]) == utc(dates[0])
            and utc(row["end_exclusive"]) == utc(dates[1]), label + ": original baseline identity/window/cost/capital"):
            continue
        relative = folder + "/" + cost + "/" + name
        for filename in CASE_FILES[:6]:
            inputs.capture(root, relative + "/" + filename)
        result = _audit_run(safe_file(inputs.roots[root], relative), dataset, spec, {}, row, monthly, daily,
            split="test", scenario=cost, algorithm="baseline", profile="baseline", key=name, checks=checks)
        old = original_audit.get("runs", {}).get(label, {})
        hours = int((utc(dates[1]) - utc(dates[0])) / pd.Timedelta(hours=1))
        checks.check(result.get("reconstructed") is True and result.get("hours") == hours
                     and old.get("reconstructed") is True and old.get("hours") == hours
                     and old.get("fingerprints") == result.get("fingerprints"),
                     label + ": unchanged control belongs to the passed172 original audit")
        result.update(source_root="main", relative_directory=relative, fees=float(row["fees"]),
            final_equity=float(row["final_equity"]), net_return=float(result["net_return"]),
            max_drawdown=float(result["max_drawdown"]))
        results[label] = result
    return frame, results


def verify_original_goal(inputs):
    goal = inputs.json("reference", "goal_candidate_evidence.json")
    original = inputs.json("reference", "independent_audit.json")
    checks = inputs.checks
    checks.check(goal.get("audit_sha256") == sha(inputs.roots["reference"] / "independent_audit.json")
                 and original.get("passed") is True and original.get("status") == "passed"
                 and original.get("test_complete") is True and original.get("run_count") == 36,
                 "First capped annual account and its actual goal evidence remain immutable")
    checks.check(goal.get("candidate") == {"family": "cap075", "algorithm": "tqc", "profile": "balanced"}
                 and goal.get("goal_performance_achieved") is False
                 and goal.get("metrics", {}).get("positive_base_training_seeds") == 2
                 and goal.get("gates", {}).get("at_least_three_positive_training_seeds") is False,
                 "Original2/5-positive-seed goal remained unsuccessful; it is never overwritten by the new replay")
    return goal


def audit_group(inputs, dataset, records, protocol, *, features=None, reference=False):
    root, folder, checks = ("reference" if reference else "annual"), "test/balanced", inputs.checks
    for name in GROUP_FILES:
        inputs.capture(root, folder + "/" + name)
    frame = inputs.csv(root, folder + "/summary.csv")
    daily, monthly = inputs.csv(root, folder + "/daily_equity.csv"), inputs.csv(root, folder + "/monthly_returns.csv")
    if reference:
        from scripts.audit_btceth_rl_exposure_cap import validate_group_rows as validate_old_group
        validate_old_group(frame, split="test", algorithm="tqc", profile="balanced", multiplier=MULTIPLIER)
    else:
        validate_group_rows(frame)
    completion = inputs.json(root, folder + "/completion.json")
    checks.check(completion.get("case_count") == 12, root + ": complete original twelve-case funded group")
    if not reference:
        checks.check(completion.get("protocol_sha256") == sha(inputs.roots["annual"] / "protocol.json"),
                     "Annual group used this separately sealed protocol")
    proofs = {row.get("run_id"): row for row in completion.get("repeats", [])}
    checks.check(len(proofs) == 12 and set(proofs) == set(frame.run_id),
                 root + ": all twelve exact financial/raw-action replay receipts")
    expected = {}
    original = inputs.json("reference", "independent_audit.json") if reference else None
    for row in frame.to_dict("records"):
        key = "ensemble" if row["primary_policy"] else str(int(row["seed"]))
        relative = f"{folder}/{row['scenario']}/{key}"
        for name in CASE_FILES:
            inputs.capture(root, relative + "/" + name)
        directory = safe_file(inputs.roots[root], relative)
        if reference:
            run, history, months = audit_reference_case(directory, dataset, row, dates=TEST,
                records={("tqc", "balanced", seed): record for seed, record in records.items()},
                profile="balanced", key=key, checks=checks, capped=True, protocol=protocol)
            old = original.get("runs", {}).get(row["run_id"], {})
            checks.check(old.get("reconstructed") is True and old.get("hours") == 8760
                         and old.get("fingerprints") == run["fingerprints"],
                         "Original capped annual reference remains in passed36 audit:" + row["run_id"])
        else:
            run, history, months = audit_case(directory, dataset, row, records, protocol, checks, features=features)
        run.update(source_root=root, relative_directory=relative)
        proof = proofs.get(row["run_id"], {})
        checks.check(proof.get("passed") is True and proof.get("fingerprints") == run["fingerprints"]
                     and proof.get("member_targets_sha256") == sha(directory / "member_targets.csv"),
                     row["run_id"] + ": group receipt binds actual account and full observation trace")
        expected[row["run_id"]] = (run, history, months)
    compare_group_tables(frame, daily, monthly, expected, checks, root)
    return frame, daily, monthly, expected


def verify_execution_chain(inputs, p):
    """Require a new explicit authorization and a distinct annual process."""
    checks, root = inputs.checks, inputs.roots["annual"]
    authorization = inputs.json("annual", "authorization.json")
    marker = inputs.json("annual", "test_started.json")
    process, completion = inputs.json("annual", "runner_process.json"), inputs.json("annual", "test_completion.json")
    receipt = inputs.json("annual", "protocol_receipt.json")
    selected = inputs.json("validation", "selection_receipt.json")
    selection_sha = sha(inputs.roots["validation"] / "selection.json")
    auth_sha, protocol_sha = sha(root / "authorization.json"), sha(root / "protocol.json")
    checks.check(authorization.get("annual_evaluation_authorized") is True
                 and authorization.get("selection_sha256") == selection_sha
                 and authorization.get("financial_parameters_fixed") is True
                 and authorization.get("no_live_trading") is True
                 and all(authorization.get(flag) is True for flag in FLAGS)
                 and p.get("authorization_sha256") == auth_sha,
                 "Only a separately authorized fixed known-year annual replay is permitted")
    checks.check(marker.get("protocol_sha256") == protocol_sha
                 and marker.get("authorization_sha256") == auth_sha
                 and marker.get("validation_selection_sha256") == selection_sha,
                 "Annual start marker binds the sealed validation, annual declaration and explicit authorization")
    command = process.get("command", [])
    checks.check(process.get("phase") == "test" and process.get("status") == "completed"
                 and process.get("exit_code") == 0 and process.get("protocol_sha256") == protocol_sha
                 and process.get("source_sha256") == p["source_sha256"]["scripts/btceth_rl_execution_band_annual.py"],
                 "True new annual supervisor reached completed0 under this source")
    checks.check(len(command) >= 5 and str(command[2]).replace("\\", "/").endswith("/btceth_rl_execution_band_annual.py")
                 and "--phase" in command and command[command.index("--phase") + 1] == "_test",
                 "Actual annual private child command is explicitly recorded")
    checks.check(isinstance(process.get("supervisor_pid"), int) and process["supervisor_pid"] > 0
                 and isinstance(process.get("child_pid"), int) and process["child_pid"] > 0
                 and process["supervisor_pid"] != process["child_pid"],
                 "Distinct real annual supervisor and child handles")
    checks.check(process.get("log_relative_path") == "test.log"
                 and process.get("log_sha256") == sha(inputs.capture("annual", "test.log")),
                 "Actual annual terminal log bound by final process evidence")
    checks.check(completion.get("case_count") == completion.get("repeat_count") == 12
                 and completion.get("protocol_sha256") == protocol_sha,
                 "Only twelve annual cases plus twelve exact repeats completed")
    checks.check(utc(selected["sealed_at"]) <= utc(authorization["authorized_at"]) <= utc(p["prepared_at"])
                 <= utc(receipt["sealed_at"]) <= utc(process["started_at"]) <= utc(marker["at"])
                 <= utc(completion["at"]) <= utc(process["completed_at"]),
                 "Validation seal and authorization precede the new annual process and its first dispatch")
    compatibility = inputs.json("validation", "adapter_protocol.json")
    checks.check(p.get("allowed_instance") == compatibility["allowed_instance"]
                 and p.get("allowed_gpu_rows") == compatibility["allowed_gpu_rows"],
                 "Only the same separately reviewed existing UI instance may pass annual GPU preflight")
    runtime = inputs.json("annual", "resource_runtime.json")
    checks.check(runtime.get("status") == "completed" and runtime.get("error") is None
                 and runtime.get("financial_rules_changed") is False,
                 "Actual annual resource checks complete without changing finance")
    identity, allowed = compatibility["allowed_instance"], compatibility["allowed_gpu_rows"]
    gpu_proof.validate_instance(identity)
    gpu_proof.verify_snapshot(runtime["before"], identity, allowed, p["physical_gpus"], checks, "Annual before")
    gpu_proof.verify_snapshot(runtime["after"], identity, allowed, p["physical_gpus"], checks, "Annual after")
    checks.check(utc(marker["at"]) <= utc(runtime["before"]["at"]) <= utc(runtime["after"]["at"])
                 <= utc(process["completed_at"]), "Actual resource observations lie inside the separately authorized annual attempt")
    if runtime.get("query_events"):
        checks.check(runtime.get("old_subprocess_run_restored") is True, "Any temporary resource proxy was fully restored")
        for event in runtime["query_events"]:
            gpu_proof.verify_query_event(event, identity, allowed, checks, "Annual exact resource query")
    return completion, process


def verify_worker_groups(inputs, protocol, group, completion):
    """Check real annual IDs without renaming frozen validation proof fields."""
    workers, checks = completion.get("workers", []), inputs.checks
    checks.check(len(workers) == 2 and {w.get("scenario") for w in workers} == set(COSTS)
                 and len({w.get("worker_pid") for w in workers}) == 2,
                 "Two distinct actual annual cost-worker handles")
    for cost in COSTS:
        folder = f"test/balanced/{cost}"
        tables = [inputs.csv("annual", folder + "/" + name) for name in GROUP_FILES[:3]]
        worker = next((w for w in workers if w.get("scenario") == cost), {})
        record = inputs.json("annual", folder + "/completion.json")
        checks.check(record.get("case_count") == worker.get("case_count") == 6
                     and record.get("worker_pid") == worker.get("worker_pid")
                     and record.get("protocol_sha256") == sha(inputs.roots["annual"] / "protocol.json"),
                     cost + ": actual annual worker completed all six cases")
        device = record.get("device", {})
        checks.check(device.get("visible_gpu_uuid") == protocol["gpu_cost_assignments"][cost],
                     cost + ": correct isolated original physical card")
        receipts = record.get("repeats", [])
        checks.check(len(receipts) == 6 and {r.get("run_id") for r in receipts} == set(tables[0].run_id),
                     cost + ": six complete repeat proofs")
        for key in (*map(str, SEEDS), "ensemble"):
            proof = inputs.json("annual", f"{folder}/{key}/replay.json")
            checks.check(proof in receipts and proof.get("worker_pid") == worker.get("worker_pid")
                         and proof.get("device") == device, cost + ": per-case actual annual worker identity")
        compare_tables(inputs, tables, group[:3], "annual", cost=cost)


def compare_tables(inputs, actual_tables, expected_tables, label, *, cost=None):
    for actual, expected, keys in zip(actual_tables, expected_tables,
                                      (["run_id"], ["run_id", "timestamp"], ["run_id", "month"])):
        if cost is not None:
            expected = expected.loc[expected.run_id.str.startswith("test__" + cost + "__")]
        checks = inputs.checks
        checks.check(set(actual.columns) == set(expected.columns) and len(actual) == len(expected),
                     label + ": exact aggregate/worker table dimensions")
        if set(actual.columns) != set(expected.columns) or len(actual) != len(expected):
            continue
        actual, expected = actual.sort_values(keys).reset_index(drop=True), expected.sort_values(keys).reset_index(drop=True)
        checks.check(not actual[keys].duplicated().any(), label + ": no duplicate financial table identities")
        for column in expected:
            missing = actual[column].isna() & expected[column].isna()
            if pd.api.types.is_numeric_dtype(expected[column]):
                checks.vector_close(actual.loc[~missing, column], expected.loc[~missing, column],
                                    label + ": financial table:" + column, atol=1e-8)
            else:
                checks.check((missing | actual[column].eq(expected[column])).all(), label + ": financial table:" + column)


def performance_evidence(summary, baselines, old_goal):
    """Performance is separate from accounting and from fresh-holdout evidence."""
    validate_group_rows(summary)
    ids = {f"test__{cost}__{name}" for cost in COSTS for name in BASELINES}
    if len(baselines) != 14 or set(baselines.run_id) != ids or baselines.run_id.duplicated().any():
        raise ValueError("All seven original baselines at both costs are required")
    base = summary.loc[summary.scenario.eq("base") & summary.primary_policy].iloc[0]
    stress = summary.loc[summary.scenario.eq("stress") & summary.primary_policy].iloc[0]
    members = summary.loc[summary.scenario.eq("base") & ~summary.primary_policy].sort_values("seed")
    equal = baselines.loc[baselines.run_id.eq("test__base__equal_hold")].iloc[0]
    positive = int(members.net_return.gt(0).sum())
    gates = {"base_return_at_least_5pct": bool(base.net_return >= .05),
        "base_max_drawdown_at_most_20pct": bool(abs(base.max_drawdown) <= .20),
        "stress_return_nonnegative": bool(stress.net_return >= 0),
        "at_least_three_positive_training_seeds": positive >= 3,
        "beats_equal_hold_return_and_drawdown": bool(base.net_return > equal.net_return and base.max_drawdown > equal.max_drawdown)}
    controls = {name: baselines.loc[baselines.run_id.eq(f"test__base__{name}")].iloc[0] for name in BASELINES}
    simple_trends = {name: bool(base.net_return > controls[name].net_return and base.max_drawdown > controls[name].max_drawdown)
                     for name in ("sma200_trend", "momentum180")}
    return {"candidate": {"family": FAMILY, "algorithm": "tqc", "profile": "balanced"},
        "gates": gates, "all_five_diagnostic_targets_met": all(gates.values()),
        "goal_performance_achieved": all(gates.values()), "positive_base_training_seeds": positive,
        "base_seed_returns": [{"seed": int(row.seed), "net_return": float(row.net_return)} for row in members.itertuples()],
        "base_net_return": float(base.net_return), "base_max_drawdown": float(base.max_drawdown),
        "stress_net_return": float(stress.net_return), "stress_max_drawdown": float(stress.max_drawdown),
        "base_fees_usdt": float(base.fees), "stress_fees_usdt": float(stress.fees),
        "beats_simple_trend_controls": simple_trends,
        "stronger_control_comparison_met": all(simple_trends.values()),
        "old_goal_performance_achieved": old_goal["goal_performance_achieved"],
        "old_positive_base_training_seeds": old_goal["metrics"]["positive_base_training_seeds"],
        "accounting_passage_is_not_goal_success": True, "fresh_holdout_success": False,
        **{flag: True for flag in FLAGS},
        "interpretation": "Diagnostic replay of an already-observed year; five seeds repeat one market history"}


def audit(output, validation, main_output, reference, dataset_path, *, source_root=ROOT):
    """Audit38 original/replayed accounts, without model inference or execution."""
    checks, started = Checks(), band.now()
    inputs = Inputs(output, validation, main_output, reference, source_root, checks)
    runs, reference_runs, baseline_runs, evidence, training = {}, {}, {}, None, None
    try:
        p, spec, records, original_audit, old_goal = verify_protocol(inputs, dataset_path)
        completion, process = verify_execution_chain(inputs, p)
        if checks.failed:
            raise ValueError("Authorization/resource/terminal evidence failed before annual account reconstruction")
        data = load_market(dataset_path, TEST[1])
        features = band.independent_market_features(data, cutoff=TEST[1])
        hours = pd.date_range(utc(TEST[0]), utc(TEST[1]), freq="h", inclusive="left")
        for field in ("open", "high", "low", "close", "volume"):
            frame = data["hourly_" + field].reindex(hours)
            values = frame.to_numpy(dtype=float)
            checks.check(len(frame) == 8760 and np.isfinite(values).all()
                         and (values >= 0 if field == "volume" else values > 0).all(),
                         "Every original annual two-coin OHLCV hour complete:" + field)
        training = band.verify_training_cutoff(records, spec, data, checks)
        group = audit_group(inputs, data, records, p, features=features)
        verify_worker_groups(inputs, p, group, completion)
        annual_tables = [inputs.csv("annual", name) for name in GROUP_FILES[:3]]
        compare_tables(inputs, annual_tables, group[:3], "Parent annual tables")
        old = audit_group(inputs, data, records, p, reference=True)
        baseline_frame, baseline_runs = audit_baselines(inputs, data, spec, original_audit)
        runs = {key: value[0] for key, value in group[3].items()}
        reference_runs = {key: value[0] for key, value in old[3].items()}
        checks.check(len(runs) == len(reference_runs) == 12 and len(baseline_runs) == 14,
                     "Exactly12 new/12 original capped/14 original baseline accounts reconstructed")
        for cost in COSTS:
            row = old[0].loc[old[0].scenario.eq(cost) & old[0].primary_policy].iloc[0]
            for name in ("net_return", "max_drawdown"):
                checks.close(row[name], old_goal["metrics"][cost + "_" + name],
                             "Original unsuccessful annual goal remains independently reconciled:" + cost + ":" + name,
                             atol=1e-10)
        old_positive = int(old[0].loc[old[0].scenario.eq("base") & ~old[0].primary_policy, "net_return"].gt(0).sum())
        checks.check(old_positive == old_goal["metrics"]["positive_base_training_seeds"] == 2,
                     "Original2/5-positive-seed failure retained separately from new annual")
        inputs.unchanged()
        if checks.failed == 0:
            evidence = performance_evidence(group[0], baseline_frame, old_goal)
    except Exception as exc:
        checks.check(False, type(exc).__name__ + ": " + str(exc))
    passed = checks.failed == 0 and len(runs) == len(reference_runs) == 12 and len(baseline_runs) == 14
    return {"schema_version": 1, "started_at": started, "audited_at": band.now(),
        "passed": passed, "status": "passed" if passed else "failed", "test_complete": passed,
        "run_count": len(runs), "repeat_count": len(runs) if passed else 0,
        "reference_run_count": len(reference_runs), "baseline_run_count": len(baseline_runs),
        "checks_total": checks.count, "failure_count": checks.failed, "failures": checks.failures,
        "maximum_errors": checks.maximum_errors, "input_sha256": inputs.hashes,
        "source_sha256": inputs.hashes["source"], "dataset_sha256": sha(dataset_path),
        "runs": runs, "reference_runs": reference_runs, "baseline_runs": baseline_runs,
        "declared_research_targets": TARGETS, "research_targets": evidence, "training_calendar": training,
        "fresh_holdout_success": False, "accounting_passage_is_not_goal_success": True,
        **{flag: True for flag in FLAGS},
        "scope": "Separate already-seen365-day diagnostic replay; original unsuccessful goal/report are immutable"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("output-dir", "validation-output", "main-output", "reference-output", "dataset"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = args.report or args.output_dir / "independent_audit.json"
    if report.exists():
        raise FileExistsError("Never overwrite an earlier annual audit or historical report")
    result = audit(args.output_dir, args.validation_output, args.main_output, args.reference_output, args.dataset)
    report.parent.mkdir(parents=True, exist_ok=True)
    with report.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({key: result[key] for key in ("passed", "status", "test_complete",
        "run_count", "repeat_count", "reference_run_count", "baseline_run_count", "checks_total", "failure_count")}))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()

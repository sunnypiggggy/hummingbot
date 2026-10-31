"""Pre-test, one-shot TQC .75 exposure extension of the frozen BTC/ETH study.

The old study is read-only. Validation and test are distinct guarded phases.
This overlay changes target exposure, with new policy observations from its own
funded ledger; it never rescales a previously observed return series.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import sys

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_name] = "1"

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from scripts import backtest_btceth_rl as formal
from scripts import btceth_rl_goal_candidate as goal
from scripts.crypto_rl_reporting import canonical_csv, result_fingerprints, save_result, summarize, write_json

MULTIPLIER = .75
PROFILES = ("return", "balanced")
DIAGNOSTIC_SHA = "2cefc724e28f8b01ec400acf48e87748a4807aa59278c74e8d85f396935f8cab"
CASE_FILES = ("equity_1h.csv.gz", "fills.csv", "weights.csv", "boundary_equity.csv", "metadata.json", "fingerprints.json", "member_targets.csv", "replay.json")
GROUP_FILES = ("summary.csv", "daily_equity.csv", "monthly_returns.csv", "completion.json")
now, sha, read_json = formal.now, formal.sha, formal.read_json
relative, stamp = goal.relative_file, goal.stamp


def write_new(path, value):
    goal.exclusive_json(Path(path), value)


def pin(root, names):
    return {name: sha(relative(root, name)) for name in sorted(names)}


def check_pins(root, values, label):
    if not isinstance(values, dict) or not values:
        raise ValueError(label + " requires nonempty SHA evidence")
    for name, expected in values.items():
        if sha(relative(root, name)) != expected:
            raise ValueError(label + " input changed: " + name)


def same_metric(actual, expected):
    if isinstance(expected, dict):
        return isinstance(actual, dict) and set(actual) == set(expected) and all(same_metric(actual[k], v) for k, v in expected.items())
    if isinstance(expected, list):
        return isinstance(actual, list) and len(actual) == len(expected) and all(same_metric(a, e) for a, e in zip(actual, expected))
    if isinstance(expected, (int, float)):
        return isinstance(actual, (int, float)) and bool(np.isclose(actual, expected, atol=1e-10, rtol=1e-10))
    return actual == expected


def separate(main, output):
    main, output = Path(main).resolve(), Path(output).resolve()
    if output == main or output.is_relative_to(main) or main.is_relative_to(output):
        raise ValueError("Extension and original study must be separate sibling trees")
    return main, output


def models_manifest(main, spec):
    models, names = [], []
    for profile in PROFILES:
        for seed in formal.SEEDS:
            folder = f"training/{formal.job_id('tqc', profile, seed)}"
            record_file, model_file = folder + "/record.json", folder + "/model.zip"
            record = read_json(relative(main, record_file))
            if ((record["algorithm"], record["profile"], record["seed"]) != ("tqc", profile, seed)
                    or record["model_relative_path"] != model_file
                    or record["frozen_spec_sha256"] != sha(Path(main) / "frozen_spec.json")
                    or record["model_sha256"] != sha(relative(main, model_file))):
                raise ValueError("TQC checkpoint identity or frozen training SHA changed")
            models.append(dict(profile=profile, seed=seed, record_file=record_file, model_file=model_file,
                record_sha256=sha(relative(main, record_file)), model_sha256=record["model_sha256"],
                parameter_sha256=record["parameter_sha256"]))
            names.extend((record_file, model_file))
    return models, pin(main, names)


def prepare(main, dataset, output, diagnostic):
    main, output = separate(main, output)
    dataset, diagnostic = Path(dataset).resolve(), Path(diagnostic).resolve()
    if output.exists():
        raise FileExistsError("Preserve any existing extension attempt")
    goal.ensure_no_test(main)
    spec = formal.check_frozen(main, dataset)
    goal.load_protocol(main)
    if sha(diagnostic / "summary.csv") != DIAGNOSTIC_SHA:
        raise ValueError("Only the already reviewed single-.75 diagnostic may motivate this extension")
    done = read_json(diagnostic / "completion.json")
    if done.get("passed") is not True or done.get("cases") != 4 or done.get("test_evaluations") != 0:
        raise ValueError("Require the complete validation-only exposure diagnostic")
    models, training = models_manifest(main, spec)
    runtime = read_json(main / "runtime.json")
    inventory = runtime["physical_gpus"]
    if len(inventory) != 2 or len({g["uuid"] for g in inventory}) != 2 or any("V100" not in g["name"] for g in inventory):
        raise ValueError("Freeze the two original distinct V100 identities")
    source = {**spec["code_sha256"], **goal.source_hashes(),
              "scripts/btceth_rl_exposure_cap.py": sha(Path(__file__))}
    check_pins(ROOT, source, "Financial/goal/extension source")
    protocol = dict(schema_version=1, prepared_at=now(), multiplier=MULTIPLIER,
        scope="Independent funded BTCUSDT/ETHUSDT/USDT TQC exposure extension; original 172 cases unchanged",
        motivation="After validation showed TQC balanced 22-24% drawdown, freeze one .75 exposure reduction before test; no multiplier search or retraining.",
        designed_after_validation=True, new_training=0, multiplier_search=False,
        original_frozen_spec_sha256=sha(main / "frozen_spec.json"),
        original_goal_protocol_sha256=sha(main / goal.PROTOCOL_FILE), source_sha256=source,
        training_source_sha256=training, models=models, dataset_sha256=spec["dataset_sha256"],
        dataset_manifest_sha256=spec["dataset_manifest_sha256"], physical_gpus=inventory,
        runtime_packages=spec["runtime_packages"], capital=10000., costs=formal.COSTS,
        profiles=list(PROFILES), seeds=list(formal.SEEDS), dates_utc={k: spec[k] for k in ("train", "validation", "test")},
        research_targets=spec["research_targets"], rebalance_band=.025, cash_floor=.05,
        max_asset_weight=.95, effective_max_coin_target_total=.7125,
        mapping="original action_to_weights -> BTC and ETH each times .75 -> USDT residual -> arithmetic mean member targets -> one real funded ledger; next obs uses actual weights/DD",
        execution="original daily 00UTC reductions, 01UTC purchases; hourly close equity; per-leg costs unchanged",
        diagnostic_summary_sha256=DIAGNOSTIC_SHA,
        diagnostic_source_sha256=pin(diagnostic, ("summary.csv", "protocol.json", "completion.json")),
        validation_cases=24, repeats_per_case=1, test_cases_if_cap_selected=12,
        selection_score=spec["validation_selection"]["score"],
        selection_population="six original new-algorithm/profile raw ensembles and two TQC cap075 ensembles; SAC control only",
        tie_break="descending fixed score, raw before cap075, algorithm then profile",
        selection_uses_test=False, seed_selection=False, known_test_year_already_seen=True)
    goal.ensure_no_test(main)
    output.mkdir(parents=True)
    for name in source:
        target = output / "sources" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    (output / "sources/original_frozen_spec.json").write_bytes((main / "frozen_spec.json").read_bytes())
    for name in protocol["diagnostic_source_sha256"]:
        shutil.copyfile(diagnostic / name, output / "sources" / ("diagnostic_" + name))
    write_new(output / "protocol.json", protocol)
    write_new(output / "protocol_receipt.json", dict(sealed_at=now(), protocol_file="protocol.json", protocol_sha256=sha(output / "protocol.json")))
    return protocol


def verify_protocol(main, dataset, output):
    main, output = separate(main, output)
    p, receipt = read_json(output / "protocol.json"), read_json(output / "protocol_receipt.json")
    spec = formal.check_frozen(main, dataset)
    if (receipt.get("protocol_sha256") != sha(output / "protocol.json") or receipt.get("protocol_file") != "protocol.json"
            or stamp(receipt["sealed_at"]) < stamp(p["prepared_at"])
            or p.get("multiplier") != MULTIPLIER or p.get("capital") != 10000.
            or p.get("costs") != formal.COSTS or p.get("profiles") != list(PROFILES) or p.get("seeds") != list(formal.SEEDS)
            or p.get("rebalance_band") != .025 or p.get("cash_floor") != .05 or p.get("max_asset_weight") != .95
            or p.get("effective_max_coin_target_total") != .7125
            or p.get("original_frozen_spec_sha256") != sha(main / "frozen_spec.json")
            or p.get("original_goal_protocol_sha256") != sha(main / goal.PROTOCOL_FILE)
            or p.get("dates_utc") != {k: spec[k] for k in ("train", "validation", "test")}
            or p.get("research_targets") != spec["research_targets"]
            or p.get("dataset_sha256") != spec["dataset_sha256"] or p.get("dataset_manifest_sha256") != spec["dataset_manifest_sha256"]
            or p.get("diagnostic_summary_sha256") != DIAGNOSTIC_SHA
            or p.get("selection_uses_test") is not False or p.get("seed_selection") is not False):
        raise ValueError("Frozen exposure-extension protocol changed")
    expected_source = {**spec["code_sha256"], **goal.source_hashes(), "scripts/btceth_rl_exposure_cap.py": sha(Path(__file__))}
    if p.get("source_sha256") != expected_source:
        raise ValueError("Prepared financial/goal/extension source no longer matches")
    check_pins(ROOT, p["source_sha256"], "Prepared source")
    check_pins(output / "sources", p["source_sha256"], "Byte snapshot")
    if sha(output / "sources/original_frozen_spec.json") != p["original_frozen_spec_sha256"]:
        raise ValueError("Original frozen spec byte snapshot changed")
    for name, expected in p["diagnostic_source_sha256"].items():
        if sha(output / "sources" / ("diagnostic_" + name)) != expected:
            raise ValueError("Reviewed diagnostic snapshot changed")
    check_pins(main, p["training_source_sha256"], "Original TQC training")
    models, training = models_manifest(main, spec)
    if p["models"] != models or p["training_source_sha256"] != training:
        raise ValueError("Prepared ten-model identity changed")
    return p


def terminal_process(main, phase):
    """GPU phases run on the original Linux host, with terminal process proof."""
    path = Path(main) / f"{phase}_process.json"
    p = read_json(path)
    if (p.get("phase") != phase or p.get("status") != "completed" or p.get("exit_code") != 0
            or p.get("frozen_spec_sha256") != sha(Path(main) / "frozen_spec.json")
            or not p.get("completed_at") or not p.get("log_relative_path")
            or sha(relative(main, p["log_relative_path"])) != p.get("log_sha256")):
        raise ValueError("Original " + phase + " must have completed with exit0 and unchanged log")
    if sys.platform != "linux" or not Path("/proc").is_dir():
        raise RuntimeError("GPU phases require Linux process inspection on the original host")
    command = p.get("command", [])
    if "--output-dir" not in command or Path(command[command.index("--output-dir") + 1]).resolve() != Path(main).resolve():
        raise ValueError("Terminal process proof belongs to another study")
    for pid in (p.get("supervisor_pid"), p.get("child_pid")):
        proc = Path("/proc") / str(pid)
        if proc.is_dir():
            try:
                state = (proc / "stat").read_text().rsplit(")", 1)[1].strip().split()[0]
                cmd = (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
            except (FileNotFoundError, ProcessLookupError):
                continue
            if state != "Z" and ("btceth_rl" in cmd or "multiprocessing" in cmd):
                raise ValueError("Original " + phase + " process is still live")
    return p


def cap_target(raw):
    target = np.asarray(raw, dtype=float).copy()
    if target.shape != (3,) or not np.isfinite(target).all() or (target < 0).any() or abs(target.sum() - 1) > 1e-10:
        raise ValueError("Three finite funded raw targets required")
    target[:2] *= MULTIPLIER
    target[2] = 1 - target[:2].sum()
    if target[:2].sum() > .7125 + 1e-12:
        raise ValueError("Original gross target must respect .95 budget")
    return target


def evaluate_cap(data, models, seeds, profile, split, costs):
    from scripts.btceth_rl_env import BTCETHPortfolioEnv
    start, end = formal.VALIDATION if split == "validation" else formal.TEST
    if (len(models) not in (1, 5) or len(models) != len(seeds) or len(set(seeds)) != len(seeds)
            or not set(seeds).issubset(formal.SEEDS) or (len(models) == 5 and set(seeds) != set(formal.SEEDS))):
        raise ValueError("A case is exactly one fixed seed or all five models")
    before = [formal.parameter_digest(m.policy) for m in models]
    env = BTCETHPortfolioEnv(data, start, end, costs=costs, record_history=True, **formal.PROFILES[profile])
    obs, _ = env.reset(seed=0)
    trace, done = [], False
    while not done:
        day, targets, members = env.current_date.isoformat(), [], []
        observation = np.asarray(obs, dtype="<f4")
        observation_sha = hashlib.sha256(observation.tobytes()).hexdigest()
        for seed, model in zip(seeds, models):
            action, _ = model.predict(obs, deterministic=True)
            raw = env.action_to_weights(action)
            target = cap_target(raw)
            targets.append(target)
            members.append(dict(day=day, seed=seed, raw_action0=float(action[0]), raw_action1=float(action[1]),
                raw_btc=float(raw[0]), raw_eth=float(raw[1]), raw_cash=float(raw[2]),
                capped_btc=float(target[0]), capped_eth=float(target[1]), capped_cash=float(target[2]),
                observation_sha256=observation_sha, observation_btc_weight=float(obs[-4]),
                observation_eth_weight=float(obs[-3]), observation_cash_weight=float(obs[-2]), observation_drawdown=float(obs[-1])))
        target = np.mean(targets, axis=0)
        for member in members:
            member.update(target_btc=float(target[0]), target_eth=float(target[1]), target_cash=float(target[2]))
            trace.append(member)
        obs, _, terminated, truncated, _ = env.step_target(target)
        done = bool(terminated or truncated)
    result = env.get_result()
    env.close()
    after = [formal.parameter_digest(m.policy) for m in models]
    if before != after:
        raise AssertionError("Evaluation changed frozen trained parameters")
    result["metadata"].update(algorithm="tqc", family="cap075", profile=profile, model_count=len(models),
        exposure_multiplier=MULTIPLIER, parameter_sha256_before=before, parameter_sha256_after=after,
        ensemble="arithmetic mean of mapped and capped targets; one real10000 USDT ledger" if len(models) == 5 else None,
        observations="this account actual weights and drawdown; original causal market features", old_actions_or_equity_scaled=False)
    return result, pd.DataFrame(trace)


def evaluate_group(task):
    import torch
    main, dataset, output = map(Path, (task["main"], task["dataset"], task["output"]))
    p = verify_protocol(main, dataset, output)
    device = formal.torch_setup(123456)
    if device["visible_gpu_uuid"] != task["uuid"] or "V100" not in device["name"]:
        raise ValueError("Overlay worker must use its UUID-isolated original V100")
    if formal.runtime_packages() != p["runtime_packages"]:
        raise ValueError("Formal inference package versions changed")
    split, profile = task["split"], task["profile"]
    directory = output / split / profile
    if directory.exists():
        raise FileExistsError("Preserve existing evaluation attempt")
    directory.mkdir(parents=True)
    data = formal.load_dataset(dataset)
    # Validation never constructs features from test-year price frames.
    if split == "validation":
        for name in list(data):
            if name.startswith("hourly_"):
                data[name] = data[name].loc[data[name].index < pd.Timestamp(formal.VALIDATION[1])].copy()
    selected_records = [r for r in p["models"] if r["profile"] == profile]
    models = []
    for record in selected_records:
        model = formal.model_class("tqc").load(relative(main, record["model_file"]), device="cuda:0")
        if formal.parameter_digest(model.policy) != record["parameter_sha256"]:
            raise ValueError("Loaded parameter digest differs from original record")
        models.append(model)
    rows, histories, months, repeats = [], [], [], []
    start, end = formal.VALIDATION if split == "validation" else formal.TEST
    for scenario, costs in formal.COSTS.items():
        plans = [(str(seed), [model], [seed], seed) for seed, model in zip(formal.SEEDS, models)] + [("ensemble", models, list(formal.SEEDS), None)]
        for key, members, seeds, seed in plans:
            run_id = f"{split}__{scenario}__cap075__tqc__{profile}__{key}"
            result, trace = evaluate_cap(data, members, seeds, profile, split, costs)
            repeat, repeated_trace = evaluate_cap(data, members, seeds, profile, split, costs)
            fingerprints, repeated = result_fingerprints(result), result_fingerprints(repeat)
            trace_bytes, repeat_bytes = canonical_csv(trace), canonical_csv(repeated_trace)
            if fingerprints != repeated or trace_bytes != repeat_bytes:
                raise AssertionError("Exact evaluation/member-target replay failed: " + run_id)
            case_dir = directory / scenario / key
            save_result(case_dir, result)
            (case_dir / "member_targets.csv").write_bytes(trace_bytes)
            selected = [r for r in selected_records if r["seed"] in seeds]
            proof = dict(run_id=run_id, passed=True, repeat_exact=True, fingerprints=fingerprints,
                repeated_fingerprints=repeated, member_targets_sha256=hashlib.sha256(trace_bytes).hexdigest(),
                repeated_member_targets_sha256=hashlib.sha256(repeat_bytes).hexdigest(),
                parameter_sha256_before=result["metadata"]["parameter_sha256_before"],
                parameter_sha256_after=result["metadata"]["parameter_sha256_after"], model_records=selected, device=device)
            write_new(case_dir / "replay.json", proof)
            row, daily, monthly = summarize(result, run_id=run_id, strategy="tqc_cap075", seed=seed,
                scenario=scenario, split=split, start=start, end=end, capital=10000., **costs)
            row.update(algorithm="tqc", family="cap075", profile=profile, primary_policy=seed is None, exposure_multiplier=MULTIPLIER)
            daily["profile"], monthly["profile"] = profile, profile
            rows.append(row); histories.append(daily); months.append(monthly); repeats.append(proof)
            print(f"CAP {run_id} return={row['net_return']:.6%} DD={row['max_drawdown']:.6%}", flush=True)
    pd.DataFrame(rows).to_csv(directory / "summary.csv", index=False, float_format="%.17g")
    pd.concat(histories, ignore_index=True).to_csv(directory / "daily_equity.csv", index=False, float_format="%.17g")
    pd.concat(months, ignore_index=True).to_csv(directory / "monthly_returns.csv", index=False, float_format="%.17g")
    write_new(directory / "completion.json", dict(at=now(), protocol_sha256=sha(output / "protocol.json"), case_count=12, repeats=repeats))
    del models
    torch.cuda.empty_cache()
    return dict(profile=profile, case_count=12)


def extension_names(split, profiles):
    return {*(f"{split}/{p}/{f}" for p in profiles for f in GROUP_FILES),
            *(f"{split}/{p}/{c}/{k}/{f}" for p in profiles for c in formal.COSTS
              for k in (*map(str, formal.SEEDS), "ensemble") for f in CASE_FILES), f"{split}_completion.json"}


def cap_rows(frame, split, profile, spec):
    copied = frame.copy()
    expected_ids = {f"{split}__{c}__cap075__tqc__{profile}__{k}" for c in formal.COSTS for k in (*map(str, formal.SEEDS), "ensemble")}
    if set(frame.run_id) != expected_ids or not frame.family.eq("cap075").all() or not frame.exposure_multiplier.eq(MULTIPLIER).all():
        raise ValueError("Overlay financial identity or fixed multiplier changed")
    copied["run_id"] = copied.run_id.str.replace("__cap075", "", regex=False)
    goal.validate_rows(copied, split=split, algorithm="tqc", profile=profile, spec=spec)


def assemble(output, main=None):
    output = Path(output)
    groups = sorted(output.glob("validation/*/summary.csv")) + sorted(output.glob("test/*/summary.csv"))
    if groups:
        for name in ("summary.csv", "daily_equity.csv", "monthly_returns.csv"):
            frames = [pd.read_csv(g.parent / name, float_precision="round_trip") for g in groups]
            done = output / "test_completion.json"
            if done.is_file() and read_json(done).get("mode") == "raw_reference":
                if main is None:
                    raise ValueError("Raw-reference assembly needs its explicit original study root")
                candidate = read_json(output / "selection.json")["candidate"]
                folder = Path(main) / f"evaluation/test/{candidate['algorithm']}/{candidate['profile']}"
                raw = pd.read_csv(folder / name, float_precision="round_trip")
                raw["family"], raw["exposure_multiplier"] = "raw", 1.
                frames.append(raw)
            pd.concat(frames, ignore_index=True).to_csv(output / name, index=False, float_format="%.17g")


def run_validation(main, dataset, output):
    p = verify_protocol(main, dataset, output)
    goal.ensure_no_test(Path(main))
    terminal_process(main, "fit")
    if (Path(output) / "validation").exists() or (Path(output) / "validation_completion.json").exists():
        raise FileExistsError("Preserve prior validation attempt")
    run_gpu_groups(main, dataset, output, "validation", list(PROFILES), p)
    write_new(Path(output) / "validation_completion.json", dict(at=now(), protocol_sha256=sha(Path(output) / "protocol.json"), case_count=24, repeat_count=24, profiles=list(PROFILES), split="validation"))
    assemble(output, main)


def run_gpu_groups(main, dataset, output, split, profiles, p):
    inventory = formal.gpu_inventory()
    if {g["uuid"] for g in inventory} != {g["uuid"] for g in p["physical_gpus"]}:
        raise ValueError("Current V100 UUID inventory differs from prepared host")
    pools, futures = [], []
    try:
        for index, profile in enumerate(profiles):
            uuid = p["physical_gpus"][index]["uuid"]
            pool = ProcessPoolExecutor(max_workers=1, mp_context=mp.get_context("spawn"), initializer=formal.gpu_initializer, initargs=(uuid,))
            pools.append(pool)
            futures.append(pool.submit(evaluate_group, dict(main=str(Path(main).resolve()), dataset=str(Path(dataset).resolve()), output=str(Path(output).resolve()), split=split, profile=profile, uuid=uuid)))
        for future in futures:
            future.result()
    finally:
        for pool in pools:
            pool.shutdown(wait=True, cancel_futures=True)


def candidates(main, output, spec, original):
    raw = goal.validation_candidates(Path(main), spec, original)
    rows = [dict(family="raw", **r) for r in raw if r["algorithm"] in goal.NEW_ALGORITHMS]
    done = read_json(Path(output) / "validation_completion.json")
    if done.get("case_count") != 24 or done.get("repeat_count") != 24 or done.get("protocol_sha256") != sha(Path(output) / "protocol.json"):
        raise ValueError("Require all24 extension validation cases and exact repeats")
    for profile in PROFILES:
        validate_group_evidence(Path(output), "validation", profile)
        frame = goal.read_summary(Path(output) / "validation" / profile / "summary.csv")
        cap_rows(frame, "validation", profile, spec)
        row = goal.candidate_row(frame, "tqc", profile)
        rows.append(dict(family="cap075", **row))
    if len(rows) != 8:
        raise ValueError("Exactly six raw new-model and two cap candidates required")
    return sorted(rows, key=lambda r: (-r["score"], 0 if r["family"] == "raw" else 1, r["algorithm"], r["profile"]))


def validate_group_evidence(output, split, profile):
    """Recheck replay proof identities before sealing validation selection."""
    p = read_json(output / "protocol.json")
    group = output / split / profile
    done = read_json(group / "completion.json")
    expected = {f"{split}__{c}__cap075__tqc__{profile}__{k}" for c in formal.COSTS for k in (*map(str, formal.SEEDS), "ensemble")}
    if (done.get("case_count") != 12 or done.get("protocol_sha256") != sha(output / "protocol.json")
            or len(done.get("repeats", [])) != 12 or {r.get("run_id") for r in done["repeats"]} != expected):
        raise ValueError("Extension group must have exactly12 complete replay proofs")
    proofs = {r["run_id"]: r for r in done["repeats"]}
    for scenario in formal.COSTS:
        for key in (*map(str, formal.SEEDS), "ensemble"):
            directory = group / scenario / key
            proof = read_json(directory / "replay.json")
            run_id = f"{split}__{scenario}__cap075__tqc__{profile}__{key}"
            models = [r for r in p["models"] if r["profile"] == profile and (key == "ensemble" or r["seed"] == int(key))]
            params = [r["parameter_sha256"] for r in models]
            if (proof != proofs[run_id] or proof.get("run_id") != run_id or proof.get("passed") is not True
                    or proof.get("repeat_exact") is not True or proof.get("fingerprints") != proof.get("repeated_fingerprints")
                    or proof.get("fingerprints") != read_json(directory / "fingerprints.json")
                    or proof.get("member_targets_sha256") != sha(directory / "member_targets.csv")
                    or proof.get("member_targets_sha256") != proof.get("repeated_member_targets_sha256")
                    or proof.get("model_records") != models or proof.get("parameter_sha256_before") != params
                    or proof.get("parameter_sha256_after") != params
                    or proof.get("device", {}).get("visible_gpu_uuid") not in {g["uuid"] for g in p["physical_gpus"]}
                    or proof.get("device", {}).get("device") != "cuda:0"):
                raise ValueError("Saved replay/source/parameter evidence changed: " + run_id)


def select(main, dataset, output):
    main, output = map(Path, (main, output))
    p = verify_protocol(main, dataset, output)
    goal.ensure_no_test(main)
    if any((output / name).exists() for name in ("test_started.json", "test_completion.json", "test")):
        raise ValueError("Extension selection must precede every extension test attempt")
    if (output / "selection.json").exists() or (output / "selection_receipt.json").exists():
        raise FileExistsError("Preserve the frozen extension selection")
    context = goal.validate_frozen_candidate(main)
    ranked = candidates(main, output, context["spec"], context["original_selection"])
    winner = ranked[0]
    if winner["family"] == "raw" and winner["profile"] != context["original_selection"]["selected_profiles"][winner["algorithm"]]:
        raise ValueError("A raw winner must reuse its original selected test profile")
    selection = dict(selected_at=now(), protocol_sha256=sha(output / "protocol.json"),
        original_validation_selection_sha256=sha(main / "validation_selection.json"),
        original_goal_selection_sha256=sha(main / goal.SELECTION_FILE), original_goal_receipt_sha256=sha(main / goal.RECEIPT_FILE),
        training_source_sha256=context["original_selection"]["training_source_sha256"],
        validation_source_sha256=context["original_selection"]["validation_source_sha256"],
        extension_source_sha256=pin(output, extension_names("validation", PROFILES)),
        original_global_validation_leader=context["original_selection"]["primary_validation_leader"],
        original_new_model_validation_leader=context["new_model_validation_leader"],
        candidates=ranked, candidate=winner, selection_uses_test=False, seed_selection=False,
        designed_after_validation=True, new_validation_is_unseen=False)
    goal.ensure_no_test(main)
    write_new(output / "selection.json", selection)
    write_new(output / "selection_receipt.json", dict(sealed_at=now(), selection_file="selection.json", selection_sha256=sha(output / "selection.json"), protocol_sha256=sha(output / "protocol.json")))
    return selection


def validate_selection(main, dataset, output, *, require_test=False):
    main, output = map(Path, (main, output))
    p = verify_protocol(main, dataset, output)
    context = goal.validate_frozen_candidate(main, require_test_started=require_test)
    s, r = read_json(output / "selection.json"), read_json(output / "selection_receipt.json")
    if (r.get("selection_file") != "selection.json" or r.get("selection_sha256") != sha(output / "selection.json")
            or r.get("protocol_sha256") != sha(output / "protocol.json") or s.get("protocol_sha256") != r["protocol_sha256"]
            or s.get("original_validation_selection_sha256") != sha(main / "validation_selection.json")
            or s.get("original_goal_selection_sha256") != sha(main / goal.SELECTION_FILE)
            or s.get("original_goal_receipt_sha256") != sha(main / goal.RECEIPT_FILE)
            or s.get("training_source_sha256") != context["original_selection"]["training_source_sha256"]
            or s.get("validation_source_sha256") != context["original_selection"]["validation_source_sha256"]
            or s.get("selection_uses_test") is not False or s.get("seed_selection") is not False):
        raise ValueError("Frozen extension selection/receipt source changed")
    protocol_receipt = read_json(output / "protocol_receipt.json")
    if (not stamp(p["prepared_at"]) <= stamp(protocol_receipt["sealed_at"]) <= stamp(s["selected_at"]) <= stamp(r["sealed_at"])
            or stamp(context["receipt"]["sealed_at"]) > stamp(s["selected_at"])):
        raise ValueError("Extension prepare/select/seal chronology invalid")
    if (main / "test_started.json").is_file() and not stamp(r["sealed_at"]) < stamp(read_json(main / "test_started.json")["at"]):
        raise ValueError("Extension selection must be sealed before original test starts")
    if set(s["extension_source_sha256"]) != extension_names("validation", PROFILES):
        raise ValueError("Selection must pin complete immutable extension validation inputs")
    check_pins(output, s["extension_source_sha256"], "Sealed extension validation")
    ranked = candidates(main, output, context["spec"], context["original_selection"])
    if (s.get("candidate") != ranked[0] or s.get("candidates") != ranked
            or s.get("original_global_validation_leader") != context["original_selection"]["primary_validation_leader"]
            or s.get("original_new_model_validation_leader") != context["new_model_validation_leader"]):
        raise ValueError("Validation-only extension ranking no longer reproduces")
    return p, s, context


def run_test(main, dataset, output):
    main, output = map(Path, (main, output))
    p, s, context = validate_selection(main, dataset, output, require_test=True)
    terminal_process(main, "test")
    completed = read_json(main / "test_completion.json")
    if completed.get("model_groups") != 4 or completed.get("baselines") != 7 or completed.get("frozen_spec_sha256") != p["original_frozen_spec_sha256"]:
        raise ValueError("Original complete172 test phase required first")
    if (output / "test_started.json").exists() or (output / "test").exists() or (output / "test_completion.json").exists():
        raise FileExistsError("Preserve the existing extension test attempt")
    candidate = s["candidate"]
    write_new(output / "test_started.json", dict(at=now(), selection_sha256=sha(output / "selection.json"), protocol_sha256=sha(output / "protocol.json"), main_test_started_sha256=sha(main / "test_started.json")))
    raw = {}
    if candidate["family"] == "cap075":
        run_gpu_groups(main, dataset, output, "test", [candidate["profile"]], p)
        mode = "cap075_evaluation"
    else:
        profile, algorithm = candidate["profile"], candidate["algorithm"]
        if context["original_selection"]["selected_profiles"][algorithm] != profile:
            raise ValueError("Raw reference must reuse the originally frozen test case")
        folder = f"evaluation/test/{algorithm}/{profile}"
        frame = goal.read_summary(main / folder / "summary.csv")
        goal.validate_rows(frame, split="test", algorithm=algorithm, profile=profile, spec=context["spec"])
        names = {f"{folder}/{f}" for f in GROUP_FILES}
        names.update(f"{folder}/{c}/{k}/{f}" for c in formal.COSTS for k in (*map(str, formal.SEEDS), "ensemble") for f in CASE_FILES[:6])
        raw = pin(main, names)
        mode = "raw_reference"
    write_new(output / "test_completion.json", dict(at=now(), protocol_sha256=sha(output / "protocol.json"), selection_sha256=sha(output / "selection.json"), mode=mode, case_count=12, repeat_count=12, main_test_completion_sha256=sha(main / "test_completion.json"), raw_source_sha256=raw))
    assemble(output, main)


def verify(main, dataset, output, audit_path=None):
    main, output = map(Path, (main, output))
    p, s, context = validate_selection(main, dataset, output, require_test=True)
    if (output / "goal_candidate_evidence.json").exists():
        raise FileExistsError("Preserve final extension goal evidence")
    audit_path = Path(audit_path) if audit_path else output / "independent_audit.json"
    audit_hash, audit = sha(audit_path), read_json(audit_path)
    if (audit.get("passed") is not True or audit.get("status") != "passed" or audit.get("test_complete") is not True
            or audit.get("failures") != [] or audit.get("failure_count", 0) != 0 or audit.get("run_count") != 36):
        raise ValueError("Complete independent extension audit required; accounting pass is not performance success")
    required = {"protocol.json", "protocol_receipt.json", "selection.json", "selection_receipt.json", "test_started.json", "test_completion.json", "summary.csv", "daily_equity.csv", "monthly_returns.csv"}
    if not required.issubset(audit.get("input_sha256", {})):
        raise ValueError("Independent audit must pin all extension contracts")
    check_pins(output, audit["input_sha256"], "Independent extension audit")
    check_pins(main, audit.get("main_input_sha256"), "Independent original study audit")
    main_audit_name = audit.get("main_final_audit_path")
    if not isinstance(main_audit_name, str) or main_audit_name not in audit["main_input_sha256"]:
        raise ValueError("Independent extension audit must pin the original final172 audit")
    main_audit = read_json(relative(main, main_audit_name))
    if (main_audit.get("passed") is not True or main_audit.get("status") != "passed"
            or main_audit.get("test_complete") is not True or main_audit.get("run_count") != 172
            or main_audit.get("failures") != []):
        raise ValueError("Original complete passed172 accounting audit required")
    done = read_json(output / "test_completion.json")
    if (done.get("case_count") != 12 or done.get("repeat_count") != 12 or done.get("selection_sha256") != sha(output / "selection.json")
            or done.get("main_test_completion_sha256") != sha(main / "test_completion.json")):
        raise ValueError("Frozen selected candidate test is incomplete")
    winner = s["candidate"]
    expected_ids = {f"validation__{c}__cap075__tqc__{pr}__{k}" for c in formal.COSTS for pr in PROFILES for k in (*map(str, formal.SEEDS), "ensemble")}
    if winner["family"] == "cap075":
        if done.get("mode") != "cap075_evaluation":
            raise ValueError("Cap selection needs actual capped tests")
        frame = goal.read_summary(output / "test" / winner["profile"] / "summary.csv")
        cap_rows(frame, "test", winner["profile"], context["spec"])
        if not extension_names("test", (winner["profile"],)).issubset(audit["input_sha256"]):
            raise ValueError("Final extension audit must pin all actual capped test cases")
    else:
        if done.get("mode") != "raw_reference":
            raise ValueError("Raw winner must reference original cases")
        check_pins(main, done["raw_source_sha256"], "Raw test reuse")
        frame = goal.read_summary(main / f"evaluation/test/{winner['algorithm']}/{winner['profile']}/summary.csv")
        goal.validate_rows(frame, split="test", algorithm=winner["algorithm"], profile=winner["profile"], spec=context["spec"])
        if not set(done["raw_source_sha256"]).issubset(audit["main_input_sha256"]):
            raise ValueError("Final extension audit must pin all referenced original test cases")
    expected_ids.update(frame.run_id)
    if set(audit.get("runs", {})) != expected_ids:
        raise ValueError("Independent audit must reconstruct exactly24 cap validation and12 selected test cases")
    for run_id in expected_ids:
        proof = audit["runs"][run_id]
        if proof.get("reconstructed") is not True or proof.get("hours") != (7080 if run_id.startswith("validation__") else 8760):
            raise ValueError("Independent actual ledger reconstruction is incomplete: " + run_id)
    baseline = goal.read_summary(main / "evaluation/test/baselines/summary.csv")
    combined = pd.concat([frame, baseline], ignore_index=True)
    metrics, gates = goal.performance_gates(combined, winner, p["research_targets"])
    audited_targets = audit.get("research_targets", {})
    if (audited_targets.get("evaluated") is not True or audited_targets.get("gates") != gates
            or audited_targets.get("goal_performance_achieved") is not all(gates.values())):
        raise ValueError("Independent audit research gates disagree with actual selected candidate")
    for key, value in metrics.items():
        actual = audited_targets.get("metrics", {}).get(key)
        if not same_metric(actual, value):
            raise ValueError("Independent audit research metrics disagree: " + key)
    result = dict(verified_at=now(), candidate={k: winner[k] for k in ("family", "algorithm", "profile")},
        validation_score=winner["score"], metrics=metrics, gates=gates, goal_performance_achieved=all(gates.values()),
        accounting_audit_passed=True, accounting_audit_is_not_performance_success=True,
        selection_uses_test=False, seed_selection=False, designed_after_validation=True,
        known_test_year_already_seen=True, original_global_and_new_only_selection_preserved=True,
        audit_sha256=audit_hash, protocol_sha256=sha(output / "protocol.json"), selection_sha256=sha(output / "selection.json"))
    check_pins(output, audit["input_sha256"], "Final extension audit")
    check_pins(main, audit["main_input_sha256"], "Final original audit")
    if sha(audit_path) != audit_hash:
        raise ValueError("Independent audit changed during verification")
    write_new(output / "goal_candidate_evidence.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=("prepare", "validate", "select", "test", "verify"))
    parser.add_argument("--main-output", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--diagnostic-dir", type=Path)
    parser.add_argument("--audit-path", type=Path)
    args = parser.parse_args()
    common = (args.main_output.resolve(), args.dataset.resolve(), args.output_dir.resolve())
    if args.phase == "prepare":
        if not args.diagnostic_dir:
            parser.error("prepare requires --diagnostic-dir")
        result = prepare(*common, args.diagnostic_dir)
    elif args.phase == "validate":
        result = run_validation(*common)
    elif args.phase == "select":
        result = select(*common)
    elif args.phase == "test":
        result = run_test(*common)
    else:
        result = verify(*common, args.audit_path)
    print(json.dumps(dict(phase=args.phase, completed=True, result=result), ensure_ascii=False, default=str), flush=True)


if __name__ == "__main__":
    main()

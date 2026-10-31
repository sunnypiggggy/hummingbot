"""Data-only annual TD3 queries, gated by complete independent evidence.

This reader imports no model, execution engine, selector or market pickle.
The ten TD3 policies used twice the earlier training budget.  Their annual
replay is an already-seen-year follow-up, never a fresh holdout, and cannot
replace the original208 or execution-band38 research scopes.
"""
from __future__ import annotations

import copy
import math
from pathlib import Path

import numpy as np
import pandas as pd

from scripts import btceth_rl_report_extension as evidence_helpers
from scripts.btceth_rl_report_extension import Sources, safe, sha, same_metrics

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ("2020-01-01T00:00:00Z", "2025-01-01T00:00:00Z")
VALIDATION = (TRAIN[1], "2025-10-23T00:00:00Z")
TEST = (VALIDATION[1], "2026-10-23T00:00:00Z")
PROFILES, BUDGETS = ("return", "balanced"), {"raw": 1., "cap075": .75}
SEEDS = tuple(range(5))
COSTS = {"base": {"fee": .001, "slippage": .0002}, "stress": {"fee": .002, "slippage": .0005}}
BASELINES = ("cash", "btc_hold", "eth_hold", "equal_hold", "equal_rebalance", "sma200_trend", "momentum180")
FLAGS = ("designed_after_first_annual", "test_year_already_seen", "not_fresh_holdout")
SCOPE = "td3_known_year_followup"
CASE_FILES = ("equity_1h.csv.gz", "fills.csv", "weights.csv", "boundary_equity.csv", "metadata.json", "fingerprints.json", "member_targets.csv", "replay.json")
GROUP_FILES = ("summary.csv", "daily_equity.csv", "monthly_returns.csv", "completion.json")
FINANCIAL = ("scripts/backtest_btceth_rl.py", "scripts/btceth_rl_env.py", "scripts/crypto_rl_portfolio_env.py", "scripts/crypto_rl_reporting.py")
SOURCE_FILES = (*FINANCIAL, "scripts/audit_btceth_rl.py", "scripts/audit_btceth_rl_exposure_cap.py", "scripts/audit_btceth_rl_execution_band.py", "scripts/btceth_rl_td3.py", "scripts/audit_btceth_rl_td3.py")
TARGETS = {"primary_base_net_return_at_least": .05, "primary_base_max_drawdown_magnitude_at_most": .20,
    "primary_stress_net_return_at_least": 0., "primary_beats_equal_hold_return_and_drawdown": True,
    "positive_base_training_seeds_at_least": 3, "stronger_claim_requires_beating_simple_trend_controls": True,
    "targets_are_not_guarantees": True}
RECIPE = dict(policy="MlpPolicy", learning_rate=.0003, gamma=.995, buffer_size=100000,
    learning_starts=512, batch_size=128, tau=.005, train_freq=[1, "step"], gradient_steps=1,
    n_steps=1, policy_delay=2, target_policy_noise=.2, target_noise_clip=.5,
    optimize_memory_usage=False, replay_buffer_kwargs={"handle_timeout_termination": True},
    policy_kwargs=dict(net_arch=[128, 128], activation_fn="ReLU", optimizer_class="Adam", n_critics=2, share_features_extractor=False),
    action_noise=dict(class_name="NormalActionNoise", mean=[0., 0.], sigma=[.1, .1], dtype="float32",
        units="normalized Box[-1,1] action units, not portfolio percentage points"))
UPSTREAM_COMMIT = "8908708f10c8ff29759c67f55c8acb56cab27463"
UPSTREAM_SHA256 = {
    "stable_baselines3/td3/td3.py": "45107ef95f85cca1224741b666a3de2afa57838a501cc301331fd7e9a41cc2e0",
    "stable_baselines3/td3/policies.py": "5b81fa2347e9e3ce85956b570cb3008f58778d62ad1598c3bc89217c3a143062",
    "stable_baselines3/common/off_policy_algorithm.py": "ec895fa4b7ce96550b0acdfea3a73561db568f863b24c0b77d106f3f67e540a7",
    "stable_baselines3/common/buffers.py": "21db27f000048615f5848facf18af9afc22e6e2a1c26148911c2cba542d1988d",
    "stable_baselines3/common/noise.py": "335d9e322c8b5bf1b43703ac284951a01bb7980822cb30c8c2b94b0f48dc4e1f",
}
LICENSE_SHA256 = "2d8fbeee75e0acd13e75b30470b3521b12fb55c26c90c2060c6bd525345870a9"
SETUP_SHA256 = "7c407d274eddac9884e3beb4ad7718a14cbb8a792940903384894ee9ea8d1e7e"
MANIFEST_SHA256 = "7ea8c1248651934265a4fe087b1979c231f6b0ab51d2180c2c9b3a55121853d9"


def utc(value):
    value = pd.Timestamp(value)
    if value.tzinfo is None:
        raise ValueError("TD3 evidence requires timezone-aware timestamps")
    return value.tz_convert("UTC")


def positive_int(value):
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def group_files(split, family, profile):
    folder = f"{split}/{family}/{profile}"
    return {*(f"{folder}/{name}" for name in GROUP_FILES),
        *(f"{folder}/{cost}/{key}/{name}" for cost in COSTS for key in (*map(str, SEEDS), "ensemble") for name in CASE_FILES)}


def baseline_files():
    folder = "evaluation/test/baselines"
    return {*(f"{folder}/{name}" for name in GROUP_FILES),
        *(f"{folder}/{cost}/{key}/{name}" for cost in COSTS for key in BASELINES for name in CASE_FILES[:6])}


def training_files():
    return {"training_completion.json", *(f"training/td3__{profile}__seed{seed}/{name}"
        for profile in PROFILES for seed in SEEDS for name in ("started.json", "record.json", "model.zip", "training_timeout_transitions.jsonl"))}


def validation_files():
    return {"validation_completion.json", "validate_completion.json", "validate_process.json", "validate.log",
        *(f"validation/{name}" for name in GROUP_FILES[:3]),
        *(name for family in BUDGETS for profile in PROFILES for name in group_files("validation", family, profile))}


def _audit(value, *, validation=False):
    wanted = dict(passed=True, status="partial_test_pending" if validation else "passed", phase="validation" if validation else "test",
        test_complete=not validation, validation_complete=True, training_model_count=10, validation_run_count=48,
        test_run_count=0 if validation else 12, baseline_run_count=0 if validation else 14,
        run_count=48 if validation else 26, repeat_count=48 if validation else 12, failure_count=0, failures=[],
        fresh_holdout_success=False, training_budget_multiple=2, accounting_passage_is_not_goal_success=True,
        **{flag: True for flag in FLAGS})
    if (any(value.get(k) != v or isinstance(v, bool) and value.get(k) is not v for k, v in wanted.items()) or value.get("declared_research_targets") != TARGETS
            or not positive_int(value.get("checks_total"))
            or any(not positive_int(value[k]) for k in ("checks", "checks_total") if k in value)
            or ("checks" in value and value["checks"] != value["checks_total"])):
        raise ValueError("Complete independent TD3 annual26 required; validation48 remains partial_test_pending")


def _bind(sources, roots, value, *, study_required=(), main_required=()):
    sources.pins(roots["study"], value.get("input_sha256"), required=study_required, label="TD3 study audit")
    sources.pins(roots["main"], value.get("main_input_sha256"), required=main_required, label="TD3 original-main audit")
    sources.pins(roots["source"], value.get("source_sha256"), required=SOURCE_FILES, label="TD3 audit implementation")


def _protocol(sources, study, main, source_root):
    p, receipt, old = (sources.json(path) for path in (study / "protocol.json", study / "protocol_receipt.json", main / "frozen_spec.json"))
    fixed = dict(algorithm="td3", profiles={"return": {"drawdown_penalty": 0., "rebalance_band": .025}, "balanced": {"drawdown_penalty": .25, "rebalance_band": .025}},
        budgets=BUDGETS, seeds=list(SEEDS), train=list(TRAIN), validation=list(VALIDATION), test=list(TEST), costs=COSTS, capital=10000.,
        timesteps_per_model=65536, old_timesteps_per_model=32768, training_budget_multiple=2, train_episode_days=32,
        model_count=10, validation_cases=48, validation_repeats=48, test_cases=12, test_repeats=12,
        cash_floor=.05, max_asset_weight=.95, hyperparameters=RECIPE, recipe_is_pure_official_default=False,
        seed_selection=False, selection_uses_test=False, new_recipe_search=False, annual_requires_separate_authorization=True,
        research_targets=TARGETS, selection_score="mean(log1p(net_return)-0.5*abs(max_drawdown)-2*max(abs(max_drawdown)-0.20,0)) across base/stress",
        tie_break="raw before cap075, then profile lexical identifier", **{flag: True for flag in FLAGS})
    if (any(p.get(k) != v for k, v in fixed.items()) or receipt.get("protocol_file") != "protocol.json"
            or receipt.get("protocol_sha256") != sha(study / "protocol.json") or utc(p["prepared_at"]) > utc(receipt["sealed_at"])
            or p.get("original_frozen_spec_sha256") != sha(main / "frozen_spec.json")
            or old.get("timesteps_per_model") != 32768 or old.get("research_targets") != TARGETS
            or any(p.get(k) != old.get(k) for k in ("dataset_sha256", "dataset_manifest_sha256"))):
        raise ValueError("Fixed single TD3 recipe, twofold budget, original data and whole-file freeze required")
    pins = p.get("source_sha256", {})
    if set(pins) != set(SOURCE_FILES) or any(pins[name] != old.get("code_sha256", {}).get(name) for name in FINANCIAL):
        raise ValueError("Exact nine-source DAG must preserve the original four financial files")
    sources.pins(source_root, pins, required=SOURCE_FILES, label="Actual TD3 sources")
    sources.pins(study / "sources", pins, required=SOURCE_FILES, label="Frozen TD3 source copies")
    sources.pins(main, p.get("original_input_sha256"), required={"frozen_spec.json", "runtime.json", "independent_audit.json"} | baseline_files(), label="Predeclared original controls")
    runtime = sources.json(main / "runtime.json")
    gpus = p.get("physical_gpus", [])
    if (gpus != runtime.get("physical_gpus") or len(gpus) != 2 or len({g["uuid"] for g in gpus}) != 2
            or any("V100" not in g["name"] for g in gpus)
            or p.get("profile_gpu_assignments") != {profile: gpu["uuid"] for profile, gpu in zip(PROFILES, gpus)}
            or p.get("runtime_packages") != old.get("runtime_packages")):
        raise ValueError("Original research dependencies and two physical profile-assigned V100s required")
    upstream_files = {*UPSTREAM_SHA256, "LICENSE", "setup.py", "verified_manifest.json", "github_manifest.json", "installed_distribution_observation.json"}
    sources.pins(study / "upstream", p.get("upstream_evidence_sha256"), required=upstream_files, label="Official TD3 source and MIT evidence")
    if set(p["upstream_evidence_sha256"]) != upstream_files:
        raise ValueError("Complete reviewed upstream source/license provenance required")
    manifest = sources.json(study / "upstream/verified_manifest.json")
    expected_sources = {**UPSTREAM_SHA256, "LICENSE": LICENSE_SHA256, "setup.py": SETUP_SHA256}
    if (p.get("sb3_source_sha256") != UPSTREAM_SHA256 or manifest.get("commit") != UPSTREAM_COMMIT or manifest.get("license") != "MIT"
            or sha(study / "upstream/verified_manifest.json") != MANIFEST_SHA256
            or p.get("upstream", {}).get("commit") != UPSTREAM_COMMIT or p.get("upstream", {}).get("version") != "2.9.0"
            or p.get("upstream", {}).get("verified_manifest_sha256") != MANIFEST_SHA256
            or set(manifest.get("files", {})) != set(expected_sources)
            or any(p["upstream_evidence_sha256"][name] != digest or manifest["files"][name].get("sha256") != digest for name, digest in expected_sources.items())):
        raise ValueError("Actual SB3 2.9.0 five-source identity and preserved MIT notice required")
    return p, receipt


def _phase(sources, study, phase, p, receipt):
    process, complete, preflight = (sources.json(study / name) for name in (phase + "_process.json", phase + "_completion.json", "preflight/" + phase + ".json"))
    command = process.get("command", [])
    if (process.get("phase") != phase or process.get("status") != "completed" or process.get("exit_code") != 0
            or process.get("protocol_sha256") != sha(study / "protocol.json") or complete.get("protocol_sha256") != sha(study / "protocol.json")
            or process.get("source_sha256") != p["source_sha256"]["scripts/btceth_rl_td3.py"]
            or len(command) < 5 or not str(command[2]).replace("\\", "/").endswith("/btceth_rl_td3.py")
            or "--phase" not in command or command[command.index("--phase") + 1] != "_" + phase
            or not positive_int(process.get("supervisor_pid")) or not positive_int(process.get("child_pid")) or process["supervisor_pid"] == process["child_pid"]
            or process.get("log_relative_path") != phase + ".log" or process.get("log_sha256") != sha(sources.file(study / (phase + ".log")))
            or preflight.get("physical_gpus") != p["physical_gpus"]
            or not (utc(receipt["sealed_at"]) <= utc(preflight["at"]) <= utc(process["started_at"]) <= utc(complete["at"]) <= utc(process["completed_at"]))):
        raise ValueError("Real terminal0 process/log/time chain required: " + phase)
    if phase == "fit":
        good = complete.get("model_count") == 10
    else:
        count = 48 if phase == "validate" else 12
        good = complete.get("case_count") == complete.get("repeat_count") == count
    if not good:
        raise ValueError("Incomplete declared TD3 phase: " + phase)
    return process, complete


def _training(sources, study, p, fit, audits):
    completion = sources.json(study / "training_completion.json")
    rows = completion.get("records", [])
    by_key = {(r.get("profile"), r.get("seed")): r for r in rows}
    if (completion.get("model_count") != 10 or len(rows) != 10 or len(by_key) != 10
            or set(by_key) != {(profile, seed) for profile in PROFILES for seed in SEEDS}
            or completion.get("protocol_sha256") != sha(study / "protocol.json")):
        raise ValueError("All ten new trained policies must remain present")
    effective = {k: copy.deepcopy(v) for k, v in RECIPE.items() if k != "policy"}
    effective["action_noise"].pop("units")
    result = []
    for profile in PROFILES:
        for seed in SEEDS:
            folder = f"training/td3__{profile}__seed{seed}"
            record, started = (sources.json(study / folder / name) for name in ("record.json", "started.json"))
            if (record != by_key[(profile, seed)] or (record.get("algorithm"), record.get("profile"), record.get("seed")) != ("td3", profile, seed)
                    or record.get("actual_timesteps") != 65536 or record.get("actual_n_updates") != 65024 or record.get("actor_update_count") != 32512
                    or record.get("hyperparameters") != RECIPE or record.get("effective_hyperparameters") != effective
                    or record.get("training_budget_multiple") != 2 or record.get("old_timesteps_per_model") != 32768
                    or record.get("recipe_is_pure_official_default") is not False or record.get("sb3_source_sha256") != UPSTREAM_SHA256
                    or record.get("algorithm_class") != "stable_baselines3.td3.td3.TD3" or record.get("policy_class") != "TD3Policy"
                    or record.get("training_buffer_class") != "AuditedReplayBuffer"
                    or record.get("protocol_sha256") != started.get("protocol_sha256") or record.get("protocol_sha256") != sha(study / "protocol.json")
                    or any(record.get(flag) is not True for flag in FLAGS)):
                raise ValueError("Actual ten-model training identity, exact budget and effective recipe required")
            for name, expected in (("model.zip", record.get("model_sha256")), ("training_timeout_transitions.jsonl", record.get("timeout_trace_sha256"))):
                if sha(sources.file(study / folder / name)) != expected or record.get("model_relative_path" if name == "model.zip" else "timeout_trace_relative_path") != folder + "/" + name:
                    raise ValueError("Original TD3 checkpoint/timeout evidence changed")
            device, semantics = record.get("device", {}), record.get("replay_buffer_semantics", {})
            if (device != started.get("device") or device.get("visible_gpu_uuid") != p["profile_gpu_assignments"][profile]
                    or device.get("device") != "cuda:0" or device.get("capability") != [7, 0] or "V100" not in device.get("name", "")
                    or semantics != dict(optimize_memory_usage=False, handle_timeout_termination=True, observations_captured_after_super_add=True)
                    or not (utc(fit["started_at"]) <= utc(started["at"]) <= utc(record["completed_at"]) <= utc(fit["completed_at"]))):
                raise ValueError("Actual profile GPU, timeout bootstrap and training time chain required")
            quality = record.get("training_window_quality", {})
            for audit in audits:
                calendar = audit.get("training_calendar", {})
                if (any(quality.get(k) != calendar.get(k) for k in ("requested_train_days", "candidate_windows", "eligible_windows", "excluded_windows"))
                        or quality.get("covered_training_days") != calendar.get("covered_days") or not positive_int(calendar.get("covered_days"))):
                    raise ValueError("Independent complete causal TRAIN calendar must agree with every model")
            result.append(dict(algorithm="td3", profile=profile, seed=seed, actual_timesteps=65536, actual_n_updates=65024, actor_update_count=32512,
                training_budget_multiple=2, old_timesteps_per_model=32768, recipe_is_pure_official_default=False,
                gpu_uuid=device["visible_gpu_uuid"], model_sha256=record["model_sha256"], parameter_sha256=record["parameter_sha256"],
                covered_training_days=quality["covered_training_days"], eligible_windows=quality["eligible_windows"],
                started_at=started["at"], completed_at=record["completed_at"], scope=SCOPE))
    return result


def _summary(frame, *, split, family=None, profile=None, baseline=False):
    dates = VALIDATION if split == "validation" else TEST
    expected = {f"test__{cost}__{key}" for cost in COSTS for key in BASELINES} if baseline else {
        f"{split}__{cost}__{family}__td3__{profile}__{key}" for cost in COSTS for key in (*map(str, SEEDS), "ensemble")}
    required = {"run_id", "strategy", "scenario", "split", "initial_capital", "start", "end_exclusive", "net_return", "max_drawdown", "final_equity", "fees", "average_exposure"}
    if (not required.issubset(frame) or len(frame) != len(expected) or frame.run_id.duplicated().any() or set(frame.run_id) != expected
            or not frame.split.eq(split).all() or not frame.initial_capital.eq(10000.).all()
            or not pd.to_datetime(frame.start, utc=True).eq(utc(dates[0])).all() or not pd.to_datetime(frame.end_exclusive, utc=True).eq(utc(dates[1])).all()):
        raise ValueError("Complete fixed TD3/baseline population, window and initial capital required")
    values = frame[["net_return", "max_drawdown", "final_equity", "fees", "average_exposure"]].to_numpy(dtype=float)
    if (not np.isfinite(values).all() or (frame.net_return <= -1).any() or (frame.final_equity <= 0).any()
            or not frame.max_drawdown.between(-1, 0, inclusive="right").all() or (frame.fees < 0).any()
            or not frame.average_exposure.between(0, 1).all() or not np.allclose(frame.final_equity / 10000 - 1, frame.net_return, atol=1e-11, rtol=1e-11)):
        raise ValueError("Reported TD3 financial metrics must be finite actual funded accounts")
    if baseline:
        if any(row.strategy not in BASELINES or row.run_id != f"test__{row.scenario}__{row.strategy}" for row in frame.itertuples()):
            raise ValueError("Baseline identities cannot be relabeled as learned TD3")
    else:
        fixed = dict(algorithm="td3", family=family, profile=profile, strategy="td3_" + family, exposure_multiplier=BUDGETS[family], rebalance_band=.025)
        if (not {*fixed, "seed", "primary_policy"}.issubset(frame) or any(not frame[k].eq(v).all() for k, v in fixed.items())
                or not frame.primary_policy.map(lambda v: isinstance(v, (bool, np.bool_))).all()):
            raise ValueError("Exact selected TD3 execution/model identity required")
        for cost in COSTS:
            group = frame.loc[frame.scenario.eq(cost)]
            primary, seeds = group.loc[group.primary_policy], group.loc[~group.primary_policy]
            if len(primary) != 1 or primary.seed.notna().any() or len(seeds) != 5 or set(seeds.seed) != set(SEEDS):
                raise ValueError("All five seeds and one genuine common-account ensemble required")
    return expected


def _history(history, months, summary):
    if (not {"run_id", "timestamp", "equity", "net_return", "drawdown"}.issubset(history)
            or not {"run_id", "month", "return"}.issubset(months) or set(history.run_id) != set(summary.run_id) or set(months.run_id) != set(summary.run_id)):
        raise ValueError("Complete annual curve and monthly population required")
    dates = pd.date_range(*TEST, freq="D")
    for row in summary.itertuples():
        curve, monthly = history.loc[history.run_id.eq(row.run_id)], months.loc[months.run_id.eq(row.run_id)]
        if (not pd.DatetimeIndex(pd.to_datetime(curve.timestamp, utc=True)).equals(dates)
                or not np.isfinite(curve[["equity", "net_return", "drawdown"]]).all().all() or (curve.equity <= 0).any()
                or (curve.drawdown > 0).any() or not np.isclose(curve.equity.iloc[0], 10000., atol=1e-8, rtol=0)
                or not np.isclose(curve.equity.iloc[-1], row.final_equity, atol=1e-7, rtol=1e-12)
                or not np.allclose(curve.net_return, curve.equity / 10000 - 1, atol=1e-11, rtol=1e-11)
                or row.max_drawdown > float(curve.drawdown.min()) + 1e-10):
            raise ValueError("Every annual boundary must reconcile with actual NAV and hourly-audited drawdown")
        daily = pd.Series(curve.equity.iloc[1:].to_numpy(), index=dates[1:] - pd.Timedelta(days=1))
        ends = daily.resample("ME").last(); expected = ends.pct_change(); expected.iloc[0] = ends.iloc[0] / 10000 - 1
        if (monthly.month.tolist() != expected.index.strftime("%Y-%m").tolist() or not np.isfinite(monthly["return"]).all()
                or not np.allclose(monthly["return"], expected, atol=1e-10, rtol=1e-10)
                or not np.isclose(np.prod(1 + monthly["return"]) - 1, row.net_return, atol=1e-10, rtol=1e-10)):
            raise ValueError("Annual monthly returns must reconcile with daily boundaries and final NAV")


def _runs(value, name, summary, *, hours):
    runs = value.get(name, {})
    if not isinstance(runs, dict) or set(runs) != set(summary.run_id):
        raise ValueError("Complete independently reconstructed account identities required")
    for row in summary.itertuples():
        proof = runs[row.run_id]
        if proof.get("reconstructed") is not True or proof.get("hours") != hours or not isinstance(proof.get("fingerprints"), dict) or not proof["fingerprints"]:
            raise ValueError("Full hourly account reconstruction required")
        for key in ("net_return", "max_drawdown", "fees", "final_equity"):
            same_metrics(proof.get(key), float(getattr(row, key)), "Independently audited TD3 metric:" + key)
    return runs


def _candidate(frame, family, profile):
    _summary(frame, split="validation", family=family, profile=profile)
    metrics = {cost: frame.loc[frame.scenario.eq(cost) & frame.primary_policy].iloc[0] for cost in COSTS}
    def one(row):
        dd = abs(row.max_drawdown)
        return math.log1p(row.net_return) - .5 * dd - 2 * max(dd - .20, 0.)
    return dict(algorithm="td3", family=family, profile=profile, exposure_multiplier=BUDGETS[family], rebalance_band=.025,
        score=sum(one(row) for row in metrics.values()) / 2,
        **{cost: dict(run_id=row.run_id, **{key: float(getattr(row, key)) for key in ("net_return", "max_drawdown", "fees", "final_equity")}) for cost, row in metrics.items()})


def _targets(frame, baselines, chosen):
    base = frame.loc[frame.scenario.eq("base") & frame.primary_policy].iloc[0]
    stress = frame.loc[frame.scenario.eq("stress") & frame.primary_policy].iloc[0]
    seeds = frame.loc[frame.scenario.eq("base") & ~frame.primary_policy].sort_values("seed")
    equal = baselines.loc[baselines.run_id.eq("test__base__equal_hold")].iloc[0]
    positive = int(seeds.net_return.gt(0).sum())
    gates = dict(base_return_at_least_5pct=bool(base.net_return >= .05), base_max_drawdown_at_most_20pct=bool(abs(base.max_drawdown) <= .20),
        stress_return_nonnegative=bool(stress.net_return >= 0), at_least_three_positive_training_seeds=positive >= 3,
        beats_equal_hold_return_and_drawdown=bool(base.net_return > equal.net_return and base.max_drawdown > equal.max_drawdown))
    trends = {name: bool(base.net_return > baselines.loc[baselines.run_id.eq("test__base__" + name)].iloc[0].net_return
        and base.max_drawdown > baselines.loc[baselines.run_id.eq("test__base__" + name)].iloc[0].max_drawdown) for name in ("sma200_trend", "momentum180")}
    return dict(candidate={key: chosen[key] for key in ("algorithm", "family", "profile")}, gates=gates,
        goal_performance_achieved=all(gates.values()), all_five_diagnostic_targets_met=all(gates.values()),
        base_net_return=float(base.net_return), base_max_drawdown=float(base.max_drawdown), stress_net_return=float(stress.net_return), stress_max_drawdown=float(stress.max_drawdown),
        positive_base_training_seeds=positive, base_seed_returns=[dict(seed=int(row.seed), net_return=float(row.net_return)) for row in seeds.itertuples()],
        base_fees_usdt=float(base.fees), stress_fees_usdt=float(stress.fees), beats_simple_trend_controls=trends, stronger_control_comparison_met=all(trends.values()),
        fresh_holdout_success=False, training_budget_multiple=2, recipe_is_pure_official_default=False, accounting_passage_is_not_goal_success=True,
        **{flag: True for flag in FLAGS})


def load_td3(study_root, main_root, *, audit_path=None, source_root=ROOT):
    """Return seven future annual query sets only after full26/48/10 gates.

    Baseline14 rows are reused reference accounts, not fourteen additional
    policy runs.  Validation scores retain their sealed Linux bytes for exact
    tie ordering; their numerical values are independently checked tolerantly.
    """
    study, main, source_root = map(lambda p: Path(p).resolve(), (study_root, main_root, source_root))
    if study == main or study.is_relative_to(main) or main.is_relative_to(study):
        raise ValueError("TD3 needs an independent research tree, preserving previous studies")
    sources = Sources(); roots = dict(study=study, main=main, source=source_root)
    selected_path = Path(audit_path).resolve() if audit_path else study / "independent_audit.json"
    selected_path.relative_to(study)
    audits, audit_paths = [], list(dict.fromkeys((selected_path, study / "independent_audit.json")))
    for path in audit_paths:
        audit = sources.json(path); _audit(audit)
        _bind(sources, roots, audit, study_required={"protocol.json", "protocol_receipt.json", "selection.json", "selection_receipt.json", "independent_audit.validation.json", "annual_authorization.json", "test_started.json", "test_process.json", "test_completion.json"} | training_files() | validation_files(),
            main_required={"frozen_spec.json", "runtime.json", "independent_audit.json"} | baseline_files())
        audits.append(audit)
    p, protocol_receipt = _protocol(sources, study, main, source_root)
    if any(audit.get("dataset_sha256") != p["dataset_sha256"] for audit in audits):
        raise ValueError("Every audit must describe the same original frozen dataset")
    fit, _ = _phase(sources, study, "fit", p, protocol_receipt)
    validate, _ = _phase(sources, study, "validate", p, protocol_receipt)
    test, completion = _phase(sources, study, "test", p, protocol_receipt)
    initial = sources.json(study / "independent_audit.validation.json"); _audit(initial, validation=True)
    _bind(sources, roots, initial, study_required=validation_files() | training_files(), main_required={"frozen_spec.json", "runtime.json"})
    training = _training(sources, study, p, fit, [*audits, initial])
    selection, seal = (sources.json(study / name) for name in ("selection.json", "selection_receipt.json"))
    if (seal.get("selection_file") != "selection.json" or seal.get("selection_sha256") != sha(study / "selection.json")
            or seal.get("protocol_sha256") != selection.get("protocol_sha256") or selection.get("protocol_sha256") != sha(study / "protocol.json")
            or selection.get("audit_sha256") != sha(study / "independent_audit.validation.json")
            or selection.get("seed_selection") is not False or selection.get("selection_uses_test") is not False
            or selection.get("test_results_available_to_selector") is not False or any(selection.get(flag) is not True for flag in FLAGS)):
        raise ValueError("Immutable validation-only four-candidate selection and receipt required")
    selected_training = training_files() - {name for name in training_files() if name.endswith("/started.json")}
    sources.pins(study, selection.get("training_source_sha256"), required=selected_training, label="Sealed ten TD3 models")
    sources.pins(study, selection.get("validation_source_sha256"), required=validation_files(), label="Sealed all48 TD3 validation accounts")
    actual, validation_frames = {}, []
    for family in BUDGETS:
        for profile in PROFILES:
            frame = sources.csv(study / f"validation/{family}/{profile}/summary.csv")
            actual[(family, profile)] = _candidate(frame, family, profile)
            validation_frames.append(frame)
    _runs(initial, "runs", pd.concat(validation_frames, ignore_index=True), hours=7080)
    pool = selection.get("candidates", [])
    if len(pool) != 4 or {(r.get("family"), r.get("profile")) for r in pool} != set(actual):
        raise ValueError("Exactly four predeclared TD3 validation candidates required")
    for item in pool:
        same_metrics(item, actual[(item["family"], item["profile"])], "Independently recomputed TD3 validation candidate")
    order = sorted(pool, key=lambda r: (-r["score"], 0 if r["family"] == "raw" else 1, r["profile"]))
    if pool != order or selection.get("candidate") != pool[0]:
        raise ValueError("Frozen score order must preserve raw-first exact ties, without test selection")
    chosen = selection["candidate"]
    annual_folder = f"test/{chosen['family']}/{chosen['profile']}"
    for audit in audits:
        sources.pins(study, audit["input_sha256"], required=group_files("test", chosen["family"], chosen["profile"]) | {"preflight/test.json", "test.log"}, label="Selected12 annual TD3 accounts")
    auth, marker = (sources.json(study / name) for name in ("annual_authorization.json", "test_started.json"))
    if (auth.get("annual_evaluation_authorized") is not True or auth.get("no_live_trading") is not True or auth.get("financial_parameters_fixed") is not True
            or auth.get("selection_sha256") != sha(study / "selection.json") or any(auth.get(flag) is not True for flag in FLAGS)
            or marker.get("protocol_sha256") != sha(study / "protocol.json") or marker.get("selection_sha256") != sha(study / "selection.json")
            or marker.get("authorization_sha256") != sha(study / "annual_authorization.json") or completion.get("candidate") != chosen
            or completion.get("selection_sha256") != sha(study / "selection.json")
            or not (utc(fit["completed_at"]) <= utc(validate["started_at"]) <= utc(validate["completed_at"]) <= utc(initial["audited_at"])
                <= utc(selection["selected_at"]) <= utc(seal["sealed_at"]) <= utc(auth["authorized_at"]) <= utc(marker["at"]) <= utc(test["started_at"]))):
        raise ValueError("Complete validation and sealed choice must precede separately authorized new annual replay")
    main_audit = sources.json(main / "independent_audit.json")
    if (main_audit.get("passed") is not True or main_audit.get("status") != "passed" or main_audit.get("test_complete") is not True
            or main_audit.get("run_count") != 172 or not isinstance(main_audit.get("runs"), dict) or len(main_audit["runs"]) != 172
            or not positive_int(main_audit.get("checks_total")) or main_audit.get("failure_count") != 0 or main_audit.get("failures") != []):
        raise ValueError("The retained original172 reference audit must remain complete")
    sources.pins(main, main_audit.get("input_sha256"), required={"frozen_spec.json", "test_completion.json"} | baseline_files(), label="Original172 completion and controls")
    annual_tables = [sources.csv(study / annual_folder / name) for name in GROUP_FILES[:3]]
    baseline_tables = [sources.csv(main / "evaluation/test/baselines" / name) for name in GROUP_FILES[:3]]
    summary, baseline = annual_tables[0], baseline_tables[0]
    _summary(summary, split="test", family=chosen["family"], profile=chosen["profile"])
    _summary(baseline, split="test", baseline=True)
    for tables in (annual_tables, baseline_tables):
        _history(tables[1], tables[2], tables[0])
    targets = _targets(summary, baseline, chosen)
    for audit in audits:
        _runs(audit, "runs", summary, hours=8760); refs = _runs(audit, "baseline_runs", baseline, hours=8760)
        if (utc(audit["audited_at"]) < utc(test["completed_at"]) or any(refs[key]["fingerprints"] != main_audit.get("runs", {}).get(key, {}).get("fingerprints") for key in refs)):
            raise ValueError("Annual audit must follow real completion and preserve original baseline fingerprints")
        same_metrics(audit.get("research_targets"), targets, "Actual TD3 annual research conditions")
    goal = sources.json(study / "goal_candidate_evidence.json")
    if (goal.get("protocol_sha256") != sha(study / "protocol.json") or goal.get("selection_sha256") != sha(study / "selection.json")
            or goal.get("audit_sha256") != sha(study / "independent_audit.json") or goal.get("accounting_audit_passed") is not True
            or utc(goal["at"]) < utc(audits[audit_paths.index(study / "independent_audit.json")]["audited_at"])):
        raise ValueError("Goal evidence must retain its actual remote audit identity, independent of local re-audit")
    for key, expected in targets.items():
        same_metrics(goal.get(key), expected, "Own TD3 goal evidence:" + key)
    sources.file(Path(__file__))
    sources.file(Path(evidence_helpers.__file__))
    tables = [[], [], []]
    for role, source_tables in (("td3_selected", annual_tables), ("original_baseline_reference", baseline_tables)):
        for frame, dest in zip(source_tables, tables):
            item = frame.copy(); item["scope"] = SCOPE; item["td3_role"] = role; item["training_budget_multiple"] = 2 if role == "td3_selected" else None
            dest.append(item)
    target_rows = [dict(gate=key, met=value, category="已观察年度的TD3自身诊断") for key, value in targets["gates"].items()]
    target_rows.extend(dict(gate=key, met=value, category="简单趋势对照") for key, value in targets["beats_simple_trend_controls"].items())
    audit_rows = [dict(role="本机明确审计" if i == 0 and path != study / "independent_audit.json" else "保留远端审计", file=path.as_posix(), sha256=sha(path),
        passed=True, new_runs=12, baseline_reference_runs=14, validation_runs=48, training_models=10, repeat_count=12, fresh_holdout_success=False) for i, path in enumerate(audit_paths)]
    frames = dict(td3_summary=pd.concat(tables[0], ignore_index=True), td3_equity=pd.concat(tables[1], ignore_index=True), td3_monthly=pd.concat(tables[2], ignore_index=True),
        td3_targets=pd.DataFrame(target_rows), td3_validation_selection=pd.DataFrame(pool), td3_training=pd.DataFrame(training), td3_audit=pd.DataFrame(audit_rows))
    source = dict(label="TD3双倍环境步数预算与已观察年度的独立研究", classification="modeled", timezone="UTC", period=dict(start=TEST[0], endExclusive=TEST[1]),
        files=list(sources.hashes), fileHashes=dict(sources.hashes), datasetSha256=p["dataset_sha256"],
        caveats=["65536环境步×10模型，是此前每模型32768环境步的2倍；实际评价和策略更新次数分别披露，不能等同于测得的2倍计算开销；共享骨架不是TD3纯官方默认参数。", "新年度已观察，不是fresh holdout；不能改写原208、5pp38或原失败目标。", "年度26行含12个新TD3账户和14个原基准引用；基准并非新增模型或新增资金。", "四个候选只用295天验证选择；每个成本保留五个种子和一个平均目标的真实资金组合。", "SAC/TQC仅可从先前审计报告另作对照，不能替代TD3自身种子稳定性条件。", "回撤来自完整小时权益独立重建；曲线展示完整日边界。"])
    queries = {}
    for name, frame in frames.items():
        context = copy.deepcopy(source)
        if name in ("td3_training", "td3_validation_selection"):
            dates = TRAIN if name == "td3_training" else VALIDATION
            context["period"] = dict(start=dates[0], endExclusive=dates[1])
        queries[name] = dict(rows=frame.astype(object).where(pd.notna(frame), None).to_dict("records"), source=context,
            methods=[dict(language="text", code="Read-only, fully SHA-bound TD3 evidence. Complete independent annual26 plus prior48 validation, all10 checkpoints, sealed four-candidate selection, separate known-year authorization and own recomputed goal required. No inference, execution, price loading or prior-report mutation.")])
    sources.unchanged()
    return dict(queries=queries, source=source, targets=targets, selection=selection, audit=audits[0], external_input_sha256=dict(sources.hashes),
        query_run_count=26, new_run_count=12, baseline_reference_run_count=14, validation_run_count=48, training_model_count=10,
        original_scopes_unchanged=True, fresh_holdout_success=False, training_budget_multiple=2)

"""Data-only follow-up queries for a complete, separately audited 5pp replay.

No inference, pickle loading, selector, HTML writer or first-report mutation.
The later design uses an already observed year; it cannot repair the first
experiment's failed goal or provide fresh-holdout evidence.
"""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.btceth_rl_report_extension import Sources, safe, sha, same_metrics

ROOT = Path(__file__).resolve().parents[1]
TEST = ("2025-10-23T00:00:00Z", "2026-10-23T00:00:00Z")
FAMILY, STRATEGY = "cap075_band05", "tqc_cap075_band05"
COSTS = {"base": {"fee": .001, "slippage": .0002}, "stress": {"fee": .002, "slippage": .0005}}
BASELINES = ("cash", "btc_hold", "eth_hold", "equal_hold", "equal_rebalance", "sma200_trend", "momentum180")
FLAGS = ("designed_after_first_annual", "test_year_already_seen", "not_fresh_holdout")
TARGETS = {"primary_base_net_return_at_least": .05, "primary_base_max_drawdown_magnitude_at_most": .20,
    "primary_stress_net_return_at_least": 0., "primary_beats_equal_hold_return_and_drawdown": True,
    "positive_base_training_seeds_at_least": 3, "stronger_claim_requires_beating_simple_trend_controls": True,
    "targets_are_not_guarantees": True}
CASE_FILES = ("equity_1h.csv.gz", "fills.csv", "weights.csv", "boundary_equity.csv", "metadata.json", "fingerprints.json", "member_targets.csv", "replay.json")
GROUP_FILES = ("summary.csv", "daily_equity.csv", "monthly_returns.csv", "completion.json")
CONTRACTS = ("protocol.json", "protocol_receipt.json", "test_started.json", "test_completion.json", "runner_process.json", "summary.csv", "daily_equity.csv", "monthly_returns.csv")
REQUIRED_SOURCES = ("scripts/btceth_rl_execution_band_annual.py", "scripts/audit_btceth_rl_execution_band_annual.py", "scripts/btceth_rl_execution_band.py", "scripts/audit_btceth_rl_execution_band.py")
SCOPE = "post_annual_execution_band_followup"


def utc(value):
    value = pd.Timestamp(value)
    if value.tzinfo is None:
        raise ValueError("Follow-up contracts require timezone-aware timestamps")
    return value.tz_convert("UTC")


def case_files(folder, *, baseline=False):
    keys = BASELINES if baseline else (*map(str, range(5)), "ensemble")
    files = CASE_FILES[:6] if baseline else CASE_FILES
    return {*(f"{folder}/{name}" for name in GROUP_FILES),
            *(f"{folder}/{cost}/{key}/{name}" for cost in COSTS for key in keys for name in files)}


def _passed(value, count, *, test_complete=True, checks_field="checks_total"):
    checks = value.get(checks_field)
    if (value.get("passed") is not True or value.get("status") != "passed"
            or value.get("test_complete") is not test_complete or value.get("run_count") != count
            or value.get("failure_count") != 0 or value.get("failures") != []
            or not isinstance(value.get("runs"), dict) or len(value["runs"]) != count
            or not isinstance(checks, int) or isinstance(checks, bool) or checks <= 0
            or any(not isinstance(value[name], int) or isinstance(value[name], bool) or value[name] <= 0
                   for name in ("checks", "checks_total") if name in value)
            or ("checks" in value and "checks_total" in value and value["checks_total"] != value["checks"])):
        raise ValueError("Complete independent accounting audit required; validation-only is insufficient")


def _pins(sources, roots, mapping, required):
    if not isinstance(mapping, dict) or set(mapping) != set(roots):
        raise ValueError("Audit must bind the five explicit evidence roots")
    for kind, root in roots.items():
        sources.pins(root, mapping[kind], required=required.get(kind, ()), label="Follow-up " + kind)


def _runs(audit, name, expected):
    runs = audit.get(name)
    if not isinstance(runs, dict) or set(runs) != expected:
        raise ValueError("Complete distinct follow-up/reference account identities required: " + name)
    for value in runs.values():
        if (value.get("reconstructed") is not True or value.get("hours") != 8760
                or not isinstance(value.get("fingerprints"), dict) or not value["fingerprints"]):
            raise ValueError("Every plotted account needs complete independent hourly reconstruction")
    return runs


def _summary(frame, family, *, baseline=False):
    common = {"run_id", "strategy", "scenario", "split", "initial_capital", "start", "end_exclusive", "net_return", "max_drawdown", "final_equity", "fees", "average_exposure"}
    if not common.issubset(frame) or frame.run_id.duplicated().any():
        raise ValueError("Follow-up summary fields or identities incomplete")
    expected = {f"test__{c}__{key}" for c in COSTS for key in BASELINES} if baseline else {
        f"test__{c}__{family}__tqc__balanced__{key}" for c in COSTS for key in (*map(str, range(5)), "ensemble")}
    if len(frame) != len(expected) or set(frame.run_id) != expected:
        raise ValueError("All fixed seeds, both costs and every baseline required")
    if (not frame.split.eq("test").all() or not frame.initial_capital.eq(10000.).all()
            or not pd.to_datetime(frame.start, utc=True).eq(utc(TEST[0])).all()
            or not pd.to_datetime(frame.end_exclusive, utc=True).eq(utc(TEST[1])).all()):
        raise ValueError("365-day UTC window and equal real 10,000 USDT accounts required")
    numeric = frame[["net_return", "max_drawdown", "final_equity", "fees", "average_exposure"]].to_numpy(dtype=float)
    if (not np.isfinite(numeric).all() or (frame.net_return <= -1).any() or (frame.final_equity <= 0).any()
            or (frame.max_drawdown > 0).any() or (frame.max_drawdown <= -1).any()
            or (frame.fees < 0).any() or not frame.average_exposure.between(0, 1).all()
            or not np.allclose(frame.final_equity / 10000 - 1, frame.net_return, atol=1e-11, rtol=1e-11)):
        raise ValueError("Financial summary is inconsistent")
    if not baseline:
        required = {"algorithm", "profile", "family", "seed", "primary_policy", "exposure_multiplier", "rebalance_band"}
        if not required.issubset(frame):
            raise ValueError("Actual model/ensemble and execution identities required")
        band = .05 if family == FAMILY else .025
        strategy = STRATEGY if family == FAMILY else "tqc_cap075"
        for key, value in dict(strategy=strategy, algorithm="tqc", profile="balanced", family=family, exposure_multiplier=.75, rebalance_band=band).items():
            if not frame[key].eq(value).all():
                raise ValueError("Only the same frozen five TQC models and declared execution variant allowed")
        if not frame.primary_policy.map(lambda x: isinstance(x, (bool, np.bool_))).all():
            raise ValueError("True actual-account ensemble flags required")
        for cost in COSTS:
            rows = frame.loc[frame.scenario.eq(cost)]
            ensemble, seeds = rows.loc[rows.primary_policy], rows.loc[~rows.primary_policy]
            if len(ensemble) != 1 or ensemble.seed.notna().any() or len(seeds) != 5 or set(seeds.seed) != set(range(5)):
                raise ValueError("No seed selection, averaged returns or missing costs")
    return expected


def legacy_execution_band(frame, reference, sources):
    """Recover the original display setting from SHA-bound case metadata.

    The old summary predates its band column; no current default is substituted
    and the original financial CSV is never rewritten.
    """
    values = []
    for row in frame.itertuples(index=False):
        if row.scenario not in COSTS:
            raise ValueError("Original cap costs must retain their audited identities")
        key = "ensemble" if bool(row.primary_policy) else str(int(row.seed))
        metadata = sources.json(safe(reference, f"test/balanced/{row.scenario}/{key}/metadata.json"))
        band = metadata.get("rebalance_band")
        if isinstance(band, bool) or not isinstance(band, (int, float)) or band != .025:
            raise ValueError("Original cap execution band must come from its audited metadata")
        values.append(float(band))
    if "rebalance_band" in frame and not np.array_equal(frame.rebalance_band.to_numpy(), values):
        raise ValueError("Original summary band contradicts audited case metadata")
    result = frame.copy()
    result["rebalance_band"] = values
    return result


def _history(history, months, summary):
    if not {"run_id", "timestamp", "equity", "net_return", "drawdown"}.issubset(history) or not {"run_id", "month", "return"}.issubset(months):
        raise ValueError("Complete daily curve and monthly evidence required")
    if set(history.run_id) != set(summary.run_id) or set(months.run_id) != set(summary.run_id):
        raise ValueError("Curve/monthly populations differ from audited accounts")
    dates = pd.date_range(*TEST, freq="D")
    for row in summary.itertuples(index=False):
        curve = history.loc[history.run_id.eq(row.run_id)]
        stamps = pd.DatetimeIndex(pd.to_datetime(curve.timestamp, utc=True))
        values = curve[["equity", "net_return", "drawdown"]].to_numpy(dtype=float)
        if not stamps.equals(dates) or not np.isfinite(values).all() or (curve.equity <= 0).any() or (curve.drawdown > 0).any():
            raise ValueError("Every daily boundary must be present, finite and ordered")
        if (not np.isclose(curve.equity.iloc[0], 10000., atol=1e-7, rtol=0)
                or not np.isclose(curve.equity.iloc[-1], row.final_equity, atol=1e-7, rtol=1e-12)
                or not np.allclose(curve.net_return, curve.equity / 10000 - 1, atol=1e-11, rtol=1e-11)
                or row.max_drawdown > float(curve.drawdown.min()) + 1e-10):
            raise ValueError("Curve equity/return/drawdown differs from the actual account")
        daily = pd.Series(curve.equity.iloc[1:].to_numpy(), index=dates[1:] - pd.Timedelta(days=1))
        ends = daily.resample("ME").last()
        expected = ends.pct_change(); expected.iloc[0] = ends.iloc[0] / 10000 - 1
        monthly = months.loc[months.run_id.eq(row.run_id)]
        if (monthly.month.tolist() != expected.index.strftime("%Y-%m").tolist()
                or not np.isfinite(monthly["return"]).all()
                or not np.allclose(monthly["return"], expected, atol=1e-10, rtol=1e-10)
                or not np.isclose(np.prod(1 + monthly["return"]) - 1, row.net_return, atol=1e-10, rtol=1e-10)):
            raise ValueError("Each monthly return must reconcile to its actual daily curve and final NAV")


def _targets(summary, baselines, old):
    base = summary.loc[summary.scenario.eq("base") & summary.primary_policy].iloc[0]
    stress = summary.loc[summary.scenario.eq("stress") & summary.primary_policy].iloc[0]
    seeds = summary.loc[summary.scenario.eq("base") & ~summary.primary_policy].sort_values("seed")
    equal = baselines.loc[baselines.run_id.eq("test__base__equal_hold")].iloc[0]
    positive = int(seeds.net_return.gt(0).sum())
    gates = dict(base_return_at_least_5pct=bool(base.net_return >= .05),
        base_max_drawdown_at_most_20pct=bool(abs(base.max_drawdown) <= .20),
        stress_return_nonnegative=bool(stress.net_return >= 0),
        at_least_three_positive_training_seeds=positive >= 3,
        beats_equal_hold_return_and_drawdown=bool(base.net_return > equal.net_return and base.max_drawdown > equal.max_drawdown))
    trends = {key: bool(base.net_return > baselines.loc[baselines.run_id.eq("test__base__" + key)].iloc[0].net_return
                       and base.max_drawdown > baselines.loc[baselines.run_id.eq("test__base__" + key)].iloc[0].max_drawdown)
              for key in ("sma200_trend", "momentum180")}
    return dict(candidate=dict(family=FAMILY, algorithm="tqc", profile="balanced"), gates=gates,
        all_five_diagnostic_targets_met=all(gates.values()), goal_performance_achieved=all(gates.values()),
        positive_base_training_seeds=positive, base_seed_returns=[dict(seed=int(r.seed), net_return=float(r.net_return)) for r in seeds.itertuples()],
        base_net_return=float(base.net_return), base_max_drawdown=float(base.max_drawdown), stress_net_return=float(stress.net_return),
        stress_max_drawdown=float(stress.max_drawdown), base_fees_usdt=float(base.fees), stress_fees_usdt=float(stress.fees),
        beats_simple_trend_controls=trends, stronger_control_comparison_met=all(trends.values()),
        old_goal_performance_achieved=old["goal_performance_achieved"], old_positive_base_training_seeds=old["metrics"]["positive_base_training_seeds"],
        accounting_passage_is_not_goal_success=True, fresh_holdout_success=False, **{flag: True for flag in FLAGS})


def load_execution_band(annual_root, validation_root, main_root, reference_root, *, audit_path=None, source_root=ROOT):
    """Read complete evidence only; return independent queries, never mutate a snapshot.

    Annual runner contract is intentionally strict. Any unfinished/preparatory
    schema is rejected until the separately authorized runner/audit is final.
    """
    roots = {key: Path(value).resolve() for key, value in (("annual", annual_root), ("validation", validation_root), ("main", main_root), ("reference", reference_root), ("source", source_root))}
    outputs = [roots[key] for key in ("annual", "validation", "main", "reference")]
    if any(a == b or a.is_relative_to(b) or b.is_relative_to(a) for i, a in enumerate(outputs) for b in outputs[i + 1:]):
        raise ValueError("Follow-up, validation, main and original cap require separate evidence trees")
    annual, validation, main, reference = (roots[key] for key in ("annual", "validation", "main", "reference"))
    sources = Sources()
    selected_audit = Path(audit_path).resolve() if audit_path else annual / "independent_audit.json"
    selected_audit.relative_to(annual)
    required = dict(annual=set(CONTRACTS) | case_files("test/balanced"),
        validation={"protocol.json", "protocol_receipt.json", "selection.json", "selection_receipt.json", "independent_audit.validation.json", "summary.csv"},
        main={"frozen_spec.json", "independent_audit.json"} | case_files("evaluation/test/baselines", baseline=True),
        reference={"protocol.json", "independent_audit.json", "goal_candidate_evidence.json", "validation/balanced/summary.csv"} | case_files("test/balanced"),
        source=set(REQUIRED_SOURCES))
    audits = []
    for path in dict.fromkeys((selected_audit, annual / "independent_audit.json")):
        audit = sources.json(path); _passed(audit, 12)
        if (audit.get("repeat_count") != 12 or audit.get("baseline_run_count") != 14
                or audit.get("declared_research_targets") != TARGETS):
            raise ValueError("Full twelve replays and fourteen independently reconstructed baselines required")
        _pins(sources, roots, audit.get("input_sha256"), required)
        audits.append(audit)
    audit = audits[0]
    p, receipt, marker, completion, process = (sources.json(annual / name) for name in CONTRACTS[:5])
    if sources.json(annual / "test_process.json") != process:
        raise ValueError("Annual process aliases must preserve the same actual terminal evidence")
    selection = sources.json(validation / "selection.json")
    seal = sources.json(validation / "selection_receipt.json")
    validation_audit = sources.json(validation / "independent_audit.validation.json")
    _passed(validation_audit, 12, test_complete=False)
    if (validation_audit.get("repeat_count") != 12 or validation_audit.get("comparator_run_count") != 12
            or seal.get("selection_file") != "selection.json" or seal.get("selection_sha256") != sha(validation / "selection.json")
            or selection.get("audit_sha256") != sha(validation / "independent_audit.validation.json")
            or seal.get("protocol_sha256") != selection.get("protocol_sha256") or selection.get("protocol_sha256") != sha(validation / "protocol.json")
            or selection.get("seed_selection") is not False or selection.get("selection_uses_test") is not False):
        raise ValueError("Actual full validation audit and sealed two-candidate decision required")
    postselection_file = p.get("validation_audit_file")
    if not isinstance(postselection_file, str) or postselection_file == "independent_audit.validation.json":
        raise ValueError("A distinct completed post-selection audit must follow the original selection audit")
    postselection_path = safe(validation, postselection_file)
    if any(postselection_file not in value["input_sha256"]["validation"] for value in audits):
        raise ValueError("Every annual audit must bind the actual post-selection validation audit identity")
    postselection_audit = sources.json(postselection_path)
    _passed(postselection_audit, 12, test_complete=False)
    if (postselection_audit.get("selection_complete") is not True or postselection_audit.get("validation_complete") is not True
            or postselection_audit.get("resource_adapter_disclosed") is not True
            or postselection_audit.get("repeat_count") != 12 or postselection_audit.get("comparator_run_count") != 12):
        raise ValueError("Full post-selection financial and resource-adapter audit required before annual preparation")
    pool = selection.get("candidates", [])
    if (len(pool) != 2 or {r.get("family") for r in pool} != {"cap075", FAMILY}
            or any(not isinstance(r.get("score"), (int, float)) or not np.isfinite(r["score"]) for r in pool)
            or pool != sorted(pool, key=lambda r: (-r["score"], 0 if r["family"] == "cap075" else 1))
            or selection.get("candidate") != pool[0] or pool[0].get("family") != FAMILY):
        raise ValueError("Only a validation-winning 5pp candidate may receive this separate annual replay")
    validation_roots = {"extension": validation, **{key: roots[key] for key in ("main", "reference", "source")}}
    if selection.get("validation_input_sha256") != validation_audit.get("input_sha256"):
        raise ValueError("Selection must retain the exact prior validation audit inputs")
    _pins(sources, validation_roots, selection["validation_input_sha256"], {})
    _pins(sources, validation_roots, postselection_audit.get("input_sha256"), {"extension": {
        "selection.json", "selection_receipt.json", "independent_audit.validation.json"}})
    for item in pool:
        band = .05 if item["family"] == FAMILY else .025
        if item.get("algorithm") != "tqc" or item.get("profile") != "balanced" or item.get("rebalance_band") != band:
            raise ValueError("SAC or another configuration cannot substitute for either fixed validation candidate")
        table = sources.csv(validation / "summary.csv" if band == .05 else reference / "validation/balanced/summary.csv")
        for cost in COSTS:
            match = table.loc[table.run_id.eq(item[cost]["run_id"])]
            if len(match) != 1 or match.iloc[0].split != "validation" or not bool(match.iloc[0].primary_policy) or match.iloc[0].scenario != cost:
                raise ValueError("Every selection metric must refer to its actual full validation ensemble")
            for metric in ("net_return", "max_drawdown", "fees", "final_equity"):
                same_metrics(item[cost][metric], float(match.iloc[0][metric]), "Prior validation:" + metric)
    fixed = dict(algorithm="tqc", profile="balanced", capital=10000., exposure_multiplier=.75, rebalance_band=.05,
        costs=COSTS, seeds=list(range(5)), new_training=0, seed_selection=False,
        selection_uses_test=False, annual_evaluation_enabled=True, **{flag: True for flag in FLAGS})
    if any(p.get(k) != value for k, value in fixed.items()) or any(selection.get(flag) is not True for flag in FLAGS):
        raise ValueError("Only the fixed, openly post-annual execution-band follow-up may be reported")
    pins = dict(validation_protocol_sha256=sha(validation / "protocol.json"), validation_selection_sha256=sha(validation / "selection.json"),
        validation_selection_receipt_sha256=sha(validation / "selection_receipt.json"), validation_audit_sha256=sha(postselection_path),
        original_frozen_spec_sha256=sha(main / "frozen_spec.json"))
    if any(p.get(k) != value for k, value in pins.items()):
        raise ValueError("Annual declaration must bind the actual prior validation and original source contract")
    spec = sources.json(main / "frozen_spec.json")
    if p.get("dataset_sha256") != spec.get("dataset_sha256") or any(value.get("dataset_sha256") != spec.get("dataset_sha256") for value in audits):
        raise ValueError("Annual audit and declaration must use the original frozen dataset identity")
    sources.pins(annual / "sources", p.get("source_sha256"), required=REQUIRED_SOURCES, label="Annual source snapshots")
    sources.pins(roots["source"], p["source_sha256"], required=REQUIRED_SOURCES, label="Actual annual sources")
    models = p.get("models", [])
    old_models = sources.json(reference / "protocol.json").get("models", [])
    if (len(models) != 5 or [m.get("seed") for m in models] != list(range(5))
            or models != [model for model in old_models if model.get("profile") == "balanced"]):
        raise ValueError("Exactly the original five training checkpoints required")
    for model in models:
        record = sources.json(safe(main, model["record_file"]))
        for field in ("record", "model"):
            name = model[field + "_file"]
            if (name not in audit["input_sha256"]["main"] or sha(sources.file(safe(main, name))) != model[field + "_sha256"]):
                raise ValueError("Original checkpoint/record differs from annual input identity")
        if ((record.get("algorithm"), record.get("profile"), record.get("seed")) != ("tqc", "balanced", model["seed"])
                or record.get("parameter_sha256") != model.get("parameter_sha256") or record.get("model_sha256") != model["model_sha256"]):
            raise ValueError("No new training, substituted seed or parameter identity allowed")
    protocol_sha = sha(annual / "protocol.json")
    if (receipt.get("protocol_file") != "protocol.json" or receipt.get("protocol_sha256") != protocol_sha
            or marker.get("protocol_sha256") != protocol_sha or marker.get("validation_selection_sha256") != pins["validation_selection_sha256"]
            or completion.get("protocol_sha256") != protocol_sha or completion.get("case_count") != 12 or completion.get("repeat_count") != 12
            or process.get("phase") != "test" or process.get("status") != "completed" or process.get("exit_code") != 0 or process.get("protocol_sha256") != protocol_sha):
        raise ValueError("Distinct annual dispatch, terminal process and completion must agree")
    if not (utc(validation_audit["audited_at"]) <= utc(selection["selected_at"]) <= utc(seal["sealed_at"])
            <= utc(postselection_audit["audited_at"]) <= utc(p["prepared_at"]) <= utc(receipt["sealed_at"])
            <= utc(marker["at"]) <= utc(process["completed_at"]) <= utc(audit["audited_at"])):
        raise ValueError("Annual must follow the sealed validation choice; accounting cannot masquerade as pre-test selection")
    log = sources.file(safe(annual, process["log_relative_path"]))
    if process.get("log_sha256") != sha(log) or process["log_relative_path"] not in audit["input_sha256"]["annual"]:
        raise ValueError("Actual terminal log identity must be audited")
    main_audit, old_audit = sources.json(main / "independent_audit.json"), sources.json(reference / "independent_audit.json")
    _passed(main_audit, 172)
    _passed(old_audit, 36, checks_field="checks")
    # The new annual audit binds the retained original audit and the reused
    # baseline accounts. Follow that original audit to its completion evidence,
    # rather than claiming the new audit read an unrelated global completion.
    sources.pins(main, main_audit.get("input_sha256"),
        required={"frozen_spec.json", "test_completion.json"} | case_files("evaluation/test/baselines", baseline=True),
        label="Retained original172 audit")
    old = sources.json(reference / "goal_candidate_evidence.json")
    if (old.get("audit_sha256") != sha(reference / "independent_audit.json") or old.get("goal_performance_achieved") is not False
            or old.get("metrics", {}).get("positive_base_training_seeds") != 2):
        raise ValueError("Preserve original failed goal and its actual two-positive-seed evidence")
    summary, history, months = (sources.csv(annual / name) for name in GROUP_FILES[:3])
    old_tables = [sources.csv(reference / "test/balanced" / name) for name in GROUP_FILES[:3]]
    old_tables[0] = legacy_execution_band(old_tables[0], reference, sources)
    baseline_tables = [sources.csv(main / "evaluation/test/baselines" / name) for name in GROUP_FILES[:3]]
    expected = _summary(summary, FAMILY)
    old_ids = _summary(old_tables[0], "cap075")
    baseline_ids = _summary(baseline_tables[0], "", baseline=True)
    for value in audits:
        _runs(value, "runs", expected); _runs(value, "baseline_runs", baseline_ids)
        if value.get("reference_run_count") != 12:
            raise ValueError("Twelve preserved original-cap reference accounts required")
        refs = _runs(value, "reference_runs", old_ids)
        for key in old_ids:
            if refs[key]["fingerprints"] != old_audit.get("runs", {}).get(key, {}).get("fingerprints"):
                raise ValueError("Original cap reference cannot be replaced by new annual results")
        for key in baseline_ids:
            if value["baseline_runs"][key]["fingerprints"] != main_audit.get("runs", {}).get(key, {}).get("fingerprints"):
                raise ValueError("Baseline reference differs from retained original172 audit")
    for tables in ((summary, history, months), old_tables, baseline_tables):
        _history(tables[1], tables[2], tables[0])
    calculated = _targets(summary, baseline_tables[0], old)
    for value in audits:
        supplied = value.get("research_targets", {})
        for key, expected_value in calculated.items():
            same_metrics(supplied.get(key), expected_value, "Follow-up target:" + key)
    sources.file(Path(__file__))
    source = dict(label="已观察年度上的独立5pp执行门槛跟进", classification="modeled", timezone="UTC",
        period=dict(start=TEST[0], endExclusive=TEST[1]), files=list(sources.hashes), fileHashes=dict(sources.hashes),
        caveats=["本轮在观察首轮年度结果后设计，年度已见，不是fresh holdout。", "首轮失败目标及主方案原样保留；SAC不能替代新模型达标。", "通过本轮五项目标不代表优于所有模型；首轮SAC两成本结果应从原已审计报告另作算法对照，不参加这两候选选择。", "五个种子重复同一段市场历史；实际主组合平均目标权重后在一个10,000 USDT账本成交。", "回撤使用完整小时权益，曲线为日边界展示；与早期下一分钟执行口径不同。"])
    summary_parts, history_parts, monthly_parts = [], [], []
    for role, tables in (("followup", (summary, history, months)), ("original_cap_reference", old_tables), ("baseline_reference", baseline_tables)):
        for source_frame, destination in zip(tables, (summary_parts, history_parts, monthly_parts)):
            frame = source_frame.copy(); frame["scope"] = SCOPE; frame["followup_role"] = role
            destination.append(frame)
    target_rows = [dict(gate=key, met=value, category="已观察年度的诊断目标") for key, value in calculated["gates"].items()]
    target_rows.extend(dict(gate=key, met=value, category="简单趋势对照") for key, value in calculated["beats_simple_trend_controls"].items())
    audit_rows = [dict(role="本机明确审计" if i == 0 else "保留远端审计", file=str(path.as_posix()), sha256=sha(path),
        passed=True, new_runs=12, old_cap_reference_runs=12, baseline_runs=14, repeat_count=12, fresh_holdout_success=False)
        for i, path in enumerate(dict.fromkeys((selected_audit, annual / "independent_audit.json")))]
    tables = dict(execution_band_summary=pd.concat(summary_parts, ignore_index=True), execution_band_equity=pd.concat(history_parts, ignore_index=True),
        execution_band_monthly=pd.concat(monthly_parts, ignore_index=True), execution_band_targets=pd.DataFrame(target_rows),
        execution_band_validation_selection=pd.DataFrame(pool), execution_band_audit=pd.DataFrame(audit_rows))
    queries = {key: dict(rows=frame.astype(object).where(pd.notna(frame), None).to_dict("records"), source=copy.deepcopy(source),
        methods=[dict(language="text", code="Independent follow-up only. Full passed annual audit and SHA-bound prior validation required; twelve new funded accounts plus twelve preserved cap references and fourteen original baselines. Never enter or replace the original208 ranking or original leader/goal.")]) for key, frame in tables.items()}
    sources.unchanged()
    return dict(queries=queries, targets=calculated, selection=selection, audit=audit, source=source,
        original_goal_preserved=True, fresh_holdout_success=False, main_ranking_unchanged=True,
        external_input_sha256=dict(sources.hashes), query_run_count=38)

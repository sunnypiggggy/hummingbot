"""Synthetic reader fixtures only; no annual prices, policies or HTML outputs."""
from copy import deepcopy
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from scripts import btceth_rl_report_execution_band as report


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def mapping(root):
    return {p.relative_to(root).as_posix(): report.sha(p) for p in root.rglob("*") if p.is_file()}


def summary(family, *, baseline=False, validation=False):
    rows = []
    for cost in report.COSTS:
        for key in report.BASELINES if baseline else (*map(str, range(5)), "ensemble"):
            primary = key == "ensemble"
            ret = .08 if cost == "base" else .03
            if not primary and not baseline:
                ret = [-.02, .03, .09, .12, .05][int(key)] - (.02 if cost == "stress" else 0)
            if family == "cap075":
                ret = (.07 if cost == "base" else -.01) if primary else [-.04, .05, .06, -.05, -.02][int(key)]
            if validation and primary:
                ret = (.20 if cost == "base" else .12) if family == report.FAMILY else (.07 if cost == "base" else .01)
            if baseline:
                ret = 0. if key == "cash" else .02 if key in ("sma200_trend", "momentum180") else -.10
            start, end = ("2025-01-01T00:00:00Z", report.TEST[0]) if validation else report.TEST
            split = "validation" if validation else "test"
            name = str(key) if baseline else f"{family}__tqc__balanced__{key}"
            strategy = str(key) if baseline else report.STRATEGY if family == report.FAMILY else "tqc_cap075"
            row = dict(run_id=f"{split}__{cost}__{name}", strategy=strategy,
                algorithm="baseline" if baseline else "tqc", profile="balanced", family=family, scenario=cost, split=split,
                seed=None if primary or baseline else int(key), primary_policy=primary, initial_capital=10000.,
                start=start, end_exclusive=end, net_return=ret, max_drawdown=0. if baseline and key == "cash" else -.40 if baseline else -.20,
                final_equity=10000 * (1 + ret), fees=10. if cost == "base" else 20., average_exposure=.30,
                exposure_multiplier=.75, rebalance_band=.05 if family == report.FAMILY else .025)
            rows.append(row)
    return pd.DataFrame(rows)


def group(root, folder, frame, *, baseline=False):
    directory = root / folder
    directory.mkdir(parents=True, exist_ok=True)
    histories, months = [], []
    for row in frame.itertuples():
        dates = pd.date_range(row.start, row.end_exclusive, freq="D")
        equity = np.linspace(10000., row.final_equity, len(dates))
        history = pd.DataFrame(dict(run_id=row.run_id, timestamp=dates.astype(str), equity=equity,
            net_return=equity / 10000 - 1, drawdown=equity / np.maximum.accumulate(equity) - 1))
        histories.append(history)
        daily = pd.Series(equity[1:], index=dates[1:] - pd.Timedelta(days=1))
        ends = daily.resample("ME").last()
        returns = ends.pct_change(); returns.iloc[0] = ends.iloc[0] / 10000 - 1
        months.append(pd.DataFrame(dict(run_id=row.run_id, month=ends.index.strftime("%Y-%m"), **{"return": returns.to_numpy()})))
    saved = frame.drop(columns="rebalance_band") if not baseline and frame.family.eq("cap075").all() else frame
    tables = saved, pd.concat(histories, ignore_index=True), pd.concat(months, ignore_index=True)
    for name, table in zip(report.GROUP_FILES, tables):
        table.to_csv(directory / name, index=False)
    dump(directory / "completion.json", dict(case_count=len(frame)))
    for name in report.case_files(folder, baseline=baseline):
        path = root / name
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(("fixture-only:" + name).encode())
    if not baseline:
        for row in frame.itertuples(index=False):
            key = "ensemble" if row.primary_policy else str(int(row.seed))
            dump(directory / row.scenario / key / "metadata.json",
                 dict(synthetic_fixture_only=True, rebalance_band=float(row.rebalance_band)))
    return tables


def proof(frame):
    return {row.run_id: dict(reconstructed=True, hours=8760, repeat_exact=True,
        fingerprints={"equity": str(i).zfill(64)}) for i, row in enumerate(frame.itertuples())}


def passed(count, runs, *, test_complete=True):
    result = dict(passed=True, status="passed", test_complete=test_complete, run_count=count,
        repeat_count=count, checks_total=100, failure_count=0, failures=[], audited_at="2026-10-30T01:05:00Z", runs=dict(runs))
    while len(result["runs"]) < count:
        result["runs"]["other_fixture_" + str(len(result["runs"]))] = dict(reconstructed=True)
    return result


@pytest.fixture
def evidence(tmp_path):
    roots = {key: tmp_path / key for key in ("annual", "validation", "main", "reference", "source")}
    for root in roots.values():
        root.mkdir()
    annual, validation, main, reference, source = (roots[key] for key in roots)
    source_hashes = {}
    for name in report.REQUIRED_SOURCES:
        path = source / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_text("# fixture only\n")
        source_hashes[name] = report.sha(path)
        target = annual / "sources" / name; target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(path.read_bytes())
    dump(main / "frozen_spec.json", dict(dataset_sha256="d" * 64))
    models = []
    for seed in range(5):
        record_name, model_name = f"training/tqc__balanced__seed{seed}/record.json", f"training/tqc__balanced__seed{seed}/model.zip"
        model_path = main / model_name; model_path.parent.mkdir(parents=True, exist_ok=True); model_path.write_bytes(f"fixture-model-{seed}".encode())
        parameter_sha = str(seed) * 64
        dump(main / record_name, dict(algorithm="tqc", profile="balanced", seed=seed, parameter_sha256=parameter_sha, model_sha256=report.sha(model_path)))
        models.append(dict(profile="balanced", seed=seed, record_file=record_name, model_file=model_name,
            record_sha256=report.sha(main / record_name), model_sha256=report.sha(model_path), parameter_sha256=parameter_sha))
    dump(reference / "protocol.json", dict(models=models))
    new = group(annual, "test/balanced", summary(report.FAMILY))
    for name, table in zip(report.GROUP_FILES, new):
        table.to_csv(annual / name, index=False)
    old = group(reference, "test/balanced", summary("cap075"))
    baselines = group(main, "evaluation/test/baselines", summary("", baseline=True), baseline=True)
    dump(main / "test_completion.json", dict(case_count=62))
    original_audit = passed(172, proof(baselines[0]))
    original_audit["input_sha256"] = mapping(main)
    dump(main / "independent_audit.json", original_audit)
    old_audit = passed(36, proof(old[0]))
    old_audit["checks"] = old_audit.pop("checks_total")
    dump(reference / "independent_audit.json", old_audit)
    old_goal = dict(goal_performance_achieved=False, metrics=dict(positive_base_training_seeds=2), audit_sha256=report.sha(reference / "independent_audit.json"))
    dump(reference / "goal_candidate_evidence.json", old_goal)
    val_new, val_old = summary(report.FAMILY, validation=True), summary("cap075", validation=True)
    val_new.to_csv(validation / "summary.csv", index=False)
    (reference / "validation/balanced").mkdir(parents=True)
    val_old.to_csv(reference / "validation/balanced/summary.csv", index=False)
    dump(validation / "protocol.json", dict(annual_evaluation_enabled=False))
    dump(validation / "protocol_receipt.json", dict(protocol_sha256=report.sha(validation / "protocol.json")))
    main_reference_map = lambda: {name: digest for name, digest in mapping(main).items() if name != "test_completion.json"}
    val_maps = dict(extension=mapping(validation), main=main_reference_map(), reference=mapping(reference), source=mapping(source))
    val_audit = passed(12, proof(val_new), test_complete=False)
    val_audit.update(audited_at="2026-10-30T00:10:00Z", comparator_run_count=12, input_sha256=val_maps)
    dump(validation / "independent_audit.validation.json", val_audit)
    pool = []
    for family, frame, band in ((report.FAMILY, val_new, .05), ("cap075", val_old, .025)):
        costs = {cost: frame.loc[frame.scenario.eq(cost) & frame.primary_policy].iloc[0] for cost in report.COSTS}
        metrics = {cost: dict(run_id=row.run_id, **{key: float(row[key]) for key in ("net_return", "max_drawdown", "final_equity", "fees")}) for cost, row in costs.items()}
        score = sum(math.log1p(row.net_return) - .5 * abs(row.max_drawdown) - 2 * max(abs(row.max_drawdown) - .2, 0) for row in costs.values()) / 2
        pool.append(dict(name="band05" if family == report.FAMILY else "reference_band025", family=family, algorithm="tqc", profile="balanced", rebalance_band=band, score=score, **metrics))
    pool.sort(key=lambda row: (-row["score"], 0 if row["family"] == "cap075" else 1))
    selection = dict(selected_at="2026-10-30T00:20:00Z", protocol_sha256=report.sha(validation / "protocol.json"),
        audit_sha256=report.sha(validation / "independent_audit.validation.json"), candidates=pool, candidate=pool[0],
        validation_input_sha256=val_maps, selection_uses_test=False, seed_selection=False, **{flag: True for flag in report.FLAGS})
    dump(validation / "selection.json", selection)
    dump(validation / "selection_receipt.json", dict(selection_file="selection.json", selection_sha256=report.sha(validation / "selection.json"),
        protocol_sha256=report.sha(validation / "protocol.json"), sealed_at="2026-10-30T00:21:00Z"))
    postselection_audit = deepcopy(val_audit)
    postselection_audit.update(audited_at="2026-10-30T00:25:00Z", selection_complete=True,
        validation_complete=True, resource_adapter_disclosed=True,
        input_sha256=dict(extension=mapping(validation), main=main_reference_map(), reference=mapping(reference), source=mapping(source)))
    dump(validation / "independent_audit.validation.postselection.json", postselection_audit)
    protocol = dict(prepared_at="2026-10-30T00:30:00Z", algorithm="tqc", profile="balanced", capital=10000., exposure_multiplier=.75,
        rebalance_band=.05, costs=report.COSTS, seeds=list(range(5)), new_training=0,
        seed_selection=False, selection_uses_test=False, annual_evaluation_enabled=True, models=models, dataset_sha256="d" * 64, source_sha256=source_hashes,
        validation_protocol_sha256=report.sha(validation / "protocol.json"), validation_selection_sha256=report.sha(validation / "selection.json"),
        validation_selection_receipt_sha256=report.sha(validation / "selection_receipt.json"),
        validation_audit_file="independent_audit.validation.postselection.json", validation_audit_sha256=report.sha(validation / "independent_audit.validation.postselection.json"),
        original_frozen_spec_sha256=report.sha(main / "frozen_spec.json"), **{flag: True for flag in report.FLAGS})
    dump(annual / "protocol.json", protocol)
    protocol_sha = report.sha(annual / "protocol.json")
    dump(annual / "protocol_receipt.json", dict(protocol_file="protocol.json", protocol_sha256=protocol_sha, sealed_at="2026-10-30T00:31:00Z"))
    dump(annual / "test_started.json", dict(at="2026-10-30T00:40:00Z", protocol_sha256=protocol_sha, validation_selection_sha256=report.sha(validation / "selection.json")))
    dump(annual / "test_completion.json", dict(protocol_sha256=protocol_sha, case_count=12, repeat_count=12))
    (annual / "test.log").write_text("fixture terminal success\n")
    dump(annual / "test_process.json", dict(phase="test", status="completed", exit_code=0, protocol_sha256=protocol_sha,
        completed_at="2026-10-30T01:00:00Z", log_relative_path="test.log", log_sha256=report.sha(annual / "test.log")))
    dump(annual / "runner_process.json", read(annual / "test_process.json"))
    audit = passed(12, proof(new[0]))
    audit.update(baseline_run_count=14, reference_run_count=12, baseline_runs=proof(baselines[0]), reference_runs=proof(old[0]),
        dataset_sha256="d" * 64, declared_research_targets=report.TARGETS, research_targets=report._targets(new[0], baselines[0], old_goal))
    def reseal():
        audit["input_sha256"] = {key: {name: value for name, value in mapping(root).items() if not (key == "annual" and name.startswith("independent_audit"))} for key, root in roots.items()}
        audit["input_sha256"]["main"] = main_reference_map()
        dump(annual / "independent_audit.json", audit)
    reseal()
    def load(**kwargs):
        return report.load_execution_band(annual, validation, main, reference, source_root=source, **kwargs)
    return SimpleNamespace(roots=roots, annual=annual, validation=validation, main=main, reference=reference, source=source, load=load, reseal=reseal, audit=audit)


def test_complete_followup_is_separate_preserves_old_goal_and_has_inspectable_sources(evidence):
    original = mapping(evidence.main), mapping(evidence.reference)
    payload = evidence.load()
    assert set(payload["queries"]) == {"execution_band_" + name for name in ("summary", "equity", "monthly", "targets", "validation_selection", "audit")}
    rows = payload["queries"]["execution_band_summary"]["rows"]
    assert len(rows) == len({row["run_id"] for row in rows}) == 38
    assert {row["followup_role"] for row in rows} == {"followup", "original_cap_reference", "baseline_reference"}
    assert sum(row["followup_role"] == "followup" and row["primary_policy"] for row in rows) == 2
    assert payload["targets"]["goal_performance_achieved"] is True
    assert payload["targets"]["old_goal_performance_achieved"] is False
    assert payload["targets"]["old_positive_base_training_seeds"] == 2
    assert payload["fresh_holdout_success"] is False and payload["main_ranking_unchanged"] is True
    assert all(Path(path).is_absolute() for path in payload["external_input_sha256"])
    assert original == (mapping(evidence.main), mapping(evidence.reference))
    assert not list(evidence.annual.rglob("*.html"))


def test_validation_only_or_missing_annual_cannot_create_fake_queries(evidence):
    evidence.audit.update(test_complete=False)
    evidence.reseal()
    with pytest.raises(ValueError, match="validation-only"):
        evidence.load()
    (evidence.annual / "independent_audit.json").unlink()
    with pytest.raises(FileNotFoundError):
        evidence.load()


def test_exact_input_sha_required_before_financial_tables_are_used(evidence):
    (evidence.annual / "summary.csv").write_text("changed\n")
    with pytest.raises(ValueError, match="SHA changed"):
        evidence.load()


def test_main_completion_is_bound_through_original_audit_not_falsely_claimed_direct(evidence):
    assert "test_completion.json" not in evidence.audit["input_sha256"]["main"]
    payload = evidence.load()
    completion = evidence.main / "test_completion.json"
    assert completion.as_posix() in payload["external_input_sha256"]
    dump(completion, {"case_count": 0})
    with pytest.raises(ValueError, match="SHA changed"):
        evidence.load()


def test_legacy_cap_count_alias_is_explicit_positive_and_unambiguous():
    value = passed(36, {})
    value["checks"] = value.pop("checks_total")
    report._passed(value, 36, checks_field="checks")
    with pytest.raises(ValueError):
        report._passed(value, 36)
    value["checks_total"] = value["checks"] + 1
    with pytest.raises(ValueError):
        report._passed(value, 36, checks_field="checks")
    canonical = passed(12, {})
    canonical["checks"] = canonical["checks_total"] + 1
    with pytest.raises(ValueError):
        report._passed(canonical, 12)
    canonical["checks"] = True
    with pytest.raises(ValueError):
        report._passed(canonical, 12)


@pytest.mark.parametrize("mutation", ["dates", "capital", "seed", "family", "cost", "algorithm", "strategy"])
def test_bound_but_wrong_summary_cannot_pass_on_audit_label_alone(evidence, mutation):
    path = evidence.annual / "summary.csv"
    frame = pd.read_csv(path)
    column, value = dict(dates=("end_exclusive", "2026-10-22T00:00:00Z"), capital=("initial_capital", 5000.),
        seed=("seed", 99), family=("family", "raw"), cost=("scenario", "other"), algorithm=("algorithm", "sac"), strategy=("strategy", "sac"))[mutation]
    frame.loc[0, column] = value; frame.to_csv(path, index=False); evidence.reseal()
    with pytest.raises(ValueError):
        evidence.load()


@pytest.mark.parametrize("filename,column", [("daily_equity.csv", "equity"), ("monthly_returns.csv", "return")])
def test_resealed_curve_or_monthly_mismatch_rejected(evidence, filename, column):
    path = evidence.annual / filename
    frame = pd.read_csv(path); frame.loc[frame.index[-1], column] += .01
    frame.to_csv(path, index=False); evidence.reseal()
    with pytest.raises(ValueError, match="Curve|monthly"):
        evidence.load()


def test_audit_performance_gate_cannot_be_used_as_only_success_label(evidence):
    evidence.audit["research_targets"]["gates"]["at_least_three_positive_training_seeds"] = False
    evidence.reseal()
    with pytest.raises(ValueError, match="target"):
        evidence.load()


def test_declared_thresholds_cannot_be_silently_lowered(evidence):
    evidence.audit["declared_research_targets"] = {**report.TARGETS, "primary_base_net_return_at_least": 0.}
    evidence.reseal()
    with pytest.raises(ValueError):
        evidence.load()


def test_sac_never_substitutes_for_frozen_candidate_or_seed_stability():
    frame, controls = summary(report.FAMILY), summary("", baseline=True)
    frame.loc[~frame.primary_policy, ["net_return", "final_equity"]] = [-.01, 9900.]
    result = report._targets(frame, controls, dict(goal_performance_achieved=False, metrics=dict(positive_base_training_seeds=2)))
    assert result["candidate"]["algorithm"] == "tqc"
    assert result["gates"]["base_return_at_least_5pct"] is True
    assert result["positive_base_training_seeds"] == 0
    assert result["goal_performance_achieved"] is False


def test_explicit_local_audit_retains_remote_identity_and_both_source_hashes(evidence):
    local = evidence.annual / "independent_audit.local.json"
    value = deepcopy(evidence.audit); value["checks_total"] += 1; dump(local, value)
    payload = evidence.load(audit_path=local)
    rows = payload["queries"]["execution_band_audit"]["rows"]
    assert len(rows) == 2 and rows[0]["file"] == local.as_posix()
    assert rows[0]["sha256"] != rows[1]["sha256"]
    assert all(row["file"] in payload["external_input_sha256"] for row in rows)


def test_missing_baseline_or_false_hourly_proof_rejected(evidence):
    key = next(iter(evidence.audit["baseline_runs"]))
    evidence.audit["baseline_runs"][key]["hours"] = 8759; evidence.reseal()
    with pytest.raises(ValueError, match="hourly"):
        evidence.load()


def test_selection_must_precede_annual_and_cannot_claim_fresh_holdout(evidence):
    marker = read(evidence.annual / "test_started.json"); marker["at"] = "2026-10-30T00:15:00Z"
    dump(evidence.annual / "test_started.json", marker); evidence.reseal()
    with pytest.raises(ValueError, match="Annual must follow"):
        evidence.load()


def test_source_path_escape_rejected(evidence):
    evidence.audit["input_sha256"]["source"]["../escape.py"] = "a" * 64
    dump(evidence.annual / "independent_audit.json", evidence.audit)
    with pytest.raises(ValueError, match="relative"):
        evidence.load()


def repin_protocol(evidence):
    path = evidence.annual / "protocol.json"
    value = read(path)
    value["validation_audit_sha256"] = report.sha(evidence.validation / value["validation_audit_file"])
    dump(path, value)
    digest = report.sha(path)
    for name in ("protocol_receipt.json", "test_started.json", "test_completion.json", "test_process.json", "runner_process.json"):
        item = read(evidence.annual / name); item["protocol_sha256"] = digest; dump(evidence.annual / name, item)
    evidence.reseal()


def test_postselection_audit_cannot_replace_the_original_selection_audit(evidence):
    path = evidence.validation / "selection.json"
    value = read(path); value["audit_sha256"] = report.sha(evidence.validation / "independent_audit.validation.postselection.json")
    dump(path, value)
    seal = read(evidence.validation / "selection_receipt.json"); seal["selection_sha256"] = report.sha(path)
    dump(evidence.validation / "selection_receipt.json", seal); evidence.reseal()
    with pytest.raises(ValueError, match="Actual full validation audit"):
        evidence.load()


def test_initial_audit_cannot_masquerade_as_postselection_completion(evidence):
    path = evidence.annual / "protocol.json"
    value = read(path); value["validation_audit_file"] = "independent_audit.validation.json"; dump(path, value)
    repin_protocol(evidence)
    with pytest.raises(ValueError, match="distinct completed post-selection"):
        evidence.load()


def test_postselection_audit_must_follow_the_actual_seal_even_with_valid_byte_pins(evidence):
    path = evidence.validation / "independent_audit.validation.postselection.json"
    value = read(path); value["audited_at"] = "2026-10-30T00:19:00Z"; dump(path, value)
    repin_protocol(evidence)
    with pytest.raises(ValueError, match="Annual must follow"):
        evidence.load()


def test_process_alias_cannot_disagree_with_audited_terminal_identity(evidence):
    path = evidence.annual / "test_process.json"
    value = read(path); value["exit_code"] = 1; dump(path, value); evidence.reseal()
    with pytest.raises(ValueError, match="process aliases"):
        evidence.load()

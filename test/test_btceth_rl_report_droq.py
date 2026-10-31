"""Synthetic reader only: no actual study, market, model or HTML."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from scripts import btceth_rl_report_droq as report


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def mapping(root):
    return {path.relative_to(root).as_posix(): report.sha(path) for path in root.rglob("*") if path.is_file()}


def touch(root, names):
    for name in names:
        path = root / name
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(("synthetic-fixture-only:" + name).encode())


def summary(*, family="cap075", profile="balanced", split="test", baseline=False, cash=False):
    rows = []; dates = report.VALIDATION if split == "validation" else report.TEST
    for cost in report.COSTS:
        for key in report.BASELINES if baseline else (*map(str, report.SEEDS), "ensemble"):
            primary = not baseline and key == "ensemble"
            if baseline:
                ret = 0. if key == "cash" else -.20
            elif cash:
                ret = 0.
            elif split == "validation":
                ret = .08 + .02*(family == "cap075") - .01*(cost == "stress")
            else:
                ret = (.12 if cost == "base" else .05) if primary else [-.01, .10, -.02, .12, -.03][int(key)]
            flat = cash or baseline and key == "cash"
            row = dict(run_id=f"test__{cost}__{key}" if baseline else f"{split}__{cost}__{family}__droq__{profile}__{key}",
                strategy=key if baseline else "droq_"+family, split=split, scenario=cost, initial_capital=10000., start=dates[0], end_exclusive=dates[1],
                net_return=ret, final_equity=10000*(1+ret), max_drawdown=0. if flat else -.40 if baseline else -.18,
                fees=0. if flat else 10. if cost == "base" else 20., average_exposure=0. if flat else .30)
            if not baseline:
                row.update(algorithm="droq", family=family, profile=profile, exposure_multiplier=report.BUDGETS[family], rebalance_band=.025,
                    seed=None if primary else int(key), primary_policy=primary)
            rows.append(row)
    return pd.DataFrame(rows)


def group(root, folder, frame, *, baseline=False):
    directory = root/folder; directory.mkdir(parents=True, exist_ok=True); curves, months = [], []
    for row in frame.itertuples():
        start, end = pd.Timestamp(row.start), pd.Timestamp(row.end_exclusive)
        dates = pd.date_range(start, end, freq="h", inclusive="left"); n = len(dates)
        peak = min(12000., row.final_equity/(1+row.max_drawdown)*.99)
        nav = np.full(n, 10000.) if row.max_drawdown == 0 else np.interp(np.arange(n), [0, n//4, n//2, n-1], [10000., peak, peak*(1+row.max_drawdown), row.final_equity])
        series = pd.Series(nav, index=dates); daily = series.resample("D").last(); dd = series/series.cummax().clip(lower=10000.)-1
        curves.append(pd.DataFrame(dict(run_id=row.run_id, timestamp=[start, *list(daily.index+pd.Timedelta(days=1))],
            equity=[10000., *daily], net_return=[0., *(daily/10000-1)], drawdown=[0., *dd.resample("D").last()])))
        ends = daily.resample("ME").last(); ret = ends.pct_change(); ret.iloc[0] = ends.iloc[0]/10000-1
        months.append(pd.DataFrame(dict(run_id=row.run_id, month=ends.index.strftime("%Y-%m"), **{"return": ret.to_numpy()})))
        if row.split == "test":
            key = row.strategy if baseline else "ensemble" if row.primary_policy else str(int(row.seed))
            case = directory/row.scenario/key; case.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(dict(timestamp=dates, equity=nav, cash=nav*(1-row.average_exposure), base_value=nav*row.average_exposure,
                exposure=row.average_exposure)).to_csv(case/"equity_1h.csv.gz", index=False, float_format="%.17g")
            trades = [dict(timestamp=start, side="BUY", qty=1., price=10000., fee=row.fees)] if row.fees else []
            pd.DataFrame(trades, columns=["timestamp", "side", "qty", "price", "fee"]).to_csv(case/"fills.csv", index=False)
            dump(case/"fingerprints.json", {"equity": hashlib.sha256(row.run_id.encode()).hexdigest()}); dump(case/"metadata.json", {})
    tables = frame, pd.concat(curves, ignore_index=True), pd.concat(months, ignore_index=True)
    for name, table in zip(report.GROUP_FILES, tables):
        table.to_csv(directory/name, index=False, float_format="%.17g")
    dump(directory/"completion.json", dict(case_count=len(frame), repeat_count=len(frame)))
    files = report.baseline_files() if baseline else report.group_files(frame.split.iloc[0], frame.family.iloc[0], frame.profile.iloc[0])
    touch(root, files)
    return tables


def proofs(frame, *, hours):
    return {r.run_id: dict(reconstructed=True, repeat_exact=True, hours=hours, fingerprints={"equity": hashlib.sha256(r.run_id.encode()).hexdigest()},
        **{k: float(getattr(r, k)) for k in ("net_return", "max_drawdown", "fees", "final_equity")}) for r in frame.itertuples()}


def audited(frame, *, validation=False):
    return dict(passed=True, status="partial_test_pending" if validation else "passed", phase="validation" if validation else "test",
        validation_complete=True, test_complete=not validation, training_model_count=5, validation_run_count=24, test_run_count=0 if validation else 12,
        baseline_run_count=0 if validation else 14, run_count=24 if validation else 26, repeat_count=24 if validation else 12, checks_total=100, failure_count=0, failures=[],
        declared_research_targets=report.TARGETS, fresh_holdout_success=False, training_budget_multiple=1, recipe_is_pure_official_default=False,
        accounting_passage_is_not_goal_success=True, runs=proofs(frame, hours=7080 if validation else 8760), dataset_sha256="d"*64,
        training_calendar=dict(requested_train_days=1827, candidate_windows=1796, eligible_windows=1500, excluded_windows=296, covered_days=1700), **{flag: True for flag in report.FLAGS})


@pytest.fixture
def evidence(tmp_path, monkeypatch):
    study, main, source = (tmp_path/name for name in ("study", "main", "source"))
    for root in (study, main, source):
        root.mkdir()
    touch(source, report.SOURCE_FILES); source_pins = mapping(source)
    for name in report.SOURCE_FILES:
        path = study/"sources"/name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes((source/name).read_bytes())
    monkeypatch.setattr(report, "RUNNER_SHA256", source_pins["scripts/btceth_rl_droq.py"])
    monkeypatch.setattr(report, "AUDITOR_SHA256", source_pins["scripts/audit_btceth_rl_droq.py"])
    upstream = study/"upstream"
    core_names = ("OriginalREDQCodebase/redq/algos/core.py", "OriginalREDQCodebase/redq/algos/redq_sac.py")
    author_names = {"LICENSE", *core_names, *(f"fake_author_source{i}.py" for i in range(24))}; touch(upstream, author_names)
    dump(upstream/"verified_manifest.json", dict(commit=report.UPSTREAM_COMMIT, license="MIT", file_count=27, all_git_blob_sha1_verified=True,
        files=[dict(path=name, sha256=report.sha(upstream/name)) for name in sorted(author_names)]))
    dump(upstream/"github_commit.json", {"synthetic_fixture_only": True}); dump(upstream/"github_tree.json", {"synthetic_fixture_only": True})
    monkeypatch.setattr(report, "MANIFEST_SHA256", report.sha(upstream/"verified_manifest.json"))
    dump(study/"effective_config_contract.json", dict(effective_constructor_kwargs=report.RECIPE, official_commit=report.UPSTREAM_COMMIT))
    monkeypatch.setattr(report, "CONFIG_SHA256", report.sha(study/"effective_config_contract.json"))
    gpus = [dict(index=i, uuid=f"GPU-fixture{i}", name="Tesla V100", memory_total_mib=32768, memory_used_mib=10, utilization_percent=0) for i in range(2)]
    packages = {"fixture": "only"}; dump(main/"runtime.json", dict(physical_gpus=gpus))
    dump(main/"frozen_spec.json", dict(timesteps_per_model=32768, research_targets=report.TARGETS, dataset_sha256="d"*64,
        dataset_manifest_sha256="e"*64, code_sha256={k: source_pins[k] for k in report.FINANCIAL}, runtime_packages=packages))
    baseline = group(main, "evaluation/test/baselines", summary(baseline=True), baseline=True)
    dump(main/"test_completion.json", dict(model_groups=4, baselines=7))
    main_audit = dict(passed=True, status="passed", test_complete=True, run_count=172, checks_total=100, failure_count=0, failures=[],
        input_sha256=mapping(main), runs=proofs(baseline[0], hours=8760))
    main_audit["runs"].update({f"other_synthetic_{i}": {"reconstructed": True} for i in range(158)})
    dump(main/"independent_audit.json", main_audit)
    fit_auth = dict(authorized_at="2026-10-29T23:59:00Z", fit_validation_authorized=True, no_live_trading=True, reviewed_source_sha256=source_pins,
        hyperparameters=report.RECIPE, seeds=list(report.SEEDS), timesteps_per_model=32768, budgets=report.BUDGETS, **{flag: True for flag in report.FLAGS})
    dump(study/"preflight/fit_validation_authorization.json", fit_auth)
    p = dict(algorithm="droq", profiles={"balanced": {"drawdown_penalty": .25, "rebalance_band": .025}}, budgets=report.BUDGETS, seeds=list(report.SEEDS),
        train=list(report.TRAIN), validation=list(report.VALIDATION), test=list(report.TEST), costs=report.COSTS, capital=10000., timesteps_per_model=32768,
        training_budget_multiple=1, environment_step_budget_multiple=1, compute_budget_comparable=False, train_episode_days=32, model_count=5,
        validation_cases=24, validation_repeats=24, test_cases=12, test_repeats=12, cash_floor=.05, max_asset_weight=.95,
        hyperparameters=report.RECIPE, expected_counters=report.COUNTS, recipe_is_pure_official_default=False, seed_selection=False, selection_uses_test=False,
        new_recipe_search=False, annual_requires_separate_authorization=True, research_targets=report.TARGETS,
        selection_score="mean(log1p(net_return)-0.5*abs(max_drawdown)-2*max(abs(max_drawdown)-0.20,0)) across base/stress", tie_break="raw before cap075, then profile lexical identifier",
        prepared_at="2026-10-30T00:00:00Z", original_frozen_spec_sha256=report.sha(main/"frozen_spec.json"), dataset_sha256="d"*64, dataset_manifest_sha256="e"*64,
        source_sha256=source_pins, original_input_sha256=mapping(main), runtime_packages=packages, physical_gpus=gpus,
        seed_gpu_assignments={str(seed): gpus[seed%2]["uuid"] for seed in report.SEEDS}, family_gpu_assignments={family: gpus[i]["uuid"] for i, family in enumerate(report.BUDGETS)},
        upstream_evidence_sha256=mapping(upstream), upstream_commit=report.UPSTREAM_COMMIT, effective_config_contract_sha256=report.CONFIG_SHA256,
        fit_validation_authorization_sha256=report.sha(study/"preflight/fit_validation_authorization.json"), **{flag: True for flag in report.FLAGS})
    dump(study/"protocol.json", p); psha = report.sha(study/"protocol.json")
    dump(study/"protocol_receipt.json", dict(protocol_file="protocol.json", protocol_sha256=psha, sealed_at="2026-10-30T00:01:00Z"))
    def phase(name, pre, start, complete, end):
        stamp = lambda v: f"2026-10-30T00:{v}:00Z"
        (study/(name+".log")).write_text("synthetic terminal only\n")
        dump(study/(name+"_process.json"), dict(phase=name, status="completed", exit_code=0, protocol_sha256=psha, source_sha256=source_pins["scripts/btceth_rl_droq.py"],
            supervisor_pid=100, child_pid=101, command=["fixture", "-u", "/fixture/scripts/btceth_rl_droq.py", "--phase", "_"+name],
            started_at=stamp(start), completed_at=stamp(end), log_relative_path=name+".log", log_sha256=report.sha(study/(name+".log"))))
        counts = dict(model_count=5) if name == "fit" else dict(case_count=24 if name == "validate" else 12, repeat_count=24 if name == "validate" else 12)
        dump(study/(name+"_completion.json"), dict(protocol_sha256=psha, at=stamp(complete), **counts))
        current = deepcopy(gpus); current[0].update(memory_used_mib=25, utilization_percent=1)
        dump(study/"preflight"/(name+".json"), dict(at=stamp(pre), physical_gpus=current))
    phase("fit", "02", "03", "11", "12"); phase("validate", "13", "14", "16", "17")
    effective = {k: report.RECIPE[k] for k in ("replay_size", "batch_size", "lr", "gamma", "polyak", "auto_alpha", "start_steps", "utd_ratio", "num_Q", "num_min", "q_target_mode", "policy_update_delay")}
    effective.update(hidden_sizes=[256, 256], delay_update_steps=5000, target_entropy=-2., dropout_per_q=[[.01,.01],[.01,.01]],
        layer_norm_per_q=[[[256],[256]],[[256],[256]]], target_training=[True,True], alpha_current=1., log_alpha_current=0.)
    records = []
    for seed in report.SEEDS:
        folder = f"training/droq__balanced__seed{seed}"; touch(study, {folder+"/model_tensors.pt", folder+"/training_timeout_transitions.jsonl"})
        device = dict(visible_gpu_uuid=p["seed_gpu_assignments"][str(seed)], device="cuda:0", name="Tesla V100", capability=[7,0])
        dump(study/folder/"started.json", dict(at="2026-10-30T00:04:00Z", protocol_sha256=psha, device=device))
        parameter = str(seed)*64
        dump(study/folder/"model.json", dict(parameter_sha256=parameter, tensor_sha256=report.sha(study/folder/"model_tensors.pt"),
            counters=report.COUNTS, upstream_commit=report.UPSTREAM_COMMIT, effective_alpha=1., log_alpha_value=0.))
        record = dict(algorithm="droq", profile="balanced", seed=seed, actual_timesteps=32768, actual_counters=report.COUNTS, hyperparameters=report.RECIPE,
            effective_hyperparameters=effective, algorithm_class="redq.algos.redq_sac.REDQSACAgent", recipe_is_pure_official_default=False,
            training_budget_multiple=1, environment_step_budget_multiple=1, compute_budget_comparable=False, protocol_sha256=psha, completed_at="2026-10-30T00:10:00Z",
            model_tensor_file=folder+"/model_tensors.pt", model_tensor_sha256=report.sha(study/folder/"model_tensors.pt"),
            model_manifest_file=folder+"/model.json", model_manifest_sha256=report.sha(study/folder/"model.json"),
            timeout_trace_relative_path=folder+"/training_timeout_transitions.jsonl", timeout_trace_sha256=report.sha(study/folder/"training_timeout_transitions.jsonl"),
            parameter_sha256=parameter, device=device, training_window_quality=dict(requested_train_days=1827, candidate_windows=1796, eligible_windows=1500, excluded_windows=296, covered_training_days=1700),
            upstream_loaded_sources={name: dict(sha256=p["upstream_evidence_sha256"][name], file="/fixture/upstream/"+name) for name in core_names}, **{flag: True for flag in report.FLAGS})
        dump(study/folder/"record.json", record); records.append(record)
    dump(study/"training_completion.json", dict(model_count=5, records=records, protocol_sha256=psha))
    groups = [group(study, f"validation/{family}/balanced", summary(family=family, split="validation")) for family in report.BUDGETS]
    for name, i in zip(report.GROUP_FILES[:3], range(3)):
        pd.concat([g[i] for g in groups], ignore_index=True).to_csv(study/"validation"/name, index=False, float_format="%.17g")
    dump(study/"validation_completion.json", dict(case_count=24, repeat_count=24, protocol_sha256=psha))
    initial = audited(pd.concat([g[0] for g in groups], ignore_index=True), validation=True)
    initial.update(audited_at="2026-10-30T00:18:00Z", input_sha256=mapping(study), main_input_sha256={k:v for k,v in mapping(main).items() if k!="test_completion.json"}, source_sha256=source_pins)
    dump(study/"independent_audit.validation.json", initial)
    pool = [report._candidate(g[0], g[0].family.iloc[0], "balanced") for g in groups]; pool.sort(key=lambda r:(-r["score"],0 if r["family"]=="raw" else 1,r["profile"]))
    selection = dict(selected_at="2026-10-30T00:19:00Z", protocol_sha256=psha, audit_sha256=report.sha(study/"independent_audit.validation.json"), candidates=pool,
        candidate=pool[0], training_source_sha256={name:report.sha(study/name) for name in report.training_files() if not name.endswith("/started.json")},
        validation_source_sha256={name:report.sha(study/name) for name in report.validation_files()}, seed_selection=False, selection_uses_test=False,
        test_results_available_to_selector=False, **{flag:True for flag in report.FLAGS})
    dump(study/"selection.json", selection)
    dump(study/"selection_receipt.json", dict(selection_file="selection.json", selection_sha256=report.sha(study/"selection.json"), protocol_sha256=psha, sealed_at="2026-10-30T00:20:00Z"))
    dump(study/"annual_authorization.json", dict(annual_evaluation_authorized=True, no_live_trading=True, financial_parameters_fixed=True, selection_sha256=report.sha(study/"selection.json"), authorized_at="2026-10-30T00:21:00Z", **{flag:True for flag in report.FLAGS}))
    dump(study/"test_started.json", dict(at="2026-10-30T00:22:00Z", protocol_sha256=psha, selection_sha256=report.sha(study/"selection.json"), authorization_sha256=report.sha(study/"annual_authorization.json")))
    phase("test", "23", "24", "25", "26")
    chosen = pool[0]; annual = group(study, f"test/{chosen['family']}/balanced", summary(family=chosen["family"]))
    for row in annual[0].itertuples():
        key = "ensemble" if row.primary_policy else str(int(row.seed)); members = records if row.primary_policy else [records[int(row.seed)]]; parameters = [r["parameter_sha256"] for r in members]
        case = study/f"test/{chosen['family']}/balanced"/row.scenario/key
        dump(case/"metadata.json", dict(algorithm="droq", family=chosen["family"], profile="balanced", model_count=len(members), parameter_sha256_before=parameters, parameter_sha256_after=parameters))
        dump(case/"replay.json", dict(passed=True, repeat_exact=True, fingerprints=proofs(annual[0],hours=8760)[row.run_id]["fingerprints"], repeated_fingerprints=proofs(annual[0],hours=8760)[row.run_id]["fingerprints"], rng_sha256_before="r"*64, rng_sha256_after_first="r"*64, rng_sha256_after_repeat="r"*64))
    complete = read(study/"test_completion.json"); complete.update(candidate=chosen, selection_sha256=report.sha(study/"selection.json")); dump(study/"test_completion.json", complete)
    audit = audited(annual[0]); audit.update(audited_at="2026-10-30T00:27:00Z", baseline_runs=proofs(baseline[0],hours=8760), research_targets=report._targets(annual[0],baseline[0],chosen))
    def reseal():
        audit.update(input_sha256={k:v for k,v in mapping(study).items() if not k.startswith("independent_audit") and k!="goal_candidate_evidence.json"}, main_input_sha256={k:v for k,v in mapping(main).items() if k!="test_completion.json"}, source_sha256=mapping(source))
        audit["input_sha256"]["independent_audit.validation.json"] = report.sha(study/"independent_audit.validation.json")
        dump(study/"independent_audit.json", audit); local = deepcopy(audit); local["checks_total"] += 1; dump(study/"independent_audit.local.json", local)
    def goal():
        dump(study/"goal_candidate_evidence.json", dict(at="2026-10-30T00:28:00Z", protocol_sha256=psha, selection_sha256=report.sha(study/"selection.json"), audit_sha256=report.sha(study/"independent_audit.json"), accounting_audit_passed=True, **audit["research_targets"]))
    reseal(); goal()
    return SimpleNamespace(study=study, main=main, source=source, audit=audit, initial=initial, annual=annual, baseline=baseline, chosen=chosen, reseal=reseal, goal=goal,
        load=lambda **kwargs:report.load_droq(study,main,source_root=source,**kwargs))


def test_complete_reader_has_seven_separate_scopes_26_accounts_and_own_failed_goal(evidence):
    before = mapping(evidence.study), mapping(evidence.main)
    value = evidence.load()
    assert set(value["queries"]) == {"droq_" + name for name in ("summary", "equity", "monthly", "targets", "validation_selection", "training", "audit")}
    assert [len(value["queries"]["droq_" + n]["rows"]) for n in ("summary", "equity", "monthly", "targets", "validation_selection", "training", "audit")] == [26, 26 * 366, 26 * 13, 7, 2, 5, 2]
    assert {row["droq_role"] for row in value["queries"]["droq_summary"]["rows"]} == {"droq_selected", "original_baseline_reference"}
    assert value["targets"]["positive_base_training_seeds"] == 2 and value["targets"]["goal_performance_achieved"] is False
    assert value["original_scopes_unchanged"] is True and value["fresh_holdout_success"] is False and value["training_budget_multiple"] == 1
    assert value["queries"]["droq_training"]["source"]["period"] == dict(start=report.TRAIN[0], endExclusive=report.TRAIN[1])
    assert value["queries"]["droq_validation_selection"]["source"]["period"] == dict(start=report.VALIDATION[0], endExclusive=report.VALIDATION[1])
    assert all(Path(name).is_absolute() for name in value["external_input_sha256"])
    assert before == (mapping(evidence.study), mapping(evidence.main)) and not list(evidence.study.rglob("*.html"))


@pytest.mark.parametrize("damage", ["partial", "missing", "failed", "missing_repeat"])
def test_no_annual_payload_without_full_evidence(evidence, damage):
    if damage == "missing":
        (evidence.study / "independent_audit.json").unlink()
    else:
        evidence.audit.update({"status": "partial_test_pending", "test_complete": False} if damage == "partial" else {"passed": False} if damage == "failed" else {"repeat_count": 11})
        evidence.reseal()
    with pytest.raises((ValueError, FileNotFoundError)):
        evidence.load()


def test_local_audit_identity_and_retained_remote_goal_identity_remain_distinct(evidence):
    local = evidence.study / "independent_audit.local.json"
    value = deepcopy(evidence.audit); value["checks_total"] += 1; dump(local, value)
    payload = evidence.load(audit_path=local)
    rows = payload["queries"]["droq_audit"]["rows"]
    assert len(rows) == 2 and rows[0]["sha256"] != rows[1]["sha256"]
    assert read(evidence.study / "goal_candidate_evidence.json")["audit_sha256"] == rows[1]["sha256"]


def test_original_completion_is_checked_indirectly_through_retained172(evidence):
    assert "test_completion.json" not in evidence.audit["main_input_sha256"]
    value = evidence.load()
    assert (evidence.main / "test_completion.json").as_posix() in value["external_input_sha256"]
    dump(evidence.main / "test_completion.json", {"changed": True})
    with pytest.raises(ValueError, match="SHA changed"):
        evidence.load()


@pytest.mark.parametrize("filename,column", [("summary.csv", "net_return"), ("daily_equity.csv", "equity"), ("monthly_returns.csv", "return")])
def test_resealed_but_inconsistent_financial_tables_are_rejected(evidence, filename, column):
    path = evidence.study / f"test/{evidence.chosen['family']}/{evidence.chosen['profile']}" / filename
    frame = pd.read_csv(path); frame.loc[frame.index[-1], column] += .01; frame.to_csv(path, index=False)
    evidence.reseal()
    with pytest.raises(ValueError):
        evidence.load()


def test_changed_model_is_not_hidden_by_final_audit_label(evidence):
    (evidence.study / "training/droq__balanced__seed0/model_tensors.pt").write_bytes(b"another checkpoint")
    evidence.reseal()
    with pytest.raises(ValueError, match="SHA changed|checkpoint"):
        evidence.load()


def test_false_goal_boolean_cannot_substitute_for_two_profitable_seeds(evidence):
    path = evidence.study / "goal_candidate_evidence.json"; value = read(path); value["goal_performance_achieved"] = True; dump(path, value)
    with pytest.raises(ValueError, match="Own DroQ goal"):
        evidence.load()


def test_resealed_audit_goal_must_match_droq_own_annual_metrics(evidence):
    evidence.audit["research_targets"]["gates"]["at_least_three_positive_training_seeds"] = True
    evidence.reseal(); evidence.goal()
    with pytest.raises(ValueError, match="Actual DroQ"):
        evidence.load()


@pytest.mark.parametrize("damage", ["model_count", "budget", "recipe", "timeout"])
def test_original_training_record_budgets_and_semantics_are_not_optional(evidence, damage):
    if damage == "model_count":
        path = evidence.study / "training_completion.json"; item = read(path); item["model_count"] = 4
    else:
        path = evidence.study / "training/droq__balanced__seed0/record.json"; item = read(path)
        if damage == "budget":
            item["actual_timesteps"] = 65536
        elif damage == "recipe":
            item["effective_hyperparameters"]["policy_update_delay"] = 3
        else:
            item["actual_counters"]["target_updates_per_network"] = [1, 1]
    dump(path, item); evidence.reseal()
    with pytest.raises(ValueError):
        evidence.load()


@pytest.mark.parametrize("field,value", [("seed", 9), ("algorithm", "sac"), ("family", "other"), ("initial_capital", 5000.), ("start", report.TEST[0])])
def test_validation_population_cannot_change_or_use_annual_dates(field, value):
    frame = summary(split="validation"); frame.loc[0, field] = value
    with pytest.raises(ValueError):
        report._candidate(frame, "cap075", "balanced")


def test_real_zero_cash_policy_is_retained_without_inventing_success_or_missing_rows():
    frame, controls = summary(cash=True), summary(baseline=True)
    report._summary(frame, split="test", family="cap075", profile="balanced")
    result = report._targets(frame, controls, dict(algorithm="droq", family="cap075", profile="balanced"))
    assert result["base_net_return"] == result["base_max_drawdown"] == result["base_fees_usdt"] == 0.
    assert result["positive_base_training_seeds"] == 0 and result["goal_performance_achieved"] is False
    with pytest.raises(ValueError):
        report._summary(frame.iloc[:-1], split="test", family="cap075", profile="balanced")


@pytest.mark.parametrize("changed", ["authorization", "marker", "early_seal"])
def test_new_year_requires_actual_authorization_after_this_validation_choice(evidence, changed):
    name = "selection_receipt.json" if changed == "early_seal" else "annual_authorization.json" if changed == "authorization" else "test_started.json"
    path = evidence.study / name; value = read(path)
    if changed == "early_seal":
        value["sealed_at"] = "2026-10-30T00:17:00Z"
    else:
        value["selection_sha256"] = "a" * 64
    dump(path, value); evidence.reseal()
    with pytest.raises(ValueError):
        evidence.load()


def test_current_source_and_archived_source_have_to_agree(evidence):
    (evidence.source / "scripts/btceth_rl_droq.py").write_text("# changed implementation\n")
    evidence.reseal()
    with pytest.raises(ValueError, match="SHA changed"):
        evidence.load()


def test_counter_schema_rejects_mixed_conflict_and_boolean_fake_counts():
    value = audited(summary())
    report._audit(value)
    for damage in ({"checks": 99}, {"checks_total": True}, {"checks": True}, {"passed": 1}):
        with pytest.raises(ValueError):
            report._audit({**value, **damage})


def test_unsafe_external_path_cannot_escape_source_tree(evidence):
    evidence.audit["source_sha256"]["../outside.py"] = "a" * 64
    dump(evidence.study / "independent_audit.json", evidence.audit)
    with pytest.raises(ValueError, match="relative"):
        evidence.load()


def test_actual_hourly_and_trade_readback_accepts_cash_zero_but_rejects_export_damage(tmp_path):
    frame = summary(baseline=True).iloc[:2].copy()
    tables = group(tmp_path, "evaluation/test/baselines", frame, baseline=True)
    proof = proofs(frame, hours=8760)
    report._case_readbacks(report.Sources(), tmp_path, frame, tables[1], proof, baseline=True)
    case = tmp_path/"evaluation/test/baselines/base/btc_hold"
    path = case/"fills.csv"; original = path.read_bytes(); trades = pd.read_csv(path); trades.loc[0, "fee"] += 1.; trades.to_csv(path, index=False)
    with pytest.raises(ValueError, match="exported spot trades"):
        report._case_readbacks(report.Sources(), tmp_path, frame, tables[1], proof, baseline=True)
    path.write_bytes(original)
    path = case/"equity_1h.csv.gz"; hours = pd.read_csv(path); hours.loc[0, "cash"] += 1.; hours.to_csv(path, index=False)
    with pytest.raises(ValueError, match="cash plus funded holdings"):
        report._case_readbacks(report.Sources(), tmp_path, frame, tables[1], proof, baseline=True)

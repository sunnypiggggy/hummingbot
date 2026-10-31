"""Source-backed report contract tests. All market values are pytest fixtures.

No actual research holdout, inference, report/HTML build or network is used.
"""
from copy import deepcopy
import json
from pathlib import Path

import pandas as pd
import pytest

from scripts import btceth_rl_report_app as app
from scripts import btceth_rl_report_extension as ext
from scripts.audit_btceth_rl import _research_target_result, recompute_selection
from scripts.audit_btceth_rl_exposure_cap import candidate_pool, objective_evidence, extension_validation_files, CASE_FILES, GROUP_FILES
from test.test_btceth_rl_report import complete_study_root, synthetic_summary, curves_for, write_json, ROOT_INPUTS


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def put_stub(root, name):
    path = root / name
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"SYNTHETIC_PYTEST_EVIDENCE_NOT_MARKET_DATA")
    return app.sha(path)


def tables(root, frame):
    root.mkdir(parents=True, exist_ok=True)
    history, months = curves_for(frame)
    for name, values in (("summary.csv", frame), ("daily_equity.csv", history), ("monthly_returns.csv", months)):
        values.to_csv(root / name, index=False, float_format="%.17g")
    return history, months


def fixture_extension(main, family):
    """Build only temporary fixture CSV/JSON; actual results remain untouched."""
    output = main.parent / (main.name + "_cap_fixture")
    output.mkdir()
    # A retained failed earlier audit is evidence, not the selected passed
    # final audit. Its presence must neither override nor be silently mutated.
    write_json(output / "independent_audit.validation.json", {"passed":False,"status":"failed",
        "test_complete":False,"run_count":24,"failure_count":824,
        "scope":"SYNTHETIC_RETAINED_FAILED_AUDIT_ONLY"})
    summary = pd.read_csv(main / "summary.csv", float_precision="round_trip")
    # SAC is the original overall leader and succeeds. The raw new-model
    # leader is CrossQ, whose seed-stability gate fails. Neither identity can
    # silently stand in for the other in the extended report.
    use = summary.algorithm.eq("sac") & summary.split.eq("validation")
    summary.loc[use, "net_return"] += .4
    summary.loc[summary.run_id.eq("test__base__crossq__balanced__2"), "net_return"] = -.03
    summary["final_equity"] = 10000 * (1 + summary.net_return)
    tables(main, summary)
    original = recompute_selection(summary)
    original.update(frozen_spec_sha256=app.sha(main / "frozen_spec.json"),
        training_source_sha256=read(main / "validation_selection.json")["training_source_sha256"])
    val_pins = {}
    for algorithm in app.ALGORITHMS:
        for profile in app.PROFILES:
            name = f"evaluation/validation/{algorithm}/{profile}/summary.csv"
            path = main / name
            path.parent.mkdir(parents=True, exist_ok=True)
            summary.loc[summary.split.eq("validation") & summary.algorithm.eq(algorithm) & summary.profile.eq(profile)].to_csv(path,index=False,float_format="%.17g")
            val_pins[name] = app.sha(path)
    original["validation_source_sha256"] = val_pins
    write_json(main / "validation_selection.json", original)
    assert original["primary_validation_leader"] == {"algorithm":"sac","profile":"balanced"}
    raw_frames = {(a,p): summary.loc[summary.split.eq("validation") & summary.algorithm.eq(a) & summary.profile.eq(p)].copy()
                  for a in ext.ALGORITHMS for p in ext.PROFILES}
    raw_pool = candidate_pool(raw_frames, {p: cap_frame(raw_frames[("tqc",p)], p, 0.) for p in ext.PROFILES})
    raw_pool = [r for r in raw_pool if r["family"] == "raw"]
    raw_leader = {key:raw_pool[0][key] for key in ("algorithm","profile")}
    assert raw_leader == {"algorithm":"crossq","profile":"balanced"}
    write_json(main / "goal_candidate_protocol.json", {"declared_at":"2026-10-29T00:01:00Z"})
    raw_selection = {"selected_at":"2026-10-29T00:02:00Z", "new_model_validation_leader":raw_leader,
        "declared_protocol_sha256":app.sha(main / "goal_candidate_protocol.json"),
        "original_validation_selection_sha256":app.sha(main / "validation_selection.json"),
        "original_frozen_spec_sha256":app.sha(main / "frozen_spec.json"),
        "selection_uses_test":False,"seed_selection":False}
    write_json(main / "goal_candidate_selection.json", raw_selection)
    write_json(main / "goal_candidate_selection_receipt.json", {"sealed_at":"2026-10-29T00:03:00Z",
        "selection_file":"goal_candidate_selection.json","selection_sha256":app.sha(main / "goal_candidate_selection.json"),
        "declared_protocol_sha256":app.sha(main / "goal_candidate_protocol.json"),
        "original_validation_selection_sha256":app.sha(main / "validation_selection.json")})
    write_json(main / "test_started.json", {"at":"2026-10-29T00:08:00Z",
        "validation_selection_sha256":app.sha(main / "validation_selection.json"),"frozen_spec_sha256":app.sha(main / "frozen_spec.json")})
    audited = read(main / "independent_audit.json")
    audited["input_sha256"] = {name:app.sha(main / name) for name in ROOT_INPUTS}
    audited["research_targets"] = _research_target_result(summary, original)
    write_json(main / "independent_audit.json", audited)
    equal = summary.loc[summary.run_id.eq("test__base__equal_hold")].iloc[0].to_dict()
    raw_test = summary.loc[summary.split.eq("test") & summary.algorithm.eq(raw_leader["algorithm"]) & summary.profile.eq(raw_leader["profile"])]
    raw_targets = objective_evidence(raw_test,{"family":"raw",**raw_leader},equal)
    assert not raw_targets["goal_performance_achieved"] and audited["research_targets"]["all_predeclared_primary_targets_met"]
    write_json(main / "goal_candidate_evidence.json", {**raw_targets,"candidate":raw_leader,
        "original_global_validation_leader":original["primary_validation_leader"],
        "accounting_audit_passed":True,"selection_uses_test":False,"seed_selection":False,
        "audit_sha256":app.sha(main / "independent_audit.json"),
        "frozen_inputs_sha256":{name:app.sha(main / name) for name in ext.MAIN_ANCHORS}})
    value = .30 if family == "cap075" else 0.
    capped = {p:cap_frame(raw_frames[("tqc",p)],p,value-(.02 if p=="return" else 0)) for p in ext.PROFILES}
    source_name = "scripts/SYNTHETIC_FROZEN_FINANCIAL_SOURCE.py"
    put_stub(output / "sources",source_name)
    protocol = {"prepared_at":"2026-10-29T00:04:00Z", "multiplier":.75,"capital":10000.,
        "profiles":list(ext.PROFILES),"seeds":list(range(5)),"effective_max_coin_target_total":.7125,
        "original_frozen_spec_sha256":app.sha(main / "frozen_spec.json"),
        "original_goal_protocol_sha256":app.sha(main / "goal_candidate_protocol.json"),
        "source_sha256":{source_name:app.sha(output / "sources" / source_name)},
        "training_source_sha256":{k:v for k,v in original["training_source_sha256"].items() if "tqc__" in k}}
    write_json(output / "protocol.json",protocol)
    write_json(output / "protocol_receipt.json",{"sealed_at":"2026-10-29T00:05:00Z","protocol_file":"protocol.json","protocol_sha256":app.sha(output / "protocol.json")})
    for p,frame in capped.items():
        tables(output / "validation" / p,frame)
    write_json(output / "validation_completion.json",{"case_count":24,"repeat_count":24})
    immutable = {name:put_stub(output,name) for name in extension_validation_files()}
    pool = candidate_pool(raw_frames,capped)
    chosen = pool[0]
    assert chosen["family"] == family
    selection = {"selected_at":"2026-10-29T00:06:00Z","protocol_sha256":app.sha(output / "protocol.json"),
        "original_validation_selection_sha256":app.sha(main / "validation_selection.json"),
        "original_goal_selection_sha256":app.sha(main / "goal_candidate_selection.json"),
        "original_goal_receipt_sha256":app.sha(main / "goal_candidate_selection_receipt.json"),
        "original_global_validation_leader":original["primary_validation_leader"],"original_new_model_validation_leader":raw_leader,
        "training_source_sha256":original["training_source_sha256"],"validation_source_sha256":val_pins,
        "extension_source_sha256":immutable,"candidates":pool,"candidate":chosen,"selection_uses_test":False,"seed_selection":False}
    write_json(output / "selection.json",selection)
    write_json(output / "selection_receipt.json",{"sealed_at":"2026-10-29T00:07:00Z","selection_file":"selection.json",
        "selection_sha256":app.sha(output / "selection.json"),"protocol_sha256":app.sha(output / "protocol.json")})
    actual = summary.loc[summary.split.eq("test") & summary.algorithm.eq(chosen["algorithm"]) & summary.profile.eq(chosen["profile"])].copy()
    raw_pins, test_pins = {}, {}
    if family == "cap075":
        actual = cap_frame(actual,chosen["profile"],.16)
        tables(output / "test" / chosen["profile"],actual)
        names = {*(f"test/{chosen['profile']}/{f}" for f in GROUP_FILES),
                 *(f"test/{chosen['profile']}/{c}/{k}/{f}" for c in ("base","stress") for k in (*map(str,range(5)),"ensemble") for f in CASE_FILES)}
        test_pins = {name:put_stub(output,name) for name in names}
    else:
        actual["family"],actual["exposure_multiplier"] = "raw",1.
        folder = f"evaluation/test/{chosen['algorithm']}/{chosen['profile']}"
        names = {*(f"{folder}/{f}" for f in GROUP_FILES),*(f"{folder}/{c}/{k}/{f}" for c in ("base","stress") for k in (*map(str,range(5)),"ensemble") for f in CASE_FILES[:6])}
        raw_pins = {name:put_stub(main,name) for name in names}
    full = pd.concat([*capped.values(),actual],ignore_index=True)
    tables(output,full)
    targets = objective_evidence(actual,chosen,equal)
    write_json(output / "test_started.json",{"selection_sha256":app.sha(output / "selection.json"),"protocol_sha256":app.sha(output / "protocol.json"),"main_test_started_sha256":app.sha(main / "test_started.json")})
    write_json(output / "test_completion.json",{"case_count":12,"repeat_count":12,"protocol_sha256":app.sha(output / "protocol.json"),
        "selection_sha256":app.sha(output / "selection.json"),"main_test_completion_sha256":app.sha(main / "test_completion.json"),
        "mode":"cap075_evaluation" if family=="cap075" else "raw_reference","raw_source_sha256":raw_pins})
    contracts = ("protocol.json","protocol_receipt.json","selection.json","selection_receipt.json","validation_completion.json","test_started.json","test_completion.json","summary.csv","daily_equity.csv","monthly_returns.csv")
    final = {"passed":True,"status":"passed","test_complete":True,"failure_count":0,"failures":[],"run_count":36,"checks":123,
        "source_sha256":{name:app.sha(ext.ROOT / name) for name in ("scripts/audit_btceth_rl_exposure_cap.py","scripts/audit_btceth_rl.py")},
        "input_sha256":{**immutable,**test_pins,**{name:app.sha(output / name) for name in contracts}},
        "main_input_sha256":{**raw_pins,**{name:app.sha(main / name) for name in ROOT_INPUTS}},
        "runs":{r.run_id:{"reconstructed":True,"hours":7080 if r.split=="validation" else 8760} for r in full.itertuples(index=False)},
        "research_targets":targets}
    write_json(output / "independent_audit.json",final)
    local = {**final,"audited_at":"2026-10-30T01:00:00Z","location":"separate_local_reaudit_fixture"}
    write_json(output / "independent_audit.local.json",local)
    write_json(output / "goal_candidate_evidence.json",{**targets,"candidate":{k:chosen[k] for k in ("family","algorithm","profile")},
        "audit_sha256":app.sha(output / "independent_audit.json"),"protocol_sha256":app.sha(output / "protocol.json"),
        "selection_sha256":app.sha(output / "selection.json"),"accounting_audit_passed":True,"selection_uses_test":False,"seed_selection":False})
    return main,output


def cap_frame(frame,profile,value):
    result=frame.copy()
    result["family"],result["algorithm"],result["strategy"],result["exposure_multiplier"]="cap075","tqc","tqc_cap075",.75
    result["profile"]=profile
    result["run_id"]=result.run_id.str.replace("__tqc__","__cap075__tqc__",regex=False)
    result["net_return"]=value-result.scenario.eq("stress")*.02
    result["final_equity"]=10000*(1+result.net_return)
    return result


@pytest.fixture(params=("cap075","raw"))
def complete_extension(complete_study_root,request):
    return fixture_extension(complete_study_root,request.param)


def test_complete_extension_uses_new_primary_and_never_counts_references_twice(complete_extension):
    main,output=complete_extension
    snapshot,evidence=app.make_reviewed_snapshot(main,extension_root=output,extension_audit_path=output/"independent_audit.local.json")
    choice=read(output/"selection.json")["candidate"]
    rows=snapshot["queries"]["strategy_summary"]["rows"]
    expected=208 if choice["family"]=="cap075" else 196
    assert len(rows)==len({r["run_id"] for r in rows})==expected==evidence["runs"]
    assert len(snapshot["queries"]["validation_selection"]["rows"])==10
    pool=snapshot["queries"]["validation_selection"]["rows"]
    assert sum(r["eligible_for_primary"] for r in pool)==8
    assert not any(r["eligible_for_primary"] for r in pool if r["algorithm"]=="sac")
    assert snapshot["report"]["originalGlobalLeader"]=={"algorithm":"sac","profile":"balanced"}
    assert snapshot["report"]["rawNewLeader"]=={"algorithm":"crossq","profile":"balanced"}
    assert snapshot["report"]["rawNewModelTargets"]["goal_performance_achieved"] is False
    assert snapshot["report"]["primaryLeader"]["algorithm"] == ("tqc_cap075" if choice["family"]=="cap075" else "crossq")
    assert len(snapshot["queries"]["training_records"]["rows"])==40
    identity=snapshot["report"]["extension"]["auditEvidence"]
    assert identity["primary_audit_sha256"] != identity["retained_remote_audit_sha256"]
    assert read(output/"goal_candidate_evidence.json")["audit_sha256"]==identity["retained_remote_audit_sha256"]
    retained_failure=output/"independent_audit.validation.json"
    assert read(retained_failure)["passed"] is False and read(retained_failure)["failure_count"]==824
    assert retained_failure.resolve().as_posix() not in evidence["external_input_sha256"]
    assert not (main/"report").exists() and not list(output.rglob("*.html"))
    assert Path(ext.__file__).resolve().as_posix() in evidence["external_input_sha256"]


def test_original_failed_raw_goal_cannot_be_made_successful_by_successful_sac(complete_extension):
    main,output=complete_extension
    path=main/"goal_candidate_evidence.json"
    value=read(path)
    assert value["goal_performance_achieved"] is False
    value["goal_performance_achieved"]=True
    write_json(path,value)
    with pytest.raises(ValueError,match="Original raw new-model goal"):
        app.make_reviewed_snapshot(main,extension_root=output)


@pytest.mark.parametrize("change",("partial","bad_remote_goal_sha","wrong_imported_source","changed_gate","missing_remote"))
def test_partial_or_conflicting_extension_never_produces_report(complete_study_root,change):
    main,output=fixture_extension(complete_study_root,"cap075")
    path=output/"independent_audit.local.json"
    value=read(path)
    if change=="partial":
        value.update(status="passed_validation_only",run_count=24,test_complete=False)
    elif change=="wrong_imported_source":
        value["source_sha256"]["scripts/audit_btceth_rl.py"]="0"*64
    elif change=="changed_gate":
        value["research_targets"]["gates"]["stress_return_nonnegative"]=False
    elif change=="missing_remote":
        (output/"independent_audit.json").unlink()
    else:
        evidence=read(output/"goal_candidate_evidence.json")
        evidence["audit_sha256"]=app.sha(path)
        write_json(output/"goal_candidate_evidence.json",evidence)
    write_json(path,value)
    with pytest.raises((ValueError,FileNotFoundError)):
        app.make_reviewed_snapshot(main,extension_root=output,extension_audit_path=path)
    assert not (main/"report").exists()


def test_external_pins_never_escape_a_supplied_source_root(tmp_path):
    for name in ("../secret","C:/outside","/outside"):
        with pytest.raises(ValueError,match="relative"):
            ext.safe(tmp_path,name)


def test_external_pin_changes_after_snapshot_are_detected(complete_extension):
    main,output=complete_extension
    _,evidence=app.make_reviewed_snapshot(main,extension_root=output)
    ext.verify_external_inputs(evidence["external_input_sha256"])
    (output/"goal_candidate_evidence.json").write_text("{}",encoding="utf-8")
    with pytest.raises(ValueError,match="external report source changed"):
        ext.verify_external_inputs(evidence["external_input_sha256"])


def test_risk_reference_is_optional_separate_and_never_adds_rank_rows(complete_study_root,monkeypatch):
    from scripts import btceth_rl_report_risk_controls as risk
    called=[]
    def load(*args,**kwargs):
        called.append((args,kwargs))
        return {"queries":{name:{"rows":[{"synthetic_fixture":True}]*6,"source":{},"methods":[]} for name in ("risk_budget_summary","risk_budget_equity","risk_budget_monthly")},
                "audit":{"passed":True,"cases":6,"full_repeats":6},"external_input_sha256":{}}
    monkeypatch.setattr(risk,"load_risk_controls",load)
    snapshot,evidence=app.make_reviewed_snapshot(complete_study_root,risk_reference_root=Path("explicit_fixture"),dataset_path=Path("unused_synthetic"))
    assert called and len(snapshot["queries"]["strategy_summary"]["rows"])==172
    assert len(snapshot["queries"]["validation_selection"]["rows"])==8
    assert evidence["risk_reference_runs"]==6 and evidence["runs"]==172
    assert snapshot["report"]["riskBudgetControls"]["includedInModelRanking"] is False
    assert snapshot["report"]["riskBudgetControls"]["exactRiskMatchClaimed"] is False


def test_optional_roots_cannot_implicitly_select_sources(tmp_path):
    with pytest.raises(ValueError,match="extension root"):
        app.make_reviewed_snapshot(tmp_path,extension_audit_path=tmp_path/"unrelated.json")
    with pytest.raises(ValueError,match="explicit frozen dataset"):
        app.make_reviewed_snapshot(tmp_path,risk_reference_root=tmp_path)

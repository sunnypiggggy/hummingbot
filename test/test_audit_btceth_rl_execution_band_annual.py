"""Synthetic annual-audit fixtures, never a policy run or a real annual read."""
from copy import deepcopy
import inspect

import numpy as np
import pandas as pd
import pytest

from scripts import audit_btceth_rl_execution_band_annual as annual
from scripts import audit_btceth_rl_execution_band as validation
from test.test_audit_btceth_rl_execution_band import hand_account, serialize_case
from test.test_audit_btceth_rl_execution_band import write_json, protocol_fixture, group_rows
from test.test_audit_btceth_rl_execution_band_gpu_guard import instance, allowed, snapshot


def rows():
    values=[]
    for cost in annual.COSTS:
        for key in (*map(str,annual.SEEDS),"ensemble"):
            ret=.06 if cost=="base" else .01
            if key!="ensemble":ret=[-.1,.1,.2,.3,-.1][int(key)]
            values.append(dict(run_id=f"test__{cost}__{annual.FAMILY}__tqc__balanced__{key}",
                algorithm="tqc",profile="balanced",family=annual.FAMILY,strategy=annual.STRATEGY,
                scenario=cost,split="test",seed=None if key=="ensemble" else int(key),primary_policy=key=="ensemble",
                initial_capital=10000.,start=annual.TEST[0],end_exclusive=annual.TEST[1],
                net_return=ret,max_drawdown=-.15,final_equity=10000*(1+ret),fees=10.,
                exposure_multiplier=.75,rebalance_band=.05))
    return pd.DataFrame(values)


def controls():
    return pd.DataFrame([dict(run_id=f"test__{cost}__{name}",net_return=-.1,max_drawdown=-.4)
                         for cost in annual.COSTS for name in annual.BASELINES])


def previous_goal():
    return {"goal_performance_achieved":False,"metrics":{"positive_base_training_seeds":2}}


def test_diagnostic_five_gates_do_not_change_old_failed_goal_or_claim_fresh_holdout():
    old=previous_goal()
    evidence=annual.performance_evidence(rows(),controls(),old)
    assert evidence["all_five_diagnostic_targets_met"]
    assert evidence["positive_base_training_seeds"]==3
    assert evidence["old_goal_performance_achieved"] is False
    assert evidence["old_positive_base_training_seeds"]==2
    assert evidence["fresh_holdout_success"] is False
    assert all(evidence[flag] is True for flag in annual.FLAGS)
    assert old==previous_goal()


@pytest.mark.parametrize("gate",["return","drawdown","stress","seed_count","equal_hold"])
def test_each_original_gate_is_required_without_new_weighting_or_seed_picking(gate):
    frame,baseline=rows(),controls()
    primary=frame.primary_policy&frame.scenario.eq("base")
    if gate=="return":
        frame.loc[primary,["net_return","final_equity"]]=[.04,10400.]
    elif gate=="drawdown":frame.loc[primary,"max_drawdown"]=-.21
    elif gate=="stress":
        frame.loc[frame.primary_policy&frame.scenario.eq("stress"),["net_return","final_equity"]]=[-.001,9990.]
    elif gate=="seed_count":
        frame.loc[frame.scenario.eq("base")&frame.seed.eq(3),["net_return","final_equity"]]=[0.,10000.]
    else:baseline.loc[baseline.run_id.eq("test__base__equal_hold"),"net_return"]=.07
    evidence=annual.performance_evidence(frame,baseline,previous_goal())
    assert not evidence["all_five_diagnostic_targets_met"]
    assert not evidence["goal_performance_achieved"]


def test_ensemble_and_stress_never_substitute_for_five_individual_base_seeds():
    frame=rows()
    frame.loc[~frame.primary_policy,["net_return","final_equity"]]=[-.01,9900.]
    result=annual.performance_evidence(frame,controls(),previous_goal())
    assert result["positive_base_training_seeds"]==0
    assert not result["gates"]["at_least_three_positive_training_seeds"]
    assert result["gates"]["base_return_at_least_5pct"]


@pytest.mark.parametrize("mutation",["missing_cost","seed_pick","validation","dates","family","band"])
def test_annual_population_dates_and_same_frozen_variant_are_required(mutation):
    frame=rows()
    if mutation=="missing_cost":frame=frame.loc[frame.scenario.eq("base")]
    elif mutation=="seed_pick":frame.loc[0,"seed"]=99
    elif mutation=="validation":frame.loc[0,"split"]="validation"
    elif mutation=="dates":frame.loc[0,"end_exclusive"]="2026-10-22T00:00:00Z"
    elif mutation=="family":frame.loc[0,"family"]="cap075"
    else:frame.loc[0,"rebalance_band"]=.025
    with pytest.raises(ValueError):annual.validate_group_rows(frame)


def test_all_seven_two_cost_controls_are_required_for_performance_comparison():
    with pytest.raises(ValueError):
        annual.performance_evidence(rows(),controls().iloc[:-1],previous_goal())
    assert len(annual.baseline_files())==14*6+4


def test_dynamic_annual_dates_do_not_mutate_frozen_validation_module_or_reuse_its_window(tmp_path):
    case=hand_account(exit_second_day=True)
    dates=(case[2].index[0].isoformat(),(case[2].index[-1]+pd.Timedelta(hours=1)).isoformat())
    old_dates=validation.VALIDATION
    models=[dict(seed=seed,parameter_sha256=str(seed)*64) for seed in annual.SEEDS]
    params=[model["parameter_sha256"] for model in models]
    metadata=deepcopy(case[3])
    metadata.update(algorithm="tqc",profile="balanced",family=annual.FAMILY,model_count=5,exposure_multiplier=.75,
        rebalance_band=.05,cash_floor_target=.05,max_asset_weight_target=.95,drawdown_penalty=.25,
        old_actions_or_equity_scaled=False,band_change_only=True,parameter_sha256_before=params,
        parameter_sha256_after=params,**{flag:True for flag in annual.FLAGS})
    run_id="test__base__cap075_band05__tqc__balanced__ensemble"
    device=dict(device="cuda:0",capability=[7,0],visible_gpu_uuid="GPU-hand",name="Tesla V100-SXM2-16GB")
    replay=dict(run_id=run_id,passed=True,repeat_exact=True,parameter_sha256_before=params,
                parameter_sha256_after=params,model_records=models,device=device)
    serialize_case(tmp_path,case,metadata,replay)
    row={**case[4]["metrics"],"run_id":run_id,"scenario":"base","primary_policy":True}
    checks=annual.Checks()
    run,history,monthly=annual.audit_case(tmp_path,case[0],row,
        {seed:{"parameter_sha256":params[seed]} for seed in annual.SEEDS},
        {"models":models,"gpu_cost_assignments":{"base":"GPU-hand"}},checks,dates=dates,
        features=pd.DataFrame(np.zeros((2,23)),index=pd.date_range(dates[0],dates[1],freq="D",inclusive="left")))
    assert checks.failed==0,checks.failures
    assert run["hours"]==48 and len(history)==3 and set(monthly)=={"2025-01"}
    assert validation.VALIDATION==old_dates


def test_annual_auditor_never_imports_execution_engine_or_loads_policy():
    source=inspect.getsource(annual)
    assert "PortfolioLedger(" not in source and ".predict(" not in source
    assert "from scripts.btceth_rl_env import" not in source
    assert "validation.VALIDATION =" not in source


@pytest.mark.parametrize("mutation",[None,"old_success","seed_count","wrong_audit","candidate"])
def test_original_two_seed_failure_is_kept_as_separate_immutable_evidence(tmp_path,mutation):
    roots=[tmp_path/name for name in ("annual","validation","main","reference","source")]
    for root in roots:root.mkdir()
    original=dict(passed=True,status="passed",test_complete=True,run_count=36)
    write_json(roots[3]/"independent_audit.json",original)
    goal=dict(candidate={"family":"cap075","algorithm":"tqc","profile":"balanced"},
        audit_sha256=annual.sha(roots[3]/"independent_audit.json"),goal_performance_achieved=False,
        metrics={"positive_base_training_seeds":2},gates={"at_least_three_positive_training_seeds":False})
    if mutation=="old_success":goal["goal_performance_achieved"]=True
    elif mutation=="seed_count":goal["metrics"]["positive_base_training_seeds"]=3
    elif mutation=="wrong_audit":goal["audit_sha256"]="changed"
    elif mutation=="candidate":goal["candidate"]["family"]=annual.FAMILY
    write_json(roots[3]/"goal_candidate_evidence.json",goal)
    checks=annual.Checks()
    value=annual.verify_original_goal(annual.Inputs(*roots,checks))
    assert value==goal
    assert checks.failed==0 if mutation is None else checks.failed>0


def test_source_and_result_roots_cannot_alias_original_research(tmp_path):
    roots=[tmp_path/name for name in ("annual","validation","main","reference","source")]
    with pytest.raises(ValueError,match="distinct"):
        annual.Inputs(roots[0],roots[0],*roots[2:],annual.Checks())


def annual_protocol_fixture(tmp_path):
    """Complete provenance-contract fixture; fake model bytes are never loaded."""
    old_roots,dataset=protocol_fixture(tmp_path)
    validation_root,main,reference,source=old_roots
    output=tmp_path/"annual";output.mkdir()
    spec=annual.read_json(main/"frozen_spec.json")
    spec.update(test=list(annual.TEST),research_targets=annual.TARGETS)
    write_json(main/"frozen_spec.json",spec)
    complete=annual.read_json(main/"training_completion.json")
    models=[]
    for seed in annual.SEEDS:
        folder=f"training/tqc__balanced__seed{seed}"
        record=annual.read_json(main/folder/"record.json")
        record["frozen_spec_sha256"]=annual.sha(main/"frozen_spec.json")
        write_json(main/folder/"record.json",record)
        models.append(dict(profile="balanced",seed=seed,record_file=folder+"/record.json",
            model_file=folder+"/model.zip",record_sha256=annual.sha(main/folder/"record.json"),
            model_sha256=record["model_sha256"],parameter_sha256=record["parameter_sha256"]))
        complete["records"]=[record if (r["algorithm"],r["profile"],r["seed"])==("tqc","balanced",seed) else r
                              for r in complete["records"]]
    complete["frozen_spec_sha256"]=annual.sha(main/"frozen_spec.json")
    write_json(main/"training_completion.json",complete)
    vp=annual.read_json(validation_root/"protocol.json");vp["models"]=models
    write_json(validation_root/"protocol.json",vp)
    write_json(validation_root/"protocol_receipt.json",dict(protocol_file="protocol.json",
        protocol_sha256=annual.sha(validation_root/"protocol.json"),sealed_at="2026-10-30T00:00:01Z"))
    for name in annual.SOURCE_FILES:
        path=source/name
        if not path.exists():path.write_text("pinned annual source fixture "+name)
        copied=output/"sources"/name;copied.parent.mkdir(parents=True,exist_ok=True);copied.write_bytes(path.read_bytes())
    for root,names in ((main,annual.baseline_files()),
                       (reference,annual.case_group_files("test")),
                       (validation_root,annual.validation_files())):
        for name in names:
            path=root/name
            if not path.exists():
                path.parent.mkdir(parents=True,exist_ok=True);path.write_text("immutable fixture "+name)
    new,old=group_rows(),group_rows(True)
    new.loc[new.primary_policy,["net_return","final_equity"]]=[.2,12000.]
    new.to_csv(validation_root/"summary.csv",index=False)
    old.to_csv(reference/"validation/balanced/summary.csv",index=False)
    ranked=annual.band.compare_validation_candidates(new,old)
    for row in ranked:row["name"]="band05" if row["family"]==annual.FAMILY else "reference_band025"
    chosen=dict(selected_at="2026-10-30T00:00:08Z",candidate=ranked[0],candidates=ranked,
        selection_uses_test=False,seed_selection=False,**{flag:True for flag in annual.FLAGS})
    write_json(validation_root/"selection.json",chosen)
    write_json(validation_root/"selection_receipt.json",dict(selection_file="selection.json",
        selection_sha256=annual.sha(validation_root/"selection.json"),sealed_at="2026-10-30T00:00:09Z"))
    main_audit=dict(passed=True,status="passed",test_complete=True,run_count=172,failures=[])
    write_json(main/"independent_audit.json",main_audit)
    old_goal=annual.read_json(reference/"goal_candidate_evidence.json")
    old_goal.update(goal_performance_achieved=False,metrics={"positive_base_training_seeds":2},
                    gates={"at_least_three_positive_training_seeds":False})
    write_json(reference/"goal_candidate_evidence.json",old_goal)
    hashes={"extension":{name:annual.sha(validation_root/name) for name in annual.validation_files()},
            "main":{},"reference":{},"source":{name:annual.sha(source/name) for name in annual.SOURCE_FILES[:-2]}}
    prior=dict(passed=True,status="passed",validation_complete=True,selection_complete=True,test_complete=False,
        run_count=12,repeat_count=12,comparator_run_count=12,failure_count=0,failures=[],
        resource_adapter_disclosed=True,audited_at="2026-10-30T00:00:10Z",input_sha256=hashes)
    write_json(validation_root/"independent_audit.selection.json",prior)
    required_main=annual.baseline_files()|{"frozen_spec.json","training_completion.json","runtime.json","independent_audit.json",
        *(model[key] for model in models for key in ("record_file","model_file"))}
    required_ref=annual.case_group_files("test")|{"protocol.json","protocol_receipt.json","independent_audit.json","goal_candidate_evidence.json"}
    for name in ("protocol.json","protocol_receipt.json"):
        if not (reference/name).exists():write_json(reference/name,{})
    required_val=annual.validation_files()|{"independent_audit.selection.json"}
    p=dict(prepared_at="2026-10-30T00:00:12Z",family=annual.FAMILY,algorithm="tqc",profile="balanced",
        test_start=annual.TEST[0],end_exclusive=annual.TEST[1],capital=10000.,costs=annual.COSTS,
        seeds=list(annual.SEEDS),cash_floor=.05,max_asset_weight=.95,drawdown_penalty=.25,
        rebalance_band=.05,exposure_multiplier=.75,new_training=0,seed_selection=False,
        selection_uses_test=False,annual_evaluation_enabled=True,research_targets=annual.TARGETS,
        source_sha256={name:annual.sha(source/name) for name in annual.SOURCE_FILES},
        original_inputs={"main":{name:annual.sha(main/name) for name in required_main},
                         "reference":{name:annual.sha(reference/name) for name in required_ref},
                         "validation":{name:annual.sha(validation_root/name) for name in required_val}},
        validation_audit_file="independent_audit.selection.json",
        validation_selection_sha256=annual.sha(validation_root/"selection.json"),models=models,
        physical_gpus=vp["physical_gpus"],gpu_cost_assignments=vp["gpu_cost_assignments"],
        runtime_packages=spec["runtime_packages"],dataset_sha256=spec["dataset_sha256"],
        dataset_manifest_sha256=spec["dataset_manifest_sha256"],**{flag:True for flag in annual.FLAGS})
    write_json(output/"protocol.json",p)
    write_json(output/"protocol_receipt.json",dict(protocol_file="protocol.json",
        protocol_sha256=annual.sha(output/"protocol.json"),sealed_at="2026-10-30T00:00:13Z"))
    return [output,validation_root,main,reference,source],dataset


def test_annual_protocol_binds_complete_validation_models_sources_and_original_controls(tmp_path):
    roots,dataset=annual_protocol_fixture(tmp_path)
    checks=annual.Checks()
    p,spec,records,old,goal=annual.verify_protocol(annual.Inputs(*roots,checks),dataset)
    assert checks.failed==0,checks.failures
    assert len(records)==5 and len(p["source_sha256"])==14
    assert old["run_count"]==172 and goal["goal_performance_achieved"] is False


@pytest.mark.parametrize("mutation",["lower_gate","source","model","validation_file","partial_audit","no_disclosure","old_winner"])
def test_annual_context_rejects_changed_rules_or_a_weaker_unsealed_validation_path(tmp_path,mutation):
    roots,dataset=annual_protocol_fixture(tmp_path)
    output,validation_root,main,reference,source=roots
    p=annual.read_json(output/"protocol.json")
    if mutation=="lower_gate":p["research_targets"]["primary_base_net_return_at_least"]=.01
    elif mutation=="source":(source/annual.SOURCE_FILES[-2]).write_text("changed annual runner")
    elif mutation=="model":(main/"training/tqc__balanced__seed0/model.zip").write_bytes(b"changed")
    elif mutation=="validation_file":(validation_root/"validation/balanced/base/0/fills.csv").write_text("changed")
    elif mutation=="partial_audit":
        prior=annual.read_json(validation_root/"independent_audit.selection.json")
        prior["input_sha256"]["extension"]={}
        write_json(validation_root/"independent_audit.selection.json",prior)
    elif mutation=="no_disclosure":p["not_fresh_holdout"]=False
    else:
        selected=annual.read_json(validation_root/"selection.json")
        selected["candidate"]=selected["candidates"][1]
        write_json(validation_root/"selection.json",selected)
    write_json(output/"protocol.json",p)
    write_json(output/"protocol_receipt.json",dict(protocol_file="protocol.json",
        protocol_sha256=annual.sha(output/"protocol.json"),sealed_at="2026-10-30T00:00:13Z"))
    checks=annual.Checks()
    with pytest.raises(ValueError):
        annual.verify_protocol(annual.Inputs(*roots,checks),dataset)
    assert checks.failed>0


def execution_fixture(tmp_path):
    roots,dataset=annual_protocol_fixture(tmp_path)
    output,validation_root,main,reference,source=roots
    p=annual.read_json(output/"protocol.json")
    auth=dict(authorized_at="2026-10-30T00:00:11Z",annual_evaluation_authorized=True,
        selection_sha256=annual.sha(validation_root/"selection.json"),financial_parameters_fixed=True,
        no_live_trading=True,**{flag:True for flag in annual.FLAGS})
    write_json(output/"authorization.json",auth)
    p.update(authorization_sha256=annual.sha(output/"authorization.json"),
             allowed_instance=instance(),allowed_gpu_rows=allowed())
    write_json(output/"protocol.json",p)
    write_json(output/"protocol_receipt.json",dict(protocol_file="protocol.json",
        protocol_sha256=annual.sha(output/"protocol.json"),sealed_at="2026-10-30T00:00:13Z"))
    write_json(output/"test_started.json",dict(at="2026-10-30T00:00:15Z",
        protocol_sha256=annual.sha(output/"protocol.json"),authorization_sha256=p["authorization_sha256"],
        validation_selection_sha256=annual.sha(validation_root/"selection.json")))
    (output/"test.log").write_text("synthetic real child terminal log")
    process=dict(phase="test",status="completed",exit_code=0,protocol_sha256=annual.sha(output/"protocol.json"),
        source_sha256=p["source_sha256"]["scripts/btceth_rl_execution_band_annual.py"],
        command=["python","-u","/research/scripts/btceth_rl_execution_band_annual.py","--phase","_test"],
        supervisor_pid=123,child_pid=456,started_at="2026-10-30T00:00:14Z",
        completed_at="2026-10-30T00:00:19Z",log_relative_path="test.log",log_sha256=annual.sha(output/"test.log"))
    write_json(output/"runner_process.json",process)
    write_json(output/"test_completion.json",dict(at="2026-10-30T00:00:18Z",case_count=12,repeat_count=12,
        protocol_sha256=annual.sha(output/"protocol.json")))
    write_json(validation_root/"adapter_protocol.json",dict(allowed_instance=instance(),allowed_gpu_rows=allowed()))
    write_json(output/"resource_runtime.json",dict(status="completed",error=None,financial_rules_changed=False,
        before=snapshot("2026-10-30T00:00:16Z"),after=snapshot("2026-10-30T00:00:17Z")))
    return roots,p


def test_only_separately_authorized_annual_marker_real_process_and_same_reviewed_ui_instance_can_pass(tmp_path):
    roots,p=execution_fixture(tmp_path)
    checks=annual.Checks()
    completion,process=annual.verify_execution_chain(annual.Inputs(*roots,checks),p)
    assert checks.failed==0,checks.failures
    assert completion["case_count"]==12 and process["exit_code"]==0


@pytest.mark.parametrize("mutation",["no_authorization","early_authorization","unsealed_marker",
    "running","failed","wrong_private_phase","changed_log","partial_repeat","other_gpu_pid","reused_ui","financial_change"])
def test_prepared_source_or_validation_success_never_substitutes_for_explicit_annual_authorization(tmp_path,mutation):
    roots,p=execution_fixture(tmp_path)
    output,validation_root,*_=roots
    auth=annual.read_json(output/"authorization.json")
    marker=annual.read_json(output/"test_started.json")
    process=annual.read_json(output/"runner_process.json")
    completion=annual.read_json(output/"test_completion.json")
    runtime=annual.read_json(output/"resource_runtime.json")
    if mutation=="no_authorization":auth["annual_evaluation_authorized"]=False
    elif mutation=="early_authorization":auth["authorized_at"]="2026-10-29T00:00:00Z"
    elif mutation=="unsealed_marker":marker["protocol_sha256"]="wrong"
    elif mutation=="running":process["status"]="running"
    elif mutation=="failed":process["exit_code"]=1
    elif mutation=="wrong_private_phase":process["command"][-1]="_validate"
    elif mutation=="changed_log":(output/"test.log").write_text("changed log")
    elif mutation=="partial_repeat":completion["repeat_count"]=11
    elif mutation=="other_gpu_pid":runtime["before"]["stdout"]+="GPU-fixture1, 99999, /usr/bin/python, 5000 MiB\n"
    elif mutation=="reused_ui":runtime["after"]["instance"]["starttime_ticks"]="1234"
    else:runtime["financial_rules_changed"]=True
    for name,data in (("authorization.json",auth),("test_started.json",marker),
        ("runner_process.json",process),("test_completion.json",completion),("resource_runtime.json",runtime)):
        write_json(output/name,data)
    checks=annual.Checks()
    annual.verify_execution_chain(annual.Inputs(*roots,checks),p)
    assert checks.failed>0

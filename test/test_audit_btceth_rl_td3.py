"""Pure synthetic TD3 audit contracts; no policy run, training or year result."""
from copy import deepcopy
import inspect
import hashlib
import io
import json
from pathlib import Path
import shutil
import zipfile

import numpy as np
import pandas as pd
import pytest

from scripts import audit_btceth_rl_td3 as audit
from test.test_audit_btceth_rl_execution_band import hand_account, serialize_case, write_json
from test.test_audit_btceth_rl_execution_band import protocol_fixture as old_protocol_fixture


def rows(family="raw",profile="balanced",split="validation",ret=.1,dd=-.1):
    dates=audit.VALIDATION if split=="validation" else audit.TEST
    return pd.DataFrame([dict(run_id=f"{split}__{cost}__{family}__td3__{profile}__{key}",
        algorithm="td3",family=family,profile=profile,scenario=cost,split=split,
        seed=None if key=="ensemble" else int(key),primary_policy=key=="ensemble",
        initial_capital=10000.,start=dates[0],end_exclusive=dates[1],net_return=ret,
        max_drawdown=dd,final_equity=10000*(1+ret),fees=1.,rebalance_band=.025,
        exposure_multiplier=audit.FAMILIES[family]) for cost in audit.COSTS
        for key in (*map(str,audit.SEEDS),"ensemble")])


def groups():
    return {(family,profile):rows(family,profile) for family in audit.FAMILIES for profile in audit.PROFILES}


def test_four_fixed_candidates_all_seeds_and_both_costs_tie_raw_then_profile():
    ranked=audit.rank_candidates(groups())
    assert [(r["family"],r["profile"]) for r in ranked]==[
        ("raw","balanced"),("raw","return"),("cap075","balanced"),("cap075","return")]


def test_only_actual_shared_ensemble_controls_ranking_never_best_seed():
    pool=groups()
    frame=pool[("cap075","return")]
    frame.loc[~frame.primary_policy,["net_return","final_equity"]]=[99.,1000000.]
    assert audit.rank_candidates(pool)[0]["family"]=="raw"
    frame.loc[frame.primary_policy,["net_return","final_equity"]]=[.2,12000.]
    assert audit.rank_candidates(pool)[0]["family"]=="cap075"


@pytest.mark.parametrize("mutation",["missing_seed","omit_cost","year_row","band_search","drop_candidate","wrong_algo"])
def test_validation_population_cannot_shrink_or_look_at_year(mutation):
    pool=groups();frame=pool[("raw","return")].copy()
    if mutation=="missing_seed":frame=frame.iloc[1:]
    elif mutation=="omit_cost":frame=frame.loc[frame.scenario.eq("base")]
    elif mutation=="year_row":frame.loc[0,"split"]="test"
    elif mutation=="band_search":frame.loc[0,"rebalance_band"]=.05
    elif mutation=="drop_candidate":pool.pop(("cap075","balanced"))
    else:frame.loc[0,"algorithm"]="tqc"
    pool[("raw","return")]=frame
    with pytest.raises(ValueError):audit.rank_candidates(pool)


def test_original_box_action_cash_flat_region_and_two_declared_budgets():
    raw,mapped=audit.action_weights([1.,1.],"raw")
    np.testing.assert_allclose(raw,[0.,.95,.05])
    np.testing.assert_allclose(mapped,raw)
    raw,mapped=audit.action_weights([1.,1.],"cap075")
    np.testing.assert_allclose(mapped,[0.,.7125,.2875])
    assert audit.action_weights([-.5,1.],"raw")[1].tolist()==[0.,0.,1.]
    with pytest.raises(ValueError):audit.action_weights([1.,0.],"searched")


def case_for_family(family):
    case=list(hand_account(exit_second_day=True))
    trace=case[5].copy()
    if family=="raw":
        trace.loc[trace.raw_action0.gt(0),"raw_action0"]*=.75
    for i,row in trace.iterrows():
        raw,mapped=audit.action_weights([row.raw_action0,row.raw_action1],family)
        for key,value in zip(("raw_btc","raw_eth","raw_cash"),raw):trace.at[i,key]=value
        for key,value in zip(("mapped_btc","mapped_eth","mapped_cash"),mapped):trace.at[i,key]=value
    trace["observation_cutoff_exclusive"]=trace.day
    case[5]=trace
    return case


@pytest.mark.parametrize("family",["raw","cap075"])
def test_hand_funded_account_raw_and_capped_targets_both_reconcile_without_old_global_patch(tmp_path,family):
    case=case_for_family(family)
    dates=(case[2].index[0].isoformat(),(case[2].index[-1]+pd.Timedelta(hours=1)).isoformat())
    params=[str(seed)*64 for seed in audit.SEEDS]
    meta=deepcopy(case[3])
    meta.update(algorithm="td3",profile="balanced",family=family,model_count=5,
        exposure_multiplier=audit.FAMILIES[family],rebalance_band=.025,cash_floor_target=.05,
        max_asset_weight_target=.95,drawdown_penalty=.25,old_actions_or_equity_scaled=False,
        parameter_sha256_before=params,parameter_sha256_after=params,training_budget_multiple=2,
        deterministic_prediction=True,**{flag:True for flag in audit.FLAGS})
    run_id=f"validation__base__{family}__td3__balanced__ensemble"
    device=dict(device="cuda:0",visible_gpu_uuid="GPU-hand",name="Tesla V100-SXM2-16GB",capability=[7,0])
    replay=dict(run_id=run_id,passed=True,repeat_exact=True,parameter_sha256_before=params,
        parameter_sha256_after=params,device=device,model_records=[{"parameter_sha256":value} for value in params])
    serialize_case(tmp_path,case,meta,replay)
    features=pd.DataFrame(np.zeros((2,23)),index=pd.date_range(*dates,freq="D",inclusive="left"))
    checks=audit.Checks()
    result,history,monthly=audit.audit_case(tmp_path,case[0],{**case[4]["metrics"],"run_id":run_id,
        "scenario":"base","primary_policy":True},{("balanced",seed):{"parameter_sha256":params[seed]} for seed in audit.SEEDS},
        checks,split="validation",family=family,profile="balanced",features=features,gpus=[{"uuid":"GPU-hand"}],dates=dates)
    assert checks.failed==0,checks.failures
    assert result["hours"]==48 and len(history)==3 and set(monthly)=={"2025-01"}


def boundaries():
    date=pd.Timestamp("2024-02-02T00:00Z")
    current=np.r_[np.zeros(23),.1,.2,.7,.05].astype("<f4")
    terminal=np.r_[np.zeros(23),.2,.3,.5,.06].astype("<f4")
    episodes=[dict(start="2024-01-01T00:00:00Z",end_exclusive=str(date),
        completed_until=str(date),completed_steps=32,terminated=False,truncated=True) for _ in range(2048)]
    trace=[]
    for i in range(2048):
        row=dict(transition_number=(i+1)*32,replay_position=(i+1)*32-1,env_index=0,
            done=1.,timeout=1.,effective_done=0.,time_limit_truncated=True,info_date=str(date))
        for key,value in (("stored_observation",current),("terminal_observation",terminal),("stored_next_observation",terminal)):
            row[key+"_values"]=value.tolist();row[key+"_sha256"]=audit.observation_digest(value)
        trace.append(row)
    return trace,episodes,pd.DataFrame(np.zeros((2,23)),index=[date-pd.Timedelta(days=1),date])


def test_all2048_actual_timeout_transitions_bootstrap_from_true_terminal_observation():
    trace,episodes,features=boundaries()
    checks=audit.Checks()
    audit.verify_replay_boundaries(trace,episodes,features,checks,"hand")
    assert checks.failed==0,checks.failures


@pytest.mark.parametrize("mutation",["missing","reset_obs","absorbing","false_timeout","future_feature","slot","outside_train"])
def test_timeout_replay_cannot_cross_into_validation_or_use_reset_state(mutation):
    trace,episodes,features=boundaries()
    if mutation=="missing":trace.pop()
    elif mutation=="reset_obs":
        value=np.r_[np.zeros(25),1.,0.].astype("<f4")
        trace[-1]["stored_next_observation_values"]=value.tolist()
        trace[-1]["stored_next_observation_sha256"]=audit.observation_digest(value)
    elif mutation=="absorbing":trace[-1]["effective_done"]=1.
    elif mutation=="false_timeout":trace[-1]["timeout"]=0.
    elif mutation=="future_feature":
        for key in ("stored_next_observation","terminal_observation"):
            trace[-1][key+"_values"][0]=2.
            trace[-1][key+"_sha256"]=audit.observation_digest(trace[-1][key+"_values"])
    elif mutation=="slot":trace[-1]["replay_position"]=0
    else:trace[-1]["info_date"]="2025-01-02T00:00:00Z"
    checks=audit.Checks()
    try:audit.verify_replay_boundaries(trace,episodes,features,checks,"tamper")
    except KeyError:checks.check(False,"outside known training feature dates")
    assert checks.failed>0


def test_official_upstream_five_sources_and_mit_notice_are_real_exact_archived_bytes():
    root=Path(__file__).resolve().parents[1]/"results/backtests/btceth_rl_td3_20261030_preflight/upstream/stable-baselines3"
    if not root.is_dir():
        pytest.skip("Optional independently archived upstream evidence is not installed in this checkout")
    checks=audit.Checks()
    hashes=audit.verify_upstream(root,checks)
    assert checks.failed==0,checks.failures
    assert len(hashes)==7


def test_source_never_imports_or_calls_financial_engine_or_policy():
    source=inspect.getsource(audit)
    assert "PortfolioLedger(" not in source and ".predict(" not in source
    assert "from scripts.btceth_rl_env import" not in source
    assert ".ALGORITHMS =" not in source


def test_actual_zip_tensor_digest_is_independent_of_claimed_record_and_never_constructs_policy(tmp_path):
    import torch
    state={"actor_target.synthetic_weight":torch.tensor([1.,2.],dtype=torch.float32),
           "critic_target.synthetic_weight":torch.tensor([[3.]],dtype=torch.float32)}
    stream=io.BytesIO();torch.save(state,stream)
    path=tmp_path/"tensor_fixture.zip"
    with zipfile.ZipFile(path,"w") as zipped:
        zipped.writestr("policy.pth",stream.getvalue());zipped.writestr("_stable_baselines3_version","2.9.0")
    expected=hashlib.sha256()
    for key,tensor in sorted(state.items()):
        value=tensor.numpy()
        expected.update(key.encode());expected.update(str((value.shape,value.dtype)).encode());expected.update(value.tobytes())
    initialized=torch.cuda.is_initialized()
    assert audit.checkpoint_parameter_digest(path)==expected.hexdigest()
    assert torch.cuda.is_initialized()==initialized


@pytest.mark.parametrize("mutation",["old_sac_or_tqc","wrong_version","duplicated_policy","non_tensor","non_finite"])
def test_old_algorithm_weight_layout_or_ambiguous_zip_cannot_masquerade_as_td3(tmp_path,mutation):
    import torch
    state={"actor_target.weight":torch.tensor([1.]),"critic_target.weight":torch.tensor([2.])}
    if mutation=="old_sac_or_tqc":state.pop("actor_target.weight");state["actor.weight"]=torch.tensor([1.])
    elif mutation=="non_tensor":state["inference_object"]="must never execute"
    elif mutation=="non_finite":state["actor_target.weight"]=torch.tensor([float("nan")])
    stream=io.BytesIO();torch.save(state,stream)
    path=tmp_path/"invalid_tensor_fixture.zip"
    with zipfile.ZipFile(path,"w") as zipped:
        zipped.writestr("policy.pth",stream.getvalue())
        zipped.writestr("_stable_baselines3_version","2.8.0" if mutation=="wrong_version" else "2.9.0")
        if mutation=="duplicated_policy":
            with pytest.warns(UserWarning):zipped.writestr("policy.pth",stream.getvalue())
    with pytest.raises(ValueError):audit.checkpoint_parameter_digest(path)


def test_plain_zip_metadata_counters_are_read_without_decoding_policy_class_pickle(tmp_path):
    record=dict(num_timesteps=65536,_n_updates=65024,policy_delay=2,learning_rate=.0003,learning_starts=512,
        batch_size=128,buffer_size=100000,gradient_steps=1,n_steps=1,gamma=.995,tau=.005,
        target_policy_noise=.2,target_noise_clip=.5,policy_class={":serialized:":"this is deliberately never decoded"})
    path=tmp_path/"plain_metadata.zip"
    with zipfile.ZipFile(path,"w") as zipped:zipped.writestr("data",json.dumps(record))
    assert audit.checkpoint_training_metadata(path)["_n_updates"]==65024
    assert "policy_class" not in audit.checkpoint_training_metadata(path)


def test_actual_official_untrained_cpu_td3_zip_matches_independent_tensor_and_metadata_reader(tmp_path):
    import gymnasium as gym
    import torch
    from stable_baselines3 import TD3
    from scripts import btceth_rl_td3 as producer
    from scripts.backtest_btceth_rl import parameter_digest
    class ShapeOnlyEnv(gym.Env):
        observation_space=gym.spaces.Box(-10.,10.,shape=(27,),dtype=np.float32)
        action_space=gym.spaces.Box(-1.,1.,shape=(2,),dtype=np.float32)
        def reset(self,*,seed=None,options=None):
            super().reset(seed=seed)
            return np.zeros(27,dtype=np.float32),{}
        def step(self,action):
            raise AssertionError("No learning or policy inference is permitted in this fixture")
    initialized=torch.cuda.is_initialized()
    model=TD3("MlpPolicy",ShapeOnlyEnv(),seed=7,device="cpu",**producer.model_parameters(tmp_path/"unused.jsonl"))
    path=tmp_path/"untrained.zip";model.save(path)
    assert model.num_timesteps==model._n_updates==0
    assert audit.checkpoint_parameter_digest(path)==parameter_digest(model.policy)
    metadata=audit.checkpoint_training_metadata(path)
    assert metadata["num_timesteps"]==metadata["_n_updates"]==0
    assert metadata["policy_delay"]==2 and metadata["n_steps"]==1
    assert torch.cuda.is_initialized()==initialized
    model.get_env().close()


def td3_protocol_fixture(tmp_path,monkeypatch):
    from scripts import btceth_rl_td3 as producer
    oldroots,dataset=old_protocol_fixture(tmp_path)
    output,main,_,source=oldroots
    old=audit.read_json(main/"frozen_spec.json")
    old.update(test=list(audit.TEST),timesteps_per_model=32768,research_targets=deepcopy(audit.TARGETS))
    write_json(main/"frozen_spec.json",old)
    for name in audit.SOURCE_FILES:
        path=source/name
        if not path.exists():path.write_text("independently pinned synthetic source "+name)
        target=output/"sources"/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(path.read_bytes())
    for name in audit.baseline_files():
        path=main/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text("immutable original control fixture")
    write_json(main/"independent_audit.json",dict(passed=True,status="passed",test_complete=True,run_count=172,failures=[]))
    upstream={*audit.UPSTREAM_SHA256,"LICENSE","setup.py","verified_manifest.json",
        "github_manifest.json","installed_distribution_observation.json"}
    for name in upstream:
        path=output/"upstream"/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text("hash fixture "+name)
    # The actual upstream verifier has a separate real-byte integration test.
    # This fixture tests declaration and ownership guards without duplicating
    # 110KiB of official source files for every malformed-protocol example.
    monkeypatch.setattr(audit,"verify_upstream",lambda root,checks:{})
    spec=producer.shared_spec()
    spec.update(prepared_at="2026-10-30T00:00:10Z",
        original_frozen_spec_sha256=audit.sha(main/"frozen_spec.json"),
        dataset_sha256=old["dataset_sha256"],dataset_manifest_sha256=old["dataset_manifest_sha256"],
        original_input_sha256={name:audit.sha(main/name) for name in audit.baseline_files()|
            {"frozen_spec.json","runtime.json","independent_audit.json"}},
        source_sha256={name:audit.sha(source/name) for name in audit.SOURCE_FILES},
        physical_gpus=audit.read_json(main/"runtime.json")["physical_gpus"],
        profile_gpu_assignments={"return":"GPU-fixture0","balanced":"GPU-fixture1"},
        runtime_packages=old["runtime_packages"],research_targets=deepcopy(audit.TARGETS),
        sb3_source_sha256=deepcopy(audit.UPSTREAM_SHA256),
        upstream_evidence_sha256={name:audit.sha(output/"upstream"/name) for name in upstream},
        allowed_compute_instance=None,allowed_gpu_rows=[])
    write_json(output/"protocol.json",spec)
    write_json(output/"protocol_receipt.json",dict(protocol_file="protocol.json",
        protocol_sha256=audit.sha(output/"protocol.json"),sealed_at="2026-10-30T00:00:11Z"))
    return output,main,source,dataset,spec


def test_independent_recipe_matches_producer_explicit_contract_and_pins_all_original_controls(tmp_path,monkeypatch):
    output,main,source,dataset,spec=td3_protocol_fixture(tmp_path,monkeypatch)
    checks=audit.Checks()
    actual,old=audit.verify_protocol(audit.Inputs(output,main,source,checks),dataset)
    assert checks.failed==0,checks.failures
    assert actual["hyperparameters"]==audit.RECIPE
    assert actual["model_count"]==10 and actual["validation_cases"]==48
    assert old["research_targets"]==audit.TARGETS


@pytest.mark.parametrize("mutation",["budget","lr_search","bigger_network","wrong_algo","class_source",
    "missing_seed","old_control","unknown_year","lower_target","upstream_changed"])
def test_protocol_refuses_unapproved_search_source_or_weakening_the_original_goal(tmp_path,monkeypatch,mutation):
    output,main,source,dataset,spec=td3_protocol_fixture(tmp_path,monkeypatch)
    if mutation=="budget":spec["timesteps_per_model"]=32768
    elif mutation=="lr_search":spec["hyperparameters"]["learning_rate"]=.001
    elif mutation=="bigger_network":spec["hyperparameters"]["policy_kwargs"]["net_arch"]=[400,300]
    elif mutation=="wrong_algo":spec["algorithm"]="sac"
    elif mutation=="class_source":(source/"scripts/btceth_rl_td3.py").write_text("changed executable")
    elif mutation=="missing_seed":spec["seeds"]=[0,1,2,3]
    elif mutation=="old_control":(main/"evaluation/test/baselines/base/cash/fills.csv").write_text("changed baseline")
    elif mutation=="unknown_year":spec["not_fresh_holdout"]=False
    elif mutation=="lower_target":spec["research_targets"]["positive_base_training_seeds_at_least"]=2
    else:spec["sb3_source_sha256"]["stable_baselines3/common/buffers.py"]="changed library"
    write_json(output/"protocol.json",spec)
    write_json(output/"protocol_receipt.json",dict(protocol_file="protocol.json",
        protocol_sha256=audit.sha(output/"protocol.json"),sealed_at="2026-10-30T00:00:11Z"))
    checks=audit.Checks()
    with pytest.raises(ValueError):audit.verify_protocol(audit.Inputs(output,main,source,checks),dataset)
    assert checks.failed>0


def test_year_audit_without_sealed_selection_fails_before_any_market_or_weight_read(tmp_path,monkeypatch):
    output=tmp_path/"study";main=tmp_path/"main";source=tmp_path/"source"
    for path in (output,main,source):path.mkdir()
    dataset=tmp_path/"dataset";dataset.write_bytes(b"never decoded")
    calls=[]
    monkeypatch.setattr(audit,"verify_protocol",lambda *a:({},{}))
    monkeypatch.setattr(audit,"verify_phase",lambda *a:({"started_at":"2026-10-30T00:00Z",
        "completed_at":"2026-10-30T00:00Z"},{"model_count":10,"case_count":48,"repeat_count":48}))
    monkeypatch.setattr(audit,"load_market",lambda *a:calls.append("forbidden"))
    result=audit.audit(output,main,dataset,phase="test",source_root=source)
    assert result["passed"] is False and result["test_complete"] is False
    assert not calls


def baseline_rows():
    return pd.DataFrame([dict(run_id=f"test__{cost}__{name}",net_return=-.1,max_drawdown=-.4)
                         for cost in audit.COSTS for name in audit.BASELINES])


def test_same_five_gates_and_known_year_disclosure_apply_to_any_selected_td3_budget():
    frame=rows("cap075","balanced","test",ret=.06,dd=-.15)
    frame.loc[frame.scenario.eq("stress"),["net_return","final_equity"]]=[.01,10100.]
    frame.loc[~frame.primary_policy&frame.seed.isin([0,4]),["net_return","final_equity"]]=[-.01,9900.]
    candidate={"algorithm":"td3","family":"cap075","profile":"balanced"}
    result=audit.performance_evidence(frame,baseline_rows(),candidate)
    assert result["goal_performance_achieved"] and result["positive_base_training_seeds"]==3
    assert result["fresh_holdout_success"] is False and result["training_budget_multiple"]==2
    frame.loc[frame.seed.eq(3),["net_return","final_equity"]]=[-.01,9900.]
    result=audit.performance_evidence(frame,baseline_rows(),candidate)
    assert not result["goal_performance_achieved"] and result["positive_base_training_seeds"]==2


def phase_fixture(tmp_path,monkeypatch):
    output,main,source,dataset,spec=td3_protocol_fixture(tmp_path,monkeypatch)
    (output/"fit.log").write_text("actual synthetic child exit0")
    proof=dict(phase="fit",status="completed",exit_code=0,protocol_sha256=audit.sha(output/"protocol.json"),
        source_sha256=spec["source_sha256"]["scripts/btceth_rl_td3.py"],
        command=["python","-u","/research/scripts/btceth_rl_td3.py","--phase","_fit"],
        supervisor_pid=111,child_pid=222,log_relative_path="fit.log",log_sha256=audit.sha(output/"fit.log"),
        started_at="2026-10-30T00:00:13Z",completed_at="2026-10-30T00:00:15Z")
    write_json(output/"fit_process.json",proof)
    write_json(output/"fit_completion.json",dict(at="2026-10-30T00:00:14Z",model_count=10,
        protocol_sha256=audit.sha(output/"protocol.json")))
    write_json(output/"preflight/fit.json",dict(at="2026-10-30T00:00:12Z",
        physical_gpus=spec["physical_gpus"],command=["nvidia-smi","--query-compute-apps=gpu_uuid,pid","--format=csv,noheader"],
        unfiltered_stdout="",allowed_compute_instance=None,allowed_gpu_rows=[]))
    return output,main,source,spec


def test_fit_source_receipt_precedes_real_supervisor_child_and_terminal_evidence(tmp_path,monkeypatch):
    output,main,source,spec=phase_fixture(tmp_path,monkeypatch)
    checks=audit.Checks()
    proof,done=audit.verify_phase(audit.Inputs(output,main,source,checks),"fit",spec)
    assert checks.failed==0,checks.failures
    assert proof["child_pid"]==222 and done["model_count"]==10


@pytest.mark.parametrize("mutation",["live","failed","source","same_pid","wrong_phase","log","before_seal","foreign_gpu"])
def test_terminal_claim_cannot_replace_actual_source_log_time_or_gpu_identity(tmp_path,monkeypatch,mutation):
    output,main,source,spec=phase_fixture(tmp_path,monkeypatch)
    proof=audit.read_json(output/"fit_process.json")
    pre=audit.read_json(output/"preflight/fit.json")
    if mutation=="live":proof["status"]="running"
    elif mutation=="failed":proof["exit_code"]=1
    elif mutation=="source":proof["source_sha256"]="different executable"
    elif mutation=="same_pid":proof["child_pid"]=111
    elif mutation=="wrong_phase":proof["command"][-1]="_test"
    elif mutation=="log":(output/"fit.log").write_text("changed terminal")
    elif mutation=="before_seal":pre["at"]="2026-10-30T00:00:09Z"
    else:pre["unfiltered_stdout"]="GPU-fixture1, 99999\n"
    write_json(output/"fit_process.json",proof);write_json(output/"preflight/fit.json",pre)
    checks=audit.Checks()
    audit.verify_phase(audit.Inputs(output,main,source,checks),"fit",spec)
    assert checks.failed>0


@pytest.mark.parametrize("mutation",[None,"not_authorized","wrong_candidate","early","market_live","fresh_year"])
def test_new_year_authorization_is_separate_and_tied_to_exact_validation_seal(tmp_path,mutation):
    roots=[tmp_path/name for name in ("study","main","source")]
    for root in roots:root.mkdir()
    output=roots[0]
    write_json(output/"protocol.json",{"pinned":"protocol"})
    write_json(output/"selection.json",{"candidate":"already sealed before annual"})
    auth=dict(annual_evaluation_authorized=True,no_live_trading=True,financial_parameters_fixed=True,
        selection_sha256=audit.sha(output/"selection.json"),authorized_at="2026-10-30T00:00:02Z",
        **{flag:True for flag in audit.FLAGS})
    if mutation=="not_authorized":auth["annual_evaluation_authorized"]=False
    elif mutation=="wrong_candidate":auth["selection_sha256"]="other"
    elif mutation=="early":auth["authorized_at"]="2026-10-30T00:00:00Z"
    elif mutation=="market_live":auth["no_live_trading"]=False
    elif mutation=="fresh_year":auth["not_fresh_holdout"]=False
    write_json(output/"annual_authorization.json",auth)
    write_json(output/"test_started.json",dict(at="2026-10-30T00:00:03Z",
        protocol_sha256=audit.sha(output/"protocol.json"),selection_sha256=audit.sha(output/"selection.json"),
        authorization_sha256=audit.sha(output/"annual_authorization.json")))
    checks=audit.Checks()
    audit.verify_annual_authorization(audit.Inputs(*roots,checks),{},{"sealed_at":"2026-10-30T00:00:01Z"})
    assert checks.failed==0 if mutation is None else checks.failed>0


@pytest.mark.parametrize("mutation",[None,"wrong_profile_card","missing_profile","disguised_worker"])
def test_each_profile_uses_its_declared_uuid_and_real_completed_worker(tmp_path,mutation):
    roots=[tmp_path/name for name in ("study","main","source")]
    for root in roots:root.mkdir()
    spec={"profile_gpu_assignments":{"return":"GPU-A","balanced":"GPU-B"}}
    workers=[]
    for i,profile in enumerate(audit.PROFILES):
        device={"visible_gpu_uuid":spec["profile_gpu_assignments"][profile]}
        worker=dict(profile=profile,split="validation",families=list(audit.FAMILIES),device=device,worker_pid=100+i)
        workers.append(worker)
        for family in audit.FAMILIES:
            write_json(roots[0]/f"validation/{family}/{profile}/completion.json",
                dict(worker_pid=worker["worker_pid"],device=device))
    if mutation=="wrong_profile_card":workers[0]["device"]={"visible_gpu_uuid":"GPU-B"}
    elif mutation=="missing_profile":workers.pop()
    elif mutation=="disguised_worker":workers[0]["worker_pid"]=999
    checks=audit.Checks()
    audit.verify_workers(audit.Inputs(*roots,checks),spec,{"workers":workers},split="validation")
    assert checks.failed==0 if mutation is None else checks.failed>0

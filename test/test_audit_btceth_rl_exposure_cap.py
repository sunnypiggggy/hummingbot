"""Synthetic, hand-built cap audit evidence; no trained policy or test-year data."""
from copy import deepcopy
import gzip
import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from scripts import audit_btceth_rl_exposure_cap as audit
from scripts.audit_btceth_rl import Checks, reconstruct_fills


START = pd.Timestamp("2025-01-01T00:00Z")
END = START + pd.Timedelta(days=2)


def group_rows(algorithm="tqc", profile="balanced", family="raw", split="validation", ret=.1, dd=-.1):
    rows = []
    for cost in audit.COSTS:
        for key in (*map(str, audit.SEEDS), "ensemble"):
            prefix = "cap075__" if family == "cap075" else ""
            rows.append(dict(run_id=f"{split}__{cost}__{prefix}{algorithm}__{profile}__{key}",
                algorithm=algorithm, profile=profile, family=family, exposure_multiplier=.75 if family == "cap075" else 1.,
                strategy="tqc_cap075" if family == "cap075" else algorithm,
                scenario=cost, split=split, seed=None if key == "ensemble" else int(key), primary_policy=key == "ensemble",
                initial_capital=10000., start=audit.SPLITS[split][0], end_exclusive=audit.SPLITS[split][1],
                net_return=ret, max_drawdown=dd, final_equity=10000 * (1 + ret), fees=12.))
    return pd.DataFrame(rows)


def pools(ret=.1):
    raw = {(a, p): group_rows(a, p, ret=ret) for a in (*audit.ALGORITHMS, "sac") for p in audit.PROFILES}
    cap = {p: group_rows(profile=p, family="cap075", ret=ret) for p in audit.PROFILES}
    return raw, cap


def test_eight_way_pool_excludes_sac_and_breaks_exact_ties_raw_then_algorithm_profile():
    raw, cap = pools()
    for p in audit.PROFILES:
        raw[("sac", p)] = group_rows("sac", p, ret=100.)
    ranked = audit.candidate_pool(raw, cap)
    assert len(ranked) == 8 and all(r["algorithm"] != "sac" for r in ranked)
    assert [r["family"] for r in ranked] == ["raw"] * 6 + ["cap075"] * 2
    assert (ranked[0]["algorithm"], ranked[0]["profile"]) == ("crossq", "balanced")


def test_material_cap_gain_can_win_but_no_single_seed_can_choose_the_winner():
    raw, cap = pools()
    frame = cap["balanced"]
    frame.loc[~frame.primary_policy, ["net_return", "final_equity"]] = [99., 1000000.]
    first = audit.candidate_pool(raw, cap)
    assert first[0]["family"] == "raw"
    frame.loc[frame.primary_policy, ["net_return", "final_equity"]] = [.2, 12000.]
    assert audit.candidate_pool(raw, cap)[0]["family"] == "cap075"


@pytest.mark.parametrize("mutation", ["missing_seed", "wrong_cost", "test_row", "wrong_multiplier", "coin_family"])
def test_partial_or_wrong_identity_validation_cannot_be_ranked(mutation):
    raw, cap = pools()
    frame = cap["balanced"].copy()
    if mutation == "missing_seed": frame = frame.drop(frame.index[0])
    elif mutation == "wrong_cost": frame.loc[0, "scenario"] = "base_typo"
    elif mutation == "test_row": frame.loc[0, "split"] = "test"
    elif mutation == "wrong_multiplier": frame.loc[0, "exposure_multiplier"] = .8
    else: frame.loc[0, "family"] = "raw"
    cap["balanced"] = frame
    with pytest.raises(ValueError): audit.candidate_pool(raw, cap)


def hand_account(drift=False):
    """Five distinct fixed actions, one funded account; two known buy fills."""
    hours = pd.date_range(START, END, freq="h", inclusive="left")
    opening = pd.DataFrame({"BTCUSDT": 100., "ETHUSDT": 200.}, index=hours)
    closing = opening.copy()
    if drift: closing.loc[START + pd.Timedelta(days=1):] *= 5
    data = {"hourly_" + field: (closing.copy() if field == "close" else opening.copy()) for field in ("open", "high", "low", "close")}
    data["hourly_volume"] = opening * 0 + 100
    target = np.array([.178125, .178125, .64375])
    gross, cost = .35625, 1.0002 * 1.001 - 1
    nav = 10000 / (1 + gross * cost)
    quantities = {"BTCUSDT": target[0] * nav / 100, "ETHUSDT": target[1] * nav / 200}
    cash, fills = 10000., []
    for symbol, price in (("BTCUSDT", 100.), ("ETHUSDT", 200.)):
        quantity, execution = quantities[symbol], price * 1.0002
        notional = quantity * execution
        fee = notional * .001
        cash -= notional + fee
        fills.append(dict(timestamp=START + pd.Timedelta(hours=1), asset_key=symbol, symbol=symbol[:-4], side="BUY",
            qty=quantity, reference_price=price, price=execution, notional=notional, fee=fee,
            slippage_cost=quantity * (execution - price), cash_after=cash, quantity_after=quantity, realized_pnl=0.))
    fills = pd.DataFrame(fills)
    equity = pd.DataFrame(index=hours, columns=["equity", "cash", "base_value", "exposure"], dtype=float)
    equity.iloc[0] = [10000, 10000, 0, 0]
    for hour in hours[1:]:
        base = sum(quantities[s] * closing.at[hour, s] for s in quantities)
        equity.loc[hour] = [cash + base, cash, base, base / (cash + base)]
    final = float(equity.equity.iloc[-1])
    fees = float(fills.fee.sum())
    notional = float(fills.notional.sum())
    base = float(equity.base_value.iloc[-1])
    metadata = dict(initial_capital=10000., final_equity=final, final_cash=cash, fees=fees, total_notional=notional,
        slippage_cost=float(fills.slippage_cost.sum()), realized_pnl=0., unrealized_pnl=base-notional-fees,
        final_liquidation_equity=cash + base * .9998 * .999, final_positions=quantities,
        bnb_fee_discount=False, fee_asset="USDT", fee_rate=.001, slippage=.0002, record_history=True,
        training=False, forced_final_liquidation=False, hourly_high_water=max(10000., final),
        current_drawdown=1-final/max(10000.,final), fill_count=2,
        final_holdings=[dict(asset_key=s,qty=q,mark_price=float(closing.at[hours[-1],s]),
            market_value=q*closing.at[hours[-1],s],weight=q*closing.at[hours[-1],s]/final) for s,q in quantities.items()])
    rebuilt = reconstruct_fills(data, fills, equity, metadata, start=START, end=END, fee=.001, slippage=.0002)
    assert rebuilt["checks"].failed == 0, rebuilt["checks"].failures
    audit.attach_inventory_replay(rebuilt, fills, closing, checks=rebuilt["checks"], label="hand", metadata=metadata)
    actions = [(-1.,0.),(0.,0.),(1.,-1.),(1.,1.),(.5,0.)]
    raw = [(0.,0.,1.),(0.,0.,1.),(.95,0.,.05),(0.,.95,.05),(.2375,.2375,.525)]
    cap = [(0.,0.,1.),(0.,0.,1.),(.7125,0.,.2875),(0.,.7125,.2875),(.178125,.178125,.64375)]
    trace, weights = [], []
    for i, day in enumerate(pd.date_range(START, END, freq="D", inclusive="left")):
        prior = [0.,0.,1.,0.] if i == 0 else [*target,1-nav/10000]
        observation = np.r_[np.zeros(23),prior].astype("<f4")
        digest = hashlib.sha256(observation.tobytes()).hexdigest()
        for seed in audit.SEEDS:
            trace.append(dict(day=day,seed=seed,raw_action0=actions[seed][0],raw_action1=actions[seed][1],
                raw_btc=raw[seed][0],raw_eth=raw[seed][1],raw_cash=raw[seed][2],capped_btc=cap[seed][0],
                capped_eth=cap[seed][1],capped_cash=cap[seed][2],target_btc=target[0],target_eth=target[1],target_cash=target[2],
                observation_sha256=digest,observation_btc_weight=float(observation[-4]),observation_eth_weight=float(observation[-3]),
                observation_cash_weight=float(observation[-2]),observation_drawdown=float(observation[-1])))
        for slot,symbol in enumerate(audit.SYMBOLS):
            close=day+pd.Timedelta(hours=23)
            weights.append(dict(date=day,slot=slot,asset_key=symbol,symbol=symbol[:-4],target_weight=target[slot],
                actual_weight=float(rebuilt["coin_values"].at[close,symbol]/equity.at[close,"equity"]),rebalanced=i==0))
    return data,fills,equity,metadata,rebuilt,pd.DataFrame(trace),pd.DataFrame(weights)


def check_trace(case):
    checks=Checks()
    audit.audit_member_targets(case[5],case[6],case[4],case[1],start=START,end=END,seeds=list(audit.SEEDS),checks=checks,label="hand")
    return checks


def inventory_only(buy_quantity, sell_quantity=None, *, expected_after=None):
    hours=pd.date_range(START,END,freq="h",inclusive="left")
    rows=[dict(timestamp=START+pd.Timedelta(hours=1),asset_key="BTCUSDT",side="BUY",
               qty=buy_quantity,quantity_after=buy_quantity)]
    if sell_quantity is not None:
        remainder=buy_quantity-min(sell_quantity,buy_quantity)
        after=(0. if remainder<=1e-14 else remainder) if expected_after is None else expected_after
        rows.append(dict(timestamp=START+pd.Timedelta(days=1),asset_key="BTCUSDT",side="SELL",
                         qty=sell_quantity,quantity_after=after))
    checks=Checks()
    value=audit.reconstruct_inventory(pd.DataFrame(rows),hours,checks=checks,label="independent inventory hand case")
    return value,checks


@pytest.mark.parametrize("owned,sold",[
    # Actual observed validation fills, represented only as a two-fill hand
    # regression. No source price, policy or test-year result is loaded.
    (.04707155977286824,.047071559772868234),
    (.00019157434411963924,.00019157434411963922),
    (.003340325929201475,.0033403259292014748),
])
def test_real_sell_dust_counterexamples_delete_inventory_and_do_not_force_next_cash_exit(owned,sold):
    assert 0<owned-sold<1e-14
    result,checks=inventory_only(owned,sold)
    assert checks.failed==0,checks.failures
    assert result["quantity"].iloc[-1].eq(0).all()
    assert not result["presence"].iloc[-1].any()
    assert result["sell_deletions"][0]["remainder_before_delete"]==owned-sold
    assert not audit.rebalance_decision(np.array([0.,0.,1.]),np.array([0.,0.,1.]))
    naive=np.array([(owned-sold)*100/10000,0.,1.])
    assert audit.rebalance_decision(np.array([0.,0.,1.]),naive)


def test_legal_tiny_buy_is_retained_and_still_forces_cash_exit():
    result,checks=inventory_only(1e-18)
    assert checks.failed==0 and result["retained_tiny_buy_fills"]==1
    assert result["quantity"].iloc[-1].BTCUSDT==1e-18 and result["presence"].iloc[-1].BTCUSDT
    weight=result["quantity"].iloc[-1].BTCUSDT*100/10000
    assert audit.rebalance_decision(np.array([0.,0.,1.]),np.array([weight,0.,1.]))


def test_partial_sell_with_remainder_above_threshold_retains_inventory_and_protective_exit():
    result,checks=inventory_only(1.,1.-2e-14)
    assert checks.failed==0 and not result["sell_deletions"]
    remainder=1.-(1.-2e-14)
    assert remainder>1e-14 and result["quantity"].iloc[-1].BTCUSDT==remainder
    assert result["presence"].iloc[-1].BTCUSDT
    assert audit.rebalance_decision(np.array([0.,0.,1.]),np.array([remainder*.01,0.,1.]))


@pytest.mark.parametrize("side",["BUY","SELL"])
def test_saved_inventory_zero_presence_cannot_hide_a_retained_tiny_position(side):
    if side=="BUY":
        hours=pd.date_range(START,END,freq="h",inclusive="left")
        fills=pd.DataFrame([dict(timestamp=START+pd.Timedelta(hours=1),asset_key="BTCUSDT",side="BUY",qty=1e-18,quantity_after=0.)])
        checks=Checks();audit.reconstruct_inventory(fills,hours,checks=checks,label="bad tiny buy")
    else:
        _,checks=inventory_only(1.,1.-2e-14,expected_after=0.)
    assert checks.failed and any("zero/present identity" in f for f in checks.failures)


def test_sell_dust_presence_and_quantity_are_independently_bound_to_existing_accounting():
    result,checks=inventory_only(.04707155977286824,.047071559772868234)
    # An additional independent inventory pass never changes the accounting
    # quantity/NAV objects. It only supplies the exact state for policy checks.
    case=hand_account()
    old_quantity=case[4]["quantity"].copy(deep=True)
    old_equity=case[4]["equity"].copy(deep=True)
    evidence=audit.attach_inventory_replay(case[4],case[1],case[0]["hourly_close"],checks=checks,label="hand",metadata=case[3])
    pd.testing.assert_frame_equal(case[4]["quantity"],old_quantity)
    pd.testing.assert_frame_equal(case[4]["equity"],old_equity)
    assert evidence["hours"]==48 and len(evidence["quantity_sha256"])==64 and len(evidence["presence_sha256"])==64
    assert checks.failed==0


def test_hand_five_model_mean_is_one_real_funded_ledger_and_actual_drift_is_allowed():
    for drift in (False,True):
        case=hand_account(drift)
        checks=check_trace(case)
        assert checks.failed==0,checks.failures
        if drift: assert case[6].groupby("date").actual_weight.sum().iloc[-1] > .7125


@pytest.mark.parametrize("mutation,label", [
    ("one_obs","identical common-account"),("borrowed_state","portfolio matches this ledger"),
    ("average_actions","true mean"),("raw_cash","cash residual"),("missing_seed","member"),
    ("swapped_slots","slots"),("extra_trade","no fills"),("no_initial_trade","actually invest")])
def test_target_or_shared_account_lineage_tampering_is_rejected(mutation,label):
    case=list(hand_account())
    if mutation=="one_obs": case[5].loc[4,"observation_sha256"]="a"*64
    elif mutation=="borrowed_state": case[5].loc[5:,"observation_btc_weight"]=.9
    elif mutation=="average_actions": case[5]["target_btc"],case[5]["target_eth"],case[5]["target_cash"] = .106875,.106875,.78625
    elif mutation=="raw_cash": case[5].loc[2,"capped_cash"]=.05
    elif mutation=="missing_seed": case[5]=case[5].drop(4)
    elif mutation=="swapped_slots": case[6]["slot"]=1-case[6]["slot"]
    elif mutation=="extra_trade":
        wrong=case[1].iloc[[0]].copy();wrong["timestamp"]=START+pd.Timedelta(days=1,hours=1)
        case[1]=pd.concat([case[1],wrong],ignore_index=True)
    else: case[1]=case[1].iloc[0:0]
    checks=check_trace(case)
    assert checks.failed and any(label in f for f in checks.failures),checks.failures


def test_cash_paper_policy_is_not_investment_success_even_with_perfect_accounting():
    frame=group_rows(split="test",ret=0.,dd=0.)
    evidence=audit.objective_evidence(frame,{"family":"raw","algorithm":"tqc","profile":"balanced"},
        {"net_return":-.1,"max_drawdown":-.2})
    assert evidence["evaluated"] and evidence["gates"]["base_max_drawdown_at_most_20pct"]
    assert not evidence["goal_performance_achieved"] and not evidence["gates"]["at_least_three_positive_training_seeds"]


def test_final_objective_requires_all_five_seeds_and_both_equal_hold_metrics():
    frame=group_rows(family="cap075",split="test",ret=.1,dd=-.1)
    selected={"family":"cap075","algorithm":"tqc","profile":"balanced"}
    first=audit.objective_evidence(frame,selected,{"net_return":.08,"max_drawdown":-.08})
    assert not first["goal_performance_achieved"]
    good=audit.objective_evidence(frame,selected,{"net_return":.08,"max_drawdown":-.2})
    assert good["goal_performance_achieved"] and len(good["metrics"]["base_seed_returns"])==5
    with pytest.raises(ValueError): audit.objective_evidence(frame.drop(frame.index[0]),selected,{"net_return":.08,"max_drawdown":-.2})
    with pytest.raises(ValueError): audit.objective_evidence(frame,{"family":"raw","algorithm":"sac","profile":"balanced"},{"net_return":0,"max_drawdown":0})


def test_partial_study_never_reports_full_validation_test_or_goal_success(tmp_path,monkeypatch):
    output,main=tmp_path/"extension",tmp_path/"main"
    output.mkdir();main.mkdir()
    monkeypatch.setattr(audit,"load_market",lambda *a,**k: pytest.fail("Missing protocol must not reach market loading"))
    value=audit.audit(output,main,tmp_path/"missing.pkl.gz",allow_validation_only=True)
    assert value["passed"] is False and value["status"]=="failed"
    assert value["validation_complete"] is False and value["test_complete"] is False
    assert value["run_count"]==0 and value["research_targets"]["evaluated"] is False


def test_final_mode_does_not_read_financial_test_data_without_sealed_completion(tmp_path,monkeypatch):
    output,main=tmp_path/"extension",tmp_path/"main"
    output.mkdir();main.mkdir()
    monkeypatch.setattr(audit,"verify_protocol",lambda *a,**k: pytest.fail("Final marker must be checked first"))
    result=audit.audit(output,main,tmp_path/"missing.pkl.gz")
    assert result["passed"] is False and result["test_complete"] is False and result["run_count"]==0
    assert any("closed" in failure for failure in result["failures"])


def test_phase_root_aggregate_can_change_without_invalidating_immutable_selection_files():
    files=audit.extension_validation_files()
    assert len(files)==201
    assert not {"summary.csv","daily_equity.csv","monthly_returns.csv"}&files
    assert {"validation/return/summary.csv","validation/balanced/summary.csv","validation_completion.json"}<=files


def exported_hand_case(path):
    data,fills,equity,metadata,rebuilt,trace,weights=hand_account()
    parameters=[hashlib.sha256(f"original parameter {s}".encode()).hexdigest() for s in audit.SEEDS]
    models=[dict(profile="return",seed=s,record_file=f"training/tqc__return__seed{s}/record.json",
        model_file=f"training/tqc__return__seed{s}/model.zip",record_sha256="a"*64,model_sha256="b"*64,
        parameter_sha256=parameters[s]) for s in audit.SEEDS]
    records={("tqc","return",s):dict(parameter_sha256=parameters[s]) for s in audit.SEEDS}
    protocol={"models":models,"physical_gpus":[{"uuid":"GPU-synthetic-v100-a"},{"uuid":"GPU-synthetic-v100-b"}]}
    metadata.update(algorithm="tqc",profile="return",family="cap075",model_count=5,exposure_multiplier=.75,
        parameter_sha256_before=parameters,parameter_sha256_after=parameters,rebalance_band=.025,cash_floor_target=.05,
        max_asset_weight_target=.95,drawdown_penalty=0.,old_actions_or_equity_scaled=False)
    daily=equity.equity.resample("D").last()
    boundary=pd.DataFrame({"equity":[10000.,*daily.to_list()],
        "reward":[0.,100*np.log(daily.iloc[0]/10000),100*np.log(daily.iloc[1]/daily.iloc[0])]},
        index=pd.date_range(START,END,freq="D"))
    payloads={}
    for name,frame,index in (("equity",equity,True),("fills",fills,False),("weights",weights,False),("boundary_equity",boundary,True)):
        payload=frame.to_csv(index=index,index_label="timestamp" if index else None,float_format="%.17g",lineterminator="\n").encode()
        payloads[name]=payload
        filename="equity_1h.csv.gz" if name=="equity" else name+".csv"
        (path/filename).write_bytes(gzip.compress(payload,mtime=0) if name=="equity" else payload)
    fingerprints={name:hashlib.sha256(value).hexdigest() for name,value in payloads.items()}
    trace_bytes=trace.to_csv(index=False,float_format="%.17g",lineterminator="\n").encode()
    (path/"member_targets.csv").write_bytes(trace_bytes)
    row={**rebuilt["metrics"],"run_id":"validation__base__cap075__tqc__return__ensemble","algorithm":"tqc",
        "profile":"return","scenario":"base","split":"validation","seed":None,"primary_policy":True}
    replay=dict(run_id=row["run_id"],passed=True,repeat_exact=True,fingerprints=fingerprints,repeated_fingerprints=fingerprints,
        member_targets_sha256=hashlib.sha256(trace_bytes).hexdigest(),repeated_member_targets_sha256=hashlib.sha256(trace_bytes).hexdigest(),
        parameter_sha256_before=parameters,parameter_sha256_after=parameters,model_records=models,
        device=dict(device="cuda:0",visible_gpu_uuid="GPU-synthetic-v100-a",name="Synthetic Tesla V100",capability=[7,0]))
    for name,value in (("metadata.json",metadata),("fingerprints.json",fingerprints),("replay.json",replay)):
        (path/name).write_text(json.dumps(value,default=lambda x:x.item())+"\n")
    return data,row,records,protocol


def test_exported_case_connects_member_targets_real_fills_repeat_and_daily_accounting(tmp_path):
    data,row,records,protocol=exported_hand_case(tmp_path)
    checks=Checks()
    run,daily,monthly=audit.audit_case(tmp_path,data,row,dates=(str(START),str(END)),records=records,
        profile="return",key="ensemble",checks=checks,capped=True,protocol=protocol)
    assert checks.failed==0,checks.failures
    assert run["hours"]==48 and run["reconstructed"] and len(daily)==3 and len(monthly)==1
    assert run["fees"]==pytest.approx(row["fees"])


@pytest.mark.parametrize("mutation,label",[("changed_weights","parameters immutable"),("wrong_cost","fill fee"),("fake_repeat","financial replay")])
def test_exported_case_does_not_accept_metadata_or_replay_claims_as_economic_proof(tmp_path,mutation,label):
    data,row,records,protocol=exported_hand_case(tmp_path)
    if mutation=="changed_weights":
        value=json.loads((tmp_path/"metadata.json").read_text());value["parameter_sha256_after"][2]="f"*64
        (tmp_path/"metadata.json").write_text(json.dumps(value))
    elif mutation=="wrong_cost":row["scenario"]="stress"
    else:
        value=json.loads((tmp_path/"replay.json").read_text());value["repeated_fingerprints"]["fills"]="f"*64
        (tmp_path/"replay.json").write_text(json.dumps(value))
    checks=Checks()
    audit.audit_case(tmp_path,data,row,dates=(str(START),str(END)),records=records,
        profile="return",key="ensemble",checks=checks,capped=True,protocol=protocol)
    assert checks.failed and any(label in value for value in checks.failures),checks.failures


def test_input_changed_between_reads_cannot_replace_the_first_evidence_sha(tmp_path):
    for name in ("extension","main","source"): (tmp_path/name).mkdir()
    path=tmp_path/"main/selection.json";path.write_text("first sealed bytes")
    checks=Checks();inputs=audit.Inputs(tmp_path/"extension",tmp_path/"main",tmp_path/"source",checks)
    inputs.capture("main","selection.json")
    original=inputs.hashes["main"]["selection.json"]
    path.write_text("changed while auditing")
    inputs.capture("main","selection.json")
    assert checks.failed==1 and inputs.hashes["main"]["selection.json"]==original
    inputs.unchanged()
    assert checks.failed==2


def old_contract_fixture(tmp_path, ulp=False):
    output,main,source=(tmp_path/name for name in ("extension","main","source"))
    for p in (output,main,source/"scripts"):p.mkdir(parents=True)
    goal_source=source/"scripts/btceth_rl_goal_candidate.py"
    goal_source.write_text("# synthetic frozen goal source; never executed\n")
    put=lambda path,value:path.write_text(json.dumps(value)+"\n")
    spec=dict(frozen_at="2026-01-01T00:00:00Z",code_sha256={"scripts/frozen.py":"a"*64},timesteps_per_model=32768,
        validation_selection={"score":"fixed log/DD score"},research_targets={"targets_are_not_guarantees":True})
    put(main/"frozen_spec.json",spec);spec_sha=audit.sha(main/"frozen_spec.json")
    records=[];training_names=[]
    for a in (*audit.ALGORITHMS,"sac"):
        for p in audit.PROFILES:
            for s in audit.SEEDS:
                folder=f"training/{a}__{p}__seed{s}";(main/folder).mkdir(parents=True)
                model=main/folder/"model.zip";model.write_bytes(f"synthetic weight {a} {p} {s}".encode())
                record=dict(algorithm=a,profile=p,seed=s,model_relative_path=folder+"/model.zip",model_sha256=audit.sha(model),
                    frozen_spec_sha256=spec_sha,actual_timesteps=32768,parameter_sha256=hashlib.sha256(model.read_bytes()).hexdigest())
                put(main/folder/"record.json",record);records.append(record)
                training_names.extend([folder+"/record.json",folder+"/model.zip"])
    put(main/"training_completion.json",dict(model_count=40,records=records,frozen_spec_sha256=spec_sha))
    training_names.append("training_completion.json")
    training_pins={name:audit.sha(main/name) for name in training_names}
    candidates=[];val_pins={}
    for a in (*audit.ALGORITHMS,"sac"):
        for p in audit.PROFILES:
            folder=f"evaluation/validation/{a}/{p}";(main/folder).mkdir(parents=True)
            frame=group_rows(a,p);frame.to_csv(main/folder/"summary.csv",index=False,float_format="%.17g")
            val_pins[folder+"/summary.csv"]=audit.sha(main/folder/"summary.csv")
            row=audit.candidate_row(frame,family="raw",algorithm=a,profile=p);row.pop("family")
            candidates.append(row)
    candidates=sorted(candidates,key=lambda r:(-r["score"],r["algorithm"],r["profile"]))
    leader={k:candidates[0][k] for k in ("algorithm","profile")}
    original=dict(frozen_at="2026-01-03T00:00:00Z",frozen_spec_sha256=spec_sha,selection_uses_test=False,
        test_results_available_to_selector=False,training_source_sha256=training_pins,validation_source_sha256=val_pins,
        selected_profiles={a:"balanced" for a in (*audit.ALGORITHMS,"sac")},primary_validation_leader=leader,candidates=candidates)
    put(main/"validation_selection.json",original)
    put(main/"validation_completion.json",dict(model_groups=8,baselines=7,frozen_spec_sha256=spec_sha))
    source_pins={"scripts/btceth_rl_goal_candidate.py":audit.sha(goal_source)}
    protocol=dict(original_frozen_spec_sha256=spec_sha,original_financial_code_sha256=spec["code_sha256"],
        new_algorithms=list(audit.ALGORITHMS),control_algorithm="sac",seeds=list(audit.SEEDS),selection_score=spec["validation_selection"]["score"],
        research_targets=spec["research_targets"],source_sha256=source_pins,declared_at="2026-01-02T00:00:00Z")
    put(main/"goal_candidate_protocol.json",protocol)
    new=deepcopy([r for r in candidates if r["algorithm"] in audit.ALGORITHMS])
    if ulp:new[0]["score"]=float(np.nextafter(new[0]["score"],np.inf))
    chosen=dict(selected_at="2026-01-04T00:00:00Z",declared_protocol_sha256=audit.sha(main/"goal_candidate_protocol.json"),
        original_validation_selection_sha256=audit.sha(main/"validation_selection.json"),original_frozen_spec_sha256=spec_sha,
        source_sha256=source_pins,seed_selection=False,selection_uses_test=False,new_validation_candidates=new,
        new_model_validation_leader=leader,original_global_validation_leader=leader,candidate=new[0],
        training_source_sha256=training_pins,validation_source_sha256=val_pins)
    put(main/"goal_candidate_selection.json",chosen)
    receipt=dict(selection_file="goal_candidate_selection.json",selection_sha256=audit.sha(main/"goal_candidate_selection.json"),
        declared_protocol_sha256=audit.sha(main/"goal_candidate_protocol.json"),original_validation_selection_sha256=audit.sha(main/"validation_selection.json"),
        sealed_at="2026-01-05T00:00:00Z")
    put(main/"goal_candidate_selection_receipt.json",receipt)
    (main/"summary.csv").write_bytes(b"Unparseable TEST-profit sentinel: this root aggregate must never be read here.")
    checks=Checks();return audit.Inputs(output,main,source,checks),checks


def test_old_contract_is_independent_of_strict_helper_and_accepts_one_ulp_score_drift(tmp_path):
    inputs,checks=old_contract_fixture(tmp_path,ulp=True)
    context=audit.independent_original_context(inputs,require_test=False)
    assert checks.failed==0,checks.failures
    assert context["test_pnl_read"] is False and context["new_model_validation_leader"]["algorithm"]=="crossq"
    assert "summary.csv" not in inputs.hashes["main"]
    assert len([name for name in inputs.hashes["main"] if name.startswith("training/")])==80


def test_original_goal_receipt_tampering_is_not_hidden_by_numeric_tolerance(tmp_path):
    inputs,checks=old_contract_fixture(tmp_path)
    path=inputs.roots["main"]/"goal_candidate_selection_receipt.json"
    value=json.loads(path.read_text());value["selection_sha256"]="f"*64;path.write_text(json.dumps(value))
    audit.independent_original_context(inputs,require_test=False)
    assert checks.failed and any("full-byte selection receipt" in f for f in checks.failures)


def test_late_original_goal_seal_is_rejected_before_test_financial_reads(tmp_path):
    inputs,checks=old_contract_fixture(tmp_path)
    root=inputs.roots["main"]
    (root/"test_started.json").write_text(json.dumps(dict(at="2026-01-04T12:00:00Z",
        validation_selection_sha256=audit.sha(root/"validation_selection.json"),frozen_spec_sha256=audit.sha(root/"frozen_spec.json"))))
    audit.independent_original_context(inputs,require_test=True)
    assert checks.failed and any("postdates both fixed choices" in f for f in checks.failures)
    assert "summary.csv" not in inputs.hashes["main"]

"""Hand arithmetic and synthetic histories; never load a trained policy."""
from copy import deepcopy
import hashlib
import gzip
import inspect
import json

import numpy as np
import pandas as pd
import pytest

from scripts import audit_btceth_rl_execution_band as audit
from scripts.audit_btceth_rl_exposure_cap import reconstruct_inventory

START = pd.Timestamp("2025-01-01T00:00Z")
END = START + pd.Timedelta(days=2)


def market(days=500):
    index = pd.date_range(START - pd.Timedelta(days=days), START + pd.Timedelta(days=3), freq="h", inclusive="left")
    values = np.arange(len(index), dtype=float)
    prices = pd.DataFrame({"BTCUSDT": 100 + values / 500, "ETHUSDT": 200 + values / 600}, index=index)
    data = {"hourly_" + key: prices.copy() for key in ("open", "high", "low", "close")}
    data["hourly_volume"] = prices * 0 + 100
    return data


def hand_account(*, exit_second_day=False):
    """Two known buys and optional complete sells, no execution-engine import."""
    index = pd.date_range(START - pd.Timedelta(hours=1), END, freq="h", inclusive="left")
    prices = pd.DataFrame({"BTCUSDT": 100., "ETHUSDT": 200.}, index=index)
    data = {"hourly_" + key: prices.copy() for key in ("open", "high", "low", "close")}
    data["hourly_volume"] = prices * 0 + 100
    weights = np.array([.178125, .178125, .64375])
    after_buy_nav = 10000 / (1 + .35625 * (1.0002 * 1.001 - 1))
    quantities = {"BTCUSDT": weights[0] * after_buy_nav / 100, "ETHUSDT": weights[1] * after_buy_nav / 200}
    cash, records, basis, current = 10000., [], {}, {}
    for symbol, reference in (("BTCUSDT", 100.), ("ETHUSDT", 200.)):
        quantity = quantities[symbol]
        price = reference * 1.0002
        notional, fee = quantity * price, quantity * price * .001
        cash -= notional + fee
        current[symbol] = quantity
        basis[symbol] = notional + fee
        records.append(dict(timestamp=START + pd.Timedelta(hours=1), asset_key=symbol, symbol=symbol[:-4],
            side="BUY", qty=quantity, reference_price=reference, price=price, notional=notional, fee=fee,
            slippage_cost=quantity * (price-reference), cash_after=cash, quantity_after=quantity, realized_pnl=0.))
    buy_cash, realized = cash, 0.
    if exit_second_day:
        for symbol, reference in (("BTCUSDT", 100.), ("ETHUSDT", 200.)):
            quantity, price = current.pop(symbol), reference * .9998
            notional, fee = quantity * price, quantity * price * .001
            cash += notional-fee
            pnl = notional-fee-basis.pop(symbol)
            realized += pnl
            records.append(dict(timestamp=START + pd.Timedelta(days=1), asset_key=symbol, symbol=symbol[:-4],
                side="SELL", qty=quantity, reference_price=reference, price=price, notional=notional, fee=fee,
                slippage_cost=quantity*(reference-price), cash_after=cash, quantity_after=0., realized_pnl=pnl))
    trades = pd.DataFrame(records)
    hours = pd.date_range(START, END, freq="h", inclusive="left")
    eq = pd.DataFrame(index=hours, columns=["equity", "cash", "base_value", "exposure"], dtype=float)
    base = sum(quantities[s] * prices.at[START,s] for s in quantities)
    for hour in hours:
        if hour == START:
            eq.loc[hour] = [10000.,10000.,0.,0.]
        elif exit_second_day and hour >= START + pd.Timedelta(days=1):
            eq.loc[hour] = [cash,cash,0.,0.]
        else:
            eq.loc[hour] = [buy_cash+base,buy_cash,base,base/(buy_cash+base)]
    final_cash, final = float(eq.cash.iloc[-1]), float(eq.equity.iloc[-1])
    final_base = float(eq.base_value.iloc[-1])
    metadata = dict(initial_capital=10000.,final_equity=final,final_cash=final_cash,
        fees=float(trades.fee.sum()), total_notional=float(trades.notional.sum()),
        slippage_cost=float(trades.slippage_cost.sum()),realized_pnl=realized,
        unrealized_pnl=final_base-sum(basis.values()),final_liquidation_equity=final_cash+final_base*.9998*.999,
        final_positions=current,bnb_fee_discount=False,fee_asset="USDT",fee_rate=.001,slippage=.0002,
        record_history=True,training=False,forced_final_liquidation=False,fill_count=len(trades),
        hourly_high_water=10000.,current_drawdown=1-final/10000.,
        final_holdings=[dict(asset_key=s,qty=q,mark_price=prices.at[hours[-1],s],
            market_value=q*prices.at[hours[-1],s],weight=q*prices.at[hours[-1],s]/final) for s,q in current.items()])
    rebuilt = audit.reconstruct_fills(data,trades,eq,metadata,start=START,end=END,**audit.COSTS["base"])
    assert rebuilt["checks"].failed == 0,rebuilt["checks"].failures
    audit.attach_inventory_replay(rebuilt,trades,data["hourly_close"],checks=rebuilt["checks"],label="hand",metadata=metadata)
    rebuilt["costs"] = audit.COSTS["base"]
    assert rebuilt["checks"].failed == 0,rebuilt["checks"].failures
    actions = [(-1.,0.),(0.,0.),(1.,-1.),(1.,1.),(.5,0.)]
    trace, daily = [], []
    for day in pd.date_range(START,END,freq="D",inclusive="left"):
        old = day > START
        prior = np.r_[weights,1-after_buy_nav/10000] if old else np.array([0.,0.,1.,0.])
        target = np.array([0.,0.,1.]) if old and exit_second_day else weights
        obs = np.r_[np.zeros(23),prior].astype("<f4")
        digest = hashlib.sha256(obs.tobytes()).hexdigest()
        for seed in audit.SEEDS:
            action = (-1.,0.) if old and exit_second_day else actions[seed]
            raw, capped = audit.mapped_target(*action)
            trace.append(dict(day=day,seed=seed,raw_action0=action[0],raw_action1=action[1],
                raw_btc=raw[0],raw_eth=raw[1],raw_cash=raw[2],capped_btc=capped[0],capped_eth=capped[1],capped_cash=capped[2],
                target_btc=target[0],target_eth=target[1],target_cash=target[2],observation_sha256=digest,
                observation_btc_weight=float(obs[-4]),observation_eth_weight=float(obs[-3]),
                observation_cash_weight=float(obs[-2]),observation_drawdown=float(obs[-1]),
                observation_asof=day-pd.Timedelta(hours=1),feature_input_cutoff=day-pd.Timedelta(hours=1),
                observation_values_json=json.dumps(obs.astype(float).tolist(),separators=(",",":"))))
        for i,symbol in enumerate(audit.SYMBOLS):
            closing=day+pd.Timedelta(hours=23)
            daily.append(dict(date=day,slot=i,asset_key=symbol,symbol=symbol[:-4],target_weight=target[i],
                actual_weight=rebuilt["coin_values"].at[closing,symbol]/eq.at[closing,"equity"],
                rebalanced=not old or exit_second_day))
    return data,trades,eq,metadata,rebuilt,pd.DataFrame(trace),pd.DataFrame(daily)


def check_trace(case, **kwargs):
    checks=audit.Checks()
    audit.audit_member_targets(case[5],case[6],case[4],case[1],start=START,end=END,
        seeds=list(audit.SEEDS),checks=checks,label="hand",require_observation_values=True,
        market_features=pd.DataFrame(np.zeros((2,23)),index=pd.date_range(START,END,freq="D",inclusive="left")),**kwargs)
    return checks


def test_hand_buys_fees_slippage_and_true_five_mean_pass_both_independent_replays():
    case=hand_account()
    checks=check_trace(case)
    audit.audit_execution_quantities(case[0],case[5],case[4],case[1],start=START,end=END,
                                    band=.05,checks=checks,label="sizing")
    assert checks.failed==0,checks.failures
    assert len(case[1])==2 and case[4]["metrics"]["fees"]>0


def test_hand_full_cash_exit_uses_sell_leg_and_costs_and_does_not_add_inventory():
    case=hand_account(exit_second_day=True)
    checks=check_trace(case)
    audit.audit_execution_quantities(case[0],case[5],case[4],case[1],start=START,end=END,
                                    band=.05,checks=checks,label="sizing")
    assert checks.failed==0,checks.failures
    assert not case[4]["inventory_presence"].iloc[-1].any()
    assert case[4]["metrics"]["sell_fills"]==2
    assert case[4]["metrics"]["final_cash"]<10000


@pytest.mark.parametrize("change",[.025,.03,.0499999])
def test_new_five_pp_band_skips_changes_that_old_band_trades(change):
    prior=np.array([.2,.2,.6])
    target=prior+np.array([change,0.,-change])
    assert not audit.rebalance_decision(target,prior,.05)
    assert audit.rebalance_decision(target,prior,.025)


def test_exact_band_threshold_and_global_three_weight_maximum():
    assert audit.rebalance_decision([.05,0.,.95],[0.,0.,1.],.05)
    # Neither coin changes5pp, but combined cash changes6pp.
    assert audit.rebalance_decision([.23,.23,.54],[.2,.2,.6],.05)


def test_cash_exit_cannot_be_stuck_by_band_even_for_present_tiny_buy():
    tiny=1e-18
    fill=pd.DataFrame([dict(timestamp=START+pd.Timedelta(hours=1),asset_key="BTCUSDT",
                           side="BUY",qty=tiny,quantity_after=tiny)])
    checks=audit.Checks()
    exact=reconstruct_inventory(fill,pd.date_range(START,END,freq="h",inclusive="left"),checks=checks,label="tinyBUY")
    assert checks.failed==0 and exact["presence"].iloc[-1].BTCUSDT
    assert audit.rebalance_decision([0.,0.,1.],[tiny,0.,1.],.05)


def test_only_sell_dust_is_deleted_and_no_phantom_cash_exit_follows():
    owned,sold=.04707155977286824,.047071559772868234
    fills=pd.DataFrame([
        dict(timestamp=START+pd.Timedelta(hours=1),asset_key="BTCUSDT",side="BUY",qty=owned,quantity_after=owned),
        dict(timestamp=START+pd.Timedelta(days=1),asset_key="BTCUSDT",side="SELL",qty=sold,quantity_after=0.)])
    checks=audit.Checks()
    result=reconstruct_inventory(fills,pd.date_range(START,END,freq="h",inclusive="left"),checks=checks,label="tinySELL")
    assert checks.failed==0 and not result["presence"].iloc[-1].any()
    assert not audit.rebalance_decision([0.,0.,1.],[0.,0.,1.],.05)


@pytest.mark.parametrize("mutation",["fee","cash","qty","sell_unowned","wrong_hour","missing_hour"])
def test_independent_fill_replay_rejects_economic_and_hourly_tampering(mutation):
    data,fills,equity,metadata,*_=hand_account()
    fills,equity=fills.copy(),equity.copy()
    if mutation=="fee": fills.loc[0,"fee"]*=.5
    elif mutation=="cash": fills.loc[0,"cash_after"]+=1
    elif mutation=="qty": fills.loc[0,"quantity_after"]+=.1
    elif mutation=="sell_unowned": fills.loc[0,"side"]="SELL"
    elif mutation=="wrong_hour": fills.loc[0,"timestamp"]=START
    else: equity=equity.drop(equity.index[5])
    checks=audit.Checks()
    audit.reconstruct_fills(data,fills,equity,metadata,start=START,end=END,
                            **audit.COSTS["base"],checks=checks,label="tampered")
    assert checks.failed>0


@pytest.mark.parametrize("mutation",["member_mean","action","own_account","obs_hash","future_cutoff","market_feature","member_missing","duplicate_member","old_band_flag"])
def test_independent_target_audit_rejects_actions_ensemble_state_and_causal_tampering(mutation):
    case=list(hand_account())
    trace,weights=case[5].copy(),case[6].copy()
    if mutation=="member_mean": trace.loc[0,"target_btc"]+=.01
    elif mutation=="action": trace.loc[0,"raw_action0"]=1
    elif mutation=="own_account": trace.loc[0,"observation_cash_weight"]=.4
    elif mutation=="obs_hash": trace.loc[0,"observation_sha256"]="a"*64
    elif mutation=="future_cutoff": trace.loc[0,"feature_input_cutoff"]=START
    elif mutation=="market_feature":
        values=json.loads(trace.loc[0,"observation_values_json"]);values[0]=1
        trace.loc[0,"observation_values_json"]=json.dumps(values)
    elif mutation=="member_missing": trace=trace.iloc[1:]
    elif mutation=="duplicate_member": trace.loc[0,"seed"]=1
    else: weights.loc[weights.date.gt(START),"rebalanced"]=True
    case[5],case[6]=trace,weights
    checks=check_trace(case)
    assert checks.failed>0


def test_internally_consistent_fills_still_must_match_independent_target_sizing():
    case=hand_account()
    trace=case[5].copy()
    trace[["target_btc","target_eth","target_cash"]]=[.25,.25,.5]
    checks=audit.Checks()
    audit.audit_execution_quantities(case[0],trace,case[4],case[1],start=START,end=END,
                                    band=.05,checks=checks,label="sizing")
    assert checks.failed>0


def test_market_features_are_invariant_to_truncating_future_and_mutating_current_day():
    data=market()
    full=audit.independent_market_features(data,cutoff=END)
    truncated={key:frame.loc[frame.index<START].copy() for key,frame in data.items()}
    past=audit.independent_market_features(truncated,cutoff=START)
    np.testing.assert_equal(full.loc[START].to_numpy(),past.loc[START].to_numpy())
    for frame in data.values():
        frame.loc[frame.index>=START]*=999
    changed=audit.independent_market_features(data,cutoff=END)
    np.testing.assert_equal(full.loc[START].to_numpy(),changed.loc[START].to_numpy())


def test_intraday_outage_keeps_real_close_but_volume_requires_full24_and_recovers30days():
    data=market()
    day=START-pd.Timedelta(days=90)
    original=audit.independent_market_features(data,cutoff=END)
    for frame in data.values():
        frame.loc[day+pd.Timedelta(hours=13),"BTCUSDT"]=np.nan
    changed=audit.independent_market_features(data,cutoff=END)
    # 90days later, SMA200 still uses the actual close23 and remains finite.
    assert np.isfinite(changed.loc[START]).all()
    np.testing.assert_equal(changed.loc[START].to_numpy(),original.loc[START].to_numpy())
    assert pd.isna(changed.loc[day+pd.Timedelta(days=1),9])
    assert pd.isna(changed.loc[day+pd.Timedelta(days=30),9])
    assert np.isfinite(changed.loc[day+pd.Timedelta(days=31),9])


def test_missing_close23_is_not_replaced_by_close22_and_needs200_days():
    data=market()
    day=START-pd.Timedelta(days=210)
    data["hourly_close"].loc[day+pd.Timedelta(hours=23),"BTCUSDT"]=np.nan
    features=audit.independent_market_features(data,cutoff=END)
    assert pd.isna(features.loc[day+pd.Timedelta(days=1),7])
    assert pd.isna(features.loc[day+pd.Timedelta(days=200),7])
    assert np.isfinite(features.loc[day+pd.Timedelta(days=201),7])


def test_two_coin_or_field_misalignment_never_gets_filled():
    data=market()
    data["hourly_volume"]=data["hourly_volume"].iloc[1:]
    with pytest.raises(ValueError,match="aligned"):
        audit.independent_market_features(data,cutoff=END)


def test_monotone_funding_solution_matches_hand_cost_equations():
    initial,gross,buycost=10000.,.4,1.0002*1.001-1
    after=audit.funding_nav(initial,buycost,[0.,0.],[.2,.2],buying=True)
    assert after==pytest.approx(initial/(1+gross*buycost),abs=1e-10)
    haircut=1-.9998*.999
    sold=audit.funding_nav(initial,haircut,[4000.,0.],[0.,0.],buying=False)
    assert sold==pytest.approx(initial-4000*haircut,abs=1e-10)


def group_rows(reference=False):
    family,strategy=("cap075","tqc_cap075") if reference else (audit.FAMILY,audit.STRATEGY)
    return pd.DataFrame([dict(run_id=f"validation__{cost}__{family}__tqc__balanced__{key}",
        algorithm="tqc",profile="balanced",family=family,strategy=strategy,scenario=cost,split="validation",
        seed=None if key=="ensemble" else int(key),primary_policy=key=="ensemble",initial_capital=10000.,
        start=audit.VALIDATION[0],end_exclusive=audit.VALIDATION[1],net_return=.1,max_drawdown=-.1,
        final_equity=11000.,fees=1.,exposure_multiplier=.75,rebalance_band=.025 if reference else .05) for cost in audit.COSTS
        for key in (*map(str,audit.SEEDS),"ensemble")])


def test_only_twelve_validation_cases_with_original_five_seeds_can_pass():
    audit.validate_group_rows(group_rows())
    audit.validate_group_rows(group_rows(True),reference=True)
    for key,value in (("split","test"),("profile","return"),("exposure_multiplier",.7),("seed",99)):
        frame=group_rows();frame.loc[0,key]=value
        with pytest.raises(ValueError):
            audit.validate_group_rows(frame)
    with pytest.raises(ValueError):
        audit.validate_group_rows(group_rows().iloc[1:])


def test_unchanged_score_exact_tie_keeps_old_and_single_seed_cannot_choose_winner():
    new,old=group_rows(),group_rows(True)
    assert audit.compare_validation_candidates(new,old)[0]["family"]=="cap075"
    new.loc[~new.primary_policy,["net_return","final_equity"]]=[99.,1000000.]
    assert audit.compare_validation_candidates(new,old)[0]["family"]=="cap075"
    new.loc[new.primary_policy,["net_return","final_equity"]]=[.2,12000.]
    assert audit.compare_validation_candidates(new,old)[0]["family"]==audit.FAMILY


def write_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2),encoding="utf-8")


def protocol_fixture(tmp_path):
    """Pinned fake byte identities only: no real prices or model are loaded."""
    roots=[tmp_path/name for name in ("new","main","ref","source")]
    for root in roots:root.mkdir()
    output,main,reference,source=roots
    sources={}
    for name in audit.SOURCE_FILES:
        path=source/name;path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text("frozen byte fixture "+name)
        sources[name]=audit.sha(path)
    data=tmp_path/"data";data.mkdir()
    (data/"dataset.pkl.gz").write_bytes(b"not a model, not real prices")
    (data/"manifest.json").write_text("{}")
    spec=dict(train=["2020-01-01T00:00:00Z",audit.VALIDATION[0]],validation=list(audit.VALIDATION),
        code_sha256={name:sources[name] for name in audit.FINANCIAL},
        dataset_sha256=audit.sha(data/"dataset.pkl.gz"),dataset_manifest_sha256=audit.sha(data/"manifest.json"),
        validation_selection={"score":"same frozen score"},runtime_packages={"stable-baselines3":"2.9.0"})
    write_json(main/"frozen_spec.json",spec)
    specsha=audit.sha(main/"frozen_spec.json")
    gpus=[dict(uuid="GPU-fixture"+str(i),name="Tesla V100-SXM2-16GB") for i in range(2)]
    write_json(main/"runtime.json",dict(physical_gpus=gpus))
    all_records=[]
    models=[]
    for algorithm in ("recurrent_ppo","tqc","crossq","sac"):
        for profile in ("return","balanced"):
            for seed in audit.SEEDS:
                row=dict(algorithm=algorithm,profile=profile,seed=seed)
                if algorithm=="tqc" and profile=="balanced":
                    folder=f"training/tqc__balanced__seed{seed}"
                    path=main/folder;path.mkdir(parents=True)
                    (path/"model.zip").write_bytes(("fake immutable policy"+str(seed)).encode())
                    row.update(model_relative_path=folder+"/model.zip",model_sha256=audit.sha(path/"model.zip"),
                        parameter_sha256=str(seed)*64,actual_timesteps=32768,frozen_spec_sha256=specsha)
                    write_json(path/"record.json",row)
                    models.append(dict(profile=profile,seed=seed,record_file=folder+"/record.json",
                        model_file=folder+"/model.zip",record_sha256=audit.sha(path/"record.json"),
                        model_sha256=row["model_sha256"],parameter_sha256=row["parameter_sha256"]))
                all_records.append(row)
    write_json(main/"training_completion.json",dict(model_count=40,records=all_records,frozen_spec_sha256=specsha))
    for name in audit.reference_files():
        path=reference/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text("immutable reference fixture")
    write_json(reference/"independent_audit.json",dict(passed=True,status="passed",test_complete=True,
        failures=[],run_count=36,audited_at="2026-10-29T00:00:00Z"))
    write_json(reference/"goal_candidate_evidence.json",{
        "audit_sha256":audit.sha(reference/"independent_audit.json"),
        "candidate":{"family":"cap075","algorithm":"tqc","profile":"balanced"}})
    mainnames={"frozen_spec.json","training_completion.json",
        *(f"training/tqc__balanced__seed{s}/{name}" for s in audit.SEEDS for name in ("record.json","model.zip"))}
    refnames=audit.reference_files()|{"independent_audit.json","goal_candidate_evidence.json"}
    p=dict(algorithm="tqc",profile="balanced",capital=10000.,costs=audit.COSTS,seeds=list(audit.SEEDS),
        cash_floor=.05,max_asset_weight=.95,drawdown_penalty=.25,new_rebalance_band=.05,reference_rebalance_band=.025,
        exposure_multiplier=.75,new_training=0,seed_selection=False,selection_uses_test=False,
        annual_evaluation_enabled=False,validation_start=audit.VALIDATION[0],end_exclusive=audit.VALIDATION[1],
        prepared_at="2026-10-30T00:00:00Z",source_sha256=sources,models=models,physical_gpus=gpus,
        runtime_packages=spec["runtime_packages"],gpu_cost_assignments={cost:gpu["uuid"] for cost,gpu in zip(audit.COSTS,gpus)},
        original_inputs={"main":{name:audit.sha(main/name) for name in mainnames},
                         "reference":{name:audit.sha(reference/name) for name in refnames}},
        dataset_sha256=spec["dataset_sha256"],dataset_manifest_sha256=spec["dataset_manifest_sha256"],
        selection_score="same frozen score",tie_break="prefer_reference",
        **{flag:True for flag in audit.FLAGS})
    write_json(output/"protocol.json",p)
    write_json(output/"protocol_receipt.json",dict(protocol_file="protocol.json",
        protocol_sha256=audit.sha(output/"protocol.json"),sealed_at="2026-10-30T00:00:01Z"))
    return roots,data/"dataset.pkl.gz"


def test_source_model_reference_dataset_and_seen_annual_identity_all_verified(tmp_path):
    roots,dataset=protocol_fixture(tmp_path)
    checks=audit.Checks()
    inputs=audit.Inputs(*roots,checks)
    p,spec,records,previous=audit.verify_protocol(inputs,dataset)
    assert checks.failed==0,checks.failures
    assert len(records)==5 and len(p["models"])==5 and previous["run_count"]==36
    assert len(inputs.hashes["source"])==10


def serialize_case(directory,case,metadata,replay):
    _,fills,equity,_,rebuilt,trace,weights=case
    daily=rebuilt["equity"].equity.resample("D").last()
    rewards=[0.]
    previous_nav,previous_dd=10000.,0.
    for _,hours in rebuilt["equity"].groupby(rebuilt["equity"].index.normalize()):
        deepest=-float(rebuilt["drawdown"].loc[hours.index].min())
        final=float(hours.equity.iloc[-1])
        rewards.append(100*(np.log(final/previous_nav)-.25*max(deepest-previous_dd,0.)))
        previous_nav,previous_dd=final,-float(rebuilt["drawdown"].loc[hours.index[-1]])
    boundary=pd.DataFrame({"equity":[10000.,*daily.tolist()],"reward":rewards},
        index=pd.DatetimeIndex([START,*list(daily.index+pd.Timedelta(days=1))]))
    payloads={}
    for name,frame in (("equity",equity),("fills",fills),("weights",weights),("boundary_equity",boundary)):
        data=frame.to_csv(index=name in ("equity","boundary_equity"),
            index_label="timestamp" if name in ("equity","boundary_equity") else None,
            float_format="%.17g",lineterminator="\n").encode()
        payloads[name]=hashlib.sha256(data).hexdigest()
        (directory/("equity_1h.csv.gz" if name=="equity" else name+".csv")).write_bytes(
            gzip.compress(data,mtime=0) if name=="equity" else data)
    trace.to_csv(directory/"member_targets.csv",index=False,float_format="%.17g")
    replay.update(fingerprints=payloads,repeated_fingerprints=payloads,
        member_targets_sha256=audit.sha(directory/"member_targets.csv"),
        repeated_member_targets_sha256=audit.sha(directory/"member_targets.csv"))
    write_json(directory/"metadata.json",metadata)
    write_json(directory/"fingerprints.json",payloads)
    write_json(directory/"replay.json",replay)


def test_complete_serialized_case_reconciles_fingerprints_boundary_monthly_targets_and_repeats(tmp_path,monkeypatch):
    monkeypatch.setattr(audit,"VALIDATION",(START.isoformat(),END.isoformat()))
    case=hand_account(exit_second_day=True)
    models=[dict(seed=seed,parameter_sha256=str(seed)*64) for seed in audit.SEEDS]
    params=[model["parameter_sha256"] for model in models]
    metadata=deepcopy(case[3])
    metadata.update(algorithm="tqc",profile="balanced",family=audit.FAMILY,model_count=5,
        exposure_multiplier=.75,rebalance_band=.05,cash_floor_target=.05,max_asset_weight_target=.95,
        drawdown_penalty=.25,old_actions_or_equity_scaled=False,band_change_only=True,
        parameter_sha256_before=params,parameter_sha256_after=params,**{key:True for key in audit.FLAGS})
    run_id="validation__base__cap075_band05__tqc__balanced__ensemble"
    device=dict(device="cuda:0",visible_gpu_uuid="GPU-hand",name="Tesla V100-SXM2-16GB",capability=[7,0])
    replay=dict(run_id=run_id,passed=True,repeat_exact=True,parameter_sha256_before=params,
                parameter_sha256_after=params,model_records=models,device=device)
    serialize_case(tmp_path,case,metadata,replay)
    row={**case[4]["metrics"],"run_id":run_id,"scenario":"base","primary_policy":True}
    checks=audit.Checks()
    result,history,monthly=audit.audit_case(tmp_path,case[0],row,
        {seed:{"parameter_sha256":params[seed]} for seed in audit.SEEDS},
        {"models":models,"physical_gpus":[{"uuid":"GPU-hand"}]},checks,
        features=pd.DataFrame(np.zeros((2,23)),index=pd.date_range(START,END,freq="D",inclusive="left")))
    assert checks.failed==0,checks.failures
    assert result["hours"]==48 and result["repeat_exact"] and len(history)==3
    assert set(monthly)=={"2025-01"}
    # Any post-serialization mutation is detected, even when another object
    # still supplies a plausible final return.
    (tmp_path/"member_targets.csv").write_text((tmp_path/"member_targets.csv").read_text()+"\n")
    checks=audit.Checks()
    audit.audit_case(tmp_path,case[0],row,{seed:{"parameter_sha256":params[seed]} for seed in audit.SEEDS},
        {"models":models,"physical_gpus":[{"uuid":"GPU-hand"}]},checks,
        features=pd.DataFrame(np.zeros((2,23)),index=pd.date_range(START,END,freq="D",inclusive="left")))
    assert checks.failed>0


@pytest.mark.parametrize("mutation",["source","model","reference","data","missing_model_pin","band","annual_disclosure","training_completion"])
def test_protocol_rejects_changed_bytes_partial_population_or_rewritten_context(tmp_path,mutation):
    roots,dataset=protocol_fixture(tmp_path)
    output,main,reference,source=roots
    p=audit.read_json(output/"protocol.json")
    if mutation=="source":(source/audit.SOURCE_FILES[0]).write_text("changed original")
    elif mutation=="model":(main/"training/tqc__balanced__seed0/model.zip").write_bytes(b"changed checkpoint")
    elif mutation=="reference":(reference/"validation/balanced/base/0/fills.csv").write_text("changed reference")
    elif mutation=="data":dataset.write_bytes(b"changed data")
    elif mutation=="missing_model_pin":del p["original_inputs"]["main"]["training/tqc__balanced__seed0/model.zip"]
    elif mutation=="band":p["new_rebalance_band"]=.075
    elif mutation=="annual_disclosure":p["not_fresh_holdout"]=False
    else:
        completion=audit.read_json(main/"training_completion.json")
        completion["records"]=completion["records"][:-1]
        write_json(main/"training_completion.json",completion)
    write_json(output/"protocol.json",p)
    write_json(output/"protocol_receipt.json",dict(protocol_file="protocol.json",
        protocol_sha256=audit.sha(output/"protocol.json"),sealed_at="2026-10-30T00:00:01Z"))
    checks=audit.Checks()
    inputs=audit.Inputs(*roots,checks)
    with pytest.raises(ValueError):
        audit.verify_protocol(inputs,dataset)
    assert checks.failed>0


def test_training_fragments_cannot_run_into_validation_or_cross_market_gaps(monkeypatch):
    eligible={pd.Timestamp("2024-01-01T00:00Z")}
    monkeypatch.setattr(audit,"independent_training_calendar",
        lambda *args,**kwargs:{"eligible":eligible,"covered_days":32,"episode_days":32})
    spec={"train":["2020-01-01T00:00:00Z",audit.VALIDATION[0]]}
    episode=dict(start="2024-01-01T00:00:00Z",end_exclusive="2024-02-02T00:00:00Z",
                 completed_until="2024-02-02T00:00:00Z",completed_steps=32,terminated=False,truncated=True)
    record=dict(actual_timesteps=32768,training_episode_history=[deepcopy(episode) for _ in range(1024)])
    checks=audit.Checks()
    audit.verify_training_cutoff({0:record},spec,{},checks)
    assert checks.failed==0
    for first,last in (("2025-01-01T00:00Z","2025-02-02T00:00Z"),("2024-01-02T00:00Z","2024-02-03T00:00Z")):
        changed=deepcopy(record)
        changed["training_episode_history"][0].update(start=first,end_exclusive=last,completed_until=last)
        checks=audit.Checks()
        audit.verify_training_cutoff({0:changed},spec,{},checks)
        assert checks.failed>0


@pytest.mark.parametrize("mutation",[None,"start_before_seal","source","log","child_command","same_pid"])
def test_real_supervisor_handle_log_hash_and_chronology_are_required(tmp_path,mutation):
    roots,dataset=protocol_fixture(tmp_path)
    output=roots[0]
    protocol=audit.read_json(output/"protocol.json")
    (output/"validation.log").write_text("actual synthetic child terminal log")
    process=dict(phase="validate",protocol_sha256=audit.sha(output/"protocol.json"),
        source_sha256=protocol["source_sha256"]["scripts/btceth_rl_execution_band.py"],
        command=["python","-u","/research/scripts/btceth_rl_execution_band.py","--phase","_validate"],
        supervisor_pid=111,child_pid=222,log_relative_path="validation.log",log_sha256=audit.sha(output/"validation.log"),
        started_at="2026-10-30T00:00:02Z",completed_at="2026-10-30T00:00:04Z")
    completion={"at":"2026-10-30T00:00:03Z"}
    if mutation=="start_before_seal":process["started_at"]="2026-10-29T00:00:00Z"
    elif mutation=="source":process["source_sha256"]="unfrozen"
    elif mutation=="log":(output/"validation.log").write_text("overwritten log")
    elif mutation=="child_command":process["command"][-1]="test"
    elif mutation=="same_pid":process["child_pid"]=111
    checks=audit.Checks()
    inputs=audit.Inputs(*roots,checks)
    audit.verify_process_chain(inputs,protocol,process,completion)
    assert checks.failed==0 if mutation is None else checks.failed>0


@pytest.mark.parametrize("mutation",[None,"new_tie","alter_score","annual_hidden","changed_audit","wrong_input_sha"])
def test_post_accounting_selection_is_recomputed_and_bound_to_previous_immutable_audit(tmp_path,mutation):
    roots,dataset=protocol_fixture(tmp_path)
    output=roots[0]
    new,old=group_rows(),group_rows(True)
    pool=audit.compare_validation_candidates(new,old)
    for row in pool:
        row["name"]="reference_band025" if row["family"]=="cap075" else "band05"
    audited=dict(passed=True,status="passed",run_count=12,comparator_run_count=12,test_complete=False,
        audited_at="2026-10-30T00:00:05Z",input_sha256={key:{} for key in ("extension","main","reference","source")})
    write_json(output/"independent_audit.validation.json",audited)
    selected=dict(selected_at="2026-10-30T00:00:06Z",protocol_sha256=audit.sha(output/"protocol.json"),
        audit_sha256=audit.sha(output/"independent_audit.validation.json"),candidates=pool,candidate=pool[0],
        selection_uses_test=False,seed_selection=False,validation_input_sha256=audited["input_sha256"],
        **{key:True for key in audit.FLAGS})
    if mutation=="new_tie":
        selected["candidates"]=list(reversed(selected["candidates"]))
        selected["candidate"]=selected["candidates"][0]
    elif mutation=="alter_score":selected["candidates"][0]["score"]+=.1
    elif mutation=="annual_hidden":selected["not_fresh_holdout"]=False
    elif mutation=="changed_audit":
        audited["audited_at"]="2026-10-30T00:00:08Z"
        write_json(output/"independent_audit.validation.json",audited)
    elif mutation=="wrong_input_sha":selected["validation_input_sha256"]={}
    write_json(output/"comparison.json",dict(candidates=selected["candidates"],
        chosen=selected["candidate"]["name"],tie_break="prefer_reference"))
    write_json(output/"selection.json",selected)
    write_json(output/"selection_receipt.json",dict(selection_file="selection.json",
        selection_sha256=audit.sha(output/"selection.json"),protocol_sha256=audit.sha(output/"protocol.json"),
        sealed_at="2026-10-30T00:00:07Z"))
    checks=audit.Checks()
    assert audit.verify_selection_if_present(audit.Inputs(*roots,checks),new,old)
    assert checks.failed==0 if mutation is None else checks.failed>0


@pytest.mark.parametrize("mutation",["live","exit1","annual","count","protocol"])
def test_terminal_guard_rejects_live_failed_annual_or_incomplete_outputs(tmp_path,mutation):
    process={"status":"completed","exit_code":0}
    completion={"case_count":12,"repeat_count":12,"protocol_sha256":"pinned"}
    if mutation=="live":process["status"]="running"
    elif mutation=="exit1":process["exit_code"]=1
    elif mutation=="annual":(tmp_path/"test_started.json").write_text("{}")
    elif mutation=="count":completion["repeat_count"]=11
    else:completion["protocol_sha256"]="wrong"
    (tmp_path/"runner_process.json").write_text(json.dumps(process))
    (tmp_path/"validation_completion.json").write_text(json.dumps(completion))
    checks=audit.Checks()
    audit.verify_terminal(tmp_path,"pinned",checks)
    assert checks.failed>0


def test_input_paths_hashes_and_temporal_no_execution_engine_imports(tmp_path):
    checks=audit.Checks()
    roots=[tmp_path/name for name in ("new","main","ref","source")]
    for root in roots:root.mkdir()
    (roots[0]/"file").write_text("immutable")
    inputs=audit.Inputs(*roots,checks)
    inputs.capture("extension","file")
    (roots[0]/"file").write_text("changed")
    inputs.unchanged()
    assert checks.failed==1
    for value in ("../secret","C:/secret","/absolute"):
        with pytest.raises(ValueError):
            inputs.capture("extension",value)
    source=inspect.getsource(audit)
    assert "from scripts.btceth_rl_env import" not in source
    assert "PortfolioLedger(" not in source
    assert ".predict(" not in source
    assert all(flag in source for flag in audit.FLAGS)

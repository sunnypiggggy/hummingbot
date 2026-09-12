"""Isolated counter and full cash-ledger regression scenarios; no OCI clients."""
import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import xgboost_long_risk_gate_v22 as v22
from grid_v22_recovery_ablation import CounterState, rollover, step
from backtest_grid_v22_recovery_counters_2026 import Book, build_orders, can_reenter, replay_grid, validate_frame


def active():
    return CounterState(v22.GateState(active=True, since=0, last_signal_ts=44*3600,
                                      last_complete_4h_ts=44*3600, previous_structure=(5.,5.,1.,1.,.2)))


def call(s, arm, hour, structure, *, structure_hour=None, fold=1, boundary=0):
    return step(s, arm=arm, pair="BTC-FDUSD", probability=.01, threshold=.1, ts=hour*3600,
                structure_ts=(hour if structure_hour is None else structure_hour)*3600,
                structure=structure, fold=fold, boundary=boundary)


def test_strong_independent_while_ordinary_fails():
    a,b=active(),active()
    for hour, v in ((48,4.),(52,3.)):
        ar=call(a,"A",hour,(v,v,1,1,.2))
        br=call(b,"B",hour,(v,v,1,1,.2))
    assert ar["risk_off"] and ar["ordinary_count"]==0
    assert not br["risk_off"] and br["strong_count"]==2
    assert br["recovery_path"]=="strong_independent"
    assert b.ordinary_count==b.strong_count==0


def test_alternating_strong_does_not_accumulate():
    s=active()
    for h,di in ((48,1),(52,-1),(56,1),(60,-1),(64,1)):
        r=call(s,"B",h,(1,1,di,1,.2))
        assert r["risk_off"]
    assert s.strong_count==1


def test_ordinary_three_without_strong():
    s=active()
    s.gate.previous_structure=(-10,-10,1,1,.8)
    for h,v in ((48,-9),(52,-8),(56,-7)):
        r=call(s,"B",h,(v,v,1,1,.8))
    assert not r["risk_off"] and r["recovery_path"]=="ordinary"


def test_48_hour_boundary():
    s=active()
    s.gate.since=4*3600
    r=call(s,"B",48,(4,4,1,1,.2))
    assert r["risk_off"]
    r=call(s,"B",52,(3,3,1,1,.2))
    assert not r["risk_off"]


def test_duplicate_and_repeated_4h_do_not_count():
    s=active()
    call(s,"B",48,(4,4,1,1,.2))
    saved=s.dump()
    call(s,"B",48,(4,4,1,1,.2),fold=99,boundary=0)
    assert s.dump()==saved
    call(s,"B",49,(4,4,1,1,.2),structure_hour=48)
    assert s.strong_count==1 and s.gate.active


def test_missing_4h_breaks_continuity():
    s=active()
    call(s,"B",48,(4,4,1,1,.2))
    r=call(s,"B",56,(3,3,1,1,.2))
    assert r["risk_off"] and r["missing_4h"] and s.strong_count==1


def test_cross_week_and_json_resume_keep_active_progress():
    s=active()
    call(s,"B",48,(4,4,1,1,.2))
    restored=CounterState.restore(json.loads(json.dumps(s.dump())))
    r=call(restored,"B",52,(3,3,1,1,.2),fold=2,boundary=50*3600)
    assert not r["risk_off"] and r["strong_count"]==2


def test_risk_on_fold_requires_new_entry_evidence():
    s=CounterState(v22.GateState(armed_until=1000000,above_entry_count=9),fold=1)
    rollover(s,2,100000)
    assert s.gate.armed_until is None and s.gate.above_entry_count==0
    assert s.gate.entry_evidence_not_before==100000


@pytest.mark.parametrize("seed", range(5))
def test_a_full_state_exact_reference_random_sequences(seed):
    rng=np.random.default_rng(seed)
    s,ref=CounterState(),CounterState()
    for hour in range(1,600):
        structure=tuple(rng.uniform(-1,1,4))+(float(rng.random()),)
        fold=hour//168
        rollover(ref,fold,fold*168*3600)
        args=dict(pair="BTC-FDUSD",probability=float(rng.random()),entry_threshold=.5,signal_ts=hour*3600,
                  last_complete_4h_ts=(hour//4)*14400,structure=structure,state=ref.gate)
        ref.gate,_=v22.advance_gate(**args)
        step(s,arm="A",pair="BTC-FDUSD",probability=args["probability"],threshold=.5,ts=hour*3600,
             structure_ts=args["last_complete_4h_ts"],structure=structure,fold=fold,boundary=fold*168*3600)
        assert s.dump()["gate"]==v22.state_to_dict(ref.gate)


def fixtures(hours=72, off=False):
    ts=np.arange(86400,86400+hours*3600,300)
    candles={p:pd.DataFrame(dict(timestamp=ts,open=100.,high=100.,low=100.,close=100.,volume=1.)) for p in v22.PAIRS}
    signals=pd.DataFrame([dict(pair=p,arm=a,signal_ts=int(t),risk_off=off) for p in v22.PAIRS for a in ("A","B")
                          for t in np.arange(86400-3600,86400+hours*3600,3600)])
    filters={p:dict(tick_size=.01,step_size=.0001,minimum_notional=5.) for p in v22.PAIRS}
    return candles,signals,filters,int(ts[0]),int(ts[-1]+300)


def test_riskoff_initially_flattens_once_and_cash_stays_flat():
    c,s,f,start,end=fixtures(off=True)
    for p in c:
        c[p].loc[1:,["open","high","low","close"]]=40.
    eq,t,e,b=replay_grid(c,s,f,"A",start,end)
    assert len(t)==2 and t.side.eq("SELL").all() and t.ts.eq(start).all()
    assert len(e)==2
    assert eq.groupby("pair").equity.nunique().eq(1).all()
    assert all(not x.active and not x.orders for x in b.values())


def test_model_signal_delayed_5_minutes():
    c,s,f,start,end=fixtures(hours=6)
    s.loc[s.signal_ts>=start+3600,"risk_off"]=True
    eq,t,e,b=replay_grid(c,s,f,"A",start,end)
    assert t.ts.eq(start+3600+300).all()


def test_future_high_cannot_fill_newly_rebuilt_orders():
    c,s,f,start,end=fixtures(hours=1)
    for p in c:
        c[p].loc[0,"high"]=200.
    eq,t,e,b=replay_grid(c,s,f,"A",start,end)
    assert t.empty  # initial grids placed AFTER first candle, spike already past


def test_recovery_waits_three_cycles_and_no_capital_injection():
    c,s,f,start,end=fixtures(hours=12,off=True)
    s.loc[s.signal_ts>=start+3600,"risk_off"]=False
    eq,t,e,b=replay_grid(c,s,f,"B",start,end)
    buys=t[t.side=="BUY"]
    assert buys.ts.eq(start+3600+300+3*300).all()
    assert len(t)==4
    assert (eq.equity<=200).all() and (eq.equity>199).all()
    assert np.isclose(sum(x.equity(100) for x in b.values()),400-t.fee.sum()-t.slippage_cost.sum())


def test_reentry_insufficient_cash_stays_blocked():
    b=Book("BTC-FDUSD",50,0,0,0,100,active=False)
    f={b.pair:dict(tick_size=.01,step_size=.0001,minimum_notional=5)}
    assert not can_reenter(b,100,f)


def test_generation_unique_and_inventory_budget_safe():
    b=Book("BTC-FDUSD",100,1,150,1,100)
    f={b.pair:dict(tick_size=.01,step_size=.0001,minimum_notional=5)}
    build_orders(b,100,0,f)
    assert len({(x[0],x[1]) for x in b.orders})==len(b.orders)
    assert sum(x[2] for x in b.orders if x[0]=="SELL")<=1
    assert all(q*p>=10 for _,p,q,_,_ in b.orders)
    assert all(p>=150*1.004 for side,p,_,_,_ in b.orders if side=="SELL")


@pytest.mark.parametrize("bad", ["duplicate","gap","ohlc","nan"])
def test_bad_candles_rejected(bad):
    c,_,_,_,_=fixtures(hours=1)
    x=c["BTC-FDUSD"].copy()
    if bad=="duplicate": x.loc[1,"timestamp"]=x.loc[0,"timestamp"]
    if bad=="gap": x=x.drop(index=1)
    if bad=="ohlc": x.loc[0,"low"]=200
    if bad=="nan": x.loc[0,"volume"]=np.nan
    with pytest.raises(ValueError): validate_frame(x,"BTC-FDUSD")


@pytest.mark.parametrize("structure_hour",[40,52])
def test_invalid_structure_time_fails_without_mutation(structure_hour):
    s=active()
    old=s.dump()
    with pytest.raises(ValueError): call(s,"B",48,(4,4,1,1,.2),structure_hour=structure_hour)
    assert s.dump()==old


def test_pair_exit_isolation_and_six_hour_cooldown():
    c,s,f,start,end=fixtures(hours=12)
    p="BTC-FDUSD"
    c[p].loc[1:,["open","high","low","close"]]=93.
    eq,t,e,b=replay_grid(c,s,f,"A",start,end)
    exits=e[e.kind=="EXIT_COMPLETE"]
    assert set(exits.pair)=={p}
    assert exits.reason.str.startswith("STRATEGY").all()
    assert (exits.cooldown_until-exits.ts).eq(6*3600).all()
    assert t[t.side=="BUY"].ts.min()>=exits.ts.iloc[0]+6*3600+600
    assert b["ETH-FDUSD"].active


def test_portfolio_exit_atomic_reentry_with_twelve_hour_cooldown():
    c,s,f,start,end=fixtures(hours=24)
    for p in c: c[p].loc[1:,["open","high","low","close"]]=70.
    eq,t,e,b=replay_grid(c,s,f,"B",start,end)
    exits=e[e.kind=="EXIT_COMPLETE"]
    assert exits.reason.str.startswith("PORTFOLIO").all() and len(exits)==2
    assert (exits.cooldown_until-exits.ts).eq(12*3600).all()
    buys=t[t.side=="BUY"]
    assert len(buys)==2 and buys.ts.nunique()==1
    assert buys.ts.min()>=exits.ts.max()+12*3600


def test_feature_pipeline_is_prefix_causal_with_real_candles():
    directory=Path(__file__).resolve().parents[1]/"results/backtests/grid_v22_recovery_counter_ablation_2026/inputs"
    if not (directory/"binance_BTC-FDUSD_5m.csv").exists():
        pytest.skip("Read-only acquired experiment input not present")
    cutoff=int(pd.Timestamp("2026-01-02",tz="UTC").timestamp())
    frames={p:pd.read_csv(directory/f"binance_{p}_5m.csv") for p in v22.PAIRS}
    before={p:x[x.timestamp<cutoff].copy() for p,x in frames.items()}
    after={p:x[x.timestamp<cutoff+86400].copy() for p,x in frames.items()}
    expected=v22.build_inference_panel(before).reset_index(drop=True)
    observed=v22.build_inference_panel(after)
    observed=observed[observed.signal_ts<=cutoff].reset_index(drop=True)
    pd.testing.assert_frame_equal(expected,observed)

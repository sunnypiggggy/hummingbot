"""Causal signal alignment, owned-inventory execution and state-shading regressions."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from prepare_v22_365d_inputs import START, END, validate
from replay_v22_365d import gate_on_bars, dca_replay
from render_v22_365d import risk_intervals


def candles(count=24):
    return pd.DataFrame({'timestamp':np.arange(START,START+count*300,300),
                         'open':100.,'high':101.,'low':100.,'close':100.,'volume':1.})


def signals(old=True,new=True):
    return pd.DataFrame([{'pair':p,'signal_ts':ts,'risk_off':off}
                         for p in ['BTC-FDUSD','ETH-FDUSD']
                         for ts,off in [(START-3600,old),(START,new),(START+3600,new)]])


def filters():
    return {p:{'step_size':.0001,'market_step_size':.0001,'tick_size':.01,'minimum_notional':5.}
            for p in ['BTC-USDT','ETH-USDT']}


def test_exact_365_day_window_and_candle_validation():
    assert END-START==365*86400
    assert (END-START)//300==105120
    frame=candles()
    validate(frame,START,START+len(frame)*300)
    for broken in [frame.iloc[:-1],pd.concat([frame,frame.iloc[[-1]]]),frame.iloc[::-1]]:
        with pytest.raises(ValueError):validate(broken,START,START+len(frame)*300)
    invalid=frame.copy();invalid.loc[0,'high']=99
    with pytest.raises(ValueError):validate(invalid,START,START+len(frame)*300)


def test_signal_only_available_after_five_minutes_and_no_fallback():
    result=gate_on_bars(signals(True,False),'BTC-FDUSD',np.array([START,START+299,START+300]))
    assert result.tolist()==[True,True,False]
    with pytest.raises(ValueError,match='no fallback'):
        gate_on_bars(signals(),'BTC-FDUSD',np.array([START-7200]))


def test_starting_risk_off_exits_startup_inventory_once_without_buy():
    data={p:candles() for p in filters()}
    equity,trades,events,books=dca_replay(data,signals(),filters(),'with_v22')
    assert len(trades)==2
    assert trades.side.eq('SELL').all()
    assert trades.quantity.eq(.95).all()
    assert events.kind.eq('EXIT_COMPLETE').all()
    assert not equity.active.any()
    assert equity.base.min()>=0 and equity.quote.min()>0
    assert all(abs(b.base)<1e-10 for b in books.values())


def test_recovery_requires_health_cycles_and_does_not_inject_equity():
    data={p:candles() for p in filters()}
    equity,trades,events,books=dca_replay(data,signals(True,False),filters(),'with_v22')
    buys=trades[trades.side=='BUY']
    assert len(buys)==2
    assert buys.ts.min()>=START+3*300
    for p,b in books.items():
        t=trades[trades.pair==p]
        initial_base=.95;initial_quote=95.
        direction=np.where(t.side=='BUY',1.,-1.)
        assert b.base==pytest.approx(initial_base+sum(t.quantity*direction))
        assert b.quote==pytest.approx(initial_quote-sum(t.notional*direction)-sum(t.fee))
        assert b.equity(100)<190


def test_future_bars_cannot_change_prefix_equity():
    all_frames={p:candles(24) for p in filters()}
    prefix={p:f.iloc[:6].copy() for p,f in all_frames.items()}
    first=dca_replay(prefix,signals(False,False),filters(),'with_v22')[0]
    # A future crash is allowed to affect only future observations.
    for frame in all_frames.values():
        frame.loc[12:,['open','close','high','low']]=80.
    second=dca_replay(all_frames,signals(False,False),filters(),'with_v22')[0]
    pd.testing.assert_frame_equal(first.reset_index(drop=True),second[second.ts<=START+6*300].reset_index(drop=True))


def test_risk_off_shadows_use_state_not_probability_and_keep_gaps():
    frame=pd.DataFrame({'pair':['BTC-FDUSD']*5,
        'signal_ts':[START-3600,START,START+3600,START+10800,START+14400],
        'risk_off':[True,True,False,True,False], 'probability':[.9,.1,.99,.1,.99]})
    assert risk_intervals(frame,'BTC-FDUSD')==[(START,START+3600),(START+10800,START+14400)]
    assert risk_intervals(frame,'ETH-FDUSD')==[]

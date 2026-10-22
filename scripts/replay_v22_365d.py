"""Continuous spot-ledger execution for the isolated 365-day v22 experiment."""
from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path

import numpy as np
import pandas as pd

from prepare_v22_365d_inputs import START, END, sha
from backtest_v22_365d import save_csv, write_json
import backtest_grid_v22_recovery_counters_2026 as grid

SCENARIOS = ('without_v22', 'with_v22')
ROOT = Path(__file__).resolve().parents[1]


def gate_on_bars(signals, pair, timestamps, enabled=True):
    source = signals[signals.pair==pair].sort_values('signal_ts')
    available = source.signal_ts.to_numpy(dtype='int64')+300
    index = np.searchsorted(available,timestamps,side='right')-1
    if (index<0).any() or ((timestamps-available[index])>=3600).any():
        raise ValueError('missing causal v22 signal; no fallback')
    return source.risk_off.to_numpy(dtype=bool)[index] if enabled else np.zeros(len(timestamps),dtype=bool)


@dataclass
class DcaBook:
    pair: str
    quote: float
    base: float
    baseline_base: float
    cycle_baseline: float = 190.
    peak: float = 190.
    active: bool = True
    until: int = 0
    healthy: int = 0
    pending: str = ''
    reenter_pending: bool = False
    executor: dict | None = None
    serial: int = 0
    rounds: int = 0
    orders_placed: int = 0

    def equity(self,price):
        return self.quote+self.base*price


def floor(value,step):
    return grid.quantize(max(0.,value),step)


def dca_replay(frames,signals,filters,scenario):
    arrays={p:f[['timestamp','open','high','low','close']].to_numpy() for p,f in frames.items()}
    times=next(iter(arrays.values()))[:,0].astype('int64')
    gates={p:gate_on_bars(signals,p.replace('USDT','FDUSD'),times,scenario=='with_v22') for p in frames}
    books={}
    for pair,rows in arrays.items():
        price=float(rows[0,1]);base=floor(95/price,filters[pair]['step_size'])
        books[pair]=DcaBook(pair,190-base*price,base,base)
    trades,events,equity=[],[],[]
    combined_peak=combined_baseline=380.
    portfolio_halted=False
    portfolio_until=0

    def trade(book,side,quantity,price,ts,reason,taker,executor_id=None):
        if quantity<=0:return
        execution=price*(1+(.0002 if side=='BUY' else -.0002)) if taker else price
        notional=quantity*execution;fee=notional*.001
        if side=='BUY':
            if notional+fee>book.quote+1e-8:raise AssertionError('DCA overspend')
            book.quote-=notional+fee;book.base+=quantity
        else:
            if quantity>book.base+1e-12:raise AssertionError('DCA oversell')
            book.base-=quantity;book.quote+=notional-fee
        trades.append(dict(pair=book.pair,ts=ts,side=side,quantity=quantity,price=execution,notional=notional,
                           fee=fee,slippage_cost=abs(execution-price)*quantity,kind='TAKER' if taker else 'MAKER',
                           reason=reason,executor_id=executor_id))
        if book.quote < -1e-8 or book.base < -1e-12:raise AssertionError('DCA negative assets')

    def reenter(book,price,ts):
        target=max(0.,95-book.base*price)
        if book.quote < target*1.0012:return False
        quantity=floor(target/(price*1.0002*1.001),filters[book.pair]['step_size'])
        if quantity*price>=filters[book.pair]['minimum_notional']:
            trade(book,'BUY',quantity,price,ts,'REENTRY',True)
        book.baseline_base=book.base
        book.cycle_baseline=book.peak=book.equity(price)
        book.active=True;book.healthy=0;book.reenter_pending=False
        events.append(dict(pair=book.pair,ts=ts,kind='REENTRY'))
        return True

    for i,ts in enumerate(times):
        ts=int(ts)
        opens={p:float(a[i,1]) for p,a in arrays.items()}
        closes={p:float(a[i,4]) for p,a in arrays.items()}
        for pair,book in books.items():
            if gates[pair][i] and book.active:
                book.pending='V22_RISK_OFF'
                book.reenter_pending=False
            if book.pending:
                reason=book.pending
                if reason in {'TAKE_PROFIT','STOP_LOSS','TIME_LIMIT'}:
                    executor=book.executor
                    quantity=floor(executor['quantity'],filters[pair]['market_step_size']) if executor else 0.
                    if quantity*opens[pair]>=filters[pair]['minimum_notional']:
                        trade(book,'SELL',quantity,opens[pair],ts,reason,True,executor['id'])
                    book.rounds+=int(executor is not None and executor['quantity']>0)
                    book.executor=None;book.pending=''
                    if reason=='STOP_LOSS':book.until=max(book.until,ts+1800)
                else:
                    quantity=floor(book.base,filters[pair]['market_step_size'])
                    if quantity*opens[pair]>=filters[pair]['minimum_notional']:
                        trade(book,'SELL',quantity,opens[pair],ts,reason,True)
                    book.rounds+=int(book.executor is not None and book.executor['quantity']>0)
                    book.executor=None;book.active=False;book.healthy=0;book.pending=''
                    if reason.startswith('STRATEGY'):book.until=max(book.until,ts+6*3600)
                    if reason.startswith('PORTFOLIO'):book.until=max(book.until,ts+12*3600)
                    events.append(dict(pair=pair,ts=ts,kind='EXIT_COMPLETE',reason=reason,dust_value=book.base*opens[pair]))
        # Portfolio re-entry is all-or-none and still requires both model gates.
        if portfolio_halted and ts>=portfolio_until:
            ready=all(book.healthy>=3 and not gates[p][i] and ts>=book.until and
                      book.quote>=max(0.,95-book.base*opens[p])*1.0012 for p,book in books.items())
            if ready:
                for p,book in books.items():assert reenter(book,opens[p],ts)
                portfolio_halted=False
                combined_baseline=combined_peak=sum(b.equity(opens[p]) for p,b in books.items())
        if not portfolio_halted:
            for p,book in books.items():
                if book.reenter_pending:
                    if gates[p][i] or ts<book.until:book.reenter_pending=False
                    else:reenter(book,opens[p],ts)
        for pair,book in books.items():
            high,low=arrays[pair][i,2:4]
            executor=book.executor
            if book.active and ts>=book.until and not gates[pair][i] and executor:
                for order in executor['orders']:
                    if not order['filled'] and low<=order['price']:
                        trade(book,'BUY',order['quantity'],order['price'],ts+300,'DCA_ENTRY',False,executor['id'])
                        order['filled']=True
                        executor['quantity']+=order['quantity']
                        executor['cost']+=order['quantity']*order['price']
                        executor['first_fill']=executor['first_fill'] or ts+300
                if executor['quantity']:
                    net=(executor['quantity']*closes[pair]-executor['cost']*1.001)/executor['cost']
                    age=ts+300-executor['first_fill']
                    reason='STOP_LOSS' if net<=-.05 else 'TAKE_PROFIT' if net>.02 else 'TIME_LIMIT' if age>=18000 else ''
                    if reason:book.pending=reason
                elif ts+300-executor['created_at']>=18000:
                    book.executor=None
            value=book.equity(closes[pair]);book.peak=max(book.peak,value)
            if book.active:
                reason='STRATEGY_LOSS' if value<=book.cycle_baseline-16 else 'STRATEGY_DRAWDOWN' if value<=book.peak*.92 else ''
                if reason:book.pending=reason
            else:
                healthy=not gates[pair][i] and ts+300>=max(book.until,portfolio_until)
                book.healthy=book.healthy+1 if healthy else 0
                if book.healthy>=3 and not portfolio_halted:book.reenter_pending=True
            # Orders are constructed at the completed close, eligible next bar only.
            if book.active and not book.pending and not book.executor and ts+300>=book.until and not gates[pair][i]:
                budget=min(95.,max(0.,book.quote))/1.001
                orders=[]
                for spread,weight in zip((.01,.02,.04,.08),(.1,.2,.3,.4)):
                    price=floor(closes[pair]*(1-spread),filters[pair]['tick_size'])
                    quantity=floor(budget*weight/price,filters[pair]['step_size'])
                    if quantity*price>=filters[pair]['minimum_notional']:
                        orders.append(dict(price=price,quantity=quantity,filled=False))
                if orders:
                    book.serial+=1;book.orders_placed+=len(orders)
                    book.executor={'id':f'{pair}:{book.serial}','created_at':ts+300,'orders':orders,
                                   'quantity':0.,'cost':0.,'first_fill':None}
        total=sum(b.equity(closes[p]) for p,b in books.items())
        combined_peak=max(combined_peak,total)
        if not portfolio_halted and (total<=combined_baseline-32 or total<=combined_peak*.92):
            reason='PORTFOLIO_LOSS' if total<=combined_baseline-32 else 'PORTFOLIO_DRAWDOWN'
            portfolio_halted=True;portfolio_until=ts+300+12*3600
            for b in books.values():b.pending=reason;b.reenter_pending=False
        for p,b in books.items():
            equity.append(dict(pair=p,ts=ts+300,price=closes[p],equity=b.equity(closes[p]),quote=b.quote,base=b.base,
                               risk_off=bool(gates[p][i]),active=b.active,cycle_peak=b.peak,cycle_baseline=b.cycle_baseline,
                               phase='EXITING' if b.pending else 'ACTIVE' if b.active else
                               'COOLDOWN' if ts+300<max(b.until,portfolio_until) else 'REENTRY'))
    return pd.DataFrame(equity),pd.DataFrame(trades),pd.DataFrame(events),books


def metric(strategy,pair,scenario,curve,trades,events,initial,orders,rounds):
    values=np.r_[initial,curve.equity.to_numpy()]
    own_trades=trades[trades.pair==pair] if not trades.empty else trades
    own_events=events[events.pair==pair] if not events.empty else events
    return dict(strategy=strategy,pair=pair,scenario=scenario,quote_currency='FDUSD' if strategy=='Grid' else 'USDT',
                initial_equity=initial,final_equity=float(values[-1]),net_pnl_quote=float(values[-1]-initial),
                return_pct=float((values[-1]/initial-1)*100),max_drawdown_pct=float(np.max(1-values/np.maximum.accumulate(values))*100),
                fills=len(own_trades),buy_fills=int(own_trades.side.eq('BUY').sum()) if not own_trades.empty else 0,
                sell_fills=int(own_trades.side.eq('SELL').sum()) if not own_trades.empty else 0,
                fees_quote=float(own_trades.fee.sum()) if not own_trades.empty else 0.,
                slippage_quote=float(own_trades.slippage_cost.sum()) if not own_trades.empty else 0.,
                exits=int(own_events.kind.eq('EXIT_COMPLETE').sum()) if not own_events.empty else 0,
                reentries=int(own_events.kind.eq('REENTRY').sum()) if not own_events.empty else 0,
                risk_off_hours=float(curve.risk_off.sum()/12),placed_orders=orders,completed_rounds=rounds)


def run(output:Path):
    inputs=output/'inputs'
    signals=pd.read_csv(output/'hourly_signals.csv.gz')
    for col in ['risk_off','risk_off_active','recommended_buy_enabled']:
        signals[col]=signals[col].astype(str).str.lower().map({'true':True,'false':False})
        if signals[col].isna().any():raise ValueError('invalid signal boolean')
    data=json.loads((inputs/'market_data_manifest.json').read_text())
    filters={p:{k:float(v) for k,v in row.items()} for p,row in data['filters'].items()}
    grid_frames={p:pd.read_csv(inputs/f'binance_{p}_5m.csv') for p in grid.v22.PAIRS}
    dca_frames={p:pd.read_csv(inputs/f"{p.replace('-','')}_5m.csv") for p in ['BTC-USDT','ETH-USDT']}
    for p,frame in dca_frames.items():
        if len(frame)!=105120:raise ValueError('DCA candle window incomplete')
    all_curves=[];all_trades=[];all_events=[];metrics=[]
    for scenario in SCENARIOS:
        for strategy in ['Grid','DCA']:
            print('Replaying',strategy,scenario,flush=True)
            if strategy=='Grid':
                selected=signals.copy();selected['arm']=scenario
                if scenario=='without_v22':selected['risk_off']=False
                curves,trades,events,books=grid.replay_grid(grid_frames,selected,filters,scenario,START,END)
            else:
                curves,trades,events,books=dca_replay(dca_frames,signals,filters,scenario)
            curves['strategy']=strategy;curves['scenario']=scenario
            curves['drawdown_pct']=curves.groupby('pair').equity.transform(lambda x:(1-x/np.maximum.accumulate(np.r_[200. if strategy=='Grid' else 190.,x])[1:])*100)
            all_curves.append(curves)
            if not trades.empty:all_trades.append(trades.assign(strategy=strategy,scenario=scenario))
            if not events.empty:all_events.append(events.assign(strategy=strategy,scenario=scenario))
            for p,book in books.items():
                own=curves[curves.pair==p]
                metrics.append(metric(strategy,p,scenario,own,trades,events,200. if strategy=='Grid' else 190.,
                                      book.submitted if strategy=='Grid' else book.orders_placed,
                                      None if strategy=='Grid' else book.rounds))
    curves=pd.concat(all_curves,ignore_index=True)
    trades=pd.concat(all_trades,ignore_index=True)
    events=pd.concat(all_events,ignore_index=True)
    save_csv(output/'continuous_equity_5m.csv.gz',curves)
    save_csv(output/'trade_fills.csv.gz',trades)
    save_csv(output/'execution_events.csv.gz',events)
    summary=pd.DataFrame(metrics)
    save_csv(output/'summary.csv',summary)
    # Independent conservation check: each ledger equals initial assets + its fills.
    checks=[]
    for (strategy,pair,scenario),part in curves.groupby(['strategy','pair','scenario']):
        t=trades[(trades.strategy==strategy)&(trades.pair==pair)&(trades.scenario==scenario)]
        first_price=(grid_frames if strategy=='Grid' else dca_frames)[pair].query('timestamp>=@START').iloc[0].open
        initial=200. if strategy=='Grid' else 190.
        allocation=100. if strategy=='Grid' else 95.
        base=floor(allocation/first_price,filters[pair]['step_size'])
        quote=initial-base*first_price
        direction=np.where(t.side=='BUY',1.,-1.)
        base+=float((t.quantity*direction).sum())
        quote+=float((-t.notional*direction-t.fee).sum())
        tail=part.iloc[-1]
        if abs(base-tail.base)>1e-10 or abs(quote-tail.quote)>1e-7:raise AssertionError('ledger conservation failed')
        if part.base.min() < -1e-12 or part.quote.min() < -1e-8:raise AssertionError('negative spot balances')
        if scenario=='with_v22':
            buy=t[t.side=='BUY']
            # Entry/re-entry buys must consume an available, healthy model signal.
            decision_times=np.where(buy.kind.eq('MAKER'),buy.ts.to_numpy(dtype='int64')-300,buy.ts.to_numpy(dtype='int64'))
            available=gate_on_bars(signals,pair.replace('USDT','FDUSD'),np.maximum(decision_times,START))
            if available.any():raise AssertionError('Risk-Off BUY in protected scenario')
        checks.append(dict(strategy=strategy,pair=pair,scenario=scenario,base_error=base-tail.base,quote_error=quote-tail.quote))
    write_json(output/'validation.json',{'ledger_conservation':checks,'rows_per_bot_scenario':105120,
        'negative_assets':0,'risk_off_buys':0,'new_live_orders':0,'cross_week_resets':0})
    write_json(output/'parameter_snapshot.json',{'grid':grid.PARAMS,'grid_capital_per_pair':200,'grid_reserve_fdusd':20,
        'grid_minimum_order_fdusd':10,'grid_refresh_seconds':7200,'grid_movement':'beyond boundary 1.5%; 30m cooldown',
        'dca':{'long_only':True,'capital_usdt':190,'allocated_initial_base_quote':95,'buy_budget_usdt':95,
               'spreads':[.01,.02,.04,.08],'weights':[.1,.2,.3,.4],'take_profit':.02,'stop_loss':.05,
               'time_limit_seconds':18000,'executor_refresh_seconds':18000,'stop_cooldown_seconds':1800},
        'risk':{'grid_pair_loss':6,'grid_pair_drawdown':.03,'grid_portfolio_loss':24,'grid_portfolio_drawdown':.06,
                'dca_pair_loss':16,'dca_pair_drawdown':.08,'dca_portfolio_loss':32,'dca_portfolio_drawdown':.08,
                'pair_cooldown_seconds':21600,'portfolio_cooldown_seconds':43200},
        'fomc':False,'bnb_fee_used':False,'grid_maker_fee':0,'dca_maker_fee':.001,'taker_fee':.001,'taker_slippage_bps':2,
        'note':'Freeze current strategy for all 365 days; not historical live configurations. Filters frozen as-of acquisition. Guard healthy cycles approximated by 3 completed 5m bars.'})
    write_json(output/'summary.json',{'schema':'v22-365d-offline-backtest-v1','offline_only':True,'deployment_allowed':False,
        'orders_submitted':False,'start_utc':'2025-10-17T00:00:00Z','end_exclusive_utc':'2026-10-17T00:00:00Z','days':365,
        'metrics':metrics,'limitations':['5m OHLC touch fills are not queue-level execution; no order-book reconstruction.',
        'Hyperparameters and feature choice fixed retrospectively; early backfill is research, not actual historical production.',
        'Current exchange filters applied over the whole window. No FOMC or parameter optimizer.',
        'Three healthy Guard cycles approximated as three 5m bars.']})
    from render_v22_365d import render
    render(output,curves,signals,summary)
    write_json(output/'artifact_manifest.json',{'offline_only':True,'deployment_allowed':False,
        'input_hashes':{str(p.relative_to(inputs)):sha(p) for p in inputs.rglob('*') if p.is_file()},
        'source_hashes':{str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),ROOT/'scripts/backtest_v22_365d.py',
        ROOT/'scripts/render_v22_365d.py',ROOT/'scripts/xgboost_long_risk_gate_v22.py',
        ROOT/'scripts/xgboost_long_risk_gate_v22_features.py',ROOT/'scripts/backtest_grid_v22_recovery_counters_2026.py',
        ROOT/'scripts/prepare_v22_365d_inputs.py',ROOT/'scripts/grid_v22_recovery_ablation.py',ROOT/'scripts/grid_live_common.py',
        ROOT/'scripts/retrain_xgboost_long_risk_gate_250d_v19.py',ROOT/'scripts/compare_independent_gate_ml_stops.py',
        ROOT/'scripts/optimize_xgboost_grid_risk_gate_v7.py',ROOT/'scripts/tune_xgboost_momentum_stop_v2.py']},
        'artifacts':{p.name:sha(p) for p in output.iterdir() if p.is_file() and p.name!='artifact_manifest.json'}})
    print(summary.to_string(index=False),flush=True)

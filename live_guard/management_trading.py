"""Sanitized read-only runtime and historical trading summary for Telegram."""
import json
import sqlite3
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

BOTS={'grid':('grid-live-fdusd-400',('BTC-FDUSD','ETH-FDUSD')),
      'dca_btc':('dca-live-btcusdt-200',('BTC-USDT',)),
      'dca_eth':('dca-live-ethusdt-200',('ETH-USDT',))}


def seconds(value):
    value=float(value)
    return value/1000 if value>10000000000 else value


def history(path,pair,now):
    midnight=datetime.fromtimestamp(now,timezone(timedelta(hours=8))).replace(hour=0,minute=0,second=0,microsecond=0).timestamp()
    counters={scope:{'BUY':0,'SELL':0,'normal':0,'stop_loss':0,'other':0} for scope in ('today','total')}
    seen={}; first=None; rounds_from=None; unverified=0; residual=0
    with sqlite3.connect(path.resolve().as_uri()+'?mode=ro',uri=True,timeout=2) as db:
        db.row_factory=sqlite3.Row; db.execute('PRAGMA query_only=ON'); db.execute('BEGIN')
        for r in db.execute('SELECT market,symbol,exchange_trade_id,trade_type,timestamp,amount,price FROM TradeFill WHERE symbol=?',(pair,)):
            at=seconds(r['timestamp']); key=(r['market'],pair,r['exchange_trade_id'])
            if not r['exchange_trade_id'] or at>now: raise ValueError('unverified_fill')
            signature=(r['trade_type'],r['amount'],r['price'],at)
            if key in seen:
                if seen[key]!=signature: raise ValueError('conflicting_fill')
                continue
            seen[key]=signature
            if r['trade_type'] not in ('BUY','SELL'): raise ValueError('unknown_side')
            first=at if first is None else min(first,at)
            counters['total'][r['trade_type']]+=1
            if at>=midnight: counters['today'][r['trade_type']]+=1
        for r in db.execute('SELECT id,config,custom_info,status,close_type,close_timestamp FROM Executors'):
            config=json.loads(r['config']); custom=json.loads(r['custom_info'] or '{}')
            if config.get('trading_pair')!=pair or r['status']!=4: continue
            progress=custom.get('management_progress')
            if not progress or not r['close_timestamp']:
                unverified+=1; continue
            opened=Decimal(progress['open_base']); closed=Decimal(progress['closed_base'])
            if opened<=0: continue
            if closed<opened:
                residual+=1; continue
            at=seconds(r['close_timestamp'])
            if at>now: raise ValueError('future_executor')
            rounds_from=at if rounds_from is None else min(rounds_from,at)
            kind='stop_loss' if r['close_type']==2 else 'normal' if r['close_type'] in (3,6,9) else 'other'
            counters['total'][kind]+=1
            if at>=midnight: counters['today'][kind]+=1
    return {**counters,'fills_from':first,'rounds_from':rounds_from,
            'unverified_rounds':unverified,'residual_rounds':residual}


def publish(bots_root,output,status_contract,now):
    rows=[]
    for key,(bot,pairs) in BOTS.items():
        root=Path(bots_root)/'instances'/bot/'data'
        try:
            snap=json.loads((root/'management_trading_snapshot.json').read_text(encoding='utf-8'))
            if snap['schema']!='management-trading-snapshot-v1' or snap['bot']!=bot: raise ValueError('binding')
            age=now-float(snap['observed_at'])
            if not 0<=age<=60: raise ValueError('stale')
        except (OSError,ValueError,KeyError,TypeError): snap=None
        for pair in pairs:
            status=next((s for s in status_contract['robots'] if s.get('bot')==bot and s.get('pair')==pair),{})
            row={'bot':bot,'strategy':'grid' if key=='grid' else 'dca','pair':pair,'status':status,
                 'snapshot_at':snap['observed_at'] if snap else None,
                 'runtime':snap.get('pairs',{}).get(pair) if snap else None}
            if key!='grid':
                try: row['history']=history(root/(bot+'.sqlite'),pair,now)
                except (OSError,ValueError,KeyError,TypeError,sqlite3.Error,ArithmeticError): row['history']=None
            rows.append(row)
    path=Path(output)/'management_trading.json'; temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps({'schema':'management-trading-v1','generated_at':now,'robots':rows},ensure_ascii=False),encoding='utf-8')
    temp.replace(path)

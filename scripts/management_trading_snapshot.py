"""Best-effort, local-only trading telemetry. Never places or cancels orders."""
import json
import time
from decimal import Decimal
from pathlib import Path


def dca_progress(executor):
    levels=[]
    for index, price in enumerate(executor.config.prices):
        tracked=[t for level,t in executor._management_level_orders.values() if level==index]
        quantities=[t.order for t in tracked if t.order is not None]
        unknown=any(t.order is None for t in tracked)
        filled=sum((o.executed_amount_base for o in quantities),Decimal(0))
        complete=bool(quantities) and not unknown and any(o.is_filled for o in quantities)
        levels.append({'index':index,'price':str(price),'budget':str(executor.config.amounts_quote[index]),
                       'filled_base':None if unknown else str(filled),
                       'state':'UNKNOWN' if unknown else 'FILLED' if complete else 'PARTIAL' if filled>0 else 'WAITING',
                       'active_orders':[{'id':o.client_order_id,'price':str(o.price)} for o in quantities if getattr(o,'is_open',False)],
                       'order_ids':[t.order_id for t in tracked]})
    return {'levels':levels,'open_base':str(executor.open_filled_amount),
            'closed_base':str(executor.close_filled_amount),
            'open_quote':str(executor.open_filled_amount_quote),
            'average_price':str(executor.current_position_average_price),
            'take_profit':None if executor.config.take_profit is None else str(executor.config.take_profit),
            'stop_loss':None if executor.config.stop_loss is None else str(executor.config.stop_loss)}


def _write(strategy, build):
    now=time.time()
    if now-getattr(strategy,'_management_snapshot_at',0)<15: return
    strategy._management_snapshot_at=now
    try:
        value=build(now)
        value.update(schema='management-trading-snapshot-v1',observed_at=now)
        target=Path('data/management_trading_snapshot.json')
        target.parent.mkdir(parents=True,exist_ok=True)
        temp=target.with_suffix('.tmp')
        temp.write_text(json.dumps(value,ensure_ascii=False,default=str),encoding='utf-8')
        temp.replace(target)
    except Exception:
        # Failure leaves the old timestamp visible; it never interrupts trading.
        return


def publish_grid(strategy):
    def build(now):
        pairs={}
        connector=strategy.connector
        active=connector.limit_orders
        tracked=connector._order_tracker.all_orders
        for pair in strategy.config.trading_pairs:
            owned=strategy.ledgers[pair].open_order_ids
            orders=[]
            for order in active:
                if order.trading_pair!=pair or order.client_order_id not in owned: continue
                detail=tracked.get(order.client_order_id)
                filled=detail.executed_amount_base if detail is not None else None
                orders.append({'id':order.client_order_id,'pair':pair,'side':'BUY' if order.is_buy else 'SELL',
                               'price':str(order.price),'amount':str(order.quantity),
                               'filled':None if filled is None else str(filled),
                               'remaining':None if filled is None else str(max(Decimal(0),order.quantity-filled)),
                               'state':str(getattr(getattr(detail,'current_state',None),'name','UNCONFIRMED'))})
            price=strategy.reference_price(pair)
            pairs[pair]={'price':str(price) if price>0 else None,'price_observed_at':now,'orders':orders}
        return {'strategy':'grid','bot':'grid-live-fdusd-400','pairs':pairs}
    _write(strategy,build)


def publish_dca(strategy):
    def build(now):
        pairs={getattr(c.config,'trading_pair',None) for c in strategy.controllers.values()}
        if len(pairs)!=1 or next(iter(pairs)) not in ('BTC-USDT','ETH-USDT'): raise ValueError('scope')
        pair=next(iter(pairs)); executors=[]; price=None
        mark=strategy.connectors['binance'].get_mid_price(pair)
        if mark is not None and mark.is_finite() and mark>0: price=str(mark)
        for info in strategy.get_all_executors():
            if getattr(info.config,'trading_pair',None)!=pair or not info.is_active: continue
            custom=info.custom_info or {}
            executors.append({'id':info.id,'started_at':info.timestamp,'side':info.side.name,
                              'progress':custom.get('management_progress')})
        return {'strategy':'dca','bot':'dca-live-'+pair.replace('-','').lower()+'-200',
                'pairs':{pair:{'price':price,'price_observed_at':now if price is not None else None,'executors':executors}}}
    _write(strategy,build)

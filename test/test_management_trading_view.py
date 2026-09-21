import json
import sqlite3
import time
from decimal import Decimal
from types import SimpleNamespace as NS
from unittest.mock import Mock

from management_bot.app import HOME_ROWS
from management_bot.trading_view import page, progress_lines
from live_guard.management_trading import history, publish
from scripts.management_trading_snapshot import dca_progress, publish_grid
import test.test_trading_management_bot as fixtures


def doc(root,now,orders=None):
    rows=[]
    for strategy,pairs in [('grid',['BTC-FDUSD','ETH-FDUSD']),('dca',['BTC-USDT','ETH-USDT'])]:
        for pair in pairs:
            rows.append({'strategy':strategy,'pair':pair,'snapshot_at':now,'status':{'trading_normal':True,'process_running':True,
                'final_permissions':{'buy_enabled':True,'sell_enabled':True}},
                'runtime':{'price':'100','price_observed_at':now,'orders':orders if orders is not None else [],'executors':[]}})
    (root/'management_trading.json').write_text(json.dumps({'schema':'management-trading-v1','generated_at':now,'robots':rows}))


def test_pages_both_assets_no_mutation_buttons_and_stale(tmp_path):
    now=time.time(); doc(tmp_path,now)
    assert not any(data=='m:audit' for row in HOME_ROWS for _,data in row)
    for strategy in ('grid','dca'):
        text,buttons=page(tmp_path,strategy,now=now)
        assert 'BTC-' in text and 'ETH-' in text and '正常交易' in text
        assert all(c.startswith('tv:') or c=='m:home' for row in buttons for _,c in row)
        assert '超过60秒' in page(tmp_path,strategy,now=now+61)[0]
        assert '超过180秒' in page(tmp_path,strategy,now=now+181)[0]


def test_orders_sort_partial_cancel_isolated_paginated(tmp_path):
    now=time.time()
    orders=[{'pair':'BTC-FDUSD','side':'BUY','price':str(100+i),'remaining':'0.3','filled':'0.2','state':'PENDING_CANCEL'} for i in range(90)]
    orders.append({'pair':'ETH-FDUSD','side':'SELL','price':'300','remaining':'1','filled':'0'})
    doc(tmp_path,now,orders)
    first,buttons=page(tmp_path,'grid',now=now)
    assert first.index('买 189')<first.index('买 188')
    assert '剩余 0.3 · 已成 0.2 · 撤单待确认' in first and len(first)<3500
    for i in range(10):
        text,_=page(tmp_path,'grid',i,now=now)
        assert '<b>BTC-FDUSD</b>' in text and '<b>ETH-FDUSD</b>' in text and len(text)<3500


def test_dca_levels_partial_retry_first_not_addition():
    def track(filled,complete):
        return NS(order_id=str(filled),order=NS(executed_amount_base=Decimal(filled),is_filled=complete))
    e=NS(config=NS(prices=[Decimal(100),Decimal(90),Decimal(80)],amounts_quote=[10,20,30],take_profit=Decimal('.03'),stop_loss=Decimal('.02')),
         _management_level_orders={'a':(0,track('1',True)),'b':(1,track('.2',False)),'c':(1,track('.8',True))},
         open_filled_amount=Decimal(2),close_filled_amount=Decimal(0),open_filled_amount_quote=Decimal(190),current_position_average_price=Decimal(95))
    p=dca_progress(e)
    assert len(p['levels'])==3 and p['levels'][1]['filled_base']=='1.0'
    lines='\n'.join(progress_lines({'id':'x','side':'BUY','progress':p}))
    assert '补仓完成 1/2档' in lines and '下一档计划价（未挂出） 80' in lines
    assert '97.85' in lines and '93.1' in lines
    e._management_level_orders['d']=(2,NS(order_id='d',order=None))
    assert dca_progress(e)['levels'][2]['filled_base'] is None


def test_grid_snapshot_filters_ownership_and_is_readonly(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path)
    orders=[NS(client_order_id='own',trading_pair='BTC-FDUSD',is_buy=True,price=Decimal(10),quantity=Decimal(2)),
            NS(client_order_id='other',trading_pair='BTC-FDUSD',is_buy=True,price=Decimal(11),quantity=Decimal(4))]
    connector=NS(limit_orders=orders,_order_tracker=NS(all_orders={'own':NS(executed_amount_base=Decimal('.5'),current_state=NS(name='PARTIALLY_FILLED'))}))
    strategy=NS(connector=connector,config=NS(trading_pairs=['BTC-FDUSD']),ledgers={'BTC-FDUSD':NS(open_order_ids={'own'})},reference_price=lambda p:Decimal(10))
    publish_grid(strategy)
    path=tmp_path/'data/management_trading_snapshot.json'; before=path.read_bytes(); value=json.loads(before)
    assert value['pairs']['BTC-FDUSD']['orders'][0]['remaining']=='1.5'
    assert len(value['pairs']['BTC-FDUSD']['orders'])==1
    publish_grid(strategy); assert path.read_bytes()==before


def test_sqlite_history_dedup_rounds_zero_fill_residual_and_midnight(tmp_path):
    path=tmp_path/'bot.sqlite'; now=1767229200.0
    with sqlite3.connect(path) as db:
        db.execute('CREATE TABLE TradeFill(market,symbol,exchange_trade_id,trade_type,timestamp,amount,price)')
        row=('binance','BTC-USDT','t1','BUY',now*1000,100,100)
        db.executemany('INSERT INTO TradeFill VALUES (?,?,?,?,?,?,?)',[row,row,('binance','BTC-USDT','t2','SELL',(now-86400)*1000,100,100)])
        db.execute('CREATE TABLE Executors(id,config,custom_info,status,close_type,close_timestamp)')
        for i,(opened,closed) in enumerate([('1','1'),('1','.5'),('0','0')]):
            db.execute('INSERT INTO Executors VALUES (?,?,?,?,?,?)',(str(i),json.dumps({'trading_pair':'BTC-USDT'}),json.dumps({'management_progress':{'open_base':opened,'closed_base':closed}}),4,2,now))
        db.execute('INSERT INTO Executors VALUES (?,?,?,?,?,?)',('old',json.dumps({'trading_pair':'BTC-USDT'}),'{}',4,3,now))
    h=history(path,'BTC-USDT',now)
    assert h['total']['BUY']==1 and h['total']['SELL']==1 and h['today']['SELL']==0
    assert h['total']['stop_loss']==1 and h['residual_rounds']==1 and h['unverified_rounds']==1
    assert history(path,'BTC-USDT',now)==h


def callback(bot,data,index=1):
    bot._handle_callback(index,{'id':str(index),'data':data,'from':{'id':7},'message':{'message_id':11,'chat':{'id':7,'type':'private'}}})


def test_old_controls_rejected_and_new_maintenance_bound_once(tmp_path):
    bot=fixtures.TelegramFlowTests()._bot(tmp_path,mutations_enabled=True)
    bot.hummingbot=Mock(); bot.contracts=Mock(); bot.contracts.snapshot.return_value={'sources':{}}
    for data in ['b:grid:stop','c:grid:stop','b:grid:restart','c:grid:start']:
        callback(bot,data)
    bot.hummingbot.stop.assert_not_called(); bot.hummingbot.start.assert_not_called(); bot.hummingbot.restart.assert_not_called()
    bot.hummingbot.stop.return_value={'status':'stopped'}
    callback(bot,'mt:grid:stop')
    confirmation=bot.telegram.edited[-1][3][0][0][1]
    assert confirmation.startswith('mc:')
    callback(bot,confirmation,2); callback(bot,confirmation,3)
    bot.hummingbot.stop.assert_called_once_with('grid')
    bot.store.close()


def test_removed_audit_button_cannot_send_stale_chart(tmp_path):
    bot=fixtures.TelegramFlowTests()._bot(tmp_path)
    callback(bot,'au:spot:chart')
    assert not bot.telegram.sent
    assert '账户稽核功能已撤回' in bot.telegram.edited[-1][2]
    bot.store.close()


def test_canceled_level_does_not_claim_active_order():
    p={'levels':[{'index':0,'state':'PARTIAL','price':'100','filled_base':'.1',
                  'order_ids':['canceled'],'active_orders':[]}],
       'average_price':'100','open_quote':'10'}
    result='\n'.join(progress_lines({'id':'x','side':'BUY','progress':p}))
    assert '计划价（未挂出） 100' in result and '下一档实际挂单价' not in result
    p['levels'][0]['active_orders']=[{'id':'new','price':'99'}]
    assert '下一档实际挂单价 99' in '\n'.join(progress_lines({'progress':p}))


def test_report_pipeline_binding_staleness_missing_history(tmp_path):
    now=time.time(); root=tmp_path/'instances'/'grid-live-fdusd-400'/'data'; root.mkdir(parents=True)
    path=root/'management_trading_snapshot.json'
    value={'schema':'management-trading-snapshot-v1','bot':'grid-live-fdusd-400','observed_at':now,
           'pairs':{'BTC-FDUSD':{'orders':[],'price':'100'}}}
    path.write_text(json.dumps(value))
    publish(tmp_path,tmp_path,{'robots':[]},now)
    result=json.loads((tmp_path/'management_trading.json').read_text())
    assert len(result['robots'])==4 and result['robots'][0]['runtime']['price']=='100'
    assert result['robots'][2]['history'] is None
    publish(tmp_path,tmp_path,{'robots':[]},now+61)
    assert json.loads((tmp_path/'management_trading.json').read_text())['robots'][0]['runtime'] is None
    value['bot']='other'; path.write_text(json.dumps(value))
    publish(tmp_path,tmp_path,{'robots':[]},now)
    assert json.loads((tmp_path/'management_trading.json').read_text())['robots'][0]['runtime'] is None


def test_telemetry_failure_never_interrupts_trading(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path)
    strategy=NS()  # unavailable connector must not propagate into on_tick
    publish_grid(strategy)
    assert not (tmp_path/'data/management_trading_snapshot.json').exists()


def test_reference_rounding_does_not_change_order_price(tmp_path):
    from management_bot.trading_view import reference_price
    assert reference_price('81956.5',integer=True)=='81,957'
    assert reference_price('2652.2600000000002',integer=True)=='2,652'
    assert reference_price('NaN',integer=True)=='未记录'
    now=time.time();doc(tmp_path,now,[{'pair':'BTC-FDUSD','side':'BUY','price':'99.57','remaining':'1','filled':'0'}])
    text,_=page(tmp_path,'grid',now=now)
    assert '参考价 100' in text and '买 99.57' in text
    assert '第1/1页' not in text
    path=tmp_path/'management_trading.json';value=json.loads(path.read_text())
    value['robots'][2]['runtime']['price']='81956.5'
    path.write_text(json.dumps(value))
    assert '参考价 81,957' in page(tmp_path,'dca',now=now)[0]


def test_dca_compact_richtext_and_missing_evidence(tmp_path):
    now=time.time();doc(tmp_path,now)
    path=tmp_path/'management_trading.json';d=json.loads(path.read_text())
    row=d['robots'][2]
    row['runtime']['executors']=[{'id':'<unsafe>','side':'BUY','progress':{
        'levels':[{'index':0,'state':'WAITING','price':'100'}],
        'open_base':'0','open_quote':'0','average_price':'0','take_profit':'.02','stop_loss':'.05'}}]
    counts={'BUY':2,'SELL':1,'normal':0,'stop_loss':0,'other':0}
    row['history']={'today':counts,'total':counts,'fills_from':now,'rounds_from':None,'unverified_rounds':12000,'residual_rounds':1}
    path.write_text(json.dumps(d));text,_=page(tmp_path,'dca',now=now)
    assert '<b>本轮买入</b>' in text and '<b>成交统计</b>' in text
    assert '均价：尚未成交' in text and '触发参考价：成交后确定' in text
    assert '12000' not in text and '历史轮次证据不完整' in text
    assert '尚有残余仓位：1' in text and '<unsafe>' not in text

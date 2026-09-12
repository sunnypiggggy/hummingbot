import hashlib
import json
import math
import sqlite3
from pathlib import Path

import pytest
from PIL import Image
from live_guard.model_probability_history import collect
from management_bot.probability_chart import snapshot, render, segments, recovery_label


def source(root, now=1788670800, probability=.04, threshold=.05):
    root.mkdir(exist_ok=True)
    manifest = json.dumps({"release_sha256": "b"*64}).encode()
    generation = hashlib.sha256(manifest).hexdigest()
    directory = root / "v22-runtime/generations" / generation
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "manifest.json").write_bytes(manifest)
    (root / "v22-runtime/current.json").write_text(json.dumps({"runtime_generation":generation}))
    gate = dict(runtime_generation=generation, release_sha256="b"*64, model_sha256="a"*64,
                source_healthy=True, generated_at=now, valid_until=now+150,
                pairs={p: dict(signal_ts=now, probability=probability, entry_threshold=threshold,
                               week_start=now-604800,week_end=now+604800,
                               model_week=41,risk_off_active=True)
                       for p in ("BTC-FDUSD","ETH-FDUSD")})
    (root / "xgboost_risk_gate.json").write_text(json.dumps(gate))
    return gate


def test_persistence_dedup_expiry_and_missing_source(tmp_path):
    root,out=tmp_path/"source",tmp_path/"out"
    source(root)
    first=collect(root,out,1788670801)
    assert len(first["pairs"]["BTC-FDUSD"]["points"]) == 1
    assert len(collect(root,out,1788670802)["pairs"]["BTC-FDUSD"]["points"]) == 1
    stale=collect(root,out,1788671100)
    assert not stale["pairs"]["BTC-FDUSD"]["current_available"]
    assert len(stale["pairs"]["BTC-FDUSD"]["points"]) == 1
    (root/"xgboost_risk_gate.json").unlink()
    assert len(collect(root,out,1788671101)["pairs"]["BTC-FDUSD"]["points"]) == 1


@pytest.mark.parametrize("strategy,asset",[(s,a) for s in ("grid","dca") for a in ("BTC","ETH")])
def test_mapping_single_point_render(tmp_path,strategy,asset):
    source(tmp_path/"source")
    data=snapshot(collect(tmp_path/"source",tmp_path/"out",1788670801),strategy,asset,1788670801)
    assert data["pair"] == f"{asset}-{'FDUSD' if strategy=='grid' else 'USDT'}"
    with Image.open(render(data,tmp_path/"image.png")) as image:
        assert image.size == (1440,1800)


@pytest.mark.parametrize("bad",[None,True,float("nan"),float("inf"),-1,2])
def test_bad_probability_rejected(tmp_path,bad):
    source(tmp_path/"source",probability=bad)
    data=collect(tmp_path/"source",tmp_path/"out",1788670801)
    assert not data["pairs"]["BTC-FDUSD"]["points"]


def test_retention_and_failed_generation(tmp_path):
    root,out=tmp_path/"source",tmp_path/"out"
    source(root)
    collect(root,out,1788670801)
    source(root,now=1788670800+31*86400)
    result=collect(root,out,1788670801+31*86400)
    assert len(result["pairs"]["BTC-FDUSD"]["points"]) == 1
    with sqlite3.connect(out/"model_probability_history.sqlite") as db:
        assert db.execute("SELECT count(*) FROM signals").fetchone()[0] == 2
    (root/"v22-runtime/current.json").write_text('{"runtime_generation":"bad"}')
    assert not collect(root,out,1788670802+31*86400)["pairs"]["BTC-FDUSD"]["current_available"]


def test_curve_gap_threshold_step_stale_view(tmp_path):
    rows=[dict(signal_ts=1788600000+i*3600,probability=.1+.2*math.sin(i)**2,
               threshold=.25 if i<80 else .35, week=40 if i<80 else 41,
               risk_off=int(40<i<75),break_before=(i==90))
          for i in range(168) if i not in (50,51,52)]
    contract={"schema":"management-probability-history-v1","generated_at":1788600000,
              "pairs":{"BTC-FDUSD":{"current_available":True,"points":rows}}}
    data=snapshot(contract,"grid","BTC",rows[-1]["signal_ts"]+300)
    assert not data["available"]
    assert len(segments(data)) == 3
    render(data,tmp_path/"curve.png")
    with pytest.raises(ValueError):
        snapshot(contract,"grid","../../escape",rows[-1]["signal_ts"])
    with pytest.raises(ValueError):
        snapshot(contract,"grid","ETH",rows[-1]["signal_ts"])


def test_database_lock(tmp_path):
    source(tmp_path/"source")
    collect(tmp_path/"source",tmp_path/"out",1788670801)
    with sqlite3.connect(tmp_path/"out/model_probability_history.sqlite") as db:
        db.execute("BEGIN EXCLUSIVE")
        with pytest.raises(sqlite3.OperationalError):
            collect(tmp_path/"source",tmp_path/"out",1788670802)


def test_sparse_audit_import_and_idempotence(tmp_path):
    root,out=tmp_path/"source",tmp_path/"out"
    gate=source(root,now=1788667200)
    gate['cutover_phase']='ACTIVE'
    (root/'risk_audit.jsonl').write_text(json.dumps({'timestamp':1788667201,'gate':gate})+'\n')
    source(root)
    result=collect(root,out,1788670801)
    points=result['pairs']['BTC-FDUSD']['points']
    assert len(points)==2
    assert points[0]['break_before']==1
    assert len(collect(root,out,1788670802)['pairs']['BTC-FDUSD']['points'])==2


@pytest.mark.parametrize('count', [0, 1, 2])
def test_current_recovery_binding_and_staleness(tmp_path, count):
    root, out = tmp_path/'source', tmp_path/'out'
    gate = source(root)
    state_path = root/'v22-runtime/generations'/gate['runtime_generation']/'gate_state.json'
    saved = {'pairs': {'BTC-FDUSD': {
        'gate_state': {'last_signal_ts': 1788670800, 'active': True, 'recovery_count': count},
        'last_snapshot': {'probability': .04, 'entry_threshold': .05, 'recovery_required_4h_bars': 3}}}}
    state_path.write_text(json.dumps(saved))
    contract = collect(root, out, 1788670801)
    for strategy in ('grid', 'dca'):
        data = snapshot(contract, strategy, 'BTC', 1788670801)
        assert f'{count} / 3' in recovery_label(data)
        render(data, tmp_path/f'{strategy}-recovery.png')
    assert '暂无可信数据' in recovery_label(snapshot(contract, 'grid', 'ETH', 1788670801))
    assert '暂无可信数据' in recovery_label(snapshot(contract, 'grid', 'BTC', 1788671200))
    saved['pairs']['BTC-FDUSD']['gate_state']['last_signal_ts'] -= 3600
    state_path.write_text(json.dumps(saved))
    assert collect(root, out, 1788670802)['pairs']['BTC-FDUSD']['current_recovery'] is None


def test_recovery_unknown_and_risk_on():
    assert '不适用' in recovery_label({'recovery': {'count': 0, 'ordinary_required': 3, 'risk_off': False}})
    for count in (None, True, -1, 1.5):
        assert '暂无可信数据' in recovery_label({'recovery': {'count': count, 'ordinary_required': 3}})


def test_prices_cached_isolated_and_filtered(tmp_path):
    root,out=tmp_path/'source',tmp_path/'out'
    source(root)
    calls=[]
    def reader(pair,now):
        calls.append(pair)
        if pair=='ETH-USDT':
            raise TimeoutError()
        return [{'timestamp':now-3600,'close':100 if pair.endswith('USDT') else 200},
                {'timestamp':now+1,'close':300}, {'timestamp':now,'close':float('nan')}]
    c=collect(root,out,1788670801,price_reader=reader)
    assert len(calls)==4
    assert snapshot(c,'grid','BTC',1788670801)['prices'][0]['close']==200
    assert snapshot(c,'dca','BTC',1788670801)['prices'][0]['close']==100
    assert snapshot(c,'dca','ETH',1788670801)['prices']==[]
    collect(root,out,1788670802,price_reader=reader)
    assert len(calls)==4
    assert len(c['market_prices']['BTC-FDUSD']['points'])==1


def test_price_chart_and_no_footer(tmp_path,monkeypatch):
    from PIL import ImageDraw
    labels=[]
    original=ImageDraw.ImageDraw.text
    def capture(self,xy,text,*args,**kwargs):
        labels.append(text)
        return original(self,xy,text,*args,**kwargs)
    monkeypatch.setattr(ImageDraw.ImageDraw,'text',capture)
    source(tmp_path/'source')
    c=collect(tmp_path/'source',tmp_path/'out',1788670801)
    c['market_prices']['BTC-FDUSD']['points']=[
        {'timestamp':1788670800-i*3600,'close':65000+i*20} for i in range(168) if i!=50]
    d=snapshot(c,'grid','BTC',1788670801)
    render(d,tmp_path/'price-curve.png')
    assert any('1小时收盘价' in label for label in labels)
    assert not any('线上记录' in label or '不重算补齐' in label or '本图不改变' in label for label in labels)


def test_market_reader_excludes_open_candle(monkeypatch):
    import requests
    from live_guard.model_probability_history import read_market_prices
    class Response:
        def raise_for_status(self):
            pass
        def json(self):
            return [[0,0,0,0,'100',0,3599999], [3600000,0,0,0,'999',0,7199999]]
    def get(url,params,timeout):
        assert params['symbol']=='BTCUSDT'
        assert params['interval']=='1h'
        assert timeout==3
        return Response()
    monkeypatch.setattr(requests,'get',get)
    assert read_market_prices('BTC-USDT',4000)==[{'timestamp':3600,'close':100.0}]

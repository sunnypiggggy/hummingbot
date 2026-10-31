"""Hand economics, common-state and causal tests; no trained models or PNL."""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import torch

from scripts import btceth_rl_execution_band as band
from scripts.btceth_rl_env import BTCETHPortfolioEnv
from scripts.crypto_rl_reporting import result_fingerprints
from test.test_btceth_rl_env import START, dataset, change_price


class FixedModel:
    def __init__(self,actions):
        self.actions=actions;self.policy=torch.nn.Linear(1,1,bias=False)
        with torch.no_grad():self.policy.weight.fill_(.123)
        self.observations=[]

    def predict(self,obs,deterministic):
        assert deterministic is True
        self.observations.append(np.asarray(obs).copy())
        i=(len(self.observations)-1)%len(self.actions)
        return np.asarray(self.actions[i],dtype=float),None


def evaluate(data=None,actions=((.5,0.),),days=2):
    model=FixedModel(actions)
    result,trace=band.evaluate_band(dataset() if data is None else data,[model],[0],band.formal.COSTS['base'],
        start=START,end=START+pd.Timedelta(days=days))
    return result,trace,model


def test_explicit_five_point_constructor_keeps_original_profile_and_entry_costs():
    profiles=deepcopy(band.formal.PROFILES)
    result,trace,model=evaluate(days=1)
    assert band.formal.PROFILES==profiles and profiles['balanced']['rebalance_band']==.025
    assert result['metadata']['rebalance_band']==.05 and result['metadata']['drawdown_penalty']==.25
    gross=.95*.5*.75;buy_cost=1.0002*1.001-1;nav=10000/(1+gross*buy_cost)
    assert result['equity'].equity.iloc[-1]==pytest.approx(nav,abs=1e-9)
    assert result['fills'].fee.sum()==pytest.approx(nav*gross*1.0002*.001)
    assert result['fills'].side.tolist()==['BUY','BUY']
    assert result['fills'].qty.tolist()==pytest.approx([nav*gross/2/100,nav*gross/2/200])
    assert trace[['target_btc','target_eth','target_cash']].iloc[0].to_numpy()==pytest.approx([gross/2,gross/2,1-gross])
    assert result['metadata']['parameter_sha256_before']==result['metadata']['parameter_sha256_after']


def test_change_between_two_point_five_and_five_points_skips_new_trade_by_hand():
    actions=((.5,0.),(.5,.16))
    result,trace,_=evaluate(actions=actions)
    # .35625 gross x .08 ETH-share change = 2.85pp in each coin.
    assert abs(trace.target_btc.iloc[1]-trace.target_btc.iloc[0])==pytest.approx(.0285)
    assert result['weights'].groupby('date').rebalanced.first().tolist()==[True,False]
    assert len(result['fills'])==2
    old=BTCETHPortfolioEnv(dataset(),START,START+pd.Timedelta(days=2),costs=band.formal.COSTS['base'],rebalance_band=.025)
    old.reset()
    for action in actions:old.step_target(band.cap.cap_target(old.action_to_weights(action)))
    reference=old.get_result();old.close()
    assert len(reference['fills'])==4 and reference['fills'].fee.sum()>result['fills'].fee.sum()


def test_material_change_above_five_points_still_executes_both_spot_legs():
    result,trace,_=evaluate(actions=((.5,0.),(.5,.30)))
    assert abs(trace.target_btc.iloc[1]-trace.target_btc.iloc[0])==pytest.approx(.0534375)
    assert result['weights'].groupby('date').rebalanced.first().tolist()==[True,True]
    trades=result['fills'];second=trades.loc[trades.timestamp.dt.normalize().eq(START+pd.Timedelta(days=1))]
    assert second.side.tolist()==['SELL','BUY']
    assert second.timestamp.dt.hour.tolist()==[0,1]
    assert second.asset_key.tolist()==['BTCUSDT','ETHUSDT']


def test_protective_all_cash_exit_remains_even_below_new_band_after_price_fall():
    data=dataset();change_price(data,START+pd.Timedelta(hours=2),btc=1.,eth=2.)
    result,trace,_=evaluate(data,actions=((1.,0.),(-1.,0.)))
    prior=trace.observation_btc_weight.iloc[1]+trace.observation_eth_weight.iloc[1]
    assert 0<prior<.05
    assert result['weights'].groupby('date').rebalanced.first().tolist()==[True,True]
    assert result['fills'].side.tolist()==['BUY','BUY','SELL','SELL']
    assert result['metadata']['final_positions']=={}


def test_five_distinct_models_share_actual_state_and_mean_targets_not_actions():
    actions=[(-1.,0.),(0.,0.),(1.,-1.),(1.,1.),(.5,0.)]
    models=[FixedModel([action]) for action in actions]
    result,trace=band.evaluate_band(dataset(),models,list(range(5)),band.formal.COSTS['base'],start=START,end=START+pd.Timedelta(days=2))
    first=trace.iloc[:5]
    assert first.observation_sha256.nunique()==1
    assert first.target_btc.tolist()==pytest.approx([.178125]*5)
    assert first.target_eth.tolist()==pytest.approx([.178125]*5)
    assert first.target_cash.tolist()==pytest.approx([.64375]*5)
    assert first.target_btc.iloc[0]!=pytest.approx(.95*max(np.mean(actions,axis=0)[0],0)*.75/2)
    for i in range(2):
        for model in models[1:]:np.testing.assert_array_equal(model.observations[i],models[0].observations[i])
    assert models[0].observations[1][-4]>0 and models[0].observations[1][-3]>0
    for row in trace.itertuples(index=False):
        values=np.asarray(json.loads(row.observation_values_json),dtype='<f4')
        assert values.shape==(27,) and hashlib.sha256(values.tobytes()).hexdigest()==row.observation_sha256
        assert pd.Timestamp(row.observation_asof)==pd.Timestamp(row.day)-pd.Timedelta(hours=1)
        assert row.observation_asof==row.feature_input_cutoff
    assert result['metadata']['model_count']==5


def test_exact_repeat_and_future_mutation_leave_prior_account_and_trace_unchanged():
    data=dataset();end=START+pd.Timedelta(days=3)
    truncated={k:v.loc[v.index<end].copy() for k,v in data.items()}
    altered={k:v.copy() for k,v in data.items()}
    for frame in altered.values():frame.loc[frame.index>=end]=99999.
    outcomes=[evaluate(d,actions=((.5,0.),(.5,.16),(.5,.3)),days=3) for d in (data,truncated,altered,data)]
    fingerprints=[result_fingerprints(item[0]) for item in outcomes]
    assert all(value==fingerprints[0] for value in fingerprints)
    for item in outcomes[1:]:pd.testing.assert_frame_equal(item[1],outcomes[0][1])


@pytest.mark.parametrize('seeds',[[0,1],[0,0,2,3,4],[0,1,2,3,5]])
def test_incomplete_or_changed_seed_population_is_rejected(seeds):
    with pytest.raises(ValueError,match='original models'):
        band.evaluate_band({},[FixedModel([(.5,0.)]) for _ in seeds],seeds,band.formal.COSTS['base'])


def candidates(ret=.1,dd=-.1):
    return pd.DataFrame([dict(run_id='synthetic__'+cost,scenario=cost,primary_policy=True,
        net_return=ret,max_drawdown=dd,final_equity=10000*(1+ret),fees=12.) for cost in band.formal.COSTS])


def test_same_declared_two_cost_score_and_exact_tie_prefers_original():
    old=band.candidate(candidates(),'reference_band025',.025)
    new=band.candidate(candidates(),'band05',.05)
    assert old['score']==pytest.approx(math.log1p(.1)-.5*.1)
    ranked=sorted([new,old],key=lambda r:(-r['score'],0 if r['family']=='cap075' else 1))
    assert ranked[0]['family']=='cap075'
    bad=band.candidate(candidates(.2,-.4),'band05',.05)
    assert bad['score']==pytest.approx(math.log1p(.2)-.5*.4-2*(.4-.2))


def test_cli_has_no_annual_phase():
    result=subprocess.run([sys.executable,str(Path(band.__file__)),'--phase','test'],capture_output=True,text=True)
    assert result.returncode==2 and 'invalid choice' in result.stderr


def test_existing_experiment_never_reprepared_or_overwritten(tmp_path,monkeypatch):
    main,reference,out=[tmp_path/n for n in ('main','reference','new')]
    for p in (main,reference,out):p.mkdir()
    monkeypatch.setattr(band.formal,'check_frozen',lambda *a:pytest.fail('Existing output rejected before reading new experiment data'))
    with pytest.raises(FileExistsError):band.prepare(main,reference,tmp_path/'absent-data',out)


def test_annual_files_close_protocol_before_any_model_or_price_loading(tmp_path,monkeypatch):
    # Small fixture checks the explicit gate independently of original models.
    main,reference,out=[tmp_path/n for n in ('main','reference','new')]
    for p in (main,reference,out):p.mkdir()
    (out/'test').mkdir()
    monkeypatch.setattr(band.formal,'check_frozen',lambda *a:pytest.fail('Annual gate must close first'))
    monkeypatch.setattr(band,'evaluate_scenario',lambda *a:pytest.fail('No model allowed for an undeclared phase'))
    with pytest.raises(ValueError,match='no annual phase'):band.verify_protocol(main,reference,tmp_path/'absent-data',out)


def prepared_fixture(tmp_path,monkeypatch):
    main,reference,out,source=[tmp_path/n for n in ('main','reference','new','source')]
    for p in (main,reference,source):p.mkdir()
    dump=lambda p,v:p.write_text(json.dumps(v),encoding='utf-8')
    for name in band.SOURCE_FILES:
        f=source/name;f.parent.mkdir(parents=True,exist_ok=True);f.write_bytes(b'SYNTHETIC_SOURCE_ONLY')
    monkeypatch.setattr(band,'ROOT',source)
    for name in band.reference_files():
        p=reference/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'SYNTHETIC_VALIDATION_REFERENCE_ONLY')
    dump(reference/'independent_audit.json',dict(passed=True,status='passed',run_count=36,test_complete=True,failure_count=0))
    dump(reference/'goal_candidate_evidence.json',dict(audit_sha256=band.sha(reference/'independent_audit.json'),candidate=dict(family='cap075',algorithm='tqc',profile='balanced')))
    for name in ['frozen_spec.json','training_completion.json','fit_process.json','fit.log']:(main/name).write_bytes(b'SYNTHETIC_ORIGINAL_FILE')
    dump(main/'runtime.json',dict(physical_gpus=[dict(uuid='GPU_fixture_a',name='Fixture V100'),dict(uuid='GPU_fixture_b',name='Fixture V100')]))
    models=[]
    for seed in range(5):
        prefix=f'training/tqc__balanced__seed{seed}'
        for name in ['record.json','model.zip']:
            f=main/prefix/name;f.parent.mkdir(parents=True,exist_ok=True);f.write_bytes(b'SYNTHETIC_NOT_A_TRAINED_MODEL')
        models.append(dict(profile='balanced',seed=seed,record_file=prefix+'/record.json',model_file=prefix+'/model.zip',parameter_sha256='a'*64))
    monkeypatch.setattr(band,'models_manifest',lambda *a:models)
    monkeypatch.setattr(band.cap,'verify_protocol',lambda *a:None)
    spec=dict(dataset_sha256='d'*64,dataset_manifest_sha256='e'*64,runtime_packages={},validation_selection={'score':'original fixed score'})
    monkeypatch.setattr(band.formal,'check_frozen',lambda *a:spec)
    monkeypatch.setattr(band.formal,'load_dataset',lambda *a:pytest.fail('Preparing and checking pins must not load prices'))
    band.prepare(main,reference,tmp_path/'unused_dataset',out)
    return main,reference,out,source


def test_prepare_freezes_sources_models_reference_and_seen_year_before_any_prices(tmp_path,monkeypatch):
    main,reference,out,source=prepared_fixture(tmp_path,monkeypatch)
    p=band.verify_protocol(main,reference,tmp_path/'unused_dataset',out)
    assert p['designed_after_first_annual'] is True and p['test_year_already_seen'] is True and p['not_fresh_holdout'] is True
    assert p['annual_evaluation_enabled'] is False and p['new_training']==0
    assert len(p['models'])==5 and len(p['source_sha256'])==10
    assert set(band.reference_files()).issubset(p['original_inputs']['reference'])
    assert not (out/'validation').exists()
    (source/band.SOURCE_FILES[0]).write_bytes(b'CHANGED_SOURCE')
    with pytest.raises(ValueError,match='changed'):band.verify_protocol(main,reference,tmp_path/'unused_dataset',out)


def test_updating_receipt_cannot_turn_one_band_into_parameter_search(tmp_path,monkeypatch):
    main,reference,out,source=prepared_fixture(tmp_path,monkeypatch)
    p=json.loads((out/'protocol.json').read_text());p['new_rebalance_band']=.055
    (out/'protocol.json').write_text(json.dumps(p))
    receipt=json.loads((out/'protocol_receipt.json').read_text());receipt['protocol_sha256']=band.sha(out/'protocol.json')
    (out/'protocol_receipt.json').write_text(json.dumps(receipt))
    with pytest.raises(ValueError,match='single post-annual'):band.verify_protocol(main,reference,tmp_path/'unused_dataset',out)

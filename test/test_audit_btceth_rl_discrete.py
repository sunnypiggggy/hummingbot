"""Invented observations/tensors only: no NN, training or market IO."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import hashlib
import io
import json
import zipfile
import numpy as np
import pytest
from scripts import audit_btceth_rl_discrete as a


def test_exact21_actions_preserve_cash_floor_and_two_original_budget_mappings():
    table=np.array([a.action_to_box(i) for i in range(21)])
    assert len(np.unique(table,axis=0))==21 and table[0].tolist()==[-1.,0.]
    np.testing.assert_array_equal(table[1:],[[g,2*s-1] for g in [.25,.5,.75,1.] for s in [0.,.25,.5,.75,1.]])
    for family,multiplier in a.BUDGETS.items():
        for i in range(21):
            target=a.action_target(i,family)
            assert np.isfinite(target).all() and (target>=0).all() and target[2]>=.05-1e-12
            assert abs(target.sum()-1)<1e-12 and target[:2].sum()<=.95*multiplier+1e-12
    np.testing.assert_allclose(a.action_target(1),[.2375,0.,.7625])
    np.testing.assert_allclose(a.action_target(16),[.95,0.,.05])
    np.testing.assert_allclose(a.action_target(20,'cap075'),[0.,.7125,.2875])


@pytest.mark.parametrize('invalid',[-1,21,1.0,True,np.array([1]),None])
def test_categorical_ids_never_coerced_or_clipped(invalid):
    with pytest.raises(ValueError):a.action_to_box(invalid)


def test_ensemble_means_five_targets_not_ids_or_raw_box_actions():
    ids=[0,16,20,1,5]
    np.testing.assert_allclose(a.ensemble_target(ids),[.2375,.2375,.525])
    np.testing.assert_allclose(a.ensemble_target(ids,'cap075'),[.178125,.178125,.64375])
    boxmean=np.mean([a.action_to_box(i) for i in ids],axis=0)
    assert abs(.95*max(boxmean[0],0)-a.ensemble_target(ids)[:2].sum())>.1
    with pytest.raises(ValueError):a.ensemble_target(ids[:4])


def test_fixed_mean_and_unsorted_low_tau_heads_crossing_are_different_policies():
    q=np.zeros((200,21));q[:50,1]=-10;q[50:,1]=10;q[:,2]=1;q[:50,3]=2;q[50:,3]=-10
    assert a.greedy_action(q,'qrdqn_mean')==1 and a.greedy_action(q,'qrdqn_lower_tail25')==3
    assert np.argmax(np.sort(q,axis=0)[:50].mean(axis=0))==2
    d=a.crossing_diagnostics(q)
    assert d['low_tail_boundary_crossing_count']==1 and d['sorted_outputs_used'] is False and d['cvar_optimality_guarantee'] is False
    assert a.greedy_action(np.zeros((200,21)),'qrdqn_mean')==0 and a.greedy_action(np.zeros(21),'dqn_mean')==0
    a.verify_inference_trace(q,'qrdqn_lower_tail25',a.inference_scores(q,'qrdqn_lower_tail25'),3)
    with pytest.raises(ValueError):a.verify_inference_trace(q,'qrdqn_lower_tail25',a.inference_scores(q,'qrdqn_lower_tail25'),2)


@pytest.mark.parametrize('shape',[(21,200),(199,21),(200,20)])
def test_wrong_quantile_orientation_or_population_rejected(shape):
    with pytest.raises(ValueError):a.inference_scores(np.zeros(shape),'qrdqn_mean')


def test_matched_recipe_and_actual_update_count_handcalculation():
    counts=a.expected_update_counts()
    assert counts['optimizer_steps']==(32768-1024)//4==7936 and counts['target_sync_calls']==32768//512==64
    assert counts['completed_episodes']==1024 and counts['unused_zero_step_resets']==1
    qr,dqn=a.matched_recipe('qrdqn'),a.matched_recipe('dqn')
    assert qr['policy_kwargs'].pop('n_quantiles')==200 and qr==dqn
    assert qr['policy_kwargs']['optimizer_class']=='Adam' and qr['policy_kwargs']['optimizer_kwargs']=={'eps':1e-8}
    a.validate_observed_counts(counts)
    for key,value in [('optimizer_steps',7935),('target_sync_calls',63),('unused_zero_step_resets',True)]:
        with pytest.raises(ValueError):a.validate_observed_counts({**counts,key:value})


def timeout_rows():
    rows=[]
    for episode in range(1,1025):
        obs=np.full(27,episode,dtype=np.float32);terminal=obs+1
        row=dict(transition_number=episode*32,replay_position=episode*32-1,env_index=0,action_id=episode%21,
            stored_done=1.,stored_timeout=1.,effective_done_from_stored_flags=0.,handle_timeout_termination=True,
            terminal_observation_present=True,stored_next_matches_terminal=True,terminated_from_stored_flags=False,
            transition_chain_sha256=hashlib.sha256(str(episode).encode()).hexdigest(),**{'TimeLimit.truncated':True})
        for key,values in [('stored_observation',obs),('terminal_observation',terminal),('stored_next_observation',terminal)]:
            row[key+'_values']=values.tolist();row[key+'_sha256']=a.observation_digest(values)
        rows.append(row)
    return rows


def test_all1024_timeouts_retain_true_terminal_and_bootstrap_mask():
    result=a.audit_timeout_rows(timeout_rows())
    assert result['timeout_rows']==1024 and result['all_terminal_next_observations_verified'] is True


def test_chain_hashes_stored_categorical_int64_and_float32_arrays_without_rng():
    first=hashlib.sha256(b'').hexdigest();obs=np.zeros(27,dtype='<f4');nxt=np.ones(27,dtype='<f4')
    expected=hashlib.sha256(bytes.fromhex(first)+obs.tobytes()+nxt.tobytes()+np.array([20],dtype='<i8').tobytes()+np.array([.5,1.,1.],dtype='<f4').tobytes()).hexdigest()
    assert a.extend_transition_chain(first,obs,nxt,20,.5,1.,1.)==expected
    assert a.extend_transition_chain(first,obs,nxt,19,.5,1.,1.)!=expected
    assert a.extend_transition_chain(expected,obs,nxt,20,.5,1.,1.)!=expected


@pytest.mark.parametrize('damage',['reset_next','terminal_done','boxed_action','wrong_position','missing_row','repeated_chain'])
def test_native_replay_failure_cases_are_not_silently_treated_as_success(damage):
    rows=timeout_rows()
    if damage=='reset_next':
        rows[0]['stored_next_observation_values']=[0.]*27;rows[0]['stored_next_observation_sha256']=a.observation_digest([0.]*27)
    elif damage=='terminal_done':rows[0]['effective_done_from_stored_flags']=1.
    elif damage=='boxed_action':rows[0]['action_id']=[.25,-1.]
    elif damage=='wrong_position':rows[0]['replay_position']=0
    elif damage=='missing_row':rows.pop()
    else:rows[1]['transition_chain_sha256']=rows[0]['transition_chain_sha256']
    with pytest.raises(ValueError):a.audit_timeout_rows(rows)


def history():
    start=datetime(2020,1,1,tzinfo=timezone.utc);end=start+timedelta(days=32)
    row=dict(start=start.isoformat(),end_exclusive=end.isoformat(),completed_until=end.isoformat(),completed_steps=32,terminated=False,truncated=True)
    rows=[deepcopy(row) for _ in range(1024)]
    rows.append({**row,'completed_until':start.isoformat(),'completed_steps':0,'truncated':False})
    return rows


def test_completed_train_calendar_and_the_single_unused_reset_are_distinct():
    rows=history();assert a.verify_training_history(rows,eligible_starts=[rows[0]['start']])['completed_steps']==32768
    for change in ['missing','trained_unused','economic_terminal','outside','ineligible']:
        bad=deepcopy(rows)
        if change=='missing':bad.pop()
        elif change=='trained_unused':bad[-1]['completed_steps']=1
        elif change=='economic_terminal':bad[0]['terminated']=True
        elif change=='outside':bad[0]['start']='2025-01-01T00:00:00Z'
        with pytest.raises(ValueError):a.verify_training_history(bad,eligible_starts=[] if change=='ineligible' else [rows[0]['start']])


@pytest.fixture
def tensors():
    import torch
    previous=torch.get_num_threads();torch.set_num_threads(1)
    def make(algorithm):
        shapes=a.expected_policy_shapes(algorithm);policy={key:torch.zeros(shape,dtype=torch.float32) for key,shape in shapes.items()}
        prefix='quantile_net' if algorithm=='qrdqn' else 'q_net';online=[name for name in shapes if name.startswith(prefix+'.')]
        state={i:dict(step=torch.tensor(7936.,dtype=torch.float32),exp_avg=torch.zeros(shapes[name]),exp_avg_sq=torch.zeros(shapes[name])) for i,name in enumerate(online)}
        optimizer=dict(state=state,param_groups=[dict(params=list(range(6)),lr=1e-4,eps=1e-8,betas=(.9,.999),weight_decay=0,amsgrad=False)])
        return policy,optimizer,dict(num_timesteps=32768,_n_updates=7936,_n_calls=32768,policy_class={':serialized:':'INERT NOT DECODED'})
    yield make
    torch.set_num_threads(previous)


@pytest.mark.parametrize('algorithm',['dqn','qrdqn'])
def test_inert_online_target_weights_and_six_real_adam_steps(tensors,algorithm):
    policy,optimizer,data=tensors(algorithm);value=a.inspect_state_dicts(policy,optimizer,data,algorithm)
    assert value['policy_tensor_count']==12 and value['actual_optimizer_steps']==[7936]*6


@pytest.mark.parametrize('damage',['wrong_head_shape','missing_target','nonfinite','fake_step','wrong_eps','missing_moment','wrong_moment_shape','fake_json_count'])
def test_weight_and_optimizer_counter_forgery_rejected(tensors,damage):
    import torch
    policy,optimizer,data=tensors('qrdqn')
    if damage=='wrong_head_shape':policy['quantile_net.quantile_net.4.weight']=torch.zeros((21,128))
    elif damage=='missing_target':policy.pop('quantile_net_target.quantile_net.4.bias')
    elif damage=='nonfinite':policy['quantile_net.quantile_net.0.bias'][0]=float('nan')
    elif damage=='fake_step':optimizer['state'][0]['step']=7936
    elif damage=='wrong_eps':optimizer['param_groups'][0]['eps']=.01/128
    elif damage=='missing_moment':optimizer['state'][0].pop('exp_avg_sq')
    elif damage=='wrong_moment_shape':optimizer['state'][0]['exp_avg']=torch.zeros(27)
    else:data['_n_updates']=7935
    with pytest.raises(ValueError):a.inspect_state_dicts(policy,optimizer,data,'qrdqn')


def make_zip(path,tensors,algorithm='dqn',extra=None):
    import torch
    policy,optimizer,data=tensors(algorithm)
    with zipfile.ZipFile(path,'w') as z:
        z.writestr('data',json.dumps(data));z.writestr('_stable_baselines3_version','2.9.0');z.writestr('system_info.txt','SYNTHETIC NO NN')
        for name,value in [('policy.pth',policy),('policy.optimizer.pth',optimizer),('pytorch_variables.pth',{})]:
            stream=io.BytesIO();torch.save(value,stream);z.writestr(name,stream.getvalue())
        if extra:z.writestr(extra,b'NOT EXTRACTED')


def test_weights_only_zip_reads_no_python_objects_or_neural_network(tensors,tmp_path):
    path=tmp_path/'synthetic.zip';make_zip(path,tensors);value=a.inspect_sb3_zip(path,'dqn')
    assert value['tensor_payload_only'] is True and len(value['model_sha256'])==64


def test_zip_traversal_and_extra_payloads_rejected_before_any_extraction(tensors,tmp_path):
    path=tmp_path/'synthetic.zip';make_zip(path,tensors,extra='../unexpected')
    with pytest.raises(ValueError):a.inspect_sb3_zip(path,'dqn')
    assert not (tmp_path.parent/'unexpected').exists()


def candidate_pool():
    return [dict(variant=variant,family=family,profile='balanced',base=dict(net_return=.1,max_drawdown=-.1),stress=dict(net_return=.05,max_drawdown=-.1)) for variant in a.VARIANTS for family in a.BUDGETS]


def test_fixed_six_pool_raw_first_lex_tie_and_all_three_preannual_variant_budgets():
    result=a.rank_candidate_pool(candidate_pool())
    assert len(result['candidates'])==6 and result['global_primary']['variant']=='dqn_mean' and result['global_primary']['family']=='raw'
    assert all(v['family']=='raw' for v in result['per_variant'].values())
    assert abs(result['global_primary']['score']-((np.log1p(.1)+np.log1p(.05))/2-.05))<1e-14
    with pytest.raises(ValueError):a.rank_candidate_pool(candidate_pool()[:-1])
    bad=candidate_pool();bad[0]['score']=10
    with pytest.raises(ValueError):a.rank_candidate_pool(bad)


def test_full_population72_validation_and36_annual_plus14_references_remain_distinct():
    validation=a.case_population('validation')
    annual=a.case_population('test',{'qrdqn_mean':'raw','qrdqn_lower_tail25':'cap075','dqn_mean':'raw'})
    assert len(validation)==72 and len(annual)==36 and len(annual)+14==50
    assert len({tuple(row.values()) for row in validation})==72
    with pytest.raises(ValueError):a.case_population('test',{'dqn_mean':'raw'})
    with pytest.raises(ValueError):a.case_population('validation',{'dqn_mean':'raw'})


def test_cash0_is_not_goal_success_and_no_seed_subset_or_missing_drawdown():
    cash=dict(net_return=0.,max_drawdown=0.);equal=dict(net_return=-.2,max_drawdown=-.4);seeds=[dict(seed=seed,net_return=0.) for seed in range(5)]
    result=a.primary_targets(cash,cash,seeds,equal)
    assert result['goal_performance_achieved'] is False and result['positive_base_training_seeds']==0
    with pytest.raises(ValueError):a.primary_targets(cash,cash,seeds[:4],equal)
    with pytest.raises(ValueError):a.primary_targets({**cash,'max_drawdown':float('nan')},cash,seeds,equal)

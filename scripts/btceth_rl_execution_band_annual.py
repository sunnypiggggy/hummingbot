"""Separately authorized annual research of the validation-selected 5pp band.

Source preparation is not permission to evaluate. An explicit authorization
file, the sealed validation winner and post-selection audit must precede a new
protocol. The already-seen annual window is never called a fresh holdout.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

for _key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[_key]='1'
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import pandas as pd
from scripts import btceth_rl_execution_band as band
from scripts import btceth_rl_execution_band_gpu_guard as resource
from scripts.crypto_rl_reporting import canonical_csv,result_fingerprints,save_result,summarize

formal=band.formal
FLAGS=('designed_after_first_annual','test_year_already_seen','not_fresh_holdout')
SOURCE_FILES=(*band.SOURCE_FILES,'scripts/btceth_rl_execution_band_gpu_guard.py',
 'scripts/audit_btceth_rl_execution_band_gpu_guard.py','scripts/btceth_rl_execution_band_annual.py',
 'scripts/audit_btceth_rl_execution_band_annual.py')
now,sha,read_json,relative,write_new=band.now,band.sha,band.read_json,band.relative,band.write_new


def authorization_gate(value,selection_sha,receipt):
    if (value.get('annual_evaluation_authorized') is not True or value.get('selection_sha256')!=selection_sha
        or value.get('financial_parameters_fixed') is not True or value.get('no_live_trading') is not True
        or any(value.get(k) is not True for k in FLAGS)
        or band.goal.stamp(value['authorized_at'])<band.goal.stamp(receipt['sealed_at'])):
        raise ValueError('Explicit post-selection research-only annual authorization is required')


def validation_gate(validation,main,reference,dataset,audit_file):
    validation=Path(validation)
    p=band.verify_protocol(main,reference,dataset,validation)
    s=read_json(validation/'selection.json');r=read_json(validation/'selection_receipt.json');a=read_json(relative(validation,audit_file))
    if (r.get('selection_file')!='selection.json' or r.get('selection_sha256')!=sha(validation/'selection.json')
        or r.get('protocol_sha256')!=sha(validation/'protocol.json')
        or s.get('candidate',{}).get('family')!=band.FAMILY or s['candidate'].get('name')!='band05'
        or s['candidate'].get('rebalance_band')!=.05 or s['candidate'].get('algorithm')!='tqc' or s['candidate'].get('profile')!='balanced'
        or any(s.get(k) is not True for k in FLAGS) or s.get('seed_selection') is not False or s.get('selection_uses_test') is not False
        or a.get('passed') is not True or a.get('status')!='passed' or a.get('selection_complete') is not True
        or a.get('resource_adapter_disclosed') is not True or a.get('run_count')!=12 or a.get('comparator_run_count')!=12
        or a.get('repeat_count')!=12 or a.get('failure_count')!=0 or a.get('test_complete') is not False
        or band.goal.stamp(a['audited_at'])<band.goal.stamp(r['sealed_at'])):
        raise ValueError('The new band must win the sealed validation comparison and pass post-selection auditing')
    roots={'extension':validation,'main':Path(main),'reference':Path(reference),'source':ROOT}
    for kind,values in a['input_sha256'].items():band.verify_pins(roots[kind],values)
    return p,s,r,a


def prepare(validation,main,reference,dataset,output,authorization,audit_file='independent_audit.validation.postselection.json'):
    validation,main,reference,dataset,output=(Path(p).resolve() for p in (validation,main,reference,dataset,output))
    if output.exists():raise FileExistsError('Preserve existing annual research attempts')
    band.separate(main,reference,output)
    if output==validation or output.is_relative_to(validation) or validation.is_relative_to(output):raise ValueError('Annual output must be a separate sibling study')
    v,s,receipt,a=validation_gate(validation,main,reference,dataset,audit_file)
    auth=read_json(authorization);authorization_gate(auth,sha(validation/'selection.json'),receipt)
    spec=formal.check_frozen(main,dataset)
    source=band.pin(ROOT,SOURCE_FILES)
    for name,expected in v['source_sha256'].items():
        if source[name]!=expected:raise ValueError('Original validation financial source changed')
    adapter=read_json(validation/'adapter_protocol.json')
    source_guard={k:source[k] for k in resource.SOURCES}
    if source_guard!=adapter['source_sha256']:raise ValueError('Reviewed resource-adapter source changed')
    val_names=set(a['input_sha256']['extension'])|{audit_file,'selection.json','selection_receipt.json','comparison.json'}
    main_names=set(a['input_sha256']['main'])|{'independent_audit.json','validation_selection.json'}
    baseline='evaluation/test/baselines'
    main_names|={f'{baseline}/{f}' for f in band.GROUP_FILES}
    main_names|={f'{baseline}/{c}/{name}/{f}' for c in formal.COSTS for name in formal.BASELINES for f in band.CASE_FILES[:6]}
    ref_names=set(a['input_sha256']['reference'])|{'independent_audit.json','goal_candidate_evidence.json'}
    ref_names|={f'test/balanced/{f}' for f in band.GROUP_FILES}
    ref_names|={f'test/balanced/{c}/{key}/{f}' for c in formal.COSTS for key in (*map(str,formal.SEEDS),'ensemble') for f in band.CASE_FILES}
    p=dict(schema_version=1,prepared_at=now(),family=band.FAMILY,algorithm='tqc',profile='balanced',
        test_start=formal.TEST[0],end_exclusive=formal.TEST[1],capital=10000.,costs=formal.COSTS,seeds=list(formal.SEEDS),
        cash_floor=.05,max_asset_weight=.95,drawdown_penalty=.25,rebalance_band=.05,exposure_multiplier=.75,
        new_training=0,seed_selection=False,selection_uses_test=False,annual_evaluation_enabled=True,
        **{flag:True for flag in FLAGS},source_sha256=source,
        original_inputs={'validation':band.pin(validation,val_names),'main':band.pin(main,main_names),'reference':band.pin(reference,ref_names)},
        models=v['models'],dataset_sha256=v['dataset_sha256'],dataset_manifest_sha256=v['dataset_manifest_sha256'],
        physical_gpus=v['physical_gpus'],gpu_cost_assignments=v['gpu_cost_assignments'],runtime_packages=v['runtime_packages'],
        research_targets=spec['research_targets'],validation_audit_file=audit_file,
        validation_protocol_sha256=sha(validation/'protocol.json'),validation_selection_sha256=sha(validation/'selection.json'),
        validation_selection_receipt_sha256=sha(validation/'selection_receipt.json'),validation_audit_sha256=sha(relative(validation,audit_file)),
        original_frozen_spec_sha256=sha(main/'frozen_spec.json'),authorization_sha256=sha(Path(authorization)),
        allowed_instance=adapter['allowed_instance'],allowed_gpu_rows=adapter['allowed_gpu_rows'],
        test_cases=12,repeat_count=12,annual_has_not_run_at_preparation=True,
        scope='Post-first-annual finite execution-band follow-up; new account, original models, no live trading and not a fresh holdout')
    output.mkdir(parents=True)
    for name in source:
        target=output/'sources'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,target)
    shutil.copyfile(authorization,output/'authorization.json')
    write_new(output/'protocol.json',p)
    write_new(output/'protocol_receipt.json',dict(protocol_file='protocol.json',protocol_sha256=sha(output/'protocol.json'),sealed_at=now()))
    return p


def verify_protocol(validation,main,reference,dataset,output):
    output=Path(output);p=read_json(output/'protocol.json');r=read_json(output/'protocol_receipt.json')
    v,s,receipt,a=validation_gate(validation,main,reference,dataset,p['validation_audit_file'])
    auth=read_json(output/'authorization.json');authorization_gate(auth,sha(Path(validation)/'selection.json'),receipt)
    if (r.get('protocol_file')!='protocol.json' or r.get('protocol_sha256')!=sha(output/'protocol.json')
        or p.get('authorization_sha256')!=sha(output/'authorization.json') or p.get('rebalance_band')!=.05
        or p.get('exposure_multiplier')!=.75 or p.get('family')!=band.FAMILY or p.get('algorithm')!='tqc' or p.get('profile')!='balanced'
        or p.get('costs')!=formal.COSTS or p.get('capital')!=10000. or p.get('seeds')!=list(formal.SEEDS)
        or p.get('cash_floor')!=.05 or p.get('max_asset_weight')!=.95 or p.get('drawdown_penalty')!=.25
        or p.get('annual_evaluation_enabled') is not True or p.get('new_training')!=0
        or p.get('seed_selection') is not False or p.get('selection_uses_test') is not False
        or any(p.get(flag) is not True for flag in FLAGS) or p.get('source_sha256')!=band.pin(ROOT,SOURCE_FILES)
        or p.get('models')!=v['models'] or p.get('test_start')!=formal.TEST[0] or p.get('end_exclusive')!=formal.TEST[1]
        or p.get('dataset_sha256')!=v['dataset_sha256'] or p.get('dataset_manifest_sha256')!=v['dataset_manifest_sha256']
        or p.get('physical_gpus')!=v['physical_gpus'] or p.get('gpu_cost_assignments')!=v['gpu_cost_assignments']
        or p.get('runtime_packages')!=v['runtime_packages'] or p.get('original_frozen_spec_sha256')!=sha(Path(main)/'frozen_spec.json')
        or p.get('validation_protocol_sha256')!=sha(Path(validation)/'protocol.json')
        or p.get('validation_selection_sha256')!=sha(Path(validation)/'selection.json')
        or p.get('validation_selection_receipt_sha256')!=sha(Path(validation)/'selection_receipt.json')
        or p.get('validation_audit_sha256')!=sha(relative(validation,p['validation_audit_file']))
        or p.get('research_targets')!=read_json(Path(main)/'frozen_spec.json')['research_targets']
        or p.get('test_cases')!=12 or p.get('repeat_count')!=12
        or p.get('allowed_instance')!=read_json(Path(validation)/'adapter_protocol.json')['allowed_instance']
        or p.get('allowed_gpu_rows')!=read_json(Path(validation)/'adapter_protocol.json')['allowed_gpu_rows']):
        raise ValueError('Annual authorization, sealed sources or fixed financial rules changed')
    if not band.goal.stamp(auth['authorized_at'])<=band.goal.stamp(p['prepared_at'])<=band.goal.stamp(r['sealed_at']):raise ValueError('Authorization must precede preparation and seal')
    band.verify_pins(output/'sources',p['source_sha256'])
    for kind,path in [('validation',validation),('main',main),('reference',reference)]:band.verify_pins(path,p['original_inputs'][kind])
    return p


def evaluate_scenario(task):
    p=verify_protocol(task['validation'],task['main'],task['reference'],task['dataset'],task['output'])
    device=formal.torch_setup(123456)
    if device['visible_gpu_uuid']!=task['uuid'] or device['capability']!=[7,0] or formal.runtime_packages()!=p['runtime_packages']:raise ValueError('Original isolated V100 and runtime required')
    data=formal.load_dataset(task['dataset'])
    for name in list(data):
        if name.startswith('hourly_'):data[name]=data[name].loc[data[name].index<pd.Timestamp(formal.TEST[1])].copy()
    models=[]
    for record in p['models']:
        model=formal.model_class('tqc').load(relative(task['main'],record['model_file']),device='cuda:0')
        if formal.parameter_digest(model.policy)!=record['parameter_sha256']:raise ValueError('Loaded policy differs from fixed original')
        models.append(model)
    cost=task['scenario'];costs=formal.COSTS[cost];output=Path(task['output']);folder=output/'test/balanced'/cost
    folder.mkdir(parents=True,exist_ok=False);rows=[];histories=[];months=[];proofs=[]
    plans=[(str(seed),[model],[seed],seed) for seed,model in zip(formal.SEEDS,models)]+[('ensemble',models,list(formal.SEEDS),None)]
    for key,members,seeds,seed in plans:
        run_id=f'test__{cost}__{band.FAMILY}__tqc__balanced__{key}'
        result,trace=band.evaluate_band(data,members,seeds,costs,start=formal.TEST[0],end=formal.TEST[1])
        repeat,again_trace=band.evaluate_band(data,members,seeds,costs,start=formal.TEST[0],end=formal.TEST[1])
        fingerprints,again=result_fingerprints(result),result_fingerprints(repeat);trace_bytes,again_bytes=canonical_csv(trace),canonical_csv(again_trace)
        if fingerprints!=again or trace_bytes!=again_bytes:raise AssertionError('Exact annual repeat failed')
        case=folder/key;save_result(case,result);(case/'member_targets.csv').write_bytes(trace_bytes)
        proof=dict(run_id=run_id,passed=True,repeat_exact=True,fingerprints=fingerprints,repeated_fingerprints=again,
            member_targets_sha256=__import__('hashlib').sha256(trace_bytes).hexdigest(),repeated_member_targets_sha256=__import__('hashlib').sha256(again_bytes).hexdigest(),
            parameter_sha256_before=result['metadata']['parameter_sha256_before'],parameter_sha256_after=result['metadata']['parameter_sha256_after'],
            model_records=[r for r in p['models'] if r['seed'] in seeds],device=device,worker_pid=os.getpid())
        write_new(case/'replay.json',proof)
        row,daily,monthly=summarize(result,run_id=run_id,strategy='tqc_cap075_band05',seed=seed,scenario=cost,split='test',start=formal.TEST[0],end=formal.TEST[1],capital=10000.,**costs)
        row.update(algorithm='tqc',profile='balanced',family=band.FAMILY,primary_policy=seed is None,exposure_multiplier=.75,rebalance_band=.05)
        daily['profile'],monthly['profile']='balanced','balanced';rows.append(row);histories.append(daily);months.append(monthly);proofs.append(proof)
        print(f'ANNUAL_BAND05 {run_id} return={row["net_return"]:.6%} DD={row["max_drawdown"]:.6%}',flush=True)
    for name,frame in [('summary.csv',pd.DataFrame(rows)),('daily_equity.csv',pd.concat(histories,ignore_index=True)),('monthly_returns.csv',pd.concat(months,ignore_index=True))]:frame.to_csv(folder/name,index=False,float_format='%.17g')
    write_new(folder/'completion.json',dict(at=now(),protocol_sha256=sha(output/'protocol.json'),case_count=6,repeats=proofs,worker_pid=os.getpid(),device=device))
    return dict(scenario=cost,case_count=6,worker_pid=os.getpid())


def run_test(validation,main,reference,dataset,output):
    p=verify_protocol(validation,main,reference,dataset,output);output=Path(output)
    if (output/'test').exists() or (output/'test_completion.json').exists():raise FileExistsError('Preserve existing annual attempt')
    marker=read_json(output/'test_started.json')
    if marker.get('protocol_sha256')!=sha(output/'protocol.json') or marker.get('authorization_sha256')!=p['authorization_sha256'] or marker.get('validation_selection_sha256')!=p['validation_selection_sha256']:raise ValueError('Annual marker is not bound to authorization and sealed selection')
    if band.goal.stamp(marker['at'])<band.goal.stamp(read_json(output/'protocol_receipt.json')['sealed_at']):raise ValueError('Annual marker predates frozen protocol')
    if sys.platform!='linux':raise RuntimeError('Use original two V100s on Linux')
    before=resource.snapshot();resource.verify_resource_snapshot(before,p)
    ctx=mp.get_context('spawn');pools=[];futures=[]
    try:
        for scenario,uuid in p['gpu_cost_assignments'].items():
            pool=ProcessPoolExecutor(max_workers=1,mp_context=ctx,initializer=formal.gpu_initializer,initargs=(uuid,));pools.append(pool)
            futures.append(pool.submit(evaluate_scenario,dict(validation=str(validation),main=str(main),reference=str(reference),dataset=str(dataset),output=str(output),scenario=scenario,uuid=uuid)))
        workers=[f.result() for f in futures]
    finally:
        for pool in pools:pool.shutdown(wait=True)
    after=resource.snapshot();resource.verify_resource_snapshot(after,p)
    write_new(output/'resource_runtime.json',dict(at=now(),status='completed',before=before,after=after,financial_rules_changed=False,error=None))
    folder=output/'test/balanced'
    for name in ['summary.csv','daily_equity.csv','monthly_returns.csv']:
        frame=pd.concat([pd.read_csv(folder/c/name,float_precision='round_trip') for c in formal.COSTS],ignore_index=True)
        frame.to_csv(folder/name,index=False,float_format='%.17g');frame.to_csv(output/name,index=False,float_format='%.17g')
    proofs=[r for c in formal.COSTS for r in read_json(folder/c/'completion.json')['repeats']]
    write_new(folder/'completion.json',dict(at=now(),protocol_sha256=sha(output/'protocol.json'),case_count=12,repeats=proofs))
    verify_protocol(validation,main,reference,dataset,output)
    write_new(output/'test_completion.json',dict(at=now(),protocol_sha256=sha(output/'protocol.json'),case_count=12,repeat_count=12,workers=workers,split='test'))
    return dict(case_count=12,repeat_count=12,workers=workers)


def supervise(validation,main,reference,dataset,output):
    p=verify_protocol(validation,main,reference,dataset,output);output=Path(output)
    if any((output/n).exists() for n in ['test_started.json','test_process.json','runner_process.json','test.log','test']):raise FileExistsError('Preserve any prior annual dispatch')
    started=now()
    write_new(output/'test_started.json',dict(at=now(),protocol_sha256=sha(output/'protocol.json'),authorization_sha256=p['authorization_sha256'],validation_selection_sha256=p['validation_selection_sha256'],supervisor_pid=os.getpid()))
    command=[sys.executable,'-u',str(Path(__file__).resolve()),'--phase','_test','--validation-output',str(validation),'--main-output',str(main),'--reference-output',str(reference),'--dataset',str(dataset),'--output-dir',str(output)]
    proof=output/'test_process.json';log=output/'test.log'
    r=dict(phase='test',status='launching',started_at=started,supervisor_pid=os.getpid(),command=command,protocol_sha256=sha(output/'protocol.json'),source_sha256=sha(Path(__file__)),log_relative_path='test.log')
    def publish():
        for name in ['test_process.json','runner_process.json']:
            q=output/name;t=q.with_suffix('.tmp');t.write_text(json.dumps(r,indent=2)+'\n');t.replace(q)
    publish()
    with log.open('x',encoding='utf-8') as stream:
        child=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT);r.update(status='running',child_pid=child.pid);publish();code=child.wait()
    r.update(status='completed' if code==0 else 'failed',exit_code=code,completed_at=now(),log_sha256=sha(log));publish()
    if code:raise RuntimeError('Preserve failed annual attempt')
    return r


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--phase',required=True,choices=('prepare','test','_test'))
    for name in ('validation-output','main-output','reference-output','dataset','output-dir'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--authorization',type=Path);p.add_argument('--validation-audit-file',default='independent_audit.validation.postselection.json')
    a=p.parse_args();args=[getattr(a,n).resolve() for n in ('validation_output','main_output','reference_output','dataset','output_dir')]
    if a.phase=='prepare':
        if a.authorization is None:p.error('An explicit authorized file is required; source preparation is not permission')
        result=prepare(*args,a.authorization,a.validation_audit_file)
    elif a.phase=='test':result=supervise(*args)
    else:
        for _ in range(20):
            r=read_json(args[-1]/'test_process.json')
            if r.get('child_pid')==os.getpid():break
            time.sleep(.05)
        if r.get('child_pid')!=os.getpid() or r.get('supervisor_pid')!=os.getppid() or r.get('status')!='running':raise ValueError('Actual annual child must match its supervisor')
        result=run_test(*args)
    print(json.dumps({'phase':a.phase,'result':result}),flush=True)


if __name__=='__main__':main()

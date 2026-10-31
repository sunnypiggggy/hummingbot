"""Transparent instance-only GPU resource adapter; frozen financial core reused.

The existing failed attempt and original c30/8f5/b4d bytes are preserved.
Only a verified sunny ptyxis instance is removed from one resource query.
All actual command lines and unfiltered nvidia-smi output are retained.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from scripts import btceth_rl_execution_band as core

CORE_SHA='8f5c4583b8ddd4da279cbbb9bae14c49d4cde4e762558ed4b007160a562a9fc8'
AUDIT_SHA='b4d699bc57b22a1b436491bd501e4ed7af784834e8156f6a0042956f5eb6285f'
PROTOCOL_SHA='c30dc8139948ffc9e1d7d201d9a5b163cfb7d63afae6670990aa227ee21ac405'
RECEIPT_SHA='befce2d2ff3009b7e6c3779eae219367bb6a8222bf392d52642bfa1ad75dde82'
FAILURES={'runner_process.json':'b3dff3e641a0f6867b975a484a06df053fc30517341beab183fc8dd990d37473',
 'validation.log':'cfdc886dab07f543275441481c86e3ba8dc3283489a90a9697a252a65b3ca12c',
 'preflight/launch.log':'53ac21d31d1ce2081cf97d9e4d7bf23698f1d1544fbaabe523c8b08d71287020'}
SOURCES=('scripts/btceth_rl_execution_band.py','scripts/audit_btceth_rl_execution_band.py',
 'scripts/btceth_rl_execution_band_gpu_guard.py','scripts/audit_btceth_rl_execution_band_gpu_guard.py')
QUERY=['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader']
now,sha,read_json,write_new=core.now,core.sha,core.read_json,core.write_new


def instance_info(pid=715236):
    import pwd
    p=Path('/proc')/str(pid)
    uid=int(next(x for x in (p/'status').read_text().splitlines() if x.startswith('Uid:')).split()[1])
    value=dict(pid=pid,uid=uid,username=pwd.getpwuid(uid).pw_name,exe=os.readlink(p/'exe'),
        starttime_ticks=(p/'stat').read_text().rsplit(')',1)[1].split()[19],
        boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())
    if pid!=715236 or value['username']!='sunny' or uid!=os.getuid() or Path(value['exe']).name!='ptyxis':
        raise ValueError('Only the explicitly reviewed sunny ptyxis instance is eligible')
    return value


def snapshot(run=subprocess.run):
    argv=['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_memory','--format=csv,noheader']
    result=run(argv,capture_output=True,text=True,check=True)
    gpu=['nvidia-smi','--query-gpu=index,uuid,name,memory.used,utilization.gpu','--format=csv,noheader']
    status=run(gpu,capture_output=True,text=True,check=True)
    return dict(at=now(),instance=instance_info(),argv=argv,stdout=result.stdout,
        gpu_status_argv=gpu,gpu_status_stdout=status.stdout)


def filter_rows(stdout,allowed):
    """Delete only the exact declared UUID/PID line; every other byte stays."""
    allowed={(r['gpu_uuid'],int(r['pid'])) for r in allowed}
    kept=[]
    for line in stdout.splitlines(keepends=True):
        fields=next(csv.reader([line]))
        try:key=(fields[0].strip(),int(fields[1].strip()))
        except (ValueError,IndexError):key=None
        if key not in allowed:kept.append(line)
    return ''.join(kept)


def verify_resource_snapshot(value,protocol):
    if value['instance']!=protocol['allowed_instance']:raise ValueError('Reviewed terminal instance changed')
    allowed={(r['gpu_uuid'],int(r['pid'])) for r in protocol['allowed_gpu_rows']}
    seen=set()
    for row in csv.reader(value['stdout'].splitlines()):
        if len(row)!=4:raise ValueError('Unexpected real GPU process output')
        key=(row[0].strip(),int(row[1].strip()));seen.add(key)
        if key not in allowed or Path(row[2].strip()).name!='ptyxis':raise ValueError('Other GPU task remains forbidden')
    if seen!=allowed:raise ValueError('Single reviewed GPU instance must remain present')


@contextmanager
def resource_adapter(protocol,events):
    original=subprocess.run
    def run(argv,*args,**kwargs):
        result=original(argv,*args,**kwargs)
        if list(argv)==QUERY:
            observed=instance_info()
            if observed!=protocol['allowed_instance']:raise ValueError('Allowed PID was reused or its identity changed')
            filtered=filter_rows(result.stdout,protocol['allowed_gpu_rows'])
            events.append(dict(at=now(),argv=list(argv),returncode=result.returncode,
                stdout_original=result.stdout,stdout_filtered=filtered,instance_observed=observed))
            return subprocess.CompletedProcess(result.args,result.returncode,filtered,result.stderr)
        return result
    subprocess.run=run
    try:yield original
    finally:subprocess.run=original


def source_pins():
    values=core.pin(ROOT,SOURCES)
    if values[SOURCES[0]]!=CORE_SHA or values[SOURCES[1]]!=AUDIT_SHA:raise ValueError('Frozen core/auditor changed')
    return values


def prepare(original,main,reference,dataset,attempt):
    original,attempt=Path(original).resolve(),Path(attempt).resolve()
    attempt.relative_to(original/'attempts')
    if attempt.exists() or (original/'validation').exists():raise FileExistsError('Preserve original failure and any existing attempt')
    core.verify_protocol(main,reference,dataset,original)
    if sha(original/'protocol.json')!=PROTOCOL_SHA or sha(original/'protocol_receipt.json')!=RECEIPT_SHA:raise ValueError('Original protocol changed')
    for name,expected in FAILURES.items():
        if sha(original/name)!=expected:raise ValueError('Original failure changed: '+name)
    failure=read_json(original/'runner_process.json')
    if failure['status']!='failed' or failure['exit_code']!=1:raise ValueError('Require the retained pre-inference failure')
    before=snapshot();identity=before['instance'];allowed=[]
    for row in csv.reader(before['stdout'].splitlines()):
        if len(row)!=4:raise ValueError('Unexpected nvidia-smi resource row')
        uuid,pid,name,memory=[x.strip() for x in row]
        if int(pid)!=715236 or Path(name).name!='ptyxis':raise ValueError('Another compute task exists; do not interrupt it')
        allowed.append(dict(gpu_uuid=uuid,pid=int(pid)))
    if len(allowed)!=1:raise ValueError('The single approved terminal instance must be observed')
    sources=source_pins();attempt.mkdir(parents=True)
    for name in ('protocol.json','protocol_receipt.json'):shutil.copyfile(original/name,attempt/name)
    shutil.copytree(original/'sources',attempt/'sources')
    for name in FAILURES:
        target=attempt/'original_failure'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(original/name,target)
    for name in sources:
        target=attempt/'adapter_sources'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,target)
    p=dict(prepared_at=now(),original_protocol_sha256=PROTOCOL_SHA,original_protocol_receipt_sha256=RECEIPT_SHA,
        original_failed_files=FAILURES,source_sha256=sources,allowed_instance=identity,allowed_gpu_rows=allowed,
        before=before,financial_rules_changed=False,no_annual=True,prices_loaded_at_prepare=False,
        scope='Only exclude the reviewed ptyxis instance from the exact resource query; all other GPU PID rows stay; no process is stopped')
    write_new(attempt/'adapter_protocol.json',p)
    write_new(attempt/'adapter_protocol_receipt.json',dict(protocol_file='adapter_protocol.json',protocol_sha256=sha(attempt/'adapter_protocol.json'),sealed_at=now()))
    return p


def verify(original,main,reference,dataset,attempt):
    original,attempt=Path(original),Path(attempt)
    p,r=read_json(attempt/'adapter_protocol.json'),read_json(attempt/'adapter_protocol_receipt.json')
    if (r.get('protocol_file')!='adapter_protocol.json' or r.get('protocol_sha256')!=sha(attempt/'adapter_protocol.json')
        or p.get('financial_rules_changed') is not False or p.get('no_annual') is not True
        or p.get('prices_loaded_at_prepare') is not False or p.get('source_sha256')!=source_pins()
        or p.get('original_failed_files')!=FAILURES or p.get('original_protocol_sha256')!=PROTOCOL_SHA
        or p.get('original_protocol_receipt_sha256')!=RECEIPT_SHA):raise ValueError('Transparent resource declaration changed')
    if len(p.get('allowed_gpu_rows',[]))!=1 or p['allowed_gpu_rows'][0].get('pid')!=715236:
        raise ValueError('Only the single explicitly approved PID can be excluded')
    core.verify_pins(attempt/'adapter_sources',p['source_sha256'])
    for name,expected in FAILURES.items():
        if sha(original/name)!=expected or sha(attempt/'original_failure'/name)!=expected:raise ValueError('Retained original failure changed')
    if sha(attempt/'protocol.json')!=PROTOCOL_SHA or sha(attempt/'protocol_receipt.json')!=RECEIPT_SHA:raise ValueError('Copied original financial declaration changed')
    core.verify_protocol(main,reference,dataset,attempt)
    if instance_info()!=p['allowed_instance']:raise ValueError('Approved terminal instance changed')
    return p


def driver(original,main,reference,dataset,attempt):
    attempt=Path(attempt);p=verify(original,main,reference,dataset,attempt);events=[];before=snapshot();ok=False;error=None;after=None
    original_run=subprocess.run
    try:
        verify_resource_snapshot(before,p)
        with resource_adapter(p,events):result=core.run_validation(main,reference,dataset,attempt)
        after=snapshot(original_run);verify_resource_snapshot(after,p)
        ok=True;return result
    except BaseException as exc:
        error=dict(type=type(exc).__name__,message=str(exc));raise
    finally:
        restored=subprocess.run is original_run
        if after is None:after=snapshot(original_run)
        write_new(attempt/'adapter_runtime.json',dict(at=now(),status='completed' if ok else 'failed',
            before=before,after=after,query_events=events,old_subprocess_run_restored=restored,error=error,
            adapter_protocol_sha256=sha(attempt/'adapter_protocol.json'),financial_rules_changed=False,no_annual=True))
        if not restored:raise AssertionError('Original subprocess.run was not restored')


def supervise(original,main,reference,dataset,attempt):
    attempt=Path(attempt);p=verify(original,main,reference,dataset,attempt)
    proof,log=attempt/'runner_process.json',attempt/'validation.log'
    if proof.exists() or log.exists() or (attempt/'validation').exists():raise FileExistsError('Preserve the unique compatibility attempt')
    command=[sys.executable,'-u',str(Path(__file__).resolve()),'--original-runner',str(ROOT/'scripts/btceth_rl_execution_band.py'),
        '--phase','_validate','--original-output',str(original),'--main-output',str(main),'--reference-output',str(reference),'--dataset',str(dataset),'--output-dir',str(attempt)]
    record=dict(phase='validate',started_at=now(),supervisor_pid=os.getpid(),command=command,status='launching',
        protocol_sha256=sha(attempt/'protocol.json'),source_sha256=CORE_SHA,core_source_sha256=CORE_SHA,
        adapter_source_sha256=sha(Path(__file__)),actual_entrypoint=str(Path(__file__).resolve()),
        source_adaptation='Declared instance-only resource adapter calling unchanged frozen core.run_validation',
        adapter_protocol_sha256=sha(attempt/'adapter_protocol.json'),log_relative_path='validation.log')
    def publish():
        temp=proof.with_suffix('.tmp');temp.write_text(json.dumps(record,indent=2)+'\n');temp.replace(proof)
    publish()
    with log.open('x',encoding='utf-8') as stream:
        child=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT)
        record.update(status='running',child_pid=child.pid);publish();code=child.wait()
    record.update(status='completed' if code==0 else 'failed',exit_code=code,completed_at=now(),log_sha256=sha(log));publish()
    if code:raise RuntimeError('Preserve failed compatibility attempt')
    return record


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',required=True,choices=('prepare','validate','_validate'))
    parser.add_argument('--original-runner',type=Path,default=ROOT/'scripts/btceth_rl_execution_band.py')
    for name in ('original-output','main-output','reference-output','dataset','output-dir'):parser.add_argument('--'+name,type=Path,required=True)
    a=parser.parse_args()
    if a.original_runner.resolve()!=(ROOT/'scripts/btceth_rl_execution_band.py').resolve():raise ValueError('Only the declared original runner is callable')
    values=[getattr(a,n).resolve() for n in ('original_output','main_output','reference_output','dataset','output_dir')]
    if a.phase=='prepare':result=prepare(*values)
    elif a.phase=='validate':result=supervise(*values)
    else:
        for _ in range(20):
            r=read_json(values[-1]/'runner_process.json')
            if r.get('child_pid')==os.getpid():break
            time.sleep(.05)
        if r.get('child_pid')!=os.getpid() or r.get('supervisor_pid')!=os.getppid() or r.get('status')!='running':raise ValueError('Private adapter child identity mismatch')
        result=driver(*values)
    print(json.dumps({'phase':a.phase,'result':result}),flush=True)


if __name__=='__main__':main()

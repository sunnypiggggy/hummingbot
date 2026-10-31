"""Versioned transport/launch controls for one explicitly authorized study.

Not part of the financial source contract.  Staging does not prepare, train,
evaluate, restart, trade, or touch SSH configuration.  Every phase launch is a
separate explicit action and captures the actual PID/start ticks and byte pins.
"""
from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
import uuid

ROOT = Path(__file__).resolve().parents[1]
STUDY = "single_pair_rolling_all_algorithms_20261031"
HOST = "v100-192-168-8-120"
REMOTE_ROOT = "/home/sunny/research/" + STUDY
REMOTE_PYTHON = "/home/sunny/research-envs/btceth-rl-20261029/bin/python"
REMOTE_DATASET = "/home/sunny/research/btceth_rl_20261029/data/crypto_rl_btceth_20261029/dataset.pkl.gz"
SOURCE_NAMES = tuple("scripts/" + name + ".py" for name in (
    "backtest_single_pair_rl", "single_pair_rl_env", "single_pair_rl_models", "audit_single_pair_rl",
    "backtest_btceth_rl", "btceth_rl_env", "crypto_rl_portfolio_env", "crypto_rl_reporting", "crypto_rl_data",
    "btceth_rl_td3", "btceth_rl_droq", "rolling_single_pair_models", "backtest_rolling_single_pair", "audit_rolling_single_pair"))
INPUT_NAMES = ("inventory.json", "selection_inputs/native_protocol.json", "selection_inputs/native_validation_summary.csv",
    "selection_inputs/c01_lr1e4_validation_summary.csv", "selection_inputs/c02_gamma099_validation_summary.csv",
    "selection_inputs/c03_lr1e4_gamma099_validation_summary.csv")


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _safe_local_study(path):
    target = Path(path).absolute()
    intended = ROOT / "results/backtests" / STUDY
    if target.resolve() != intended.resolve():
        raise ValueError("Only the named new local study is permitted")
    if any(p.is_symlink() or hasattr(p, "is_junction") and p.is_junction() for p in (target, *target.parents)):
        raise ValueError("Study paths cannot cross links or junctions")
    return target


def stage_plan(local_study, *, sources_only=False):
    study = _safe_local_study(local_study)
    files = {"source/" + name: ROOT / name for name in SOURCE_NAMES}
    if not sources_only:
        files.update({name: study / name for name in INPUT_NAMES})
    files["control_source/rolling_single_pair_remote_ops.py"] = Path(__file__).resolve()
    for name, path in files.items():
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError("Required regular staged input is missing: " + name)
    return dict(study=STUDY, remote_root=REMOTE_ROOT, staged_not_frozen=True, sources_only=bool(sources_only),
        file_sha256={name: sha(path) for name, path in sorted(files.items())},
        file_sizes={name: path.stat().st_size for name, path in sorted(files.items())},
        missing_formal_inputs=[name for name in INPUT_NAMES if name not in files],
        local_files=files)


_REMOTE = r'''
import base64,datetime,gzip,hashlib,io,json,os,pathlib,shutil,subprocess,tarfile,time
root=pathlib.Path(REQUEST['remote_root'])
assert str(root)==EXPECTED_ROOT and root.parent==pathlib.Path('/home/sunny/research')
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def safe(name):
    assert isinstance(name,str) and name and not name.startswith('/') and chr(92) not in name and ':' not in name
    assert all(part not in ('','.','..') for part in name.split('/'))
    p=root.joinpath(*name.split('/'))
    for item in (p,*p.parents):
        assert not item.is_symlink(),'Symlink path is forbidden'
        if item==root.parent:break
    assert p.resolve().is_relative_to(root.resolve())
    return p
def read(p):return json.loads(p.read_text())
def write_new(p,value):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as stream:json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')
def proc(pid):
    p=pathlib.Path('/proc')/str(pid)
    try:
        raw=(p/'stat').read_text();tail=raw[raw.rfind(')')+2:].split()
        args=[x.decode(errors='replace') for x in (p/'cmdline').read_bytes().split(b'\0') if x]
        return {'pid':int(pid),'start_ticks':int(tail[19]),'state':tail[0],'argv':args}
    except (FileNotFoundError,PermissionError,ProcessLookupError):return None
def active():
    found=[]
    folder=safe('control/launches')
    for item in folder.glob('*/launched.json') if folder.exists() else []:
        receipt=read(item);current=proc(receipt['pid'])
        if current and current['start_ticks']==receipt['start_ticks'] and current['argv']==receipt['argv'] and current['state']!='Z':found.append(current)
    return found
def guard_stage(pins):
    assert not active(),'A recorded study phase is still active'
    for entry in pathlib.Path('/proc').iterdir():
        if not entry.name.isdigit():continue
        p=proc(int(entry.name))
        assert not(p and p['state']!='Z' and any(str(root) in arg for arg in p['argv'])),'An unrecorded process references this study'
    for name,digest in pins.items():
        path=safe(name)
        if path.exists() and sha(path)!=digest:
            assert not safe('protocol.json').exists(),'Frozen study files cannot be replaced'
            assert name in REPLACEABLE_SOURCES or name=='control_source/rolling_single_pair_remote_ops.py','Only the exact staged source whitelist may change'
    if safe('protocol.json').exists():
        for name,digest in pins.items():assert safe(name).is_file() and sha(safe(name))==digest,'Frozen input mismatch'
        return 'already_frozen_identical'
    return 'staged_not_frozen'
action=REQUEST['action']
if action=='check_stage':
    print(json.dumps({'at':now(),'state':guard_stage(REQUEST['file_sha256'])}))
elif action=='stage':
    pins=REQUEST['file_sha256'];state=guard_stage(pins)
    raw=base64.b64decode(REQUEST['archive_base64'],validate=True)
    assert hashlib.sha256(raw).hexdigest()==REQUEST['archive_sha256']
    changed={};old_pins={};members={}
    with tarfile.open(fileobj=io.BytesIO(raw),mode='r:gz') as archive:
        items=archive.getmembers()
        assert len(items)==len(pins) and {x.name for x in items}==set(pins)
        for item in items:
            assert item.isfile() and not item.issym() and not item.islnk() and item.size==REQUEST['file_sizes'][item.name]
            safe(item.name);content=archive.extractfile(item).read()
            assert hashlib.sha256(content).hexdigest()==pins[item.name]
            members[item.name]=content
    # Validate every member before creating or replacing even one staged file.
    revision=REQUEST['operation_id'];backups=safe('control/stage_revisions/'+revision)
    for name,content in members.items():
        destination=safe(name)
        if destination.exists():
            old_pins[name]=sha(destination)
            if old_pins[name]==pins[name]:continue
            backup=backups/name;backup.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(destination,backup)
            assert sha(backup)==old_pins[name]
            temporary=destination.with_name(destination.name+'.'+revision+'.pending')
            with temporary.open('xb') as stream:stream.write(content)
            assert sha(destination)==old_pins[name],'Concurrent staged source change'
            os.replace(temporary,destination)
            changed[name]={'before':old_pins[name],'after':pins[name],'backup':backup.relative_to(root).as_posix()}
        else:
            destination.parent.mkdir(parents=True,exist_ok=True)
            with destination.open('xb') as stream:stream.write(content)
    assert all(sha(safe(name))==digest for name,digest in pins.items())
    result={'at':now(),'operation_id':revision,'state':state,'staged_not_frozen':not safe('protocol.json').exists(),
        'sources_only':REQUEST['sources_only'],'file_sha256':pins,'archive_sha256':REQUEST['archive_sha256'],
        'replaced_source_history':changed,'control_source_sha256':REQUEST['control_source_sha256'],
        'no_prepare_preflight_fit_or_production':True}
    write_new(safe('control/stage_receipts/'+revision+'.json'),result)
    print(json.dumps(result))
elif action=='launch':
    phase=REQUEST['phase'];assert phase in ('prepare','preflight','fit','evaluate','pipeline')
    assert not active(),'A recorded study phase is still active'
    pins=REQUEST['file_sha256']
    assert set(REPLACEABLE_SOURCES)<=set(pins) and set(REQUIRED_INPUTS)<=set(pins)
    assert all(safe(name).is_file() and sha(safe(name))==digest for name,digest in pins.items()),'Recheck all exact staged source and validation inputs'
    for entry in pathlib.Path('/proc').iterdir():
        if not entry.name.isdigit():continue
        p=proc(int(entry.name))
        assert not(p and p['state']!='Z' and any(str(root) in arg for arg in p['argv'])),'An unrecorded study process is active'
    if phase=='prepare':assert not safe('protocol.json').exists(),'Existing protocol cannot be replaced'
    else:
        spec=read(safe('protocol.json'));sidecar=read(safe('protocol_receipt.json'))
        assert sidecar['protocol_sha256']==sha(safe('protocol.json'))
        assert all(pins['source/'+name]==digest for name,digest in spec['sources'].items())
        assert sha(safe('inventory.json'))==spec['inventory_sha256'] and sha(safe('selection.json'))==spec['selection_sha256']
    operation=REQUEST['operation_id'];folder=safe('control/launches/'+operation);folder.mkdir(parents=True,exist_ok=False)
    command=[REQUEST['remote_python'],'-B',str(safe('source/scripts/backtest_rolling_single_pair.py')),phase,'--output',str(root),'--dataset',REQUEST['dataset']]
    if REQUEST.get('max_workers') is not None:
        assert type(REQUEST['max_workers']) is int and 1<=REQUEST['max_workers']<=128
        command+=['--max-workers',str(REQUEST['max_workers'])]
    plan={'at':now(),'phase':phase,'argv':command,'source_and_selection_sha256':pins,
        'control_source_sha256':REQUEST['control_source_sha256'],'protocol_sha256':sha(safe('protocol.json')) if safe('protocol.json').exists() else None,
        'no_production_change':True}
    write_new(folder/'launch_plan.json',plan)
    with (folder/'stdout.log').open('xb') as stream:
        child=subprocess.Popen(command,cwd=safe('source'),stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,close_fds=True,start_new_session=True)
    current=proc(child.pid)
    assert current is not None and current['argv']==command,'Actual launched process identity unavailable; do not restart blindly'
    receipt=dict(plan,**current,launch_plan_sha256=sha(folder/'launch_plan.json'),log_path=(folder/'stdout.log').relative_to(root).as_posix())
    write_new(folder/'launched.json',receipt)
    print(json.dumps(receipt))
elif action=='status':
    result={'at':now(),'remote_root':str(root),'exists':root.exists(),'protocol_frozen':safe('protocol.json').is_file(),'active_recorded_phases':active()}
    markers={}
    for name in ('pipeline_status.json','fit_status.json','fit_complete.json','evaluation_status.json','evaluation_complete.json','completion_gate.json','preflight_complete.json'):
        p=safe(name)
        if p.is_file():
            item=read(p);markers[name]={k:v for k,v in item.items() if k in ('at','status','phase','pid','passed','failure_count','completed','total','remaining','models','accounts','actual_gpu_models','type','error')}
    result['markers']=markers
    directories=list(safe('models').glob('*/*/seed*/v*')) if safe('models').exists() else []
    result['models']={'started':len(directories),'completed':sum((p/'record.json').is_file() for p in directories),'failed':sum((p/'failed.json').is_file() for p in directories)}
    result['available_cpus']=len(os.sched_getaffinity(0));result['load_average']=os.getloadavg()
    for key,args in (('gpus',['nvidia-smi','--query-gpu=uuid,memory.total,memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits']),('gpu_processes',['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'])):
        answer=subprocess.run(args,capture_output=True,text=True)
        result[key]=answer.stdout.splitlines() if answer.returncode==0 else {'unavailable':True,'returncode':answer.returncode}
    result['mem_available_kb']=next((int(line.split()[1]) for line in pathlib.Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:')),None)
    print(json.dumps(result))
else:raise ValueError('Unknown control action')
'''


def remote_request(action, payload, *, timeout=60):
    request = dict(payload, action=action, remote_root=REMOTE_ROOT, remote_python=REMOTE_PYTHON,
                   control_source_sha256=sha(__file__))
    code = "REQUEST=" + repr(request) + "\nEXPECTED_ROOT=" + repr(REMOTE_ROOT)
    code += "\nREPLACEABLE_SOURCES=" + repr({"source/" + p for p in SOURCE_NAMES})
    code += "\nREQUIRED_INPUTS=" + repr(set(INPUT_NAMES)) + "\n" + _REMOTE
    result = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", HOST,
        REMOTE_PYTHON + " -"], input=code, capture_output=True, text=True, encoding="utf-8", timeout=timeout)
    if result.returncode:
        raise RuntimeError("Remote control failed without restart: " + result.stderr[-3000:])
    return json.loads(result.stdout)


def _write_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def stage(local_study, *, sources_only=False):
    plan = stage_plan(local_study, sources_only=sources_only)
    remote_request("check_stage", {k: v for k, v in plan.items() if k != "local_files"})
    study = _safe_local_study(local_study)
    operation = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:10]
    files = plan.pop("local_files")
    # Maintain a local mirror with reversible, versioned pre-freeze replacements.
    for name, source in files.items():
        if not name.startswith("source/"):
            continue
        destination = study / name
        if destination.exists():
            if sha(destination) == plan["file_sha256"][name]:
                continue
            if (study / "protocol.json").exists():
                raise ValueError("The local study is frozen")
            backup = study / "control/stage_revisions" / operation / name
            backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(destination, backup)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        if sha(destination) != plan["file_sha256"][name]:
            raise ValueError("Source changed while staging")
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w:gz") as archive:
        for name, path in files.items():
            if sha(path) != plan["file_sha256"][name]:
                raise ValueError("Input changed before transport")
            archive.add(path, arcname=name, recursive=False)
    raw = stream.getvalue()
    receipt = remote_request("stage", dict(plan, operation_id=operation, archive_sha256=hashlib.sha256(raw).hexdigest(),
        archive_base64=base64.b64encode(raw).decode("ascii")), timeout=90)
    if receipt["file_sha256"] != plan["file_sha256"]:
        raise ValueError("Remote staged receipt differs")
    _write_new(study / "control/stage_receipts" / (operation + ".json"), receipt)
    return receipt


def launch(local_study, phase, *, max_workers=None):
    if phase not in ("prepare", "preflight", "fit", "evaluate", "pipeline"):
        raise ValueError("An explicit valid phase is required")
    plan = stage_plan(local_study)
    pins = plan["file_sha256"]
    operation = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:10]
    receipt = remote_request("launch", dict(file_sha256=pins, operation_id=operation, phase=phase,
                                           dataset=REMOTE_DATASET, max_workers=max_workers))
    _write_new(_safe_local_study(local_study) / "control/launches" / operation / "launched.json", receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("dry-run", "stage", "status", "launch"))
    parser.add_argument("--local-study", type=Path, default=ROOT / "results/backtests" / STUDY)
    parser.add_argument("--sources-only", action="store_true")
    parser.add_argument("--phase", choices=("prepare", "preflight", "fit", "evaluate", "pipeline"))
    parser.add_argument("--max-workers", type=int)
    args = parser.parse_args()
    if args.action == "status":
        result = remote_request("status", {})
    elif args.action == "dry-run":
        result = stage_plan(args.local_study, sources_only=args.sources_only)
        result.pop("local_files")
    elif args.action == "stage":
        result = stage(args.local_study, sources_only=args.sources_only)
    else:
        if args.phase is None or args.sources_only:
            parser.error("launch requires explicit --phase and complete inputs")
        result = launch(args.local_study, args.phase, max_workers=args.max_workers)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()

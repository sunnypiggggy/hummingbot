"""Collect only the completed, frozen rolling study without replacing evidence.

This is a transport/control source, separate from the fourteen frozen financial
sources.  It never trains, resumes, evaluates, repairs, or changes a process.
Both the remote snapshot and every local tar member are independently hashed.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import stat
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[1]
STUDY = "single_pair_rolling_all_algorithms_20261031"
REMOTE_ROOT = "/home/sunny/research/" + STUDY
REMOTE_PYTHON = "/home/sunny/research-envs/btceth-rl-20261029/bin/python"
AUDITOR_NAME = "scripts/audit_rolling_single_pair.py"
AUDITOR_SHA256 = "d7cc3c2fc0bfce9473ed4ecb3ed8f004bc7f2a02e7107c7e24628c2bdfcacdc0"
HEX = re.compile(r"^[0-9a-f]{64}$")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative(name):
    if (not isinstance(name, str) or not name or "\\" in name or ":" in name
            or any(part in ("", ".", "..") for part in name.split("/"))
            or any(ord(char) < 32 for char in name)):
        raise ValueError("Only canonical relative evidence names are permitted")
    path = PurePosixPath(name)
    if path.is_absolute() or path.as_posix() != name:
        raise ValueError("Only canonical relative evidence names are permitted")
    return path


def reject_links(path):
    """Also detect Windows junction/reparse metadata on Python before 3.12."""
    path = Path(path).absolute()
    for entry in (path, *path.parents):
        if entry.is_symlink() or hasattr(entry, "is_junction") and entry.is_junction():
            raise ValueError("Evidence cannot cross a link or junction")
        try:
            info = entry.lstat()
        except FileNotFoundError:
            continue
        if getattr(info, "st_file_attributes", 0) & 0x400:
            raise ValueError("Evidence cannot cross a reparse point")
    return path


REMOTE_EXPORT = r'''
from pathlib import Path,PurePosixPath
from datetime import datetime,timezone
import hashlib,importlib.util,json,os,stat,sys,tarfile,uuid
sys.dont_write_bytecode=True
unresolved=Path(REQUEST_ROOT)
assert unresolved.is_absolute() and unresolved.name==EXPECTED_STUDY,'Only the named rolling study may be exported'
def no_links(path):
    for entry in (path,*path.parents):
        assert not entry.is_symlink(),'Linked evidence is forbidden'
        assert not(hasattr(entry,'is_junction') and entry.is_junction()),'Junction evidence is forbidden'
        try:info=entry.lstat()
        except FileNotFoundError:continue
        assert not(getattr(info,'st_file_attributes',0)&0x400),'Reparse-point evidence is forbidden'
no_links(unresolved)
root=unresolved.resolve();assert root.is_dir()
def digest(path):
    value=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):value.update(block)
    return value.hexdigest()
def canonical(name):
    assert isinstance(name,str) and name and chr(92) not in name and ':' not in name
    assert not any(ord(c)<32 for c in name)
    assert all(part not in ('','.','..') for part in name.split('/'))
    part=PurePosixPath(name)
    assert not part.is_absolute() and part.as_posix()==name
    return part
def safe(name):
    part=canonical(name);path=root.joinpath(*part.parts);path.resolve().relative_to(root)
    no_links(path)
    assert path.is_file() and stat.S_ISREG(path.stat().st_mode),'Only regular evidence files are permitted'
    return path
bound={}
def read_bound(name):
    path=safe(name);body=path.read_bytes();value=hashlib.sha256(body).hexdigest()
    assert name not in bound or bound[name]==value,'Mixed evidence generations'
    bound[name]=value
    return json.loads(body)
status=read_bound('pipeline_status.json')
assert status.get('status')=='completed' and status.get('phase')=='complete','Pipeline has not completed its final audit'
assert status.get('no_production_change') is True
assert type(status.get('pid')) is int and status['pid']>1
spec=read_bound('protocol.json')
assert spec.get('test')==['2025-10-23T00:00:00Z','2026-10-23T00:00:00Z']
assert spec.get('seeds')==list(range(5)) and all(type(s) is int for s in spec['seeds'])
assert spec.get('capital_per_account')==10000 and spec.get('capital_shared_between_pairs') is False and spec.get('ensemble') is False
assert spec.get('selection_uses_test') is False and spec.get('fresh_holdout') is False and spec.get('test_year_already_observed') is True
assert spec.get('cadences')=={'fixed':None,'every_4w':28,'every_week':7}
assert spec.get('costs')=={'base':{'fee':.001,'slippage':.0002},'stress':{'fee':.002,'slippage':.0005}}
assert (spec.get('train_days'),spec.get('lag_days'),spec.get('preheat_days'))==(1827,1,200)
assert spec.get('selected')=={'BTCUSDT':['td3_default','sac_default','sac_gamma099'],'ETHUSDT':['sac_default','tqc_default','sac_gamma099']}
assert spec.get('expected_models')==1590 and spec.get('expected_accounts')==180
assert read_bound('protocol_receipt.json').get('protocol_sha256')==bound['protocol.json']
source_name='source/scripts/audit_rolling_single_pair.py'
assert spec['sources'].get('scripts/audit_rolling_single_pair.py')==EXPECTED_AUDITOR
assert digest(safe(source_name))==EXPECTED_AUDITOR,'Frozen verifier source changed'
bound[source_name]=EXPECTED_AUDITOR
complete=read_bound('completion_gate.json')
assert complete.get('passed') is True and complete.get('failure_count')==0 and complete.get('failures')==[] and complete.get('checks_total',0)>0
assert (complete.get('expected_models'),complete.get('models_checked'))==(1590,1590)
assert (complete.get('expected_accounts'),complete.get('accounts_checked'),complete.get('repeats_checked'))==(180,180,180)
assert complete.get('source_sha256')==EXPECTED_AUDITOR
assert complete.get('protocol_sha256')==bound['protocol.json'] and complete.get('summary_sha256')==digest(safe('summary.csv'))
assert complete.get('inputs_sha256') and complete.get('external_inputs_sha256')
for name,expected in complete['inputs_sha256'].items():
    assert digest(safe(name))==expected and (name not in bound or bound[name]==expected),'Passed input evidence changed'
    bound[name]=expected
external_bound=dict(complete['external_inputs_sha256'])
assert len(external_bound)==2,'Precisely one frozen market archive and manifest are required'
dataset_paths=[Path(name) for name,value in external_bound.items() if value==spec['dataset_sha256']]
manifest_paths=[Path(name) for name,value in external_bound.items() if value==spec['dataset_manifest_sha256']]
assert len(dataset_paths)==len(manifest_paths)==1
dataset,market_manifest=dataset_paths[0],manifest_paths[0]
assert market_manifest.name=='manifest.json' and dataset.parent==market_manifest.parent
def verify_external():
    for name,expected in external_bound.items():
        path=Path(name)
        assert path.is_absolute() and str(path)==name and '..' not in path.parts and path.is_relative_to(root.parent)
        no_links(path)
        assert path.is_file() and stat.S_ISREG(path.stat().st_mode)
        assert digest(path)==expected,'External frozen market input changed'
verify_external()
# Match the exact recorded pipeline launch, never a prepare/preflight PID.
launches=[]
folder=root/'control/launches'
assert folder.is_dir()
for path in sorted(folder.glob('*/launched.json')):
    relative=path.relative_to(root).as_posix();candidate=read_bound(relative)
    if candidate.get('phase')=='pipeline' and candidate.get('pid')==status['pid'] and candidate.get('protocol_sha256')==bound['protocol.json']:
        launches.append((relative,candidate))
assert len(launches)==1,'Exactly one matching pipeline launch receipt is required'
launch_name,launch=launches[0]
assert type(launch.get('pid')) is int and launch['pid']>1 and type(launch.get('start_ticks')) is int and launch['start_ticks']>0
assert launch.get('state') not in (None,'Z')
plan_name=(PurePosixPath(launch_name).parent/'launch_plan.json').as_posix()
plan=read_bound(plan_name)
assert launch.get('launch_plan_sha256')==bound[plan_name]
assert all(launch.get(name)==value for name,value in plan.items()),'Launch receipt differs from its exact command plan'
assert plan.get('no_production_change') is True
log_name=(PurePosixPath(launch_name).parent/'stdout.log').as_posix()
assert launch.get('log_path')==log_name,'Exact launched process stdout path required'
bound[log_name]=digest(safe(log_name))
command=launch.get('argv')
expected_command=[REQUEST_PYTHON,'-B',str(root/'source/scripts/backtest_rolling_single_pair.py'),'pipeline','--output',str(root),'--dataset',str(dataset)]
assert isinstance(command,list) and command[:len(expected_command)]==expected_command
assert len(command)==len(expected_command) or (len(command)==len(expected_command)+2 and command[-2]=='--max-workers' and command[-1].isdigit() and 1<=int(command[-1])<=128)
pins=launch.get('source_and_selection_sha256')
assert isinstance(pins,dict) and pins
for name,expected in pins.items():
    assert digest(safe(name))==expected and (name not in bound or bound[name]==expected),'Actually launched source or selection input changed'
    bound[name]=expected
assert all(pins.get('source/'+name)==value for name,value in spec['sources'].items())
assert pins.get('control_source/rolling_single_pair_remote_ops.py')==launch.get('control_source_sha256')
assert not(Path('/proc')/str(launch['pid'])).exists(),'Wait for the exact pipeline PID to exit; no process will be stopped'
loader=importlib.util.spec_from_file_location('frozen_rolling_export_verifier',safe(source_name))
auditor=importlib.util.module_from_spec(loader);sys.modules[loader.name]=auditor;loader.loader.exec_module(auditor)
verified=auditor.verify_completed_study(root)
assert verified.get('passed') is True and verified.get('no_training_inference_or_repair') is True
assert verified.get('completion_gate_sha256')==bound['completion_gate.json']
assert verified.get('inputs_verified')==len(complete['inputs_sha256'])+len(external_bound)
verify_external()
stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'_'+uuid.uuid4().hex
archive=root.parent/(root.name+'_audited_'+stamp+'.tar.gz')
receipt_path=archive.with_suffix(archive.suffix+'.json')
assert not archive.exists() and not archive.is_symlink() and not receipt_path.exists() and not receipt_path.is_symlink(),'Previous export evidence cannot be replaced'
files={}
def snapshot_files():
    result={}
    for path in sorted(root.rglob('*')):
        no_links(path)
        info=path.stat()
        if stat.S_ISDIR(info.st_mode):continue
        assert stat.S_ISREG(info.st_mode),'Do not omit nonregular or linked evidence'
        if '__pycache__' in path.parts or path.suffix=='.pyc' or path.name=='latest_progress_snapshot.json':continue
        name=path.relative_to(root).as_posix();canonical(name);result[name]=digest(path)
    return result
files=snapshot_files()
assert all(files.get(name)==expected for name,expected in bound.items()),'Audit bindings changed before archive snapshot'
with tarfile.open(archive,'x:gz',compresslevel=1) as stream:
    for name in files:stream.add(root/name,arcname=name,recursive=False)
assert snapshot_files()==files,'Evidence or exact archive membership changed during export'
assert all(digest(safe(name))==expected for name,expected in bound.items()),'Passed evidence changed during export'
verify_external()
assert not(Path('/proc')/str(launch['pid'])).exists(),'Pipeline PID was reused during export; refuse ambiguous collection'
receipt=dict(schema_version=1,study=EXPECTED_STUDY,exported_at=datetime.now(timezone.utc).isoformat(),
    remote_root=str(root),archive=str(archive),archive_sha256=digest(archive),files=files,
    protocol_sha256=bound['protocol.json'],completion_gate_sha256=bound['completion_gate.json'],
    verifier_source_sha256=EXPECTED_AUDITOR,pipeline_pid=launch['pid'],pipeline_start_ticks=launch['start_ticks'],
    pipeline_launch_receipt=launch_name,pipeline_launch_receipt_sha256=bound[launch_name],launch_plan_sha256=bound[plan_name],
    pipeline_log_path=log_name,pipeline_log_sha256=bound[log_name],
    inputs_sha256=dict(complete['inputs_sha256']),external_inputs_sha256=external_bound,
    external_verified_sha256=external_bound,external_rechecked_after_archive=True,
    no_training_inference_or_repair=True,no_production_change=True)
with receipt_path.open('x',encoding='utf-8') as stream:stream.write(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(receipt,ensure_ascii=False))
'''


def export_program(remote_root, remote_python, auditor_sha256=AUDITOR_SHA256):
    if not HEX.fullmatch(auditor_sha256):
        raise ValueError("An exact frozen verifier SHA256 is required")
    replacements = {"REQUEST_ROOT": repr(str(remote_root)), "REQUEST_PYTHON": repr(str(remote_python)),
                    "EXPECTED_STUDY": repr(STUDY), "EXPECTED_AUDITOR": repr(auditor_sha256)}
    source = REMOTE_EXPORT
    for key, value in replacements.items():
        source = source.replace(key, value)
    compile(source, "<frozen-rolling-export>", "exec")
    return source


def _validate_receipt(receipt, remote_root):
    if not isinstance(receipt, dict) or receipt.get("schema_version") != 1 or receipt.get("study") != STUDY or receipt.get("remote_root") != remote_root:
        raise ValueError("Remote export receipt belongs to another study")
    for key in ("archive_sha256", "protocol_sha256", "completion_gate_sha256", "verifier_source_sha256",
                "pipeline_launch_receipt_sha256", "launch_plan_sha256"):
        if not isinstance(receipt.get(key), str) or not HEX.fullmatch(receipt[key]):
            raise ValueError("Missing exact remote receipt SHA256: " + key)
    if (receipt["verifier_source_sha256"] != AUDITOR_SHA256 or receipt.get("external_rechecked_after_archive") is not True
            or receipt.get("no_training_inference_or_repair") is not True or receipt.get("no_production_change") is not True):
        raise ValueError("Remote verification/preservation contract is incomplete")
    files = receipt.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("Exact remote snapshot member map is required")
    for name, digest in files.items():
        _relative(name)
        if not isinstance(digest, str) or not HEX.fullmatch(digest):
            raise ValueError("Invalid snapshot file SHA256")
    for name, expected in {"protocol.json": receipt["protocol_sha256"], "completion_gate.json": receipt["completion_gate_sha256"],
                           "source/" + AUDITOR_NAME: AUDITOR_SHA256,
                           receipt.get("pipeline_launch_receipt", ""): receipt["pipeline_launch_receipt_sha256"]}.items():
        _relative(name)
        if files.get(name) != expected:
            raise ValueError("Remote receipt does not bind required current evidence:" + name)
    launch_name = _relative(receipt["pipeline_launch_receipt"])
    if launch_name.parts[:2] != ("control", "launches") or launch_name.name != "launched.json" or len(launch_name.parts) != 4:
        raise ValueError("Wrong exact pipeline launch receipt path")
    if files.get((launch_name.parent / "launch_plan.json").as_posix()) != receipt["launch_plan_sha256"]:
        raise ValueError("Exact pipeline launch plan is not bound")
    for key in ("pipeline_pid", "pipeline_start_ticks"):
        if type(receipt.get(key)) is not int or receipt[key] <= 1:
            raise ValueError("Exact launched process identity required")
    if (not isinstance(receipt.get("inputs_sha256"), dict) or not receipt["inputs_sha256"]
            or not all(files.get(name) == digest for name, digest in receipt["inputs_sha256"].items())):
        raise ValueError("Current passed inputs do not match snapshot membership")
    if (not isinstance(receipt.get("external_inputs_sha256"), dict) or len(receipt["external_inputs_sha256"]) != 2
            or receipt.get("external_verified_sha256") != receipt["external_inputs_sha256"]):
        raise ValueError("Current external market verification is incomplete")
    for name, digest in receipt["external_inputs_sha256"].items():
        external = PurePosixPath(name)
        if (not external.is_absolute() or external.as_posix() != name or ".." in external.parts
                or "\\" in name or ":" in name or not isinstance(digest, str) or not HEX.fullmatch(digest)):
            raise ValueError("Canonical absolute public-market input and SHA256 required")
    archive = PurePosixPath(receipt.get("archive", ""))
    expected_parent = PurePosixPath(remote_root).parent
    if (not archive.is_absolute() or archive.parent != expected_parent or str(archive) != receipt["archive"]
            or not re.fullmatch(re.escape(STUDY) + r"_audited_[A-Za-z0-9_]+\.tar\.gz", archive.name)):
        raise ValueError("Remote archive is outside the named research export directory")
    return archive.name


def _target(output, name):
    relative = _relative(name)
    target = reject_links(output.joinpath(*relative.parts))
    target.resolve().relative_to(output)
    return target


def _archive_pass(archive, output, receipt, *, materialize=False):
    seen = set()
    with tarfile.open(archive, "r|gz") as stream:
        for member in stream:
            if member.type not in (tarfile.REGTYPE, tarfile.AREGTYPE) or not member.isfile():
                raise ValueError("Only regular archive members are permitted")
            if member.name in seen or member.name not in receipt["files"]:
                raise ValueError("Exact unique archive member count and names required")
            seen.add(member.name)
            target = _target(output, member.name)
            if target.exists() and (not target.is_file() or sha(target) != receipt["files"][member.name]):
                raise FileExistsError("Preserve changed local evidence:" + member.name)
            body = stream.extractfile(member)
            if body is None:
                raise ValueError("Missing regular archive member body")
            new_file = materialize and not target.exists()
            if new_file:
                target.parent.mkdir(parents=True, exist_ok=True)
                reject_links(target.parent)
            digest = hashlib.sha256()
            if new_file:
                with target.open("xb") as saved:
                    for block in iter(lambda: body.read(1024 * 1024), b""):
                        digest.update(block)
                        saved.write(block)
            else:
                for block in iter(lambda: body.read(1024 * 1024), b""):
                    digest.update(block)
            if digest.hexdigest() != receipt["files"][member.name]:
                raise ValueError("Archive member SHA256 mismatch:" + member.name)
    if len(seen) != len(receipt["files"]) or seen != set(receipt["files"]):
        raise ValueError("Exact complete archive membership required")


def collect(host, remote_root, output, remote_python):
    if remote_root != REMOTE_ROOT:
        raise ValueError("Only the explicitly named remote rolling study can be collected")
    output = reject_links(output).resolve()
    if output.name != STUDY:
        raise ValueError("Only the named local rolling study can receive this archive")
    local_verifier = ROOT / AUDITOR_NAME
    if sha(local_verifier) != AUDITOR_SHA256:
        raise ValueError("Frozen independent verifier changed; use a recorded versioned revision")
    source = export_program(remote_root, remote_python)
    answer = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", host,
                             shlex.quote(remote_python) + " -B -"], input=source, capture_output=True,
                            text=True, encoding="utf-8", check=True)
    receipt = json.loads(answer.stdout)
    basename = _validate_receipt(receipt, remote_root)
    output.mkdir(parents=True, exist_ok=True)
    transport = reject_links(output / "transport")
    transport.mkdir(exist_ok=True)
    archive = reject_links(transport / basename)
    receipt_path = reject_links(transport / (basename + ".receipt.json"))
    remote_receipt_path = reject_links(transport / (basename + ".remote-receipt.json"))
    for path in (archive, receipt_path, remote_receipt_path):
        if path.exists():
            raise FileExistsError("Preserve previous archive or transfer receipt:" + path.name)
    # Keep the exact received remote attestation even if download or extraction
    # later fails. A failed attempt is never relabelled a successful receipt.
    with remote_receipt_path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(answer.stdout)
    subprocess.run(["scp", "-q", host + ":" + receipt["archive"], str(archive)], check=True)
    if sha(archive) != receipt["archive_sha256"]:
        raise ValueError("Transferred archive SHA256 mismatch")
    _archive_pass(archive, output, receipt, materialize=False)
    _archive_pass(archive, output, receipt, materialize=True)
    if sha(archive) != receipt["archive_sha256"]:
        raise ValueError("Transferred archive changed during extraction")
    for name, expected in receipt["files"].items():
        if sha(_target(output, name)) != expected:
            raise ValueError("Collected file changed after extraction:" + name)
    local_receipt = dict(receipt, archive=basename, remote_archive=receipt["archive"], archive_file="transport/" + basename,
                         collected_at=datetime.now(timezone.utc).isoformat(), local_output=str(output),
                         remote_export_receipt_sha256=sha(remote_receipt_path),
                         remote_export_receipt_file="transport/" + remote_receipt_path.name,
                         all_archive_members_hashed_before_materialization=True,
                         existing_identical_files_preserved=True)
    with receipt_path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(local_receipt, ensure_ascii=False, indent=2) + "\n")
    return {"files_verified": len(receipt["files"]), "archive_sha256": receipt["archive_sha256"],
            "output": str(output), "receipt": str(receipt_path), "completion_gate_sha256": receipt["completion_gate_sha256"],
            "no_training_inference_or_repair": True, "no_production_change": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="v100-192-168-8-120")
    parser.add_argument("--remote-root", default=REMOTE_ROOT)
    parser.add_argument("--remote-python", default=REMOTE_PYTHON)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(collect(args.host, args.remote_root, args.output, args.remote_python)))


if __name__ == "__main__":
    main()

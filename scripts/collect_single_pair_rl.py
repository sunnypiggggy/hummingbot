"""Collect a completed, audited independent-pair study without replacing evidence."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import shlex
import subprocess
import tarfile


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


REMOTE_EXPORT = '''
from pathlib import Path,PurePosixPath
from datetime import datetime,timezone
import hashlib,json,os,tarfile
unresolved=Path(REMOTE_ROOT)
assert not any(p.is_symlink() for p in (unresolved,*unresolved.parents)),'Do not export a linked root'
root=unresolved.resolve()
assert root.is_dir()
read=lambda p:json.loads(p.read_text())
digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
bound={}
def safe(name):
    relative=PurePosixPath(name)
    assert name and relative.as_posix()==name and not relative.is_absolute() and '..' not in relative.parts and ':' not in name and chr(92) not in name
    path=root.joinpath(*relative.parts);path.resolve().relative_to(root)
    assert path.is_file() and not any(p.is_symlink() for p in (path,*path.parents))
    return path
def read_bound(name):
    body=safe(name).read_bytes();bound[name]=hashlib.sha256(body).hexdigest()
    return json.loads(body)
status=read_bound('pipeline_status.json')
assert status['status']=='completed','Study has not completed its phase pipeline'
launch=read_bound('pipeline_launch.json')
assert not Path('/proc'/Path(str(launch['pid']))).exists(),'Wait for the exact supervisor to finish writing'
spec=read_bound('protocol.json')
assert spec['ensemble'] is False and spec['capital_per_account']==10000 and spec['capital_shared_between_pairs'] is False
assert spec['symbols']==['BTCUSDT','ETHUSDT'] and spec['seeds']==list(range(5)) and set(spec['costs'])=={'base','stress'}
assert spec['algorithms'] and len(spec['algorithms'])==len(set(spec['algorithms'])) and set(spec['algorithms'])<=set(['sac','tqc','crossq','recurrent_ppo','td3','droq'])
assert spec['baselines']==['cash','buy_hold','sma200_trend','momentum180']
result_files=['equity_1h.csv.gz','fills.csv','weights.csv','boundary_equity.csv','metadata.json','fingerprints.json','actions.json','summary.json','daily_equity.csv','monthly_returns.csv']
for split in ['validation','test']:
    audit=read_bound(split+'_audit.json')
    expected={f'evaluation/{split}/{symbol}/{algorithm}/seed{seed}/{scenario}' for symbol in spec['symbols'] for algorithm in spec['algorithms'] for seed in spec['seeds'] for scenario in spec['costs']}
    expected.update(f'evaluation/{split}/{symbol}/{algorithm}/baseline/{scenario}' for symbol in spec['symbols'] for algorithm in spec['baselines'] for scenario in spec['costs'])
    count=len(expected)
    assert audit['passed'] is True and audit['failure_count']==0 and not audit['failures'] and audit['phase']==split and audit['checks_total']>0
    assert all(audit[k]==count for k in ['expected_accounts','accounts_checked','repeats_checked','accounts_count'])
    assert len(audit['cases'])==count and {case['relative_path'] for case in audit['cases']}==expected
    assert audit['protocol_sha256']==bound['protocol.json'] and audit['summary_sha256']==digest(safe(split+'_summary.csv'))
    inputs=dict(audit['inputs_sha256'])
    assert set(inputs)=={'protocol.json',split+'_summary.csv',split+'_complete.json'}
    for case in audit['cases']:
        assert case['passed'] is True and case['failure_count']==0 and not case['failures'] and case['checks_total']>0
        directory=case['relative_path']
        expected_inputs={directory+'/'+name for name in result_files}|{directory.replace('evaluation/','repeat/',1)+'/'+name for name in result_files}
        assert set(case['inputs_sha256'])==expected_inputs
        for name,expected in case['inputs_sha256'].items():
            assert name not in inputs or inputs[name]==expected
            inputs[name]=expected
    assert all(digest(safe(name))==expected for name,expected in inputs.items()),'Current evidence differs from passed audit'
    for name,expected in inputs.items():
        assert name not in bound or bound[name]==expected
        bound[name]=expected
stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
archive=root.parent/(root.name+'_audited_'+stamp+'.tar.gz')
receipt_path=archive.with_suffix(archive.suffix+'.json')
assert not receipt_path.exists() and not receipt_path.is_symlink(),'Preserve previous remote export receipt'
files={}
for path in sorted(root.rglob('*')):
    assert not path.is_symlink(),'Do not export linked paths'
    if not path.is_file() or '__pycache__' in path.parts or path.suffix=='.pyc' or path.name=='latest_progress_snapshot.json': continue
    files[path.relative_to(root).as_posix()]=digest(path)
assert all(files.get(name)==expected for name,expected in bound.items()),'Audit bindings changed before export snapshot'
with tarfile.open(archive,'x:gz') as stream:
    for name in files: stream.add(root/name,arcname=name,recursive=False)
assert all(digest(root/name)==expected for name,expected in files.items()),'Evidence changed during export'
assert all(digest(safe(name))==expected for name,expected in bound.items()),'Audited evidence changed during export'
receipt=dict(exported_at=datetime.now(timezone.utc).isoformat(),remote_root=str(root),archive=str(archive),archive_sha256=digest(archive),files=files,protocol_sha256=digest(root/'protocol.json'),no_production_change=True)
with receipt_path.open('x') as stream:stream.write(json.dumps(receipt,indent=2)+'\\n')
print(json.dumps(receipt))
'''


def reject_links(path):
    path = Path(path).absolute()
    if any(p.is_symlink() or hasattr(p, "is_junction") and p.is_junction() for p in (path, *path.parents)):
        raise ValueError("Evidence cannot cross a local link")
    return path


def collect(host, remote_root, output, remote_python):
    output = reject_links(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    source = REMOTE_EXPORT.replace("REMOTE_ROOT", repr(str(remote_root)))
    exported = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", host, shlex.quote(remote_python) + " -"],
                              input=source, capture_output=True, text=True, encoding="utf-8", check=True)
    receipt = json.loads(exported.stdout)
    downloads = reject_links(output / "transport")
    downloads.mkdir(exist_ok=True)
    archive = reject_links(downloads / PurePosixPath(receipt["archive"]).name)
    receipt_path = reject_links(downloads / (archive.name + ".receipt.json"))
    if archive.exists():
        raise FileExistsError("Preserve previous transferred archive")
    if receipt_path.exists():
        raise FileExistsError("Preserve previous transfer receipt")
    subprocess.run(["scp", "-q", host + ":" + receipt["archive"], str(archive)], check=True)
    if sha(archive) != receipt["archive_sha256"]:
        raise ValueError("Transferred archive hash mismatch")
    with tarfile.open(archive, "r:gz") as stream:
        members = stream.getmembers()
        if {member.name for member in members} != set(receipt["files"]) or len(members) != len(receipt["files"]):
            raise ValueError("Exact export membership required")
        for member in members:
            relative = PurePosixPath(member.name)
            if (not member.isfile() or not member.name or relative.as_posix() != member.name or relative.is_absolute() or ".." in relative.parts or ":" in member.name or "\\" in member.name):
                raise ValueError("Only relative regular evidence files are allowed")
            target = output.joinpath(*relative.parts)
            target.resolve().relative_to(output)
            if any(parent.is_symlink() or hasattr(parent, "is_junction") and parent.is_junction()
                   for parent in (target, *target.parents) if parent != output.parent):
                raise ValueError("Evidence cannot cross a local link")
            if target.exists():
                if sha(target) != receipt["files"][member.name]:
                    raise FileExistsError("Preserve changed local evidence: " + member.name)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            data = stream.extractfile(member)
            if data is None:
                raise ValueError("Missing regular-file archive body")
            with target.open("xb") as saved:
                for block in iter(lambda: data.read(1024 * 1024), b""):
                    saved.write(block)
            if sha(target) != receipt["files"][member.name]:
                raise ValueError("Extracted file hash mismatch: " + member.name)
    receipt["collected_at"] = datetime.now(timezone.utc).isoformat()
    receipt["local_output"] = str(output)
    with receipt_path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
    return {"files_verified": len(receipt["files"]), "archive_sha256": receipt["archive_sha256"], "output": str(output)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="v100-192-168-8-120")
    parser.add_argument("--remote-root", default="/home/sunny/research/single_pair_rl_20261030")
    parser.add_argument("--output", required=True)
    parser.add_argument("--remote-python", default="/home/sunny/research-envs/btceth-rl-20261029/bin/python")
    args = parser.parse_args()
    print(json.dumps(collect(args.host, args.remote_root, args.output, args.remote_python)))


if __name__ == "__main__":
    main()

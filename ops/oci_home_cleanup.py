"""Reversible, narrowly allowlisted OCI home cleanup. Default is preview only."""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

HOME_ROOT=Path('/home/ubuntu')
CACHE=HOME_ROOT/'anaconda3/pkgs'
DEST=HOME_ROOT/'extra_drive/maintenance-quarantine/20261010-home-cleanup'
STAGING_DIRS=('tg-price-curve-staging','account-auditor-staging','serverdocker_old',' \\',';','cp',' ','\\')
STAGING_FILES=(
 'ACCOUNT_AUDITOR-eight-hour.md','ACCOUNT_AUDITOR.md','TRADING_MANAGEMENT_BOT_V3.md',
 'account-audit-eight-hour.tar.gz','account-audit-four-build.log','account-audit-four.tar.gz',
 'account-auditor-candidate.tar.gz','account-auditor-final.tar.gz','account-auditor-integration.patch',
 'audit-schedule.py','bot-display.tar.gz','dca-fee-fix-20260920.tar',
 'deploy_audit_comparison.py','deploy_audit_eight_hour.py','deploy_audit_quantity_check.py',
 'deploy_dual_audit.py','deploy_host_overview.py','deploy_simple_audit.py','dual-audit.tar.gz',
 'finalize_simple_audit.py','fix_overview.py','overview-fix.tar','readonly-menu.tar.gz',
 'rollback-audit-docs.tar','rollback-before-audit-source.tar','rollback_audit.py',
 'simple-audit-source.tar','simple_audit.py','simple_audit_view.py','soak.pid',
 'telegram-notify-deploy.tgz','telegram-png-only-4852c2fd3.tar','telegram-png-only-674e7322a.tar',
 'test_account_auditor_four_assets.py','tg-model-probability-history-deploy.py')


def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as stream:
  for block in iter(lambda:stream.read(1048576),b''):h.update(block)
 return h.hexdigest()


def referenced(path,processes,mounts,services):
 name=str(path)
 return any(name==ref or name.startswith(ref.rstrip('/')+'/') or ref.startswith(name.rstrip('/')+'/')
            for ref in mounts) or any(name in text for text in processes+services)


def main():
 parser=argparse.ArgumentParser()
 parser.add_argument('--apply',action='store_true')
 args=parser.parse_args()
 if not HOME_ROOT.is_dir() or not CACHE.is_dir():raise RuntimeError('not expected OCI host layout')
 ids=subprocess.check_output(['docker','ps','-aq'],text=True).split()
 containers=json.loads(subprocess.check_output(['docker','inspect',*ids],text=True))
 mounts=[m['Source'] for c in containers for m in c['Mounts']]
 processes=[]
 for proc in Path('/proc').glob('[0-9]*'):
  try:
   processes.append((proc/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace'))
   processes.append(os.readlink(proc/'cwd'))
  except (OSError,PermissionError):pass
 services=[]
 for root in (Path('/etc/systemd/system'),Path('/etc/cron.d'),Path('/var/spool/cron/crontabs')):
  for p in root.rglob('*'):
   if not p.is_file() or p.is_symlink():continue
   try:services.append(p.read_text())
   except (OSError,UnicodeError):pass
 # Conda package archives are never imported by Python. Extracted package dirs
 # and environments are deliberately excluded (they may have links/references).
 if any('/anaconda3/bin/conda' in p and 'clean' not in p for p in processes):
  raise RuntimeError('Conda operation active; skip package archive cleanup')
 paths=sorted([p for p in CACHE.iterdir() if p.is_file() and not p.is_symlink()
               and (p.name.endswith('.conda') or p.name.endswith('.tar.bz2'))])
 paths += [HOME_ROOT/n for n in (*STAGING_FILES,*STAGING_DIRS) if (HOME_ROOT/n).exists()]
 selected=[]
 for p in paths:
  if p.is_symlink() or not p.resolve().is_relative_to(HOME_ROOT) or p.resolve()==HOME_ROOT:
   raise RuntimeError('unsafe cleanup target')
  if referenced(p,processes,mounts,services):continue
  if p.is_dir() and any(x.is_symlink() for x in p.rglob('*')):continue
  selected.append(p)
 def size(p):return p.stat().st_size if p.is_file() else sum(x.stat().st_size for x in p.rglob('*') if x.is_file())
 preview={'apply':args.apply,'selected_count':len(selected),'bytes':sum(size(p) for p in selected),
          'destination':str(DEST),'paths':[str(p) for p in selected if p.parent!=CACHE],
          'conda_archives':sum(p.parent==CACHE for p in selected)}
 print(json.dumps(preview),flush=True)
 if not args.apply:return
 if DEST.resolve()!=DEST or not DEST.resolve().is_relative_to(HOME_ROOT/'extra_drive/maintenance-quarantine'):
  raise RuntimeError('unsafe quarantine path')
 if shutil.disk_usage(HOME_ROOT/'extra_drive').free<preview['bytes']+1024**3:
  raise RuntimeError('insufficient quarantine headroom')
 os.umask(0o077)
 DEST.mkdir(parents=True,exist_ok=True)
 receipt=DEST/'manifest.jsonl'
 for index,p in enumerate(selected):
  target=DEST/p.relative_to(HOME_ROOT)
  if target.exists():raise RuntimeError('quarantine destination already exists; manual reconciliation required')
  target.parent.mkdir(parents=True,exist_ok=True)
  sources=[p] if p.is_file() else [x for x in p.rglob('*') if x.is_file()]
  hashes={str(x.relative_to(p) if p.is_dir() else Path(p.name)):digest(x) for x in sources}
  with receipt.open('a') as log:
   log.write(json.dumps({'source':str(p),'destination':str(target),'phase':'PREPARED','hashes':hashes,
                        'uid':p.stat().st_uid,'gid':p.stat().st_gid,'mode':p.stat().st_mode})+'\n')
  if p.is_file():
   shutil.copy2(p,target)
   if digest(target)!=next(iter(hashes.values())):raise RuntimeError('copy checksum mismatch')
   p.unlink()
  else:
   shutil.copytree(p,target)
   if any(digest(target/k)!=h for k,h in hashes.items()):raise RuntimeError('directory checksum mismatch')
   # Exact non-symlink staging target; no home/workspace-root deletion.
   shutil.rmtree(p)
  with receipt.open('a') as log:log.write(json.dumps({'source':str(p),'destination':str(target),'phase':'MOVED'})+'\n')
  if (index+1)%100==0:print(json.dumps({'moved':index+1}),flush=True)
 print(json.dumps({'completed':True,'moved':len(selected),'manifest':str(receipt)}),flush=True)


if __name__=='__main__':main()

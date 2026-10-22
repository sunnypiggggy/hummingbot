"""Prepare private backups and a minimal Guard image; never place orders."""
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
import subprocess

ROOT = Path('/home/ubuntu/extra_drive/hummingbot')
INCOMING = Path('/home/ubuntu/.cache/hb-reentry-20261021')
ROLLOUT = ROOT / '.reentry-rollouts/20261021-recovery-v1'


def inspect(name):
    return json.loads(subprocess.check_output(['docker', 'inspect', name]))[0]


def main():
    backup = ROLLOUT / 'backup'
    backup.mkdir(parents=True, mode=0o700, exist_ok=False)
    names = ['dca-live-guard', 'grid-live-guard', 'dca-live-report',
             'grid-live-fdusd-400', 'dca-live-btcusdt-200', 'dca-live-ethusdt-200',
             'grid-live-fdusd-scheduler', 'sunnypiggy-trade-bot']
    baseline = {}
    for name in names:
        value = inspect(name)
        baseline[name] = {'id': value['Id'], 'image': value['Image'],
                          'started_at': value['State']['StartedAt']}
    (backup / 'containers.json').write_text(json.dumps(baseline, indent=2))
    for rel in ['dca-live-data/guard_state.json', 'grid-live-fdusd-data/guard_state.json',
                'api-files/bots/instances/grid-live-fdusd-400/data/live_grid_runtime_state.json',
                'live_guard/dca_reentry.py', 'scripts/walk_forward_portfolio_grid_live.py',
                'api-files/bots/scripts/walk_forward_portfolio_grid_live.py']:
        target = backup / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / rel, target)
    for rel in ['account-inventory-data/account_inventory.sqlite',
                'api-files/bots/instances/grid-live-fdusd-400/data',
                'api-files/bots/instances/dca-live-btcusdt-200/data',
                'api-files/bots/instances/dca-live-ethusdt-200/data']:
        path = ROOT / rel
        databases = [path] if path.is_file() else list(path.glob('*.sqlite'))
        for source in databases:
            target = backup / source.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            with sqlite3.connect(f'file:{source}?mode=ro', uri=True) as original:
                with sqlite3.connect(target) as saved:
                    original.backup(saved)
                    assert saved.execute('PRAGMA quick_check').fetchone()[0] == 'ok'
    context = ROLLOUT / 'context'
    (context / 'live_guard').mkdir(parents=True)
    for src, dst in [('live_guard/dca_reentry.py', 'live_guard/dca_reentry.py'),
                     ('Dockerfile.dca-guard', 'Dockerfile'),
                     ('compose.reentry.yml', 'compose.reentry.yml')]:
        shutil.copy2(INCOMING / src, context / dst)
    old = baseline['dca-live-guard']['image']
    base = 'hummingbot/dca-live-guard:reentry-base-20261021-v1'
    subprocess.run(['docker', 'tag', old, base], check=True)
    subprocess.run(['docker', 'build', '--build-arg', f'BASE_IMAGE={base}', '-t',
                    'hummingbot/dca-live-guard:reentry-cohort-20261021-v1', str(context)], check=True)
    print('Prepared private backups and minimal image:', ROLLOUT)
    print('Candidate source SHA256:', hashlib.sha256((context / 'live_guard/dca_reentry.py').read_bytes()).hexdigest())


if __name__ == '__main__':
    main()

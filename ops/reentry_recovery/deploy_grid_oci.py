"""Reload the Grid after proving no active Grid orders or pending exits."""
import hashlib
import json
from pathlib import Path
import os
import shutil
import subprocess

ROOT = Path('/home/ubuntu/extra_drive/hummingbot')
ROLLOUT = ROOT / '.reentry-rollouts/20261021-recovery-v1'
SOURCE = Path('/home/ubuntu/.cache/hb-reentry-20261021/scripts/walk_forward_portfolio_grid_live.py')


def exchange_orders(*, known_ids=None):
    code = """import sys,json
from pathlib import Path
sys.path[:0]=['/app','/app/scripts']
from dca_live_guard import BinanceEmergencyClient
c=BinanceEmergencyClient.from_secret_file(Path('/run/secrets/dca_binance_emergency_credentials'))
print(json.dumps({p:[o['clientOrderId'] for o in c.open_orders(p)] for p in ['BTC-FDUSD','ETH-FDUSD']}))
"""
    output = subprocess.check_output(['docker', 'exec', 'dca-live-guard', 'python', '-c', code], text=True)
    result = json.loads(output)
    if known_ids is None:
        assert not any(result.values()), 'Grid orders remain; pause maintenance'
    else:
        assert all(set(ids) <= known_ids[pair] for pair, ids in result.items()), 'Unknown Grid orders; pause maintenance'
    print('Exchange Grid order check:', {p: len(ids) for p, ids in result.items()})


def main():
    path = ROOT / 'api-files/bots/instances/grid-live-fdusd-400/data/live_grid_runtime_state.json'
    state = json.loads(path.read_text())
    assert not state['pending_flatten'] and not state['pending_inventory_exit']
    assert all(s['phase'] == 'ACTIVE' for s in state['pair_recovery'].values())
    assert state['portfolio_recovery']['phase'] == 'ACTIVE'
    assert all(i['state'] == 'COMPLETED' for i in state['reentry_intents'].values())
    exchange_orders(known_ids={p: set(ledger['open_order_ids']) for p, ledger in state['ledgers'].items()})
    subprocess.run(['docker', 'stop', '--time', '60', 'grid-live-fdusd-400'], check=True)
    info = json.loads(subprocess.check_output(['docker', 'inspect', 'grid-live-fdusd-400']))[0]
    assert info['State']['ExitCode'] == 0, 'Grid shutdown incomplete; inspect before continuing'
    exchange_orders()
    shutil.copy2(path, ROLLOUT / 'backup/grid_stopped_runtime.json')
    for rel in ['scripts/walk_forward_portfolio_grid_live.py',
                'api-files/bots/scripts/walk_forward_portfolio_grid_live.py']:
        target = ROOT / rel
        temp = target.with_suffix('.reentry-new')
        shutil.copy2(SOURCE, temp)
        os.replace(temp, target)
    subprocess.run(['docker', 'start', 'grid-live-fdusd-400'], check=True)
    print('Grid reloaded with source SHA256:', hashlib.sha256(SOURCE.read_bytes()).hexdigest())
    print('No balance, ledger, risk state or model authorization rewritten.')


if __name__ == '__main__':
    main()

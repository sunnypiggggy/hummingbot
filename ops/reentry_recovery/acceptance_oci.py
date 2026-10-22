"""Read-only acceptance of restored orders, ownership and loaded artifacts."""
import hashlib
import json
from pathlib import Path
import subprocess
import time

ROOT = Path('/home/ubuntu/extra_drive/hummingbot')
ROLLOUT = ROOT / '.reentry-rollouts/20261021-recovery-v1'


def main():
    state = json.loads((ROOT / 'api-files/bots/instances/grid-live-fdusd-400/data/live_grid_runtime_state.json').read_text())
    assert time.time() - state['updated_at'] < 30
    source = ROOT / 'scripts/walk_forward_portfolio_grid_live.py'
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    assert state['loaded_source_sha256'] == state['disk_source_sha256'] == digest
    scheduler_digest = subprocess.check_output(['docker', 'exec', 'grid-live-fdusd-scheduler',
                                               'sha256sum', '/app/walk_forward_portfolio_grid_live.py'], text=True).split()[0]
    assert scheduler_digest == digest
    code = """import sys,json
from pathlib import Path
from decimal import Decimal
sys.path[:0]=['/app','/app/scripts']
from dca_live_guard import BinanceEmergencyClient
c=BinanceEmergencyClient.from_secret_file(Path('/run/secrets/dca_binance_emergency_credentials'))
result={}
for pair in ['BTC-FDUSD','ETH-FDUSD','BTC-USDT','ETH-USDT']:
 orders=c.open_orders(pair)
 keys=[(o['side'],str(Decimal(o['price']).normalize())) for o in orders]
 assert len(keys)==len(set(keys)), 'duplicate price-side orders: '+pair
 result[pair]={'buy':sum(o['side']=='BUY' for o in orders),'sell':sum(o['side']=='SELL' for o in orders),'sell_remaining':str(sum((Decimal(o['origQty'])-Decimal(o['executedQty']) for o in orders if o['side']=='SELL'),Decimal('0'))),'ids':[o['clientOrderId'] for o in orders]}
print(json.dumps(result))
"""
    orders = json.loads(subprocess.check_output(['docker', 'exec', 'dca-live-guard', 'python', '-c', code], text=True))
    from decimal import Decimal
    for pair in state['ledgers']:
        ledger = state['ledgers'][pair]
        assert Decimal(orders[pair]['sell_remaining']) <= Decimal(ledger['base'])
        assert set(orders[pair]['ids']) == set(ledger['open_order_ids']), pair
        assert state['order_build_status'][pair]['state'] in {'HEALTHY', 'HEALTHY_DEFERRED'}
    guard = json.loads((ROOT / 'dca-live-data/guard_state.json').read_text())
    assert time.time() - guard['last_success_at'] < 30
    assert not guard.get('last_monitor_error')
    for name, bot in guard['bots'].items():
        gate = guard['gate_aggregate']['bots'][name]
        assert bot['recovery']['phase'] == 'ACTIVE', name
        assert gate['controller_applied']
        assert gate['controller_actual_buy_enabled'] and gate['controller_actual_sell_enabled']
    inventory = json.loads((ROOT / 'account-inventory-data/account_inventory_status.json').read_text())
    assert time.time() - inventory['generated_at'] < 30
    assert inventory['sources_healthy']
    assert all(Decimal(a['ownership_deficit']) == 0 for a in inventory['assets'].values())
    errors = json.loads((ROOT / 'dca-live-data/telegram/current_runtime_errors.json').read_text())
    assert not errors['errors']
    baseline = json.loads((ROLLOUT / 'backup/containers.json').read_text())
    for name, old in baseline.items():
        live = json.loads(subprocess.check_output(['docker', 'inspect', name]))[0]
        if name not in {'dca-live-guard', 'grid-live-fdusd-scheduler', 'grid-live-fdusd-400'}:
            assert live['Id'] == old['id'] and live['State']['StartedAt'] == old['started_at'], name
    result = {'checked_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
              'grid_source_sha256': digest,
              'orders': {pair: {k: v for k, v in value.items() if k != 'ids'} for pair, value in orders.items()},
              'dca_phases': {n: b['recovery']['phase'] for n, b in guard['bots'].items()},
              'ownership_deficits': {a: v['ownership_deficit'] for a, v in inventory['assets'].items()},
              'current_runtime_errors': 0, 'non_target_containers_unchanged': True}
    (ROLLOUT / 'acceptance.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()

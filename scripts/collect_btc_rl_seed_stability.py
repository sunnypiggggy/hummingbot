"""Collect only the complete audited BTC thirty-seed comparison.

Reuse the tested immutable transfer/path-safety implementation. Its export
contract is explicitly extended to one coin and thirty seeds, including the
separate seed-contract oracle; no financial calculations or files are changed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts import collect_single_pair_rl as transfer

BASE_COLLECTOR_SHA256 = "29ca746540e22c4ee45531b8456ff4526629a42c1c39be7f808ce82143bc2751"


def extended_export(template):
    replacements = [
        ("assert spec['symbols']==['BTCUSDT','ETHUSDT'] and spec['seeds']==list(range(5)) and set(spec['costs'])=={'base','stress'}",
         "assert spec['symbols']==['BTCUSDT'] and spec['seeds']==list(range(30)) and set(spec['costs'])=={'base','stress'}\nassert spec['training_model_count']==60"),
        ("assert spec['algorithms'] and len(spec['algorithms'])==len(set(spec['algorithms'])) and set(spec['algorithms'])<=set(['sac','tqc','crossq','recurrent_ppo','td3','droq'])",
         "assert len(spec['algorithms'])==2 and set(spec['algorithms'])=={'sac','td3'}"),
        ("stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')", """
for split in ['validation','test']:
    extra=read_bound(split+'_seed_contract_audit.json')
    assert extra['passed'] is True and extra['failure_count']==0 and not extra['failures'] and extra['checks_total']>0
    assert extra['core_financial_audit_passed'] is True and extra['core_financial_audit_sha256']==bound[split+'_audit.json']
    assert extra['inputs_sha256']['protocol.json']==bound['protocol.json']
    for name,expected in extra['inputs_sha256'].items():
        assert digest(safe(name))==expected and (name not in bound or bound[name]==expected)
        bound[name]=expected
counter=read_bound('training_counter_audit.json')
assert counter['passed'] is True and counter['models_checked']==60 and counter['failure_count']==0 and not counter['failures']
assert counter.get('core_protocol_sha256',counter.get('protocol_sha256'))==bound['protocol.json']
assert len(counter['models'])==60 and {(r['symbol'],r['algorithm'],r['seed']) for r in counter['models']}=={('BTCUSDT',a,s) for a in ['sac','td3'] for s in range(30)}
for model in counter['models']:
    name='training/BTCUSDT/'+model['algorithm']+'/seed'+str(model['seed'])+'/record.json'
    assert digest(safe(name))==model['record_sha256']
    bound[name]=model['record_sha256']
stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
""".strip()),
    ]
    for original, replacement in replacements:
        if template.count(original) != 1:
            raise ValueError("The immutable transfer template contract changed")
        template = template.replace(original, replacement)
    compile(template.replace("REMOTE_ROOT", repr("/unused/export-contract-check")), "<eth-seed-export>", "exec")
    return template


def collect(host, remote_root, output, remote_python):
    if (Path(transfer.__file__).resolve() != ROOT / "scripts/collect_single_pair_rl.py"
            or transfer.sha(transfer.__file__) != BASE_COLLECTOR_SHA256):
        raise ValueError("Use the reviewed unchanged base transfer implementation")
    original = transfer.REMOTE_EXPORT
    try:
        transfer.REMOTE_EXPORT = extended_export(original)
        return transfer.collect(host, remote_root, output, remote_python)
    finally:
        transfer.REMOTE_EXPORT = original


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="v100-192-168-8-120")
    parser.add_argument("--remote-root", default="/home/sunny/research/btc_rl_seed_stability_20261031")
    parser.add_argument("--output", required=True)
    parser.add_argument("--remote-python", default="/home/sunny/research-envs/btceth-rl-20261029/bin/python")
    args = parser.parse_args()
    print(json.dumps(collect(args.host, args.remote_root, args.output, args.remote_python)))


if __name__ == "__main__":
    main()

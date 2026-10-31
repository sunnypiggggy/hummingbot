"""Transfer the complete SAC search and final comparison without changing evidence.

The financial and selection audits belong to the frozen independent verifier.
This entry extends only the remote export contract; archive membership, link
checks, checksums and preservation of local files reuse the reviewed transfer.
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
PLAN_SHA256 = "8ab80ceb21720691af055c7ed3505b69997087437f87315484ba2c64af625c36"
AUDITOR_NAME = "scripts/audit_sac_validation_optimization_v3.py"
ORIGINAL_AUDITOR_SHA256 = "f84e18ea2d587cbce962f3c9ec9f7385fb56ce5f91284f7dd8c102f6d92fe3ad"
PARENT_AUDITOR_SHA256 = "a53f412c748899bec6cc961316eedad52c354f522248a5f8b9d6d99b279ab468"


def optimization_export(template, auditor_sha256):
    if len(auditor_sha256) != 64 or any(c not in "0123456789abcdef" for c in auditor_sha256):
        raise ValueError("A frozen independent verifier checksum is required")
    first = "status=read_bound('pipeline_status.json')"
    last = "stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')"
    if template.count(first) != 1 or template.count(last) != 1:
        raise ValueError("The immutable base export boundary changed")
    start, stop = template.index(first), template.index(last)
    contract = '''
status=read_bound('pipeline_status.json')
assert status['status']=='completed','Study has not completed its pipeline'
launch=read_bound('pipeline_launch.json')
assert type(launch['pid']) is int and launch['pid']>1
assert not (Path('/proc')/str(launch['pid'])).exists(),'Wait for the exact supervisor to finish writing'
spec=read_bound('protocol.json')
assert spec['symbols']==['BTCUSDT','ETHUSDT'] and spec['capital_per_account']==10000
assert spec['ensemble'] is False and spec['capital_shared_between_pairs'] is False
assert spec['costs']=={'base':{'fee':.001,'slippage':.0002},'stress':{'fee':.002,'slippage':.0005}}
assert digest(safe('search_plan.json'))==PLAN_DIGEST,'The predeclared search plan changed'
bound['search_plan.json']=PLAN_DIGEST
revision=read_bound('audit_revision.json')
assert revision['schema_version']==1 and revision['revision_id']=='native_sac_policy_kwargs_default_false_v2'
assert revision['reason_code']=='sb3_sac_saved_policy_kwargs_use_sde_false'
assert revision['original_protocol_sha256']==bound['protocol.json'] and revision['search_plan_sha256']==PLAN_DIGEST
assert revision['original_audit_source_sha256']==BASE_AUDIT_DIGEST
assert digest(safe('extension_source/scripts/audit_sac_validation_optimization.py'))==BASE_AUDIT_DIGEST
bound['extension_source/scripts/audit_sac_validation_optimization.py']=BASE_AUDIT_DIGEST
extension=read_bound('extension_manifest.json')
assert revision['original_extension_manifest_sha256']==bound['extension_manifest.json']
assert revision['original_pipeline_launch_sha256']==bound['pipeline_launch.json']
assert all(revision[k] is False for k in ['financial_math_changed','model_recipe_changed','training_restarted'])
parent_source_name='audit_revision_source/scripts/audit_sac_validation_optimization_v2.py'
assert revision['revised_audit_source_path']==parent_source_name and revision['revised_audit_source_sha256']==PARENT_AUDIT_DIGEST
assert digest(safe(parent_source_name))==PARENT_AUDIT_DIGEST,'The passed predecessor verifier changed'
bound[parent_source_name]=PARENT_AUDIT_DIGEST
controller_name='audit_revision_source/scripts/continue_sac_validation_optimization.py'
assert revision['controller_source_path']==controller_name
assert digest(safe(controller_name))==revision['controller_source_sha256']
bound[controller_name]=revision['controller_source_sha256']
active_launch=read_bound('pipeline_launch_revision.json')
assert type(active_launch['pid']) is int and active_launch['pid']>1 and active_launch['training_restarted'] is False
assert active_launch['original_pipeline_launch_sha256']==bound['pipeline_launch.json']
assert active_launch['audit_revision_sha256']==bound['audit_revision.json']
assert active_launch['controller_source_sha256']==bound[controller_name]
assert active_launch['revised_audit_source_sha256']==bound[parent_source_name]
assert not (Path('/proc')/str(active_launch['pid'])).exists(),'Wait for the versioned audit controller to finish writing'
import_revision=read_bound('audit_import_revision.json')
assert import_revision['schema_version']==1 and import_revision['revision_id']=='retained_import_population_shadowing_v3'
assert import_revision['reason_code']=='import_manifest_expected_digest_shadowing'
assert import_revision['original_protocol_sha256']==bound['protocol.json'] and import_revision['search_plan_sha256']==PLAN_DIGEST
assert import_revision['parent_audit_revision_sha256']==bound['audit_revision.json']
assert import_revision['parent_auditor_source_sha256']==bound[parent_source_name]
assert all(import_revision[k] is False for k in ['financial_math_changed','model_recipe_changed','selection_changed','training_restarted'])
selection=read_bound('selection.json');selection_audit=read_bound('selection_audit.json')
assert import_revision['selection_sha256']==bound['selection.json'] and import_revision['selection_audit_sha256']==bound['selection_audit.json']
failed=read_bound('final/validation_optimization_audit.json')
assert failed['passed'] is False and failed['failure_count']>0
assert import_revision['failed_final_validation_audit_sha256']==bound['final/validation_optimization_audit.json']
sidecar=read_bound('audit_import_revision_receipt.json')
assert sidecar['sha256']==bound['audit_import_revision.json']
source_name='audit_import_revision_source/scripts/audit_sac_validation_optimization_v3.py'
assert import_revision['revised_audit_source_path']==source_name and import_revision['revised_audit_source_sha256']==AUDITOR_DIGEST
assert digest(safe(source_name))==AUDITOR_DIGEST,'The final independent verifier changed'
bound[source_name]=AUDITOR_DIGEST
finisher_name='audit_import_revision_source/scripts/finish_sac_fallback_audit_v3.py'
assert import_revision['finisher_source_path']==finisher_name
assert digest(safe(finisher_name))==import_revision['finisher_source_sha256']
bound[finisher_name]=import_revision['finisher_source_sha256']
final_launch=read_bound('pipeline_launch_import_revision.json')
assert type(final_launch['pid']) is int and final_launch['pid']>1 and final_launch['training_restarted'] is False
assert final_launch['audit_import_revision_sha256']==bound['audit_import_revision.json']
assert final_launch['finisher_source_sha256']==bound[finisher_name] and final_launch['revised_audit_source_sha256']==bound[source_name]
assert final_launch['parent_pipeline_launch_revision_sha256']==bound['pipeline_launch_revision.json']
assert not (Path('/proc')/str(final_launch['pid'])).exists(),'Wait for final versioned audit delivery to finish writing'
complete=read_bound('completion_contract_audit.json')
assert complete['passed'] is True and complete['failure_count']==0 and not complete['failures'] and complete['checks_total']>0
assert complete['protocol_sha256']==bound['protocol.json']
assert complete['expected_tuning_models']==40 and complete['final_models']==60
assert complete['inputs_sha256'] and complete['external_inputs_sha256']
for name,expected in complete['inputs_sha256'].items():
    assert digest(safe(name))==expected and (name not in bound or bound[name]==expected),'Current study differs from completed independent audits'
    bound[name]=expected
external_bound=dict(complete['external_inputs_sha256'])
def verify_external():
    for name,expected in external_bound.items():
        path=Path(name)
        assert path.is_absolute() and str(path)==name and '..' not in path.parts
        assert path.is_relative_to(root.parent) and path.is_file()
        assert not any(p.is_symlink() for p in (path,*path.parents))
        assert digest(path)==expected,'Retained reference evidence changed'
verify_external()
import importlib.util,sys
loader=importlib.util.spec_from_file_location('frozen_sac_export_verifier',safe(source_name))
auditor=importlib.util.module_from_spec(loader);sys.modules[loader.name]=auditor;loader.loader.exec_module(auditor)
verified=auditor.verify_completed_study(root)
assert verified['passed'] is True and verified['failure_count']==0 and not verified['failures']
assert verified['inputs_sha256']==complete['inputs_sha256'] and verified['external_inputs_sha256']==external_bound
verify_external()
'''.replace("PLAN_DIGEST", repr(PLAN_SHA256)).replace("BASE_AUDIT_DIGEST", repr(ORIGINAL_AUDITOR_SHA256)).replace("PARENT_AUDIT_DIGEST", repr(PARENT_AUDITOR_SHA256)).replace("AUDITOR_DIGEST", repr(auditor_sha256))
    result = template[:start] + contract.strip() + "\n" + template[stop:]
    before = "assert all(digest(safe(name))==expected for name,expected in bound.items()),'Audited evidence changed during export'"
    if result.count(before) != 1:
        raise ValueError("The base archive recheck boundary changed")
    result = result.replace(before, before + "\nverify_external()")
    needle = "receipt=dict(exported_at="
    if result.count(needle) != 1:
        raise ValueError("The base receipt contract changed")
    result = result.replace(needle, "receipt=dict(completion_contract_audit_sha256=bound['completion_contract_audit.json'],audit_revision_sha256=bound['audit_revision.json'],audit_import_revision_sha256=bound['audit_import_revision.json'],external_inputs_sha256=external_bound,external_verified_sha256=external_bound,external_rechecked_after_archive=True,verifier_source_sha256=bound[source_name],exported_at=")
    compile(result.replace("REMOTE_ROOT", repr("/unused/sac-export-contract")), "<sac-optimization-export>", "exec")
    return result


def collect(host, remote_root, output, remote_python):
    if Path(transfer.__file__).resolve() != ROOT / "scripts/collect_single_pair_rl.py" or transfer.sha(transfer.__file__) != BASE_COLLECTOR_SHA256:
        raise ValueError("Use the unchanged reviewed transfer implementation")
    auditor = ROOT / AUDITOR_NAME
    if not auditor.is_file():
        raise FileNotFoundError("Finish and freeze the independent optimization verifier first")
    original = transfer.REMOTE_EXPORT
    try:
        transfer.REMOTE_EXPORT = optimization_export(original, transfer.sha(auditor))
        return transfer.collect(host, remote_root, output, remote_python)
    finally:
        transfer.REMOTE_EXPORT = original


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="v100-192-168-8-120")
    parser.add_argument("--remote-root", default="/home/sunny/research/sac_validation_optimization_20261031")
    parser.add_argument("--remote-python", default="/home/sunny/research-envs/btceth-rl-20261029/bin/python")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(collect(args.host, args.remote_root, args.output, args.remote_python)))


if __name__ == "__main__":
    main()

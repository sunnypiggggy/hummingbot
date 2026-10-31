"""Future annual permission/source tests only; no real annual market is read."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import pytest
from scripts import btceth_rl_execution_band_annual as annual


def auth():
    return dict(authorized_at='2026-10-30T00:02:00Z',annual_evaluation_authorized=True,
        selection_sha256='a'*64,financial_parameters_fixed=True,no_live_trading=True,
        designed_after_first_annual=True,test_year_already_seen=True,not_fresh_holdout=True)


def test_explicit_authorization_is_bound_to_sealed_winner_and_later_time():
    annual.authorization_gate(auth(),'a'*64,{'sealed_at':'2026-10-30T00:01:00Z'})


@pytest.mark.parametrize('field,value',[
    ('annual_evaluation_authorized',False),('financial_parameters_fixed',False),('no_live_trading',False),
    ('selection_sha256','b'*64),('designed_after_first_annual',False),('test_year_already_seen',False),
    ('not_fresh_holdout',False),('authorized_at','2026-10-30T00:00:00Z'),
])
def test_missing_permission_different_selection_or_false_fresh_claim_rejected(field,value):
    value_auth=auth();value_auth[field]=value
    with pytest.raises(ValueError,match='authorization'):annual.authorization_gate(value_auth,'a'*64,{'sealed_at':'2026-10-30T00:01:00Z'})


def write(path,value):
    path.write_text(json.dumps(value),encoding='utf-8')


@pytest.mark.parametrize('mutation', ['old_winner','partial_audit','not_selected','false_seen'])
def test_only_independently_audited_strict_new_validation_winner_is_eligible(tmp_path,monkeypatch,mutation):
    monkeypatch.setattr(annual.band,'verify_protocol',lambda *args:{})
    selected=dict(candidate=dict(family='cap075_band05',name='band05',rebalance_band=.05,algorithm='tqc',profile='balanced'),
        seed_selection=False,selection_uses_test=False,**{k:True for k in annual.FLAGS})
    passed=dict(passed=True,status='passed',selection_complete=True,resource_adapter_disclosed=True,
        run_count=12,comparator_run_count=12,repeat_count=12,failure_count=0,test_complete=False,audited_at='2026-10-30T00:03:00Z')
    if mutation=='old_winner':selected['candidate'].update(family='cap075',name='reference_band025',rebalance_band=.025)
    elif mutation=='partial_audit':passed['run_count']=6
    elif mutation=='not_selected':passed['selection_complete']=False
    else:selected['not_fresh_holdout']=False
    write(tmp_path/'selection.json',selected);write(tmp_path/'audit.json',passed)
    write(tmp_path/'protocol.json',{})
    write(tmp_path/'selection_receipt.json',dict(selection_file='selection.json',selection_sha256=annual.sha(tmp_path/'selection.json'),
        protocol_sha256=annual.sha(tmp_path/'protocol.json'),sealed_at='2026-10-30T00:02:00Z'))
    with pytest.raises(ValueError,match='must win'):annual.validation_gate(tmp_path,tmp_path,tmp_path,tmp_path/'unused','audit.json')


def test_no_authorization_or_protocol_reaches_price_loader(tmp_path,monkeypatch):
    monkeypatch.setattr(annual.formal,'load_dataset',lambda *a:pytest.fail('Annual price loading forbidden'))
    monkeypatch.setattr(annual.resource,'snapshot',lambda *a:pytest.fail('No GPU activity permitted before gate'))
    with pytest.raises(FileNotFoundError):annual.run_test(tmp_path,tmp_path,tmp_path,tmp_path/'absent',tmp_path)
    assert not (tmp_path/'test_started.json').exists() and not (tmp_path/'test').exists()


def test_marker_must_match_authorization_before_any_gpu_or_price_load(tmp_path,monkeypatch):
    monkeypatch.setattr(annual,'verify_protocol',lambda *a:dict(authorization_sha256='a',validation_selection_sha256='b'))
    write(tmp_path/'test_started.json',dict(protocol_sha256='wrong',authorization_sha256='wrong',validation_selection_sha256='wrong'))
    write(tmp_path/'protocol.json',{})
    monkeypatch.setattr(annual.formal,'load_dataset',lambda *a:pytest.fail('No prices on bad marker'))
    monkeypatch.setattr(annual.resource,'snapshot',lambda *a:pytest.fail('No GPU on bad marker'))
    with pytest.raises(ValueError,match='marker'):annual.run_test(tmp_path,tmp_path,tmp_path,tmp_path/'absent',tmp_path)


def test_source_preparation_without_explicit_auth_cannot_create_annual_output(tmp_path):
    command=[sys.executable,annual.__file__,'--phase','prepare']
    for name in ['validation-output','main-output','reference-output','dataset','output-dir']:command+=['--'+name,str(tmp_path/'unused')]
    result=subprocess.run(command,capture_output=True,text=True)
    assert result.returncode==2 and 'authorized file' in result.stderr
    assert not (tmp_path/'unused').exists()


def test_existing_annual_dispatch_is_preserved_before_any_process_spawn(tmp_path,monkeypatch):
    monkeypatch.setattr(annual,'verify_protocol',lambda *a:{})
    (tmp_path/'test.log').write_bytes(b'PRESERVE_SYNTHETIC_PREVIOUS_ATTEMPT')
    monkeypatch.setattr(annual.subprocess,'Popen',lambda *a,**k:pytest.fail('Must not restart an existing annual'))
    with pytest.raises(FileExistsError):annual.supervise(tmp_path,tmp_path,tmp_path,tmp_path/'absent',tmp_path)
    assert (tmp_path/'test.log').read_bytes()==b'PRESERVE_SYNTHETIC_PREVIOUS_ATTEMPT'


def test_executor_and_independent_audit_declare_same_fourteen_source_dag():
    from scripts import audit_btceth_rl_execution_band_annual as independent
    assert len(annual.SOURCE_FILES)==len(set(annual.SOURCE_FILES))==14
    assert set(annual.SOURCE_FILES)==set(independent.SOURCE_FILES)

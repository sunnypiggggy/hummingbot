"""Instance/resource-only adapter checks; no actual GPU or policy is used."""
from copy import deepcopy
import subprocess
import pytest

from scripts import btceth_rl_execution_band_gpu_guard as adapter


def declaration():
    identity=dict(pid=715236,uid=1000,username='sunny',exe='/usr/bin/ptyxis',starttime_ticks='123456',boot_id='fixture-boot')
    return dict(allowed_instance=identity,allowed_gpu_rows=[dict(gpu_uuid='GPU_A',pid=715236)])


def test_filter_deletes_only_exact_declared_row_preserving_other_bytes():
    text='GPU_A, 715236\nGPU_A, 999999\r\nGPU_B, 715236\n'
    assert adapter.filter_rows(text,declaration()['allowed_gpu_rows'])=='GPU_A, 999999\r\nGPU_B, 715236\n'
    assert adapter.filter_rows('GPU_A, 999999\n',declaration()['allowed_gpu_rows'])=='GPU_A, 999999\n'


def test_only_exact_resource_command_is_adapted_and_always_restored(monkeypatch):
    p=declaration();calls=[];events=[]
    def run(argv,*args,**kwargs):
        calls.append((argv,args,kwargs));return subprocess.CompletedProcess(argv,0,'GPU_A, 715236\nGPU_A, 999999\n','original stderr')
    monkeypatch.setattr(adapter.subprocess,'run',run)
    monkeypatch.setattr(adapter,'instance_info',lambda:p['allowed_instance'])
    with adapter.resource_adapter(p,events):
        result=adapter.subprocess.run(adapter.QUERY,capture_output=True,text=True,check=True)
        assert result.stdout=='GPU_A, 999999\n' and result.stderr=='original stderr'
        other=adapter.subprocess.run(['nvidia-smi','--query-gpu=uuid'])
        assert other.stdout=='GPU_A, 715236\nGPU_A, 999999\n'
    assert adapter.subprocess.run is run and len(events)==1 and len(calls)==2
    assert events[0]['stdout_original']=='GPU_A, 715236\nGPU_A, 999999\n'
    assert events[0]['stdout_filtered']=='GPU_A, 999999\n'


@pytest.mark.parametrize('field,value',[('starttime_ticks','new-instance'),('uid',999),('exe','/usr/bin/other'),('boot_id','other-boot')])
def test_reused_pid_or_changed_instance_is_rejected_without_hiding_task(monkeypatch,field,value):
    p=declaration();actual=deepcopy(p['allowed_instance']);actual[field]=value
    run=lambda argv,*a,**k:subprocess.CompletedProcess(argv,0,'GPU_A, 715236\n','')
    monkeypatch.setattr(adapter.subprocess,'run',run);monkeypatch.setattr(adapter,'instance_info',lambda:actual)
    with pytest.raises(ValueError,match='reused|identity'):
        with adapter.resource_adapter(p,[]):adapter.subprocess.run(adapter.QUERY)
    assert adapter.subprocess.run is run


def test_context_restores_original_subprocess_even_if_core_execution_raises(monkeypatch):
    p=declaration();run=adapter.subprocess.run
    with pytest.raises(RuntimeError):
        with adapter.resource_adapter(p,[]):raise RuntimeError('synthetic failed core')
    assert adapter.subprocess.run is run


@pytest.mark.parametrize('extra',[
    'GPU_A, 999999, python, 100 MiB\n',
    'GPU_B, 888888, python, 200 MiB\n',
])
def test_before_after_snapshots_refuse_every_other_gpu_task(extra):
    p=declaration();snapshot=dict(instance=p['allowed_instance'],stdout='GPU_A, 715236, ptyxis, 19 MiB\n'+extra)
    with pytest.raises(ValueError,match='Other GPU task'):adapter.verify_resource_snapshot(snapshot,p)


def test_declared_instance_snapshot_is_valid_without_freezing_transient_gpu_usage():
    p=declaration()
    for memory in ['19 MiB','20 MiB']:
        adapter.verify_resource_snapshot(dict(instance=p['allowed_instance'],stdout=f'GPU_A, 715236, ptyxis, {memory}\n'),p)


def test_missing_declared_terminal_is_not_silently_replaced_by_another_instance():
    with pytest.raises(ValueError,match='remain present'):
        adapter.verify_resource_snapshot(dict(instance=declaration()['allowed_instance'],stdout=''),declaration())

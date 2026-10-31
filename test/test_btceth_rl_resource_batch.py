"""Synthetic batch gates; no real SSH, PID, market, GPU or training access."""
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def batch(tmp_path, monkeypatch):
    path = Path(__file__).resolve().parents[1] / 'results/backtests/btceth_rl_droq_20261030_preflight/performance/wait_then_parallel_validation.py'
    spec = importlib.util.spec_from_file_location('_resource_batch_test', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    base, output = tmp_path / 'source', tmp_path / 'study'
    base.mkdir(); output.mkdir()
    monkeypatch.setattr(module, 'BASE', base); monkeypatch.setattr(module, 'OUT', output)
    pins = {}
    for name in module.PINS:
        p = base / name; p.write_text('synthetic source, never imported')
        pins[name] = hashlib.sha256(p.read_bytes()).hexdigest()
    monkeypatch.setattr(module, 'PINS', pins)
    module.put(output / 'accelerated_fit_process.json', dict(status='failed', exit_code=1, supervisor_pid=123))
    calls = []
    scheduler = SimpleNamespace(proc_identity=lambda pid: dict(pid=pid, exe='synthetic', starttime_ticks='1', boot_id='fixture'),
        resource_snapshot=lambda *args: {'synthetic': True})
    audit = SimpleNamespace(verify_accelerated_fit=lambda *args, **kwargs: dict(status='failed', passed=False))
    parallel = SimpleNamespace(run_parallel=lambda *args, **kwargs: calls.append((args, kwargs)))
    def load(name):
        calls.append(name)
        return {'btceth_rl_droq_scheduling.py': scheduler, 'btceth_rl_droq_scheduling_audit.py': audit,
                'btceth_rl_droq_parallel_eval.py': parallel}[name]
    monkeypatch.setattr(module, 'load', load)
    return SimpleNamespace(module=module, output=output, calls=calls, scheduler=scheduler, audit=audit)


@pytest.mark.parametrize('proof', [dict(status='running', exit_code=0), dict(status='completed', exit_code=True),
                                   dict(status='completed', exit_code=1), dict(status='failed', exit_code=0)])
def test_only_real_integer_completed_zero_permits_validation(batch, proof):
    with pytest.raises(ValueError): batch.module.gate(proof)
    batch.module.gate(dict(status='completed', exit_code=0))


def test_original_failure_is_preserved_and_no_validation_is_authorized(batch):
    m = batch.module; original = (batch.output / 'accelerated_fit_process.json').read_bytes()
    with pytest.raises(ValueError): m.supervise()
    assert m.read(batch.output / 'resource_batch_process.json')['status'] == 'failed'
    assert original == (batch.output / 'accelerated_fit_process.json').read_bytes()
    assert not (batch.output / 'parallel_validation_authorization.json').exists()
    assert 'btceth_rl_droq_parallel_eval.py' not in batch.calls


def test_incomplete_or_failed_scheduling_audit_cannot_start_financial_work(batch):
    m = batch.module
    (batch.output / 'accelerated_fit_process.json').write_text(json.dumps(dict(status='completed', exit_code=0, supervisor_pid=123)))
    with pytest.raises(RuntimeError, match='scheduling audit'): m.supervise()
    assert m.read(batch.output / 'scheduling_audit.fit.json')['passed'] is False
    assert not (batch.output / 'parallel_validation_authorization.json').exists()
    assert 'btceth_rl_droq_parallel_eval.py' not in batch.calls


def test_validation_child_uses_exact24_plan_and12_workers_per_gpu(batch):
    batch.module.run_validation()
    args, kwargs = batch.calls[-1]
    assert kwargs['workers_per_gpu'] == 12 and 'family' not in kwargs and 'split' not in kwargs
    assert kwargs['authorization'] == batch.output / 'parallel_validation_authorization.json'
    assert kwargs['source_root'] == batch.module.SOURCE


def test_changed_source_stops_before_owner_claim_or_fit_access(batch):
    m = batch.module
    (m.BASE / next(iter(m.PINS))).write_text('mutated after review')
    with pytest.raises(ValueError, match='Source identity'): m.supervise()
    assert not (batch.output / 'resource_batch.owner.json').exists()
    assert not batch.calls

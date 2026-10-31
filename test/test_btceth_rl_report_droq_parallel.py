"""Synthetic report adapter tests; original financial readbacks execute unchanged."""
from copy import deepcopy
from pathlib import Path

import pytest

from scripts import btceth_rl_report_droq_parallel as p
from scripts import btceth_rl_report_droq as h
from test.test_btceth_rl_report_droq import evidence, read, dump, mapping, touch


@pytest.fixture
def parallel_evidence(evidence, tmp_path):
    e = evidence; study = e.study; protocol = read(study / 'protocol.json'); protocol_sha = h.sha(study / 'protocol.json')
    external = tmp_path / 'explicit_external'; external.mkdir()
    execution = external / 'btceth_rl_droq_parallel_execution.py'; execution.write_bytes(b'# synthetic resource-only implementation')
    emap = {'/remote/performance/btceth_rl_droq_parallel_execution.py': str(execution), execution.name: str(execution)}
    ext_pins = {k: h.sha(v) for k, v in emap.items()}
    touch(study, p.required_files(e.chosen['family']) | p.validation_files())
    old = read(study / 'fit_process.json'); old.update(status='failed', exit_code=1); dump(study / 'fit_process.json', old)
    accelerated_complete = dict(training_completion_sha256=h.sha(study / 'training_completion.json'))
    dump(study / 'accelerated_fit_completion.json', accelerated_complete)
    (study / 'accelerated_seed4.log').write_bytes(b'synthetic completed actual new worker\n')
    accelerated = dict(status='completed', exit_code=0, worker_exit_code=0, supervisor_pid=701, child_pid=702,
        source_sha256=p.SCHEDULER_SHA, started_at='2026-10-30T00:03:00Z', completed_at='2026-10-30T00:11:00Z',
        log_sha256=h.sha(study / 'accelerated_seed4.log'), completion_sha256=h.sha(study / 'accelerated_fit_completion.json'))
    dump(study / 'accelerated_fit_process.json', accelerated)
    dump(study / 'scheduling_audit.fit.json', dict(passed=True, status='passed', training_model_count=5, external_input_sha256=ext_pins,
        original_collision_verified=True, original_0_1_preserved=True, original_2_3_start_instances_preserved=True,
        derived_cohort_envelope=dict(kind='derived_training_time_bounds_only', original_terminal_relabelled=False,
            original_terminal_status='failed', accelerated_terminal_status='completed', started_at=old['started_at'], completed_at=accelerated['completed_at'],
            original_fit_process_sha256=h.sha(study / 'fit_process.json'), accelerated_process_sha256=h.sha(study / 'accelerated_fit_process.json'))))
    (study / 'parallel_validation_runner.log').write_bytes(b'synthetic24 complete\n')
    validation = dict(status='completed', total_cases=24, completed_cases=24, coordinator_pid=801, parent_pid=800,
        original_runner_sha256=h.RUNNER_SHA256, completion_sha256=h.sha(study / 'validation_completion.json'),
        started_at='2026-10-30T00:13:00Z', completed_at='2026-10-30T00:17:00Z')
    dump(study / 'parallel_validation_process.json', validation)
    dump(study / 'resource_batch_process.json', dict(status='completed_validation_awaiting_financial_audit', exit_code=0,
        child_exit_code=0, child_pid=801, pid=800, log_sha256=h.sha(study / 'parallel_validation_runner.log')))
    initial = deepcopy(e.initial); initial['external_input_sha256'] = ext_pins
    initial['input_sha256'] = {name: h.sha(study / name) for name in p.validation_files() | h.training_files() | {'protocol.json', 'protocol_receipt.json'}}
    dump(study / p.VALIDATION_AUDIT, initial)
    selection = read(study / 'selection.json'); selection.update(audit_file=p.VALIDATION_AUDIT,
        audit_sha256=h.sha(study / p.VALIDATION_AUDIT), extension_source_sha256={execution.name: h.sha(execution)},
        validation_source_sha256={name: h.sha(study / name) for name in p.validation_files()})
    dump(study / 'selection.json', selection)
    seal = read(study / 'selection_receipt.json'); seal['selection_sha256'] = h.sha(study / 'selection.json'); dump(study / 'selection_receipt.json', seal)
    auth = read(study / 'annual_authorization.json'); auth['selection_sha256'] = h.sha(study / 'selection.json'); dump(study / 'annual_authorization.json', auth)
    marker = read(study / 'test_started.json'); marker.update(selection_sha256=h.sha(study / 'selection.json'), authorization_sha256=h.sha(study / 'annual_authorization.json')); dump(study / 'test_started.json', marker)
    completion = read(study / 'test_completion.json'); completion['selection_sha256'] = h.sha(study / 'selection.json'); dump(study / 'test_completion.json', completion)
    (study / 'parallel_test_runner.log').write_bytes(b'synthetic12 and12repeats\n')
    dump(study / 'parallel_test_process.json', dict(status='completed', exit_code=0, supervisor_pid=901, child_pid=902,
        source_sha256=h.sha(execution), started_at='2026-10-30T00:23:00Z', completed_at='2026-10-30T00:26:00Z',
        log_sha256=h.sha(study / 'parallel_test_runner.log')))
    gpus = [g['uuid'] for g in protocol['physical_gpus']]
    assignments = {row.run_id: gpus[(5 if row.primary_policy else int(row.seed)) % 2] for row in e.annual[0].itertuples()}
    dump(study / 'parallel_annual_resource_annex.json', {'case_gpu_assignments': assignments})
    group = read(study / f"test/{e.chosen['family']}/balanced/completion.json")
    group.update(device=None, case_workers=[dict(run_id=k, worker_pid=1000+i, device={'visible_gpu_uuid': v}) for i, (k, v) in enumerate(assignments.items())])
    dump(study / f"test/{e.chosen['family']}/balanced/completion.json", group)
    audit = deepcopy(e.audit); audit['external_input_sha256'] = ext_pins
    def reseal():
        audit['input_sha256'] = {k: v for k, v in mapping(study).items() if not k.startswith('independent_') and k not in ('goal_candidate_evidence.json', 'parallel_transfer_receipt.json')}
        audit['input_sha256'][p.VALIDATION_AUDIT] = h.sha(study / p.VALIDATION_AUDIT)
        dump(study / p.REMOTE_AUDIT, audit); local = deepcopy(audit); local['checks_total'] += 1; dump(study / p.LOCAL_AUDIT, local)
        dump(study / 'goal_candidate_evidence.json', dict(at='2026-10-30T00:28:00Z', protocol_sha256=protocol_sha,
            selection_sha256=h.sha(study / 'selection.json'), audit_sha256=h.sha(study / p.REMOTE_AUDIT), accounting_audit_passed=True, **audit['research_targets']))
    reseal()
    e.parallel_audit = audit; e.parallel_reseal = reseal
    e.parallel_load = lambda **kwargs: p.load_droq(study, e.main, source_root=e.source, external_map=emap, **kwargs)
    return e


def test_actual_new_metadata_keeps_all_seven_queries_and_original_failed_goal(parallel_evidence):
    e = parallel_evidence; before = mapping(e.study), mapping(e.main)
    value = e.parallel_load()
    assert [len(value['queries']['droq_' + n]['rows']) for n in ('summary', 'equity', 'monthly', 'targets', 'validation_selection', 'training', 'audit')] == [26, 9516, 338, 7, 2, 5, 2]
    assert value['targets']['positive_base_training_seeds'] == 2 and value['targets']['goal_performance_achieved'] is False
    assert read(e.study / 'fit_process.json')['status'] == 'failed'
    assert before == (mapping(e.study), mapping(e.main))


def test_even_resealed_snapshot_cannot_hide_invalid_process_or_finance(parallel_evidence):
    e = parallel_evidence
    for damage in ('fake_old_fit_success', 'partial_annual', 'misassigned_gpu', 'changed_nav'):
        if damage == 'fake_old_fit_success':
            f = e.study / 'fit_process.json'; original = f.read_bytes(); v = read(f); v['status'] = 'completed'; dump(f, v)
        elif damage == 'partial_annual':
            f = e.study / 'parallel_test_process.json'; original = f.read_bytes(); v = read(f); v['exit_code'] = 1; dump(f, v)
        elif damage == 'misassigned_gpu':
            f = e.study / 'parallel_annual_resource_annex.json'; original = f.read_bytes(); v = read(f); v['case_gpu_assignments'] = {k: 'foreign' for k in v['case_gpu_assignments']}; dump(f, v)
        else:
            f = e.study / f"test/{e.chosen['family']}/balanced/summary.csv"; original = f.read_bytes(); frame = h.pd.read_csv(f); frame.loc[0, 'net_return'] += .01; frame.to_csv(f, index=False)
        e.parallel_reseal()
        with pytest.raises(ValueError): e.parallel_load()
        f.write_bytes(original); e.parallel_reseal()


def test_external_label_mapping_is_explicit_and_hash_bound(tmp_path):
    path = tmp_path / 'source.py'; path.write_bytes(b'original')
    value = {'extension_source_sha256': {'source.py': h.sha(path)}}
    with pytest.raises(ValueError, match='explicit local path'): p.external_pins(h.Sources(), value, {})
    p.external_pins(h.Sources(), value, {'source.py': str(path)})
    path.write_bytes(b'changed')
    with pytest.raises(ValueError, match='exact local mapping'): p.external_pins(h.Sources(), value, {'source.py': str(path)})


@pytest.mark.parametrize('relative', ['scripts/portable.py', 'results/backtests/btceth_rl_droq_20261030_preflight/performance/portable.py'])
def test_local_portability_adapter_is_owned_hash_bound_and_listed_as_source(tmp_path, relative):
    source_root = tmp_path / 'source'; adapter = source_root / relative; adapter.parent.mkdir(parents=True); adapter.write_bytes(b'# synthetic metadata adapter')
    audit = {'local_metadata_portability': dict(adapter_file=str(adapter), adapter_sha256=h.sha(adapter), scope=p.PORTABILITY_SCOPE)}
    sources = h.Sources(); p.local_portability_source(sources, audit, source_root, local=True)
    assert sources.hashes[adapter.resolve().as_posix()] == h.sha(adapter)
    adapter.write_bytes(b'changed after review')
    with pytest.raises(ValueError, match='changed after'): sources.unchanged()


@pytest.mark.parametrize('damage', ['foreign_path', 'wrong_hash', 'wrong_scope', 'remote_audit', 'relative_path'])
def test_invalid_or_undeclared_portability_cannot_enter_report_sources(tmp_path, damage):
    source_root = tmp_path / 'source'; adapter = source_root / 'scripts/portable.py'; adapter.parent.mkdir(parents=True); adapter.write_bytes(b'# fixture')
    value = dict(adapter_file=str(adapter), adapter_sha256=h.sha(adapter), scope=p.PORTABILITY_SCOPE)
    local = True
    if damage == 'foreign_path':
        other = tmp_path / 'other.py'; other.write_bytes(adapter.read_bytes()); value['adapter_file'] = str(other)
    elif damage == 'wrong_hash': value['adapter_sha256'] = '0' * 64
    elif damage == 'wrong_scope': value['scope'] = 'all paths'
    elif damage == 'remote_audit': local = False
    else: value['adapter_file'] = 'scripts/portable.py'
    with pytest.raises(ValueError):
        p.local_portability_source(h.Sources(), {'local_metadata_portability': value}, source_root, local=local)

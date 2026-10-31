"""Byte-only collector fixtures; no pickle, model, network or GPU access."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import collect_btceth_rl_droq_parallel as c


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    study, main, source, external = [tmp_path / s for s in ('study', 'main', 'source', 'external')]
    for p in (study, main, source, external): p.mkdir()
    t = c.load_transport(Path(__file__).resolve().parents[1] / 'scripts/collect_btceth_rl_results.py')
    code = {}
    for i in range(9):
        p = source / f'scripts/synthetic_{i}.py'; p.parent.mkdir(exist_ok=True); p.write_bytes(str(i).encode()); code[p.relative_to(source).as_posix()] = c.sha(p)
    for name in c.required_files('cap075'): put(study / name, {})
    put(study / 'selection.json', {'candidate': {'family': 'cap075'}})
    put(study / 'protocol.json', {'source_sha256': code})
    monkeypatch.setattr(c, 'PROTOCOL_SHA', c.sha(study / 'protocol.json'))
    put(study / 'protocol_receipt.json', {'protocol_sha256': c.PROTOCOL_SHA})
    for name in ('parallel_test_process.json', 'parallel_validation_process.json'): put(study / name, {'status': 'completed'})
    put(study / 'fit_process.json', {'status': 'failed', 'actual_original_collision': True})
    for name in ('frozen_spec.json', 'runtime.json', 'independent_audit.json'): put(main / name, {'synthetic': True})
    auditor = external / 'auditor.py'; auditor.write_bytes(b'# reviewed synthetic independent auditor')
    raw = external / 'data.bin'; raw.write_bytes(b'not a pickle')
    mapping = {'/remote/auditor.py': str(auditor), '/remote/data.bin': str(raw)}
    audit = dict(passed=True, status='passed', phase='test', test_complete=True, validation_complete=True,
        training_model_count=5, validation_run_count=24, test_run_count=12, baseline_run_count=14, run_count=26, repeat_count=12,
        failure_count=0, failures=[], fresh_holdout_success=False, training_budget_multiple=1, accounting_passage_is_not_goal_success=True,
        **{flag: True for flag in c.FLAGS}, checks_total=123,
        runs={str(i): {'synthetic': True} for i in range(12)}, baseline_runs={str(i): {'synthetic': True} for i in range(14)},
        input_sha256={name: c.sha(study / name) for name in c.required_files('cap075')},
        main_input_sha256={p.name: c.sha(p) for p in main.iterdir()}, source_sha256=code,
        external_input_sha256={name: c.sha(p) for name, p in mapping.items()})
    put(study / c.AUDIT_FILE, audit)
    args = dict(transport=t, audit_source=auditor, audit_source_sha=c.sha(auditor), external_map=mapping, local=True)
    return study, main, source, t, auditor, audit, args


def test_complete_snapshot_guard_preserves_original_failed_fit(fixture):
    study, main, source, _, _, _, args = fixture
    before = (study / 'fit_process.json').read_bytes()
    result = c.guard(study, main, source, **args)
    assert result['run_count'] == 26 and result['chosen_family'] == 'cap075'
    assert before == (study / 'fit_process.json').read_bytes()


@pytest.mark.parametrize('field,value', [('passed', False), ('run_count', 25), ('training_model_count', 4), ('failure_count', 1), ('checks_total', True)])
def test_partial_or_failed_financial_audit_never_collects(fixture, field, value):
    study, main, source, _, _, audit, args = fixture
    audit[field] = value; put(study / c.AUDIT_FILE, audit)
    with pytest.raises(ValueError): c.guard(study, main, source, **args)


def test_hash_mutation_or_missing_explicit_external_mapping_is_rejected(fixture):
    study, main, source, _, _, _, args = fixture
    bad = {**args, 'external_map': {'/remote/auditor.py': str(args['audit_source'])}}
    with pytest.raises(ValueError, match='exact explicit'): c.guard(study, main, source, **bad)
    (study / 'training/droq__balanced__seed0/model_tensors.pt').write_bytes(b'changed')
    with pytest.raises(ValueError, match='changed bytes'): c.guard(study, main, source, **args)


def test_unreviewed_independent_audit_source_is_not_accepted(fixture):
    study, main, source, _, _, _, args = fixture
    with pytest.raises(ValueError, match='Root must pin'):
        c.guard(study, main, source, **{**args, 'audit_source_sha': '0' * 64})


def test_custom_archive_uses_reviewed_safe_extractor_and_exact_bytes(fixture, tmp_path, monkeypatch):
    study, main, source, t, auditor, _, args = fixture
    result = c.guard(study, main, source, **args)
    monkeypatch.setattr(c, 'guard', lambda *a, **kw: result)
    response = c.export(study, main, source, transport=t, audit_file=c.AUDIT_FILE,
        audit_source=auditor, audit_source_sha=c.sha(auditor), export_parent=tmp_path / 'exports')
    manifest = c.read(response['manifest_path']); destination = tmp_path / 'new_snapshot'
    t.verify_extract(response['archive_path'], manifest, destination)
    assert t.inventory(destination) == t.inventory(study)
    assert c.read(destination / 'fit_process.json')['status'] == 'failed'
    with pytest.raises(FileExistsError): t.verify_extract(response['archive_path'], manifest, destination)


def test_frozen_transport_source_is_not_mutated(tmp_path):
    fake = tmp_path / 'fake.py'; fake.write_bytes(b'not the reviewed transport')
    with pytest.raises(ValueError, match='reviewed54ef'): c.load_transport(fake)


def test_relative_extension_source_labels_need_exact_local_mapping(fixture):
    study, main, source, _, auditor, audit, args = fixture
    audit['external_input_sha256'].pop('/remote/auditor.py')
    audit['extension_source_sha256'] = {auditor.name: c.sha(auditor)}
    put(study / c.AUDIT_FILE, audit)
    with pytest.raises(ValueError, match='exact explicit'): c.guard(study, main, source, **args)
    mapping = {**args['external_map'], auditor.name: str(auditor)}
    assert c.guard(study, main, source, **{**args, 'external_map': mapping})['run_count'] == 26


def test_collection_cannot_move_a_snapshot_into_original_reference_tree(tmp_path):
    main = tmp_path / 'original'; main.mkdir()
    with pytest.raises(ValueError, match='independent'):
        c.collect(SimpleNamespace(output=main / 'new', main=main), transport=None)
    assert not list(main.iterdir())


def test_collection_never_overwrites_existing_output(tmp_path):
    output = tmp_path / 'existing'; output.mkdir(); sentinel = output / 'sentinel'; sentinel.write_bytes(b'preserve')
    with pytest.raises(FileExistsError):
        c.collect(SimpleNamespace(output=output, main=tmp_path / 'main'), transport=None)
    assert sentinel.read_bytes() == b'preserve'


def test_remote_export_parent_stays_an_opaque_posix_string(monkeypatch):
    found = {}
    monkeypatch.setattr(c.sys, 'argv', ['collector', '--collect', '--transport-source', 'unused', '--export-parent', '/home/sunny/research/exports'])
    monkeypatch.setattr(c, 'load_transport', lambda path: object())
    monkeypatch.setattr(c, 'collect', lambda args, **kwargs: found.update(parent=args.export_parent))
    c.main()
    assert type(found['parent']) is str and found['parent'] == '/home/sunny/research/exports'

"""Reader-routing fixtures only; no genuine parallel audit is implied."""
from copy import deepcopy

import pytest

from scripts import btceth_rl_report_app as app
from scripts import btceth_rl_report_droq as legacy
from scripts import btceth_rl_report_droq_parallel as parallel
from test.test_btceth_rl_report import complete_study_root, synthetic_summary, _droq_integration_payload


def test_explicit_parallel_marker_selects_new_reader_without_touching_old_queries(complete_study_root, monkeypatch):
    original, _ = app.make_reviewed_snapshot(complete_study_root)
    study = complete_study_root / 'parallel-synthetic'; study.mkdir(); (study / 'parallel_test_process.json').write_text('{}')
    payload = _droq_integration_payload()
    monkeypatch.setattr(legacy, 'load_droq', lambda *a, **k: pytest.fail('Parallel marker fell through to legacy terminal gates'))
    monkeypatch.setattr(parallel, 'load_droq', lambda *a, **k: deepcopy(payload))
    updated, evidence = app.make_reviewed_snapshot(complete_study_root, droq_root=study)
    assert all(updated['queries'][k] == value for k, value in original['queries'].items())
    assert evidence['droq_scope_runs'] == 26 and evidence['additional_unique_droq_runs'] == 12
    assert not (complete_study_root / 'report').exists()


def test_failed_parallel_reader_never_falls_back_to_legacy(complete_study_root, monkeypatch):
    study = complete_study_root / 'parallel-synthetic'; study.mkdir(); (study / 'parallel_test_process.json').write_text('{}')
    def reject(*a, **k): raise ValueError('Incomplete parallel financial audit')
    monkeypatch.setattr(parallel, 'load_droq', reject)
    monkeypatch.setattr(legacy, 'load_droq', lambda *a, **k: pytest.fail('A failed new gate cannot fall back'))
    with pytest.raises(ValueError, match='Incomplete parallel'):
        app.make_reviewed_snapshot(complete_study_root, droq_root=study)

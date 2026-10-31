"""One recorded-path conversion, no bypass of original checks or filesystem."""
from pathlib import Path, PurePosixPath
from types import SimpleNamespace

import pytest

from scripts.audit_btceth_rl_droq_parallel_local import stored_posix_performance_path


@pytest.mark.parametrize('raises', [False, True])
def test_only_exact_recorded_string_uses_posix_and_every_binding_is_restored(raises):
    recorded = '/home/sunny/research/performance/droq_execution_adapter.py'
    helper = SimpleNamespace(Path=Path)
    seen = []
    def original(inputs, annex, worker, actual):
        seen.append(helper.Path(recorded))
        assert type(helper.Path(actual)) is type(Path(actual))
        assert type(helper.Path(recorded + '.other')) is type(Path(recorded + '.other'))
        assert str(helper.Path(recorded).parent / 'dependency.py') == '/home/sunny/research/performance/dependency.py'
        if raises: raise ValueError('Original validator failure must propagate')
        return 'original checks completed'
    helper.verify_performance = original
    with stored_posix_performance_path(helper):
        if raises:
            with pytest.raises(ValueError, match='must propagate'):
                helper.verify_performance(None, {'performance_adapter': {'path': recorded}}, None, Path.cwd())
        else:
            assert helper.verify_performance(None, {'performance_adapter': {'path': recorded}}, None, Path.cwd()) == 'original checks completed'
    assert isinstance(seen[0], PurePosixPath)
    assert helper.Path is Path and helper.verify_performance is original

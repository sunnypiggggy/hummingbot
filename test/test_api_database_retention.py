import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from scripts.api_database_retention import DATABASE, TEST_DATABASE, plan, run


def test_default_is_nonmutating_and_target_is_fixed():
    sql=plan()
    assert f"current_database()<>'{DATABASE}'" in sql
    assert 'DELETE FROM' not in sql
    assert '370 days' in sql
    assert 'newer.timestamp>a.timestamp' in sql
    assert 'TRUNCATE' not in sql and 'CASCADE' not in sql


def test_apply_deletes_only_reviewed_snapshots_and_children_atomically():
    sql=plan(apply=True,testing=True)
    assert f"current_database()<>'{TEST_DATABASE}'" in sql
    assert sql.index('DELETE FROM public.token_states') < sql.index('DELETE FROM public.account_states')
    assert sql.count('BEGIN;')==1 and sql.count('COMMIT;')==1
    assert 'LIMIT 1000' in sql and "lock_timeout='1s'" in sql
    assert 'unreviewed incoming dependency' in sql and 'unreviewed maintenance trigger' in sql
    for table in ('orders','trades','executors','position_snapshots','bot_runs','funding_payments'):
        assert f'DELETE FROM public.{table}' not in sql


def test_busy_database_skips_without_dumping_errors():
    with patch('scripts.api_database_retention.subprocess.run',return_value=SimpleNamespace(returncode=1,stderr='55P03 secret',stdout='')):
        assert run(apply=True)['reason']=='database_busy_retry_next_day'


def test_unknown_error_fails_job_no_secret_echo():
    with patch('scripts.api_database_retention.subprocess.run',return_value=SimpleNamespace(returncode=1,stderr='secret',stdout='')):
        with pytest.raises(RuntimeError,match='no partial transaction committed') as error:
            run(apply=True)
        assert 'secret' not in str(error.value)


def test_fixed_psql_no_credentials_in_command():
    result={'database':DATABASE,'deleted':{}}
    with patch('scripts.api_database_retention.subprocess.run',return_value=SimpleNamespace(returncode=0,stderr='',stdout=json.dumps(result))) as mocked:
        assert run()==result
    assert DATABASE in mocked.call_args.args[0]
    assert 'binance-stocks-runtime' not in mocked.call_args.args[0]

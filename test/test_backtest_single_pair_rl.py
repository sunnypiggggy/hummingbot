"""Independent runner acceptance using synthetic markets and frozen fake policies.

Real SinglePairEnv, metrics, canonical saves and replays are exercised. Model
construction/GPU dispatch and real market loading are explicit mock boundaries.
"""
from copy import deepcopy
import hashlib
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from scripts import backtest_single_pair_rl as runner
from scripts import backtest_btceth_rl as old_loader
from scripts import single_pair_rl_models as models
from scripts.single_pair_rl_env import SinglePairEnv


@pytest.fixture(scope='module')
def synthetic_market():
    index = pd.date_range('2019-01-01', '2026-10-25', freq='h', inclusive='left', tz='UTC', name='timestamp')
    close = pd.DataFrame({'BTCUSDT': 100., 'ETHUSDT': 200.}, index=index)
    data = {f'hourly_{f}': close.copy() for f in ('open', 'high', 'low', 'close', 'volume')}
    data['hourly_high'] *= 1.001; data['hourly_low'] *= .999; data['hourly_volume'].loc[:, :] = 10.
    data['assets'] = {s: {'symbol': s[:-4], 'binance_symbol': s} for s in runner.SYMBOLS}
    data['cached_portfolio_features'] = 'This must never enter a new feature constructor'
    return data


@pytest.fixture
def prepared(tmp_path, monkeypatch, synthetic_market):
    root = tmp_path / 'source'; names = ('scripts/backtest_single_pair_rl.py', 'scripts/single_pair_rl_env.py', 'scripts/single_pair_rl_models.py')
    for name in names:
        p = root / name; p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(('synthetic source pin:' + name).encode())
    monkeypatch.setattr(runner, 'ROOT', root); monkeypatch.setattr(runner, 'SOURCE_FILES', names)
    monkeypatch.setattr(old_loader, 'load_dataset', lambda path: synthetic_market)
    monkeypatch.setattr(models, 'recipe', lambda algorithm: {'synthetic_algorithm': algorithm, 'observation_dim': 13, 'action_dim': 1})
    monkeypatch.setattr(models, 'runtime_versions', lambda: {'synthetic_fixture_only': 'pinned'})
    data = tmp_path / 'data/dataset.pkl.gz'; data.parent.mkdir(); data.write_bytes(b'Synthetic bytes only, never unpickled')
    runner.write(data.parent / 'manifest.json', {'synthetic_only': True})
    output = tmp_path / 'study'; spec = runner.prepare(output, data, ('sac', 'tqc'))
    return SimpleNamespace(root=root, dataset=data, output=output, spec=spec, names=names)


def test_crop_excludes_future_raw_ohlcv_and_cached_portfolio_state_before_features(synthetic_market):
    end = pd.Timestamp(runner.TRAIN[1]); cut = runner.cropped_data(synthetic_market, end)
    assert set(cut) == {'assets', *(f'hourly_{f}' for f in ('open', 'high', 'low', 'close', 'volume'))}
    for f in ('open', 'high', 'low', 'close', 'volume'):
        actual = cut['hourly_' + f]; expected = synthetic_market['hourly_' + f].loc[lambda x: x.index < end]
        assert_frame_equal(actual, expected, check_exact=True)
        assert actual.index.max() == end - pd.Timedelta(hours=1)
    # Building the actual new features receives only the already cropped inputs.
    e = SinglePairEnv(cut, 'BTCUSDT', *runner.TRAIN, training=True, episode_days=32, record_history=False)
    assert e._index[-1] < end and max(e._features) == end
    cut['hourly_open'].iloc[0, 0] = -1
    assert synthetic_market['hourly_open'].iloc[0, 0] == 100.


def test_prepare_freezes_twenty_independent_13_by1_models_and_exact_source_copies(prepared):
    p = prepared; s = p.spec
    assert s['training_model_count'] == 20 and s['algorithms'] == ['sac', 'tqc']
    assert s['capital_per_account'] == 10000. and s['capital_shared_between_pairs'] is False and s['ensemble'] is False
    assert s['environment']['observation_dim'] == 13 and s['environment']['action_dim'] == 1
    assert s['primary_seed'] == 0 and s['primary_seed_fixed_before_training'] is True
    assert s['validation_selection']['seed_selection'] is False and s['fresh_holdout'] is False
    assert runner.read(p.output / 'protocol_receipt.json')['protocol_sha256'] == runner.sha(p.output / 'protocol.json')
    assert set(s['sources']) == set(p.names)
    for name in p.names: assert (p.root / name).read_bytes() == (p.output / 'frozen_source' / name).read_bytes()
    assert runner.check_protocol(p.output, p.dataset) == s
    with pytest.raises(FileExistsError): runner.prepare(p.output, p.dataset, ('sac', 'tqc'))


@pytest.mark.parametrize('damage', ['source', 'dataset', 'manifest', 'protocol', 'missing_source_pin', 'primary_seed', 'observation_dim', 'shared_cash'])
def test_frozen_identity_rejects_byte_tampering_and_resealed_wrong_semantics(prepared, damage):
    p = prepared
    if damage == 'source': (p.root / p.names[0]).write_bytes(b'changed producer')
    elif damage == 'dataset': p.dataset.write_bytes(b'changed market')
    elif damage == 'manifest': runner.write(p.dataset.parent / 'manifest.json', {'changed': True})
    else:
        s = deepcopy(p.spec)
        if damage == 'missing_source_pin': s['sources'].pop(p.names[0])
        elif damage == 'primary_seed': s['primary_seed'] = 4
        elif damage == 'observation_dim': s['environment']['observation_dim'] = 27
        elif damage == 'shared_cash': s['capital_shared_between_pairs'] = True
        else: s['capital_per_account'] = 5000.
        runner.write(p.output / 'protocol.json', s)
        if damage != 'protocol':
            receipt = runner.read(p.output / 'protocol_receipt.json'); receipt['protocol_sha256'] = runner.sha(p.output / 'protocol.json'); runner.write(p.output / 'protocol_receipt.json', receipt)
    with pytest.raises(ValueError): runner.check_protocol(p.output, p.dataset)


def test_fit_grid_and_validation_grid_are20_models_and56_separate_accounts(prepared, monkeypatch):
    p = prepared; grids = {}
    def fake_train_pool(output, tasks, function, phase, max_workers=None):
        grids[phase] = list(tasks)
        for task in tasks:
            write_training_record(output, task['symbol'], task['algorithm'], task['seed'], p.spec)
        return []
    monkeypatch.setattr(runner, 'run_parallel', fake_train_pool); runner.fit(p.output, p.dataset)
    assert len(grids['fit']) == 20 and len({(x['symbol'], x['algorithm'], x['seed']) for x in grids['fit']}) == 20
    assert runner.read(p.output / 'fit_complete.json')['model_count'] == 20
    def fake_eval_pool(output, tasks, function, phase, max_workers=None):
        grids[phase] = list(tasks)
        return [{'summary': dict(run_id=f"{x['split']}__{x['scenario']}__{x['symbol']}__{x['algorithm']}__{x['seed'] if x['seed'] is not None else 'baseline'}", symbol=x['symbol'], algorithm=x['algorithm'], seed=x['seed'], scenario=x['scenario'], split=x['split']), 'repeat_equal': True} for x in tasks]
    monkeypatch.setattr(runner, 'run_parallel', fake_eval_pool); frame = runner.evaluate(p.output, p.dataset, 'validation')
    assert len(frame) == len(grids['validation']) == 56
    assert sum(x['seed'] is not None for x in grids['validation']) == 40
    assert sum(x['seed'] is None for x in grids['validation']) == 16
    assert len({(x['symbol'], x['algorithm'], x['seed'], x['scenario']) for x in grids['validation']}) == 56


def selection_fixture(p):
    rows = []
    for symbol in runner.SYMBOLS:
        for algorithm in p.spec['algorithms']:
            returns = ([10., -.1, -.1, -.1, -.1] if algorithm == 'sac' else [.04] * 5) if symbol == 'BTCUSDT' else ([.06] * 5 if algorithm == 'sac' else [10., 0., 0., 0., 0.])
            for seed in runner.SEEDS:
                for scenario in runner.COSTS:
                    rows.append(dict(run_id=f'validation__{scenario}__{symbol}__{algorithm}__{seed}', symbol=symbol, algorithm=algorithm, seed=seed, scenario=scenario, split='validation', initial_capital=10000., net_return=returns[seed], max_drawdown=-.1, ensemble=False, primary_seed=seed == 0))
        for algorithm in runner.BASELINES:
            for scenario in runner.COSTS:
                rows.append(dict(run_id=f'validation__{scenario}__{symbol}__{algorithm}__baseline', symbol=symbol, algorithm=algorithm, seed=None, scenario=scenario, split='validation', initial_capital=10000., net_return=0., max_drawdown=0., ensemble=False, primary_seed=False))
    frame = pd.DataFrame(rows); frame.to_csv(p.output / 'validation_summary.csv', index=False)
    audit = dict(passed=True, phase='validation', expected_accounts=56, accounts_checked=56, repeats_checked=56, checks_total=123, failure_count=0, failures=[], summary_sha256=runner.sha(p.output / 'validation_summary.csv'), protocol_sha256=runner.sha(p.output / 'protocol.json'))
    runner.write(p.output / 'validation_audit.json', audit)
    return frame, audit


def test_median_algorithm_selection_uses_only_audited_validation_and_fixed_seed_zero(prepared):
    p = prepared; selection_fixture(p); chosen = runner.select(p.output)
    assert chosen['BTCUSDT']['algorithm'] == 'tqc' and chosen['ETHUSDT']['algorithm'] == 'sac'
    assert all(value['seed'] == 0 for value in chosen.values())
    actual = runner.read(p.output / 'validation_selection.json')
    assert actual['uses_test'] is False and actual['seed_selection'] is False
    assert actual['validation_summary_sha256'] == runner.sha(p.output / 'validation_summary.csv')
    assert not (p.output / 'evaluation').exists() and not (p.output / 'test_started.json').exists()


@pytest.mark.parametrize('damage', ['audit_failed', 'changed_summary', 'mixed_test', 'duplicate_base', 'wrong_capital', 'wrong_protocol_hash'])
def test_selector_rejects_failed_stale_or_wrong_population_even_when_summary_resealed(prepared, damage):
    p = prepared; frame, audit = selection_fixture(p)
    if damage == 'audit_failed': audit['passed'] = False
    elif damage == 'wrong_protocol_hash': audit['protocol_sha256'] = '0' * 64
    elif damage == 'mixed_test': frame.loc[0, 'split'] = 'test'
    elif damage == 'duplicate_base': frame.loc[1] = frame.loc[0]
    elif damage == 'wrong_capital': frame.loc[0, 'initial_capital'] = 5000.
    else: frame.loc[0, 'net_return'] += .5
    frame.to_csv(p.output / 'validation_summary.csv', index=False)
    if damage != 'changed_summary': audit['summary_sha256'] = runner.sha(p.output / 'validation_summary.csv')
    runner.write(p.output / 'validation_audit.json', audit)
    with pytest.raises(ValueError): runner.select(p.output)
    assert not (p.output / 'validation_selection.json').exists()


class FixedPredictor:
    weight = .6
    def __init__(self, symbol='BTCUSDT', algorithm='sac', seed=0):
        self.calls = []
        self._single_pair_identity = dict(symbol=symbol, algorithm=algorithm, seed=seed,
            observation_shape=[13], action_shape=[1], independent_account=True, multi_model_target_average=False)


def fake_predict(model, obs, *, algorithm, state=None, episode_start=False):
    assert algorithm == 'sac' and obs.shape == (13,) and obs.dtype == np.float32
    assert (state is None) is bool(episode_start)
    model.calls.append((episode_start, obs.copy()))
    return np.array([model.weight], dtype=np.float32), 1 if state is None else state + 1


@pytest.fixture(scope='module')
def annual_results(tmp_path_factory, synthetic_market):
    output = tmp_path_factory.mktemp('single_annual'); loaded = []; digests = []; results = {}
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(old_loader, 'load_dataset', lambda path: synthetic_market)
        def load(path, *, algorithm, device, upstream_root=None):
            assert algorithm == 'sac' and device == 'cuda:0'
            path = Path(path)
            model = FixedPredictor(path.parents[2].name, algorithm, int(path.parent.name.removeprefix('seed')))
            loaded.append(model); return model
        def digest(model, algorithm):
            assert algorithm == 'sac'; digests.append((id(model), algorithm)); return hashlib.sha256(str(model.weight).encode()).hexdigest()
        patch.setattr(models, 'load_model', load); patch.setattr(models, 'parameter_digest', digest); patch.setattr(models, 'predict', fake_predict)
        for symbol in runner.SYMBOLS:
            for scenario in runner.COSTS:
                results[(symbol, scenario)] = runner.evaluate_case(dict(output=str(output), dataset='synthetic_not_read.pkl', symbol=symbol, algorithm='sac', seed=0, scenario=scenario, split='test'))
        yield SimpleNamespace(output=output, results=results, loaded=loaded, digests=digests)


def test_full365_canonical_case_and_repeat_saved_from_real_one_coin_ledgers(annual_results):
    case = annual_results
    assert len(case.loaded) == 4 and len({id(m) for m in case.loaded}) == 4
    for (symbol, scenario), result in case.results.items():
        canonical = f'evaluation/test/{symbol}/sac/seed0/{scenario}'; repeated = f'repeat/test/{symbol}/sac/seed0/{scenario}'
        assert result['path'] == canonical and result['repeat_path'] == repeated and result['repeat_equal'] is True
        a, b = case.output / canonical, case.output / repeated
        assert {p.name for p in a.iterdir()} == {p.name for p in b.iterdir()}
        assert all(p.read_bytes() == (b / p.name).read_bytes() for p in a.iterdir())
        hourly = pd.read_csv(a / 'equity_1h.csv.gz'); daily = pd.read_csv(a / 'daily_equity.csv'); months = pd.read_csv(a / 'monthly_returns.csv')
        assert len(hourly) == 8760 and len(daily) == 366 and len(months) == 13
        assert pd.to_datetime(hourly.timestamp, utc=True).iloc[0] == pd.Timestamp(runner.TEST[0])
        assert pd.to_datetime(hourly.timestamp, utc=True).iloc[-1] == pd.Timestamp(runner.TEST[1]) - pd.Timedelta(hours=1)
        metadata = runner.read(a / 'metadata.json'); fills = pd.read_csv(a / 'fills.csv'); actions = runner.read(a / 'actions.json')
        assert metadata['pair'] == symbol and metadata['initial_capital'] == metadata['independent_capital'] == 10000. and metadata['no_ensemble'] is True
        assert set(metadata['final_positions']).issubset({symbol}) and set(fills.asset_key) == {symbol}
        assert len(actions) == 365 and all(len(x['action']) == 1 and len(x['target']) == 2 for x in actions)
        assert (hourly.cash >= 0).all() and (hourly.base_value >= 0).all()
        assert result['summary']['ensemble'] is False and result['summary']['primary_seed'] is True
        assert result['summary']['fees'] == pytest.approx(float(fills.fee.sum()), abs=1e-9)
    for model in case.loaded:
        assert len(model.calls) == 730 and [i for i, (first, _) in enumerate(model.calls) if first] == [0, 365]
        assert case.digests.count((id(model), 'sac')) == 2
    for symbol in runner.SYMBOLS:
        assert case.results[(symbol, 'stress')]['summary']['net_return'] < case.results[(symbol, 'base')]['summary']['net_return']


@pytest.mark.parametrize('damage', ['split', 'symbol', 'scenario', 'seed', 'algorithm'])
def test_invalid_case_identity_stops_before_market_or_model_loading(tmp_path, monkeypatch, damage):
    task = dict(output=str(tmp_path), dataset='never_read', symbol='BTCUSDT', algorithm='sac', seed=0, scenario='base', split='test')
    task[damage] = {'split': 'other', 'symbol': 'ADAUSDT', 'scenario': 'cheap', 'seed': 9, 'algorithm': 'unknown'}[damage]
    monkeypatch.setattr(old_loader, 'load_dataset', lambda p: pytest.fail('Invalid case read market data'))
    with pytest.raises(ValueError): runner.evaluate_case(task)


def test_different_raw_actions_with_identical_cash_accounts_are_not_exact_replay(tmp_path, monkeypatch, synthetic_market):
    monkeypatch.setattr(runner, 'TEST', (runner.TEST[0], '2025-10-26T00:00:00Z'))
    monkeypatch.setattr(old_loader, 'load_dataset', lambda p: synthetic_market)
    model = FixedPredictor(); model.pass_number = 0
    monkeypatch.setattr(models, 'load_model', lambda *a, **k: model)
    monkeypatch.setattr(models, 'parameter_digest', lambda model, algorithm: 'frozen_parameter_sha')
    def negative(model, obs, *, algorithm, state=None, episode_start=False):
        if episode_start: model.pass_number += 1
        return np.array([-.1 * model.pass_number], dtype=np.float32), None
    monkeypatch.setattr(models, 'predict', negative)
    with pytest.raises(ValueError, match='replay'):
        runner.evaluate_case(dict(output=str(tmp_path), dataset='not_read', symbol='BTCUSDT', algorithm='sac', seed=0, scenario='base', split='test'))


def write_training_record(output, symbol, algorithm, seed, spec):
    """Byte/JSON fixture only; does not claim that a model was actually trained."""
    d = Path(output) / 'training' / symbol / algorithm / f'seed{seed}'
    d.mkdir(parents=True, exist_ok=True); cp = d / 'checkpoint/fake.bin'; cp.parent.mkdir(exist_ok=True); cp.write_bytes(b'synthetic checkpoint bytes, not a model')
    steps = spec['timesteps_per_model'][algorithm]; updates = steps - 512
    training = dict(actual_steps=steps, counters=dict(environment_steps=steps, gradient_update_rounds=updates,
        replay_transitions=steps, replay_add_calls=steps, optimizer_steps={'actor': updates, 'critic': updates, 'entropy': updates}))
    manifest = dict(symbol=symbol, algorithm=algorithm, seed=seed, observation_shape=[13], action_shape=[1],
        independent_account=True, multi_model_target_average=False, adapter_sha256=spec['sources']['scripts/single_pair_rl_models.py'], training=deepcopy(training))
    runner.write(d / 'checkpoint/single_pair_model.json', manifest)
    record = dict(symbol=symbol, algorithm=algorithm, seed=seed, capital=10000., ensemble=False,
        observation_shape=[13], action_shape=[1], training=training, checkpoint=deepcopy(manifest),
        maximum_input_hour=str(pd.Timestamp(runner.TRAIN[1]) - pd.Timedelta(hours=1)),
        files={p.relative_to(d).as_posix(): runner.sha(p) for p in cp.parent.iterdir()})
    runner.write(d / 'record.json', record)
    return d, record, manifest


@pytest.mark.parametrize('algorithm', ['sac', 'tqc'])
def test_completed_training_record_has_exact_owned_manifest_and_real_counter_contract(tmp_path, algorithm):
    spec = {'timesteps_per_model': {algorithm: 32768}, 'sources': {'scripts/single_pair_rl_models.py': 'a' * 64}}
    _, record, _ = write_training_record(tmp_path, 'BTCUSDT', algorithm, 0, spec)
    assert runner.validate_training_record(tmp_path, 'BTCUSDT', algorithm, 0, spec) == record


@pytest.mark.parametrize('damage', ['wrong_symbol', 'wrong_seed', 'steps', 'optimizer', 'replay', 'input_cutoff'])
def test_resealed_manifest_cannot_hide_foreign_model_or_wrong_training_counter(tmp_path, damage):
    spec = {'timesteps_per_model': {'sac': 32768}, 'sources': {'scripts/single_pair_rl_models.py': 'a' * 64}}
    d, record, manifest = write_training_record(tmp_path, 'BTCUSDT', 'sac', 0, spec)
    if damage == 'wrong_symbol': manifest['symbol'] = 'ETHUSDT'
    elif damage == 'wrong_seed': manifest['seed'] = 4
    elif damage == 'steps': record['training']['counters']['environment_steps'] -= 1
    elif damage == 'optimizer': record['training']['counters']['optimizer_steps']['actor'] -= 1
    elif damage == 'replay': record['training']['counters']['replay_add_calls'] -= 1
    else: record['maximum_input_hour'] = runner.TRAIN[1]
    # Reseal matching JSON and file hashes to exercise semantics, not just a
    # stale outer digest. No actual checkpoint payload is deserialized.
    manifest['training'] = deepcopy(record['training']); record['checkpoint'] = deepcopy(manifest)
    runner.write(d / 'checkpoint/single_pair_model.json', manifest)
    record['files'] = {p.relative_to(d).as_posix(): runner.sha(p) for p in (d / 'checkpoint').iterdir()}
    runner.write(d / 'record.json', record)
    with pytest.raises(ValueError): runner.validate_training_record(tmp_path, 'BTCUSDT', 'sac', 0, spec)


@pytest.mark.parametrize('wrong', ['symbol', 'seed'])
def test_loaded_foreign_pair_or_seed_stops_before_prediction_and_account_creation(tmp_path, monkeypatch, synthetic_market, wrong):
    monkeypatch.setattr(old_loader, 'load_dataset', lambda path: synthetic_market)
    model = FixedPredictor(); model._single_pair_identity[wrong] = 'ETHUSDT' if wrong == 'symbol' else 4
    monkeypatch.setattr(models, 'load_model', lambda *args, **kwargs: model)
    monkeypatch.setattr(models, 'parameter_digest', lambda *args: pytest.fail('Identity mismatch reached parameter/inference work'))
    monkeypatch.setattr(models, 'predict', lambda *args, **kwargs: pytest.fail('Wrong model made a prediction'))
    with pytest.raises(ValueError, match='exact independent'):
        runner.evaluate_case(dict(output=str(tmp_path), dataset='not_read', symbol='BTCUSDT', algorithm='sac', seed=0, scenario='base', split='test'))
    assert not (tmp_path / 'evaluation').exists() and not (tmp_path / 'repeat').exists()

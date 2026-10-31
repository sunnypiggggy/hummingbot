"""Single-coin economic/isolation oracles; synthetic OHLCV, no training or GPU."""
from copy import deepcopy
import math

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from scripts.single_pair_rl_env import SinglePairEnv

START = pd.Timestamp('2020-01-01T00:00:00Z')
FIELDS = ('open', 'high', 'low', 'close', 'volume')
COSTS = ({'fee': .001, 'slippage': .0002}, {'fee': .002, 'slippage': .0005})


def market(*, flat=False):
    index = pd.date_range('2019-01-01', '2020-06-01', freq='h', inclusive='left', tz='UTC')
    t = np.arange(len(index), dtype=float)
    close = pd.DataFrame({symbol: base * (np.ones(len(t)) if flat else 1 + t * .00002)
                          for symbol, base in [('BTCUSDT', 100.), ('ETHUSDT', 200.)]}, index=index)
    data = {f'hourly_{field}': close.copy() for field in FIELDS}
    data['hourly_high'] *= 1.001; data['hourly_low'] *= .999
    data['hourly_volume'].loc[:, :] = 10.
    data['assets'] = {s: {'symbol': s[:-4], 'binance_symbol': s} for s in close.columns}
    data['registry'] = deepcopy(data['assets'])
    data['manifest'] = {'scope': 'synthetic test only, no market or model'}
    return data


def only_coin(data, symbol):
    return {**{f'hourly_{f}': data[f'hourly_{f}'].loc[:, [symbol]].copy() for f in FIELDS},
            'assets': {symbol: deepcopy(data['assets'][symbol])}, 'registry': {symbol: deepcopy(data['registry'][symbol])}}


def env(data, symbol='BTCUSDT', days=6, **kwargs):
    return SinglePairEnv(data, symbol, START, START + pd.Timedelta(days=days), **kwargs)


@pytest.mark.parametrize('symbol', ['BTCUSDT', 'ETHUSDT'])
def test_true_one_column_market_13_observation_and_one_gross_action(symbol):
    e = env(only_coin(market(), symbol), symbol); obs, _ = e.reset(seed=9)
    assert e.SYMBOLS == (symbol,) and set(e.assets) == {symbol}
    assert e.action_space.shape == (1,) and e.observation_space.shape == obs.shape == (13,)
    assert obs.dtype == np.float32 and len(e.FEATURE_NAMES) == 13
    assert all(name.startswith(symbol + ':') for name in e.FEATURE_NAMES[:10])
    assert e.FEATURE_NAMES[-3:] == ('coin_weight', 'cash_weight', 'current_drawdown')
    assert all(list(frame.columns) == [symbol] for frame in e._frames.values())
    np.testing.assert_array_equal(obs[-3:], [0, 1, 0])
    for action, weight in [(-2, 0.), (-1, 0.), (0, 0.), (.5, .475), (1, .95), (2, .95)]:
        np.testing.assert_allclose(e.action_to_weights([action]), [weight, 1 - weight], atol=1e-15)
    for invalid in ([0, 1], np.nan, [np.inf], [np.nan]):
        with pytest.raises(ValueError): e.action_to_weights(invalid)


def test_two_ten_thousand_accounts_share_no_cash_inventory_or_high_water():
    data = market(); btc, eth = env(data), env(data, 'ETHUSDT')
    btc.reset(seed=1); eth.reset(seed=1)
    assert btc.ledger is not eth.ledger and btc.ledger.cash == eth.ledger.cash == 10000.
    assert btc.ledger.positions is not eth.ledger.positions and btc.ledger.cost_basis is not eth.ledger.cost_basis
    btc.step([1]); before = deepcopy((btc.ledger.cash, btc.ledger.positions, btc.ledger.cost_basis, btc._nav, btc._high_water))
    assert eth.ledger.cash == eth._nav == eth._high_water == 10000. and eth.ledger.positions == {}
    eth.step([.5])
    assert before == (btc.ledger.cash, btc.ledger.positions, btc.ledger.cost_basis, btc._nav, btc._high_water)
    for e, symbol in ((btc, 'BTCUSDT'), (eth, 'ETHUSDT')):
        result = e.get_result(); m = result['metadata']
        assert m['initial_capital'] == m['independent_capital'] == 10000.
        assert m['pair'] == symbol and m['tradable_symbols'] == [symbol] and m['no_ensemble'] is True
        assert set(result['fills'].asset_key) == {symbol} and set(result['weights'].asset_key) == {symbol}


@pytest.mark.parametrize('symbol', ['BTCUSDT', 'ETHUSDT'])
def test_other_coin_shocks_and_gaps_never_change_features_windows_or_account(symbol):
    data = market(); changed = deepcopy(data); other = 'ETHUSDT' if symbol == 'BTCUSDT' else 'BTCUSDT'
    for field in FIELDS:
        changed['hourly_' + field].loc[:, other] = np.nan
        changed['hourly_' + field].loc[START:, other] = 1e12 if field != 'volume' else -10.
    first = env(data, symbol, days=75, training=True, episode_days=32)
    second = env(changed, symbol, days=75, training=True, episode_days=32)
    assert first._eligible_starts == second._eligible_starts
    assert first.training_window_metadata == second.training_window_metadata
    a, _ = first.reset(seed=42); b, _ = second.reset(seed=42); np.testing.assert_array_equal(a, b)
    for action in ([1], [.5], [-1], [.8]):
        ar = first.step(action); br = second.step(action)
        np.testing.assert_array_equal(ar[0], br[0]); assert ar[1:4] == br[1:4]
        np.testing.assert_array_equal(ar[4]['target_weights'], br[4]['target_weights'])
        assert {k: v for k, v in ar[4].items() if k != 'target_weights'} == {k: v for k, v in br[4].items() if k != 'target_weights'}
    for key in ('equity', 'fills', 'weights', 'boundary_equity'):
        assert_frame_equal(first.get_result()[key], second.get_result()[key], check_exact=True)


def one_coin_oracle(data, symbol, actions, costs, *, penalty=.25, band=.025):
    """Closed-form one-coin target accounting, never calls env/ledger helpers."""
    cash, quantity, nav, peak = 10000., 0., 10000., 10000.
    marks, fills, rewards = [], [], []
    fee, slip = costs['fee'], costs['slippage']; buy_cost = (1 + slip) * (1 + fee) - 1; sell_cost = 1 - (1 - slip) * (1 - fee)
    for n, action in enumerate(actions):
        day = START + pd.Timedelta(days=n); weight = .95 * max(min(max(action, -1.), 1.), 0.)
        previous_nav = nav; old_dd = 1 - nav / peak; deepest = old_dd
        before = float(data['hourly_close'].loc[day - pd.Timedelta(hours=1), symbol]); current = quantity * before / nav
        execute = (weight == 0 and current > 0) or abs(weight - current) >= band
        if execute:
            opening = float(data['hourly_open'].loc[day, symbol]); total = cash + quantity * opening
            reduction = max((quantity * opening - weight * total) / (1 - weight * sell_cost), 0.)
            if reduction > 1e-9:
                sold = min(reduction / opening, quantity); price = opening * (1 - slip); charge = sold * price * fee
                cash += sold * price - charge; quantity -= sold
                if quantity <= 1e-14: quantity = 0.
                fills.append((day, 'SELL', sold, price, charge, cash, quantity))
        # First mark occurs before the01UTC purchase.
        for hour in range(24):
            stamp = day + pd.Timedelta(hours=hour)
            if hour == 1 and execute:
                opening = float(data['hourly_open'].loc[stamp, symbol]); total = cash + quantity * opening
                increase = max((weight * total - quantity * opening) / (1 + weight * buy_cost), 0.)
                if increase > 1e-9:
                    bought = increase / opening; price = opening * (1 + slip); charge = bought * price * fee
                    cash -= bought * price + charge; quantity += bought
                    fills.append((stamp, 'BUY', bought, price, charge, cash, quantity))
            price = float(data['hourly_close'].loc[stamp, symbol]); nav = cash + quantity * price; peak = max(peak, nav)
            deepest = max(deepest, 1 - nav / peak)
            marks.append((stamp, nav, cash, quantity * price, (quantity * price) / nav))
        rewards.append(100 * (math.log(nav / previous_nav) - penalty * max(deepest - old_dd, 0.)))
    return marks, fills, rewards, cash, quantity, peak


@pytest.mark.parametrize('symbol', ['BTCUSDT', 'ETHUSDT'])
@pytest.mark.parametrize('costs', COSTS)
def test_real_six_day_flow_matches_independent_closed_form_cash_inventory_and_fees(symbol, costs):
    data = market(); actions = [1., .4, .42, -1., .9, .2]
    for day in range(6):
        stamp = START + pd.Timedelta(days=day, hours=8)
        data['hourly_close'].loc[stamp, symbol] *= .75
        data['hourly_low'].loc[stamp, symbol] = data['hourly_close'].loc[stamp, symbol] * .999
    e = env(data, symbol, costs=costs); e.reset(seed=2); observed_rewards = []
    for a in actions: observed_rewards.append(e.step([a])[1])
    result = e.get_result(); marks, fills, rewards, cash, quantity, peak = one_coin_oracle(data, symbol, actions, costs)
    expected = pd.DataFrame(marks, columns=['timestamp', 'equity', 'cash', 'base_value', 'exposure']).set_index('timestamp')
    assert_frame_equal(result['equity'], expected, check_exact=False, atol=1e-7, rtol=1e-12)
    assert len(result['fills']) == len(fills)
    for actual, expected_fill in zip(result['fills'].itertuples(), fills):
        assert actual.timestamp == expected_fill[0] and actual.side == expected_fill[1] and actual.asset_key == symbol
        np.testing.assert_allclose([actual.qty, actual.price, actual.fee, actual.cash_after, actual.quantity_after], expected_fill[2:], atol=1e-7, rtol=1e-12)
    np.testing.assert_allclose(observed_rewards, rewards, atol=1e-9, rtol=1e-12)
    metadata = result['metadata']; assert metadata['final_cash'] == pytest.approx(cash, abs=1e-7)
    assert metadata['final_positions'].get(symbol, 0.) == pytest.approx(quantity, abs=1e-9)
    assert metadata['hourly_high_water'] == pytest.approx(peak, abs=1e-7)
    assert metadata['fees'] == pytest.approx(sum(x[4] for x in fills), abs=1e-8)
    assert metadata['final_liquidation_equity'] == pytest.approx(cash + quantity * data['hourly_close'].loc[START + pd.Timedelta(days=6, hours=-1), symbol] * (1 - costs['slippage']) * (1 - costs['fee']), abs=1e-7)
    assert (result['equity'].cash >= -1e-8).all() and (result['equity'].base_value >= -1e-8).all()
    assert result['fills'].query("side=='SELL'").timestamp.dt.hour.eq(0).all()
    assert result['fills'].query("side=='BUY'").timestamp.dt.hour.eq(1).all()


def test_hourly_drawdown_retains_intraday_peak_and_daily_reward_penalty():
    data = market(flat=True); symbol = 'BTCUSDT'
    data['hourly_close'].loc[START + pd.Timedelta(hours=2), symbol] = 200.
    data['hourly_high'].loc[START + pd.Timedelta(hours=2), symbol] = 200.
    data['hourly_close'].loc[START + pd.Timedelta(hours=3), symbol] = 50.
    data['hourly_low'].loc[START + pd.Timedelta(hours=3), symbol] = 50.
    e = env(data, days=1); e.reset(); _, reward, terminated, truncated, _ = e.step([1])
    result = e.get_result(); nav = result['equity'].equity; dd = nav / nav.cummax().clip(lower=10000.) - 1
    expected_nav = 10000 / (1 + .95 * ((1.0002 * 1.001) - 1))
    assert nav.iloc[-1] == pytest.approx(expected_nav, abs=1e-7)
    assert dd.min() == pytest.approx(.525 / 1.95 - 1, abs=1e-12)
    assert reward == pytest.approx(100 * (math.log(expected_nav / 10000) - .25 * (1 - .525 / 1.95)), abs=1e-9)
    assert len(nav) == 24 and terminated is True and truncated is False


def test_band_skips_small_changes_but_cash_exit_forces_sale_of_small_holding():
    data = market(flat=True); e = env(data, days=4); e.reset(); e.step([.5]); first = len(e.ledger.fills)
    current = e.current_weights[0]; _, _, _, _, info = e.step_target([current + .0249, 1 - current - .0249])
    assert info['rebalanced'] is False and len(e.ledger.fills) == first
    _, _, _, _, info = e.step_target([current + .0251, 1 - current - .0251])
    assert info['rebalanced'] is True and len(e.ledger.fills) > first
    crash = market(flat=True)
    for field in ('open', 'high', 'low', 'close'): crash['hourly_' + field].loc[START + pd.Timedelta(days=1):, 'BTCUSDT'] = .01
    tiny = env(crash, days=3); tiny.reset(); tiny.step([1]); tiny.step_hold()
    assert 0 < tiny.current_weights[0] < .025
    _, _, _, _, info = tiny.step([0]); assert info['rebalanced'] is True and tiny.ledger.positions == {}
    assert tiny.ledger.fills[-1]['side'] == 'SELL'


@pytest.mark.parametrize('key', ['positions', 'cost_basis'])
def test_foreign_inventory_is_rejected_before_any_economic_change(key):
    e = env(market()); e.reset(); getattr(e.ledger, key)['ETHUSDT'] = 1.
    before = deepcopy((e.ledger.cash, e.ledger.positions, e.ledger.cost_basis, e.ledger.fills))
    with pytest.raises(ValueError, match='foreign'): e.step([1])
    assert before == (e.ledger.cash, e.ledger.positions, e.ledger.cost_basis, e.ledger.fills)


def test_training_ends_as_timeout_and_evaluation_has_no_reset_or_future_feature_leakage():
    data = market(); e = env(data, days=75, training=True, episode_days=4); e.reset(seed=11)
    for i in range(4):
        observation, _, terminated, truncated, _ = e.step([.6])
        assert observation.shape == (13,) and terminated is False and truncated is (i == 3)
    assert e.training_episode_history[-1]['completed_steps'] == 4 and e.training_episode_history[-1]['truncated'] is True
    assert observation[-3] > .5  # Real terminal held-coin state, not a reset all-cash observation.
    cutoff = START + pd.Timedelta(days=5)
    cropped = {**data, **{f'hourly_{f}': data[f'hourly_{f}'].loc[lambda x: x.index < cutoff].copy() for f in FIELDS}}
    a, b = env(data, days=5), env(cropped, days=5); ao, _ = a.reset(seed=6); bo, _ = b.reset(seed=6); np.testing.assert_array_equal(ao, bo)
    for action in ([1], [.3], [0], [.8], [.5]):
        ar, br = a.step(action), b.step(action); np.testing.assert_array_equal(ar[0], br[0]); assert ar[1:4] == br[1:4]
    for key in ('equity', 'fills', 'weights', 'boundary_equity'): assert_frame_equal(a.get_result()[key], b.get_result()[key], check_exact=True)
    with pytest.raises(RuntimeError): a.step([0])


def test_own_missing_23utc_is_not_replaced_by_22utc_and_bad_training_day_is_excluded():
    data = market(); bad_day = START + pd.Timedelta(days=5)
    data['hourly_close'].loc[bad_day + pd.Timedelta(hours=23), 'BTCUSDT'] = np.nan
    with pytest.raises(ValueError, match='complete'):
        env(data, days=10)
    train = env(data, days=75, training=True, episode_days=4)
    assert bad_day not in train._eligible_starts
    assert not train._feature_valid[bad_day + pd.Timedelta(days=1)]
    assert all(not (start <= bad_day < start + pd.Timedelta(days=4)) for start in train._eligible_starts)


def test_invalid_single_targets_never_change_the_ledger():
    e = env(market()); e.reset()
    for target in ([.3, .3, .4], [.9, .3], [-.1, 1.1], [.99, .01], [np.nan, np.nan]):
        with pytest.raises(ValueError): e.step_target(target)
        assert e.ledger.cash == 10000. and e.ledger.positions == {} and not e.ledger.fills


def test_input_market_frames_cannot_mutate_an_existing_private_account():
    data = market(flat=True); tested, reference = env(data), env(deepcopy(data))
    tested.reset(seed=5); reference.reset(seed=5)
    for field in FIELDS: data['hourly_' + field].loc[START:, 'BTCUSDT'] *= 10.
    for action in ([1], [.3]):
        a, b = tested.step(action), reference.step(action)
        np.testing.assert_array_equal(a[0], b[0]); assert a[1:4] == b[1:4]
    for key in ('equity', 'fills', 'weights', 'boundary_equity'):
        assert_frame_equal(tested.get_result()[key], reference.get_result()[key], check_exact=True)


def test_missing_own_hourly_grid_row_is_not_filled_or_silently_removed():
    data = market(); missing = START - pd.Timedelta(hours=2)
    for field in FIELDS: data['hourly_' + field] = data['hourly_' + field].drop(index=missing)
    with pytest.raises(ValueError, match='omitted'): env(data)


def test_insufficient_cash_and_overselling_have_no_economic_side_effect():
    e = env(market(flat=True)); e.reset()
    operations = [lambda: e.ledger.sell('BTCUSDT', 1., 100., START),
                  lambda: e.ledger.buy_with_cash('BTCUSDT', 10001., 100., START),
                  lambda: e.ledger.buy('BTCUSDT', 100., 100., START)]
    for operation in operations:
        before = deepcopy((e.ledger.cash, e.ledger.positions, e.ledger.cost_basis, e.ledger.fills))
        with pytest.raises(ValueError): operation()
        assert before == (e.ledger.cash, e.ledger.positions, e.ledger.cost_basis, e.ledger.fills)
    e.step([1]); before = deepcopy((e.ledger.cash, e.ledger.positions, e.ledger.cost_basis, e.ledger.fills))
    with pytest.raises(ValueError): e.ledger.sell('BTCUSDT', e.ledger.positions['BTCUSDT'] + 1., 100., e.current_date)
    assert before == (e.ledger.cash, e.ledger.positions, e.ledger.cost_basis, e.ledger.fills)

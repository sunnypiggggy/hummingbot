"""Independent hand-calculated timing, economics and causal-state checks."""
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from scripts.btceth_rl_env import BTCETHPortfolioEnv
from scripts.crypto_rl_portfolio_env import utc
from scripts.crypto_rl_reporting import result_fingerprints, save_result, summarize


START = utc("2023-08-01")


def dataset(days=420):
    index = pd.date_range("2023-01-01T00:00Z", periods=days * 24, freq="h")
    frame = pd.DataFrame({"BTCUSDT": 100., "ETHUSDT": 200.}, index=index)
    data = {"hourly_" + field: frame.copy() for field in ("open", "high", "low", "close")}
    data["hourly_volume"] = frame * 0 + 100
    return data


def env(data=None, days=3, **kwargs):
    return BTCETHPortfolioEnv(dataset() if data is None else data, START,
                             START + pd.Timedelta(days=days), **kwargs)


def change_price(data, start, end=None, *, btc=None, eth=None):
    for field in ("open", "high", "low", "close"):
        frame = data["hourly_" + field]
        for column, price in (("BTCUSDT", btc), ("ETHUSDT", eth)):
            if price is not None:
                frame.loc[start:end, column] = price


def test_action_cash_allocation_caps_and_observation_are_reachable():
    e = env()
    obs, _ = e.reset(seed=2)
    assert obs.shape == (27,) and obs.dtype == np.float32
    assert e.observation_space.contains(obs)
    np.testing.assert_array_equal(e.action_to_weights([0, .8]), [0, 0, 1])
    np.testing.assert_array_equal(e.action_to_weights([-1, -.8]), [0, 0, 1])
    assert e.action_to_weights([1, 0]) == pytest.approx([.475, .475, .05])
    assert e.action_to_weights([1, -1]) == pytest.approx([.95, 0, .05])
    assert e.action_to_weights([1, 1]) == pytest.approx([0, .95, .05])
    capped = env(max_asset_weight=.6)
    assert capped.action_to_weights([1, -1]) == pytest.approx([.6, 0, .4])


def test_flat_market_entry_cost_cash_floor_and_reporting_by_hand(tmp_path):
    e = env(days=1)
    e.reset()
    _, reward, done, truncated, info = e.step_target([.475, .475, .05])
    result = e.get_result()
    cost = 1.0002 * 1.001 - 1
    nav = 10000 / (1 + .95 * cost)
    assert result["metadata"]["final_equity"] == pytest.approx(nav)
    assert result["metadata"]["final_cash"] == pytest.approx(.05 * nav)
    assert result["metadata"]["fees"] == pytest.approx(.95 * nav * 1.0002 * .001)
    assert result["metadata"]["slippage_cost"] == pytest.approx(.95 * nav * .0002)
    assert reward == pytest.approx(100 * (np.log(nav / 10000) - .25 * (1 - nav / 10000)))
    assert info["drawdown_increase"] == pytest.approx(1 - nav / 10000)
    assert done and not truncated
    assert len(result["equity"]) == 24
    assert result["fills"].side.tolist() == ["BUY", "BUY"]
    assert set(result["fills"].timestamp) == {START + pd.Timedelta(hours=1)}
    assert save_result(tmp_path, result) == result_fingerprints(result)
    summary, _, _ = summarize(result, run_id="hand", strategy="equal", scenario="base",
                              split="test", start=START, end=START + pd.Timedelta(days=1))
    assert summary["net_return"] == pytest.approx(nav / 10000 - 1)


def test_daily_repeated_target_and_hold_do_not_create_phantom_fills():
    e = env(days=3)
    e.reset()
    e.step_target([.475, .475, .05])
    count = len(e.ledger.fills)
    e.step_target([.475, .475, .05])
    e.step_hold()
    assert len(e.ledger.fills) == count == 2
    assert len(e.get_result()["equity"]) == 72


def test_coin_rotation_sells_then_buys_next_hour_and_charges_both_legs():
    e = env(days=2)
    e.reset()
    e.step_target([.95, 0, .05])
    e.step_target([0, .95, .05])
    fills = e.get_result()["fills"]
    assert fills.side.tolist() == ["BUY", "SELL", "BUY"]
    assert fills.asset_key.tolist() == ["BTCUSDT", "BTCUSDT", "ETHUSDT"]
    assert fills.timestamp.tolist() == [START + pd.Timedelta(hours=1), START + pd.Timedelta(days=1),
                                      START + pd.Timedelta(days=1, hours=1)]
    assert (fills.fee > 0).all()
    assert "BTCUSDT" not in e.ledger.positions
    assert e.ledger.cash >= 0


def test_small_target_changes_skip_but_complete_exit_bypasses_band():
    e = env(days=3, rebalance_band=0)
    e.reset()
    e.step_target([.01, 0, .99])
    count = len(e.ledger.fills)
    e.rebalance_band = .025
    _, _, _, _, info = e.step_target([.011, 0, .989])
    assert not info["rebalanced"] and len(e.ledger.fills) == count
    _, _, _, _, info = e.step_target([0, 0, 1])
    assert info["rebalanced"]
    assert e.ledger.positions == {}
    assert e.ledger.fills[-1]["side"] == "SELL"


def test_cash_and_no_coin_sales_and_bad_actions_cannot_borrow():
    e = env()
    e.reset()
    with pytest.raises(ValueError, match="inventory"):
        e.ledger.sell("BTCUSDT", 1, 100, START)
    with pytest.raises(ValueError, match="Insufficient"):
        e.ledger.buy("BTCUSDT", 1000, 100, START)
    for target in ([1, 0, 0], [-.1, .5, .6], [0, 0, .5], [np.nan, 0, 1]):
        with pytest.raises(ValueError):
            e.step_target(target)
    for action in ([0], [0, 0, 0], [np.nan, 0]):
        with pytest.raises(ValueError):
            e.action_to_weights(action)
    for _ in range(3):
        e.step([-1, 0])
    result = e.get_result()
    assert result["fills"].empty
    np.testing.assert_array_equal(result["equity"].equity, 10000)


def test_next_open_is_execution_only_and_unheld_gap_is_not_profit():
    data = dataset()
    change_price(data, START, btc=200)
    e = env(data, days=1, costs={"fee": 0, "slippage": 0})
    e.reset()
    _, reward, *_ = e.step_target([.95, 0, .05])
    assert e.ledger.fills[0]["price"] == 200
    assert reward == pytest.approx(0)


def test_reward_telescopes_gap_and_terminal_never_peeks_next_open():
    data = dataset()
    jump = START + pd.Timedelta(days=1)
    change_price(data, jump, btc=200)
    e = env(data, days=2, costs={"fee": 0, "slippage": 0}, drawdown_penalty=0)
    e.reset()
    first = e.step_target([.5, 0, .5])[1]
    second = e.step_hold()[1]
    assert first == 0
    assert second == pytest.approx(100 * np.log(1.5))
    assert e.get_result()["metadata"]["final_equity"] == pytest.approx(15000)
    # An opening price outside this evaluation cannot affect terminal reward.
    changed = deepcopy(data)
    change_price(changed, START + pd.Timedelta(days=2), btc=9000)
    repeat = env(changed, days=2, costs={"fee": 0, "slippage": 0}, drawdown_penalty=0)
    repeat.reset()
    repeat.step_target([.5, 0, .5])
    assert repeat.step_hold()[1] == second


def test_hourly_drawdown_penalty_captures_intraday_loss_that_recovers():
    data = dataset()
    change_price(data, START + pd.Timedelta(hours=12), START + pd.Timedelta(hours=12), btc=80)
    e = env(data, days=1, costs={"fee": 0, "slippage": 0}, drawdown_penalty=.25)
    e.reset()
    obs, reward, _, _, info = e.step_target([.5, 0, .5])
    assert info["log_net_return"] == pytest.approx(0)
    assert info["drawdown_increase"] == pytest.approx(.1)
    assert reward == pytest.approx(-2.5)
    assert info["current_drawdown"] == 0
    assert obs[-1] == 0


def test_future_and_current_day_cannot_change_decision_observation():
    data = dataset()
    altered = deepcopy(data)
    for name, frame in altered.items():
        frame.loc[START:] *= 4 if name != "hourly_volume" else 9
    a, b = env(data), env(altered)
    np.testing.assert_array_equal(a.reset()[0], b.reset()[0])
    assert a.action_to_weights([.8, .3]).tolist() == b.action_to_weights([.8, .3]).tolist()


def test_future_truncation_preserves_all_historical_observations_and_fills():
    data = dataset()
    cut = START + pd.Timedelta(days=4)
    prefix = {name: frame.loc[:cut - pd.Timedelta(hours=1)].copy() for name, frame in data.items()}
    a, b = env(data), env(prefix)
    np.testing.assert_array_equal(a.reset()[0], b.reset()[0])
    for action in ([.8, -.2], [.5, .7], [-.1, 0]):
        ra, rb = a.step(action), b.step(action)
        np.testing.assert_array_equal(ra[0], rb[0])
        assert ra[1:4] == rb[1:4]
    assert result_fingerprints(a.get_result()) == result_fingerprints(b.get_result())


def test_missing_features_or_market_fail_evaluation_and_filter_train_episodes():
    data = dataset()
    missing = START + pd.Timedelta(days=50, hours=3)
    data["hourly_close"].loc[missing, "ETHUSDT"] = np.nan
    with pytest.raises(ValueError, match="Evaluation must be complete"):
        env(data, days=90)
    e = env(data, days=140, training=True, episode_days=16)
    quality = e.training_window_metadata
    assert quality["excluded_windows"] > 0
    assert quality["eligible_windows"] > 0
    for first in e._eligible_starts:
        assert not first <= missing.normalize() < first + pd.Timedelta(days=16)
    with pytest.raises(ValueError, match="Evaluation must be complete"):
        BTCETHPortfolioEnv(dataset(), "2023-02-01", "2023-02-03")


def test_time_alignment_duplicate_and_omitted_hours_are_rejected():
    for fault in ("alignment", "duplicates", "omitted"):
        data = dataset()
        if fault == "alignment":
            data["hourly_volume"].index += pd.Timedelta(hours=1)
        elif fault == "duplicates":
            frame = data["hourly_open"]
            data["hourly_open"] = pd.concat([frame.iloc[:1], frame])
        else:
            data = {name: frame.drop(frame.index[30]) for name, frame in data.items()}
        with pytest.raises(ValueError):
            env(data)


def test_intraday_gap_does_not_erase_actual_23utc_close_or_long_trend_after_volume_recovers():
    data = dataset(days=600)
    gap = START + pd.Timedelta(hours=13)
    for frame in data.values():
        frame.loc[gap, "BTCUSDT"] = np.nan
    later = START + pd.Timedelta(days=90)
    e = BTCETHPortfolioEnv(data, later, later + pd.Timedelta(days=2))
    e.reset()
    assert e._feature_valid[later]
    assert not e._market_valid[START]
    # The true close exists at 23UTC; long close-only features are unaffected.
    assert np.isfinite(e._features[START + pd.Timedelta(days=1)][:9]).all()
    # Volume is not filled, so its trailing 30-day feature stays unavailable.
    assert np.isnan(e._features[START + pd.Timedelta(days=30)][9])
    assert np.isfinite(e._features[START + pd.Timedelta(days=31)][9])


def test_missing_23utc_close_cannot_fall_back_to_22utc_and_requires_200_true_closes():
    data = dataset(days=650)
    gap = START + pd.Timedelta(hours=23)
    data["hourly_close"].loc[gap, "BTCUSDT"] = np.nan
    later = START + pd.Timedelta(days=202)
    e = BTCETHPortfolioEnv(data, later, later + pd.Timedelta(days=2))
    # Every earlier intraday price remains available, but it is not a daily close.
    assert np.isnan(e._features[START + pd.Timedelta(days=1)][7])
    assert not e._feature_valid[START + pd.Timedelta(days=200)]
    assert e._feature_valid[START + pd.Timedelta(days=201)]
    assert e._feature_valid[later]


def test_volume_feature_requires_24_real_volume_hours_independently_of_ohlc():
    data = dataset(days=500)
    gap = START + pd.Timedelta(hours=13)
    data["hourly_open"].loc[gap, "BTCUSDT"] = np.nan
    later = START + pd.Timedelta(days=2)
    e = BTCETHPortfolioEnv(data, later, later + pd.Timedelta(days=2))
    # Missing open does not corrupt actual volume or any observed 23UTC close.
    assert e._feature_valid[START + pd.Timedelta(days=1)]
    assert np.isfinite(e._features[START + pd.Timedelta(days=1)][9])
    missing_volume = deepcopy(data)
    missing_volume["hourly_volume"].loc[gap, "BTCUSDT"] = np.nan
    after_recovery = START + pd.Timedelta(days=31)
    other = BTCETHPortfolioEnv(missing_volume, after_recovery, after_recovery + pd.Timedelta(days=2))
    assert np.isnan(other._features[START + pd.Timedelta(days=1)][9])
    assert not other._feature_valid[START + pd.Timedelta(days=30)]
    assert other._feature_valid[after_recovery]


def test_seeded_train_reset_is_reproducible_and_real_episode_end_is_truncated():
    e = env(days=160, training=True, episode_days=16, record_history=False)
    first, info = e.reset(seed=19)
    date = e.current_date
    second, other = e.reset(seed=19)
    np.testing.assert_array_equal(first, second)
    assert info == other and date == e.current_date
    done = False
    for _ in range(16):
        _, _, terminated, truncated, _ = e.step([.2, 0])
        done = terminated or truncated
    assert done
    assert truncated and not terminated
    assert e.current_date == date + pd.Timedelta(days=16)
    assert e.training_episode_history[-1]["completed_steps"] == 16
    assert e.get_result()["equity"].empty
    with pytest.raises(RuntimeError, match="Episode ended"):
        e.step([0, 0])


def test_history_off_preserves_cash_cost_reward_and_hourly_high_water():
    data = dataset()
    change_price(data, START + pd.Timedelta(days=1, hours=4),
                 START + pd.Timedelta(days=1, hours=9), btc=110, eth=180)
    a, b = env(data), env(data, record_history=False)
    a.reset()
    b.reset()
    for action in ([.9, 0], [.7, .2], [0, 0]):
        ra, rb = a.step(action), b.step(action)
        np.testing.assert_array_equal(ra[0], rb[0])
        assert ra[1] == rb[1]
    ma, mb = a.get_result()["metadata"], b.get_result()["metadata"]
    for name in ("final_equity", "final_cash", "fees", "total_notional", "slippage_cost",
                 "fill_count", "hourly_high_water", "current_drawdown", "realized_pnl", "unrealized_pnl"):
        assert ma[name] == mb[name]


def test_train_fragment_at_dataset_right_edge_is_time_truncated_without_future_open():
    data = dataset()
    end = START + pd.Timedelta(days=2)
    prefix = {name: frame.loc[:end - pd.Timedelta(hours=1)].copy() for name, frame in data.items()}
    e = env(prefix, days=2, training=True, episode_days=2, record_history=False)
    e.reset(seed=1)
    e.step([.7, 0])
    _, _, terminated, truncated, info = e.step([.7, 0])
    assert truncated and not terminated
    assert info["date"] == end
    assert e.episode_end == e.end


@pytest.mark.parametrize("fee,slippage", [(.001, .0002), (.002, .0005)])
def test_round_trip_net_cash_matches_hand_calculation_under_both_cost_scenarios(fee, slippage):
    e = env(days=2, costs={"fee": fee, "slippage": slippage})
    e.reset()
    e.step_target([.95, 0, .05])
    buy_cost = (1 + slippage) * (1 + fee) - 1
    entered_nav = 10000 / (1 + .95 * buy_cost)
    coin_value = entered_nav * .95
    expected_cash = entered_nav * .05 + coin_value * (1 - slippage) * (1 - fee)
    e.step_target([0, 0, 1])
    result = e.get_result()
    assert result["metadata"]["final_equity"] == pytest.approx(expected_cash)
    assert result["metadata"]["final_cash"] == pytest.approx(expected_cash)
    assert result["metadata"]["final_positions"] == {}


def test_gymnasium_contract_is_accepted_by_installed_sb3():
    checker = pytest.importorskip("stable_baselines3.common.env_checker")
    checker.check_env(env(), warn=True)

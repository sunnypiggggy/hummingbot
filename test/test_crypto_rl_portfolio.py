"""Hand-calculated cash/coin and timing tests; no account/network/training."""
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from scripts.crypto_rl_portfolio_env import CryptoPortfolioEnv, PortfolioLedger, GYM_AVAILABLE, utc


def dataset(days=240):
    hours = pd.date_range("2024-01-01T00:00Z", periods=24 * days, freq="h")
    keys = [str(i) for i in range(11)]
    prices = pd.DataFrame({k: 100. + int(k) * 10 for k in keys}, index=hours)
    volume = pd.DataFrame(100., index=hours, columns=keys)
    rows = []
    for day in hours[::24]:
        for slot in range(10):
            rows.append({"date": day, "slot": slot, "asset_key": str(slot), "cap": float(11 - slot),
                         "cmc_rank": slot + 1, "symbol": f"C{slot}", "snapshot_date": day - pd.Timedelta(days=3),
                         "available_after": day - pd.Timedelta(days=1), "anchor": day})
    return {"hourly_open": prices.copy(), "hourly_high": prices.copy(), "hourly_low": prices.copy(),
            "hourly_close": prices.copy(), "hourly_volume": volume,
            "universe": pd.DataFrame(rows), "assets": {k: {"symbol": "C" + k} for k in keys}}


START = utc("2024-03-01")


def env(data=None, days=3, **kwargs):
    return CryptoPortfolioEnv(data or dataset(), START, START + pd.Timedelta(days=days), **kwargs)


def target(*pairs):
    out = np.zeros(11)
    for slot, value in pairs:
        out[slot] = value
    out[10] = 1 - out[:10].sum()
    return out


def test_ledger_entry_and_exit_fees_slippage_are_cash_conserving_by_hand():
    ledger = PortfolioLedger(1000, fee=.01, slippage=.02)
    entry = ledger.buy_with_cash("a", 900, 100, START)
    qty = 900 / (102 * 1.01)
    assert entry["qty"] == pytest.approx(qty)
    assert ledger.cash == pytest.approx(100)
    assert ledger.value({"a": 100}) == pytest.approx(100 + 100 * qty)
    sale = ledger.sell("a", qty, 110, START + pd.Timedelta(days=1))
    assert sale["price"] == pytest.approx(107.8)
    assert sale["fee"] == pytest.approx(qty * 107.8 * .01)
    assert ledger.cash == pytest.approx(100 + qty * 107.8 * .99)
    assert ledger.positions == {}
    assert ledger.realized_pnl == pytest.approx(ledger.cash - 1000)


def test_partial_add_and_sell_cost_basis_reconciles_without_borrowing():
    ledger = PortfolioLedger(1000, fee=.001, slippage=0)
    ledger.buy("a", 2, 100, START)
    ledger.buy("a", 1, 120, START)
    average = 320.32 / 3
    ledger.sell("a", 1, 140, START)
    assert ledger.cost_basis["a"] == pytest.approx(average * 2)
    assert ledger.realized_pnl == pytest.approx(139.86 - average)
    final = ledger.value({"a": 150})
    assert final - 1000 == pytest.approx(ledger.realized_pnl + 300 - ledger.cost_basis["a"])
    with pytest.raises(ValueError, match="inventory"):
        ledger.sell("a", 3, 100, START)
    with pytest.raises(ValueError, match="Insufficient"):
        ledger.buy("a", 100, 100, START)
    assert ledger.cash >= 0


def test_missing_held_mark_fails_instead_of_zero_or_stale_price():
    ledger = PortfolioLedger()
    ledger.buy("a", 1, 100, START)
    for prices in ({}, {"a": np.nan}, {"a": 0.}):
        with pytest.raises(ValueError, match="mark"):
            ledger.value(prices)


@pytest.mark.parametrize("price,quantity", [(1e-5, 486349523.4729573), (1e-8, 486349523472.95734)])
def test_low_price_large_inventory_full_reduction_is_bounded_before_strict_ledger(price, quantity):
    data = dataset()
    for field in ("open", "high", "low", "close"):
        data["hourly_" + field]["0"] = price
    e = env(data, days=1)
    e.reset()
    e.ledger.buy("0", quantity, price, START)
    # A caller explicitly requesting even one ULP above owned units is denied.
    with pytest.raises(ValueError, match="inventory"):
        e.ledger.sell("0", np.nextafter(quantity, np.inf), price, START)
    e.step_target(target())
    assert not e.ledger.positions
    sale = e.ledger.fills[-1]
    assert sale["qty"] == quantity
    assert sale["side"] == "SELL" and sale["quantity_after"] == 0
    assert e.ledger.cash >= 0


def test_score_mapping_cash_floor_asset_cap_and_full_cash_are_reachable():
    e = env()
    obs, _ = e.reset(seed=2)
    assert obs.shape == (92,) and obs.dtype == np.float32
    assert e.observation_space.contains(obs)
    assert e.action_to_weights(np.full(11, -1)).tolist() == target().tolist()
    concentrated = np.full(11, -1.)
    concentrated[0] = 1
    assert e.action_to_weights(concentrated).tolist() == target((0, .25)).tolist()
    equal = e.action_to_weights(np.r_[np.ones(10), -1.])
    assert equal[:10] == pytest.approx(np.full(10, .095))
    assert equal[-1] == pytest.approx(.05)


def test_equal_targets_solve_post_fee_equity_and_reserve_cash_exactly():
    e = env(days=1)
    e.reset()
    _, reward, done, truncated, _ = e.step_target(np.r_[np.full(10, .095), .05])
    result = e.get_result()
    cost = (1 + .0002) * (1 + .001) - 1
    expected_equity = 10000 / (1 + cost * .95)
    assert result["metadata"]["final_equity"] == pytest.approx(expected_equity)
    assert result["metadata"]["final_cash"] == pytest.approx(.05 * expected_equity)
    assert all(h["weight"] <= .25 + 1e-12 for h in result["metadata"]["final_holdings"])
    assert reward == pytest.approx(100 * np.log(expected_equity / 10000))
    assert done and not truncated
    assert len(result["equity"]) == 24
    assert (result["fills"].timestamp.dt.hour == 1).all()
    assert result["fills"].side.tolist() == ["BUY"] * 10


def test_repeated_identical_flat_price_target_does_not_make_phantom_turnover():
    e = env(days=2)
    e.reset()
    e.step_target(target((0, .25)))
    n = len(e.ledger.fills)
    e.step_target(target((0, .25)))
    assert len(e.ledger.fills) == n


def test_same_rank_changes_identity_with_real_sell_then_next_hour_buy():
    data = dataset()
    swap_day = START + pd.Timedelta(days=1)
    u = data["universe"]
    u.loc[(u.date >= swap_day) & (u.slot == 0), "asset_key"] = "10"
    u.loc[(u.date >= swap_day) & (u.slot == 0), "symbol"] = "C10"
    e = env(data, days=2)
    e.reset()
    obs, *_ = e.step_target(target((0, .25)))
    assert obs[-1] == pytest.approx(.25), "old asset is visible as outside-universe exposure"
    e.step_target(target((0, .25)))
    fills = e.get_result()["fills"]
    swap = fills.loc[fills.timestamp >= swap_day]
    assert swap.side.tolist() == ["SELL", "BUY"]
    assert swap.asset_key.tolist() == ["0", "10"]
    assert swap.timestamp.tolist() == [swap_day, swap_day + pd.Timedelta(hours=1)]
    assert swap.fee.gt(0).all()
    assert "0" not in e.ledger.positions and e.ledger.positions["10"] > 0


def test_feature_observation_cannot_see_same_day_or_later_prices():
    data = dataset()
    altered = deepcopy(data)
    for field in ("open", "high", "low", "close"):
        altered["hourly_" + field].loc[START:, :] *= 4
    altered["hourly_volume"].loc[START:, :] *= 99
    a, b = env(data), env(altered)
    np.testing.assert_array_equal(a.reset()[0], b.reset()[0])


def test_truncating_future_preserves_earlier_observations_and_fills():
    data = dataset()
    cut = START + pd.Timedelta(days=4)
    prefix = deepcopy(data)
    for field in ("open", "high", "low", "close", "volume"):
        prefix["hourly_" + field] = prefix["hourly_" + field].loc[:cut - pd.Timedelta(hours=1)]
    prefix["universe"] = prefix["universe"].loc[prefix["universe"].date < cut]
    a, b = env(data, days=3), env(prefix, days=3)
    np.testing.assert_array_equal(a.reset()[0], b.reset()[0])
    for _ in range(3):
        ra, rb = a.step_target(target((0, .25))), b.step_target(target((0, .25)))
        np.testing.assert_array_equal(ra[0], rb[0])
        assert ra[1] == rb[1]
    pd.testing.assert_frame_equal(a.get_result()["fills"], b.get_result()["fills"])


def test_prelisting_nan_is_masked_cash_not_fabricated_history():
    data = dataset()
    for field in ("open", "high", "low", "close", "volume"):
        data["hourly_" + field].loc[:START - pd.Timedelta(days=5), "0"] = np.nan
    e = env(data)
    obs, _ = e.reset()
    assert not e.current_valid_mask[0]
    assert obs[8] == 0
    assert e.action_to_weights(np.r_[1., np.full(10, -1.)])[0] == 0
    e.step_target(target((0, .25)))
    assert not e.ledger.fills


def test_any_missing_hourly_held_price_fails_even_with_day_close_present():
    data = dataset()
    at = START + pd.Timedelta(hours=3)
    data["hourly_close"].loc[at, "0"] = np.nan
    e = env(data, days=1)
    e.reset()
    with pytest.raises(ValueError, match="mark"):
        e.step_target(target((0, .25)))


def test_next_open_price_is_execution_only_not_previous_candle_return():
    data = dataset()
    for field in ("open", "high", "low", "close"):
        data["hourly_" + field].loc[START:, "0"] = 200
    e = env(data, days=1, costs={"fee": 0., "slippage": 0.})
    e.reset()
    _, reward, *_ = e.step_target(target((0, .25)))
    assert e.ledger.fills[0]["price"] == 200
    assert reward == pytest.approx(0), "unheld pre-entry price rise cannot create a profit"


def test_daily_rewards_telescope_across_open_gaps_without_double_count():
    data = dataset()
    jump = START + pd.Timedelta(days=1)
    for field in ("open", "high", "low", "close"):
        data["hourly_" + field].loc[jump:, "0"] *= 2
    e = env(data, days=2, costs={"fee": 0., "slippage": 0.})
    e.reset()
    a = e.step_target(target((0, .25)))
    b = e.step_target(target())
    result = e.get_result()
    assert result["metadata"]["final_equity"] == pytest.approx(12500)
    assert a[1] + b[1] == pytest.approx(100 * np.log(1.25))
    assert b[1] == pytest.approx(0)


def test_cash_policy_is_exactly_flat_with_zero_fees_turnover():
    e = env()
    e.reset()
    for _ in range(3):
        e.step_target(target())
    result = e.get_result()
    assert result["equity"].equity.eq(10000).all()
    assert result["metadata"]["fees"] == result["metadata"]["total_notional"] == 0
    assert result["fills"].empty


def test_training_random_windows_seed_reproducible_and_strictly_inside_training():
    data = dataset(days=400)
    e = CryptoPortfolioEnv(data, START, START + pd.Timedelta(days=250), training=True, episode_days=128)
    e.reset(seed=9)
    first = e.episode_start
    assert e.episode_start >= e.start and e.episode_end <= e.end
    assert (e.episode_end - e.episode_start).days == 128
    e.reset(seed=9)
    assert e.episode_start == first
    for _ in range(128):
        last = e.step_target(target())
    assert last[3] and not last[2]


def test_full_evaluation_never_randomizes_or_resets_inside_window():
    e = env(days=3, training=False, episode_days=1)
    e.reset(seed=99)
    assert e.episode_start == START and (e.episode_end - START).days == 3
    for _ in range(3):
        last = e.step_hold()
    assert last[2] and not last[3]
    assert len(e.get_result()["equity"]) == 72
    with pytest.raises(RuntimeError, match="Episode ended"):
        e.step_hold()
    with pytest.raises(RuntimeError, match="internal"):
        e.rebalance_target(target())


def test_unavailable_snapshot_rejected_before_any_trade():
    data = dataset()
    data["universe"].loc[data["universe"].date == START, "available_after"] = START + pd.Timedelta(days=1)
    with pytest.raises(ValueError, match="unavailable"):
        env(data)


def test_train_candidates_exclude_closed_market_day_without_filling_prices():
    data = dataset(days=400)
    closed_day = START + pd.Timedelta(days=100)
    closed_hour = closed_day + pd.Timedelta(hours=13)
    for field in ("open", "high", "low", "close", "volume"):
        data["hourly_" + field].loc[closed_hour, "0"] = np.nan
    e = CryptoPortfolioEnv(data, START, START + pd.Timedelta(days=250),
                           training=True, episode_days=128)
    assert e.training_window_metadata["excluded_windows"] > 0
    assert e.training_window_metadata["covered_training_days"] < 250
    assert len(e.training_window_metadata["incomplete_days"]) == 1
    assert not e._feature_valid[closed_day + pd.Timedelta(days=1)][0]
    for start in e._eligible_starts:
        assert not start <= closed_day < start + pd.Timedelta(days=128)
    for _ in range(10):
        e.reset()
        assert not e.episode_start <= closed_day < e.episode_end
    assert np.isnan(data["hourly_open"].loc[closed_hour, "0"])


def test_no_complete_train_episode_is_explicit_failure():
    data = dataset()
    for field in ("open", "close"):
        data["hourly_" + field].loc[START + pd.Timedelta(hours=5), "0"] = np.nan
    with pytest.raises(ValueError, match="No complete training"):
        CryptoPortfolioEnv(data, START, START + pd.Timedelta(days=10),
                           training=True, episode_days=128)


def delisting_dataset(notice):
    data = dataset()
    until = START + pd.Timedelta(days=1, hours=3)
    data["assets"]["0"].update(effective_until=until, delisting_notice_at=notice)
    for field in ("open", "high", "low", "close", "volume"):
        data["hourly_" + field].loc[until:, "0"] = np.nan
    return data


def test_announced_delisting_protects_at_midnight_and_records_actual_cost():
    data = delisting_dataset(START - pd.Timedelta(days=3))
    e = env(data, days=2)
    e.reset()
    e.step_target(target((0, .25)))
    assert not e.current_valid_mask[0]
    e.step_target(target((0, .25)))
    fills = e.get_result()["fills"]
    assert fills.side.tolist() == ["BUY", "SELL"]
    assert fills.reason.tolist()[-1] == "known_delisting_protection"
    assert fills.timestamp.tolist()[-1] == START + pd.Timedelta(days=1)
    assert fills.fee.gt(0).all()
    assert not e.ledger.positions


def test_unannounced_delisting_does_not_get_hindsight_exit():
    # Notice published at 02:00 cannot inform the frozen daily 00:00 decision.
    data = delisting_dataset(START + pd.Timedelta(days=1, hours=2))
    e = env(data, days=2)
    e.reset()
    e.step_target(target((0, .25)))
    assert e.current_valid_mask[0]
    with pytest.raises(ValueError, match="mark"):
        e.step_target(target((0, .25)))
    assert not any(f["reason"] == "known_delisting_protection" for f in e.ledger.fills)


def test_terminal_observation_does_not_read_next_split_universe():
    data = dataset()
    end = START + pd.Timedelta(days=1)
    data["universe"].loc[(data["universe"].date >= end) & (data["universe"].slot == 0), "asset_key"] = "10"
    e = env(data, days=1)
    e.reset()
    e.step_target(target((0, .25)))
    assert e.current_assets[0] == "0"


def test_synthetic_short_ppo_can_collect_rollouts_and_update():
    if not GYM_AVAILABLE:
        pytest.skip("Gymnasium optional in original trading environment")
    sb3 = pytest.importorskip("stable_baselines3")
    torch = pytest.importorskip("torch")
    torch.set_num_threads(1)
    e = env(training=True, episode_days=2)
    model = sb3.PPO("MlpPolicy", e, n_steps=16, batch_size=16, n_epochs=1,
                    seed=0, policy_kwargs={"net_arch": [8, 8]}, device="cpu", verbose=0)
    model.learn(total_timesteps=32)
    assert model.num_timesteps == 32
    observation, _ = e.reset(seed=0)
    action, _ = model.predict(observation, deterministic=True)
    assert action.shape == (11,) and np.isfinite(action).all()


@pytest.mark.parametrize("bad", [np.full(11, np.nan), np.zeros(10), np.r_[.5, np.zeros(9), .5], np.r_[np.full(10, .1), 0.]])
def test_invalid_actions_or_targets_never_reach_order_ledger(bad):
    e = env()
    e.reset()
    with pytest.raises(ValueError):
        e.step_target(bad)
    assert not e.ledger.fills


@pytest.mark.skipif(not GYM_AVAILABLE, reason="Gymnasium intentionally optional in original environment")
def test_gymnasium_and_sb3_environment_interfaces():
    from gymnasium.utils.env_checker import check_env
    e = env()
    check_env(e, skip_render_check=True)
    try:
        from stable_baselines3.common.env_checker import check_env as sb3_check
    except ImportError:
        return
    sb3_check(e)

"""Independent, synthetic checks for the isolated cash-only spot study.

No network access, production data, exchange account, or trading connector is
needed. Prices and fees below are deliberately small enough to audit by hand.
"""

import hashlib
import json
import zipfile

import numpy as np
import pandas as pd
import pytest

from scripts import spot_strategy_suite_engine as engine
from scripts import spot_strategy_suite_data as market_data
from scripts.spot_strategy_suite_data import validate_minutes
from scripts.spot_strategy_suite_reporting import combine_sleeves, result_fingerprints, summarize


def candles(prices, *, start="2026-01-01T00:00:00Z", freq="min"):
    values = np.asarray(prices, dtype=float)
    index = pd.date_range(start, periods=len(values), freq=freq)
    return pd.DataFrame(
        {"open": values, "high": values, "low": values,
         "close": values, "volume": np.ones(len(values))},
        index=index,
    )


def signal_rows(frame, entries=(), exits=()):
    offsets = sorted(set(entries) | set(exits))
    return pd.DataFrame(
        {"enter": [i in entries for i in offsets], "exit": [i in exits for i in offsets]},
        index=frame.index[offsets],
    )


def single(frame, signals, *, strategy="rsi2", **kwargs):
    return engine.simulate_single(
        frame, signals, symbol="BTCUSDT", strategy=strategy,
        start=frame.index[0], end=frame.index[-1] + pd.Timedelta(minutes=1),
        capital=1000, **kwargs,
    )


def test_complete_aggregation_uses_closed_boundaries_and_real_ohlcv():
    frame = candles(np.arange(100.0, 111.0))
    frame.loc[frame.index[2], "high"] = 130
    frame.loc[frame.index[3], "low"] = 90
    frame["volume"] = np.arange(1.0, 12.0)
    grouped = engine.aggregate_candles(frame, 5)
    assert grouped.index.tolist() == [frame.index[5], frame.index[10]]
    assert grouped.iloc[0].to_dict() == {
        "open": 100, "high": 130, "low": 90, "close": 104, "volume": 15,
    }
    assert grouped.iloc[1]["close"] == 109


def test_aggregation_drops_incomplete_edges_and_missing_minute_group():
    frame = candles(np.arange(100.0, 121.0))
    frame = frame.iloc[1:].drop(frame.index[12])
    grouped = engine.aggregate_candles(frame, 5)
    assert grouped.index.tolist() == [pd.Timestamp("2026-01-01T00:10Z"),
                                      pd.Timestamp("2026-01-01T00:20Z")]


def test_aggregation_prefix_does_not_change_when_future_prices_arrive():
    frame = candles(np.arange(100.0, 201.0))
    prefix = engine.aggregate_candles(frame.iloc[:73], 5)
    future = frame.copy()
    future.loc[future.index[73:], ["open", "high", "low", "close"]] = 10000
    full = engine.aggregate_candles(future, 5)
    pd.testing.assert_frame_equal(prefix, full.loc[prefix.index])


def test_validate_minutes_accepts_exact_half_open_utc_window():
    frame = candles([100, 101, 102])
    result = validate_minutes(frame, frame.index[0], frame.index[-1] + pd.Timedelta(minutes=1))
    assert result["validated"] is True
    assert result["rows"] == 3
    assert result["missing_minutes"] == result["duplicate_minutes"] == 0


@pytest.mark.parametrize("problem", ["missing", "duplicate", "unsorted", "naive", "extra", "nan",
                                     "negative_volume", "bad_high", "bad_low", "nonpositive"])
def test_validate_minutes_rejects_untrustworthy_data(problem):
    frame = candles([100, 101, 102])
    start, end = frame.index[0], frame.index[-1] + pd.Timedelta(minutes=1)
    if problem == "missing":
        frame = frame.drop(frame.index[1])
    elif problem == "duplicate":
        frame = pd.concat([frame, frame.iloc[-1:]])
    elif problem == "unsorted":
        frame = frame.iloc[::-1]
    elif problem == "naive":
        frame.index = frame.index.tz_localize(None)
    elif problem == "extra":
        frame = pd.concat([frame, candles([103], start=end)])
    elif problem == "nan":
        frame.iloc[1, frame.columns.get_loc("close")] = np.nan
    elif problem == "negative_volume":
        frame.iloc[1, frame.columns.get_loc("volume")] = -1
    elif problem == "bad_high":
        frame.iloc[1, frame.columns.get_loc("high")] = 100
    elif problem == "bad_low":
        frame.iloc[1, frame.columns.get_loc("low")] = 102
    else:
        frame.iloc[1, :4] = 0
    with pytest.raises(ValueError):
        validate_minutes(frame, start, end)


def test_donchian_excludes_current_high_and_requires_sma200_warmup():
    frame = candles(np.arange(100.0, 350.0), freq="h")
    frame["high"] += 0.5
    frame["low"] -= 0.5
    signals = engine.generate_signals(frame, "donchian")
    assert not signals.enter.iloc[:199].any()
    assert bool(signals.enter.iloc[199])
    assert bool(signals.enter.iloc[-1])  # Impossible if current high belongs to the channel.
    frame.loc[frame.index[-1], ["open", "high", "low", "close"]] = [50, 51, 49, 50]
    assert bool(engine.generate_signals(frame, "donchian").exit.iloc[-1])


def test_cluc_deep_dip_entry_middle_exit_and_prior_volume_guard():
    frame = candles([100.0] * 80 + [80.0, 101.0], freq="5min")
    signals = engine.generate_signals(frame, "cluc")
    assert bool(signals.enter.iloc[-2])
    assert bool(signals.exit.iloc[-1])
    frame.loc[frame.index[-2], "volume"] = 21
    assert not bool(engine.generate_signals(frame, "cluc").enter.iloc[-2])


def test_rsi2_is_trend_pullback_and_exits_above_sma5():
    prices = np.r_[np.arange(100.0, 340.0), 334, 329, 324, 319, 314]
    frame = candles(prices, freq="h")
    signals = engine.generate_signals(frame, "rsi2")
    assert bool(signals.enter.iloc[-1])
    assert not bool(signals.exit.iloc[-1])
    below_trend = frame.copy()
    below_trend.loc[below_trend.index[-1], ["open", "high", "low", "close"]] = 50
    assert not bool(engine.generate_signals(below_trend, "rsi2").enter.iloc[-1])
    rebound = pd.concat([frame, candles([350], start=frame.index[-1] + pd.Timedelta(hours=1), freq="h")])
    assert bool(engine.generate_signals(rebound, "rsi2").exit.iloc[-1])


def test_macd_ema_requires_both_state_conditions_not_a_cross(monkeypatch):
    frame = candles([110, 110, 90, 90], freq="h")
    monkeypatch.setattr(engine.talib, "EMA", lambda values, **kw: np.full(len(values), 100.0))
    monkeypatch.setattr(engine.talib, "MACD", lambda values, **kw: (
        np.array([2.0, 0.0, 2.0, 0.0]), np.ones(4), np.zeros(4),
    ))
    signals = engine.generate_signals(frame, "macd_ema")
    assert signals.enter.tolist() == [True, False, False, False]
    assert signals.exit.tolist() == [False, False, False, True]


@pytest.mark.parametrize("strategy", engine.STRATEGY_IDS)
def test_signal_prefix_invariance_under_extreme_future_candles(strategy):
    values = 150 + np.arange(360) * 0.1 + 4 * np.sin(np.arange(360) / 7)
    frame = candles(values, freq="h")
    prefix = engine.generate_signals(frame.iloc[:270], strategy)
    frame.loc[frame.index[270:], ["open", "high", "low", "close"]] = 5000
    full = engine.generate_signals(frame, strategy)
    pd.testing.assert_frame_equal(prefix, full.loc[prefix.index])


def test_cash_fees_slippage_and_return_match_hand_calculation_without_adding():
    frame = candles([100.0] * 6)
    result = single(frame, signal_rows(frame, entries=(0, 1, 2), exits=(4,)), fee=.001, slippage=.0002)
    fills, trade, meta = result["fills"], result["trades"].iloc[0], result["metadata"]
    qty = 950 / (100.02 * 1.001)
    buy_fee, sell_fee = qty * 100.02 * .001, qty * 99.98 * .001
    final_cash = 50 + qty * 99.98 - sell_fee
    assert fills.side.tolist() == ["BUY", "SELL"]
    assert fills.qty.tolist() == pytest.approx([qty, qty])
    assert fills.price.tolist() == pytest.approx([100.02, 99.98])
    assert fills.fee.tolist() == pytest.approx([buy_fee, sell_fee])
    assert meta["final_cash"] == pytest.approx(final_cash)
    assert meta["fees"] == pytest.approx(buy_fee + sell_fee)
    assert meta["slippage_cost"] == pytest.approx(qty * .04)
    assert meta["total_notional"] == pytest.approx(qty * (100.02 + 99.98))
    assert trade.pnl == pytest.approx(final_cash - 1000)
    assert trade.return_pct == pytest.approx((final_cash - 1000) / 950 * 100)
    assert trade.holding_minutes == 4
    assert meta["final_positions"] == {}
    assert (result["equity"].cash >= 0).all()


def test_entry_at_close_boundary_uses_following_minute_open_never_signal_close():
    frame = candles([100.0] * 12)
    frame.loc[frame.index[5], ["open", "high", "low", "close"]] = 120
    signals = pd.DataFrame({"enter": [True], "exit": [False]}, index=engine.aggregate_candles(frame, 5).index[:1])
    result = single(frame, signals, fee=0, slippage=0)
    assert result["fills"].timestamp.tolist() == [frame.index[5]]
    assert result["fills"].price.tolist() == [120]
    assert result["equity"].base_value.iloc[:5].eq(0).all()


def test_missing_or_nan_signal_is_not_forward_filled_or_treated_as_truthy():
    frame = candles([100.0] * 6)
    signals = pd.DataFrame({"enter": [np.nan, True], "exit": [False, False]},
                           index=frame.index[[0, 3]])
    # Retain actual boolean scalars instead of allowing pandas to coerce to float.
    signals["enter"] = pd.Series([np.nan, True], index=signals.index, dtype=object)
    result = single(frame, signals, fee=0, slippage=0)
    assert result["fills"].timestamp.tolist() == [frame.index[3]]


def test_terminal_position_is_marked_to_market_without_fictitious_sell():
    frame = candles([100, 105, 110])
    result = single(frame, signal_rows(frame, entries=(0,)), fee=.001, slippage=.0002)
    qty = 950 / (100.02 * 1.001)
    meta = result["metadata"]
    assert result["fills"].side.tolist() == ["BUY"]
    assert result["trades"].empty
    assert meta["final_positions"] == pytest.approx({"BTCUSDT": qty})
    assert result["equity"].equity.iloc[-1] == pytest.approx(50 + qty * 110)
    assert meta["unrealized_pnl"] == pytest.approx(qty * 110 - 950)
    assert meta["final_liquidation_equity"] == pytest.approx(50 + qty * 110 * .9998 * .999)
    assert meta["final_liquidation_equity"] < result["equity"].equity.iloc[-1]


def test_cluc_roi_is_one_percent_after_both_fees_and_slippage():
    frame = candles([100, 100, 100])
    frame.loc[frame.index[1], "high"] = 102
    result = single(frame, signal_rows(frame, entries=(0,)), strategy="cluc", fee=.001, slippage=.0002)
    trade = result["trades"].iloc[0]
    assert trade.exit_reason == "roi"
    assert trade.return_pct == pytest.approx(1.0)
    assert trade.pnl == pytest.approx(9.5)
    assert result["metadata"]["final_cash"] == pytest.approx(1009.5)


def test_same_minute_cluc_stop_has_priority_over_roi_even_on_entry_minute():
    frame = candles([100, 100])
    frame.loc[frame.index[0], ["high", "low"]] = [120, 80]
    result = single(frame, signal_rows(frame, entries=(0,)), strategy="cluc", fee=0, slippage=0)
    assert result["fills"].side.tolist() == ["BUY", "SELL"]
    assert result["fills"].timestamp.tolist() == [frame.index[0], frame.index[0]]
    assert result["trades"].exit_reason.tolist() == ["stop_loss"]
    assert result["trades"].exit_price.tolist() == pytest.approx([95])


def test_protective_stop_uses_gap_open_and_waits_for_a_new_boundary():
    frame = candles([100, 90, 100, 100, 100, 100, 100])
    result = single(frame, signal_rows(frame, entries=(0, 5)), protective_stop=.05, fee=0, slippage=0)
    fills = result["fills"]
    assert fills.side.tolist() == ["BUY", "SELL", "BUY"]
    assert fills.timestamp.tolist() == [frame.index[0], frame.index[1], frame.index[5]]
    assert fills.price.tolist() == pytest.approx([100, 90, 100])
    assert result["trades"].return_pct.tolist() == pytest.approx([-10])


def test_signal_exit_does_not_reenter_on_same_boundary():
    frame = candles([100.0] * 4)
    result = single(frame, signal_rows(frame, entries=(0, 2), exits=(2,)), fee=0, slippage=0)
    assert result["fills"].side.tolist() == ["BUY", "SELL"]


def test_filters_round_quantity_down_and_skip_orders_below_minimum_notional():
    frame = candles([100.0] * 3)
    result = single(frame, signal_rows(frame, entries=(0,)), fee=0, slippage=0,
                    filters={"step_size": 1, "minimum_notional": 10})
    assert result["fills"].qty.tolist() == [9]
    assert result["metadata"]["final_cash"] == 100
    skipped = single(frame, signal_rows(frame, entries=(0,)), fee=0, slippage=0,
                     filters={"minimum_notional": 951})
    assert skipped["fills"].empty
    assert skipped["metadata"]["final_cash"] == 1000


def test_rotation_has_separate_sell_and_buy_minutes_and_funds_both_legs():
    btc, eth = candles([100.0] * 7), candles([200.0] * 7)
    targets = pd.Series(["BTCUSDT", "ETHUSDT", "USDT"], index=btc.index[[0, 3, 6]])
    result = engine.simulate_rotation(
        {"BTCUSDT": btc, "ETHUSDT": eth}, targets,
        start=btc.index[0], end=btc.index[-1] + pd.Timedelta(minutes=1),
        capital=1000, fee=.001, slippage=.0002,
    )
    fills = result["fills"]
    assert list(zip(fills.symbol, fills.side)) == [
        ("BTCUSDT", "BUY"), ("BTCUSDT", "SELL"),
        ("ETHUSDT", "BUY"), ("ETHUSDT", "SELL"),
    ]
    assert fills.timestamp.tolist() == btc.index[[0, 3, 4, 6]].tolist()
    assert result["equity"].base_value.iloc[3] == 0
    cash_after_btc = 50 + 950 / (100.02 * 1.001) * 99.98 * .999
    eth_spend = cash_after_btc * .95
    eth_qty = eth_spend / (200.04 * 1.001)
    expected_cash = cash_after_btc - eth_spend + eth_qty * 199.96 * .999
    assert fills.qty.iloc[2] == pytest.approx(eth_qty)
    assert result["metadata"]["final_cash"] == pytest.approx(expected_cash)
    assert result["metadata"]["final_positions"] == {}
    assert result["metadata"]["fees"] == pytest.approx(fills.fee.sum())


def test_rotation_stop_does_not_reenter_until_a_later_target_boundary():
    btc, eth = candles([100, 90, 100, 100, 100, 100, 100]), candles([200.0] * 7)
    targets = pd.Series(["BTCUSDT", "BTCUSDT"], index=btc.index[[0, 5]])
    result = engine.simulate_rotation(
        {"BTCUSDT": btc, "ETHUSDT": eth}, targets,
        start=btc.index[0], end=btc.index[-1] + pd.Timedelta(minutes=1),
        capital=1000, fee=0, slippage=0, protective_stop=.05,
    )
    assert result["fills"].timestamp.tolist() == btc.index[[0, 1, 5]].tolist()


def test_rotation_requires_indicator_warmup_cash_guard_and_aligned_days():
    btc = candles(np.arange(100.0, 350.0), freq="D")
    eth = candles(np.arange(100.0, 350.0) ** 1.1, freq="D")
    targets = engine.rotation_targets({"BTCUSDT": btc, "ETHUSDT": eth})
    assert targets.iloc[:199].eq("USDT").all()
    assert targets.iloc[-1] == "ETHUSDT"
    eth.loc[eth.index[-1], ["open", "high", "low", "close"]] = 100
    btc.loc[btc.index[-1], ["open", "high", "low", "close"]] = 50
    assert engine.rotation_targets({"BTCUSDT": btc, "ETHUSDT": eth}).iloc[-1] == "USDT"
    with pytest.raises(ValueError):
        engine.rotation_targets({"BTCUSDT": btc, "ETHUSDT": eth.iloc[1:]})


def test_rotation_target_prefix_invariance():
    btc = candles(100 + np.arange(280) * .4, freq="D")
    eth = candles(200 + np.arange(280) * .9 + np.sin(np.arange(280)), freq="D")
    prefix = engine.rotation_targets({"BTCUSDT": btc.iloc[:230], "ETHUSDT": eth.iloc[:230]})
    eth.loc[eth.index[230:], ["open", "high", "low", "close"]] = 10000
    full = engine.rotation_targets({"BTCUSDT": btc, "ETHUSDT": eth})
    pd.testing.assert_series_equal(prefix, full.loc[prefix.index])


def test_buy_hold_uses_same_cash_allocation_and_costs_as_strategy_entry():
    frame = candles([100, 105, 110])
    baseline = engine.simulate_buy_hold(
        frame, symbol="BTCUSDT", start=frame.index[0],
        end=frame.index[-1] + pd.Timedelta(minutes=1), capital=1000,
        fee=.001, slippage=.0002,
    )
    expected = single(frame, signal_rows(frame, entries=(0,)), fee=.001, slippage=.0002)
    pd.testing.assert_frame_equal(baseline["equity"], expected["equity"])
    assert baseline["fills"].side.tolist() == ["BUY"]


def test_simulation_equity_and_fills_prefix_cannot_use_future_prices():
    frame = candles([100, 101, 102, 103, 104, 105, 106, 107, 108, 109])
    signals = signal_rows(frame, entries=(0, 8), exits=(5,))
    prefix = engine.simulate_single(
        frame.iloc[:8], signals.loc[signals.index < frame.index[8]],
        symbol="BTCUSDT", strategy="rsi2", start=frame.index[0],
        end=frame.index[8], capital=1000, fee=.001, slippage=.0002,
    )
    frame.loc[frame.index[8:], ["open", "high", "low", "close"]] = 10000
    full = single(frame, signals, fee=.001, slippage=.0002)
    pd.testing.assert_frame_equal(prefix["equity"], full["equity"].loc[prefix["equity"].index])
    pd.testing.assert_frame_equal(prefix["fills"], full["fills"].loc[full["fills"].timestamp < frame.index[8]])


def test_end_boundary_signal_is_excluded_and_empty_tables_keep_their_schema():
    frame = candles([100, 100, 100])
    end = frame.index[-1] + pd.Timedelta(minutes=1)
    signals = pd.DataFrame({"enter": [True], "exit": [False]}, index=pd.DatetimeIndex([end]))
    result = single(frame, signals, fee=0, slippage=0)
    assert result["fills"].empty and result["trades"].empty
    assert set(result["fills"].columns) >= {"timestamp", "symbol", "side", "qty", "price", "fee", "reason"}
    assert set(result["trades"].columns) >= {"entry_time", "exit_time", "pnl", "return_pct", "exit_reason"}
    assert result["equity"].equity.eq(1000).all()
    assert result["metadata"]["final_cash"] == 1000


def test_windows_require_minute_boundaries():
    frame = candles([100, 101, 102])
    with pytest.raises(ValueError):
        validate_minutes(frame, frame.index[0] + pd.Timedelta(seconds=1), frame.index[-1])
    with pytest.raises(ValueError):
        engine.simulate_single(
            frame, signal_rows(frame), symbol="BTCUSDT", strategy="rsi2",
            start=frame.index[0] + pd.Timedelta(seconds=1), end=frame.index[-1],
        )


def summarize_fixture(result, frame):
    return summarize(
        result, run_id="synthetic", strategy="rsi2", pair="BTCUSDT", scenario="base",
        variant="original", start=frame.index[0],
        end=frame.index[-1] + pd.Timedelta(minutes=1),
    )


def test_summary_drawdown_and_daily_sharpe_include_initial_entry_cost():
    frame = candles([100.0] * 2880)
    result = single(frame, signal_rows(frame, entries=(0,)), fee=.001, slippage=.0002)
    row, history, monthly = summarize_fixture(result, frame)
    qty = 950 / (100.02 * 1.001)
    cost_return = (50 + qty * 100) / 1000 - 1
    assert row["max_drawdown"] == pytest.approx(cost_return)
    assert history.equity.iloc[0] == 1000
    assert history.net_return.iloc[0] == 0
    assert history.net_return.iloc[1] == pytest.approx(cost_return)
    # Two daily returns are [entry cost loss, 0]; this uses the first day,
    # rather than discarding it through a bare pct_change().dropna().
    assert row["sharpe_daily"] == pytest.approx(-np.sqrt(365 / 2))
    assert monthly["return"].iloc[0] == pytest.approx(cost_return)


def test_summary_reconciles_closed_and_open_pnl_and_rejects_a_broken_bridge():
    frame = candles([100, 110, 120, 100, 130])
    result = single(frame, signal_rows(frame, entries=(0, 3), exits=(2,)), fee=.001, slippage=.0002)
    row, _, _ = summarize_fixture(result, frame)
    first_qty = 950 / (100.02 * 1.001)
    closed_pnl = first_qty * 119.976 * .999 - 950
    cash_after_sale = 1000 + closed_pnl
    second_cost = cash_after_sale * .95
    second_qty = second_cost / (100.02 * 1.001)
    open_pnl = second_qty * 130 - second_cost
    assert row["closed_trades"] == 1
    assert row["realized_pnl"] == pytest.approx(closed_pnl)
    assert row["unrealized_pnl"] == pytest.approx(open_pnl)
    assert row["net_pnl"] == pytest.approx(closed_pnl + open_pnl)
    assert row["reconciliation_error"] == pytest.approx(0, abs=1e-9)
    assert row["cagr"] is None and row["calmar"] is None
    broken = {**result, "metadata": {**result["metadata"], "unrealized_pnl": open_pnl + 1}}
    with pytest.raises(AssertionError, match="reconcile"):
        summarize_fixture(broken, frame)


def test_combined_exposure_is_value_weighted_and_cash_is_not_double_counted():
    frame = candles([100, 200])
    invested = single(frame, signal_rows(frame, entries=(0,)), fee=0, slippage=0)
    cash_only = engine.simulate_single(
        frame, signal_rows(frame), symbol="ETHUSDT", strategy="rsi2",
        start=frame.index[0], end=frame.index[-1] + pd.Timedelta(minutes=1),
        capital=3000, fee=0, slippage=0,
    )
    combined = combine_sleeves([invested, cash_only])
    last = combined["equity"].iloc[-1]
    assert last["equity"] == 4950
    assert last["cash"] == 3050
    assert last["base_value"] == 1900
    assert last["exposure"] == pytest.approx(1900 / 4950)
    assert last["exposure"] != pytest.approx(invested["equity"].exposure.iloc[-1] / 2)
    assert combined["metadata"]["initial_capital"] == 4000
    assert combined["metadata"]["unrealized_pnl"] == 950


def test_identical_results_have_identical_fingerprints_and_changed_equity_does_not():
    frame = candles([100, 105, 110])
    signals = signal_rows(frame, entries=(0,))
    first = single(frame, signals, fee=.001, slippage=.0002)
    repeated = single(frame.copy(), signals.copy(), fee=.001, slippage=.0002)
    original_hashes = result_fingerprints(first)
    assert original_hashes == result_fingerprints(repeated)
    assert set(original_hashes) == {"equity", "fills", "trades"}
    assert all(len(digest) == 64 for digest in original_hashes.values())
    frame.loc[frame.index[-1], ["open", "high", "low", "close"]] = 120
    changed = result_fingerprints(single(frame, signals, fee=.001, slippage=.0002))
    assert changed["equity"] != original_hashes["equity"]
    assert changed["fills"] == original_hashes["fills"]


def archive_fixture(path, units):
    path.parent.mkdir(parents=True, exist_ok=True)
    index = pd.date_range("2026-01-01T00:00Z", periods=3, freq="min")
    divisor = 1_000_000 if units == "ms" else 1000
    payload = "".join(f"{stamp.value // divisor},100,101,99,100,1\n" for stamp in index)
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("synthetic-candles.csv", payload)
    return index


@pytest.mark.parametrize("units", ["ms", "us"])
def test_public_archive_millisecond_and_microsecond_units_have_same_minutes(tmp_path, units):
    path = tmp_path / "synthetic.zip"
    expected = archive_fixture(path, units)
    frame, detected = market_data._read_archive(path)
    assert detected == units
    assert frame.index.equals(expected)
    assert frame.close.tolist() == [100, 100, 100]
    assert frame.volume.tolist() == [1, 1, 1]


def test_offline_archive_rechecks_checksum_without_network_or_repair(tmp_path, monkeypatch):
    relative = "monthly/klines/BTCUSDT/1m/synthetic.zip"
    path = tmp_path / "raw" / relative
    expected = archive_fixture(path, "us")
    original_digest = hashlib.sha256(path.read_bytes()).hexdigest()
    checksum = path.with_name(path.name + ".CHECKSUM")
    checksum.write_text(original_digest + "  synthetic.zip\n", encoding="utf-8")
    monkeypatch.setattr(market_data, "_response", lambda *a, **kw: pytest.fail("Offline read attempted a network call"))
    frame, metadata = market_data._get_archive(relative, tmp_path, offline=True)
    assert frame.index.equals(expected)
    assert metadata["sha256"] == original_digest
    assert metadata["validated"] is True
    assert not path.with_name(path.name + ".metadata.json").exists()
    corrupted = path.read_bytes() + b"changed-after-download"
    path.write_bytes(corrupted)
    with pytest.raises(ValueError, match="checksum mismatch"):
        market_data._get_archive(relative, tmp_path, offline=True)
    assert path.read_bytes() == corrupted
    assert checksum.read_text(encoding="utf-8").split()[0] == original_digest


def test_exchange_filters_use_compact_symbols_and_explicit_current_snapshot(monkeypatch):
    calls = []
    payload = {"symbols": [{"symbol": symbol, "filters": [
        {"filterType": "PRICE_FILTER", "tickSize": "0.01"},
        {"filterType": "LOT_SIZE", "stepSize": "0.00001"},
        {"filterType": "NOTIONAL", "minNotional": "5.00"},
    ]} for symbol in ("BTCUSDT", "ETHUSDT")]}

    class PublicResponse:
        url = "https://data-api.binance.vision/api/v3/exchangeInfo"
        content = json.dumps(payload).encode("utf-8")

        def json(self):
            return payload

        def close(self):
            pass

    def request(url, *, params):
        calls.append((url, params))
        assert params == {"symbols": '["BTCUSDT","ETHUSDT"]'}
        return PublicResponse()

    monkeypatch.setattr(market_data, "_response", request)
    filters, metadata = market_data.fetch_exchange_filters(["BTCUSDT", "ETHUSDT"])
    assert len(calls) == 1
    assert set(filters) == {"BTCUSDT", "ETHUSDT"}
    assert filters["BTCUSDT"] == {"tick_size": .01, "step_size": .00001, "minimum_notional": 5}
    assert metadata["historical"] is False
    assert metadata["sha256"] == hashlib.sha256(PublicResponse.content).hexdigest()
    assert metadata["retrieved_at"]

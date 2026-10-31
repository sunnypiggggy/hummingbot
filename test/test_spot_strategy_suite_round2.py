"""Independent synthetic validation for the second frozen spot experiment.

No network, account, production configuration or full-year results are used.
Indicator references below use elementary arithmetic rather than the strategy
implementation, so changes in seeding, crossing or decision timing are visible.
"""

import numpy as np
import pandas as pd
import pytest

from scripts import spot_strategy_round2 as round2
from scripts import spot_strategy_suite_engine as original
from scripts.spot_strategy_suite_reporting import result_fingerprints


IDS = ("supertrend", "squeeze", "ibs", "mfi", "absolute_momentum")


def candles(prices, *, start="2025-01-01T00:00:00Z", freq="4h", spread=1.0):
    values = np.asarray(prices, dtype=float)
    index = pd.date_range(start, periods=len(values), freq=freq)
    return pd.DataFrame({"open": values, "high": values + spread,
                         "low": values - spread, "close": values,
                         "volume": np.arange(len(values), dtype=float) % 17 + 1}, index=index)


def signals(frame, entries=(), exits=()):
    offsets = sorted(set(entries) | set(exits))
    return pd.DataFrame({"enter": [i in entries for i in offsets],
                         "exit": [i in exits for i in offsets]}, index=frame.index[offsets])


def as_full(frame, strategy):
    result = round2.generate_signals(frame, strategy)
    assert set(("enter", "exit")).issubset(result.columns)
    assert result.index.is_unique
    assert result.index.isin(frame.index).all()
    return result.reindex(frame.index, fill_value=False)[["enter", "exit"]]


def wilder_atr(frame, period):
    """TA-Lib's ATR seed excludes bar zero, then applies Wilder recursion."""
    tr = []
    for i in range(1, len(frame)):
        tr.append(max(frame.high.iloc[i] - frame.low.iloc[i],
                      abs(frame.high.iloc[i] - frame.close.iloc[i - 1]),
                      abs(frame.low.iloc[i] - frame.close.iloc[i - 1])))
    result = np.full(len(frame), np.nan)
    if len(frame) > period:
        result[period] = sum(tr[:period]) / period
        for i in range(period + 1, len(frame)):
            result[i] = (result[i - 1] * (period - 1) + tr[i - 1]) / period
    return result


def expected_supertrend(frame):
    atr = wilder_atr(frame, 10)
    upper = lower = None
    up = False
    entries, exits = [], []
    for i in range(len(frame)):
        if np.isnan(atr[i]):
            entries.append(False)
            exits.append(False)
            continue
        middle = (frame.high.iloc[i] + frame.low.iloc[i]) / 2
        raw_upper, raw_lower = middle + 3 * atr[i], middle - 3 * atr[i]
        if upper is None:
            upper, lower = raw_upper, raw_lower
            up = False
        else:
            previous_upper, previous_lower = upper, lower
            upper = raw_upper if raw_upper < previous_upper or frame.close.iloc[i - 1] > previous_upper else previous_upper
            lower = raw_lower if raw_lower > previous_lower or frame.close.iloc[i - 1] < previous_lower else previous_lower
            up = frame.close.iloc[i] >= lower if up else frame.close.iloc[i] > upper
        entries.append(bool(up))
        exits.append(not bool(up))
    return pd.DataFrame({"enter": entries, "exit": exits}, index=frame.index)


@pytest.mark.parametrize("strategy", IDS)
def test_truncating_future_candles_does_not_change_past_signals(strategy):
    rng = np.random.default_rng(1729)
    close = 100 + np.cumsum(rng.normal(0, 0.8, 450))
    frame = candles(close, freq="D" if strategy in ("ibs", "absolute_momentum") else "4h")
    prefix = frame.iloc[:327]
    future = frame.copy()
    future.loc[frame.index[327:], ["open", "high", "low", "close"]] *= 7
    future.loc[frame.index[327:], "volume"] *= 99
    pd.testing.assert_frame_equal(as_full(prefix, strategy), as_full(future, strategy).loc[prefix.index])


def test_ibs_uses_close_position_and_includes_threshold_equalities():
    frame = candles([2, 8, 5, 3], freq="D", spread=0)
    frame["high"] = [10.0, 10.0, 5.0, 10.0]
    frame["low"] = [0.1, 0.1, 5, 0.1]
    # Use exactly representable levels to avoid testing binary float tolerance.
    frame.loc[frame.index[:2], "low"] = 0.5
    frame.loc[frame.index[:2], "high"] = 10.5
    frame.loc[frame.index[0], ["open", "close"]] = 2.5
    frame.loc[frame.index[1], ["open", "close"]] = 8.5
    result = as_full(frame, "ibs")
    assert result.enter.tolist() == [True, False, False, False]
    assert result.exit.tolist() == [False, True, False, False]


def test_ibs_zero_range_is_neutral_and_has_no_phantom_signal():
    result = as_full(candles([100] * 12, freq="D", spread=0), "ibs")
    assert not result.to_numpy().any()


def test_supertrend_matches_independent_true_range_recursion_and_band_ratchet():
    price = np.r_[np.linspace(100, 103, 25), np.linspace(103, 170, 40),
                  np.linspace(170, 90, 40), np.linspace(90, 135, 35)]
    frame = candles(price, spread=1.5)
    result = as_full(frame, "supertrend")
    pd.testing.assert_frame_equal(result, expected_supertrend(frame))
    assert result.enter.any() and result.exit.any()
    assert not result.iloc[:10].to_numpy().any()


def test_supertrend_is_state_not_only_one_crossover_pulse():
    frame = candles(np.r_[np.full(20, 100), np.arange(101, 181)], spread=1)
    result = as_full(frame, "supertrend")
    assert result.enter.iloc[-10:].all()
    assert not result.exit.iloc[-10:].any()


def test_squeeze_fires_on_release_once_and_exits_on_reversal():
    frame = candles([100] * 45 + [120, 121, 122, 90], spread=0.2)
    result = as_full(frame, "squeeze")
    assert not result.enter.iloc[:45].any()
    assert bool(result.enter.iloc[45])
    assert not result.enter.iloc[46:48].any(), "Continuing expansion is not a new release"
    assert bool(result.exit.iloc[48])


def test_squeeze_does_not_trade_an_expansion_without_preceding_compression():
    # A gradual trend has wide close dispersion relative to its small true
    # ranges, so BB remains outside KC before and during this later expansion.
    price = np.r_[np.arange(100., 160.), [200., 210., 220.]]
    frame = candles(price, spread=0.1)
    result = as_full(frame, "squeeze")
    assert not result.enter.iloc[-3:].any()


def test_mfi_uses_price_times_volume_and_directional_crosses():
    price = np.array([100, 110] * 15 + list(np.arange(109, 89, -1)) + list(np.arange(90, 121)), dtype=float)
    frame = candles(price, spread=1)
    typical = (frame.high.to_numpy() + frame.low.to_numpy() + price) / 3
    raw = typical * frame.volume.to_numpy()
    mfi = np.full(len(price), np.nan)
    for i in range(14, len(price)):
        plus = minus = 0.0
        for j in range(i - 13, i + 1):
            if typical[j] > typical[j - 1]:
                plus += raw[j]
            elif typical[j] < typical[j - 1]:
                minus += raw[j]
        mfi[i] = 100 * plus / (plus + minus) if plus + minus else 0
    expected_enter = [False] + [bool(mfi[i - 1] >= 20 and mfi[i] < 20) for i in range(1, len(price))]
    expected_exit = [False] + [bool(mfi[i - 1] <= 80 and mfi[i] > 80) for i in range(1, len(price))]
    actual = as_full(frame, "mfi")
    assert actual.enter.tolist() == expected_enter
    assert actual.exit.tolist() == expected_exit
    assert actual.enter.sum() > 0 and actual.exit.sum() > 0


def test_mfi_sustained_extreme_is_not_repeated_crossing():
    frame = candles(np.linspace(200, 100, 100), spread=0.2)
    result = as_full(frame, "mfi")
    assert not result.enter.iloc[30:].any()


def test_absolute_momentum_uses_180_completed_calendar_days_and_monday_only():
    frame = candles(np.linspace(100, 500, 410), freq="D")
    result = as_full(frame, "absolute_momentum")
    eligible = (frame.index.dayofweek == 0) & (np.arange(len(frame)) >= 180)
    assert np.array_equal(result.enter.to_numpy(), eligible)
    assert not result.exit.any()
    assert not result.iloc[:180].to_numpy().any()


def test_absolute_momentum_equal_price_goes_to_cash_on_weekly_boundary():
    frame = candles(np.full(250, 100), freq="D")
    result = as_full(frame, "absolute_momentum")
    eligible = (frame.index.dayofweek == 0) & (np.arange(len(frame)) >= 180)
    assert not result.enter.any()
    assert np.array_equal(result.exit.to_numpy(), eligible)


def test_absolute_momentum_does_not_respond_midweek():
    frame = candles(np.r_[np.full(210, 100), np.full(14, 70)], freq="D")
    result = as_full(frame, "absolute_momentum")
    decisions = result.index[result.any(axis=1)]
    assert len(decisions)
    assert (decisions.dayofweek == 0).all()
    assert (decisions.hour == 0).all()


@pytest.mark.parametrize("strategy", IDS)
@pytest.mark.parametrize("stop", [None, 0.05])
def test_external_signal_adapter_preserves_verified_cash_inventory_and_costs(strategy, stop):
    frame = candles([100, 101, 90, 95, 102, 98, 103, 104, 100], freq="min", spread=0.1)
    signal = signals(frame, entries=(0, 1, 3, 5, 7), exits=(4, 8))
    kwargs = dict(symbol="BTCUSDT", start=frame.index[0], end=frame.index[-1] + pd.Timedelta(minutes=1),
                  capital=1000, fee=0.001, slippage=0.0002, protective_stop=stop,
                  filters={"step_size":0.00001,"minimum_notional":5})
    expected = original.simulate_single(frame, signal, strategy="buy_hold", **kwargs)
    actual = round2.simulate_strategy(frame, signal, strategy=strategy, **kwargs)
    assert result_fingerprints(actual) == result_fingerprints(expected)
    assert actual["metadata"]["strategy"] == strategy
    assert actual["metadata"]["roi"] is None
    assert (actual["equity"].cash >= 0).all()
    assert (actual["equity"].base_value >= 0).all()


def test_protective_exit_waits_for_next_actual_strategy_decision():
    frame = candles([100, 90, 100, 100, 100, 100, 100], freq="min", spread=0)
    signal = signals(frame, entries=(0, 5))
    result = round2.simulate_strategy(frame, signal, strategy="supertrend", symbol="BTCUSDT",
                                      start=frame.index[0], end=frame.index[-1] + pd.Timedelta(minutes=1),
                                      capital=1000, fee=0, slippage=0, protective_stop=0.05)
    buys = result["fills"].loc[result["fills"].side == "BUY", "timestamp"]
    assert buys.tolist() == [frame.index[0], frame.index[5]]
    sell = result["fills"].loc[result["fills"].side == "SELL"].iloc[0]
    assert sell.price == 90, "A gap through a stop uses the worse opening price"


def test_deterministic_adapter_repeat():
    frame = candles([100, 110, 101, 96, 105], freq="min", spread=0.2)
    signal = signals(frame, entries=(0, 3), exits=(2, 4))
    kwargs = dict(strategy="ibs", symbol="ETHUSDT", start=frame.index[0],
                  end=frame.index[-1] + pd.Timedelta(minutes=1),capital=5000,
                  fee=0.002,slippage=0.0005,protective_stop=0.05)
    assert result_fingerprints(round2.simulate_strategy(frame,signal,**kwargs)) == result_fingerprints(round2.simulate_strategy(frame,signal,**kwargs))


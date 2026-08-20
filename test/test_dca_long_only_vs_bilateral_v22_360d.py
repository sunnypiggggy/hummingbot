from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from backtest_dca_long_only_vs_bilateral_v22_360d import build_v22_gate  # noqa: E402
from backtest_dca_momentum_guard import run_pair_guarded  # noqa: E402


def candles(prices: list[float], start: int = 1_700_000_000) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "timestamp": start + offset * 300,
            "open": close,
            "high": close * 1.02,
            "low": close * 0.98,
            "close": close,
            "volume": 1.0,
        }
        for offset, close in enumerate(prices)
    ])


def test_long_only_removes_sell_executor_without_changing_buy_path() -> None:
    frame = candles([100.0] * 30)
    gate = pd.Series(True, index=frame.index)
    long_summary, long_trades, _ = run_pair_guarded(
        frame, gate, "BTC-USDT", 0.001, 2.0,
        refresh_seconds=900, time_limit_seconds=900,
        guarded_sides=("BUY",), active_sides=("BUY",),
    )
    bilateral_summary, bilateral_trades, _ = run_pair_guarded(
        frame, gate, "BTC-USDT", 0.001, 2.0,
        refresh_seconds=900, time_limit_seconds=900,
        guarded_sides=("BUY", "SELL"), active_sides=("BUY", "SELL"),
    )

    assert long_summary["active_sides"] == "BUY"
    assert set(long_trades.side) == {"BUY"}
    assert long_summary["positioned_executors_by_side"]["SELL"] == 0
    assert bilateral_summary["positioned_executors_by_side"]["SELL"] > 0
    assert set(bilateral_trades.side) == {"BUY", "SELL"}
    assert long_summary["positioned_executors_by_side"]["BUY"] == bilateral_summary["positioned_executors_by_side"]["BUY"]


def test_v22_gate_is_fail_closed_outside_signed_coverage() -> None:
    start = 1_700_000_000
    frame = candles([100.0] * 30, start=start)
    states = pd.DataFrame([
        {
            "pair": "BTC-FDUSD", "signal_ts": start + 300,
            "risk_off_active": False, "recommended_buy_enabled": True,
        },
        {
            "pair": "BTC-FDUSD", "signal_ts": start + 3900,
            "risk_off_active": True, "recommended_buy_enabled": False,
        },
    ])

    gate, audit, coverage = build_v22_gate(frame, states, "BTC-USDT")

    assert coverage == {"start": start + 300, "end": start + 7500}
    assert not bool(gate.iloc[0])
    assert bool(gate.iloc[1])
    assert not bool(gate.loc[frame.timestamp.ge(start + 3900)].iloc[0])
    assert not audit.loc[frame.timestamp.ge(start + 7500), "v22_available"].any()
    assert not gate.loc[frame.timestamp.ge(start + 7500)].any()


def test_guarded_sides_cannot_include_an_inactive_side() -> None:
    frame = candles([100.0] * 5)
    gate = pd.Series(True, index=frame.index)
    try:
        run_pair_guarded(
            frame, gate, "ETH-USDT", 0.001, 2.0,
            guarded_sides=("BUY", "SELL"), active_sides=("BUY",),
        )
    except ValueError as exc:
        assert "subset" in str(exc)
    else:
        raise AssertionError("inactive guarded side must be rejected")

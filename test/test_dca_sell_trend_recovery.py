from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from backtest_dca_sell_trend_recovery_360d import run_pair, trend_features  # noqa: E402


def candles(prices: list[float]) -> pd.DataFrame:
    rows = []
    for offset, close in enumerate(prices):
        rows.append({
            "timestamp": 1_700_000_000 + offset * 300,
            "open": close,
            "high": close * 1.001,
            "low": close * 0.999,
            "close": close,
            "volume": 1.0,
        })
    return pd.DataFrame(rows)


def test_strong_uptrend_uses_only_current_and_prior_completed_bars() -> None:
    base = [100.0] * 60
    frame = candles(base + [101.0 + offset for offset in range(20)])
    features = trend_features(frame)
    first_trigger = features.index[features.trigger].min()
    assert pd.notna(first_trigger)

    changed_future = frame.copy()
    changed_future.loc[first_trigger + 1:, "close"] = 1.0
    replay = trend_features(changed_future)
    assert bool(replay.trigger.loc[first_trigger]) == bool(features.trigger.loc[first_trigger])


def test_sell_trend_gate_blocks_new_sell_without_blocking_buy_lifecycle() -> None:
    prices = [100.0] * 60 + [100.0 * (1.01 ** offset) for offset in range(1, 60)]
    frame = candles(prices)
    summary, trades, curve = run_pair(
        frame,
        "ETH-USDT",
        fee_rate=0.001,
        trend_gate=True,
        sell_stop_cooldown_seconds=1800,
    )

    assert summary["sell_blocked_hours"] > 0
    assert summary["trend_blocks"] >= 1
    assert curve.sell_blocked.any()
    assert "BUY" in set(trades.side)


def test_validated_cooldowns_are_deterministic_and_never_bypass_trend_recovery() -> None:
    prices = [100.0] * 60 + [100.0 * (1.006 ** offset) for offset in range(1, 40)] + [126.0] * 40
    frame = candles(prices)
    results = []
    for cooldown in (1800, 7200, 21600):
        summary, _, _ = run_pair(
            frame,
            "ETH-USDT",
            fee_rate=0.001,
            trend_gate=True,
            sell_stop_cooldown_seconds=cooldown,
        )
        results.append(summary)

    assert all(item["sell_blocked_hours"] > 0 for item in results)
    assert results[0]["sell_blocked_hours"] <= results[1]["sell_blocked_hours"] <= results[2]["sell_blocked_hours"]

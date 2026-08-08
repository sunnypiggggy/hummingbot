from __future__ import annotations

import pandas as pd
import pytest

from scripts.compare_momentum_boosting_models import (
    ExperimentConfig,
    chronological_split,
    simulate_symbol,
    strategy_metrics,
)


def test_chronological_split_purges_the_label_horizon() -> None:
    timestamps = pd.date_range("2026-01-01", periods=100, freq="1h", tz="UTC")
    panel = pd.DataFrame(
        {
            "datetime": timestamps,
            "symbol": "BTC",
            "target": 0,
        }
    )
    config = ExperimentConfig(symbols=("BTC",), horizon_bars=6)

    splits, info = chronological_split(panel, config)

    train_cut = pd.Timestamp(info["train_cut_utc"])
    validation_cut = pd.Timestamp(info["validation_cut_utc"])
    assert splits["train"]["datetime"].max() + pd.Timedelta(hours=6) < train_cut
    assert splits["validation"]["datetime"].max() + pd.Timedelta(hours=6) < validation_cut
    assert splits["validation"]["datetime"].min() == train_cut
    assert splits["test"]["datetime"].min() == validation_cut


def test_momentum_atr_stop_uses_carried_stop_and_gap_open() -> None:
    frame = pd.DataFrame(
        {
            "datetime": pd.date_range("2026-01-01", periods=3, freq="1h", tz="UTC"),
            "open": [100.0, 100.0, 98.0],
            "high": [101.0, 102.0, 99.0],
            "low": [99.0, 99.0, 97.0],
            "close": [100.0, 101.0, 97.5],
            "atr": [1.0, 1.0, 1.0],
            "momentum_score": [2, 2, -2],
            "probability": [0.80, 0.80, 0.20],
        }
    )
    config = ExperimentConfig(
        symbols=("BTC",),
        fee_bps=0.0,
        slippage_bps=0.0,
        normal_stop_atr=2.5,
        weak_stop_atr=1.25,
    )

    path, trades = simulate_symbol(frame, "BTC", 0.60, 0.50, config)

    assert len(trades) == 1
    assert trades.iloc[0]["exit_reason"] == "momentum_atr_stop"
    assert trades.iloc[0]["exit_price"] == pytest.approx(98.0)
    assert int(path["stop_event"].sum()) == 1


def test_strategy_metrics_include_the_move_from_initial_capital() -> None:
    index = pd.date_range("2026-01-01", periods=2, freq="1h", tz="UTC")
    equity = pd.Series([0.90, 1.00], index=index)
    exposure = pd.Series([1.0, 1.0], index=index)
    stop_events = pd.Series([0, 0], index=index)

    metrics = strategy_metrics(equity, exposure, pd.DataFrame(), stop_events, "1h")

    assert metrics.total_return_pct == pytest.approx(0.0)
    assert metrics.max_drawdown_pct == pytest.approx(-10.0)

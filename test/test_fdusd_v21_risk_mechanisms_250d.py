from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from backtest_fdusd_v21_risk_mechanisms_250d import scenario_matrix  # noqa: E402
from validate_grid_live import (  # noqa: E402
    Candidate, ExecutionFilter, RiskMechanismConfig, simulate,
)


def candles(start: float, count: int = 1_000, decline: float = .20) -> pd.DataFrame:
    timestamps = 1_700_000_000 + np.arange(count) * 300
    close = start * (1 - decline * np.arange(count) / (count - 1))
    return pd.DataFrame({
        "timestamp": timestamps, "open": close,
        "high": close * 1.002, "low": close * .998,
        "close": close, "volume": 10.0,
    })


def pair_candles() -> dict[str, pd.DataFrame]:
    return {"BTC-FDUSD": candles(65_000), "ETH-FDUSD": candles(3_500)}


def disabled(**overrides: bool) -> RiskMechanismConfig:
    values = dict(pair_loss=False, pair_drawdown=False, portfolio_loss=False,
                  portfolio_drawdown=False, continue_after_portfolio_stop=True,
                  restore_portfolio_inventory=True)
    values.update(overrides)
    return RiskMechanismConfig(**values)


def test_mechanism_matrix_has_independent_singles_and_exact_leave_one_out() -> None:
    matrix = {item.scenario: item for item in scenario_matrix()}
    assert len(matrix) == 16
    for number in range(2, 7):
        single = matrix[f"mechanism_{number}"]
        enabled = [i for i in range(2, 7) if getattr(single, f"mechanism_{i}")]
        assert enabled == [number]
    for removed in range(1, 7):
        item = matrix[f"leave_out_{removed}"]
        enabled = [item.gate == "v21"] + [getattr(item, f"mechanism_{i}") for i in range(2, 7)]
        assert enabled[removed - 1] is False
        assert sum(enabled) == 5


def test_pair_loss_and_drawdown_breakers_do_not_share_a_hidden_master_switch() -> None:
    trade_log: list[dict] = []
    result, curve, _ = simulate(
        pair_candles(), Candidate(.03, .006, .006, .015), 0.0, taker_fee=.001,
        risk_breakers_enabled=False, risk_mechanisms=disabled(pair_loss=True),
        cost_floor_enabled=False, trade_log=trade_log,
    )
    triggers = {row.get("trigger") for row in trade_log if row["reason"] == "pair_breaker_flatten"}
    assert triggers == {"pair_loss"}
    assert result["risk_mechanisms"]["pair_drawdown"] is False
    assert not result["liquidated"]
    assert len(curve) == len(pair_candles()["BTC-FDUSD"]) - 1


def test_portfolio_breaker_restores_inventory_and_continues_to_window_end() -> None:
    trade_log: list[dict] = []
    source = pair_candles()
    result, curve, pairs = simulate(
        source, Candidate(.03, .006, .006, .015), 0.0, taker_fee=.001,
        risk_breakers_enabled=False, risk_mechanisms=disabled(portfolio_loss=True),
        cost_floor_enabled=False, trade_log=trade_log,
    )
    stops = [row for row in trade_log if row["reason"] == "portfolio_breaker"]
    assert len(stops) == 1 and stops[0]["trigger"] == "portfolio_loss"
    assert result["liquidated"]
    assert len(curve) == len(source["BTC-FDUSD"]) - 1
    assert int(curve.timestamp.iloc[-1]) == int(source["BTC-FDUSD"].timestamp.iloc[-1])
    assert all(abs(item["inventory_delta"]) < 1e-12 for item in pairs.values())


def test_pair_specific_v21_style_gate_only_suppresses_its_own_buys() -> None:
    source = pair_candles()
    btc_off = {int(ts): False for ts in source["BTC-FDUSD"].timestamp}
    eth_on = {int(ts): True for ts in source["ETH-FDUSD"].timestamp}
    _, _, pairs = simulate(
        source, Candidate(.03, .006, .006, .015), 0.0,
        technical_buy_gate={"BTC-FDUSD": btc_off, "ETH-FDUSD": eth_on},
        risk_breakers_enabled=False, risk_mechanisms=disabled(),
        cost_floor_enabled=False,
    )
    assert pairs["BTC-FDUSD"]["buys"] == 0
    assert pairs["ETH-FDUSD"]["buys"] > 0


def test_exchange_filter_snapshot_quantizes_grid_price_quantity_and_notional() -> None:
    filters = {
        "BTC-FDUSD": ExecutionFilter(.01, .00001, 5),
        "ETH-FDUSD": ExecutionFilter(.01, .0001, 5),
    }
    trades: list[dict] = []
    simulate(
        pair_candles(), Candidate(.03, .006, .006, .015), 0.0,
        risk_breakers_enabled=False, risk_mechanisms=disabled(),
        cost_floor_enabled=False, execution_filters=filters, trade_log=trades,
    )
    fills = [row for row in trades if row["reason"] == "grid_fill"]
    assert fills
    for fill in fills:
        item = filters[fill["pair"]]
        assert abs(fill["price"] / item.tick_size - round(fill["price"] / item.tick_size)) < 1e-7
        assert abs(fill["amount"] / item.step_size - round(fill["amount"] / item.step_size)) < 1e-7
        assert fill["quote_notional"] >= item.min_notional

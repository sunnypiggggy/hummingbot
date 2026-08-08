from decimal import Decimal
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from grid_live_common import PORTFOLIOS, build_live_config  # noqa: E402


def test_grid_risk_execution_switches_are_independent(monkeypatch) -> None:
    monkeypatch.setenv("GRID_RISK_V21_BUY_GATE_ENABLED", "false")
    monkeypatch.setenv("GRID_RISK_FOMC_GATE_ENABLED", "true")
    monkeypatch.setenv("GRID_RISK_STRATEGY_LOSS_BREAKER_ENABLED", "false")
    monkeypatch.setenv("GRID_RISK_STRATEGY_DRAWDOWN_BREAKER_ENABLED", "true")
    monkeypatch.setenv("GRID_RISK_PORTFOLIO_LOSS_BREAKER_ENABLED", "true")
    monkeypatch.setenv("GRID_RISK_PORTFOLIO_DRAWDOWN_BREAKER_ENABLED", "false")
    monkeypatch.setenv("GRID_RISK_POSITION_PROTECTION_ENABLED", "false")
    portfolio = PORTFOLIOS["FDUSD"]
    config = build_live_config(
        portfolio,
        {"BTC-FDUSD": Decimal("65000"), "ETH-FDUSD": Decimal("3500")},
        Decimal("0"),
    )
    assert config["technical_buy_gate_enabled"] is False
    assert config["macro_gate_enabled"] is True
    assert config["pair_loss_breaker_enabled"] is False
    assert config["pair_drawdown_breaker_enabled"] is True
    assert config["portfolio_loss_breaker_enabled"] is True
    assert config["portfolio_drawdown_breaker_enabled"] is False
    assert config["cost_floor_enabled"] is False
    assert config["inventory_exit_enabled"] is False

"""Compatibility checks for the shared Guard clients and fill accounting."""

from copy import deepcopy
from decimal import Decimal
from pathlib import Path
import subprocess
import sys

import pytest

from live_guard.emergency_accounting import emergency_fill_metrics


@pytest.mark.parametrize("side,base,cash", [("BUY", "2", "-20"), ("SELL", "-2", "20")])
@pytest.mark.parametrize("asset,base_fee,quote_fee", [
    ("BTC", "0.01", "0.10"), ("USDT", "0", "0.01"), ("OTHER", "0", "0"),
])
def test_shared_metrics_preserve_existing_fee_and_cashflow_semantics(side, base, cash, asset, base_fee, quote_fee):
    response = {
        "executedQty": "2", "cummulativeQuoteQty": "20",
        "fills": [{"price": "10", "commission": "0.01", "commissionAsset": asset}],
    }
    original = deepcopy(response)
    result = emergency_fill_metrics("BTC-USDT", side, response)
    assert Decimal(result["base_delta"]) == Decimal(base) - Decimal(base_fee)
    assert result["quote_cashflow"] == cash
    assert Decimal(result["fee_quote"]) == Decimal(quote_fee)
    assert result["fee_details"] == [{"asset": asset, "commission": "0.01", "fill_price": "10"}]
    assert response == original


def test_shared_metrics_preserve_no_execution_error():
    response = {"executedQty": "0"}
    with pytest.raises(RuntimeError, match="emergency order reported no executed quantity"):
        emergency_fill_metrics("BTC-USDT", "SELL", response)


@pytest.mark.parametrize("layout", ["package", "flat"])
def test_clients_import_without_guard_orchestration(layout):
    root = Path(__file__).resolve().parents[1]
    paths = [root] if layout == "package" else [root / "live_guard", root / "scripts"]
    module = "live_guard.emergency_clients" if layout == "package" else "emergency_clients"
    code = (
        f"import sys; sys.path[:0] = {[str(path) for path in paths]!r}; "
        f"import {module} as clients; "
        "assert clients.BinanceEmergencyClient; assert clients.DockerEmergencyClient; "
        "assert not any(name in sys.modules for name in "
        "('dca_live_guard', 'live_guard.dca_live_guard', 'grid_live_guard', 'live_guard.grid_live_guard'))"
    )
    result = subprocess.run(
        [sys.executable, "-I", "-c", code], capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr

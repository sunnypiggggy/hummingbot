import json
from decimal import Decimal as D

import pytest

from scripts.dca_live_common import base_fee_correction, trade_pnl_from_rows
from live_guard.dca_live_report import row_metrics
from live_guard.telegram_notifications import _format_inventory_impact


def fill(side="BUY", token="ETH", fee="0.00000720"):
    return (side, 2564590000, 7200, 18465, 1789873350000,
            json.dumps({"percent": "0", "flat_fees": [{"token": token, "amount": fee}]}), "ETH-USDT")


def test_actual_base_fee_reduces_inventory_not_cash_twice():
    row = fill()
    result = trade_pnl_from_rows([row], D("2564.59"))
    report = row_metrics([row])
    assert result["net_base"] == D("0.0071928")
    assert result["quote_cashflow"] - result["fees_quote"] == -D("0.0072") * D("2564.59")
    assert report["net_base"] == result["net_base"]
    assert report["cashflow_quote"] == result["quote_cashflow"]
    assert trade_pnl_from_rows([row], D("2564.59")) == result


def test_quote_and_third_currency_do_not_change_base():
    for token in ("USDT", "BNB"):
        assert base_fee_correction(fill(token=token)) == (D(0), D(0))


def test_sell_fee_and_partial_fills_are_actual_units():
    result = trade_pnl_from_rows([fill("SELL"), fill("BUY")], D("2564.59"))
    assert result["net_base"] == -D("0.0000144")


def test_legacy_rows_are_not_guessed_and_invalid_evidence_is_rejected():
    assert base_fee_correction(fill()[:5]) == (D(0), D(0))
    with pytest.raises(ValueError):
        base_fee_correction(fill(fee="NaN"))
    with pytest.raises(ValueError):
        base_fee_correction((*fill()[:5], "broken", "ETH-USDT"))


def test_deficit_message_does_not_claim_normal_trading():
    text = _format_inventory_impact({"ownership_deficit": "0.00000552", "runtime": {"trading_normal": True}})
    assert "交易正常" not in text
    assert "禁止按此账本执行无归属清仓" in text

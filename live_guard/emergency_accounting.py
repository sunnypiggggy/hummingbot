"""Pure emergency fill metrics shared by the Grid and DCA Guards."""

from decimal import Decimal
from typing import Any, Dict


def emergency_fill_metrics(pair: str, side: str, response: dict) -> Dict[str, Any]:
    executed = Decimal(str(response.get("executedQty", "0")))
    quote = Decimal(str(response.get("cummulativeQuoteQty", "0")))
    if executed <= 0:
        raise RuntimeError(f"emergency order reported no executed quantity: {response}")
    base_asset, quote_asset = pair.split("-", 1)
    base_delta = executed if side == "BUY" else -executed
    quote_cashflow = -quote if side == "BUY" else quote
    fee_quote = Decimal("0")
    fee_details = []
    for fill in response.get("fills", []):
        commission = Decimal(str(fill.get("commission", "0")))
        asset = str(fill.get("commissionAsset", ""))
        fill_price = Decimal(str(fill.get("price", "0")))
        fee_details.append({
            "asset": asset, "commission": str(commission),
            "fill_price": str(fill_price),
        })
        if asset == quote_asset:
            fee_quote += commission
        elif asset == base_asset:
            base_delta -= commission
            fee_quote += commission * fill_price
        # Third-asset fees remain explicit in the audit response and are
        # intentionally not guessed without an independent conversion.
    return {
        "base_delta": str(base_delta),
        "quote_cashflow": str(quote_cashflow),
        "fee_quote": str(fee_quote),
        "fee_details": fee_details,
    }

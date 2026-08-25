from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, Iterable, Mapping, Optional


@dataclass(frozen=True)
class WhitelistQuoteRefresh:
    symbols: tuple[str, ...]
    prices: Dict[str, Decimal]
    errors: Dict[str, str]


def enabled_whitelist_symbols(
    rows: Iterable[Mapping[str, Any]], available_symbols: Iterable[str]
) -> tuple[str, ...]:
    """Return only enabled, directly tradable whitelist symbols."""
    available = {str(symbol).upper() for symbol in available_symbols}
    return tuple(sorted({
        str(row.get("symbol", "")).upper()
        for row in rows
        if bool(row.get("enabled")) and str(row.get("symbol", "")).upper() in available
    }))


def executable_mid(quote: Mapping[str, Any]) -> Decimal:
    try:
        bid = Decimal(str(quote.get("bidPrice", quote.get("bid", "0"))))
        ask = Decimal(str(quote.get("askPrice", quote.get("ask", "0"))))
    except (InvalidOperation, TypeError, ValueError):
        return Decimal("0")
    if bid > 0 and ask > 0:
        return (bid + ask) / Decimal("2")
    return ask if ask > 0 else bid


async def refresh_whitelist_quotes(
    *,
    client: Any,
    ledger: Any,
    available_symbols: Iterable[str],
    connector: Optional[Any] = None,
    market_data_service: Optional[Any] = None,
) -> WhitelistQuoteRefresh:
    """Fetch one Quote per enabled whitelist symbol, never the full market catalog."""
    rows = await ledger.whitelist_rows()
    symbols = enabled_whitelist_symbols(rows, available_symbols)
    prices: Dict[str, Decimal] = {}
    errors: Dict[str, str] = {}
    for symbol in symbols:
        try:
            quote = await client.quote(symbol)
            price = executable_mid(quote)
            if price <= 0:
                raise ValueError("quote has no positive bid or ask")
            prices[symbol] = price
            if connector is not None:
                connector.process_quote_event(quote)
            if market_data_service is not None:
                market_data_service.set_price(f"{symbol}-USDC", price)
        except Exception as exc:  # one bad ticker must not hide the remaining whitelist
            errors[symbol] = f"{type(exc).__name__}: {exc}"
    return WhitelistQuoteRefresh(symbols=symbols, prices=prices, errors=errors)

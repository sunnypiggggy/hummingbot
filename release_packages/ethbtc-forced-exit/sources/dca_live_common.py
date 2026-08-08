"""Shared configuration and accounting helpers for the small live DCA bots."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Dict, Iterable, Mapping, Sequence


CONNECTOR = "binance"
ACCOUNT_NAME = "binance_live_dca_200"
STRATEGY_BUDGET_QUOTE = Decimal("190")
CAPITAL_LIMIT_QUOTE = Decimal("200")
RESERVE_QUOTE = CAPITAL_LIMIT_QUOTE - STRATEGY_BUDGET_QUOTE
SINGLE_BOT_LOSS_LIMIT = Decimal("16")
COMBINED_LOSS_LIMIT = Decimal("32")
DCA_SPREADS = [Decimal("0.01"), Decimal("0.02"), Decimal("0.04"), Decimal("0.08")]
DCA_AMOUNTS = [Decimal("0.10"), Decimal("0.20"), Decimal("0.30"), Decimal("0.40")]
LIVE_TIME_LIMIT_SECONDS = 18000
LIVE_EXECUTOR_REFRESH_SECONDS = 18000


@dataclass(frozen=True)
class LivePair:
    trading_pair: str
    config_name: str
    bot_name: str
    base_asset: str


LIVE_PAIRS: Dict[str, LivePair] = {
    "BTC-USDT": LivePair("BTC-USDT", "dca_btcusdt_live_200", "dca-live-btcusdt-200", "BTC"),
    "ETH-USDT": LivePair("ETH-USDT", "dca_ethusdt_live_200", "dca-live-ethusdt-200", "ETH"),
}


def live_controller_config(pair: str) -> Dict[str, Any]:
    spec = LIVE_PAIRS[pair]
    return {
        "id": spec.config_name,
        "controller_name": "dman_maker_v3_macro",
        "controller_type": "market_making",
        "connector_name": CONNECTOR,
        "trading_pair": pair,
        "total_amount_quote": float(STRATEGY_BUDGET_QUOTE),
        "buy_spreads": [0.0],
        "sell_spreads": [0.0],
        "buy_amounts_pct": [1.0],
        "sell_amounts_pct": [1.0],
        "dca_spreads": [float(value) for value in DCA_SPREADS],
        "dca_amounts": [float(value) for value in DCA_AMOUNTS],
        # Unfilled ladders refresh after five hours. Once the first level
        # fills, the position receives its own five-hour holding deadline.
        "executor_refresh_time": LIVE_EXECUTOR_REFRESH_SECONDS,
        "cooldown_time": 15,
        "leverage": 1,
        "position_mode": "ONEWAY",
        "stop_loss": 0.05,
        "take_profit": 0.02,
        "time_limit": LIVE_TIME_LIMIT_SECONDS,
        "time_limit_from_first_fill": True,
        "stop_loss_on_partial_fills": True,
        "shutdown_retry_seconds": 1.0,
        "take_profit_order_type": "LIMIT",
        "skip_rebalance": True,
        "macro_buy_enabled": True,
        "macro_sell_enabled": True,
        "macro_decision_id": "bootstrap",
        "policy_version": "dca-macro-v3",
    }


def side_budget() -> Decimal:
    return STRATEGY_BUDGET_QUOTE / Decimal("2")


def layer_quote_amounts() -> list[Decimal]:
    return [side_budget() * weight for weight in DCA_AMOUNTS]


def required_balances(prices: Mapping[str, Decimal]) -> Dict[str, Decimal]:
    """Return the shared-account minimum balances for both live bots."""
    requirements = {"USDT": side_budget() * len(LIVE_PAIRS) + RESERVE_QUOTE * len(LIVE_PAIRS)}
    for spec in LIVE_PAIRS.values():
        requirements[spec.base_asset] = side_budget() / prices[spec.trading_pair]
    return requirements


def validate_config(config: Mapping[str, Any]) -> None:
    if config.get("connector_name") != CONNECTOR:
        raise ValueError("Live DCA must use the Binance spot connector.")
    if config.get("leverage") != 1:
        raise ValueError("Live DCA leverage must be 1.")
    if Decimal(str(config.get("total_amount_quote"))) != STRATEGY_BUDGET_QUOTE:
        raise ValueError("Live DCA strategy budget must be 190 USDT.")
    if config.get("trading_pair") not in LIVE_PAIRS:
        raise ValueError("Only BTC-USDT and ETH-USDT are permitted.")
    if "perpetual" in str(config.get("connector_name", "")):
        raise ValueError("Perpetual connectors are forbidden for this deployment.")
    if int(config.get("time_limit", 0)) != LIVE_TIME_LIMIT_SECONDS:
        raise ValueError("Live DCA time limit must remain 18000 seconds.")
    if int(config.get("executor_refresh_time", 0)) != LIVE_EXECUTOR_REFRESH_SECONDS:
        raise ValueError("Live DCA executor refresh must remain 18000 seconds.")
    if config.get("time_limit_from_first_fill") is not True:
        raise ValueError("Live DCA time limit must start at first fill.")
    if config.get("stop_loss_on_partial_fills") is not True:
        raise ValueError("Live DCA stop loss must protect partial fills.")
    if float(config.get("shutdown_retry_seconds", 0)) != 1.0:
        raise ValueError("Live DCA shutdown verification must run every second.")


def validate_exchange_filters(symbol_info: Mapping[str, Any], price: Decimal) -> None:
    filters = {entry["filterType"]: entry for entry in symbol_info.get("filters", [])}
    notional_filter = filters.get("NOTIONAL") or filters.get("MIN_NOTIONAL")
    if notional_filter is None:
        raise ValueError(f"No notional filter returned for {symbol_info.get('symbol')}.")
    minimum_notional = Decimal(str(notional_filter["minNotional"]))
    lot = filters.get("LOT_SIZE")
    if lot is None:
        raise ValueError(f"No LOT_SIZE filter returned for {symbol_info.get('symbol')}.")
    minimum_quantity = Decimal(str(lot["minQty"]))
    for amount_quote in layer_quote_amounts():
        if amount_quote < minimum_notional:
            raise ValueError(f"DCA layer {amount_quote} is below min notional {minimum_notional}.")
        if amount_quote / price < minimum_quantity:
            raise ValueError(f"DCA layer {amount_quote} is below min quantity {minimum_quantity}.")


def extract_balances(payload: Any, account_name: str = ACCOUNT_NAME,
                     connector_name: str = CONNECTOR) -> Dict[str, Decimal]:
    """Normalize the API's portfolio response across known response shapes."""
    node = payload
    if isinstance(node, Mapping) and account_name in node:
        node = node[account_name]
    if isinstance(node, Mapping) and connector_name in node:
        node = node[connector_name]
    if isinstance(node, Mapping) and "balances" in node:
        node = node["balances"]

    balances: Dict[str, Decimal] = {}
    if isinstance(node, Mapping):
        for token, value in node.items():
            if isinstance(value, Mapping):
                raw = value.get("units", value.get("total", value.get("balance", 0)))
            else:
                raw = value
            try:
                balances[str(token).upper()] = Decimal(str(raw))
            except Exception:
                continue
    elif isinstance(node, Sequence) and not isinstance(node, (str, bytes)):
        for entry in node:
            if not isinstance(entry, Mapping):
                continue
            token = entry.get("token") or entry.get("asset") or entry.get("symbol")
            raw = entry.get("units", entry.get("total", entry.get("balance", 0)))
            if token is not None:
                balances[str(token).upper()] = Decimal(str(raw))
    return balances


def trade_pnl_from_rows(rows: Iterable[Sequence[Any]], mark_price: Decimal) -> Dict[str, Decimal]:
    """Calculate bot-attributable PnL from raw SQLite TradeFill rows."""
    quote_cashflow = Decimal("0")
    net_base = Decimal("0")
    fees = Decimal("0")
    trades = 0
    scale = Decimal("1000000")
    for trade_type, raw_price, raw_amount, raw_fee, *_ in rows:
        price = Decimal(raw_price) / scale
        amount = Decimal(raw_amount) / scale
        fee = Decimal(raw_fee or 0) / scale
        notional = price * amount
        if str(trade_type).upper() == "BUY":
            quote_cashflow -= notional
            net_base += amount
        elif str(trade_type).upper() == "SELL":
            quote_cashflow += notional
            net_base -= amount
        else:
            continue
        fees += fee
        trades += 1
    pnl = quote_cashflow - fees + net_base * mark_price
    return {
        "pnl_quote": pnl,
        "quote_cashflow": quote_cashflow,
        "net_base": net_base,
        "fees_quote": fees,
        "trades": Decimal(trades),
    }

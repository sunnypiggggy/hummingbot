from __future__ import annotations

from copy import deepcopy
from decimal import Decimal
from typing import Any, Dict, Optional


TRADE_SIDE_VALUES = {"BUY": 1, "SELL": 2, "1": 1, "2": 2}
ORDER_TYPE_VALUES = {"MARKET": 1, "LIMIT": 2, "1": 1, "2": 2}


def _name(value: Any) -> str:
    return str(getattr(value, "name", getattr(value, "value", value))).upper()


def trade_side_name(value: Any) -> str:
    name = _name(value)
    if name in {"1", "BUY"}:
        return "BUY"
    if name in {"2", "SELL"}:
        return "SELL"
    return name


def order_type_name(value: Any) -> str:
    name = _name(value)
    if name in {"1", "MARKET"}:
        return "MARKET"
    if name in {"2", "LIMIT"}:
        return "LIMIT"
    return name


def normalize_executor_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Convert the public human-readable request into Hummingbot enum values.

    The managed ledger stores this normalized form, so process recovery uses
    exactly the same configuration as the original executor creation.
    """
    value = deepcopy(config)
    side = trade_side_name(value.get("side"))
    if side not in {"BUY", "SELL"}:
        raise ValueError("side must be BUY or SELL")
    value["side"] = TRADE_SIDE_VALUES[side]
    executor_type = str(value.get("type", ""))
    if executor_type == "position_executor":
        barriers = dict(value.get("triple_barrier_config") or {})
        for field, default in (
            ("open_order_type", "LIMIT"),
            ("take_profit_order_type", "LIMIT"),
            ("stop_loss_order_type", "MARKET"),
            ("time_limit_order_type", "MARKET"),
        ):
            order_name = order_type_name(barriers.get(field, default))
            if order_name not in ORDER_TYPE_VALUES:
                raise ValueError(f"{field} must be LIMIT or MARKET")
            barriers[field] = ORDER_TYPE_VALUES[order_name]
        value["triple_barrier_config"] = barriers
    elif executor_type == "order_executor":
        strategy = order_type_name(value.get("execution_strategy"))
        if strategy not in {"LIMIT", "MARKET"}:
            raise ValueError("execution_strategy must be LIMIT or MARKET")
        value["execution_strategy"] = strategy
    return value


def build_order_executor_config(
    *, executor_id: str, symbol: str, side: str, amount: Decimal,
    order_type: str, price: Optional[Decimal], source_owner: Optional[str] = None,
) -> Dict[str, Any]:
    config: Dict[str, Any] = {
        "id": executor_id,
        "type": "order_executor",
        "connector_name": "binance_stocks",
        "trading_pair": f"{symbol.upper()}-USDC",
        "side": side.upper(),
        "amount": str(amount),
        "execution_strategy": order_type.upper(),
        "position_action": "OPEN" if side.upper() == "BUY" else "CLOSE",
    }
    if price is not None:
        config["price"] = str(price)
    if source_owner:
        config["managed_source_owner"] = source_owner
    return config


def build_position_executor_config(
    *, executor_id: str, symbol: str, amount: Decimal, entry_order_type: str,
    entry_price: Optional[Decimal], stop_loss: Decimal, time_limit: int,
    take_profit: Optional[Decimal], trailing_activation: Optional[Decimal],
    trailing_delta: Optional[Decimal],
) -> Dict[str, Any]:
    barriers: Dict[str, Any] = {
        "stop_loss": str(stop_loss),
        "time_limit": time_limit,
        "open_order_type": entry_order_type.upper(),
        "take_profit_order_type": "LIMIT",
        "stop_loss_order_type": "MARKET",
        "time_limit_order_type": "MARKET",
    }
    if take_profit is not None:
        barriers["take_profit"] = str(take_profit)
    if trailing_activation is not None or trailing_delta is not None:
        if trailing_activation is None or trailing_delta is None:
            raise ValueError("trailing_activation and trailing_delta must be supplied together")
        barriers["trailing_stop"] = {
            "activation_price": str(trailing_activation),
            "trailing_delta": str(trailing_delta),
        }
    config: Dict[str, Any] = {
        "id": executor_id,
        "type": "position_executor",
        "connector_name": "binance_stocks",
        "trading_pair": f"{symbol.upper()}-USDC",
        "side": "BUY",
        "amount": str(amount),
        "triple_barrier_config": barriers,
    }
    if entry_price is not None:
        config["entry_price"] = str(entry_price)
    return config

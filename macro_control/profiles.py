from __future__ import annotations

from copy import deepcopy

from .policy import POLICY_VERSION

MAX_TOTAL_AMOUNT_QUOTE = 190.0
LIVE_TIME_LIMIT_SECONDS = 18000
LIVE_EXECUTOR_REFRESH_SECONDS = 18000

LIVE_PROFILE = {
    "total_amount_quote": 190.0,
    "dca_spreads": [0.01, 0.02, 0.04, 0.08],
    "dca_amounts": [0.10, 0.20, 0.30, 0.40],
    "take_profit": 0.02,
    "stop_loss": 0.05,
    "time_limit": LIVE_TIME_LIMIT_SECONDS,
    "executor_refresh_time": LIVE_EXECUTOR_REFRESH_SECONDS,
    "time_limit_from_first_fill": True,
    "stop_loss_on_partial_fills": True,
}


def event_config(
    trading_pair: str,
    *,
    macro_buy_enabled: bool,
    macro_sell_enabled: bool = True,
    decision_id: str,
    policy_version: str = POLICY_VERSION,
) -> dict:
    if trading_pair not in {"BTC-USDT", "ETH-USDT"}:
        raise ValueError("macro DCA is restricted to BTC-USDT and ETH-USDT")
    profile = deepcopy(LIVE_PROFILE)
    profile.update(
        {
            "controller_name": "dman_maker_v3_macro",
            "connector_name": "binance",
            "trading_pair": trading_pair,
            "leverage": 1,
            "macro_buy_enabled": macro_buy_enabled,
            "macro_sell_enabled": macro_sell_enabled,
            "macro_decision_id": decision_id,
            "policy_version": policy_version,
        }
    )
    validate_profile(profile)
    return profile


def validate_profile(profile: dict) -> None:
    if not isinstance(profile.get("macro_buy_enabled"), bool):
        raise ValueError("macro_buy_enabled must be a boolean")
    if not isinstance(profile.get("macro_sell_enabled"), bool):
        raise ValueError("macro_sell_enabled must be a boolean")
    if profile.get("policy_version") != POLICY_VERSION:
        raise ValueError("invalid policy version")
    if profile.get("connector_name") != "binance" or int(profile.get("leverage", 0)) != 1:
        raise ValueError("only Binance spot with leverage=1 is supported")
    if profile.get("trading_pair") not in {"BTC-USDT", "ETH-USDT"}:
        raise ValueError("unexpected trading pair")
    total = float(profile["total_amount_quote"])
    if total <= 0 or total > MAX_TOTAL_AMOUNT_QUOTE:
        raise ValueError("profile attempts to exceed the 190 USDT hard limit")
    spreads = [float(value) for value in profile["dca_spreads"]]
    amounts = [float(value) for value in profile["dca_amounts"]]
    if len(spreads) != 4 or sorted(spreads) != spreads or any(value <= 0 for value in spreads):
        raise ValueError("DCA spreads must be four ascending positive values")
    if len(amounts) != 4 or abs(sum(amounts) - 1.0) > 1e-9:
        raise ValueError("DCA amount weights must contain four values totaling 1")
    if float(profile["stop_loss"]) != 0.05:
        raise ValueError("the 5% controller stop loss is immutable")
    if int(profile["time_limit"]) != LIVE_TIME_LIMIT_SECONDS:
        raise ValueError("the 18000 second time limit is immutable")
    if int(profile["executor_refresh_time"]) != LIVE_EXECUTOR_REFRESH_SECONDS:
        raise ValueError("the 18000 second executor refresh is immutable")
    if profile.get("time_limit_from_first_fill") is not True:
        raise ValueError("the position time limit must start at first fill")
    if profile.get("stop_loss_on_partial_fills") is not True:
        raise ValueError("partial fills must remain protected by stop loss")

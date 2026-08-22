from decimal import Decimal
from typing import Any, Dict

from pydantic import ConfigDict, Field, SecretStr, field_validator

from hummingbot.client.config.config_data_types import BaseConnectorConfigMap
from hummingbot.core.data_type.trade_fee import TradeFeeSchema

from . import binance_stocks_constants as CONSTANTS

CENTRALIZED = True
EXAMPLE_PAIR = "AAPL-USDC"
DEFAULT_FEES = TradeFeeSchema(
    maker_percent_fee_decimal=Decimal("0"),
    taker_percent_fee_decimal=Decimal("0"),
)


def extract_payload(response: Any) -> Any:
    """Unwrap the response envelopes used by different Stocks API endpoints."""
    if not isinstance(response, dict):
        return response
    for key in ("data", "result"):
        if key in response and response[key] is not None:
            return response[key]
    return response


def is_exchange_information_valid(exchange_info: Dict[str, Any]) -> bool:
    symbol = str(exchange_info.get("symbol", "")).strip()
    if not symbol or exchange_info.get("tokenized") is True:
        return False
    status = str(exchange_info.get("status", exchange_info.get("tradingStatus", "TRADING"))).upper()
    return status not in {"DELISTED", "INACTIVE", "UNAVAILABLE"}


class BinanceStocksConfigMap(BaseConnectorConfigMap):
    connector: str = CONSTANTS.EXCHANGE_NAME
    binance_stocks_api_key: SecretStr = Field(
        default=...,
        json_schema_extra={
            "prompt": lambda cm: "Enter your Binance Stocks API key",
            "is_secure": True,
            "is_connect_key": True,
            "prompt_on_new": True,
        },
    )
    binance_stocks_api_secret: SecretStr = Field(
        default=...,
        json_schema_extra={
            "prompt": lambda cm: "Enter your Binance Stocks HMAC API secret",
            "is_secure": True,
            "is_connect_key": True,
            "prompt_on_new": True,
        },
    )
    quote_asset: str = CONSTANTS.DEFAULT_QUOTE_ASSET
    wallet_type: str = CONSTANTS.DEFAULT_WALLET_TYPE
    trading_session: str = CONSTANTS.DEFAULT_TRADING_SESSION
    time_in_force: str = CONSTANTS.DEFAULT_TIME_IN_FORCE
    disclaimer_confirmed: bool = False

    @field_validator("quote_asset")
    @classmethod
    def validate_quote_asset(cls, value: str) -> str:
        value = value.upper()
        if value != CONSTANTS.DEFAULT_QUOTE_ASSET:
            raise ValueError("binance_stocks v1 supports USDC quote only")
        return value

    @field_validator("wallet_type")
    @classmethod
    def validate_wallet_type(cls, value: str) -> str:
        value = value.upper()
        if value != CONSTANTS.DEFAULT_WALLET_TYPE:
            raise ValueError("binance_stocks v1 requires CARD Funding Wallet")
        return value

    @field_validator("trading_session")
    @classmethod
    def validate_trading_session(cls, value: str) -> str:
        value = value.upper()
        if value not in CONSTANTS.SUPPORTED_TRADING_SESSIONS:
            raise ValueError("binance_stocks v1 supports EXTENDED limit orders only")
        return value

    @field_validator("time_in_force")
    @classmethod
    def validate_time_in_force(cls, value: str) -> str:
        value = value.upper()
        if value not in CONSTANTS.SUPPORTED_TIME_IN_FORCE:
            raise ValueError("binance_stocks v1 supports DAY limit orders only")
        return value

    model_config = ConfigDict(title=CONSTANTS.EXCHANGE_NAME)


KEYS = BinanceStocksConfigMap.model_construct()

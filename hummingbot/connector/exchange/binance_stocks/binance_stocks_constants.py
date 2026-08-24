from hummingbot.core.api_throttler.data_types import LinkedLimitWeightPair, RateLimit
from hummingbot.core.data_type.in_flight_order import OrderState

EXCHANGE_NAME = "binance_stocks"
DEFAULT_DOMAIN = "com"
REST_URL = "https://api.binance.com"
WS_URL = "wss://nbstream.binance.com/equity"

EXCHANGE_INFO_PATH_URL = "/sapi/v1/equity/market/exchangeInfo"
QUOTE_PATH_URL = "/sapi/v1/equity/market/quote"
ORDER_PLACE_PATH_URL = "/sapi/v1/equity/order/place"
ORDER_CANCEL_PATH_URL = "/sapi/v1/equity/order/cancel"
OPEN_ORDERS_PATH_URL = "/sapi/v1/equity/order/open-orders"
ORDER_HISTORY_PATH_URL = "/sapi/v1/equity/order/history"
ORDER_DETAIL_PATH_URL = "/sapi/v1/equity/order/detail"
TRADE_HISTORY_PATH_URL = "/sapi/v1/equity/trade/history"
LISTEN_KEY_PATH_URL = "/sapi/v1/equity/listenKey"
SERVER_TIME_PATH_URL = "/api/v3/time"
MARKET_DATA_PATHS = {EXCHANGE_INFO_PATH_URL, QUOTE_PATH_URL}

HBOT_ORDER_ID_PREFIX = "x-HBSTK"
MAX_ORDER_ID_LEN = 32

DEFAULT_QUOTE_ASSET = "USDC"
DEFAULT_WALLET_TYPE = "CARD"
DEFAULT_TRADING_SESSION = "EXTENDED"
DEFAULT_TIME_IN_FORCE = "DAY"
SUPPORTED_TRADING_SESSIONS = {"EXTENDED"}
SUPPORTED_TIME_IN_FORCE = {"DAY"}

QUOTE_STALE_SECONDS = 10.0
POSITION_SNAPSHOT_STALE_SECONDS = 30.0
LISTEN_KEY_KEEP_ALIVE_SECONDS = 45 * 60
WS_HEARTBEAT_SECONDS = 30.0

MARKET_PHASES = {
    "PRE_MARKET",
    "MARKET_OPEN",
    "POST_MARKET",
    "OVERNIGHT",
    "MARKET_CLOSED",
}
EXTENDED_OPEN_PHASES = {"PRE_MARKET", "MARKET_OPEN", "POST_MARKET"}
RTH_OPEN_PHASES = {"MARKET_OPEN"}

ORDER_STATE = {
    "NEW": OrderState.OPEN,
    "ACCEPTED": OrderState.OPEN,
    # The local PAPER venue uses the normalized lifecycle vocabulary while
    # Binance orderReport uses NEW/ACCEPTED. Both represent the same state.
    "OPEN": OrderState.OPEN,
    "PARTIALLY_FILLED": OrderState.PARTIALLY_FILLED,
    "FILLED": OrderState.FILLED,
    "CANCELED": OrderState.CANCELED,
    "CANCELLED": OrderState.CANCELED,
    "EXPIRED": OrderState.FAILED,
    "REJECTED": OrderState.FAILED,
}

REQUEST_WEIGHT = "BINANCE_STOCKS_REQUEST_WEIGHT"
ORDER_PLACE = "BINANCE_STOCKS_ORDER_PLACE"
ONE_MINUTE = 60

_PUBLIC_LIMIT = 1200
_PRIVATE_LIMIT = 1200

RATE_LIMITS = [
    RateLimit(limit_id=REQUEST_WEIGHT, limit=1200, time_interval=ONE_MINUTE),
    RateLimit(limit_id=ORDER_PLACE, limit=200, time_interval=ONE_MINUTE),
    RateLimit(
        limit_id=EXCHANGE_INFO_PATH_URL,
        limit=_PUBLIC_LIMIT,
        time_interval=ONE_MINUTE,
        linked_limits=[LinkedLimitWeightPair(REQUEST_WEIGHT, 1)],
    ),
    RateLimit(
        limit_id=QUOTE_PATH_URL,
        limit=_PUBLIC_LIMIT,
        time_interval=ONE_MINUTE,
        linked_limits=[LinkedLimitWeightPair(REQUEST_WEIGHT, 1)],
    ),
    RateLimit(
        limit_id=SERVER_TIME_PATH_URL,
        limit=_PUBLIC_LIMIT,
        time_interval=ONE_MINUTE,
        linked_limits=[LinkedLimitWeightPair(REQUEST_WEIGHT, 1)],
    ),
    RateLimit(
        limit_id=ORDER_PLACE_PATH_URL,
        limit=200,
        time_interval=ONE_MINUTE,
        linked_limits=[LinkedLimitWeightPair(REQUEST_WEIGHT, 1), LinkedLimitWeightPair(ORDER_PLACE, 1)],
    ),
    RateLimit(
        limit_id=ORDER_CANCEL_PATH_URL,
        limit=_PRIVATE_LIMIT,
        time_interval=ONE_MINUTE,
        linked_limits=[LinkedLimitWeightPair(REQUEST_WEIGHT, 1)],
    ),
    RateLimit(
        limit_id=OPEN_ORDERS_PATH_URL,
        limit=_PRIVATE_LIMIT,
        time_interval=ONE_MINUTE,
        linked_limits=[LinkedLimitWeightPair(REQUEST_WEIGHT, 1)],
    ),
    RateLimit(
        limit_id=ORDER_HISTORY_PATH_URL,
        limit=_PRIVATE_LIMIT,
        time_interval=ONE_MINUTE,
        linked_limits=[LinkedLimitWeightPair(REQUEST_WEIGHT, 1)],
    ),
    RateLimit(
        limit_id=ORDER_DETAIL_PATH_URL,
        limit=_PRIVATE_LIMIT,
        time_interval=ONE_MINUTE,
        linked_limits=[LinkedLimitWeightPair(REQUEST_WEIGHT, 1)],
    ),
    RateLimit(
        limit_id=TRADE_HISTORY_PATH_URL,
        limit=_PRIVATE_LIMIT,
        time_interval=ONE_MINUTE,
        linked_limits=[LinkedLimitWeightPair(REQUEST_WEIGHT, 1)],
    ),
    RateLimit(
        limit_id=LISTEN_KEY_PATH_URL,
        limit=_PRIVATE_LIMIT,
        time_interval=ONE_MINUTE,
        linked_limits=[LinkedLimitWeightPair(REQUEST_WEIGHT, 1)],
    ),
]

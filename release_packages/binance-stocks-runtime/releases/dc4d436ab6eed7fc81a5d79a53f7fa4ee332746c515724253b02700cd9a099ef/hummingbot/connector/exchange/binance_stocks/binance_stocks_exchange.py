import asyncio
import math
from decimal import ROUND_UP, Decimal
from typing import Any, Dict, List, Optional, Tuple

from bidict import bidict

from hummingbot.connector.constants import s_decimal_NaN
from hummingbot.connector.exchange_py_base import ExchangePyBase
from hummingbot.connector.trading_rule import TradingRule
from hummingbot.connector.utils import combine_to_hb_trading_pair, split_hb_trading_pair
from hummingbot.core.data_type.common import OrderType, TradeType
from hummingbot.core.data_type.in_flight_order import InFlightOrder, OrderUpdate, TradeUpdate
from hummingbot.core.data_type.order_book_tracker_data_source import OrderBookTrackerDataSource
from hummingbot.core.data_type.trade_fee import (
    AddedToCostTradeFee,
    DeductedFromReturnsTradeFee,
    TokenAmount,
    TradeFeeBase,
)
from hummingbot.core.data_type.user_stream_tracker_data_source import UserStreamTrackerDataSource
from hummingbot.core.web_assistant.web_assistants_factory import WebAssistantsFactory

from . import (
    binance_stocks_constants as CONSTANTS,
    binance_stocks_utils as utils,
    binance_stocks_web_utils as web_utils,
)
from .binance_stocks_api_order_book_data_source import BinanceStocksAPIOrderBookDataSource
from .binance_stocks_api_user_stream_data_source import BinanceStocksAPIUserStreamDataSource
from .binance_stocks_auth import BinanceStocksAuth
from .binance_stocks_position_provider import (
    EquityAccountSnapshot,
    EquityPositionProvider,
    ManagedLedgerEquityPositionProvider,
    UnavailableEquityPositionProvider,
)


class BinanceStocksExchange(ExchangePyBase):
    """Direct US equity connector for Binance Stocks Trading.

    Live order placement is deliberately unavailable with the default position
    provider.  Binance currently exposes order history but no authoritative
    equity-position snapshot, and order history must not be used to infer a
    sellable position.
    """

    web_utils = web_utils

    def __init__(
        self,
        binance_stocks_api_key: str,
        binance_stocks_api_secret: str,
        quote_asset: str = CONSTANTS.DEFAULT_QUOTE_ASSET,
        wallet_type: str = CONSTANTS.DEFAULT_WALLET_TYPE,
        trading_session: str = CONSTANTS.DEFAULT_TRADING_SESSION,
        time_in_force: str = CONSTANTS.DEFAULT_TIME_IN_FORCE,
        disclaimer_confirmed: bool = False,
        balance_asset_limit: Optional[Dict[str, Dict[str, Decimal]]] = None,
        rate_limits_share_pct: Decimal = Decimal("100"),
        trading_pairs: Optional[List[str]] = None,
        trading_required: bool = True,
        domain: str = CONSTANTS.DEFAULT_DOMAIN,
        position_provider: Optional[EquityPositionProvider] = None,
    ):
        self.api_key = binance_stocks_api_key
        self.secret_key = binance_stocks_api_secret
        self._domain = domain
        self._trading_pairs = trading_pairs or []
        self._trading_required = trading_required
        self._quote_asset = quote_asset.upper()
        self._wallet_type = wallet_type.upper()
        self._trading_session = trading_session.upper()
        self._time_in_force = time_in_force.upper()
        self._disclaimer_confirmed = bool(disclaimer_confirmed)
        self._validate_static_configuration()

        self._position_provider = position_provider or UnavailableEquityPositionProvider()
        self._position_reconciliation_ready = False
        self._last_position_snapshot: Optional[EquityAccountSnapshot] = None
        self._account_authorized = False
        self._market_phase = "UNKNOWN"
        self._market_phase_source = "UNKNOWN"
        self._market_event_timestamp = 0.0
        self._market_trading_date: Optional[str] = None
        self._market_valid_until = 0.0
        self._market_state_conflict = False
        self._last_quote_timestamp: Dict[str, float] = {}
        self._last_quote: Dict[str, Tuple[Decimal, Decimal]] = {}
        self._trading_status: Dict[str, str] = {}
        self._tradability: Dict[str, str] = {}
        self._fractional_supported: Dict[str, bool] = {}
        super().__init__(balance_asset_limit, rate_limits_share_pct)

    def _validate_static_configuration(self):
        if self._quote_asset != CONSTANTS.DEFAULT_QUOTE_ASSET:
            raise ValueError("binance_stocks v1 only supports USDC quote pairs")
        if self._wallet_type != CONSTANTS.DEFAULT_WALLET_TYPE:
            raise ValueError("binance_stocks v1 requires CARD Funding Wallet")
        if self._trading_session not in CONSTANTS.SUPPORTED_TRADING_SESSIONS:
            raise ValueError("binance_stocks v1 only supports EXTENDED limit orders")
        if self._time_in_force not in CONSTANTS.SUPPORTED_TIME_IN_FORCE:
            raise ValueError("binance_stocks v1 only supports DAY limit orders")
        for trading_pair in self._trading_pairs:
            _, quote = split_hb_trading_pair(trading_pair)
            if quote != self._quote_asset:
                raise ValueError(f"Unsupported Binance Stocks quote asset in {trading_pair}")

    @property
    def authenticator(self) -> BinanceStocksAuth:
        return BinanceStocksAuth(self.api_key, self.secret_key, self._time_synchronizer)

    @property
    def name(self) -> str:
        return CONSTANTS.EXCHANGE_NAME

    @property
    def rate_limits_rules(self):
        return CONSTANTS.RATE_LIMITS

    @property
    def domain(self):
        return self._domain

    @property
    def client_order_id_max_length(self):
        return CONSTANTS.MAX_ORDER_ID_LEN

    @property
    def client_order_id_prefix(self):
        return CONSTANTS.HBOT_ORDER_ID_PREFIX

    @property
    def trading_rules_request_path(self):
        return CONSTANTS.EXCHANGE_INFO_PATH_URL

    @property
    def trading_pairs_request_path(self):
        return CONSTANTS.EXCHANGE_INFO_PATH_URL

    @property
    def check_network_request_path(self):
        return CONSTANTS.EXCHANGE_INFO_PATH_URL

    @property
    def trading_pairs(self):
        return self._trading_pairs

    @property
    def is_cancel_request_in_exchange_synchronous(self) -> bool:
        # S/F is only an API request acknowledgement. orderReport/detail owns state.
        return False

    @property
    def is_trading_required(self) -> bool:
        return self._trading_required

    @property
    def status_dict(self) -> Dict[str, bool]:
        status = super().status_dict
        if self.is_trading_required:
            status.update(
                {
                    "account_eligible": self._account_authorized,
                    "disclaimer_confirmed": self._disclaimer_confirmed,
                    "market_state_initialized": self._market_phase != "UNKNOWN",
                    "quotes_fresh_or_market_closed": self._quotes_are_ready(),
                    "position_reconciliation_ready": self._position_reconciliation_ready,
                }
            )
        return status

    @property
    def market_data_headers(self) -> Dict[str, str]:
        return {"X-MBX-APIKEY": self.api_key} if self.api_key else {}

    @property
    def market_phase(self) -> str:
        return self._market_phase

    @property
    def market_state_metadata(self) -> Dict[str, Any]:
        return {
            "phase": self._market_phase,
            "source": self._market_phase_source,
            "event_timestamp": self._market_event_timestamp,
            "valid_until": self._market_valid_until,
            "trading_date": self._market_trading_date,
            "conflict": self._market_state_conflict,
        }

    def restore_market_state(
        self, phase: str, *, source: str, event_timestamp: float,
        valid_until: float, trading_date: Optional[str], conflict: bool = False,
    ) -> None:
        phase = str(phase).upper()
        if phase in CONSTANTS.MARKET_PHASES:
            self._market_phase = phase
            self._market_phase_source = str(source).upper()
            self._market_event_timestamp = float(event_timestamp)
            self._market_valid_until = float(valid_until)
            self._market_trading_date = trading_date
            self._market_state_conflict = bool(conflict)

    def latest_quote(self, symbol: str) -> Optional[Tuple[Decimal, Decimal]]:
        """Return the current one-level quote only when it is still executable."""
        symbol = symbol.upper()
        if not self._quote_is_fresh(symbol):
            return None
        return self._last_quote.get(symbol)

    def supported_order_types(self):
        return [OrderType.LIMIT, OrderType.MARKET]

    def set_account_authorized(self, authorized: bool):
        self._account_authorized = authorized

    def process_quote_event(self, event: Dict[str, Any]):
        symbol = str(event.get("s", event.get("symbol", ""))).upper()
        if not symbol:
            return
        bid = self._decimal(event.get("bp", event.get("bidPrice")))
        ask = self._decimal(event.get("ap", event.get("askPrice")))
        event_time = self._seconds(event.get("T", event.get("E")))
        self._last_quote_timestamp[symbol] = event_time or self._clock_time()
        if bid > 0 and ask > 0:
            self._last_quote[symbol] = (bid, ask)
            # Hummingbot asks for fees in both base and quote terms when it
            # builds Executor metrics. Stocks fees are always USDC, so expose
            # the executable conversion rate and avoid falling back to an
            # unrelated crypto rate source.
            from hummingbot.core.rate_oracle.rate_oracle import RateOracle
            RateOracle.get_instance().set_price(
                combine_to_hb_trading_pair(symbol, self._quote_asset), (bid + ask) / Decimal("2")
            )

    def process_market_state_event(self, event: Dict[str, Any]):
        event_type = str(event.get("e", event.get("eventType", "")))
        symbol = str(event.get("s", event.get("symbol", ""))).upper()
        if event_type == "calendar":
            # The official transition-only Calendar stream names the new phase
            # `to`.  Keep legacy aliases for scenario fixtures and old captures.
            phase = str(event.get(
                "to", event.get("phase", event.get("marketPhase", event.get("status", "UNKNOWN")))
            )).upper()
            if phase in CONSTANTS.MARKET_PHASES:
                self._market_phase = phase
                self._market_phase_source = "BINANCE"
                self._market_event_timestamp = self._seconds(event.get("ts", event.get("T", event.get("E")))) or self._clock_time()
                # A transition must be reconciled against XNYS before an
                # asynchronous order is activated.
                self._market_valid_until = 0.0
                value = event.get("tradingDate", event.get("trading_date"))
                # Calendar is transition-only and may omit tradingDate.  Do
                # not retain a previous session's date; the runtime reconciler
                # binds this fresh transition to the current XNYS date.
                self._market_trading_date = str(value) if value else None
                self._market_state_conflict = False
        elif event_type == "tradingStatus" and symbol:
            self._trading_status[symbol] = str(event.get("status", event.get("tradingStatus", "UNKNOWN"))).upper()
        elif event_type == "tradability" and symbol:
            value = event.get("tradability", event.get("direction", event.get("status", "NONE")))
            self._tradability[symbol] = str(value).upper()

    def _quotes_are_ready(self) -> bool:
        if self._market_phase == "MARKET_CLOSED":
            return True
        return all(self._quote_is_fresh(pair.split("-")[0]) for pair in self._trading_pairs)

    def _quote_is_fresh(self, symbol: str) -> bool:
        timestamp = self._last_quote_timestamp.get(symbol.upper(), 0)
        return timestamp > 0 and self._clock_time() - timestamp <= CONSTANTS.QUOTE_STALE_SECONDS

    def _create_web_assistants_factory(self) -> WebAssistantsFactory:
        return web_utils.build_api_factory(
            throttler=self._throttler,
            time_synchronizer=self._time_synchronizer,
            domain=self._domain,
            auth=self._auth,
        )

    async def _api_request(self, path_url, headers: Optional[Dict[str, Any]] = None, **kwargs):
        if path_url in CONSTANTS.MARKET_DATA_PATHS:
            headers = dict(headers or {})
            headers.update(self.market_data_headers)
        return await super()._api_request(path_url=path_url, headers=headers, **kwargs)

    def _create_order_book_data_source(self) -> OrderBookTrackerDataSource:
        return BinanceStocksAPIOrderBookDataSource(
            trading_pairs=self._trading_pairs,
            connector=self,
            api_factory=self._web_assistants_factory,
            domain=self._domain,
        )

    def _create_user_stream_data_source(self) -> UserStreamTrackerDataSource:
        return BinanceStocksAPIUserStreamDataSource(
            auth=self._auth,
            trading_pairs=self._trading_pairs,
            connector=self,
            api_factory=self._web_assistants_factory,
            domain=self._domain,
        )

    def _is_request_exception_related_to_time_synchronizer(self, request_exception: Exception):
        text = str(request_exception)
        return "-1021" in text or "timestamp" in text.lower() and "recvwindow" in text.lower()

    def _is_order_not_found_during_status_update_error(self, status_update_exception: Exception) -> bool:
        text = str(status_update_exception).lower()
        return "order" in text and ("not found" in text or "does not exist" in text)

    def _is_order_not_found_during_cancelation_error(self, cancelation_exception: Exception) -> bool:
        return self._is_order_not_found_during_status_update_error(cancelation_exception)

    def _get_fee(
        self,
        base_currency: str,
        quote_currency: str,
        order_type: OrderType,
        order_side: TradeType,
        amount: Decimal,
        price: Decimal = s_decimal_NaN,
        is_maker: Optional[bool] = None,
    ) -> TradeFeeBase:
        notional = amount * price if not price.is_nan() else Decimal("0")
        fee_class = AddedToCostTradeFee if order_side is TradeType.BUY else DeductedFromReturnsTradeFee
        if notional > Decimal("350"):
            return fee_class(percent=Decimal("0.001"))
        return fee_class(flat_fees=[TokenAmount(token=self._quote_asset, amount=Decimal("0.35"))])

    async def _place_order(
        self,
        order_id: str,
        trading_pair: str,
        amount: Decimal,
        trade_type: TradeType,
        order_type: OrderType,
        price: Decimal,
        **kwargs,
    ) -> Tuple[str, float]:
        symbol = await self.exchange_symbol_associated_to_pair(trading_pair=trading_pair)
        executor_id = self._runtime_executor_id()
        self._assert_order_can_be_placed(
            symbol, amount, trade_type, order_type, price, managed_executor_id=executor_id,
        )
        if hasattr(self._position_provider, "register_order"):
            await self._position_provider.register_order(
                client_order_id=order_id,
                executor_id=executor_id,
                symbol=symbol,
                side=trade_type.name,
                requested_base=amount,
                order_type=order_type.name,
            )
        payload: Dict[str, Any] = {
            "symbol": symbol,
            "side": trade_type.name,
            "orderType": order_type.name,
            "clientOrderId": order_id,
            "quoteAsset": self._quote_asset,
        }
        # Binance Stocks MARKET BUY is not quantity based.  The exchange API
        # requires quote notional, while Hummingbot executors express their
        # target in base units.  Convert with the same fresh ask used by the
        # pre-trade budget check and round up to cents so the requested base
        # quantity is not silently under-funded.
        if trade_type is TradeType.BUY and order_type is OrderType.MARKET:
            _bid, ask = self._last_quote[symbol]
            notional = (amount * ask).quantize(Decimal("0.01"), rounding=ROUND_UP)
            payload["notional"] = f"{notional:f}"
        else:
            payload["quantity"] = f"{amount:f}"
        if trade_type is TradeType.BUY:
            payload["walletType"] = self._wallet_type
        if order_type is OrderType.LIMIT:
            payload.update(
                {
                    "price": f"{price:f}",
                    "tradingSession": self._trading_session,
                    "timeInForce": self._time_in_force,
                }
            )
        try:
            response = await self._api_post(
                path_url=CONSTANTS.ORDER_PLACE_PATH_URL,
                data=payload,
                is_auth_required=True,
            )
            result = self._successful_ack_payload(response, "place")
        except IOError as original_error:
            # Request outcome is unknown. Query the deterministic client ID; do not repost.
            try:
                result = await self._request_order_detail(client_order_id=order_id)
            except Exception:
                raise original_error
        exchange_order_id = result.get("orderId", result.get("id"))
        if exchange_order_id is None:
            raise IOError(f"Binance Stocks place acknowledgement lacks orderId: {result}")
        if hasattr(self._position_provider, "bind_exchange_order"):
            await self._position_provider.bind_exchange_order(order_id, str(exchange_order_id))
        event_time = self._seconds(result.get("transactTime", result.get("time"))) or self._clock_time()
        return str(exchange_order_id), event_time

    async def _place_cancel(self, order_id: str, tracked_order: InFlightOrder):
        exchange_order_id = await tracked_order.get_exchange_order_id()
        response = await self._api_post(
            path_url=CONSTANTS.ORDER_CANCEL_PATH_URL,
            data={"orderId": exchange_order_id},
            is_auth_required=True,
        )
        self._successful_ack_payload(response, "cancel")
        return True

    def _assert_order_can_be_placed(
        self,
        symbol: str,
        amount: Decimal,
        side: TradeType,
        order_type: OrderType,
        price: Decimal,
        managed_executor_id: Optional[str] = None,
    ):
        if order_type not in self.supported_order_types():
            raise ValueError("Binance Stocks v1 supports LIMIT and MARKET only; LIMIT_MAKER is forbidden")
        if not self._disclaimer_confirmed:
            raise PermissionError("Binance Stocks disclaimer is not confirmed")
        if not self._account_authorized:
            raise PermissionError("Binance Stocks API/account eligibility is not verified")
        if not self._position_reconciliation_ready or self._last_position_snapshot is None:
            raise PermissionError("Authoritative Binance Stocks position snapshot is unavailable; order blocked")
        if not self._quote_is_fresh(symbol):
            raise PermissionError(f"Binance Stocks quote for {symbol} is stale")
        if order_type is OrderType.MARKET and self._market_phase not in CONSTANTS.RTH_OPEN_PHASES:
            raise PermissionError("Binance Stocks MARKET orders are allowed only during MARKET_OPEN")
        if order_type is OrderType.LIMIT and self._market_phase not in CONSTANTS.EXTENDED_OPEN_PHASES:
            raise PermissionError("Binance Stocks EXTENDED limit order requested outside an eligible session")
        status = self._trading_status.get(symbol, "UNKNOWN")
        if status not in {"TRADING", "ACTIVE", "NORMAL"}:
            raise PermissionError(f"Binance Stocks {symbol} trading status is {status}")
        tradability = self._tradability.get(symbol, "NONE")
        allowed = {"BOTH", "BUY_SELL", "ALL", side.name}
        if tradability not in allowed:
            raise PermissionError(f"Binance Stocks {symbol} does not currently allow {side.name}")
        if not self._fractional_supported.get(symbol, False) and amount != amount.to_integral_value():
            raise ValueError(f"Binance Stocks {symbol} does not support fractional quantity")
        snapshot = self._last_position_snapshot
        if side is TradeType.SELL:
            available = snapshot.positions.get(symbol)
            if available is None or available.available < amount:
                trusted_reservation = bool(
                    managed_executor_id
                    and isinstance(self._position_provider, ManagedLedgerEquityPositionProvider)
                )
                if not trusted_reservation:
                    raise PermissionError(f"Authoritative available {symbol} position is insufficient")
        else:
            effective_price = price
            if effective_price.is_nan() or effective_price <= 0:
                bid, ask = self._last_quote[symbol]
                effective_price = ask
            estimated_fee = (
                Decimal("0.35") if amount * effective_price <= 350 else amount * effective_price * Decimal("0.001")
            )
            if snapshot.quote_available < amount * effective_price + estimated_fee:
                raise PermissionError(f"Authoritative available {self._quote_asset} balance is insufficient")

    @staticmethod
    def _successful_ack_payload(response: Any, operation: str) -> Dict[str, Any]:
        if isinstance(response, dict) and str(response.get("status", "S")).upper() == "F":
            raise IOError(f"Binance Stocks {operation} request failed: {response}")
        result = utils.extract_payload(response)
        if not isinstance(result, dict):
            raise IOError(f"Unexpected Binance Stocks {operation} response: {response}")
        return result

    async def _request_order_detail(
        self,
        tracked_order: Optional[InFlightOrder] = None,
        client_order_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        params: Dict[str, Any] = {}
        if tracked_order is not None and tracked_order.exchange_order_id is not None:
            params["orderId"] = tracked_order.exchange_order_id
        elif client_order_id or tracked_order is not None:
            params["clientOrderId"] = client_order_id or tracked_order.client_order_id
        else:
            raise ValueError("orderId or clientOrderId is required")
        response = await self._api_get(
            path_url=CONSTANTS.ORDER_DETAIL_PATH_URL,
            params=params,
            is_auth_required=True,
        )
        result = utils.extract_payload(response)
        if not isinstance(result, dict):
            raise IOError(f"Unexpected Binance Stocks order detail response: {response}")
        self._account_authorized = True
        return result

    async def _request_order_status(self, tracked_order: InFlightOrder) -> OrderUpdate:
        detail = await self._request_order_detail(tracked_order=tracked_order)
        status = str(detail.get("orderStatus", detail.get("status", ""))).upper()
        if status not in CONSTANTS.ORDER_STATE:
            raise IOError(f"Unknown Binance Stocks order state: {status}")
        return OrderUpdate(
            trading_pair=tracked_order.trading_pair,
            update_timestamp=self._seconds(detail.get("updateTime", detail.get("time"))) or self._clock_time(),
            new_state=CONSTANTS.ORDER_STATE[status],
            client_order_id=tracked_order.client_order_id,
            exchange_order_id=str(detail.get("orderId", tracked_order.exchange_order_id)),
        )

    async def _all_trade_updates_for_order(self, order: InFlightOrder) -> List[TradeUpdate]:
        detail = await self._request_order_detail(tracked_order=order)
        cumulative_base = self._decimal(
            detail.get("filledQuantity", detail.get("executedQty", detail.get("cumQuantity", "0")))
        )
        cumulative_quote = self._decimal(
            detail.get("filledAmount", detail.get("cummulativeQuoteQty", detail.get("cumulativeQuoteQty", "0")))
        )
        if cumulative_quote <= 0 and cumulative_base > 0:
            average = self._decimal(detail.get("avgFilledPrice", detail.get("averagePrice", "0")))
            cumulative_quote = cumulative_base * average
        delta_base = cumulative_base - order.executed_amount_base
        delta_quote = cumulative_quote - order.executed_amount_quote
        if delta_base <= 0:
            return []
        fill_price = delta_quote / delta_base if delta_quote > 0 else self._decimal(detail.get("avgFilledPrice"))
        cumulative_fee = self._decimal(detail.get("fee", detail.get("totalFee", "0")))
        if hasattr(self._position_provider, "record_cumulative_fill"):
            await self._position_provider.record_cumulative_fill(
                client_order_id=order.client_order_id,
                exchange_order_id=str(detail.get("orderId", order.exchange_order_id)),
                cumulative_base=cumulative_base,
                cumulative_quote=cumulative_quote,
                cumulative_fee=cumulative_fee,
                status=str(detail.get("orderStatus", detail.get("status", "UNKNOWN"))).upper(),
            )
        already_paid = order.cumulative_fee_paid(self._quote_asset)
        delta_fee = max(Decimal("0"), cumulative_fee - already_paid)
        fee_class = AddedToCostTradeFee if order.trade_type is TradeType.BUY else DeductedFromReturnsTradeFee
        fee = fee_class(flat_fees=[TokenAmount(self._quote_asset, delta_fee)] if delta_fee else [])
        exchange_order_id = str(detail.get("orderId", order.exchange_order_id))
        trade_id = f"{exchange_order_id}:{cumulative_base:f}"
        return [
            TradeUpdate(
                trade_id=trade_id,
                client_order_id=order.client_order_id,
                exchange_order_id=exchange_order_id,
                trading_pair=order.trading_pair,
                fill_timestamp=self._seconds(detail.get("updateTime", detail.get("time"))) or self._clock_time(),
                fill_price=fill_price,
                fill_base_amount=delta_base,
                fill_quote_amount=delta_quote,
                fee=fee,
                is_taker=order.order_type is OrderType.MARKET,
            )
        ]

    @staticmethod
    def _runtime_executor_id() -> Optional[str]:
        """Return the API ExecutorService context without coupling the connector to it."""
        try:
            from utils.executor_log_capture import current_executor_id
            return current_executor_id.get()
        except (ImportError, LookupError):
            return None

    async def _user_stream_event_listener(self):
        async for event in self._iter_user_event_queue():
            try:
                client_order_id = str(event.get("clientOrderId", event.get("c", "")))
                tracked_order = self._order_tracker.all_updatable_orders.get(client_order_id)
                if tracked_order is None:
                    continue
                for trade_update in await self._all_trade_updates_for_order(tracked_order):
                    self._order_tracker.process_trade_update(trade_update)
                raw_status = str(event.get("orderStatus", event.get("status", event.get("X", "")))).upper()
                if raw_status in CONSTANTS.ORDER_STATE:
                    self._order_tracker.process_order_update(
                        OrderUpdate(
                            trading_pair=tracked_order.trading_pair,
                            update_timestamp=self._seconds(event.get("E", event.get("updateTime")))
                            or self._clock_time(),
                            new_state=CONSTANTS.ORDER_STATE[raw_status],
                            client_order_id=client_order_id,
                            exchange_order_id=str(
                                event.get("orderId", event.get("i", tracked_order.exchange_order_id))
                            ),
                        )
                    )
            except asyncio.CancelledError:
                raise
            except Exception:
                self.logger().exception("Unexpected Binance Stocks orderReport processing error")

    async def _update_balances(self):
        self._position_reconciliation_ready = False
        self._last_position_snapshot = None
        self._account_balances.clear()
        self._account_available_balances.clear()
        if not self._position_provider.available:
            return
        snapshot = await self._position_provider.get_snapshot()
        if snapshot is None or self._clock_time() - snapshot.timestamp > CONSTANTS.POSITION_SNAPSHOT_STALE_SECONDS:
            return
        self._last_position_snapshot = snapshot
        self._position_reconciliation_ready = True
        self._account_balances[self._quote_asset] = snapshot.quote_total
        self._account_available_balances[self._quote_asset] = snapshot.quote_available
        for symbol, position in snapshot.positions.items():
            self._account_balances[symbol] = position.total
            self._account_available_balances[symbol] = position.available

    async def _update_trading_fees(self):
        return

    async def _format_trading_rules(self, exchange_info_dict: Dict[str, Any]) -> List[TradingRule]:
        symbols = self._symbols_from_exchange_info(exchange_info_dict)
        rules: List[TradingRule] = []
        for entry in filter(utils.is_exchange_information_valid, symbols):
            try:
                symbol = str(entry.get("symbol", entry.get("ticker"))).upper()
                trading_pair = combine_to_hb_trading_pair(symbol, self._quote_asset)
                filters = {item.get("filterType"): item for item in entry.get("filters", [])}
                price_filter = filters.get("PRICE_FILTER", {})
                lot_filter = filters.get("LOT_SIZE", filters.get("MARKET_LOT_SIZE", {}))
                notional_filter = filters.get("NOTIONAL", filters.get("MIN_NOTIONAL", {}))
                tick_size = self._decimal(
                    entry.get("priceIncrement", entry.get("tickSize", price_filter.get("tickSize", "0.01")))
                )
                # Current Stocks exchangeInfo calls this field `fractionable`.
                # Keep the earlier catalog aliases for captured fixtures and
                # backwards-compatible deployments.
                fractional = bool(entry.get(
                    "fractionable", entry.get("fractional", entry.get("fractionalTrading", False))
                ))
                raw_step = entry.get("quantityIncrement", entry.get("stepSize", lot_filter.get("stepSize")))
                if raw_step is None and fractional:
                    raise ValueError("fractional ticker is missing quantity increment")
                step_size = self._decimal(raw_step if raw_step is not None else "1")
                minimum_quantity = self._decimal(entry.get("minQuantity", lot_filter.get("minQty", step_size)))
                minimum_notional = self._decimal(entry.get(
                    "minNotional", entry.get("minOrderValue", notional_filter.get("minNotional", "0"))
                ))
                self._fractional_supported[symbol] = fractional
                rules.append(
                    TradingRule(
                        trading_pair=trading_pair,
                        min_order_size=minimum_quantity,
                        min_price_increment=tick_size,
                        min_base_amount_increment=step_size,
                        min_notional_size=minimum_notional,
                        supports_limit_orders=True,
                        supports_market_orders=True,
                    )
                )
            except Exception:
                self.logger().exception(f"Error parsing Binance Stocks rule {entry}. Skipping.")
        return rules

    def _initialize_trading_pair_symbols_from_exchange_info(self, exchange_info: Dict[str, Any]):
        mapping = bidict()
        for entry in filter(utils.is_exchange_information_valid, self._symbols_from_exchange_info(exchange_info)):
            symbol = str(entry.get("symbol", entry.get("ticker", ""))).upper()
            if symbol:
                mapping[symbol] = combine_to_hb_trading_pair(symbol, self._quote_asset)
                self._trading_status[symbol] = str(entry.get("status", entry.get("tradingStatus", "TRADING"))).upper()
                self._tradability[symbol] = str(entry.get("tradability", "NONE")).upper()
        self._set_trading_pair_symbol_map(mapping)

    @staticmethod
    def _symbols_from_exchange_info(exchange_info: Any) -> List[Dict[str, Any]]:
        payload = utils.extract_payload(exchange_info)
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict):
            symbols = payload.get("symbols", payload.get("rows", []))
            return symbols if isinstance(symbols, list) else []
        return []

    async def _get_last_traded_price(self, trading_pair: str) -> float:
        prices = await self._orderbook_ds.get_last_traded_prices([trading_pair])
        return prices[trading_pair]

    @staticmethod
    def _decimal(value: Any) -> Decimal:
        if value in (None, ""):
            return Decimal("0")
        return Decimal(str(value))

    @staticmethod
    def _seconds(value: Any) -> float:
        if value in (None, ""):
            return 0.0
        numeric = float(value)
        return numeric / 1000 if numeric > 10_000_000_000 else numeric

    def _clock_time(self) -> float:
        value = self.current_timestamp
        return value if value > 0 and math.isfinite(value) else self._time_synchronizer.time()

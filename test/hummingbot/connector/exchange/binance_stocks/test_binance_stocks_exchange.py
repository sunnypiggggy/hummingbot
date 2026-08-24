from decimal import Decimal
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock

from bidict import bidict

from hummingbot.client.settings import AllConnectorSettings
from hummingbot.connector.exchange.binance_stocks.binance_stocks_exchange import BinanceStocksExchange
from hummingbot.connector.exchange.binance_stocks.binance_stocks_position_provider import (
    EquityAccountSnapshot,
    EquityPosition,
    EquityPositionProvider,
)
from hummingbot.core.data_type.common import OrderType, TradeType
from hummingbot.core.data_type.in_flight_order import InFlightOrder
from hummingbot.core.data_type.trade_fee import AddedToCostTradeFee, DeductedFromReturnsTradeFee


class StaticPositionProvider(EquityPositionProvider):
    def __init__(self, snapshot: EquityAccountSnapshot):
        self.snapshot = snapshot

    @property
    def available(self) -> bool:
        return True

    async def get_snapshot(self):
        return self.snapshot


def configured_exchange(
    trading_required: bool = True,
    provider: EquityPositionProvider = None,
) -> BinanceStocksExchange:
    exchange = BinanceStocksExchange(
        binance_stocks_api_key="key",
        binance_stocks_api_secret="secret",
        trading_pairs=["AAPL-USDC"],
        trading_required=trading_required,
        disclaimer_confirmed=True,
        position_provider=provider,
    )
    exchange._set_current_timestamp(1_700_000_000)
    exchange._set_trading_pair_symbol_map(bidict({"AAPL": "AAPL-USDC"}))
    exchange.process_market_state_event({"e": "calendar", "phase": "MARKET_OPEN"})
    exchange.process_market_state_event({"e": "tradingStatus", "s": "AAPL", "status": "TRADING"})
    exchange.process_market_state_event({"e": "tradability", "s": "AAPL", "tradability": "BOTH"})
    exchange.process_quote_event({"e": "quote", "s": "AAPL", "bp": "199.90", "ap": "200.00", "T": 1_700_000_000_000})
    exchange._fractional_supported["AAPL"] = True
    exchange.set_account_authorized(True)
    return exchange


class BinanceStocksDiscoveryTests(IsolatedAsyncioTestCase):
    async def test_connector_is_auto_discovered(self):
        setting = AllConnectorSettings.get_connector_settings()["binance_stocks"]
        self.assertEqual("AAPL-USDC", setting.example_pair)
        self.assertEqual("binance_stocks", setting.name)
        read_only = setting.non_trading_connector_instance_with_default_configuration(["AAPL-USDC"])
        self.assertFalse(read_only.is_trading_required)

        try:
            AllConnectorSettings.initialize_paper_trade_settings(["binance_stocks"])
            self.assertIn("binance_stocks_paper_trade", AllConnectorSettings.get_connector_settings())
        finally:
            AllConnectorSettings.get_connector_settings().pop("binance_stocks_paper_trade", None)
            AllConnectorSettings.paper_trade_connectors_names = []


class BinanceStocksExchangeTests(IsolatedAsyncioTestCase):
    async def test_official_calendar_to_field_updates_market_phase(self):
        exchange = configured_exchange(trading_required=False)
        exchange.process_market_state_event({
            "e": "calendar", "from": "PRE_MARKET", "to": "MARKET_OPEN", "ts": 1_700_000_001_000,
        })
        self.assertEqual("MARKET_OPEN", exchange.market_phase)
        self.assertEqual("BINANCE", exchange.market_state_metadata["source"])

    async def test_default_position_provider_blocks_order_before_network(self):
        exchange = configured_exchange()
        exchange._api_post = AsyncMock()

        for side in (TradeType.BUY, TradeType.SELL):
            with self.assertRaisesRegex(PermissionError, "position snapshot is unavailable"):
                await exchange._place_order(
                    f"client-{side.name}",
                    "AAPL-USDC",
                    Decimal("0.5"),
                    side,
                    OrderType.LIMIT,
                    Decimal("200"),
                )

        exchange._api_post.assert_not_awaited()
        self.assertFalse(exchange.status_dict["position_reconciliation_ready"])

    async def test_limit_order_uses_extended_day_card_and_bare_ticker(self):
        provider = StaticPositionProvider(
            EquityAccountSnapshot(
                positions={"AAPL": EquityPosition(total=Decimal("2"), available=Decimal("2"))},
                quote_total=Decimal("1000"),
                quote_available=Decimal("1000"),
                source="test-authoritative",
                timestamp=1_700_000_000,
            )
        )
        exchange = configured_exchange(provider=provider)
        await exchange._update_balances()
        exchange._api_post = AsyncMock(
            return_value={
                "status": "S",
                "data": {"orderId": "123", "transactTime": 1_700_000_001_000},
            }
        )

        order_id, timestamp = await exchange._place_order(
            "client-1",
            "AAPL-USDC",
            Decimal("0.5"),
            TradeType.BUY,
            OrderType.LIMIT,
            Decimal("200"),
        )

        self.assertEqual("123", order_id)
        self.assertEqual(1_700_000_001, timestamp)
        payload = exchange._api_post.await_args.kwargs["data"]
        self.assertEqual("AAPL", payload["symbol"])
        self.assertEqual("EXTENDED", payload["tradingSession"])
        self.assertEqual("DAY", payload["timeInForce"])
        self.assertEqual("CARD", payload["walletType"])
        self.assertEqual("LIMIT", payload["orderType"])
        self.assertNotIn("type", payload)
        self.assertNotIn("tokenize", payload)

    async def test_market_order_is_rth_only_and_omits_session_and_tif(self):
        provider = StaticPositionProvider(
            EquityAccountSnapshot(
                quote_total=Decimal("1000"),
                quote_available=Decimal("1000"),
                source="test",
                timestamp=1_700_000_000,
            )
        )
        exchange = configured_exchange(provider=provider)
        await exchange._update_balances()
        exchange._api_post = AsyncMock(return_value={"status": "S", "data": {"orderId": "1"}})

        await exchange._place_order(
            "market-1",
            "AAPL-USDC",
            Decimal("0.5"),
            TradeType.BUY,
            OrderType.MARKET,
            Decimal("NaN"),
        )
        payload = exchange._api_post.await_args.kwargs["data"]
        self.assertNotIn("tradingSession", payload)
        self.assertNotIn("timeInForce", payload)
        self.assertNotIn("price", payload)
        self.assertEqual("100.00", payload["notional"])
        self.assertNotIn("quantity", payload)

        exchange.process_market_state_event({"e": "calendar", "phase": "PRE_MARKET"})
        with self.assertRaisesRegex(PermissionError, "MARKET orders are allowed only"):
            await exchange._place_order(
                "market-2",
                "AAPL-USDC",
                Decimal("0.5"),
                TradeType.BUY,
                OrderType.MARKET,
                Decimal("NaN"),
            )

    async def test_sell_omits_buy_only_wallet_type(self):
        provider = StaticPositionProvider(
            EquityAccountSnapshot(
                positions={"AAPL": EquityPosition(total=Decimal("2"), available=Decimal("2"))},
                quote_total=Decimal("10"),
                quote_available=Decimal("10"),
                source="test",
                timestamp=1_700_000_000,
            )
        )
        exchange = configured_exchange(provider=provider)
        await exchange._update_balances()
        exchange._api_post = AsyncMock(return_value={"status": "S", "orderId": "2"})

        await exchange._place_order(
            "sell-1",
            "AAPL-USDC",
            Decimal("0.5"),
            TradeType.SELL,
            OrderType.LIMIT,
            Decimal("200"),
        )
        payload = exchange._api_post.await_args.kwargs["data"]
        self.assertNotIn("walletType", payload)

    async def test_stale_quote_blocks_while_market_closed_is_not_an_infrastructure_error(self):
        exchange = configured_exchange()
        exchange._set_current_timestamp(1_700_000_011)
        self.assertFalse(exchange._quotes_are_ready())
        exchange.process_market_state_event({"e": "calendar", "phase": "MARKET_CLOSED"})
        self.assertTrue(exchange._quotes_are_ready())

    async def test_direction_and_fractional_rules_are_enforced_before_network(self):
        provider = StaticPositionProvider(
            EquityAccountSnapshot(
                positions={"AAPL": EquityPosition(total=Decimal("2"), available=Decimal("2"))},
                quote_total=Decimal("1000"),
                quote_available=Decimal("1000"),
                source="test",
                timestamp=1_700_000_000,
            )
        )
        exchange = configured_exchange(provider=provider)
        await exchange._update_balances()
        exchange._tradability["AAPL"] = "BUY"
        with self.assertRaisesRegex(PermissionError, "does not currently allow SELL"):
            exchange._assert_order_can_be_placed("AAPL", Decimal("1"), TradeType.SELL, OrderType.LIMIT, Decimal("200"))
        exchange._tradability["AAPL"] = "BUY_SELL"
        exchange._fractional_supported["AAPL"] = False
        with self.assertRaisesRegex(ValueError, "does not support fractional"):
            exchange._assert_order_can_be_placed("AAPL", Decimal("0.5"), TradeType.BUY, OrderType.LIMIT, Decimal("200"))

    async def test_limit_maker_and_non_usdc_are_rejected_locally(self):
        with self.assertRaisesRegex(ValueError, "quote asset"):
            BinanceStocksExchange("k", "s", trading_pairs=["AAPL-USDT"], trading_required=False)
        exchange = configured_exchange()
        with self.assertRaisesRegex(ValueError, "LIMIT_MAKER"):
            exchange._assert_order_can_be_placed(
                "AAPL", Decimal("1"), TradeType.SELL, OrderType.LIMIT_MAKER, Decimal("200")
            )

    async def test_cumulative_detail_generates_only_incremental_fill_and_fee(self):
        exchange = configured_exchange(trading_required=False)
        order = InFlightOrder(
            client_order_id="client-1",
            exchange_order_id="123",
            trading_pair="AAPL-USDC",
            order_type=OrderType.LIMIT,
            trade_type=TradeType.BUY,
            amount=Decimal("1"),
            price=Decimal("200"),
            creation_timestamp=1_700_000_000,
        )
        exchange._request_order_detail = AsyncMock(
            return_value={
                "orderId": "123",
                "filledQuantity": "0.5",
                "filledAmount": "100",
                "avgFilledPrice": "200",
                "fee": "0.35",
                "updateTime": 1_700_000_001_000,
            }
        )
        first = (await exchange._all_trade_updates_for_order(order))[0]
        self.assertEqual(Decimal("0.5"), first.fill_base_amount)
        self.assertEqual(Decimal("0.35"), first.fee.flat_fees[0].amount)
        order.executed_amount_base = Decimal("0.5")
        order.executed_amount_quote = Decimal("100")
        order.order_fills[first.trade_id] = first
        exchange._request_order_detail.return_value = {
            "orderId": "123",
            "filledQuantity": "0.75",
            "filledAmount": "150",
            "avgFilledPrice": "200",
            "fee": "0.35",
            "updateTime": 1_700_000_002_000,
        }
        second = (await exchange._all_trade_updates_for_order(order))[0]
        self.assertEqual(Decimal("0.25"), second.fill_base_amount)
        self.assertEqual([], second.fee.flat_fees)

    async def test_lost_place_response_queries_client_id_without_second_post(self):
        provider = StaticPositionProvider(
            EquityAccountSnapshot(
                quote_total=Decimal("1000"),
                quote_available=Decimal("1000"),
                source="test",
                timestamp=1_700_000_000,
            )
        )
        exchange = configured_exchange(provider=provider)
        await exchange._update_balances()
        exchange._api_post = AsyncMock(side_effect=IOError("response lost after broker accepted order"))
        exchange._request_order_detail = AsyncMock(
            return_value={"orderId": "recovered-1", "clientOrderId": "stable-id", "status": "NEW"}
        )

        order_id, _ = await exchange._place_order(
            "stable-id",
            "AAPL-USDC",
            Decimal("0.5"),
            TradeType.BUY,
            OrderType.LIMIT,
            Decimal("200"),
        )

        self.assertEqual("recovered-1", order_id)
        exchange._api_post.assert_awaited_once()
        exchange._request_order_detail.assert_awaited_once_with(client_order_id="stable-id")
        self.assertFalse(exchange.is_cancel_request_in_exchange_synchronous)

    def test_fee_model_is_usdc_only_and_side_aware(self):
        exchange = configured_exchange(trading_required=False)
        small_buy = exchange._get_fee("AAPL", "USDC", OrderType.LIMIT, TradeType.BUY, Decimal("1"), Decimal("200"))
        large_sell = exchange._get_fee("AAPL", "USDC", OrderType.LIMIT, TradeType.SELL, Decimal("2"), Decimal("200"))
        self.assertIsInstance(small_buy, AddedToCostTradeFee)
        self.assertEqual("USDC", small_buy.flat_fees[0].token)
        self.assertIsInstance(large_sell, DeductedFromReturnsTradeFee)
        self.assertEqual(Decimal("0.001"), large_sell.percent)

    async def test_exchange_info_parses_direct_fractional_equity_and_skips_tokenized(self):
        exchange = configured_exchange(trading_required=False)
        response = {
            "data": {
                "symbols": [
                    {
                        "symbol": "AAPL",
                        "status": "TRADING",
                        "tradability": "BUY_SELL",
                        "fractionalTrading": True,
                        "priceIncrement": "0.01",
                        "quantityIncrement": "0.0001",
                        "minQuantity": "0.0001",
                        "minOrderValue": "5",
                    },
                    {"symbol": "AAPLB", "status": "TRADING", "tokenized": True},
                ]
            }
        }
        exchange._initialize_trading_pair_symbols_from_exchange_info(response)
        rules = await exchange._format_trading_rules(response)
        self.assertEqual(1, len(rules))
        self.assertEqual("AAPL-USDC", rules[0].trading_pair)
        self.assertEqual(Decimal("0.0001"), rules[0].min_base_amount_increment)
        self.assertEqual(Decimal("5"), rules[0].min_notional_size)
        self.assertEqual("BUY_SELL", exchange._tradability["AAPL"])

    async def test_exchange_info_parses_current_fractionable_and_top_level_min_notional(self):
        exchange = configured_exchange(trading_required=False)
        response = {
            "data": {
                "symbols": [{
                    "symbol": "AAPL",
                    "tradability": "BUY_SELL",
                    "fractionable": True,
                    "stepSize": "0.000000001",
                    "minNotional": "5.00000000",
                }]
            }
        }
        rules = await exchange._format_trading_rules(response)
        self.assertEqual(1, len(rules))
        self.assertTrue(exchange._fractional_supported["AAPL"])
        self.assertEqual(Decimal("0.000000001"), rules[0].min_base_amount_increment)
        self.assertEqual(Decimal("5.00000000"), rules[0].min_notional_size)

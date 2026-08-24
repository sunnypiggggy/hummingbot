import json
import os
import tempfile
from decimal import Decimal
from pathlib import Path
from unittest import TestCase, mock

from hummingbot.core.data_type.common import OrderType, TradeType
from hummingbot.strategy_v2.executors.order_executor.data_types import OrderExecutorConfig
from hummingbot.strategy_v2.executors.position_executor.data_types import PositionExecutorConfig

from stocks_runtime.executor_config import (
    build_order_executor_config,
    build_position_executor_config,
    normalize_executor_config,
)
from stocks_runtime.paper_exchange import BinanceStocksPaperExchange
from stocks_runtime.router import OrderExecutorRequest, PositionExecutorRequest
from stocks_runtime.settings import StocksRuntimeSettings


class ExecutorConfigTests(TestCase):
    def test_human_order_request_normalizes_to_hummingbot_config(self):
        public = build_order_executor_config(
            executor_id="limit-aapl-0001", symbol="aapl", side="BUY", amount=Decimal("0.5"),
            order_type="LIMIT", price=Decimal("200"),
        )
        typed = OrderExecutorConfig(**normalize_executor_config(public))
        self.assertIs(typed.side, TradeType.BUY)
        self.assertEqual("LIMIT", typed.execution_strategy.value)
        self.assertEqual(Decimal("200"), typed.price)

    def test_human_position_request_normalizes_all_barrier_order_types(self):
        public = build_position_executor_config(
            executor_id="position-aapl-01", symbol="AAPL", amount=Decimal("0.5"),
            entry_order_type="LIMIT", entry_price=Decimal("200"), stop_loss=Decimal("0.02"),
            time_limit=3600, take_profit=Decimal("0.04"), trailing_activation=Decimal("0.02"),
            trailing_delta=Decimal("0.005"),
        )
        typed = PositionExecutorConfig(**normalize_executor_config(public))
        self.assertIs(typed.side, TradeType.BUY)
        self.assertIs(typed.triple_barrier_config.open_order_type, OrderType.LIMIT)
        self.assertIs(typed.triple_barrier_config.take_profit_order_type, OrderType.LIMIT)
        self.assertIs(typed.triple_barrier_config.stop_loss_order_type, OrderType.MARKET)
        self.assertEqual(Decimal("0.005"), typed.triple_barrier_config.trailing_stop.trailing_delta)

    def test_public_models_reject_ambiguous_price_and_trailing_input(self):
        with self.assertRaisesRegex(Exception, "LIMIT requires price"):
            OrderExecutorRequest(
                id="limit-aapl-0002", symbol="AAPL", side="BUY", amount="0.5", order_type="LIMIT"
            )
        with self.assertRaisesRegex(Exception, "supplied together"):
            PositionExecutorRequest(
                id="position-aapl-02", symbol="AAPL", amount="0.5", entry_order_type="LIMIT",
                entry_price="200", stop_loss="0.02", time_limit=3600, trailing_activation="0.01",
            )

    def test_paper_accepts_market_data_only_key_but_live_does_not(self):
        with tempfile.TemporaryDirectory() as directory:
            secret = Path(directory) / "stocks.json"
            secret.write_text(json.dumps({"api_key": "market-key", "api_secret": ""}), encoding="utf-8")
            env = {
                "DATABASE_URL": "postgresql+asyncpg://u:p@db:5432/main",
                "BINANCE_STOCKS_CREDENTIALS_FILE": str(secret),
                "BINANCE_STOCKS_RUNTIME_MODE": "PAPER",
            }
            with mock.patch.dict(os.environ, env, clear=False):
                paper = StocksRuntimeSettings.from_env()
                self.assertEqual("market-key", paper.read_credentials()["binance_stocks_api_key"])
                self.assertEqual("", paper.read_credentials()["binance_stocks_api_secret"])
            env["BINANCE_STOCKS_RUNTIME_MODE"] = "LIVE"
            env["STOCKS_API_USERNAME"] = "user"
            env["STOCKS_API_PASSWORD"] = "pass"
            with mock.patch.dict(os.environ, env, clear=False):
                live = StocksRuntimeSettings.from_env()
                with self.assertRaisesRegex(ValueError, "api_secret"):
                    live.read_credentials()

    def test_non_official_endpoints_require_scenario_mode(self):
        env = {
            "DATABASE_URL": "postgresql+asyncpg://u:p@db:5432/main",
            "BINANCE_STOCKS_RUNTIME_MODE": "PAPER",
            "BINANCE_STOCKS_REST_URL": "http://market:8080",
        }
        with mock.patch.dict(os.environ, env, clear=False):
            with self.assertRaisesRegex(ValueError, "SCENARIO_MODE"):
                StocksRuntimeSettings.from_env()

    def test_paper_cancel_is_synchronous_while_live_connector_remains_async(self):
        self.assertTrue(BinanceStocksPaperExchange.is_cancel_request_in_exchange_synchronous.fget(None))

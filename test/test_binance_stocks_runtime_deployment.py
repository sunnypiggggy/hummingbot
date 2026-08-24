from pathlib import Path
from unittest import TestCase

import yaml


ROOT = Path(__file__).resolve().parents[1]


class BinanceStocksRuntimeDeploymentTests(TestCase):
    def test_compose_has_one_isolated_loopback_runtime(self):
        compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
        service = compose["services"]["binance-stocks-runtime"]
        self.assertEqual(["stocks"], service["profiles"])
        self.assertEqual(["127.0.0.1:8001:8000"], service["ports"])
        self.assertNotIn("/var/run/docker.sock:/var/run/docker.sock", service.get("volumes", []))
        self.assertEqual(["binance_stocks_credentials"], service["secrets"])
        self.assertEqual("hummingbot_stocks", service["environment"]["BINANCE_STOCKS_DATABASE_NAME"])
        self.assertNotIn("BINANCE_STOCKS_LIVE_AUTHORIZED", service["environment"])
        for name in (
            "BINANCE_STOCKS_MAX_ORDER_USDC", "BINANCE_STOCKS_MAX_SYMBOL_USDC",
            "BINANCE_STOCKS_MAX_EXPOSURE_USDC", "BINANCE_STOCKS_DAILY_LOSS_USDC",
        ):
            self.assertNotIn(name, service["environment"])

    def test_image_defaults_are_fail_closed(self):
        dockerfile = (ROOT / "Dockerfile.binance-stocks-runtime").read_text(encoding="utf-8")
        self.assertIn("BINANCE_STOCKS_RUNTIME_MODE=PAPER", dockerfile)
        self.assertIn("BINANCE_STOCKS_LIVE_AUTHORIZED=false", dockerfile)
        self.assertNotIn("order/place", dockerfile)

    def test_paper_secret_is_empty(self):
        self.assertEqual("{}", (ROOT / "config/binance_stocks_credentials.paper.json").read_text().strip())

    def test_paper_market_stream_is_limited_to_operator_whitelist(self):
        source = (ROOT / "stocks_runtime" / "app.py").read_text(encoding="utf-8")
        self.assertIn('whitelist_pairs = [f"{symbol}-USDC"', source)
        self.assertIn("trading_pairs=whitelist_pairs", source)
        self.assertNotIn("trading_pairs=[]", source)
        data_source = (ROOT / "hummingbot" / "connector" / "exchange" / "binance_stocks" /
                       "binance_stocks_api_order_book_data_source.py").read_text(encoding="utf-8")
        self.assertNotIn("@price", data_source)

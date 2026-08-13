import sys
import tempfile
import unittest
from decimal import Decimal
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "live_guard"))

from dca_live_common import (  # noqa: E402
    LIVE_PAIRS,
    extract_balances,
    layer_quote_amounts,
    live_controller_config,
    required_balances,
    trade_pnl_from_rows,
    validate_config,
    validate_exchange_filters,
)
from dca_live_guard import Guard  # noqa: E402


class FakeApi:
    def __init__(self):
        self.stopped = []
        self.orders = []

    def stop_bot(self, bot_name):
        self.stopped.append(bot_name)
        return {"status": "stopped"}

    def market_order(self, pair, side, amount):
        self.orders.append((pair, side, amount))
        return {"status": "submitted"}


class DcaLiveSafetyTest(unittest.TestCase):
    def test_configs_are_spot_only_and_budgeted(self):
        for pair in LIVE_PAIRS:
            config = live_controller_config(pair)
            validate_config(config)
            self.assertEqual("binance", config["connector_name"])
            self.assertEqual(1, config["leverage"])
            self.assertEqual(190.0, config["total_amount_quote"])
            self.assertTrue(config["skip_rebalance"])

    def test_each_side_has_expected_layers(self):
        self.assertEqual(
            [Decimal("9.50"), Decimal("19.00"), Decimal("28.50"), Decimal("38.00")],
            layer_quote_amounts(),
        )

    def test_min_notional_validation(self):
        symbol_info = {
            "symbol": "BTCUSDT",
            "filters": [
                {"filterType": "LOT_SIZE", "minQty": "0.00001000", "stepSize": "0.00001000"},
                {"filterType": "NOTIONAL", "minNotional": "5.00000000"},
            ],
        }
        validate_exchange_filters(symbol_info, Decimal("65000"))
        symbol_info["filters"][1]["minNotional"] = "10"
        with self.assertRaises(ValueError):
            validate_exchange_filters(symbol_info, Decimal("65000"))

    def test_shared_balance_requirements_reserve_ten_per_bot(self):
        requirements = required_balances({
            "BTC-USDT": Decimal("65000"),
            "ETH-USDT": Decimal("2000"),
        })
        self.assertEqual(Decimal("210"), requirements["USDT"])
        self.assertEqual(Decimal("95") / Decimal("65000"), requirements["BTC"])
        self.assertEqual(Decimal("95") / Decimal("2000"), requirements["ETH"])

    def test_portfolio_response_is_normalized(self):
        payload = {
            "binance_live_dca_200": {
                "binance": [
                    {"token": "USDT", "units": 210},
                    {"token": "BTC", "units": "0.002"},
                ]
            }
        }
        self.assertEqual(
            {"USDT": Decimal("210"), "BTC": Decimal("0.002")},
            extract_balances(payload),
        )

    def test_trade_pnl_ignores_starting_account_inventory(self):
        # Buy 0.01 BTC at 60k, sell 0.005 at 64k, then mark remaining at 65k.
        rows = [
            ("BUY", 60_000_000_000, 10_000, 120_000, 1),
            ("SELL", 64_000_000_000, 5_000, 64_000, 2),
        ]
        metrics = trade_pnl_from_rows(rows, Decimal("65000"))
        self.assertEqual(Decimal("0.005"), metrics["net_base"])
        self.assertEqual(Decimal("44.816"), metrics["pnl_quote"])
        self.assertEqual(Decimal("0.184"), metrics["fees_quote"])

    def test_flatten_restores_only_bot_inventory_delta(self):
        guard = Guard.__new__(Guard)
        guard.api = FakeApi()
        guard._lot_filter = lambda pair: (Decimal("0.00001"), Decimal("5"))
        result = guard._flatten({
            "pair": "BTC-USDT",
            "net_base": "0.00123456",
            "mark_price": "65000",
        })
        self.assertEqual("SELL", result["side"])
        self.assertEqual("0.00123", result["amount"])
        self.assertEqual(
            [("BTC-USDT", "SELL", Decimal("0.00123"))],
            guard.api.orders,
        )

    def test_trip_is_persistent_and_not_repeated(self):
        with tempfile.TemporaryDirectory() as directory:
            guard = Guard.__new__(Guard)
            guard.api = FakeApi()
            guard.state_dir = Path(directory)
            guard.state_path = Path(directory) / "guard_state.json"
            guard.audit_path = Path(directory) / "audit.jsonl"
            guard.state = {"bots": {}}
            guard._notify = lambda message: None
            guard._flatten = lambda snapshot: {"status": "not_required"}
            guard._trip("dca-live-btcusdt-200", "test loss", {
                "pair": "BTC-USDT",
                "net_base": "0",
                "mark_price": "65000",
            })
            guard._trip("dca-live-btcusdt-200", "test loss", None)
            state = guard.state["bots"]["dca-live-btcusdt-200"]
            self.assertTrue(state["tripped"])
            self.assertTrue(state["action_complete"])
            self.assertEqual(["dca-live-btcusdt-200"], guard.api.stopped)


if __name__ == "__main__":
    unittest.main()

import sys
import types
import unittest
from types import SimpleNamespace
from unittest import mock

from hummingbot.core.data_type.common import TradeType

from scripts.dca_live_common import live_controller_config, validate_config


class DcaLongOnlyLiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with mock.patch.dict(
            sys.modules, {"pandas_ta": types.ModuleType("pandas_ta")}
        ):
            from controllers.market_making.dman_maker_v3_macro import DManMakerV3Macro

        cls.controller_class = DManMakerV3Macro

    def controller(self, *, macro_buy=True, macro_sell=True, long_only=True):
        controller = self.controller_class.__new__(self.controller_class)
        controller.config = SimpleNamespace(
            macro_buy_enabled=macro_buy,
            macro_sell_enabled=macro_sell,
            long_only_enabled=long_only,
            sell_trend_gate_enabled=True,
        )
        controller._sell_trend_blocked = False
        return controller

    def test_live_profiles_are_long_only(self):
        for pair in ("BTC-USDT", "ETH-USDT"):
            config = live_controller_config(pair)
            validate_config(config)
            self.assertIs(config["long_only_enabled"], True)
            self.assertIs(config["macro_sell_enabled"], True)

    def test_only_buy_executors_can_be_created(self):
        controller = self.controller()
        self.assertTrue(controller.creation_side_enabled(TradeType.BUY))
        self.assertFalse(controller.creation_side_enabled(TradeType.SELL))
        self.assertEqual([], controller.get_candles_config())

    def test_long_only_does_not_force_stop_existing_sell_executor(self):
        controller = self.controller(macro_sell=True)
        self.assertFalse(controller.force_stop_side_required(TradeType.SELL))
        controller.config.macro_sell_enabled = False
        self.assertTrue(controller.force_stop_side_required(TradeType.SELL))

    def test_macro_buy_gate_still_blocks_buy_creation_and_exposure(self):
        controller = self.controller(macro_buy=False)
        self.assertFalse(controller.creation_side_enabled(TradeType.BUY))
        self.assertTrue(controller.force_stop_side_required(TradeType.BUY))

    def test_validation_rejects_bilateral_live_profile(self):
        config = live_controller_config("BTC-USDT")
        config["long_only_enabled"] = False
        with self.assertRaisesRegex(ValueError, "long-only"):
            validate_config(config)


if __name__ == "__main__":
    unittest.main()

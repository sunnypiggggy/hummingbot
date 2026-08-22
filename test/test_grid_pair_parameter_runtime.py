import sys
import unittest
from decimal import Decimal, ROUND_DOWN
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.grid_live_common import (  # noqa: E402
    clip_quantized_buy_levels,
    clip_quantized_sell_levels,
)


class PairParameterRuntimeTest(unittest.TestCase):
    @staticmethod
    def quantizer(step):
        return lambda amount: (
            (Decimal(amount) / step).to_integral_value(rounding=ROUND_DOWN) * step
        )

    def test_btc_sell_layers_are_clipped_instead_of_disappearing(self):
        levels = [Decimal("70000") + Decimal(index * 500) for index in range(9)]
        orders = clip_quantized_sell_levels(
            levels, Decimal("0.000768"), Decimal("10"),
            lambda level: max(level, Decimal("69000") * Decimal("1.004"), Decimal("68000")),
            self.quantizer(Decimal("0.00001")),
        )
        self.assertEqual(len(orders), 5)
        self.assertTrue(all(price * amount >= Decimal("10") for price, amount in orders))
        self.assertLessEqual(sum(amount for _, amount in orders), Decimal("0.000768"))

    def test_eth_inventory_can_support_all_nine_sell_layers(self):
        levels = [Decimal("1950") + Decimal(index * 50) for index in range(9)]
        orders = clip_quantized_sell_levels(
            levels, Decimal("0.05069"), Decimal("10"),
            lambda level: max(level, Decimal("1900") * Decimal("1.014179761072002472"), Decimal("1850")),
            self.quantizer(Decimal("0.0001")),
        )
        self.assertEqual(len(orders), 9)
        self.assertTrue(all(price * amount >= Decimal("10") for price, amount in orders))
        self.assertLessEqual(sum(amount for _, amount in orders), Decimal("0.05069"))

    def test_btc_exact_minimum_budget_rounds_buy_up_and_clips_farthest(self):
        # Production incident: 13 lower levels, 100 FDUSD budget, a 10 FDUSD
        # floor and BTC's 0.00001 amount step at roughly 71,532 FDUSD.
        levels = [
            Decimal("65143.50") + Decimal(index) * Decimal("679.46")
            for index in range(13)
        ]
        orders = clip_quantized_buy_levels(
            levels,
            Decimal("100"),
            Decimal("10"),
            lambda level: level.quantize(Decimal("0.01")),
            self.quantizer(Decimal("0.00001")),
            amount_step=Decimal("0.00001"),
            minimum_amount=Decimal("0.00001"),
        )
        self.assertGreaterEqual(len(orders), 1)
        self.assertLess(len(orders), 10)  # ten upward-rounded orders exceed 100
        self.assertTrue(all(price * amount >= Decimal("10") for price, amount in orders))
        self.assertLessEqual(
            sum((price * amount for price, amount in orders), Decimal("0")),
            Decimal("100"),
        )
        self.assertEqual(
            sorted((price for price, _ in orders), reverse=True),
            [price for price, _ in orders],
        )

    def test_buy_builder_respects_dynamic_minimum_and_min_quantity(self):
        levels = [Decimal("1800"), Decimal("1850"), Decimal("1900")]
        orders = clip_quantized_buy_levels(
            levels, Decimal("35"), Decimal("11"), lambda value: value,
            self.quantizer(Decimal("0.0001")),
            amount_step=Decimal("0.0001"), minimum_amount=Decimal("0.006"),
        )
        self.assertTrue(orders)
        self.assertTrue(all(amount >= Decimal("0.006") for _, amount in orders))
        self.assertTrue(all(price * amount >= Decimal("11") for price, amount in orders))
        self.assertLessEqual(sum(price * amount for price, amount in orders), Decimal("35"))

    def test_eth_single_layer_uses_executable_minimum_instead_of_overspending(self):
        # 2026-08-22 live incident: using the full 11.223 FDUSD layer budget
        # rounds 0.00464 ETH up to 0.0047 and overspends.  A 0.0042 ETH order
        # clears the 10 FDUSD minimum and must be retained.
        price = Decimal("2418.22")
        budget = Decimal("11.223135676")
        orders = clip_quantized_buy_levels(
            [price], budget, Decimal("10"), lambda value: value,
            self.quantizer(Decimal("0.0001")),
            amount_step=Decimal("0.0001"), minimum_amount=Decimal("0.0001"),
        )
        self.assertEqual([(price, Decimal("0.0046"))], orders)
        self.assertGreaterEqual(price * orders[0][1], Decimal("10"))
        self.assertLessEqual(price * orders[0][1], budget)

    def test_buy_builder_returns_explicit_empty_below_minimum_budget(self):
        self.assertEqual([], clip_quantized_buy_levels(
            [Decimal("70000")], Decimal("9.99"), Decimal("10"), lambda value: value,
            self.quantizer(Decimal("0.00001")), amount_step=Decimal("0.00001"),
        ))


if __name__ == "__main__":
    unittest.main()

import sys
import unittest
from decimal import Decimal, ROUND_DOWN
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.grid_live_common import clip_quantized_sell_levels  # noqa: E402


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


if __name__ == "__main__":
    unittest.main()

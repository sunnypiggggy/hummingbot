import unittest

import pandas as pd

from scripts.portfolio_grid_core import GridParams, simulate_portfolio


class PortfolioGridCoreFeeTest(unittest.TestCase):
    def candles(self):
        return {
            "BTC-FDUSD": pd.DataFrame([
                {"timestamp": 0, "open": 100, "high": 100, "low": 100, "close": 100},
                {"timestamp": 300, "open": 100, "high": 102, "low": 98, "close": 101},
            ])
        }

    def test_zero_maker_fee_improves_grid_result(self):
        params = GridParams(0.04, 3, 0.1, 0.003, 0.5)
        zero_maker = simulate_portfolio(
            self.candles(), params, 1000, 0, 0.08, 24,
            maker_fee_rate=0, taker_fee_rate=0.0002,
        )
        paid_maker = simulate_portfolio(
            self.candles(), params, 1000, 0.0002, 0.08, 24,
            maker_fee_rate=0.0002, taker_fee_rate=0.0002,
        )
        self.assertGreater(zero_maker["net_pnl_quote"], paid_maker["net_pnl_quote"])


if __name__ == "__main__":
    unittest.main()

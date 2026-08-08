import sys
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from compare_live_grid_momentum_models import build_stop_timeline  # noqa: E402
from validate_grid_live import Candidate, simulate  # noqa: E402


def candle_frame() -> pd.DataFrame:
    return pd.DataFrame([
        {"timestamp": 0, "open": 100, "high": 100, "low": 100, "close": 100, "volume": 10},
        {"timestamp": 300, "open": 100, "high": 100, "low": 96, "close": 97, "volume": 10},
        {"timestamp": 600, "open": 97, "high": 98, "low": 96, "close": 97, "volume": 10},
        {"timestamp": 900, "open": 97, "high": 98, "low": 97, "close": 98, "volume": 10},
    ])


class LiveGridMomentumStopTest(unittest.TestCase):
    def test_signal_starts_at_closed_hour_and_never_before_it(self):
        predictions = pd.DataFrame([{
            "pair": "BTC-FDUSD", "signal_ts": 3_600, "LightGBM": 0.9,
        }])
        timeline = build_stop_timeline(
            predictions, "LightGBM", 0.8, 0, 30_000, horizon_hours=6
        )
        self.assertNotIn(3_300, timeline["BTC-FDUSD"])
        self.assertEqual(0.9, timeline["BTC-FDUSD"][3_600])
        self.assertIn(24_900, timeline["BTC-FDUSD"])
        self.assertNotIn(25_200, timeline["BTC-FDUSD"])

    def test_stop_flattens_only_excess_inventory(self):
        candidate = Candidate(0.03, 0.006, 0.006, 0.015, 1_800)
        trades = []
        result, _, pairs = simulate(
            {"BTC-FDUSD": candle_frame()}, candidate, maker_fee=0.0,
            taker_fee=0.001, risk_breakers_enabled=False,
            momentum_stop_timeline={"BTC-FDUSD": {600: 0.9}},
            momentum_stop_threshold=0.8, trade_log=trades,
        )
        exits = [trade for trade in trades if trade["reason"] == "momentum_stop_exit"]
        self.assertEqual(1, len(exits))
        self.assertGreater(exits[0]["amount"], 0)
        self.assertAlmostEqual(0.0, pairs["BTC-FDUSD"]["inventory_delta"], places=12)
        self.assertEqual(1, result["momentum_stop_exits"])
        self.assertGreater(result["fees_quote"], 0)

    def test_empty_overlay_preserves_baseline_numerics(self):
        candidate = Candidate(0.03, 0.006, 0.006, 0.015, 1_800)
        baseline, _, baseline_pairs = simulate(
            {"BTC-FDUSD": candle_frame()}, candidate, maker_fee=0.0,
            taker_fee=0.001, risk_breakers_enabled=False,
        )
        overlay, _, overlay_pairs = simulate(
            {"BTC-FDUSD": candle_frame()}, candidate, maker_fee=0.0,
            taker_fee=0.001, risk_breakers_enabled=False,
            momentum_stop_timeline={"BTC-FDUSD": {}},
        )
        for key in ("final_equity", "net_pnl_quote", "max_drawdown_pct", "trades", "fees_quote"):
            self.assertEqual(baseline[key], overlay[key])
        self.assertEqual(
            baseline_pairs["BTC-FDUSD"]["inventory_delta"],
            overlay_pairs["BTC-FDUSD"]["inventory_delta"],
        )


if __name__ == "__main__":
    unittest.main()

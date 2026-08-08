import sys
import unittest
from pathlib import Path

import pandas as pd


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from fdusd_live_grid_optimizer import DAY_SECONDS, rolling_validation_windows  # noqa: E402
from search_fdusd_inventory_exit import (  # noqa: E402
    INITIAL_APPROVED,
    add_balanced_score,
    approved_candidate,
    holdout_windows,
    inventory_policy_space,
    policy_id,
    stop_metrics,
)
from validate_grid_live import Candidate  # noqa: E402
from build_fdusd_technical_gate_kline import count_trades_in_regions  # noqa: E402


class FDUSDInventoryExitSearchTest(unittest.TestCase):
    def test_search_space_has_420_unique_valid_policies(self):
        policies = inventory_policy_space()
        self.assertEqual(len(policies), 420)
        self.assertEqual(len({policy_id(policy) for policy in policies}), 420)
        self.assertTrue(all(policy.stage_two_profit_rate == 0 for policy in policies))
        self.assertTrue(all(
            0 < policy.stage_one_fraction < policy.stage_two_fraction < 1
            for policy in policies
        ))

    def test_development_and_holdout_windows_do_not_look_forward(self):
        start = 1_700_000_000
        holdout_start = start + 120 * DAY_SECONDS
        end = holdout_start + 60 * DAY_SECONDS
        development = rolling_validation_windows(start, holdout_start)
        holdout = holdout_windows(holdout_start, end)
        self.assertTrue(development)
        self.assertTrue(holdout)
        self.assertTrue(all(train_end <= test_start < test_end <= holdout_start
                            for _, train_end, test_start, test_end in development))
        self.assertTrue(all(train_end <= test_start < test_end <= end
                            for _, train_end, test_start, test_end in holdout))
        self.assertTrue(all(test_start >= holdout_start for _, _, test_start, _ in holdout))

    def test_no_eligible_candidate_retains_previous_parameters(self):
        selected = Candidate(0.05, 0.01, 0.01, 0.03)
        approved, retained = approved_candidate(selected, 0, INITIAL_APPROVED)
        self.assertEqual(approved, INITIAL_APPROVED)
        self.assertTrue(retained)
        approved, retained = approved_candidate(selected, 1, INITIAL_APPROVED)
        self.assertEqual(approved, selected)
        self.assertFalse(retained)

    def test_balanced_score_rewards_profit_drawdown_and_less_stopping(self):
        frame = pd.DataFrame([
            {"policy_id": "better", "oos_pnl_fdusd": 5.0, "worst_drawdown_pct": -2.0,
             "portfolio_stop_hours": 0.0, "pair_stop_hours": 10.0},
            {"policy_id": "worse", "oos_pnl_fdusd": -5.0, "worst_drawdown_pct": -6.0,
             "portfolio_stop_hours": 20.0, "pair_stop_hours": 100.0},
        ])
        ranked = add_balanced_score(frame)
        self.assertEqual(ranked.iloc[0].policy_id, "better")
        self.assertGreater(ranked.iloc[0].balanced_score, ranked.iloc[1].balanced_score)

    def test_zero_notional_breaker_is_still_counted_as_stopped(self):
        curve = pd.DataFrame([{"timestamp": 1000, "equity": 420, "drawdown_pct": 0}])
        trades = [{
            "timestamp": 900, "pair": "BTC-FDUSD", "side": "SELL", "price": 1,
            "amount": 0, "quote_notional": 0, "reason": "pair_breaker_flatten",
        }]
        metrics = stop_metrics({"liquidated": False}, curve, trades, 4500)
        self.assertEqual(metrics["pair_stop_events"], 1)
        self.assertEqual(metrics["pair_stop_hours"], 1.0)

    def test_technical_pause_audit_counts_sells_but_no_buys(self):
        trades = pd.DataFrame([
            {"period": "holdout", "scenario": "online", "timestamp": 120,
             "side": "SELL", "reason": "grid_fill"},
            {"period": "holdout", "scenario": "online", "timestamp": 220,
             "side": "BUY", "reason": "grid_fill"},
            {"period": "holdout", "scenario": "new", "timestamp": 150,
             "side": "SELL", "reason": "grid_fill"},
        ])
        regions = pd.DataFrame([{"start_ts": 100, "end_ts": 200}])
        audit = count_trades_in_regions(trades, regions).set_index("scenario")
        self.assertEqual(audit.loc["online", "grid_buys_during_pause"], 0)
        self.assertEqual(audit.loc["online", "sells_during_pause"], 1)
        self.assertEqual(audit.loc["new", "grid_buys_during_pause"], 0)
        self.assertEqual(audit.loc["new", "sells_during_pause"], 1)


if __name__ == "__main__":
    unittest.main()

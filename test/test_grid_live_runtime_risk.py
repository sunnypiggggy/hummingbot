import sys
import unittest
from unittest.mock import patch
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from grid_live_common import PairLedger  # noqa: E402
from walk_forward_portfolio_grid_live import LivePortfolioGrid  # noqa: E402


class Order:
    def __init__(self, order_id: str, pair: str = "BTC-FDUSD"):
        self.client_order_id = order_id
        self.trading_pair = pair


class GridLiveRuntimeRiskTest(unittest.TestCase):
    def strategy(self):
        strategy = LivePortfolioGrid.__new__(LivePortfolioGrid)
        strategy.config = SimpleNamespace(
            exchange="binance",
            fail_closed_seconds=60,
            reserve_quote=Decimal("20"),
            pair_budget_quote=Decimal("200"),
            pair_stop_loss_quote=Decimal("6"),
            pair_drawdown_limit_pct=Decimal("0.03"),
            capital_limit_quote=Decimal("420"),
            portfolio_stop_loss_quote=Decimal("24"),
            portfolio_drawdown_limit_pct=Decimal("0.06"),
            trading_pairs=["BTC-FDUSD", "ETH-FDUSD"],
        )
        strategy.ledgers = {
            "BTC-FDUSD": PairLedger.create("BTC-FDUSD", Decimal("0.002")),
            "ETH-FDUSD": PairLedger.create("ETH-FDUSD", Decimal("0.05")),
        }
        strategy.flatten_order_ids = {}
        strategy.buy_order_ids = set()
        strategy.sell_order_ids = set()
        strategy.first_cycle_failure_at = None
        strategy.pending_flatten = set()
        strategy.portfolio_tripped = False
        strategy.peak_equity = Decimal("420")
        strategy.notify = lambda message: None
        strategy._current_timestamp = 1_000.0
        strategy.next_refresh = 9_999.0
        strategy.next_macro_poll = 0.0
        strategy.next_technical_poll = 0.0
        strategy.macro_paused = False
        strategy.macro_gate_healthy = True
        strategy.macro_active_lease_ids = []
        strategy.macro_reason = "active"
        strategy.macro_transition_key = None
        strategy.technical_buy_enabled = True
        strategy.technical_gate_healthy = True
        strategy.technical_reason = "active"
        strategy.technical_signal = {}
        strategy.technical_transition_key = None
        strategy.runtime_events = []
        return strategy

    def test_foreign_orders_do_not_block_or_get_cancelled(self):
        strategy = self.strategy()
        strategy.ledgers["BTC-FDUSD"].open_order_ids.add("owned")
        active = [Order("owned"), Order("foreign")]
        strategy.get_active_orders = lambda exchange: active
        cancelled = []
        strategy.cancel = lambda exchange, pair, order_id: cancelled.append(order_id)

        self.assertEqual(
            ["owned"],
            [order.client_order_id for order in strategy._owned_active_orders(active)],
        )
        strategy.cancel_owned_orders()
        self.assertEqual(["owned"], cancelled)

    def test_flatten_waits_for_grid_cancel_and_existing_flatten(self):
        strategy = self.strategy()
        ledger = strategy.ledgers["BTC-FDUSD"]
        ledger.open_order_ids.update({"grid", "flatten"})
        strategy.flatten_order_ids = {"BTC-FDUSD": "flatten"}

        grid, flatten = strategy._partition_flatten_orders(
            [Order("grid"), Order("flatten"), Order("foreign")]
        )
        self.assertEqual({"grid"}, grid)
        self.assertEqual({"flatten"}, flatten)
        self.assertEqual((set(), set()), strategy._partition_flatten_orders([]))

    def test_terminal_order_events_remove_owned_ids(self):
        strategy = self.strategy()
        strategy.ledgers["BTC-FDUSD"].open_order_ids.add("done")
        strategy.buy_order_ids.add("done")
        strategy._forget_order("done")
        self.assertNotIn("done", strategy.ledgers["BTC-FDUSD"].open_order_ids)
        self.assertNotIn("done", strategy.buy_order_ids)

    def test_technical_risk_off_cancels_buy_but_preserves_sell(self):
        strategy = self.strategy()
        strategy.ledgers["BTC-FDUSD"].open_order_ids.update({"buy", "sell"})
        strategy.buy_order_ids.add("buy")
        strategy.sell_order_ids.add("sell")
        active = [Order("buy"), Order("sell")]
        strategy.get_active_orders = lambda exchange: active
        cancelled = []
        strategy.cancel = lambda exchange, pair, order_id: cancelled.append(order_id)
        self.assertFalse(strategy.cancel_owned_buy_orders())
        self.assertEqual(["buy"], cancelled)

    def test_unknown_order_sides_cancel_all_and_report_rebuild_required(self):
        strategy = self.strategy()
        strategy.ledgers["BTC-FDUSD"].open_order_ids.update({"one", "two"})
        strategy.get_active_orders = lambda exchange: [Order("one"), Order("two")]
        cancelled = []
        strategy.cancel = lambda exchange, pair, order_id: cancelled.append(order_id)
        self.assertTrue(strategy.cancel_owned_buy_orders())
        self.assertEqual(["one", "two"], cancelled)

    def test_fomc_resume_forces_immediate_refresh(self):
        strategy = self.strategy()
        strategy.config.macro_gate_enabled = True
        strategy.config.macro_gate_poll_seconds = 5
        strategy.config.macro_gate_file = "unused"
        strategy.config.macro_gate_max_age_seconds = 150
        strategy.config.macro_fail_closed = True
        strategy.macro_paused = True
        with patch("walk_forward_portfolio_grid_live.load_runtime_macro_gate", return_value={
            "runtime_gate_healthy": True,
            "pause_new_orders": False,
            "active_lease_ids": [],
            "reason": "no_active_fomc_lease",
        }):
            strategy._poll_macro_gate()
        self.assertEqual(0.0, strategy.next_refresh)
        self.assertEqual("fomc_gate_resumed_immediate_refresh", strategy.runtime_events[-1]["event"])

    def test_unknown_sides_risk_off_forces_sell_only_refresh(self):
        strategy = self.strategy()
        strategy.config.technical_buy_gate_enabled = True
        strategy.config.technical_buy_gate_poll_seconds = 5
        strategy.config.technical_buy_gate_file = "unused"
        strategy.config.technical_buy_gate_max_age_seconds = 150
        strategy.config.technical_buy_fail_closed = True
        strategy.cancel_owned_buy_orders = lambda: True
        with patch("walk_forward_portfolio_grid_live.load_runtime_technical_gate", return_value={
            "runtime_gate_healthy": True,
            "buy_enabled": False,
            "reason": "roc_sqzmom_combined_risk_off",
            "signal": {"roc_48h_pct": -6, "sqzmom_pct": -2},
        }):
            strategy._poll_technical_buy_gate()
        self.assertEqual(0.0, strategy.next_refresh)
        self.assertFalse(strategy.technical_buy_enabled)
        self.assertEqual(
            "technical_risk_off_unknown_sides_sell_rebuild_scheduled",
            strategy.runtime_events[-1]["event"],
        )

    def test_continuous_cycle_failure_trips_after_sixty_seconds(self):
        strategy = self.strategy()
        self.assertFalse(strategy._cycle_failure_requires_trip(100.0))
        self.assertFalse(strategy._cycle_failure_requires_trip(159.9))
        self.assertTrue(strategy._cycle_failure_requires_trip(160.0))

    def test_unknown_sides_cold_start_risk_off_also_forces_refresh(self):
        strategy = self.strategy()
        strategy.config.technical_buy_gate_enabled = True
        strategy.config.technical_buy_gate_poll_seconds = 5
        strategy.config.technical_buy_gate_file = "unused"
        strategy.config.technical_buy_gate_max_age_seconds = 150
        strategy.config.technical_buy_fail_closed = True
        strategy.technical_buy_enabled = False
        strategy.cancel_owned_buy_orders = lambda: True
        with patch("walk_forward_portfolio_grid_live.load_runtime_technical_gate", return_value={
            "runtime_gate_healthy": True,
            "buy_enabled": False,
            "reason": "roc_sqzmom_combined_risk_off",
            "signal": {},
        }):
            strategy._poll_technical_buy_gate()
        self.assertEqual(0.0, strategy.next_refresh)

    def test_fixed_pair_loss_trips_even_before_drawdown_threshold(self):
        strategy = self.strategy()
        ledger = strategy.ledgers["BTC-FDUSD"]
        ledger.quote = Decimal("193.5")
        ledger.base = Decimal("0")
        ledger.initial_base = Decimal("0")
        ledger.peak_equity = Decimal("193.5")
        strategy._control_risk({
            "BTC-FDUSD": Decimal("65000"),
            "ETH-FDUSD": Decimal("3500"),
        })
        self.assertTrue(ledger.halted)
        self.assertIn("BTC-FDUSD", strategy.pending_flatten)

    def test_fixed_portfolio_loss_trips_all_pairs(self):
        strategy = self.strategy()
        for ledger in strategy.ledgers.values():
            ledger.quote = Decimal("187")
            ledger.base = Decimal("0")
            ledger.initial_base = Decimal("0")
            ledger.peak_equity = Decimal("200")
        strategy._control_risk({
            "BTC-FDUSD": Decimal("65000"),
            "ETH-FDUSD": Decimal("3500"),
        })
        self.assertTrue(strategy.portfolio_tripped)
        self.assertEqual(set(strategy.config.trading_pairs), strategy.pending_flatten)


if __name__ == "__main__":
    unittest.main()

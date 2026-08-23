import sys
import signal
import unittest
from unittest.mock import patch
from decimal import Decimal, ROUND_DOWN
from pathlib import Path
from types import SimpleNamespace

from hummingbot.core.data_type.common import OrderType, TradeType
from hummingbot.core.data_type.trade_fee import DeductedFromReturnsTradeFee

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from grid_live_common import PairLedger  # noqa: E402
from walk_forward_portfolio_grid_live import (  # noqa: E402
    GridState,
    LivePortfolioGrid,
    ParameterBuildError,
)
from risk_recovery import ACTIVE, EXITING, REENTRY, active_state, normalize_state, trigger_state  # noqa: E402


class Order:
    def __init__(self, order_id: str, pair: str = "BTC-FDUSD"):
        self.client_order_id = order_id
        self.trading_pair = pair


class Connector:
    def __init__(self):
        self.limit_orders = []
        self.available = {
            "FDUSD": Decimal("1000"),
            "BTC": Decimal("1"),
            "ETH": Decimal("10"),
        }

    def get_available_balance(self, asset):
        return self.available.get(asset, Decimal("0"))


class GridLiveRuntimeRiskTest(unittest.TestCase):
    def strategy(self):
        strategy = LivePortfolioGrid.__new__(LivePortfolioGrid)
        strategy.config = SimpleNamespace(
            exchange="binance",
            quote_asset="FDUSD",
            fail_closed_seconds=60,
            reserve_quote=Decimal("20"),
            pair_budget_quote=Decimal("200"),
            pair_stop_loss_quote=Decimal("6"),
            pair_drawdown_limit_pct=Decimal("0.03"),
            capital_limit_quote=Decimal("420"),
            portfolio_stop_loss_quote=Decimal("24"),
            portfolio_drawdown_limit_pct=Decimal("0.06"),
            trading_pairs=["BTC-FDUSD", "ETH-FDUSD"],
            pair_breakers_enabled=True,
            portfolio_breakers_enabled=True,
            cost_floor_enabled=True,
            inventory_exit_enabled=True,
            max_extra_inventory_quote=Decimal("10"),
            profit_protection_seconds=24 * 3600,
            max_extra_inventory_hold_seconds=48 * 3600,
            take_profit=Decimal("0.006"),
            min_order_quote=Decimal("5"),
            side_budget_quote=Decimal("100"),
            move_threshold=Decimal("0.015"),
            min_grid_move_seconds=1800,
            startup_order_reconcile_seconds=30,
            fee_rate=Decimal("0"),
            taker_fee_rate=Decimal("0.001"),
        )
        strategy.connectors = {"binance": Connector()}
        strategy.ledgers = {
            "BTC-FDUSD": PairLedger.create("BTC-FDUSD", Decimal("0.002")),
            "ETH-FDUSD": PairLedger.create("ETH-FDUSD", Decimal("0.05")),
        }
        strategy.flatten_order_ids = {}
        strategy.pending_inventory_exit = set()
        strategy.inventory_exit_order_ids = {}
        strategy.excess_inventory_started_at = {
            pair: None for pair in strategy.config.trading_pairs
        }
        strategy.buy_order_ids = set()
        strategy.sell_order_ids = set()
        strategy.first_cycle_failure_at = None
        strategy.pending_flatten = set()
        strategy.portfolio_tripped = False
        strategy.peak_equity = Decimal("420")
        strategy.notify = lambda message: None
        strategy._set_current_timestamp(1_000.0)
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
        strategy.technical_buy_enabled_by_pair = {pair: True for pair in strategy.config.trading_pairs}
        strategy.technical_gate_healthy_by_pair = {pair: True for pair in strategy.config.trading_pairs}
        strategy.technical_reason_by_pair = {pair: "active" for pair in strategy.config.trading_pairs}
        strategy.technical_signal_by_pair = {pair: {} for pair in strategy.config.trading_pairs}
        strategy.technical_transition_key = None
        strategy.pair_parameters = {
            pair: {
                "profile": "legacy_shared",
                "grid_range": Decimal("0.06"),
                "grid_levels": 10,
                "take_profit": Decimal("0.006"),
                "minimum_order_quote": Decimal("5"),
                "move_threshold": Decimal("0.015"),
                "min_grid_move_seconds": 1800,
                "order_refresh_seconds": 7200,
            }
            for pair in strategy.config.trading_pairs
        }
        strategy.grid_states = {}
        strategy.pair_recovery = {
            pair: active_state() for pair in strategy.config.trading_pairs
        }
        strategy.portfolio_recovery = active_state()
        strategy.parameter_blocked_pairs = {}
        strategy.order_build_status = {
            pair: strategy._empty_order_build_status()
            for pair in strategy.config.trading_pairs
        }
        strategy.active_parameter_version = "test-legacy-shared"
        strategy.active_parameter_sha256 = "test"
        strategy.integrity_failure_grace = {}
        strategy.runtime_events = []
        strategy._append_notification_event = lambda *args, **kwargs: None
        return strategy

    def test_btc_incident_builds_executable_buy_orders_with_exact_budget(self):
        strategy = self.strategy()
        strategy.pair_parameters["BTC-FDUSD"].update({
            "profile": "medium_sideways",
            "grid_range": Decimal("0.12698379475402316"),
            "grid_levels": 18,
            "minimum_order_quote": Decimal("10"),
        })
        strategy.connector.trading_rules = {
            "BTC-FDUSD": SimpleNamespace(
                min_notional_size=Decimal("10"),
                min_base_amount_increment=Decimal("0.00001"),
                min_order_size=Decimal("0.00001"),
            )
        }
        strategy.connector.quantize_order_amount = lambda pair, amount: (
            (Decimal(amount) / Decimal("0.00001")).to_integral_value(
                rounding=ROUND_DOWN
            ) * Decimal("0.00001")
        )
        strategy.connector.quantize_order_price = lambda pair, order_price: (
            Decimal(order_price).quantize(Decimal("0.01"), rounding=ROUND_DOWN)
        )
        price = Decimal("71532")
        strategy.grid_states["BTC-FDUSD"] = strategy._new_grid("BTC-FDUSD", price)
        strategy.connector.available["FDUSD"] = Decimal("100")
        strategy.connector.available["BTC"] = Decimal("0")
        # Reproduce the live inventory deficit that makes the full 100 FDUSD
        # BUY-side budget available (rather than the 10 FDUSD extra-inventory cap).
        strategy.ledgers["BTC-FDUSD"].base = Decimal("0")
        strategy.ledgers["BTC-FDUSD"].base_cost_quote = Decimal("0")
        submitted = []
        strategy.buy = lambda exchange, pair, amount, order_type, order_price: (
            submitted.append((pair, amount, order_price)) or f"buy-{len(submitted)}"
        )
        strategy.sell = lambda *args, **kwargs: self.fail("unexpected sell")

        remaining = strategy._place_pair_grid("BTC-FDUSD", price, Decimal("100"))

        self.assertTrue(submitted)
        self.assertTrue(all(amount * order_price >= Decimal("10")
                            for _, amount, order_price in submitted))
        total = sum((amount * order_price for _, amount, order_price in submitted), Decimal("0"))
        self.assertLessEqual(total, Decimal("100"))
        self.assertEqual(Decimal("100") - total, remaining)
        self.assertEqual("HEALTHY", strategy.order_build_status["BTC-FDUSD"]["state"])

    def test_eth_single_layer_incident_builds_without_exceeding_budget(self):
        strategy = self.strategy()
        strategy.pair_parameters["ETH-FDUSD"].update({
            "profile": "long_volatility",
            "minimum_order_quote": Decimal("10"),
        })
        strategy.connector.trading_rules = {
            "ETH-FDUSD": SimpleNamespace(
                min_notional_size=Decimal("10"),
                min_base_amount_increment=Decimal("0.0001"),
                min_order_size=Decimal("0.0001"),
            )
        }
        strategy.connector.quantize_order_amount = lambda pair, amount: (
            (Decimal(amount) / Decimal("0.0001")).to_integral_value(
                rounding=ROUND_DOWN
            ) * Decimal("0.0001")
        )
        strategy.connector.quantize_order_price = lambda pair, order_price: (
            Decimal(order_price).quantize(Decimal("0.01"), rounding=ROUND_DOWN)
        )

        budget = Decimal("11.223135676")
        orders = strategy._build_buy_orders(
            "ETH-FDUSD", [Decimal("2418.22")], budget,
        )

        self.assertEqual(1, len(orders))
        self.assertGreaterEqual(orders[0][0] * orders[0][1], Decimal("10"))
        self.assertLessEqual(orders[0][0] * orders[0][1], budget)

    def test_buy_build_failure_preserves_safe_sell_side_and_remains_degraded(self):
        strategy = self.strategy()
        pair = "ETH-FDUSD"
        price = Decimal("2418.22")
        strategy.grid_states[pair] = strategy._new_grid(pair, price)
        strategy._build_buy_orders = lambda *args, **kwargs: []
        strategy._build_sell_orders = lambda *args, **kwargs: [
            (Decimal("2500"), Decimal("0.004")),
        ]
        submitted_sells = []
        strategy.buy = lambda *args, **kwargs: self.fail("unexpected buy")
        strategy.sell = lambda exchange, trading_pair, amount, order_type, order_price: (
            submitted_sells.append((trading_pair, amount, order_price)) or "sell-safe"
        )

        with self.assertRaisesRegex(ParameterBuildError, "BUY build produced no executable"):
            strategy._place_pair_grid(pair, price, Decimal("100"))

        self.assertEqual([(pair, Decimal("0.004"), Decimal("2500"))], submitted_sells)
        status = strategy.order_build_status[pair]
        self.assertGreaterEqual(status["expected_buy_layers"], 1)
        self.assertEqual(1, status["expected_sell_layers"])
        self.assertEqual(0, status["actual_buy_layers"])
        self.assertEqual(1, status["actual_sell_layers"])

        status.update({
            "state": "RETRYING", "consecutive_empty_cycles": 3,
            "first_empty_at": 900.0, "first_failure_at": 900.0,
            "next_retry_at": 999.0, "retry_count": 1,
        })
        rebuilt = []
        strategy._place_pair_grid = lambda *args: rebuilt.append(args) or Decimal("0")
        strategy._check_order_liveness_and_rebuild(
            {"BTC-FDUSD": Decimal("71532"), pair: price},
            [Order("sell-safe", pair)],
        )
        self.assertEqual(pair, rebuilt[0][0])

    def test_targeted_btc_rebuild_does_not_touch_eth_orders(self):
        strategy = self.strategy()
        strategy.pair_recovery = {pair: active_state() for pair in strategy.config.trading_pairs}
        strategy.ledgers["BTC-FDUSD"].halted = False
        status = strategy.order_build_status["BTC-FDUSD"]
        status.update({
            "state": "RETRYING", "expected_buy_layers": 6,
            "expected_sell_layers": 0, "consecutive_empty_cycles": 3,
            "first_empty_at": 900.0, "first_failure_at": 900.0,
            "next_retry_at": 999.0, "retry_count": 1,
        })
        eth_order = Order("eth-live", "ETH-FDUSD")
        strategy.sell_order_ids.add("eth-live")
        strategy.ledgers["ETH-FDUSD"].open_order_ids.add("eth-live")
        rebuilt = []
        strategy._place_pair_grid = lambda pair, price, quote: (
            rebuilt.append((pair, price, quote)) or quote
        )

        changed = strategy._check_order_liveness_and_rebuild(
            {"BTC-FDUSD": Decimal("71532"), "ETH-FDUSD": Decimal("2100")},
            [eth_order],
        )

        self.assertTrue(changed)
        self.assertEqual(["BTC-FDUSD"], [value[0] for value in rebuilt])
        self.assertIn("eth-live", strategy.ledgers["ETH-FDUSD"].open_order_ids)

    def test_risk_off_pair_does_not_trigger_zero_order_rebuild(self):
        strategy = self.strategy()
        strategy.order_build_status["BTC-FDUSD"].update({
            "state": "HEALTHY", "expected_buy_layers": 6,
            "expected_sell_layers": 0,
        })
        strategy.technical_gate_healthy_by_pair["BTC-FDUSD"] = False
        calls = []
        strategy._place_pair_grid = lambda *args: calls.append(args)
        strategy._check_order_liveness_and_rebuild(
            {"BTC-FDUSD": Decimal("71532"), "ETH-FDUSD": Decimal("2100")}, []
        )
        self.assertEqual([], calls)

    def test_market_fill_fee_conversion_failure_uses_taker_not_maker_rate(self):
        strategy = self.strategy()
        order_id = "market-reentry"
        ledger = strategy.ledgers["ETH-FDUSD"]
        ledger.open_order_ids.add(order_id)

        class MissingRateFee:
            @staticmethod
            def fee_amount_in_token(*_args, **_kwargs):
                raise ValueError("quote conversion unavailable")

        event = SimpleNamespace(
            trading_pair="ETH-FDUSD",
            order_id=order_id,
            price=Decimal("2000"),
            amount=Decimal("0.05"),
            order_type=SimpleNamespace(name="MARKET"),
            trade_type=SimpleNamespace(name="BUY"),
            trade_fee=MissingRateFee(),
        )

        strategy.did_fill_order(event)

        self.assertEqual(Decimal("0.1"), ledger.fees_quote)
        self.assertEqual("quote_fee_fallback_applied", strategy.runtime_events[-1]["event"])
        self.assertEqual("0.001", strategy.runtime_events[-1]["fee_rate"])

    def test_market_buy_fee_deducted_from_returns_owns_only_received_base(self):
        strategy = self.strategy()
        order_id = "market-reentry-base-fee"
        ledger = strategy.ledgers["ETH-FDUSD"]
        ledger.open_order_ids.add(order_id)
        initial_quote = ledger.quote
        initial_base = ledger.base
        initial_cost = ledger.base_cost_quote
        initial_fees = ledger.fees_quote

        event = SimpleNamespace(
            trading_pair="ETH-FDUSD",
            order_id=order_id,
            price=Decimal("2000"),
            amount=Decimal("0.05"),
            order_type=OrderType.MARKET,
            trade_type=TradeType.BUY,
            trade_fee=DeductedFromReturnsTradeFee(percent=Decimal("0.001")),
        )

        strategy.did_fill_order(event)

        self.assertEqual(initial_quote - Decimal("100"), ledger.quote)
        self.assertEqual(initial_base + Decimal("0.04995"), ledger.base)
        self.assertEqual(initial_cost + Decimal("100"), ledger.base_cost_quote)
        self.assertEqual(initial_fees + Decimal("0.1"), ledger.fees_quote)

    def test_shutdown_flag_blocks_new_orders_without_competing_cancellation(self):
        strategy = LivePortfolioGrid.__new__(LivePortfolioGrid)
        strategy._is_stop_triggered = True
        calls = []
        strategy.cancel_strategy_pair_orders = lambda: calls.append("cancel")
        strategy.on_tick()
        self.assertEqual([], calls)

    def test_btc_reentry_does_not_block_active_eth_grid(self):
        strategy = self.strategy()
        strategy.config.risk_auto_reentry_enabled = True
        strategy.config.side_budget_quote = Decimal("100")
        strategy.pair_recovery = {
            "BTC-FDUSD": normalize_state({
                **active_state(),
                "phase": REENTRY,
                "mechanism": "v22_weekly_buy_gate",
                "scope": "technical",
            }),
            "ETH-FDUSD": active_state(),
        }
        strategy.portfolio_recovery = active_state()
        strategy.reentry_order_ids = {}
        strategy.ledgers["BTC-FDUSD"].halted = True
        strategy.ledgers["ETH-FDUSD"].halted = False
        strategy.technical_buy_enabled_by_pair = {
            "BTC-FDUSD": False,
            "ETH-FDUSD": True,
        }
        strategy.grid_states = {
            "BTC-FDUSD": GridState(
                lower=Decimal("63000"), upper=Decimal("67000"),
                levels=[Decimal("63000"), Decimal("67000")],
            ),
            "ETH-FDUSD": GridState(
                lower=Decimal("1800"), upper=Decimal("2000"),
                levels=[Decimal("1800"), Decimal("2000")],
            ),
        }
        submitted = []
        strategy.buy = lambda exchange, pair, amount, order_type, price: (
            submitted.append(("BUY", pair)) or f"buy-{pair}"
        )
        strategy.sell = lambda exchange, pair, amount, order_type, price: (
            submitted.append(("SELL", pair)) or f"sell-{pair}"
        )

        blocks_all_grids = strategy._advance_risk_recovery(
            {"BTC-FDUSD": Decimal("65000"), "ETH-FDUSD": Decimal("1900")}, []
        )
        if not blocks_all_grids:
            strategy._place_grids(
                {"BTC-FDUSD": Decimal("65000"), "ETH-FDUSD": Decimal("1900")}
            )

        self.assertFalse(blocks_all_grids)
        self.assertEqual(REENTRY, strategy.pair_recovery["BTC-FDUSD"]["phase"])
        self.assertTrue(submitted)
        self.assertEqual({"ETH-FDUSD"}, {pair for _, pair in submitted})
        self.assertEqual(ACTIVE, strategy.pair_recovery["ETH-FDUSD"]["phase"])

    def test_shutdown_pair_cancellation_is_rate_limited_but_retried(self):
        strategy = self.strategy()
        strategy.connector.limit_orders = [Order("old", "ETH-FDUSD")]
        strategy._set_current_timestamp(1000)
        cancelled = []
        strategy.cancel = lambda exchange, pair, order_id: cancelled.append(order_id)

        self.assertEqual(1, strategy.cancel_strategy_pair_orders())
        self.assertEqual(1, strategy.cancel_strategy_pair_orders())
        self.assertEqual(["old"], cancelled)
        strategy._set_current_timestamp(1005)
        self.assertEqual(1, strategy.cancel_strategy_pair_orders())
        self.assertEqual(["old", "old"], cancelled)

    def test_sigterm_blocks_orders_and_schedules_only_one_graceful_shutdown(self):
        strategy = self.strategy()
        scheduled = []

        class Loop:
            @staticmethod
            def is_running():
                return True

            @staticmethod
            def create_task(coroutine):
                scheduled.append(coroutine)
                return "shutdown-task"

        strategy._termination_signal_received = None
        strategy._termination_loop = Loop()
        strategy._request_termination(signal.SIGTERM)
        strategy._request_termination(signal.SIGTERM)

        self.assertTrue(strategy._is_stop_triggered)
        self.assertEqual(signal.SIGTERM, strategy._termination_signal_received)
        self.assertEqual(1, len(scheduled))
        self.assertEqual("shutdown-task", strategy._termination_task)
        scheduled[0].close()

    def test_startup_reconciliation_cancels_restored_orders_before_new_grid(self):
        strategy = self.strategy()
        strategy.startup_reconcile_started_at = None
        strategy.startup_reconcile_quiet_cycles = 0
        strategy.startup_reconcile_cancel_attempts = {}
        strategy.startup_reconcile_complete = False
        strategy.ledgers["ETH-FDUSD"].open_order_ids.add("old-sell")
        restored = Order("old-sell", "ETH-FDUSD")
        strategy.connector.limit_orders = [restored]
        cancelled = []
        strategy.cancel = lambda exchange, pair, order_id: cancelled.append(order_id)

        self.assertTrue(strategy._startup_order_reconciliation([]))
        self.assertEqual(["old-sell"], cancelled)
        strategy.connector.limit_orders = []
        strategy._set_current_timestamp(1030)
        self.assertTrue(strategy._startup_order_reconciliation([]))
        strategy._set_current_timestamp(1031)
        self.assertTrue(strategy._startup_order_reconciliation([]))
        strategy._set_current_timestamp(1032)
        self.assertFalse(strategy._startup_order_reconciliation([]))
        self.assertTrue(strategy.startup_reconcile_complete)

    def test_confirmed_inactive_orders_are_pruned_before_new_generation(self):
        strategy = self.strategy()
        strategy.buy_order_ids = {"old-buy", "still-active"}
        strategy.sell_order_ids = {"old-sell"}
        strategy.ledgers["BTC-FDUSD"].open_order_ids.update(
            {"old-buy", "old-sell", "still-active"}
        )

        strategy._prune_inactive_order_ownership([Order("still-active")])

        self.assertEqual({"still-active"}, strategy.buy_order_ids)
        self.assertEqual(set(), strategy.sell_order_ids)
        self.assertEqual(
            {"still-active"}, strategy.ledgers["BTC-FDUSD"].open_order_ids,
        )

    def test_grid_orders_are_capped_by_exchange_available_balances(self):
        strategy = self.strategy()
        strategy.config.trading_pairs = ["ETH-FDUSD"]
        strategy.config.side_budget_quote = Decimal("100")
        strategy.ledgers = {"ETH-FDUSD": strategy.ledgers["ETH-FDUSD"]}
        strategy.grid_states = {"ETH-FDUSD": GridState(
            lower=Decimal("1900"), upper=Decimal("2100"),
            levels=[Decimal("1900"), Decimal("2100")],
        )}
        strategy.connector.available["FDUSD"] = Decimal("6")
        strategy.connector.available["ETH"] = Decimal("0.01")
        buys, sells = [], []
        strategy.buy = lambda exchange, pair, amount, order_type, price: (
            buys.append(amount * price) or "buy"
        )
        strategy.sell = lambda exchange, pair, amount, order_type, price: (
            sells.append(amount) or "sell"
        )

        strategy._place_grids({"ETH-FDUSD": Decimal("2000")})

        self.assertEqual([Decimal("6")], buys)
        self.assertEqual([Decimal("0.01")], sells)

    def test_notification_failure_does_not_interrupt_runtime_risk_event(self):
        strategy = self.strategy()
        strategy.logger = lambda: SimpleNamespace(error=lambda *args, **kwargs: None)
        strategy._append_notification_event = lambda *args, **kwargs: (_ for _ in ()).throw(
            OSError("notification volume is read-only")
        )

        strategy._record_runtime_event("risk_breaker_triggered", pair="BTC-FDUSD")

        self.assertEqual("risk_breaker_triggered", strategy.runtime_events[-1]["event"])

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

    def test_pair_exit_cancels_only_the_affected_pair_orders(self):
        strategy = self.strategy()
        strategy.pending_flatten = {"ETH-FDUSD"}
        strategy.portfolio_recovery = active_state()
        strategy.ledgers["BTC-FDUSD"].open_order_ids.add("btc-grid")
        strategy.ledgers["ETH-FDUSD"].open_order_ids.add("eth-grid")
        active = [Order("btc-grid", "BTC-FDUSD"), Order("eth-grid", "ETH-FDUSD")]
        strategy.get_active_orders = lambda exchange: active
        cancelled = []
        strategy.cancel = lambda exchange, pair, order_id: cancelled.append((pair, order_id))

        targets = strategy._flatten_target_pairs()
        grid, flatten = strategy._partition_flatten_orders(active, pairs=targets)
        strategy.cancel_owned_orders(pairs=targets)

        self.assertEqual({"ETH-FDUSD"}, targets)
        self.assertEqual({"eth-grid"}, grid)
        self.assertEqual(set(), flatten)
        self.assertEqual([("ETH-FDUSD", "eth-grid")], cancelled)

    def test_portfolio_exit_cancels_both_pair_order_sets(self):
        strategy = self.strategy()
        strategy.pending_flatten = {"ETH-FDUSD"}
        strategy.portfolio_recovery = trigger_state(
            mechanism="portfolio_drawdown_breaker", scope="portfolio", now=1000,
            trigger_value=Decimal("0.06"), signal_price={}, reason="portfolio drawdown",
        )
        for pair, order_id in (("BTC-FDUSD", "btc-grid"), ("ETH-FDUSD", "eth-grid")):
            strategy.ledgers[pair].open_order_ids.add(order_id)
        active = [Order("btc-grid", "BTC-FDUSD"), Order("eth-grid", "ETH-FDUSD")]
        strategy.get_active_orders = lambda exchange: active
        cancelled = []
        strategy.cancel = lambda exchange, pair, order_id: cancelled.append((pair, order_id))

        targets = strategy._flatten_target_pairs()
        strategy.cancel_owned_orders(pairs=targets)

        self.assertEqual(set(strategy.config.trading_pairs), targets)
        self.assertEqual(
            {("BTC-FDUSD", "btc-grid"), ("ETH-FDUSD", "eth-grid")},
            set(cancelled),
        )

    def test_breaker_exit_sells_all_managed_base_not_only_inventory_delta(self):
        strategy = self.strategy()
        pair = "BTC-FDUSD"
        strategy.config.trading_pairs = [pair]
        strategy.ledgers = {pair: strategy.ledgers[pair]}
        strategy.ledgers[pair].base += Decimal("0.0002")
        strategy.pending_flatten = {pair}
        strategy.pair_recovery = {pair: trigger_state(
            mechanism="strategy_loss_breaker", scope="strategy", now=1000,
            trigger_value=-6, signal_price=65000, reason="loss",
        )}
        strategy.portfolio_recovery = {"phase": "ACTIVE"}
        submitted = []
        strategy.sell = lambda exchange, trading_pair, amount, order_type: (
            submitted.append(amount) or "quote-only-exit"
        )
        strategy._restore_inventory({pair: Decimal("65000")})
        self.assertEqual([Decimal("0.0022")], submitted)
        self.assertEqual(EXITING, strategy.pair_recovery[pair]["phase"])

    def test_terminal_order_events_remove_owned_ids(self):
        strategy = self.strategy()
        strategy.ledgers["BTC-FDUSD"].open_order_ids.add("done")
        strategy.buy_order_ids.add("done")
        strategy._forget_order("done")
        self.assertNotIn("done", strategy.ledgers["BTC-FDUSD"].open_order_ids)
        self.assertNotIn("done", strategy.buy_order_ids)

    def test_grid_sell_price_respects_average_cost_floor(self):
        strategy = self.strategy()
        strategy.config.trading_pairs = ["BTC-FDUSD"]
        strategy.config.side_budget_quote = Decimal("100")
        strategy.config.take_profit = Decimal("0.006")
        strategy.config.min_order_quote = Decimal("5")
        strategy.config.move_threshold = Decimal("0.015")
        strategy.config.min_grid_move_seconds = 1800
        strategy.technical_buy_enabled_by_pair["BTC-FDUSD"] = False
        ledger = strategy.ledgers["BTC-FDUSD"]
        ledger.apply_fill("BUY", Decimal("40000"), Decimal("0.001"), Decimal("0"))
        strategy.ledgers = {"BTC-FDUSD": ledger}
        strategy.grid_states = {
            "BTC-FDUSD": GridState(
                lower=Decimal("39000"), upper=Decimal("41000"),
                levels=[Decimal("39000"), Decimal("41000")],
            )
        }
        strategy.buy = lambda *args, **kwargs: "buy"
        submitted = []
        strategy.sell = lambda exchange, pair, amount, order_type, price: (
            submitted.append((pair, amount, price)) or "sell"
        )

        strategy._place_grids({"BTC-FDUSD": Decimal("40000")})

        self.assertEqual(1, len(submitted))
        self.assertEqual(
            ledger.minimum_profitable_sell_price(Decimal("0.006")),
            submitted[0][2],
        )

    def test_grid_buy_budget_is_capped_at_ten_quote_above_baseline(self):
        strategy = self.strategy()
        strategy.config.trading_pairs = ["BTC-FDUSD"]
        strategy.config.side_budget_quote = Decimal("100")
        strategy.config.move_threshold = Decimal("0.015")
        strategy.config.min_grid_move_seconds = 1800
        strategy.ledgers = {"BTC-FDUSD": strategy.ledgers["BTC-FDUSD"]}
        strategy.grid_states = {"BTC-FDUSD": GridState(
            lower=Decimal("39000"), upper=Decimal("41000"),
            levels=[Decimal("39000"), Decimal("41000")],
        )}
        submitted = []
        strategy.buy = lambda exchange, pair, amount, order_type, price: (
            submitted.append(amount * price) or "buy"
        )
        strategy.sell = lambda *args, **kwargs: "sell"
        strategy._place_grids({"BTC-FDUSD": Decimal("40000")})
        self.assertEqual([Decimal("10")], submitted)

    def test_grid_inventory_cap_keeps_nearest_affordable_buy_level(self):
        strategy = self.strategy()
        strategy.config.trading_pairs = ["BTC-FDUSD"]
        strategy.config.side_budget_quote = Decimal("100")
        strategy.config.min_order_quote = Decimal("5.25")
        strategy.pair_parameters["BTC-FDUSD"]["minimum_order_quote"] = Decimal("5.25")
        strategy.ledgers = {"BTC-FDUSD": strategy.ledgers["BTC-FDUSD"]}
        strategy.grid_states = {"BTC-FDUSD": GridState(
            lower=Decimal("37000"), upper=Decimal("41000"),
            levels=[Decimal("37000"), Decimal("38000"), Decimal("39000"), Decimal("41000")],
        )}
        submitted = []
        strategy.buy = lambda exchange, pair, amount, order_type, price: (
            submitted.append((price, amount * price)) or "buy"
        )
        strategy.sell = lambda *args, **kwargs: "sell"

        strategy._place_grids({"BTC-FDUSD": Decimal("40000")})

        self.assertEqual([(Decimal("39000"), Decimal("10"))], submitted)

    def test_cost_floor_relaxes_to_break_even_after_24_hours(self):
        strategy = self.strategy()
        strategy.excess_inventory_started_at["BTC-FDUSD"] = (
            strategy.current_timestamp - 25 * 3600
        )
        self.assertEqual(Decimal("0"), strategy._inventory_profit_floor_rate("BTC-FDUSD"))

    def test_48_hour_exit_sells_only_inventory_above_baseline(self):
        strategy = self.strategy()
        pair = "BTC-FDUSD"
        ledger = strategy.ledgers[pair]
        extra = Decimal("0.0002")
        ledger.base += extra
        strategy.excess_inventory_started_at[pair] = strategy.current_timestamp - 48 * 3600
        submitted = []
        strategy.sell = lambda exchange, trading_pair, amount, order_type: (
            submitted.append((trading_pair, amount, order_type)) or "inventory-exit"
        )
        self.assertTrue(strategy._process_inventory_exit_policy(
            {"BTC-FDUSD": Decimal("60000"), "ETH-FDUSD": Decimal("3000")}, []
        ))
        self.assertEqual(extra, submitted[0][1])
        self.assertEqual("inventory-exit", strategy.inventory_exit_order_ids[pair])
        self.assertEqual(ledger.initial_base + extra, ledger.base)

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
        self.assertEqual(
            ["fomc_gate_resumed_immediate_refresh", "fomc_gate_transition"],
            [event["event"] for event in strategy.runtime_events[-2:]],
        )

    def test_unknown_sides_risk_off_forces_sell_only_refresh(self):
        strategy = self.strategy()
        strategy.config.technical_buy_gate_enabled = True
        strategy.config.technical_buy_gate_poll_seconds = 5
        strategy.config.technical_buy_gate_file = "unused"
        strategy.config.technical_buy_gate_max_age_seconds = 150
        strategy.config.technical_buy_fail_closed = True
        strategy.config.technical_model_sha256 = ""
        strategy.config.technical_feature_sha256 = ""
        strategy.cancel_owned_buy_orders = lambda pair=None: True
        with patch("walk_forward_portfolio_grid_live.load_runtime_xgboost_gate", return_value={
            "runtime_gate_healthy": True,
            "reason": "xgboost_gate_healthy",
            "pairs": {
                "BTC-FDUSD": {"buy_enabled": False, "reason": "channel_or_risk_off"},
                "ETH-FDUSD": {"buy_enabled": True, "reason": "all_channels_clear"},
            },
        }):
            strategy._poll_technical_buy_gate()
        self.assertEqual(0.0, strategy.next_refresh)
        self.assertFalse(strategy.technical_buy_enabled)
        self.assertEqual(
            "xgboost_risk_off_unknown_sides_sell_rebuild_scheduled",
            strategy.runtime_events[-1]["event"],
        )

    def test_transient_technical_failure_blocks_buys_without_immediate_latch(self):
        strategy = self.strategy()
        strategy.config.technical_buy_gate_enabled = True
        strategy.config.technical_buy_gate_poll_seconds = 5
        strategy.config.technical_buy_gate_file = "unused"
        strategy.config.technical_buy_gate_max_age_seconds = 150
        strategy.config.technical_buy_fail_closed = True
        strategy.config.technical_model_sha256 = ""
        strategy.config.technical_feature_sha256 = ""
        strategy.cancel_owned_buy_orders = lambda pair=None: False
        strategy._latch_integrity_failure = lambda reason: self.fail(reason)
        with patch("walk_forward_portfolio_grid_live.load_runtime_xgboost_gate", return_value={
            "runtime_gate_healthy": False,
            "reason": (
                "fail_closed:500 Server Error: Internal Server Error for url: "
                "https://api.binance.com/api/v3/time"
            ),
            "pairs": {},
        }):
            strategy._poll_technical_buy_gate()
        self.assertFalse(strategy.technical_buy_enabled)
        self.assertFalse(strategy.integrity_failure_grace["technical_contract"]["expired"])
        self.assertEqual(
            "technical_contract_transport_grace_started",
            strategy.runtime_events[-1]["event"],
        )

    def test_deterministic_technical_failure_still_latches_immediately(self):
        strategy = self.strategy()
        strategy.config.technical_buy_gate_enabled = True
        strategy.config.technical_buy_gate_poll_seconds = 5
        strategy.config.technical_buy_gate_file = "unused"
        strategy.config.technical_buy_gate_max_age_seconds = 150
        strategy.config.technical_buy_fail_closed = True
        strategy.config.technical_model_sha256 = ""
        strategy.config.technical_feature_sha256 = ""
        strategy.cancel_owned_buy_orders = lambda pair=None: False
        latched = []
        strategy._latch_integrity_failure = latched.append
        with patch("walk_forward_portfolio_grid_live.load_runtime_xgboost_gate", return_value={
            "runtime_gate_healthy": False,
            "reason": "fail_closed:model hash mismatch",
            "pairs": {},
        }):
            strategy._poll_technical_buy_gate()
        self.assertEqual(1, len(latched))
        self.assertIn("model hash mismatch", latched[0])

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
        strategy.config.technical_model_sha256 = ""
        strategy.config.technical_feature_sha256 = ""
        strategy.technical_buy_enabled = False
        strategy.cancel_owned_buy_orders = lambda pair=None: True
        with patch("walk_forward_portfolio_grid_live.load_runtime_xgboost_gate", return_value={
            "runtime_gate_healthy": True,
            "reason": "xgboost_gate_healthy",
            "pairs": {
                "BTC-FDUSD": {"buy_enabled": False, "reason": "channel_or_risk_off"},
                "ETH-FDUSD": {"buy_enabled": False, "reason": "channel_or_risk_off"},
            },
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

    def test_fdusd_mechanisms_1_to_3_do_not_enable_portfolio_breaker(self):
        strategy = self.strategy()
        strategy.config.portfolio_breakers_enabled = False
        for ledger in strategy.ledgers.values():
            ledger.quote = Decimal("187")
            ledger.base = Decimal("0")
            ledger.initial_base = Decimal("0")
            ledger.peak_equity = Decimal("200")
        strategy._control_risk({
            "BTC-FDUSD": Decimal("65000"),
            "ETH-FDUSD": Decimal("3500"),
        })
        self.assertFalse(strategy.portfolio_tripped)
        # Pair-local breakers still act independently.
        self.assertEqual(set(strategy.config.trading_pairs), strategy.pending_flatten)

    def test_inventory_cap_is_absent_when_inventory_exit_mechanism_is_disabled(self):
        strategy = self.strategy()
        strategy.config.trading_pairs = ["BTC-FDUSD"]
        strategy.config.side_budget_quote = Decimal("100")
        strategy.config.move_threshold = Decimal("0.015")
        strategy.config.min_grid_move_seconds = 1800
        strategy.config.inventory_exit_enabled = False
        strategy.ledgers = {"BTC-FDUSD": strategy.ledgers["BTC-FDUSD"]}
        strategy.grid_states = {"BTC-FDUSD": GridState(
            lower=Decimal("39000"), upper=Decimal("41000"),
            levels=[Decimal("39000"), Decimal("41000")],
        )}
        submitted = []
        strategy.buy = lambda exchange, pair, amount, order_type, price: (
            submitted.append(amount * price) or "buy"
        )
        strategy.sell = lambda *args, **kwargs: "sell"
        strategy._place_grids({"BTC-FDUSD": Decimal("40000")})
        self.assertEqual([Decimal("100")], submitted)


if __name__ == "__main__":
    unittest.main()

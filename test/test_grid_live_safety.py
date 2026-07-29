import json
import sys
import tempfile
import unittest
from decimal import Decimal
from pathlib import Path

import pandas as pd


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from grid_live_common import (  # noqa: E402
    ACTIVE_SELECTION_SCHEMA_VERSION,
    CAPITAL_LIMIT,
    FDUSD_RECOMMENDED_BALANCE,
    FDUSD_BUDGET,
    ORDER_REFRESH_SECONDS,
    PAIR_BUDGET,
    PAIR_DRAWDOWN_LIMIT_PCT,
    PORTFOLIOS,
    PORTFOLIO_DRAWDOWN_LIMIT_PCT,
    RISK_STATE_PERSIST_SECONDS,
    RESERVE_QUOTE,
    SIDE_BUDGET,
    STRATEGY_BUDGET,
    USDT_BUDGET,
    PairLedger,
    build_fdusd_bootstrap_plan,
    build_live_config,
    effective_take_profit,
    required_balances,
    validate_active_selection,
    validate_live_config,
)
from validate_grid_live import Candidate, simulate  # noqa: E402
from fdusd_live_grid_optimizer import (  # noqa: E402
    rolling_validation_windows,
    weekly_cutoff,
)
from prepare_fdusd_live_grid import weighted_ask  # noqa: E402
from deploy_fdusd_live_grid import (  # noqa: E402
    ApiClient,
    NO_GO_OVERRIDE_CONFIRMATION,
    guard_readiness,
    load_bootstrap_receipt,
    validation_authorization,
)
from live_guard.grid_live_guard import Guard, fill_pnl, peak_drawdown  # noqa: E402


class GridLiveSafetyTest(unittest.TestCase):
    def setUp(self):
        self.prices = {
            "BTC-USDT": Decimal("65000"), "ETH-USDT": Decimal("3500"),
            "BTC-FDUSD": Decimal("65010"), "ETH-FDUSD": Decimal("3501"),
        }

    def test_quote_specific_approved_budgets(self):
        self.assertEqual(CAPITAL_LIMIT, Decimal("500"))
        self.assertEqual(STRATEGY_BUDGET, Decimal("475"))
        self.assertEqual(RESERVE_QUOTE, Decimal("25"))
        self.assertEqual(PAIR_BUDGET, Decimal("237.5"))
        self.assertEqual(SIDE_BUDGET, Decimal("118.75"))
        self.assertEqual(USDT_BUDGET.capital_limit, Decimal("500"))
        self.assertEqual(FDUSD_BUDGET.capital_limit, Decimal("420"))
        self.assertEqual(FDUSD_BUDGET.strategy_budget, Decimal("400"))
        self.assertEqual(FDUSD_BUDGET.reserve_quote, Decimal("20"))
        self.assertEqual(FDUSD_BUDGET.pair_budget, Decimal("200"))
        self.assertEqual(FDUSD_BUDGET.side_budget, Decimal("100"))
        self.assertEqual(FDUSD_BUDGET.pair_loss_limit, Decimal("6"))
        self.assertEqual(FDUSD_BUDGET.portfolio_loss_limit, Decimal("24"))
        self.assertEqual(PAIR_DRAWDOWN_LIMIT_PCT, Decimal("0.03"))
        self.assertEqual(PORTFOLIO_DRAWDOWN_LIMIT_PCT, Decimal("0.06"))
        self.assertEqual(ORDER_REFRESH_SECONDS, 7200)
        self.assertEqual(RISK_STATE_PERSIST_SECONDS, 5)

    def test_configs_are_disabled_spot_and_btc_eth_only(self):
        for portfolio in PORTFOLIOS.values():
            config = build_live_config(portfolio, self.prices, Decimal("0.001"))
            validate_live_config(config)
            self.assertEqual(config["exchange"], "binance")
            self.assertFalse(config["trading_enabled"])
            self.assertEqual([pair.split("-")[0] for pair in config["trading_pairs"]], ["BTC", "ETH"])
            self.assertNotIn("perpetual", str(config).lower())
            expected_budget = FDUSD_BUDGET if portfolio.quote_asset == "FDUSD" else USDT_BUDGET
            self.assertEqual(Decimal(str(config["capital_limit_quote"])), expected_budget.capital_limit)
            self.assertEqual(Decimal(str(config["pair_budget_quote"])), expected_budget.pair_budget)
            self.assertEqual(Decimal(str(config["pair_drawdown_limit_pct"])), Decimal("0.03"))
            self.assertEqual(Decimal(str(config["portfolio_drawdown_limit_pct"])), Decimal("0.06"))
            self.assertEqual(config["order_refresh_time"], 7200)
            self.assertEqual(config["risk_state_persist_seconds"], 5)
            if portfolio.quote_asset == "FDUSD":
                self.assertTrue(config["macro_gate_enabled"])
                self.assertTrue(config["macro_fail_closed"])
                self.assertEqual(config["macro_gate_max_age_seconds"], 150)
                self.assertTrue(config["technical_buy_gate_enabled"])
                self.assertTrue(config["technical_buy_fail_closed"])
                self.assertEqual(config["technical_buy_gate_max_age_seconds"], 150)
            else:
                self.assertFalse(config["macro_gate_enabled"])

    def test_fdusd_live_identity_is_fixed_and_has_no_timestamp(self):
        portfolio = PORTFOLIOS["FDUSD"]
        self.assertEqual(portfolio.bot_name, "grid-live-fdusd-400")
        self.assertEqual(portfolio.profile_name, "binance_live_grid_fdusd_400")
        self.assertEqual(
            portfolio.config_name,
            "walk_forward_portfolio_grid_live_fdusd_400.yml",
        )
        client = ApiClient.__new__(ApiClient)
        calls = []
        client.request = lambda method, path, payload: calls.append(
            (method, path, payload)
        ) or {"unique_instance_name": portfolio.bot_name}
        client.deploy(portfolio.profile_name, portfolio.config_name)
        self.assertEqual(
            "/bot-orchestration/deploy-v2-script?use_timestamp=false",
            calls[0][1],
        )

    def test_runtime_authority_is_validated_instance_config(self):
        source = (SCRIPTS / "walk_forward_portfolio_grid_live.py").read_text(
            encoding="utf-8"
        )
        on_tick = source.split("def on_tick(self):", 1)[1].split(
            "def _poll_macro_gate", 1
        )[0]
        self.assertIn("if not self.config.trading_enabled:", on_tick)
        self.assertNotIn("GRID_LIVE_TRADING_ENABLED", on_tick)

    def test_two_portfolios_reserve_shared_account_once(self):
        balances = required_balances(self.prices)
        self.assertEqual(balances["USDT"], Decimal("262.50"))
        self.assertEqual(balances["FDUSD"], Decimal("220"))
        expected_btc = (
            USDT_BUDGET.side_budget / self.prices["BTC-USDT"]
            + FDUSD_BUDGET.side_budget / self.prices["BTC-FDUSD"]
        )
        self.assertEqual(balances["BTC"], expected_btc)

    def test_fdusd_quote_only_bootstrap_requires_no_initial_base(self):
        balances = required_balances(self.prices, quote_only_fdusd=True)
        self.assertEqual(balances, {"FDUSD": Decimal("420")})
        plan = build_fdusd_bootstrap_plan(self.prices)
        self.assertEqual(plan["minimum_balance"], "420")
        self.assertEqual(plan["recommended_balance"], str(FDUSD_RECOMMENDED_BALANCE))
        self.assertEqual(plan["purchases"]["BTC-FDUSD"]["quote_amount"], "100")
        self.assertEqual(plan["expected_remaining_strategy_fdusd"], "220")
        self.assertFalse(plan["automatic_rollback"])

    def test_fee_floor_covers_round_trip_and_buffer(self):
        self.assertEqual(effective_take_profit(Decimal("0")), Decimal("0.008"))
        self.assertEqual(effective_take_profit(Decimal("0.003")), Decimal("0.010"))

    def test_ledger_tracks_only_bot_inventory_delta(self):
        ledger = PairLedger.create("BTC-USDT", Decimal("0.002"))
        ledger.apply_fill("BUY", Decimal("65000"), Decimal("0.0001"), Decimal("0.0065"))
        self.assertEqual(ledger.inventory_delta(), Decimal("0.0001"))
        ledger.apply_fill("SELL", Decimal("65500"), Decimal("0.0001"), Decimal("0.00655"))
        self.assertEqual(ledger.inventory_delta(), Decimal("0"))
        self.assertEqual(ledger.buys + ledger.sells, 2)
        restored = PairLedger.from_mapping({
            **ledger.__dict__,
            "open_order_ids": ["owned-order"],
        })
        self.assertEqual(restored.inventory_delta(), Decimal("0"))
        self.assertEqual(restored.open_order_ids, {"owned-order"})
        self.assertEqual(restored.peak_equity, ledger.peak_equity)

    def test_old_ledger_state_defaults_pair_peak_to_initial_pair_budget(self):
        restored = PairLedger.from_mapping({
            "trading_pair": "BTC-FDUSD",
            "initial_quote": "100",
            "initial_base": "0.002",
            "quote": "100",
            "base": "0.002",
        })
        self.assertEqual(restored.peak_equity, Decimal("200"))

    def test_active_selection_is_bounded_and_fee_adjusted(self):
        payload = {
            "schema_version": ACTIVE_SELECTION_SCHEMA_VERSION,
            "parameter_version": "weekly-2026-07-20",
            "trading_pairs": ["BTC-FDUSD", "ETH-FDUSD"],
            "parameters": {
                "half_range": 0.04,
                "minimum_spread": 0.008,
                "take_profit": 0.006,
                "move_threshold": 0.02,
                "min_grid_move_seconds": 1800,
            },
        }
        params = validate_active_selection(payload, Decimal("0.002"))
        self.assertEqual(params["grid_range"], Decimal("0.08"))
        self.assertEqual(params["grid_levels"], 10)
        self.assertEqual(params["take_profit"], Decimal("0.008"))

    def test_rolling_windows_use_30_days_then_14_days(self):
        day = 86400
        start = 1_700_000_000
        windows = rolling_validation_windows(start, start + 70 * day)
        self.assertGreaterEqual(len(windows), 5)
        first = windows[0]
        second = windows[1]
        self.assertEqual(first[1] - first[0], 30 * day)
        self.assertEqual(second[1] - second[0], 14 * day)
        self.assertEqual(first[3] - first[2], 7 * day)
        self.assertEqual(second[3] - second[2], 7 * day)
        self.assertLessEqual(first[1], first[2])
        self.assertLessEqual(second[1], second[2])

    def test_weekly_cutoff_waits_until_monday_0010_shanghai(self):
        before = int(pd.Timestamp("2026-07-20 00:09:00", tz="Asia/Shanghai").timestamp())
        after = int(pd.Timestamp("2026-07-20 00:10:00", tz="Asia/Shanghai").timestamp())
        cutoff_before, _ = weekly_cutoff(before)
        cutoff_after, run_after = weekly_cutoff(after)
        self.assertEqual(
            pd.Timestamp(cutoff_before, unit="s", tz="Asia/Shanghai").date().isoformat(),
            "2026-07-13",
        )
        self.assertEqual(
            pd.Timestamp(cutoff_after, unit="s", tz="Asia/Shanghai").date().isoformat(),
            "2026-07-20",
        )
        self.assertEqual(run_after - cutoff_after, 600)

    def test_weighted_order_book_slippage(self):
        average, slippage, base = weighted_ask(
            [["100", "1"], ["100.10", "1"]],
            Decimal("150"),
        )
        self.assertGreater(average, Decimal("100"))
        self.assertLess(slippage, Decimal("0.001"))
        self.assertGreater(base, Decimal("1.49"))

    @staticmethod
    def candles(start_price: float, count: int = 600, decline: float = 0.0) -> pd.DataFrame:
        rows = []
        for index in range(count):
            price = start_price * (1 - decline * index / max(count - 1, 1))
            rows.append({"timestamp": 1_700_000_000 + index * 300, "open": price,
                         "high": price * 1.002, "low": price * 0.998, "close": price, "volume": 10})
        return pd.DataFrame(rows)

    def test_flat_market_stays_within_capital(self):
        candles = {"BTC-USDT": self.candles(65000), "ETH-USDT": self.candles(3500)}
        result, _, pair_stats = simulate(candles, Candidate(0.04, 0.008, 0.008, 0.02), 0.001)
        self.assertLessEqual(result["initial_equity"], 500)
        self.assertFalse(result["liquidated"])
        self.assertTrue(all(abs(stats["inventory_delta"]) < 1 for stats in pair_stats.values()))

    def test_fdusd_simulation_uses_400_strategy_plus_20_reserve(self):
        candles = {
            "BTC-FDUSD": self.candles(65000),
            "ETH-FDUSD": self.candles(3500),
        }
        result, _, _ = simulate(
            candles, Candidate(0.04, 0.008, 0.008, 0.02), 0.0, taker_fee=0.001
        )
        self.assertEqual(result["initial_equity"], 420)
        self.assertFalse(result["liquidated"])

    def test_simulation_models_configured_order_lifetime(self):
        candles = {
            "BTC-FDUSD": self.candles(65000, count=50),
            "ETH-FDUSD": self.candles(3500, count=50),
        }
        result, _, _ = simulate(
            candles,
            Candidate(0.04, 0.008, 0.008, 0.02),
            0.0,
            order_refresh_seconds=7200,
        )
        self.assertEqual(result["order_refresh_seconds"], 7200)
        with self.assertRaisesRegex(ValueError, "must be positive"):
            simulate(candles, Candidate(0.04, 0.008, 0.008, 0.02), 0.0,
                     order_refresh_seconds=0)

    def test_technical_risk_off_suppresses_only_new_buys(self):
        candles = {
            "BTC-FDUSD": self.candles(65000, count=50),
            "ETH-FDUSD": self.candles(3500, count=50),
        }
        timeline = {
            int(timestamp): False
            for timestamp in candles["BTC-FDUSD"].timestamp
        }
        result, _, pair_stats = simulate(
            candles,
            Candidate(0.04, 0.008, 0.008, 0.02),
            0.0,
            technical_buy_gate=timeline,
        )
        self.assertTrue(result["technical_buy_gate_enabled"])
        self.assertEqual(result["technical_risk_off_bars"], 49)
        self.assertTrue(all(metrics["buys"] == 0 for metrics in pair_stats.values()))

    def test_large_decline_trips_before_loss_exceeds_design_by_wide_margin(self):
        candles = {"BTC-USDT": self.candles(65000, decline=0.30),
                   "ETH-USDT": self.candles(3500, decline=0.30)}
        result, _, _ = simulate(candles, Candidate(0.04, 0.008, 0.008, 0.02), 0.001, slippage=0.001)
        self.assertTrue(result["liquidated"])
        self.assertLessEqual(result["max_drawdown_pct"], -0.06)
        self.assertGreater(result["net_pnl_quote"], -70)

    def test_exploratory_backtest_can_disable_breakers_without_changing_defaults(self):
        candles = {
            "BTC-FDUSD": self.candles(65000, decline=0.30),
            "ETH-FDUSD": self.candles(3500, decline=0.30),
        }
        default_result, default_curve, _ = simulate(
            candles, Candidate(0.04, 0.008, 0.008, 0.02), 0.001
        )
        exploratory, full_curve, _ = simulate(
            candles,
            Candidate(0.04, 0.008, 0.008, 0.02),
            0.001,
            risk_breakers_enabled=False,
        )
        self.assertTrue(default_result["risk_breakers_enabled"])
        self.assertTrue(default_result["liquidated"])
        self.assertFalse(exploratory["risk_breakers_enabled"])
        self.assertFalse(exploratory["liquidated"])
        self.assertLess(len(default_curve), len(full_curve))
        self.assertEqual(len(full_curve), len(candles["BTC-FDUSD"]) - 1)

    def test_guard_pnl_excludes_unreserved_main_account_inventory(self):
        rows = [("BUY", 65_000 * 1_000_000, 0.001 * 1_000_000, 0.05 * 1_000_000),
                ("SELL", 66_000 * 1_000_000, 0.001 * 1_000_000, 0.05 * 1_000_000)]
        pnl, net_base = fill_pnl(rows, Decimal("66000"))
        self.assertEqual(net_base, Decimal("0.000"))
        self.assertEqual(pnl, Decimal("0.900"))

    def test_current_portfolio_rows_use_available_not_locked_units(self):
        from grid_live_common import extract_balances

        payload = {
            "binance_live_grid_fdusd_400": {
                "binance": [
                    {"token": "FDUSD", "units": 430.69, "available_units": 430.69},
                    {"token": "USDT", "units": 765.43, "available_units": 596.38},
                ]
            }
        }
        balances = extract_balances(payload)
        self.assertEqual(Decimal("430.69"), balances["FDUSD"])
        self.assertEqual(Decimal("596.38"), balances["USDT"])

    def test_guard_uses_persisted_peak_drawdown(self):
        peak, drawdown = peak_drawdown(Decimal("423"), Decimal("450"), Decimal("420"))
        self.assertEqual(peak, Decimal("450"))
        self.assertEqual(drawdown, Decimal("0.06"))

    def test_guard_does_not_repeat_completed_flatten_pair(self):
        guard = Guard.__new__(Guard)
        calls = []
        guard.api = type("Api", (), {
            "market": lambda self, *args: calls.append(args) or {"ok": True}
        })()
        guard.save = lambda: None
        bot = {"flatten": {"BTC-FDUSD": {"ok": True}}}
        snapshot = {
            "pairs": {
                "BTC-FDUSD": {"net_base": "0.001", "mark": "65000"},
                "ETH-FDUSD": {"net_base": "0", "mark": "3500"},
            }
        }
        results = guard.flatten_deltas("FDUSD", snapshot, bot)
        self.assertEqual({"ok": True}, results["BTC-FDUSD"])
        self.assertEqual("dust", results["ETH-FDUSD"])
        self.assertEqual([], calls)

    def test_guard_emergency_stop_cancels_exchange_and_stops_container(self):
        guard = Guard.__new__(Guard)
        guard.api = type("Api", (), {
            "stop": lambda self, name: {"success": True, "name": name},
            "active_containers": lambda self, name: [{"name": f"{name}-instance"}],
        })()

        class Exchange:
            def __init__(self):
                self.orders = {
                    "BTC-FDUSD": [{"orderId": "btc-order"}],
                    "ETH-FDUSD": [{"orderId": "eth-order"}],
                }
                self.cancelled = []

            def open_orders(self, pair):
                return list(self.orders[pair])

            def cancel_all_orders(self, pair):
                self.cancelled.append(pair)
                self.orders[pair] = []

        class Docker:
            def __init__(self):
                self.live = ["grid-live-fdusd-400-instance"]
                self.stopped = []

            def matching_containers(self, name):
                return list(self.live)

            def stop(self, name):
                self.stopped.append(name)
                self.live.remove(name)
                return {"ok": True}

        guard.emergency_exchange = Exchange()
        guard.emergency_docker = Docker()
        result = guard._secure_stop("FDUSD")
        self.assertEqual(
            ["BTC-FDUSD", "ETH-FDUSD"], guard.emergency_exchange.cancelled
        )
        self.assertEqual(
            ["grid-live-fdusd-400-instance"], guard.emergency_docker.stopped
        )
        self.assertTrue(result["verified_no_active_orders"])
        self.assertTrue(result["verified_no_live_instances"])

    def test_guard_trip_reconciles_after_stop_before_flatten(self):
        guard = Guard.__new__(Guard)
        guard.state = {"bots": {}}
        guard.save = lambda: None
        guard.audit = lambda *args, **kwargs: None
        guard._secure_stop = lambda key: {"stopped": key}
        post_stop = {
            "pnl": "0",
            "pairs": {
                "BTC-FDUSD": {"net_base": "0", "mark": "65000"},
                "ETH-FDUSD": {"net_base": "0", "mark": "3500"},
            },
        }
        snapshots = iter([post_stop, post_stop])
        guard.snapshot = lambda key: next(snapshots)
        flattened = []
        guard.flatten_deltas = (
            lambda key, snapshot, bot: flattened.append(snapshot) or {}
        )
        stale = {
            "pairs": {
                "BTC-FDUSD": {"net_base": "0.01", "mark": "65000"},
                "ETH-FDUSD": {"net_base": "1", "mark": "3500"},
            }
        }
        guard.trip("FDUSD", "test", stale)
        self.assertEqual([post_stop], flattened)
        self.assertTrue(
            guard.state["bots"][PORTFOLIOS["FDUSD"].bot_name]["action_complete"]
        )

    def test_emergency_fill_metrics_accounts_for_base_fee(self):
        metrics = Guard._emergency_fill_metrics("BTC-FDUSD", "SELL", {
            "executedQty": "0.01",
            "cummulativeQuoteQty": "650",
            "fills": [{
                "commission": "0.00001",
                "commissionAsset": "BTC",
                "price": "65000",
            }],
        })
        self.assertEqual(metrics["base_delta"], "-0.01001")
        self.assertEqual(metrics["quote_cashflow"], "650")
        self.assertEqual(metrics["fee_quote"], "0.65000")

    def test_guard_monitors_docker_instance_when_mqtt_status_omits_bot(self):
        guard = Guard.__new__(Guard)
        guard.portfolio_keys = ["FDUSD"]
        guard.manifest_path = Path("missing-grid-live-manifest.json")
        guard.manifest = {"reservations": {}}
        guard.api = type("Api", (), {"status": lambda self: {}})()
        guard.emergency_docker = type("Docker", (), {
            "matching_containers": lambda self, name: [f"{name}-instance"]
        })()
        guard.state = {"bots": {}, "first_failure_at": None}
        guard.save = lambda: None
        guard.snapshot = lambda key: {
            "pnl": "0",
            "pairs": {
                "BTC-FDUSD": {"pnl": "0", "net_base": "0", "mark": "65000"},
                "ETH-FDUSD": {"pnl": "0", "net_base": "0", "mark": "3500"},
            },
        }
        guard.publish_technical_buy_gate = lambda: {"buy_enabled": True}
        guard.trip = lambda *args, **kwargs: self.fail("safe snapshot must not trip")
        guard.cycle()
        bot = guard.state["bots"][PORTFOLIOS["FDUSD"].bot_name]
        self.assertEqual("0", bot["latest"]["pnl"])

    def test_live_deployment_requires_running_fresh_emergency_guard(self):
        ready = guard_readiness(
            {"emergency_ready": True, "last_success_at": 990,
             "armed": True, "shadow": False},
            [{"name": "grid-live-guard", "status": "running"}],
            now=1000,
        )
        self.assertTrue(all(ready.values()))
        stale = guard_readiness(
            {"emergency_ready": True, "last_success_at": 900,
             "armed": True, "shadow": False},
            [{"name": "grid-live-guard", "status": "running"}],
            now=1000,
        )
        self.assertFalse(stale["guard_observation_fresh"])
        missing = guard_readiness({}, [], now=1000)
        self.assertFalse(any(missing.values()))

    def test_no_go_override_requires_both_flag_and_exact_second_confirmation(self):
        validation = {"validation_decision": "NO-GO"}
        self.assertEqual(
            (False, False),
            validation_authorization(
                validation, accept_no_go=True, override_confirmation="wrong"
            ),
        )
        self.assertEqual(
            (True, True),
            validation_authorization(
                validation,
                accept_no_go=True,
                override_confirmation=NO_GO_OVERRIDE_CONFIRMATION,
            ),
        )

    def test_profit_peak_drawdown_trips_exactly_at_six_percent(self):
        peak, below = peak_drawdown(Decimal("423.01"), Decimal("450"), Decimal("420"))
        _, at_limit = peak_drawdown(Decimal("423"), peak, Decimal("420"))
        self.assertLess(below, PORTFOLIO_DRAWDOWN_LIMIT_PCT)
        self.assertEqual(PORTFOLIO_DRAWDOWN_LIMIT_PCT, at_limit)

    def test_bootstrap_receipt_requires_two_filled_buys_near_100_fdusd(self):
        receipt = {
            "schema_version": "fdusd-bootstrap-receipt-v1",
            "source": "binance-signed-api",
            "profile": "binance_live_grid_fdusd_400",
            "orders": {
                "BTC-FDUSD": {"side": "BUY", "status": "FILLED", "order_id": "1",
                               "executed_base": "0.00156", "quote_spent": "99.86"},
                "ETH-FDUSD": {"side": "BUY", "status": "FILLED", "order_id": "2",
                               "executed_base": "0.052", "quote_spent": "99.90"},
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "receipt.json"
            path.write_text(json.dumps(receipt), encoding="utf-8")
            loaded, digest = load_bootstrap_receipt(
                path, "binance_live_grid_fdusd_400"
            )
            self.assertEqual(receipt, loaded)
            self.assertEqual(64, len(digest))


if __name__ == "__main__":
    unittest.main()

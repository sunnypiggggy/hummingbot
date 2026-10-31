"""Stopped, verified Grid valuation refreshes without any recovery action."""
from copy import deepcopy
from decimal import Decimal
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

try:
    from grid_live_guard import Guard, PORTFOLIOS
    import simple_audit
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from live_guard.grid_live_guard import Guard, PORTFOLIOS
    from live_guard import simple_audit


NOW = 2_000_000_000
BOT = PORTFOLIOS["FDUSD"].bot_name


class GridStoppedValuationTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.database = self.root / "grid.sqlite"
        self.database.write_bytes(b"unchanged database")
        self.runtime_path = self.root / "live_grid_runtime_state.json"
        self.runtime_path.write_text(json.dumps({
            "schema_version": 12,
            "ledgers": {pair: {"initial_base": base, "base": base, "quote": "170"}
                        for pair, base in (("BTC-FDUSD", "0.0001"), ("ETH-FDUSD", "0.01"))},
        }), encoding="utf-8")
        self.prices = {"BTC-FDUSD": Decimal("82000"), "ETH-FDUSD": Decimal("2400")}

    def guard(self):
        guard = Guard.__new__(Guard)
        guard.portfolio_keys = ["FDUSD"]
        guard.manifest_path = self.root / "absent-manifest.json"
        guard.state = {"bots": {BOT: {
            "tripped": True, "action_complete": True, "stop_complete": True,
            "stop": {"verified_no_active_orders": True, "verified_no_live_instances": True},
            "peak_equity": "420", "tripped_at": NOW - 500,
            "reason": "existing completed exit", "flatten": {"BTC-FDUSD": "dust", "ETH-FDUSD": "dust"},
            "latest": {"observed_at": NOW - 400, "pnl": "old"},
        }}, "combined_peak_equity": "920"}
        guard.api = Mock()
        guard.api.status.return_value = {}
        guard.emergency_docker = Mock()
        guard.emergency_docker.matching_containers.return_value = []
        guard.emergency_exchange = Mock()
        guard.database = Mock(return_value=self.database)
        guard.price = Mock(side_effect=lambda pair: self.prices[pair])
        guard.fdusd_external_breakers_enabled = True
        for name in ("_refresh_emergency_readiness", "_record_read_retry_events", "publish_technical_buy_gate",
                     "save", "trip", "_secure_stop", "flatten_deltas", "_effective_order_status",
                     "_monitor_order_execution"):
            setattr(guard, name, Mock())
        guard._stuck_recoverable_exit = Mock(return_value=(False, ""))
        return guard

    def assert_no_actions(self, guard):
        for action in (guard.trip, guard._secure_stop, guard.flatten_deltas,
                       guard.api.stop, guard.emergency_exchange.market_order,
                       guard.emergency_exchange.cancel_all_orders):
            action.assert_not_called()
        if guard.emergency_docker is not None:
            guard.emergency_docker.stop.assert_not_called()

    def test_each_cycle_reads_real_ledger_and_current_prices_without_changing_latch(self):
        guard = self.guard()
        original_bot = deepcopy(guard.state["bots"][BOT])
        source = self.runtime_path.read_bytes()
        database = self.database.read_bytes()
        with patch.object(sys.modules[Guard.__module__].time, "time", return_value=NOW):
            guard.cycle()
        first = deepcopy(guard.state["bots"][BOT]["latest"])
        self.assertEqual(NOW, first["observed_at"])
        self.assertEqual(Decimal("392.2"), Decimal(first["equity"]))
        self.assertEqual(Decimal("170"), Decimal(first["pairs"]["BTC-FDUSD"]["quote_balance"]))
        self.prices["BTC-FDUSD"] = Decimal("92000")
        self.prices["ETH-FDUSD"] = Decimal("3000")
        with patch.object(sys.modules[Guard.__module__].time, "time", return_value=NOW + 60):
            guard.cycle()
        latest = guard.state["bots"][BOT]["latest"]
        self.assertEqual(NOW + 60, latest["observed_at"])
        self.assertEqual(Decimal("399.2"), Decimal(latest["equity"]))
        self.assertNotEqual(first["drawdown_pct"], latest["drawdown_pct"])
        self.assertEqual("92000", latest["pairs"]["BTC-FDUSD"]["mark"])
        self.assertEqual(4, guard.price.call_count)
        self.assertEqual(source, self.runtime_path.read_bytes())
        self.assertEqual(database, self.database.read_bytes())
        self.assertEqual({k: v for k, v in original_bot.items() if k != "latest"},
                         {k: v for k, v in guard.state["bots"][BOT].items() if k != "latest"})
        guard._effective_order_status.assert_not_called()
        guard._monitor_order_execution.assert_not_called()
        guard._stuck_recoverable_exit.assert_not_called()
        self.assert_no_actions(guard)

    def test_price_failure_preserves_previous_snapshot_and_never_replays_actions(self):
        guard = self.guard()
        before = deepcopy(guard.state["bots"][BOT])
        guard.price.side_effect = RuntimeError("public price unavailable")
        with self.assertRaisesRegex(RuntimeError, "public price unavailable"):
            guard.cycle()
        self.assertEqual(before, guard.state["bots"][BOT])
        self.assert_no_actions(guard)

    def test_missing_database_preserves_previous_snapshot(self):
        guard = self.guard()
        before = deepcopy(guard.state["bots"][BOT])
        guard.database.return_value = None
        with self.assertRaisesRegex(RuntimeError, "Missing SQLite"):
            guard.cycle()
        self.assertEqual(before, guard.state["bots"][BOT])
        self.assert_no_actions(guard)

    def test_incomplete_or_unknown_stop_evidence_does_not_refresh(self):
        for fault in ("unlatched", "pending_exit", "pending_stop", "orders_unverified",
                      "instances_unverified", "missing_docker", "unknown_docker", "unknown_bot"):
            with self.subTest(fault=fault):
                guard = self.guard()
                bot = guard.state["bots"][BOT]
                if fault == "unlatched": bot["tripped"] = False
                elif fault == "pending_exit": bot["action_complete"] = False
                elif fault == "pending_stop": bot["stop_complete"] = False
                elif fault == "orders_unverified": bot["stop"]["verified_no_active_orders"] = False
                elif fault == "instances_unverified": bot["stop"].pop("verified_no_live_instances")
                elif fault == "missing_docker": guard.emergency_docker = None
                elif fault == "unknown_docker": guard.emergency_docker.matching_containers.return_value = None
                elif fault == "unknown_bot": guard.state["bots"].clear()
                before = deepcopy(guard.state["bots"])
                guard.cycle()
                self.assertEqual(before, guard.state["bots"])
                guard.database.assert_not_called()
                self.assert_no_actions(guard)

    def test_docker_read_error_cannot_fabricate_a_fresh_snapshot(self):
        guard = self.guard()
        before = deepcopy(guard.state["bots"][BOT])
        guard.emergency_docker.matching_containers.side_effect = RuntimeError("Docker read unavailable")
        with self.assertRaisesRegex(RuntimeError, "Docker read unavailable"):
            guard.cycle()
        self.assertEqual(before, guard.state["bots"][BOT])
        guard.database.assert_not_called()
        self.assert_no_actions(guard)

    def test_stopped_valuation_is_excluded_from_combined_risk_with_another_live_portfolio(self):
        guard = self.guard()
        guard.portfolio_keys = ["FDUSD", "USDT"]
        live_bot = PORTFOLIOS["USDT"].bot_name
        guard.api.status.return_value = {live_bot: "running"}
        guard.emergency_docker.matching_containers.side_effect = lambda name: [name] if name == live_bot else []
        original_snapshot = guard.snapshot
        guard.snapshot = Mock(side_effect=lambda key: original_snapshot(key) if key == "FDUSD" else {
            "pnl": "0", "pairs": {pair: {"pnl": "0"} for pair in PORTFOLIOS["USDT"].pairs},
        })
        guard.cycle()
        self.assertEqual("0", guard.state["combined_pnl"])
        self.assertEqual("920", guard.state["combined_peak_equity"])
        self.assertLess(Decimal(guard.state["bots"][BOT]["latest"]["pnl"]), Decimal("-24"))
        guard._stuck_recoverable_exit.assert_not_called()
        guard._monitor_order_execution.assert_not_called()
        self.assert_no_actions(guard)

    def test_running_container_keeps_existing_monitoring_path(self):
        guard = self.guard()
        guard.fdusd_external_breakers_enabled = False
        guard.emergency_docker.matching_containers.return_value = [BOT]
        guard.cycle()
        guard._effective_order_status.assert_called_once()
        guard._monitor_order_execution.assert_called_once()
        guard._stuck_recoverable_exit.assert_called_once()
        self.assertNotEqual("0", guard.state["combined_pnl"])

    def test_refreshed_stopped_grid_is_usable_by_current_simple_audit(self):
        guard = self.guard()
        with patch.object(sys.modules[Guard.__module__].time, "time", return_value=NOW):
            guard.cycle()
        dca = {"bots": {"dca-live-" + pair.replace("-", "").lower() + "-200": {
            "latest": {"observed_at": NOW, "quote_balance": "20", "owned_base": base}}
            for pair, base in (("BTC-USDT", ".001"), ("ETH-USDT", ".02"))}}
        balance = {"account": "fake-account", "observed_at": NOW, "assets": {
            asset: {"free": value, "locked": "0", "total": value}
            for asset, value in (("BTC", ".0011"), ("ETH", ".03"), ("FDUSD", "400"), ("USDT", "50"))}}
        inventory = {"account_fingerprint": "fake-account", "generated_at": NOW, "sources_healthy": True,
                     "assets": {asset: {"owned_total": balance["assets"][asset]["total"],
                                        "exchange": {"total": balance["assets"][asset]["total"]}}
                                for asset in ("BTC", "ETH")}}
        result = simple_audit.collect(balance, guard.state, dca, inventory,
            {"observed_at": NOW, "prices": {"BTC": "82000", "ETH": "2400", "FDUSD": "1", "USDT": "1"}}, NOW)
        self.assertEqual([], result["issues"])
        self.assertEqual(4, len(result["robots"]))
        self.assertTrue(all(row["equity"] is not None for row in result["robots"]))
        self.assertIsNotNone(simple_audit.comparison_value(result))
        self.assert_no_actions(guard)


if __name__ == "__main__":
    unittest.main()

"""A latch-only reset preserves stopped valuation and future emergency actions."""
from copy import deepcopy
from decimal import Decimal
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from live_guard import grid_live_guard as module  # noqa: E402


NOW = 2_000_000_000
BOT = "grid-live-fdusd-400"


class GridGuardResetReceiptTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.database = self.root / "grid.sqlite"
        self.database.write_bytes(b"unchanged database")
        self.runtime_path = self.root / "live_grid_runtime_state.json"
        self.runtime = {
            "schema_version": 13,
            "portfolio_recovery": {
                "phase": "EXITING", "triggered_at": NOW - 1000,
                "mechanism": "portfolio_drawdown_breaker", "scope": "portfolio",
                "exit_completed_at": None, "cooldown_until": None, "healthy_cycles": 0,
            },
            "pending_flatten": ["BTC-FDUSD", "ETH-FDUSD"],
            "ledgers": {pair: {"initial_base": base, "base": base, "quote": "170"}
                        for pair, base in (("BTC-FDUSD", "0.00000971"), ("ETH-FDUSD", "0.000076"))},
        }
        self.write_runtime()
        self.prices = {"BTC-FDUSD": Decimal("82000"), "ETH-FDUSD": Decimal("2400")}
        self.guard = module.Guard.__new__(module.Guard)
        self.guard.portfolio_keys = ["FDUSD"]
        self.guard.manifest_path = self.root / "absent-manifest.json"
        self.guard.state = {"bots": {BOT: {
            "tripped": False, "action_complete": False, "stop_complete": False,
            "peak_equity": "420", "latest": {"observed_at": NOW - 400, "pnl": "old"},
            "emergency_adjustments": [{"pair": "BTC-FDUSD", "base_delta": "0",
                                       "quote_cashflow": "1", "fee_quote": ".2"}],
        }}, "combined_peak_equity": "920", "last_success_at": NOW - 400}
        self.bot = self.guard.state["bots"][BOT]
        self.receipt = {
            "schema": "grid-guard-latch-reset-v1", "scope": "grid_guard_latch_only", "bot": BOT,
            "reset_id": "offline-reset-1", "event_id": "c" * 64,
            "event_sha256": "a" * 64, "preflight_sha256": "b" * 64,
            "performed_at": NOW - 10, "runtime_sha256": module.canonical_sha256(self.runtime),
            "runtime_episode": {field: self.runtime["portfolio_recovery"][field]
                                for field in ("triggered_at", "mechanism", "scope")},
            "prior": {
                "tripped": True, "tripped_at": NOW - 500, "reason": "completed protective stop",
                "action_complete": True, "stop_complete": True,
                "stop": {"verified_no_active_orders": True, "verified_no_live_instances": True,
                         "method": "offline secure stop evidence"},
                "flatten": {"BTC-FDUSD": "dust", "ETH-FDUSD": "dust"},
            },
        }
        self.bot["manual_guard_reset"] = self.receipt
        self.guard.api = Mock()
        self.guard.api.status.return_value = {}
        self.guard.emergency_docker = Mock()
        self.guard.emergency_docker.matching_containers.return_value = []
        self.guard.emergency_exchange = Mock()
        self.guard.database = Mock(return_value=self.database)
        self.guard.price = Mock(side_effect=lambda pair: self.prices[pair])
        self.guard.fdusd_external_breakers_enabled = True
        for name in ("_refresh_emergency_readiness", "_record_read_retry_events", "publish_technical_buy_gate",
                     "save", "trip", "_secure_stop", "flatten_deltas", "_effective_order_status",
                     "_monitor_order_execution", "audit"):
            setattr(self.guard, name, Mock())
        self.guard._stuck_recoverable_exit = Mock(return_value=(False, ""))

    def write_runtime(self):
        self.runtime_path.write_text(json.dumps(self.runtime), encoding="utf-8")

    def cycle(self, now=NOW):
        with patch.object(module.time, "time", return_value=now):
            self.guard.cycle()

    def assert_no_actions(self):
        for action in (self.guard.trip, self.guard._secure_stop, self.guard.flatten_deltas,
                       self.guard.api.stop, self.guard.emergency_exchange.market_order,
                       self.guard.emergency_exchange.cancel_all_orders):
            action.assert_not_called()
        if self.guard.emergency_docker is not None:
            self.guard.emergency_docker.stop.assert_not_called()

    def assert_rejected(self):
        before = deepcopy(self.bot)
        with self.assertRaisesRegex(RuntimeError, "Grid manual guard reset invalid"):
            self.cycle()
        self.assertEqual(before, self.bot)
        self.assertEqual(NOW - 400, self.guard.state["last_success_at"])
        self.guard.save.assert_not_called()
        self.assert_no_actions()

    def test_repeated_cycles_revalue_without_recovery_or_accounting_mutation(self):
        before = deepcopy(self.bot)
        source, database = self.runtime_path.read_bytes(), self.database.read_bytes()
        self.cycle()
        first = deepcopy(self.bot["latest"])
        self.prices["BTC-FDUSD"] = Decimal("92000")
        self.cycle(NOW + 60)
        latest = self.bot["latest"]
        self.assertEqual(NOW + 60, latest["observed_at"])
        self.assertNotEqual(first["equity"], latest["equity"])
        self.assertEqual("170.8", latest["pairs"]["BTC-FDUSD"]["quote_balance"])
        self.assertEqual("0.2", latest["pairs"]["BTC-FDUSD"]["fees_quote"])
        self.assertEqual({k: v for k, v in before.items() if k != "latest"},
                         {k: v for k, v in self.bot.items() if k != "latest"})
        self.assertEqual(source, self.runtime_path.read_bytes())
        self.assertEqual(database, self.database.read_bytes())
        self.assertEqual("920", self.guard.state["combined_peak_equity"])
        self.assertEqual("0", self.guard.state["combined_pnl"])
        self.guard._effective_order_status.assert_not_called()
        self.guard._monitor_order_execution.assert_not_called()
        self.guard._stuck_recoverable_exit.assert_not_called()
        self.assert_no_actions()

    def test_frozen_raw_and_old_receipt_are_not_falsely_expired(self):
        self.cycle(NOW + 7 * 24 * 3600)
        self.assertEqual(NOW + 7 * 24 * 3600, self.bot["latest"]["observed_at"])
        self.assertEqual("EXITING", self.runtime["portfolio_recovery"]["phase"])
        self.assert_no_actions()

    def test_reset_stopped_bot_is_excluded_from_other_live_portfolio_risk(self):
        self.guard.portfolio_keys = ["FDUSD", "USDT"]
        live_bot = module.PORTFOLIOS["USDT"].bot_name
        self.guard.api.status.return_value = {live_bot: "running"}
        self.guard.emergency_docker.matching_containers.side_effect = (
            lambda name: [name] if name == live_bot else []
        )
        original_snapshot = self.guard.snapshot
        self.guard.snapshot = Mock(side_effect=lambda key: original_snapshot(key) if key == "FDUSD" else {
            "pnl": "0", "pairs": {pair: {"pnl": "0"} for pair in module.PORTFOLIOS["USDT"].pairs},
        })
        self.cycle()
        self.assertEqual(NOW, self.bot["latest"]["observed_at"])
        self.assertEqual("0", self.guard.state["combined_pnl"])
        self.assertEqual("920", self.guard.state["combined_peak_equity"])
        self.assert_no_actions()

    def test_receipt_shape_identity_and_hashes_are_strict(self):
        for field, value in (
            ("schema", "other"), ("scope", "resume_trading"), ("bot", "dca-live-btcusdt-200"),
            ("reset_id", ""), ("reset_id", "bad\nID"), ("event_id", None),
            ("event_sha256", "a" * 63), ("event_sha256", True),
            ("preflight_sha256", "g" * 64), ("runtime_sha256", "c" * 64),
        ):
            with self.subTest(field=field, value=value):
                original = self.receipt[field]
                self.receipt[field] = value
                self.assert_rejected()
                self.receipt[field] = original
        for value in (None, [], "receipt"):
            with self.subTest(receipt=value):
                self.bot["manual_guard_reset"] = value
                self.assert_rejected()
        self.bot["manual_guard_reset"] = self.receipt

    def test_current_flags_require_explicit_false(self):
        for field in ("tripped", "action_complete", "stop_complete"):
            for value in (None, "false", 0, 1, True):
                if field == "tripped" and value is True:
                    continue  # A new actual trip uses the original completion path.
                with self.subTest(field=field, value=value):
                    self.bot[field] = value
                    self.assert_rejected()
                    self.bot[field] = False

    def test_prior_completion_and_stop_proof_require_explicit_true(self):
        for mapping, fields in ((self.receipt["prior"], ("tripped", "action_complete", "stop_complete")),
                                (self.receipt["prior"]["stop"], ("verified_no_active_orders",
                                                                 "verified_no_live_instances"))):
            for field in fields:
                for value in (False, None, "true", 1):
                    with self.subTest(field=field, value=value):
                        mapping[field] = value
                        self.assert_rejected()
                        mapping[field] = True
        for field, value in (("stop", None), ("reason", ""), ("flatten", [])):
            with self.subTest(field=field):
                original = self.receipt["prior"][field]
                self.receipt["prior"][field] = value
                self.assert_rejected()
                self.receipt["prior"][field] = original

    def test_episode_and_source_changes_are_rejected(self):
        for field, value in (("triggered_at", NOW - 999), ("mechanism", "another_breaker"),
                             ("scope", "technical")):
            with self.subTest(field=field):
                original = self.receipt["runtime_episode"][field]
                self.receipt["runtime_episode"][field] = value
                self.assert_rejected()
                self.receipt["runtime_episode"][field] = original
        self.runtime["ledgers"]["BTC-FDUSD"]["base"] = "0.00001"
        self.write_runtime()
        self.assert_rejected()

    def test_unreviewed_runtime_schema_is_rejected_even_with_matching_hash(self):
        for value in (12, 14, True, "13", None):
            with self.subTest(schema_version=value):
                self.runtime["schema_version"] = value
                self.write_runtime()
                self.receipt["runtime_sha256"] = module.canonical_sha256(self.runtime)
                self.assert_rejected()

    def test_invalid_timestamps_and_reordered_chronology_are_rejected(self):
        for value in (None, True, "NaN", "Infinity", -1, 0, NOW + 6, NOW - 1001):
            with self.subTest(performed_at=value):
                self.receipt["performed_at"] = value
                self.assert_rejected()
        self.receipt["performed_at"] = NOW - 10
        for value in (None, True, "NaN", "Infinity", 0, NOW - 1001, NOW):
            with self.subTest(tripped_at=value):
                self.receipt["prior"]["tripped_at"] = value
                self.assert_rejected()

    def test_unknown_stop_missing_database_and_read_failure_cannot_fake_freshness(self):
        self.guard.emergency_docker.matching_containers.return_value = None
        self.assert_rejected()
        self.guard.emergency_docker.matching_containers.return_value = []
        self.guard.database.return_value = None
        self.assert_rejected()
        self.guard.database.return_value = self.database
        self.runtime_path.unlink()
        before = deepcopy(self.bot)
        with self.assertRaises(OSError):
            self.cycle()
        self.assertEqual(before, self.bot)
        self.assert_no_actions()

    def test_market_failure_does_not_stamp_old_values_as_fresh(self):
        before = deepcopy(self.bot)
        self.guard.price.side_effect = RuntimeError("public price unavailable")
        with self.assertRaisesRegex(RuntimeError, "public price unavailable"):
            self.cycle()
        self.assertEqual(before, self.bot)
        self.assertEqual(NOW - 400, self.guard.state["last_success_at"])
        self.assert_no_actions()

    def test_runtime_change_between_validation_and_snapshot_is_rejected(self):
        original_snapshot = self.guard.snapshot
        def changed_snapshot(key):
            self.runtime["ledgers"]["ETH-FDUSD"]["quote"] = "171"
            self.write_runtime()
            return original_snapshot(key)
        self.guard.snapshot = Mock(side_effect=changed_snapshot)
        self.assert_rejected()

    def test_original_completed_latch_branch_still_works(self):
        self.bot.pop("manual_guard_reset")
        self.bot.update(deepcopy(self.receipt["prior"]))
        self.cycle()
        self.assertTrue(self.bot["tripped"])
        self.assertEqual(NOW, self.bot["latest"]["observed_at"])
        self.assert_no_actions()

    def test_new_trip_is_not_suppressed_by_archived_completion(self):
        self.guard._secure_stop.return_value = deepcopy(self.receipt["prior"]["stop"])
        self.guard.flatten_deltas.return_value = {}
        with patch.object(module.time, "time", return_value=NOW):
            module.Guard.trip(self.guard, "FDUSD", "new independent fault", self.bot["latest"])
        self.guard._secure_stop.assert_called_once_with("FDUSD")
        self.guard.flatten_deltas.assert_called_once()
        self.assertTrue(self.bot["tripped"])
        self.assertTrue(self.bot["action_complete"])
        self.assertTrue(self.bot["stop_complete"])
        self.assertEqual(NOW, self.bot["tripped_at"])
        self.assertEqual("new independent fault", self.bot["reason"])
        self.assertEqual(NOW - 500, self.receipt["prior"]["tripped_at"])
        self.guard.emergency_exchange.market_order.assert_not_called()
        self.guard.emergency_exchange.cancel_all_orders.assert_not_called()


if __name__ == "__main__":
    unittest.main()

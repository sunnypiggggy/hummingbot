import json
import os
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scheduler"))

from fdusd_live_grid_scheduler import Scheduler  # noqa: E402
from live_guard.telegram_notifications import canonical_sha256  # noqa: E402


class GridMacroSchedulerTest(unittest.TestCase):
    def test_gate_reconcile_interval_is_bounded_to_ten_seconds(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(os.environ, {
                "GRID_LIVE_FDUSD_STATE_PATH": directory,
                "BOTS_PATH": str(Path(directory) / "bots"),
                "GRID_LIVE_GATE_RECONCILE_SECONDS": "5",
            }):
                self.assertEqual(5, Scheduler().reconcile_seconds)
            with patch.dict(os.environ, {
                "GRID_LIVE_FDUSD_STATE_PATH": directory,
                "BOTS_PATH": str(Path(directory) / "bots"),
                "GRID_LIVE_GATE_RECONCILE_SECONDS": "11",
            }):
                with self.assertRaisesRegex(ValueError, "between 1 and 10"):
                    Scheduler()

    def test_fixed_mode_publishes_canonical_selection_without_optimizer(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state_root = root / "grid-state"
            bots = root / "bots"
            macro = root / "macro" / "state.json"
            macro.parent.mkdir(parents=True)
            macro.write_text("{}", encoding="utf-8")
            with patch.dict(os.environ, {
                "GRID_LIVE_FDUSD_STATE_PATH": str(state_root),
                "BOTS_PATH": str(bots),
                "GRID_LIVE_MACRO_STATE_PATH": str(macro),
                "GRID_LIVE_PARAMETER_UPDATES_ENABLED": "false",
                "PARAMETER_EVIDENCE_RECEIPT_ROOT": str(root / "receipts"),
            }):
                scheduler = Scheduler()
                with patch.object(scheduler, "ensure_staging"), patch.object(
                    scheduler, "publish_macro_gate"
                ), patch.object(scheduler, "verified_fees") as fees:
                    scheduler.reconcile()
                pending = json.loads((state_root / "scheduler_state.json").read_text(encoding="utf-8"))
                self.assertEqual("AWAITING_TELEGRAM_EVIDENCE", pending["phase"])
                self.assertFalse((state_root / "active_selection.json").exists())
                parameter_sha = pending["pending_parameter_sha256"]
                receipt = {
                    "schema": "telegram-evidence-delivery-receipt-v1",
                    "identity_sha256": parameter_sha, "source_event_id": pending["pending_event_id"],
                    "release_sha256": "", "model_sha256": "",
                    "parameter_sha256": parameter_sha, "report_request": "grid_360d",
                    "expected_photo_count": 6, "photo_sha256": [str(i) * 64 for i in range(1, 7)],
                    "delivered_at": "2026-08-20T00:00:00+00:00",
                    "telegram_message_ids": ["1"] * 8,
                }
                receipt["delivery_receipt_sha256"] = canonical_sha256(receipt)
                receipt_root = root / "receipts"
                receipt_root.mkdir()
                (receipt_root / f"{parameter_sha}.json").write_text(
                    json.dumps(receipt), encoding="utf-8",
                )
                scheduler.ensure_fixed_selection()
            fees.assert_not_called()
            selection = json.loads(
                (state_root / "active_selection.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                "binance-ai-btc-medium-sideways-eth-long-volatility-v1",
                selection["parameter_version"],
            )
            self.assertEqual(2, selection["schema_version"])
            self.assertEqual("medium_sideways", selection["pair_parameters"]["BTC-FDUSD"]["profile"])
            self.assertEqual("long_volatility", selection["pair_parameters"]["ETH-FDUSD"]["profile"])
            self.assertEqual("ethbtc-forced-exit-live-contract-v1", selection["technical_buy_gate"]["schema"])
            self.assertFalse(selection["technical_buy_gate"]["short_spike_enabled"])
            self.assertFalse(selection["technical_buy_gate"]["mechanism1_runtime_fallback"])
            self.assertEqual(
                0.12698379475402316,
                selection["pair_parameters"]["BTC-FDUSD"]["grid_range"],
            )
            self.assertEqual(18, selection["pair_parameters"]["BTC-FDUSD"]["grid_levels"])
            self.assertEqual(
                0.5246511596640915,
                selection["pair_parameters"]["ETH-FDUSD"]["grid_range"],
            )
            self.assertEqual(10.0, selection["pair_parameters"]["ETH-FDUSD"]["minimum_order_quote"])

    def test_consumer_first_interlock_keeps_schema_v1(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state_root = root / "grid-state"
            bots = root / "bots"
            with patch.dict(os.environ, {
                "GRID_LIVE_FDUSD_STATE_PATH": str(state_root),
                "BOTS_PATH": str(bots),
                "GRID_LIVE_PARAMETER_UPDATES_ENABLED": "false",
                "GRID_PAIR_PARAMETER_SCHEMA_V2_ENABLED": "false",
            }):
                scheduler = Scheduler()
                scheduler.ensure_fixed_selection()
            selection = json.loads(
                (state_root / "active_selection.json").read_text(encoding="utf-8")
            )
            self.assertEqual(1, selection["schema_version"])
            self.assertEqual("fixed-grid-6pct-ethbtc-forced-exit-v22", selection["parameter_version"])
            self.assertEqual(0.03, selection["parameters"]["half_range"])

    def test_scheduler_publishes_active_fomc_gate_to_fixed_instance(self):
        now = datetime.now(timezone.utc)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state_root = root / "grid-state"
            bots = root / "bots"
            macro = root / "macro" / "state.json"
            instance = bots / "instances" / "grid-live-fdusd-400"
            instance.mkdir(parents=True)
            macro.parent.mkdir(parents=True)
            macro.write_text(
                json.dumps({
                    "schema_version": 3,
                    "last_reconcile": now.isoformat(),
                    "leases": {
                        "approved-fomc": {
                            "decision_id": "approved-fomc",
                            "event_id": "fomc-2026-09-16",
                            "event_kind": "fomc",
                            "market_impact": "negative",
                            "effective_at": (now - timedelta(minutes=5)).isoformat(),
                            "resume_at": (now + timedelta(hours=1)).isoformat(),
                            "approval": {"status": "approved", "action": "approve"},
                        }
                    },
                }),
                encoding="utf-8",
            )
            with patch.dict(os.environ, {
                "GRID_LIVE_FDUSD_STATE_PATH": str(state_root),
                "BOTS_PATH": str(bots),
                "GRID_LIVE_MACRO_STATE_PATH": str(macro),
                "GRID_LIVE_FOMC_EXECUTION_ENABLED": "true",
            }):
                scheduler = Scheduler()
                self.assertEqual(1, scheduler.publish_macro_gate())

            canonical = json.loads(
                (state_root / "macro_gate.json").read_text(encoding="utf-8")
            )
            published = json.loads(
                (instance / "data" / "macro_gate.json").read_text(encoding="utf-8")
            )
            self.assertTrue(canonical["pause_new_orders"])
            self.assertEqual(["approved-fomc"], canonical["active_lease_ids"])
            self.assertEqual(canonical, published)


if __name__ == "__main__":
    unittest.main()

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
            }):
                scheduler = Scheduler()
                with patch.object(scheduler, "ensure_staging"), patch.object(
                    scheduler, "publish_macro_gate"
                ), patch.object(scheduler, "verified_fees") as fees:
                    scheduler.reconcile()
            fees.assert_not_called()
            selection = json.loads(
                (state_root / "active_selection.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                "fixed-grid-6pct-roc5-sqz1-recovery-roc1-sqz3-v2",
                selection["parameter_version"],
            )
            self.assertEqual(1.0, selection["technical_buy_gate"]["recovery"]["roc_pct"])
            self.assertEqual(-3.0, selection["technical_buy_gate"]["recovery"]["sqzmom_pct"])
            self.assertEqual(0.03, selection["parameters"]["half_range"])
            self.assertEqual(10, selection["parameters"]["levels"])

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

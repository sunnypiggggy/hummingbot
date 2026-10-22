import json
import os
import sys
import tempfile
import unittest
from contextlib import ExitStack
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scheduler"))

from fdusd_live_grid_scheduler import Scheduler  # noqa: E402
from live_guard.telegram_notifications import canonical_sha256  # noqa: E402


class GridMacroSchedulerTest(unittest.TestCase):
    @staticmethod
    def _evidence_receipt(parameter_sha, **updates):
        receipt = {
            "schema": "telegram-evidence-delivery-receipt-v1",
            "identity_sha256": parameter_sha,
            "parameter_sha256": parameter_sha,
            "report_request": "grid_360d",
            "expected_photo_count": 6,
            "photo_sha256": [str(i) * 64 for i in range(1, 7)],
            **updates,
        }
        receipt["delivery_receipt_sha256"] = canonical_sha256(receipt)
        return receipt

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

    def test_both_parameter_paths_wait_for_valid_evidence_without_duplicate_notifications(self):
        for dynamic in (False, True):
            with self.subTest(dynamic=dynamic), tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
                root = Path(directory)
                stack.enter_context(patch.dict(os.environ, {
                    "GRID_LIVE_FDUSD_STATE_PATH": str(root / "state"),
                    "BOTS_PATH": str(root / "bots"),
                    "PARAMETER_EVIDENCE_RECEIPT_ROOT": str(root / "receipts"),
                    "GRID_LIVE_PARAMETER_UPDATES_ENABLED": str(dynamic).lower(),
                    "GRID_PAIR_PARAMETER_SCHEMA_V2_ENABLED": "true",
                    "V22_WEEKLY_AUTO_UPDATE_ENABLED": "false",
                }))
                scheduler = Scheduler()
                for name in ("ensure_staging", "publish_macro_gate"):
                    stack.enter_context(patch.object(scheduler, name))
                publish = stack.enter_context(patch.object(
                    scheduler, "publish_to_active_instances", return_value=2,
                ))
                write_config = stack.enter_context(patch.object(scheduler, "write_disabled_config"))
                if dynamic:
                    stack.enter_context(patch.object(
                        scheduler, "target", return_value=("weekly-test", 14, 1800000000),
                    ))
                    stack.enter_context(patch.object(scheduler, "verified_fees", return_value=(0.0, 0.0002)))
                    stack.enter_context(patch("fdusd_live_grid_scheduler.load_candles", return_value=pd.DataFrame()))
                    selected = SimpleNamespace(
                        half_range=0.03, min_spread=0.006, take_profit=0.006,
                        move_threshold=0.015, levels=10, move_cooldown_seconds=1800,
                    )
                    evaluations = pd.DataFrame([{"score": 1.0}])
                    evaluations.attrs["eligible_count"] = 1
                    stack.enter_context(patch(
                        "fdusd_live_grid_scheduler.select_candidate", return_value=(selected, evaluations),
                    ))
                    clock = stack.enter_context(patch("fdusd_live_grid_scheduler.datetime", wraps=datetime))
                    clock.now.return_value = datetime(2026, 10, 21, tzinfo=timezone.utc)
                previous = {"schema_version": 1, "parameters": {"levels": 7}}
                scheduler.atomic_json(scheduler.canonical_selection, previous)
                scheduler.reconcile()
                pending = scheduler.read_json(scheduler.state_path, {})
                parameter_sha = pending["pending_parameter_sha256"]
                receipt_path = scheduler.evidence_receipt_root / f"{parameter_sha}.json"
                bad_hash = self._evidence_receipt(parameter_sha)
                bad_hash["delivery_receipt_sha256"] = "0" * 64
                missing_photo = self._evidence_receipt(parameter_sha, photo_sha256=["1" * 64] * 5)
                for receipt in (None, bad_hash, missing_photo):
                    if receipt is not None:
                        scheduler.atomic_json(receipt_path, receipt)
                    scheduler.reconcile()
                    self.assertEqual(previous, scheduler.read_json(scheduler.canonical_selection, {}))
                    self.assertEqual(1, len(scheduler.notification_path.read_text(encoding="utf-8").splitlines()))
                    write_config.assert_not_called()
                    self.assertTrue(all(call.args[0] == previous for call in publish.call_args_list))
                receipt = self._evidence_receipt(parameter_sha)
                scheduler.atomic_json(receipt_path, receipt)
                scheduler.reconcile()
                active = scheduler.read_json(scheduler.canonical_selection, {})
                self.assertEqual(parameter_sha, canonical_sha256(active))
                self.assertFalse(scheduler.read_json(scheduler.state_path, {})["trading_enabled"])
                self.assertEqual(active, publish.call_args.args[0])
                if dynamic:
                    write_config.assert_called_once()
                scheduler.reconcile()
                events = [
                    json.loads(line) for line in scheduler.notification_path.read_text(encoding="utf-8").splitlines()
                ]
                self.assertEqual(
                    ["PARAMETER_APPROVAL_PENDING", "PARAMETER_ACTIVATED"], [e["transition"] for e in events],
                )
                self.assertEqual(
                    receipt["delivery_receipt_sha256"], events[-1]["details"]["evidence_delivery_receipt_sha256"],
                )

    def test_photo_count_error_keeps_short_circuit_and_legacy_bypass(self):
        parameter_sha = "a" * 64
        receipt = self._evidence_receipt(parameter_sha, expected_photo_count="invalid")
        unchanged = dict(receipt)
        with self.assertRaises(ValueError):
            Scheduler._parameter_evidence_receipt(receipt, parameter_sha)
        self.assertEqual(
            (True, receipt["delivery_receipt_sha256"]),
            Scheduler._parameter_evidence_receipt(receipt, parameter_sha, required=False),
        )
        self.assertEqual(unchanged, receipt)
        receipt["schema"] = "unsupported"
        self.assertEqual(
            (False, receipt["delivery_receipt_sha256"]),
            Scheduler._parameter_evidence_receipt(receipt, parameter_sha),
        )

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

import json
import tempfile
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from live_guard.grid_live_guard import Guard
from live_guard.telegram_notifications import RuntimeErrorChannel


class GridExecutionMonitorTest(unittest.TestCase):
    def test_persistent_fault_restart_and_verified_recovery_emit_once(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            guard = Guard.__new__(Guard)
            def channel():
                return RuntimeErrorChannel(event_path=root / "events.jsonl", state_path=root / "errors.json",
                                           source="grid-live-guard", strategy="grid", bot="grid", pair="ETH-FDUSD")
            guard.runtime_errors = channel()
            statuses = {"ETH-FDUSD": {"state": "RESTRICTED", "trading_expected": True,
                                     "reason": "one active order remains"}}
            for now in (1000, 1010, 1016, 1017):
                with patch("live_guard.grid_live_guard.time.time", return_value=now):
                    guard._monitor_order_execution(statuses)
            guard.runtime_errors = channel()  # True on-disk restart, not an in-memory mock.
            with patch("live_guard.grid_live_guard.time.time", return_value=1020):
                guard._monitor_order_execution(statuses)
            self.assertTrue(guard.runtime_errors.state["components"]["grid_order_execution:ETH-FDUSD"]["active"])
            row = statuses["ETH-FDUSD"]
            row.update(state="HEALTHY", actual_sell_layers=5, exchange_orders_verified=False)
            guard._monitor_order_execution(statuses)
            self.assertTrue(guard.runtime_errors.state["components"]["grid_order_execution:ETH-FDUSD"]["active"])
            row.update(exchange_orders_verified=True)
            guard._monitor_order_execution(statuses)
            guard._monitor_order_execution(statuses)
            events = [json.loads(line) for line in (root / "events.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(["ERROR_OCCURRED", "ERROR_RECOVERED"], [event["transition"] for event in events])

    def test_exchange_evidence_wait_grace_and_persistent_fault(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            guard = Guard.__new__(Guard)
            guard.runtime_errors = RuntimeErrorChannel(
                event_path=root / "events.jsonl", state_path=root / "errors.json",
                source="grid-live-guard", strategy="grid", bot="grid", pair="ETH-FDUSD",
            )
            statuses = {"ETH-FDUSD": {"state": "REBUILDING", "trading_expected": True,
                                     "refresh_requested_at": 1000, "reason": "awaiting exchange proof"}}
            for now in (1000, 1015, 1029):
                with patch("live_guard.grid_live_guard.time.time", return_value=now):
                    guard._monitor_order_execution(statuses)
            self.assertFalse((root / "events.jsonl").exists())
            for now in (1030, 1046, 1050):
                with patch("live_guard.grid_live_guard.time.time", return_value=now):
                    guard._monitor_order_execution(statuses)
            events = [json.loads(line) for line in (root / "events.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(["ERROR_OCCURRED"], [event["transition"] for event in events])
            self.assertTrue(all(event["mechanism"] == "runtime_error" for event in events))

    def test_risk_off_is_not_execution_failure_or_proof_of_recovery(self):
        guard = Guard.__new__(Guard)
        guard._monitor_order_execution({"BTC-FDUSD": {"state": "EXPECTED_EMPTY", "trading_expected": False}})


if __name__ == "__main__":
    unittest.main()

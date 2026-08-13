import sys
import tempfile
import unittest
import json
import time
from decimal import Decimal
from pathlib import Path
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from live_guard.grid_live_guard import Guard  # noqa: E402
from ethbtc_forced_exit_contract import MODEL_VERSION, PACKAGE_ID, SCHEMA  # noqa: E402


class GridGuardShadowTest(unittest.TestCase):
    @patch("live_guard.grid_live_guard.atomic_gate_json")
    def test_cached_technical_gate_is_republished_to_new_instance(self, atomic_json):
        guard = Guard.__new__(Guard)
        guard.next_technical_refresh = float("inf")
        gate = {"schema": "grid-xgboost-long-risk-gate-v1",
                "model_version": "xgboost-grid-long-risk-gate-v21-250d", "source_healthy": True}
        guard.state = {"xgboost_risk_gate": gate}
        guard.technical_gate_path = Path("state/xgboost_risk_gate.json")
        target = Path("instance/data/xgboost_risk_gate.json")
        guard._technical_gate_targets = Mock(return_value=[target])

        self.assertEqual(gate, guard.publish_technical_buy_gate())
        atomic_json.assert_called_once_with(target, gate)

    @patch("live_guard.grid_live_guard.atomic_gate_json")
    @patch("live_guard.grid_live_guard.atomic_v22_json")
    @patch("live_guard.grid_live_guard.load_runtime_v22_contract")
    def test_guard_distributes_v22_contract_from_the_single_in_process_producer(
        self, load_gate, atomic_v22_json, atomic_json,
    ):
        with tempfile.TemporaryDirectory() as directory:
            guard = Guard.__new__(Guard)
            guard.next_technical_refresh = 0
            guard.technical_refresh_seconds = 30
            guard.state = {"xgboost_risk_gate": {}}
            guard.technical_gate_path = Path(directory) / "xgboost_risk_gate.json"
            gate = {"schema": SCHEMA, "package_id": PACKAGE_ID,
                    "model_version": MODEL_VERSION, "execution_authorized": True,
                    "release_sha256": "a" * 64, "pairs": {}}
            guard.v22_producer = Mock()
            guard.v22_producer.produce.return_value = gate
            guard.v22_observation_gate_path = Path(directory) / "ethbtc_forced_exit_observation.json"
            guard.v22_execution_mode = "live"
            guard.v21_producer = None
            target = Path(directory) / "instance" / "data" / "xgboost_risk_gate.json"
            guard._technical_gate_targets = Mock(return_value=[guard.technical_gate_path, target])
            guard.audit = Mock()
            load_gate.return_value = {"runtime_gate_healthy": True, "reason": "healthy"}

            self.assertEqual(gate, guard.publish_technical_buy_gate(force=True))
            atomic_json.assert_called_once_with(target, gate)
            atomic_v22_json.assert_called_once_with(guard.technical_gate_path, gate)
            load_gate.assert_called_once_with(guard.v22_observation_gate_path)
            self.assertEqual("v22", guard.state["active_technical_producer"])

    @patch("live_guard.grid_live_guard.atomic_gate_json")
    def test_guard_never_distributes_v21_shadow_contract(self, atomic_json):
        guard = Guard.__new__(Guard)
        guard.next_technical_refresh = float("inf")
        shadow = {
            "schema": "grid-xgboost-long-risk-gate-v2",
            "model_version": "xgboost-grid-long-risk-gate-v21-250d",
            "shadow_mode": True,
            "deployment_allowed": False,
        }
        guard.state = {"xgboost_risk_gate": shadow}
        guard.technical_gate_path = Path("state/xgboost_risk_gate.json")
        guard._technical_gate_targets = Mock(return_value=[
            guard.technical_gate_path, Path("instance/data/xgboost_risk_gate.json")
        ])

        self.assertEqual(shadow, guard.publish_technical_buy_gate())
        atomic_json.assert_not_called()

    def test_shadow_preflight_uses_no_fill_test_orders(self):
        with tempfile.TemporaryDirectory() as directory:
            guard = Guard.__new__(Guard)
            guard.emergency_exchange = Mock()
            guard.emergency_exchange.open_orders.return_value = []
            guard.emergency_exchange.account_balances.return_value = {
                "BTC": {"total": Decimal("0.002")},
                "ETH": {"total": Decimal("0.05")},
            }
            guard.inventory_ledger = Mock()
            guard.inventory_ledger.status_path = Path(directory) / "account_inventory_status.json"
            guard.inventory_ledger.status_path.write_text(json.dumps({
                "schema": "account-inventory-status-v3",
                "generated_at": time.time(),
                "healthy": False,
                "sources_healthy": True,
                "account_fingerprint": "f" * 64,
                "evidence_sha256": "e" * 64,
                "active_order_count": 25,
                "open_order_counts": {
                    "BTC-FDUSD": 4, "ETH-FDUSD": 5,
                    "BTC-USDT": 8, "ETH-USDT": 8,
                },
                "assets": {
                    "BTC": {"ownership_deficit": "0", "owners": {
                        "grid:grid-live-fdusd-400": "0",
                        "dca:dca-live-btcusdt-200": "0.0015",
                    }},
                    "ETH": {"ownership_deficit": "0", "owners": {
                        "grid:grid-live-fdusd-400": "0",
                        "dca:dca-live-ethusdt-200": "0.05",
                    }},
                },
            }), encoding="utf-8")
            guard.emergency_exchange._signed.side_effect = [
                {
                    "enableWithdrawals": False,
                    "ipRestrict": True,
                    "enableFutures": False,
                    "enableMargin": False,
                },
                [{"makerCommission": "0", "takerCommission": "0.001"}],
                {},
                [{"makerCommission": "0", "takerCommission": "0.001"}],
                {},
            ]
            result = guard.verify_shadow_exchange_ready(["BTC-FDUSD", "ETH-FDUSD"])
            self.assertTrue(result["test_order_no_fill"])
            self.assertEqual(5, guard.emergency_exchange._signed.call_count)
            self.assertEqual(
                {"BTC-FDUSD": 0, "ETH-FDUSD": 0},
                result["open_order_counts"],
            )
            self.assertEqual(
                "shared_account_inventory_v3",
                result["ownership_coverage"]["BTC-FDUSD"]["source"],
            )
            self.assertTrue(result["ownership_coverage"]["BTC-FDUSD"]["covered"])
            self.assertFalse(result["inventory_contract"]["overall_healthy"])
            self.assertEqual(25, result["inventory_contract"]["active_order_count"])
            self.assertTrue(
                result["inventory_contract"]["orders_must_be_cancelled_before_liquidation"]
            )
            test_calls = guard.emergency_exchange._signed.call_args_list[2::2]
            self.assertTrue(all(call.args[1] == "/api/v3/order/test" for call in test_calls))
            self.assertEqual("0.001", result["commissions"]["BTC-FDUSD"]["taker_fee"])

    def test_shadow_preflight_rejects_stale_shared_inventory(self):
        with tempfile.TemporaryDirectory() as directory:
            guard = Guard.__new__(Guard)
            guard.emergency_exchange = Mock()
            guard.emergency_exchange.open_orders.return_value = []
            guard.emergency_exchange.account_balances.return_value = {}
            guard.inventory_ledger = Mock()
            guard.inventory_ledger.status_path = Path(directory) / "account_inventory_status.json"
            guard.inventory_ledger.status_path.write_text(json.dumps({
                "schema": "account-inventory-status-v3",
                "generated_at": time.time() - 31,
                "healthy": True,
                "sources_healthy": True,
                "account_fingerprint": "f" * 64,
                "evidence_sha256": "e" * 64,
                "active_order_count": 0,
                "assets": {},
            }), encoding="utf-8")
            guard.emergency_exchange._signed.side_effect = [
                {"enableWithdrawals": False, "ipRestrict": True,
                 "enableFutures": False, "enableMargin": False},
                [{"makerCommission": "0", "takerCommission": "0.001"}], {},
            ]
            with self.assertRaisesRegex(RuntimeError, "shared inventory status is stale"):
                guard.verify_shadow_exchange_ready(["BTC-FDUSD"])

    def test_shadow_preflight_rejects_unhealthy_inventory_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            guard = Guard.__new__(Guard)
            guard.emergency_exchange = Mock()
            guard.emergency_exchange.open_orders.return_value = []
            guard.emergency_exchange.account_balances.return_value = {}
            guard.inventory_ledger = Mock()
            guard.inventory_ledger.status_path = Path(directory) / "account_inventory_status.json"
            guard.inventory_ledger.status_path.write_text(json.dumps({
                "schema": "account-inventory-status-v3",
                "generated_at": time.time(),
                "healthy": False,
                "sources_healthy": False,
                "account_fingerprint": "f" * 64,
                "evidence_sha256": "e" * 64,
                "active_order_count": 25,
                "assets": {},
            }), encoding="utf-8")
            guard.emergency_exchange._signed.side_effect = [
                {"enableWithdrawals": False, "ipRestrict": True,
                 "enableFutures": False, "enableMargin": False},
                [{"makerCommission": "0", "takerCommission": "0.001"}], {},
            ]
            with self.assertRaisesRegex(RuntimeError, "inventory sources are unhealthy"):
                guard.verify_shadow_exchange_ready(["BTC-FDUSD"])

    def test_shadow_trip_only_audits_would_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            guard = Guard.__new__(Guard)
            guard.shadow = True
            guard.state = {"bots": {}}
            guard.state_path = Path(directory) / "guard_state.json"
            guard.audit_path = Path(directory) / "risk_audit.jsonl"
            guard._secure_stop = Mock(side_effect=AssertionError("must not stop"))
            guard.flatten_deltas = Mock(side_effect=AssertionError("must not trade"))
            guard.trip("FDUSD", "test threshold", {"pnl": "-24"})
            bot = guard.state["bots"]["grid-live-fdusd-400"]
            self.assertEqual("test threshold", bot["would_trip"]["reason"])
            guard._secure_stop.assert_not_called()
            guard.flatten_deltas.assert_not_called()
            self.assertIn(
                "grid_circuit_breaker_would_trip",
                guard.audit_path.read_text(encoding="utf-8"),
            )


if __name__ == "__main__":
    unittest.main()

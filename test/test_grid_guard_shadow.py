import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from live_guard.grid_live_guard import Guard  # noqa: E402


class GridGuardShadowTest(unittest.TestCase):
    @patch("live_guard.grid_live_guard.atomic_gate_json")
    def test_cached_technical_gate_is_republished_to_new_instance(self, atomic_json):
        guard = Guard.__new__(Guard)
        guard.next_technical_refresh = float("inf")
        gate = {"source_healthy": True, "buy_enabled": True}
        guard.state = {"technical_buy_gate": gate}
        target = Path("instance/data/technical_buy_gate.json")
        guard._technical_gate_targets = Mock(return_value=[target])

        self.assertEqual(gate, guard.publish_technical_buy_gate())
        atomic_json.assert_called_once_with(target, gate)

    @patch("live_guard.grid_live_guard.atomic_gate_json")
    @patch("live_guard.grid_live_guard.build_technical_buy_gate")
    @patch("live_guard.grid_live_guard.roc_sqz_signal_from_klines")
    @patch("live_guard.grid_live_guard.requests.get")
    def test_recovery_model_migration_forces_latest_closed_bar_re_evaluation(
        self, get, signal, build, atomic_json,
    ):
        server, klines = Mock(), Mock()
        server.json.return_value = {"serverTime": 1_000_000}
        klines.json.return_value = [[0, 1, 2, 0.5, 1, 0, 999_000]]
        get.side_effect = [server, klines]
        signal.return_value = {"bar_close_time": 999_000}
        build.return_value = {
            "source_healthy": True,
            "risk_off_active": True,
            "recovery_rule_version": "combined-roc1-sqz3-improving-v1",
        }
        guard = Guard.__new__(Guard)
        guard.next_technical_refresh = 0
        guard.technical_refresh_seconds = 60
        guard.state = {"technical_buy_gate": {
            "risk_off_active": True,
            "last_evaluated_bar_close_time": 123,
        }}
        guard.roc_risk_off_pct = -5
        guard.sqzmom_risk_off_pct = -1
        guard.roc_recovery_pct = 1
        guard.sqzmom_recovery_pct = -3
        guard._technical_gate_targets = Mock(return_value=[])
        guard.audit = Mock()

        guard.publish_technical_buy_gate(force=True)

        self.assertIsNone(build.call_args.kwargs["previous_bar_close_time"])
        self.assertTrue(guard.audit.call_args.kwargs["model_changed"])
        atomic_json.assert_not_called()

    def test_shadow_preflight_uses_no_fill_test_orders(self):
        guard = Guard.__new__(Guard)
        guard.emergency_exchange = Mock()
        guard.emergency_exchange.open_orders.return_value = []
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
        test_calls = guard.emergency_exchange._signed.call_args_list[2::2]
        self.assertTrue(all(call.args[1] == "/api/v3/order/test" for call in test_calls))
        self.assertEqual("0.001", result["commissions"]["BTC-FDUSD"]["taker_fee"])

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

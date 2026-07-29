import json
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from grid_technical_gate import (  # noqa: E402
    build_technical_buy_gate,
    failed_technical_buy_gate,
    load_runtime_technical_gate,
    roc_sqz_signal_from_klines,
)
from live_guard.dca_live_guard import Guard as DcaGuard  # noqa: E402


NOW = datetime(2026, 7, 28, 6, 0, tzinfo=timezone.utc)


def falling_klines(count: int = 64) -> list[list]:
    rows = []
    for index in range(count):
        close = 100 - index * 1.2
        opened = index * 14_400_000
        rows.append([opened, close + 1, close + 2, close - 2, close, 1, opened + 14_399_999])
    return rows


class GridTechnicalGateTest(unittest.TestCase):
    def test_signal_math_matches_existing_dca_guard(self):
        rows = falling_klines()
        grid = roc_sqz_signal_from_klines(rows)
        dca = DcaGuard._roc_sqz_signal_from_klines(rows)
        for key in ("roc_48h_pct", "sqzmom", "sqzmom_previous", "sqzmom_pct"):
            self.assertAlmostEqual(grid[key], dca[key], places=12)
        self.assertEqual(grid["sqzmom_red"], dca["sqzmom_red"])
        expected_color = (
            "lime" if grid["sqzmom"] > 0 and grid["sqzmom"] > grid["sqzmom_previous"]
            else "green" if grid["sqzmom"] > 0
            else "red" if grid["sqzmom"] < grid["sqzmom_previous"]
            else "maroon"
        )
        self.assertEqual(grid["sqzmom_color"], expected_color)

    def test_combined_threshold_disables_and_combined_signal_recovers(self):
        red = {
            "roc_48h_pct": -8.1,
            "sqzmom_pct": -3.1,
            "sqzmom": -4,
            "sqzmom_previous": -3,
            "sqzmom_color": "red",
            "bar_close_time": 100,
        }
        risk_off = build_technical_buy_gate(red, previously_active=False, now=NOW)
        self.assertFalse(risk_off["buy_enabled"])
        self.assertTrue(risk_off["risk_off_condition"])
        self.assertTrue(risk_off["trigger"])
        duplicate_poll = build_technical_buy_gate(
            red,
            previously_active=True,
            previous_bar_close_time=100,
            previous_sqzmom_color="red",
            now=NOW,
        )
        self.assertFalse(duplicate_poll["buy_enabled"])
        self.assertFalse(duplicate_poll["trigger"])
        recovered = build_technical_buy_gate(
            {**red, "sqzmom": -2, "sqzmom_previous": -4,
             "sqzmom_color": "maroon", "bar_close_time": 200,
             "sqzmom_pct": -2.0, "roc_48h_pct": 2.0},
            previously_active=True,
            previous_bar_close_time=100,
            previous_sqzmom_color="red",
            now=NOW,
        )
        self.assertTrue(recovered["buy_enabled"])
        self.assertTrue(recovered["recover"])
        self.assertTrue(recovered["recovery_condition"])
        self.assertEqual("roc_sqzmom_combined_recovery", recovered["reason"])
        insufficient_roc = build_technical_buy_gate(
            {**red, "sqzmom": -2, "sqzmom_previous": -4,
             "sqzmom_color": "maroon", "bar_close_time": 201,
             "sqzmom_pct": -2.0, "roc_48h_pct": 0.9},
            previously_active=True,
            previous_bar_close_time=100,
            now=NOW,
        )
        self.assertFalse(insufficient_roc["recover"])
        self.assertFalse(insufficient_roc["buy_enabled"])
        not_adverse = build_technical_buy_gate(
            {**red, "roc_48h_pct": -4.9, "bar_close_time": 300},
            previously_active=False,
            previous_bar_close_time=200,
            previous_sqzmom_color="maroon",
            now=NOW,
        )
        self.assertFalse(not_adverse["risk_off_condition"])
        self.assertFalse(not_adverse["trigger"])

    def test_recommended_thresholds_and_audit_rule_are_exact(self):
        signal = {
            "roc_48h_pct": -5.0,
            "sqzmom_pct": -1.0,
            "sqzmom": -2,
            "sqzmom_previous": -1,
            "sqzmom_color": "red",
            "bar_close_time": 100,
        }
        gate = build_technical_buy_gate(signal, previously_active=False, now=NOW)
        self.assertTrue(gate["trigger"])
        self.assertEqual(-5.0, gate["roc_risk_off_pct"])
        self.assertEqual(-1.0, gate["sqzmom_risk_off_pct"])
        self.assertIn("ROC48 <= -5%", gate["decision_rule"])
        self.assertIn("SQZMOM <= -1%", gate["decision_rule"])
        self.assertIn("ROC48 >= 1%", gate["decision_rule"])
        self.assertIn("SQZMOM >= -3%", gate["decision_rule"])
        self.assertEqual(1.0, gate["roc_recovery_pct"])
        self.assertEqual(-3.0, gate["sqzmom_recovery_pct"])
        self.assertEqual("combined-roc1-sqz3-improving-v1", gate["recovery_rule_version"])
        roc_only = build_technical_buy_gate(
            {**signal, "sqzmom_pct": -0.99, "bar_close_time": 200},
            previously_active=False,
            now=NOW,
        )
        self.assertFalse(roc_only["trigger"])
        self.assertEqual("normal_conditions", roc_only["reason"])

    def test_missing_stale_or_unhealthy_gate_disables_buy(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "technical_buy_gate.json"
            self.assertFalse(load_runtime_technical_gate(path, now=NOW)["buy_enabled"])
            unhealthy = failed_technical_buy_gate("test", now=NOW)
            path.write_text(json.dumps(unhealthy), encoding="utf-8")
            self.assertFalse(load_runtime_technical_gate(path, now=NOW)["buy_enabled"])
            healthy = build_technical_buy_gate({
                "roc_48h_pct": 1, "sqzmom_pct": 1,
                "sqzmom": 2, "sqzmom_previous": 1,
                "sqzmom_color": "lime",
            }, previously_active=False, now=NOW - timedelta(seconds=151))
            path.write_text(json.dumps(healthy), encoding="utf-8")
            loaded = load_runtime_technical_gate(path, now=NOW)
            self.assertFalse(loaded["buy_enabled"])
            self.assertFalse(loaded["runtime_gate_healthy"])


if __name__ == "__main__":
    unittest.main()

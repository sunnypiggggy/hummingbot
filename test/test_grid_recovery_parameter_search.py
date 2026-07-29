import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from search_grid_roc_sqz_recovery_parameters import advance_combined_gate, gate_intervals  # noqa: E402


class GridRecoveryParameterSearchTest(unittest.TestCase):
    def advance(self, active, roc, sqz, current, previous):
        return advance_combined_gate(
            active,
            {
                "roc_48h_pct": roc,
                "sqzmom_pct": sqz,
                "sqzmom": current,
                "sqzmom_previous": previous,
            },
            roc_risk_off_pct=-5,
            sqzmom_risk_off_pct=-1,
            roc_recovery_pct=1,
            sqzmom_recovery_pct=-2,
        )

    def test_risk_off_requires_both_adverse_signals(self):
        self.assertEqual((True, True, False), self.advance(False, -6, -2, -3, -2))
        self.assertEqual((False, False, False), self.advance(False, -4, -2, -3, -2))

    def test_recovery_requires_both_thresholds_and_improving_sqzmom(self):
        self.assertEqual((True, False, False), self.advance(True, 0.5, -1, -1, -2))
        self.assertEqual((True, False, False), self.advance(True, 2, -2.5, -1, -2))
        self.assertEqual((True, False, False), self.advance(True, 2, -1, -3, -2))
        self.assertEqual((False, False, True), self.advance(True, 2, -1, -1, -2))

    def test_recovery_cannot_fire_while_original_risk_off_remains_true(self):
        self.assertEqual((True, False, False), self.advance(True, -6, -2, -1, -2))

    def test_gate_intervals_marks_disabled_and_recovered_ranges(self):
        timeline = {100: True, 400: False, 700: False, 1000: True}
        self.assertEqual(
            [(100, 400, True), (400, 1000, False), (1000, 1300, True)],
            gate_intervals(timeline, 100, 1300),
        )


if __name__ == "__main__":
    unittest.main()

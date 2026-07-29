import json
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from grid_macro_gate import build_grid_macro_gate, load_runtime_macro_gate  # noqa: E402


NOW = datetime(2026, 9, 16, 17, 0, tzinfo=timezone.utc)


def lease(kind: str = "fomc", **overrides) -> dict:
    value = {
        "decision_id": "fomc-2026-09-16-approved",
        "event_id": "fomc-2026-09-16",
        "event_kind": kind,
        "market_impact": "negative",
        "effective_at": (NOW - timedelta(hours=1)).isoformat(),
        "resume_at": (NOW + timedelta(hours=1)).isoformat(),
        "approval": {"status": "approved", "action": "approve"},
    }
    value.update(overrides)
    return value


def state(*leases: dict, reconciled_at: datetime = NOW) -> dict:
    return {
        "schema_version": 3,
        "last_reconcile": reconciled_at.isoformat(),
        "leases": {item["decision_id"]: item for item in leases},
    }


class GridMacroGateTest(unittest.TestCase):
    def test_approved_active_fomc_pauses_both_sides(self):
        gate = build_grid_macro_gate(state(lease()), now=NOW)
        self.assertTrue(gate["source_healthy"])
        self.assertTrue(gate["pause_new_orders"])
        self.assertEqual(
            gate["active_lease_ids"], ["fomc-2026-09-16-approved"]
        )
        self.assertEqual(gate["reason"], "approved_fomc_window_active")

    def test_no_active_fomc_allows_grid(self):
        expired = lease(resume_at=(NOW - timedelta(seconds=1)).isoformat())
        cpi = lease(
            kind="cpi",
            decision_id="cpi-2026-09",
            event_id="cpi-2026-09",
        )
        gate = build_grid_macro_gate(state(expired, cpi), now=NOW)
        self.assertTrue(gate["source_healthy"])
        self.assertFalse(gate["pause_new_orders"])
        self.assertEqual(gate["active_lease_ids"], [])

    def test_shadow_fomc_is_visible_but_cannot_pause_orders(self):
        gate = build_grid_macro_gate(
            state(lease()), now=NOW, execution_enabled=False
        )
        self.assertTrue(gate["source_healthy"])
        self.assertFalse(gate["pause_new_orders"])
        self.assertTrue(gate["shadow_pause_new_orders"])
        self.assertEqual(gate["reason"], "shadow_fomc_window_active")

    def test_revoked_or_unapproved_fomc_is_not_active(self):
        revoked = lease(revoked_at=(NOW - timedelta(seconds=1)).isoformat())
        unapproved = lease(
            decision_id="fomc-unapproved",
            approval={"status": "rejected", "action": "approve"},
        )
        gate = build_grid_macro_gate(state(revoked, unapproved), now=NOW)
        self.assertFalse(gate["pause_new_orders"])

    def test_missing_or_stale_macro_state_fails_closed(self):
        missing = build_grid_macro_gate(None, now=NOW)
        stale = build_grid_macro_gate(
            state(reconciled_at=NOW - timedelta(seconds=151)), now=NOW
        )
        self.assertTrue(missing["pause_new_orders"])
        self.assertFalse(missing["source_healthy"])
        self.assertTrue(stale["pause_new_orders"])
        self.assertIn("stale", stale["reason"])

    def test_runtime_gate_rejects_stale_and_corrupt_files(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "macro_gate.json"
            path.write_text("{broken", encoding="utf-8")
            self.assertTrue(load_runtime_macro_gate(path, now=NOW)["pause_new_orders"])

            gate = build_grid_macro_gate(state(), now=NOW - timedelta(seconds=151))
            path.write_text(json.dumps(gate), encoding="utf-8")
            loaded = load_runtime_macro_gate(path, now=NOW)
            self.assertTrue(loaded["pause_new_orders"])
            self.assertFalse(loaded["runtime_gate_healthy"])

    def test_runtime_gate_accepts_fresh_resume_state(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "macro_gate.json"
            gate = build_grid_macro_gate(state(), now=NOW)
            path.write_text(json.dumps(gate), encoding="utf-8")
            loaded = load_runtime_macro_gate(path, now=NOW)
            self.assertTrue(loaded["runtime_gate_healthy"])
            self.assertFalse(loaded["pause_new_orders"])


if __name__ == "__main__":
    unittest.main()

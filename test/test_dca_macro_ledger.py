import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from macro_control.approval import proposal_sha256
from macro_control.ledger import (
    LedgerValidationError,
    append_record,
    load_records,
    new_record,
    proposal_record,
    replay_records,
    validate_records,
)


UTC = timezone.utc
DECISION_AT = datetime(2026, 8, 10, 11, 0, tzinfo=UTC)
EFFECTIVE_AT = datetime(2026, 8, 10, 12, 0, tzinfo=UTC)
EVENT_AT = datetime(2026, 8, 10, 14, 0, tzinfo=UTC)
RESUME_AT = datetime(2026, 8, 10, 15, 0, tzinfo=UTC)


def proposal(proposal_id: str, impact: str) -> dict:
    return {
        "decision_id": proposal_id,
        "event_id": "cpi-2026-08",
        "event_kind": "cpi",
        "event_at": EVENT_AT.isoformat(),
        "event_source_url": "https://www.bls.gov/cpi/",
        "decision_at": DECISION_AT.isoformat(),
        "market_impact": impact,
        "confidence": 0.9,
        "effective_at": EFFECTIVE_AT.isoformat(),
        "resume_at": RESUME_AT.isoformat(),
        "reason": "Hermes event assessment.",
        "evidence": [
            {
                "source_id": "one",
                "source_url": "https://example.com/one",
                "observed_at": (DECISION_AT - timedelta(minutes=20)).isoformat(),
                "summary": "First source.",
                "kind": "decision",
            },
            {
                "source_id": "two",
                "source_url": "https://example.org/two",
                "observed_at": (DECISION_AT - timedelta(minutes=10)).isoformat(),
                "summary": "Second source.",
                "kind": "decision",
            },
        ],
        "hermes": {
            "agent_id": "hermes-macro",
            "model": "test",
            "prompt_version": "test",
            "run_id": proposal_id,
        },
        "policy_version": "dca-macro-v3",
    }


def approval_record(value: dict, approved_at: datetime, callback: str) -> dict:
    digest = proposal_sha256(value)
    return new_record(
        "approval",
        proposal_id=value["decision_id"],
        proposal_sha256=digest,
        approval={
            "status": "approved",
            "channel": "telegram",
            "action": "approve",
            "telegram_user_id": "owner",
            "telegram_chat_id": "chat",
            "telegram_update_id": f"update-{callback}",
            "telegram_callback_query_id": callback,
            "telegram_message_id": f"message-{callback}",
            "approved_at": approved_at.isoformat(),
            "proposal_sha256": digest,
        },
    )


class LedgerTests(unittest.TestCase):
    def test_append_validate_and_load(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ledger.jsonl"
            value = proposal("negative", "negative")
            append_record(path, proposal_record(value))
            append_record(
                path,
                approval_record(value, EFFECTIVE_AT, "negative-approval"),
            )
            records = load_records(path)
            report = validate_records(records)
            self.assertTrue(report["valid"], report["errors"])
            self.assertEqual(2, report["records"])

    def test_late_approval_is_replayed_from_approval_time_and_marked(self):
        value = proposal("late", "negative")
        approved_at = EFFECTIVE_AT + timedelta(minutes=20)
        result = replay_records(
            [
                proposal_record(value),
                approval_record(value, approved_at, "late-approval"),
            ]
        )
        self.assertEqual(1, result["late_approval_lease_count"])
        self.assertEqual(
            approved_at.isoformat(), result["timeline"][0]["starts_at"]
        )
        self.assertFalse(result["timeline"][0]["macro_buy_enabled"])
        self.assertTrue(result["timeline"][0]["macro_sell_enabled"])

    def test_overlapping_impacts_disable_both_sides(self):
        negative = proposal("negative", "negative")
        positive = proposal("positive", "positive")
        result = replay_records(
            [
                proposal_record(negative),
                approval_record(negative, EFFECTIVE_AT, "negative"),
                proposal_record(positive),
                approval_record(positive, EFFECTIVE_AT, "positive"),
            ]
        )
        self.assertFalse(result["timeline"][0]["macro_buy_enabled"])
        self.assertFalse(result["timeline"][0]["macro_sell_enabled"])

    def test_revocation_truncates_lease(self):
        value = proposal("revoked", "negative")
        digest = proposal_sha256(value)
        revoked_at = EFFECTIVE_AT + timedelta(minutes=30)
        revoke_approval = {
            **approval_record(value, EFFECTIVE_AT, "unused")["approval"],
            "action": "revoke",
            "telegram_callback_query_id": "revoke",
        }
        result = replay_records(
            [
                proposal_record(value),
                approval_record(value, EFFECTIVE_AT, "approved"),
                new_record(
                    "revocation",
                    proposal_id="revoked",
                    proposal_sha256=digest,
                    revoked_at=revoked_at.isoformat(),
                    approval=revoke_approval,
                ),
            ]
        )
        self.assertEqual(
            revoked_at.isoformat(), result["timeline"][-1]["ends_at"]
        )

    def test_corrupt_json_and_hash_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ledger.jsonl"
            path.write_text("{broken\n", encoding="utf-8")
            with self.assertRaises(LedgerValidationError):
                load_records(path)

        value = proposal("bad-hash", "negative")
        record = proposal_record(value)
        record["proposal_sha256"] = "0" * 64
        report = validate_records([record])
        self.assertFalse(report["valid"])
        self.assertIn("proposal hash mismatch", report["errors"][0])


if __name__ == "__main__":
    unittest.main()

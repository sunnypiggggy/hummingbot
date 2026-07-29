import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from macro_control.approval import ApprovalProof, validate_approval_proof
from macro_control.hermes_conversation import (
    create_approval_request,
    resolve_approval_request,
)
from macro_control.ledger import (
    append_record,
    load_records,
    new_record,
    proposal_record,
    validate_records,
)
from test.test_dca_macro_ledger import DECISION_AT, RESUME_AT, proposal


class HermesConversationApprovalTests(unittest.TestCase):
    def test_exact_clarify_choice_creates_gateway_valid_proof(self):
        value = proposal("conversation", "negative")
        request = create_approval_request(
            value,
            interaction_id="interaction-1",
        )
        approval = resolve_approval_request(
            request,
            request["choices"][0],
            approver_user_id="owner",
            chat_id="chat",
            surface="telegram",
            approved_at=DECISION_AT,
        )
        proof = ApprovalProof.from_mapping(approval)
        validate_approval_proof(
            proof,
            expected_proposal_sha256=request["proposal_sha256"],
            expected_user_id="owner",
            expected_chat_id="chat",
            decision_at=DECISION_AT,
            resume_at=RESUME_AT,
            action="approve",
            received_at=datetime(2026, 8, 10, 12, tzinfo=timezone.utc),
        )
        self.assertEqual("approved", approval["status"])
        self.assertEqual("hermes_conversation", approval["channel"])
        self.assertEqual("hermes:interaction-1", proof.replay_key)

    def test_paraphrase_cannot_approve(self):
        request = create_approval_request(
            proposal("no-paraphrase", "positive"),
            interaction_id="interaction-2",
        )
        with self.assertRaisesRegex(ValueError, "exact Hermes approval choice"):
            resolve_approval_request(
                request,
                "yes",
                approver_user_id="owner",
                chat_id="chat",
                surface="telegram",
            )

    def test_rejection_is_audited_but_not_approved(self):
        value = proposal("rejected", "negative")
        request = create_approval_request(
            value,
            interaction_id="interaction-3",
        )
        approval = resolve_approval_request(
            request,
            request["choices"][1],
            approver_user_id="owner",
            chat_id="chat",
            surface="telegram",
            approved_at=DECISION_AT,
        )
        self.assertEqual("rejected", approval["status"])
        with tempfile.TemporaryDirectory() as directory:
            ledger = Path(directory) / "ledger.jsonl"
            append_record(ledger, proposal_record(value))
            append_record(
                ledger,
                new_record(
                    "approval",
                    proposal_id=value["decision_id"],
                    proposal_sha256=request["proposal_sha256"],
                    approval=approval,
                ),
            )
            report = validate_records(load_records(ledger))
            self.assertTrue(report["valid"], report["errors"])

    def test_response_hash_and_interaction_replay_are_rejected(self):
        value = proposal("tampered", "negative")
        request = create_approval_request(
            value,
            interaction_id="same-interaction",
        )
        approval = resolve_approval_request(
            request,
            request["choices"][0],
            approver_user_id="owner",
            chat_id="chat",
            surface="telegram",
            approved_at=DECISION_AT,
        )
        tampered = {**approval, "hermes_response_sha256": "0" * 64}
        proof = ApprovalProof.from_mapping(tampered)
        with self.assertRaisesRegex(ValueError, "response hash"):
            validate_approval_proof(
                proof,
                expected_proposal_sha256=request["proposal_sha256"],
                expected_user_id="owner",
                expected_chat_id="chat",
                decision_at=DECISION_AT,
                resume_at=RESUME_AT,
                action="approve",
            )

        records = [
            proposal_record(value),
            new_record(
                "approval",
                proposal_id=value["decision_id"],
                proposal_sha256=request["proposal_sha256"],
                approval=approval,
            ),
            new_record(
                "approval",
                proposal_id=value["decision_id"],
                proposal_sha256=request["proposal_sha256"],
                approval=approval,
            ),
        ]
        report = validate_records(records)
        self.assertFalse(report["valid"])
        self.assertTrue(
            any("interaction is duplicated" in error for error in report["errors"])
        )


if __name__ == "__main__":
    unittest.main()

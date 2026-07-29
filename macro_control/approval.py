from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Mapping


PROPOSAL_FIELDS = (
    "decision_id",
    "event_id",
    "event_kind",
    "event_at",
    "event_source_url",
    "decision_at",
    "market_impact",
    "confidence",
    "effective_at",
    "resume_at",
    "reason",
    "evidence",
    "hermes",
    "policy_version",
)


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def proposal_payload(value: Mapping[str, Any]) -> dict:
    return {field: value[field] for field in PROPOSAL_FIELDS}


def proposal_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json(proposal_payload(value))).hexdigest()


def hermes_conversation_choice(
    action: str,
    interaction_id: str,
    proposal_hash: str,
    *,
    approved: bool,
) -> str:
    if action not in {"approve", "revoke"}:
        raise ValueError("action must be approve or revoke")
    verb = "批准执行" if action == "approve" else "批准撤销"
    if not approved:
        verb = "拒绝执行" if action == "approve" else "拒绝撤销"
    return f"{verb} {interaction_id} {proposal_hash[:16]}"


@dataclass(frozen=True)
class ApprovalProof:
    status: str
    channel: str
    action: str
    approved_at: datetime
    proposal_sha256: str
    telegram_user_id: str = ""
    telegram_chat_id: str = ""
    telegram_update_id: str = ""
    telegram_callback_query_id: str = ""
    telegram_message_id: str = ""
    hermes_interaction_id: str = ""
    hermes_surface: str = ""
    hermes_approver_user_id: str = ""
    hermes_chat_id: str = ""
    hermes_response_sha256: str = ""

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ApprovalProof":
        try:
            approved_at = datetime.fromisoformat(str(value["approved_at"]))
            if approved_at.tzinfo is None:
                raise ValueError("approved_at must include a timezone")
            return cls(
                status=str(value["status"]).lower(),
                channel=str(value["channel"]).lower(),
                action=str(value.get("action", "approve")).lower(),
                approved_at=approved_at.astimezone(timezone.utc),
                proposal_sha256=str(value["proposal_sha256"]).lower(),
                telegram_user_id=str(value.get("telegram_user_id", "")),
                telegram_chat_id=str(value.get("telegram_chat_id", "")),
                telegram_update_id=str(value.get("telegram_update_id", "")),
                telegram_callback_query_id=str(
                    value.get("telegram_callback_query_id", "")
                ),
                telegram_message_id=str(value.get("telegram_message_id", "")),
                hermes_interaction_id=str(
                    value.get("hermes_interaction_id", "")
                ),
                hermes_surface=str(value.get("hermes_surface", "")).lower(),
                hermes_approver_user_id=str(
                    value.get("hermes_approver_user_id", "")
                ),
                hermes_chat_id=str(value.get("hermes_chat_id", "")),
                hermes_response_sha256=str(
                    value.get("hermes_response_sha256", "")
                ).lower(),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid approval proof: {exc}") from exc

    def to_dict(self) -> dict:
        result = {
            "status": self.status,
            "channel": self.channel,
            "action": self.action,
            "approved_at": self.approved_at.isoformat(),
            "proposal_sha256": self.proposal_sha256,
        }
        if self.channel == "telegram":
            result.update(
                {
                    "telegram_user_id": self.telegram_user_id,
                    "telegram_chat_id": self.telegram_chat_id,
                    "telegram_update_id": self.telegram_update_id,
                    "telegram_callback_query_id": (
                        self.telegram_callback_query_id
                    ),
                    "telegram_message_id": self.telegram_message_id,
                }
            )
        elif self.channel == "hermes_conversation":
            result.update(
                {
                    "hermes_interaction_id": self.hermes_interaction_id,
                    "hermes_surface": self.hermes_surface,
                    "hermes_approver_user_id": self.hermes_approver_user_id,
                    "hermes_chat_id": self.hermes_chat_id,
                    "hermes_response_sha256": self.hermes_response_sha256,
                }
            )
        return result

    @property
    def replay_key(self) -> str:
        if self.channel == "telegram":
            return f"telegram:{self.telegram_callback_query_id}"
        if self.channel == "hermes_conversation":
            return f"hermes:{self.hermes_interaction_id}"
        return ""


def validate_approval_proof(
    approval: ApprovalProof,
    *,
    expected_proposal_sha256: str,
    expected_user_id: str,
    expected_chat_id: str,
    decision_at: datetime,
    resume_at: datetime,
    action: str,
    received_at: datetime | None = None,
) -> None:
    if approval.status != "approved":
        raise ValueError("decision was not approved")
    if approval.channel not in {"telegram", "hermes_conversation"}:
        raise ValueError("unsupported approval channel")
    if approval.action != action:
        raise ValueError(f"approval action must be {action}")
    if not expected_user_id or not expected_chat_id:
        raise ValueError("approval owner is not configured")
    if approval.proposal_sha256 != expected_proposal_sha256:
        raise ValueError("approval proposal hash does not match")
    decision_at = decision_at.astimezone(timezone.utc)
    resume_at = resume_at.astimezone(timezone.utc)
    if approval.approved_at < decision_at:
        raise ValueError("approval predates the Hermes decision")
    if approval.approved_at > resume_at:
        raise ValueError("approval is outside the event window")
    if received_at is not None and approval.approved_at > (
        received_at.astimezone(timezone.utc) + timedelta(seconds=60)
    ):
        raise ValueError("approval timestamp is in the future")
    if approval.channel == "telegram":
        if approval.telegram_user_id != str(expected_user_id):
            raise ValueError("Telegram approver user ID does not match")
        if approval.telegram_chat_id != str(expected_chat_id):
            raise ValueError("Telegram approver chat ID does not match")
        if (
            not approval.telegram_update_id
            or not approval.telegram_callback_query_id
        ):
            raise ValueError("Telegram callback identifiers are required")
        return

    if approval.hermes_approver_user_id != str(expected_user_id):
        raise ValueError("Hermes conversation approver user ID does not match")
    if approval.hermes_chat_id != str(expected_chat_id):
        raise ValueError("Hermes conversation chat ID does not match")
    if approval.hermes_surface not in {"telegram", "cli", "dashboard"}:
        raise ValueError("Hermes conversation surface is invalid")
    if not approval.hermes_interaction_id:
        raise ValueError("Hermes conversation interaction ID is required")
    expected_choice = hermes_conversation_choice(
        action,
        approval.hermes_interaction_id,
        expected_proposal_sha256,
        approved=True,
    )
    expected_response_hash = hashlib.sha256(
        expected_choice.encode("utf-8")
    ).hexdigest()
    if approval.hermes_response_sha256 != expected_response_hash:
        raise ValueError("Hermes conversation response hash does not match")


# Backwards-compatible names for callers that still use the direct Bot adapter.
TelegramApproval = ApprovalProof
validate_telegram_approval = validate_approval_proof

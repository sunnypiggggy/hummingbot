from __future__ import annotations

import json
import os
import uuid
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from .approval import hermes_conversation_choice, proposal_sha256
from .policy import POLICY_VERSION, RiskWindowDecision, RiskWindowPolicy


LEDGER_SCHEMA_VERSION = "hermes-dca-event-ledger-v3"
RECORD_TYPES = {"event", "proposal", "approval", "execution", "revocation"}


class LedgerValidationError(ValueError):
    pass


def _approval_replay_key(approval: dict, proposal_hash: str) -> str:
    channel = str(approval.get("channel", ""))
    if channel == "telegram":
        required = (
            "telegram_user_id",
            "telegram_chat_id",
            "telegram_update_id",
            "telegram_callback_query_id",
            "telegram_message_id",
        )
        if any(not str(approval.get(field, "")) for field in required):
            raise LedgerValidationError(
                "Telegram approval identifiers are incomplete"
            )
        return f"telegram:{approval['telegram_callback_query_id']}"
    if channel == "hermes_conversation":
        required = (
            "hermes_interaction_id",
            "hermes_surface",
            "hermes_approver_user_id",
            "hermes_chat_id",
            "hermes_response_sha256",
        )
        if any(not str(approval.get(field, "")) for field in required):
            raise LedgerValidationError(
                "Hermes conversation approval identifiers are incomplete"
            )
        action = str(approval.get("action", "approve"))
        choice = hermes_conversation_choice(
            action,
            str(approval["hermes_interaction_id"]),
            proposal_hash,
            approved=approval.get("status") == "approved",
        )
        expected = hashlib.sha256(choice.encode("utf-8")).hexdigest()
        if approval.get("hermes_response_sha256") != expected:
            raise LedgerValidationError(
                "Hermes conversation response hash mismatch"
            )
        return f"hermes:{approval['hermes_interaction_id']}"
    raise LedgerValidationError("unsupported approval channel")


def _time(value: object, field: str) -> datetime:
    try:
        result = datetime.fromisoformat(str(value))
    except ValueError as exc:
        raise LedgerValidationError(f"{field} is not ISO-8601") from exc
    if result.tzinfo is None:
        raise LedgerValidationError(f"{field} must include a timezone")
    return result.astimezone(timezone.utc)


def new_record(record_type: str, **values) -> dict:
    if record_type not in RECORD_TYPES:
        raise ValueError(f"unsupported record_type: {record_type}")
    return {
        "schema_version": LEDGER_SCHEMA_VERSION,
        "record_type": record_type,
        "record_id": str(values.pop("record_id", uuid.uuid4().hex)),
        "recorded_at": str(
            values.pop("recorded_at", datetime.now(timezone.utc).isoformat())
        ),
        **values,
    }


def proposal_record(proposal: dict, **metadata) -> dict:
    proposal = dict(proposal)
    proposal.pop("snapshot_id", None)
    proposal.pop("approval", None)
    return new_record(
        "proposal",
        record_id=metadata.pop("record_id", proposal["decision_id"]),
        proposal=proposal,
        proposal_sha256=proposal_sha256(proposal),
        **metadata,
    )


def append_record(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(
            record,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")
    descriptor = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
    try:
        os.write(descriptor, encoded)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def load_records(path: Path) -> list[dict]:
    records: list[dict] = []
    if not path.exists():
        return records
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise LedgerValidationError(
                f"line {line_number} is invalid JSON: {exc.msg}"
            ) from exc
        if not isinstance(value, dict):
            raise LedgerValidationError(
                f"line {line_number} must contain one JSON object"
            )
        records.append(value)
    return records


def validate_records(records: Iterable[dict]) -> dict:
    records = list(records)
    errors: list[str] = []
    record_ids: set[str] = set()
    approval_ids: set[str] = set()
    proposals: dict[str, dict] = {}
    approvals: dict[str, dict] = {}
    executions = 0
    revocations = 0

    for index, record in enumerate(records, start=1):
        prefix = f"record {index}"
        try:
            if record.get("schema_version") != LEDGER_SCHEMA_VERSION:
                raise LedgerValidationError("schema_version mismatch")
            record_type = str(record.get("record_type", ""))
            if record_type not in RECORD_TYPES:
                raise LedgerValidationError("unsupported record_type")
            record_id = str(record.get("record_id", ""))
            if not record_id or record_id in record_ids:
                raise LedgerValidationError("record_id is empty or duplicated")
            record_ids.add(record_id)
            _time(record.get("recorded_at"), "recorded_at")

            if record_type == "event":
                event = record.get("event")
                if not isinstance(event, dict):
                    raise LedgerValidationError("event payload is required")
                if event.get("kind") not in {"fomc", "cpi", "nfp"}:
                    raise LedgerValidationError("unsupported event kind")
                _time(event.get("starts_at"), "event.starts_at")
                if not str(event.get("source_url", "")).startswith("https://"):
                    raise LedgerValidationError("official HTTPS source is required")
                source_hash = str(record.get("source_sha256", ""))
                if len(source_hash) != 64 or any(
                    character not in "0123456789abcdef"
                    for character in source_hash
                ):
                    raise LedgerValidationError(
                        "event source_sha256 must be a lowercase SHA-256"
                    )

            elif record_type == "proposal":
                proposal = record.get("proposal")
                if not isinstance(proposal, dict):
                    raise LedgerValidationError("proposal payload is required")
                if "snapshot_id" in proposal or "approval" in proposal:
                    raise LedgerValidationError(
                        "proposal must not contain snapshot_id or approval"
                    )
                expected_hash = proposal_sha256(proposal)
                if record.get("proposal_sha256") != expected_hash:
                    raise LedgerValidationError("proposal hash mismatch")
                decision = RiskWindowDecision.from_mapping(
                    {**proposal, "snapshot_id": "ledger-validation"}
                )
                assessment = RiskWindowPolicy().assess(decision)
                if not assessment.accepted:
                    raise LedgerValidationError(
                        "proposal policy rejected: " + ",".join(assessment.reasons)
                    )
                if decision.policy_version != POLICY_VERSION:
                    raise LedgerValidationError("proposal policy_version mismatch")
                proposals[decision.decision_id] = record

            elif record_type == "approval":
                proposal_id = str(record.get("proposal_id", ""))
                proposal = proposals.get(proposal_id)
                if proposal is None:
                    raise LedgerValidationError(
                        "approval must follow its proposal"
                    )
                approval = record.get("approval")
                if not isinstance(approval, dict):
                    raise LedgerValidationError("approval payload is required")
                if approval.get("status") not in {"approved", "rejected"}:
                    raise LedgerValidationError("approval status is invalid")
                if approval.get("action", "approve") != "approve":
                    raise LedgerValidationError("approval action must be approve")
                if approval.get("proposal_sha256") != proposal["proposal_sha256"]:
                    raise LedgerValidationError("approval hash mismatch")
                approval_id = _approval_replay_key(
                    approval, proposal["proposal_sha256"]
                )
                if approval_id in approval_ids:
                    raise LedgerValidationError(
                        "approval interaction is duplicated"
                    )
                approval_ids.add(approval_id)
                approved_at = _time(approval.get("approved_at"), "approved_at")
                decision_at = _time(
                    proposal["proposal"]["decision_at"], "decision_at"
                )
                resume_at = _time(
                    proposal["proposal"]["resume_at"], "resume_at"
                )
                if approved_at < decision_at or approved_at > resume_at:
                    raise LedgerValidationError(
                        "approval is outside the decision window"
                    )
                approvals[proposal_id] = record

            elif record_type == "execution":
                proposal_id = str(record.get("proposal_id", ""))
                proposal = proposals.get(proposal_id)
                if proposal is None:
                    raise LedgerValidationError("execution proposal is unknown")
                if proposal["proposal"]["market_impact"] != "neutral":
                    approval = approvals.get(proposal_id)
                    if (
                        approval is None
                        or approval["approval"].get("status") != "approved"
                    ):
                        raise LedgerValidationError(
                            "execution requires an approved proposal"
                        )
                if record.get("proposal_sha256") != proposal["proposal_sha256"]:
                    raise LedgerValidationError("execution hash mismatch")
                _time(record.get("submitted_at"), "submitted_at")
                if not record.get("snapshot_id"):
                    raise LedgerValidationError("execution snapshot_id is required")
                executions += 1

            elif record_type == "revocation":
                proposal_id = str(record.get("proposal_id", ""))
                proposal = proposals.get(proposal_id)
                if proposal is None:
                    raise LedgerValidationError("revocation proposal is unknown")
                approval = record.get("approval")
                if not isinstance(approval, dict):
                    raise LedgerValidationError(
                        "revocation approval payload is required"
                    )
                if approval.get("status") != "approved":
                    raise LedgerValidationError("revocation must be approved")
                if approval.get("action") != "revoke":
                    raise LedgerValidationError(
                        "revocation approval action must be revoke"
                    )
                if approval.get("proposal_sha256") != proposal["proposal_sha256"]:
                    raise LedgerValidationError("revocation hash mismatch")
                approval_id = _approval_replay_key(
                    approval, proposal["proposal_sha256"]
                )
                if approval_id in approval_ids:
                    raise LedgerValidationError(
                        "approval interaction is duplicated"
                    )
                approval_ids.add(approval_id)
                _time(record.get("revoked_at"), "revoked_at")
                revocations += 1
        except (KeyError, TypeError, ValueError, LedgerValidationError) as exc:
            errors.append(f"{prefix}: {exc}")

    return {
        "valid": not errors,
        "records": len(records),
        "proposals": len(proposals),
        "approvals": len(approvals),
        "executions": executions,
        "revocations": revocations,
        "errors": errors,
    }


def replay_records(records: Iterable[dict]) -> dict:
    records = list(records)
    report = validate_records(records)
    if not report["valid"]:
        raise LedgerValidationError("; ".join(report["errors"]))

    proposals = {
        record["proposal"]["decision_id"]: record
        for record in records
        if record["record_type"] == "proposal"
    }
    approvals = {
        record["proposal_id"]: record
        for record in records
        if record["record_type"] == "approval"
        and record["approval"]["status"] == "approved"
    }
    revocations = {
        record["proposal_id"]: record
        for record in records
        if record["record_type"] == "revocation"
    }
    leases: list[dict] = []
    for proposal_id, record in proposals.items():
        proposal = record["proposal"]
        impact = proposal["market_impact"]
        if impact == "neutral":
            continue
        approval_record = approvals.get(proposal_id)
        if approval_record is None:
            continue
        approved_at = _time(
            approval_record["approval"]["approved_at"], "approved_at"
        )
        effective_at = _time(proposal["effective_at"], "effective_at")
        resume_at = _time(proposal["resume_at"], "resume_at")
        starts_at = max(effective_at, approved_at)
        revocation = revocations.get(proposal_id)
        ends_at = (
            min(resume_at, _time(revocation["revoked_at"], "revoked_at"))
            if revocation
            else resume_at
        )
        if starts_at >= ends_at:
            continue
        leases.append(
            {
                "proposal_id": proposal_id,
                "market_impact": impact,
                "starts_at": starts_at,
                "ends_at": ends_at,
                "approval_timing": (
                    "late_approval"
                    if approved_at > effective_at
                    else "on_time"
                ),
            }
        )

    boundaries = sorted(
        {value for lease in leases for value in (lease["starts_at"], lease["ends_at"])}
    )
    timeline = []
    for start, end in zip(boundaries, boundaries[1:]):
        active = [
            lease
            for lease in leases
            if lease["starts_at"] <= start < lease["ends_at"]
        ]
        if not active:
            continue
        timeline.append(
            {
                "starts_at": start.isoformat(),
                "ends_at": end.isoformat(),
                "macro_buy_enabled": not any(
                    lease["market_impact"] == "negative" for lease in active
                ),
                "macro_sell_enabled": not any(
                    lease["market_impact"] == "positive" for lease in active
                ),
                "active_proposal_ids": sorted(
                    lease["proposal_id"] for lease in active
                ),
                "approval_timing": sorted(
                    {lease["approval_timing"] for lease in active}
                ),
            }
        )
    return {
        "schema_version": LEDGER_SCHEMA_VERSION,
        "lease_count": len(leases),
        "timeline": timeline,
        "on_time_lease_count": sum(
            lease["approval_timing"] == "on_time" for lease in leases
        ),
        "late_approval_lease_count": sum(
            lease["approval_timing"] == "late_approval" for lease in leases
        ),
    }

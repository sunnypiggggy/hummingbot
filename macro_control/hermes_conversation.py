from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timezone
from typing import Mapping

from .approval import (
    hermes_conversation_choice,
    proposal_payload,
    proposal_sha256,
)


def create_approval_request(
    proposal: Mapping,
    *,
    action: str = "approve",
    interaction_id: str | None = None,
) -> dict:
    if action not in {"approve", "revoke"}:
        raise ValueError("action must be approve or revoke")
    payload = proposal_payload(proposal)
    digest = proposal_sha256(payload)
    interaction_id = interaction_id or uuid.uuid4().hex
    approve_choice = hermes_conversation_choice(
        action, interaction_id, digest, approved=True
    )
    reject_choice = hermes_conversation_choice(
        action, interaction_id, digest, approved=False
    )
    direction = {
        "negative": "关闭 BUY",
        "positive": "关闭 SELL",
        "neutral": "仅审计",
    }[str(payload["market_impact"])]
    prompt = (
        f"DCA 宏观控制审批\n"
        f"事件：{payload['event_kind']} / {payload['event_id']}\n"
        f"影响：{payload['market_impact']}（{direction}）\n"
        f"置信度：{float(payload['confidence']):.2f}\n"
        f"窗口：{payload['effective_at']} → {payload['resume_at']}\n"
        f"理由：{payload['reason']}\n"
        f"提案哈希：{digest}\n"
        "请选择；未响应或选择拒绝不会改变交易。"
    )
    return {
        "schema_version": "hermes-conversation-approval-v1",
        "interaction_id": interaction_id,
        "action": action,
        "proposal_sha256": digest,
        "proposal": payload,
        "prompt": prompt,
        "choices": [approve_choice, reject_choice],
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def resolve_approval_request(
    request: Mapping,
    response: str,
    *,
    approver_user_id: str,
    chat_id: str,
    surface: str,
    approved_at: datetime | None = None,
) -> dict:
    if request.get("schema_version") != "hermes-conversation-approval-v1":
        raise ValueError("unsupported Hermes conversation approval request")
    payload = proposal_payload(request["proposal"])
    digest = proposal_sha256(payload)
    if digest != request.get("proposal_sha256"):
        raise ValueError("Hermes conversation proposal hash changed")
    action = str(request["action"])
    interaction_id = str(request["interaction_id"])
    approved_choice = hermes_conversation_choice(
        action, interaction_id, digest, approved=True
    )
    rejected_choice = hermes_conversation_choice(
        action, interaction_id, digest, approved=False
    )
    response = str(response).strip()
    if response not in {approved_choice, rejected_choice}:
        raise ValueError("response is not an exact Hermes approval choice")
    if not approver_user_id or not chat_id:
        raise ValueError("Hermes conversation approver is not configured")
    surface = str(surface).lower()
    if surface not in {"telegram", "cli", "dashboard"}:
        raise ValueError("surface must be telegram, cli, or dashboard")
    return {
        "status": "approved" if response == approved_choice else "rejected",
        "channel": "hermes_conversation",
        "action": action,
        "hermes_interaction_id": interaction_id,
        "hermes_surface": surface,
        "hermes_approver_user_id": str(approver_user_id),
        "hermes_chat_id": str(chat_id),
        "hermes_response_sha256": hashlib.sha256(
            response.encode("utf-8")
        ).hexdigest(),
        "approved_at": (
            approved_at or datetime.now(timezone.utc)
        ).astimezone(timezone.utc).isoformat(),
        "proposal_sha256": digest,
    }


def dumps_request(request: Mapping) -> str:
    return json.dumps(request, ensure_ascii=False, indent=2)

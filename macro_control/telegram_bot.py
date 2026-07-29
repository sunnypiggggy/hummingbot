from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from typing import Callable

from .approval import proposal_sha256


DEFAULT_APPROVAL_TIMEOUT_SECONDS = 12 * 60 * 60


class TelegramApprovalTimeout(TimeoutError):
    pass


class TelegramApprovalBot:
    """Legacy direct-Bot adapter.

    Hermes Agent integrations should use its native ``clarify`` conversation
    flow instead. Running this adapter next to Hermes with the same token would
    create two competing ``getUpdates`` consumers.
    """

    def __init__(
        self,
        token: str,
        approver_user_id: str,
        approver_chat_id: str,
        *,
        api_base: str = "https://api.telegram.org",
        opener: Callable = urllib.request.urlopen,
    ) -> None:
        if not token or not approver_user_id or not approver_chat_id:
            raise ValueError(
                "Telegram token, approver user ID and chat ID are required"
            )
        self.base_url = f"{api_base.rstrip('/')}/bot{token}"
        self.approver_user_id = str(approver_user_id)
        self.approver_chat_id = str(approver_chat_id)
        self.opener = opener

    def _request(self, method: str, payload: dict) -> dict:
        request = urllib.request.Request(
            f"{self.base_url}/{method}",
            data=urllib.parse.urlencode(payload).encode(),
            method="POST",
        )
        with self.opener(request, timeout=35) as response:
            result = json.loads(response.read().decode())
        if not result.get("ok"):
            raise RuntimeError(f"Telegram {method} failed")
        return result["result"]

    @staticmethod
    def _callback_data(action: str, proposal_hash: str) -> str:
        return f"dca:{action}:{proposal_hash[:24]}"

    @staticmethod
    def _proposal_text(proposal: dict, proposal_hash: str, action: str) -> str:
        verb = "提前撤销" if action == "revoke" else "执行"
        return (
            f"DCA 宏观控制请求：{verb}\n"
            f"事件：{proposal['event_kind']} / {proposal['event_id']}\n"
            f"影响：{proposal['market_impact']}\n"
            f"置信度：{float(proposal['confidence']):.2f}\n"
            f"生效：{proposal['effective_at']}\n"
            f"恢复：{proposal['resume_at']}\n"
            f"理由：{proposal['reason']}\n"
            f"提案哈希：{proposal_hash[:16]}…"
        )

    def request_approval(self, proposal: dict, *, action: str = "approve") -> dict:
        if action not in {"approve", "revoke"}:
            raise ValueError("action must be approve or revoke")
        proposal_hash = proposal_sha256(proposal)
        approve_action = "revoke" if action == "revoke" else "approve"
        keyboard = {
            "inline_keyboard": [
                [
                    {
                        "text": "批准",
                        "callback_data": self._callback_data(
                            approve_action, proposal_hash
                        ),
                    },
                    {
                        "text": "拒绝",
                        "callback_data": self._callback_data(
                            "reject", proposal_hash
                        ),
                    },
                ]
            ]
        }
        message = self._request(
            "sendMessage",
            {
                "chat_id": self.approver_chat_id,
                "text": self._proposal_text(proposal, proposal_hash, action),
                "reply_markup": json.dumps(keyboard, separators=(",", ":")),
            },
        )
        return {
            "proposal_sha256": proposal_hash,
            "telegram_message_id": str(message["message_id"]),
            "requested_at": datetime.now(timezone.utc).isoformat(),
            "action": action,
        }

    def wait_for_approval(
        self,
        request_record: dict,
        *,
        timeout_seconds: float = DEFAULT_APPROVAL_TIMEOUT_SECONDS,
        offset: int | None = None,
    ) -> dict:
        proposal_hash = str(request_record["proposal_sha256"])
        message_id = str(request_record["telegram_message_id"])
        expected_action = (
            "revoke" if request_record.get("action") == "revoke" else "approve"
        )
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            poll_seconds = max(1, min(30, int(deadline - time.monotonic())))
            payload = {
                "timeout": poll_seconds,
                "allowed_updates": json.dumps(["callback_query"]),
            }
            if offset is not None:
                payload["offset"] = offset
            updates = self._request("getUpdates", payload)
            for update in updates:
                offset = int(update["update_id"]) + 1
                callback = update.get("callback_query", {})
                message = callback.get("message", {})
                sender = callback.get("from", {})
                chat = message.get("chat", {})
                if (
                    str(sender.get("id")) != self.approver_user_id
                    or str(chat.get("id")) != self.approver_chat_id
                    or str(message.get("message_id")) != message_id
                ):
                    continue
                data = str(callback.get("data", ""))
                approved_value = self._callback_data(
                    expected_action, proposal_hash
                )
                rejected_value = self._callback_data("reject", proposal_hash)
                if data not in {approved_value, rejected_value}:
                    continue
                status = "approved" if data == approved_value else "rejected"
                self._request(
                    "answerCallbackQuery",
                    {
                        "callback_query_id": callback["id"],
                        "text": "已批准" if status == "approved" else "已拒绝",
                    },
                )
                return {
                    "status": status,
                    "channel": "telegram",
                    "action": expected_action,
                    "telegram_user_id": self.approver_user_id,
                    "telegram_chat_id": self.approver_chat_id,
                    "telegram_update_id": str(update["update_id"]),
                    "telegram_callback_query_id": str(callback["id"]),
                    "telegram_message_id": message_id,
                    "approved_at": datetime.now(timezone.utc).isoformat(),
                    "proposal_sha256": proposal_hash,
                }
        raise TelegramApprovalTimeout("Telegram approval timed out")

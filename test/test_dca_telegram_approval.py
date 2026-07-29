import json
import unittest

from macro_control.approval import proposal_sha256
from macro_control.hermes_cli import build_parser
from macro_control.telegram_bot import (
    DEFAULT_APPROVAL_TIMEOUT_SECONDS,
    TelegramApprovalBot,
)
from test.test_dca_macro_ledger import proposal


class FakeResponse:
    def __init__(self, value):
        self.value = value

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def read(self):
        return json.dumps(self.value).encode()


class FakeTelegram:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    def __call__(self, request, timeout=0):
        self.requests.append((request, timeout))
        return FakeResponse(self.responses.pop(0))


class TelegramApprovalTests(unittest.TestCase):
    def test_approval_and_revocation_default_to_twelve_hours(self):
        self.assertEqual(43_200, DEFAULT_APPROVAL_TIMEOUT_SECONDS)
        parser = build_parser()
        approve = parser.parse_args(
            ["approve", "--dossier", "proposal.json", "--output", "approved.json"]
        )
        revoke = parser.parse_args(
            ["revoke", "decision-id", "--dossier", "proposal.json"]
        )
        self.assertEqual(43_200, approve.timeout_seconds)
        self.assertEqual(43_200, revoke.timeout_seconds)

    def test_only_configured_user_and_message_can_approve(self):
        value = proposal("telegram", "negative")
        digest = proposal_sha256(value)
        callback_data = f"dca:approve:{digest[:24]}"
        opener = FakeTelegram(
            [
                {"ok": True, "result": {"message_id": 42}},
                {
                    "ok": True,
                    "result": [
                        {
                            "update_id": 1,
                            "callback_query": {
                                "id": "wrong",
                                "from": {"id": 999},
                                "message": {
                                    "message_id": 42,
                                    "chat": {"id": 200},
                                },
                                "data": callback_data,
                            },
                        },
                        {
                            "update_id": 2,
                            "callback_query": {
                                "id": "approved",
                                "from": {"id": 100},
                                "message": {
                                    "message_id": 42,
                                    "chat": {"id": 200},
                                },
                                "data": callback_data,
                            },
                        },
                    ],
                },
                {"ok": True, "result": True},
            ]
        )
        bot = TelegramApprovalBot(
            "token",
            "100",
            "200",
            api_base="https://telegram.invalid",
            opener=opener,
        )
        request = bot.request_approval(value)
        approval = bot.wait_for_approval(request, timeout_seconds=1)
        self.assertEqual("approved", approval["status"])
        self.assertEqual("approved", approval["telegram_callback_query_id"])
        self.assertEqual(digest, approval["proposal_sha256"])

    def test_reject_button_records_rejected_status(self):
        value = proposal("telegram-reject", "positive")
        digest = proposal_sha256(value)
        opener = FakeTelegram(
            [
                {"ok": True, "result": {"message_id": 43}},
                {
                    "ok": True,
                    "result": [
                        {
                            "update_id": 3,
                            "callback_query": {
                                "id": "rejected",
                                "from": {"id": 100},
                                "message": {
                                    "message_id": 43,
                                    "chat": {"id": 200},
                                },
                                "data": f"dca:reject:{digest[:24]}",
                            },
                        }
                    ],
                },
                {"ok": True, "result": True},
            ]
        )
        bot = TelegramApprovalBot("token", "100", "200", opener=opener)
        approval = bot.wait_for_approval(
            bot.request_approval(value), timeout_seconds=1
        )
        self.assertEqual("rejected", approval["status"])


if __name__ == "__main__":
    unittest.main()

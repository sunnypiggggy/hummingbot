from types import SimpleNamespace
from unittest import IsolatedAsyncioTestCase

from fastapi import HTTPException

from stocks_runtime.policy import PolicyViolation
from stocks_runtime.executor_config import build_order_executor_config, normalize_executor_config
from stocks_runtime.router import LimitsUpdate, _create_managed_executor, _preview_managed_executor


class _Policy:
    def __init__(self, error=None):
        self.error = error

    async def preview(self, *_args):
        if self.error:
            raise self.error
        return {"allowed": True}


def _request(policy=None, ledger=None):
    state = SimpleNamespace(
        stocks_policy=policy or _Policy(),
        stocks_settings=SimpleNamespace(mode="PAPER"),
        stocks_ledger=ledger,
        executor_service=SimpleNamespace(),
    )
    return SimpleNamespace(app=SimpleNamespace(state=state))


class RouterSemanticsTests(IsolatedAsyncioTestCase):
    async def test_business_preflight_failure_is_http_200_payload(self):
        result = await _preview_managed_executor(
            {}, "telegram-management-bot",
            _request(_Policy(PolicyViolation(
                "超过单笔本金限额", "请求本金超过当前单笔运行限额",
                requested="600", current="0", available="500", limit="500",
            ))),
        )
        self.assertFalse(result["allowed"])
        self.assertEqual("超过单笔本金限额", result["violation"]["code"])

    async def test_malformed_request_maps_to_422_and_transient_to_503(self):
        with self.assertRaises(HTTPException) as malformed:
            await _preview_managed_executor({}, "telegram-management-bot", _request(_Policy(ValueError("bad"))))
        self.assertEqual(422, malformed.exception.status_code)
        with self.assertRaises(HTTPException) as transient:
            await _preview_managed_executor({}, "telegram-management-bot", _request(_Policy(RuntimeError("db down"))))
        self.assertEqual(503, transient.exception.status_code)

    async def test_same_id_different_config_is_the_remaining_409(self):
        original = build_order_executor_config(
            executor_id="same-id-0001", symbol="AAPL", side="BUY", amount="1",
            order_type="LIMIT", price="100",
        )
        changed = build_order_executor_config(
            executor_id="same-id-0001", symbol="AAPL", side="BUY", amount="2",
            order_type="LIMIT", price="100",
        )

        class Ledger:
            async def executor_record(self, _executor_id):
                return {"config": normalize_executor_config(original)}

        with self.assertRaises(HTTPException) as conflict:
            await _create_managed_executor(
                changed,
                "telegram-management-bot", _request(ledger=Ledger()),
            )
        self.assertEqual(409, conflict.exception.status_code)

    def test_legacy_three_field_limit_update_keeps_daily_loss_optional(self):
        value = LimitsUpdate(
            max_order_notional="500", max_symbol_exposure="1000", max_managed_exposure="2000"
        )
        self.assertIsNone(value.daily_loss_limit)

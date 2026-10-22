from decimal import InvalidOperation
from types import SimpleNamespace
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, patch

from fastapi import HTTPException

from stocks_runtime.policy import PolicyViolation
from stocks_runtime.executor_config import build_order_executor_config, normalize_executor_config
from stocks_runtime.ledger import LedgerConflict, LedgerLimitExceeded
from stocks_runtime.router import (
    LimitsUpdate, OrderExecutorRequest, PositionExecutorRequest,
    _create_managed_executor, _preview_managed_executor,
    create_order_executor, create_position_executor, preview_order_executor, preview_position_executor,
)


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


def _typed_requests():
    yield (
        OrderExecutorRequest(
            id="order-aapl-0001", symbol="aapl", side="SELL", amount="1.25", order_type="LIMIT",
            price="101.50", source_owner="owner-aapl-0001", controller_id="custom-controller",
        ),
        create_order_executor, preview_order_executor,
        {
            "id": "order-aapl-0001", "type": "order_executor", "connector_name": "binance_stocks",
            "trading_pair": "AAPL-USDC", "side": "SELL", "amount": "1.25", "execution_strategy": "LIMIT",
            "position_action": "CLOSE", "price": "101.50", "managed_source_owner": "owner-aapl-0001",
        },
    )
    yield (
        PositionExecutorRequest(
            id="position-aapl-0001", symbol="aapl", amount="1.25", entry_price="101.50",
            stop_loss="0.05", time_limit=3600, take_profit="0.10",
            trailing_activation="0.04", trailing_delta="0.02", controller_id="custom-controller",
        ),
        create_position_executor, preview_position_executor,
        {
            "id": "position-aapl-0001", "type": "position_executor", "connector_name": "binance_stocks",
            "trading_pair": "AAPL-USDC", "side": "BUY", "amount": "1.25", "entry_price": "101.50",
            "triple_barrier_config": {
                "stop_loss": "0.05", "time_limit": 3600, "open_order_type": "LIMIT",
                "take_profit_order_type": "LIMIT", "stop_loss_order_type": "MARKET",
                "time_limit_order_type": "MARKET", "take_profit": "0.10",
                "trailing_stop": {"activation_price": "0.04", "trailing_delta": "0.02"},
            },
        },
    )


def _typed_request_context(mode="PAPER", phase="MARKET_OPEN"):
    request = _request()
    state = request.app.state
    state.stocks_settings.mode = mode
    state.stocks_policy = SimpleNamespace(preview=AsyncMock(return_value={"allowed": True}))
    state.stocks_ledger = SimpleNamespace(executor_record=AsyncMock(return_value=None))
    state.executor_service = SimpleNamespace(create_executor=AsyncMock(return_value={"executor_id": "created"}))
    state.stocks_async_scheduler = SimpleNamespace(
        enqueue=AsyncMock(return_value={"disposition": "QUEUED", "schedule_id": "schedule-aapl"})
    )
    state.stocks_connector = SimpleNamespace(market_phase=phase)
    return request


class RouterSemanticsTests(IsolatedAsyncioTestCase):
    async def test_typed_preview_and_create_preserve_config_and_execution_scope(self):
        for mode in ("PAPER", "LIVE"):
            for payload, create, preview, config in _typed_requests():
                with self.subTest(mode=mode, executor_type=config["type"]):
                    request = _typed_request_context(mode)
                    state = request.app.state
                    scope = {
                        "execution_scope": mode.lower(), "binance_economic_request": False if mode == "PAPER" else None,
                    }
                    self.assertEqual({"allowed": True, **scope}, await preview(payload, request))
                    state.stocks_policy.preview.assert_awaited_once_with(
                        config, "stocks_managed", payload.controller_id
                    )
                    state.stocks_ledger.executor_record.assert_not_awaited()
                    state.executor_service.create_executor.assert_not_awaited()
                    state.stocks_async_scheduler.enqueue.assert_not_awaited()

                    self.assertEqual({
                        "idempotent_replay": False, "disposition": "CREATED",
                        "executor": {"executor_id": "created"},
                        **scope,
                    }, await create(payload, request))
                    state.executor_service.create_executor.assert_awaited_once_with(
                        config, "stocks_managed", payload.controller_id
                    )
                    state.stocks_async_scheduler.enqueue.assert_not_awaited()

    async def test_closed_session_preview_does_not_queue_and_create_preserves_schedule_payload(self):
        for mode in ("PAPER", "LIVE"):
            for original, create, preview, config in _typed_requests():
                with self.subTest(mode=mode, executor_type=config["type"]):
                    payload = original.model_copy(update={"activation_policy": "QUEUE_IF_CLOSED"})
                    request = _typed_request_context(mode, "MARKET_CLOSED")
                    state = request.app.state
                    await preview(payload, request)
                    state.stocks_async_scheduler.enqueue.assert_not_awaited()
                    state.stocks_ledger.executor_record.assert_not_awaited()
                    state.executor_service.create_executor.assert_not_awaited()

                    result = await create(payload, request)
                    self.assertEqual("QUEUED", result["disposition"])
                    self.assertEqual("schedule-aapl", result["schedule_id"])
                    self.assertEqual(mode.lower(), result["execution_scope"])
                    self.assertIs(False if mode == "PAPER" else None, result["binance_economic_request"])
                    state.stocks_async_scheduler.enqueue.assert_awaited_once_with(
                        executor_config=config, request_payload=payload.model_dump(mode="json"),
                        controller_id=payload.controller_id, quote_budget=None,
                    )
                    state.stocks_ledger.executor_record.assert_not_awaited()
                    state.executor_service.create_executor.assert_not_awaited()

    async def test_typed_queue_failures_preserve_http_status_and_details(self):
        for original, create, _preview, config in _typed_requests():
            for error, status in (
                (LedgerConflict("conflict"), 409), (LedgerLimitExceeded("limit"), 422),
                (ValueError("invalid"), 422), (InvalidOperation("decimal"), 422),
                (HTTPException(status_code=429, detail="retry later"), 429), (RuntimeError("db down"), 503),
            ):
                with self.subTest(executor_type=config["type"], error=type(error).__name__):
                    request = _typed_request_context(phase="MARKET_CLOSED")
                    request.app.state.stocks_async_scheduler.enqueue.side_effect = error
                    payload = original.model_copy(update={"activation_policy": "QUEUE_IF_CLOSED"})
                    with self.assertRaises(HTTPException) as raised:
                        await create(payload, request)
                    self.assertEqual(status, raised.exception.status_code)
                    if isinstance(error, HTTPException):
                        self.assertIs(error, raised.exception)
                    else:
                        self.assertEqual(str(error), raised.exception.detail)
                    request.app.state.executor_service.create_executor.assert_not_awaited()

    async def test_typed_config_errors_keep_original_exception_boundary(self):
        for payload, create, preview, config in _typed_requests():
            builder = (
                "build_order_executor_config" if config["type"] == "order_executor"
                else "build_position_executor_config"
            )
            for route in (create, preview):
                with self.subTest(executor_type=config["type"], route=route.__name__):
                    request = _typed_request_context()
                    error = ValueError("config construction failed")
                    with patch(f"stocks_runtime.router.{builder}", side_effect=error):
                        with self.assertRaises(ValueError) as raised:
                            await route(payload, request)
                    self.assertIs(error, raised.exception)
                    request.app.state.stocks_policy.preview.assert_not_awaited()
                    request.app.state.executor_service.create_executor.assert_not_awaited()
                    request.app.state.stocks_async_scheduler.enqueue.assert_not_awaited()

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

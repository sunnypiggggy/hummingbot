from __future__ import annotations

from decimal import Decimal, InvalidOperation
from datetime import datetime
from typing import Any, Dict, Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field, model_validator

from stocks_runtime.auth import auth_user
from stocks_runtime.async_orders import activation_target, session_eligible
from stocks_runtime.executor_config import build_order_executor_config, build_position_executor_config


router = APIRouter(prefix="/stocks", tags=["Binance Stocks Runtime"], dependencies=[Depends(auth_user)])


class WhitelistUpdate(BaseModel):
    symbol: str = Field(min_length=1, max_length=15)
    enabled: bool = True
    max_position_notional: Decimal = Field(default=Decimal("200"), gt=0)


class LimitsUpdate(BaseModel):
    max_order_notional: Decimal = Field(gt=0)
    max_symbol_exposure: Decimal = Field(gt=0)
    max_managed_exposure: Decimal = Field(gt=0)


class ExecutorRequest(BaseModel):
    executor_config: Dict[str, Any]
    controller_id: str = Field(default="telegram-management-bot", min_length=3, max_length=64)


class OrderExecutorRequest(BaseModel):
    id: str = Field(min_length=8, max_length=64)
    symbol: str = Field(min_length=1, max_length=15)
    side: Literal["BUY", "SELL"]
    amount: Decimal = Field(gt=0)
    order_type: Literal["LIMIT", "MARKET"]
    price: Optional[Decimal] = Field(default=None, gt=0)
    source_owner: Optional[str] = Field(default=None, min_length=8, max_length=64)
    controller_id: str = Field(default="stocks-paper-api", min_length=3, max_length=64)
    activation_policy: Literal["IMMEDIATE", "QUEUE_IF_CLOSED"] = "IMMEDIATE"
    quote_budget: Optional[Decimal] = Field(default=None, gt=0)

    @model_validator(mode="after")
    def validate_order(self):
        if self.order_type == "LIMIT" and self.price is None:
            raise ValueError("LIMIT requires price")
        if self.order_type == "MARKET" and self.price is not None:
            raise ValueError("MARKET must not include price")
        if self.quote_budget is not None and not (self.order_type == "MARKET" and self.side == "BUY"):
            raise ValueError("quote_budget is supported only for BUY MARKET")
        return self


class PositionExecutorRequest(BaseModel):
    id: str = Field(min_length=8, max_length=64)
    symbol: str = Field(min_length=1, max_length=15)
    amount: Decimal = Field(gt=0)
    entry_order_type: Literal["LIMIT", "MARKET"] = "LIMIT"
    entry_price: Optional[Decimal] = Field(default=None, gt=0)
    stop_loss: Decimal = Field(gt=0, lt=1)
    time_limit: int = Field(gt=0, le=31_536_000)
    take_profit: Optional[Decimal] = Field(default=None, gt=0, lt=1)
    trailing_activation: Optional[Decimal] = Field(default=None, gt=0, lt=1)
    trailing_delta: Optional[Decimal] = Field(default=None, gt=0, lt=1)
    controller_id: str = Field(default="stocks-paper-api", min_length=3, max_length=64)
    activation_policy: Literal["IMMEDIATE", "QUEUE_IF_CLOSED"] = "IMMEDIATE"
    quote_budget: Optional[Decimal] = Field(default=None, gt=0)

    @model_validator(mode="after")
    def validate_position(self):
        if self.entry_order_type == "LIMIT" and self.entry_price is None:
            raise ValueError("LIMIT position entry requires entry_price")
        if self.entry_order_type == "MARKET" and self.entry_price is not None:
            raise ValueError("MARKET position entry must not include entry_price")
        if self.quote_budget is not None and self.entry_order_type != "MARKET":
            raise ValueError("quote_budget is supported only for MARKET position entry")
        if (self.trailing_activation is None) != (self.trailing_delta is None):
            raise ValueError("trailing_activation and trailing_delta must be supplied together")
        return self


class ReduceRequest(BaseModel):
    amount: Decimal = Field(gt=0)
    request_id: str = Field(min_length=8, max_length=64)


class PaperResetRequest(BaseModel):
    confirmation: str


class ScenarioMarketStateRequest(BaseModel):
    symbol: str = Field(default="AAPL", min_length=1, max_length=15)
    phase: Literal["PRE_MARKET", "MARKET_OPEN", "POST_MARKET", "OVERNIGHT", "MARKET_CLOSED"]
    trading_status: str = "TRADING"
    tradability: Literal["BUY", "SELL", "BOTH", "NONE"] = "BOTH"
    trading_date: str


class ScenarioQuoteRequest(BaseModel):
    symbol: str = Field(min_length=1, max_length=15)
    bid: Decimal = Field(gt=0)
    ask: Decimal = Field(gt=0)
    bid_size: Decimal = Field(gt=0)
    ask_size: Decimal = Field(gt=0)
    event_time_ms: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_spread(self):
        if self.bid > self.ask:
            raise ValueError("bid must not exceed ask")
        return self


def _tag_managed(payload: Any, managed_ids: set[str]) -> Any:
    if isinstance(payload, list):
        return [_tag_managed(item, managed_ids) for item in payload]
    if not isinstance(payload, dict):
        return payload
    item = dict(payload)
    client_id = str(item.get("clientOrderId", item.get("client_order_id", "")))
    item["managed_by_this_runtime"] = client_id in managed_ids
    item["external_inventory_ownership"] = "unknown" if client_id not in managed_ids else "not_applicable"
    return item


@router.get("/health")
async def health(request: Request) -> Dict[str, Any]:
    ledger = request.app.state.stocks_ledger
    connector = getattr(request.app.state, "stocks_connector", None)
    settings = request.app.state.stocks_settings
    payload = {
        "status": "healthy" if await ledger.is_fresh() else "degraded",
        "runtime_mode": settings.mode,
        "live_authorized": settings.mode == "LIVE" and settings.live_authorized,
        "disclaimer_confirmed": settings.disclaimer_confirmed,
        "connector_ready": bool(connector and connector.ready),
        "market_phase": connector.market_phase if connector else "PUBLIC_DATA_ONLY",
        "position_source": "paper_ledger" if settings.mode == "PAPER" else "managed_ledger_non_authoritative",
        "external_positions_unknown": settings.mode != "PAPER",
        "economic_requests_enabled": settings.mode == "LIVE" and settings.live_authorized,
        "scenario_mode": settings.scenario_mode,
    }
    if settings.mode == "PAPER":
        broker = request.app.state.stocks_paper_broker
        account = await broker.account()
        payload.update({
            "paper_run_id": broker.run_id,
            "paper_equity": account["equity"],
            "paper_available_usdc": account["available_cash"],
            "paper_recovery_required": account["recovery_required"],
            "economic_http_request_count": getattr(connector, "economic_http_request_count", 0),
            "economic_requests_enabled": False,
        })
        if account["recovery_required"] or not connector:
            payload["status"] = "degraded"
    return payload


@router.get("/markets")
async def markets(request: Request) -> Any:
    return request.app.state.stocks_market_info


@router.get("/whitelist")
async def whitelist(request: Request) -> Dict[str, Any]:
    return {"items": await request.app.state.stocks_ledger.whitelist_rows()}


@router.put("/whitelist/{symbol}")
async def put_whitelist(symbol: str, payload: WhitelistUpdate, request: Request) -> Dict[str, Any]:
    symbol = symbol.upper().strip()
    if payload.symbol.upper().strip() != symbol:
        raise HTTPException(status_code=422, detail="path and payload symbols must match")
    if symbol not in request.app.state.stocks_policy.symbols:
        raise HTTPException(status_code=409, detail=f"{symbol} is not a current direct-equity market")
    try:
        row = await request.app.state.stocks_ledger.upsert_whitelist(
            symbol, payload.enabled, payload.max_position_notional
        )
    except Exception as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"updated": True, "item": row}


@router.delete("/whitelist/{symbol}")
async def delete_whitelist(symbol: str, request: Request) -> Dict[str, Any]:
    deleted = await request.app.state.stocks_ledger.delete_whitelist(symbol.upper().strip())
    return {
        "deleted": deleted,
        "existing_positions_unchanged": True,
        "new_buy_blocked": True,
    }


@router.get("/limits")
async def limits(request: Request) -> Dict[str, Any]:
    active = await request.app.state.stocks_ledger.active_limits()
    hard = request.app.state.stocks_ledger.hard_limits
    return {
        "active": {
            "max_order_notional": str(active.max_order_notional),
            "max_symbol_exposure": str(active.max_symbol_exposure),
            "max_managed_exposure": str(active.max_managed_exposure),
        },
        "hard_ceiling": {
            "max_order_notional": str(hard.max_order_notional),
            "max_symbol_exposure": str(hard.max_symbol_exposure),
            "max_managed_exposure": str(hard.max_managed_exposure),
        },
    }


@router.put("/limits")
async def put_limits(payload: LimitsUpdate, request: Request) -> Dict[str, Any]:
    try:
        active = await request.app.state.stocks_ledger.update_limits(
            payload.max_order_notional,
            payload.max_symbol_exposure,
            payload.max_managed_exposure,
        )
    except Exception as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {
        "updated": True,
        "active": {
            "max_order_notional": str(active.max_order_notional),
            "max_symbol_exposure": str(active.max_symbol_exposure),
            "max_managed_exposure": str(active.max_managed_exposure),
        },
    }


@router.get("/quotes/{symbol}")
async def quote(symbol: str, request: Request) -> Any:
    if not request.app.state.stocks_read_client.api_key:
        raise HTTPException(status_code=503, detail="credential-free Paper mode has no live Binance quote")
    return await request.app.state.stocks_read_client.quote(symbol)


@router.get("/market-status/{symbol}")
async def market_status(symbol: str, request: Request) -> Dict[str, Any]:
    symbol = symbol.upper()
    connector = getattr(request.app.state, "stocks_connector", None)
    return {
        "symbol": symbol,
        "market_phase": connector.market_phase if connector else "PUBLIC_DATA_ONLY",
        "trading_status": getattr(connector, "_trading_status", {}).get(symbol, "UNKNOWN"),
        "tradability": getattr(connector, "_tradability", {}).get(symbol, "UNKNOWN"),
        "quote_fresh": bool(connector and connector.latest_quote(symbol)),
    }


@router.get("/account-summary")
async def account_summary(request: Request) -> Dict[str, Any]:
    if request.app.state.stocks_settings.mode == "PAPER":
        return await request.app.state.stocks_paper_broker.account()
    summary = await request.app.state.stocks_ledger.summary()
    summary.update({
        "runtime_mode": request.app.state.stocks_settings.mode,
        "account_scope": "this runtime only",
        "account_total_equity_positions_available": False,
    })
    return summary


@router.get("/managed-positions")
async def managed_positions(request: Request) -> Dict[str, Any]:
    if request.app.state.stocks_settings.mode == "PAPER":
        account = await request.app.state.stocks_paper_broker.account()
        return {
            "position_source": "paper_ledger",
            "external_positions_unknown": False,
            "account_scope": "paper",
            "items": account["positions"],
        }
    return {
        "position_source": "managed_ledger_non_authoritative",
        "external_positions_unknown": True,
        "items": await request.app.state.stocks_ledger.managed_position_rows(),
    }


@router.get("/open-orders")
async def open_orders(request: Request, symbol: Optional[str] = None) -> Any:
    if request.app.state.stocks_settings.mode == "PAPER":
        return await request.app.state.stocks_paper_broker.orders(symbol=symbol, open_only=True)
    ledger = request.app.state.stocks_ledger
    return _tag_managed(
        await request.app.state.stocks_read_client.open_orders(symbol),
        await ledger.managed_order_ids(),
    )


@router.get("/order-history")
async def order_history(
    request: Request, symbol: Optional[str] = None, limit: int = Query(100, ge=1, le=500)
) -> Any:
    if request.app.state.stocks_settings.mode == "PAPER":
        return await request.app.state.stocks_paper_broker.orders(symbol=symbol, limit=limit)
    ledger = request.app.state.stocks_ledger
    return _tag_managed(
        await request.app.state.stocks_read_client.order_history(symbol, limit),
        await ledger.managed_order_ids(),
    )


@router.get("/trade-history")
async def trade_history(
    request: Request, symbol: Optional[str] = None, limit: int = Query(100, ge=1, le=500)
) -> Any:
    if request.app.state.stocks_settings.mode == "PAPER":
        return await request.app.state.stocks_paper_broker.trades(symbol=symbol, limit=limit)
    ledger = request.app.state.stocks_ledger
    return _tag_managed(
        await request.app.state.stocks_read_client.trade_history(symbol, limit),
        await ledger.managed_order_ids(),
    )


@router.post("/executors/preview")
async def preview_executor(payload: ExecutorRequest, request: Request) -> Dict[str, Any]:
    return await _preview_managed_executor(payload.executor_config, payload.controller_id, request)


async def _preview_managed_executor(
    executor_config: Dict[str, Any], controller_id: str, request: Request
) -> Dict[str, Any]:
    try:
        result = await request.app.state.stocks_policy.preview(
            executor_config, "stocks_managed", controller_id
        )
    except (ValueError, InvalidOperation) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    paper = request.app.state.stocks_settings.mode == "PAPER"
    result.update({
        "execution_scope": "paper" if paper else "live",
        "binance_economic_request": False if paper else None,
    })
    return result


@router.get("/executors")
async def list_managed_executors(request: Request, active_only: bool = True) -> Dict[str, Any]:
    return {"items": await request.app.state.stocks_ledger.executor_rows(active_only=active_only)}


@router.post("/executors")
async def create_executor(payload: ExecutorRequest, request: Request) -> Dict[str, Any]:
    return await _create_managed_executor(payload.executor_config, payload.controller_id, request)


async def _create_managed_executor(
    executor_config: Dict[str, Any], controller_id: str, request: Request
) -> Dict[str, Any]:
    executor_id = str(executor_config.get("id", ""))
    existing = await request.app.state.stocks_ledger.executor_record(executor_id)
    if existing is not None:
        scheduled = await request.app.state.stocks_ledger.scheduled_by_executor(executor_id)
        if scheduled is not None:
            return {
                "idempotent_replay": True, "disposition": "QUEUED",
                "schedule_id": scheduled["schedule_id"], "schedule": scheduled,
            }
        return {"idempotent_replay": True, "disposition": "CREATED", "executor": existing}
    try:
        created = await request.app.state.executor_service.create_executor(
            executor_config, "stocks_managed", controller_id
        )
    except (ValueError, InvalidOperation) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"idempotent_replay": False, "disposition": "CREATED", "executor": created}


async def _create_or_queue(
    *, executor_config: Dict[str, Any], payload: BaseModel, controller_id: str,
    activation_policy: str, quote_budget: Optional[Decimal], request: Request,
) -> Dict[str, Any]:
    target = activation_target(executor_config)
    connector = getattr(request.app.state, "stocks_connector", None)
    phase = str(getattr(connector, "market_phase", "UNKNOWN") or "UNKNOWN").upper()
    mode = str(request.app.state.stocks_settings.mode).upper()
    if (
        activation_policy == "QUEUE_IF_CLOSED" and mode in {"PAPER", "LIVE"}
        and not session_eligible(target, phase)
    ):
        return await request.app.state.stocks_async_scheduler.enqueue(
            executor_config=executor_config,
            request_payload=payload.model_dump(mode="json"),
            controller_id=controller_id,
            quote_budget=quote_budget,
        )
    if quote_budget is not None and target == "MARKET_OPEN":
        symbol = str(executor_config["trading_pair"]).rsplit("-", 1)[0]
        quote = connector.latest_quote(symbol) if connector is not None else None
        if quote is None:
            try:
                raw_quote = await request.app.state.stocks_read_client.quote(symbol)
                if connector is not None:
                    connector.process_quote_event(raw_quote)
                    quote = connector.latest_quote(symbol)
            except Exception as exc:
                raise HTTPException(status_code=409, detail=f"fresh {symbol} quote is unavailable: {exc}") from exc
        if quote is None:
            raise HTTPException(status_code=409, detail=f"fresh {symbol} quote is unavailable")
        ask = Decimal(str(quote[1]))
        pair = str(executor_config["trading_pair"])
        amount = connector.quantize_order_amount(pair, Decimal(quote_budget) / ask)
        if amount <= 0 or amount * ask > Decimal(quote_budget):
            raise HTTPException(status_code=409, detail="fixed quote budget cannot produce a valid quantity")
        executor_config = dict(executor_config)
        executor_config["amount"] = str(amount)
    return await _create_managed_executor(executor_config, controller_id, request)


@router.post("/order-executors")
async def create_order_executor(payload: OrderExecutorRequest, request: Request) -> Dict[str, Any]:
    """Create a real OrderExecutor lifecycle backed by the local Paper venue."""
    config = build_order_executor_config(
        executor_id=payload.id,
        symbol=payload.symbol,
        side=payload.side,
        amount=payload.amount,
        order_type=payload.order_type,
        price=payload.price,
        source_owner=payload.source_owner,
    )
    result = await _create_or_queue(
        executor_config=config, payload=payload, controller_id=payload.controller_id,
        activation_policy=payload.activation_policy, quote_budget=payload.quote_budget, request=request,
    )
    result["execution_scope"] = "paper" if request.app.state.stocks_settings.mode == "PAPER" else "live"
    result["binance_economic_request"] = False if request.app.state.stocks_settings.mode == "PAPER" else None
    return result


@router.post("/order-executors/preview")
async def preview_order_executor(payload: OrderExecutorRequest, request: Request) -> Dict[str, Any]:
    config = build_order_executor_config(
        executor_id=payload.id,
        symbol=payload.symbol,
        side=payload.side,
        amount=payload.amount,
        order_type=payload.order_type,
        price=payload.price,
        source_owner=payload.source_owner,
    )
    return await _preview_managed_executor(config, payload.controller_id, request)


@router.post("/position-executors")
async def create_position_executor(payload: PositionExecutorRequest, request: Request) -> Dict[str, Any]:
    """Create a long PositionExecutor with real barrier and time-exit processing."""
    config = build_position_executor_config(
        executor_id=payload.id,
        symbol=payload.symbol,
        amount=payload.amount,
        entry_order_type=payload.entry_order_type,
        entry_price=payload.entry_price,
        stop_loss=payload.stop_loss,
        time_limit=payload.time_limit,
        take_profit=payload.take_profit,
        trailing_activation=payload.trailing_activation,
        trailing_delta=payload.trailing_delta,
    )
    result = await _create_or_queue(
        executor_config=config, payload=payload, controller_id=payload.controller_id,
        activation_policy=payload.activation_policy, quote_budget=payload.quote_budget, request=request,
    )
    result["execution_scope"] = "paper" if request.app.state.stocks_settings.mode == "PAPER" else "live"
    result["binance_economic_request"] = False if request.app.state.stocks_settings.mode == "PAPER" else None
    return result


@router.post("/position-executors/preview")
async def preview_position_executor(payload: PositionExecutorRequest, request: Request) -> Dict[str, Any]:
    config = build_position_executor_config(
        executor_id=payload.id,
        symbol=payload.symbol,
        amount=payload.amount,
        entry_order_type=payload.entry_order_type,
        entry_price=payload.entry_price,
        stop_loss=payload.stop_loss,
        time_limit=payload.time_limit,
        take_profit=payload.take_profit,
        trailing_activation=payload.trailing_activation,
        trailing_delta=payload.trailing_delta,
    )
    return await _preview_managed_executor(config, payload.controller_id, request)


@router.get("/scheduled-executors")
async def scheduled_executors(
    request: Request,
    active_only: bool = True,
    updated_after: Optional[datetime] = None,
    limit: int = Query(100, ge=1, le=500),
) -> Dict[str, Any]:
    return {"items": await request.app.state.stocks_ledger.scheduled_rows(
        active_only=active_only, updated_after=updated_after, limit=limit
    )}


@router.get("/scheduled-executors/{schedule_id}")
async def scheduled_executor(schedule_id: str, request: Request) -> Dict[str, Any]:
    row = await request.app.state.stocks_ledger.scheduled_record(schedule_id)
    if row is None:
        raise HTTPException(status_code=404, detail="scheduled executor not found")
    return row


@router.post("/scheduled-executors/{schedule_id}/refresh")
async def refresh_scheduled_executor(schedule_id: str, request: Request) -> Dict[str, Any]:
    if await request.app.state.stocks_ledger.scheduled_record(schedule_id) is None:
        raise HTTPException(status_code=404, detail="scheduled executor not found")
    await request.app.state.stocks_async_scheduler.tick()
    return await request.app.state.stocks_ledger.scheduled_record(schedule_id)


@router.post("/scheduled-executors/{schedule_id}/cancel")
async def cancel_scheduled_executor(schedule_id: str, request: Request) -> Dict[str, Any]:
    try:
        result = await request.app.state.stocks_async_scheduler.cancel(schedule_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if result.get("executor_active"):
        raise HTTPException(
            status_code=409,
            detail="scheduled request is already active; use the Executor cancel/close endpoint",
        )
    return result


@router.get("/executors/{executor_id}")
async def executor_result(executor_id: str, request: Request) -> Dict[str, Any]:
    record = await request.app.state.stocks_ledger.executor_record(executor_id)
    runtime = await request.app.state.executor_service.get_executor(executor_id)
    if record is None and not runtime:
        raise HTTPException(status_code=404, detail="managed Stocks executor not found")
    return {"ledger": record, "runtime": runtime}


@router.post("/executors/{executor_id}/reduce")
async def reduce_executor(executor_id: str, payload: ReduceRequest, request: Request) -> Dict[str, Any]:
    record = await request.app.state.stocks_ledger.executor_record(executor_id)
    if not record or record.get("executor_type") != "position_executor":
        raise HTTPException(status_code=404, detail="managed PositionExecutor not found")
    symbol = str(record["symbol"])
    reduce_config = {
        "id": f"reduce-{payload.request_id}"[:64],
        "type": "order_executor",
        "connector_name": "binance_stocks",
        "trading_pair": f"{symbol}-USDC",
        "side": "SELL",
        "amount": str(payload.amount),
        "execution_strategy": "MARKET",
        "position_action": "CLOSE",
        "managed_source_owner": executor_id,
    }
    existing = await request.app.state.stocks_ledger.executor_record(reduce_config["id"])
    if existing is not None:
        return {"idempotent_replay": True, "executor": existing}
    try:
        created = await request.app.state.executor_service.create_executor(
            reduce_config, "stocks_managed", "telegram-management-bot"
        )
    except Exception as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"idempotent_replay": False, "executor": created}


async def _managed_executor_or_404(request: Request, executor_id: str) -> Dict[str, Any]:
    service = request.app.state.executor_service
    executor = await service.get_executor(executor_id)
    if not executor or executor.get("account_name") != "stocks_managed" or executor.get("connector_name") != "binance_stocks":
        raise HTTPException(status_code=404, detail="managed Stocks executor not found")
    return executor


@router.post("/executors/{executor_id}/cancel")
async def cancel_executor(executor_id: str, request: Request) -> Dict[str, Any]:
    await _managed_executor_or_404(request, executor_id)
    if request.app.state.stocks_settings.mode == "SHADOW":
        raise HTTPException(status_code=409, detail="SHADOW mode has no executable order")
    return await request.app.state.executor_service.stop_executor(executor_id, keep_position=True)


@router.post("/executors/{executor_id}/close")
async def close_executor(executor_id: str, request: Request) -> Dict[str, Any]:
    await _managed_executor_or_404(request, executor_id)
    if request.app.state.stocks_settings.mode == "SHADOW":
        raise HTTPException(status_code=409, detail="SHADOW mode has no executable position")
    return await request.app.state.executor_service.stop_executor(executor_id, keep_position=False)


@router.get("/paper/account")
async def paper_account(request: Request) -> Dict[str, Any]:
    _paper_only(request)
    return await request.app.state.stocks_paper_broker.account()


@router.get("/paper/summary")
async def paper_summary(request: Request) -> Dict[str, Any]:
    _paper_only(request)
    connector = getattr(request.app.state, "stocks_connector", None)
    service = request.app.state.executor_service
    return await request.app.state.stocks_paper_broker.summary(
        market_phase=connector.market_phase if connector else "PUBLIC_DATA_ONLY",
        connector_ready=bool(connector and connector.ready),
        active_executor_count=len(service._active_executors),
    )


@router.get("/paper/positions")
async def paper_positions(request: Request) -> Dict[str, Any]:
    _paper_only(request)
    account = await request.app.state.stocks_paper_broker.account()
    return {"account_scope": "paper", "paper_run_id": account["paper_run_id"], "items": account["positions"]}


@router.get("/paper/orders")
async def paper_orders(
    request: Request,
    symbol: Optional[str] = None,
    open_only: bool = False,
    limit: int = Query(500, ge=1, le=2000),
) -> Dict[str, Any]:
    _paper_only(request)
    return {"account_scope": "paper", "items": await request.app.state.stocks_paper_broker.orders(
        symbol=symbol, open_only=open_only, limit=limit
    )}


@router.get("/paper/trades")
async def paper_trades(
    request: Request, symbol: Optional[str] = None, limit: int = Query(500, ge=1, le=2000)
) -> Dict[str, Any]:
    _paper_only(request)
    return {"account_scope": "paper", "items": await request.app.state.stocks_paper_broker.trades(
        symbol=symbol, limit=limit
    )}


@router.get("/paper/performance")
async def paper_performance(
    request: Request, window: str = Query("all", pattern="^(4h|24h|7d|all)$")
) -> Dict[str, Any]:
    _paper_only(request)
    seconds = {"4h": 4 * 3600, "24h": 24 * 3600, "7d": 7 * 86400, "all": None}[window]
    result = await request.app.state.stocks_paper_broker.performance(seconds)
    result["window"] = window
    return result


@router.get("/paper/equity")
async def paper_equity(
    request: Request,
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
    limit: int = Query(5000, ge=1, le=10000),
) -> Dict[str, Any]:
    _paper_only(request)
    return {"account_scope": "paper", "items": await request.app.state.stocks_paper_broker.equity(
        start=start, end=end, limit=limit
    )}


@router.post("/paper/reset")
async def reset_paper(payload: PaperResetRequest, request: Request) -> Dict[str, Any]:
    _paper_only(request)
    try:
        return await request.app.state.stocks_paper_broker.reset(
            payload.confirmation, len(request.app.state.executor_service._active_executors)
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


def _paper_only(request: Request) -> None:
    if request.app.state.stocks_settings.mode != "PAPER":
        raise HTTPException(status_code=409, detail="endpoint is available only in PAPER mode")


def _scenario_only(request: Request) -> None:
    _paper_only(request)
    if not request.app.state.stocks_settings.scenario_mode:
        raise HTTPException(status_code=404, detail="scenario endpoint is disabled")


@router.post("/scenario/market-state", include_in_schema=False)
async def scenario_market_state(payload: ScenarioMarketStateRequest, request: Request) -> Dict[str, Any]:
    _scenario_only(request)
    connector = request.app.state.stocks_connector
    if connector is None:
        raise HTTPException(status_code=503, detail="paper connector is not ready")
    symbol = payload.symbol.upper()
    connector.process_market_state_event({"e": "calendar", "phase": payload.phase, "tradingDate": payload.trading_date})
    connector.process_market_state_event({"e": "tradingStatus", "s": symbol, "status": payload.trading_status})
    connector.process_market_state_event({"e": "tradability", "s": symbol, "tradability": payload.tradability})
    request.app.state.stocks_paper_broker.update_market_state(
        payload.phase,
        {symbol: payload.trading_status.upper()},
        {symbol: payload.tradability.upper()},
        payload.trading_date,
    )
    return {"accepted": True, "phase": payload.phase, "symbol": symbol}


@router.post("/scenario/quote", include_in_schema=False)
async def scenario_quote(payload: ScenarioQuoteRequest, request: Request) -> Dict[str, Any]:
    _scenario_only(request)
    connector = request.app.state.stocks_connector
    if connector is None:
        raise HTTPException(status_code=503, detail="paper connector is not ready")
    event = {
        "e": "quote", "s": payload.symbol.upper(), "bp": str(payload.bid), "ap": str(payload.ask),
        "bs": str(payload.bid_size), "as": str(payload.ask_size), "T": payload.event_time_ms,
    }
    connector.process_quote_event(event)
    return {"accepted": True, "event_time_ms": payload.event_time_ms}

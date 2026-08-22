from __future__ import annotations

from decimal import Decimal, InvalidOperation
from datetime import datetime
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field

from stocks_runtime.auth import auth_user


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


class ReduceRequest(BaseModel):
    amount: Decimal = Field(gt=0)
    request_id: str = Field(min_length=8, max_length=64)


class PaperResetRequest(BaseModel):
    confirmation: str


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
    try:
        return await request.app.state.stocks_policy.preview(
            payload.executor_config, "stocks_managed", payload.controller_id
        )
    except (ValueError, InvalidOperation) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.get("/executors")
async def list_managed_executors(request: Request, active_only: bool = True) -> Dict[str, Any]:
    return {"items": await request.app.state.stocks_ledger.executor_rows(active_only=active_only)}


@router.post("/executors")
async def create_executor(payload: ExecutorRequest, request: Request) -> Dict[str, Any]:
    executor_id = str(payload.executor_config.get("id", ""))
    existing = await request.app.state.stocks_ledger.executor_record(executor_id)
    if existing is not None:
        return {"idempotent_replay": True, "executor": existing}
    try:
        created = await request.app.state.executor_service.create_executor(
            payload.executor_config, "stocks_managed", payload.controller_id
        )
    except (ValueError, InvalidOperation) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"idempotent_replay": False, "executor": created}


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

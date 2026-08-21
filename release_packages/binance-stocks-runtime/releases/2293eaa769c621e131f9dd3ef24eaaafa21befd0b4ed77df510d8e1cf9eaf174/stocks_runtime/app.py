from __future__ import annotations

import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse

from stocks_runtime.settings import dedicated_database_url

os.environ["DATABASE_URL"] = dedicated_database_url(
    os.environ["DATABASE_URL"], os.getenv("BINANCE_STOCKS_DATABASE_NAME", "hummingbot_stocks")
)

from config import settings as api_settings  # noqa: E402
from database import AsyncDatabaseManager  # noqa: E402
from hummingbot.client.config.config_crypt import ETHKeyFileSecretManger  # noqa: E402
from hummingbot.strategy_v2.executors.binance_stocks_order_executor import (  # noqa: E402
    BinanceStocksOrderExecutor,
)
from hummingbot.strategy_v2.executors.binance_stocks_position_executor import (  # noqa: E402
    BinanceStocksPositionExecutor,
)
from hummingbot.strategy_v2.executors.order_executor.data_types import OrderExecutorConfig  # noqa: E402
from hummingbot.strategy_v2.executors.position_executor.data_types import PositionExecutorConfig  # noqa: E402

from stocks_runtime.binance_client import BinanceStocksReadClient  # noqa: E402
from stocks_runtime.auth import auth_user  # noqa: E402
from stocks_runtime.ledger import LedgerLimits, PostgresManagedLedger  # noqa: E402
from stocks_runtime.policy import StocksExecutorPolicy  # noqa: E402
from stocks_runtime.router import router as stocks_router  # noqa: E402
from stocks_runtime.settings import StocksRuntimeSettings  # noqa: E402
from routers import executors as executors_router  # noqa: E402
from services.accounts_service import AccountsService  # noqa: E402
from services.executor_service import ExecutorService  # noqa: E402
from services.market_data_service import MarketDataService  # noqa: E402
from services.trading_service import TradingService  # noqa: E402
from services.unified_connector_service import UnifiedConnectorService  # noqa: E402
from utils.security import BackendAPISecurity  # noqa: E402


logger = logging.getLogger(__name__)


def _items(payload: Any) -> list[Dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        for key in ("symbols", "data", "items"):
            if isinstance(payload.get(key), list):
                return [item for item in payload[key] if isinstance(item, dict)]
    return []


def _direct_symbols(market_info: Any) -> set[str]:
    result = set()
    for item in _items(market_info):
        symbol = str(item.get("symbol", "")).upper()
        status = str(item.get("status", item.get("tradingStatus", "TRADING"))).upper()
        if symbol and item.get("tokenized") is not True and status not in {"DELISTED", "INACTIVE", "UNAVAILABLE"}:
            result.add(symbol)
    return result


def _ensure_master_account_scaffold() -> None:
    """Create only non-secret API config files in the isolated credential volume."""
    root = Path("/hummingbot-api/bots/credentials/master_account")
    connectors = root / "connectors"
    connectors.mkdir(parents=True, exist_ok=True)
    defaults = {
        "conf_client.yml": "global_token:\n  global_token_name: USDC\n",
        "conf_fee_overrides.yml": "{}\n",
        "hummingbot_logs.yml": "{}\n",
    }
    for name, content in defaults.items():
        path = root / name
        if not path.exists():
            path.write_text(content, encoding="utf-8")


def _trading_date(market_info: Any) -> str | None:
    if isinstance(market_info, dict):
        value = market_info.get("tradingDate", market_info.get("trading_date"))
        if value:
            return str(value)
    for item in _items(market_info):
        value = item.get("tradingDate", item.get("trading_date"))
        if value:
            return str(value)
    return None


async def _refresh_runtime(app, stop: asyncio.Event) -> None:
    client = app.state.stocks_read_client
    ledger = app.state.stocks_ledger
    policy = app.state.stocks_policy
    while not stop.is_set():
        try:
            if app.state.stocks_settings.mode == "PAPER" and not client.api_key:
                total = available = Decimal("0")
            else:
                total, available = await client.funding_usdc()
            ledger.set_quote_balances(total, available)
            symbols = _direct_symbols(app.state.stocks_market_info)
            prices: Dict[str, Decimal] = {}
            # Only managed/active symbols need repeated private-risk valuation.
            managed = await ledger.managed_positions()
            active_pairs = {
                str(meta.get("trading_pair", "")).split("-")[0]
                for meta in app.state.executor_service._executor_metadata.values()
            }
            for symbol in (set(managed) | active_pairs):
                if symbol in symbols:
                    quote = await client.quote(symbol)
                    bid = Decimal(str(quote.get("bidPrice", quote.get("bid", "0"))))
                    ask = Decimal(str(quote.get("askPrice", quote.get("ask", "0"))))
                    prices[symbol] = ask if ask > 0 else bid
            trading_date = _trading_date(app.state.stocks_market_info)
            policy.update_market(symbols, prices, trading_date)
            if client.api_key:
                await _detect_external_activity(app)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.error("Stocks reconciliation refresh failed: %s", exc, exc_info=True)
        try:
            await asyncio.wait_for(stop.wait(), timeout=5)
        except asyncio.TimeoutError:
            pass


async def _detect_external_activity(app) -> None:
    """Latch only new post-activation non-runtime trades; historical holdings stay external."""
    activated_ms = app.state.stocks_activation_ms
    if not activated_ms:
        return
    history = _items(await app.state.stocks_read_client.trade_history(limit=100))
    managed = await app.state.stocks_ledger.managed_order_ids()
    for trade in history:
        ts = int(trade.get("time", trade.get("tradeTime", trade.get("updateTime", 0))) or 0)
        if 0 < ts < 10**12:
            ts *= 1000
        client_id = str(trade.get("clientOrderId", ""))
        if ts >= activated_ms and client_id not in managed:
            await app.state.stocks_ledger.mark_external_activity({
                "time": ts,
                "symbol": trade.get("symbol"),
                "side": trade.get("side"),
                "reason": "non_runtime_prefix_equity_trade_after_activation",
            })
            for executor_id, executor in list(app.state.executor_service._active_executors.items()):
                metadata = app.state.executor_service._executor_metadata.get(executor_id, {})
                if metadata.get("account_name") != "stocks_managed":
                    continue
                if hasattr(executor, "freeze_entry_due_external_activity"):
                    executor.freeze_entry_due_external_activity()
                elif str(metadata.get("config", {}).get("side", "")).upper() == "BUY":
                    await app.state.executor_service.stop_executor(executor_id, keep_position=True)
            return


def _install_executor_policy(app) -> None:
    service = app.state.executor_service
    service.EXECUTOR_REGISTRY = {
        "order_executor": (BinanceStocksOrderExecutor, OrderExecutorConfig),
        "position_executor": (BinanceStocksPositionExecutor, PositionExecutorConfig),
    }
    original_create = service.create_executor
    original_complete = service._handle_executor_completion

    async def guarded_create(executor_config, account_name=None, controller_id=None):
        pair = str(executor_config.get("trading_pair", ""))
        symbol = pair.rsplit("-", 1)[0].upper()
        if symbol in app.state.stocks_policy.symbols:
            try:
                quote = await app.state.stocks_read_client.quote(symbol)
                bid = Decimal(str(quote.get("bidPrice", quote.get("bid", "0"))))
                ask = Decimal(str(quote.get("askPrice", quote.get("ask", "0"))))
                executable = ask if ask > 0 else bid
                if executable > 0:
                    app.state.stocks_policy.prices[symbol] = executable
            except Exception:
                # Credential-free Paper validates explicit LIMIT prices. It
                # cannot pretend to have a fresh BBO for MARKET requests.
                if app.state.stocks_settings.mode != "PAPER":
                    raise
        preview = await app.state.stocks_policy.validate_and_reserve(
            executor_config, account_name or "stocks_managed", controller_id
        )
        executor_id = str(executor_config["id"])
        if not preview["would_submit"]:
            await app.state.stocks_ledger.release_intent(executor_id, f"{app.state.stocks_settings.mode}_VALIDATED")
            return {
                "executor_id": executor_id,
                "executor_type": executor_config["type"],
                "connector_name": "binance_stocks",
                "trading_pair": executor_config["trading_pair"],
                "controller_id": controller_id or "stocks-runtime",
                "status": f"{app.state.stocks_settings.mode}_VALIDATED_NO_ORDER",
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
        if not app.state.stocks_settings.disclaimer_confirmed:
            await app.state.stocks_ledger.release_intent(executor_id, "BLOCKED_DISCLAIMER")
            raise HTTPException(status_code=409, detail="Binance Stocks disclaimer is not confirmed")
        try:
            return await original_create(executor_config, "stocks_managed", controller_id or "stocks-runtime")
        except Exception:
            await app.state.stocks_ledger.release_intent(executor_id, "CREATE_FAILED")
            raise

    async def guarded_complete(executor_id: str):
        executor = service._active_executors.get(executor_id)
        close_type = getattr(getattr(executor, "close_type", None), "name", "COMPLETED")
        await original_complete(executor_id)
        await app.state.stocks_ledger.release_intent(executor_id, close_type or "COMPLETED")

    service.create_executor = guarded_create
    service._handle_executor_completion = guarded_complete


@asynccontextmanager
async def stocks_lifespan(runtime_app):
    settings = StocksRuntimeSettings.from_env()
    _ensure_master_account_scaffold()
    secrets_manager = ETHKeyFileSecretManger(password=api_settings.security.config_password)
    if BackendAPISecurity.new_password_required():
        BackendAPISecurity.store_password_verification(secrets_manager)
    if not BackendAPISecurity.login_account("master_account", secrets_manager):
        raise RuntimeError("CONFIG_PASSWORD does not match the isolated Stocks credential volume")
    db_manager = AsyncDatabaseManager(settings.database_url)
    await db_manager.create_tables()
    connector_service = UnifiedConnectorService(secrets_manager=secrets_manager, db_manager=db_manager)
    market_data_service = MarketDataService(connector_service=connector_service, quote_token="USDC")
    connector_service.set_rate_provider(market_data_service)
    trading_service = TradingService(connector_service=connector_service, market_data_service=market_data_service)
    accounts_service = AccountsService(
        db_manager=db_manager,
        connector_service=connector_service,
        market_data_service=market_data_service,
        trading_service=trading_service,
    )
    executor_service = ExecutorService(
        trading_service=trading_service,
        db_manager=db_manager,
        default_account=settings.account_name,
        update_interval=1.0,
        max_retries=3,
    )
    runtime_app.state.db_manager = db_manager
    runtime_app.state.connector_service = connector_service
    runtime_app.state.market_data_service = market_data_service
    runtime_app.state.trading_service = trading_service
    runtime_app.state.accounts_service = accounts_service
    runtime_app.state.executor_service = executor_service
    await connector_service.initialize_all_trading_connectors()
    market_data_service.start()
    executor_service.start()
    await executor_service.cleanup_orphaned_executors()
    await executor_service.recover_positions_from_db()
    ledger = client = None
    try:
        limits = LedgerLimits(
            max_order_notional=settings.max_order_notional,
            max_managed_exposure=settings.max_managed_exposure,
            daily_loss_limit=settings.daily_loss_limit,
            max_symbol_exposure=settings.max_symbol_exposure,
        )
        ledger = PostgresManagedLedger(settings.database_url, limits, settings.order_prefix)
        await ledger.initialize()
        await ledger.set_live_authorized(settings.mode == "LIVE" and settings.live_authorized)
        credentials = settings.read_credentials()
        client = BinanceStocksReadClient(
            settings.rest_url,
            str(credentials.get("binance_stocks_api_key", "")),
            str(credentials.get("binance_stocks_api_secret", "")),
        )
        await client.start()
        if credentials:
            market_info = await client.exchange_info()
        else:
            paper_symbols = [
                symbol.strip().upper()
                for symbol in os.getenv("BINANCE_STOCKS_PAPER_SYMBOLS", "AAPL,TSLA,SPY,QQQ").split(",")
                if symbol.strip()
            ]
            market_info = {
                "mode": "PAPER_STATIC_CATALOG",
                "symbols": [
                    {"symbol": symbol, "status": "PAPER_ONLY", "fractionalTrading": True}
                    for symbol in paper_symbols
                ],
            }
        connector = None
        if credentials:
            accounts = runtime_app.state.accounts_service
            if settings.account_name not in accounts.list_accounts():
                accounts.add_account(settings.account_name)
            await accounts.add_credentials(settings.account_name, "binance_stocks", credentials)
            connector = await accounts.get_connector_instance(settings.account_name, "binance_stocks")
            from hummingbot.connector.exchange.binance_stocks.binance_stocks_position_provider import (
                ManagedLedgerEquityPositionProvider,
            )
            connector._position_provider = ManagedLedgerEquityPositionProvider(ledger)
        runtime_app.state.stocks_settings = settings
        runtime_app.state.stocks_ledger = ledger
        runtime_app.state.stocks_read_client = client
        runtime_app.state.stocks_market_info = market_info
        runtime_app.state.stocks_connector = connector
        runtime_app.state.stocks_activation_ms = int(time.time() * 1000)
        runtime_app.state.stocks_policy = StocksExecutorPolicy(ledger, settings.mode, settings.live_authorized)
        direct_symbols = _direct_symbols(market_info)
        default_whitelist = {
            symbol.strip().upper()
            for symbol in os.getenv("BINANCE_STOCKS_DEFAULT_WHITELIST", "AAPL,TSLA,SPY,QQQ").split(",")
            if symbol.strip().upper() in direct_symbols
        }
        await ledger.ensure_whitelist(default_whitelist)
        runtime_app.state.stocks_policy.update_market(direct_symbols, {}, _trading_date(market_info))
        _install_executor_policy(runtime_app)
        stop = asyncio.Event()
        refresh_task = asyncio.create_task(_refresh_runtime(runtime_app, stop))
        try:
            yield
        finally:
            stop.set()
            refresh_task.cancel()
            await asyncio.gather(refresh_task, return_exceptions=True)
    finally:
        if client is not None:
            await client.close()
        if ledger is not None:
            await ledger.close()
        await executor_service.stop()
        market_data_service.stop()
        await connector_service.stop_all()
        await db_manager.close()


app = FastAPI(
    title="Binance Stocks Executor Runtime",
    version="1.0.0",
    lifespan=stocks_lifespan,
    redirect_slashes=False,
)
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.include_router(executors_router.router, dependencies=[Depends(auth_user)])
app.include_router(stocks_router)


@app.middleware("http")
async def restrict_runtime_surface(request: Request, call_next):
    path = request.url.path
    allowed = path.startswith("/stocks/") or path in {"/docs", "/openapi.json", "/redoc"}
    allowed = allowed or path.startswith("/executors") or path == "/"
    if not allowed:
        return JSONResponse(status_code=403, content={"detail": "endpoint disabled in Stocks runtime"})
    if path.startswith("/executors/positions"):
        return JSONResponse(status_code=403, content={"detail": "position hold bypass endpoints are disabled"})
    if path.startswith("/executors/") and request.method not in {"GET"}:
        if not (path == "/executors/" and request.method == "POST") and not (
            path == "/executors/search" and request.method == "POST"
        ):
            return JSONResponse(
                status_code=403,
                content={"detail": "use the ownership-checked /stocks/executors cancel/close endpoints"},
            )
    return await call_next(request)

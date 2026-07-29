from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response

from .executor import DecisionRejected, MacroExecutor
from .bot_overview import BotOverviewProvider, BotOverviewUnavailable
from .file_telemetry import JsonFileTelemetryProvider
from .hummingbot_api import HummingbotAPI
from .security import NonceCache, verify_request
from .storage import StateStore
from .telemetry import TelemetryProvider
from .trading_report import JsonTradingReportProvider, TradingReportUnavailable

LOG = logging.getLogger("dca-macro-gateway")


class SlidingRateLimiter:
    def __init__(self, requests_per_minute: int = 30) -> None:
        self.limit = requests_per_minute
        self.hits: dict[str, deque[float]] = defaultdict(deque)

    def allow(self, identity: str, now: float | None = None) -> bool:
        now = time.time() if now is None else now
        queue = self.hits[identity]
        while queue and queue[0] < now - 60:
            queue.popleft()
        if len(queue) >= self.limit:
            return False
        queue.append(now)
        return True


def create_app(
    executor: MacroExecutor,
    telemetry: TelemetryProvider,
    hmac_secret: str | dict[str, str],
    require_mtls_proxy: bool = True,
    reconcile_interval_seconds: float = 5,
    trading_report: JsonTradingReportProvider | None = None,
    bot_overview: BotOverviewProvider | None = None,
) -> FastAPI:
    async def reconcile_loop() -> None:
        while True:
            try:
                await asyncio.to_thread(executor.reconcile)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                LOG.error("macro gate reconcile failed: %s", exc)
            await asyncio.sleep(max(2.0, reconcile_interval_seconds))

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        task = asyncio.create_task(reconcile_loop())
        try:
            yield
        finally:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    app = FastAPI(
        title="DCA Macro Event Execution Gateway V3",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        lifespan=lifespan,
    )
    nonces = NonceCache(state_path=executor.store.directory / "nonces.json")
    limiter = SlidingRateLimiter()

    @app.middleware("http")
    async def authenticate(request: Request, call_next):
        if request.url.path == "/healthz":
            return await call_next(request)
        if request.url.path not in {
            "/v1/telemetry",
            "/v1/event-decisions",
            "/v1/heartbeat",
            "/v1/status",
            "/v1/trading-report",
            "/v1/trading-chart",
            "/v1/bots",
        } and not request.url.path.startswith("/v1/decisions/"):
            return JSONResponse({"detail": "not found"}, status_code=404)
        if require_mtls_proxy and request.headers.get("x-client-cert-verified") != "SUCCESS":
            return JSONResponse({"detail": "mTLS client certificate required"}, status_code=401)
        identity = request.headers.get("x-client-cert-fingerprint", "unknown")
        if not limiter.allow(identity):
            return JSONResponse({"detail": "rate limit exceeded"}, status_code=429)
        body = await request.body()
        ok, reason = verify_request(
            hmac_secret,
            request.method,
            request.url.path,
            request.headers.get("x-hermes-timestamp", ""),
            request.headers.get("x-hermes-nonce", ""),
            body,
            request.headers.get("x-hermes-signature", ""),
            nonces,
            request.headers.get("x-hermes-key-id", ""),
        )
        if not ok:
            return JSONResponse({"detail": reason}, status_code=401)
        request._body = body
        return await call_next(request)

    @app.get("/v1/telemetry")
    async def get_telemetry():
        return telemetry.snapshot()

    @app.post("/v1/event-decisions")
    async def set_event_decision(request: Request):
        try:
            return executor.apply(await request.json())
        except DecisionRejected as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.post("/v1/heartbeat")
    async def heartbeat(request: Request):
        payload = await request.json()
        try:
            return executor.heartbeat(str(payload.get("decision_id", "")))
        except DecisionRejected as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.get("/v1/decisions/{decision_id}")
    async def get_decision(decision_id: str):
        value = executor.decision(decision_id)
        if value is None:
            raise HTTPException(status_code=404, detail="decision not found")
        return value

    @app.post("/v1/decisions/{decision_id}/revoke")
    async def revoke_decision(decision_id: str, request: Request):
        payload = await request.json()
        try:
            return executor.revoke(
                decision_id,
                dict(payload.get("approval", {})),
            )
        except DecisionRejected as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.get("/v1/status")
    async def get_status():
        return executor.status()

    @app.get("/v1/trading-report")
    async def get_trading_report():
        if trading_report is None:
            raise HTTPException(
                status_code=503, detail="trading report is not configured"
            )
        try:
            return trading_report.report()
        except TradingReportUnavailable as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.get("/v1/trading-chart")
    async def get_trading_chart():
        if trading_report is None:
            raise HTTPException(
                status_code=503, detail="trading report is not configured"
            )
        try:
            chart, report_id = trading_report.chart()
        except TradingReportUnavailable as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return Response(
            content=chart,
            media_type="image/png",
            headers={"X-DCA-Report-Id": report_id},
        )

    @app.get("/v1/bots")
    async def get_bots():
        if bot_overview is None:
            raise HTTPException(
                status_code=503, detail="bot overview is not configured"
            )
        try:
            return bot_overview.snapshot()
        except BotOverviewUnavailable as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.get("/healthz")
    async def healthz():
        return {
            "status": "healthy",
            "policy_version": "dca-macro-v3",
            "execution_enabled": executor.execution_enabled,
        }

    return app


def _disabled_reader(name: str):
    def read():
        raise RuntimeError(
            f"{name} reader is not configured; inject a local OCI adapter before deployment"
        )

    return read


def app_from_environment() -> FastAPI:
    secret_file = os.environ.get("HERMES_HMAC_SECRETS_FILE", "")
    if secret_file:
        secret = json.loads(Path(secret_file).read_text(encoding="utf-8"))
        if not isinstance(secret, dict) or not secret or not all(
            isinstance(key, str)
            and key
            and isinstance(value, str)
            and len(value) >= 32
            for key, value in secret.items()
        ):
            raise RuntimeError(
                "HERMES_HMAC_SECRETS_FILE must map key IDs to 32+ character secrets"
            )
    else:
        secret = os.environ.get("HERMES_HMAC_SECRET", "")
        if not secret:
            raise RuntimeError(
                "HERMES_HMAC_SECRETS_FILE or HERMES_HMAC_SECRET is required"
            )
    api = HummingbotAPI(
        os.environ.get("HUMMINGBOT_API_URL", "http://hummingbot-api:8000"),
        os.environ.get("HUMMINGBOT_API_USER", os.environ.get("USERNAME", "")),
        os.environ.get(
            "HUMMINGBOT_API_PASSWORD", os.environ.get("PASSWORD", "")
        ),
    )
    telemetry_input = os.environ.get("DCA_MACRO_TELEMETRY_INPUT")
    telemetry = (
        JsonFileTelemetryProvider(Path(telemetry_input))
        if telemetry_input
        else TelemetryProvider(
            _disabled_reader("bot telemetry"), _disabled_reader("market telemetry")
        )
    )
    report_input = os.environ.get("DCA_MACRO_TRADING_REPORT_INPUT", "")
    chart_input = os.environ.get("DCA_MACRO_TRADING_CHART_INPUT", "")
    trading_report = (
        JsonTradingReportProvider(
            Path(report_input),
            Path(chart_input),
            max_age_seconds=float(
                os.environ.get("DCA_MACRO_TRADING_REPORT_MAX_AGE", "900")
            ),
        )
        if report_input and chart_input
        else None
    )
    grid_guard_input = os.environ.get("DCA_MACRO_GRID_GUARD_INPUT", "")
    bot_overview = (
        BotOverviewProvider(api, Path(grid_guard_input))
        if grid_guard_input
        else None
    )
    targets = json.loads(os.environ.get("DCA_MACRO_BOT_TARGETS", "[]"))
    executor = MacroExecutor(
        api,
        telemetry,
        StateStore(Path(os.environ.get("DCA_MACRO_STATE_DIR", "data/macro-control"))),
        targets,
        allowed_event_kinds={
            value.strip().lower()
            for value in os.environ.get(
                "DCA_MACRO_ALLOWED_EVENTS", "fomc,cpi,nfp"
            ).split(",")
            if value.strip()
        },
        max_pre_event_hours=float(
            os.environ.get("DCA_MACRO_MAX_PRE_EVENT_HOURS", "24")
        ),
        max_post_event_hours=float(
            os.environ.get("DCA_MACRO_MAX_POST_EVENT_HOURS", "6")
        ),
        execution_enabled=os.environ.get(
            "DCA_MACRO_EXECUTION_ENABLED", "false"
        ).lower()
        == "true",
        telegram_approver_user_id=os.environ.get(
            "HERMES_APPROVER_TELEGRAM_USER_ID",
            os.environ.get("ADMIN_USER_ID", ""),
        ),
        telegram_approver_chat_id=os.environ.get(
            "HERMES_APPROVER_TELEGRAM_CHAT_ID",
            os.environ.get("ADMIN_USER_ID", ""),
        ),
    )
    return create_app(
        executor,
        telemetry,
        secret,
        reconcile_interval_seconds=float(
            os.environ.get("DCA_MACRO_RECONCILE_INTERVAL", "5")
        ),
        trading_report=trading_report,
        bot_overview=bot_overview,
    )

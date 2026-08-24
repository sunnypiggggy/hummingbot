from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from datetime import datetime, timedelta, timezone
from decimal import Decimal, ROUND_DOWN
from typing import Any, Dict, Optional

from stocks_runtime.executor_config import normalize_executor_config, order_type_name, trade_side_name
from stocks_runtime.ledger import SCHEDULED_INTENT_STATES


logger = logging.getLogger(__name__)

LIMIT_PHASES = {"PRE_MARKET", "MARKET_OPEN", "POST_MARKET", "AFTER_HOURS"}
MARKET_PHASES = {"MARKET_OPEN"}
TERMINAL_STATES = {"CANCELED", "EXPIRED", "REJECTED", "FAILED"}
TERMINAL_EXECUTOR_STATES = {"TERMINATED", "FAILED", "COMPLETED", "CANCELED", "CANCELLED", "STOPPED"}


def schedule_id_for(executor_id: str) -> str:
    return "sch-" + hashlib.sha256(executor_id.encode("utf-8")).hexdigest()[:24]


def activation_target(config: Dict[str, Any]) -> str:
    if str(config.get("type")) == "position_executor":
        strategy = order_type_name((config.get("triple_barrier_config") or {}).get("open_order_type", "LIMIT"))
    else:
        strategy = order_type_name(config.get("execution_strategy", "LIMIT"))
    return "MARKET_OPEN" if strategy == "MARKET" else "EXTENDED"


def session_eligible(target: str, phase: str) -> bool:
    phase = str(phase or "UNKNOWN").upper()
    return phase in (MARKET_PHASES if target == "MARKET_OPEN" else LIMIT_PHASES)


def _hard_rejection(exc: Exception) -> bool:
    text = str(exc).lower()
    transient = (
        "stale", "unknown", "closed", "outside an eligible session", "market_open",
        "temporar", "timeout", "connection", "429", "5xx", "trading status",
        "tradability", "currently allow", "quote", "suspend",
    )
    if any(token in text for token in transient):
        return False
    hard = (
        "whitelist", "not enabled", "insufficient", "exceed", "daily", "invalid",
        "requires", "forbidden", "long-only", "disclaimer", "eligibility", "authorized",
        "position snapshot", "fractional", "minimum", "notional",
    )
    return any(token in text for token in hard)


def _runtime_executor_is_active(runtime: Optional[Dict[str, Any]]) -> bool:
    if not runtime:
        return False
    if "is_active" in runtime:
        return bool(runtime["is_active"])
    return str(runtime.get("status", "UNKNOWN")).upper() not in TERMINAL_EXECUTOR_STATES


class AsyncStocksOrderScheduler:
    """Durable session-aware activation for Stocks Executors.

    The queue owns no second balance model: the normal managed ledger reserves
    the request in the same transaction that creates the schedule.  Activation
    only converts that reservation into a running Executor.
    """

    def __init__(self, app, poll_seconds: float = 2.0):
        self.app = app
        self.poll_seconds = max(0.2, float(poll_seconds))
        self._tick_lock = asyncio.Lock()

    @property
    def ledger(self):
        return self.app.state.stocks_ledger

    @property
    def policy(self):
        return self.app.state.stocks_policy

    def _market_context(self, symbol: str) -> tuple[str, Optional[str], bool, str]:
        connector = getattr(self.app.state, "stocks_connector", None)
        if connector is None:
            return "UNKNOWN", None, False, "connector_unavailable"
        phase = str(getattr(connector, "market_phase", "UNKNOWN") or "UNKNOWN").upper()
        metadata = getattr(connector, "market_state_metadata", {})
        trading_date = getattr(getattr(self.app.state, "stocks_paper_broker", None), "trading_date", None)
        trading_date = trading_date or self.policy.trading_date
        quote_fresh = connector.latest_quote(symbol) is not None
        if phase == "UNKNOWN":
            return phase, trading_date, quote_fresh, "market_state_unknown"
        if metadata and float(metadata.get("valid_until") or 0) <= time.time():
            return phase, trading_date, quote_fresh, "market_state_reconciliation_pending"
        if metadata.get("conflict"):
            return phase, trading_date, quote_fresh, "binance_xnys_market_state_conflict"
        return phase, trading_date, quote_fresh, ""

    async def enqueue(
        self,
        *,
        executor_config: Dict[str, Any],
        request_payload: Dict[str, Any],
        controller_id: str,
        quote_budget: Optional[Decimal] = None,
    ) -> Dict[str, Any]:
        config = normalize_executor_config(executor_config)
        executor_id = str(config["id"])
        existing = await self.ledger.scheduled_by_executor(executor_id)
        if existing is not None:
            return {"idempotent_replay": True, "disposition": "QUEUED", "schedule": existing}
        target = activation_target(config)
        pair = str(config["trading_pair"])
        symbol = pair.rsplit("-", 1)[0]
        price = Decimal(str(config.get("price") or config.get("entry_price") or self.policy.prices.get(symbol, 0)))
        amount = Decimal(str(config.get("amount", 0)))
        if amount <= 0 or price <= 0:
            raise ValueError("queued order requires a positive reference quantity and price for reservation")
        basis = "QUOTE_BUDGET" if quote_budget is not None else "FIXED_SHARES"
        if quote_budget is not None:
            budget = Decimal(quote_budget)
            if budget <= 0:
                raise ValueError("quote_budget must be positive")
            # Reserve the exact budget.  Activation can only reduce base shares,
            # never increase economic exposure beyond this amount.
            config["amount"] = str(amount)
            reservation_price = budget / amount
            if str(config.get("type")) == "position_executor":
                config["entry_price"] = str(reservation_price)
            else:
                config["price"] = str(reservation_price)
        schedule_id = schedule_id_for(executor_id)
        phase, _, _, reason = self._market_context(symbol)
        schedule = {
            "schedule_id": schedule_id,
            "request_type": str(config["type"]),
            "request_payload": request_payload,
            "target_session": target,
            "amount_basis": basis,
            "quote_budget": str(quote_budget) if quote_budget is not None else None,
            "frozen_price": str(price) if target == "EXTENDED" else None,
            "hard_expires_at": datetime.now(timezone.utc) + timedelta(days=7),
            "last_block_reason": reason or f"waiting_for_{target.lower()}",
        }
        reserved = await self.policy.validate_and_reserve(
            config, "stocks_managed", controller_id, schedule=schedule
        )
        row = await self.ledger.scheduled_record(schedule_id)
        return {
            "idempotent_replay": False,
            "disposition": "QUEUED",
            "schedule_id": schedule_id,
            "target_session": target,
            "reservation": reserved,
            "market_phase": phase,
            "schedule": row,
        }

    async def cancel(self, schedule_id: str) -> Dict[str, Any]:
        async with self._tick_lock:
            row = await self.ledger.scheduled_record(schedule_id)
            if row is None:
                raise KeyError("scheduled executor not found")
            status = str(row["status"])
            if status == "ACTIVE":
                return {"schedule": row, "executor_active": True, "cancel_route": "executor"}
            if status in TERMINAL_STATES:
                return {"schedule": row, "idempotent_replay": True}
            canceled = await self.ledger.terminalize_schedule(
                schedule_id, "CANCELED", "canceled_by_operator"
            )
            return {"schedule": canceled, "idempotent_replay": False}

    async def _activation_config(self, row: Dict[str, Any], symbol: str) -> tuple[Dict[str, Any], Decimal]:
        config = dict(row["executor_config"])
        if str(row["amount_basis"]) != "QUOTE_BUDGET":
            price = Decimal(str(row.get("frozen_price") or config.get("price") or config.get("entry_price")))
            return config, Decimal(str(config["amount"])) * price
        connector = self.app.state.stocks_connector
        quote = connector.latest_quote(symbol)
        if quote is None:
            raise PermissionError(f"Binance Stocks quote for {symbol} is stale")
        _, ask = quote
        budget = Decimal(str(row["quote_budget"]))
        raw_amount = budget / ask
        pair = str(config["trading_pair"])
        amount = connector.quantize_order_amount(pair, raw_amount)
        # A conservative fallback for a connector whose rules are still warming.
        if amount <= 0:
            amount = raw_amount.quantize(Decimal("0.000001"), rounding=ROUND_DOWN)
        notional = amount * ask
        if amount <= 0 or notional <= 0 or notional > budget:
            raise ValueError("fixed quote budget cannot produce a valid order quantity")
        config["amount"] = str(amount)
        if str(config.get("type")) == "position_executor":
            config.pop("entry_price", None)
        else:
            config.pop("price", None)
        return config, notional

    def _connector_preflight(self, config: Dict[str, Any], symbol: str) -> None:
        connector = self.app.state.stocks_connector
        if not bool(getattr(connector, "ready", False)):
            raise PermissionError("Binance Stocks connector is not ready")
        side = trade_side_name(config.get("side"))
        status = str(getattr(connector, "_trading_status", {}).get(symbol, "UNKNOWN")).upper()
        if status not in {"TRADING", "ACTIVE", "NORMAL"}:
            raise PermissionError(f"Binance Stocks {symbol} trading status is {status}")
        direction = str(getattr(connector, "_tradability", {}).get(symbol, "NONE")).upper()
        if direction not in {"BOTH", "BUY_SELL", "ALL", side}:
            raise PermissionError(f"Binance Stocks {symbol} does not currently allow {side}")
        settings = self.app.state.stocks_settings
        if settings.mode == "LIVE":
            if not settings.live_authorized:
                raise PermissionError("Binance Stocks LIVE is not authorized")
            if not settings.disclaimer_confirmed:
                raise PermissionError("Binance Stocks disclaimer is not confirmed")
            status_dict = getattr(connector, "status_dict", {})
            for key in ("account_eligible", "position_reconciliation_ready"):
                if not bool(status_dict.get(key)):
                    raise PermissionError(f"Binance Stocks {key} is not ready")

    async def _process(self, row: Dict[str, Any]) -> None:
        schedule_id = str(row["schedule_id"])
        status = str(row["status"])
        if status not in SCHEDULED_INTENT_STATES:
            return
        if row.get("cancel_requested"):
            await self.ledger.terminalize_schedule(schedule_id, "CANCELED", "canceled_by_operator")
            return
        expires_at = row["hard_expires_at"]
        if expires_at and datetime.now(timezone.utc) >= expires_at:
            await self.ledger.terminalize_schedule(schedule_id, "EXPIRED", "seven_day_market_state_safety_limit")
            return
        config = dict(row["executor_config"])
        symbol = str(config["trading_pair"]).rsplit("-", 1)[0]
        phase, trading_date, quote_fresh, reason = self._market_context(symbol)
        if reason == "binance_xnys_market_state_conflict":
            await self.ledger.transition_schedule(
                schedule_id, "WAITING_PREFLIGHT", expected=set(SCHEDULED_INTENT_STATES),
                reason=reason, target_trading_date=trading_date, next_attempt_seconds=30,
            )
            return
        if not session_eligible(str(row["target_session"]), phase):
            await self.ledger.transition_schedule(
                schedule_id, "WAITING_SESSION", expected=set(SCHEDULED_INTENT_STATES),
                reason=reason or f"waiting_for_{str(row['target_session']).lower()}",
                next_attempt_seconds=5,
            )
            return
        # Once the first eligible day is observed, never carry the request into
        # another trading date.  It expires after that session closes.
        target_date = row.get("target_trading_date")
        if target_date and trading_date and str(target_date) != str(trading_date):
            await self.ledger.terminalize_schedule(schedule_id, "EXPIRED", "target_trading_day_ended")
            return
        if not quote_fresh:
            await self.ledger.transition_schedule(
                schedule_id, "WAITING_PREFLIGHT", expected=set(SCHEDULED_INTENT_STATES),
                reason="quote_stale", target_trading_date=trading_date, next_attempt_seconds=5,
            )
            return
        try:
            self._connector_preflight(config, symbol)
            activation_config, notional = await self._activation_config(row, symbol)
            await self.policy.revalidate_reserved(
                activation_config, "stocks_managed", "stocks-async-scheduler"
            )
        except Exception as exc:
            if _hard_rejection(exc):
                await self.ledger.terminalize_schedule(schedule_id, "REJECTED", str(exc))
            else:
                attempts = int(row.get("attempt_count") or 0)
                await self.ledger.transition_schedule(
                    schedule_id, "WAITING_PREFLIGHT", expected=set(SCHEDULED_INTENT_STATES),
                    reason=str(exc), target_trading_date=trading_date,
                    next_attempt_seconds=min(60, 5 * (2 ** min(attempts, 3))),
                )
            return
        claimed = await self.ledger.transition_schedule(
            schedule_id, "ACTIVATING", expected={"QUEUED", "WAITING_SESSION", "WAITING_PREFLIGHT", "ACTIVATING"},
            reason="preflight_passed", target_trading_date=trading_date,
            executor_config=activation_config,
            requested_shares=Decimal(str(activation_config["amount"])),
            estimated_notional=notional,
            next_attempt_seconds=10,
        )
        if claimed is None:
            return
        executor_id = str(row["executor_id"])
        existing_runtime = await self.app.state.executor_service.get_executor(executor_id)
        record = await self.ledger.executor_record(executor_id)
        # A previously failed, zero-fill Executor may be explicitly re-queued
        # with the same stable ID after its pre-order cause is fixed. A terminal
        # history row is audit evidence, not an active economic order.
        if _runtime_executor_is_active(existing_runtime) or (record and record.get("orders")):
            await self.ledger.transition_schedule(
                schedule_id, "ACTIVE", expected={"ACTIVATING"}, reason="executor_recovered",
                resulting_executor_id=executor_id,
            )
            return
        try:
            created = await self.app.state.stocks_raw_create(
                activation_config, "stocks_managed", "stocks-async-scheduler"
            )
            resulting = str(created.get("executor_id", executor_id)) if isinstance(created, dict) else executor_id
            await self.ledger.transition_schedule(
                schedule_id, "ACTIVE", expected={"ACTIVATING"}, reason="executor_created",
                resulting_executor_id=resulting,
            )
        except Exception as exc:
            # No terminal failure is inferred from a single transport or process
            # error.  The next loop first reconciles the deterministic executor ID.
            await self.ledger.transition_schedule(
                schedule_id, "WAITING_PREFLIGHT", expected={"ACTIVATING"},
                reason=f"activation_retry:{exc}", next_attempt_seconds=10,
            )

    async def tick(self) -> None:
        async with self._tick_lock:
            for row in await self.ledger.due_scheduled_rows():
                try:
                    await self._process(row)
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.exception("scheduled Stocks activation failed schedule_id=%s", row.get("schedule_id"))

    async def run(self, stop: asyncio.Event) -> None:
        while not stop.is_set():
            await self.tick()
            try:
                await asyncio.wait_for(stop.wait(), timeout=self.poll_seconds)
            except asyncio.TimeoutError:
                pass

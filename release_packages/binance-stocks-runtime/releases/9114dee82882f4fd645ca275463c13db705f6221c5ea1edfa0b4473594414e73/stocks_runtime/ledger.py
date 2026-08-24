from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass
from decimal import Decimal
from datetime import datetime
from typing import Any, Dict, Optional

from hummingbot.connector.exchange.binance_stocks.binance_stocks_position_provider import EquityPosition


# POSITION_HOLD is an Executor terminal outcome. Its filled shares live in
# inventory_lots; treating the intent as active double-counts that exposure and
# leaves fee reserves locked after the order has completed.
SCHEDULED_INTENT_STATES = {
    "QUEUED", "WAITING_SESSION", "WAITING_PREFLIGHT", "ACTIVATING",
}
ACTIVE_INTENT_STATES = {
    "RESERVED", "RUNNING", "EXIT_PENDING", "CLOSING", *SCHEDULED_INTENT_STATES,
}
TERMINAL_ORDER_STATES = {"FILLED", "CANCELED", "EXPIRED", "REJECTED", "FAILED"}


def _decode_jsonb_fields(row: Any, *fields: str) -> Optional[Dict[str, Any]]:
    """Normalize asyncpg JSONB values across pool/codec configurations.

    asyncpg returns JSONB as text unless a JSON codec is registered.  Tests and
    some deployments register a codec and therefore return dictionaries.  Keep
    the ledger contract stable for both cases so runtime consumers never need
    to guess which representation the connection pool uses.
    """
    if row is None:
        return None
    result = dict(row)
    for field in fields:
        value = result.get(field)
        if isinstance(value, str):
            result[field] = json.loads(value)
    return result


@dataclass(frozen=True)
class LedgerLimits:
    max_order_notional: Decimal = Decimal("200")
    max_managed_exposure: Decimal = Decimal("2000")
    daily_loss_limit: Decimal = Decimal("200")
    max_symbol_exposure: Decimal = Decimal("200")


class LedgerConflict(RuntimeError):
    pass


class LedgerLimitExceeded(RuntimeError):
    pass


class PostgresManagedLedger:
    """Transaction ledger for inventory created by the dedicated Stocks runtime.

    The ledger never imports pre-existing account equity.  An order is managed
    only when its executor intent was reserved here and its client order ID was
    registered by the connector running in the same executor context.
    """

    SCHEMA = "binance_stocks"
    LEADER_LOCK_ID = 0x484253544F434B  # "HBSTOCK"
    TX_LOCK_ID = LEADER_LOCK_ID + 1

    def __init__(
        self,
        database_url: str,
        limits: LedgerLimits = LedgerLimits(),
        order_prefix: str = "x-HBSTK",
        freshness_seconds: float = 30.0,
        schema: str = "binance_stocks",
        leader_lock_id: Optional[int] = None,
    ):
        if not re.fullmatch(r"[a-z][a-z0-9_]{0,62}", schema):
            raise ValueError("unsafe PostgreSQL schema name")
        self.SCHEMA = schema
        if leader_lock_id is not None:
            self.LEADER_LOCK_ID = int(leader_lock_id)
            self.TX_LOCK_ID = self.LEADER_LOCK_ID + 1
        self.database_url = database_url.replace("postgresql+asyncpg://", "postgresql://", 1)
        self.hard_limits = limits
        self.limits = limits
        self.order_prefix = order_prefix
        self.freshness_seconds = freshness_seconds
        self._pool = None
        self._leader_connection = None
        self._quote_total = Decimal("0")
        self._quote_available = Decimal("0")
        self._quote_updated_at = 0.0

    async def initialize(self) -> None:
        import asyncpg

        self._pool = await asyncpg.create_pool(self.database_url, min_size=1, max_size=4)
        self._leader_connection = await self._pool.acquire()
        leader = await self._leader_connection.fetchval("SELECT pg_try_advisory_lock($1)", self.LEADER_LOCK_ID)
        if not leader:
            await self._pool.release(self._leader_connection)
            self._leader_connection = None
            raise LedgerConflict("another binance-stocks-runtime instance owns the ledger lease")
        async with self._pool.acquire() as connection:
            await connection.execute(f"CREATE SCHEMA IF NOT EXISTS {self.SCHEMA}")
            await connection.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.SCHEMA}.executor_intents (
                    executor_id TEXT PRIMARY KEY,
                    executor_type TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    requested_base NUMERIC NOT NULL,
                    estimated_notional NUMERIC NOT NULL,
                    fee_reserve NUMERIC NOT NULL DEFAULT 0,
                    source_owner TEXT,
                    status TEXT NOT NULL,
                    config JSONB NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
                )
                """
            )
            await connection.execute(
                f"ALTER TABLE {self.SCHEMA}.executor_intents "
                "ADD COLUMN IF NOT EXISTS fee_reserve NUMERIC NOT NULL DEFAULT 0"
            )
            await connection.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.SCHEMA}.managed_orders (
                    client_order_id TEXT PRIMARY KEY,
                    executor_id TEXT NOT NULL REFERENCES {self.SCHEMA}.executor_intents(executor_id),
                    exchange_order_id TEXT,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    order_type TEXT NOT NULL,
                    requested_base NUMERIC NOT NULL,
                    cumulative_base NUMERIC NOT NULL DEFAULT 0,
                    cumulative_quote NUMERIC NOT NULL DEFAULT 0,
                    cumulative_fee NUMERIC NOT NULL DEFAULT 0,
                    status TEXT NOT NULL DEFAULT 'PENDING',
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
                )
                """
            )
            await connection.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.SCHEMA}.inventory_lots (
                    owner_id TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    total_base NUMERIC NOT NULL DEFAULT 0,
                    available_base NUMERIC NOT NULL DEFAULT 0,
                    cost_quote NUMERIC NOT NULL DEFAULT 0,
                    realized_pnl_quote NUMERIC NOT NULL DEFAULT 0,
                    fees_quote NUMERIC NOT NULL DEFAULT 0,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    PRIMARY KEY(owner_id, symbol)
                )
                """
            )
            await connection.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.SCHEMA}.runtime_state (
                    singleton BOOLEAN PRIMARY KEY DEFAULT TRUE CHECK (singleton),
                    trading_date TEXT,
                    external_activity_latched BOOLEAN NOT NULL DEFAULT FALSE,
                    live_authorized BOOLEAN NOT NULL DEFAULT FALSE,
                    session_start_pnl NUMERIC NOT NULL DEFAULT 0,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
                )
                """
            )
            await connection.execute(
                f"INSERT INTO {self.SCHEMA}.runtime_state(singleton) VALUES(TRUE) ON CONFLICT DO NOTHING"
            )
            await connection.execute(
                f"ALTER TABLE {self.SCHEMA}.runtime_state "
                "ADD COLUMN IF NOT EXISTS session_start_pnl NUMERIC NOT NULL DEFAULT 0"
            )
            await connection.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.SCHEMA}.audit_events (
                    event_id TEXT PRIMARY KEY,
                    event_type TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    payload JSONB NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
                )
                """
            )
            await connection.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.SCHEMA}.operator_limits (
                    singleton BOOLEAN PRIMARY KEY DEFAULT TRUE CHECK (singleton),
                    max_order_notional NUMERIC NOT NULL,
                    max_symbol_exposure NUMERIC NOT NULL,
                    max_managed_exposure NUMERIC NOT NULL,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
                )
                """
            )
            await connection.execute(
                f"""
                INSERT INTO {self.SCHEMA}.operator_limits
                  (singleton,max_order_notional,max_symbol_exposure,max_managed_exposure)
                VALUES(TRUE,$1,$2,$3) ON CONFLICT DO NOTHING
                """,
                self.hard_limits.max_order_notional,
                self.hard_limits.max_symbol_exposure,
                self.hard_limits.max_managed_exposure,
            )
            await connection.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.SCHEMA}.symbol_whitelist (
                    symbol TEXT PRIMARY KEY,
                    enabled BOOLEAN NOT NULL DEFAULT TRUE,
                    max_position_notional NUMERIC NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
                )
                """
            )
            await connection.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.SCHEMA}.scheduled_executors (
                    schedule_id TEXT PRIMARY KEY,
                    executor_id TEXT NOT NULL UNIQUE
                      REFERENCES {self.SCHEMA}.executor_intents(executor_id),
                    request_type TEXT NOT NULL,
                    request_payload JSONB NOT NULL,
                    executor_config JSONB NOT NULL,
                    target_session TEXT NOT NULL,
                    amount_basis TEXT NOT NULL,
                    quote_budget NUMERIC,
                    frozen_price NUMERIC,
                    requested_shares NUMERIC NOT NULL,
                    status TEXT NOT NULL,
                    target_trading_date TEXT,
                    hard_expires_at TIMESTAMPTZ NOT NULL,
                    next_attempt_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    attempt_count INTEGER NOT NULL DEFAULT 0,
                    last_block_reason TEXT,
                    resulting_executor_id TEXT,
                    cancel_requested BOOLEAN NOT NULL DEFAULT FALSE,
                    version BIGINT NOT NULL DEFAULT 1,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    activated_at TIMESTAMPTZ,
                    terminal_at TIMESTAMPTZ
                )
                """
            )
            await connection.execute(
                f"CREATE INDEX IF NOT EXISTS scheduled_executors_due_idx "
                f"ON {self.SCHEMA}.scheduled_executors(status,next_attempt_at)"
            )

    async def close(self) -> None:
        if self._leader_connection is not None:
            await self._leader_connection.execute("SELECT pg_advisory_unlock($1)", self.LEADER_LOCK_ID)
            await self._pool.release(self._leader_connection)
            self._leader_connection = None
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    def set_quote_balances(self, total: Decimal, available: Decimal, timestamp: Optional[float] = None) -> None:
        self._quote_total = Decimal(total)
        self._quote_available = Decimal(available)
        self._quote_updated_at = timestamp if timestamp is not None else time.time()

    async def quote_balances(self) -> tuple[Decimal, Decimal]:
        return self._quote_total, self._quote_available

    async def is_fresh(self) -> bool:
        if self._pool is None or time.time() - self._quote_updated_at > self.freshness_seconds:
            return False
        try:
            async with self._pool.acquire() as connection:
                return bool(await connection.fetchval("SELECT TRUE"))
        except Exception:
            return False

    async def managed_positions(self) -> Dict[str, EquityPosition]:
        async with self._pool.acquire() as connection:
            rows = await connection.fetch(
                f"""
                SELECT symbol, SUM(total_base) AS total, SUM(available_base) AS available
                FROM {self.SCHEMA}.inventory_lots
                GROUP BY symbol
                HAVING SUM(total_base) > 0
                """
            )
        return {
            str(row["symbol"]): EquityPosition(
                total=Decimal(row["total"]),
                available=max(Decimal("0"), Decimal(row["available"])),
            )
            for row in rows
        }

    async def ensure_whitelist(self, symbols: set[str]) -> None:
        """Seed an empty whitelist without re-enabling operator-disabled symbols."""
        if not symbols:
            return
        async with self._pool.acquire() as connection:
            async with connection.transaction():
                for symbol in sorted(symbols):
                    await connection.execute(
                        f"""
                        INSERT INTO {self.SCHEMA}.symbol_whitelist(symbol,enabled,max_position_notional)
                        VALUES($1,TRUE,$2) ON CONFLICT DO NOTHING
                        """,
                        symbol.upper(),
                        self.hard_limits.max_symbol_exposure,
                    )

    async def whitelist_rows(self) -> list[Dict[str, Any]]:
        async with self._pool.acquire() as connection:
            rows = await connection.fetch(
                f"SELECT symbol,enabled,max_position_notional,created_at,updated_at "
                f"FROM {self.SCHEMA}.symbol_whitelist ORDER BY symbol"
            )
        return [dict(row) for row in rows]

    async def whitelist_entry(self, symbol: str) -> Optional[Dict[str, Any]]:
        async with self._pool.acquire() as connection:
            row = await connection.fetchrow(
                f"SELECT symbol,enabled,max_position_notional,created_at,updated_at "
                f"FROM {self.SCHEMA}.symbol_whitelist WHERE symbol=$1",
                symbol.upper(),
            )
        return dict(row) if row else None

    async def upsert_whitelist(self, symbol: str, enabled: bool, max_position_notional: Decimal) -> Dict[str, Any]:
        symbol = symbol.upper()
        limit = Decimal(max_position_notional)
        if limit <= 0 or limit > self.hard_limits.max_symbol_exposure:
            raise LedgerLimitExceeded(
                f"symbol limit must be within (0, {self.hard_limits.max_symbol_exposure}] USDC"
            )
        async with self._pool.acquire() as connection:
            row = await connection.fetchrow(
                f"""
                INSERT INTO {self.SCHEMA}.symbol_whitelist(symbol,enabled,max_position_notional)
                VALUES($1,$2,$3)
                ON CONFLICT(symbol) DO UPDATE SET enabled=$2,max_position_notional=$3,updated_at=now()
                RETURNING symbol,enabled,max_position_notional,created_at,updated_at
                """,
                symbol,
                bool(enabled),
                limit,
            )
        return dict(row)

    async def delete_whitelist(self, symbol: str) -> bool:
        async with self._pool.acquire() as connection:
            result = await connection.execute(
                f"DELETE FROM {self.SCHEMA}.symbol_whitelist WHERE symbol=$1",
                symbol.upper(),
            )
        return result.endswith("1")

    async def active_limits(self) -> LedgerLimits:
        async with self._pool.acquire() as connection:
            row = await connection.fetchrow(
                f"SELECT max_order_notional,max_symbol_exposure,max_managed_exposure "
                f"FROM {self.SCHEMA}.operator_limits WHERE singleton=TRUE"
            )
        if not row:
            return self.hard_limits
        limits = LedgerLimits(
            max_order_notional=Decimal(row["max_order_notional"]),
            max_managed_exposure=Decimal(row["max_managed_exposure"]),
            daily_loss_limit=self.hard_limits.daily_loss_limit,
            max_symbol_exposure=Decimal(row["max_symbol_exposure"]),
        )
        self.limits = limits
        return limits

    async def update_limits(
        self, max_order_notional: Decimal, max_symbol_exposure: Decimal, max_managed_exposure: Decimal
    ) -> LedgerLimits:
        order = Decimal(max_order_notional)
        symbol = Decimal(max_symbol_exposure)
        total = Decimal(max_managed_exposure)
        if not (Decimal("0") < order <= symbol <= total):
            raise LedgerLimitExceeded("limits must satisfy 0 < order <= symbol <= total")
        if order > self.hard_limits.max_order_notional:
            raise LedgerLimitExceeded(f"order limit exceeds hard ceiling {self.hard_limits.max_order_notional}")
        if symbol > self.hard_limits.max_symbol_exposure:
            raise LedgerLimitExceeded(f"symbol limit exceeds hard ceiling {self.hard_limits.max_symbol_exposure}")
        if total > self.hard_limits.max_managed_exposure:
            raise LedgerLimitExceeded(f"total limit exceeds hard ceiling {self.hard_limits.max_managed_exposure}")
        async with self._pool.acquire() as connection:
            await connection.execute(
                f"""
                UPDATE {self.SCHEMA}.operator_limits SET max_order_notional=$1,
                  max_symbol_exposure=$2,max_managed_exposure=$3,updated_at=now()
                WHERE singleton=TRUE
                """,
                order,
                symbol,
                total,
            )
        self.limits = LedgerLimits(order, total, self.hard_limits.daily_loss_limit, symbol)
        return self.limits

    async def managed_symbol_exposure(
        self, symbol: str, price: Decimal, exclude_executor_id: Optional[str] = None
    ) -> Decimal:
        async with self._pool.acquire() as connection:
            position = await connection.fetchval(
                f"SELECT COALESCE(SUM(total_base),0) FROM {self.SCHEMA}.inventory_lots WHERE symbol=$1",
                symbol,
            )
            pending = await connection.fetchval(
                f"""
                SELECT COALESCE(SUM(estimated_notional + fee_reserve),0)
                FROM {self.SCHEMA}.executor_intents
                WHERE symbol=$1 AND side='BUY' AND status = ANY($2::text[])
                  AND ($3::text IS NULL OR executor_id<>$3)
                """,
                symbol,
                list(ACTIVE_INTENT_STATES),
                exclude_executor_id,
            )
        return Decimal(position or 0) * Decimal(price) + Decimal(pending or 0)

    async def managed_available(self, owner_id: str, symbol: str) -> Decimal:
        async with self._pool.acquire() as connection:
            value = await connection.fetchval(
                f"SELECT available_base FROM {self.SCHEMA}.inventory_lots WHERE owner_id=$1 AND symbol=$2",
                owner_id,
                symbol,
            )
        return Decimal(value or 0)

    async def executor_record(self, executor_id: str) -> Optional[Dict[str, Any]]:
        async with self._pool.acquire() as connection:
            intent = await connection.fetchrow(
                f"SELECT * FROM {self.SCHEMA}.executor_intents WHERE executor_id=$1",
                executor_id,
            )
            if not intent:
                return None
            orders = await connection.fetch(
                f"SELECT * FROM {self.SCHEMA}.managed_orders WHERE executor_id=$1 ORDER BY created_at",
                executor_id,
            )
        result = dict(intent)
        result["orders"] = [dict(row) for row in orders]
        return result

    async def executor_rows(self, active_only: bool = False) -> list[Dict[str, Any]]:
        where = "WHERE status = ANY($1::text[])" if active_only else ""
        async with self._pool.acquire() as connection:
            if active_only:
                rows = await connection.fetch(
                    f"SELECT * FROM {self.SCHEMA}.executor_intents {where} ORDER BY updated_at DESC",
                    list(ACTIVE_INTENT_STATES),
                )
            else:
                rows = await connection.fetch(
                    f"SELECT * FROM {self.SCHEMA}.executor_intents ORDER BY updated_at DESC LIMIT 200"
                )
        return [dict(row) for row in rows]

    async def reserve_intent(
        self,
        *,
        executor_id: str,
        executor_type: str,
        symbol: str,
        side: str,
        requested_base: Decimal,
        estimated_notional: Decimal,
        fee_reserve: Decimal = Decimal("0"),
        config: Dict[str, Any],
        source_owner: Optional[str] = None,
        schedule: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        side = side.upper()
        estimated_notional = Decimal(estimated_notional)
        fee_reserve = Decimal(fee_reserve)
        limits = await self.active_limits()
        if estimated_notional > limits.max_order_notional:
            raise LedgerLimitExceeded(
                f"order notional {estimated_notional} exceeds {limits.max_order_notional} USDC"
            )
        async with self._pool.acquire() as connection:
            async with connection.transaction():
                await connection.execute("SELECT pg_advisory_xact_lock($1)", self.TX_LOCK_ID)
                existing = await connection.fetchrow(
                    f"SELECT * FROM {self.SCHEMA}.executor_intents WHERE executor_id=$1", executor_id
                )
                canonical = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str)
                if existing:
                    existing_config = existing["config"]
                    if isinstance(existing_config, str):
                        existing_config = json.loads(existing_config)
                    if existing_config != json.loads(canonical):
                        raise LedgerConflict(f"executor id {executor_id} already exists with different config")
                    return dict(existing)
                state = await connection.fetchrow(
                    f"SELECT external_activity_latched FROM {self.SCHEMA}.runtime_state WHERE singleton=TRUE"
                )
                if side == "BUY" and bool(state["external_activity_latched"]):
                    raise LedgerLimitExceeded("external equity activity is latched; new BUY is blocked")
                exposure = Decimal(
                    await connection.fetchval(
                        f"""
                        SELECT COALESCE(SUM(estimated_notional + fee_reserve), 0)
                        FROM {self.SCHEMA}.executor_intents
                        WHERE side='BUY' AND status = ANY($1::text[])
                        """,
                        list(ACTIVE_INTENT_STATES),
                    )
                )
                if side == "BUY" and exposure + estimated_notional + fee_reserve > limits.max_managed_exposure:
                    raise LedgerLimitExceeded(
                        f"managed exposure would exceed {limits.max_managed_exposure} USDC"
                    )
                if side == "SELL":
                    owner = source_owner or "unassigned"
                    available = await connection.fetchval(
                        f"SELECT available_base FROM {self.SCHEMA}.inventory_lots WHERE owner_id=$1 AND symbol=$2",
                        owner,
                        symbol,
                    )
                    if available is None or Decimal(available) < Decimal(requested_base):
                        raise LedgerLimitExceeded(f"managed {symbol} inventory is insufficient for SELL")
                    await connection.execute(
                        f"""
                        UPDATE {self.SCHEMA}.inventory_lots
                        SET available_base=available_base-$3, updated_at=now()
                        WHERE owner_id=$1 AND symbol=$2
                        """,
                        owner,
                        symbol,
                        Decimal(requested_base),
                    )
                    source_owner = owner
                await connection.execute(
                    f"""
                    INSERT INTO {self.SCHEMA}.executor_intents
                    (executor_id, executor_type, symbol, side, requested_base, estimated_notional,
                     fee_reserve, source_owner, status, config)
                    VALUES($1,$2,$3,$4,$5,$6,$7,$8,'RESERVED',$9::jsonb)
                    """,
                    executor_id,
                    executor_type,
                    symbol,
                    side,
                    Decimal(requested_base),
                    estimated_notional,
                    fee_reserve,
                    source_owner,
                    canonical,
                )
                if schedule is not None:
                    schedule_id = str(schedule["schedule_id"])
                    request_payload = json.dumps(
                        schedule.get("request_payload", {}), sort_keys=True,
                        separators=(",", ":"), default=str,
                    )
                    await connection.execute(
                        f"""
                        INSERT INTO {self.SCHEMA}.scheduled_executors
                          (schedule_id,executor_id,request_type,request_payload,executor_config,
                           target_session,amount_basis,quote_budget,frozen_price,requested_shares,
                           status,hard_expires_at,last_block_reason)
                        VALUES($1,$2,$3,$4::jsonb,$5::jsonb,$6,$7,$8,$9,$10,'QUEUED',$11,$12)
                        """,
                        schedule_id,
                        executor_id,
                        str(schedule["request_type"]),
                        request_payload,
                        canonical,
                        str(schedule["target_session"]),
                        str(schedule["amount_basis"]),
                        Decimal(str(schedule["quote_budget"])) if schedule.get("quote_budget") is not None else None,
                        Decimal(str(schedule["frozen_price"])) if schedule.get("frozen_price") is not None else None,
                        Decimal(requested_base),
                        schedule["hard_expires_at"],
                        str(schedule.get("last_block_reason") or "waiting_for_eligible_session"),
                    )
                    await connection.execute(
                        f"UPDATE {self.SCHEMA}.executor_intents SET status='QUEUED',updated_at=now() "
                        "WHERE executor_id=$1",
                        executor_id,
                    )
        return {
            "executor_id": executor_id,
            "status": "QUEUED" if schedule is not None else "RESERVED",
            **({"schedule_id": str(schedule["schedule_id"])} if schedule is not None else {}),
        }

    async def set_live_authorized(self, authorized: bool) -> None:
        async with self._pool.acquire() as connection:
            await connection.execute(
                f"UPDATE {self.SCHEMA}.runtime_state SET live_authorized=$1, updated_at=now() "
                "WHERE singleton=TRUE",
                bool(authorized),
            )

    async def set_trading_date(self, trading_date: str, current_pnl: Decimal) -> None:
        """Reset only the daily loss baseline when Binance tradingDate changes."""
        async with self._pool.acquire() as connection:
            async with connection.transaction():
                await connection.execute("SELECT pg_advisory_xact_lock($1)", self.TX_LOCK_ID)
                previous = await connection.fetchval(
                    f"SELECT trading_date FROM {self.SCHEMA}.runtime_state WHERE singleton=TRUE FOR UPDATE"
                )
                if previous != trading_date:
                    await connection.execute(
                        f"UPDATE {self.SCHEMA}.runtime_state SET trading_date=$1, "
                        "session_start_pnl=$2, updated_at=now() WHERE singleton=TRUE",
                        trading_date,
                        Decimal(current_pnl),
                    )

    async def daily_pnl(self, current_pnl: Decimal) -> Decimal:
        async with self._pool.acquire() as connection:
            baseline = await connection.fetchval(
                f"SELECT session_start_pnl FROM {self.SCHEMA}.runtime_state WHERE singleton=TRUE"
            )
        return Decimal(current_pnl) - Decimal(baseline or 0)

    async def realized_pnl(self) -> Decimal:
        async with self._pool.acquire() as connection:
            value = await connection.fetchval(
                f"SELECT COALESCE(SUM(realized_pnl_quote), 0) FROM {self.SCHEMA}.inventory_lots"
            )
        return Decimal(value or 0)

    async def managed_pnl(self, prices: Dict[str, Decimal]) -> Decimal:
        """Realized + unrealized managed PnL after all USDC fees."""
        async with self._pool.acquire() as connection:
            rows = await connection.fetch(
                f"SELECT symbol,total_base,cost_quote,realized_pnl_quote,fees_quote "
                f"FROM {self.SCHEMA}.inventory_lots"
            )
        pnl = Decimal("0")
        for row in rows:
            mark = Decimal(prices.get(str(row["symbol"]), 0))
            pnl += Decimal(row["realized_pnl_quote"])
            pnl += Decimal(row["total_base"]) * mark - Decimal(row["cost_quote"])
            pnl -= Decimal(row["fees_quote"])
        return pnl

    async def managed_exposure(
        self, prices: Dict[str, Decimal], exclude_executor_id: Optional[str] = None
    ) -> Decimal:
        """Managed MTM plus unfilled BUY reservations; external positions are excluded."""
        async with self._pool.acquire() as connection:
            lots = await connection.fetch(
                f"SELECT symbol, SUM(total_base) AS total FROM {self.SCHEMA}.inventory_lots "
                "GROUP BY symbol HAVING SUM(total_base) > 0"
            )
            pending = await connection.fetchval(
                f"""
                SELECT COALESCE(SUM(
                  GREATEST(0, i.estimated_notional - COALESCE(o.filled_quote, 0)) + i.fee_reserve
                ), 0)
                FROM {self.SCHEMA}.executor_intents i
                LEFT JOIN (
                  SELECT executor_id, SUM(cumulative_quote) AS filled_quote
                  FROM {self.SCHEMA}.managed_orders WHERE side='BUY' GROUP BY executor_id
                ) o ON o.executor_id=i.executor_id
                WHERE i.side='BUY' AND i.status = ANY($1::text[])
                  AND ($2::text IS NULL OR i.executor_id<>$2)
                """,
                list(ACTIVE_INTENT_STATES),
                exclude_executor_id,
            )
        mtm = sum(
            (Decimal(row["total"]) * Decimal(prices.get(str(row["symbol"]), 0)) for row in lots),
            Decimal("0"),
        )
        return mtm + Decimal(pending or 0)

    async def release_intent(self, executor_id: str, status: str) -> None:
        """Release unfilled SELL reservation exactly once when an executor terminates."""
        async with self._pool.acquire() as connection:
            async with connection.transaction():
                await connection.execute("SELECT pg_advisory_xact_lock($1)", self.TX_LOCK_ID)
                intent = await connection.fetchrow(
                    f"SELECT * FROM {self.SCHEMA}.executor_intents WHERE executor_id=$1 FOR UPDATE",
                    executor_id,
                )
                if intent is None or intent["status"] not in ACTIVE_INTENT_STATES:
                    return
                if intent["side"] == "SELL":
                    filled = Decimal(
                        await connection.fetchval(
                            f"SELECT COALESCE(SUM(cumulative_base), 0) FROM {self.SCHEMA}.managed_orders "
                            "WHERE executor_id=$1 AND side='SELL'",
                            executor_id,
                        ) or 0
                    )
                    unfilled = max(Decimal("0"), Decimal(intent["requested_base"]) - filled)
                    if unfilled > 0:
                        await connection.execute(
                            f"UPDATE {self.SCHEMA}.inventory_lots SET available_base=available_base+$3, "
                            "updated_at=now() WHERE owner_id=$1 AND symbol=$2",
                            intent["source_owner"] or "unassigned",
                            intent["symbol"],
                            unfilled,
                        )
                await connection.execute(
                    f"UPDATE {self.SCHEMA}.executor_intents SET status=$2, updated_at=now() WHERE executor_id=$1",
                    executor_id,
                    status,
                )

    async def scheduled_record(self, schedule_id: str) -> Optional[Dict[str, Any]]:
        async with self._pool.acquire() as connection:
            row = await connection.fetchrow(
                f"SELECT * FROM {self.SCHEMA}.scheduled_executors WHERE schedule_id=$1",
                schedule_id,
            )
        return _decode_jsonb_fields(row, "request_payload", "executor_config")

    async def scheduled_by_executor(self, executor_id: str) -> Optional[Dict[str, Any]]:
        async with self._pool.acquire() as connection:
            row = await connection.fetchrow(
                f"SELECT * FROM {self.SCHEMA}.scheduled_executors WHERE executor_id=$1",
                executor_id,
            )
        return _decode_jsonb_fields(row, "request_payload", "executor_config")

    async def scheduled_rows(
        self, *, active_only: bool = True, updated_after: Optional[datetime] = None, limit: int = 200
    ) -> list[Dict[str, Any]]:
        statuses = list(SCHEDULED_INTENT_STATES)
        async with self._pool.acquire() as connection:
            rows = await connection.fetch(
                f"""
                SELECT * FROM {self.SCHEMA}.scheduled_executors
                WHERE ($1::boolean=FALSE OR status=ANY($2::text[]))
                  AND ($3::timestamptz IS NULL OR updated_at>$3)
                ORDER BY updated_at DESC LIMIT $4
                """,
                active_only,
                statuses,
                updated_after,
                int(limit),
            )
        return [
            _decode_jsonb_fields(row, "request_payload", "executor_config")
            for row in rows
        ]

    async def due_scheduled_rows(self, limit: int = 50) -> list[Dict[str, Any]]:
        async with self._pool.acquire() as connection:
            rows = await connection.fetch(
                f"""
                SELECT * FROM {self.SCHEMA}.scheduled_executors
                WHERE status=ANY($1::text[]) AND next_attempt_at<=now()
                ORDER BY created_at LIMIT $2 FOR UPDATE SKIP LOCKED
                """,
                list(SCHEDULED_INTENT_STATES),
                int(limit),
            )
        return [
            _decode_jsonb_fields(row, "request_payload", "executor_config")
            for row in rows
        ]

    async def transition_schedule(
        self,
        schedule_id: str,
        status: str,
        *,
        expected: Optional[set[str]] = None,
        reason: Optional[str] = None,
        target_trading_date: Optional[str] = None,
        next_attempt_seconds: float = 0,
        executor_config: Optional[Dict[str, Any]] = None,
        requested_shares: Optional[Decimal] = None,
        estimated_notional: Optional[Decimal] = None,
        resulting_executor_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        intent_status = "RUNNING" if status == "ACTIVE" else status
        async with self._pool.acquire() as connection:
            async with connection.transaction():
                await connection.execute("SELECT pg_advisory_xact_lock($1)", self.TX_LOCK_ID)
                row = await connection.fetchrow(
                    f"SELECT * FROM {self.SCHEMA}.scheduled_executors WHERE schedule_id=$1 FOR UPDATE",
                    schedule_id,
                )
                if row is None or (expected is not None and str(row["status"]) not in expected):
                    return None
                # Closed-market and transient-preflight retries are scheduling
                # heartbeats, not lifecycle transitions. Keep polling due time
                # (and the preflight backoff counter) current without creating
                # a new externally observable version every few seconds.
                same_wait_state = (
                    status in {"WAITING_SESSION", "WAITING_PREFLIGHT"}
                    and str(row["status"]) == status
                    and (reason is None or str(row["last_block_reason"] or "") == str(reason))
                    and (
                        target_trading_date is None
                        or str(row["target_trading_date"] or "") == str(target_trading_date)
                    )
                    and executor_config is None
                    and requested_shares is None
                    and estimated_notional is None
                    and resulting_executor_id is None
                )
                if same_wait_state:
                    await connection.execute(
                        f"""
                        UPDATE {self.SCHEMA}.scheduled_executors SET
                          next_attempt_at=now()+($2::double precision*interval '1 second'),
                          attempt_count=attempt_count+CASE WHEN $3='WAITING_PREFLIGHT' THEN 1 ELSE 0 END
                        WHERE schedule_id=$1
                        """,
                        schedule_id, float(next_attempt_seconds), status,
                    )
                    return _decode_jsonb_fields(
                        await connection.fetchrow(
                            f"SELECT * FROM {self.SCHEMA}.scheduled_executors WHERE schedule_id=$1", schedule_id
                        ),
                        "request_payload",
                        "executor_config",
                    )
                terminal = status in {"CANCELED", "EXPIRED", "REJECTED", "FAILED"}
                activated = status == "ACTIVE"
                config_json = None
                if executor_config is not None:
                    config_json = json.dumps(executor_config, sort_keys=True, separators=(",", ":"), default=str)
                await connection.execute(
                    f"""
                    UPDATE {self.SCHEMA}.scheduled_executors SET
                      status=$2,last_block_reason=COALESCE($3,last_block_reason),
                      target_trading_date=COALESCE($4,target_trading_date),
                      next_attempt_at=now()+($5::double precision*interval '1 second'),
                      attempt_count=attempt_count+CASE WHEN $2 IN ('WAITING_PREFLIGHT','FAILED') THEN 1 ELSE 0 END,
                      executor_config=COALESCE($6::jsonb,executor_config),
                      requested_shares=COALESCE($7,requested_shares),
                      resulting_executor_id=COALESCE($8,resulting_executor_id),
                      activated_at=CASE WHEN $9 THEN now() ELSE activated_at END,
                      terminal_at=CASE WHEN $10 THEN now() ELSE terminal_at END,
                      version=version+1,updated_at=now()
                    WHERE schedule_id=$1
                    """,
                    schedule_id, status, reason, target_trading_date, float(next_attempt_seconds),
                    config_json, requested_shares, resulting_executor_id, activated, terminal,
                )
                await connection.execute(
                    f"""
                    UPDATE {self.SCHEMA}.executor_intents SET
                      status=$2,
                      config=COALESCE($3::jsonb,config),
                      requested_base=COALESCE($4,requested_base),
                      estimated_notional=COALESCE($5,estimated_notional),updated_at=now()
                    WHERE executor_id=$1
                    """,
                    row["executor_id"], intent_status, config_json, requested_shares, estimated_notional,
                )
                return _decode_jsonb_fields(
                    await connection.fetchrow(
                        f"SELECT * FROM {self.SCHEMA}.scheduled_executors WHERE schedule_id=$1", schedule_id
                    ),
                    "request_payload",
                    "executor_config",
                )

    async def terminalize_schedule(self, schedule_id: str, status: str, reason: str) -> Optional[Dict[str, Any]]:
        row = await self.scheduled_record(schedule_id)
        if row is None:
            return None
        if str(row["status"]) in {"CANCELED", "EXPIRED", "REJECTED", "FAILED", "ACTIVE"}:
            return row
        await self.release_intent(str(row["executor_id"]), status)
        return await self.transition_schedule(
            schedule_id, status, expected=set(SCHEDULED_INTENT_STATES), reason=reason
        )

    async def is_managed_order(self, client_order_id: str) -> bool:
        async with self._pool.acquire() as connection:
            return bool(await connection.fetchval(
                f"SELECT EXISTS(SELECT 1 FROM {self.SCHEMA}.managed_orders WHERE client_order_id=$1)",
                client_order_id,
            ))

    async def managed_order_ids(self) -> set[str]:
        async with self._pool.acquire() as connection:
            rows = await connection.fetch(f"SELECT client_order_id FROM {self.SCHEMA}.managed_orders")
        return {str(row["client_order_id"]) for row in rows}

    async def managed_position_rows(self) -> list[Dict[str, Any]]:
        async with self._pool.acquire() as connection:
            rows = await connection.fetch(
                f"SELECT * FROM {self.SCHEMA}.inventory_lots WHERE total_base > 0 ORDER BY symbol, owner_id"
            )
        return [dict(row) for row in rows]

    async def register_order(
        self,
        *,
        client_order_id: str,
        executor_id: Optional[str],
        symbol: str,
        side: str,
        requested_base: Decimal,
        order_type: str,
    ) -> None:
        if not client_order_id.startswith(self.order_prefix):
            raise LedgerConflict("connector generated an order outside the dedicated Stocks namespace")
        if not executor_id:
            raise LedgerConflict("Stocks order has no ExecutorService ownership context")
        async with self._pool.acquire() as connection:
            await connection.execute(
                f"""
                INSERT INTO {self.SCHEMA}.managed_orders
                (client_order_id, executor_id, symbol, side, order_type, requested_base)
                VALUES($1,$2,$3,$4,$5,$6)
                ON CONFLICT(client_order_id) DO UPDATE SET updated_at=now()
                """,
                client_order_id,
                executor_id,
                symbol,
                side.upper(),
                order_type,
                Decimal(requested_base),
            )
            await connection.execute(
                f"UPDATE {self.SCHEMA}.executor_intents SET status='RUNNING', updated_at=now() WHERE executor_id=$1",
                executor_id,
            )

    async def bind_exchange_order(self, client_order_id: str, exchange_order_id: str) -> None:
        async with self._pool.acquire() as connection:
            await connection.execute(
                f"""
                UPDATE {self.SCHEMA}.managed_orders
                SET exchange_order_id=$2, updated_at=now()
                WHERE client_order_id=$1
                """,
                client_order_id,
                exchange_order_id,
            )

    async def record_cumulative_fill(
        self,
        *,
        client_order_id: str,
        exchange_order_id: str,
        cumulative_base: Decimal,
        cumulative_quote: Decimal,
        cumulative_fee: Decimal,
        status: str,
    ) -> None:
        async with self._pool.acquire() as connection:
            async with connection.transaction():
                await connection.execute("SELECT pg_advisory_xact_lock($1)", self.TX_LOCK_ID)
                order = await connection.fetchrow(
                    f"SELECT * FROM {self.SCHEMA}.managed_orders WHERE client_order_id=$1 FOR UPDATE",
                    client_order_id,
                )
                if order is None:
                    raise LedgerConflict(f"fill for unmanaged Stocks order {client_order_id}")
                delta_base = Decimal(cumulative_base) - Decimal(order["cumulative_base"])
                delta_quote = Decimal(cumulative_quote) - Decimal(order["cumulative_quote"])
                delta_fee = Decimal(cumulative_fee) - Decimal(order["cumulative_fee"])
                if delta_base < 0 or delta_quote < 0 or delta_fee < 0:
                    raise LedgerConflict(f"non-monotonic cumulative fill for {client_order_id}")
                intent = await connection.fetchrow(
                    f"SELECT * FROM {self.SCHEMA}.executor_intents WHERE executor_id=$1",
                    order["executor_id"],
                )
                if delta_base > 0:
                    if order["side"] == "BUY":
                        owner = "unassigned" if intent["executor_type"] == "order_executor" else intent["executor_id"]
                        await connection.execute(
                            f"""
                            INSERT INTO {self.SCHEMA}.inventory_lots
                            (owner_id, symbol, total_base, available_base, cost_quote, fees_quote)
                            VALUES($1,$2,$3,$3,$4,$5)
                            ON CONFLICT(owner_id,symbol) DO UPDATE SET
                              total_base={self.SCHEMA}.inventory_lots.total_base+$3,
                              available_base={self.SCHEMA}.inventory_lots.available_base+$3,
                              cost_quote={self.SCHEMA}.inventory_lots.cost_quote+$4,
                              fees_quote={self.SCHEMA}.inventory_lots.fees_quote+$5,
                              updated_at=now()
                            """,
                            owner,
                            order["symbol"],
                            delta_base,
                            delta_quote,
                            max(Decimal("0"), delta_fee),
                        )
                    else:
                        owner = intent["source_owner"] or intent["executor_id"]
                        lot = await connection.fetchrow(
                            f"SELECT * FROM {self.SCHEMA}.inventory_lots WHERE owner_id=$1 AND symbol=$2 FOR UPDATE",
                            owner,
                            order["symbol"],
                        )
                        if lot is None or Decimal(lot["total_base"]) < delta_base:
                            raise LedgerConflict("confirmed SELL exceeds its managed inventory lot")
                        average_cost = (
                            Decimal(lot["cost_quote"]) / Decimal(lot["total_base"])
                            if Decimal(lot["total_base"]) > 0 else Decimal("0")
                        )
                        realized = delta_quote - delta_base * average_cost
                        await connection.execute(
                            f"""
                            UPDATE {self.SCHEMA}.inventory_lots SET
                              total_base=total_base-$3,
                              cost_quote=GREATEST(0,cost_quote-($3*$4)),
                              realized_pnl_quote=realized_pnl_quote+$5,
                              fees_quote=fees_quote+$6,
                              updated_at=now()
                            WHERE owner_id=$1 AND symbol=$2
                            """,
                            owner,
                            order["symbol"],
                            delta_base,
                            average_cost,
                            realized,
                            max(Decimal("0"), delta_fee),
                        )
                        # Direct SELL intents reserve available_base before the order
                        # is submitted. PositionExecutor exits are internal orders and
                        # therefore consume availability only when their fill confirms.
                        if intent["source_owner"] is None:
                            await connection.execute(
                                f"UPDATE {self.SCHEMA}.inventory_lots SET "
                                "available_base=GREATEST(0, available_base-$3), updated_at=now() "
                                "WHERE owner_id=$1 AND symbol=$2",
                                owner,
                                order["symbol"],
                                delta_base,
                            )
                await connection.execute(
                    f"""
                    UPDATE {self.SCHEMA}.managed_orders SET
                      exchange_order_id=$2, cumulative_base=$3, cumulative_quote=$4,
                      cumulative_fee=$5, status=$6, updated_at=now()
                    WHERE client_order_id=$1
                    """,
                    client_order_id,
                    exchange_order_id,
                    Decimal(cumulative_base),
                    Decimal(cumulative_quote),
                    Decimal(cumulative_fee),
                    status,
                )

    async def mark_external_activity(self, payload: Dict[str, Any]) -> None:
        event_id = hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()
        async with self._pool.acquire() as connection:
            async with connection.transaction():
                await connection.execute(
                    f"UPDATE {self.SCHEMA}.runtime_state SET external_activity_latched=TRUE, updated_at=now()"
                )
                await connection.execute(
                    f"""
                    INSERT INTO {self.SCHEMA}.audit_events(event_id,event_type,severity,payload)
                    VALUES($1,'EXTERNAL_EQUITY_ACTIVITY','CRITICAL',$2::jsonb)
                    ON CONFLICT DO NOTHING
                    """,
                    event_id,
                    json.dumps(payload, default=str),
                )

    async def summary(self) -> Dict[str, Any]:
        positions = await self.managed_positions()
        async with self._pool.acquire() as connection:
            state = await connection.fetchrow(
                f"SELECT * FROM {self.SCHEMA}.runtime_state WHERE singleton=TRUE"
            )
            active = await connection.fetchval(
                f"SELECT COUNT(*) FROM {self.SCHEMA}.executor_intents WHERE status = ANY($1::text[])",
                list(ACTIVE_INTENT_STATES),
            )
        return {
            "position_source": "managed_ledger_non_authoritative",
            "external_positions_unknown": True,
            "external_activity_latched": bool(state["external_activity_latched"]),
            "live_authorized": bool(state["live_authorized"]),
            "active_intents": int(active),
            "positions": {
                symbol: {"total": str(position.total), "available": str(position.available)}
                for symbol, position in positions.items()
            },
            "quote_total": str(self._quote_total),
            "quote_available": str(self._quote_available),
            "fresh": await self.is_fresh(),
        }

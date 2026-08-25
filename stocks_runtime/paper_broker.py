from __future__ import annotations

import hashlib
import json
import math
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, Optional

from stocks_runtime.ledger import ACTIVE_INTENT_STATES, LedgerConflict, PostgresManagedLedger


PAPER_OPEN_STATES = {"OPEN", "PARTIALLY_FILLED"}
RTH_PHASES = {"MARKET_OPEN", "RTH", "REGULAR"}
EXTENDED_PHASES = RTH_PHASES | {"PRE_MARKET", "POST_MARKET", "EXTENDED"}


def _decimal(value: Any) -> Decimal:
    if value in (None, ""):
        return Decimal("0")
    return Decimal(str(value))


def _jsonable(row: Any) -> Dict[str, Any]:
    return {key: value for key, value in dict(row).items()}


def _decode_checkpoint_row(row: Any) -> Dict[str, Any]:
    result = _jsonable(row)
    for field in ("config", "metadata", "state"):
        value = result.get(field)
        if isinstance(value, str):
            result[field] = json.loads(value)
    return result


def reconcile_checkpoint_terminal_orders(
    state: Dict[str, Any], orders_by_id: Dict[str, Dict[str, Any]]
) -> Dict[str, Any]:
    """Fold terminal PAPER orders into Executor backup counters on restart.

    An earlier failed recovery intentionally cancels open orders. The following
    healthy restart must not retain dead TrackedOrder references, lose confirmed
    fills, or create a second entry for an already-filled position.
    """
    result = dict(state)
    processed: set[str] = set()
    mappings = (
        ("open_order_id", "entry_filled_backup", "entry_quote_backup"),
        ("close_order_id", "exit_filled_backup", "exit_quote_backup"),
        ("take_profit_order_id", "exit_filled_backup", "exit_quote_backup"),
    )
    for id_field, base_field, quote_field in mappings:
        order_id = result.get(id_field)
        if not order_id:
            continue
        row = orders_by_id.get(str(order_id))
        if row is None:
            raise RuntimeError(f"paper checkpoint order is missing: {order_id}")
        if str(row.get("status", "UNKNOWN")).upper() in PAPER_OPEN_STATES:
            continue
        if str(order_id) not in processed:
            filled_base = _decimal(row.get("filled_base"))
            result[base_field] = str(_decimal(result.get(base_field)) + filled_base)
            result[quote_field] = str(
                _decimal(result.get(quote_field)) + _decimal(row.get("filled_quote"))
            )
            result["fees_quote_backup"] = str(
                _decimal(result.get("fees_quote_backup")) + _decimal(row.get("cumulative_fee"))
            )
            if id_field == "open_order_id" and filled_base > 0:
                result["recovered_entry_frozen"] = True
            processed.add(str(order_id))
        result[id_field] = None
    return result


def _checkpoint_json(value: Any) -> str:
    """Serialize runtime state to PostgreSQL's strict JSON representation."""
    def clean(item: Any) -> Any:
        if isinstance(item, Decimal):
            return str(item)
        if isinstance(item, float):
            return item if math.isfinite(item) else str(item)
        if isinstance(item, datetime):
            return item.isoformat()
        if isinstance(item, dict):
            return {str(key): clean(nested) for key, nested in item.items()}
        if isinstance(item, (list, tuple, set)):
            return [clean(nested) for nested in item]
        if hasattr(item, "model_dump"):
            return clean(item.model_dump())
        if hasattr(item, "value"):
            return clean(item.value)
        if item is None or isinstance(item, (str, int, bool)):
            return item
        return str(item)

    return json.dumps(clean(value), ensure_ascii=False, allow_nan=False)


@dataclass(frozen=True)
class PaperQuote:
    symbol: str
    bid: Decimal
    ask: Decimal
    bid_size: Decimal
    ask_size: Decimal
    event_time: float
    event_id: str

    @classmethod
    def from_payload(cls, payload: Dict[str, Any], now: Optional[float] = None) -> "PaperQuote":
        symbol = str(payload.get("s", payload.get("symbol", ""))).upper()
        bid = _decimal(payload.get("bp", payload.get("bidPrice", payload.get("bid"))))
        ask = _decimal(payload.get("ap", payload.get("askPrice", payload.get("ask"))))
        bid_size = _decimal(payload.get("bs", payload.get("bidQty", payload.get("bidSize"))))
        ask_size = _decimal(payload.get("as", payload.get("askQty", payload.get("askSize"))))
        raw_time = payload.get(
            "T", payload.get("E", payload.get("eventTime", payload.get("time")))
        )
        event_time = float(raw_time or now or time.time())
        if event_time > 10_000_000_000:
            event_time /= 1000
        fingerprint = "|".join(
            (symbol, str(event_time), str(bid), str(ask), str(bid_size), str(ask_size))
        )
        return cls(
            symbol=symbol,
            bid=bid,
            ask=ask,
            bid_size=bid_size,
            ask_size=ask_size,
            event_time=event_time,
            event_id=hashlib.sha256(fingerprint.encode()).hexdigest(),
        )

    @property
    def valid(self) -> bool:
        return bool(self.symbol and self.bid > 0 and self.ask > 0 and self.bid <= self.ask)


class PostgresPaperBroker:
    """Persistent, L1-only execution venue for Binance Stocks PAPER mode.

    It deliberately has no Binance order endpoint. Quotes enter through
    ``process_quote`` and all economic state is committed to the dedicated
    paper schema before Hummingbot order events are emitted.
    """

    def __init__(
        self,
        ledger: PostgresManagedLedger,
        *,
        initial_usdc: Decimal = Decimal("2000"),
        latency_ms: int = 1000,
        market_timeout_seconds: float = 5.0,
        quote_max_age_seconds: float = 10.0,
        equity_snapshot_seconds: float = 60.0,
    ):
        self.ledger = ledger
        self.schema = ledger.SCHEMA
        self.initial_usdc = Decimal(initial_usdc)
        self.latency_seconds = max(0.0, latency_ms / 1000)
        self.market_timeout_seconds = max(0.1, float(market_timeout_seconds))
        self.quote_max_age_seconds = max(0.1, float(quote_max_age_seconds))
        self.equity_snapshot_seconds = max(1.0, float(equity_snapshot_seconds))
        self._run_id = ""
        self._last_snapshot = 0.0
        self._latest_quotes: Dict[str, PaperQuote] = {}
        self._market_phase = "UNKNOWN"
        self._trading_date: Optional[str] = None
        self._trading_status: Dict[str, str] = {}
        self._tradability: Dict[str, str] = {}

    @property
    def run_id(self) -> str:
        return self._run_id

    async def initialize(self) -> None:
        pool = self.ledger._pool
        if pool is None:
            raise RuntimeError("paper ledger must be initialized first")
        async with pool.acquire() as connection:
            await connection.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.schema}.paper_runs (
                    run_id TEXT PRIMARY KEY,
                    initial_usdc NUMERIC NOT NULL,
                    cash_balance NUMERIC NOT NULL,
                    peak_equity NUMERIC NOT NULL,
                    status TEXT NOT NULL DEFAULT 'ACTIVE',
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    closed_at TIMESTAMPTZ
                )
                """
            )
            await connection.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.schema}.paper_state (
                    singleton BOOLEAN PRIMARY KEY DEFAULT TRUE CHECK(singleton),
                    active_run_id TEXT,
                    recovery_required BOOLEAN NOT NULL DEFAULT FALSE,
                    recovery_reason TEXT,
                    last_quote_at TIMESTAMPTZ,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
                )
                """
            )
            await connection.execute(
                f"INSERT INTO {self.schema}.paper_state(singleton) VALUES(TRUE) ON CONFLICT DO NOTHING"
            )
            await connection.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.schema}.paper_orders (
                    sequence BIGSERIAL PRIMARY KEY,
                    run_id TEXT NOT NULL REFERENCES {self.schema}.paper_runs(run_id),
                    client_order_id TEXT NOT NULL UNIQUE,
                    exchange_order_id TEXT NOT NULL UNIQUE,
                    executor_id TEXT,
                    source_owner TEXT,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    order_type TEXT NOT NULL,
                    limit_price NUMERIC,
                    requested_base NUMERIC NOT NULL,
                    filled_base NUMERIC NOT NULL DEFAULT 0,
                    filled_quote NUMERIC NOT NULL DEFAULT 0,
                    cumulative_fee NUMERIC NOT NULL DEFAULT 0,
                    status TEXT NOT NULL,
                    trading_date TEXT,
                    accepted_at TIMESTAMPTZ NOT NULL,
                    eligible_at TIMESTAMPTZ NOT NULL,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    terminal_at TIMESTAMPTZ
                )
                """
            )
            await connection.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.schema}.paper_quote_events (
                    run_id TEXT NOT NULL,
                    event_id TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    bid NUMERIC NOT NULL,
                    ask NUMERIC NOT NULL,
                    bid_size NUMERIC NOT NULL,
                    ask_size NUMERIC NOT NULL,
                    event_time TIMESTAMPTZ NOT NULL,
                    processed_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    PRIMARY KEY(run_id,event_id)
                )
                """
            )
            await connection.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.schema}.paper_trades (
                    trade_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    client_order_id TEXT NOT NULL,
                    executor_id TEXT,
                    quote_event_id TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    quantity NUMERIC NOT NULL,
                    price NUMERIC NOT NULL,
                    quote_amount NUMERIC NOT NULL,
                    fee_delta NUMERIC NOT NULL,
                    is_taker BOOLEAN NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL
                )
                """
            )
            await connection.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.schema}.paper_equity_snapshots (
                    run_id TEXT NOT NULL,
                    snapshot_at TIMESTAMPTZ NOT NULL,
                    cash_balance NUMERIC NOT NULL,
                    reserved_cash NUMERIC NOT NULL,
                    positions_value NUMERIC NOT NULL,
                    equity NUMERIC NOT NULL,
                    peak_equity NUMERIC NOT NULL,
                    drawdown_pct NUMERIC NOT NULL,
                    PRIMARY KEY(run_id,snapshot_at)
                )
                """
            )
            await connection.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.schema}.paper_executor_checkpoints (
                    run_id TEXT NOT NULL,
                    executor_id TEXT NOT NULL,
                    config JSONB NOT NULL,
                    metadata JSONB NOT NULL,
                    state JSONB NOT NULL,
                    status TEXT NOT NULL,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    PRIMARY KEY(run_id,executor_id)
                )
                """
            )
            state = await connection.fetchrow(
                f"SELECT active_run_id FROM {self.schema}.paper_state WHERE singleton=TRUE FOR UPDATE"
            )
            self._run_id = str(state["active_run_id"] or "")
            if not self._run_id:
                self._run_id = uuid.uuid4().hex
                await connection.execute(
                    f"INSERT INTO {self.schema}.paper_runs(run_id,initial_usdc,cash_balance,peak_equity) "
                    "VALUES($1,$2,$2,$2)",
                    self._run_id,
                    self.initial_usdc,
                )
                await connection.execute(
                    f"UPDATE {self.schema}.paper_state SET active_run_id=$1,updated_at=now() WHERE singleton=TRUE",
                    self._run_id,
                )
        await self._load_latest_quotes()
        account = await self.account()
        self.ledger.set_quote_balances(
            Decimal(account["cash_balance"]), Decimal(account["available_cash"])
        )

    async def _load_latest_quotes(self) -> None:
        async with self.ledger._pool.acquire() as connection:
            rows = await connection.fetch(
                f"""
                SELECT DISTINCT ON(symbol) symbol,bid,ask,bid_size,ask_size,event_time,event_id
                FROM {self.schema}.paper_quote_events WHERE run_id=$1
                ORDER BY symbol,event_time DESC
                """,
                self._run_id,
            )
        for row in rows:
            self._latest_quotes[str(row["symbol"])] = PaperQuote(
                symbol=str(row["symbol"]),
                bid=Decimal(row["bid"]), ask=Decimal(row["ask"]),
                bid_size=Decimal(row["bid_size"]), ask_size=Decimal(row["ask_size"]),
                event_time=row["event_time"].timestamp(), event_id=str(row["event_id"]),
            )

    def update_market_state(
        self,
        market_phase: str,
        trading_status: Optional[Dict[str, str]] = None,
        tradability: Optional[Dict[str, str]] = None,
        trading_date: Optional[str] = None,
    ) -> None:
        self._market_phase = str(market_phase or "UNKNOWN").upper()
        if trading_status is not None:
            self._trading_status = {str(k).upper(): str(v).upper() for k, v in trading_status.items()}
        if tradability is not None:
            self._tradability = {str(k).upper(): str(v).upper() for k, v in tradability.items()}
        if trading_date:
            self._trading_date = str(trading_date)

    @property
    def trading_date(self) -> Optional[str]:
        return self._trading_date

    def latest_quote(self, symbol: str) -> Optional[PaperQuote]:
        quote = self._latest_quotes.get(symbol.upper())
        if quote is None or time.time() - quote.event_time > self.quote_max_age_seconds:
            return None
        return quote

    @staticmethod
    def cumulative_fee(notional: Decimal) -> Decimal:
        if notional <= 0:
            return Decimal("0")
        return Decimal("0.35") if notional <= Decimal("350") else notional * Decimal("0.001")

    async def create_order(
        self,
        *,
        client_order_id: str,
        executor_id: Optional[str],
        symbol: str,
        side: str,
        order_type: str,
        amount: Decimal,
        limit_price: Optional[Decimal],
        trading_date: Optional[str],
    ) -> Dict[str, Any]:
        symbol = symbol.upper()
        side = side.upper()
        order_type = order_type.upper()
        amount = Decimal(amount)
        now = datetime.now(timezone.utc)
        quote = self.latest_quote(symbol)
        reference = Decimal(limit_price or 0)
        if order_type == "MARKET":
            reference = quote.ask if quote and side == "BUY" else quote.bid if quote else Decimal("0")
        if amount <= 0 or reference <= 0:
            raise ValueError("paper order requires positive amount and fresh executable price")
        if side == "BUY":
            account = await self.account()
            required = amount * reference + self.cumulative_fee(amount * reference)
            if Decimal(account["available_cash"]) < required:
                raise PermissionError("paper USDC balance is insufficient")
        async with self.ledger._pool.acquire() as connection:
            intent = await connection.fetchrow(
                f"SELECT source_owner FROM {self.schema}.executor_intents WHERE executor_id=$1",
                executor_id,
            )
            source_owner = intent["source_owner"] if intent else None
            exchange_id = f"PAPER-{hashlib.sha256(client_order_id.encode()).hexdigest()[:24]}"
            row = await connection.fetchrow(
                f"""
                INSERT INTO {self.schema}.paper_orders
                  (run_id,client_order_id,exchange_order_id,executor_id,source_owner,symbol,side,
                   order_type,limit_price,requested_base,status,trading_date,accepted_at,eligible_at)
                VALUES($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,'OPEN',$11,$12,$13)
                ON CONFLICT(client_order_id) DO UPDATE SET updated_at=now()
                RETURNING *
                """,
                self._run_id, client_order_id, exchange_id, executor_id, source_owner, symbol, side,
                order_type, limit_price, amount, trading_date, now,
                datetime.fromtimestamp(now.timestamp() + self.latency_seconds, tz=timezone.utc),
            )
        return _jsonable(row)

    async def cancel_order(self, client_order_id: str, status: str = "CANCELED") -> bool:
        async with self.ledger._pool.acquire() as connection:
            result = await connection.execute(
                f"""
                UPDATE {self.schema}.paper_orders SET status=$2,terminal_at=now(),updated_at=now()
                WHERE run_id=$1 AND client_order_id=$3 AND status=ANY($4::text[])
                """,
                self._run_id, status, client_order_id, list(PAPER_OPEN_STATES),
            )
        return result.endswith("1")

    async def order(self, client_order_id: str) -> Optional[Dict[str, Any]]:
        async with self.ledger._pool.acquire() as connection:
            row = await connection.fetchrow(
                f"SELECT * FROM {self.schema}.paper_orders WHERE run_id=$1 AND client_order_id=$2",
                self._run_id, client_order_id,
            )
        return _jsonable(row) if row else None

    def _direction_allowed(self, symbol: str, side: str) -> bool:
        status = self._trading_status.get(symbol, "UNKNOWN")
        if status not in {"TRADING", "ACTIVE", "NORMAL"}:
            return False
        tradability = self._tradability.get(symbol, "NONE")
        return tradability in {"BOTH", "BUY_SELL", "ALL", side}

    async def process_quote(self, payload: Dict[str, Any]) -> list[str]:
        quote = PaperQuote.from_payload(payload)
        if not quote.valid:
            return []
        self._latest_quotes[quote.symbol] = quote
        changed: list[str] = []
        async with self.ledger._pool.acquire() as connection:
            async with connection.transaction():
                inserted = await connection.fetchval(
                    f"""
                    INSERT INTO {self.schema}.paper_quote_events
                      (run_id,event_id,symbol,bid,ask,bid_size,ask_size,event_time)
                    VALUES($1,$2,$3,$4,$5,$6,$7,$8)
                    ON CONFLICT DO NOTHING RETURNING TRUE
                    """,
                    self._run_id, quote.event_id, quote.symbol, quote.bid, quote.ask,
                    quote.bid_size, quote.ask_size,
                    datetime.fromtimestamp(quote.event_time, tz=timezone.utc),
                )
                if inserted:
                    await connection.execute(
                        f"UPDATE {self.schema}.paper_state SET last_quote_at=$1,updated_at=now() WHERE singleton=TRUE",
                        datetime.fromtimestamp(quote.event_time, tz=timezone.utc),
                    )
                orders = await connection.fetch(
                    f"""
                    SELECT * FROM {self.schema}.paper_orders
                    WHERE run_id=$1 AND symbol=$2 AND status=ANY($3::text[])
                    ORDER BY sequence FOR UPDATE
                    """,
                    self._run_id, quote.symbol, list(PAPER_OPEN_STATES),
                )
                # A repeated REST/WS snapshot can advance order timeout/expiry,
                # but its displayed size must never be consumed twice.
                bid_liquidity = quote.bid_size if inserted else Decimal("0")
                ask_liquidity = quote.ask_size if inserted else Decimal("0")
                now = datetime.now(timezone.utc)
                for order in orders:
                    client_id = str(order["client_order_id"])
                    if (
                        order["trading_date"] and self._trading_date
                        and str(order["trading_date"]) != self._trading_date
                    ):
                        await self._terminal(connection, client_id, "EXPIRED")
                        changed.append(client_id)
                        continue
                    age = now.timestamp() - order["accepted_at"].timestamp()
                    if order["order_type"] == "MARKET" and age >= self.market_timeout_seconds:
                        await self._terminal(connection, client_id, "CANCELED")
                        changed.append(client_id)
                        continue
                    if now < order["eligible_at"] or not self._direction_allowed(quote.symbol, order["side"]):
                        continue
                    if order["order_type"] == "MARKET" and self._market_phase not in RTH_PHASES:
                        continue
                    if order["order_type"] == "LIMIT" and self._market_phase not in EXTENDED_PHASES:
                        continue
                    side = str(order["side"])
                    limit = Decimal(order["limit_price"] or 0)
                    if side == "BUY":
                        if ask_liquidity <= 0 or (order["order_type"] == "LIMIT" and quote.ask > limit):
                            continue
                        available = ask_liquidity
                        price = quote.ask
                    else:
                        if bid_liquidity <= 0 or (order["order_type"] == "LIMIT" and quote.bid < limit):
                            continue
                        available = bid_liquidity
                        price = quote.bid
                    remaining = Decimal(order["requested_base"]) - Decimal(order["filled_base"])
                    quantity = min(remaining, available)
                    if side == "SELL":
                        quantity = await self._owned_sell_quantity(connection, order, quantity)
                    if quantity <= 0:
                        continue
                    await self._apply_fill(connection, order, quote, quantity, price)
                    changed.append(client_id)
                    if side == "BUY":
                        ask_liquidity -= quantity
                    else:
                        bid_liquidity -= quantity
        # Economic cash and the ownership ledger must move as one observable
        # unit. Otherwise a SELL briefly appears as both proceeds and inventory,
        # fabricating an equity peak before the next reconciliation loop.
        if changed:
            await self.reconcile_managed_fills()
        await self.snapshot_equity_if_due()
        account = await self.account()
        self.ledger.set_quote_balances(
            Decimal(account["cash_balance"]), Decimal(account["available_cash"])
        )
        return changed

    async def _owned_sell_quantity(self, connection, order, proposed: Decimal) -> Decimal:
        """Cap a simulated SELL to inventory owned by its Executor.

        A PositionExecutor does not reserve its internal exit order.  A
        concurrent reduce Executor can therefore consume part of the lot after
        that exit was created.  Rechecking ownership at the actual quote event
        prevents the paper exchange from fabricating a short sale.  Direct
        SELL intents already reserve availability, so their own reservation is
        bounded by total_base instead.
        """
        owner = order["source_owner"] or order["executor_id"]
        if not owner:
            return Decimal("0")
        lot = await connection.fetchrow(
            f"SELECT total_base,available_base FROM {self.schema}.inventory_lots "
            "WHERE owner_id=$1 AND symbol=$2 FOR UPDATE",
            str(owner), str(order["symbol"]),
        )
        if lot is None:
            return Decimal("0")
        owned = Decimal(lot["total_base"])
        if order["source_owner"] is None:
            owned = min(owned, Decimal(lot["available_base"]))
        return min(Decimal(proposed), max(Decimal("0"), owned))

    async def _terminal(self, connection, client_id: str, status: str) -> None:
        await connection.execute(
            f"UPDATE {self.schema}.paper_orders SET status=$2,terminal_at=now(),updated_at=now() "
            "WHERE client_order_id=$1",
            client_id, status,
        )

    async def _apply_fill(self, connection, order, quote: PaperQuote, quantity: Decimal, price: Decimal) -> None:
        old_base = Decimal(order["filled_base"])
        old_quote = Decimal(order["filled_quote"])
        old_fee = Decimal(order["cumulative_fee"])
        new_base = old_base + quantity
        quote_delta = quantity * price
        new_quote = old_quote + quote_delta
        new_fee = self.cumulative_fee(new_quote)
        fee_delta = new_fee - old_fee
        final = new_base >= Decimal(order["requested_base"])
        status = "FILLED" if final else "PARTIALLY_FILLED"
        if order["side"] == "BUY":
            cash_delta = -(quote_delta + fee_delta)
        else:
            cash_delta = quote_delta - fee_delta
        cash = await connection.fetchval(
            f"UPDATE {self.schema}.paper_runs SET cash_balance=cash_balance+$2 "
            "WHERE run_id=$1 RETURNING cash_balance",
            self._run_id, cash_delta,
        )
        if Decimal(cash) < 0:
            raise PermissionError("paper fill would make USDC cash negative")
        await connection.execute(
            f"""
            UPDATE {self.schema}.paper_orders SET filled_base=$2,filled_quote=$3,cumulative_fee=$4,
              status=$5,updated_at=now(),terminal_at=CASE WHEN $5='FILLED' THEN now() ELSE NULL END
            WHERE client_order_id=$1
            """,
            order["client_order_id"], new_base, new_quote, new_fee, status,
        )
        trade_id = hashlib.sha256(
            f"{self._run_id}|{order['client_order_id']}|{new_base}|{quote.event_id}".encode()
        ).hexdigest()
        await connection.execute(
            f"""
            INSERT INTO {self.schema}.paper_trades
              (trade_id,run_id,client_order_id,executor_id,quote_event_id,symbol,side,
               quantity,price,quote_amount,fee_delta,is_taker,created_at)
            VALUES($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13)
            ON CONFLICT DO NOTHING
            """,
            trade_id, self._run_id, order["client_order_id"], order["executor_id"], quote.event_id,
            order["symbol"], order["side"], quantity, price, quote_delta, fee_delta,
            order["order_type"] == "MARKET",
            datetime.fromtimestamp(quote.event_time, tz=timezone.utc),
        )

    async def _reserved_cash(self, connection) -> Decimal:
        rows = await connection.fetch(
            f"""
            SELECT symbol,order_type,limit_price,requested_base,filled_base,filled_quote,cumulative_fee
            FROM {self.schema}.paper_orders
            WHERE run_id=$1 AND side='BUY' AND status=ANY($2::text[])
            """,
            self._run_id, list(PAPER_OPEN_STATES),
        )
        reserved = Decimal("0")
        for row in rows:
            remaining = Decimal(row["requested_base"]) - Decimal(row["filled_base"])
            quote = self._latest_quotes.get(str(row["symbol"]))
            price = Decimal(row["limit_price"] or 0)
            if row["order_type"] == "MARKET" and quote:
                price = quote.ask
            remaining_quote = max(Decimal("0"), remaining * price)
            total_fee = self.cumulative_fee(Decimal(row["filled_quote"]) + remaining_quote)
            reserved += remaining_quote + max(Decimal("0"), total_fee - Decimal(row["cumulative_fee"]))
        return reserved

    async def account(self) -> Dict[str, Any]:
        async with self.ledger._pool.acquire() as connection:
            run = await connection.fetchrow(
                f"SELECT * FROM {self.schema}.paper_runs WHERE run_id=$1", self._run_id
            )
            reserved = await self._reserved_cash(connection)
            state = await connection.fetchrow(
                f"SELECT recovery_required,recovery_reason,last_quote_at FROM {self.schema}.paper_state "
                "WHERE singleton=TRUE"
            )
        positions = await self.ledger.managed_positions()
        positions_value = Decimal("0")
        position_rows = []
        for symbol, position in sorted(positions.items()):
            quote = self._latest_quotes.get(symbol)
            bid = quote.bid if quote else Decimal("0")
            value = position.total * bid
            positions_value += value
            position_rows.append({
                "symbol": symbol, "total": str(position.total), "available": str(position.available),
                "mark_bid": str(bid), "market_value": str(value),
            })
        cash = Decimal(run["cash_balance"])
        equity = cash + positions_value
        peak = max(Decimal(run["peak_equity"]), equity)
        if peak != Decimal(run["peak_equity"]):
            async with self.ledger._pool.acquire() as connection:
                await connection.execute(
                    f"UPDATE {self.schema}.paper_runs SET peak_equity=$2 WHERE run_id=$1",
                    self._run_id, peak,
                )
        drawdown = (peak - equity) / peak if peak > 0 else Decimal("0")
        return {
            "account_scope": "paper",
            "paper_run_id": self._run_id,
            "initial_usdc": str(run["initial_usdc"]),
            "cash_balance": str(cash),
            "reserved_cash": str(reserved),
            "available_cash": str(max(Decimal("0"), cash - reserved)),
            "positions_value": str(positions_value),
            "equity": str(equity),
            "peak_equity": str(peak),
            "drawdown_pct": str(drawdown * 100),
            "net_pnl": str(equity - Decimal(run["initial_usdc"])),
            "positions": position_rows,
            "recovery_required": bool(state["recovery_required"]),
            "recovery_reason": state["recovery_reason"],
            "last_quote_at": state["last_quote_at"],
        }

    async def snapshot_equity_if_due(self, force: bool = False) -> None:
        now = time.time()
        if not force and now - self._last_snapshot < self.equity_snapshot_seconds:
            return
        account = await self.account()
        instant = datetime.fromtimestamp(now, tz=timezone.utc).replace(microsecond=0)
        async with self.ledger._pool.acquire() as connection:
            await connection.execute(
                f"""
                INSERT INTO {self.schema}.paper_equity_snapshots
                  (run_id,snapshot_at,cash_balance,reserved_cash,positions_value,equity,peak_equity,drawdown_pct)
                VALUES($1,$2,$3,$4,$5,$6,$7,$8) ON CONFLICT(run_id,snapshot_at) DO NOTHING
                """,
                self._run_id, instant, Decimal(account["cash_balance"]), Decimal(account["reserved_cash"]),
                Decimal(account["positions_value"]), Decimal(account["equity"]),
                Decimal(account["peak_equity"]), Decimal(account["drawdown_pct"]),
            )
        self._last_snapshot = now

    async def orders(self, symbol: Optional[str] = None, open_only: bool = False, limit: int = 500):
        clauses = ["run_id=$1"]
        args: list[Any] = [self._run_id]
        if symbol:
            args.append(symbol.upper())
            clauses.append(f"symbol=${len(args)}")
        if open_only:
            args.append(list(PAPER_OPEN_STATES))
            clauses.append(f"status=ANY(${len(args)}::text[])")
        args.append(max(1, min(int(limit), 2000)))
        async with self.ledger._pool.acquire() as connection:
            rows = await connection.fetch(
                f"SELECT * FROM {self.schema}.paper_orders WHERE {' AND '.join(clauses)} "
                f"ORDER BY sequence DESC LIMIT ${len(args)}",
                *args,
            )
        return [_jsonable(row) for row in rows]

    async def trades(self, symbol: Optional[str] = None, limit: int = 500):
        args: list[Any] = [self._run_id]
        where = "run_id=$1"
        if symbol:
            args.append(symbol.upper())
            where += f" AND symbol=${len(args)}"
        args.append(max(1, min(int(limit), 2000)))
        async with self.ledger._pool.acquire() as connection:
            rows = await connection.fetch(
                f"SELECT * FROM {self.schema}.paper_trades WHERE {where} "
                f"ORDER BY created_at DESC LIMIT ${len(args)}",
                *args,
            )
        return [_jsonable(row) for row in rows]

    async def equity(self, start: Optional[datetime] = None, end: Optional[datetime] = None, limit: int = 5000):
        args: list[Any] = [self._run_id]
        where = "run_id=$1"
        if start:
            args.append(start)
            where += f" AND snapshot_at>=${len(args)}"
        if end:
            args.append(end)
            where += f" AND snapshot_at<=${len(args)}"
        args.append(max(1, min(int(limit), 10000)))
        async with self.ledger._pool.acquire() as connection:
            rows = await connection.fetch(
                f"SELECT * FROM {self.schema}.paper_equity_snapshots WHERE {where} "
                f"ORDER BY snapshot_at LIMIT ${len(args)}", *args,
            )
        return [_jsonable(row) for row in rows]

    async def performance(self, window_seconds: Optional[int] = None) -> Dict[str, Any]:
        await self.snapshot_equity_if_due(force=True)
        account = await self.account()
        async with self.ledger._pool.acquire() as connection:
            first = None
            if window_seconds:
                first = await connection.fetchrow(
                    f"SELECT * FROM {self.schema}.paper_equity_snapshots "
                    "WHERE run_id=$1 AND snapshot_at<=now()-($2 * interval '1 second') "
                    "ORDER BY snapshot_at DESC LIMIT 1",
                    self._run_id, int(window_seconds),
                )
            totals = await connection.fetchrow(
                f"SELECT COALESCE(SUM(fee_delta),0) fees,COUNT(*) fills,"
                f"COALESCE(SUM(quote_amount),0) volume FROM {self.schema}.paper_trades WHERE run_id=$1",
                self._run_id,
            )
        baseline = Decimal(first["equity"]) if first else Decimal(account["initial_usdc"])
        current = Decimal(account["equity"])
        return {
            **account,
            "window_seconds": window_seconds,
            "window_start_equity": str(baseline),
            "window_pnl": str(current - baseline),
            "fees": str(totals["fees"]),
            "fill_count": int(totals["fills"]),
            "volume": str(totals["volume"]),
        }

    async def summary(
        self,
        *,
        market_phase: str,
        connector_ready: bool,
        active_executor_count: int,
    ) -> Dict[str, Any]:
        """Return one authoritative PAPER performance snapshot for operators.

        The Telegram client must not reconstruct PnL from unrelated account,
        trade and position requests.  This method pins every value to the
        active run and reconciles symbol PnL back to account equity.
        """
        await self.snapshot_equity_if_due(force=True)
        account = await self.account()
        now = datetime.now(timezone.utc)
        async with self.ledger._pool.acquire() as connection:
            run = await connection.fetchrow(
                f"SELECT created_at FROM {self.schema}.paper_runs WHERE run_id=$1",
                self._run_id,
            )
            totals = await connection.fetchrow(
                f"SELECT COALESCE(SUM(fee_delta),0) fees,COUNT(*) fills,"
                f"COALESCE(SUM(quote_amount),0) volume FROM {self.schema}.paper_trades WHERE run_id=$1",
                self._run_id,
            )
            lots = await connection.fetch(
                f"""
                SELECT symbol,SUM(total_base) total_base,SUM(available_base) available_base,
                       SUM(cost_quote) cost_quote,SUM(realized_pnl_quote) realized_pnl_quote,
                       SUM(fees_quote) fees_quote
                FROM {self.schema}.inventory_lots
                GROUP BY symbol
                HAVING SUM(total_base)<>0 OR SUM(realized_pnl_quote)<>0 OR SUM(fees_quote)<>0
                ORDER BY symbol
                """
            )
            open_order_count = await connection.fetchval(
                f"SELECT COUNT(*) FROM {self.schema}.paper_orders "
                "WHERE run_id=$1 AND status=ANY($2::text[])",
                self._run_id,
                list(PAPER_OPEN_STATES),
            )
            snapshots = await connection.fetch(
                f"SELECT snapshot_at,equity FROM {self.schema}.paper_equity_snapshots "
                "WHERE run_id=$1 ORDER BY snapshot_at",
                self._run_id,
            )

        created_at = run["created_at"] if run else now
        run_age_seconds = max(0, int((now - created_at).total_seconds()))
        current_equity = Decimal(account["equity"])
        initial_equity = Decimal(account["initial_usdc"])
        windows: Dict[str, Dict[str, Any]] = {}
        for name, seconds in (("4h", 4 * 3600), ("24h", 24 * 3600), ("7d", 7 * 86400)):
            cutoff = now.timestamp() - seconds
            baseline_row = next(
                (row for row in reversed(snapshots) if row["snapshot_at"].timestamp() <= cutoff),
                None,
            )
            complete = run_age_seconds >= seconds and baseline_row is not None
            baseline_equity = Decimal(baseline_row["equity"]) if baseline_row else initial_equity
            baseline_at = baseline_row["snapshot_at"] if baseline_row else created_at
            windows[name] = {
                "pnl": str(current_equity - baseline_equity),
                "baseline_equity": str(baseline_equity),
                "baseline_at": baseline_at.isoformat(),
                "window_complete": complete,
            }
        windows["all"] = {
            "pnl": str(current_equity - initial_equity),
            "baseline_equity": str(initial_equity),
            "baseline_at": created_at.isoformat(),
            "window_complete": True,
        }

        positions: list[Dict[str, Any]] = []
        missing_marks: list[str] = []
        symbol_net_total = Decimal("0")
        for row in lots:
            symbol = str(row["symbol"])
            total = Decimal(row["total_base"])
            available = max(Decimal("0"), Decimal(row["available_base"]))
            cost = Decimal(row["cost_quote"])
            realized = Decimal(row["realized_pnl_quote"])
            fees = Decimal(row["fees_quote"])
            quote = self._latest_quotes.get(symbol)
            if total > 0 and quote is None:
                missing_marks.append(symbol)
                mark_bid = None
                mark_at = None
                market_value = unrealized = net_pnl = None
            else:
                mark_bid_value = quote.bid if quote else Decimal("0")
                market_value_value = total * mark_bid_value
                unrealized_value = market_value_value - cost
                net_pnl_value = realized + unrealized_value - fees
                mark_bid = str(mark_bid_value) if quote else None
                mark_at = (
                    datetime.fromtimestamp(quote.event_time, tz=timezone.utc).isoformat()
                    if quote else None
                )
                market_value = str(market_value_value)
                unrealized = str(unrealized_value)
                net_pnl = str(net_pnl_value)
                symbol_net_total += net_pnl_value
            positions.append({
                "symbol": symbol,
                "total": str(total),
                "available": str(available),
                "cost_quote": str(cost),
                "average_cost": str(cost / total) if total > 0 else "0",
                "mark_bid": mark_bid,
                "mark_at": mark_at,
                "market_value": market_value,
                "realized_pnl": str(realized),
                "unrealized_pnl": unrealized,
                "fees": str(fees),
                "net_pnl": net_pnl,
            })

        account_net = Decimal(account["net_pnl"])
        difference = symbol_net_total - account_net
        reconciled = not missing_marks and abs(difference) <= Decimal("0.000001")
        phase = str(market_phase or self._market_phase or "UNKNOWN").upper()
        closed = phase in {"MARKET_CLOSED", "OVERNIGHT", "CLOSED"}
        last_quote_at = account.get("last_quote_at")
        quote_recent = bool(
            isinstance(last_quote_at, datetime)
            and (now - last_quote_at).total_seconds() <= max(30.0, self.quote_max_age_seconds * 3)
        )
        if account["recovery_required"]:
            quote_health = "RECOVERY_REQUIRED"
        elif closed and last_quote_at and not missing_marks:
            quote_health = "MARKET_CLOSED_LAST_TRUSTED"
        elif missing_marks:
            quote_health = "MARKET_DATA_UNAVAILABLE"
        elif phase in {"UNKNOWN", "PUBLIC_DATA_ONLY"} and quote_recent:
            quote_health = "MARKET_STATE_UNAVAILABLE"
        elif quote_recent:
            quote_health = "FRESH"
        elif not connector_ready:
            quote_health = "MARKET_DATA_UNAVAILABLE"
        else:
            quote_health = "AWAITING_FIRST_QUOTE"

        reconciliation_error = None
        if missing_marks:
            reconciliation_error = "missing_position_mark:" + ",".join(missing_marks)
        elif not reconciled:
            reconciliation_error = f"symbol_account_pnl_difference:{difference}"
        return {
            "schema": "binance-stocks-paper-summary-v1",
            "generated_at": now.isoformat(),
            "account_scope": "paper",
            "paper_run_id": self._run_id,
            "run_created_at": created_at.isoformat(),
            "run_age_seconds": run_age_seconds,
            "market_phase": phase,
            "quote_health": quote_health,
            "last_quote_at": account.get("last_quote_at"),
            "valuation_complete": not missing_marks,
            "account": account,
            "windows": windows,
            "totals": {
                "fees": str(totals["fees"]),
                "fill_count": int(totals["fills"]),
                "volume": str(totals["volume"]),
                "open_order_count": int(open_order_count or 0),
                "active_executor_count": int(active_executor_count),
            },
            "positions": positions,
            "reconciliation": {
                "ok": reconciled,
                "symbol_net_pnl": str(symbol_net_total) if not missing_marks else None,
                "account_net_pnl": str(account_net),
                "difference": str(difference) if not missing_marks else None,
                "error": reconciliation_error,
            },
        }

    async def reconcile_managed_fills(self) -> int:
        """Replay paper cumulative fills into the ownership ledger idempotently."""
        async with self.ledger._pool.acquire() as connection:
            rows = await connection.fetch(
                f"""
                SELECT p.client_order_id,p.exchange_order_id,p.filled_base,p.filled_quote,
                       p.cumulative_fee,p.status,m.client_order_id AS managed_id,
                       m.cumulative_base AS managed_base,m.cumulative_quote AS managed_quote,
                       m.cumulative_fee AS managed_fee
                FROM {self.schema}.paper_orders p
                LEFT JOIN {self.schema}.managed_orders m USING(client_order_id)
                WHERE p.run_id=$1 AND p.filled_base>0
                  AND (m.client_order_id IS NULL OR m.cumulative_base<p.filled_base
                       OR m.cumulative_quote<p.filled_quote OR m.cumulative_fee<p.cumulative_fee)
                ORDER BY p.sequence
                """,
                self._run_id,
            )
        repaired = 0
        for row in rows:
            if row["managed_id"] is None:
                await self.mark_recovery_required(
                    f"paper_fill_without_managed_order:{row['client_order_id']}"
                )
                continue
            try:
                await self.ledger.record_cumulative_fill(
                    client_order_id=str(row["client_order_id"]),
                    exchange_order_id=str(row["exchange_order_id"]),
                    cumulative_base=Decimal(row["filled_base"]),
                    cumulative_quote=Decimal(row["filled_quote"]),
                    cumulative_fee=Decimal(row["cumulative_fee"]),
                    status=str(row["status"]),
                )
                repaired += 1
            except LedgerConflict as exc:
                # Preserve the account and keep the API available for an
                # operator-owned flatten/reset.  A single corrupt historical
                # fill must not crash startup or suppress reconciliation of
                # later valid orders.
                await self.mark_recovery_required(
                    f"paper_fill_ownership_conflict:{row['client_order_id']}:{exc}"
                )
        return repaired

    async def save_checkpoint(
        self, executor_id: str, config: Dict[str, Any], metadata: Dict[str, Any], state: Dict[str, Any], status: str
    ) -> None:
        async with self.ledger._pool.acquire() as connection:
            await connection.execute(
                f"""
                INSERT INTO {self.schema}.paper_executor_checkpoints
                  (run_id,executor_id,config,metadata,state,status)
                VALUES($1,$2,$3::jsonb,$4::jsonb,$5::jsonb,$6)
                ON CONFLICT(run_id,executor_id) DO UPDATE SET
                  config=$3::jsonb,metadata=$4::jsonb,state=$5::jsonb,status=$6,updated_at=now()
                """,
                self._run_id, executor_id, _checkpoint_json(config),
                _checkpoint_json(metadata), _checkpoint_json(state), status,
            )

    async def active_checkpoints(self) -> list[Dict[str, Any]]:
        async with self.ledger._pool.acquire() as connection:
            rows = await connection.fetch(
                f"SELECT * FROM {self.schema}.paper_executor_checkpoints "
                "WHERE run_id=$1 AND status NOT IN('TERMINATED','COMPLETED') ORDER BY updated_at",
                self._run_id,
            )
        return [_decode_checkpoint_row(row) for row in rows]

    async def mark_recovery_required(self, reason: str) -> None:
        async with self.ledger._pool.acquire() as connection:
            await connection.execute(
                f"UPDATE {self.schema}.paper_state SET recovery_required=TRUE,recovery_reason=$1,updated_at=now() "
                "WHERE singleton=TRUE",
                reason[:1000],
            )

    async def clear_executor_recovery_error(self) -> None:
        """Clear only the recoverable Executor checkpoint decoding latch."""
        async with self.ledger._pool.acquire() as connection:
            await connection.execute(
                f"UPDATE {self.schema}.paper_state SET recovery_required=FALSE,recovery_reason=NULL,updated_at=now() "
                "WHERE singleton=TRUE AND recovery_reason LIKE 'executor_recovery_failed:%'"
            )

    async def reset(self, confirmation: str, active_executor_count: int) -> Dict[str, Any]:
        if confirmation != "RESET PAPER ACCOUNT TO 2000 USDC":
            raise ValueError("explicit reset confirmation does not match")
        if active_executor_count:
            raise RuntimeError("paper account has active executors")
        if await self.orders(open_only=True):
            raise RuntimeError("paper account has open orders")
        async with self.ledger._pool.acquire() as connection:
            active_intents = await connection.fetchval(
                f"SELECT COUNT(*) FROM {self.schema}.executor_intents WHERE status=ANY($1::text[])",
                list(ACTIVE_INTENT_STATES),
            )
        if int(active_intents or 0):
            raise RuntimeError("paper account has active executor intents")
        positions = await self.ledger.managed_positions()
        if any(position.total > 0 for position in positions.values()):
            raise RuntimeError("paper account has open positions")
        previous = self._run_id
        new_run = uuid.uuid4().hex
        async with self.ledger._pool.acquire() as connection:
            async with connection.transaction():
                await connection.execute(
                    f"UPDATE {self.schema}.paper_runs SET status='CLOSED',closed_at=now() WHERE run_id=$1",
                    previous,
                )
                await connection.execute(
                    f"INSERT INTO {self.schema}.paper_runs(run_id,initial_usdc,cash_balance,peak_equity) "
                    "VALUES($1,$2,$2,$2)",
                    new_run, self.initial_usdc,
                )
                await connection.execute(
                    f"UPDATE {self.schema}.paper_state SET active_run_id=$1,recovery_required=FALSE,"
                    "recovery_reason=NULL,updated_at=now() WHERE singleton=TRUE",
                    new_run,
                )
                # scheduled_executors and managed_orders both reference
                # executor_intents.  Clear dependants first so a completed
                # asynchronous plan cannot make an otherwise safe reset fail.
                await connection.execute(f"DELETE FROM {self.schema}.scheduled_executors")
                await connection.execute(f"DELETE FROM {self.schema}.managed_orders")
                await connection.execute(f"DELETE FROM {self.schema}.executor_intents")
                await connection.execute(f"DELETE FROM {self.schema}.inventory_lots")
        self._run_id = new_run
        self._latest_quotes.clear()
        self.ledger.set_quote_balances(self.initial_usdc, self.initial_usdc)
        return {"previous_run_id": previous, "paper_run_id": new_run, "initial_usdc": str(self.initial_usdc)}

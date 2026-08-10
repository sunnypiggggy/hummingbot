"""Idempotent, journalled emergency market execution for Grid and DCA."""

from __future__ import annotations

import json
import threading
import time
from decimal import Decimal
from functools import wraps
from typing import Any, Mapping


TERMINAL_STATUSES = {"FILLED", "CANCELED", "EXPIRED", "REJECTED", "EXPIRED_IN_MATCH"}


class ExecutionPending(RuntimeError):
    """The exchange has not yet made the existing order terminal."""


class _LeaseHeartbeat:
    """Keep an asset lease alive while a synchronous exchange call blocks."""

    def __init__(self, ledger: Any, asset: str, holder: str, ttl_seconds: int):
        self.ledger = ledger
        self.asset = asset
        self.holder = holder
        self.ttl_seconds = max(int(ttl_seconds), 1)
        self.interval = max(min(self.ttl_seconds / 3, 10.0), 0.1)
        self.stopped = threading.Event()
        self.lost = threading.Event()
        self.thread: threading.Thread | None = None

    def start(self) -> None:
        if self.ledger is None:
            return
        if not self.ledger.renew_lease(
            self.asset, self.holder, ttl_seconds=self.ttl_seconds
        ):
            raise RuntimeError("inventory lease was not owned before exchange execution")
        self.thread = threading.Thread(
            target=self._run,
            name=f"inventory-lease-{self.asset}-{self.holder}",
            daemon=True,
        )
        self.thread.start()

    def _run(self) -> None:
        while not self.stopped.wait(self.interval):
            try:
                renewed = self.ledger.renew_lease(
                    self.asset, self.holder, ttl_seconds=self.ttl_seconds
                )
            except Exception:
                renewed = False
            if not renewed:
                self.lost.set()
                return

    def ensure_owned(self) -> None:
        if self.lost.is_set():
            raise RuntimeError("inventory lease lost during blocking exchange call")

    def stop(self) -> None:
        self.stopped.set()
        if self.thread is not None:
            self.thread.join(timeout=max(self.interval * 2, 1.0))


def _with_lease_heartbeat(function):
    """Renew the caller-owned lease for the full synchronous operation."""

    @wraps(function)
    def wrapped(*args, **kwargs):
        heartbeat = _LeaseHeartbeat(
            kwargs.get("ledger"), str(kwargs.get("lease_asset", "")),
            str(kwargs.get("lease_holder", "")),
            int(kwargs.get("lease_ttl_seconds", 45)),
        )
        heartbeat.start()
        try:
            result = function(*args, **kwargs)
            heartbeat.ensure_owned()
            return result
        finally:
            heartbeat.stop()

    return wrapped


def _decimal(value: Any) -> Decimal:
    return Decimal(str(value if value is not None else "0"))


def _child_client_id(prefix: str, sequence: int) -> str:
    if sequence == 0:
        return prefix[:36]
    suffix = f"-r{sequence}"
    return f"{prefix[:36-len(suffix)]}{suffix}"


@_with_lease_heartbeat
def execute_market_liquidation(
    *, exchange: Any, ledger: Any, job_id: str, pair: str, side: str,
    target_quantity: Decimal, client_order_id: str, step_size: Decimal,
    minimum_notional: Decimal, mark_price: Decimal, lease_asset: str,
    lease_holder: str, lease_ttl_seconds: int = 45, max_attempts: int = 5,
    poll_attempts: int = 3, poll_seconds: float = 1.0,
) -> dict[str, Any]:
    """Execute at most one live order at a time and journal every attempt.

    A residual child order is created only after the previous order is proven
    terminal.  On timeouts the deterministic client id is queried before any
    retry, so an accepted order cannot turn into a second economic fill.
    """
    aggregate_fills: list[dict[str, Any]] = []
    aggregate_executed = Decimal("0")
    aggregate_quote = Decimal("0")
    order_ids: list[str] = []
    attempt_audit: list[dict[str, Any]] = []

    existing_attempts = {int(row["sequence"]): row for row in ledger.attempts(job_id)}
    for sequence in range(max_attempts):
        residual = max(target_quantity - aggregate_executed, Decimal("0"))
        if residual < step_size or residual * mark_price < minimum_notional:
            break
        attempt_id = _child_client_id(client_order_id, sequence)
        attempt = existing_attempts.get(sequence) or ledger.start_attempt(
            job_id=job_id, sequence=sequence, client_order_id=attempt_id,
            requested_quantity=residual,
        )
        stored = {}
        if attempt.get("response_json"):
            try:
                stored = json.loads(attempt["response_json"])
            except (TypeError, json.JSONDecodeError):
                stored = {}
        order = exchange.order_by_client_id(pair, attempt_id)
        if order is None and stored.get("status") in TERMINAL_STATUSES:
            order = stored
        if order is None:
            if not ledger.renew_lease(
                lease_asset, lease_holder, ttl_seconds=lease_ttl_seconds
            ):
                raise RuntimeError(f"inventory lease lost before order {attempt_id}")
            try:
                order = exchange.market_order(pair, side, residual, attempt_id)
            except Exception as exc:
                # A timeout can happen after Binance accepted and filled the
                # order. Query the same client id before declaring failure.
                order = exchange.order_by_client_id(pair, attempt_id)
                if order is None:
                    ledger.finish_attempt(
                        job_id=job_id, sequence=sequence, status="UNKNOWN",
                        error=repr(exc),
                    )
                    raise

        status = str(order.get("status", "UNKNOWN")).upper()
        for _ in range(poll_attempts):
            if status in TERMINAL_STATUSES:
                break
            time.sleep(poll_seconds)
            refreshed = exchange.order_by_client_id(pair, attempt_id)
            if refreshed is not None:
                order = refreshed
                status = str(order.get("status", "UNKNOWN")).upper()
        ledger.finish_attempt(
            job_id=job_id, sequence=sequence,
            status=status if status in TERMINAL_STATUSES else "PENDING",
            response=order,
        )
        if status not in TERMINAL_STATUSES:
            raise ExecutionPending(
                f"existing market order {attempt_id} remains {status}; no residual submitted"
            )

        executed = _decimal(order.get("executedQty"))
        quote = _decimal(order.get("cummulativeQuoteQty"))
        if executed < 0 or executed > residual + step_size:
            raise RuntimeError(
                f"exchange execution {executed} exceeds requested residual {residual}"
            )
        aggregate_executed += executed
        aggregate_quote += quote
        aggregate_fills.extend(
            row for row in order.get("fills", []) if isinstance(row, dict)
        )
        if order.get("orderId") is not None:
            order_ids.append(str(order["orderId"]))
        attempt_audit.append({
            "sequence": sequence, "client_order_id": attempt_id,
            "status": status, "requested_quantity": str(residual),
            "executed_quantity": str(executed),
        })
        if executed == 0 and status != "FILLED":
            # The order is proven terminal, so a deterministic residual child
            # may safely retry the unchanged quantity.
            continue

    residual = max(target_quantity - aggregate_executed, Decimal("0"))
    if residual >= step_size and residual * mark_price >= minimum_notional:
        raise RuntimeError(
            f"market liquidation exhausted {max_attempts} attempts with residual {residual}"
        )
    return {
        "orderId": order_ids[-1] if order_ids else "",
        "orderIds": order_ids,
        "status": "FILLED" if residual < step_size else "PARTIALLY_FILLED",
        "executedQty": str(aggregate_executed),
        "cummulativeQuoteQty": str(aggregate_quote),
        "fills": aggregate_fills,
        "attempts": attempt_audit,
        "residualQty": str(residual),
    }


@_with_lease_heartbeat
def verify_market_liquidation(
    *, exchange: Any, pair: str, response: Mapping[str, Any],
    requested_quantity: Decimal, before_total: Decimal,
    step_size: Decimal, minimum_notional: Decimal, mark_price: Decimal,
    ledger: Any | None = None, lease_asset: str = "", lease_holder: str = "",
    lease_ttl_seconds: int = 45,
) -> dict[str, Any]:
    """Perform the mandatory order/trade, balance, and open-order verification."""
    executed = _decimal(response.get("executedQty"))
    attempts = list(response.get("attempts", []))
    order_verified = bool(attempts)
    def renew() -> None:
        if ledger is not None and not ledger.renew_lease(
            lease_asset, lease_holder, ttl_seconds=lease_ttl_seconds
        ):
            raise RuntimeError("inventory lease lost during post-trade verification")

    for attempt in attempts:
        renew()
        current = exchange.order_by_client_id(pair, str(attempt["client_order_id"]))
        if (
            current is None
            or str(current.get("status", "")).upper() not in TERMINAL_STATUSES
            or (
                _decimal(current.get("executedQty")) > 0
                and not current.get("fills")
            )
        ):
            order_verified = False
            break
    renew()
    balances = exchange.account_balances()
    base_asset = pair.split("-", 1)[0]
    after_total = _decimal(balances.get(base_asset, {}).get("total"))
    # Base-asset commission may make the drop slightly larger than executed;
    # a higher post-balance means the account has not reflected the fill yet.
    balance_verified = after_total <= before_total - executed + step_size
    renew()
    active_orders = exchange.open_orders(pair)
    residual = max(requested_quantity - executed, Decimal("0"))
    requested_verified = bool(
        executed <= requested_quantity + step_size
        and (residual < step_size or residual * mark_price < minimum_notional)
    )
    return {
        "order_verified": order_verified,
        "balance_verified": balance_verified,
        "no_active_orders": not active_orders,
        "requested_quantity_verified": requested_verified,
        "before_total": str(before_total),
        "after_total": str(after_total),
        "executed_quantity": str(executed),
        "remaining_requested_quantity": str(residual),
        "checked_at": time.time(),
    }

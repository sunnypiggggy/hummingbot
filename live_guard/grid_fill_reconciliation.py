"""Read-only recovery of complete flatten fills absent from a stopped Grid checkpoint.

This is deliberately not a general TradeFill replay cursor. Schema 13 only
records a fill marker for flatten/reentry orders; ordinary fills are never
inferred from timestamps or buy/sell counters here.
"""
from __future__ import annotations

from copy import deepcopy
from contextlib import closing
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import sqlite3


SCALE = Decimal("1000000")


def _decimal(value, *, positive=False):
    if value is None or isinstance(value, bool):
        raise ValueError("missing numeric evidence")
    result = Decimal(str(value))
    if not result.is_finite() or (result <= 0 if positive else result < 0):
        raise ValueError("invalid numeric evidence")
    return result


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     default=str).encode()).hexdigest()


def flatten_candidates(runtime):
    """Return only current EXITING orders for which no fill was checkpointed."""
    result = {}
    portfolio = runtime.get("portfolio_recovery") or {}
    for pair, order_id in (runtime.get("flatten_order_ids") or {}).items():
        recovery = portfolio if portfolio.get("phase") == "EXITING" else (
            (runtime.get("pair_recovery") or {}).get(pair) or {})
        if recovery.get("phase") == "EXITING" and order_id and not recovery.get("last_exit_fill"):
            result[pair] = (str(order_id), recovery)
    return result


def fold_flatten_fills(runtime, orders, fills, *, stopped):
    """Return a disposable view and receipt; never mutate a producer document."""
    original = deepcopy(runtime)
    receipt = {"status": "unchanged", "stopped_writer": stopped,
               "raw_sha256": _digest(runtime), "pairs": {}}
    candidates = flatten_candidates(runtime)
    if not candidates:
        return original, receipt
    if stopped is not True:
        receipt.update(status="unverifiable", reason="stopped Grid writer not proven")
        return original, receipt
    try:
        if runtime.get("schema_version") != 13:
            raise ValueError("unsupported flatten checkpoint schema")
        if set(runtime.get("trading_pairs", [])) != set(runtime.get("ledgers", {})):
            raise ValueError("runtime pair identity is incomplete")
        triggers = [_decimal(state.get("triggered_at"), positive=True)
                    for _, state in candidates.values()]
        trigger = min(triggers)
        for intent in (runtime.get("reentry_intents") or {}).values():
            if intent.get("state") != "COMPLETED":
                raise ValueError("reentry checkpoint is not complete")
            for key in ("submitted_at", "terminal_observed_at", "reconciled_at"):
                if _decimal(intent.get(key), positive=True) > trigger:
                    raise ValueError("reentry checkpoint overlaps flatten episode")
        wanted = {order_id for order_id, _ in candidates.values()}
        if len(wanted) != len(candidates):
            raise ValueError("flatten order identity overlaps pairs")
        # Reject concurrent ordinary/reentry fills rather than guessing whether
        # they compensated an already-booked flatten fill in the saved balance.
        if any(str(fill.get("order_id")) not in wanted for fill in fills):
            raise ValueError("other fills overlap flatten episode")
        order_map = {}
        for order in orders:
            key = str(order.get("id"))
            if key in order_map:
                raise ValueError("duplicate order evidence")
            order_map[key] = order
        view = deepcopy(runtime)
        seen = set()
        for pair, (order_id, state) in candidates.items():
            raw = runtime["ledgers"][pair]
            if raw.get("trading_pair") != pair:
                raise ValueError("runtime ledger pair identity disagrees")
            if order_id not in raw.get("open_order_ids", []):
                raise ValueError("flatten order lacks saved ownership")
            if "last_exit_fill" in state and state["last_exit_fill"] is not None:
                raise ValueError("invalid saved exit fill marker")
            episode = _decimal(state.get("triggered_at"), positive=True)
            first = _decimal(state.get("first_exit_order_at"), positive=True)
            updated = _decimal(runtime.get("updated_at"), positive=True)
            if not episode <= first <= updated:
                raise ValueError("flatten episode timestamps disagree")
            order = order_map[order_id]
            if not all(order.get(key) for key in ("market", "strategy", "config_file_path")):
                raise ValueError("order source identity is missing")
            created = _decimal(order.get("creation_timestamp"), positive=True) / 1000
            completed = _decimal(order.get("last_update_timestamp"), positive=True) / 1000
            if (order.get("symbol") != pair or order.get("order_type") != "MARKET"
                    or order.get("last_status") != "SellOrderCompleted"
                    or not episode <= first <= created <= completed):
                raise ValueError("flatten order is not a current completed market sell")
            amount = _decimal(order.get("amount"), positive=True) / SCALE
            pair_fills = [f for f in fills if str(f.get("order_id")) == order_id]
            if not pair_fills:
                raise ValueError("completed flatten has no fills")
            total = Decimal(0)
            decoded = []
            for fill in pair_fills:
                identity = (fill.get("market"), pair, str(fill.get("exchange_trade_id") or ""))
                if not identity[2] or identity in seen:
                    raise ValueError("missing or duplicate trade identity")
                seen.add(identity)
                stamp = _decimal(fill.get("timestamp"), positive=True) / 1000
                if (fill.get("symbol") != pair or fill.get("trade_type") != "SELL"
                        or fill.get("order_type") != "MARKET"
                        or any(fill.get(key) != order.get(key)
                               for key in ("market", "strategy", "config_file_path"))
                        or not int(created) <= stamp <= int(completed) + 1):
                    raise ValueError("fill is outside its flatten order")
                quantity = _decimal(fill.get("amount"), positive=True) / SCALE
                price = _decimal(fill.get("price"), positive=True) / SCALE
                fee = fill.get("trade_fee")
                fee = json.loads(fee) if isinstance(fee, str) else fee
                if not isinstance(fee, dict) or _decimal(fee.get("percent", 0)) != 0:
                    raise ValueError("unsupported flatten fee evidence")
                quote_fee = Decimal(0)
                flat_fees = fee.get("flat_fees", [])
                if not isinstance(flat_fees, list) or any(not isinstance(item, dict) for item in flat_fees):
                    raise ValueError("invalid flatten fee structure")
                for item in flat_fees:
                    if item.get("token") != pair.rsplit("-", 1)[1]:
                        raise ValueError("non-quote flatten fee requires review")
                    quote_fee += _decimal(item.get("amount"))
                reported = _decimal(fill.get("trade_fee_in_quote", 0)) / SCALE
                if abs(quote_fee - reported) >= Decimal("0.000001"):
                    raise ValueError("flatten fee evidence disagrees")
                if quote_fee >= quantity * price:
                    raise ValueError("flatten fee exhausts the fill proceeds")
                decoded.append((quantity, price, quote_fee, identity))
                total += quantity
            if total != amount:
                raise ValueError("flatten fills do not cover the completed order")
            base = _decimal(raw.get("base"))
            # Extra consistency only: absence of the schema-13 marker plus the
            # exclusion of overlapping reentry checkpoints provides the replay
            # proof. A residual/counter alone cannot establish an unbooked fill.
            if base < amount or any(quantity <= base - amount for quantity, *_ in decoded):
                raise ValueError("saved base is inconsistent with the unbooked whole flatten")
            ledger = view["ledgers"][pair]
            quote = _decimal(raw.get("quote"))
            cost = _decimal(raw.get("base_cost_quote"))
            fees = _decimal(raw.get("fees_quote"))
            sells = raw.get("sells")
            if type(sells) is not int or sells < 0:
                raise ValueError("invalid saved sell count")
            for quantity, price, fee, _ in decoded:
                remaining = base - quantity
                cost *= remaining / base
                base = remaining
                quote += price * quantity - fee
                fees += fee
                sells += 1
            ledger.update(base=str(base), quote=str(quote), base_cost_quote=str(cost),
                          fees_quote=str(fees), sells=sells)
            receipt["pairs"][pair] = {"order_sha256": _digest(order_id),
                "trade_sha256": [_digest(x[3]) for x in decoded],
                "applied_fill_count": len(decoded), "base_before": raw["base"],
                "base_after": str(base)}
        receipt.update(status="applied", view_sha256=_digest(view))
        return view, receipt
    except (ArithmeticError, ValueError, TypeError, KeyError) as exc:
        reason = "missing flatten proof" if isinstance(exc, KeyError) else str(exc)
        receipt.update(status="unverifiable", reason=reason, pairs={})
        return original, receipt


def read_grid_accounting_view(runtime, database, *, stopped, runtime_path=None):
    """Read Order and TradeFill in one RO transaction, then verify the source CAS."""
    candidates = flatten_candidates(runtime)
    if not candidates or stopped is not True:
        return fold_flatten_fills(runtime, [], [], stopped=stopped)
    try:
        trigger = min(_decimal(state.get("triggered_at"), positive=True)
                      for _, state in candidates.values())
        ids = [order_id for order_id, _ in candidates.values()]
        placeholders = ",".join("?" for _ in ids)
        with closing(sqlite3.connect(Path(database).resolve().as_uri() + "?mode=ro", uri=True, timeout=5)) as db:
            db.row_factory = sqlite3.Row
            db.execute("PRAGMA query_only=ON")
            db.execute("BEGIN")
            orders = [dict(row) for row in db.execute(
                f'SELECT * FROM "Order" WHERE id IN ({placeholders})', ids)]
            fills = [dict(row) for row in db.execute(
                "SELECT * FROM TradeFill WHERE timestamp>=? ORDER BY timestamp,rowid",
                (int(trigger) * 1000,))]
        if runtime_path is not None:
            if Path(runtime_path).resolve().parent != Path(database).resolve().parent:
                raise ValueError("Grid database is outside its checkpoint directory")
            current = json.loads(Path(runtime_path).read_text(encoding="utf-8"))
            if current != runtime:
                raise ValueError("Grid checkpoint changed during fill read")
        return fold_flatten_fills(runtime, orders, fills, stopped=stopped)
    except (OSError, sqlite3.Error, ArithmeticError, ValueError, TypeError) as exc:
        return deepcopy(runtime), {"status": "unverifiable", "stopped_writer": stopped,
                                   "reason": type(exc).__name__, "raw_sha256": _digest(runtime), "pairs": {}}

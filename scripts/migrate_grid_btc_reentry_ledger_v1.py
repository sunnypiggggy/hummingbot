#!/usr/bin/env python3
"""Repair one proven missed Grid BTC reentry fill and liquidate its owned risk.

The command is deliberately dry-run by default.  Live execution requires the
exact preflight SHA printed by a preceding run.  Every economic action is
journalled in the shared inventory SQLite ledger under a deterministic job id.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
import tempfile
import time
from decimal import Decimal, ROUND_DOWN
from pathlib import Path
from typing import Any, Mapping

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
for candidate in (REPOSITORY_ROOT, REPOSITORY_ROOT / "scripts", REPOSITORY_ROOT / "live_guard"):
    if candidate.is_dir() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

try:
    from account_inventory import (
        UnifiedInventoryLedger,
        api_key_fingerprint,
        canonical_sha256,
    )
    from dca_live_guard import BinanceEmergencyClient
    from emergency_execution import execute_market_liquidation, verify_market_liquidation
    from telegram_notifications import append_event, build_event
except ModuleNotFoundError:
    from live_guard.account_inventory import (
        UnifiedInventoryLedger,
        api_key_fingerprint,
        canonical_sha256,
    )
    from live_guard.dca_live_guard import BinanceEmergencyClient
    from live_guard.emergency_execution import (
        execute_market_liquidation,
        verify_market_liquidation,
    )
    from live_guard.telegram_notifications import append_event, build_event


PAIR = "BTC-FDUSD"
ASSET = "BTC"
OWNER = "grid:grid-live-fdusd-400"
MIGRATION_ID = "grid-btc-missed-reentry-fill-v1"
SCALE = Decimal("1000000")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False,
    ) as output:
        json.dump(dict(value), output, ensure_ascii=True, indent=2, sort_keys=True)
        temporary = output.name
    Path(temporary).replace(path)


def _fee_document(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    try:
        value = json.loads(str(raw or "{}"))
    except json.JSONDecodeError:
        value = {}
    return value if isinstance(value, dict) else {}


def _candidate_fills(database: Path, start_ms: int, end_ms: int) -> list[dict[str, Any]]:
    connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True, timeout=10)
    try:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            "SELECT timestamp,trade_type,order_type,price,amount,"
            "trade_fee_in_quote,trade_fee,order_id,exchange_trade_id "
            "FROM TradeFill WHERE symbol=? AND trade_type='BUY' "
            "AND order_type='MARKET' AND timestamp BETWEEN ? AND ? "
            "ORDER BY timestamp,rowid",
            (PAIR, start_ms, end_ms),
        ).fetchall()
    finally:
        connection.close()
    result = []
    for row in rows:
        price = Decimal(row["price"]) / SCALE
        amount = Decimal(row["amount"]) / SCALE
        fee_quote = Decimal(row["trade_fee_in_quote"] or 0) / SCALE
        fee_doc = _fee_document(row["trade_fee"])
        base_fee = sum(
            (
                Decimal(str(item.get("amount", "0")))
                for item in fee_doc.get("flat_fees", [])
                if str(item.get("token", "")).upper() == ASSET
            ),
            Decimal("0"),
        )
        # SQLite's trade_fee_in_quote column is fixed to six decimals, while
        # the base-asset commission in trade_fee retains the exact amount.
        # Preserve the more precise economic fee for the Runtime ledger.
        fee_quote = max(fee_quote, base_fee * price)
        result.append({
            "timestamp": int(row["timestamp"]),
            "order_id": str(row["order_id"]),
            "exchange_trade_id": str(row["exchange_trade_id"]),
            "price": str(price), "amount": str(amount),
            "notional": str(price * amount),
            "fee_quote": str(fee_quote), "base_fee": str(base_fee),
            "net_base": str(amount - base_fee),
        })
    return result


def _market_rules(exchange: BinanceEmergencyClient) -> tuple[Decimal, Decimal, Decimal]:
    response = exchange.session.get(
        f"{exchange.base_url}/api/v3/exchangeInfo",
        params={"symbol": exchange.symbol(PAIR)}, timeout=10,
    )
    response.raise_for_status()
    symbols = response.json().get("symbols", [])
    if len(symbols) != 1:
        raise RuntimeError("BTC-FDUSD exchange filters unavailable")
    filters = {row["filterType"]: row for row in symbols[0].get("filters", [])}
    step = Decimal("0")
    for lot in (filters.get("MARKET_LOT_SIZE") or {}, filters.get("LOT_SIZE") or {}):
        candidate = Decimal(str(lot.get("stepSize", "0")))
        if candidate > 0:
            step = candidate
            break
    minimum = Decimal(str((filters.get("NOTIONAL") or filters.get("MIN_NOTIONAL") or {}).get(
        "minNotional", "0",
    )))
    ticker = exchange.session.get(
        f"{exchange.base_url}/api/v3/ticker/price",
        params={"symbol": exchange.symbol(PAIR)}, timeout=10,
    )
    ticker.raise_for_status()
    mark = Decimal(str(ticker.json().get("price", "0")))
    if step <= 0 or minimum <= 0 or mark <= 0:
        raise RuntimeError("BTC-FDUSD exchange filters or mark are invalid")
    return step, minimum, mark


def _fill_metrics(response: Mapping[str, Any]) -> dict[str, Any]:
    executed = Decimal(str(response.get("executedQty", "0")))
    quote = Decimal(str(response.get("cummulativeQuoteQty", "0")))
    if executed <= 0 or quote <= 0:
        raise RuntimeError("migration SELL has no confirmed execution")
    quote_fee = Decimal("0")
    base_fee = Decimal("0")
    fee_details = []
    for fill in response.get("fills", []):
        asset = str(fill.get("commissionAsset", "")).upper()
        commission = Decimal(str(fill.get("commission", "0")))
        price = Decimal(str(fill.get("price", "0")))
        if asset == "BNB":
            raise RuntimeError("BNB commission is forbidden for Grid/DCA accounting")
        if commission and asset not in {"FDUSD", "BTC"}:
            raise RuntimeError(f"unsupported commission asset {asset}")
        if asset == "FDUSD":
            quote_fee += commission
        elif asset == "BTC":
            base_fee += commission
        fee_details.append({
            "asset": asset, "commission": str(commission), "fill_price": str(price),
        })
    return {
        "executed": executed, "quote": quote,
        "average_price": quote / executed,
        "quote_fee": quote_fee, "base_fee": base_fee,
        "fee_quote": quote_fee + base_fee * (quote / executed),
        "fee_details": fee_details,
    }


def _apply_missing_buy(ledger: dict[str, Any], fill: Mapping[str, Any]) -> None:
    price = Decimal(str(fill["price"]))
    gross = Decimal(str(fill["amount"]))
    base_fee = Decimal(str(fill["base_fee"]))
    fee_quote = Decimal(str(fill["fee_quote"]))
    notional = price * gross
    ledger["quote"] = str(Decimal(str(ledger["quote"])) - notional)
    ledger["base"] = str(Decimal(str(ledger["base"])) + gross - base_fee)
    ledger["base_cost_quote"] = str(
        Decimal(str(ledger.get("base_cost_quote", "0"))) + notional
    )
    ledger["fees_quote"] = str(
        Decimal(str(ledger.get("fees_quote", "0"))) + fee_quote
    )
    ledger["buys"] = int(ledger.get("buys", 0)) + 1


def _apply_sell(ledger: dict[str, Any], metrics: Mapping[str, Any]) -> None:
    before_base = Decimal(str(ledger["base"]))
    executed = Decimal(str(metrics["executed"]))
    base_fee = Decimal(str(metrics["base_fee"]))
    if executed + base_fee > before_base:
        raise RuntimeError("confirmed SELL exceeds corrected Grid BTC ownership")
    remaining = before_base - executed - base_fee
    old_cost = Decimal(str(ledger.get("base_cost_quote", "0")))
    ledger["base_cost_quote"] = str(
        old_cost * remaining / before_base if before_base > 0 else Decimal("0")
    )
    ledger["base"] = str(remaining)
    ledger["quote"] = str(
        Decimal(str(ledger["quote"]))
        + Decimal(str(metrics["quote"]))
        - Decimal(str(metrics["quote_fee"]))
    )
    ledger["fees_quote"] = str(
        Decimal(str(ledger.get("fees_quote", "0")))
        + Decimal(str(metrics["fee_quote"]))
    )
    ledger["sells"] = int(ledger.get("sells", 0)) + 1


def _completed_job_response(
    shared: UnifiedInventoryLedger, job: Mapping[str, Any],
) -> dict[str, Any]:
    """Rebuild immutable fill evidence for crash-safe Runtime finalization."""
    fills: list[dict[str, Any]] = []
    for attempt in shared.attempts(str(job["job_id"])):
        try:
            response = json.loads(str(attempt.get("response_json") or "{}"))
        except json.JSONDecodeError:
            continue
        if isinstance(response, dict):
            fills.extend(response.get("fills", []))
    if not fills:
        try:
            details = json.loads(str(job.get("fee_details") or "[]"))
        except json.JSONDecodeError:
            details = []
        fills = [
            {
                "commissionAsset": row.get("asset", ""),
                "commission": row.get("commission", "0"),
                "price": row.get("fill_price", "0"),
            }
            for row in details if isinstance(row, dict)
        ]
    return {
        "executedQty": job.get("executed_quantity", "0"),
        "cummulativeQuoteQty": job.get("quote_quantity", "0"),
        "fills": fills,
    }


def _finalize_runtime_from_completed_job(
    runtime_path: Path, runtime: dict[str, Any], job: Mapping[str, Any],
    shared: UnifiedInventoryLedger,
) -> dict[str, Any]:
    if not shared.completed_job_verified(job):
        raise RuntimeError("existing completed migration job lacks verification")
    events = list(runtime.get("accounting_migrations", []))
    migration = next(
        (row for row in events if row.get("migration_id") == MIGRATION_ID), None,
    )
    if migration is None:
        raise RuntimeError("completed liquidation exists without Runtime correction evidence")
    metrics = _fill_metrics(_completed_job_response(shared, job))
    if migration.get("stage") != "COMPLETED":
        ledger = dict(runtime["ledgers"][PAIR])
        _apply_sell(ledger, metrics)
        migration.update({
            "stage": "COMPLETED", "completed_at": time.time(),
            "job_id": str(job["job_id"]),
            "client_order_id": str(job.get("client_order_id") or ""),
            "exchange_order_id": job.get("exchange_order_id"),
            "executed_quantity": str(metrics["executed"]),
            "quote_quantity": str(metrics["quote"]),
            "fee_quote": str(metrics["fee_quote"]),
            "remaining_grid_base": ledger["base"],
        })
        runtime["ledgers"][PAIR] = ledger
        runtime["accounting_migrations"] = events
        runtime["updated_at"] = time.time()
        _atomic_json(runtime_path, runtime)
    return metrics


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--inventory-status", type=Path, required=True)
    parser.add_argument("--inventory-dir", type=Path, required=True)
    parser.add_argument("--secret", type=Path, required=True)
    parser.add_argument("--start-ms", type=int, default=1788206300000)
    parser.add_argument("--end-ms", type=int, default=1788206500000)
    parser.add_argument("--missing-order-id", default="")
    parser.add_argument("--missing-trade-id", default="")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirmation", default="")
    parser.add_argument("--telegram-events", type=Path)
    return parser.parse_args()


def main() -> int:
    args = _args()
    runtime = _read_json(args.runtime)
    inventory = _read_json(args.inventory_status)
    candidates = _candidate_fills(args.database, args.start_ms, args.end_ms)
    selected = [
        row for row in candidates
        if (not args.missing_order_id or row["order_id"] == args.missing_order_id)
        and (not args.missing_trade_id or row["exchange_trade_id"] == args.missing_trade_id)
    ]
    if not args.missing_order_id or not args.missing_trade_id:
        print(json.dumps({"mode": "DISCOVERY", "candidates": candidates}, indent=2))
        return 0
    if len(selected) != 1:
        raise RuntimeError("exactly one hash-bound missing fill must match")
    fill = selected[0]
    exchange = BinanceEmergencyClient.from_secret_file(args.secret)
    fee_policy = exchange.verify_ready([PAIR, "BTC-USDT"])
    balances = exchange.account_balances()
    open_orders = {
        PAIR: exchange.open_orders(PAIR),
        "BTC-USDT": exchange.open_orders("BTC-USDT"),
    }
    step, minimum, mark = _market_rules(exchange)
    ledger_before = dict(runtime.get("ledgers", {}).get(PAIR, {}))
    if not ledger_before or int(runtime.get("schema_version", 0)) < 12:
        raise RuntimeError("Grid Runtime PairLedger is unavailable")
    migration_events = list(runtime.get("accounting_migrations", []))
    already_applied = next(
        (row for row in migration_events if row.get("migration_id") == MIGRATION_ID), None,
    )
    corrected = dict(ledger_before)
    if already_applied is None:
        _apply_missing_buy(corrected, fill)
    shared = UnifiedInventoryLedger(args.inventory_dir)
    shared.bind_account(api_key_fingerprint(exchange.api_key))
    job_id = hashlib.sha256(f"{MIGRATION_ID}:{fill['exchange_trade_id']}".encode()).hexdigest()
    client_id = f"inv-{job_id[:24]}"
    existing_job = shared.get_job(job_id)
    if existing_job is not None and str(existing_job.get("status")) == "COMPLETED":
        metrics = _finalize_runtime_from_completed_job(
            args.runtime, runtime, existing_job, shared,
        )
        print(json.dumps({
            "status": "COMPLETED_RECOVERED", "migration_id": MIGRATION_ID,
            "job_id": job_id, "executed_quantity": str(metrics["executed"]),
            "quote_quantity": str(metrics["quote"]),
            "fee_quote": str(metrics["fee_quote"]),
            "remaining_grid_base": _read_json(args.runtime)["ledgers"][PAIR]["base"],
        }, indent=2))
        return 0
    current_owner = Decimal(str(
        inventory.get("assets", {}).get(ASSET, {}).get("owners", {}).get(OWNER, "0")
    ))
    corrected_owner = Decimal(str(corrected["base"]))
    dca_owner = Decimal(str(
        inventory.get("assets", {}).get(ASSET, {}).get("owners", {}).get(
            "dca:dca-live-btcusdt-200", "0",
        )
    ))
    unattributed = Decimal(str(
        inventory.get("assets", {}).get(ASSET, {}).get("unattributed", "0")
    ))
    if not inventory.get("sources_healthy"):
        raise RuntimeError("unified inventory sources are unhealthy")
    if Decimal(str(inventory["assets"][ASSET].get("ownership_deficit", "0"))) > 0:
        raise RuntimeError("unified inventory has an ownership deficit")
    if any(open_orders.values()):
        raise RuntimeError("BTC orders are active; migration cannot proceed")
    if corrected_owner != current_owner:
        raise RuntimeError(
            f"corrected Runtime owner {corrected_owner} != independently derived owner {current_owner}"
        )
    actual_total = Decimal(str(balances.get(ASSET, {}).get("total", "0")))
    if corrected_owner + dca_owner + unattributed > actual_total + step:
        raise RuntimeError("corrected ownership exceeds Binance BTC balance")
    sell_quantity = (corrected_owner / step).to_integral_value(rounding=ROUND_DOWN) * step
    if sell_quantity * mark < minimum:
        sell_quantity = Decimal("0")
    # Keep the approval digest stable between a dry-run and execution.  The
    # ticker price, generated status hashes and observation timestamps are
    # intentionally excluded because they change without changing the
    # approved economic scope.  They are still printed and, more importantly,
    # revalidated immediately before the order is submitted.
    approval_scope = {
        "migration_id": MIGRATION_ID,
        "missing_fill": fill,
        "ledger_before": ledger_before,
        "ledger_after_correction": corrected,
        "grid_owner": str(current_owner), "dca_owner_unchanged": str(dca_owner),
        "unattributed_unchanged": str(unattributed),
        "exchange_total": str(actual_total), "exchange_free": str(
            balances.get(ASSET, {}).get("free", "0")
        ),
        "step_size": str(step), "minimum_notional": str(minimum),
        "sell_quantity": str(sell_quantity), "fee_policy": fee_policy,
        "already_applied": bool(already_applied),
    }
    preflight_sha = canonical_sha256(approval_scope)
    preflight = {
        "approval_scope": approval_scope,
        "runtime_sha256": canonical_sha256(runtime),
        "inventory_evidence_sha256": inventory.get("evidence_sha256"),
        "mark_price": str(mark),
        "estimated_notional": str(sell_quantity * mark),
        "open_orders": {key: len(value) for key, value in open_orders.items()},
    }
    print(json.dumps({"mode": "EXECUTE" if args.execute else "DRY_RUN",
                      "preflight_sha256": preflight_sha, **preflight}, indent=2))
    if not args.execute:
        return 0
    if args.confirmation != preflight_sha:
        raise RuntimeError("--confirmation must equal the current preflight_sha256")
    if sell_quantity <= 0:
        raise RuntimeError("corrected Grid BTC ownership is already dust")

    if args.telegram_events:
        append_event(args.telegram_events, build_event(
            source="grid-ledger-migration", strategy="grid",
            bot="grid-live-fdusd-400", pair=PAIR,
            mechanism="account_inventory",
            transition="INVENTORY_LIQUIDATION_STARTED",
            reason="补记已确认的BTC重入成交后，退出Grid可成交归属库存",
            severity="warning", action="hash_bound_grid_owned_market_exit",
            correlation_id=f"{MIGRATION_ID}:started",
            details={
                "ownership_scope": OWNER, "quantity": str(sell_quantity),
                "estimated_notional": str(sell_quantity * mark),
                "minimum_notional": str(minimum),
                "preflight_sha256": preflight_sha,
                "dca_owner_unchanged": str(dca_owner),
                "unattributed_unchanged": str(unattributed),
            },
        ))

    if already_applied is None:
        migration_events.append({
            "migration_id": MIGRATION_ID, "stage": "LEDGER_CORRECTED",
            "applied_at": time.time(), "fill": fill,
            "preflight_sha256": preflight_sha,
        })
        runtime["schema_version"] = 13
        runtime["ledgers"][PAIR] = corrected
        runtime["accounting_migrations"] = migration_events
        runtime["updated_at"] = time.time()
        _atomic_json(args.runtime, runtime)

    holder = f"migration:{MIGRATION_ID}"
    with shared.lease(ASSET, holder, ttl_seconds=45):
        shared.assert_exit_allowed(
            asset=ASSET, exchange_total=actual_total, owner_key=OWNER,
            requested_quantity=sell_quantity, tolerance=step,
        )
        job = shared.start_job(
            job_id=job_id, asset=ASSET, scope=OWNER, pair=PAIR,
            requested_quantity=sell_quantity, client_order_id=client_id,
        )
        if job.get("status") == "COMPLETED":
            if not shared.completed_job_verified(job):
                raise RuntimeError("existing completed migration job lacks verification")
            response = {
                "orderId": job.get("exchange_order_id"), "status": "FILLED",
                "executedQty": job.get("executed_quantity"),
                "cummulativeQuoteQty": job.get("quote_quantity"),
                "fills": json.loads(str(job.get("fee_details") or "[]")),
                "attempts": shared.attempts(job_id),
            }
        else:
            response = execute_market_liquidation(
                exchange=exchange, ledger=shared, job_id=job_id,
                pair=PAIR, side="SELL", target_quantity=sell_quantity,
                client_order_id=client_id, step_size=step,
                minimum_notional=minimum, mark_price=mark,
                lease_asset=ASSET, lease_holder=holder,
            )
            verification = verify_market_liquidation(
                exchange=exchange, pair=PAIR, response=response,
                requested_quantity=sell_quantity, before_total=actual_total,
                step_size=step, minimum_notional=minimum, mark_price=mark,
                ledger=shared, lease_asset=ASSET, lease_holder=holder,
            )
            metrics = _fill_metrics(response)
            shared.finish_job(
                job_id, status="COMPLETED",
                exchange_order_id=str(response.get("orderId", "")),
                executed_quantity=response.get("executedQty", "0"),
                quote_quantity=response.get("cummulativeQuoteQty", "0"),
                fee_quote=metrics["fee_quote"], fee_details=metrics["fee_details"],
                verification=verification,
            )

    # Re-query the deterministic job so a rerun and the first run use the same
    # confirmed fee evidence.
    job = shared.start_job(
        job_id=job_id, asset=ASSET, scope=OWNER, pair=PAIR,
        requested_quantity=sell_quantity, client_order_id=client_id,
    )
    runtime = _read_json(args.runtime)
    metrics = _finalize_runtime_from_completed_job(args.runtime, runtime, job, shared)
    if args.telegram_events:
        append_event(args.telegram_events, build_event(
            source="grid-ledger-migration", strategy="grid",
            bot="grid-live-fdusd-400", pair=PAIR,
            mechanism="account_inventory",
            transition="INVENTORY_LIQUIDATION_COMPLETED",
            reason="Grid BTC重入账本已修正，可成交归属库存已卖回FDUSD",
            severity="info", action="ledger_reconciled_and_grid_inventory_flattened",
            correlation_id=f"{MIGRATION_ID}:completed",
            details={
                "ownership_scope": OWNER,
                "quantity": str(metrics["executed"]),
                "quote_quantity": str(metrics["quote"]),
                "fee_quote": str(metrics["fee_quote"]),
                "fee_details": metrics["fee_details"],
                "order_id": str(job.get("exchange_order_id") or ""),
                "remaining_grid_base": _read_json(args.runtime)["ledgers"][PAIR]["base"],
                "preflight_sha256": preflight_sha,
            },
        ))
    print(json.dumps({
        "status": "COMPLETED", "migration_id": MIGRATION_ID,
        "job_id": job_id, "executed_quantity": str(metrics["executed"]),
        "quote_quantity": str(metrics["quote"]),
        "fee_quote": str(metrics["fee_quote"]),
        "remaining_grid_base": _read_json(args.runtime)["ledgers"][PAIR]["base"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import asyncio
import json
import os
import time
from decimal import Decimal

from stocks_runtime.ledger import PostgresManagedLedger
from stocks_runtime.paper_broker import PostgresPaperBroker
from stocks_runtime.settings import dedicated_database_url


SCHEMA = os.getenv("BINANCE_STOCKS_SCENARIO_SCHEMA", "paper_scenario_restart_fifo_v2")
LOCK_ID = int(os.getenv("BINANCE_STOCKS_SCENARIO_LOCK_ID", str(0x484253560222)))


def database_url() -> str:
    return dedicated_database_url(
        os.environ["DATABASE_URL"],
        os.getenv("BINANCE_STOCKS_DATABASE_NAME", "hummingbot_stocks_paper_test"),
    )


async def new_runtime() -> tuple[PostgresManagedLedger, PostgresPaperBroker]:
    ledger = PostgresManagedLedger(database_url(), schema=SCHEMA, leader_lock_id=LOCK_ID)
    await ledger.initialize()
    broker = PostgresPaperBroker(
        ledger,
        latency_ms=0,
        market_timeout_seconds=0.2,
        quote_max_age_seconds=60,
    )
    await broker.initialize()
    broker.update_market_state(
        "MARKET_OPEN",
        {"AAPL": "TRADING"},
        {"AAPL": "BOTH"},
        "2026-08-21",
    )
    return ledger, broker


async def reserve_buy(
    ledger: PostgresManagedLedger,
    executor_id: str,
    client_order_id: str,
    amount: Decimal,
    order_type: str = "LIMIT",
) -> None:
    await ledger.reserve_intent(
        executor_id=executor_id,
        executor_type="order_executor",
        symbol="AAPL",
        side="BUY",
        requested_base=amount,
        estimated_notional=amount * Decimal("101"),
        fee_reserve=Decimal("0.35"),
        config={"id": executor_id},
    )
    await ledger.register_order(
        client_order_id=client_order_id,
        executor_id=executor_id,
        symbol="AAPL",
        side="BUY",
        requested_base=amount,
        order_type=order_type,
    )


async def reset_scenario_schema() -> None:
    if not SCHEMA.startswith("paper_scenario_"):
        raise ValueError("scenario schema must start with paper_scenario_")
    import asyncpg

    connection = await asyncpg.connect(database_url().replace("postgresql+asyncpg://", "postgresql://", 1))
    try:
        await connection.execute(f"DROP SCHEMA IF EXISTS {SCHEMA} CASCADE")
    finally:
        await connection.close()


async def main() -> None:
    await reset_scenario_schema()
    base_event_ms = int(time.time() * 1000)
    ledger, broker = await new_runtime()
    initial_quote = {
        "symbol": "AAPL", "bid": "100", "ask": "101",
        "bidQty": "1", "askQty": "1", "eventTime": base_event_ms,
    }
    await broker.process_quote(initial_quote)
    await reserve_buy(ledger, "fifo-1", "x-HBSTK-FIFO1", Decimal("0.4"))
    await reserve_buy(ledger, "fifo-2", "x-HBSTK-FIFO2", Decimal("0.4"))
    for executor_id, client_order_id in (
        ("fifo-1", "x-HBSTK-FIFO1"),
        ("fifo-2", "x-HBSTK-FIFO2"),
    ):
        await broker.create_order(
            client_order_id=client_order_id,
            executor_id=executor_id,
            symbol="AAPL",
            side="BUY",
            order_type="LIMIT",
            amount=Decimal("0.4"),
            limit_price=Decimal("101"),
            trading_date="2026-08-21",
        )

    first_fill_quote = {
        "symbol": "AAPL", "bid": "100", "ask": "101",
        "bidQty": "1", "askQty": "0.5", "eventTime": base_event_ms + 1,
    }
    await broker.process_quote(first_fill_quote)
    await broker.reconcile_managed_fills()
    first = await broker.order("x-HBSTK-FIFO1")
    second = await broker.order("x-HBSTK-FIFO2")
    assert Decimal(first["filled_base"]) == Decimal("0.4")
    assert Decimal(second["filled_base"]) == Decimal("0.1")

    # Replaying an identical BBO must not consume the same displayed size twice.
    await broker.process_quote(first_fill_quote)
    second_after_duplicate = await broker.order("x-HBSTK-FIFO2")
    assert Decimal(second_after_duplicate["filled_base"]) == Decimal("0.1")

    run_id = broker.run_id
    await broker.save_checkpoint(
        "fifo-2", {"id": "fifo-2"}, {"kind": "restart"}, {"filled": "0.1"}, "RUNNING"
    )
    await ledger.close()

    # A fresh process must recover the same run, partial order, inventory, and checkpoint.
    recovered_ledger, recovered_broker = await new_runtime()
    assert recovered_broker.run_id == run_id
    checkpoints = await recovered_broker.active_checkpoints()
    assert len(checkpoints) == 1 and checkpoints[0]["executor_id"] == "fifo-2"
    positions = await recovered_ledger.managed_position_rows()
    assert sum(Decimal(row["total_base"]) for row in positions) == Decimal("0.5")

    await recovered_broker.process_quote({
        "symbol": "AAPL", "bid": "100", "ask": "101",
        "bidQty": "1", "askQty": "0.3", "eventTime": base_event_ms + 2,
    })
    await recovered_broker.reconcile_managed_fills()
    completed = await recovered_broker.order("x-HBSTK-FIFO2")
    assert completed["status"] == "FILLED"
    assert Decimal(completed["filled_base"]) == Decimal("0.4")

    # Duplicate/stale liquidity may advance timeout, but must never fabricate a fill.
    await reserve_buy(
        recovered_ledger, "timeout-1", "x-HBSTK-TIMEOUT1", Decimal("0.1"), "MARKET"
    )
    await recovered_broker.create_order(
        client_order_id="x-HBSTK-TIMEOUT1",
        executor_id="timeout-1",
        symbol="AAPL",
        side="BUY",
        order_type="MARKET",
        amount=Decimal("0.1"),
        limit_price=None,
        trading_date="2026-08-21",
    )
    await asyncio.sleep(0.25)
    await recovered_broker.process_quote({
        "symbol": "AAPL", "bid": "100", "ask": "101",
        "bidQty": "1", "askQty": "0.3", "eventTime": base_event_ms + 2,
    })
    timed_out = await recovered_broker.order("x-HBSTK-TIMEOUT1")
    assert timed_out["status"] == "CANCELED"
    assert Decimal(timed_out["filled_base"]) == 0

    account = await recovered_broker.account()
    trades = await recovered_broker.trades()
    assert len(trades) == 3
    assert Decimal(account["positions"][0]["total"]) == Decimal("0.8")
    try:
        await recovered_broker.reset("RESET PAPER ACCOUNT TO 2000 USDC", 0)
    except (ValueError, RuntimeError):
        pass
    else:
        raise AssertionError("reset unexpectedly succeeded while a position remained")
    await recovered_ledger.close()

    print(json.dumps({
        "scenario": "restart_fifo_timeout",
        "run_id_stable": True,
        "fills": len(trades),
        "position_base": account["positions"][0]["total"],
        "net_pnl": account["net_pnl"],
        "timeout_status": timed_out["status"],
        "economic_requests": 0,
    }, default=str, sort_keys=True))


if __name__ == "__main__":
    asyncio.run(main())

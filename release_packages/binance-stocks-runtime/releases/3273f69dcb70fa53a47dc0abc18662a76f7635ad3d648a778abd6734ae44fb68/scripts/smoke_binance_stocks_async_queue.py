from __future__ import annotations

import asyncio
import os
from datetime import datetime, timedelta, timezone
from decimal import Decimal

from stocks_runtime.ledger import LedgerLimits, PostgresManagedLedger
from stocks_runtime.settings import dedicated_database_url


async def main() -> None:
    database_name = os.environ.get("BINANCE_STOCKS_ASYNC_TEST_DATABASE", "")
    if not database_name.startswith("hummingbot_stocks_async_test_"):
        raise RuntimeError("refusing to run async queue smoke against a non-test database")
    database_url = dedicated_database_url(os.environ["DATABASE_URL"], database_name)
    ledger = PostgresManagedLedger(
        database_url,
        LedgerLimits(),
        schema="binance_stocks_paper",
        leader_lock_id=0x4842534153594E43,  # "HBSASYNC"
    )
    await ledger.initialize()
    try:
        ledger.set_quote_balances(Decimal("2000"), Decimal("2000"))
        await ledger.ensure_whitelist({"AAPL"})
        config = {
            "id": "smoke-async-aapl-0001", "type": "order_executor",
            "connector_name": "binance_stocks", "trading_pair": "AAPL-USDC",
            "side": "BUY", "amount": "0.5", "execution_strategy": "MARKET",
        }
        result = await ledger.reserve_intent(
            executor_id=config["id"], executor_type=config["type"], symbol="AAPL", side="BUY",
            requested_base=Decimal("0.5"), estimated_notional=Decimal("100"),
            fee_reserve=Decimal("0.35"), config=config,
            schedule={
                "schedule_id": "sch-smoke-aapl-0001", "request_type": config["type"],
                "request_payload": {"symbol": "AAPL", "activation_policy": "QUEUE_IF_CLOSED"},
                "target_session": "MARKET_OPEN", "amount_basis": "QUOTE_BUDGET",
                "quote_budget": "100", "frozen_price": None,
                "hard_expires_at": datetime.now(timezone.utc) + timedelta(days=7),
            },
        )
        assert result["status"] == "QUEUED"
        row = await ledger.scheduled_record("sch-smoke-aapl-0001")
        assert row and row["status"] == "QUEUED" and Decimal(row["quote_budget"]) == Decimal("100")
        await ledger.transition_schedule(
            row["schedule_id"], "WAITING_SESSION", expected={"QUEUED"},
            reason="market_closed", next_attempt_seconds=1,
        )
        active = await ledger.scheduled_rows(active_only=True)
        assert len(active) == 1 and active[0]["status"] == "WAITING_SESSION"
        canceled = await ledger.terminalize_schedule(row["schedule_id"], "CANCELED", "smoke_cancel")
        assert canceled and canceled["status"] == "CANCELED"
        assert not await ledger.scheduled_rows(active_only=True)
        print("ASYNC_QUEUE_POSTGRES_SMOKE_PASS")
    finally:
        await ledger.close()


if __name__ == "__main__":
    asyncio.run(main())

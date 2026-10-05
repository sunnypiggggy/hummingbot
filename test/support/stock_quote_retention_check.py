"""Real PostgreSQL rehearsal. Refuses any database except the dedicated test DB."""
import asyncio
import json
import os
from datetime import datetime, timedelta, timezone
from decimal import Decimal

from stocks_runtime.paper_broker import PaperQuote, PostgresPaperBroker
from stocks_runtime.quote_store import CompactQuoteStore
from stocks_runtime.quote_maintenance import migrate_batch, verify


async def check():
    import asyncpg
    from stocks_runtime.settings import dedicated_database_url
    url=dedicated_database_url(os.environ["DATABASE_URL"],"hummingbot_stocks_retention_test").replace("postgresql+asyncpg://","postgresql://")
    db=await asyncpg.connect(url)
    assert await db.fetchval("SELECT current_database()") == "hummingbot_stocks_retention_test"
    await db.execute("DROP SCHEMA IF EXISTS binance_stocks_paper CASCADE")
    await db.execute("CREATE SCHEMA IF NOT EXISTS binance_stocks_paper")
    s="binance_stocks_paper"
    await db.execute(f"""
      CREATE TABLE IF NOT EXISTS {s}.paper_quote_events (
        run_id TEXT,event_id TEXT,symbol TEXT,bid NUMERIC,ask NUMERIC,bid_size NUMERIC,ask_size NUMERIC,
        event_time TIMESTAMPTZ,processed_at TIMESTAMPTZ DEFAULT now(),PRIMARY KEY(run_id,event_id));
      CREATE TABLE IF NOT EXISTS {s}.paper_trades (run_id TEXT,quote_event_id TEXT);
      CREATE TABLE IF NOT EXISTS {s}.paper_equity_snapshots(snapshot_at TIMESTAMPTZ);
    """)
    store=CompactQuoteStore(s)
    await store.initialize(db)
    await db.execute(f"TRUNCATE {s}.paper_quote_events,{s}.paper_trades,{s}.paper_quote_latest,"
                     f"{s}.paper_quote_summaries,{s}.paper_fill_quote_evidence,{s}.paper_quote_migration,{s}.paper_equity_snapshots")
    now=datetime.now(timezone.utc)
    def quote(at,ask="101"):
        return PaperQuote.from_payload(dict(symbol="AAPL",bid="100",ask=ask,bidQty="1",askQty="0.5",eventTime=at.timestamp()))
    old=quote(now-timedelta(days=40))
    for q in (old,quote(now-timedelta(days=2)),quote(now-timedelta(days=371))):
        await db.execute(f"INSERT INTO {s}.paper_quote_events VALUES($1,$2,$3,$4,$5,$6,$7,$8,now())",
                         "run",q.event_id,q.symbol,q.bid,q.ask,q.bid_size,q.ask_size,
                         datetime.fromtimestamp(q.event_time,timezone.utc))
    await db.execute(f"INSERT INTO {s}.paper_trades VALUES('run',$1)",old.event_id)
    await store.seed_legacy(db,"run")
    assert not await store.accept(db,"run",old)
    fresh=quote(now)
    async with db.transaction():
        assert await store.accept(db,"run",fresh)
        await store.protect_fill(db,"run",fresh)
    assert not await store.accept(db,"run",fresh)
    restarted=CompactQuoteStore(s)
    assert not await restarted.accept(db,"run",fresh)
    assert await restarted.accept(db,"run",quote(now,"102"))
    assert not await restarted.accept(db,"run",quote(now-timedelta(seconds=1)))
    # Crash/rollback does not advance watermark or add summary samples.
    future=quote(now+timedelta(seconds=1))
    try:
        async with db.transaction():
            assert await store.accept(db,"run",future)
            raise RuntimeError("process killed before commit")
    except RuntimeError:
        pass
    assert await store.accept(db,"run",future)
    # Concurrent readers compete on the persistent row, only one consumes liquidity.
    c1=await asyncpg.connect(url)
    c2=await asyncpg.connect(url)
    simultaneous=quote(now+timedelta(seconds=2))
    async def accept(c):
        async with c.transaction():
            return await store.accept(c,"run",simultaneous)
    assert sum(await asyncio.gather(accept(c1),accept(c2))) == 1
    await c1.close(); await c2.close()
    while not (await migrate_batch(db,limit=1))["completed"]:
        pass
    receipt=await verify(db)
    assert receipt["safe_to_reclaim"],receipt
    assert await db.fetchval(f"SELECT count(*) FROM {s}.paper_fill_quote_evidence WHERE event_id=$1",old.event_id) == 1
    assert await db.fetchval(f"SELECT count(*) FROM {s}.paper_quote_summaries WHERE seconds=3600") == 1
    await db.execute(f"TRUNCATE {s}.paper_quote_events")
    assert not await store.accept(db,"run",fresh)
    value=await store.maintenance(db)
    assert value["writes_blocked"] is False
    # Full broker economics, not only storage SQL. This schema belongs exclusively
    # to the allowlisted rehearsal DB; production cannot reach this reset.
    await db.execute("DROP SCHEMA binance_stocks_paper CASCADE")
    await db.close()
    from stocks_runtime.ledger import PostgresManagedLedger
    ledger=PostgresManagedLedger(url,schema=s,leader_lock_id=123456789)
    await ledger.initialize()
    await ledger.ensure_whitelist({"AAPL"})
    broker=PostgresPaperBroker(ledger,latency_ms=0)
    await broker.initialize()
    broker.update_market_state("MARKET_OPEN",{"AAPL":"TRADING"},{"AAPL":"BOTH"})
    await ledger.reserve_intent(executor_id="test-executor",executor_type="order_executor",
                               symbol="AAPL",side="BUY",requested_base=Decimal("0.6"),
                               estimated_notional=Decimal("60.6"),config={"id":"test-executor"})
    order_id="x-HBSTK-rehearsal"
    await ledger.register_order(client_order_id=order_id,executor_id="test-executor",symbol="AAPL",
                                side="BUY",requested_base=Decimal("0.6"),order_type="LIMIT")
    await broker.create_order(client_order_id=order_id,executor_id="test-executor",symbol="AAPL",
                              side="BUY",order_type="LIMIT",amount=Decimal("0.6"),
                              limit_price=Decimal("101"),trading_date=None)
    def payload(q):
        return dict(symbol=q.symbol,bid=str(q.bid),ask=str(q.ask),bidQty=str(q.bid_size),askQty=str(q.ask_size),eventTime=q.event_time)
    q=quote(datetime.now(timezone.utc))
    assert (await broker.process_quote(payload(q))) == [order_id]
    assert (await broker.order(order_id))["filled_base"] == Decimal("0.5")
    cash=(await broker.account())["cash_balance"]
    assert (await broker.process_quote(payload(q))) == []
    assert (await broker.account())["cash_balance"] == cash
    restarted_broker=PostgresPaperBroker(ledger,latency_ms=0)
    await restarted_broker.initialize()
    restarted_broker.update_market_state("MARKET_OPEN",{"AAPL":"TRADING"},{"AAPL":"BOTH"})
    assert restarted_broker.run_id == broker.run_id
    assert (await restarted_broker.process_quote(payload(q))) == []
    q2=quote(datetime.now(timezone.utc)+timedelta(seconds=1))
    assert (await restarted_broker.process_quote(payload(q2))) == [order_id]
    order=await restarted_broker.order(order_id)
    assert order["status"] == "FILLED" and order["filled_base"] == Decimal("0.6")
    assert order["cumulative_fee"] == Decimal("0.35")
    async with ledger._pool.acquire() as c:
        assert await c.fetchval(f"SELECT count(*) FROM {s}.paper_trades") == 2
        assert await c.fetchval(f"SELECT count(*) FROM {s}.paper_fill_quote_evidence") == 2
        assert await c.fetchval("SELECT to_regclass($1)", f"{s}.paper_quote_events") is None
    assert Decimal((await restarted_broker.account())["cash_balance"]) == Decimal("1939.05")
    assert (await ledger.managed_positions())["AAPL"].total == Decimal("0.6")
    await ledger.close()
    # Physical migration cursor must reject a rewritten legacy relation instead
    # of silently skipping/duplicating heap pages. This is an isolated test DB.
    db = await asyncpg.connect(url)
    await db.execute(f"""CREATE TABLE {s}.paper_quote_events (
        run_id TEXT,event_id TEXT,symbol TEXT,bid NUMERIC,ask NUMERIC,bid_size NUMERIC,ask_size NUMERIC,
        event_time TIMESTAMPTZ,processed_at TIMESTAMPTZ DEFAULT now(),PRIMARY KEY(run_id,event_id))""")
    await db.execute(f"INSERT INTO {s}.paper_quote_events VALUES($1,$2,$3,$4,$5,$6,$7,$8,now())",
                     "rewritten",old.event_id,old.symbol,old.bid,old.ask,old.bid_size,old.ask_size,
                     datetime.fromtimestamp(old.event_time,timezone.utc))
    await migrate_batch(db,limit=1)
    before = await db.fetchval(f"SELECT migrated FROM {s}.paper_quote_migration WHERE singleton")
    await db.execute(f"VACUUM FULL {s}.paper_quote_events")
    try:
        await migrate_batch(db,limit=1)
        raise AssertionError("rewritten heap must fail closed")
    except RuntimeError as exc:
        assert "relation changed" in str(exc)
    assert await db.fetchval(f"SELECT migrated FROM {s}.paper_quote_migration WHERE singleton") == before
    await db.close()
    print(json.dumps({"passed":True,"scenarios":15,"receipt":receipt,"capacity":value,
                      "economic_fills":2,"cumulative_fee":"0.35","restart_replay_fills":0}))


if __name__ == "__main__":
    asyncio.run(check())

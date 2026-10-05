"""Resumable legacy conversion; old quotes are reclaimed only after verification."""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re

from stocks_runtime.quote_store import CompactQuoteStore


async def migrate_batch(connection, *, limit=50000):
    s = "binance_stocks_paper"
    async with connection.transaction():
        await connection.execute("SET LOCAL lock_timeout='2s'")
        await connection.execute("SET LOCAL statement_timeout='60s'")
        await connection.execute(f"INSERT INTO {s}.paper_quote_migration(singleton) VALUES(TRUE) ON CONFLICT DO NOTHING")
        progress = await connection.fetchrow(f"SELECT * FROM {s}.paper_quote_migration WHERE singleton FOR UPDATE")
        if progress["completed"]:
            return {"completed": True, "migrated": progress["migrated"]}
        # Read immutable legacy heap pages sequentially, not random hash-PK
        # order. PostgreSQL 15 TidRangeScan avoids rescanning a multi-GB table.
        relation = await connection.fetchrow(f"SELECT relfilenode::bigint AS identity, "
            f"ceil(pg_relation_size(oid)/8192.0)::bigint AS blocks FROM pg_class "
            f"WHERE oid='{s}.paper_quote_events'::regclass")
        if progress["heap_identity"] is None:
            if progress["migrated"]:
                raise RuntimeError("old migration cursor requires explicit review")
            await connection.execute(f"UPDATE {s}.paper_quote_migration SET heap_identity=$1,heap_blocks=$2 WHERE singleton",
                                     relation["identity"], relation["blocks"])
        elif (relation["identity"] != progress["heap_identity"] or relation["blocks"] != progress["heap_blocks"]):
            raise RuntimeError("legacy relation changed during migration; do not reclaim")
        start = progress["next_block"]
        if start >= relation["blocks"]:
            await connection.execute(f"UPDATE {s}.paper_quote_migration SET completed=TRUE WHERE singleton")
            return {"completed": True, "migrated": progress["migrated"]}
        end = min(relation["blocks"], start+max(1, limit//50))
        # Temporary bounded page batch and progress commit in one transaction.
        await connection.execute(f"""
          CREATE TEMP TABLE quote_batch ON COMMIT DROP AS
          SELECT * FROM {s}.paper_quote_events
          WHERE ctid >= $1::text::tid AND ctid < $2::text::tid
        """, f"({start},0)", f"({end},0)")
        count = await connection.fetchval("SELECT count(*) FROM quote_batch")
        if not count:
            await connection.execute(f"UPDATE {s}.paper_quote_migration SET next_block=$1 WHERE singleton", end)
            return {"completed": False, "migrated": progress["migrated"]}
        await connection.execute(f"""
          INSERT INTO {s}.paper_fill_quote_evidence
          SELECT q.run_id,q.event_id,q.symbol,q.event_time,q.bid,q.ask,q.bid_size,q.ask_size
          FROM quote_batch q WHERE EXISTS (SELECT 1 FROM {s}.paper_trades t
            WHERE t.run_id=q.run_id AND t.quote_event_id=q.event_id) ON CONFLICT DO NOTHING
        """)
        await connection.execute(f"""
          WITH bucketed AS (SELECT *,
            CASE WHEN event_time<$1::timestamptz-interval '30 days' THEN 3600 ELSE 60 END AS seconds,
            CASE WHEN event_time<$1::timestamptz-interval '30 days'
              THEN date_trunc('hour',event_time) ELSE date_trunc('minute',event_time) END AS bucket
            FROM quote_batch WHERE event_time >= $1::timestamptz-interval '370 days'),
          grouped AS (SELECT run_id,symbol,bucket,seconds,min(event_time) AS first_time,max(event_time) AS last_time,
            (array_agg(bid ORDER BY event_time,event_id))[1] AS first_bid,
            (array_agg(ask ORDER BY event_time,event_id))[1] AS first_ask,
            (array_agg(bid ORDER BY event_time DESC,event_id DESC))[1] AS last_bid,
            (array_agg(ask ORDER BY event_time DESC,event_id DESC))[1] AS last_ask,
            min(bid) AS min_bid,max(ask) AS max_ask,count(*) AS samples
            FROM bucketed GROUP BY run_id,symbol,bucket,seconds)
          INSERT INTO {s}.paper_quote_summaries AS old SELECT * FROM grouped
          ON CONFLICT(run_id,symbol,bucket,seconds) DO UPDATE SET
            first_bid=CASE WHEN excluded.first_time<old.first_time THEN excluded.first_bid ELSE old.first_bid END,
            first_ask=CASE WHEN excluded.first_time<old.first_time THEN excluded.first_ask ELSE old.first_ask END,
            last_bid=CASE WHEN excluded.last_time>=old.last_time THEN excluded.last_bid ELSE old.last_bid END,
            last_ask=CASE WHEN excluded.last_time>=old.last_time THEN excluded.last_ask ELSE old.last_ask END,
            first_time=least(old.first_time,excluded.first_time),last_time=greatest(old.last_time,excluded.last_time),
            min_bid=least(old.min_bid,excluded.min_bid),max_ask=greatest(old.max_ask,excluded.max_ask),
            samples=old.samples+excluded.samples
        """, progress["started_at"])
        await connection.execute(f"UPDATE {s}.paper_quote_migration SET next_block=$1,migrated=migrated+$2 WHERE singleton",
                                 end, count)
        return {"completed": False, "migrated": progress["migrated"]+count}


async def verify(connection):
    s = "binance_stocks_paper"
    row = await connection.fetchrow(f"SELECT * FROM {s}.paper_quote_migration WHERE singleton")
    count = await connection.fetchval(f"SELECT count(*) FROM {s}.paper_quote_events")
    missing = await connection.fetchval(f"SELECT count(*) FROM {s}.paper_trades t WHERE NOT EXISTS "
                                        f"(SELECT 1 FROM {s}.paper_fill_quote_evidence e "
                                        "WHERE e.run_id=t.run_id AND e.event_id=t.quote_event_id)")
    # Watermarks may advance after migration; all legacy quotes must be older/equal.
    uncovered = await connection.fetchval(f"SELECT count(*) FROM {s}.paper_quote_events q "
                                          f"WHERE NOT EXISTS(SELECT 1 FROM {s}.paper_quote_latest l "
                                          "WHERE l.run_id=q.run_id AND l.symbol=q.symbol AND l.event_time>=q.event_time)")
    return {"completed": bool(row and row["completed"]), "legacy_rows": count,
            "migrated_rows": row["migrated"] if row else 0, "missing_fill_evidence": missing,
            "uncovered_quotes": uncovered,
            "safe_to_reclaim": bool(row and row["completed"] and row["migrated"] == count and missing == 0 and uncovered == 0)}


async def main_async(args):
    import asyncpg
    from stocks_runtime.settings import dedicated_database_url
    url = os.getenv("STOCK_MAINTENANCE_DATABASE_URL") or dedicated_database_url(
        os.environ["DATABASE_URL"], os.getenv("BINANCE_STOCKS_DATABASE_NAME", "hummingbot_stocks"))
    url = url.replace("postgresql+asyncpg://", "postgresql://")
    connection = await asyncpg.connect(url)
    try:
        name = await connection.fetchval("SELECT current_database()")
        if name not in {"hummingbot_stocks", "hummingbot_stocks_retention_test"}:
            raise ValueError("database is not allowlisted")
        store = CompactQuoteStore("binance_stocks_paper")
        await store.initialize(connection)
        if args.seed:
            # Enumerate tiny run metadata, not millions of raw quote records.
            for row in await connection.fetch("SELECT run_id FROM binance_stocks_paper.paper_runs"):
                await store.seed_legacy(connection, row["run_id"])
        if args.migrate:
            # Runtime must already use compact storage. Legacy must be immutable.
            while True:
                result = await migrate_batch(connection)
                print(json.dumps(result), flush=True)
                if result["completed"]:
                    break
                await asyncio.sleep(.1)
        if args.verify and not args.reclaim:
            result = await verify(connection)
            print(json.dumps(result), flush=True)
        if args.reclaim:
            if not re.fullmatch(r"[0-9a-f]{64}", args.backup_sha256 or ""):
                raise RuntimeError("verified backup SHA256 required")
            async with connection.transaction():
                await connection.execute("SET LOCAL lock_timeout='2s'")
                await connection.execute("LOCK TABLE binance_stocks_paper.paper_quote_events IN ACCESS EXCLUSIVE MODE")
                # One complete verification under lock, not repeated multi-GB
                # scans before and after acquiring it. Matching uses new tables.
                result = await verify(connection)
                print(json.dumps(result), flush=True)
                if not result["safe_to_reclaim"]:
                    raise RuntimeError("legacy changed or migration evidence incomplete")
                # Preserve compatibility relation, release data AND index files.
                await connection.execute("TRUNCATE binance_stocks_paper.paper_quote_events")
        if args.maintain:
            async with connection.transaction():
                await connection.execute("SET LOCAL lock_timeout='2s'")
                result = await store.maintenance(connection)
            from stocks_runtime.database_capacity import publish
            import time
            publish(os.getenv("STOCK_DATABASE_CAPACITY_ROOT","/hummingbot-api/logs"),result,time.time())
            print(json.dumps(result), flush=True)
    finally:
        await connection.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--migrate", action="store_true")
    parser.add_argument("--seed", action="store_true")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--reclaim", action="store_true")
    parser.add_argument("--backup-sha256")
    parser.add_argument("--maintain", action="store_true")
    asyncio.run(main_async(parser.parse_args()))


if __name__ == "__main__":
    main()

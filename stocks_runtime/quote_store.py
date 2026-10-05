"""Bounded PAPER quote storage; summaries are never execution inputs."""
from __future__ import annotations

import json
from datetime import datetime, timezone


class CompactQuoteStore:
    def __init__(self, schema):
        if schema != "binance_stocks_paper":
            raise ValueError("quote maintenance is restricted to the PAPER schema")
        self.schema = schema

    async def initialize(self, connection):
        s = self.schema
        await connection.execute(f"""
          CREATE TABLE IF NOT EXISTS {s}.paper_quote_latest (
            run_id TEXT,symbol TEXT,event_time TIMESTAMPTZ,event_ids JSONB NOT NULL,
            event_id TEXT,bid NUMERIC,ask NUMERIC,bid_size NUMERIC,ask_size NUMERIC,
            PRIMARY KEY(run_id,symbol));
          CREATE TABLE IF NOT EXISTS {s}.paper_quote_summaries (
            run_id TEXT,symbol TEXT,bucket TIMESTAMPTZ,seconds INTEGER,
            first_time TIMESTAMPTZ,last_time TIMESTAMPTZ,first_bid NUMERIC,first_ask NUMERIC,
            last_bid NUMERIC,last_ask NUMERIC,min_bid NUMERIC,max_ask NUMERIC,samples BIGINT,
            PRIMARY KEY(run_id,symbol,bucket,seconds));
          CREATE TABLE IF NOT EXISTS {s}.paper_fill_quote_evidence (
            run_id TEXT,event_id TEXT,symbol TEXT,event_time TIMESTAMPTZ,
            bid NUMERIC,ask NUMERIC,bid_size NUMERIC,ask_size NUMERIC,
            PRIMARY KEY(run_id,event_id));
          CREATE TABLE IF NOT EXISTS {s}.paper_quote_migration (
            singleton BOOLEAN PRIMARY KEY DEFAULT TRUE CHECK(singleton),
            last_run TEXT NOT NULL DEFAULT '',last_event TEXT NOT NULL DEFAULT '',
            migrated BIGINT NOT NULL DEFAULT 0,completed BOOLEAN NOT NULL DEFAULT FALSE,
            started_at TIMESTAMPTZ NOT NULL DEFAULT now());
          ALTER TABLE {s}.paper_quote_migration ADD COLUMN IF NOT EXISTS next_block BIGINT NOT NULL DEFAULT 0;
          ALTER TABLE {s}.paper_quote_migration ADD COLUMN IF NOT EXISTS heap_identity BIGINT;
          ALTER TABLE {s}.paper_quote_migration ADD COLUMN IF NOT EXISTS heap_blocks BIGINT;
        """)

    async def seed_legacy(self, connection, run_id):
        """One-time watermark bootstrap. Never use summaries as latest quotes."""
        s = self.schema
        if await connection.fetchval(f"SELECT EXISTS(SELECT 1 FROM {s}.paper_quote_latest WHERE run_id=$1)", run_id):
            return
        await connection.execute(f"""
          INSERT INTO {s}.paper_quote_latest
          WITH maxima AS MATERIALIZED (
            SELECT symbol,max(event_time) AS event_time FROM {s}.paper_quote_events
            WHERE run_id=$1 GROUP BY symbol)
          SELECT q.run_id,q.symbol,q.event_time,jsonb_agg(q.event_id),
            (array_agg(q.event_id ORDER BY q.event_id DESC))[1],
            (array_agg(q.bid ORDER BY q.event_id DESC))[1],
            (array_agg(q.ask ORDER BY q.event_id DESC))[1],
            (array_agg(q.bid_size ORDER BY q.event_id DESC))[1],
            (array_agg(q.ask_size ORDER BY q.event_id DESC))[1]
          FROM {s}.paper_quote_events q JOIN maxima m USING(symbol,event_time)
          WHERE q.run_id=$1 GROUP BY q.run_id,q.symbol,q.event_time
          ON CONFLICT DO NOTHING
        """, run_id)

    async def accept(self, connection, run_id, quote):
        s = self.schema
        stamp = datetime.fromtimestamp(quote.event_time, tz=timezone.utc)
        await connection.execute(f"INSERT INTO {s}.paper_quote_latest "
                                 "(run_id,symbol,event_time,event_ids) VALUES($1,$2,'-infinity','[]') "
                                 "ON CONFLICT DO NOTHING", run_id, quote.symbol)
        row = await connection.fetchrow(f"SELECT event_time,event_ids FROM {s}.paper_quote_latest "
                                        "WHERE run_id=$1 AND symbol=$2 FOR UPDATE", run_id, quote.symbol)
        ids = json.loads(row["event_ids"]) if isinstance(row["event_ids"], str) else row["event_ids"]
        last_time = row["event_time"]
        # asyncpg represents PostgreSQL -infinity as naive datetime.min.
        if last_time.tzinfo is None:
            last_time = last_time.replace(tzinfo=timezone.utc)
        if stamp < last_time or (stamp == last_time and quote.event_id in ids):
            return False
        # Bounded same-timestamp set: reject additional ambiguous liquidity rather
        # than forgetting IDs and permitting replay. Other timestamps unaffected.
        if stamp == last_time and len(ids) >= 256:
            return False
        ids = [quote.event_id] if stamp > last_time else [*ids, quote.event_id]
        await connection.execute(f"UPDATE {s}.paper_quote_latest SET event_time=$3,event_ids=$4::jsonb,"
                                 "event_id=$5,bid=$6,ask=$7,bid_size=$8,ask_size=$9 "
                                 "WHERE run_id=$1 AND symbol=$2", run_id, quote.symbol, stamp,
                                 json.dumps(ids), quote.event_id, quote.bid, quote.ask,
                                 quote.bid_size, quote.ask_size)
        await self.record_summary(connection, run_id, quote, seconds=60)
        return True

    async def record_summary(self, connection, run_id, quote, *, seconds):
        stamp = datetime.fromtimestamp(quote.event_time, tz=timezone.utc)
        bucket = datetime.fromtimestamp(int(quote.event_time)//seconds*seconds, tz=timezone.utc)
        await connection.execute(f"""
          INSERT INTO {self.schema}.paper_quote_summaries AS old
            VALUES($1,$2,$3,$4,$5,$5,$6,$7,$6,$7,$6,$7,1)
          ON CONFLICT(run_id,symbol,bucket,seconds) DO UPDATE SET
            first_bid=CASE WHEN excluded.first_time<old.first_time THEN excluded.first_bid ELSE old.first_bid END,
            first_ask=CASE WHEN excluded.first_time<old.first_time THEN excluded.first_ask ELSE old.first_ask END,
            last_bid=CASE WHEN excluded.last_time>=old.last_time THEN excluded.last_bid ELSE old.last_bid END,
            last_ask=CASE WHEN excluded.last_time>=old.last_time THEN excluded.last_ask ELSE old.last_ask END,
            first_time=least(old.first_time,excluded.first_time),last_time=greatest(old.last_time,excluded.last_time),
            min_bid=least(old.min_bid,excluded.min_bid),max_ask=greatest(old.max_ask,excluded.max_ask),
            samples=old.samples+excluded.samples
        """, run_id, quote.symbol, bucket, seconds, stamp, quote.bid, quote.ask)

    async def protect_fill(self, connection, run_id, quote):
        await connection.execute(f"INSERT INTO {self.schema}.paper_fill_quote_evidence VALUES($1,$2,$3,$4,$5,$6,$7,$8) "
                                 "ON CONFLICT DO NOTHING", run_id, quote.event_id, quote.symbol,
                                 datetime.fromtimestamp(quote.event_time, tz=timezone.utc),
                                 quote.bid, quote.ask, quote.bid_size, quote.ask_size)

    async def maintenance(self, connection):
        s = self.schema
        # Fold old minute summaries transactionally; do not lose sample counts.
        await connection.execute(f"""
          WITH removed AS (DELETE FROM {s}.paper_quote_summaries
            WHERE (run_id,symbol,bucket,seconds) IN
              (SELECT run_id,symbol,bucket,seconds FROM {s}.paper_quote_summaries
               WHERE seconds=60 AND bucket<now()-interval '30 days' LIMIT 10000)
            RETURNING *), folded AS (
            SELECT run_id,symbol,date_trunc('hour',bucket) AS bucket,3600 AS seconds,
              min(first_time) AS first_time,max(last_time) AS last_time,
              (array_agg(first_bid ORDER BY first_time))[1] AS first_bid,
              (array_agg(first_ask ORDER BY first_time))[1] AS first_ask,
              (array_agg(last_bid ORDER BY last_time DESC))[1] AS last_bid,
              (array_agg(last_ask ORDER BY last_time DESC))[1] AS last_ask,
              min(min_bid) AS min_bid,max(max_ask) AS max_ask,sum(samples)::bigint AS samples
            FROM removed GROUP BY run_id,symbol,date_trunc('hour',bucket))
          INSERT INTO {s}.paper_quote_summaries AS old SELECT * FROM folded
          ON CONFLICT(run_id,symbol,bucket,seconds) DO UPDATE SET
            first_bid=CASE WHEN excluded.first_time<old.first_time THEN excluded.first_bid ELSE old.first_bid END,
            first_ask=CASE WHEN excluded.first_time<old.first_time THEN excluded.first_ask ELSE old.first_ask END,
            last_bid=CASE WHEN excluded.last_time>=old.last_time THEN excluded.last_bid ELSE old.last_bid END,
            last_ask=CASE WHEN excluded.last_time>=old.last_time THEN excluded.last_ask ELSE old.last_ask END,
            first_time=least(old.first_time,excluded.first_time),last_time=greatest(old.last_time,excluded.last_time),
            min_bid=least(old.min_bid,excluded.min_bid),max_ask=greatest(old.max_ask,excluded.max_ask),
            samples=old.samples+excluded.samples
        """)
        await connection.execute(f"DELETE FROM {s}.paper_quote_summaries WHERE bucket<now()-interval '370 days'")
        await connection.execute(f"DELETE FROM {s}.paper_equity_snapshots WHERE snapshot_at<now()-interval '370 days'")
        size = await connection.fetchval("SELECT pg_database_size(current_database())")
        mib = size / (1024*1024)
        return {"database_bytes": size, "target_mib": 500,
                "level": "critical" if mib >= 500 else "warning" if mib >= 450 else "maintenance" if mib >= 400 else "healthy",
                "writes_blocked": False}

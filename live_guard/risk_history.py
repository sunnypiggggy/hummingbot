"""Report-owned observed risk history. No trading/notification dependencies."""
from __future__ import annotations

import hashlib
import json
import math
import os
import sqlite3
import zlib
from contextlib import closing
from datetime import datetime
from pathlib import Path

RETENTION_SECONDS = 370 * 86400


def timestamp(value):
    try:
        number = float(value) if isinstance(value, (int, float)) else datetime.fromisoformat(
            str(value).replace("Z", "+00:00")).timestamp()
        return number if math.isfinite(number) else 0.0
    except (ValueError, TypeError, OverflowError):
        return 0.0


def pack(value):
    return zlib.compress(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode())


def unpack(value):
    return json.loads(zlib.decompress(value))


class RiskHistoryReader:
    """Read-only connections; live permissions must never be inferred here."""

    def __init__(self, path):
        self.path = Path(path)

    def connect(self):
        db = sqlite3.connect(self.path.resolve().as_uri() + "?mode=ro", uri=True, timeout=1)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA query_only=ON")
        return db

    def events(self, *, start, end, bot=None, mechanism=None, limit=500):
        conditions, values = ["occurred>=?", "occurred<=?"], [start, end]
        for name, value in (("bot", bot), ("mechanism", mechanism)):
            if value is not None:
                conditions.append(name + "=?")
                values.append(value)
        with closing(self.connect()) as db:
            db.execute("BEGIN")
            rows = db.execute("SELECT * FROM events WHERE " + " AND ".join(conditions)
                              + " ORDER BY occurred,id LIMIT ?", (*values, min(5000, limit))).fetchall()
            return [dict(source=r["source"], collected_at=r["collected"],
                         occurred_at=r["occurred"], event=unpack(r["payload"])) for r in rows]

    def intervals(self, *, strategy, pair, start, end, mechanism="v22_weekly_buy_gate", known=False):
        with closing(self.connect()) as db:
            db.execute("BEGIN")
            rows = db.execute(
                "SELECT observed,source_time,payload FROM snapshots WHERE strategy=? AND pair=? "
                "AND observed>=? AND observed<=? ORDER BY observed", (strategy, pair, start-180, end)
            ).fetchall()
        result = []
        for a, b in zip(rows, rows[1:]):
            if not (0 < b["observed"]-a["observed"] <= 180
                    and 0 <= a["observed"]-a["source_time"] <= 180
                    and 0 <= b["observed"]-b["source_time"] <= 180):
                continue
            states = [next((g for g in unpack(r["payload"]).get("gate_statuses", [])
                            if g.get("mechanism") == mechanism), {}) for r in (a, b)]
            allowed = {"RISK_ON", "RISK_OFF"} if known else {"RISK_OFF"}
            if known == "permissions":
                eligibility = all(
                    type(unpack(r["payload"]).get("process_running")) is bool
                    and all(type(unpack(r["payload"]).get("final_permissions", {}).get(k)) is bool
                            for k in ("buy_enabled", "sell_enabled")) for r in (a,b))
            else:
                eligibility = all(g.get("state") in allowed and g.get("health") == "HEALTHY"
                                  and g.get("enabled") is True for g in states)
            if eligibility:
                left, right = max(start, a["observed"]), min(end, b["observed"])
                if left < right:
                    if result and result[-1]["end"] == left:
                        result[-1]["end"] = right
                    else:
                        result.append({"start": left, "end": right})
        return result

    def permission_intervals(self, *, strategy, pair, start, end):
        with closing(self.connect()) as db:
            rows = db.execute("SELECT observed,source_time,payload FROM snapshots WHERE strategy=? AND pair=? "
                              "AND observed>=? AND observed<=? ORDER BY observed", (strategy,pair,start-180,end)).fetchall()
        def description(payload):
            state = unpack(payload)
            final = state.get("final_permissions", {})
            if state.get("process_running") is False:
                return ("STOPPED", "all", "机器人进程已停止")
            if final.get("buy_enabled") is False or final.get("sell_enabled") is False:
                scope = "all" if final.get("buy_enabled") is False and final.get("sell_enabled") is False else "buy" if final.get("buy_enabled") is False else "sell"
                labels = [g.get("label") or g.get("mechanism") for g in state.get("gate_statuses", [])
                          if g.get("enabled") and (g.get("buy_enabled") is False or g.get("sell_enabled") is False)]
                return ("RESTRICTED",scope,("／".join(labels) or "风控限制普通交易")[:48])
            return None
        result=[]
        for a,b in zip(rows,rows[1:]):
            if not (0 < b["observed"]-a["observed"] <= 180
                    and 0 <= a["observed"]-a["source_time"] <= 180
                    and 0 <= b["observed"]-b["source_time"] <= 180):
                continue
            desc = description(a["payload"])
            if not desc or desc != description(b["payload"]):
                continue
            left,right=max(start,a["observed"]),min(end,b["observed"])
            if left>=right:
                continue
            status,scope,reason=desc
            if result and result[-1]["end_at"]==left and result[-1]["reason"]==reason and result[-1]["scope"]==scope:
                result[-1]["end_at"]=right
            else:
                result.append(dict(start_at=left,end_at=right,status=status,scope=scope,reason=reason))
        return result


class RiskHistory:
    def __init__(self, root, *, now, sources=()):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "risk_history.sqlite"
        self.db = sqlite3.connect(self.path, timeout=1)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.execute("PRAGMA wal_autocheckpoint=1000")
        self.db.executescript("""
          CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY,value REAL);
          CREATE TABLE IF NOT EXISTS cursors(source TEXT PRIMARY KEY,identity TEXT,offset INTEGER);
          CREATE TABLE IF NOT EXISTS events(id INTEGER PRIMARY KEY,source TEXT,event_id TEXT,
            digest TEXT,occurred REAL,collected REAL,bot TEXT,mechanism TEXT,payload BLOB,
            UNIQUE(source,event_id,digest));
          CREATE INDEX IF NOT EXISTS event_time ON events(occurred);
          CREATE INDEX IF NOT EXISTS event_identity ON events(source,event_id);
          CREATE TABLE IF NOT EXISTS snapshots(strategy TEXT,pair TEXT,observed REAL,
            source_time REAL,payload BLOB,PRIMARY KEY(strategy,pair,observed));
          CREATE TABLE IF NOT EXISTS issues(source TEXT,kind TEXT,first_seen REAL,last_seen REAL,
            count INTEGER,PRIMARY KEY(source,kind));
        """)
        if not self.db.execute("SELECT 1 FROM meta WHERE key='epoch'").fetchone():
            with self.db:
                self.db.execute("INSERT INTO meta VALUES('epoch',?)", (now,))
                for source in sources:
                    path = Path(source)
                    if path.exists():
                        stat = path.stat()
                        self.db.execute("INSERT OR REPLACE INTO cursors VALUES(?,?,?)",
                                        (str(path), self.identity(stat), stat.st_size))
        self.epoch = self.db.execute("SELECT value FROM meta WHERE key='epoch'").fetchone()[0]

    @staticmethod
    def identity(stat):
        return f"{stat.st_dev}:{stat.st_ino}"

    def issue(self, source, kind, now):
        self.db.execute("INSERT INTO issues VALUES(?,?,?,?,1) ON CONFLICT(source,kind) "
                        "DO UPDATE SET last_seen=excluded.last_seen,count=count+1",
                        (str(source), kind, now, now))

    def ingest(self, path, *, now, limit=1000):
        path = Path(path)
        if not path.exists():
            return
        try:
            with path.open("rb") as file, self.db:
                stat = os.fstat(file.fileno())
                identity = self.identity(stat)
                cursor = self.db.execute("SELECT * FROM cursors WHERE source=?", (str(path),)).fetchone()
                offset = cursor["offset"] if cursor else 0
                if cursor and (identity != cursor["identity"] or offset > stat.st_size):
                    offset = 0
                    self.issue(path, "source_rotated_or_truncated", now)
                file.seek(offset)
                for _ in range(limit):
                    line = file.readline(1024*1024+1)
                    if not line:
                        break
                    if not line.endswith(b"\n"):
                        # Never commit a partial record. Oversize records are quarantined.
                        if len(line) > 1024*1024:
                            self.issue(path, "oversize_record", now)
                            while line and not line.endswith(b"\n"):
                                line = file.readline(1024*1024)
                            offset = file.tell()
                        break
                    offset = file.tell()
                    try:
                        event = json.loads(line)
                        if not isinstance(event, dict):
                            raise ValueError("not an event")
                        occurred = timestamp(event.get("occurred_at") or event.get("timestamp")
                                             or event.get("time") or event.get("generated_at"))
                        if occurred and occurred < self.epoch:
                            continue
                        payload = pack(event)
                        digest = hashlib.sha256(payload).hexdigest()
                        event_id = str(event.get("event_id") or digest)
                        existing = self.db.execute(
                            "SELECT digest FROM events WHERE source=? AND event_id=? LIMIT 1",
                            (str(path), event_id)).fetchone()
                        if existing and existing[0] != digest:
                            self.issue(path, "event_identity_conflict", now)
                        self.db.execute("INSERT OR IGNORE INTO events(source,event_id,digest,occurred,"
                                        "collected,bot,mechanism,payload) VALUES(?,?,?,?,?,?,?,?)",
                                        (str(path), event_id, digest, occurred or now, now,
                                         str(event.get("bot") or ""),
                                         str(event.get("mechanism") or ""), payload))
                    except (ValueError, TypeError, UnicodeError):
                        self.issue(path, "invalid_record", now)
                self.db.execute("INSERT OR REPLACE INTO cursors VALUES(?,?,?)",
                                (str(path), identity, offset))
        except (OSError, sqlite3.Error):
            with self.db:
                self.issue(path, "source_unavailable", now)

    def sample(self, robots, *, now, source_times=None):
        with self.db:
            for robot in robots:
                status = robot.get("trading_status", robot)
                source = timestamp((source_times or {}).get(status["strategy"]))
                self.db.execute("INSERT OR IGNORE INTO snapshots VALUES(?,?,?,?,?)",
                                (status["strategy"], status["pair"], now, source, pack(status)))

    def ingest_inventory(self, path, *, now):
        """Do not acknowledge delivery or modify the Guard-owned database."""
        path = Path(path)
        if not path.exists():
            return
        source = str(path) + "#events"
        try:
            with closing(sqlite3.connect(path.resolve().as_uri()+"?mode=ro", uri=True, timeout=1)) as upstream, self.db:
                upstream.execute("PRAGMA query_only=ON")
                upstream.execute("BEGIN")
                row = self.db.execute("SELECT offset FROM cursors WHERE source=?", (source,)).fetchone()
                offset = row[0] if row else 0
                records = upstream.execute("SELECT rowid,event_id,kind,payload,created_at FROM events "
                                           "WHERE rowid>? ORDER BY rowid LIMIT 1000", (offset,)).fetchall()
                for index, event_id, kind, raw, occurred in records:
                    offset = index
                    if occurred < self.epoch:
                        continue
                    event = {"event_id": event_id, "mechanism": "account_inventory", "kind": kind,
                             "occurred_at": occurred, "details": json.loads(raw)}
                    payload = pack(event)
                    digest = hashlib.sha256(payload).hexdigest()
                    self.db.execute("INSERT OR IGNORE INTO events(source,event_id,digest,occurred,collected,bot,mechanism,payload) "
                                    "VALUES(?,?,?,?,?,?,?,?)", (source,event_id,digest,occurred,now,"","account_inventory",payload))
                self.db.execute("INSERT OR REPLACE INTO cursors VALUES(?,?,?)", (source,"sqlite",offset))
        except (OSError, sqlite3.Error, ValueError):
            with self.db:
                self.issue(source, "inventory_source_unavailable", now)

    def maintain(self, *, now):
        last = self.db.execute("SELECT value FROM meta WHERE key='maintenance'").fetchone()
        if last and now-last[0] < 86400:
            return
        with self.db:
            for table, column in (("events", "occurred"), ("snapshots", "observed"), ("issues", "last_seen")):
                self.db.execute(f"DELETE FROM {table} WHERE rowid IN "
                                f"(SELECT rowid FROM {table} WHERE {column}<? LIMIT 50000)",
                                (now-RETENTION_SECONDS,))
            self.db.execute("INSERT OR REPLACE INTO meta VALUES('maintenance',?)", (now,))
        self.db.execute("PRAGMA wal_checkpoint(PASSIVE)")

    def publish(self, *, now):
        reader = RiskHistoryReader(self.path)
        curves = {}
        coverage = {}
        pauses = {}
        permission_coverage = {}
        for strategy in ("grid", "dca"):
            for asset in ("BTC", "ETH"):
                pair = f"{asset}-{'FDUSD' if strategy == 'grid' else 'USDT'}"
                curves[f"{strategy}:{pair}"] = reader.intervals(
                    strategy=strategy, pair=pair, start=now-168*3600, end=now)
                coverage[f"{strategy}:{pair}"] = reader.intervals(
                    strategy=strategy, pair=pair, start=now-168*3600, end=now, known=True)
                pauses[f"{strategy}:{pair}"] = reader.permission_intervals(
                    strategy=strategy,pair=pair,start=now-12*3600,end=now)[-8:]
                permission_coverage[f"{strategy}:{pair}"] = reader.intervals(
                    strategy=strategy,pair=pair,start=now-12*3600,end=now,known="permissions")
        value = {"schema": "report-risk-history-v1", "epoch": self.epoch, "generated_at": now,
                 "retention_days": 370, "risk_off_intervals": curves,
                 "known_signal_intervals": coverage,
                 "permission_intervals": pauses,
                 "permission_coverage": permission_coverage,
                 "issues": [dict(r) for r in self.db.execute("SELECT * FROM issues")],
                 "sqlite_bytes": sum(p.stat().st_size for p in
                                     (self.path, Path(str(self.path)+"-wal"), Path(str(self.path)+"-shm"))
                                     if p.exists())}
        temp = self.root / ".risk_history.json.tmp"
        temp.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
        os.replace(temp, self.root / "risk_history.json")
        return value

    def close(self):
        self.db.close()

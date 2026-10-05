import json
import sqlite3
from pathlib import Path

import pytest

from live_guard.risk_history import RETENTION_SECONDS, RiskHistory, RiskHistoryReader


def append(path, event):
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(event)+"\n")


def event(identity="a", timestamp=101, phase="EXITING"):
    return {"event_id": identity, "timestamp": timestamp, "bot": "grid",
            "mechanism": "strategy_drawdown_breaker", "phase_to": phase}


def status():
    return {"strategy": "grid", "pair": "ETH-FDUSD", "phase": "ACTIVE",
            "final_permissions": {"buy_enabled": False, "sell_enabled": False},
            "gate_statuses": [{"mechanism": "v22_weekly_buy_gate", "state": "RISK_OFF",
                               "health": "HEALTHY", "enabled": True}]}


def test_starts_at_eof_no_backfill_and_restarts(tmp_path):
    source = tmp_path/"events.jsonl"
    append(source, event("old", 50))
    archive = RiskHistory(tmp_path/"out", now=100, sources=[source])
    archive.ingest(source, now=101)
    assert archive.db.execute("SELECT count(*) FROM events").fetchone()[0] == 0
    append(source, event())
    archive.ingest(source, now=102)
    archive.close()
    archive = RiskHistory(tmp_path/"out", now=103, sources=[source])
    archive.ingest(source, now=103)
    assert len(RiskHistoryReader(archive.path).events(start=100,end=200)) == 1
    assert archive.epoch == 100
    archive.close()


def test_duplicate_conflict_and_legitimate_stages(tmp_path):
    source = tmp_path/"events.jsonl"
    archive = RiskHistory(tmp_path/"out", now=100)
    for value in (event(), event(), event(phase="COOLDOWN"), event("b", phase="COOLDOWN")):
        append(source, value)
    archive.ingest(source, now=102)
    assert archive.db.execute("SELECT count(*) FROM events").fetchone()[0] == 3
    assert archive.db.execute("SELECT kind FROM issues").fetchone()[0] == "event_identity_conflict"
    archive.close()


def test_partial_invalid_rotation_and_new_source(tmp_path):
    source = tmp_path/"events.jsonl"
    archive = RiskHistory(tmp_path/"out", now=100)
    source.write_text('{"event_id":"partial","timestamp":101',encoding="utf-8")
    archive.ingest(source, now=102)
    assert archive.db.execute("SELECT offset FROM cursors").fetchone()[0] == 0
    with source.open("a") as file:
        file.write('}\ninvalid\n')
    archive.ingest(source, now=103)
    source.rename(tmp_path/"rotated")
    append(source, event("new",104))
    archive.ingest(source, now=105)
    assert archive.db.execute("SELECT count(*) FROM events").fetchone()[0] == 2
    assert archive.db.execute("SELECT count(*) FROM issues").fetchone()[0] == 2
    archive.close()


def test_sample_gaps_stale_and_no_extend_last_state(tmp_path):
    archive = RiskHistory(tmp_path,now=100)
    for stamp in (100,160,220,500,560):
        archive.sample([status()], now=stamp,source_times={"grid":stamp})
    archive.sample([status()], now=620,source_times={"grid":100})
    spans = RiskHistoryReader(archive.path).intervals(strategy="grid",pair="ETH-FDUSD",start=100,end=1000)
    assert spans == [{"start":100,"end":220},{"start":500,"end":560}]
    assert RiskHistoryReader(archive.path).intervals(strategy="grid",pair="ETH-FDUSD",start=100,end=1000,known=True) == spans
    archive.publish(now=1000)
    archive.close()


def test_retention_reader_is_readonly_and_cursors_protected(tmp_path):
    source = tmp_path/"events.jsonl"
    archive = RiskHistory(tmp_path/"out",now=100)
    append(source,event(timestamp=101))
    archive.ingest(source,now=102)
    archive.maintain(now=101+RETENTION_SECONDS)
    assert len(RiskHistoryReader(archive.path).events(start=0,end=102)) == 1
    archive.maintain(now=102+RETENTION_SECONDS+86400)
    assert archive.db.execute("SELECT count(*) FROM events").fetchone()[0] == 0
    assert archive.db.execute("SELECT count(*) FROM cursors").fetchone()[0] == 1
    reader=RiskHistoryReader(archive.path)
    db=reader.connect()
    with pytest.raises(sqlite3.OperationalError):
        db.execute("DELETE FROM cursors")
    db.close()
    archive.close()


def test_inventory_archive_does_not_mark_delivery(tmp_path):
    source=tmp_path/"inventory.sqlite"
    with sqlite3.connect(source) as db:
        db.execute("CREATE TABLE events(event_id,kind,payload,created_at,delivered)")
        db.execute("INSERT INTO events VALUES('old','x','{}',50,0)")
        db.execute("INSERT INTO events VALUES('new','dust','{}',101,0)")
    archive=RiskHistory(tmp_path/"out",now=100)
    archive.ingest_inventory(source,now=102)
    archive.ingest_inventory(source,now=103)
    assert archive.db.execute("SELECT count(*) FROM events").fetchone()[0] == 1
    with sqlite3.connect(source) as db:
        assert db.execute("SELECT sum(delivered) FROM events").fetchone()[0] == 0
    archive.close()


def test_telegram_retention_code_unchanged():
    # Locks in the explicitly exempt outbox policy, not the canonical archive.
    from live_guard.telegram_notifications import TelegramOutbox
    import inspect
    code=inspect.getsource(TelegramOutbox.maintain)
    assert "7 * 86400" in code or "7*86400" in code


def test_real_report_archives_with_notification_disabled_and_outbox_deleted(tmp_path, monkeypatch):
    from datetime import datetime, timezone
    from live_guard.dca_live_report import UnifiedTelegramReporting
    from live_guard.telegram_notifications import TelegramOutbox
    monkeypatch.setenv("TELEGRAM_NOTIFY_ENABLED", "false")
    report = UnifiedTelegramReporting.__new__(UnifiedTelegramReporting)
    report.output = tmp_path/"report"
    report.dca_state = tmp_path/"dca"
    report.grid_state = tmp_path/"grid"
    report.bots_path = tmp_path/"bots"
    report.dca_state.mkdir()
    report.events = report.dca_state/"telegram_events.jsonl"
    report.archive_risk_sources(datetime.fromtimestamp(100, timezone.utc))
    append(report.events, event("inventory-first",101))
    append(report.events, event("inventory-exit",102,phase="COOLDOWN"))
    report.archive_risk_sources(datetime.fromtimestamp(103, timezone.utc))
    outbox = TelegramOutbox(tmp_path/"outbox.sqlite", channel_id="-100-test")
    outbox.enqueue(event_id="inventory-first",kind="message",text="test")
    outbox.connection.execute("UPDATE outbox SET status='sent',sent_at=101")
    outbox.connection.commit()
    outbox.maintain(force=True,now=101+100*86400)
    assert outbox.connection.execute("SELECT count(*) FROM outbox").fetchone()[0] == 0
    assert len(RiskHistoryReader(report.risk_history.path).events(start=100,end=200)) == 2
    outbox.connection.close()
    report.risk_history.close()

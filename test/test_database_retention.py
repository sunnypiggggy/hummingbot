import sqlite3
from unittest.mock import patch

from management_bot.storage import BotStore
from scripts.database_retention import maintain
from stocks_runtime.database_capacity import publish


def test_discovery_never_includes_instance_backup(tmp_path):
    from scripts.database_retention import discover
    data = tmp_path/"api-files/bots/instances/grid-live-fdusd-400/data"
    data.mkdir(parents=True)
    runtime = data/"walk_forward_portfolio_grid_live_fdusd_400.sqlite"
    runtime.touch()
    (data/"walk_forward_portfolio_grid_live_fdusd_400.before_fix.sqlite").touch()
    assert list(discover(tmp_path)) == [(runtime, "instance")]


def test_market_data_decimal_units_and_financial_flows_protected(tmp_path):
    path=tmp_path/"bot.sqlite"
    now=1800000000
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE MarketData(timestamp INTEGER PRIMARY KEY)")
        db.execute("CREATE TABLE TradeFill(timestamp INTEGER)")
        db.executemany("INSERT INTO MarketData VALUES(?)",[(int((now-371*86400)*1e6),),(int((now-369*86400)*1e6),)])
        db.execute("INSERT INTO TradeFill VALUES(1)")
    preview=maintain(path,"instance",now=now)
    assert preview["candidates"][0]["eligible"] == 1
    assert maintain(path,"instance",now=now,apply=True)["deleted"] == 1
    with sqlite3.connect(path) as db:
        assert db.execute("SELECT count(*) FROM TradeFill").fetchone()[0] == 1
        assert db.execute("SELECT count(*) FROM MarketData").fetchone()[0] == 1


def test_incoming_fk_prevents_cleanup(tmp_path):
    path=tmp_path/"bot.sqlite"
    with sqlite3.connect(path) as db:
        db.executescript("CREATE TABLE MarketData(timestamp INTEGER PRIMARY KEY);"
                         "CREATE TABLE referenced(x REFERENCES MarketData(timestamp));")
    assert maintain(path,"instance",now=1800000000,apply=True)["candidates"] == []


def test_trigger_is_unreviewed_dependency(tmp_path):
    path=tmp_path/"bot.sqlite"
    with sqlite3.connect(path) as db:
        db.executescript('CREATE TABLE MarketData(timestamp INTEGER); CREATE TABLE TradeFill(timestamp INTEGER);'
                        'CREATE TRIGGER side_effect AFTER DELETE ON MarketData BEGIN DELETE FROM TradeFill; END;'
                        'INSERT INTO MarketData VALUES(1); INSERT INTO TradeFill VALUES(1);')
    assert maintain(path,"instance",now=1800000000,apply=True)["deleted"]==0
    with sqlite3.connect(path) as db:
        assert db.execute('SELECT count(*) FROM TradeFill').fetchone()[0]==1


def test_exact_boundary_bounded_batch_and_repeated_run(tmp_path):
    path=tmp_path/"bot.sqlite"
    now=1800000000
    cutoff=(now-370*86400)*1_000_000
    with sqlite3.connect(path) as db:
        db.execute('CREATE TABLE MarketData(timestamp INTEGER)')
        db.executemany('INSERT INTO MarketData VALUES(?)',[(cutoff-1,)]*5001+[(cutoff,),(cutoff+1,)])
    result=maintain(path,"instance",now=now,apply=True)
    assert result['deleted']==5000 and result['remaining']['MarketData']==1
    assert maintain(path,"instance",now=now,apply=True)['deleted']==1
    assert maintain(path,"instance",now=now,apply=True)['deleted']==0
    with sqlite3.connect(path) as db:
        assert db.execute('SELECT count(*) FROM MarketData').fetchone()[0]==2


def test_writer_lock_does_not_mutate_economic_evidence(tmp_path):
    import pytest
    path=tmp_path/"bot.sqlite"
    with sqlite3.connect(path) as db:
        db.executescript('CREATE TABLE MarketData(timestamp INTEGER); INSERT INTO MarketData VALUES(1);'
                        'CREATE TABLE TradeFill(timestamp INTEGER); INSERT INTO TradeFill VALUES(1);')
    with sqlite3.connect(path) as writer:
        writer.execute('BEGIN IMMEDIATE')
        with pytest.raises(sqlite3.OperationalError,match='locked'):
            maintain(path,'instance',now=1800000000,apply=True)
    assert maintain(path,'instance',now=1800000000,apply=True)['deleted']==1
    with sqlite3.connect(path) as db:
        assert db.execute('SELECT count(*) FROM TradeFill').fetchone()[0]==1


def test_inventory_cleanup_keeps_open_episode_unsent_and_financial_events(tmp_path):
    path=tmp_path/'inventory.sqlite'
    with sqlite3.connect(path) as db:
        db.executescript('CREATE TABLE events(created_at REAL,delivered INTEGER,kind TEXT,payload TEXT);'
                        'CREATE TABLE inventory_episodes(episode_id TEXT); INSERT INTO inventory_episodes VALUES("active");')
        db.executemany('INSERT INTO events VALUES(1,?,?,?)',[
            (1,'inventory_dust_classified','{"episode_id":"old"}'),
            (1,'inventory_dust_classified','{"episode_id":"active"}'),
            (0,'inventory_dust_classified','{"episode_id":"old"}'),
            (1,'inventory_liquidation_completed','{"episode_id":"old"}')])
    assert maintain(path,'inventory',now=1800000000,apply=True)['deleted']==1
    with sqlite3.connect(path) as db:
        assert db.execute('SELECT count(*) FROM events').fetchone()[0]==3


def test_bot_370_day_logs_indefinite_actions_and_session_ttl(tmp_path):
    store=BotStore(tmp_path/"bot.sqlite")
    with patch("management_bot.storage.time.time",return_value=100):
        store.claim_action("do-not-repeat")
        store.claim_update(1)
        session=store.create_session(1,1,"wizard")
    with patch("management_bot.storage.time.time",return_value=100+369*86400):
        assert not store.claim_update(1)
        assert store.get_session(session["session_id"]) is None
    with patch("management_bot.storage.time.time",return_value=100+371*86400):
        store.audit("CHECK")
        store.claim_update(2)
        assert store.db.execute("SELECT count(*) FROM processed_updates").fetchone()[0] == 1
        assert store.claim_action("do-not-repeat")[0] is False
    store.close()


def test_capacity_alert_dedup_and_writes_not_blocked(tmp_path):
    value={"level":"critical","database_bytes":600*1048576,"writes_blocked":False}
    publish(tmp_path,value,100)
    publish(tmp_path,value,200)
    assert len((tmp_path/"telegram_events.jsonl").read_text(encoding="utf-8").splitlines()) == 1
    publish(tmp_path,{**value,"level":"healthy"},300)
    assert len((tmp_path/"telegram_events.jsonl").read_text(encoding="utf-8").splitlines()) == 2
    from live_guard.telegram_notifications import format_event
    import json
    event=json.loads((tmp_path/"telegram_events.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert "500" in format_event(event)


def test_capacity_health_observes_external_upkeep_without_restart(tmp_path):
    from stocks_runtime.database_capacity import latest
    value = {"level": "critical", "database_bytes": 600*1048576, "writes_blocked": False}
    old = publish(tmp_path, value, 100)
    new = publish(tmp_path, {**value, "level": "healthy", "database_bytes": 100*1048576}, 200)
    with patch.dict("os.environ", {"STOCK_DATABASE_CAPACITY_ROOT": str(tmp_path)}):
        assert latest(old) == new
        (tmp_path/"database_capacity.json").write_text("incomplete", encoding="utf-8")
        assert latest(old) == old

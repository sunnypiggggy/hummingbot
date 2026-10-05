import json
import sqlite3
import time
from pathlib import Path
from unittest.mock import Mock

import pytest
import requests

from cardputer_monitor.collector import ROBOTS, SnapshotCollector
from cardputer_monitor.__main__ import main
from cardputer_monitor.config import Config, TELEMETRY_URL
from cardputer_monitor.history import HistoryCollector, PRICE_URL, PublicKlinePrices, sampled_curve
from cardputer_monitor.publisher import PublishError, TelemetryPublisher


@pytest.fixture
def history_database(tmp_path):
    now = int(time.time())
    connection = sqlite3.connect(tmp_path / "telegram_outbox.sqlite")
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("CREATE TABLE profit_snapshot(strategy TEXT,pair TEXT,observed_at REAL,"
                       "mtm_quote REAL,equity REAL,drawdown_pct REAL,payload_json TEXT,"
                       "PRIMARY KEY(strategy,pair,observed_at))")
    for strategy, pair, _, _ in ROBOTS:
        for index in range(73):
            connection.execute("INSERT INTO profit_snapshot VALUES (?,?,?,?,?,?,?)", (
                strategy, pair, now - 43200 + index * 600 - 10,
                -10 + index * 0.25, 200, 0.5, '{"schema":"telegram-profit-snapshot-v2"}',
            ))
    connection.commit()
    yield tmp_path, now, connection
    connection.close()


def test_canonical_mtm_difference_keeps_real_zero_and_fixed_currencies(history_database):
    directory, now, _ = history_database
    history = HistoryCollector(directory / "telegram_outbox.sqlite").collect(now=now)
    assert list(history) == [f"{s}:{p}" for s, p, _, _ in ROBOTS]
    for row in history.values():
        assert row["end_at"] - row["start_at"] == 43200
        assert len(row["profit_points"]) == len(row["price_points"]) == 73
        assert [point["value"] for point in row["profit_points"]] == [index * .25 for index in range(73)]
        assert row["profit_observed_at"] == now - 10 and row["window_complete"] is True
        assert all(point["value"] is None for point in row["price_points"])
        assert row["price_observed_at"] is None
        assert row["pauses"] == [] and row["status_coverage_start_at"] is None
        assert row["data_state"] == "FRESH"


def test_risk_is_readonly_report_coverage_and_stale_is_unknown(history_database):
    directory, now, _ = history_database
    path = directory/"risk_history.json"
    path.write_text(json.dumps({"schema":"report-risk-history-v1","epoch":now-43200,
        "permission_coverage":{"grid:BTC-FDUSD":[{"start":now-120,"end":now-60}]},
        "permission_intervals":{"grid:BTC-FDUSD":[{"start_at":now-300,"end_at":now,
            "status":"RESTRICTED","scope":"buy","reason":"v22限制买入"}]}}),encoding="utf-8")
    collector=HistoryCollector(directory/"telegram_outbox.sqlite")
    value=collector.collect(now=now)["grid:BTC-FDUSD"]
    assert value["status_coverage_start_at"] == now-120
    assert value["status_coverage_end_at"] == now-60
    assert value["pauses"][0]["start_at"] == now-120
    assert value["pauses"][0]["end_at"] == now-60
    stale=collector.collect(now=now+181)["grid:BTC-FDUSD"]
    assert stale["pauses"] == [] and stale["status_coverage_start_at"] is None


@pytest.mark.parametrize("case", ["baseline_missing", "baseline_old", "baseline_null", "gap", "nan", "infinity"])
def test_missing_or_invalid_mtm_is_not_filled_or_interpolated(history_database, case):
    directory, now, writer = history_database
    first = now - 43200 - 10
    if case == "baseline_missing":
        writer.execute("DELETE FROM profit_snapshot WHERE observed_at=?", (first,))
    elif case == "baseline_old":
        writer.execute("UPDATE profit_snapshot SET observed_at=? WHERE observed_at=?", (now - 43200 - 301, first))
    else:
        observed = first if case == "baseline_null" else first + 10 * 600
        value = None if case in ("baseline_null", "gap") else "NaN" if case == "nan" else "Infinity"
        writer.execute("UPDATE profit_snapshot SET mtm_quote=? WHERE observed_at=?", (value, observed))
    writer.commit()
    history = HistoryCollector(directory / "telegram_outbox.sqlite").collect(now=now)
    for row in history.values():
        assert row["window_complete"] is False
        values = [point["value"] for point in row["profit_points"]]
        if case.startswith("baseline"):
            assert values == [None] * 73
        else:
            assert values[10] is None and values[11] == 2.75
    json.dumps(history, allow_nan=False)


@pytest.mark.parametrize("age,state", [(300, "FRESH"), (301, "STALE"), (-31, "UNAVAILABLE")])
def test_history_uses_actual_source_age_not_collection_heartbeat(history_database, age, state):
    directory, now, writer = history_database
    writer.execute("UPDATE profit_snapshot SET observed_at=? WHERE observed_at=?", (now - age, now - 10))
    writer.commit()
    history = HistoryCollector(directory / "telegram_outbox.sqlite").collect(now=now)
    for row in history.values():
        assert row["data_state"] == state
        assert row["profit_observed_at"] == (None if state == "UNAVAILABLE" else now - age)


def test_wal_transaction_sees_one_version_across_all_points_and_robots(history_database, monkeypatch):
    directory, now, writer = history_database
    real_connect = sqlite3.connect
    trace = []
    connections = []
    changed = False

    def connect(database, *args, **kwargs):
        nonlocal changed
        assert database.endswith("?mode=ro") and kwargs["uri"] is True
        connection = real_connect(database, *args, **kwargs)
        connections.append(connection)

        def observed(query):
            nonlocal changed
            trace.append(query)
            if query.startswith("SELECT") and sum(q.startswith("SELECT") for q in trace) == 2 and not changed:
                writer.execute("UPDATE profit_snapshot SET mtm_quote=mtm_quote*2")
                writer.commit()
                changed = True

        connection.set_trace_callback(observed)
        return connection

    monkeypatch.setattr("cardputer_monitor.history.sqlite3.connect", connect)
    first = HistoryCollector(directory / "telegram_outbox.sqlite").collect(now=now)
    assert changed and all(row["profit_points"][-1]["value"] == 18 for row in first.values()), trace[:4]
    assert trace[0] == "BEGIN" and all(q.startswith(("BEGIN", "SELECT")) for q in trace)
    second = HistoryCollector(directory / "telegram_outbox.sqlite").collect(now=now)
    assert all(row["profit_points"][-1]["value"] == 36 for row in second.values())
    for connection in connections:
        with pytest.raises(sqlite3.ProgrammingError):
            connection.execute("SELECT 1")


def test_missing_database_is_never_created(tmp_path):
    history = HistoryCollector(tmp_path / "missing.sqlite").collect(now=100000)
    assert not (tmp_path / "missing.sqlite").exists()
    assert all(row["data_state"] == "UNAVAILABLE" for row in history.values())


def candle(open_seconds, price="100"):
    return [open_seconds * 1000, "100", "101", "99", price, "10",
            (open_seconds + 300) * 1000 - 1, "1000", 5, "5", "500", "0"]


def price_client(rows=None, status=200, raw=None):
    session = Mock()
    session.headers = {"Authorization": "must-be-removed"}
    response = Mock(status_code=status)
    body = raw if raw is not None else json.dumps(rows).encode()
    response.iter_content.return_value = [body]
    session.get.return_value = response
    return PublicKlinePrices(session=session), session, response


def test_public_market_client_is_keyless_bounded_tls_and_uses_only_closed_5m_candles():
    end = 180000
    rows = [candle(end - 600, "123"), candle(end - 300, "124"), candle(end, "999"),
            candle(end - 1200, "NaN"), candle(end - 1500, "0"), candle(end - 2100, "-1")]
    client, session, response = price_client(rows)
    samples = client.samples("BTC-FDUSD", end - 43200, end)
    assert samples == [(end - 300.001, 123), (end - .001, 124)]
    call = session.get.call_args
    assert call.args == (PRICE_URL,)
    assert call.kwargs["params"]["symbol"] == "BTCFDUSD"
    assert call.kwargs["params"]["interval"] == "5m" and call.kwargs["params"]["limit"] == 150
    assert call.kwargs["verify"] is True and call.kwargs["allow_redirects"] is False
    assert session.trust_env is False and session.auth is None and session.headers == {}
    session.cookies.clear.assert_called_once()
    response.close.assert_called_once()
    session.get.reset_mock()
    assert client.samples("BTC-USDC", end - 43200, end) == []
    session.get.assert_not_called()


@pytest.mark.parametrize("case", ["redirect", "oversize", "malformed", "huge_list", "network", "duplicate", "bad_interval"])
def test_public_market_failure_does_not_invent_price(case):
    rows = [candle(179700)]
    client, session, response = price_client(rows)
    if case == "redirect":
        response.status_code = 302
    elif case == "oversize":
        response.iter_content.return_value = [b"x" * (128 * 1024 + 1)]
    elif case == "malformed":
        response.iter_content.return_value = [b'{"error":"malformed"}']
    elif case == "huge_list":
        response.iter_content.return_value = [json.dumps(rows * 151).encode()]
    elif case == "network":
        session.get.side_effect = requests.Timeout("redacted")
    else:
        if case == "duplicate":
            rows *= 2
        else:
            rows[0][6] += 1
        response.iter_content.return_value = [json.dumps(rows).encode()]
    assert client.samples("BTC-FDUSD", 136800, 180000) == []
    if case != "network":
        response.close.assert_called_once()


def test_sampling_has_no_lookahead_or_cross_gap_carry():
    curve = sampled_curve([(100, 1), (701, 2)], [100, 400, 401, 700, 701])
    assert [row["value"] for row in curve] == [1, 1, None, None, 2]


def test_price_failure_and_profit_failure_are_independent(history_database):
    directory, now, writer = history_database
    prices = Mock()
    prices.samples.side_effect = lambda pair, start, end: [(end - 1, 60000 if pair.startswith("BTC") else 2000)]
    collector = HistoryCollector(directory / "telegram_outbox.sqlite", prices=prices)
    rows = collector.collect(now=now)
    assert all(row["profit_points"][-1]["value"] == 18 for row in rows.values())
    assert all(row["price_points"][-1]["value"] is not None for row in rows.values())
    writer.execute("DROP TABLE profit_snapshot")
    writer.commit()
    rows = collector.collect(now=now)
    assert all(row["profit_observed_at"] is None and row["data_state"] == "FRESH" for row in rows.values())
    assert all(row["price_points"][-1]["value"] is not None for row in rows.values())


@pytest.mark.parametrize("failure", ["future", "null"])
def test_invalid_latest_profit_does_not_invalidate_valid_price_payload(history_database, failure):
    directory, now, writer = history_database
    if failure == "future":
        writer.execute("UPDATE profit_snapshot SET observed_at=? WHERE observed_at=?", (now + 31, now - 10))
    else:
        writer.execute("UPDATE profit_snapshot SET mtm_quote=NULL WHERE observed_at=?", (now - 10,))
    writer.commit()
    prices = Mock()
    prices.samples.side_effect = lambda pair, start, end: [(end - 1, 100)]
    rows = HistoryCollector(directory / "telegram_outbox.sqlite", prices=prices).collect(now=now)
    assert all(row["profit_observed_at"] is None for row in rows.values())
    assert all(point["value"] is None for row in rows.values() for point in row["profit_points"])
    assert all(row["price_points"][-1]["value"] == 100 and row["data_state"] == "FRESH" for row in rows.values())


def test_history_is_optional_to_existing_snapshot_but_attached_by_cli_collector(history_database):
    directory, now, _ = history_database
    history = HistoryCollector(directory / "telegram_outbox.sqlite")
    sample = SnapshotCollector(directory, history=history).collect(now=now)
    assert len(sample["robots"]) == 4
    assert all(row["history"]["window_complete"] for row in sample["robots"])
    assert all("history" not in row for row in SnapshotCollector(directory).collect(now=now)["robots"])
    assert len(json.dumps(sample, ensure_ascii=False, separators=(",", ":")).encode()) < 64 * 1024


def test_dry_run_with_configured_public_provider_still_never_opens_network(tmp_path, monkeypatch, capsys):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"reports_dir": str(tmp_path), "price_provider": "binance-public"}), encoding="utf-8")
    public = Mock(side_effect=AssertionError("dry-run must not create network client"))
    publisher = Mock(side_effect=AssertionError("dry-run must not create publisher"))
    monkeypatch.setattr("cardputer_monitor.__main__.PublicKlinePrices", public)
    monkeypatch.setattr("cardputer_monitor.__main__.TelemetryPublisher", publisher)
    assert main(["--config", str(path), "--once"]) == 0
    sample = json.loads(capsys.readouterr().out)
    assert len(sample["robots"][0]["history"]["profit_points"]) == 73
    assert sample["robots"][0]["history"]["price_observed_at"] is None
    public.assert_not_called()
    publisher.assert_not_called()


@pytest.mark.parametrize("configuration", [{"price_provider": "trading"}, {"price_timeout_seconds": True},
                                             {"price_timeout_seconds": 11}])
def test_price_configuration_has_explicit_provider_and_bounded_timeout(tmp_path, configuration):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(configuration), encoding="utf-8")
    with pytest.raises(ValueError):
        Config.load(path)


def test_oversize_telemetry_rejected_before_network(tmp_path):
    token = tmp_path / "token"
    token.write_text("synthetic-token", encoding="utf-8")
    session = Mock()
    with pytest.raises(PublishError, match="64 KiB"):
        TelemetryPublisher(TELEMETRY_URL, token, session=session).publish({"value": "x" * 65536})
    session.post.assert_not_called()


def test_shared_history_fixture_matches_locked_contract():
    path = Path(__file__).resolve().parents[1] / "cardputer_monitor/fixtures/trading-history.json"
    sample = json.loads(path.read_text(encoding="utf-8"))
    assert len(json.dumps(sample, ensure_ascii=False, separators=(",", ":")).encode()) < 65536
    for robot in sample["robots"]:
        history = robot["history"]
        assert history["end_at"] - history["start_at"] == 43200
        for name in ("profit_points", "price_points"):
            times = [point["ts"] for point in history[name]]
            assert len(times) == 73 and times == sorted(set(times))
        assert len(history["pauses"]) <= 8
        assert all(len(pause["reason"]) <= 48 for pause in history["pauses"])

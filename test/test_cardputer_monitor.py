import copy
import json
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import requests

from cardputer_monitor.__main__ import main
from cardputer_monitor.collector import MAX_AGE_SECONDS, ROBOTS, STATUS_SCHEMA, SnapshotCollector
from cardputer_monitor.config import Config, TELEMETRY_URL
from cardputer_monitor.publisher import PublishError, TelemetryPublisher
from live_guard.trading_status import evaluate_status, gate_row
from management_bot.clients import OperationsReportReader


def status_document(now):
    rows = []
    for strategy, pair, _, bot in ROBOTS:
        rows.append(evaluate_status(
            generated_at=datetime.fromtimestamp(now, timezone.utc).isoformat(),
            strategy=strategy, bot=bot, pair=pair, process_running=True, phase="ACTIVE",
            gates=[gate_row("v22_weekly_buy_gate"), gate_row("controller_application_gate")],
        ))
    return {"schema": STATUS_SCHEMA, "generated_at": rows[0]["generated_at"], "robots": rows}


def write_status(root, now, transform=None):
    document = status_document(now)
    if transform:
        transform(document)
    (root / "trading_status.json").write_text(json.dumps(document), encoding="utf-8")
    return document


def profit_database(root, now):
    connection = sqlite3.connect(root / "telegram_outbox.sqlite")
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute(
        "CREATE TABLE profit_snapshot(strategy TEXT,pair TEXT,observed_at REAL,mtm_quote REAL,"
        "equity REAL,drawdown_pct REAL,payload_json TEXT,PRIMARY KEY(strategy,pair,observed_at))"
    )
    for strategy, pair, _, _ in ROBOTS:
        for hours, mtm in ((168, 1.0), (24, 2.0), (4, 3.0), (0, 5.0)):
            connection.execute("INSERT INTO profit_snapshot VALUES (?,?,?,?,?,?,?)", (
                strategy, pair, now - hours * 3600, mtm, 200 + mtm, 0.4,
                '{"schema":"telegram-profit-snapshot-v2"}',
            ))
    connection.commit()
    return connection


@pytest.fixture
def reports(tmp_path):
    now = int(time.time())
    write_status(tmp_path, now)
    writer = profit_database(tmp_path, now)
    yield tmp_path, now, writer
    writer.close()


def test_real_reader_exports_four_units_and_owned_mtm_windows(reports):
    root, now, _ = reports
    sample = SnapshotCollector(root).collect(now=now)
    assert sample["source_id"] == "hummingbot-main" and sample["schema_version"] == 1
    assert len(sample["robots"]) == 4
    assert [row["id"] for row in sample["robots"]] == [f"{s}:{p}" for s, p, _, _ in ROBOTS]
    assert [row["quote_asset"] for row in sample["robots"]] == ["FDUSD", "FDUSD", "USDT", "USDT"]
    assert len({row["bot_name"] for row in sample["robots"]}) == 3
    for row in sample["robots"]:
        assert row["status"] == "NORMAL"
        assert row["profit"] == {"4h": 2.0, "24h": 3.0, "7d": 4.0, "all": 5.0}
        assert row["window_complete"] == {"4h": True, "24h": True, "7d": True}
        assert row["status_observed_at"] == row["profit_observed_at"] == now
    json.dumps(sample, allow_nan=False)


def test_live_wal_is_read_inside_transaction_without_writes(reports, monkeypatch):
    root, now, writer = reports
    assert (root / "telegram_outbox.sqlite-wal").exists()
    trace, connections = [], []
    connect = sqlite3.connect

    def readonly_connect(database, *args, **kwargs):
        connections.append((database, kwargs))
        connection = connect(database, *args, **kwargs)
        connection.set_trace_callback(trace.append)
        return connection

    monkeypatch.setattr("management_bot.clients.sqlite3.connect", readonly_connect)
    collector = SnapshotCollector(root)
    assert collector.collect(now=now)["robots"][0]["profit"]["all"] == 5
    writer.execute("UPDATE profit_snapshot SET mtm_quote=6 WHERE strategy='grid' AND pair='BTC-FDUSD' AND observed_at=?", (now,))
    writer.commit()
    assert collector.collect(now=now)["robots"][0]["profit"]["all"] == 6
    assert len(connections) == 2
    assert all("mode=ro" in uri and kwargs["uri"] is True for uri, kwargs in connections)
    assert trace.count("BEGIN") == 2
    assert all(query.startswith(("BEGIN", "SELECT")) for query in trace)


@pytest.mark.parametrize("source", ["status", "profit"])
def test_sources_degrade_independently(reports, source):
    root, now, writer = reports
    if source == "status":
        (root / "trading_status.json").write_text("not JSON", encoding="utf-8")
    else:
        writer.execute("DROP TABLE profit_snapshot")
        writer.commit()
    sample = SnapshotCollector(root).collect(now=now)
    for row in sample["robots"]:
        if source == "status":
            assert row["status"] == "UNAVAILABLE" and row["buy_enabled"] is None
            assert row["profit"]["all"] == 5
        else:
            assert row["status"] == "NORMAL" and row["profit_data_state"] == "UNAVAILABLE"
            assert all(value is None for value in row["profit"].values())


@pytest.mark.parametrize("age,state", [(300, "FRESH"), (301, "STALE"), (-31, "UNAVAILABLE")])
def test_source_time_controls_freshness_even_when_heartbeat_is_new(reports, age, state, monkeypatch):
    root, now, writer = reports
    monkeypatch.setattr("management_bot.clients.time.time", lambda: now)
    observed = now - age
    write_status(root, observed)
    writer.execute("UPDATE profit_snapshot SET observed_at=? WHERE observed_at=?", (observed, now))
    writer.commit()
    sample = SnapshotCollector(root).collect(now=now)
    assert sample["collected_at"] == now
    for row in sample["robots"]:
        assert row["status_data_state"] == row["profit_data_state"] == state
        if state != "FRESH":
            assert row["status"] == "UNAVAILABLE" and row["buy_enabled"] is None
            assert row["profit"]["all"] is None
        if state == "UNAVAILABLE":
            assert row["status_observed_at"] is None and row["profit_observed_at"] is None


def test_missing_duplicate_and_wrong_bot_cannot_borrow_other_units(reports):
    root, now, _ = reports

    def corrupt(document):
        document["robots"].append(copy.deepcopy(document["robots"][0]))
        document["robots"][1]["bot"] = "other-bot"
        document["robots"] = [row for row in document["robots"] if row["pair"] != "ETH-USDT"]

    write_status(root, now, corrupt)
    rows = SnapshotCollector(root).collect(now=now)["robots"]
    assert [row["status"] for row in rows] == ["UNAVAILABLE", "UNAVAILABLE", "NORMAL", "UNAVAILABLE"]
    assert all(row["profit"]["all"] == 5 for row in rows)


@pytest.mark.parametrize("broken", ["schema", "naive_time", "missing_time", "normal_claim", "boolean_type"])
def test_invalid_status_contract_is_never_normal(reports, broken):
    root, now, _ = reports

    def corrupt(document):
        if broken == "schema":
            document["schema"] = "wrong"
        elif broken == "naive_time":
            document["generated_at"] = datetime.fromtimestamp(now).isoformat()
        elif broken == "missing_time":
            document.pop("generated_at")
        elif broken == "normal_claim":
            document["robots"][0]["final_permissions"]["buy_enabled"] = False
        else:
            document["robots"][0]["process_running"] = "true"

    write_status(root, now, corrupt)
    assert SnapshotCollector(root).collect(now=now)["robots"][0]["status"] == "UNAVAILABLE"


def test_sample_shortage_null_valuation_and_real_zero(reports):
    root, now, writer = reports
    writer.execute("DELETE FROM profit_snapshot WHERE observed_at<?", (now,))
    writer.execute("UPDATE profit_snapshot SET mtm_quote=0 WHERE strategy='grid' AND pair='BTC-FDUSD'")
    writer.execute("UPDATE profit_snapshot SET mtm_quote=NULL WHERE strategy='dca' AND pair='ETH-USDT'")
    writer.commit()
    rows = SnapshotCollector(root).collect(now=now)["robots"]
    assert rows[0]["profit"]["all"] == 0
    assert rows[0]["profit"]["4h"] is None
    assert rows[0]["window_complete"] == {"4h": False, "24h": False, "7d": False}
    assert rows[3]["profit_data_state"] == "UNAVAILABLE" and rows[3]["equity"] is None
    assert all(value is None for value in rows[3]["profit"].values())


def test_risk_statuses_follow_authority_and_raw_secrets_are_not_exported(reports):
    root, now, _ = reports
    secret = "SECRET_BEARER_SHOULD_NEVER_LEAVE"
    source = status_document(now)
    gates = [
        [gate_row("capital_budget_gate", state="ALERT_ONLY", reason=secret)],
        [gate_row("v22_weekly_buy_gate", state="RISK_OFF", buy_enabled=False, reason=secret)],
        [gate_row("order_execution_gate", state="RETRYING", health="FAILED", reason=secret)],
        [gate_row("infrastructure_integrity_breaker", buy_enabled=False, sell_enabled=False, health="FAILED", reason=secret)],
    ]
    for index, row in enumerate(source["robots"]):
        row.update(evaluate_status(
            generated_at=source["generated_at"], strategy=row["strategy"], bot=row["bot"], pair=row["pair"],
            process_running=True, phase="LATCHED" if index == 3 else "ACTIVE", gates=gates[index],
        ))
        row["secret"] = secret
        row["recovery"] = {"phase": "LATCHED", "token": secret, "reentry_block_reason": secret}
    (root / "trading_status.json").write_text(json.dumps(source), encoding="utf-8")
    sample = SnapshotCollector(root).collect(now=now)
    assert [row["status"] for row in sample["robots"]] == ["NORMAL", "RESTRICTED", "RESTRICTED", "RESTRICTED"]
    assert sample["robots"][2]["trade_mode"] == "EXECUTION_DEGRADED"
    assert sample["robots"][3]["phase"] == "LATCHED"
    assert secret not in json.dumps(sample, ensure_ascii=False)


def test_atomic_replacement_is_reopened_and_mismatch_is_retried(reports):
    root, now, _ = reports
    delegate = OperationsReportReader(root / "trading_status.json", root / "telegram_outbox.sqlite")
    replacement = status_document(now)
    replacement["robots"][0]["process_running"] = False
    replacement["robots"][0]["trading_normal"] = False
    replacement["robots"][0]["trade_mode"] = "STOPPED"
    calls = 0

    def status():
        nonlocal calls
        calls += 1
        if calls == 1:
            temporary = root / "replacement.tmp"
            temporary.write_text(json.dumps(replacement), encoding="utf-8")
            temporary.replace(root / "trading_status.json")
        return delegate.status()

    reader = SimpleNamespace(status=status, profits=delegate.profits)
    collector = SnapshotCollector(root, reader=reader)
    assert collector.collect(now=now)["robots"][0]["status"] == "STOPPED"
    assert calls == 2
    assert collector.collect(now=now)["robots"][0]["status"] == "STOPPED"


def test_nonfinite_values_never_escape_strict_json(reports):
    root, now, _ = reports
    delegate = OperationsReportReader(root / "trading_status.json", root / "telegram_outbox.sqlite")
    profits = delegate.profits()
    row = next(item for item in profits["robots"] if item["strategy"] == "grid" and item["pair"] == "BTC-FDUSD")
    row["profit"]["four_hour_mtm_quote"] = float("nan")
    row["equity"] = float("inf")
    row["drawdown_pct"] = True
    reader = SimpleNamespace(status=delegate.status, profits=lambda: profits)
    sample = SnapshotCollector(root, reader=reader).collect(now=now)
    assert sample["robots"][0]["profit"]["4h"] is None
    assert sample["robots"][0]["equity"] is None
    assert sample["robots"][0]["drawdown_pct"] is None
    json.dumps(sample, allow_nan=False)


def test_dry_run_never_reads_token_or_opens_network(tmp_path, monkeypatch, capsys):
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"reports_dir": str(tmp_path), "token_file": str(tmp_path / "missing-token")}), encoding="utf-8")
    publisher = Mock(side_effect=AssertionError("publisher must not be created in dry-run"))
    monkeypatch.setattr("cardputer_monitor.__main__.TelemetryPublisher", publisher)
    assert main(["--config", str(config), "--once"]) == 0
    sample = json.loads(capsys.readouterr().out)
    assert len(sample["robots"]) == 4 and all(row["status"] == "UNAVAILABLE" for row in sample["robots"])
    publisher.assert_not_called()


def test_https_publisher_sends_only_bearer_and_retries_same_sample(tmp_path):
    token = tmp_path / "telemetry-token"
    token.write_text("test-only-token", encoding="utf-8")
    responses = [Mock(status_code=503), Mock(status_code=204)]
    session = Mock()
    session.post.side_effect = responses
    sleep = Mock()
    sample = {"sample_id": "same-sample", "robots": []}
    publisher = TelemetryPublisher(TELEMETRY_URL, token, session=session, sleep=sleep)
    publisher.publish(sample)
    assert session.post.call_count == 2 and sleep.call_args.args == (1,)
    assert session.post.call_args_list[0] == session.post.call_args_list[1]
    options = session.post.call_args.kwargs
    assert options["headers"]["Authorization"] == "Bearer test-only-token"
    assert options["allow_redirects"] is False and options["verify"] is True
    assert options["timeout"] == 10 and session.trust_env is False
    assert json.loads(options["data"]) == sample
    assert all(response.close.called for response in responses)


@pytest.mark.parametrize("code", [301, 401, 403, 422, 429])
def test_redirect_and_client_failure_cannot_leak_response_or_retry(tmp_path, code):
    token = tmp_path / "telemetry-token"
    token.write_text("test-token", encoding="utf-8")
    session = Mock()
    session.post.return_value = Mock(status_code=code, text="SECRET_RESPONSE")
    with pytest.raises(PublishError, match=f"telemetry HTTP {code}") as error:
        TelemetryPublisher(TELEMETRY_URL, token, session=session).publish({})
    assert session.post.call_count == 1 and "SECRET" not in str(error.value)


def test_connection_failure_is_sanitized_and_retry_is_bounded(tmp_path):
    token = tmp_path / "telemetry-token"
    token.write_text("test-token", encoding="utf-8")
    session = Mock()
    session.post.side_effect = requests.ConnectionError("https://token-secret@host SECRET")
    with pytest.raises(PublishError) as error:
        TelemetryPublisher(TELEMETRY_URL, token, session=session, sleep=Mock()).publish({})
    assert session.post.call_count == 2 and "SECRET" not in str(error.value)
    assert "token-secret" not in str(error.value)


@pytest.mark.parametrize("configuration", [
    {"telemetry_url": "http://example.invalid/api/cardputer/v1/telemetry/trading"},
    {"telemetry_url": "https://user:secret@example.invalid/api/cardputer/v1/telemetry/trading"},
    {"telemetry_url": "https://example.invalid/api/cardputer/v1/telemetry/trading?token=secret"},
    {"telemetry_url": "https://example.invalid/\napi/cardputer/v1/telemetry/trading"},
    {"interval_seconds": 1}, {"request_timeout_seconds": float("nan")}, {"token": "secret"},
])
def test_invalid_configuration_is_rejected_without_echoing_secrets(tmp_path, configuration, capsys):
    config = tmp_path / "config.json"
    config.write_text(json.dumps(configuration), encoding="utf-8")
    with pytest.raises(ValueError):
        Config.load(config)
    assert main(["--config", str(config), "--once"]) == 2
    assert "secret" not in capsys.readouterr().err


def test_missing_publish_token_causes_no_request(tmp_path):
    session = Mock()
    with pytest.raises(PublishError, match="token file is unavailable"):
        TelemetryPublisher(TELEMETRY_URL, tmp_path / "missing-token", session=session).publish({})
    session.post.assert_not_called()


def test_shared_fixture_is_synthetic_and_matches_complete_exporter_shape(reports):
    root, now, _ = reports
    fixture = json.loads((Path(__file__).parents[1] / "cardputer_monitor/fixtures/trading.json").read_text(encoding="utf-8"))
    actual = SnapshotCollector(root).collect(now=now)
    assert set(fixture) == set(actual)
    assert [row["status"] for row in fixture["robots"]] == ["NORMAL", "RESTRICTED", "STOPPED", "UNAVAILABLE"]
    assert [row["id"] for row in fixture["robots"]] == [row["id"] for row in actual["robots"]]
    assert all(set(row) == set(actual["robots"][index]) for index, row in enumerate(fixture["robots"]))

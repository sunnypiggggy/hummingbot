import copy
import json
import sqlite3
import ssl
import threading
import time
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from cardputer_monitor import mqtt_reporter as reporter
from cardputer_monitor.collector import ROBOTS, STATUS_SCHEMA, SnapshotCollector
from live_guard.trading_status import evaluate_status, gate_row


def wait_until(predicate, timeout=2):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(.005)
    assert predicate(), "background condition was not reached"


def sample(directory, now=100000):
    value = SnapshotCollector(directory).collect(now=now)
    for robot in value["robots"]:
        robot["history"] = reporter._empty_histories(now)[robot["id"]]
    return value


class Samples:
    def __init__(self, directory, *, blocked=None, entered=None):
        self.directory = directory
        self.blocked, self.entered = blocked, entered
        self.count = 0
        self.thread_ids = []

    def collect(self, *, now):
        self.count += 1
        self.thread_ids.append(threading.get_ident())
        if self.entered:
            self.entered.set()
        if self.blocked:
            assert self.blocked.wait(5)
        result = sample(self.directory, now)
        result["sample_id"] = f"sample-{self.count}"
        return result


class Message:
    def __init__(self, topic, payload, *, rc=0):
        self.topic, self.payload, self.rc = topic, payload, rc
        self.acknowledged = False

    def is_published(self):
        return self.acknowledged


class FakeBroker:
    def __init__(self):
        self.available = True
        self.ack = True
        self.force_failure = False
        self.sent = []
        self.clients = []
        self.max_pending = 0

    def client(self, config):
        instance = FakeClient(self)
        self.clients.append(instance)
        return instance

    @property
    def snapshots(self):
        return [message.payload for message in self.sent if message.topic == reporter.SNAPSHOT_TOPIC]


class FakeClient:
    def __init__(self, broker):
        self.broker = broker
        self.ready = False
        self.pending = []
        self.sock = SimpleNamespace(close=Mock())
        self.disconnected = False

    def connect(self, host, port, keepalive):
        self.connection = (host, port, keepalive)
        if not self.broker.available:
            raise OSError("must-not-log-password")
        return 0

    def loop(self, timeout):
        if self.broker.force_failure:
            return 7
        if not self.ready:
            self.ready = True
            self.on_connect(self, None, None, 0, None)
        if self.broker.ack:
            for message in self.pending:
                message.acknowledged = True
            self.pending.clear()
        return 0

    def publish(self, topic, payload, qos, retain):
        assert qos == 1 and retain is True
        message = Message(topic, payload, rc=15 if self.pending else 0)
        if message.rc == 0:
            self.pending.append(message)
            self.broker.max_pending = max(self.broker.max_pending, len(self.pending))
            self.broker.sent.append(message)
        return message

    def socket(self):
        return self.sock

    def disconnect(self):
        self.disconnected = True
        return 0


def publisher(directory, *, broker=None, samples=None, **kwargs):
    broker = broker or FakeBroker()
    samples = samples or Samples(directory)
    instance = reporter.MqttReportPublisher(
        directory, reporter.MqttReportConfig(enabled=True),
        client_factory=broker.client, collector_factory=lambda: samples,
        wall_clock=lambda: 100000, retry_min=.02, retry_max=.05,
        ack_timeout=.15, **kwargs,
    )
    return instance, broker, samples


def test_disabled_default_performs_no_io_import_client_or_thread(tmp_path):
    factory = Mock(side_effect=AssertionError("must not initialize"))
    instance = reporter.MqttReportPublisher.from_environment(
        tmp_path, environment={"CARDPUTER_MQTT_HOST": "invalid://secret"},
        client_factory=factory, collector_factory=factory,
    )
    assert instance.enabled is False
    assert instance.start() is False
    assert instance.notify_report_ready() is False
    assert instance.flush_once(.01) is False
    assert instance.stop(.01) is True
    assert instance._thread is None
    factory.assert_not_called()


@pytest.mark.parametrize("field,value", [
    ("CARDPUTER_MQTT_ENABLED", "maybe"),
    ("CARDPUTER_MQTT_HOST", "mqtt://user:secret@host"),
    ("CARDPUTER_MQTT_HOST", "host\nsecret"),
    ("CARDPUTER_MQTT_HOST", ""),
    ("CARDPUTER_MQTT_PORT", "invalid"),
    ("CARDPUTER_MQTT_PORT", "65536"),
    ("CARDPUTER_MQTT_PORT", "0"),
    ("CARDPUTER_MQTT_CREDENTIALS_FILE", ""),
    ("CARDPUTER_PRICE_PROVIDER", "account-trading-api"),
])
def test_bad_environment_fails_closed_without_secret_logs(tmp_path, caplog, field, value):
    instance = reporter.MqttReportPublisher.from_environment(
        tmp_path, environment={"CARDPUTER_MQTT_ENABLED": "true", field: value})
    assert instance.enabled is False and instance.start() is False
    assert "configuration_invalid" in caplog.text
    assert "secret" not in caplog.text and "account-trading" not in caplog.text


@pytest.mark.parametrize("raw", [None, b"not-json-secret", b"x" * 4097,
    b'{"username":"test","password":"secret","extra":1}',
    b'{"username":false,"password":"secret"}',
    b'{"username":"test","password":""}',
    b'{"username":"test","password":"line\\nsecret"}',
])
def test_bad_credentials_are_isolated_and_flush_does_not_claim_success(tmp_path, caplog, raw):
    secret = tmp_path / "credentials.json"
    if raw is not None:
        secret.write_bytes(raw)
    instance = reporter.MqttReportPublisher(
        tmp_path, reporter.MqttReportConfig(enabled=True, credentials_file=secret),
        collector_factory=lambda: Samples(tmp_path), wall_clock=lambda: 100000,
        retry_min=.05, retry_max=.05,
    )
    try:
        assert instance.start() is True
        assert instance.flush_once(.5) is False
        assert instance._thread.is_alive()
        assert "configuration_unavailable" in caplog.text
        assert "secret" not in caplog.text and "not-json" not in caplog.text
    finally:
        assert instance.stop(.5)


def test_bad_ca_does_not_attempt_connection(tmp_path, monkeypatch, caplog):
    secret = tmp_path / "credentials.json"
    secret.write_text('{"username":"test","password":"never-log"}')
    ca = tmp_path / "invalid-ca.pem"
    ca.write_text("invalid-private-content")
    import paho.mqtt.client as mqtt
    factory = Mock(side_effect=AssertionError("must validate TLS first"))
    monkeypatch.setattr(mqtt, "Client", factory)
    instance = reporter.MqttReportPublisher(
        tmp_path, reporter.MqttReportConfig(enabled=True, credentials_file=secret, ca_file=ca),
        collector_factory=lambda: Samples(tmp_path), wall_clock=lambda: 100000,
    )
    try:
        assert instance.flush_once(.5) is False
        factory.assert_not_called()
        assert "never-log" not in caplog.text and "private-content" not in caplog.text
    finally:
        assert instance.stop(.5)


def test_mqtt_factory_pins_protocol_identity_tls_will_and_bounded_queue(tmp_path, monkeypatch):
    secret = tmp_path / "credentials.json"
    secret.write_text('{"username":"test-publisher","password":"dummy-password"}')
    import paho.mqtt.client as mqtt
    client = Mock()
    constructor = Mock(return_value=client)
    monkeypatch.setattr(mqtt, "Client", constructor)
    result = reporter._mqtt_client(reporter.MqttReportConfig(enabled=True, credentials_file=secret))
    assert result is client
    assert constructor.call_args.args == (mqtt.CallbackAPIVersion.VERSION2,)
    assert constructor.call_args.kwargs == {
        "client_id": "hummingbot-cardputer-report", "protocol": mqtt.MQTTv311,
        "clean_session": True, "reconnect_on_failure": False,
    }
    context = client.tls_set_context.call_args.args[0]
    assert context.verify_mode == ssl.CERT_REQUIRED and context.check_hostname is True
    assert context.minimum_version >= ssl.TLSVersion.TLSv1_2
    client.tls_insecure_set.assert_not_called()
    client.will_set.assert_called_once_with(reporter.AVAILABILITY_TOPIC, b"offline", qos=1, retain=True)
    client.max_queued_messages_set.assert_called_once_with(1)
    client.max_inflight_messages_set.assert_called_once_with(1)
    assert client.connect_timeout == 3


def test_publish_ack_then_online_and_bounded_normal_offline(tmp_path):
    instance, broker, samples = publisher(tmp_path)
    try:
        assert instance.flush_once(2) is True
        wait_until(lambda: any(message.payload == b"online" for message in broker.sent))
        assert broker.sent[0].topic == reporter.SNAPSHOT_TOPIC
        assert len(broker.snapshots) == 1 and samples.count == 1
        body = json.loads(broker.snapshots[0])
        assert body["schema_version"] == 1 and body["source_id"] == "hummingbot-main"
        assert [r["id"] for r in body["robots"]] == [f"{s}:{p}" for s,p,_,_ in ROBOTS]
        assert [r["quote_asset"] for r in body["robots"]] == ["FDUSD", "FDUSD", "USDT", "USDT"]
        health = instance.health()
        assert health["connected"] and health["published_count"] == 1
        assert health["last_sample_id"] == "sample-1" and health["pending"] == 0
        assert health["last_published_at"] == 100000
    finally:
        assert instance.stop(1)
    assert broker.sent[-1].payload == b"offline"
    assert broker.sent[-1].acknowledged
    assert broker.clients[0].disconnected
    assert broker.max_pending == 1


def test_flush_uses_report_signal_without_creating_duplicate_cycle(tmp_path):
    instance, broker, samples = publisher(tmp_path)
    try:
        assert instance.start() and instance.notify_report_ready()
        assert instance.flush_once(2)
        assert instance.flush_once(.01)
        assert samples.count == 1 and len(broker.snapshots) == 1
        instance.notify_report_ready()
        assert instance.flush_once(2)
        assert samples.count == 2 and len(broker.snapshots) == 2
    finally:
        instance.stop(1)


def test_notify_is_immediate_and_many_refreshes_coalesce_while_collecting(tmp_path):
    release, entered = threading.Event(), threading.Event()
    samples = Samples(tmp_path, blocked=release, entered=entered)
    instance, broker, _ = publisher(tmp_path, samples=samples)
    try:
        assert instance.start() and instance.notify_report_ready()
        assert entered.wait(1)
        began = time.monotonic()
        for _ in range(10000):
            assert instance.notify_report_ready()
        assert time.monotonic() - began < .2
        assert instance.health()["pending"] == 1
        release.set()
        wait_until(lambda: instance.health()["pending"] == 0)
        assert samples.count == 2
        assert len(broker.snapshots) == 1
        assert json.loads(broker.snapshots[0])["sample_id"] == "sample-2"
        assert all(thread != threading.get_ident() for thread in samples.thread_ids)
    finally:
        release.set()
        instance.stop(1)


def test_offline_refreshes_keep_latest_generation_not_old_backlog(tmp_path, caplog):
    broker = FakeBroker()
    broker.available = False
    instance, _, samples = publisher(tmp_path, broker=broker)
    try:
        instance.start()
        for i in range(1, 8):
            instance.notify_report_ready()
            wait_until(lambda: samples.count >= i)
        assert instance.health()["pending"] == 1 and not broker.snapshots
        broker.available = True
        wait_until(lambda: len(broker.snapshots) == 1)
        assert json.loads(broker.snapshots[0])["sample_id"] == "sample-7"
        assert "must-not-log-password" not in caplog.text
        assert broker.max_pending == 1
    finally:
        instance.stop(1)


def test_uncertain_ack_retries_identical_bytes_id_and_original_timestamps(tmp_path):
    broker = FakeBroker()
    broker.ack = False
    instance, _, samples = publisher(tmp_path, broker=broker)
    try:
        instance.start()
        instance.notify_report_ready()
        wait_until(lambda: len(broker.snapshots) >= 2)
        broker.ack = True
        wait_until(lambda: instance.health()["pending"] == 0)
        assert samples.count == 1
        assert len(set(broker.snapshots)) == 1
        assert json.loads(broker.snapshots[0])["collected_at"] == 100000
        assert json.loads(broker.snapshots[0])["sample_id"] == "sample-1"
    finally:
        instance.stop(1)


def test_new_snapshot_supersedes_uncertain_frame_before_reconnect(tmp_path):
    broker = FakeBroker()
    broker.ack = False
    instance, _, samples = publisher(tmp_path, broker=broker)
    try:
        instance.start()
        instance.notify_report_ready()
        wait_until(lambda: len(broker.snapshots) == 1)
        instance.notify_report_ready()
        wait_until(lambda: samples.count == 2)
        broker.force_failure = True
        wait_until(lambda: instance.state == "offline")
        broker.force_failure = False
        broker.ack = True
        wait_until(lambda: instance.health()["pending"] == 0)
        ids = [json.loads(raw)["sample_id"] for raw in broker.snapshots]
        assert ids[0] == "sample-1" and ids[-1] == "sample-2"
        assert all(sample_id == "sample-2" for sample_id in ids[ids.index("sample-2"):])
        assert broker.max_pending == 1
    finally:
        instance.stop(1)


def test_stop_is_bounded_during_a_blocked_dns_tls_connect(tmp_path):
    entered, release = threading.Event(), threading.Event()
    broker = FakeBroker()

    def factory(config):
        client = broker.client(config)
        def connect(*args, **kwargs):
            entered.set()
            assert release.wait(5)
            raise OSError("connection unavailable")
        client.connect = connect
        return client

    instance = reporter.MqttReportPublisher(
        tmp_path, reporter.MqttReportConfig(enabled=True), client_factory=factory,
        collector_factory=lambda: Samples(tmp_path), wall_clock=lambda: 100000,
    )
    try:
        instance.start()
        instance.notify_report_ready()
        assert entered.wait(1)
        began = time.monotonic()
        assert instance.stop(.03) is False
        assert time.monotonic() - began < .15
        assert instance.notify_report_ready() is False
    finally:
        release.set()
        wait_until(lambda: not instance._thread.is_alive())


def test_source_risk_pause_history_and_unknown_keys_are_not_published(tmp_path):
    snapshot = sample(tmp_path)
    history = snapshot["robots"][0]["history"]
    history["profit_points"][1]["value"] = 0.0
    history["price_points"][1]["value"] = 61000
    history["status_coverage_start_at"] = 1
    history["status_coverage_end_at"] = 100000
    history["pauses"] = [{"start_at": 1, "end_at": 100000, "scope": "sell", "reason": "private"}]
    history["extra_private_metadata"] = "do-not-export"
    original = copy.deepcopy(snapshot)
    encoded = reporter.encode_snapshot(snapshot)
    clean = json.loads(encoded)["robots"][0]["history"]
    assert clean["pauses"] == []
    assert clean["status_coverage_start_at"] is None and clean["status_coverage_end_at"] is None
    assert clean["profit_points"] == history["profit_points"]
    assert clean["price_points"] == history["price_points"]
    assert "extra_private_metadata" not in clean
    assert b"do-not-export" not in encoded and b"private" not in encoded
    assert snapshot == original


@pytest.mark.parametrize("case", ["oversize", "nan", "infinity", "identity", "wrong_source", "partial", "clock", "curve_size"])
def test_invalid_frames_never_publish_and_flush_reports_failure(tmp_path, case):
    source = sample(tmp_path)
    if case == "oversize":
        source["robots"][0]["blockers"] = ["汉" * 65536]
    elif case in ("nan", "infinity"):
        source["robots"][0]["profit"]["4h"] = float("nan" if case == "nan" else "inf")
    elif case == "identity":
        source["robots"][0]["id"] = "grid:ETH-FDUSD"
    elif case == "wrong_source":
        source["source_id"] = "another-publisher"
    elif case == "partial":
        source["robots"].pop()
    elif case == "clock":
        source["collected_at"] = False
    else:
        source["robots"][0]["history"]["price_points"] *= 2
    samples = SimpleNamespace(collect=lambda **kwargs: source)
    instance, broker, _ = publisher(tmp_path, samples=samples)
    try:
        assert instance.flush_once(.5) is False
        assert broker.sent == [] and instance._thread.is_alive()
    finally:
        instance.stop(1)


def test_corrupt_optional_history_degrades_only_curves(tmp_path, monkeypatch):
    class BrokenHistory:
        def __init__(self, *args, **kwargs):
            pass
        def collect(self, **kwargs):
            raise KeyError("raw-secret-archive")
    monkeypatch.setattr(reporter, "HistoryCollector", BrokenHistory)
    broker = FakeBroker()
    instance = reporter.MqttReportPublisher(tmp_path, reporter.MqttReportConfig(enabled=True),
        client_factory=broker.client, wall_clock=lambda: 100000)
    try:
        assert instance.flush_once(2) is True
        body = json.loads(broker.snapshots[0])
        assert len(body["robots"]) == 4
        assert all(len(row["history"]["profit_points"]) == 73 for row in body["robots"])
        assert instance.health()["last_error_type"] == "history_unavailable"
    finally:
        instance.stop(1)


def test_canonical_wal_reads_on_worker_close_before_market_and_publish(tmp_path, monkeypatch):
    now = int(time.time())
    observed = now - 10
    writer = sqlite3.connect(tmp_path / "telegram_outbox.sqlite")
    writer.execute("PRAGMA journal_mode=WAL")
    writer.execute("CREATE TABLE profit_snapshot(strategy TEXT,pair TEXT,observed_at REAL,"
        "mtm_quote REAL,equity REAL,drawdown_pct REAL,payload_json TEXT,"
        "PRIMARY KEY(strategy,pair,observed_at))")
    status_rows = []
    for strategy, pair, _, bot in ROBOTS:
        for i in range(73):
            writer.execute("INSERT INTO profit_snapshot VALUES (?,?,?,?,?,?,?)", (
                strategy, pair, observed - 43200 + i * 600, i * .25, 200, .4,
                '{"schema":"telegram-profit-snapshot-v2"}',
            ))
        status_rows.append(evaluate_status(generated_at=datetime.fromtimestamp(observed, timezone.utc).isoformat(),
            strategy=strategy, bot=bot, pair=pair, process_running=True, phase="ACTIVE",
            gates=[gate_row("v22_weekly_buy_gate"), gate_row("controller_application_gate")]))
    writer.commit()
    (tmp_path / "trading_status.json").write_text(json.dumps({"schema":STATUS_SCHEMA,
        "generated_at":status_rows[0]["generated_at"],"robots":status_rows}), encoding="utf-8")
    real_connect = sqlite3.connect
    open_readers, queries, reader_threads, price_requests = [], [], [], []

    class Connection:
        def __init__(self, connection):
            self.connection = connection
            open_readers.append(self)
            connection.set_trace_callback(queries.append)
        def execute(self, *args):
            return self.connection.execute(*args)
        def close(self):
            self.connection.close()
            open_readers.remove(self)

    def connect(database, **kwargs):
        assert database.endswith("?mode=ro") and kwargs["uri"] is True
        reader_threads.append(threading.get_ident())
        return Connection(real_connect(database, **kwargs))

    class Prices:
        def __init__(self, **kwargs):
            self.session = Mock()
            self.session.close.side_effect = lambda: price_requests.append("closed")
        def samples(self, pair, start, end):
            assert open_readers == [], "market HTTP must not hold a SQLite transaction"
            price_requests.append(pair)
            return [(observed - 43200 + i * 600, 100 + i) for i in range(73)]

    monkeypatch.setattr(sqlite3, "connect", connect)
    monkeypatch.setattr(reporter, "PublicKlinePrices", Prices)
    broker = FakeBroker()
    factory = broker.client
    def client_factory(config):
        assert not open_readers, "MQTT must not hold a SQLite transaction"
        return factory(config)
    instance = reporter.MqttReportPublisher(tmp_path,
        reporter.MqttReportConfig(enabled=True, price_provider="binance-public"),
        client_factory=client_factory, wall_clock=lambda: now)
    try:
        assert instance.flush_once(2)
        body = json.loads(broker.snapshots[0])
        assert body["collected_at"] == now
        for row in body["robots"]:
            assert row["status"] == "NORMAL"
            assert row["status_observed_at"] == row["profit_observed_at"] == observed
            assert row["profit"]["4h"] == 6
            assert row["history"]["profit_observed_at"] == observed
            assert row["history"]["profit_points"][0]["value"] == 0
            assert row["history"]["profit_points"][-1]["value"] == 18
            assert row["history"]["price_observed_at"] == observed
        assert len(broker.snapshots[0]) > 1024 and len(broker.snapshots[0]) <= 65536
        assert all(thread != threading.get_ident() for thread in reader_threads)
        assert len(reader_threads) == 2
        assert all(query.startswith(("BEGIN", "SELECT")) for query in queries)
        assert price_requests == [pair for _,pair,_,_ in ROBOTS]
    finally:
        instance.stop(1)
        writer.close()
    assert price_requests[-1] == "closed" and not open_readers


def test_initial_worker_failure_isolated_and_stop_is_idempotent(tmp_path, caplog):
    factory = Mock(side_effect=RuntimeError("raw-private-source"))
    instance = reporter.MqttReportPublisher(tmp_path, reporter.MqttReportConfig(enabled=True), collector_factory=factory)
    assert instance.flush_once(.5) is False
    assert instance.stop(.5) and instance.stop(.5)
    assert "worker_unavailable" in caplog.text and "raw-private" not in caplog.text

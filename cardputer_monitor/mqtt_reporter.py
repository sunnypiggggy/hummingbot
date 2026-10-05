"""Optional, read-only MQTT export owned by the existing report process.

Only the daemon worker touches SQLite, market HTTP, credentials, TLS or MQTT.
The report loop hands it a coalescing signal after its canonical writes finish.
"""
from __future__ import annotations

import json
import logging
import math
import ssl
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping

from .collector import ROBOTS, SOURCE_ID, SnapshotCollector
from .history import HistoryCollector, MAX_POINTS, PublicKlinePrices, STEP_SECONDS, WINDOW_SECONDS


SNAPSHOT_TOPIC = "cardputer/v1/trading/hummingbot-main/snapshot"
AVAILABILITY_TOPIC = "cardputer/v1/trading/hummingbot-main/availability"
CLIENT_ID = "hummingbot-cardputer-report"
MAX_PAYLOAD_BYTES = 65536
HISTORY_FIELDS = (
    "start_at", "end_at", "profit_observed_at", "price_observed_at",
    "window_complete", "profit_points", "price_points", "data_state",
)


class ConfigurationError(ValueError):
    """The message is a fixed category, never a credential or source error."""


@dataclass(frozen=True)
class MqttReportConfig:
    enabled: bool = False
    host: str = "sh.sunnypiggy.top"
    port: int = 8883
    credentials_file: Path = Path("/run/secrets/cardputer-report-mqtt.json")
    ca_file: Path | None = None
    price_provider: str = "none"
    price_timeout: float = 5.0

    @classmethod
    def from_environment(cls, environment: Mapping[str, str]) -> "MqttReportConfig":
        enabled = environment.get("CARDPUTER_MQTT_ENABLED", "false").strip().lower()
        if enabled in ("", "false", "0", "no", "off"):
            # Disabled means no credential reads, dependency import or thread.
            return cls()
        if enabled not in ("true", "1", "yes", "on"):
            raise ConfigurationError("enabled_invalid")
        host = environment.get("CARDPUTER_MQTT_HOST", cls.host).strip()
        if not host or len(host) > 253 or any(c.isspace() or ord(c) < 33 or c in "/@?#\\" for c in host):
            raise ConfigurationError("host_invalid")
        try:
            port = int(environment.get("CARDPUTER_MQTT_PORT", "8883"))
        except (TypeError, ValueError, OverflowError):
            raise ConfigurationError("port_invalid") from None
        if not 1 <= port <= 65535:
            raise ConfigurationError("port_invalid")
        provider = environment.get("CARDPUTER_PRICE_PROVIDER", "none").strip()
        if provider not in ("none", "binance-public"):
            raise ConfigurationError("price_provider_invalid")
        credentials = environment.get("CARDPUTER_MQTT_CREDENTIALS_FILE", str(cls.credentials_file)).strip()
        ca_file = environment.get("CARDPUTER_MQTT_CA_FILE", "").strip()
        if not credentials or "\0" in credentials or "\0" in ca_file:
            raise ConfigurationError("path_invalid")
        return cls(True, host, port, Path(credentials), Path(ca_file) if ca_file else None, provider)


def _credentials(path: Path) -> tuple[str, str]:
    try:
        with path.open("rb") as source:
            raw = source.read(4097)
        if len(raw) > 4096:
            raise ConfigurationError("credentials_invalid")
        document = json.loads(raw)
        if not isinstance(document, dict) or set(document) != {"username", "password"}:
            raise ConfigurationError("credentials_invalid")
        username, password = document["username"], document["password"]
        for value, limit in ((username, 128), (password, 512)):
            if (not isinstance(value, str) or not value or len(value) > limit
                    or any(ord(c) < 33 or ord(c) > 126 for c in value)):
                raise ConfigurationError("credentials_invalid")
        return username, password
    except (OSError, UnicodeError, ValueError, TypeError):
        raise ConfigurationError("credentials_invalid") from None


def _mqtt_client(config: MqttReportConfig):
    # Lazy import is intentional: default-disabled reports need no MQTT package.
    import paho.mqtt.client as mqtt

    username, password = _credentials(config.credentials_file)
    try:
        context = ssl.create_default_context(cafile=str(config.ca_file) if config.ca_file else None)
        context.minimum_version = ssl.TLSVersion.TLSv1_2
    except (OSError, ssl.SSLError, ValueError):
        raise ConfigurationError("tls_configuration_invalid") from None
    client = mqtt.Client(
        mqtt.CallbackAPIVersion.VERSION2, client_id=CLIENT_ID,
        protocol=mqtt.MQTTv311, clean_session=True, reconnect_on_failure=False,
    )
    client.suppress_exceptions = True
    client.connect_timeout = 3
    client.username_pw_set(username, password)
    client.tls_set_context(context)
    client.will_set(AVAILABILITY_TOPIC, b"offline", qos=1, retain=True)
    client.max_queued_messages_set(1)
    client.max_inflight_messages_set(1)
    return client


def _empty_histories(now: float) -> dict:
    end = int(now)
    ticks = range(end - WINDOW_SECONDS, end + 1, STEP_SECONDS)
    return {
        f"{strategy}:{pair}": {
            "start_at": end - WINDOW_SECONDS, "end_at": end,
            "profit_observed_at": None, "price_observed_at": None,
            "window_complete": False, "data_state": "UNAVAILABLE",
            "profit_points": [{"ts": tick, "value": None} for tick in ticks],
            "price_points": [{"ts": tick, "value": None} for tick in ticks],
            "status_coverage_start_at": None, "status_coverage_end_at": None, "pauses": [],
        } for strategy, pair, _, _ in ROBOTS
    }


class _IsolatedHistory:
    """A corrupt optional archive must not discard valid current status/MTM."""

    def __init__(self, history: HistoryCollector, report_error: Callable[[str], None]):
        self.history, self.report_error = history, report_error

    def collect(self, *, now: float) -> dict:
        try:
            return self.history.collect(now=now)
        except Exception:
            self.report_error("history_unavailable")
            return _empty_histories(now)


def encode_snapshot(snapshot: dict) -> bytes:
    """Whitelist the existing device contract; cloud owns observed pause history."""
    if (not isinstance(snapshot, dict) or type(snapshot.get("schema_version")) is not int
            or snapshot.get("schema_version") != 1
            or snapshot.get("source_id") != SOURCE_ID
            or not isinstance(snapshot.get("sample_id"), str) or not snapshot["sample_id"]):
        raise ValueError("snapshot_invalid")
    collected = snapshot.get("collected_at")
    if (type(collected) not in (int, float) or not math.isfinite(collected) or collected <= WINDOW_SECONDS):
        raise ValueError("snapshot_invalid")
    robots = snapshot.get("robots")
    if not isinstance(robots, list) or len(robots) != len(ROBOTS):
        raise ValueError("snapshot_invalid")
    result = {key: snapshot[key] for key in ("schema_version", "source_id", "sample_id", "collected_at")}
    result["robots"] = []
    for source, (strategy, pair, name, bot_name) in zip(robots, ROBOTS):
        if (not isinstance(source, dict) or source.get("id") != f"{strategy}:{pair}"
                or source.get("name") != name or source.get("pair") != pair
                or source.get("bot_name") != bot_name or source.get("quote_asset") != pair.split("-")[-1]):
            raise ValueError("snapshot_invalid")
        row = dict(source)
        history = source.get("history")
        if history is not None:
            if not isinstance(history, dict) or not all(key in history for key in HISTORY_FIELDS):
                raise ValueError("history_invalid")
            for key in ("profit_points", "price_points"):
                if not isinstance(history[key], list) or len(history[key]) > MAX_POINTS:
                    raise ValueError("history_invalid")
            row["history"] = {key: history[key] for key in HISTORY_FIELDS}
            row["history"].update(status_coverage_start_at=None, status_coverage_end_at=None, pauses=[])
        result["robots"].append(row)
    payload = json.dumps(result, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")
    if len(payload) > MAX_PAYLOAD_BYTES:
        raise ValueError("snapshot_oversize")
    return payload


@dataclass(frozen=True)
class _Frame:
    generation: int
    payload: bytes
    sample_id: str


class MqttReportPublisher:
    """One coalesced refresh, one latest frame and at most one MQTT in-flight."""

    def __init__(self, reports_dir: Path, config: MqttReportConfig | None = None, *, logger=None,
                 client_factory=None, collector_factory=None, wall_clock=time.time,
                 ack_timeout: float = 5, retry_min: float = 1, retry_max: float = 30):
        self.reports_dir = Path(reports_dir)
        self.config = config or MqttReportConfig()
        self.logger = logger or logging.getLogger(__name__)
        self.client_factory = client_factory or _mqtt_client
        self.collector_factory = collector_factory
        self.wall_clock, self.ack_timeout = wall_clock, ack_timeout
        self.retry_min, self.retry_max = retry_min, retry_max
        self._condition = threading.Condition()
        self._stop = threading.Event()
        self._thread = None
        self._requested = self._assembled = self._published = self._failed = 0
        self._shutdown_deadline = 0.0
        self._last_error = None
        self._last_error_logged_at = 0.0
        self._startup_error = False
        self._connected = False
        self._published_count = 0
        self._last_published_at = None
        self.last_sample_id = None
        self.state = "disabled" if not self.config.enabled else "waiting"

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    def health(self) -> dict:
        with self._condition:
            return {
                "requested": self.config.enabled, "enabled": self.enabled,
                "connected": self._connected, "state": self.state,
                "last_sample_id": self.last_sample_id,
                "published_count": self._published_count,
                "last_published_at": self._last_published_at,
                "last_error_type": self._last_error,
                "refresh_count": self._requested,
                "pending": int(self._requested > self._published and not self._stop.is_set()),
            }

    @classmethod
    def from_environment(cls, reports_dir: Path, *, logger=None, environment=None, **kwargs):
        import os
        try:
            config = MqttReportConfig.from_environment(os.environ if environment is None else environment)
        except Exception:
            instance = cls(reports_dir, logger=logger, **kwargs)
            instance._error("configuration_invalid")
            return instance
        return cls(reports_dir, config, logger=logger, **kwargs)

    def _error(self, category: str) -> None:
        now = time.monotonic()
        # Fixed strings only: do not attach exception text, payloads or traceback.
        if category != self._last_error or now - self._last_error_logged_at >= 60:
            try:
                self.logger.warning("Cardputer MQTT: %s", category)
            except Exception:
                pass
            self._last_error_logged_at = now
        self._last_error = category

    def start(self) -> bool:
        if not self.config.enabled or self._stop.is_set():
            return False
        with self._condition:
            if self._thread is not None:
                return self._thread.is_alive()
            try:
                self._thread = threading.Thread(target=self._run, name="cardputer-report-mqtt", daemon=True)
                self._thread.start()
            except Exception:
                self._error("worker_start_failed")
                self.state = "unavailable"
                return False
        return True

    def notify_report_ready(self) -> bool:
        # No IO, collection, network calls or worker-owned locks in this method.
        if not self.config.enabled or self._stop.is_set() or self._thread is None:
            return False
        with self._condition:
            self._requested += 1
            self._condition.notify_all()
        return True

    def flush_once(self, timeout: float = 30) -> bool:
        if not self.start():
            return False
        deadline = time.monotonic() + max(0, timeout)
        with self._condition:
            # The report's update_snapshots may already have signalled this
            # cycle. Flush it; do not silently manufacture a second sample.
            if self._requested == 0:
                self._requested = 1
                self._condition.notify_all()
            generation = self._requested
            while self._published < generation:
                remaining = deadline - time.monotonic()
                if (remaining <= 0 or self._failed >= generation or self._startup_error
                        or self._stop.is_set() or not self._thread.is_alive()):
                    return False
                self._condition.wait(min(remaining, 0.1))
            return True

    def stop(self, timeout: float = 2) -> bool:
        self._shutdown_deadline = time.monotonic() + max(0, timeout)
        self._stop.set()
        with self._condition:
            self._condition.notify_all()
        if self._thread is None:
            return True
        if threading.current_thread() is self._thread:
            return False
        self._thread.join(max(0, self._shutdown_deadline - time.monotonic()))
        return not self._thread.is_alive()

    @staticmethod
    def _close(client, *, graceful: bool = False) -> None:
        if client is None:
            return
        try:
            if graceful:
                client.disconnect()
                client.loop(timeout=0.01)
            sock = client.socket()
            if sock is not None:
                sock.close()
        except Exception:
            pass

    def _run(self) -> None:
        client = prices = collector = None
        frame = flight_frame = flight = availability = None
        connected = False
        online = False
        flight_deadline = connect_deadline = next_connect = 0.0
        confirmed_payload = None
        retry_delay = self.retry_min

        def on_connect(_client, _userdata, _flags, reason, _properties):
            nonlocal connected
            connected = not bool(getattr(reason, "is_failure", reason != 0))
            self._connected = connected

        def on_disconnect(_client, _userdata, _flags, _reason, _properties):
            nonlocal connected
            connected = False
            self._connected = False

        try:
            if self.collector_factory:
                collector = self.collector_factory()
            else:
                prices = PublicKlinePrices(timeout=self.config.price_timeout) if self.config.price_provider == "binance-public" else None
                history = _IsolatedHistory(HistoryCollector(self.reports_dir / "telegram_outbox.sqlite", prices=prices), self._error)
                collector = SnapshotCollector(self.reports_dir, history=history)
            while not self._stop.is_set():
                with self._condition:
                    requested = self._requested
                if requested > self._assembled:
                    self._assembled = requested
                    try:
                        snapshot = collector.collect(now=self.wall_clock())
                        payload = encode_snapshot(snapshot)
                        frame = _Frame(requested, payload, snapshot["sample_id"])
                    except Exception:
                        self._error("snapshot_invalid")
                        with self._condition:
                            self._failed = requested
                            self._condition.notify_all()
                    # A newer signal during slow collection replaces this frame
                    # before sending, without queuing old report generations.
                    with self._condition:
                        if self._requested > self._assembled:
                            continue
                now = time.monotonic()
                if client is None and frame is not None and now >= next_connect:
                    try:
                        client = self.client_factory(self.config)
                        client.on_connect = on_connect
                        client.on_disconnect = on_disconnect
                        rc = client.connect(self.config.host, self.config.port, keepalive=30)
                        if rc != 0:
                            raise RuntimeError("connection_failed")
                        connect_deadline = time.monotonic() + 3
                        self._startup_error = False
                    except Exception as exc:
                        self._startup_error = isinstance(exc, ConfigurationError)
                        self._error("configuration_unavailable" if self._startup_error else "connection_unavailable")
                        self._close(client)
                        client = None
                        next_connect = time.monotonic() + retry_delay
                        retry_delay = min(self.retry_max, retry_delay * 2)
                if client is not None:
                    try:
                        if client.loop(timeout=0.02) != 0:
                            raise RuntimeError("network_failed")
                        now = time.monotonic()
                        if not connected and now >= connect_deadline:
                            raise RuntimeError("connection_failed")
                        if connected:
                            retry_delay = self.retry_min
                            if flight is not None and flight.is_published():
                                confirmed_payload = flight_frame.payload
                                self.last_sample_id = flight_frame.sample_id
                                self.state = "published"
                                with self._condition:
                                    self._published = max(self._published, flight_frame.generation)
                                    self._published_count += 1
                                    self._last_published_at = self.wall_clock()
                                    self._condition.notify_all()
                                flight = flight_frame = None
                            if availability is not None and availability.is_published():
                                availability = None
                                online = True
                            if (flight is not None or availability is not None) and now >= flight_deadline:
                                raise RuntimeError("ack_timeout")
                            if flight is None and availability is None:
                                if confirmed_payload is not None and not online:
                                    availability = client.publish(AVAILABILITY_TOPIC, b"online", qos=1, retain=True)
                                    if availability.rc != 0:
                                        raise RuntimeError("publish_failed")
                                    flight_deadline = now + self.ack_timeout
                                elif frame is not None and frame.payload != confirmed_payload:
                                    flight = client.publish(SNAPSHOT_TOPIC, frame.payload, qos=1, retain=True)
                                    if flight.rc != 0:
                                        raise RuntimeError("publish_failed")
                                    flight_frame = frame
                                    flight_deadline = now + self.ack_timeout
                    except Exception:
                        self._error("connection_unavailable")
                        self.state = "offline"
                        self._close(client)
                        client = None
                        connected = online = False
                        self._connected = False
                        flight = flight_frame = availability = None
                        confirmed_payload = None
                        next_connect = time.monotonic() + retry_delay
                        retry_delay = min(self.retry_max, retry_delay * 2)
                with self._condition:
                    if self._requested <= self._assembled and not self._stop.is_set():
                        self._condition.wait(0.05)
        except Exception:
            self._error("worker_unavailable")
            self.state = "unavailable"
        finally:
            # No unbounded network join. If offline cannot be acknowledged,
            # close abruptly so the broker's offline will remains authoritative.
            graceful = False
            if client is not None and connected:
                try:
                    deadline = min(self._shutdown_deadline, time.monotonic() + 0.75)
                    # A just-queued online/snapshot occupies the single slot.
                    # Finish it within the same bounded cleanup budget first.
                    while time.monotonic() < deadline and any(
                        info is not None and not info.is_published() for info in (flight, availability)
                    ):
                        if client.loop(timeout=0.02) != 0:
                            break
                    offline = client.publish(AVAILABILITY_TOPIC, b"offline", qos=1, retain=True)
                    while offline.rc == 0 and time.monotonic() < deadline:
                        if client.loop(timeout=0.02) != 0:
                            break
                        if offline.is_published():
                            graceful = True
                            break
                except Exception:
                    pass
            self._close(client, graceful=graceful)
            if prices is not None:
                try:
                    prices.session.close()
                except Exception:
                    pass
            self.state = "stopped"
            self._connected = False
            with self._condition:
                self._condition.notify_all()

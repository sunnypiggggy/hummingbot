"""Report lifecycle integration; all external network boundaries are replaced."""

import copy
import json
import signal
import sys
import time
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from cardputer_monitor.collector import ROBOTS, SnapshotCollector
from live_guard import dca_live_report as report_module
from live_guard.telegram_notifications import append_event, build_event
from live_guard.trading_status import evaluate_status, gate_row


@pytest.fixture
def reporting(tmp_path, monkeypatch):
    """Keep canonical SQLite, JSON, risk history and outbox writes real."""
    monkeypatch.setenv("TELEGRAM_NOTIFY_ENABLED", "false")
    monkeypatch.setenv("CARDPUTER_MQTT_ENABLED", "true")
    monkeypatch.setenv("GRID_LIVE_STATE_PATH", str(tmp_path / "grid"))
    monkeypatch.setenv("ACCOUNT_INVENTORY_LEDGER_PATH", str(tmp_path / "inventory"))
    monkeypatch.setenv("ETHBTC_RELEASE_FAMILY_PATH", str(tmp_path / "releases"))
    worker = SimpleNamespace(
        schedule=Mock(return_value=[]), poll=Mock(return_value={"active": False}),
        finalize_delivery_receipts=Mock(return_value=0),
    )
    monkeypatch.setattr(report_module, "ParameterReportWorker", Mock(return_value=worker))
    parameter_publisher = SimpleNamespace(publish=Mock(return_value={"catalog_sha256": "a" * 64}))
    monkeypatch.setattr(report_module, "ManagementParameterPublisher", Mock(return_value=parameter_publisher))
    instance = report_module.UnifiedTelegramReporting(bots_path=tmp_path / "bots", dca_state=tmp_path / "dca")
    instance.simple_audit = SimpleNamespace(cycle=Mock())
    now = datetime.fromtimestamp(int(time.time()), timezone.utc)
    robots = []
    for strategy, pair, _, bot in ROBOTS:
        status = evaluate_status(
            generated_at=now.isoformat(), strategy=strategy, pair=pair, bot=bot,
            process_running=True, phase="ACTIVE",
            gates=[gate_row("v22_weekly_buy_gate"), gate_row("controller_application_gate")],
        )
        row = {
            "strategy": strategy, "pair": pair, "quote_asset": pair.split("-")[1],
            "trading_status": status, "profit": {"all_time_mtm_quote": 11.0},
            "equity": 211.0, "drawdown_pct": 0.0,
        }
        anchor = copy.deepcopy(row)
        anchor["profit"]["all_time_mtm_quote"] = 7.0
        instance.outbox.record_profit(anchor, observed_at=now.timestamp() - 4 * 3600)
        robots.append(row)
    instance._grid_cards = lambda current: copy.deepcopy(robots[:2])
    instance._dca_cards = lambda report, current: copy.deepcopy(robots[2:])
    for root in (instance.grid_state, instance.dca_state):
        root.mkdir(parents=True, exist_ok=True)
        (root / "guard_state.json").write_text(json.dumps({"last_success_at": now.timestamp()}))
    yield instance, now
    instance.outbox.close()
    if hasattr(instance, "risk_history"):
        instance.risk_history.close()


def test_disabled_telegram_notifies_only_after_canonical_sources_are_readable(reporting):
    instance, now = reporting
    samples = []

    def report_ready():
        sample = SnapshotCollector(instance.output).collect(now=now.timestamp())
        risk = json.loads((instance.output / "risk_history.json").read_text(encoding="utf-8"))
        assert risk["schema"] == "report-risk-history-v1"
        assert risk["generated_at"] == now.timestamp()
        assert set(risk["permission_coverage"]) == {f"{s}:{p}" for s, p, _, _ in ROBOTS}
        samples.append(sample)

    instance.mqtt_reporter = SimpleNamespace(notify_report_ready=Mock(side_effect=report_ready))
    result = instance.cycle({}, now=now)
    assert result["enabled"] is False and result["sent"] == 0
    assert len(samples) == 1
    assert [r["id"] for r in samples[0]["robots"]] == [f"{s}:{p}" for s, p, _, _ in ROBOTS]
    assert [r["quote_asset"] for r in samples[0]["robots"]] == ["FDUSD", "FDUSD", "USDT", "USDT"]
    for robot in samples[0]["robots"]:
        assert robot["status"] == "NORMAL"
        assert robot["status_observed_at"] == robot["profit_observed_at"] == now.timestamp()
        assert robot["profit"]["4h"] == 4.0 and robot["profit"]["all"] == 11.0
    instance.parameter_worker.poll.assert_called_once()


def test_notify_exception_preserves_archive_outbox_and_telegram_delivery(reporting, caplog):
    instance, now = reporting
    # Risk history starts observing at its epoch; historical input is not
    # silently backfilled on first startup.
    instance.archive_risk_sources(now)
    event = build_event(
        source="test", strategy="grid", bot="grid-live-fdusd-400", pair="BTC-FDUSD",
        mechanism="v22_weekly_buy_gate", transition="TRIGGERED", reason="risk_off",
        correlation_id="cardputer-report-integration",
    )
    append_event(instance.events, event)
    instance.enabled = True
    instance.profit_enabled = False
    instance.client = object()
    instance.outbox.drain = Mock(return_value=1)
    instance.mqtt_reporter = SimpleNamespace(notify_report_ready=Mock(side_effect=RuntimeError("test-only failure")))
    result = instance.cycle({}, now=now)
    assert result["sent"] == 1
    instance.outbox.drain.assert_called_once_with(instance.client)
    instance.parameter_worker.poll.assert_called_once()
    instance.management_parameters.publish.assert_called_once()
    assert instance.outbox.connection.execute("SELECT count(*) FROM outbox").fetchone()[0] >= 1
    assert instance.risk_history.db.execute("SELECT count(*) FROM events").fetchone()[0] >= 1
    assert len(SnapshotCollector(instance.output).collect(now=now.timestamp())["robots"]) == 4
    assert "Cardputer MQTT refresh unavailable (RuntimeError)" in caplog.text
    assert "test-only failure" not in caplog.text


@pytest.fixture
def main_runtime(tmp_path, monkeypatch):
    """Run the real main and report loop without external report collection."""
    collector = SimpleNamespace(collect=Mock(return_value={
        "generated_at": datetime.now(timezone.utc).isoformat(), "report_id": "test-report",
        "bots": [], "warnings": [],
    }))
    telegram = SimpleNamespace(archive_risk_sources=Mock(), cycle=Mock(return_value={
        "retrying": 0, "profit_report_error": "",
    }))
    errors = SimpleNamespace(failure=Mock(), recovered=Mock())
    publisher = SimpleNamespace(
        flush_once=Mock(return_value=True), stop=Mock(), health=Mock(return_value={"connected": True}),
    )
    monkeypatch.setattr(report_module, "DcaLiveReportCollector", Mock(return_value=collector))
    monkeypatch.setattr(report_module, "UnifiedTelegramReporting", Mock(return_value=telegram))
    monkeypatch.setattr(report_module, "RuntimeErrorChannel", Mock(return_value=errors))
    starter = Mock(side_effect=lambda root: publisher if report_module._cardputer_mqtt_requested() else None)
    monkeypatch.setattr(report_module, "_start_cardputer_mqtt", starter)
    previous = {signal.SIGTERM: object(), signal.SIGINT: object()}
    handlers = dict(previous)

    def install_signal(signum, handler):
        old, handlers[signum] = handlers[signum], handler
        return old

    monkeypatch.setattr(report_module.signal, "signal", install_signal)
    monkeypatch.setattr(sys, "argv", ["dca-live-report", "--output-dir", str(tmp_path), "--once"])
    monkeypatch.setenv("CARDPUTER_MQTT_ENABLED", "true")
    return SimpleNamespace(collector=collector, telegram=telegram, errors=errors,
                           publisher=publisher, starter=starter, handlers=handlers, previous=previous)


@pytest.mark.parametrize("healthy,expected", [(True, 0), (False, 1)])
def test_healthcheck_does_not_initialize_report_or_mqtt(tmp_path, monkeypatch, healthy, expected):
    monkeypatch.setenv("CARDPUTER_MQTT_ENABLED", "true")
    monkeypatch.setattr(sys, "argv", ["dca-live-report", "--output-dir", str(tmp_path), "--healthcheck"])
    monkeypatch.setattr(report_module, "healthcheck", lambda directory: healthy)
    constructors = [Mock(side_effect=AssertionError("must not start")) for _ in range(3)]
    for name, constructor in zip(("DcaLiveReportCollector", "UnifiedTelegramReporting", "_start_cardputer_mqtt"), constructors):
        monkeypatch.setattr(report_module, name, constructor)
    assert report_module.main() == expected
    assert all(not constructor.called for constructor in constructors)


@pytest.mark.parametrize("enabled,confirmation,expected", [
    (True, True, 0), (True, False, 1), (True, "exception", 1), (True, "absent", 1), (False, False, 0),
])
def test_once_reports_confirmation_and_always_cleans_up(main_runtime, monkeypatch, enabled, confirmation, expected):
    runtime = main_runtime
    monkeypatch.setenv("CARDPUTER_MQTT_ENABLED", str(enabled).lower())
    if confirmation == "absent":
        runtime.starter.side_effect = None
        runtime.starter.return_value = None
    elif confirmation == "exception":
        runtime.publisher.flush_once.side_effect = RuntimeError("test-only timeout")
    else:
        runtime.publisher.flush_once.return_value = confirmation
    assert report_module.main() == expected
    runtime.telegram.cycle.assert_called_once()
    if enabled and confirmation != "absent":
        runtime.publisher.flush_once.assert_called_once_with(timeout=30)
        runtime.publisher.stop.assert_called_once_with(timeout=2)
    else:
        runtime.publisher.flush_once.assert_not_called()
        runtime.publisher.stop.assert_not_called()
    assert runtime.handlers == runtime.previous


def test_once_failed_canonical_cycle_does_not_claim_publication_success(main_runtime):
    runtime = main_runtime
    runtime.collector.collect.side_effect = RuntimeError("test-only unavailable source")
    assert report_module.main() == 1
    runtime.publisher.flush_once.assert_not_called()
    runtime.publisher.stop.assert_called_once_with(timeout=2)
    runtime.errors.failure.assert_called_once()
    assert runtime.handlers == runtime.previous


def test_sigterm_finishes_current_report_without_interval_sleep_and_stops_publisher(main_runtime, monkeypatch):
    runtime = main_runtime
    monkeypatch.setattr(sys, "argv", ["dca-live-report"])
    source = runtime.collector.collect.return_value

    def stopped_collection():
        runtime.handlers[signal.SIGTERM](signal.SIGTERM, None)
        return source

    runtime.collector.collect.side_effect = stopped_collection
    assert report_module.main() == 0
    runtime.telegram.cycle.assert_called_once()
    runtime.publisher.flush_once.assert_not_called()
    runtime.publisher.stop.assert_called_once_with(timeout=2)
    assert runtime.handlers == runtime.previous


def test_shutdown_failure_does_not_override_success_or_leave_signal_handlers(main_runtime, caplog):
    runtime = main_runtime
    runtime.publisher.stop.side_effect = RuntimeError("test-only shutdown failure")
    assert report_module.main() == 0
    assert runtime.handlers == runtime.previous
    assert "Cardputer MQTT shutdown unavailable (RuntimeError)" in caplog.text
    assert "test-only shutdown failure" not in caplog.text


def test_unexpected_loop_failure_still_stops_publisher_and_restores_signals(main_runtime, monkeypatch):
    runtime = main_runtime
    monkeypatch.setattr(report_module, "_run_report_loop", Mock(side_effect=KeyboardInterrupt))
    with pytest.raises(KeyboardInterrupt):
        report_module.main()
    runtime.publisher.stop.assert_called_once_with(timeout=2)
    assert runtime.handlers == runtime.previous

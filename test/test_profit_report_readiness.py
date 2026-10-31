"""Four-hour readiness uses current market health; durable delivery stays real."""

from copy import deepcopy
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from cardputer_monitor.collector import ROBOTS
from live_guard import dca_live_report as report_module
from live_guard import telegram_notifications as notifications
from live_guard.trading_status import evaluate_status, gate_row


NOW = datetime(2026, 10, 23, 4, 0, 0, 165000, tzinfo=timezone.utc)
SLOT = "2026-10-23T12:00:00+08:00"
MTM_WINDOWS = ("four_hour_mtm_quote", "twenty_four_hour_mtm_quote", "seven_day_mtm_quote")


def source_report(now, market_age):
    bots = []
    for strategy, pair, _, bot in ROBOTS:
        if strategy != "dca":
            continue
        candles = [] if market_age is None else [{
            "timestamp": (now - timedelta(seconds=market_age)).timestamp(), "close": "100000",
        }]
        bots.append(report_module.calculate_pair_report(
            bot_name=bot, pair=pair, rows=[], candles=candles, database_age_seconds=7200,
            now=now, managed_base=Decimal(".01"), managed_base_cost_quote=Decimal("1000"),
        ))
    return {"bots": bots, "warnings": []}


class Client:
    def __init__(self, *, fail_message_once=False):
        self.fail_message_once = fail_message_once
        self.sent = []

    def send_message(self, text):
        if self.fail_message_once:
            self.fail_message_once = False
            raise RuntimeError("test-only Telegram failure")
        self.sent.append(("message", text))
        return str(len(self.sent))

    def send_file(self, path, *, caption, kind):
        self.sent.append((kind, path.name))
        return str(len(self.sent))


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    # Only external workers/rendering/transports are replaced. Snapshot writes,
    # event JSONL, archive, slot state, source cursors and outbox delivery are real.
    monkeypatch.setenv("TELEGRAM_NOTIFY_ENABLED", "false")
    monkeypatch.setenv("GRID_LIVE_STATE_PATH", str(tmp_path / "grid"))
    monkeypatch.setenv("ACCOUNT_INVENTORY_LEDGER_PATH", str(tmp_path / "inventory"))
    monkeypatch.setenv("ETHBTC_RELEASE_FAMILY_PATH", str(tmp_path / "releases"))
    monkeypatch.setattr(report_module, "ParameterReportWorker", lambda **_: SimpleNamespace(
        schedule=Mock(return_value=[]), poll=Mock(return_value={"active": False}),
        finalize_delivery_receipts=Mock(return_value=0),
    ))
    monkeypatch.setattr(report_module, "ManagementParameterPublisher", lambda **_: SimpleNamespace(
        publish=Mock(return_value={"catalog_sha256": "a" * 64}),
    ))
    clock = [NOW.timestamp()]
    monkeypatch.setattr(notifications, "time", SimpleNamespace(time=lambda: clock[0], sleep=lambda _: None))

    def render(item, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"test-image:{item['strategy']}:{item['pair']}".encode())

    renderer = Mock(side_effect=render)
    monkeypatch.setattr(report_module, "render_mobile_profit_card", renderer)
    instances = []

    def card(strategy, pair, bot, current, profit, healthy=True):
        value = profit.get("all_time_mtm_quote")
        return {
            "strategy": strategy, "pair": pair, "quote_asset": pair.split("-")[-1],
            "profit": deepcopy(profit), "equity": None if value is None else 200 + float(value),
            "drawdown_pct": 0, "trading_status": evaluate_status(
                generated_at=current.isoformat(), strategy=strategy, pair=pair, bot=bot,
                process_running=True, phase="ACTIVE", gates=[gate_row(
                    "infrastructure_integrity_breaker", buy_enabled=healthy, sell_enabled=healthy,
                    state="ALLOW" if healthy else "BLOCK", reason="test-source-health",
                )],
            ),
        }

    def make(client=None):
        instance = report_module.UnifiedTelegramReporting(bots_path=tmp_path / "bots", dca_state=tmp_path / "dca")
        instance.simple_audit = SimpleNamespace(cycle=Mock())
        instance.mqtt_reporter = SimpleNamespace(notify_report_ready=Mock())
        instance.enabled = instance.profit_enabled = True
        instance.client = client or Client()
        instance._grid_cards = lambda current: [
            card(strategy, pair, bot, current, {"all_time_mtm_quote": 11})
            for strategy, pair, _, bot in ROBOTS if strategy == "grid"
        ]
        instance._dca_cards = lambda report, current: [
            card("dca", bot["trading_pair"], bot["bot_name"], current,
                 bot["profit"], bot["market_data_healthy"])
            for bot in report["bots"]
        ]
        for root in (instance.grid_state, instance.dca_state):
            root.mkdir(parents=True, exist_ok=True)
            (root / "guard_state.json").write_text(json.dumps({"last_success_at": clock[0]}))
        # Establish the archive epoch before test producers append their events.
        instance.archive_risk_sources(datetime.fromtimestamp(clock[0] - 1, timezone.utc))
        instances.append(instance)
        return instance

    def close(instance):
        instance.outbox.close()
        instance.risk_history.close()
        instances.remove(instance)

    yield SimpleNamespace(make=make, close=close, clock=clock, renderer=renderer)
    for instance in instances:
        instance.outbox.close()
        instance.risk_history.close()


def profit_events(instance):
    if not instance.events.exists():
        return []
    return [row for line in instance.events.read_text(encoding="utf-8").splitlines()
            if (row := json.loads(line)).get("mechanism") == "profit_report"]


@pytest.mark.parametrize("market_age,deferred", [(900, False), (900.165, True), (None, True)])
def test_current_market_boundary_does_not_change_900_second_rule(runtime, market_age, deferred):
    instance = runtime.make()
    source = source_report(NOW, market_age)
    result = instance.cycle(source, now=NOW)
    assert report_module.MAX_DATA_AGE_SECONDS == 900
    assert all(bot["market_data_healthy"] is (not deferred) for bot in source["bots"])
    assert result["profit_report_deferred_pairs"] == (["BTC-USDT", "ETH-USDT"] if deferred else [])
    assert result["profit_report_error"] == ""
    assert instance.outbox.slot_due(now=NOW) == (deferred, SLOT)
    assert runtime.renderer.call_count == (0 if deferred else 4)
    assert len(profit_events(instance)) == (0 if deferred else 1)
    if deferred:
        assert instance.outbox.connection.execute(
            "SELECT COUNT(*) FROM profit_snapshot WHERE strategy='dca' AND mtm_quote IS NULL"
        ).fetchone()[0] == 2


@pytest.mark.parametrize("restart_while_waiting", [False, True])
def test_fresh_next_cycle_sends_same_slot_once_even_across_restart(runtime, restart_while_waiting):
    instance = runtime.make()
    instance.cycle(source_report(NOW, 900.165), now=NOW)
    if restart_while_waiting:
        runtime.close(instance)
        instance = runtime.make()
    later = NOW + timedelta(seconds=70)
    runtime.clock[0] = later.timestamp()
    result = instance.cycle(source_report(later, 1), now=later)
    assert result["profit_report_deferred_pairs"] == [] and result["sent"] == 5
    assert instance.outbox.slot_due(now=later) == (False, SLOT)
    assert [event["details"]["slot"] for event in profit_events(instance)] == [SLOT]
    assert runtime.renderer.call_count == 4
    runtime.close(instance)
    restarted = runtime.make()
    assert restarted.cycle(source_report(later, 1), now=later)["sent"] == 0
    assert runtime.renderer.call_count == 4 and len(profit_events(restarted)) == 1


def test_waiting_keeps_risk_parameter_archive_and_mqtt_paths_running(runtime):
    instance = runtime.make()
    for mechanism, transition in (("v22_weekly_buy_gate", "TRIGGERED"),
                                   ("parameter_update", "MODEL_CUTOVER_PREWARMED")):
        notifications.append_event(instance.events, notifications.build_event(
            source="test", strategy="dca", bot="test-bot", pair="BTC-USDT", mechanism=mechanism,
            transition=transition, reason="test-only event", correlation_id=mechanism,
            occurred_at=NOW.isoformat(),
        ))
    result = instance.cycle(source_report(NOW, 900.165), now=NOW)
    assert result["sent"] == 2 and result["pending"] == 0
    assert instance.outbox.slot_due(now=NOW) == (True, SLOT)
    assert instance.outbox.connection.execute("SELECT COUNT(*) FROM profit_snapshot").fetchone()[0] == 4
    assert instance.risk_history.db.execute("SELECT COUNT(*) FROM events").fetchone()[0] == 2
    assert (instance.output / "trading_status.json").is_file()
    assert (instance.output / "risk_history.json").is_file()
    instance.mqtt_reporter.notify_report_ready.assert_called_once()
    instance.management_parameters.publish.assert_called_once()
    assert instance.parameter_worker.schedule.call_count == 2
    instance.parameter_worker.poll.assert_called_once()
    instance.parameter_worker.finalize_delivery_receipts.assert_called_once()
    assert not profit_events(instance) and not runtime.renderer.called


def test_missing_historical_windows_do_not_defer_fresh_current_valuation(runtime):
    instance = runtime.make()
    source = source_report(NOW, 1)
    for bot in source["bots"]:
        bot["profit"].update({field: None for field in MTM_WINDOWS})
    result = instance.cycle(source, now=NOW)
    assert result["profit_report_deferred_pairs"] == [] and result["sent"] == 5
    assert instance.outbox.slot_due(now=NOW) == (False, SLOT)
    event, = profit_events(instance)
    dca = [row for row in event["details"]["robots"] if row["strategy"] == "dca"]
    assert all(row["profit"]["all_time_mtm_quote"] == "0.00" for row in dca)
    assert all(all(row["profit"].get(field) is None for field in MTM_WINDOWS) for row in dca)


def test_render_failure_keeps_slot_due_and_retries_generation(runtime):
    instance = runtime.make()
    render = runtime.renderer.side_effect
    runtime.renderer.side_effect = RuntimeError("test-only PNG failure")
    result = instance.cycle(source_report(NOW, 1), now=NOW)
    assert result["profit_report_error"] == "RuntimeError: test-only PNG failure"
    assert instance.outbox.slot_due(now=NOW) == (True, SLOT)
    assert not profit_events(instance)
    instance.parameter_worker.poll.assert_called_once()
    runtime.renderer.side_effect = render
    result = instance.cycle(source_report(NOW, 1), now=NOW)
    assert result["profit_report_error"] == "" and result["sent"] == 5
    assert len(profit_events(instance)) == 1


def test_transport_retry_survives_restart_without_regenerating_same_slot(runtime):
    instance = runtime.make(Client(fail_message_once=True))
    first = instance.cycle(source_report(NOW, 1), now=NOW)
    assert first["sent"] == 4 and first["retrying"] == 1
    assert instance.outbox.slot_due(now=NOW) == (False, SLOT)
    runtime.close(instance)
    restarted = runtime.make()
    later = NOW + timedelta(seconds=70)
    runtime.clock[0] = later.timestamp()
    result = restarted.cycle(source_report(later, 1), now=later)
    assert result["sent"] == 1 and result["pending"] == result["retrying"] == 0
    assert runtime.renderer.call_count == 4 and len(profit_events(restarted)) == 1
    assert restarted.outbox.connection.execute(
        "SELECT kind,status,attempts FROM outbox ORDER BY id"
    ).fetchall() == [("message", "sent", 1), *[("photo", "sent", 0)] * 4]


def test_existing_sent_slot_is_preserved_without_replay(runtime):
    instance = runtime.make()
    instance.outbox.enqueue(event_id="already-sent-noon-report", kind="message", text="existing sent evidence")
    assert instance.outbox.drain(instance.client) == 1
    instance.outbox.mark_slot(SLOT)
    runtime.close(instance)
    restarted = runtime.make()
    result = restarted.cycle(source_report(NOW, 1), now=NOW)
    assert result["sent"] == 0 and result["profit_report_deferred_pairs"] == []
    assert restarted.outbox.slot_due(now=NOW) == (False, SLOT)
    assert restarted.outbox.connection.execute("SELECT COUNT(*) FROM outbox WHERE status='sent'").fetchone()[0] == 1
    assert not profit_events(restarted) and not runtime.renderer.called

import json
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import requests
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from live_guard.telegram_notifications import (
    MARKDOWN_MESSAGE_PREFIX,
    RuntimeErrorChannel,
    TelegramChannelClient,
    TelegramOutbox,
    append_event,
    build_event,
    dust_metric,
    dust_usdt_display,
    explain_event,
    format_event,
    hermes_recovery_prompt,
    phase_display,
    render_mobile_profit_card,
    runtime_error_fingerprint_text,
    runtime_error_lines,
    sanitize_runtime_error,
    system_health_display,
    trade_mode_display,
)
from live_guard.telegram_parameter_report import (
    _resolve_report_inputs,
    build_parameter_attachments,
)
from live_guard.dca_live_guard import Guard as DcaGuard
from live_guard.grid_live_guard import Guard as GridGuard
from live_guard.dca_live_report import ParameterReportWorker, UnifiedTelegramReporting


def event(**overrides):
    values = {
        "source": "test", "strategy": "grid", "bot": "grid-btc",
        "pair": "BTC-FDUSD", "mechanism": "v22_weekly_buy_gate",
        "transition": "TRIGGERED", "reason": "risk_off",
        "correlation_id": "source-event-1",
    }
    values.update(overrides)
    return build_event(**values)


def test_model_cutover_lifecycle_events_are_channel_safe():
    for transition in (
        "MODEL_CUTOVER_PREWARMED", "MODEL_CUTOVER_STABLE",
        "MODEL_CUTOVER_PRECHECK_FAILED", "MODEL_FOLD_ACTIVATED",
        "MODEL_RETENTION_PRUNED", "MODEL_RETENTION_FAILED",
    ):
        value = event(
            mechanism="parameter_update", transition=transition,
            reason="candidate isolation status",
        )
        assert transition in format_event(value)


def test_runtime_error_channel_alerts_once_then_reports_recovery(tmp_path):
    events = tmp_path / "events.jsonl"
    state = tmp_path / "runtime-errors.json"
    channel = RuntimeErrorChannel(
        event_path=events, state_path=state, source="grid-live-guard",
        strategy="grid", bot="grid-live-fdusd-400",
        pair="BTC-FDUSD,ETH-FDUSD",
    )
    assert channel.failure(
        "guard_cycle", ConnectionError("connection reset by peer"),
        trading_impact="automatic retry", now=100,
    )
    assert not channel.failure(
        "guard_cycle", ConnectionError("connection reset by peer"),
        trading_impact="automatic retry", now=102,
    )
    restarted = RuntimeErrorChannel(
        event_path=events, state_path=state, source="grid-live-guard",
        strategy="grid", bot="grid-live-fdusd-400",
        pair="BTC-FDUSD,ETH-FDUSD",
    )
    assert not restarted.failure(
        "guard_cycle", ConnectionError("connection reset by peer"),
        trading_impact="automatic retry", now=104,
    )
    assert restarted.recovered("guard_cycle", now=110)
    assert not restarted.recovered("guard_cycle", now=111)
    rows = [json.loads(line) for line in events.read_text(encoding="utf-8").splitlines()]
    assert [row["transition"] for row in rows] == ["ERROR_OCCURRED", "ERROR_RECOVERED"]
    assert rows[1]["details"]["occurrences"] == 3
    assert rows[1]["details"]["duration_seconds"] == 10
    assert "运行错误告警" in format_event(rows[0])
    assert "运行错误已恢复" in format_event(rows[1])


def test_timestamped_runtime_log_lines_share_one_error_episode(tmp_path):
    events = tmp_path / "events.jsonl"
    state = tmp_path / "runtime-errors.json"
    channel = RuntimeErrorChannel(
        event_path=events, state_path=state, source="grid-live-guard",
        strategy="grid", bot="grid-live-fdusd-400",
        pair="BTC-FDUSD,ETH-FDUSD",
    )
    first = (
        "2026-08-23T15:32:17.005298890Z 15:32:17 - strategy_v2_base - "
        "Live grid cycle failed: 'infrastructure'"
    )
    second = (
        "2026-08-23T15:32:19.002862618Z 15:32:19 - strategy_v2_base - "
        "Live grid cycle failed: 'infrastructure'"
    )
    assert runtime_error_fingerprint_text(first) == (
        "strategy_v2_base - Live grid cycle failed: 'infrastructure'"
    )
    assert runtime_error_fingerprint_text(first) == runtime_error_fingerprint_text(second)
    assert channel.failure("container_log:grid", first, trading_impact="monitor", now=100)
    assert not channel.failure("container_log:grid", second, trading_impact="monitor", now=102)
    assert channel.recovered("container_log:grid", now=110)
    rows = [json.loads(line) for line in events.read_text(encoding="utf-8").splitlines()]
    assert [row["transition"] for row in rows] == ["ERROR_OCCURRED", "ERROR_RECOVERED"]
    assert rows[1]["details"]["occurrences"] == 2
    saved = json.loads(state.read_text(encoding="utf-8"))
    assert saved["schema"] == "runtime-error-channel-v3"


def test_runtime_error_fingerprint_keeps_distinct_failure_reasons():
    prefix = "2026-08-23T15:32:19.002862618Z 15:32:19 - strategy - "
    assert runtime_error_fingerprint_text(prefix + "cycle failed: stale contract") != (
        runtime_error_fingerprint_text(prefix + "cycle failed: insufficient balance")
    )


def test_short_runtime_error_is_suppressed_and_kept_for_four_hour_summary(tmp_path):
    events = tmp_path / "events.jsonl"
    state = tmp_path / "runtime-errors.json"
    channel = RuntimeErrorChannel(
        event_path=events, state_path=state, source="grid-live-guard",
        strategy="grid", bot="grid-live-fdusd-400",
        pair="BTC-FDUSD,ETH-FDUSD",
    )
    assert not channel.failure(
        "guard_cycle", ConnectionError("remote end closed connection"),
        trading_impact="automatic retry", now=100, notify_after_seconds=6,
    )
    assert not channel.failure(
        "guard_cycle", ConnectionError("remote end closed connection"),
        trading_impact="automatic retry", now=104, notify_after_seconds=6,
    )
    assert not channel.recovered("guard_cycle", now=105)
    assert not events.exists()
    saved = json.loads(state.read_text(encoding="utf-8"))
    assert saved["history"][-1]["suppressed_as_transient"] is True
    assert saved["history"][-1]["occurrences"] == 2
    assert saved["history"][-1]["duration_seconds"] == 5


def test_runtime_error_alerts_only_after_delay_then_reports_recovery(tmp_path):
    events = tmp_path / "events.jsonl"
    channel = RuntimeErrorChannel(
        event_path=events, state_path=tmp_path / "state.json",
        source="grid-live-guard", strategy="grid", bot="grid",
        pair="BTC-FDUSD,ETH-FDUSD",
    )
    assert not channel.failure(
        "guard_cycle", "api unavailable", trading_impact="retry",
        now=200, notify_after_seconds=6,
    )
    assert channel.failure(
        "guard_cycle", "api unavailable", trading_impact="retry",
        now=206, notify_after_seconds=6,
    )
    assert channel.recovered("guard_cycle", now=208)
    rows = [json.loads(line) for line in events.read_text(encoding="utf-8").splitlines()]
    assert [row["transition"] for row in rows] == ["ERROR_OCCURRED", "ERROR_RECOVERED"]
    assert rows[0]["details"]["notification_delay_seconds"] == 6
    assert rows[0]["details"]["occurrences"] == 2


def test_internal_get_retry_is_recorded_without_telegram_event(tmp_path):
    state = tmp_path / "state.json"
    events = tmp_path / "events.jsonl"
    channel = RuntimeErrorChannel(
        event_path=events, state_path=state, source="grid-live-guard",
        strategy="grid", bot="grid", pair="BTC-FDUSD,ETH-FDUSD",
    )
    channel.record_transient_recovery(
        "guard_cycle", "RemoteDisconnected", occurrences=1,
        duration_seconds=0.2, now=300,
    )
    assert not events.exists()
    saved = json.loads(state.read_text(encoding="utf-8"))
    assert saved["history"][-1]["alert_sent"] is False
    assert saved["history"][-1]["suppressed_as_transient"] is True


def test_four_hour_report_counts_only_recent_suppressed_grid_recoveries(tmp_path):
    reporting = UnifiedTelegramReporting.__new__(UnifiedTelegramReporting)
    reporting.grid_state = tmp_path
    now = datetime.fromtimestamp(20_000, timezone.utc)
    (tmp_path / "runtime_error_state.json").write_text(json.dumps({
        "history": [
            {"component": "guard_cycle", "recovered_at": 19_000,
             "occurrences": 2, "summary": "recent", "suppressed_as_transient": True},
            {"component": "guard_cycle", "recovered_at": 1_000,
             "occurrences": 5, "summary": "old", "suppressed_as_transient": True},
            {"component": "report_cycle", "recovered_at": 19_500,
             "occurrences": 3, "summary": "other", "suppressed_as_transient": True},
            {"component": "guard_cycle", "recovered_at": 19_800,
             "occurrences": 1, "summary": "alerted", "suppressed_as_transient": False},
        ]
    }), encoding="utf-8")
    summary = reporting._grid_transport_summary(now)
    assert summary["recovered_episodes"] == 1
    assert summary["retry_attempts"] == 2
    assert summary["last_reason"] == "recent"


def test_runtime_order_error_recovers_only_after_quiet_window(tmp_path):
    channel = RuntimeErrorChannel(
        event_path=tmp_path / "events.jsonl", state_path=tmp_path / "state.json",
        source="strategy", strategy="dca", bot="dca", pair="ETH-USDT",
    )
    channel.failure("order_submission", "balance insufficient", trading_impact="rejected", now=10)
    assert not channel.recover_if_quiet("order_submission", quiet_seconds=300, now=309)
    assert channel.recover_if_quiet("order_submission", quiet_seconds=300, now=310)


def test_runtime_error_is_redacted_and_truncated():
    raw = (
        "GET https://api.telegram.org/bot123456:SECRET/sendMessage?signature=abc "
        "api_key=visible password=hunter2 token=secret " + "x" * 1000
    )
    value = sanitize_runtime_error(raw)
    assert "123456:SECRET" not in value
    assert "visible" not in value
    assert "hunter2" not in value
    assert "token=secret" not in value
    assert "[REDACTED]" in value
    assert len(value) <= 600


def test_runtime_error_notification_failure_never_breaks_producer(tmp_path):
    channel = RuntimeErrorChannel(
        event_path=tmp_path / "directory", state_path=tmp_path / "state.json",
        source="test", strategy="dca", bot="bot", pair="BTC-USDT",
    )
    channel.event_path.mkdir()
    assert not channel.failure(
        "cycle", RuntimeError("boom"), trading_impact="none", now=10
    )


def test_outbox_health_exposes_sanitized_delivery_retry(tmp_path):
    outbox = TelegramOutbox(tmp_path / "outbox.sqlite", channel_id="channel")
    outbox.enqueue(event_id="event", kind="message", text="alert")
    outbox.connection.execute(
        "UPDATE outbox SET attempts=2,last_error=? WHERE event_id='event'",
        ("token=very-secret transport timeout",),
    )
    outbox.connection.commit()
    health = outbox.health()
    assert health["retrying"] == 1
    assert health["max_attempts"] == 2
    assert "very-secret" not in health["last_error"]
    outbox.close()


def test_runtime_log_filter_selects_errors_and_redacts_secrets():
    lines = [
        "2026-08-13 INFO strategy started successfully",
        "2026-08-13 ERROR order rejected token=secret balance insufficient",
        "2026-08-13 client_order_tracker Order abc has failed at Binance",
        "2026-08-13 INFO validation completed with 0 errors",
    ]
    found = runtime_error_lines(lines)
    assert len(found) == 2
    assert "order rejected" in found[0]
    assert "secret" not in found[0]


def test_correlated_event_id_is_stable_across_guard_cycles():
    first = event(occurred_at="2026-08-08T00:00:00+00:00")
    second = event(occurred_at="2026-08-08T00:00:02+00:00")
    assert first["event_id"] == second["event_id"]
    assert first["occurred_at"] != second["occurred_at"]


def test_inventory_message_uses_actual_active_runtime_instead_of_latched_claim():
    value = event(
        strategy="account", bot="shared-binance-spot", pair="BTC-USDT",
        mechanism="account_inventory",
        transition="INVENTORY_UNATTRIBUTED_DETECTED",
        correlation_id="inventory-episode-btc",
        details={
            "inventory_phase": "DETECTED",
            "confirmation": {"cycles": 1, "confirmed": False},
            "runtime": {
                "trading_normal": True,
                "active_order_count": 24,
                "robots": {
                    "grid-live-fdusd-400": {"phase": "ACTIVE", "running": True},
                    "dca-live-btcusdt-200": {"phase": "ACTIVE", "running": True},
                    "dca-live-ethusdt-200": {"phase": "ACTIVE", "running": True},
                },
            },
        },
    )
    message = format_event(value)
    assert "当前机器人为 ACTIVE，交易正常" in message
    assert "活动订单=24" in message
    assert "机器人保持停止和 LATCHED" not in message


def test_inventory_dust_classified_is_supported_and_explains_no_trade():
    value = event(
        strategy="account", bot="shared-binance-spot", pair="ETH-USDT",
        mechanism="account_inventory", transition="INVENTORY_DUST_CLASSIFIED",
        reason="unattributed_inventory_below_exchange_minimum",
        action="record_dust_no_order", correlation_id="inventory-episode-eth",
        details={
            "inventory_phase": "DUST", "quantity": "0.0022",
            "tradable_quantity": "0.0022", "estimated_notional": "4.24",
            "minimum_notional": "5", "dust_reason": "below_minimum_notional",
            "runtime": {"trading_normal": True, "robots": {}},
        },
    )
    message = format_event(value)
    assert "Dust" in message
    assert "record_dust_no_order" in message


def test_inventory_deficit_message_uses_shortage_fields_not_dust_fields():
    value = event(
        strategy="account", bot="shared-binance-spot", pair="ETH-USDT",
        mechanism="account_inventory", transition="INVENTORY_OWNERSHIP_DEFICIT",
        reason="confirmed_strategy_ownership_exceeds_exchange_balance",
        action="fail_closed_no_liquidation", correlation_id="deficit-eth",
        details={
            "inventory_phase": "RECOVERED",
            "ownership_deficit": "0.0048",
            "deficit_quantity": "0.0048",
            "deficit_estimated_notional": "9.14",
            "tradable_quantity": "0",
            "minimum_notional": "5",
            "dust_reason": "rounded_quantity_zero",
            "confirmation": {"cycles": 0, "confirmed": False},
            "deficit_confirmation": {"cycles": 3, "confirmed": True},
            "runtime": {"trading_normal": False, "robots": {}},
        },
    )
    message = format_event(value)
    assert "缺口数量/预估金额：0.0048 / 9.14 USDT" in message
    assert "确认：3/3，已确认=True" in message
    assert "库存阶段：RECOVERED" not in message
    assert "可成交数量" not in message
    assert "Dust 原因" not in message


def test_outbox_ingestion_and_restart_are_idempotent():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source = root / "events.jsonl"
        append_event(source, event())
        append_event(source, event())
        outbox = TelegramOutbox(root / "outbox.sqlite")
        assert outbox.ingest(source) == 1
        assert outbox.ingest(source) == 0
        assert outbox.health()["pending"] == 1
        outbox.close()
        restarted = TelegramOutbox(root / "outbox.sqlite")
        assert restarted.ingest(source) == 0
        assert restarted.health()["pending"] == 1
        restarted.close()


class FakeResponse:
    def raise_for_status(self):
        return None

    def json(self):
        return {"ok": True, "result": {"message_id": 7}}


class FakeSession:
    def __init__(self):
        self.urls = []

    def post(self, url, **kwargs):
        self.urls.append(url)
        return FakeResponse()


def test_notification_client_only_uses_one_way_send_apis():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        token = root / "token"
        token.write_text("notify-token", encoding="utf-8")
        photo = root / "card.png"
        photo.write_bytes(b"png")
        session = FakeSession()
        client = TelegramChannelClient(token, "-100123", session=session)
        client.send_message("hello")
        client.send_file(photo, kind="photo")
        client.send_file(photo, kind="document")
        assert [url.rsplit("/", 1)[-1] for url in session.urls] == [
            "sendMessage", "sendPhoto", "sendDocument",
        ]
        assert all("getUpdates" not in url for url in session.urls)


def test_profit_report_is_sent_as_telegram_markdown():
    class MarkdownSession:
        def __init__(self):
            self.data = None

        def post(self, url, **kwargs):
            self.data = kwargs["data"]
            return FakeResponse()

    report = build_event(
        source="test", strategy="grid+dca", bot="4 robots",
        pair="BTC-FDUSD", mechanism="profit_report",
        transition="PROFIT_REPORT", reason="scheduled",
        details={
            "slot": "2026-08-09T12:00:00+08:00",
            "robots": [{
                "strategy": "grid", "pair": "BTC-FDUSD", "quote_asset": "FDUSD",
                "profit": {
                    "four_hour_mtm_quote": None,
                    "twenty_four_hour_mtm_quote": None,
                    "seven_day_mtm_quote": None,
                    "all_time_mtm_quote": 7.665,
                },
                "runtime_transport": {"recovered_episodes": 2},
            }],
        },
    )
    text = format_event(report)
    assert text.startswith(MARKDOWN_MESSAGE_PREFIX)
    assert "*GRID · BTC-FDUSD*" in text
    assert "- 累计：`+7.6650 FDUSD`" in text
    assert "Guard连接瞬时恢复（4h）：`2 次`，交易权限未受影响" in text
    with tempfile.TemporaryDirectory() as directory:
        token = Path(directory) / "token"
        token.write_text("notify-token", encoding="utf-8")
        session = MarkdownSession()
        TelegramChannelClient(token, "-100123", session=session).send_message(text)
        assert session.data["parse_mode"] == "Markdown"
        assert MARKDOWN_MESSAGE_PREFIX not in session.data["text"]


def test_telegram_transport_error_never_leaks_bot_token():
    class FailingSession:
        def post(self, url, **kwargs):
            raise requests.Timeout(f"timeout for {url}")

    with tempfile.TemporaryDirectory() as directory:
        token = Path(directory) / "token"
        token.write_text("super-secret-notify-token", encoding="utf-8")
        client = TelegramChannelClient(token, "-100123", session=FailingSession())
        try:
            client.send_message("hello")
        except RuntimeError as exc:
            assert "super-secret-notify-token" not in str(exc)
        else:
            raise AssertionError("transport failure should raise")


def test_send_failure_is_persisted_for_exponential_retry_without_trade_exception():
    class FailingClient:
        def send_message(self, text):
            raise TimeoutError("telegram timeout")

    with tempfile.TemporaryDirectory() as directory:
        outbox = TelegramOutbox(Path(directory) / "outbox.sqlite", channel_id="-100123")
        outbox.enqueue(event_id="event-1", kind="message", text="alert")
        assert outbox.drain(FailingClient(), now=10**12) == 0
        row = outbox.connection.execute(
            "SELECT status,attempts,last_error,next_attempt FROM outbox"
        ).fetchone()
        assert row[0] == "pending" and row[1] == 1
        assert "TimeoutError" in row[2] and row[3] > 0
        outbox.close()


def test_beijing_slot_only_sends_current_period_once():
    with tempfile.TemporaryDirectory() as directory:
        outbox = TelegramOutbox(Path(directory) / "outbox.sqlite")
        now = datetime(2026, 8, 8, 1, 59, tzinfo=timezone.utc)  # 09:59 BJT
        due, slot = outbox.slot_due(now=now)
        assert due and slot.startswith("2026-08-08T08:00:00")
        outbox.mark_slot(slot)
        assert outbox.slot_due(now=now) == (False, slot)
        outbox.close()


def test_mobile_card_is_one_robot_1440_by_3200_png_with_gate_table():
    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory) / "grid_btc.png"
        render_mobile_profit_card({
            "strategy": "grid", "pair": "BTC-FDUSD", "quote_asset": "FDUSD",
            "generated_at_bjt": "2026-08-08T12:00:00+08:00", "data_age_seconds": 3,
            "profit": {"four_hour_mtm_quote": 1, "twenty_four_hour_mtm_quote": 2,
                       "seven_day_mtm_quote": -1, "all_time_mtm_quote": 8},
            "equity": 208, "peak_equity": 210, "drawdown_pct": 0.95,
            "owned_base": "0.002", "fees_quote": 0, "buys": 3, "sells": 2,
            "phase": "ACTIVE", "v22_gate": "放行", "fomc_gate": "放行",
            "active_runtime": {"orders": 4},
            "trading_status": {
                "system_health": "HEALTHY", "trade_mode": "NORMAL",
                "trading_normal": True, "phase": "ACTIVE",
                "final_permissions": {"buy_enabled": True, "sell_enabled": True},
                "runtime_generation": "a" * 64, "release_sha256": "b" * 64,
                "model_week": 38, "cutover_phase": "ACTIVE",
                "gate_statuses": [{
                    "mechanism": "v22_weekly_buy_gate", "label": "v22周度BUY门",
                    "enabled": True, "applicable": True, "health": "HEALTHY",
                    "state": "RISK_ON", "buy_enabled": True, "sell_enabled": True,
                    "reason": "long_risk_gate_clear",
                }],
            },
            "equity_series": [200, 203, 208], "drawdown_series": [0, 1, 0.95],
        }, output)
        with Image.open(output) as image:
            assert image.size == (1440, 3200)
            assert image.format == "PNG"


def test_mobile_card_dust_text_only_contains_usdt_valuation():
    value = dust_usdt_display({
        "quantity": "0.002160779092935063",
        "estimated_notional": "3.983318",
    })
    assert value == "约 3.9833 USDT"
    assert "ETH" not in value
    assert "0.002160" not in value


def test_mobile_card_status_and_phase_are_clear_chinese():
    assert system_health_display("HEALTHY") == "健康"
    assert system_health_display("FAILED") == "故障"
    assert trade_mode_display("NORMAL") == "正常交易"
    assert trade_mode_display("REENTRY") == "等待重入，暂停交易"
    assert phase_display("ACTIVE") == "正常交易"
    assert phase_display("LATCHED") == "已锁存"


def test_every_robot_card_has_asset_scoped_dust_row():
    assert dust_metric("ETH-FDUSD", {
        "quantity": "0.002160779092935063",
        "estimated_notional": "3.983318",
    }) == ("共享账户 ETH Dust", "约 3.9833 USDT")
    assert dust_metric("BTC-USDT", None) == ("共享账户 BTC Dust", "无")


def test_grid_parameter_report_requires_hash_bound_six_png_evidence(monkeypatch):
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        evidence = root / "grid-evidence"
        evidence.mkdir()
        parameter_sha = "a" * 64
        images = []
        for pair in ("BTC-FDUSD", "ETH-FDUSD"):
            for window in ("360d", "2026_jan_feb", "2026_may_june"):
                path = evidence / f"{pair}_{window}.png"
                Image.new("RGB", (1440, 2400), "white").save(path)
                import hashlib
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                images.append({"pair": pair, "window": window, "path": path.name,
                               "sha256": digest})
        (evidence / "grid_parameter_evidence_manifest.json").write_text(json.dumps({
            "schema": "grid-parameter-mobile-evidence-v1",
            "parameter_sha256": parameter_sha,
            "evidence_complete": True,
            "images": images,
        }), encoding="utf-8")
        monkeypatch.setenv("GRID_PARAMETER_EVIDENCE_ROOT", str(evidence))
        value = event(
            mechanism="parameter_update", transition="PARAMETER_ACTIVATED",
            parameter_sha256=parameter_sha,
            details={"report_request": "grid_360d", "candidate": {"levels": 10}},
        )
        attachments = build_parameter_attachments(
            value, release_root=root / "release", output_root=root / "out",
        )
        assert len(attachments) == 6
        assert sum(item["kind"] == "photo" for item in attachments) == 6
        assert {item["kind"] for item in attachments} == {"photo"}
        assert all(item["evidence_complete"] is True for item in attachments)
        assert all(Path(item["path"]).is_file() for item in attachments)


def test_v22_parameter_report_is_twelve_pngs_without_documents(tmp_path):
    release = ROOT / "release_packages" / "ethbtc-forced-exit"
    value = event(
        mechanism="parameter_update",
        transition="MODEL_APPROVAL_PENDING",
        strategy="grid+dca",
        model_sha256="a" * 64,
        details={"report_request": "v22_png_windows"},
    )
    attachments = build_parameter_attachments(
        value, release_root=release, output_root=tmp_path / "out",
    )
    assert len(attachments) == 12
    assert {item["kind"] for item in attachments} == {"photo"}
    assert all(Path(item["path"]).is_file() for item in attachments)
    captions = "\n".join(str(item["caption"]) for item in attachments)
    assert "过去360天" in captions
    assert "2026年1–2月" in captions
    assert "2026年5–6月" in captions


def test_pending_weekly_candidate_uses_family_replay_evidence(tmp_path):
    release_sha = "c" * 64
    family = tmp_path / "ethbtc-forced-exit"
    candidate = family / "releases" / release_sha
    (candidate / "shadow_package").mkdir(parents=True)
    (candidate / "shadow_package" / "shadow_lock.json").write_text(
        json.dumps({"effective_start": 123}), encoding="utf-8",
    )
    evidence = family / "evidence"
    evidence.mkdir(parents=True)
    for name in ("summary.json", "audit_series.csv.gz", "risk_intervals.csv"):
        (evidence / name).write_bytes(b"evidence")

    identity_root, evidence_root = _resolve_report_inputs(
        family, {"release_sha256": release_sha},
    )

    assert identity_root == candidate
    assert evidence_root == family


def test_evidence_receipt_is_written_only_after_all_photos_are_sent(tmp_path):
    outbox = TelegramOutbox(tmp_path / "outbox.sqlite", channel_id="-100-test")
    worker = ParameterReportWorker(
        root=tmp_path / "telegram", release_root=tmp_path / "release", outbox=outbox,
    )
    event_id = "e" * 64
    attachments = []
    for index in range(12):
        path = tmp_path / f"photo-{index}.png"
        path.write_bytes(f"photo-{index}".encode())
        import hashlib
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        attachments.append({"path": str(path), "kind": "photo", "sha256": digest,
                            "evidence_complete": True})
        outbox.enqueue(event_id=event_id, kind="photo", text=str(index),
                       file_path=path, file_sha256=digest)
    outbox.enqueue(event_id=event_id, kind="message", text="approval")
    marker = worker.jobs / f"{event_id}.done"
    marker.write_text(json.dumps({
        "event_id": event_id, "release_sha256": "c" * 64,
        "model_sha256": "b" * 64, "parameter_sha256": "",
        "report_request": "v22_png_windows", "attachments": attachments,
    }), encoding="utf-8")
    assert worker.finalize_delivery_receipts() == 0
    outbox.connection.execute(
        "UPDATE outbox SET status='sent',sent_at=1,telegram_message_id='42' WHERE event_id=?",
        (event_id,),
    )
    outbox.connection.commit()
    assert worker.finalize_delivery_receipts() == 1
    receipt = json.loads((
        worker.receipts / f"{'c' * 64}.json"
    ).read_text(encoding="utf-8"))
    assert receipt["expected_photo_count"] == 12
    assert len(receipt["photo_sha256"]) == 12
    assert receipt["delivery_receipt_sha256"]
    worker.executor.shutdown(wait=True)
    outbox.close()


def test_latched_prompt_contains_binding_but_no_secret_or_command():
    value = event(
        mechanism="infrastructure_integrity_breaker", transition="LATCHED",
        release_sha256="a" * 64, model_sha256="b" * 64,
        requires_manual_action=True, phase_to="LATCHED",
    )
    prompt = hermes_recovery_prompt(value)
    assert value["event_id"] in prompt
    assert "release=" in prompt and "model=" in prompt
    assert "Token" not in prompt and "reset --" not in prompt


def test_grid_integrity_failure_uses_stable_event_id_across_contract_refreshes(tmp_path):
    guard = GridGuard.__new__(GridGuard)
    guard.notification_path = tmp_path / "events.jsonl"
    details = {
        "runtime_healthy": False,
        "reason": "fail_closed:no signed weekly model covers BTC-FDUSD",
        "gate": {"generated_at": "first", "release_sha256": "a" * 64,
                 "model_sha256": "b" * 64, "pairs": {}},
    }
    guard._emit_notification("grid_xgboost_risk_gate_transition", details)
    details["gate"]["generated_at"] = "second"
    guard._emit_notification("grid_xgboost_risk_gate_transition", details)
    rows = [json.loads(line) for line in guard.notification_path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 2
    assert rows[0]["event_id"] == rows[1]["event_id"]


def test_risk_event_message_explains_cause_and_follow_up_impact_in_one_line():
    value = event(
        strategy="grid", mechanism="strategy_drawdown_breaker",
        transition="TRIGGERED", reason="drawdown 3.01% >= 3.00%",
        action="cancel_orders_and_flatten",
    )
    explanation = explain_event(value)
    assert explanation.startswith("解释：由于单机器人权益从峰值回撤达到策略保护条件")
    assert "原始原因：drawdown 3.01% >= 3.00%" in explanation
    assert "停止新增订单、撤销活动订单并退出归属库存" in explanation
    assert explanation.count("\n") == 0 and explanation.endswith("。")
    message = format_event(value)
    assert explanation in message
    assert message.index("原因：") < message.index("解释：") < message.index("动作：")

    multiline = event(reason="first line\nsecond line")
    assert "\n" not in explain_event(multiline)


def test_grid_drawdown_exit_message_has_threshold_cooldown_dust_and_reentry_conditions():
    value = event(
        strategy="grid", pair="ETH-FDUSD",
        mechanism="strategy_drawdown_breaker", transition="COOLDOWN",
        reason="drawdown=3.03%", trigger_value="0.0303437469709546",
        threshold="0.03", phase_to="COOLDOWN", action="risk_exit_complete",
        details={
            "cooldown_until": 1787482112,
            "remaining_dust_base": "0.00009420",
            "remaining_dust_quote": "0.225644706",
            "quote_asset": "FDUSD",
            "auto_reentry_enabled": True,
            "healthy_cycles_required": 3,
        },
    )
    message = format_event(value)
    assert "触发值：3.0344%" in message
    assert "保护阈值：3.0000%" in message
    assert "冷却结束：2026-08-23 18:48:32（北京时间）" in message
    assert "剩余 Dust：0.00009420 ETH / 约 0.225644706 FDUSD" in message
    assert "自动恢复：已开启" in message
    assert "连续3个健康周期" in message
    assert "v22/FOMC及其他风控门全部放行" in message


def test_capital_budget_alert_explains_that_trading_is_not_blocked():
    explanation = explain_event({
        "mechanism": "capital_budget_gate",
        "strategy": "dca",
        "transition": "TRIGGERED",
        "reason": "insufficient_quote_budget",
        "details": {"free_quote": "18.68", "required_quote": "190"},
    })
    assert "仅发送告警" in explanation
    assert "不会关闭 BUY/SELL" in explanation
    assert "18.68" in explanation


def test_recovered_explanation_never_claims_all_gates_are_open():
    value = event(
        strategy="dca", mechanism="v22_weekly_buy_gate",
        transition="RECOVERED", reason="weekly signal returned risk-on",
        phase_from="REENTRY", phase_to="ACTIVE",
        details={
            "buy_enabled": True,
            "effective_buy_enabled": True,
            "effective_sell_enabled": True,
            "recovery_phase": "ACTIVE",
            "execution_applied": True,
            "controller_update_status": "applied",
        },
    )
    explanation = explain_event(value)
    assert "v22 BUY 门=放行（Risk-On）" in explanation
    assert "DCA 聚合门 BUY=放行、SELL=放行" in explanation
    assert "交易正常" in explanation


def test_dca_v22_recovered_but_another_gate_still_blocks_buy_is_explicit():
    value = event(
        strategy="dca", mechanism="v22_weekly_buy_gate",
        transition="RECOVERED", reason="long_risk_gate_clear",
        phase_from="RISK_OFF", phase_to="RISK_ON",
        details={
            "buy_enabled": True,
            "effective_buy_enabled": False,
            "effective_sell_enabled": True,
            "recovery_phase": "COOLDOWN",
            "execution_applied": True,
            "controller_update_status": "unchanged",
        },
    )
    explanation = explain_event(value)
    assert "v22 BUY 门=放行（Risk-On）" in explanation
    assert "DCA 聚合门 BUY=阻止、SELL=放行" in explanation
    assert "不会创建新的普通 BUY executor" in explanation
    assert "交易处于受限状态" in explanation


def test_dca_v22_legacy_event_without_aggregate_state_does_not_guess():
    value = event(
        strategy="dca", mechanism="v22_weekly_buy_gate",
        transition="RECOVERED", reason="long_risk_gate_clear",
    )
    explanation = explain_event(value)
    assert "v22 BUY 门=放行（Risk-On）" in explanation
    assert "不能据此判断交易是否正常" in explanation


def test_dca_v22_producer_preserves_aggregate_and_controller_state():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "events.jsonl"
        guard = DcaGuard.__new__(DcaGuard)
        guard.notification_path = path
        guard._emit_notification("v22_gate_transition", {
            "bot": "dca-live-btcusdt-200",
            "pair": "BTC-USDT",
            "buy_enabled": True,
            "sell_enabled": True,
            "effective_buy_enabled": False,
            "effective_sell_enabled": True,
            "recovery_phase": "COOLDOWN",
            "execution_applied": True,
            "controller_update_status": "unchanged",
            "reason": "long_risk_gate_clear",
            "correlation_id": "v22-btc-event",
        })
        value = json.loads(path.read_text(encoding="utf-8"))
        assert value["phase_from"] == "RISK_OFF"
        assert value["phase_to"] == "RISK_ON"
        assert value["details"]["effective_buy_enabled"] is False
        assert value["details"]["effective_sell_enabled"] is True
        assert value["details"]["recovery_phase"] == "COOLDOWN"
        assert value["details"]["execution_applied"] is True


def test_hourly_v22_event_id_refresh_is_not_a_state_transition():
    dca_source = (ROOT / "live_guard" / "dca_live_guard.py").read_text(encoding="utf-8")
    grid_source = (ROOT / "live_guard" / "grid_live_guard.py").read_text(encoding="utf-8")
    changed_block = dca_source.split("v22_changed = (", 1)[1].split(
        "controller_result", 1
    )[0]
    assert "v22_buy_enabled" in changed_block
    assert "v22_event_id" not in changed_block
    assert "previous_risk_off" in grid_source
    assert "previous_event_ids" not in grid_source


def test_integrity_latched_explanation_requires_manual_recovery():
    value = event(
        mechanism="infrastructure_integrity_breaker", transition="LATCHED",
        reason="contract hash mismatch", requires_manual_action=True,
    )
    explanation = explain_event(value)
    assert "完整性检查失败" in explanation
    assert "Hermes 或 OCI 的人工复核" in explanation


def test_dca_recovery_lifecycle_emits_each_transition_and_manual_reentry_prompt():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "events.jsonl"
        guard = DcaGuard.__new__(DcaGuard)
        guard.notification_path = path
        guard.auto_reentry_enabled = False
        guard.state = {"v22_observation": {
            "release_sha256": "a" * 64, "model_sha256": "b" * 64,
        }}
        recovery = {
            "phase": "EXITING", "mechanism": "strategy_loss_breaker",
            "reason": "loss", "triggered_at": 100, "trigger_value": "-16",
        }
        guard._emit_notification("recoverable_breaker_triggered", {
            "bot": "dca-live-btcusdt-200", "pair": "BTC-USDT",
            "recovery": recovery,
        })
        cooldown = {**recovery, "phase": "COOLDOWN"}
        guard._emit_notification("recoverable_exit_complete", {
            "bot": "dca-live-btcusdt-200", "pair": "BTC-USDT",
            "recovery": cooldown,
        })
        reentry = {**recovery, "phase": "REENTRY"}
        guard._emit_notification("recoverable_reentry_ready", {
            "bot": "dca-live-btcusdt-200", "pair": "BTC-USDT",
            "recovery": reentry,
        })
        values = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        assert [value["transition"] for value in values] == [
            "TRIGGERED", "EXITING", "EXIT_COMPLETE", "COOLDOWN", "REENTRY",
        ]
        assert values[-1]["requires_manual_action"] is True

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
    TelegramChannelClient,
    TelegramOutbox,
    append_event,
    build_event,
    format_event,
    hermes_recovery_prompt,
    render_mobile_profit_card,
)
from live_guard.telegram_parameter_report import build_parameter_attachments
from live_guard.dca_live_guard import Guard as DcaGuard


def event(**overrides):
    values = {
        "source": "test", "strategy": "grid", "bot": "grid-btc",
        "pair": "BTC-FDUSD", "mechanism": "v22_weekly_buy_gate",
        "transition": "TRIGGERED", "reason": "risk_off",
        "correlation_id": "source-event-1",
    }
    values.update(overrides)
    return build_event(**values)


def test_correlated_event_id_is_stable_across_guard_cycles():
    first = event(occurred_at="2026-08-08T00:00:00+00:00")
    second = event(occurred_at="2026-08-08T00:00:02+00:00")
    assert first["event_id"] == second["event_id"]
    assert first["occurred_at"] != second["occurred_at"]


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
            }],
        },
    )
    text = format_event(report)
    assert text.startswith(MARKDOWN_MESSAGE_PREFIX)
    assert "*GRID · BTC-FDUSD*" in text
    assert "- 累计：`+7.6650 FDUSD`" in text
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


def test_mobile_card_is_one_robot_1440_by_2400_png():
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
            "equity_series": [200, 203, 208], "drawdown_series": [0, 1, 0.95],
        }, output)
        with Image.open(output) as image:
            assert image.size == (1440, 2400)
            assert image.format == "PNG"


def test_grid_parameter_report_marks_missing_evidence_without_fake_png():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        value = event(
            mechanism="parameter_update", transition="PARAMETER_ACTIVATED",
            parameter_sha256="a" * 64,
            details={"report_request": "grid_360d", "candidate": {"levels": 10}},
        )
        attachments = build_parameter_attachments(
            value, release_root=root / "release", output_root=root / "out",
        )
        assert len(attachments) == 1
        assert attachments[0]["kind"] == "document"
        assert attachments[0]["evidence_complete"] is False
        assert Path(attachments[0]["path"]).is_file()
        assert not list((root / "out").rglob("*.png"))


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

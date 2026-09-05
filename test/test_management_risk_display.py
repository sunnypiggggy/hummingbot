from datetime import datetime, timedelta, timezone
from unittest.mock import Mock

from management_bot.app import _reason_cn, _state_cn
from management_bot.risk_display import RULES, RichText, render_risk
from management_bot.telegram_api import TelegramAPI


def robot(gates, phase="COOLDOWN", **extra):
    return dict(strategy="grid", pair="ETH-FDUSD", phase=phase,
                process_running=True, trading_normal=False,
                final_permissions={"buy_enabled": False, "sell_enabled": False},
                gate_statuses=gates, **extra)


def gate(mechanism, **extra):
    return dict(mechanism=mechanism, label="回撤保护 <ETH>", enabled=True,
                buy_enabled=False, sell_enabled=False, reason="unmapped_internal_code", **extra)


def render(value, **kwargs):
    return render_risk([value], 2, _state_cn, _reason_cn, detail=True, **kwargs)


def test_real_cooldown_is_not_a_promise_of_trading():
    r = robot([gate("strategy_drawdown_breaker")], recovery={
        "mechanism": "strategy_drawdown_breaker",
        "cooldown_until": (datetime.now(timezone.utc) + timedelta(hours=2)).timestamp(),
    })
    text = render(r)
    assert "北京时间" in text and "届时仍需通过恢复检查" in text
    assert "权益从峰值回撤" in text
    assert "&lt;ETH&gt;" in text and "unmapped_internal_code" not in text


def test_model_ttl_and_other_mechanism_timer_are_not_release_estimates():
    text = render(robot([gate("v22_weekly_buy_gate")], phase="REENTRY", recovery={
        "mechanism": "strategy_drawdown_breaker", "cooldown_until": 9999999999,
    }, valid_until=9999999999))
    assert "无法按时间预计" in text
    assert "冷却结束" not in text


def test_latched_requires_manual_recovery_and_disabled_gate_does_not_block():
    text = render(robot([gate("infrastructure_integrity_breaker")], phase="LATCHED"))
    assert "不会自动解除" in text and "待人工处理" in text
    g = gate("strategy_drawdown_breaker")
    g["enabled"] = False
    assert "⛔" not in render(robot([g]))


def test_every_blocker_has_chinese_condition_and_pagination():
    r = robot([gate(m) for m in RULES])
    first = render(r)
    assert first.pages > 1
    count = 0
    for page in range(first.pages):
        text = render(r, page=page)
        assert len(text) < 4096
        count += text.count("⛔")
        assert text.count("解除条件：") == text.count("⛔")
        assert text.count("预计解除：") == text.count("⛔")
    assert count == len(RULES)


def test_transport_formats_only_explicit_risk_cards():
    api = TelegramAPI("test")
    api._call = Mock(return_value={})
    for text in (RichText("<b>风控状态</b>"), "plain <text>"):
        api.send(1, text)
        assert (api._call.call_args.args[1].get("parse_mode") == "HTML") == isinstance(text, RichText)
        api.edit(1, 2, text)
        assert (api._call.call_args.args[1].get("parse_mode") == "HTML") == isinstance(text, RichText)

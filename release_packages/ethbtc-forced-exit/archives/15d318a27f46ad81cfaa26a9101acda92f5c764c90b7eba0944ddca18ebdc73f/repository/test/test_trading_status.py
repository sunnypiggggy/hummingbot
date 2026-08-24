from datetime import datetime, timezone

from live_guard.dca_live_report import UnifiedTelegramReporting
from live_guard.trading_status import evaluate_status, gate_row


def status(*rows, running=True, phase="ACTIVE"):
    return evaluate_status(
        process_running=running, phase=phase, gates=rows,
        generated_at="2026-08-17T00:00:00Z", strategy="grid",
        bot="grid-live-fdusd-400", pair="BTC-FDUSD",
    )


def test_normal_requires_healthy_active_and_both_final_permissions():
    value = status(
        gate_row("v22_weekly_buy_gate"),
        gate_row("fomc_gate"),
        gate_row("controller_application_gate"),
    )
    assert value["system_health"] == "HEALTHY"
    assert value["trade_mode"] == "NORMAL"
    assert value["trading_normal"] is True


def test_healthy_risk_off_is_restricted_not_system_failure():
    value = status(
        gate_row("v22_weekly_buy_gate", state="RISK_OFF",
                 buy_enabled=False, sell_enabled=True),
        gate_row("controller_application_gate"),
    )
    assert value["system_health"] == "HEALTHY"
    assert value["trade_mode"] == "BUY_BLOCKED"
    assert value["trading_normal"] is False


def test_integrity_unavailable_is_failed_and_not_model_risk_off():
    value = status(
        gate_row("v22_weekly_buy_gate", state="UNAVAILABLE",
                 buy_enabled=False, sell_enabled=True, health="FAILED"),
        gate_row("infrastructure_integrity_breaker", state="BLOCK",
                 buy_enabled=False, sell_enabled=False, health="FAILED"),
    )
    assert value["system_health"] == "FAILED"
    assert value["trade_mode"] == "BOTH_BLOCKED"
    assert value["gate_statuses"][0]["state"] == "UNAVAILABLE"


def test_disabled_and_not_applicable_gates_do_not_block():
    value = status(
        gate_row("v22_weekly_buy_gate"),
        gate_row("capital_budget_gate", applicable=False),
        gate_row("fomc_gate", enabled=False),
    )
    assert value["trading_normal"] is True


def test_capital_budget_alert_only_does_not_block_normal_trading():
    value = status(
        gate_row("v22_weekly_buy_gate"),
        gate_row("capital_budget_gate", state="ALERT_ONLY",
                 buy_enabled=True, sell_enabled=True,
                 reason="insufficient_quote_budget"),
        gate_row("controller_application_gate"),
    )
    assert value["trade_mode"] == "NORMAL"
    assert value["trading_normal"] is True
    assert value["blockers"] == []


def test_zero_order_execution_failure_is_not_reported_as_normal_trading():
    value = status(
        gate_row("v22_weekly_buy_gate"),
        gate_row("order_execution_gate", state="RETRYING",
                 buy_enabled=True, sell_enabled=True, health="FAILED",
                 reason="expected_orders_missing"),
    )
    assert value["system_health"] == "FAILED"
    assert value["trade_mode"] == "EXECUTION_DEGRADED"
    assert value["trading_normal"] is False
    assert value["final_permissions"] == {"buy_enabled": True, "sell_enabled": True}


def test_report_marks_isolated_prewarm_without_changing_current_model_semantics():
    phase = UnifiedTelegramReporting._reported_cutover_phase(
        {
            "schema": "ethbtc-forced-exit-cutover-status-v1",
            "phase": "PREWARMED_PENDING_ACTIVATION", "fold_boundary": 200,
        },
        {"cutover_phase": "ACTIVE"},
        datetime.fromtimestamp(100, timezone.utc),
    )
    assert phase == "PREWARMING_CURRENT_MODEL_ACTIVE"

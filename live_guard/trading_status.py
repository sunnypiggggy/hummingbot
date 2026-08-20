"""Canonical Grid/DCA trading-status semantics used by reports and guards."""

from __future__ import annotations

from typing import Any, Iterable, Mapping


SCHEMA = "grid-dca-trading-status-v1"
GATE_LABELS = {
    "v22_weekly_buy_gate": "v22周度BUY门",
    "fomc_gate": "FOMC门",
    "strategy_loss_breaker": "策略亏损熔断",
    "strategy_drawdown_breaker": "策略回撤熔断",
    "portfolio_loss_breaker": "组合亏损熔断",
    "portfolio_drawdown_breaker": "组合回撤熔断",
    "position_protection": "持仓保护",
    "infrastructure_integrity_breaker": "基础设施/完整性",
    "capital_budget_gate": "资金预算告警",
    "inventory_ownership_gate": "库存归属门",
    "order_execution_gate": "挂单执行状态",
    "recovery_phase_gate": "退出/冷却/重入",
    "controller_application_gate": "控制器落地",
}
GATE_LABELS["strategy_mode_gate"] = "DCA只做多模式"
GATE_ORDER = tuple(GATE_LABELS)


def gate_row(
    mechanism: str, *, enabled: bool = True, applicable: bool = True,
    state: str = "ALLOW", buy_enabled: bool | None = True,
    sell_enabled: bool | None = True, health: str = "HEALTHY",
    reason: str = "healthy", source: str = "guard",
) -> dict[str, Any]:
    if not applicable:
        state, buy_enabled, sell_enabled, health = "N/A", None, None, "HEALTHY"
    elif not enabled:
        state, buy_enabled, sell_enabled, health = "DISABLED", True, True, "HEALTHY"
    return {
        "mechanism": mechanism,
        "label": GATE_LABELS.get(mechanism, mechanism),
        "enabled": bool(enabled),
        "applicable": bool(applicable),
        "health": str(health),
        "state": str(state),
        "buy_enabled": buy_enabled,
        "sell_enabled": sell_enabled,
        "reason": str(reason),
        "source": str(source),
    }

def evaluate_status(
    *, process_running: bool, phase: str, gates: Iterable[Mapping[str, Any]],
    generated_at: str, strategy: str, bot: str, pair: str,
    runtime_generation: str | None = None, release_sha256: str | None = None,
    model_week: int | None = None, cutover_phase: str | None = None,
) -> dict[str, Any]:
    rows_by_id = {str(row["mechanism"]): dict(row) for row in gates}
    rows = [rows_by_id[name] for name in GATE_ORDER if name in rows_by_id]
    rows.extend(row for name, row in rows_by_id.items() if name not in GATE_ORDER)
    effective = [row for row in rows if row.get("enabled") and row.get("applicable")]
    failed = any(row.get("health") == "FAILED" for row in effective)
    degraded = any(row.get("health") in {"DEGRADED", "UNKNOWN"} for row in effective)
    unknown_decision = any(
        row.get("buy_enabled") is None or row.get("sell_enabled") is None
        for row in effective
    )
    buy_enabled = bool(effective) and not unknown_decision and all(
        row.get("buy_enabled") is True for row in effective
    )
    sell_enabled = bool(effective) and not unknown_decision and all(
        row.get("sell_enabled") is True for row in effective
    )
    normalized_phase = str(phase or "UNKNOWN").upper()
    phase_modes = {
        "EXITING": "EXITING", "COOLDOWN": "COOLDOWN",
        "REENTRY": "REENTRY", "LATCHED": "LATCHED",
    }
    if not process_running:
        trade_mode = "STOPPED"
    elif normalized_phase in phase_modes:
        trade_mode = phase_modes[normalized_phase]
    elif unknown_decision:
        trade_mode = "UNKNOWN"
    elif failed and buy_enabled and sell_enabled:
        # Diagnostic execution failures (for example an expected Grid with no
        # live orders) do not revoke risk permissions, but must never be
        # presented as normal trading.
        trade_mode = "EXECUTION_DEGRADED"
    elif buy_enabled and sell_enabled:
        trade_mode = "NORMAL"
    elif not buy_enabled and sell_enabled:
        trade_mode = "BUY_BLOCKED"
    else:
        trade_mode = "BOTH_BLOCKED"
    system_health = "FAILED" if failed else "DEGRADED" if degraded else "HEALTHY"
    trading_normal = bool(
        process_running and normalized_phase == "ACTIVE"
        and system_health == "HEALTHY" and buy_enabled and sell_enabled
    )
    blockers = [
        {"mechanism": row["mechanism"], "reason": row.get("reason")}
        for row in effective
        if row.get("buy_enabled") is not True or row.get("sell_enabled") is not True
        or row.get("health") != "HEALTHY"
    ]
    return {
        "schema": SCHEMA, "generated_at": generated_at,
        "strategy": strategy, "bot": bot, "pair": pair,
        "system_health": system_health, "trade_mode": trade_mode,
        "trading_normal": trading_normal, "process_running": bool(process_running),
        "phase": normalized_phase,
        "final_permissions": {"buy_enabled": buy_enabled, "sell_enabled": sell_enabled},
        "runtime_generation": runtime_generation, "release_sha256": release_sha256,
        "model_week": model_week, "cutover_phase": cutover_phase,
        "blockers": blockers, "gate_statuses": rows,
    }

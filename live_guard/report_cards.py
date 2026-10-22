"""Build separate Grid/DCA display cards from snapshots and read-only callbacks.

File and SQLite access stays in the report service. Callbacks are evaluated at
its original observation points; this module never publishes or trades.
"""
from __future__ import annotations

from datetime import datetime
from decimal import Decimal, InvalidOperation
from typing import Any, Callable, Mapping
from zoneinfo import ZoneInfo

try:
    from trading_status import evaluate_status, gate_row
except ModuleNotFoundError:
    from live_guard.trading_status import evaluate_status, gate_row
try:
    from dca_live_common import STRATEGY_BUDGET_QUOTE
except ModuleNotFoundError:
    from scripts.dca_live_common import STRATEGY_BUDGET_QUOTE


MAX_DATA_AGE_SECONDS = 900
MAX_GUARD_STATUS_AGE_SECONDS = 90
SHANGHAI = ZoneInfo("Asia/Shanghai")


def dust_usdt_text(dust: Mapping[str, Any]) -> str:
    """Return a display-only dust value without exposing base-asset quantity."""
    try:
        value = float(dust.get("estimated_notional"))
    except (TypeError, ValueError):
        return "共享账户 Dust：无可信美元估值"
    return f"共享账户 Dust：约 {value:.4f} USDT"


def dca_report_infrastructure_status(
    *, bot: Mapping[str, Any], guard: Mapping[str, Any],
    inventory_status: Mapping[str, Any], v22_contract: Mapping[str, Any],
    macro_contract: Mapping[str, Any], now_ts: float,
) -> tuple[bool, str]:
    """Evaluate source health without treating a quiet trade DB as stale.

    A DCA SQLite file may not be written for hours when orders do not fill. Its
    mtime remains report metadata, while Guard observations provide runtime
    liveness and market candles provide valuation freshness.
    """
    monitor_error = str(guard.get("last_monitor_error") or "").strip()
    if monitor_error:
        return False, monitor_error
    if not bot.get("database_available"):
        return False, "sqlite_unavailable"
    if not bot.get("market_data_healthy"):
        return False, "market_data_stale_or_missing"
    try:
        market_age = float(bot.get("market_data_age_seconds"))
    except (TypeError, ValueError):
        return False, "market_data_age_unavailable"
    if market_age > MAX_DATA_AGE_SECONDS:
        return False, "market_data_stale"
    try:
        guard_age = max(0.0, now_ts - float(guard.get("last_success_at")))
    except (TypeError, ValueError):
        return False, "guard_observation_unavailable"
    if guard_age > MAX_GUARD_STATUS_AGE_SECONDS:
        return False, "guard_observation_stale"
    if inventory_status.get("sources_healthy") is not True:
        return False, "inventory_sources_unhealthy"
    if not v22_contract.get("healthy"):
        return False, str(v22_contract.get("reason") or "v22_contract_unhealthy")
    if not macro_contract.get("healthy"):
        return False, str(macro_contract.get("reason") or "macro_contract_unhealthy")
    return True, "all_sources_fresh"


def inventory_exposure_quote(
    owner_value: Any, mark_price: Any, remaining_value: Any = None,
) -> tuple[float | None, float | None]:
    """Value human-decimal inventory contract fields in quote currency."""
    try:
        owner = Decimal(str(owner_value))
        mark = Decimal(str(mark_price))
        remaining = owner if remaining_value is None else Decimal(str(remaining_value))
        if not all(value.is_finite() for value in (owner, mark, remaining)):
            return None, None
        owned_quote = owner * mark
        remaining_quote = remaining * mark
        if not owned_quote.is_finite() or not remaining_quote.is_finite():
            return None, None
        return float(owned_quote), float(remaining_quote)
    except (InvalidOperation, OverflowError, TypeError, ValueError):
        return None, None


def reported_cutover_phase(
    cutover: Mapping[str, Any], contract: Mapping[str, Any], now: datetime,
) -> str | None:
    if contract.get("fold_handover_active") is True:
        return "FOLD_HANDOVER_STABLE"
    phase = str(cutover.get("phase") or "")
    boundary = int(cutover.get("fold_boundary") or 0)
    if (
        cutover.get("schema") == "ethbtc-forced-exit-cutover-status-v1"
        and phase == "PREWARMED_PENDING_ACTIVATION"
        and now.timestamp() < boundary
    ):
        return "PREWARMING_CURRENT_MODEL_ACTIVE"
    return contract.get("cutover_phase")


def breaker_rows(mechanisms: Mapping[str, Any], recovery: Mapping[str, Any]) -> list[dict[str, Any]]:
    active_mechanism = str(recovery.get("mechanism") or "")
    phase = str(recovery.get("phase") or "ACTIVE").upper()
    rows = []
    for mechanism in (
        "strategy_loss_breaker", "strategy_drawdown_breaker",
        "portfolio_loss_breaker", "portfolio_drawdown_breaker",
        "position_protection",
    ):
        blocked = active_mechanism == mechanism and phase != "ACTIVE"
        rows.append(gate_row(
            mechanism, enabled=bool(mechanisms.get(mechanism, True)),
            state=phase if blocked else "ALLOW",
            buy_enabled=not blocked, sell_enabled=not blocked,
            reason=str(recovery.get("reason") or active_mechanism) if blocked else "not_triggered",
            source="risk_recovery",
        ))
    return rows


def build_dca_cards(
    report: Mapping[str, Any], now: datetime, *,
    guard: Mapping[str, Any], v22_observation: Mapping[str, Any],
    dca_read_telemetry: Mapping[str, Any], inventory_status: Mapping[str, Any],
    cutover: Mapping[str, Any], profit_history: Callable[..., list],
    inventory_dust: Callable[[str], dict | None], inventory_row: Callable[[str], dict],
    breaker_rows: Callable[..., list], reported_cutover_phase: Callable[..., str | None],
) -> list[dict[str, Any]]:
    """Keep DCA source/permission semantics and per-bot read ordering."""
    output = []
    for bot in report.get("bots", []):
        pair = str(bot["trading_pair"])
        state = guard.get("bots", {}).get(bot["bot_name"], {})
        recovery = state.get("recovery", {})
        executor_counts = state.get("latest", {}).get("executor_counts", {})
        aggregate = guard.get("gate_aggregate", {}).get("bots", {}).get(bot["bot_name"], {})
        v22_contract = guard.get("gate_aggregate", {}).get("v22", {})
        macro_contract = guard.get("gate_aggregate", {}).get("macro", {})
        capital = guard.get("gate_aggregate", {}).get("capital", {})
        technical_row = v22_contract.get("pairs", {}).get(pair, {})
        all_time = bot.get("profit", {}).get("all_time_mtm_quote")
        initial_equity = float(STRATEGY_BUDGET_QUOTE)
        equity = None if all_time is None else initial_equity + float(all_time)
        history = profit_history("dca", pair, days=370)
        historic_peak = max([
            initial_equity,
            *(row["equity"] for row in history if row["equity"] is not None),
        ])
        peak = max(historic_peak, float(state.get("peak_equity", historic_peak)),
                   equity or historic_peak)
        drawdown = None if equity is None or peak <= 0 else (peak - equity) / peak * 100
        technical = guard.get("v22_observation", {}).get("event_ids", {}).get(pair, "")
        # A quiet strategy can legitimately have no SQLite writes for
        # hours. The visible age and valuation gate therefore use market
        # freshness, not the age of the last persisted trade.
        data_age = bot.get("market_data_age_seconds")
        warnings = list(report.get("warnings", []))
        dust = inventory_dust(pair.split("-", 1)[0])
        if dust:
            warnings.append(dust_usdt_text(dust))
        if not bot.get("database_available"):
            warnings.append(f"{bot['bot_name']}: SQLite无可信数据")
        if not bot.get("market_data_healthy"):
            warnings.append(f"{pair}: Binance行情过期或缺失")
        if data_age is not None and data_age > MAX_DATA_AGE_SECONDS:
            warnings.append(f"{pair}: 数据年龄{data_age:.0f}秒")
        runtime_robot = inventory_status.get("runtime", {}).get("robots", {}).get(
            bot["bot_name"], {}
        )
        process_running = bool(runtime_robot.get("running", False))
        v22_healthy = bool(v22_contract.get("healthy"))
        macro_healthy = bool(macro_contract.get("healthy"))
        infra_healthy, infra_reason = dca_report_infrastructure_status(
            bot=bot, guard=guard, inventory_status=inventory_status,
            v22_contract=v22_contract, macro_contract=macro_contract,
            now_ts=now.timestamp(),
        )
        mechanisms = guard.get("mechanisms", {})
        phase = str(recovery.get("phase", "ACTIVE"))
        gates = [
            gate_row(
                "v22_weekly_buy_gate",
                enabled=bool(mechanisms.get("v22_weekly_buy_gate", True)),
                state=("UNAVAILABLE" if not v22_healthy else str(
                    technical_row.get("model_signal") or (
                        "RISK_ON" if aggregate.get("v22_buy_enabled") else "RISK_OFF"
                    )
                )),
                buy_enabled=(aggregate.get("v22_buy_enabled") if v22_healthy else False),
                sell_enabled=True,
                health="HEALTHY" if v22_healthy else "FAILED",
                reason=str(technical_row.get("reason") or v22_contract.get("reason") or "unknown"),
                source="shared_v22_contract",
            ),
            gate_row(
                "fomc_gate", enabled=bool(mechanisms.get("fomc_gate", True)),
                state="ALLOW" if macro_healthy and macro_contract.get("buy_enabled")
                and macro_contract.get("sell_enabled") else "BLOCK",
                buy_enabled=bool(macro_contract.get("buy_enabled")) if macro_healthy else False,
                sell_enabled=bool(macro_contract.get("sell_enabled")) if macro_healthy else False,
                health="HEALTHY" if macro_healthy else "FAILED",
                reason=str(macro_contract.get("reason") or "unknown"), source="macro_contract",
            ),
            *breaker_rows(mechanisms, recovery),
            gate_row(
                "infrastructure_integrity_breaker",
                state="ALLOW" if infra_healthy else "BLOCK",
                buy_enabled=infra_healthy, sell_enabled=infra_healthy,
                health="HEALTHY" if infra_healthy else "FAILED",
                reason=infra_reason, source="dca_guard",
            ),
            gate_row(
                "capital_budget_gate",
                state="OK" if capital.get("buy_ready") else "ALERT_ONLY",
                # Advisory only: expose the observation without changing
                # the final controller permissions or normal-trading flag.
                buy_enabled=True, sell_enabled=True, health="HEALTHY",
                reason=str(capital.get("reason") or "unknown"), source="quote_budget",
            ),
            gate_row(
                "strategy_mode_gate",
                state=("LONG_ONLY" if aggregate.get("long_only_enabled") else "BILATERAL"),
                # Long-only changes ordinary executor creation, not the
                # controller's permission to close a BUY with a SELL.
                buy_enabled=True, sell_enabled=True, health="HEALTHY",
                reason=(
                    "ordinary_sell_creation_disabled_protective_exits_allowed"
                    if aggregate.get("long_only_enabled")
                    else "ordinary_buy_and_sell_creation_enabled"
                ),
                source="dca_controller",
            ),
            inventory_row(pair.split("-", 1)[0]),
            gate_row(
                "recovery_phase_gate", state=phase,
                buy_enabled=phase == "ACTIVE", sell_enabled=phase == "ACTIVE",
                reason=str(recovery.get("reason") or phase), source="risk_recovery",
            ),
            gate_row(
                "controller_application_gate",
                state="APPLIED" if aggregate.get("controller_applied") else "MISMATCH",
                buy_enabled=bool(aggregate.get("controller_applied")),
                sell_enabled=bool(aggregate.get("controller_applied")),
                health="HEALTHY" if aggregate.get("controller_applied") else "FAILED",
                reason=str(aggregate.get("controller_update_error") or
                           aggregate.get("controller_update_status") or "unknown"),
                source="dca_controller",
            ),
        ]
        trading_status = evaluate_status(
            process_running=process_running, phase=phase, gates=gates,
            generated_at=now.astimezone(SHANGHAI).isoformat(), strategy="dca",
            bot=bot["bot_name"], pair=pair,
            runtime_generation=v22_contract.get("runtime_generation"),
            release_sha256=v22_contract.get("release_sha256"),
            model_week=technical_row.get("model_week"),
            cutover_phase=reported_cutover_phase(cutover, v22_contract, now),
        )
        trading_status["recovery"] = {key: recovery[key] for key in (
            "mechanism", "phase", "cooldown_until", "healthy_cycles",
            "exit_completed_at", "reentry_block_reason",
        ) if key in recovery}
        base_asset = pair.split("-", 1)[0]
        asset_status = inventory_status.get("assets", {}).get(base_asset, {})
        owner_key = f"dca:{bot['bot_name']}"
        owner_value = asset_status.get("owners", {}).get(owner_key)
        mark_price = bot.get("mark_price")
        owned_notional = None
        tradable_risk = None
        if owner_value is not None and mark_price is not None:
            # The Guard persists the exchange-filter result in recovery;
            # without it, reporting the full owner as exposure is safer
            # than incorrectly declaring the position dust.
            remaining = recovery.get("remaining_base", {}).get(pair, owner_value)
            owned_notional, tradable_risk = inventory_exposure_quote(
                owner_value, mark_price, remaining,
            )
            if recovery.get("exit_verification", {}).get(
                "residual_below_exchange_minimum"
            ) is True:
                tradable_risk = 0.0
        risk_exit_trades = len([
            row for row in state.get("emergency_adjustments", [])
            if str(row.get("pair")) == pair and str(row.get("side", "")).upper() == "SELL"
        ])
        item = {
            "strategy": "dca", "bot": bot["bot_name"], "pair": pair,
            "quote_asset": "USDT", "generated_at_bjt": now.astimezone(SHANGHAI).isoformat(),
            "data_age_seconds": data_age,
            "data_sources": ["Hummingbot SQLite TradeFill", "Binance 15m OHLCV",
                             "dca-live-guard state"],
            "profit": dict(bot.get("profit", {})), "equity": equity,
            "peak_equity": peak, "drawdown_pct": drawdown,
            "owned_base": owner_value,
            "strategy_net_base_delta": bot.get("position", {}).get("net_base"),
            "owned_inventory_mtm_quote": owned_notional,
            "tradable_risk_exposure_quote": tradable_risk,
            "risk_exit_trades": risk_exit_trades,
            "ordinary_trade_count": max(
                0, int(bot.get("trades", {}).get("all_time_buys") or 0)
                + int(bot.get("trades", {}).get("all_time_sells") or 0)
                - risk_exit_trades,
            ),
            "fees_quote": bot.get("profit", {}).get("all_time_fees_quote"),
            "buys": bot.get("trades", {}).get("all_time_buys"),
            "sells": bot.get("trades", {}).get("all_time_sells"),
            "phase": phase,
            "exit_execution": {
                "takeover_stage": recovery.get("executor_takeover_stage"),
                "remaining_base": recovery.get("remaining_base", {}).get(pair),
                "verification": recovery.get("exit_verification"),
                "last_fill": recovery.get("last_exit_fill"),
            },
            "active_runtime": {
                "orders": executor_counts.get("open_orders"),
                "buy_executors": executor_counts.get("active_buy_executors"),
                "sell_executors": executor_counts.get("active_sell_executors"),
            },
            "v22_gate": (
                "放行" if aggregate.get("v22_buy_enabled") is True else
                "关闭" if aggregate.get("v22_buy_enabled") is False else
                "event:" + str(technical)[:8] if technical else "无可信数据"
            ),
            "fomc_gate": (
                f"BUY={'开' if aggregate.get('fomc_buy_enabled') else '关'} / "
                f"SELL={'开' if aggregate.get('fomc_sell_enabled') else '关'}"
                if aggregate else "无可信数据"
            ),
            "warnings": warnings,
            "unattributed_dust": dust,
            "runtime_health": {
                "current_error": v22_observation.get("last_error"),
                "last_recovered_error": v22_observation.get("last_recovered_error"),
                "last_recovered_at": v22_observation.get("last_recovered_at"),
                "source_error_total": int(v22_observation.get("source_error_total", 0)),
                "integrity_error_total": int(v22_observation.get("integrity_error_total", 0)),
                "read_retry_attempts": int(dca_read_telemetry.get("retry_attempts", 0)),
                "pool_replacements": int(dca_read_telemetry.get("pool_replacements", 0)),
                "longest_read_recovery_seconds": float(
                    dca_read_telemetry.get("longest_recovery_seconds", 0.0)
                ),
            },
            "trading_status": trading_status,
        }
        output.append(item)
    return output


def build_grid_cards(
    now: datetime, *, state: Mapping[str, Any], v22_observation: Mapping[str, Any],
    grid_read_telemetry: Mapping[str, Any], macro: Mapping[str, Any], bot_name: str,
    latest: Mapping[str, Any], gate: Mapping[str, Any], cutover: Mapping[str, Any],
    runtime: Mapping[str, Any], transport_summary: Mapping[str, Any],
    runtime_buy_ids: set[str], runtime_sell_ids: set[str],
    active_pair_parameters: Mapping[str, Any], profit_history: Callable[..., list],
    trade_counts: Callable[..., tuple], inventory_dust: Callable[[str], dict | None],
    inventory_row: Callable[[str], dict], clock: Callable[[], float],
    breaker_rows: Callable[..., list], reported_cutover_phase: Callable[..., str | None],
) -> list[dict[str, Any]]:
    """Keep Grid runtime observations, clock reads and gate semantics separate."""
    output = []
    for pair in ("BTC-FDUSD", "ETH-FDUSD"):
        values = latest.get("pairs", {}).get(pair, {})
        pnl = float(values["pnl"]) if values.get("pnl") is not None else None
        equity = None if pnl is None else 200 + pnl
        history = profit_history("grid", pair, days=370)
        historic_peak = max([200.0, *(row["equity"] for row in history if row["equity"] is not None)])
        peak = max(historic_peak, equity or historic_peak)
        drawdown = None if equity is None else (peak - equity) / peak * 100
        buys, sells, fees = trade_counts(str(latest.get("database", "")), pair)
        data_age = (max(0, clock() - float(latest.get("updated_at", 0)))
                    if latest else None)
        warnings = [] if latest else ["Grid Guard快照无可信数据"]
        dust = inventory_dust(pair.split("-", 1)[0])
        if dust:
            warnings.append(dust_usdt_text(dust))
        if data_age is not None and data_age > MAX_DATA_AGE_SECONDS:
            warnings.append(f"{pair}: Grid快照年龄{data_age:.0f}秒")
        pair_recovery = runtime.get("pair_recovery", {}).get(pair, {})
        portfolio_recovery = runtime.get("portfolio_recovery", {})
        recovery = (
            portfolio_recovery
            if str(portfolio_recovery.get("phase", "ACTIVE")) != "ACTIVE"
            else pair_recovery
        )
        phase = str(recovery.get("phase") or "ACTIVE")
        pair_order_ids = {
            str(value)
            for value in runtime.get("ledgers", {}).get(pair, {}).get("open_order_ids", [])
        }
        pair_params = active_pair_parameters.get(pair, {})
        effective_buy_layers = len(pair_order_ids & runtime_buy_ids)
        effective_sell_layers = len(pair_order_ids & runtime_sell_ids)
        order_build = state.get("effective_order_status", {}).get(
            pair, runtime.get("order_build_status", {}).get(pair, {})
        )
        order_build_state = str(order_build.get("state") or "UNKNOWN").upper()
        order_build_healthy = order_build_state in {
            "HEALTHY", "EXPECTED_EMPTY", "INTENTIONAL_IDLE",
            "REFRESH_REQUESTED", "CANCEL_PENDING", "REBUILDING",
            "HEALTHY_DEFERRED", "MAKER_WAIT",
        }
        deferred_layers = list(order_build.get("maker_deferred_layers") or [])
        mechanisms = state.get("mechanisms", {})
        technical_contract = gate.get("pairs", {}).get(pair, {})
        runtime_technical = runtime.get("technical_buy_gate", {}).get(
            "pairs", {}
        ).get(pair, {})
        runtime_macro = runtime.get("macro_gate", {})
        v22_healthy = bool(
            gate.get("source_healthy") and runtime_technical.get("healthy", True)
        )
        macro_healthy = bool(runtime_macro.get("healthy", not macro.get("pause_new_orders")))
        process_running = bool(runtime and data_age is not None and data_age <= 90)
        data_healthy = bool(data_age is not None and data_age <= MAX_DATA_AGE_SECONDS)
        infra_healthy = bool(
            process_running and data_healthy and v22_healthy and macro_healthy
            and not state.get("last_monitor_error")
        )
        v22_buy = bool(runtime_technical.get(
            "buy_enabled", technical_contract.get("buy_enabled", False)
        )) if v22_healthy else False
        macro_allow = bool(not runtime_macro.get(
            "paused", macro.get("pause_new_orders", False)
        )) if macro_healthy else False
        controller_applied = bool(
            runtime_technical
            and v22_buy == bool(technical_contract.get("buy_enabled"))
            and macro_allow == (not bool(macro.get("pause_new_orders")))
        )
        gates = [
            gate_row(
                "v22_weekly_buy_gate",
                enabled=bool(mechanisms.get("v22_weekly_buy_gate", True)),
                state=("UNAVAILABLE" if not v22_healthy else str(
                    technical_contract.get("model_signal") or (
                        "RISK_OFF" if technical_contract.get("risk_off_active") else
                        "RISK_ON"
                    )
                )),
                buy_enabled=v22_buy, sell_enabled=True,
                health="HEALTHY" if v22_healthy else "FAILED",
                reason=str(technical_contract.get("reason") or gate.get("reason") or "unknown"),
                source="shared_v22_contract",
            ),
            gate_row(
                "fomc_gate", enabled=bool(mechanisms.get("fomc_gate", True)),
                state="ALLOW" if macro_allow else "BLOCK",
                buy_enabled=macro_allow, sell_enabled=macro_allow,
                health="HEALTHY" if macro_healthy else "FAILED",
                reason=str(runtime_macro.get("reason") or "macro_gate"), source="grid_runtime",
            ),
            *breaker_rows(mechanisms, recovery),
            gate_row(
                "infrastructure_integrity_breaker",
                state="ALLOW" if infra_healthy else "BLOCK",
                buy_enabled=infra_healthy, sell_enabled=infra_healthy,
                health="HEALTHY" if infra_healthy else "FAILED",
                reason="all_sources_fresh" if infra_healthy else str(
                    state.get("last_monitor_error") or gate.get("reason") or "runtime_unhealthy"
                ), source="grid_guard",
            ),
            gate_row("capital_budget_gate", applicable=False, reason="not_applicable_to_grid"),
            inventory_row(pair.split("-", 1)[0]),
            gate_row(
                "order_execution_gate",
                state=order_build_state,
                buy_enabled=True, sell_enabled=True,
                health=(
                    "DEGRADED" if order_build_state == "MAKER_WAIT"
                    else "HEALTHY" if order_build_healthy else "FAILED"
                ),
                reason=str(order_build.get("reason") or "runtime_order_status_unavailable"),
                source="grid_runtime",
            ),
            gate_row(
                "recovery_phase_gate", state=phase,
                buy_enabled=phase == "ACTIVE", sell_enabled=phase == "ACTIVE",
                reason=str(recovery.get("reason") or phase), source="grid_runtime",
            ),
            gate_row(
                "controller_application_gate",
                state="APPLIED" if controller_applied else "MISMATCH",
                buy_enabled=controller_applied, sell_enabled=controller_applied,
                health="HEALTHY" if controller_applied else "FAILED",
                reason="runtime_consumed_current_gates" if controller_applied
                else "runtime_gate_state_mismatch", source="grid_runtime",
            ),
        ]
        trading_status = evaluate_status(
            process_running=process_running, phase=phase, gates=gates,
            generated_at=now.astimezone(SHANGHAI).isoformat(), strategy="grid",
            bot=bot_name, pair=pair,
            runtime_generation=gate.get("runtime_generation"),
            release_sha256=gate.get("release_sha256"),
            model_week=technical_contract.get("model_week"),
            cutover_phase=reported_cutover_phase(cutover, gate, now),
        )
        trading_status["recovery"] = {key: recovery[key] for key in (
            "mechanism", "phase", "cooldown_until", "healthy_cycles",
            "exit_completed_at", "reentry_block_reason",
        ) if key in recovery}
        item = {
            "strategy": "grid", "bot": bot_name, "pair": pair,
            "quote_asset": "FDUSD", "generated_at_bjt": now.astimezone(SHANGHAI).isoformat(),
            "data_age_seconds": data_age,
            "data_sources": ["Hummingbot SQLite TradeFill",
                             "grid-live-guard snapshot", "v22 shared contract"],
            "profit": {"all_time_mtm_quote": pnl}, "equity": equity,
            "peak_equity": peak, "drawdown_pct": drawdown,
            "owned_base": values.get("owned_base"),
            "strategy_net_base_delta": values.get("net_base"),
            "cash_balance_quote": values.get("quote_balance"),
            "owned_inventory_mtm_quote": (
                float(values.get("owned_base", 0)) * float(values.get("mark", 0))
                if values.get("owned_base") is not None and values.get("mark") is not None
                else None
            ),
            "equity_source": values.get("equity_source"),
            "reconciliation_status": (
                "RECONCILED" if values.get("owned_base") is not None else
                "EQUITY_UNRECONCILED"
            ),
            "fees_quote": values.get("fees_quote", fees),
            "buys": buys, "sells": sells,
            "phase": phase,
            "active_runtime": {
                "orders": len(pair_order_ids),
                "expected_buy_layers": order_build.get("expected_buy_layers"),
                "expected_sell_layers": order_build.get("expected_sell_layers"),
                "actual_buy_layers": order_build.get(
                    "actual_buy_layers", effective_buy_layers
                ),
                "actual_sell_layers": order_build.get(
                    "actual_sell_layers", effective_sell_layers
                ),
                "order_build_state": order_build_state,
                "order_build_reason": order_build.get("reason"),
                "maker_deferred_layers": len(deferred_layers),
                "maker_rejection_count": order_build.get("maker_rejection_count", 0),
                "maker_topology_refresh_count": order_build.get(
                    "maker_topology_refresh_count", 0
                ),
                "maker_longest_recovery_seconds": order_build.get(
                    "maker_longest_recovery_seconds", 0
                ),
            },
            "order_build_status": order_build,
            "runtime_code": {
                "loaded_source_sha256": runtime.get("loaded_source_sha256"),
                "disk_source_sha256": runtime.get("disk_source_sha256"),
                "loaded_at": runtime.get("loaded_at"),
                "current": bool(
                    runtime.get("loaded_source_sha256")
                    and runtime.get("loaded_source_sha256")
                    == runtime.get("disk_source_sha256")
                ),
            },
            "grid_parameters": {
                "profile": pair_params.get("profile"),
                "grid_range": pair_params.get("grid_range"),
                "grid_levels": pair_params.get("grid_levels"),
                "take_profit": pair_params.get("take_profit"),
                "minimum_order_quote": pair_params.get("minimum_order_quote"),
                "effective_buy_layers": effective_buy_layers,
                "effective_sell_layers": effective_sell_layers,
                "parameter_version": runtime.get("active_parameter_version"),
                "parameter_sha256": runtime.get("active_parameter_sha256"),
            },
            "v22_gate": (
                "不可用" if not v22_healthy else
                "Risk-Off" if gate.get("pairs", {}).get(pair, {}).get("risk_off_active")
                else "Risk-On"
            ),
            "fomc_gate": "暂停" if macro.get("pause_new_orders") else "放行",
            "warnings": warnings,
            "unattributed_dust": dust,
            "runtime_transport": transport_summary,
            "runtime_health": {
                "current_error": v22_observation.get("last_error"),
                "last_recovered_error": v22_observation.get("last_recovered_error"),
                "last_recovered_at": v22_observation.get("last_recovered_at"),
                "source_error_total": int(v22_observation.get("source_error_total", 0)),
                "integrity_error_total": int(v22_observation.get("integrity_error_total", 0)),
                "read_retry_attempts": sum(
                    int(row.get("retry_attempts", 0))
                    for row in grid_read_telemetry.values()
                    if isinstance(row, Mapping)
                ),
                "pool_replacements": sum(
                    int(row.get("pool_replacements", 0))
                    for row in grid_read_telemetry.values()
                    if isinstance(row, Mapping)
                ),
                "longest_read_recovery_seconds": max(
                    (float(row.get("longest_recovery_seconds", 0.0))
                     for row in grid_read_telemetry.values()
                     if isinstance(row, Mapping)),
                    default=0.0,
                ),
            },
            "trading_status": trading_status,
        }
        output.append(item)
    return output

"""Fail-safe Telegram channel notifications for live Grid and DCA.

This module deliberately has no trading dependencies. Producers append immutable
JSONL events; the existing dca-live-report service is the sole process allowed
to deliver them to Telegram.
"""

from __future__ import annotations

import hashlib
import json
import mimetypes
import os
import re
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping
from zoneinfo import ZoneInfo

import requests

try:
    from runtime_endpoints import telegram_api_base
except ModuleNotFoundError:
    try:
        # API-managed Hummingbot instances mount this module under the
        # namespace-package directory /home/hummingbot/scripts.
        from scripts.runtime_endpoints import telegram_api_base
    except ModuleNotFoundError:
        from live_guard.runtime_endpoints import telegram_api_base

try:
    from PIL import Image, ImageDraw, ImageFont
except ModuleNotFoundError:  # Trading producers only need JSONL event helpers.
    Image = ImageDraw = ImageFont = None


SCHEMA = "ethbtc-telegram-event-v1"
MARKDOWN_MESSAGE_PREFIX = "__TELEGRAM_MARKDOWN_V1__\n"
MECHANISMS = (
    "v22_weekly_buy_gate",
    "fomc_gate",
    "strategy_loss_breaker",
    "strategy_drawdown_breaker",
    "portfolio_loss_breaker",
    "portfolio_drawdown_breaker",
    "position_protection",
    "capital_budget_gate",
)
LIFECYCLE_TRANSITIONS = {
    "TRIGGERED", "EXITING", "EXIT_COMPLETE", "COOLDOWN", "REENTRY",
    "RECOVERED", "LATCHED", "EXIT_DELAY", "ACTION_FAILED",
}
RUNTIME_ERROR_TRANSITIONS = {"ERROR_OCCURRED", "ERROR_RECOVERED"}
MODEL_CUTOVER_TRANSITIONS = {
    "MODEL_CUTOVER_PREWARMED", "MODEL_CUTOVER_STABLE",
    "MODEL_CUTOVER_PRECHECK_FAILED", "MODEL_FOLD_ACTIVATED",
    "MODEL_RETENTION_PRUNED", "MODEL_RETENTION_FAILED",
}
INVENTORY_TRANSITIONS = {
    "INVENTORY_UNATTRIBUTED_DETECTED",
    "INVENTORY_DUST_CLASSIFIED",
    "INVENTORY_STATUS_CORRECTED",
    "INVENTORY_LIQUIDATION_STARTED",
    "INVENTORY_LIQUIDATION_COMPLETED",
    "INVENTORY_LIQUIDATION_FAILED",
    "INVENTORY_OWNERSHIP_DEFICIT",
    "INVENTORY_RECONCILIATION_RECOVERED",
    "INVENTORY_LIQUIDATION_FEE_RECONCILED",
}
SHANGHAI = ZoneInfo("Asia/Shanghai")
MOBILE_CARD_SIZE = (1440, 3200)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(raw).hexdigest()


def _clean(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _clean(item) for key, item in value.items() if item is not None}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    return value


def build_event(
    *, source: str, strategy: str, bot: str, pair: str, mechanism: str,
    transition: str, reason: str, severity: str = "warning",
    phase_from: str = "", phase_to: str = "", action: str = "",
    trigger_value: Any = None, threshold: Any = None,
    release_sha256: str = "", model_sha256: str = "",
    parameter_sha256: str = "", attachments: Iterable[Mapping[str, Any]] = (),
    requires_manual_action: bool = False, occurred_at: str | None = None,
    correlation_id: str = "", details: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    transition = transition.upper()
    if transition not in LIFECYCLE_TRANSITIONS and transition not in INVENTORY_TRANSITIONS and transition not in RUNTIME_ERROR_TRANSITIONS and transition not in MODEL_CUTOVER_TRANSITIONS and transition not in {
        "PARAMETER_CANDIDATE", "PARAMETER_APPROVAL_PENDING",
        "PARAMETER_ACTIVATED", "PARAMETER_RETAINED",
        "MODEL_APPROVAL_PENDING", "MODEL_DEFAULT_APPROVED",
        "MODEL_APPROVAL_REJECTED", "MODEL_UPDATE_BLOCKED",
        "REPORT_EVIDENCE_MISSING", "PROFIT_REPORT",
    }:
        raise ValueError(f"unsupported notification transition {transition!r}")
    if mechanism not in MECHANISMS and mechanism not in {
        "infrastructure_integrity_breaker", "parameter_update", "profit_report",
        "account_inventory", "runtime_error",
    }:
        raise ValueError(f"unsupported notification mechanism {mechanism!r}")
    occurred_at = occurred_at or datetime.now(timezone.utc).isoformat()
    identity = {
        "source": source, "strategy": strategy, "bot": bot, "pair": pair,
        "mechanism": mechanism, "transition": transition,
        "phase_from": phase_from, "phase_to": phase_to,
        "correlation_id": correlation_id, "reason": reason,
    }
    # Producers may observe the same transition on every guard cycle.  A
    # stable correlation id therefore owns the notification identity; the
    # wall-clock timestamp is only part of the identity when no durable source
    # event id exists.
    if not correlation_id:
        identity["occurred_at"] = occurred_at
    return _clean({
        "schema": SCHEMA,
        "event_id": canonical_sha256(identity),
        "occurred_at": occurred_at,
        **identity,
        "severity": severity,
        "action": action,
        "trigger_value": trigger_value,
        "threshold": threshold,
        "release_sha256": release_sha256,
        "model_sha256": model_sha256,
        "parameter_sha256": parameter_sha256,
        "requires_manual_action": bool(requires_manual_action),
        "attachments": [dict(item) for item in attachments],
        "details": dict(details or {}),
    })


_SECRET_PATTERNS = (
    (re.compile(r"(?i)(authorization\s*[:=]\s*)(?:bearer\s+)?[^\s,;]+"), r"\1[REDACTED]"),
    (re.compile(r"(?i)((?:api[_-]?key|api[_-]?secret|token|password|signature)\s*[:=]\s*)[^\s,;]+"), r"\1[REDACTED]"),
    (re.compile(r"https://api\.telegram\.org/bot[^/\s]+"), "https://api.telegram.org/bot[REDACTED]"),
    (re.compile(r"([?&](?:signature|timestamp|recvWindow|apiKey)=[^&\s]+)", re.I), ""),
)


def sanitize_runtime_error(error: BaseException | str, *, limit: int = 600) -> str:
    """Return a channel-safe one-line error without credentials or request signatures."""
    if isinstance(error, BaseException):
        text = f"{type(error).__name__}: {error}"
    else:
        text = str(error)
    text = " ".join(text.replace("\x00", " ").split())
    for pattern, replacement in _SECRET_PATTERNS:
        text = pattern.sub(replacement, text)
    return (text or "unknown runtime error")[:limit]


_RUNTIME_LOG_ERROR = re.compile(
    r"(?i)(\bERROR\b|\bCRITICAL\b|Traceback \(most recent call last\)|"
    r"\bexception\b|cycle failed|\border\b.*\b(?:failed|rejected)\b|"
    r"failed to (?:start|load|submit|place))"
)


def runtime_error_lines(lines: Iterable[str]) -> list[str]:
    """Return relevant log errors, excluding known non-error status phrases."""
    found = []
    for raw in lines:
        line = " ".join(str(raw).split())
        if not line or "0 errors" in line.lower():
            continue
        if _RUNTIME_LOG_ERROR.search(line):
            found.append(sanitize_runtime_error(line))
    return found


class RuntimeErrorChannel:
    """Persist and emit one alert per runtime-error episode plus one recovery.

    The journal is deliberately separate from trading state. Notification I/O
    can fail without changing any trading gate, and a process restart does not
    create a duplicate alert for an already active error fingerprint.
    """

    def __init__(
        self, *, event_path: Path, state_path: Path, source: str, strategy: str,
        bot: str, pair: str,
    ) -> None:
        self.event_path = Path(event_path)
        self.state_path = Path(state_path)
        self.source = source
        self.strategy = strategy
        self.bot = bot
        self.pair = pair
        try:
            self.state = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            self.state = {"schema": "runtime-error-channel-v2", "components": {}}
        self.state["schema"] = "runtime-error-channel-v2"
        self.state.setdefault("components", {})
        self.state.setdefault("history", [])

    def _save(self) -> None:
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.state_path.with_suffix(self.state_path.suffix + ".tmp")
            temporary.write_text(
                json.dumps(self.state, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            temporary.replace(self.state_path)
        except OSError:
            # Notification persistence must never break a trading/guard loop.
            return

    def _emit(self, event: Mapping[str, Any]) -> bool:
        try:
            append_event(self.event_path, event)
            return True
        except (OSError, ValueError, TypeError):
            # The report service will retry persisted events, but a producer
            # unable to write its event file must still continue risk work.
            return False

    def _failure_event(self, component: str, row: Mapping[str, Any]) -> dict[str, Any]:
        return build_event(
            source=self.source, strategy=self.strategy, bot=self.bot, pair=self.pair,
            mechanism="runtime_error", transition="ERROR_OCCURRED",
            reason=str(row.get("summary") or "unknown runtime error"),
            severity=str(row.get("severity") or "warning"),
            phase_to="ERROR_ACTIVE", action=str(row.get("action") or "automatic_retry"),
            correlation_id=str(row["episode_id"]), details={
                "component": component, "error_summary": row.get("summary", ""),
                "first_seen_at": datetime.fromtimestamp(
                    float(row.get("first_seen_at", time.time())), timezone.utc
                ).isoformat(),
                "occurrences": int(row.get("occurrences", 1)),
                "trading_impact": row.get("trading_impact", ""),
                "notification_delay_seconds": float(
                    row.get("notification_delay_seconds", 0)
                ),
                **dict(row.get("details") or {}),
            },
        )

    def _remember_recovery(self, component: str, row: Mapping[str, Any], now: float) -> None:
        history = self.state.setdefault("history", [])
        history.append({
            "component": component,
            "episode_id": row.get("episode_id"),
            "summary": row.get("summary", ""),
            "first_seen_at": float(row.get("first_seen_at", now)),
            "recovered_at": now,
            "duration_seconds": max(0, now - float(row.get("first_seen_at", now))),
            "occurrences": int(row.get("occurrences", 1)),
            "alert_sent": bool(row.get("notified")),
            "suppressed_as_transient": not bool(row.get("notified")),
        })
        self.state["history"] = history[-1000:]

    def record_transient_recovery(
        self, component: str, error: BaseException | str, *, occurrences: int = 1,
        duration_seconds: float = 0.0, now: float | None = None,
    ) -> None:
        """Persist an internally recovered transport retry without alerting."""
        now = time.time() if now is None else float(now)
        duration = max(0.0, float(duration_seconds))
        row = {
            "episode_id": canonical_sha256({
                "source": self.source, "component": component,
                "summary": sanitize_runtime_error(error), "recovered_at": now,
            }),
            "summary": sanitize_runtime_error(error),
            "first_seen_at": now - duration,
            "occurrences": max(1, int(occurrences)),
            "notified": False,
        }
        self._remember_recovery(component, row, now)
        self._save()

    def failure(
        self, component: str, error: BaseException | str, *,
        trading_impact: str, severity: str = "warning", action: str = "automatic_retry",
        details: Mapping[str, Any] | None = None, now: float | None = None,
        notify_after_seconds: float = 0,
    ) -> bool:
        now = time.time() if now is None else float(now)
        notify_after_seconds = max(0.0, float(notify_after_seconds))
        summary = sanitize_runtime_error(error)
        fingerprint = canonical_sha256({"component": component, "summary": summary})
        components = self.state["components"]
        previous = components.get(component, {})
        if previous.get("active") and previous.get("fingerprint") == fingerprint:
            previous["occurrences"] = int(previous.get("occurrences", 1)) + 1
            previous["last_seen_at"] = now
            previous["details"] = dict(details or previous.get("details") or {})
            emitted = False
            elapsed = now - float(previous.get("first_seen_at", now))
            if not previous.get("notified") and elapsed >= float(
                previous.get("notification_delay_seconds", notify_after_seconds)
            ):
                emitted = self._emit(self._failure_event(component, previous))
                if emitted:
                    previous["notified"] = True
                    previous["notified_at"] = now
            self._save()
            return emitted
        episode_id = canonical_sha256({
            "source": self.source, "component": component,
            "fingerprint": fingerprint, "first_seen_at": now,
        })
        row = {
            "active": True, "episode_id": episode_id, "fingerprint": fingerprint,
            "summary": summary, "first_seen_at": now, "last_seen_at": now,
            "occurrences": 1, "trading_impact": trading_impact,
            "severity": severity, "action": action, "details": dict(details or {}),
            "notification_delay_seconds": notify_after_seconds, "notified": False,
        }
        components[component] = row
        emitted = False
        if notify_after_seconds <= 0:
            emitted = self._emit(self._failure_event(component, row))
            if emitted:
                row["notified"] = True
                row["notified_at"] = now
        self._save()
        return emitted

    def recovered(
        self, component: str, *, trading_status: str = "normal",
        details: Mapping[str, Any] | None = None, now: float | None = None,
    ) -> bool:
        now = time.time() if now is None else float(now)
        row = self.state["components"].get(component, {})
        if not row.get("active"):
            return False
        row["active"] = False
        row["recovered_at"] = now
        self._remember_recovery(component, row, now)
        emitted = False
        if row.get("notified"):
            emitted = self._emit(build_event(
                source=self.source, strategy=self.strategy, bot=self.bot, pair=self.pair,
                mechanism="runtime_error", transition="ERROR_RECOVERED",
                reason=f"{component} recovered", severity="info",
                phase_from="ERROR_ACTIVE", phase_to="HEALTHY",
                action="resume_normal_monitoring",
                correlation_id=f"{row['episode_id']}:recovered", details={
                    "component": component, "error_summary": row.get("summary", ""),
                    "first_seen_at": datetime.fromtimestamp(
                        float(row.get("first_seen_at", now)), timezone.utc
                    ).isoformat(),
                    "recovered_at": datetime.fromtimestamp(now, timezone.utc).isoformat(),
                    "duration_seconds": max(
                        0, now - float(row.get("first_seen_at", now))
                    ),
                    "occurrences": int(row.get("occurrences", 1)),
                    "trading_status": trading_status, **dict(details or {}),
                },
            ))
        self._save()
        return emitted

    def recover_if_quiet(
        self, component: str, *, quiet_seconds: float,
        trading_status: str = "normal", now: float | None = None,
    ) -> bool:
        now = time.time() if now is None else float(now)
        row = self.state["components"].get(component, {})
        if not row.get("active"):
            return False
        if now - float(row.get("last_seen_at", now)) < float(quiet_seconds):
            return False
        return self.recovered(
            component, trading_status=trading_status, now=now,
            details={"recovery_basis": f"no recurrence for {quiet_seconds:g} seconds"},
        )


def append_event(path: Path, event: Mapping[str, Any]) -> None:
    if event.get("schema") != SCHEMA:
        raise ValueError("notification event schema mismatch")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(dict(event), ensure_ascii=False, default=str) + "\n")
        output.flush()


def mechanism_enabled(mechanism: str, environ: Mapping[str, str] | None = None) -> bool:
    environ = os.environ if environ is None else environ
    if mechanism in MECHANISMS:
        key = f"TELEGRAM_ALERT_{mechanism.upper()}_ENABLED"
        return str(environ.get(key, "true")).lower() == "true"
    return True


def hermes_recovery_prompt(event: Mapping[str, Any]) -> str:
    return (
        "请在 Hermes 私聊中发送以下提示词：\n"
        f"检查恢复事件 {event.get('event_id')}，机器人 {event.get('bot') or '-'}，"
        f"交易对 {event.get('pair') or '-'}，机制 {event.get('mechanism')}，"
        f"当前阶段 {event.get('phase_to') or event.get('transition')}。"
        f"release={event.get('release_sha256') or '-'}，"
        f"model={event.get('model_sha256') or '-'}。"
        "先只读核验退出已完成、无活动订单、资金归属一致、合同与过滤器新鲜、"
        "其他风控门全部健康；通过后生成与本事件哈希绑定的一次性恢复审批。"
        "不要绕过仍生效的风控门，也不要直接修改交易开关。"
    )


MECHANISM_EXPLANATIONS = {
    "v22_weekly_buy_gate": "v22 周度模型信号改变了该交易对的 BUY 风险状态",
    "fomc_gate": "FOMC 宏观租约改变了交易方向权限",
    "strategy_loss_breaker": "单机器人累计亏损达到策略保护条件",
    "strategy_drawdown_breaker": "单机器人权益从峰值回撤达到策略保护条件",
    "portfolio_loss_breaker": "同一策略的 BTC/ETH 组合累计亏损达到组合保护条件",
    "portfolio_drawdown_breaker": "同一策略的 BTC/ETH 组合权益回撤达到组合保护条件",
    "position_protection": "机器人归属持仓达到止损或异常持仓保护条件",
    "capital_budget_gate": "可用 USDT 低于计划资金预算；该机制仅告警，不改变 BUY/SELL 权限",
    "infrastructure_integrity_breaker": "模型、合同、行情、API 或数据完整性检查失败",
    "parameter_update": "逐交易对 Grid 参数或订单构建校验失败",
}


def explain_event(event: Mapping[str, Any]) -> str:
    """Return one plain-language sentence covering cause and next impact."""
    mechanism = str(event.get("mechanism", ""))
    strategy = str(event.get("strategy", "")).lower()
    transition = str(event.get("transition", "")).upper()
    reason = " ".join(str(event.get("reason") or "未提供原始原因").split())
    cause = MECHANISM_EXPLANATIONS.get(mechanism, "该风控机制的状态发生变化")
    details = event.get("details", {})
    details = details if isinstance(details, Mapping) else {}

    if mechanism == "capital_budget_gate":
        free_quote = details.get("free_quote", event.get("trigger_value", "未知"))
        required_quote = details.get("required_quote", event.get("threshold", "未知"))
        state = "已恢复预算水平" if transition == "RECOVERED" else "低于预算水平"
        return (
            f"解释：可用 USDT {free_quote}，计划预算 {required_quote}，当前{state}"
            f"（原始原因：{reason}）；该机制仅发送告警和展示状态，"
            "不会关闭 BUY/SELL、不会撤单，也不会改变交易机器人正常运行权限。"
        )

    if mechanism == "v22_weekly_buy_gate" and strategy == "dca":
        mechanism_buy = details.get("buy_enabled")
        if mechanism_buy is None:
            mechanism_buy = transition == "RECOVERED"
        gate_state = "放行（Risk-On）" if mechanism_buy else "阻止（Risk-Off）"
        effective_buy = details.get("effective_buy_enabled")
        effective_sell = details.get("effective_sell_enabled")
        recovery_phase = str(details.get("recovery_phase") or event.get("phase_to") or "未知")
        execution_applied = details.get("execution_applied")
        controller_status = str(details.get("controller_update_status") or "未提供")
        state = f"当前状态：v22 BUY 门={gate_state}，恢复阶段={recovery_phase}"
        if effective_buy is None or effective_sell is None or execution_applied is None:
            impact = (
                "影响：事件没有携带 DCA 聚合门和控制器落地结果，"
                "只能确认本机制状态，不能据此判断交易是否正常"
            )
        elif not execution_applied:
            impact = (
                "影响：控制器更新未确认"
                f"（状态={controller_status}），按 Fail-Closed 视为交易未正常恢复"
            )
        elif bool(effective_buy) and bool(effective_sell) and recovery_phase == "ACTIVE":
            impact = (
                "影响：DCA 聚合门 BUY=放行、SELL=放行；"
                "可正常创建普通 BUY executor并执行SELL，交易正常"
            )
        elif not bool(effective_buy):
            sell_text = "放行" if effective_sell else "阻止"
            impact = (
                f"影响：DCA 聚合门 BUY=阻止、SELL={sell_text}；"
                "不会创建新的普通 BUY executor，交易处于受限状态"
            )
        else:
            impact = (
                "影响：DCA 聚合门 BUY=放行、SELL=阻止；"
                "不能视为双侧正常交易"
            )
        return f"解释：由于{cause}（原始原因：{reason}）。{state}。{impact}。"

    if transition == "TRIGGERED":
        if mechanism == "v22_weekly_buy_gate":
            impact = "后续该交易对停止新增普通 BUY，并在已授权时进入撤单和归属库存退出流程"
        elif mechanism == "fomc_gate":
            buy = details.get("buy_enabled")
            sell = details.get("sell_enabled")
            if buy is not None or sell is not None:
                impact = (
                    "后续仅按当前方向门执行"
                    f"（BUY={'放行' if buy else '关闭'}、SELL={'放行' if sell else '关闭'}）"
                )
            else:
                impact = "后续受限方向停止创建新交易，直至宏观租约结束或撤销"
        elif mechanism == "infrastructure_integrity_breaker":
            impact = "后续停止新增风险并退出归属库存，退出完成后保持 LATCHED 等待人工处理"
        else:
            impact = "后续停止新增订单、撤销活动订单并退出归属库存，再按该机制的冷却和恢复条件处理"
    elif transition == "EXITING":
        impact = "后续保持双侧关闭并持续撤单、复核成交和清理归属库存，直至只剩不可成交 dust"
    elif transition == "EXIT_COMPLETE":
        impact = "后续不再承担可成交归属库存的价格风险，并进入冷却、重入或锁存阶段"
    elif transition == "COOLDOWN":
        impact = "后续继续禁止新交易，冷却结束且健康条件满足后才进入重入检查"
    elif transition == "REENTRY":
        impact = "后续等待其他所有风控门放行和连续健康检查，通过后才重建基础库存并恢复策略"
    elif transition == "RECOVERED":
        impact = "后续只解除本机制的限制，最终交易权限仍需其他所有已启用风控门共同放行"
    elif transition == "LATCHED":
        impact = "后续保持禁止重入，必须先修复根因并通过 Hermes 或 OCI 的人工复核流程"
    elif transition == "EXIT_DELAY":
        impact = "后续继续 Fail-Closed 重试撤单和退出，同时保持严重告警，不能提前恢复交易"
    elif transition == "ACTION_FAILED":
        impact = "后续保持 Fail-Closed 并重试或等待人工处理，不会因本次执行失败自动放行"
    else:
        action = " ".join(
            str(event.get("action") or "继续按当前风控阶段处理").split()
        )
        impact = f"后续执行“{action}”，最终权限仍由全部已启用风控门共同决定"
    return f"解释：由于{cause}（原始原因：{reason}）；{impact}。"


def _format_inventory_impact(details: Mapping[str, Any]) -> str:
    runtime = details.get("runtime", {}) if isinstance(details, Mapping) else {}
    robots = runtime.get("robots", {}) if isinstance(runtime, Mapping) else {}
    states = []
    for name, row in robots.items():
        if not isinstance(row, Mapping):
            continue
        phase = str(row.get("phase") or "UNKNOWN")
        running = bool(row.get("running"))
        states.append(f"{name}={phase}/{'运行中' if running else '未运行'}")
    state_text = "，".join(states) if states else "无可信运行状态"
    if any(
        isinstance(row, Mapping) and str(row.get("phase")) == "LATCHED"
        for row in robots.values()
    ):
        effect = "至少一个机器人处于 LATCHED；本库存事件不会解除任何风控门"
    elif runtime.get("trading_normal") is True:
        effect = "当前机器人为 ACTIVE，交易正常；本库存事件不改变交易权限"
    elif robots:
        effect = "当前交易状态受限；最终权限仍由全部已启用风控门共同决定"
    else:
        effect = "无法仅凭本事件判断交易是否正常；本事件不改变其他风控门"
    orders = runtime.get("active_order_count")
    order_text = f"，活动订单={orders}" if orders is not None else ""
    return f"影响：{effect}。运行状态：{state_text}{order_text}。"


def format_event(event: Mapping[str, Any]) -> str:
    if event.get("mechanism") == "runtime_error":
        details = event.get("details", {})
        recovered = event.get("transition") == "ERROR_RECOVERED"
        icon = "🟢" if recovered else ("🔴" if event.get("severity") == "critical" else "🟠")
        lines = [
            f"{icon} 运行错误{'已恢复' if recovered else '告警'}",
            f"组件：{details.get('component') or event.get('source') or '-'}",
            f"机器人：{event.get('bot') or '-'}",
            f"交易对：{event.get('pair') or '-'}",
        ]
        if recovered:
            lines.extend((
                f"原错误：{details.get('error_summary') or '-'}",
                f"持续时间：{float(details.get('duration_seconds', 0)):.1f} 秒",
                f"失败次数：{details.get('occurrences', 1)}",
                f"当前状态：{details.get('trading_status') or '已恢复正常监控'}",
            ))
        else:
            lines.extend((
                f"错误：{details.get('error_summary') or event.get('reason') or '-'}",
                f"影响：{details.get('trading_impact') or '正在自动重试，交易权限由现有风控门决定'}",
                f"动作：{event.get('action') or 'automatic_retry'}",
                "说明：同类错误持续期间不会重复刷屏；恢复后会另发一条通知。",
            ))
        lines.extend((
            f"时间：{event.get('occurred_at')}",
            f"事件ID：{str(event.get('event_id', ''))[:20]}",
        ))
        return "\n".join(lines)[:4096]
    if event.get("mechanism") == "account_inventory":
        details = event.get("details", {})
        transition = str(event.get("transition", ""))
        names = {
            "INVENTORY_UNATTRIBUTED_DETECTED": "发现无归属库存",
            "INVENTORY_DUST_CLASSIFIED": "无归属库存已归类为 Dust",
            "INVENTORY_STATUS_CORRECTED": "库存状态更正",
            "INVENTORY_LIQUIDATION_STARTED": "无归属库存开始清仓",
            "INVENTORY_LIQUIDATION_COMPLETED": "无归属库存清仓完成",
            "INVENTORY_LIQUIDATION_FAILED": "无归属库存清仓失败",
            "INVENTORY_OWNERSHIP_DEFICIT": "库存归属缺口",
            "INVENTORY_RECONCILIATION_RECOVERED": "库存归属恢复健康",
            "INVENTORY_LIQUIDATION_FEE_RECONCILED": "清仓手续费复核完成",
        }
        lines = [
            "🔐 统一库存风控事件",
            f"状态：{names.get(transition, transition)}",
            f"资产/交易对：{event.get('pair') or '-'}",
            f"原因：{event.get('reason') or '-'}",
            f"动作：{event.get('action') or '-'}",
        ]
        for label, key in (
            ("数量", "quantity"), ("USDT到账", "quote_quantity"),
            ("手续费", "fee_quote"), ("订单ID", "order_id"),
        ):
            if details.get(key) not in (None, ""):
                lines.append(f"{label}：{details.get(key)}")
        if details.get("fee_details"):
            fees = ", ".join(
                f"{row.get('commission')} {row.get('asset')}"
                for row in details.get("fee_details", [])
            )
            lines.append(f"手续费资产明细：{fees}")
        is_deficit = transition == "INVENTORY_OWNERSHIP_DEFICIT"
        if details.get("inventory_phase") and not is_deficit:
            lines.append(f"库存阶段：{details.get('inventory_phase')}")
        if is_deficit:
            lines.append(
                "缺口数量/预估金额："
                f"{details.get('deficit_quantity', details.get('ownership_deficit', '-'))} / "
                f"{details.get('deficit_estimated_notional', '-')} USDT"
            )
        elif details.get("tradable_quantity") is not None:
            lines.append(
                "可成交数量/预估金额/最小金额："
                f"{details.get('tradable_quantity')} / "
                f"{details.get('estimated_notional', '-')} / "
                f"{details.get('minimum_notional', '-')}"
            )
        if details.get("dust_reason") and not is_deficit:
            lines.append(f"Dust 原因：{details.get('dust_reason')}")
        confirmation_key = "deficit_confirmation" if is_deficit else "confirmation"
        if details.get(confirmation_key):
            confirmation = details[confirmation_key]
            lines.append(
                f"确认：{confirmation.get('cycles', 0)}/3，"
                f"已确认={confirmation.get('confirmed', False)}"
            )
        lines.extend((
            "影响：交易机器人保持停止和 LATCHED；本事件不会启动交易或解除其他风控。",
            f"时间：{event.get('occurred_at')}",
            f"事件ID：{str(event.get('event_id', ''))[:20]}",
        ))
        lines[-3] = _format_inventory_impact(details)
        return "\n".join(lines)[:4096]
    if event.get("mechanism") == "profit_report":
        lines = [
            "📊 *Grid / DCA 每4小时策略归属 MTM*",
            "",
            "🕒 *北京时间时段*",
            f"`{event.get('details', {}).get('slot', '-')}`",
        ]
        for item in event.get("details", {}).get("robots", []):
            profit = item.get("profit", {})
            quote = item.get("quote_asset", "")
            lines.extend((
                "",
                f"*{str(item.get('strategy', '')).upper()} · {item.get('pair')}*",
                f"- 4h：`{_number(profit.get('four_hour_mtm_quote'))}`",
                f"- 24h：`{_number(profit.get('twenty_four_hour_mtm_quote'))}`",
                f"- 7d：`{_number(profit.get('seven_day_mtm_quote'))}`",
                f"- 累计：`{_number(profit.get('all_time_mtm_quote'))} {quote}`",
            ))
            transport = item.get("runtime_transport", {})
            if str(item.get("strategy", "")).lower() == "grid" and isinstance(
                transport, Mapping
            ):
                lines.append(
                    "- Guard连接瞬时恢复（4h）："
                    f"`{int(transport.get('recovered_episodes', 0) or 0)} 次`"
                    "，交易权限未受影响"
                )
        lines.extend(("", "_不同报价币种不折算、不合并；详见四张单机器人 PNG。_"))
        return MARKDOWN_MESSAGE_PREFIX + "\n".join(lines)[:4096]
    if event.get("mechanism") == "parameter_update":
        lines = [
            "🧪 参数/模型更新报告",
            f"阶段：{event.get('transition')}",
            f"策略/交易对：{event.get('strategy')} / {event.get('pair')}",
            f"原因：{event.get('reason')}",
            f"Release：{event.get('release_sha256') or '-'}",
            f"模型：{event.get('model_sha256') or '-'}",
            f"参数：{event.get('parameter_sha256') or '-'}",
            f"事件ID：{str(event.get('event_id', ''))[:20]}",
        ]
        details = event.get("details", {})
        if event.get("transition") == "MODEL_APPROVAL_PENDING":
            deadline = details.get("review_deadline")
            if deadline:
                shown = datetime.fromtimestamp(int(deadline), timezone.utc).astimezone().isoformat()
                lines.append(f"默认审批截止：{shown}")
            lines.extend((
                "默认行为：截止前无人拒绝且全部硬门槛持续通过，则自动批准；任一校验失败不会自动放行。",
                "审批等待不影响当前模型交易；候选会在边界前隔离预热并原子切换 runtime generation，周边界只切换内部 fold。",
                "",
                "请在 Hermes 私聊中发送以下提示词：",
                str(details.get("prompt") or "请检查该 v22 周模型候选并选择批准或拒绝。"),
            ))
        return "\n".join(lines)[:4096]
    severity = {"critical": "🔴", "warning": "🟠", "info": "🔵"}.get(
        str(event.get("severity", "warning")).lower(), "🟠"
    )
    lines = [
        f"{severity} {event.get('strategy', '').upper()} 风控事件",
        f"机器人：{event.get('bot') or '-'}",
        f"交易对：{event.get('pair') or '-'}",
        f"机制：{event.get('mechanism')}",
        f"转换：{event.get('phase_from') or '-'} → {event.get('phase_to') or event.get('transition')}",
        f"原因：{event.get('reason')}",
        explain_event(event),
        f"动作：{event.get('action') or '-'}",
        f"时间：{event.get('occurred_at')}",
        f"事件ID：{str(event.get('event_id', ''))[:20]}",
    ]
    if event.get("trigger_value") is not None or event.get("threshold") is not None:
        lines.append(f"触发值/阈值：{event.get('trigger_value', '-')} / {event.get('threshold', '-')}")
    if event.get("mechanism") == "fomc_gate":
        details = event.get("details", {})
        if details.get("buy_enabled") is not None or details.get("sell_enabled") is not None:
            lines.append(
                f"限制方向：BUY={'放行' if details.get('buy_enabled') else '关闭'} / "
                f"SELL={'放行' if details.get('sell_enabled') else '关闭'}"
            )
    if event.get("requires_manual_action"):
        lines.extend(("", hermes_recovery_prompt(event)))
    return "\n".join(lines)[:4096]


class TelegramChannelClient:
    def __init__(self, token_file: Path, channel_id: str,
                 *, session: requests.Session | None = None) -> None:
        token = token_file.read_text(encoding="utf-8").strip()
        if not token or not channel_id:
            raise ValueError("Telegram notification token and channel ID are required")
        self.base_url = f"{telegram_api_base()}/bot{token}"
        self.channel_id = str(channel_id)
        self.session = session or requests.Session()

    def _request(self, method: str, *, data: Mapping[str, Any],
                 files: Mapping[str, Any] | None = None) -> str:
        try:
            response = self.session.post(
                f"{self.base_url}/{method}", data=dict(data), files=files, timeout=35,
            )
        except requests.RequestException as exc:
            # requests exceptions normally include the URL, which contains the
            # Bot token. Never persist that exception text in the outbox.
            raise RuntimeError(
                f"Telegram {method} transport failure: {type(exc).__name__}"
            ) from None
        if int(getattr(response, "status_code", 200)) >= 400:
            raise RuntimeError(f"Telegram {method} HTTP {response.status_code}")
        try:
            payload = response.json()
        except ValueError:
            raise RuntimeError(f"Telegram {method} returned invalid JSON") from None
        if not payload.get("ok"):
            raise RuntimeError(f"Telegram {method} failed")
        return str(payload.get("result", {}).get("message_id", ""))

    def send_message(self, text: str) -> str:
        data = {
            "chat_id": self.channel_id,
            "text": text[:4096],
            "disable_web_page_preview": "true",
        }
        if text.startswith(MARKDOWN_MESSAGE_PREFIX):
            data["text"] = text[len(MARKDOWN_MESSAGE_PREFIX):][:4096]
            data["parse_mode"] = "Markdown"
        return self._request("sendMessage", data=data)

    def send_file(self, path: Path, *, caption: str = "", kind: str = "document") -> str:
        if kind not in {"photo", "document"}:
            raise ValueError("Telegram attachment kind must be photo or document")
        field = "photo" if kind == "photo" else "document"
        method = "sendPhoto" if kind == "photo" else "sendDocument"
        mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        with path.open("rb") as stream:
            return self._request(method, data={
                "chat_id": self.channel_id, "caption": caption[:1024],
            }, files={field: (path.name, stream, mime)})


class TelegramOutbox:
    def __init__(self, path: Path, *, channel_id: str = "") -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.channel_id = str(channel_id)
        self.connection = sqlite3.connect(path, timeout=30)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.executescript("""
        CREATE TABLE IF NOT EXISTS outbox (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          dedupe_key TEXT NOT NULL UNIQUE,
          event_id TEXT NOT NULL,
          kind TEXT NOT NULL,
          text TEXT NOT NULL,
          file_path TEXT,
          file_sha256 TEXT,
          status TEXT NOT NULL DEFAULT 'pending',
          attempts INTEGER NOT NULL DEFAULT 0,
          next_attempt REAL NOT NULL,
          created_at REAL NOT NULL,
          sent_at REAL,
          telegram_message_id TEXT,
          last_error TEXT
        );
        CREATE TABLE IF NOT EXISTS source_cursor (
          source_path TEXT PRIMARY KEY,
          offset INTEGER NOT NULL DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS schedule_slot (
          schedule TEXT PRIMARY KEY,
          slot TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS profit_snapshot (
          strategy TEXT NOT NULL,
          pair TEXT NOT NULL,
          observed_at REAL NOT NULL,
          mtm_quote REAL,
          equity REAL,
          drawdown_pct REAL,
          payload_json TEXT NOT NULL,
          PRIMARY KEY(strategy,pair,observed_at)
        );
        CREATE INDEX IF NOT EXISTS profit_snapshot_lookup
          ON profit_snapshot(strategy,pair,observed_at);
        """)
        self.connection.commit()

    def enqueue(self, *, event_id: str, kind: str, text: str,
                file_path: Path | None = None, file_sha256: str = "") -> bool:
        if kind not in {"message", "photo", "document"}:
            raise ValueError("unsupported outbox kind")
        if file_path is not None:
            file_path = file_path.resolve()
            actual = sha256_file(file_path)
            if file_sha256 and actual != file_sha256:
                raise ValueError("notification attachment hash mismatch")
            file_sha256 = actual
        dedupe = canonical_sha256({
            "event_id": event_id, "kind": kind,
            "file_sha256": file_sha256, "channel_id": self.channel_id,
        })
        cursor = self.connection.execute(
            "INSERT OR IGNORE INTO outbox "
            "(dedupe_key,event_id,kind,text,file_path,file_sha256,next_attempt,created_at) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (dedupe, event_id, kind, text, str(file_path) if file_path else None,
             file_sha256 or None, time.time(), time.time()),
        )
        self.connection.commit()
        return cursor.rowcount == 1

    def close(self) -> None:
        self.connection.close()

    def ingest(self, source: Path, *,
               attachment_builder: Callable[[dict[str, Any]], list[dict[str, Any]]] | None = None) -> int:
        source_key = str(source.resolve())
        row = self.connection.execute(
            "SELECT offset FROM source_cursor WHERE source_path=?", (source_key,),
        ).fetchone()
        offset = int(row[0]) if row else 0
        if not source.exists():
            return 0
        size = source.stat().st_size
        if size < offset:
            offset = 0
        added = 0
        with source.open("r", encoding="utf-8") as stream:
            stream.seek(offset)
            while True:
                position = stream.tell()
                line = stream.readline()
                if not line:
                    offset = position
                    break
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    offset = position
                    break
                offset = stream.tell()
                if event.get("schema") != SCHEMA or not mechanism_enabled(str(event.get("mechanism"))):
                    continue
                if attachment_builder is not None:
                    try:
                        event["attachments"] = [
                            *event.get("attachments", []), *attachment_builder(event),
                        ]
                    except Exception as exc:
                        event["severity"] = "critical"
                        event["transition"] = "REPORT_EVIDENCE_MISSING"
                        event["reason"] = (
                            f"{event.get('reason', '')}; 360天或重点窗口PNG证据生成失败："
                            f"{type(exc).__name__}: {exc}"
                        )
                        event.setdefault("details", {})["evidence_complete"] = False
                event_id = str(event["event_id"])
                added += int(self.enqueue(
                    event_id=event_id, kind="message", text=format_event(event),
                ))
                for number, attachment in enumerate(event.get("attachments", []), 1):
                    path = Path(str(attachment["path"]))
                    caption = str(attachment.get("caption") or f"{event.get('pair') or event.get('bot')} 附件 {number}")
                    added += int(self.enqueue(
                        event_id=event_id, kind=str(attachment.get("kind", "document")),
                        text=caption, file_path=path,
                        file_sha256=str(attachment.get("sha256", "")),
                    ))
        self.connection.execute(
            "INSERT INTO source_cursor(source_path,offset) VALUES (?,?) "
            "ON CONFLICT(source_path) DO UPDATE SET offset=excluded.offset",
            (source_key, offset),
        )
        self.connection.commit()
        return added

    def drain(self, client: TelegramChannelClient, *, limit: int = 10,
              now: float | None = None) -> int:
        now = time.time() if now is None else now
        rows = self.connection.execute(
            "SELECT id,kind,text,file_path,file_sha256,attempts FROM outbox "
            "WHERE status='pending' AND next_attempt<=? ORDER BY id LIMIT ?",
            (now, limit),
        ).fetchall()
        sent = 0
        for row_id, kind, text, raw_path, expected_sha, attempts in rows:
            try:
                if kind == "message":
                    message_id = client.send_message(text)
                else:
                    path = Path(raw_path)
                    if not path.is_file() or sha256_file(path) != expected_sha:
                        raise ValueError("queued Telegram attachment changed or disappeared")
                    message_id = client.send_file(path, caption=text, kind=kind)
                self.connection.execute(
                    "UPDATE outbox SET status='sent',sent_at=?,telegram_message_id=?,last_error=NULL "
                    "WHERE id=?", (time.time(), message_id, row_id),
                )
                self.connection.commit()
                sent += 1
                time.sleep(1.05)
            except Exception as exc:
                attempts = int(attempts) + 1
                delay = min(3600, 5 * 2 ** min(attempts - 1, 9))
                self.connection.execute(
                    "UPDATE outbox SET attempts=?,next_attempt=?,last_error=? WHERE id=?",
                    (attempts, time.time() + delay,
                     f"{type(exc).__name__}: {exc}"[:1000], row_id),
                )
                self.connection.commit()
        return sent

    def slot_due(self, *, now: datetime | None = None) -> tuple[bool, str]:
        local = (now or datetime.now(timezone.utc)).astimezone(SHANGHAI)
        hour = local.hour - local.hour % 4
        slot = local.replace(hour=hour, minute=0, second=0, microsecond=0).isoformat()
        row = self.connection.execute(
            "SELECT slot FROM schedule_slot WHERE schedule='profit_4h'"
        ).fetchone()
        return row is None or row[0] != slot, slot

    def mark_slot(self, slot: str) -> None:
        self.connection.execute(
            "INSERT INTO schedule_slot(schedule,slot) VALUES ('profit_4h',?) "
            "ON CONFLICT(schedule) DO UPDATE SET slot=excluded.slot", (slot,),
        )
        self.connection.commit()

    def health(self) -> dict[str, Any]:
        pending, oldest = self.connection.execute(
            "SELECT COUNT(*),MIN(created_at) FROM outbox WHERE status='pending'"
        ).fetchone()
        retrying, max_attempts, last_error = self.connection.execute(
            "SELECT COUNT(*),COALESCE(MAX(attempts),0),MAX(last_error) "
            "FROM outbox WHERE status='pending' AND attempts>0"
        ).fetchone()
        return {
            "pending": int(pending),
            "oldest_pending_age_seconds": max(0.0, time.time() - oldest) if oldest else 0.0,
            "retrying": int(retrying),
            "max_attempts": int(max_attempts),
            "last_error": sanitize_runtime_error(last_error or "") if retrying else None,
        }

    def event_delivery(self, event_id: str) -> dict[str, Any]:
        """Return durable Telegram delivery evidence for one event."""
        rows = self.connection.execute(
            "SELECT kind,status,file_sha256,sent_at,telegram_message_id,last_error "
            "FROM outbox WHERE event_id=? ORDER BY id", (str(event_id),),
        ).fetchall()
        return {
            "event_id": str(event_id),
            "items": [
                {
                    "kind": row[0], "status": row[1], "file_sha256": row[2],
                    "sent_at": row[3], "telegram_message_id": row[4],
                    "last_error": sanitize_runtime_error(row[5] or "") or None,
                }
                for row in rows
            ],
            "all_sent": bool(rows) and all(row[1] == "sent" for row in rows),
        }

    def record_profit(self, report: Mapping[str, Any], *, observed_at: float) -> None:
        profit = report.get("profit", {})
        self.connection.execute(
            "INSERT OR REPLACE INTO profit_snapshot "
            "(strategy,pair,observed_at,mtm_quote,equity,drawdown_pct,payload_json) "
            "VALUES (?,?,?,?,?,?,?)",
            (str(report["strategy"]), str(report["pair"]), float(observed_at),
             profit.get("all_time_mtm_quote"), report.get("equity"),
             report.get("drawdown_pct"),
             json.dumps(dict(report), ensure_ascii=False, default=str)),
        )
        self.connection.execute(
            "DELETE FROM profit_snapshot WHERE observed_at<?", (time.time() - 370 * 86400,),
        )
        self.connection.commit()

    def profit_history(self, strategy: str, pair: str, *, days: int = 7) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            "SELECT observed_at,mtm_quote,equity,drawdown_pct,payload_json "
            "FROM profit_snapshot WHERE strategy=? AND pair=? AND observed_at>=? "
            "ORDER BY observed_at",
            (strategy, pair, time.time() - days * 86400),
        ).fetchall()
        return [{"observed_at": row[0], "mtm_quote": row[1], "equity": row[2],
                 "drawdown_pct": row[3], "payload": json.loads(row[4])} for row in rows]

    def mtm_delta(self, strategy: str, pair: str, hours: int,
                  current: float | None) -> float | None:
        if current is None:
            return None
        row = self.connection.execute(
            "SELECT mtm_quote FROM profit_snapshot WHERE strategy=? AND pair=? "
            "AND observed_at<=? AND mtm_quote IS NOT NULL ORDER BY observed_at DESC LIMIT 1",
            (strategy, pair, time.time() - hours * 3600),
        ).fetchone()
        return None if row is None else float(current) - float(row[0])


def _require_pillow() -> None:
    if Image is None or ImageDraw is None or ImageFont is None:
        raise RuntimeError("Pillow is required only for Telegram image/PDF rendering")


def report_font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    _require_pillow()
    configured = os.getenv("TELEGRAM_REPORT_FONT_PATH", "").strip()
    candidates = [
        configured,
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc" if bold else
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "C:/Windows/Fonts/msyhbd.ttc" if bold else "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return ImageFont.truetype(candidate, size=size)
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def _number(value: Any, suffix: str = "") -> str:
    if value is None or value == "":
        return "无可信数据"
    try:
        return f"{float(value):+,.4f}{suffix}"
    except (TypeError, ValueError):
        return f"{value}{suffix}"


def dust_usdt_display(dust: Mapping[str, Any]) -> str:
    """Format mobile-card dust exclusively in quote-value terms."""
    try:
        return f"约 {float(dust.get('estimated_notional')):.4f} USDT"
    except (TypeError, ValueError):
        return "无可信美元估值"


SYSTEM_HEALTH_DISPLAY = {
    "HEALTHY": "健康",
    "DEGRADED": "部分降级",
    "FAILED": "故障",
    "UNKNOWN": "未知",
}
TRADE_MODE_DISPLAY = {
    "NORMAL": "正常交易",
    "BUY_BLOCKED": "仅买入受限",
    "BOTH_BLOCKED": "买卖均受限",
    "EXITING": "清仓中，暂停交易",
    "COOLDOWN": "冷却中，暂停交易",
    "REENTRY": "等待重入，暂停交易",
    "LATCHED": "已锁存，暂停交易",
    "STOPPED": "机器人已停止",
    "EXECUTION_DEGRADED": "交易异常/自动重建中",
    "UNKNOWN": "交易状态未知",
}
PHASE_DISPLAY = {
    "ACTIVE": "正常交易",
    "EXITING": "清仓中",
    "COOLDOWN": "冷却中",
    "REENTRY": "等待重入",
    "LATCHED": "已锁存",
    "STOPPED": "已停止",
    "UNKNOWN": "未知",
}
GATE_STATE_DISPLAY = {
    "RISK_ON": "模型放行",
    "RISK_OFF": "模型风控",
    "ALLOW": "放行",
    "BLOCK": "阻止",
    "ACTIVE": "正常交易",
    "APPLIED": "已落地",
    "MISMATCH": "未同步",
    "DISABLED": "已关闭",
    "UNAVAILABLE": "不可用",
    "HEALTHY": "挂单正常",
    "EXPECTED_EMPTY": "预期无挂单",
    "INTENTIONAL_IDLE": "门控预期暂停",
    "MISSING": "挂单缺失",
    "RETRYING": "自动重建中",
    "RESTRICTED": "该交易对受限",
    "N/A": "不适用",
}
GATE_REASON_DISPLAY = {
    "long_risk_gate_clear": "模型风险解除",
    "adaptive_structural_relief_confirmed": "结构性恢复已确认",
    "no_active_fomc_window": "无生效FOMC限制",
    "macro_state_healthy": "宏观数据正常",
    "not_triggered": "未触发",
    "all_sources_fresh": "全部数据源正常",
    "not_applicable_to_grid": "Grid不适用",
    "already_classified_dust": "已归类Dust，不影响交易",
    "quote_budget_available": "资金预算充足",
    "runtime_consumed_current_gates": "运行时已应用最新门控",
    "orders_submitted": "挂单已提交",
    "active_orders_confirmed": "活动挂单已确认",
    "expected_orders_missing": "应有挂单但当前为零",
    "pair_order_set_rebuilt": "挂单已自动恢复",
    "technical_buy_gate_blocks_buy": "模型门预期阻止BUY",
    "insufficient_budget_or_inventory_for_legal_order": "余额不足以形成合格订单",
    "runtime_order_status_unavailable": "暂无挂单执行状态",
    "unchanged": "控制器已同步",
    "ACTIVE": "正常交易",
}


def system_health_display(value: Any) -> str:
    text = str(value or "UNKNOWN").upper()
    return SYSTEM_HEALTH_DISPLAY.get(text, text)


def trade_mode_display(value: Any) -> str:
    text = str(value or "UNKNOWN").upper()
    return TRADE_MODE_DISPLAY.get(text, text)


def phase_display(value: Any) -> str:
    text = str(value or "UNKNOWN").upper()
    return PHASE_DISPLAY.get(text, text)


def gate_state_display(value: Any) -> str:
    text = str(value or "UNKNOWN").upper()
    return GATE_STATE_DISPLAY.get(text, str(value or "未知"))


def gate_reason_display(value: Any) -> str:
    text = str(value or "unknown")
    return GATE_REASON_DISPLAY.get(text, text)


def dust_metric(pair: Any, dust: Any) -> tuple[str, str]:
    """Return one stable account-level dust row for every robot card."""
    base_asset = str(pair or "").split("-", 1)[0] or "资产"
    value = dust_usdt_display(dust) if isinstance(dust, Mapping) else "无"
    return f"共享账户 {base_asset} Dust", value


def _runtime_summary(value: Any) -> str:
    if not isinstance(value, Mapping) or not value:
        return "无可信数据"
    parts = []
    for key, label in (("orders", "订单"), ("buy_executors", "BUY执行器"),
                       ("sell_executors", "SELL执行器")):
        if value.get(key) is not None:
            parts.append(f"{label}{value[key]}")
    return " / ".join(parts) if parts else "无可信数据"


def _line(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int],
          values: list[float], color: str) -> None:
    left, top, right, bottom = box
    draw.rounded_rectangle(box, radius=18, fill="#ffffff", outline="#dce3ea", width=2)
    if len(values) < 2:
        draw.text((left + 30, top + 50), "无可信时间序列", fill="#b91c1c",
                  font=report_font(34))
        return
    low, high = min(values), max(values)
    if high == low:
        high += 1
        low -= 1
    points = []
    for index, value in enumerate(values):
        x = left + 30 + int(index / (len(values) - 1) * (right - left - 60))
        y = top + 30 + int((high - value) / (high - low) * (bottom - top - 60))
        points.append((x, y))
    for grid in range(5):
        y = top + 30 + int(grid / 4 * (bottom - top - 60))
        draw.line((left + 30, y, right - 30, y), fill="#e5e7eb", width=2)
    draw.line(points, fill=color, width=5, joint="curve")
    draw.text((left + 30, top + 35), f"高 {high:,.4f}", fill="#64748b", font=report_font(26))
    draw.text((left + 30, bottom - 70), f"低 {low:,.4f}", fill="#64748b", font=report_font(26))


def render_mobile_profit_card(report: Mapping[str, Any], output: Path) -> None:
    """Render one strategy/pair card; never combine Grid and DCA equity."""
    image = Image.new("RGB", MOBILE_CARD_SIZE, "#f5f7fa")
    draw = ImageDraw.Draw(image)
    title = f"{str(report.get('strategy', '')).upper()} {report.get('pair')} 策略归属MTM"
    draw.text((70, 55), title, fill="#172033", font=report_font(52, bold=True))
    draw.text((70, 130), f"北京时间：{report.get('generated_at_bjt', '-')}  数据年龄：{report.get('data_age_seconds', '-')}秒",
              fill="#64748b", font=report_font(28))
    status = report.get("trading_status", {})
    system_health = str(status.get("system_health") or "UNKNOWN")
    trade_mode = str(status.get("trade_mode") or "UNKNOWN")
    trading_normal = status.get("trading_normal") is True
    if system_health == "FAILED":
        status_color, status_fill = "#b91c1c", "#fef2f2"
    elif system_health == "DEGRADED":
        status_color, status_fill = "#b45309", "#fff7ed"
    elif trading_normal:
        status_color, status_fill = "#15803d", "#f0fdf4"
    else:
        status_color, status_fill = "#a16207", "#fefce8"
    draw.rounded_rectangle((70, 180, 1370, 370), radius=18, fill=status_fill,
                           outline=status_color, width=3)
    permissions = status.get("final_permissions", {})
    trade_text = "正常交易" if trading_normal else trade_mode_display(trade_mode)
    draw.text((105, 205), f"系统：{system_health_display(system_health)}   交易状态：{trade_text}",
              fill=status_color, font=report_font(36, bold=True))
    draw.text(
        (105, 262),
        f"最终权限：BUY={'放行' if permissions.get('buy_enabled') else '阻止'} / "
        f"SELL={'放行' if permissions.get('sell_enabled') else '阻止'}   "
        f"阶段：{phase_display(status.get('phase'))}",
        fill="#334155", font=report_font(28),
    )
    generation = str(status.get("runtime_generation") or "-")
    release = str(status.get("release_sha256") or "-")
    cutover_phase = str(status.get("cutover_phase") or "-")
    cutover_label = (
        "候选预热中，当前模型继续交易"
        if cutover_phase == "PREWARMING_CURRENT_MODEL_ACTIVE" else
        "已生效" if cutover_phase == "ACTIVE" else cutover_phase
    )
    draw.text(
        (105, 315),
        f"运行代次：{generation[:12]}  发布版本：{release[:12]}  "
        f"模型周：{status.get('model_week', '-')}  切换状态：{cutover_label}",
        fill="#64748b", font=report_font(23),
    )
    windows = report.get("profit", {})
    cards = [
        ("最近4小时", windows.get("four_hour_mtm_quote")),
        ("最近24小时", windows.get("twenty_four_hour_mtm_quote")),
        ("最近7天", windows.get("seven_day_mtm_quote")),
        ("上线以来", windows.get("all_time_mtm_quote")),
    ]
    for index, (label, value) in enumerate(cards):
        left = 70 + (index % 2) * 650
        top = 410 + (index // 2) * 170
        draw.rounded_rectangle((left, top, left + 600, top + 150), radius=18,
                               fill="#ffffff", outline="#dce3ea", width=2)
        draw.text((left + 28, top + 20), label, fill="#64748b", font=report_font(28))
        try:
            non_negative = value is not None and float(value) >= 0
            color = "#15803d" if non_negative else "#b91c1c"
        except (TypeError, ValueError):
            color = "#64748b"
        draw.text((left + 28, top + 68), _number(value, f" {report.get('quote_asset', '')}"),
                  fill=color, font=report_font(38, bold=True))
    metrics = [
        ("当前权益", _number(report.get("equity"), f" {report.get('quote_asset', '')}")),
        ("峰值权益", _number(report.get("peak_equity"), f" {report.get('quote_asset', '')}")),
        ("峰值回撤", _number(report.get("drawdown_pct"), "%")),
        ("归属基础币", str(report.get("owned_base") or "无可信数据")),
        ("费用", _number(report.get("fees_quote"), f" {report.get('quote_asset', '')}")),
        ("买/卖成交", f"{report.get('buys', '-')} / {report.get('sells', '-') }"),
        ("恢复状态", phase_display(report.get("phase"))),
        ("活动订单/执行器", _runtime_summary(report.get("active_runtime", {}))),
    ]
    order_runtime = report.get("active_runtime", {})
    if report.get("strategy") == "grid" and isinstance(order_runtime, Mapping):
        metrics.append((
            "挂单执行",
            f"{order_runtime.get('order_build_state', 'UNKNOWN')} "
            f"B{order_runtime.get('actual_buy_layers', '-')}/"
            f"{order_runtime.get('expected_buy_layers', '-')} "
            f"S{order_runtime.get('actual_sell_layers', '-')}/"
            f"{order_runtime.get('expected_sell_layers', '-')}",
        ))
    grid_parameters = report.get("grid_parameters")
    if isinstance(grid_parameters, Mapping) and grid_parameters:
        profile_labels = {
            "medium_sideways": "中短横盘",
            "long_volatility": "长期波动",
            "legacy_shared": "原固定参数",
            "configured_shared": "原固定参数",
        }
        profile = profile_labels.get(
            str(grid_parameters.get("profile")), str(grid_parameters.get("profile") or "未知")
        )
        try:
            range_pct = float(grid_parameters.get("grid_range")) * 100.0
            minimum = float(grid_parameters.get("minimum_order_quote"))
            parameter_value = (
                f"{profile} {range_pct:.2f}%/{grid_parameters.get('grid_levels', '-')}格 "
                f"B{grid_parameters.get('effective_buy_layers', '-')}/"
                f"S{grid_parameters.get('effective_sell_layers', '-')} ≥{minimum:g} "
                f"#{str(grid_parameters.get('parameter_sha256') or '-')[:8]}"
            )
        except (TypeError, ValueError):
            parameter_value = "无可信逐交易对参数"
        metrics.append(("Grid参数", parameter_value))
    metrics.append(dust_metric(report.get("pair"), report.get("unattributed_dust")))
    top = 790
    draw.rounded_rectangle((70, top, 1370, top + 390), radius=18,
                           fill="#ffffff", outline="#dce3ea", width=2)
    for index, (label, value) in enumerate(metrics):
        x = 105 + (index % 2) * 650
        y = top + 30 + (index // 2) * 82
        draw.text((x, y), f"{label}：{value}", fill="#334155", font=report_font(29))
    draw.text((70, 1220), "全部有效门控", fill="#172033", font=report_font(34, bold=True))
    table_top = 1270
    draw.rounded_rectangle((70, table_top, 1370, table_top + 710), radius=18,
                           fill="#ffffff", outline="#dce3ea", width=2)
    headers = (("机制", 90), ("开关", 390), ("状态", 520),
               ("BUY", 720), ("SELL", 840), ("原因", 970))
    for label, x in headers:
        draw.text((x, table_top + 16), label, fill="#172033",
                  font=report_font(24, bold=True))
    draw.line((85, table_top + 55, 1355, table_top + 55), fill="#cbd5e1", width=2)
    for index, row in enumerate(status.get("gate_statuses", [])[:13]):
        y = table_top + 65 + index * 49
        if index % 2:
            draw.rectangle((82, y - 5, 1358, y + 43), fill="#f8fafc")
        blocked = (
            row.get("health") != "HEALTHY"
            or row.get("buy_enabled") is False or row.get("sell_enabled") is False
        )
        color = "#b91c1c" if row.get("health") == "FAILED" else (
            "#a16207" if blocked else "#166534"
        )
        draw.text((90, y), str(row.get("label") or row.get("mechanism"))[:15],
                  fill="#334155", font=report_font(22))
        switch = "N/A" if not row.get("applicable") else (
            "开" if row.get("enabled") else "关"
        )
        draw.text((390, y), switch, fill="#334155", font=report_font(22))
        draw.text((520, y), gate_state_display(row.get("state"))[:14],
                  fill=color, font=report_font(22, bold=True))
        draw.text((720, y), "放行" if row.get("buy_enabled") is True else
                  "阻止" if row.get("buy_enabled") is False else "N/A",
                  fill=color, font=report_font(22))
        draw.text((840, y), "放行" if row.get("sell_enabled") is True else
                  "阻止" if row.get("sell_enabled") is False else "N/A",
                  fill=color, font=report_font(22))
        draw.text((970, y), gate_reason_display(row.get("reason"))[:30],
                  fill="#475569", font=report_font(20))
    draw.text((70, 2020), "单机器人连续权益", fill="#172033", font=report_font(34, bold=True))
    _line(draw, (70, 2070, 1370, 2430), [float(x) for x in report.get("equity_series", [])], "#2563eb")
    draw.text((70, 2470), "单机器人回撤（%）", fill="#172033", font=report_font(34, bold=True))
    _line(draw, (70, 2520, 1370, 2820), [float(x) for x in report.get("drawdown_series", [])], "#dc2626")
    warnings = report.get("warnings", [])
    warning = "；".join(str(item) for item in warnings) if warnings else "数据完整；未跨策略合并权益。"
    sources = " / ".join(str(value) for value in report.get("data_sources", []))
    source_text = f"数据源：{sources or '无可信数据源'}"
    for index, line in enumerate((source_text[:70], source_text[70:140])):
        if line:
            draw.text((70, 2860 + index * 36), line, fill="#64748b",
                      font=report_font(23))
    for index, line in enumerate((warning[:68], warning[68:136])):
        if line:
            draw.text((70, 2940 + index * 36), line,
                      fill="#b45309" if warnings else "#64748b",
                      font=report_font(24))
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output, "PNG", optimize=True)


def render_analysis_pdf(report: Mapping[str, Any], output: Path) -> None:
    """Render a UTF-8 Chinese mobile report as rasterized PDF pages."""
    width, height = 1240, 1754
    pages: list[Image.Image] = []

    def new_page(*, continued: bool = False) -> tuple[Image.Image, ImageDraw.ImageDraw, int]:
        page = Image.new("RGB", (width, height), "#ffffff")
        canvas = ImageDraw.Draw(page)
        title = str(report.get("title", "参数分析报告")) + ("（续）" if continued else "")
        canvas.text((70, 70), title, fill="#172033", font=report_font(44, bold=True))
        return page, canvas, 155

    image, draw, y = new_page()
    rows = [
        ("报告ID", report.get("report_id")),
        ("状态", report.get("status")),
        ("生成时间", report.get("generated_at")),
        ("参数版本", report.get("parameter_version")),
        ("Release", report.get("release_sha256")),
        ("模型哈希", report.get("model_sha256")),
        ("360天证据", "完整" if report.get("evidence_complete") else "缺失或不完整"),
        ("结论", report.get("conclusion")),
    ]
    for label, value in rows:
        draw.text((70, y), f"{label}：{value or '-'}", fill="#334155", font=report_font(28))
        y += 54
    y += 20
    sections = report.get("sections", [])
    for section in sections:
        if y > height - 180:
            pages.append(image)
            image, draw, y = new_page(continued=True)
        draw.text((70, y), str(section.get("title", "分析")), fill="#172033",
                  font=report_font(32, bold=True))
        y += 55
        text = str(section.get("text", ""))
        while text:
            line, text = text[:46], text[46:]
            draw.text((85, y), line, fill="#475569", font=report_font(25))
            y += 43
            if y > height - 100:
                pages.append(image)
                image, draw, y = new_page(continued=True)
        y += 28
    pages.append(image)
    output.parent.mkdir(parents=True, exist_ok=True)
    pages[0].save(
        output, "PDF", resolution=150.0, save_all=True,
        append_images=pages[1:],
    )

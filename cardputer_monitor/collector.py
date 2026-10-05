from __future__ import annotations

import json
import math
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from live_guard.trading_status import GATE_LABELS
from management_bot.clients import OperationsReportReader
from management_bot.risk_display import RULES


SOURCE_ID = "hummingbot-main"
STATUS_SCHEMA = "grid-dca-trading-status-collection-v1"
MAX_AGE_SECONDS = 300
FUTURE_TOLERANCE_SECONDS = 30
MAX_SOURCE_BYTES = 1024 * 1024
MAX_BLOCKERS = 8
MAX_GATES = 16
ROBOTS = (
    ("grid", "BTC-FDUSD", "Grid BTC", "grid-live-fdusd-400"),
    ("grid", "ETH-FDUSD", "Grid ETH", "grid-live-fdusd-400"),
    ("dca", "BTC-USDT", "DCA BTC", "dca-live-btcusdt-200"),
    ("dca", "ETH-USDT", "DCA ETH", "dca-live-ethusdt-200"),
)
PROFIT_KEYS = {
    "4h": "four_hour_mtm_quote", "24h": "twenty_four_hour_mtm_quote",
    "7d": "seven_day_mtm_quote", "all": "all_time_mtm_quote",
}
PHASES = {"ACTIVE", "EXITING", "COOLDOWN", "REENTRY", "LATCHED", "UNKNOWN"}
TRADE_MODES = PHASES | {"NORMAL", "BUY_BLOCKED", "BOTH_BLOCKED", "STOPPED", "EXECUTION_DEGRADED"}
HEALTH_STATES = {"HEALTHY", "DEGRADED", "FAILED", "UNKNOWN"}
GATE_STATES = PHASES | {
    "ALLOW", "BLOCK", "OK", "DISABLED", "N/A", "RISK_ON", "RISK_OFF",
    "UNAVAILABLE", "APPLIED", "MISMATCH", "ALERT_ONLY", "LONG_ONLY", "BILATERAL",
    "RETRYING", "TRIGGERED", "HEALTHY", "FAILED", "WAITING", "IDLE", "STOPPED",
}


def number(value: Any) -> float | None:
    """Preserve a real zero; reject booleans and non-finite numeric values."""
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def timestamp(value: Any) -> float | None:
    if isinstance(value, str) and not value.replace(".", "", 1).isdigit():
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                return None
            return number(parsed.timestamp())
        except (ValueError, TypeError, OverflowError, OSError):
            return None
    result = number(value)
    return result if result is not None and result > 0 else None


def data_state(observed_at: float | None, now: float) -> str:
    if observed_at is None or observed_at > now + FUTURE_TOLERANCE_SECONDS:
        return "UNAVAILABLE"
    return "FRESH" if now - observed_at <= MAX_AGE_SECONDS else "STALE"


def boolean(value: Any) -> bool | None:
    return value if type(value) is bool else None


def enum(value: Any, choices: set[str]) -> str | None:
    return value if isinstance(value, str) and value in choices else None


def reason_text(value: Any, mechanism: str = "") -> str:
    """Map source reasons to public text; never export an arbitrary log string."""
    text = str(value or "").lower()
    if "stale" in text or "missing" in text or "unavailable" in text:
        return "缺少新鲜可信数据，等待来源恢复"
    if "hash" in text or "integrity" in text:
        return "完整性检查未通过"
    if "ownership_deficit" in text:
        return "库存归属核对存在缺口"
    if "signed_week" in text:
        return "等待有效签名周模型"
    if "disabled" in text and mechanism == "recovery_phase_gate":
        return "自动重入未开启，等待恢复授权"
    if mechanism == "capital_budget_gate":
        return "资金预算告警；以最终交易权限为准"
    if mechanism == "strategy_mode_gate":
        return "只做多模式仍允许保护性退出" if "protective" in text else "以控制器最终交易权限为准"
    return RULES.get(mechanism, ("等待权威状态确认", ""))[0][:120]


def sanitize_gates(value: Any) -> list[dict]:
    if not isinstance(value, list):
        return []
    result = []
    for row in value[:MAX_GATES]:
        if not isinstance(row, dict):
            continue
        mechanism = row.get("mechanism")
        if not isinstance(mechanism, str) or mechanism not in GATE_LABELS:
            continue
        healthy = (row.get("buy_enabled") is True and row.get("sell_enabled") is True
                   and row.get("health", "HEALTHY") == "HEALTHY"
                   and row.get("state") != "ALERT_ONLY")
        result.append({
            "mechanism": mechanism[:64], "label": GATE_LABELS[mechanism][:40],
            "enabled": boolean(row.get("enabled")), "applicable": boolean(row.get("applicable")),
            "health": enum(row.get("health"), HEALTH_STATES),
            "state": enum(row.get("state"), GATE_STATES),
            "buy_enabled": boolean(row.get("buy_enabled")),
            "sell_enabled": boolean(row.get("sell_enabled")),
            "reason": "检查通过" if healthy else reason_text(row.get("reason"), mechanism),
        })
    return result


def sanitize_recovery(value: Any, now: float) -> dict:
    if not isinstance(value, dict):
        return {}
    result = {}
    mechanism = value.get("mechanism")
    if isinstance(mechanism, str) and mechanism in GATE_LABELS:
        result["mechanism"] = mechanism
    phase = enum(value.get("phase"), PHASES)
    if phase is not None:
        result["phase"] = phase
    for key in ("cooldown_until", "exit_completed_at"):
        observed = timestamp(value.get(key))
        if observed is not None and (key == "cooldown_until" or observed <= now + FUTURE_TOLERANCE_SECONDS):
            result[key] = observed
    cycles = value.get("healthy_cycles")
    if type(cycles) is int and 0 <= cycles <= 100000:
        result["healthy_cycles"] = cycles
    if value.get("reentry_block_reason"):
        result["reentry_block_reason"] = reason_text(value["reentry_block_reason"], "recovery_phase_gate")
    return result


def empty_robot(strategy: str, pair: str, name: str, bot: str) -> dict:
    return {
        "id": f"{strategy}:{pair}", "name": name, "bot_name": bot, "pair": pair,
        "quote_asset": pair.split("-")[-1], "status": "UNAVAILABLE",
        "status_observed_at": None, "status_data_state": "UNAVAILABLE",
        "process_running": None, "trade_mode": None, "system_health": None, "phase": None,
        "buy_enabled": None, "sell_enabled": None,
        "blockers": ["缺少可信状态数据"], "gates": [], "recovery": {},
        "profit_observed_at": None, "profit_data_state": "UNAVAILABLE",
        "profit": {key: None for key in PROFIT_KEYS},
        "window_complete": {key: False for key in PROFIT_KEYS if key != "all"},
        "equity": None, "drawdown_pct": None,
    }


def index_rows(rows: Any) -> dict[tuple[str, str], dict]:
    """Duplicate source identities invalidate that unit instead of choosing one."""
    indexed = {}
    duplicates = set()
    if not isinstance(rows, list):
        return indexed
    for row in rows:
        if not isinstance(row, dict):
            continue
        strategy, pair = row.get("strategy"), row.get("pair")
        if not isinstance(strategy, str) or not isinstance(pair, str):
            continue
        key = strategy, pair
        if key in indexed:
            duplicates.add(key)
        else:
            indexed[key] = row
    return {key: row for key, row in indexed.items() if key not in duplicates}


class SnapshotCollector:
    def __init__(self, reports_dir: Path, *, reader: OperationsReportReader | None = None, history=None):
        self.status_path = Path(reports_dir) / "trading_status.json"
        self.reader = reader or OperationsReportReader(
            self.status_path, Path(reports_dir) / "telegram_outbox.sqlite", MAX_AGE_SECONDS,
        )
        self.history = history

    def _statuses(self, now: float) -> tuple[dict, float | None, str]:
        # A bounded re-read protects schema validation if atomic replacement
        # occurs between this read and OperationsReportReader.status().
        for _ in range(2):
            try:
                with self.status_path.open("rb") as source:
                    raw = source.read(MAX_SOURCE_BYTES + 1)
                if len(raw) > MAX_SOURCE_BYTES:
                    return {}, None, "UNAVAILABLE"
                document = json.loads(raw)
                if not isinstance(document, dict) or document.get("schema") != STATUS_SCHEMA:
                    return {}, None, "UNAVAILABLE"
                observed = timestamp(document.get("generated_at"))
                state = data_state(observed, now)
                if state != "FRESH":
                    # Do not let one invalid future source timestamp make the
                    # cloud reject the otherwise usable four-unit batch.
                    return {}, observed if state == "STALE" else None, state
                status = self.reader.status()
                if status.get("generated_at") == observed and status.get("robots") == document.get("robots"):
                    return index_rows(status.get("robots")), observed, state
            except (OSError, ValueError, TypeError, RuntimeError, OverflowError):
                return {}, None, "UNAVAILABLE"
        return {}, None, "UNAVAILABLE"

    def _profits(self) -> dict:
        try:
            return index_rows(self.reader.profits().get("robots"))
        except (OSError, ValueError, TypeError, RuntimeError, OverflowError):
            return {}

    @staticmethod
    def _apply_status(target: dict, source: dict, observed: float, now: float) -> None:
        if source.get("bot") != target["bot_name"]:
            return
        row_observed = timestamp(source.get("generated_at"))
        if row_observed is None or row_observed != observed:
            return
        process = boolean(source.get("process_running"))
        normal = boolean(source.get("trading_normal"))
        mode = enum(source.get("trade_mode"), TRADE_MODES)
        health = enum(source.get("system_health"), HEALTH_STATES)
        phase = enum(source.get("phase"), PHASES)
        permissions = source.get("final_permissions")
        if process is None or normal is None or mode is None or health is None or phase is None or not isinstance(permissions, dict):
            return
        buy, sell = boolean(permissions.get("buy_enabled")), boolean(permissions.get("sell_enabled"))
        if normal and not (process and mode == "NORMAL" and health == "HEALTHY" and phase == "ACTIVE" and buy and sell):
            return
        gates = sanitize_gates(source.get("gate_statuses"))
        blockers = [f"{row['label']}：{row['reason']}"[:120] for row in gates
                    if row["enabled"] is True and row["applicable"] is True
                    and (row["buy_enabled"] is not True or row["sell_enabled"] is not True or row["health"] != "HEALTHY")]
        # Canonical blockers remain visible even if a future mechanism has no
        # public gate definition; raw reason/log text is never exported.
        canonical = source.get("blockers", [])
        if not blockers and isinstance(canonical, list) and canonical:
            blockers.append("存在交易限制，等待权威风控状态确认")
        if not process:
            blockers.insert(0, "机器人进程已停止")
        status = "NORMAL" if normal else "STOPPED" if not process else "UNAVAILABLE" if mode == "UNKNOWN" else "RESTRICTED"
        target.update({
            "status": status, "status_observed_at": observed, "status_data_state": "FRESH",
            "process_running": process, "trade_mode": mode, "system_health": health, "phase": phase,
            "buy_enabled": buy, "sell_enabled": sell, "blockers": blockers[:MAX_BLOCKERS],
            "gates": gates, "recovery": sanitize_recovery(source.get("recovery"), now),
        })

    @staticmethod
    def _apply_profit(target: dict, source: dict, now: float) -> None:
        observed = timestamp(source.get("observed_at"))
        state = data_state(observed, now)
        if state == "UNAVAILABLE":
            observed = None
        target.update({"profit_observed_at": observed, "profit_data_state": state})
        if state != "FRESH":
            return
        values = source.get("profit")
        complete = source.get("window_complete")
        if not isinstance(values, dict) or not isinstance(complete, dict):
            target["profit_data_state"] = "UNAVAILABLE"
            return
        cumulative = number(values.get(PROFIT_KEYS["all"]))
        if cumulative is None:
            target["profit_data_state"] = "UNAVAILABLE"
            return
        target["profit"]["all"] = cumulative
        for key, field in PROFIT_KEYS.items():
            if key == "all":
                continue
            amount = number(values.get(field))
            valid = complete.get(field) is True and amount is not None
            target["window_complete"][key] = valid
            target["profit"][key] = amount if valid else None
        target["equity"] = number(source.get("equity"))
        target["drawdown_pct"] = number(source.get("drawdown_pct"))

    def collect(self, *, now: float | None = None) -> dict:
        collected = time.time() if now is None else number(now)
        if collected is None or collected <= 0:
            raise ValueError("collection time is invalid")
        statuses, observed, state = self._statuses(collected)
        profits = self._profits()
        robots = []
        for strategy, pair, name, bot in ROBOTS:
            row = empty_robot(strategy, pair, name, bot)
            row.update({"status_observed_at": observed, "status_data_state": state if state != "FRESH" else "UNAVAILABLE"})
            if state == "STALE":
                row["blockers"] = ["状态源数据已过期"]
            if (strategy, pair) in statuses:
                self._apply_status(row, statuses[(strategy, pair)], observed, collected)
            if (strategy, pair) in profits:
                self._apply_profit(row, profits[(strategy, pair)], collected)
            robots.append(row)
        if self.history is not None:
            histories = self.history.collect(now=collected)
            for row in robots:
                row["history"] = histories[row["id"]]
        return {"schema_version": 1, "source_id": SOURCE_ID, "sample_id": str(uuid.uuid4()), "collected_at": collected, "robots": robots}

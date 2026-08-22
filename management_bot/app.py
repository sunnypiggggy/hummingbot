from __future__ import annotations

import hashlib
import json
import logging
import os
import signal
import time
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation, ROUND_DOWN
from typing import Any, Optional

from management_bot.approvals import ApprovalStore
from management_bot.clients import (
    ContractReader, HummingbotClient, OperationsReportReader, StocksClient,
)
from management_bot.config import Settings
from management_bot.storage import BotStore
from management_bot.telegram_api import TelegramAPI, TelegramError


logger = logging.getLogger("trading-management-bot")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(message)s")
BEIJING_TZ = timezone(timedelta(hours=8))


HOME_ROWS = [
    [("📊 系统总览", "m:overview"), ("💰 盈亏报告", "m:profit")],
    [("🟦 Grid", "m:bot:grid"), ("🟩 DCA", "m:dca")],
    [("📈 Stock", "m:stock"), ("🛡 风控状态", "m:risk")],
    [("🧠 模型审批", "m:approvals"), ("⚠️ 当前异常", "m:errors")],
    [("⚙️ 模型与参数", "m:models"), ("🔧 系统维护", "m:maintenance")],
]


def _d(value: Any, default: str = "0") -> Decimal:
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError):
        raise ValueError("请输入有效数字")
    if not result.is_finite() or result <= 0:
        raise ValueError("数值必须大于0")
    return result


def _quote_price(payload: dict, side: str) -> Decimal:
    bid = Decimal(str(payload.get("bidPrice", payload.get("bid", "0")) or "0"))
    ask = Decimal(str(payload.get("askPrice", payload.get("ask", "0")) or "0"))
    price = ask if side == "BUY" else bid
    if price <= 0:
        price = bid if bid > 0 else ask
    if price <= 0:
        raise ValueError("没有可用的最新报价")
    return price


def _quote_timestamp(payload: dict) -> Optional[float]:
    for key in ("T", "E", "eventTime", "quoteTime", "updateTime", "timestamp", "time"):
        value = payload.get(key)
        if value in (None, ""):
            continue
        try:
            result = float(value)
        except (TypeError, ValueError):
            try:
                return datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp()
            except ValueError:
                continue
        if result > 10**12:
            result /= 1000
        return result
    return None


def _safe_text(value: Any, limit: int = 500) -> str:
    return str(value).replace("`", "'")[:limit]


def _permission_cn(value: Any) -> str:
    if value is True:
        return "放行"
    if value is False:
        return "阻止"
    return "不适用"


def _state_cn(value: Any) -> str:
    raw = str(value or "未知").upper()
    return {
        "RISK_ON": "正常（Risk-On）", "RISK_OFF": "风险关闭（Risk-Off）",
        "ALLOW": "放行", "BLOCK": "阻止", "BLOCKED": "阻止",
        "HEALTHY": "健康", "FAILED": "故障", "UNAVAILABLE": "不可用",
        "ACTIVE": "正常", "EXITING": "清仓中", "COOLDOWN": "冷却中",
        "REENTRY": "等待重入", "LATCHED": "锁存", "APPLIED": "已落地",
        "ALERT_ONLY": "提醒（不阻塞）", "LONG_ONLY": "只做多模式",
        "MARKET_OPEN": "正常交易时段", "PRE_MARKET": "盘前",
        "AFTER_HOURS": "盘后", "OVERNIGHT": "夜盘", "MARKET_CLOSED": "休市",
        "TRADING": "可交易", "UNKNOWN": "未知",
        "N/A": "不适用",
    }.get(raw, str(value or "未知"))


def _reason_cn(value: Any) -> str:
    raw = str(value or "").strip()
    return {
        "long_risk_gate_clear": "模型允许新增BUY",
        "no_active_fomc_window": "当前无FOMC限制窗口",
        "macro_state_healthy": "宏观状态正常",
        "not_triggered": "未触发",
        "all_sources_fresh": "全部数据源新鲜",
        "already_classified_dust": "仅有已归类Dust，不影响交易",
        "orders_submitted": "挂单已正常提交",
        "ACTIVE": "处于正常交易阶段",
        "runtime_consumed_current_gates": "运行时已应用当前门控",
        "unchanged": "控制器已同步",
        "insufficient_quote_budget": "报价币预算不足，仅告警",
        "ordinary_sell_creation_disabled_protective_exits_allowed": (
            "普通SELL建仓关闭，保护性退出仍允许"
        ),
    }.get(raw, _safe_text(raw or "无附加原因", 140))


class TradingManagementBot:
    def __init__(self, settings: Settings):
        self.settings = settings
        settings.state_dir.mkdir(parents=True, exist_ok=True)
        settings.approval_decision_root.mkdir(parents=True, exist_ok=True)
        self.store = BotStore(settings.state_dir / "management_bot.sqlite", settings.session_ttl_seconds)
        self.telegram = TelegramAPI(settings.read_token())
        self.hummingbot = HummingbotClient(
            settings.hummingbot_api_url,
            settings.hummingbot_api_username,
            settings.hummingbot_api_password,
        )
        self.stocks = StocksClient(
            settings.stocks_api_url,
            settings.stocks_api_username,
            settings.stocks_api_password,
        )
        self.contracts = ContractReader(
            settings.grid_guard_path, settings.dca_guard_path, settings.inventory_path
        )
        self.reports = OperationsReportReader(
            settings.trading_status_path,
            settings.profit_snapshot_db_path,
            settings.operations_report_max_age_seconds,
        )
        self.approvals = ApprovalStore(
            settings.approval_request_root,
            settings.approval_evidence_root,
            settings.approval_decision_root,
        )
        self.running = True

    @property
    def mode_label(self) -> str:
        return "执行模式" if self.settings.mutations_enabled else "只读观察模式"

    def stop(self, *_: Any) -> None:
        self.running = False

    def _authorized(self, user_id: int, chat_id: int, chat_type: str) -> bool:
        return user_id == self.settings.admin_user_id and chat_id == user_id and chat_type == "private"

    def _home(self) -> tuple[str, list[list[tuple[str, str]]]]:
        return (
            "🐷 交易系统维护管理 Bot V3\n\n"
            f"当前：{self.mode_label}\n"
            "Grid / DCA / Stock 的交易判断仍由各自风控容器负责。\n"
            "请选择功能：",
            HOME_ROWS,
        )

    @staticmethod
    def _back(target: str = "m:home") -> list[list[tuple[str, str]]]:
        return [[("⬅️ 返回", target), ("🏠 主菜单", "m:home")]]

    def _edit_or_send(self, chat_id: int, message_id: Optional[int], text: str,
                      rows: Optional[list[list[tuple[str, str]]]] = None) -> dict:
        if message_id:
            try:
                value = self.telegram.edit(chat_id, message_id, text, rows)
                return value if isinstance(value, dict) else {"message_id": message_id}
            except TelegramError:
                pass
        return self.telegram.send(chat_id, text, rows)

    def _api_bots(self) -> dict:
        value = self.hummingbot.status().get("data", {})
        return value if isinstance(value, dict) else {}

    @staticmethod
    def _bot_line(name: str, raw: dict) -> str:
        status = str(raw.get("status", "unknown")).lower()
        chinese = {
            "running": "运行中", "stopped": "已停止", "starting": "启动中",
            "stopping": "停止中", "error": "错误",
        }.get(status, "状态未知")
        errors = raw.get("error_logs", [])
        count = len(errors) if isinstance(errors, list) else 0
        return f"• {name}：{chinese}" + (f"，错误{count}条" if count else "")

    def _overview(self) -> str:
        try:
            bots = self._api_bots()
            lines = ["📊 系统总览", f"模式：{self.mode_label}", ""]
            wanted = {v["bot_name"] for v in self.settings.bots.values()}
            for name in sorted(wanted):
                lines.append(self._bot_line(name, bots.get(name, {})))
            try:
                stock = self.stocks.health()
                stock_ok = stock.get("status") == "healthy"
                lines.append(f"• Stock Runtime：{'健康' if stock_ok else '降级'} / {stock.get('runtime_mode', '-')}")
            except Exception as exc:
                lines.append(f"• Stock Runtime：不可用（{type(exc).__name__}）")
            contracts = self.contracts.snapshot()
            lines.append(f"• 风控合同：{'正常' if not contracts['errors'] else '存在缺失'}")
            return "\n".join(lines)
        except Exception as exc:
            return f"📊 系统总览\n\n数据不可用：{_safe_text(exc)}"

    def _profit(self) -> str:
        try:
            snapshot = self.reports.profits()
        except Exception as exc:
            snapshot = None
            reports_error = _safe_text(exc, 160)
        else:
            reports_error = None

        def amount(value: Any, quote: str) -> str:
            try:
                return f"{Decimal(str(value)):+.4f} {quote}"
            except (InvalidOperation, TypeError, ValueError):
                return "无可信数据"

        lines = ["💰 Grid / DCA / Stock 收益"]
        if snapshot is None:
            lines.append(f"\n🟦 Grid / 🟩 DCA\n• 数据不可用：{reports_error}")
        else:
            lines.append(f"Grid/DCA数据年龄：{int(snapshot['age_seconds'])}秒")
            for strategy, title in (("grid", "🟦 Grid"), ("dca", "🟩 DCA")):
                lines.extend(("", title))
                rows = [row for row in snapshot["robots"] if row["strategy"] == strategy]
                for row in rows:
                    quote = row["pair"].split("-")[-1]
                    profit = row.get("profit", {})
                    lines.append(
                        f"• {row['pair']}\n"
                        f"  4h {amount(profit.get('four_hour_mtm_quote'), quote)}｜"
                        f"24h {amount(profit.get('twenty_four_hour_mtm_quote'), quote)}\n"
                        f"  7d {amount(profit.get('seven_day_mtm_quote'), quote)}｜"
                        f"累计 {amount(profit.get('all_time_mtm_quote'), quote)}"
                    )
        lines.extend(("", "🧪 Stock PAPER"))
        try:
            paper = self.stocks.paper_summary()
            if not paper.get("valuation_complete") or not paper.get("reconciliation", {}).get("ok"):
                lines.append("• 收益数据无法对账，暂不展示收益数值。")
            else:
                windows = paper.get("windows", {})
                def paper_window(name: str) -> str:
                    window = windows.get(name, {})
                    suffix = "" if window.get("window_complete") else "（运行期不足）"
                    return f"{amount(window.get('pnl'), 'USDC')}{suffix}"
                account = paper.get("account", {})
                lines.append(
                    f"• 4h {paper_window('4h')}｜24h {paper_window('24h')}\n"
                    f"  7d {paper_window('7d')}｜累计 {paper_window('all')}\n"
                    f"  权益 {amount(account.get('equity'), 'USDC').lstrip('+')}｜"
                    f"峰值 {amount(account.get('peak_equity'), 'USDC').lstrip('+')}｜"
                    f"回撤 {Decimal(str(account.get('drawdown_pct', 0))):.4f}%"
                )
        except Exception as exc:
            lines.append(f"• 数据不可用：{_safe_text(exc, 160)}")
        lines.append(
            "\n口径：Grid/DCA为机器人归属MTM；Stock为独立2000 USDC Paper账户，币种不合并。"
        )
        return "\n".join(lines)

    def _risk(self) -> str:
        try:
            snapshot = self.reports.status()
        except Exception as exc:
            return f"🛡 风控状态\n\n数据不可用：{_safe_text(exc)}"
        robots = snapshot["robots"]
        normal = sum(1 for item in robots if item.get("trading_normal"))
        lines = [
            "🛡 Grid / DCA 风控状态",
            f"数据年龄：{int(snapshot['age_seconds'])}秒",
            f"结论：{'✅' if normal == len(robots) else '🔴'} {normal}/{len(robots)} 正常交易",
            "",
        ]
        for robot in robots:
            strategy = str(robot.get("strategy", "")).upper()
            pair = str(robot.get("pair", "未知交易对"))
            permissions = robot.get("final_permissions", {})
            buy = "放行" if permissions.get("buy_enabled") else "阻止"
            sell = "放行" if permissions.get("sell_enabled") else "阻止"
            alerts = sum(
                1 for gate in robot.get("gate_statuses", [])
                if isinstance(gate, dict) and gate.get("applicable", True)
                and gate.get("state") == "ALERT_ONLY"
            )
            state = "正常交易" if robot.get("trading_normal") else "交易受限"
            suffix = f"｜{alerts}项提醒" if alerts else ""
            lines.append(
                f"{'✅' if robot.get('trading_normal') else '🔴'} {strategy} {pair}\n"
                f"  {state}｜BUY {buy}｜SELL {sell}｜阶段 {_state_cn(robot.get('phase'))}{suffix}"
            )
        lines.append("\n点击下方交易对查看全部生效门控。")
        return "\n".join(lines)

    @staticmethod
    def _risk_rows() -> list[list[tuple[str, str]]]:
        return [
            [("Grid BTC", "r:grid:BTC-FDUSD"), ("Grid ETH", "r:grid:ETH-FDUSD")],
            [("DCA BTC", "r:dca:BTC-USDT"), ("DCA ETH", "r:dca:ETH-USDT")],
            [("🏠 主菜单", "m:home")],
        ]

    def _risk_detail(self, strategy: str, pair: str) -> str:
        try:
            snapshot = self.reports.status()
        except Exception as exc:
            return f"🛡 风控详情\n\n数据不可用：{_safe_text(exc)}"
        robot = next((
            item for item in snapshot["robots"]
            if str(item.get("strategy")) == strategy and str(item.get("pair")) == pair
        ), None)
        if not robot:
            return f"🛡 风控详情\n\n没有找到 {strategy.upper()} {pair} 的状态。"
        permissions = robot.get("final_permissions", {})
        buy = "放行" if permissions.get("buy_enabled") else "阻止"
        sell = "放行" if permissions.get("sell_enabled") else "阻止"
        lines = [
            f"🛡 {strategy.upper()} {pair} 风控详情",
            f"交易状态：{'✅ 正常交易' if robot.get('trading_normal') else '🔴 交易受限'}",
            f"最终权限：BUY {buy}｜SELL {sell}",
            f"恢复阶段：{_state_cn(robot.get('phase'))}",
            "",
            "全部生效门控：",
        ]
        for gate in robot.get("gate_statuses", []):
            if not isinstance(gate, dict) or not gate.get("applicable", True):
                continue
            gate_buy = _permission_cn(gate.get("buy_enabled"))
            gate_sell = _permission_cn(gate.get("sell_enabled"))
            lines.append(
                f"• {gate.get('label', gate.get('mechanism', '未知门控'))}："
                f"{_state_cn(gate.get('state'))}\n"
                f"  BUY {gate_buy}｜SELL {gate_sell}｜{_reason_cn(gate.get('reason'))}"
            )
        blockers = robot.get("blockers", [])
        if blockers:
            lines.extend(("", "当前阻塞原因："))
            lines.extend(f"• {_safe_text(item, 160)}" for item in blockers)
        else:
            lines.append("\n当前没有阻塞交易的风控门。")
        return "\n".join(lines)

    def _errors(self) -> str:
        blockers: list[str] = []
        warnings: list[str] = []
        normal = 0
        total = 0
        budget_warning_bots: set[str] = set()
        try:
            status = self.reports.status()
            for robot in status["robots"]:
                total += 1
                pair = str(robot.get("pair", "未知交易对"))
                strategy = str(robot.get("strategy", "")).upper()
                bot_name = str(robot.get("bot", pair))
                permissions = robot.get("final_permissions", {})
                buy = "放行" if permissions.get("buy_enabled") else "阻止"
                sell = "放行" if permissions.get("sell_enabled") else "阻止"
                if robot.get("trading_normal"):
                    normal += 1
                else:
                    reasons = robot.get("blockers", [])
                    reason = "；".join(_safe_text(item, 100) for item in reasons) or "存在生效中的风控门"
                    blockers.append(
                        f"{strategy} {pair}：交易受限，BUY={buy}、SELL={sell}；{reason}"
                    )
                for gate in robot.get("gate_statuses", []):
                    if not isinstance(gate, dict) or not gate.get("applicable", True):
                        continue
                    if gate.get("state") == "ALERT_ONLY":
                        budget_warning_bots.add(bot_name)
                        quote = pair.split("-")[-1]
                        warnings.append(
                            f"{strategy} {pair}：可用 {quote} 低于预算提醒值；"
                            "仅告警，BUY/SELL仍放行。"
                        )
                    elif str(gate.get("health", "HEALTHY")).upper() != "HEALTHY" and robot.get("trading_normal"):
                        warnings.append(
                            f"{strategy} {pair}：{gate.get('label', '风控数据')}状态异常，"
                            "当前尚未阻塞交易。"
                        )
        except Exception as exc:
            blockers.append(_safe_text(exc))

        # Hummingbot's error_logs is a rolling history, not a current alarm list.
        # Only recent unique errors are shown and repeated per-second messages are collapsed.
        cutoff = time.time() - 900
        recent: dict[tuple[str, str], int] = {}
        try:
            for name, raw in self._api_bots().items():
                values = raw.get("error_logs", []) if isinstance(raw, dict) else []
                if isinstance(values, list):
                    for value in values:
                        if not isinstance(value, dict) or float(value.get("timestamp", 0) or 0) < cutoff:
                            continue
                        msg = str(value.get("msg", "未知运行错误"))
                        if name in budget_warning_bots and "Not enough budget" in msg:
                            continue
                        key = (name, msg)
                        recent[key] = recent.get(key, 0) + 1
        except Exception as exc:
            blockers.append(f"Hummingbot API不可用：{_safe_text(exc)}")
        for (name, message), count in recent.items():
            suffix = f"（近15分钟重复{count}次）" if count > 1 else ""
            warnings.append(f"{name}：{_safe_text(message, 140)}{suffix}；请关注，当前权限以风控状态为准。")

        snapshot = self.contracts.snapshot()
        for error in snapshot["errors"]:
            blockers.append(f"风控合同不可用：{error}")

        lines = ["⚠️ 当前异常"]
        if blockers:
            lines.append(f"结论：🔴 {len(blockers)}项正在影响或可能影响交易；正常交易 {normal}/{total}。")
            lines.extend(("", "交易阻塞："))
            lines.extend(f"• {item}" for item in blockers)
        else:
            lines.append(f"结论：✅ {normal}/{total} 正常交易；没有生效中的交易阻塞。")
        if warnings:
            lines.extend(("", "提醒（不阻塞交易）："))
            lines.extend(f"• {item}" for item in warnings)
        elif not blockers:
            lines.append("当前也没有需要处理的运行提醒。")
        return "\n".join(lines)

    def _models(self) -> str:
        candidates = self.approvals.pending()
        pending = [item for item in candidates if item["status"] == "PENDING"]
        lines = ["⚙️ 模型与参数", f"待审批模型：{len(pending)}"]
        if candidates:
            current = candidates[0]
            lines.extend((
                f"最近模型：{current['model_type']}",
                f"Release：{current['release_sha256'][:16]}",
                f"状态：{current['status']}",
            ))
        else:
            lines.append("当前没有可读取的模型候选记录。")
        lines.append("普通 Grid 参数更新不进入模型审批中心。")
        return "\n".join(lines)

    def _bot_menu(self, key: str) -> tuple[str, list[list[tuple[str, str]]]]:
        definition = self.settings.bots[key]
        name = definition["bot_name"]
        try:
            raw = self._api_bots().get(name, {})
            status = self._bot_line(name, raw)
        except Exception as exc:
            status = f"{name}：数据不可用（{type(exc).__name__}）"
        rows = [
            [("⏸ 停止并撤单", f"b:{key}:stop"), ("▶️ 恢复", f"b:{key}:start")],
            [("🔄 安全重启", f"b:{key}:restart"), ("刷新", f"b:{key}:view")],
            [("🏠 主菜单", "m:home")],
        ]
        return f"🤖 机器人管理\n\n{status}\n\n变更前会再次预检并要求确认。", rows

    def _dca_menu(self) -> tuple[str, list[list[tuple[str, str]]]]:
        rows = [
            [("BTC-USDT", "m:bot:dca_btc"), ("ETH-USDT", "m:bot:dca_eth")],
            [("🏠 主菜单", "m:home")],
        ]
        return "🟩 DCA\n\n请选择机器人：", rows

    def _stock_menu(self) -> tuple[str, list[list[tuple[str, str]]]]:
        try:
            health = self.stocks.health()
            mode = str(health.get("runtime_mode", "-")).upper()
            if mode == "PAPER":
                status = self._paper_status(self.stocks.paper_summary()).lstrip("✅🔴🟦🟠 ")
            else:
                status = f"Runtime {health.get('status', 'unknown')} / {mode}"
        except Exception as exc:
            status = f"不可用（{type(exc).__name__}）"
        rows = [
            [("🧾 单笔订单", "s:new:order"), ("📈 仓位交易", "s:new:position")],
            [("💰 Paper 收益", "s:paper_profit"), ("📊 Paper 持仓", "s:paper_positions")],
            [("🧾 Paper 成交", "s:paper_trades"), ("🔄 刷新", "m:stock")],
            [("📋 白名单", "s:whitelist"), ("📏 交易限制", "s:limits")],
            [("Executor管理", "s:positions")],
            [("🏠 主菜单", "m:home")],
        ]
        paper_switch = "已开启" if self.settings.stocks_paper_trading_enabled else "已关闭"
        return (
            f"📈 Stock 管理\n\n状态：{status}\n"
            f"Telegram Paper下单：{paper_switch}\n"
            "交易时段：盘前 + 正常时段 + 盘后（EXTENDED）",
            rows,
        )

    @staticmethod
    def _paper_status(summary: dict) -> str:
        if summary.get("account", {}).get("recovery_required"):
            return "🔴 恢复校验失败，禁止新增Paper仓位"
        if not summary.get("valuation_complete") or not summary.get("reconciliation", {}).get("ok"):
            return "🔴 收益数据无法对账"
        health = str(summary.get("quote_health", "UNKNOWN")).upper()
        if health == "FRESH":
            return "✅ 模拟交易正常"
        if health == "MARKET_CLOSED_LAST_TRUSTED":
            return "🟦 市场休市，使用最后可信行情估值"
        if health == "MARKET_STATE_UNAVAILABLE":
            return "🟠 行情已接入，等待交易时段状态"
        if health in {"MARKET_DATA_UNAVAILABLE", "AWAITING_FIRST_QUOTE"}:
            return "🟠 行情尚未接入，Paper未运行"
        return f"🟠 Paper状态：{health}"

    @staticmethod
    def _paper_money(value: Any, *, signed: bool = True) -> str:
        try:
            number = Decimal(str(value))
        except (InvalidOperation, TypeError, ValueError):
            return "无可信数据"
        return f"{number:+.4f} USDC" if signed else f"{number:.4f} USDC"

    def _stock_paper_profit(self) -> str:
        summary = self.stocks.paper_summary()
        lines = ["💰 Stock PAPER 收益", self._paper_status(summary), ""]
        if not summary.get("valuation_complete") or not summary.get("reconciliation", {}).get("ok"):
            error = summary.get("reconciliation", {}).get("error", "unknown")
            lines.append(f"原因：{_safe_text(error, 180)}\n收益数值已隐藏，避免展示错误估值。")
            return "\n".join(lines)
        windows = summary.get("windows", {})
        for key, label in (("4h", "最近4小时"), ("24h", "最近24小时"), ("7d", "最近7天"), ("all", "上线以来")):
            window = windows.get(key, {})
            incomplete = "（运行期不足，按本次Paper run）" if not window.get("window_complete") else ""
            lines.append(f"• {label}：{self._paper_money(window.get('pnl'))}{incomplete}")
        account = summary.get("account", {})
        totals = summary.get("totals", {})
        lines.extend((
            "",
            f"当前权益：{self._paper_money(account.get('equity'), signed=False)}",
            f"峰值权益：{self._paper_money(account.get('peak_equity'), signed=False)}",
            f"当前回撤：{Decimal(str(account.get('drawdown_pct', 0))):.4f}%",
            f"可用现金：{self._paper_money(account.get('available_cash'), signed=False)}",
            f"持仓市值：{self._paper_money(account.get('positions_value'), signed=False)}",
            f"累计费用：{self._paper_money(totals.get('fees'), signed=False)}",
            f"成交：{totals.get('fill_count', 0)}笔｜活动订单：{totals.get('open_order_count', 0)}｜"
            f"活动Executor：{totals.get('active_executor_count', 0)}",
            f"Paper run：{str(summary.get('paper_run_id', '-'))[:16]}",
        ))
        return "\n".join(lines)

    def _stock_paper_positions(self) -> str:
        summary = self.stocks.paper_summary()
        lines = ["📊 Stock PAPER 持仓", self._paper_status(summary), ""]
        positions = [row for row in summary.get("positions", []) if Decimal(str(row.get("total", 0))) > 0]
        if not positions:
            lines.append("当前没有Paper持仓。")
            return "\n".join(lines)
        for row in positions:
            lines.append(
                f"• {row.get('symbol', '-')}：{row.get('total', '0')}股\n"
                f"  成本 {self._paper_money(row.get('cost_quote'), signed=False)}｜"
                f"均价 {self._paper_money(row.get('average_cost'), signed=False)}\n"
                f"  市值 {self._paper_money(row.get('market_value'), signed=False)}｜"
                f"最新Bid {row.get('mark_bid') or '无可信数据'}\n"
                f"  已实现 {self._paper_money(row.get('realized_pnl'))}｜"
                f"未实现 {self._paper_money(row.get('unrealized_pnl'))}\n"
                f"  费用 {self._paper_money(row.get('fees'), signed=False)}｜"
                f"净收益 {self._paper_money(row.get('net_pnl'))}"
            )
        return "\n".join(lines)

    def _stock_paper_trades(self) -> str:
        trades = self.stocks.paper_trades(limit=12)
        lines = ["🧾 Stock PAPER 最近成交", ""]
        if not trades:
            lines.append("当前没有Paper成交。")
            return "\n".join(lines)
        for row in trades:
            lines.append(
                f"• {row.get('symbol', '-')} {row.get('side', '-')} {row.get('quantity', '0')}股 "
                f"@ {row.get('price', '-')}｜费用 {row.get('fee_delta', '0')} USDC\n"
                f"  {str(row.get('created_at', '-'))[:19]}"
            )
        return "\n".join(lines)

    def _enabled_symbols(self) -> list[str]:
        return [str(row["symbol"]) for row in self.stocks.whitelist() if row.get("enabled")]

    def _new_stock_session(self, flow: str, user_id: int, chat_id: int, message_id: int) -> tuple[str, list]:
        session = self.store.create_session(user_id, chat_id, flow, message_id)
        symbols = self._enabled_symbols()
        rows = [[(symbol, f"w:{session['session_id']}:symbol:{symbol}")] for symbol in symbols[:12]]
        rows.append([("取消", f"w:{session['session_id']}:cancel:-")])
        return (f"📈 {'单笔订单' if flow == 'stock_order' else '仓位交易'}\n\n请选择白名单股票：", rows)

    def _stock_order_step(self, session: dict, action: str, value: str) -> tuple[str, list]:
        sid = session["session_id"]
        if action == "symbol":
            session = self.store.update_session(sid, step="side", payload={"symbol": value})
            return "请选择方向：", [[("BUY", f"w:{sid}:side:BUY"), ("SELL", f"w:{sid}:side:SELL")], [("取消", f"w:{sid}:cancel:-")]]
        if action == "side":
            self.store.update_session(sid, step="order_type", payload={"side": value})
            return "请选择订单类型：", [[("LIMIT（推荐）", f"w:{sid}:otype:LIMIT"), ("MARKET", f"w:{sid}:otype:MARKET")], [("取消", f"w:{sid}:cancel:-")]]
        if action == "otype":
            self.store.update_session(sid, step="amount", payload={"order_type": value})
            return "请选择下单金额或输入自定义值：", [
                [("100 USDC（推荐）", f"w:{sid}:amount:100"), ("50 USDC", f"w:{sid}:amount:50")],
                [("自定义金额", f"w:{sid}:input:amount"), ("自定义股数", f"w:{sid}:input:shares")],
                [("取消", f"w:{sid}:cancel:-")],
            ]
        if action == "amount":
            session = self.store.update_session(sid, payload={"amount_usdc": value, "amount_basis": "quote"})
            if session["payload"]["order_type"] == "LIMIT":
                return self._limit_price_step(session)
            return self._stock_preview(session)
        if action == "price_latest":
            session = self._capture_latest_limit_price(session)
            return self._stock_preview(session)
        if action == "price_refresh":
            return self._limit_price_step(session)
        if action == "confirm":
            return self._execute_stock(session)
        if action == "refresh":
            return self._executor_result(session)
        raise ValueError("未知订单向导动作")

    def _stock_position_step(self, session: dict, action: str, value: str) -> tuple[str, list]:
        sid = session["session_id"]
        if action == "symbol":
            self.store.update_session(sid, step="amount", payload={"symbol": value, "side": "BUY"})
            return "请选择目标仓位金额：", [
                [("100 USDC（推荐）", f"w:{sid}:amount:100"), ("50 USDC", f"w:{sid}:amount:50")],
                [("自定义金额", f"w:{sid}:input:amount")], [("取消", f"w:{sid}:cancel:-")],
            ]
        if action == "amount":
            self.store.update_session(sid, step="entry", payload={"amount_usdc": value, "amount_basis": "quote"})
            return "请选择入场方式：", [[("LIMIT（推荐）", f"w:{sid}:entry:LIMIT"), ("MARKET", f"w:{sid}:entry:MARKET")], [("取消", f"w:{sid}:cancel:-")]]
        if action == "entry":
            session = self.store.update_session(sid, step="barriers", payload={
                "order_type": value, "take_profit": "0.03", "stop_loss": "0.02", "days": "7",
            })
            if value == "LIMIT":
                return self._limit_price_step(session)
            return self._position_barriers_step(session)
        if action == "barriers":
            return self._stock_preview(self.store.get_session(sid) or session)
        if action == "price_latest":
            session = self._capture_latest_limit_price(session)
            return self._position_barriers_step(session)
        if action == "price_refresh":
            return self._limit_price_step(session)
        if action == "confirm":
            return self._execute_stock(session)
        if action == "refresh":
            return self._executor_result(session)
        raise ValueError("未知仓位向导动作")

    def _limit_price_step(self, session: dict) -> tuple[str, list]:
        sid = session["session_id"]
        self.store.update_session(sid, step="price")
        side = str(session["payload"].get("side", "BUY"))
        context, quote = self._stock_quote_context(session["payload"]["symbol"], side)
        reference = Decimal(str(quote["reference"]))
        reference_label = "卖一" if side.upper() == "BUY" else "买一"
        return (
            f"限价设置\n\n{context}\n\n"
            f"建议：{side.upper()} LIMIT 可从最新{reference_label}价开始判断；偏离盘口会影响成交速度。",
            [[(f"按最新{reference_label} {reference}", f"w:{sid}:price_latest:-"),
              ("自定义限价", f"w:{sid}:input:price")],
             [("刷新行情", f"w:{sid}:price_refresh:-"), ("取消", f"w:{sid}:cancel:-")]],
        )

    def _stock_quote_context(self, symbol: str, side: str) -> tuple[str, dict]:
        raw = self.stocks.quote(symbol)
        bid = Decimal(str(raw.get("bidPrice", raw.get("bid", "0")) or "0"))
        ask = Decimal(str(raw.get("askPrice", raw.get("ask", "0")) or "0"))
        if bid <= 0 or ask <= 0 or ask < bid:
            raise ValueError("最新Bid/Ask无效，暂时不能辅助定价")
        mid = (bid + ask) / 2
        spread = ask - bid
        spread_pct = spread / mid * 100 if mid > 0 else Decimal("0")
        quote_ts = _quote_timestamp(raw)
        if quote_ts is None:
            quote_time = datetime.now(BEIJING_TZ).strftime("查询于 %Y-%m-%d %H:%M:%S %z")
            age_text = "交易所未提供事件时间，无法计算"
        else:
            quote_time = datetime.fromtimestamp(quote_ts, tz=timezone.utc).astimezone(BEIJING_TZ).strftime(
                "%Y-%m-%d %H:%M:%S %z"
            )
            age_text = f"{max(0, time.time() - quote_ts):.1f}秒"
        try:
            market = self.stocks.market_status(symbol)
        except Exception:
            market = {}
        phase = _state_cn(market.get("market_phase", "未知"))
        trading = _state_cn(market.get("trading_status", "未知"))
        reference = ask if side.upper() == "BUY" else bid
        text = (
            f"📍 {symbol} 最新行情\n"
            f"买一 Bid：{bid} USDC｜卖一 Ask：{ask} USDC\n"
            f"中间价：{mid.quantize(Decimal('0.0001'))} USDC｜"
            f"价差：{spread} USDC（{spread_pct.quantize(Decimal('0.0001'))}%）\n"
            f"{side.upper()}参考价：{reference} USDC\n"
            f"行情时间：{quote_time}｜数据年龄：{age_text}\n"
            f"市场阶段：{phase}｜标的状态：{trading}"
        )
        return text, {
            "bid": str(bid), "ask": str(ask), "mid": str(mid),
            "reference": str(reference), "quote_ts": quote_ts,
        }

    def _capture_latest_limit_price(self, session: dict) -> dict:
        side = str(session["payload"].get("side", "BUY"))
        _, quote = self._stock_quote_context(session["payload"]["symbol"], side)
        return self.store.update_session(session["session_id"], payload={
            "price": quote["reference"],
            "price_source": "latest_ask" if side.upper() == "BUY" else "latest_bid",
            "quote_bid": quote["bid"], "quote_ask": quote["ask"],
            "quote_ts": quote["quote_ts"],
        })

    def _position_barriers_step(self, session: dict) -> tuple[str, list]:
        sid = session["session_id"]
        session = self.store.update_session(sid, step="barriers")
        data = session["payload"]
        context, quote = self._stock_quote_context(data["symbol"], "BUY")
        entry = Decimal(str(data.get("price", quote["ask"])))
        amount = Decimal(str(data.get("amount_usdc", "100")))
        tp = Decimal(str(data.get("take_profit", "0.03")))
        sl = Decimal(str(data.get("stop_loss", "0.02")))
        days = Decimal(str(data.get("days", "7")))
        tp_price = entry * (1 + tp)
        sl_price = entry * (1 - sl)
        reward = amount * tp
        risk = amount * sl
        ratio = reward / risk if risk > 0 else Decimal("0")
        entry_note = (
            "LIMIT入场价" if data.get("order_type") == "LIMIT"
            else "MARKET估算入场价（最终成交可能滑移）"
        )
        text = (
            f"仓位退出参数\n\n{context}\n\n"
            f"{entry_note}：{entry} USDC\n"
            f"止盈3% → 约 {tp_price.quantize(Decimal('0.0001'))} USDC，"
            f"预计 +{reward.quantize(Decimal('0.01'))} USDC\n"
            f"止损2% → 约 {sl_price.quantize(Decimal('0.0001'))} USDC，"
            f"预计 -{risk.quantize(Decimal('0.01'))} USDC\n"
            f"预期盈亏比：{ratio.quantize(Decimal('0.01'))}:1｜最长持仓：{days}天\n\n"
            "以上为价格辅助，不含跳空、滑点和费用；最终参数由Stock Runtime再次校验。"
        )
        return text, [
            [("采用以上 3% / 2% / 7天", f"w:{sid}:barriers:default"),
             ("自定义参数", f"w:{sid}:input:barriers")],
            [("取消", f"w:{sid}:cancel:-")],
        ]

    def _build_stock_request(self, session: dict) -> tuple[dict, Decimal]:
        data = session["payload"]
        symbol = data["symbol"]
        side = data.get("side", "BUY")
        quote = self.stocks.quote(symbol)
        market_price = _quote_price(quote, side)
        price = _d(data.get("price", market_price))
        if data.get("amount_basis") == "shares":
            shares = _d(data["shares"])
        else:
            shares = (_d(data.get("amount_usdc", "100")) / price).quantize(Decimal("0.000001"), rounding=ROUND_DOWN)
        if shares <= 0:
            raise ValueError("金额低于可表示的最小股数")
        raw = f"{session['session_id']}|{symbol}|{side}|{shares}|{price}|{session['flow']}"
        executor_id = "tg-" + hashlib.sha256(raw.encode()).hexdigest()[:24]
        order_type = data.get("order_type", "LIMIT")
        common = {
            "id": executor_id,
            "symbol": symbol,
            "amount": str(shares),
            "controller_id": "telegram-management-bot",
        }
        if session["flow"] == "stock_order":
            request = {**common, "side": side, "order_type": order_type}
            if order_type == "LIMIT":
                request["price"] = str(price)
        else:
            request = {
                **common,
                "entry_order_type": order_type,
                "stop_loss": data.get("stop_loss", "0.02"),
                "take_profit": data.get("take_profit", "0.03"),
                "time_limit": int(Decimal(data.get("days", "7")) * 86400),
            }
            if order_type == "LIMIT":
                request["entry_price"] = str(price)
        return request, price

    def _stock_preview(self, session: dict) -> tuple[str, list]:
        request, price = self._build_stock_request(session)
        if session["flow"] == "stock_order":
            preview = self.stocks.preview_order(request)
            request_type = "order"
        else:
            preview = self.stocks.preview_position(request)
            request_type = "position"
        session = self.store.update_session(
            session["session_id"], step="confirm",
            payload={"request": request, "request_type": request_type},
        )
        notional = Decimal(request["amount"]) * price
        loss = notional * Decimal(str(request.get("stop_loss", "0")))
        side = str(session["payload"].get("side", "BUY"))
        quote_context, latest_quote = self._stock_quote_context(session["payload"]["symbol"], side)
        lines = [
            "✅ Stock 权威预检通过",
            f"股票：{session['payload']['symbol']} / {request.get('side', 'BUY')}",
            f"类型：{'OrderExecutor' if request_type == 'order' else 'PositionExecutor'} / "
            f"{session['payload'].get('order_type', 'LIMIT')}",
            quote_context,
            f"数量：{request['amount']} 股",
            f"参考价格：{price} USDC",
            f"预计金额：{notional.quantize(Decimal('0.01'))} USDC",
            f"预计费用预留：{preview.get('fee_reserve', '-')} USDC",
            "执行范围：PAPER（本地持久化撮合）",
            "不会发送Binance真实下单或撤单请求。",
        ]
        if session["payload"].get("order_type") == "LIMIT":
            latest_reference = Decimal(str(latest_quote["reference"]))
            deviation = (price / latest_reference - 1) * 100 if latest_reference > 0 else Decimal("0")
            lines.append(
                f"限价相对最新{'卖一' if side.upper() == 'BUY' else '买一'}："
                f"{deviation.quantize(Decimal('0.0001'))}%"
            )
        if loss > 0:
            lines.append(f"止损最大估算：{loss.quantize(Decimal('0.01'))} USDC（不含跳空滑点）")
        if request_type == "position":
            take_profit = Decimal(str(request.get("take_profit", "0")))
            stop_loss = Decimal(str(request.get("stop_loss", "0")))
            lines.extend([
                f"止盈触发参考：{(price * (1 + take_profit)).quantize(Decimal('0.0001'))} USDC "
                f"（+{take_profit * 100}%）",
                f"止损触发参考：{(price * (1 - stop_loss)).quantize(Decimal('0.0001'))} USDC "
                f"（-{stop_loss * 100}%）",
                f"最长持仓：{Decimal(request['time_limit']) / Decimal(86400)}天",
            ])
        lines.append("确认后由 Stock Runtime 再次执行同一套校验。")
        return "\n".join(lines), [[("✅ 确认创建", f"w:{session['session_id']}:confirm:-"), ("取消", f"w:{session['session_id']}:cancel:-")]]

    def _execute_stock(self, session: dict) -> tuple[str, list]:
        request = dict(session["payload"].get("request", {}))
        request_type = str(session["payload"].get("request_type", ""))
        if not request or request_type not in {"order", "position"}:
            raise ValueError("订单预览已失效，请重新创建")
        if not self.settings.stocks_paper_trading_enabled:
            return "🔒 Telegram Paper下单开关已关闭，未创建任何订单。", self._back("m:stock")
        health = self.stocks.health()
        if str(health.get("runtime_mode", "")).upper() != "PAPER":
            return "⛔ Stock Runtime并非PAPER模式，管理Bot拒绝提交。", self._back("m:stock")
        if health.get("economic_requests_enabled") or health.get("live_authorized"):
            return "⛔ 检测到实盘经济请求权限，管理Bot拒绝Paper提交。", self._back("m:stock")
        key = f"stock-paper:{request['id']}"
        claimed, existing = self.store.claim_action(key)
        if claimed:
            try:
                result = (
                    self.stocks.create_order(request)
                    if request_type == "order" else self.stocks.create_position(request)
                )
                self.store.finish_action(key, "SUBMITTED", result)
            except Exception as exc:
                result = {"error": _safe_text(exc)}
                self.store.finish_action(key, "FAILED", result)
                raise
        else:
            result = existing or {}
        self.store.update_session(session["session_id"], step="result", payload={"executor_id": request["id"]})
        return (
            f"📨 PAPER创建请求已处理\nExecutor：{request['id']}\n"
            f"范围：PAPER / Binance真实经济请求=False\n"
            f"状态：{_safe_text(result.get('status', result))}",
            [[("刷新执行结果", f"w:{session['session_id']}:refresh:-"), ("Stock菜单", "m:stock")]],
        )

    def _executor_result(self, session: dict) -> tuple[str, list]:
        executor_id = str(session["payload"].get("executor_id", ""))
        if not executor_id:
            raise ValueError("没有可查询的Executor")
        result = self.stocks.executor(executor_id)
        ledger = result.get("ledger") or {}
        runtime = result.get("runtime") or {}
        orders = ledger.get("orders", []) if isinstance(ledger, dict) else []
        filled = sum(Decimal(str(item.get("cumulative_base", 0))) for item in orders if isinstance(item, dict))
        fees = sum(Decimal(str(item.get("cumulative_fee", 0))) for item in orders if isinstance(item, dict))
        text = (
            f"📈 Executor 执行结果\nID：{executor_id}\n"
            f"账本状态：{ledger.get('status', '-')}\n运行状态：{runtime.get('status', '-')}\n"
            f"累计成交：{filled} 股\n累计费用：{fees} USDC"
        )
        return text, [[("刷新", f"w:{session['session_id']}:refresh:-"), ("Stock菜单", "m:stock")]]

    def _stock_whitelist(self) -> tuple[str, list]:
        try:
            items = self.stocks.whitelist()
        except Exception as exc:
            return f"📋 白名单\n\n数据不可用：{_safe_text(exc)}", self._back("m:stock")
        lines = ["📋 Stock 白名单", ""]
        rows: list[list[tuple[str, str]]] = []
        for item in items[:20]:
            symbol = str(item.get("symbol"))
            enabled = bool(item.get("enabled"))
            lines.append(f"• {symbol}：{'启用' if enabled else '停用'}；上限 {item.get('max_position_notional')} USDC")
            rows.append([
                (f"{'停用' if enabled else '启用'} {symbol}", f"s:wl_toggle:{symbol}:{int(not enabled)}"),
                (f"删除 {symbol}", f"s:wl_delete:{symbol}"),
            ])
        rows.append([("➕ 添加/修改", "s:wl_input"), ("⬅️ 返回", "m:stock")])
        return "\n".join(lines), rows

    def _stock_limits(self) -> str:
        value = self.stocks.limits()
        active, hard = value.get("active", {}), value.get("hard_ceiling", {})
        return (
            "📏 Stock 交易限制\n\n"
            f"单笔：{active.get('max_order_notional')} / 硬上限 {hard.get('max_order_notional')} USDC\n"
            f"单股票：{active.get('max_symbol_exposure')} / 硬上限 {hard.get('max_symbol_exposure')} USDC\n"
            f"总持仓：{active.get('max_managed_exposure')} / 硬上限 {hard.get('max_managed_exposure')} USDC"
        )

    def _stock_executors_menu(self, user_id: int, chat_id: int, message_id: int) -> tuple[str, list]:
        items = self.stocks.executors(active_only=True)
        lines = ["📈 Stock 活动仓位与订单", ""]
        rows: list[list[tuple[str, str]]] = []
        for item in items[:12]:
            executor_id = str(item.get("executor_id", ""))
            lines.append(
                f"• {item.get('symbol', '-')} / {item.get('executor_type', '-')} / {item.get('status', '-')}"
            )
            session = self.store.create_session(user_id, chat_id, "executor_action", message_id)
            self.store.update_session(session["session_id"], payload={"executor_id": executor_id})
            rows.append([("管理 " + str(item.get("symbol", "-")), f"x:{session['session_id']}:exec_view")])
        if not items:
            lines.append("当前没有活动 Executor。")
        rows.append([("刷新", "s:positions"), ("⬅️ 返回", "m:stock")])
        return "\n".join(lines), rows

    def _stock_executor_detail(self, session: dict) -> tuple[str, list]:
        executor_id = session["payload"]["executor_id"]
        result = self.stocks.executor(executor_id)
        ledger = result.get("ledger") or {}
        rows = [[
            ("⏸ 暂停", f"x:{session['session_id']}:exec_pause"),
            ("📉 减仓", f"x:{session['session_id']}:exec_reduce"),
            ("🛑 平仓", f"x:{session['session_id']}:exec_close"),
        ], [("⬅️ 返回", "s:positions")]]
        return (
            f"📈 Executor\nID：{executor_id}\n股票：{ledger.get('symbol', '-')}\n"
            f"类型：{ledger.get('executor_type', '-')}\n状态：{ledger.get('status', '-')}",
            rows,
        )

    def _approvals_menu(self) -> tuple[str, list]:
        pending = [item for item in self.approvals.pending() if item["status"] == "PENDING"]
        lines = ["🧠 模型审批", f"待审批：{len(pending)}", ""]
        rows: list[list[tuple[str, str]]] = []
        for item in pending[:10]:
            deadline = item.get("review_deadline")
            remain = max(0, int(deadline) - int(time.time())) if deadline else 0
            lines.append(f"• {item['model_type']} / {item['candidate_id']} / 剩余{remain // 3600}小时")
            rows.append([(f"查看 {item['candidate_id']}", f"a:{item['candidate_id']}:view")])
        if not pending:
            lines.append("当前没有待审批模型。")
        rows.append([("刷新", "m:approvals"), ("🏠 主菜单", "m:home")])
        return "\n".join(lines), rows

    def _notify_new_model_candidates(self) -> None:
        """Send one private inline approval card per hash-bound pending candidate."""
        for item in self.approvals.pending():
            if item.get("status") != "PENDING":
                continue
            release = item["release_sha256"]
            marker = f"approval_notified:{release}"
            if self.store.metadata(marker) == "true":
                continue
            deadline = item.get("review_deadline")
            deadline_text = (
                time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime(int(deadline)))
                if deadline else "-"
            )
            self.telegram.send(
                self.settings.admin_user_id,
                "🧠 新模型候选等待审批\n\n"
                f"类型：{item['model_type']}\n候选：{item['candidate_id']}\n"
                f"默认审批截止：{deadline_text}\n"
                "截止前无人拒绝且硬门槛持续通过时自动批准；当前模型继续交易。",
                [[("查看证据并审批", f"a:{item['candidate_id']}:view")]],
            )
            self.store.set_metadata(marker, "true")

    def _approval_detail(self, candidate_id: str) -> tuple[str, list]:
        item = self.approvals.find(candidate_id)
        if not item:
            return "模型候选不存在或已归档。", self._back("m:approvals")
        evidence = self.approvals.evidence(item)
        checks = item.get("checks", {})
        check_lines = [f"• {key}：{'通过' if value else '失败'}" for key, value in checks.items()]
        lines = [
            f"🧠 {item['model_type']}",
            f"候选：{item['candidate_id']}",
            f"Release：{item['release_sha256']}",
            f"模型：{item['model_sha256']}",
            f"状态：{item['status']}",
            f"证据附件：{len(evidence['attachments'])}",
            "硬门槛：",
            *(check_lines or ["• 无门槛数据"]),
            "默认行为：截止前无人拒绝且硬门槛持续通过则自动批准。",
        ]
        rows = [
            [("查看PNG/PDF证据", f"a:{candidate_id}:evidence")],
            [("✅ 批准", f"a:{candidate_id}:approve"), ("❌ 拒绝", f"a:{candidate_id}:reject")],
            [("⬅️ 返回", "m:approvals")],
        ]
        return "\n".join(lines), rows

    def _handle_model_approval(self, data: str, callback: dict, update_id: int) -> tuple[str, list]:
        _, candidate_id, action = data.split(":", 2)
        if action == "view":
            return self._approval_detail(candidate_id)
        candidate = self.approvals.find(candidate_id)
        if not candidate or candidate.get("status") != "PENDING":
            return "候选已不再等待审批。", self._back("m:approvals")
        if action == "evidence":
            evidence = self.approvals.evidence(candidate)
            attachments = [
                item for item in evidence["attachments"]
                if str(item["name"]).lower().endswith((".png", ".jpg", ".jpeg", ".pdf"))
            ]
            for item in attachments[:8]:
                self.telegram.send_file(
                    int(callback["message"]["chat"]["id"]),
                    item["path"],
                    f"{candidate_id} · {item['name']} · sha256 {item['sha256'][:16]}",
                )
            return (
                f"已发送 {min(len(attachments), 8)} 个PNG/PDF证据附件。"
                if attachments else "该候选当前没有可发送的PNG/PDF证据附件。",
                self._back(f"a:{candidate_id}:view"),
            )
        if action == "approve":
            return (
                f"确认批准模型候选 {candidate_id}？\n决定将绑定请求和证据哈希。",
                [[("确认批准", f"a:{candidate_id}:approve2"), ("取消", f"a:{candidate_id}:view")]],
            )
        if action == "reject":
            session = self.store.create_session(
                int(callback["from"]["id"]), int(callback["message"]["chat"]["id"]),
                "model_reject", int(callback["message"]["message_id"]),
            )
            self.store.update_session(session["session_id"], step="reason", payload={"candidate_id": candidate_id})
            return "请输入拒绝原因：", [[("取消", "m:approvals")]]
        if action == "approve2":
            if not self.settings.mutations_enabled:
                return "🔒 当前为只读观察模式，未写入审批决定。", self._back("m:approvals")
            key = f"approval:{candidate['release_sha256']}:approve"
            claimed, existing = self.store.claim_action(key)
            if claimed:
                result = self.approvals.decide(
                    candidate, "approve", operator=f"telegram:{self.settings.admin_user_id}", reason="",
                    telegram_user_id=int(callback["from"]["id"]),
                    telegram_chat_id=int(callback["message"]["chat"]["id"]),
                    telegram_update_id=update_id,
                    telegram_callback_query_id=str(callback["id"]),
                )
                self.store.finish_action(key, "APPROVED", result)
            else:
                result = existing or {}
            return f"✅ 模型候选已批准\n{candidate_id}\nScheduler 将再次验证后预热和激活。", self._back("m:approvals")
        raise ValueError("未知审批动作")

    def _handle_text_session(self, message: dict, session: dict) -> tuple[str, list]:
        text = str(message.get("text", "")).strip()
        sid = session["session_id"]
        flow, step = session["flow"], session["step"]
        if flow in {"stock_order", "stock_position"}:
            if step == "input_amount":
                _d(text)
                session = self.store.update_session(sid, payload={"amount_usdc": text, "amount_basis": "quote"})
                if flow == "stock_position" and "order_type" not in session["payload"]:
                    return "请选择入场方式：", [[("LIMIT（推荐）", f"w:{sid}:entry:LIMIT"), ("MARKET", f"w:{sid}:entry:MARKET")]]
                if session["payload"].get("order_type") == "LIMIT":
                    return self._limit_price_step(session)
                return self._stock_preview(session)
            if step == "input_shares":
                _d(text)
                session = self.store.update_session(sid, payload={"shares": text, "amount_basis": "shares"})
                if session["payload"].get("order_type") == "LIMIT":
                    return self._limit_price_step(session)
                return self._stock_preview(session)
            if step == "input_price":
                _d(text)
                session = self.store.update_session(sid, payload={"price": text})
                if flow == "stock_position":
                    return self._position_barriers_step(session)
                return self._stock_preview(session)
            if step == "input_barriers":
                parts = [part.strip() for part in text.replace("，", ",").split(",")]
                if len(parts) != 3:
                    raise ValueError("请输入：止盈%,止损%,持仓天数，例如 3,2,7")
                tp, sl, days = (_d(value) for value in parts)
                if tp > 50 or sl > 20 or days > 365:
                    raise ValueError("止盈≤50%，止损≤20%，持仓≤365天")
                session = self.store.update_session(sid, payload={
                    "take_profit": str(tp / 100), "stop_loss": str(sl / 100), "days": str(days),
                })
                return self._stock_preview(session)
        if flow == "whitelist_input":
            parts = text.upper().split()
            if len(parts) not in {1, 2}:
                raise ValueError("请输入：股票代码 [单股票上限]，例如 AAPL 200")
            limit = str(_d(parts[1] if len(parts) == 2 else "200"))
            session = self.store.update_session(sid, step="confirm", payload={"symbol": parts[0], "limit": limit})
            return f"确认添加/更新 {parts[0]}，单股票上限 {limit} USDC？", [[("确认", f"x:{sid}:wl_confirm"), ("取消", "m:stock")]]
        if flow == "limits_input":
            parts = [part.strip() for part in text.replace("，", ",").split(",")]
            if len(parts) != 3:
                raise ValueError("请输入：单笔,单股票,总持仓，例如 200,200,2000")
            values = [str(_d(part)) for part in parts]
            if not (Decimal(values[0]) <= Decimal(values[1]) <= Decimal(values[2])):
                raise ValueError("必须满足：单笔 ≤ 单股票 ≤ 总持仓")
            self.store.update_session(sid, step="confirm", payload={"limits": values})
            return f"确认更新限额为 {values[0]} / {values[1]} / {values[2]} USDC？", [[("确认", f"x:{sid}:limits_confirm"), ("取消", "m:stock")]]
        if flow == "model_reject" and step == "reason":
            if len(text) < 3:
                raise ValueError("拒绝原因至少3个字符")
            self.store.update_session(sid, step="confirm", payload={"reason": text})
            return f"确认拒绝候选 {session['payload']['candidate_id']}？\n原因：{text}", [[("确认拒绝", f"x:{sid}:reject_confirm"), ("取消", "m:approvals")]]
        if flow == "executor_action" and step == "input_reduce":
            amount = str(_d(text))
            self.store.update_session(sid, step="confirm", payload={"reduce_amount": amount})
            return (
                f"确认减仓 {amount} 股？\nExecutor：{session['payload']['executor_id']}",
                [[("确认减仓", f"x:{sid}:exec_reduce_confirm"), ("取消", f"x:{sid}:exec_view")]],
            )
        raise ValueError("当前没有等待文本输入的向导")

    def _handle_session_action(self, data: str, callback: dict, update_id: int) -> tuple[str, list]:
        _, sid, action = data.split(":", 2)
        session = self.store.get_session(sid)
        if not session:
            return "向导已过期，请重新开始。", self._back()
        if action == "wl_toggle_confirm":
            if not self.settings.mutations_enabled:
                return "🔒 只读观察模式，白名单未修改。", self._back("s:whitelist")
            payload = session["payload"]
            result = self.stocks.put_whitelist(
                payload["symbol"], bool(payload["enabled"]), payload["limit"]
            )
            self.store.delete_session(sid)
            return f"✅ {payload['symbol']} 已{'启用' if payload['enabled'] else '停用'}\n{_safe_text(result)}", self._back("s:whitelist")
        if action == "wl_delete_confirm":
            if not self.settings.mutations_enabled:
                return "🔒 只读观察模式，白名单未删除。", self._back("s:whitelist")
            symbol = session["payload"]["symbol"]
            result = self.stocks.delete_whitelist(symbol)
            self.store.delete_session(sid)
            return (
                f"✅ {symbol} 已从白名单移除\n已有仓位保持不变，仍可SELL或平仓。\n{_safe_text(result)}",
                self._back("s:whitelist"),
            )
        if action == "wl_confirm":
            if not self.settings.mutations_enabled:
                return "🔒 只读观察模式，白名单未修改。", self._back("m:stock")
            payload = session["payload"]
            result = self.stocks.put_whitelist(payload["symbol"], True, payload["limit"])
            self.store.delete_session(sid)
            return f"✅ 白名单已更新\n{_safe_text(result)}", self._back("s:whitelist")
        if action == "limits_confirm":
            if not self.settings.mutations_enabled:
                return "🔒 只读观察模式，限额未修改。", self._back("m:stock")
            order, symbol, total = session["payload"]["limits"]
            result = self.stocks.put_limits(order, symbol, total)
            self.store.delete_session(sid)
            return f"✅ 限额已更新\n{_safe_text(result)}", self._back("m:stock")
        if action == "reject_confirm":
            if not self.settings.mutations_enabled:
                return "🔒 只读观察模式，未写入拒绝决定。", self._back("m:approvals")
            candidate = self.approvals.find(session["payload"]["candidate_id"])
            if not candidate:
                raise ValueError("候选已不存在")
            result = self.approvals.decide(
                candidate, "reject", operator=f"telegram:{self.settings.admin_user_id}",
                reason=session["payload"]["reason"],
                telegram_user_id=int(callback["from"]["id"]),
                telegram_chat_id=int(callback["message"]["chat"]["id"]),
                telegram_update_id=update_id,
                telegram_callback_query_id=str(callback["id"]),
            )
            self.store.delete_session(sid)
            return f"❌ 模型候选已拒绝\n{result['release_sha256'][:16]}", self._back("m:approvals")
        if action == "exec_view":
            return self._stock_executor_detail(session)
        if action in {"exec_pause", "exec_close"}:
            label = "暂停并保留仓位" if action == "exec_pause" else "关闭并退出仓位"
            return (
                f"确认{label}？\nExecutor：{session['payload']['executor_id']}",
                [[("确认执行", f"x:{sid}:{action}2"), ("取消", f"x:{sid}:exec_view")]],
            )
        if action in {"exec_pause2", "exec_close2"}:
            if not self.settings.stocks_paper_trading_enabled:
                return "🔒 Telegram Paper下单开关已关闭，未执行Executor变更。", self._back("s:positions")
            health = self.stocks.health()
            if str(health.get("runtime_mode", "")).upper() != "PAPER":
                return "⛔ Stock Runtime并非PAPER模式，拒绝Executor变更。", self._back("s:positions")
            executor_id = session["payload"]["executor_id"]
            key = f"executor:{executor_id}:{action}"
            claimed, existing = self.store.claim_action(key)
            if claimed:
                result = self.stocks.pause(executor_id) if action == "exec_pause2" else self.stocks.close(executor_id)
                self.store.finish_action(key, "COMPLETE", result)
            else:
                result = existing or {}
            return f"✅ Executor 操作完成\n{_safe_text(result)}", self._back("s:positions")
        if action == "exec_reduce":
            self.store.update_session(sid, step="input_reduce")
            return "请输入需要减仓的股数：", [[("取消", f"x:{sid}:exec_view")]]
        if action == "exec_reduce_confirm":
            if not self.settings.stocks_paper_trading_enabled:
                return "🔒 Telegram Paper下单开关已关闭，未执行减仓。", self._back("s:positions")
            health = self.stocks.health()
            if str(health.get("runtime_mode", "")).upper() != "PAPER":
                return "⛔ Stock Runtime并非PAPER模式，拒绝减仓。", self._back("s:positions")
            executor_id = session["payload"]["executor_id"]
            amount = session["payload"]["reduce_amount"]
            request_id = hashlib.sha256(f"{sid}|{executor_id}|{amount}".encode()).hexdigest()[:24]
            key = f"executor:{executor_id}:reduce:{request_id}"
            claimed, existing = self.store.claim_action(key)
            if claimed:
                result = self.stocks.reduce(executor_id, amount, request_id)
                self.store.finish_action(key, "COMPLETE", result)
            else:
                result = existing or {}
            return f"✅ 减仓请求已处理\n{_safe_text(result)}", self._back("s:positions")
        raise ValueError("未知确认动作")

    def _maintenance_action(self, key: str, action: str) -> tuple[str, list]:
        definition = self.settings.bots[key]
        if action == "view":
            return self._bot_menu(key)
        if action in {"stop", "start", "restart"}:
            allowed, reason = ContractReader.resume_allowed(
                "grid" if key == "grid" else "dca", self.contracts.snapshot()
            )
            if action in {"start", "restart"} and not allowed:
                return f"⛔ 当前不能恢复/重启交易\n原因：{reason}", self._back(f"m:bot:{key}")
            label = {"stop": "停止并撤单", "start": "恢复交易", "restart": "安全重启"}[action]
            return (
                f"确认{label} {definition['bot_name']}？\n风控预检：{reason}",
                [[("确认执行", f"c:{key}:{action}"), ("取消", f"m:bot:{key}")]],
            )
        raise ValueError("未知机器人动作")

    def _confirm_maintenance(self, key: str, action: str) -> tuple[str, list]:
        if not self.settings.mutations_enabled:
            return "🔒 当前为只读观察模式，未执行机器人变更。", self._back(f"m:bot:{key}")
        definition = self.settings.bots[key]
        idempotency = f"bot:{definition['bot_name']}:{action}:{int(time.time()) // 30}"
        claimed, existing = self.store.claim_action(idempotency)
        if claimed:
            try:
                if action == "stop":
                    result = self.hummingbot.stop(definition["bot_name"])
                elif action == "start":
                    result = self.hummingbot.start(definition)
                else:
                    result = self.hummingbot.restart(definition)
                self.store.finish_action(idempotency, "COMPLETE", result)
            except Exception as exc:
                self.store.finish_action(idempotency, "FAILED", {"error": _safe_text(exc)})
                raise
        else:
            result = existing or {}
        return f"✅ 操作已执行\n机器人：{definition['bot_name']}\n结果：{_safe_text(result)}", self._back(f"m:bot:{key}")

    def _handle_callback(self, update_id: int, callback: dict) -> None:
        user_id = int(callback.get("from", {}).get("id", 0))
        message = callback.get("message", {})
        chat = message.get("chat", {})
        chat_id = int(chat.get("id", 0))
        message_id = int(message.get("message_id", 0))
        callback_id = str(callback.get("id", ""))
        data = str(callback.get("data", ""))
        if not self._authorized(user_id, chat_id, str(chat.get("type", ""))):
            self.store.audit("ACCESS_DENIED", user_id=user_id, chat_id=chat_id, update_id=update_id,
                             callback_id=callback_id, details={"surface": "callback"})
            self.telegram.answer_callback(callback_id, "无权访问", True)
            return
        try:
            if data.startswith("m:") or data in {
                "s:whitelist", "s:limits", "s:positions", "s:paper_profit",
                "s:paper_positions", "s:paper_trades",
            }:
                self.store.clear_sessions(user_id, chat_id)
            if data == "m:home":
                text, rows = self._home()
            elif data == "m:overview":
                text, rows = self._overview(), self._back()
            elif data == "m:profit":
                text, rows = self._profit(), self._back()
            elif data == "m:risk":
                text, rows = self._risk(), self._risk_rows()
            elif data.startswith("r:"):
                _, strategy, pair = data.split(":", 2)
                text, rows = self._risk_detail(strategy, pair), [
                    [("⬅️ 风控总览", "m:risk"), ("🏠 主菜单", "m:home")]
                ]
            elif data == "m:errors":
                text, rows = self._errors(), self._back()
            elif data == "m:models":
                text, rows = self._models(), self._back()
            elif data == "m:approvals":
                text, rows = self._approvals_menu()
            elif data == "m:stock":
                text, rows = self._stock_menu()
            elif data == "m:dca":
                text, rows = self._dca_menu()
            elif data == "m:maintenance":
                text, rows = "🔧 系统维护\n\n所有变更都需二次确认；恢复不会绕过风控门。", [[("Grid", "m:bot:grid"), ("DCA", "m:dca")], [("🏠 主菜单", "m:home")]]
            elif data.startswith("m:bot:"):
                text, rows = self._bot_menu(data.split(":", 2)[2])
            elif data.startswith("b:"):
                _, key, action = data.split(":", 2)
                text, rows = self._maintenance_action(key, action)
            elif data.startswith("c:"):
                _, key, action = data.split(":", 2)
                text, rows = self._confirm_maintenance(key, action)
            elif data == "s:new:order":
                text, rows = self._new_stock_session("stock_order", user_id, chat_id, message_id)
            elif data == "s:new:position":
                text, rows = self._new_stock_session("stock_position", user_id, chat_id, message_id)
            elif data == "s:whitelist":
                text, rows = self._stock_whitelist()
            elif data == "s:limits":
                text, rows = self._stock_limits(), [[("修改", "s:limits_input"), ("⬅️ 返回", "m:stock")]]
            elif data == "s:positions":
                text, rows = self._stock_executors_menu(user_id, chat_id, message_id)
            elif data == "s:paper_profit":
                text, rows = self._stock_paper_profit(), [[
                    ("🔄 刷新", "s:paper_profit"), ("⬅️ Stock菜单", "m:stock")
                ]]
            elif data == "s:paper_positions":
                text, rows = self._stock_paper_positions(), [[
                    ("🔄 刷新", "s:paper_positions"), ("⬅️ Stock菜单", "m:stock")
                ]]
            elif data == "s:paper_trades":
                text, rows = self._stock_paper_trades(), [[
                    ("🔄 刷新", "s:paper_trades"), ("⬅️ Stock菜单", "m:stock")
                ]]
            elif data == "s:wl_input":
                session = self.store.create_session(user_id, chat_id, "whitelist_input", message_id)
                self.store.update_session(session["session_id"], step="input")
                text, rows = "请输入：股票代码 [单股票上限]\n例如：AAPL 200", [[("取消", "m:stock")]]
            elif data == "s:limits_input":
                session = self.store.create_session(user_id, chat_id, "limits_input", message_id)
                self.store.update_session(session["session_id"], step="input")
                text, rows = "请输入：单笔,单股票,总持仓\n例如：200,200,2000", [[("取消", "m:stock")]]
            elif data.startswith("s:wl_toggle:"):
                _, _, symbol, enabled = data.split(":", 3)
                row = next((item for item in self.stocks.whitelist() if item.get("symbol") == symbol), None)
                if not row:
                    raise ValueError("白名单项目不存在")
                session = self.store.create_session(user_id, chat_id, "whitelist_toggle", message_id)
                self.store.update_session(session["session_id"], step="confirm", payload={
                    "symbol": symbol,
                    "enabled": bool(int(enabled)),
                    "limit": str(row["max_position_notional"]),
                })
                target = "启用" if int(enabled) else "停用（已有仓位不会自动卖出）"
                text, rows = (
                    f"确认{target} {symbol}？",
                    [[("确认", f"x:{session['session_id']}:wl_toggle_confirm"), ("取消", "s:whitelist")]],
                )
            elif data.startswith("s:wl_delete:"):
                symbol = data.split(":", 2)[2]
                row = next((item for item in self.stocks.whitelist() if item.get("symbol") == symbol), None)
                if not row:
                    raise ValueError("白名单项目不存在")
                session = self.store.create_session(user_id, chat_id, "whitelist_delete", message_id)
                self.store.update_session(session["session_id"], step="confirm", payload={"symbol": symbol})
                text, rows = (
                    f"确认删除 {symbol}？\n这会立即禁止新增BUY，但不会自动卖出已有仓位。",
                    [[("确认删除", f"x:{session['session_id']}:wl_delete_confirm"), ("取消", "s:whitelist")]],
                )
            elif data.startswith("w:"):
                _, sid, action, value = data.split(":", 3)
                session = self.store.get_session(sid)
                if not session:
                    text, rows = "向导已过期，请重新开始。", self._back("m:stock")
                elif action == "cancel":
                    self.store.delete_session(sid)
                    text, rows = "已取消，未执行任何变更。", self._back("m:stock")
                elif action == "input":
                    step = {"amount": "input_amount", "shares": "input_shares", "price": "input_price", "barriers": "input_barriers"}[value]
                    self.store.update_session(sid, step=step)
                    prompts = {
                        "amount": "请输入USDC金额：", "shares": "请输入股数：", "price": "请输入限价：",
                        "barriers": "请输入：止盈%,止损%,持仓天数，例如 3,2,7",
                    }
                    if value == "price":
                        side = str(session["payload"].get("side", "BUY"))
                        context, _ = self._stock_quote_context(session["payload"]["symbol"], side)
                        text = f"{context}\n\n请输入自定义LIMIT入场价（USDC）："
                    elif value == "barriers" and session["flow"] == "stock_position":
                        context, quote = self._stock_quote_context(session["payload"]["symbol"], "BUY")
                        entry = session["payload"].get("price", quote["ask"])
                        text = (
                            f"{context}\n\n当前入场参考：{entry} USDC\n"
                            "请输入：止盈%,止损%,持仓天数，例如 3,2,7"
                        )
                    else:
                        text = prompts[value]
                    rows = [[("取消", f"w:{sid}:cancel:-")]]
                elif session["flow"] == "stock_order":
                    text, rows = self._stock_order_step(session, action, value)
                else:
                    text, rows = self._stock_position_step(session, action, value)
            elif data.startswith("a:"):
                text, rows = self._handle_model_approval(data, callback, update_id)
            elif data.startswith("x:"):
                text, rows = self._handle_session_action(data, callback, update_id)
            else:
                raise ValueError("未知操作")
            self._edit_or_send(chat_id, message_id, text, rows)
            self.telegram.answer_callback(callback_id, "已处理")
            self.store.audit("CALLBACK_COMPLETE", user_id=user_id, chat_id=chat_id, update_id=update_id,
                             callback_id=callback_id, details={"action": data})
        except Exception as exc:
            logger.warning("callback failed action=%s type=%s", data, type(exc).__name__)
            self.telegram.answer_callback(callback_id, _safe_text(exc, 180), True)
            self.store.audit("CALLBACK_FAILED", user_id=user_id, chat_id=chat_id, update_id=update_id,
                             callback_id=callback_id, details={"action": data, "error": _safe_text(exc)})

    def _handle_message(self, update_id: int, message: dict) -> None:
        user_id = int(message.get("from", {}).get("id", 0))
        chat = message.get("chat", {})
        chat_id = int(chat.get("id", 0))
        if not self._authorized(user_id, chat_id, str(chat.get("type", ""))):
            self.store.audit("ACCESS_DENIED", user_id=user_id, chat_id=chat_id, update_id=update_id,
                             details={"surface": "message"})
            return
        text = str(message.get("text", "")).strip()
        try:
            if text in {"/start", "/menu", "菜单"}:
                self.store.clear_sessions(user_id, chat_id)
                rendered, rows = self._home()
            else:
                session = self.store.active_session(user_id, chat_id)
                if not session:
                    rendered, rows = "请使用菜单按钮开始操作。", HOME_ROWS
                else:
                    rendered, rows = self._handle_text_session(message, session)
            self.telegram.send(chat_id, rendered, rows)
            self.store.audit("MESSAGE_COMPLETE", user_id=user_id, chat_id=chat_id, update_id=update_id,
                             details={"command": text[:50]})
        except Exception as exc:
            self.telegram.send(chat_id, f"输入无效：{_safe_text(exc)}", [[("🏠 主菜单", "m:home")]])
            self.store.audit("MESSAGE_FAILED", user_id=user_id, chat_id=chat_id, update_id=update_id,
                             details={"error": _safe_text(exc)})

    def handle_update(self, update: dict) -> None:
        update_id = int(update.get("update_id", -1))
        if update_id < 0 or not self.store.claim_update(update_id):
            return
        if isinstance(update.get("callback_query"), dict):
            self._handle_callback(update_id, update["callback_query"])
        elif isinstance(update.get("message"), dict):
            self._handle_message(update_id, update["message"])
        self.store.set_metadata("telegram_offset", update_id + 1)

    def _health(self, error: str = "") -> None:
        payload = {
            "schema": "trading-management-bot-health-v1",
            "generated_at": time.time(),
            "mode": self.mode_label,
            "admin_configured": self.settings.admin_user_id > 0,
            "last_error": error,
        }
        temporary = self.settings.health_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        os.replace(temporary, self.settings.health_path)

    def run(self) -> None:
        if self.store.metadata("telegram_initialized") != "true":
            self.telegram.delete_webhook(drop_pending_updates=True)
            self.store.set_metadata("telegram_initialized", "true")
        offset = int(self.store.metadata("telegram_offset", "0") or 0)
        self._health()
        while self.running:
            try:
                for update in self.telegram.get_updates(offset, timeout=25):
                    self.handle_update(update)
                    offset = max(offset, int(update.get("update_id", -1)) + 1)
                    self.store.set_metadata("telegram_offset", offset)
                self._notify_new_model_candidates()
                self._health()
            except TelegramError as exc:
                logger.warning("Telegram polling transient failure: %s", exc)
                self._health(str(exc))
                time.sleep(3)
            except Exception as exc:
                logger.exception("management loop failed")
                self._health(f"{type(exc).__name__}: {_safe_text(exc)}")
                time.sleep(3)


def main() -> int:
    bot = TradingManagementBot(Settings.from_env())
    signal.signal(signal.SIGTERM, bot.stop)
    signal.signal(signal.SIGINT, bot.stop)
    try:
        bot.run()
    finally:
        bot.store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

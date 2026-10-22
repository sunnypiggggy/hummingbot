"""Read-only model and parameter pages; callers own data access and the clock."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from typing import Any, Mapping


BEIJING_TZ = timezone(timedelta(hours=8))


def permission_cn(value: Any) -> str:
    if value is True:
        return "放行"
    if value is False:
        return "阻止"
    return "不适用"


def state_cn(value: Any) -> str:
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


def pct(value: Any) -> str:
    try:
        return f"{Decimal(str(value)) * 100:.4f}%"
    except (InvalidOperation, ValueError, TypeError):
        return "无可信数据"


def duration(value: Any) -> str:
    try:
        seconds = int(value)
    except (TypeError, ValueError):
        return "-"
    if seconds % 86400 == 0:
        return f"{seconds // 86400}天"
    if seconds % 3600 == 0:
        return f"{seconds // 3600}小时"
    if seconds % 60 == 0:
        return f"{seconds // 60}分钟"
    return f"{seconds}秒"


def parameter_text(parameters: Mapping[str, Any]) -> str:
    labels = {
        "loss_limit_quote": "亏损阈值", "drawdown_limit_pct": "回撤阈值",
        "cooldown_seconds": "冷却", "stop_loss_pct": "止损",
        "contract_max_age_seconds": "最大数据年龄",
        "maximum_extra_inventory_hold_seconds": "额外库存最长持有",
        "healthy_cycles_before_reentry": "恢复健康周期",
        "mode": "模式", "update_cycle": "更新周期",
    }
    values = []
    for key, value in parameters.items():
        if key not in labels:
            continue
        if key.endswith("_seconds"):
            shown = duration(value)
        elif key.endswith("_pct"):
            shown = pct(value)
        else:
            shown = str(value)
        values.append(f"{labels[key]}={shown}")
    return "；".join(values) or "无固定阈值"


def bj_time(value: Any) -> str:
    if value in (None, ""):
        return "无可信记录"
    try:
        stamp = float(value)
    except (TypeError, ValueError):
        try:
            stamp = datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp()
        except ValueError:
            return "无可信记录"
    return datetime.fromtimestamp(stamp, timezone.utc).astimezone(BEIJING_TZ).strftime("%Y-%m-%d %H:%M")


def model_status_cn(value: Any) -> str:
    return {
        "AWAITING_APPROVAL": "等待审批", "APPROVED_PENDING_PREWARM": "已批准，等待预热",
        "PREWARMED_PENDING_ACTIVATION": "预热完成，等待激活",
        "WARM_ACTIVE_PENDING_FOLD": "已热切换，等待进入目标周",
        "ACTIVE": "已激活", "BLOCKED": "检查失败", "SIGNED_WEEK_UNAVAILABLE": "签名周不可用",
    }.get(str(value), state_cn(value))


def check_label(value: Any) -> str:
    key = str(value)
    labels = {
        "hashes": "模型文件完整性", "model_hashes": "模型文件完整性",
        "continuity": "签名周连续性", "week_continuity": "签名周连续性",
        "evidence": "回测证据完整性", "telegram_png_evidence": "频道证据送达",
        "current_week": "当前签名周有效", "candidate_not_expired": "候选有效期",
        "minimum_signed_runway": "签名覆盖时长", "grid_emergency_channel_ready": "Grid退出通道",
        "dca_emergency_channel_ready": "DCA退出通道", "grid_ownership_covered": "Grid库存归属",
        "dca_ownership_covered": "DCA库存归属", "exchange_filters_verified": "交易所规则校验",
        "candidate_stays_closed": "候选默认保持未授权",
        "fallback_forbidden": "禁止回退旧模型",
        "immutable_package_integrity": "发布包完整性",
        "release_directory_matches": "发布目录身份一致",
        "signed_week_is_contiguous": "签名周连续性",
    }
    return labels.get(key, "候选完整性检查")


def model_type_cn(value: Any) -> str:
    return {"v22_weekly_buy_gate": "v22 周度模型"}.get(str(value), "模型候选")


def robot_line(active: Mapping[str, Any], strategy: str, pair: str) -> str:
    row = active.get("trading", {}).get(f"{strategy}:{pair}", {})
    if not row:
        return f"{strategy.upper()} {pair}：数据不可用（最终权限请查看风控状态）"
    permissions = row.get("final_permissions", {})
    buy = permissions.get("buy_enabled", permissions.get("buy"))
    sell = permissions.get("sell_enabled", permissions.get("sell"))
    state = "正常交易" if row.get("trading_normal") else "交易受限"
    return f"{strategy.upper()} {pair}：{state}（BUY {permission_cn(buy)} / SELL {permission_cn(sell)}）"


def models(catalog: Mapping[str, Any]) -> tuple[str, list]:
    models = catalog.get("models", {})
    pending = models.get("candidate", [])
    lines = [
        "⚙️ 模型与参数（只读）",
        f"快照年龄：{catalog.get('age_seconds', 0):.0f}秒",
        f"当前候选：{'1个' if pending else '无'}",
        f"已上线历史模型：{min(3, len(models.get('history', [])))}个",
        "",
        "模型按当前、候选、历史分开查看；此页面不能修改参数或审批模型。",
    ]
    rows = [
        [("🟦 Grid 参数", "p:grid"), ("🟩 DCA 参数", "p:dca")],
        [("🛡 风控门参数", "p:risk")],
        [("📈 近一周概率与阈值", "p:probability")],
        [("🧠 当前模型", "p:model"), ("🆕 候选模型", "p:candidate")],
        [("🕒 历史模型", "p:model_history")],
        [("🔄 刷新", "m:models"), ("🏠 主菜单", "m:home")],
    ]
    return "\n".join(lines), rows


def grid_parameters(catalog: Mapping[str, Any], asset: str) -> tuple[str, list]:
    grid = catalog.get("grid", {})
    pair = f"{asset}-FDUSD"
    effective = grid.get("pairs", {}).get(pair, {}).get("effective", {})
    configured = grid.get("configured", {})
    runtime = grid.get("runtime", {})
    value = {**configured, **effective}
    parameter_version = runtime.get("active_parameter_version") or value.get("active_parameter_version", "-")
    lines = [
        f"🟦 Grid {pair} 当前生效参数",
        f"落地状态：{state_cn(grid.get('application_state'))}",
        f"参数版本：{parameter_version}",
        "",
        f"单对资金：{value.get('pair_budget_quote', '-')} FDUSD",
        f"单侧资金：{value.get('side_budget_quote', '-')} FDUSD",
        f"Grid总范围：{pct(value.get('grid_range'))}",
        f"Grid层数：{value.get('grid_levels', '-')}",
        f"止盈：{pct(value.get('take_profit'))}",
        f"移动阈值：{pct(value.get('move_threshold'))}",
        f"移动冷却：{duration(value.get('min_grid_move_seconds'))}",
        f"挂单刷新：{duration(value.get('order_refresh_time') or value.get('order_refresh_seconds'))}",
        f"最小订单：{value.get('minimum_order_quote', value.get('min_order_quote', '-'))} FDUSD",
        f"Maker费用：{pct(value.get('fee_rate', 0))}",
    ]
    if grid.get("runtime_override_active"):
        lines.extend(["", "参数来源：已批准的运行时参数覆盖（现网生效）"])
    rows = [[("BTC-FDUSD", "p:grid:BTC"), ("ETH-FDUSD", "p:grid:ETH")],
            [("⬅️ 模型与参数", "m:models"), ("🏠 主菜单", "m:home")]]
    return "\n".join(lines), rows


def dca_parameters(catalog: Mapping[str, Any], asset: str) -> tuple[str, list]:
    pair = f"{asset}-USDT"
    row = catalog.get("dca", {}).get(pair, {})
    value = row.get("effective", {})
    lines = [
        f"🟩 DCA {pair} 当前生效参数",
        f"落地状态：{state_cn(row.get('application_state'))}",
        "",
        f"执行资金：{value.get('total_amount_quote', '-')} USDT",
        f"DCA跌幅档位：{value.get('dca_spreads', '-')}",
        f"资金比例：{value.get('dca_amounts', '-')}",
        f"止盈：{pct(value.get('take_profit'))}",
        f"止损：{pct(value.get('stop_loss'))}",
        f"Executor刷新：{duration(value.get('executor_refresh_time'))}",
        f"持仓时限：{duration(value.get('time_limit'))}",
        f"普通冷却：{duration(value.get('cooldown_time'))}",
        f"SELL止损冷却：{duration(value.get('sell_stop_cooldown_seconds'))}",
        f"方向模式：{'只做多' if value.get('long_only_enabled', True) else '双向'}",
        f"SELL趋势门：{'开启' if value.get('sell_trend_gate_enabled') else '关闭'}",
    ]
    rows = [[("BTC-USDT", "p:dca:BTC"), ("ETH-USDT", "p:dca:ETH")],
            [("⬅️ 模型与参数", "m:models"), ("🏠 主菜单", "m:home")]]
    return "\n".join(lines), rows


def risk_parameters(catalog: Mapping[str, Any], strategy: str, asset: str) -> tuple[str, list]:
    pair = f"{asset}-{'FDUSD' if strategy == 'grid' else 'USDT'}"
    risks = catalog.get("risks", {})
    configured = risks.get(strategy, {})
    current = risks.get("current", {}).get(f"{strategy}:{pair}", {})
    parameter_map = configured.get("parameters", {})
    permissions = current.get("final_permissions", {})
    final_buy = permissions.get("buy_enabled", permissions.get("buy"))
    final_sell = permissions.get("sell_enabled", permissions.get("sell"))
    lines = [
        f"🛡 {strategy.upper()} {pair} 风控门参数",
        f"交易结论：{'正常交易' if current.get('trading_normal') else '交易受限'}",
        f"最终权限：BUY={permission_cn(final_buy)} / SELL={permission_cn(final_sell)}",
        "",
    ]
    for gate in current.get("gates", []):
        mechanism = str(gate.get("mechanism", ""))
        params = parameter_map.get(mechanism, {})
        enabled = "开" if gate.get("enabled") else "关"
        lines.append(
            f"• {gate.get('label', mechanism)}｜{enabled}｜{state_cn(gate.get('state'))}\n"
            f"  BUY {permission_cn(gate.get('buy_enabled'))} / SELL {permission_cn(gate.get('sell_enabled'))}\n"
            f"  {parameter_text(params)}"
        )
    rows = [
        [("Grid BTC", "p:risk:grid:BTC"), ("Grid ETH", "p:risk:grid:ETH")],
        [("DCA BTC", "p:risk:dca:BTC"), ("DCA ETH", "p:risk:dca:ETH")],
        [("⬅️ 模型与参数", "m:models"), ("🏠 主菜单", "m:home")],
    ]
    return "\n".join(lines)[:4000], rows


def current_model(catalog: Mapping[str, Any]) -> tuple[str, list]:
    active = catalog.get("models", {}).get("active", {})
    healthy = bool(active.get("system_healthy"))
    lines = [
        "🧠 v22 当前模型",
        f"系统状态：{'正常' if healthy else '异常'}",
        f"v22门：{'信号正常' if healthy else '模型信号不可用'}",
        f"当前模型周：第{active.get('model_week', '-')}周",
        f"有效期（北京时间）：{bj_time(active.get('week_start'))} 至 {bj_time(active.get('week_end'))}",
        "",
    ]
    for asset in ("BTC", "ETH"):
        pair = f"{asset}-FDUSD"
        row = active.get("pairs", {}).get(pair, {})
        signal = row.get("model_signal")
        if not signal and "risk_off_active" in row:
            signal = "RISK_OFF" if row.get("risk_off_active") else "RISK_ON"
        if signal == "RISK_ON":
            lines.append(f"{asset}：Risk-On（普通BUY放行）")
        elif signal == "RISK_OFF":
            lines.append(f"{asset}：Risk-Off（普通BUY阻止）")
        else:
            lines.append(f"{asset}：无可信模型信号（普通BUY权限以完整风控门为准）")
    lines.extend(["", robot_line(active, "grid", "BTC-FDUSD"),
                  robot_line(active, "grid", "ETH-FDUSD"),
                  robot_line(active, "dca", "BTC-USDT"),
                  robot_line(active, "dca", "ETH-USDT"), ""])
    rows: list[list[tuple[str, str]]] = []
    if int(active.get("exact_evidence_count") or 0) >= 4:
        rows.extend([
            [("Grid BTC 360天", "p:img:current:grid:BTC"), ("Grid ETH 360天", "p:img:current:grid:ETH")],
            [("DCA BTC 360天", "p:img:current:dca:BTC"), ("DCA ETH 360天", "p:img:current:dca:ETH")],
        ])
    else:
        lines.append("当前模型精确360天回测：缺失")
        lines.append("不会使用历史图片替代，也不会自动补生成。")
    rows.extend([[("📈 近一周概率与阈值", "p:probability")],
                 [("🆕 候选模型", "p:candidate"), ("🕒 历史模型", "p:model_history")],
                 [("⬅️ 模型与参数", "m:models"), ("🏠 主菜单", "m:home")]])
    return "\n".join(lines), rows


def candidate_model(catalog: Mapping[str, Any], *, now: float) -> tuple[str, list]:
    candidates = catalog.get("models", {}).get("candidate", [])
    if not candidates:
        return "🆕 v22 候选模型\n\n当前没有尚未激活的候选模型。", [
            [("🧠 模型审批", "m:approvals"), ("⬅️ 模型与参数", "m:models")]]
    item = candidates[0]
    checks = item.get("checks", {})
    passed = sum(value is True for value in checks.values())
    failed = [check_label(key) for key, value in checks.items() if value is not True]
    deadline = item.get("review_deadline")
    remaining = max(0, int(float(deadline)) - int(now)) if deadline else 0
    evidence_count = int(item.get("exact_evidence_count") or 0)
    lines = [
        "🆕 v22 候选模型",
        f"状态：{model_status_cn(item.get('status'))}",
        f"目标模型周：第{item.get('model_week', '-')}周",
        f"有效期（北京时间）：{bj_time(item.get('effective_start'))} 至 {bj_time(item.get('effective_end'))}",
        f"自动审批倒计时：{remaining // 3600}小时{remaining % 3600 // 60}分钟" if deadline else "自动审批倒计时：未开始",
        f"硬门槛：{passed}/{len(checks)}项通过" if checks else "硬门槛：无可信检查结果",
        f"失败原因：{'、'.join(failed) if failed else ('候选检查失败，请查看审批证据' if item.get('last_error') else '无')}",
        f"360天证据：{min(evidence_count, 4)}/4张已验证",
        "",
        "本页只读；批准或拒绝请进入模型审批。",
    ]
    candidate_id = str(item.get("release_sha256") or "")[:16]
    rows = [[("前往模型审批", f"a:{candidate_id}:view")],
            [("⬅️ 模型与参数", "m:models"), ("🏠 主菜单", "m:home")]]
    return "\n".join(lines), rows


def model_history(catalog: Mapping[str, Any]) -> tuple[str, list]:
    history = catalog.get("models", {}).get("history", [])[:3]
    lines = ["🕒 v22 历史模型", "仅展示有可信激活记录、曾参与实盘且已下线的最近3个模型。", ""]
    rows = []
    if not history:
        lines.append("当前没有可确认的已下线实盘模型记录。")
    for index, item in enumerate(history):
        lines.append(f"{index + 1}. 上线 {bj_time(item.get('activated_at'))}｜下线 {bj_time(item.get('retired_at'))}")
        rows.append([(f"查看历史模型 {index + 1}", f"p:hist_model:{index}")])
    rows.append([("⬅️ 模型与参数", "m:models"), ("🏠 主菜单", "m:home")])
    return "\n".join(lines), rows


def model_history_detail(catalog: Mapping[str, Any], index: int) -> tuple[str, list]:
    history = catalog.get("models", {}).get("history", [])[:3]
    if index < 0 or index >= len(history):
        raise ValueError("历史模型不存在或列表已更新")
    item = history[index]
    lines = [
        f"🕒 历史模型 {index + 1}",
        f"模型周：第{item.get('model_week', '-')}周",
        f"签名有效期（北京时间）：{bj_time(item.get('week_start'))} 至 {bj_time(item.get('week_end'))}",
        f"实际上线：{bj_time(item.get('activated_at'))}",
        f"实际下线：{bj_time(item.get('retired_at'))}",
        f"替换原因：{'周度模型稳定切换' if item.get('replacement_reason') == 'MODEL_CUTOVER_STABLE' else '无可信记录'}",
    ]
    rows: list[list[tuple[str, str]]] = []
    if int(item.get("exact_evidence_count") or 0) >= 4:
        rows.extend([
            [("Grid BTC 360天", f"p:img:history:{index}:grid:BTC"),
             ("Grid ETH 360天", f"p:img:history:{index}:grid:ETH")],
            [("DCA BTC 360天", f"p:img:history:{index}:dca:BTC"),
             ("DCA ETH 360天", f"p:img:history:{index}:dca:ETH")],
        ])
    else:
        lines.append("精确360天证据：无可信记录")
    rows.append([("⬅️ 历史模型", "p:model_history"), ("🏠 主菜单", "m:home")])
    return "\n".join(lines), rows

"""Read-only, escaped Telegram risk cards; never infer a release deadline."""
from datetime import datetime, timezone, timedelta
from html import escape


class RichText(str):
    """Explicit opt-in to Telegram HTML (bold headings and readable cards)."""


RULES = {
    "v22_weekly_buy_gate": ("v22模型判断当前不适合新增买入", "等待本交易对模型恢复放行，并通过其他风控门及重入检查"),
    "fomc_gate": ("宏观事件限制窗口正在生效", "等待已批准限制窗口结束，并确认宏观状态更新正常"),
    "strategy_loss_breaker": ("本机器人累计亏损达到保护阈值", "完成归属库存退出、冷却及健康检查后，等待其他门放行再重入"),
    "strategy_drawdown_breaker": ("本机器人权益从峰值回撤达到保护阈值", "完成归属库存退出、冷却及健康检查后，等待其他门放行再重入"),
    "portfolio_loss_breaker": ("本策略组合亏损达到保护阈值", "组合相关机器人全部完成退出、冷却及恢复检查"),
    "portfolio_drawdown_breaker": ("本策略组合权益回撤达到保护阈值", "组合相关机器人全部完成退出、冷却及恢复检查"),
    "position_protection": ("持仓触发止损或持仓期限保护", "保护性退出完成、冷却结束且其余门放行后恢复"),
    "inventory_ownership_gate": ("库存归属或账户余额核对尚未通过", "等待新鲜账本与交易所余额核对一致、归属缺口归零"),
    "infrastructure_integrity_breaker": ("运行数据或完整性检查尚未通过", "修复对应故障，确认数据新鲜及完整性通过；若已锁存，还需人工复核解锁"),
    "controller_application_gate": ("控制器尚未确认应用最新交易权限", "等待控制器同步成功并复核实际权限"),
    "order_execution_gate": ("挂单执行或订单核对尚未完成", "完成撤单及订单核对，确认挂单恢复正常"),
    "recovery_phase_gate": ("机器人正在退出、冷却或等待重入", "完成当前恢复阶段，且所有生效门共同放行"),
}


def attention(gate):
    return (gate.get("enabled", True) and gate.get("applicable", True)
            and (gate.get("buy_enabled") is not True or gate.get("sell_enabled") is not True
                 or gate.get("health", "HEALTHY") != "HEALTHY"))


def explanation(gate, robot, reason_cn):
    mechanism = gate.get("mechanism", "")
    cause, condition = RULES.get(mechanism, ("该项风控检查未通过，具体原因尚未提供中文说明", "等待权威风控状态确认该门恢复放行"))
    raw = str(gate.get("reason") or "")
    translated = reason_cn(raw)
    if translated != raw and translated and raw not in {"unknown", "healthy"}:
        cause = translated
    for token, message in (
        ("stale", "数据已过期，无法确认当前交易权限"),
        ("hash", "模型或数据校验值不一致，完整性检查失败"),
        ("ownership_deficit", "归属库存超过账户实际余额，存在归属缺口"),
        ("drawdown=", "权益从峰值回撤达到保护阈值"),
        ("signed_week", "当前时段缺少有效签名周模型"),
    ):
        if token in raw:
            cause = message
            break
    recovery = robot.get("recovery", {})
    if not isinstance(recovery, dict):
        recovery = {}
    technical_wait = recovery.get("mechanism") == "v22_weekly_buy_gate"
    if mechanism == "recovery_phase_gate" and technical_wait:
        cause = "技术风控退出后仍在等待v22模型恢复放行"
        condition = "等待v22恢复放行、连续健康检查通过，且其他门放行后自动重入"
    if gate.get("state") == "UNAVAILABLE" or "signed_week" in raw or "hash" in raw or "stale" in raw:
        condition = "等待有效模型及新鲜数据通过完整性检查；若已锁存，还需人工复核解锁"
    phase = str(robot.get("phase", "UNKNOWN")).upper()
    eta = "无法按时间预计，取决于上述解除条件"
    if phase == "LATCHED":
        condition = "先修复根因并复核退出与其他风控门，再通过人工审批解锁；不会自动解除"
        eta = "待人工处理，无自动解除时间"
    elif phase == "EXITING":
        eta = "退出尚未完成，暂不能预计恢复时间"
    elif phase == "REENTRY" and "disabled" in str(recovery.get("reentry_block_reason", "")):
        condition = "自动重入开关未开启；需授权开启，并确认其余风控门放行"
        eta = "等待恢复授权，时间未定"
    else:
        # Only use a timer belonging to this mechanism, never a model TTL.
        deadline = gate.get("cooldown_until")
        if mechanism in {recovery.get("mechanism"), "recovery_phase_gate"} and not technical_wait:
            deadline = deadline or recovery.get("cooldown_until")
        if mechanism == "v22_weekly_buy_gate":
            deadline = None
        try:
            if deadline:
                dt = (datetime.fromtimestamp(float(deadline), timezone.utc)
                      if isinstance(deadline, (int, float)) or str(deadline).replace('.', '', 1).isdigit() else
                      datetime.fromisoformat(str(deadline).replace("Z", "+00:00")))
                if dt.tzinfo is None:
                    raise ValueError("timezone required")
                when = dt.astimezone(timezone(timedelta(hours=8))).strftime("%m月%d日 %H:%M:%S")
                eta = (f"北京时间 {when} 冷却结束；届时仍需通过恢复检查"
                       if dt > datetime.now(timezone.utc) else
                       f"冷却已于北京时间 {when} 结束；仍在等待恢复条件")
        except (ValueError, TypeError, OverflowError, OSError):
            pass
    return cause, condition, eta


def render_risk(robots, age, state_cn, reason_cn, *, detail=False, page=0):
    pages = 1
    lines = ["<b>🛡 Grid / DCA 风控状态</b>", f"数据年龄：{int(age)}秒（时间均为北京时间）"]
    if not detail:
        lines += [f"{sum(bool(r.get('trading_normal')) for r in robots)}/{len(robots)} 正常交易"]
    for robot in robots:
        normal = bool(robot.get("trading_normal"))
        status = ("正常交易" if normal else "停止交易" if robot.get("process_running") is False
                  else "数据不可用" if robot.get("trade_mode") == "UNKNOWN" else "交易受限")
        perms = robot.get("final_permissions", {})
        permission = lambda v: "放行" if v is True else "阻止" if v is False else "未知"
        lines += ["", f"<b>{escape(str(robot.get('strategy', '')).upper())} {escape(str(robot.get('pair', '')))}</b>",
                  f"{'✅' if normal else '🔴'} {status}",
                  f"BUY {permission(perms.get('buy_enabled'))}｜SELL {permission(perms.get('sell_enabled'))}",
                  f"阶段：{escape(state_cn(robot.get('phase')))}"]
        gates = [g for g in robot.get("gate_statuses", []) if isinstance(g, dict) and g.get("applicable", True)]
        blocked = [g for g in gates if attention(g)]
        if robot.get("process_running") is False:
            lines += ["原因：机器人进程已停止", "解除条件：确认停止原因并启动后复核全部风控门", "预计解除：等待维护操作，时间未定"]
        pages = max(1, (len(blocked) + 2) // 3)
        page = min(max(0, page), pages - 1)
        selected = blocked[page * 3:page * 3 + 3] if detail else blocked[:1]
        if detail and pages > 1:
            lines += [f"阻塞详情：第 {page+1}/{pages} 页，共 {len(blocked)} 项"]
        for gate in selected:
            cause, condition, eta = explanation(gate, robot, reason_cn)
            label = escape(str(gate.get("label", "风控检查"))[:40])
            lines += [f"\n<b>⛔ {label}</b>", f"原因：{escape(cause)}"]
            if detail:
                lines += [f"影响：BUY {permission(gate.get('buy_enabled'))}｜SELL {permission(gate.get('sell_enabled'))}"]
            if detail:
                lines += [f"解除条件：{escape(condition)}", f"预计解除：{escape(eta)}"]
            else:
                lines += [f"共 {len(blocked)} 项限制；点交易对查看解除条件与预计时间"]
        if not blocked:
            lines += ["当前没有阻塞交易的风控门。" if normal else "状态尚未确认正常；需复核进程、阶段及数据，预计恢复时间未定。"]
        if detail:
            lines += ["", "<b>其余门控</b>"]
            for gate in gates:
                if not attention(gate):
                    state = "关闭" if not gate.get("enabled", True) else "正常" if gate.get("state") == "OK" else state_cn(gate.get("state"))
                    lines += [escape(f"• {gate.get('label', '风控检查')}：{state}")]
                    if gate.get("state") == "ALERT_ONLY":
                        lines += [escape(reason_cn(gate.get("reason")))]
    result = RichText("\n".join(lines))
    result.pages = pages if detail else 1
    result.page = page
    return result

"""Read-only scheduled order presentation using persisted parameters."""
import hashlib
import time
from datetime import datetime, timezone, timedelta
from decimal import Decimal, InvalidOperation
from html import escape

from management_bot.risk_display import RichText

CANCELABLE = {"QUEUED", "WAITING_SESSION", "WAITING_PREFLIGHT", "ACTIVATING"}


def number(value):
    try:
        v = Decimal(str(value))
        return v if v.is_finite() else None
    except (InvalidOperation, ValueError, TypeError):
        return None


def fmt(value):
    v = number(value)
    return "未记录" if v is None else format(v.quantize(Decimal("0.0001")), "f").rstrip("0").rstrip(".") or "0"


def short_id(item):
    return hashlib.sha256(str(item.get("schedule_id", "")).encode()).hexdigest()[:10].upper()


def identity(item):
    p = item.get("request_payload") or {}
    position = item.get("request_type") == "position_executor"
    side = p.get("side", "BUY" if position else "UNKNOWN")
    return str(p.get("symbol") or "股票未记录"), {"BUY": "买入", "SELL": "卖出"}.get(str(side), "方向未记录")


def label(item):
    symbol, side = identity(item)
    budget = item.get("quote_budget")
    amount = f"{fmt(budget)} USDC" if budget is not None else f"{fmt(item.get('requested_shares'))}股"
    return f"{symbol} · {side} · {amount} · #{short_id(item)}"


def bj(value):
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            return "未记录可信时间"
        return dt.astimezone(timezone(timedelta(hours=8))).strftime("%Y-%m-%d %H:%M 北京时间")
    except (ValueError, TypeError):
        return "未记录可信时间"


def waiting(item):
    reason = str(item.get("last_block_reason") or "")
    if item.get("status") == "ACTIVE":
        return "已激活，转由Executor管理", "请在Executor管理中查看成交和退出状态"
    if item.get("status") not in CANCELABLE:
        return "该计划已结束", "不再等待激活"
    for token, message, condition in (
        ("stale", "行情已过期", "等待可信新行情并重新通过开市预检"),
        ("cash", "可用资金检查未通过", "资金检查通过且目标交易时段开放"),
        ("whitelist", "股票白名单检查未通过", "白名单允许交易且开市预检通过"),
        ("recovery", "模拟账户恢复检查未通过", "完成账户恢复核验后重新预检"),
        ("unknown", "市场状态暂不可确认", "等待可信市场状态及订单预检通过"),
    ):
        if token in reason.lower():
            return message, condition
    if reason.startswith("waiting_for_") or any(t in reason.lower() for t in ("closed", "session", "market_not")):
        return "尚未进入该订单允许的交易时段", "目标交易时段开放，且行情、资金、白名单及风控预检通过后激活"
    return "等待开市或权威预检通过；具体阻塞尚无中文说明", "等待Runtime确认目标时段开放及全部预检通过"


def detail(item, status_cn, session_cn, *, quote_text="行情暂不可用", quote=None, mode="未确认"):
    p = item.get("request_payload") or {}
    position = item.get("request_type") == "position_executor"
    order_type = str(p.get("entry_order_type" if position else "order_type") or "UNKNOWN").upper()
    price = number(item.get("frozen_price"))
    if price is None:
        price = number(p.get("entry_price" if position else "price"))
    budget = item.get("quote_budget")
    shares = number(item.get("requested_shares"))
    reference = None
    q = quote or {}
    ts = number(q.get("quote_ts"))
    fresh = ts is not None and 0 <= time.time() - float(ts) <= 60
    if fresh:
        reference = number(q.get("reference"))
    base = price if order_type == "LIMIT" else reference
    cause, condition = waiting(item)
    lines = ["<b>⏰ 待开市订单详情</b>", escape(label(item)),
             f"状态：{escape(status_cn(item.get('status')))}", f"执行范围：{escape(mode)}",
             f"类型：{'仓位交易 PositionExecutor' if position else '单笔订单 OrderExecutor'}",
             "", "<b>入场条件</b>",
             f"订单类型：{ {'LIMIT':'限价', 'MARKET':'市价'}.get(order_type, '未记录')}",
             f"委托价：{fmt(price)} USDC" if order_type == "LIMIT" else "执行价格：开市按当时行情执行，成交价未确定",
             f"固定预算：{fmt(budget)} USDC，开市按最新行情重算股数" if budget is not None else f"固定股数：{fmt(shares)} 股",
             f"预计金额：{fmt(budget if budget is not None else shares * base if shares is not None and base is not None else None)} USDC",
             "", "<b>行情参考</b>", escape(quote_text)]
    if not fresh:
        lines.append("行情过期或时间未确认，仅供查看，不用于市价触发价估算。")
    if order_type == "LIMIT" and price is not None and reference is not None and reference > 0:
        lines.append(f"委托价相对当前买卖参考价：{fmt((price / reference - 1) * 100)}%")
    lines += ["", "<b>止盈止损</b>"]
    if position:
        # Saved request values only; old orders must never inherit today's defaults.
        for key, title, sign in (("take_profit", "止盈", 1), ("stop_loss", "止损", -1)):
            v = number(p.get(key))
            if key not in p:
                lines.append(f"{title}：未记录")
            elif v is None or v == 0:
                lines.append(f"{title}：未设置")
            else:
                lines.append(f"{title}比例：{fmt(v * 100)}%" + (
                    f"｜触发参考价：{fmt(base * (1 + sign * v))} USDC" if base is not None else "｜触发价：成交后确定"))
        lines.append(f"最长持仓：{fmt(number(p.get('time_limit')) / 86400)} 天" if number(p.get("time_limit")) is not None else "最长持仓：未记录")
        lines.append("参考价按委托入场价计算；实际触发以Executor成交入场基准为准。" if order_type == "LIMIT" else "当前行情参考，成交后确定实际触发价。")
    else:
        lines.append("单笔订单不附带仓位止盈止损。")
    lines += ["", "<b>等待与有效期</b>", f"目标时段：{escape(session_cn(item.get('target_session')))}",
              f"目标交易日：{escape(str(item.get('target_trading_date') or '尚未确定'))}",
              "有效期：目标有效交易日结束；最迟保留至 " + bj(item.get("hard_expires_at")),
              f"原因：{escape(cause)}", f"解除条件：{escape(condition)}",
              "预计开市：" + (bj(item["next_market_open_at"]) if item.get("next_market_open_at") else "暂无可信时间，等待交易日历确认")]
    return RichText("\n".join(lines))

"""Read-only simple audit presentation; no trading clients or credentials."""
import hashlib
import json
import time
from datetime import datetime, timedelta, timezone
from html import escape
from pathlib import Path

from management_bot.risk_display import RichText

ROWS = [[("📈 账户与机器人总额 · 30天", "sa:account")],
        [("📊 四机器人权益 · 30天", "sa:robots")],
        [("🔄 刷新", "m:simple_audit"), ("🏠 主菜单", "m:home")]]


def date(value):
    try:
        return datetime.fromtimestamp(float(value), timezone(timedelta(hours=8))).strftime("%m-%d %H:%M")
    except (ValueError, TypeError, OSError):
        return "未记录"


def amount(value):
    try:
        return f"{float(value):,.2f}"
    except (TypeError, ValueError):
        return "无可信数据"


def load(root):
    path = Path(root) / "simple_audit" / "summary.json"
    result = json.loads(path.read_text(encoding="utf-8"))
    if result.get("schema") != "simple-audit-v1":
        raise ValueError("schema")
    return result


def page(root, now=None):
    now = time.time() if now is None else now
    try:
        data = load(root)
    except Exception:
        return RichText("<b>🔎 简单稽核</b>\n\n尚无有效采集结果，请稍后刷新。"), ROWS
    stale = not -5 <= now - float(data.get("observed_at", 0)) <= 4500
    lines = ["<b>🔎 简单稽核</b>", "每小时更新 · 金额核对，不是独立账本审计", "",
             "<b>账户资产｜USDT计价</b>"]
    for asset in ("BTC", "ETH", "FDUSD", "USDT"):
        row = data.get("assets", {}).get(asset, {})
        try:
            quantity = f"{float(row['quantity']):.8f}".rstrip('0').rstrip('.')
        except (KeyError, TypeError, ValueError):
            quantity = "未记录"
        lines.append(f"• {asset}：{quantity}｜{amount(row.get('value'))} USDT")
    lines.extend([f"<b>账户总额：{amount(data.get('account_total'))} USDT</b>", "", "<b>机器人归属权益</b>"])
    for row in data.get("robots", []):
        lines.append(f"• {escape(str(row['strategy']).upper())} {escape(str(row['pair']))}："
                     f"{amount(row.get('equity'))} {escape(str(row['quote']))}")
    lines.extend([f"• Grid公共储备：{amount(data.get('reserve'))} FDUSD", "", "<b>金额核对</b>"])
    if stale:
        lines.append("结果已过期，以下为上次采集；当前合计核对不可用。")
    elif data.get("difference") is None:
        lines.append("核对不完整：暂不展示机器人合计和差额。")
    else:
        difference = float(data["difference"])
        label = "未分配／待核实" if difference > .01 else "归属或统计口径待核实" if difference < -.01 else "本次金额一致（不代表历史可信）"
        lines.extend([f"机器人及储备合计：{amount(data['robot_total'])} USDT",
                      f"账户减机器人：<b>{difference:+,.2f} USDT</b>", label])
    for issue in data.get("issues", [])[:8]:
        lines.append("• " + escape(str(issue)))
    coverage = data.get("coverage", {}).get("account") or {}
    lines.extend(["", "<b>数据时间｜北京时间</b>", f"采集：{date(data.get('observed_at'))}",
                  f"下次：{date(data.get('next_at'))}",
                  f"账户历史：{date(coverage.get('start'))} — {date(coverage.get('end'))}（{coverage.get('count',0)}点）",
                  "不足30天按实际记录展示；各机器人覆盖区间见图。"])
    return RichText("\n".join(lines)), ROWS


def chart(root, key):
    if key not in {"account", "robots"}:
        raise ValueError("unknown chart")
    item = load(root).get("charts", {}).get(key, {})
    folder = (Path(root) / "simple_audit").resolve()
    path = (folder / str(item.get("file", ""))).resolve()
    if path.parent != folder or path.name != key + ".png":
        raise ValueError("invalid path")
    if hashlib.sha256(path.read_bytes()).hexdigest() != item.get("sha256"):
        raise ValueError("invalid hash")
    return path

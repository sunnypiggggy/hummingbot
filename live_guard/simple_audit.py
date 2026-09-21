"""Read-only hourly asset overview. No ledger migration or trading operations."""
from __future__ import annotations

import hashlib
import json
import math
import os
import sqlite3
import time
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

ASSETS = ("BTC", "ETH", "FDUSD", "USDT")
ROBOTS = (("grid", "BTC-FDUSD"), ("grid", "ETH-FDUSD"),
          ("dca", "BTC-USDT"), ("dca", "ETH-USDT"))
BJT = timezone(timedelta(hours=8))


def number(value):
    value = Decimal(str(value))
    if not value.is_finite():
        raise ValueError("nonfinite")
    return value


def fresh(value, now, limit=180):
    try:
        return -5 <= now - float(value) <= limit
    except (ValueError, TypeError):
        return False


def balance_snapshot(balances, account, observed_at):
    """Caller supplies the completed, existing account query; never query again."""
    return {"schema": "simple-audit-balance-v1", "account": account,
            "observed_at": observed_at, "assets": {
                asset: {k: str(balances.get(asset, {}).get(k, 0)) for k in ("free", "locked", "total")}
                for asset in ASSETS}}


def collect(balance, grid, dca, inventory, quotes, now):
    result = {"schema": "simple-audit-v1", "observed_at": now,
              "next_at": (int(now) // 3600 + 1) * 3600, "assets": {}, "robots": [],
              "account_total": None, "robot_total": None, "difference": None,
              "reserve": None, "issues": [], "account": balance.get("account")}
    try:
        if not balance.get("account") or not fresh(balance.get("observed_at"), now, 60):
            raise ValueError("账户余额缺失或过期")
        if not fresh(quotes.get("observed_at"), now, 60):
            raise ValueError("估值行情缺失或过期")
        prices = {a: number(quotes["prices"][a]) for a in ASSETS}
        if any(v <= 0 for v in prices.values()) or prices["USDT"] != 1:
            raise ValueError("估值行情异常")
        total = Decimal(0)
        for asset in ASSETS:
            row = balance["assets"][asset]
            free, locked, quantity = (number(row[k]) for k in ("free", "locked", "total"))
            if min(free, locked, quantity) < 0 or free + locked != quantity:
                raise ValueError("账户可用与冻结余额不一致")
            value = quantity * prices[asset]
            result["assets"][asset] = {"quantity": str(quantity), "value": str(value)}
            total += value
        result.update(account_total=str(total), prices={a: str(p) for a, p in prices.items()},
                      balance_at=balance["observed_at"], quote_at=quotes["observed_at"])
    except (ValueError, KeyError, TypeError, ArithmeticError) as exc:
        result["issues"].append(str(exc) if isinstance(exc, ValueError) else "四币余额或行情不完整")
        return result
    latest_grid = grid.get("bots", {}).get("grid-live-fdusd-400", {}).get("latest", {})
    totals = {a: Decimal(0) for a in ASSETS}
    bindings_ok = (inventory.get("account_fingerprint") == balance["account"]
                   and inventory.get("sources_healthy") is True
                   and fresh(inventory.get("generated_at"), now))
    if not bindings_ok:
        result["issues"].append("账户绑定或归属证据不可用，暂停金额核对")
    for strategy, pair in ROBOTS:
        base, quote = pair.split("-")
        row = latest_grid.get("pairs", {}).get(pair, {}) if strategy == "grid" else (
            dca.get("bots", {}).get("dca-live-" + pair.replace("-", "").lower() + "-200", {}).get("latest", {}))
        stamp = latest_grid.get("observed_at") if strategy == "grid" else row.get("observed_at")
        item = {"strategy": strategy, "pair": pair, "quote": quote, "equity": None}
        try:
            if not fresh(stamp, now) or abs(float(stamp) - float(balance["observed_at"])) > 60:
                raise ValueError("归属快照与余额时点不一致")
            cash, owned = number(row["quote_balance"]), number(row["owned_base"])
            if min(cash, owned) < 0:
                raise ValueError("存在负归属")
            value = cash * prices[quote] + owned * prices[base]
            item.update(equity=str(value / prices[quote]), value=str(value),
                        cash=str(cash), owned_base=str(owned), source_at=stamp)
            totals[quote] += cash
            totals[base] += owned
        except (ValueError, KeyError, TypeError, ArithmeticError) as exc:
            result["issues"].append(f"{strategy.upper()} {pair}：" + (
                str(exc) if isinstance(exc, ValueError) else "现金或库存证据缺失"))
        result["robots"].append(item)
    try:
        # Same runtime snapshot: portfolio equity includes the common reserve once.
        reserve = number(latest_grid["equity"]) - sum(
            (number(latest_grid["pairs"][p]["equity"]) for s, p in ROBOTS if s == "grid"), Decimal(0))
        if reserve < 0 or not fresh(latest_grid.get("observed_at"), now):
            raise ValueError("invalid reserve")
        result["reserve"] = str(reserve)
        totals["FDUSD"] += reserve
    except (ValueError, KeyError, TypeError, ArithmeticError):
        result["issues"].append("Grid公共储备缺少同一运行快照证据")
    # Detect a concurrent fill/ownership update instead of reporting a false match.
    for asset in ("BTC", "ETH"):
        try:
            inv = inventory["assets"][asset]
            if abs(number(inv["owned_total"]) - totals[asset]) > Decimal("0.000000000001"):
                result["issues"].append(f"{asset}归属更新不同步，待稳定复核")
            if number(inv["exchange"]["total"]) != number(balance["assets"][asset]["total"]):
                result["issues"].append(f"{asset}采样期间余额变化，待稳定复核")
            deficit = totals[asset] - number(balance["assets"][asset]["total"])
            if deficit > Decimal("0.000000000001"):
                result["issues"].append(f"{asset}账面归属超出账户余额 {deficit:.8f}（约{deficit * prices[asset]:.4f} USDT），待核实")
        except (ValueError, KeyError, TypeError, ArithmeticError):
            result["issues"].append(f"{asset}归属组成缺失")
    if not result["issues"]:
        robot_total = sum((totals[a] * prices[a] for a in ASSETS), Decimal(0))
        result.update(robot_total=str(robot_total), difference=str(number(result["account_total"]) - robot_total))
    return result


def atomic_json(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    os.replace(temp, path)


def quote_snapshot(api_base):
    import requests
    started = time.time()
    response = requests.get(api_base + "/api/v3/ticker/bookTicker",
                            params={"symbols": json.dumps(["BTCUSDT", "ETHUSDT", "FDUSDUSDT"], separators=(",", ":"))}, timeout=8)
    response.raise_for_status()
    rows = {r["symbol"]: r for r in response.json()}
    return {"observed_at": started, "prices": {**{a: rows[a + "USDT"]["bidPrice"] for a in ASSETS[:-1]}, "USDT": "1"}}


def history_rows(connection, now):
    """Keep native stored equities; do not recalculate old accounting periods."""
    output = {}
    for strategy, pair in ROBOTS:
        rows = connection.execute(
            "SELECT observed_at,equity FROM profit_snapshot WHERE strategy=? AND pair=? "
            "AND observed_at>=? AND observed_at<=? ORDER BY observed_at",
            (strategy, pair, now - 30 * 86400, now)).fetchall()
        output[f"{strategy}:{pair}"] = [(float(t), None if v is None or not math.isfinite(float(v)) else float(v)) for t, v in rows]
    return output


def segments(points, now, hourly=False):
    output, segment, previous = [], [], None
    for t, value in points:
        # Existing report DB retains minute, hour and older daily resolutions.
        gap = 5400 if hourly or now - t <= 30 * 86400 else 129600
        if not hourly and now - t <= 8 * 86400:
            gap = 180
        if value is None or (previous is not None and t - previous > gap):
            if segment:
                output.append(segment)
            segment = []
        if value is not None:
            segment.append((t, value))
        previous = t
    if segment:
        output.append(segment)
    return output


def render(path, series, now, *, account=False):
    from PIL import Image, ImageDraw
    try:
        from telegram_notifications import report_font
    except ModuleNotFoundError:
        from live_guard.telegram_notifications import report_font
    image = Image.new("RGB", (1440, 1800 if account else 3200), "#ffffff")
    draw = ImageDraw.Draw(image)
    font, small, title = report_font(32), report_font(27), report_font(42, bold=True)
    draw.text((64, 40), "账户与机器人资产 · USDT" if account else "四机器人权益 · 近30天", fill="#17212f", font=title)
    draw.text((64, 106), datetime.fromtimestamp(now, BJT).strftime("采集：%Y-%m-%d %H:%M 北京时间 · 每小时更新"), fill="#526070", font=small)
    draw.text((64, 152), "蓝线：账户  灰线：机器人＋储备｜差额不代表盈亏" if account else "历史报告口径 · 各图独立纵轴，币种分别标注", fill="#526070", font=small)
    panels = [("账户四币合计", "USDT", series.get("account", [])),
              ("账户减机器人 · 差额", "USDT", series.get("difference", []))] if account else [
        (f"{s.upper()} {p}", p.split("-")[1], series.get(f"{s}:{p}", [])) for s, p in ROBOTS]
    start = now - 30 * 86400
    panel_h = 760 if account else 735
    for i, (label, unit, points) in enumerate(panels):
        top = 235 + i * panel_h
        valid = [(t, v) for t, v in points if v is not None and start <= t <= now]
        latest = f"{valid[-1][1]:,.2f} {unit}" if valid else "无可信数据"
        draw.text((64, top), f"{label}   {latest}", fill="#17212f", font=font)
        coverage = "暂无历史" if not valid else "覆盖 " + " — ".join(datetime.fromtimestamp(t, BJT).strftime("%m-%d %H:%M") for t in (valid[0][0], valid[-1][0]))
        draw.text((64, top+50), coverage + (" · 历史不足30天" if not valid or valid[0][0] > start + 3600 else ""), fill="#526070", font=small)
        left, right, y1, y2 = 175, 1345, top + 120, top + panel_h - 110
        overlay = series.get("robot_total", []) if account and i == 0 else []
        scale_values = [v for _,v in valid] + [v for t,v in overlay if v is not None and start <= t <= now]
        low, high = (min(scale_values), max(scale_values)) if scale_values else (0, 1)
        padding = max((high - low) * .1, abs(high) * .002, .01)
        low, high = low - padding, high + padding
        for j in range(4):
            y = y2 - j * (y2-y1)/3
            draw.line((left, y, right, y), fill="#e2e5ea", width=2)
            draw.text((20, y-18), f"{low+(high-low)*j/3:,.1f}", fill="#526070", font=small)
        for j in range(6):
            t = start + j * (now-start)/5
            x = left + j * (right-left)/5
            draw.text((x-46, y2+20), datetime.fromtimestamp(t, BJT).strftime("%m-%d"), fill="#526070", font=small)
        draw.line((left,y1,left,y2,right,y2), fill="#526070", width=2)
        for segment in segments(points, now, hourly=account):
            coords = [(left+(t-start)/(now-start)*(right-left), y2-(v-low)/(high-low)*(y2-y1)) for t,v in segment if start <= t <= now]
            if len(coords)>1:
                draw.line(coords, fill="#2563a6", width=4)
            elif coords:
                x,y=coords[0];draw.ellipse((x-5,y-5,x+5,y+5),fill="#2563a6")
        for segment in segments(overlay, now, hourly=True):
            coords = [(left+(t-start)/(now-start)*(right-left), y2-(v-low)/(high-low)*(y2-y1)) for t,v in segment if start <= t <= now]
            if len(coords)>1:
                draw.line(coords, fill="#626b76", width=4)
            elif coords:
                x,y=coords[0];draw.rectangle((x-6,y-6,x+6,y+6),outline="#626b76",width=3)
        if account and i == 0:
            values = [(t,v) for t,v in overlay if v is not None]
            text = "机器人＋储备：" + (f"{values[-1][1]:,.2f} USDT" if values and points and values[-1][0] == points[-1][0] else "核对不完整，当前不绘制")
            draw.text((64, y2+65), text, fill="#626b76", font=small)
        if not valid:
            draw.text((500,(y1+y2)/2),"暂无有效权益记录",fill="#526070",font=font)
    temp = path.with_suffix(".tmp")
    image.save(temp, format="PNG")
    os.replace(temp, path)


class SimpleAudit:
    def __init__(self, output):
        self.root = Path(output) / "simple_audit"
        self.root.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.root / "history.sqlite", timeout=5)
        self.db.execute("CREATE TABLE IF NOT EXISTS hours(slot INTEGER PRIMARY KEY, account TEXT, observed_at REAL, payload TEXT)")
        self.db.commit()

    def cycle(self, *, grid, dca, inventory, history_connection, api_base, now=None, quotes=None):
        now = time.time() if now is None else now
        slot = int(now)//3600
        previous = self.db.execute("SELECT payload FROM hours WHERE slot=?", (slot,)).fetchone()
        if previous and (self.root / "summary.json").exists():
            try:
                if json.loads((self.root / "summary.json").read_text(encoding="utf-8")).get("render_version") == 3:
                    return
            except (ValueError, OSError):
                pass
        try:
            quotes = quotes if quotes is not None else quote_snapshot(api_base)
            result = collect(dca.get("simple_audit_balance", {}), grid, dca, inventory, quotes, now)
        except Exception:
            result = {"schema":"simple-audit-v1", "observed_at":now, "next_at":(slot+1)*3600,
                      "account_total":None,"robots":[],"issues":["行情采集失败，等待重试"]}
        if previous:
            result = json.loads(previous[0])
        # Recheck saved quantities when upgrading display logic; never rewrite
        # the stored account history or invent a new same-hour observation.
        for asset in ("BTC", "ETH"):
            try:
                owned = sum((number(r["owned_base"]) for r in result["robots"] if r["pair"].startswith(asset+"-")), Decimal(0))
                deficit = owned - number(result["assets"][asset]["quantity"])
                if deficit > Decimal("0.000000000001"):
                    if not any(asset+"账面归属超出" in s for s in result["issues"]):
                        result["issues"].append(f"{asset}账面归属超出账户余额 {deficit:.8f}（约{deficit * number(result['prices'][asset]):.4f} USDT），待核实")
                    result.update(robot_total=None, difference=None)
            except (ValueError, KeyError, TypeError, ArithmeticError):
                pass
        result["render_version"] = 3
        account = result.get("account")
        saved_history = [(t, json.loads(p)) for t,p in self.db.execute(
            "SELECT observed_at,payload FROM hours WHERE account=? AND observed_at>=? ORDER BY observed_at", (account, now-30*86400))]
        if result.get("account_total") is not None and not previous:
            saved_history.append((now,result))
        series = history_rows(history_connection, now)
        series["account"] = [(t,None if row.get("account_total") is None else float(row["account_total"])) for t,row in saved_history]
        series["robot_total"] = [(t, comparison_value(row)) for t,row in saved_history]
        series["difference"] = [(t, None if comparison_value(row) is None or row.get("account_total") is None
                                  else float(number(row["account_total"]) - number(row["robot_total"]))) for t,row in saved_history]
        result["coverage"] = {}
        for key, points in series.items():
            valid_times = [t for t,v in points if v is not None]
            result["coverage"][key] = {"start":min(valid_times), "end":max(valid_times), "count":len(valid_times)} if valid_times else None
        result["charts"] = {}
        for key in ("account", "robots"):
            filename = key + ".png"
            try:
                render(self.root/filename, series, now, account=key=="account")
                result["charts"][key] = {"file":filename,"sha256":hashlib.sha256((self.root/filename).read_bytes()).hexdigest()}
            except Exception:
                result["issues"].append("图片生成失败："+key)
        atomic_json(self.root/"summary.json",result)
        # Failures retry next report cycle, never mark a failed observation done.
        if result.get("account_total") is not None and len(result["charts"])==2:
            self.db.execute("INSERT OR IGNORE INTO hours VALUES(?,?,?,?)",(slot,account,now,json.dumps(result,ensure_ascii=False)))
            self.db.execute("DELETE FROM hours WHERE observed_at<?", (now-90*86400,))
            self.db.commit()


def comparison_value(row):
    """Never expose legacy aggregate values that hide a per-asset deficit."""
    try:
        if row.get("issues") or row.get("robot_total") is None or len(row["robots"]) != 4:
            return None
        for asset in ("BTC", "ETH"):
            owned = sum((number(r["owned_base"]) for r in row["robots"] if r["pair"].startswith(asset+"-")), Decimal(0))
            if owned - number(row["assets"][asset]["quantity"]) > Decimal("0.000000000001"):
                return None
        return float(number(row["robot_total"]))
    except (ValueError, KeyError, TypeError, ArithmeticError):
        return None

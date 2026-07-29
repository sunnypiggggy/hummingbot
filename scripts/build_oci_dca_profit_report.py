#!/usr/bin/env python3
"""Build a detailed, auditable OCI DCA performance report from a saved snapshot."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime, timezone
from html import escape
from pathlib import Path

import plotly.graph_objects as go
from plotly.subplots import make_subplots


def fnum(value, digits=2):
    return f"{float(value or 0):,.{digits}f}"


def ts_text(value):
    if not value:
        return "-"
    value = float(value)
    if value > 10_000_000_000:
        value /= 1000
    return datetime.fromtimestamp(value, timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S")


def write_csv(path, rows):
    rows = list(rows)
    if not rows:
        path.unlink(missing_ok=True)
        return
    with path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate_executors(bot):
    side = defaultdict(lambda: {"executors": 0, "filled": 0, "pnl": 0.0, "fees": 0.0, "volume": 0.0})
    close = defaultdict(lambda: {"executors": 0, "filled": 0, "pnl": 0.0, "fees": 0.0, "volume": 0.0})
    level = defaultdict(lambda: {"executors": 0, "filled": 0, "pnl": 0.0, "fees": 0.0, "volume": 0.0})
    for row in bot["executors"]:
        for key, bucket in ((row["side"], side), (row["close_type"], close), (row.get("level_id") or "unknown", level)):
            item = bucket[key]
            item["executors"] += 1
            item["filled"] += int(float(row.get("filled_quote") or 0) > 0)
            item["pnl"] += float(row.get("net_pnl_quote") or 0)
            item["fees"] += float(row.get("fees") or 0)
            item["volume"] += float(row.get("filled_quote") or 0)
    return side, close, level


def chart_html(data):
    bots = data["bots"]
    fig = make_subplots(
        rows=3, cols=2,
        specs=[[{}, {}], [{}, {}], [{"colspan": 2}, None]],
        subplot_titles=("逐币市值口径收益", "净库存的当前市值", "买入与卖出成交额", "成交与费用", "每日累计成交现金流（未含每日库存重估）"),
        vertical_spacing=0.12,
    )
    colors = {"BTC-USDT": "#2563eb", "ETH-USDT": "#d97706", "BUY": "#2563eb", "SELL": "#d97706"}
    fig.add_trace(go.Bar(x=[b["pair"] for b in bots], y=[b["fill_summary"]["flow_mtm_pnl"] for b in bots],
                         text=[fnum(b["fill_summary"]["flow_mtm_pnl"]) for b in bots], textposition="outside",
                         marker_color=[colors.get(b["pair"], "#64748b") for b in bots], showlegend=False), row=1, col=1)
    fig.add_trace(go.Bar(x=[b["pair"] for b in bots],
                         y=[b["fill_summary"]["net_base"] * b["fill_summary"]["mark_price"] for b in bots],
                         marker_color=[colors[b["pair"]] for b in bots], showlegend=False), row=1, col=2)
    for bot in bots:
        fs = bot["fill_summary"]
        fig.add_trace(go.Bar(name=bot["pair"], x=["BUY", "SELL"], y=[fs["buy_quote"], fs["sell_quote"]],
                             marker_color=colors[bot["pair"]]), row=2, col=1)
    fig.add_trace(go.Bar(name="成交数", x=[b["pair"] for b in bots], y=[b["fill_summary"]["trades"] for b in bots],
                         marker_color="#2563eb"), row=2, col=2)
    fig.add_trace(go.Bar(name="费用 USDT", x=[b["pair"] for b in bots], y=[b["fill_summary"]["fees"] for b in bots],
                         marker_color="#d97706"), row=2, col=2)
    for bot in bots:
        running = 0.0
        xs, ys = [], []
        for d in bot["daily_fills"]:
            running += float(d["sell_quote"]) - float(d["buy_quote"]) - float(d["fees"])
            xs.append(d["date"]); ys.append(running)
        fig.add_trace(go.Scatter(name=bot["pair"], x=xs, y=ys, mode="lines+markers",
                                 line={"color": colors[bot["pair"]], "width": 2}), row=3, col=1)
    fig.update_layout(height=1120, barmode="group", margin={"l": 45, "r": 30, "t": 80, "b": 45},
                      paper_bgcolor="#ffffff", plot_bgcolor="#ffffff", font={"family": "Arial, sans-serif", "color": "#172033"},
                      legend={"orientation": "h", "y": -0.05})
    fig.update_yaxes(gridcolor="#e5e7eb", zerolinecolor="#64748b")
    return fig.to_html(full_html=False, include_plotlyjs=True, config={"displaylogo": False, "responsive": True})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", default="results/oci_dca_report/snapshot.json")
    parser.add_argument("--output-dir", default="results/oci_dca_report")
    args = parser.parse_args()
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    data = json.loads(Path(args.snapshot).read_text(encoding="utf-8"))
    data_status = data.get("data_status", "live_read_only_snapshot")
    source_as_of = data.get("source_as_of", data["generated_at"])

    bot_rows, side_rows, close_rows, level_rows, daily_rows, executor_rows, fill_rows = [], [], [], [], [], [], []
    total_pnl = total_fees = 0.0; total_trades = 0
    for bot in data["bots"]:
        fs = bot["fill_summary"]; sides, closes, levels = aggregate_executors(bot)
        executor_available = bool(bot["executors"])
        realized = sum(float(x.get("net_pnl_quote") or 0) for x in bot["executors"] if not x.get("is_active")) if executor_available else None
        active_pnl = sum(float(x.get("net_pnl_quote") or 0) for x in bot["executors"] if x.get("is_active")) if executor_available else None
        row = {"pair": bot["pair"], "bot": bot["bot"], "first_fill": ts_text(fs["first_fill"]), "last_fill": ts_text(fs["last_fill"]),
               "trades": fs["trades"], "buys": fs["buys"], "sells": fs["sells"], "fees_usdt": fs["fees"],
               "buy_quote": fs["buy_quote"], "sell_quote": fs["sell_quote"], "net_base": fs["net_base"], "mark_price": fs["mark_price"],
               "flow_mtm_pnl_usdt": fs["flow_mtm_pnl"], "closed_executor_pnl_usdt": realized,
               "active_executor_pnl_usdt": active_pnl, "executors": len(bot["executors"]),
               "active_executors": sum(bool(x.get("is_active")) for x in bot["executors"])}
        bot_rows.append(row); total_pnl += fs["flow_mtm_pnl"]; total_fees += fs["fees"]; total_trades += fs["trades"]
        for name, value in sides.items(): side_rows.append({"pair": bot["pair"], "side": name, **value})
        for name, value in closes.items(): close_rows.append({"pair": bot["pair"], "close_type": name, **value})
        for name, value in levels.items(): level_rows.append({"pair": bot["pair"], "level": name, **value})
        for row2 in bot["daily_fills"]: daily_rows.append({"pair": bot["pair"], **row2})
        for row2 in bot["executors"]: executor_rows.append({"pair": bot["pair"], **row2})
        for row2 in bot["fills"]: fill_rows.append({"pair": bot["pair"], "time": ts_text(row2["timestamp"]), **row2})
    write_csv(output / "bot_summary.csv", bot_rows); write_csv(output / "side_summary.csv", side_rows)
    write_csv(output / "close_type_summary.csv", close_rows); write_csv(output / "level_summary.csv", level_rows)
    write_csv(output / "daily_fills.csv", daily_rows); write_csv(output / "executors_latest.csv", executor_rows)
    write_csv(output / "trade_fills.csv", fill_rows)

    cards = "".join(f'<div class="metric"><span>{escape(k)}</span><strong>{escape(v)}</strong></div>' for k, v in [
        ("组合市值口径收益", f"{total_pnl:+,.2f} USDT"), ("逐笔成交", f"{total_trades:,}"),
        ("记录费用", f"{total_fees:,.2f} USDT"), ("数据时间", ts_text(max((b["fill_summary"]["last_fill"] or 0) for b in data["bots"])))])
    summary_tr = "".join("<tr>" + "".join(f"<td>{escape(str(v))}</td>" for v in [r["pair"], r["trades"], f'{r["buys"]}/{r["sells"]}',
        f'{r["fees_usdt"]:.4f}', f'{r["net_base"]:.8f}', f'{r["mark_price"]:.4f}', f'{r["flow_mtm_pnl_usdt"]:+.4f}',
        (f'{r["closed_executor_pnl_usdt"]:+.4f}' if r["closed_executor_pnl_usdt"] is not None else "未读取"),
        r["executors"], r["active_executors"]]) + "</tr>" for r in bot_rows)
    def detail_table(rows, key):
        return "".join("<tr>"+"".join(f"<td>{escape(str(v))}</td>" for v in [r["pair"], r[key], r["executors"], r["filled"], f'{r["volume"]:.2f}', f'{r["fees"]:.4f}', f'{r["pnl"]:+.4f}'])+"</tr>" for r in rows)
    status_note = ("<div class=\"callout\"><strong>数据状态：</strong>当前 TradeFill 只读快照。为避免再次压满服务器 I/O，未扫描约 4 GB 的重复执行器历史；止盈、止损、超时和层级收益暂不展示。</div>" if data_status == "live_tradefill_snapshot" else
                   f'<div class="callout"><strong>数据状态：</strong>最近一次成功读取的汇总快照（{escape(source_as_of)}），不是实时值。</div>')
    unavailable = '<tr><td colspan="7" style="text-align:left;color:#64748b">未读取：完整去重扫描会对线上 SQLite 造成不可接受的 I/O 压力。</td></tr>'
    html = f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>DCA Bot 收益详细报告</title><style>
:root{{--ink:#172033;--muted:#64748b;--line:#dce3ea;--bg:#f5f7fa;--blue:#2563eb;--gold:#d97706}}*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font-family:Arial,"Microsoft YaHei",sans-serif}}main{{max-width:1280px;margin:auto;padding:34px 22px 70px}}h1{{font-size:34px;margin:0 0 8px;letter-spacing:0}}h2{{margin:34px 0 12px;font-size:23px;letter-spacing:0}}p{{line-height:1.75;color:#334155}}.muted{{color:var(--muted)}}.metrics{{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:12px;margin:22px 0}}.metric{{background:white;border:1px solid var(--line);border-radius:7px;padding:17px}}.metric span{{display:block;color:var(--muted);font-size:13px;margin-bottom:8px}}.metric strong{{font-size:24px}}.callout{{border-left:4px solid var(--gold);background:#fff7ed;padding:14px 16px;margin:16px 0}}.panel{{background:white;border:1px solid var(--line);border-radius:7px;padding:18px;overflow:auto}}table{{width:100%;border-collapse:collapse;font-size:13px}}th,td{{padding:10px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}}th:first-child,td:first-child,th:nth-child(2),td:nth-child(2){{text-align:left}}th{{background:#f8fafc;color:#475569}}code{{background:#eef2f7;padding:2px 5px;border-radius:3px}}ul{{line-height:1.8}}@media(max-width:700px){{main{{padding:22px 12px}}h1{{font-size:27px}}.metric strong{{font-size:20px}}}}
</style></head><body><main><h1>DCA Bot 收益详细报告</h1><p class="muted">OCI Binance paper trading | 生成时间 {escape(data["generated_at"])} | BTC-USDT 与 ETH-USDT</p>
<h2>技术摘要</h2><div class="metrics">{cards}</div>{status_note}
<p><strong>组合的逐笔成交市值口径收益为 {total_pnl:+,.2f} USDT。</strong>该数值把成交造成的净基础币变化按报告时 Binance 现价估值，并扣除 TradeFill 记录费用，适合回答“策略成交相对启动库存创造了多少增量价值”。它不是 paper 钱包总余额变化。</p>
<div class="callout"><strong>重要口径：</strong>市值收益 = 卖出收入 - 买入成本 + 净基础币变化 × 当前价格 - 已记录费用。它包含未平库存的浮动价值，不等于已实现利润，也不等于 paper 钱包总权益变化。</div>
<h2>BTC 与 ETH 的收益、库存和费用</h2><p>净基础币为正表示策略相对启动时增持，负值表示相对启动库存净卖出。因 paper 账户预置了基础币，负库存变化不是裸做空。</p>
<div class="panel"><table><thead><tr><th>交易对</th><th>成交数</th><th>买/卖</th><th>费用 USDT</th><th>净基础币</th><th>估值价格</th><th>市值收益</th><th>已关闭执行器收益</th><th>执行器</th><th>活动</th></tr></thead><tbody>{summary_tr}</tbody></table></div>
<h2>收益结构可视化</h2><p>前四图用于比较逐币市值收益、净库存敞口、买卖成交额和费用；底部现金流曲线不含每天库存重估，因此不能单独视为每日净值曲线。</p><div class="panel">{chart_html(data)}</div>
<h2>买侧与卖侧</h2><p>DCA 是双边现货做市：买侧在价格下方逐层买入后退出，卖侧使用 paper 账户已有基础币在价格上方逐层卖出。侧别收益以去重后的执行器最终状态为准。</p>
<div class="panel"><table><thead><tr><th>交易对</th><th>方向</th><th>执行器</th><th>有成交</th><th>成交额</th><th>费用</th><th>净收益</th></tr></thead><tbody>{detail_table(side_rows,"side") if side_rows else unavailable}</tbody></table></div>
<h2>退出原因</h2><p><code>TAKE_PROFIT</code> 为达到 2% 止盈，<code>STOP_LOSS</code> 为达到 5% 止损，<code>TIME_LIMIT</code> 为 45 分钟期限退出。大量未成交超时执行器会增加计数，但通常不产生收益。</p>
<div class="panel"><table><thead><tr><th>交易对</th><th>退出原因</th><th>执行器</th><th>有成交</th><th>成交额</th><th>费用</th><th>净收益</th></tr></thead><tbody>{detail_table(close_rows,"close_type") if close_rows else unavailable}</tbody></table></div>
<h2>四层 DCA 的贡献</h2><p>层级对应距参考价约 1%、2%、4%、8%，资金权重约 10%、20%、30%、40%。更深层成交较少但单次风险更集中。</p>
<div class="panel"><table><thead><tr><th>交易对</th><th>层级</th><th>执行器</th><th>有成交</th><th>成交额</th><th>费用</th><th>净收益</th></tr></thead><tbody>{detail_table(level_rows,"level") if level_rows else unavailable}</tbody></table></div>
<h2>风险与结论</h2><ul><li>当前为 paper trading，初始虚拟 BTC/ETH 库存很大，不能直接外推到小额实盘。</li><li>市值口径对报告时价格敏感；净库存绝对值越大，短期收益波动越受方向行情影响。</li><li>费用来自 TradeFill 的记录值；若 paper 与真实账户费率不同，实盘净收益会变化。</li><li>执行器数据库异常膨胀至约 2 GB/实例，原因是周期快照重复写入；本版没有扫描该表，应另行治理保留策略后再补退出原因统计。</li></ul>
<p><strong>建议：</strong>先观察买卖侧和 STOP_LOSS/TIME_LIMIT 的净贡献。若某一侧持续亏损，应调整该侧层距、时限或暂时关闭，而不是只看组合正收益。</p>
<h2>可审计文件</h2><p class="muted">同目录包含 bot 汇总、每日成交、474 笔逐笔成交 CSV 和抓取时的 snapshot.json。线上数据库仅只读访问，bot 未被修改或重启。</p>
</main></body></html>'''
    (output / "dca_profit_report.html").write_text(html, encoding="utf-8")
    md = f"# DCA Bot 收益摘要\n\n- 报告时间：{data['generated_at']}\n- 组合市值口径收益：{total_pnl:+,.4f} USDT\n- 成交：{total_trades}\n- 费用：{total_fees:,.4f} USDT\n\n" + "\n".join(f"- {r['pair']}: {r['flow_mtm_pnl_usdt']:+,.4f} USDT，成交 {r['trades']}，费用 {r['fees_usdt']:.4f}" for r in bot_rows)
    (output / "result.md").write_text(md + "\n", encoding="utf-8")
    print(output / "dca_profit_report.html")


if __name__ == "__main__":
    main()

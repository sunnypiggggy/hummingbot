#!/usr/bin/env python3
"""Build a detailed, self-contained report from an OCI grid snapshot."""

from __future__ import annotations

import argparse
import html
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def local_time(timestamp_ms: int | float) -> str:
    return datetime.fromtimestamp(float(timestamp_ms) / 1000, timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, default=Path("results/oci_grid_report/snapshot.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/oci_grid_report"))
    parser.add_argument("--loss-cutoff", type=float, default=-500.0)
    args = parser.parse_args()
    payload = json.loads(args.snapshot.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    pair_rows, daily_rows, bot_rows = [], [], []
    for bot_name, bot in payload["bots"].items():
        runtime = bot["runtime"]
        drawdown = (runtime["current_equity"] - runtime["peak_equity"]) / runtime["peak_equity"]
        bot_rows.append({
            "bot": bot_name, "pnl": bot["pnl"], "fees": bot["fees"], "trades": bot["trades"],
            "current_equity": runtime["current_equity"], "peak_equity": runtime["peak_equity"],
            "drawdown_from_peak": drawdown, "liquidated": runtime["liquidated"],
            "first_fill": local_time(bot["first_ts"]), "last_fill": local_time(bot["last_ts"]),
        })
        for pair in bot["pairs"]:
            pair_rows.append({
                "bot": bot_name, "pair": pair["pair"], "pnl": pair["pnl"], "fees": pair["fees"],
                "trades": pair["trades"], "buys": pair["buys"], "sells": pair["sells"],
                "buy_notional": pair["buy_notional"], "sell_notional": pair["sell_notional"],
                "net_inventory": pair["inventory"], "inventory_value": pair["inventory"] * pair["mark"],
                "mark": pair["mark"], "first_fill": local_time(pair["first_ts"]),
                "last_fill": local_time(pair["last_ts"]),
            })
            for daily in pair["daily"]:
                daily_rows.append({"bot": bot_name, "pair": pair["pair"], **daily})
    bots, pairs, daily = pd.DataFrame(bot_rows), pd.DataFrame(pair_rows), pd.DataFrame(daily_rows)
    pairs["remove_candidate"] = pairs.pnl < args.loss_cutoff
    pairs["pnl_share_pct"] = pairs.pnl / pairs.pnl.abs().sum()
    pairs.to_csv(args.output_dir / "pair_metrics.csv", index=False)
    daily.to_csv(args.output_dir / "daily_mark_attribution.csv", index=False)
    bots.to_csv(args.output_dir / "bot_summary.csv", index=False)

    sorted_pairs = pairs.sort_values("pnl")
    colors = ["#b42318" if value < 0 else "#067647" for value in sorted_pairs.pnl]
    fig = make_subplots(rows=2, cols=2,
                        subplot_titles=("逐交易对净贡献", "费用与成交数", "每日成交批次按当前价格归因", "当前净库存敞口"),
                        specs=[[{}, {"secondary_y": True}], [{}, {}]], vertical_spacing=0.15)
    fig.add_trace(go.Bar(x=sorted_pairs.pair, y=sorted_pairs.pnl, marker_color=colors,
                         text=[f"{value:+,.0f}" for value in sorted_pairs.pnl], textposition="outside",
                         name="净贡献"), row=1, col=1)
    fig.add_trace(go.Bar(x=pairs.pair, y=pairs.fees, marker_color="#d97706", name="手续费"), row=1, col=2)
    fig.add_trace(go.Scatter(x=pairs.pair, y=pairs.trades, mode="lines+markers", line_color="#2563eb",
                             name="成交数"), row=1, col=2, secondary_y=True)
    daily_group = daily.groupby(["date", "bot"], as_index=False).pnl_at_current_mark.sum()
    for bot_name, frame in daily_group.groupby("bot"):
        fig.add_trace(go.Bar(x=frame.date, y=frame.pnl_at_current_mark, name=f"{bot_name} 日归因"), row=2, col=1)
    exposure_colors = ["#7c3aed" if value >= 0 else "#db2777" for value in pairs.inventory_value]
    fig.add_trace(go.Bar(x=pairs.pair, y=pairs.inventory_value, marker_color=exposure_colors,
                         name="库存市值"), row=2, col=2)
    fig.update_layout(height=900, template="plotly_white", barmode="relative", showlegend=True,
                      margin=dict(l=40, r=30, t=70, b=50), font=dict(family="Arial", size=12))
    chart = fig.to_html(full_html=False, include_plotlyjs=True, config={"displaylogo": False, "responsive": True})

    total_pnl, total_fees, total_trades = pairs.pnl.sum(), pairs.fees.sum(), int(pairs.trades.sum())
    removed = pairs[pairs.remove_candidate].sort_values("pnl")
    retained = pairs[~pairs.remove_candidate]
    hindsight_pnl = retained.pnl.sum()
    removal_names = ", ".join(removed.pair.tolist())
    bot_table = "".join(
        f"<tr><td>{html.escape(row.bot)}</td><td class='{('loss' if row.pnl < 0 else 'profit')}'>{row.pnl:+,.2f}</td>"
        f"<td>{row.fees:,.2f}</td><td>{int(row.trades):,}</td><td>{row.current_equity:,.2f}</td>"
        f"<td>{row.drawdown_from_peak:.2%}</td><td>{row.first_fill}</td><td>{row.last_fill}</td></tr>"
        for row in bots.itertuples(index=False)
    )
    pair_table = "".join(
        f"<tr><td>{html.escape(row.bot)}</td><td>{row.pair}</td><td class='{('loss' if row.pnl < 0 else 'profit')}'>{row.pnl:+,.2f}</td>"
        f"<td>{row.fees:,.2f}</td><td>{int(row.trades)}</td><td>{int(row.buys)}/{int(row.sells)}</td>"
        f"<td>{row.inventory_value:+,.2f}</td><td>{row.buy_notional:,.0f}</td><td>{row.sell_notional:,.0f}</td>"
        f"<td>{'建议剔除' if row.remove_candidate else '保留观察'}</td></tr>"
        for row in pairs.sort_values(["bot", "pnl"]).itertuples(index=False)
    )
    generated = payload.get("generated_at", "unknown")
    document = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>OCI Grid 详细收益报告</title>
<style>body{{font-family:Arial,sans-serif;background:#f5f7f9;color:#17202a;margin:0}}main{{max-width:1500px;margin:auto;padding:24px}}
h1,h2{{letter-spacing:0}}.note{{background:#fff7ed;border-left:5px solid #d97706;padding:14px;margin:14px 0}}
.metrics{{display:grid;grid-template-columns:repeat(4,minmax(150px,1fr));gap:10px}}.metric{{background:white;border:1px solid #d8dee5;border-radius:6px;padding:13px}}
.metric span{{display:block;color:#667085;font-size:13px}}.metric strong{{font-size:22px}}.loss{{color:#b42318;font-weight:700}}.profit{{color:#067647;font-weight:700}}
.panel{{background:white;border:1px solid #d8dee5;margin:18px 0;padding:14px;overflow:auto}}table{{border-collapse:collapse;width:100%;font-size:13px}}
th,td{{border-bottom:1px solid #e6e9ed;padding:8px;text-align:right;white-space:nowrap}}th:first-child,td:first-child,th:nth-child(2),td:nth-child(2){{text-align:left}}th{{background:#eef2f5}}
@media(max-width:760px){{main{{padding:12px}}.metrics{{grid-template-columns:1fr 1fr}}}}</style></head><body><main>
<h1>OCI Grid 详细收益报告</h1><p>快照：{html.escape(generated)}；口径：卖出现金流 - 买入现金流 + 净库存按实时价格估值 - 已记录手续费。</p>
<div class="metrics"><div class="metric"><span>两套 Grid 净贡献</span><strong class="loss">{total_pnl:+,.2f}</strong></div>
<div class="metric"><span>成交数</span><strong>{total_trades:,}</strong></div><div class="metric"><span>手续费</span><strong>{total_fees:,.2f}</strong></div>
<div class="metric"><span>亏损超过 500 的交易对</span><strong>{len(removed)}</strong></div></div>
<div class="note"><strong>结论：</strong>主要亏损来源为 {html.escape(removal_names)}。若从一开始静态排除这些交易对，按同一批历史成交做机械归因，保留组贡献为 <strong>{hindsight_pnl:+,.2f}</strong>；这是事后筛选，不代表未来可复制收益。</div>
<section class="panel"><h2>Bot 总览</h2><table><thead><tr><th>Bot</th><th>净贡献</th><th>费用</th><th>成交</th><th>当前权益</th><th>峰值回撤</th><th>首笔</th><th>末笔</th></tr></thead><tbody>{bot_table}</tbody></table></section>
<section class="panel"><h2>图表分析</h2>{chart}</section>
<section class="panel"><h2>逐交易对明细</h2><table><thead><tr><th>Bot</th><th>交易对</th><th>净贡献</th><th>费用</th><th>成交</th><th>买/卖</th><th>净库存市值</th><th>买入额</th><th>卖出额</th><th>判断</th></tr></thead><tbody>{pair_table}</tbody></table></section>
<div class="note"><strong>重要限制：</strong>这是 paper bot 相对于原始虚拟持仓的成交增量归因，不等于 Binance 钱包余额收益。负库存表示策略卖出了初始虚拟基础币，并不代表现货借币做空。</div>
</main></body></html>"""
    (args.output_dir / "grid_profit_report.html").write_text(document, encoding="utf-8")
    (args.output_dir / "grid_profit_report.md").write_text(
        f"# OCI Grid Profit Report\n\n- total pnl: {total_pnl:+.2f}\n- fees: {total_fees:.2f}\n"
        f"- trades: {total_trades}\n- removal candidates: {removal_names}\n"
        f"- retained historical attribution: {hindsight_pnl:+.2f}\n", encoding="utf-8")
    print(json.dumps({"total_pnl": total_pnl, "fees": total_fees, "trades": total_trades,
                      "removal_candidates": removed[["bot", "pair", "pnl"]].to_dict("records"),
                      "retained_hindsight_pnl": hindsight_pnl}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

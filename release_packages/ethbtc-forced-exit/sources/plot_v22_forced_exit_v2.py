#!/usr/bin/env python3
"""Self-contained UTF-8 Plotly report for the forced-exit-v2 audit."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


MECHANISMS = (
    "v22_weekly_buy_gate", "fomc_gate", "strategy_loss_breaker",
    "strategy_drawdown_breaker", "portfolio_loss_breaker",
    "portfolio_drawdown_breaker", "position_protection",
)
LABELS = {
    "v22_weekly_buy_gate": "v22 周度 Risk-Off",
    "fomc_gate": "FOMC 宏观风控（无数据）",
    "strategy_loss_breaker": "单策略绝对亏损",
    "strategy_drawdown_breaker": "单策略峰值回撤",
    "portfolio_loss_breaker": "组合绝对亏损",
    "portfolio_drawdown_breaker": "组合峰值回撤",
    "position_protection": "持仓保护",
}
STYLES = {
    "v22_weekly_buy_gate": ("rgba(217,119,6,.16)", "#92400e", "triangle-down"),
    "fomc_gate": ("rgba(124,58,237,.12)", "#5b21b6", "diamond"),
    "strategy_loss_breaker": ("rgba(190,24,93,.12)", "#9d174d", "x"),
    "strategy_drawdown_breaker": ("rgba(234,88,12,.12)", "#9a3412", "cross"),
    "portfolio_loss_breaker": ("rgba(71,85,105,.12)", "#334155", "square-x"),
    "portfolio_drawdown_breaker": ("rgba(30,64,175,.10)", "#1e3a8a", "star"),
    "position_protection": ("rgba(5,150,105,.10)", "#065f46", "circle-open"),
}
PAIRS = {"grid": ("BTC-FDUSD", "ETH-FDUSD"), "dca": ("BTC-USDT", "ETH-USDT")}


def _utc(values):
    return pd.to_datetime(values, unit="s", utc=True)


def _figure(strategy: str, series: pd.DataFrame, intervals: pd.DataFrame):
    titles = []
    currency, initial = ("FDUSD", 200) if strategy == "grid" else ("USDT", 190)
    for pair in PAIRS[strategy]:
        titles += [f"{pair} 价格与执行阶段", f"{pair} v22 周概率与 fold-local 阈值",
                   f"{pair} 机器人连续权益（初始 {initial} {currency}）"]
    fig = make_subplots(rows=6, cols=1, shared_xaxes=True, vertical_spacing=.025,
                        row_heights=[.22,.12,.16,.22,.12,.16],
                        specs=[[{}],[{}],[{"secondary_y":True}],[{}],[{}],[{"secondary_y":True}]],
                        subplot_titles=titles)
    groups = {mechanism: {"shapes": [], "traces": []} for mechanism in MECHANISMS}
    shapes, legend_seen = [], set()
    axis_refs = {1:("x","y domain"),3:("x3","y3 domain"),4:("x4","y5 domain"),6:("x6","y7 domain")}
    for pair_index, pair in enumerate(PAIRS[strategy]):
        price_row, prob_row, equity_row = 1+pair_index*3, 2+pair_index*3, 3+pair_index*3
        full_data = series[(series.strategy == strategy)&(series.pair == pair)].sort_values("timestamp")
        # Keep the audit CSV at 5-minute resolution, but embed hourly display
        # points only.  The old four-panel 5-minute payload was ~80 MB and
        # could freeze the browser during JSON parsing and initial rendering.
        data = (full_data.assign(datetime=_utc(full_data.timestamp)).set_index("datetime")
                .resample("1h").last().dropna(subset=["price", "equity"]).reset_index(drop=True))
        fig.add_trace(go.Scattergl(x=_utc(data.timestamp), y=data.price, name=f"{pair} 价格",
                                 line={"color":"#1e40af","width":1.1}), row=price_row, col=1)
        probability = data[data.probability.notna()]
        fig.add_trace(go.Scattergl(x=_utc(probability.timestamp), y=probability.probability,
                                 name=f"{pair} 周概率", line={"color":"#d97706","width":1.1}), row=prob_row,col=1)
        fig.add_trace(go.Scattergl(x=_utc(probability.timestamp), y=probability.entry_threshold,
                                 name=f"{pair} 阈值", line={"color":"#334155","width":1,"dash":"dot"}), row=prob_row,col=1)
        fig.add_trace(go.Scattergl(x=_utc(data.timestamp), y=data.equity, name=f"{pair} 机器人权益",
                                 line={"color":"#1e40af","width":1.3}), row=equity_row,col=1,secondary_y=False)
        fig.add_trace(go.Scattergl(x=_utc(data.timestamp), y=data.peak_equity, name=f"{pair} 权益峰值",
                                 line={"color":"#64748b","width":1,"dash":"dash"}), row=equity_row,col=1,secondary_y=False)
        fig.add_trace(go.Scattergl(x=_utc(data.timestamp), y=data.drawdown_pct, name=f"{pair} 回撤 %",
                                 line={"color":"#be185d","width":1,"dash":"dot"}), row=equity_row,col=1,secondary_y=True)
        selected = intervals[(intervals.strategy == strategy)&intervals.pair.isin((pair,"ALL"))]
        for mechanism in MECHANISMS:
            fill, line, marker = STYLES[mechanism]; xs, ys, hover = [], [], []
            for event in selected[selected.mechanism == mechanism].itertuples(index=False):
                dash = {"EXITING":"solid","COOLDOWN":"dash","REENTRY":"dot"}.get(str(event.phase).upper(),"dashdot")
                for panel in (price_row,equity_row):
                    xref,yref=axis_refs[panel]
                    shapes.append({"type":"rect","xref":xref,"yref":yref,"x0":_utc(event.start_ts),"x1":_utc(event.end_ts),
                                   "y0":0,"y1":1,"fillcolor":fill,"line":{"color":line,"width":1,"dash":dash},"layer":"below"})
                    groups[mechanism]["shapes"].append({"index":len(shapes)-1,"pair":pair})
                near=(full_data.timestamp.astype(float)-float(event.start_ts)).abs().idxmin()
                xs.append(_utc(event.start_ts)); ys.append(float(full_data.loc[near,"price"]))
                hover.append(f"{LABELS[mechanism]}<br>阶段: {event.phase}<br>动作: {event.action}<br>开始: {_utc(event.start_ts)}<br>结束: {_utc(event.end_ts)}<br>来源: {event.source}<br>执行开关: {event.enabled}")
            if xs:
                fig.add_trace(go.Scatter(x=xs,y=ys,text=hover,mode="markers",name=LABELS[mechanism],
                                         legendgroup=mechanism,showlegend=mechanism not in legend_seen,
                                         marker={"symbol":marker,"size":9,"color":line},hovertemplate="%{text}<extra></extra>"),row=price_row,col=1)
                groups[mechanism]["traces"].append({"index":len(fig.data)-1,"pair":pair}); legend_seen.add(mechanism)
    fig.update_layout(template="plotly_white",height=1760,hovermode="x unified",shapes=shapes,
                      title=f"{'FDUSD Grid' if strategy=='grid' else 'USDT DCA'} · v22信号+强制退出执行覆盖层 · weekly walk-forward · offline validation · NO-GO",
                      legend={"orientation":"h","y":1.02},font={"family":"Arial, Microsoft YaHei, sans-serif","color":"#172033"})
    for row in (3,6):
        fig.update_yaxes(title_text=f"机器人权益 ({currency})",row=row,col=1,secondary_y=False)
        fig.update_yaxes(title_text="回撤 %",row=row,col=1,secondary_y=True)
    return fig, groups


def _ablation(metrics: pd.DataFrame) -> str:
    combined = metrics[metrics.pair.eq("ALL")].copy()
    combined["label"] = combined.strategy.str.upper()+" · "+combined.scenario
    fig=make_subplots(rows=1,cols=2,subplot_titles=("净收益（报价币）","最大回撤（%）"))
    fig.add_trace(go.Bar(x=combined.label,y=combined.net_pnl_quote,name="净收益",marker_color="#1e40af"),row=1,col=1)
    fig.add_trace(go.Bar(x=combined.label,y=combined.max_drawdown_pct,name="最大回撤",marker_color="#be185d"),row=1,col=2)
    fig.update_layout(template="plotly_white",height=470,title="legacy BUY-only 与 forced-exit-v2 消融对照",showlegend=False,
                      font={"family":"Arial, Microsoft YaHei, sans-serif"})
    fig.update_xaxes(tickangle=-15)
    return fig.to_html(full_html=False,include_plotlyjs=False,div_id="forced-exit-ablation",config={"responsive":True,"displaylogo":False})


def render_dashboard(series: pd.DataFrame, intervals: pd.DataFrame, metrics: pd.DataFrame, output: Path) -> dict:
    figs, groups = {}, {}
    for strategy in ("grid","dca"):
        figs[strategy],groups[strategy]=_figure(strategy,series,intervals)
    chart={s:f.to_html(full_html=False,include_plotlyjs=(s=="grid"),div_id=f"{s}-forced-exit-chart",config={"responsive":True,"displaylogo":False}) for s,f in figs.items()}
    mechanism_controls="".join(f"<label><input type='checkbox' data-mechanism='{m}' checked>{LABELS[m]}</label>" for m in MECHANISMS)
    pair_controls="".join(f"<label><input type='checkbox' data-pair='{p}' checked>{p} 机器人阴影</label>" for p in (*PAIRS['grid'],*PAIRS['dca']))
    encoded=json.dumps(groups,ensure_ascii=False)
    html=f"""<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>
<title>v22 forced-exit-v2 离线审计</title><style>body{{margin:0;background:#f8fafc;color:#172033;font:14px Arial,'Microsoft YaHei',sans-serif}}main{{max-width:1500px;margin:auto;padding:22px}}.warning{{background:#fff7ed;border-left:5px solid #d97706;padding:14px}}.controls,.tabs{{display:flex;gap:9px;flex-wrap:wrap;margin:13px 0}}label,button{{background:#fff;border:1px solid #cbd5e1;border-radius:7px;padding:8px 11px;cursor:pointer}}button.active{{border-color:#1e40af;color:#1e40af}}.panel{{display:none;background:#fff}}.panel.active{{display:block}}</style></head><body><main>
<h1>v22 Risk-Off 强制退出与连续权益修正</h1><p class='warning'><strong>NO-GO：</strong>这是“冻结 v22 信号 + forced-exit-v2 执行覆盖层”的离线反事实。offline_only=true，deployment_allowed=false，不是原冻结 v22 精确回放，也不授予实盘权限。</p>
<p>BTC 与 ETH 分别展示各机器人连续权益，不再重复组合权益。审计数据保持 5 分钟精度；图表展示按小时取样，以避免浏览器解析超大 Plotly 数据时卡死。退出和重入标记仍保留 5 分钟精度。FOMC 无可信历史数据，因此明确显示“无数据”。</p>{_ablation(metrics)}
<div class='tabs'><button data-tab='grid' class='active'>FDUSD Grid</button><button data-tab='dca'>USDT DCA</button></div><div class='controls'>{mechanism_controls}</div><div class='controls'>{pair_controls}</div>
<section id='grid-panel' class='panel active'>{chart['grid']}</section><section id='dca-panel' class='panel'>{chart['dca']}</section>
<script>const groups={encoded};const state={{mechanisms:Object.fromEntries([...document.querySelectorAll('[data-mechanism]')].map(x=>[x.dataset.mechanism,true])),pairs:Object.fromEntries([...document.querySelectorAll('[data-pair]')].map(x=>[x.dataset.pair,true]))}};
function refresh(){{for(const strategy of ['grid','dca']){{const div=document.getElementById(strategy+'-forced-exit-chart'),patch={{}};for(const [m,g] of Object.entries(groups[strategy])){{for(const x of g.shapes||[])patch[`shapes[${{x.index}}].visible`]=state.mechanisms[m]!==false&&state.pairs[x.pair]!==false;for(const x of g.traces||[])Plotly.restyle(div,{{visible:state.mechanisms[m]!==false&&state.pairs[x.pair]!==false}},[x.index]);}}if(Object.keys(patch).length)Plotly.relayout(div,patch);}}}}
document.querySelectorAll('[data-tab]').forEach(b=>b.onclick=()=>{{document.querySelectorAll('[data-tab]').forEach(x=>x.classList.toggle('active',x===b));document.querySelectorAll('.panel').forEach(x=>x.classList.toggle('active',x.id===b.dataset.tab+'-panel'));window.dispatchEvent(new Event('resize'));}});document.querySelectorAll('[data-mechanism]').forEach(x=>x.onchange=()=>{{state.mechanisms[x.dataset.mechanism]=x.checked;refresh()}});document.querySelectorAll('[data-pair]').forEach(x=>x.onchange=()=>{{state.pairs[x.dataset.pair]=x.checked;refresh()}});</script></main></body></html>"""
    output.write_text(html,encoding="utf-8")
    return groups

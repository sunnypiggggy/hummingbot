"""Self-contained Plotly, four independent robot tabs and state-derived shadows."""
from __future__ import annotations

import html
import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs
from plotly.subplots import make_subplots

from prepare_v22_365d_inputs import START, END

COLORS={'without_v22':'#a36a12','with_v22':'#3268a8'}
LABELS={'without_v22':'不加 v22（其他保护相同）','with_v22':'v22 + 强制退出'}


def local_time(ts):
    return pd.Timestamp(int(ts),unit='s',tz='UTC').tz_convert('Asia/Shanghai').tz_localize(None).isoformat()


def risk_intervals(signals,pair):
    source=signals[signals.pair==pair].sort_values('signal_ts')
    intervals=[];begin=None;last=None
    for row in source.itertuples(index=False):
        ts=int(row.signal_ts)
        if last is not None and ts-last!=3600:
            if begin is not None:intervals.append((begin,last+3600));begin=None
        active=str(row.risk_off).lower()=='true'
        if active and begin is None:begin=ts
        if not active and begin is not None:intervals.append((begin,ts));begin=None
        last=ts
    if begin is not None:intervals.append((begin,min(END,last+3600)))
    return [(max(a,START),min(b,END)) for a,b in intervals if b>START and a<END]


def figure(curves,signals,summary,strategy,pair):
    quote='FDUSD' if strategy=='Grid' else 'USDT'
    source_pair=pair.replace('USDT','FDUSD')
    own=curves[(curves.strategy==strategy)&(curves.pair==pair)].copy()
    own['date']=pd.to_datetime(own.ts,unit='s',utc=True)
    hourly=own.set_index('date').groupby('scenario').resample('1h',include_groups=False).last().reset_index()
    hourly['date']=hourly.date.dt.tz_convert('Asia/Shanghai').dt.tz_localize(None)
    # Preserve the exact final point; only presentation is downsampled.
    for scenario in COLORS:
        last=own[own.scenario==scenario].iloc[-1]
        if not ((hourly.scenario==scenario)&(hourly.ts==last.ts)).any():
            record=last.to_dict();record['date']=pd.Timestamp(last.date).tz_convert('Asia/Shanghai').tz_localize(None)
            hourly=pd.concat([hourly,pd.DataFrame([record])],ignore_index=True)
    hourly=hourly.sort_values(['scenario','ts'])
    fig=make_subplots(rows=4,cols=1,shared_xaxes=True,vertical_spacing=.055,
                     subplot_titles=[f'{pair} 价格（{quote}）',f'单机器人连续权益（{quote}）','峰值回撤（%）','v22 概率与逐周阈值（%）'])
    price=hourly[hourly.scenario=='with_v22']
    fig.add_trace(go.Scatter(x=price.date,y=price.price,name='价格',line=dict(color='#555c65',width=1.2)),row=1,col=1)
    for scenario,color in COLORS.items():
        data=hourly[hourly.scenario==scenario]
        for row,field in [(2,'equity'),(3,'drawdown_pct')]:
            fig.add_trace(go.Scatter(x=data.date,y=data[field],name=LABELS[scenario],legendgroup=scenario,
                                    showlegend=row==2,line=dict(color=color,width=1.5,dash='solid' if scenario=='with_v22' else 'dash')),
                          row=row,col=1)
    st=signals[(signals.pair==source_pair)&(signals.signal_ts>=START)&(signals.signal_ts<END)].copy()
    st['date']=[local_time(ts) for ts in st.signal_ts]
    fig.add_trace(go.Scatter(x=st.date,y=st.probability*100,name='预测概率',line=dict(color='#3268a8',width=1.2)),row=4,col=1)
    fig.add_trace(go.Scatter(x=st.date,y=st.threshold*100,name='当周阈值',line=dict(color='#555c65',width=1.3,dash='dash',shape='hv')),row=4,col=1)
    for start,end in risk_intervals(signals,source_pair):
        for row in range(1,5):
            fig.add_vrect(x0=local_time(start),x1=local_time(end),fillcolor='#e4a244',opacity=.15,line_width=0,
                          layer='below',row=row,col=1)
    original_start=int(pd.Timestamp('2025-11-23T15:00:00Z').timestamp())
    for row in range(1,5):
        fig.add_vline(x=local_time(original_start),line=dict(color='#68717c',width=1,dash='dot'),row=row,col=1)
    protected=summary[(summary.strategy==strategy)&(summary.pair==pair)&(summary.scenario=='with_v22')].iloc[0]
    title=f'{strategy} {pair} · 365天离线回测'
    subtitle=f'v22收益 {protected.net_pnl_quote:+.4f} {quote}（{protected.return_pct:+.2f}%）｜最大回撤 {protected.max_drawdown_pct:.2f}%｜浅橙阴影：模型 Risk-Off'
    fig.update_layout(title=dict(text=title+'<br><sup>'+subtitle+'</sup>',font=dict(size=19)),
                      template='plotly_white',height=1250,margin=dict(l=75,r=35,t=110,b=65),
                      hovermode='x unified',font=dict(family='Arial, Microsoft YaHei, sans-serif',size=13),
                      legend=dict(orientation='h',y=1.02,x=0),uirevision=pair)
    fig.update_yaxes(range=[0,100],row=4,col=1)
    fig.update_xaxes(range=[local_time(START),local_time(END)],tickformat='%m-%d',showticklabels=True)
    fig.update_xaxes(title_text='北京时间',row=4,col=1)
    fig.update_yaxes(gridcolor='#e7e9ec',zerolinecolor='#ccd1d7')
    return fig


def render(output:Path,curves,signals,summary):
    specs=[];buttons=[]
    for index,(strategy,pair) in enumerate([('Grid','BTC-FDUSD'),('Grid','ETH-FDUSD'),('DCA','BTC-USDT'),('DCA','ETH-USDT')]):
        fig=figure(curves,signals,summary,strategy,pair)
        buttons.append(f'<button class="tab {"active" if index==0 else ""}" onclick="showTab({index},this)">{html.escape(strategy+" "+pair)}</button>')
        specs.append(f'<section id="tab-{index}" class="page {"active" if index==0 else ""}"><div id="plot-{index}"></div><script type="application/json" id="spec-{index}">{fig.to_json().replace("</","<\\/")}</script></section>')
    rows=[]
    for _,row in summary.iterrows():
        rows.append('<tr>'+''.join(f'<td>{html.escape(str(x))}</td>' for x in [row.strategy,row.pair,LABELS[row.scenario],
            f'{row.net_pnl_quote:+.4f} {row.quote_currency}',f'{row.return_pct:+.2f}%',f'{row.max_drawdown_pct:.2f}%',row.fills,
            f'{row.fees_quote:.4f}',f'{row.risk_off_hours:.1f}h'])+'</tr>')
    content='''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>v22 · 365天回测</title><style>body{margin:0;background:#f6f7f9;color:#1f2937;font:15px Arial,'Microsoft YaHei',sans-serif}header{padding:18px 22px;background:#fff}h1{font-size:23px;margin:0 0 8px}p{margin:7px 0;line-height:1.6}nav{position:sticky;top:0;background:#fff;z-index:4;padding:10px;overflow-x:auto;white-space:nowrap;border-bottom:1px solid #ddd}button{padding:9px 12px;margin:3px;background:white;border:1px solid #ccd2da;border-radius:6px;cursor:pointer}button.active{background:#304f73;color:white}.page{display:none;background:white;margin:10px}.page.active{display:block}.table{overflow:auto;margin:15px;background:#fff}table{border-collapse:collapse;white-space:nowrap;width:100%}td,th{text-align:left;padding:9px;border-bottom:1px solid #e6e8ec}details{padding:12px 22px;background:white}.warn{color:#875c13}</style><script>''' + get_plotlyjs()+'''</script></head><body>
<header><h1>v22 · 过去365天离线回测</h1><p>UTC：2025-10-17 00:00 → 2026-10-17 00:00（末端不含）；图表横轴为北京时间。</p>
<p class="warn">2025-11-23 15:00 UTC之前采用本次离线补训周（图中竖虚线左侧），之后使用原47周模型。补训不等于历史实盘；本报告不授予交易权限。</p>
<p>四个机器人分别计价、资金与状态跨周连续。FOMC、参数优化和Momentum TP不参与。浅橙阴影仅表示模型Risk-Off，不等于所有其他风控阶段。</p></header><nav>'''+''.join(buttons)+'''
<span>｜</span><button onclick="setWindow('2025-10-17T08:00:00','2026-10-17T08:00:00')">365天</button><button onclick="setWindow('2026-01-01T08:00:00','2026-03-01T08:00:00')">2026年1–2月</button><button onclick="setWindow('2026-05-01T08:00:00','2026-07-01T08:00:00')">5–6月</button><button onclick="toggleRisk()">Risk-Off阴影 开/关</button></nav>'''+''.join(specs)+'''
<div class="table"><table><thead><tr><th>策略</th><th>交易对</th><th>口径</th><th>净MTM</th><th>收益率</th><th>最大回撤</th><th>成交</th><th>费用</th><th>Risk-Off时长</th></tr></thead><tbody>'''+''.join(rows)+'''</tbody></table></div>
<details><summary>回测边界与计算口径</summary><p>当前策略参数固定用于整个365天，不冒充当时实盘配置。Grid每对200 FDUSD、公共储备20 FDUSD；DCA每对190 USDT，95 USDT等值启动库存，普通执行器只做多。不同稳定币收益不相加。</p><p>Grid Maker费0%；DCA Maker费0.1%；Taker费0.1%、不利滑点2bp，不使用BNB。5分钟OHLC触价不能重建订单簿排队；健康恢复周期近似为3根5分钟K线。小时降采样只影响展示，统计使用完整5分钟数据。</p><p>特征与超参数为回溯固定方案，不能将离线补训视为当时已确定的前瞻决策。完整逐周模型审计、成交、权益及资产守恒核对见同目录文件。</p></details>
<script>const made=new Set();let active=0;function draw(i){if(made.has(i))return;let s=JSON.parse(document.getElementById('spec-'+i).textContent);Plotly.newPlot('plot-'+i,s.data,s.layout,{responsive:true,displaylogo:false});made.add(i)}function showTab(i,b){active=i;document.querySelectorAll('.page,.tab').forEach(x=>x.classList.remove('active'));document.getElementById('tab-'+i).classList.add('active');b.classList.add('active');draw(i);setTimeout(()=>Plotly.Plots.resize(document.getElementById('plot-'+i)),20)}function setWindow(a,b){let p=document.getElementById('plot-'+active),u={};for(let i=1;i<=4;i++)u[(i==1?'xaxis':'xaxis'+i)+'.range']=[a,b];Plotly.relayout(p,u)}function toggleRisk(){let p=document.getElementById('plot-'+active),u={};p.__riskHidden=!p.__riskHidden;(p.layout.shapes||[]).forEach((s,i)=>{if(s.type==='rect')u['shapes['+i+'].visible']=!p.__riskHidden});Plotly.relayout(p,u)}draw(0);</script></body></html>'''
    target=output/'v22_365d_plotly.html'
    target.write_text(content,encoding='utf-8')
    print('Plotly:',target,flush=True)

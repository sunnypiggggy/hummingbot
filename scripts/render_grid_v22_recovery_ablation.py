"""Requested self-contained Plotly + mobile PNG + Chinese experiment audit."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

COLORS = {"A": "#2563eb", "B": "#d97706"}
LABELS = {"A": "A 共用计数", "B": "B 独立计数"}


def intervals(signals, pair, arm, start, end):
    h = signals[(signals.pair == pair) & (signals.arm == arm)].sort_values("signal_ts")
    result, beginning, prev = [], None, None
    for row in h.itertuples():
        t = max(start, int(row.signal_ts))
        if row.signal_ts+3600 <= start or row.signal_ts >= end:
            continue
        if prev is not None and row.signal_ts != prev+3600 and beginning is not None:
            result.append((beginning, min(end, prev+3600)))
            beginning = None
        if row.risk_off and beginning is None:
            beginning = t
        if not row.risk_off and beginning is not None:
            result.append((beginning, t))
            beginning = None
        prev = int(row.signal_ts)
    if beginning is not None:
        result.append((beginning, min(end, prev+3600)))
    return result


def display_time(ts):
    # Plotly naive ISO is deliberate: axis explicitly says Beijing UTC+8.
    return pd.Timestamp(int(ts), unit="s", tz="UTC").tz_convert("Asia/Shanghai").tz_localize(None)


def values_at(frame, timestamp):
    ix = np.searchsorted(frame.ts.to_numpy(), timestamp, side="right")-1
    return 200. if ix < 0 else float(frame.equity.iloc[ix])


def divergences(signals, equities, start, end):
    records = []
    for pair in ("BTC-FDUSD", "ETH-FDUSD"):
        a = signals[(signals.pair == pair) & (signals.arm == "A")].set_index("signal_ts")
        b = signals[(signals.pair == pair) & (signals.arm == "B")].set_index("signal_ts")
        common = a.index[(a.index >= start) & (a.index < end)]
        mask = a.loc[common, "risk_off"].to_numpy() != b.loc[common, "risk_off"].to_numpy()
        changes = np.flatnonzero(np.r_[True, mask[1:] != mask[:-1], True])
        paths = {arm: equities[(equities.pair == pair) & (equities.arm == arm)].sort_values("ts") for arm in ("A", "B")}
        for left, right in zip(changes[:-1], changes[1:]):
            if not mask[left]:
                continue
            s, e = int(common[left]), min(end, int(common[right-1])+3600)
            ar, br = a.loc[s], b.loc[s]
            future_a = a[(a.index >= s) & (a.index < end) & (a.transition == "recover")]
            future_b = b[(b.index >= s) & (b.index < end) & (b.transition == "recover")]
            earlier_b = not bool(br.risk_off) and bool(ar.risk_off)
            counterpart = future_a if earlier_b else future_b
            gap = (int(counterpart.index[0])-s)/3600 if len(counterpart) else None
            after = min(end, s+48*3600)
            vals = {arm: values_at(paths[arm], after)-values_at(paths[arm], s) for arm in ("A", "B")}
            records.append(dict(pair=pair, start_utc=pd.Timestamp(s, unit="s", tz="UTC").isoformat(),
                                end_utc=pd.Timestamp(e, unit="s", tz="UTC").isoformat(),
                                divergence_hours=(e-s)/3600, a_risk_off=bool(ar.risk_off), b_risk_off=bool(br.risk_off),
                                a_ordinary_count=int(ar.ordinary_count), b_ordinary_count=int(br.ordinary_count),
                                b_strong_count=int(br.strong_count), ordinary_condition=bool(br.ordinary_condition),
                                strong_condition=bool(br.strong_condition), b_earlier=earlier_b,
                                next_other_recovery_hours=gap,
                                recovery_matching="next observed recovery, not causal episode pairing; censored if absent",
                                followup_hours=(after-s)/3600, a_next_48h_equity_change=vals["A"],
                                b_next_48h_equity_change=vals["B"], followup_increment=vals["B"]-vals["A"]))
    return pd.DataFrame(records)


def render(output: Path, report, signals, equities, end):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    import matplotlib.dates as mdates
    import plotly.graph_objects as go
    from plotly.offline.offline import get_plotlyjs
    from plotly.subplots import make_subplots

    font_paths = [Path("C:/Windows/Fonts/msyh.ttc"), Path("/mnt/c/Windows/Fonts/msyh.ttc"),
                  Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")]
    font = next((p for p in font_paths if p.exists()), None)
    if font is None:
        raise RuntimeError("Chinese font missing; refusing mojibake PNG")
    font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({"font.family": font_manager.FontProperties(fname=str(font)).get_name(),
                         "axes.unicode_minus": False, "font.size": 12})
    start = int(pd.Timestamp(report["start"]).timestamp())
    div = divergences(signals, equities, start, end)
    div.to_csv(output/"divergence_episodes.csv", index=False, encoding="utf-8", lineterminator="\n")
    signals[(signals.signal_ts >= int(pd.Timestamp("2026-09-01", tz="UTC").timestamp()))].to_csv(
        output/"september_diagnostic_signals.csv", index=False, lineterminator="\n")
    html_parts, group_map = [], {}
    for pair in ("BTC-FDUSD", "ETH-FDUSD", "PORTFOLIO"):
        paths = {}
        for arm in ("A", "B"):
            if pair == "PORTFOLIO":
                p = equities[equities.arm == arm].groupby("ts", as_index=False).equity.sum()
                p.equity += 20
                p["price"] = np.nan
                initial = 420
            else:
                p = equities[(equities.arm == arm) & (equities.pair == pair)].sort_values("ts").copy()
                initial = 200
            p["dd"] = (1-p.equity/np.maximum.accumulate(np.r_[initial, p.equity])[1:])*100
            # Exact statistics use every 5m point; plot keeps each completed hour and final point.
            p = p[(p.ts % 3600 == 0) | (p.ts == p.ts.max())].copy()
            p["time"] = p.ts.map(display_time)
            paths[arm] = p
        delta = paths["B"].equity.to_numpy()-paths["A"].equity.to_numpy()
        titles = ("价格（FDUSD）" if pair != "PORTFOLIO" else "组合总资金：420 FDUSD（含20储备）",
                  "单机器人连续权益（FDUSD）" if pair != "PORTFOLIO" else "独立组合图：BTC+ETH连续权益",
                  "从累计权益峰值回撤（%）", "B − A 权益差（FDUSD）")
        fig = make_subplots(rows=4, cols=1, shared_xaxes=True, subplot_titles=titles,
                            vertical_spacing=.055, row_heights=[.25,.3,.2,.25])
        if pair != "PORTFOLIO":
            fig.add_trace(go.Scatter(x=paths["A"].time, y=paths["A"].price, name="价格", line=dict(color="#4b5563")), row=1,col=1)
        for arm in ("A", "B"):
            p = paths[arm]
            for row, metric in ((2,"equity"),(3,"dd")):
                fig.add_trace(go.Scatter(x=p.time, y=p[metric], name=LABELS[arm], legendgroup=arm,
                                        showlegend=row==2, line=dict(color=COLORS[arm], dash="solid" if arm=="A" else "dash")),row=row,col=1)
        fig.add_trace(go.Scatter(x=paths["A"].time,y=delta,name="权益差 B−A",line=dict(color="#334155")),row=4,col=1)
        groups = {"A": [], "B": []}
        if pair != "PORTFOLIO":
            for arm in ("A", "B"):
                for s,e in intervals(signals,pair,arm,start,end):
                    for row in range(1,5):
                        groups[arm].append(len(fig.layout.shapes or []))
                        fig.add_vrect(x0=display_time(s+300), x1=display_time(min(end,e+300)), fillcolor=COLORS[arm],
                                      opacity=.09, line_width=1 if arm=="B" else 0, line_dash="dot",
                                      row=row,col=1,layer="below")
        fig.update_layout(height=1050,template="plotly_white",title=f"{pair} · 2026年内恢复计数对照 · 仅离线，不授权实盘",
                          hovermode="x unified", margin=dict(l=65,r=35,t=100,b=50))
        fig.update_xaxes(title_text="北京时间（UTC+8）",row=4,col=1,
                         rangeselector=dict(buttons=[dict(count=1,label="近1月",step="month",stepmode="backward"),dict(step="all",label="全部")] ))
        chart_id = "chart_"+pair.replace("-", "_")
        group_map[chart_id] = groups
        buttons = "" if pair == "PORTFOLIO" else ''.join(
            f'<label><input type="checkbox" checked onchange="shade(\'{chart_id}\',\'{a}\',this.checked)">{LABELS[a]} Risk-Off阴影</label> '
            for a in ("A","B"))
        html_parts.append(f'<section id="tab_{chart_id}" style="display:{"block" if pair=="BTC-FDUSD" else "none"}">{buttons}'+
                          fig.to_html(full_html=False, include_plotlyjs=False, div_id=chart_id)+"</section>")
        if pair != "PORTFOLIO":
            mobile, axes = plt.subplots(4,1,figsize=(12,20),dpi=120,sharex=True,
                                        gridspec_kw=dict(height_ratios=[1,1.35,.8,1],hspace=.15))
            axes[0].plot(paths["A"].time,paths["A"].price,color="#4b5563",linewidth=1)
            for arm in ("A","B"):
                p=paths[arm]
                for ax,metric in ((axes[1],"equity"),(axes[2],"dd")):
                    ax.plot(p.time,p[metric],label=LABELS[arm],color=COLORS[arm],linewidth=1.4,
                            linestyle="-" if arm=="A" else "--")
                for s,e in intervals(signals,pair,arm,start,end):
                    for ax in axes:
                        ax.axvspan(display_time(s+300),display_time(min(end,e+300)),color=COLORS[arm],alpha=.085)
            axes[3].plot(paths["A"].time,delta,color="#334155",linewidth=1.2)
            axes[3].axhline(0,color="#999",linewidth=.7)
            for ax,title in zip(axes,titles):
                ax.set_title(title,loc="left",fontsize=14)
                ax.grid(alpha=.15)
                ax.spines[["top","right"]].set_visible(False)
            axes[1].legend(loc="best")
            axes[3].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
            axes[3].set_xlabel("北京时间 UTC+8；阴影=该组可信 Risk-Off（执行延后5分钟）")
            a,b=report["results"]["A"][pair],report["results"]["B"][pair]
            mobile.suptitle(f"{pair} · v22恢复计数对照\n2026-01-01 至 {pd.Timestamp(end,unit='s').strftime('%Y-%m-%d')} UTC（截止日不含）\n"
                            f"A净收益 {a['pnl']:+.2f} / B {b['pnl']:+.2f} FDUSD\n"
                            f"A最大回撤 {a['max_drawdown_pct']:.2f}% / B {b['max_drawdown_pct']:.2f}%",
                            fontsize=17,y=.975)
            mobile.subplots_adjust(top=.87,bottom=.08,left=.12,right=.96)
            mobile.text(.06,.023,"历史模型离线反事实实验 · 固定现行参数 · 不改现网\n5分钟撮合近似；未模拟盘口排队；更早恢复不代表收益更高。",fontsize=11,color="#555")
            mobile.savefig(output/f"{pair}_comparison.png",dpi=120)
            plt.close(mobile)
    nav=''.join(f'<button onclick="tab(\'chart_{p.replace("-","_")}\')">{p}</button>' for p in ("BTC-FDUSD","ETH-FDUSD","PORTFOLIO"))
    html='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">'+\
         '<title>2026 v22恢复计数离线对照</title><style>body{font-family:system-ui;margin:18px;color:#1e293b}button,label{margin:8px;padding:8px}section{max-width:1500px}</style>'+\
         '<script>'+get_plotlyjs()+'</script><h2>共用计数 vs 独立计数 · 历史模型离线反事实</h2>'+\
         '<p>单机器人权益，不使用组合替代。小时抽样仅用于展示；统计使用全部5分钟记录。阴影开关不影响曲线。</p>'+nav+''.join(html_parts)+\
         '<script>const groups='+json.dumps(group_map)+';function shade(id,a,on){const u={};groups[id][a].forEach(i=>u["shapes["+i+"].visible"]=on);Plotly.relayout(id,u)}'+\
         'function tab(id){document.querySelectorAll("section").forEach(s=>s.style.display=s.id==="tab_"+id?"block":"none");Plotly.Plots.resize(document.getElementById(id))}</script></html>'
    (output/"comparison.html").write_text(html,encoding="utf-8")
    rows=["# Grid 2026年内：v22恢复计数离线对照", "", "本实验不修改现网，不生成审批或上线授权。", "",
          f"区间：{report['start']} 至 {report['end_exclusive']}（右开），{report['days']}天。",
          "历史周模型 + 现行固定参数的反事实，不是历史实盘收益复原。", "",
          "|范围|A净收益 FDUSD|B净收益 FDUSD|B−A|A最大回撤|B最大回撤|",
          "|---|---:|---:|---:|---:|---:|"]
    for p in ("BTC-FDUSD","ETH-FDUSD","PORTFOLIO"):
        a,b=report["results"]["A"][p],report["results"]["B"][p]
        rows.append(f"|{p}|{a['pnl']:+.4f}|{b['pnl']:+.4f}|{b['pnl']-a['pnl']:+.4f}|{a['max_drawdown_pct']:.3f}%|{b['max_drawdown_pct']:.3f}%|")
    rows += ["", "## 恢复行为", "", "|交易对/组|Risk-Off小时|最长小时|普通恢复|强恢复|恢复48小时内再触发|", "|---|---:|---:|---:|---:|---:|"]
    for p in ("BTC-FDUSD","ETH-FDUSD"):
        for arm in ("A","B"):
            r=report["results"][arm][p]
            rows.append(f"|{p}/{arm}|{r['risk_off_hours']}|{r['longest_risk_off_hours']}|{r['ordinary_recoveries']}|{r['strong_recoveries']}|{r['retrigger_within_48h']}|")
    rows += ["", "## 成交与执行成本", "", "|交易对/组|成交数|费用FDUSD|全部退出次数|重入次数|重入成本FDUSD|48小时额外库存退出|",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for p in ("BTC-FDUSD","ETH-FDUSD"):
        for arm in ("A","B"):
            r=report["results"][arm][p]
            rows.append(f"|{p}/{arm}|{r['trades']}|{r['fees']:.4f}|{r['exits']}|{r['reentries']}|{r['reentry_cost']:.4f}|{r['excess_exits']}|")
    rows += ["", "## 完整性与局限", "",
             f"A逐点与未修改生产advance_gate核对：{report['a_reference_parity_points']}点，0差异；两组采用相同周边界入场证据屏障。",
             "每周模型UBJ哈希、模型包/特征/策略哈希、原冻结训练面板SHA256、训练成熟截止及校准时序通过检查；追加周的原始训练行情快照未逐周重建，未进行复训。",
             "A参考源码与只读复制的OCI Guard源码内容一致（仅CRLF换行不同）。",
             "两组从相同签名起点空状态预热，预热状态连续带入年初；只在2026年起计入资金和收益，年内不按fold注资。",
             "B独立计数，普通连续3次或强连续2次且Risk-Off至少48小时才能恢复；缺失4小时周期不凑数。",
             "执行采用5分钟OHLC触价撮合、下一可执行开盘退出；已完成小时信号延后5分钟。3个健康5分钟周期近似真实Guard周期。",
             "使用同一过滤器快照，未重建全年过滤器变更、盘口排队或成交延迟。因此收益是本执行模型下的对照估计，不等于实盘保证。",
             "普通成本底线、额外库存10 FDUSD上限、24小时利润保护/48小时额外库存退出均相同；FOMC、优化、Momentum不参与。",
             "分歧明细的‘另一组下次恢复’可能已是不同事件，标记为观察比较，不当作同次事件的因果提前量。",
             f"9月诊断见september_diagnostic_signals.csv，至{report['diagnostic_end_exclusive']}（右开），主回测截止之后的诊断信号不计收益。", "",
             "## 产物", "", "- comparison.html：离线自包含Plotly，BTC/ETH及独立组合页，A/B阴影独立开关。",
             "- BTC-FDUSD_comparison.png / ETH-FDUSD_comparison.png：1440×2400手机图。",
             "- summary.json、hourly_signals.csv.gz、A/B_equity_5m.csv.gz、A/B_trades.csv：完整统计与流水。",
             "- divergence_episodes.csv：每段状态分歧及之后最多48小时权益变化。",
             "- weekly_model_audit.json、parameter_snapshot.json、artifact_manifest.json：时序、参数和哈希。", "",
             "## 结论", "", "独立强恢复能解决‘强条件成立、普通条件不改善时计数归零’的语义限制，但增加的市场敞口可能改善或恶化收益；应分别看BTC、ETH和回撤，不能仅按恢复更快决定上线。",
             "3%/6%风控阈值约束的是每次恢复后重设的风险周期；图中最大回撤使用不重置的累计权益峰值，因此可以超过单周期阈值。"]
    for p in ("BTC-FDUSD","ETH-FDUSD","PORTFOLIO"):
        a,b=report["results"]["A"][p],report["results"]["B"][p]
        rows.append(f"{p}：B−A净收益{b['pnl']-a['pnl']:+.4f} FDUSD；最大回撤变化{b['max_drawdown_pct']-a['max_drawdown_pct']:+.3f}个百分点。")
    eth=signals[(signals.pair=="ETH-FDUSD")&(signals.signal_ts>=start)&(signals.signal_ts<end)].pivot(index="signal_ts",columns="arm",values="risk_off")
    if eth.A.equals(eth.B):
        rows.append("ETH模型Risk-Off序列在主窗口完全相同；ETH交易收益仍可能受BTC传导的组合熔断影响。")
        ae=equities[(equities.pair=="ETH-FDUSD")&(equities.arm=="A")].set_index("ts")
        be=equities[(equities.pair=="ETH-FDUSD")&(equities.arm=="B")].set_index("ts")
        differences=(ae.equity-be.equity).abs()>1e-8
        if differences.any():
            first=int(differences[differences].index[0])
            rows.append(f"ETH首次权益分歧记录：{pd.Timestamp(first,unit='s',tz='UTC').isoformat()}。对应退出原因：")
            for arm in ("A","B"):
                es=pd.read_csv(output/f"{arm}_execution_events.csv")
                near=es[(es.ts>=first-600)&(es.ts<=first)&(es.kind=="EXIT_COMPLETE")]
                rows.append(f"- {arm}："+('；'.join(near.pair+':'+near.reason) if len(near) else "该时点无退出事件"))
    recent=signals[(signals.signal_ts>=end)&(signals.transition=="recover")]
    for r in recent.itertuples():
        rows.append(f"诊断（未计收益）：{r.pair} {r.arm} 于 {pd.Timestamp(r.signal_ts,unit='s',tz='UTC').isoformat()} 恢复，路径 {r.recovery_path}。")
    rows += ["", "## 复现与验收命令", "", "```sh",
             "OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python scripts/backtest_grid_v22_recovery_counters_2026.py",
             "python -m pytest test/test_grid_v22_recovery_ablation.py -q -p no:cacheprovider",
             "python scripts/validate_grid_v22_recovery_ablation.py", "```", "",
             "模型joblib来自Linux OCI；Windows XGBoost的同版本反序列化失败，本次使用本地WSL Linux隔离环境，未转换原模型。实际依赖版本见summary.json/runtime_versions。",
             "HTML阴影/trace索引完成静态检查；浏览器本地file URL被安全策略阻止，未宣称完成浏览器点击验收。PNG通过尺寸和中文人工目视检查。"]
    (output/"REPORT.md").write_text('\n'.join(rows)+'\n',encoding="utf-8")

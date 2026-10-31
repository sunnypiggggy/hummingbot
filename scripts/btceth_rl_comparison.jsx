import React, { useMemo, useState } from "react";
import { DataComponent, DataTable, Dropdown, EvidenceChart, ReportSection, RichNarrative, useDataApp } from "../../data-app-public.jsx";
import { SUMMARY_QUERIES, HISTORY_QUERIES, MODEL_NAMES, finite, comparisonAccounts, sourceScope, comparisonDisplay } from "./comparison-data.mjs";

const pct = value => finite(value) ? `${value > 0 ? "+" : ""}${(value * 100).toFixed(2)}%` : "—";
const magnitude = value => finite(value) ? `${(value * 100).toFixed(2)}%` : "—";
const money = value => finite(value) ? value.toLocaleString("zh-CN", {maximumFractionDigits: 2, minimumFractionDigits: 2}) : "—";
const returnCell = value => <span className={value > 0 ? "compare-positive" : value < 0 ? "compare-negative" : ""}>{pct(value)}</span>;
const ddCell = value => <span className={value > .2 ? "compare-deep-dd" : ""}>{magnitude(value)}</span>;
const COLORS = ["#2563eb", "#b45309", "#7c3aed", "#087f5b", "#64748b"];
const ordered = (rows, sort) => [...rows].sort((a,b) => sort === "drawdown" ? Math.abs(a.base.max_drawdown) - Math.abs(b.base.max_drawdown) : sort === "stress" ? b.stress.net_return - a.stress.net_return : b.base.net_return - a.base.net_return);
const ensembleDefaults = all => ["sac", "tqc", "tqc_band05"].map(id => all.find(p => p.kind === "ensemble" && p.variant === id)?.key).filter(Boolean);
const seedDefaults = all => ["tqc", "tqc_cap075", "tqc_band05"].map(id => all.find(p => p.kind === "seed" && p.variant === id && p.seed === 1)?.key).filter(Boolean);

function Binding({ id, title, accounts, queries, children, includeSeedEvidence = false, ...props }) {
  const sources = sourceScope(accounts, queries, {includeSeedEvidence});
  const ids = Object.keys(sources);
  return <DataComponent id={id} title={title} queryId={ids[0] ?? "strategy_summary"} queryIds={ids} sourceRowsByQuery={sources} displayRows={comparisonDisplay(accounts)} kind="table" {...props}>{children}</DataComponent>;
}

export function DirectComparison() {
  const { reviewedPeriodRows } = useDataApp();
  const queries = Object.fromEntries(SUMMARY_QUERIES.map(id => [id, reviewedPeriodRows(id, {period:"history"})]));
  const histories = Object.fromEntries(HISTORY_QUERIES.map(id => [id, reviewedPeriodRows(id, {period:"history"})]));
  const accounts = useMemo(() => comparisonAccounts(queries), SUMMARY_QUERIES.map(id => queries[id]));
  const [kind, setKind] = useState("ensemble");
  const [algorithm, setAlgorithm] = useState("all_models");
  const [sort, setSort] = useState("return");
  const [selected, setSelected] = useState(() => ensembleDefaults(accounts));
  const [scenario, setScenario] = useState("base");
  const [metric, setMetric] = useState("net_return");
  const [passive, setPassive] = useState(false);
  const changeKind = next => { setKind(next); setAlgorithm(next === "seed" ? "tqc" : "all_models"); setSelected(next === "seed" ? seedDefaults(accounts) : ensembleDefaults(accounts)); };
  const filtered = ordered(accounts.filter(p => p.kind === kind && (algorithm === "all_models" || (algorithm === "tqc" ? p.variant.startsWith("tqc") : p.variant === algorithm))), sort);
  const baseline = accounts.filter(p => p.kind === "baseline");
  const select = row => setSelected(old => old.includes(row.key) ? old.filter(key => key !== row.key) : [...old, row.key].slice(-4));
  const chosen = selected.map(key => accounts.find(p => p.key === key)).filter(Boolean);
  const plotted = [...chosen, ...(passive ? baseline.filter(p => p.variant === "equal_hold") : [])];
  const chartSources = {};
  const plot = new Map();
  plotted.forEach(pair => {
    const rows = histories[pair.historyQuery].filter(row => row.run_id === pair[scenario].run_id);
    (chartSources[pair.historyQuery] ??= []).push(...rows);
    rows.forEach(row => {
      if (!plot.has(row.timestamp)) plot.set(row.timestamp, { date: row.timestamp.replace(" ", "T") });
      plot.get(row.timestamp)[`${pair.label}(%)`] = finite(row[metric]) ? row[metric] * 100 : null;
    });
  });
  const fields = plotted.map(p => `${p.label}(%)`);
  const chartIds = Object.keys(chartSources);
  const chartRows = [...plot.entries()].sort(([a],[b]) => a.localeCompare(b)).map(([,row]) => row);
  const sac = accounts.find(p => p.variant === "sac" && p.kind === "ensemble");
  const tqc = accounts.find(p => p.variant === "tqc_cap075" && p.kind === "seed" && p.seed === 1);
  const overviewAccounts = [sac,tqc].filter(Boolean);
  const overviewSources = sourceScope(overviewAccounts, queries, {includeSeedEvidence:true});
  const columns = [
    {field:"name", label:"模型 / 交易变体", renderCell: (value,row) => <span className="compare-model-name">{value}{kind === "seed" && <small>单模型 · seed {row.seed}</small>}</span>},
    {field:"base_return", label:"主成本收益", renderCell:returnCell}, {field:"base_dd",label:"主成本回撤",renderCell:ddCell},
    {field:"stress_return", label:"压力成本收益", renderCell:returnCell}, {field:"stress_dd",label:"压力成本回撤",renderCell:ddCell},
    ...(kind === "ensemble" ? [{field:"positive_seeds",label:"盈利种子",renderCell:value => <span className="compare-seeds">{value}</span>}] : []),
    {field:"key",label:"曲线",renderCell:(value,row) => <button className="compare-line-button" aria-pressed={selected.includes(value)} aria-label={`${selected.includes(value) ? "移除" : "加入"}${row.label}曲线`} onClick={() => select(row)}>{selected.includes(value) ? "已选" : "加入"}</button>}
  ];

  return <div className="compare-content" aria-label="直接比较回测结果">
    <ReportSection id="compare-summary" title="先看这两种结果的区别" queryId="strategy_summary" queryIds={Object.keys(overviewSources)} sourceRowsByQuery={overviewSources} showHeading={false}>
      <RichNarrative id="compare-summary:body" value={`**SAC 五模型组合：收益 ${pct(sac?.base.net_return)} / 回撤 ${magnitude(Math.abs(sac?.base.max_drawdown))}；TQC 单模型 seed 1（75%预算）：${pct(tqc?.base.net_return)} / ${magnitude(Math.abs(tqc?.base.max_drawdown))}。** 对应盈利种子分别为 ${sac?.positiveSeeds}/5、${tqc?.positiveSeeds}/5；单模型高收益不等于组合或稳定复现。`} />
    </ReportSection>
    <div className="compare-workspace">
      <div className="compare-control-row">
        <div className="compare-segment" role="group" aria-label="比较对象">
          <button aria-pressed={kind === "ensemble"} onClick={() => changeKind("ensemble")}>五模型组合 <span>8组</span></button>
          <button aria-pressed={kind === "seed"} onClick={() => changeKind("seed")}>单模型 <span>全部40份</span></button>
        </div>
        <div className="compare-dropdowns">
          <Dropdown label="筛选模型" showLabel value={algorithm} choices={["all_models","tqc","sac","crossq","recurrent_ppo","td3","droq"]} choiceLabels={{all_models:"全部模型",tqc:"TQC 三种交易变体",sac:"SAC",crossq:"CrossQ",recurrent_ppo:"RecurrentPPO",td3:"TD3",droq:"DroQ"}} onChange={setAlgorithm} />
          <Dropdown label="排序" showLabel value={sort} choices={["return","drawdown","stress"]} choiceLabels={{return:"主成本收益从高到低",drawdown:"主成本回撤从小到大",stress:"压力成本收益从高到低"}} onChange={setSort}/>
        </div>
      </div>
      <RichNarrative id="compare-table-note" value={`**${kind === "ensemble" ? "五模型组合" : "单模型"} · 当前 ${filtered.length} 项。** ${kind === "ensemble" ? "盈利种子＝五份单模型中主成本盈利数。" : "保留全部五个种子，切换“全部模型”可查看40份。"} 最大回撤显示跌幅，越小越好。`} />
      <Binding id="compare-main-table" title="收益与最大回撤：两档成本并排比较" accounts={filtered} queries={queries} includeSeedEvidence={kind === "ensemble"}>
        <DataTable rows={comparisonDisplay(filtered)} rowKey="key" selectedRowKey={null} searchable={false} compactNumbers={false} caption="2025-10-23至2026-10-23 UTC · 365天 · 每个账本本金10,000 USDT" columns={columns}/>
      </Binding>
      <RichNarrative id="compare-cost-note" className="compare-fine-print" value="主成本：单边手续费0.10%＋滑点0.02%；压力成本：0.20%＋0.05%。净收益已扣两者，期末按持仓市值估值。75%预算＝原币仓目标×0.75，目标币仓上限71.25%（原上限95%）。该年度已被观察，收益排序不是新盲测结论；持现金的0%不代表学到有效交易。" />
    </div>
    <section className="compare-curves" aria-label="所选结果曲线">
      <div className="compare-control-row"><RichNarrative id="compare-curve-title" value="### 选中结果放在同一张图\n\n在表格点击“加入”，最多比较4项。"/>
        <div className="compare-dropdowns"><Dropdown label="曲线成本" showLabel value={scenario} choices={["base","stress"]} choiceLabels={{base:"主成本",stress:"压力成本"}} onChange={setScenario}/><Dropdown label="曲线指标" showLabel value={metric} choices={["net_return","drawdown"]} choiceLabels={{net_return:"累计净收益",drawdown:"回撤"}} onChange={setMetric}/><label className="compare-check"><input type="checkbox" checked={passive} onChange={event => setPassive(event.target.checked)}/>加入等权持有</label></div>
      </div>
      <div className="compare-selected" aria-label="已选曲线" data-reviewed-rows>{chosen.map((pair,index) => <button key={pair.key} onClick={() => select(pair)} aria-label={`移除${pair.label}曲线`}><i style={{background:COLORS[index]}}/>{pair.label}<span aria-hidden="true">×</span></button>)}</div>
      {chartRows.length > 0 ? <EvidenceChart id="compare-equity-chart" title={`${scenario === "base" ? "主成本" : "压力成本"} · ${metric === "net_return" ? "累计扣费净收益" : "日末相对小时历史峰值回撤"}`} queryId={chartIds[0]} queryIds={chartIds} sourceRowsByQuery={chartSources} rows={chartRows} spec={{type:"line",x:"date",y:fields[0],fields,stackable:false,valueDecimals:2, colors:Object.fromEntries(fields.map((field,index) => [field,COLORS[index]])),legend:{labels:Object.fromEntries(fields.map(field=>[field,field.replace(/\(%\)$/u,"")]))},yLabel:metric === "net_return" ? "累计净收益（%）" : "回撤（%）"}} height={360}/> : <RichNarrative id="compare-no-curves" value="尚未选择曲线。点击上方表格的“加入”。"/>}
    </section>
    <Binding id="compare-baselines" title="同窗口基准：收益和回撤的参照" accounts={baseline} queries={queries}>
      <DataTable rows={comparisonDisplay(baseline)} rowKey="key" searchable={false} compactNumbers={false} caption="独立10,000 USDT账本 · 与模型使用相同成本及成交口径" columns={[{field:"name",label:"固定规则基准"},...columns.slice(1,5)]}/>
    </Binding>
    <Binding id="compare-selected-details" title="所选结果的费用、仓位与期末清算" accounts={chosen} queries={queries}>
      <DataTable rows={comparisonDisplay(chosen)} rowKey="key" searchable={false} compactNumbers={false} caption="费用为手续费；净收益中还扣除了滑点。币仓为主成本全小时平均值。" columns={[{field:"label",label:"结果"},{field:"base_fees",label:"主成本手续费 USDT",renderCell:money},{field:"stress_fees",label:"压力手续费 USDT",renderCell:money},{field:"average_exposure",label:"平均币仓",renderCell:magnitude},{field:"base_liquidation",label:"净清算收益 / 主成本",renderCell:returnCell},{field:"stress_liquidation",label:"净清算收益 / 压力",renderCell:returnCell}]}/>
    </Binding>
    <RichNarrative id="compare-method-brief" className="compare-method-brief" value="### 如何理解比较\n\n范围固定为 BTCUSDT、ETHUSDT 与 USDT 现金，现货、无杠杆，每份独立本金10,000 USDT。五模型组合把同一账户状态下的五个目标权重取平均后实际成交；组合收益不是五份最终收益的平均数。\n\n本页使用各轮验证期选定的收益＋回撤惩罚配置；收益排序只描述这个已观察年度，不改变冻结主方案。TQC 三种变体复用同一组训练模型，区别在币仓预算和调仓门槛；它们不是三组独立训练。5pp 指5个百分点，其他变体门槛为2.5个百分点。TD3训练65,536环境步，其余32,768步；DroQ每步的训练更新量更多，计算预算不同，不能把差异全归因于算法。\n\n表格的最大回撤使用全部8,760个小时估值点计算；图中为日末采样，可能看不到小时内的最深回撤。完整训练、验证选择、月度结果和审计仍保存在顶部的“研究档案”中。"/>
  </div>;
}

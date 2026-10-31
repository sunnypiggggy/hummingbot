import React, { useState } from "react";
import { DataComponent, DataTable, Dropdown, EvidenceChart, ReportSection, RichNarrative, useDataApp } from "../../data-app-public.jsx";

const COST = { base: "主成本：费0.10%＋滑点0.02%", stress: "压力成本：费0.20%＋滑点0.05%" };
const PROFILE = { balanced: "收益＋回撤惩罚" };
const BUDGET = { raw: "原币仓预算", cap075: "75%币仓预算" };
const BASELINE = { cash: "USDT现金", equal_hold: "BTC/ETH等权被动持有", sma200_trend: "SMA200趋势", momentum180: "180日动量" };
const COLORS = { droq: "#7c3aed", cash: "#64748b", equal_hold: "var(--text)", sma200_trend: "#0f766e", momentum180: "#be123c" };
const GATES = {
  base_return_at_least_5pct: ["主成本净收益至少5%", "≥5%"],
  base_max_drawdown_at_most_20pct: ["小时最大回撤幅度不超过20%", "≤20%"],
  stress_return_nonnegative: ["压力成本净收益非负", "≥0%"],
  at_least_three_positive_training_seeds: ["至少三个训练种子主成本盈利", "≥3/5"],
  beats_equal_hold_return_and_drawdown: ["收益与回撤均优于等权被动持有", "收益更高且回撤幅度更小"],
  sma200_trend: ["收益与回撤均优于SMA200趋势", "收益更高且回撤幅度更小"],
  momentum180: ["收益与回撤均优于180日动量", "收益更高且回撤幅度更小"]
};
const finite = (value) => typeof value === "number" && Number.isFinite(value);
const pct = (value) => finite(value) ? `${(value * 100).toFixed(2)}%` : "—";
const num = (value, digits = 2) => finite(value) ? value.toLocaleString("zh-CN", { minimumFractionDigits: digits, maximumFractionDigits: digits }) : "—";
const pc = (field, label) => ({ field, label, renderCell: pct });
const nc = (field, label, digits = 2) => ({ field, label, renderCell: (value) => num(value, digits) });
const isModel = (row) => row.droq_role === "droq_selected";
const isEnsemble = (row) => isModel(row) && row.primary_policy === true;
const role = (row) => isModel(row) ? "droq" : row.strategy;
const name = (row) => isModel(row) ? "DroQ实际主组合" : BASELINE[row.strategy] ?? row.strategy;
const display = (row) => ({ ...row, _cost: COST[row.scenario], _name: name(row) });

function ready(summary, history, months, targets, choices, training, audits) {
  return summary.length === 26 && summary.filter(isEnsemble).length === 2
    && summary.filter((row) => isModel(row) && row.primary_policy === false && finite(row.seed)).length === 10
    && history.length === 9516 && months.length === 338 && targets.length === 7 && choices.length === 2
    && training.length === 5 && audits.length === 2 && audits.every((row) => row.passed === true);
}

function curveScope(summary, history, scenario) {
  const accounts = summary.filter((row) => row.scenario === scenario && (isEnsemble(row)
    || row.droq_role === "original_baseline_reference" && Object.hasOwn(BASELINE, row.strategy)));
  const lookup = new Map(accounts.map((row) => [row.run_id, row]));
  return { accounts, lookup, rows: history.filter((row) => lookup.has(row.run_id)) };
}

function pivot(rows, lookup, field) {
  const points = new Map();
  rows.forEach((row) => {
    const account = lookup.get(row.run_id);
    if (!account) return;
    if (!points.has(row.timestamp)) points.set(row.timestamp, { timestamp: row.timestamp, date: row.timestamp.replace(" ", "T") });
    points.get(row.timestamp)[`${name(account)}(%)`] = finite(row[field]) ? row[field] * 100 : null;
  });
  return [...points.values()].sort((a, b) => a.timestamp.localeCompare(b.timestamp));
}

function spec(accounts, field) {
  const fields = accounts.map((row) => `${name(row)}(%)`);
  return { type: /** @type {"line"} */ ("line"), x: "date", xLabel: "日期（UTC）", y: fields[0] ?? "扣费净收益(%)", fields,
    stackable: false, valueDecimals: 2, colors: Object.fromEntries(accounts.map((row) => [`${name(row)}(%)`, COLORS[role(row)]])),
    legend: { labels: Object.fromEntries(fields.map((field) => [field, field.replace(/\(%\)$/u, "")])) },
    yLabel: field === "drawdown" ? "相对完整小时历史峰值（%）" : "累计扣费净收益（%）" };
}

function gateRows(targets, summary) {
  const base = summary.find((row) => isEnsemble(row) && row.scenario === "base");
  const stress = summary.find((row) => isEnsemble(row) && row.scenario === "stress");
  const positive = summary.filter((row) => isModel(row) && row.primary_policy === false && row.scenario === "base" && row.net_return > 0).length;
  return targets.map((row) => {
    const control = summary.find((item) => item.droq_role === "original_baseline_reference" && item.scenario === "base"
      && item.strategy === (row.gate === "beats_equal_hold_return_and_drawdown" ? "equal_hold" : row.gate));
    const observed = row.gate === "base_return_at_least_5pct" ? pct(base?.net_return)
      : row.gate === "base_max_drawdown_at_most_20pct" ? pct(finite(base?.max_drawdown) ? Math.abs(base.max_drawdown) : null)
      : row.gate === "stress_return_nonnegative" ? pct(stress?.net_return)
      : row.gate === "at_least_three_positive_training_seeds" ? `${positive}/5`
      : control ? `对照收益 ${pct(control.net_return)}；回撤 ${pct(control.max_drawdown)}` : "—";
    return { ...row, _label: GATES[row.gate]?.[0] ?? row.gate, _criterion: GATES[row.gate]?.[1] ?? "—", _observed: observed };
  });
}

function monthRows(rows, lookup) {
  const months = new Map();
  rows.forEach((row) => {
    const account = lookup.get(row.run_id);
    if (!account) return;
    if (!months.has(row.month)) months.set(row.month, { month: row.month });
    months.get(row.month)[role(account)] = finite(row.return) ? row.return : null;
  });
  return [...months.values()].sort((a, b) => a.month.localeCompare(b.month));
}

export function DroQFollowup() {
  const { snapshot, reviewedPeriodRows } = useDataApp();
  const [scenario, setScenario] = useState("base");
  if (!snapshot.report?.droqFollowup) return null;
  const summary = reviewedPeriodRows("droq_summary", { period: "history" });
  const history = reviewedPeriodRows("droq_equity", { period: "history" });
  const months = reviewedPeriodRows("droq_monthly", { period: "history" });
  const targets = reviewedPeriodRows("droq_targets", { period: "history" });
  const choices = reviewedPeriodRows("droq_validation_selection", { period: "history" });
  const training = reviewedPeriodRows("droq_training", { period: "history" });
  const audits = reviewedPeriodRows("droq_audit", { period: "history" });
  if (!ready(summary, history, months, targets, choices, training, audits)) return null;
  const selected = summary.find((row) => isEnsemble(row) && row.scenario === scenario);
  const diagnostic = targets.filter((row) => !["sma200_trend", "momentum180"].includes(row.gate));
  const met = diagnostic.filter((row) => row.met === true).length;
  const inactive = selected && selected.average_exposure < .005 ? "该组合平均币仓不足0.5%，结果主要反映持现金，不能认定学到了有效交易。" : "";
  const opening = `## 新模型：DroQ的收益与回撤\n\n只用295天验证，在原币仓和75%币仓两个事先固定预算中选定 **${PROFILE[selected?.profile] ?? "—"} · ${BUDGET[selected?.family] ?? "—"}**。当前查看 **${COST[scenario]}**：实际主组合扣费净收益 **${pct(selected?.net_return)}**，小时最大回撤 **${pct(selected?.max_drawdown)}**，平均币仓 **${pct(selected?.average_exposure)}**。${inactive}\n\nDroQ自身五项诊断条件达到 **${met}/5**，${met === 5 ? "达到本轮固定研究条件" : "尚未同时达到本轮固定研究条件"}。下表保留全部五个种子，主组合由同一真实账户上的五份目标平均后成交。\n\n本轮固定一种收益与回撤奖励，训练五个模型，每模型 **32,768个环境步**。每模型执行 **555,360轮联合Q更新**、两个Q优化器合计 **1,110,720次更新**，策略与熵系数各 **27,768次更新**。环境步数与首轮相同，训练计算量更高，不能将效果差异只归因于算法。0.01 dropout和LayerNorm为事先固定研究参数，没有在年度结果上搜索。\n\n这是在观察过此前年度结果后开展的研究，回放 **2025-10-23至2026-10-23 UTC**，**不是新的盲测样本**。`;
  const original = reviewedPeriodRows("strategy_summary", { period: "history" });
  const sac = original.filter((row) => row.algorithm === "sac" && row.split === "test" && row.primary_policy === true && row.selected_profile === true);
  const oldTQC = original.filter((row) => row.algorithm === "tqc_cap075" && row.split === "test" && row.primary_policy === true && row.selected_profile === true);
  const band = snapshot.report.executionBand ? reviewedPeriodRows("execution_band_summary", { period: "history" }).filter((row) => row.followup_role === "followup" && row.primary_policy === true) : [];
  const td3 = snapshot.report.td3Followup ? reviewedPeriodRows("td3_summary", { period: "history" }).filter((row) => row.td3_role === "td3_selected" && row.primary_policy === true) : [];
  const ensembles = summary.filter(isEnsemble);
  const comparisons = [...ensembles.map((row) => ({ ...row, _name: "DroQ验证选定主组合", _budget: "32,768环境步；每模型555,360轮联合Q更新" })),
    ...sac.map((row) => ({ ...row, _name: "首轮SAC对照", _budget: "每模型32,768环境步" })),
    ...oldTQC.map((row) => ({ ...row, _name: "首轮TQC75%预算", _budget: "每模型32,768环境步" })),
    ...band.map((row) => ({ ...row, _name: "TQC5个百分点门槛", _budget: "复用首轮32,768步模型" })),
    ...td3.map((row) => ({ ...row, _name: "TD3验证选定主组合", _budget: "每模型65,536环境步" }))].map((row) => ({ ...row, _id: `${row._name}:${row.run_id}`, _cost: COST[row.scenario] }));
  const comparisonSources = { droq_summary: ensembles, strategy_summary: [...sac, ...oldTQC], ...(band.length ? { execution_band_summary: band } : {}), ...(td3.length ? { td3_summary: td3 } : {}) };
  const chart = curveScope(summary, history, scenario);
  const seeds = summary.filter((row) => isModel(row) && row.primary_policy === false && row.scenario === scenario);
  const gates = gateRows(targets, summary);
  const monthSource = months.filter((row) => chart.lookup.has(row.run_id));
  const chosen = snapshot.report.droqFollowup.chosen ?? {};
  const candidateRows = choices.map((row) => ({ ...row, _id: `${row.family}:${row.profile}`, _profile: PROFILE[row.profile], _budget: BUDGET[row.family],
    _decision: row.family === chosen.family && row.profile === chosen.profile ? "验证选中并封存" : "保留验证证据",
    _baseReturn: row.base?.net_return, _baseDD: row.base?.max_drawdown, _stressReturn: row.stress?.net_return, _stressDD: row.stress?.max_drawdown }));
  const trainingRows = training.map((row) => ({ ...row, _id: `${row.profile}:${row.seed}`, _profile: PROFILE[row.profile] }));
  return <section aria-label="已观察年度的DroQ独立跟进">
    <ReportSection id="btceth-droq-opening" title="DroQ独立研究结果" queryId="droq_summary" queryIds={["droq_summary", "droq_targets"]} sourceRowsByQuery={{ droq_summary: ensembles, droq_targets: targets }} showHeading={false}><RichNarrative id="btceth-droq-opening:body" value={opening} /></ReportSection>
    <DataComponent id="btceth-droq-comparison" title="两档成本：DroQ与此前SAC、TQC、TD3的实际主组合" queryId="droq_summary" queryIds={Object.keys(comparisonSources)} sourceRowsByQuery={comparisonSources} displayRows={comparisons} kind="table"><DataTable rows={comparisons} rowKey="_id" searchable={false} compactNumbers={false} caption="所有账户各自初始10,000 USDT；旧结果直接引用原审计来源，训练预算不同，不能归因于算法单一差异" columns={[{ field: "_name", label: "实际方案" }, { field: "_cost", label: "成本条件" }, pc("net_return", "扣费净收益"), pc("max_drawdown", "小时最大回撤"), nc("fees", "手续费USDT"), pc("average_exposure", "平均币仓"), { field: "_budget", label: "训练预算" }]} /></DataComponent>
    <div className="btceth-controls" aria-label="DroQ曲线、种子与月度的成本条件"><Dropdown label="DroQ显示成本" showLabel value={scenario} choices={["base", "stress"]} choiceLabels={COST} onChange={setScenario} /></div>
    <DataComponent id="btceth-droq-ledger" title="DroQ主组合完整指标：两档成本独立账本" queryId="droq_summary" sourceRows={ensembles} displayRows={ensembles.map(display)} kind="table"><DataTable rows={ensembles.map(display)} rowKey="run_id" searchable={false} compactNumbers={false} caption="五个模型读取同一实际账户后平均目标并成交；不平均五份期末收益。费用与滑点均已计入净收益" columns={[{ field: "_cost", label: "成本" }, pc("net_return", "扣费净收益"), pc("max_drawdown", "小时最大回撤"), nc("sharpe_daily", "日Sharpe"), nc("calmar", "Calmar"), nc("fees", "手续费USDT"), nc("slippage_cost", "滑点USDT"), nc("turnover", "周转倍数"), pc("average_exposure", "平均币仓"), pc("net_liquidation_return", "扣期末退出成本后")]} /></DataComponent>
    <EvidenceChart id="btceth-droq-return" title="当前成本：DroQ与现金、被动持有及简单趋势的扣费收益" queryId="droq_equity" sourceRows={chart.rows} rows={pivot(chart.rows, chart.lookup, "net_return")} spec={spec(chart.accounts, "net_return")} height={350} />
    <EvidenceChart id="btceth-droq-drawdown" title="当前成本：日边界相对完整小时峰值的回撤" queryId="droq_equity" sourceRows={chart.rows} rows={pivot(chart.rows, chart.lookup, "drawdown")} spec={spec(chart.accounts, "drawdown")} height={270} />
    <DataComponent id="btceth-droq-targets" title="DroQ自身五项诊断目标与两个简单趋势对照" queryId="droq_targets" queryIds={["droq_targets", "droq_summary"]} sourceRowsByQuery={{ droq_targets: targets, droq_summary: summary }} displayRows={gates} kind="table"><DataTable rows={gates} rowKey="gate" searchable={false} caption="固定年度、固定两成本与全部五种子；不随显示成本切换，不修改原模型的失败目标" columns={[{ field: "_label", label: "检查条件" }, { field: "category", label: "类别" }, { field: "_observed", label: "实际值／对照值" }, { field: "_criterion", label: "标准" }, { field: "met", label: "结果", renderCell: (value) => value === true ? "达到" : value === false ? "未达到" : "缺少证据" }]} /></DataComponent>
    <DataComponent id="btceth-droq-seeds" title="当前成本：选定配置全部五个训练种子" queryId="droq_summary" sourceRows={seeds} displayRows={seeds} kind="table"><DataTable rows={seeds} rowKey="run_id" searchable={false} caption="独立训练重复同一市场历史，描述训练不稳定性；它们不是五个独立市场样本" columns={[nc("seed", "训练种子", 0), pc("net_return", "扣费净收益"), pc("max_drawdown", "小时最大回撤"), nc("fees", "手续费USDT"), pc("average_exposure", "平均币仓")]} /></DataComponent>
    <DataComponent id="btceth-droq-selection" title="295天验证：两个事先固定的币仓预算" queryId="droq_validation_selection" sourceRows={choices} displayRows={candidateRows} kind="table"><DataTable rows={candidateRows} rowKey="_id" searchable={false} caption="两成本既定收益与回撤评分；精确同分优先原预算；年度结果没有参与选择" columns={[{ field: "_profile", label: "奖励配置" }, { field: "_budget", label: "币仓预算" }, { field: "_decision", label: "封存选择" }, nc("score", "验证评分", 6), pc("_baseReturn", "主成本收益"), pc("_baseDD", "主成本小时回撤"), pc("_stressReturn", "压力成本收益"), pc("_stressDD", "压力成本小时回撤")]} /></DataComponent>
    <DataComponent id="btceth-droq-months" title="当前成本：DroQ及四个基准的月度精确值" queryId="droq_monthly" sourceRows={monthSource} displayRows={monthRows(monthSource, chart.lookup)} kind="table"><DataTable rows={monthRows(monthSource, chart.lookup)} rowKey="month" searchable={false} caption="UTC真实首尾区间；未补齐首尾月；与权益和期末收益对账" columns={[{ field: "month", label: "UTC月份" }, pc("droq", "DroQ"), ...Object.entries(BASELINE).map(([field, label]) => pc(field, label))]} /></DataComponent>
    <ReportSection id="btceth-droq-method" title="DroQ实现与研究边界" queryId="droq_training" sourceRows={training} showHeading={false}><RichNarrative id="btceth-droq-method:body" value={"## DroQ实现与研究边界\n\n使用[DroQ原作者实现](https://github.com/TakuyaHiraoka/Dropout-Q-Functions-for-Doubly-Efficient-Reinforcement-Learning/blob/4c7adddbaf7573ff9256b16091a3bbdbd536aad4/OriginalREDQCodebase/redq/algos/redq_sac.py)，固定两个评价网络、每次经验20轮Q更新、Dropout与LayerNorm。0.01 dropout是本轮预设研究假设，不能当作适用于所有市场的官方默认。保留作者实际更新顺序和自动熵系数逻辑；推理使用确定性策略均值，保存的评价网络与优化器状态由独立工具核验。\n\n训练仅使用2020-01-01至2025-01-01真实完整32日片段，验证为随后295日。每个模型单卡V100训练32,768步，全部五种子保留。32日片段边界保留真实终点观测并自举；每个模型1024次截断均需独立核对因果行情前缀与实际回放数组。\n\n两个推理预算只改变目标币仓：原预算最高95%，75%预算最高71.25%，实际权重可随价格漂移。2.5个百分点再平衡门槛、00UTC先卖/01UTC再买、双边费用与小时估值沿用前轮。五模型读取同一个实际资金账户反馈后平均目标，统一使用10,000 USDT现金与本组合持币成交。\n\n完整会计审计通过证明结果可对账；收益、回撤及种子稳定性分别判断。已观察年度不能提供新的盲测证明，本研究未接入实盘或Paper交易。"} /></ReportSection>
    <DataComponent id="btceth-droq-training" title="五个固定DroQ模型：真实训练与优化器更新" queryId="droq_training" sourceRows={training} displayRows={trainingRows} kind="table"><DataTable rows={trainingRows} rowKey="_id" searchable={false} compactNumbers={false} caption="环境交互步数相同不等于计算量相同；每Q真实Adam次数与完整参数状态均需独立核验" columns={[nc("seed", "种子", 0), nc("actual_timesteps", "环境步数", 0), nc("q_inner_rounds", "联合Q更新轮数", 0), nc("total_q_optimizer_steps", "两个Q合计更新", 0), nc("policy_optimizer_steps", "策略网络更新", 0), nc("alpha_optimizer_steps", "熵系数更新", 0), nc("covered_training_days", "可抽样日期并集", 0), { field: "gpu_uuid", label: "实际物理GPU" }, { field: "model_tensor_sha256", label: "模型Tensor SHA256", renderCell: (value) => <span className="btceth-hash">{value}</span> }]} /></DataComponent>
    <DataComponent id="btceth-droq-audit" title="DroQ完整独立审计与原始来源" queryId="droq_audit" sourceRows={audits} displayRows={audits} kind="table"><DataTable rows={audits} rowKey="file" searchable={false} caption="五模型、24验证及重复、12年度及重复、14原基准引用；引用不计作新模型或新市场样本" columns={[{ field: "role", label: "审计身份" }, nc("new_runs", "新年度账本", 0), nc("baseline_reference_runs", "原基准引用", 0), nc("validation_runs", "验证账本", 0), nc("repeat_count", "年度精确重放", 0), { field: "file", label: "实际文件" }, { field: "sha256", label: "SHA256", renderCell: (value) => <span className="btceth-hash">{value}</span> }]} /></DataComponent>
  </section>;
}

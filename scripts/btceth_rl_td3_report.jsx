import React, { useState } from "react";
import { DataComponent, DataTable, Dropdown, EvidenceChart, ReportSection, RichNarrative, useDataApp } from "../../data-app-public.jsx";

const COST = { base: "主成本：费0.10%＋滑点0.02%", stress: "压力成本：费0.20%＋滑点0.05%" };
const PROFILE = { return: "净收益奖励", balanced: "收益＋回撤惩罚" };
const BUDGET = { raw: "原币仓预算", cap075: "75%币仓预算" };
const BASELINE = { cash: "USDT现金", equal_hold: "BTC/ETH等权被动持有", sma200_trend: "SMA200趋势", momentum180: "180日动量" };
const COLORS = { td3: "#2563eb", cash: "#64748b", equal_hold: "var(--text)", sma200_trend: "#0f766e", momentum180: "#be123c" };
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
const isModel = (row) => row.td3_role === "td3_selected";
const isEnsemble = (row) => isModel(row) && row.primary_policy === true;
const role = (row) => isModel(row) ? "td3" : row.strategy;
const name = (row) => isModel(row) ? "TD3实际主组合" : BASELINE[row.strategy] ?? row.strategy;
const display = (row) => ({ ...row, _cost: COST[row.scenario], _name: name(row) });

function ready(summary, history, months, targets, choices, training, audits) {
  return summary.length === 26 && summary.filter(isEnsemble).length === 2
    && summary.filter((row) => isModel(row) && row.primary_policy === false && finite(row.seed)).length === 10
    && history.length > 0 && months.length > 0 && targets.length === 7 && choices.length === 4
    && training.length === 10 && audits.length > 0 && audits.every((row) => row.passed === true);
}

function curveScope(summary, history, scenario) {
  const accounts = summary.filter((row) => row.scenario === scenario && (isEnsemble(row)
    || row.td3_role === "original_baseline_reference" && Object.hasOwn(BASELINE, row.strategy)));
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
    const control = summary.find((item) => item.td3_role === "original_baseline_reference" && item.scenario === "base"
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

export function TD3Followup() {
  const { snapshot, reviewedPeriodRows } = useDataApp();
  const [scenario, setScenario] = useState("base");
  if (!snapshot.report?.td3Followup) return null;
  const summary = reviewedPeriodRows("td3_summary", { period: "history" });
  const history = reviewedPeriodRows("td3_equity", { period: "history" });
  const months = reviewedPeriodRows("td3_monthly", { period: "history" });
  const targets = reviewedPeriodRows("td3_targets", { period: "history" });
  const choices = reviewedPeriodRows("td3_validation_selection", { period: "history" });
  const training = reviewedPeriodRows("td3_training", { period: "history" });
  const audits = reviewedPeriodRows("td3_audit", { period: "history" });
  if (!ready(summary, history, months, targets, choices, training, audits)) return null;
  const selected = summary.find((row) => isEnsemble(row) && row.scenario === scenario);
  const diagnostic = targets.filter((row) => !["sma200_trend", "momentum180"].includes(row.gate));
  const met = diagnostic.filter((row) => row.met === true).length;
  const inactive = selected && selected.average_exposure < .005 ? "该组合平均币仓不足0.5%，结果主要反映持现金，不能认定学到了有效交易。" : "";
  const opening = `## 新模型：TD3的收益与回撤\n\n只用295天验证，在两种奖励配置与两种币仓预算共四个固定候选中选定 **${PROFILE[selected?.profile] ?? "—"} · ${BUDGET[selected?.family] ?? "—"}**。当前查看 **${COST[scenario]}**：实际主组合扣费净收益 **${pct(selected?.net_return)}**，小时最大回撤 **${pct(selected?.max_drawdown)}**，平均币仓 **${pct(selected?.average_exposure)}**。${inactive}\n\nTD3自身五项诊断条件达到 **${met}/5**，${met === 5 ? "达到本轮固定研究条件" : "尚未同时达到本轮固定研究条件"}。下表保留全部五个种子，不用最好的单个种子替代主组合，也不借用SAC或TQC的结果判断TD3达标。\n\n每模型训练 **65,536步**，为此前32,768步的 **2倍环境步数预算**；两种奖励各五种子共十个模型。训练数据和资金成交口径沿用前轮，增加训练步数没有增加独立市场样本。这是在观察过首轮年度结果后开展的独立跟进，回放 **2025-10-23至2026-10-23 UTC**，**不是新的盲测样本**。`;
  const original = reviewedPeriodRows("strategy_summary", { period: "history" });
  const sac = original.filter((row) => row.algorithm === "sac" && row.split === "test" && row.primary_policy === true && row.selected_profile === true);
  const oldTQC = original.filter((row) => row.algorithm === "tqc_cap075" && row.split === "test" && row.primary_policy === true && row.selected_profile === true);
  const band = snapshot.report.executionBand ? reviewedPeriodRows("execution_band_summary", { period: "history" }).filter((row) => row.followup_role === "followup" && row.primary_policy === true) : [];
  const ensembles = summary.filter(isEnsemble);
  const comparisons = [...ensembles.map((row) => ({ ...row, _name: "TD3验证选定主组合", _budget: "每模型65,536步" })),
    ...sac.map((row) => ({ ...row, _name: "首轮SAC对照", _budget: "每模型32,768步" })),
    ...oldTQC.map((row) => ({ ...row, _name: "首轮TQC75%预算", _budget: "每模型32,768步" })),
    ...band.map((row) => ({ ...row, _name: "TQC5个百分点门槛", _budget: "复用首轮32,768步模型" }))].map((row) => ({ ...row, _id: `${row._name}:${row.run_id}`, _cost: COST[row.scenario] }));
  const comparisonSources = { td3_summary: ensembles, strategy_summary: [...sac, ...oldTQC], ...(band.length ? { execution_band_summary: band } : {}) };
  const chart = curveScope(summary, history, scenario);
  const seeds = summary.filter((row) => isModel(row) && row.primary_policy === false && row.scenario === scenario);
  const gates = gateRows(targets, summary);
  const monthSource = months.filter((row) => chart.lookup.has(row.run_id));
  const chosen = snapshot.report.td3Followup.selection?.candidate ?? {};
  const candidateRows = choices.map((row) => ({ ...row, _id: `${row.family}:${row.profile}`, _profile: PROFILE[row.profile], _budget: BUDGET[row.family],
    _decision: row.family === chosen.family && row.profile === chosen.profile ? "验证选中并封存" : "保留验证证据",
    _baseReturn: row.base?.net_return, _baseDD: row.base?.max_drawdown, _stressReturn: row.stress?.net_return, _stressDD: row.stress?.max_drawdown }));
  const trainingRows = training.map((row) => ({ ...row, _id: `${row.profile}:${row.seed}`, _profile: PROFILE[row.profile] }));
  return <section aria-label="已观察年度的TD3独立跟进">
    <ReportSection id="btceth-td3-opening" title="TD3独立研究结果" queryId="td3_summary" queryIds={["td3_summary", "td3_targets"]} sourceRowsByQuery={{ td3_summary: ensembles, td3_targets: targets }} showHeading={false}><RichNarrative id="btceth-td3-opening:body" value={opening} /></ReportSection>
    <DataComponent id="btceth-td3-comparison" title="两档成本：TD3与此前SAC、TQC的实际主组合" queryId="td3_summary" queryIds={Object.keys(comparisonSources)} sourceRowsByQuery={comparisonSources} displayRows={comparisons} kind="table"><DataTable rows={comparisons} rowKey="_id" searchable={false} compactNumbers={false} caption="所有账户各自初始10,000 USDT；旧结果直接引用原审计来源，训练预算不同，不能归因于算法单一差异" columns={[{ field: "_name", label: "实际方案" }, { field: "_cost", label: "成本条件" }, pc("net_return", "扣费净收益"), pc("max_drawdown", "小时最大回撤"), nc("fees", "手续费USDT"), pc("average_exposure", "平均币仓"), { field: "_budget", label: "训练预算" }]} /></DataComponent>
    <div className="btceth-controls" aria-label="TD3曲线、种子与月度的成本条件"><Dropdown label="TD3显示成本" showLabel value={scenario} choices={["base", "stress"]} choiceLabels={COST} onChange={setScenario} /></div>
    <DataComponent id="btceth-td3-ledger" title="TD3主组合完整指标：两档成本独立账本" queryId="td3_summary" sourceRows={ensembles} displayRows={ensembles.map(display)} kind="table"><DataTable rows={ensembles.map(display)} rowKey="run_id" searchable={false} compactNumbers={false} caption="五个模型读取同一实际账户后平均目标并成交；不平均五份期末收益。费用与滑点均已计入净收益" columns={[{ field: "_cost", label: "成本" }, pc("net_return", "扣费净收益"), pc("max_drawdown", "小时最大回撤"), nc("sharpe_daily", "日Sharpe"), nc("calmar", "Calmar"), nc("fees", "手续费USDT"), nc("slippage_cost", "滑点USDT"), nc("turnover", "周转倍数"), pc("average_exposure", "平均币仓"), pc("net_liquidation_return", "扣期末退出成本后")]} /></DataComponent>
    <EvidenceChart id="btceth-td3-return" title="当前成本：TD3与现金、被动持有及简单趋势的扣费收益" queryId="td3_equity" sourceRows={chart.rows} rows={pivot(chart.rows, chart.lookup, "net_return")} spec={spec(chart.accounts, "net_return")} height={350} />
    <EvidenceChart id="btceth-td3-drawdown" title="当前成本：日边界相对完整小时峰值的回撤" queryId="td3_equity" sourceRows={chart.rows} rows={pivot(chart.rows, chart.lookup, "drawdown")} spec={spec(chart.accounts, "drawdown")} height={270} />
    <DataComponent id="btceth-td3-targets" title="TD3自身五项诊断目标与两个简单趋势对照" queryId="td3_targets" queryIds={["td3_targets", "td3_summary"]} sourceRowsByQuery={{ td3_targets: targets, td3_summary: summary }} displayRows={gates} kind="table"><DataTable rows={gates} rowKey="gate" searchable={false} caption="固定年度、固定两成本与全部五种子；不随显示成本切换，不修改原模型的失败目标" columns={[{ field: "_label", label: "检查条件" }, { field: "category", label: "类别" }, { field: "_observed", label: "实际值／对照值" }, { field: "_criterion", label: "标准" }, { field: "met", label: "结果", renderCell: (value) => value === true ? "达到" : value === false ? "未达到" : "缺少证据" }]} /></DataComponent>
    <DataComponent id="btceth-td3-seeds" title="当前成本：选定配置全部五个训练种子" queryId="td3_summary" sourceRows={seeds} displayRows={seeds} kind="table"><DataTable rows={seeds} rowKey="run_id" searchable={false} caption="独立训练重复同一市场历史，描述训练不稳定性；它们不是五个独立市场样本" columns={[nc("seed", "训练种子", 0), pc("net_return", "扣费净收益"), pc("max_drawdown", "小时最大回撤"), nc("fees", "手续费USDT"), pc("average_exposure", "平均币仓")]} /></DataComponent>
    <DataComponent id="btceth-td3-selection" title="295天验证：四个事先固定的候选" queryId="td3_validation_selection" sourceRows={choices} displayRows={candidateRows} kind="table"><DataTable rows={candidateRows} rowKey="_id" searchable={false} caption="两成本既定收益与回撤评分；精确同分先原预算，再按配置标识排序；年度结果没有参与选择" columns={[{ field: "_profile", label: "奖励配置" }, { field: "_budget", label: "币仓预算" }, { field: "_decision", label: "封存选择" }, nc("score", "验证评分", 6), pc("_baseReturn", "主成本收益"), pc("_baseDD", "主成本小时回撤"), pc("_stressReturn", "压力成本收益"), pc("_stressDD", "压力成本小时回撤")]} /></DataComponent>
    <DataComponent id="btceth-td3-months" title="当前成本：TD3及四个基准的月度精确值" queryId="td3_monthly" sourceRows={monthSource} displayRows={monthRows(monthSource, chart.lookup)} kind="table"><DataTable rows={monthRows(monthSource, chart.lookup)} rowKey="month" searchable={false} caption="UTC真实首尾区间；未补齐首尾月；与权益和期末收益对账" columns={[{ field: "month", label: "UTC月份" }, pc("td3", "TD3"), ...Object.entries(BASELINE).map(([field, label]) => pc(field, label))]} /></DataComponent>
    <ReportSection id="btceth-td3-method" title="TD3实现与研究边界" queryId="td3_training" sourceRows={training} showHeading={false}><RichNarrative id="btceth-td3-method:body" value="## TD3实现与训练预算\n\n使用[Stable-Baselines3 2.9.0的TD3](https://stable-baselines3.readthedocs.io/en/v2.9.0/modules/td3.html)，固定双评价网络、延迟策略更新和目标动作平滑。沿用前轮离线研究的网络、折扣与训练骨架，并加入固定高斯探索噪声；这不是纯官方默认参数，也未根据年度结果调整动作偏置或探索幅度。\n\n训练仅使用2020-01-01至2025-01-01真实完整32日片段，验证为随后295日。两种奖励配置各五种子，每个模型单卡V100训练65,536步。32日片段边界按时间截断保留终点观测并自举，不当作经济终结；独立检查全部训练记录和终点观测。\n\n两种推理预算只改变目标币仓：原预算最高95%，75%预算最高71.25%，实际权重可随价格漂移。2.5个百分点再平衡门槛、00UTC先卖/01UTC再买、双边费用与小时估值沿用前轮。完整会计审计通过只证明结果可对账；目标达标也不等于获得实盘授权。" /></ReportSection>
    <DataComponent id="btceth-td3-training" title="十个固定TD3模型：真实训练记录" queryId="td3_training" sourceRows={training} displayRows={trainingRows} kind="table"><DataTable rows={trainingRows} rowKey="_id" searchable={false} caption="环境步数与实际评价、策略更新次数分别披露；没有增加独立市场历史或按年度选种子" columns={[{ field: "_profile", label: "配置" }, nc("seed", "种子", 0), nc("actual_timesteps", "真实步数", 0), nc("actual_n_updates", "评价网络更新", 0), nc("actor_update_count", "策略网络更新", 0), nc("covered_training_days", "可抽样日期并集", 0), { field: "gpu_uuid", label: "实际物理GPU" }, { field: "model_sha256", label: "模型SHA256", renderCell: (value) => <span className="btceth-hash">{value}</span> }]} /></DataComponent>
    <DataComponent id="btceth-td3-audit" title="TD3完整独立审计与原始来源" queryId="td3_audit" sourceRows={audits} displayRows={audits} kind="table"><DataTable rows={audits} rowKey="file" searchable={false} caption="十模型、48验证及重复、12年度及重复、14原基准引用；引用不计作新模型或新市场样本" columns={[{ field: "role", label: "审计身份" }, nc("new_runs", "新年度账本", 0), nc("baseline_reference_runs", "原基准引用", 0), nc("validation_runs", "验证账本", 0), nc("repeat_count", "年度精确重放", 0), { field: "file", label: "实际文件" }, { field: "sha256", label: "SHA256", renderCell: (value) => <span className="btceth-hash">{value}</span> }]} /></DataComponent>
  </section>;
}

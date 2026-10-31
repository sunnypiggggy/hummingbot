import React, { useState } from "react";
import { DataComponent, DataTable, Dropdown, EvidenceChart, ReportSection, RichNarrative, useDataApp } from "../../data-app-public.jsx";

const COST = { base: "主成本：费0.10%＋滑点0.02%", stress: "压力成本：费0.20%＋滑点0.05%" };
const ROLE = { followup: "5个百分点新组合", original_cap_reference: "原2.5个百分点组合", equal_hold: "BTC/ETH等权被动持有", cash: "USDT现金" };
const COLOR = { followup: "#0891b2", original_cap_reference: "#7c3aed", equal_hold: "var(--text)", cash: "#64748b" };
const GATES = {
  base_return_at_least_5pct: ["主成本净收益至少5%", "≥5%"],
  base_max_drawdown_at_most_20pct: ["小时最大回撤幅度不超过20%", "≤20%"],
  stress_return_nonnegative: ["压力成本收益非负", "≥0%"],
  at_least_three_positive_training_seeds: ["至少三个训练种子主成本收益为正", "≥3/5；保留全部五个种子"],
  beats_equal_hold_return_and_drawdown: ["收益和回撤均优于等权被动持有", "净收益更高、回撤幅度更小"],
  sma200_trend: ["收益和回撤均优于SMA200趋势", "净收益更高、回撤幅度更小"],
  momentum180: ["收益和回撤均优于180日动量", "净收益更高、回撤幅度更小"]
};
const finite = (value) => typeof value === "number" && Number.isFinite(value);
const pct = (value) => finite(value) ? `${(value * 100).toFixed(2)}%` : "—";
const number = (value, digits = 2) => finite(value) ? value.toLocaleString("zh-CN", { minimumFractionDigits: digits, maximumFractionDigits: digits }) : "—";
const pc = (field, label) => ({ field, label, renderCell: pct });
const nc = (field, label, digits = 2) => ({ field, label, renderCell: (value) => number(value, digits) });
const isModel = (row) => ["followup", "original_cap_reference"].includes(row.followup_role);
const isEnsemble = (row) => isModel(row) && row.primary_policy === true;
const rowRole = (row) => isModel(row) ? row.followup_role : row.strategy;
const roleName = (row) => ROLE[rowRole(row)] ?? row.strategy ?? row.run_id;
const displayRow = (row) => ({ ...row, _role: roleName(row), _cost: COST[row.scenario] });

function bandReady(summary, history, months, targets, choices, audits) {
  return summary.length === 38 && summary.filter(isEnsemble).length === 4
    && summary.filter((row) => isModel(row) && row.primary_policy === false && finite(row.seed)).length === 20
    && history.length > 0 && months.length > 0 && targets.length >= 5 && choices.length === 2
    && audits.length > 0 && audits.every((row) => row.passed === true);
}

function bandChartScope(summary, history, scenario) {
  const accounts = summary.filter((row) => row.scenario === scenario && (isEnsemble(row)
    || row.followup_role === "baseline_reference" && ["equal_hold", "cash"].includes(row.strategy)));
  const lookup = new Map(accounts.map((row) => [row.run_id, row]));
  return { accounts, lookup, sourceRows: history.filter((row) => lookup.has(row.run_id)) };
}

function bandPivot(rows, lookup, field) {
  const points = new Map();
  rows.forEach((row) => {
    const account = lookup.get(row.run_id);
    if (!account) return;
    if (!points.has(row.timestamp)) points.set(row.timestamp, { timestamp: row.timestamp, date: row.timestamp.replace(" ", "T") });
    points.get(row.timestamp)[`${roleName(account)}(%)`] = finite(row[field]) ? row[field] * 100 : null;
  });
  return [...points.values()].sort((a, b) => a.timestamp.localeCompare(b.timestamp));
}

function bandSpec(accounts, field) {
  const fields = accounts.map((row) => `${roleName(row)}(%)`);
  return { type: /** @type {"line"} */ ("line"), x: "date", xLabel: "日期（UTC）", y: fields[0] ?? "扣费收益(%)", fields,
    stackable: false, valueDecimals: 2, colors: Object.fromEntries(accounts.map((row) => [`${roleName(row)}(%)`, COLOR[rowRole(row)]])),
    legend: { labels: Object.fromEntries(fields.map((field) => [field, field.replace(/\(%\)$/u, "")])) },
    yLabel: field === "drawdown" ? "相对完整小时历史峰值（%）" : "累计扣费净收益（%）" };
}

function bandMonthRows(rows, lookup) {
  const months = new Map();
  rows.forEach((row) => {
    const account = lookup.get(row.run_id);
    if (!account) return;
    if (!months.has(row.month)) months.set(row.month, { month: row.month });
    months.get(row.month)[rowRole(account)] = finite(row.return) ? row.return : null;
  });
  return [...months.values()].sort((a, b) => a.month.localeCompare(b.month));
}

function bandGateRows(targets, summary) {
  const primary = (cost) => summary.find((row) => row.followup_role === "followup" && row.primary_policy === true && row.scenario === cost);
  const base = primary("base"), stress = primary("stress");
  const positives = summary.filter((row) => row.followup_role === "followup" && row.primary_policy === false && row.scenario === "base" && row.net_return > 0).length;
  return targets.map((row) => {
    const control = summary.find((item) => item.followup_role === "baseline_reference" && item.scenario === "base" && item.strategy === (row.gate === "beats_equal_hold_return_and_drawdown" ? "equal_hold" : row.gate));
    const observed = row.gate === "base_return_at_least_5pct" ? pct(base?.net_return)
      : row.gate === "base_max_drawdown_at_most_20pct" ? pct(finite(base?.max_drawdown) ? Math.abs(base.max_drawdown) : null)
      : row.gate === "stress_return_nonnegative" ? pct(stress?.net_return)
      : row.gate === "at_least_three_positive_training_seeds" ? `${positives}/5`
      : control ? `基准收益 ${pct(control.net_return)}；回撤 ${pct(control.max_drawdown)}` : "—";
    return { ...row, _label: GATES[row.gate]?.[0] ?? row.gate, _criterion: GATES[row.gate]?.[1] ?? "—", _observed: observed };
  });
}

function changeText(value, increase, decrease) {
  return finite(value) ? value === 0 ? "一致" : `${value > 0 ? increase : decrease}${Math.abs(value * 100).toFixed(2)}个百分点` : "缺少可比较证据";
}

export function BandFollowup() {
  const { snapshot, reviewedPeriodRows } = useDataApp();
  const [scenario, setScenario] = useState("base");
  if (!snapshot.report?.executionBand) return null;
  const summary = reviewedPeriodRows("execution_band_summary", { period: "history" });
  const history = reviewedPeriodRows("execution_band_equity", { period: "history" });
  const months = reviewedPeriodRows("execution_band_monthly", { period: "history" });
  const targets = reviewedPeriodRows("execution_band_targets", { period: "history" });
  const choices = reviewedPeriodRows("execution_band_validation_selection", { period: "history" });
  const audits = reviewedPeriodRows("execution_band_audit", { period: "history" });
  if (!bandReady(summary, history, months, targets, choices, audits)) return null;
  const original = reviewedPeriodRows("strategy_summary", { period: "history" });
  const sacRows = original.filter((row) => row.algorithm === "sac" && row.split === "test" && row.primary_policy === true && row.selected_profile === true);
  const completeSAC = sacRows.length === 2 && new Set(sacRows.map((row) => row.scenario)).size === 2;
  const ensembles = summary.filter(isEnsemble);
  const selected = ensembles.find((row) => row.followup_role === "followup" && row.scenario === scenario);
  const previous = ensembles.find((row) => row.followup_role === "original_cap_reference" && row.scenario === scenario);
  const diagnosticGates = targets.filter((row) => !["sma200_trend", "momentum180"].includes(row.gate));
  const met = diagnosticGates.filter((row) => row.met === true).length;
  const opening = `## 后续研究：把执行门槛固定为5个百分点\n\n本轮继续使用同一批五个收益与回撤平衡配置的TQC模型及75%币仓预算，只把再平衡门槛由2.5个百分点改为5个百分点。它在观察首轮年度结果后提出，再用验证295天比较新旧两个固定候选；随后回放的仍是 **2025-10-23至2026-10-23的已观察365天**，**不是新的盲测样本**。\n\n当前查看 **${COST[scenario]}**：新实际主组合扣费收益 **${pct(selected?.net_return)}**、小时最大回撤 **${pct(selected?.max_drawdown)}**；旧主组合分别为 **${pct(previous?.net_return)}**、**${pct(previous?.max_drawdown)}**。相对旧方案，净收益${changeText(selected?.net_return - previous?.net_return, "增加", "减少")}，回撤幅度${changeText(selected?.max_drawdown - previous?.max_drawdown, "缩小", "扩大")}。\n\n五项诊断目标达到 **${met}/5**；会计审计通过与研究效果达标分别判断，通过这些目标也不表示优于所有模型。首轮主方案、原208例排名与原失败目标完整保留。SAC在下方用原已审计结果单列，不参加这两个候选的选择或新模型目标判断。`;
  const chart = bandChartScope(summary, history, scenario);
  const selectedMonths = months.filter((row) => chart.lookup.has(row.run_id));
  const seeds = summary.filter((row) => isModel(row) && row.primary_policy === false && row.scenario === scenario);
  const gates = bandGateRows(targets, summary);
  const candidates = choices.map((row) => ({ ...row, _name: row.family === "cap075_band05" ? ROLE.followup : ROLE.original_cap_reference,
    _chosen: row.family === "cap075_band05" ? "验证胜出，随后独立回放" : "保留原执行规则对照",
    _baseReturn: row.base?.net_return, _baseDD: row.base?.max_drawdown, _baseFees: row.base?.fees,
    _stressReturn: row.stress?.net_return, _stressDD: row.stress?.max_drawdown, _stressFees: row.stress?.fees }));
  const monthRows = bandMonthRows(selectedMonths, chart.lookup);
  return <section aria-label="已观察年度的5个百分点调仓门槛独立跟进">
    <ReportSection id="btceth-band-opening" title="独立跟进结果" queryId="execution_band_summary" queryIds={["execution_band_summary", "execution_band_targets"]} sourceRowsByQuery={{ execution_band_summary: ensembles, execution_band_targets: targets }} showHeading={false}><RichNarrative id="btceth-band-opening:body" value={opening} /></ReportSection>
    <div className="btceth-controls" aria-label="跟进曲线、种子及月度的成本条件"><Dropdown label="跟进显示成本" showLabel value={scenario} choices={["base", "stress"]} choiceLabels={COST} onChange={setScenario} /></div>
    <DataComponent id="btceth-band-ensembles" title="新旧实际主组合：两档成本四个账本" queryId="execution_band_summary" sourceRows={ensembles} displayRows={ensembles.map(displayRow)} kind="table"><DataTable rows={ensembles.map(displayRow)} rowKey="run_id" searchable={false} compactNumbers={false} caption="固定展示两成本；每行是独立10,000 USDT真实平均目标账本，未平均五份最终收益" columns={[{ field: "_role", label: "方案" }, { field: "_cost", label: "成本" }, pc("net_return", "扣费净收益"), pc("max_drawdown", "小时最大回撤"), nc("sharpe_daily", "日Sharpe"), nc("calmar", "Calmar"), nc("fees", "手续费USDT"), nc("slippage_cost", "滑点USDT"), nc("turnover", "周转倍数"), pc("average_exposure", "平均币仓"), pc("net_liquidation_return", "扣期末退出成本后")]} /></DataComponent>
    {completeSAC && <DataComponent id="btceth-band-sac-control" title="首轮SAC同环境算法对照" queryId="strategy_summary" sourceRows={sacRows} displayRows={sacRows.map(displayRow)} kind="table"><DataTable rows={sacRows.map(displayRow)} rowKey="run_id" searchable={false} caption="直接引用首轮已审计两成本真实主组合，未重跑或参加本轮两候选选择；通过本轮目标不代表优于SAC" columns={[{ field: "_cost", label: "成本" }, pc("net_return", "SAC扣费净收益"), pc("max_drawdown", "SAC小时最大回撤"), nc("fees", "手续费USDT"), pc("average_exposure", "平均币仓")]} /></DataComponent>}
    <EvidenceChart id="btceth-band-return" title="当前成本：新旧主组合与现金、等权持有的累计扣费收益" queryId="execution_band_equity" sourceRows={chart.sourceRows} rows={bandPivot(chart.sourceRows, chart.lookup, "net_return")} spec={bandSpec(chart.accounts, "net_return")} height={350} />
    <EvidenceChart id="btceth-band-drawdown" title="当前成本：日边界相对完整小时峰值的回撤" queryId="execution_band_equity" sourceRows={chart.sourceRows} rows={bandPivot(chart.sourceRows, chart.lookup, "drawdown")} spec={bandSpec(chart.accounts, "drawdown")} height={270} />
    <DataComponent id="btceth-band-targets" title="五项诊断目标及两个简单趋势对照" queryId="execution_band_targets" queryIds={["execution_band_targets", "execution_band_summary"]} sourceRowsByQuery={{ execution_band_targets: targets, execution_band_summary: summary }} displayRows={gates} kind="table"><DataTable rows={gates} rowKey="gate" searchable={false} caption="固定365天的新主组合；不随成本下拉切换，不修改首轮失败目标，也不构成新盲测证据" columns={[{ field: "_label", label: "检查条件" }, { field: "category", label: "类别" }, { field: "_observed", label: "实际值／对照值" }, { field: "_criterion", label: "标准" }, { field: "met", label: "结果", renderCell: (value) => value === true ? "达到" : value === false ? "未达到" : "缺少证据" }]} /></DataComponent>
    <DataComponent id="btceth-band-seeds" title="当前成本：新旧方案全部五个训练种子" queryId="execution_band_summary" sourceRows={seeds} displayRows={seeds.map(displayRow)} kind="table"><DataTable rows={seeds.map(displayRow)} rowKey="run_id" searchable={false} caption="成本下拉可查看另一档全部种子；不隐藏亏损种子，不以最好种子代替实际主组合。五种子重复同一市场历史，不是独立市场样本" columns={[{ field: "_role", label: "执行规则" }, nc("seed", "训练种子", 0), pc("net_return", "扣费净收益"), pc("max_drawdown", "小时最大回撤"), nc("fees", "手续费USDT"), pc("average_exposure", "平均币仓")]} /></DataComponent>
    <DataComponent id="btceth-band-selection" title="295天验证：两个固定候选及实际评分" queryId="execution_band_validation_selection" sourceRows={choices} displayRows={candidates} kind="table"><DataTable rows={candidates} rowKey="family" searchable={false} caption="完整两候选；使用既定两成本收益与回撤评分，精确同分保留原规则。不是按这次年度结果再选" columns={[{ field: "_name", label: "执行方案" }, { field: "_chosen", label: "冻结选择" }, nc("score", "固定验证评分", 6), pc("_baseReturn", "主成本收益"), pc("_baseDD", "主成本小时回撤"), nc("_baseFees", "主成本手续费"), pc("_stressReturn", "压力成本收益"), pc("_stressDD", "压力成本小时回撤"), nc("_stressFees", "压力成本手续费")]} /></DataComponent>
    <DataComponent id="btceth-band-months" title="当前成本：新旧方案及两个基准的月度精确值" queryId="execution_band_monthly" sourceRows={selectedMonths} displayRows={monthRows} kind="table"><DataTable rows={monthRows} rowKey="month" searchable={false} caption="真实UTC首尾区间不补齐；月收益与每日曲线、期末权益均已对账" columns={[{ field: "month", label: "UTC月份" }, pc("followup", ROLE.followup), pc("original_cap_reference", ROLE.original_cap_reference), pc("equal_hold", ROLE.equal_hold), pc("cash", ROLE.cash)]} /></DataComponent>
    <DataComponent id="btceth-band-audit" title="独立跟进审计与原始来源身份" queryId="execution_band_audit" sourceRows={audits} displayRows={audits} kind="table"><DataTable rows={audits} rowKey="file" searchable={false} caption="新12账本＋12次精确重放；另独立核对原cap12与原基准14，引用不计作新试验" columns={[{ field: "role", label: "审计身份" }, nc("new_runs", "新账本", 0), nc("old_cap_reference_runs", "原cap引用", 0), nc("baseline_runs", "基准引用", 0), nc("repeat_count", "精确重放", 0), { field: "file", label: "实际来源文件" }, { field: "sha256", label: "SHA256", renderCell: (value) => <span className="btceth-hash">{value ?? "—"}</span> }]} /></DataComponent>
  </section>;
}

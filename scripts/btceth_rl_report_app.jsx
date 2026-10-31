import React, { useState } from "react";
import { DataComponent, DataTable, Dropdown, EvidenceChart, ReportSection, RichNarrative, useDataApp } from "../../data-app-public.jsx";
import { BandFollowup } from "./BandFollowup.jsx";
import { TD3Followup } from "./TD3Followup.jsx";
import { DroQFollowup } from "./DroQFollowup.jsx";
import { DirectComparison } from "./DirectComparison.jsx";
import "./btceth-rl-report.css";

const MODEL = { recurrent_ppo: "RecurrentPPO（LSTM）", tqc: "TQC", crossq: "CrossQ", sac: "SAC同环境对照", tqc_cap075: "TQC75%币仓预算" };
const RISK = { btc25_cash75_hold: "初始25%BTC＋现金", btceth25_cash75_hold: "初始各12.5%BTC/ETH＋现金", vol20_inverse_vol30: "20%年化波动目标组合" };
const SPLIT = { test: "测试365天", validation: "验证295天" };
const COST = { base: "主成本：费0.10%＋滑点0.02%", stress: "压力成本：费0.20%＋滑点0.05%" };
const PROFILE = { return: "净收益奖励", balanced: "收益＋回撤惩罚" };
const BASELINE = { cash: "USDT现金", btc_hold: "BTC被动持有", eth_hold: "ETH被动持有", equal_hold: "BTC/ETH等权被动持有", equal_rebalance: "BTC/ETH等权再平衡", sma200_trend: "SMA200趋势对照", momentum180: "180日动量对照" };
const SERIES_COLORS = { recurrent_ppo: "#15803d", tqc: "#d97706", crossq: "#2563eb", sac: "#db2777", tqc_cap075: "#7c3aed", cash: "#64748b", btc_hold: "#ea580c", eth_hold: "#9333ea", equal_hold: "var(--text)", equal_rebalance: "#0891b2", sma200_trend: "#0f766e", momentum180: "#be123c", btc25_cash75_hold: "#b45309", btceth25_cash75_hold: "#4f46e5", vol20_inverse_vol30: "#047857" };
const SEED_COLORS = ["#4c1d95", "#0e7490", "#b45309", "#b91c1c", "#4d7c0f"];
const valid = (value) => typeof value === "number" && Number.isFinite(value);
const pct = (value) => valid(value) ? `${(value * 100).toFixed(2)}%` : "—";
const num = (value, digits = 2) => valid(value) ? value.toLocaleString("zh-CN", { minimumFractionDigits: digits, maximumFractionDigits: digits }) : "—";
const pc = (field, label) => ({ field, label, renderCell: pct });
const nc = (field, label, digits = 2) => ({ field, label, renderCell: (value) => num(value, digits) });
const isPrimary = (row) => row.primary_policy === true && row.algorithm !== "baseline";
const isBaseline = (row) => row.algorithm === "baseline";
const isSeed = (row) => !isPrimary(row) && !isBaseline(row) && valid(row.seed);
const displayName = (row) => row.scope === "independent_risk_budget_reference" ? RISK[row.strategy] ?? row.name : isBaseline(row) ? BASELINE[row.strategy] ?? row.label : `${MODEL[row.algorithm] ?? row.algorithm}${isPrimary(row) ? "主组合" : ` seed ${row.seed}`}`;
const seriesColor = (row) => isSeed(row) ? SEED_COLORS[row.seed] : SERIES_COLORS[isBaseline(row) || row.scope === "independent_risk_budget_reference" ? row.strategy : row.algorithm];

function pivot(rows, field) {
  const output = new Map();
  rows.forEach((row) => {
    // ISO display alias lets the public renderer infer UTC date ticks; Sources retains the original timestamp.
    if (!output.has(row.timestamp)) output.set(row.timestamp, { timestamp: row.timestamp, date: row.timestamp.replace(" ", "T") });
    output.get(row.timestamp)[`${displayName(row)}(%)`] = valid(row[field]) ? row[field] * 100 : null;
  });
  return [...output.values()].sort((left, right) => left.timestamp.localeCompare(right.timestamp));
}

function curveSpec(rows, field) {
  const fields = [...new Set(rows.map((row) => `${displayName(row)}(%)`))];
  return { type: /** @type {"line"} */ ("line"), x: "date", xLabel: "日期（UTC）", y: fields[0] ?? "收益(%)", fields, stackable: false, valueDecimals: 2,
    colors: Object.fromEntries(rows.map((row) => [`${displayName(row)}(%)`, seriesColor(row)])),
    legend: { labels: Object.fromEntries(fields.map((name) => [name, name.replace(/\(%\)$/u, "")])) },
    yLabel: field === "drawdown" ? "相对小时历史峰值（%）" : "累计扣费净收益（%）" };
}

export function ReportContent() {
  const { appTitle, setAppTitle, canEdit, mode } = useDataApp();
  const [page, setPage] = useState("compare");
  return <article className="report-content btceth-rl-report btceth-comparison-report" aria-label="BTC与ETH回测直接比较">
    <header className="report-hero"><h1 data-data-app-title contentEditable={canEdit && mode === "edit"} suppressContentEditableWarning onBlur={canEdit && mode === "edit" ? event => setAppTitle(event.currentTarget.textContent?.trim() || appTitle) : undefined}>{appTitle}</h1><RichNarrative id="btceth:deck" className="report-deck" value="BTCUSDT / ETHUSDT / USDT · 2025-10-23—2026-10-23 UTC · 365天 · 每份本金10,000 USDT"/></header>
    <nav className="compare-page-nav" aria-label="报告内容"><button aria-current={page === "compare" ? "page" : undefined} onClick={() => setPage("compare")}>结果对比</button><button aria-current={page === "research" ? "page" : undefined} onClick={() => setPage("research")}>研究档案</button></nav>
    {page === "compare" ? <DirectComparison/> : <ResearchDetails/>}
  </article>;
}

function ResearchDetails() {
  const { snapshot, reviewedPeriodRows, appTitle, setAppTitle, canEdit, mode } = useDataApp();
  const summary = reviewedPeriodRows("strategy_summary", { period: "history" });
  const history = reviewedPeriodRows("equity_history", { period: "history" });
  const months = reviewedPeriodRows("monthly_returns", { period: "history" });
  const statistics = reviewedPeriodRows("seed_statistics", { period: "history" });
  const candidates = reviewedPeriodRows("validation_selection", { period: "history" });
  const training = reviewedPeriodRows("training_records", { period: "history" });
  const targets = reviewedPeriodRows("research_targets", { period: "history" });
  const audits = reviewedPeriodRows("audit_evidence", { period: "history" });
  const report = snapshot.report ?? {};
  const hasFollowup = Boolean(report.executionBand || report.td3Followup || report.droqFollowup);
  const extension = report.extension;
  const hasRisk = report.riskBudgetControls?.audit?.passed === true;
  const riskSummary = hasRisk ? reviewedPeriodRows("risk_budget_summary", { period: "history" }) : [];
  const riskHistory = hasRisk ? reviewedPeriodRows("risk_budget_equity", { period: "history" }) : [];
  const riskMonths = hasRisk ? reviewedPeriodRows("risk_budget_monthly", { period: "history" }) : [];
  /** @type {{timesteps_per_model?: number, train_episode_days?: number}} */
  const config = report.studyConfig ?? {};
  /** @type {{algorithm?: string, profile?: string}} */
  const leader = report.primaryLeader ?? {};
  /** @type {{requested_train_days?: number, eligible_windows?: number, covered_days?: number}} */
  const calendar = report.trainingCalendar ?? {};
  /** @type {{all_predeclared_primary_targets_met?: boolean}} */
  const researchTargets = report.researchTargets ?? {};
  const [split, setSplit] = useState("test"), [scenario, setScenario] = useState("base");
  const [curves, setCurves] = useState("primary_controls"), [monthChoice, setMonthChoice] = useState("");
  const selected = summary.filter((row) => row.split === split && row.scenario === scenario && row.selected_profile);
  const primary = selected.filter(isPrimary), baselines = selected.filter(isBaseline), seeds = selected.filter(isSeed);
  const primaryLeader = primary.find((row) => row.algorithm === leader.algorithm && row.profile === leader.profile);
  const passive = baselines.find((row) => row.strategy === "equal_hold");
  const stressLeader = summary.find((row) => row.split === "test" && row.scenario === "stress" && row.algorithm === leader.algorithm && row.profile === leader.profile && isPrimary(row));
  const fixedBase = summary.find((row) => row.split === "test" && row.scenario === "base" && row.algorithm === leader.algorithm && row.profile === leader.profile && isPrimary(row));
  const testPrimary = summary.filter((row) => row.split === "test" && row.scenario === "base" && row.selected_profile && isPrimary(row));
  const positive = testPrimary.filter((row) => row.net_return > 0).length;
  const ledgerRows = [...primary, ...baselines].map((row) => ({ ...row, _name: displayName(row), _role: isPrimary(row) ? row.algorithm === leader.algorithm ? extension ? "验证期选定新模型主方案" : "原全算法主方案" : row.algorithm === "sac" ? "原算法SAC对照" : "其他模型对照" : "固定规则基准", _cash: 1 - row.average_exposure, _profile: isBaseline(row) ? "—" : PROFILE[row.profile] }));
  const targetMet = researchTargets.all_predeclared_primary_targets_met === true;
  const inactiveNote = primaryLeader && primaryLeader.average_exposure < .005 ? "该主方案平均仓位低于0.5%，当前结果主要反映持现金；不能据模型名称认定学到了有效交易。" : "";
  const originalChoices = extension ? `\n\n扩展主方案来自测试启动前封存的8个新模型候选：6个原规则候选与2个TQC75%币仓预算候选。另保留2个SAC配置作对照，不参与这8选1。原全算法主方案 **${MODEL[report.originalGlobalLeader?.algorithm] ?? "—"} · ${PROFILE[report.originalGlobalLeader?.profile] ?? "—"}** 与原新模型主方案 **${MODEL[report.rawNewLeader?.algorithm] ?? "—"} · ${PROFILE[report.rawNewLeader?.profile] ?? "—"}** 均保留；后者的原条件${report.rawNewModelTargets?.goal_performance_achieved === true ? "全部达到" : "尚未全部达到"}，未借用SAC结果替代。75%预算是在观察验证表现后提出，未重训模型、未按测试结果改选。` : "";
  const opening = `## ${hasFollowup ? "首轮结果摘要（原2.5个百分点门槛）" : "结果摘要"}\n\n事前只用验证期选出的主方案是 **${MODEL[leader.algorithm ?? ""] ?? "—"} · ${PROFILE[leader.profile ?? ""] ?? "—"}**。当前查看 **${SPLIT[split]} · ${COST[scenario]}**：主方案扣费净收益 **${pct(primaryLeader?.net_return)}**，小时最大回撤 **${pct(primaryLeader?.max_drawdown)}**，平均仓位 **${pct(primaryLeader?.average_exposure)}**。${inactiveNote}\n\n同窗口BTC/ETH等权被动持有收益 **${pct(passive?.net_return)}**，回撤 **${pct(passive?.max_drawdown)}**。本轮资产固定为 **BTCUSDT、ETHUSDT与USDT现金**，所有主组合与基准各自使用一个10,000 USDT账本。\n\n固定365天测试中，验证期主方案的主成本收益为 **${pct(fixedBase?.net_return)}**、回撤 **${pct(fixedBase?.max_drawdown)}**，压力成本收益为 **${pct(stressLeader?.net_return)}**。已评估的${testPrimary.length}个模型主组合中有 **${positive}/${testPrimary.length}** 主成本净收益为正；主方案${targetMet ? "达到" : "尚未同时达到"}预先声明的收益、回撤、成本压力及种子稳定性条件，具体逐项列在下方。${originalChoices}\n\nSAC是同环境算法对照；每种算法保留全部五个训练种子，主组合对五个目标权重取平均后实际成交。其他模型的测试排名只作描述，不替换验证期主方案。该测试年度已被此前研究观察，不是全新盲测样本。`;
  const selectedHistory = history.filter((row) => row.split === split && row.scenario === scenario && row.selected_profile);
  const availableAlgorithms = [...new Set(primary.map((row) => row.algorithm))];
  const curveLabels = { primary_controls: "实际主组合＋现金及简单对照", baselines: "七种固定规则基准", ...Object.fromEntries(availableAlgorithms.map((value) => [value, `${MODEL[value]}五种子＋实际主组合`])) };
  const activeCurves = availableAlgorithms.includes(curves) || ["primary_controls", "baselines"].includes(curves) ? curves : "primary_controls";
  const curveRows = selectedHistory.filter((row) => activeCurves === "primary_controls"
    ? isPrimary(row) || ["cash", "equal_hold", "sma200_trend", "momentum180"].includes(row.strategy)
    : activeCurves === "baselines" ? isBaseline(row) : row.algorithm === activeCurves || row.strategy === "equal_hold");
  const selectedStatistics = statistics.filter((row) => row.split === split && row.scenario === scenario && row.selected_profile);
  const statsRows = selectedStatistics.map((row) => ({ ...row, _name: MODEL[row.algorithm], _profile: PROFILE[row.profile], _key: `${row.algorithm}:${row.profile}` }));
  const monthlyChoices = [...primary, ...baselines, ...seeds];
  const chosen = monthlyChoices.find((row) => row.run_id === monthChoice) ?? primaryLeader ?? monthlyChoices[0];
  const selectedMonths = months.filter((row) => row.run_id === chosen?.run_id);
  const candidateRows = candidates.map((row) => ({ ...row, _key: `${row.algorithm}:${row.profile}`, _name: MODEL[row.algorithm], _profile: PROFILE[row.profile], _decision: row.primary_leader ? extension ? "冻结新模型主方案" : "原全算法主方案" : row.eligible_for_primary === false ? "SAC对照，不参加8选1" : row.original_raw_new_leader ? "保留原新模型主方案" : row.selected ? "本算法选中" : "保留验证证据", _original: row.original_global_leader ? "原全算法主方案" : row.original_raw_new_leader ? "原新模型主方案" : "—" }));
  const riskRows = riskSummary.map((row) => ({ ...row, _name: RISK[row.strategy], _cost: COST[row.scenario], _cash: 1 - row.average_exposure, _positiveMonths: riskMonths.filter((m) => m.run_id === row.run_id && m.return > 0).length, _observedMonths: riskMonths.filter((m) => m.run_id === row.run_id).length }));
  const riskCurves = riskHistory.filter((row) => row.scenario === scenario);
  const trainingRows = training.map((row) => ({ ...row, _name: MODEL[row.algorithm], _profile: PROFILE[row.profile] }));
  const trainingText = `## ${hasFollowup ? "首轮" : ""}训练、验证与研究边界\n\n训练为 **2020-01-01至2025-01-01**，名义 **${num(calendar.requested_train_days, 0)}天**；验证为2025-01-01至2025-10-23，共295天；测试为2025-10-23至2026-10-23，共365天，全部UTC、右端不含。\n\n每个算法有净收益奖励与加入回撤惩罚的两组训练配置，各训练五个种子，共40个模型，每模型 **${num(config.timesteps_per_model, 0)}步**。训练从完整 **${num(config.train_episode_days, 0)}日**片段抽样；${num(calendar.eligible_windows, 0)}个可用窗口的日期并集覆盖 **${num(calendar.covered_days, 0)}天**。这表示可供抽样的真实日期覆盖，不代表每个种子都遍历了全部日期，也不把不同种子当作独立市场样本。\n\n历史缺小时不插值。日价特征读取当天23UTC真实收盘价；交易日小时价格、当日成交量及其滚动窗口按冻结规则验证，跨缺口或特征不完整的训练片段排除。验证和测试连续完整，不删除亏损日或缺行情日。特征使用固定因果缩放，无全年度拟合归一化。\n\n训练在双V100上按物理GPU隔离进程，每模型单卡运行；验证只比较固定的两组配置，按事前定义的主／压力成本收益回撤评分选每算法配置及总主方案。测试不重训、不挑最好种子、不再按测试表现更换主方案。本轮未使用Kronos、Chronos或其他预训练预测模型。`;
  const methodText = `## ${hasFollowup ? "首轮" : ""}成交、费用与指标口径\n\n每个独立账本初始10,000 USDT，现货、无借币和杠杆，不使用BNB抵扣。模型控制总币仓及ETH占比，非正总仓位动作可选择全现金；目标总币仓最多95%、单币最多95%，余款留USDT。五模型目标权重取平均后由一个账本执行，不平均五份最终收益。目标权重与当前权重差异不足2.5个百分点时不再平衡，全现金退出按冻结规则执行；实际比例可随价格漂移。\n\n每日00UTC使用此前已完成行情产生目标；00小时开盘先卖，01小时开盘再用净现金买，两腿分别按taker扣费。这与此前RL小时执行口径一致，与早期规则策略的下一分钟成交口径不同。主成本单边手续费0.10%＋不利滑点0.02%；压力为0.20%＋0.05%，均是研究假设。两档评估复用同一模型，分别运行资金和持币账本。\n\n小时收盘权益和初始本金用于最大回撤；日末曲线只是展示采样，不能据图中最低点替代小时最大回撤，也不捕捉小时内更深跌幅。日收益Sharpe无风险利率设0。期末保留真实持仓估值，另列扣一次退出成本的净清算收益；小数币模拟未重建历史盘口、数量步长和最小下单金额。\n\n七个基准为现金、BTC持有、ETH持有、BTC/ETH等权被动持有、等权再平衡、SMA200趋势和180日动量，费用、执行与估值口径一致。持有基准初始买入后持有，趋势基准有自己的固定买卖规则。会计审计通过仅表示结果和账本一致，不能替代是否达到研究效果的判断。`;

  const extensionMethods = extension ? `\n\n预算扩展对每个TQC模型原始映射的BTC、ETH目标分别乘0.75，剩余权重留现金，单次目标总币仓上限为71.25%；成交后实际比例可随价格漂移。五个模型读取同一真实组合状态，各自保留原训练参数；平均缩减后的目标，由一个新账本重新执行。该结果来自实际推理和成交，未按比例缩放原净值。原172例保持原身份；预算扩展独立审计36例，其中${extension.rawTestReferencesReused ? "12个测试例引用原结果，不额外计入唯一案例" : "12个测试例使用自己的真实账本"}。` : "";
  const officialMethods = "\n\n实际实现锁定SB3/SB3-Contrib 2.9.0：[RecurrentPPO](https://sb3-contrib.readthedocs.io/en/v2.9.0/modules/ppo_recurrent.html)、[TQC](https://sb3-contrib.readthedocs.io/en/v2.9.0/modules/tqc.html)、[CrossQ](https://sb3-contrib.readthedocs.io/en/v2.9.0/modules/crossq.html)、[SAC](https://stable-baselines3.readthedocs.io/en/v2.9.0/modules/sac.html)。LSTM推理跨交易日保留状态，片段边界重置。官方其他环境的基准表现不构成本次币市收益证据。";
  return <article className="report-content btceth-rl-report" aria-label="固定BTC与ETH现货强化学习研究报告">
    <DroQFollowup />
    <TD3Followup />
    <BandFollowup />
    <ReportSection id="btceth-summary" title="结果摘要" queryId="strategy_summary" queryIds={["strategy_summary", "research_targets"]} sourceRowsByQuery={{ strategy_summary: summary.filter((row) => row.selected_profile), research_targets: targets }} showHeading={false}><RichNarrative id="btceth-summary:body" value={opening} /></ReportSection>
    <div className="btceth-controls" aria-label="评估显示条件"><Dropdown label="评估窗口" showLabel value={split} choices={["test", "validation"]} choiceLabels={SPLIT} onChange={setSplit} /><Dropdown label="成本情景" showLabel value={scenario} choices={["base", "stress"]} choiceLabels={COST} onChange={setScenario} /><Dropdown label="曲线比较" showLabel value={availableAlgorithms.includes(curves) || ["primary_controls", "baselines"].includes(curves) ? curves : "primary_controls"} choices={["primary_controls", "baselines", ...availableAlgorithms]} choiceLabels={curveLabels} onChange={setCurves} /></div>
    <EvidenceChart id="btceth-return" title="同一资金口径的累计扣费净收益" queryId="equity_history" sourceRows={curveRows} rows={pivot(curveRows, "net_return")} spec={curveSpec(curveRows, "net_return")} height={410} />
    <EvidenceChart id="btceth-drawdown" title="日末相对完整小时历史峰值的回撤" queryId="equity_history" sourceRows={curveRows} rows={pivot(curveRows, "drawdown")} spec={curveSpec(curveRows, "drawdown")} height={310} />
    <DataComponent id="btceth-main" title={hasFollowup ? "首轮模型主组合与七个基准" : "实际模型主组合与七个基准"} queryId="strategy_summary" sourceRows={[...primary, ...baselines]} displayRows={ledgerRows} kind="table"><DataTable rows={ledgerRows} rowKey="run_id" searchable={false} compactNumbers={false} caption="每行各自使用10,000 USDT，主组合不挑选测试最好种子" columns={[{ field: "_name", label: "模型／基准" }, { field: "_role", label: "比较角色" }, { field: "_profile", label: "验证所选配置" }, pc("net_return", "扣费净收益"), pc("max_drawdown", "小时最大回撤"), nc("sharpe_daily", "日Sharpe"), nc("calmar", "Calmar"), nc("fees", "手续费USDT"), nc("slippage_cost", "滑点USDT"), nc("turnover", "周转倍数"), pc("average_exposure", "平均币仓"), pc("_cash", "平均现金"), pc("net_liquidation_return", "扣期末退出成本后")]} /></DataComponent>
    <ReportSection id="btceth-targets-heading" title="固定测试期的研究条件" queryId="research_targets" sourceRows={targets} showHeading={false}><RichNarrative id="btceth-targets:body" value={`## ${hasFollowup ? "首轮验证期主方案是否达到研究条件" : "验证期主方案是否达到研究条件"}\n\n下表始终对应 **${MODEL[leader.algorithm ?? ""] ?? "—"} · ${PROFILE[leader.profile ?? ""] ?? "—"}的固定365天测试**，不随上方验证／成本显示切换。五项事前研究条件${targetMet ? "全部达到" : "尚未全部达到"}；另列两种简单趋势对照，避免仅超过被动持有就宣称模型有优势。`} /></ReportSection>
    <DataComponent id="btceth-targets" title="收益、回撤、成本压力与稳定性逐项核对" queryId="research_targets" sourceRows={targets} displayRows={targets} kind="table"><DataTable rows={targets} rowKey="gate" searchable={false} caption="固定测试主方案达标检查，与审计通过分开" columns={[{ field: "label", label: "检查条件" }, { field: "category", label: "类别" }, { field: "observed", label: "实际值／基准值" }, { field: "criterion", label: "比较标准" }, { field: "met", label: "结果", renderCell: (value) => value === true ? "达到" : value === false ? "未达到" : "缺少证据" }]} /></DataComponent>
    <ReportSection id="btceth-seeds-heading" title="五次训练的结果分散程度" queryId="seed_statistics" sourceRows={selectedStatistics} showHeading={false}><RichNarrative id="btceth-seeds:body" value="## 主组合之外，保留全部五次训练\n\n区间描述同一市场历史上五次独立训练的结果变化，不是独立市场置信区间，也不用于挑选最好测试种子。主组合收益来自真实平均目标权重账本，与五份最终收益的均值不同。" /></ReportSection>
    <DataComponent id="btceth-seed-ranges" title="验证所选配置的五种子结果区间" queryId="seed_statistics" sourceRows={selectedStatistics} displayRows={statsRows} kind="table"><DataTable rows={statsRows} rowKey="_key" searchable={false} caption="当前窗口及成本的五种子分散程度" columns={[{ field: "_name", label: "算法" }, { field: "_profile", label: "配置" }, pc("worst_return", "最差净收益"), pc("median_return", "净收益中位数"), pc("best_return", "最好净收益"), pc("worst_drawdown", "最深小时回撤"), pc("best_drawdown", "最浅小时回撤"), nc("positive_seeds", "正收益种子数", 0)]} /></DataComponent>
    <DataComponent id="btceth-seed-table" title="全部训练种子的实际账本结果" queryId="strategy_summary" sourceRows={seeds} displayRows={seeds.map((row) => ({ ...row, _name: displayName(row) }))} kind="table"><DataTable rows={seeds.map((row) => ({ ...row, _name: displayName(row) }))} rowKey="run_id" searchable={false} caption="不隐藏亏损种子" columns={[{ field: "_name", label: "模型与种子" }, pc("net_return", "扣费净收益"), pc("max_drawdown", "小时最大回撤"), nc("fees", "费用USDT"), pc("average_exposure", "平均币仓")]} /></DataComponent>
    <EvidenceChart id="btceth-monthly" title="所选组合的月度扣费收益" queryId="monthly_returns" sourceRows={selectedMonths} rows={selectedMonths.map((row) => ({ ...row, "月度净收益(%)": valid(row.return) ? row.return * 100 : null }))} spec={{ type: "bar", x: "month", y: "月度净收益(%)", stackable: false, valueDecimals: 2 }} height={300} headerControls={<Dropdown label="月度查看组合" showLabel value={chosen?.run_id ?? ""} choices={monthlyChoices.map((row) => row.run_id)} choiceLabels={Object.fromEntries(monthlyChoices.map((row) => [row.run_id, displayName(row)]))} onChange={setMonthChoice} />} />
    <DataComponent id="btceth-month-table" title="所选组合月度精确数值" queryId="monthly_returns" sourceRows={selectedMonths} displayRows={selectedMonths} kind="table"><DataTable rows={selectedMonths} rowKey="month" searchable={false} caption={displayName(chosen ?? {}) + "：实际区间首尾月份不补齐"} columns={[{ field: "month", label: "UTC月份" }, pc("return", "扣费净收益")]} /></DataComponent>
    <ReportSection id="btceth-selection-heading" title="只使用验证期选择配置" queryId="validation_selection" sourceRows={candidates} showHeading={false}><RichNarrative id="btceth-selection:body" value={extension ? "## 新模型主方案如何事前选出\n\n完整保留8个新模型候选及2个SAC配置对照。6个原规则新候选与2个TQC75%预算候选使用同一固定评分；精确同分优先原规则，再按算法及配置标识排序。SAC不参与这8选1，原全算法选择保留。评分同时使用295天验证主／压力成本收益及回撤；五模型平均目标的真实账本为评分单位。预算在观察验证表现后提出；原模型、预算和8候选主方案均在测试启动前封存，未通过测试排名再选。" : "## 主方案如何事前选出\n\n每种算法只比较预先固定的两组奖励配置；下表完整保留八个候选。评分同时使用验证期主成本和压力成本，惩罚回撤以及超过20%的回撤；五种子平均目标组合是评分单位。总主方案及每算法所选配置在测试启动前冻结，测试排名只作描述。"} /></ReportSection>
    <DataComponent id="btceth-selection" title={extension ? "八个新候选、两个SAC对照与冻结选择" : "八个验证候选与冻结选择"} queryId="validation_selection" sourceRows={candidates} displayRows={candidateRows} kind="table"><DataTable rows={candidateRows} rowKey="_key" searchable={false} caption="验证295天的选模证据，不读取测试排名" columns={[{ field: "_name", label: "算法" }, { field: "_profile", label: "配置" }, { field: "_decision", label: "冻结选择" }, { field: "_original", label: "保留原身份" }, nc("score", "固定验证评分", 4), pc("base_return", "主成本收益"), pc("base_drawdown", "主成本小时回撤"), pc("stress_return", "压力成本收益"), pc("stress_drawdown", "压力成本小时回撤")]} /></DataComponent>
    <ReportSection id="btceth-training-heading" title="实际训练覆盖" queryId="training_records" sourceRows={training} showHeading={false}><RichNarrative id="btceth-training:body" value={trainingText} /></ReportSection>
    <DataComponent id="btceth-training" title={hasFollowup ? "首轮40个训练模型的冻结记录" : "全部40个训练模型的冻结记录"} queryId="training_records" sourceRows={training} displayRows={trainingRows} kind="table"><DataTable rows={trainingRows} rowKey="job_id" searchable caption="训练预算、实际覆盖与模型SHA" columns={[{ field: "_name", label: "算法" }, { field: "_profile", label: "奖励配置" }, nc("seed", "种子", 0), nc("timesteps", "训练步数", 0), nc("episode_days", "片段日数", 0), nc("covered_training_days", "可抽样覆盖天数", 0), nc("elapsed_seconds", "训练秒数", 1), { field: "gpu", label: "实际GPU" }, { field: "model_sha256", label: "模型SHA256", renderCell: (value) => <span className="btceth-hash">{value ?? "—"}</span> }]} /></DataComponent>
    {hasRisk && <>
      <ReportSection id="btceth-risk-heading" title="独立风险预算对照" queryId="risk_budget_summary" sourceRows={riskSummary} showHeading={false}><RichNarrative id="btceth-risk:body" value="## 低仓位本身能解释多少收益与回撤\n\n以下3个预先固定的控制各运行主／压力两档成本，使用独立10,000 USDT账本。不加入原172例排名、预算扩展36例或8候选选择。25%为初始币仓，随后持有库存并随价格漂移；20%为基于此前30个完整日收益协方差的年化波动目标，不保证实际波动为20%，也不表示与模型风险完全匹配。每日因果调仓、费用和全小时回撤均已独立对账。表格固定展示完整365天，两张曲线跟随上方成本切换，不跟随验证窗口切换。" /></ReportSection>
      <DataComponent id="btceth-risk-table" title="三种风险预算控制的六个真实账本" queryId="risk_budget_summary" sourceRows={riskSummary} displayRows={riskRows} kind="table"><DataTable rows={riskRows} rowKey="run_id" searchable={false} caption="独立对照，未参与选模与模型排名" columns={[{ field: "_name", label: "风险预算控制" }, { field: "_cost", label: "成本" }, pc("net_return", "扣费净收益"), pc("max_drawdown", "小时最大回撤"), nc("fees", "手续费USDT"), pc("average_exposure", "全小时平均币仓"), pc("_cash", "平均现金"), nc("_positiveMonths", "正收益月份", 0), nc("_observedMonths", "实际月份", 0)]} /></DataComponent>
      <EvidenceChart id="btceth-risk-return" title="风险预算对照：365天累计扣费收益" queryId="risk_budget_equity" sourceRows={riskCurves} rows={pivot(riskCurves, "net_return")} spec={curveSpec(riskCurves, "net_return")} height={330} />
      <EvidenceChart id="btceth-risk-drawdown" title="风险预算对照：日末相对小时峰值回撤" queryId="risk_budget_equity" sourceRows={riskCurves} rows={pivot(riskCurves, "drawdown")} spec={curveSpec(riskCurves, "drawdown")} height={270} />
      <DataComponent id="btceth-risk-months" title="风险预算对照的月度精确值" queryId="risk_budget_monthly" sourceRows={riskMonths} displayRows={riskMonths.map((row) => ({ ...row, _key: `${row.run_id}:${row.month}`, _name: RISK[row.strategy], _cost: COST[row.scenario] }))} kind="table"><DataTable rows={riskMonths.map((row) => ({ ...row, _key: `${row.run_id}:${row.month}`, _name: RISK[row.strategy], _cost: COST[row.scenario] }))} rowKey="_key" searchable caption="月份使用真实窗口，首尾月不补齐" columns={[{ field: "_name", label: "控制" }, { field: "_cost", label: "成本" }, { field: "month", label: "UTC月份" }, pc("return", "扣费收益")]} /></DataComponent>
    </>}
    <ReportSection id="btceth-methods" title="交易和指标口径" queryId="strategy_summary" sourceRows={summary} showHeading={false}><RichNarrative id="btceth-methods:body" value={methodText + extensionMethods + officialMethods} /></ReportSection>
    <DataComponent id="btceth-audit" title="最终独立审计证据" queryId="audit_evidence" sourceRows={audits} displayRows={audits} kind="table"><DataTable rows={audits} searchable={false} caption={extension ? "原172例与独立扩展36例分别审计；raw引用不会重复计成新账本。会计通过与研究目标达成分开" : "完整172个实际账本的独立核对，未以通过审计冒充策略盈利"} columns={[{ field: "audit_role", label: "审计范围" }, { field: "status", label: "审计状态" }, nc("runs_checked", "核对账本数", 0), nc("checks_total", "检查数", 0), nc("failures", "失败数", 0), { field: "audit_completed_at", label: "审计完成UTC" }, { field: "dataset_sha256", label: "行情SHA256", renderCell: (value) => <span className="btceth-hash">{value ?? "—"}</span> }]} /></DataComponent>
  </article>;
}

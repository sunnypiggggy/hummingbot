import React, { useState } from "react";
import { DataComponent, DataTable, Dropdown, EvidenceChart, ReportSection, RichNarrative, useDataApp } from "../../data-app-public.jsx";
import "./rl-framework-report.css";

const COST = { base: "主成本", stress: "压力成本" };
const SPLIT = { test: "测试365天", validation: "验证90天", val: "验证90天" };
const CURVES = { ensemble: "PPO／A2C主组合与四个基准", PPO: "PPO全部种子与主组合", A2C: "A2C全部种子与主组合" };
const valid = (v) => typeof v === "number" && Number.isFinite(v);
const pct = (v, digits = 2) => valid(v) ? `${(v * 100).toFixed(digits)}%` : "—";
const num = (v, digits = 2) => valid(v) ? v.toLocaleString("zh-CN", { minimumFractionDigits: digits, maximumFractionDigits: digits }) : "—";
const pc = (field, label) => ({ field, label, renderCell: (value) => pct(value) });
const nc = (field, label, digits = 2) => ({ field, label, renderCell: (value) => num(value, digits) });
const isSeed = (r) => r.seed !== null && r.seed !== undefined && valid(Number(r.seed));
const isEnsemble = (r) => ["ppo_ensemble", "a2c_ensemble"].includes(r.strategy);
const algorithm = (r) => r.strategy?.startsWith("a2c") ? "A2C" : r.strategy?.startsWith("ppo") ? "PPO" : "共用基准";
const seriesName = (r) => isEnsemble(r) ? `${algorithm(r)}五模型平均权重组合` : isSeed(r) ? `${algorithm(r)}种子${r.seed}` : r.label;
const display = (rows) => rows.map((r) => ({ ...r, _algorithm: algorithm(r), _label: seriesName(r) }));
const first = (row, keys, fallback = null) => { for (const key of keys) if (row[key] !== undefined && row[key] !== null) return row[key]; return fallback; };
const median = (values) => { const v = values.filter(valid).sort((a, b) => a - b); return v.length ? v.length % 2 ? v[(v.length - 1) / 2] : (v[v.length / 2 - 1] + v[v.length / 2]) / 2 : null; };

function pivot(rows, measure) {
  const data = new Map();
  rows.forEach((row) => { if (!data.has(row.timestamp)) data.set(row.timestamp, { timestamp: row.timestamp }); data.get(row.timestamp)[`${seriesName(row)}(%)`] = valid(row[measure]) ? row[measure] * 100 : null; });
  return [...data.values()].sort((a, b) => a.timestamp.localeCompare(b.timestamp));
}
function spec(rows, measure, curveMode) {
  const labels = [...new Set(rows.map(seriesName))];
  const fields = labels.map((label) => `${label}(%)`);
  const colors = Object.fromEntries(labels.map((label) => {
    const row = rows.find((r) => seriesName(r) === label);
    const color = isSeed(row) ? `var(--chart-${Number(row.seed) % 5 + 1})`
      : row.strategy === "ppo_ensemble" ? curveMode === "ensemble" ? "var(--chart-1)" : "var(--text)"
      : row.strategy === "a2c_ensemble" ? curveMode === "ensemble" ? "var(--chart-2)" : "var(--text)"
      : row.strategy === "top10_cap" ? "#797979" : row.strategy === "btc_eth_hold" ? "#a7a7a7"
      : row.strategy === "cash" ? "var(--secondary)" : "var(--text)";
    return [`${label}(%)`, color];
  }));
  return { type: "line", x: "timestamp", y: fields[0] ?? "净收益(%)", fields, colors, stackable: false, valueDecimals: 2,
    legend: { labels: Object.fromEntries(fields.map((field) => [field, field.replace(/\(%\)$/u, "")])) },
    yLabel: measure === "drawdown" ? "距小时历史峰值（%）" : "扣费累计净收益（%）" };
}

export function ReportContent() {
  const { snapshot, reviewedPeriodRows, appTitle, setAppTitle, canEdit, mode } = useDataApp();
  const summary = reviewedPeriodRows("strategy_summary", { period: "history" });
  const history = reviewedPeriodRows("equity_history", { period: "history" });
  const months = reviewedPeriodRows("monthly_returns", { period: "history" });
  const frameworks = reviewedPeriodRows("framework_comparison", { period: "history" });
  const statistics = reviewedPeriodRows("seed_statistics", { period: "history" });
  const training = reviewedPeriodRows("training_metrics", { period: "history" });
  const splitChoices = [...new Set(summary.map((r) => r.split))].filter((v) => ["test", "validation", "val"].includes(v));
  const [split, setSplit] = useState(splitChoices.includes("test") ? "test" : splitChoices[0]);
  const [cost, setCost] = useState("base");
  const [curveMode, setCurveMode] = useState("ensemble");
  const [frameworkName, setFrameworkName] = useState(frameworks[0]?.name ?? "");
  const selected = summary.filter((r) => r.split === split && r.scenario === cost);
  const ppo = selected.find((r) => r.strategy === "ppo_ensemble");
  const a2c = selected.find((r) => r.strategy === "a2c_ensemble");
  const cap = selected.find((r) => r.strategy === "top10_cap");
  const main = selected.filter((r) => !isSeed(r));
  const seeds = selected.filter(isSeed);
  const selectedStats = statistics.filter((r) => r.split === split && r.scenario === cost);
  const seedsFor = (name) => seeds.filter((r) => algorithm(r) === name);
  const dispersion = (name) => { const rows = seedsFor(name); const s = selectedStats.find((r) => String(r.algorithm).toUpperCase() === name); return { count: s?.seeds ?? rows.length, median: s?.median_net_return ?? median(rows.map((r) => r.net_return)), worst: s?.worst_net_return ?? (rows.length ? Math.min(...rows.map((r) => r.net_return).filter(valid)) : null) }; };
  const pd = dispersion("PPO"), ad = dispersion("A2C");
  const testModel = (name, scenario) => summary.find((r) => r.strategy === `${name}_ensemble` && r.split === "test" && r.scenario === scenario);
  const bothLoss = valid(ppo?.net_return) && valid(a2c?.net_return) && ppo.net_return < 0 && a2c.net_return < 0;
  const opening = `## 结果摘要\n\n- **${SPLIT[split] ?? split} · ${COST[cost]}**：PPO主组合扣费净收益 **${pct(ppo?.net_return)}**，小时最大回撤 **${pct(ppo?.max_drawdown)}**；A2C主组合为 **${pct(a2c?.net_return)}**、**${pct(a2c?.max_drawdown)}**。${bothLoss ? "两套主组合在这一窗口均亏损。" : "结果仅代表所选历史窗口与冻结训练设置。"}每套主组合均将事前固定的五个模型目标权重取平均，在一个现金／库存账本中实际执行。\n- 市值加权限仓基准收益 **${pct(cap?.net_return)}**，回撤 **${pct(cap?.max_drawdown)}**；其平均仓位 **${pct(cap?.average_exposure)}**，PPO为 **${pct(ppo?.average_exposure)}**，A2C为 **${pct(a2c?.average_exposure)}**。单币限仓后的余款留现金，收益和回撤差异同时伴随风险暴露差异，不能单独解释为择时或选币能力。\n- 全部五种子均保留：当前PPO单种子收益中位数 **${pct(pd.median)}**、最差 **${pct(pd.worst)}**；A2C中位数 **${pct(ad.median)}**、最差 **${pct(ad.worst)}**。固定测试期压力成本下，PPO主组合为 **${pct(testModel("ppo", "stress")?.net_return)}**，A2C为 **${pct(testModel("a2c", "stress")?.net_return)}**，两档成本复用各自同一套冻结模型。\n- 本次实测是**共用现货环境中的SB3 PPO／A2C算法对照**。A2C是TradeMaster列示算法的另一实现；五个被审查框架的原生复现状态另列。研究者已看过本年策略表现，不能把这365天称为整个研究过程全新未见的样本。`;
  const allHistory = history.filter((r) => r.split === split && r.scenario === cost);
  const curveRows = allHistory.filter((r) => curveMode === "ensemble" ? !isSeed(r) : algorithm(r) === curveMode && (isSeed(r) || isEnsemble(r)));
  const detailRows = selected;
  const detailChoices = detailRows.map((r) => r.run_id);
  const detailLabels = Object.fromEntries(detailRows.map((r) => [r.run_id, seriesName(r)]));
  const [detail, setDetail] = useState("");
  const chosen = detailRows.find((r) => r.run_id === detail) ?? ppo ?? detailRows[0];
  const selectedMonths = months.filter((r) => r.run_id === chosen?.run_id);
  const framework = frameworks.find((r) => r.name === frameworkName) ?? frameworks[0];
  const frameworkDetails = framework ? [
    ["原生任务", framework.task], ["组合适配", framework.portfolio_fit], ["数据需求", framework.data_requirement],
    ["需要修改", framework.adaptation], ["资金与费用", framework.cost_caveat], ["本地兼容", framework.local_compatibility],
    ["本次原生回测状态", framework.native_backtest_status], ["建议", framework.decision], ["固定提交", framework.commit],
  ].map(([field, value]) => ({ field, value: value ?? "未提供" })) : [];
  const frameworkLinks = (framework?.source_urls ?? []).map((entry, index) => typeof entry === "string" ? `- [官方证据${index + 1}](${entry})` : `- [${entry.label ?? entry.title ?? `官方证据${index + 1}`}](${entry.url})`).join("\n");
  const seedPairs = [...new Map(summary.filter(isSeed).map((r) => [`${algorithm(r)}:${r.seed}`, { algorithm: algorithm(r), seed: Number(r.seed) }])).values()];
  const seedComparison = seedPairs.map((pair) => {
    const row = { ...pair, _key: `${pair.algorithm}:${pair.seed}` };
    for (const subset of splitChoices) for (const scenario of ["base", "stress"]) {
      const evidence = summary.find((r) => algorithm(r) === pair.algorithm && isSeed(r) && Number(r.seed) === pair.seed && r.split === subset && r.scenario === scenario);
      row[`${subset}_${scenario}_return`] = evidence?.net_return ?? null;
      row[`${subset}_${scenario}_drawdown`] = evidence?.max_drawdown ?? null;
    }
    return row;
  });
  const trainingRows = training.map((row, index) => ({ ...row, _row: index,
    _algorithm: String(row.algorithm ?? "").toUpperCase(), _steps: first(row, ["timesteps", "total_timesteps", "trained_steps", "steps"]),
    _elapsed: first(row, ["wall_seconds", "elapsed_seconds", "training_seconds"]),
    _valReturn: first(row, ["validation_net_return", "validation_return", "val_return"]),
    _hash: first(row, ["model_sha256", "checkpoint_sha256", "sha256"]),
  }));
  const quality = training[0]?.training_window_metadata ?? {};
  const trainingText = `## 训练与样本边界\n\n- **相同数据与交互预算。** 使用独立Gymnasium现货组合账本，以Stable-Baselines3分别训练PPO和A2C；各5个种子，每种子32,768步，MLP为[32,32]，CPU运行。每步推进一日，奖励为扣费后净值变化对数乘100。优化器与更新频率各异，相同市场交互步数不等于相同算力预算。\n- **时间分离。** 名义训练2023-01-01至2025-07-25；验证90天2025-07-25至2025-10-23；测试365天2025-10-23至2026-10-23。全部UTC、右端不含。验证和测试仅做冻结模型推理；没有按测试排名挑种子。\n- **真实缺口。** 名义训练${num(quality.requested_train_days, 0)}天，按${num(quality.episode_days, 0)}日片段抽样；${num(quality.candidate_windows, 0)}个候选中可用${num(quality.eligible_windows, 0)}个，排除${num(quality.excluded_windows, 0)}个。跨越2023-03-24 13:00 UTC真实停市缺失小时的片段被排除，可用片段并集覆盖${num(quality.covered_training_days, 0)}天，这不表示每个种子遍历全部日期。验证与测试没有删日或填造缺失价格。\n- **证据范围。** 两算法的四个共用基准在每个窗口和成本情景逐字节一致，报告仅保留一份；各自独立审计重建现金、库存与小时权益。五种子共享同一市场历史，不能解释为五份独立市场样本。本次没有搜索框架或算法参数，尚不能推断它们经过其他预算配置后的表现。`;
  const methodText = `## 组合、成本与估值口径\n\n**历史资产池。** 月初及各数据切分起点选择历史市值靠前、剔除稳定币和包装币且当时Binance现货可交易的10个资产；CMC周快照使用日期加2天滞后。历史网页为回溯版本，无法证明原始发布时点。每个槽位需31个完整历史UTC日及有效价量特征才可买入。特征未就绪的成分不可买入；资金留在可投成分或现金。测试末HYPE仍未就绪。10个槽位不表示始终实际持有10币。\n\n**目标与执行。** 每次初始10,000 USDT。PPO、A2C及Top10等权／市值加权限仓基准的扣费后目标现金至少5%、单币最多25%，限仓余款留现金。BTC／ETH等权持有是被动参考，各币初始47.5%，允许超过25%；其余费用、时序、估值一致。每日00:00使用已收盘信息，在00:00小时开盘先卖出、01:00小时开盘使用净所得买入，两腿分别扣费；无借币、杠杆或BNB手续费抵扣，现金与持币数量不得为负。实际权重会随价格漂移。\n\n**成本假设。** 主场景单边手续费0.10%、不利滑点0.02%；压力场景0.20%、0.05%。同一算法的冻结模型复用两档成本，不重新训练。分币数量可为小数，未重建历史数量步长、最小名义金额和盘口深度。期末按真实剩余库存估值，另列假设全部退出再扣一次成本的净清算收益。\n\n**收益与回撤。** 完整小时收盘权益加期初本金用于最大回撤，日末权益用于曲线和日Sharpe。日末图保留小时历史峰值，图上最深点仍可能浅于完整小时最大回撤；小时内更深跌幅未被捕捉，也不能与此前分钟回撤混作同一精度。首尾月按实际区间。多次加减仓组合保留费用、换手和仓位，不把每条成交腿当作一笔完整交易计算胜率。`;
  const references = (snapshot.report?.officialReferences ?? []).map((r) => `- [${r.label}](${r.url})`).join("\n");
  const previews = {
    "https://arxiv.org/abs/2309.12891": { title: "EarnHFT官方论文", summary: "单交易对分层强化学习：秒级底层与分钟级上层，依赖订单簿与成交数据；其原生任务与本次日频多币资金组合不同。", approvedForReport: true },
    "https://arxiv.org/abs/2406.14537": { title: "MacroHFT官方论文", summary: "分钟级分层强化学习，用市场趋势和波动上下文调整子策略；原生实验使用单交易对以及订单簿和价量特征。", approvedForReport: true },
  };

  return <article className="report-content rl-framework-report" aria-label="强化学习框架与统一组合回测报告">
    <header className="report-hero"><h1 data-data-app-title contentEditable={canEdit && mode === "edit"} suppressContentEditableWarning onBlur={canEdit && mode === "edit" ? (event) => setAppTitle(event.currentTarget.textContent.trim() || appTitle) : undefined}>{appTitle}</h1><RichNarrative id="rl-framework:deck" className="report-deck" value="五个框架适配审查 · 历史动态Top10 · PPO／A2C同口径算法对照" /></header>
    <ReportSection id="rl-framework-summary" title="结果摘要" queryId="strategy_summary" sourceRows={summary} showHeading={false}><RichNarrative id="rl-framework-summary:body" value={opening} /></ReportSection>
    <ReportSection id="rl-framework-fit-heading" title="五框架适配" queryId="framework_comparison" sourceRows={frameworks} showHeading={false}><RichNarrative id="rl-framework-fit:body" value="## 五个项目分别能解决什么问题\n\n先核对原生任务、行情需求和资金口径，再判断是否能公平接入现有数据。框架能力审查与下面的算法收益来自不同证据：原生未运行的项目不填收益，不用本地A2C结果代表TradeMaster原项目。" /></ReportSection>
    <DataComponent id="rl-framework-fit-table" title="框架任务、数据及采用建议" queryId="framework_comparison" sourceRows={frameworks} displayRows={frameworks} kind="table"><DataTable rows={frameworks} rowKey="name" searchable={false} caption="五个框架的适配审查" columns={[{ field: "name", label: "框架" }, { field: "task", label: "原生任务" }, { field: "portfolio_fit", label: "多币组合适配" }, { field: "data_requirement", label: "数据需求" }, { field: "decision", label: "采用建议" }]} /></DataComponent>
    <div className="rl-framework-controls"><Dropdown label="查看固定版本和修改要求" showLabel value={framework?.name ?? ""} choices={frameworks.map((r) => r.name)} onChange={setFrameworkName} /></div>
    <DataComponent id="rl-framework-detail" title={`${framework?.name ?? "框架"}：资金、成本与本地兼容`} queryId="framework_comparison" sourceRows={framework ? [framework] : []} displayRows={frameworkDetails} kind="table"><DataTable rows={frameworkDetails} rowKey="field" searchable={false} caption="所选框架的审查细节" columns={[{ field: "field", label: "检查项" }, { field: "value", label: "审查结论" }]} /></DataComponent>
    {frameworkLinks && <ReportSection id="rl-framework-official-sources" title="所选框架的官方证据" queryId="framework_comparison" sourceRows={framework ? [framework] : []} showHeading={false}><RichNarrative id="rl-framework-official-sources:body" value={`### ${framework?.name}官方源码与论文\n\n${frameworkLinks}`} sourcePreviews={previews} /></ReportSection>}
    <div className="rl-framework-controls" aria-label="收益评估条件"><Dropdown label="评估窗口" showLabel value={split} choices={splitChoices} choiceLabels={SPLIT} onChange={setSplit} /><Dropdown label="成本情景" showLabel value={cost} choices={["base", "stress"]} choiceLabels={COST} onChange={setCost} /><Dropdown label="曲线对照" showLabel value={curveMode} choices={["ensemble", "PPO", "A2C"]} choiceLabels={CURVES} onChange={setCurveMode} /></div>
    <ReportSection id="rl-framework-numeric-heading" title="实际执行组合" queryId="equity_history" sourceRows={curveRows} showHeading={false}><RichNarrative id="rl-framework-curves:body" value={`## ${SPLIT[split] ?? split}的扣费收益与回撤\n\n当前显示**${COST[cost]} · ${CURVES[curveMode]}**。PPO与A2C均使用实际执行的五模型平均目标权重组合；曲线按日末展示，最大回撤指标读取完整小时权益。`} /></ReportSection>
    <EvidenceChart id="rl-framework-return" title="累计扣费净收益" queryId="equity_history" sourceRows={curveRows} rows={pivot(curveRows, "net_return")} spec={spec(curveRows, "net_return", curveMode)} height={410} />
    <EvidenceChart id="rl-framework-drawdown" title="日末相对小时历史峰值的回撤" queryId="equity_history" sourceRows={curveRows} rows={pivot(curveRows, "drawdown")} spec={spec(curveRows, "drawdown", curveMode)} height={320} />
    <DataComponent id="rl-framework-main-table" title="两套主组合与四个共用基准" queryId="strategy_summary" sourceRows={main} displayRows={display(main)} kind="table"><DataTable rows={display(main)} rowKey="run_id" searchable={false} compactNumbers={false} caption="主组合与共用基准比较" columns={[{ field: "_label", label: "组合" }, pc("net_return", "扣费净收益"), pc("max_drawdown", "小时最大回撤"), nc("sharpe_daily", "日Sharpe"), nc("calmar", "Calmar"), nc("fees", "费用USDT"), nc("slippage_cost", "滑点USDT"), nc("turnover", "周转倍数"), pc("average_exposure", "平均仓位"), nc("final_cash", "末端现金USDT"), pc("net_liquidation_return", "扣退出成本后")]} /></DataComponent>
    <DataComponent id="rl-framework-dispersion" title="全部种子的收益分布" queryId="seed_statistics" sourceRows={selectedStats} displayRows={selectedStats} kind="table"><DataTable rows={selectedStats} rowKey="algorithm" searchable={false} caption="所选窗口与成本的种子分布" columns={[{ field: "algorithm", label: "算法" }, nc("seeds", "种子数", 0), pc("median_net_return", "收益中位数"), pc("worst_net_return", "最差收益"), pc("best_net_return", "最高收益"), pc("median_max_drawdown", "回撤中位数"), pc("worst_max_drawdown", "最差回撤"), nc("positive_seeds", "正收益种子", 0)]} /></DataComponent>
    <DataComponent id="rl-framework-seeds" title="所选窗口的十个单种子结果" queryId="strategy_summary" sourceRows={seeds} displayRows={display(seeds)} kind="table"><DataTable rows={display(seeds)} rowKey="run_id" searchable={false} compactNumbers={false} caption="PPO与A2C全部单种子结果" columns={[{ field: "_algorithm", label: "算法" }, nc("seed", "种子", 0), pc("net_return", "扣费净收益"), pc("max_drawdown", "小时最大回撤"), nc("fees", "费用USDT"), nc("slippage_cost", "滑点USDT"), nc("turnover", "周转倍数"), pc("average_exposure", "平均仓位"), nc("fills", "成交腿数", 0), pc("net_liquidation_return", "扣退出成本后")]} /></DataComponent>
    <ReportSection id="rl-framework-splits-heading" title="验证与测试分开" queryId="strategy_summary" sourceRows={summary.filter(isSeed)} showHeading={false}><RichNarrative id="rl-framework-splits:body" value="## 十个单模型在验证与测试中表现如何\n\n每个种子完整列示两个窗口、两档成本。表中最高收益用于描述分散程度，不作为事后替换主组合的选择规则。" /></ReportSection>
    <DataComponent id="rl-framework-split-table" title="全部种子跨窗口及成本比较" queryId="strategy_summary" sourceRows={summary.filter(isSeed)} displayRows={seedComparison} kind="table"><DataTable rows={seedComparison} rowKey="_key" searchable={false} caption="所有单种子的验证与测试结果" columns={[{ field: "algorithm", label: "算法" }, nc("seed", "种子", 0), ...splitChoices.flatMap((subset) => ["base", "stress"].flatMap((scenario) => [pc(`${subset}_${scenario}_return`, `${SPLIT[subset]}·${COST[scenario]}收益`), pc(`${subset}_${scenario}_drawdown`, `${SPLIT[subset]}·${COST[scenario]}回撤`)]))]} /></DataComponent>
    <EvidenceChart id="rl-framework-months" title="所选组合的月度扣费净收益" queryId="monthly_returns" sourceRows={selectedMonths} rows={selectedMonths.map((r) => ({ ...r, "月收益(%)": valid(r.return) ? r.return * 100 : null }))} spec={{ type: "bar", x: "month", y: "月收益(%)", stackable: false, valueDecimals: 2 }} height={310} headerControls={<Dropdown label="查看组合" showLabel value={chosen?.run_id ?? ""} choices={detailChoices} choiceLabels={detailLabels} onChange={setDetail} />} />
    <DataComponent id="rl-framework-month-table" title="所选组合的月度精确结果" queryId="monthly_returns" sourceRows={selectedMonths} displayRows={selectedMonths} kind="table"><DataTable rows={selectedMonths} searchable={false} caption="月度净收益" columns={[{ field: "month", label: "UTC月份" }, pc("return", "净收益")]} /></DataComponent>
    <ReportSection id="rl-framework-training-heading" title="训练与研究边界" queryId="training_metrics" sourceRows={training} showHeading={false}><RichNarrative id="rl-framework-training:body" value={trainingText} /></ReportSection>
    <DataComponent id="rl-framework-training-table" title="十个冻结模型的训练记录" queryId="training_metrics" sourceRows={training} displayRows={trainingRows} kind="table"><DataTable rows={trainingRows} rowKey="_row" searchable={false} caption="训练预算及冻结模型" columns={[{ field: "_algorithm", label: "算法" }, nc("seed", "种子", 0), nc("_steps", "训练步数", 0), nc("_elapsed", "训练秒数", 1), pc("_valReturn", "验证净收益"), { field: "_hash", label: "模型SHA256" }]} /></DataComponent>
    <ReportSection id="rl-framework-methods" title="统一交易与指标口径" queryId="strategy_summary" sourceRows={summary} showHeading={false}><RichNarrative id="rl-framework-methods:body" value={methodText} /><RichNarrative id="rl-framework-references:body" value={`### 算法和行情官方参考\n\n${references}`} /></ReportSection>
  </article>;
}

import React, { useState } from "react";
import { DataComponent, DataTable, Dropdown, EvidenceChart, ReportSection, RichNarrative, useDataApp } from "../../data-app-public.jsx";
import "./foundation-model-report.css";

const COST = { base: "主成本", stress: "压力成本" };
const SPLIT = { test: "测试365天", validation: "验证90天" };
const VARIANT = { original: "原资产池", ex_ada: "剔除ADA后顺位补足" };
const IMPLEMENTATION = { sb3: "SB3", elegantrl: "ElegantRL", foundation: "预训练预测", shared: "共用基准" };
const MODEL = { kronos_small: "Kronos-small", kronos_base: "Kronos-base", chronos2: "Chronos-2" };
const valid = (v) => typeof v === "number" && Number.isFinite(v);
const pct = (v) => valid(v) ? `${(v * 100).toFixed(2)}%` : "—";
const pp = (v) => valid(v) ? `${v > 0 ? "+" : ""}${(v * 100).toFixed(2)}个百分点` : "—";
const num = (v, digits = 2) => valid(v) ? v.toLocaleString("zh-CN", { minimumFractionDigits: digits, maximumFractionDigits: digits }) : "—";
const pc = (field, label) => ({ field, label, renderCell: pct });
const dc = (field, label) => ({ field, label, renderCell: pp });
const nc = (field, label, digits = 2) => ({ field, label, renderCell: (v) => num(v, digits) });
const difference = (a, b) => valid(a) && valid(b) ? a - b : null;
const isPrimary = (r) => r.primary_policy === true || r.policy_role === "primary" || String(r.strategy ?? "").endsWith("_ensemble");
const isSeed = (r) => !isPrimary(r) && r.seed !== null && r.seed !== undefined && valid(Number(r.seed));
const isBaseline = (r) => r.policy_role === "baseline" || r.algorithm === "SHARED" || r.implementation === "shared";
const key = (r) => `${r.implementation}:${r.model_key ?? r.algorithm}`;
const modelName = (r) => r.implementation === "foundation" ? MODEL[r.model_key] ?? r.algorithm : `${IMPLEMENTATION[r.implementation] ?? r.implementation} ${r.algorithm}`;
const name = (r) => isPrimary(r) ? `${modelName(r)}主组合` : isSeed(r) ? `${modelName(r)}${r.implementation === "foundation" ? "采样" : "种子"}${r.seed}` : r.label;
const listText = (v) => Array.isArray(v) ? v.map((e) => typeof e === "object" ? e.symbol ?? e.asset_key ?? e.name ?? JSON.stringify(e) : String(e)).join("、") : v === null || v === undefined ? "—" : typeof v === "object" ? JSON.stringify(v) : String(v);
const first = (r, fields) => { for (const f of fields) if (r[f] !== undefined && r[f] !== null) return r[f]; return null; };
const display = (rows) => rows.map((r) => ({ ...r, _implementation: IMPLEMENTATION[r.implementation] ?? r.implementation, _variant: VARIANT[r.universe_variant] ?? r.universe_variant, _model: modelName(r), _name: name(r), _averageCash: valid(r.average_exposure) ? 1 - r.average_exposure : null, _seedType: r.implementation === "foundation" ? "固定权重的预测采样" : "独立训练种子" }));

function pivot(rows, field) {
  const data = new Map();
  rows.forEach((r) => { if (!data.has(r.timestamp)) data.set(r.timestamp, { timestamp: r.timestamp }); const point = data.get(r.timestamp); point[`${name(r)}(%)`] = valid(r[field]) ? r[field] * 100 : null; if (r.availability_event && r.model_key) point[`availability_${r.model_key}`] = r.availability_event; });
  return [...data.values()].sort((a, b) => a.timestamp.localeCompare(b.timestamp));
}
function chartSpec(rows, field) {
  const labels = [...new Set(rows.map(name))], fields = labels.map((v) => `${v}(%)`);
  const colors = ["var(--chart-1)", "var(--chart-2)", "var(--chart-3)", "var(--chart-4)", "var(--chart-5)", "var(--chart-6, #0891b2)", "#0f766e", "#a16207"];
  const events = [...new Map(rows.filter((r) => r.availability_event && r.model_key).map((r) => [`${r.model_key}:${r.timestamp}`, r])).values()];
  return { type: "line", x: "timestamp", y: fields[0] ?? "收益(%)", fields, stackable: false, valueDecimals: 2,
    annotations: events.slice(0, 8).map((r) => ({ id: `available-${r.model_key}`, kind: "event", at: r.timestamp, field: `availability_${r.model_key}`, label: r.availability_event })),
    colors: Object.fromEntries(labels.map((label, i) => [`${label}(%)`, colors[i % colors.length]])),
    legend: { labels: Object.fromEntries(fields.map((f) => [f, f.replace(/\(%\)$/u, "")])) },
    yLabel: field === "drawdown" ? "距小时历史峰值（%）" : "扣费累计净收益（%）" };
}

export function ReportContent() {
  const { snapshot, reviewedPeriodRows, appTitle, setAppTitle, canEdit, mode } = useDataApp();
  const summary = reviewedPeriodRows("strategy_summary", { period: "history" });
  const history = reviewedPeriodRows("equity_history", { period: "history" });
  const months = reviewedPeriodRows("monthly_returns", { period: "history" });
  const statistics = reviewedPeriodRows("seed_statistics", { period: "history" });
  const training = reviewedPeriodRows("training_metrics", { period: "history" });
  const changes = reviewedPeriodRows("universe_changes", { period: "history" });
  const models = reviewedPeriodRows("model_provenance", { period: "history" });
  const diagnostics = reviewedPeriodRows("forecast_diagnostics", { period: "history" });
  const hasImplementation = Boolean(snapshot.queries?.implementation_review);
  const implementation = hasImplementation ? reviewedPeriodRows("implementation_review", { period: "history" }) : [];
  const [variant, setVariant] = useState("ex_ada"), [split, setSplit] = useState("test"), [cost, setCost] = useState("base");
  const [curveMode, setCurveMode] = useState("new_and_refs"), [detail, setDetail] = useState("");
  const selected = summary.filter((r) => r.universe_variant === variant && r.split === split && r.scenario === cost);
  const primary = selected.filter(isPrimary), seeds = selected.filter(isSeed), baselines = selected.filter(isBaseline);
  const newPrimary = primary.filter((r) => r.implementation === "foundation");
  const groups = [...new Map(primary.map((r) => [key(r), r])).values()];
  const groupLabels = Object.fromEntries(groups.map((r) => [key(r), `${modelName(r)}主组合与采样明细`]));
  const a2c = primary.find((r) => r.implementation === "sb3" && r.algorithm === "A2C");
  const bestNew = [...newPrimary].filter((r) => valid(r.net_return)).sort((a, b) => b.net_return - a.net_return)[0];
  const positive = newPrimary.filter((r) => valid(r.net_return) && r.net_return > 0).length;
  const capReference = baselines.find((r) => r.strategy === "top10_cap"), cashReference = baselines.find((r) => r.strategy === "cash");
  const capDominates = capReference && newPrimary.length === 3 && newPrimary.every((r) => valid(r.net_return) && valid(r.max_drawdown) && r.net_return < capReference.net_return && r.max_drawdown < capReference.max_drawdown);
  const benchmarkNote = `${capReference ? `同场景的**动态市值加权基准**：扣费收益 **${pct(capReference.net_return)}**，小时最大回撤 **${pct(capReference.max_drawdown)}**。` : ""}${cashReference ? `现金基准为 **${pct(cashReference.net_return)}／${pct(cashReference.max_drawdown)}**（收益／回撤）。` : ""}${capDominates ? "当前三组新模型的收益均低于市值加权基准，回撤也更深；这一判断仅限本轮固定映射与当前场景。" : ""}`;
  const opening = `## 结果摘要\n\n当前为**${VARIANT[variant]} · ${SPLIT[split]} · ${COST[cost]}**。新模型均使用固定预训练权重和事前声明的预测转仓位规则，在原现货账本中逐日模拟执行。\n\n${newPrimary.map((r) => `- **${modelName(r)}**：扣费净收益 **${pct(r.net_return)}**，小时最大回撤 **${pct(r.max_drawdown)}**，平均仓位 **${pct(r.average_exposure)}**。`).join("\n")}\n\n${newPrimary.length === 3 && positive === 0 ? "三套新模型主组合在当前场景均未超过初始本金。" : `当前三套新模型中有${positive}套净收益为正。`}${bestNew && a2c ? `净收益最高的新模型${modelName(bestNew)}，相对原SB3 A2C的收益变化 **${pp(difference(bestNew.net_return, a2c.net_return))}**，回撤变化 **${pp(difference(bestNew.max_drawdown, a2c.max_drawdown))}**。` : ""}\n\n${benchmarkNote}\n\nKronos的多次采样来自同一预训练权重，不能当作五个独立训练模型；Chronos-2保留一个确定性策略。模型首次可核实可用日、此前的现金等待以及预训练数据截止证据分别列出。该年度已用于此前研究，比较不构成全新独立市场验证。`;
  const selectedHistory = history.filter((r) => r.universe_variant === variant && r.split === split && r.scenario === cost);
  const curveRows = selectedHistory.filter((r) => curveMode === "new_and_refs" ? (isPrimary(r) && (r.implementation === "foundation" || r.implementation === "sb3")) || r.strategy === "btc_eth_hold" : curveMode === "all_primary" ? isPrimary(r) : curveMode === "baselines" ? isBaseline(r) : key(r) === curveMode && (isPrimary(r) || isSeed(r)));
  const allPrimary = summary.filter((r) => isPrimary(r) && r.split === split && r.scenario === cost);
  const poolChanges = groups.map((r) => {
    const old = allPrimary.find((e) => key(e) === key(r) && e.universe_variant === "original");
    const newer = allPrimary.find((e) => key(e) === key(r) && e.universe_variant === "ex_ada");
    return { _key: key(r), model: modelName(r), original_return: old?.net_return ?? null, ex_ada_return: newer?.net_return ?? null, return_delta: difference(newer?.net_return, old?.net_return), original_drawdown: old?.max_drawdown ?? null, ex_ada_drawdown: newer?.max_drawdown ?? null, drawdown_delta: difference(newer?.max_drawdown, old?.max_drawdown), original_exposure: old?.average_exposure ?? null, ex_ada_exposure: newer?.average_exposure ?? null };
  });
  const selectedStats = statistics.filter((r) => r.universe_variant === variant && r.split === split && r.scenario === cost);
  const statsRows = display(selectedStats).map((r) => ({ ...r, _key: key(r) }));
  const detailRows = selected.filter((r) => isPrimary(r) || isSeed(r));
  const chosen = detailRows.find((r) => r.run_id === detail) ?? newPrimary[0] ?? primary[0];
  const selectedMonths = months.filter((r) => r.run_id === chosen?.run_id);
  const modelRows = models.map((r, i) => ({ ...r, _key: i, _model: MODEL[r.model_key] ?? r.label ?? r.model_key,
    _revision: first(r, ["weight_revision", "revision", "checkpoint_revision"]), _weightHash: listText(first(r, ["weight_sha256", "weights_sha256", "checkpoint_sha256"])),
    _available: first(r, ["first_verified_available_at", "published_at", "available_after"]), _decision: first(r, ["first_decision_date", "first_usable_decision"]),
    _cutoff: listText(first(r, ["training_cutoff_status", "training_data_cutoff", "cutoff_evidence"]) ?? (r.pretraining_cutoff_independently_verified === false ? "未完整独立核实" : null)),
    _context: first(r, ["context_days", "context_length"]), _horizon: first(r, ["horizon_days", "prediction_length"]),
    _sampling: listText(first(r, ["sampling_description", "inference_description"])), _fit: r.local_fit === false ? "无本地训练" : listText(r.local_fit) }));
  const periodConfig = snapshot.report?.studyConfig ?? {};
  const diagnosticStart = String(split === "test" ? periodConfig.test_start ?? "2025-10-23" : periodConfig.validation_start ?? "2025-07-25").slice(0, 10);
  const diagnosticEnd = String(split === "test" ? periodConfig.test_end ?? "2026-10-23" : periodConfig.validation_end ?? "2025-10-23").slice(0, 10);
  const selectedDiagnostics = diagnostics.filter((r) => !r.universe_variant || r.universe_variant === variant).filter((r) => r.split ? r.split === split : (() => { const day = String(first(r, ["date", "decision_date", "timestamp"]) ?? "").slice(0, 10); return day >= diagnosticStart && day < diagnosticEnd; })());
  const diagnosticRows = selectedDiagnostics.map((r, i) => ({ ...r, _key: i, _model: MODEL[r.model_key] ?? r.label ?? r.algorithm,
    _date: first(r, ["date", "decision_date", "timestamp"]), _paths: first(r, ["sampling_paths", "forecast_paths"]), _contextCount: first(r, ["context_ready_count", "eligible_context_assets"]),
    _poolCount: first(r, ["asset_count", "pool_assets", "selected_assets"]), _forecastCount: first(r, ["forecast_available_count", "forecast_assets", "predicted_assets"]),
    _positiveCount: first(r, ["score_positive_count", "positive_scores", "positive_assets"]), _reason: listText(first(r, ["exclusion_reason", "reason", "cash_reason"])),
    _available: typeof r.model_available === "boolean" ? (r.model_available ? "可用" : "等待发布") : listText(r.model_available) }));
  const availabilityText = modelRows.map((r) => { const waiting = selectedDiagnostics.filter((d) => d.model_key === r.model_key && d.model_available === false).length; return `**${r._model}**首个可用决策日 ${r._decision ? String(r._decision).slice(0, 10) + " UTC" : "未提供"}；当前窗口因尚未公开而持现金 **${waiting}日**。`; }).join("\n\n");
  const trainRows = display(training).map((r, i) => ({ ...r, _key: i, _steps: first(r, ["timesteps", "total_timesteps", "steps"]), _hash: first(r, ["model_sha256", "checkpoint_sha256"]) }));
  const changeRows = changes.map((r, i) => ({ ...r, _key: i, _old: listText(r.original_assets), _new: listText(r.ex_ada_assets), _removed: listText(r.removed_assets), _added: listText(r.added_assets), _detail: listText(r.replacement_detail) }));
  const implementationRows = implementation.map((r, i) => ({ ...r, _key: i, _implementation: IMPLEMENTATION[r.implementation] ?? r.implementation, _details: listText(r.details), _adapter: listText(r.adapter_changes), _budget: listText(r.budget_caveat) }));
  const mappingText = snapshot.report?.forecastMappingDescription ?? "预测转仓位规则以冻结协议为准。这里只展示实际执行结果，不从预测误差推导交易利润，不对缺失预测补零或回填未来行情。";
  const executionText = snapshot.report?.inferenceExecutionDescription ?? "实际推理环境和调度方式见冻结协议。";
  const methodText = `## 样本与交易口径\n\n每次独立本金10,000 USDT，无杠杆、借币和BNB抵扣；目标现金至少5%、单币目标最多25%。每日00:00使用已收盘信息先卖出，01:00小时开盘再用净所得买入，两腿分别计费。主成本单边手续费0.10%＋不利滑点0.02%；压力成本0.20%＋0.05%。新模型的两档成本使用相同预测及目标权重，各自计算真实现金、库存和费用；旧RL同模型也在各档成本账本中分别评估。\n\n验证窗口为2025-07-25至2025-10-23，测试365天为2025-10-23至2026-10-23，均UTC、右端不含。预训练预测仅使用决策日前完整历史K线，不提供未来价格或未来外生变量；权重首次可核实可用日不足以证明训练数据全部截止时间，证据缺口保留在模型表。模型尚不可用的决策日持现金，不倒填预测；完整365天窗口保留。可用日期按权重公开证据设门，本轮用固定推理代码重演历史，不表示当时已部署同一交易系统。\n\n${mappingText}\n\n旧强化学习模型的五个种子分别训练；Kronos的采样种子只改变固定权重下的生成随机性，两者不是同一种重复实验。Kronos主组合对各采样目标权重取平均后在一个独立账本成交；Chronos-2只保留确定性主策略。全部策略和采样均保留，不按本年度收益挑选。\n\n${executionText}\n\n小时收盘权益和初始本金用于最大回撤；图表只按日末采样，不能捕捉小时内更深跌幅。期末真实持仓估值与假设全部退出后的净清算收益分别列示。平均现金占比为1减同一小时估值口径的平均仓位。数量允许小数币，未重建历史步长、最小名义金额和盘口深度。BTC／ETH被动参考初始各47.5%，是25%单币上限的明确例外。\n\n历史Top10使用回溯CMC快照及可用滞后假设，并要求当时Binance现货可交易。剔除ADA后按历史排名顺位补足，来自此前亏损归因；差值同时包含补入资产和策略路径变化，不能解释为单一剔除的因果收益。同一年市场已被前序研究观察，不称为全新独立样本。`;

  return <article className="report-content foundation-model-report" aria-label="预训练预测模型与强化学习组合比较">
    <header className="report-hero"><h1 data-data-app-title contentEditable={canEdit && mode === "edit"} suppressContentEditableWarning onBlur={canEdit && mode === "edit" ? (e) => setAppTitle(e.currentTarget.textContent.trim() || appTitle) : undefined}>{appTitle}</h1><RichNarrative id="foundation:deck" className="report-deck" value="Kronos-small／Kronos-base／Chronos-2 · 同一现货账本 · 原池与剔除ADA" /></header>
    <ReportSection id="foundation-summary" title="结果摘要" queryId="strategy_summary" sourceRows={selected} showHeading={false}><RichNarrative id="foundation-summary:body" value={opening} /></ReportSection>
    <div className="foundation-controls" aria-label="比较显示条件"><Dropdown label="资产池" showLabel value={variant} choices={["original", "ex_ada"]} choiceLabels={VARIANT} onChange={setVariant} /><Dropdown label="评估窗口" showLabel value={split} choices={["test", "validation"]} choiceLabels={SPLIT} onChange={setSplit} /><Dropdown label="成本情景" showLabel value={cost} choices={["base", "stress"]} choiceLabels={COST} onChange={setCost} /><Dropdown label="曲线对照" showLabel value={curveMode} choices={["new_and_refs", "all_primary", "baselines", ...groups.map(key)]} choiceLabels={{ new_and_refs: "新模型、SB3与BTC／ETH参考", all_primary: "全部实际主组合", baselines: "共用被动基准", ...groupLabels }} onChange={setCurveMode} /></div>
    <ReportSection id="foundation-availability" title="图中模型可用时间" queryId="model_provenance" queryIds={["model_provenance", "forecast_diagnostics"]} sourceRowsByQuery={{ model_provenance: models, forecast_diagnostics: selectedDiagnostics }} showHeading={false}><RichNarrative id="foundation-availability:body" value={availabilityText} /></ReportSection>
    <EvidenceChart id="foundation-return" title="同一资产池的累计扣费净收益" queryId="equity_history" sourceRows={curveRows} rows={pivot(curveRows, "net_return")} spec={chartSpec(curveRows, "net_return")} height={420} />
    <EvidenceChart id="foundation-drawdown" title="日末相对小时历史峰值的回撤" queryId="equity_history" sourceRows={curveRows} rows={pivot(curveRows, "drawdown")} spec={chartSpec(curveRows, "drawdown")} height={320} />
    <DataComponent id="foundation-main" title="新模型与既有强化学习的实际主组合" queryId="strategy_summary" sourceRows={primary} displayRows={display(primary)} kind="table"><DataTable rows={display(primary)} rowKey="run_id" searchable={false} compactNumbers={false} caption="同池主组合扣费比较" columns={[{ field: "_model", label: "模型／实现" }, pc("net_return", "扣费净收益"), pc("max_drawdown", "小时最大回撤"), nc("sharpe_daily", "日Sharpe"), nc("fees", "费用USDT"), nc("slippage_cost", "滑点USDT"), nc("turnover", "周转倍数"), pc("average_exposure", "平均仓位"), pc("_averageCash", "平均现金"), pc("net_liquidation_return", "扣退出成本后")]} /></DataComponent>
    <DataComponent id="foundation-baselines" title="当前资产池的共用基准" queryId="strategy_summary" sourceRows={baselines} displayRows={baselines} kind="table"><DataTable rows={baselines} rowKey="run_id" searchable={false} caption="同口径被动基准" columns={[{ field: "label", label: "基准" }, pc("net_return", "扣费净收益"), pc("max_drawdown", "小时最大回撤"), nc("fees", "费用USDT"), pc("average_exposure", "平均仓位")]} /></DataComponent>
    <ReportSection id="foundation-sampling-note" title="采样与训练的分散程度" queryId="seed_statistics" sourceRows={selectedStats} showHeading={false}><RichNarrative id="foundation-sampling-note:body" value="## 采样与训练的分散程度\n\nKronos各采样使用同一预训练权重，差异表示生成随机性；旧RL种子分别训练。它们共享同一年行情，不能当作独立市场重复。Chronos-2为单一确定性策略，不制造五种子分布。" /></ReportSection>
    <DataComponent id="foundation-statistics" title="全部真实采样／训练种子的结果分布" queryId="seed_statistics" sourceRows={selectedStats} displayRows={statsRows} kind="table"><DataTable rows={statsRows} rowKey="_key" searchable={false} caption="实际重复策略的描述性分布" columns={[{ field: "_model", label: "模型／实现" }, { field: "_seedType", label: "重复来源" }, nc("seeds", "次数", 0), pc("median_net_return", "收益中位数"), pc("worst_net_return", "最差收益"), pc("best_net_return", "最高收益"), pc("worst_max_drawdown", "最差回撤")]} /></DataComponent>
    <DataComponent id="foundation-seeds" title="全部单采样与训练种子的扣费结果" queryId="strategy_summary" sourceRows={seeds} displayRows={display(seeds)} kind="table"><DataTable rows={display(seeds)} rowKey="run_id" searchable caption="单次采样／训练种子" columns={[{ field: "_model", label: "模型／实现" }, { field: "_seedType", label: "重复来源" }, nc("seed", "编号", 0), pc("net_return", "扣费净收益"), pc("max_drawdown", "小时最大回撤"), nc("fees", "费用USDT"), pc("average_exposure", "平均仓位")]} /></DataComponent>
    <DataComponent id="foundation-pool-deltas" title="同一策略在原池和剔除ADA后的变化" queryId="strategy_summary" sourceRows={allPrimary} displayRows={poolChanges} kind="table"><DataTable rows={poolChanges} rowKey="_key" searchable={false} caption="剔除ADA后减原池的主组合差值" columns={[{ field: "model", label: "模型／实现" }, pc("original_return", "原池收益"), pc("ex_ada_return", "剔ADA收益"), dc("return_delta", "收益变化"), pc("original_drawdown", "原池小时回撤"), pc("ex_ada_drawdown", "剔ADA小时回撤"), dc("drawdown_delta", "回撤变化"), pc("original_exposure", "原池仓位"), pc("ex_ada_exposure", "剔ADA仓位")]} /></DataComponent>
    <div className="foundation-detail-control"><Dropdown label="查看组合" showLabel value={chosen?.run_id ?? ""} choices={detailRows.map((r) => r.run_id)} choiceLabels={Object.fromEntries(detailRows.map((r) => [r.run_id, name(r)]))} onChange={setDetail} /></div>
    <EvidenceChart id="foundation-months" title="所选组合的月度扣费收益" queryId="monthly_returns" sourceRows={selectedMonths} rows={selectedMonths.map((r) => ({ ...r, "月收益(%)": valid(r.return) ? r.return * 100 : null }))} spec={{ type: "bar", x: "month", y: "月收益(%)", stackable: false, valueDecimals: 2 }} height={310} />
    <DataComponent id="foundation-month-table" title="所选组合的月度精确结果" queryId="monthly_returns" sourceRows={selectedMonths} displayRows={selectedMonths} kind="table"><DataTable rows={selectedMonths} searchable={false} caption="月度净收益" columns={[{ field: "month", label: "UTC月份" }, pc("return", "净收益")]} /></DataComponent>
    <DataComponent id="foundation-model-provenance" title="真正使用的预训练权重与可用日期" queryId="model_provenance" sourceRows={models} displayRows={modelRows} kind="table"><DataTable rows={modelRows} rowKey="_key" searchable={false} caption="权重版本及时间证据" columns={[{ field: "_model", label: "模型" }, { field: "_revision", label: "权重固定版本" }, { field: "_available", label: "首次可核可用时间" }, { field: "_decision", label: "首个可用决策日" }, { field: "_cutoff", label: "预训练截止证据" }, nc("_context", "历史日数", 0), nc("_horizon", "预测日数", 0), { field: "_sampling", label: "推理采样" }, { field: "_fit", label: "本地拟合" }, { field: "_weightHash", label: "权重SHA256" }, { field: "source_url", label: "官方来源", renderCell: (v) => v ? <a href={v} target="_blank" rel="noreferrer">官方来源</a> : "—" }]} /></DataComponent>
    <DataComponent id="foundation-diagnostics" title="历史预测覆盖与现金等待" queryId="forecast_diagnostics" sourceRows={selectedDiagnostics} displayRows={diagnosticRows} kind="table"><DataTable rows={diagnosticRows} rowKey="_key" searchable caption="所选资产池与窗口的预测可用性" columns={[{ field: "_date", label: "决策日UTC" }, { field: "_model", label: "模型" }, nc("_paths", "预测路径数", 0), { field: "_available", label: "模型状态" }, nc("_poolCount", "成分数", 0), nc("_contextCount", "128日上下文就绪", 0), nc("_forecastCount", "可用预测数（路径最小）", 0), nc("_positiveCount", "正分数数（路径均值）", 1), { field: "_reason", label: "未投或等待原因" }]} /></DataComponent>
    {hasImplementation && <DataComponent id="foundation-implementation" title="已保存的算法与适配来源" queryId="implementation_review" sourceRows={implementation} displayRows={implementationRows} kind="table"><DataTable rows={implementationRows} rowKey="_key" searchable={false} caption="固定实现及适配记录" columns={[{ field: "_implementation", label: "实现" }, { field: "algorithm", label: "算法／模型" }, { field: "commit", label: "固定提交" }, { field: "_details", label: "实际实现" }, { field: "_adapter", label: "适配范围" }, { field: "_budget", label: "预算与边界" }]} /></DataComponent>}
    <DataComponent id="foundation-legacy-training" title="旧RL对照的真实训练来源" queryId="training_metrics" sourceRows={training} displayRows={trainRows} kind="table"><DataTable rows={trainRows} rowKey="_key" searchable caption="只读保留的旧强化学习训练记录" columns={[{ field: "_variant", label: "资产池" }, { field: "_model", label: "模型／实现" }, nc("seed", "种子", 0), nc("_steps", "训练步数", 0), { field: "_hash", label: "原模型SHA256" }]} /></DataComponent>
    <DataComponent id="foundation-universe" title="原池与剔除ADA后的历史替补" queryId="universe_changes" sourceRows={changes} displayRows={changeRows} kind="table"><DataTable rows={changeRows} rowKey="_key" searchable caption="历史资产池选择边界" columns={[{ field: "anchor", label: "选择边界UTC" }, { field: "_removed", label: "移除" }, { field: "_added", label: "补入" }, { field: "_detail", label: "顺位选择" }, { field: "_old", label: "原池" }, { field: "_new", label: "剔ADA后池" }]} /></DataComponent>
    <ReportSection id="foundation-methods" title="样本与交易口径" queryId="strategy_summary" queryIds={["strategy_summary", "model_provenance", "training_metrics"]} sourceRowsByQuery={{ strategy_summary: summary, model_provenance: models, training_metrics: training }} showHeading={false}><RichNarrative id="foundation-methods:body" value={methodText} /></ReportSection>
  </article>;
}

import React, { useState } from "react";
import { DataComponent, DataTable, Dropdown, EvidenceChart, ReportSection, RichNarrative, useDataApp } from "../../data-app-public.jsx";
import "./rl-elegantrl-report.css";

const COST = { base: "主成本", stress: "压力成本" };
const SPLIT = { test: "测试365天", validation: "验证90天" };
const VARIANT = { original: "原资产池", ex_ada: "剔除ADA后顺位补足" };
const IMPLEMENTATION = { sb3: "SB3", elegantrl: "ElegantRL", shared: "共用基准" };
const valid = (v) => typeof v === "number" && Number.isFinite(v);
const pct = (v) => valid(v) ? `${(v * 100).toFixed(2)}%` : "—";
const pp = (v) => valid(v) ? `${v > 0 ? "+" : ""}${(v * 100).toFixed(2)}个百分点` : "—";
const num = (v, digits = 2) => valid(v) ? v.toLocaleString("zh-CN", { minimumFractionDigits: digits, maximumFractionDigits: digits }) : "—";
const pc = (field, label) => ({ field, label, renderCell: pct });
const dc = (field, label) => ({ field, label, renderCell: pp });
const nc = (field, label, digits = 2) => ({ field, label, renderCell: (v) => num(v, digits) });
const difference = (newValue, oldValue) => valid(newValue) && valid(oldValue) ? newValue - oldValue : null;
const isSeed = (r) => r.seed !== null && r.seed !== undefined && valid(Number(r.seed));
const isEnsemble = (r) => String(r.strategy ?? "").endsWith("_ensemble");
const modelKey = (r) => `${r.implementation}:${r.algorithm}`;
const modelName = (r) => `${IMPLEMENTATION[r.implementation] ?? r.implementation} ${r.algorithm}`;
const name = (r) => isEnsemble(r) ? `${modelName(r)}主组合` : isSeed(r) ? `${modelName(r)}种子${r.seed}` : r.label;
const fullName = (r) => `${VARIANT[r.universe_variant] ?? r.universe_variant} · ${name(r)}`;
const display = (rows) => rows.map((r) => ({ ...r, _implementation: IMPLEMENTATION[r.implementation] ?? r.implementation, _variant: VARIANT[r.universe_variant] ?? r.universe_variant, _name: name(r), _fullName: fullName(r) }));
const listText = (v) => Array.isArray(v) ? v.map((e) => typeof e === "object" ? e.symbol ?? e.asset_key ?? e.name ?? JSON.stringify(e) : String(e)).join("、") : v === null || v === undefined ? "—" : typeof v === "object" ? JSON.stringify(v) : String(v);
const first = (r, keys) => { for (const k of keys) if (r[k] !== undefined && r[k] !== null) return r[k]; return null; };

function pivot(rows, measure) {
  const data = new Map();
  rows.forEach((r) => { if (!data.has(r.timestamp)) data.set(r.timestamp, { timestamp: r.timestamp }); data.get(r.timestamp)[`${name(r)}(%)`] = valid(r[measure]) ? r[measure] * 100 : null; });
  return [...data.values()].sort((a, b) => a.timestamp.localeCompare(b.timestamp));
}
function spec(rows, measure) {
  const labels = [...new Set(rows.map(name))], fields = labels.map((v) => `${v}(%)`);
  return { type: "line", x: "timestamp", y: fields[0] ?? "收益(%)", fields, stackable: false, valueDecimals: 2,
    colors: Object.fromEntries(labels.map((label, i) => [`${label}(%)`, `var(--chart-${i % 6 + 1}, #0891b2)`])),
    legend: { labels: Object.fromEntries(fields.map((f) => [f, f.replace(/\(%\)$/u, "")])) },
    yLabel: measure === "drawdown" ? "距小时历史峰值（%）" : "扣费累计净收益（%）" };
}

export function ReportContent() {
  const { snapshot, reviewedPeriodRows, appTitle, setAppTitle, canEdit, mode } = useDataApp();
  const summary = reviewedPeriodRows("strategy_summary", { period: "history" });
  const history = reviewedPeriodRows("equity_history", { period: "history" });
  const months = reviewedPeriodRows("monthly_returns", { period: "history" });
  const statistics = reviewedPeriodRows("seed_statistics", { period: "history" });
  const changes = reviewedPeriodRows("universe_changes", { period: "history" });
  const training = reviewedPeriodRows("training_metrics", { period: "history" });
  const hasReview = Boolean(snapshot.queries?.implementation_review);
  const implementationReview = hasReview ? reviewedPeriodRows("implementation_review", { period: "history" }) : snapshot.report?.implementationNotes ?? [];
  const [variant, setVariant] = useState("original"), [split, setSplit] = useState("test"), [cost, setCost] = useState("base");
  const [curveMode, setCurveMode] = useState("ensembles"), [detail, setDetail] = useState("");
  const selected = summary.filter((r) => r.universe_variant === variant && r.split === split && r.scenario === cost);
  const ensembles = selected.filter(isEnsemble), seeds = selected.filter(isSeed), baselines = selected.filter((r) => !isSeed(r) && !isEnsemble(r));
  const allEnsembles = summary.filter((r) => isEnsemble(r) && r.split === split && r.scenario === cost);
  const allSeeds = summary.filter((r) => isSeed(r) && r.split === split && r.scenario === cost);
  const groups = [...new Map(ensembles.map((r) => [modelKey(r), r])).values()];
  const groupLabels = Object.fromEntries(groups.map((r) => [modelKey(r), `${modelName(r)}全部种子与主组合`]));
  const sbPpo = ensembles.find((r) => r.implementation === "sb3" && r.algorithm === "PPO");
  const elPpo = ensembles.find((r) => r.implementation === "elegantrl" && r.algorithm === "PPO");
  const opening = `## 结果摘要\n\n当前为**${VARIANT[variant]} · ${SPLIT[split]} · ${COST[cost]}**。各主组合均事前固定为五个种子模型的平均目标权重，在独立现金／库存账本中执行。\n\n${ensembles.map((r) => `- **${modelName(r)}**：扣费净收益 **${pct(r.net_return)}**，小时最大回撤 **${pct(r.max_drawdown)}**，平均仓位 **${pct(r.average_exposure)}**。`).join("\n")}\n\n${sbPpo && elPpo ? `同池PPO对照中，ElegantRL相对SB3的收益变化为 **${pp(difference(elPpo.net_return, sbPpo.net_return))}**，回撤变化 **${pp(difference(elPpo.max_drawdown, sbPpo.max_drawdown))}**。` : "所选切分缺少完整的两套PPO实现对照，不生成差值。"}PPO的原生更新细节和训练适配也有差异，这不是严格只更换库的实验。\n\n这些365天已用于前序策略选择；剔除ADA也源自此前亏损归因。新增框架与资产池比较均属于已见年度上的研究延伸，不能称为全新独立样本验证。`;
  const selectedHistory = history.filter((r) => r.universe_variant === variant && r.split === split && r.scenario === cost);
  const curveRows = selectedHistory.filter((r) => curveMode === "ensembles" ? isEnsemble(r) : modelKey(r) === curveMode && (isEnsemble(r) || isSeed(r)));
  const poolChanges = groups.map((r) => {
    const old = allEnsembles.find((e) => modelKey(e) === modelKey(r) && e.universe_variant === "original");
    const newer = allEnsembles.find((e) => modelKey(e) === modelKey(r) && e.universe_variant === "ex_ada");
    return { _key: modelKey(r), model: modelName(r), original_return: old?.net_return ?? null, ex_ada_return: newer?.net_return ?? null,
      return_delta: difference(newer?.net_return, old?.net_return), original_drawdown: old?.max_drawdown ?? null, ex_ada_drawdown: newer?.max_drawdown ?? null,
      drawdown_delta: difference(newer?.max_drawdown, old?.max_drawdown), original_exposure: old?.average_exposure ?? null, ex_ada_exposure: newer?.average_exposure ?? null };
  });
  const seedKeys = [...new Map(allSeeds.map((r) => [`${modelKey(r)}:${r.seed}`, r])).values()];
  const pairs = seedKeys.map((r) => {
    const find = (v) => allSeeds.find((e) => modelKey(e) === modelKey(r) && Number(e.seed) === Number(r.seed) && e.universe_variant === v);
    const old = find("original"), newer = find("ex_ada");
    return { _key: `${modelKey(r)}:${r.seed}`, model: modelName(r), seed: Number(r.seed), original_return: old?.net_return ?? null, ex_ada_return: newer?.net_return ?? null,
      return_delta: difference(newer?.net_return, old?.net_return), original_drawdown: old?.max_drawdown ?? null, ex_ada_drawdown: newer?.max_drawdown ?? null,
      drawdown_delta: difference(newer?.max_drawdown, old?.max_drawdown), fee_delta: difference(newer?.fees, old?.fees) };
  });
  const selectedStatsRaw = statistics.filter((r) => r.universe_variant === variant && r.split === split && r.scenario === cost);
  const selectedStats = selectedStatsRaw.map((r) => ({ ...r, _key: modelKey(r), _implementation: IMPLEMENTATION[r.implementation] ?? r.implementation }));
  const detailRows = selected.filter((r) => isEnsemble(r) || isSeed(r));
  const chosen = detailRows.find((r) => r.run_id === detail) ?? ensembles.find((r) => r.implementation === "elegantrl") ?? detailRows[0];
  const selectedMonths = months.filter((r) => r.run_id === chosen?.run_id);
  const changeRows = changes.map((r, i) => ({ ...r, _key: i, _old: listText(r.original_assets), _new: listText(r.ex_ada_assets), _removed: listText(r.removed_assets), _added: listText(r.added_assets), _detail: listText(r.replacement_detail) }));
  const trainRows = training.map((r, i) => ({ ...r, _key: i, _implementation: IMPLEMENTATION[r.implementation] ?? r.implementation, _variant: VARIANT[r.universe_variant] ?? r.universe_variant,
    _steps: first(r, ["timesteps", "total_timesteps", "steps"]), _elapsed: first(r, ["elapsed_seconds", "wall_seconds", "training_seconds"]), _hash: first(r, ["model_sha256", "checkpoint_sha256", "sha256"]) }));
  const reviewRows = implementationReview.map((r, i) => ({ ...r, _key: i, _implementation: IMPLEMENTATION[r.implementation] ?? r.implementation, _details: listText(r.details), _adapter: listText(r.adapter_changes), _budget: listText(r.budget_caveat) }));
  const implementationText = `## 原生算法与共用账本怎样衔接\n\nElegantRL结果只对应本次固定源码、固定训练参数与适配版本。原生PPO、SAC、TD3使用各自的策略更新和数据使用方式，PPO与SB3在初始化、激活函数、critic损失、采样和动作输出上不同，具体来源与适配见下表。相同训练交互步数不等于相同优化器更新次数或算力预算，不能把结果差异全部归为库本身。\n\n共用环境负责历史资产池、因果特征、目标权重投影、现金与库存限制、两段成交时序和费用；训练适配负责把各算法接口与同一状态／动作／奖励合同连接。SAC、TD3的探索、回放和确定性评估处理应以固定来源及适配记录为准。未提供的细节显示缺失，不从算法名称推断配置。`;
  const methodText = `## 样本与交易口径\n\n**研究边界。** 原池与排除ADA（CMC ID 2010）后按历史排名顺位补足10个槽位的池分别训练。剔除建议是在此前同年测试归因之后提出，且替补成分、状态和重训路径均改变；差值是事后敏感性结果，不能解释为单独移除ADA的因果收益。五个种子共享同一段行情，匹配编号只用于描述；完整保留全部种子，不挑测试最优者。每个主组合平均五个模型目标权重后独立执行，收益不由单种子收益简单相加。\n\n**时间与信息。** 名义训练2023-01-01至2025-07-25，验证2025-07-25至2025-10-23，测试2025-10-23至2026-10-23，全部UTC、右端不含。历史CMC周快照加2日可用滞后属于回溯口径，无法证明原始发布时间；资产还必须当时Binance现货可交易。31完整历史UTC日特征未就绪的成分不可买入，资金可分到可投成分或现金；10槽位不代表始终持有10币。\n\n**资金与成交。** 每次10,000 USDT，无杠杆、借币或BNB抵扣；扣费后目标现金至少5%、单币最多25%。每日00:00用已收盘信息，在00:00小时开盘先卖出、01:00小时开盘用净所得买入，双边分别计费。BTC／ETH持有参考各初始47.5%，属于25%上限的被动基准例外。市值限仓余款留现金，比较回撤时同时看平均仓位。\n\n**成本与估值。** 主成本为单边手续费0.10%、不利滑点0.02%，压力成本0.20%、0.05%；同一冻结模型复用两档评估成本。完整小时收盘权益及初始本金用于最大回撤，日末用于绘图，小时内更深跌幅不能捕捉，也不能与先前分钟回撤直接比较。期末按实际持仓估值，另列假设全部退出后净清算收益；小数币数量未重建历史数量步长、最小名义金额和订单簿深度。`;

  return <article className="report-content rl-elegantrl-report" aria-label="ElegantRL与SB3统一组合比较">
    <header className="report-hero"><h1 data-data-app-title contentEditable={canEdit && mode === "edit"} suppressContentEditableWarning onBlur={canEdit && mode === "edit" ? (e) => setAppTitle(e.currentTarget.textContent.trim() || appTitle) : undefined}>{appTitle}</h1><RichNarrative id="rl-elegantrl:deck" className="report-deck" value="同一现货账本与历史动态Top10 · 原池／剔除ADA · 框架原生算法与适配版本的实际对照" /></header>
    <ReportSection id="rl-elegantrl-summary" title="结果摘要" queryId="strategy_summary" sourceRows={selected} showHeading={false}><RichNarrative id="rl-elegantrl-summary:body" value={opening} /></ReportSection>
    <div className="rl-elegantrl-controls" aria-label="比较显示条件"><Dropdown label="资产池" showLabel value={variant} choices={["original", "ex_ada"]} choiceLabels={VARIANT} onChange={setVariant} /><Dropdown label="评估窗口" showLabel value={split} choices={["test", "validation"]} choiceLabels={SPLIT} onChange={setSplit} /><Dropdown label="成本情景" showLabel value={cost} choices={["base", "stress"]} choiceLabels={COST} onChange={setCost} /><Dropdown label="曲线对照" showLabel value={curveMode} choices={["ensembles", ...groups.map(modelKey)]} choiceLabels={{ ensembles: "同池全部实际主组合", ...groupLabels }} onChange={setCurveMode} /></div>
    <EvidenceChart id="rl-elegantrl-return" title="同一资产池的累计扣费净收益" queryId="equity_history" sourceRows={curveRows} rows={pivot(curveRows, "net_return")} spec={spec(curveRows, "net_return")} height={420} />
    <EvidenceChart id="rl-elegantrl-drawdown" title="日末相对小时历史峰值的回撤" queryId="equity_history" sourceRows={curveRows} rows={pivot(curveRows, "drawdown")} spec={spec(curveRows, "drawdown")} height={320} />
    <DataComponent id="rl-elegantrl-main" title="同池主组合的收益、费用与仓位" queryId="strategy_summary" sourceRows={ensembles} displayRows={display(ensembles)} kind="table"><DataTable rows={display(ensembles)} rowKey="run_id" searchable={false} compactNumbers={false} caption="实际执行的模型主组合" columns={[{ field: "_implementation", label: "实现" }, { field: "algorithm", label: "算法" }, pc("net_return", "扣费净收益"), pc("max_drawdown", "小时最大回撤"), nc("sharpe_daily", "日Sharpe"), nc("fees", "费用USDT"), nc("slippage_cost", "滑点USDT"), nc("turnover", "周转倍数"), pc("average_exposure", "平均仓位"), pc("net_liquidation_return", "扣退出成本后")]} /></DataComponent>
    <DataComponent id="rl-elegantrl-baselines" title="当前资产池的共用基准" queryId="strategy_summary" sourceRows={baselines} displayRows={baselines} kind="table"><DataTable rows={baselines} rowKey="run_id" searchable={false} caption="同口径被动基准" columns={[{ field: "label", label: "基准" }, pc("net_return", "扣费净收益"), pc("max_drawdown", "小时最大回撤"), nc("fees", "费用USDT"), pc("average_exposure", "平均仓位"), pc("net_liquidation_return", "扣退出成本后")]} /></DataComponent>
    <DataComponent id="rl-elegantrl-statistics" title="全部种子的分散程度" queryId="seed_statistics" sourceRows={selectedStatsRaw} displayRows={selectedStats} kind="table"><DataTable rows={selectedStats} rowKey="_key" searchable={false} caption="所选窗口和资产池的种子分布" columns={[{ field: "_implementation", label: "实现" }, { field: "algorithm", label: "算法" }, nc("seeds", "种子数", 0), pc("median_net_return", "收益中位数"), pc("worst_net_return", "最差收益"), pc("best_net_return", "最高收益"), pc("median_max_drawdown", "回撤中位数"), pc("worst_max_drawdown", "最差回撤")]} /></DataComponent>
    <DataComponent id="rl-elegantrl-seeds" title="当前资产池的每个单种子" queryId="strategy_summary" sourceRows={seeds} displayRows={display(seeds)} kind="table"><DataTable rows={display(seeds)} rowKey="run_id" searchable caption="全部单种子扣费结果" columns={[{ field: "_implementation", label: "实现" }, { field: "algorithm", label: "算法" }, nc("seed", "种子", 0), pc("net_return", "扣费净收益"), pc("max_drawdown", "小时最大回撤"), nc("fees", "费用USDT"), nc("turnover", "周转倍数"), pc("average_exposure", "平均仓位")]} /></DataComponent>
    <ReportSection id="rl-elegantrl-pool-heading" title="剔除ADA的变化" queryId="strategy_summary" sourceRows={allEnsembles} showHeading={false}><RichNarrative id="rl-elegantrl-pool:body" value="## 同一实现与算法换资产池后改变多少\n\n以下为剔除ADA后减原池，收益与回撤差值使用百分点。回撤为负数，变化为正表示较浅；新池替补和重新训练同时发生，变化不具有单币移除的独立因果解释。" /></ReportSection>
    <DataComponent id="rl-elegantrl-pool-deltas" title="每套主组合的两池变化" queryId="strategy_summary" sourceRows={allEnsembles} displayRows={poolChanges} kind="table"><DataTable rows={poolChanges} rowKey="_key" searchable={false} caption="两资产池的主组合差值" columns={[{ field: "model", label: "实现与算法" }, pc("original_return", "原池收益"), pc("ex_ada_return", "剔ADA收益"), dc("return_delta", "收益变化"), pc("original_drawdown", "原池小时回撤"), pc("ex_ada_drawdown", "剔ADA小时回撤"), dc("drawdown_delta", "回撤变化"), pc("original_exposure", "原池平均仓位"), pc("ex_ada_exposure", "剔ADA平均仓位")]} /></DataComponent>
    <DataComponent id="rl-elegantrl-paired-seeds" title="同编号种子的两池变化" queryId="strategy_summary" sourceRows={allSeeds} displayRows={pairs} kind="table"><DataTable rows={pairs} rowKey="_key" searchable caption="匹配实现、算法与种子编号的变化" columns={[{ field: "model", label: "实现与算法" }, nc("seed", "种子", 0), pc("original_return", "原池收益"), pc("ex_ada_return", "剔ADA收益"), dc("return_delta", "收益变化"), pc("original_drawdown", "原池小时回撤"), pc("ex_ada_drawdown", "剔ADA小时回撤"), dc("drawdown_delta", "回撤变化"), nc("fee_delta", "费用变化USDT")]} /></DataComponent>
    <div className="rl-elegantrl-detail-control"><Dropdown label="查看组合" showLabel value={chosen?.run_id ?? ""} choices={detailRows.map((r) => r.run_id)} choiceLabels={Object.fromEntries(detailRows.map((r) => [r.run_id, name(r)]))} onChange={setDetail} /></div>
    <EvidenceChart id="rl-elegantrl-months" title="所选组合的月度扣费收益" queryId="monthly_returns" sourceRows={selectedMonths} rows={selectedMonths.map((r) => ({ ...r, "月收益(%)": valid(r.return) ? r.return * 100 : null }))} spec={{ type: "bar", x: "month", y: "月收益(%)", stackable: false, valueDecimals: 2 }} height={310} />
    <DataComponent id="rl-elegantrl-month-table" title="所选组合的月度精确结果" queryId="monthly_returns" sourceRows={selectedMonths} displayRows={selectedMonths} kind="table"><DataTable rows={selectedMonths} searchable={false} caption="月度净收益" columns={[{ field: "month", label: "UTC月份" }, pc("return", "净收益")]} /></DataComponent>
    <ReportSection id="rl-elegantrl-implementation" title="原生算法与适配边界" queryId={hasReview ? "implementation_review" : "training_metrics"} sourceRows={hasReview ? implementationReview : training} showHeading={false}><RichNarrative id="rl-elegantrl-implementation:body" value={implementationText} /></ReportSection>
    {reviewRows.length > 0 && <DataComponent id="rl-elegantrl-adapters" title="固定源码及适配记录" queryId={hasReview ? "implementation_review" : "training_metrics"} sourceRows={hasReview ? implementationReview : training} displayRows={reviewRows} kind="table"><DataTable rows={reviewRows} rowKey="_key" searchable={false} caption="各原生算法及本地适配" columns={[{ field: "_implementation", label: "实现" }, { field: "algorithm", label: "算法" }, { field: "commit", label: "固定提交" }, { field: "_details", label: "原生算法说明" }, { field: "_adapter", label: "适配改版" }, { field: "_budget", label: "训练预算边界" }, { field: "source_url", label: "官方源码", renderCell: (value) => value ? <a href={value} target="_blank" rel="noreferrer">官方源码</a> : "—" }]} /></DataComponent>}
    <DataComponent id="rl-elegantrl-training" title="每个种子的训练与冻结来源" queryId="training_metrics" sourceRows={training} displayRows={trainRows} kind="table"><DataTable rows={trainRows} rowKey="_key" searchable caption="所有实际训练的冻结模型" columns={[{ field: "_variant", label: "资产池" }, { field: "_implementation", label: "实现" }, { field: "algorithm", label: "算法" }, nc("seed", "种子", 0), nc("_steps", "训练步数", 0), nc("_elapsed", "训练秒数", 1), { field: "_hash", label: "模型SHA256" }]} /></DataComponent>
    <DataComponent id="rl-elegantrl-universe" title="原池与剔除ADA后的历史替补" queryId="universe_changes" sourceRows={changes} displayRows={changeRows} kind="table"><DataTable rows={changeRows} rowKey="_key" searchable caption="历史资产池选择边界" columns={[{ field: "anchor", label: "选择边界UTC" }, { field: "_removed", label: "移除" }, { field: "_added", label: "补入" }, { field: "_detail", label: "顺位选择说明" }, { field: "_old", label: "原池" }, { field: "_new", label: "剔ADA后池" }]} /></DataComponent>
    <ReportSection id="rl-elegantrl-methods" title="样本和交易口径" queryId="strategy_summary" queryIds={["strategy_summary", "training_metrics"]} sourceRowsByQuery={{ strategy_summary: summary, training_metrics: training }} showHeading={false}><RichNarrative id="rl-elegantrl-methods:body" value={methodText + "\n\n训练只抽取行情完整的128日窗口，实际覆盖日期和被排除缺口见模型训练记录；验证和测试保持连续。"} /></ReportSection>
  </article>;
}

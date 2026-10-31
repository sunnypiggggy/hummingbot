import React, { useState } from "react";
import { DataComponent, DataTable, Dropdown, EvidenceChart, ReportSection, RichNarrative, useDataApp } from "../../data-app-public.jsx";
import "./rl-ex-ada-report.css";

const COST = { base: "主成本", stress: "压力成本" };
const SPLIT = { test: "测试365天", validation: "验证90天" };
const VARIANT = { original: "原资产池", ex_ada: "剔除ADA后顺位补足" };
const CURVES = { ensembles: "四套主组合", PPO: "PPO：原池与剔除ADA全部种子", A2C: "A2C：原池与剔除ADA全部种子" };
const valid = (v) => typeof v === "number" && Number.isFinite(v);
const pct = (v, digits = 2) => valid(v) ? `${(v * 100).toFixed(digits)}%` : "—";
const pp = (v) => valid(v) ? `${v > 0 ? "+" : ""}${(v * 100).toFixed(2)}个百分点` : "—";
const num = (v, digits = 2) => valid(v) ? v.toLocaleString("zh-CN", { minimumFractionDigits: digits, maximumFractionDigits: digits }) : "—";
const pc = (field, label) => ({ field, label, renderCell: (v) => pct(v) });
const dc = (field, label) => ({ field, label, renderCell: (v) => pp(v) });
const nc = (field, label, digits = 2) => ({ field, label, renderCell: (v) => num(v, digits) });
const difference = (newValue, oldValue) => valid(newValue) && valid(oldValue) ? newValue - oldValue : null;
const isSeed = (r) => r.seed !== null && r.seed !== undefined && valid(Number(r.seed));
const isEnsemble = (r) => ["ppo_ensemble", "a2c_ensemble"].includes(r.strategy);
const algorithm = (r) => r.strategy?.startsWith("a2c") ? "A2C" : r.strategy?.startsWith("ppo") ? "PPO" : "基准";
const name = (r) => `${VARIANT[r.universe_variant] ?? r.universe_variant} · ${isEnsemble(r) ? `${algorithm(r)}主组合` : isSeed(r) ? `${algorithm(r)}种子${r.seed}` : r.label}`;
const display = (rows) => rows.map((r) => ({ ...r, _variant: VARIANT[r.universe_variant] ?? r.universe_variant, _algorithm: algorithm(r), _name: name(r) }));
const listText = (v) => Array.isArray(v) ? v.map((entry) => typeof entry === "object" ? entry.symbol ?? entry.asset_key ?? entry.name ?? JSON.stringify(entry) : String(entry)).join("、") : v === null || v === undefined ? "—" : typeof v === "object" ? JSON.stringify(v) : String(v);
const first = (row, keys) => { for (const key of keys) if (row[key] !== undefined && row[key] !== null) return row[key]; return null; };

function pivot(rows, measure) {
  const data = new Map();
  rows.forEach((r) => { if (!data.has(r.timestamp)) data.set(r.timestamp, { timestamp: r.timestamp }); data.get(r.timestamp)[`${name(r)}(%)`] = valid(r[measure]) ? r[measure] * 100 : null; });
  return [...data.values()].sort((a, b) => a.timestamp.localeCompare(b.timestamp));
}
function spec(rows, measure) {
  const labels = [...new Set(rows.map(name))];
  const fields = labels.map((label) => `${label}(%)`);
  const colors = Object.fromEntries(labels.map((label) => {
    const r = rows.find((row) => name(row) === label);
    const color = isSeed(r) ? `var(--chart-${Number(r.seed) % 5 + 1})` : algorithm(r) === "PPO" ? r.universe_variant === "original" ? "#9a9a9a" : "var(--chart-1)" : r.universe_variant === "original" ? "#bbbbbb" : "var(--chart-2)";
    return [`${label}(%)`, color];
  }));
  return { type: "line", x: "timestamp", y: fields[0] ?? "收益(%)", fields, colors, stackable: false, valueDecimals: 2,
    legend: { labels: Object.fromEntries(fields.map((field) => [field, field.replace(/\(%\)$/u, "")])) },
    yLabel: measure === "drawdown" ? "距小时历史峰值（%）" : "扣费累计净收益（%）" };
}

export function ReportContent() {
  const { reviewedPeriodRows, appTitle, setAppTitle, canEdit, mode } = useDataApp();
  const summary = reviewedPeriodRows("strategy_summary", { period: "history" });
  const history = reviewedPeriodRows("equity_history", { period: "history" });
  const months = reviewedPeriodRows("monthly_returns", { period: "history" });
  const statistics = reviewedPeriodRows("seed_statistics", { period: "history" });
  const changes = reviewedPeriodRows("universe_changes", { period: "history" });
  const training = reviewedPeriodRows("training_metrics", { period: "history" });
  const [split, setSplit] = useState("test");
  const [cost, setCost] = useState("base");
  const [curveMode, setCurveMode] = useState("ensembles");
  const [detail, setDetail] = useState("");
  const selected = summary.filter((r) => r.split === split && r.scenario === cost);
  const ensembles = selected.filter(isEnsemble);
  const models = selected.filter(isSeed);
  const baselines = selected.filter((r) => !isSeed(r) && !isEnsemble(r));
  const ensemble = (variant, algo) => ensembles.find((r) => r.universe_variant === variant && algorithm(r) === algo);
  const oldP = ensemble("original", "PPO"), newP = ensemble("ex_ada", "PPO");
  const oldA = ensemble("original", "A2C"), newA = ensemble("ex_ada", "A2C");
  const deltaRows = ["PPO", "A2C"].map((algo) => {
    const old = ensemble("original", algo), newer = ensemble("ex_ada", algo);
    return { algorithm: algo, original_return: old?.net_return ?? null, ex_ada_return: newer?.net_return ?? null, return_delta: difference(newer?.net_return, old?.net_return),
      original_drawdown: old?.max_drawdown ?? null, ex_ada_drawdown: newer?.max_drawdown ?? null, drawdown_delta: difference(newer?.max_drawdown, old?.max_drawdown),
      original_fees: old?.fees ?? null, ex_ada_fees: newer?.fees ?? null, original_exposure: old?.average_exposure ?? null, ex_ada_exposure: newer?.average_exposure ?? null };
  });
  const pairedKeys = [...new Map(models.map((r) => [`${algorithm(r)}:${r.seed}`, { algorithm: algorithm(r), seed: Number(r.seed) }])).values()];
  const pairs = pairedKeys.map((pair) => {
    const old = models.find((r) => r.universe_variant === "original" && algorithm(r) === pair.algorithm && Number(r.seed) === pair.seed);
    const newer = models.find((r) => r.universe_variant === "ex_ada" && algorithm(r) === pair.algorithm && Number(r.seed) === pair.seed);
    return { ...pair, _key: `${pair.algorithm}:${pair.seed}`, original_return: old?.net_return ?? null, ex_ada_return: newer?.net_return ?? null,
      return_delta: difference(newer?.net_return, old?.net_return), original_drawdown: old?.max_drawdown ?? null, ex_ada_drawdown: newer?.max_drawdown ?? null,
      drawdown_delta: difference(newer?.max_drawdown, old?.max_drawdown), fee_delta: difference(newer?.fees, old?.fees), exposure_delta: difference(newer?.average_exposure, old?.average_exposure) };
  });
  const opening = `## 结果摘要\n\n- 当前显示 **${SPLIT[split]} · ${COST[cost]}**。PPO原池主组合收益 **${pct(oldP?.net_return)}**、小时最大回撤 **${pct(oldP?.max_drawdown)}**；剔除ADA并按历史排名补足后的主组合为 **${pct(newP?.net_return)}**、**${pct(newP?.max_drawdown)}**，收益变化 **${pp(difference(newP?.net_return, oldP?.net_return))}**。\n- A2C原池主组合收益 **${pct(oldA?.net_return)}**、小时最大回撤 **${pct(oldA?.max_drawdown)}**；新主组合为 **${pct(newA?.net_return)}**、**${pct(newA?.max_drawdown)}**，收益变化 **${pp(difference(newA?.net_return, oldA?.net_return))}**。四个主组合均平均五个模型的目标权重，再在各自账本执行，未挑选最高测试收益种子。\n- 剔除ADA的建议来自此前同一年测试亏损归因，属于**看过结果后的敏感性实验**。排名补位会改变其他成分，模型也重新训练，因此变化同时包含资产池替换、权重分配与训练路径变化，不能解释为单独移除ADA的因果收益，也不是全新样本上的验证。`;
  const selectedHistory = history.filter((r) => r.split === split && r.scenario === cost);
  const curveRows = selectedHistory.filter((r) => curveMode === "ensembles" ? isEnsemble(r) : algorithm(r) === curveMode && (isEnsemble(r) || isSeed(r)));
  const detailOptions = selected.filter((r) => isEnsemble(r) || isSeed(r));
  const chosen = detailOptions.find((r) => r.run_id === detail) ?? newP ?? detailOptions[0];
  const selectedMonths = months.filter((r) => r.run_id === chosen?.run_id);
  const selectedStats = statistics.filter((r) => r.split === split && r.scenario === cost).map((r) => ({ ...r, _key: `${r.universe_variant}:${r.algorithm}`, _variant: VARIANT[r.universe_variant] ?? r.universe_variant }));
  const changesRows = changes.map((r, index) => ({ ...r, _key: index, _old: listText(r.original_assets), _new: listText(r.ex_ada_assets), _removed: listText(r.removed_assets), _added: listText(r.added_assets), _detail: listText(r.replacement_detail), _changed: r.changed === true ? "有变化" : r.changed === false ? "相同" : "未提供" }));
  const trainingRows = training.map((r, index) => ({ ...r, _key: index, _variant: VARIANT[r.universe_variant] ?? r.universe_variant, _steps: first(r, ["timesteps", "total_timesteps", "steps"]), _elapsed: first(r, ["elapsed_seconds", "wall_seconds", "training_seconds"]), _hash: first(r, ["model_sha256", "checkpoint_sha256", "sha256"]) }));
  const limits = `## 怎样解释本次变化\n\n**实验起点。** 新方案明确排除CMC资产ID 2010（ADA），其他排除条件与当时Binance现货可交易条件保持既定口径，按历史排名继续向后选择直到补足10个资产槽位。它不是简单把原ADA目标权重永久换成现金。每个选择边界的实际移除、补入与完整成分保留在资产池记录中。\n\n**重新训练。** PPO与A2C各保留5个种子、每个32,768步、MLP[32,32]；新模型在剔除ADA后的训练资产池中训练，主组合仍平均五个模型目标权重，分别执行现金／库存账本。相同编号种子的配对用于描述变化，不构成五份独立市场实验，也不据此进行显著性或普遍盈利推断。\n\n**验证与测试。** 名义训练2023-01-01至2025-07-25；验证2025-07-25至2025-10-23；测试2025-10-23至2026-10-23，均UTC、右端不含。特征需要31个完整历史UTC日，未就绪成分不可买入，资金可分给可投成分或现金。两档评估成本复用各自同一套冻结模型；压力成本不重新训练。研究过程已使用过该365天，测试称谓仅表示模型切分，不表示研究选择未见过这段行情。\n\n**相同交易口径。** 每次10,000 USDT，无借币、杠杆或BNB抵扣；目标现金至少5%、单币最多25%。每日00:00使用已收盘信息，00:00小时开盘先卖、01:00小时开盘再买，两腿分别扣费。BTC／ETH持有参考各初始47.5%，属于25%上限的被动基准例外。主成本单边手续费0.10%、不利滑点0.02%；压力成本0.20%、0.05%。市值加权限仓后的余款留现金，比较收益和回撤时同时查看平均仓位。\n\n**数据精度。** 权益与最大回撤采用完整小时收盘及初始资金，图表按日末显示；小时内更深跌幅不可见，不能直接与此前分钟回撤比较。历史市值网页为回溯版本，使用周快照加2日滞后，不能证明严格原始发布时可得。允许小数币数量，未重建历史交易所数量步长、最小名义金额和订单簿深度。`;

  return <article className="report-content rl-ex-ada-report" aria-label="剔除ADA后历史动态Top10敏感性实验">
    <header className="report-hero"><h1 data-data-app-title contentEditable={canEdit && mode === "edit"} suppressContentEditableWarning onBlur={canEdit && mode === "edit" ? (event) => setAppTitle(event.currentTarget.textContent.trim() || appTitle) : undefined}>{appTitle}</h1><RichNarrative id="rl-ex-ada:deck" className="report-deck" value="原资产池与剔除ADA后顺位补足10币 · PPO／A2C重新训练 · 同一年事后敏感性实验" /></header>
    <ReportSection id="rl-ex-ada-summary" title="结果摘要" queryId="strategy_summary" sourceRows={selected} showHeading={false}><RichNarrative id="rl-ex-ada-summary:body" value={opening} /></ReportSection>
    <div className="rl-ex-ada-controls" aria-label="评估显示条件"><Dropdown label="评估窗口" showLabel value={split} choices={["test", "validation"]} choiceLabels={SPLIT} onChange={setSplit} /><Dropdown label="成本情景" showLabel value={cost} choices={["base", "stress"]} choiceLabels={COST} onChange={setCost} /><Dropdown label="曲线对照" showLabel value={curveMode} choices={["ensembles", "PPO", "A2C"]} choiceLabels={CURVES} onChange={setCurveMode} /></div>
    <EvidenceChart id="rl-ex-ada-return" title="原池与剔除ADA后的累计扣费净收益" queryId="equity_history" sourceRows={curveRows} rows={pivot(curveRows, "net_return")} spec={spec(curveRows, "net_return")} height={410} />
    <EvidenceChart id="rl-ex-ada-drawdown" title="日末相对小时历史峰值的回撤" queryId="equity_history" sourceRows={curveRows} rows={pivot(curveRows, "drawdown")} spec={spec(curveRows, "drawdown")} height={320} />
    <ReportSection id="rl-ex-ada-deltas-heading" title="主组合变化" queryId="strategy_summary" sourceRows={ensembles} showHeading={false}><RichNarrative id="rl-ex-ada-deltas:body" value="## 两套算法的主组合分别改变多少\n\n变化统一为“剔除ADA后减原池”，以百分点表示；小时回撤为负值，因此回撤变化为正代表较浅。比较时同时查看费用和平均仓位。" /></ReportSection>
    <DataComponent id="rl-ex-ada-deltas" title="主组合配对变化" queryId="strategy_summary" sourceRows={ensembles} displayRows={deltaRows} kind="table"><DataTable rows={deltaRows} rowKey="algorithm" searchable={false} compactNumbers={false} caption="剔除ADA后与原池主组合的差值" columns={[{ field: "algorithm", label: "算法" }, pc("original_return", "原池收益"), pc("ex_ada_return", "新池收益"), dc("return_delta", "收益变化"), pc("original_drawdown", "原池小时回撤"), pc("ex_ada_drawdown", "新池小时回撤"), dc("drawdown_delta", "回撤变化"), nc("original_fees", "原池费用USDT"), nc("ex_ada_fees", "新池费用USDT"), pc("original_exposure", "原池平均仓位"), pc("ex_ada_exposure", "新池平均仓位")]} /></DataComponent>
    <DataComponent id="rl-ex-ada-main-table" title="四套主组合完整指标" queryId="strategy_summary" sourceRows={ensembles} displayRows={display(ensembles)} kind="table"><DataTable rows={display(ensembles)} rowKey="run_id" searchable={false} compactNumbers={false} caption="四套实际执行主组合" columns={[{ field: "_name", label: "主组合" }, pc("net_return", "扣费净收益"), pc("max_drawdown", "小时最大回撤"), nc("sharpe_daily", "日Sharpe"), nc("fees", "费用USDT"), nc("slippage_cost", "滑点USDT"), nc("turnover", "周转倍数"), pc("average_exposure", "平均仓位"), pc("net_liquidation_return", "扣退出成本后")]} /></DataComponent>
    <ReportSection id="rl-ex-ada-seed-heading" title="匹配种子变化" queryId="strategy_summary" sourceRows={models} showHeading={false}><RichNarrative id="rl-ex-ada-seed:body" value="## 全部种子按编号配对\n\n每种算法的五个种子均保留。编号相同便于核对固定预算下的变化；资产池与训练轨迹不同，不能将差值解释为只移除一次ADA仓位的收益。" /></ReportSection>
    <DataComponent id="rl-ex-ada-seed-pairs" title="十组配对种子变化" queryId="strategy_summary" sourceRows={models} displayRows={pairs} kind="table"><DataTable rows={pairs} rowKey="_key" searchable={false} compactNumbers={false} caption="相同算法和种子编号的差值" columns={[{ field: "algorithm", label: "算法" }, nc("seed", "种子", 0), pc("original_return", "原池收益"), pc("ex_ada_return", "新池收益"), dc("return_delta", "收益变化"), pc("original_drawdown", "原池小时回撤"), pc("ex_ada_drawdown", "新池小时回撤"), dc("drawdown_delta", "回撤变化"), nc("fee_delta", "费用变化USDT"), dc("exposure_delta", "平均仓位变化")]} /></DataComponent>
    <DataComponent id="rl-ex-ada-statistics" title="四组五种子的分散程度" queryId="seed_statistics" sourceRows={statistics.filter((r) => r.split === split && r.scenario === cost)} displayRows={selectedStats} kind="table"><DataTable rows={selectedStats} rowKey="_key" searchable={false} caption="原池与新池的完整种子统计" columns={[{ field: "_variant", label: "资产池" }, { field: "algorithm", label: "算法" }, nc("seeds", "种子数", 0), pc("median_net_return", "收益中位数"), pc("worst_net_return", "最差收益"), pc("best_net_return", "最高收益"), pc("median_max_drawdown", "回撤中位数"), pc("worst_max_drawdown", "最差回撤")]} /></DataComponent>
    <DataComponent id="rl-ex-ada-baselines" title="原池与新池的同口径基准" queryId="strategy_summary" sourceRows={baselines} displayRows={display(baselines)} kind="table"><DataTable rows={display(baselines)} rowKey="run_id" searchable={false} caption="资产池变化下的基准收益与仓位" columns={[{ field: "_name", label: "基准" }, pc("net_return", "扣费净收益"), pc("max_drawdown", "小时最大回撤"), nc("fees", "费用USDT"), pc("average_exposure", "平均仓位"), pc("net_liquidation_return", "扣退出成本后")]} /></DataComponent>
    <EvidenceChart id="rl-ex-ada-months" title="所选主组合或种子的月度净收益" queryId="monthly_returns" sourceRows={selectedMonths} rows={selectedMonths.map((r) => ({ ...r, "月收益(%)": valid(r.return) ? r.return * 100 : null }))} spec={{ type: "bar", x: "month", y: "月收益(%)", stackable: false, valueDecimals: 2 }} height={310} headerControls={<Dropdown label="查看组合" showLabel value={chosen?.run_id ?? ""} choices={detailOptions.map((r) => r.run_id)} choiceLabels={Object.fromEntries(detailOptions.map((r) => [r.run_id, name(r)]))} onChange={setDetail} />} />
    <DataComponent id="rl-ex-ada-month-table" title="所选组合的月度精确结果" queryId="monthly_returns" sourceRows={selectedMonths} displayRows={selectedMonths} kind="table"><DataTable rows={selectedMonths} searchable={false} caption="月度扣费净收益" columns={[{ field: "month", label: "UTC月份" }, pc("return", "净收益")]} /></DataComponent>
    <ReportSection id="rl-ex-ada-universe-heading" title="每个选择边界的实际替补" queryId="universe_changes" sourceRows={changes} showHeading={false}><RichNarrative id="rl-ex-ada-universe:body" value="## 剔除ADA后由哪些资产补入\n\n每次仍按当时历史排名及交易可用条件顺位选足10个槽位。表中保留完整成分和变化边界；同一时点的特征就绪掩码另决定哪些资产可以买入，资产池成分不等于实际持仓。" /></ReportSection>
    <DataComponent id="rl-ex-ada-universe-table" title="历史选择边界及顺位替补" queryId="universe_changes" sourceRows={changes} displayRows={changesRows} kind="table"><DataTable rows={changesRows} rowKey="_key" searchable caption="每次历史资产池选择的变化" columns={[{ field: "anchor", label: "选择边界UTC" }, { field: "_changed", label: "成分变化" }, { field: "_removed", label: "移除" }, { field: "_added", label: "新增" }, { field: "_detail", label: "排名／替补说明" }, { field: "_old", label: "原池10槽位" }, { field: "_new", label: "新池10槽位" }]} /></DataComponent>
    <DataComponent id="rl-ex-ada-training" title="原池与新池的全部冻结模型" queryId="training_metrics" sourceRows={training} displayRows={trainingRows} kind="table"><DataTable rows={trainingRows} rowKey="_key" searchable={false} caption="全部模型的固定训练预算与来源" columns={[{ field: "_variant", label: "资产池" }, { field: "algorithm", label: "算法" }, nc("seed", "种子", 0), nc("_steps", "训练步数", 0), nc("_elapsed", "训练秒数", 1), { field: "_hash", label: "模型SHA256" }]} /></DataComponent>
    <ReportSection id="rl-ex-ada-methods" title="敏感性实验的解释边界" queryId="strategy_summary" sourceRows={summary} showHeading={false}><RichNarrative id="rl-ex-ada-methods:body" value={limits} /></ReportSection>
  </article>;
}

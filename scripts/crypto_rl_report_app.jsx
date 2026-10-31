import React, { useState } from "react";
import { DataComponent, DataTable, Dropdown, EvidenceChart, ReportSection, RichNarrative, useDataApp } from "../../data-app-public.jsx";
import "./crypto-rl-report.css";

const COST = { base: "主成本", stress: "压力成本" };
const SPLIT = { test: "测试365天", validation: "验证90天", val: "验证90天" };
const CURVES = { ensemble: "主组合与全部基准", all_seeds: "五个种子与主组合" };
const valid = (v) => typeof v === "number" && Number.isFinite(v);
const pct = (v, digits = 2) => valid(v) ? `${(v * 100).toFixed(digits)}%` : "—";
const num = (v, digits = 2) => valid(v) ? v.toLocaleString("zh-CN", { minimumFractionDigits: digits, maximumFractionDigits: digits }) : "—";
const median = (values) => { const v = values.filter(valid).sort((a, b) => a - b); return v.length ? v.length % 2 ? v[(v.length - 1) / 2] : (v[v.length / 2 - 1] + v[v.length / 2]) / 2 : null; };
const pc = (field, label) => ({ field, label, renderCell: (value) => pct(value) });
const nc = (field, label, digits = 2) => ({ field, label, renderCell: (value) => num(value, digits) });
const isModel = (r) => r.seed !== null && r.seed !== undefined && valid(Number(r.seed));
const first = (row, keys, fallback = null) => { for (const key of keys) if (row[key] !== undefined && row[key] !== null) return row[key]; return fallback; };
const dateText = (value) => value === null || value === undefined ? "—" : String(value).replace("T", " ").replace(/\.000Z$/u, "Z");

function pivot(rows, measure) {
  const data = new Map();
  rows.forEach((row) => { if (!data.has(row.timestamp)) data.set(row.timestamp, { timestamp: row.timestamp }); data.get(row.timestamp)[`${row.label}(%)`] = valid(row[measure]) ? row[measure] * 100 : null; });
  return [...data.values()].sort((a, b) => a.timestamp.localeCompare(b.timestamp));
}
function spec(rows, measure, curveMode = "ensemble") {
  const labels = [...new Set(rows.map((r) => r.label))];
  const fields = labels.map((label) => `${label}(%)`);
  const seedLabels = [...new Set(rows.filter(isModel).map((r) => r.label))];
  return { type: "line", x: "timestamp", y: fields[0] ?? "净收益(%)", fields, stackable: false, valueDecimals: 2,
    colors: Object.fromEntries(labels.map((label) => {
      const row = rows.find((r) => r.label === label);
      const color = seedLabels.includes(label) ? `var(--chart-${seedLabels.indexOf(label) % 5 + 1})`
        : row?.strategy === "ppo_ensemble" ? curveMode === "ensemble" ? "var(--chart-1)" : "var(--text)"
        : row?.strategy === "top10_cap" ? "#797979" : row?.strategy === "btc_eth_hold" ? "#a7a7a7"
        : row?.strategy === "cash" ? "var(--secondary)" : "var(--text)";
      return [`${label}(%)`, color];
    })),
    legend: { labels: Object.fromEntries(fields.map((field) => [field, field.replace(/\(%\)$/u, "")])) },
    yLabel: measure === "drawdown" ? "距小时历史峰值（%）" : "扣费累计净收益（%）" };
}

export function ReportContent() {
  const { snapshot, reviewedPeriodRows, appTitle, setAppTitle, canEdit, mode, visible } = useDataApp();
  const summary = reviewedPeriodRows("strategy_summary", { period: "history" });
  const history = reviewedPeriodRows("equity_history", { period: "history" });
  const months = reviewedPeriodRows("monthly_returns", { period: "history" });
  const universe = reviewedPeriodRows("universe_history", { period: "history" });
  const training = reviewedPeriodRows("training_metrics", { period: "history" });
  const holdings = snapshot.queries?.final_holdings ? reviewedPeriodRows("final_holdings", { period: "history" }) : [];
  const config = snapshot.report?.studyConfig ?? {};
  const splitChoices = [...new Set(summary.map((r) => r.split))].filter((value) => ["test", "validation", "val"].includes(value));
  const [split, setSplit] = useState(splitChoices.includes("test") ? "test" : splitChoices[0]);
  const [cost, setCost] = useState("base");
  const [curveMode, setCurveMode] = useState("ensemble");
  const seedChoices = [...new Set(summary.filter(isModel).map((r) => String(r.seed)))].sort((a, b) => Number(a) - Number(b));
  const detailChoices = ["ensemble", ...seedChoices];
  const detailLabels = { ensemble: "五模型平均权重主组合", ...Object.fromEntries(seedChoices.map((id) => [id, `单模型 seed ${id}`])) };
  const [seed, setSeed] = useState("ensemble");
  const selected = summary.filter((r) => r.split === split && r.scenario === cost);
  const models = selected.filter(isModel);
  const ensemble = selected.find((r) => r.strategy === "ppo_ensemble");
  const benchmarks = selected.filter((r) => !isModel(r) && r.strategy !== "ppo_ensemble");
  const testRows = summary.filter((r) => r.split === "test" && r.scenario === "base" && isModel(r));
  const testStress = summary.filter((r) => r.split === "test" && r.scenario === "stress" && isModel(r));
  const returns = models.map((r) => r.net_return).filter(valid);
  const drawdowns = models.map((r) => r.max_drawdown).filter(valid);
  const equal = benchmarks.find((r) => /equal/u.test(r.strategy) && !/hold/u.test(r.strategy));
  const capped = benchmarks.find((r) => r.strategy === "top10_cap");
  const stressEnsemble = summary.find((r) => r.strategy === "ppo_ensemble" && r.split === "test" && r.scenario === "stress");
  const baseEnsemble = summary.find((r) => r.strategy === "ppo_ensemble" && r.split === "test" && r.scenario === "base");
  const opening = models.length
    ? `## 结果摘要\n\n- 当前为 **${SPLIT[split] ?? split} · ${COST[cost]}**。事前固定的**五模型平均目标权重组合**，扣费收益 **${pct(ensemble?.net_return)}**，小时最大回撤 **${pct(ensemble?.max_drawdown)}**；扣末端退出成本后为 **${pct(ensemble?.net_liquidation_return)}**。它是单独执行、独立记账的实际模拟组合，没有用最佳测试种子代替主策略。\n- 同窗口等权限仓基准收益 **${pct(equal?.net_return)}**，回撤 **${pct(equal?.max_drawdown)}**；市值加权限仓基准收益 **${pct(capped?.net_return)}**，回撤 **${pct(capped?.max_drawdown)}**。主组合平均仓位 **${pct(ensemble?.average_exposure)}**，市值加权限仓版 **${pct(capped?.average_exposure)}**。限仓后余款留现金，因此风险暴露不同，不能把收益／回撤差异单独归为择时或选币能力。\n- 固定测试期主组合在主成本下为 **${pct(baseEnsemble?.net_return)}**，压力成本下为 **${pct(stressEnsemble?.net_return)}**，均复用同一套冻结模型。当前窗口全部 **${models.length} 个单种子**收益中位数 **${pct(median(returns))}**，最差 **${pct(returns.length ? Math.min(...returns) : null)}**；这描述训练波动，不替代主组合的账本结果。\n- 资产池来自**历史快照回溯**，不能证明原始发布时已可获得的严格point-in-time信息。模型训练不读取测试数据，但研究者已看过本年其他策略结果；这不是完全独立的新研究样本。本次回撤使用**小时收盘权益**。`
    : "## 结果摘要\n\n当前选择没有完整PPO评估记录，暂不生成收益结论。";
  const allHistory = history.filter((r) => r.split === split && r.scenario === cost);
  const curveRows = allHistory.filter((r) => curveMode === "ensemble" ? !isModel(r) : isModel(r) || r.strategy === "ppo_ensemble");
  const dates = [...new Set(curveRows.map((r) => r.timestamp))].sort();
  const selectedMonths = months.filter((r) => r.split === split && r.scenario === cost && (seed === "ensemble" ? r.strategy === "ppo_ensemble" : isModel(r) && String(r.seed) === seed));
  const seedComparison = seedChoices.map((id) => {
    const row = { seed: Number(id) };
    for (const subset of splitChoices) for (const scenario of ["base", "stress"]) {
      const evidence = summary.find((r) => String(r.seed) === id && r.split === subset && r.scenario === scenario);
      row[`${subset}_${scenario}_return`] = evidence?.net_return ?? null;
      row[`${subset}_${scenario}_drawdown`] = evidence?.max_drawdown ?? null;
    }
    return row;
  });
  const allModelRows = summary.filter(isModel);
  const universeRows = universe.map((row, index) => ({ ...row, _row: index,
    _date: first(row, ["anchor", "rebalance_time", "rebalance_at", "selection_time", "effective_time", "date", "timestamp"]),
    _snapshot: first(row, ["snapshot_date", "source_date", "as_of_date", "rank_date"]),
    _available: first(row, ["available_after", "available_at", "available_time", "snapshot_available_at"]),
    _rank: first(row, ["cmc_rank", "rank", "market_cap_rank", "global_rank"]), _asset: first(row, ["symbol", "asset_key", "name"]),
    _cap: first(row, ["market_cap", "market_cap_usd", "cap"]),
  }));
  const trainingRows = training.map((row, index) => ({ ...row, _row: index,
    _steps: first(row, ["timesteps", "total_timesteps", "trained_steps", "steps"]),
    _elapsed: first(row, ["wall_seconds", "elapsed_seconds", "training_seconds"]),
    _valReturn: first(row, ["validation_net_return", "validation_return", "val_return"]),
    _hash: first(row, ["model_sha256", "checkpoint_sha256", "sha256"]),
  }));
  const quality = training[0]?.training_window_metadata ?? {};
  const selectedHoldings = holdings.filter((r) => r.split === split && r.scenario === cost && (seed === "ensemble" ? r.strategy === "ppo_ensemble" : isModel(r) && String(r.seed) === seed));
  const trainingText = `## 怎样训练，怎样检验\n\n- **算法与框架。** 使用独立Gymnasium现货组合环境，由Stable-Baselines3的PPO学习目标权重。每个环境步推进一个模拟交易日，奖励为扣费后净值变化的对数乘100。\n- **时间分离。** 名义训练为2023-01-01至2025-07-25，验证90天为2025-07-25至2025-10-23；固定测试为2025-10-23至2026-10-23，全部UTC、右端不含。特征采用事前固定因果比例缩放与裁剪；验证与测试在本报告分开展示。\n- **实际训练覆盖。** 名义训练 **${num(quality.requested_train_days, 0)}天**，按 **${num(quality.episode_days, 0)}日**完整片段抽样；${num(quality.candidate_windows, 0)}个候选窗口中，可用${num(quality.eligible_windows, 0)}个，排除${num(quality.excluded_windows, 0)}个。跨越2023-03-24 13:00 UTC真实停市缺失小时的片段被排除，可用片段并集覆盖 **${num(quality.covered_training_days, 0)}天**；这不表示每个种子都遍历了全部日期。验证和测试没有删除日期，也没有填造缺失价格。\n- **训练预算与主组合。** 每个种子32,768步，MLP为[32,32]，CPU独立环境；5个种子全部报告。主策略事前固定为五个模型目标权重取平均，在一个组合账本中实际执行。两档评估成本复用同一套冻结模型，不针对压力结果重新训练。\n- **统计边界。** 五个种子是在同一市场历史上的五次训练，不是五份独立行情；不以最佳测试种子作为主结论。研究者已看过本年其他策略表现，模型训练虽不读取测试，这仍不是全新独立研究样本。`;
  const methodText = `## 组合规则、费用与行情精度\n\n**资产池与特征就绪。** 月初及训练／验证／测试起点选取当时市值排名靠前、符合排除规则且当时Binance现货可交易的10个资产。使用CMC周快照日期加2天滞后；当前取得的历史页面是回溯材料，不能证明原始发布时点。稳定币及包装资产剔除，身份映射与每次成分详见资产池证据。每个槽位需要连续31个完整历史UTC日及有效价量特征才允许买入；未就绪的成分不可投资，测试末HYPE仍未就绪。10个池内槽位不代表始终实际持有10币，余款可留现金。\n\n**权重与执行。** 初始10,000 USDT；PPO及动态Top10等权／市值加权限仓版的扣费后目标现金至少5%、单币最多25%，超限余款留现金。BTC／ETH等权持有是被动参考，各币初始47.5%，无25%单币上限；费用、成交时序与估值频率一致。全部无杠杆、不借币、不使用BNB抵扣。实际权重可随价格漂移，目标上限不代表连续实时限制。每日00:00边界使用已收盘信息，先在00:00小时开盘卖出，再于01:00小时开盘用净现金买入；两腿分别扣费，现金与持币数量不得为负。\n\n**成本与数量精度。** 主成本单边手续费0.10%、不利滑点0.02%；压力成本分别0.20%、0.05%，均为研究假设。成交数量允许小数币，没有重建历史数量步长、最小下单量、最小名义金额及盘口深度，不能据此认定逐笔满足当时全部交易所过滤器。期末保留实际持仓估值，另按全部持币市值扣一次假设退出费与滑点计算净清算值。\n\n**权益与指标。** 价格和估值粒度为1小时，最大回撤读取完整小时收盘权益并包含期初资金；小时内部更深的跌幅可能未被捕捉，不能把它与前两轮分钟回撤直接比较。图表按日末显示，因此图上最深值不一定等于小时最大回撤。日Sharpe以UTC日末收益计算，包含首日成本、无风险利率设0。首尾月按实际区间，不补齐。多次加减仓的组合不计算单仓胜率或逐笔盈亏比；费用、周转和末端持仓均来自账本。`;
  const references = (snapshot.report?.officialReferences ?? []).map((item) => `- [${item.label}](${item.url})`).join("\n");

  return <article className="report-content crypto-rl-report" aria-label="动态Top10 PPO组合研究报告">
    <header className="report-hero"><h1 data-data-app-title contentEditable={canEdit && mode === "edit"} suppressContentEditableWarning onBlur={canEdit && mode === "edit" ? (event) => setAppTitle(event.currentTarget.textContent.trim() || appTitle) : undefined}>{appTitle}</h1>
      <RichNarrative id="crypto-rl:deck" className="report-deck" value="历史动态Top10现货组合 · 固定测试期2025-10-23至2026-10-23（UTC，右端不含）" />
    </header>
    {visible("crypto-rl-summary") && <ReportSection id="crypto-rl-summary" title="结果摘要" queryId="strategy_summary" sourceRows={summary} showHeading={false}><RichNarrative id="crypto-rl-summary:body" value={opening} /></ReportSection>}
    <div className="crypto-rl-controls" aria-label="评估显示条件">
      <Dropdown label="评估窗口" showLabel value={split} choices={splitChoices} choiceLabels={SPLIT} onChange={setSplit} />
      <Dropdown label="成本情景" showLabel value={cost} choices={["base", "stress"]} choiceLabels={COST} onChange={setCost} />
    </div>
    <ReportSection id="crypto-rl-curves-heading" title="实际组合曲线" queryId="equity_history" sourceRows={curveRows} showHeading={false}><RichNarrative id="crypto-rl-curves:body" value={`## 主组合、基准与种子差异\n\n当前为**${SPLIT[split] ?? split} · ${COST[cost]}**，覆盖${dates[0]?.slice(0, 10) ?? "—"}至${dates.at(-1)?.slice(0, 10) ?? "—"}的日收盘边界。曲线显示**${CURVES[curveMode]}**；全部种子和基准也保留在下方精确表中。`} /></ReportSection>
    <div className="crypto-rl-controls" aria-label="曲线比较条件"><Dropdown label="曲线对照" showLabel value={curveMode} choices={["ensemble", "all_seeds"]} choiceLabels={CURVES} onChange={setCurveMode} /></div>
    <EvidenceChart id="crypto-rl-return" title="实际组合的累计扣费净收益" queryId="equity_history" sourceRows={curveRows} rows={pivot(curveRows, "net_return")} spec={spec(curveRows, "net_return", curveMode)} height={400} />
    <EvidenceChart id="crypto-rl-drawdown" title="日末相对小时历史峰值的回撤" queryId="equity_history" sourceRows={curveRows} rows={pivot(curveRows, "drawdown")} spec={spec(curveRows, "drawdown", curveMode)} height={320} />
    <DataComponent id="crypto-rl-summary-table" title="全部种子与同口径基准" queryId="strategy_summary" sourceRows={selected} displayRows={selected} kind="table">
      <DataTable rows={selected} rowKey="run_id" searchable={false} compactNumbers={false} caption="PPO种子与同口径基准结果" columns={[
        { field: "label", label: "模型／基准" }, pc("net_return", "扣费净收益"), pc("max_drawdown", "小时最大回撤"), nc("sharpe_daily", "日Sharpe"), nc("calmar", "Calmar"), nc("fills", "成交腿数", 0), nc("fees", "费用USDT"), nc("turnover", "周转倍数"), pc("average_exposure", "平均仓位"), nc("final_cash", "期末现金USDT"), pc("net_liquidation_return", "扣末端退出成本后"),
      ]} />
    </DataComponent>
    <ReportSection id="crypto-rl-splits-heading" title="验证与测试分开" queryId="strategy_summary" sourceRows={allModelRows} showHeading={false}><RichNarrative id="crypto-rl-splits:body" value="## 验证、测试与成本压力\n\n下表每个种子始终保留，验证与测试分别计算收益及小时回撤。压力成本复用同一模型，没有按测试排名筛选种子。" /></ReportSection>
    <DataComponent id="crypto-rl-seed-table" title="每个种子的验证与测试结果" queryId="strategy_summary" sourceRows={allModelRows} displayRows={seedComparison} kind="table"><DataTable rows={seedComparison} rowKey="seed" searchable={false} compactNumbers={false} caption="全部PPO种子跨窗口与成本比较" columns={[nc("seed", "种子", 0), ...splitChoices.flatMap((subset) => ["base", "stress"].flatMap((scenario) => [pc(`${subset}_${scenario}_return`, `${SPLIT[subset]}·${COST[scenario]}收益`), pc(`${subset}_${scenario}_drawdown`, `${SPLIT[subset]}·${COST[scenario]}回撤`)]))]} /></DataComponent>
    <EvidenceChart id="crypto-rl-months" title="所选模型组合的月度扣费收益" queryId="monthly_returns" sourceRows={selectedMonths} rows={selectedMonths.map((r) => ({ ...r, "月收益(%)": valid(r.return) ? r.return * 100 : null }))} spec={{ type: "bar", x: "month", y: "月收益(%)", stackable: false, valueDecimals: 2 }} height={300} headerControls={<Dropdown label="查看模型组合" showLabel value={seed} choices={detailChoices} choiceLabels={detailLabels} onChange={setSeed} />} />
    <DataComponent id="crypto-rl-month-table" title="所选模型组合的月度精确结果" queryId="monthly_returns" sourceRows={selectedMonths} displayRows={selectedMonths} kind="table"><DataTable rows={selectedMonths} searchable={false} caption="月度净收益" columns={[{ field: "month", label: "UTC月份" }, pc("return", "净收益")]} /></DataComponent>
    {holdings.length > 0 && <DataComponent id="crypto-rl-holdings" title="所选模型组合的期末实际持仓" queryId="final_holdings" sourceRows={selectedHoldings} displayRows={selectedHoldings} kind="table"><DataTable rows={selectedHoldings} searchable={false} compactNumbers={false} caption="期末持币账本" columns={[{ field: "symbol", label: "资产" }, nc("qty", "数量", 8), nc("mark_price", "估值价格USDT", 6), nc("market_value", "市值USDT"), pc("weight", "实际权重")]} /></DataComponent>}
    <ReportSection id="crypto-rl-training-heading" title="训练与验证边界" queryId="training_metrics" sourceRows={training} showHeading={false}><RichNarrative id="crypto-rl-training:body" value={trainingText} /></ReportSection>
    <DataComponent id="crypto-rl-training-table" title="每个种子的训练记录" queryId="training_metrics" sourceRows={training} displayRows={trainingRows} kind="table"><DataTable rows={trainingRows} rowKey="_row" searchable={false} caption="训练预算与冻结模型记录" columns={[nc("seed", "种子", 0), nc("_steps", "训练步数", 0), nc("_elapsed", "训练秒数", 1), pc("_valReturn", "验证净收益"), { field: "_hash", label: "冻结模型SHA256", renderCell: (value) => value ? String(value) : "—" }]} /></DataComponent>
    <ReportSection id="crypto-rl-universe-heading" title="历史资产池" queryId="universe_history" sourceRows={universe} showHeading={false}><RichNarrative id="crypto-rl-universe:body" value="## 每次选择实际使用哪些币\n\n这里保留历史资产池变更、来源快照与可用时间；缺少的字段显示缺口，不补为当时已知。历史回溯数据与严格point-in-time发布时间证据有区别。" /></ReportSection>
    <DataComponent id="crypto-rl-universe-table" title="动态Top10成分及历史来源" queryId="universe_history" sourceRows={universe} displayRows={universeRows} kind="table"><DataTable rows={universeRows} rowKey="_row" searchable caption="历史Top10选择记录" columns={[{ field: "_date", label: "选择边界UTC", renderCell: dateText }, { field: "_snapshot", label: "来源快照日期", renderCell: dateText }, { field: "_available", label: "使用滞后后的时间", renderCell: dateText }, nc("_rank", "市值排名", 0), { field: "_asset", label: "资产" }, nc("_cap", "市值USD", 0)]} /></DataComponent>
    <ReportSection id="crypto-rl-methods" title="交易与指标口径" queryId="strategy_summary" sourceRows={summary} showHeading={false}><RichNarrative id="crypto-rl-methods:body" value={methodText} />{references && <RichNarrative id="crypto-rl-references:body" value={`### 框架和数据的官方参考\n\n${references}\n\n框架文档说明接口，不能为本次策略收益背书。完整方法、历史数据来源及复现命令保存在METHODOLOGY.md与来源清单中。`} />}</ReportSection>
  </article>;
}

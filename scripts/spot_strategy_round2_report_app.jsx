import React, { useMemo, useState } from "react";
import { DataComponent, DataTable, Dropdown, EvidenceChart, ReportSection, RichNarrative, useDataApp } from "../../data-app-public.jsx";
import "./spot-round2-report.css";

const STRATEGIES = ["supertrend", "squeeze", "ibs", "mfi", "absolute_momentum"];
const COLORS = Object.fromEntries(STRATEGIES.map((id, index) => [id, `var(--chart-${index + 1})`]));
COLORS.buy_hold_equal = "var(--text)";
COLORS.cash = "var(--secondary)";
const SCENARIO = { base: "主成本", stress: "压力成本" };
const VARIANT = { original: "原规则", stop5: "5% 保护", benchmark: "基准" };
const valid = (value) => typeof value === "number" && Number.isFinite(value);
const pct = (value, digits = 2) => valid(value) ? `${(value * 100).toFixed(digits)}%` : "—";
const num = (value, digits = 2) => valid(value) ? value.toLocaleString("zh-CN", { minimumFractionDigits: digits, maximumFractionDigits: digits }) : "—";
const pc = (field, label) => ({ field, label, renderCell: (value) => pct(value) });
const nc = (field, label, digits = 2) => ({ field, label, renderCell: (value) => num(value, digits) });
const scope = (rows, scenario, variant) => rows.filter((r) => r.pair === "PORTFOLIO" && r.scenario === scenario && (r.variant === variant || r.variant === "benchmark"));

function pivot(rows, measure, labeler = (row) => row.label) {
  const records = new Map();
  rows.forEach((row) => {
    if (!records.has(row.timestamp)) records.set(row.timestamp, { timestamp: row.timestamp });
    records.get(row.timestamp)[`${labeler(row)}(%)`] = valid(row[measure]) ? row[measure] * 100 : null;
  });
  return [...records.values()].sort((a, b) => a.timestamp.localeCompare(b.timestamp));
}

function curveSpec(rows, measure, labeler = (row) => row.label) {
  const fields = [...new Set(rows.map((row) => `${labeler(row)}(%)`))];
  return { type: "line", x: "timestamp", y: fields[0] ?? "净收益(%)", fields,
    colors: Object.fromEntries(rows.map((r) => [`${labeler(r)}(%)`, COLORS[r.strategy] ?? "var(--secondary)"])),
    stackable: false, valueDecimals: 2,
    yLabel: measure === "drawdown" ? "距历史峰值（%）" : "累计净收益（%）",
    legend: { labels: Object.fromEntries(fields.map((field) => [field, field.replace(/\(%\)$/u, "")])) } };
}

function SummaryTable({ id, title, rows }) {
  return <DataComponent id={id} title={title} queryId="strategy_summary" kind="table" sourceRows={rows} displayRows={rows}>
    <DataTable rows={rows} rowKey="run_id" searchable={false} compactNumbers={false} caption={title} columns={[
      { field: "label", label: "策略" },
      { field: "pair", label: "资金分组", renderCell: (value) => value === "PORTFOLIO" ? "两币组合" : value.replace("USDT", "／USDT") },
      pc("net_return", "扣费净收益"), pc("max_drawdown", "分钟最大回撤"), nc("sharpe_daily", "日收益 Sharpe"), nc("calmar", "Calmar"),
      nc("closed_trades", "平仓笔数", 0), pc("win_rate", "胜率"), nc("payoff_ratio", "平均盈亏比"), nc("fees", "费用 USDT"),
      nc("turnover", "周转倍数"), pc("exposure", "平均仓位"), nc("average_holding_hours", "平均持仓小时"),
      nc("unrealized_pnl", "未实现盈亏 USDT"), pc("net_liquidation_return", "扣期末退出成本后"),
    ]} />
  </DataComponent>;
}

export function ReportContent() {
  const { snapshot, reviewedPeriodRows, appTitle, setAppTitle, canEdit, mode, visible } = useDataApp();
  const summary = reviewedPeriodRows("strategy_summary", { period: "history" });
  const histories = reviewedPeriodRows("equity_history", { period: "history" });
  const months = reviewedPeriodRows("monthly_returns", { period: "history" });
  const previous = reviewedPeriodRows("previous_summary", { period: "history" });
  const labels = Object.fromEntries(summary.map((row) => [row.strategy, row.label]));
  const [scenario, setScenario] = useState("base");
  const [variant, setVariant] = useState("original");
  const [focus, setFocus] = useState("supertrend");
  const config = snapshot.report?.suiteConfig ?? {};
  const main = scope(summary, "base", "original").filter((r) => STRATEGIES.includes(r.strategy));
  const oldMain = scope(previous, "base", "original").filter((r) => !["cash", "buy_hold_equal"].includes(r.strategy));
  const ranked = [...main].filter((r) => valid(r.net_return)).sort((a, b) => b.net_return - a.net_return);
  const winner = ranked[0];
  const oldWinner = [...oldMain].filter((r) => valid(r.net_return)).sort((a, b) => b.net_return - a.net_return)[0];
  const stressMain = scope(summary, "stress", "original").filter((r) => STRATEGIES.includes(r.strategy));
  const stressWinner = winner && stressMain.find((r) => r.strategy === winner.strategy);
  const profitable = main.filter((r) => r.net_return > 0).length;
  const stressProfitable = stressMain.filter((r) => r.net_return > 0).length;
  const benchmark = summary.find((r) => r.strategy === "buy_hold_equal" && r.scenario === "base" && r.pair === "PORTFOLIO");
  const opening = winner
    ? `## 结果摘要\n\n- 新五组在主成本、原规则下有 **${profitable} 组**取得正净收益；${winner.label}为 **${pct(winner.net_return)}**，分钟最大回撤 **${pct(winner.max_drawdown)}**，已平仓 **${num(winner.closed_trades, 0)} 笔**。总收益包括期末持仓浮盈亏；扣末端退出成本后为 **${pct(winner.net_liquidation_return)}**。\n- 压力成本下，新五组有 **${stressProfitable} 组**仍为正收益；主版收益最高者在压力成本下为 **${pct(stressWinner?.net_return)}**。同资金 BTC／ETH 等权持有主成本收益为 **${pct(benchmark?.net_return)}**。\n- 上一轮最高收益是${oldWinner?.label ?? "—"}的 **${pct(oldWinner?.net_return)}**。两轮均使用同一365天行情；这次是追加候选比较，**没有新增样本外验证**，也不根据排名反复修改参数。`
    : "## 结果摘要\n\n当前证据不足以比较完整的新五组，暂不生成收益排名。";
  const selected = scope(summary, scenario, variant);
  const selectedHistory = scope(histories, scenario, variant);
  const dates = [...new Set(selectedHistory.map((r) => r.timestamp))].sort();
  const coinRows = summary.filter((r) => ["BTCUSDT", "ETHUSDT"].includes(r.pair) && r.scenario === scenario && (r.variant === variant || r.variant === "benchmark"));
  const oldSelected = scope(previous, scenario, variant).filter((r) => !["cash", "buy_hold_equal"].includes(r.strategy));
  const tenRows = [...selected.filter((r) => STRATEGIES.includes(r.strategy)).map((r) => ({ ...r, round: "第二轮" })), ...oldSelected.map((r) => ({ ...r, round: "第一轮" }))].sort((a, b) => b.net_return - a.net_return);
  const comparisonSource = summary.filter((r) => r.pair === "PORTFOLIO" && STRATEGIES.includes(r.strategy));
  const comparisons = STRATEGIES.map((strategy) => {
    const row = { strategy, label: labels[strategy] ?? strategy };
    for (const cost of ["base", "stress"]) for (const ver of ["original", "stop5"]) {
      const evidence = comparisonSource.find((r) => r.strategy === strategy && r.scenario === cost && r.variant === ver);
      row[`${cost}_${ver}_return`] = evidence?.net_return ?? null;
      row[`${cost}_${ver}_drawdown`] = evidence?.max_drawdown ?? null;
    }
    return row;
  });
  const focusHistory = histories.filter((r) => r.pair === "PORTFOLIO" && r.scenario === scenario && r.strategy === focus && ["original", "stop5"].includes(r.variant));
  const focusSpec = curveSpec(focusHistory, "net_return", (r) => VARIANT[r.variant]);
  focusSpec.colors = { "原规则(%)": COLORS[focus], "5% 保护(%)": "var(--text)" };
  const monthSource = scope(months, scenario, variant).filter((r) => STRATEGIES.includes(r.strategy));
  const monthlyRows = useMemo(() => {
    const rows = new Map();
    monthSource.forEach((r) => {
      if (!rows.has(r.month)) rows.set(r.month, { month: r.month });
      rows.get(r.month)[r.strategy] = r.return;
    });
    return [...rows.values()].sort((a, b) => a.month.localeCompare(b.month));
  }, [months, scenario, variant]);
  const stability = STRATEGIES.map((strategy) => {
    const rows = monthSource.filter((r) => r.strategy === strategy && valid(r.return));
    return { label: labels[strategy] ?? strategy, months: rows.length, positive: rows.filter((r) => r.return > 0).length,
      best: rows.length ? Math.max(...rows.map((r) => r.return)) : null, worst: rows.length ? Math.min(...rows.map((r) => r.return)) : null };
  });
  const focusedMonths = monthSource.filter((r) => r.strategy === focus);
  const referenceRows = snapshot.report?.strategySources ?? [];
  const sourceLinks = referenceRows.map((source) => `- [${source.label ?? source.title ?? source.strategy_id ?? "规则来源"}](${source.source_url})${source.license ? `（${source.license}）` : ""}。`).join("\n");
  const previews = Object.fromEntries(referenceRows.filter((s) => s.preview || s.evidence_summary).map((s) => [s.source_url, { title: s.label ?? s.title ?? s.strategy_id, summary: s.preview ?? s.evidence_summary, source: s.publisher ?? "公开原始参考", approvedForReport: true }]));
  const rules = `## 新五组怎样产生信号\n\n- **${labels.supertrend ?? "Supertrend"}｜4小时。** ATR(10)×3 自适应趋势带；按TradingView递推公式收紧或重置上下带，收盘确认上行状态时空仓买入，下行状态退出。使用状态判断，带值不是分钟触价止损。\n- **${labels.squeeze ?? "波动压缩释放"}｜4小时。** 布林带为SMA20±2×总体标准差（ddof=0），Keltner为EMA20±1.5×ATR20。前根布林带严格位于Keltner内、本根释放，且收盘超过本根布林上轨、20根价格动量为正时入场；收盘跌破SMA20或动量≤0退出。\n- **${labels.ibs ?? "IBS反转"}｜日线。** IBS=(收盘−最低)/(最高−最低)，≤0.2入场、≥0.8退出；零振幅根取0.5，不发买卖信号，无额外趋势过滤。\n- **${labels.mfi ?? "MFI反转"}｜4小时。** MFI(14)加入典型价与成交量信息；前值≥20、当前<20时入场，前值≤80、当前>80退出。按穿越判断，不把持续超卖或超买重复视为新穿越。\n- **${labels.absolute_momentum ?? "绝对动量"}｜每周。** 每周一00:00 UTC判断截至此前已闭合日线的180日收益（当前收盘/180根前收盘−1）；>0持有本币，否则持USDT。同币继续持有，不按周加仓或再平衡；USDT收益率设0。\n\n这些规则在本轮数值运行前冻结。周期选择、单仓现货执行均为本轮设定；Squeeze的突破确认、普通20根价格差与退出，以及绝对动量的180日／周频／USDT替代均为明示改版，不声称复刻作者完整交易系统。三组含趋势成分，两组偏反转，不能把五组视为相互独立。`;
  const methods = `## 成本、成交与结果范围\n\n**统一资金与成本。** 每组初始 ${num(config.capital, 0)} USDT；BTC与ETH各 ${num(config.sleeve_capital, 0)} USDT 独立记账，再合为组合。每次入场最多花费现金95%，包含手续费；单仓、不加仓、不借币、不使用BNB抵扣。主成本单边手续费 ${pct(config.costs?.base?.fee)}、不利滑点 ${pct(config.costs?.base?.slippage)}；压力成本 ${pct(config.costs?.stress?.fee)}、${pct(config.costs?.stress?.slippage)}。费用为研究假设，未冒充历史账户费率。\n\n**可用信息与时点。** 共用上一轮已核验的Binance现货1分钟数据，另有210天预热，不计收益。所有信号仅使用已完成K线，最早在随后的分钟开盘成交。主版不额外加入价格止损或止盈；5%保护版单列，止损按实际入场成交价计算，跳空按更不利开盘成交并扣滑点，保护退出后等下一次严格更晚的策略判断。\n\n**权益与回撤。** 曲线用UTC日末权益；最大回撤使用完整1分钟权益，包含期初本金。期末未强平，收益含浮盈亏；表中另列假设扣除期末卖出成本的净清算收益。胜率、平均盈亏比和平均持仓时间仅计已平仓交易，无交易或分母缺失显示“—”。月度按月末权益计算，首尾月为部分月份，不填补。\n\n**比较的局限。** 第一轮和第二轮反复使用同一历史窗口，追加候选会增加选择偏差；没有新增样本外证据。历史盘口、排队与完整历史手续费／过滤器未重建，过滤器仅用带采集时间的快照。各组只有现货多头和现金，无法天然回避全部市场下跌；结果不构成实盘授权。`;

  return <article className="report-content spot-round2-report" aria-label="第二轮现货策略研究报告">
    <header className="report-hero">
      <h1 data-data-app-title contentEditable={canEdit && mode === "edit"} suppressContentEditableWarning onBlur={canEdit && mode === "edit" ? (event) => setAppTitle(event.currentTarget.textContent.trim() || appTitle) : undefined}>{appTitle}</h1>
      <RichNarrative id="round2:deck" className="report-deck" value={`BTCUSDT／ETHUSDT · ${String(config.start ?? "").slice(0, 10)} 至 ${String(config.end ?? "").slice(0, 10)}（右端不含，UTC）`} />
    </header>
    {visible("round2-summary") && <ReportSection id="round2-summary" title="结果摘要" queryId="strategy_summary" queryIds={["strategy_summary", "previous_summary"]} sourceRowsByQuery={{ strategy_summary: [...main, ...stressMain, ...(benchmark ? [benchmark] : [])], previous_summary: oldMain }} showHeading={false}>
      <RichNarrative id="round2-summary:body" value={opening} />
    </ReportSection>}
    <div className="spot-round2-controls" aria-label="报告显示条件">
      <Dropdown label="成本情景" showLabel value={scenario} choices={["base", "stress"]} choiceLabels={SCENARIO} onChange={setScenario} />
      <Dropdown label="策略版本" showLabel value={variant} choices={["original", "stop5"]} choiceLabels={VARIANT} onChange={setVariant} />
    </div>
    <ReportSection id="round2-curve-heading" title="组合比较" queryId="equity_history" sourceRows={selectedHistory} showHeading={false}>
      <RichNarrative id="round2-curves:body" value={`## 新五组的扣费收益与回撤\n\n当前选择：**${SCENARIO[scenario]} · ${VARIANT[variant]}**。曲线实际覆盖 **${dates[0]?.slice(0, 10) ?? "—"} 至 ${dates.at(-1)?.slice(0, 10) ?? "—"}（UTC）**；右端为日收盘边界，代表此前最后一分钟的权益。间隔刻度不改变实际覆盖范围。`} />
    </ReportSection>
    <EvidenceChart id="round2-return" title="组合累计净收益" queryId="equity_history" rows={pivot(selectedHistory, "net_return")} sourceRows={selectedHistory} spec={curveSpec(selectedHistory, "net_return")} height={400} />
    <EvidenceChart id="round2-drawdown" title="组合距历史峰值的回撤" queryId="equity_history" rows={pivot(selectedHistory, "drawdown")} sourceRows={selectedHistory} spec={curveSpec(selectedHistory, "drawdown")} height={320} />
    <SummaryTable id="round2-portfolio" title="新五组与基准：精确结果" rows={selected} />
    <ReportSection id="round2-old-comparison-heading" title="两轮对照" queryId="strategy_summary" queryIds={["strategy_summary", "previous_summary"]} sourceRowsByQuery={{ strategy_summary: selected, previous_summary: oldSelected }} showHeading={false}>
      <RichNarrative id="round2-old-comparison:body" value={`## 与上一轮五组放在一起比较\n\n下表统一为${SCENARIO[scenario]}、${VARIANT[variant]}，按净收益排序。第一轮结果直接读取已交付文件，保留原有规则与资金口径；这是同窗口候选对照，不是独立验证。`} />
    </ReportSection>
    <DataComponent id="round2-ten-table" title="两轮十组组合对照" queryId="strategy_summary" queryIds={["strategy_summary", "previous_summary"]} sourceRowsByQuery={{ strategy_summary: selected.filter((r) => STRATEGIES.includes(r.strategy)), previous_summary: oldSelected }} kind="table" displayRows={tenRows}>
      <DataTable rows={tenRows} rowKey="run_id" searchable={false} compactNumbers={false} caption="两轮十组组合对照" columns={[
        { field: "round", label: "轮次" }, { field: "label", label: "策略" }, pc("net_return", "扣费净收益"), pc("max_drawdown", "分钟最大回撤"), nc("closed_trades", "平仓笔数", 0), nc("fees", "费用 USDT"), pc("exposure", "平均仓位"), pc("net_liquidation_return", "扣期末退出成本后"),
      ]} />
    </DataComponent>
    <ReportSection id="round2-cost-heading" title="成本与退出对照" queryId="strategy_summary" sourceRows={comparisonSource} showHeading={false}>
      <RichNarrative id="round2-cost:body" value="## 成本加倍与保护止损会改变结果多少\n\n同时列出四种组合，不只挑选收益最高的版本。原规则没有统一价格止损；5%保护版沿用前轮的执行、跳空与重入规则。" />
    </ReportSection>
    <DataComponent id="round2-cost-table" title="四情景的收益与最大回撤" queryId="strategy_summary" sourceRows={comparisonSource} displayRows={comparisons} kind="table">
      <DataTable rows={comparisons} rowKey="strategy" searchable={false} compactNumbers={false} caption="新五组四情景比较" columns={[
        { field: "label", label: "策略" }, ...["base", "stress"].flatMap((cost) => ["original", "stop5"].flatMap((ver) => [pc(`${cost}_${ver}_return`, `${SCENARIO[cost]}·${VARIANT[ver]}收益`), pc(`${cost}_${ver}_drawdown`, `${SCENARIO[cost]}·${VARIANT[ver]}回撤`)])),
      ]} />
    </DataComponent>
    <EvidenceChart id="round2-version-curve" title="所选策略：原规则与5%保护" queryId="equity_history" sourceRows={focusHistory} rows={pivot(focusHistory, "net_return", (r) => VARIANT[r.variant])} spec={focusSpec} height={320} headerControls={<Dropdown label="比较策略" showLabel value={focus} choices={STRATEGIES} choiceLabels={labels} onChange={setFocus} />} />
    <RichNarrative id="round2-coins:heading" value={`## BTC与ETH的单币结果\n\n每币 ${num(config.sleeve_capital, 0)} USDT 独立记账；显示${SCENARIO[scenario]}、${VARIANT[variant]}，附单币买入持有。组合是两份权益相加，不相互借用现金或库存。`} />
    <SummaryTable id="round2-coins" title="新五组各币精确结果" rows={coinRows} />
    <ReportSection id="round2-month-heading" title="月度稳定性" queryId="monthly_returns" sourceRows={monthSource} showHeading={false}>
      <RichNarrative id="round2-month:body" value="## 净收益是否依赖少数月份\n\n月收益按月末权益变化计算，首月相对期初本金；首尾月仅覆盖实际区间。正收益月数是样本描述，不代表未来胜率。" />
    </ReportSection>
    <DataComponent id="round2-month-table" title="每月扣费净收益" queryId="monthly_returns" sourceRows={monthSource} displayRows={monthlyRows} kind="table">
      <DataTable rows={monthlyRows} searchable={false} caption="新五组月度收益" columns={[{ field: "month", label: "UTC月份" }, ...STRATEGIES.map((id) => pc(id, labels[id] ?? id))]} />
    </DataComponent>
    <DataComponent id="round2-stability-table" title="月度收益分布" queryId="monthly_returns" sourceRows={monthSource} displayRows={stability} kind="table">
      <DataTable rows={stability} searchable={false} caption="新五组月度分布" columns={[{ field: "label", label: "策略" }, nc("months", "覆盖月份", 0), nc("positive", "正收益月份", 0), pc("best", "最好月"), pc("worst", "最差月")]} />
    </DataComponent>
    <EvidenceChart id="round2-month-bars" title="所选策略每月净收益" queryId="monthly_returns" sourceRows={focusedMonths} rows={focusedMonths.map((r) => ({ ...r, "月收益(%)": valid(r.return) ? r.return * 100 : null }))} spec={{ type: "bar", x: "month", y: "月收益(%)", colors: { "月收益(%)": COLORS[focus] }, colorBySign: false, stackable: false, valueDecimals: 2 }} height={300} headerControls={<Dropdown label="月度策略" showLabel value={focus} choices={STRATEGIES} choiceLabels={labels} onChange={setFocus} />} />
    <ReportSection id="round2-rules" title="固定规则" queryId="strategy_summary" sourceRows={summary} showHeading={false}><RichNarrative id="round2-rules:body" value={rules} /></ReportSection>
    <ReportSection id="round2-methods" title="成本与结果口径" queryId="strategy_summary" sourceRows={summary} showHeading={false}>
      <RichNarrative id="round2-methods:body" value={methods} />
      {sourceLinks && <RichNarrative id="round2-sources:body" sourcePreviews={previews} value={`### 公开规则与原始参考\n\n${sourceLinks}\n\n来源、公式与明示改版见 sources/sources.json；固定参数、依赖、行情和源码哈希以及复现命令见 manifest.json。`} />}
    </ReportSection>
  </article>;
}

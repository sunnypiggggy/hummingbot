import React, { useMemo, useState } from "react";
import {
  DataComponent, DataTable, Dropdown, EvidenceChart, ReportSection, RichNarrative, useDataApp,
} from "../../data-app-public.jsx";
import "./spot-suite-report.css";

const STRATEGIES = ["donchian", "cluc", "rsi2", "macd_ema", "rotation"];
const ROOTS = Object.fromEntries(STRATEGIES.map((id, index) => [id, `var(--chart-${index + 1})`]));
ROOTS.buy_hold_equal = "var(--text)";
ROOTS.cash = "var(--secondary)";
const SCENARIO = { base: "主成本", stress: "压力成本" };
const VARIANT = { original: "原规则", stop5: "5% 保护", benchmark: "基准" };
const valid = (v) => typeof v === "number" && Number.isFinite(v);
const pct = (v, decimals = 2) => valid(v) ? `${(v * 100).toFixed(decimals)}%` : "—";
const num = (v, decimals = 2) => valid(v) ? v.toLocaleString("zh-CN", { minimumFractionDigits: decimals, maximumFractionDigits: decimals }) : "—";
const percentage = (field, label) => ({ field, label, renderCell: (value) => pct(value) });
const number = (field, label, decimals = 2) => ({ field, label, renderCell: (value) => num(value, decimals) });
const groupRows = (rows, scenario, variant) => rows.filter((r) => r.pair === "PORTFOLIO" && r.scenario === scenario && (r.variant === variant || r.variant === "benchmark"));

function pivot(rows, measure, labeler = (row) => row.label) {
  const records = new Map();
  rows.forEach((row) => {
    if (!records.has(row.timestamp)) records.set(row.timestamp, { timestamp: row.timestamp });
    records.get(row.timestamp)[`${labeler(row)}(%)`] = valid(row[measure]) ? row[measure] * 100 : null;
  });
  return [...records.values()].sort((a, b) => a.timestamp.localeCompare(b.timestamp));
}

function curveSpec(rows, measure, labeler) {
  const fields = [...new Set(rows.map((row) => `${labeler ? labeler(row) : row.label}(%)`))];
  const colors = Object.fromEntries(rows.map((row) => [`${labeler ? labeler(row) : row.label}(%)`, ROOTS[row.strategy] ?? "var(--secondary)"]));
  return {
    type: "line", x: "timestamp", y: fields[0] ?? "净收益(%)", fields, colors,
    stackable: false, valueDecimals: 2,
    yLabel: measure === "drawdown" ? "距历史峰值（%）" : "累计净收益（%）",
    legend: { labels: Object.fromEntries(fields.map((field) => [field, field.replace(/\(%\)$/u, "")])) },
  };
}

function SummaryTable({ id, title, rows, extra = false }) {
  const columns = [
    { field: "label", label: "策略" },
    { field: "pair", label: "资金分组", renderCell: (value) => value === "PORTFOLIO" ? "两币组合" : value.replace("USDT", "／USDT") },
    percentage("net_return", "净收益"), percentage("max_drawdown", "最大回撤"),
    number("sharpe_daily", "日收益 Sharpe"), number("calmar", "Calmar"),
    number("closed_trades", "平仓笔数", 0), percentage("win_rate", "胜率"),
    number("payoff_ratio", "平均盈亏比"), number("fees", "费用 USDT"),
    number("turnover", "周转倍数"), percentage("exposure", "平均仓位"),
    ...(extra ? [number("average_holding_hours", "平均持仓小时"), number("unrealized_pnl", "未实现盈亏 USDT"), percentage("net_liquidation_return", "计入末端卖出成本后")] : []),
  ];
  return <DataComponent id={id} queryId="strategy_summary" title={title} kind="table" sourceRows={rows} displayRows={rows}>
    <DataTable rows={rows} columns={columns} rowKey="run_id" searchable={false} compactNumbers={false} caption={title} />
  </DataComponent>;
}

export function ReportContent() {
  const { snapshot, reviewedPeriodRows, appTitle, setAppTitle, canEdit, mode, visible } = useDataApp();
  const summary = reviewedPeriodRows("strategy_summary", { period: "history" });
  const histories = reviewedPeriodRows("equity_history", { period: "history" });
  const months = reviewedPeriodRows("monthly_returns", { period: "history" });
  const labels = Object.fromEntries(summary.map((row) => [row.strategy, row.label]));
  const [scenario, setScenario] = useState("base");
  const [variant, setVariant] = useState("original");
  const [focus, setFocus] = useState("cluc");
  const config = snapshot.report?.suiteConfig ?? {};
  const main = groupRows(summary, "base", "original").filter((row) => STRATEGIES.includes(row.strategy));
  const ranked = [...main].filter((row) => valid(row.net_return)).sort((a, b) => b.net_return - a.net_return);
  const winner = ranked[0];
  const profitable = main.filter((row) => valid(row.net_return) && row.net_return > 0).length;
  const benchmark = summary.find((row) => row.strategy === "buy_hold_equal" && row.scenario === "base" && row.pair === "PORTFOLIO");
  const stressWinner = winner && summary.find((row) => row.strategy === winner.strategy && row.scenario === "stress" && row.variant === "original" && row.pair === "PORTFOLIO");
  const clucMain = main.find((row) => row.strategy === "cluc");
  const clucStress = summary.find((row) => row.strategy === "cluc" && row.scenario === "stress" && row.variant === "original" && row.pair === "PORTFOLIO");
  const rotationStress = summary.find((row) => row.strategy === "rotation" && row.scenario === "stress" && row.variant === "original" && row.pair === "PORTFOLIO");
  const opening = winner
    ? `## 结果摘要\n\n- 主成本、原规则下，五组中有 **${profitable} 组**取得正净收益；${winner.label}的收益最高，为 **${pct(winner.net_return)}**，分钟最大回撤为 **${pct(winner.max_drawdown)}**，实际已平仓 **${num(winner.closed_trades, 0)} 笔**。净收益仍包含期末未平仓持仓的市值变化，不能把这一排名视为充分的交易样本证据。\n- 同资金、同执行成本的 BTC／ETH 等权持有收益为 **${pct(benchmark?.net_return)}**。这一基准同样保留 5% 初始现金。\n- Cluc 主版仅有 **${num(clucMain?.closed_trades, 0)} 笔已平仓交易**，平均资金暴露仅 **${pct(clucMain?.exposure)}**（分钟仓位均值），属于交易很少、资金暴露很低的样本。\n- 压力成本下，Cluc 原规则净收益为 **${pct(clucStress?.net_return)}**，仍仅 **${num(clucStress?.closed_trades, 0)} 笔已平仓交易**；轮动原规则为 **${pct(rotationStress?.net_return)}**。这份年度样本尚未证明稳健优胜者，需结合样本量、费用敏感性与完整对照判断。`
    : "## 结果摘要\n\n当前证据未提供完整的五组组合结果。请保留这一缺口，暂不按收益排名。";
  const selectedSummary = groupRows(summary, scenario, variant);
  const selectedHistory = groupRows(histories, scenario, variant);
  const curveDates = [...new Set(selectedHistory.map((row) => row.timestamp))].sort();
  const curveStart = curveDates[0]?.slice(0, 10) ?? "—";
  const curveEnd = curveDates.at(-1)?.slice(0, 10) ?? "—";
  const coinSummary = summary.filter((row) => ["BTCUSDT", "ETHUSDT"].includes(row.pair) && row.scenario === scenario && (row.variant === variant || row.variant === "benchmark"));
  const comparisonSource = summary.filter((row) => row.pair === "PORTFOLIO" && STRATEGIES.includes(row.strategy));
  const comparisons = STRATEGIES.map((strategy) => {
    const record = { strategy, label: labels[strategy] ?? strategy };
    for (const cost of ["base", "stress"]) for (const ver of ["original", "stop5"]) {
      const row = comparisonSource.find((item) => item.strategy === strategy && item.scenario === cost && item.variant === ver);
      record[`${cost}_${ver}_return`] = row?.net_return ?? null;
      record[`${cost}_${ver}_drawdown`] = row?.max_drawdown ?? null;
    }
    return record;
  });
  const focusHistory = histories.filter((row) => row.pair === "PORTFOLIO" && row.scenario === scenario && row.strategy === focus && ["original", "stop5"].includes(row.variant));
  const focusSpec = curveSpec(focusHistory, "net_return", (row) => VARIANT[row.variant]);
  focusSpec.colors = { "原规则(%)": ROOTS[focus], "5% 保护(%)": "var(--text)" };
  const monthSource = groupRows(months, scenario, variant).filter((row) => STRATEGIES.includes(row.strategy));
  const monthlyTable = useMemo(() => {
    const rows = new Map();
    monthSource.forEach((row) => {
      if (!rows.has(row.month)) rows.set(row.month, { month: row.month });
      rows.get(row.month)[row.strategy] = row.return;
    });
    return [...rows.values()].sort((a, b) => a.month.localeCompare(b.month));
  }, [months, scenario, variant]);
  const stability = STRATEGIES.map((strategy) => {
    const rows = monthSource.filter((row) => row.strategy === strategy && valid(row.return));
    return { label: labels[strategy] ?? strategy, reviewed_months: rows.length,
      positive_months: rows.filter((row) => row.return > 0).length,
      best_month: rows.length ? Math.max(...rows.map((row) => row.return)) : null,
      worst_month: rows.length ? Math.min(...rows.map((row) => row.return)) : null };
  });
  const selectedMonthRows = monthSource.filter((row) => row.strategy === focus);
  const cost = (costId, kind) => config.costs?.[costId]?.[kind];
  const methods = `## 怎样理解这些结果\n\n**费用与执行。** 主成本为单边手续费 ${pct(cost("base", "fee"))}、不利滑点 ${pct(cost("base", "slippage"))}；压力成本为 ${pct(cost("stress", "fee"))}、${pct(cost("stress", "slippage"))}。这些是研究情景，未冒充账户历史实际费率。每组初始 ${num(config.capital, 0)} USDT；前四组 BTC／ETH 各占一半独立资金，轮动组共用资金。每次最多花费可用现金的 95%，包含入场费，不借币、不加仓、不使用 BNB 抵扣。\n\n**时点与价格。** 指标仅使用已闭合 K 线；信号在随后分钟开盘加不利滑点成交。换币先卖出，下一分钟才买入。Cluc 保留净 ROI 1% 与 5% 价格止损；其余原规则没有额外价格止损。5% 保护版单列；Cluc 两版相同。同根 K 线同时触发止损与 ROI 时优先止损，止损跳空按更不利开盘价成交；保护退出后等下一条严格更晚的策略边界信号。\n\n**收益与回撤。** 报告的曲线按 UTC 日末显示，最大回撤仍以完整 1 分钟权益计算，并包含期初资金。期末保留真实模拟持仓，净收益包含浮盈亏；表中的末端卖出成本后收益另扣一次假设清算的手续费与滑点。胜率、平均盈亏比与持仓时间只统计已平仓交易；无交易或分母不可用显示“—”。首尾月可能只有部分月份，不补齐。\n\n**来源与改版。** 本次是五组固定规则的独立现货复现：Donchian、RSI2 与 MACD＋EMA 使用 1h，Cluc 使用 5m，轮动使用日线；数据另有 ${num(config.warmup_days, 0)} 天预热，但不纳入收益。RSI2 去除空头；Cluc 的变量名 ema100 实际是 EMA50；MACD＋EMA 保留状态条件而非改成穿越；轮动增加所选币 SMA200 趋势门。各组不全是震荡策略：Cluc 为均值回归，RSI2 为趋势内回落，其余主要捕捉趋势或相对强弱。全年收益不能单独证明震荡时段有效。\n\n**证据范围。** 全部数值来自本地 reviewed 数据与模拟账本。没有历史盘口队列、逐笔流动性与完整历史费率重建；当前过滤器仅是带采集时间的快照。五组共享现货多头风险，收益不代表彼此独立；本报告没有变更实盘权限。`;
  const sourceLinks = (snapshot.report?.strategySources ?? []).map((source) => `- [${source.repository} · ${source.repo_path}](${source.source_url})（${source.license}；固定提交 ${source.commit.slice(0, 12)}）。`).join("\n");

  return <article className="report-content spot-suite-report" aria-label="五组现货策略研究报告">
    <header className="report-hero">
      <h1 data-data-app-title contentEditable={canEdit && mode === "edit"} suppressContentEditableWarning
        onBlur={canEdit && mode === "edit" ? (event) => setAppTitle(event.currentTarget.textContent.trim() || appTitle) : undefined}>{appTitle}</h1>
      <RichNarrative id="spot-suite:deck" className="report-deck" value={`BTCUSDT／ETHUSDT · ${String(config.start ?? "").slice(0, 10)} 至 ${String(config.end ?? "").slice(0, 10)}（右端不含，UTC）`} />
    </header>

    {visible("suite-summary") && <ReportSection id="suite-summary" title="结果摘要" queryId="strategy_summary" sourceRows={[...main, ...(benchmark ? [benchmark] : []), ...(stressWinner ? [stressWinner] : []), ...(clucStress ? [clucStress] : []), ...(rotationStress && rotationStress !== stressWinner ? [rotationStress] : [])]} showHeading={false}>
      <RichNarrative id="suite-summary:body" value={opening} />
    </ReportSection>}

    <div className="spot-suite-controls" aria-label="报告显示条件">
      <Dropdown label="成本情景" showLabel value={scenario} choices={["base", "stress"]} choiceLabels={SCENARIO} onChange={setScenario} />
      <Dropdown label="策略版本" showLabel value={variant} choices={["original", "stop5"]} choiceLabels={VARIANT} onChange={setVariant} />
    </div>

    <ReportSection id="suite-curve-period" title="曲线范围" queryId="equity_history" sourceRows={selectedHistory} showHeading={false}>
      <RichNarrative id="suite-curves:heading" value={`## 五组组合的收益与回撤\n\n当前选择：**${SCENARIO[scenario]} · ${VARIANT[variant]}**。每组初始资金一致；现金与等权买入持有以中性色显示。下方精确表对应同一选择。\n\n**曲线实际起止：${curveStart} 至 ${curveEnd}（UTC）**。横轴使用间隔日期刻度；最右端点仍为 ${curveEnd} 的日收盘边界，代表该边界之前最后一分钟的权益。`} />
    </ReportSection>
    <EvidenceChart id="suite-return" title="组合累计净收益" queryId="equity_history" rows={pivot(selectedHistory, "net_return")} sourceRows={selectedHistory} spec={curveSpec(selectedHistory, "net_return")} height={400} />
    <EvidenceChart id="suite-drawdown" title="组合距历史峰值的回撤" queryId="equity_history" rows={pivot(selectedHistory, "drawdown")} sourceRows={selectedHistory} spec={curveSpec(selectedHistory, "drawdown")} height={320} />
    <SummaryTable id="suite-portfolio-table" title="五组与基准：完整收益、风险和成本" rows={selectedSummary} extra />

    <ReportSection id="suite-comparison-heading" title="成本与保护退出" queryId="strategy_summary" sourceRows={comparisonSource} showHeading={false}>
      <RichNarrative id="suite-comparison:body" value="## 原规则、保护退出与成本压力\n\n下表同时保留四种组合，避免只选择看起来最好的版本。5% 保护按实际成交入场价计算；Cluc 原规则已经包含同一止损，因此其两版一致。" />
    </ReportSection>
    <DataComponent id="suite-scenario-table" queryId="strategy_summary" title="四种组合的净收益与分钟最大回撤" kind="table" displayRows={comparisons} sourceRows={comparisonSource}>
      <DataTable rows={comparisons} rowKey="strategy" searchable={false} compactNumbers={false} caption="五组策略四情景比较" columns={[
        { field: "label", label: "策略" },
        ...["base", "stress"].flatMap((costId) => ["original", "stop5"].flatMap((version) => [
          percentage(`${costId}_${version}_return`, `${SCENARIO[costId]}·${VARIANT[version]}收益`),
          percentage(`${costId}_${version}_drawdown`, `${SCENARIO[costId]}·${VARIANT[version]}回撤`),
        ])),
      ]} />
    </DataComponent>
    <EvidenceChart id="suite-version-curve" queryId="equity_history" title="所选策略：原规则与 5% 保护" rows={pivot(focusHistory, "net_return", (row) => VARIANT[row.variant])} sourceRows={focusHistory} spec={focusSpec} height={340}
      headerControls={<Dropdown label="比较策略" showLabel value={focus} choices={STRATEGIES} choiceLabels={labels} onChange={setFocus} />} />

    <RichNarrative id="suite-coins:heading" value={`## BTC 与 ETH 分开核对\n\n前四组各币使用 ${num(config.sleeve_capital, 0)} USDT，分别记账；轮动为共享资金，因此不拆成两个独立回测。这里同样显示${SCENARIO[scenario]}、${VARIANT[variant]}，并附单币买入持有基准。`} />
    <SummaryTable id="suite-coin-table" title="单币精确结果" rows={coinSummary} extra />

    <ReportSection id="suite-month-heading" queryId="monthly_returns" title="月度稳定性" sourceRows={monthSource} showHeading={false}>
      <RichNarrative id="suite-month:body" value="## 收益是否集中在少数月份\n\n月收益由月末权益相对前月末计算，首月以初始资金为基准。首尾月为实际覆盖的部分月份，不作全年化填补。正收益月数只描述这段样本内的分布，不代表未来胜率。" />
    </ReportSection>
    <DataComponent id="suite-monthly-table" queryId="monthly_returns" title="每月净收益" kind="table" displayRows={monthlyTable} sourceRows={monthSource}>
      <DataTable rows={monthlyTable} searchable={false} caption="五组月度收益精确表" columns={[{ field: "month", label: "UTC 月份" }, ...STRATEGIES.map((id) => percentage(id, labels[id] ?? id))]} />
    </DataComponent>
    <DataComponent id="suite-monthly-stability" queryId="monthly_returns" title="月度收益分布" kind="table" displayRows={stability} sourceRows={monthSource}>
      <DataTable rows={stability} searchable={false} caption="样本内正收益月份与最好最差月份" columns={[
        { field: "label", label: "策略" }, number("reviewed_months", "覆盖月份", 0), number("positive_months", "正收益月份", 0), percentage("best_month", "最好月"), percentage("worst_month", "最差月"),
      ]} />
    </DataComponent>
    <EvidenceChart id="suite-monthly-bars" queryId="monthly_returns" title="所选策略每月净收益" sourceRows={selectedMonthRows} rows={selectedMonthRows.map((row) => ({ ...row, "月收益(%)": valid(row.return) ? row.return * 100 : null }))}
      spec={{ type: "bar", x: "month", y: "月收益(%)", colors: { "月收益(%)": ROOTS[focus] }, colorBySign: false, stackable: false, valueDecimals: 2 }} height={300}
      headerControls={<Dropdown label="月度策略" showLabel value={focus} choices={STRATEGIES} choiceLabels={labels} onChange={setFocus} />} />

    <ReportSection id="suite-methods" title="规则、成本与期末持仓" queryId="strategy_summary" sourceRows={summary} showHeading={false}>
      <RichNarrative id="suite-methods:body" value={methods} />
      {sourceLinks && <RichNarrative id="suite-sources:body" value={`### 固定版本的开源参考\n\n${sourceLinks}\n\n完整源文件、许可与 SHA256 保存在随报告交付的 sources/ 及 manifest.json；链接用于核对实际规则，不是收益背书。`} />}
    </ReportSection>
  </article>;
}

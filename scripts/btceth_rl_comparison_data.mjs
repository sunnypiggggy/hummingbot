// Presentation-only joins of reviewed accounts. No backtest or model calculations.
export const SUMMARY_QUERIES = ["strategy_summary", "execution_band_summary", "td3_summary", "droq_summary"];
export const HISTORY_QUERIES = ["equity_history", "execution_band_equity", "td3_equity", "droq_equity"];
export const MODEL_NAMES = { sac: "SAC", tqc: "TQC · 原预算", tqc_cap075: "TQC · 75%预算", tqc_band05: "TQC · 75%预算 / 5pp门槛", crossq: "CrossQ", recurrent_ppo: "RecurrentPPO", td3: "TD3", droq: "DroQ · 75%预算" };
export const BASELINE_NAMES = { cash: "USDT现金", btc_hold: "BTC持有", eth_hold: "ETH持有", equal_hold: "BTC/ETH等权持有", equal_rebalance: "BTC/ETH等权再平衡", sma200_trend: "SMA200趋势", momentum180: "180日动量" };
export const finite = value => typeof value === "number" && Number.isFinite(value);
export const accountKey = runId => runId.replace("__base__", "__COST__").replace("__stress__", "__COST__");

export function comparisonAccounts(queries) {
  const pairs = new Map();
  SUMMARY_QUERIES.forEach((queryId, index) => {
    for (const row of queries[queryId] ?? []) {
      if (row.split !== "test" || !["base", "stress"].includes(row.scenario)) continue;
      let variant;
      if (index === 0) {
        if (row.algorithm !== "baseline" && row.selected_profile !== true) continue;
        variant = row.algorithm === "baseline" ? row.strategy : row.algorithm;
      } else if (index === 1 && row.followup_role === "followup") variant = "tqc_band05";
      else if (index === 2 && row.td3_role === "td3_selected") variant = "td3";
      else if (index === 3 && row.droq_role === "droq_selected") variant = "droq";
      else continue; // Reused baseline/cap references keep their original query identity.
      const kind = row.algorithm === "baseline" ? "baseline" : row.primary_policy === true ? "ensemble" : "seed";
      if (kind === "seed" && !finite(row.seed)) throw Error("Single model needs an observed seed");
      const key = accountKey(row.run_id);
      if (!pairs.has(key)) pairs.set(key, { key, variant, kind, seed: row.seed, queryId, historyQuery: HISTORY_QUERIES[index], name: MODEL_NAMES[variant] ?? BASELINE_NAMES[variant] ?? variant });
      const pair = pairs.get(key);
      if (pair[row.scenario]) throw Error("Duplicate reviewed account: " + row.run_id);
      pair[row.scenario] = row;
    }
  });
  const accounts = [...pairs.values()];
  for (const pair of accounts) {
    if (!pair.base || !pair.stress) throw Error("Both cost accounts are required: " + pair.key);
    if (pair.base.initial_capital !== 10000 || pair.stress.initial_capital !== 10000) throw Error("Capital differs");
    const siblings = accounts.filter(p => p.kind === "seed" && p.variant === pair.variant);
    pair.positiveSeeds = siblings.filter(p => p.base.net_return > 0).length;
    pair.seedCount = siblings.length;
    if (pair.kind !== "baseline" && pair.base.fills === 0 && pair.base.average_exposure === 0) pair.name += " · 持现金";
    pair.label = pair.name + (pair.kind === "seed" ? ` · seed ${pair.seed}` : pair.kind === "ensemble" ? " · 五模型组合" : "");
  }
  return accounts;
}

export function sourceScope(accounts, queries, {includeSeedEvidence = false} = {}) {
  const byQuery = new Map();
  const include = (queryId, runId) => { if (!byQuery.has(queryId)) byQuery.set(queryId, new Set()); byQuery.get(queryId).add(runId); };
  accounts.forEach(pair => { include(pair.queryId, pair.base.run_id); include(pair.queryId, pair.stress.run_id); });
  if (includeSeedEvidence) {
    const variants = new Set(accounts.filter(pair => pair.kind !== "baseline").map(pair => pair.variant));
    comparisonAccounts(queries).filter(pair => pair.kind === "seed" && variants.has(pair.variant))
      .forEach(pair => include(pair.queryId, pair.base.run_id));
  }
  return Object.fromEntries(SUMMARY_QUERIES.filter(id => byQuery.has(id)).map(id => [id, (queries[id] ?? []).filter(row => byQuery.get(id).has(row.run_id))]));
}

export function comparisonDisplay(accounts) {
  return accounts.map(pair => ({ key: pair.key, name: pair.name, seed: pair.kind === "seed" ? pair.seed : null, label: pair.label, kind: pair.kind,
    base_return: pair.base.net_return, base_dd: Math.abs(pair.base.max_drawdown), stress_return: pair.stress.net_return, stress_dd: Math.abs(pair.stress.max_drawdown),
    positive_seeds: pair.kind === "baseline" ? null : `${pair.positiveSeeds}/${pair.seedCount}`, base_fees: pair.base.fees, stress_fees: pair.stress.fees,
    base_liquidation: pair.base.net_liquidation_return, stress_liquidation: pair.stress.net_liquidation_return, average_exposure: pair.base.average_exposure,
    profile: pair.base.profile, run_id: pair.base.run_id }));
}

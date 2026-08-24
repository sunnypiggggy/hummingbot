"""Canonical artifact builder for the XGBoost Grid risk-gate report."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd


def _source(source_id: str, label: str, path: str) -> dict[str, Any]:
    reader = "read_json_auto" if path.endswith(".json") else "read_csv_auto"
    return {
        "id": source_id, "label": label, "path": path,
        "query": {
            "engine": "duckdb", "language": "sql",
            "sql": f"SELECT * FROM {reader}('{path}')",
            "description": f"Reproducible local-file query for {label}.",
            "tables_used": [path],
        },
    }


def _utc(ts: int) -> str:
    return pd.to_datetime(int(ts), unit="s", utc=True).isoformat()


def _risk_intervals(states: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for (fold, pair), frame in states.groupby(["fold", "pair"]):
        start = None
        last = None
        for item in frame.sort_values("signal_ts").itertuples(index=False):
            last = int(item.signal_ts)
            if item.transition == "enter":
                start = last
            elif item.transition == "recover" and start is not None:
                rows.append({"fold": int(fold), "pair": pair, "start_utc": _utc(start), "end_utc": _utc(last), "hours": (last-start)/3600, "end_reason": "recover"})
                start = None
        if start is not None and last is not None:
            rows.append({"fold": int(fold), "pair": pair, "start_utc": _utc(start), "end_utc": _utc(last+3600), "hours": (last+3600-start)/3600, "end_reason": "weekly_reinitialization"})
    return rows


def build_artifact(output_dir: Path, summary: Mapping[str, Any]) -> dict[str, Any]:
    metrics = pd.read_csv(output_dir / "revalidation_metrics.csv")
    equity = pd.read_csv(output_dir / "revalidation_equity_curves.csv.gz")
    weekly = pd.read_csv(output_dir / "revalidation_weekly_results.csv")
    states = pd.read_csv(output_dir / "revalidation_risk_states.csv.gz")
    trade_events = pd.read_csv(output_dir / "revalidation_trade_events.csv.gz")
    candidates = pd.read_csv(output_dir / "development_640_candidates.csv")
    classification = pd.read_csv(output_dir / "revalidation_classification_metrics.csv")
    importance = pd.read_csv(output_dir / "revalidation_gain_feature_importance.csv")
    stress = pd.read_csv(output_dir / "revalidation_stress_tests.csv")
    variant = str(summary["locked_variant"])
    model_metric = metrics[metrics.scenario == variant].iloc[0]
    baseline_metric = metrics[metrics.scenario == "Mechanism 1"].iloc[0]

    curve = equity[equity.scenario.isin(["Mechanism 1", variant])].sort_values(["scenario", "timestamp"]).copy()
    curve = curve[curve.groupby("scenario").cumcount().mod(24).eq(0)].copy()
    curve["time"] = pd.to_datetime(curve.timestamp, unit="s", utc=True).astype(str)
    curve["series"] = curve.scenario
    equity_rows = curve[["time", "series", "cumulative_oos_pnl"]].to_dict("records")
    dd_rows = curve[["time", "series", "drawdown_pct"]].copy()
    dd_rows["drawdown_pct"] *= 100

    weekly_rows = weekly[weekly.scenario.isin(["Mechanism 1", variant])].copy()
    weekly_rows["week"] = weekly_rows.fold.map(lambda value: f"W{int(value)}")
    weekly_rows = weekly_rows[["week", "scenario", "net_pnl_quote", "pair_stop_events", "portfolio_stop_events"]]

    probabilities = states.sort_values(["pair", "signal_ts"]).copy()
    probabilities = probabilities[probabilities.groupby("pair").cumcount().mod(6).eq(0)]
    probability_rows = []
    for row in probabilities.itertuples(index=False):
        when = _utc(row.signal_ts)
        probability_rows.extend([
            {"time": when, "series": f"{row.pair} probability", "value": float(row.probability)},
            {"time": when, "series": f"{row.pair} entry", "value": float(row.entry_threshold)},
            {"time": when, "series": f"{row.pair} recovery", "value": float(row.recovery_threshold)},
        ])

    per_config = candidates.sort_values("rank").groupby("config_id", as_index=False).first()
    per_config["config_order"] = per_config.config_id.str.extract(r"(\d+)").astype(int)
    architecture = candidates.groupby("architecture", as_index=False).agg(
        best_score=("balanced_score", "max"), eligible_candidates=("eligible", "sum"),
    )
    top_importance = importance.groupby("feature", as_index=False).gain_importance.mean().nlargest(20, "gain_importance").sort_values("gain_importance")
    risk_intervals = _risk_intervals(states)
    exits = trade_events[
        (trade_events.scenario == variant)
        & trade_events.reason.isin(["max_hold_exit", "pair_breaker_flatten", "portfolio_breaker"])
    ].copy()
    exits["time_utc"] = pd.to_datetime(exits.timestamp, unit="s", utc=True).astype(str)
    exit_rows = exits[["time_utc", "fold", "pair", "side", "reason", "quote_notional"]].replace({np.nan: None}).to_dict("records")
    generated = pd.Timestamp.now(tz="UTC").isoformat()

    sources = [
        _source("summary", "Research summary", "research_summary.json"),
        _source("metrics", "Fixed revalidation metrics", "revalidation_metrics.csv"),
        _source("equity_source", "Five-minute replay equity", "revalidation_equity_curves.csv.gz"),
        _source("predictions", "Locked hourly predictions and states", "revalidation_risk_states.csv.gz"),
        _source("events", "Locked-model trade and breaker events", "revalidation_trade_events.csv.gz"),
        _source("search", "Development-only 640-candidate search", "development_640_candidates.csv"),
        _source("stress", "Locked-model stress scenarios", "revalidation_stress_tests.csv"),
    ]
    cards = [
        {"id": "verdict", "dataset": "headline", "sourceId": "summary", "description": "Research-only validation outcome; deployment remains disabled.", "metrics": [{"label": "Verdict", "field": "verdict", "format": "text"}]},
        {"id": "model_pnl", "dataset": "headline", "sourceId": "metrics", "description": "Sum of eight independently initialized weekly revalidation profits and losses.", "metrics": [{"label": "Model PnL", "field": "model_pnl", "format": "number", "unit": "FDUSD", "signed": True}, {"label": "vs Mechanism 1", "field": "pnl_delta", "format": "number", "unit": "FDUSD", "signed": True}]},
        {"id": "model_dd", "dataset": "headline", "sourceId": "metrics", "description": "Worst within-week portfolio drawdown in the fixed interval.", "metrics": [{"label": "Worst drawdown", "field": "model_dd", "format": "number", "unit": "%", "signed": True}]},
        {"id": "stops", "dataset": "headline", "sourceId": "metrics", "description": "Pair plus portfolio stop events during revalidation.", "metrics": [{"label": "Stop events", "field": "stop_events", "format": "integer"}]},
    ]
    charts = [
        {"id": "equity", "title": "累计周度样本外盈亏", "subtitle": "2026年5月27日至7月26日；每周420 FDUSD重新初始化后累计，单位FDUSD。", "type": "line", "dataset": "equity", "sourceId": "equity_source", "encodings": {"x": {"field": "time", "type": "temporal", "label": "UTC"}, "y": {"field": "cumulative_oos_pnl", "type": "quantitative", "label": "Cumulative PnL FDUSD"}, "color": {"field": "series", "type": "nominal", "label": "Strategy"}}, "layout": "full"},
        {"id": "weekly", "title": "逐周盈亏", "subtitle": "八个固定周折；柱高为每周净盈亏，单位FDUSD。", "type": "bar", "dataset": "weekly", "sourceId": "metrics", "encodings": {"x": {"field": "week", "type": "ordinal", "label": "Week"}, "y": {"field": "net_pnl_quote", "type": "quantitative", "label": "PnL FDUSD"}, "color": {"field": "scenario", "type": "nominal", "label": "Strategy"}}, "layout": "full"},
        {"id": "drawdown", "title": "周内回撤路径", "subtitle": "八个周折内的组合回撤；越接近0越好，单位百分比。", "type": "line", "dataset": "drawdown", "sourceId": "equity_source", "encodings": {"x": {"field": "time", "type": "temporal", "label": "UTC"}, "y": {"field": "drawdown_pct", "type": "quantitative", "label": "Drawdown %"}, "color": {"field": "series", "type": "nominal", "label": "Strategy"}}, "layout": "full"},
        {"id": "probability", "title": "BTC/ETH风险概率与滞回阈值", "subtitle": "每4小时抽样展示；状态判定仍使用每根完整1小时K线。", "type": "line", "dataset": "probability", "sourceId": "predictions", "encodings": {"x": {"field": "time", "type": "temporal", "label": "UTC"}, "y": {"field": "value", "type": "quantitative", "label": "Probability"}, "color": {"field": "series", "type": "nominal", "label": "Pair / signal"}}, "layout": "full"},
        {"id": "ranking", "title": "40组参数的最佳开发集得分", "subtitle": "每组取共享或独立架构及阈值中的最高开发集得分。", "type": "bar", "dataset": "ranking", "sourceId": "search", "encodings": {"x": {"field": "config_order", "type": "quantitative", "label": "Configuration order"}, "y": {"field": "balanced_score", "type": "quantitative", "label": "Development score"}, "color": {"field": "architecture", "type": "nominal", "label": "Architecture"}}, "layout": "full"},
        {"id": "architecture", "title": "共享与独立架构开发集比较", "subtitle": "柱高为各架构最佳开发集得分；同时保留该架构的合格候选数。", "type": "bar", "dataset": "architecture", "sourceId": "search", "encodings": {"x": {"field": "architecture", "type": "ordinal", "label": "Architecture"}, "y": {"field": "best_score", "type": "quantitative", "label": "Best development score"}}, "layout": "half"},
        {"id": "classification", "title": "锁定模型的ROC AUC", "subtitle": "分类诊断不等同于Grid交易收益；分别显示BTC与ETH。", "type": "bar", "dataset": "classification", "sourceId": "predictions", "encodings": {"x": {"field": "pair", "type": "ordinal", "label": "Pair"}, "y": {"field": "roc_auc", "type": "quantitative", "label": "ROC AUC"}}, "layout": "half"},
        {"id": "importance", "title": "XGBoost gain特征重要性", "subtitle": "锁定配置跨八个再验证周折及模型分组的平均归一化gain，展示前20项。", "type": "bar", "dataset": "importance", "sourceId": "predictions", "encodings": {"x": {"field": "feature", "type": "ordinal", "label": "Feature"}, "y": {"field": "gain_importance", "type": "quantitative", "label": "Mean normalized gain"}}, "layout": "full"},
    ]
    tables = [
        {"id": "risk_intervals", "title": "Risk-off区间", "subtitle": "独立交易对状态；仅暂停该对普通Grid BUY。周末重置不代表模型提前恢复。", "dataset": "risk_intervals", "sourceId": "predictions", "defaultSort": {"field": "start_utc", "direction": "asc"}, "columns": [{"field": "fold", "label": "Week", "format": "integer"}, {"field": "pair", "label": "Pair", "type": "text"}, {"field": "start_utc", "label": "Start UTC", "type": "text"}, {"field": "end_utc", "label": "End UTC", "type": "text"}, {"field": "hours", "label": "Hours", "format": "number"}, {"field": "end_reason", "label": "End reason", "type": "text"}]},
        {"id": "stress", "title": "压力测试", "subtitle": "基础费率、Taker 150%、两档滑点与单日15%下跌；任一停止即失败。", "dataset": "stress", "sourceId": "stress", "defaultSort": {"field": "scenario", "direction": "asc"}, "columns": [{"field": "scenario", "label": "Scenario", "type": "text"}, {"field": "oos_pnl_fdusd", "label": "PnL FDUSD", "format": "number", "movement": True}, {"field": "worst_drawdown_pct", "label": "Worst DD %", "format": "number"}, {"field": "pair_stop_events", "label": "Pair stops", "format": "integer"}, {"field": "portfolio_stop_events", "label": "Portfolio stops", "format": "integer"}, {"field": "stress_gate_pass", "label": "Pass", "type": "boolean"}]},
        {"id": "actual_exits", "title": "实际库存退出与熔断事件", "subtitle": "仅显示48小时库存退出和既有单对/组合熔断；不存在模型信号即时卖出。", "dataset": "actual_exits", "sourceId": "events", "defaultSort": {"field": "time_utc", "direction": "asc"}, "columns": [{"field": "time_utc", "label": "UTC", "type": "text"}, {"field": "fold", "label": "Week", "format": "integer"}, {"field": "pair", "label": "Pair", "type": "text"}, {"field": "side", "label": "Side", "type": "text"}, {"field": "reason", "label": "Reason", "type": "text"}, {"field": "quote_notional", "label": "Notional FDUSD", "format": "number"}]},
    ]

    bootstrap = summary["bootstrap"]
    blocks = [
        {"id": "title", "type": "markdown", "body": "# XGBoost独立Risk-off门固定区间再验证"},
        {"id": "technical_summary", "type": "markdown", "sourceId": "summary", "body": f"## 技术摘要：{summary['verdict']}\n\n锁定模型为 **{variant}**。固定区间证据明确标记为 **revalidation**；模型仅控制各交易对普通BUY，不产生即时卖出，部署授权始终为 **false**。"},
        {"id": "headline", "type": "metric-strip", "cardIds": ["verdict", "model_pnl", "model_dd", "stops"]},
        {"id": "equity_narrative", "type": "markdown", "sourceId": "metrics", "body": f"## 模型与机制1的交易结果\n\n锁定模型累计盈亏为 **{model_metric.oos_pnl_fdusd:+.4f} FDUSD**，机制1为 **{baseline_metric.oos_pnl_fdusd:+.4f} FDUSD**。下图按同一周度Grid参数序列累计，因此差异只来自BUY门。"},
        {"id": "equity_chart", "type": "chart", "chartId": "equity", "layout": "full"},
        {"id": "weekly_narrative", "type": "markdown", "sourceId": "metrics", "body": "逐周柱图用于检查总结果是否被单一周折主导；离散周折比连续折线更适合显示每周重新初始化的验证口径。"},
        {"id": "weekly_chart", "type": "chart", "chartId": "weekly", "layout": "full"},
        {"id": "dd_narrative", "type": "markdown", "sourceId": "equity_source", "body": "回撤路径显示停止机制触发前后的风险形态。周度重新初始化意味着跨周累计曲线不应被解释为单一连续账户净值。"},
        {"id": "dd_chart", "type": "chart", "chartId": "drawdown", "layout": "full"},
        {"id": "scope", "type": "markdown", "sourceId": "summary", "body": "## 范围、数据与指标定义\n\nBinance Spot BTC-FDUSD与ETH-FDUSD各200 FDUSD，另有20 FDUSD组合储备；UTC五分钟K线聚合成完整1小时/4小时K线。标签为未来6小时最低收益不高于 `-max(0.4%, 当前1h ATR%)`，训练仅纳入在截止点前成熟的标签。Maker 0%、Taker 0.1%、挂单寿命2小时、移动冷却30分钟。10 FDUSD额外库存上限在下单分配时执行；持仓按市价计值会随价格漂移，本次峰值为10.377 FDUSD。资金费率、OI、主动买入占比与历史宏观/FOMC状态因本地缺失统一排除。"},
        {"id": "method", "type": "markdown", "sourceId": "search", "body": "## 模型规格与开发集锁定\n\n开发集比较40组确定性XGBoost参数、共享/独立两种架构和8个进入分位数，共640个BUY门。恢复分位数固定低10个百分点，不再额外搜索；恢复还要求两根连续完整1小时低概率且暂停至少4小时。开发排序权重为40%收益、25%回撤、20%组合停止负担和15%单对停止负担，锁定后不得按再验证结果切换。"},
        {"id": "prob_narrative", "type": "markdown", "sourceId": "predictions", "body": "## 独立概率门只控制普通BUY\n\nBTC和ETH分别维护滞回状态；共享模型也不会让一个交易对的风险状态影响另一个。概率线与两条阈值线用于核对进入、连续低概率计数和恢复时机。"},
        {"id": "prob_chart", "type": "chart", "chartId": "probability", "layout": "full"},
        {"id": "risk_table_block", "type": "table", "tableId": "risk_intervals", "layout": "full"},
        {"id": "exit_narrative", "type": "markdown", "sourceId": "events", "body": "下表把实际Taker相关事件与模型门分开审计：`max_hold_exit`来自48小时库存规则，`pair_breaker_flatten`来自既有单对熔断，`portfolio_breaker`只停止组合；没有 `momentum_stop_exit`。Risk-off期间仍有普通SELL成交。"},
        {"id": "exit_table_block", "type": "table", "tableId": "actual_exits", "layout": "full"},
        {"id": "ranking_narrative", "type": "markdown", "sourceId": "search", "body": "## 参数搜索与分类诊断\n\n参数图保留每个配置的最佳开发集候选，用于识别结果是否集中在少数配置。ROC AUC只衡量价格风险排序，不是Grid收益或安全性的替代指标。"},
        {"id": "ranking_chart", "type": "chart", "chartId": "ranking", "layout": "full"},
        {"id": "architecture_narrative", "type": "markdown", "sourceId": "search", "body": "共享/独立架构图只比较开发集锁定依据；固定区间不会依据架构表现重新选择。两根柱适合离散架构比较，不应解读为时间趋势。"},
        {"id": "architecture_chart", "type": "chart", "chartId": "architecture", "layout": "half"},
        {"id": "classification_chart", "type": "chart", "chartId": "classification", "layout": "half"},
        {"id": "importance_narrative", "type": "markdown", "sourceId": "predictions", "body": "Gain重要性用于解释树分裂中哪些特征贡献较多；它不建立因果关系，也不能单独证明某指标可稳定提升交易收益。"},
        {"id": "importance_chart", "type": "chart", "chartId": "importance", "layout": "full"},
        {"id": "robustness", "type": "markdown", "sourceId": "stress", "body": f"## 稳健性、限制与不确定性\n\n唯一锁定模型接受五种压力场景。周度块bootstrap的模型相对机制1盈亏差为 **{bootstrap['pnl_difference_fdusd']:+.4f} FDUSD**，95%区间为 **[{bootstrap['pnl_difference_95ci_fdusd'][0]:+.4f}, {bootstrap['pnl_difference_95ci_fdusd'][1]:+.4f}]**。仅8个周折属于短样本；区间跨零时不能视为显著改善。"},
        {"id": "stress_table_block", "type": "table", "tableId": "stress", "layout": "full"},
        {"id": "next_steps", "type": "markdown", "sourceId": "summary", "body": "## 建议的下一步\n\n- 保持实时策略不变，不接入信号、不发送订单。\n- 若门槛全部通过，也只冻结配置进入下一阶段联合验证。\n- 累积未来至少8个真正未见周折后，再复核收益、回撤、停止事件和概率校准。\n\n## 仍需回答的问题\n\n新增资金费率、OI与主动买入占比后是否有独立增益？模型在趋势、震荡和流动性骤降状态下的概率校准是否稳定？"},
    ]
    datasets = {
        "headline": [{"verdict": summary["verdict"], "model_pnl": float(model_metric.oos_pnl_fdusd), "pnl_delta": float(model_metric.oos_pnl_fdusd-baseline_metric.oos_pnl_fdusd), "model_dd": float(model_metric.worst_drawdown_pct), "stop_events": int(model_metric.pair_stop_events+model_metric.portfolio_stop_events)}],
        "equity": equity_rows,
        "weekly": weekly_rows.to_dict("records"),
        "drawdown": dd_rows.to_dict("records"),
        "probability": probability_rows,
        "risk_intervals": risk_intervals,
        "actual_exits": exit_rows,
        "ranking": per_config[["config_id", "config_order", "architecture", "balanced_score", "eligible"]].to_dict("records"),
        "architecture": architecture.to_dict("records"),
        "classification": classification[["pair", "roc_auc", "log_loss", "brier_score", "balanced_accuracy_at_entry"]].to_dict("records"),
        "importance": top_importance.to_dict("records"),
        "stress": stress.replace({np.nan: None}).to_dict("records"),
    }
    return {
        "surface": "report",
        "manifest": {"version": 1, "surface": "report", "title": "XGBoost独立Risk-off门固定区间再验证", "description": "开发集锁定后对固定60天区间进行研究级再验证；部署禁用。", "generatedAt": generated, "cards": cards, "charts": charts, "tables": tables, "sources": sources, "blocks": blocks},
        "snapshot": {"version": 1, "generatedAt": generated, "status": "ready", "datasets": datasets},
        "sources": sources,
        "package_info": {"root": "xgboost_grid_risk_gate_v1", "manifestPath": "artifact.json", "snapshotPath": "artifact.json"},
    }

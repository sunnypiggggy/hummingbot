#!/usr/bin/env python3
"""Diagnose sparse crash-only ROC/SQZMOM gates and retest the live DCA."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from backtest_dca_momentum_guard import (
    PAIRS,
    build_signal_bars,
    combine_scenario,
    floor_five_minutes,
    load_window,
    run_pair_guarded,
)


EVENTS = (
    ("February crash", pd.Timestamp("2026-02-03T12:00:00Z"), pd.Timestamp("2026-02-06T12:00:00Z")),
    ("June crash", pd.Timestamp("2026-06-01T08:00:00Z"), pd.Timestamp("2026-06-03T12:00:00Z")),
)


def stateful_gate(signals: pd.DataFrame, trigger: pd.Series, recovery: pd.Series) -> tuple[pd.Series, list[tuple[pd.Timestamp, pd.Timestamp]]]:
    enabled = True
    states: list[bool] = []
    periods: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    off_at = None
    for ts, should_stop, should_resume in zip(signals.effective_time, trigger.fillna(False), recovery.fillna(False)):
        if enabled and should_stop:
            enabled, off_at = False, ts
        elif not enabled and should_resume:
            periods.append((off_at, ts))
            enabled, off_at = True, None
        states.append(enabled)
    if not enabled:
        periods.append((off_at, signals.effective_time.iloc[-1] + pd.Timedelta(hours=4)))
    return pd.Series(states, index=signals.index, dtype=bool), periods


def summarize_candidate(name: str, periods: list[tuple[pd.Timestamp, pd.Timestamp]], total_hours: float) -> dict:
    hits = []
    for event_name, start, end in EVENTS:
        hits.append(any(left < end and right > start for left, right in periods))
    false_episodes = sum(not any(left < end and right > start for _, start, end in EVENTS)
                         for left, right in periods)
    off_hours = sum((right - left).total_seconds() / 3600 for left, right in periods)
    return {
        "candidate": name,
        "february_hit": hits[0],
        "june_hit": hits[1],
        "risk_off_episodes": len(periods),
        "other_episodes": false_episodes,
        "risk_off_hours": off_hours,
        "risk_off_pct": off_hours / total_hours * 100,
    }


def map_gate(frame: pd.DataFrame, signals: pd.DataFrame, state: pd.Series) -> pd.Series:
    schedule = pd.Series(state.to_numpy(), index=signals.effective_time)
    target = pd.to_datetime(frame.timestamp, unit="s", utc=True)
    return pd.Series(schedule.reindex(target, method="ffill").fillna(True).to_numpy(), index=frame.index, dtype=bool)


def run_strategy(frames, signals, crash_state, fee_rate, slippage_bps, name, guarded_sides):
    pair_summaries, curves = [], {}
    for pair, frame in frames.items():
        gate = pd.Series(True, index=frame.index) if name in {"Baseline", "无门控"} else map_gate(frame, signals, crash_state)
        summary, _, curve = run_pair_guarded(
            frame, gate, pair, fee_rate, slippage_bps, guarded_sides=guarded_sides
        )
        pair_summaries.append(summary)
        curves[pair] = curve
    combined, _ = combine_scenario(curves, pair_summaries, name)
    return {
        "scenario": name,
        "net_pnl_quote": float(combined["combined_net_pnl_quote"]),
        "return_pct": float(combined["return_on_380_pct"]),
        "max_drawdown_pct": float(combined["combined_max_drawdown_pct"]),
        "fees_quote": float(combined["fees_quote"]),
        "positioned_executors": int(combined["positioned_executors"]),
        "risk_flatten_positions": int(combined["risk_flatten_positions"]),
        "enabled_pct": float(combined["enabled_pct"]),
    }


def build_artifact(output: Path, generated: str, threshold_rows: list[dict], strategy_rows: list[dict], event_rows: list[dict]) -> None:
    relative_root = output.resolve().relative_to(Path.cwd().resolve()).as_posix()
    def csv_source(source_id: str, label: str, filename: str) -> dict:
        return {
            "id": source_id,
            "label": label,
            "query": {
                "engine": "DuckDB",
                "sql": f"SELECT * FROM read_csv_auto('{relative_root}/{filename}');",
                "description": "Read the reviewed CSV produced by the Python threshold and DCA replay.",
                "tables_used": [filename],
            },
        }
    strategy_source = csv_source("strategy_source", "DCA scenario replay", "strategy_summary.csv")
    threshold_source = csv_source("threshold_source", "Crash threshold sweep", "threshold_summary.csv")
    event_source = csv_source("event_source", "Recommended gate event timeline", "event_summary.csv")
    sources = [strategy_source, threshold_source, event_source]
    artifact = {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": "BTC 快跌风控阈值诊断",
            "description": "只针对2月与6月快速下跌的ROC/SQZMOM门控诊断",
            "generatedAt": generated,
            "sources": sources,
            "blocks": [
                {"id": "title", "type": "markdown", "body": "# BTC 快跌风控阈值诊断"},
                {"id": "summary", "type": "markdown", "sourceId": "threshold_source",
                 "body": "## 技术结论\n\n推荐的稀疏触发器是 **48小时 ROC ≤ -8%、标准化 SQZMOM ≤ -3%，且 SQZMOM 为红色下降柱**。恢复条件为 SQZMOM 重新大于零（lime 或 green）。180天内它只产生两段暂停，覆盖指定的2月和6月下跌；但对当前双向 DCA 全部清仓后，收益与回撤没有改善，因此更适合做风险敞口开关，而不是收益增强器。"},
                {"id": "finding", "type": "markdown", "sourceId": "strategy_source",
                 "body": "## 全停没有改善当前双向 DCA\n\n当前 DCA 同时运行 BUY 与 SELL 执行器。下跌时把两边一起清掉，也会关闭可能受益于下行波动的 SELL 侧。下图比较无门控、两侧全停和仅停止 BUY 侧；每项均使用相同手续费与2bp风控清仓滑点。"},
                {"id": "chart", "type": "chart", "chartId": "strategy_return"},
                {"id": "scope", "type": "markdown", "body": "## 范围与指标定义\n\n分析窗口为2026年1月28日至7月27日，BTC 4小时指标控制 BTC/ETH 两个 DCA。SQZMOM 使用价格百分比标准化，避免固定点数随币价变化失真；信号在4小时收盘确认，下一根5分钟执行。"},
                {"id": "method", "type": "markdown", "sourceId": "threshold_source",
                 "body": "## 阈值筛选结果\n\n候选阈值必须覆盖两段指定事件，并比较其他暂停段数和总暂停时间。阈值越宽松，6月触发越早，但误停明显增加。"},
                {"id": "threshold_table", "type": "table", "tableId": "thresholds"},
                {"id": "events", "type": "markdown", "sourceId": "event_source",
                 "body": "## 两次事件的触发与恢复\n\n2月门控在目标日期前已经进入 Risk-Off；6月门控在快速下跌途中触发，并一直保持到 SQZMOM 回到零轴上方。"},
                {"id": "event_table", "type": "table", "tableId": "events"},
                {"id": "limits", "type": "markdown",
                 "body": "## 局限与稳健性\n\n阈值是在仅180天、两个指定事件上筛选，存在明显过拟合风险。5分钟 OHLC 回放把触及限价视为成交，未模拟排队；常规退出未计滑点，只有风控市价清仓计2bp。此规则应先纸面运行，并用更长的熊市、横盘和闪崩样本做样本外验证。"},
                {"id": "next", "type": "markdown",
                 "body": "## 建议的下一步\n\n1. 若目标是账户风险上限，采用推荐稀疏触发器，但先只关闭 BUY 侧。\n2. SELL 侧保留运行，除非账户库存或交易所规则要求全平。\n3. 连续纸面监控触发时间、清仓滑点和恢复后的首批成交。"},
                {"id": "questions", "type": "markdown",
                 "body": "## 仍需回答的问题\n\n- 是否必须把 SELL 侧库存也恢复到初始基准？\n- 可接受的最大提前暂停时间是多少？\n- 是否要加入1小时闪崩通道，捕捉4小时信号来不及处理的行情？"},
            ],
            "charts": [{
                "id": "strategy_return", "title": "180天 DCA 收益比较",
                "subtitle": "BTC+ETH，初始资金380 USDT；负值越接近零越好",
                "type": "bar", "dataset": "strategy_results", "sourceId": "strategy_source",
                "encodings": {
                    "x": {"field": "scenario", "type": "nominal", "label": "风控方案"},
                    "y": {"field": "return_pct", "type": "quantitative", "label": "收益率", "unit": "%"},
                    "tooltip": [
                        {"field": "net_pnl_quote", "type": "quantitative", "label": "净盈亏", "unit": "USDT"},
                        {"field": "max_drawdown_pct", "type": "quantitative", "label": "最大回撤", "unit": "%"},
                        {"field": "fees_quote", "type": "quantitative", "label": "手续费", "unit": "USDT"},
                    ],
                },
                "layout": "full", "maxRows": 10,
            }],
            "tables": [
                {"id": "thresholds", "title": "候选阈值的事件覆盖与误停",
                 "subtitle": "180天；其他暂停段越少越符合只抓快速下跌的目标",
                 "dataset": "threshold_results", "sourceId": "threshold_source", "density": "spacious",
                 "columns": [
                     {"field": "candidate", "label": "候选规则", "type": "text"},
                     {"field": "february_hit", "label": "覆盖2月", "type": "text"},
                     {"field": "june_hit", "label": "覆盖6月", "type": "text"},
                     {"field": "risk_off_episodes", "label": "暂停段数", "format": "number"},
                     {"field": "other_episodes", "label": "其他暂停", "format": "number"},
                     {"field": "risk_off_pct", "label": "暂停占比", "format": "number"},
                 ]},
                {"id": "events", "title": "推荐规则的事件时间线",
                 "subtitle": "UTC；触发在4小时收盘确认后生效",
                 "dataset": "event_results", "sourceId": "event_source", "density": "spacious",
                 "columns": [
                     {"field": "event", "label": "事件", "type": "text"},
                     {"field": "trigger_time", "label": "触发", "type": "text"},
                     {"field": "recovery_time", "label": "恢复", "type": "text"},
                     {"field": "roc_48h_pct", "label": "48h ROC", "format": "number"},
                     {"field": "sqzmom_pct", "label": "SQZMOM/价格", "format": "number"},
                     {"field": "post_trigger_low_pct", "label": "触发后最低跌幅", "format": "number", "movement": True},
                 ]},
            ],
        },
        "snapshot": {
            "version": 1, "generatedAt": generated, "status": "ready",
            "datasets": {
                "strategy_results": strategy_rows,
                "threshold_results": threshold_rows,
                "event_results": event_rows,
            },
        },
        "sources": sources,
    }
    (output / "artifact.json").write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--end", default="2026-07-27T02:00:00Z")
    parser.add_argument("--days", type=int, default=180)
    parser.add_argument("--cache-dir", type=Path, default=Path("data/backtesting_candles"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/backtests/dca_crash_guard_thresholds_180d"))
    args = parser.parse_args()
    end = floor_five_minutes(datetime.fromisoformat(args.end.replace("Z", "+00:00")))
    start = end - timedelta(days=args.days)
    signals = build_signal_bars(args.cache_dir / "BTCUSDT_5m.csv", "4h", 12, 20)
    signals = signals[(signals.effective_time >= start) & (signals.effective_time < end)].copy()
    signals["sqz_pct"] = signals.sqz_val / signals.close * 100
    signals["roc3"] = signals.close.pct_change(3) * 100
    green = signals.sqz_val > 0
    total_hours = (end - start).total_seconds() / 3600
    candidates = []
    candidate_defs = [
        ("SQZMOM ≤ -2%, green恢复", signals.sqz_pct <= -2, green),
        ("SQZMOM ≤ -4%, green恢复", signals.sqz_pct <= -4, green),
        ("SQZMOM ≤ -5%, green恢复", signals.sqz_pct <= -5, green),
        ("ROC(12) ≤ -5%且红柱, green恢复", (signals.roc <= -5) & signals.sqz_color.eq("red"), green),
        ("ROC(3) ≤ -4.5%, ROC>-1%两根恢复", signals.roc3 <= -4.5,
         (signals.roc3 > -1).rolling(2).sum().eq(2)),
        ("推荐: ROC(12)≤-8% + SQZMOM≤-3% + 红柱", (signals.roc <= -8) &
         (signals.sqz_pct <= -3) & signals.sqz_color.eq("red"), green),
    ]
    recommended_state = None
    recommended_periods = None
    for name, trigger, recovery in candidate_defs:
        state, periods = stateful_gate(signals, trigger, recovery)
        candidates.append(summarize_candidate(name, periods, total_hours))
        if name.startswith("推荐"):
            recommended_state, recommended_periods = state, periods

    frames = {pair: load_window(args.cache_dir / f"{symbol}_5m.csv", start, end)
              for pair, symbol in PAIRS.items()}
    strategy_rows = [
        run_strategy(frames, signals, recommended_state, 0.001, 2, "无门控", ("BUY", "SELL")),
        run_strategy(frames, signals, recommended_state, 0.001, 2, "快跌时两侧全停", ("BUY", "SELL")),
        run_strategy(frames, signals, recommended_state, 0.001, 2, "快跌时仅停BUY", ("BUY",)),
    ]
    event_rows = []
    for index, ((left, right), (event_name, _, _)) in enumerate(zip(recommended_periods, EVENTS)):
        trigger_row = signals.loc[signals.effective_time.eq(left)].iloc[0]
        market = signals[(signals.effective_time >= left) & (signals.effective_time <= right)]
        trigger_price = float(trigger_row.close)
        event_rows.append({
            "event": event_name,
            "trigger_time": left.strftime("%Y-%m-%d %H:%M"),
            "recovery_time": right.strftime("%Y-%m-%d %H:%M"),
            "roc_48h_pct": round(float(trigger_row.roc), 3),
            "sqzmom_pct": round(float(trigger_row.sqz_pct), 3),
            "trigger_price": round(trigger_price, 2),
            "minimum_price": round(float(market.low.min()), 2),
            "post_trigger_low_pct": round((float(market.low.min()) / trigger_price - 1) * 100, 3),
            "pause_hours": (right - left).total_seconds() / 3600,
        })
    output = args.output_dir / end.strftime("%Y%m%dT%H%M%SZ")
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(candidates).to_csv(output / "threshold_summary.csv", index=False)
    pd.DataFrame(strategy_rows).to_csv(output / "strategy_summary.csv", index=False)
    pd.DataFrame(event_rows).to_csv(output / "event_summary.csv", index=False)
    generated = datetime.now(timezone.utc).isoformat()
    build_artifact(output, generated, candidates, strategy_rows, event_rows)
    print(pd.DataFrame(candidates).to_string(index=False))
    print(pd.DataFrame(strategy_rows).to_string(index=False))
    print(pd.DataFrame(event_rows).to_string(index=False))
    print(output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

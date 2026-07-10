#!/usr/bin/env python
"""Create a self-contained strategy catalog and performance report.

The generator intentionally separates backtests from paper trading. It only
uses formal result files, keeps untested strategy entrypoints visible, and
writes an offline HTML report plus CSV/JSON audit artifacts.
"""

from __future__ import annotations

import argparse
import html
import json
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from strategy_execution_metrics import replay_fixed, replay_walk_forward


TOP10_PAIRS = ["BTC-USDT", "ETH-USDT", "SOL-USDT", "BNB-USDT", "XRP-USDT", "DOGE-USDT", "ADA-USDT", "LINK-USDT", "AVAX-USDT", "TRX-USDT"]
TICKER_URL = "https://api.binance.com/api/v3/ticker/24hr"
PLUGIN_ROOT = Path(r"C:\Users\sunny\.codex\plugins\cache\openai-curated-remote\data-analytics\0.2.6-d37358633e00")


@dataclass
class StrategyRecord:
    key: str
    name: str
    technical_name: str
    run_type: str
    period: str
    initial_equity: float | None
    final_equity: float | None
    pnl_usdt: float | None
    pnl_pct: float | None
    max_drawdown_pct: float | None
    liquidations: int | None
    trades: int | None
    grid_moves: int | None
    fee_rate: float | None
    parameters: str
    source: str


FORMAL_BACKTESTS = {
    "fixed_original_frequency_top10_6m_fee002.csv": {
        "key": "grid_fixed_original_frequency",
        "name": "组合移动网格-原频率基准｜R04-L24-Q2%-TP0.3%-M0.5%",
        "technical_name": "backtest_portfolio_3commas_grid.py",
        "run_type": "固定参数回测",
        "fee_rate": 0.0002,
    },
    "fixed_new_params_top10_6m.csv": {
        "key": "grid_fixed_low_frequency",
        "name": "组合移动网格-低频对照｜R04-L24-Q2%-TP0.3%-M1.5%-CD30m",
        "technical_name": "backtest_portfolio_3commas_grid.py",
        "run_type": "固定参数回测",
        "fee_rate": 0.0002,
    },
}

UNTESTED = [
    ("做市-PMM Mister｜inventory-skew", "backtest_pmm_mister.py", "做市", "缺少正式回测结果"),
    ("网格-Grid Strike｜trend-grid", "backtest_grid_strike.py", "网格", "缺少正式回测结果"),
    ("布林带做市-Bollinger V2｜mean-reversion", "backtest_bollinger_v2.py", "均值回归", "缺少正式回测结果"),
    ("V2 控制器策略｜controller", "backtest_v2_controller.py", "控制器", "缺少正式回测结果"),
    ("简单做市-Simple PMM｜basic", "simple_pmm.py", "做市", "缺少正式回测结果"),
    ("VWAP 执行｜execution", "simple_vwap.py", "执行", "缺少正式回测结果"),
    ("跨所做市-XEMM｜cross-exchange", "simple_xemm.py", "套利", "缺少正式回测结果"),
    ("资金费率套利｜funding-rate", "v2_funding_rate_arb.py", "套利", "缺少正式回测结果"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backtests-dir", type=Path, default=Path("results/backtests"))
    parser.add_argument("--paper-dir", type=Path, default=Path("results/server_trade_report/2026-07-10/new_strategy_since_restart"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/strategy_catalog"))
    parser.add_argument("--scripts-dir", type=Path, default=Path("scripts"))
    parser.add_argument("--cache-dir", type=Path, default=Path("data/backtesting_candles"))
    parser.add_argument("--skip-market", action="store_true", help="Generate without querying Binance public tickers.")
    parser.add_argument("--market-timeout", type=float, default=8.0, help="Per-pair Binance public API timeout in seconds.")
    return parser.parse_args()


def safe_number(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def compact_params(row: pd.Series) -> str:
    labels = [("grid_range", "R"), ("grid_levels", "L"), ("order_quote_pct", "Q"), ("take_profit", "TP"), ("move_threshold", "M")]
    values: list[str] = []
    for column, label in labels:
        if column not in row or pd.isna(row[column]):
            continue
        value = float(row[column])
        values.append(f"{label}{value * 100:g}%" if label in {"R", "Q", "TP", "M"} else f"{label}{int(value)}")
    if "min_grid_move_seconds" in row and not pd.isna(row["min_grid_move_seconds"]) and float(row["min_grid_move_seconds"]) > 0:
        values.append(f"CD{float(row['min_grid_move_seconds']) / 60:g}m")
    return " ".join(values)


def fixed_records(backtests_dir: Path) -> list[StrategyRecord]:
    records: list[StrategyRecord] = []
    for filename, metadata in FORMAL_BACKTESTS.items():
        path = backtests_dir / filename
        if not path.exists():
            continue
        row = pd.read_csv(path).iloc[0]
        records.append(StrategyRecord(
            key=metadata["key"], name=metadata["name"], technical_name=metadata["technical_name"],
            run_type=metadata["run_type"], period="Top10 USDT / 6 months / 5m", initial_equity=safe_number(row.get("initial_quote")),
            final_equity=safe_number(row.get("final_equity")), pnl_usdt=safe_number(row.get("net_pnl_quote")),
            pnl_pct=safe_number(row.get("net_pnl_pct")), max_drawdown_pct=safe_number(row.get("max_drawdown_pct")),
            liquidations=int(bool(row.get("liquidated"))), trades=int(row.get("trades", 0)), grid_moves=int(row.get("grid_moves", 0)),
            fee_rate=metadata["fee_rate"], parameters=compact_params(row), source=str(path),
        ))
    return records


def md_value(text: str, label: str) -> str | None:
    match = re.search(rf"^- {re.escape(label)}:\s*(.+)$", text, flags=re.MULTILINE)
    return match.group(1).strip() if match else None


def md_float(text: str, label: str) -> float | None:
    value = md_value(text, label)
    if not value:
        return None
    match = re.search(r"[-+]?\d+(?:\.\d+)?", value)
    return float(match.group()) if match else None


def optimized_records(backtests_dir: Path) -> list[StrategyRecord]:
    staged = backtests_dir / "optimize_fixed_top10_6m_fee002_result.md"
    walk = backtests_dir / "walk_forward_top_candidates_top10_6m_fee002_result.md"
    records: list[StrategyRecord] = []
    if staged.exists():
        text = staged.read_text(encoding="utf-8")
        records.append(StrategyRecord(
            key="grid_staged_fixed", name="组合移动网格-阶段优化固定版｜R10-L08-Q0.5%-TP0.5%-M0.5%",
            technical_name="optimize_portfolio_grid_staged.py", run_type="优化后固定参数回测",
            period="Top10 USDT / 6 months / 5m", initial_equity=10000.0, final_equity=md_float(text, "final_equity"),
            pnl_usdt=md_float(text, "net_pnl"), pnl_pct=(md_float(text, "net_pnl") or 0) / 10000,
            max_drawdown_pct=(md_float(text, "max_drawdown") or 0) / 100, liquidations=0 if md_value(text, "liquidated") == "False" else 1,
            trades=int(md_float(text, "trades") or 0), grid_moves=int(md_float(text, "grid_moves") or 0), fee_rate=0.0002,
            parameters="R10% L8 Q0.5% TP0.5% M0.5%", source=str(staged),
        ))
    if walk.exists():
        text = walk.read_text(encoding="utf-8")
        records.append(StrategyRecord(
            key="grid_walk_forward_yield", name="滑窗组合网格-收益优先版｜WF-R08-L08-Q0.5%-TP0.5%-M0.5%",
            technical_name="backtest_walk_forward_portfolio_grid.py", run_type="滑窗 walk-forward 回测",
            period="Top10 USDT / 2026-02-07 to 2026-07-09 / 5m", initial_equity=10000.0,
            final_equity=md_float(text, "Final equity"), pnl_usdt=md_float(text, "Net PnL"),
            pnl_pct=(md_float(text, "Net PnL") or 0) / 10000, max_drawdown_pct=(md_float(text, "Max weekly-window drawdown") or 0) / 100,
            liquidations=int(md_float(text, "Total liquidations") or 0), trades=None, grid_moves=int(md_float(text, "Total grid moves") or 0),
            fee_rate=0.0002, parameters="WF R8% L8 Q0.5% TP0.5% M0.5% SL4%", source=str(walk),
        ))
    return records


def paper_record(paper_dir: Path) -> StrategyRecord | None:
    summary_path = paper_dir / "pair_summary.csv"
    fills_path = paper_dir / "trade_fills.csv"
    if not summary_path.exists() or not fills_path.exists():
        return None
    summary = pd.read_csv(summary_path)
    fills = pd.read_csv(fills_path)
    timestamps = pd.to_datetime(fills["timestamp"], utc=True)
    rates = pd.to_numeric(fills.get("fee_rate"), errors="coerce").dropna()
    return StrategyRecord(
        key="grid_walk_forward_paper", name="滑窗组合网格-Paper当前运行｜R04-L24-Q2%-TP0.3%-M0.5%",
        technical_name="walk_forward_portfolio_grid.py", run_type="Binance paper trading", period=f"{timestamps.min():%Y-%m-%d %H:%M} to {timestamps.max():%Y-%m-%d %H:%M} UTC",
        initial_equity=None, final_equity=None, pnl_usdt=float(summary["mark_to_market"].sum()), pnl_pct=None,
        max_drawdown_pct=None, liquidations=None, trades=int(summary["trades"].sum()), grid_moves=None,
        fee_rate=float(rates.iloc[0]) if not rates.empty else None, parameters="R4% L24 Q2% TP0.3% M0.5%",
        source=str(paper_dir),
    )


def execution_details(backtests_dir: Path, paper_dir: Path, cache_dir: Path) -> list[dict[str, Any]]:
    """Collect replayed grid events and code-derived documentation entries."""
    fixed = [
        ("grid_fixed_original_frequency", "组合移动网格-原频率基准｜R04-L24-Q2%-TP0.3%-M0.5%", 28124, 218, {
            "grid_range": 0.04, "grid_levels": 24, "order_quote_pct": 0.02, "take_profit": 0.003,
            "move_threshold": 0.005, "portfolio_stop_loss": 0.08, "min_grid_move_seconds": 0.0,
        }),
        ("grid_fixed_low_frequency", "组合移动网格-低频对照｜R04-L24-Q2%-TP0.3%-M1.5%-CD30m", 23929, 130, {
            "grid_range": 0.04, "grid_levels": 24, "order_quote_pct": 0.02, "take_profit": 0.003,
            "move_threshold": 0.015, "portfolio_stop_loss": 0.08, "min_grid_move_seconds": 1800.0,
        }),
        ("grid_staged_fixed", "组合移动网格-阶段优化固定版｜R10-L08-Q0.5%-TP0.5%-M0.5%", 28076, 555, {
            "grid_range": 0.10, "grid_levels": 8, "order_quote_pct": 0.005, "take_profit": 0.005,
            "move_threshold": 0.005, "portfolio_stop_loss": 0.08, "min_grid_move_seconds": 0.0,
        }),
    ]
    details: list[dict[str, Any]] = []
    for key, name, archived_trades, archived_moves, params in fixed:
        try:
            stats = replay_fixed(cache_dir, params)
            stats["archived_trades"] = archived_trades
            stats["archived_grid_moves"] = archived_moves
            if stats["trades"] == archived_trades and stats["grid_moves"] == archived_moves:
                status = "已重放验证"
            else:
                status = "重放完成，归档存在差异"
        except (FileNotFoundError, RuntimeError, ValueError) as error:
            stats, status = {"error": str(error)}, "统计不可用"
        details.append({"key": key, "name": name, "kind": "组合移动网格", "status": status, "params": params, "stats": stats,
                        "steps": ["按 Top10 等权分配组合权益", "价格下穿网格层时限价买入", "达到该笔入场价加止盈幅度时卖出", "越出网格边界后按阈值移动网格"],
                        "risk": "组合权益相对历史峰值回撤达到 8%：取消挂单、卖出非 USDT、冷却 24 小时。", "source": "本地 5 分钟缓存与正式回测参数"})

    walk_summary = backtests_dir / "walk_forward_top_candidates_top10_6m_fee002_summary.csv"
    try:
        walk_stats = replay_walk_forward(cache_dir, walk_summary)
        walk_status = "已按每周选参重放"
    except (FileNotFoundError, RuntimeError, ValueError, KeyError) as error:
        walk_stats, walk_status = {"error": str(error), "weekly": []}, "统计不可用"
    details.append({"key": "grid_walk_forward_yield", "name": "滑窗组合网格-收益优先版｜WF-R08-L08-Q0.5%-TP0.5%-M0.5%", "kind": "滑窗组合移动网格", "status": walk_status,
                    "params": {"lookback_days": 7, "trade_days": 7, "portfolio_stop_loss": 0.08, "stop_loss": 0.04}, "stats": walk_stats,
                    "steps": ["首月训练，不交易", "每周回看前 7 天的候选参数表现", "按收益减回撤惩罚分数选出参数", "用该参数交易下一周并保留组合状态"],
                    "risk": "组合回撤达到 8% 时清仓并冷却 24 小时。stop_loss=4% 当前只作为候选参数/评分字段，不是独立单交易对清仓触发。", "source": str(walk_summary)})

    fills_path = paper_dir / "trade_fills.csv"
    if fills_path.exists():
        fills = pd.read_csv(fills_path)
        buy_count = int((fills["trade_type"].str.upper() == "BUY").sum())
        sell_count = int((fills["trade_type"].str.upper() == "SELL").sum())
        by_pair = fills.groupby(["symbol", "trade_type"]).size().unstack(fill_value=0)
        details.append({"key": "grid_walk_forward_paper", "name": "滑窗组合网格-Paper当前运行｜R04-L24-Q2%-TP0.3%-M0.5%", "kind": "Paper 组合移动网格", "status": "逐笔成交统计", 
                        "params": {"grid_range": 0.04, "grid_levels": 24, "order_quote_pct": 0.02, "take_profit": 0.003, "move_threshold": 0.005, "portfolio_stop_loss": 0.08, "refresh_seconds": 60},
                        "stats": {"buy_entries": buy_count, "take_profit_sells": sell_count, "trades": len(fills), "per_pair": by_pair.reset_index().to_dict("records")},
                        "steps": ["每 60 秒刷新组合网格", "每个交易对在当前价格下方挂多层买单", "持仓按上方网格/止盈价挂卖单", "价格越界 0.5% 时移动网格"],
                        "risk": "组合回撤达到 8% 时按策略逻辑全清仓并冷却 24 小时。", "source": str(fills_path)})
    else:
        details.append({"key": "grid_walk_forward_paper", "name": "滑窗组合网格-Paper当前运行｜R04-L24-Q2%-TP0.3%-M0.5%", "kind": "Paper 组合移动网格", "status": "统计不可用", "params": {}, "stats": {"error": f"缺少 {fills_path}"}, "steps": [], "risk": "组合回撤 8% 清仓并冷却 24 小时。", "source": str(fills_path)})

    for name, script, family, status in UNTESTED:
        details.append({"key": script, "name": name, "kind": family, "status": "待验证", "params": {}, "stats": {}, "source": f"scripts/{script}", "risk": "尚未产生正式回测统计。",
                        "steps": untested_steps(script)})
    return details


def untested_steps(script: str) -> list[str]:
    specs = {
        "backtest_pmm_mister.py": ["根据库存偏离调整双边报价", "成交后重新平衡库存", "以库存风险和报价偏斜控制敞口"],
        "backtest_grid_strike.py": ["按方向与最小订单间距创建网格", "最多维持限定数量开放订单", "达到止盈条件后退出"],
        "backtest_bollinger_v2.py": ["用布林带识别价格偏离", "按方向开仓", "由止盈、止损、时间限制和冷却控制退出"],
        "backtest_v2_controller.py": ["控制器解析买卖价差层", "可选 DCA 价差分层补仓", "由止盈和止损退出执行器"],
        "simple_pmm.py": ["围绕参考价两侧挂限价单", "买卖价差由 bid/ask spread 决定", "按刷新周期撤单后重挂"],
        "simple_vwap.py": ["按 VWAP 执行计划切分订单", "根据成交进度调整剩余数量", "完成目标量后停止"],
        "simple_xemm.py": ["Maker 侧挂双边报价", "任一 Maker 成交后在 Taker 侧对冲", "预期利润低于阈值时撤单重报"],
        "v2_funding_rate_arb.py": ["比较两侧资金费率与建仓成本", "构建多空对冲仓位", "达到收益目标或费率差恶化时退出"],
    }
    return specs.get(script, ["代码入口已发现，尚待补充正式回测。"])


def market_snapshot(skip_market: bool, timeout: float) -> tuple[list[dict[str, Any]], str]:
    generated = datetime.now(timezone.utc).isoformat()
    if skip_market:
        return [], "Skipped by --skip-market"
    def fetch(pair: str) -> dict[str, Any]:
        try:
            response = requests.get(TICKER_URL, params={"symbol": pair.replace("-", "")}, timeout=timeout)
            response.raise_for_status()
            payload = response.json()
            return {"pair": pair, "last_price": float(payload["lastPrice"]), "change_pct": float(payload["priceChangePercent"]) / 100, "quote_volume": float(payload["quoteVolume"]), "close_time": int(payload["closeTime"])}
        except (requests.RequestException, KeyError, TypeError, ValueError) as error:
            return {"pair": pair, "error": str(error)}
    with ThreadPoolExecutor(max_workers=5, thread_name_prefix="binance-ticker") as pool:
        rows = list(pool.map(fetch, TOP10_PAIRS))
    return rows, generated


def tooltip(value: str, source: str, tooltip_id: int) -> str:
    return (f'<span class="source-tooltip" tabindex="0" aria-describedby="src-{tooltip_id}">{value}'
            f'<span class="source-tooltip-content" id="src-{tooltip_id}" role="tooltip">Source: local report inputs<br>File: {html.escape(source)}</span></span>')


def money(value: float | None) -> str:
    return "-" if value is None else f"{value:+,.2f}"


def percent(value: float | None) -> str:
    return "-" if value is None else f"{value:+.2%}"


def bar_svg(records: list[StrategyRecord]) -> str:
    chart_rows = [record for record in records if record.pnl_pct is not None]
    if not chart_rows:
        return "<table><tr><td>No comparable return data available.</td></tr></table>"
    width, height, left, top = 960, 390, 250, 35
    max_abs = max(abs(record.pnl_pct or 0) for record in chart_rows) or 0.01
    zero = left + (width - left - 55) / 2
    scale = (width - left - 55) / (2 * max_abs)
    spacing = 42
    parts = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="Backtest return comparison"><line x1="{zero:.1f}" y1="20" x2="{zero:.1f}" y2="{height - 25}" stroke="#8f8f8f"/>']
    for index, record in enumerate(chart_rows):
        y = top + index * spacing
        value = record.pnl_pct or 0
        bar_width = abs(value) * scale
        x = zero if value >= 0 else zero - bar_width
        color = "#0169cc" if value >= 0 else "#e25507"
        label = html.escape(record.name.split("｜")[0])
        parts.append(f'<text x="{left - 8}" y="{y + 15}" text-anchor="end" font-size="12" fill="#5d5d5d">{label}</text>')
        parts.append(f'<rect x="{x:.1f}" y="{y}" width="{bar_width:.1f}" height="22" rx="2" fill="{color}"/>')
        parts.append(f'<text x="{(x + bar_width + 6) if value >= 0 else (x - 6):.1f}" y="{y + 15}" text-anchor="{"start" if value >= 0 else "end"}" font-size="12" fill="#0d0d0d">{value:+.2%}</text>')
    parts.append("</svg>")
    return "".join(parts)


def execution_flow_svg(steps: list[str]) -> str:
    if not steps:
        return "<p class=\"chart-note\">暂无可展示的执行步骤。</p>"
    width = 920
    box_width = max(150, min(220, (width - 40 * (len(steps) - 1)) / len(steps)))
    parts = [f'<svg class="flow-svg" viewBox="0 0 {width} 150" role="img" aria-label="策略执行流程图">']
    for index, step in enumerate(steps):
        x = 12 + index * (box_width + 40)
        text = html.escape(step)
        parts.append(f'<rect x="{x:.1f}" y="40" width="{box_width:.1f}" height="65" fill="#eef5fb" stroke="#0169cc"/>')
        parts.append(f'<text x="{x + box_width / 2:.1f}" y="67" text-anchor="middle" font-size="12" fill="#0d0d0d">{text[:17]}</text>')
        if len(text) > 17:
            parts.append(f'<text x="{x + box_width / 2:.1f}" y="84" text-anchor="middle" font-size="12" fill="#0d0d0d">{text[17:34]}</text>')
        if index < len(steps) - 1:
            parts.append(f'<path d="M{x + box_width + 8:.1f} 72 L{x + box_width + 30:.1f} 72" stroke="#5d5d5d"/><path d="M{x + box_width + 30:.1f} 72 l-7 -5 M{x + box_width + 30:.1f} 72 l-7 5" stroke="#5d5d5d" fill="none"/>')
    return "".join(parts) + "</svg>"


def execution_stats_svg(details: list[dict[str, Any]]) -> str:
    rows = [detail for detail in details if detail["kind"].endswith("网格") and detail["stats"].get("buy_entries") is not None]
    if not rows:
        return "<p>暂无可比较的重放统计。</p>"
    width, height, left = 960, 90 + 52 * len(rows), 270
    maximum = max(max(int(row["stats"].get(key, 0)) for key in ("buy_entries", "take_profit_sells", "grid_moves")) for row in rows) or 1
    colors = {"buy_entries": "#0169cc", "take_profit_sells": "#d89b00", "grid_moves": "#8046d9"}
    labels = {"buy_entries": "补仓", "take_profit_sells": "止盈", "grid_moves": "移动"}
    parts = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="网格策略执行事件比较">']
    for index, detail in enumerate(rows):
        y = 44 + index * 52
        parts.append(f'<text x="{left - 10}" y="{y + 16}" text-anchor="end" font-size="12" fill="#0d0d0d">{html.escape(detail["name"].split("｜")[0])}</text>')
        cursor = left
        for key in ("buy_entries", "take_profit_sells", "grid_moves"):
            value = int(detail["stats"].get(key, 0))
            bar = value / maximum * 600
            parts.append(f'<rect x="{cursor:.1f}" y="{y}" width="{bar:.1f}" height="18" fill="{colors[key]}"/>')
            if bar > 28:
                parts.append(f'<text x="{cursor + 4:.1f}" y="{y + 13}" font-size="11" fill="#fff">{labels[key]} {value}</text>')
            cursor += bar
    parts.append('</svg><p class="chart-note">蓝色：补仓买入；金色：止盈卖出；紫色：网格移动。不同策略按各自时间段重放，不代表同一绝对交易频率。</p>')
    return "".join(parts)


def execution_detail_html(details: list[dict[str, Any]]) -> str:
    cards: list[str] = []
    for detail in details:
        params = detail["params"]
        stats = detail["stats"]
        params_text = "<br>".join(f"<b>{html.escape(key)}</b>: {value}" for key, value in params.items()) or "使用脚本默认参数；尚未形成正式运行配置。"
        if "error" in stats:
            metrics = f"<p class=\"chart-note\">统计不可用：{html.escape(str(stats['error']))}</p>"
        elif detail["key"] == "grid_walk_forward_yield":
            metrics = (f"<div class=\"execution-metrics\"><span>补仓 {stats.get('buy_entries', 0):,}</span><span>止盈 {stats.get('take_profit_sells', 0):,}</span><span>移动 {stats.get('grid_moves', 0):,}</span><span>清仓 {stats.get('liquidations', 0)}</span></div>"
                       f"<p class=\"chart-note\">每周实际参数重放：{len(stats.get('weekly', []))} 周；每周参数和补仓次数显示在下方时间线。</p>" + walk_week_table(stats.get("weekly", [])))
        elif detail["key"] == "grid_walk_forward_paper":
            metrics = (f"<div class=\"execution-metrics\"><span>买入成交 {int(stats.get('buy_entries', 0)):,}</span><span>卖出成交 {int(stats.get('take_profit_sells', 0)):,}</span><span>总成交 {int(stats.get('trades', 0)):,}</span></div>"
                       "<p class=\"chart-note\">Paper 快照只能精确识别成交方向，卖出成交不等同于逐笔止盈成交。</p>")
        elif stats:
            metrics = (f"<div class=\"execution-metrics\"><span>补仓 {int(stats.get('buy_entries', 0)):,}</span><span>止盈 {int(stats.get('take_profit_sells', 0)):,}</span><span>移动 {int(stats.get('grid_moves', 0)):,}</span><span>清仓 {int(stats.get('liquidations', 0))}</span></div>"
                       f"<p class=\"chart-note\">完成循环：{int(stats.get('completed_cycles', 0)):,}；清仓卖出：{int(stats.get('liquidation_sells', 0)):,}；总交易：{int(stats.get('trades', 0)):,}。{'归档总交易：' + str(stats['archived_trades']) + '；归档移动：' + str(stats['archived_grid_moves']) + '。' if 'archived_trades' in stats else ''}</p>")
        else:
            metrics = "<p class=\"chart-note\">尚无正式运行统计。</p>"
        spacing = "-"
        if params.get("grid_range") and params.get("grid_levels", 0) > 1:
            spacing = f"{params['grid_range'] / (params['grid_levels'] - 1):.4%}（相邻网格）"
        open_attr = " open" if detail["key"] in {"grid_fixed_original_frequency", "grid_fixed_low_frequency", "grid_staged_fixed", "grid_walk_forward_yield", "grid_walk_forward_paper"} else ""
        cards.append(f'''<details class="execution-card"{open_attr}>
          <summary><span>{html.escape(detail['name'])}</span><span class="pill {'warn' if detail['status'] != '已重放验证' and '逐笔' not in detail['status'] and '已按' not in detail['status'] else ''}">{html.escape(detail['status'])}</span></summary>
          <div class="execution-body"><div class="flow-wrap">{execution_flow_svg(detail['steps'])}</div>
          <div class="execution-columns"><div><h3>执行规则</h3><p>网格边界 = 中心价 × (1 ± grid_range / 2)。相邻间距：{spacing}。{params_text}</p></div><div><h3>风控与限制</h3><p>{html.escape(detail['risk'])}</p><p class="chart-note">来源：{html.escape(detail['source'])}</p></div></div>{metrics}</div></details>''')
    return "".join(cards)


def walk_week_table(weekly: list[dict[str, object]]) -> str:
    if not weekly:
        return ""
    rows = "".join(f"<tr><td>{item['week_index']}</td><td>{float(item['pnl_pct']):+.2%}</td><td>{float(item['drawdown_pct']):+.2%}</td><td>R{float(item['grid_range']):.0%}/L{item['grid_levels']}/M{float(item['move_threshold']):.1%}</td><td>{item['buy_entries']}</td><td>{item['grid_moves']}</td></tr>" for item in weekly)
    return f'<div class="table-scroll"><table class="week-table"><thead><tr><th>周</th><th>收益</th><th>回撤</th><th>实际参数</th><th>补仓</th><th>移动</th></tr></thead><tbody>{rows}</tbody></table></div>'


def render_html(records: list[StrategyRecord], pending: list[tuple[str, str, str, str]], details: list[dict[str, Any]], market: list[dict[str, Any]], market_generated: str, output: Path) -> tuple[str, dict[str, Any]]:
    template = (PLUGIN_ROOT / "assets" / "html-report-shell.html").read_text(encoding="utf-8")
    template = template.replace("</style>", """
    .execution-card { margin: 14px 0; border: 1px solid var(--border); border-radius: 8px; background: var(--surface); }
    .execution-card summary { display: flex; gap: 12px; align-items: center; justify-content: space-between; padding: 14px 16px; cursor: pointer; font-weight: 600; }
    .execution-body { padding: 0 16px 18px; border-top: 1px solid var(--border); }
    .flow-wrap { margin: 16px 0; overflow-x: auto; border-bottom: 1px solid var(--border); }
    .flow-svg { display: block; min-width: 720px; width: 100%; height: auto; }
    .execution-columns { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
    .execution-columns p { margin: 6px 0; color: var(--secondary); }
    .execution-metrics { display: flex; flex-wrap: wrap; gap: 8px; margin: 16px 0 4px; }
    .execution-metrics span { padding: 5px 8px; border: 1px solid var(--border-strong); font-variant-numeric: tabular-nums; }
    .week-table { margin-top: 12px; }
    @media (max-width: 650px) { .execution-columns { grid-template-columns: 1fr; gap: 8px; } .execution-card summary { align-items: flex-start; } }
  </style>""", 1)
    backtests = [record for record in records if record.run_type != "Binance paper trading"]
    paper = next((record for record in records if record.run_type == "Binance paper trading"), None)
    profitable = sum(1 for record in backtests if (record.pnl_pct or 0) > 0)
    tooltip_index = 0
    def cell(value: str, source: str, class_name: str = "") -> str:
        nonlocal tooltip_index
        tooltip_index += 1
        return f'<td class="{class_name}">{tooltip(value, source, tooltip_index)}</td>'

    backtest_rows = []
    for record in sorted(backtests, key=lambda item: item.pnl_pct or -999, reverse=True):
        css = "positive" if (record.pnl_pct or 0) > 0 else "negative" if (record.pnl_pct or 0) < 0 else ""
        backtest_rows.append("<tr>" + cell(html.escape(record.name), record.source) + cell(record.run_type, record.source) + cell(percent(record.pnl_pct), record.source, css) + cell(money(record.pnl_usdt), record.source, css) + cell(percent(record.max_drawdown_pct), record.source, "negative") + cell(str(record.trades or "-"), record.source) + cell(str(record.grid_moves or "-"), record.source) + cell(record.parameters, record.source) + "</tr>")
    performance_table = ("<table><thead><tr><th>策略</th><th>类型</th><th>收益率</th><th>收益 (USDT)</th><th>最大回撤</th><th>交易数</th><th>网格移动</th><th>参数</th></tr></thead><tbody>" + "".join(backtest_rows) + "</tbody></table>")
    paper_html = "<p>本地 paper 快照不可用。</p>"
    if paper:
        paper_html = ("<table><thead><tr><th>策略</th><th>运行区间</th><th>成交数</th><th>记录手续费</th><th>成交流量市值口径</th><th>手续费率</th></tr></thead><tbody><tr>" +
                      cell(html.escape(paper.name), paper.source) + cell(paper.period, paper.source) + cell(str(paper.trades), paper.source) +
                      cell("见 trade_fills.csv", paper.source) + cell(money(paper.pnl_usdt) + " USDT", paper.source, "positive" if (paper.pnl_usdt or 0) >= 0 else "negative") +
                      cell(percent(paper.fee_rate), paper.source) + "</tr></tbody></table><p class=\"chart-note\">成交流量市值口径为成交现金流加净库存按报告末端价格估值，不等同于 Hummingbot paper 账户的虚拟总余额。</p>")
    pending_rows = "".join(f"<tr><td>{html.escape(name)}</td><td><code>{html.escape(script)}</code></td><td>{html.escape(family)}</td><td>{html.escape(status)}</td></tr>" for name, script, family, status in pending)
    market_rows = "".join(
        f"<tr><td>{html.escape(item['pair'])}</td><td>{item.get('last_price', '-'):,.8g}</td><td class=\"{'positive' if item.get('change_pct', 0) >= 0 else 'negative'}\">{item.get('change_pct', 0):+.2%}</td><td>{item.get('quote_volume', 0):,.0f}</td></tr>"
        if "error" not in item else f"<tr><td>{html.escape(item['pair'])}</td><td colspan=\"3\">Unavailable: {html.escape(item['error'])}</td></tr>"
        for item in market
    ) or "<tr><td colspan=\"4\">Market snapshot not fetched.</td></tr>"
    summary = (f"<p><strong>{len(backtests)} 个正式回测记录中 {profitable} 个为正收益。</strong> 当前最佳正式回测为滑窗组合网格，收益 {percent(max((record.pnl_pct for record in backtests if record.pnl_pct is not None), default=None))}；固定原频率基准触发了组合清仓。</p>"
               f"<p><strong>Paper 记录单列。</strong> {'当前本地快照收益 ' + money(paper.pnl_usdt) + ' USDT。' if paper else '当前未找到本地 paper 快照。'}</p>")
    cards = (f'<div class="metric"><div class="metric-label">正式回测</div><div class="metric-value">{len(backtests)}</div><div class="metric-note">不含 smoke 测试</div></div>'
             f'<div class="metric"><div class="metric-label">正收益回测</div><div class="metric-value">{profitable}</div><div class="metric-note">同一回测口径内比较</div></div>'
             f'<div class="metric"><div class="metric-label">Paper 状态</div><div class="metric-value">{"已载入" if paper else "未找到"}</div><div class="metric-note">使用本地快照</div></div>'
             f'<div class="metric"><div class="metric-label">报告时间</div><div class="metric-value">{datetime.now():%m-%d %H:%M}</div><div class="metric-note">本地生成</div></div>')
    extra = f'''<section class="narrative" data-contract-section="execution-details"><h2>策略如何执行</h2><p>以下统计将补仓买入、止盈卖出、网格移动与组合清仓分开记录。正式回测使用本地 K 线和同参数重放；待验证策略仅解释代码逻辑。</p><div class="card table-card"><div class="chart-wrap">{execution_stats_svg(details)}</div></div>{execution_detail_html(details)}</section>
      <section class="narrative"><h2>Paper trading 当前运行</h2>{paper_html}</section>
      <section class="narrative"><h2>待验证策略目录</h2><div class="card table-card"><div class="table-scroll"><table><thead><tr><th>策略</th><th>脚本</th><th>类别</th><th>状态</th></tr></thead><tbody>{pending_rows}</tbody></table></div></div></section>
      <section class="narrative"><h2>Binance 市场快照</h2><p>公开 Binance Spot 24 小时价格变化与 USDT 成交额。抓取时间：{html.escape(market_generated)}。</p><div class="card table-card"><div class="table-scroll"><table><thead><tr><th>交易对</th><th>最新价</th><th>24h 涨跌</th><th>24h USDT 成交额</th></tr></thead><tbody>{market_rows}</tbody></table></div></div></section>'''
    replacements = {
        "{{TITLE}}": "策略目录与收益报告", "{{REPORT_AUDIENCE}}": "product stakeholders", "{{SOURCE_AND_DATE}}": "本地回测结果、paper 快照与 Binance Spot 市场数据",
        "{{KICKER}}": "HUMMINGBOT STRATEGY CATALOG", "{{ANSWER_FIRST_DECK}}": "已验证策略以统一的收益、回撤和交易活动指标比较；paper 记录与回测严格分区。",
        "{{SUMMARY_PARAGRAPHS}}": summary, "{{OPTIONAL_METRIC_CARDS}}": cards,
        "{{INSIGHT_LED_FINDING}}": "滑窗优化是唯一正收益回测", "{{CLAIM_EVIDENCE_INTERPRETATION}}": "固定原频率与低频对照都触发了组合级风险控制。后续应优先扩大滑窗策略的样本外验证，而不是直接依据短期 paper 表现切换资金。",
        "{{CHART_TITLE}}": "正式回测收益率比较", "{{CHART_SUBTITLE}}": "仅纳入正式 6 个月 Top10 USDT 结果；纸面交易不与回测并列。", "{{CHART_ALT_TEXT}}": "正式回测收益率比较条形图", "{{CHART_NOTE}}": "Source: results/backtests formal CSV and Markdown result files.",
        "{{TABLE_TITLE}}": "正式回测比较", "{{TABLE_SUBTITLE}}": "可核对参数和来源；不同运行类型不混合排名。", "{{SEMANTIC_TABLE}}": performance_table,
        "{{NEXT_STEPS}}": "<ol><li>以滑窗组合网格作为主要候选，继续延长 paper 观察窗口。</li><li>为 PMM、Grid Strike 与 Bollinger 等策略补齐同口径 BTC-USDT 或 Top10 回测。</li></ol>" ,
        "{{FURTHER_QUESTIONS}}": "<p>后续可增加风险调整收益、按交易对归因和不同手续费情景的敏感性比较。</p>",
        "{{MATERIAL_CAVEATS}}": "回测不代表未来收益。paper 的成交流量市值口径不等于账户总权益；市场快照是公开行情，仅用于环境参考。",
        "{{source system or connector}}": "local strategy catalog inputs",
        "{{fully qualified table or view; list all material tables}}": "results/backtests/*.csv, *_result.md, and local paper snapshot CSV",
    }
    for token, value in replacements.items():
        template = template.replace(token, value)
    template = template.replace("<!-- Replace with a readable inline SVG or compact fallback table generated from the same reviewed rows. -->", bar_svg(backtests))
    template = template.replace("</article>\n    </main>", extra + "</article>\n    </main>", 1)
    payload = {"charts": [{"id": "chart-1", "height": 320, "type": "bar", "dataset": {"id": "strategy_returns", "title": "Strategy return", "data": [{"strategy": record.name.split("｜")[0], "return_pct": record.pnl_pct or 0} for record in backtests if record.pnl_pct is not None], "chart_spec": {"id": "chart-1", "dataset": "strategy_returns", "title": "Strategy return", "type": "bar", "encodings": {"x": {"field": "strategy", "type": "nominal"}, "y": {"field": "return_pct", "type": "quantitative"}}, "xAxisTitle": "", "yAxisTitle": "Return", "valueFormat": "percent", "settings": {"orientation": "horizontal"}}}}]}
    shell_path = output / "strategy_catalog_shell.html"
    payload_path = output / "strategy_catalog_payload.json"
    shell_path.write_text(template, encoding="utf-8")
    payload_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return str(shell_path), payload


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = fixed_records(args.backtests_dir) + optimized_records(args.backtests_dir)
    paper = paper_record(args.paper_dir)
    if paper:
        records.append(paper)
    market, market_generated = market_snapshot(args.skip_market, args.market_timeout)
    pd.DataFrame([asdict(record) for record in records]).to_csv(
        args.output_dir / "strategy_catalog.csv", index=False, encoding="utf-8-sig"
    )
    (args.output_dir / "market_snapshot.json").write_text(json.dumps({"generated_at": market_generated, "rows": market}, ensure_ascii=False, indent=2), encoding="utf-8")
    details = execution_details(args.backtests_dir, args.paper_dir, args.cache_dir)
    pd.DataFrame([{key: value for key, value in detail.items() if key not in {"steps", "stats"}} | {"steps": " | ".join(detail["steps"]), "stats": json.dumps(detail["stats"], ensure_ascii=False)} for detail in details]).to_csv(
        args.output_dir / "strategy_execution_details.csv", index=False, encoding="utf-8-sig"
    )
    shell, _ = render_html(records, UNTESTED, details, market, market_generated, args.output_dir)
    helper = PLUGIN_ROOT / "skills" / "build-report" / "scripts" / "embed_html_report_runtime.py"
    output = args.output_dir / "strategy_catalog.html"
    command = [sys.executable, str(helper), "--input", shell, "--payload", str(args.output_dir / "strategy_catalog_payload.json"), "--output", str(output)]
    try:
        subprocess.run(command, check=True)
    except (OSError, subprocess.CalledProcessError) as error:
        Path(shell).replace(output)
        print(f"Recharts runtime unavailable; wrote static fallback: {error}")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()

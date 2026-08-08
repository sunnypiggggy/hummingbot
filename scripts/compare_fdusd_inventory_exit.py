#!/usr/bin/env python3
"""Plot the full live FDUSD grid against the bounded-inventory variant."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from fdusd_live_grid_optimizer import rolling_validation_windows, select_candidate
from validate_grid_live import (
    Candidate,
    InventoryExitPolicy,
    read_cache,
    simulate,
    slice_window,
    technical_buy_gate_timeline,
)


INITIAL_EQUITY = 420.0
ONLINE = "线上模型"
NEW = "新机制"
PALETTE = {ONLINE: "#374151", NEW: "#2563EB"}
INITIAL_APPROVED = Candidate(0.03, 0.006, 0.006, 0.015, 1800)


def scenario_result(candles, candidate, gate, policy):
    trades: list[dict] = []
    result, curve, pairs = simulate(
        candles, candidate, maker_fee=0.0, taker_fee=0.001,
        order_refresh_seconds=7200, technical_buy_gate=gate,
        trade_log=trades, risk_breakers_enabled=True,
        cost_floor_enabled=True, inventory_exit_policy=policy,
    )
    return result, curve, pairs, trades


def technical_pause_regions(gate: dict[int, bool], start: int, end: int) -> list[dict]:
    regions: list[dict] = []
    paused_at: int | None = None
    for timestamp in range(start, end, 300):
        paused = not bool(gate.get(timestamp, False))
        if paused and paused_at is None:
            paused_at = timestamp
        elif not paused and paused_at is not None:
            regions.append({
                "scenario": "公共机制", "scope": "技术门：暂停新买单",
                "kind": "technical_pause", "start_ts": paused_at, "end_ts": timestamp,
            })
            paused_at = None
    if paused_at is not None:
        regions.append({
            "scenario": "公共机制", "scope": "技术门：暂停新买单",
            "kind": "technical_pause", "start_ts": paused_at, "end_ts": end,
        })
    return regions


def run_comparison(candles: dict[str, pd.DataFrame], start_ts: int, end_ts: int):
    gate = technical_buy_gate_timeline(candles["BTC-FDUSD"])
    policy = InventoryExitPolicy(25, 24 * 3600)
    weekly_rows, equity_rows, pair_rows, stop_rows = [], [], [], []
    cumulative = {ONLINE: 0.0, NEW: 0.0}
    previous = {ONLINE: INITIAL_APPROVED, NEW: INITIAL_APPROVED}
    folds = rolling_validation_windows(start_ts, end_ts)
    stop_rows.extend(technical_pause_regions(gate, folds[0][2], folds[-1][3]))

    for fold, (train_start, train_end, test_start, test_end) in enumerate(folds, 1):
        training = slice_window(candles, train_start, train_end)
        testing = slice_window(candles, test_start, test_end)
        selections = {}
        for label, policy_for_selection in ((ONLINE, None), (NEW, policy)):
            selected, candidates = select_candidate(
                training, 0.0, taker_fee=0.001, require_eligible=False,
                technical_buy_gate=gate, cost_floor_enabled=True,
                inventory_exit_policy=policy_for_selection,
            )
            eligible_count = int(candidates.attrs.get("eligible_count", 0))
            retained = eligible_count == 0
            if retained:
                selected = previous[label]
            else:
                previous[label] = selected
            selections[label] = (selected, eligible_count, retained, policy_for_selection)

        for label, (candidate, eligible_count, retained, scenario_policy) in selections.items():
            result, curve, pairs, trades = scenario_result(testing, candidate, gate, scenario_policy)
            weekly_rows.append({
                "fold": fold, "test_start": test_start, "test_end": test_end,
                "scenario": label, "training_eligible_candidates": eligible_count,
                "parameters_retained": retained, **asdict(candidate), **result,
            })
            starting_pnl = cumulative[label]
            for row in curve.itertuples(index=False):
                equity_rows.append({
                    "timestamp": int(row.timestamp),
                    "datetime": pd.to_datetime(row.timestamp, unit="s", utc=True),
                    "fold": fold, "scenario": label,
                    "cumulative_oos_pnl": starting_pnl + float(row.equity) - INITIAL_EQUITY,
                    "fold_drawdown_pct": float(row.drawdown_pct) * 100,
                })
            cumulative[label] += float(result["net_pnl_quote"])
            for pair, metrics in pairs.items():
                pair_rows.append({"fold": fold, "scenario": label, "pair": pair, **metrics})
            for trade in trades:
                if trade["reason"] == "pair_breaker_flatten":
                    stop_rows.append({
                        "scenario": label,
                        "scope": f"{label}：{trade['pair'].replace('-FDUSD', '')} 停止",
                        "kind": "pair_stop", "start_ts": int(trade["timestamp"]),
                        "end_ts": test_end,
                    })
            if result["liquidated"] and not curve.empty:
                stop_rows.append({
                    "scenario": label, "scope": f"{label}：组合全部停止",
                    "kind": "portfolio_stop", "start_ts": int(curve.timestamp.iloc[-1]),
                    "end_ts": test_end,
                })
    stops = pd.DataFrame(stop_rows)
    stops["start"] = pd.to_datetime(stops.start_ts, unit="s", utc=True)
    stops["end"] = pd.to_datetime(stops.end_ts, unit="s", utc=True)
    return (
        pd.DataFrame(weekly_rows), pd.DataFrame(equity_rows),
        pd.DataFrame(pair_rows), stops, policy,
    )


def summarize(weekly: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scenario, frame in weekly.groupby("scenario", sort=False):
        pair_frame = pairs[pairs.scenario == scenario]
        rows.append({
            "scenario": scenario,
            "oos_pnl_fdusd": float(frame.net_pnl_quote.sum()),
            "positive_folds": int((frame.net_pnl_quote > 0).sum()),
            "worst_fold_pnl": float(frame.net_pnl_quote.min()),
            "worst_drawdown_pct": float(frame.max_drawdown_pct.min()) * 100,
            "portfolio_stop_folds": int(frame.liquidated.sum()),
            "pair_stop_folds": int((pair_frame.groupby("fold").liquidations.max() > 0).sum()),
            "eligible_training_folds": int((frame.training_eligible_candidates > 0).sum()),
            "retained_parameter_folds": int(frame.parameters_retained.sum()),
            "forced_exits": int(frame.forced_inventory_exits.sum()),
            "trades": int(frame.trades.sum()),
            "fees_fdusd": float(frame.fees_quote.sum()),
        })
    return pd.DataFrame(rows)


def add_stop_timeline(fig: go.Figure, stops: pd.DataFrame, row: int) -> None:
    scope_order = [
        "技术门：暂停新买单",
        f"{ONLINE}：BTC 停止", f"{ONLINE}：ETH 停止", f"{ONLINE}：组合全部停止",
        f"{NEW}：BTC 停止", f"{NEW}：ETH 停止", f"{NEW}：组合全部停止",
    ]
    styles = {
        "technical_pause": ("#9CA3AF", "技术门暂停新买单"),
        "pair_stop": ("#D97706", "单对停止"),
        "portfolio_stop": ("#B91C1C", "组合全部停止"),
    }
    shown: set[str] = set()
    for scope in scope_order:
        frame = stops[stops.scope == scope]
        for item in frame.itertuples(index=False):
            color, legend = styles[item.kind]
            fig.add_trace(go.Scatter(
                x=[item.start, item.end], y=[scope, scope], mode="lines",
                line={"color": color, "width": 11},
                name=legend, legendgroup=item.kind, showlegend=item.kind not in shown,
                customdata=[[item.end], [item.end]],
                hovertemplate=(
                    "%{y}<br>开始：%{x|%Y-%m-%d %H:%M UTC}"
                    "<br>结束：%{customdata[0]|%Y-%m-%d %H:%M UTC}<extra></extra>"
                ),
            ), row=row, col=1)
            shown.add(item.kind)
    fig.update_yaxes(categoryorder="array", categoryarray=list(reversed(scope_order)), row=row, col=1)


def build_figure(weekly, equity, pairs, stops, summary, start_ts, end_ts) -> go.Figure:
    fig = make_subplots(
        rows=5, cols=2,
        specs=[
            [{"colspan": 2}, None], [{"colspan": 2}, None],
            [{}, {}], [{}, {}], [{"type": "table", "colspan": 2}, None],
        ],
        row_heights=[0.25, 0.16, 0.18, 0.16, 0.25],
        vertical_spacing=0.06, horizontal_spacing=0.10,
        subplot_titles=(
            "累计样本外盈亏", "停止交易与暂停买入时间区域",
            "逐周样本外盈亏", "逐周最大回撤",
            "风险事件与参数部署门槛", "BTC / ETH 盈亏贡献", "机制矩阵",
        ),
    )
    dash = {ONLINE: "solid", NEW: "dash"}
    symbol = {ONLINE: "circle", NEW: "diamond"}
    for label, color in PALETTE.items():
        curve = equity[equity.scenario == label]
        fig.add_trace(go.Scatter(
            x=curve.datetime, y=curve.cumulative_oos_pnl, name=label,
            mode="lines", line={"color": color, "width": 2.3, "dash": dash[label]},
            hovertemplate="%{x|%Y-%m-%d}<br>%{y:+.2f} FDUSD<extra>" + label + "</extra>",
        ), row=1, col=1)
    add_stop_timeline(fig, stops, 2)

    for label, color in PALETTE.items():
        frame = weekly[weekly.scenario == label]
        weeks = [f"W{int(value):02d}" for value in frame.fold]
        fig.add_trace(go.Bar(
            x=weeks, y=frame.net_pnl_quote, marker_color=color,
            name=label, legendgroup=label, showlegend=False,
            hovertemplate="%{x}<br>%{y:+.2f} FDUSD<extra>" + label + "</extra>",
        ), row=3, col=1)
        fig.add_trace(go.Scatter(
            x=weeks, y=frame.max_drawdown_pct * 100, mode="lines+markers",
            marker={"symbol": symbol[label], "size": 6},
            line={"color": color, "width": 1.8, "dash": dash[label]},
            name=label, legendgroup=label, showlegend=False,
            hovertemplate="%{x}<br>%{y:.2f}%<extra>" + label + "</extra>",
        ), row=3, col=2)

    risk = summary.melt(
        id_vars="scenario",
        value_vars=["pair_stop_folds", "portfolio_stop_folds", "retained_parameter_folds"],
        var_name="metric", value_name="folds",
    )
    risk_names = {
        "pair_stop_folds": "单对停止周", "portfolio_stop_folds": "组合停止周",
        "retained_parameter_folds": "无合格参数·维持旧参数周",
    }
    for label, color in PALETTE.items():
        frame = risk[risk.scenario == label]
        fig.add_trace(go.Bar(
            x=[risk_names[value] for value in frame.metric], y=frame.folds,
            marker_color=color, name=label, legendgroup=label, showlegend=False,
            text=frame.folds, textposition="outside",
        ), row=4, col=1)
    pair_totals = pairs.groupby(["scenario", "pair"], as_index=False).net_pnl_quote.sum()
    for label, color in PALETTE.items():
        frame = pair_totals[pair_totals.scenario == label]
        fig.add_trace(go.Bar(
            x=frame.pair.str.replace("-FDUSD", "", regex=False), y=frame.net_pnl_quote,
            marker_color=color, name=label, legendgroup=label, showlegend=False,
            text=[f"{value:+.2f}" for value in frame.net_pnl_quote], textposition="outside",
            hovertemplate="%{x}<br>%{y:+.2f} FDUSD<extra>" + label + "</extra>",
        ), row=4, col=2)

    mechanisms = [
        ("资金隔离", "420 FDUSD；每对 200；预留 20", "相同"),
        ("周度参数", "30/14 日训练；无合格参数维持旧参数", "相同；用新机制重新评估候选"),
        ("网格移动", "动态网格；30 分钟移动冷却", "相同"),
        ("挂单生命周期", "2 小时刷新", "相同"),
        ("费用模型", "Maker 0%；风险退出 Taker 0.1%", "相同"),
        ("普通卖出", "网格/现价止盈/移动平均成本底线取最高", "相同"),
        ("技术买入门", "4h ROC + SQZMOM；risk-off 仅暂停新买单", "相同；暂停区已标出"),
        ("宏观门", "本次回测关闭", "本次回测关闭"),
        ("单对风险", "亏损 -6 FDUSD 或峰值回撤 3% 即停止", "相同"),
        ("组合风险", "亏损 -24 FDUSD 或峰值回撤 6% 即全部停止", "相同"),
        ("额外库存上限", "仅受账户预算约束", "每对 25 FDUSD"),
        ("最长持有", "无限制", "24 小时"),
        ("分级成本底线", "始终保持选定止盈率", "0–8h 原止盈；8–16h +0.2%；16–24h 保本"),
        ("超时退出", "无", "仅将超出初始库存部分按 Taker 卖出"),
    ]
    fig.add_trace(go.Table(
        header={"values": ["机制", ONLINE, NEW], "fill_color": "#E5E7EB",
                "font": {"color": "#111827", "size": 12}, "align": "left", "height": 28},
        cells={"values": list(zip(*mechanisms)),
               "fill_color": [["#FFFFFF", "#F9FAFB"] * 7] * 3,
               "font": {"color": "#374151", "size": 11}, "align": "left", "height": 26},
        columnwidth=[0.18, 0.41, 0.41],
    ), row=5, col=1)

    start = pd.to_datetime(start_ts, unit="s", utc=True).strftime("%Y-%m-%d")
    end = pd.to_datetime(end_ts, unit="s", utc=True).strftime("%Y-%m-%d")
    fig.update_layout(
        title={"text": (
            "FDUSD Grid：全部线上机制与新库存机制对比"
            f"<br><sup>{start} 至 {end} UTC｜21 个周度样本外窗口｜宏观门关闭｜BTC/ETH-FDUSD｜420 FDUSD</sup>"
        ), "x": 0.02, "xanchor": "left"},
        template="plotly_white", height=2050, barmode="group",
        margin={"l": 85, "r": 40, "t": 125, "b": 40},
        font={"family": "Arial, Microsoft YaHei, sans-serif", "color": "#1F2937"},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.01, "x": 0.55},
        hovermode="closest",
    )
    fig.update_yaxes(title_text="FDUSD", zeroline=True, zerolinecolor="#6B7280", row=1, col=1)
    fig.update_yaxes(title_text="FDUSD", zeroline=True, zerolinecolor="#6B7280", row=3, col=1)
    fig.update_yaxes(title_text="回撤 %", range=[-6.6, 0.3], row=3, col=2)
    fig.add_shape(type="line", xref="x4 domain", yref="y4", x0=0, x1=1, y0=-6, y1=-6,
                  line={"color": "#6B7280", "width": 1.2, "dash": "dash"})
    fig.add_annotation(xref="x4 domain", yref="y4", x=0.99, y=-6,
                       text="组合回撤门槛 -6%", showarrow=False, xanchor="right",
                       yanchor="bottom", font={"size": 10, "color": "#6B7280"})
    fig.update_yaxes(title_text="周数（共 21 周）", rangemode="tozero", row=4, col=1)
    fig.update_yaxes(title_text="FDUSD", zeroline=True, zerolinecolor="#6B7280", row=4, col=2)
    return fig


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, default=Path("data/backtesting_candles"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("results/backtests/fdusd_inventory_exit_comparison_plotly"))
    parser.add_argument("--days", type=int, default=180)
    args = parser.parse_args()
    candles = {pair: read_cache(args.cache_dir / f"binance_{pair}_5m.csv")
               for pair in ("BTC-FDUSD", "ETH-FDUSD")}
    if any(frame.empty for frame in candles.values()):
        raise RuntimeError("BTC/ETH FDUSD candle cache is required.")
    end_ts = min(int(frame.timestamp.max()) for frame in candles.values()) + 300
    start_ts = end_ts - args.days * 86400
    if any(int(frame.timestamp.min()) > start_ts for frame in candles.values()):
        raise RuntimeError("Candle cache does not cover the requested comparison period.")

    weekly, equity, pairs, stops, policy = run_comparison(candles, start_ts, end_ts)
    summary = summarize(weekly, pairs)
    figure = build_figure(weekly, equity, pairs, stops, summary, start_ts, end_ts)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    weekly.to_csv(args.output_dir / "weekly_comparison.csv", index=False)
    equity.to_csv(args.output_dir / "equity_comparison.csv", index=False)
    pairs.to_csv(args.output_dir / "pair_comparison.csv", index=False)
    stops.to_csv(args.output_dir / "stopped_trading_regions.csv", index=False)
    payload = {
        "range": {"start_ts": start_ts, "end_ts": end_ts, "days": args.days},
        "macro_gate_enabled": False,
        "initial_approved_parameters": asdict(INITIAL_APPROVED),
        "inventory_exit_policy": asdict(policy),
        "summary": summary.to_dict("records"),
        "limitations": [
            "The macro gate is intentionally disabled for both scenarios.",
            "The simulation uses 5-minute candles and limit-touch fills; queue position and partial fills are not modeled.",
            "Risk-stop regions run from the breaker timestamp to the end of that weekly fold; the next fold starts a fresh simulation.",
        ],
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    figure.write_html(args.output_dir / "fdusd_live_vs_inventory_exit.html",
                      include_plotlyjs=True, full_html=True,
                      config={"displaylogo": False, "responsive": True})
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

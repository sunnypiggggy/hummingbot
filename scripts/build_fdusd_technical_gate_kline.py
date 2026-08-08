#!/usr/bin/env python3
"""Render holdout candlesticks with technical BUY pauses and risk-stop regions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from validate_grid_live import read_cache


PANELS = (
    ("online", "BTC-FDUSD", "线上模型 · BTC-FDUSD"),
    ("new", "BTC-FDUSD", "新机制 · BTC-FDUSD"),
    ("online", "ETH-FDUSD", "线上模型 · ETH-FDUSD"),
    ("new", "ETH-FDUSD", "新机制 · ETH-FDUSD"),
)


def count_trades_in_regions(trades: pd.DataFrame, regions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scenario in ("online", "new"):
        subset = trades[(trades.period == "holdout") & (trades.scenario == scenario)]
        inside = pd.Series(False, index=subset.index)
        for region in regions.itertuples(index=False):
            inside |= subset.timestamp.between(int(region.start_ts), int(region.end_ts), inclusive="left")
        paused = subset[inside]
        grid_buys = paused[(paused.side == "BUY") & (paused.reason == "grid_fill")]
        risk_restores = paused[
            (paused.side == "BUY") & (paused.reason == "pair_breaker_flatten")
        ]
        rows.append({
            "scenario": scenario,
            "technical_pause_regions": len(regions),
            "technical_pause_hours": float(((regions.end_ts - regions.start_ts) / 3600).sum()),
            "grid_buys_during_pause": len(grid_buys),
            "risk_restore_buys_during_pause": len(risk_restores),
            "sells_during_pause": int((paused.side == "SELL").sum()),
            "all_trades_during_pause": len(paused),
        })
    return pd.DataFrame(rows)


def _add_trade_markers(fig: go.Figure, trades: pd.DataFrame, row: int) -> None:
    styles = {
        ("grid_fill", "BUY"): ("普通网格买入", "triangle-up", "#0891B2", 8),
        ("grid_fill", "SELL"): ("普通网格卖出", "triangle-down", "#7C3AED", 8),
        ("max_hold_exit", "SELL"): ("最长持有超时退出", "x", "#111827", 11),
        ("pair_breaker_flatten", "BUY"): ("单对风控恢复基准库存", "x", "#DC2626", 11),
        ("pair_breaker_flatten", "SELL"): ("单对风控卖出额外库存", "x", "#DC2626", 11),
    }
    for (reason, side), frame in trades.groupby(["reason", "side"], sort=False):
        name, symbol, color, size = styles[(reason, side)]
        fig.add_trace(go.Scatter(
            x=pd.to_datetime(frame.timestamp, unit="s", utc=True), y=frame.price,
            mode="markers", name=name, legendgroup=f"{reason}-{side}",
            showlegend=row == 1,
            marker={"symbol": symbol, "color": color, "size": size,
                    "line": {"color": "#FFFFFF", "width": 0.6}},
            customdata=frame[["amount", "quote_notional", "reason"]],
            hovertemplate=(
                "%{x|%Y-%m-%d %H:%M UTC}<br>价格 %{y:.6g}"
                "<br>数量 %{customdata[0]:.8g}<br>金额 %{customdata[1]:.2f} FDUSD"
                "<br>%{customdata[2]}<extra></extra>"
            ),
        ), row=row, col=1)


def build_kline_figure(candles: dict[str, pd.DataFrame], trades: pd.DataFrame,
                       stops: pd.DataFrame, start_ts: int, end_ts: int) -> go.Figure:
    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.035,
        subplot_titles=[title for _, _, title in PANELS],
    )
    technical = stops[stops.kind == "technical_pause"]
    region_styles = {
        "technical_pause": ("#2563EB", 0.20, "dot"),
        "pair_stop": ("#F59E0B", 0.24, "dash"),
        "portfolio_stop": ("#DC2626", 0.26, "solid"),
    }
    for row, (scenario, pair, _) in enumerate(PANELS, 1):
        frame = candles[pair]
        fig.add_trace(go.Candlestick(
            x=pd.to_datetime(frame.timestamp, unit="s", utc=True),
            open=frame.open, high=frame.high, low=frame.low, close=frame.close,
            name=f"{scenario}-{pair}", legendgroup=f"candle-{scenario}-{pair}",
            showlegend=False,
            increasing={"line": {"color": "#334155", "width": 0.8},
                        "fillcolor": "#CBD5E1"},
            decreasing={"line": {"color": "#64748B", "width": 0.8},
                        "fillcolor": "#F8FAFC"},
            hovertext=pair,
        ), row=row, col=1)
        scenario_prefix = "线上：" if scenario == "online" else "新机制："
        pair_scope = pair.replace("-FDUSD", "")
        relevant = pd.concat([
            technical,
            stops[(stops.scope == f"{scenario_prefix}{pair_scope}")
                  | (stops.scope == f"{scenario_prefix}组合")],
        ]).drop_duplicates(["scope", "kind", "start_ts", "end_ts"])
        for region in relevant.itertuples(index=False):
            color, opacity, dash = region_styles[region.kind]
            fig.add_vrect(
                x0=region.start, x1=region.end, fillcolor=color, opacity=opacity,
                line={"color": color, "width": 1.25, "dash": dash},
                layer="below", row=row, col=1,
            )
        panel_trades = trades[
            (trades.period == "holdout") & (trades.scenario == scenario) & (trades.pair == pair)
        ]
        _add_trade_markers(fig, panel_trades, row)
        fig.update_yaxes(title_text=pair.split("-")[0] + " 价格", row=row, col=1)
        fig.update_xaxes(rangeslider_visible=False, row=row, col=1)

    pause_hours = float(((technical.end_ts - technical.start_ts) / 3600).sum())
    fig.update_layout(
        title={
            "text": (
                "FDUSD Grid：技术门暂停买入 K线观察"
                f"<br><sup>{pd.to_datetime(start_ts, unit='s', utc=True):%Y-%m-%d} 至 "
                f"{pd.to_datetime(end_ts, unit='s', utc=True):%Y-%m-%d} UTC｜5分钟K线｜"
                f"蓝色点线=技术门暂停普通买入（{len(technical)}段、共{pause_hours:.0f}小时）｜"
                "橙色虚线=单对停止｜红色实线=组合停止</sup>"
            ),
            "x": 0.02, "xanchor": "left",
        },
        template="plotly_white", height=1800,
        margin={"l": 85, "r": 40, "t": 225, "b": 55},
        font={"family": "Arial, Microsoft YaHei, sans-serif", "color": "#1F2937"},
        legend={"orientation": "h", "y": 1.015, "x": 0.02,
                "bgcolor": "rgba(255,255,255,0.88)", "bordercolor": "#CBD5E1",
                "borderwidth": 1},
        hovermode="x unified",
    )
    fig.update_xaxes(title_text="UTC时间", row=4, col=1)
    return fig


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, default=Path("data/backtesting_candles"))
    parser.add_argument("--result-dir", type=Path,
                        default=Path("results/backtests/fdusd_inventory_exit_parameter_search"))
    args = parser.parse_args()
    summary = json.loads((args.result_dir / "summary.json").read_text(encoding="utf-8"))
    start_ts = int(summary["range"]["holdout_start"])
    weekly = pd.read_csv(args.result_dir / "weekly_results.csv")
    end_ts = int(weekly[weekly.period == "holdout"].test_end.max())
    candles = {
        pair: read_cache(args.cache_dir / f"binance_{pair}_5m.csv").query(
            "timestamp >= @start_ts and timestamp < @end_ts"
        ).reset_index(drop=True)
        for pair in ("BTC-FDUSD", "ETH-FDUSD")
    }
    trades = pd.read_csv(args.result_dir / "trades.csv")
    stops = pd.read_csv(args.result_dir / "stopped_trading_regions.csv", parse_dates=["start", "end"])
    technical = stops[stops.kind == "technical_pause"].copy()
    audit = count_trades_in_regions(trades, technical)
    if bool((audit.grid_buys_during_pause != 0).any()):
        raise RuntimeError("Technical BUY gate audit failed: a new Grid BUY occurred during a pause.")
    audit.to_csv(args.result_dir / "technical_gate_kline_audit.csv", index=False)
    figure = build_kline_figure(candles, trades, stops, start_ts, end_ts)
    figure.write_html(
        args.result_dir / "technical_gate_kline_observation.html",
        include_plotlyjs=True, full_html=True,
        config={"displaylogo": False, "responsive": True, "scrollZoom": True},
    )
    print(audit.to_json(orient="records", force_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

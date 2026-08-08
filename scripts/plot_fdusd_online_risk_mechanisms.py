#!/usr/bin/env python3
"""Build one self-contained Plotly chart per FDUSD online Grid risk mechanism."""

from __future__ import annotations

import argparse
import html
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

from grid_live_common import (
    FDUSD_BUDGET,
    PAIR_DRAWDOWN_LIMIT_PCT,
    PORTFOLIO_DRAWDOWN_LIMIT_PCT,
)
from validate_grid_live import (
    Candidate,
    read_cache,
    simulate,
    technical_buy_gate_timeline,
)


PAIRS = ("BTC-FDUSD", "ETH-FDUSD")
COLORS = {"BTC-FDUSD": "#0891B2", "ETH-FDUSD": "#7C3AED"}
TRIGGER_COLOR = "#DC2626"
THRESHOLD_COLOR = "#111827"
PAUSE_COLOR = "#2563EB"


def _candidate(row) -> Candidate:
    return Candidate(
        float(row.half_range), float(row.min_spread), float(row.take_profit),
        float(row.move_threshold), int(row.move_cooldown_seconds),
    )


def replay_online(candles: dict[str, pd.DataFrame], selections: pd.DataFrame):
    gate = technical_buy_gate_timeline(candles["BTC-FDUSD"])
    curves, trades = [], []
    for row in selections.itertuples(index=False):
        start, end = int(row.test_start), int(row.test_end)
        window = {
            pair: frame[(frame.timestamp >= start) & (frame.timestamp < end)].reset_index(drop=True)
            for pair, frame in candles.items()
        }
        trade_log: list[dict] = []
        _, curve, _ = simulate(
            window, _candidate(row), maker_fee=0.0, taker_fee=0.001,
            order_refresh_seconds=7200, technical_buy_gate=gate,
            trade_log=trade_log, risk_breakers_enabled=True,
            cost_floor_enabled=True, inventory_exit_policy=None,
        )
        if not curve.empty:
            curve = curve.copy()
            curve["period"] = row.period
            curve["fold"] = int(row.fold)
            curve["test_start"] = start
            curve["test_end"] = end
            curves.append(curve)
        trades.extend({"period": row.period, "fold": int(row.fold), "test_end": end, **item}
                      for item in trade_log)
    return pd.concat(curves, ignore_index=True), pd.DataFrame(trades), gate


def _base_figure(title: str, subtitle: str, explanation: str, y_title: str) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        title={"text": f"{title}<br><sup>{subtitle}</sup>", "x": 0.02, "xanchor": "left",
               "y": 0.985, "yanchor": "top"},
        template="plotly_white", height=760,
        margin={"l": 90, "r": 45, "t": 225, "b": 70},
        font={"family": "Arial, Microsoft YaHei, sans-serif", "color": "#1F2937"},
        legend={"orientation": "h", "x": 0.02, "y": 1.04,
                "bgcolor": "rgba(255,255,255,0.9)", "bordercolor": "#CBD5E1",
                "borderwidth": 1},
        hovermode="x unified",
        annotations=[{
            "xref": "paper", "yref": "paper", "x": 0.01, "y": 1.28,
            "xanchor": "left", "yanchor": "top", "align": "left", "showarrow": False,
            "text": explanation, "font": {"size": 13, "color": "#334155"},
            "bgcolor": "#F8FAFC", "bordercolor": "#94A3B8", "borderwidth": 1,
            "borderpad": 10,
        }],
    )
    fig.update_xaxes(title_text="UTC时间", showgrid=True, gridcolor="#E2E8F0")
    fig.update_yaxes(title_text=y_title, showgrid=True, gridcolor="#E2E8F0")
    return fig


def _add_segmented_line(fig: go.Figure, curve: pd.DataFrame, column: str, pair: str,
                        name: str) -> None:
    first = True
    for _, fold in curve.groupby(["period", "fold"], sort=False):
        fig.add_trace(go.Scatter(
            x=pd.to_datetime(fold.timestamp, unit="s", utc=True), y=fold[column],
            mode="lines", name=name, legendgroup=name, showlegend=first,
            line={"color": COLORS[pair], "width": 1.35},
            hovertemplate="%{x|%Y-%m-%d %H:%M UTC}<br>%{y:.3f}<extra></extra>",
        ))
        first = False


def _trigger_markers(fig: go.Figure, trades: pd.DataFrame, trigger: str, y_field: str,
                     name: str) -> int:
    events = trades[trades.get("trigger", pd.Series(index=trades.index, dtype=str)) == trigger]
    if events.empty:
        return 0
    fig.add_trace(go.Scatter(
        x=pd.to_datetime(events.timestamp, unit="s", utc=True), y=events[y_field],
        mode="markers", name=name,
        marker={"symbol": "x", "size": 12, "color": TRIGGER_COLOR,
                "line": {"color": "#7F1D1D", "width": 1}},
        customdata=events[["pair", "trigger_pnl_quote", "trigger_drawdown_pct"]],
        hovertemplate=("%{x|%Y-%m-%d %H:%M UTC}<br>%{customdata[0]}"
                       "<br>盈亏 %{customdata[1]:.2f} FDUSD"
                       "<br>回撤 %{customdata[2]:.2f}%<extra></extra>"),
    ))
    return len(events)


def _add_stop_regions(fig: go.Figure, trades: pd.DataFrame, trigger: str) -> None:
    events = trades[trades.get("trigger", pd.Series(index=trades.index, dtype=str)) == trigger]
    for event in events.itertuples(index=False):
        fig.add_vrect(
            x0=pd.to_datetime(event.timestamp, unit="s", utc=True),
            x1=pd.to_datetime(event.test_end, unit="s", utc=True),
            fillcolor=TRIGGER_COLOR, opacity=0.10,
            line={"color": TRIGGER_COLOR, "width": 1, "dash": "dot"}, layer="below",
        )


def technical_chart(candles, selections, gate) -> go.Figure:
    start, end = int(selections.test_start.min()), int(selections.test_end.max())
    explanation = (
        "<b>机制：</b>BTC 4小时K线 ROC48≤-5% 且 SQZMOM≤-1% 时暂停BTC/ETH普通网格买入；"
        "ROC48≥1%、SQZMOM≥-3%且动量改善时恢复。<br>"
        "<b>动作：</b>只撤普通买单，卖单及单对风控恢复基准库存仍可执行；线上信号失效时fail-closed。"
    )
    fig = _base_figure(
        "线上 Grid 风控 1：ROC/SQZMOM 技术买入门",
        "价格指数以观察期首=100；蓝色背景为禁止普通网格买入", explanation, "价格指数",
    )
    for pair in PAIRS:
        frame = candles[pair][(candles[pair].timestamp >= start) & (candles[pair].timestamp < end)]
        indexed = frame.close / float(frame.close.iloc[0]) * 100
        fig.add_trace(go.Scatter(
            x=pd.to_datetime(frame.timestamp, unit="s", utc=True), y=indexed,
            mode="lines", name=pair, line={"color": COLORS[pair], "width": 1.25},
        ))
    timestamps = sorted(ts for ts in gate if start <= int(ts) < end)
    region_start = None
    regions = []
    previous = None
    for ts in timestamps:
        paused = not bool(gate[ts])
        if paused and region_start is None:
            region_start = ts
        if not paused and region_start is not None:
            regions.append((region_start, ts))
            region_start = None
        previous = ts
    if region_start is not None and previous is not None:
        regions.append((region_start, min(previous + 300, end)))
    for left, right in regions:
        fig.add_vrect(
            x0=pd.to_datetime(left, unit="s", utc=True),
            x1=pd.to_datetime(right, unit="s", utc=True),
            fillcolor=PAUSE_COLOR, opacity=0.16,
            line={"color": PAUSE_COLOR, "width": 1, "dash": "dot"}, layer="below",
        )
    fig.add_annotation(
        xref="paper", yref="paper", x=0.99, y=0.02, showarrow=False, xanchor="right",
        text=f"技术暂停 {len(regions)} 段，共 {sum(b-a for a,b in regions)/3600:.0f} 小时",
        bgcolor="#DBEAFE", bordercolor=PAUSE_COLOR, borderwidth=1,
    )
    return fig


def pair_risk_chart(curve, trades, *, trigger: str, column_suffix: str, title: str,
                    subtitle: str, explanation: str, threshold: float, y_title: str) -> go.Figure:
    fig = _base_figure(title, subtitle, explanation, y_title)
    for pair in PAIRS:
        _add_segmented_line(fig, curve, f"{pair}_{column_suffix}", pair, pair)
    fig.add_hline(y=threshold, line={"color": THRESHOLD_COLOR, "width": 1.5, "dash": "dash"},
                  annotation_text=f"触发阈值 {threshold:g}", annotation_position="top left")
    count = _trigger_markers(
        fig, trades, trigger,
        "trigger_pnl_quote" if trigger == "pair_loss" else "trigger_drawdown_pct",
        "实际触发点",
    )
    _add_stop_regions(fig, trades, trigger)
    fig.add_annotation(xref="paper", yref="paper", x=0.99, y=0.02, xanchor="right",
                       showarrow=False, text=f"实际触发 {count} 次",
                       bgcolor="#FEE2E2", bordercolor=TRIGGER_COLOR, borderwidth=1)
    return fig


def portfolio_risk_chart(curve, trades, *, trigger: str, column: str, title: str,
                         subtitle: str, explanation: str, threshold: float,
                         y_title: str) -> go.Figure:
    fig = _base_figure(title, subtitle, explanation, y_title)
    first = True
    for _, fold in curve.groupby(["period", "fold"], sort=False):
        y = fold[column] * 100 if column == "drawdown_pct" else fold[column]
        fig.add_trace(go.Scatter(
            x=pd.to_datetime(fold.timestamp, unit="s", utc=True), y=y,
            mode="lines", name="BTC+ETH组合", legendgroup="portfolio", showlegend=first,
            line={"color": "#0F766E", "width": 1.4},
        ))
        first = False
    fig.add_hline(y=threshold, line={"color": THRESHOLD_COLOR, "width": 1.5, "dash": "dash"},
                  annotation_text=f"触发阈值 {threshold:g}", annotation_position="top left")
    count = _trigger_markers(
        fig, trades, trigger,
        "trigger_pnl_quote" if trigger == "portfolio_loss" else "trigger_drawdown_pct",
        "实际触发点",
    )
    _add_stop_regions(fig, trades, trigger)
    fig.add_annotation(xref="paper", yref="paper", x=0.99, y=0.02, xanchor="right",
                       showarrow=False, text=f"实际触发 {count} 次",
                       bgcolor="#FEE2E2", bordercolor=TRIGGER_COLOR, borderwidth=1)
    return fig


def cost_floor_chart(trades: pd.DataFrame) -> go.Figure:
    explanation = (
        "<b>机制：</b>普通卖单价格取 max(网格档位、当前价×止盈率、移动平均持仓成本×止盈率)。<br>"
        "<b>动作：</b>Maker成交价不得低于成本利润底线；FDUSD Maker按0%，周度止盈参数为0.6%/0.8%/1.0%。"
    )
    fig = _base_figure(
        "线上 Grid 风控 6：移动平均成本卖出底线",
        "仅展示实际成交的普通网格卖单；纵轴为相对移动平均成本的收益率", explanation,
        "相对平均成本收益率（%）",
    )
    sells = trades[(trades.reason == "grid_fill") & (trades.side == "SELL")].copy()
    sells = sells.dropna(subset=["average_cost", "cost_floor"])
    sells["actual_pct"] = (sells.price / sells.average_cost - 1) * 100
    sells["floor_pct"] = (sells.cost_floor / sells.average_cost - 1) * 100
    for pair in PAIRS:
        frame = sells[sells.pair == pair]
        fig.add_trace(go.Scatter(
            x=pd.to_datetime(frame.timestamp, unit="s", utc=True), y=frame.actual_pct,
            mode="markers", name=f"{pair}实际成交",
            marker={"color": COLORS[pair], "size": 7, "symbol": "circle"},
            customdata=frame[["price", "average_cost", "floor_pct"]],
            hovertemplate=("%{x|%Y-%m-%d %H:%M UTC}<br>实际收益 %{y:.3f}%"
                           "<br>成交价 %{customdata[0]:.6g}<br>平均成本 %{customdata[1]:.6g}"
                           "<br>成本底线收益 %{customdata[2]:.3f}%<extra></extra>"),
        ))
        fig.add_trace(go.Scatter(
            x=pd.to_datetime(frame.timestamp, unit="s", utc=True), y=frame.floor_pct,
            mode="markers", name=f"{pair}成本底线",
            marker={"color": COLORS[pair], "size": 6, "symbol": "line-ew-open"},
        ))
    violations = int((sells.actual_pct + 1e-9 < sells.floor_pct).sum())
    fig.add_hline(y=0, line={"color": THRESHOLD_COLOR, "width": 1, "dash": "dash"})
    fig.add_annotation(
        xref="paper", yref="paper", x=0.99, y=0.02, xanchor="right", showarrow=False,
        text=f"成交 {len(sells)} 笔；低于成本底线 {violations} 笔",
        bgcolor="#ECFDF5" if violations == 0 else "#FEE2E2",
        bordercolor="#0F766E" if violations == 0 else TRIGGER_COLOR, borderwidth=1,
    )
    return fig


def _write_figure(fig: go.Figure, path: Path) -> None:
    fig.write_html(path, include_plotlyjs=True, full_html=True,
                   config={"displaylogo": False, "responsive": True, "scrollZoom": True})


def write_index(output_dir: Path, charts: list[tuple[str, str]], curves: pd.DataFrame,
                trades: pd.DataFrame) -> None:
    cards = "\n".join(
        f'<li><a href="{html.escape(filename)}">{html.escape(title)}</a></li>'
        for filename, title in charts
    )
    content = f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<title>线上FDUSD Grid风控图</title><style>
body{{font-family:Arial,'Microsoft YaHei',sans-serif;background:#f8fafc;color:#1f2937;margin:32px}}
main{{max-width:980px;margin:auto}}section{{background:white;border:1px solid #cbd5e1;padding:22px;border-radius:8px}}
li{{margin:14px 0}}a{{color:#1d4ed8;font-size:17px}}code{{background:#e2e8f0;padding:2px 5px}}
</style><main><h1>线上 FDUSD Grid：逐项风控图</h1><section>
<p>口径：仅线上 Grid，BTC-FDUSD + ETH-FDUSD，Maker 0%、Taker 0.1%、2小时挂单、周度参数。每项机制一张自包含Plotly图。</p>
<ol>{cards}</ol>
<p>宏观/FOMC门未纳入：当前回测没有完整历史审批状态，避免制造不存在的事件区间。</p>
<p>轨迹点 {len(curves):,}；成交/风控事件 {len(trades):,}。</p>
</section></main></html>"""
    (output_dir / "index.html").write_text(content, encoding="utf-8")


def write_combined_index(output_dir: Path, charts: list[tuple[str, go.Figure]],
                         curves: pd.DataFrame, trades: pd.DataFrame) -> None:
    """Write all risk charts into one self-contained, navigable HTML page."""
    nav_items: list[str] = []
    chart_sections: list[str] = []
    plot_config = {"displaylogo": False, "responsive": True, "scrollZoom": True}
    for index, (filename, figure) in enumerate(charts, start=1):
        title = str(figure.layout.title.text).split("<br>", 1)[0]
        anchor = f"risk-{index}"
        nav_items.append(f'<li><a href="#{anchor}">{title}</a></li>')
        chart_html = figure.to_html(
            full_html=False,
            include_plotlyjs=True if index == 1 else False,
            config=plot_config,
        )
        chart_sections.append(
            f'<section class="chart" id="{anchor}">{chart_html}'
            f'<p class="single"><a href="{html.escape(filename)}">单独打开这张图</a></p></section>'
        )
    content = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>线上 FDUSD Grid 风控机制总览</title><style>
html{{scroll-behavior:smooth}}body{{font-family:Arial,'Microsoft YaHei',sans-serif;background:#f8fafc;color:#1f2937;margin:0}}
main{{max-width:1500px;margin:auto;padding:28px 22px 64px}}header,.chart{{background:white;border:1px solid #cbd5e1;border-radius:10px}}
header{{padding:22px 28px;margin-bottom:24px}}h1{{margin:0 0 12px}}p{{line-height:1.65}}ol{{columns:2;column-gap:40px;padding-left:24px}}
li{{margin:10px 0;break-inside:avoid}}a{{color:#1d4ed8}}.chart{{padding:8px 12px 14px;margin:24px 0;scroll-margin-top:16px}}
.single{{text-align:right;margin:0 12px 4px;font-size:14px}}@media(max-width:760px){{main{{padding:14px 8px 36px}}ol{{columns:1}}.chart{{padding:2px}}}}
</style></head><body><main><header><h1>线上 FDUSD Grid：全部风控机制</h1>
<p>口径：BTC-FDUSD + ETH-FDUSD，Maker 0%、Taker 0.1%、2小时挂单、周度参数。每张图均包含机制解释、阈值、触发动作及停止交易区域。</p>
<ol>{''.join(nav_items)}</ol>
<p>宏观/FOMC 门未纳入：当前回测没有完整历史审批状态，避免制造不存在的事件区间。轨迹点 {len(curves):,}；成交及风控记录 {len(trades):,}。</p>
</header>{''.join(chart_sections)}</main></body></html>"""
    (output_dir / "index.html").write_text(content, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, default=Path("data/backtesting_candles"))
    parser.add_argument(
        "--weekly-results", type=Path,
        default=Path("results/backtests/fdusd_inventory_exit_parameter_search/weekly_results.csv"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("results/backtests/fdusd_online_risk_mechanisms"),
    )
    args = parser.parse_args()
    selections = pd.read_csv(args.weekly_results)
    selections = selections[selections.scenario == "online"].sort_values(["test_start", "fold"])
    if selections.empty:
        raise RuntimeError("No online weekly selections were found.")
    candles = {
        pair: read_cache(args.cache_dir / f"binance_{pair}_5m.csv") for pair in PAIRS
    }
    curves, trades, gate = replay_online(candles, selections)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    charts = [
        ("01_technical_buy_gate.html", technical_chart(candles, selections, gate)),
        ("02_pair_absolute_loss.html", pair_risk_chart(
            curves, trades, trigger="pair_loss", column_suffix="pnl_quote",
            title="线上 Grid 风控 2：单对绝对亏损停止",
            subtitle="BTC/ETH每周独立按200 FDUSD重新计量；红色背景为停止至本周结束",
            explanation=("<b>机制：</b>任一交易对权益相对200 FDUSD预算亏损达到6 FDUSD。<br>"
                         "<b>动作：</b>用Taker恢复到启动基准库存，撤单并停止该交易对至本周结束；另一交易对可继续。"),
            threshold=-float(FDUSD_BUDGET.pair_loss_limit), y_title="单对盈亏（FDUSD）",
        )),
        ("03_pair_peak_drawdown.html", pair_risk_chart(
            curves, trades, trigger="pair_drawdown", column_suffix="drawdown_pct",
            title="线上 Grid 风控 3：单对峰值回撤停止",
            subtitle="回撤从各交易对当周权益峰值计算；红色背景为停止至本周结束",
            explanation=("<b>机制：</b>任一交易对从当周权益最高点回撤达到3%。<br>"
                         "<b>动作：</b>用Taker恢复到启动基准库存，撤单并停止该交易对至本周结束；下周重新计量。"),
            threshold=-float(PAIR_DRAWDOWN_LIMIT_PCT) * 100, y_title="单对峰值回撤（%）",
        )),
        ("04_portfolio_absolute_loss.html", portfolio_risk_chart(
            curves, trades, trigger="portfolio_loss", column="portfolio_pnl_quote",
            title="线上 Grid 风控 4：组合绝对亏损停止",
            subtitle="BTC+ETH+20 FDUSD储备按420 FDUSD每周重新计量；红色背景为组合停止",
            explanation=("<b>机制：</b>BTC与ETH组合权益相对420 FDUSD亏损达到24 FDUSD。<br>"
                         "<b>动作：</b>立即停止两个交易对至本周结束；组合停止本身不强制卖出全部基础仓位。"),
            threshold=-float(FDUSD_BUDGET.portfolio_loss_limit), y_title="组合盈亏（FDUSD）",
        )),
        ("05_portfolio_peak_drawdown.html", portfolio_risk_chart(
            curves, trades, trigger="portfolio_drawdown", column="drawdown_pct",
            title="线上 Grid 风控 5：组合峰值回撤停止",
            subtitle="BTC+ETH组合从当周权益峰值计算；红色背景为组合停止",
            explanation=("<b>机制：</b>BTC与ETH组合权益从当周最高点回撤达到6%。<br>"
                         "<b>动作：</b>立即停止两个交易对至本周结束；下周重新建立风险基准。"),
            threshold=-float(PORTFOLIO_DRAWDOWN_LIMIT_PCT) * 100, y_title="组合峰值回撤（%）",
        )),
        ("06_cost_floor.html", cost_floor_chart(trades)),
    ]
    index_entries = []
    for filename, figure in charts:
        _write_figure(figure, args.output_dir / filename)
        index_entries.append((filename, str(figure.layout.title.text).split("<br>", 1)[0]))
    curves.to_csv(args.output_dir / "online_risk_curve.csv", index=False)
    trades.to_csv(args.output_dir / "online_risk_events.csv", index=False)
    write_combined_index(args.output_dir, charts, curves, trades)
    print(f"Wrote {len(charts)} charts to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

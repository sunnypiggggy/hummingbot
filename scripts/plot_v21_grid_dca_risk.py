#!/usr/bin/env python3
"""Render Grid and DCA risk intervals as one self-contained Plotly dashboard."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


MECHANISMS = (
    "v21_buy_gate",
    "fomc_gate",
    "strategy_loss_breaker",
    "strategy_drawdown_breaker",
    "portfolio_loss_breaker",
    "portfolio_drawdown_breaker",
    "position_protection",
)
LABELS = {
    "v21_buy_gate": "v21 BUY 风控",
    "fomc_gate": "FOMC 宏观风控",
    "strategy_loss_breaker": "单策略绝对亏损",
    "strategy_drawdown_breaker": "单策略峰值回撤",
    "portfolio_loss_breaker": "组合绝对亏损",
    "portfolio_drawdown_breaker": "组合峰值回撤",
    "position_protection": "成本/持仓保护",
}
COLORS = {
    "v21_buy_gate": "rgba(245,158,11,0.20)",
    "fomc_gate": "rgba(139,92,246,0.22)",
    "strategy_loss_breaker": "rgba(239,68,68,0.20)",
    "strategy_drawdown_breaker": "rgba(249,115,22,0.20)",
    "portfolio_loss_breaker": "rgba(190,24,93,0.20)",
    "portfolio_drawdown_breaker": "rgba(244,63,94,0.16)",
    "position_protection": "rgba(14,165,233,0.18)",
}
MARKERS = {
    "v21_buy_gate": "triangle-down",
    "fomc_gate": "diamond",
    "strategy_loss_breaker": "x",
    "strategy_drawdown_breaker": "cross",
    "portfolio_loss_breaker": "square-x",
    "portfolio_drawdown_breaker": "star",
    "position_protection": "circle-open",
}
PAIR_ORDER = {
    "grid": ("BTC-FDUSD", "ETH-FDUSD"),
    "dca": ("BTC-USDT", "ETH-USDT"),
}


def _empty_series() -> pd.DataFrame:
    return pd.DataFrame(columns=(
        "strategy", "pair", "timestamp", "price", "equity",
        "peak_equity", "drawdown_pct",
    ))


def _empty_intervals() -> pd.DataFrame:
    return pd.DataFrame(columns=(
        "strategy", "pair", "mechanism", "start_ts", "end_ts",
        "trigger_value", "threshold", "action", "source", "enabled",
    ))


def read_csv(path: Path | None, *, intervals: bool = False) -> pd.DataFrame:
    if path is None or not path.exists():
        return _empty_intervals() if intervals else _empty_series()
    frame = pd.read_csv(path)
    required = (
        {"strategy", "pair", "mechanism", "start_ts", "end_ts"}
        if intervals else {"strategy", "pair", "timestamp", "price"}
    )
    if not required.issubset(frame.columns):
        raise ValueError(f"{path} is missing columns {sorted(required - set(frame.columns))}")
    if intervals:
        unknown = set(frame.mechanism.dropna()) - set(MECHANISMS)
        if unknown:
            raise ValueError(f"unknown risk mechanisms: {sorted(unknown)}")
        for column in ("trigger_value", "threshold", "action", "source", "enabled"):
            if column not in frame:
                frame[column] = ""
    return frame


def _utc(values: Any) -> Any:
    return pd.to_datetime(values, unit="s", utc=True)


def _nearest_price(frame: pd.DataFrame, timestamp: float) -> float | None:
    if frame.empty:
        return None
    index = (frame.timestamp.astype(float) - float(timestamp)).abs().idxmin()
    return float(frame.loc[index, "price"])


def build_figure(
    strategy: str, series: pd.DataFrame, intervals: pd.DataFrame,
) -> tuple[go.Figure, dict[str, dict[str, list[int]]]]:
    pairs = PAIR_ORDER[strategy]
    figure = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
        subplot_titles=pairs,
    )
    groups = {
        mechanism: {"traces": [], "shapes": []} for mechanism in MECHANISMS
    }
    for row, pair in enumerate(pairs, 1):
        prices = series[(series.strategy == strategy) & (series.pair == pair)].sort_values("timestamp")
        figure.add_trace(go.Scatter(
            x=_utc(prices.timestamp), y=prices.price, mode="lines",
            name=f"{pair} 价格", legendgroup=f"price-{pair}",
            line={"color": "#38bdf8", "width": 1.3},
            hovertemplate="%{x}<br>价格 %{y:,.4f}<extra></extra>",
        ), row=row, col=1)
        if "equity" in prices and prices.equity.notna().any():
            figure.add_trace(go.Scatter(
                x=_utc(prices.timestamp), y=prices.equity, mode="lines",
                name=f"{pair} 权益", legendgroup=f"equity-{pair}",
                line={"color": "#22c55e", "width": 1, "dash": "dot"},
                hovertemplate="%{x}<br>权益 %{y:,.4f}<extra></extra>",
            ), row=row, col=1, secondary_y=False)
        selected = intervals[
            (intervals.strategy == strategy) & (intervals.pair.isin((pair, "ALL")))
        ]
        for mechanism in MECHANISMS:
            events = selected[selected.mechanism == mechanism]
            if events.empty:
                continue
            marker_x, marker_y, marker_text = [], [], []
            for event in events.itertuples(index=False):
                end_ts = float(event.end_ts) if pd.notna(event.end_ts) else float(event.start_ts)
                figure.add_vrect(
                    x0=pd.to_datetime(float(event.start_ts), unit="s", utc=True),
                    x1=pd.to_datetime(end_ts, unit="s", utc=True),
                    fillcolor=COLORS[mechanism], opacity=1,
                    line={"color": COLORS[mechanism].replace("0.20", "0.75").replace("0.22", "0.75").replace("0.16", "0.75").replace("0.18", "0.75"), "width": 1},
                    row=row, col=1,
                )
                groups[mechanism]["shapes"].append(len(figure.layout.shapes) - 1)
                price = _nearest_price(prices, float(event.start_ts))
                if price is not None:
                    marker_x.append(pd.to_datetime(float(event.start_ts), unit="s", utc=True))
                    marker_y.append(price)
                    marker_text.append(
                        f"{LABELS[mechanism]}<br>动作: {event.action}<br>"
                        f"触发值: {event.trigger_value}<br>阈值: {event.threshold}<br>"
                        f"来源: {event.source}<br>执行开关: {event.enabled}"
                    )
            if marker_x:
                index = len(figure.data)
                figure.add_trace(go.Scatter(
                    x=marker_x, y=marker_y, text=marker_text, mode="markers",
                    name=LABELS[mechanism], legendgroup=mechanism,
                    marker={"symbol": MARKERS[mechanism], "size": 10, "color": COLORS[mechanism]},
                    hovertemplate="%{x}<br>%{text}<extra></extra>",
                ), row=row, col=1)
                groups[mechanism]["traces"].append(index)
    figure.update_layout(
        template="plotly_dark", height=820, hovermode="x unified",
        title=f"{'FDUSD Grid' if strategy == 'grid' else 'USDT DCA'} 风控生效区间",
        margin={"t": 80, "l": 70, "r": 30, "b": 45},
        legend={"orientation": "h", "y": 1.04},
    )
    return figure, groups


def render_dashboard(
    series: pd.DataFrame, intervals: pd.DataFrame, output: Path,
) -> dict[str, dict[str, dict[str, list[int]]]]:
    figures, groups = {}, {}
    for strategy in ("grid", "dca"):
        figures[strategy], groups[strategy] = build_figure(strategy, series, intervals)
    plot_html = {
        strategy: figure.to_html(
            full_html=False, include_plotlyjs=(strategy == "grid"),
            div_id=f"{strategy}-risk-chart",
        ) for strategy, figure in figures.items()
    }
    controls = "".join(
        f"<label><input type='checkbox' data-mechanism='{key}' checked>"
        f"{LABELS[key]}</label>" for key in MECHANISMS
    )
    encoded = json.dumps(groups, ensure_ascii=False)
    html = f"""<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'>
<title>Grid 与 DCA 风控阴影</title><style>
body{{margin:0;background:#07111f;color:#dbeafe;font:14px system-ui}}main{{max-width:1500px;margin:auto;padding:22px}}
.controls,.tabs{{display:flex;gap:10px;flex-wrap:wrap;margin:14px 0}}label,button{{background:#102033;color:#dbeafe;border:1px solid #29415d;border-radius:9px;padding:10px 13px;cursor:pointer}}
.panel{{display:none}}.panel.active{{display:block}}button.active{{border-color:#38bdf8;color:#7dd3fc}}.note{{color:#a9bfd7}}
</style></head><body><main><h1>Grid 与 DCA 七层风控</h1>
<p class='note'>阴影仅来自传入的历史回放或线上审计区间；缺失区间保持为空，不推测事件。</p>
<div class='tabs'><button data-tab='grid' class='active'>FDUSD Grid</button><button data-tab='dca'>USDT DCA</button></div>
<div class='controls'>{controls}</div>
<section id='grid-panel' class='panel active'>{plot_html['grid']}</section>
<section id='dca-panel' class='panel'>{plot_html['dca']}</section>
<script>const groups={encoded};
document.querySelectorAll('[data-tab]').forEach(b=>b.onclick=()=>{{document.querySelectorAll('[data-tab]').forEach(x=>x.classList.toggle('active',x===b));document.querySelectorAll('.panel').forEach(x=>x.classList.toggle('active',x.id===b.dataset.tab+'-panel'));window.dispatchEvent(new Event('resize'));}});
document.querySelectorAll('[data-mechanism]').forEach(cb=>cb.onchange=()=>{{for(const strategy of ['grid','dca']){{const chart=strategy+'-risk-chart',g=groups[strategy][cb.dataset.mechanism];(g.traces||[]).forEach(i=>Plotly.restyle(chart,{{visible:cb.checked}},[i]));const patch={{}};(g.shapes||[]).forEach(i=>patch[`shapes[${{i}}].visible`]=cb.checked);Plotly.relayout(chart,patch);}}}});
</script></main></body></html>"""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")
    return groups


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--series", type=Path)
    parser.add_argument("--intervals", type=Path)
    parser.add_argument("--output", type=Path, default=Path(
        "results/backtests/v21_grid_dca_risk/v21_grid_dca_risk_plotly.html"
    ))
    args = parser.parse_args()
    series = read_csv(args.series)
    intervals = read_csv(args.intervals, intervals=True)
    groups = render_dashboard(series, intervals, args.output)
    print(json.dumps({"output": str(args.output), "mechanisms": list(MECHANISMS),
                      "groups": groups}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

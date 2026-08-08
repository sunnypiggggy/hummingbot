#!/usr/bin/env python3
"""Build the self-contained v22 Grid/DCA offline risk audit dashboard."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


MECHANISMS = (
    "v22_weekly_buy_gate",
    "fomc_gate",
    "strategy_loss_breaker",
    "strategy_drawdown_breaker",
    "portfolio_loss_breaker",
    "portfolio_drawdown_breaker",
    "position_protection",
)
LABELS = {
    "v22_weekly_buy_gate": "v22 周度 BUY 风控",
    "fomc_gate": "FOMC 宏观风控",
    "strategy_loss_breaker": "单策略绝对亏损",
    "strategy_drawdown_breaker": "单策略峰值回撤",
    "portfolio_loss_breaker": "组合绝对亏损",
    "portfolio_drawdown_breaker": "组合峰值回撤",
    "position_protection": "持仓保护",
}
# Five restrained palette roots plus line style/marker distinctions; no state relies on colour alone.
STYLES = {
    "v22_weekly_buy_gate": ("rgba(217,119,6,0.18)", "#92400e", "solid", "triangle-down"),
    "fomc_gate": ("rgba(124,58,237,0.14)", "#5b21b6", "dash", "diamond"),
    "strategy_loss_breaker": ("rgba(190,24,93,0.14)", "#9d174d", "dot", "x"),
    "strategy_drawdown_breaker": ("rgba(234,88,12,0.13)", "#9a3412", "dashdot", "cross"),
    "portfolio_loss_breaker": ("rgba(71,85,105,0.14)", "#334155", "longdash", "square-x"),
    "portfolio_drawdown_breaker": ("rgba(30,64,175,0.12)", "#1e3a8a", "longdashdot", "star"),
    "position_protection": ("rgba(5,150,105,0.12)", "#065f46", "dot", "circle-open"),
}
PAIR_ORDER = {
    "grid": ("BTC-FDUSD", "ETH-FDUSD"),
    "dca": ("BTC-USDT", "ETH-USDT"),
}
SERIES_COLUMNS = (
    "strategy", "pair", "timestamp", "price", "equity", "peak_equity", "drawdown_pct",
    "probability", "entry_threshold", "fold",
)
INTERVAL_COLUMNS = (
    "strategy", "pair", "mechanism", "start_ts", "end_ts", "trigger_value", "threshold",
    "action", "reason", "source", "enabled", "model_week", "model_sha256",
    "feature_schema_sha256", "strategy_schema_sha256",
    "phase",
)


def _empty(columns: tuple[str, ...]) -> pd.DataFrame:
    return pd.DataFrame(columns=columns)


def read_series(path: Path | None) -> pd.DataFrame:
    if path is None or not path.exists():
        return _empty(SERIES_COLUMNS)
    frame = pd.read_csv(path)
    required = {"strategy", "pair", "timestamp", "price"}
    if not required.issubset(frame):
        raise ValueError(f"series missing {sorted(required - set(frame))}")
    for column in SERIES_COLUMNS:
        if column not in frame:
            frame[column] = pd.NA
    return frame


def read_intervals(path: Path | None) -> pd.DataFrame:
    if path is None or not path.exists():
        return _empty(INTERVAL_COLUMNS)
    frame = pd.read_csv(path)
    required = {"strategy", "pair", "mechanism", "start_ts", "end_ts"}
    if not required.issubset(frame):
        raise ValueError(f"intervals missing {sorted(required - set(frame))}")
    unknown = set(frame.mechanism.dropna()) - set(MECHANISMS)
    if unknown:
        raise ValueError(f"unknown mechanisms: {sorted(unknown)}")
    for column in INTERVAL_COLUMNS:
        if column not in frame:
            frame[column] = pd.NA
    return frame


def _utc(values: Any) -> Any:
    return pd.to_datetime(values, unit="s", utc=True)


def _nearest_price(frame: pd.DataFrame, timestamp: float) -> float | None:
    if frame.empty:
        return None
    index = (frame.timestamp.astype(float) - timestamp).abs().idxmin()
    return float(frame.loc[index, "price"])


def _hover(event: Any, mechanism: str) -> str:
    def value(name: str) -> str:
        raw = getattr(event, name, "")
        return "" if pd.isna(raw) else str(raw)
    return (
        f"{LABELS[mechanism]}<br>阶段: {value('phase')}<br>动作: {value('action')}<br>原因: {value('reason')}<br>"
        f"触发值: {value('trigger_value')}<br>阈值: {value('threshold')}<br>"
        f"模型周: {value('model_week')}<br>来源: {value('source')}<br>"
        f"执行开关: {value('enabled')}<br>模型哈希: {value('model_sha256')}<br>"
        f"特征哈希: {value('feature_schema_sha256')}<br>策略哈希: {value('strategy_schema_sha256')}"
    )


def build_figure(strategy: str, series: pd.DataFrame, intervals: pd.DataFrame) -> tuple[go.Figure, dict[str, Any]]:
    pairs = PAIR_ORDER[strategy]
    titles: list[str] = []
    for pair in pairs:
        equity_title = (
            "BTC+ETH DCA 组合权益、组合峰值与组合回撤（初始 380 USDT）"
            if strategy == "dca"
            else "BTC+ETH Grid 组合权益、组合峰值与组合回撤（初始 400 FDUSD）"
        )
        titles.extend((f"{pair} 价格与风控区间", f"{pair} v22 周概率与 fold-local 阈值", equity_title))
    figure = make_subplots(
        rows=6, cols=1, shared_xaxes=True, vertical_spacing=.025,
        row_heights=[.22, .12, .16, .22, .12, .16],
        specs=[[{}], [{}], [{"secondary_y": True}], [{}], [{}], [{"secondary_y": True}]],
        subplot_titles=titles,
    )
    groups: dict[str, Any] = {key: {"shapes": []} for key in MECHANISMS}
    pending_shapes: list[dict[str, Any]] = []
    legend_seen: set[str] = set()
    for pair_index, pair in enumerate(pairs):
        price_row, probability_row, equity_row = 1 + pair_index * 3, 2 + pair_index * 3, 3 + pair_index * 3
        item = series[(series.strategy == strategy) & (series.pair == pair)].sort_values("timestamp")
        figure.add_trace(go.Scatter(
            x=_utc(item.timestamp), y=item.price, mode="lines", name=f"{pair} 价格",
            line={"color": "#1e40af", "width": 1.2},
            hovertemplate="%{x}<br>价格 %{y:,.6f}<extra></extra>",
        ), row=price_row, col=1)
        probability = item[item.probability.notna()]
        if not probability.empty:
            custom = probability[["fold"]].fillna("").to_numpy()
            figure.add_trace(go.Scatter(
                x=_utc(probability.timestamp), y=probability.probability, customdata=custom,
                name=f"{pair} 周概率", line={"color": "#d97706", "width": 1.2},
                hovertemplate="%{x}<br>概率 %{y:.6f}<br>fold %{customdata[0]}<extra></extra>",
            ), row=probability_row, col=1)
            figure.add_trace(go.Scatter(
                x=_utc(probability.timestamp), y=probability.entry_threshold,
                name=f"{pair} fold-local 阈值", line={"color": "#334155", "width": 1, "dash": "dot"},
                hovertemplate="%{x}<br>阈值 %{y:.6f}<extra></extra>",
            ), row=probability_row, col=1)
        equity = item[item.equity.notna()]
        if not equity.empty:
            equity_prefix = "DCA 组合" if strategy == "dca" else "Grid 组合"
            figure.add_trace(go.Scatter(
                x=_utc(equity.timestamp), y=equity.equity, name=f"{equity_prefix}权益",
                line={"color": "#1e40af", "width": 1.3},
            ), row=equity_row, col=1, secondary_y=False)
            figure.add_trace(go.Scatter(
                x=_utc(equity.timestamp), y=equity.peak_equity, name=f"{equity_prefix}峰值",
                line={"color": "#64748b", "width": 1, "dash": "dash"},
            ), row=equity_row, col=1, secondary_y=False)
            figure.add_trace(go.Scatter(
                x=_utc(equity.timestamp), y=equity.drawdown_pct, name=f"{equity_prefix}回撤 %",
                line={"color": "#be185d", "width": 1, "dash": "dot"},
            ), row=equity_row, col=1, secondary_y=True)
        selected = intervals[(intervals.strategy == strategy) & intervals.pair.isin((pair, "ALL"))]
        for mechanism in MECHANISMS:
            fill, line, dash, marker = STYLES[mechanism]
            marker_x, marker_y, marker_text = [], [], []
            for event in selected[selected.mechanism == mechanism].itertuples(index=False):
                end = float(event.end_ts) if pd.notna(event.end_ts) else float(item.timestamp.max())
                raw_phase = getattr(event, "phase", "")
                phase = "" if pd.isna(raw_phase) else str(raw_phase).upper()
                phase_dash = {
                    "EXITING": "solid", "COOLDOWN": "dash", "REENTRY": "dot",
                }.get(phase, dash)
                # Repeat each band on the price and final-equity panels so the
                # equity path can be checked against the exact active mechanism.
                axis_refs = {
                    1: ("x", "y domain"), 3: ("x3", "y3 domain"),
                    4: ("x4", "y5 domain"), 6: ("x6", "y7 domain"),
                }
                for panel_row in (price_row, equity_row):
                    xref, yref = axis_refs[panel_row]
                    pending_shapes.append({
                        "type": "rect", "xref": xref, "yref": yref,
                        "x0": pd.to_datetime(float(event.start_ts), unit="s", utc=True),
                        "x1": pd.to_datetime(end, unit="s", utc=True), "y0": 0, "y1": 1,
                        "fillcolor": fill, "opacity": 1,
                        "line": {"color": line, "width": 1, "dash": phase_dash}, "layer": "below",
                    })
                    groups[mechanism]["shapes"].append({
                        "index": len(pending_shapes) - 1, "pair": pair,
                    })
                price = _nearest_price(item, float(event.start_ts))
                if price is not None:
                    marker_x.append(pd.to_datetime(float(event.start_ts), unit="s", utc=True))
                    marker_y.append(price); marker_text.append(_hover(event, mechanism))
            if marker_x:
                figure.add_trace(go.Scatter(
                    x=marker_x, y=marker_y, text=marker_text, mode="markers", name=LABELS[mechanism],
                    legendgroup=mechanism, showlegend=mechanism not in legend_seen,
                    marker={"symbol": marker, "size": 9, "color": line},
                    hovertemplate="%{x}<br>%{text}<extra></extra>",
                ), row=price_row, col=1)
                legend_seen.add(mechanism)
    figure.update_layout(
        template="plotly_white", height=1780, hovermode="x unified",
        title=(f"{'FDUSD Grid' if strategy == 'grid' else 'USDT DCA'} · v22 weekly walk-forward "
               "· offline validation · NO-GO"),
        margin={"t": 95, "l": 72, "r": 72, "b": 50},
        legend={"orientation": "h", "y": 1.02},
        font={"family": "Arial, Microsoft YaHei, sans-serif", "color": "#172033"},
        shapes=pending_shapes,
    )
    for row in (3, 6):
        figure.update_yaxes(title_text="组合权益", row=row, col=1, secondary_y=False)
        figure.update_yaxes(title_text="回撤 %", row=row, col=1, secondary_y=True)
    return figure, groups


def render_dashboard(series: pd.DataFrame, intervals: pd.DataFrame, output: Path, summary: dict[str, Any] | None = None) -> dict[str, Any]:
    figures, groups = {}, {}
    for strategy in ("grid", "dca"):
        figures[strategy], groups[strategy] = build_figure(strategy, series, intervals)
    plot_html = {
        strategy: figure.to_html(full_html=False, include_plotlyjs=(strategy == "grid"),
                                 div_id=f"{strategy}-v22-risk-chart", config={"responsive": True, "displaylogo": False})
        for strategy, figure in figures.items()
    }
    scenario_rows = pd.DataFrame((summary or {}).get("dca_scenarios", []))
    ablation_html = "<p class='note'>消融实验无数据</p>"
    ablation_order = ("baseline", "v22_btc_only", "v22_eth_only", "v22")
    if not scenario_rows.empty and set(ablation_order).issubset(set(scenario_rows.scenario)):
        ablation = scenario_rows.set_index("scenario").loc[list(ablation_order)].reset_index()
        labels = ["无技术门", "仅 BTC 机器人 v22", "仅 ETH 机器人 v22", "BTC+ETH v22"]
        patterns = ["", "/", "x", "."]
        comparison = make_subplots(rows=1, cols=3, subplot_titles=("组合净收益（USDT）", "最大回撤（%）", "暂停时长（pair-hours）"))
        for column, col, color in (("combined_net_pnl_quote", 1, "#1e40af"),
                                   ("combined_max_drawdown_pct", 2, "#be185d"),
                                   ("buy_disabled_pair_hours", 3, "#d97706")):
            comparison.add_trace(go.Bar(
                x=labels, y=ablation[column], showlegend=False,
                marker={"color": color, "pattern": {"shape": patterns}, "line": {"color": "#334155", "width": 1}},
                customdata=ablation[["scenario"]],
                hovertemplate="%{x}<br>%{y:.4f}<br>%{customdata[0]}<extra></extra>",
            ), row=1, col=col)
            comparison.update_yaxes(rangemode="tozero", row=1, col=col)
        comparison.update_xaxes(tickangle=-18)
        comparison.update_layout(
            template="plotly_white", height=510, title="DCA v22 机器人级消融实验",
            margin={"t": 80, "l": 55, "r": 25, "b": 110},
            font={"family": "Arial, Microsoft YaHei, sans-serif", "color": "#172033"},
        )
        ablation_html = comparison.to_html(full_html=False, include_plotlyjs=False,
                                            div_id="dca-v22-ablation-chart",
                                            config={"responsive": True, "displaylogo": False})
    controls = "".join(
        f"<label><input type='checkbox' data-mechanism='{key}' checked>{LABELS[key]}</label>"
        for key in MECHANISMS
    )
    pair_controls = "".join(
        f"<label><input type='checkbox' data-bot-pair='{pair}' checked>{pair} 机器人阴影</label>"
        for pair in ("BTC-FDUSD", "ETH-FDUSD", "BTC-USDT", "ETH-USDT")
    )
    encoded = json.dumps(groups, ensure_ascii=False)
    html = f"""<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'>
<meta name='viewport' content='width=device-width,initial-scale=1'><title>v22 Grid/DCA 离线风控审计</title><style>
body{{margin:0;background:#f8fafc;color:#172033;font:14px Arial,'Microsoft YaHei',sans-serif}}
main{{max-width:1500px;margin:auto;padding:22px}}h1{{margin-bottom:6px}}.warning{{background:#fff7ed;border-left:5px solid #d97706;padding:14px}}
.controls,.tabs{{display:flex;gap:9px;flex-wrap:wrap;margin:13px 0}}label,button{{background:#fff;color:#172033;border:1px solid #cbd5e1;border-radius:7px;padding:8px 11px;cursor:pointer}}
button.active{{border-color:#1e40af;color:#1e40af}}.panel{{display:none;background:#fff}}.panel.active{{display:block}}.note{{color:#475569}}
@media(max-width:800px){{main{{padding:10px}}}}
</style></head><body><main><h1>v22 Grid/DCA 离线风控审计</h1>
<p class='warning'><strong>NO-GO：</strong>本报告仅使用冻结签名周的反事实建议；deployment_allowed=false，promotion_authorized=false，不代表实盘 BUY 权限。</p>
<p class='note'>阴影复选框只控制对应风控区间；价格、权益、峰值、回撤、概率、阈值及事件标记始终保留。没有可信事件的机制明确显示“无数据”。</p>
<div class='tabs'><button data-tab='grid' class='active'>FDUSD Grid</button><button data-tab='dca'>USDT DCA</button></div>
<div class='controls'>{controls}</div><div class='controls'>{pair_controls}</div>
<section id='grid-panel' class='panel active'>{plot_html['grid']}</section>
<section id='dca-panel' class='panel'><h2>DCA 机器人级消融</h2>{ablation_html}{plot_html['dca']}</section>
<script>const groups={encoded};
const state={{mechanisms:Object.fromEntries([...document.querySelectorAll('[data-mechanism]')].map(x=>[x.dataset.mechanism,true])),pairs:Object.fromEntries([...document.querySelectorAll('[data-bot-pair]')].map(x=>[x.dataset.botPair,true]))}};
function refresh(){{for(const strategy of ['grid','dca']){{const patch={{}};for(const [mechanism,group] of Object.entries(groups[strategy])){{for(const item of group.shapes||[]){{const pairAllowed=item.pair==='ALL'||state.pairs[item.pair]!==false;patch[`shapes[${{item.index}}].visible`]=state.mechanisms[mechanism]!==false&&pairAllowed;}}}}if(Object.keys(patch).length)Plotly.relayout(strategy+'-v22-risk-chart',patch);}}}}
document.querySelectorAll('[data-tab]').forEach(b=>b.onclick=()=>{{document.querySelectorAll('[data-tab]').forEach(x=>x.classList.toggle('active',x===b));document.querySelectorAll('.panel').forEach(x=>x.classList.toggle('active',x.id===b.dataset.tab+'-panel'));window.dispatchEvent(new Event('resize'));}});
document.querySelectorAll('[data-mechanism]').forEach(cb=>cb.onchange=()=>{{state.mechanisms[cb.dataset.mechanism]=cb.checked;refresh();}});
document.querySelectorAll('[data-bot-pair]').forEach(cb=>cb.onchange=()=>{{state.pairs[cb.dataset.botPair]=cb.checked;refresh();}});
</script></main></body></html>"""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")
    return groups


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--series", type=Path)
    parser.add_argument("--intervals", type=Path)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--output", type=Path, default=Path("results/backtests/v22_grid_dca_offline_audit/v22_grid_dca_risk_plotly.html"))
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text(encoding="utf-8")) if args.summary and args.summary.exists() else {}
    render_dashboard(read_series(args.series), read_intervals(args.intervals), args.output, summary)
    print(json.dumps({"output": str(args.output), "mechanisms": list(MECHANISMS)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

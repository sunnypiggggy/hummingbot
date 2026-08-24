#!/usr/bin/env python3
"""Run every Binance AI-style Grid preset under the frozen v22 gate.

All presets share the same 360-day candles, 200 FDUSD per pair, execution
filters and risk state machine.  A 10 FDUSD post-quantisation order minimum
limits every preset to nine BUY and nine SELL levels.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import OrderedDict
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
from plotly.offline.offline import get_plotlyjs
from plotly.subplots import make_subplots

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from scripts.backtest_binance_short_sideways_long_vs_bidirectional_360d import (
    END_TS,
    MINIMUM_ORDER,
    PAIR_CAPITAL,
    ROOT,
    ROWS_PER_PAIR,
    SIDE_BUDGET,
    START_TS,
    Preset,
    add_v22_shapes,
    expand_v22_gate,
    hourly,
    load_inputs,
    risk_intervals,
    sha256,
    simulate_arm,
    summarise,
    validate_simulation,
)


DEFAULT_OUTPUT = ROOT / "results/backtests/binance_ai_grid_all_10usd_v22_360d"

# Effective take-profit values preserve the previously approved Binance AI
# mapping.  Only the executable grid count changes to enforce the 10 FDUSD
# post-quantisation minimum with a 100 FDUSD side budget.
AI_PRESETS: OrderedDict[str, Preset] = OrderedDict([
    ("short_sideways", Preset("short_sideways", "短期横盘", 0.07695669969152726, 18, 0.004)),
    ("medium_sideways", Preset("medium_sideways", "中短期横盘", 0.12698379475402316, 18, 0.004)),
    ("medium_volatility", Preset("medium_volatility", "中期波动", 0.08618475700686561, 18, 0.004)),
    ("long_volatility", Preset("long_volatility", "长期波动", 0.5246511596640915, 18, 0.014179761072002472)),
])

SOURCE_LEVELS = {
    "short_sideways": 25,
    "medium_sideways": 42,
    "medium_volatility": 28,
    "long_volatility": 80,
}

PALETTE = {
    "short_sideways": "#d4a017",
    "medium_sideways": "#3465a4",
    "medium_volatility": "#d97706",
    "long_volatility": "#7a5195",
}


def fixed_baseline() -> pd.DataFrame:
    path = ROOT / "results/backtests/binance_short_sideways_10usd_long_vs_bidirectional_360d/summary.csv"
    frame = pd.read_csv(path)
    return frame[(frame.scope == "protected_v22") & (frame.preset == "fixed_current")][
        ["pair", "mode", "net_pnl_fdusd", "max_drawdown_pct"]
    ].rename(columns={"net_pnl_fdusd": "fixed_pnl_fdusd", "max_drawdown_pct": "fixed_drawdown_pct"})


def returns_matrix(summary: pd.DataFrame) -> pd.DataFrame:
    matrix = summary.pivot(index=["pair", "mode"], columns="preset", values="net_pnl_fdusd").reset_index()
    matrix = matrix[["pair", "mode", *AI_PRESETS.keys()]]
    matrix["mode"] = matrix["mode"].map({"bidirectional": "双向", "long_only": "只做多"})
    matrix["pair"] = matrix["pair"].str.replace("-FDUSD", "", regex=False)
    return matrix


def write_markdown(output: Path, summary: pd.DataFrame, matrix: pd.DataFrame, validation: dict[str, Any]) -> None:
    labels = {key: preset.label for key, preset in AI_PRESETS.items()}
    lines = [
        "# Binance AI 风格Grid参数：v22口径360天收益",
        "",
        "区间：2025-08-24 00:00—2026-08-19 00:00 UTC。每对200 FDUSD；每侧100 FDUSD；Maker 0%；FOMC未参与。",
        "",
        "| 交易对 | 模式 | 短期横盘 | 中短期横盘 | 中期波动 | 长期波动 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in matrix.itertuples(index=False):
        lines.append(f"| {row.pair} | {row.mode} | {row.short_sideways:+.4f} | {row.medium_sideways:+.4f} | "
                     f"{row.medium_volatility:+.4f} | {row.long_volatility:+.4f} |")
    lines.extend([
        "",
        "单位：FDUSD；均为单交易对连续权益的净收益，不是BTC+ETH组合收益。",
        "",
        "## 参数与执行约束",
        "",
        "| 参数 | 原始格数 | 执行格数 | BUY/SELL格数 | 实际格距 | 止盈 |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for key, preset in AI_PRESETS.items():
        lines.append(f"| {labels[key]} | {SOURCE_LEVELS[key]} | {preset.levels} | 9 / 9 | "
                     f"{preset.actual_step * 100:.4f}% | {preset.take_profit * 100:.4f}% |")
    lines.extend([
        "",
        "## 验收",
        "",
        f"- 实际挂单 `{validation['order_rows']}` 笔；最低金额 `{validation['minimum_placed_order_fdusd']:.8f} FDUSD`。",
        f"- 低于10 FDUSD：`{validation['orders_below_10_fdusd']}`；负仓：`{validation['negative_inventory_observations']}`。",
        f"- 无BUY支撑的只做多止盈SELL：`{validation['long_only_unbacked_take_profit_sells']}`。",
        f"- v22阻止期间普通BUY：`{validation['protected_buys_during_v22_block']}`。",
        "- 现货双向SELL只出售已有库存，不建立合约空仓。",
    ])
    (output / "V22_RETURNS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def overview_figure(summary: pd.DataFrame) -> go.Figure:
    labels = ["BTC 双向", "BTC 只做多", "ETH 双向", "ETH 只做多"]
    keys = [("BTC-FDUSD", "bidirectional"), ("BTC-FDUSD", "long_only"),
            ("ETH-FDUSD", "bidirectional"), ("ETH-FDUSD", "long_only")]
    figure = go.Figure()
    for preset_id, preset in AI_PRESETS.items():
        values = []
        drawdowns = []
        for pair, mode in keys:
            row = summary[(summary.pair == pair) & (summary["mode"] == mode) & (summary.preset == preset_id)].iloc[0]
            values.append(float(row.net_pnl_fdusd))
            drawdowns.append(float(row.max_drawdown_pct))
        figure.add_trace(go.Bar(
            x=labels, y=values, name=preset.label, marker_color=PALETTE[preset_id],
            text=[f"{value:+.2f}" for value in values], textposition="outside",
            customdata=drawdowns,
            hovertemplate="%{x}<br>收益=%{y:+.4f} FDUSD<br>最大回撤=%{customdata:.4f}%<extra>%{fullData.name}</extra>",
        ))
    figure.add_hline(y=0, line_color="#273746", line_width=1)
    figure.update_layout(
        title="四组Binance AI Grid参数 · v22口径360天收益",
        template="plotly_white", barmode="group", height=720,
        yaxis_title="净收益（FDUSD）", xaxis_title="单交易对 / 模式",
        legend=dict(orientation="h", y=1.08, x=0), margin=dict(l=70, r=35, t=110, b=70),
    )
    return figure


def strategy_figure(pair: str, mode: str, equity: pd.DataFrame, summary: pd.DataFrame) -> go.Figure:
    mode_label = "双向" if mode == "bidirectional" else "只做多"
    subset = equity[(equity.pair == pair) & (equity["mode"] == mode)]
    figure = make_subplots(
        rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.045,
        row_heights=[0.24, 0.31, 0.22, 0.23],
        subplot_titles=("标的价格", "单交易对连续权益", "回撤", "基础币市场敞口"),
    )
    first = hourly(subset[subset.preset == next(iter(AI_PRESETS))])
    figure.add_trace(go.Scatter(x=first.datetime, y=first.price, name=f"{pair}价格",
                                line=dict(color="#273746", width=1.2)), row=1, col=1)
    for preset_id, preset in AI_PRESETS.items():
        data = hourly(subset[subset.preset == preset_id])
        metric = summary[(summary.pair == pair) & (summary["mode"] == mode) & (summary.preset == preset_id)].iloc[0]
        name = f"{preset.label}（{metric.net_pnl_fdusd:+.2f}）"
        figure.add_trace(go.Scatter(x=data.datetime, y=data.equity, name=name,
                                    line=dict(color=PALETTE[preset_id], width=1.5)), row=2, col=1)
        drawdown = (data.equity / data.equity.cummax() - 1.0) * 100.0
        figure.add_trace(go.Scatter(x=data.datetime, y=drawdown, name=f"{preset.label}回撤",
                                    legendgroup=preset_id, showlegend=False,
                                    line=dict(color=PALETTE[preset_id], width=1.2)), row=3, col=1)
        figure.add_trace(go.Scatter(x=data.datetime, y=data.base_value, name=f"{preset.label}敞口",
                                    legendgroup=preset_id, showlegend=False,
                                    line=dict(color=PALETTE[preset_id], width=1.2)), row=4, col=1)
    add_v22_shapes(figure, risk_intervals(subset[subset.preset == next(iter(AI_PRESETS))]), 5)
    figure.update_layout(
        title=f"{pair} · {mode_label} · 四组Binance AI参数", template="plotly_white", height=1100,
        hovermode="x unified", legend=dict(orientation="h", y=1.055, x=0),
        margin=dict(l=70, r=35, t=110, b=50),
    )
    figure.update_yaxes(title_text="价格", row=1, col=1)
    figure.update_yaxes(title_text="FDUSD", row=2, col=1)
    figure.update_yaxes(title_text="%", row=3, col=1)
    figure.update_yaxes(title_text="FDUSD", row=4, col=1)
    return figure


def write_plotly(output: Path, equity: pd.DataFrame, summary: pd.DataFrame) -> Path:
    tabs: list[tuple[str, go.Figure]] = [("收益矩阵", overview_figure(summary))]
    for pair in ("BTC-FDUSD", "ETH-FDUSD"):
        for mode in ("bidirectional", "long_only"):
            tabs.append((f"{pair.split('-')[0]} {'双向' if mode == 'bidirectional' else '只做多'}",
                         strategy_figure(pair, mode, equity, summary)))
    sections = []
    for index, (_, figure) in enumerate(tabs):
        payload = figure.to_json(pretty=False, remove_uids=True).replace("</", "<\\/")
        sections.append(
            f'<section id="tab-{index}" class="plot-tab {"active" if index == 0 else ""}">'
            f'<div id="plot-{index}" class="plot-target"></div><script id="spec-{index}" type="application/json">{payload}</script></section>'
        )
    buttons = "".join(
        f'<button class="tab-button {"active" if index == 0 else ""}" onclick="showTab({index},this)">{label}</button>'
        for index, (label, _) in enumerate(tabs)
    )
    html = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Binance AI Grid参数 · v22口径</title><style>body{{font-family:Arial,'Microsoft YaHei',sans-serif;margin:0;background:#f6f7f9;color:#1f2937}}header{{padding:18px 24px;background:#fff;border-bottom:1px solid #ddd}}h1{{font-size:22px;margin:0 0 8px}}header p{{margin:4px 0;color:#52606d}}nav{{position:sticky;top:0;z-index:5;background:#fff;padding:10px 18px;border-bottom:1px solid #ddd;overflow-x:auto;white-space:nowrap}}button{{padding:8px 13px;margin:2px;border:1px solid #cbd5e1;background:#fff;border-radius:7px;cursor:pointer}}button.active{{background:#273746;color:#fff}}.plot-tab{{display:none;background:#fff;margin:14px;box-shadow:0 1px 4px #ddd;min-height:650px}}.plot-tab.active{{display:block}}.plot-target{{width:100%}}@media(max-width:700px){{header{{padding:14px}}.plot-tab{{margin:4px}}}}</style><script>{get_plotlyjs()}</script></head><body>
<header><h1>Binance AI 风格Grid参数 · v22口径360天回测</h1><p>BTC/ETH分别展示，双向与只做多独立；每对200 FDUSD，量化后每单不少于10 FDUSD。</p><p>区间：2025-08-24—2026-08-19 UTC；Maker 0%；FOMC未参与；仅离线验证，不构成实盘授权。</p></header><nav>{buttons}</nav>{''.join(sections)}
<div style="position:fixed;right:12px;bottom:12px;z-index:8;background:#fff;padding:7px;border:1px solid #ddd;border-radius:8px"><button onclick="setWindow('2025-08-24','2026-08-19')">360天</button><button onclick="setWindow('2026-01-01','2026-03-01')">2026年1–2月</button><button onclick="setWindow('2026-05-01','2026-07-01')">2026年5–6月</button><button onclick="toggleV22()">v22阴影</button></div>
<script>const rendered=new Set();function renderTab(i){{if(rendered.has(i))return;const s=JSON.parse(document.getElementById('spec-'+i).textContent);Plotly.newPlot('plot-'+i,s.data,s.layout,{{responsive:true,displaylogo:false}});rendered.add(i);}}function showTab(i,b){{document.querySelectorAll('.plot-tab').forEach(x=>x.classList.remove('active'));document.querySelectorAll('.tab-button').forEach(x=>x.classList.remove('active'));document.getElementById('tab-'+i).classList.add('active');b.classList.add('active');renderTab(i);setTimeout(()=>window.dispatchEvent(new Event('resize')),50);}}function setWindow(a,b){{const p=document.querySelector('.plot-tab.active .js-plotly-plot');if(!p||!p.layout.xaxis||p.layout.xaxis.type==='category')return;const u={{}};for(let i=1;i<=4;i++){{const k=i===1?'xaxis':'xaxis'+i;u[k+'.range']=[a,b];}}Plotly.relayout(p,u);}}function toggleV22(){{const p=document.querySelector('.plot-tab.active .js-plotly-plot');if(!p||!(p.layout.shapes||[]).length)return;p.__v22Visible=p.__v22Visible===false;const u={{}};(p.layout.shapes||[]).forEach((_,i)=>u['shapes['+i+'].visible']=p.__v22Visible);Plotly.relayout(p,u);}}renderTab(0);</script></body></html>"""
    path = output / "binance_ai_grid_all_v22_plotly.html"
    path.write_text(html, encoding="utf-8")
    return path


def run(output: Path) -> None:
    candles, gate, filters, evidence = load_inputs()
    gate_arrays = {pair: expand_v22_gate(candles[pair], gate, pair) for pair in candles}
    summaries: list[dict[str, Any]] = []
    equity_frames: list[pd.DataFrame] = []
    event_frames: list[pd.DataFrame] = []

    for mode in ("bidirectional", "long_only"):
        for preset in AI_PRESETS.values():
            states, equity, events = simulate_arm(candles, gate_arrays, filters, mode, preset, "protected_v22")
            summaries.extend(summarise(states, equity, "protected_v22", mode, preset))
            equity_frames.append(equity)
            if not events.empty:
                events["scope"] = "protected_v22"
                event_frames.append(events)

    equity = pd.concat(equity_frames, ignore_index=True)
    events = pd.concat(event_frames, ignore_index=True)
    validation = validate_simulation(equity, events)
    summary = pd.DataFrame(summaries).merge(fixed_baseline(), on=["pair", "mode"], how="left")
    summary["pnl_delta_vs_fixed_fdusd"] = summary.net_pnl_fdusd - summary.fixed_pnl_fdusd
    summary["drawdown_delta_vs_fixed_pp"] = summary.max_drawdown_pct - summary.fixed_drawdown_pct
    matrix = returns_matrix(summary)

    output.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output / "summary.csv", index=False, encoding="utf-8-sig")
    matrix.to_csv(output / "v22_returns_matrix.csv", index=False, encoding="utf-8-sig")
    equity.to_csv(output / "continuous_equity.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    events.to_csv(output / "trade_and_risk_events.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    parameters = pd.DataFrame([{
        "preset": key,
        "label": preset.label,
        "source_levels": SOURCE_LEVELS[key],
        "executable_levels": preset.levels,
        "side_levels": preset.side_levels,
        "total_range_pct": preset.total_range * 100,
        "actual_step_pct": preset.actual_step * 100,
        "take_profit_pct": preset.take_profit * 100,
        "minimum_order_fdusd": MINIMUM_ORDER,
    } for key, preset in AI_PRESETS.items()])
    parameters.to_csv(output / "parameter_mapping.csv", index=False, encoding="utf-8-sig")
    (output / "validation.json").write_text(json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(output, summary, matrix, validation)
    write_plotly(output, equity, summary)
    result = {
        "schema": "binance-ai-grid-all-10usd-v22-360d-v1",
        "offline_only": True,
        "deployment_allowed": False,
        "oci_modified": False,
        "window": {"start": START_TS, "end_exclusive": END_TS, "rows_per_pair": ROWS_PER_PAIR},
        "execution": {"pair_capital_fdusd": PAIR_CAPITAL, "side_budget_fdusd": SIDE_BUDGET,
                      "minimum_order_fdusd": MINIMUM_ORDER, "v22_enabled": True, "fomc_enabled": False,
                      "maker_fee": 0.0, "bnb_fee_used": False},
        "parameters": [asdict(preset) | {"source_levels": SOURCE_LEVELS[key], "actual_step": preset.actual_step}
                       for key, preset in AI_PRESETS.items()],
        "summaries": summaries,
        "validation": validation,
        "evidence": evidence,
    }
    (output / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    artifacts = {}
    for path in sorted(output.iterdir()):
        if path.is_file() and path.name != "manifest.json":
            artifacts[path.name] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    (output / "manifest.json").write_text(json.dumps({
        "schema": "binance-ai-grid-all-10usd-v22-360d-manifest-v1", "artifacts": artifacts,
    }, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run(args.output.resolve())


if __name__ == "__main__":
    main()

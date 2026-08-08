#!/usr/bin/env python3
"""Build a self-contained Plotly page for FDUSD live mechanisms 1-3."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = ROOT / "results/backtests/fdusd_grid_v21_mechanisms_250d"
DEFAULT_OUTPUT = (
    ROOT / "results/backtests/fdusd_live_mechanisms_1_3/"
    "fdusd_live_mechanisms_1_3_plotly.html"
)
COLORS = {
    "v21": "rgba(245,158,11,0.18)",
    "fomc": "rgba(139,92,246,0.24)",
    "pair_loss": "rgba(239,68,68,0.20)",
    "pair_drawdown": "rgba(249,115,22,0.20)",
}


def utc(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, unit="s", utc=True)


def read_fomc(path: Path | None) -> pd.DataFrame:
    """Read real FOMC intervals; an absent source deliberately yields none."""
    if path is None:
        return pd.DataFrame(columns=["start_ts", "end_ts", "reason"])
    frame = pd.read_csv(path)
    required = {"start_ts", "end_ts"}
    if not required.issubset(frame.columns):
        raise ValueError(f"FOMC interval file needs columns {sorted(required)}")
    if "reason" not in frame:
        frame["reason"] = "FOMC macro pause"
    return frame


def current_macro_status(path: Path | None) -> tuple[str, str]:
    if path is None or not path.exists():
        return "未提供线上快照", "报告不会据此猜测当前 FOMC 状态"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        generated = datetime.fromisoformat(str(payload["generated_at"]).replace("Z", "+00:00"))
        age = (datetime.now(timezone.utc) - generated.astimezone(timezone.utc)).total_seconds()
        healthy = bool(payload.get("source_healthy")) and -10 <= age <= 150
        paused = bool(payload.get("pause_trading", payload.get("pause_new_orders", True)))
        if not healthy:
            return "Fail-Closed", f"宏观状态异常或过期（年龄 {age:.0f} 秒）"
        return ("FOMC 生效中" if paused else "健康、无活动窗口"), f"快照年龄 {age:.0f} 秒"
    except Exception as exc:
        return "Fail-Closed", f"无法读取宏观状态：{exc}"


def nearest_price(states: pd.DataFrame, pair: str, timestamps: pd.Series) -> list[float]:
    source = states[states.pair == pair].sort_values("signal_ts")[["signal_ts", "price"]]
    query = pd.DataFrame({"signal_ts": timestamps.astype("int64")}).sort_values("signal_ts")
    return pd.merge_asof(query, source, on="signal_ts", direction="nearest")["price"].tolist()


def build_figure(
    results: Path, fomc: pd.DataFrame,
) -> tuple[go.Figure, dict[str, object], dict[str, int]]:
    states = pd.read_csv(results / "v21_states.csv.gz")
    intervals = pd.read_csv(results / "risk_intervals.csv")
    v21 = intervals[
        (intervals.scenario == "v21_gate") & (intervals.mechanism == "v21")
    ].copy()
    breakers = intervals[
        ((intervals.scenario == "mechanism_2") & (intervals.mechanism == "pair_loss"))
        | ((intervals.scenario == "mechanism_3") & (intervals.mechanism == "pair_drawdown"))
    ].copy()
    pairs = ["BTC-FDUSD", "ETH-FDUSD"]
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
        subplot_titles=pairs,
    )
    trace_groups: dict[str, list[int]] = {"v21": [], "fomc": [], "pair_breaker": []}
    shape_groups: dict[str, list[int]] = {"v21": [], "fomc": [], "pair_breaker": []}

    for row, pair in enumerate(pairs, 1):
        frame = states[states.pair == pair].sort_values("signal_ts")
        fig.add_trace(
            go.Scatter(
                x=utc(frame.signal_ts), y=frame.price, name=f"{pair} 价格",
                mode="lines", line={"width": 1.2, "color": "#38bdf8"},
                hovertemplate="%{x}<br>%{y:,.2f}<extra></extra>",
            ), row=row, col=1,
        )
        pair_v21 = v21[v21.pair == pair]
        for label, field, symbol in (
            ("进入", "start_ts", "triangle-down"),
            ("退出", "end_ts", "triangle-up"),
        ):
            ts = pair_v21[field].astype("int64")
            index = len(fig.data)
            fig.add_trace(
                go.Scatter(
                    x=utc(ts), y=nearest_price(states, pair, ts), mode="markers",
                    name=f"v21 {label} · {pair[:3]}", meta={"mechanism": "v21"},
                    marker={"symbol": symbol, "size": 9, "color": "#f59e0b"},
                ), row=row, col=1,
            )
            trace_groups["v21"].append(index)

        pair_breakers = breakers[breakers.pair == pair]
        if not pair_breakers.empty:
            ts = pair_breakers.start_ts.astype("int64")
            index = len(fig.data)
            fig.add_trace(
                go.Scatter(
                    x=utc(ts), y=nearest_price(states, pair, ts), mode="markers",
                    text=pair_breakers.mechanism, name=f"双熔断触发 · {pair[:3]}",
                    meta={"mechanism": "pair_breaker"},
                    marker={"symbol": "x", "size": 11, "color": "#ef4444"},
                    hovertemplate="%{x}<br>%{text}<extra></extra>",
                ), row=row, col=1,
            )
            trace_groups["pair_breaker"].append(index)

        for _, item in pair_v21.iterrows():
            fig.add_vrect(
                x0=pd.to_datetime(item.start_ts, unit="s", utc=True),
                x1=pd.to_datetime(item.end_ts, unit="s", utc=True),
                fillcolor=COLORS["v21"], opacity=1, line_width=0, row=row, col=1,
            )
            shape_groups["v21"].append(len(fig.layout.shapes) - 1)
        for _, item in pair_breakers.iterrows():
            fig.add_vrect(
                x0=pd.to_datetime(item.start_ts, unit="s", utc=True),
                x1=pd.to_datetime(item.end_ts, unit="s", utc=True),
                fillcolor=COLORS[str(item.mechanism)], opacity=1, line_width=0,
                row=row, col=1,
            )
            shape_groups["pair_breaker"].append(len(fig.layout.shapes) - 1)
        for _, item in fomc.iterrows():
            fig.add_vrect(
                x0=pd.to_datetime(item.start_ts, unit="s", utc=True),
                x1=pd.to_datetime(item.end_ts, unit="s", utc=True),
                fillcolor=COLORS["fomc"], opacity=1, line_width=0, row=row, col=1,
            )
            shape_groups["fomc"].append(len(fig.layout.shapes) - 1)

    fig.update_layout(
        template="plotly_dark", height=850,
        title="FDUSD Live Grid · 机制 1–3 进入/退出与生效区间",
        hovermode="x unified", margin={"t": 85, "l": 70, "r": 30, "b": 45},
        legend={"orientation": "h", "y": 1.04},
    )
    fig.update_yaxes(title_text="BTC 价格", row=1, col=1)
    fig.update_yaxes(title_text="ETH 价格", row=2, col=1)
    groups: dict[str, object] = {**trace_groups, "shapes": shape_groups}
    counts = {
        "v21_intervals": len(v21),
        "fomc_intervals": len(fomc),
        "pair_breaker_intervals": len(breakers),
    }
    return fig, groups, counts


def render(
    fig: go.Figure, groups: dict[str, object], counts: dict[str, int],
    macro_status: tuple[str, str], output: Path,
) -> None:
    plot = fig.to_html(full_html=False, include_plotlyjs=True, div_id="risk-chart")
    trace_groups = {key: value for key, value in groups.items() if key != "shapes"}
    controls = json.dumps({"traces": trace_groups, "shapes": groups["shapes"]})
    html = f"""<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'>
<title>FDUSD Live Grid 风控机制</title><style>
body{{margin:0;background:#07111f;color:#dbeafe;font:14px system-ui}}main{{max-width:1500px;margin:auto;padding:22px}}
.controls,.cards{{display:flex;gap:14px;flex-wrap:wrap;margin:12px 0}}label,.card{{background:#102033;border:1px solid #29415d;border-radius:10px;padding:12px 16px}}
label{{cursor:pointer;font-weight:700}}input{{margin-right:8px}}.card{{flex:1;min-width:250px}}h1{{margin-bottom:6px}}p{{color:#a9bfd7;line-height:1.55}}code{{color:#fbbf24}}
</style></head><body><main><h1>FDUSD Live Grid：三个线上风控机制</h1>
<p>三个勾选框彼此独立；取消后同时隐藏该机制的触发点和生效阴影。机制 1 和 3 使用冻结的 250 天离线轨迹展示时机，不代表当前 OCI 实时状态；FOMC 只显示传入的真实区间，不构造历史审批。</p>
<div class='controls'>
<label><input type='checkbox' data-mechanism='v21' checked>机制 1 · v21 买入门（{counts['v21_intervals']} 段）</label>
<label><input type='checkbox' data-mechanism='fomc' checked>机制 2 · FOMC 宏观门（{counts['fomc_intervals']} 段）</label>
<label><input type='checkbox' data-mechanism='pair_breaker' checked>机制 3 · 单对双熔断（{counts['pair_breaker_intervals']} 段）</label></div>
{plot}
<div class='cards'><section class='card'><h3>1 · v21 独立买入门</h3><p>BTC 和 ETH 各自推理，只暂停对应交易对的普通 BUY；SELL 和风险恢复单不受控制。橙色阴影为 Risk-Off，三角形标出进入和退出。</p></section>
<section class='card'><h3>2 · FOMC 宏观门</h3><p>每 5 秒同步；活动窗口内撤销两对 Grid 订单并停止 BUY/SELL。文件缺失、审批源异常或超过 150 秒会 Fail-Closed；窗口结束后立即刷新。当前快照：<code>{macro_status[0]}</code>（{macro_status[1]}）。</p></section>
<section class='card'><h3>3 · 单交易对双熔断</h3><p>相对 200 FDUSD 基准亏损 6 FDUSD，或相对该对历史峰值回撤 3% 时触发。只 halt 对应交易对、撤单并用 Taker 恢复初始库存；小于 5.25 FDUSD 留作 dust。状态持久化且不自动恢复。红/橙阴影来自两类独立触发回放。</p></section></div>
<script>const groups={controls};document.querySelectorAll('[data-mechanism]').forEach(cb=>cb.addEventListener('change',()=>{{
const key=cb.dataset.mechanism,visible=cb.checked;(groups.traces[key]||[]).forEach(i=>Plotly.restyle('risk-chart',{{visible:visible}},[i]));
const patch={{}};(groups.shapes[key]||[]).forEach(i=>patch[`shapes[${{i}}].visible`]=visible);Plotly.relayout('risk-chart',patch);
}}));</script></main></body></html>"""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--fomc-intervals", type=Path)
    parser.add_argument("--macro-gate", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    fomc = read_fomc(args.fomc_intervals)
    figure, groups, counts = build_figure(args.results, fomc)
    render(figure, groups, counts, current_macro_status(args.macro_gate), args.output)
    print(json.dumps({"output": str(args.output), **counts}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

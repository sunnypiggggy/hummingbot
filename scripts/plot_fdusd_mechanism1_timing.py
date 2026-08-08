#!/usr/bin/env python3
"""Plot exact mechanism-1 risk-off entry and recovery timing for BTC/ETH."""

from __future__ import annotations

import argparse
import html
import json
from dataclasses import fields
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from search_fdusd_ytd_risk_mechanisms import (
    PAIRS,
    TechnicalParameters,
    technical_observations,
)
from validate_grid_live import read_cache


PAIR_COLORS = {"BTC-FDUSD": "#0891B2", "ETH-FDUSD": "#7C3AED"}
RISK_OFF_COLOR = "#DC2626"
RECOVERY_COLOR = "#2563EB"
BASELINE_COLOR = "#64748B"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result-dir", type=Path,
        default=Path("results/backtests/fdusd_ytd_risk_mechanisms_1_3_final_20260731_1540utc"),
    )
    parser.add_argument("--cache-dir", type=Path, default=Path("data/backtesting_candles"))
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def transition_events(observations: pd.DataFrame, params: TechnicalParameters,
                      start_ts: int, end_ts: int) -> pd.DataFrame:
    """Return exact closed-bar entry/recovery events without future data."""
    active = False
    events = []
    for row in observations.itertuples(index=False):
        timestamp = int(row.timestamp)
        roc = float(row.roc_48h_pct)
        sqz = float(row.sqzmom_pct)
        improving = float(row.sqzmom) > float(row.sqzmom_previous)
        adverse = roc <= params.roc_trigger_pct and sqz <= params.sqz_trigger_pct
        event = None
        if not active and adverse:
            active = True
            event = "risk_off_entry"
        elif (
            active and not adverse and improving
            and roc >= params.roc_recovery_pct
            and sqz >= params.sqz_recovery_pct
        ):
            active = False
            event = "risk_off_exit"
        if event is not None and start_ts <= timestamp < end_ts:
            events.append({
                "timestamp": timestamp,
                "event": event,
                "price": float(row.close),
                "roc_48h_pct": roc,
                "sqzmom_pct": sqz,
                "sqzmom_improving": improving,
            })
    return pd.DataFrame(events)


def pause_intervals(events: pd.DataFrame, end_ts: int) -> list[tuple[int, int]]:
    intervals = []
    opened = None
    for row in events.sort_values("timestamp").itertuples(index=False):
        if row.event == "risk_off_entry":
            opened = int(row.timestamp)
        elif row.event == "risk_off_exit" and opened is not None:
            intervals.append((opened, int(row.timestamp)))
            opened = None
    if opened is not None:
        intervals.append((opened, end_ts))
    return intervals


def add_event_markers(fig: go.Figure, events: pd.DataFrame, row: int) -> None:
    for event, name, color, symbol in (
        ("risk_off_entry", "进入Risk-Off：停止新买单", RISK_OFF_COLOR, "triangle-down"),
        ("risk_off_exit", "退出Risk-Off：恢复买单", RECOVERY_COLOR, "triangle-up"),
    ):
        frame = events[events.event == event]
        fig.add_trace(go.Scatter(
            x=pd.to_datetime(frame.timestamp, unit="s", utc=True), y=frame.price,
            mode="markers", name=name,
            marker={"color": color, "symbol": symbol, "size": 13,
                    "line": {"color": "#111827", "width": 1}},
            customdata=frame[["roc_48h_pct", "sqzmom_pct"]],
            hovertemplate=("%{x|%Y-%m-%d %H:%M UTC}<br>价格=%{y:.6g}"
                           "<br>ROC48=%{customdata[0]:.2f}%"
                           "<br>SQZMOM=%{customdata[1]:.2f}%<extra></extra>"),
        ), row=row, col=1)


def pair_figure(pair: str, observations: pd.DataFrame, events: pd.DataFrame,
                intervals: list[tuple[int, int]], params: TechnicalParameters,
                curves: pd.DataFrame, trades: pd.DataFrame, split_ts: int) -> go.Figure:
    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.055,
        row_heights=[0.42, 0.16, 0.16, 0.26],
        subplot_titles=(
            f"{pair} 4小时收盘价与机制进出时机",
            "ROC48（%）", "标准化SQZMOM（%）", "单对权益：机制1与无风控基线",
        ),
    )
    time_axis = pd.to_datetime(observations.timestamp, unit="s", utc=True)
    fig.add_trace(go.Scatter(
        x=time_axis, y=observations.close, mode="lines", name=f"{pair}价格",
        line={"color": PAIR_COLORS[pair], "width": 1.5},
        hovertemplate="%{x|%Y-%m-%d %H:%M UTC}<br>收盘=%{y:.6g}<extra></extra>",
    ), row=1, col=1)
    add_event_markers(fig, events, 1)

    pair_trades = trades[(trades.scenario == "mechanism_1") & (trades.pair == pair)
                         & (trades.reason == "grid_fill")]
    for side, color, symbol in (("BUY", "#0F766E", "circle"), ("SELL", "#D97706", "x")):
        frame = pair_trades[pair_trades.side == side]
        fig.add_trace(go.Scatter(
            x=pd.to_datetime(frame.timestamp, unit="s", utc=True), y=frame.price,
            mode="markers", name=f"实际Grid {side}成交（点击显示）", visible="legendonly",
            marker={"color": color, "symbol": symbol, "size": 6, "opacity": 0.65},
            hovertemplate="%{x|%Y-%m-%d %H:%M UTC}<br>成交价=%{y:.6g}<extra></extra>",
        ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=time_axis, y=observations.roc_48h_pct, mode="lines", name="ROC48",
        line={"color": "#0F766E", "width": 1.3}, showlegend=False,
    ), row=2, col=1)
    fig.add_hline(y=params.roc_trigger_pct, line={"color": RISK_OFF_COLOR, "dash": "dash"},
                  annotation_text=f"进入阈值 {params.roc_trigger_pct:g}%", row=2, col=1)
    fig.add_hline(y=params.roc_recovery_pct, line={"color": RECOVERY_COLOR, "dash": "dot"},
                  annotation_text=f"退出阈值 {params.roc_recovery_pct:g}%", row=2, col=1)

    fig.add_trace(go.Bar(
        x=time_axis, y=observations.sqzmom_pct, name="SQZMOM", showlegend=False,
        marker_color=[RECOVERY_COLOR if value >= 0 else RISK_OFF_COLOR
                      for value in observations.sqzmom_pct], opacity=0.65,
    ), row=3, col=1)
    fig.add_hline(y=params.sqz_trigger_pct, line={"color": RISK_OFF_COLOR, "dash": "dash"},
                  annotation_text=f"进入阈值 {params.sqz_trigger_pct:g}%", row=3, col=1)
    fig.add_hline(y=params.sqz_recovery_pct, line={"color": RECOVERY_COLOR, "dash": "dot"},
                  annotation_text=f"退出阈值 {params.sqz_recovery_pct:g}%", row=3, col=1)

    for scenario, name, color in (
        ("baseline", "无风控基线", BASELINE_COLOR),
        ("mechanism_1", "机制1", PAIR_COLORS[pair]),
    ):
        for segment in ("development", "holdout"):
            frame = curves[(curves.scenario == scenario) & (curves.segment == segment)
                           & (curves.pair == pair) & curves.pair_equity.notna()]
            fig.add_trace(go.Scatter(
                x=pd.to_datetime(frame.timestamp, unit="s", utc=True), y=frame.pair_equity,
                mode="lines", name=name, legendgroup=name,
                showlegend=segment == "development",
                line={"color": color, "width": 1.4},
                hovertemplate="%{x|%Y-%m-%d %H:%M UTC}<br>权益=%{y:.2f} FDUSD<extra></extra>",
            ), row=4, col=1)

    for left, right in intervals:
        for chart_row in range(1, 5):
            fig.add_vrect(
                x0=pd.to_datetime(left, unit="s", utc=True),
                x1=pd.to_datetime(right, unit="s", utc=True),
                fillcolor=RISK_OFF_COLOR, opacity=0.09,
                line={"color": RISK_OFF_COLOR, "width": 0.8, "dash": "dot"},
                layer="below", row=chart_row, col=1,
            )
    split_time = pd.to_datetime(split_ts, unit="s", utc=True)
    fig.add_vline(x=split_time, line={"color": "#111827", "dash": "dash", "width": 1.2})
    fig.add_annotation(x=split_time, y=1.01, xref="x", yref="paper",
                       text="70/30锁参边界", showarrow=False, xanchor="left")
    fig.update_yaxes(title_text="价格", row=1, col=1)
    fig.update_yaxes(title_text="%", row=2, col=1)
    fig.update_yaxes(title_text="%", row=3, col=1)
    fig.update_yaxes(title_text="FDUSD", row=4, col=1)
    fig.update_xaxes(title_text="UTC时间", row=4, col=1, rangeslider_visible=True)
    fig.update_layout(
        title={
            "text": (
                f"机制1进出时机 · {pair}<br><sup>进入=停止该交易对新买单；退出=恢复买单；"
                "卖单始终保留，不代表清仓或卖出基础库存</sup>"
            ),
            "x": 0.02,
        },
        template="plotly_white", height=1160, margin={"t": 125, "l": 80, "r": 70, "b": 70},
        hovermode="x unified", font={"family": "Arial, Microsoft YaHei, sans-serif"},
        legend={"orientation": "h", "x": 0.01, "y": 1.04},
    )
    return fig


def add_local_times(events: pd.DataFrame, pair: str,
                    intervals: list[tuple[int, int]]) -> pd.DataFrame:
    result = events.copy()
    result["pair"] = pair
    result["utc_time"] = pd.to_datetime(result.timestamp, unit="s", utc=True).dt.strftime(
        "%Y-%m-%d %H:%M UTC"
    )
    result["shanghai_time"] = pd.to_datetime(
        result.timestamp, unit="s", utc=True,
    ).dt.tz_convert("Asia/Shanghai").dt.strftime("%Y-%m-%d %H:%M CST")
    durations = {}
    for left, right in intervals:
        durations[left] = (right - left) / 3600
    result["pause_hours"] = result.apply(
        lambda row: durations.get(int(row.timestamp)) if row.event == "risk_off_entry" else None,
        axis=1,
    )
    return result


def write_report(output_dir: Path, figures: list[tuple[str, go.Figure]], events: pd.DataFrame,
                 summary: dict) -> None:
    sections = []
    nav = []
    for index, (pair, figure) in enumerate(figures):
        nav.append(f'<a href="#{html.escape(pair)}">{html.escape(pair)}</a>')
        fragment = figure.to_html(
            full_html=False, include_plotlyjs=index == 0,
            config={"displaylogo": False, "responsive": True, "scrollZoom": True},
        )
        sections.append(f'<section id="{html.escape(pair)}">{fragment}</section>')
    table_rows = []
    for row in events.sort_values(["timestamp", "pair"]).itertuples(index=False):
        label = "进入Risk-Off / 停止新买单" if row.event == "risk_off_entry" else "退出Risk-Off / 恢复买单"
        duration = "" if pd.isna(row.pause_hours) else f"{row.pause_hours:.1f}"
        table_rows.append(
            f"<tr><td>{html.escape(row.pair)}</td><td>{html.escape(label)}</td>"
            f"<td>{html.escape(row.utc_time)}</td><td>{html.escape(row.shanghai_time)}</td>"
            f"<td>{row.price:.6g}</td><td>{row.roc_48h_pct:.2f}%</td>"
            f"<td>{row.sqzmom_pct:.2f}%</td><td>{duration}</td></tr>"
        )
    document = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>机制1进出时机</title><style>
body{{margin:0;background:#f8fafc;color:#1f2937;font-family:Arial,'Microsoft YaHei',sans-serif}}
main{{max-width:1500px;margin:auto;padding:24px 18px 60px}}header,section{{background:white;border:1px solid #cbd5e1;border-radius:10px}}
header{{padding:24px 28px;margin-bottom:22px}}section{{margin:22px 0;padding:8px}}nav a{{margin-right:20px;color:#1d4ed8;font-weight:700}}
table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{border-bottom:1px solid #e2e8f0;padding:8px;text-align:right}}
th:first-child,td:first-child,th:nth-child(2),td:nth-child(2){{text-align:left}}.note{{background:#fff7ed;border-left:4px solid #d97706;padding:12px}}
</style></head><body><main><header><h1>FDUSD Grid 机制1：进入/退出时机</h1>
<p>数据范围：{html.escape(summary['period']['start_iso'])} 至 {html.escape(summary['period']['end_iso'])}。红色背景为停止新买单区间。</p>
<p class="note"><b>重要：</b>这里的“进入”不是买入，“退出”也不是卖出或清仓。进入Risk-Off只停止对应交易对的新买单，原有卖单继续；退出Risk-Off表示恢复新买单。</p>
<nav>{' '.join(nav)}</nav><h2>全部切换时间</h2><table><thead><tr><th>交易对</th><th>动作</th><th>UTC</th><th>上海时间</th><th>价格</th><th>ROC48</th><th>SQZMOM</th><th>暂停小时</th></tr></thead>
<tbody>{''.join(table_rows)}</tbody></table></header>{''.join(sections)}</main></body></html>"""
    (output_dir / "report.html").write_text(document, encoding="utf-8")


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir or args.result_dir / "mechanism1_timing"
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = json.loads((args.result_dir / "summary.json").read_text(encoding="utf-8"))
    curves = pd.read_csv(
        args.result_dir / "locked_curves.csv",
        usecols=["timestamp", "pair_equity", "scenario", "pair", "segment"],
        low_memory=False,
    )
    trades = pd.read_csv(
        args.result_dir / "locked_events.csv",
        usecols=["scenario", "pair", "reason", "side", "timestamp", "price"],
        low_memory=False,
    )
    start_ts = int(summary["period"]["start_ts"])
    end_ts = int(summary["period"]["end_ts"])
    split_ts = int(summary["period"]["split_ts"])
    figures = []
    event_frames = []
    signal_frames = []
    for pair in PAIRS:
        raw = summary["locked_parameters"]["mechanism_1"][pair]
        params = TechnicalParameters(**{
            field.name: float(raw[field.name]) for field in fields(TechnicalParameters)
        })
        candles = read_cache(args.cache_dir / f"binance_{pair}_5m.csv")
        candles = candles[
            (candles.timestamp >= int(summary["period"]["warmup_start_ts"]))
            & (candles.timestamp < end_ts)
        ].reset_index(drop=True)
        _, observations_raw = technical_observations(candles)
        observations_all = pd.DataFrame(observations_raw)
        events = transition_events(observations_all, params, start_ts, end_ts)
        observations = observations_all[
            (observations_all.timestamp >= start_ts) & (observations_all.timestamp < end_ts)
        ].reset_index(drop=True)
        intervals = pause_intervals(events, end_ts)
        events = add_local_times(events, pair, intervals)
        observations["pair"] = pair
        event_frames.append(events)
        signal_frames.append(observations)
        figures.append((pair, pair_figure(
            pair, observations, events, intervals, params, curves, trades, split_ts,
        )))
    all_events = pd.concat(event_frames, ignore_index=True)
    all_signals = pd.concat(signal_frames, ignore_index=True)
    all_events.to_csv(output_dir / "mechanism1_transitions.csv", index=False)
    all_signals.to_csv(output_dir / "mechanism1_4h_signals.csv", index=False)
    write_report(output_dir, figures, all_events, summary)
    counts = {
        f"{pair}/{event}": int(count)
        for (pair, event), count in all_events.groupby(["pair", "event"]).size().items()
    }
    print(json.dumps({
        "output_dir": str(output_dir),
        "transition_counts": counts,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

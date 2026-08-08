#!/usr/bin/env python3
"""Plot BTC/ETH prices with exact XGBoost risk-gate entry and exit times."""

from __future__ import annotations

import argparse
import html
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


PAIRS = ("BTC-FDUSD", "ETH-FDUSD")
PAIR_LABEL = {"BTC-FDUSD": "BTC", "ETH-FDUSD": "ETH"}
PAIR_COLOR = {"BTC-FDUSD": "#2563EB", "ETH-FDUSD": "#C2417B"}
ENTRY_COLOR = "#C2410C"
RECOVERY_COLOR = "#2563EB"
RESET_COLOR = "#B7791F"
RISK_FILL = "rgba(194, 65, 12, 0.11)"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=Path("data/backtesting_candles"))
    parser.add_argument("--result-dir", type=Path, default=Path("results/backtests/xgboost_grid_risk_gate_v1"))
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def load_price(path: Path, start_ts: int, end_ts: int) -> pd.DataFrame:
    frame = pd.read_csv(path, usecols=["timestamp", "close"])
    frame["timestamp"] = frame.timestamp.astype("int64")
    frame = frame[(frame.timestamp >= start_ts) & (frame.timestamp < end_ts)].copy()
    frame["time"] = pd.to_datetime(frame.timestamp, unit="s", utc=True)
    return frame.sort_values("timestamp")


def nearest_prices(events: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return events.assign(close=pd.Series(dtype=float))
    return pd.merge_asof(
        events.sort_values("timestamp"),
        prices[["timestamp", "close"]].sort_values("timestamp"),
        on="timestamp", direction="backward",
    )


def event_table(events: pd.DataFrame) -> str:
    rows = []
    for item in events.sort_values(["timestamp", "pair"]).itertuples(index=False):
        label = {"enter": "进入 Risk-off", "recover": "模型恢复 BUY", "weekly_reinitialization": "周度重置"}[item.event]
        rows.append(
            "<tr>"
            f"<td>{html.escape(item.time_utc)}</td>"
            f"<td>{html.escape(item.pair)}</td>"
            f"<td>{html.escape(label)}</td>"
            f"<td>{item.probability:.6f}</td>" if np.isfinite(item.probability) else
            "<tr>"
            f"<td>{html.escape(item.time_utc)}</td>"
            f"<td>{html.escape(item.pair)}</td>"
            f"<td>{html.escape(label)}</td>"
            "<td>—</td>"
        )
        rows[-1] += f"<td>{item.entry_threshold:.6f}</td><td>{item.recovery_threshold:.6f}</td><td>{item.price:,.4f}</td></tr>"
    return "".join(rows)


def build_plot(cache_dir: Path, result_dir: Path, output: Path) -> tuple[Path, Path]:
    states = pd.read_csv(result_dir / "revalidation_risk_states.csv.gz")
    intervals = pd.read_csv(result_dir / "revalidation_risk_off_intervals.csv")
    grid = pd.read_csv(result_dir / "revalidation_grid_selections.csv")
    start_ts, end_ts = int(grid.test_start.min()), int(grid.test_end.max())
    prices = {
        pair: load_price(cache_dir / f"binance_{pair}_5m.csv", start_ts, end_ts)
        for pair in PAIRS
    }

    transition_rows = []
    for row in states[states.transition.isin(["enter", "recover"])].itertuples(index=False):
        transition_rows.append({
            "timestamp": int(row.signal_ts), "pair": row.pair,
            "event": str(row.transition), "probability": float(row.probability),
            "entry_threshold": float(row.entry_threshold),
            "recovery_threshold": float(row.recovery_threshold),
        })
    for row in intervals[intervals.end_reason == "weekly_reinitialization"].itertuples(index=False):
        prior = states[(states.fold == row.fold) & (states.pair == row.pair) & (states.signal_ts < row.end_ts)].sort_values("signal_ts").tail(1)
        transition_rows.append({
            "timestamp": int(row.end_ts), "pair": row.pair,
            "event": "weekly_reinitialization",
            "probability": float(prior.probability.iloc[0]) if not prior.empty else np.nan,
            "entry_threshold": float(prior.entry_threshold.iloc[0]) if not prior.empty else np.nan,
            "recovery_threshold": float(prior.recovery_threshold.iloc[0]) if not prior.empty else np.nan,
        })
    events = pd.DataFrame(transition_rows)
    event_frames = []
    for pair in PAIRS:
        item = nearest_prices(events[events.pair == pair], prices[pair])
        item["price"] = item.close
        event_frames.append(item.drop(columns=["close"]))
    events = pd.concat(event_frames, ignore_index=True).sort_values(["timestamp", "pair"])
    events["time"] = pd.to_datetime(events.timestamp, unit="s", utc=True)
    events["time_utc"] = events.time.dt.strftime("%Y-%m-%d %H:%M UTC")
    events.to_csv(result_dir / "plotly_risk_gate_transitions.csv", index=False)

    figure = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.10,
        specs=[[{"secondary_y": True}], [{"secondary_y": True}]],
        subplot_titles=("BTC-FDUSD", "ETH-FDUSD"),
    )
    for row_index, pair in enumerate(PAIRS, 1):
        price = prices[pair]
        pair_states = states[states.pair == pair].sort_values("signal_ts").copy()
        pair_states["time"] = pd.to_datetime(pair_states.signal_ts, unit="s", utc=True)
        figure.add_trace(go.Scattergl(
            x=price.time, y=price.close, mode="lines", name=f"{PAIR_LABEL[pair]} close",
            line={"color": PAIR_COLOR[pair], "width": 1.4}, legendgroup=pair,
            hovertemplate="%{x|%Y-%m-%d %H:%M UTC}<br>Close %{y:,.4f} FDUSD<extra></extra>",
        ), row=row_index, col=1, secondary_y=False)
        figure.add_trace(go.Scatter(
            x=pair_states.time, y=pair_states.probability, mode="lines", name=f"{PAIR_LABEL[pair]} risk probability",
            line={"color": ENTRY_COLOR, "width": 1.2, "dash": "dash"}, legendgroup=f"{pair}-risk",
            customdata=np.column_stack((pair_states.entry_threshold, pair_states.recovery_threshold, pair_states.transition)),
            hovertemplate=("%{x|%Y-%m-%d %H:%M UTC}<br>Probability %{y:.6f}"
                           "<br>Entry %{customdata[0]:.6f}<br>Recovery %{customdata[1]:.6f}"
                           "<br>State %{customdata[2]}<extra></extra>"),
        ), row=row_index, col=1, secondary_y=True)
        figure.add_trace(go.Scatter(
            x=pair_states.time, y=pair_states.entry_threshold, mode="lines", name=f"{PAIR_LABEL[pair]} entry threshold",
            line={"color": "#4B5563", "width": 1, "dash": "dot"}, legendgroup=f"{pair}-threshold",
            hovertemplate="%{x|%Y-%m-%d %H:%M UTC}<br>Entry threshold %{y:.6f}<extra></extra>",
        ), row=row_index, col=1, secondary_y=True)
        figure.add_trace(go.Scatter(
            x=pair_states.time, y=pair_states.recovery_threshold, mode="lines", name=f"{PAIR_LABEL[pair]} recovery threshold",
            line={"color": "#6B7280", "width": 1, "dash": "dashdot"}, legendgroup=f"{pair}-threshold",
            hovertemplate="%{x|%Y-%m-%d %H:%M UTC}<br>Recovery threshold %{y:.6f}<extra></extra>",
        ), row=row_index, col=1, secondary_y=True)

        for event, label, symbol, color in (
            ("enter", "Risk-off进入", "triangle-down", ENTRY_COLOR),
            ("recover", "模型恢复", "triangle-up", RECOVERY_COLOR),
            ("weekly_reinitialization", "周度重置", "x", RESET_COLOR),
        ):
            marked = events[(events.pair == pair) & (events.event == event)]
            figure.add_trace(go.Scatter(
                x=marked.time, y=marked.price, mode="markers", name=f"{PAIR_LABEL[pair]} {label}",
                marker={"symbol": symbol, "size": 11, "color": color, "line": {"color": "#111827", "width": 0.8}},
                legendgroup=f"{pair}-{event}",
                customdata=np.column_stack((marked.probability, marked.entry_threshold, marked.recovery_threshold, marked.time_utc)) if not marked.empty else None,
                hovertemplate=(f"<b>{label}</b><br>%{{customdata[3]}}<br>Price %{{y:,.4f}} FDUSD"
                               "<br>Probability %{customdata[0]:.6f}<br>Entry %{customdata[1]:.6f}"
                               "<br>Recovery %{customdata[2]:.6f}<extra></extra>"),
            ), row=row_index, col=1, secondary_y=False)

        for interval in intervals[intervals.pair == pair].itertuples(index=False):
            figure.add_vrect(
                x0=pd.to_datetime(interval.start_ts, unit="s", utc=True),
                x1=pd.to_datetime(interval.end_ts, unit="s", utc=True),
                fillcolor=RISK_FILL, line_width=0, layer="below",
                row=row_index, col=1,
            )

        figure.update_yaxes(title_text=f"{PAIR_LABEL[pair]} price (FDUSD)", row=row_index, col=1, secondary_y=False, showgrid=True, gridcolor="#E5E7EB")
        figure.update_yaxes(title_text="Risk probability", range=[0, 1], row=row_index, col=1, secondary_y=True, showgrid=False)

    figure.update_xaxes(
        title_text="UTC", row=2, col=1, showgrid=True, gridcolor="#F3F4F6",
        rangeslider={"visible": True, "thickness": 0.06},
        rangeselector={"buttons": [
            {"count": 7, "label": "7d", "step": "day", "stepmode": "backward"},
            {"count": 14, "label": "14d", "step": "day", "stepmode": "backward"},
            {"step": "all", "label": "全部"},
        ]},
    )
    figure.update_xaxes(showgrid=True, gridcolor="#F3F4F6", row=1, col=1)
    figure.update_layout(
        template="plotly_white", height=1050, hovermode="x unified",
        margin={"l": 70, "r": 70, "t": 150, "b": 80},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.035, "xanchor": "left", "x": 0, "font": {"size": 11}},
        font={"family": "Arial, Microsoft YaHei, sans-serif", "color": "#1F2937"},
        paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
    )

    plot = figure.to_html(
        full_html=False, include_plotlyjs=True, config={
            "responsive": True, "displaylogo": False, "scrollZoom": True,
            "modeBarButtonsToRemove": ["lasso2d", "select2d"],
        }, div_id="risk-gate-plot",
    )
    table_rows = event_table(events)
    page = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>XGBoost独立Risk-off门：BTC/ETH进入与退出时间</title>
<style>
html,body{{max-width:100%;overflow-x:hidden}}body{{margin:0;background:#fff;color:#1f2937;font-family:Arial,'Microsoft YaHei',sans-serif}}main{{max-width:1500px;margin:auto;padding:16px;overflow-x:hidden}}
h1{{font-size:26px;margin:8px 0 4px;overflow-wrap:anywhere}}.subtitle{{color:#4b5563;margin:0 0 12px}}
.note{{margin:8px 0 16px;padding:12px 14px;background:#fff7ed;border-left:4px solid #c2410c;line-height:1.6}}
#risk-gate-plot,.plotly-graph-div,.js-plotly-plot,.plot-container,.svg-container{{width:100%!important;max-width:100%!important;min-width:0!important}}
.table-wrap{{overflow:auto;border:1px solid #e5e7eb;border-radius:8px;margin-top:18px}}table{{border-collapse:collapse;width:100%;font-size:13px}}
th,td{{padding:8px 10px;border-bottom:1px solid #e5e7eb;text-align:right;white-space:nowrap}}th{{background:#f9fafb;position:sticky;top:0}}th:nth-child(1),td:nth-child(1),th:nth-child(2),td:nth-child(2),th:nth-child(3),td:nth-child(3){{text-align:left}}
@media(max-width:600px){{main{{padding:8px}}h1{{font-size:19px}}.subtitle,.note{{font-size:12px}}}}
</style></head><body><main>
<h1>XGBoost独立Risk-off门：BTC/ETH进入与退出时间</h1>
<p class="subtitle">固定再验证区间 2026-05-27 16:00 至 2026-07-26 16:00 UTC；橙色背景为Risk-off</p>
<div class="note"><b>读图：</b>橙色三角为进入Risk-off，蓝色三角为满足“两根连续低概率＋至少4小时”后的模型恢复，金色×为周度回放重置，并非模型提前恢复。Risk-off只暂停对应交易对普通BUY；价格、概率线和阈值都可在图例中单独开关。</div>
{plot}
<h2>精确进入与退出时间（UTC）</h2>
<div class="table-wrap"><table><thead><tr><th>UTC</th><th>Pair</th><th>Event</th><th>Probability</th><th>Entry threshold</th><th>Recovery threshold</th><th>Price</th></tr></thead><tbody>{table_rows}</tbody></table></div>
</main>
<script>
(function(){{
  const chart = document.getElementById('risk-gate-plot');
  function adapt(){{
    if (!chart || !window.Plotly) return;
    const mobile = window.innerWidth <= 600;
    Plotly.relayout(chart, mobile ? {{
      width: Math.max(window.innerWidth - 16, 320), height: 1180,
      'margin.l': 48, 'margin.r': 45, 'margin.t': 245, 'margin.b': 70,
      'legend.font.size': 9, 'legend.y': 1.035
    }} : {{
      width: Math.min(window.innerWidth - 32, 1500), height: 1050,
      'margin.l': 70, 'margin.r': 70, 'margin.t': 150, 'margin.b': 80,
      'legend.font.size': 11, 'legend.y': 1.035
    }});
    Plotly.Plots.resize(chart);
  }}
  window.addEventListener('load', adapt);
  let timer;
  window.addEventListener('resize', function(){{ clearTimeout(timer); timer=setTimeout(adapt,120); }});
}})();
</script></body></html>"""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(page, encoding="utf-8")
    return output, result_dir / "plotly_risk_gate_transitions.csv"


def main() -> int:
    args = parse_args()
    output = args.output or args.result_dir / "xgboost_risk_gate_entry_exit_plotly.html"
    html_path, csv_path = build_plot(args.cache_dir, args.result_dir, output)
    print(f"Wrote {html_path}")
    print(f"Wrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Plot the 180-day FDUSD Grid backtest with ROC/SQZMOM BUY-off windows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from grid_technical_gate import build_technical_buy_gate
from validate_grid_live import (
    Candidate,
    read_cache,
    roc_sqz_signal_from_klines,
    simulate,
    technical_buy_gate_timeline,
)


INTERVAL_SECONDS = 300
DAY_SECONDS = 86_400


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=180)
    parser.add_argument("--end-ts", type=int, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path("data/backtesting_candles"))
    parser.add_argument("--validation-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--maker-fee", type=float, default=0.0)
    parser.add_argument("--taker-fee", type=float, default=0.001)
    parser.add_argument("--roc-risk-off-pct", type=float, default=-8.0)
    parser.add_argument("--sqzmom-risk-off-pct", type=float, default=-3.0)
    parser.add_argument(
        "--include-hard-breaker", action="store_true",
        help="Model production pair/portfolio breakers. Default is disabled for exploratory backtests.",
    )
    return parser.parse_args()


def load_candles(cache_dir: Path, start_ts: int, end_ts: int) -> dict[str, pd.DataFrame]:
    candles = {}
    for pair in ("BTC-FDUSD", "ETH-FDUSD"):
        frame = read_cache(cache_dir / f"binance_{pair}_5m.csv")
        frame = frame[
            (frame.timestamp >= start_ts - 8 * DAY_SECONDS)
            & (frame.timestamp < end_ts)
        ].sort_values("timestamp").reset_index(drop=True)
        if frame.empty:
            raise RuntimeError(f"No cached candles for {pair}")
        candles[pair] = frame
    return candles


def signal_frame(
    frame: pd.DataFrame, *, roc_risk_off_pct: float = -8.0,
    sqzmom_risk_off_pct: float = -3.0,
) -> pd.DataFrame:
    source = frame.sort_values("timestamp").copy()
    source["bucket"] = (source.timestamp.astype("int64") // 14_400) * 14_400
    bars = source.groupby("bucket", sort=True).agg(
        open=("open", "first"), high=("high", "max"), low=("low", "min"),
        close=("close", "last"), rows=("close", "size"),
    ).reset_index()
    bars = bars[bars.rows == 48]
    klines: list[list] = []
    records = []
    active = False
    previous_bar_close_time = None
    previous_sqzmom_color = None
    for row in bars.itertuples(index=False):
        close_time = int(row.bucket) + 14_400 - 1
        klines.append([
            int(row.bucket) * 1000, row.open, row.high, row.low, row.close, 0,
            close_time * 1000,
        ])
        if len(klines) < 40:
            continue
        signal = roc_sqz_signal_from_klines(klines[-64:])
        gate = build_technical_buy_gate(
            signal,
            previously_active=active,
            previous_bar_close_time=previous_bar_close_time,
            previous_sqzmom_color=previous_sqzmom_color,
            roc_risk_off_pct=roc_risk_off_pct,
            sqzmom_risk_off_pct=sqzmom_risk_off_pct,
        )
        active = bool(gate["risk_off_active"])
        previous_bar_close_time = int(gate["last_evaluated_bar_close_time"])
        previous_sqzmom_color = str(gate["last_sqzmom_color"])
        records.append({
            "timestamp": close_time + 1,
            "roc_48h_pct": float(signal["roc_48h_pct"]),
            "sqzmom_pct": float(signal["sqzmom_pct"]),
            "sqzmom": float(signal["sqzmom"]),
            "sqzmom_previous": float(signal["sqzmom_previous"]),
            "sqzmom_improving": float(signal["sqzmom"]) > float(signal["sqzmom_previous"]),
            "sqzmom_above_zero": bool(signal["sqzmom_green"]),
            "sqzmom_color": str(signal["sqzmom_color"]),
            "buy_enabled": bool(gate["buy_enabled"]),
            "risk_off_condition": bool(gate["risk_off_condition"]),
            "entry_trigger": bool(gate["trigger"]),
            "recovery_signal": bool(gate["recover"]),
        })
    result = pd.DataFrame(records)
    result["datetime"] = pd.to_datetime(result.timestamp, unit="s", utc=True)
    return result


def decision_ranges(signals: pd.DataFrame, end_ts: int) -> pd.DataFrame:
    adverse = signals[~signals.buy_enabled].copy()
    if adverse.empty:
        return pd.DataFrame(columns=["start", "end", "bars", "min_roc_pct", "min_sqzmom_pct"])
    adverse["group"] = adverse.timestamp.diff().ne(14_400).cumsum()
    rows = []
    for _, group in adverse.groupby("group"):
        start = int(group.timestamp.iloc[0])
        last = int(group.timestamp.iloc[-1])
        rows.append({
            "start": pd.to_datetime(start, unit="s", utc=True),
            "end": pd.to_datetime(min(last + 14_400, end_ts), unit="s", utc=True),
            "bars": len(group),
            "min_roc_pct": float(group.roc_48h_pct.min()),
            "min_sqzmom_pct": float(group.sqzmom_pct.min()),
        })
    return pd.DataFrame(rows)


def add_trade_markers(fig: go.Figure, trades: pd.DataFrame, pair: str, row: int) -> None:
    subset = trades[trades.pair == pair]
    for side, color, symbol in (
        ("BUY", "#2563eb", "triangle-up"),
        ("SELL", "#d97706", "triangle-down"),
    ):
        points = subset[subset.side == side]
        if points.empty:
            continue
        fig.add_trace(go.Scatter(
            x=points.datetime, y=points.price, mode="markers", name=f"{pair} {side}",
            marker={"color": color, "symbol": symbol, "size": 8,
                    "line": {"color": "white", "width": 0.8}},
            customdata=points[["amount", "quote_notional", "reason"]],
            hovertemplate=(
                "%{x}<br>price=%{y:.4f}<br>amount=%{customdata[0]:.8f}"
                "<br>notional=%{customdata[1]:.2f} FDUSD<br>%{customdata[2]}<extra></extra>"
            ),
        ), row=row, col=1)


def build_figure(
    candles: dict[str, pd.DataFrame], signals: pd.DataFrame, ranges: pd.DataFrame,
    curve: pd.DataFrame, trades: pd.DataFrame, summary: dict,
) -> go.Figure:
    fig = make_subplots(
        rows=6, cols=1, shared_xaxes=True, vertical_spacing=0.025,
        row_heights=[0.23, 0.21, 0.13, 0.13, 0.18, 0.12],
        subplot_titles=(
            "BTC-FDUSD 5分钟收盘价与成交", "ETH-FDUSD 5分钟收盘价与成交",
            "ROC(12×4h)：与SQZMOM组合触发risk-off", "LazyBear SQZMOM：maroon恢复BUY",
            "Grid组合权益", "组合回撤",
        ),
    )
    for row, pair in ((1, "BTC-FDUSD"), (2, "ETH-FDUSD")):
        frame = candles[pair]
        fig.add_trace(go.Scatter(
            x=frame.datetime, y=frame.close, name=pair, mode="lines",
            line={"color": "#334155", "width": 1},
            hovertemplate="%{x}<br>%{y:.4f} FDUSD<extra></extra>",
        ), row=row, col=1)
        add_trade_markers(fig, trades, pair, row)
    fig.add_trace(go.Scatter(
        x=signals.datetime, y=signals.roc_48h_pct, name="ROC 48h",
        line={"color": "#2563eb", "width": 1.5},
    ), row=3, col=1)
    fig.add_hline(y=0, line={"color": "#64748b", "dash": "dot", "width": 1}, row=3, col=1)
    fig.add_hline(
        y=summary["roc_risk_off_pct"],
        line={"color": "#d97706", "dash": "dash", "width": 1},
        annotation_text=f"ROC risk-off {summary['roc_risk_off_pct']:g}%",
        annotation_position="bottom right", row=3, col=1,
    )
    sqz_colors = {
        "lime": "#22c55e", "green": "#15803d",
        "red": "#dc2626", "maroon": "#7f1d1d",
    }
    fig.add_trace(go.Bar(
        x=signals.datetime, y=signals.sqzmom_pct, name="SQZMOM histogram",
        marker_color=signals.sqzmom_color.map(sqz_colors),
        customdata=signals[["sqzmom_color", "risk_off_condition", "entry_trigger", "recovery_signal"]],
        hovertemplate=(
            "%{x}<br>SQZMOM=%{y:.3f}%<br>color=%{customdata[0]}"
            "<br>combined adverse=%{customdata[1]}<br>risk-off=%{customdata[2]}"
            "<br>maroon recovery=%{customdata[3]}<extra></extra>"
        ),
    ), row=4, col=1)
    fig.add_hline(y=0, line={"color": "#0f172a", "width": 1}, row=4, col=1)
    fig.add_hline(
        y=summary["sqzmom_risk_off_pct"],
        line={"color": "#d97706", "dash": "dash", "width": 1},
        annotation_text=f"SQZMOM risk-off {summary['sqzmom_risk_off_pct']:g}%",
        annotation_position="bottom right", row=4, col=1,
    )
    fig.add_trace(go.Scatter(
        x=curve.datetime, y=curve.equity, name="组合权益",
        line={"color": "#2563eb", "width": 1.8},
    ), row=5, col=1)
    fig.add_trace(go.Scatter(
        x=curve.datetime, y=curve.drawdown_pct * 100, name="回撤 %",
        fill="tozeroy", line={"color": "#d97706", "width": 1.2},
        fillcolor="rgba(217,119,6,0.16)",
    ), row=6, col=1)
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="markers", name="BUY OFF 决策范围",
        marker={"symbol": "square", "size": 11, "color": "rgba(217,119,6,0.24)"},
        hoverinfo="skip",
    ), row=1, col=1)
    for decision in ranges.itertuples(index=False):
        fig.add_vrect(
            x0=decision.start, x1=decision.end,
            fillcolor="rgba(217,119,6,0.14)", line_width=0,
        )
    risk_note = (
        f"硬熔断启用，权益曲线于{summary['simulation_end_utc']}停止"
        if summary["risk_breakers_enabled"]
        else "仅回测关闭硬熔断；生产Guard保持不变"
    )
    subtitle = (
        f"UTC；阴影=ROC≤{summary['roc_risk_off_pct']:g}%且"
        f"SQZMOM≤{summary['sqzmom_risk_off_pct']:g}%后BUY关闭；"
        f"SQZMOM首次maroon后恢复；"
        f"{summary['decision_windows']}个窗口 / {summary['decision_hours']:.2f}小时；"
        f"{risk_note}"
    )
    fig.update_layout(
        title={"text": f"FDUSD Grid 180天回测与技术决策范围<br><sup>{subtitle}</sup>", "x": 0.02},
        template="plotly_white", height=1500, hovermode="x unified",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.01, "x": 0},
        margin={"l": 75, "r": 35, "t": 125, "b": 55},
    )
    fig.update_yaxes(title_text="FDUSD", row=1, col=1)
    fig.update_yaxes(title_text="FDUSD", row=2, col=1)
    fig.update_yaxes(title_text="%", row=3, col=1)
    fig.update_yaxes(title_text="%", row=4, col=1)
    fig.update_yaxes(title_text="FDUSD", row=5, col=1)
    fig.update_yaxes(title_text="%", row=6, col=1)
    fig.update_xaxes(showspikes=True, spikemode="across", spikesnap="cursor")
    return fig


def main() -> int:
    args = parse_args()
    end_ts = args.end_ts // INTERVAL_SECONDS * INTERVAL_SECONDS
    start_ts = end_ts - args.days * DAY_SECONDS
    all_candles = load_candles(args.cache_dir, start_ts, end_ts)
    signals = signal_frame(
        all_candles["BTC-FDUSD"],
        roc_risk_off_pct=args.roc_risk_off_pct,
        sqzmom_risk_off_pct=args.sqzmom_risk_off_pct,
    )
    signals = signals[(signals.timestamp >= start_ts) & (signals.timestamp < end_ts)].reset_index(drop=True)
    ranges = decision_ranges(signals, end_ts)
    timeline = technical_buy_gate_timeline(
        all_candles["BTC-FDUSD"],
        roc_risk_off_pct=args.roc_risk_off_pct,
        sqzmom_risk_off_pct=args.sqzmom_risk_off_pct,
    )
    candles = {
        pair: frame[(frame.timestamp >= start_ts) & (frame.timestamp < end_ts)].reset_index(drop=True)
        for pair, frame in all_candles.items()
    }
    for frame in candles.values():
        frame["datetime"] = pd.to_datetime(frame.timestamp, unit="s", utc=True)

    selection = json.loads((args.validation_dir / "active_selection.json").read_text(encoding="utf-8"))
    params = selection["parameters"]
    candidate = Candidate(
        half_range=float(params["half_range"]),
        min_spread=float(params["minimum_spread"]),
        take_profit=float(params["take_profit"]),
        move_threshold=float(params["move_threshold"]),
        move_cooldown_seconds=int(params["min_grid_move_seconds"]),
    )
    trades: list[dict] = []
    result, curve, per_pair = simulate(
        candles, candidate, args.maker_fee, taker_fee=args.taker_fee,
        technical_buy_gate=timeline, trade_log=trades,
        risk_breakers_enabled=args.include_hard_breaker,
    )
    curve["datetime"] = pd.to_datetime(curve.timestamp, unit="s", utc=True)
    trade_frame = pd.DataFrame(trades)
    if not trade_frame.empty:
        trade_frame["datetime"] = pd.to_datetime(trade_frame.timestamp, unit="s", utc=True)
    simulation_end_ts = int(curve.timestamp.iloc[-1])
    active_ranges = ranges[ranges.start < pd.to_datetime(simulation_end_ts, unit="s", utc=True)]
    active_hours = 0.0
    for decision in active_ranges.itertuples(index=False):
        active_hours += max(
            0.0,
            (min(decision.end, pd.to_datetime(simulation_end_ts, unit="s", utc=True))
             - decision.start).total_seconds() / 3600,
        )
    summary = {
        "schema_version": "grid-roc-sqz-maroon-plot-v3",
        "period": {"start_ts": start_ts, "end_ts": end_ts, "days": args.days},
        "candidate": params,
        "decision_rule": (
            f"ROC48 <= {args.roc_risk_off_pct:g}% AND "
            f"SQZMOM <= {args.sqzmom_risk_off_pct:g}%"
        ),
        "roc_risk_off_pct": args.roc_risk_off_pct,
        "sqzmom_risk_off_pct": args.sqzmom_risk_off_pct,
        "recovery_rule": "while risk-off is active, first LazyBear maroon bar",
        "roc_role": "combined risk-off trigger",
        "decision_action": "cancel/suppress BUY only; preserve SELL",
        "risk_breakers_enabled": bool(args.include_hard_breaker),
        "production_risk_settings_changed": False,
        "decision_windows": int(len(ranges)),
        "decision_hours": float(ranges.apply(
            lambda row: (row["end"] - row["start"]).total_seconds() / 3600, axis=1
        ).sum()) if not ranges.empty else 0.0,
        "decision_windows_while_bot_running": int(len(active_ranges)),
        "decision_hours_while_bot_running": active_hours,
        "simulation_start_utc": str(curve.datetime.iloc[0]),
        "simulation_end_utc": str(curve.datetime.iloc[-1]),
        "liquidated_at_utc": str(curve.datetime.iloc[-1]) if result["liquidated"] else None,
        "backtest": result,
        "per_pair": per_pair,
        "trade_rows": len(trade_frame),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    signals.to_csv(args.output_dir / "technical_signals_4h.csv", index=False)
    ranges.to_csv(args.output_dir / "decision_ranges.csv", index=False)
    trade_frame.to_csv(args.output_dir / "trades.csv", index=False)
    curve.to_csv(args.output_dir / "equity_curve.csv", index=False)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    fig = build_figure(candles, signals, ranges, curve, trade_frame, summary)
    output = args.output_dir / "fdusd_grid_180d_roc_sqz_decisions.html"
    fig.write_html(output, include_plotlyjs=True, full_html=True)
    fig.write_json(args.output_dir / "fdusd_grid_180d_roc_sqz_decisions.plotly.json")
    june_ranges = ranges[
        (ranges.start < pd.Timestamp("2026-06-09", tz="UTC"))
        & (ranges.end >= pd.Timestamp("2026-06-01", tz="UTC"))
    ]
    if not june_ranges.empty:
        june = go.Figure(fig)
        june.update_xaxes(range=["2026-06-01", "2026-06-09"])
        june.update_layout(title={
            "text": (
                "FDUSD Grid：2026年6月2日下跌决策覆盖与恢复"
                "<br><sup>阴影期间仅BUY关闭；阴影结束=SQZMOM首次maroon</sup>"
            ),
            "x": 0.02,
        })
        june.add_vline(
            x=june_ranges.iloc[0]["end"], row=1, col=1,
            line={"color": "#2563eb", "dash": "dot", "width": 1.5},
            annotation_text="maroon: BUY恢复", annotation_position="top right",
        )
        june_output = args.output_dir / "fdusd_grid_june2_decision_zoom.html"
        june.write_html(june_output, include_plotlyjs=True, full_html=True)
        june.write_json(args.output_dir / "fdusd_grid_june2_decision_zoom.plotly.json")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Backtest the live BTC/ETH DCA with legacy and ML BUY risk gates.

BTC 4h signals are confirmed at bar close. A risk-off transition cancels all
active ladders and liquidates filled inventory at the next 5m open. No new DCA
executors are created until the selected gate recovers.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from backtest_dca_live_local import (
    BAR_SECONDS,
    PAIRS,
    SIDE_BUDGET,
    SPREADS,
    STOP_LOSS,
    TAKE_PROFIT,
    TOTAL_BUDGET,
    WEIGHTS,
    Executor,
    Fill,
    close_executor,
    create_executor,
    floor_five_minutes,
    load_window,
    mark_to_market,
)
from compare_roc_sqzmom_plotly import add_indicators, load_and_resample
from dca_live_common import LIVE_EXECUTOR_REFRESH_SECONDS, LIVE_TIME_LIMIT_SECONDS


SCENARIOS = ("baseline", "roc", "sqzmom", "combined", "v21", "v22")
FDUSD_PAIR_MAP = {"BTC-USDT": "BTC-FDUSD", "ETH-USDT": "ETH-FDUSD"}
V21_PAIR_MAP = FDUSD_PAIR_MAP  # compatibility for existing imports


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=180)
    parser.add_argument("--end", default="2026-07-27T02:00:00Z")
    parser.add_argument("--cache-dir", type=Path, default=Path("data/backtesting_candles"))
    parser.add_argument("--fee-rate", type=float, default=0.001, help="Fee per fill (0.001 = 0.10%%)")
    parser.add_argument("--risk-slippage-bps", type=float, default=2.0)
    parser.add_argument("--signal-timeframe", default="4h")
    parser.add_argument("--roc-length", type=int, default=12)
    parser.add_argument("--sqz-length", type=int, default=20)
    parser.add_argument(
        "--v21-states", type=Path,
        default=Path("results/backtests/fdusd_grid_v21_mechanisms_250d/v21_states.csv.gz"),
    )
    parser.add_argument(
        "--v22-states", type=Path,
        default=Path("results/backtests/xgboost_grid_long_risk_gate_v22_weekly_250d/application_bundle/risk_states.csv.gz"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/backtests/dca_momentum_guard_180d"))
    return parser.parse_args()


def build_signal_bars(path: Path, timeframe: str, roc_length: int, sqz_length: int) -> pd.DataFrame:
    bars, quality = load_and_resample(path, timeframe)
    bars = add_indicators(bars, sqz_length, 2.0, 1.5, roc_length)
    sqz_tradable = []
    enabled = True
    for row in bars.itertuples():
        if bool(row.sqz_sell):
            enabled = False
        elif bool(row.sqz_buy):
            enabled = True
        sqz_tradable.append(enabled)
    bars["sqzmom_enabled"] = sqz_tradable
    bars["roc_enabled"] = bars["roc"].gt(0).where(bars["roc"].notna(), True)
    bars["combined_enabled"] = bars["sqzmom_enabled"] & bars["roc_enabled"]
    bars["effective_time"] = bars.index + pd.Timedelta(timeframe)
    bars.attrs["quality"] = quality
    return bars


def gate_for_frame(
    frame: pd.DataFrame, signals: pd.DataFrame, scenario: str, *, pair: str = "",
    v21_states: pd.DataFrame | None = None, v22_states: pd.DataFrame | None = None,
) -> pd.Series:
    if scenario == "baseline":
        return pd.Series(True, index=frame.index, dtype=bool)
    if scenario in {"v21", "v22"}:
        states = v21_states if scenario == "v21" else v22_states
        if states is None or pair not in FDUSD_PAIR_MAP:
            raise ValueError(f"{scenario} scenario requires mapped pair states")
        source = states[states.pair.eq(FDUSD_PAIR_MAP[pair])].sort_values("signal_ts")
        if source.empty:
            raise ValueError(f"{scenario} states are missing {FDUSD_PAIR_MAP[pair]}")
        schedule = pd.Series(
            source.recommended_buy_enabled.astype(bool).to_numpy(),
            index=pd.to_datetime(source.signal_ts, unit="s", utc=True),
        )
        targets = pd.to_datetime(frame["timestamp"], unit="s", utc=True)
        values = schedule.reindex(targets, method="ffill").astype("boolean").fillna(False).astype(bool).to_numpy()
        return pd.Series(values, index=frame.index, dtype=bool)
    source_col = {"roc": "roc_enabled", "sqzmom": "sqzmom_enabled", "combined": "combined_enabled"}[scenario]
    schedule = pd.Series(signals[source_col].to_numpy(), index=signals["effective_time"])
    targets = pd.to_datetime(frame["timestamp"], unit="s", utc=True)
    values = schedule.reindex(targets, method="ffill").astype("boolean").fillna(True).astype(bool).to_numpy()
    return pd.Series(values, index=frame.index, dtype=bool)


def adjusted_risk_exit(mark: float, side: str, slippage_bps: float) -> float:
    slip = slippage_bps / 10_000
    return mark * (1 - slip if side == "BUY" else 1 + slip)


def run_pair_guarded(
    frame: pd.DataFrame,
    gate: pd.Series,
    pair: str,
    fee_rate: float,
    risk_slippage_bps: float,
    refresh_seconds: int = LIVE_EXECUTOR_REFRESH_SECONDS,
    time_limit_seconds: int = LIVE_TIME_LIMIT_SECONDS,
    guarded_sides: tuple[str, ...] = ("BUY", "SELL"),
    flatten_on_risk_off: bool = True,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    if len(frame) != len(gate):
        raise ValueError("gate and market frame lengths differ")
    active: dict[str, Executor | None] = {"BUY": None, "SELL": None}
    next_id = 1
    records: list[dict] = []
    equity_rows: list[dict] = []
    realized_pnl = 0.0
    counters = Counter()
    layer_fills = {"BUY": Counter(), "SELL": Counter()}

    def ensure_executors(timestamp: int, reference: float, gate_enabled: bool = True) -> None:
        nonlocal next_id
        for side in ("BUY", "SELL"):
            if not gate_enabled and side in guarded_sides:
                continue
            if active[side] is None:
                active[side] = create_executor(next_id, side, timestamp, reference, SPREADS)
                next_id += 1

    def record_close(executor: Executor, timestamp: int, mark: float, close_type: str) -> None:
        nonlocal realized_pnl
        record = close_executor(executor, timestamp, mark, fee_rate, close_type)
        if executor.has_position:
            records.append({"pair": pair, **record})
            realized_pnl += record["net_pnl"]
            counters[f"position_{close_type}"] += 1
        else:
            counters[f"empty_{close_type}"] += 1

    first = frame.iloc[0]
    if bool(gate.iloc[0]):
        ensure_executors(int(first.timestamp), float(first.close))
    else:
        ensure_executors(int(first.timestamp), float(first.close), gate_enabled=False)
    equity_rows.append({"timestamp": int(first.timestamp), "equity": TOTAL_BUDGET, "enabled": bool(gate.iloc[0])})
    previous_enabled = bool(gate.iloc[0])

    for offset, row in enumerate(frame.iloc[1:].itertuples(index=False), start=1):
        timestamp = int(row.timestamp)
        high, low, close, open_price = float(row.high), float(row.low), float(row.close), float(row.open)
        enabled = bool(gate.iloc[offset])

        if previous_enabled and not enabled:
            counters["risk_off_transitions"] += 1
            if flatten_on_risk_off:
                for side in guarded_sides:
                    executor = active[side]
                    if executor is not None:
                        exit_mark = adjusted_risk_exit(open_price, side, risk_slippage_bps)
                        record_close(executor, timestamp, exit_mark, "RISK_FLATTEN")
                        active[side] = None
        elif not previous_enabled and enabled:
            counters["risk_on_transitions"] += 1

        for side in ("BUY", "SELL"):
            side_enabled = enabled or side not in guarded_sides
            if side_enabled:
                executor = active[side]
                if executor is None:
                    continue
                for level, (price, quote) in enumerate(zip(executor.prices, executor.amounts_quote)):
                    if level in executor.filled_levels:
                        continue
                    touched = low <= price if side == "BUY" else high >= price
                    if touched:
                        executor.fills.append(Fill(level, price, quote, timestamp))
                        executor.filled_levels.add(level)
                        layer_fills[side][level + 1] += 1

                close_type = None
                if executor.has_position:
                    net_pnl_pct = (
                        executor.gross_pnl(close) - executor.open_quote * fee_rate
                    ) / executor.open_quote
                    age_from_fill = timestamp - executor.fills[0].timestamp
                    if net_pnl_pct <= -STOP_LOSS:
                        close_type = "STOP_LOSS"
                    elif net_pnl_pct > TAKE_PROFIT:
                        close_type = "TAKE_PROFIT"
                    elif age_from_fill >= time_limit_seconds:
                        close_type = "TIME_LIMIT"
                elif timestamp - executor.created_at >= refresh_seconds:
                    close_type = "REFRESH"

                if close_type:
                    record_close(executor, timestamp, close, close_type)
                    active[side] = None
        ensure_executors(timestamp, close, gate_enabled=enabled)

        unrealized = sum(
            mark_to_market(executor, close, fee_rate)
            for executor in active.values()
            if executor is not None
        )
        equity_rows.append({"timestamp": timestamp, "equity": TOTAL_BUDGET + realized_pnl + unrealized,
                            "enabled": enabled})
        previous_enabled = enabled

    final = frame.iloc[-1]
    for side in ("BUY", "SELL"):
        executor = active[side]
        if executor is not None:
            record_close(executor, int(final.timestamp), float(final.close), "END_OF_BACKTEST")
            active[side] = None
    equity_rows[-1]["equity"] = TOTAL_BUDGET + realized_pnl

    trades = pd.DataFrame(records)
    curve = pd.DataFrame(equity_rows)
    curve["datetime"] = pd.to_datetime(curve["timestamp"], unit="s", utc=True)
    curve = curve.set_index("datetime")
    returns = curve["equity"].pct_change().fillna(0)
    drawdown = curve["equity"] / curve["equity"].cummax() - 1
    annualizer = math.sqrt(365.25 * 24 * 12)
    positioned = len(trades)
    wins = int((trades["net_pnl"] > 0).sum()) if positioned else 0
    summary = {
        "pair": pair,
        "net_pnl_quote": realized_pnl,
        "return_on_190_pct": realized_pnl / TOTAL_BUDGET * 100,
        "max_drawdown_pct": drawdown.min() * 100,
        "sharpe_5m": returns.mean() / returns.std(ddof=0) * annualizer if returns.std(ddof=0) else 0.0,
        "positioned_executors": positioned,
        "win_rate_pct": wins / positioned * 100 if positioned else 0.0,
        "fees_quote": float(trades["fees"].sum()) if positioned else 0.0,
        "turnover_quote": float((trades["open_quote"] + trades["close_notional"]).sum()) if positioned else 0.0,
        "risk_flatten_positions": int(counters["position_RISK_FLATTEN"]),
        "risk_off_transitions": int(counters["risk_off_transitions"]),
        "risk_on_transitions": int(counters["risk_on_transitions"]),
        "stop_loss_positions": int(counters["position_STOP_LOSS"]),
        "buy_disabled_hours": float((~curve["enabled"]).sum() * BAR_SECONDS / 3600),
        "enabled_pct": curve["enabled"].mean() * 100,
        "close_types": {str(k): int(v) for k, v in sorted(Counter(trades["close_type"]).items())} if positioned else {},
        "layer_fills": {side: {str(level): int(layer_fills[side][level]) for level in range(1, 5)}
                        for side in ("BUY", "SELL")},
    }
    return summary, trades, curve


def combine_scenario(pair_curves: dict[str, pd.DataFrame], pair_summaries: list[dict], scenario: str) -> tuple[dict, pd.DataFrame]:
    combined = pd.concat([curve["equity"].rename(pair) for pair, curve in pair_curves.items()], axis=1)
    combined["equity"] = combined.sum(axis=1)
    combined["drawdown_pct"] = (combined["equity"] / combined["equity"].cummax() - 1) * 100
    combined["timestamp"] = (combined.index.astype("int64") // 10**9).astype("int64")
    returns = combined["equity"].pct_change().fillna(0)
    annualizer = math.sqrt(365.25 * 24 * 12)
    summary = {
        "scenario": scenario,
        "combined_net_pnl_quote": combined["equity"].iloc[-1] - TOTAL_BUDGET * len(PAIRS),
        "return_on_380_pct": (combined["equity"].iloc[-1] / (TOTAL_BUDGET * len(PAIRS)) - 1) * 100,
        "combined_max_drawdown_pct": combined["drawdown_pct"].min(),
        "combined_sharpe_5m": returns.mean() / returns.std(ddof=0) * annualizer if returns.std(ddof=0) else 0.0,
        "fees_quote": sum(item["fees_quote"] for item in pair_summaries),
        "positioned_executors": sum(item["positioned_executors"] for item in pair_summaries),
        "risk_flatten_positions": sum(item["risk_flatten_positions"] for item in pair_summaries),
        "enabled_pct": sum(item["enabled_pct"] for item in pair_summaries) / len(pair_summaries),
    }
    return summary, combined


def guard_periods(signals: pd.DataFrame, start: datetime, end: datetime, column: str) -> pd.DataFrame:
    series = pd.Series(signals[column].to_numpy(), index=signals["effective_time"])
    boundaries = series.loc[(series.index >= start) & (series.index < end)]
    initial = bool(series.loc[series.index < start].iloc[-1]) if (series.index < start).any() else True
    periods = []
    state = initial
    state_start = pd.Timestamp(start) if not state else None
    for ts, new_state in boundaries.items():
        new_state = bool(new_state)
        if new_state == state:
            continue
        if state is False and state_start is not None:
            periods.append({"start": state_start, "end": ts, "hours": (ts - state_start).total_seconds() / 3600})
            state_start = None
        elif state is True and new_state is False:
            state_start = ts
        state = new_state
    if state is False and state_start is not None:
        final_ts = pd.Timestamp(end)
        periods.append({"start": state_start, "end": final_ts,
                        "hours": (final_ts - state_start).total_seconds() / 3600})
    return pd.DataFrame(periods)


def make_figure(signals: pd.DataFrame, curves: dict[str, pd.DataFrame], periods: pd.DataFrame,
                start: datetime, end: datetime, fee_rate: float, slippage_bps: float) -> go.Figure:
    plot_signals = signals[(signals["effective_time"] >= start) & (signals["effective_time"] < end)].copy()
    x = plot_signals["effective_time"]
    fig = make_subplots(rows=5, cols=1, shared_xaxes=True, vertical_spacing=0.025,
                        row_heights=[0.32, 0.15, 0.14, 0.24, 0.15],
                        subplot_titles=("BTC-USDT 4小时价格与组合 Risk-Off 区间", "SQZMOM",
                                        "ROC(12)", "BTC+ETH DCA 组合净值（初始 380 USDT）", "组合回撤"))
    fig.add_trace(go.Candlestick(x=x, open=plot_signals.open, high=plot_signals.high, low=plot_signals.low,
                                 close=plot_signals.close, name="BTC-USDT",
                                 increasing_line_color="#2563eb", decreasing_line_color="#9ca3af"), row=1, col=1)
    for row in periods.itertuples(index=False):
        fig.add_vrect(x0=row.start, x1=row.end, fillcolor="#d97706", opacity=0.12, line_width=0, row=1, col=1)
    hist_colors = {"lime": "#2563eb", "green": "#93c5fd", "red": "#d97706", "maroon": "#fed7aa"}
    fig.add_trace(go.Bar(x=x, y=plot_signals.sqz_val, marker_color=plot_signals.sqz_color.map(hist_colors),
                         name="SQZMOM", showlegend=False), row=2, col=1)
    fig.add_hline(y=0, line_color="#374151", line_width=1, row=2, col=1)
    fig.add_trace(go.Scatter(x=x, y=plot_signals.roc, name="ROC(12)",
                             line=dict(color="#d97706", width=1.5)), row=3, col=1)
    fig.add_hline(y=0, line_color="#374151", line_width=1, row=3, col=1)
    palette = {"baseline": "#4b5563", "roc": "#d97706", "sqzmom": "#2563eb",
               "combined": "#111827", "v21": "#7c3aed", "v22": "#c2410c"}
    labels = {"baseline": "无指标风控", "roc": "ROC 风控", "sqzmom": "SQZMOM 风控", "combined": "组合风控"}
    labels["v21"] = "v21 BUY risk gate"
    labels["v22"] = "v22 weekly BUY risk gate"
    dashes = {"baseline": "dot", "roc": "solid", "sqzmom": "solid",
              "combined": "dash", "v21": "solid", "v22": "longdash"}
    for scenario in SCENARIOS:
        # Keep calculations at 5m, but use hourly endpoints for a compact,
        # responsive interactive chart without changing any reported metric.
        curve = curves[scenario].resample("1h").last()
        fig.add_trace(go.Scatter(x=curve.index, y=curve.equity, name=labels[scenario],
                                 line=dict(color=palette[scenario], width=2, dash=dashes[scenario])), row=4, col=1)
        fig.add_trace(go.Scatter(x=curve.index, y=curve.drawdown_pct, name=f"{labels[scenario]}回撤",
                                 line=dict(color=palette[scenario], width=1.5, dash=dashes[scenario]),
                                 showlegend=False), row=5, col=1)
    subtitle = (f"{start:%Y-%m-%d %H:%M} 至 {end:%Y-%m-%d %H:%M} UTC；BTC 4h 收盘确认，下一根5m执行；"
                f"手续费 {fee_rate:.2%}/边，风控清仓滑点 {slippage_bps:.0f}bp")
    fig.update_layout(title=dict(text=f"Live DCA：ROC / SQZMOM 清仓暂停风控（180天）<br><sup>{subtitle}</sup>", x=0.02),
                      template="plotly_white", height=1280, hovermode="x unified",
                      legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
                      margin=dict(l=70, r=35, t=110, b=55), xaxis_rangeslider_visible=False)
    fig.update_yaxes(title_text="USDT", row=1, col=1)
    fig.update_yaxes(title_text="动量", row=2, col=1)
    fig.update_yaxes(title_text="%", row=3, col=1)
    fig.update_yaxes(title_text="USDT", row=4, col=1)
    fig.update_yaxes(title_text="%", row=5, col=1)
    fig.update_xaxes(showspikes=True, spikemode="across", spikesnap="cursor", spikethickness=1)
    return fig


def main() -> int:
    args = parse_args()
    if args.days != 180:
        raise ValueError("this comparison is fixed to 180 days")
    end = floor_five_minutes(datetime.fromisoformat(args.end.replace("Z", "+00:00")))
    start = end - timedelta(days=args.days)
    frames = {pair: load_window(args.cache_dir / f"{symbol}_5m.csv", start, end)
              for pair, symbol in PAIRS.items()}
    signals = build_signal_bars(args.cache_dir / "BTCUSDT_5m.csv", args.signal_timeframe,
                                args.roc_length, args.sqz_length)
    v21_states = pd.read_csv(args.v21_states)
    v22_states = pd.read_csv(args.v22_states)
    required_v21 = {"pair", "signal_ts", "recommended_buy_enabled"}
    if not required_v21.issubset(v21_states.columns):
        raise ValueError(f"v21 state file is missing {sorted(required_v21 - set(v21_states.columns))}")
    if not required_v21.issubset(v22_states.columns):
        raise ValueError(f"v22 state file is missing {sorted(required_v21 - set(v22_states.columns))}")
    scenario_summaries, pair_summaries, all_trades, combined_curves = [], [], [], {}
    for scenario in SCENARIOS:
        curves = {}
        scenario_pair_summaries = []
        for pair, frame in frames.items():
            gate = gate_for_frame(
                frame, signals, scenario, pair=pair, v21_states=v21_states, v22_states=v22_states,
            )
            ml_buy_only = scenario in {"v21", "v22"}
            summary, trades, curve = run_pair_guarded(frame, gate, pair, args.fee_rate,
                                                       args.risk_slippage_bps,
                                                       guarded_sides=("BUY",) if ml_buy_only else ("BUY", "SELL"),
                                                       flatten_on_risk_off=not ml_buy_only)
            summary["scenario"] = scenario
            trades.insert(0, "scenario", scenario)
            scenario_pair_summaries.append(summary)
            all_trades.append(trades)
            curves[pair] = curve
        aggregate, combined = combine_scenario(curves, scenario_pair_summaries, scenario)
        scenario_summaries.append(aggregate)
        pair_summaries.extend(scenario_pair_summaries)
        combined_curves[scenario] = combined
        print(f"{scenario}: pnl={aggregate['combined_net_pnl_quote']:+.4f}, "
              f"return={aggregate['return_on_380_pct']:+.3f}%, dd={aggregate['combined_max_drawdown_pct']:.3f}%",
              flush=True)

    output = args.output_dir / end.strftime("%Y%m%dT%H%M%SZ")
    output.mkdir(parents=True, exist_ok=True)
    summary_frame = pd.DataFrame(scenario_summaries)
    summary_frame.to_csv(output / "summary.csv", index=False)
    pd.DataFrame(pair_summaries).to_json(output / "pair_summary.json", orient="records", indent=2)
    pd.concat(all_trades, ignore_index=True).to_csv(output / "positioned_executors.csv", index=False)
    periods = guard_periods(signals, start, end, "combined_enabled")
    periods.to_csv(output / "combined_risk_off_periods.csv", index=False)
    audit = {
        "window": {"start": start.isoformat(), "end": end.isoformat(), "days": args.days},
        "data_source": "local Binance spot BTCUSDT/ETHUSDT 5m klines",
        "signal": {
            "market": "BTC-USDT", "timeframe": args.signal_timeframe,
            "roc": f"ROC({args.roc_length}) < 0 risk-off; > 0 recovery",
            "sqzmom": "transition to red risk-off; transition to lime recovery",
            "combined": "either off triggers; both must recover",
            "v21": "frozen FDUSD v21 BUY-only state mapped by BTC/ETH base asset to USDT DCA",
            "v22": "signed weekly FDUSD v22 counterfactual BUY-only state mapped by BTC/ETH base asset to USDT DCA",
            "execution": "confirmed 4h close, next 5m open risk flatten",
            "pine_compatibility": "supplied code uses multKC=1.5 for BB stdev; BB mult input remains unused",
        },
        "live_dca": {
            "pairs": list(PAIRS), "budget_per_pair": TOTAL_BUDGET, "side_budget": SIDE_BUDGET,
            "spreads": SPREADS, "weights": WEIGHTS,
            "refresh_seconds": LIVE_EXECUTOR_REFRESH_SECONDS,
            "time_limit_seconds_from_first_fill": LIVE_TIME_LIMIT_SECONDS,
            "take_profit": TAKE_PROFIT, "stop_loss_on_partial_fills": STOP_LOSS,
        },
        "costs": {"fee_rate_per_fill": args.fee_rate, "risk_flatten_slippage_bps": args.risk_slippage_bps},
        "signal_data_quality": signals.attrs.get("quality", {}),
    }
    (output / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    fig = make_figure(signals, combined_curves, periods, start, end, args.fee_rate, args.risk_slippage_bps)
    fig.write_html(output / "dca_momentum_guard_180d.html", include_plotlyjs=True, full_html=True)
    print(summary_frame.round(4).to_string(index=False))
    print(json.dumps({"output": str(output.resolve()), "orders_submitted": False}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

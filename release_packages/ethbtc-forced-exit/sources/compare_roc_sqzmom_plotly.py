#!/usr/bin/env python3
"""Compare a zero-cross ROC strategy with LazyBear SQZMOM on OHLCV data.

The SQZMOM implementation intentionally preserves the supplied Pine v4 code,
including its use of ``multKC`` (rather than ``mult``) for Bollinger Bands.
Signals are calculated at bar close and executed at the next bar open.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


@dataclass
class Metrics:
    total_return_pct: float
    annualized_return_pct: float
    max_drawdown_pct: float
    sharpe: float
    trades: int
    win_rate_pct: float
    profit_factor: float
    exposure_pct: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=Path("data/backtesting_candles/BTCUSDT_5m.csv"))
    parser.add_argument("--symbol", default="BTC-USDT")
    parser.add_argument("--timeframe", default="4h")
    parser.add_argument("--days", type=int, default=180)
    parser.add_argument("--roc-length", type=int, default=12)
    parser.add_argument("--sqz-length", type=int, default=20)
    parser.add_argument("--bb-mult", type=float, default=2.0)
    parser.add_argument("--kc-mult", type=float, default=1.5)
    parser.add_argument("--fee-bps", type=float, default=10.0)
    parser.add_argument("--slippage-bps", type=float, default=2.0)
    parser.add_argument("--initial-capital", type=float, default=10_000.0)
    parser.add_argument("--output-dir", type=Path, default=Path("results/backtests/roc_vs_sqzmom_180d"))
    return parser.parse_args()


def load_and_resample(path: Path, timeframe: str) -> tuple[pd.DataFrame, dict]:
    raw = pd.read_csv(path)
    required = {"timestamp", "open", "high", "low", "close", "volume"}
    missing = sorted(required.difference(raw.columns))
    if missing:
        raise ValueError(f"Missing required CSV columns: {missing}")
    timestamp_unit = "ms" if raw["timestamp"].abs().median() > 10_000_000_000 else "s"
    raw["datetime"] = pd.to_datetime(raw["timestamp"], unit=timestamp_unit, utc=True)
    raw = raw.sort_values("datetime").drop_duplicates("datetime", keep="last").set_index("datetime")
    expected_minutes = pd.Timedelta(timeframe).total_seconds() / 60
    source_minutes = raw.index.to_series().diff().dropna().median().total_seconds() / 60
    expected_source_rows = int(round(expected_minutes / source_minutes))
    bars = raw.resample(timeframe, label="left", closed="left", origin="epoch").agg(
        open=("open", "first"), high=("high", "max"), low=("low", "min"),
        close=("close", "last"), volume=("volume", "sum"), source_rows=("close", "count")
    )
    bars = bars.dropna(subset=["open", "high", "low", "close"])
    if len(bars) and bars.iloc[-1]["source_rows"] < expected_source_rows:
        bars = bars.iloc[:-1]
    partial = bars["source_rows"] < expected_source_rows
    quality = {
        "raw_rows": int(len(raw)),
        "duplicate_timestamps_removed": int(pd.read_csv(path, usecols=["timestamp"]).duplicated().sum()),
        "timestamp_unit": timestamp_unit,
        "source_interval_minutes": float(source_minutes),
        "expected_source_rows_per_bar": expected_source_rows,
        "resampled_bars": int(len(bars)),
        "partial_boundary_bars_retained": int(partial.iloc[[0]].sum()) if len(partial) else 0,
        "incomplete_non_boundary_bars": int(partial.iloc[1:].sum()) if len(partial) > 1 else 0,
    }
    return bars, quality


def pine_linreg_last(values: np.ndarray) -> float:
    if np.isnan(values).any():
        return np.nan
    n = len(values)
    x = np.arange(n, dtype=float)
    x_mean = (n - 1) / 2
    slope = np.dot(x - x_mean, values - values.mean()) / np.dot(x - x_mean, x - x_mean)
    return float(values.mean() + slope * ((n - 1) - x_mean))


def add_indicators(df: pd.DataFrame, length: int, bb_mult: float, kc_mult: float, roc_length: int) -> pd.DataFrame:
    out = df.copy()
    basis = out["close"].rolling(length).mean()
    # Pine stdev() uses the population standard deviation (ddof=0). Preserve the
    # supplied script's multKC multiplier; bb_mult is retained for auditability.
    dev = kc_mult * out["close"].rolling(length).std(ddof=0)
    out["upper_bb"] = basis + dev
    out["lower_bb"] = basis - dev
    ma = out["close"].rolling(length).mean()
    prev_close = out["close"].shift(1)
    true_range = pd.concat(
        [(out["high"] - out["low"]), (out["high"] - prev_close).abs(), (out["low"] - prev_close).abs()], axis=1
    ).max(axis=1)
    range_ma = true_range.rolling(length).mean()
    out["upper_kc"] = ma + range_ma * kc_mult
    out["lower_kc"] = ma - range_ma * kc_mult
    out["sqz_on"] = (out["lower_bb"] > out["lower_kc"]) & (out["upper_bb"] < out["upper_kc"])
    out["sqz_off"] = (out["lower_bb"] < out["lower_kc"]) & (out["upper_bb"] > out["upper_kc"])
    out["no_sqz"] = ~(out["sqz_on"] | out["sqz_off"])
    midpoint = ((out["high"].rolling(length).max() + out["low"].rolling(length).min()) / 2 + ma) / 2
    source = out["close"] - midpoint
    out["sqz_val"] = source.rolling(length).apply(pine_linreg_last, raw=True)
    prior = out["sqz_val"].shift(1)
    out["sqz_color"] = np.select(
        [(out["sqz_val"] > 0) & (out["sqz_val"] > prior),
         out["sqz_val"] > 0,
         out["sqz_val"] < prior],
        ["lime", "green", "red"], default="maroon"
    )
    lime = out["sqz_color"].eq("lime")
    red = out["sqz_color"].eq("red")
    out["sqz_buy"] = lime & ~lime.shift(1, fill_value=False)
    out["sqz_sell"] = red & ~red.shift(1, fill_value=False)
    out["roc"] = out["close"].pct_change(roc_length) * 100
    out["roc_buy"] = (out["roc"] > 0) & (out["roc"].shift(1) <= 0)
    out["roc_sell"] = (out["roc"] < 0) & (out["roc"].shift(1) >= 0)
    out.attrs["bb_mult_input_unused_by_supplied_pine"] = bb_mult
    return out


def backtest(df: pd.DataFrame, buy_col: str, sell_col: str, initial: float, fee_bps: float, slippage_bps: float):
    fee = fee_bps / 10_000
    slip = slippage_bps / 10_000
    cash, qty, in_position = initial, 0.0, False
    equities, positions, trades, pending = [], [], [], None
    entry = None
    for ts, row in df.iterrows():
        if pending == "buy" and not in_position:
            fill = float(row.open) * (1 + slip)
            entry_cash = cash
            qty = cash * (1 - fee) / fill
            cash, in_position = 0.0, True
            entry = {"entry_time": ts, "entry_price": fill, "entry_equity": entry_cash}
        elif pending == "sell" and in_position:
            fill = float(row.open) * (1 - slip)
            cash = qty * fill * (1 - fee)
            trade_return = cash / entry["entry_equity"] - 1
            trades.append({**entry, "exit_time": ts, "exit_price": fill, "return_pct": trade_return * 100, "forced_exit": False})
            qty, in_position, entry = 0.0, False, None
        pending = None
        equity = cash if not in_position else qty * float(row.close)
        equities.append(equity)
        positions.append(int(in_position))
        if in_position and bool(row[sell_col]):
            pending = "sell"
        elif not in_position and bool(row[buy_col]):
            pending = "buy"
    if in_position:
        fill = float(df.iloc[-1].close) * (1 - slip)
        cash = qty * fill * (1 - fee)
        trades.append({**entry, "exit_time": df.index[-1], "exit_price": fill,
                       "return_pct": (cash / entry["entry_equity"] - 1) * 100, "forced_exit": True})
        equities[-1] = cash
    equity = pd.Series(equities, index=df.index, dtype=float)
    position = pd.Series(positions, index=df.index, dtype=int)
    return equity, position, pd.DataFrame(trades)


def metrics(equity: pd.Series, position: pd.Series, trades: pd.DataFrame, initial: float, timeframe: str) -> Metrics:
    returns = equity.pct_change().fillna(0)
    years = max((equity.index[-1] - equity.index[0]).total_seconds() / (365.25 * 86400), 1 / 365.25)
    total = equity.iloc[-1] / initial - 1
    annualized = (equity.iloc[-1] / initial) ** (1 / years) - 1
    drawdown = equity / equity.cummax() - 1
    periods_per_year = pd.Timedelta(days=365.25) / pd.Timedelta(timeframe)
    sharpe = returns.mean() / returns.std(ddof=0) * math.sqrt(periods_per_year) if returns.std(ddof=0) else 0.0
    wins = trades["return_pct"] > 0 if not trades.empty else pd.Series(dtype=bool)
    gross_profit = trades.loc[trades.return_pct > 0, "return_pct"].sum() if not trades.empty else 0
    gross_loss = -trades.loc[trades.return_pct < 0, "return_pct"].sum() if not trades.empty else 0
    pf = gross_profit / gross_loss if gross_loss else (math.inf if gross_profit else 0.0)
    return Metrics(total * 100, annualized * 100, drawdown.min() * 100, float(sharpe), len(trades),
                   float(wins.mean() * 100) if len(wins) else 0.0, float(pf), position.mean() * 100)


def buy_hold(df: pd.DataFrame, initial: float, fee_bps: float, slippage_bps: float) -> pd.Series:
    fee, slip = fee_bps / 10_000, slippage_bps / 10_000
    qty = initial * (1 - fee) / (float(df.iloc[0].open) * (1 + slip))
    equity = qty * df["close"]
    equity.iloc[-1] = qty * float(df.iloc[-1].close) * (1 - slip) * (1 - fee)
    return equity


def make_figure(df, roc_eq, sqz_eq, hold_eq, roc_trades, sqz_trades, subtitle: str) -> go.Figure:
    fig = make_subplots(rows=5, cols=1, shared_xaxes=True, vertical_spacing=0.025,
                        row_heights=[0.38, 0.16, 0.14, 0.20, 0.12],
                        subplot_titles=("BTC-USDT 4小时价格与交易点", "SQZMOM（原 PineScript 逻辑）",
                                        "ROC(12)", "策略净值（起始=10,000）", "回撤"))
    fig.add_trace(go.Candlestick(x=df.index, open=df.open, high=df.high, low=df.low, close=df.close,
                                 name="BTC-USDT", increasing_line_color="#2563eb", decreasing_line_color="#9ca3af"), row=1, col=1)
    for trades, name, color, symbol in [(roc_trades, "ROC", "#d97706", "triangle-up"),
                                         (sqz_trades, "SQZMOM", "#2563eb", "diamond")]:
        if not trades.empty:
            fig.add_trace(go.Scatter(x=trades.entry_time, y=trades.entry_price, mode="markers", name=f"{name} 买入",
                                     marker=dict(color=color, symbol=symbol, size=9, line=dict(color="white", width=1))), row=1, col=1)
            fig.add_trace(go.Scatter(x=trades.exit_time, y=trades.exit_price, mode="markers", name=f"{name} 卖出",
                                     marker=dict(color=color, symbol="x", size=9)), row=1, col=1)
    hist_colors = {"lime": "#2563eb", "green": "#93c5fd", "red": "#d97706", "maroon": "#fed7aa"}
    fig.add_trace(go.Bar(x=df.index, y=df.sqz_val, name="SQZMOM", marker_color=df.sqz_color.map(hist_colors), showlegend=False), row=2, col=1)
    fig.add_hline(y=0, line_color="#374151", line_width=1, row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df.roc, name="ROC(12)", line=dict(color="#d97706", width=1.5)), row=3, col=1)
    fig.add_hline(y=0, line_color="#374151", line_width=1, row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=roc_eq, name="ROC 策略", line=dict(color="#d97706", width=2)), row=4, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=sqz_eq, name="SQZMOM 策略", line=dict(color="#2563eb", width=2)), row=4, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=hold_eq, name="买入持有", line=dict(color="#4b5563", width=1.5, dash="dot")), row=4, col=1)
    for eq, name, color in [(roc_eq, "ROC 回撤", "#d97706"), (sqz_eq, "SQZMOM 回撤", "#2563eb")]:
        dd = (eq / eq.cummax() - 1) * 100
        fig.add_trace(go.Scatter(x=df.index, y=dd, name=name, line=dict(color=color, width=1.5)), row=5, col=1)
    fig.update_layout(title=dict(text=f"ROC 与 Squeeze Momentum：过去180天对比<br><sup>{subtitle}</sup>", x=0.02),
                      template="plotly_white", height=1250, hovermode="x unified", barmode="relative",
                      legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
                      margin=dict(l=70, r=35, t=110, b=55), xaxis_rangeslider_visible=False)
    fig.update_yaxes(title_text="USDT", row=1, col=1)
    fig.update_yaxes(title_text="动量", row=2, col=1)
    fig.update_yaxes(title_text="%", row=3, col=1)
    fig.update_yaxes(title_text="USDT", row=4, col=1)
    fig.update_yaxes(title_text="%", row=5, col=1)
    fig.update_xaxes(showspikes=True, spikemode="across", spikesnap="cursor", spikethickness=1)
    return fig


def main() -> None:
    args = parse_args()
    bars, quality = load_and_resample(args.csv, args.timeframe)
    full = add_indicators(bars, args.sqz_length, args.bb_mult, args.kc_mult, args.roc_length)
    end = full.index[-1]
    start = end - pd.Timedelta(days=args.days)
    df = full.loc[full.index >= start].copy()
    if len(df) < 30:
        raise ValueError("Not enough bars in requested comparison window")
    roc_eq, roc_pos, roc_trades = backtest(df, "roc_buy", "roc_sell", args.initial_capital, args.fee_bps, args.slippage_bps)
    sqz_eq, sqz_pos, sqz_trades = backtest(df, "sqz_buy", "sqz_sell", args.initial_capital, args.fee_bps, args.slippage_bps)
    hold_eq = buy_hold(df, args.initial_capital, args.fee_bps, args.slippage_bps)
    hold_pos = pd.Series(1, index=df.index)
    summary = pd.DataFrame({
        "ROC(12)": asdict(metrics(roc_eq, roc_pos, roc_trades, args.initial_capital, args.timeframe)),
        "SQZMOM": asdict(metrics(sqz_eq, sqz_pos, sqz_trades, args.initial_capital, args.timeframe)),
        "Buy & Hold": asdict(metrics(hold_eq, hold_pos, pd.DataFrame(), args.initial_capital, args.timeframe)),
    }).T
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_dir / "summary.csv", index_label="strategy")
    roc_trades.to_csv(args.output_dir / "roc_trades.csv", index=False)
    sqz_trades.to_csv(args.output_dir / "sqzmom_trades.csv", index=False)
    audit = {
        "source": str(args.csv), "symbol": args.symbol, "timeframe": args.timeframe,
        "window_start_utc": str(df.index[0]), "window_end_utc": str(df.index[-1]), "bars": len(df),
        "signal_execution": "bar-close signal, next-bar open execution",
        "fee_bps_per_side": args.fee_bps, "slippage_bps_per_side": args.slippage_bps,
        "roc_rule": f"buy cross above zero / sell cross below zero, ROC({args.roc_length})",
        "sqzmom_rule": "buy on transition to lime / sell on transition to red",
        "sqzmom_note": "Supplied Pine uses multKC for BB stdev; bb_mult input is unused",
        "data_quality": quality,
    }
    (args.output_dir / "audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    subtitle = (f"{df.index[0]:%Y-%m-%d %H:%M} 至 {df.index[-1]:%Y-%m-%d %H:%M} UTC；"
                f"信号后下一根开盘成交；单边手续费 {args.fee_bps:.0f}bp + 滑点 {args.slippage_bps:.0f}bp")
    fig = make_figure(df, roc_eq, sqz_eq, hold_eq, roc_trades, sqz_trades, subtitle)
    fig.write_html(args.output_dir / "roc_vs_sqzmom_180d.html", include_plotlyjs=True, full_html=True)
    print(summary.round(3).to_string())
    print(f"\nWindow: {df.index[0]} -> {df.index[-1]} ({len(df)} bars)")
    print(f"Output: {(args.output_dir / 'roc_vs_sqzmom_180d.html').resolve()}")


if __name__ == "__main__":
    main()

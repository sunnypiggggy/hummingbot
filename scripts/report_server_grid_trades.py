#!/usr/bin/env python
"""Build an offline HTML report for paper-grid trade fills.

The report reads a downloaded Hummingbot SQLite database, aligns each fill with
cached Binance 5-minute candles, and marks every buy and sell on a per-pair
price chart. Candle files are updated incrementally only when their tail does
not cover the requested reporting interval.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import time
from pathlib import Path
from typing import Iterable

import pandas as pd
import plotly.graph_objects as go
import requests
from plotly.io import to_html
from plotly.offline.offline import get_plotlyjs
from plotly.subplots import make_subplots


BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"
CANDLE_COLUMNS = [
    "timestamp", "open", "high", "low", "close", "volume", "quote_asset_volume",
    "n_trades", "taker_buy_base_volume", "taker_buy_quote_volume",
]
DECIMAL_SCALE = 1_000_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, required=True, help="Downloaded Hummingbot SQLite database.")
    parser.add_argument("--cache-dir", type=Path, default=Path("data/backtesting_candles"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--interval", default="5m")
    parser.add_argument("--padding-minutes", type=int, default=60)
    parser.add_argument("--start", help="Optional UTC start timestamp, e.g. 2026-07-09T15:00:23Z.")
    parser.add_argument("--pairs", nargs="*", help="Optional pairs to render, including pairs with no fills.")
    return parser.parse_args()


def read_trades(database: Path) -> pd.DataFrame:
    query = """
        SELECT timestamp, symbol, trade_type, order_type, price, amount,
               trade_fee, trade_fee_in_quote, order_id, exchange_trade_id
        FROM TradeFill
        ORDER BY timestamp, symbol, order_id
    """
    with sqlite3.connect(database) as connection:
        trades = pd.read_sql_query(query, connection)
    if trades.empty:
        raise RuntimeError("No TradeFill records found in the database.")

    for column in ("price", "amount", "trade_fee_in_quote"):
        trades[column] = pd.to_numeric(trades[column], errors="coerce") / DECIMAL_SCALE
    trades["timestamp"] = pd.to_datetime(trades["timestamp"], unit="ms", utc=True)
    trades["notional_usdt"] = trades["price"] * trades["amount"]
    trades["fee_rate"] = trades["trade_fee"].map(extract_fee_rate)
    return trades


def extract_fee_rate(raw_fee: str) -> float:
    try:
        return float(json.loads(raw_fee).get("percent", 0.0))
    except (TypeError, ValueError, json.JSONDecodeError):
        return 0.0


def cache_path(cache_dir: Path, pair: str, interval: str) -> Path:
    return cache_dir / f"binance_{pair}_{interval}.csv"


def read_cache(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=CANDLE_COLUMNS)
    candles = pd.read_csv(path)
    if "timestamp" not in candles:
        raise RuntimeError(f"Candle cache is missing timestamp: {path}")
    return candles[CANDLE_COLUMNS].copy()


def fetch_klines(pair: str, interval: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    rows: list[list[object]] = []
    cursor = start_ms
    while cursor <= end_ms:
        response = requests.get(
            BINANCE_KLINES_URL,
            params={
                "symbol": pair.replace("-", ""),
                "interval": interval,
                "startTime": cursor,
                "endTime": end_ms,
                "limit": 1000,
            },
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()
        if not payload:
            break
        rows.extend(payload)
        next_cursor = int(payload[-1][0]) + 1
        if next_cursor <= cursor:
            break
        cursor = next_cursor
        if len(payload) < 1000:
            break
        time.sleep(0.1)

    if not rows:
        return pd.DataFrame(columns=CANDLE_COLUMNS)
    frame = pd.DataFrame(rows).iloc[:, :10]
    frame.columns = CANDLE_COLUMNS
    numeric_columns = [column for column in CANDLE_COLUMNS if column != "timestamp"]
    frame["timestamp"] = pd.to_numeric(frame["timestamp"], errors="coerce") / 1000
    frame[numeric_columns] = frame[numeric_columns].apply(pd.to_numeric, errors="coerce")
    return frame.dropna(subset=["timestamp", "close"])


def load_candles(pair: str, interval: str, cache_dir: Path, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    path = cache_path(cache_dir, pair, interval)
    cached = read_cache(path)
    start_seconds = start.timestamp()
    end_seconds = end.timestamp()
    latest_cached = float(cached["timestamp"].max()) if not cached.empty else 0.0
    earliest_cached = float(cached["timestamp"].min()) if not cached.empty else float("inf")

    requested: list[pd.DataFrame] = []
    if earliest_cached > start_seconds:
        requested.append(fetch_klines(pair, interval, int(start_seconds * 1000), int((earliest_cached - 1) * 1000)))
    if latest_cached < end_seconds:
        requested.append(fetch_klines(pair, interval, int(max(start_seconds, latest_cached + 1) * 1000), int(end_seconds * 1000)))

    if requested:
        frames = [frame for frame in [cached, *requested] if not frame.empty]
        if not frames:
            raise RuntimeError(f"Binance returned no candle data for {pair}.")
        merged = pd.concat(frames, ignore_index=True)
        merged = merged.drop_duplicates(subset="timestamp", keep="last").sort_values("timestamp")
        cache_dir.mkdir(parents=True, exist_ok=True)
        merged.to_csv(path, index=False)
        cached = merged

    candles = cached[(cached["timestamp"] >= start_seconds) & (cached["timestamp"] <= end_seconds)].copy()
    if candles.empty:
        raise RuntimeError(f"No {interval} candles available for {pair} in the reporting window.")
    candles["timestamp"] = pd.to_datetime(candles["timestamp"], unit="s", utc=True)
    return candles


def pair_metrics(trades: pd.DataFrame, last_price: float) -> dict[str, float]:
    side = trades["trade_type"].str.upper()
    cash_flow = (trades.loc[side == "SELL", "notional_usdt"].sum()
                 - trades.loc[side == "BUY", "notional_usdt"].sum()
                 - trades["trade_fee_in_quote"].fillna(0).sum())
    inventory = (trades.loc[side == "BUY", "amount"].sum()
                 - trades.loc[side == "SELL", "amount"].sum())
    return {
        "trades": int(len(trades)),
        "fees": float(trades["trade_fee_in_quote"].fillna(0).sum()),
        "inventory": float(inventory),
        "mark_to_market": float(cash_flow + inventory * last_price),
        "buy_notional": float(trades.loc[side == "BUY", "notional_usdt"].sum()),
        "sell_notional": float(trades.loc[side == "SELL", "notional_usdt"].sum()),
    }


def make_pair_figure(pair: str, candles: pd.DataFrame, trades: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=candles["timestamp"], y=candles["close"], mode="lines", name="5m close",
        line={"color": "#4c78a8", "width": 1.4},
        hovertemplate="%{x|%Y-%m-%d %H:%M UTC}<br>close=%{y:,.8g}<extra></extra>",
    ))
    for side, color, marker in (("BUY", "#2a9d8f", "triangle-up"), ("SELL", "#e76f51", "triangle-down")):
        fills = trades[trades["trade_type"].str.upper() == side]
        fig.add_trace(go.Scatter(
            x=fills["timestamp"], y=fills["price"], mode="markers", name=side.title(),
            marker={"color": color, "symbol": marker, "size": 10, "line": {"color": "#ffffff", "width": 0.8}},
            customdata=fills[["amount", "notional_usdt", "trade_fee_in_quote", "fee_rate", "order_id"]],
            hovertemplate=("%{x|%Y-%m-%d %H:%M:%S UTC}<br>" + side.lower() + " price=%{y:,.8g}<br>"
                           "amount=%{customdata[0]:,.8g}<br>notional=%{customdata[1]:,.2f} USDT<br>"
                           "fee=%{customdata[2]:,.4f} USDT (%{customdata[3]:.3%})<br>"
                           "order=%{customdata[4]}<extra></extra>"),
        ))
    metrics = pair_metrics(trades, float(candles.iloc[-1]["close"]))
    fig.update_layout(
        title={"text": f"{pair}: {metrics['trades']} fills, mark-to-market {metrics['mark_to_market']:+,.2f} USDT", "x": 0.01},
        template="plotly_white", height=430, hovermode="closest", margin={"l": 70, "r": 30, "t": 60, "b": 55},
        legend={"orientation": "h", "y": 1.02, "x": 0.01},
    )
    fig.update_xaxes(title_text="UTC time", showgrid=True, gridcolor="#eceff1")
    fig.update_yaxes(title_text="Price (USDT)", showgrid=True, gridcolor="#eceff1", tickformat=",.8g")
    return fig


def write_report(output_dir: Path, trades: pd.DataFrame, candles_by_pair: dict[str, pd.DataFrame]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, object]] = []
    fragments: list[str] = []
    for pair in sorted(candles_by_pair):
        pair_trades = trades[trades["symbol"] == pair].copy()
        candles = candles_by_pair[pair]
        metrics = pair_metrics(pair_trades, float(candles.iloc[-1]["close"]))
        summaries.append({"pair": pair, **metrics, "last_price": float(candles.iloc[-1]["close"])})
        fragments.append(to_html(make_pair_figure(pair, candles, pair_trades), full_html=False, include_plotlyjs=False))

    summary = pd.DataFrame(summaries).sort_values("mark_to_market", ascending=False)
    summary.to_csv(output_dir / "pair_summary.csv", index=False)
    exports = trades.copy()
    exports["timestamp"] = exports["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    exports.to_csv(output_dir / "trade_fills.csv", index=False)

    first_trade = trades["timestamp"].min()
    last_trade = trades["timestamp"].max()
    total_fees = summary["fees"].sum()
    total_mtm = summary["mark_to_market"].sum()
    rows = "".join(
        "<tr>" + "".join(f"<td>{value}</td>" for value in (
            item.pair, f"{item.trades:,}", f"{item.mark_to_market:+,.2f}", f"{item.fees:,.2f}", f"{item.inventory:,.8g}",
        )) + "</tr>"
        for item in summary.itertuples()
    )
    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Paper Grid Trade Report</title><style>
body {{ margin: 0; background: #f7f9fb; color: #17212b; font: 14px/1.5 Arial, sans-serif; }}
main {{ max-width: 1320px; margin: auto; padding: 28px 18px 48px; }} h1 {{ margin: 0; }}
.summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 12px; margin: 18px 0; }}
.metric, .table-wrap, .chart {{ background: #fff; border: 1px solid #dfe5ea; border-radius: 6px; padding: 14px; }}
.metric strong {{ display: block; font-size: 24px; }} .chart {{ margin-top: 16px; padding: 4px; }}
table {{ width: 100%; border-collapse: collapse; }} th, td {{ padding: 8px; border-bottom: 1px solid #edf0f2; text-align: right; }} th:first-child, td:first-child {{ text-align: left; }}
.note {{ color: #53606c; }} @media (prefers-color-scheme: dark) {{ body {{ background:#111820; color:#e7edf3; }} .metric,.table-wrap,.chart {{ background:#17212b; border-color:#35424e; }} th,td {{ border-color:#2b3640; }} }}
</style><script>{get_plotlyjs()}</script></head><body><main>
<h1>Paper Grid Trade Report</h1>
<p class="note">Source: downloaded Hummingbot <code>TradeFill</code> records and local Binance 5-minute candle cache. All times are UTC. Mark-to-market is trade cash flow plus net filled inventory valued at the last displayed candle; it is not the paper account balance.</p>
<div class="summary"><div class="metric"><span>Trade window</span><strong>{first_trade:%Y-%m-%d} to {last_trade:%Y-%m-%d}</strong></div>
<div class="metric"><span>Total fills</span><strong>{len(trades):,}</strong></div><div class="metric"><span>Net filled-flow MTM</span><strong>{total_mtm:+,.2f} USDT</strong></div>
<div class="metric"><span>Recorded fees</span><strong>{total_fees:,.2f} USDT</strong></div></div>
<section class="table-wrap"><h2>By trading pair</h2><table><thead><tr><th>Pair</th><th>Fills</th><th>Filled-flow MTM (USDT)</th><th>Fees (USDT)</th><th>Net inventory</th></tr></thead><tbody>{rows}</tbody></table></section>
<section><h2>Price and every fill</h2>{''.join(f'<div class="chart">{fragment}</div>' for fragment in fragments)}</section>
</main></body></html>"""
    (output_dir / "server_grid_trade_report.html").write_text(html, encoding="utf-8")


def main() -> None:
    args = parse_args()
    trades = read_trades(args.database)
    if args.start:
        start_filter = pd.Timestamp(args.start)
        if start_filter.tzinfo is None:
            start_filter = start_filter.tz_localize("UTC")
        else:
            start_filter = start_filter.tz_convert("UTC")
        trades = trades[trades["timestamp"] >= start_filter].copy()
        if trades.empty:
            raise RuntimeError(f"No fills found on or after {start_filter.isoformat()}.")
    padding = pd.Timedelta(minutes=args.padding_minutes)
    start = trades["timestamp"].min() - padding
    end = trades["timestamp"].max() + padding
    pairs = sorted(set(args.pairs or []).union(trades["symbol"].unique()))
    candles_by_pair = {
        pair: load_candles(pair, args.interval, args.cache_dir, start, end)
        for pair in pairs
    }
    write_report(args.output_dir, trades, candles_by_pair)
    print(f"Wrote {args.output_dir / 'server_grid_trade_report.html'}")


if __name__ == "__main__":
    main()

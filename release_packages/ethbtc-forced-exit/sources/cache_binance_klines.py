from __future__ import annotations

import argparse
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests


API_URL = "https://api.binance.com/api/v3/klines"
COLUMNS = [
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "trades",
    "taker_base",
    "taker_quote",
    "ignore",
]


def parse_time(value: str) -> datetime:
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None:
        result = result.replace(tzinfo=timezone.utc)
    return result.astimezone(timezone.utc)


def download(symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
    cursor = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)
    rows: list[list] = []
    session = requests.Session()
    while cursor < end_ms:
        response = session.get(
            API_URL,
            params={
                "symbol": symbol,
                "interval": "5m",
                "startTime": cursor,
                "endTime": end_ms - 1,
                "limit": 1000,
            },
            timeout=30,
        )
        response.raise_for_status()
        batch = response.json()
        if not batch:
            break
        rows.extend(batch)
        next_cursor = int(batch[-1][0]) + 5 * 60 * 1000
        if next_cursor <= cursor:
            raise RuntimeError("Binance kline cursor did not advance")
        cursor = next_cursor
        time.sleep(0.05)
    return pd.DataFrame(rows, columns=COLUMNS)


def update_cache(path: Path, symbol: str, start: datetime, end: datetime) -> int:
    existing = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=COLUMNS)
    missing_start = start
    if not existing.empty:
        last = pd.to_datetime(existing["timestamp"].max(), unit="ms", utc=True)
        missing_start = max(start, last.to_pydatetime() + timedelta(minutes=5))
    if missing_start >= end:
        return 0
    fresh = download(symbol, missing_start, end)
    combined = pd.concat([existing, fresh], ignore_index=True)
    combined = combined.drop_duplicates("timestamp", keep="last").sort_values("timestamp")
    path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(path, index=False)
    return len(fresh)


def main() -> int:
    parser = argparse.ArgumentParser(description="Incremental Binance 5m cache")
    parser.add_argument("--symbols", default="BTCUSDT,ETHUSDT")
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/backtesting_candles")
    )
    args = parser.parse_args()
    start = parse_time(args.start)
    end = parse_time(args.end)
    for symbol in (value.strip().upper() for value in args.symbols.split(",")):
        path = args.output_dir / f"{symbol}_5m.csv"
        count = update_cache(path, symbol, start, end)
        print(f"{symbol}: added {count} rows -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

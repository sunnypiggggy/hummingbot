"""Acquire public, closed 5m candles into an isolated offline experiment."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
FIELDS = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
START = int(pd.Timestamp('2025-10-17T00:00:00Z').timestamp())
END = int(pd.Timestamp('2026-10-17T00:00:00Z').timestamp())
TRAIN_START = int(pd.Timestamp('2025-07-11T00:00:00Z').timestamp())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def get(session, path, params):
    for attempt in range(6):
        try:
            response = session.get('https://api.binance.com' + path, params=params, timeout=30)
            if response.status_code == 429 or response.status_code >= 500:
                time.sleep(min(2 ** attempt, 15))
                continue
            response.raise_for_status()
            return response.json()
        except requests.RequestException:
            if attempt == 5:
                raise
            time.sleep(min(2 ** attempt, 15))
    raise RuntimeError('public Binance reads exhausted')


def validate(frame, start, end):
    expected = np.arange(start, end, 300, dtype=np.int64)
    if not np.array_equal(frame.timestamp.to_numpy(), expected):
        raise ValueError('missing, duplicate, unclosed, or unordered 5m candle')
    x = frame[FIELDS[1:]]
    if not np.isfinite(x).all().all() or (x.iloc[:, :4] <= 0).any().any() or (x.volume < 0).any():
        raise ValueError('invalid candle values')
    if not ((frame.high >= frame[['open', 'close']].max(axis=1)) &
            (frame.low <= frame[['open', 'close']].min(axis=1)) & (frame.high >= frame.low)).all():
        raise ValueError('invalid OHLC bounds')


def acquire(item, directory):
    pair, symbol = item
    is_grid = pair.endswith('FDUSD')
    start = TRAIN_START if is_grid else START
    target = directory / (f'binance_{pair}_5m.csv' if is_grid else f'{symbol}_5m.csv')
    seed = ROOT / 'data/backtesting_candles' / (f'binance_{pair}_5m.csv' if is_grid else f'{symbol}_5m.csv')
    input_file = target if target.exists() else seed
    frame = pd.read_csv(input_file, usecols=FIELDS) if input_file.exists() else pd.DataFrame(columns=FIELDS)
    if not frame.empty:
        frame.timestamp = pd.to_numeric(frame.timestamp).astype('int64')
        if frame.timestamp.max() > 10_000_000_000:
            frame.timestamp //= 1000
        frame = frame[(frame.timestamp >= start) & (frame.timestamp < END)].copy()
        for column in FIELDS[1:]:
            frame[column] = pd.to_numeric(frame[column])
    if not frame.empty and (frame.timestamp.duplicated().any() or not frame.timestamp.diff().dropna().eq(300).all()):
        raise ValueError(f'{pair} existing cache has gaps or duplicates')
    ranges = [(start, END)] if frame.empty else [(start, int(frame.timestamp.min())), (int(frame.timestamp.max()) + 300, END)]
    parts = [frame] if not frame.empty else []
    session = requests.Session()
    calls = 0
    for first, last in ranges:
        cursor = first
        while cursor < last:
            rows = get(session, '/api/v3/klines', {'symbol': symbol, 'interval': '5m',
                       'startTime': cursor * 1000, 'endTime': last * 1000 - 1, 'limit': 1000})
            if not rows:
                raise ValueError(f'{pair} unavailable history at {cursor}')
            batch = pd.DataFrame([[int(row[0]) // 1000, *map(float, row[1:6])] for row in rows], columns=FIELDS)
            next_cursor = int(batch.timestamp.iloc[-1]) + 300
            if next_cursor <= cursor:
                raise ValueError('non-advancing market-data cursor')
            parts.append(batch)
            cursor = next_cursor
            calls += 1
            if calls % 25 == 0:
                print(pair, 'download pages', calls, flush=True)
    result = pd.concat(parts, ignore_index=True).sort_values('timestamp').reset_index(drop=True)
    result.timestamp = result.timestamp.astype('int64')
    validate(result, start, END)
    result.to_csv(target, index=False, float_format='%.17g', lineterminator='\n')
    print(pair, 'validated', len(result), flush=True)
    return {'pair': pair, 'path': str(target), 'sha256': sha(target), 'rows': len(result),
            'start': start, 'end_exclusive': END, 'public_get_pages': calls,
            'seed_path': str(seed), 'seed_sha256': sha(seed) if seed.exists() else None}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT/'results/backtests/v22_365d_20261017/inputs')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    pairs = [('BTC-FDUSD', 'BTCFDUSD'), ('ETH-FDUSD', 'ETHFDUSD'), ('BTC-USDT', 'BTCUSDT'), ('ETH-USDT', 'ETHUSDT')]
    with ThreadPoolExecutor(max_workers=4) as pool:
        evidence = list(pool.map(lambda item: acquire(item, args.output), pairs))
    info = get(requests.Session(), '/api/v3/exchangeInfo', {'symbols': json.dumps([x[1] for x in pairs], separators=(',', ':'))})
    filters = {}
    for item in info['symbols']:
        pair = next(pair for pair, symbol in pairs if symbol == item['symbol'])
        f = {row['filterType']: row for row in item['filters']}
        lot = f['LOT_SIZE']
        market = f.get('MARKET_LOT_SIZE', lot)
        minimum = f.get('NOTIONAL', f.get('MIN_NOTIONAL'))
        filters[pair] = {'tick_size': f['PRICE_FILTER']['tickSize'], 'step_size': lot['stepSize'],
                         'market_step_size': market['stepSize'] if float(market['stepSize']) else lot['stepSize'],
                         'minimum_notional': minimum['minNotional']}
    (args.output/'market_data_manifest.json').write_text(json.dumps({'schema': 'v22-365d-public-data-v1',
        'source': 'Binance Spot /api/v3/klines and /exchangeInfo', 'window_start': START,
        'window_end_exclusive': END, 'days': 365, 'complete_5m_rows_per_bot': 105120,
        'evidence': evidence, 'filters_as_of': '2026-10-17', 'filters': filters}, indent=2), encoding='utf-8')
    print('All market inputs complete; no signed/private API used.', flush=True)


if __name__ == '__main__':
    main()

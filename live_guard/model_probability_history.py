"""Observed production signals only; never reconstruct probability history."""
import hashlib
import json
import math
import sqlite3
from contextlib import closing
from datetime import datetime
from pathlib import Path


def read_market_prices(pair, now):
    """Public, completed hourly candles only; never use exchange credentials."""
    import requests
    try:
        from runtime_endpoints import binance_api_base
    except ImportError:
        from live_guard.runtime_endpoints import binance_api_base
    response = requests.get(f'{binance_api_base()}/api/v3/klines', params={
        'symbol': pair.replace('-', ''), 'interval': '1h',
        'startTime': int((now-169*3600)*1000), 'endTime': int(now*1000), 'limit': 180,
    }, timeout=3)
    response.raise_for_status()
    return [{'timestamp': (int(r[0])/1000)+3600, 'close': float(r[4])}
            for r in response.json() if int(r[6]) < now*1000]


def ts(value):
    try:
        result = float(value) if isinstance(value, (int, float)) else datetime.fromisoformat(str(value).replace('Z', '+00:00')).timestamp()
        return result if math.isfinite(result) else 0
    except (ValueError, TypeError, OverflowError):
        return 0


def collect(grid_state, output, now, *, price_reader=None):
    grid_state, output = Path(grid_state), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    try:
        gate = json.loads((grid_state / 'xgboost_risk_gate.json').read_text())
        pointer = json.loads((grid_state / 'v22-runtime/current.json').read_text())
    except (OSError, ValueError):
        gate, pointer = {}, {}
    generation = str(pointer.get('runtime_generation', ''))
    committed = False
    state_pairs = {}
    if len(generation) == 64 and all(c in '0123456789abcdef' for c in generation):
        try:
            raw = (grid_state / 'v22-runtime/generations' / generation / 'manifest.json').read_bytes()
            manifest = json.loads(raw)
            committed = (hashlib.sha256(raw).hexdigest() == generation
                         and gate.get('runtime_generation') == generation
                         and manifest.get('release_sha256') == gate.get('release_sha256'))
        except (OSError, ValueError):
            pass
        if committed:
            try:
                state_pairs = json.loads((grid_state / 'v22-runtime/generations' / generation /
                                          'gate_state.json').read_text(encoding='utf-8')).get('pairs', {})
            except (OSError, ValueError, AttributeError):
                pass
    result = {'schema': 'management-probability-history-v1', 'generated_at': now,
              'window_start': now - 168*3600, 'window_end': now, 'pairs': {}}
    with closing(sqlite3.connect(output / 'model_probability_history.sqlite', timeout=2)) as db:
        db.row_factory = sqlite3.Row
        db.execute('PRAGMA journal_mode=WAL')
        db.execute('''CREATE TABLE IF NOT EXISTS signals (
          pair TEXT, model TEXT, release TEXT, signal_ts REAL, probability REAL,
          threshold REAL, week TEXT, risk_off INTEGER, break_before INTEGER,
          PRIMARY KEY(pair,model,release,signal_ts))''')
        db.execute('CREATE TABLE IF NOT EXISTS sampling (pair TEXT PRIMARY KEY, seen REAL, broken INTEGER)')
        db.execute('CREATE TABLE IF NOT EXISTS imports (name TEXT PRIMARY KEY)')
        db.execute('CREATE TABLE IF NOT EXISTS prices (pair TEXT, timestamp REAL, close REAL, PRIMARY KEY(pair,timestamp))')
        db.execute('CREATE TABLE IF NOT EXISTS price_poll (pair TEXT PRIMARY KEY, attempted REAL)')
        result['market_prices'] = {}
        for market in ('BTC-FDUSD', 'ETH-FDUSD', 'BTC-USDT', 'ETH-USDT'):
            poll = db.execute('SELECT attempted FROM price_poll WHERE pair=?', (market,)).fetchone()
            error = None
            if price_reader is not None and (poll is None or now-poll[0] >= 900):
                try:
                    for candle in price_reader(market, now):
                        stamp, close = candle['timestamp'], candle['close']
                        if (all(type(v) in (int, float) and math.isfinite(v) for v in (stamp,close))
                                and now-168*3600 <= stamp <= now and close > 0):
                            db.execute('INSERT OR REPLACE INTO prices VALUES (?,?,?)', (market,stamp,close))
                except Exception as exc:
                    error = type(exc).__name__
                db.execute('INSERT OR REPLACE INTO price_poll VALUES (?,?)', (market,now))
            result['market_prices'][market] = {
                'source': 'Binance Spot 1h close', 'interval_seconds': 3600, 'error': error,
                'points': [dict(r) for r in db.execute(
                    'SELECT timestamp,close FROM prices WHERE pair=? AND timestamp>=? AND timestamp<=? ORDER BY timestamp',
                    (market,now-168*3600,now))]}
        db.execute('DELETE FROM prices WHERE timestamp<?', (now-30*86400,))
        audit = grid_state / 'risk_audit.jsonl'
        if not db.execute("SELECT 1 FROM imports WHERE name='risk_audit_v1'").fetchone():
            if audit.exists() and audit.stat().st_size <= 8*1024*1024:
                for line in audit.read_text().splitlines():
                    try:
                        event = json.loads(line)
                        observed = ts(event.get('timestamp'))
                        history = event.get('gate', {})
                        gid = str(history.get('runtime_generation', ''))
                        if not (now-30*86400 <= observed <= now and history.get('source_healthy') is True
                                and history.get('cutover_phase') == 'ACTIVE'
                                and 0 <= observed-ts(history.get('generated_at')) <= 300
                                and observed < ts(history.get('valid_until'))
                                and len(gid) == 64 and all(c in '0123456789abcdef' for c in gid)):
                            continue
                        raw = (grid_state / 'v22-runtime/generations' / gid / 'manifest.json').read_bytes()
                        if hashlib.sha256(raw).hexdigest() != gid or json.loads(raw).get('release_sha256') != history.get('release_sha256'):
                            continue
                        for pair, r in history.get('pairs', {}).items():
                            if pair not in ('BTC-FDUSD','ETH-FDUSD'):
                                continue
                            signal = ts(r.get('signal_ts'))
                            p,t = r.get('probability'),r.get('entry_threshold')
                            if not (0 <= observed-signal <= 3900
                                    and ts(r.get('week_start')) <= signal < ts(r.get('week_end'))
                                    and isinstance(r.get('risk_off_active'),bool)
                                    and all(isinstance(v,(int,float)) and not isinstance(v,bool)
                                            and math.isfinite(v) and 0 <= v <= 1 for v in (p,t))
                                    and len(str(history.get('model_sha256',''))) == 64):
                                continue
                            db.execute('INSERT OR IGNORE INTO signals VALUES (?,?,?,?,?,?,?,?,?)',
                                       (pair,history['model_sha256'],history['release_sha256'],signal,p,t,
                                        str(r.get('model_week','')),int(r['risk_off_active']),1))
                    except (OSError, ValueError, TypeError, KeyError):
                        continue
                db.execute("INSERT INTO imports VALUES ('risk_audit_v1')")
        for pair in ('BTC-FDUSD', 'ETH-FDUSD'):
            row = gate.get('pairs', {}).get(pair, {})
            healthy = False
            try:
                p, t = row.get('probability'), row.get('entry_threshold')
                signal = ts(row.get('signal_ts'))
                healthy = (committed and gate.get('source_healthy') is True
                           and 0 <= now-ts(gate.get('generated_at')) <= 300
                           and now < ts(gate.get('valid_until'))
                           and ts(row.get('week_start')) <= signal < ts(row.get('week_end'))
                           and now < ts(row.get('week_end')) and 0 <= now-signal <= 3900
                           and isinstance(row.get('risk_off_active'), bool)
                           and all(isinstance(v, (int,float)) and not isinstance(v,bool)
                                   and math.isfinite(v) and 0 <= v <= 1 for v in (p,t))
                           and all(len(str(gate.get(k, ''))) == 64 for k in ('model_sha256','release_sha256')))
            except (TypeError, ValueError):
                healthy = False
            previous = db.execute('SELECT * FROM sampling WHERE pair=?', (pair,)).fetchone()
            broken = not previous or previous['broken'] or now-previous['seen'] > 300
            inserted = False
            if healthy:
                cursor = db.execute('INSERT OR IGNORE INTO signals VALUES (?,?,?,?,?,?,?,?,?)', (
                    pair, gate['model_sha256'], gate['release_sha256'], signal, p,t,
                    str(row.get('model_week', '')), int(row['risk_off_active']), int(bool(broken))))
                inserted = cursor.rowcount > 0
            db.execute('INSERT OR REPLACE INTO sampling VALUES (?,?,?)',
                       (pair, now, int(not healthy or (broken and not inserted))))
            points = [dict(r) for r in db.execute('SELECT * FROM signals WHERE pair=? AND signal_ts>=? AND signal_ts<=? ORDER BY signal_ts',
                                                (pair, now-168*3600, now))]
            # Multiple model identities at the same timestamp are ambiguous: omit, never choose silently.
            counts = {}
            for point in points:
                counts[point['signal_ts']] = counts.get(point['signal_ts'], 0) + 1
            points = [p for p in points if counts[p['signal_ts']] == 1]
            for point in points:
                point.pop('model'); point.pop('release'); point.pop('pair')
            recovery = None
            if healthy and isinstance(state_pairs, dict):
                saved = state_pairs.get(pair, {})
                saved = saved if isinstance(saved, dict) else {}
                state, last = saved.get('gate_state', {}), saved.get('last_snapshot', {})
                state = state if isinstance(state, dict) else {}
                last = last if isinstance(last, dict) else {}
                count, required = state.get('recovery_count'), last.get('recovery_required_4h_bars')
                if (state.get('last_signal_ts') == signal
                        and state.get('active') is row['risk_off_active']
                        and last.get('probability') == p and last.get('entry_threshold') == t
                        and type(count) is int and count >= 0
                        and type(required) is int and required > 0):
                    recovery = {'count': count, 'ordinary_required': required,
                                'risk_off': row['risk_off_active'], 'signal_ts': signal}
            result['pairs'][pair] = {'current_available': bool(healthy), 'points': points,
                                     'current_recovery': recovery}
        db.execute('DELETE FROM signals WHERE signal_ts<?', (now-30*86400,))
        db.commit()
    return result

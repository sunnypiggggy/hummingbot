from __future__ import annotations

import json
import sqlite3
from bisect import bisect_right
from pathlib import Path

import requests

from .collector import FUTURE_TOLERANCE_SECONDS, MAX_AGE_SECONDS, ROBOTS, data_state, number


WINDOW_SECONDS = 12 * 3600
STEP_SECONDS = 600
MAX_POINTS = 73
PRICE_URL = "https://data-api.binance.vision/api/v3/klines"
MAX_PRICE_BYTES = 128 * 1024


class PublicKlinePrices:
    """Fixed-symbol, keyless Binance market data; never a trading client."""

    def __init__(self, *, timeout: float = 5, session=None):
        self.timeout = timeout
        self.session = session if session is not None else requests.Session()
        self.session.trust_env = False
        # This client must not inherit account authentication from a caller.
        self.session.headers.clear()
        self.session.auth = None
        self.session.cookies.clear()

    def samples(self, pair: str, start: int, end: int) -> list[tuple[float, float]]:
        if pair not in {p for _, p, _, _ in ROBOTS}:
            return []
        response = None
        try:
            response = self.session.get(
                PRICE_URL,
                params={"symbol": pair.replace("-", ""), "interval": "5m",
                        # REST filters candle open time; our curve uses close
                        # time. Include the preceding bar at an unaligned edge.
                        "startTime": (start - 600) * 1000, "endTime": end * 1000,
                        "limit": 150, "timeZone": "0"},
                timeout=self.timeout, allow_redirects=False, verify=True, stream=True,
            )
            if response.status_code != 200:
                return []
            raw = bytearray()
            for block in response.iter_content(8192):
                raw.extend(block)
                if len(raw) > MAX_PRICE_BYTES:
                    return []
            rows = json.loads(raw)
            if not isinstance(rows, list) or len(rows) > 150:
                return []
            result = {}
            duplicates = set()
            for row in rows:
                if not isinstance(row, list) or len(row) != 12:
                    continue
                opened, closed, price = number(row[0]), number(row[6]), number(row[4])
                if (opened is None or closed is None or price is None or price <= 0
                        or opened != int(opened) or closed != int(closed)
                        or int(opened) % 300000 or closed - opened != 299999):
                    continue
                observed = closed / 1000
                # Open candles and a future close are never final market prices.
                if not start - MAX_AGE_SECONDS <= observed <= end:
                    continue
                if observed in result:
                    duplicates.add(observed)
                result[observed] = price
            return sorted((ts, value) for ts, value in result.items() if ts not in duplicates)
        except (requests.RequestException, ValueError, TypeError, OverflowError):
            return []
        finally:
            if response is not None:
                response.close()


def sampled_curve(samples: list[tuple[float, float]], ticks: list[int]) -> list[dict]:
    """Use a prior observation within 300s; do not interpolate or carry gaps."""
    samples = sorted(samples)
    times = [row[0] for row in samples]
    result = []
    for tick in ticks:
        index = bisect_right(times, tick) - 1
        valid = index >= 0 and tick - times[index] <= MAX_AGE_SECONDS
        result.append({"ts": tick, "value": samples[index][1] if valid else None})
    return result


class HistoryCollector:
    """Read canonical MTM in one RO transaction; current risk is no history."""

    def __init__(self, profit_db_path: Path, *, prices: PublicKlinePrices | None = None):
        self.profit_db_path = Path(profit_db_path)
        self.prices = prices

    def _profits(self, ticks: list[int], now: float) -> dict:
        result = {}
        connection = None
        try:
            uri = self.profit_db_path.resolve().as_uri() + "?mode=ro"
            connection = sqlite3.connect(uri, uri=True, timeout=3)
            connection.execute("BEGIN")
            for strategy, pair, _, _ in ROBOTS:
                latest = connection.execute(
                    "SELECT observed_at,mtm_quote FROM profit_snapshot "
                    "WHERE strategy=? AND pair=? ORDER BY observed_at DESC LIMIT 1",
                    (strategy, pair),
                ).fetchone()
                observed = number(latest[0]) if latest else None
                cumulative = number(latest[1]) if latest else None
                if (observed is None or observed <= 0 or observed > now + FUTURE_TOLERANCE_SECONDS
                        or cumulative is None):
                    observed = None
                values = []
                for tick in ticks:
                    row = connection.execute(
                        "SELECT observed_at,mtm_quote FROM profit_snapshot "
                        "WHERE strategy=? AND pair=? AND observed_at<=? "
                        "ORDER BY observed_at DESC LIMIT 1", (strategy, pair, tick),
                    ).fetchone()
                    source_time = number(row[0]) if row else None
                    amount = number(row[1]) if row else None
                    valid = source_time is not None and source_time > 0 and 0 <= tick - source_time <= MAX_AGE_SECONDS
                    values.append(amount if valid else None)
                baseline = values[0]
                curve = [{"ts": tick, "value": None if observed is None or baseline is None or value is None
                          else number(value - baseline)} for tick, value in zip(ticks, values)]
                result[f"{strategy}:{pair}"] = (observed, curve)
            return result
        except (sqlite3.Error, OSError, ValueError, TypeError, OverflowError):
            return {}
        finally:
            if connection is not None:
                connection.close()

    def collect(self, *, now: float) -> dict[str, dict]:
        collected = number(now)
        if collected is None or collected <= WINDOW_SECONDS:
            raise ValueError("history collection time is invalid")
        end = int(collected)
        start = end - WINDOW_SECONDS
        ticks = list(range(start, end + 1, STEP_SECONDS))
        profits = self._profits(ticks, collected)
        try:
            risk = json.loads((self.profit_db_path.parent / "risk_history.json").read_text(encoding="utf-8"))
            if risk.get("schema") != "report-risk-history-v1":
                risk = {}
        except (OSError, ValueError):
            risk = {}
        result = {}
        for strategy, pair, _, _ in ROBOTS:
            identity = f"{strategy}:{pair}"
            coverage = risk.get("permission_coverage", {}).get(identity, [])
            coverage = coverage[-1] if coverage else {}
            coverage_start, coverage_end = number(coverage.get("start")), number(coverage.get("end"))
            if (coverage_end is None or not 0 <= collected-coverage_end <= 180
                    or coverage_start is None or coverage_start >= coverage_end):
                coverage_start = coverage_end = None
            observed, profit_curve = profits.get(identity, (None, [{"ts": t, "value": None} for t in ticks]))
            price_samples = self.prices.samples(pair, start, end) if self.prices else []
            price_observed = max((row[0] for row in price_samples), default=None)
            price_curve = sampled_curve(price_samples, ticks)
            states = (data_state(observed, collected), data_state(price_observed, collected))
            state = "FRESH" if "FRESH" in states else "STALE" if "STALE" in states else "UNAVAILABLE"
            result[identity] = {
                "start_at": start, "end_at": end, "profit_observed_at": observed,
                "price_observed_at": price_observed,
                "window_complete": all(point["value"] is not None for point in profit_curve),
                "status_coverage_start_at": coverage_start,
                "status_coverage_end_at": coverage_end,
                "profit_points": profit_curve, "price_points": price_curve,
                # Report is the sole history writer; never infer past risk from
                # the current gate or probability. Clip to this request window.
                "pauses": [{**p,"start_at":max(start,coverage_start,p["start_at"]),
                            "end_at":min(end,coverage_end,p["end_at"])}
                           for p in risk.get("permission_intervals", {}).get(identity, [])
                           if coverage_start is not None and max(start,coverage_start) < p.get("end_at", 0)
                           and p.get("start_at",end) < min(end,coverage_end)][-8:],
                "data_state": state,
            }
        return result

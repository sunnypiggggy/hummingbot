"""Reproducible public Binance Spot candles for the isolated strategy study.

No credentials, production state, or trading endpoints are used.  Raw archives
are retained with their official checksums; missing minutes are fetched from
public klines endpoints and never interpolated.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
from typing import Iterable
import uuid
import zipfile

import numpy as np
import pandas as pd
import requests


DEFAULT_CACHE_DIR = Path("data/spot_strategy_suite_365d")
ARCHIVE_BASE = "https://data.binance.vision/data/spot"
API_BASES = ("https://data-api.binance.vision", "https://api.binance.com")
OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]
SCHEMA_VERSION = 1


class MarketDataError(ValueError):
    """Invalid, incomplete, or non-reproducible study data."""


class ArchiveUnavailable(RuntimeError):
    """An archive is unavailable and may be replaced by a public data source."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _stamp(value) -> pd.Timestamp:
    result = pd.Timestamp(value)
    result = result.tz_localize("UTC") if result.tzinfo is None else result.tz_convert("UTC")
    if result != result.floor("min"):
        raise MarketDataError(f"Expected a minute-aligned UTC boundary: {value!r}")
    return result


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def validate_minutes(frame: pd.DataFrame, start, end) -> dict:
    """Raise for missing/duplicate minutes, non-UTC time, or invalid OHLCV."""
    start, end = _stamp(start), _stamp(end)
    if end <= start:
        raise MarketDataError("end must be strictly later than start")
    if not isinstance(frame.index, pd.DatetimeIndex) or frame.index.tz is None:
        raise MarketDataError("Candles require a timezone-aware UTC DatetimeIndex")
    if str(frame.index.tz) != "UTC":
        raise MarketDataError("Candle index must use UTC")
    if frame.index.has_duplicates:
        examples = frame.index[frame.index.duplicated()].unique()[:3].astype(str).tolist()
        raise MarketDataError(f"Duplicate candle minutes: {examples}")
    if not frame.index.is_monotonic_increasing:
        raise MarketDataError("Candle minutes are not sorted")
    if not all(column in frame for column in OHLCV_COLUMNS):
        raise MarketDataError(f"Missing OHLCV columns: {set(OHLCV_COLUMNS) - set(frame.columns)}")
    expected = pd.date_range(start, end, freq="min", inclusive="left")
    missing, extra = expected.difference(frame.index), frame.index.difference(expected)
    if len(missing) or len(extra):
        raise MarketDataError(
            f"Minute coverage failed: missing={len(missing)}, extra={len(extra)}, "
            f"first_missing={missing[:3].astype(str).tolist()}, "
            f"first_extra={extra[:3].astype(str).tolist()}"
        )
    try:
        values = frame[OHLCV_COLUMNS].to_numpy(dtype=float)
    except (ValueError, TypeError) as error:
        raise MarketDataError("OHLCV values must be numeric") from error
    finite = np.isfinite(values).all(axis=1)
    o, h, low, c, volume = values.T
    valid = finite & (o > 0) & (h > 0) & (low > 0) & (c > 0) & (volume >= 0)
    valid &= (h >= np.maximum(o, c)) & (low <= np.minimum(o, c)) & (h >= low)
    if not valid.all():
        examples = frame.index[~valid][:3].astype(str).tolist()
        raise MarketDataError(f"Invalid OHLCV candles: {examples}")
    return {
        "validated": True,
        "rows": len(frame),
        "expected_rows": len(expected),
        "missing_minutes": 0,
        "duplicate_minutes": 0,
        "coverage_start": start.isoformat(),
        "coverage_end_exclusive": end.isoformat(),
        "index_timezone": "UTC",
        "index_semantics": "minute_open",
    }


def _response(url: str, *, params: dict | None = None, stream: bool = False):
    last_error = None
    for attempt in range(3):
        try:
            response = requests.get(url, params=params, timeout=(15, 90), stream=stream)
            if response.status_code == 404:
                response.close()
                raise ArchiveUnavailable(f"HTTP 404: {url}")
            if response.status_code in (418, 429, 500, 502, 503, 504):
                response.close()
                raise requests.HTTPError(f"HTTP {response.status_code}: {url}")
            response.raise_for_status()
            return response
        except ArchiveUnavailable:
            raise
        except requests.RequestException as error:
            if "response" in locals():
                response.close()
            last_error = error
            if attempt < 2:
                time.sleep(1 + attempt * 2)
    raise ArchiveUnavailable(f"Public data request failed: {url}: {last_error}")


def _archive_path(symbol: str, frequency: str, label: str) -> str:
    return f"{frequency}/klines/{symbol}/1m/{symbol}-1m-{label}.zip"


def _read_archive(path: Path) -> tuple[pd.DataFrame, str]:
    try:
        with zipfile.ZipFile(path) as archive:
            names = [name for name in archive.namelist() if name.endswith(".csv")]
            if len(names) != 1:
                raise MarketDataError(f"Expected one CSV in {path.name}")
            with archive.open(names[0]) as handle:
                raw = pd.read_csv(handle, header=None, usecols=range(6), dtype=str)
    except (zipfile.BadZipFile, pd.errors.ParserError) as error:
        raise MarketDataError(f"Unreadable archive: {path}") from error
    timestamps = pd.to_numeric(raw.iloc[:, 0], errors="coerce")
    if timestamps.isna().any():
        # Some public data formats use a header; arbitrary malformed rows fail.
        bad = np.flatnonzero(timestamps.isna().to_numpy()).tolist()
        if bad != [0] or str(raw.iloc[0, 0]).lower() not in ("open_time", "open time", "timestamp"):
            raise MarketDataError(f"Invalid timestamp rows in {path.name}: {bad[:3]}")
        raw, timestamps = raw.iloc[1:], timestamps.iloc[1:]
    timestamps = timestamps.astype("int64")
    unit = "us" if (timestamps >= 100_000_000_000_000).all() else "ms"
    if unit == "ms" and not timestamps.between(100_000_000_000, 99_999_999_999_999).all():
        raise MarketDataError(f"Mixed or unsupported timestamp units in {path.name}")
    index = pd.DatetimeIndex(pd.to_datetime(timestamps, unit=unit, utc=True)).as_unit("ns")
    values = raw.iloc[:, 1:6].apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)
    frame = pd.DataFrame(values, index=index, columns=OHLCV_COLUMNS)
    frame.index.name = "timestamp"
    return frame, unit


def _validate_archive_rows(frame: pd.DataFrame) -> int:
    """Validate observed rows, allowing gaps to be filled explicitly later."""
    if frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
        raise MarketDataError("Archive contains duplicate or unsorted candle minutes")
    if not (frame.index == frame.index.floor("min")).all():
        raise MarketDataError("Archive contains non-minute-aligned timestamps")
    split_at = np.flatnonzero(np.diff(frame.index.asi8) != 60_000_000_000) + 1
    for positions in np.split(np.arange(len(frame)), split_at):
        segment = frame.iloc[positions]
        validate_minutes(segment, segment.index[0], segment.index[-1] + pd.Timedelta(minutes=1))
    expected = int((frame.index[-1] - frame.index[0]).total_seconds() / 60) + 1
    return expected - len(frame)


def _get_archive(relative: str, cache_dir: Path, offline: bool) -> tuple[pd.DataFrame, dict]:
    path = cache_dir / "raw" / relative
    checksum_path = path.with_name(path.name + ".CHECKSUM")
    metadata_path = path.with_name(path.name + ".metadata.json")
    url = f"{ARCHIVE_BASE}/{relative}"
    if not path.exists() or not checksum_path.exists():
        if offline:
            raise ArchiveUnavailable(f"Archive not cached: {relative}")
        path.parent.mkdir(parents=True, exist_ok=True)
        if not checksum_path.exists():
            response = _response(url + ".CHECKSUM")
            checksum_text = response.text
            response.close()
            # Never replace an existing cached checksum with today's version.
            checksum_path.write_text(checksum_text, encoding="utf-8")
        if not path.exists():
            for attempt in range(3):
                response = _response(url, stream=True)
                temp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
                try:
                    with temp.open("wb") as handle:
                        for chunk in response.iter_content(1024 * 1024):
                            handle.write(chunk)
                    temp.replace(path)
                    break
                except requests.RequestException as error:
                    if attempt == 2:
                        raise ArchiveUnavailable(f"Archive stream failed: {url}: {error}") from error
                    time.sleep(1 + 2 * attempt)
                finally:
                    response.close()
                    temp.unlink(missing_ok=True)
    expected_sha = checksum_path.read_text(encoding="utf-8").strip().split()[0].lower()
    if len(expected_sha) != 64 or any(ch not in "0123456789abcdef" for ch in expected_sha):
        raise MarketDataError(f"Invalid official checksum: {checksum_path}")
    actual_sha = _sha(path)
    if expected_sha != actual_sha:
        raise MarketDataError(f"Archive checksum mismatch (preserved for diagnosis): {path}")
    frame, unit = _read_archive(path)
    if frame.empty:
        raise MarketDataError(f"Empty archive: {relative}")
    archive_missing = _validate_archive_rows(frame)
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    else:
        metadata = {"retrieved_at": _now()}
    metadata.update({
        "kind": "official_archive", "url": url, "checksum_url": url + ".CHECKSUM",
        "path": str(path.resolve()), "checksum_path": str(checksum_path.resolve()),
        "checksum_file_sha256": _sha(checksum_path),
        "sha256": actual_sha, "units": unit, "rows": len(frame),
        "coverage_start": frame.index[0].isoformat(),
        "coverage_end_exclusive": (frame.index[-1] + pd.Timedelta(minutes=1)).isoformat(),
        "validated": True, "missing_minutes_within_source": archive_missing,
    })
    if not offline:
        _atomic_json(metadata_path, metadata)
    return frame, metadata


def _archive_specs(symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> list[tuple[str, pd.Timestamp, pd.Timestamp]]:
    specs = []
    month = start.normalize().replace(day=1)
    while month < end:
        next_month = month + pd.offsets.MonthBegin(1)
        left, right = max(start, month), min(end, next_month)
        if next_month <= end:
            specs.append((_archive_path(symbol, "monthly", month.strftime("%Y-%m")), left, right))
        else:
            day = left.normalize()
            while day < right:
                next_day = day + pd.Timedelta(days=1)
                specs.append((_archive_path(symbol, "daily", day.strftime("%Y-%m-%d")), max(left, day), min(right, next_day)))
                day = next_day
        month = next_month
    return specs


def _public_klines(symbol: str, start: pd.Timestamp, end: pd.Timestamp, cache_dir: Path) -> tuple[pd.DataFrame, list[dict]]:
    chunks, sources = [], []
    cursor = start
    while cursor < end:
        params = {
            "symbol": symbol, "interval": "1m", "startTime": int(cursor.timestamp() * 1000),
            "endTime": int(end.timestamp() * 1000) - 1, "limit": 1000,
        }
        response = None
        for base in API_BASES:
            try:
                response = _response(base + "/api/v3/klines", params=params)
                data = response.json()
                if not isinstance(data, list) or not data:
                    raise ArchiveUnavailable(f"Empty public klines response: {symbol} {cursor}")
                break
            except (ArchiveUnavailable, ValueError) as error:
                if response is not None:
                    response.close()
                response = None
                last_error = error
        if response is None:
            raise MarketDataError(f"Could not fill exact minute gap {symbol}: {cursor} .. {end}: {last_error}")
        raw_content = response.content
        source_url = response.url
        response.close()
        content_sha = hashlib.sha256(raw_content).hexdigest()
        raw_path = cache_dir / "raw_rest" / symbol / (str(params["startTime"]) + "-" + content_sha + ".json")
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        if not raw_path.exists():
            raw_path.write_bytes(raw_content)
        index = pd.DatetimeIndex(pd.to_datetime([row[0] for row in data], unit="ms", utc=True)).as_unit("ns")
        frame = pd.DataFrame([[float(value) for value in row[1:6]] for row in data], index=index, columns=OHLCV_COLUMNS)
        frame.index.name = "timestamp"
        frame = frame[(frame.index >= cursor) & (frame.index < end)]
        if frame.empty or frame.index[0] != cursor:
            raise MarketDataError(f"REST could not supply the first missing minute: {symbol} {cursor}")
        chunk_end = frame.index[-1] + pd.Timedelta(minutes=1)
        validate_minutes(frame, cursor, chunk_end)
        sources.append({
            "kind": "public_rest_gap_fill", "url": source_url, "sha256": content_sha,
            "path": str(raw_path.resolve()), "units": "ms", "rows": len(frame),
            "coverage_start": cursor.isoformat(), "coverage_end_exclusive": chunk_end.isoformat(),
            "retrieved_at": _now(), "validated": True,
        })
        chunks.append(frame)
        cursor = chunk_end
        time.sleep(0.05)
    return pd.concat(chunks), sources


def _gaps(index: pd.DatetimeIndex, start: pd.Timestamp, end: pd.Timestamp):
    missing = pd.date_range(start, end, freq="min", inclusive="left").difference(index)
    if not len(missing):
        return []
    breaks = np.flatnonzero(np.diff(missing.asi8) != 60_000_000_000) + 1
    return [(group[0], group[-1] + pd.Timedelta(minutes=1)) for group in np.split(missing, breaks)]


def load_market_data(symbols: Iterable[str], start, end, warmup_days: int = 210,
                     cache_dir: Path = DEFAULT_CACHE_DIR, offline: bool = False) -> tuple[dict[str, pd.DataFrame], dict]:
    """Return strict one-minute OHLCV frames over [start-warmup, end)."""
    start, end, cache_dir = _stamp(start), _stamp(end), Path(cache_dir)
    if end <= start or not isinstance(warmup_days, int) or warmup_days < 0:
        raise MarketDataError("Require end > start and a nonnegative integer warmup_days")
    symbols = list(dict.fromkeys(str(symbol).upper() for symbol in symbols))
    if not symbols or any(not symbol.isalnum() for symbol in symbols):
        raise MarketDataError("Expected nonempty alphanumeric Binance symbols")
    data_start = start - pd.Timedelta(days=warmup_days)
    frames = {}
    manifest = {
        "schema_version": SCHEMA_VERSION, "market": "binance_spot", "interval": "1m",
        "study_start": start.isoformat(), "study_end_exclusive": end.isoformat(),
        "warmup_days": warmup_days, "data_start": data_start.isoformat(),
        "generated_at": _now(), "symbols": {}, "validated": False,
    }
    for symbol in symbols:
        key = f"{symbol}-{data_start.strftime('%Y%m%dT%H%M')}-{end.strftime('%Y%m%dT%H%M')}"
        normalized_path = cache_dir / "normalized" / (key + ".pkl.gz")
        symbol_manifest_path = cache_dir / "manifests" / (key + ".json")
        if normalized_path.exists() and symbol_manifest_path.exists():
            metadata = json.loads(symbol_manifest_path.read_text(encoding="utf-8"))
            if metadata.get("schema_version") != SCHEMA_VERSION:
                raise MarketDataError(f"Unsupported normalized cache version: {symbol}")
            if _sha(normalized_path) != metadata.get("normalized_cache_sha256"):
                raise MarketDataError(f"Normalized cache checksum mismatch: {symbol}")
            # These pickles are only read from this module's hash-verified local cache.
            frame = pd.read_pickle(normalized_path, compression="gzip")
            validation = validate_minutes(frame, data_start, end)
            for source in metadata["sources"]:
                source_path = Path(source["path"])
                if not source_path.exists() or _sha(source_path) != source["sha256"]:
                    raise MarketDataError(f"Source cache checksum mismatch or missing: {source_path}")
                if source.get("kind") == "official_archive":
                    checksum_path = Path(source["checksum_path"])
                    if not checksum_path.exists():
                        raise MarketDataError(f"Official checksum missing: {checksum_path}")
                    recorded_checksum_sha = source.get("checksum_file_sha256")
                    if recorded_checksum_sha and _sha(checksum_path) != recorded_checksum_sha:
                        raise MarketDataError(f"Official checksum file changed: {checksum_path}")
                    official_sha = checksum_path.read_text(encoding="utf-8").strip().split()[0].lower()
                    if official_sha != source["sha256"]:
                        raise MarketDataError(f"Official checksum disagrees with pinned source: {source_path}")
            metadata.update(validation)
            frames[symbol], manifest["symbols"][symbol] = frame, metadata
            print(f"{symbol}: verified normalized cache, {len(frame):,} minutes", flush=True)
            continue
        if offline:
            raise MarketDataError(f"No complete normalized cache for offline run: {symbol}")
        chunks, sources, unavailable = [], [], []
        specs = _archive_specs(symbol, data_start, end)
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = {pool.submit(_get_archive, relative, cache_dir, False): (relative, left, right)
                       for relative, left, right in specs}
            for number, future in enumerate(as_completed(futures), 1):
                relative, left, right = futures[future]
                try:
                    archive_frame, source = future.result()
                    chunks.append(archive_frame[(archive_frame.index >= data_start) & (archive_frame.index < end)])
                    sources.append(source)
                except ArchiveUnavailable as error:
                    unavailable.append({"url": f"{ARCHIVE_BASE}/{relative}", "reason": str(error)})
                    # A missing monthly archive is preferably replaced with daily archives.
                    if relative.startswith("monthly/"):
                        day = left.normalize()
                        while day < right:
                            daily = _archive_path(symbol, "daily", day.strftime("%Y-%m-%d"))
                            try:
                                daily_frame, daily_source = _get_archive(daily, cache_dir, False)
                                chunks.append(daily_frame[(daily_frame.index >= data_start) & (daily_frame.index < end)])
                                sources.append(daily_source)
                            except ArchiveUnavailable as daily_error:
                                unavailable.append({"url": f"{ARCHIVE_BASE}/{daily}", "reason": str(daily_error)})
                            day += pd.Timedelta(days=1)
                if number == len(specs) or number % 10 == 0:
                    print(f"{symbol}: checked {number}/{len(specs)} archives", flush=True)
        if chunks:
            frame = pd.concat(chunks).sort_index()
            if frame.index.has_duplicates:
                raise MarketDataError(f"Overlapping archive minute data: {symbol}")
        else:
            frame = pd.DataFrame(columns=OHLCV_COLUMNS, index=pd.DatetimeIndex([], tz="UTC"))
        gaps = _gaps(frame.index, data_start, end)
        if gaps:
            print(f"{symbol}: filling {len(gaps)} exact gaps from public REST", flush=True)
        for left, right in gaps:
            extra_frame, extra_sources = _public_klines(symbol, left, right, cache_dir)
            chunks.append(extra_frame)
            sources.extend(extra_sources)
        if gaps:
            frame = pd.concat(chunks).sort_index()
        frame = frame[OHLCV_COLUMNS].astype(float)
        validation = validate_minutes(frame, data_start, end)
        normalized_path.parent.mkdir(parents=True, exist_ok=True)
        temp = normalized_path.with_name(normalized_path.name + "." + uuid.uuid4().hex + ".tmp")
        try:
            frame.to_pickle(temp, compression="gzip")
            temp.replace(normalized_path)
        finally:
            temp.unlink(missing_ok=True)
        metadata = {
            "schema_version": SCHEMA_VERSION, "symbol": symbol, "interval": "1m",
            **validation, "normalized_cache": str(normalized_path.resolve()),
            "normalized_cache_sha256": _sha(normalized_path),
            "sources": sorted(sources, key=lambda item: item["coverage_start"]),
            "unavailable_archives": unavailable, "retrieved_at": _now(),
            "archive_timestamp_policy": "Spot archives since 2025-01-01 use microseconds; parsed units verified from values",
        }
        _atomic_json(symbol_manifest_path, metadata)
        frames[symbol], manifest["symbols"][symbol] = frame, metadata
        print(f"{symbol}: validated {len(frame):,} minutes, {data_start} .. {end}", flush=True)
    manifest["validated"] = True
    if not offline:
        _atomic_json(cache_dir / "manifest.json", manifest)
    return frames, manifest


def fetch_exchange_filters(symbols: Iterable[str]) -> tuple[dict, dict]:
    """Fetch current public exchangeInfo; this is explicitly not historical data."""
    symbols = list(dict.fromkeys(str(symbol).upper() for symbol in symbols))
    response = None
    for base in API_BASES:
        try:
            response = _response(base + "/api/v3/exchangeInfo", params={
                "symbols": json.dumps(symbols, separators=(",", ":")),
            })
            payload = response.json()
            break
        except (ArchiveUnavailable, ValueError) as error:
            if response is not None:
                response.close()
                response = None
            last_error = error
    if response is None:
        raise MarketDataError(f"Could not fetch current public filters: {last_error}")
    source_url, raw_sha = response.url, hashlib.sha256(response.content).hexdigest()
    response.close()
    result = {}
    for item in payload.get("symbols", []):
        by_type = {value["filterType"]: value for value in item["filters"]}
        notional = by_type.get("NOTIONAL", by_type.get("MIN_NOTIONAL", {}))
        result[item["symbol"]] = {
            "tick_size": float(by_type["PRICE_FILTER"]["tickSize"]),
            "step_size": float(by_type["LOT_SIZE"]["stepSize"]),
            "minimum_notional": float(notional.get("minNotional", 0)),
        }
    if set(result) != set(symbols):
        raise MarketDataError(f"Missing requested exchange filters: {set(symbols) - set(result)}")
    return result, {
        "url": source_url, "sha256": raw_sha, "retrieved_at": _now(),
        "historical": False, "scope": "current exchangeInfo snapshot; not historical filters",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", nargs="+", default=["BTCUSDT", "ETHUSDT"])
    parser.add_argument("--start", default="2025-10-23T00:00:00Z")
    parser.add_argument("--end", default="2026-10-23T00:00:00Z")
    parser.add_argument("--warmup-days", type=int, default=210)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--exchange-filters", action="store_true")
    args = parser.parse_args()
    _, manifest = load_market_data(args.symbols, args.start, args.end, args.warmup_days, args.cache_dir, args.offline)
    if args.exchange_filters and not args.offline:
        filters, metadata = fetch_exchange_filters(args.symbols)
        _atomic_json(args.cache_dir / "exchange_filters_current.json", {"filters": filters, "metadata": metadata})
    print(json.dumps({"validated": manifest["validated"], "rows": {
        symbol: metadata["rows"] for symbol, metadata in manifest["symbols"].items()
    }}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

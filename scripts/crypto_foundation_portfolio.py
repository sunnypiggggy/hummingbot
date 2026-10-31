"""Causal daily inputs and fixed forecast-to-portfolio rule for local research.

Prices and trading still come from the immutable, audited historical datasets.
True quote turnover is recovered from the same public Binance archive bytes.
"""
from __future__ import annotations

import gzip
import hashlib
import io
from pathlib import Path
import pickle
import re
import zipfile

import numpy as np
import pandas as pd

from scripts.backtest_crypto_rl_top10 import load_dataset, sha, VAL_START, TEST_END
from scripts.crypto_rl_reporting import canonical_csv, write_json

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results/backtests/crypto_foundation_models_20261024"
DATASETS = {"original": ROOT / "data/crypto_rl_top10/dataset.pkl.gz",
            "ex_ada": ROOT / "data/crypto_rl_top10_ex_ada/dataset.pkl.gz"}
FIELDS = ("open", "high", "low", "close", "volume", "amount")
CONTEXT_DAYS, HURDLE, VOL_FLOOR = 128, .0024, .01


def utc(value):
    stamp = pd.Timestamp(value)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def frame_digest(frame):
    return hashlib.sha256(canonical_csv(frame, index=True)).hexdigest()


def fixed_weights(forecasts, volatilities, valid, *, hurdle=HURDLE, vol_floor=VOL_FLOOR,
                  cash_floor=.05, coin_cap=.25):
    """Positive forecast excess / trailing realized risk, capped without refill.

    hurdle is a fixed approximate round-trip cost buffer, not an exact net ROI.
    Clipped risk weight is left in USDT; surviving positions are not renormalized.
    """
    predicted = np.asarray(forecasts, dtype=float)
    vol = np.asarray(volatilities, dtype=float)
    valid = np.asarray(valid, dtype=bool)
    if predicted.shape != (10,) or vol.shape != (10,) or valid.shape != (10,):
        raise ValueError("Exactly ten historical slots are required")
    good = valid & np.isfinite(predicted) & np.isfinite(vol) & (vol >= 0)
    scores = np.zeros(10)
    scores[good] = np.maximum(predicted[good] - hurdle, 0) / np.maximum(vol[good], vol_floor)
    weights = np.zeros(11)
    if scores.sum() > 0:
        weights[:10] = np.minimum((1 - cash_floor) * scores / scores.sum(), coin_cap)
    weights[-1] = 1 - weights[:10].sum()
    if weights[-1] < cash_floor - 1e-12 or (weights < 0).any():
        raise AssertionError("Forecast mapper violated funded-spot constraints")
    return weights


def known_exit_today(asset, date):
    notice, until = asset.get("delisting_notice_at"), asset.get("effective_until")
    return bool(notice and until and utc(notice) <= date and date < utc(until) <= date + pd.Timedelta(days=1))


def build_market(output=OUTPUT):
    """Derive real OHLCVA without changing any frozen source matrix or archive."""
    output = Path(output)
    market_path = output / "market/daily_ohlcva.pkl.gz"
    manifest_path = output / "market/manifest.json"
    if market_path.exists() or manifest_path.exists():
        raise FileExistsError("Preserve existing prepared foundation-model inputs")
    dataset = load_dataset(DATASETS["original"])
    start, end = utc(VAL_START) - pd.Timedelta(days=CONTEXT_DAYS), utc(TEST_END)
    index = dataset["hourly_close"].index
    index = index[(index >= start) & (index < end)]
    quote = pd.DataFrame(np.nan, index=index, columns=dataset["hourly_close"].columns)
    ticker_to_key = {row["binance_symbol"]: str(key) for key, row in dataset["assets"].items()}
    evidence, maximum_error = [], 0.
    for entry in sorted(dataset["manifest"]["archive_evidence"], key=lambda row: row.get("path", "")):
        if not entry.get("verified") or not entry.get("path"):
            continue
        path = Path(entry["path"])
        match = re.fullmatch(r"(.+)-1h-(\d{4}-\d{2}(?:-\d{2})?)\.zip", path.name)
        if not match or match[1] not in ticker_to_key:
            continue
        archive_start = utc(match[2])
        archive_end = archive_start + (pd.offsets.MonthBegin(1) if len(match[2]) == 7 else pd.Timedelta(days=1))
        if archive_end <= start or archive_start >= end:
            continue
        raw = path.read_bytes()
        expected = Path(entry["checksum_path"]).read_text(encoding="utf-8").split()[0].lower()
        if hashlib.sha256(raw).hexdigest() != expected or expected != entry["sha256"]:
            raise AssertionError("Original archive/checksum differs: " + str(path))
        with zipfile.ZipFile(io.BytesIO(raw)) as zipped:
            members = [name for name in zipped.namelist() if name.endswith(".csv")]
            if len(members) != 1:
                raise ValueError("Archive must contain one CSV")
            frame = pd.read_csv(zipped.open(members[0]), header=None)
        frame = frame.loc[frame[0].astype(str).str.fullmatch(r"\d+")].copy()
        stamps = pd.to_numeric(frame[0], errors="raise").astype("int64")
        unit = "us" if len(stamps) and int(stamps.iloc[0]) > 10**14 else "ms"
        frame.index = pd.DatetimeIndex(pd.to_datetime(stamps, unit=unit, utc=True))
        frame = frame.loc[frame.index.intersection(index)].astype(float)
        if frame.index.has_duplicates:
            raise AssertionError("Repeated real archive hours")
        key = ticker_to_key[match[1]]
        for col, field in enumerate(FIELDS[:5], 1):
            original = dataset["hourly_" + field].loc[frame.index, key].to_numpy()
            values = frame[col].to_numpy()
            error = float(np.max(np.abs(original - values))) if len(frame) else 0.
            maximum_error = max(maximum_error, error)
            if not np.allclose(original, values, rtol=1e-12, atol=0, equal_nan=False):
                raise AssertionError("Quote-turnover archive OHLCV differs from frozen data")
        amount = frame[7].to_numpy()
        if not np.isfinite(amount).all() or (amount < 0).any():
            raise ValueError("Invalid real quote-asset turnover")
        prior = quote.loc[frame.index, key]
        if prior.notna().any() and not np.array_equal(prior.dropna(), frame.loc[prior.notna(), 7]):
            raise AssertionError("Conflicting archive quote turnover")
        quote.loc[frame.index, key] = amount
        evidence.append({**entry, "quote_volume_column": 7, "used_rows": len(frame)})
    complete = np.isfinite(quote) & quote.ge(0)
    hourly = {}
    for field in FIELDS[:5]:
        hourly[field] = dataset["hourly_" + field].loc[index]
        complete &= np.isfinite(hourly[field]) & (hourly[field].ge(0) if field == "volume" else hourly[field].gt(0))
    counts = complete.resample("D").sum()
    daily = {"open": hourly["open"].resample("D").first(),
             "high": hourly["high"].resample("D").max(),
             "low": hourly["low"].resample("D").min(),
             "close": hourly["close"].resample("D").last(),
             "volume": hourly["volume"].resample("D").sum(min_count=24),
             "amount": quote.resample("D").sum(min_count=24)}
    daily = {field: frame.where(counts.eq(24)) for field, frame in daily.items()}
    if any(not frame.index.equals(next(iter(daily.values())).index) for frame in daily.values()):
        raise AssertionError("Daily field grids diverged")
    payload = {"daily": daily, "complete_hours": counts, "start": start, "end": end}
    market_path.parent.mkdir(parents=True, exist_ok=True)
    market_path.write_bytes(gzip.compress(pickle.dumps(payload, protocol=5), mtime=0))
    manifest = {"dataset_sha256_by_variant": {key: sha(path) for key, path in DATASETS.items()},
                "market_path": str(market_path.resolve()), "market_sha256": sha(market_path),
                "start": str(start), "end_exclusive": str(end), "daily_fields": list(FIELDS),
                "amount": "True USDT quote-asset volume from Binance original CSV column7, summed over24 complete hours; never volume*close",
                "aggregation": {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum", "amount": "sum"},
                "context_days": CONTEXT_DAYS, "all_24_hours_required": True,
                "interpolation": False, "original_archive_ohlcv_max_absolute_difference": maximum_error,
                "archive_evidence": evidence, "daily_sha256": {k: frame_digest(v) for k, v in daily.items()}}
    write_json(manifest_path, manifest)
    return manifest


def load_market(output=OUTPUT):
    import json
    path = Path(output) / "market/daily_ohlcva.pkl.gz"
    manifest = json.loads((path.parent / "manifest.json").read_text(encoding="utf-8"))
    if sha(path) != manifest["market_sha256"]:
        raise AssertionError("Prepared model inputs changed")
    return pickle.loads(gzip.decompress(path.read_bytes())), manifest


def context_frame(market, date, key):
    date = utc(date)
    dates = pd.date_range(date - pd.Timedelta(days=CONTEXT_DAYS), date, freq="D", inclusive="left")
    frame = pd.DataFrame({field: market["daily"][field].reindex(dates)[str(key)] for field in FIELDS})
    frame.index.name = "timestamp"
    if len(frame) != CONTEXT_DAYS or not np.isfinite(frame.to_numpy()).all():
        return None
    if (frame[list(FIELDS[:4])] <= 0).any().any() or (frame[["volume", "amount"]] < 0).any().any():
        return None
    if frame.volume.iloc[-30:].mean() <= 0:
        return None
    return frame


def volatility(frame):
    returns = frame.close.iloc[-31:].pct_change(fill_method=None).iloc[1:]
    if len(returns) != 30 or not np.isfinite(returns).all():
        raise AssertionError("Thirty realized daily returns must use only completed closes")
    return float(returns.std(ddof=1))

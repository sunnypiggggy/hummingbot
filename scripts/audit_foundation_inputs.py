"""Rebuild six daily model channels independently from immutable raw archives."""
from __future__ import annotations

from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
from pathlib import Path
import pickle
import re
import zipfile

import numpy as np
import pandas as pd

from scripts.foundation_model_audit_helpers import Checks, read
from scripts.audit_crypto_rl_top10 import sha, stamp

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results/backtests/crypto_foundation_models_20261024"


def audit_market(output=OUTPUT):
    output = Path(output)
    review = Checks()
    check = review.check
    metadata = read(output / "market/manifest.json")
    path = Path(metadata["market_path"])
    check(sha(path) == metadata["market_sha256"], "prepared market file SHA")
    data = pd.read_pickle(ROOT / "data/crypto_rl_top10/dataset.pkl.gz")
    raw_metadata = read(ROOT / "data/crypto_rl_top10/manifest.json")
    prior_evidence = {row["path"]: row for row in raw_metadata["archive_evidence"] if row.get("path")}
    with gzip.open(path, "rb") as stream:
        market = pickle.load(stream)
    start, end = stamp(metadata["start"]), stamp(metadata["end_exclusive"])
    index = pd.date_range(start, end, freq="h", inclusive="left")
    keys = [str(key) for key in data["hourly_close"].columns]
    ticker_to_key = {row["binance_symbol"]: str(key) for key, row in data["assets"].items()}
    amount = pd.DataFrame(np.nan, index=index, columns=keys)
    checksums, raw_files, used_hours = set(), set(), 0
    maximum_original_difference = 0.
    for row in metadata["archive_evidence"]:
        source = Path(row["path"])
        checksum_path = Path(row["checksum_path"])
        raw = source.read_bytes()
        expected = checksum_path.read_text(encoding="utf-8").split()[0].lower()
        check(hashlib.sha256(raw).hexdigest() == expected == row["sha256"] == prior_evidence[str(source)]["sha256"],
              "original raw archive hash: " + source.name)
        check(sha(checksum_path) == row["checksum_sha256"] == prior_evidence[str(source)]["checksum_sha256"],
              "original checksum text hash: " + source.name)
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            files = [name for name in archive.namelist() if name.endswith(".csv")]
            check(len(files) == 1, "one CSV member: " + source.name)
            frame = pd.read_csv(archive.open(files[0]), header=None)
        frame = frame.loc[frame[0].astype(str).str.fullmatch(r"\d+")].copy()
        check(frame.shape[1] == 12, "twelve official kline fields: " + source.name)
        epochs = pd.to_numeric(frame[0], errors="raise").astype("int64")
        lengths = epochs.astype(str).str.len().unique()
        check(len(lengths) == 1 and lengths[0] in (13, 16), "known epoch units per archive: " + source.name)
        unit = "us" if lengths[0] == 16 else "ms"
        check(row["timestamp_unit"] == unit, "original epoch-unit evidence: " + source.name)
        stamps = pd.DatetimeIndex(pd.to_datetime(epochs, unit=unit, utc=True))
        check(stamps.is_unique and stamps.equals(stamps.floor("h")), "unique exact hour opens: " + source.name)
        frame.index = stamps
        frame = frame.loc[(frame.index >= start) & (frame.index < end)]
        match = re.fullmatch(r"(.+)-1h-\d{4}-\d{2}(?:-\d{2})?\.zip", source.name)
        check(bool(match) and match[1] in ticker_to_key, "dated symbol maps to stable ID: " + source.name)
        key = ticker_to_key[match[1]]
        check(len(frame) == row["used_rows"], "actual selected archive row count: " + source.name)
        for col, field in enumerate(("open", "high", "low", "close", "volume"), 1):
            observed = pd.to_numeric(frame[col], errors="raise").to_numpy(float)
            original = data["hourly_" + field].loc[frame.index, key].to_numpy(float)
            check(np.array_equal(observed, original, equal_nan=True), "unchanged real " + field + ": " + source.name)
            if len(frame):
                maximum_original_difference = max(maximum_original_difference, float(np.max(np.abs(observed - original))))
        real_amount = pd.to_numeric(frame[7], errors="raise").to_numpy(float)
        check(np.isfinite(real_amount).all() and (real_amount >= 0).all(), "finite quote-asset volume: " + source.name)
        previous = amount.loc[frame.index, key].to_numpy()
        overlapping = np.isfinite(previous)
        check(np.array_equal(previous[overlapping], real_amount[overlapping]), "any overlapping quote rows agree: " + source.name)
        amount.loc[frame.index, key] = real_amount
        checksums.add(str(checksum_path))
        raw_files.add(str(source))
        used_hours += len(frame)
    arrays = {name: data["hourly_" + name].loc[index].to_numpy(float)
              for name in ("open", "high", "low", "close", "volume")}
    arrays["amount"] = amount.to_numpy(float)
    complete = np.ones(arrays["open"].shape, dtype=bool)
    for name, array in arrays.items():
        complete &= np.isfinite(array) & (array >= 0 if name in ("volume", "amount") else array > 0)
    days, count = len(index) // 24, len(keys)
    daily_count = complete.reshape(days, 24, count).sum(axis=1)
    mask = daily_count == 24
    expected_daily = {}
    channel_errors = {}
    for name, array in arrays.items():
        block = array.reshape(days, 24, count)
        if name == "open":
            values = block[:, 0]
        elif name == "close":
            values = block[:, -1]
        elif name == "high":
            values = np.max(block, axis=1)
        elif name == "low":
            values = np.min(block, axis=1)
        else:
            values = np.sum(block, axis=1)
        values = np.where(mask, values, np.nan)
        expected_daily[name] = pd.DataFrame(values, index=index[::24], columns=keys)
        actual = market["daily"][name]
        check(actual.index.equals(index[::24]) and list(actual.columns) == keys,
              "complete daily UTC axes: " + name)
        observed = actual.to_numpy(float)
        finite = np.isfinite(observed) & np.isfinite(values)
        channel_errors[name] = {"max_absolute_difference": float(np.max(np.abs(observed[finite] - values[finite]))) if finite.any() else 0.,
                                "max_relative_difference": float(np.max(np.abs(observed[finite] - values[finite]) / np.maximum(np.abs(values[finite]), 1e-30))) if finite.any() else 0.}
        # pandas groupby sum uses compensated reduction; NumPy's independent
        # reduction can differ by a few final bits for 24 nonnegative values.
        # Do not mistake that arithmetic order for a changed source candle.
        if name in ("volume", "amount"):
            same = np.allclose(observed, values, rtol=5e-15, atol=1e-12, equal_nan=True)
        else:
            same = np.array_equal(observed, values, equal_nan=True)
        check(same, "independent 24-hour real aggregation: " + name)
        serialized = actual.to_csv(index=True, index_label="timestamp", float_format="%.17g", lineterminator="\n").encode()
        check(hashlib.sha256(serialized).hexdigest() == metadata["daily_sha256"][name], "daily channel canonical digest: " + name)
    check(np.array_equal(market["complete_hours"].to_numpy(), daily_count), "joint six-channel complete-hour counts")
    check(metadata.get("interpolation") is False and metadata.get("all_24_hours_required") is True, "no interpolation or partial-day acceptance")
    check(maximum_original_difference == metadata["original_archive_ohlcv_max_absolute_difference"] == 0., "all original OHLCV channels byte-equivalent float values")
    ready31 = pd.DataFrame(mask, index=index[::24], columns=keys).rolling(31, min_periods=31).sum().eq(31)
    ready128 = pd.DataFrame(mask, index=index[::24], columns=keys).rolling(128, min_periods=128).sum().eq(128)
    volume_ready = expected_daily["volume"].rolling(30, min_periods=30).mean().gt(0)
    ready31 &= volume_ready
    ready128 &= volume_ready
    ready31.index = ready31.index + pd.Timedelta(days=1)
    ready128.index = ready128.index + pd.Timedelta(days=1)
    counts = {}
    for variant, name in (("original", "crypto_rl_top10"), ("ex_ada", "crypto_rl_top10_ex_ada")):
        other = pd.read_pickle(ROOT / "data" / name / "dataset.pkl.gz")
        univ = other["universe"].copy()
        univ["date"] = pd.to_datetime(univ.date, utc=True)
        rows = univ.loc[univ.date.ge(pd.Timestamp("2025-07-25", tz="UTC")) & univ.date.lt(end)]
        split_counts = {}
        for split, low, high in (("validation", "2025-07-25", "2025-10-23"), ("test", "2025-10-23", "2026-10-23")):
            selected = rows.loc[rows.date.ge(pd.Timestamp(low, tz="UTC")) & rows.date.lt(pd.Timestamp(high, tz="UTC"))]
            n31 = n128 = 0
            exclusions = []
            for row in selected.itertuples(index=False):
                key = str(row.asset_key)
                n31 += bool(ready31.loc[row.date, key])
                n128 += bool(ready128.loc[row.date, key])
                if ready31.loc[row.date, key] and not ready128.loc[row.date, key]:
                    exclusions.append({"date": row.date.isoformat(), "asset_key": key})
            split_counts[split] = {"slot_days": len(selected), "ready31": n31, "ready128": n128,
                                   "extra_context128_exclusions": exclusions}
            check(not exclusions, variant + "/" + split + " real amount does not change 128-day eligible slots")
        counts[variant] = split_counts
    result = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "status": "passed" if not review.failures else "failed",
              "checks_total": review.count, "checks_failed": len(review.failures), "failures": review.failures,
              "raw_archives_checked": len(raw_files), "checksum_files_checked": len(checksums), "real_quote_hour_rows": used_hours,
              "original_ohlcv_maximum_difference": maximum_original_difference, "quote_amount_is_real_column_7": True,
              "all_six_daily_channels_rebuilt_independently": True, "market_sha256": sha(path),
              "daily_reduction_differences": channel_errors,
              "sum_reduction_tolerance": {"relative": 5e-15, "absolute": 1e-12,
                                           "reason": "Independent NumPy24 sum versus pandas compensated group sum; OHLC fields require exact equality"},
              "eligibility": counts, "auditor_source_sha256": sha(Path(__file__))}
    target = output / "sources/independent_market_audit.json"
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    result = audit_market()
    print(json.dumps({key: result[key] for key in ("status", "checks_total", "checks_failed", "raw_archives_checked", "market_sha256")}, ensure_ascii=False))
    raise SystemExit(int(bool(result["checks_failed"])))

"""Extend frozen BTC/ETH hourly research candles with official 2019--2022 data.

Public Binance archives only; no credentials, training, trading, interpolation,
or writes to earlier research caches. Existing frozen prices remain unchanged.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import pickle
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.crypto_rl_data import BASE_DATA, OHLCV, _archive, _repair_gaps

SYMBOL_KEYS = {"BTCUSDT": "1", "ETHUSDT": "1027"}
START = pd.Timestamp("2019-01-01T00:00:00Z")
BOUNDARY = pd.Timestamp("2022-11-01T00:00:00Z")
TRAIN_START = pd.Timestamp("2020-01-01T00:00:00Z")
VALIDATION_START = pd.Timestamp("2025-01-01T00:00:00Z")
TEST_START = pd.Timestamp("2025-10-23T00:00:00Z")
END = pd.Timestamp("2026-10-23T00:00:00Z")
DEFAULT_SOURCE = ROOT / "data/crypto_rl_top10/dataset.pkl.gz"
DEFAULT_DESTINATION = ROOT / "data/crypto_rl_btceth_20261029"


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_frozen_source(source):
    source = Path(source).resolve()
    metadata_path = source.parent / "manifest.json"
    metadata = read_json(metadata_path)
    actual = sha(source)
    if actual != metadata["dataset_sha256"]:
        raise ValueError("Original frozen dataset differs from its source manifest")
    with gzip.open(source, "rb") as stream:
        original = pickle.load(stream)
    return original, metadata, {"path": str(source), "sha256": actual,
                                "manifest_path": str(metadata_path), "manifest_sha256": sha(metadata_path)}


def archive_jobs(destination, start=START, boundary=BOUNDARY):
    jobs = []
    for symbol in SYMBOL_KEYS:
        for month in pd.date_range(start, boundary, freq="MS", inclusive="left"):
            suffix = f"monthly/klines/{symbol}/1h/{symbol}-1h-{month:%Y-%m}.zip"
            jobs.append((symbol, f"{BASE_DATA}/{suffix}", Path(destination) / "raw" / suffix))
    return jobs


def verified_archive(job, offline=False):
    symbol, url, path = job
    existed, started = path.exists(), now()
    frame, evidence = _archive(url, path, offline)
    if frame is None:
        raise FileNotFoundError(f"Required official monthly archive unavailable; no silent omission: {url}")
    receipt_path = Path(str(path) + ".source.json")
    if receipt_path.exists():
        receipt = read_json(receipt_path)
        for field in ("url", "sha256", "checksum_sha256", "rows", "timestamp_unit"):
            if receipt.get(field) != evidence.get(field):
                raise ValueError(f"Frozen archive evidence changed ({field}): {path}")
    else:
        receipt = {**evidence, "checksum_url": url + ".CHECKSUM", "first_verified_at": now(),
                   "download_started_at": None if existed else started,
                   "download_completed_at": None if existed else now(),
                   "download_time_note": "Existing local archive time unknown" if existed else "Public archive/checksum downloaded in this call"}
        write_json(receipt_path, receipt)
    return symbol, frame, {**receipt, "receipt_path": str(receipt_path), "receipt_sha256": sha(receipt_path)}


def assemble_matrices(prefix, original, start=START, boundary=BOUNDARY, end=END):
    expected = pd.date_range(start, end, freq="h", inclusive="left", name="timestamp")
    prefix_index = expected[expected < boundary]
    old_index = expected[expected >= boundary]
    matrices = {f"hourly_{column}": pd.DataFrame(index=expected, columns=list(SYMBOL_KEYS), dtype=float)
                for column in OHLCV}
    for symbol, old_key in SYMBOL_KEYS.items():
        history = prefix[symbol]
        if history.index.has_duplicates or not history.index.is_monotonic_increasing:
            raise ValueError(f"Duplicate or unsorted new historical candles: {symbol}")
        if len(history.index.difference(prefix_index)):
            raise ValueError(f"New history overlaps or lies outside the requested prefix: {symbol}")
        for column in OHLCV:
            old = original[f"hourly_{column}"]
            if old.index.has_duplicates or not old.index.is_monotonic_increasing:
                raise ValueError("Original frozen hours are duplicated or unsorted")
            if not old_index.isin(old.index).all():
                raise ValueError("Original frozen dataset does not cover the expected later period")
            target = matrices[f"hourly_{column}"]
            target.loc[history.index, symbol] = history[column].to_numpy()
            target.loc[old_index, symbol] = old.loc[old_index, old_key].to_numpy()
            # Equality includes the exact pre-existing NaN locations. Never fill
            # the earlier study's exchange-closure hole or change its prices.
            pd.testing.assert_series_equal(target.loc[old_index, symbol], old.loc[old_index, old_key],
                                           check_names=False, check_exact=True)
    return matrices


def validate_coverage(matrices, validation_start=VALIDATION_START, test_start=TEST_START, end=END):
    reference = matrices["hourly_open"]
    expected = pd.date_range(reference.index[0], end, freq="h", inclusive="left", name="timestamp")
    for column in OHLCV:
        frame = matrices[f"hourly_{column}"]
        if not frame.index.equals(expected) or frame.index.has_duplicates or list(frame.columns) != list(SYMBOL_KEYS):
            raise ValueError("Hourly grid/column contract violated")
    coverage = {}
    for symbol in SYMBOL_KEYS:
        frame = pd.DataFrame({column: matrices[f"hourly_{column}"][symbol] for column in OHLCV})
        missing = frame.isna().all(axis=1)
        if (frame.isna().any(axis=1) & ~missing).any():
            raise ValueError(f"Partially missing OHLCV row: {symbol}")
        observed = frame.loc[~missing]
        if not np.isfinite(observed.to_numpy()).all() or (observed[["open", "high", "low", "close"]] <= 0).any().any() or (observed.volume < 0).any():
            raise ValueError(f"Invalid observed OHLCV: {symbol}")
        if (observed.high < observed[["open", "close", "low"]].max(axis=1)).any() or (observed.low > observed[["open", "close", "high"]].min(axis=1)).any():
            raise ValueError(f"Inconsistent observed OHLCV: {symbol}")
        for name, left, right in (("validation", validation_start, test_start), ("test", test_start, end)):
            section = missing.loc[(missing.index >= left) & (missing.index < right)]
            if int(section.sum()):
                raise ValueError(f"{name} must be complete; missing {int(section.sum())} hours for {symbol}")
        coverage[symbol] = {"expected_hours": len(frame), "observed_hours": len(observed),
                            "missing_hours": int(missing.sum()), "complete_fraction": float((~missing).mean()),
                            "missing_timestamps_utc": frame.index[missing].astype(str).tolist(),
                            "no_interpolation": True, "start": expected[0].isoformat(), "end_exclusive": end.isoformat(),
                            "validation_missing_hours": 0, "test_missing_hours": 0}
    return coverage


def write_frozen_dataset(path, payload):
    encoded = gzip.compress(pickle.dumps(payload, protocol=5), mtime=0)
    digest = hashlib.sha256(encoded).hexdigest()
    path = Path(path)
    if path.exists():
        if sha(path) != digest:
            raise FileExistsError("Refuse to overwrite an existing frozen dataset with different bytes")
    else:
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_bytes(encoded)
        temporary.replace(path)
    return digest


def prepare_dataset(source=DEFAULT_SOURCE, destination=DEFAULT_DESTINATION, *, workers=6, offline=False):
    if not 1 <= workers <= 6:
        raise ValueError("Use between one and six public archive workers")
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if destination == source.parent or destination.is_relative_to(source.parent):
        raise ValueError("New data must use a separate cache, outside the original source cache")
    original, old_manifest, old_evidence = load_frozen_source(source)
    destination.mkdir(parents=True, exist_ok=True)
    jobs, pieces, sources = archive_jobs(destination), {s: [] for s in SYMBOL_KEYS}, []
    print(f"Verifying/downloading {len(jobs)} public archive/checksum pairs using {workers} workers", flush=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(verified_archive, job, offline) for job in jobs]
        for number, future in enumerate(as_completed(futures), 1):
            symbol, frame, evidence = future.result()
            pieces[symbol].append(frame)
            sources.append(evidence)
            print(f"archive {number}/{len(jobs)} {Path(evidence['path']).name} SHA verified", flush=True)
    prefix, repairs = {}, {}
    prefix_index = pd.date_range(START, BOUNDARY, freq="h", inclusive="left", name="timestamp")
    for symbol in SYMBOL_KEYS:
        frame = pd.concat(pieces[symbol]).sort_index()
        if frame.index.has_duplicates:
            raise ValueError(f"Overlapping official archive rows: {symbol}")
        frame = frame.loc[(frame.index >= START) & (frame.index < BOUNDARY)]
        frame, evidence = _repair_gaps(frame, prefix_index, {"binance_symbol": symbol}, destination, offline)
        prefix[symbol], repairs[symbol] = frame, evidence
        print(f"{symbol} prefix: {len(frame)}/{len(prefix_index)} hours, REST gap groups {len(evidence)}", flush=True)
    matrices = assemble_matrices(prefix, original)
    coverage = validate_coverage(matrices)
    assets = {symbol: {**original["assets"][key], "asset_key": symbol, "symbol": symbol[:-4],
                       "binance_symbol": symbol, "original_frozen_asset_key": key}
              for symbol, key in SYMBOL_KEYS.items()}
    contract = {"schema_version": 1, "scope": "isolated public-data research; no production or training",
                "data_start": START.isoformat(), "end_exclusive": END.isoformat(),
                "train": [TRAIN_START.isoformat(), VALIDATION_START.isoformat()],
                "validation": [VALIDATION_START.isoformat(), TEST_START.isoformat()],
                "test": [TEST_START.isoformat(), END.isoformat()],
                "hourly_timestamp": "UTC hour opening time; OHLCV for the following closed hour",
                "source_dataset_sha256": old_evidence["sha256"],
                "source_extension_boundary": BOUNDARY.isoformat(), "no_interpolation": True}
    payload = {**matrices, "assets": assets, "registry": assets, "manifest": contract}
    dataset_path = destination / "dataset.pkl.gz"
    dataset_sha = write_frozen_dataset(dataset_path, payload)
    manifest_path = destination / "manifest.json"
    sources.sort(key=lambda row: row["url"])
    source_scripts = {str(path.relative_to(ROOT)): sha(path) for path in
                      (Path(__file__).resolve(), ROOT / "scripts/crypto_rl_data.py")}
    manifest = {**contract, "created_at": now(), "dataset_path": str(dataset_path), "dataset_sha256": dataset_sha,
                "gzip_mtime": 0, "original_frozen_source": old_evidence,
                "source_dataset_unchanged": sha(source) == old_evidence["sha256"],
                "source_manifest_unchanged": sha(source.parent / "manifest.json") == old_evidence["manifest_sha256"],
                "preservation_check": "Every later hourly float/NaN preserved exactly for BTC/ETH, OHLCV five columns",
                "original_hourly_archive_evidence": [row for row in old_manifest["archive_evidence"] if any('/' + symbol + '/' in row["url"] for symbol in SYMBOL_KEYS)],
                "archive_evidence": sources, "rest_gap_evidence": repairs,
                "original_gap_evidence": {symbol: old_manifest["coverage"][key]["rest_gap_evidence"] for symbol, key in SYMBOL_KEYS.items()},
                "coverage": coverage, "symbols": list(SYMBOL_KEYS), "hour_count": len(matrices["hourly_open"]),
                "source_code_sha256": source_scripts, "worker_count": workers,
                "training_gap_policy": "Retain missing rows; downstream training must reject affected episodes; validation and test cannot omit any hours",
                "execution_resolution": "Hourly OHLCV only; cannot infer minute01 opening prices before the separate minute cache starts"}
    if not manifest["source_dataset_unchanged"] or not manifest["source_manifest_unchanged"]:
        raise AssertionError("Original research source changed during data preparation")
    if manifest_path.exists():
        prior = read_json(manifest_path)
        if prior["dataset_sha256"] != dataset_sha or prior["original_frozen_source"] != old_evidence or prior["archive_evidence"] != sources:
            raise FileExistsError("Refuse to overwrite different frozen data evidence")
        manifest = prior
    else:
        write_json(manifest_path, manifest)
    print(json.dumps({"dataset": str(dataset_path), "sha256": dataset_sha, "coverage": coverage}, ensure_ascii=False), flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--destination", type=Path, default=DEFAULT_DESTINATION)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    prepare_dataset(args.source, args.destination, workers=args.workers, offline=args.offline)


if __name__ == "__main__":
    main()

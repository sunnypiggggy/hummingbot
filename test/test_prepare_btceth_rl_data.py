"""Offline source-preservation and missing-price tests for BTC/ETH extension."""
from copy import deepcopy
import gzip
import hashlib
import json
import pickle

import numpy as np
import pandas as pd
import pytest

from scripts.prepare_btceth_rl_data import (
    OHLCV, SYMBOL_KEYS, assemble_matrices, load_frozen_source,
    validate_coverage, verified_archive, write_frozen_dataset,
)


def sample():
    start = pd.Timestamp("2024-01-01T00:00Z")
    boundary, end = start + pd.Timedelta(hours=24), start + pd.Timedelta(hours=72)
    old_index = pd.date_range(boundary, end, freq="h", inclusive="left", name="timestamp")
    old = {"hourly_" + c: pd.DataFrame({"1": np.arange(48, dtype=float) + 100,
                                       "1027": np.arange(48, dtype=float) + 50}, index=old_index) for c in OHLCV}
    prefix = {s: pd.DataFrame(100., columns=OHLCV, index=pd.date_range(start, boundary, freq="h", inclusive="left", name="timestamp")) for s in SYMBOL_KEYS}
    return start, boundary, end, old, prefix


def test_earlier_history_preserves_all_old_values_and_known_gap():
    start, boundary, end, old, prefix = sample()
    for column in OHLCV:
        old["hourly_" + column].iloc[0] = np.nan
    before = deepcopy(old)
    frames = assemble_matrices(prefix, old, start, boundary, end)
    for column in OHLCV:
        pd.testing.assert_frame_equal(old["hourly_" + column], before["hourly_" + column], check_exact=True)
        for symbol, key in SYMBOL_KEYS.items():
            pd.testing.assert_series_equal(frames["hourly_" + column].loc[boundary:, symbol], old["hourly_" + column][key], check_names=False, check_exact=True)


def test_missing_training_price_stays_nan_without_interpolation():
    start, boundary, end, old, prefix = sample()
    stamp = start + pd.Timedelta(hours=7)
    for symbol in prefix:
        prefix[symbol] = prefix[symbol].drop(stamp)
    frames = assemble_matrices(prefix, old, start, boundary, end)
    for matrix in frames.values():
        assert matrix.loc[stamp].isna().all()
    coverage = validate_coverage(frames, boundary, boundary + pd.Timedelta(days=1), end)
    assert coverage["BTCUSDT"]["missing_hours"] == 1
    assert coverage["ETHUSDT"]["test_missing_hours"] == 0


def test_missing_validation_or_test_hour_fails():
    start, boundary, end, old, prefix = sample()
    for c in OHLCV:
        old["hourly_" + c].iloc[0] = np.nan
    frames = assemble_matrices(prefix, old, start, boundary, end)
    with pytest.raises(ValueError, match="validation must be complete"):
        validate_coverage(frames, boundary, boundary + pd.Timedelta(days=1), end)


def test_partially_missing_ohlcv_is_rejected():
    start, boundary, end, old, prefix = sample()
    frames = assemble_matrices(prefix, old, start, boundary, end)
    frames["hourly_volume"].iloc[0, 0] = np.nan
    with pytest.raises(ValueError, match="Partially missing"):
        validate_coverage(frames, boundary, boundary + pd.Timedelta(days=1), end)


def test_overlapping_prefix_is_rejected():
    start, boundary, end, old, prefix = sample()
    prefix["BTCUSDT"].loc[boundary] = 100.
    with pytest.raises(ValueError, match="overlaps"):
        assemble_matrices(prefix, old, start, boundary, end)


def test_frozen_output_is_repeatable_and_never_overwritten(tmp_path):
    path = tmp_path / "dataset.pkl.gz"
    payload = {"prices": np.array([1., np.nan, 2.])}
    first = write_frozen_dataset(path, payload)
    encoded = path.read_bytes()
    assert int.from_bytes(encoded[4:8], "little") == 0
    assert write_frozen_dataset(path, payload) == first
    assert path.read_bytes() == encoded
    with pytest.raises(FileExistsError, match="Refuse"):
        write_frozen_dataset(path, {"prices": np.array([3.])})
    assert path.read_bytes() == encoded


def test_source_manifest_sha_failure_before_unpickling(tmp_path):
    path = tmp_path / "dataset.pkl.gz"
    path.write_bytes(b"not trusted pickle")
    (tmp_path / "manifest.json").write_text(json.dumps({"dataset_sha256": "0" * 64}), encoding="utf-8")
    with pytest.raises(ValueError, match="differs"):
        load_frozen_source(path)


def test_archive_missing_failure_is_not_silently_omitted(tmp_path, monkeypatch):
    monkeypatch.setattr("scripts.prepare_btceth_rl_data._archive", lambda *a: (None, {"missing": True}))
    with pytest.raises(FileNotFoundError, match="no silent omission"):
        verified_archive(("BTCUSDT", "https://example.test/missing.zip", tmp_path / "missing.zip"))


def test_cached_source_receipt_pins_checksum_and_archive_hash(tmp_path, monkeypatch):
    path = tmp_path / "archive.zip"
    path.write_bytes(b"archive")
    evidence = {"url": "https://example.test/a.zip", "sha256": "a" * 64,
                "checksum_sha256": "b" * 64, "rows": 1, "timestamp_unit": "ms"}
    receipt = dict(evidence, checksum_sha256="c" * 64)
    (tmp_path / "archive.zip.source.json").write_text(json.dumps(receipt), encoding="utf-8")
    monkeypatch.setattr("scripts.prepare_btceth_rl_data._archive", lambda *a: (pd.DataFrame(), evidence))
    with pytest.raises(ValueError, match="checksum_sha256"):
        verified_archive(("BTCUSDT", evidence["url"], path), offline=True)

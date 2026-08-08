"""Small version-local I/O helpers for v22 artifacts."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

import pandas as pd


PAIRS = ("BTC-FDUSD", "ETH-FDUSD")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    os.replace(temporary, path)


def load_candles(cache_dir: Path, *, nested: bool = False,
                 start_ts: int | None = None, end_ts: int | None = None) -> dict[str, pd.DataFrame]:
    base = cache_dir / "extended_candles" if nested else cache_dir
    output: dict[str, pd.DataFrame] = {}
    for pair in PAIRS:
        frame = pd.read_csv(base / f"binance_{pair}_5m.csv")
        frame["timestamp"] = pd.to_numeric(frame.timestamp, errors="raise").astype("int64")
        if frame.timestamp.max() > 10_000_000_000:
            frame["timestamp"] //= 1000
        frame = frame.drop_duplicates("timestamp", keep="last").sort_values("timestamp")
        if start_ts is not None:
            frame = frame[frame.timestamp >= int(start_ts)]
        if end_ts is not None:
            frame = frame[frame.timestamp < int(end_ts)]
        if frame.empty or not frame.timestamp.diff().dropna().eq(300).all():
            raise RuntimeError(f"{pair} candle sequence is incomplete")
        output[pair] = frame.reset_index(drop=True)
    return output


def combined_file_hash(paths: Mapping[str, Path]) -> str:
    digest = hashlib.sha256()
    for key in sorted(paths):
        digest.update(key.encode()); digest.update(sha256_file(paths[key]).encode())
    return digest.hexdigest()

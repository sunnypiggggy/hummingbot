"""Public, point-in-time-lagged universe and hourly data for local PPO research.

No credentials, account endpoint, production state, or current exchangeInfo
selection is used. CMC historical pages are retrospective provider snapshots,
not a guarantee of their original publication vintage. A conservative two-day
date-label lag is imposed before monthly and split-boundary universe selection.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
from pathlib import Path
import pickle
import re
import threading
import time
from typing import Any
import urllib.robotparser
import zipfile

import numpy as np
import pandas as pd
import requests


ROOT = Path(__file__).resolve().parents[1]
DATA_START = pd.Timestamp("2022-11-01T00:00Z")
TRAIN_START = pd.Timestamp("2023-01-01T00:00Z")
VAL_START = pd.Timestamp("2025-07-25T00:00Z")
TEST_START = pd.Timestamp("2025-10-23T00:00Z")
END = pd.Timestamp("2026-10-23T00:00Z")
BASE_DATA = "https://data.binance.vision/data/spot"
BASE_REST = "https://data-api.binance.vision/api/v3/klines"
OHLCV = ["open", "high", "low", "close", "volume"]
_LOCAL = threading.local()

# Asset design, rather than price-near-one, determines this exclusion. Snapshot
# tags supplement this explicit list; all decisions retain their exact reasons.
STABLE_SYMBOLS = {
    "USDT", "USDC", "BUSD", "DAI", "USDE", "USDD", "USDS", "TUSD", "FDUSD",
    "FRAX", "USDP", "PAX", "GUSD", "PYUSD", "RLUSD", "USD1", "USD0", "USDF",
    "USDX", "USDY", "USDTB", "USDG", "EURC", "EURS", "EURT", "UST", "USTC",
}
WRAPPED_SYMBOLS = {"WBTC", "STETH", "WSTETH", "WBETH", "BETH", "BTCB", "CBETH", "RETH", "WEETH", "EZETH", "RSETH", "CBBTC", "WETH"}
EVENTS = {
    "328": {"binance_symbol": "XMRUSDT", "until": "2024-02-20T03:00:00Z", "delisting_notice_at": "2024-02-06T07:00:00Z",
            "source": "https://www.binance.com/en/support/announcement/detail/f73b083ba6834771b07dbe5319917ae5"},
    "3890": {"binance_symbol": "MATICUSDT", "until": "2024-09-10T03:00:00Z", "delisting_notice_at": "2024-08-28T09:00:00Z",
             "source": "https://www.binance.com/en/support/announcement/detail/6a6de383727f4659a3050f7982e1620f"},
    "20314": {"binance_symbol": "LUNAUSDT", "from": "2022-05-31T06:00:00Z",
              "source": "https://www.binance.com/en/support/announcement/detail/d044a6742e484b77a170111460b0eed3",
              "note": "New Terra LUNA only; Binance LUNAUSDT chart contains old Terra Classic history, which must be clipped."},
    "4172": {"binance_symbol": "LUNCUSDT", "from": "2022-09-09T08:00:00Z",
             "source": "https://www.binance.com/en/support/announcement/detail/d044a6742e484b77a170111460b0eed3",
             "note": "Terra Classic UCID4172 is distinct from new Terra UCID20314; use actual LUNCUSDT spot, never concatenate LUNA."},
    "32196": {"binance_symbol": "HYPEUSDT", "from": "2026-09-24T11:00:00Z",
              "source": "https://www.binance.com/en-BH/support/announcement/detail/d49de208bb6d4a26b02677d7c3d89b5d"},
    "11419": {"binance_symbol": "TONUSDT", "until": "2026-06-30T03:00:00Z", "delisting_notice_at": "2026-06-12T06:00:00Z",
              "source": "https://www.binance.com/en/support/announcement/detail/fe307fba935b44698fde4db01e84a7eb",
              "note": "Old TON ticker only; documented 1:1 GRAM rebranding is not silently joined. This asset leaves the selected pool before the event."},
}


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _session() -> requests.Session:
    if not hasattr(_LOCAL, "session"):
        _LOCAL.session = requests.Session()
    return _LOCAL.session


def _request(url: str, *, missing_ok: bool = False) -> bytes | None:
    for attempt in range(3):
        try:
            response = _session().get(url, timeout=40)
            if response.status_code == 404 and missing_ok:
                return None
            if response.status_code in (401, 403, 418):
                raise RuntimeError(f"Public source denied access; no bypass attempted: {url} ({response.status_code})")
            if response.status_code == 429:
                if attempt == 2:
                    response.raise_for_status()
                delay = float(response.headers.get("Retry-After", "5"))
                if delay > 60:
                    raise RuntimeError(f"Public source requests a longer cooldown: {url}")
                time.sleep(max(1, delay))
                continue
            response.raise_for_status()
            return response.content
        except (requests.ConnectionError, requests.Timeout):
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError(f"Public request failed: {url}")


def _cached_json_request(url: str, path: Path, offline: bool) -> tuple[Any, dict]:
    if path.exists():
        stored = _json(path)
        if stored.get("url") != url:
            raise ValueError(f"Cached request identity mismatch: {path}")
        return stored["data"], {k: v for k, v in stored.items() if k != "data"}
    if offline:
        raise FileNotFoundError(f"Offline cache missing: {path}")
    data = _request(url)
    decoded = json.loads(data)
    meta = {"url": url, "retrieved_at": datetime.now(timezone.utc).isoformat(), "sha256": _sha(data)}
    _write_json(path, {**meta, "data": decoded})
    return decoded, meta


def _snapshot_date(anchor: pd.Timestamp) -> pd.Timestamp:
    candidate = anchor.normalize() - pd.Timedelta(days=2)
    return candidate - pd.Timedelta(days=(candidate.dayofweek + 1) % 7)


def _load_snapshot(date: pd.Timestamp, cache: Path, offline: bool) -> tuple[list[dict], dict]:
    url = f"https://coinmarketcap.com/historical/{date:%Y%m%d}/"
    path = cache / "snapshots" / f"{date:%Y%m%d}.json"
    if path.exists():
        stored = _json(path)
        if stored["metadata"]["snapshot_date"] != date.isoformat():
            raise ValueError(f"Snapshot date mismatch: {path}")
        return stored["rows"], stored["metadata"]
    if offline:
        raise FileNotFoundError(f"Offline snapshot missing: {path}")
    raw = _request(url)
    match = re.search(rb'<script[^>]*id=["\']__NEXT_DATA__["\'][^>]*>(.*?)</script>', raw, re.S)
    if not match:
        raise ValueError(f"Public snapshot has no readable embedded data: {url}")
    page = json.loads(match.group(1))
    if page["props"]["pageProps"]["dateWithHyphens"] != date.strftime("%Y-%m-%d"):
        raise ValueError(f"Provider returned another snapshot date: {url}")
    state = page["props"]["initialState"]
    state = json.loads(state) if isinstance(state, str) else state
    rows = state["cryptocurrency"]["listingHistorical"]["data"]
    selected = []
    for asset in rows:
        quote = asset["quote"].get("USD", asset["quote"])
        cap = float(quote.get("marketCap", quote.get("market_cap", 0)))
        rank = int(asset.get("cmcRank", asset.get("rank", 0)))
        if rank < 1 or not np.isfinite(cap) or cap <= 0:
            continue
        selected.append({"id": int(asset["id"]), "name": str(asset["name"]), "symbol": str(asset["symbol"]),
                         "slug": str(asset["slug"]), "cmc_rank": rank, "market_cap": cap,
                         "tags": asset.get("tags", []), "provider_date_label": asset.get("lastUpdated")})
    selected.sort(key=lambda a: (a["cmc_rank"], a["id"]))
    if len(selected) < 20 or len({a["id"] for a in selected}) != len(selected):
        raise ValueError(f"Insufficient or duplicate ranked snapshot data: {url}")
    metadata = {"url": url, "snapshot_date": date.isoformat(), "available_after": (date + pd.Timedelta(days=2)).isoformat(),
                "retrieved_at": datetime.now(timezone.utc).isoformat(), "html_sha256": _sha(raw),
                "records": len(selected), "provider_version": "retrospective historical snapshot; original publication vintage unverified"}
    raw_path = cache / "snapshots" / f"{date:%Y%m%d}.html.gz"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_bytes(gzip.compress(raw, mtime=0))
    metadata["raw_gzip_sha256"] = _sha(raw_path.read_bytes())
    _write_json(path, {"metadata": metadata, "rows": selected})
    return selected, metadata


def _exclude(asset: dict) -> str | None:
    symbol, name = asset["symbol"].upper(), asset["name"].lower()
    tags = {str(t.get("slug", t.get("name", "")) if isinstance(t, dict) else t).lower() for t in asset.get("tags", [])}
    if symbol in STABLE_SYMBOLS or any("stablecoin" in t for t in tags):
        return "stablecoin/pegged-value design"
    if symbol in WRAPPED_SYMBOLS or "wrapped" in name or "staked" in name or tags.intersection({"wrapped-tokens", "liquid-staking-derivatives", "liquid-restaking"}):
        return "wrapped/staked duplicate underlying exposure"
    return None


def _identity(asset: dict) -> dict:
    key = str(asset["id"])
    symbol = asset["symbol"].upper()
    if not re.fullmatch(r"[A-Z0-9]{1,25}", symbol):
        raise ValueError(f"Unreviewed exchange ticker identity: UCID{key} / {symbol}")
    event = EVENTS.get(key, {})
    ticker = event.get("binance_symbol", symbol + "USDT")
    if key == "3890" and symbol != "MATIC":
        raise ValueError("UCID3890 ticker changed; review MATIC/POL identity instead of automatically joining")
    return {"asset_key": key, "cmc_id": int(key), "symbol": symbol, "name": asset["name"], "slug": asset["slug"],
            "binance_symbol": ticker, "identity_mapping": "CMC numeric identity with dated exchange ticker; no cross-UCID stitching",
            "effective_from": event.get("from"), "effective_until": event.get("until"),
            "delisting_notice_at": event.get("delisting_notice_at"),
            "last_tradable_hour": (pd.Timestamp(event["until"]) - pd.Timedelta(hours=1)).isoformat() if event.get("until") else None,
            "source": event.get("source", "https://developers.binance.com/en/docs/catalog/core-trading-spot-trading/api/rest-api/market"),
            "identity_note": event.get("note", "Historical spot trading existence is checked using closed-hour data before each selection anchor.")}


def _probe(asset: dict, anchor: pd.Timestamp, cache: Path, offline: bool) -> tuple[bool, dict]:
    identity = _identity(asset)
    since, until = identity["effective_from"], identity["effective_until"]
    if (since and anchor < pd.Timestamp(since)) or (until and anchor >= pd.Timestamp(until)):
        return False, {"reason": "outside documented spot identity/trading interval", "identity": identity}
    start = anchor - pd.Timedelta(days=1)
    url = f"{BASE_REST}?symbol={identity['binance_symbol']}&interval=1h&startTime={start.value // 10**6}&endTime={anchor.value // 10**6 - 1}&limit=24"
    path = cache / "probes" / f"{anchor:%Y%m%d}_{identity['binance_symbol']}.json"
    if path.exists():
        saved = _json(path)
    elif offline:
        raise FileNotFoundError(f"Offline historical eligibility probe missing: {path}")
    else:
        response = _session().get(url, timeout=40)
        if response.status_code in (401, 403, 418, 429):
            raise RuntimeError(f"Eligibility request denied/throttled; do not treat this as an unavailable coin: {url} ({response.status_code})")
        data = response.json()
        if response.status_code != 200 and not (response.status_code == 400 and isinstance(data, dict) and data.get("code") == -1121):
            response.raise_for_status()
        saved = {"url": url, "status": response.status_code, "data": data,
                 "retrieved_at": datetime.now(timezone.utc).isoformat(), "sha256": _sha(response.content)}
        _write_json(path, saved)
    bars = saved["data"]
    recent = isinstance(bars, list) and len(bars) > 0 and int(bars[-1][0]) == (anchor - pd.Timedelta(hours=1)).value // 10**6
    positive = recent and any(float(row[5]) > 0 for row in bars)
    return bool(positive), {"url": url, "path": str(path), "sha256": saved["sha256"], "bars": len(bars) if isinstance(bars, list) else 0,
                            "last_closed_hour_present": bool(recent), "reason": "recent closed-hour spot trade evidence" if positive else "no confirmed recent spot trade data"}


def _first_hour(identity: dict, cache: Path, offline: bool) -> tuple[pd.Timestamp, dict]:
    url = f"{BASE_REST}?symbol={identity['binance_symbol']}&interval=1h&startTime=0&limit=1"
    data, meta = _cached_json_request(url, cache / "first_hour" / f"{identity['binance_symbol']}.json", offline)
    if not isinstance(data, list) or not data:
        raise ValueError(f"No historical candle for selected identity: {identity['binance_symbol']}")
    first = pd.Timestamp(int(data[0][0]), unit="ms", tz="UTC")
    if identity["effective_from"]:
        first = max(first, pd.Timestamp(identity["effective_from"]))
    return first, meta


def _select_anchor(anchor: pd.Timestamp, snapshots: dict, cache: Path, offline: bool) -> tuple[list[dict], dict]:
    """Selection uses only dated inputs; fetch order has no economic meaning."""
    date = _snapshot_date(anchor)
    rows, meta = snapshots[date]
    chosen, considered = [], []
    for asset in rows:
        reason = _exclude(asset)
        if reason:
            considered.append({"cmc_id": asset["id"], "symbol": asset["symbol"], "rank": asset["cmc_rank"], "excluded": reason})
            continue
        accepted, evidence = _probe(asset, anchor, cache, offline)
        considered.append({"cmc_id": asset["id"], "symbol": asset["symbol"], "rank": asset["cmc_rank"], "accepted": accepted, "evidence": evidence})
        if not accepted:
            continue
        chosen.append({"asset_key": str(asset["id"]), "symbol": asset["symbol"], "cap": asset["market_cap"], "market_cap": asset["market_cap"],
                       "cmc_rank": asset["cmc_rank"], "snapshot_date": date, "available_after": date + pd.Timedelta(days=2), "anchor": anchor,
                       "_identity": _identity(asset)})
        if len(chosen) == 10:
            break
    if len(chosen) != 10:
        raise RuntimeError(f"Only {len(chosen)} confirmed eligible coins for {anchor}; do not invent replacement members")
    evidence = {"anchor": anchor.isoformat(), "snapshot": meta, "considered": considered,
                "selected_asset_keys": [r["asset_key"] for r in chosen]}
    return chosen, evidence


def _repair_gaps(frame: pd.DataFrame, expected: pd.DatetimeIndex, identity: dict, cache: Path, offline: bool) -> tuple[pd.DataFrame, list[dict]]:
    """Ask the public REST source about archive gaps; an empty result stays empty."""
    missing = expected.difference(frame.index)
    evidence, pieces, groups = [], [], []
    for stamp in missing:
        if not groups or len(groups[-1]) == 1000 or stamp != groups[-1][-1] + pd.Timedelta(hours=1):
            groups.append([])
        groups[-1].append(stamp)
    for group in groups:
        stamp, finish = group[0], group[-1] + pd.Timedelta(hours=1)
        url = f"{BASE_REST}?symbol={identity['binance_symbol']}&interval=1h&startTime={stamp.value // 10**6}&endTime={finish.value // 10**6 - 1}&limit={len(group)}"
        suffix = "" if len(group) == 1 else f"_n{len(group)}"
        data, meta = _cached_json_request(url, cache / "patches" / f"{identity['binance_symbol']}_{stamp:%Y%m%dT%H%M}{suffix}.json", offline)
        if not isinstance(data, list):
            raise ValueError(f"Gap lookup failed instead of returning valid bars: {url}")
        wanted = {t.value // 10**6 for t in group}
        accepted = [row for row in data if int(row[0]) in wanted]
        if accepted:
            times = pd.to_datetime([int(row[0]) for row in accepted], unit="ms", utc=True)
            bar = pd.DataFrame([row[1:6] for row in accepted], columns=OHLCV, index=pd.DatetimeIndex(times, name="timestamp")).astype(float)
            if bar.index.has_duplicates:
                raise ValueError(f"Duplicate REST gap candles: {url}")
            if not np.isfinite(bar.to_numpy()).all() or (bar[["open", "high", "low", "close"]] <= 0).any().any() or (bar.volume < 0).any():
                raise ValueError(f"Invalid REST gap candle: {url}")
            if (bar.high < bar[["open", "close", "low"]].max(axis=1)).any() or (bar.low > bar[["open", "close", "high"]].min(axis=1)).any():
                raise ValueError(f"Inconsistent REST gap candle: {url}")
            pieces.append(bar)
        evidence.append({**meta, "timestamp": stamp.isoformat(), "end_exclusive": finish.isoformat(), "requested_hours": len(group),
                         "returned_rows": len(accepted), "interpolated": False})
    if pieces:
        frame = pd.concat([frame, *pieces]).sort_index()
    return frame, evidence


def _archive(url: str, path: Path, offline: bool) -> tuple[pd.DataFrame | None, dict]:
    checksum_path = Path(str(path) + ".CHECKSUM")
    missing_path = Path(str(path) + ".missing.json")
    if not (path.exists() and checksum_path.exists()):
        if offline:
            if missing_path.exists():
                return None, _json(missing_path)
            raise FileNotFoundError(f"Offline archive/checksum missing: {path}")
        checksum = _request(url + ".CHECKSUM", missing_ok=True)
        if checksum is None:
            meta = {"url": url, "missing": True, "retrieved_at": datetime.now(timezone.utc).isoformat()}
            _write_json(missing_path, meta)
            return None, meta
        raw = _request(url)
        expected = checksum.decode("utf-8").strip().split()[0].lower()
        if _sha(raw) != expected:
            raise ValueError(f"Official archive checksum mismatch: {url}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        checksum_path.write_bytes(checksum)
    raw = path.read_bytes()
    expected = checksum_path.read_text(encoding="utf-8").strip().split()[0].lower()
    if not re.fullmatch(r"[0-9a-f]{64}", expected) or _sha(raw) != expected:
        raise ValueError(f"Cached official archive hash mismatch: {path}")
    with zipfile.ZipFile(io.BytesIO(raw)) as zipped:
        csvs = [n for n in zipped.namelist() if n.endswith(".csv")]
        if len(csvs) != 1:
            raise ValueError(f"Unexpected archive members: {path}")
        frame = pd.read_csv(zipped.open(csvs[0]), header=None, usecols=range(6), names=["timestamp", *OHLCV])
    frame = frame.loc[frame.timestamp.astype(str).str.fullmatch(r"\d+")].copy()
    stamps = pd.to_numeric(frame.pop("timestamp"), errors="raise").astype("int64")
    unit = "us" if len(stamps) and int(stamps.iloc[0]) > 10**14 else "ms"
    frame.index = pd.DatetimeIndex(pd.to_datetime(stamps, unit=unit, utc=True), name="timestamp")
    frame = frame.astype(float)
    values = frame.to_numpy()
    if frame.index.has_duplicates or not frame.index.is_monotonic_increasing or (frame.index != frame.index.floor("h")).any():
        raise ValueError(f"Invalid hour timestamps: {path}")
    if not np.isfinite(values).all() or (frame[["open", "high", "low", "close"]] <= 0).any().any() or (frame.volume < 0).any():
        raise ValueError(f"Invalid hour OHLCV values: {path}")
    if (frame.high < frame[["open", "close", "low"]].max(axis=1)).any() or (frame.low > frame[["open", "close", "high"]].min(axis=1)).any():
        raise ValueError(f"Inconsistent hourly candle: {path}")
    return frame, {"url": url, "path": str(path), "checksum_path": str(checksum_path), "sha256": expected,
                   "checksum_sha256": _sha(checksum_path.read_bytes()), "rows": len(frame), "timestamp_unit": unit, "verified": True}


def _hourly_jobs(identity: dict, first: pd.Timestamp, cache: Path) -> list[tuple[str, Path]]:
    begin = max(DATA_START, first.floor("h"))
    until = min(END, pd.Timestamp(identity["effective_until"])) if identity["effective_until"] else END
    if begin >= until:
        return []
    jobs = []
    ticker = identity["binance_symbol"]
    month = begin.normalize().replace(day=1)
    while month < until:
        next_month = month + pd.offsets.MonthBegin(1)
        if next_month <= END:
            suffix = f"monthly/klines/{ticker}/1h/{ticker}-1h-{month:%Y-%m}.zip"
            jobs.append((f"{BASE_DATA}/{suffix}", cache / "raw" / suffix))
        else:
            for day in pd.date_range(max(month, begin.normalize()), until.normalize(), freq="D", inclusive="left"):
                suffix = f"daily/klines/{ticker}/1h/{ticker}-1h-{day:%Y-%m-%d}.zip"
                jobs.append((f"{BASE_DATA}/{suffix}", cache / "raw" / suffix))
        month = next_month
    return jobs


def build_dataset(cache_dir: str | Path, output_dir: str | Path, offline: bool = False) -> dict:
    """Fetch/verify fixed study evidence and return hourly matrices and universe.

    Hourly matrices use UTC hour-open labels. ``universe`` contains one row per
    day/slot, with CMC ID strings as stable asset keys; its cap and market_cap
    fields both contain the same historical USD market capitalization.
    """
    cache, output = Path(cache_dir).resolve(), Path(output_dir).resolve()
    cache.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    if not offline:
        robots_url = "https://coinmarketcap.com/robots.txt"
        robots = _request(robots_url)
        parser = urllib.robotparser.RobotFileParser()
        parser.parse(robots.decode("utf-8").splitlines())
        if not parser.can_fetch(_session().headers.get("User-Agent", "python-requests"), "https://coinmarketcap.com/historical/20230101/"):
            raise RuntimeError("CMC robots disallows this client; no bypass attempted")
        if parser.crawl_delay(_session().headers.get("User-Agent", "python-requests")):
            raise RuntimeError("CMC declares crawl delay; use a paced collection workflow instead")
        (cache / "robots.txt").write_bytes(robots)
    anchors = sorted(set(pd.date_range(TRAIN_START, END, freq="MS", inclusive="left").tolist() + [VAL_START, TEST_START]))
    dates = sorted({_snapshot_date(t) for t in anchors})
    print(f"Reading {len(dates)} historical CMC snapshots for {len(anchors)} selection anchors...", flush=True)
    snapshots = {}
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = {pool.submit(_load_snapshot, date, cache, offline): date for date in dates}
        for future in as_completed(futures):
            snapshots[futures[future]] = future.result()
    registry, selections, selection_evidence = {}, {}, []
    # Independent source retrieval is concurrent; registry merging is still
    # chronological. Probes inspect only the already closed preceding 24 hours.
    with ThreadPoolExecutor(max_workers=6) as pool:
        selected = list(pool.map(lambda t: _select_anchor(t, snapshots, cache, offline), anchors))
    for anchor, (chosen, evidence) in zip(anchors, selected):
        for asset in chosen:
            key = asset["asset_key"]
            identity = asset.pop("_identity")
            if key in registry and registry[key]["binance_symbol"] != identity["binance_symbol"]:
                raise ValueError(f"Ticker identity changes within one UCID; explicit event mapping required: {key}")
            registry.setdefault(key, identity)
        selections[anchor] = chosen
        selection_evidence.append(evidence)
        print(f"{anchor:%Y-%m-%d}: " + " ".join(row["symbol"] for row in chosen), flush=True)
    print("Historical selected union: " + ", ".join(f"{a['asset_key']}:{a['symbol']}" for a in registry.values()), flush=True)
    _write_json(cache / "universe_selection.json", selection_evidence)
    rows = []
    for i, anchor in enumerate(anchors):
        until = anchors[i + 1] if i + 1 < len(anchors) else END
        for day in pd.date_range(anchor, until, freq="D", inclusive="left"):
            rows.extend({"date": day, "slot": slot, **asset} for slot, asset in enumerate(selections[anchor]))
    universe = pd.DataFrame(rows)
    firsts = {}
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = {pool.submit(_first_hour, identity, cache, offline): key for key, identity in registry.items()}
        for future in as_completed(futures):
            key = futures[future]
            firsts[key], meta = future.result()
            registry[key]["observed_first_hour"] = firsts[key].isoformat()
            registry[key]["first_hour_evidence"] = meta
    jobs = {key: _hourly_jobs(identity, firsts[key], cache) for key, identity in registry.items()}
    all_jobs = {str(path): (url, path) for group in jobs.values() for url, path in group}
    print(f"Verifying {len(all_jobs)} official hourly archive/checksum pairs with 6 workers...", flush=True)
    loaded = {}
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = {pool.submit(_archive, url, path, offline): name for name, (url, path) in all_jobs.items()}
        for n, future in enumerate(as_completed(futures), 1):
            loaded[futures[future]] = future.result()
            if n % 100 == 0 or n == len(futures):
                print(f"Hourly archives verified: {n}/{len(futures)}", flush=True)
    index = pd.date_range(DATA_START, END, freq="h", inclusive="left", name="timestamp")
    matrices = {column: pd.DataFrame(index=index, columns=sorted(registry, key=int), dtype=float) for column in OHLCV}
    coverage = {}
    for key, identity in registry.items():
        pieces = [loaded[str(path)][0] for _, path in jobs[key] if loaded[str(path)][0] is not None]
        if not pieces:
            raise ValueError(f"Selected asset has no official hourly archives: {key}")
        frame = pd.concat(pieces).sort_index()
        if frame.index.has_duplicates:
            raise ValueError(f"Overlapping hourly archive rows: {key}")
        begin = max(DATA_START, firsts[key])
        until = min(END, pd.Timestamp(identity["effective_until"])) if identity["effective_until"] else END
        frame = frame.loc[(frame.index >= begin) & (frame.index < until)]
        expected = index[(index >= begin) & (index < until)]
        frame, gap_evidence = _repair_gaps(frame, expected, identity, cache, offline)
        missing = expected.difference(frame.index)
        for column in OHLCV:
            matrices[column].loc[frame.index, key] = frame[column]
        coverage[key] = {"symbol": identity["symbol"], "rows": len(frame), "expected_within_documented_span": len(expected),
                         "missing_hours": len(missing), "missing_timestamps_utc": [str(t) for t in missing],
                         "coverage_start": frame.index[0].isoformat() if len(frame) else None,
                         "coverage_end_exclusive": (frame.index[-1] + pd.Timedelta(hours=1)).isoformat() if len(frame) else None,
                         "no_interpolation": True, "rest_gap_evidence": gap_evidence}
    manifest = {"created_at": datetime.now(timezone.utc).isoformat(), "scope": "local research only; no live authorization",
                "data_start": DATA_START.isoformat(), "train": [TRAIN_START.isoformat(), VAL_START.isoformat()],
                "validation": [VAL_START.isoformat(), TEST_START.isoformat()], "test": [TEST_START.isoformat(), END.isoformat()],
                "hourly_timestamp": "UTC hour opening time; OHLCV for the following closed hour",
                "universe_frequency": "month starts plus validation/test start; fixed within each selection interval",
                "snapshot_available_after": "snapshot date label + 2 calendar days 00:00 UTC; retrospective provider vintage",
                "eligibility": "filter stable/wrapped designs, then historical rank; latest closed-hour data before anchor plus documented identity/trading intervals",
                "known_limitations": ["Current retrieval of historical snapshots may contain retrospective provider revisions; not original publication vintage.",
                                      "Snapshot tags are retrospective metadata and are used only as asset-design exclusions, not predictive features.",
                                      "A historical trade proves market existence, not continuous account-specific permission; missing holding prices must fail in the simulator.",
                                      "No current exchangeInfo survivors filter or automatic cross-identity ticker stitching is used."],
                "registry": registry, "coverage": coverage, "selection_evidence": selection_evidence,
                "archive_evidence": [meta for _, meta in loaded.values()], "asset_count": len(registry),
                "hour_count": len(index), "universe_rows": len(universe), "worker_count": 6,
                "confirmed_market_events": [{"missing_hour": "2023-03-24T13:00:00Z", "spot_resumed_at": "2023-03-24T14:00:00Z",
                                              "source": "https://www.binance.com/en/support/announcement/detail/813a31506e9f478ea8c1058b425df87a",
                                              "source_published_at": "2023-03-24T13:38:00Z", "classification": "documented spot maintenance; no replacement candle invented"}]}
    payload = {**{f"hourly_{name}": value for name, value in matrices.items()}, "universe": universe,
               "assets": registry, "registry": registry, "manifest": manifest}
    encoded = gzip.compress(pickle.dumps(payload, protocol=5), mtime=0)
    (cache / "dataset.pkl.gz").write_bytes(encoded)
    manifest["dataset_path"] = str(cache / "dataset.pkl.gz")
    manifest["dataset_sha256"] = _sha(encoded)
    _write_json(cache / "manifest.json", manifest)
    _write_json(output / "sources/data_manifest.json", manifest)
    universe.to_csv(output / "sources/universe_daily.csv", index=False, date_format="%Y-%m-%dT%H:%M:%SZ")
    _write_json(output / "sources/asset_registry.json", registry)
    print(f"Dataset ready: {len(registry)} assets, {len(index)} hours, {len(universe)} daily slot rows", flush=True)
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "data/crypto_rl_top10")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/backtests/crypto_rl_top10_20261024")
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    build_dataset(args.cache_dir, args.output_dir, args.offline)

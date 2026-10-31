"""Bounded official-wheel range probe/download, verified against index SHA256."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import threading
import time

import requests

URL = "https://download.pytorch.org/whl/cu118/torch-2.5.1%2Bcu118-cp311-cp311-win_amd64.whl"
SIZE = 2700152906
EXPECTED = "7e09af36c3d0bf8e6554cacae794f9fcd348d096e21f8be1dd42ed3bc48b4e2f"
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/backtests/crypto_foundation_models_20261024/sources"
CACHE = Path("C:/Users/sunny/.cache/codex-research-vendors/torch-cu118-2.5.1-cp311")
LOCAL = threading.local()
CHUNK = 8 * 1024 * 1024


def download(index):
    start, end = index * CHUNK, min(SIZE - 1, (index + 1) * CHUNK - 1)
    path = CACHE / "parts" / f"{index:04d}.part"
    expected = end - start + 1
    if path.exists() and path.stat().st_size == expected:
        return {"index": index, "bytes": expected, "cached": True, "seconds": 0}
    if not hasattr(LOCAL, "session"):
        LOCAL.session = requests.Session()
    begin = time.monotonic()
    for attempt in range(3):
        try:
            with LOCAL.session.get(URL, headers={"Range": f"bytes={start}-{end}"}, stream=True, timeout=(10, 15)) as response:
                if response.status_code != 206 or response.headers.get("Content-Range") != f"bytes {start}-{end}/{SIZE}" or int(response.headers.get("Content-Length", -1)) != expected:
                    raise AssertionError("Official server returned another byte range")
                temporary = path.with_suffix(".partial")
                total = 0
                with temporary.open("wb") as stream:
                    for part in response.iter_content(1024 * 1024):
                        total += len(part)
                        if total > expected:
                            raise AssertionError("Overlong official byte range")
                        stream.write(part)
                if total != expected:
                    raise AssertionError("Incomplete official byte range")
                temporary.replace(path)
            return {"index": index, "bytes": expected, "cached": False, "seconds": time.monotonic() - begin}
        except Exception:
            if attempt == 2:
                raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--prefix", type=Path)
    args = parser.parse_args()
    (CACHE / "parts").mkdir(parents=True, exist_ok=True)
    if args.prefix:
        # A stopped pip download is an immutable contiguous prefix. Reuse only
        # complete chunks; the final assembled wheel still needs official SHA.
        with args.prefix.open("rb") as stream:
            for index in range(args.prefix.stat().st_size // CHUNK):
                raw = stream.read(CHUNK)
                if len(raw) != CHUNK:
                    raise AssertionError("Stopped pip prefix changed")
                target = CACHE / "parts" / f"{index:04d}.part"
                if not target.exists():
                    target.write_bytes(raw)
    indices = list(range(64, 72)) if args.probe else list(range((SIZE + CHUNK - 1) // CHUNK))
    started, records = time.monotonic(), []
    with ThreadPoolExecutor(max_workers=8) as pool:
        for future in as_completed([pool.submit(download, index) for index in indices]):
            records.append(future.result())
            if not args.probe and len(records) % 16 == 0:
                print(f"RANGE {len(records)}/{len(indices)} done", flush=True)
    elapsed = time.monotonic() - started
    fresh_bytes = sum(r["bytes"] for r in records if not r["cached"])
    result = {"official_url": URL, "official_index_url": "https://download.pytorch.org/whl/cu118/torch/",
              "expected_sha256": EXPECTED, "bytes": fresh_bytes, "seconds": elapsed,
              "fresh_mb_per_second": fresh_bytes / 1e6 / elapsed, "ranges": records, "probe": args.probe}
    if not args.probe:
        target = CACHE / "torch-2.5.1+cu118-cp311-cp311-win_amd64.whl"
        if target.exists():
            raise FileExistsError("Preserve existing official wheel")
        digest = hashlib.sha256()
        with target.open("wb") as stream:
            for index in range(len(indices)):
                raw = (CACHE / "parts" / f"{index:04d}.part").read_bytes()
                digest.update(raw)
                stream.write(raw)
        if target.stat().st_size != SIZE or digest.hexdigest() != EXPECTED:
            raise AssertionError("Assembled wheel differs from official index SHA256")
        result.update(wheel_path=str(target), sha256=digest.hexdigest(), official_hash_passed=True)
    output = OUT / ("cuda118_parallel_probe.json" if args.probe else "cuda118_parallel_download.json")
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("fresh_mb_per_second", "seconds", "bytes", "wheel_path", "official_hash_passed") if k in result}), flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run the unchanged live Guard plus a side-effect-free v22 observer in one container."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from ethbtc_forced_exit_contract import atomic_json, load_runtime_contract


PAIR_MAP = {"BTC-USDT": "BTC-FDUSD", "ETH-USDT": "ETH-FDUSD"}


def update_status(path: Path, contract: dict, now: float, role: str,
                  error: str | None = None) -> None:
    current = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    release = str(contract.get("release_sha256", ""))
    valid_release = bool(
        len(release) == 64
        and release != "0" * 64
        and all(character in "0123456789abcdef" for character in release.lower())
    )
    if valid_release and current.get("release_sha256") != release:
        previous_release = current.get("release_sha256")
        current = {
            "schema": "ethbtc-forced-exit-observer-status-v1",
            "release_sha256": release,
            "started_at": now,
            "cycles": 0,
            "source_errors": 0,
            "integrity_errors": 0,
            "previous_release_sha256": previous_release,
            "release_change_count": int(current.get("release_change_count", 0)) + 1,
        }
    current["last_seen_at"] = now
    current["cycles"] = int(current.get("cycles", 0)) + 1
    current["source_healthy"] = bool(contract.get("runtime_gate_healthy", contract.get("source_healthy")))
    current["execution_authorized"] = bool(contract.get("execution_authorized"))
    if role == "grid":
        mapped_events = {
            pair: contract.get("pairs", {}).get(pair, {}).get("event_id")
            for pair in PAIR_MAP.values()
        }
    else:
        mapped_events = {
            pair: contract.get("pairs", {}).get(source, {}).get("event_id")
            for pair, source in PAIR_MAP.items()
        }
    # A failed runtime view deliberately carries an all-zero synthetic release
    # and event IDs. Keep the last healthy parity evidence while counting the
    # failure; otherwise one transient error can erase the 24h audit window.
    if valid_release and all(mapped_events.values()):
        current["event_ids"] = mapped_events
    if error:
        category = "source_errors" if any(
            marker in error.lower() for marker in ("timeout", "connection", "temporarily")
        ) else "integrity_errors"
        current[category] = int(current.get(category, 0)) + 1
        current["last_error"] = error
    atomic_json(path, current)


def run_worker(role: str, child: subprocess.Popen) -> None:
    state_dir = Path(os.getenv("GRID_LIVE_STATE_PATH" if role == "grid" else "DCA_LIVE_STATE_PATH", "/workspace/state"))
    status_path = state_dir / "ethbtc_forced_exit_observer_status.json"
    contract_path = (
        state_dir / "ethbtc_forced_exit_observation.json"
        if role == "grid"
        else Path(os.getenv("DCA_V22_OBSERVATION_GATE_PATH", "/workspace/technical/ethbtc_forced_exit_observation.json"))
    )
    producer = None
    if role == "grid":
        from grid_v22_live_gate import V22LiveGateProducer
        producer = V22LiveGateProducer(
            package_dir=Path(os.getenv("GRID_V22_PACKAGE_PATH", "/workspace/v22-package")),
            cache_dir=Path(os.getenv("GRID_V22_CANDLE_PATH", "/workspace/technical-candles")),
            seed_cache_dir=Path(os.getenv("GRID_V22_SEED_CANDLE_PATH", "/workspace/research-candles")),
            state_dir=state_dir,
            authorization_path=state_dir / "observation-mode-no-authorization.json",
            refresh_binance=True,
        )
        producer.output = contract_path
    while True:
        if child.poll() is not None:
            raise RuntimeError(f"legacy {role} Guard exited with {child.returncode}")
        now = time.time()
        try:
            if producer is not None:
                producer.produce(int(now))
            contract = load_runtime_contract(
                contract_path,
                now=datetime.fromtimestamp(now, timezone.utc),
            )
            error = None if contract.get("runtime_gate_healthy") else str(contract.get("reason", "unhealthy"))
            update_status(status_path, contract, now, role, error)
        except Exception as exc:
            update_status(status_path, {}, now, role, repr(exc))
        time.sleep(30)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=("grid", "dca"), required=True)
    parser.add_argument("--guard", type=Path, required=True)
    args = parser.parse_args()
    child = subprocess.Popen([sys.executable, str(args.guard)])

    def stop(_signum, _frame):
        child.terminate()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        run_worker(args.role, child)
    finally:
        child.terminate()
        child.wait(timeout=15)
    return child.returncode or 0


if __name__ == "__main__":
    raise SystemExit(main())

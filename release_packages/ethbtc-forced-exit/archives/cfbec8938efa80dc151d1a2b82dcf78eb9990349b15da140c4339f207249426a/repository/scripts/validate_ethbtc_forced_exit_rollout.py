#!/usr/bin/env python3
"""Create approval evidence for ethbtc-forced-exit observation and OCI preflight."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from ethbtc_forced_exit_contract import atomic_json, load_runtime_contract


PAIR_MAP = {"BTC-USDT": "BTC-FDUSD", "ETH-USDT": "ETH-FDUSD"}


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def production(package: Path) -> dict:
    return load(package / "production_lock.json")


def observation(args: argparse.Namespace) -> int:
    now = int(time.time())
    lock = production(args.package)
    release = str(lock["release_sha256"])
    grid_raw = load(args.grid_state)
    dca_raw = load(args.dca_state)
    grid = grid_raw.get("v22_observation", grid_raw)
    dca = dca_raw.get("v22_observation", dca_raw)
    contract = load_runtime_contract(args.contract)
    starts = [int(grid.get("started_at", now)), int(dca.get("started_at", now))]
    ended = min(int(grid.get("last_seen_at", 0)), int(dca.get("last_seen_at", 0)))
    grid_events = grid.get("event_ids", {})
    dca_events = dca.get("event_ids", {})
    parity = all(dca_events.get(pair) == grid_events.get(source) for pair, source in PAIR_MAP.items())
    grid_cycles = max(int(grid.get("cycles", 0)), 1)
    dca_cycles = max(int(dca.get("cycles", 0)), 1)
    checks = {
        "release_matches": all(value.get("release_sha256") == release for value in (grid, dca, contract)),
        "duration_24h": ended - max(starts) >= int(lock.get("observation_required_seconds", 86400)),
        "contract_fresh_and_healthy": bool(contract.get("runtime_gate_healthy")),
        "still_observation_only": not bool(contract.get("execution_authorized")),
        "zero_grid_integrity_errors": int(grid.get("integrity_errors", 0)) == 0,
        "zero_dca_integrity_errors": int(dca.get("integrity_errors", 0)) == 0,
        "grid_source_availability_99_9pct": 1 - int(grid.get("source_errors", 0)) / grid_cycles >= .999,
        "dca_source_availability_99_9pct": 1 - int(dca.get("source_errors", 0)) / dca_cycles >= .999,
        "pair_event_parity": parity,
        "both_pairs_present": set(contract.get("pairs", {})) == {"BTC-FDUSD", "ETH-FDUSD"},
    }
    payload = {
        "schema": "ethbtc-forced-exit-observation-v1",
        "release_sha256": release, "started_at": max(starts), "ended_at": ended,
        "checks": checks, "passed": all(checks.values()),
        "grid_cycles": grid_cycles, "dca_cycles": dca_cycles,
        "event_ids": grid_events,
    }
    atomic_json(args.output, payload)
    print(json.dumps(payload, indent=2))
    return 0 if payload["passed"] else 2


def preflight(args: argparse.Namespace) -> int:
    lock = production(args.package)
    release = str(lock["release_sha256"])
    grid = load(args.grid_state)
    dca = load(args.dca_state)
    reservations = load(args.grid_reservations).get("reservations", {}).get("FDUSD", {}).get("base", {})
    managed = load(args.dca_inventory).get("pairs", {})
    grid_coverage = grid.get("shadow_preflight", {}).get("ownership_coverage", {})
    dca_coverage = dca.get("ownership_preflight", {})
    checks = {
        "grid_emergency_ready": bool(grid.get("emergency_ready")),
        "dca_emergency_ready": bool(dca.get("emergency_ready")),
        "grid_btc_eth_owned": all(float(reservations.get(asset, 0)) > 0 for asset in ("BTC", "ETH")),
        "dca_btc_eth_owned": all(float(managed.get(pair, {}).get("managed_base", 0)) > 0 for pair in PAIR_MAP),
        "grid_balance_covered": bool(grid_coverage) and all(item.get("covered") for item in grid_coverage.values()),
        "dca_balance_covered": bool(dca_coverage) and all(item.get("covered") for item in dca_coverage.values()),
        "grid_filters_and_test_orders": bool(grid.get("shadow_preflight", {}).get("test_order_no_fill")),
        "release_not_expired": int(time.time()) < int(lock["effective_end"]),
    }
    payload = {
        "schema": "ethbtc-forced-exit-preflight-v1",
        "release_sha256": release, "generated_at": int(time.time()),
        "checks": checks, "passed": all(checks.values()),
    }
    atomic_json(args.output, payload)
    print(json.dumps(payload, indent=2))
    return 0 if payload["passed"] else 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    observed = sub.add_parser("observation")
    observed.add_argument("--package", type=Path, required=True)
    observed.add_argument("--grid-state", type=Path, required=True)
    observed.add_argument("--dca-state", type=Path, required=True)
    observed.add_argument("--contract", type=Path, required=True)
    observed.add_argument("--output", type=Path, required=True)
    observed.set_defaults(handler=observation)
    checked = sub.add_parser("preflight")
    checked.add_argument("--package", type=Path, required=True)
    checked.add_argument("--grid-state", type=Path, required=True)
    checked.add_argument("--dca-state", type=Path, required=True)
    checked.add_argument("--grid-reservations", type=Path, required=True)
    checked.add_argument("--dca-inventory", type=Path, required=True)
    checked.add_argument("--output", type=Path, required=True)
    checked.set_defaults(handler=preflight)
    args = parser.parse_args()
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())

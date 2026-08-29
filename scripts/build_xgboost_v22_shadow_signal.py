#!/usr/bin/env python3
"""Produce the v22 weekly-model shadow heartbeat; never authorize Grid."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from grid_xgboost_shadow_gate_v22 import atomic_json, build_contract, failed_contract
from xgboost_v22_io import load_candles, sha256_file
from xgboost_long_risk_gate_v22 import (
    MODEL_VERSION, PAIRS, STATE_SCHEMA, GateState, build_inference_panel,
    feature_schema_sha256, run_weekly_bundle_strategy, state_from_dict, state_to_dict,
    strategy_schema_sha256, validate_weekly_bundle, week_for_timestamp,
)


DEFAULT_PACKAGE = Path("results/backtests/xgboost_grid_long_risk_gate_v22_weekly_250d/shadow_package")


def ensure_live_cache(cache_dir: Path, seed_cache_dir: Path | None) -> None:
    """Seed v22 candles without importing any legacy model implementation."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    for pair in PAIRS:
        target = cache_dir / f"binance_{pair}_5m.csv"
        if target.exists():
            continue
        if seed_cache_dir is None:
            raise FileNotFoundError(f"v22 candle cache is missing: {target}")
        source = seed_cache_dir / target.name
        if not source.exists():
            raise FileNotFoundError(f"v22 seed candle is missing: {source}")
        shutil.copy2(source, target)


try:
    from get_only_read_client import GetOnlyReadClient
except ModuleNotFoundError:
    from scripts.get_only_read_client import GetOnlyReadClient
try:
    from runtime_endpoints import binance_api_base
except ModuleNotFoundError:
    from live_guard.runtime_endpoints import binance_api_base


def refresh_binance_cache(
    cache_dir: Path, *, read_client: GetOnlyReadClient | None = None,
) -> None:
    """Append complete Binance Spot 5m candles; this is v22 self-contained I/O."""
    client = read_client or GetOnlyReadClient(binance_api_base())
    server = client.request("GET", "/api/v3/time", timeout=15)
    server_ms = int(server["serverTime"])
    for pair in PAIRS:
        path = cache_dir / f"binance_{pair}_5m.csv"
        frame = pd.read_csv(path)
        cursor = int(float(frame.timestamp.max()) * 1000) + 300_000
        additions = []
        while cursor < server_ms:
            response = client.request(
                "GET", "/api/v3/klines",
                params={"symbol": pair.replace("-", ""), "interval": "5m",
                        "startTime": cursor, "limit": 1000},
                timeout=20,
            )
            rows = [item for item in response if int(item[6]) < server_ms]
            if not rows:
                break
            additions.extend({
                "timestamp": int(item[0]) // 1000,
                "open": float(item[1]), "high": float(item[2]),
                "low": float(item[3]), "close": float(item[4]),
                "volume": float(item[5]),
            } for item in rows)
            next_cursor = int(rows[-1][0]) + 300_000
            if next_cursor <= cursor:
                break
            cursor = next_cursor
        if additions:
            merged = pd.concat([frame, pd.DataFrame(additions)], ignore_index=True)
            merged = merged.drop_duplicates("timestamp", keep="last").sort_values("timestamp")
            temporary = path.with_suffix(path.suffix + ".tmp")
            merged.to_csv(temporary, index=False)
            temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, default=DEFAULT_PACKAGE / "shadow_lock.json")
    parser.add_argument("--cache-dir", type=Path, default=Path("grid-xgboost-v22-shadow-candles"))
    parser.add_argument("--seed-cache-dir", type=Path)
    parser.add_argument("--output", type=Path, default=Path("grid-xgboost-v22-shadow-data/xgboost_risk_gate_v22_shadow.json"))
    parser.add_argument("--state", type=Path, default=Path("grid-xgboost-v22-shadow-data/xgboost_risk_gate_v22_state.json"))
    parser.add_argument("--refresh-binance", action="store_true")
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--observed-at", type=int)
    return parser.parse_args()


def combined_data_hash(cache_dir: Path) -> str:
    digest = hashlib.sha256()
    for pair in PAIRS: digest.update(sha256_file(cache_dir / f"binance_{pair}_5m.csv").encode())
    return digest.hexdigest()


def load_state(path: Path, expected: dict[str, Any], *,
               rollover_from_lock_sha256: str | None = None) -> dict[str, Any]:
    if not path.exists(): return {**expected, "schema": STATE_SCHEMA, "model_version": MODEL_VERSION, "pairs": {}}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != STATE_SCHEMA or payload.get("model_version") != MODEL_VERSION:
        raise ValueError("v22 state schema/model mismatch")
    mismatches = {key for key, value in expected.items() if payload.get(key) != value}
    metadata_only = mismatches <= {"training_data_sha256"}
    contiguous_rollover = bool(
        rollover_from_lock_sha256
        and payload.get("candidate_lock_sha256") == rollover_from_lock_sha256
        and payload.get("feature_schema_sha256") == expected["feature_schema_sha256"]
        and int(payload.get("manifest_effective_end", 0)) < int(expected["manifest_effective_end"])
    )
    if mismatches and not metadata_only and not contiguous_rollover:
        raise ValueError("v22 state hash/manifest mismatch")
    if mismatches:
        previous_effective_end = int(payload.get("manifest_effective_end", 0))
        previous_strategy = payload.get("strategy_schema_sha256")
        payload.update(expected)
        payload["rollover_from_lock_sha256"] = rollover_from_lock_sha256
        # A reviewed weekly release can extend its package-level training
        # policy (and therefore its strategy hash) without changing GateState.
        # State continuity still requires the exact predecessor lock, the same
        # feature schema, increasing coverage, and the model/state version
        # checks above; an unrelated package cannot claim this rollover.
        payload["rollover_strategy_changed"] = bool(
            previous_strategy != expected.get("strategy_schema_sha256")
        )
        if contiguous_rollover:
            # Do not let probability/arming evidence produced by the old
            # release trigger Risk-Off immediately after a weekly cutover.
            # Existing Risk-Off and recovery state remains continuous.
            for pair_state in payload.get("pairs", {}).values():
                gate = pair_state.get("gate_state", {})
                if not bool(gate.get("active")):
                    gate["above_entry_count"] = 0
                    gate["armed_until"] = None
                    gate["entry_evidence_not_before"] = previous_effective_end
                    pair_state["gate_state"] = gate
            payload["rollover_entry_evidence_reset_at"] = previous_effective_end
    return payload


def produce_once(args: argparse.Namespace) -> dict[str, Any]:
    observed = int(args.observed_at if args.observed_at is not None else time.time())
    lock = json.loads(args.lock.read_text(encoding="utf-8")); lock_hash = sha256_file(args.lock)
    if lock.get("model_version") != MODEL_VERSION or lock.get("deployment_allowed") is not False:
        raise ValueError("v22 lock version/authorization mismatch")
    model_path = Path(lock["model_path"])
    if not model_path.exists(): model_path = args.lock.parent / "models" / model_path.name
    if sha256_file(model_path) != lock["model_sha256"]: raise ValueError("v22 model hash mismatch")
    bundle = joblib.load(model_path); validate_weekly_bundle(bundle)
    if feature_schema_sha256() != lock["feature_schema_sha256"]: raise ValueError("feature hash mismatch")
    if strategy_schema_sha256(bundle["strategy_spec"]) != lock["strategy_schema_sha256"]:
        raise ValueError("strategy hash mismatch")
    ensure_live_cache(args.cache_dir, args.seed_cache_dir)
    if args.refresh_binance:
        refresh_binance_cache(
            args.cache_dir, read_client=getattr(args, "read_client", None),
        )
    candles = load_candles(args.cache_dir)
    for pair, frame in candles.items():
        if len(frame) < 45 * 24 * 12 or not np.all(np.diff(frame.timestamp.to_numpy(np.int64)) == 300):
            raise RuntimeError(f"{pair} candle history incomplete")
    panel = build_inference_panel(candles)
    panel = panel[(panel.signal_ts >= int(lock["effective_start"])) & (panel.signal_ts <= observed)].copy()
    if panel.empty: raise RuntimeError("no signed complete inference hour")
    latest = int(panel.signal_ts.max())
    # A current heartbeat is healthy only if the latest complete hour is fresh
    # and a signed fold covers it for both pairs.
    source_healthy = 0 <= observed - latest <= 5400
    expected = {"model_sha256": lock["model_sha256"], "feature_schema_sha256": lock["feature_schema_sha256"],
        "strategy_schema_sha256": lock["strategy_schema_sha256"], "candidate_lock_sha256": lock_hash,
        "training_data_sha256": lock["training_panel_sha256"],
        "manifest_effective_end": int(lock["effective_end"])}
    state_payload = load_state(
        args.state, expected,
        rollover_from_lock_sha256=lock.get("source_package_lock_sha256"),
    ); snapshots = {}; last_1h = {}; last_4h = {}
    for pair in PAIRS:
        saved = state_payload.setdefault("pairs", {}).setdefault(pair, {})
        state = state_from_dict(saved.get("gate_state", {})) if saved else GateState()
        rows = panel[(panel.pair == pair) & (panel.signal_ts > int(state.last_signal_ts or -1))]
        snapshot = saved.get("last_snapshot")
        if not rows.empty:
            values, state = run_weekly_bundle_strategy(rows, pair=pair,
                pair_bundle=bundle["pairs"][pair], state=state)
            snapshot = values.iloc[-1].drop(labels=["signal_ts"]).to_dict()
        if snapshot is None: raise RuntimeError(f"{pair} has no v22 snapshot")
        week = week_for_timestamp(bundle["pairs"][pair], int(state.last_signal_ts or latest))
        following = next((candidate for candidate in bundle["pairs"][pair]["weeks"]
                          if int(candidate["test_start"]) == int(week["test_end"])), None)
        snapshot.update({"fold": int(week["fold"]), "week_train_cutoff": int(week["train_cutoff"]),
                         "week_test_start": int(week["test_start"]), "week_test_end": int(week["test_end"]),
                         "week_model_sha256": week["model_sha256"],
                         "next_week_start": int(following["test_start"]) if following else None,
                         "next_week_end": int(following["test_end"]) if following else None,
                         "next_week_model_sha256": following["model_sha256"] if following else None,
                         "calibration_threshold": float(week["calibration_threshold"])})
        saved.update({"gate_state": state_to_dict(state), "last_snapshot": snapshot})
        snapshots[pair] = snapshot; last_1h[pair] = int(state.last_signal_ts or latest)
        last_4h[pair] = int(state.last_complete_4h_ts or latest)
    state_payload["updated_at"] = observed; atomic_json(args.state, state_payload)
    hashes = {"model_sha256": lock["model_sha256"], "feature_schema_sha256": lock["feature_schema_sha256"],
        "strategy_schema_sha256": lock["strategy_schema_sha256"],
        "training_data_sha256": lock["training_panel_sha256"], "candidate_lock_sha256": lock_hash,
        "state_sha256": sha256_file(args.state)}
    contract = build_contract(generated_at=observed, hashes=hashes, source_healthy=source_healthy,
        snapshots=snapshots, last_1h=last_1h, last_4h=last_4h,
        reason="shadow_signal_healthy" if source_healthy else "source_unhealthy")
    atomic_json(args.output, contract); return contract


def main() -> int:
    args = parse_args()
    if args.loop and args.observed_at is not None: raise ValueError("--observed-at is one-shot only")
    while True:
        try:
            value = produce_once(args); print(json.dumps({"generated_at": value["generated_at"],
                "source_healthy": value["source_healthy"], "folds": {p: value["pairs"][p]["long"]["fold"] for p in PAIRS}},
                ensure_ascii=False), flush=True)
        except Exception as exc:
            observed = int(args.observed_at if args.observed_at is not None else time.time())
            atomic_json(args.output, failed_contract(str(exc), observed))
            print(json.dumps({"error": str(exc), "fail_closed": True}, ensure_ascii=False), flush=True)
            if not args.loop: raise
        if not args.loop: return 0
        time.sleep(max(1, int(args.poll_seconds)))


if __name__ == "__main__":
    raise SystemExit(main())

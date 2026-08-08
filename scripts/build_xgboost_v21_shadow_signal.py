#!/usr/bin/env python3
"""Generate the isolated v21 long-risk shadow heartbeat; never authorize Grid."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from build_xgboost_risk_gate_signal import ensure_live_cache, refresh_binance_cache
from compare_independent_gate_ml_stops import PAIRS, load_candles
from grid_xgboost_shadow_gate_v21 import atomic_json, build_contract, failed_contract
from retrain_xgboost_long_risk_gate_250d_v19 import sha256_file
from xgboost_long_risk_gate_v21 import (
    FEATURES, MODEL_BUNDLE_SCHEMA, MODEL_VERSION, STATE_SCHEMA, GateState,
    build_inference_panel, feature_schema_sha256, run_bundle_strategy,
    state_from_dict, state_to_dict, strategy_schema_sha256, validate_strategy_bundle,
)


DEFAULT_PACKAGE = Path("results/backtests/xgboost_grid_long_risk_gate_v21_250d/shadow_package")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, default=DEFAULT_PACKAGE / "shadow_lock.json")
    parser.add_argument("--cache-dir", type=Path, default=Path("grid-xgboost-v21-shadow-candles"))
    parser.add_argument("--seed-cache-dir", type=Path)
    parser.add_argument("--output", type=Path, default=Path("grid-xgboost-v21-shadow-data/xgboost_risk_gate_v21_shadow.json"))
    parser.add_argument("--state", type=Path, default=Path("grid-xgboost-v21-shadow-data/xgboost_risk_gate_v21_state.json"))
    parser.add_argument("--refresh-binance", action="store_true")
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--observed-at", type=int, help="Deterministic one-shot timestamp; forbidden with --loop")
    return parser.parse_args()


def combined_data_hash(cache_dir: Path) -> str:
    digest = hashlib.sha256()
    for pair in PAIRS: digest.update(sha256_file(cache_dir / f"binance_{pair}_5m.csv").encode())
    return digest.hexdigest()


def load_state(path: Path, lock: dict[str, Any]) -> dict[str, Any]:
    if not path.exists():
        return {"schema": STATE_SCHEMA, "model_version": MODEL_VERSION,
                "model_sha256": lock["model_sha256"],
                "feature_schema_sha256": lock["feature_schema_sha256"],
                "strategy_schema_sha256": lock["strategy_schema_sha256"],
                "candidate_lock_sha256": lock["candidate_lock_sha256"],
                "training_data_sha256": lock["training_panel_sha256"], "pairs": {}}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != STATE_SCHEMA:
        raise ValueError("state schema mismatch")
    if payload.get("model_version") != MODEL_VERSION:
        raise ValueError("state model version mismatch")
    expected = {
        "model_sha256": lock["model_sha256"],
        "feature_schema_sha256": lock["feature_schema_sha256"],
        "strategy_schema_sha256": lock["strategy_schema_sha256"],
        "candidate_lock_sha256": lock["candidate_lock_sha256"],
        "training_data_sha256": lock["training_panel_sha256"],
    }
    if any(payload.get(key) != value for key, value in expected.items()):
        raise ValueError("state model/feature hash mismatch")
    return payload


def validate_cache(candles: dict[str, pd.DataFrame]) -> None:
    minimum = 45 * 24 * 12
    for pair, frame in candles.items():
        if len(frame) < minimum: raise RuntimeError(f"{pair} needs at least 45 days of 5m candles")
        timestamps = frame.timestamp.to_numpy(np.int64)
        if not np.all(np.diff(timestamps) == 300): raise RuntimeError(f"{pair} candle gap detected")


def produce_once(args: argparse.Namespace) -> dict[str, Any]:
    observed_ts = int(args.observed_at if args.observed_at is not None else time.time())
    lock = json.loads(args.lock.read_text(encoding="utf-8"))
    if lock.get("model_version") != MODEL_VERSION or lock.get("deployment_allowed") is not False:
        raise ValueError("shadow lock version/authorization mismatch")
    if lock.get("promotion_authorized") is not False or lock.get("historical_verdict") != "NO-GO":
        raise ValueError("shadow lock must preserve NO-GO and deny promotion")
    model_path = Path(lock["model_path"])
    if not model_path.exists():
        # Portable read-only package mount used by the isolated container.
        packaged = args.lock.parent / "models" / model_path.name
        if not packaged.exists():
            raise FileNotFoundError(f"model package is missing: {model_path}")
        model_path = packaged
    if sha256_file(model_path) != lock["model_sha256"]: raise ValueError("model hash mismatch")
    bundle = joblib.load(model_path)
    validate_strategy_bundle(bundle)
    if feature_schema_sha256() != lock["feature_schema_sha256"]: raise ValueError("feature schema mismatch")
    if strategy_schema_sha256() != lock["strategy_schema_sha256"]: raise ValueError("strategy schema mismatch")
    ensure_live_cache(args.cache_dir, args.seed_cache_dir)
    if args.refresh_binance: refresh_binance_cache(args.cache_dir)
    candles, _ = load_candles(args.cache_dir); validate_cache(candles)
    panel = build_inference_panel(candles)
    panel = panel[panel.signal_ts <= observed_ts].copy()
    if panel.empty: raise RuntimeError("no complete inference hour at observed time")
    latest_signal = int(panel.signal_ts.max())
    source_healthy = 0 <= observed_ts - latest_signal <= 5400
    state_payload = load_state(args.state, lock); pair_snapshots = {}
    last_1h, last_4h = {}, {}
    for pair in PAIRS:
        saved = state_payload.setdefault("pairs", {}).setdefault(pair, {})
        state = state_from_dict(saved.get("gate_state", {})) if saved else GateState()
        rows = panel[(panel.pair == pair) &
                     (panel.signal_ts > int(state.last_signal_ts or -1))].sort_values("signal_ts").copy()
        pair_bundle = bundle["pairs"][pair]; features = list(FEATURES[pair])
        if features != list(pair_bundle["features"]): raise ValueError(f"{pair} feature order mismatch")
        snapshot = saved.get("last_snapshot")
        if not rows.empty:
            strategy_rows, state = run_bundle_strategy(
                rows, pair=pair, pair_bundle=pair_bundle, state=state,
            )
            snapshot = strategy_rows.iloc[-1].drop(labels=["signal_ts"]).to_dict()
        if snapshot is None: raise RuntimeError(f"{pair} has no state snapshot")
        saved.update({"gate_state": state_to_dict(state), "last_snapshot": snapshot})
        pair_snapshots[pair] = snapshot; last_1h[pair] = int(state.last_signal_ts or latest_signal)
        last_4h[pair] = int(state.last_complete_4h_ts or latest_signal)
    state_payload["updated_at"] = observed_ts
    atomic_json(args.state, state_payload); state_hash = sha256_file(args.state)
    contract = build_contract(generated_at=observed_ts, model_sha256=lock["model_sha256"],
        feature_sha256=lock["feature_schema_sha256"], strategy_sha256=lock["strategy_schema_sha256"],
        training_data_sha256=lock["training_panel_sha256"],
        candidate_lock_sha256=lock["candidate_lock_sha256"], state_sha256=state_hash,
        source_healthy=source_healthy, pair_snapshots=pair_snapshots,
        last_complete_1h=last_1h, last_complete_4h=last_4h,
        reason="shadow_signal_healthy" if source_healthy else "source_unhealthy")
    atomic_json(args.output, contract); return contract


def main() -> int:
    args = parse_args()
    if args.loop and args.observed_at is not None: raise ValueError("--observed-at is one-shot only")
    while True:
        try:
            contract = produce_once(args)
            print(json.dumps({"generated_at": contract["generated_at"],
                              "source_healthy": contract["source_healthy"],
                              "recommended_buy_enabled": {pair: contract["pairs"][pair]["long"]["recommended_buy_enabled"] for pair in PAIRS}}, ensure_ascii=False), flush=True)
        except Exception as exc:
            observed = int(args.observed_at if args.observed_at is not None else time.time())
            contract = failed_contract(str(exc), observed); atomic_json(args.output, contract)
            print(json.dumps({"error": str(exc), "fail_closed": True}, ensure_ascii=False), flush=True)
            if not args.loop: raise
        if not args.loop: return 0
        time.sleep(max(1, int(args.poll_seconds)))


if __name__ == "__main__":
    raise SystemExit(main())

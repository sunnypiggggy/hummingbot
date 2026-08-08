#!/usr/bin/env python3
"""Produce the pair-specific XGBoost v16 long-only BUY-gate heartbeat."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd
import requests

import backtest_xgboost_long_risk_gate_180d as base
import optimize_xgboost_dual_risk_gate_180d_v5 as v5
from compare_independent_gate_ml_stops import PAIRS, load_candles
from grid_xgboost_risk_gate import (
    MODEL_VERSION, STALE_AFTER_SECONDS, atomic_json, build_contract,
    combine_pair_channels, feature_schema_hash,
)
from tune_xgboost_momentum_stop_v2 import sha256_file


DEFAULT_LOCK = Path("results/backtests/xgboost_grid_long_risk_gate_v16/locked_configuration.json")
DEFAULT_OUTPUT = Path("data/xgboost_risk_gate.json")
DEFAULT_STATE = Path("data/xgboost_risk_gate_state.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--cache-dir", type=Path, default=Path("data/backtesting_candles"))
    parser.add_argument("--seed-cache-dir", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--source-max-age-seconds", type=int, default=5400)
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--refresh-binance", action="store_true")
    return parser.parse_args()


def combined_data_hash(cache_dir: Path) -> str:
    digest = hashlib.sha256()
    for pair in PAIRS:
        digest.update(sha256_file(cache_dir / f"binance_{pair}_5m.csv").encode())
    return digest.hexdigest()


def ensure_live_cache(cache_dir: Path, seed_cache_dir: Path | None) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    for pair in PAIRS:
        target = cache_dir / f"binance_{pair}_5m.csv"
        if target.exists():
            continue
        if seed_cache_dir is None:
            raise FileNotFoundError(f"live candle cache is missing: {target}")
        source = seed_cache_dir / target.name
        if not source.exists():
            raise FileNotFoundError(f"research seed candle is missing: {source}")
        shutil.copy2(source, target)


def refresh_binance_cache(cache_dir: Path) -> None:
    """Append only complete Binance Spot 5m candles to the local feature cache."""
    server = requests.get("https://api.binance.com/api/v3/time", timeout=15)
    server.raise_for_status()
    server_ms = int(server.json()["serverTime"])
    for pair in PAIRS:
        path = cache_dir / f"binance_{pair}_5m.csv"
        frame = pd.read_csv(path)
        last_ms = int(float(frame.timestamp.max()) * 1000)
        cursor = last_ms + 300_000
        additions = []
        while cursor < server_ms:
            response = requests.get(
                "https://api.binance.com/api/v3/klines",
                params={"symbol": pair.replace("-", ""), "interval": "5m",
                        "startTime": cursor, "limit": 1000}, timeout=20,
            )
            response.raise_for_status()
            rows = [item for item in response.json() if int(item[6]) < server_ms]
            if not rows:
                break
            additions.extend({
                "timestamp": int(item[0]) // 1000, "open": float(item[1]),
                "high": float(item[2]), "low": float(item[3]), "close": float(item[4]),
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


def load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"schema": "xgboost-long-risk-gate-state-v1", "pairs": {}}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "xgboost-long-risk-gate-state-v1":
        raise ValueError("unsupported XGBoost state schema")
    return payload


def _gate_state(raw: Mapping[str, Any] | None) -> v5.GateState:
    value = dict(raw or {})
    return v5.GateState(
        active=bool(value.get("active", False)),
        since=int(value["since"]) if value.get("since") is not None else None,
        entry_count=int(value.get("entry_count", 0)),
        recovery_count=int(value.get("recovery_count", 0)),
        last_recovery=(int(value["last_recovery"]) if value.get("last_recovery") is not None else None),
    )


def _predict(channel: Mapping[str, Any], rows) -> np.ndarray:
    architecture = str(channel["architecture"])
    features = list(channel["features"])
    if architecture == "shared":
        return channel["models"]["ALL"].predict_proba(rows[features])[:, 1]
    result = np.empty(len(rows), dtype=float)
    for pair in PAIRS:
        mask = rows.pair == pair
        result[mask.to_numpy()] = channel["models"][pair].predict_proba(rows.loc[mask, features])[:, 1]
    return result


def persistent_entry_evidence(
    history: list[dict[str, float]], probability: float, entry: float, recovery: float,
    roc: float, sqz: float,
) -> tuple[bool, bool, bool]:
    """Return the locked v15 probability/ROC-SQZ long-entry evidence."""
    values = [*history[-8:], {"probability": probability, "roc": roc, "sqz": sqz}]
    probability_rising = False
    technical_worsening = False
    if len(values) >= 3:
        recent = values[-3:]
        minimum_rise = max(1e-4, 0.25 * max(entry - recovery, 0.0))
        probability_rising = bool(
            recent[2]["probability"] > recent[1]["probability"]
            and recent[1]["probability"] > recent[0]["probability"]
            and recent[2]["probability"] - recent[0]["probability"] >= minimum_rise
        )
    if len(values) >= 9:
        current, lag4, lag8 = values[-1], values[-5], values[-9]
        technical_worsening = bool(
            current["roc"] < lag4["roc"] <= lag8["roc"]
            and current["sqz"] < lag4["sqz"] <= lag8["sqz"]
            and current["roc"] < 0.0 and current["sqz"] < 0.0
        )
    return probability_rising or technical_worsening, probability_rising, technical_worsening


def produce_once(args: argparse.Namespace) -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    now_ts = int(now.timestamp())
    lock = json.loads(args.lock.read_text(encoding="utf-8"))
    model_path = Path(lock["model_path"])
    if not model_path.is_absolute():
        model_path = Path.cwd() / model_path
    actual_model_hash = sha256_file(model_path)
    if actual_model_hash != lock["model_sha256"]:
        raise ValueError("locked model hash mismatch")
    bundle = joblib.load(model_path)
    expected_feature_hash = hashlib.sha256(json.dumps(
        {name: value["features"] for name, value in bundle["channels"].items()},
        sort_keys=True, separators=(",", ":"),
    ).encode()).hexdigest()
    if expected_feature_hash != lock["feature_schema_sha256"]:
        raise ValueError("locked feature schema hash mismatch")

    ensure_live_cache(args.cache_dir, args.seed_cache_dir)
    if args.refresh_binance:
        refresh_binance_cache(args.cache_dir)
    candles, _ = load_candles(args.cache_dir)
    panel = base.build_multi_horizon_panel(candles)
    latest_signal = int(panel.signal_ts.max())
    source_healthy = now_ts - latest_signal <= args.source_max_age_seconds
    state_payload = load_state(args.state)
    if (
        state_payload.get("model_sha256") not in (None, actual_model_hash)
        or state_payload.get("feature_schema_sha256") not in (None, expected_feature_hash)
    ):
        state_payload = {"schema": "xgboost-long-risk-gate-state-v1", "pairs": {}}
    pair_state = state_payload.setdefault("pairs", {})
    channel_outputs: dict[str, dict[str, dict[str, Any]]] = {pair: {} for pair in PAIRS}
    for channel_name, channel in bundle["channels"].items():
        for pair in PAIRS:
            gate = v5.GateParameters(**channel["gates"][pair])
            saved = pair_state.setdefault(pair, {}).setdefault(channel_name, {})
            last_signal = saved.get("last_signal_ts")
            if last_signal is None:
                last_signal = latest_signal - 30 * 24 * 3600
            rows = panel[(panel.pair == pair) & (panel.signal_ts > int(last_signal))].sort_values("signal_ts").copy()
            state = _gate_state(saved.get("gate_state"))
            probability = float(saved.get("probability", 0.0))
            transition, reason = "hold", "heartbeat_without_new_closed_bar"
            probability_rising = bool(saved.get("probability_rising_3h", False))
            technical_worsening = bool(saved.get("roc_sqz_worsening_8h", False))
            history = list(saved.get("entry_evidence_history", []))[-8:]
            thresholds = channel["thresholds"][pair]
            if not rows.empty:
                rows["probability"] = _predict(channel, rows)
                for row in rows.itertuples(index=False):
                    probability = float(row.probability)
                    entry = float(thresholds["entry"])
                    recovery = float(thresholds["recovery"])
                    evidence, probability_rising, technical_worsening = persistent_entry_evidence(
                        history, probability, entry, recovery,
                        float(row.roc_48h_4h), float(row.sqzmom_pct_4h),
                    )
                    effective_probability = probability
                    if not state.active and not evidence:
                        effective_probability = min(
                            probability, float(np.nextafter(entry, -np.inf))
                        )
                    state, transition, reason = v5.step_gate(
                        effective_probability, entry, recovery,
                        int(row.signal_ts), state, gate,
                    )
                    if not state.active and probability >= entry and not evidence:
                        reason = "entry_blocked_no_persistent_probability_or_roc_sqz_deterioration"
                    history = [*history, {
                        "signal_ts": int(row.signal_ts), "probability": probability,
                        "roc": float(row.roc_48h_4h), "sqz": float(row.sqzmom_pct_4h),
                    }][-8:]
                    last_signal = int(row.signal_ts)
            saved.update({
                "last_signal_ts": int(last_signal), "probability": probability,
                "gate_state": asdict(state), "transition": transition, "reason": reason,
                "entry_evidence_history": history,
                "probability_rising_3h": probability_rising,
                "roc_sqz_worsening_8h": technical_worsening,
            })
            channel_outputs[pair][channel_name] = {
                "probability": probability,
                "entry_threshold": float(thresholds["entry"]),
                "recovery_threshold": float(thresholds["recovery"]),
                "risk_off_active": bool(state.active),
                "risk_off_since": (
                    datetime.fromtimestamp(state.since, timezone.utc).isoformat().replace("+00:00", "Z")
                    if state.since is not None else None
                ),
                "entry_count": int(state.entry_count),
                "recovery_count": int(state.recovery_count),
                "probability_rising_3h": probability_rising,
                "roc_sqz_worsening_8h": technical_worsening,
                "transition": transition, "reason": reason,
                "last_complete_1h": datetime.fromtimestamp(int(last_signal), timezone.utc).isoformat().replace("+00:00", "Z"),
            }
    pair_signals = {
        pair: combine_pair_channels(
            pair=pair, channels=channel_outputs[pair], signal_ts=latest_signal,
            model_version=MODEL_VERSION,
        ) for pair in PAIRS
    }
    last_4h = {
        pair: int(panel.loc[panel.pair == pair, "last_complete_4h_ts"].max()) for pair in PAIRS
    }
    lock_allowed = bool(lock.get("deployment_allowed", False))
    contract = build_contract(
        generated_at=now_ts, valid_until=now_ts + STALE_AFTER_SECONDS,
        model_version=MODEL_VERSION, model_sha256=actual_model_hash,
        feature_sha256=expected_feature_hash, data_sha256=combined_data_hash(args.cache_dir),
        source_healthy=source_healthy,
        deployment_allowed=False,
        pair_signals=pair_signals,
        last_complete_1h={pair: latest_signal for pair in PAIRS}, last_complete_4h=last_4h,
    )
    contract["shadow_mode"] = True
    contract["lock_deployment_allowed"] = lock_allowed
    state_payload["updated_at"] = now.isoformat()
    state_payload["model_sha256"] = actual_model_hash
    state_payload["feature_schema_sha256"] = expected_feature_hash
    atomic_json(args.state, state_payload)
    atomic_json(args.output, contract)
    return contract


def main() -> int:
    args = parse_args()
    while True:
        try:
            contract = produce_once(args)
            print(json.dumps({
                "generated_at": contract["generated_at"],
                "source_healthy": contract["source_healthy"],
                "deployment_allowed": contract["deployment_allowed"],
                "buy_enabled": {pair: value["buy_enabled"] for pair, value in contract["pairs"].items()},
            }, ensure_ascii=False), flush=True)
        except Exception as exc:
            print(json.dumps({"error": str(exc), "fail_closed": True}, ensure_ascii=False), flush=True)
            if not args.loop:
                raise
        if not args.loop:
            break
        time.sleep(max(1, int(args.poll_seconds)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

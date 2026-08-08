#!/usr/bin/env python3
"""Run frozen v21 inference inside Grid Live Guard and publish a BUY-only gate."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from build_xgboost_v21_shadow_signal import produce_once
from grid_xgboost_risk_gate import (
    MODEL_VERSION, STALE_AFTER_SECONDS, atomic_json, build_contract,
    combine_pair_channels,
)
from xgboost_long_risk_gate_v21 import MODEL_VERSION as V21_MODEL_VERSION, PAIRS


if MODEL_VERSION != V21_MODEL_VERSION:
    raise RuntimeError("live gate contract and frozen v21 model version differ")


def _ts(value: str) -> int:
    return int(datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp())


def convert_shadow_to_live(
    shadow: Mapping[str, Any], *, authorized: bool, observed_at: int,
) -> dict[str, Any]:
    """Promote only recommended BUY state; never add SELL or flatten authority."""
    if shadow.get("model_version") != MODEL_VERSION:
        raise ValueError("v21 shadow model version mismatch")
    pair_signals: dict[str, dict[str, Any]] = {}
    last_1h: dict[str, int] = {}
    last_4h: dict[str, int] = {}
    for pair in PAIRS:
        raw = dict(shadow["pairs"][pair]["long"])
        risk_off = bool(raw["risk_off_active"])
        if bool(raw["recommended_buy_enabled"]) == risk_off:
            raise ValueError(f"{pair} recommended BUY is inconsistent with Risk-Off")
        channel = {
            **raw,
            "probability": float(raw["probability"]),
            "entry_threshold": float(raw["entry_threshold"]),
            # v21 recovery is structural, not probability-based.  This field is
            # retained only for the stable live contract schema.
            "recovery_threshold": 0.0,
            "risk_off_active": risk_off,
            "buy_enabled": not risk_off,
            "recovery_mode": "v21_adaptive_structural_relief",
        }
        pair_signals[pair] = combine_pair_channels(
            pair=pair, channels={"long": channel}, signal_ts=observed_at,
            model_version=MODEL_VERSION,
        )
        last_1h[pair] = _ts(raw["last_complete_1h"])
        last_4h[pair] = _ts(raw["last_complete_4h"])
    live = build_contract(
        generated_at=observed_at,
        valid_until=observed_at + STALE_AFTER_SECONDS,
        model_version=MODEL_VERSION,
        model_sha256=str(shadow["model_sha256"]),
        feature_sha256=str(shadow["feature_schema_sha256"]),
        data_sha256=str(shadow["state_sha256"]),
        source_healthy=bool(shadow["source_healthy"]),
        deployment_allowed=authorized,
        pair_signals=pair_signals, last_complete_1h=last_1h, last_complete_4h=last_4h,
    )
    live.update({
        "shadow_mode": False,
        "runtime_action": "pause_ordinary_buy_only",
        "producer": "grid-live-guard",
        "known_window_no_go_acknowledged": True,
        "v21_live_authorized": bool(authorized),
        "candidate_lock_sha256": shadow["candidate_lock_sha256"],
        "strategy_schema_sha256": shadow["strategy_schema_sha256"],
        "training_data_sha256": shadow["training_data_sha256"],
        "state_sha256": shadow["state_sha256"],
    })
    return live


class V21LiveGateProducer:
    def __init__(self, *, package_dir: Path, cache_dir: Path, seed_cache_dir: Path,
                 state_dir: Path, authorized: bool, refresh_binance: bool = True):
        self.package_dir = package_dir
        self.cache_dir = cache_dir
        self.seed_cache_dir = seed_cache_dir
        self.state_dir = state_dir
        self.authorized = bool(authorized)
        self.refresh_binance = bool(refresh_binance)
        self.shadow_output = state_dir / "xgboost_risk_gate_v21_internal.json"
        self.v21_state = state_dir / "xgboost_risk_gate_v21_state.json"
        self.live_output = state_dir / "xgboost_risk_gate.json"

    def produce(self, observed_at: int | None = None) -> dict[str, Any]:
        observed = int(observed_at if observed_at is not None else datetime.now(timezone.utc).timestamp())
        args = argparse.Namespace(
            lock=self.package_dir / "shadow_lock.json",
            cache_dir=self.cache_dir,
            seed_cache_dir=self.seed_cache_dir,
            output=self.shadow_output,
            state=self.v21_state,
            refresh_binance=self.refresh_binance,
            loop=False, poll_seconds=60, observed_at=observed,
        )
        shadow = produce_once(args)
        live = convert_shadow_to_live(shadow, authorized=self.authorized, observed_at=observed)
        atomic_json(self.live_output, live)
        return live


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--seed-cache-dir", type=Path, required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--authorized", action="store_true")
    parser.add_argument("--no-refresh-binance", action="store_true")
    args = parser.parse_args()
    result = V21LiveGateProducer(
        package_dir=args.package_dir, cache_dir=args.cache_dir,
        seed_cache_dir=args.seed_cache_dir, state_dir=args.state_dir,
        authorized=args.authorized, refresh_binance=not args.no_refresh_binance,
    ).produce()
    print(json.dumps({"generated_at": result["generated_at"],
                      "authorized": result["v21_live_authorized"],
                      "buy_enabled": {pair: result["pairs"][pair]["buy_enabled"] for pair in PAIRS}}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Produce the sole ethbtc-forced-exit v22 contract inside Grid Live Guard."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

try:
    from build_xgboost_v22_shadow_signal import produce_once
    from ethbtc_forced_exit_contract import (
        EXECUTION_POLICY_VERSION, MODEL_VERSION, PACKAGE_ID, REQUIRED_PAIRS,
        STALE_AFTER_SECONDS, atomic_json, event_id, failed_contract, sha256_file, utc,
    )
except ModuleNotFoundError:  # Package import from the repository root/tests.
    from scripts.build_xgboost_v22_shadow_signal import produce_once
    from scripts.ethbtc_forced_exit_contract import (
        EXECUTION_POLICY_VERSION, MODEL_VERSION, PACKAGE_ID, REQUIRED_PAIRS,
        STALE_AFTER_SECONDS, atomic_json, event_id, failed_contract, sha256_file, utc,
    )


CONFIRMATION = "PROMOTE-ETHBTC-FORCED-EXIT"


def _resolve_package(package_dir: Path) -> tuple[Path, Path]:
    lock = package_dir / "shadow_package/shadow_lock.json"
    production = package_dir / "production_lock.json"
    if not lock.exists():
        lock = package_dir / "shadow_lock.json"
    if not lock.is_file() or not production.is_file():
        raise FileNotFoundError("v22 candidate lacks shadow_package or production_lock.json")
    return lock, production


def _authorization(path: Path, production: dict[str, Any], observed: int) -> tuple[bool, dict[str, Any], str | None]:
    if not path.is_file():
        return False, {}, None
    raw = path.read_bytes()
    receipt = json.loads(raw.decode("utf-8"))
    expected = {
        "schema": "ethbtc-forced-exit-authorization-v1",
        "package_id": PACKAGE_ID,
        "release_sha256": production["release_sha256"],
        "model_sha256": production["model_sha256"],
        "confirmation": CONFIRMATION,
    }
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise ValueError("v22 authorization receipt does not match candidate")
    activation = int(receipt["activate_at"])
    if activation % 60 or activation < int(receipt["approved_at"]):
        raise ValueError("v22 authorization activation boundary is invalid")
    if activation >= int(production["effective_end"]):
        raise ValueError("v22 authorization starts after signed coverage")
    if not receipt.get("observation_report_sha256") or not receipt.get("preflight_sha256"):
        raise ValueError("v22 authorization lacks observation/preflight evidence")
    return observed >= activation, receipt, sha256_file(path)


class V22LiveGateProducer:
    def __init__(self, *, package_dir: Path, cache_dir: Path, seed_cache_dir: Path,
                 state_dir: Path, authorization_path: Path, refresh_binance: bool = True):
        self.package_dir = package_dir
        self.cache_dir = cache_dir
        self.seed_cache_dir = seed_cache_dir
        self.state_dir = state_dir
        self.authorization_path = authorization_path
        self.refresh_binance = refresh_binance
        self.output = state_dir / "xgboost_risk_gate.json"
        self.shadow_output = state_dir / "xgboost_risk_gate_v22_internal.json"
        self.shadow_state = state_dir / "xgboost_risk_gate_v22_state.json"

    def produce(self, observed_at: int | None = None) -> dict[str, Any]:
        observed = int(observed_at if observed_at is not None else time.time())
        lock_path, production_path = _resolve_package(self.package_dir)
        production = json.loads(production_path.read_text(encoding="utf-8"))
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        metadata = {
            "release_sha256": production.get("release_sha256"),
            "model_sha256": lock.get("model_sha256"),
            "feature_schema_sha256": lock.get("feature_schema_sha256"),
            "strategy_schema_sha256": lock.get("strategy_schema_sha256"),
            "training_data_sha256": lock.get(
                "training_candle_sha256", lock.get("training_panel_sha256")
            ),
        }
        try:
            if production.get("package_id") != PACKAGE_ID:
                raise ValueError("candidate package id mismatch")
            if production.get("execution_policy_version") != EXECUTION_POLICY_VERSION:
                raise ValueError("candidate execution policy mismatch")
            if production.get("model_sha256") != lock.get("model_sha256"):
                raise ValueError("production/shadow model hash mismatch")
            if sha256_file(lock_path) != production.get("shadow_lock_sha256"):
                raise ValueError("candidate shadow lock hash mismatch")
            shadow = produce_once(SimpleNamespace(
                observed_at=observed, lock=lock_path, cache_dir=self.cache_dir,
                seed_cache_dir=self.seed_cache_dir, output=self.shadow_output,
                state=self.shadow_state, refresh_binance=self.refresh_binance,
            ))
            authorized, receipt, receipt_hash = _authorization(
                self.authorization_path, production, observed,
            )
            source_healthy = bool(shadow.get("source_healthy"))
            pairs = {}
            for pair in REQUIRED_PAIRS:
                source = shadow["pairs"][pair]
                long = source["long"]
                week_start = int(long["week_test_start"])
                week_end = int(long["week_test_end"])
                # The shadow producer stores the authoritative latest signal in
                # its persistent state; use it rather than parsing display time.
                shadow_state = json.loads(self.shadow_state.read_text(encoding="utf-8"))
                signal_ts = int(shadow_state["pairs"][pair]["gate_state"]["last_signal_ts"])
                risk_off = bool(source["risk_off_active"])
                transition = str(long.get("transition", "hold"))
                pairs[pair] = {
                    "pair": pair, "source_pair": pair, "signal_ts": signal_ts,
                    "model_week": int(long["fold"]), "week_start": week_start,
                    "week_end": week_end, "week_model_sha256": long["week_model_sha256"],
                    "probability": float(long["probability"]),
                    "entry_threshold": float(long["entry_threshold"]),
                    "risk_off_active": risk_off,
                    "recommended_buy_enabled": not risk_off,
                    "buy_enabled": bool(authorized and not risk_off),
                    "force_exit": bool(authorized and risk_off),
                    "transition": transition, "reason": str(source.get("reason", long.get("reason", "v22"))),
                    "event_id": event_id(production["release_sha256"], pair, signal_ts, transition),
                }
            valid_until = min(observed + STALE_AFTER_SECONDS, int(production["effective_end"]))
            contract = {
                "schema": "ethbtc-forced-exit-live-contract-v1", "package_id": PACKAGE_ID,
                "execution_policy_version": EXECUTION_POLICY_VERSION,
                "model_version": MODEL_VERSION, "generated_at": utc(observed),
                "valid_until": utc(valid_until), "stale_after_seconds": STALE_AFTER_SECONDS,
                **metadata, "source_healthy": source_healthy,
                "execution_authorized": bool(authorized), "observation_mode": not authorized,
                "activation_at": receipt.get("activate_at"),
                "approval_receipt_sha256": receipt_hash,
                "deployment_allowed": bool(authorized), "promotion_authorized": bool(authorized),
                "market_sell_action": True, "previous_model_fallback_allowed": False,
                "runtime_action": "execute" if authorized else "observe_only",
                "reason": "v22_live_healthy" if authorized else "v22_observation_healthy",
                "pairs": pairs,
            }
        except Exception as exc:
            contract = failed_contract(
                generated_at=observed, reason=f"fail_closed:{exc}", metadata=metadata,
            )
        atomic_json(self.output, contract)
        return contract


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--seed-cache-dir", type=Path, required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--observed-at", type=int)
    args = parser.parse_args()
    producer = V22LiveGateProducer(
        package_dir=args.package_dir, cache_dir=args.cache_dir,
        seed_cache_dir=args.seed_cache_dir, state_dir=args.state_dir,
        authorization_path=args.authorization,
    )
    print(json.dumps(producer.produce(args.observed_at), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

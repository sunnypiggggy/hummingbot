#!/usr/bin/env python3
"""Validate the frozen v21 shadow package without authorizing deployment."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tempfile
import tracemalloc
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import joblib
import numpy as np
import pandas as pd

from build_xgboost_v21_shadow_signal import produce_once
from grid_xgboost_shadow_gate_v21 import atomic_json, load_shadow_contract
from retrain_xgboost_long_risk_gate_250d_v19 import sha256_file
from xgboost_long_risk_gate_v21 import (
    FEATURES, GATES, MODEL_BUNDLE_SCHEMA, MODEL_VERSION, PAIRS, GateState,
    advance_gate, feature_schema_sha256, strategy_schema_sha256, validate_strategy_bundle,
)


DEFAULT_PACKAGE = Path("results/backtests/xgboost_grid_long_risk_gate_v21_250d/shadow_package")
DEFAULT_SEED = Path("results/backtests/eth_xgboost_long_risk_gate_v15_250d/extended_candles")
PANEL = Path("results/backtests/xgboost_grid_long_risk_gate_v19_250d/feature_panel.csv.gz")


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, default=DEFAULT_PACKAGE)
    parser.add_argument("--seed-cache-dir", type=Path, default=DEFAULT_SEED)
    parser.add_argument("--container-evidence", type=Path)
    return parser.parse_args()


def _predict_worker(payload: tuple[str, str, list[str], list[list[float]]]) -> list[float]:
    model_path, pair, features, values = payload
    bundle = joblib.load(model_path)
    frame = pd.DataFrame(values, columns=features)
    return bundle["pairs"][pair]["model"].predict_proba(frame)[:, 1].tolist()


def verify_container_evidence(path: Path | None) -> tuple[bool, str]:
    if path is None or not path.exists():
        return False, "Docker engine validation evidence is absent"
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema") == "xgboost-v21-shadow-container-validation-failure-v1":
        return False, str(value.get("reason", "container validation failed"))
    required = ("image_built", "one_shot_passed", "heartbeat_passed", "restart_passed",
                "read_only_model_passed", "atomic_replace_passed", "healthcheck_passed")
    if value.get("schema") != "xgboost-v21-shadow-container-evidence-v1":
        return False, "container evidence schema mismatch"
    if not all(value.get(item) is True for item in required):
        return False, "container evidence is incomplete"
    return True, "container validation evidence passed"


def main() -> int:
    args = arguments()
    package = args.package.resolve()
    lock_path = package / "shadow_lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    checks: dict[str, Any] = {}

    checks["immutable_safety_lock"] = (
        lock.get("historical_verdict") == "NO-GO"
        and lock.get("deployment_allowed") is False
        and lock.get("promotion_authorized") is False
        and lock.get("shadow_mode") is True
        and lock.get("short_spike_enabled") is False
        and lock.get("market_sell_action") is False
        and lock.get("mechanism1_fallback_allowed") is False
        and lock.get("forward_shadow_weeks_required") == 8
    )
    checks["candidate_lock_hash"] = sha256_file(Path(lock["candidate_lock_path"])) == lock["candidate_lock_sha256"]
    checks["training_panel_hash"] = sha256_file(PANEL) == lock["training_panel_sha256"]
    checks["feature_schema_hash"] = feature_schema_sha256() == lock["feature_schema_sha256"]
    checks["strategy_schema_hash"] = strategy_schema_sha256() == lock["strategy_schema_sha256"]
    model_path = Path(lock["model_path"])
    checks["model_hash"] = sha256_file(model_path) == lock["model_sha256"]
    bundle = joblib.load(model_path)
    checks["bundle_schema"] = bundle.get("schema") == MODEL_BUNDLE_SCHEMA and bundle.get("model_version") == MODEL_VERSION
    try:
        validate_strategy_bundle(bundle)
        checks["embedded_complete_strategy"] = True
    except ValueError:
        checks["embedded_complete_strategy"] = False
    checks["pair_model_contract"] = all(
        bundle["pairs"][pair]["features"] == list(FEATURES[pair])
        and bundle["pairs"][pair]["gate"] == GATES[pair].__dict__
        and bundle["pairs"][pair]["entry_threshold"] == lock["pairs"][pair]["entry_threshold"]
        and bundle["pairs"][pair]["best_tree_count"] == lock["pairs"][pair]["best_tree_count"]
        and bundle["pairs"][pair]["model"].get_params()["n_estimators"] == lock["pairs"][pair]["best_tree_count"]
        for pair in PAIRS
    )

    audit = pd.read_csv(package / "final_training_audit.csv")
    checks["label_maturity_and_calibration_isolation"] = bool(
        len(audit) == 2
        and audit.label_maturity_hours.eq(96).all()
        and audit.calibration_excluded_from_final_fit.astype(bool).all()
        and (audit.last_label_ready_ts <= audit.cutoff).all()
        and (audit.development_last_ts < audit.calibration_first_ts).all()
        and (audit.early_train_rows + audit.early_stop_rows == audit.development_rows).all()
        and (audit.development_rows + audit.calibration_rows == audit.mature_rows).all()
    )
    checks["serialization_tolerance"] = bool(
        lock["serialization_check"]["passed"]
        and lock["serialization_check"]["maximum_probability_absolute_error"] <= 1e-12
    )
    test_evidence_path = package / "test_evidence.json"
    test_evidence = json.loads(test_evidence_path.read_text(encoding="utf-8"))
    checks["pytest_and_runtime_regressions"] = bool(
        test_evidence.get("schema") == "xgboost-v21-shadow-test-evidence-v1"
        and int(test_evidence.get("passed", 0)) >= 85
        and int(test_evidence.get("failed", -1)) == 0
    )
    report_path = package.parent / "xgboost-grid-long-risk-gate-v21-250d_plotly.html"
    report_text = report_path.read_text(encoding="utf-8")
    checks["plotly_long_only_independent_controls"] = bool(
        "BTC Risk-off" in report_text and "ETH Risk-off" in report_text
        and "Short Risk-off" not in report_text and "short_spike" not in report_text
        and "frozen application bundle exact replay" in report_text
    )
    checks["application_replay_artifact_hashes"] = bool(
        sha256_file(Path(lock["application_plotly"])) == lock["application_plotly_sha256"]
        and sha256_file(Path(lock["application_bundle_summary"])) == lock["application_bundle_summary_sha256"]
        and sha256_file(package.parent / "application_bundle" / "risk_states.csv.gz")
        == lock["application_risk_states_sha256"]
        and sha256_file(package.parent / "application_bundle" / "risk_intervals.csv")
        == lock["application_risk_intervals_sha256"]
    )

    # Single- and multi-process probability paths must be bit-identical.
    panel = pd.read_csv(PANEL).replace([np.inf, -np.inf], np.nan)
    multiprocess_errors = {}
    for pair in PAIRS:
        frame = panel[panel.pair.eq(pair)].dropna(subset=list(FEATURES[pair])).tail(256)
        single = bundle["pairs"][pair]["model"].predict_proba(frame[list(FEATURES[pair])])[:, 1]
        chunks = np.array_split(frame[list(FEATURES[pair])], 2)
        payloads = [(str(model_path), pair, list(FEATURES[pair]), chunk.values.tolist()) for chunk in chunks]
        with ProcessPoolExecutor(max_workers=2) as pool:
            multi = np.concatenate([np.asarray(value) for value in pool.map(_predict_worker, payloads)])
        multiprocess_errors[pair] = float(np.max(np.abs(single - multi)))
    checks["single_multi_process_probability_parity"] = max(multiprocess_errors.values()) <= 1e-12

    # Run the real producer twice from the same persisted state. This verifies
    # cold bootstrap, atomic state/output writes and restart idempotence.
    validation_dir = package / "validation"
    validation_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="v21-shadow-") as temporary:
        work = Path(temporary)
        producer_args = SimpleNamespace(
            lock=lock_path, cache_dir=work / "candles", seed_cache_dir=args.seed_cache_dir,
            output=work / "xgboost_risk_gate_v21_shadow.json", state=work / "state.json",
            refresh_binance=False, loop=False, poll_seconds=60,
            observed_at=int(lock["training_cutoff_ts"]) - 3600 + 300,
        )
        first = produce_once(producer_args)
        first_state_hash = sha256_file(producer_args.state)
        second = produce_once(producer_args)
        second_state_hash = sha256_file(producer_args.state)
        load_shadow_contract(
            producer_args.output,
            now=pd.Timestamp(producer_args.observed_at, unit="s", tz="UTC").to_pydatetime(),
        )
        checks["one_shot_signal"] = first.get("source_healthy") is True
        checks["restart_state_idempotence"] = first_state_hash == second_state_hash
        checks["public_buy_always_disabled"] = all(
            second["pairs"][pair]["long"]["buy_enabled"] is False for pair in PAIRS
        )
        application_states = pd.read_csv(
            package.parent / "application_bundle" / "risk_states.csv.gz"
        )
        parity = []
        for pair in PAIRS:
            live = second["pairs"][pair]["long"]
            expected = application_states[
                application_states.pair.eq(pair)
                & application_states.signal_ts.eq(int(pd.Timestamp(live["last_complete_1h"]).timestamp()))
            ].iloc[-1]
            parity.append(
                abs(float(live["probability"]) - float(expected.probability)) <= 1e-12
                and abs(float(live["entry_threshold"]) - float(expected.entry_threshold)) <= 1e-15
                and bool(live["risk_off_active"]) == bool(expected.risk_off_active)
                and bool(live["recommended_buy_enabled"]) == bool(expected.recommended_buy_enabled)
                and str(live["transition"]) == str(expected.transition)
                and str(live["event_id"]) == str(expected.event_id)
            )
        checks["producer_plotly_state_parity"] = all(parity)
        (validation_dir / "xgboost_risk_gate_v21_shadow.sample.json").write_text(
            json.dumps(second, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    # Accelerated 48h sequential state soak using the shared inference/state path.
    tracemalloc.start()
    states = {pair: GateState() for pair in PAIRS}
    seen_events: set[str] = set()
    duplicate_transitions = 0
    soak_rows = panel.dropna(subset=["roc_48h_4h", "sqzmom_pct_4h", "di_spread",
                                    "ema20_slope_atr_12h", "below_ema20_ratio_72h"])
    soak_rows = soak_rows[soak_rows.signal_ts > int(lock["training_cutoff_ts"]) - 48 * 3600]
    for pair in PAIRS:
        rows = soak_rows[soak_rows.pair.eq(pair)].sort_values("signal_ts").tail(48)
        probabilities = bundle["pairs"][pair]["model"].predict_proba(rows[list(FEATURES[pair])])[:, 1]
        for row, probability in zip(rows.itertuples(index=False), probabilities):
            states[pair], snap = advance_gate(
                pair=pair, probability=float(probability),
                entry_threshold=float(lock["pairs"][pair]["entry_threshold"]),
                signal_ts=int(row.signal_ts), last_complete_4h_ts=int(row.last_complete_4h_ts),
                structure=(row.roc_48h_4h, row.sqzmom_pct_4h, row.di_spread,
                           row.ema20_slope_atr_12h, row.below_ema20_ratio_72h), state=states[pair],
            )
            if snap["transition"] in {"enter", "recover"}:
                duplicate_transitions += int(snap["event_id"] in seen_events)
                seen_events.add(snap["event_id"])
    _, peak_bytes = tracemalloc.get_traced_memory(); tracemalloc.stop()
    checks["accelerated_48h_soak"] = duplicate_transitions == 0 and peak_bytes < 64 * 1024 * 1024

    compose = subprocess.run(
        ["docker", "compose", "config", "--quiet"], text=True, capture_output=True, check=False
    )
    checks["compose_configuration"] = compose.returncode == 0
    evidence_path = args.container_evidence
    if evidence_path is None:
        failure_evidence = package / "container_validation_failure.json"
        evidence_path = failure_evidence if failure_evidence.exists() else None
    container_ok, container_reason = verify_container_evidence(evidence_path)
    checks["container_runtime_validation"] = container_ok

    required_without_container = all(value is True for key, value in checks.items()
                                     if key != "container_runtime_validation")
    status = "SHADOW_READY" if required_without_container and container_ok else (
        "PACKAGE_VALIDATED_DOCKER_PENDING" if required_without_container else "VALIDATION_FAILED"
    )
    readiness = {
        "schema": "xgboost-v21-shadow-readiness-v1",
        "model_version": MODEL_VERSION,
        "status": status,
        "historical_verdict": "NO-GO",
        "deployment_allowed": False,
        "promotion_authorized": False,
        "shadow_mode": True,
        "forward_shadow_weeks_required": 8,
        "checks": checks,
        "multiprocess_max_probability_error": multiprocess_errors,
        "accelerated_soak_peak_bytes": peak_bytes,
        "accelerated_soak_duplicate_transition_events": duplicate_transitions,
        "container_validation": container_reason,
        "note": "SHADOW_READY is packaging readiness only; it never authorizes Grid or starts the eight-week clock.",
    }
    atomic_json(package / "shadow_readiness.json", readiness)
    manifest = {}
    for path in sorted(item for item in package.rglob("*") if item.is_file()
                       and item.name != "artifact_manifest.json"):
        manifest[path.relative_to(package).as_posix()] = {
            "sha256": sha256_file(path), "bytes": path.stat().st_size,
        }
    atomic_json(package / "artifact_manifest.json", {
        "schema": "xgboost-v21-shadow-artifact-manifest-v1",
        "model_version": MODEL_VERSION,
        "deployment_allowed": False,
        "promotion_authorized": False,
        "artifacts": manifest,
    })
    print(json.dumps(readiness, ensure_ascii=False, indent=2))
    return 0 if status != "VALIDATION_FAILED" else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Stage an immutable ethbtc-forced-exit production candidate by content hash."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

import joblib

from ethbtc_forced_exit_contract import EXECUTION_POLICY_VERSION, MODEL_VERSION, PACKAGE_ID, sha256_file
from xgboost_long_risk_gate_v22 import validate_weekly_bundle

try:
    from telegram_notifications import append_event, build_event
except ModuleNotFoundError:
    from live_guard.telegram_notifications import append_event, build_event


def canonical_hash(value: object) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shadow-package", type=Path, required=True)
    parser.add_argument("--lineage-package", type=Path,
                        default=Path("release_packages/ethbtc-forced-exit"))
    parser.add_argument("--release-root", type=Path,
                        default=Path("release_packages/ethbtc-forced-exit/releases"))
    parser.add_argument("--notification-events", type=Path, default=(
        Path(os.environ["TELEGRAM_NOTIFICATION_EVENTS_PATH"])
        if os.environ.get("TELEGRAM_NOTIFICATION_EVENTS_PATH") else None
    ))
    args = parser.parse_args()
    source_lock_path = args.shadow_package / "shadow_lock.json"
    source_lock = json.loads(source_lock_path.read_text(encoding="utf-8"))
    model = Path(source_lock["model_path"])
    if not model.exists():
        model = args.shadow_package / "models" / model.name
    if sha256_file(model) != source_lock["model_sha256"]:
        raise RuntimeError("staged v22 model hash mismatch")
    bundle = joblib.load(model)
    validate_weekly_bundle(bundle)
    if source_lock.get("model_version") != MODEL_VERSION:
        raise RuntimeError("staged package is not v22")
    weeks = {
        pair: [(int(item["test_start"]), int(item["test_end"]), int(item["fold"]))
               for item in bundle["pairs"][pair]["weeks"]]
        for pair in ("BTC-FDUSD", "ETH-FDUSD")
    }
    if weeks["BTC-FDUSD"] != weeks["ETH-FDUSD"]:
        raise RuntimeError("BTC/ETH signed week manifests differ")
    if weeks["BTC-FDUSD"][-1][1] != int(source_lock["effective_end"]):
        raise RuntimeError("lock effective_end does not match the signed model")
    weekly_report = {}
    for pair in ("BTC-FDUSD", "ETH-FDUSD"):
        signed = bundle["pairs"][pair]["weeks"]
        latest = signed[-1]
        weekly_report[pair] = {
            "weeks": len(signed),
            "signed_coverage": [int(signed[0]["test_start"]), int(latest["test_end"])],
            "latest_fold": int(latest["fold"]),
            "training_label_ready_end": int(latest["last_label_ready_ts"]),
            "development_end": int(latest["development_last_ts"]),
            "calibration_start": int(latest["calibration_first_ts"]),
            "calibration_rows": int(latest["calibration_rows"]),
            "oos_start": int(latest["test_start"]),
            "oos_end": int(latest["test_end"]),
            "entry_threshold": float(latest["entry_threshold"]),
            "entry_quantile": float(latest["entry_quantile"]),
            "best_tree_count": int(latest["best_tree_count"]),
            "fold_model_sha256": str(latest["model_sha256"]),
        }
    execution_policy = json.loads(
        (args.lineage_package / "evidence/execution_policy.json").read_text(encoding="utf-8")
    )
    if execution_policy.get("package_id") != PACKAGE_ID:
        raise RuntimeError("lineage execution policy package mismatch")
    documentation = args.lineage_package / "documentation"
    if not documentation.is_dir() or not (documentation / "README.md").is_file():
        raise RuntimeError("release lineage documentation is missing")
    documentation_hashes = {
        path.relative_to(documentation).as_posix(): sha256_file(path)
        for path in sorted(documentation.rglob("*")) if path.is_file()
    }
    documentation_sha256 = canonical_hash(documentation_hashes)
    identity = {
        "package_id": PACKAGE_ID,
        "execution_policy_version": EXECUTION_POLICY_VERSION,
        "model_sha256": source_lock["model_sha256"],
        "feature_schema_sha256": source_lock["feature_schema_sha256"],
        "strategy_schema_sha256": source_lock["strategy_schema_sha256"],
        "training_data_sha256": source_lock.get(
            "training_candle_sha256", source_lock["training_panel_sha256"]
        ),
        "effective_start": int(source_lock["effective_start"]),
        "effective_end": int(source_lock["effective_end"]),
        "execution_policy_sha256": execution_policy["execution_policy_sha256"],
        "lineage_manifest_sha256": sha256_file(args.lineage_package / "MANIFEST.sha256"),
        "documentation_sha256": documentation_sha256,
    }
    release_sha = canonical_hash(identity)
    target = args.release_root / release_sha
    if target.exists():
        raise FileExistsError(f"release already exists: {target}")
    staging = args.release_root / f".{release_sha}.staging"
    if staging.exists():
        raise FileExistsError(staging)
    staging.mkdir(parents=True)
    try:
        shadow = staging / "shadow_package"
        shutil.copytree(args.shadow_package, shadow)
        staged_lock_path = shadow / "shadow_lock.json"
        staged_lock = json.loads(staged_lock_path.read_text(encoding="utf-8"))
        staged_lock["model_path"] = "models/xgboost_long_risk_gate_v22_weekly.joblib"
        atomic_json(staged_lock_path, staged_lock)
        shutil.copy2(args.lineage_package / "evidence/execution_policy.json",
                     staging / "execution_policy.json")
        shutil.copytree(documentation, staging / "documentation")
        production = {
            "schema": "ethbtc-forced-exit-production-lock-v1",
            **identity, "release_sha256": release_sha,
            "shadow_lock_sha256": sha256_file(staged_lock_path),
            "observation_required_seconds": 86400,
            "deployment_allowed": False, "promotion_authorized": False,
            "previous_model_fallback_allowed": False,
        }
        atomic_json(staging / "production_lock.json", production)
        release = {
            "schema": "ethbtc-forced-exit-candidate-v1",
            "package_id": PACKAGE_ID, "release_sha256": release_sha,
            "release_stage": "observation_candidate", "offline_only": False,
            "deployment_allowed": False, "promotion_authorized": False,
            "effective_start": identity["effective_start"],
            "effective_end": identity["effective_end"],
            "source_offline_verdict": "NO-GO",
            "promotion_policy": "current_week_integrity_plus_24h_observation_plus_local_cli_approval",
        }
        atomic_json(staging / "release.json", release)
        files = sorted(path for path in staging.rglob("*") if path.is_file())
        manifest = "\n".join(
            f"{sha256_file(path)}  {path.relative_to(staging).as_posix()}" for path in files
        ) + "\n"
        (staging / "MANIFEST.sha256").write_text(manifest, encoding="utf-8")
        target.parent.mkdir(parents=True, exist_ok=True)
        os.replace(staging, target)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    if args.notification_events is not None:
        append_event(args.notification_events, build_event(
            source="stage-ethbtc-forced-exit-release", strategy="grid+dca",
            bot="grid-live-fdusd-400,dca-live-btcusdt-200,dca-live-ethusdt-200",
            pair="BTC-FDUSD,ETH-FDUSD,BTC-USDT,ETH-USDT",
            mechanism="parameter_update", transition="PARAMETER_CANDIDATE",
            reason="v22 周模型生产候选已封包", severity="info",
            action="render_analysis_then_observe_and_approve",
            release_sha256=release_sha, model_sha256=source_lock["model_sha256"],
            correlation_id=release_sha,
            details={"report_request": "v22_png_windows", "release_path": str(target),
                     "effective_start": identity["effective_start"],
                     "effective_end": identity["effective_end"],
                     "weekly_report": weekly_report,
                     "feature_schema_sha256": identity["feature_schema_sha256"],
                     "strategy_schema_sha256": identity["strategy_schema_sha256"],
                     "training_data_sha256": identity["training_data_sha256"],
                     "deployment_allowed": False},
        ))
    print(json.dumps({"release": str(target), "release_sha256": release_sha,
                      "deployment_allowed": False}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

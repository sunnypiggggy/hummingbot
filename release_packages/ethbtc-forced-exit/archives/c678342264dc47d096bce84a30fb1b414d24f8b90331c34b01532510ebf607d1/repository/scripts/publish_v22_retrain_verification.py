#!/usr/bin/env python3
"""Publish a hash-verified v22 reproducibility retrain to the Telegram event queue."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from live_guard.telegram_notifications import append_event, build_event


MODEL_NAME = "xgboost_long_risk_gate_v22_weekly.joblib"


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_verified_event(
    release_root: Path,
    retrain_dir: Path,
    *,
    attachment_root: str = "/workspace/releases",
) -> dict:
    current = release_root / "current"
    production = read_json(current / "production_lock.json")
    active_lock = read_json(current / "shadow_package/shadow_lock.json")
    retrain_lock = read_json(retrain_dir / "shadow_lock.json")
    report = read_json(
        release_root / "audits" / f"retrain-verification-{production['release_sha256']}"
        / "retrain_report.json"
    )
    active_model = current / "shadow_package/models" / MODEL_NAME
    retrain_model = retrain_dir / "models" / MODEL_NAME
    active_hash = sha256_file(active_model)
    retrain_hash = sha256_file(retrain_model)
    expected = str(production["model_sha256"])
    checks = {
        "release_matches_report": report.get("release_sha256") == production.get("release_sha256"),
        "active_lock_matches_production": active_lock.get("model_sha256") == expected,
        "retrain_lock_matches_production": retrain_lock.get("model_sha256") == expected,
        "active_file_matches_production": active_hash == expected,
        "retrain_file_matches_production": retrain_hash == expected,
        "byte_for_byte_match": active_hash == retrain_hash and report.get("byte_for_byte_match") is True,
        "same_effective_end": retrain_lock.get("effective_end") == production.get("effective_end"),
        "retrain_stays_unapproved": retrain_lock.get("deployment_allowed") is False,
        "no_production_switch": report.get("production_switch_performed") is False,
    }
    if not all(checks.values()):
        raise RuntimeError(f"retrain verification failed: {checks}")
    release_sha = str(production["release_sha256"])
    reason = (
        "真实 Binance 行情独立重训完成：第38周 BTC/ETH 模型与当前生产模型逐字节一致；"
        "当前交易继续使用原发布，未切换权限。下一周候选将在北京时间 "
        "2026-08-16 10:00 按连续周规则训练。频道只发送回测PNG，不发送模型文件。"
    )
    return build_event(
        source="v22-retrain-verification",
        strategy="grid+dca",
        bot="grid-live-fdusd-400,dca-live-btcusdt-200,dca-live-ethusdt-200",
        pair="BTC-FDUSD,ETH-FDUSD,BTC-USDT,ETH-USDT",
        mechanism="parameter_update",
        transition="PARAMETER_RETAINED",
        reason=reason,
        severity="info",
        action="retain_active_release_after_reproducible_retrain",
        release_sha256=release_sha,
        model_sha256=expected,
        correlation_id=f"real-retrain-verified:{release_sha}:{expected}",
        details={
            "report_request": "v22_png_windows",
            "model_attachment_included": False,
            "training_mode": report["mode"],
            "fold": report["fold"],
            "training_cutoff": report["training_cutoff"],
            "effective_start": report["effective_start"],
            "effective_end": report["effective_end"],
            "feature_schema_sha256": report["feature_schema_sha256"],
            "strategy_schema_sha256": report["strategy_schema_sha256"],
            "training_data_sha256": report["training_data_sha256"],
            "pairs": report["pairs"],
            "checks": checks,
            "production_switch_performed": False,
            "next_scheduled_candidate_training_bjt": report[
                "next_scheduled_candidate_training_bjt"
            ],
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-root", type=Path, required=True)
    parser.add_argument("--retrain-dir", type=Path, required=True)
    parser.add_argument("--notification-path", type=Path, required=True)
    parser.add_argument("--attachment-root", default="/workspace/releases")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    event = build_verified_event(
        args.release_root,
        args.retrain_dir,
        attachment_root=args.attachment_root,
    )
    if not args.dry_run:
        append_event(args.notification_path, event)
    print(json.dumps(event, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

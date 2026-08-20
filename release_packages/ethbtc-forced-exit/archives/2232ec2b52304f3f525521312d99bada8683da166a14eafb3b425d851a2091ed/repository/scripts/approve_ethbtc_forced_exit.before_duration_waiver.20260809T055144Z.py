#!/usr/bin/env python3
"""OCI-local approval/revocation CLI for an observed ethbtc-forced-exit release."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from ethbtc_forced_exit_contract import PACKAGE_ID, atomic_json, sha256_file

try:
    from telegram_notifications import append_event, build_event
except ModuleNotFoundError:
    from live_guard.telegram_notifications import append_event, build_event


# Kept literal here so the OCI-local approval CLI has no ML/joblib dependency.
CONFIRMATION = "PROMOTE-ETHBTC-FORCED-EXIT"


def checked_evidence(path: Path, schema: str, release_sha: str) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != schema or payload.get("release_sha256") != release_sha:
        raise ValueError(f"invalid {schema} evidence")
    if payload.get("passed") is not True:
        raise ValueError(f"{schema} did not pass")
    return payload


def approve(args: argparse.Namespace) -> int:
    production_path = args.package / "production_lock.json"
    production = json.loads(production_path.read_text(encoding="utf-8"))
    release_sha = str(production["release_sha256"])
    now = int(time.time())
    if production.get("package_id") != PACKAGE_ID or args.release_sha256 != release_sha:
        raise ValueError("approval release hash mismatch")
    if args.confirm != CONFIRMATION:
        raise ValueError(f"explicit --confirm {CONFIRMATION} is required")
    observation = checked_evidence(
        args.observation_report, "ethbtc-forced-exit-observation-v1", release_sha,
    )
    if int(observation["ended_at"]) - int(observation["started_at"]) < 86400:
        raise ValueError("v22 observation is shorter than 24 hours")
    checked_evidence(args.preflight, "ethbtc-forced-exit-preflight-v1", release_sha)
    if now >= int(production["effective_end"]):
        raise ValueError("signed v22 week is already expired")
    activate_at = ((now + 119) // 60) * 60
    if activate_at >= int(production["effective_end"]):
        raise ValueError("signed week expires before activation")
    receipt = {
        "schema": "ethbtc-forced-exit-authorization-v1",
        "package_id": PACKAGE_ID, "release_sha256": release_sha,
        "model_sha256": production["model_sha256"],
        "operator": args.operator, "confirmation": CONFIRMATION,
        "approved_at": now, "activate_at": activate_at,
        "effective_end": int(production["effective_end"]),
        "observation_report_sha256": sha256_file(args.observation_report),
        "preflight_sha256": sha256_file(args.preflight),
        "auto_reentry_authorized": True, "consumed": False,
    }
    atomic_json(args.authorization, receipt)
    try:
        os.chmod(args.authorization, 0o600)
    except OSError:
        pass
    if args.notification_events is not None:
        append_event(args.notification_events, build_event(
            source="approve-ethbtc-forced-exit", strategy="grid+dca",
            bot="grid-live-fdusd-400,dca-live-btcusdt-200,dca-live-ethusdt-200",
            pair="BTC-FDUSD,ETH-FDUSD,BTC-USDT,ETH-USDT",
            mechanism="parameter_update", transition="PARAMETER_ACTIVATED",
            reason=f"v22 已批准，将在 {activate_at} 激活", severity="info",
            action="atomic_activation_at_minute_boundary",
            release_sha256=release_sha, model_sha256=production["model_sha256"],
            correlation_id=release_sha,
            details={"activate_at": activate_at, "effective_end": production["effective_end"],
                     "approval_receipt_sha256": sha256_file(args.authorization)},
        ))
    print(json.dumps(receipt, indent=2))
    return 0


def revoke(args: argparse.Namespace) -> int:
    if args.confirm != "REVOKE-ETHBTC-FORCED-EXIT":
        raise ValueError("explicit revocation confirmation is required")
    args.authorization.unlink(missing_ok=True)
    revoked_path = args.authorization.with_name("ethbtc_forced_exit_revoked.json")
    atomic_json(revoked_path, {
        "schema": "ethbtc-forced-exit-revocation-v1", "revoked_at": int(time.time()),
        "operator": args.operator, "reason": args.reason,
        "execution_allowed": False, "automatic_fallback": False,
    })
    if args.notification_events is not None:
        append_event(args.notification_events, build_event(
            source="approve-ethbtc-forced-exit", strategy="grid+dca",
            bot="grid-live-fdusd-400,dca-live-btcusdt-200,dca-live-ethusdt-200",
            pair="BTC-FDUSD,ETH-FDUSD,BTC-USDT,ETH-USDT",
            mechanism="infrastructure_integrity_breaker", transition="LATCHED",
            reason=f"v22 实盘授权已撤销：{args.reason}", severity="critical",
            action="fail_closed_cancel_and_exit_no_fallback",
            requires_manual_action=True,
            correlation_id=f"revocation:{sha256_file(revoked_path)}",
            details={"revocation_sha256": sha256_file(revoked_path)},
        ))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    approved = sub.add_parser("approve")
    approved.add_argument("--package", type=Path, required=True)
    approved.add_argument("--release-sha256", required=True)
    approved.add_argument("--observation-report", type=Path, required=True)
    approved.add_argument("--preflight", type=Path, required=True)
    approved.add_argument("--authorization", type=Path, required=True)
    approved.add_argument("--operator", required=True)
    approved.add_argument("--confirm", required=True)
    approved.add_argument("--notification-events", type=Path, default=(
        Path(os.environ["TELEGRAM_NOTIFICATION_EVENTS_PATH"])
        if os.environ.get("TELEGRAM_NOTIFICATION_EVENTS_PATH") else None
    ))
    approved.set_defaults(handler=approve)
    revoked = sub.add_parser("revoke")
    revoked.add_argument("--authorization", type=Path, required=True)
    revoked.add_argument("--operator", required=True)
    revoked.add_argument("--reason", required=True)
    revoked.add_argument("--confirm", required=True)
    revoked.add_argument("--notification-events", type=Path, default=(
        Path(os.environ["TELEGRAM_NOTIFICATION_EVENTS_PATH"])
        if os.environ.get("TELEGRAM_NOTIFICATION_EVENTS_PATH") else None
    ))
    revoked.set_defaults(handler=revoke)
    args = parser.parse_args()
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())

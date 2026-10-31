"""Offline, event-bound reset of the Grid Guard latch; never starts trading.

The operator must stop the sole Guard writer and provide a fresh read-only
preflight. This adapter uses no Docker, network, Telegram or exchange client.
Default CLI only plans. Producer runtime and all accounting stay unchanged.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import tempfile
import time

BOT = "grid-live-fdusd-400"
PAIRS = ("BTC-FDUSD", "ETH-FDUSD")
SCOPE = "grid_guard_latch_only"
FLAGS = ("tripped", "action_complete", "stop_complete")
CHECKS = (
    "unique_guard_writer_stopped", "grid_physically_stopped", "no_grid_orders",
    "exit_fills_verified", "residuals_classified_dust", "grid_ownership_reconciled",
    "ownership_deficit_zero", "owned_within_exchange_balance", "sources_fresh",
    "model_current_signed_week", "model_hashes_authorized", "model_both_risk_on",
    "market_prices_fresh", "exchange_filters_fresh", "quote_snapshot_fresh",
    "other_gate_state_preserved",
)


class ResetRejected(ValueError):
    """A static reason code, safe to return without source documents."""


def require(condition, code):
    if not condition:
        raise ResetRejected(code)


def canonical_sha256(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode("utf-8")).hexdigest()


def bytes_sha256(value):
    return hashlib.sha256(value).hexdigest()


def valid_sha(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def positive_time(value):
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and value > 0)


def stamp(value):
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        require(parsed.tzinfo is not None, "event_time_requires_timezone")
        result = parsed.timestamp()
    except (AttributeError, TypeError, ValueError):
        raise ResetRejected("invalid_event_time") from None
    require(positive_time(result), "invalid_event_time")
    return result


def fresh(value, now):
    return positive_time(value) and -5 <= now - value <= 60


def episode_of(raw):
    require(type(raw.get("schema_version")) is int and raw["schema_version"] == 13,
            "unsupported_runtime_schema")
    recovery = raw.get("portfolio_recovery", {})
    require(recovery.get("phase") == "EXITING" and recovery.get("scope") == "portfolio",
            "producer_must_remain_portfolio_exiting")
    episode = {key: recovery.get(key) for key in ("triggered_at", "mechanism", "scope")}
    require(positive_time(episode["triggered_at"]) and isinstance(episode["mechanism"], str)
            and bool(episode["mechanism"]), "invalid_runtime_episode")
    return episode


def episode_matches(value, episode, *, event=False):
    try:
        triggered = float(value.get("triggered_at"))
    except (TypeError, ValueError, OverflowError):
        return False
    return (not isinstance(value.get("triggered_at"), bool) and math.isfinite(triggered)
            and triggered == episode["triggered_at"]
            and value.get("mechanism") == episode["mechanism"]
            and value.get("scope") == ("PORTFOLIO" if event else episode["scope"]))


def validate(state, raw, event, proof, *, guard_state_sha256, now):
    """Validate complete in-memory documents; performs no I/O or mutation."""
    require(positive_time(now), "invalid_clock")
    require(valid_sha(guard_state_sha256), "invalid_state_sha256")
    require(proof.get("schema") == "grid-guard-latch-preflight-v1"
            and proof.get("scope") == SCOPE and proof.get("bot") == BOT, "invalid_proof_scope")
    require(proof.get("authorization") == {"type": "direct_user_request", "scope": SCOPE},
            "direct_scope_authorization_required")
    require(fresh(proof.get("observed_at"), now), "stale_preflight")
    for key in CHECKS:
        check = proof.get("checks", {}).get(key, {})
        require(check.get("ok") is True and fresh(check.get("observed_at"), now), "failed_check:" + key)
    snapshot = proof.get("snapshot", {})
    require(all(isinstance(snapshot.get(key), str) and snapshot[key]
                for key in ("guard_container_id", "grid_container_id")), "container_identity_required")
    require(type(snapshot.get("running_guard_writers")) is int and snapshot["running_guard_writers"] == 0
            and snapshot.get("guard_running") is False and snapshot.get("grid_running") is False,
            "writer_or_grid_not_stopped")
    counts = snapshot.get("grid_open_orders", {})
    require(set(counts) == set(PAIRS) and all(type(counts[pair]) is int and counts[pair] == 0 for pair in PAIRS),
            "grid_orders_not_zero")
    require(snapshot.get("portfolio_phase") == "EXITING" and snapshot.get("ordinary_permissions") == {
        "buy_enabled": False, "sell_enabled": False}, "ordinary_permissions_must_stay_closed")
    require(proof.get("guard_state_sha256") == guard_state_sha256, "preflight_state_mismatch")
    runtime_sha = canonical_sha256(raw)
    require(proof.get("runtime_sha256") == runtime_sha, "preflight_runtime_mismatch")
    episode = episode_of(raw)
    bot = state.get("bots", {}).get(BOT, {})
    require(all(bot.get(key) is True for key in FLAGS), "latch_not_in_verified_completed_state")
    require(positive_time(bot.get("tripped_at")) and episode["triggered_at"] <= bot["tripped_at"] <= now + 5,
            "invalid_trip_clock")
    require(isinstance(bot.get("reason"), str) and bool(bot["reason"]), "trip_reason_required")
    require(not bot.get("last_action_error"), "current_action_error")
    stop = bot.get("stop", {})
    require(stop.get("logical_bot_name") == BOT and all(stop.get(key) is True for key in (
        "verified_no_active_orders", "verified_no_live_instances")), "stop_not_verified")
    flatten = bot.get("flatten", {})
    require(set(flatten) == set(PAIRS) and all(flatten[pair] in ("dust", "dust_after_exchange_rounding")
                                            for pair in PAIRS), "exit_dust_not_verified")
    require(event.get("schema") == "ethbtc-telegram-event-v1" and event.get("source") == "grid-live-guard"
            and event.get("strategy") == "grid" and event.get("bot") == BOT
            and event.get("pair") == ",".join(PAIRS) and event.get("mechanism") == "infrastructure_integrity_breaker"
            and event.get("transition") == event.get("phase_to") == "LATCHED"
            and event.get("requires_manual_action") is True and event.get("action") == "manual_recovery_required",
            "wrong_canonical_event")
    identity = {key: event.get(key, "") for key in (
        "source", "strategy", "bot", "pair", "mechanism", "transition", "phase_from", "phase_to",
        "correlation_id", "reason")}
    require(bool(identity["correlation_id"]) and valid_sha(event.get("event_id"))
            and canonical_sha256(identity) == event["event_id"], "event_identity_mismatch")
    details = event.get("details", {})
    event_episode = details.get("recovery_episode", {})
    require(event_episode.get("bot") == BOT and episode_matches(event_episode, episode, event=True)
            and details.get("failure_type") == "complete", "event_episode_mismatch")
    event_time = stamp(event.get("occurred_at"))
    require(0 <= event_time - bot["tripped_at"] <= 5, "event_trip_mismatch")
    event_sha = canonical_sha256(event)
    require(proof.get("event_id") == event["event_id"] and proof.get("event_sha256") == event_sha,
            "preflight_event_mismatch")
    audit = proof.get("completion_audit", {})
    audit_episode = audit.get("snapshot", {}).get("recovery_episodes", {}).get("PORTFOLIO", {})
    require(audit.get("event") == "grid_circuit_breaker_complete" and audit.get("bot") == BOT
            and audit.get("reason") == bot["reason"] and audit.get("stop") == stop
            and audit.get("flatten") == flatten and episode_matches(audit_episode, episode)
            and 0 <= event_time - stamp(audit.get("timestamp")) <= 5, "completion_audit_mismatch")
    return {"episode": episode, "event_sha256": event_sha, "runtime_sha256": runtime_sha,
            "preflight_sha256": canonical_sha256(proof)}


def plan(state, raw, event, proof, *, reset_id, guard_state_sha256, now):
    require(isinstance(reset_id, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}", reset_id) is not None,
            "invalid_reset_id")
    bot = state.get("bots", {}).get(BOT, {})
    require(bot.get("manual_guard_reset", {}).get("reset_id") != reset_id, "reset_id_already_consumed")
    bound = validate(state, raw, event, proof, guard_state_sha256=guard_state_sha256, now=now)
    prior = {key: deepcopy(bot[key]) for key in (*FLAGS, "tripped_at", "reason", "stop", "flatten")}
    receipt = {"schema": "grid-guard-latch-reset-v1", "scope": SCOPE, "bot": BOT,
               "reset_id": reset_id, "event_id": event["event_id"], "performed_at": float(now),
               "event_sha256": bound["event_sha256"], "preflight_sha256": bound["preflight_sha256"],
               "runtime_sha256": bound["runtime_sha256"], "runtime_episode": deepcopy(bound["episode"]), "prior": prior}
    updated = deepcopy(state)
    target = updated["bots"][BOT]
    target.update({key: False for key in FLAGS})
    target.pop("reason", None)
    target.pop("tripped_at", None)
    target["manual_guard_reset"] = receipt
    summary = {"schema": "grid-guard-latch-reset-plan-v1", "scope": SCOPE, "bot": BOT, "reset_id": reset_id,
               "event_id": event["event_id"], "guard_state_sha256": guard_state_sha256,
               "runtime_sha256": bound["runtime_sha256"], "event_sha256": bound["event_sha256"],
               "preflight_sha256": bound["preflight_sha256"], "flags": {key: False for key in FLAGS},
               "producer_phase": "EXITING", "ordinary_permissions": {"buy_enabled": False, "sell_enabled": False},
               "starts_grid": False, "orders_submitted": 0}
    return {"new_state": updated, "summary": summary}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def accounting_sha256(state):
    """Ignore changing marks/health logs, retain every Grid economic field."""
    bot = deepcopy(state.get("bots", {}).get(BOT, {}))
    latest = bot.pop("latest", {})
    bot.pop("manual_guard_reset", None)
    bot["latest_accounting"] = {
        pair: {key: row.get(key) for key in ("owned_base", "quote_balance", "net_base", "fees_quote",
                                            "equity_source", "runtime_schema_version")}
        for pair, row in latest.get("pairs", {}).items()
    }
    return canonical_sha256(bot)


def plan_rollback(state, raw, post_reset_state, proof, *, reset_id, guard_state_sha256, now):
    """Pure selective rollback plan only; intentionally has no apply/CLI path.

    Root must supply the immutable state.after.json produced by the reset and
    another fresh stopped-writer/no-orders/no-economic-progress proof. This
    restores only the archived latch flags/reason/time, never a whole old state.
    """
    bot = state.get("bots", {}).get(BOT, {})
    receipt = bot.get("manual_guard_reset", {})
    saved = post_reset_state.get("bots", {}).get(BOT, {}).get("manual_guard_reset")
    require(receipt == saved and receipt.get("schema") == "grid-guard-latch-reset-v1"
            and receipt.get("scope") == SCOPE and receipt.get("bot") == BOT
            and receipt.get("reset_id") == reset_id and "rollback" not in receipt, "rollback_receipt_mismatch")
    require(all(bot.get(key) is False for key in FLAGS) and not bot.get("reason") and not bot.get("tripped_at"),
            "new_trip_or_reset_state_change")
    require(receipt.get("runtime_sha256") == canonical_sha256(raw)
            and receipt.get("runtime_episode") == episode_of(raw), "rollback_runtime_changed")
    require(proof.get("schema") == "grid-guard-latch-rollback-preflight-v1"
            and proof.get("scope") == "grid_guard_latch_rollback_only" and proof.get("bot") == BOT
            and proof.get("reset_id") == reset_id and fresh(proof.get("observed_at"), now)
            and valid_sha(guard_state_sha256) and proof.get("guard_state_sha256") == guard_state_sha256,
            "invalid_rollback_preflight")
    for key in ("unique_guard_writer_stopped", "grid_physically_stopped", "no_grid_orders", "no_new_economic_progress"):
        check = proof.get("checks", {}).get(key, {})
        require(check.get("ok") is True and fresh(check.get("observed_at"), now), "failed_rollback_check:" + key)
    snapshot = proof.get("snapshot", {})
    require(type(snapshot.get("running_guard_writers")) is int and snapshot["running_guard_writers"] == 0
            and snapshot.get("guard_running") is False and snapshot.get("grid_running") is False,
            "rollback_writer_or_grid_not_stopped")
    orders = snapshot.get("grid_open_orders", {})
    require(set(orders) == set(PAIRS) and all(type(orders[pair]) is int and orders[pair] == 0 for pair in PAIRS),
            "rollback_grid_orders_not_zero")
    expected_accounting = accounting_sha256(post_reset_state)
    require(accounting_sha256(state) == expected_accounting
            and proof.get("post_reset_accounting_sha256") == expected_accounting, "rollback_economic_state_changed")
    prior = receipt.get("prior", {})
    require(all(prior.get(key) is True for key in FLAGS) and positive_time(prior.get("tripped_at"))
            and isinstance(prior.get("reason"), str) and bool(prior["reason"]), "invalid_rollback_prior")
    updated = deepcopy(state)
    target = updated["bots"][BOT]
    for key in (*FLAGS, "reason", "tripped_at"):
        target[key] = deepcopy(prior[key])
    record = {"scope": "grid_guard_latch_rollback_only", "performed_at": float(now),
              "preflight_sha256": canonical_sha256(proof)}
    target["manual_guard_reset"]["rollback"] = record
    target["manual_guard_reset"].setdefault("history", []).append({"event": "selective_latch_rollback", **record})
    return {"new_state": updated, "summary": {"scope": record["scope"], "reset_id": reset_id,
            "planned_only": True, "starts_grid": False, "producer_written": False, "accounting_preserved": True}}


def private_write(path, content):
    with Path(path).open("xb") as handle:
        os.chmod(path, 0o600)
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())


def verify_inputs(paths, expected):
    for key, path in paths.items():
        value = bytes_sha256(path.read_bytes()) if key == "state" else canonical_sha256(read_json(path))
        require(value == expected[key], "input_changed:" + key)


def apply(state_path, raw_path, event_path, proof_path, *, reset_id, expected_state_sha256, private_dir, now=None):
    """Explicit local-file commit. Caller supplies fresh proof after stopping Guard."""
    paths = {key: Path(value) for key, value in (
        ("state", state_path), ("runtime", raw_path), ("event", event_path), ("proof", proof_path))}
    require(not paths["state"].is_symlink(), "state_symlink_refused")
    original = paths["state"].read_bytes()
    require(bytes_sha256(original) == expected_state_sha256, "state_cas_mismatch")
    docs = {key: json.loads(original) if key == "state" else read_json(path) for key, path in paths.items()}
    clock = time.time() if now is None else now
    planned = plan(docs["state"], docs["runtime"], docs["event"], docs["proof"], reset_id=reset_id,
                   guard_state_sha256=expected_state_sha256, now=clock)
    expected = {key: expected_state_sha256 if key == "state" else canonical_sha256(value) for key, value in docs.items()}
    before = paths["state"].stat()
    lock = paths["state"].with_name(paths["state"].name + ".reset.lock")
    lock_fd = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    os.close(lock_fd)
    temporary = None
    result = None
    try:
        artifacts = Path(private_dir) / reset_id
        artifacts.mkdir(mode=0o700, parents=True, exist_ok=False)
        private_write(artifacts / "state.before.json", original)
        encoded = (json.dumps(planned["new_state"], indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
        private_write(artifacts / "state.after.json", encoded)
        audit = {**planned["summary"], "performed_at": float(clock), "state_after_sha256": bytes_sha256(encoded),
                 "backup_sha256": bytes_sha256(original), "status": "prepared"}
        private_write(artifacts / "prepared-audit.json", (json.dumps(audit, indent=2) + "\n").encode())
        fd, name = tempfile.mkstemp(prefix="." + paths["state"].name + ".reset-", dir=paths["state"].parent)
        temporary = Path(name)
        with os.fdopen(fd, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, stat.S_IMODE(before.st_mode))
        if hasattr(os, "chown"):
            os.chown(temporary, before.st_uid, before.st_gid)
        verify_inputs(paths, expected)
        validate(docs["state"], docs["runtime"], docs["event"], docs["proof"],
                 guard_state_sha256=expected_state_sha256, now=time.time() if now is None else now)
        current = paths["state"].stat()
        require((current.st_uid, current.st_gid, stat.S_IMODE(current.st_mode)) ==
                (before.st_uid, before.st_gid, stat.S_IMODE(before.st_mode)), "state_permissions_changed")
        os.replace(temporary, paths["state"])
        temporary = None
        result = {**planned["summary"], "applied": True, "state_after_sha256": bytes_sha256(encoded),
                  "audit_directory": str(artifacts)}
        # Once replace succeeds the reset is committed. A later audit/directory
        # error must never be reported as "not applied" or invite a retry.
        if hasattr(os, "O_DIRECTORY"):
            try:
                directory_fd = os.open(paths["state"].parent, os.O_RDONLY | os.O_DIRECTORY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
            except OSError as exc:
                result["directory_sync_error_type"] = type(exc).__name__
        try:
            private_write(artifacts / "applied.json", (json.dumps(result, indent=2) + "\n").encode())
        except Exception as exc:
            result["audit_finalize_error_type"] = type(exc).__name__
        return result
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        try:
            lock.unlink()
        except OSError as exc:
            if result is None:
                raise
            result["lock_cleanup_error_type"] = type(exc).__name__


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("state", "runtime", "event", "proof"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--reset-id", required=True)
    parser.add_argument("--expected-state-sha256", required=True)
    parser.add_argument("--private-dir", type=Path)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    if args.apply:
        if args.private_dir is None:
            parser.error("--apply requires --private-dir")
        result = apply(args.state, args.runtime, args.event, args.proof, reset_id=args.reset_id,
                       expected_state_sha256=args.expected_state_sha256, private_dir=args.private_dir)
    else:
        raw_state = args.state.read_bytes()
        require(bytes_sha256(raw_state) == args.expected_state_sha256, "state_cas_mismatch")
        result = plan(json.loads(raw_state), read_json(args.runtime), read_json(args.event), read_json(args.proof),
                      reset_id=args.reset_id, guard_state_sha256=args.expected_state_sha256, now=time.time())["summary"]
        result["applied"] = False
    print(json.dumps(result, ensure_ascii=True))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(json.dumps({"applied": False, "error_type": type(exc).__name__,
                          "reason": str(exc) if isinstance(exc, ResetRejected) else "local_adapter_failed"}))
        raise SystemExit(1)

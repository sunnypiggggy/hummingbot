"""Breaker notification deduplication must not suppress audit or trading retries."""

from copy import deepcopy
import json
from pathlib import Path
import sys
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from live_guard import grid_live_guard as module


BOT = "grid-live-fdusd-400"
TRIGGERED = 1_792_658_250


def make_guard(tmp_path):
    guard = module.Guard.__new__(module.Guard)
    guard.audit_path = tmp_path / "audit.jsonl"
    guard.notification_path = tmp_path / "notifications.jsonl"
    guard.state = {"bots": {BOT: {"tripped": True, "tripped_at": TRIGGERED + 4,
        "action_complete": False, "latest": {"recovery_episodes": {
            "PORTFOLIO": {"triggered_at": TRIGGERED, "scope": "portfolio",
                          "mechanism": "portfolio_drawdown_breaker"},
            "BTC-FDUSD": {"triggered_at": TRIGGERED, "scope": "pair",
                          "mechanism": "strategy_drawdown_breaker"},
        }}}}}
    return guard


def failure(age=4.2, *, asset="BTC", amount="0.00167", scope="PORTFOLIO"):
    return {"bot": BOT,
            "reason": f"recoverable {scope} exit remained incomplete for {age:.1f}s",
            "error": f"RuntimeError('Grid owned {asset} {amount} exceeds exchange total 0.001')"}


def events(guard):
    if not guard.notification_path.exists():
        return []
    return [json.loads(line) for line in guard.notification_path.read_text(encoding="utf-8").splitlines()]


def test_same_failure_keeps_every_audit_and_preserves_risk_state(tmp_path):
    guard = make_guard(tmp_path)
    for age in (4.2, 8.0, 3600.1):
        guard.state["bots"][BOT]["tripped_at"] = TRIGGERED + age
        before = deepcopy(guard.state["bots"])
        guard.audit("grid_circuit_breaker_action_failed", **failure(age, amount=str(age)))
        assert guard.state["bots"] == before
    assert len(guard.audit_path.read_text(encoding="utf-8").splitlines()) == 3
    rows = events(guard)
    assert len(rows) == 1
    assert rows[0]["transition"] == "ACTION_FAILED"
    assert rows[0]["details"]["failure_type"] == "ownership_exceeds_exchange:BTC"
    assert "4.2s" not in rows[0]["reason"]
    assert rows[0]["details"]["exit_age_seconds"] == 4.2


def test_notification_dedup_survives_guard_state_reload(tmp_path):
    first = make_guard(tmp_path)
    first.audit("grid_circuit_breaker_action_failed", **failure())
    second = make_guard(tmp_path)
    second.state = json.loads(json.dumps(first.state))
    second.audit("grid_circuit_breaker_action_failed", **failure(500))
    assert len(events(second)) == 1


def test_new_original_recovery_time_creates_a_new_notification(tmp_path):
    guard = make_guard(tmp_path)
    guard.audit("grid_circuit_breaker_action_failed", **failure())
    guard.state["bots"][BOT]["latest"]["recovery_episodes"]["PORTFOLIO"]["triggered_at"] += 86400
    guard.audit("grid_circuit_breaker_action_failed", **failure())
    assert len({row["event_id"] for row in events(guard)}) == 2


def test_failure_class_changes_and_returns_are_each_reported(tmp_path):
    guard = make_guard(tmp_path)
    for asset in ("BTC", "ETH", "BTC"):
        guard.audit("grid_circuit_breaker_action_failed", **failure(asset=asset))
    rows = events(guard)
    assert len(rows) == len({row["event_id"] for row in rows}) == 3
    assert guard.state["breaker_notification_state"][BOT + ":PORTFOLIO"]["sequence"] == 3


def test_complete_transition_is_not_hidden_by_failure(tmp_path):
    guard = make_guard(tmp_path)
    guard.audit("grid_circuit_breaker_action_failed", **failure())
    guard.audit("grid_circuit_breaker_complete", **failure(10))
    guard.audit("grid_circuit_breaker_complete", **failure(20))
    rows = events(guard)
    assert [row["transition"] for row in rows] == ["ACTION_FAILED", "LATCHED"]
    assert rows[-1]["requires_manual_action"] is True


@pytest.mark.parametrize("value", [None, "NaN", "Infinity", -1, True])
def test_missing_or_invalid_episode_does_not_silence_legacy_notifications(tmp_path, value):
    guard = make_guard(tmp_path)
    guard.state["bots"][BOT]["latest"]["recovery_episodes"]["PORTFOLIO"]["triggered_at"] = value
    guard.audit("grid_circuit_breaker_action_failed", **failure())
    guard.audit("grid_circuit_breaker_action_failed", **failure(8))
    assert len(events(guard)) == 2


def test_pair_and_portfolio_episodes_do_not_merge(tmp_path):
    guard = make_guard(tmp_path)
    guard.audit("grid_circuit_breaker_action_failed", **failure())
    guard.audit("grid_circuit_breaker_action_failed", **failure(scope="BTC-FDUSD"))
    assert len(events(guard)) == 2


def test_unknown_errors_are_not_reduced_to_one_exception_class(tmp_path):
    guard = make_guard(tmp_path)
    for text in ("RuntimeError('connection unavailable')", "RuntimeError('filter unavailable')"):
        detail = failure()
        detail["error"] = text
        guard.audit("grid_circuit_breaker_action_failed", **detail)
    assert len(events(guard)) == 2


def test_failed_notification_write_can_be_retried(tmp_path):
    guard = make_guard(tmp_path)
    with patch.object(module, "append_event", side_effect=OSError("test volume unavailable")):
        with pytest.raises(OSError):
            guard._emit_notification("grid_circuit_breaker_action_failed", failure())
    guard._emit_notification("grid_circuit_breaker_action_failed", failure())
    assert len(events(guard)) == 1


def test_unrelated_breaker_reason_keeps_original_notification(tmp_path):
    guard = make_guard(tmp_path)
    detail = {"bot": BOT, "reason": "independent integrity fault", "error": "RuntimeError('test')"}
    guard._emit_notification("grid_circuit_breaker_action_failed", detail)
    row = events(guard)[0]
    assert row["reason"] == detail["reason"]
    assert row["correlation_id"] == detail["reason"]

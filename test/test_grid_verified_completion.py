"""Only a verified breaker completion clears its current diagnostic error."""
import sys
from pathlib import Path
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from live_guard.grid_live_guard import Guard  # noqa: E402


def guard_with_previous_error():
    guard = object.__new__(Guard)
    guard.state = {"bots": {"grid-live-fdusd-400": {
        "tripped": True, "action_complete": False, "last_action_error": "previous failure",
    }}}
    guard.save = Mock()
    guard.audit = Mock()
    guard._secure_stop = Mock(return_value={"verified_no_active_orders": True})
    guard.snapshot = Mock(return_value={"pairs": {}})
    guard.flatten_deltas = Mock(return_value={"BTC-FDUSD": "dust", "ETH-FDUSD": "dust"})
    return guard


def test_verified_stop_and_reconciliation_clear_current_error():
    guard = guard_with_previous_error()
    guard.trip("FDUSD", "recoverable exit incomplete", {"pairs": {}})
    state = guard.state["bots"]["grid-live-fdusd-400"]
    assert state["action_complete"] is True
    assert state["stop_complete"] is True
    assert "last_action_error" not in state
    assert guard.audit.call_args.args == ("grid_circuit_breaker_complete",)
    assert state["flatten"] == {}


def test_failed_verification_keeps_current_error_and_pending_exit():
    guard = guard_with_previous_error()
    guard._secure_stop.side_effect = RuntimeError("orders remain")
    guard.trip("FDUSD", "recoverable exit incomplete", {"pairs": {}})
    state = guard.state["bots"]["grid-live-fdusd-400"]
    assert state["action_complete"] is False
    assert "orders remain" in state["last_action_error"]
    assert "stop_complete" not in state
    guard.flatten_deltas.assert_not_called()
    assert guard.audit.call_args.args == ("grid_circuit_breaker_action_failed",)

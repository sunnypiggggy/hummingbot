import json
from pathlib import Path

import pytest

from live_guard.telegram_notifications import build_event, format_event
from scripts.grid_v22_live_gate import AUTO_CONFIRMATION, _authorization
from scripts.v22_weekly_release_manager import (
    DEFAULT_DELAY_SECONDS,
    DEFAULT_GENERATION_LEAD_SECONDS,
    Policy,
    WeeklyReleaseManager,
    approval_prompt,
)


HASH = "a" * 64
MODEL = "b" * 64


def manager(tmp_path: Path, now: int, effective_end: int) -> WeeklyReleaseManager:
    root = tmp_path / "release"
    current = root / "current"
    current.mkdir(parents=True)
    (current / "production_lock.json").write_text(json.dumps({
        "package_id": "ethbtc-forced-exit", "release_sha256": HASH,
        "model_sha256": MODEL, "effective_end": effective_end,
    }), encoding="utf-8")
    return WeeklyReleaseManager(
        release_root=root, work_root=tmp_path / "work", candle_dir=tmp_path / "candles",
        state_path=tmp_path / "work/state.json", authorization_path=tmp_path / "auth.json",
        notification_path=tmp_path / "events.jsonl", grid_state_path=tmp_path / "grid.json",
        dca_state_path=tmp_path / "dca.json", policy=Policy(), now=lambda: now,
    )


def test_default_schedule_generates_13h_before_boundary_and_reviews_for_12h(tmp_path: Path):
    boundary = 2_000_000_000
    value = manager(tmp_path, boundary - DEFAULT_GENERATION_LEAD_SECONDS - 1, boundary).reconcile()
    assert value["phase"] == "SCHEDULED"
    assert value["next_generation_at"] == boundary - DEFAULT_GENERATION_LEAD_SECONDS
    assert Policy().approval_delay_seconds == DEFAULT_DELAY_SECONDS == 43_200


def test_approval_wait_does_not_touch_current_release(tmp_path: Path):
    value = manager(tmp_path, 1_000, 2_000)
    state = {
        "phase": "AWAITING_APPROVAL", "candidate_release_sha256": "c" * 64,
        "candidate_path": str(tmp_path / "candidate"), "source_effective_end": 2_000,
        "review_started_at": 900, "review_deadline": 1_100,
    }
    before = (value.current / "production_lock.json").read_bytes()
    assert value._approve(state, 1_000, {}) == state
    assert (value.current / "production_lock.json").read_bytes() == before
    assert not value.authorization_path.exists()


def test_auto_authorization_requires_full_12h_review():
    production = {"release_sha256": HASH, "model_sha256": MODEL, "effective_end": 200_000}
    receipt = {
        "schema": "ethbtc-forced-exit-authorization-v1", "package_id": "ethbtc-forced-exit",
        "release_sha256": HASH, "model_sha256": MODEL, "confirmation": AUTO_CONFIRMATION,
        "approval_mode": "automatic_default_after_12h", "review_started_at": 10_000,
        "review_deadline": 53_200, "approved_at": 53_200, "activate_at": 60_000,
        "approval_request_sha256": "c" * 64, "observation_report_sha256": "d" * 64,
        "preflight_sha256": "e" * 64,
    }
    assert _authorization(receipt, production, 59_999)[0] is False
    assert _authorization(receipt, production, 60_000)[0] is True
    broken = {**receipt, "review_deadline": 53_199}
    with pytest.raises(ValueError, match="12h review"):
        _authorization(broken, production, 60_000)


def test_candidate_notification_contains_default_pass_prompt():
    prompt = approval_prompt(HASH, 2_000_000_000)
    event = build_event(
        source="test", strategy="grid+dca", bot="bots", pair="pairs",
        mechanism="parameter_update", transition="MODEL_APPROVAL_PENDING",
        reason="candidate ready", release_sha256=HASH,
        correlation_id=HASH, details={"review_deadline": 2_000_000_000, "prompt": prompt},
    )
    text = format_event(event)
    assert "默认行为" in text
    assert "审批等待不影响当前模型交易" in text
    assert "Hermes" in text and HASH in text

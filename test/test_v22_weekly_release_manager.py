import json
import hashlib
from pathlib import Path
from unittest.mock import patch

import pytest

from live_guard.telegram_notifications import build_event, format_event
from scripts.grid_v22_live_gate import AUTO_CONFIRMATION, _authorization
from scripts.ethbtc_forced_exit_contract import atomic_json
from scripts.v22_weekly_release_manager import (
    DEFAULT_DELAY_SECONDS,
    DEFAULT_GENERATION_LEAD_SECONDS,
    DEFAULT_RETAIN_OLD_RELEASES,
    Policy,
    V22LiveGateProducer,
    WeeklyReleaseManager,
    approval_prompt,
    policy_from_environment,
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
    value = WeeklyReleaseManager(
        release_root=root, work_root=tmp_path / "work", candle_dir=tmp_path / "candles",
        state_path=tmp_path / "work/state.json", authorization_path=tmp_path / "auth.json",
        notification_path=tmp_path / "events.jsonl", grid_state_path=tmp_path / "grid.json",
        dca_state_path=tmp_path / "dca.json", policy=Policy(), now=lambda: now,
    )
    receipt_root = tmp_path / "telegram" / "evidence_receipts"
    receipt_root.mkdir(parents=True)
    receipt = {
        "schema": "telegram-evidence-delivery-receipt-v1",
        "identity_sha256": "c" * 64, "source_event_id": "candidate-event",
        "release_sha256": "c" * 64, "model_sha256": MODEL,
        "parameter_sha256": "", "report_request": "v22_png_windows",
        "expected_photo_count": 12, "photo_sha256": [str(i) * 64 for i in range(1, 10)] + [
            "a" * 64, "b" * 64, "c" * 64,
        ],
        "delivered_at": "2033-05-18T03:33:20+00:00", "telegram_message_ids": ["1"] * 13,
    }
    receipt["delivery_receipt_sha256"] = hashlib.sha256(
        json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    (receipt_root / f"{'c' * 64}.json").write_text(json.dumps(receipt), encoding="utf-8")
    return value


def test_default_schedule_generates_16h_before_boundary_and_reviews_for_12h(tmp_path: Path):
    boundary = 2_000_000_000
    value = manager(tmp_path, boundary - DEFAULT_GENERATION_LEAD_SECONDS - 1, boundary).reconcile()
    assert value["phase"] == "SCHEDULED"
    assert value["next_generation_at"] == boundary - DEFAULT_GENERATION_LEAD_SECONDS
    assert Policy().approval_delay_seconds == DEFAULT_DELAY_SECONDS == 43_200


def test_public_approval_contracts_are_readable_by_unprivileged_management_bot(tmp_path: Path):
    value = manager(tmp_path, 1_000, 2_000)
    release = "c" * 64
    request = tmp_path / "work" / f"approval-request-{release}.json"
    atomic_json(request, {"release_sha256": release, "model_sha256": MODEL})
    state = {
        "schema": "ethbtc-forced-exit-weekly-automation-v1",
        "phase": "AWAITING_APPROVAL", "candidate_release_sha256": release,
        "request_path": str(request), "review_started_at": 900,
        "review_deadline": 1_100, "activation_boundary": 2_000,
    }
    with patch("scripts.v22_weekly_release_manager.os.chmod") as chmod:
        value._save(state)
    public = tmp_path / "work" / "approval_public"
    assert (public / "automation_state.json").is_file()
    assert (public / request.name).is_file()
    assert {Path(call.args[0]) for call in chmod.call_args_list} == {
        public / "automation_state.json", public / request.name,
    }
    assert all(call.args[1] == 0o644 for call in chmod.call_args_list)


def test_early_activation_commits_only_runtime_pointer_not_mutable_aliases(tmp_path: Path):
    boundary = 2_000_000_000
    value = manager(tmp_path, boundary - 1_800, boundary)
    candidate = "c" * 64
    generation = "d" * 64
    receipt = {"model_sha256": MODEL}
    pending = tmp_path / "work/pending.json"
    prepared_pointer = tmp_path / "work/prepared-pointer.json"
    atomic_json(pending, receipt)
    atomic_json(prepared_pointer, {
        "schema": "ethbtc-forced-exit-runtime-pointer-v1",
        "runtime_generation": generation,
        "generation_manifest_sha256": generation,
    })
    state = {
        "candidate_release_sha256": candidate,
        "prepared_pointer_path": str(prepared_pointer),
        "pending_authorization_path": str(pending),
        "activation_boundary": boundary,
        "approval_mode": "manual_hermes",
    }

    def commit(producer, pointer):
        atomic_json(producer.runtime_root / "current.json", dict(pointer))

    with patch.object(V22LiveGateProducer, "commit_generation", commit):
        activated = value._activate(state, boundary - 1_800)

    assert activated["phase"] == "WARM_ACTIVE_PENDING_FOLD"
    assert json.loads((value.runtime_root / "current.json").read_text(encoding="utf-8"))[
        "runtime_generation"
    ] == generation
    assert not (value.release_root / "active_deployment.json").exists()
    assert not value.authorization_path.exists()


def test_three_warm_failures_restore_signed_predecessor_before_boundary(tmp_path: Path):
    boundary = 2_000_000_000
    value = manager(tmp_path, boundary - 1_700, boundary)
    old_pointer = {
        "schema": "ethbtc-forced-exit-runtime-pointer-v1",
        "runtime_generation": "e" * 64,
        "generation_manifest_sha256": "e" * 64,
    }
    candidate_pointer = {
        "schema": "ethbtc-forced-exit-runtime-pointer-v1",
        "runtime_generation": "d" * 64,
        "generation_manifest_sha256": "d" * 64,
    }
    atomic_json(value.runtime_root / "current.json", candidate_pointer)
    observation = value.grid_state_path.parent / "ethbtc_forced_exit_observation.json"
    atomic_json(observation, {
        "source_healthy": False, "runtime_generation": "d" * 64,
    })
    state = {
        "phase": "WARM_ACTIVE_PENDING_FOLD",
        "candidate_release_sha256": "c" * 64,
        "runtime_generation": "d" * 64,
        "activation_boundary": boundary,
        "previous_runtime_pointer": old_pointer,
        "warm_verified_cycles": 0,
        "warm_failures": 0,
    }
    for offset in range(3):
        state = value._monitor_warm_generation(state, boundary - 1_700 + offset)
    assert state["phase"] == "APPROVED_PENDING_PREWARM"
    restored = json.loads((value.runtime_root / "current.json").read_text(encoding="utf-8"))
    assert restored == old_pointer


def test_late_recovery_waits_for_three_healthy_producer_cycles(tmp_path: Path):
    boundary = 2_000_000_000
    value = manager(tmp_path, boundary + 100, boundary)
    generation = "d" * 64
    observation = value.grid_state_path.parent / "ethbtc_forced_exit_observation.json"
    atomic_json(observation, {
        "source_healthy": True, "runtime_generation": generation,
    })
    state = {
        "phase": "ACTIVE_UNAVAILABLE",
        "candidate_release_sha256": "c" * 64,
        "runtime_generation": generation,
        "activation_boundary": boundary,
        "late_signed_week_recovery": True,
        "warm_verified_cycles": 0,
        "warm_failures": 1,
    }
    with patch.object(value, "_finalize_fold", return_value={"phase": "ACTIVE"}) as finalize:
        first = value._monitor_warm_generation(state, boundary + 100)
        second = value._monitor_warm_generation(first, boundary + 101)
        third = value._monitor_warm_generation(second, boundary + 102)
    assert first["phase"] == "ACTIVE_UNAVAILABLE"
    assert second["phase"] == "ACTIVE_UNAVAILABLE"
    assert third == {"phase": "ACTIVE"}
    finalize.assert_called_once()


def test_t_minus_60m_rechecks_all_hard_gates_before_prewarm(tmp_path: Path):
    boundary = 2_000_000_000
    value = manager(tmp_path, boundary - 3_600, boundary)
    state = {
        "phase": "APPROVED_PENDING_PREWARM",
        "candidate_release_sha256": "c" * 64,
        "candidate_path": str(tmp_path / "candidate"),
        "source_effective_end": boundary,
        "activation_boundary": boundary,
        "prewarm_at": boundary - 2_100,
        "final_check_at": boundary - 3_600,
        "final_check_complete": False,
    }
    atomic_json(value.state_path, state)
    with patch.object(value, "_candidate_checks", return_value={"hashes": True}), \
            patch.object(value, "_runtime_checks", return_value={"account": True}):
        checked = value.reconcile()
    assert checked["final_check_complete"] is True
    assert checked["final_checks"] == {"hashes": True, "account": True}
    assert checked["phase"] == "APPROVED_PENDING_PREWARM"


def test_late_runtime_commit_is_refused_at_fold_boundary(tmp_path: Path):
    boundary = 2_000_000_000
    value = manager(tmp_path, boundary, boundary)
    atomic_json(value.state_path, {
        "phase": "PREWARMED_PENDING_ACTIVATION",
        "candidate_release_sha256": "c" * 64,
        "activation_boundary": boundary,
        "activate_at": boundary - 1_800,
    })
    result = value.reconcile()
    assert result["phase"] == "SIGNED_WEEK_UNAVAILABLE"
    assert not (value.runtime_root / "current.json").exists()


def test_approval_wait_does_not_touch_current_release(tmp_path: Path):
    value = manager(tmp_path, 1_000, 2_000)
    state = {
        "phase": "AWAITING_APPROVAL", "candidate_release_sha256": "c" * 64,
        "candidate_path": str(tmp_path / "candidate"), "source_effective_end": 2_000,
        "review_started_at": 900, "review_deadline": 1_100,
        "activation_boundary": 2_000,
    }
    before = (value.current / "production_lock.json").read_bytes()
    assert value._approve(state, 1_000, {}) == state
    assert (value.current / "production_lock.json").read_bytes() == before
    assert not value.authorization_path.exists()


def test_model_approval_is_blocked_until_telegram_png_receipt_exists(tmp_path: Path):
    boundary = 2_000_000_000
    value = manager(tmp_path, boundary - 4_000, boundary)
    (value.evidence_receipt_root / f"{'c' * 64}.json").unlink()
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    request = tmp_path / "request.json"
    atomic_json(request, {"candidate": "c" * 64})
    state = {
        "phase": "AWAITING_APPROVAL", "candidate_release_sha256": "c" * 64,
        "candidate_path": str(candidate), "source_effective_end": boundary,
        "review_started_at": boundary - 50_000, "review_deadline": boundary - 5_000,
        "activation_boundary": boundary, "request_path": str(request),
        "last_event_id": "candidate-event",
    }
    with patch.object(value, "_candidate_checks", return_value={"package": True}), \
            patch.object(value, "_runtime_checks", return_value={"runtime": True}), \
            patch.object(value, "_production", return_value={
                "model_sha256": MODEL, "effective_end": boundary + 7 * 86400,
            }):
        result = value._approve(state, boundary - 4_000, {})
    assert result["phase"] == "AWAITING_APPROVAL"
    assert "telegram_model_evidence_delivered': False" in result["last_error"]
    assert not value.authorization_path.exists()


def test_missed_approval_boundary_is_terminal_before_hard_gate_recheck(tmp_path: Path):
    boundary = 2_000
    value = manager(tmp_path, boundary, boundary)
    state = {
        "phase": "AWAITING_APPROVAL", "candidate_release_sha256": "c" * 64,
        "candidate_path": str(tmp_path / "candidate"), "source_effective_end": boundary,
        "review_started_at": 1_000, "review_deadline": 1_500,
        "activation_boundary": boundary,
    }
    with patch.object(value, "_candidate_checks") as candidate_checks, \
            patch.object(value, "_runtime_checks") as runtime_checks:
        result = value._approve(state, boundary, {})
    assert result["phase"] == "SIGNED_WEEK_UNAVAILABLE"
    assert "missed the signed week boundary" in result["last_error"]
    candidate_checks.assert_not_called()
    runtime_checks.assert_not_called()


def test_late_signed_week_recovery_can_enter_approval_after_boundary(tmp_path: Path):
    boundary = 2_000
    value = manager(tmp_path, boundary + 100, boundary)
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    state = {
        "phase": "AWAITING_APPROVAL", "candidate_release_sha256": "c" * 64,
        "candidate_path": str(candidate), "source_effective_end": boundary,
        "review_started_at": boundary, "review_deadline": boundary + 100,
        "activation_boundary": boundary, "late_signed_week_recovery": True,
        "request_path": str(tmp_path / "request.json"),
    }
    atomic_json(Path(state["request_path"]), {"candidate": "c" * 64})
    production = {
        "model_sha256": MODEL, "effective_end": boundary + 7 * 86400,
    }
    with patch.object(value, "_candidate_checks", return_value={"package": True}), \
            patch.object(value, "_runtime_checks", return_value={"runtime": True}), \
            patch.object(value, "_production", return_value=production):
        result = value._approve(state, boundary + 100, {})
    assert result["phase"] == "APPROVED_PENDING_PREWARM"
    assert result["late_signed_week_recovery"] is True
    assert result["activate_at"] > boundary + 100


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


def _write_release(value: WeeklyReleaseManager, release_sha: str, effective_end: int) -> Path:
    path = value.release_root / "releases" / release_sha
    path.mkdir(parents=True)
    atomic_json(path / "production_lock.json", {
        "package_id": "ethbtc-forced-exit",
        "release_sha256": release_sha,
        "effective_start": 1_000,
        "effective_end": effective_end,
    })
    (path / "model.bin").write_bytes(release_sha.encode())
    return path


def _write_generation(value: WeeklyReleaseManager, generation: str, *, release: str,
                      predecessor: str | None = None) -> Path:
    path = value.runtime_root / "generations" / generation
    path.mkdir(parents=True)
    atomic_json(path / "manifest.json", {
        "release_sha256": release,
        "predecessor_release_sha256": predecessor,
    })
    return path


def test_release_retention_keeps_current_plus_three_old_and_preserves_png_evidence(tmp_path: Path):
    value = manager(tmp_path, 2_000_000_000, 2_000_000_000)
    old = [str(number) * 64 for number in range(1, 6)]
    _write_release(value, HASH, 700)
    for number, release_sha in enumerate(old, 1):
        _write_release(value, release_sha, number * 100)
    atomic_json(value.release_root / "active_deployment.json", {"release_sha256": HASH})

    current_generation = "d" * 64
    stale_generation = "e" * 64
    _write_generation(value, current_generation, release=HASH, predecessor=old[-1])
    stale_path = _write_generation(value, stale_generation, release=old[0])
    atomic_json(value.runtime_root / "current.json", {
        "runtime_generation": current_generation, "release_sha256": HASH,
    })

    receipt = value.evidence_receipt_root / f"{old[0]}.json"
    atomic_json(receipt, {"release_sha256": old[0], "photo_sha256": ["f" * 64]})
    png = value.evidence_receipt_root.parent / "parameters" / "old-event" / "btc-360d.png"
    png.parent.mkdir(parents=True)
    png.write_bytes(b"png-evidence")

    report = value._prune_release_history({"candidate_release_sha256": HASH}, 1234)

    remaining = {path.name for path in (value.release_root / "releases").iterdir() if path.is_dir()}
    assert remaining == {HASH, old[2], old[3], old[4]}
    assert report["retain_old_releases"] == DEFAULT_RETAIN_OLD_RELEASES == 3
    assert report["removed_release_sha256"] == [old[0], old[1]]
    assert stale_generation in report["removed_runtime_generation"]
    assert not stale_path.exists()
    assert (value.runtime_root / "generations" / current_generation).is_dir()
    assert receipt.is_file() and png.read_bytes() == b"png-evidence"
    assert report["evidence_png_and_receipts_preserved"] is True


def test_release_retention_never_breaks_the_current_runtime_generation(tmp_path: Path):
    value = manager(tmp_path, 2_000_000_000, 2_000_000_000)
    value.policy = Policy(retain_old_releases=0)
    predecessor = "1" * 64
    _write_release(value, HASH, 200)
    _write_release(value, predecessor, 100)
    atomic_json(value.release_root / "active_deployment.json", {"release_sha256": HASH})
    generation = "d" * 64
    _write_generation(value, generation, release=HASH, predecessor=predecessor)
    atomic_json(value.runtime_root / "current.json", {
        "runtime_generation": generation, "release_sha256": HASH,
    })

    report = value._prune_release_history({"candidate_release_sha256": HASH}, 1234)

    assert (value.release_root / "releases" / predecessor).is_dir()
    assert (value.runtime_root / "generations" / generation).is_dir()
    assert report["removed_release_sha256"] == []


def test_release_retention_ignores_unverified_directories(tmp_path: Path):
    value = manager(tmp_path, 2_000_000_000, 2_000_000_000)
    _write_release(value, HASH, 200)
    atomic_json(value.release_root / "active_deployment.json", {"release_sha256": HASH})
    invalid = value.release_root / "releases" / ("f" * 64)
    invalid.mkdir(parents=True)
    atomic_json(invalid / "production_lock.json", {
        "package_id": "wrong-package", "release_sha256": invalid.name,
    })

    value._prune_release_history({"candidate_release_sha256": HASH}, 1234)

    assert invalid.is_dir()


def test_release_retention_failure_is_nonfatal_and_audited(tmp_path: Path):
    value = manager(tmp_path, 2_000_000_000, 2_000_000_000)
    with patch.object(value, "_prune_release_history", side_effect=OSError("disk busy")):
        report = value._apply_release_retention({"candidate_release_sha256": HASH}, 1234)

    assert report["status"] == "FAILED"
    assert "disk busy" in report["error"]
    persisted = json.loads(
        (value.work_root / "release_retention.json").read_text(encoding="utf-8")
    )
    assert persisted == report
    assert "MODEL_RETENTION_FAILED" in value.notification_path.read_text(encoding="utf-8")


def test_release_retention_policy_defaults_to_three(monkeypatch):
    monkeypatch.delenv("V22_WEEKLY_RETAIN_OLD_RELEASES", raising=False)
    assert policy_from_environment().retain_old_releases == 3
    monkeypatch.setenv("V22_WEEKLY_RETAIN_OLD_RELEASES", "5")
    assert policy_from_environment().retain_old_releases == 5


def test_release_retention_runs_only_after_a_healthy_fold_activation(tmp_path: Path):
    boundary = 2_000_000_000
    value = manager(tmp_path, boundary, boundary)
    pending = value.work_root / "pending.json"
    atomic_json(pending, {"model_sha256": MODEL})
    base_state = {
        "candidate_release_sha256": "c" * 64,
        "pending_authorization_path": str(pending),
        "activation_boundary": boundary,
        "approval_mode": "automatic_default_after_12h",
    }

    with patch.object(value, "_switch_current"), \
            patch.object(value, "_apply_release_retention") as retention:
        unavailable = value._finalize_fold(dict(base_state), boundary, generation_healthy=False)
    assert unavailable["phase"] == "ACTIVE_UNAVAILABLE"
    retention.assert_not_called()

    retention_report = {
        "schema": "ethbtc-forced-exit-release-retention-v1",
        "removed_release_sha256": [],
    }
    with patch.object(value, "_switch_current"), \
            patch.object(value, "_apply_release_retention", return_value=retention_report) as retention:
        active = value._finalize_fold(dict(base_state), boundary, generation_healthy=True)
    assert active["phase"] == "ACTIVE"
    assert active["release_retention"] == retention_report
    retention.assert_called_once()

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "live_guard"))

from ethbtc_forced_exit_contract import (  # noqa: E402
    CUTOVER_PHASE_WARM_ACTIVE_PENDING_FOLD,
    EXECUTION_POLICY_VERSION,
    MODEL_VERSION,
    PACKAGE_ID,
    SCHEMA,
    event_id,
    load_runtime_contract,
    utc,
    validate_cutover_transition,
)
from grid_v22_live_gate import V22LiveGateProducer  # noqa: E402
from risk_recovery import LATCHED, mark_exit_complete, trigger_state  # noqa: E402
from run_guard_with_v22_observation import update_status  # noqa: E402


HASH = "a" * 64


def contract(now: int, *, authorized: bool = False, risk_off_pair: str | None = None):
    pairs = {}
    for pair in ("BTC-FDUSD", "ETH-FDUSD"):
        risk_off = pair == risk_off_pair
        pairs[pair] = {
            "pair": pair,
            "source_pair": pair,
            "signal_ts": now - 60,
            "model_week": 37,
            "week_start": now - 3600,
            "week_end": now + 3600,
            "week_model_sha256": HASH,
            "probability": 0.9 if risk_off else 0.1,
            "entry_threshold": 0.5,
            "risk_off_active": risk_off,
            "recommended_buy_enabled": not risk_off,
            "buy_enabled": authorized and not risk_off,
            "force_exit": authorized and risk_off,
            "transition": "enter" if risk_off else "clear",
            "reason": "test",
            "event_id": event_id(HASH, pair, now - 60, "enter" if risk_off else "clear"),
        }
    return {
        "schema": SCHEMA,
        "package_id": PACKAGE_ID,
        "execution_policy_version": EXECUTION_POLICY_VERSION,
        "model_version": MODEL_VERSION,
        "generated_at": utc(now),
        "valid_until": utc(now + 120),
        "stale_after_seconds": 150,
        "release_sha256": HASH,
        "model_sha256": HASH,
        "feature_schema_sha256": HASH,
        "strategy_schema_sha256": HASH,
        "training_data_sha256": HASH,
        "source_healthy": True,
        "execution_authorized": authorized,
        "observation_mode": not authorized,
        "activation_at": now if authorized else None,
        "approval_receipt_sha256": HASH if authorized else None,
        "deployment_allowed": authorized,
        "promotion_authorized": authorized,
        "market_sell_action": True,
        "previous_model_fallback_allowed": False,
        "runtime_action": "execute" if authorized else "observe_only",
        "reason": "test",
        "pairs": pairs,
    }


def test_observation_never_authorizes_buy_or_exit(tmp_path: Path) -> None:
    now = 1_800_000_000
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(contract(now)), encoding="utf-8")
    loaded = load_runtime_contract(path, now=datetime.fromtimestamp(now, timezone.utc))
    assert loaded["runtime_gate_healthy"] is True
    assert loaded["execution_authorized"] is False
    assert all(not item["buy_enabled"] and not item["force_exit"] for item in loaded["pairs"].values())


def test_authorized_risk_off_is_pair_isolated_and_forces_exit(tmp_path: Path) -> None:
    now = 1_800_000_000
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(contract(now, authorized=True, risk_off_pair="BTC-FDUSD")), encoding="utf-8")
    loaded = load_runtime_contract(path, now=datetime.fromtimestamp(now, timezone.utc))
    assert loaded["pairs"]["BTC-FDUSD"]["force_exit"] is True
    assert loaded["pairs"]["BTC-FDUSD"]["buy_enabled"] is False
    assert loaded["pairs"]["ETH-FDUSD"]["force_exit"] is False
    assert loaded["pairs"]["ETH-FDUSD"]["buy_enabled"] is True


def test_tampered_or_expired_contract_fails_closed(tmp_path: Path) -> None:
    now = 1_800_000_000
    payload = contract(now, authorized=True)
    payload["previous_model_fallback_allowed"] = True
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    loaded = load_runtime_contract(path, now=datetime.fromtimestamp(now, timezone.utc))
    assert loaded["runtime_gate_healthy"] is False
    assert loaded["execution_authorized"] is False
    assert all(item["force_exit"] for item in loaded["pairs"].values())


def test_committed_generation_has_bounded_signed_fold_handover(tmp_path: Path) -> None:
    now = 1_800_000_000
    payload = contract(now, authorized=True)
    boundary = now + 3600
    payload.update({
        "generated_at": utc(boundary - 10),
        "valid_until": utc(boundary + 140),
        "runtime_generation": HASH,
        "predecessor_release_sha256": "b" * 64,
        "state_lineage_sha256": "c" * 64,
        "cutover_phase": "WARM_ACTIVE_PENDING_FOLD",
        "fold_boundary": boundary,
        "system_health": "HEALTHY",
    })
    for item in payload["pairs"].values():
        item.update({
            "next_week_start": boundary,
            "next_week_end": boundary + 7 * 24 * 3600,
            "next_week_model_sha256": "d" * 64,
            "model_signal": "RISK_ON",
        })
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    handover = load_runtime_contract(
        path, now=datetime.fromtimestamp(boundary + 30, timezone.utc),
    )
    assert handover["runtime_gate_healthy"] is True
    assert handover["fold_handover_active"] is True
    assert all(item["buy_enabled"] for item in handover["pairs"].values())

    expired = load_runtime_contract(
        path, now=datetime.fromtimestamp(boundary + 60, timezone.utc),
    )
    assert expired["runtime_gate_healthy"] is False
    assert all(item["force_exit"] for item in expired["pairs"].values())


def test_legacy_warm_active_pointer_is_normalized_without_boundary_gap(
    tmp_path: Path,
) -> None:
    now = 1_800_000_000
    boundary = now + 3600
    payload = contract(now, authorized=True)
    payload.update({
        "generated_at": utc(boundary - 10),
        "valid_until": utc(boundary + 140),
        "runtime_generation": HASH,
        "predecessor_release_sha256": "b" * 64,
        "state_lineage_sha256": "c" * 64,
        "cutover_phase": "WARM_ACTIVE",
        "fold_boundary": boundary,
        "system_health": "HEALTHY",
    })
    for item in payload["pairs"].values():
        item.update({
            "next_week_start": boundary,
            "next_week_end": boundary + 7 * 24 * 3600,
            "next_week_model_sha256": "d" * 64,
            "model_signal": "RISK_ON",
        })
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    loaded = load_runtime_contract(
        path, now=datetime.fromtimestamp(boundary + 3, timezone.utc),
    )
    assert loaded["runtime_gate_healthy"] is True
    assert loaded["cutover_phase"] == CUTOVER_PHASE_WARM_ACTIVE_PENDING_FOLD
    assert loaded["fold_handover_active"] is True
    assert all(not item["force_exit"] for item in loaded["pairs"].values())


def test_exact_incident_timeline_stays_healthy_without_scheduler_finalize(
    tmp_path: Path,
) -> None:
    boundary = 1_788_102_000
    payload = contract(boundary - 3, authorized=True)
    payload.update({
        "generated_at": utc(boundary - 3),
        "valid_until": utc(boundary + 147),
        "runtime_generation": HASH,
        "predecessor_release_sha256": "b" * 64,
        "state_lineage_sha256": "c" * 64,
        "cutover_phase": CUTOVER_PHASE_WARM_ACTIVE_PENDING_FOLD,
        "fold_boundary": boundary,
        "system_health": "HEALTHY",
    })
    for item in payload["pairs"].values():
        item.update({
            "week_end": boundary,
            "next_week_start": boundary,
            "next_week_end": boundary + 7 * 24 * 3600,
            "next_week_model_sha256": "d" * 64,
            "model_signal": "RISK_ON",
        })
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    at_three_seconds = load_runtime_contract(
        path, now=datetime.fromtimestamp(boundary + 3, timezone.utc),
    )
    assert at_three_seconds["runtime_gate_healthy"] is True
    assert at_three_seconds["fold_handover_active"] is True

    payload["generated_at"] = utc(boundary + 33)
    payload["valid_until"] = utc(boundary + 183)
    for pair, item in payload["pairs"].items():
        item.update({
            "signal_ts": boundary,
            "model_week": 38,
            "week_start": boundary,
            "week_end": boundary + 7 * 24 * 3600,
            "week_model_sha256": "d" * 64,
            "event_id": event_id(HASH, pair, boundary, "clear"),
        })
    path.write_text(json.dumps(payload), encoding="utf-8")
    at_thirty_three_seconds = load_runtime_contract(
        path, now=datetime.fromtimestamp(boundary + 33, timezone.utc),
    )
    assert at_thirty_three_seconds["runtime_gate_healthy"] is True
    assert at_thirty_three_seconds["fold_handover_active"] is False
    assert all(
        item["buy_enabled"] and not item["force_exit"]
        for item in at_thirty_three_seconds["pairs"].values()
    )

    payload["cutover_phase"] = "ACTIVE"
    path.write_text(json.dumps(payload), encoding="utf-8")
    at_fifty_eight_seconds = load_runtime_contract(
        path, now=datetime.fromtimestamp(boundary + 58, timezone.utc),
    )
    assert at_fifty_eight_seconds["runtime_gate_healthy"] is True
    assert at_fifty_eight_seconds["cutover_phase"] == "ACTIVE"


def test_cutover_state_machine_rejects_skipping_warm_activation() -> None:
    with pytest.raises(ValueError, match="invalid v22 cutover transition"):
        validate_cutover_transition("PREWARMED_PENDING_ACTIVATION", "ACTIVE")


def test_fold_handover_requires_contiguous_next_signed_model(tmp_path: Path) -> None:
    now = 1_800_000_000
    payload = contract(now, authorized=True)
    boundary = now + 3600
    payload.update({
        "generated_at": utc(boundary - 10),
        "valid_until": utc(boundary + 140),
        "runtime_generation": HASH,
        "predecessor_release_sha256": "b" * 64,
        "state_lineage_sha256": "c" * 64,
        "cutover_phase": "WARM_ACTIVE_PENDING_FOLD",
        "fold_boundary": boundary,
        "system_health": "HEALTHY",
    })
    for item in payload["pairs"].values():
        item["model_signal"] = "RISK_ON"
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    loaded = load_runtime_contract(
        path, now=datetime.fromtimestamp(boundary + 1, timezone.utc),
    )
    assert loaded["runtime_gate_healthy"] is False
    assert "signed week has expired" in loaded["reason"]


def test_integrity_failure_exits_before_becoming_latched() -> None:
    state = trigger_state(
        mechanism="infrastructure_integrity_breaker",
        scope="infrastructure",
        now=100,
        trigger_value="hash",
        signal_price="",
        reason="hash mismatch",
        latch_after_exit=True,
    )
    assert state["phase"] == "EXITING"
    state = mark_exit_complete(state, now=103, remaining_base={}, execution={"attempts": 1})
    assert state["phase"] == LATCHED
    assert state["cooldown_until"] is None


def test_compose_has_no_v22_service_and_dca_mount_is_read_only() -> None:
    compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    assert "grid-xgboost-v22:" not in compose
    assert "./grid-live-fdusd-data:/workspace/technical:ro" in compose
    assert "./release_packages/ethbtc-forced-exit:/workspace/v22-family:ro" in compose
    assert "GRID_V22_PACKAGE_PATH: /workspace/v22-family" in compose
    assert "DCA_V22_GATE_PATH: /workspace/technical/xgboost_risk_gate.json" in compose


def test_observation_status_maps_the_same_events_without_authorizing(tmp_path: Path) -> None:
    now = 1_800_000_000
    payload = contract(now)
    payload["runtime_gate_healthy"] = True
    grid_path = tmp_path / "grid.json"
    dca_path = tmp_path / "dca.json"
    update_status(grid_path, payload, now, "grid")
    update_status(dca_path, payload, now, "dca")
    grid = json.loads(grid_path.read_text(encoding="utf-8"))
    dca = json.loads(dca_path.read_text(encoding="utf-8"))
    assert grid["execution_authorized"] is False
    assert grid["event_ids"]["BTC-FDUSD"] == dca["event_ids"]["BTC-USDT"]
    assert grid["event_ids"]["ETH-FDUSD"] == dca["event_ids"]["ETH-USDT"]


def test_failed_runtime_does_not_erase_observation_window(tmp_path: Path) -> None:
    now = 1_800_000_000
    path = tmp_path / "status.json"
    healthy = contract(now)
    healthy["runtime_gate_healthy"] = True
    update_status(path, healthy, now, "grid")
    before = json.loads(path.read_text(encoding="utf-8"))
    failed = {
        "release_sha256": "0" * 64,
        "runtime_gate_healthy": False,
        "execution_authorized": False,
        "pairs": {},
    }
    update_status(path, failed, now + 30, "grid", "timeout")
    after = json.loads(path.read_text(encoding="utf-8"))
    assert after["release_sha256"] == healthy["release_sha256"]
    assert after["started_at"] == before["started_at"]
    assert after["event_ids"] == before["event_ids"]
    assert after["source_errors"] == 1


def test_live_producer_preserves_unhealthy_shadow_reason(tmp_path: Path) -> None:
    now = 1_800_000_000
    package = tmp_path / "package"
    shadow_package = package / "shadow_package"
    shadow_package.mkdir(parents=True)
    lock_path = shadow_package / "shadow_lock.json"
    lock = {
        "model_sha256": HASH,
        "feature_schema_sha256": HASH,
        "strategy_schema_sha256": HASH,
        "training_candle_sha256": HASH,
    }
    lock_path.write_text(json.dumps(lock), encoding="utf-8")

    import hashlib

    lock_sha = hashlib.sha256(lock_path.read_bytes()).hexdigest()
    (package / "production_lock.json").write_text(json.dumps({
        "package_id": PACKAGE_ID,
        "execution_policy_version": EXECUTION_POLICY_VERSION,
        "release_sha256": HASH,
        "model_sha256": HASH,
        "shadow_lock_sha256": lock_sha,
        "effective_end": now + 3600,
    }), encoding="utf-8")

    state_dir = tmp_path / "state"
    state_dir.mkdir()
    producer = V22LiveGateProducer(
        package_dir=package,
        cache_dir=tmp_path / "cache",
        seed_cache_dir=tmp_path / "seed",
        state_dir=state_dir,
        authorization_path=tmp_path / "missing-authorization.json",
        refresh_binance=False,
    )
    primary_reason = "fail_closed:ConnectionResetError(104, connection reset by peer)"
    unhealthy_shadow = {
        "source_healthy": False,
        "reason": primary_reason,
        "pairs": {
            pair: {"long": {"probability": None, "entry_threshold": None}}
            for pair in ("BTC-FDUSD", "ETH-FDUSD")
        },
    }
    with patch("grid_v22_live_gate.produce_once", return_value=unhealthy_shadow):
        produced = producer.produce(now)

    assert produced["source_healthy"] is False
    assert produced["reason"] == primary_reason
    assert "float()" not in produced["reason"]
    loaded = load_runtime_contract(
        producer.output, now=datetime.fromtimestamp(now, timezone.utc),
    )
    assert loaded["runtime_gate_healthy"] is False
    assert loaded["reason"] == primary_reason
    assert all(item["force_exit"] for item in loaded["pairs"].values())

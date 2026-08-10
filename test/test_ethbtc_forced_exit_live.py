import json
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "live_guard"))

from ethbtc_forced_exit_contract import (  # noqa: E402
    EXECUTION_POLICY_VERSION,
    MODEL_VERSION,
    PACKAGE_ID,
    SCHEMA,
    event_id,
    load_runtime_contract,
    utc,
)
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

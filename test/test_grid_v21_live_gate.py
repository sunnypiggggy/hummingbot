from datetime import datetime, timezone
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from grid_v21_live_gate import convert_shadow_to_live  # noqa: E402


def _shadow() -> dict:
    stamp = "2026-08-06T00:00:00Z"
    pair = lambda active: {  # noqa: E731
        "long": {
            "probability": .99 if active else .01,
            "entry_threshold": .98,
            "risk_off_active": active,
            "recommended_buy_enabled": not active,
            "transition": "hold" if active else "clear",
            "last_complete_1h": stamp,
            "last_complete_4h": stamp,
        }
    }
    digest = "a" * 64
    return {
        "model_version": "xgboost-grid-long-risk-gate-v21-250d",
        "source_healthy": True,
        "model_sha256": digest,
        "feature_schema_sha256": digest,
        "strategy_schema_sha256": digest,
        "training_data_sha256": digest,
        "candidate_lock_sha256": digest,
        "state_sha256": digest,
        "pairs": {"BTC-FDUSD": pair(True), "ETH-FDUSD": pair(False)},
    }


def test_authorized_v21_contract_is_pair_specific_buy_only() -> None:
    observed = int(datetime(2026, 8, 6, tzinfo=timezone.utc).timestamp())
    live = convert_shadow_to_live(_shadow(), authorized=True, observed_at=observed)
    assert live["shadow_mode"] is False
    assert live["runtime_action"] == "pause_ordinary_buy_only"
    assert live["market_sell_action"] is False
    assert live["pairs"]["BTC-FDUSD"]["buy_enabled"] is False
    assert live["pairs"]["ETH-FDUSD"]["buy_enabled"] is True


def test_unauthorized_v21_contract_fails_closed_for_both_pairs() -> None:
    observed = int(datetime(2026, 8, 6, tzinfo=timezone.utc).timestamp())
    live = convert_shadow_to_live(_shadow(), authorized=False, observed_at=observed)
    assert live["deployment_allowed"] is False
    assert all(not value["buy_enabled"] for value in live["pairs"].values())
